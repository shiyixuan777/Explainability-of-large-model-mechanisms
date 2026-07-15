from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from scripts.run_completion_margin_steering import (
    build_country_frame,
    completion_logprobs,
)
from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import probe_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/capital_balanced.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital_balanced")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--random-directions", type=int, default=50)
    parser.add_argument("--permutation-directions", type=int, default=20)
    parser.add_argument("--position-mode", default="prompt-final-only")
    parser.add_argument("--out-details", default="figures/completion_margin_steering_null_distribution.csv")
    parser.add_argument("--out-summary", default="figures/completion_margin_steering_null_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def normalized_random_direction(width: int, dtype: torch.dtype, rng: np.random.Generator) -> torch.Tensor:
    values = rng.normal(size=width)
    direction = torch.tensor(values, dtype=dtype)
    return direction / (direction.norm() + 1e-8)


def score_direction(
    model,
    country_data: pd.DataFrame,
    *,
    direction: torch.Tensor,
    alpha: float,
    layer: int,
    position_mode: str,
    baseline_avg_margin: np.ndarray,
    baseline_prefers_correct: np.ndarray,
) -> dict[str, float | int]:
    completion_prompts = country_data["completion_prompt"].tolist()
    correct_completions = [f" {capital}" for capital in country_data["correct_capital"].tolist()]
    false_completions = [f" {capital}" for capital in country_data["false_capital"].tolist()]
    correct_scores = completion_logprobs(
        model,
        completion_prompts,
        correct_completions,
        direction=direction,
        alpha=alpha,
        layer=layer,
        position_mode=position_mode,
    )
    false_scores = completion_logprobs(
        model,
        completion_prompts,
        false_completions,
        direction=direction,
        alpha=alpha,
        layer=layer,
        position_mode=position_mode,
    )
    margins: list[float] = []
    for correct, false in zip(correct_scores, false_scores):
        correct_total, correct_tokens = correct
        false_total, false_tokens = false
        margins.append((correct_total / correct_tokens) - (false_total / false_tokens))

    margin_array = np.array(margins)
    delta = margin_array - baseline_avg_margin
    prefers_correct = margin_array > 0
    return {
        "mean_delta_avg_token_margin": float(delta.mean()),
        "std_delta_avg_token_margin": float(delta.std(ddof=1)),
        "pairwise_avg_token_accuracy": float(prefers_correct.mean()),
        "sign_flip_to_correct": int(((baseline_prefers_correct == 0) & (prefers_correct == 1)).sum()),
        "sign_flip_to_false": int(((baseline_prefers_correct == 1) & (prefers_correct == 0)).sum()),
        "sign_flip_total": int((baseline_prefers_correct != prefers_correct).sum()),
    }


def summarize_null(details: pd.DataFrame, learned_effect: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for control_type, frame in details.groupby("control_type", sort=True):
        values = frame["mean_delta_avg_token_margin"].to_numpy()
        if control_type == "learned_probe":
            rows.append(
                {
                    "control_type": control_type,
                    "directions": int(len(frame)),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=0)),
                    "q025": float(values.mean()),
                    "q975": float(values.mean()),
                    "learned_effect": learned_effect,
                    "learned_percentile": 1.0,
                    "empirical_p_ge_learned": float("nan"),
                }
            )
            continue

        percentile = float((values <= learned_effect).mean())
        empirical_p = float((1 + (values >= learned_effect).sum()) / (len(values) + 1))
        rows.append(
            {
                "control_type": control_type,
                "directions": int(len(frame)),
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
                "learned_effect": learned_effect,
                "learned_percentile": percentile,
                "empirical_p_ge_learned": empirical_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    country_data = build_country_frame(data)

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    test_pair_ids = set(data.iloc[test_idx]["pair_id"].tolist())
    heldout_country_data = country_data[country_data["pair_id"].isin(test_pair_ids)].reset_index(drop=True)

    completion_prompts = heldout_country_data["completion_prompt"].tolist()
    correct_completions = [f" {capital}" for capital in heldout_country_data["correct_capital"].tolist()]
    false_completions = [f" {capital}" for capital in heldout_country_data["false_capital"].tolist()]
    baseline_correct = completion_logprobs(model, completion_prompts, correct_completions)
    baseline_false = completion_logprobs(model, completion_prompts, false_completions)
    baseline_avg_margin = np.array(
        [
            (correct_total / correct_tokens) - (false_total / false_tokens)
            for (correct_total, correct_tokens), (false_total, false_tokens) in zip(baseline_correct, baseline_false)
        ]
    )
    baseline_prefers_correct = baseline_avg_margin > 0

    train_activations = activations[train_idx]
    train_labels = labels[train_idx]
    learned = probe_direction(train_activations, train_labels).to(model.cfg.device)

    rows: list[dict[str, object]] = []
    learned_scores = score_direction(
        model,
        heldout_country_data,
        direction=learned,
        alpha=args.alpha,
        layer=args.layer,
        position_mode=args.position_mode,
        baseline_avg_margin=baseline_avg_margin,
        baseline_prefers_correct=baseline_prefers_correct,
    )
    rows.append(
        {
            "control_type": "learned_probe",
            "direction_index": 0,
            "seed": args.seed,
            "alpha": args.alpha,
            "position_mode": args.position_mode,
            **learned_scores,
        }
    )
    print(f"learned_probe delta={learned_scores['mean_delta_avg_token_margin']:+.4f}")

    for index in range(args.random_directions):
        direction = normalized_random_direction(train_activations.shape[1], train_activations.dtype, rng).to(
            model.cfg.device
        )
        scores = score_direction(
            model,
            heldout_country_data,
            direction=direction,
            alpha=args.alpha,
            layer=args.layer,
            position_mode=args.position_mode,
            baseline_avg_margin=baseline_avg_margin,
            baseline_prefers_correct=baseline_prefers_correct,
        )
        rows.append(
            {
                "control_type": "random_direction",
                "direction_index": index,
                "seed": args.seed,
                "alpha": args.alpha,
                "position_mode": args.position_mode,
                **scores,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"random_direction {index + 1}/{args.random_directions}")

    for index in range(args.permutation_directions):
        permuted_labels = rng.permutation(train_labels)
        direction = probe_direction(train_activations, permuted_labels).to(model.cfg.device)
        scores = score_direction(
            model,
            heldout_country_data,
            direction=direction,
            alpha=args.alpha,
            layer=args.layer,
            position_mode=args.position_mode,
            baseline_avg_margin=baseline_avg_margin,
            baseline_prefers_correct=baseline_prefers_correct,
        )
        rows.append(
            {
                "control_type": "label_permutation",
                "direction_index": index,
                "seed": args.seed,
                "alpha": args.alpha,
                "position_mode": args.position_mode,
                **scores,
            }
        )
        if (index + 1) % 5 == 0:
            print(f"label_permutation {index + 1}/{args.permutation_directions}")

    details = pd.DataFrame(rows)
    learned_effect = float(details.loc[details["control_type"] == "learned_probe", "mean_delta_avg_token_margin"].iloc[0])
    summary = summarize_null(details, learned_effect)

    details_path = Path(args.out_details)
    summary_path = Path(args.out_summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved null distribution details to {details_path}")
    print(f"Saved null distribution summary to {summary_path}")


if __name__ == "__main__":
    main()
