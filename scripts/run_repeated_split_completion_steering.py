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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--random-directions", type=int, default=10)
    parser.add_argument("--permutation-directions", type=int, default=5)
    parser.add_argument("--position-mode", default="prompt-final-only")
    parser.add_argument("--out-details", default="figures/repeated_split_completion_steering_details.csv")
    parser.add_argument("--out-summary", default="figures/repeated_split_completion_steering_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def avg_margin_rows(model, countries: pd.DataFrame, direction: torch.Tensor | None, alpha: float, layer: int, position_mode: str):
    prompts = countries["completion_prompt"].tolist()
    correct = [f" {capital}" for capital in countries["correct_capital"].tolist()]
    false = [f" {capital}" for capital in countries["false_capital"].tolist()]
    correct_scores = completion_logprobs(
        model,
        prompts,
        correct,
        direction=direction,
        alpha=alpha,
        layer=layer,
        position_mode=position_mode,
    )
    false_scores = completion_logprobs(
        model,
        prompts,
        false,
        direction=direction,
        alpha=alpha,
        layer=layer,
        position_mode=position_mode,
    )
    margins: list[float] = []
    correct_logprobs: list[float] = []
    false_logprobs: list[float] = []
    for (correct_total, correct_tokens), (false_total, false_tokens) in zip(correct_scores, false_scores):
        correct_avg = correct_total / correct_tokens
        false_avg = false_total / false_tokens
        correct_logprobs.append(correct_avg)
        false_logprobs.append(false_avg)
        margins.append(correct_avg - false_avg)
    return np.array(margins), np.array(correct_logprobs), np.array(false_logprobs)


def summarize_split(details: pd.DataFrame, seed: int) -> dict[str, object]:
    learned_row = details[details["direction_family"] == "learned_probe"].iloc[0]
    learned = float(learned_row["mean_delta_avg_token_margin"])
    random = details[details["direction_family"] == "random_direction"]["mean_delta_avg_token_margin"].to_numpy()
    permutation = details[details["direction_family"] == "label_permutation"]["mean_delta_avg_token_margin"].to_numpy()
    return {
        "seed": seed,
        "heldout_blocks": int(details["heldout_blocks"].iloc[0]),
        "heldout_countries": int(details["heldout_countries"].iloc[0]),
        "learned_delta": float(learned),
        "random_mean_delta": float(random.mean()),
        "random_q975_delta": float(np.quantile(random, 0.975)),
        "permutation_mean_delta": float(permutation.mean()),
        "permutation_q975_delta": float(np.quantile(permutation, 0.975)),
        "learned_minus_random_mean": float(learned - random.mean()),
        "learned_minus_permutation_mean": float(learned - permutation.mean()),
        "learned_gt_all_random": int(learned > random.max()),
        "learned_gt_all_permutation": int(learned > permutation.max()),
        "baseline_pairwise_accuracy": float(learned_row["baseline_pairwise_accuracy"]),
        "learned_pairwise_accuracy": float(learned_row["pairwise_avg_accuracy"]),
        "pairwise_accuracy_change": float(learned_row["pairwise_accuracy_change"]),
        "learned_sign_flips": int(learned_row["sign_flips"]),
        "wrong_to_correct_flips": int(learned_row["wrong_to_correct_flips"]),
        "correct_to_wrong_flips": int(learned_row["correct_to_wrong_flips"]),
    }


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    countries = build_country_frame(data)
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]

    base_margins, _, _ = avg_margin_rows(
        model,
        countries,
        direction=None,
        alpha=0.0,
        layer=args.layer,
        position_mode=args.position_mode,
    )
    countries = countries.copy()
    countries["baseline_avg_token_margin"] = base_margins
    countries["baseline_prefers_correct"] = base_margins > 0

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
        test_pair_ids = set(data.iloc[test_idx]["pair_id"].tolist())
        heldout = countries[countries["pair_id"].isin(test_pair_ids)].copy().reset_index(drop=True)
        rng = np.random.default_rng(seed)

        directions: list[tuple[str, str, torch.Tensor]] = []
        learned = probe_direction(activations[train_idx], labels[train_idx]).to(model.cfg.device)
        directions.append(("learned_probe", "learned_probe", learned))
        for idx in range(args.random_directions):
            random_np = rng.normal(size=activations.shape[1])
            random_direction = torch.tensor(random_np, dtype=activations.dtype)
            random_direction = (random_direction / (random_direction.norm() + 1e-8)).to(model.cfg.device)
            directions.append(("random_direction", f"random_{idx:02d}", random_direction))
        for idx in range(args.permutation_directions):
            permuted = rng.permutation(labels[train_idx])
            permutation_direction = probe_direction(activations[train_idx], permuted).to(model.cfg.device)
            directions.append(("label_permutation", f"permutation_{idx:02d}", permutation_direction))

        split_rows: list[dict[str, object]] = []
        for direction_family, direction_id, direction in directions:
            margins, correct_logprobs, false_logprobs = avg_margin_rows(
                model,
                heldout,
                direction=direction,
                alpha=args.alpha,
                layer=args.layer,
                position_mode=args.position_mode,
            )
            delta = margins - heldout["baseline_avg_token_margin"].to_numpy()
            prefers_correct = margins > 0
            baseline_prefers_correct = heldout["baseline_prefers_correct"].to_numpy()
            sign_flips = prefers_correct != baseline_prefers_correct
            wrong_to_correct = (~baseline_prefers_correct) & prefers_correct
            correct_to_wrong = baseline_prefers_correct & (~prefers_correct)
            baseline_pairwise_accuracy = float(baseline_prefers_correct.mean())
            pairwise_accuracy = float(prefers_correct.mean())
            row = {
                "seed": seed,
                "layer": args.layer,
                "alpha": args.alpha,
                "position_mode": args.position_mode,
                "direction_family": direction_family,
                "direction_id": direction_id,
                "heldout_blocks": int(heldout["pair_id"].nunique()),
                "heldout_countries": int(len(heldout)),
                "mean_delta_avg_token_margin": float(delta.mean()),
                "baseline_pairwise_accuracy": baseline_pairwise_accuracy,
                "pairwise_avg_accuracy": pairwise_accuracy,
                "pairwise_accuracy_change": float(pairwise_accuracy - baseline_pairwise_accuracy),
                "sign_flips": int(sign_flips.sum()),
                "wrong_to_correct_flips": int(wrong_to_correct.sum()),
                "correct_to_wrong_flips": int(correct_to_wrong.sum()),
                "mean_baseline_avg_token_margin": float(heldout["baseline_avg_token_margin"].mean()),
                "mean_avg_token_margin": float(margins.mean()),
            }
            split_rows.append(row)
            detail_rows.append(row)
            print(
                f"seed={seed:02d} {direction_id:<15} "
                f"delta={row['mean_delta_avg_token_margin']:+.4f} "
                f"pair_acc={row['baseline_pairwise_accuracy']:.3f}->{row['pairwise_avg_accuracy']:.3f} "
                f"flips={row['wrong_to_correct_flips']}/{row['correct_to_wrong_flips']}"
            )
        summary_rows.append(summarize_split(pd.DataFrame(split_rows), seed))

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    aggregate = {
        "seed": "aggregate",
        "heldout_blocks": int(summary["heldout_blocks"].median()),
        "heldout_countries": int(summary["heldout_countries"].median()),
        "learned_delta": float(summary["learned_delta"].mean()),
        "random_mean_delta": float(summary["random_mean_delta"].mean()),
        "random_q975_delta": float(summary["random_q975_delta"].mean()),
        "permutation_mean_delta": float(summary["permutation_mean_delta"].mean()),
        "permutation_q975_delta": float(summary["permutation_q975_delta"].mean()),
        "learned_delta_std": float(summary["learned_delta"].std(ddof=1)),
        "learned_delta_min": float(summary["learned_delta"].min()),
        "learned_delta_max": float(summary["learned_delta"].max()),
        "learned_minus_random_mean": float(summary["learned_minus_random_mean"].mean()),
        "learned_minus_permutation_mean": float(summary["learned_minus_permutation_mean"].mean()),
        "learned_gt_all_random": int(summary["learned_gt_all_random"].sum()),
        "learned_gt_all_permutation": int(summary["learned_gt_all_permutation"].sum()),
        "baseline_pairwise_accuracy": float(summary["baseline_pairwise_accuracy"].mean()),
        "learned_pairwise_accuracy": float(summary["learned_pairwise_accuracy"].mean()),
        "pairwise_accuracy_change": float(summary["pairwise_accuracy_change"].mean()),
        "learned_sign_flips": int(summary["learned_sign_flips"].sum()),
        "wrong_to_correct_flips": int(summary["wrong_to_correct_flips"].sum()),
        "correct_to_wrong_flips": int(summary["correct_to_wrong_flips"].sum()),
    }
    summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)

    details_path = Path(args.out_details)
    summary_path = Path(args.out_summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved repeated split steering details to {details_path}")
    print(f"Saved repeated split steering summary to {summary_path}")


if __name__ == "__main__":
    main()
