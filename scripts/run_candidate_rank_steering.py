from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts.run_completion_margin_steering import build_country_frame, completion_logprobs, make_directions
from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/capital_balanced.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital_balanced")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--position-mode", default="prompt-final-only")
    parser.add_argument("--out-details", default="figures/candidate_rank_steering_details.csv")
    parser.add_argument("--out-summary", default="figures/candidate_rank_steering_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def candidate_scores(model, prompt: str, candidates: list[str], *, direction=None, alpha: float, layer: int, position_mode: str) -> np.ndarray:
    prompts = [prompt] * len(candidates)
    completions = [f" {candidate}" for candidate in candidates]
    scores = completion_logprobs(
        model,
        prompts,
        completions,
        direction=direction,
        alpha=alpha,
        layer=layer,
        position_mode=position_mode,
    )
    return np.array([total / tokens for total, tokens in scores])


def rank_of(scores: np.ndarray, index: int) -> int:
    # 1 means highest score.
    return int((scores > scores[index]).sum() + 1)


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    countries = build_country_frame(data)
    candidate_capitals = sorted(countries["correct_capital"].unique().tolist())

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    test_pair_ids = set(data.iloc[test_idx]["pair_id"])
    heldout = countries[countries["pair_id"].isin(test_pair_ids)].copy().reset_index(drop=True)

    directions = make_directions(activations[train_idx], labels[train_idx], args.seed)
    learned = directions["learned_probe"].to(model.cfg.device)

    rows: list[dict[str, object]] = []
    for row in heldout.itertuples(index=False):
        correct_index = candidate_capitals.index(row.correct_capital)
        false_index = candidate_capitals.index(row.false_capital)
        baseline_scores = candidate_scores(
            model,
            row.completion_prompt,
            candidate_capitals,
            direction=None,
            alpha=0.0,
            layer=args.layer,
            position_mode=args.position_mode,
        )
        steered_scores = candidate_scores(
            model,
            row.completion_prompt,
            candidate_capitals,
            direction=learned,
            alpha=args.alpha,
            layer=args.layer,
            position_mode=args.position_mode,
        )
        baseline_correct_rank = rank_of(baseline_scores, correct_index)
        steered_correct_rank = rank_of(steered_scores, correct_index)
        baseline_false_rank = rank_of(baseline_scores, false_index)
        steered_false_rank = rank_of(steered_scores, false_index)
        rows.append(
            {
                "country": row.country,
                "pair_id": row.pair_id,
                "correct_capital": row.correct_capital,
                "false_capital": row.false_capital,
                "candidate_count": len(candidate_capitals),
                "baseline_correct_rank": baseline_correct_rank,
                "steered_correct_rank": steered_correct_rank,
                "rank_delta": baseline_correct_rank - steered_correct_rank,
                "baseline_false_rank": baseline_false_rank,
                "steered_false_rank": steered_false_rank,
                "baseline_top_candidate": candidate_capitals[int(np.argmax(baseline_scores))],
                "steered_top_candidate": candidate_capitals[int(np.argmax(steered_scores))],
                "baseline_top1_correct": int(np.argmax(baseline_scores) == correct_index),
                "steered_top1_correct": int(np.argmax(steered_scores) == correct_index),
                "top1_changed": int(np.argmax(baseline_scores) != np.argmax(steered_scores)),
                "correct_avg_logprob_delta": float(steered_scores[correct_index] - baseline_scores[correct_index]),
                "false_avg_logprob_delta": float(steered_scores[false_index] - baseline_scores[false_index]),
                "selected_pair_margin_delta": float(
                    (steered_scores[correct_index] - steered_scores[false_index])
                    - (baseline_scores[correct_index] - baseline_scores[false_index])
                ),
            }
        )
        print(
            f"{row.country:<24} rank {baseline_correct_rank:02d}->{steered_correct_rank:02d} "
            f"top1 {rows[-1]['baseline_top1_correct']}->{rows[-1]['steered_top1_correct']}"
        )

    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "layer": args.layer,
                "seed": args.seed,
                "alpha": args.alpha,
                "position_mode": args.position_mode,
                "heldout_countries": int(len(details)),
                "candidate_count": int(len(candidate_capitals)),
                "mean_baseline_correct_rank": float(details["baseline_correct_rank"].mean()),
                "mean_steered_correct_rank": float(details["steered_correct_rank"].mean()),
                "mean_rank_delta": float(details["rank_delta"].mean()),
                "median_rank_delta": float(details["rank_delta"].median()),
                "rank_improved_count": int((details["rank_delta"] > 0).sum()),
                "rank_worsened_count": int((details["rank_delta"] < 0).sum()),
                "baseline_top1_accuracy": float(details["baseline_top1_correct"].mean()),
                "steered_top1_accuracy": float(details["steered_top1_correct"].mean()),
                "top1_changed_count": int(details["top1_changed"].sum()),
                "mean_correct_avg_logprob_delta": float(details["correct_avg_logprob_delta"].mean()),
                "mean_false_avg_logprob_delta": float(details["false_avg_logprob_delta"].mean()),
                "mean_selected_pair_margin_delta": float(details["selected_pair_margin_delta"].mean()),
            }
        ]
    )

    details_path = Path(args.out_details)
    summary_path = Path(args.out_summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved candidate rank steering details to {details_path}")
    print(f"Saved candidate rank steering summary to {summary_path}")


if __name__ == "__main__":
    main()
