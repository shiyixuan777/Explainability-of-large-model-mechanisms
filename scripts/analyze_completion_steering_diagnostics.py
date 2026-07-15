from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", default="figures/completion_margin_steering_details.csv")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--negative-alpha", type=float, default=-4.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-decomposition", default="figures/completion_margin_steering_decomposition.csv")
    parser.add_argument("--out-paired", default="figures/completion_margin_steering_paired_bootstrap.csv")
    return parser.parse_args()


def corr_or_nan(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return float("nan")
    return float(left.corr(right))


def enrich_with_baselines(details: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["direction_type", "split", "pair_id", "country"]
    baseline = details.loc[details["alpha"] == 0, key_cols + [
        "correct_avg_token_logprob",
        "false_avg_token_logprob",
        "avg_token_margin",
        "prefers_correct_avg_token",
    ]].copy()
    baseline = baseline.rename(
        columns={
            "correct_avg_token_logprob": "baseline_correct_avg_token_logprob",
            "false_avg_token_logprob": "baseline_false_avg_token_logprob",
            "avg_token_margin": "baseline_joined_avg_token_margin",
            "prefers_correct_avg_token": "baseline_prefers_correct_avg_token",
        }
    )
    merged = details.merge(baseline, on=key_cols, how="left", validate="many_to_one")
    if merged["baseline_correct_avg_token_logprob"].isna().any():
        missing = merged.loc[merged["baseline_correct_avg_token_logprob"].isna(), key_cols].head()
        raise ValueError(f"Missing alpha=0 baseline rows for:\n{missing}")

    merged["delta_correct_avg_token_logprob"] = (
        merged["correct_avg_token_logprob"] - merged["baseline_correct_avg_token_logprob"]
    )
    merged["delta_false_avg_token_logprob"] = (
        merged["false_avg_token_logprob"] - merged["baseline_false_avg_token_logprob"]
    )
    merged["changed_to_correct"] = (
        (merged["baseline_prefers_correct_avg_token"] == 0) & (merged["prefers_correct_avg_token"] == 1)
    ).astype(int)
    merged["changed_to_false"] = (
        (merged["baseline_prefers_correct_avg_token"] == 1) & (merged["prefers_correct_avg_token"] == 0)
    ).astype(int)
    merged["changed_preference"] = (merged["changed_to_correct"] | merged["changed_to_false"]).astype(int)
    return merged


def baseline_preference_diff_ci(frame: pd.DataFrame, *, n_samples: int, seed: int) -> dict[str, float | int]:
    groups = np.array(sorted(frame["pair_id"].unique()))
    if len(groups) == 0 or n_samples <= 0:
        return {
            "baseline_correct_minus_wrong_delta_margin_ci_low": float("nan"),
            "baseline_correct_minus_wrong_delta_margin_ci_high": float("nan"),
            "baseline_preference_diff_bootstrap_samples": 0,
        }

    rng = np.random.default_rng(seed)
    correct_sums: list[float] = []
    correct_counts: list[int] = []
    false_sums: list[float] = []
    false_counts: list[int] = []
    for group in groups:
        block = frame[frame["pair_id"] == group]
        baseline_correct = block[block["baseline_prefers_correct_avg_token"] == 1]
        baseline_false = block[block["baseline_prefers_correct_avg_token"] == 0]
        correct_sums.append(float(baseline_correct["delta_avg_token_margin"].sum()))
        correct_counts.append(int(len(baseline_correct)))
        false_sums.append(float(baseline_false["delta_avg_token_margin"].sum()))
        false_counts.append(int(len(baseline_false)))

    correct_sums_array = np.array(correct_sums)
    correct_counts_array = np.array(correct_counts)
    false_sums_array = np.array(false_sums)
    false_counts_array = np.array(false_counts)

    values: list[float] = []
    for _ in range(n_samples):
        sampled_indices = rng.integers(0, len(groups), size=len(groups))
        correct_count = int(correct_counts_array[sampled_indices].sum())
        false_count = int(false_counts_array[sampled_indices].sum())
        if correct_count == 0 or false_count == 0:
            continue
        correct_mean = float(correct_sums_array[sampled_indices].sum() / correct_count)
        false_mean = float(false_sums_array[sampled_indices].sum() / false_count)
        values.append(correct_mean - false_mean)

    if not values:
        return {
            "baseline_correct_minus_wrong_delta_margin_ci_low": float("nan"),
            "baseline_correct_minus_wrong_delta_margin_ci_high": float("nan"),
            "baseline_preference_diff_bootstrap_samples": 0,
        }

    return {
        "baseline_correct_minus_wrong_delta_margin_ci_low": float(np.quantile(values, 0.025)),
        "baseline_correct_minus_wrong_delta_margin_ci_high": float(np.quantile(values, 0.975)),
        "baseline_preference_diff_bootstrap_samples": len(values),
    }


def summarize_decomposition(
    frame: pd.DataFrame,
    split_name: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["direction_type", "alpha"], sort=True)
    for (direction_type, alpha), group in grouped:
        baseline_correct = group[group["baseline_prefers_correct_avg_token"] == 1]
        baseline_false = group[group["baseline_prefers_correct_avg_token"] == 0]
        baseline_correct_shift = (
            float(baseline_correct["delta_avg_token_margin"].mean()) if not baseline_correct.empty else float("nan")
        )
        baseline_false_shift = (
            float(baseline_false["delta_avg_token_margin"].mean()) if not baseline_false.empty else float("nan")
        )
        baseline_diff = baseline_correct_shift - baseline_false_shift
        baseline_diff_ci = baseline_preference_diff_ci(
            group,
            n_samples=n_bootstrap,
            seed=seed + int(round((float(alpha) + 100.0) * 10)) + len(str(direction_type)),
        )
        rows.append(
            {
                "split": split_name,
                "direction_type": direction_type,
                "alpha": float(alpha),
                "countries": int(len(group)),
                "blocks": int(group["pair_id"].nunique()),
                "mean_delta_correct_avg_token_logprob": float(group["delta_correct_avg_token_logprob"].mean()),
                "mean_delta_false_avg_token_logprob": float(group["delta_false_avg_token_logprob"].mean()),
                "mean_delta_avg_token_margin": float(group["delta_avg_token_margin"].mean()),
                "std_delta_avg_token_margin": float(group["delta_avg_token_margin"].std(ddof=1)),
                "median_delta_avg_token_margin": float(group["delta_avg_token_margin"].median()),
                "min_delta_avg_token_margin": float(group["delta_avg_token_margin"].min()),
                "max_delta_avg_token_margin": float(group["delta_avg_token_margin"].max()),
                "baseline_prefers_correct_countries": int(len(baseline_correct)),
                "baseline_prefers_false_countries": int(len(baseline_false)),
                "mean_delta_margin_when_baseline_prefers_correct": baseline_correct_shift,
                "mean_delta_margin_when_baseline_prefers_false": baseline_false_shift,
                "baseline_correct_minus_wrong_delta_margin": baseline_diff,
                **baseline_diff_ci,
                "mean_delta_correct_when_baseline_prefers_correct": float(
                    baseline_correct["delta_correct_avg_token_logprob"].mean()
                )
                if not baseline_correct.empty
                else float("nan"),
                "mean_delta_false_when_baseline_prefers_correct": float(
                    baseline_correct["delta_false_avg_token_logprob"].mean()
                )
                if not baseline_correct.empty
                else float("nan"),
                "mean_delta_correct_when_baseline_prefers_false": float(
                    baseline_false["delta_correct_avg_token_logprob"].mean()
                )
                if not baseline_false.empty
                else float("nan"),
                "mean_delta_false_when_baseline_prefers_false": float(
                    baseline_false["delta_false_avg_token_logprob"].mean()
                )
                if not baseline_false.empty
                else float("nan"),
                "sign_flip_to_correct": int(group["changed_to_correct"].sum()),
                "sign_flip_to_false": int(group["changed_to_false"].sum()),
                "sign_flip_total": int(group["changed_preference"].sum()),
                "corr_baseline_margin_delta_margin": corr_or_nan(
                    group["baseline_avg_token_margin"], group["delta_avg_token_margin"]
                ),
            }
        )
    return rows


def paired_bootstrap_diff(
    frame: pd.DataFrame,
    *,
    learned_col: str,
    control_col: str,
    pair_ids: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    by_pair = {
        pair_id: frame.loc[frame["pair_id"] == pair_id, [learned_col, control_col]].to_numpy()
        for pair_id in pair_ids
    }
    values: list[float] = []
    for _ in range(n_samples):
        sampled_pair_ids = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sampled = np.concatenate([by_pair[pair_id] for pair_id in sampled_pair_ids], axis=0)
        values.append(float((sampled[:, 0] - sampled[:, 1]).mean()))
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_samples": len(values),
    }


def paired_bootstrap(details: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("learned_probe", "random_direction"),
        ("learned_probe", "label_permutation"),
    ]

    for split_name, split_frame in [
        ("all_countries", details),
        ("heldout_countries", details[details["split"] == "test"].copy()),
    ]:
        alpha_frame = split_frame[split_frame["alpha"] == args.alpha].copy()
        pivot = alpha_frame.pivot_table(
            index=["pair_id", "country"],
            columns="direction_type",
            values="delta_avg_token_margin",
            aggfunc="first",
        ).reset_index()
        pair_ids = np.array(sorted(pivot["pair_id"].unique()))
        for learned, control in comparisons:
            estimate = float((pivot[learned] - pivot[control]).mean())
            ci = paired_bootstrap_diff(
                pivot,
                learned_col=learned,
                control_col=control,
                pair_ids=pair_ids,
                n_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            rows.append(
                {
                    "split": split_name,
                    "metric": f"delta_avg_token_margin_alpha_{args.alpha:g}",
                    "comparison": f"{learned}_minus_{control}",
                    "estimate": estimate,
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "ci_unit": "pair_id_block",
                    "bootstrap_samples": ci["bootstrap_samples"],
                }
            )

        slope_source = split_frame[split_frame["alpha"].isin([args.negative_alpha, args.alpha])].copy()
        slope_pivot = slope_source.pivot_table(
            index=["pair_id", "country", "direction_type"],
            columns="alpha",
            values="delta_avg_token_margin",
            aggfunc="first",
        ).reset_index()
        denom = args.alpha - args.negative_alpha
        slope_pivot["slope"] = (slope_pivot[args.alpha] - slope_pivot[args.negative_alpha]) / denom
        slope_wide = slope_pivot.pivot_table(
            index=["pair_id", "country"],
            columns="direction_type",
            values="slope",
            aggfunc="first",
        ).reset_index()
        slope_pair_ids = np.array(sorted(slope_wide["pair_id"].unique()))
        for learned, control in comparisons:
            estimate = float((slope_wide[learned] - slope_wide[control]).mean())
            ci = paired_bootstrap_diff(
                slope_wide,
                learned_col=learned,
                control_col=control,
                pair_ids=slope_pair_ids,
                n_samples=args.bootstrap_samples,
                seed=args.seed + 17,
            )
            rows.append(
                {
                    "split": split_name,
                    "metric": f"slope_delta_avg_token_margin_{args.negative_alpha:g}_to_{args.alpha:g}",
                    "comparison": f"{learned}_minus_{control}",
                    "estimate": estimate,
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "ci_unit": "pair_id_block",
                    "bootstrap_samples": ci["bootstrap_samples"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    details = pd.read_csv(args.details)
    enriched = enrich_with_baselines(details)

    decomposition_rows: list[dict[str, object]] = []
    decomposition_rows.extend(
        summarize_decomposition(
            enriched,
            "all_countries",
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    decomposition_rows.extend(
        summarize_decomposition(
            enriched[enriched["split"] == "test"].copy(),
            "heldout_countries",
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    decomposition = pd.DataFrame(decomposition_rows)
    paired = paired_bootstrap(enriched, args)

    decomposition_path = Path(args.out_decomposition)
    paired_path = Path(args.out_paired)
    decomposition_path.parent.mkdir(parents=True, exist_ok=True)
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    decomposition.to_csv(decomposition_path, index=False)
    paired.to_csv(paired_path, index=False)

    key = decomposition[
        (decomposition["split"] == "heldout_countries")
        & (decomposition["direction_type"] == "learned_probe")
        & (decomposition["alpha"] == args.alpha)
    ].iloc[0]
    print(
        "heldout learned alpha={alpha:+.1f}: delta_margin={delta:+.4f}, "
        "delta_correct={dc:+.4f}, delta_false={df:+.4f}, flips={flips}".format(
            alpha=args.alpha,
            delta=float(key["mean_delta_avg_token_margin"]),
            dc=float(key["mean_delta_correct_avg_token_logprob"]),
            df=float(key["mean_delta_false_avg_token_logprob"]),
            flips=int(key["sign_flip_total"]),
        )
    )
    print(f"Saved decomposition diagnostics to {decomposition_path}")
    print(f"Saved paired bootstrap diagnostics to {paired_path}")


if __name__ == "__main__":
    main()
