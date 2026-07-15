from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from scripts.run_completion_margin_steering import build_country_frame, make_directions
from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/capital_balanced.csv")
    parser.add_argument("--details", default="figures/completion_margin_steering_details.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital_balanced")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-details", default="figures/unembedding_projection_baseline_details.csv")
    parser.add_argument("--out-summary", default="figures/unembedding_projection_baseline_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def completion_token_ids(model, prompt: str, completion: str) -> list[int]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(prompt + completion, prepend_bos=True)
    prompt_len = int(prompt_tokens.shape[1])
    full_len = int(full_tokens.shape[1])
    if full_len <= prompt_len:
        raise ValueError(f"Completion produced no new tokens: {completion!r}")
    return [int(token) for token in full_tokens[0, prompt_len:full_len].tolist()]


def average_unembedding_projection(model, direction: torch.Tensor, token_ids: list[int]) -> float:
    w_u = model.W_U.detach().to(direction.device)
    direction = direction.to(w_u.device)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=w_u.device)
    projections = direction @ w_u[:, token_tensor]
    return float(projections.mean().detach().cpu())


def corr_or_nan(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or float(left.std(ddof=0)) == 0.0 or float(right.std(ddof=0)) == 0.0:
        return float("nan")
    return float(left.corr(right))


def regression_slope_or_nan(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or float(x.var(ddof=0)) == 0.0:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / x.var(ddof=0))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, direction_type, alpha), group in frame.groupby(["split", "direction_type", "alpha"], sort=True):
        residual = group["observed_delta_avg_token_margin"] - group["predicted_delta_avg_token_margin"]
        corr = corr_or_nan(group["predicted_delta_avg_token_margin"], group["observed_delta_avg_token_margin"])
        rows.append(
            {
                "split": split,
                "direction_type": direction_type,
                "alpha": float(alpha),
                "countries": int(len(group)),
                "blocks": int(group["pair_id"].nunique()),
                "mean_observed_delta_avg_token_margin": float(group["observed_delta_avg_token_margin"].mean()),
                "mean_predicted_delta_avg_token_margin": float(group["predicted_delta_avg_token_margin"].mean()),
                "mean_abs_observed_delta_avg_token_margin": float(
                    group["observed_delta_avg_token_margin"].abs().mean()
                ),
                "mean_abs_predicted_delta_avg_token_margin": float(
                    group["predicted_delta_avg_token_margin"].abs().mean()
                ),
                "std_observed_delta_avg_token_margin": float(group["observed_delta_avg_token_margin"].std(ddof=1)),
                "std_predicted_delta_avg_token_margin": float(group["predicted_delta_avg_token_margin"].std(ddof=1)),
                "mean_residual_observed_minus_predicted": float(residual.mean()),
                "mean_abs_residual": float(residual.abs().mean()),
                "corr_predicted_observed": corr,
                "corr_squared_predicted_observed": float(corr * corr) if not np.isnan(corr) else float("nan"),
                "regression_slope_observed_on_predicted": regression_slope_or_nan(
                    group["predicted_delta_avg_token_margin"], group["observed_delta_avg_token_margin"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
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
    country_data["split"] = np.where(country_data["pair_id"].isin(test_pair_ids), "test", "train")

    directions = make_directions(activations[train_idx], labels[train_idx], args.seed)
    directions = {name: direction.to(model.cfg.device) for name, direction in directions.items()}

    projection_rows: list[dict[str, object]] = []
    for direction_type, direction in directions.items():
        for row in country_data.itertuples(index=False):
            prompt = row.completion_prompt
            correct_tokens = completion_token_ids(model, prompt, f" {row.correct_capital}")
            false_tokens = completion_token_ids(model, prompt, f" {row.false_capital}")
            correct_projection = average_unembedding_projection(model, direction, correct_tokens)
            false_projection = average_unembedding_projection(model, direction, false_tokens)
            projection_rows.append(
                {
                    "direction_type": direction_type,
                    "layer": args.layer,
                    "seed": args.seed,
                    "split": row.split,
                    "pair_id": row.pair_id,
                    "country": row.country,
                    "correct_capital": row.correct_capital,
                    "false_capital": row.false_capital,
                    "correct_completion_tokens": len(correct_tokens),
                    "false_completion_tokens": len(false_tokens),
                    "correct_unembedding_projection_avg": correct_projection,
                    "false_unembedding_projection_avg": false_projection,
                    "unembedding_projection_margin": correct_projection - false_projection,
                    "projection_note": "static W_U projection of the steering direction; ignores final layernorm and downstream layers",
                }
            )

    projections = pd.DataFrame(projection_rows)
    observed = pd.read_csv(args.details)
    observed = observed.rename(columns={"delta_avg_token_margin": "observed_delta_avg_token_margin"})
    join_cols = ["direction_type", "split", "pair_id", "country"]
    merged = observed.merge(
        projections,
        on=join_cols,
        how="left",
        suffixes=("", "_projection"),
        validate="many_to_one",
    )
    if merged["unembedding_projection_margin"].isna().any():
        missing = merged.loc[merged["unembedding_projection_margin"].isna(), join_cols].head()
        raise ValueError(f"Missing projection rows for:\n{missing}")

    merged["predicted_delta_avg_token_margin"] = merged["alpha"] * merged["unembedding_projection_margin"]
    merged["projection_residual_observed_minus_predicted"] = (
        merged["observed_delta_avg_token_margin"] - merged["predicted_delta_avg_token_margin"]
    )
    summary = summarize(merged)

    details_path = Path(args.out_details)
    summary_path = Path(args.out_summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    key = summary[
        (summary["split"] == "test")
        & (summary["direction_type"] == "learned_probe")
        & (summary["alpha"] == 4.0)
    ].iloc[0]
    print(
        "heldout learned alpha=+4: observed={obs:+.4f}, predicted={pred:+.4f}, corr={corr:.3f}".format(
            obs=float(key["mean_observed_delta_avg_token_margin"]),
            pred=float(key["mean_predicted_delta_avg_token_margin"]),
            corr=float(key["corr_predicted_observed"]),
        )
    )
    print(f"Saved unembedding projection details to {details_path}")
    print(f"Saved unembedding projection summary to {summary_path}")


if __name__ == "__main__":
    main()
