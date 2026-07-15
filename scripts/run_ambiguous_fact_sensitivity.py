from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_completion_margin_steering import (
    build_country_frame,
    completion_logprobs,
    make_directions,
    parse_capital_statement,
)
from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


AMBIGUOUS_COUNTRIES = {"South Africa", "Bolivia", "Israel"}
AMBIGUOUS_CAPITALS = {"Pretoria", "Sucre", "Jerusalem"}


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
    parser.add_argument("--out-data", default="data/capital_balanced_no_ambiguous.csv")
    parser.add_argument("--out-details", default="figures/ambiguous_fact_sensitivity_details.csv")
    parser.add_argument("--out-summary", default="figures/ambiguous_fact_sensitivity_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def fit_probe(train_x: torch.Tensor, train_y: np.ndarray):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_x.numpy(), train_y)
    return clf


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def filter_ambiguous_blocks(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    country_frame = build_country_frame(data)
    ambiguous_pair_ids = sorted(
        country_frame[
            country_frame["country"].isin(AMBIGUOUS_COUNTRIES)
            | country_frame["correct_capital"].isin(AMBIGUOUS_CAPITALS)
            | country_frame["false_capital"].isin(AMBIGUOUS_CAPITALS)
        ]["pair_id"].unique()
    )
    filtered = data[~data["pair_id"].isin(ambiguous_pair_ids)].copy().reset_index(drop=True)
    return filtered, ambiguous_pair_ids


def completion_margins(model, countries: pd.DataFrame, *, direction=None, alpha: float = 0.0, layer: int = 6, position_mode: str = "prompt-final-only"):
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
    totals: list[float] = []
    avgs: list[float] = []
    correct_avgs: list[float] = []
    false_avgs: list[float] = []
    for (correct_total, correct_tokens), (false_total, false_tokens) in zip(correct_scores, false_scores):
        correct_avg = correct_total / correct_tokens
        false_avg = false_total / false_tokens
        totals.append(correct_total - false_total)
        avgs.append(correct_avg - false_avg)
        correct_avgs.append(correct_avg)
        false_avgs.append(false_avg)
    return np.array(totals), np.array(avgs), np.array(correct_avgs), np.array(false_avgs)


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    filtered, removed_pair_ids = filter_ambiguous_blocks(data)
    out_data = Path(args.out_data)
    out_data.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_data, index=False)

    prompts = make_prompts(filtered["statement"].tolist(), args.prompt_template)
    labels = filtered["label"].to_numpy().astype(int)
    groups = filtered["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    train_pair_ids = set(filtered.iloc[train_idx]["pair_id"])
    test_pair_ids = set(filtered.iloc[test_idx]["pair_id"])

    clf = fit_probe(activations[train_idx], labels[train_idx])
    probs = clf.predict_proba(activations.numpy())[:, 1]
    filtered["probe_prob_true"] = probs
    filtered["probe_predicted_label"] = (probs >= 0.5).astype(int)
    filtered["split"] = np.where(filtered["pair_id"].isin(test_pair_ids), "test", "train")

    countries = build_country_frame(filtered)
    countries["split"] = np.where(countries["pair_id"].isin(test_pair_ids), "test", "train")
    total_margin, avg_margin, _, _ = completion_margins(
        model,
        countries,
        layer=args.layer,
        position_mode=args.position_mode,
    )
    countries["baseline_total_margin"] = total_margin
    countries["baseline_avg_token_margin"] = avg_margin
    countries["baseline_prefers_correct"] = avg_margin > 0

    directions = make_directions(activations[train_idx], labels[train_idx], args.seed)
    directions = {name: direction.to(model.cfg.device) for name, direction in directions.items()}
    heldout_countries = countries[countries["split"] == "test"].copy().reset_index(drop=True)

    steering_rows: list[dict[str, object]] = []
    for direction_name, direction in directions.items():
        _, steered_avg_margin, steered_correct_avg, steered_false_avg = completion_margins(
            model,
            heldout_countries,
            direction=direction,
            alpha=args.alpha,
            layer=args.layer,
            position_mode=args.position_mode,
        )
        delta = steered_avg_margin - heldout_countries["baseline_avg_token_margin"].to_numpy()
        prefers_correct = steered_avg_margin > 0
        sign_flips = prefers_correct != heldout_countries["baseline_prefers_correct"].to_numpy()
        steering_rows.append(
            {
                "analysis": "prompt_final_steering",
                "direction": direction_name,
                "layer": args.layer,
                "alpha": args.alpha,
                "position_mode": args.position_mode,
                "heldout_blocks": int(heldout_countries["pair_id"].nunique()),
                "heldout_countries": int(len(heldout_countries)),
                "mean_delta_avg_token_margin": float(delta.mean()),
                "pairwise_avg_accuracy": float(prefers_correct.mean()),
                "sign_flips": int(sign_flips.sum()),
            }
        )

    test_rows = filtered[filtered["split"] == "test"].copy()
    parsed = test_rows["statement"].map(parse_capital_statement)
    test_rows["country"] = [item[0] for item in parsed]
    test_rows = test_rows.merge(
        countries[["pair_id", "country", "baseline_total_margin", "baseline_avg_token_margin"]],
        on=["pair_id", "country"],
        how="left",
    )
    test_rows["completion_total_score"] = np.where(
        test_rows["label"] == 1,
        test_rows["baseline_total_margin"],
        -test_rows["baseline_total_margin"],
    )
    test_rows["completion_avg_token_score"] = np.where(
        test_rows["label"] == 1,
        test_rows["baseline_avg_token_margin"],
        -test_rows["baseline_avg_token_margin"],
    )
    test_rows["completion_total_predicted_label"] = (test_rows["completion_total_score"] > 0).astype(int)
    test_rows["completion_avg_token_predicted_label"] = (test_rows["completion_avg_token_score"] > 0).astype(int)

    probe_auc = safe_auc(test_rows["label"].to_numpy(), test_rows["probe_prob_true"].to_numpy())
    completion_total_auc = safe_auc(test_rows["label"].to_numpy(), test_rows["completion_total_score"].to_numpy())
    completion_avg_auc = safe_auc(test_rows["label"].to_numpy(), test_rows["completion_avg_token_score"].to_numpy())

    summary_rows = [
        {
            "analysis": "dataset",
            "rows": int(len(filtered)),
            "blocks": int(filtered["pair_id"].nunique()),
            "removed_blocks": ";".join(removed_pair_ids),
            "removed_block_count": int(len(removed_pair_ids)),
        },
        {
            "analysis": "residual_probe",
            "layer": args.layer,
            "heldout_rows": int(len(test_rows)),
            "heldout_blocks": int(test_rows["pair_id"].nunique()),
            "accuracy": float(accuracy_score(test_rows["label"], test_rows["probe_predicted_label"])),
            "auc": probe_auc,
        },
        {
            "analysis": "completion_total",
            "heldout_rows": int(len(test_rows)),
            "heldout_blocks": int(test_rows["pair_id"].nunique()),
            "accuracy": float(accuracy_score(test_rows["label"], test_rows["completion_total_predicted_label"])),
            "auc": completion_total_auc,
        },
        {
            "analysis": "completion_avg_token",
            "heldout_rows": int(len(test_rows)),
            "heldout_blocks": int(test_rows["pair_id"].nunique()),
            "accuracy": float(accuracy_score(test_rows["label"], test_rows["completion_avg_token_predicted_label"])),
            "auc": completion_avg_auc,
        },
        *steering_rows,
    ]

    details = countries.copy()
    details["removed_blocks"] = ";".join(removed_pair_ids)
    details_path = Path(args.out_details)
    summary_path = Path(args.out_summary)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Removed ambiguous blocks: {', '.join(removed_pair_ids)}")
    print(f"Saved filtered dataset to {out_data}")
    print(f"Saved ambiguous sensitivity details to {details_path}")
    print(f"Saved ambiguous sensitivity summary to {summary_path}")


if __name__ == "__main__":
    main()
