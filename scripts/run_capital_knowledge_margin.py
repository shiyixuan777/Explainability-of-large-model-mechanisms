from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


CAPITAL_RE = re.compile(r"^The capital of (?P<country>.+?) is (?P<capital>.+?)\.$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/capital_balanced.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital_balanced")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out-details", default="figures/capital_knowledge_margin_details.csv")
    parser.add_argument("--out-summary", default="figures/capital_knowledge_margin_summary.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def parse_capital_statement(statement: str) -> tuple[str, str]:
    match = CAPITAL_RE.match(statement)
    if match is None:
        raise ValueError(f"Cannot parse capital statement: {statement}")
    return match.group("country"), match.group("capital")


def completion_logprob(model, prompt: str, completion: str) -> tuple[float, int]:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(prompt + completion, prepend_bos=True)
    if full_tokens.shape[1] <= prompt_tokens.shape[1]:
        raise ValueError(f"Completion produced no new tokens: {completion!r}")

    with torch.no_grad():
        logits = model(full_tokens)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    total = 0.0
    for token_pos in range(prompt_tokens.shape[1], full_tokens.shape[1]):
        token_id = int(full_tokens[0, token_pos])
        total += float(log_probs[0, token_pos - 1, token_id])
    return total, int(full_tokens.shape[1] - prompt_tokens.shape[1])


def fit_probe(train_x: torch.Tensor, train_y: np.ndarray):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_x.numpy(), train_y)
    return clf


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(set(labels.tolist())) < 2:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels, scores))
    return auc, max(auc, 1.0 - auc)


def block_bootstrap_auc(
    frame: pd.DataFrame,
    score_col: str,
    *,
    n_samples: int,
    seed: int,
    group_col: str = "pair_id",
    label_col: str = "label",
) -> tuple[float, float, int]:
    groups = np.array(sorted(frame[group_col].unique()))
    if len(groups) == 0 or n_samples <= 0:
        return float("nan"), float("nan"), 0

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    grouped_indices = {
        group: frame.index[frame[group_col] == group].to_numpy()
        for group in groups
    }
    for _ in range(n_samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([grouped_indices[group] for group in sampled_groups])
        sampled = frame.loc[sampled_indices]
        labels = sampled[label_col].to_numpy().astype(int)
        if len(set(labels.tolist())) < 2:
            continue
        scores = sampled[score_col].to_numpy()
        aucs.append(float(roc_auc_score(labels, scores)))

    if not aucs:
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(aucs, 0.025)),
        float(np.quantile(aucs, 0.975)),
        len(aucs),
    )


def metric_row(
    *,
    analysis: str,
    group: str,
    frame: pd.DataFrame,
    score_col: str,
    pred_col: str,
    margin_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    labels = frame["label"].to_numpy().astype(int)
    scores = frame[score_col].to_numpy()
    preds = frame[pred_col].to_numpy().astype(int)
    margins = frame[margin_col].to_numpy()
    auc, sep_auc = safe_auc(labels, scores)
    ci_low, ci_high, bootstrap_samples = block_bootstrap_auc(
        frame,
        score_col,
        n_samples=n_bootstrap,
        seed=seed,
    )
    return {
        "analysis": analysis,
        "group": group,
        "rows": int(len(frame)),
        "blocks": int(frame["pair_id"].nunique()),
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": auc,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "bootstrap_samples": bootstrap_samples,
        "separability_auc": sep_auc,
        "mean_margin": float(np.mean(margins)) if len(margins) else float("nan"),
        "median_margin": float(np.median(margins)) if len(margins) else float("nan"),
        "score_column": score_col,
        "margin_column": margin_col,
        "ci_unit": "pair_id_block",
    }


def append_metric_rows(
    rows: list[dict[str, object]],
    frame_name: str,
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> None:
    specs = [
        (
            "completion_total",
            "completion_total_score",
            "completion_total_predicted_label",
            "completion_total_margin",
        ),
        (
            "completion_avg_token",
            "completion_avg_token_score",
            "completion_avg_token_predicted_label",
            "completion_avg_token_margin",
        ),
        (
            "residual_probe",
            "probe_prob_true",
            "probe_predicted_label",
            "completion_avg_token_margin",
        ),
    ]
    for analysis, score_col, pred_col, margin_col in specs:
        rows.append(
            metric_row(
                analysis=analysis,
                group=frame_name,
                frame=frame,
                score_col=score_col,
                pred_col=pred_col,
                margin_col=margin_col,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain).copy()
    parsed = data["statement"].map(parse_capital_statement)
    data["country"] = [item[0] for item in parsed]
    data["stated_capital"] = [item[1] for item in parsed]

    true_capitals = (
        data[data["label"] == 1]
        .drop_duplicates("country")
        .set_index("country")["stated_capital"]
        .to_dict()
    )
    false_capitals = (
        data[data["label"] == 0]
        .drop_duplicates("country")
        .set_index("country")["stated_capital"]
        .to_dict()
    )
    missing = sorted((set(data["country"]) - set(true_capitals)) | (set(data["country"]) - set(false_capitals)))
    if missing:
        raise ValueError(f"Countries missing true/false capital rows: {missing}")

    model = load_hooked_transformer(args.model)

    country_rows: list[dict[str, object]] = []
    for country in sorted(true_capitals):
        correct = true_capitals[country]
        false = false_capitals[country]
        prompt = f"The capital of {country} is"
        correct_total, correct_tokens = completion_logprob(model, prompt, f" {correct}")
        false_total, false_tokens = completion_logprob(model, prompt, f" {false}")
        correct_avg = correct_total / correct_tokens
        false_avg = false_total / false_tokens
        total_margin = correct_total - false_total
        avg_token_margin = correct_avg - false_avg
        country_rows.append(
            {
                "country": country,
                "correct_capital": correct,
                "false_capital": false,
                "correct_total_logprob": correct_total,
                "false_total_logprob": false_total,
                "correct_avg_token_logprob": correct_avg,
                "false_avg_token_logprob": false_avg,
                "completion_total_margin": total_margin,
                "completion_avg_token_margin": avg_token_margin,
                "model_prefers_correct_total": int(total_margin > 0),
                "model_prefers_correct_avg_token": int(avg_token_margin > 0),
                "correct_completion_tokens": correct_tokens,
                "false_completion_tokens": false_tokens,
            }
        )
        print(
            f"country={country:<24} total_margin={total_margin:+.3f} "
            f"avg_token_margin={avg_token_margin:+.3f} "
            f"tokens={correct_tokens}/{false_tokens}"
        )

    country_df = pd.DataFrame(country_rows)
    data = data.merge(country_df, on="country", how="left")

    data["stated_total_logprob"] = np.where(data["label"] == 1, data["correct_total_logprob"], data["false_total_logprob"])
    data["alternative_total_logprob"] = np.where(data["label"] == 1, data["false_total_logprob"], data["correct_total_logprob"])
    data["stated_avg_token_logprob"] = np.where(
        data["label"] == 1,
        data["correct_avg_token_logprob"],
        data["false_avg_token_logprob"],
    )
    data["alternative_avg_token_logprob"] = np.where(
        data["label"] == 1,
        data["false_avg_token_logprob"],
        data["correct_avg_token_logprob"],
    )
    data["completion_total_score"] = data["stated_total_logprob"] - data["alternative_total_logprob"]
    data["completion_avg_token_score"] = data["stated_avg_token_logprob"] - data["alternative_avg_token_logprob"]
    data["completion_total_predicted_label"] = (data["completion_total_score"] > 0).astype(int)
    data["completion_avg_token_predicted_label"] = (data["completion_avg_token_score"] > 0).astype(int)

    # Keep these aliases for plotting compatibility, but make them length-normalized.
    data["knowledge_margin"] = data["completion_avg_token_margin"]
    data["completion_compatibility_score"] = data["completion_avg_token_score"]
    data["completion_predicted_label"] = data["completion_avg_token_predicted_label"]

    median_avg_margin = float(country_df["completion_avg_token_margin"].median())
    data["knowledge_bin"] = np.where(
        data["completion_avg_token_margin"] >= median_avg_margin,
        "high_avg_token_margin",
        "low_avg_token_margin",
    )

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    clf = fit_probe(activations[train_idx], labels[train_idx])
    probe_probs = clf.predict_proba(activations.numpy())[:, 1]
    data["probe_prob_true"] = probe_probs
    data["probe_predicted_label"] = (probe_probs >= 0.5).astype(int)
    data["split"] = "train"
    data.loc[test_idx, "split"] = "test"
    data["probe_layer"] = args.layer
    data["seed"] = args.seed

    summary_rows: list[dict[str, object]] = []
    test_data = data[data["split"] == "test"].copy()
    append_metric_rows(
        summary_rows,
        "all_rows",
        data,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    append_metric_rows(
        summary_rows,
        "heldout_rows",
        test_data,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
    )

    for bin_name, frame in test_data.groupby("knowledge_bin", sort=True):
        append_metric_rows(
            summary_rows,
            f"heldout_{bin_name}",
            frame,
            n_bootstrap=args.bootstrap_samples,
            seed=args.seed + 2,
        )

    details_out = Path(args.out_details)
    summary_out = Path(args.out_summary)
    details_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(details_out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    print(f"Saved knowledge-margin details to {details_out}")
    print(f"Saved knowledge-margin summary to {summary_out}")


if __name__ == "__main__":
    main()
