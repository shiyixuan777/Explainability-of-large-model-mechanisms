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

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--out-transfer", default="figures/domain_transfer_layer8.csv")
    parser.add_argument("--out-cosine", default="figures/domain_direction_cosine_layer8.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def fit_probe(train_x: np.ndarray, train_y: np.ndarray):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_x, train_y)
    return clf


def direction_from_probe(clf) -> np.ndarray:
    scaler = clf.named_steps["standardscaler"]
    logistic = clf.named_steps["logisticregression"]
    direction = logistic.coef_[0] / (scaler.scale_ + 1e-8)
    return direction / (np.linalg.norm(direction) + 1e-8)


def evaluate(clf, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    probs = clf.predict_proba(test_x)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(test_y, probs))
    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
        "predicted_true_rate": float(preds.mean()),
    }


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language)
    domain_counts = data["domain"].value_counts()
    domains = sorted(domain for domain, count in domain_counts.items() if int(count) >= args.min_rows)
    if len(domains) < 2:
        raise ValueError("Need at least two domains for domain-consistency analysis.")

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer].numpy()

    domain_indices = {domain: np.flatnonzero(data["domain"].to_numpy() == domain) for domain in domains}
    domain_clfs = {}
    domain_directions = {}
    for domain in domains:
        idx = domain_indices[domain]
        clf = fit_probe(activations[idx], labels[idx])
        domain_clfs[domain] = clf
        domain_directions[domain] = direction_from_probe(clf)

    transfer_rows: list[dict[str, object]] = []
    for source in domains:
        for target in domains:
            target_idx = domain_indices[target]
            if source == target:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
                source_idx = domain_indices[source]
                train_local, test_local = next(
                    splitter.split(activations[source_idx], labels[source_idx], groups=groups[source_idx])
                )
                train_idx = source_idx[train_local]
                test_idx = source_idx[test_local]
                clf = fit_probe(activations[train_idx], labels[train_idx])
                split = "source_group_heldout"
            else:
                source_idx = domain_indices[source]
                test_idx = target_idx
                clf = domain_clfs[source]
                split = "source_all_to_target_all"

            metrics = evaluate(clf, activations[test_idx], labels[test_idx])
            transfer_rows.append(
                {
                    "model": args.model,
                    "layer": args.layer,
                    "seed": args.seed,
                    "source_domain": source,
                    "target_domain": target,
                    "source_rows": int(len(domain_indices[source])),
                    "target_rows": int(len(test_idx)),
                    "split": split,
                    **metrics,
                }
            )
            print(
                f"source={source:<18} target={target:<18} "
                f"accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
                f"sep_auc={metrics['separability_auc']:.3f}"
            )

    cosine_rows: list[dict[str, object]] = []
    for source in domains:
        for target in domains:
            cosine = float(np.dot(domain_directions[source], domain_directions[target]))
            cosine_rows.append(
                {
                    "model": args.model,
                    "layer": args.layer,
                    "source_domain": source,
                    "target_domain": target,
                    "cosine_similarity": cosine,
                }
            )

    transfer_path = Path(args.out_transfer)
    transfer_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)
    print(f"Saved domain transfer results to {transfer_path}")

    cosine_path = Path(args.out_cosine)
    cosine_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cosine_rows).to_csv(cosine_path, index=False)
    print(f"Saved direction cosine results to {cosine_path}")


if __name__ == "__main__":
    main()
