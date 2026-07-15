from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import filter_dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domains", nargs="+", default=["all", "capital"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="figures/surface_baselines.csv")
    return parser.parse_args()


def surface_features(statements: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "char_len": statements.str.len(),
            "word_count": statements.str.split().str.len(),
            "comma_count": statements.str.count(","),
            "digit_count": statements.str.count(r"\d"),
            "uppercase_count": statements.apply(lambda text: sum(ch.isupper() for ch in str(text))),
            "period_count": statements.str.count(r"\."),
        }
    )


def evaluate_numeric_features(train_df, test_df, train_y, test_y) -> dict[str, float]:
    train_x = surface_features(train_df["statement"])
    test_x = surface_features(test_df["statement"])
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_x, train_y)
    probs = clf.predict_proba(test_x)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(test_y, probs))
    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
    }


def evaluate_bow(train_df, test_df, train_y, test_y) -> dict[str, float]:
    clf = make_pipeline(
        CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_df["statement"], train_y)
    probs = clf.predict_proba(test_df["statement"])[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(test_y, probs))
    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
    }


def split_grouped(data: pd.DataFrame, seed: int):
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(data)), labels, groups=groups))
    train_df = data.iloc[train_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df, labels[train_idx], labels[test_idx]


def main() -> None:
    args = parse_args()
    full_data = filter_dataset(load_dataset(args.data), language=args.language)

    rows: list[dict[str, object]] = []
    for domain in args.domains:
        if domain == "all":
            data = full_data.reset_index(drop=True)
            domain_name = "all"
        else:
            data = filter_dataset(full_data, domain=domain)
            domain_name = domain

        train_df, test_df, train_y, test_y = split_grouped(data, args.seed)
        for baseline_name, evaluator in [
            ("numeric_surface", evaluate_numeric_features),
            ("bag_of_words", evaluate_bow),
        ]:
            metrics = evaluator(train_df, test_df, train_y, test_y)
            rows.append(
                {
                    "domain": domain_name,
                    "baseline": baseline_name,
                    "seed": args.seed,
                    "split": "group",
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                    **metrics,
                }
            )
            print(
                f"domain={domain_name:<8} baseline={baseline_name:<16} "
                f"accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
                f"sep_auc={metrics['separability_auc']:.3f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved surface baselines to {out_path}")


if __name__ == "__main__":
    main()
