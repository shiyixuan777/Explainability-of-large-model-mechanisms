from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
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
    parser.add_argument("--domain", default="capital", help="Comma-separated domain filter, or 'all'")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="figures/error_analysis_capital_layer8.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = None if args.domain == "all" else args.domain
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=domain)
    if data.empty:
        raise ValueError("No rows matched the requested filters.")

    print(f"Using {len(data)} rows from {args.data}")
    print(data["domain"].value_counts().sort_index().to_string())

    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations_by_layer = collect_resid_post_by_layer(model, prompts)
    if args.layer not in activations_by_layer:
        raise ValueError(f"Layer {args.layer} is unavailable for {args.model}.")

    x = activations_by_layer[args.layer].numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, labels, groups=groups))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(x[train_idx], labels[train_idx])

    probs = clf.predict_proba(x[test_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    test_labels = labels[test_idx]

    result = data.iloc[test_idx].copy()
    result["layer"] = args.layer
    result["split"] = "test"
    result["prob_true"] = probs
    result["predicted_label"] = preds
    result["predicted_name"] = result["predicted_label"].map({0: "false", 1: "true"})
    result["label_name"] = result["label"].map({0: "false", 1: "true"})
    result["correct"] = result["predicted_label"] == result["label"]
    result["confidence"] = result["prob_true"].where(result["predicted_label"] == 1, 1 - result["prob_true"])
    result["margin_from_threshold"] = (result["prob_true"] - 0.5).abs()

    accuracy = float(accuracy_score(test_labels, preds))
    auc = float(roc_auc_score(test_labels, probs))
    print(f"layer={args.layer:02d} test_accuracy={accuracy:.3f} test_auc={auc:.3f}")
    print("Errors by domain:")
    print((~result["correct"]).groupby(result["domain"]).sum().sort_index().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["correct", "confidence"], ascending=[True, False]).to_csv(out_path, index=False)

    errors_path = out_path.with_name(f"{out_path.stem}_errors{out_path.suffix}")
    result.loc[~result["correct"]].sort_values("confidence", ascending=False).to_csv(
        errors_path, index=False
    )
    print(f"Saved sample-level predictions to {out_path}")
    print(f"Saved misclassified examples to {errors_path}")


if __name__ == "__main__":
    main()
