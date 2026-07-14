from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import filter_dataset, load_dataset
from src.model_hooks import collect_resid_post_by_layer, load_hooked_transformer, make_prompts
from src.truth_direction import mean_difference_direction, probe_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--data", default="data/facts.csv")
    parser.add_argument("--language", default="en")
    parser.add_argument("--domain", default="capital")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--direction-method", choices=["probe", "mean_diff"], default="probe")
    parser.add_argument("--out", default="figures/ablation_capital_probe_layer8.csv")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def remove_direction(activations: torch.Tensor, direction: torch.Tensor, strength: float) -> torch.Tensor:
    direction = direction.to(activations.device)
    projection = activations @ direction
    return activations - strength * projection[:, None] * direction[None, :]


def fit_probe_and_evaluate(
    train_activations: torch.Tensor,
    train_labels,
    test_activations: torch.Tensor,
    test_labels,
) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_activations.numpy(), train_labels)
    probs = clf.predict_proba(test_activations.numpy())[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(test_labels, probs))
    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
    }


def main() -> None:
    args = parse_args()
    data = filter_dataset(load_dataset(args.data), language=args.language, domain=args.domain)
    prompts = make_prompts(data["statement"].tolist(), args.prompt_template)
    labels = data["label"].to_numpy().astype(int)
    groups = data["pair_id"].to_numpy()

    model = load_hooked_transformer(args.model)
    activations = collect_resid_post_by_layer(model, prompts)[args.layer]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(activations.numpy(), labels, groups=groups))
    train_activations = activations[train_idx]
    test_activations = activations[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    if args.direction_method == "probe":
        direction = probe_direction(train_activations, train_labels)
    else:
        direction = mean_difference_direction(train_activations, train_labels)

    baseline = fit_probe_and_evaluate(train_activations, train_labels, test_activations, test_labels)
    train_scores = train_activations @ direction
    fixed_threshold = float(0.5 * (train_scores[train_labels == 1].mean() + train_scores[train_labels == 0].mean()))
    baseline_fixed_scores = test_activations @ direction
    baseline_fixed_preds = (baseline_fixed_scores.numpy() > fixed_threshold).astype(int)
    baseline_fixed_accuracy = float(accuracy_score(test_labels, baseline_fixed_preds))
    baseline_fixed_auc = float(roc_auc_score(test_labels, baseline_fixed_scores.numpy()))
    baseline_fixed_gap = float(
        baseline_fixed_scores[test_labels == 1].mean() - baseline_fixed_scores[test_labels == 0].mean()
    )
    print(
        f"baseline accuracy={baseline['accuracy']:.3f} "
        f"auc={baseline['auc']:.3f} sep_auc={baseline['separability_auc']:.3f} "
        f"fixed_direction_acc={baseline_fixed_accuracy:.3f}"
    )

    rows = []
    for strength in args.strengths:
        ablated_train = remove_direction(train_activations, direction, strength)
        ablated_test = remove_direction(test_activations, direction, strength)
        metrics = fit_probe_and_evaluate(ablated_train, train_labels, ablated_test, test_labels)
        fixed_scores = ablated_test @ direction
        fixed_preds = (fixed_scores.numpy() > fixed_threshold).astype(int)
        fixed_auc = float(roc_auc_score(test_labels, fixed_scores.numpy()))
        fixed_accuracy = float(accuracy_score(test_labels, fixed_preds))
        fixed_gap = float(fixed_scores[test_labels == 1].mean() - fixed_scores[test_labels == 0].mean())
        fixed_std = float(fixed_scores.std())
        rows.append(
            {
                "layer": args.layer,
                "direction_method": args.direction_method,
                "strength": strength,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_auc": baseline["auc"],
                "baseline_separability_auc": baseline["separability_auc"],
                "fixed_probe_threshold": fixed_threshold,
                "baseline_fixed_direction_accuracy": baseline_fixed_accuracy,
                "baseline_fixed_direction_auc": baseline_fixed_auc,
                "baseline_fixed_direction_score_gap": baseline_fixed_gap,
                "fixed_direction_accuracy": fixed_accuracy,
                "fixed_direction_auc": fixed_auc,
                "fixed_direction_separability_auc": max(fixed_auc, 1.0 - fixed_auc),
                "fixed_direction_score_gap": fixed_gap,
                "fixed_direction_score_std": fixed_std,
                **metrics,
            }
        )
        print(
            f"strength={strength:.2f} accuracy={metrics['accuracy']:.3f} "
            f"auc={metrics['auc']:.3f} sep_auc={metrics['separability_auc']:.3f} "
            f"fixed_acc={fixed_accuracy:.3f} fixed_gap={fixed_gap:+.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved ablation results to {out_path}")


if __name__ == "__main__":
    main()
