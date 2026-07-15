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
    parser.add_argument("--domain", default="capital")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--max-directions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="figures/iterative_ablation_capital_layer8.csv")
    parser.add_argument(
        "--prompt-template",
        default="Statement: {statement}\nAnswer true or false:",
    )
    return parser.parse_args()


def remove_direction(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction.to(activations.device)
    projection = activations @ direction
    return activations - projection[:, None] * direction[None, :]


def fit_probe(train_x: torch.Tensor, train_y: np.ndarray):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train_x.numpy(), train_y)
    return clf


def direction_from_probe(clf, dtype: torch.dtype) -> torch.Tensor:
    scaler = clf.named_steps["standardscaler"]
    logistic = clf.named_steps["logisticregression"]
    direction_np = logistic.coef_[0] / (scaler.scale_ + 1e-8)
    direction = torch.tensor(direction_np, dtype=dtype)
    return direction / (direction.norm() + 1e-8)


def evaluate_probe(clf, test_x: torch.Tensor, test_y: np.ndarray) -> dict[str, float]:
    probs = clf.predict_proba(test_x.numpy())[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(test_y, probs))
    return {
        "accuracy": float(accuracy_score(test_y, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
    }


def run_control(
    control: str,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    test_x: torch.Tensor,
    test_y: np.ndarray,
    *,
    max_directions: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    current_train = train_x.clone()
    current_test = test_x.clone()
    rows: list[dict[str, object]] = []

    for removed in range(max_directions + 1):
        clf = fit_probe(current_train, train_y)
        metrics = evaluate_probe(clf, current_test, test_y)
        rows.append(
            {
                "control": control,
                "directions_removed": removed,
                **metrics,
            }
        )
        print(
            f"control={control:<17} removed={removed:02d} "
            f"accuracy={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
            f"sep_auc={metrics['separability_auc']:.3f}"
        )
        if removed == max_directions:
            break

        if control == "learned_iterative":
            direction = direction_from_probe(clf, current_train.dtype)
        elif control == "label_permutation":
            permuted_labels = rng.permutation(train_y)
            permuted_clf = fit_probe(current_train, permuted_labels)
            direction = direction_from_probe(permuted_clf, current_train.dtype)
        elif control == "random_direction":
            direction = torch.tensor(rng.normal(size=current_train.shape[1]), dtype=current_train.dtype)
            direction = direction / (direction.norm() + 1e-8)
        else:
            raise ValueError(f"Unknown control: {control}")

        current_train = remove_direction(current_train, direction)
        current_test = remove_direction(current_test, direction)

    return rows


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
    train_x = activations[train_idx]
    test_x = activations[test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]

    all_rows: list[dict[str, object]] = []
    for control in ["learned_iterative", "random_direction", "label_permutation"]:
        rows = run_control(
            control,
            train_x,
            train_y,
            test_x,
            test_y,
            max_directions=args.max_directions,
            seed=args.seed,
        )
        for row in rows:
            row.update(
                {
                    "model": args.model,
                    "domain": args.domain,
                    "layer": args.layer,
                    "seed": args.seed,
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "split": "group",
                }
            )
        all_rows.extend(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved iterative ablation results to {out_path}")


if __name__ == "__main__":
    main()
