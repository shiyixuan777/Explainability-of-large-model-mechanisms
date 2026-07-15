from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mean_difference_direction(activations: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    true_mean = activations[labels == 1].mean(dim=0)
    false_mean = activations[labels == 0].mean(dim=0)
    direction = true_mean - false_mean
    return direction / (direction.norm() + 1e-8)


def probe_direction(activations: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    x = activations.numpy()
    y = labels.astype(int)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x, y)
    scaler = clf.named_steps["standardscaler"]
    logistic = clf.named_steps["logisticregression"]
    # Convert the standardized-space linear probe back into the original activation basis.
    direction_np = logistic.coef_[0] / (scaler.scale_ + 1e-8)
    direction = torch.tensor(direction_np, dtype=activations.dtype)
    return direction / (direction.norm() + 1e-8)


def projection_scores(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction.to(activations.device)
    return activations @ direction


def evaluate_layer_probe(
    activations: torch.Tensor,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, float]:
    x = activations.numpy()
    y = labels.astype(int)

    if len(set(y.tolist())) < 2:
        raise ValueError("Probe requires both true and false examples.")

    if groups is None:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=seed, stratify=y
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups))
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", fit_intercept=True, max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, probs))
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": auc,
        "separability_auc": max(auc, 1.0 - auc),
    }


def logit_diff(logits: torch.Tensor, true_token: int, false_token: int) -> torch.Tensor:
    final_logits = logits[:, -1, :]
    return final_logits[:, true_token] - final_logits[:, false_token]
