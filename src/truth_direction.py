from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def mean_difference_direction(activations: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    true_mean = activations[labels == 1].mean(dim=0)
    false_mean = activations[labels == 0].mean(dim=0)
    direction = true_mean - false_mean
    return direction / (direction.norm() + 1e-8)


def evaluate_layer_probe(activations: torch.Tensor, labels: np.ndarray, seed: int = 42) -> dict[str, float]:
    x = activations.numpy()
    y = labels.astype(int)

    if len(set(y.tolist())) < 2:
        raise ValueError("Probe requires both true and false examples.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
    }


def logit_diff(logits: torch.Tensor, true_token: int, false_token: int) -> torch.Tensor:
    final_logits = logits[:, -1, :]
    return final_logits[:, true_token] - final_logits[:, false_token]
