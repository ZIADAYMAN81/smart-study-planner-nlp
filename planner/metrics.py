"""Shared evaluation helpers."""
from __future__ import annotations

from typing import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .data import LABEL_NAMES


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "labels": LABEL_NAMES,
        "test_size": len(y_true),
    }
