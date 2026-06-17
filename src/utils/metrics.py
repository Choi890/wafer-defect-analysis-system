from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
) -> dict[str, float | list[list[int]]]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_score": float(precision),
        "recall_score": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def defect_rate(labels: Sequence[str], normal_label: str = "Normal") -> float:
    values = np.asarray(labels)
    if len(values) == 0:
        return 0.0
    return float((values != normal_label).mean())
