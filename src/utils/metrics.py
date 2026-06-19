from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    names = list(label_names) if label_names is not None else [str(label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_score": float(precision),
        "recall_score": float(recall),
        "f1_score": float(f1),
        "macro_precision_score": float(macro_precision),
        "macro_recall_score": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": [
            {
                "label": names[index],
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1_score": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(labels))
        ],
    }


def defect_rate(labels: Sequence[str], normal_label: str = "Normal") -> float:
    values = np.asarray(labels)
    if len(values) == 0:
        return 0.0
    return float((values != normal_label).mean())
