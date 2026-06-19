from __future__ import annotations

from src.utils.metrics import classification_metrics


def test_classification_metrics_include_per_class_and_macro_scores():
    metrics = classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        labels=[0, 1],
        label_names=["Normal", "Center"],
    )

    assert metrics["accuracy"] == 0.75
    assert "macro_f1_score" in metrics
    assert metrics["per_class"][0]["label"] == "Normal"
    assert metrics["per_class"][1]["support"] == 2
