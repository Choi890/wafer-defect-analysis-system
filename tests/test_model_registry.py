from __future__ import annotations

from pathlib import Path

from src.models.registry import latest_registered_model, load_model_registry, register_model


def test_register_model_appends_registry_entry(tmp_path):
    registry_path = tmp_path / "model_registry.json"
    entry = register_model(
        model_name="CNN_test",
        model_path=Path("saved_models/test.pt"),
        labels=["Normal", "Center"],
        metrics={"accuracy": 0.9, "precision_score": 0.8, "recall_score": 0.7, "f1_score": 0.75},
        metrics_path=Path("reports/model_metrics.json"),
        confusion_matrix_path=Path("reports/confusion_matrix.csv"),
        registry_path=registry_path,
    )

    registry = load_model_registry(registry_path)

    assert len(registry) == 1
    assert registry[0] == entry
    assert latest_registered_model(registry_path)["model_name"] == "CNN_test"
    assert registry[0]["metrics"]["f1_score"] == 0.75
