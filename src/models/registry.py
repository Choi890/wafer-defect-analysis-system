from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import IMAGE_SIZE, MODEL_REGISTRY_PATH, ensure_directories


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Model registry must be a JSON list: {path}")


def latest_registered_model(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any] | None:
    registry = load_model_registry(path)
    return registry[-1] if registry else None


def register_model(
    model_name: str,
    model_path: Path,
    labels: list[str],
    metrics: dict[str, Any],
    metrics_path: Path,
    confusion_matrix_path: Path,
    classification_report_path: Path | None = None,
    training_history_path: Path | None = None,
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    ensure_directories()
    entry = {
        "model_name": model_name,
        "model_path": str(model_path),
        "labels": labels,
        "image_size": IMAGE_SIZE,
        "framework": "torch",
        "metrics": {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision_score": float(metrics.get("precision_score", 0.0)),
            "recall_score": float(metrics.get("recall_score", 0.0)),
            "f1_score": float(metrics.get("f1_score", 0.0)),
            "macro_f1_score": float(metrics.get("macro_f1_score", 0.0)),
            "best_validation_f1": float(metrics.get("best_validation_f1", 0.0)),
        },
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "classification_report_path": str(classification_report_path) if classification_report_path else None,
        "training_history_path": str(training_history_path) if training_history_path else None,
        "registered_at": _utc_now(),
    }
    registry = load_model_registry(registry_path)
    registry.append(entry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return entry
