from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaferInfo:
    wafer_id: str
    lot_id: str
    failure_type: str
    die_size: int
    inspection_date: str


@dataclass(frozen=True)
class PredictionResult:
    wafer_id: str
    predicted_label: str
    confidence: float
    model_name: str
    created_at: str


@dataclass(frozen=True)
class ModelMetric:
    model_name: str
    accuracy: float
    precision_score: float
    recall_score: float
    f1_score: float
    trained_at: str
