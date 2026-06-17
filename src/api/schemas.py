from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    wafer_id: str | None = Field(default=None, examples=["WAFER_000001"])
    lot_id: str | None = Field(default=None, examples=["LOT_A001"])
    wafer_map: list[list[int]] | None = None


class PredictResponse(BaseModel):
    wafer_id: str
    predicted_label: str
    confidence: float
    model_name: str


class HealthResponse(BaseModel):
    status: str
    database_ready: bool
    model_ready: bool
    wafer_count: int
    prediction_count: int
    version: str
    environment: str


class QualitySummaryResponse(BaseModel):
    total_wafers: int
    total_lots: int
    defect_wafers: int
    defect_rate: float
    top_defect: str
    risk_lots: int
    critical_lots: int
    avg_confidence: float
    low_confidence_count: int
    model_f1: float
