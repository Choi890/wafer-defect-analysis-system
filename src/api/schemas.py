from __future__ import annotations

from typing import Literal

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


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    requested_count: int


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    requested_count: int
    completed_count: int
    error_message: str | None = None
    created_at: str
    updated_at: str


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
