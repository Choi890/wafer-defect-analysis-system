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
