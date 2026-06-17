from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.config import DATABASE_PATH
from src.data.load_data import load_or_create_raw_dataset
from src.database import repository
from src.models.predict import Predictor


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", database_ready=DATABASE_PATH.exists())


def _lookup_wafer(wafer_id: str) -> tuple[np.ndarray, dict[str, object]]:
    df = load_or_create_raw_dataset()
    match = df[df["wafer_id"] == wafer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown wafer_id: {wafer_id}")
    row = match.iloc[0]
    return np.asarray(row["wafer_map"]), row.to_dict()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    repository.bootstrap_database()
    wafer_id = request.wafer_id or f"API_{uuid.uuid4().hex[:10].upper()}"

    if request.wafer_map is not None:
        wafer_map = np.asarray(request.wafer_map, dtype=np.uint8)
        wafer_info = pd.DataFrame(
            [
                {
                    "wafer_id": wafer_id,
                    "lot_id": request.lot_id or "LOT_API",
                    "failure_type": "Unknown",
                    "die_size": int((wafer_map > 0).sum()),
                    "inspection_date": pd.Timestamp.utcnow().date().isoformat(),
                }
            ]
        )
        repository.upsert_wafer_info(wafer_info)
    else:
        wafer_map, _ = _lookup_wafer(wafer_id)

    result = Predictor().predict(wafer_map)
    repository.insert_prediction(
        wafer_id=wafer_id,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        model_name=result["model_name"],
    )
    return PredictResponse(wafer_id=wafer_id, **result)


@router.get("/results")
def results(limit: int = Query(default=100, ge=1, le=1000)):
    repository.bootstrap_database()
    return repository.list_results(limit=limit)


@router.get("/results/{wafer_id}")
def result_by_wafer(wafer_id: str):
    repository.bootstrap_database()
    result = repository.get_result_by_wafer_id(wafer_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for {wafer_id}")
    return result


@router.get("/metrics")
def metrics():
    repository.bootstrap_database()
    return {
        "latest": repository.get_latest_metrics(),
        "history": repository.get_metrics_history(),
    }


@router.get("/statistics/lot")
def lot_statistics():
    repository.bootstrap_database()
    return repository.get_lot_statistics()


@router.get("/statistics/defect")
def defect_statistics():
    repository.bootstrap_database()
    return repository.get_defect_statistics()
