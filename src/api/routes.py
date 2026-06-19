from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status

from src.api.auth import require_api_key
from src.api.schemas import (
    BatchJobResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    QualitySummaryResponse,
)
from src.config import APP_ENV, APP_VERSION, DATABASE_PATH, MODEL_PATH
from src.data.load_data import load_or_create_raw_dataset
from src.database import repository
from src.models.predict import Predictor
from src.models.registry import latest_registered_model, load_model_registry
from src.utils.quality import build_quality_summary


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    repository.bootstrap_database()
    return HealthResponse(
        status="ok",
        database_ready=DATABASE_PATH.exists(),
        model_ready=MODEL_PATH.exists(),
        wafer_count=repository.get_table_count("wafer_info"),
        prediction_count=repository.get_table_count("prediction_result"),
        version=APP_VERSION,
        environment=APP_ENV,
    )


def _lookup_wafer(wafer_id: str) -> tuple[np.ndarray, dict[str, object]]:
    df = load_or_create_raw_dataset()
    match = df[df["wafer_id"] == wafer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown wafer_id: {wafer_id}")
    row = match.iloc[0]
    return np.asarray(row["wafer_map"]), row.to_dict()


def _predict_and_persist(request: PredictRequest, predictor: Predictor | None = None) -> PredictResponse:
    predictor = predictor or Predictor()
    wafer_id = request.wafer_id or f"API_{uuid.uuid4().hex[:10].upper()}"

    if request.wafer_map is not None:
        wafer_map = np.asarray(request.wafer_map, dtype=np.uint8)
        if wafer_map.ndim != 2:
            raise HTTPException(status_code=400, detail="wafer_map must be a 2D array")
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
        if request.wafer_id is None:
            raise HTTPException(status_code=400, detail="Provide either wafer_id or wafer_map")
        wafer_map, _ = _lookup_wafer(wafer_id)

    result = predictor.predict(wafer_map)
    repository.insert_prediction(
        wafer_id=wafer_id,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        model_name=result["model_name"],
    )
    return PredictResponse(wafer_id=wafer_id, **result)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    repository.bootstrap_database()
    return _predict_and_persist(request)


def _run_batch_prediction(job_id: str, items: list[PredictRequest]) -> None:
    completed = 0
    repository.update_batch_job(job_id, status="running", completed_count=completed)
    predictor = Predictor()
    try:
        for item in items:
            _predict_and_persist(item, predictor=predictor)
            completed += 1
            repository.update_batch_job(job_id, completed_count=completed)
    except Exception as exc:
        repository.update_batch_job(
            job_id,
            status="failed",
            completed_count=completed,
            error_message=str(exc),
        )
        return
    repository.update_batch_job(job_id, status="completed", completed_count=completed)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def batch_predict(request: BatchPredictRequest, background_tasks: BackgroundTasks) -> BatchPredictResponse:
    repository.bootstrap_database()
    job_id = f"JOB_{uuid.uuid4().hex[:12].upper()}"
    repository.create_batch_job(job_id, requested_count=len(request.items))
    background_tasks.add_task(_run_batch_prediction, job_id, request.items)
    return BatchPredictResponse(job_id=job_id, status="queued", requested_count=len(request.items))


@router.get("/jobs", response_model=list[BatchJobResponse])
def batch_jobs(limit: int = Query(default=50, ge=1, le=200)):
    repository.bootstrap_database()
    return repository.list_batch_jobs(limit=limit)


@router.get("/jobs/{job_id}", response_model=BatchJobResponse)
def batch_job(job_id: str):
    repository.bootstrap_database()
    result = repository.get_batch_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No batch job found for {job_id}")
    return result


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


@router.get("/models")
def models():
    return {
        "latest": latest_registered_model(),
        "registry": load_model_registry(),
    }


@router.get("/summary", response_model=QualitySummaryResponse)
def quality_summary() -> QualitySummaryResponse:
    repository.bootstrap_database()
    wafer_df = repository.get_wafer_info_frame()
    result_df = repository.get_latest_results_frame()
    summary = build_quality_summary(wafer_df, result_df, repository.get_latest_metrics())
    return QualitySummaryResponse(**summary)


@router.get("/statistics/lot")
def lot_statistics():
    repository.bootstrap_database()
    return repository.get_lot_statistics()


@router.get("/statistics/defect")
def defect_statistics():
    repository.bootstrap_database()
    return repository.get_defect_statistics()


def _build_monitoring_snapshot() -> dict[str, Any]:
    repository.bootstrap_database()
    wafer_df = repository.get_wafer_info_frame()
    result_df = repository.get_latest_results_frame()
    summary = build_quality_summary(wafer_df, result_df, repository.get_latest_metrics())
    return {
        "status": "ok",
        "version": APP_VERSION,
        "environment": APP_ENV,
        "database_ready": DATABASE_PATH.exists(),
        "model_ready": MODEL_PATH.exists(),
        "summary": summary,
        "latest_metric": repository.get_latest_metrics(),
        "recent_batch_jobs": repository.list_batch_jobs(limit=5),
        "model_registry_entries": len(load_model_registry()),
    }


@router.get("/monitoring")
def monitoring():
    return _build_monitoring_snapshot()


@router.get("/monitoring/prometheus")
def prometheus_metrics():
    snapshot = _build_monitoring_snapshot()
    summary = snapshot["summary"]
    latest_metric = snapshot["latest_metric"] or {}
    lines = [
        "# HELP wafer_total Total wafers tracked by the quality system.",
        "# TYPE wafer_total gauge",
        f"wafer_total {summary['total_wafers']}",
        "# HELP wafer_defect_rate Overall wafer defect rate.",
        "# TYPE wafer_defect_rate gauge",
        f"wafer_defect_rate {summary['defect_rate']}",
        "# HELP wafer_risk_lots Lots above warning or critical defect thresholds.",
        "# TYPE wafer_risk_lots gauge",
        f"wafer_risk_lots {summary['risk_lots']}",
        "# HELP wafer_low_confidence_predictions Predictions below the confidence threshold.",
        "# TYPE wafer_low_confidence_predictions gauge",
        f"wafer_low_confidence_predictions {summary['low_confidence_count']}",
        "# HELP wafer_model_f1 Latest weighted F1 score.",
        "# TYPE wafer_model_f1 gauge",
        f"wafer_model_f1 {latest_metric.get('f1_score', 0.0)}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
