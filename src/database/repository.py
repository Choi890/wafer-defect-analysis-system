from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH
from src.data.load_data import load_or_create_raw_dataset
from src.data.preprocess import preprocess_dataset
from src.database.db import get_connection, init_db


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def upsert_wafer_info(df: pd.DataFrame) -> None:
    init_db()
    rows = [
        (
            str(row["wafer_id"]),
            str(row["lot_id"]),
            str(row["failure_type"]),
            int(row["die_size"]),
            str(row["inspection_date"]),
        )
        for _, row in df.iterrows()
    ]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO wafer_info
                (wafer_id, lot_id, failure_type, die_size, inspection_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def clear_predictions() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM prediction_result")


def insert_prediction(
    wafer_id: str,
    predicted_label: str,
    confidence: float,
    model_name: str,
    created_at: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_result
                (wafer_id, predicted_label, confidence, model_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (wafer_id, predicted_label, float(confidence), model_name, created_at or utc_now()),
        )


def insert_model_metric(model_name: str, metrics: dict[str, Any], trained_at: str | None = None) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO model_metric
                (
                    model_name,
                    accuracy,
                    precision_score,
                    recall_score,
                    f1_score,
                    macro_f1_score,
                    best_validation_f1,
                    best_epoch,
                    device,
                    trained_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                float(metrics["accuracy"]),
                float(metrics["precision_score"]),
                float(metrics["recall_score"]),
                float(metrics["f1_score"]),
                float(metrics.get("macro_f1_score", 0.0)),
                float(metrics.get("best_validation_f1", 0.0)),
                int(metrics.get("best_epoch", 0)),
                str(metrics.get("device", "cpu")),
                trained_at or utc_now(),
            ),
        )


def create_batch_job(job_id: str, requested_count: int) -> None:
    init_db()
    now = utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO batch_prediction_job
                (job_id, status, requested_count, completed_count, error_message, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, "queued", int(requested_count), 0, None, now, now),
        )


def update_batch_job(
    job_id: str,
    status: str | None = None,
    completed_count: int | None = None,
    error_message: str | None = None,
) -> None:
    assignments: list[str] = []
    params: list[Any] = []
    if status is not None:
        assignments.append("status = ?")
        params.append(status)
    if completed_count is not None:
        assignments.append("completed_count = ?")
        params.append(int(completed_count))
    if error_message is not None:
        assignments.append("error_message = ?")
        params.append(error_message)
    assignments.append("updated_at = ?")
    params.append(utc_now())
    params.append(job_id)

    with get_connection() as conn:
        conn.execute(
            f"""
            UPDATE batch_prediction_job
            SET {", ".join(assignments)}
            WHERE job_id = ?
            """,
            tuple(params),
        )


def get_batch_job(job_id: str) -> dict[str, Any] | None:
    init_db()
    rows = _rows(
        """
        SELECT job_id, status, requested_count, completed_count, error_message, created_at, updated_at
        FROM batch_prediction_job
        WHERE job_id = ?
        """,
        (job_id,),
    )
    return rows[0] if rows else None


def list_batch_jobs(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 200))
    return _rows(
        f"""
        SELECT job_id, status, requested_count, completed_count, error_message, created_at, updated_at
        FROM batch_prediction_job
        ORDER BY created_at DESC
        LIMIT {limit}
        """
    )


def refresh_defect_statistics(df: pd.DataFrame) -> None:
    calculated_at = utc_now()
    lot_totals = df.groupby("lot_id").size().to_dict()
    defect_counts = (
        df[df["failure_type"] != "Normal"]
        .groupby(["lot_id", "failure_type"])
        .size()
        .reset_index(name="defect_count")
    )

    rows = []
    for _, row in defect_counts.iterrows():
        total = lot_totals.get(row["lot_id"], 1)
        rows.append(
            (
                str(row["lot_id"]),
                str(row["failure_type"]),
                int(row["defect_count"]),
                float(row["defect_count"] / total),
                calculated_at,
            )
        )

    with get_connection() as conn:
        conn.execute("DELETE FROM defect_statistics")
        conn.executemany(
            """
            INSERT INTO defect_statistics
                (lot_id, defect_type, defect_count, defect_rate, calculated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def list_results(limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    return _rows(
        f"""
        SELECT
            p.id,
            p.wafer_id,
            w.lot_id,
            w.failure_type AS actual_label,
            p.predicted_label,
            p.confidence,
            p.model_name,
            p.created_at
        FROM prediction_result p
        JOIN (
            SELECT wafer_id, MAX(id) AS latest_id
            FROM prediction_result
            GROUP BY wafer_id
        ) latest ON latest.latest_id = p.id
        LEFT JOIN wafer_info w ON w.wafer_id = p.wafer_id
        ORDER BY p.created_at DESC
        LIMIT {limit}
        """
    )


def get_result_by_wafer_id(wafer_id: str) -> dict[str, Any] | None:
    rows = _rows(
        """
        SELECT
            p.id,
            p.wafer_id,
            w.lot_id,
            w.failure_type AS actual_label,
            p.predicted_label,
            p.confidence,
            p.model_name,
            p.created_at
        FROM prediction_result p
        LEFT JOIN wafer_info w ON w.wafer_id = p.wafer_id
        WHERE p.wafer_id = ?
        ORDER BY p.id DESC
        LIMIT 1
        """,
        (wafer_id,),
    )
    return rows[0] if rows else None


def get_latest_metrics() -> dict[str, Any] | None:
    init_db()
    rows = _rows(
        """
        SELECT
            model_name,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            macro_f1_score,
            best_validation_f1,
            best_epoch,
            device,
            trained_at
        FROM model_metric
        ORDER BY id DESC
        LIMIT 1
        """
    )
    return rows[0] if rows else None


def get_metrics_history(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 100))
    return _rows(
        f"""
        SELECT
            model_name,
            accuracy,
            precision_score,
            recall_score,
            f1_score,
            macro_f1_score,
            best_validation_f1,
            best_epoch,
            device,
            trained_at
        FROM model_metric
        ORDER BY id DESC
        LIMIT {limit}
        """
    )


def get_lot_statistics() -> list[dict[str, Any]]:
    return _rows(
        """
        SELECT
            lot_id,
            COUNT(*) AS total_count,
            SUM(CASE WHEN failure_type != 'Normal' THEN 1 ELSE 0 END) AS defect_count,
            CAST(SUM(CASE WHEN failure_type != 'Normal' THEN 1 ELSE 0 END) AS REAL)
                / COUNT(*) AS defect_rate
        FROM wafer_info
        GROUP BY lot_id
        ORDER BY lot_id
        """
    )


def get_defect_statistics() -> list[dict[str, Any]]:
    return _rows(
        """
        SELECT lot_id, defect_type, defect_count, defect_rate, calculated_at
        FROM defect_statistics
        ORDER BY lot_id, defect_type
        """
    )


def get_wafer_info_frame() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT wafer_id, lot_id, failure_type, die_size, inspection_date
            FROM wafer_info
            ORDER BY wafer_id
            """,
            conn,
        )


def get_latest_results_frame() -> pd.DataFrame:
    return pd.DataFrame(list_results(limit=1000))


def get_table_count(table_name: str) -> int:
    allowed_tables = {"wafer_info", "prediction_result", "model_metric", "defect_statistics"}
    if table_name not in allowed_tables:
        raise ValueError(f"Unsupported table name: {table_name}")
    init_db()
    with get_connection() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def get_wafer_ids() -> list[str]:
    rows = _rows("SELECT wafer_id FROM wafer_info ORDER BY wafer_id")
    return [row["wafer_id"] for row in rows]


def load_wafer_map(wafer_id: str) -> np.ndarray | None:
    df = load_or_create_raw_dataset()
    match = df[df["wafer_id"] == wafer_id]
    if match.empty:
        return None
    return np.asarray(match.iloc[0]["wafer_map"])


def bootstrap_database(force: bool = False) -> None:
    init_db()
    df = load_or_create_raw_dataset(force=force)
    if force or not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists()):
        preprocess_dataset(df)
    upsert_wafer_info(df)
    refresh_defect_statistics(df)

    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_result").fetchone()[0]
    if count > 0 and not force:
        return

    from src.models.predict import Predictor

    clear_predictions()
    predictor = Predictor()
    for _, row in df.iterrows():
        predicted = predictor.predict(np.asarray(row["wafer_map"]))
        insert_prediction(
            wafer_id=str(row["wafer_id"]),
            predicted_label=predicted["predicted_label"],
            confidence=float(predicted["confidence"]),
            model_name=predicted["model_name"],
        )


def database_exists(path: Path = DATABASE_PATH) -> bool:
    return path.exists()
