from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DATABASE_PATH, ensure_directories


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wafer_info (
    wafer_id TEXT PRIMARY KEY,
    lot_id TEXT,
    failure_type TEXT,
    die_size INTEGER,
    inspection_date TEXT
);

CREATE TABLE IF NOT EXISTS prediction_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id TEXT,
    predicted_label TEXT,
    confidence REAL,
    model_name TEXT,
    created_at TEXT,
    FOREIGN KEY (wafer_id) REFERENCES wafer_info(wafer_id)
);

CREATE TABLE IF NOT EXISTS model_metric (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    macro_f1_score REAL,
    best_validation_f1 REAL,
    best_epoch INTEGER,
    device TEXT,
    trained_at TEXT
);

CREATE TABLE IF NOT EXISTS defect_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lot_id TEXT,
    defect_type TEXT,
    defect_count INTEGER,
    defect_rate REAL,
    calculated_at TEXT
);

CREATE TABLE IF NOT EXISTS batch_prediction_job (
    job_id TEXT PRIMARY KEY,
    status TEXT,
    requested_count INTEGER,
    completed_count INTEGER,
    error_message TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS schema_migration (
    version INTEGER PRIMARY KEY,
    name TEXT,
    applied_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_prediction_wafer_id ON prediction_result(wafer_id);
CREATE INDEX IF NOT EXISTS idx_wafer_lot_id ON wafer_info(lot_id);
CREATE INDEX IF NOT EXISTS idx_defect_statistics_lot ON defect_statistics(lot_id);
CREATE INDEX IF NOT EXISTS idx_batch_prediction_job_status ON batch_prediction_job(status);
"""

MODEL_METRIC_COLUMNS = {
    "macro_f1_score": "REAL",
    "best_validation_f1": "REAL",
    "best_epoch": "INTEGER",
    "device": "TEXT",
}


def get_connection(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DATABASE_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(model_metric)").fetchall()}
        for column_name, column_type in MODEL_METRIC_COLUMNS.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE model_metric ADD COLUMN {column_name} {column_type}")
