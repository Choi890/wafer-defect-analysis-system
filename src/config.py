from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_MAP_DIR = PROCESSED_DATA_DIR / "maps"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "saved_models"
DOCS_DIR = ROOT_DIR / "docs"


def _path_from_env(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT_DIR / path


RAW_DATA_PATH_FROM_ENV = "WAFER_RAW_DATA_PATH" in os.environ
RAW_DATA_PATH = _path_from_env("WAFER_RAW_DATA_PATH", RAW_DATA_DIR / "wafer_map_dataset.pkl")
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
ACTIVE_TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train_active.csv"
VALIDATION_CSV_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"
LABEL_MAP_PATH = PROCESSED_DATA_DIR / "label_map.json"
SAMPLE_WAFER_PATH = SAMPLE_DATA_DIR / "sample_wafer.npy"
MODEL_PATH = _path_from_env("WAFER_MODEL_PATH", MODEL_DIR / "wafer_cnn_model.pt")
MODEL_REGISTRY_PATH = _path_from_env("WAFER_MODEL_REGISTRY_PATH", MODEL_DIR / "model_registry.json")
DATABASE_PATH = _path_from_env("WAFER_DATABASE_PATH", DATA_DIR / "wafer_quality.db")
METRICS_JSON_PATH = REPORT_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "classification_report.csv"
TRAINING_HISTORY_PATH = REPORT_DIR / "training_history.csv"
EXCEL_REPORT_PATH = REPORT_DIR / "defect_analysis_report.xlsx"
PDF_REPORT_PATH = REPORT_DIR / "model_performance_report.pdf"

APP_NAME = "Wafer Defect Analysis System"
APP_VERSION = "0.2.0"
APP_ENV = os.getenv("APP_ENV", "local")
IMAGE_SIZE = 32
RANDOM_SEED = 42
MODEL_NAME = "CNN_v1"
WARNING_DEFECT_RATE = float(os.getenv("WARNING_DEFECT_RATE", "0.70"))
CRITICAL_DEFECT_RATE = float(os.getenv("CRITICAL_DEFECT_RATE", "0.85"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.70"))
ENABLE_SYNTHETIC_DATA = os.getenv("WAFER_ENABLE_SYNTHETIC_DATA", "true").lower() not in {"0", "false", "no"}
API_KEY = os.getenv("WAFER_API_KEY", "")

DEFECT_CLASSES = [
    "Normal",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_MAP_DIR,
        SAMPLE_DATA_DIR,
        REPORT_DIR,
        MODEL_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
