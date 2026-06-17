from __future__ import annotations

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

RAW_DATA_PATH = RAW_DATA_DIR / "wafer_map_dataset.pkl"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"
LABEL_MAP_PATH = PROCESSED_DATA_DIR / "label_map.json"
SAMPLE_WAFER_PATH = SAMPLE_DATA_DIR / "sample_wafer.npy"
MODEL_PATH = MODEL_DIR / "wafer_cnn_model.pt"
DATABASE_PATH = DATA_DIR / "wafer_quality.db"
METRICS_JSON_PATH = REPORT_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.csv"
EXCEL_REPORT_PATH = REPORT_DIR / "defect_analysis_report.xlsx"
PDF_REPORT_PATH = REPORT_DIR / "model_performance_report.pdf"

IMAGE_SIZE = 32
RANDOM_SEED = 42
MODEL_NAME = "CNN_v1"

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
