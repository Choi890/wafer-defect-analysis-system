from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    DOCS_DIR,
    EXCEL_REPORT_PATH,
    PDF_REPORT_PATH,
    TRAINING_HISTORY_PATH,
    ensure_directories,
)
from src.database import repository


def _safe_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def generate_reports(
    excel_path: Path = EXCEL_REPORT_PATH,
    pdf_path: Path = PDF_REPORT_PATH,
) -> dict[str, str]:
    ensure_directories()
    repository.bootstrap_database()
    wafer_df = repository.get_wafer_info_frame()
    result_df = repository.get_latest_results_frame()
    lot_stats = _safe_frame(repository.get_lot_statistics())
    defect_stats = _safe_frame(repository.get_defect_statistics())
    metrics = repository.get_latest_metrics() or {}
    class_report = _read_csv_if_exists(CLASSIFICATION_REPORT_PATH)
    training_history = _read_csv_if_exists(TRAINING_HISTORY_PATH)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        wafer_df.to_excel(writer, sheet_name="wafer_info", index=False)
        result_df.to_excel(writer, sheet_name="prediction_result", index=False)
        lot_stats.to_excel(writer, sheet_name="lot_statistics", index=False)
        defect_stats.to_excel(writer, sheet_name="defect_statistics", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="model_metric", index=False)
        class_report.to_excel(writer, sheet_name="class_report", index=False)
        training_history.to_excel(writer, sheet_name="training_history", index=False)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = wafer_df["failure_type"].value_counts().sort_values(ascending=False)
        counts.plot(kind="bar", ax=ax, color="#2563eb")
        ax.set_title("Wafer Defect Type Distribution")
        ax.set_xlabel("Defect Type")
        ax.set_ylabel("Count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        if not lot_stats.empty:
            ax.plot(lot_stats["lot_id"], lot_stats["defect_rate"] * 100, marker="o", color="#059669")
            ax.set_ylim(0, 105)
        ax.set_title("Lot Defect Rate")
        ax.set_xlabel("Lot ID")
        ax.set_ylabel("Defect Rate (%)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        metric_lines = [
            f"Model: {metrics.get('model_name', 'Not trained')}",
            f"Accuracy: {float(metrics.get('accuracy', 0.0)):.3f}",
            f"Precision: {float(metrics.get('precision_score', 0.0)):.3f}",
            f"Recall: {float(metrics.get('recall_score', 0.0)):.3f}",
            f"F1-score: {float(metrics.get('f1_score', 0.0)):.3f}",
        ]
        ax.text(0.02, 0.8, "\n".join(metric_lines), fontsize=14, va="top")
        ax.set_title("Model Performance Summary")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not training_history.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(training_history["epoch"], training_history["train_loss"], marker="o", label="Train loss")
            ax.plot(training_history["epoch"], training_history["validation_loss"], marker="o", label="Validation loss")
            ax2 = ax.twinx()
            ax2.plot(training_history["epoch"], training_history["validation_f1"], color="#dc2626", marker="s", label="Validation F1")
            ax.set_title("Training History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax2.set_ylabel("Validation F1")
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if not class_report.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            class_report.sort_values("f1_score").plot(kind="barh", x="label", y="f1_score", ax=ax, color="#7c3aed")
            ax.set_xlim(0, 1)
            ax.set_title("Per-Class F1 Score")
            ax.set_xlabel("F1 Score")
            ax.set_ylabel("Defect Class")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return {"excel_report": str(excel_path), "pdf_report": str(pdf_path)}


def generate_erd_png(path: Path = DOCS_DIR / "erd.png") -> str:
    ensure_directories()
    tables = {
        "wafer_info": ["wafer_id PK", "lot_id", "failure_type", "die_size", "inspection_date"],
        "prediction_result": ["id PK", "wafer_id FK", "predicted_label", "confidence", "created_at"],
        "model_metric": ["id PK", "model_name", "accuracy", "precision", "recall", "f1_score"],
        "defect_statistics": ["id PK", "lot_id", "defect_type", "defect_count", "defect_rate"],
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    positions = {
        "wafer_info": (0.08, 0.58),
        "prediction_result": (0.58, 0.58),
        "model_metric": (0.08, 0.10),
        "defect_statistics": (0.58, 0.10),
    }
    for table, fields in tables.items():
        x, y = positions[table]
        ax.add_patch(plt.Rectangle((x, y), 0.34, 0.30, fill=False, linewidth=1.5, edgecolor="#1f2937"))
        ax.text(x + 0.02, y + 0.26, table, fontsize=13, weight="bold")
        ax.text(x + 0.02, y + 0.22, "\n".join(fields), fontsize=10, va="top")
    ax.annotate("", xy=(0.58, 0.74), xytext=(0.42, 0.74), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.44, 0.77, "wafer_id", fontsize=10)
    ax.annotate("", xy=(0.58, 0.25), xytext=(0.42, 0.25), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.44, 0.28, "lot_id", fontsize=10)
    ax.set_title("Wafer Defect Analysis System ERD", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    print(generate_reports())
    print(generate_erd_png())
