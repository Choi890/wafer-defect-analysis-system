from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFUSION_MATRIX_PATH, DEFECT_CLASSES, EXCEL_REPORT_PATH, PDF_REPORT_PATH
from src.database import repository
from src.utils.report_generator import generate_reports
from src.utils.visualization import (
    confusion_matrix_figure,
    defect_bar_chart,
    defect_pie_chart,
    lot_defect_rate_chart,
    wafer_map_figure,
)


st.set_page_config(page_title="Wafer Defect Analysis", layout="wide")


@st.cache_data(ttl=60)
def load_dashboard_data():
    repository.bootstrap_database()
    wafer_df = repository.get_wafer_info_frame()
    result_df = repository.get_latest_results_frame()
    lot_stats = pd.DataFrame(repository.get_lot_statistics())
    defect_stats = pd.DataFrame(repository.get_defect_statistics())
    metrics = repository.get_latest_metrics()
    return wafer_df, result_df, lot_stats, defect_stats, metrics


def render_overview(wafer_df: pd.DataFrame, metrics: dict | None) -> None:
    total = len(wafer_df)
    normal = int((wafer_df["failure_type"] == "Normal").sum())
    defective = total - normal
    defect_rate = defective / total if total else 0.0
    top_defect = (
        wafer_df.loc[wafer_df["failure_type"] != "Normal", "failure_type"].value_counts().idxmax()
        if defective
        else "None"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Wafers", f"{total:,}")
    c2.metric("Normal", f"{normal:,}")
    c3.metric("Defective", f"{defective:,}")
    c4.metric("Defect Rate", f"{defect_rate:.1%}")
    c5.metric("Top Defect", top_defect)

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(defect_bar_chart(wafer_df), use_container_width=True)
    with right:
        if metrics:
            st.subheader("Latest Model Metric")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision", f"{metrics['precision_score']:.3f}")
            m3.metric("Recall", f"{metrics['recall_score']:.3f}")
            m4.metric("F1", f"{metrics['f1_score']:.3f}")
        else:
            st.info("Run `python -m src.pipeline` to train the CNN and store model metrics.")


def render_distribution(wafer_df: pd.DataFrame, lot_stats: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(defect_bar_chart(wafer_df), use_container_width=True)
    with right:
        st.plotly_chart(defect_pie_chart(wafer_df), use_container_width=True)
    st.plotly_chart(lot_defect_rate_chart(wafer_df), use_container_width=True)
    st.dataframe(lot_stats, use_container_width=True, hide_index=True)


def render_wafer_viewer(result_df: pd.DataFrame) -> None:
    wafer_ids = repository.get_wafer_ids()
    selected = st.selectbox("Wafer ID", wafer_ids, index=0 if wafer_ids else None)
    if not selected:
        return

    wafer_map = repository.load_wafer_map(selected)
    result = repository.get_result_by_wafer_id(selected)
    actual = result.get("actual_label") if result else "Unknown"
    predicted = result.get("predicted_label") if result else "Not predicted"
    confidence = float(result.get("confidence", 0.0)) if result else 0.0

    left, right = st.columns([1, 1.2])
    with left:
        if wafer_map is not None:
            st.pyplot(wafer_map_figure(wafer_map, title=selected), clear_figure=True)
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual Label", actual)
        c2.metric("Predicted Label", predicted)
        c3.metric("Confidence", f"{confidence:.3f}")
        st.dataframe(
            result_df[result_df["wafer_id"] == selected],
            use_container_width=True,
            hide_index=True,
        )


def render_model_performance(metrics: dict | None) -> None:
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{metrics['precision_score']:.3f}")
        c3.metric("Recall", f"{metrics['recall_score']:.3f}")
        c4.metric("F1-score", f"{metrics['f1_score']:.3f}")
    else:
        st.info("No trained model metric stored yet.")

    if CONFUSION_MATRIX_PATH.exists():
        matrix_df = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
        matrix = matrix_df.to_numpy()
        labels = matrix_df.index.tolist() or DEFECT_CLASSES
        st.plotly_chart(confusion_matrix_figure(matrix, labels), use_container_width=True)


def render_report(defect_stats: pd.DataFrame, metrics: dict | None) -> None:
    if st.button("Generate Reports", type="primary"):
        generate_reports()
        st.cache_data.clear()

    left, right = st.columns(2)
    with left:
        st.subheader("Lot / Defect Statistics")
        st.dataframe(defect_stats, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Model Summary")
        st.json(metrics or {"status": "No trained model metric stored yet."})

    if EXCEL_REPORT_PATH.exists():
        st.download_button(
            "Download Excel Report",
            data=EXCEL_REPORT_PATH.read_bytes(),
            file_name=EXCEL_REPORT_PATH.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if PDF_REPORT_PATH.exists():
        st.download_button(
            "Download PDF Report",
            data=PDF_REPORT_PATH.read_bytes(),
            file_name=PDF_REPORT_PATH.name,
            mime="application/pdf",
        )


def main() -> None:
    st.title("Wafer Defect Analysis System")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Defect Distribution", "Wafer Map Viewer", "Model Performance", "Report"],
    )
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()

    wafer_df, result_df, lot_stats, defect_stats, metrics = load_dashboard_data()
    if page == "Overview":
        render_overview(wafer_df, metrics)
    elif page == "Defect Distribution":
        render_distribution(wafer_df, lot_stats)
    elif page == "Wafer Map Viewer":
        render_wafer_viewer(result_df)
    elif page == "Model Performance":
        render_model_performance(metrics)
    else:
        render_report(defect_stats, metrics)


if __name__ == "__main__":
    main()
