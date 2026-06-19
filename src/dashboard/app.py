from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    APP_VERSION,
    CLASSIFICATION_REPORT_PATH,
    CRITICAL_DEFECT_RATE,
    CONFUSION_MATRIX_PATH,
    DEFECT_CLASSES,
    EXCEL_REPORT_PATH,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    PDF_REPORT_PATH,
    TRAINING_HISTORY_PATH,
    WARNING_DEFECT_RATE,
)
from src.database import repository  # noqa: E402
from src.utils.quality import (  # noqa: E402
    QualityThresholds,
    build_lot_quality_table,
    build_quality_summary,
    filter_quality_frames,
)
from src.utils.labels import apply_defect_labels, defect_label  # noqa: E402
from src.utils.report_generator import generate_reports  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    class_performance_chart,
    confidence_distribution_chart,
    confusion_matrix_figure,
    defect_bar_chart,
    defect_pareto_chart,
    defect_pie_chart,
    lot_defect_rate_chart,
    lot_risk_chart,
    training_history_chart,
    wafer_heatmap_figure,
)


st.set_page_config(
    page_title="Wafer 품질 관제",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --text: #111827;
                --muted: #64748b;
                --line: #e5e7eb;
                --surface: #ffffff;
                --soft: #f8fafc;
                --blue: #2563eb;
                --green: #16a34a;
                --amber: #f59e0b;
                --red: #dc2626;
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.4rem;
                max-width: 1440px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 14px 16px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            [data-testid="stMetricLabel"] {
                color: var(--muted);
            }
            [data-testid="stMetricValue"] {
                color: var(--text);
                font-size: 1.55rem;
            }
            .ops-header {
                border-bottom: 1px solid var(--line);
                padding-bottom: 18px;
                margin-bottom: 18px;
            }
            .ops-title {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 24px;
                flex-wrap: wrap;
            }
            .ops-title h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.15;
            }
            .ops-subtitle {
                margin-top: 8px;
                color: var(--muted);
                font-size: 0.94rem;
            }
            .status-strip {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }
            .status-pill {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 6px 10px;
                font-size: 0.82rem;
                color: var(--text);
                background: var(--soft);
                white-space: nowrap;
            }
            .status-ok {
                color: #166534;
                background: #f0fdf4;
                border-color: #bbf7d0;
            }
            .status-warn {
                color: #92400e;
                background: #fffbeb;
                border-color: #fde68a;
            }
            .section-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: .06em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .action-row {
                display: flex;
                gap: 8px;
                align-items: center;
                flex-wrap: wrap;
            }
            .small-muted {
                color: var(--muted);
                font-size: 0.86rem;
            }
            .risk-critical {
                color: #991b1b;
                font-weight: 700;
            }
            .risk-warning {
                color: #92400e;
                font-weight: 700;
            }
            .risk-normal {
                color: #166534;
                font-weight: 700;
            }
            div[data-testid="stTabs"] button {
                font-size: 0.94rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def load_dashboard_data():
    repository.bootstrap_database()
    wafer_df = repository.get_wafer_info_frame()
    result_df = repository.get_latest_results_frame()
    defect_stats = pd.DataFrame(repository.get_defect_statistics())
    metrics = repository.get_latest_metrics()
    training_history = pd.read_csv(TRAINING_HISTORY_PATH) if TRAINING_HISTORY_PATH.exists() else pd.DataFrame()
    class_report = pd.read_csv(CLASSIFICATION_REPORT_PATH) if CLASSIFICATION_REPORT_PATH.exists() else pd.DataFrame()
    return wafer_df, result_df, defect_stats, metrics, training_history, class_report


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def percent2(value: float) -> str:
    return f"{value * 100:.2f}%"


def risk_label(value: str) -> str:
    return {"Critical": "심각", "Warning": "주의", "Normal": "정상"}.get(value, value)


def render_header(summary: dict[str, object], metrics: dict | None) -> None:
    model_ready = MODEL_PATH.exists()
    model_label = metrics.get("model_name", "RuleBaseline") if metrics else "RuleBaseline"
    model_status = "CNN 모델 사용" if model_ready else "규칙 기반 예측"
    risk_status = "정상" if int(summary["critical_lots"]) == 0 else "주의 필요"
    risk_class = "status-ok" if risk_status == "정상" else "status-warn"

    st.markdown(
        f"""
        <div class="ops-header">
          <div class="ops-title">
            <div>
              <h1>Wafer 품질 관제</h1>
              <div class="ops-subtitle">Lot 위험도, Wafer 불량 패턴, 모델 신뢰도를 한 화면에서 확인합니다.</div>
            </div>
            <div class="status-strip">
              <span class="status-pill status-ok">API v{html.escape(APP_VERSION)}</span>
              <span class="status-pill status-ok">DB 정상</span>
              <span class="status-pill {risk_class}">Lot 상태 {risk_status}</span>
              <span class="status-pill">{html.escape(model_label)} · {html.escape(model_status)}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(summary: dict[str, object]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Wafer 수", f"{summary['total_wafers']:,}")
    c2.metric("Lot 수", f"{summary['total_lots']:,}")
    c3.metric("전체 불량률", percent(float(summary["defect_rate"])))
    c4.metric("위험 Lot", f"{summary['risk_lots']:,}")
    c5.metric("평균 신뢰도", percent(float(summary["avg_confidence"])))
    c6.metric("모델 F1", f"{float(summary['model_f1']):.3f}")


def render_sidebar(wafer_df: pd.DataFrame) -> tuple[list[str], list[str], QualityThresholds]:
    st.sidebar.header("운영 필터")
    lots = sorted(wafer_df["lot_id"].dropna().unique().tolist())
    defects = sorted(wafer_df["failure_type"].dropna().unique().tolist())
    selected_lots = st.sidebar.multiselect("Lot", lots, default=[], placeholder="전체 Lot")
    selected_defects = st.sidebar.multiselect(
        "불량 유형",
        defects,
        default=[],
        placeholder="전체 불량 유형",
        format_func=defect_label,
    )

    st.sidebar.divider()
    st.sidebar.header("관리 기준")
    warning = st.sidebar.slider("Warning 불량률", 0.0, 1.0, WARNING_DEFECT_RATE, 0.05)
    critical = st.sidebar.slider("Critical 불량률", warning, 1.0, CRITICAL_DEFECT_RATE, 0.05)
    low_confidence = st.sidebar.slider("낮은 신뢰도 기준", 0.0, 1.0, LOW_CONFIDENCE_THRESHOLD, 0.05)

    st.sidebar.divider()
    if st.sidebar.button("데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return selected_lots, selected_defects, QualityThresholds(
        warning_defect_rate=warning,
        critical_defect_rate=critical,
        low_confidence=low_confidence,
    )


def render_control_tower(
    wafer_df: pd.DataFrame,
    result_df: pd.DataFrame,
    lot_quality_df: pd.DataFrame,
    thresholds: QualityThresholds,
) -> None:
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(lot_risk_chart(lot_quality_df), use_container_width=True)
    with right:
        st.plotly_chart(defect_pareto_chart(wafer_df), use_container_width=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="section-label">위험 Lot 우선순위</div>', unsafe_allow_html=True)
        risk_queue = lot_quality_df[lot_quality_df["risk_level"].isin(["Warning", "Critical"])].copy()
        risk_queue = apply_defect_labels(risk_queue, ["dominant_defect"])
        risk_queue["불량률"] = risk_queue["defect_rate"].map(lambda value: percent2(float(value)))
        risk_queue["평균 신뢰도"] = risk_queue["avg_confidence"].map(lambda value: percent2(float(value)))
        risk_queue["risk_level"] = risk_queue["risk_level"].map(risk_label)
        risk_queue = risk_queue.rename(
            columns={
                "lot_id": "Lot",
                "dominant_defect": "주요 불량",
                "risk_level": "위험도",
            }
        )[["Lot", "불량률", "주요 불량", "평균 신뢰도", "위험도"]]
        st.dataframe(
            risk_queue,
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown('<div class="section-label">낮은 신뢰도 예측</div>', unsafe_allow_html=True)
        if result_df.empty:
            st.info("예측 결과가 없습니다.")
        else:
            low_confidence = result_df[result_df["confidence"].lt(thresholds.low_confidence)].sort_values("confidence")
            low_display = low_confidence[["wafer_id", "lot_id", "actual_label", "predicted_label", "confidence"]].head(12).copy()
            low_display = apply_defect_labels(low_display, ["actual_label", "predicted_label"])
            low_display["신뢰도"] = low_display["confidence"].map(lambda value: percent2(float(value)))
            low_display = low_display.rename(
                columns={
                    "wafer_id": "Wafer",
                    "lot_id": "Lot",
                    "actual_label": "실제",
                    "predicted_label": "예측",
                }
            )[["Wafer", "Lot", "실제", "예측", "신뢰도"]]
            st.dataframe(
                low_display,
                use_container_width=True,
                hide_index=True,
            )


def render_lot_analytics(wafer_df: pd.DataFrame, lot_quality_df: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(lot_defect_rate_chart(wafer_df), use_container_width=True)
    with right:
        st.plotly_chart(defect_bar_chart(wafer_df), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(defect_pie_chart(wafer_df), use_container_width=True)
    with right:
        table = lot_quality_df.copy()
        table = apply_defect_labels(table, ["dominant_defect"])
        table["불량률"] = table["defect_rate"].map(lambda value: percent2(float(value)))
        table["평균 신뢰도"] = table["avg_confidence"].map(lambda value: percent2(float(value)))
        table["risk_level"] = table["risk_level"].map(risk_label)
        table = table.rename(
            columns={
                "lot_id": "Lot",
                "total_wafers": "Wafer 수",
                "defect_wafers": "불량 수",
                "dominant_defect": "주요 불량",
                "risk_level": "위험도",
            }
        )[["Lot", "불량률", "평균 신뢰도", "주요 불량", "위험도"]]
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


def render_wafer_review(wafer_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    wafer_ids = wafer_df["wafer_id"].tolist()
    if not wafer_ids:
        st.info("선택된 필터에 해당하는 wafer가 없습니다.")
        return

    selected = st.selectbox("Wafer ID", wafer_ids, index=0)
    wafer_map = repository.load_wafer_map(selected)
    result = repository.get_result_by_wafer_id(selected) or {}
    wafer_info = wafer_df[wafer_df["wafer_id"] == selected].iloc[0].to_dict()

    left, right = st.columns([1, 1.15])
    with left:
        if wafer_map is not None:
            st.plotly_chart(wafer_heatmap_figure(wafer_map, title=selected), use_container_width=True)
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("실제 불량", defect_label(result.get("actual_label", wafer_info.get("failure_type", "Unknown"))))
        c2.metric("예측 불량", defect_label(result.get("predicted_label", "Not predicted")))
        c3.metric("예측 신뢰도", percent(float(result.get("confidence", 0.0))))

        detail = pd.DataFrame(
            [
                {
                    "Wafer": selected,
                    "Lot": wafer_info.get("lot_id"),
                    "검사일": wafer_info.get("inspection_date"),
                    "Die 수": wafer_info.get("die_size"),
                    "모델": result.get("model_name"),
                }
            ]
        )
        st.dataframe(detail, use_container_width=True, hide_index=True)

        if not result_df.empty:
            st.markdown('<div class="section-label">같은 Lot 최근 예측</div>', unsafe_allow_html=True)
            recent = result_df[result_df["lot_id"] == wafer_info.get("lot_id")].head(10)
            recent_display = recent[["wafer_id", "actual_label", "predicted_label", "confidence"]].copy()
            recent_display = apply_defect_labels(recent_display, ["actual_label", "predicted_label"])
            recent_display["신뢰도"] = recent_display["confidence"].map(lambda value: percent2(float(value)))
            recent_display = recent_display.rename(
                columns={
                    "wafer_id": "Wafer",
                    "actual_label": "실제",
                    "predicted_label": "예측",
                }
            )[["Wafer", "실제", "예측", "신뢰도"]]
            st.dataframe(
                recent_display,
                use_container_width=True,
                hide_index=True,
            )


def render_model_ops(
    result_df: pd.DataFrame,
    metrics: dict | None,
    training_history: pd.DataFrame,
    class_report: pd.DataFrame,
) -> None:
    if metrics:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("모델", metrics["model_name"])
        c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c3.metric("Precision", f"{metrics['precision_score']:.3f}")
        c4.metric("Recall", f"{metrics['recall_score']:.3f}")
        c5.metric("F1", f"{metrics['f1_score']:.3f}")
        c6.metric("최고 검증 F1", f"{float(metrics.get('best_validation_f1', 0.0)):.3f}")
    else:
        st.info("학습된 모델 지표가 없습니다. `python -m src.pipeline --epochs 6` 실행 후 지표가 표시됩니다.")

    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(confidence_distribution_chart(result_df), use_container_width=True)
    with right:
        if CONFUSION_MATRIX_PATH.exists():
            matrix_df = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
            labels = matrix_df.index.tolist() or DEFECT_CLASSES
            st.plotly_chart(confusion_matrix_figure(matrix_df.to_numpy(), labels), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(training_history_chart(training_history), use_container_width=True)
    with right:
        st.plotly_chart(class_performance_chart(class_report), use_container_width=True)

    if not class_report.empty:
        st.markdown('<div class="section-label">불량 유형별 모델 성능</div>', unsafe_allow_html=True)
        table = class_report.copy()
        table = apply_defect_labels(table, ["label"])
        for column in ["precision", "recall", "f1_score"]:
            table[column] = table[column].map(lambda value: percent2(float(value)))
        table = table.rename(
            columns={
                "label": "불량 유형",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1",
                "support": "검증 샘플 수",
            }
        )
        st.dataframe(
            table[["불량 유형", "Precision", "Recall", "F1", "검증 샘플 수"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="section-label">예측 감사 로그</div>', unsafe_allow_html=True)
    if result_df.empty:
        st.info("예측 로그가 없습니다.")
    else:
        audit = result_df[["created_at", "wafer_id", "lot_id", "actual_label", "predicted_label", "confidence", "model_name"]].copy()
        audit = apply_defect_labels(audit, ["actual_label", "predicted_label"])
        audit["신뢰도"] = audit["confidence"].map(lambda value: percent2(float(value)))
        audit = audit.rename(
            columns={
                "created_at": "생성 시각",
                "wafer_id": "Wafer",
                "lot_id": "Lot",
                "actual_label": "실제",
                "predicted_label": "예측",
                "model_name": "모델",
            }
        )[["생성 시각", "Wafer", "Lot", "실제", "예측", "신뢰도", "모델"]]
        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True,
        )


def build_report_summary_table(summary: dict[str, object], metrics: dict | None) -> pd.DataFrame:
    rows = [
        {"항목": "전체 Wafer", "값": f"{int(summary['total_wafers']):,}"},
        {"항목": "전체 Lot", "값": f"{int(summary['total_lots']):,}"},
        {"항목": "전체 불량률", "값": percent(float(summary["defect_rate"]))},
        {"항목": "주요 불량", "값": defect_label(summary.get("top_defect"))},
        {"항목": "위험 Lot", "값": f"{int(summary['risk_lots']):,}"},
        {"항목": "Critical Lot", "값": f"{int(summary['critical_lots']):,}"},
        {"항목": "평균 신뢰도", "값": percent(float(summary["avg_confidence"]))},
        {"항목": "낮은 신뢰도 예측", "값": f"{int(summary['low_confidence_count']):,}"},
        {"항목": "모델 F1", "값": f"{float(summary['model_f1']):.3f}"},
    ]
    if metrics:
        rows.extend(
            [
                {"항목": "모델명", "값": str(metrics.get("model_name", "-"))},
                {"항목": "Accuracy", "값": f"{float(metrics.get('accuracy', 0.0)):.3f}"},
                {"항목": "학습 장치", "값": str(metrics.get("device", "-"))},
                {"항목": "학습 시각", "값": str(metrics.get("trained_at", "-"))},
            ]
        )
    return pd.DataFrame(rows)


def render_reports(defect_stats: pd.DataFrame, summary: dict[str, object], metrics: dict | None) -> None:
    left, right = st.columns([1, 1])
    with left:
        if st.button("리포트 재생성", type="primary", use_container_width=True):
            generate_reports()
            st.cache_data.clear()
            st.success("리포트를 재생성했습니다.")

        if EXCEL_REPORT_PATH.exists():
            st.download_button(
                "Excel 리포트 다운로드",
                data=EXCEL_REPORT_PATH.read_bytes(),
                file_name=EXCEL_REPORT_PATH.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        if PDF_REPORT_PATH.exists():
            st.download_button(
                "PDF 리포트 다운로드",
                data=PDF_REPORT_PATH.read_bytes(),
                file_name=PDF_REPORT_PATH.name,
                mime="application/pdf",
                use_container_width=True,
            )

    with right:
        st.markdown('<div class="section-label">운영 요약</div>', unsafe_allow_html=True)
        st.dataframe(build_report_summary_table(summary, metrics), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Lot별 불량 통계</div>', unsafe_allow_html=True)
    stats = apply_defect_labels(defect_stats, ["defect_type"])
    if "defect_rate" in stats.columns:
        stats["defect_rate"] = stats["defect_rate"].map(lambda value: percent2(float(value)))
    stats = stats.rename(
        columns={
            "lot_id": "Lot",
            "defect_type": "불량 유형",
            "defect_count": "불량 수",
            "defect_rate": "불량률",
            "calculated_at": "계산 시각",
        }
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    wafer_df, result_df, defect_stats, metrics, training_history, class_report = load_dashboard_data()
    selected_lots, selected_defects, thresholds = render_sidebar(wafer_df)
    filtered_wafer, filtered_result = filter_quality_frames(wafer_df, result_df, selected_lots, selected_defects)
    summary = build_quality_summary(filtered_wafer, filtered_result, metrics, thresholds)
    lot_quality_df = build_lot_quality_table(filtered_wafer, filtered_result, thresholds)

    render_header(summary, metrics)
    render_kpis(summary)

    tabs = st.tabs(["관제", "Lot 분석", "Wafer 리뷰", "모델 운영", "리포트"])
    with tabs[0]:
        render_control_tower(filtered_wafer, filtered_result, lot_quality_df, thresholds)
    with tabs[1]:
        render_lot_analytics(filtered_wafer, lot_quality_df)
    with tabs[2]:
        render_wafer_review(filtered_wafer, filtered_result)
    with tabs[3]:
        render_model_ops(filtered_result, metrics, training_history, class_report)
    with tabs[4]:
        render_reports(defect_stats, summary, metrics)


if __name__ == "__main__":
    main()
