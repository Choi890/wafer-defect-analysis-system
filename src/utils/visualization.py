from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


WAFER_CMAP = ListedColormap(["#111827", "#d1d5db", "#ef4444"])
QUALITY_PALETTE = {
    "Normal": "#16a34a",
    "Center": "#2563eb",
    "Donut": "#7c3aed",
    "Edge-Loc": "#f97316",
    "Edge-Ring": "#dc2626",
    "Loc": "#0891b2",
    "Near-full": "#be123c",
    "Random": "#64748b",
    "Scratch": "#ca8a04",
}
RISK_COLORS = {"Normal": "#16a34a", "Warning": "#f59e0b", "Critical": "#dc2626"}


def apply_plot_theme(fig, height: int = 360):
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=16, r=16, t=52, b=24),
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#111827"),
        title=dict(font=dict(size=16, color="#111827")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#eef2f7", linecolor="#e5e7eb")
    return fig


def wafer_map_figure(wafer_map: np.ndarray, title: str | None = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wafer_map, cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def defect_bar_chart(df: pd.DataFrame):
    counts = df["failure_type"].value_counts().rename_axis("failure_type").reset_index(name="count")
    fig = px.bar(
        counts,
        x="failure_type",
        y="count",
        color="failure_type",
        color_discrete_map=QUALITY_PALETTE,
        title="불량 유형별 발생 수",
    )
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>count=%{y}<extra></extra>")
    return apply_plot_theme(fig)


def defect_pie_chart(df: pd.DataFrame):
    counts = df["failure_type"].value_counts().rename_axis("failure_type").reset_index(name="count")
    fig = px.pie(
        counts,
        names="failure_type",
        values="count",
        title="불량 유형 비중",
        color="failure_type",
        color_discrete_map=QUALITY_PALETTE,
        hole=0.58,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>%{value}건<extra></extra>")
    return apply_plot_theme(fig)


def lot_defect_rate_chart(df: pd.DataFrame):
    lot_stats = (
        df.assign(is_defect=df["failure_type"].ne("Normal").astype(int))
        .groupby("lot_id", as_index=False)["is_defect"]
        .mean()
        .rename(columns={"is_defect": "defect_rate"})
    )
    lot_stats["defect_rate"] = lot_stats["defect_rate"] * 100
    fig = px.line(lot_stats, x="lot_id", y="defect_rate", markers=True, title="Lot별 불량률 추이")
    fig.update_traces(line=dict(color="#2563eb", width=3), marker=dict(size=7), hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>")
    fig.update_yaxes(ticksuffix="%", range=[0, 105])
    return apply_plot_theme(fig)


def confusion_matrix_figure(matrix: np.ndarray, labels: list[str]):
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale=["#eff6ff", "#2563eb", "#111827"],
        text_auto=True,
        title="Confusion Matrix",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return apply_plot_theme(fig, height=520)


def wafer_heatmap_figure(wafer_map: np.ndarray, title: str | None = None):
    fig = go.Figure(
        data=go.Heatmap(
            z=wafer_map,
            colorscale=[[0, "#111827"], [0.5, "#d1d5db"], [1, "#ef4444"]],
            zmin=0,
            zmax=2,
            showscale=False,
            hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or "Wafer Map",
        height=470,
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def confidence_distribution_chart(result_df: pd.DataFrame):
    if result_df.empty or "confidence" not in result_df.columns:
        return apply_plot_theme(go.Figure(), height=320)
    fig = px.histogram(
        result_df,
        x="confidence",
        nbins=20,
        color_discrete_sequence=["#0891b2"],
        title="예측 신뢰도 분포",
    )
    fig.update_traces(marker_line_width=0, hovertemplate="confidence=%{x:.3f}<br>count=%{y}<extra></extra>")
    fig.update_xaxes(range=[0, 1])
    return apply_plot_theme(fig, height=320)


def lot_risk_chart(lot_quality_df: pd.DataFrame):
    if lot_quality_df.empty:
        return apply_plot_theme(go.Figure(), height=320)
    frame = lot_quality_df.sort_values("defect_rate", ascending=False).head(15).copy()
    frame["defect_rate_pct"] = frame["defect_rate"] * 100
    fig = px.bar(
        frame,
        x="defect_rate_pct",
        y="lot_id",
        orientation="h",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        title="Lot 위험도 Top 15",
        hover_data={"dominant_defect": True, "avg_confidence": ":.3f", "defect_rate_pct": ":.1f"},
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    fig.update_xaxes(ticksuffix="%", range=[0, 105])
    fig.update_traces(marker_line_width=0)
    return apply_plot_theme(fig, height=410)


def defect_pareto_chart(df: pd.DataFrame):
    counts = df[df["failure_type"].ne("Normal")]["failure_type"].value_counts().reset_index()
    counts.columns = ["failure_type", "count"]
    if counts.empty:
        return apply_plot_theme(go.Figure(), height=340)
    counts["cumulative_rate"] = counts["count"].cumsum() / counts["count"].sum() * 100
    fig = go.Figure()
    fig.add_bar(
        x=counts["failure_type"],
        y=counts["count"],
        marker_color=[QUALITY_PALETTE.get(label, "#64748b") for label in counts["failure_type"]],
        name="Count",
    )
    fig.add_scatter(
        x=counts["failure_type"],
        y=counts["cumulative_rate"],
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#111827", width=3),
        marker=dict(size=7),
        name="Cumulative",
    )
    fig.update_layout(
        title="불량 Pareto",
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105], ticksuffix="%"),
    )
    return apply_plot_theme(fig, height=360)
