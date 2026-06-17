from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


WAFER_CMAP = ListedColormap(["#111827", "#d1d5db", "#ef4444"])


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
    return px.bar(
        counts,
        x="failure_type",
        y="count",
        color="failure_type",
        title="Defect Type Count",
    )


def defect_pie_chart(df: pd.DataFrame):
    counts = df["failure_type"].value_counts().rename_axis("failure_type").reset_index(name="count")
    return px.pie(counts, names="failure_type", values="count", title="Defect Type Ratio")


def lot_defect_rate_chart(df: pd.DataFrame):
    lot_stats = (
        df.assign(is_defect=df["failure_type"].ne("Normal").astype(int))
        .groupby("lot_id", as_index=False)["is_defect"]
        .mean()
        .rename(columns={"is_defect": "defect_rate"})
    )
    lot_stats["defect_rate"] = lot_stats["defect_rate"] * 100
    return px.line(lot_stats, x="lot_id", y="defect_rate", markers=True, title="Lot Defect Rate (%)")


def confusion_matrix_figure(matrix: np.ndarray, labels: list[str]):
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return fig
