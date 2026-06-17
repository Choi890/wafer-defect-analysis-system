from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityThresholds:
    warning_defect_rate: float = 0.70
    critical_defect_rate: float = 0.85
    low_confidence: float = 0.70


def classify_lot_risk(defect_rate: float, thresholds: QualityThresholds | None = None) -> str:
    thresholds = thresholds or QualityThresholds()
    if defect_rate >= thresholds.critical_defect_rate:
        return "Critical"
    if defect_rate >= thresholds.warning_defect_rate:
        return "Warning"
    return "Normal"


def build_lot_quality_table(
    wafer_df: pd.DataFrame,
    result_df: pd.DataFrame,
    thresholds: QualityThresholds | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or QualityThresholds()
    if wafer_df.empty:
        return pd.DataFrame(
            columns=[
                "lot_id",
                "total_wafers",
                "defect_wafers",
                "defect_rate",
                "avg_confidence",
                "dominant_defect",
                "risk_level",
            ]
        )

    base = wafer_df.copy()
    base["is_defect"] = base["failure_type"].ne("Normal")
    lot_stats = (
        base.groupby("lot_id", as_index=False)
        .agg(total_wafers=("wafer_id", "count"), defect_wafers=("is_defect", "sum"))
        .assign(defect_rate=lambda frame: frame["defect_wafers"] / frame["total_wafers"])
    )

    defect_only = base[base["failure_type"].ne("Normal")]
    if defect_only.empty:
        dominant = pd.DataFrame({"lot_id": lot_stats["lot_id"], "dominant_defect": "None"})
    else:
        dominant = (
            defect_only.groupby(["lot_id", "failure_type"])
            .size()
            .reset_index(name="count")
            .sort_values(["lot_id", "count"], ascending=[True, False])
            .drop_duplicates("lot_id")
            .rename(columns={"failure_type": "dominant_defect"})[["lot_id", "dominant_defect"]]
        )

    if result_df.empty or "confidence" not in result_df.columns:
        confidence = pd.DataFrame({"lot_id": lot_stats["lot_id"], "avg_confidence": 0.0})
    else:
        confidence = result_df.groupby("lot_id", as_index=False)["confidence"].mean()
        confidence = confidence.rename(columns={"confidence": "avg_confidence"})

    table = (
        lot_stats.merge(dominant, on="lot_id", how="left")
        .merge(confidence, on="lot_id", how="left")
        .fillna({"dominant_defect": "None", "avg_confidence": 0.0})
    )
    table["risk_level"] = table["defect_rate"].apply(lambda value: classify_lot_risk(float(value), thresholds))
    risk_order = {"Critical": 0, "Warning": 1, "Normal": 2}
    table["risk_priority"] = table["risk_level"].map(risk_order).fillna(3)
    return (
        table.sort_values(["risk_priority", "defect_rate"], ascending=[True, False])
        .drop(columns=["risk_priority"])
        .reset_index(drop=True)
    )


def build_quality_summary(
    wafer_df: pd.DataFrame,
    result_df: pd.DataFrame,
    metrics: dict | None,
    thresholds: QualityThresholds | None = None,
) -> dict[str, object]:
    thresholds = thresholds or QualityThresholds()
    total_wafers = int(len(wafer_df))
    defect_wafers = int(wafer_df["failure_type"].ne("Normal").sum()) if total_wafers else 0
    defect_rate = defect_wafers / total_wafers if total_wafers else 0.0
    lots = int(wafer_df["lot_id"].nunique()) if total_wafers else 0

    lot_table = build_lot_quality_table(wafer_df, result_df, thresholds)
    risk_lots = int(lot_table["risk_level"].isin(["Warning", "Critical"]).sum()) if not lot_table.empty else 0
    critical_lots = int(lot_table["risk_level"].eq("Critical").sum()) if not lot_table.empty else 0

    avg_confidence = float(result_df["confidence"].mean()) if not result_df.empty and "confidence" in result_df else 0.0
    low_confidence_count = (
        int(result_df["confidence"].lt(thresholds.low_confidence).sum())
        if not result_df.empty and "confidence" in result_df
        else 0
    )
    model_f1 = float(metrics.get("f1_score", 0.0)) if metrics else 0.0

    if defect_wafers:
        top_defect = wafer_df.loc[wafer_df["failure_type"].ne("Normal"), "failure_type"].value_counts().idxmax()
    else:
        top_defect = "None"

    return {
        "total_wafers": total_wafers,
        "total_lots": lots,
        "defect_wafers": defect_wafers,
        "defect_rate": defect_rate,
        "top_defect": top_defect,
        "risk_lots": risk_lots,
        "critical_lots": critical_lots,
        "avg_confidence": avg_confidence,
        "low_confidence_count": low_confidence_count,
        "model_f1": model_f1,
    }


def filter_quality_frames(
    wafer_df: pd.DataFrame,
    result_df: pd.DataFrame,
    selected_lots: list[str] | None = None,
    selected_defects: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_wafer = wafer_df.copy()
    if selected_lots:
        filtered_wafer = filtered_wafer[filtered_wafer["lot_id"].isin(selected_lots)]
    if selected_defects:
        filtered_wafer = filtered_wafer[filtered_wafer["failure_type"].isin(selected_defects)]

    wafer_ids = set(filtered_wafer["wafer_id"])
    filtered_result = result_df[result_df["wafer_id"].isin(wafer_ids)].copy() if not result_df.empty else result_df
    return filtered_wafer.reset_index(drop=True), filtered_result.reset_index(drop=True)
