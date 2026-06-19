from __future__ import annotations

import pandas as pd


DEFECT_LABELS = {
    "Normal": "정상",
    "Center": "중앙",
    "Donut": "도넛",
    "Edge-Loc": "엣지-국부",
    "Edge-Ring": "엣지-링",
    "Loc": "국부",
    "Near-full": "거의전체",
    "Random": "산발",
    "Scratch": "스크래치",
}


def defect_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = str(value)
    return DEFECT_LABELS.get(text, text)


def apply_defect_labels(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    display = frame.copy()
    for column in columns:
        if column in display.columns:
            display[column] = display[column].map(defect_label)
    return display
