from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFECT_CLASSES,
    IMAGE_SIZE,
    LABEL_MAP_PATH,
    PROCESSED_MAP_DIR,
    RANDOM_SEED,
    ROOT_DIR,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    ensure_directories,
)
from src.data.load_data import load_or_create_raw_dataset


def resize_nearest(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    if image.shape == (size, size):
        return image.astype(np.uint8)
    y_idx = np.linspace(0, image.shape[0] - 1, size).round().astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, size).round().astype(int)
    return image[np.ix_(y_idx, x_idx)].astype(np.uint8)


def _relative_path(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def build_label_map(labels: list[str] | None = None) -> dict[str, dict[str, int] | dict[str, str]]:
    labels = labels or DEFECT_CLASSES
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    return {"label_to_id": label_to_id, "id_to_label": id_to_label}


def preprocess_dataset(
    df: pd.DataFrame | None = None,
    image_size: int = IMAGE_SIZE,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    if df is None:
        df = load_or_create_raw_dataset()

    label_map = build_label_map()
    label_to_id = label_map["label_to_id"]
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        wafer_map = resize_nearest(np.asarray(row["wafer_map"]), size=image_size)
        map_path = PROCESSED_MAP_DIR / f"{row['wafer_id']}.npy"
        np.save(map_path, wafer_map)
        records.append(
            {
                "wafer_id": row["wafer_id"],
                "lot_id": row["lot_id"],
                "failure_type": row["failure_type"],
                "label": int(label_to_id[row["failure_type"]]),
                "die_size": int(row["die_size"]),
                "inspection_date": row["inspection_date"],
                "map_path": _relative_path(map_path),
            }
        )

    processed = pd.DataFrame(records)
    train_df, test_df = train_test_split(
        processed,
        test_size=test_size,
        random_state=seed,
        stratify=processed["label"],
    )

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)
    LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    train, test = preprocess_dataset()
    print(f"Saved {len(train)} train rows and {len(test)} test rows")
