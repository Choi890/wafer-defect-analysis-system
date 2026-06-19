from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load_data import read_raw_dataset


def test_read_csv_dataset_with_json_wafer_map(tmp_path):
    raw_path = tmp_path / "wafer.csv"
    pd.DataFrame(
        [
            {
                "wafer_id": "WAFER_REAL_001",
                "lot_id": "LOT_REAL_001",
                "failure_type": "Center",
                "wafer_map": json.dumps([[0, 1, 0], [1, 2, 1], [0, 1, 0]]),
                "inspection_date": "2026-06-18",
            }
        ]
    ).to_csv(raw_path, index=False)

    dataset = read_raw_dataset(raw_path)

    assert dataset.loc[0, "wafer_id"] == "WAFER_REAL_001"
    assert dataset.loc[0, "failure_type"] == "Center"
    assert dataset.loc[0, "die_size"] == 5
    np.testing.assert_array_equal(dataset.loc[0, "wafer_map"], np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.uint8))


def test_read_csv_dataset_with_npy_wafer_map_path(tmp_path):
    map_path = tmp_path / "map.npy"
    np.save(map_path, np.array([[1, 1], [0, 2]], dtype=np.uint8))
    raw_path = tmp_path / "wafer.csv"
    pd.DataFrame(
        [
            {
                "wafer_id": "WAFER_REAL_002",
                "lot_id": "LOT_REAL_002",
                "failure_type": "Scratch",
                "wafer_map_path": map_path.name,
            }
        ]
    ).to_csv(raw_path, index=False)

    dataset = read_raw_dataset(raw_path)

    assert dataset.loc[0, "die_size"] == 3
    np.testing.assert_array_equal(dataset.loc[0, "wafer_map"], np.array([[1, 1], [0, 2]], dtype=np.uint8))
