from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFECT_CLASSES,
    IMAGE_SIZE,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SAMPLE_WAFER_PATH,
    ensure_directories,
)


def _wafer_mask(size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis)
    return (xx**2 + yy**2) <= 0.92**2


def _add_cluster(defects: np.ndarray, center: tuple[int, int], radius: int) -> None:
    size = defects.shape[0]
    yy, xx = np.ogrid[:size, :size]
    cx, cy = center
    defects[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = True


def _add_line(defects: np.ndarray, rng: np.random.Generator, width: int = 1) -> None:
    size = defects.shape[0]
    slope = rng.uniform(-0.9, 0.9)
    intercept = rng.integers(size // 6, size - size // 6)
    for x in range(size):
        y = int(slope * (x - size / 2) + intercept)
        for offset in range(-width, width + 1):
            yy = y + offset
            if 0 <= yy < size:
                defects[yy, x] = True


def generate_wafer_map(label: str, size: int = IMAGE_SIZE, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = _wafer_mask(size)
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis)
    radius = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)

    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[mask] = 1
    defects = np.zeros((size, size), dtype=bool)

    if label == "Normal":
        random_noise = rng.random((size, size)) < 0.003
        defects |= random_noise & mask
    elif label == "Center":
        defects |= radius < rng.uniform(0.18, 0.28)
    elif label == "Donut":
        inner = rng.uniform(0.22, 0.30)
        outer = inner + rng.uniform(0.12, 0.18)
        defects |= (radius > inner) & (radius < outer)
    elif label == "Edge-Loc":
        center_angle = rng.uniform(-np.pi, np.pi)
        angular_distance = np.abs(np.angle(np.exp(1j * (angle - center_angle))))
        defects |= (radius > 0.62) & (radius < 0.92) & (angular_distance < 0.45)
    elif label == "Edge-Ring":
        defects |= (radius > rng.uniform(0.68, 0.76)) & (radius < 0.92)
    elif label == "Loc":
        for _ in range(rng.integers(1, 4)):
            cx = int(rng.integers(size // 4, size - size // 4))
            cy = int(rng.integers(size // 4, size - size // 4))
            _add_cluster(defects, (cx, cy), int(rng.integers(2, 5)))
    elif label == "Near-full":
        defects |= mask & (rng.random((size, size)) < rng.uniform(0.42, 0.62))
    elif label == "Random":
        defects |= mask & (rng.random((size, size)) < rng.uniform(0.08, 0.16))
    elif label == "Scratch":
        _add_line(defects, rng, width=int(rng.integers(1, 3)))
    else:
        raise ValueError(f"Unsupported wafer defect label: {label}")

    sparse_noise = mask & (rng.random((size, size)) < 0.006)
    wafer[(defects | sparse_noise) & mask] = 2
    return wafer


def generate_synthetic_dataset(
    samples_per_class: int = 40,
    size: int = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = date.today() - timedelta(days=120)
    records: list[dict[str, object]] = []
    wafer_index = 1

    for label in DEFECT_CLASSES:
        for _ in range(samples_per_class):
            lot_number = ((wafer_index - 1) // 25) + 1
            wafer_map = generate_wafer_map(label, size=size, seed=int(rng.integers(0, 1_000_000)))
            records.append(
                {
                    "wafer_id": f"WAFER_{wafer_index:06d}",
                    "lot_id": f"LOT_A{lot_number:03d}",
                    "wafer_map": wafer_map,
                    "failure_type": label,
                    "die_size": int((wafer_map > 0).sum()),
                    "inspection_date": (start_date + timedelta(days=lot_number)).isoformat(),
                }
            )
            wafer_index += 1

    df = pd.DataFrame(records)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_or_create_raw_dataset(
    raw_path: Path = RAW_DATA_PATH,
    samples_per_class: int = 40,
    force: bool = False,
) -> pd.DataFrame:
    ensure_directories()
    if raw_path.exists() and not force:
        return pd.read_pickle(raw_path)

    df = generate_synthetic_dataset(samples_per_class=samples_per_class)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(raw_path)
    np.save(SAMPLE_WAFER_PATH, df.iloc[0]["wafer_map"])
    return df


if __name__ == "__main__":
    dataset = load_or_create_raw_dataset(force=True)
    print(f"Generated {len(dataset)} wafer records at {RAW_DATA_PATH}")
