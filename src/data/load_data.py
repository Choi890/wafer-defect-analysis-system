from __future__ import annotations

import ast
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFECT_CLASSES,
    ENABLE_SYNTHETIC_DATA,
    IMAGE_SIZE,
    RANDOM_SEED,
    RAW_DATA_PATH,
    RAW_DATA_PATH_FROM_ENV,
    ROOT_DIR,
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


def _parse_numeric_rows(text: str) -> np.ndarray | None:
    rows: list[list[int]] = []
    for raw_row in text.replace("|", ";").split(";"):
        tokens = raw_row.replace(",", " ").split()
        if tokens:
            rows.append([int(float(token)) for token in tokens])
    if not rows:
        return None
    return np.asarray(rows, dtype=np.uint8)


def _parse_wafer_map(value: object, base_dir: Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        wafer_map = value
    elif isinstance(value, (list, tuple)):
        wafer_map = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        if candidate.exists() and candidate.suffix.lower() == ".npy":
            wafer_map = np.load(candidate)
        else:
            parsed: object | None = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
            if parsed is None:
                parsed_rows = _parse_numeric_rows(text)
                if parsed_rows is None:
                    raise ValueError("wafer_map must be a 2D array, JSON string, delimited rows, or .npy path")
                wafer_map = parsed_rows
            else:
                wafer_map = np.asarray(parsed)
    else:
        raise ValueError("wafer_map must be a 2D array, JSON string, delimited rows, or .npy path")

    wafer_map = np.asarray(wafer_map, dtype=np.uint8)
    if wafer_map.ndim != 2:
        raise ValueError(f"wafer_map must be 2D, got shape={wafer_map.shape}")
    return wafer_map


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (np.ndarray, list, tuple)):
        return False
    return bool(pd.isna(value))


def _string_or_default(value: object, default: str) -> str:
    if _is_missing_scalar(value):
        return default
    return str(value)


def normalize_raw_dataset(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    required_columns = {"wafer_id", "lot_id", "failure_type"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Raw wafer dataset is missing required columns: {', '.join(missing)}")
    if "wafer_map" not in df.columns and "wafer_map_path" not in df.columns:
        raise ValueError("Raw wafer dataset must include either wafer_map or wafer_map_path")

    records: list[dict[str, object]] = []
    unsupported_labels: set[str] = set()
    today = date.today().isoformat()

    for _, row in df.iterrows():
        map_source = row["wafer_map"] if "wafer_map" in df.columns else None
        if _is_missing_scalar(map_source) and "wafer_map_path" in df.columns:
            map_source = row["wafer_map_path"]
        wafer_map = _parse_wafer_map(map_source, base_dir=base_dir)
        failure_type = _string_or_default(row["failure_type"], "Unknown")
        if failure_type not in DEFECT_CLASSES:
            unsupported_labels.add(failure_type)

        die_size = row["die_size"] if "die_size" in df.columns else None
        if die_size is None or pd.isna(die_size):
            die_size = int((wafer_map > 0).sum())

        inspection_date = row["inspection_date"] if "inspection_date" in df.columns else None
        records.append(
            {
                "wafer_id": _string_or_default(row["wafer_id"], ""),
                "lot_id": _string_or_default(row["lot_id"], ""),
                "wafer_map": wafer_map,
                "failure_type": failure_type,
                "die_size": int(die_size),
                "inspection_date": _string_or_default(inspection_date, today),
            }
        )

    if unsupported_labels:
        labels = ", ".join(sorted(unsupported_labels))
        supported = ", ".join(DEFECT_CLASSES)
        raise ValueError(f"Unsupported failure_type values: {labels}. Supported labels: {supported}")
    return pd.DataFrame(records)


def read_raw_dataset(raw_path: Path) -> pd.DataFrame:
    suffix = raw_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(raw_path)
    elif suffix == ".csv":
        df = pd.read_csv(raw_path)
    elif suffix == ".jsonl":
        df = pd.read_json(raw_path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(raw_path)
    else:
        raise ValueError(f"Unsupported raw dataset format: {raw_path.suffix}")
    return normalize_raw_dataset(df, base_dir=raw_path.parent)


def load_or_create_raw_dataset(
    raw_path: Path = RAW_DATA_PATH,
    samples_per_class: int = 40,
    force: bool = False,
) -> pd.DataFrame:
    ensure_directories()
    raw_path = raw_path if raw_path.is_absolute() else ROOT_DIR / raw_path
    if raw_path.exists() and not force:
        return read_raw_dataset(raw_path)

    if force and raw_path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("Synthetic dataset generation can only write .pkl or .pickle files")
    if RAW_DATA_PATH_FROM_ENV and not raw_path.exists():
        raise FileNotFoundError(f"WAFER_RAW_DATA_PATH does not exist: {raw_path}")
    if not ENABLE_SYNTHETIC_DATA:
        raise FileNotFoundError(f"No raw wafer dataset found at {raw_path} and synthetic data is disabled")

    df = generate_synthetic_dataset(samples_per_class=samples_per_class)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(raw_path)
    np.save(SAMPLE_WAFER_PATH, df.iloc[0]["wafer_map"])
    return df


if __name__ == "__main__":
    dataset = load_or_create_raw_dataset(force=True)
    print(f"Generated {len(dataset)} wafer records at {RAW_DATA_PATH}")
