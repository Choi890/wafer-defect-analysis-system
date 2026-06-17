from __future__ import annotations

import argparse
import json

from src.data.load_data import load_or_create_raw_dataset
from src.data.preprocess import preprocess_dataset
from src.models.train import train_model
from src.utils.report_generator import generate_erd_png, generate_reports


def run_pipeline(samples_per_class: int = 40, epochs: int = 6, force_data: bool = False) -> dict[str, object]:
    df = load_or_create_raw_dataset(samples_per_class=samples_per_class, force=force_data)
    preprocess_dataset(df)
    metrics = train_model(epochs=epochs)
    generate_reports()
    generate_erd_png()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wafer defect analysis demo pipeline.")
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    metrics = run_pipeline(
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        force_data=args.force_data,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
