from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CONFUSION_MATRIX_PATH,
    LABEL_MAP_PATH,
    METRICS_JSON_PATH,
    MODEL_NAME,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    ensure_directories,
)
from src.data.dataset import WaferMapDataset
from src.data.load_data import load_or_create_raw_dataset
from src.data.preprocess import preprocess_dataset
from src.database import repository
from src.models.cnn_model import WaferCNN
from src.utils.metrics import classification_metrics


def _load_labels() -> list[str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return [payload["id_to_label"][str(index)] for index in range(len(payload["id_to_label"]))]


def _evaluate(model: WaferCNN, loader: DataLoader, labels: list[str]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predictions.numpy().tolist())
    metrics = classification_metrics(y_true, y_pred, labels=list(range(len(labels))))
    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(CONFUSION_MATRIX_PATH)
    return metrics


def _predict_records(model: WaferCNN, frame: pd.DataFrame, labels: list[str]) -> list[tuple[str, str, float]]:
    model.eval()
    predictions: list[tuple[str, str, float]] = []
    with torch.no_grad():
        for _, row in frame.iterrows():
            wafer_map = np.load(Path(__file__).resolve().parents[2] / str(row["map_path"]))
            tensor = torch.from_numpy(wafer_map.astype("float32") / 2.0).unsqueeze(0).unsqueeze(0)
            probabilities = torch.softmax(model(tensor).squeeze(0), dim=0)
            class_index = int(probabilities.argmax().item())
            predictions.append((str(row["wafer_id"]), labels[class_index], float(probabilities[class_index].item())))
    return predictions


def train_model(
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = RANDOM_SEED,
    persist_db: bool = True,
) -> dict[str, object]:
    ensure_directories()
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_df = load_or_create_raw_dataset()
    if not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists() and LABEL_MAP_PATH.exists()):
        preprocess_dataset(raw_df)

    labels = _load_labels()
    train_loader = DataLoader(WaferMapDataset(TRAIN_CSV_PATH), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(WaferMapDataset(TEST_CSV_PATH), batch_size=batch_size, shuffle=False)

    model = WaferCNN(num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
        average_loss = total_loss / len(train_loader.dataset)
        print(f"epoch={epoch} loss={average_loss:.4f}")

    metrics = _evaluate(model, test_loader, labels)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if persist_db:
        repository.init_db()
        repository.upsert_wafer_info(raw_df)
        repository.refresh_defect_statistics(raw_df)
        repository.clear_predictions()
        all_processed = pd.concat([pd.read_csv(TRAIN_CSV_PATH), pd.read_csv(TEST_CSV_PATH)], ignore_index=True)
        for wafer_id, predicted_label, confidence in _predict_records(model, all_processed, labels):
            repository.insert_prediction(
                wafer_id=wafer_id,
                predicted_label=predicted_label,
                confidence=confidence,
                model_name=MODEL_NAME,
            )
        repository.insert_model_metric(MODEL_NAME, metrics)

    return metrics


if __name__ == "__main__":
    final_metrics = train_model()
    print(json.dumps(final_metrics, indent=2))
