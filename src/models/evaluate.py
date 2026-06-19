from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CLASSIFICATION_REPORT_PATH, CONFUSION_MATRIX_PATH, LABEL_MAP_PATH, MODEL_PATH, TEST_CSV_PATH
from src.data.dataset import WaferMapDataset
from src.models.cnn_model import WaferCNN
from src.utils.metrics import classification_metrics


def evaluate_model(
    model_path: Path = MODEL_PATH,
    test_csv_path: Path = TEST_CSV_PATH,
    batch_size: int = 64,
) -> dict[str, object]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    labels = [payload["id_to_label"][str(index)] for index in range(len(payload["id_to_label"]))]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = WaferCNN(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(WaferMapDataset(test_csv_path), batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(predictions.numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, labels=list(range(len(labels))), label_names=labels)
    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(CONFUSION_MATRIX_PATH)
    pd.DataFrame(metrics.get("per_class", [])).to_csv(CLASSIFICATION_REPORT_PATH, index=False)
    return metrics


if __name__ == "__main__":
    print(evaluate_model())
