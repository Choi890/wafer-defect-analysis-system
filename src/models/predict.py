from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFECT_CLASSES, IMAGE_SIZE, LABEL_MAP_PATH, MODEL_NAME, MODEL_PATH
from src.data.preprocess import resize_nearest

try:
    import torch

    from src.models.cnn_model import WaferCNN

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    WaferCNN = None
    TORCH_AVAILABLE = False


def load_labels() -> list[str]:
    if LABEL_MAP_PATH.exists():
        payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        id_to_label = payload["id_to_label"]
        return [id_to_label[str(index)] for index in range(len(id_to_label))]
    return DEFECT_CLASSES


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def heuristic_predict(wafer_map: np.ndarray) -> dict[str, Any]:
    wafer_map = resize_nearest(np.asarray(wafer_map), IMAGE_SIZE)
    mask = wafer_map > 0
    defects = wafer_map >= 2
    defect_count = int(defects.sum())
    total_die = int(mask.sum()) or 1
    defect_ratio = defect_count / total_die

    if defect_count <= max(2, total_die * 0.01):
        return {"predicted_label": "Normal", "confidence": 0.93, "model_name": "RuleBaseline"}

    coords = np.argwhere(defects)
    center = np.array([(wafer_map.shape[0] - 1) / 2, (wafer_map.shape[1] - 1) / 2])
    distances = np.linalg.norm(coords - center, axis=1) / (wafer_map.shape[0] / 2)
    radial_mean = float(distances.mean())
    radial_std = float(distances.std())
    center_ratio = float((distances < 0.35).mean())
    edge_ratio = float((distances > 0.68).mean())

    if defect_ratio > 0.38:
        return {"predicted_label": "Near-full", "confidence": min(0.96, 0.70 + defect_ratio / 2), "model_name": "RuleBaseline"}

    if len(coords) >= 8:
        centered = coords - coords.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        line_score = singular_values[0] / (singular_values[1] + 1e-6)
        if line_score > 5.0 and defect_ratio < 0.22:
            return {"predicted_label": "Scratch", "confidence": 0.82, "model_name": "RuleBaseline"}

    if edge_ratio > 0.72 and radial_std < 0.18:
        return {"predicted_label": "Edge-Ring", "confidence": 0.86, "model_name": "RuleBaseline"}
    if edge_ratio > 0.55:
        return {"predicted_label": "Edge-Loc", "confidence": 0.78, "model_name": "RuleBaseline"}
    if center_ratio > 0.58:
        return {"predicted_label": "Center", "confidence": 0.84, "model_name": "RuleBaseline"}
    if 0.32 < radial_mean < 0.62 and radial_std < 0.16:
        return {"predicted_label": "Donut", "confidence": 0.76, "model_name": "RuleBaseline"}
    if defect_ratio < 0.07:
        return {"predicted_label": "Loc", "confidence": 0.72, "model_name": "RuleBaseline"}
    return {"predicted_label": "Random", "confidence": 0.70, "model_name": "RuleBaseline"}


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.labels = load_labels()
        self.model_name = "RuleBaseline"
        self.model = None
        if TORCH_AVAILABLE and model_path.exists():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            self.labels = checkpoint.get("labels", self.labels)
            model = WaferCNN(num_classes=len(self.labels))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model
            self.model_name = checkpoint.get("model_name", MODEL_NAME)

    def predict(self, wafer_map: np.ndarray) -> dict[str, Any]:
        if self.model is None or not TORCH_AVAILABLE:
            return heuristic_predict(wafer_map)

        prepared = resize_nearest(np.asarray(wafer_map), IMAGE_SIZE).astype("float32") / 2.0
        tensor = torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(0).numpy()
        probabilities = _softmax(logits)
        class_index = int(probabilities.argmax())
        return {
            "predicted_label": self.labels[class_index],
            "confidence": float(probabilities[class_index]),
            "model_name": self.model_name,
        }
