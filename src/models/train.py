from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    ACTIVE_TRAIN_CSV_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    LABEL_MAP_PATH,
    METRICS_JSON_PATH,
    MODEL_NAME,
    MODEL_PATH,
    RANDOM_SEED,
    ROOT_DIR,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    TRAINING_HISTORY_PATH,
    VALIDATION_CSV_PATH,
    ensure_directories,
)
from src.data.dataset import WaferMapDataset
from src.data.load_data import load_or_create_raw_dataset
from src.data.preprocess import preprocess_dataset
from src.database import repository
from src.models.cnn_model import WaferCNN
from src.models.registry import register_model
from src.utils.metrics import classification_metrics


def _load_labels() -> list[str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return [payload["id_to_label"][str(index)] for index in range(len(payload["id_to_label"]))]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_train_validation_split(seed: int, validation_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = pd.read_csv(TRAIN_CSV_PATH)
    class_counts = train_frame["label"].value_counts()
    if len(train_frame) < 10 or train_frame["label"].nunique() < 2 or int(class_counts.min()) < 2:
        train_frame.to_csv(ACTIVE_TRAIN_CSV_PATH, index=False)
        train_frame.to_csv(VALIDATION_CSV_PATH, index=False)
        return train_frame, train_frame.copy()

    train_split, validation_split = train_test_split(
        train_frame,
        test_size=validation_size,
        random_state=seed,
        stratify=train_frame["label"],
    )
    train_split.to_csv(ACTIVE_TRAIN_CSV_PATH, index=False)
    validation_split.to_csv(VALIDATION_CSV_PATH, index=False)
    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)


def _class_weights(frame: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(range(num_classes), fill_value=0).sort_index().to_numpy()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (num_classes * counts[nonzero])
    weights[~nonzero] = 0.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_loader(csv_path: Path, batch_size: int, shuffle: bool, augment: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        WaferMapDataset(csv_path, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _evaluate(
    model: WaferCNN,
    loader: DataLoader,
    labels: list[str],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, targets)
                total_loss += float(loss.item()) * len(images)
                total_count += len(images)
            predictions = logits.argmax(dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, labels=list(range(len(labels))), label_names=labels)
    metrics["loss"] = total_loss / total_count if total_count else 0.0
    return metrics


def _write_metric_artifacts(metrics: dict[str, Any], labels: list[str]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(CONFUSION_MATRIX_PATH)
    pd.DataFrame(metrics.get("per_class", [])).to_csv(CLASSIFICATION_REPORT_PATH, index=False)
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _predict_records(
    model: WaferCNN,
    frame: pd.DataFrame,
    labels: list[str],
    device: torch.device,
) -> list[tuple[str, str, float]]:
    model.eval()
    predictions: list[tuple[str, str, float]] = []
    with torch.no_grad():
        for _, row in frame.iterrows():
            wafer_map = np.load(ROOT_DIR / str(row["map_path"]))
            tensor = torch.from_numpy(wafer_map.astype("float32") / 2.0).unsqueeze(0).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor).squeeze(0), dim=0).cpu()
            class_index = int(probabilities.argmax().item())
            predictions.append((str(row["wafer_id"]), labels[class_index], float(probabilities[class_index].item())))
    return predictions


def train_model(
    epochs: int = 6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = RANDOM_SEED,
    persist_db: bool = True,
    validation_size: float = 0.15,
    early_stopping_patience: int = 8,
) -> dict[str, Any]:
    ensure_directories()
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_df = load_or_create_raw_dataset()
    if not (TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists() and LABEL_MAP_PATH.exists()):
        preprocess_dataset(raw_df)

    labels = _load_labels()
    train_frame, _ = _prepare_train_validation_split(seed=seed, validation_size=validation_size)
    device = _device()

    train_loader = _make_loader(ACTIVE_TRAIN_CSV_PATH, batch_size=batch_size, shuffle=True, augment=True, seed=seed)
    validation_loader = _make_loader(VALIDATION_CSV_PATH, batch_size=batch_size, shuffle=False, augment=False, seed=seed)
    test_loader = _make_loader(TEST_CSV_PATH, batch_size=batch_size, shuffle=False, augment=False, seed=seed)

    model = WaferCNN(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_frame, len(labels), device=device))

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)

        train_loss = total_loss / len(train_loader.dataset)
        validation_metrics = _evaluate(model, validation_loader, labels, device=device, criterion=criterion)
        validation_f1 = float(validation_metrics["f1_score"])
        scheduler.step(validation_f1)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": float(validation_metrics["loss"]),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_f1": validation_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(history_row)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} "
            "val_loss={validation_loss:.4f} val_f1={validation_f1:.4f}".format(**history_row)
        )

        if validation_f1 > best_score:
            best_score = validation_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch} best_val_f1={best_score:.4f}")
            break

    model.load_state_dict(best_state)
    metrics = _evaluate(model, test_loader, labels, device=device, criterion=criterion)
    metrics["best_epoch"] = best_epoch
    metrics["best_validation_f1"] = float(best_score)
    metrics["device"] = str(device)

    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "metrics": metrics,
            "training_history": history,
        },
        MODEL_PATH,
    )
    _write_metric_artifacts(metrics, labels)
    pd.DataFrame(history).to_csv(TRAINING_HISTORY_PATH, index=False)
    register_model(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        labels=labels,
        metrics=metrics,
        metrics_path=METRICS_JSON_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        classification_report_path=CLASSIFICATION_REPORT_PATH,
        training_history_path=TRAINING_HISTORY_PATH,
    )

    if persist_db:
        repository.init_db()
        repository.upsert_wafer_info(raw_df)
        repository.refresh_defect_statistics(raw_df)
        repository.clear_predictions()
        all_processed = pd.concat([pd.read_csv(TRAIN_CSV_PATH), pd.read_csv(TEST_CSV_PATH)], ignore_index=True)
        for wafer_id, predicted_label, confidence in _predict_records(model, all_processed, labels, device=device):
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
