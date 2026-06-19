from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import ROOT_DIR


class WaferMapDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path = ROOT_DIR, augment: bool = False) -> None:
        self.frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        map_path = self.root_dir / str(row["map_path"])
        wafer_map = np.load(map_path).astype("float32") / 2.0
        image = torch.from_numpy(wafer_map).unsqueeze(0)
        if self.augment:
            image = self._augment(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label

    @staticmethod
    def _augment(image: torch.Tensor) -> torch.Tensor:
        # Rotations and flips preserve wafer failure-pattern labels while improving class robustness.
        rotations = int(torch.randint(0, 4, (1,)).item())
        if rotations:
            image = torch.rot90(image, k=rotations, dims=(1, 2))
        if bool(torch.randint(0, 2, (1,)).item()):
            image = torch.flip(image, dims=(1,))
        if bool(torch.randint(0, 2, (1,)).item()):
            image = torch.flip(image, dims=(2,))
        return image.contiguous()
