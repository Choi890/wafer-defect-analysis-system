from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import ROOT_DIR


class WaferMapDataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path = ROOT_DIR) -> None:
        self.frame = pd.read_csv(csv_path)
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        map_path = self.root_dir / str(row["map_path"])
        wafer_map = np.load(map_path).astype("float32") / 2.0
        image = torch.from_numpy(wafer_map).unsqueeze(0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label
