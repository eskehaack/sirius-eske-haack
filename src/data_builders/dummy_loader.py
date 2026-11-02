from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        config_path: str = "./src/configs/training_config.toml",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config_path = config_path

    def setup(self, stage: Optional[str] = None):

        self.train_ds = TrainingDataset(
        )
        print(f"Training dataset size: {len(self.train_ds)} samples")

        self.val_ds = TrainingDataset(
        )
        print(f"Validation dataset size: {len(self.val_ds)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class TrainingDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 4000

    def __getitem__(self, idx):
        x = torch.zeros((18, 51, 131))
        y = torch.zeros((4, 412, 424))
        y = {
            var: y[i]
            for i, var in enumerate(["tas", "tasmin", "tasmax", "pr"])
        }
        static = torch.zeros((2, 412, 424))

        return x, y, static, idx