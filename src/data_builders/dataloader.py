from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl

from torch.utils.data import DataLoader

from src.data_builders.smhi_dataloader import TrainingDataset, ClimateDataBuilder, parse_config


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

        date_config = parse_config(config_path=self.config_path, config_keyword="dates")
        self.data_builder = ClimateDataBuilder(
            date_config=date_config,
            preprocessing_config = parse_config(config_path=self.config_path, config_keyword="preprocessing"),
            predictor_config = parse_config(config_path=self.config_path, config_keyword="predictors"),
            static_features_config = parse_config(config_path=self.config_path, config_keyword="static_features"),
            target_config = parse_config(config_path=self.config_path, config_keyword="targets"),
        )

        data_path_config = parse_config(config_path=self.config_path, config_keyword="preprocessing")

        self.train_ds = TrainingDataset(
            predictors_path=data_path_config["predictors_output"],
            targets_path=data_path_config["targets_output"],
            static_features_path=data_path_config["static_output"],
            blocks=date_config.get("train_blocks"),
        )
        print(f"Training dataset size: {len(self.train_ds)} samples")

        self.val_ds = TrainingDataset(
            predictors_path=data_path_config["predictors_output"],
            targets_path=data_path_config["targets_output"],
            static_features_path=data_path_config["static_output"],
            blocks=date_config.get("val_blocks"),
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
