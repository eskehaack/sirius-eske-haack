from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl

from torch.utils.data import DataLoader

from src.data_builders.smhi_dataloader import ClimateDataBuilder, parse_config
from src.data_builders.dataset import ChunkDataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        config_path: str = "./src/configs/training_config.toml",
        seed: int = 42
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config_path = config_path
        self.seed = seed

        self.date_config = parse_config(config_path=self.config_path, config_keyword="dates")
        self.data_builder = ClimateDataBuilder(
            date_config=self.date_config,
            preprocessing_config = parse_config(config_path=self.config_path, config_keyword="preprocessing"),
            predictor_config = parse_config(config_path=self.config_path, config_keyword="predictors"),
            static_features_config = parse_config(config_path=self.config_path, config_keyword="static_features"),
            target_config = parse_config(config_path=self.config_path, config_keyword="targets"),
        )

        self.data_path_config = parse_config(config_path=self.config_path, config_keyword="preprocessing")

        self.block_map = {
            "train": "train_blocks",
            "val": "val_blocks",
            "test": "inference_blocks",
        }

    def _setup_dataset(self, split: str) -> ChunkDataset:
        if split not in self.block_map:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.block_map.keys())}")

        dataset = ChunkDataset(
            predictors_path=self.data_path_config["predictors_output"],
            targets_path=self.data_path_config["targets_output"],
            static_features_path=self.data_path_config["static_output"],
            blocks=self.date_config.get(self.block_map[split]),
        )
        print(f"{split.capitalize()} dataset size: {len(dataset)} samples")
        return dataset

    def setup(self, stage: Optional[str] = None):

        self.train_ds = self._setup_dataset("train")
        self.val_ds = self._setup_dataset("val")
        self.test_ds = self._setup_dataset("test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.train_ds.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.val_ds.collate,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.test_ds.collate,
        )
