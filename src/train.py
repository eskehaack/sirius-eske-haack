import argparse
from pathlib import Path
import os

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_builders.dataloader import DataModule
from src.data_builders.smhi_dataloader import parse_config
from src.models import MODELREGISTRY


def train_ddpm(run_id: str, config_path: str = "./src/configs/training_config.toml"):

    seed = 42
    pl.seed_everything(seed, workers=True)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config: dict = parse_config("training", config_path)

    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    num_workers = int(config["num_workers"])
    checkpoint_dir = Path(config["checkpoint_dir"]) / str(run_id)
    checkpoint_every_n_steps = None if int(config["checkpoint_every_n_steps"]) == 0 else int(config["checkpoint_every_n_steps"])
    log_every_n_step = int(config["log_every_n_step"])

    hyperparameters: dict = parse_config("hyperparams", config_path)
    if "max_epochs" in hyperparameters and hyperparameters["max_epochs"] != max_epochs:
        rank_zero_info(f"Warning: Overriding max_epochs in hyperparameters ({hyperparameters['max_epochs']}) with value from config ({max_epochs}).")
    hyperparameters["max_epochs"] = max_epochs
    hyperparameters["run_id"] = run_id

    datamodule = DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        config_path=config_path,
        seed=seed,
    )
    datamodule.setup()

    static_features = datamodule.train_ds.static_features
    model = MODELREGISTRY[hyperparameters["diffusion_model"]](
        backbone=hyperparameters["backbone"],
        hyperparameters=hyperparameters,
        static_features=static_features
    )

    model.save_hyperparameters(config)
    parallel = int(os.environ.get("WORLD_SIZE", 1)) > 1

    rank_zero_info("Checkpoint dir:" + str((Path(checkpoint_dir)).resolve()))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ddpm-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        strategy='ddp' if parallel else "auto",
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        profiler="simple",
        max_epochs=max_epochs,
        accumulate_grad_batches=1 if parallel else 4,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0, patience=8, check_on_train_epoch_end=False)
        ],
        logger=TensorBoardLogger(save_dir="logs", name=str(run_id)),
        log_every_n_steps=log_every_n_step,
        enable_progress_bar=False,
    )

    rank_zero_info(f"Training on {torch.cuda.device_count()} devices.")
    trainer.fit(model, datamodule=datamodule)

    rank_zero_info(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")
    trainer.test(model, datamodule=datamodule)
    return model, trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/training_config.toml",
        help="Path to the training configuration file.",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="ID for the current training run.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = parse_args()

    train_ddpm(run_id=args.run_id, config_path=args.config_path)
