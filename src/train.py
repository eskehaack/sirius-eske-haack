import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_builders.dataloader import DataModule
from src.data_builders.smhi_dataloader import parse_config
from src.models import MODELREGISTRY
from src.callbacks.device_logger import AMDGPUMonitor


def train_ddpm(run_id: str, config_path: str = "./src/configs/training_config.toml"):

    pl.seed_everything(42, workers=True)

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
        print(f"Warning: Overriding max_epochs in hyperparameters ({hyperparameters['max_epochs']}) with value from config ({max_epochs}).")
    hyperparameters["max_epochs"] = max_epochs

    datamodule = DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        config_path=config_path,
    )

    model = MODELREGISTRY[hyperparameters["diffusion_model"]](
        backbone=hyperparameters["backbone"],
        hyperparameters=hyperparameters
    )

    model.save_hyperparameters(config)

    print("Checkpoint dir:" + str((Path(checkpoint_dir)).resolve()))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ddpm-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=checkpoint_every_n_steps
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        accumulate_grad_batches=4,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            AMDGPUMonitor(interval=1.0, device_index=0),
            EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0, patience=8)
        ],
        logger=TensorBoardLogger(save_dir="logs", name=str(run_id)),
        log_every_n_steps=log_every_n_step,
        val_check_interval=checkpoint_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)

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
    print("Torch Version:", torch.__version__)
    print("Torch CUDA Available:", torch.cuda.is_available())
    print("Torch TPU Available:", torch.backends.mps.is_available())
    print("Torch CUDA Device:", torch.cuda.get_device_name(0))
    print("Torch HIP Version:", torch.version.hip)

    print("Tensorboard log dir:" + str((Path("logs") / "conditional_ddpm").resolve()))

    torch.set_float32_matmul_precision("medium")

    args = parse_args()

    train_ddpm(run_id=args.run_id, config_path=args.config_path)
