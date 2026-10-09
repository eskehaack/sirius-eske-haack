import argparse
from pathlib import Path
import toml
import yaml

import xarray as xr
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.model import LitConditionalDDPM, LitConditionalUNet
from src.data_builders.data_utils import load_sample
from src.postprocessing.inference.plot_samples import plot_predictions, plot_prediction_distribution

def load_checkpoint(run_id: str, checkpoint: str = "last") -> LitConditionalDDPM:
    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(f"./checkpoints/{run_id}/{checkpoint}.ckpt")
    log_path = Path(f"./logs/{run_id}/version_0/hparams.yaml")
    hparams = yaml.safe_load(open(log_path, "r"))
    m_spec = hparams.get('model', "conditional_ddpm")
    mclass = {
        "conditional_ddpm": LitConditionalDDPM,
        "conditional_unet": LitConditionalUNet,
    }

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}\nModel training did not complete as expected.")
    
    model = mclass[m_spec].load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def get_resized(img_size, target_shape, x, static) -> tuple[torch.Tensor]:
    # ----------------------------------------------------
    # Match interpolation from training
    # ----------------------------------------------------

    pad_x = (img_size - target_shape[0]) // 2
    pad_y = (img_size - target_shape[1]) // 2
    pad_tuple = (pad_y, pad_y, pad_x, pad_x)  # (left, right, top, bottom)

    # Interpolate the condition to match the target shape
    condition = F.interpolate(
        x,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    # Then pad to meet model requirements
    condition = F.pad(
        condition,
        pad=pad_tuple,
        mode="constant",
        value=0.0,
    )

    # Interpolate the statics to match the target shape
    static = F.interpolate(
        static,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    # Then pad to meet model requirements
    static = F.pad(
        static,
        pad=pad_tuple,
        mode="constant",
        value=0.0,
    )

    return torch.cat([condition, static], dim=1), (pad_x, pad_y)

def normalize(x, stats):
    for i, var in enumerate(stats.data_vars):
        mean = torch.tensor(stats[var].sel(statistic="pred_mean").values).to(x.device)
        std = torch.tensor(stats[var].sel(statistic="pred_std").values).to(x.device)
        x[:, i, :, :] = (x[:, i, :, :] - mean) / std
    return x

def unnormalize(prediction, stats, target_vars):
    for i, var in enumerate(target_vars):
        mean = torch.tensor(stats[var].sel(statistic="targ_mean").values).to(prediction.device)
        std = torch.tensor(stats[var].sel(statistic="targ_std").values).to(prediction.device)
        prediction[:, i, :, :] = prediction[:, i, :, :] * std + mean
    return prediction

def unresidualise(regridded, prediction, target_vars):
    prediction = prediction.clone()  # Avoid modifying the original tensor
    for ensemble in prediction:
        for i, var in enumerate(target_vars):
            ensemble[i, :, :] += torch.tensor(regridded[var].values).to(prediction.device)
    return prediction

def main(
    run_id: str, 
    checkpoint: str = "last", 
    ensemble_size: int = 5, 
    timesteps: int = 500,
    scenario: str = "historical",
    member: str = "r1i1p1f1",
    date: str = "1951-01-01",
    config_path: str = "./src/configs/sample_config.toml", 
):

    out_dir = Path(f"./samples/{run_id}")
    out_dir = out_dir / f"{scenario}_{member}_{date.replace('-', '')}"

    model = load_checkpoint(run_id, checkpoint)
    config = toml.load(config_path)
    data = load_sample(config, date=date, scenario=scenario, member=member)

    def ds_to_channels(ds: xr.Dataset) -> np.ndarray:
        """
        Convert a Dataset to a (C, H, W) array.
        Surface variables (H, W) become 1 channel each.
        Multi-level variables (plev, H, W) are flattened into plev channels.
        """
        channels = []
        for v in ds.data_vars:
            arr = ds[v].values  # (H, W) or (plev, H, W)
            if arr.ndim == 2:
                channels.append(arr[np.newaxis])  # → (1, H, W)
            else:
                channels.append(arr)              # → (plev, H, W)
        return np.concatenate(channels, axis=0)   # (C_total, H, W)

    # Stack all variables into (1, C, H, W) tensors
    x = torch.from_numpy(ds_to_channels(data['predictors'])
    ).unsqueeze(0).float().to(model.device)  # (1, C_pred, H, W)

    static = torch.from_numpy(ds_to_channels(data['static'])
    ).unsqueeze(0).float().to(model.device)  # (1, C_static, H, W)

    target_shape = data['targets']['lon'].shape
    target_vars = ["tas", "tasmin", "tasmax", "pr"]


    stats = xr.open_dataset(config['preprocessing']['normalization_stats_path'])
    x = normalize(x, stats)
    condition, (pad_x, pad_y) = get_resized(model.image_size, target_shape, x, static)
    condition = condition.to(model.device)

    condition[condition != condition] = 0.0  # Replace NaNs with zeros

    # ----------------------------------------------------
    # Sample
    # ----------------------------------------------------
    with torch.no_grad():
        prediction = torch.concat([
            model.sample(
                condition,
                num_steps=timesteps,
            )
            for _ in range(ensemble_size)
        ], dim=0)

    # Undo normalization
    prediction = unnormalize(prediction, stats, target_vars)
    # Crop back to original size (guard against pad == 0, where slice [0:-0] is empty)
    prediction = prediction[
        :, :,
        pad_x : prediction.shape[2] - pad_x if pad_x > 0 else None,
        pad_y : prediction.shape[3] - pad_y if pad_y > 0 else None,
    ]
    prediction = prediction.cpu()

    # prediction here is the residual (model output), save before unresidualising
    res_prediction = prediction.detach().cpu()
    abs_prediction = unresidualise(data['regridded'], prediction, target_vars).cpu()

    # ----------------------------------------------------
    # Save arrays
    # ----------------------------------------------------

    out_dir.mkdir(exist_ok=True, parents=True)
    np.save(out_dir / f"res_prediction_{ensemble_size}_{timesteps}.npy", res_prediction.numpy())
    np.save(out_dir / f"abs_prediction_{ensemble_size}_{timesteps}.npy", abs_prediction.numpy())

    plot_predictions(
        abs_prediction, 
        data["targets"], 
        date=date, 
        ensemble_size=ensemble_size,
        timesteps=timesteps,
        output_path=out_dir / f"prediction_diagnostics_{ensemble_size}_{timesteps}.png"
    )

    plot_prediction_distribution(
        abs_prediction, 
        data["targets"], 
        date=date, 
        ensemble_size=ensemble_size,
        timesteps=timesteps,
        output_path=out_dir / f"prediction_distributions_{ensemble_size}_{timesteps}.png"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a DDPM model.")

    parser.add_argument(
        "--run_id", type=str, help="Run ID defining directory for checkpoint file"
    )

    parser.add_argument(
        "--checkpoint", type=str, default="last", help="Name of checkpoint file"
    )

    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="How many images in the ensamble to generate",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Number of timesteps to use in the sampling process",
    )

    parser.add_argument(
        "--scenario", type=str, default="historical", help="Scenario to sample from"
    )

    parser.add_argument(
        "--member", type=str, default="r1i1p1f1", help="Member to sample from"
    )

    parser.add_argument(
        "--date", type=str, default="1951-01-01", help="Date to sample from"
    )

    parser.add_argument(
        "--config_path", type=str, default="./src/configs/sample_config.toml", help="Path to the config file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        ensemble_size=args.ensemble_size,
        timesteps=args.timesteps,
        scenario=args.scenario,
        member=args.member,
        date=args.date,
        config_path=args.config_path,
    )