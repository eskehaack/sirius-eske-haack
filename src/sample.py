import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.model import LitConditionalDDPM
from src.data_builders.dataloader import DataModule


def load_checkpoint(run_id: str, checkpoint: str = "last") -> LitConditionalDDPM:
    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(f"./checkpoints/{run_id}/{checkpoint}.ckpt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}\nModel training did not complete as expected.")
    
    model = LitConditionalDDPM.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def load_data():
    # ----------------------------------------------------
    # Load conditioning image
    # ----------------------------------------------------

    datamodule = DataModule(
        batch_size=1,
        num_workers=1,
    )
    datamodule.setup()
    datamodule.data_builder._get_global_stats()
    stats = datamodule.data_builder.stats
    if stats is None:
        raise ValueError("Normalization stats not found.")

    return datamodule


def get_interpolated(img_size, x, static) -> tuple[torch.Tensor]:
    # ----------------------------------------------------
    # Match interpolation from training
    # ----------------------------------------------------

    condition = F.interpolate(
        x,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    static = F.interpolate(
        static,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    return torch.cat([condition, static], dim=1)

def normalize(condition, n_variables, stats, device):
    # ----------------------------------------------------
    # IMPORTANT:
    # Use EXACTLY the same normalization as training
    # ----------------------------------------------------
    mean = torch.stack([torch.tensor(stat.mean().values) for stat in stats.values()])
    std = torch.stack([torch.tensor(stat.std().values) for stat in stats.values()]) + 1e-6

    mean = F.pad(mean, (0, condition.shape[1] - mean.shape[0]), value=0.0).reshape(1, -1, 1, 1)
    std = F.pad(std, (0, condition.shape[1] - std.shape[0]), value=1.0).reshape(1, -1, 1, 1)

    condition_norm = (condition - mean) / std
    condition_tensor = condition_norm.to(device)

    mean_target_vars = mean[:, :n_variables, :, :]
    std_target_vars = std[:, :n_variables, :, :]

    return condition_tensor, mean_target_vars, std_target_vars

def plot_prediction(metrics, variables, out_dir, cbar_labels):
    n_vars = len(variables)
    n_metrics = len(metrics)

    # Scale figure size with number of variables
    fig, axes = plt.subplots(
        n_metrics,
        n_vars,
        figsize=(3.5 * n_vars, 3.0 * n_metrics),
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, variable in enumerate(variables):
            img = metrics[metric][j]

            ax = axes[i, j]

            mappable = ax.imshow(
                img,
                cmap="coolwarm",
                origin="lower",
                vmin=img.min().item(),
                vmax=img.max().item(),
            )

            # Variable name on top of each column
            if i == 0:
                ax.set_title(str(variable), fontsize=12, pad=8)

            ax.axis("off")

            # Small colorbar attached directly to the image
            cbar = fig.colorbar(
                mappable,
                ax=ax,
                fraction=0.035,
                pad=0.02,
                aspect=30,
            )
            cbar.set_label(cbar_labels[j][i], fontsize=10)

        # Metric name on the left of each row
        axes[i, 0].text(
            -0.25,
            0.5,
            str(metric),
            transform=axes[i, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
        )

    fig.savefig(out_dir / "comparison.png", dpi=300)
    plt.close(fig)


def main(
    run_id: str, checkpoint: str = "last", ensemble_size: int = 5, load_from_npy: str = None
):

    if load_from_npy is not None and Path(load_from_npy).exists():
        out_dir = Path(load_from_npy)
        if (out_dir / "prediction.npy").exists() and (out_dir / "target.npy").exists():
            prediction = torch.from_numpy(np.load(out_dir / "prediction.npy"))
            target = torch.from_numpy(np.load(out_dir / "target.npy"))
            print(f"Loaded prediction and target from {out_dir}")

    else:
        out_dir = Path(f"./samples/{run_id}")
        out_dir = out_dir / time.strftime("%Y%m%d%H%M%S")

        model = load_checkpoint(run_id, checkpoint)
        datamodule = load_data()
        x, y, static, idx = next(iter(datamodule.val_dataloader()))

        # Create Target
        target = torch.stack(list(y.values()), dim =1)
        n_targets = target.shape[1]

        condition = get_interpolated(model.image_size, x, static)
        condition_tensor, mean_target_vars, std_target_vars = normalize(condition, n_targets, datamodule.data_builder.stats, model.device)

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------
        with torch.no_grad():
            prediction = torch.concat([
                model.sample(
                    condition_tensor,
                    num_steps=500,
                )
                for _ in range(ensemble_size)
            ], dim=0)

        # Interpolate back to original size
        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        prediction = prediction.cpu()

        # Undo normalization
        # prediction = prediction * std_target_vars + mean_target_vars
        # target = target * std_target_vars + mean_target_vars

        # ----------------------------------------------------
        # Save arrays
        # ----------------------------------------------------

        out_dir.mkdir(exist_ok=False, parents=True)
        np.save(out_dir / "prediction.npy", prediction)
        np.save(out_dir / "target.npy", target)

        prediction = prediction.detach().cpu()
        target = target.detach().cpu()

    # ----------------------------------------------------
    # Calculate prediction statistics
    # ----------------------------------------------------

    prediction_mean = prediction.mean(dim=0)
    prediction_std = prediction.std(dim=0)

    metrics = {
        "Prediction Std": prediction_std, 
        "Prediction Mean": prediction_mean, 
        "Ground Truth": target.squeeze()
    }
    colorbar_labels = [["Uncertainty", "Temperature [K]", "Temperature [K]"] for _ in range(3)]
    colorbar_labels.append(["Uncertainty", "Precipitation [mm/second]", "Precipitation [mm/second]"])
    variables = ["Mean Temperature", "Minimum Temperature", "Maximum Temperature", "Precipitation"]

    plot_prediction(metrics, variables=variables, out_dir=out_dir, cbar_labels=colorbar_labels)



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
        "--load_from_npy",
        type=str,
        default=None,
        help="Path to the .npy file containing the data to load",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id, args.checkpoint, args.ensemble_size, args.load_from_npy)
