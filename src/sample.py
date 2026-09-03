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


def get_interpolated(img_size, x, static, target) -> tuple[torch.Tensor]:
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

    target = F.interpolate(
        target,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    return torch.cat([condition, static], dim=1), target

def unnormalize(prediction, target, stats, n_variables):
    # ----------------------------------------------------
    # IMPORTANT:
    # Use EXACTLY the same normalization as training
    # ----------------------------------------------------
    stat_tensor = torch.stack([torch.tensor(stat.values) for stat in stats.values()]).T
    stat_tensor = stat_tensor.to(prediction.device)
    mean, std = stat_tensor
    mean = mean[None,:,None,None] # Map to shape 1, channels, 1, 1 for broadcast
    std = std[None,:,None,None] # Map to shape 1, channels, 1, 1 for broadcast

    mean_target_vars = mean[:, :n_variables, :, :]
    std_target_vars = std[:, :n_variables, :, :]

    prediction = prediction * std_target_vars + mean_target_vars
    target = target * std_target_vars + mean_target_vars

    return prediction, target

def plot_prediction(metrics, variables, scales, out_dir, cbar_labels, ensemble_size):
    n_vars = len(variables)
    n_metrics = len(metrics)
    ensemble_size = ensemble_size

    # Scale figure size with number of variables
    fig, axes = plt.subplots(
        n_metrics,
        n_vars,
        figsize=(3.5 * n_vars, 3.0 * n_metrics),
        squeeze=False,
    )

    fig.suptitle(f"Comparison of Prediction and Ground Truth [Ensemble Size: {ensemble_size}]")

    for i, metric in enumerate(metrics):
        for j, variable in enumerate(variables):
            img = metrics[metric][j]
            vmin, vmax = scales[i][j]
            ax = axes[i, j]

            mappable = ax.imshow(
                img,
                cmap="coolwarm",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
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
        target = target.to(model.device)
        n_targets = target.shape[1]
        org_img_shape = target.shape[-2:]

        condition, target = get_interpolated(model.image_size, x, static, target)
        condition = condition.to(model.device)

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------
        with torch.no_grad():
            prediction = torch.concat([
                model.sample(
                    condition,
                    num_steps=500,
                )
                for _ in range(ensemble_size)
            ], dim=0)

        # Undo normalization
        prediction, target = unnormalize(prediction, target, datamodule.data_builder.stats, n_targets)

        # Interpolate back to original size
        prediction = F.interpolate(
            prediction,
            size=org_img_shape,
            mode="bilinear",
            align_corners=False,
        )

        # Interpolate back to original size
        target = F.interpolate(
            target,
            size=org_img_shape,
            mode="bilinear",
            align_corners=False,
        )

        prediction = prediction.cpu()
        target = target.cpu()

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

    # Scale precipitation to mm/day for plotting
    prediction[:, 3, :, :] *= 86400
    target[:, 3, :, :] *= 86400

    prediction_mean = prediction.mean(dim=0)
    prediction_std = prediction.std(dim=0)
    target = target.squeeze()
    asb_err = torch.abs(prediction_mean - target)

    min_temp_std = prediction_std[0:3].min().item()
    max_temp_std = prediction_std[0:3].max().item()
    min_prec_std = prediction_std[3].min().item()
    max_prec_std = prediction_std[3].max().item()
    scale_std = [(min_temp_std, max_temp_std), (min_temp_std, max_temp_std), (min_temp_std, max_temp_std), (min_prec_std, max_prec_std)]

    min_temp = min(prediction_mean[1].min().item(), target[1].min().item())
    max_temp = max(prediction_mean[2].max().item(), target[2].max().item())
    min_prec = min(prediction_mean[3].min().item(), target[3].min().item())
    max_prec = max(prediction_mean[3].max().item(), target[3].max().item())
    scale_mean = [(min_temp, max_temp), (min_temp, max_temp), (min_temp, max_temp), (min_prec, max_prec)]

    min_temp_err = asb_err[0:3].min().item()
    max_temp_err = asb_err[0:3].max().item()
    min_prec_err = asb_err[3].min().item()
    max_prec_err = asb_err[3].max().item()
    scale_err = [(min_temp_err, max_temp_err), (min_temp_err, max_temp_err), (min_temp_err, max_temp_err), (min_prec_err, max_prec_err)]

    scales = [scale_std, scale_mean, scale_mean, scale_err]

    metrics = {
        "Prediction Std": prediction_std,
        "Prediction Mean": prediction_mean,
        "Ground Truth": target,
        "Absolute Error": asb_err
    }
    colorbar_labels = [["Uncertainty", "Temperature [K]", "Temperature [K]", "Absolute Error"] for _ in range(3)]
    colorbar_labels.append(["Uncertainty", "Precipitation [kg/m²/day]", "Precipitation [kg/m²/day]", "Absolute Error"])
    variables = ["Mean Temperature", "Minimum Temperature", "Maximum Temperature", "Precipitation"]

    plot_prediction(
        metrics, 
        variables=variables, 
        scales=scales,
        out_dir=out_dir, 
        cbar_labels=colorbar_labels, 
        ensemble_size=args.ensemble_size
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
        "--load_from_npy",
        type=str,
        default=None,
        help="Path to the .npy file containing the data to load",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run_id, args.checkpoint, args.ensemble_size, args.load_from_npy)
