"""
Visualisation of DDPM downscaling predictions.

Grid layout: one row per variable (tas, tasmin, tasmax, pr),
four columns: ground truth | ensemble mean | ensemble std | absolute error.

Usage
-----
    from visualise import plot_predictions
    fig = plot_predictions(abs_prediction, data["targets"], date="1952-03-15")
    fig.savefig("prediction.png", dpi=150, bbox_inches="tight")
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from src.postprocessing import plt_guide as pg

# Variable display metadata: (label, unit, colormap)
_VAR_META = {
    "tas":    ("Mean temp.",     "K",       "RdYlBu_r"),
    "tasmin": ("Min temp.",      "K",       "RdYlBu_r"),
    "tasmax": ("Max temp.",      "K",       "RdYlBu_r"),
    "pr":     ("Precipitation",  "mm/day",  "YlGnBu"),
}
_COL_TITLES = ["Ground truth", "Ensemble mean", "Ensemble std", "Absolute error"]


def _to_numpy(x) -> np.ndarray:
    """Accept torch.Tensor, xr.DataArray, or np.ndarray → np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, xr.DataArray):
        return x.values
    return np.asarray(x)


def _extract_targets(targets: xr.Dataset, var_names: list[str]) -> dict[str, np.ndarray]:
    """Pull each variable from the target Dataset into a plain array."""
    out = {}
    for v in var_names:
        if v not in targets.data_vars:
            raise KeyError(f"Variable '{v}' not found in targets dataset.")
        out[v] = _to_numpy(targets[v])
    return out


def _symmetric_norm(data: np.ndarray) -> mcolors.Normalize:
    """Diverging normalisation centred at zero (for error maps)."""
    vmax = np.nanmax(np.abs(data))
    return mcolors.Normalize(vmin=-vmax, vmax=vmax)


def plot_predictions(
    prediction: torch.Tensor | np.ndarray,
    targets: xr.Dataset,
    date: str = "",
    output_path: str | None = None
) -> plt.Figure:
    """
    Plot ground truth, ensemble mean, ensemble std, and absolute error
    for each of the four target variables on a spatial grid.

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray, shape (ensemble, 4, H, W)
        Absolute (un-residualised) predictions from the model.
    targets : xr.Dataset
        Target dataset with variables tas, tasmin, tasmax, pr.
        Must contain lat/lon coordinates.
    date : str
        Date string shown in the figure title.

    Returns
    -------
    matplotlib Figure
    """

    pg.setup()

    pred = _to_numpy(prediction)           # (E, 4, H, W)
    var_names = list(_VAR_META.keys())
    gt_arrays = _extract_targets(targets, var_names)

    n_vars = len(var_names)
    n_cols = 4
    fig = plt.figure(figsize=(pg.FULL_WIDTH, pg.FULL_WIDTH*0.9))

    temp_norm = mcolors.Normalize(vmin=gt_arrays['tasmin'].min(), vmax=gt_arrays['tasmax'].max())
    data_crs = ccrs.PlateCarree()
    rp = targets['rotated_latitude_longitude'].attrs
    proj = ccrs.RotatedPole(
        pole_longitude=rp['grid_north_pole_longitude'],
        pole_latitude=rp['grid_north_pole_latitude'],
    )

    axes = np.array(
        [
            [
                fig.add_subplot(4, 4, i + 1 + j * n_cols, projection=proj) 
                for i in range(n_cols)
            ] for j in range(n_vars)
        ]
    )

    # Get spatial coordinates for axis labels
    lats = targets.lat.values if "lat" in targets.coords else None
    lons = targets.lon.values if "lon" in targets.coords else None

    title = f"Prediction diagnostics"
    if date:
        title += f"  ·  {date}"
    fig.suptitle(title, y=1.01)

    for col_idx, col_title in enumerate(_COL_TITLES):
        axes[0, col_idx].set_title(col_title, pad=6)

    for row_idx, var in enumerate(var_names):
        label, unit, cmap_name = _VAR_META[var]
        var_idx = row_idx                 # variable order matches _VAR_META

        if var == "pr":
            # Precipitation is in kg/m²/s, convert to mm/day for plotting
            gt_arrays[var] *= 86400.0
            pred[:, var_idx] *= 86400.0

        gt     = gt_arrays[var]           # (H, W)
        ens    = pred[:, var_idx]         # (E, H, W)
        mean   = ens.mean(axis=0)         # (H, W)
        std    = ens.std(axis=0)          # (H, W)
        error  = mean - gt                # (H, W)  signed error

        # Add a plain axes spanning the row's left edge for the ylabel
        row_label_ax = fig.add_axes(
            [0.0, 1 - (row_idx + 1) / n_vars, 0.02, 1 / n_vars]
        )
        row_label_ax.axis("off")
        row_label_ax.text(
            -0.5, 0.5, f"{label} - [{unit}]",
            transform=row_label_ax.transAxes,
            va="center", ha="center",
            rotation=90
        )

        cmap = plt.get_cmap(cmap_name)
        if not "tas" in var:
            # Shared colour limits for truth / mean (so they're directly comparable)
            all_vals = np.concatenate([gt.ravel(), mean.ravel()])
            vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = temp_norm

        panels = [
            (gt,    cmap,                    norm,                     False),
            (mean,  cmap,                    norm,                     False),
            (std,   plt.get_cmap("Oranges"), None,                     False),
            (error, plt.get_cmap("bwr"),     _symmetric_norm(error),   True),
        ]

        for col_idx, (arr, cm, norm, is_error) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            norm = norm or mcolors.Normalize(
                vmin=np.nanpercentile(arr, 2),
                vmax=np.nanpercentile(arr, 98),
            )

            # Use real coordinates when available, pixel indices otherwise
            if lats is not None and lons is not None:
                im = ax.pcolormesh(lons, lats, arr, cmap=cm, norm=norm, rasterized=True, transform=data_crs)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5) 
                ax.gridlines(linewidth=0.5, color="grey", alpha=0.4, linestyle="--")
                if row_idx == n_vars - 1:
                    ax.set_xlabel("Longitude")
            else:
                im = ax.imshow(arr, cmap=cm, norm=norm, origin="upper", aspect="auto")

            # Colourbar
            cb = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cm),
                ax=ax,
                orientation="vertical",
                fraction=0.048,
                pad=0.01,
            )
            ax.spines['geo'].set_visible(False)

            # Annotate stats in corner
            stats_txt = f"μ={arr.mean():.2f}  σ={arr.std():.2f}"
            ax.text(
                0.00, -0.02, 
                stats_txt,
                transform=ax.transAxes,
                va="top", ha="left",
                color="black",
            )
    if output_path:
        pg.save(fig, output_path)
        plt.close(fig)

    return fig

if __name__ == "__main__":
    import argparse
    import toml
    from pathlib import Path
    import xarray as xr
    from src.data_builders.data_utils import load_sample

    parser = argparse.ArgumentParser(description="Plot DDPM downscaling predictions.")
    parser.add_argument("prediction", type=Path, help="Path to prediction .npy file (ensemble, 4, H, W).")
    parser.add_argument("--date", type=str, default="1951-01-01", help="Date string for figure title.")
    parser.add_argument("--output", type=Path, default="test.png", help="Output path for saved figure.")
    args = parser.parse_args()

    # Load data
    config_path = Path("./src/configs/sample_config.toml")
    config = toml.load(config_path)

    prediction = np.load(args.prediction)
    targets = load_sample(config=config, scenario="historical", member="r1i1p1f1", date=args.date)["targets"]
    # Plot
    fig = plot_predictions(prediction, targets, date=args.date)

    # Save or show
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.output}")
    else:
        plt.show()