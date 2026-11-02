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
import matplotlib as mpl
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
    ensemble_size: int = 5,
    timesteps: int = 500,
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
    title += f"  ·  Ensemble size: {ensemble_size}, Timesteps: {timesteps}"
    fig.suptitle(title, y=1.01)

    for col_idx, col_title in enumerate(_COL_TITLES):
        axes[0, col_idx].set_title(col_title, pad=6)

    for row_idx, var in enumerate(var_names):
        label, unit, cmap_name = _VAR_META[var]
        var_idx = row_idx                 # variable order matches _VAR_META

        if var == "pr":
            # Precipitation is in kg/m²/s, convert to mm/day for plotting
            gt     = gt_arrays[var].copy()    # (H, W)
            ens    = pred[:, var_idx].copy()  # (E, H, W)
            gt    *= 86400.0
            ens   *= 86400.0
            
            mean   = ens.mean(axis=0)         # (H, W)
            std    = ens.std(axis=0)          # (H, W)
            error  = mean - gt

        else:
            gt     = gt_arrays[var].copy()    # (H, W)
            ens    = pred[:, var_idx].copy()  # (E, H, W)
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
                vmin=np.nanpercentile(arr, 10),
                vmax=np.nanpercentile(arr, 90),
            )

            if not is_error:
                # Colourbar
                cb = fig.colorbar(
                    ScalarMappable(norm=norm, cmap=cm),
                    ax=ax,
                    orientation="vertical",
                    fraction=0.048,
                    pad=0.01,
                )
                ax.spines['geo'].set_visible(False)
            else:
                # Use sequential colourbar for error maps
                bounds = np.linspace(-np.nanmax(np.abs(arr)), np.nanmax(np.abs(arr)) + 1, 11, dtype=np.int32)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=cm)

                cb = fig.colorbar(
                    mpl.colorizer.ColorizingArtist(colorizer),
                    ax=ax,
                    orientation="vertical",
                    fraction=0.048,
                    pad=0.01,
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


            # Annotate stats in corner
            stats_txt = f"μ={arr.mean():.2f}  σ={arr.std():.2f}"
            if col_idx == 1:  # mean panel
                stats_txt += f"\nMSE={(np.mean((mean - gt) ** 2)):.2f}"
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

def plot_prediction_distribution(
    prediction: torch.Tensor | np.ndarray,
    targets: xr.Dataset,
    date: str = "",
    ensemble_size: int = 5,
    timesteps: int = 500,
    output_path: str | None = None
):
    """
    Plot the distribution of predictions for each variable.

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray, shape (ensemble, 4, H, W)
        Absolute (un-residualised) predictions from the model.
    targets : xr.Dataset
        Target dataset with variables tas, tasmin, tasmax, pr.
        Must contain lat/lon coordinates.
    date : str
        Date string shown in the figure title.
    """
    pg.setup()

    pred = _to_numpy(prediction)           # (E, 4, H, W)
    var_names = list(_VAR_META.keys())
    gt_arrays = _extract_targets(targets, var_names)

    fig = plt.figure()
    fig.suptitle(f'Distribution of Predictions · {date} · Ensemble size: {ensemble_size}, Timesteps: {timesteps}')

    # Plot the distribution of predictions for each variable
    for i, var in enumerate(var_names):

        ax = fig.add_subplot(2, 2, i + 1)

        label, unit, _ = _VAR_META[var]
        var_idx = i                 # variable order matches _VAR_META

        if var == "pr":
            # Precipitation is in kg/m²/s, convert to mm/day for plotting
            gt     = gt_arrays[var].copy()    # (H, W)
            ens    = pred[:, var_idx].copy()  # (E, H, W)
            gt    *= 86400.0
            ens   *= 86400.0
            mean   = ens.mean(axis=0)         # (H, W)
            
        else:
            gt     = gt_arrays[var].copy()    # (H, W)
            ens    = pred[:, var_idx].copy()  # (E, H, W)
            mean   = ens.mean(axis=0)         # (H, W)

        all_vals  = np.concatenate([mean.ravel(), gt.ravel()])
        bin_edges = np.linspace(np.nanpercentile(all_vals, 1),
                                np.nanpercentile(all_vals, 99), 51)
        ax.hist(mean.ravel(), bins=bin_edges, alpha=0.5, density=True, label='Ensemble Mean')
        ax.hist(gt.ravel(),   bins=bin_edges, alpha=0.5, density=True, label='Ground Truth')
        ax.set_title(f'{label}')
        ax.set_xlabel(f'{label} [{unit}]')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()
    
    if output_path:
        pg.save(fig, output_path)
        plt.close(fig)

    return fig

def plot_ensemble_members_spatial(
    prediction: torch.Tensor | np.ndarray,
    targets: xr.Dataset,
    date: str = "",
    ensemble_size: int = 5,
    timesteps: int = 500,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Generated by: claude-sonnet-4-6
    Prompt summary: Plot a grid of spatial maps with one row per variable and one column per
    ensemble member, mirroring the pcolormesh style of plot_predictions but showing each member
    individually rather than the ensemble mean.

    Plot a spatial map for every ensemble member × variable combination.

    Grid layout: one row per variable (tas, tasmin, tasmax, pr),
    one column per ensemble member.

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray, shape (ensemble, 4, H, W)
        Absolute (un-residualised) predictions from the model.
    targets : xr.Dataset
        Target dataset with variables tas, tasmin, tasmax, pr.
        Must contain lat/lon coordinates.
    date : str
        Date string shown in the figure title.
    ensemble_size : int
        Number of ensemble members (used for the title only; actual member
        count is taken from the first axis of *prediction*).
    timesteps : int
        Number of diffusion timesteps (shown in title).
    output_path : str or None
        If given the figure is saved here and closed.

    Returns
    -------
    matplotlib Figure
    """
    pg.setup()

    pred = _to_numpy(prediction)           # (E, 4, H, W)
    n_members = pred.shape[0]
    var_names  = list(_VAR_META.keys())
    n_vars     = len(var_names)

    gt_arrays = _extract_targets(targets, var_names)

    data_crs = ccrs.PlateCarree()
    rp = targets["rotated_latitude_longitude"].attrs
    proj = ccrs.RotatedPole(
        pole_longitude=rp["grid_north_pole_longitude"],
        pole_latitude=rp["grid_north_pole_latitude"],
    )

    lats = targets.lat.values  if "lat"  in targets.coords else None
    lons = targets.lon.values  if "lon"  in targets.coords else None

    fig_w = pg.FULL_WIDTH * (n_members / 4)
    fig_h = pg.FULL_WIDTH * 0.9
    fig   = plt.figure(figsize=(fig_w, fig_h))

    title = "Ensemble members – spatial maps"
    if date:
        title += f"  ·  {date}"
    title += f"  ·  Ensemble size: {ensemble_size}, Timesteps: {timesteps}"
    fig.suptitle(title, y=1.01)

    axes = np.array(
        [
            [
                fig.add_subplot(n_vars, n_members, row * n_members + col + 1, projection=proj)
                for col in range(n_members)
            ]
            for row in range(n_vars)
        ]
    )  # shape (n_vars, n_members)

    # Column headers
    for col in range(n_members):
        axes[0, col].set_title(f"Member {col + 1}", pad=6)

    # Shared temperature normalisation across all three temperature variables
    temp_vmin = min(gt_arrays["tasmin"].min(), pred[:, 1].min())
    temp_vmax = max(gt_arrays["tasmax"].max(), pred[:, 2].max())
    temp_norm = mcolors.Normalize(vmin=temp_vmin, vmax=temp_vmax)

    for row_idx, var in enumerate(var_names):
        label, unit, cmap_name = _VAR_META[var]
        cmap = plt.get_cmap(cmap_name)
        var_idx = row_idx

        # Build per-row colour limits from ground truth + all members
        if var == "pr":
            gt  = gt_arrays[var].copy() * 86400.0
            ens = pred[:, var_idx].copy() * 86400.0
        else:
            gt  = gt_arrays[var].copy()
            ens = pred[:, var_idx].copy()

        if "tas" in var:
            norm = temp_norm
        else:
            all_vals = np.concatenate([gt.ravel(), ens.ravel()])
            vmin     = np.nanpercentile(all_vals, 2)
            vmax     = np.nanpercentile(all_vals, 98)
            norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Row label
        row_label_ax = fig.add_axes(
            [0.0, 1 - (row_idx + 1) / n_vars, 0.02, 1 / n_vars]
        )
        row_label_ax.axis("off")
        row_label_ax.text(
            -0.5, 0.5, f"{label} – [{unit}]",
            transform=row_label_ax.transAxes,
            va="center", ha="center", rotation=90,
        )

        for col_idx in range(n_members):
            ax  = axes[row_idx, col_idx]
            arr = ens[col_idx]             # (H, W) for this member

            if lats is not None and lons is not None:
                ax.pcolormesh(lons, lats, arr, cmap=cmap, norm=norm,
                              rasterized=True, transform=data_crs)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.gridlines(linewidth=0.5, color="grey", alpha=0.4, linestyle="--")
                if row_idx == n_vars - 1:
                    ax.set_xlabel("Longitude")
            else:
                ax.imshow(arr, cmap=cmap, norm=norm, origin="upper", aspect="auto")

            ax.spines["geo"].set_visible(False)

            stats_txt = f"μ={arr.mean():.2f}  σ={arr.std():.2f}"
            ax.text(0.00, -0.02, stats_txt, transform=ax.transAxes,
                    va="top", ha="left", color="black")

        # One shared colourbar per row, attached to the last axes
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[row_idx, :],
            orientation="vertical",
            fraction=0.02,
            pad=0.02,
        )

    if output_path:
        pg.save(fig, output_path)
        plt.close(fig)

    return fig


def plot_ensemble_members_distribution(
    prediction: torch.Tensor | np.ndarray,
    targets: xr.Dataset,
    date: str = "",
    ensemble_size: int = 5,
    timesteps: int = 500,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Generated by: claude-sonnet-4-6
    Prompt summary: Plot per-ensemble-member histograms alongside the ground truth for each
    variable, extending plot_prediction_distribution so every member gets its own curve rather
    than showing only the ensemble mean.

    Plot the pixel-value distribution for each ensemble member and variable.

    Grid layout: one row per variable (tas, tasmin, tasmax, pr),
    one column per ensemble member.  Each panel overlays the member's
    histogram with the ground-truth histogram for easy comparison.

    Parameters
    ----------
    prediction : torch.Tensor or np.ndarray, shape (ensemble, 4, H, W)
        Absolute (un-residualised) predictions from the model.
    targets : xr.Dataset
        Target dataset with variables tas, tasmin, tasmax, pr.
    date : str
        Date string shown in the figure title.
    ensemble_size : int
        Number of ensemble members (shown in title).
    timesteps : int
        Number of diffusion timesteps (shown in title).
    output_path : str or None
        If given the figure is saved here and closed.

    Returns
    -------
    matplotlib Figure
    """
    pg.setup()

    pred      = _to_numpy(prediction)     # (E, 4, H, W)
    n_members = pred.shape[0]
    var_names  = list(_VAR_META.keys())
    n_vars     = len(var_names)

    gt_arrays = _extract_targets(targets, var_names)

    fig, axes = plt.subplots(
        n_vars, n_members,
        figsize=(pg.FULL_WIDTH * (n_members / 4), pg.FULL_WIDTH * 0.9),
        squeeze=False,
    )

    title = "Ensemble members – distributions"
    if date:
        title += f"  ·  {date}"
    title += f"  ·  Ensemble size: {ensemble_size}, Timesteps: {timesteps}"
    fig.suptitle(title, y=1.01)

    # Column headers
    for col in range(n_members):
        axes[0, col].set_title(f"Member {col + 1}", pad=6)

    for row_idx, var in enumerate(var_names):
        label, unit, _ = _VAR_META[var]
        var_idx = row_idx

        if var == "pr":
            gt  = gt_arrays[var].copy() * 86400.0
            ens = pred[:, var_idx].copy() * 86400.0
        else:
            gt  = gt_arrays[var].copy()
            ens = pred[:, var_idx].copy()

        # Shared bin edges across ground truth and all members so bar widths
        # are identical within each panel and x-limits are comparable across columns
        all_vals = np.concatenate([gt.ravel(), ens.ravel()])
        x_min    = np.nanpercentile(all_vals, 1)
        x_max    = np.nanpercentile(all_vals, 99)
        bin_edges = np.linspace(x_min, x_max, 51)  # 51 edges → 50 bins

        for col_idx in range(n_members):
            ax     = axes[row_idx, col_idx]
            member = ens[col_idx].ravel()

            ax.hist(gt.ravel(),  bins=bin_edges, alpha=0.5, density=True,
                    label="Ground truth", color="steelblue")
            ax.hist(member,      bins=bin_edges, alpha=0.5, density=True,
                    label=f"Member {col_idx + 1}", color="darkorange")

            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(f"{label} [{unit}]")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.4)

            # Only show legend on the first column to reduce clutter
            if col_idx == 0:
                ax.legend(fontsize="small")

        # Row label on the far left
        axes[row_idx, 0].annotate(
            label,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.35, 0.5),
            textcoords="axes fraction",
            va="center", ha="center",
            rotation=90,
            annotation_clip=False,
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
    parser.add_argument("--scenario", type=str, default="historical", help="Scenario to load data from")
    parser.add_argument("--output", type=Path, default="test.png", help="Output path for saved figure.")
    args = parser.parse_args()

    # Load data
    config_path = Path("./src/configs/sample_config.toml")
    config = toml.load(config_path)

    prediction = np.load(args.prediction)
    targets = load_sample(config=config, scenario=args.scenario, member="r1i1p1f1", date=args.date)["targets"]
    # Plot
    # fig = plot_predictions(prediction, targets, date=args.date, output_path="test_diagnostics.png")
    # fig = plot_prediction_distribution(prediction, targets, date=args.date, output_path="test_distribution.png")

    fig = plot_ensemble_members_spatial(prediction, targets, date=args.date, output_path="test_diagnostics.png")
    fig = plot_ensemble_members_distribution(prediction, targets, date=args.date, output_path="test_distribution.png")