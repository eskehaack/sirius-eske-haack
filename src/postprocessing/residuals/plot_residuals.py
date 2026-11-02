from pathlib import Path
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from src.postprocessing import plt_guide as pg


def plot_residuals(x='tas', period='historical'):
    regridded_path = Path(f"/scratch/project_465002687/ec_earth/metrics/regridded_r1i1p1f1_{x}_mean_{period}.nc")
    hclim_path = Path(f"/scratch/project_465002687/ec_earth/metrics/hclim_r1i1p1f1_{x}_mean_{period}.nc")
    residual_path = Path(f"/scratch/project_465002687/ec_earth/metrics/residuals_r1i1p1f1_{x}_mean_{period}.nc")

    ds_reg = xr.open_dataset(regridded_path).isel(time=0)
    ds_hcl = xr.open_dataset(hclim_path).isel(time=0)
    ds_res = xr.open_dataset(residual_path).isel(time=0)

    regridded = ds_reg[x]
    hclim     = ds_hcl[x]
    residual  = ds_res[x]

    if x == 'pr':
        regridded = regridded * 86400.0
        hclim     = hclim * 86400.0
        residual  = residual * 86400.0

    rp = ds_hcl['rotated_latitude_longitude'].attrs
    proj = ccrs.RotatedPole(
        pole_longitude=rp['grid_north_pole_longitude'],
        pole_latitude=rp['grid_north_pole_latitude'],
    )

    # Shared colour range for the two temperature maps; symmetric range for residuals
    vmin = round(min(np.nanpercentile(regridded.values, 2), np.nanpercentile(hclim.values, 2)), 1)
    vmax = round(max(np.nanpercentile(regridded.values, 98), np.nanpercentile(hclim.values, 98)), 1)
    res_abs = round(np.nanpercentile(np.abs(residual.values), 98), 1)

    fig = plt.figure(figsize=(pg.FIG_FULL[0]*1.5, pg.FIG_FULL[1]), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    map_axes  = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
    hist_axes = [fig.add_subplot(gs[1, i])                  for i in range(3)]

    cmap = mpl.colormaps["RdBu_r"]
    abs_bounds = np.linspace(vmin, vmax, 11, dtype=np.float32)
    abs_norm = mpl.colors.BoundaryNorm(abs_bounds, cmap.N)
    res_bounds = np.linspace(-res_abs, res_abs, 11, dtype=np.float32)
    res_norm = mpl.colors.BoundaryNorm(res_bounds, cmap.N)

    def plot_map(ax, data, title, cmap, norm):
        im = ax.pcolormesh(
            ds_hcl['rlon'], ds_hcl['rlat'], data.values,
            cmap=cmap, transform=proj, norm=norm
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="k")
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")
        ax.set_title(title, pad=5)
        return im

    im0 = plot_map(map_axes[0], regridded, f"EC-Earth (regridded)",       cmap, abs_norm)
    im1 = plot_map(map_axes[1], hclim,     f"HCLIM",                      cmap, abs_norm)
    im2 = plot_map(map_axes[2], residual,  f"Residual (EC-Earth - HCLIM)", cmap, res_norm)

    # One shared colorbar for the two temperature maps
    colorizer = mpl.colorizer.Colorizer(norm=abs_norm, cmap=cmap)
    cb_temp = fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer), 
                           ax=map_axes[:2], orientation="vertical",
                           fraction=0.046, pad=0.04, shrink=0.85, extend='both')
    cb_temp.set_label(f"{x} [{regridded.attrs.get('units', '')}]")
    cb_temp.ax.tick_params()

    # Separate colorbar for residuals
    colorizer = mpl.colorizer.Colorizer(norm=res_norm, cmap=cmap)
    cb_res = fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer), 
                          ax=map_axes[2], orientation="vertical",
                          fraction=0.046, pad=0.04, shrink=0.85, extend='both')
    cb_res.set_label(f"Δ{x} [{regridded.attrs.get('units', '')}]")
    cb_res.ax.tick_params()

    def plot_hist(ax, data, xlabel):
        vals = data.values.flatten()
        ax.hist(vals, bins=60, density=True, alpha=0.85, edgecolor="black")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.tick_params()
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    plot_hist(hist_axes[0], regridded, f"{x} [{regridded.attrs.get('units', '')}]")
    plot_hist(hist_axes[1], hclim,     f"{x} [{hclim.attrs.get('units', '')}]")
    plot_hist(hist_axes[2], residual,  f"Δ{x} [{regridded.attrs.get('units', '')}]")

    period_title = {
        "historical": "Historical (1985-2014)",
        "mid": "Mid range (2020-2050)",
        "late": "Late range (2070-2100)",
    }

    fig.suptitle(f"EC-Earth {x.upper()} - HCLIM {x.upper()} ({period_title.get(period, period)})", fontweight="bold")
    pg.save(plt, f"./figures/method/residuals_{x}_{period}", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_residuals(x='tas', period='historical')
    plot_residuals(x='pr', period='historical')
    # plot_residuals(x='tasmin', period='historical')
    # plot_residuals(x='tasmax', period='historical')
    