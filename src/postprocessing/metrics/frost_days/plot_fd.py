import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import regionmask

def frost_days_plot(member="r1i1p1f1"):
    """
    3-row figure:
      Row 1: Absolute 30-year mean frost days (4 panels: mid/late x ssp126/ssp370)
      Row 2: Anomaly vs historical baseline (same 4 panels, diverging colormap)
      Row 3: Histogram of frost days per grid point (5 curves: historical + 4 combos)
    """

    VAR = "frost_days_index_per_time_period"

    # --- Load data ---
    ds126 = xr.open_dataset(f"/scratch/project_465002687/ec_earth/metrics/fd_hclim_{member}_ssp126.nc")
    ds370 = xr.open_dataset(f"/scratch/project_465002687/ec_earth/metrics/fd_hclim_{member}_ssp370.nc")

    lat = ds126.lat.values
    lon = ds126.lon.values

    # 30-year means (time indices: 0=historical, 1=mid, 2=late)
    hist     = ds126[VAR].isel(time=0).values / 30
    mid_126  = ds126[VAR].isel(time=1).values / 30
    late_126 = ds126[VAR].isel(time=2).values / 30
    mid_370  = ds370[VAR].isel(time=1).values / 30
    late_370 = ds370[VAR].isel(time=2).values / 30

    # Anomalies vs historical
    anom_mid_126  = mid_126  - hist
    anom_late_126 = late_126 - hist
    anom_mid_370  = mid_370  - hist
    anom_late_370 = late_370 - hist

    # --- Projection ---
    data_crs = ccrs.PlateCarree()
    rp = ds126['rotated_latitude_longitude'].attrs
    proj = ccrs.RotatedPole(
        pole_longitude=rp['grid_north_pole_longitude'],
        pole_latitude=rp['grid_north_pole_latitude'],
    )

    # --- Colormaps and norms ---
    abs_cmap = "Blues"
    abs_norm = mcolors.Normalize(vmin=0,   vmax=366)

    div_cmap = "RdYlBu"
    div_norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-200, vmax=200)

    # --- Layout ---
    titles  = ["Mid SSP126\n(2020-2049)", "Late SSP126\n(2070-2099)",
               "Mid SSP370\n(2020-2049)", "Late SSP370\n(2070-2099)"]
    datasets     = [mid_126,       late_126,       mid_370,       late_370      ]
    anom_datasets = [anom_mid_126, anom_late_126,  anom_mid_370,  anom_late_370 ]

    fig     = plt.figure(figsize=(18,14), layout="constrained")
    subfigs = fig.subfigures(2,1, height_ratios=[4.0, 1.5])

    # Row 1 & 2: 4 map panels each
    axes_abs  = []
    axes_anom = []
    for col in range(4):
        ax = subfigs[0].add_subplot(2, 4, col + 1, projection=proj)
        axes_abs.append(ax)
        ax = subfigs[0].add_subplot(2, 4, col + 5, projection=proj)
        axes_anom.append(ax)

    # Row 3: regional histograms
    axes_hist = []
    for i in range(3):
        ax = subfigs[1].add_subplot(1, 3, i+1)
        axes_hist.append(ax)

    # --- Helper: plot one map panel ---
    def plot_map(ax, data, norm, cmap, title):
        im = ax.pcolormesh(lon, lat, data, transform=data_crs, cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8) 
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=12)
        return im

    # --- Row 1: absolute means ---
    for col, (data, title) in enumerate(zip(datasets, titles)):
        im_abs = plot_map(axes_abs[col], data, abs_norm, abs_cmap, title)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=abs_norm, cmap=abs_cmap),
        ax=axes_abs, orientation="vertical", fraction=0.02, pad=0.04,
        label="Frost days per year"
    )

    # Row labels
    axes_abs[0].text(
        -0.12, 0.5, "Absolute mean", transform=axes_abs[0].transAxes,
        fontsize=12, va="center", rotation=90, fontweight="bold"
    )

    # --- Row 2: anomalies ---
    for col, (data, title) in enumerate(zip(anom_datasets, titles)):
        im_div = plot_map(axes_anom[col], data, div_norm, div_cmap, title)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=div_norm, cmap=div_cmap),
        ax=axes_anom, orientation="vertical", fraction=0.02, pad=0.04,
        label="Δ Frost days vs historical"
    )

    axes_anom[0].text(
        -0.12, 0.5, "Anomaly vs historical", transform=axes_anom[0].transAxes,
        fontsize=12, va="center", rotation=90, fontweight="bold"
    )

    area_keys = ['NEU', 'CEU', 'MED']
    for i, area in enumerate(area_keys):
        mask = regionmask.defined_regions.srex.mask(ds126)
        data = ds126.where(mask.cf == area)
        hist_vals = (data[VAR].isel(time=0).values / 30).ravel()

        label = "Historical (1985-2014)"
        axes_hist[i].hist(hist_vals, bins=50, density=True, histtype="step", label=label)

        for dataset, scenario in zip([ds126, ds370], ["ssp126", "ssp370"]):
            data = dataset.where(mask.cf == area)
            for time_idx, title in zip([1, 2], ["Mid", "Late"]):
                hist_vals = (data[VAR].isel(time=time_idx).values / 30).ravel()
                label = f"{title} {scenario.upper()}"
                axes_hist[i].hist(hist_vals, bins=50, density=True, histtype="step", label=label)

        axes_hist[i].set_xlabel("Frost days per year")
        axes_hist[i].set_ylabel("Log Density")
        axes_hist[i].set_title(f"Distribution of frost days across {area} region", fontsize=12)
        axes_hist[i].set_yscale("log")
        axes_hist[i].legend()
        axes_hist[i].set_xlim(left=0, right=366)

    # --- Final touches ---
    plt.suptitle("Annual Frost Days (TASMIN < 0°C)", fontsize=18, fontweight="bold")
    return fig

if __name__ == "__main__":
    fig = frost_days_plot()
    plt.savefig("./test.png")