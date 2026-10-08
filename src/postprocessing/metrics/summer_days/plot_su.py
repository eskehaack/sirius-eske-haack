import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import regionmask

def summer_days_plot(member="r1i1p1f1"):
    """
    3-row figure:
      Row 1: Absolute 30-year mean summer days (4 panels: mid/late x ssp126/ssp370)
      Row 2: Anomaly vs historical baseline (same 4 panels, diverging colormap)
      Row 3: Histogram of summer days per grid point (5 curves: historical + 4 combos)
    """

    VAR = "summer_days_index_per_time_period"

    # --- Load data ---
    ds126 = xr.open_dataset(f"/scratch/project_465002687/ec_earth/metrics/su_hclim_{member}_ssp126.nc")
    ds370 = xr.open_dataset(f"/scratch/project_465002687/ec_earth/metrics/su_hclim_{member}_ssp370.nc")

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
    abs_cmap = mpl.colormaps["YlOrRd"]
    bounds = [0, 10, 50, 100, 200, 300, 366]
    abs_norm = mpl.colors.BoundaryNorm(bounds, abs_cmap.N)
    abs_colorizer = mpl.colorizer.Colorizer(norm=abs_norm, cmap='YlOrRd')

    div_cmap = mpl.colormaps["magma_r"]
    bounds = [-10, 0, 10, 20, 30, 50, 80, 100]
    div_norm = mpl.colors.BoundaryNorm(bounds, div_cmap.N, extend='both')
    div_colorizer = mpl.colorizer.Colorizer(norm=div_norm, cmap='magma_r')

    # --- Layout ---
    titles  = ["Mid SSP126 (2020-2049)", "Late SSP126 (2070-2099)",
               "Mid SSP370 (2020-2049)", "Late SSP370 (2070-2099)"]
    datasets     = [mid_126,       late_126,       mid_370,       late_370      ]
    anom_datasets = [anom_mid_126, anom_late_126,  anom_mid_370,  anom_late_370 ]

    fig     = plt.figure(figsize=(6.4, 6.4/1.3))
    subfigs = fig.subfigures(2,1, height_ratios=[2.0, 1.0])

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
    def plot_map(ax, data, norm, cmap, title=None):
        im = ax.pcolormesh(lon, lat, data, transform=data_crs, cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5) 
        ax.gridlines()
        ax.set_aspect("auto")
        if title:
            ax.set_title(title)
        return im

    # --- Row 1: absolute means ---
    for col, (data, title) in enumerate(zip(datasets, titles)):
        im_abs = plot_map(axes_abs[col], data, abs_norm, abs_cmap, title)

    fig.colorbar(
        mpl.colorizer.ColorizingArtist(abs_colorizer),
        ax=axes_abs, orientation="vertical", fraction=0.02, pad=0.02,
        label="Summer days per year"
    )

    # Row labels
    axes_abs[0].set_ylabel("Absolute mean")

    # --- Row 2: anomalies ---
    for col, (data, title) in enumerate(zip(anom_datasets, titles)):
        im_div = plot_map(axes_anom[col], data, div_norm, div_cmap)

    fig.colorbar(
        mpl.colorizer.ColorizingArtist(div_colorizer),
        ax=axes_anom, orientation="vertical", fraction=0.02, pad=0.02,
        label="Δ Summer days vs historical"
    )

    axes_anom[0].set_ylabel("Anomaly vs historical")

    area_keys = ['NEU', 'CEU', 'MED']
    for i, area in enumerate(area_keys):
        mask = regionmask.defined_regions.srex.mask(ds126)
        data = ds126.where(mask.cf == area)
        hist_vals = (data[VAR].isel(time=0).values / 30).ravel()

        label = "Historical (1985-2014)"
        axes_hist[i].hist(hist_vals, bins=50, density=True, histtype="step", label=label, linewidth=0.5,)

        for dataset, scenario in zip([ds126, ds370], ["ssp126", "ssp370"]):
            data = dataset.where(mask.cf == area)
            for time_idx, title in zip([1, 2], ["Mid", "Late"]):
                hist_vals = (data[VAR].isel(time=time_idx).values / 30).ravel()
                label = f"{title} {scenario.upper()}"
                axes_hist[i].hist(hist_vals, bins=50, density=True, histtype="step", label=label, linewidth=0.5)

        axes_hist[i].set_xlabel("Summer days per year")
        axes_hist[i].set_ylabel("Log Density")
        axes_hist[i].set_title(f"Summer days across {area} region")
        axes_hist[i].set_yscale("log")
        axes_hist[i].set_xlim(left=0, right=366)

    handles, labels = axes_hist[0].get_legend_handles_labels()
    subfigs[1].legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        borderaxespad=0.5,
        ncols=5,
    )

    # --- Final touches ---
    plt.suptitle("Annual Summer Days (TASMAX > 25°C)")
    return fig

if __name__ == "__main__":
    fig = summer_days_plot()
    plt.savefig("./test-su.png")