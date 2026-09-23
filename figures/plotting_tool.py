from os import name
from pathlib import Path

import xarray as xr
import xclim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from figures import plt_guide as pg

dpath = "/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/historical/r1i1p1f1"
HISTORICAL = Path(dpath)
TRANSPARENT = False

FIGSIZE = (12, 6)
COLORS = [
    "#2E6FD9",
    "#E8821A",
    "#2E9E6B",
    "#D94040",
    "#7B52D9",
    "#1FA8A8",
    "#888888",
    "#CCCCCC",
    "#1A1A2E",
    "#FFFFFF",
]

def precip_distribution():
    """
    Shows the distribution of precipitation amount on days
    log/reg histogram
    """
    data_path = HISTORICAL / "pr_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc"
    data = xr.open_dataset(data_path)
    pr = data.pr.values.flatten()

    plt.figure(figsize=FIGSIZE)
    plt.hist(pr, bins=50, log=True, color=COLORS[0])
    plt.ylabel("Number of Days [Log Scale]")
    plt.xlabel("Precipitation [kg/m²/day]")
    plt.title("Distribution of Precipitation Amount on Days for historical EC Earth data")
    plt.savefig("./figures/precip_distribution.png", dpi=300, transparent=TRANSPARENT)
    plt.close()


def plot_temp_hclim():
    """
    Shows the domain of the temperature data
    """
    data_path = Path("/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/r1i1p1f1/day/tas/tas_EUR-12_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_19510101-19551231.nc")
    data = xr.open_dataset(data_path)
    tas = data.tas

    plt.figure(figsize=FIGSIZE)
    p = tas.isel(time=0).plot(
        x="lon", y="lat", cmap="coolwarm", cbar_kwargs={"label": "Temperature [K]"},
        subplot_kws=dict(projection=ccrs.Orthographic(10.45, 51.2), facecolor="gray"),
        transform=ccrs.PlateCarree()
    )
    p.axes.set_global()
    p.axes.coastlines()
    plt.title("Domain of Temperature Data for historical EC Earth data")
    plt.savefig("./figures/data_section/hclim_tas.png", dpi=300, transparent=TRANSPARENT)
    plt.close()


def plot_domain():
    """
    Shows the domain of the two datasets.
    """

    ec_earth_path = (
        HISTORICAL
        / "tas_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc"
    )

    hclim_path = Path(
        "/scratch/project_465002687/ec_earth/targets/HCLIM/"
        "EC-Earth3-Veg/historical/r1i1p1f1/day/tas/"
        "tas_EUR-12_EC-Earth3-Veg_historical_r1i1p1f1_"
        "HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_"
        "19510101-19551231.nc"
    )

    paths = [ec_earth_path, hclim_path]
    labels = ["EC-Earth", "HCLIM"]

    # ---------------------------------------------------------
    # Create globe
    # ---------------------------------------------------------

    fig = plt.figure(figsize=FIGSIZE)

    ax = plt.axes(
        projection=ccrs.Orthographic(10.45, 51.2)
    )

    ax.set_global()

    # Coastlines
    ax.coastlines(
        linewidth=0.8,
        color="gray",
    )

    # ---------------------------------------------------------
    # Plot domain boundaries
    # ---------------------------------------------------------

    for i, data_path in enumerate(paths):

        if not data_path.exists():
            raise FileNotFoundError(
                f"File {data_path} does not exist."
            )

        with xr.open_dataset(data_path) as data:

            tas = data.tas

            lon = tas.lon.values
            lat = tas.lat.values

            # Convert 1D coordinates to 2D
            if lon.ndim == 1:
                lon = np.tile(lon, (lat.shape[0], 1))

            if lat.ndim == 1:
                lat = np.tile(lat, (lon.shape[1], 1)).T

            # -------------------------------------------------
            # Construct boundary in a continuous order
            #
            #        top:  ---> 
            #              |
            #              |
            #        bottom: <---
            # -------------------------------------------------

            top_lon = lon[0, :]
            top_lat = lat[0, :]

            right_lon = lon[:, -1]
            right_lat = lat[:, -1]

            bottom_lon = lon[-1, ::-1]
            bottom_lat = lat[-1, ::-1]

            left_lon = lon[::-1, 0]
            left_lat = lat[::-1, 0]

            # Join all four edges
            boundary_lon = np.concatenate([
                top_lon,
                right_lon,
                bottom_lon,
                left_lon,
            ])

            boundary_lat = np.concatenate([
                top_lat,
                right_lat,
                bottom_lat,
                left_lat,
            ])

            # Plot continuous boundary
            ax.plot(
                boundary_lon,
                boundary_lat,
                transform=ccrs.PlateCarree(),
                color=COLORS[i],
                linewidth=2.5,
                label=labels[i],
            )

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    plt.title(
        "Model Domains",
        fontsize=16,
        pad=15,
    )

    plt.savefig(
        "./figures/data_section/domains.png",
        dpi=300,
        transparent=TRANSPARENT,
        bbox_inches="tight",
    )

    plt.close()

def plot_timeseries(x='tas'):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """

    fig = plt.figure(figsize=FIGSIZE)
    running_mean_window = 3 * 365 # days
    transparency = [1, 0.7, 1]

    for i in range(3):
        realization = f"r{i+1}i1p1f1"
        base_path = Path("/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2")
        historical_path = base_path / "historical" / realization / f"{x}_EUR-12_day_EC-Earth3-Veg_historical_{realization}_r360x180_1951-2014.nc"
        ssp126_path = base_path / "ssp126" / realization / f"{x}_EUR-12_day_EC-Earth3-Veg_ssp126_{realization}_r360x180_2015-2100.nc"
        ssp370_path = base_path / "ssp370" / realization / f"{x}_EUR-12_day_EC-Earth3-Veg_ssp370_{realization}_r360x180_2015-2100.nc"

        paths = [historical_path, ssp126_path, ssp370_path]

        for j, path in enumerate(paths):
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist.")

            name = path.parent.parent.name
            member = path.parent.name

            label = f"{member}-{x}-{running_mean_window} day running mean" if j == 0 else None

            with xr.open_dataset(path) as data:

                data = data.sel(time=slice("1950-01-01", "2100-12-31"))

                weights = np.cos(np.deg2rad(data.lat))
                weights.name = "weights"

                var = data[x]
                var = var.mean(dim=['lon'])

                if var.shape[0] == 0:
                    print(f"No data available for {name}-{member} in the specified time range.")
                    continue

                size_weighted = var.weighted(weights)
                var_mean = size_weighted.mean(dim=["lat"])

                run_mean = var_mean.rolling(time=running_mean_window, center=True).mean()
                plt.plot(run_mean.time, run_mean, color=COLORS[i], label=label, alpha=transparency[j])

    xpoint = data.time[0] if not name == "historical" else data.time[-1] 
    plt.vlines(x=xpoint, ymin=282.5, ymax=292.5, color="black", linestyle="--", alpha=0.2, label="End of Historical Period")
    plt.grid(axis="y", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel(f"{x} (Kelvin)")
    plt.title(f"Time Series of {x} on EC-Earth data - all members and scenarios")
    plt.legend()
    plt.savefig(f"./figures/data_section/climatology/timeseries_{x}.png", dpi=300, transparent=TRANSPARENT, bbox_inches="tight")
    plt.close()

def plot_timeseries_merged(x='tas', running_mean_window=11, base_path="/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/merged"):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """
    base_path = Path(base_path)

    transparency = [0.7, 1.0]

    total_min = np.inf; total_max = -np.inf

    fig = plt.figure()

    for member in range(1, 4):


        for j, scenario in enumerate(["ssp126", "ssp370"]):
            data_path = base_path / f"{x}_historical_{scenario}_r{member}i1p1f1_yearly.nc"
            data = xr.open_dataset(data_path)

            weights = np.cos(np.deg2rad(data.lat))
            weights.name = "weights"

            var = data[x]
            var = var.mean(dim=['lon'])

            if var.shape[0] == 0:
                print(f"No data available for {name}-{member} in the specified time range.")
                continue

            size_weighted = var.weighted(weights)
            var_mean = size_weighted.mean(dim=["lat"])
            run_mean = var_mean.rolling(time=running_mean_window, center=True).mean()

            label = f"r{member}i1p1f1-{x}-{scenario}"
            plt.plot(run_mean.time, run_mean, color=COLORS[member-1], label=label, alpha=transparency[j])

            total_min = min(total_min, run_mean.min().values)
            total_max = max(total_max, run_mean.max().values)

    plt.vlines(x=np.datetime64("2014-12-31"), ymin=total_min, ymax=total_max, color="black", linestyle="--", alpha=0.2, label="End of Historical Period")
    plt.grid(axis="y", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel(f"{x} ({var.attrs['units']})")
    plt.title(f"Time Series of {x} on EC-Earth data")
    plt.legend()
    pg.save(fig, f"./figures/data_section/climatology/timeseries_{x}_{running_mean_window}.png")
    plt.close()

def plot_timeseries_merged_hclim(x='tas', running_mean_window=11, base_path="/scratch/project_465002687/ec_earth/targets/HCLIM/merged"):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """
    base_path = Path(base_path)

    transparency = [0.7, 1.0]

    total_min = np.inf; total_max = -np.inf

    fig = plt.figure()

    for member in range(1, 4):

        for j, scenario in enumerate(["ssp126", "ssp370"]):
            data_path = base_path / f"{x}_historical_{scenario}_r{member}i1p1f1_yearly.nc"
            data = xr.open_dataset(data_path)

            weights = np.cos(np.deg2rad(data.rlat))
            weights.name = "weights"

            var = data[x]
            var = var.mean(dim=['rlon'])

            if var.shape[0] == 0:
                print(f"No data available for {name}-{member} in the specified time range.")
                continue

            size_weighted = var.weighted(weights)
            var_mean = size_weighted.mean(dim=["rlat"])
            run_mean = var_mean.rolling(time=running_mean_window, center=True).mean()

            label = f"r{member}i1p1f1-{x}-{scenario}"
            plt.plot(run_mean.time, run_mean, color=COLORS[member-1], label=label, alpha=transparency[j])

            total_min = min(total_min, run_mean.min().values)
            total_max = max(total_max, run_mean.max().values)

    plt.vlines(x=np.datetime64("2014-12-31"), ymin=total_min, ymax=total_max, color="black", linestyle="--", alpha=0.2, label="End of Historical Period")
    plt.grid(axis="y", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel(f"{x} ({var.attrs['units']})")
    plt.title(f"Time Series of {x} on HCLIM data")
    plt.legend()
    pg.save(fig, f"./figures/data_section/climatology/timeseries_{x}_{running_mean_window}_hclim.png")
    plt.close()
    

class dataFile:
    def __init__(self, path: list, dataset: str, time_range: slice, name: str, data:xr.Dataset=None):
        self.path = path
        self.dataset = dataset
        self.time_range = time_range
        self.name = name
        self.data = data

class biasData:
    def __init__(self, x='tas', source='EC-Earth'):
        if source == 'EC-Earth':
            base_path = Path("/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2")
            historical_path = lambda i: [base_path / "historical" / f"r{i}i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_historical_r{i}i1p1f1_r360x180_1951-2014.nc"]
            ssp370_path = lambda i: [base_path / "ssp370" / f"r{i}i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_ssp370_r{i}i1p1f1_r360x180_2015-2100.nc"]
        elif source == 'HCLIM':
            base_path = Path("/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg")
            historical_path = lambda i: (base_path / "historical" / f"r{i}i1p1f1/day/{x}/").glob('*.nc')
            ssp370_path = lambda i: (base_path / "ssp370" / f"r{i}i1p1f1/day/{x}/").glob('*.nc')
        else:
            raise ValueError(f"Source {source} not supported.")

        self.n_members = 3

        self.files = [
            dataFile(
                path=historical_path,
                dataset='historical',
                time_range=slice("1985-01-01", "2014-12-31"),
                name="Historical (1985-2014)"
            ),
            dataFile(
                path=ssp370_path,
                dataset='ssp370',
                time_range=slice("2020-01-01", "2050-12-31"),
                name="Mid range (2020-2050)"
            ),
            dataFile(
                path=ssp370_path,
                dataset='ssp370',
                time_range=slice("2070-01-01", "2100-12-31"),
                name="Late range (2070-2100)"
            ),
        ]

    def _get_data(self, time_range=0):
        file = self.files[time_range]
        if file.data is not None:
            return file.data
        
        dataset = xr.open_mfdataset(file.path(1), data_vars='minimal', coords='minimal', compat='override')
        for i in range(1, self.n_members):
            tmp = xr.open_mfdataset(file.path(i+1), data_vars='minimal', coords='minimal', compat='override')
            dataset = xr.concat([dataset, tmp], dim="member")

        sliced = dataset.sel(time=file.time_range)
        file.data = sliced # save for later
        return sliced

def plot_yearly_max_days(x='tas', title="Distribution of Warmest Days", unit="Kelvin", data_source='EC-Earth'):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """

    dataObj = biasData(x=x, source=data_source)
    fig = plt.figure(figsize=FIGSIZE)

    for i in range(3):
        data = dataObj._get_data(time_range=i)
        dims = list(set(data.dims) - set(["time"]))
        maxes = data.groupby("time.year").max(dim=dims)

        plt.hist(maxes[x].values, bins=50, density=True, alpha=0.5, label=f"{dataObj.files[i].name}")

    plt.grid(axis="y", alpha=0.2)
    plt.xlabel(f"{unit} [{x}]")
    plt.ylabel(f"Probability")
    plt.title(title)
    plt.legend()
    pg.save(fig, f"./figures/data_section/climatology/max_{x}_days_{data_source}.png")
    plt.close()

def plot_warmest_days():
    plot_yearly_max_days(x='tas', title="Distribution of Warmest Days on EC-Earth all members and ssp370", unit="Kelvin")

def plot_wettest_days():
    plot_yearly_max_days(x='pr', title="Distribution of Wettest Days on EC-Earth all members and ssp370", unit="kg/m²/s")

def plot_warmest_days_hclim():
    plot_yearly_max_days(x='tas', title="Distribution of Warmest Days on HCLIM all members and ssp370", unit="Kelvin", data_source='HCLIM')

def plot_wettest_days_hclim():
    plot_yearly_max_days(x='pr', title="Distribution of Wettest Days on HCLIM all members and ssp370", unit="kg/m²/s", data_source='HCLIM')

def plot_hwfi_days(x='tas', title="Warm Spell Duration Index on EC-Earth all members and ssp370", unit="Days per Year"):
    """
    Plots the Warm Spell Duration Index (WSDI) for historical+scenario EC Earth data.
    The 90th-percentile threshold is always derived from the reference period (time_range=0).
    """
    dataObj = biasData(x)
    fig = plt.figure(figsize=FIGSIZE)

    # Compute threshold once from the reference period so all three periods
    # are evaluated against the same baseline climatology.
    ref_data = dataObj._get_data(time_range=0)
    per = xclim.core.calendar.percentile_doy(
        ref_data[x], per=90, window=5,
    )

    for i in range(3):
        data = dataObj._get_data(time_range=i)

        warm_spell_days = xclim.indicators.atmos.warm_spell_duration_index(
            tasmax=data[x],
            tasmax_per=per,
            window=6,
            freq="YS",
        )

        # Spatially average to get one value per year (swap .mean for .max/.sum if preferred)
        annual = warm_spell_days.mean(dim=["lat", "lon", "member"])

        plt.hist(
            annual.values,
            bins=50,
            range=(0, 365),
            density=True,
            color=COLORS[i],
            label=dataObj.files[i].name,
            alpha=0.5,
        )

    plt.grid(axis="y", alpha=0.2)
    plt.xlabel(f"Warm spell days per year [{x}]")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.savefig("./figures/data_section/climatology/warm_spell_days.png", dpi=300, transparent=TRANSPARENT, bbox_inches="tight")
    plt.close()

def summer_days_plot():
    """
    3-row figure:
      Row 1: Absolute 30-year mean summer days (4 panels: mid/late x ssp126/ssp370)
      Row 2: Anomaly vs historical baseline (same 4 panels, diverging colormap)
      Row 3: KDE of summer days per grid point (5 curves: historical + 4 combos)
    """

    VAR = "summer_days_index_per_time_period"

    # --- Load data ---
    ds126 = xr.open_dataset("/scratch/project_465002687/ec_earth/metrics/su_hclim_r1i1p1f1_ssp126.nc")
    ds370 = xr.open_dataset("/scratch/project_465002687/ec_earth/metrics/su_hclim_r1i1p1f1_ssp370.nc")

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
    lon_0    = float(ds126.lon.mean())
    lat_0    = float(ds126.lat.mean())
    data_crs = ccrs.PlateCarree()
    proj = ccrs.LambertConformal(
        central_longitude=lon_0,
        central_latitude=lat_0,
        standard_parallels=(35, 65)  # standard for European domains
    )

    # --- Colormaps and norms ---
    abs_cmap = "YlOrRd"
    abs_norm = mcolors.Normalize(vmin=0,   vmax=np.nanmax([mid_126, late_126, mid_370, late_370]))

    anom_max = np.nanmax(np.abs([anom_mid_126, anom_late_126, anom_mid_370, anom_late_370]))
    div_cmap = "YlGnBu"
    div_norm = mcolors.Normalize(vmin=0, vmax=anom_max)

    # --- Layout ---
    fig     = plt.figure(figsize=(18, 14))
    titles  = ["Mid SSP126\n(2020-2049)", "Late SSP126\n(2070-2099)",
               "Mid SSP370\n(2020-2049)", "Late SSP370\n(2070-2099)"]
    datasets     = [mid_126,       late_126,       mid_370,       late_370      ]
    anom_datasets = [anom_mid_126, anom_late_126,  anom_mid_370,  anom_late_370 ]

    # Row 1 & 2: 4 map panels each
    axes_abs  = []
    axes_anom = []
    for col in range(4):
        ax = fig.add_subplot(3, 4, col + 1, projection=proj)
        axes_abs.append(ax)
        ax = fig.add_subplot(3, 4, col + 5, projection=proj)
        axes_anom.append(ax)

    # Row 3: single wide hist panel
    ax_hist = fig.add_subplot(3, 1, 3)

    # --- Helper: plot one map panel ---
    def plot_map(ax, data, norm, cmap, title):
        im = ax.pcolormesh(lon, lat, data, transform=data_crs, cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=":")
        ax.add_feature(cfeature.LAND,      facecolor="lightgrey", zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="aliceblue", zorder=0)
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=9)
        return im

    # --- Row 1: absolute means ---
    for col, (data, title) in enumerate(zip(datasets, titles)):
        im_abs = plot_map(axes_abs[col], data, abs_norm, abs_cmap, title)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=abs_norm, cmap=abs_cmap),
        ax=axes_abs, orientation="vertical", fraction=0.02, pad=0.04,
        label="Summer days per year"
    )

    # Row labels
    axes_abs[0].text(
        -0.12, 0.5, "Absolute mean", transform=axes_abs[0].transAxes,
        fontsize=10, va="center", rotation=90, fontweight="bold"
    )

    # --- Row 2: anomalies ---
    for col, (data, title) in enumerate(zip(anom_datasets, titles)):
        im_div = plot_map(axes_anom[col], data, div_norm, div_cmap, title)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=div_norm, cmap=div_cmap),
        ax=axes_anom, orientation="vertical", fraction=0.02, pad=0.04,
        label="Δ Summer days vs historical"
    )

    axes_anom[0].text(
        -0.12, 0.5, "Anomaly vs historical", transform=axes_anom[0].transAxes,
        fontsize=10, va="center", rotation=90, fontweight="bold"
    )

    # --- Row 3: hist ---

    hist_data = [
        (hist, "Historical (1985-2014)"),
        (mid_126, "Mid SSP126 (2020-2049)"),
        (late_126, "Late SSP126 (2070-2099)"),
        (mid_370, "Mid SSP370 (2020-2049)"),
        (late_370, "Late SSP370 (2070-2099)")
    ]
    for data, label in hist_data:
        flat = data.flatten()
        ax_hist.hist(flat, bins=50, density=True, histtype="step", label=label)

    ax_hist.set_xlabel("Summer days per year", fontsize=10)
    ax_hist.set_ylabel("Log Frequency", fontsize=10)
    ax_hist.set_title("Distribution of summer days across grid points", fontsize=10)
    ax_hist.set_yscale("log")
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(left=0)

    # --- Final touches ---
    plt.suptitle("Annual Summer Days (TASMAX > 25°C)", fontsize=13, fontweight="bold", y=0.95)
    pg.save(fig, "./figures/data_section/climatology/summer_days.png")


def plot_days_above_threshold_hclim(x='tasmax', threshold=25, title="Days Above Threshold on EC-Earth all members and ssp370", unit="Days per Year"):
    """
    Plots the yearly number of days above a given threshold for historical+scenario EC Earth data.
    """
    dataObj = biasData(x)
    fig = plt.figure(figsize=FIGSIZE)

    threshold += 273.15 if "tas" in x else 0  # Convert to Kelvin if temperature

    for i in range(3):
        data = dataObj._get_data(time_range=i)

        days_above_threshold = (data[x] > threshold).sum(dim=["time", "lat", "lon", "member"])

        plt.hist(
            days_above_threshold.values,
            bins=50,
            density=True,
            color=COLORS[i],
            label=dataObj.files[i].name,
            alpha=0.5,
        )

    plt.grid(axis="y", alpha=0.2)
    plt.xlabel(f"Days above {threshold} [{x}]")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    pg.save(fig, "./figures/data_section/climatology/days_above_25.png")
    plt.close()

def plot_residuals(x='tas'):
    regridded_path = Path(f"/scratch/project_465002687/ec_earth/predictors/regridded/historical/r1i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc")
    hclim_path = Path(f"/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/r1i1p1f1/day/{x}/{x}_EUR-12_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_19510101-19551231.nc")

    ds_reg = xr.open_dataset(regridded_path).isel(time=0)
    ds_hcl = xr.open_dataset(hclim_path).isel(time=0)

    regridded = ds_reg[x]
    hclim     = ds_hcl[x]
    residual  = regridded - hclim

    rp = ds_hcl['rotated_latitude_longitude'].attrs
    proj = ccrs.RotatedPole(
        pole_longitude=rp['grid_north_pole_longitude'],
        pole_latitude=rp['grid_north_pole_latitude'],
    )

    # Shared colour range for the two temperature maps; symmetric range for residuals
    vmin = min(float(regridded.min()), float(hclim.min()))
    vmax = max(float(regridded.max()), float(hclim.max()))
    res_abs = float(abs(residual).max())

    fig = plt.figure(figsize=(FIGSIZE[0]*1.5, FIGSIZE[1]*1.5), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    map_axes  = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]
    hist_axes = [fig.add_subplot(gs[1, i])                  for i in range(3)]

    def plot_map(ax, data, title, cmap, vmin, vmax):
        im = ax.pcolormesh(
            ds_hcl['rlon'], ds_hcl['rlat'], data.values,
            cmap=cmap, transform=proj, vmin=vmin, vmax=vmax,
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="k")
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")
        ax.set_title(title, fontsize=10, pad=5)
        return im

    im0 = plot_map(map_axes[0], regridded, f"EC-Earth (regridded)",       "RdBu_r", vmin, vmax)
    im1 = plot_map(map_axes[1], hclim,     f"HCLIM",                      "RdBu_r", vmin, vmax)
    im2 = plot_map(map_axes[2], residual,  f"Residual (EC-Earth - HCLIM)", "RdBu_r", -res_abs, res_abs)

    # One shared colorbar for the two temperature maps
    cb_temp = fig.colorbar(im1, ax=map_axes[:2], orientation="vertical",
                           fraction=0.018, pad=0.02, shrink=0.85)
    cb_temp.set_label(f"{x} [{regridded.attrs.get('units', '')}]", fontsize=9)
    cb_temp.ax.tick_params(labelsize=8)

    # Separate colorbar for residuals
    cb_res = fig.colorbar(im2, ax=map_axes[2], orientation="vertical",
                          fraction=0.046, pad=0.04, shrink=0.85)
    cb_res.set_label(f"Δ{x} [{regridded.attrs.get('units', '')}]", fontsize=9)
    cb_res.ax.tick_params(labelsize=8)

    def plot_hist(ax, data, xlabel):
        vals = data.values.flatten()
        ax.hist(vals, bins=60, density=True, alpha=0.85, edgecolor="black")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)

    plot_hist(hist_axes[0], regridded, f"{x} [{regridded.attrs.get('units', '')}]")
    plot_hist(hist_axes[1], hclim,     f"{x} [{hclim.attrs.get('units', '')}]")
    plot_hist(hist_axes[2], residual,  f"Δ{x} [{regridded.attrs.get('units', '')}]")

    fig.suptitle(f"EC-Earth vs. HCLIM — {x}", fontsize=13, fontweight="bold")
    pg.save(plt, f"./figures/method/residuals_{x}_2")
    plt.close()

if __name__ == "__main__":
    
    pg.setup()

    # precip_distribution()
    # plot_temp_hclim()
    # plot_timeseries(x='pr')
    # plot_timeseries_merged(x='pr', running_mean_window=11)
    # plot_timeseries_merged_hclim(x='pr', running_mean_window=11)
    summer_days_plot()
    # plot_domain()
    # plot_warmest_days()
    # plot_wettest_days()
    # plot_warmest_days_hclim()
    # plot_wettest_days_hclim()
    # plot_hwfi_days(x="tasmax")
    # plot_residuals(x='tas')

    pass