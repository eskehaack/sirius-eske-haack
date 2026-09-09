from os import name
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import xclim

dpath = "/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/historical/r1i1p1f1"
HISTORICAL = Path(dpath)
TRANSPARENT = False

FIGSIZE = (8, 6)
COLORS = ["cornflowerblue", "orangered", "violet", "black", "gold"]

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

class dataFile:
    def __init__(self, path, member, dataset, time_range, name):
        self.path = path
        self.member = member
        self.dataset = dataset
        self.time_range = time_range
        self.name = name

class biasData:
    def __init__(self, x='tas'):
        base_path = Path("/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2")
        historical_path = lambda i: base_path / "historical" / f"r{i}i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_historical_r{i}i1p1f1_r360x180_1951-2014.nc"
        ssp370_path = lambda i: base_path / "ssp370" / f"r{i}i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_ssp370_r{i}i1p1f1_r360x180_2015-2100.nc"
        self.n_members = 3

        self.files = [
            dataFile(
                path=historical_path,
                member=historical_path(0).parent.name,
                dataset=historical_path(0).parent.parent.name,
                time_range=slice("1985-01-01", "2014-12-31"),
                name="Historical (1985-2014)"
            ),
            dataFile(
                path=ssp370_path,
                member=ssp370_path(0).parent.name,
                dataset=ssp370_path(0).parent.parent.name,
                time_range=slice("2020-01-01", "2050-12-31"),
                name="Mid range (2020-2050)"
            ),
            dataFile(
                path=ssp370_path,
                member=ssp370_path(0).parent.name,
                dataset=ssp370_path(0).parent.parent.name,
                time_range=slice("2070-01-01", "2100-12-31"),
                name="Late range (2070-2100)"
            ),
        ]

    def _get_data(self, time_range=0):
        file = self.files[time_range]
        dataset = xr.open_dataset(file.path(1))
        for i in range(1, self.n_members):
            tmp = xr.open_dataset(file.path(i+1))
            dataset = xr.concat([dataset, tmp], dim="member")

        return dataset.sel(time=file.time_range)

def plot_yearly_max_days(x='tas', title="Distribution of Warmest Days", unit="Kelvin"):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """

    dataObj = biasData(x=x)
    fig = plt.figure(figsize=FIGSIZE)

    for i in range(3):
        data = dataObj._get_data(time_range=i)
        maxes = data.groupby("time.year").max(dim=["lat", "lon", "member"])

        plt.hist(maxes[x].values, bins=50, density=True, color=COLORS[i], alpha=0.5, label=f"{dataObj.files[i].name}")

    plt.grid(axis="y", alpha=0.2)
    plt.xlabel(f"{unit} [{x}]")
    plt.ylabel(f"Probability")
    plt.title(title)
    plt.legend()
    plt.savefig(f"./figures/data_section/climatology/max_{x}_days.png", dpi=300, transparent=TRANSPARENT, bbox_inches="tight")
    plt.close()

def plot_warmest_days():
    plot_yearly_max_days(x='tas', title="Distribution of Warmest Days on EC-Earth all members and ssp370", unit="Kelvin")

def plot_wettest_days():
    plot_yearly_max_days(x='pr', title="Distribution of Wettest Days on EC-Earth all members and ssp370", unit="kg/m²/s")

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

def plot_residuals(x='tas'):
    """
    Plot the ressiduals between coarse (regridded) data and high res data 
    """

    regrided_path = Path(f"/scratch/project_465002687/ec_earth/predictors/regridded/historical/r1i1p1f1/{x}_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc")
    hclim_path = Path(f"/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/r1i1p1f1/day/{x}/{x}_EUR-12_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_19510101-19551231.nc")

    regridded = xr.open_dataset(regrided_path).isel(time=0)[x]
    hclim = xr.open_dataset(hclim_path).isel(time=0)[x]

    residual = regridded - hclim

    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0]*2, FIGSIZE[1]), width_ratios=[1, 1, 1], height_ratios=[1, 1])

    # --- Regridded ---
    im0 = axes[0, 0].imshow(regridded.values, cmap="coolwarm", origin="lower", aspect="auto")
    axes[0, 0].set_title(f"Regridded EC-Earth {x}")
    plt.colorbar(im0, ax=axes[0, 0], orientation="vertical", fraction=0.046, pad=0.04)

    axes[1, 0].hist(regridded.values.flatten(), bins=50, color=COLORS[0], alpha=0.5, density=True)
    axes[1, 0].set_title(f"Distribution - Regridded EC-Earth {x}")
    axes[1, 0].set_xlabel(f"{x} [{regridded.units}]")
    axes[1, 0].set_ylabel("Probability")

    # --- HCLIM ---
    im1 = axes[0, 1].imshow(hclim.values, cmap="coolwarm", origin="lower", aspect="auto")
    axes[0, 1].set_title(f"HCLIM {x}")
    plt.colorbar(im1, ax=axes[0, 1], orientation="vertical", fraction=0.046, pad=0.04)

    axes[1, 1].hist(hclim.values.flatten(), bins=50, color=COLORS[1], alpha=0.5, density=True)
    axes[1, 1].set_title(f"Distribution - HCLIM {x}")
    axes[1, 1].set_xlabel(f"{x} [{hclim.units}]")
    axes[1, 1].set_ylabel("Probability")

    # --- Residuals ---
    im2 = axes[0, 2].imshow(residual.values, cmap="coolwarm", origin="lower", aspect="auto")
    axes[0, 2].set_title(f"Residuals (EC-Earth - HCLIM) {x}")
    plt.colorbar(im2, ax=axes[0, 2], orientation="vertical", fraction=0.046, pad=0.04)

    axes[1, 2].hist(residual.values.flatten(), bins=50, color=COLORS[2], alpha=0.5, density=True)
    axes[1, 2].set_title(f"Distribution - Residuals {x}")
    axes[1, 2].set_xlabel(f"Residuals [{residual.units}]")
    axes[1, 2].set_ylabel("Probability")

    plt.tight_layout()
    plt.savefig(f"./figures/method/residuals_{x}.png", dpi=300, transparent=TRANSPARENT, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # precip_distribution()
    # plot_temp_hclim()
    # plot_timeseries()
    # plot_domain()
    # plot_warmest_days()
    # plot_wettest_days()
    # plot_hwfi_days(x="tasmax")
    plot_residuals()

    pass