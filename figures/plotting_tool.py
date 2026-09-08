from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

dpath = "/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2/historical/r1i1p1f1"
HISTORICAL = Path(dpath)

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
    plt.savefig("./figures/precip_distribution.png", dpi=300, transparent=True)
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
    plt.savefig("./figures/hclim_tas.png", dpi=300, transparent=True)
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
        "./figures/domains.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

    plt.close()

def plot_timeseries(x='tas'):
    """
    Plots a time series of variable X for historical+scenario EC Earth data.
    """

    base_path = Path("/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg-v2")
    historical_path = base_path / "historical" / "r1i1p1f1/tas_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc"
    ssp126_path = base_path / "ssp126" / "r1i1p1f1/tas_EUR-12_day_EC-Earth3-Veg_ssp126_r1i1p1f1_r360x180_2015-2100.nc"
    ssp370_path = base_path / "ssp370" / "r1i1p1f1/tas_EUR-12_day_EC-Earth3-Veg_ssp370_r1i1p1f1_r360x180_2015-2100.nc"

    paths = [historical_path, ssp126_path, ssp370_path]

    fig = plt.figure(figsize=FIGSIZE)

    running_mean_window = 365 # days

    for i, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        name = path.parent.parent.name

        with xr.open_dataset(path) as data:
            if "historical" in name:
                data = data.sel(time=slice("1951-01-01", "2014-12-31"))
            elif "ssp" in name:
                data = data.sel(time=slice("2015-01-01", "2100-12-31"))

            var = data[x]
            var_mean = var.mean(dim=["lat", "lon"])
            var_std = var.std(dim=["lat", "lon"])
            time = data.time.values

            plt.plot(time, var_mean, color=COLORS[i], label=name)
            plt.fill_between(time, var_mean - var_std, var_mean + var_std, alpha=0.2, color=COLORS[i])

            # run_mean = var_mean.rolling(time=running_mean_window, center=True).mean()
            # plt.plot(run_mean.time, run_mean, color=COLORS[i], linestyle="--", label=f"{running_mean_window}-day Running Mean")

    plt.xlabel("Time")
    plt.ylabel(f"{x} (mean)")
    plt.title(f"Time Series of {x}")
    plt.legend()
    plt.savefig(f"./figures/timeseries_{x}.png", dpi=300, transparent=False, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # precip_distribution()
    # plot_temp_hclim()
    plot_timeseries()
    # plot_domain()
