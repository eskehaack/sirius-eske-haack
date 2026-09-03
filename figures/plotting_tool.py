from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

dpath = "/scratch/project_465002687/ec_earth/predictors/EC-Earth3-Veg/historical/r1i1p1f1"
HISTORICAL = Path(dpath)

FIGSIZE = (8, 6)
COLORS = ["cornflowerblue", "orangered", "violet"]

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

    ec_earth_path = HISTORICAL / "tas_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc"
    hclim_path = Path("/scratch/project_465002687/ec_earth/targets/HCLIM/EC-Earth3-Veg/historical/r1i1p1f1/day/tas/tas_EUR-12_EC-Earth3-Veg_historical_r1i1p1f1_HCLIMcom-SMHI_HCLIM43-ALADIN_v1-r1_day_19510101-19551231.nc")

    labels = ["EC-Earth Temperature Data", "HCLIM Temperature Data"]

    # Create globe
    fig = plt.figure(figsize=FIGSIZE)

    ax = plt.axes(
        projection=ccrs.Orthographic(10.45, 51.2)
    )

    ax.set_global()
    ax.coastlines()

    for i, data_path in enumerate([ec_earth_path, hclim_path]):
        if not data_path.exists():
            raise FileNotFoundError(f"File {data_path} does not exist.")
        
        data = xr.open_dataset(data_path)
        tas = data.tas

        # Get latitude and longitude coordinates
        lon = tas.lon.values
        lat = tas.lat.values
        if lon.ndim == 1:
            lon = np.tile(lon, (lat.shape[0], 1))
        if lat.ndim == 1:
            lat = np.tile(lat, (lon.shape[1], 1)).T

        lon_edges = np.empty((2*lon.shape[0]+2*lon.shape[1]))
        lat_edges = np.empty((2*lat.shape[0]+2*lat.shape[1]))

        for source, target in [(lat, lat_edges), (lon, lon_edges)]:
            idx = 0

            for j in [0,-1]:
                edge = source[j,:]
                start = idx
                idx += edge.shape[0]
                target[start:idx] = edge

                edge = source[:,j]
                start = idx
                idx += edge.shape[0]
                target[start:idx] = edge

        ax.scatter(lon_edges, lat_edges, s=1, color=COLORS[i], label=labels[i], transform=ccrs.PlateCarree())


    ax.legend()

    plt.title("Domain of Datasets")

    plt.savefig(
        "./figures/domains.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )

    plt.close()

if __name__ == "__main__":
    # precip_distribution()
    # plot_temp_hclim()
    plot_domain()
