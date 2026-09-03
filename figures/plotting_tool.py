from pathlib import Path

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

def temp_domain():
    """
    Shows the domain of the temperature data
    """
    data_path = HISTORICAL / "tas_EUR-12_day_EC-Earth3-Veg_historical_r1i1p1f1_r360x180_1951-2014.nc"
    data = xr.open_dataset(data_path)
    tas = data.tas

    plt.figure(figsize=FIGSIZE)
    p = tas.isel(time=0).plot(
        subplot_kws=dict(projection=ccrs.Orthographic(10.45, 51.2), facecolor="gray"),
        transform=ccrs.PlateCarree()
    )
    p.axes.set_global()
    p.axes.coastlines()
    plt.title("Domain of Temperature Data for historical EC Earth data")
    plt.savefig("./figures/temp_domain.png", dpi=300, transparent=True)
    plt.close()


if __name__ == "__main__":
    precip_distribution()
    temp_domain()
