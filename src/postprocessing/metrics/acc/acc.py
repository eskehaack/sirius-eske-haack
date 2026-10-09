import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
from scipy import stats
import numpy as np

DIR = "/scratch/project_465002687/ec_earth/metrics"

def acc_plot(scenario='ssp126', member='r1i1p1f1'):
    """
    Function to plot the Anomaly Correlation Coefficient.
    """

    fig = plt.figure(figsize=(6.4, 6.4/(1.618*2)))
    data_crs = ccrs.PlateCarree()

    cmap = mpl.colormaps["Reds"]
    bounds = [-1, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colorizer = mpl.colorizer.Colorizer(norm=norm, cmap='Reds')

    def plot_map(ax, data, norm, cmap, title):
        im = ax.pcolormesh(lon, lat, data, transform=data_crs, cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5) 
        ax.gridlines(linewidth=0.5, color="grey", alpha=0.4, linestyle="--")
        ax.set_aspect("auto")
        ax.set_title(title)
        return im

    for i, x in enumerate(["tas", "tasmin", "tasmax", "pr"]):
        dsacc = xr.open_dataset(f"{DIR}/acc_{x}_{scenario}_{member}.nc")
        lon_str = "lon" if "lon" in dsacc.coords else "rlon"
        lat_str = "lat" if "lat" in dsacc.coords else "rlat"
        lon = dsacc[lon_str].values
        lat = dsacc[lat_str].values

        data = dsacc[x].values[0]

        n = 54_787

        r = data
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
        p_values = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

        rp = dsacc['rotated_latitude_longitude'].attrs
        proj = ccrs.RotatedPole(
            pole_longitude=rp['grid_north_pole_longitude'],
            pole_latitude=rp['grid_north_pole_latitude'],
        )

        ax = fig.add_subplot(1, 4, i + 1, projection=proj)

        im = plot_map(ax, data, norm, cmap, f"ACC - {x.upper()}")


    fig.colorbar(
        mpl.colorizer.ColorizingArtist(colorizer),
        ax=ax, orientation="vertical", fraction=0.15, pad=0.04,
        label="Anomaly Correlation Coefficient"
    )

    fig.suptitle(f"Anomaly Correlation Coefficient - {scenario.upper()} - {member}")

    return fig

if __name__ == "__main__":
    fig = acc_plot(scenario='ssp370')
    plt.savefig(f"test-acc.png", dpi=300)
    plt.close()