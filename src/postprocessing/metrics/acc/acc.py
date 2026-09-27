import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

DIR = "/scratch/project_465002687/ec_earth/metrics"

def acc_plot(scenario='ssp126', member='r1i1p1f1'):
    """
    Function to plot the Anomaly Correlation Coefficient.
    """

    fig = plt.figure(figsize=(11.8,11.8/(1.618*2)), layout="constrained")
    data_crs = ccrs.PlateCarree()

    cmap = "gray"
    norm = mcolors.Normalize(vmin=0, vmax=1)

    def plot_map(ax, data, norm, cmap, title):
        im = ax.pcolormesh(lon, lat, data, transform=data_crs, cmap=cmap, norm=norm)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3) 
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.4, linestyle="--")
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=12)
        return im

    for i, x in enumerate(["tas", "tasmax", "tasmin", "pr"]):
        dsacc = xr.open_dataset(f"{DIR}/acc_{x}_{scenario}_{member}.nc")
        lon_str = "lon" if "lon" in dsacc.coords else "rlon"
        lat_str = "lat" if "lat" in dsacc.coords else "rlat"
        lon = dsacc[lon_str].values
        lat = dsacc[lat_str].values

        data = dsacc[x].values[0]

        rp = dsacc['rotated_latitude_longitude'].attrs
        proj = ccrs.RotatedPole(
            pole_longitude=rp['grid_north_pole_longitude'],
            pole_latitude=rp['grid_north_pole_latitude'],
        )

        ax = fig.add_subplot(1, 4, i + 1, projection=proj)

        im = plot_map(ax, data, norm, cmap, f"ACC - {x.upper()}")


    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, orientation="vertical", fraction=0.15, pad=0.04,
        label="Anomaly Correlation Coefficient"
    )

    fig.suptitle(f"Anomaly Correlation Coefficient - {scenario.upper()} - {member}", fontsize=14)

    return fig

if __name__ == "__main__":
    fig = acc_plot()
    plt.savefig(f"test.png", dpi=300)
    plt.close()