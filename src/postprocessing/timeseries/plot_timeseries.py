from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import src.postprocessing.plt_guide as pg

def plot_timeseries_merged(x='tas', running_mean_window=11, scenario='ssp126'):
    """
    Plots a time series of variable X for historical+scenario.
    """
    pg.setup()
    regridded_path = Path("/scratch/project_465002687/ec_earth/predictors/regridded/merged")
    hclim_path = Path("/scratch/project_465002687/ec_earth/targets/HCLIM/merged")

    transparency = 0.7 if scenario == "ssp126" else 1.0
    lims = (281, 290) if x == "tas" else (1.85e-5, 2.4e-5)

    fig = plt.figure(figsize=pg.FIG_HALF)

    for member in range(1, 4):

        regridded_file = regridded_path / f"{x}_historical_{scenario}_r{member}i1p1f1_yearly.nc"
        hclim_file = hclim_path / f"{x}_historical_{scenario}_r{member}i1p1f1_yearly.nc"

        for name, file in [("EC-Earth", regridded_file), ("HCLIM", hclim_file)]:
            data = xr.open_dataset(file)

            var = data[x]
            var_mean = var.mean(dim=['rlon', 'rlat'])
            run_mean = var_mean.rolling(time=running_mean_window, center=True).mean()

            linestyle = '--' if "hclim" in name.lower() else '-'
            label = f"r{member}i1p1f1 - {name}"

            plt.plot(run_mean.time, run_mean, color=pg.PALETTE[member-1], label=label, alpha=transparency, linestyle=linestyle)

    plt.ylim(lims)
    plt.vlines(x=np.datetime64("2014-12-31"), ymin=lims[0], ymax=lims[1], color="black", linestyle="--", alpha=0.2, label="End of Historical Period")
    plt.grid(axis="y", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel(f"{x.upper()} ({var.attrs['units']})")
    plt.title(f"{running_mean_window}-year RM of {x.upper()} for {scenario.upper()}")
    plt.legend(loc="upper left")
    pg.save(fig, f"./figures/data_section/climatology/timeseries_{x}_{running_mean_window}_{scenario}.png")
    plt.close()

if __name__ == "__main__":
    plot_timeseries_merged(x='tas', running_mean_window=11, scenario='ssp126')
    plot_timeseries_merged(x='tas', running_mean_window=11, scenario='ssp370')
    plot_timeseries_merged(x='pr', running_mean_window=11, scenario='ssp126')
    plot_timeseries_merged(x='pr', running_mean_window=11, scenario='ssp370')