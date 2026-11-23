import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from src.postprocessing import plt_guide as pg

pg.setup()

run_id = "23522809"

means = np.load(f"samples/{run_id}/datamodule_val/batch00000/means.npy")
stds = np.load(f"samples/{run_id}/datamodule_val/batch00000/stds.npy")

target = xr.open_dataset("/scratch/project_465002687/ec_earth/sample/targets.zarr").sel(time="1956-01-01").to_array().values[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=pg.FIG_WIDE, layout="tight")

t = np.linspace(len(means)-1, 0, len(means))
t = t[:-1]
means = means[:-1]
stds = stds[:-1]

for i, var in enumerate(["tas", "tasmin", "tasmax", "pr"]):
    ax1.plot(t, means[:, 0, i])
    ax1.axhline(target[i].mean(), color=ax1.get_lines()[-1].get_color(), linestyle="--")
    print(f"{var} mean: {means[-2, 0, i]:.4f}, target mean: {target[i].mean():.4f}")

    ax2.plot(t, stds[:, 0, i], label=var)
    ax2.axhline(target[i].std(), color=ax2.get_lines()[-1].get_color(), linestyle="--", label="target")
    print(f"{var} std: {stds[-2, 0, i].mean():.4f}, target std: {target[i].std():.4f}")


fig.legend(loc="outside lower center", ncol=4, bbox_to_anchor=(0.5, 0))
fig.tight_layout(rect=[0, 0.12, 1, 1])

ax1.set_title("Mean values")
ax2.set_title("Standard deviation")
ax1.set_xlabel("Step")
ax1.set_ylabel("Value")
ax2.set_xlabel("Step")
ax2.set_ylabel("Value")
ax2.set_yscale("log")
ax1.xaxis.set_inverted(True)
ax2.xaxis.set_inverted(True)
pg.save(fig, f"samples/{run_id}/datamodule_val/batch00000/variances_plot.png")