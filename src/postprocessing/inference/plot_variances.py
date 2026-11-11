import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

means = np.load("dist_test/means_black.npy")
stds = np.load("dist_test/stds_black.npy")

target = xr.open_dataset("/scratch/project_465002687/ec_earth/sample/targets.zarr").sel(time="1956-01-01").to_array().values[0]
# target = np.zeros((4,412,424))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4,  4), layout="constrained")

t = np.linspace(len(means)-1, 0, len(means))

t = t[:-1]
means = means[:-1]
stds = stds[:-1]
for i, var in enumerate(["tas", "tasmin", "tasmax", "pr"]):
    ax1.plot(t, means[:, 0, i], label=var)
    ax1.axhline(target[i].mean(), color=ax1.get_lines()[-1].get_color(), linestyle="--")
    print(f"{var} mean: {means[-2, 0, i]:.4f}, target mean: {target[i].mean():.4f}")

    ax2.plot(t, stds[:, 0, i], label=var)
    ax2.axhline(target[i].std(), color=ax2.get_lines()[-1].get_color(), linestyle="--")
    print(f"{var} std: {stds[-2, 0, i].mean():.4f}, target std: {target[i].std():.4f}")

ax1.legend()
ax2.legend()
ax1.set_title("Mean values")
ax2.set_title("Standard deviation")
ax1.set_xlabel("Step")
ax1.set_ylabel("Value")
ax2.set_xlabel("Step")
ax2.set_ylabel("Value")
ax2.set_yscale("log")
ax1.xaxis.set_inverted(True)
ax2.xaxis.set_inverted(True)
plt.savefig("dist_test/variances_plot_superddpm.png", dpi=300)