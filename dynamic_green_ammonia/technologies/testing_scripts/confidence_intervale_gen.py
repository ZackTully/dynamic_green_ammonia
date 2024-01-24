import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from pathlib import Path

data_path = Path(__file__).parents[2] / "data" / "LCOA_runs"
save_path = Path(__file__).parents[2] / "plots"


gen_profiles = np.load(data_path / "H2_gen.npy").T
wind_profiles = np.load(data_path / "wind_gen.npy").T
solar_profiles = np.load(data_path / "solar_gen.npy").T

gen_profiles = wind_profiles + solar_profiles


df_all = pd.read_pickle(data_path / "hopp_sweep.pkl")
df_full = pd.read_csv(data_path / "full_sweep_main_df.csv")

data = gen_profiles[:, 0]

width = 24 * 30 * 3

cs = np.cumsum(data)
filtered_data = (cs[width:] - cs[:-width]) / width
std_dev = np.zeros(len(filtered_data))
for i in range(len(std_dev)):
    std_dev[i] = np.std(data[i : i + width])

filtered_data = np.interp(
    np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), filtered_data
)
std_dev = np.interp(
    np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), std_dev
)

fig, ax = plt.subplots(1, 1)

sorted_data = np.sort(data)
mean_ind = np.argmin(np.abs(sorted_data - np.mean(sorted_data)))

for frac in np.linspace(0.01, 0.99, 100):
    frac_in = frac
    low_ind = mean_ind + int(frac_in / 2 * len(data))
    low_ind = np.min([low_ind, len(data) - 1])
    high_ind = mean_ind - int(frac_in / 2 * len(data))
    high_ind = np.max([high_ind, 0])
    frac_captured = (high_ind - low_ind) / len(data)
    print(
        f"mean: {np.mean(data):.0f}, frac: {frac_in}, low: {sorted_data[low_ind]:.0f}, high: {sorted_data[high_ind]:.0f}"
    )
    ax.plot([sorted_data[high_ind], sorted_data[low_ind]], 2 * [frac], color="black")
plt.show()
[]
