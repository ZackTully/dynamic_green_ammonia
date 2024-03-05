"""
Try calculating the turndown flexibility based off of the max and min of some moving average filter of generation data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from dynamic_green_ammonia.tools.file_management import FileMan
from dynamic_green_ammonia.tools.common_calculations import moving_average
from dynamic_green_ammonia.technologies.demand import DemandOptimization

FM = FileMan()
FM.set_analysis_case("LCOA")

H2_gen, wind_gen, solar_gen = FM.load_gen_data()
hybrid_gen = wind_gen + solar_gen

df_all, df_full = FM.load_sweep_data()
H2_gen = H2_gen[0, 0:1000]
# H2_gen = H2_gen[0, :]

lat = df_all[df_all["gen_ind"] == 0]["lat"].unique()[0]
case_df = df_full[df_full["HOPP.site.lat"] == lat]


# widths = np.arange(int(0.5*len(H2_gen)), 2, -100)
widths = np.flip(np.logspace(1, 3.7, 5).astype(int))

H2_caps = np.zeros(len(widths))
max_dems = np.zeros(len(widths))
min_dems = np.zeros(len(widths))
fake_storage = np.zeros(len(widths))
fig, ax = plt.subplots(2, 1)
ax[0].plot(H2_gen, zorder=-1)

zord = 1.9

for i, width in enumerate(widths):


    ma, std = moving_average(H2_gen, width)

    max_demand = np.max(ma)
    min_demand = np.min(ma)

    # over_max = np.where(H2_gen > max_demand)[0]
    # under_min = np.where(H2_gen < min_demand)[0]

    # storage_gen = np.zeros(len(H2_gen))
    # storage_gen[over_max] = H2_gen[over_max] - max_demand
    # storage_gen[under_min] = H2_gen[under_min] - min_demand

    # fake_storage[i] = np.max(np.cumsum(storage_gen)) - np.min(np.cumsum(storage_gen))

    ramp_lim  = max_demand * 0.2

    DO = DemandOptimization(H2_gen, ramp_lim, min_demand, max_demand)

    x, success, res = DO.optimize()

    max_dems[i] = max_demand
    min_dems[i] = min_demand
    H2_caps[i] = DO.capacity
    print(i, width)

    zord -= 0.001
    ax[0].plot(ma, zorder=zord)

ax[1].plot(min_dems/max_dems, H2_caps, ".-")
# ax1t = ax[1].twinx()
ax[1].plot(case_df["run_params.turndown"], case_df["H2_storage.capacity_kg"])
ax[1].plot(min_dems/max_dems, fake_storage)

[]