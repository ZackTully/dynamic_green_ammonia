import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


from pathlib import Path

plot_fig = [False, False, False, True, True]
# plot_fig = [True] * 5

top_dir = Path(__file__).parents[1]
fname = "main_df_opt.csv"
fname = "main_df.csv"

main_df = pd.read_csv(top_dir / "data" / "DL_runs" / fname)


techs = ["wind", "pv", "pipe", "lined", "salt", "EL", "battery", "ASU", "HB"]

capex_cols = [col for col in main_df.columns if "capex" in col]
opex_cols = [col for col in main_df.columns if "opex" in col]
cost_cols = [col for col in main_df.columns if "pex" in col]
rating_cols = [col for col in main_df.columns if "rating" in col]
LCOA_cols = [col for col in main_df.columns if "LCOA" in col]

ramp_lims = np.unique(main_df["ramp_lim"].to_numpy())
plant_mins = np.unique(main_df["plant_min"].to_numpy())

rl_ex = ramp_lims[np.argmin(np.abs(ramp_lims - 0.2))]
rl_df = main_df[main_df["ramp_lim"] == rl_ex].set_index(plant_mins)

pm_ex = plant_mins[np.argmin(np.abs(plant_mins - 0.25))]
pm_df = main_df[main_df["plant_min"] == pm_ex].set_index(ramp_lims)


fig, ax = plt.subplots(2, 1)
rl_df[rating_cols].plot(ax=ax[0])
ax[0].set_xlabel("plant min")
pm_df[rating_cols].plot(ax=ax[1])
ax[1].set_xlabel("ramp limit")


fig, ax = plt.subplots(2, 1)
rl_df[cost_cols].sum(axis=1).plot(ax=ax[0])

ax0 = ax[0].twinx()
rl_df["storage_capacity_kg"].plot(ax=ax0)
ax0.plot([0, 1], [19.63e3] * 2)
ax0.plot([0, 1], [25.95e3] * 2)

rl_df[LCOA_cols].plot(ax=ax[1])


# ax[0].set_xlabel("plant min")
# pm_df[cost_cols].sum(axis=1).plot(ax=ax[1])
# pm_df["storage_capacity_kg"].plot(ax=ax[1].twinx())
# ax[1].set_xlabel("ramp limit")


[]
