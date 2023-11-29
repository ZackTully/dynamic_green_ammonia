import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

from pathlib import Path

plot_fig = [False, False, False, True, True]
# plot_fig = [True] * 5

top_dir = Path(__file__).parents[1]
fname = "main_df_opt.csv"
fname = "main_df_IN.csv"
fname = "main_df_TX.csv"

main_df = pd.read_csv(top_dir / "data" / "DL_runs" / fname)


techs = ["wind", "pv", "pipe", "lined", "salt", "EL", "battery", "ASU", "HB"]

capex_cols = [col for col in main_df.columns if "capex" in col]
opex_cols = [col for col in main_df.columns if "opex" in col]
cost_cols = [col for col in main_df.columns if "pex" in col]
rating_cols = [col for col in main_df.columns if "rating" in col]
LCOA_cols = [col for col in main_df.columns if "LCOA" in col]

ramp_lims = np.unique(main_df["ramp_lim"].to_numpy())
plant_mins = np.unique(main_df["plant_min"].to_numpy())

rl_ex = ramp_lims[np.argmin(np.abs(ramp_lims - 0.01))]
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


fig, ax = plt.subplots(2, 2, sharex="col", sharey="row")

tds, rls = np.meshgrid(plant_mins, ramp_lims)


def get_surf(col_name):
    out = np.zeros([len(ramp_lims), len(plant_mins)])
    for i in range(len(ramp_lims)):
        for j in range(len(plant_mins)):
            out[i, j] = main_df[
                (main_df["ramp_lim"] == ramp_lims[i])
                & (main_df["plant_min"] == plant_mins[j])
            ][col_name]
    return out


capacity_surf = get_surf("storage_capacity_kg")
c = ax[0, 0].pcolormesh(rls, tds, capacity_surf, cmap="RdBu")
fig.colorbar(c, ax=ax[0, 0])
ax[0, 0].set_title("H2 Storage capacity [kg]")

steady_ish = main_df[
    (main_df["ramp_lim"] == main_df["ramp_lim"].min())
    & (main_df["plant_min"] == main_df["plant_min"].max())
]
LCOA_pipe = get_surf("LCOA_pipe")
LCOA_lined = get_surf("LCOA_lined")
LCOA_salt = get_surf("LCOA_salt")

pipe_percdiff = (1 - LCOA_pipe / steady_ish["LCOA_pipe"].to_numpy()) * 100
lined_percdiff = (1 - LCOA_lined / steady_ish["LCOA_lined"].to_numpy()) * 100
salt_percdiff = (1 - LCOA_salt / steady_ish["LCOA_salt"].to_numpy()) * 100


pipe_norm = TwoSlopeNorm(vmin=-pipe_percdiff.max(), vcenter=0, vmax=pipe_percdiff.max())

lined_norm = TwoSlopeNorm(
    vmin=lined_percdiff.min() - 1e-5, vcenter=0, vmax=lined_percdiff.max()
)

salt_norm = TwoSlopeNorm(
    vmin=salt_percdiff.min() - 1e-5, vcenter=0, vmax=salt_percdiff.max()
)


c = ax[0, 1].pcolormesh(rls, tds, pipe_percdiff, norm=pipe_norm, cmap="RdBu")
fig.colorbar(c, ax=ax[0, 1])
ax[0, 1].set_title("Pipe LCOA Percent reduction")


c = ax[1, 0].pcolormesh(rls, tds, lined_percdiff, norm=pipe_norm, cmap="RdBu")
fig.colorbar(c, ax=ax[1, 0])
ax[1, 0].set_title("Lined LCOA Percent reduction")

c = ax[1, 1].pcolormesh(rls, tds, salt_percdiff, norm=pipe_norm, cmap="RdBu")
fig.colorbar(c, ax=ax[1, 1])
ax[1, 1].set_title("Salt LCOA Percent reduction")


ax[0, 0].set_ylabel("TD Ratio [% rated]")
ax[1, 0].set_ylabel("TD Ratio [% rated]")
ax[1, 0].set_xlabel("Ramp rate [% rated/hr]")
ax[1, 1].set_xlabel("Ramp rate [% rated/hr]")
# Sanity Check
sanity_check = False
if sanity_check:
    n_cols, n_rows = 3, 3

    n_figs = int(np.ceil((main_df.shape[1] - 2) / (n_cols * n_rows)))

    col_count = 0

    for fig_num in range(n_figs):
        fig, ax = plt.subplots(n_cols, n_rows, sharex="col")

        for col in range(n_cols):
            for row in range(n_rows):
                col_name = main_df.columns[col_count]
                ax[row, col].set_title(col_name)
                for i in range(len(ramp_lims)):
                    ax[row, col].plot(
                        plant_mins,
                        main_df[main_df["ramp_lim"] == ramp_lims[i]][col_name],
                    )
                col_count += 1
                if col_count >= main_df.shape[1]:
                    break
plt.show()
[]
