# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path


# %% Load style sheet

style = "paper"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")

# %% load data files

location = "IN"
analysis_type = "full_sweep"

# data_path = Path(__file__).parents[1] / "data" / "notable_runs" / "march-3_start"
# data_path = Path(__file__).parents[1] / "data" / "notable_runs" / "january_start"
data_path = Path(__file__).parents[1] / "data" / "DL_runs"
save_path = Path(__file__).parents[1] / "plots"

# capacities = np.load("dynamic_green_ammonia/technologies/demand_opt_capacities.npy")

# if location == "IN":
#     main_df_IN = pd.read_csv(data_path / f"{analysis_type}_main_df_IN.csv")
#     main_df = main_df_IN
# elif location == "TX":
#     main_df_TX = pd.read_csv(data_path / f"{analysis_type}_main_df_TX.csv")
#     main_df = main_df_TX


df_all = pd.read_csv(data_path / "full_sweep_main_df.csv")
lats = np.unique(df_all["HOPP.site.lat"])
years = np.unique(df_all["HOPP.site.year"])
if len(years) == 0:
    years = [years]
wind_caps = np.unique(df_all["HOPP.wind.rating_kw"])
wind_aep = np.unique(
    df_all[
        (df_all["HOPP.site.lat"] == lats[0])
        & (df_all["HOPP.site.year"] == years[0])
        & (df_all["HOPP.wind.rating_kw"] == wind_caps[2])
    ]["HOPP.wind.annual_energy"]
)
if len(wind_aep) == 0:
    wind_aep = [wind_aep]
main_df = df_all[
    (df_all["HOPP.site.lat"] == lats[0])
    & (df_all["HOPP.site.year"] == years[0])
    & (df_all["HOPP.wind.rating_kw"] == wind_caps[2])
    & (df_all["HOPP.wind.annual_energy"] == wind_aep[0])
]
# main_df = df_all

# %%

# ramp_lims = np.unique(main_df["run_params"]["ramp_lim"].to_numpy())
# turndowns = np.unique(main_df["run_params"]["turndown"].to_numpy())
ramp_lims = np.unique(main_df["run_params.ramp_lim"].to_numpy())
turndowns = np.unique(main_df["run_params.turndown"].to_numpy())

rl_realistic = 0.2
td_realistic = 0.6


# %% Plotting helper methods


def get_df_at_ramp_lim(df, ramp_lim):
    rl_idx = np.argmin(np.abs(ramp_lims - ramp_lim))
    df_rl_constant = df[df["run_params.ramp_lim"] == ramp_lims[rl_idx]]
    df_rl_constant = pd.concat(
        [
            df[df["run_params.turndown"] == 0],
            df_rl_constant,
            df[df["run_params.turndown"] == 1],
        ]
    )
    return df_rl_constant


def get_df_at_turndown(df, turndown):
    td_idx = np.argmin(np.abs(turndown - turndowns))
    df_td_constant = df[df["run_params.plant_min"] == turndowns[td_idx]]
    return df_td_constant


def plot_layer(ax: plt.Axes, df: pd.DataFrame, component: str, x, last_y: np.ndarray):
    # col_names = [column for column in df.columns if component in column]

    col_names = []
    for column in df.columns:
        if component in column:
            if ("capex" in column) or ("opex" in column):
                if component == "EL":
                    # if "rated_power" in column:
                    # col_names.append(column)
                    # if "kgpday" in column:
                    # col_names.append(column)
                    if "HOPP" in column:
                        col_names.append(column)
                else:
                    col_names.append(column)

    this_y = df[col_names].sum(axis=1).to_numpy() / df["LT_NH3"].to_numpy() + last_y
    if (component == "pipe") or (component == "lined") or (component == "salt"):
        component = "storage"
    ax.fill_between(x, last_y, this_y, label=component)
    return this_y


def plot_bars(ax, x, df, tech):
    steady_components = ["wind", "pv", "EL"]
    storage_component = tech
    varying_components = ["battery", "ASU", "HB"]
    last_y = np.zeros(len(df))
    for i in range(len(steady_components)):
        last_y = plot_layer(ax, df, steady_components[i], x, last_y)

    for i in range(len(varying_components)):
        last_y = plot_layer(ax, df, varying_components[i], x, last_y)
    y_bottom = np.mean(last_y)
    last_y = plot_layer(ax, df, storage_component, x, last_y)

    # ax.set_xscale("log")
    ax.set_ylim([0.75 * y_bottom, ax.get_ylim()[1]])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=True)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def get_3d_things(ramp_lims, turndowns, data_name):
    # flip ramp lims and turndowns here so that I don't have to flip anything later
    rl_fake = np.linspace(1, 0, len(ramp_lims))
    td_fake = np.linspace(1, 0, len(turndowns))
    RL, TD = np.meshgrid(rl_fake, td_fake)
    data_surface = np.zeros(np.shape(RL))
    for i in range(np.shape(RL)[0]):
        for j in range(np.shape(RL)[1]):
            if (ramp_lims[j] == 0) or (turndowns[i] == 1):  # inflexible case,
                data_surface[i, j] = main_df[
                    (main_df["run_params.ramp_lim"] == ramp_lims[0])
                    & (main_df["run_params.turndown"] == turndowns[-1])
                ][data_name]
            elif (ramp_lims[j] == 1) or (turndowns[i] == 0):
                data_surface[i, j] = main_df[
                    (main_df["run_params.ramp_lim"] == ramp_lims[-1])
                    & (main_df["run_params.turndown"] == turndowns[0])
                ][data_name]
            else:
                data_surface[i, j] = main_df[
                    (main_df["run_params.ramp_lim"] == ramp_lims[j])
                    & (main_df["run_params.turndown"] == turndowns[i])
                ][data_name]
    return RL, TD, data_surface


def plot_surface(fig, ax, data_name, RL=None, TD=None, data=None):
    if isinstance(data_name, str):
        RL, TD, data = get_3d_things(ramp_lims, turndowns, data_name)
    else:
        data_name = ""

    surf = ax.plot_surface(RL, TD, data, cmap=cm.plasma)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

    ax.set_xticks(RL[0, :], np.flip(ramp_lims))
    ax.set_yticks(turndowns, np.round(turndowns, 2))
    ax.set_zticklabels([])
    ax.set_xlabel("ramp limit")
    ax.set_ylabel("turndown ratio")
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    cbar.set_label(data_name)
    ax.view_init(elev=35, azim=-45, roll=0)
    # ax.view_init(elev=90, azim=0, roll=0)


def plot_heat(fig, ax, data_name, RL=None, TD=None, data=None):
    if isinstance(data_name, str):
        RL, TD, data = get_3d_things(ramp_lims, turndowns, data_name)
    else:
        data_name = ""

    n_levels = 15
    curviness = 1
    interp_locs = (
        np.log(np.linspace(np.exp(0), np.exp(curviness), n_levels)) / curviness
    )
    levels = np.interp(interp_locs, [0, 1], [np.min(data), np.max(data)])
    color_kwargs = {"cmap": cm.plasma, "vmin": np.min(data), "vmax": np.max(data)}

    CSf = ax.contourf(RL, TD, data, alpha=1, levels=levels, **color_kwargs)
    CS1 = ax.contour(RL, TD, data, levels=levels, **color_kwargs)

    rl_real_loc = np.interp(rl_realistic, ramp_lims, np.linspace(1, 0, len(ramp_lims)))

    rect_kwargs = {"alpha": 0.5, "facecolor": "white"}
    # rect1 = Rectangle([0, 0], rl_real_loc, 1, **rect_kwargs)
    # rect2 = Rectangle(
    #     [rl_real_loc, 1 - td_realistic], 1 - rl_real_loc, td_realistic, **rect_kwargs
    # )
    rect1 = Rectangle([0, 0], 1 - rl_real_loc, td_realistic, **rect_kwargs)
    rect2 = Rectangle([1 - rl_real_loc, 0], rl_real_loc, 1, **rect_kwargs)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.plot([1 - rl_real_loc, 1 - rl_real_loc], [td_realistic, 1], color="black")
    ax.plot([0, 1 - rl_real_loc], [td_realistic, td_realistic], color="black")
    ax.clabel(CS1, CS1.levels, inline=True, colors="black")
    cbar = fig.colorbar(CSf)
    ax.set_xticks(RL[0, :], np.flip(ramp_lims))
    ax.set_yticks(turndowns, np.round(turndowns, 2))
    ax.invert_xaxis()
    # ax.invert_yaxis()
    ax.set_xlabel("ramp limit")
    ax.set_ylabel("turndown ratio")

    cbar.set_label(data_name)


def plot_lines(ax, column):
    for rl in ramp_lims:
        if (rl == 0) or (rl == 1):
            continue
        ax.plot(
            turndowns[1:-1],  # OMITTING FIRST AND LAST turndown
            main_df[main_df["run_params.ramp_lim"] == rl][column],
            color=cm.plasma(rl / ramp_lims[-1]),
        )
    ax.set_xlabel("tds")
    ax.set_ylabel(column)


def get_colors():
    pass


# %%

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
plot_surface(fig, ax, "H2_storage.capacity_kg")
# fig.savefig(save_path / f"Capacity_surface_{style}.png", format="png")
# fig.savefig(save_path / f"Capacity_surface_{style}.pdf", format="pdf")

fig, ax = plt.subplots(1, 1)
plot_heat(fig, ax, "H2_storage.capacity_kg")
# fig.savefig(save_path / f"Capacity_heat_{style}.png", format="png")
# fig.savefig(save_path / f"Capacity_heat_{style}.pdf", format="pdf")

df_rl_constant = get_df_at_ramp_lim(main_df, rl_realistic)
# x = turndowns
x = turndowns


fig, ax = plt.subplots(1, 3, sharex="row", sharey="row")
fig.suptitle("LCOA breakdown")
handles, labels = plot_bars(ax[0], x, df_rl_constant, "pipe")
ax[0].set_ylabel("LCOA [USD/t]")
ax[0].set_title("Pipe storage")
ax[0].set_xlabel("Turndown Ratio")
handles, labels = plot_bars(ax[1], x, df_rl_constant, "lined")
ax[1].set_title("Lined cavern")
ax[1].set_xlabel("Turndown Ratio")
handles, labels = plot_bars(ax[2], x, df_rl_constant, "salt")
ax[2].set_title("Salt cavern")
ax[2].set_xlabel("Turndown Ratio")
fig.legend(handles[::-1], labels[::-1], loc="outside right")

# fig.savefig(save_path / f"LCOA_by_type_{style}.png", format="png")
# fig.savefig(save_path / f"LCOA_by_type_{style}.pdf", format="pdf")

# %%

# fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(15, 5))
# plot_surface(fig, ax[0], "H2_storage.initial_state_kg")
# ax[0].set_title("Initial state [kg]")

# RL, TD, data_cap = get_3d_things(ramp_lims, turndowns, "H2_storage.capacity_kg")
# RL, TD, data_state = get_3d_things(ramp_lims, turndowns, "H2_storage.initial_state_kg")
# soc_data = data_state / data_cap

# # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# plot_surface(fig, ax[1], None, RL=RL, TD=TD, data=soc_data)
# ax[1].set_title("Initial SOC ")

# %%

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# plot_surface(fig, ax, "H2_storage.min_state_index")

# %% troubleshooting plots

# fig, ax = plt.subplots(2, 2, sharex="col")

# plot_lines(ax[0, 0], "LT_NH3")
# plot_lines(ax[0, 1], "HOPP.wind.annual_energy")
# plot_lines(ax[1, 0], "Electrolyzer.HOPP_EL.H2_annual_output")
# plot_lines(ax[1, 1], "DGA.EL.H2_max")


# %%

# # fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# fig, ax = plt.subplots(1, 1)

# RL, TD, data_cap = get_3d_things(ramp_lims, turndowns, "H2_storage.capacity_kg")
# RL, TD, data_flow = get_3d_things(ramp_lims, turndowns, "H2_storage.max_demand")
# duration_data = data_cap / data_flow / 24
# plot_heat(fig, ax, None, RL=RL, TD=TD, data=duration_data)
# fig.suptitle("Storage duration based on max demand [days]")

# fig, ax = plt.subplots(1, 1)

# RL, TD, data_cap = get_3d_things(ramp_lims, turndowns, "H2_storage.capacity_kg")
# RL, TD, max_flow = get_3d_things(ramp_lims, turndowns, "H2_storage.max_chg_kgphr")
# RL, TD, min_flow = get_3d_things(ramp_lims, turndowns, "H2_storage.min_chg_kgphr")
# duration_data = data_cap / (-min_flow) / 24
# plot_heat(fig, ax, None, RL=RL, TD=TD, data=duration_data)
# fig.suptitle("Storage duration based on max discharge [days]")

# %%
plt.show()
