import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path


class plottin_helper():
    def __init__(self):

        ramp_lims
        turndowns

        style = "paper"

        if style == "paper":
            plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
        elif style == "pres":
        plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")


    def get_df_at_ramp_lim(df, ramp_lim):
        rl_idx = np.argmin(np.abs(ramp_lims - ramp_lim))
        df_rl_constant = df[df["run_params.ramp_lim"] == ramp_lims[rl_idx]]
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
                        #     col_names.append(column)
                        # if "kgpday" in column:
                        #     col_names.append(column)
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
        y_bottom = np.mean(last_y)
        last_y = plot_layer(ax, df, storage_component, x, last_y)
        for i in range(len(varying_components)):
            last_y = plot_layer(ax, df, varying_components[i], x, last_y)
        # ax.set_xscale("log")
        ax.set_ylim([0.9 * y_bottom, ax.get_ylim()[1]])
        handles, labels = ax.get_legend_handles_labels()
        return handles, labels


    def get_3d_things(ramp_lims, turndowns, data_name):
        # flip ramp lims and turndowns here so that I don't have to flip anything later

        rl_fake = np.linspace(0, 1, len(ramp_lims))
        td_fake = np.linspace(0, 1, len(turndowns))
        RL, TD = np.meshgrid(rl_fake, td_fake)
        data_surface = np.zeros(np.shape(RL))
        for i in range(np.shape(RL)[0]):
            for j in range(np.shape(RL)[1]):
                data_surface[i, j] = main_df[
                    (main_df["run_params.ramp_lim"] == ramp_lims[i])
                    & (main_df["run_params.turndown"] == turndowns[j])
                ][data_name]
        return RL, TD, data_surface


    def plot_surface(fig, ax, data_name):
        RL, TD, data = get_3d_things(ramp_lims, turndowns, data_name)
        surf = ax.plot_surface(RL, TD, data, cmap=cm.plasma)
        cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

        ax.set_xticks(RL[0, :], ramp_lims)
        ax.set_yticks(turndowns, np.round(turndowns, 2))
        ax.set_zticklabels([])
        ax.set_xlabel("ramp limit")
        ax.set_ylabel("turndown ratio")
        # ax.invert_yaxis()
        # ax.invert_xaxis()
        cbar.set_label(data_name)
        ax.view_init(elev=35, azim=135, roll=0)


    def plot_heat(fig, ax, data_name):
        RL, TD, data = get_3d_things(ramp_lims, turndowns, data_name)

        n_levels = 15
        curviness = 1
        interp_locs = (
            np.log(np.linspace(np.exp(0), np.exp(curviness), n_levels)) / curviness
        )
        levels = np.interp(interp_locs, [0, 1], [np.min(data), np.max(data)])
        color_kwargs = {"cmap": cm.plasma, "vmin": 1e5, "vmax": np.max(data)}

        CSf = ax.contourf(RL, TD, data, alpha=1, levels=levels, **color_kwargs)
        CS1 = ax.contour(RL, TD, data, levels=levels, **color_kwargs)

        rect_kwargs = {"alpha": 0.5, "facecolor": "white"}
        rect1 = Rectangle([0, 0], rl_realistic, 1, **rect_kwargs)
        rect2 = Rectangle(
            [rl_realistic, 1 - td_realistic], 1 - rl_realistic, td_realistic, **rect_kwargs
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        ax.plot([rl_realistic, rl_realistic], [0, 1 - td_realistic], color="black")
        ax.plot([rl_realistic, 1], [1 - td_realistic, 1 - td_realistic], color="black")
        ax.clabel(CS1, CS1.levels, inline=True, colors="black")
        cbar = fig.colorbar(CSf)
        ax.set_xticks(RL[0, :], np.flip(ramp_lims))
        ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
        ax.invert_xaxis()
        # ax.invert_yaxis()
        ax.set_xlabel("ramp limit")
        ax.set_ylabel("turndown ratio")

        cbar.set_label(data_name)


    def get_colors():
        pass