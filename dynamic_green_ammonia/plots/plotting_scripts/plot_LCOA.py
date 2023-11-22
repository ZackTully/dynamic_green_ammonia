import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


from pathlib import Path

plot_fig = [True, False, False, False, True, True]
plot_fig = [True] * 6

top_dir = Path(__file__).parents[2]
fname = "main_df_opt.csv"
fname = "main_df_TX.csv"

main_df = pd.read_csv(top_dir / "data" / "DL_runs" / fname)


ramp_lims = np.unique(main_df["ramp_lim"].to_numpy())
plant_mins = np.unique(main_df["plant_min"].to_numpy())

ramp_costs = main_df.loc[main_df["plant_min"] == plant_mins[1]]
plant_costs = main_df.loc[main_df["ramp_lim"] == ramp_lims[0]]


def plot_layer(ax: plt.Axes, df: pd.DataFrame, component: str, x, last_y: np.ndarray):
    col_names = [column for column in df.columns if component in column]
    this_y = df[col_names].sum(axis=1).to_numpy() / df["LT_NH3"].to_numpy() + last_y
    if (component == "pipe") or (component == "lined") or (component == "salt"):
        component = "storage"
    ax.fill_between(x, last_y, this_y, label=component)
    return this_y


def plot_LCOA_bars(ax: plt.Axes, df: pd.DataFrame, tech: str):
    steady_components = ["wind", "pv", "EL"]
    storage_component = tech
    varying_components = ["battery", "ASU", "HB"]

    # x = ramp_costs["ramp_lim"].to_numpy()
    x = plant_costs["plant_min"].to_numpy()
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


if plot_fig[0]:
    fig, ax = plt.subplots(3, 1, sharex="col", sharey="col")
    fig.suptitle("LCOA breakdown")
    handles, labels = plot_LCOA_bars(ax[0], plant_costs, "pipe")
    ax[0].set_ylabel("Pipe LCOA")
    handles, labels = plot_LCOA_bars(ax[1], plant_costs, "lined")
    ax[1].set_ylabel("Lined LCOA")
    handles, labels = plot_LCOA_bars(ax[2], plant_costs, "salt")
    ax[2].set_ylabel("Salt LCOA")
    ax[2].set_xlabel("Turndown Ratio")
    fig.legend(handles[::-1], labels[::-1], loc="outside right")

# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()


def make_colors(how_many):
    color_start = [val / 255 for val in (28, 99, 214)]
    color_end = [val / 255 for val in (217, 183, 13)]
    colors = np.linspace(color_start, color_end, len(how_many))
    return colors


if plot_fig[2]:
    colors = make_colors(ramp_lims)
    fig, ax = plt.subplots(3, 1, sharex="col")
    fig.suptitle("Troubleshooting")

    fig1, ax1 = plt.subplots(1, 1)
    fig1.suptitle("LCOA")

    ax0 = ax[0].twinx()

    for i, rl in enumerate(ramp_lims):
        LT_NH3 = main_df[main_df["ramp_lim"] == rl]["LT_NH3"]
        max_NH3 = main_df[main_df["ramp_lim"] == rl]["NH3_max"]
        H2_cap = main_df[main_df["ramp_lim"] == rl]["storage_capacity_kg"]
        H2_flow = main_df[main_df["ramp_lim"] == rl]["storage_flow_rate_kgphr"]
        H2_soc_f = main_df[main_df["ramp_lim"] == rl]["storage_soc_f"]

        LCOA_pipe = main_df[main_df["ramp_lim"] == rl]["LCOA_pipe"]
        LCOA_lined = main_df[main_df["ramp_lim"] == rl]["LCOA_lined"]
        LCOA_salt = main_df[main_df["ramp_lim"] == rl]["LCOA_salt"]

        ax[0].plot(plant_mins, LT_NH3, color=colors[i])
        ax0.plot(plant_mins, max_NH3)
        ax1.plot(plant_mins, LCOA_pipe, linestyle="solid", color=colors[i])
        ax1.plot(plant_mins, LCOA_lined, linestyle="dashed", color=colors[i])
        ax1.plot(plant_mins, LCOA_salt, linestyle="dotted", color=colors[i])

        ax[1].plot(plant_mins, H2_cap, color=colors[i])
        # ax1.plot(plant_mins, H2_flow, color=colors[i])
        ax[2].plot(plant_mins, H2_soc_f, color=colors[i], label=f"{rl:.3f}")

    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right")

    ax[2].set_xlabel("turndown [% rated]")
    ax0.set_ylabel("max NH3 rate [kg/hr]")
    ax[0].set_ylabel("LT NH3 [t]")
    ax[1].set_ylabel("storage capacity [kg]")
    ax[2].set_ylabel("final SOC [kg]")

    ax1.set_xlabel("turndown [% rated]")
    ax1.set_ylabel("LCOA [USD/t]")

    ax1.legend(
        handles=[
            Line2D([0, 0], [0, 0], linestyle="solid", color=colors[0]),
            Line2D([0, 0], [0, 0], linestyle="dashed", color=colors[0]),
            Line2D([0, 0], [0, 0], linestyle="dotted", color=colors[0]),
        ],
        labels=["pipe", "lined", "salt"],
    )
    fig1.legend(handles, labels, loc="outside right")

if plot_fig[3]:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle("H2 Capacity")

    rl, pm = np.meshgrid(ramp_lims, plant_mins)

    H2_cap = np.zeros(np.shape(rl))

    for i in range(np.shape(rl)[0]):
        for j in range(np.shape(rl)[1]):
            H2_cap[i, j] = main_df[
                (main_df["plant_min"] == pm[i, j]) & (main_df["ramp_lim"] == rl[i, j])
            ]["storage_capacity_kg"]

    LRC_min = 19.63e3 * np.ones_like(rl)
    salt_min = 25.96e3 * np.ones_like(rl)

    H2_surf = ax.plot_surface(rl, pm, H2_cap, cmap=cm.coolwarm)
    ax.plot_surface(rl, pm, LRC_min)
    ax.plot_surface(rl, pm, salt_min)

    ax.set_xlabel("ramp rate limit [%/hr]")
    # ax.set_yscale("log")
    ax.set_ylabel("turndown limit [% rated]")
    ax.set_zlabel("storage capacity [kg]")
    # ax.set_zscale("log", "inverse")

    fig.colorbar(H2_surf, shrink=0.5, aspect=5)


if plot_fig[5]:
    colors = make_colors(ramp_lims)
    fig, ax = plt.subplots(4, 1, sharex="col", sharey="col")
    fig.suptitle("Storage Cost")

    # ax[0].set_xscale("log")

    ax3 = ax[3].twinx()

    for i, pm in enumerate(ramp_lims):
        H2_cap = main_df[main_df["ramp_lim"] == pm]["storage_capacity_kg"]
        pipe_cost = (
            main_df[main_df["ramp_lim"] == pm]["pipe_capex"]
            + main_df[main_df["ramp_lim"] == pm]["pipe_opex"]
        )
        lined_cost = (
            main_df[main_df["ramp_lim"] == pm]["lined_capex"]
            + main_df[main_df["ramp_lim"] == pm]["lined_opex"]
        )
        salt_cost = (
            main_df[main_df["ramp_lim"] == pm]["salt_capex"]
            + main_df[main_df["ramp_lim"] == pm]["salt_opex"]
        )
        HB_cost = (
            main_df[main_df["ramp_lim"] == pm]["HB_capex"]
            + main_df[main_df["ramp_lim"] == pm]["HB_opex"]
        )

        ax[0].plot(plant_mins, pipe_cost, color=colors[i])
        ax[1].plot(plant_mins, lined_cost, color=colors[i])
        ax[2].plot(plant_mins, salt_cost, color=colors[i], label=f"{pm:.3f}")
        ax[3].plot(plant_mins, HB_cost, color=colors[i])
        ax3.plot(plant_mins, H2_cap, color=colors[i])

    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right")

    ax[2].set_xlabel("plant min [% rated]")
    ax[0].set_ylabel("pipe cost [USD]")
    ax[1].set_ylabel("lined cost [USD]")
    ax[2].set_ylabel("salt cost [USD]")
    ax[3].set_ylabel("Haber Bosch cost [USD]")
    ax3.set_ylabel("Storage capacity [kg]")


# Save all figures

count = 0

while len(plt.gcf().axes) > 0:
    cf = plt.gcf()
    name = cf._suptitle.get_text()
    cf.savefig(f"dynamic_green_ammonia/plots/{name}.png", format="png", dpi=300)
    plt.close(cf)

    if count > 5:
        break
    count += 1
plt.close()

[]
