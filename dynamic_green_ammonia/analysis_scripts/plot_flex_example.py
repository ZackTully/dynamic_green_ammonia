import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dynamic_green_ammonia.tools.file_management import FileMan
from dynamic_green_ammonia.technologies.demand import DemandOptimization

example = False
style = "paper"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")

FM = FileMan()
FM.set_analysis_case("cases_check")

wind_gen = np.load(FM.data_path / "wind_gen.npy")
solar_gen = np.load(FM.data_path / "solar_gen.npy")
H2_gen = np.load(FM.data_path / "H2_gen.npy")
hybrid_gen = wind_gen + solar_gen

if example:
    t_start = 257
    t_end = 327
    dt = 1

    time = np.arange(t_start, t_end, 1)

    gen = hybrid_gen[0, time]
    steady = np.ones(len(gen)) * np.mean(gen)
    flexible = 0.5 * gen + 0.5 * steady
    width = 5
    cs = np.cumsum(flexible)
    filtered_data = (cs[width:] - cs[:-width]) / width
    flexible = np.interp(
        time, np.linspace(t_start, t_end, len(time) - width), filtered_data
    )
    full_flex = gen

else:
    t_start = 867
    t_stop = 920
    dt = 1
    time = np.arange(t_start, t_stop, dt)
    H2_gen = H2_gen[0, t_start:t_stop]
    gen = H2_gen
    N = len(H2_gen)

    RL = [0.01, 0.99]
    TD = [0, 0.25]

    demands = []
    storages = []
    d_constraints = []

    for i in range(len(RL)):
        rl = RL[i]
        td = TD[i]

        A = np.array([[1, -np.max(H2_gen)], [1, -np.mean(H2_gen)]])
        b = np.array([0, np.mean(H2_gen)])
        coeffs = np.linalg.inv(A) @ b
        max_demand = coeffs[0] / (td + coeffs[1])
        min_demand = td * coeffs[0] / (td + coeffs[1])

        DO = DemandOptimization(H2_gen, rl * max_demand, min_demand, max_demand)
        x, success, res = DO.optimize()
        demand = x[0:N]
        storage = x[N : 2 * N]

        demands.append(demand)
        storages.append(storage)
        d_constraints.append([min_demand, max_demand])


hbr_color = (210 / 255, 107 / 255, 102 / 255)
storage_color = (97 / 255, 15 / 255, 161 / 255)


def make_plot(ax, time, gen, dem, case, storage=None):
    if storage is None:
        storage = np.cumsum(gen - dem)

    # fig, ax = plt.subplots(2, 1, sharex="col")

    ax[0].fill_between(time, gen, dem, alpha=0.25, color=storage_color)
    ax[0].plot(time, gen, label="Generation", color="blue")
    ax[0].plot(time, dem, label="Demand", color=hbr_color)
    ax[1].plot(time, storage, label="Storage", color=storage_color)

    for axis in ax:
        axis.spines[
            [
                "top",
                # "left",
                # "bottom",
                "right",
            ]
        ].set_visible(False)
        axis.spines[:].set_color("gray")
        axis.spines[:].set_alpha(0.25)
        axis.set_xticks([])
        axis.set_yticks([])

    ax[0].set_ylabel("Inputs")
    ax[1].set_ylabel("Storage")
    ax[1].set_xlabel("Time")

    leg_kwargs = {"loc": "upper right", "frameon": False}
    ax[0].legend(**leg_kwargs)
    ax[1].legend(**leg_kwargs)

    ax[0].set_title(case)

    # return fig, ax
    return [], []


sp_kwargs = {"figsize": (3.6, 3.5), "sharex": "col"}
sp_adj = {"left": 0.1, "top": 0.9, "right": 0.95, "bottom": 0.1}

if example:
    fig1, ax1 = plt.subplots(2, 1, **sp_kwargs)
    _, _ = make_plot(ax1, time, gen, steady, "Inflexible")

    fig2, ax2 = plt.subplots(2, 1, **sp_kwargs)
    _, _ = make_plot(ax2, time, gen, flexible, "Partial flexibility")

    fig3, ax3 = plt.subplots(2, 1, **sp_kwargs)
    _, _ = make_plot(ax3, time, gen, full_flex, "Full flexibility")

    for ax in [ax2, ax3]:
        ax[1].set_ylim(ax1[1].get_ylim())

    for fig in [fig1, fig2, fig3]:
        fig.subplots_adjust(**sp_adj)
        fig.savefig(
            FM.plot_path
            / "presentation_plots"
            / f"flex_ex_{fig.get_axes()[0].title.get_text()}.png",
            format="png",
        )
else:
    hline_kwargs = {
        "linewidth": 0.5,
        "color": "black",
        "linestyle": "dashed",
        "zorder": 0.5,
    }

    fig1, ax1 = plt.subplots(2, 1, **sp_kwargs)
    _, _ = make_plot(ax1, time, gen, demands[0], "Ramping constraint", storages[0])

    ax1[0].hlines(
        [d_constraints[0][0], d_constraints[0][1]], time[0], time[-1], **hline_kwargs
    )

    text_offset = 250
    ax1[0].text(t_start, d_constraints[0][0] + text_offset, "$\\underline{d}$")
    ax1[0].text(t_start, d_constraints[0][1] + text_offset, "$\\overline{d}$")

    fig2, ax2 = plt.subplots(2, 1, **sp_kwargs)
    _, _ = make_plot(ax2, time, gen, demands[1], "Turndown constraint", storages[1])

    ax2[0].hlines(
        [d_constraints[1][0], d_constraints[1][1]], time[0], time[-1], **hline_kwargs
    )

    ax2[0].text(t_start, d_constraints[1][0] + text_offset, "$\\underline{d}$")
    ax2[0].text(t_start, d_constraints[1][1] + text_offset, "$\\overline{d}$")

    y_low = np.min([ax1[1].get_ylim()[0], ax2[1].get_ylim()[0]])
    y_high = np.max([ax1[1].get_ylim()[1], ax2[1].get_ylim()[1]])
    ax1[1].set_ylim([y_low, y_high])
    ax2[1].set_ylim([y_low, y_high])

    for fig in [fig1, fig2]:
        fig.subplots_adjust(**sp_adj)
        fig.savefig(
            FM.plot_path
            / "presentation_plots"
            / f"flex_ex_{fig.get_axes()[0].title.get_text()}.png",
            format="png",
        )

    fig, ax = plt.subplots(2, 1, **sp_kwargs, sharey="col", layout="constrained")
    for axis in ax:
        axis.spines[
            [
                "top",
                # "left",
                # "bottom",
                "right",
            ]
        ].set_visible(False)
        axis.spines[:].set_color("gray")
        axis.spines[:].set_alpha(0.25)
        axis.set_xticks([])
        axis.set_yticks([])

    ax[0].fill_between(time, gen, demands[0], alpha=0.25, color=storage_color)
    ax[0].plot(time, gen, label="Generation", color="blue")
    ax[0].plot(time, demands[0], label="NH$_3$ plant", color=hbr_color)

    ax[1].fill_between(time, gen, demands[1], alpha=0.25, color=storage_color)
    ax[1].plot(time, gen, label="Generation", color="blue")
    ax[1].plot(time, demands[1], label="NH$_3$ plant", color=hbr_color)
    ax[1].hlines(
        [d_constraints[1][0], d_constraints[1][1]], time[0], time[-1], **hline_kwargs
    )

    ax[1].text(t_start, d_constraints[1][0] + text_offset, "$\\underline{d}$")
    ax[1].text(t_start, d_constraints[1][1] + text_offset, "$\\overline{d}$")

    ax[0].set_ylabel("Inputs")
    ax[1].set_ylabel("Inputs")
    ax[1].set_xlabel("Time")
    ax[0].set_xlabel("Time")

    leg_kwargs = {"loc": "upper right", "frameon": False}
    ax[0].legend(**leg_kwargs)
    ax[1].legend(**leg_kwargs)
    ax[0].set_title("Ramping constraint")
    ax[1].set_title("Turndown constraint")
    fig.subplots_adjust(hspace=0.2)
    fig.savefig(FM.plot_path / "presentation_plots" / "flex_cons_stacks.png", format="png")
[]
# fig, ax = plt.subplots(2, 2, figsize=(7.2, 3.5), sharex="col", sharey="row")


# fig1, ax1 = make_plot(ax[:, 0], time, gen, steady, "steady")
# fig2, ax2 = make_plot(ax[:, 1], time, gen, flexible, "Flexiible")


# ax[0, 0].set_ylabel("Inputs")
# ax[1, 0].set_ylabel("Storage")
# ax[1, 0].set_xlabel("Time")
# ax[1, 1].set_xlabel("Time")


# leg_kwargs = {"loc": "upper right", "frameon": False}
# ax[0, 0].legend(**leg_kwargs)
# ax[1, 0].legend(**leg_kwargs)
# fig.subplots_adjust(left=.05, right=.95, top=.95, bottom=.1)

# fig.savefig(FM.plot_path / "presentation_plots/flex_ex.png", format="png")


# ylim_low = np.min([ax1[1].get_ylim()[0], ax2[1].get_ylim()[0]])
# ylim_high = np.min([ax1[1].get_ylim()[1], ax2[1].get_ylim()[1]])


[]
