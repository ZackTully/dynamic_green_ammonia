# TODO: make a file with locations and state names then call that file based on the lat lon information rather than manually setting state names in plots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from pathlib import Path

style = "paper"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")


data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
save_path = Path(__file__).parents[1] / "plots"


main_df = pd.read_csv(data_path / "full_sweep_main_df.csv")
sweep_df = pd.read_pickle(data_path / "hopp_sweep.pkl")
lats = np.unique(main_df["HOPP.site.lat"])
years = np.unique(main_df["HOPP.site.year"])
splits = []
tds = []
rls = []

print(lats)


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


def plot_layer(
    ax: plt.Axes, df: pd.DataFrame, component: str, color, x, last_y: np.ndarray
):
    # col_names = [column for column in df.columns if component in column]
    col_names = []
    for column in df.columns:
        if component.upper() in column.upper():
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
        component = "H$_2$ Storage"
    if component == "HB":
        component = "HBR"
    ax.fill_between(x, last_y, this_y, color=color, label=component)
    return this_y


def plot_bars(ax, x, df, tech):
    generation_comps = ["Wind", "PV", "EL"]
    storage_comps = ["Battery", tech]
    enduse_comps = ["ASU", "HB"]

    # steady_components = ["wind", "pv", "EL"]
    # storage_component = tech
    # varying_components = ["battery", "ASU", "HB"]
    last_y = np.zeros(len(df))

    for i in range(len(generation_comps)):
        last_y = plot_layer(
            ax, df, generation_comps[i], cm.plasma(0.75 + i / 10), x, last_y
        )
    y_bottom = np.mean(last_y)

    for i in range(len(enduse_comps)):
        last_y = plot_layer(ax, df, enduse_comps[i], cm.plasma(0.5 + i / 10), x, last_y)

    for i in range(len(storage_comps)):
        last_y = plot_layer(
            ax, df, storage_comps[i], cm.plasma(0.1 + i / 10), x, last_y
        )

    # for i in range(len(steady_components)):
    #     last_y = plot_layer(ax, df, steady_components[i], x, last_y)

    # for i in range(len(varying_components)):
    #     last_y = plot_layer(ax, df, varying_components[i], x, last_y)
    # y_bottom = np.mean(last_y)
    # last_y = plot_layer(ax, df, storage_component, x, last_y)

    ax.plot(
        [td_realistic, td_realistic],
        [0, np.interp(td_realistic, x, last_y)],
        linestyle="dashed",
        color="black",
        linewidth=0.75,
    )
    ax.text(td_realistic - 0.1, 0.8 * y_bottom, f"$T={td_realistic}$", rotation=90)
    print(tech)
    print(f"LCOA inflexible: {last_y[-1]:.2f}")
    print(f"LCOA BAT: {np.interp(td_realistic, x, last_y):.2f}")
    print(f"LCOA flexible: {last_y[0]:.2f}")
    ax.set_ylim([0.75 * y_bottom, ax.get_ylim()[1]])

    minor_ticks = 5
    major_ticks = 3

    ax.set_xticks(np.linspace(0, 1, minor_ticks), minor=True)
    # ax.set_xticks(
    #     np.linspace(0, 1, major_ticks), np.linspace(1, 0, major_ticks), minor=False
    # )
    ax.set_xticks(np.linspace(0, 1, major_ticks), ["0", "0.5", "1"], minor=False)

    # ax.set_xticks(np.linspace(1, 0, 5), minor=True)
    # ax.set_xticks(np.linspace(2, 1, 3))

    ax.invert_xaxis()
    ax.set_xlabel("$f_T$")

    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


ramp_lims = np.unique(main_df["run_params.ramp_lim"].to_numpy())
turndowns = np.unique(main_df["run_params.turndown"].to_numpy())

rl_realistic = 0.2
td_realistic = 0.6

df_rl_constant = get_df_at_ramp_lim(
    main_df[main_df["HOPP.site.lat"] == lats[1]], rl_realistic
)


locs = ["TX", "IA"]

figs = []
axs = []

for i, loc in enumerate(locs):
    df_rl_constant = get_df_at_ramp_lim(
        main_df[main_df["HOPP.site.lat"] == lats[i]], rl_realistic
    )

    x = turndowns

    fig, ax = plt.subplots(1, 3, sharex="row", sharey="row")
    fig.suptitle(f"LCOA {loc}")
    handles, labels = plot_bars(ax[0], x, df_rl_constant, "pipe")
    ax[0].set_ylabel("LCOA [USD/t]")
    ax[0].set_title("Pipe storage")
    # ax[0].set_xlabel("Turndown Ratio")
    handles, labels = plot_bars(ax[1], x, df_rl_constant, "lined")
    ax[1].set_title("Lined cavern")
    # ax[1].set_xlabel("Turndown Ratio")
    handles, labels = plot_bars(ax[2], x, df_rl_constant, "salt")
    ax[2].set_title("Salt cavern")

    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="center",
        bbox_to_anchor=(0.775, 0.675),
        framealpha=1,
    )
    fig.tight_layout()
    figs.append(fig)
    axs.append(ax)

ylim_low = ax[0].get_ylim()[1]
ylim_high = 0
for i, ax in enumerate(axs):
    if ax[0].get_ylim()[1] > ylim_high:
        ylim_high = ax[0].get_ylim()[1]

    if ax[0].get_ylim()[0] < ylim_low:
        ylim_low = ax[0].get_ylim()[0]


for i, loc in enumerate(locs):
    axs[i][0].set_ylim([ylim_low, ylim_high])
    figs[i].savefig(save_path / f"LCOA_by_type_{loc}_{style}.png", format="png")
    figs[i].savefig(save_path / f"LCOA_by_type_{loc}_{style}.pdf", format="pdf")

plt.show()
