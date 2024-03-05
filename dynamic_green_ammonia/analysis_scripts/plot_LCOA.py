# TODO: make a file with locations and state names then call that file based on the lat lon information rather than manually setting state names in plots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from pathlib import Path

style = "paper"
style = "pres"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")


# data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
# data_path = Path(__file__).parents[1] / "data" / "optimal_sizing"
save_path = Path(__file__).parents[1] / "plots"
input_path = Path(__file__).parents[1] / "inputs"

loc_info = pd.read_csv(input_path / "location_info.csv")
main_df = pd.read_csv(data_path / "full_sweep_main_df.csv")
sweep_df = pd.read_pickle(data_path / "hopp_sweep.pkl")
lats = main_df["HOPP.site.lat"].unique()
lons = main_df["HOPP.site.lon"].unique()
years = np.unique(main_df["HOPP.site.year"])
splits = []
tds = []
rls = []

print(lats)


loc_df = loc_info[loc_info["lon"].isin(lons)]


def get_df_at_ramp_lim(df, ramp_lim):
    rl_idx = np.argmin(np.abs(ramp_lims - ramp_lim))
    df_rl_constant = df[df["run_params.ramp_lim"] == ramp_lims[rl_idx]]
    # df_rl_constant = pd.concat(
    # [
    # df[df["run_params.turndown"] == 0],
    # df_rl_constant,
    # df[df["run_params.turndown"] == 1],
    # ]
    # )
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
        component = "H$_2$ Stor."
    if component == "HB":
        component = "HBR"
    ax.fill_between(x, last_y, this_y, color=color, label=component)
    return this_y


def plot_bars(ax, x, df, tech):
    generation_comps = ["Wind", "PV", "EL"]
    # generation_comps = []
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
    # ax.text(td_realistic - 0.1, 0.8 * y_bottom, f"$f_T={1-td_realistic}$", rotation=90)
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
    ax.set_xticks(np.linspace(1, 0, major_ticks), ["0", "0.5", "1"], minor=False)

    # ax.set_xticks(np.linspace(1, 0, 5), minor=True)
    # ax.set_xticks(np.linspace(2, 1, 3))

    # ax.invert_xaxis()
    ax.set_xlabel("$f_T$")

    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels, last_y


def add_annotations(ax, LCOA, both):
    LCOA_BAT = np.interp(td_realistic, np.linspace(0, 1, len(LCOA)), LCOA)
    arrowprops = {"color": "black", "width": 0.5, "headwidth": 2, "headlength": 3}

    ylim_top = ax.get_ylim()[1]

    y_inf = 0.95 * ylim_top
    y_bat = 0.8 * ylim_top

    if both:
        ax.annotate(
            "", xy=(td_realistic, LCOA_BAT), xytext=(0.51, y_bat + 5), arrowprops=arrowprops
        )
        ax.annotate(
            f"BAT: {LCOA_BAT:.0f}",
            (td_realistic, LCOA_BAT),
            (0.5, y_bat),
            # arrowprops=arrowprops,
        )
    ax.annotate("", (1, LCOA[-1]), (0.61, y_inf + 5), arrowprops=arrowprops)
    ax.annotate(
        f"Inflexible: {LCOA[-1]:.0f}",
        (1, LCOA[-1]),
        (0.6, y_inf)
        # (1, LCOA[-1]),
        # (0.85, LCOA[-1]),
        # arrowprops=arrowprops,
    )


ramp_lims = np.unique(main_df["run_params.ramp_lim"].to_numpy())
turndowns = np.unique(main_df["run_params.turndown"].to_numpy())

rl_realistic = 0.2
td_realistic = 0.6

# site_type = "green steel sites"
site_type = "CF and storage"

loc1 = loc_df[(loc_df["loc"] == "TX") & (loc_df["note"] == site_type)]
loc2 = loc_df[(loc_df["loc"] == "IA")] # & (loc_df["note"] == site_type)]


df_rl_constant_site0 = get_df_at_ramp_lim(
    main_df[main_df["HOPP.site.lon"] == loc1["lon"].iloc[0]], rl_realistic
)
df_rl_constant_site1 = get_df_at_ramp_lim(
    main_df[main_df["HOPP.site.lon"] == loc2["lon"].iloc[0]], rl_realistic
)


x = turndowns
fig, ax = plt.subplots(1, 3, sharex="row", sharey="row", figsize=(7.2, 3.5))
# fig.suptitle(f"LCOA")

handles, labels, LCOA = plot_bars(ax[0], x, df_rl_constant_site0, "pipe")
ax[0].set_ylabel("LCOA [USD/t]")
ax[0].set_title(f"Pipe, {loc1['loc'].iloc[0]}")
add_annotations(ax[0], LCOA, False)
ax[0].text(0.025, 0.96, "a.", transform=ax[0].transAxes)

handles, labels, LCOA = plot_bars(ax[1], x, df_rl_constant_site0, "salt")
ax[1].set_title(f"Salt, {loc1['loc'].iloc[0]}")
add_annotations(ax[1], LCOA, False)
ax[1].text(0.025, 0.96, "b.", transform=ax[1].transAxes)

ax[1].set_ylim((286.4518542378712, 635.140380203837))
# ax[2].spines[:].set_visible(False)
ax[2].axis("off")


# handles, labels, LCOA = plot_bars(ax[2], x, df_rl_constant_site1, "pipe")
# ax[2].set_title(f"Pipe, {loc2['loc'].iloc[0]}")
# add_annotations(ax[2], LCOA)
# ax[2].text(0.025, 0.96, "c.", transform=ax[2].transAxes)

ax[0].invert_xaxis()

# fig.legend(
#     # handles[::-1],
#     # labels[::-1],
#     handles,
#     labels,
#     loc="center",
#     bbox_to_anchor=(0.48, 0.75),
#     # bbox_to_anchor=(0.95, 0.5),
#     framealpha=1,
#     ncol=3,
#     handleheight=1,
#     labelspacing=0.1,
#     columnspacing=0.25,
#     fontsize=10,
#     handletextpad=0.25,
# )
fig.legend(
    # handles[::-1],
    # labels[::-1],
    np.flip(handles),
    np.flip(labels),
    loc="center",
    bbox_to_anchor=(0.9, 0.265),
    # bbox_to_anchor=(0.95, 0.5),
    framealpha=0.9,
    ncol=1,
    handleheight=1,
    labelspacing=0.1,
    columnspacing=0.25,
    fontsize=8,
    handletextpad=0.25,
)
fig.subplots_adjust(left=0.1, right=0.95, top=0.875, bottom=0.125, wspace=0.1)


if style == "pres":
    fig.savefig(save_path / f"LCOA_by_type_{style}_2.png", format="png")
    # fig.savefig(save_path / f"LCOA_by_type_{style}.pdf", format="pdf")
    ax[2].axis("on")
    handles, labels, LCOA = plot_bars(ax[2], x, df_rl_constant_site1, "pipe")
    ax[2].set_title(f"Pipe, {loc2['loc'].iloc[0]}")
    add_annotations(ax[2], LCOA, True)
    ax[2].text(0.025, 0.96, "c.", transform=ax[2].transAxes)

    fig.savefig(save_path / f"LCOA_by_type_{style}_3.png", format="png")
else:
    ax[2].axis("on")
    handles, labels, LCOA = plot_bars(ax[2], x, df_rl_constant_site1, "pipe")
    ax[2].set_title(f"Pipe, {loc2['loc'].iloc[0]}")
    add_annotations(ax[2], LCOA, True)
    ax[2].text(0.025, 0.96, "c.", transform=ax[2].transAxes)
    fig.savefig(save_path / f"LCOA_by_type_{style}.png", format="png")
    fig.savefig(save_path / f"LCOA_by_type_{style}.pdf", format="pdf")

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(4, 3))

loc1_CF = ((df_rl_constant_site0["LT_NH3"] / 30 / 8760 * 1e3) / df_rl_constant_site0["DGA.HB.rating_NH3"]).to_numpy()
loc2_CF = ((df_rl_constant_site1["LT_NH3"] / 30 / 8760 * 1e3) / df_rl_constant_site1["DGA.HB.rating_NH3"]).to_numpy()

ax[0].plot(turndowns, loc1_CF, '.-')
ax[1].plot(turndowns, loc2_CF, '.-')

ax[0].set_title("TX ammonia CF")
ax[1].set_title("IA ammonia CF")

ax[0].set_xticks(np.linspace(0, 1, 5), np.linspace(1, 0, 5))
ax[1].set_xticks(np.linspace(0, 1, 5), np.linspace(1, 0, 5))

ax[0].set_xlabel("$f_T$")
ax[1].set_xlabel("$f_T$")

ax[0].set_ylabel("Capacity Factor")
ax[0].invert_xaxis()
ax[1].invert_xaxis()
fig.tight_layout()

[]

plt.show()
