import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
import scipy

style = "paper"


if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")


# data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
save_path = Path(__file__).parents[1] / "plots"

df_all = pd.read_csv(data_path / "full_sweep_main_df.csv")

# change storage units from kg to t
df_all["H2_storage.capacity_kg"] = df_all["H2_storage.capacity_kg"] / 1e3
lats = np.unique(df_all["HOPP.site.lat"])


locations = ["TX", "IA"]

fig, ax = plt.subplots(1, 2, sharey="row", figsize=(7.2, 3.5))  # , dpi=150)

for i, loc in enumerate(locations):
    main_df = df_all[df_all["HOPP.site.lat"] == lats[i]]

    ramp_lims = np.unique(main_df["run_params.ramp_lim"].to_numpy())
    turndowns = np.unique(main_df["run_params.turndown"].to_numpy())

    rl_realistic = 0.2
    td_realistic = 0.6

    # fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    data_name = "H2_storage.capacity_kg"

    f_R = ramp_lims
    f_T = 1 - turndowns

    # ax.plot([0, 0], [1, 1])

    ax[i].set_xlabel("Ramping flexibility $f_R$")

    # rl_fake = np.linspace(0, 1, len(ramp_lims))
    rl_fake = np.concatenate([[0], np.linspace(0.1, 0.9, len(ramp_lims) - 2), [1]])

    b = (0.4 - 0.2**2) / (0.2 - 0.2**2)
    a = 1 - b
    # rl_fake = a * ramp_lims**2 + b * ramp_lims

    td_fake = 1 - turndowns  # turndown flexibility

    data = np.zeros([len(ramp_lims), len(turndowns)])
    RL = np.zeros(np.shape(data))
    TD = np.zeros(np.shape(data))
    for k in range(len(ramp_lims)):
        for j in range(len(turndowns)):
            # if (ramp_lims[i] == 0) or (turndowns[j] == 1):  # inflexible case,
            #     data_point = main_df[
            #         (main_df["run_params.ramp_lim"] == ramp_lims[0])
            #         & (main_df["run_params.turndown"] == turndowns[-1])
            #     ][data_name]
            # elif (ramp_lims[i] == 1) or (turndowns[j] == 0):
            #     data_point = main_df[
            #         (main_df["run_params.ramp_lim"] == ramp_lims[-1])
            #         & (main_df["run_params.turndown"] == turndowns[0])
            #     ][data_name]
            if False:
                pass
            else:
                data_point = main_df[
                    (main_df["run_params.ramp_lim"] == ramp_lims[k])
                    & (main_df["run_params.turndown"] == turndowns[j])
                ][data_name]

            data[k, j] = data_point
            RL[k, j] = rl_fake[k]
            TD[k, j] = td_fake[j]

    n_levels = 10
    levels = np.linspace(
        df_all["H2_storage.capacity_kg"].min(),
        df_all["H2_storage.capacity_kg"].max(),
        n_levels,
    )

    CS0 = ax[i].contourf(RL, TD, data, levels=levels, cmap=cm.plasma)
    CS1 = ax[i].contour(RL, TD, data, levels=levels, cmap=cm.plasma, linewidths=0.25)

    y_locs = np.interp(levels, data[-1, :], np.linspace(1, 0, len(turndowns)))
    # y_locs = y_locs[np.where(levels <= np.max(data))[0]]
    y_locs = y_locs[np.where(y_locs > 0)[0]]

    x_locs = 0.6 * np.ones(len(y_locs))

    manual_locations = np.stack([x_locs, y_locs])[:, 1:].T
    clabels = ax[i].clabel(
        CS1,
        inline=True,
        fontsize=7,
        colors="black",
        manual=manual_locations,
    )

    # for txt in clabels:
    #     txt.set_bbox(
    #         # {"facecolor": "white", "edgecolor": "white", "pad": 0, "alpha": 0.5}
    #         {"color": "white", "pad": 0}
    #     )

    rl_fake_realistic = np.interp(rl_realistic, ramp_lims, rl_fake)

    BAT_x = [0, rl_fake_realistic, rl_fake_realistic]
    BAT_y = [1 - td_realistic, 1 - td_realistic, 0]
    BAT_cap = np.interp(
        td_realistic,
        turndowns,
        [
            np.interp(rl_realistic, ramp_lims, data[:, i])
            for i in range(np.shape(data)[1])
        ],
    )

    ax[i].plot(BAT_x, BAT_y, linestyle="dashed", color="black")
    # ax[i].text(
    #     rl_fake_realistic,
    #     1 - td_realistic,
    #     f"BAT:\n{BAT_cap:.0f}",
    #     backgroundcolor="white",
    # )
    arrowprops = dict(
        {"fc": "0.8", "ec": ".8", "width": 0.25, "headwidth": 1.5, "headlength": 3},
    )

    bbox = dict({"boxstyle": "round", "fc": "0.8", "ec": ".8"})
    ax[i].annotate(
        f"BAT:\n{BAT_cap:.0f} t",
        (rl_fake_realistic, 1 - td_realistic),
        (0.75, 0.5),
        bbox=bbox,
        arrowprops=arrowprops,
    )
    ax[i].annotate(
        f"Inflexible:\n{np.max(data):.0f} t",
        (0, 0),
        (0.2, 0.7),
        bbox=bbox,
        arrowprops=arrowprops,
    )
    print(
        f"loc: {loc}, BAT storage: {1 - BAT_cap/np.max(data)} frac. of inflexible reduction"
    )

    # XTICK locations
    # minor_ticks = np.linspace(0, 1, 11)
    minor_ticks = np.concatenate(
        [
            np.linspace(0, 0.0009, 10),
            np.linspace(0.001, 0.009, 10),
            np.linspace(0.01, 0.09, 10),
            np.linspace(0.1, 0.2, 2),
            [1],
        ]
    )
    # ax[i].set_xticks(xticks, np.round(np.interp(xticks, rl_fake, ramp_lims), 2))
    ax[i].set_xticks(
        np.interp(minor_ticks, ramp_lims, rl_fake),
        minor_ticks,
        minor=True,
        visible=False,
    )

    # major_ticks = np.linspace(0, 1, 3)
    # major_ticks = [0, 0.1, 0.2, 1]
    major_ticks = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    major_tick_labels = [
        "0",
        "$10^{-4}$",
        "$10^{-3}$",
        "$10^{-2}$",
        "$10^{-1}$",
        "1",
    ]
    # ax[i].set_xticks(xticks, np.round(np.interp(xticks, rl_fake, ramp_lims), 2))
    ax[i].set_xticks(
        np.interp(major_ticks, ramp_lims, rl_fake),
        # np.round(major_ticks, 2),
        major_tick_labels,
        minor=False,
    )

    d = 0.125
    kwargs = dict(
        marker=[(-d, -0.25), (d, 0.25)],
        markersize=6,
        linestyle="none",
        linewidth=0.5,
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )

    ax[i].plot(
        [0.025, 0.035, 0.9655, 0.975], [0, 0, 0, 0], transform=ax[i].transAxes, **kwargs
    )

    ax[i].set_title(f"{loc}")


# YTICK locations


# major_ticks = np.array([0, 0.4, 0.5, 1.0])
# ax[0].set_yticks(major_ticks, major_ticks, minor=False, visible=True)
# # ax[1].set_yticks([], [], minor=False, visible=False)

# minor_ticks = np.linspace(0, 1, 11)
# ax[0].set_yticks(minor_ticks, [], minor=True, visible=False)
# # ax[1].set_yticks(minor_ticks, [], minor=True, visible=False)

# ax[1].tick_params(axis="y", which="both", bottom=False)

# [label.set_visible(False) for label in ax[1].get_yticklabels()]

# plt.setp(ax[1].get_yticklabels(), visible=False)
# yticks = np.linspace(0, 1, 3)
# ax[0].set_yticks(yticks, np.interp(yticks, td_fake, turndowns))
ax[0].yaxis.set_major_formatter("{x:.1f}")
ax[0].set_ylabel("Turndown flexibility $f_T$")

sp_top = 0.9
sp_bottom = 0.125
sp_right = 0.825
sp_wspace = 0.125

fig.subplots_adjust(
    left=0.08, bottom=sp_bottom, right=sp_right, top=sp_top, wspace=sp_wspace
)
cbar_ax = fig.add_axes([sp_right + 0.025, sp_bottom, 0.05, sp_top - sp_bottom])
cbar = fig.colorbar(CS0, cax=cbar_ax)
cbar.ax.ticklabel_format(useMathText=True)
cbar.set_label("$H_2$ Storage capacity (t)")

# fig.tight_layout()
fig.savefig(save_path / f"Capacity_heat_{style}.png", format="png")
fig.savefig(save_path / f"Capacity_heat_{style}.pdf", format="pdf")


plt.show()
[]
