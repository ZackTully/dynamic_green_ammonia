import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path

style = "paper"


if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")


location = "IN"
analysis_type = "full_sweep"

# data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
save_path = Path(__file__).parents[1] / "plots"

df_all = pd.read_csv(data_path / "full_sweep_main_df.csv")
lats = np.unique(df_all["HOPP.site.lat"])


locations = ["TX", "IA"]

figs = []
axs = []
cbars = []

for i, loc in enumerate(locations):
    main_df = df_all[df_all["HOPP.site.lat"] == lats[i]]

    ramp_lims = np.unique(main_df["run_params.ramp_lim"].to_numpy())
    turndowns = np.unique(main_df["run_params.turndown"].to_numpy())

    rl_realistic = 0.2
    td_realistic = 0.6

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    data_name = "H2_storage.capacity_kg"

    f_R = ramp_lims
    f_T = 1 - turndowns

    ax.plot([0, 0], [1, 1])

    ax.set_xlabel("Ramping flexibility $f_R$")
    ax.set_ylabel("Turndown flexibility $f_T$")

    # f_R = ramp_lims
    # f_T = 1 - turndowns

    # rl_fake = np.log(np.linspace(np.exp(0), np.exp(1), len(ramp_lims)))
    rl_fake = np.linspace(0, 1, len(ramp_lims))
    # rl_fake = ramp_lims
    # rl_fake = np.concatenate([[0], np.logspace(-1, 0, len(ramp_lims) - 1)])
    # rl_logs = np.concatenate([1 - np.logspace(0, -1, 7), [1]])
    # rl_fake = np.concatenate([1 - np.exp(-ramp_lims[:-1]), [1]])

    b = (0.4 - 0.2**2) / (0.2 - 0.2**2)
    a = 1 - b
    # rl_fake = a * ramp_lims**2 + b * ramp_lims

    td_fake = 1 - turndowns  # turndown flexibility

    data = np.zeros([len(ramp_lims), len(turndowns)])
    RL = np.zeros(np.shape(data))
    TD = np.zeros(np.shape(data))
    for i in range(len(ramp_lims)):
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
                    (main_df["run_params.ramp_lim"] == ramp_lims[i])
                    & (main_df["run_params.turndown"] == turndowns[j])
                ][data_name]

            data[i, j] = data_point
            RL[i, j] = rl_fake[i]
            TD[i, j] = td_fake[j]

    # data[-1, -1] = -np.max(data)
    n_levels = 20
    # levels = np.linspace(np.min(data), np.max(data), 20)
    levels = np.linspace(
        df_all["H2_storage.capacity_kg"].min(),
        df_all["H2_storage.capacity_kg"].max(),
        n_levels,
    )

    # n_levels = 15
    # curviness = 1
    # interp_locs = np.log(np.linspace(np.exp(0), np.exp(curviness), n_levels)) / curviness
    # levels = np.interp(interp_locs, [0, 1], [np.min(data), np.max(data)])

    CS0 = ax.contourf(RL, TD, data, levels=levels, cmap=cm.plasma)

    rl_fake_realistic = np.interp(rl_realistic, ramp_lims, rl_fake)

    BAT_x = [0, rl_fake_realistic, rl_fake_realistic]
    BAT_y = [1 - td_realistic, 1 - td_realistic, 0]

    ax.plot(BAT_x, BAT_y, linestyle="dashed", color="black")

    ax.text(rl_fake_realistic, 1 - td_realistic, "BAT")

    cbar = fig.colorbar(CS0)

    # data = data_surface
    # # if isinstance(data_name, str):
    # #     RL, TD, data = get_3d_things(ramp_lims, turndowns, data_name)
    # # else:
    # #     data_name = ""

    # n_levels = 15
    # curviness = 1
    # interp_locs = np.log(np.linspace(np.exp(0), np.exp(curviness), n_levels)) / curviness
    # levels = np.interp(interp_locs, [0, 1], [np.min(data), np.max(data)])
    # color_kwargs = {"cmap": cm.plasma, "vmin": np.min(data), "vmax": np.max(data)}

    # CSf = ax.contourf(RL, TD, data, alpha=1, levels=levels, **color_kwargs)
    # CS1 = ax.contour(RL, TD, data, levels=levels, **color_kwargs)

    # rl_real_loc = np.interp(rl_realistic, ramp_lims, np.linspace(1, 0, len(ramp_lims)))

    # rect_kwargs = {"alpha": 0.5, "facecolor": "white"}
    # # rect1 = Rectangle([0, 0], rl_real_loc, 1, **rect_kwargs)
    # # rect2 = Rectangle(
    # #     [rl_real_loc, 1 - td_realistic], 1 - rl_real_loc, td_realistic, **rect_kwargs
    # # )
    # rect1 = Rectangle([0, 0], 1 - rl_real_loc, td_realistic, **rect_kwargs)
    # rect2 = Rectangle([1 - rl_real_loc, 0], rl_real_loc, 1, **rect_kwargs)
    # ax.add_patch(rect1)
    # ax.add_patch(rect2)

    # ax.plot([1 - rl_real_loc, 1 - rl_real_loc], [td_realistic, 1], color="black")
    # ax.plot([0, 1 - rl_real_loc], [td_realistic, td_realistic], color="black")
    # ax.clabel(CS1, CS1.levels, inline=True, colors="black")
    # cbar = fig.colorbar(CSf)
    # ax.set_xticks(RL[0, :], np.flip(ramp_lims))
    # ax.set_yticks(turndowns, np.round(turndowns, 2))
    # ax.invert_xaxis()
    # # ax.invert_yaxis()
    # ax.set_xlabel("ramp limit")
    # ax.set_ylabel("turndown ratio")

    # cbar.set_label(data_name)

    # ax.set_xticks(rl_fake, ramp_lims)
    # ax.set_yticks(td_fake, turndowns)

    xticks = np.linspace(0, 1, 6)
    ax.set_xticks(xticks, np.round(np.interp(xticks, rl_fake, ramp_lims), 2))

    # xticks = np.linspace(0, 1, 6)
    # ax.set_xticks(np.interp(xticks, ramp_lims, rl_fake), np.round(xticks, 2))

    # ax.xaxis.set_major_formatter("{x:.2f}")

    # ax.set_xscale("log")

    yticks = np.linspace(0, 1, 6)
    ax.set_yticks(yticks, np.interp(yticks, td_fake, turndowns))
    ax.yaxis.set_major_formatter("{x:.1f}")

    # cticks = np.linspace(np.min(data), np.max(data), 6)
    # cbar.set_ticks(cticks, labels=cticks, format=lambda x: f"{x:.2f}")
    # cbar.ax.yaxis.set_major_formatter("{x:.2g}")
    cbar.ax.ticklabel_format(useMathText=True)

    cbar.set_label("$H_2$ Storage capacity (kg)")

    ax.set_title(f"{loc}")

    figs.append(fig)
    axs.append(ax)
    cbars.append(cbar)

    # cb_low = np.min(cbar.boundaries)
    # cb_high = np.max(cbar.boundaries)

    # for cb in cbars:
    #     if np.min(cb.boundaries) < cb_low:
    #         cb_low = np.min(cb.boundaries)
    #     if np.max(cb.boundaries) > cb_high:
    #         cb_high = np.max(cb.boundaries)

    # for fig in figs:
    fig.tight_layout()
    fig.savefig(save_path / f"Capacity_heat_{loc}_{style}.png", format="png")
    fig.savefig(save_path / f"Capacity_heat_{loc}_{style}.pdf", format="pdf")


plt.show()
[]
