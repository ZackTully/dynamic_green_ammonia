import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from pathlib import Path
import calendar
from datetime import datetime
import seaborn

from dynamic_green_ammonia.tools.file_management import FileMan

FM = FileMan()
FM.set_analysis_case("heat")




year = 2012
# style = "paper"
style = "pres"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")

data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
# data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
save_path = Path(__file__).parents[1] / "plots"
input_path = Path(__file__).parents[1] / "inputs"

gen_profiles = np.load(data_path / "H2_gen.npy").T
wind_profiles = np.load(data_path / "wind_gen.npy").T
solar_profiles = np.load(data_path / "solar_gen.npy").T

gen_profiles = wind_profiles + solar_profiles

loc_df = pd.read_csv(input_path / "location_info.csv")
# site_type = "green steel sites"
site_type = "CF and storage"

loc1 = loc_df[(loc_df["loc"] == "TX") & (loc_df["note"] == site_type)]
loc2 = loc_df[(loc_df["loc"] == "IA") & (loc_df["note"] == site_type)]


df_all = pd.read_pickle(data_path / "hopp_sweep.pkl")
df_full = pd.read_csv(data_path / "full_sweep_main_df.csv")

lats = np.unique(df_all["lat"])

# df = df_all[df_all["lat"] == lats[0]]
df = df_all[df_all["lat"] == lats[1]]

splits = np.unique(df["split"])

RL = np.unique(df_all["rl"])
TD = np.unique(df_all["td"])

rl_BAT = RL[np.argmin(np.abs(RL - 0.2))]
td_BAT = TD[np.argmin(np.abs(TD - 0.6))]


# fig, ax = plt.subplots(3, 2, sharey="row", figsize=(7.2, 4), dpi=300)


# plot hourly generation profile
def get_intervals(data, num_intervals):
    sorted_data = np.sort(data)
    mean_ind = np.argmin(np.abs(sorted_data - np.mean(data)))

    intrvls = np.linspace(0.01, 0.99, num_intervals)
    intervals = np.zeros([num_intervals, 2])
    for i, interval in enumerate(intrvls):
        frac_in = interval
        low_ind = mean_ind + int(frac_in / 2 * len(data))
        low_ind = np.min([low_ind, len(data) - 1])
        high_ind = mean_ind - int(frac_in / 2 * len(data))
        high_ind = np.max([high_ind, 0])
        frac_captured = (high_ind - low_ind) / len(data)
        intervals[i, :] = [sorted_data[high_ind], sorted_data[low_ind]]
    return intervals


def LPF_ma(data, width):
    # width = 24 * 30 * 3
    if width <= 1:
        return data
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width]) / width
    std_dev = np.zeros(len(filtered_data))
    for i in range(len(std_dev)):
        std_dev[i] = np.std(data[i : i + width])
    num_intervals = 10
    # intervals = np.zeros([len(filtered_data), num_intervals, 2])
    # for i in range(len(filtered_data)):
    #     intervals[i, :, :] = get_intervals(data[i : i + width], num_intervals)

    filtered_data = np.interp(
        np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), filtered_data
    )
    std_dev = np.interp(
        np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), std_dev
    )
    intervals_8760 = np.zeros([8760, num_intervals, 2])

    return filtered_data, std_dev, intervals_8760


ma_widths = [1, 24 * 30, 90 * 24]
labels = ["Hourly", "1-mo. avg.", "3-mo. avg."]


# maws = np.linspace(508, 730, 25).astype(int)
# # maws = ma_widths[1:]

# factor = 1e-3
# variabilities = np.zeros([len(maws), gen_profiles.shape[1]])
# for i, maw in enumerate(maws):
#     for j in range(gen_profiles.shape[1]):
#         ma, std, intervals = LPF_ma(factor * gen_profiles[:, j], maw)
#         variabilities[i, j] = np.max(ma) - np.min(ma)
#     # ma, std, intervals = LPF_ma(factor * gen_profiles[:, 1], maw)
#     # variabilities[i, 1] = np.max(ma) - np.min(ma)

# fig, ax = plt.subplots(1, 1)
# ax.plot(maws, variabilities, ".-")
# # ax.plot(maws, variabilities)

# plt.show()

hrs = np.linspace(0, 8760, 8760)


def time_difference(profile, time_diff):
    width = time_diff
    factor = 1e-3
    ma, std, intervals = LPF_ma(factor * profile, width)

    max_diff = np.max(np.abs(ma - np.roll(ma, time_diff)))
    return max_diff


# fig, ax = plt.subplots(1, 1)

# tx_diff = []
# ia_diff = []

# diffs = np.linspace(150, 800, 50).astype(int)
# for diff in diffs:
#     tx_diff.append(time_difference(gen_profiles[:, 0], diff))
#     ia_diff.append(time_difference(gen_profiles[:, 1], diff))

# ax.plot(diffs, tx_diff, ".-")
# ax.plot(diffs, ia_diff, ".-")


def plot_mean_std(profile, ax, plot_caret):
    width = ma_widths[-2]
    factor = 1e-3
    ma, std, intervals = LPF_ma(factor * profile[:, 0], width)
    ax.plot(hrs, ma, color="blue", label="TX 3-mo. avg.")
    ax.fill_between(
        np.linspace(0, 8760, 8760),
        ma - std,
        ma + std,
        zorder=0.95,
        color="blue",
        alpha=0.125,
    )

    if plot_caret:
        ax.plot(np.argmax(ma), np.max(ma), marker=7, color="blue", markersize=4)
        ax.plot(np.argmin(ma), np.min(ma), marker=6, color="blue", markersize=4)

    print(
        f"TX, max: {np.max(ma):.4f}, min: {np.min(ma):.4f}, seasonal range: {np.max(ma) - np.min(ma):.4f}"
    )

    ax.plot(hrs, ma - std, color="blue", linestyle="dashed", alpha=0.5)
    ax.plot(
        hrs, ma + std, color="blue", linestyle="dashed", alpha=0.5, label="std. dev."
    )

    ma, std, intervals = LPF_ma(factor * profile[:, 1], width)
    ax.plot(hrs, ma, color="orange", label="IA 3-mo. avg.")
    ax.fill_between(
        np.linspace(0, 8760, 8760),
        ma - std,
        ma + std,
        zorder=0.95,
        color="orange",
        alpha=0.125,
    )
    if plot_caret:
        ax.plot(np.argmax(ma), np.max(ma), marker=7, color="orange", markersize=4)
        ax.plot(np.argmin(ma), np.min(ma), marker=6, color="orange", markersize=4)

    print(
        f"IA, max: {np.max(ma):.4f}, min: {np.min(ma):.4f}, seasonal range: {np.max(ma) - np.min(ma):.4f}"
    )

    ax.plot(hrs, ma - std, color="orange", linestyle="dashed", alpha=0.5)
    ax.plot(
        hrs, ma + std, color="orange", linestyle="dashed", alpha=0.5, label="std. dev."
    )


if style == "paper":
    fig, ax = plt.subplots(1, 3, sharey="all", sharex="all", figsize=(7.2, 3))
    plot_mean_std(wind_profiles, ax[0], False)
    plot_mean_std(solar_profiles, ax[1], False)
    plot_mean_std(gen_profiles, ax[2], True)
    ax[1].legend()

    ax[0].set_title("Wind generation")
    ax[1].set_title("Solar generation")
    ax[2].set_title("Hybrid generation")
    ax[0].set_xticks(np.arange(0, 8760, 2400))
    ax[0].set_ylabel("MW")

    for axis in ax:
        axis.set_xlabel("Hour")

    fig.tight_layout()
    fig.savefig(
        save_path / "generation_statistics/wind_solar_hybrid_stats.png",
        format="png",
        dpi=300,
    )

elif style == "pres":

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.5), layout="constrained")

    factor = 1e-3

    profile = factor * gen_profiles[:, 1]


    widths = [
        2,
        24,
        24 * 7,
        24 * 30,
        24 * 30 * 3,
        # 24 * 30 * 6,
    ]
    lws = [0.25, 0.5, 0.75, 1, 1.5, 2]
    alphas = [0.125, 0.25, 0.5, 0.75, 0.9, 1]

    prof_mean = np.mean(profile)*np.ones(len(profile))

    width = 30
    prof_kwargs = {"color": "blue"}

    ax.plot(profile, linewidth=0.5, alpha=0.125, **prof_kwargs, label="Hourly")
    # for i, width in enumerate(widths):

    width = 24*30*3
    ma, std, intervals = LPF_ma(profile, width)
    ax.fill_between(np.linspace(0, 8760, 8760), prof_mean, ma, color="blue", alpha=.125)
    ax.plot(prof_mean, linewidth=1, alpha=1, linestyle="dashdot", **prof_kwargs, label="Mean")
    ax.hlines([np.min(ma), np.max(ma)], 0, len(profile), linestyle="dashed", color="black", linewidth=.5)
    ax.plot(ma, linewidth=2, alpha=.9, **prof_kwargs, label="3-month avg.")
    

    

    ax.legend(loc="upper right")

    
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Energy generation [MWh]")

    fig.savefig(FM.plot_path / "presentation_plots" / "variability_no_ylim.png", format="png")

    ax.set_ylim([2e2, 3.75e2])
    fig.savefig(FM.plot_path / "presentation_plots" / "variability_ylim.png", format="png")


    plt.close(fig)

    fig, ax = plt.subplots(6, 1, sharey="col", dpi=150)

    prof = profile - np.mean(profile)
    ax[0].plot(prof)
    

    total = 0

    for (
        i,
        width,
    ) in enumerate(np.flip(widths)):
        
        # ma, std, intervals = LPF_ma(prof, width)
        ma, std, intervals = LPF_ma(prof, width)
        ax[i + 1].plot(prof-ma, linewidth=.5, alpha=.5, color="orange")
        ax[i + 1].plot(ma, linewidth=lws[0], alpha=alphas[i + 1], **prof_kwargs)
        storage = np.cumsum(prof-ma).max() - np.cumsum(prof-ma).min()
        ax[i + 1].set_ylabel(f"{storage:.2e}")

        total += storage

        # prof = prof - ma

    fig.subplots_adjust(hspace=0)

    []


[]
