import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from dynamic_green_ammonia.tools.file_management import FileMan

FM = FileMan()
FM.set_analysis_case("cases_check")

df_all, df_full = FM.load_sweep_data()
df_full.insert(1, "Case", len(df_full) * [""])
cost_excel = pd.ExcelFile(FM.costs_path / "cost_models.xlsx")

loc_info = pd.read_csv(FM.input_path / "location_info.csv")


# data_path = Path(__file__).parents[1] / "data" / "cases_check"
# data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"

# df_all = pd.read_pickle(data_path / "hopp_sweep.pkl")
# df_full = pd.read_csv(data_path / "full_sweep_main_df.csv")
wg = np.load(FM.data_path / "wind_gen.npy")
sg = np.load(FM.data_path / "solar_gen.npy")

cost_df = pd.read_csv(Path(__file__).parents[0] / "cost_models" / "ESG_locations.csv")
df_print = pd.read_csv(Path(__file__).parents[0] / "cost_models" / "df_print.csv")

# use these indices

# 10     TX inflexible
# 120    IA inflexible
# 94     TX BAT
# 204    IA BAT

# loc_case = [
#     "TX GS",
#     "TX CF",
#     "TX LCOH",
#     "TX Storage",
#     "TX Comp",
#     "TX CF_ST",
#     "IA GS",
#     "IA CF",
#     "IA LCOH",
#     "IA Storage",
#     "IA Comp",
#     "IA CF_ST",
# ]

# locations = np.array(
#     [
#         [34.22, -102.75],
#         [30.735, -102.457],
#         [32.257, -99.256],
#         [36.069, -102.819],
#         [29.556, -99.636],  # complimentarity
#         [33.364, -98.526],  # CF and storage
#         [42.55, -90.69],  # green steel
#         [41.817, -94.881],
#         [43.401, -91.220],
#         [41.494, -92.834],
#         [40.674, -91.425],  # complimentarity
#         [41.817, -94.882],  # CF and storage
#     ]
# )

# # pearson


# pearson = np.zeros(np.shape(wg)[0])
# df_list = []
# for i in range(np.shape(wg)[0]):
#     pearson[i] = np.cov([wg[i, :], sg[i, :]])[0, 1] / (
#         np.std(wg[i, :]) * np.std(sg[i, :])
#     )
#     # pearson_IA = np.cov([wg[1, :], sg[1, :]])[0, 1] / (np.std(wg[1, :]) * np.std(sg[1, :]))

#     lat_ind = np.where(locations[:, 0] == df_all.iloc[2 * i]["lat"])[0][0]
#     lon_ind = np.where(locations[:, 1] == df_all.iloc[2 * i]["lon"])[0][0]

#     # if lat_ind == lon_ind:
#     case = loc_case[lon_ind]

#     df_list.append(
#         pd.concat(
#             [
#                 df_all.iloc[2 * i][["lat", "lon"]],
#                 pd.Series({"pearson": pearson[i], "Case": case}),
#             ]
#         )
#     )

# df = pd.DataFrame(df_list)

hg = wg + sg


def plot_stats(ax, signal):
    ax.plot(np.linspace(0, 1, len(signal)), np.flip(np.sort(signal)), color="black")
    ax.grid(linewidth=0.5, alpha=0.5)


def LPF_ma(data, width):
    # width = 24 * 30 * 3
    if width <= 1:
        return data
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width]) / width
    std_dev = np.zeros(len(filtered_data))
    for i in range(len(std_dev)):
        std_dev[i] = np.std(data[i : i + width])
    std_dev = np.interp(
        np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), std_dev
    )

    return filtered_data, std_dev


def calc_variability(ax, signal, case):
    # ax.plot(signal)

    # ma_widths = np.linspace(30 * 24 * 3, 30, 5).astype(int)
    ma_widths = [20 * 24 * 3]

    filtered = []
    stds = []
    for maw in ma_widths:
        filt, std = LPF_ma(signal, maw)
        filtered.append(np.max(filt) - np.min(filt))
        stds.append(std)

    ax.plot(filtered, ".-")
    ax.text(0.25, 0.75, case, transform=ax.transAxes)
    # ax.plot(stds)


fig, ax = plt.subplots(np.shape(wg)[0], 3, sharex="all", sharey="all")
fig2, ax2 = plt.subplots(int(np.shape(wg)[0] / 4), 4, sharex="all", sharey="all")
fig2.subplots_adjust(hspace=0, wspace=0)
ax2 = np.reshape(ax2, [len(wg)])

for i in range(np.shape(wg)[0]):
    lat_ind = loc_info[loc_info["lat"] == df_all.iloc[2 * i]["lat"]].index
    lon_ind = loc_info[loc_info["lon"] == df_all.iloc[2 * i]["lon"]].index

    # if lat_ind == lon_ind:/
    case = f'{loc_info.loc[lon_ind]["loc"].iloc[0]} {loc_info.loc[lon_ind]["note"].iloc[0]}'

    plot_stats(ax[i, 0], wg[i, :])
    plot_stats(ax[i, 1], sg[i, :])
    plot_stats(ax[i, 2], hg[i, :])

    case_pearson = np.cov([wg[i, :], sg[i, :]])[0, 1] / (
        np.std(wg[i, :]) * np.std(sg[i, :])
    )

    cost_case = cost_df[cost_df["Case"] == f"{case} INF"]
    print_case = df_print[df_print["Case"] == f"{case} INF"]

    # d_min = df_print[df_print["Case"] == f"{case} BAT"]["H2_storage.min_demand"].iloc[0]
    # d_max = df_print[df_print["Case"] == f"{case} BAT"]["H2_storage.max_demand"].iloc[0]
    # ax[i, 2].plot([0, 0], [d_min, d_max])

    calc_variability(ax2[i], hg[i, :], case)

    ax[i, 0].text(
        0.6,
        0.75,
        f"CF: {print_case['HOPP.wind.CF'].iloc[0]/100:.2f}",
        transform=ax[i, 0].transAxes,
    )
    ax[i, 1].text(
        0.6,
        0.75,
        f"CF: {print_case['HOPP.pv.CF'].iloc[0]/100:.2f}",
        transform=ax[i, 1].transAxes,
    )
    ax[i, 0].text(
        0.6,
        0.6,
        f"AEP: {np.sum(wg[i,:]):.2e}",
        transform=ax[i, 0].transAxes,
    )
    ax[i, 1].text(
        0.6,
        0.6,
        f"AEP: {np.sum(sg[i,:]):.2e}",
        transform=ax[i, 1].transAxes,
    )

    cavern_cost = (
        cost_df[cost_df["Case"] == f"{case} INF"][["Cavern capex", "Cavern opex"]]
        .sum(axis=1)
        .iloc[0]
    )
    pipe_cost = (
        cost_df[cost_df["Case"] == f"{case} INF"][["Pipe capex", "Pipe opex"]]
        .sum(axis=1)
        .iloc[0]
    )

    ax[i, 2].text(
        0.6,
        0.75,
        f'Pipe LCOA: {cost_df[cost_df["Case"] == f"{case} INF"]["Pipe LCOA"].iloc[0]:.2f}',
        transform=ax[i, 2].transAxes,
    )
    ax[i, 2].text(0.6, 0.6, f"AEP {np.sum(hg[i,:]):.2e}", transform=ax[i, 2].transAxes)
    ax[i, 2].text(
        0.6, 0.45, f"Cavern $ {cavern_cost:.2e}", transform=ax[i, 2].transAxes
    )
    ax[i, 2].text(0.6, 0.3, f"Pipe $ {pipe_cost:.2e}", transform=ax[i, 2].transAxes)
    ax[i, 2].text(
        0.3,
        0.75,
        f"H2 cap. {print_case['H2_storage.capacity_kg'].iloc[0]:.2e}",
        transform=ax[i, 2].transAxes,
    )

    print(
        f'{        cost_df[cost_df["Case"] == f"{case} BAT"]["Pipe LCOA"].iloc[0]        * np.sum(hg[i, :]):.2e}'
    )

    ax[i, 0].set_ylabel(case)

ax[0, 0].set_title("Wind gen. duration curve")
ax[0, 1].set_title("PV gen. duration curve")
ax[0, 2].set_title("Hybrid gen. duration curve")

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0, left=0.05, bottom=0.025, right=0.99, top=0.975)


[]
