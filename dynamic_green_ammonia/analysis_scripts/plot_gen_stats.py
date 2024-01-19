import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from pathlib import Path
import calendar
from datetime import datetime
import seaborn

year = 2012
style = "paper"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")

# plt.rcParams["text.usetex"] = True


data_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
save_path = Path(__file__).parents[1] / "plots"


gen_profiles = np.load(data_path / "H2_gen.npy").T
wind_profiles = np.load(data_path / "wind_gen.npy").T
solar_profiles = np.load(data_path / "solar_gen.npy").T

gen_profiles = wind_profiles + solar_profiles


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


fig, ax = plt.subplots(3, 2, sharey="row", figsize=(7.2, 4), dpi=300)

# plot hourly generation profile


def LPF_ma(data, width):
    # width = 24 * 30 * 3
    if width <= 1:
        return data
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width]) / width
    std_dev = np.zeros(len(filtered_data))
    for i in range(len(std_dev)):
        std_dev[i] = np.std(data[i : i + width])

    filtered_data = np.interp(
        np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), filtered_data
    )
    std_dev = np.interp(
        np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760 - width)), std_dev
    )

    return filtered_data, std_dev


# ma_widths = [1, 24, 30 * 24, 90 * 24]
# ma_widths = [1]
ma_widths = [1, 24 * 30, 90 * 24]
labels = ["Hourly", "1-mo. avg.", "3-mo. avg."]
# ma_widths = np.round(np.linspace(1, 8750, 10)).astype(int)

# for i in range(len(ma_widths)):
#     for j in [0, 1]:
#         ax[0, j].plot(
#             LPF_ma(gen_profiles[:, j], ma_widths[i]),
#             color=cm.plasma(1 - i / len(ma_widths)),
#             linewidth=2 * (i + 1) / len(ma_widths),
#             label=labels[i],
#         )
# ax[0, 0].legend()
# ax[0, 1].legend()
# ax[0, 0].set_title("TX\nGeneration timeseries")
# ax[0, 1].set_title("IN\nGeneration timeseries")
# ax[0, 0].set_ylabel("Generation [kWh]")

# Generation duration curve - kind of like the load duration curve
# ax[1, 0].plot(np.flip(np.sort(gen_profiles[:, 0])))
# ax[1, 1].plot(np.flip(np.sort(gen_profiles[:, 1])))
# ax[1, 0].set_ylabel("Generation [kWh]")
# ax[1, 0].set_title("Generation duration curve")
# ax[1, 1].set_title("Generation duration curve")

for i in [0, 1]:
    data = []
    for mon in np.arange(1, 13, 3):
        days_in_month = calendar.monthrange(year, mon + 2)[1]
        start_ind = datetime(year, mon, 1).timetuple().tm_yday * 24
        end_ind = datetime(year, mon + 2, days_in_month).timetuple().tm_yday * 24
        data.append(gen_profiles[start_ind:end_ind, i])
    # ax[0, i].violinplot(data, np.arange(1, 5, 1), showmeans=True, showextrema=True)
    # ax[0, i].plot(
    # np.linspace(1, 4, 8760), LPF_ma(gen_profiles[:, i], ma_widths[-1]), zorder=0.95
    # )

    ma, std = LPF_ma(gen_profiles[:, i], ma_widths[-1])
    ax[0, i].fill_between(np.linspace(0, 8760, 8760), ma - std, ma + std, zorder=0.95)
    ax[0, i].plot(np.linspace(0, 8760, 8760), ma, color="orange")

    data = []
    for mon in np.arange(1, 13, 1):
        days_in_month = calendar.monthrange(year, mon)[1]
        start_ind = datetime(year, mon, 1).timetuple().tm_yday * 24
        end_ind = datetime(year, mon, days_in_month).timetuple().tm_yday * 24
        month_data = gen_profiles[start_ind:end_ind, i]
        data.append(month_data)
    ax[1, i].violinplot(data, np.arange(1, 13, 1), showmeans=True, showextrema=True)
    ma, std = LPF_ma(gen_profiles[:, i], ma_widths[-1])
    ax[1, i].plot(np.linspace(1, 12, 8760), ma, zorder=0.95)
    ax[2, i].boxplot(data)

    ax[0, i].set_xlabel("Quarter")
    ax[1, i].set_xlabel("Month")
    ax[2, i].set_xlabel("Month")

ax[0, 0].set_title("Texas generation")
ax[0, 1].set_title("Indiana generation")

# for i in [0, 1]:
#     gen_daily = np.reshape(gen_profiles[:, i], [365, 24]).T
#     daily_mean = np.mean(gen_daily, axis=1)
#     # ax[2, i].plot(gen_daily, linewidth=0.5, color="blue", alpha=0.25)
#     # ax[2, i].plot(daily_mean, color="black")
#     ax[2, i].imshow(gen_daily, aspect=3.5)

# ax[2, 0].set_yticks(np.arange(0, 24, 6))
# ax[2, 0].set_ylabel("Hour")
# ax[2, 0].set_xlabel("Day")
# ax[2, 1].set_xlabel("Day")

fig.tight_layout()


# print out information
df_TX = df_full[df_full["HOPP.site.lat"] == lats[0]].iloc[0]
df_IN = df_full[df_full["HOPP.site.lat"] == lats[1]].iloc[0]


# print(
# f'Electrolyzer CF: & {df_TX["H.CF"]/100:.2f} & {df_IN["HOPP.wind.CF"]/100:.2f} \\\\'
# )
print(
    f'Wind CF: & {df_TX["HOPP.wind.CF"]/100:.2f} & {df_IN["HOPP.wind.CF"]/100:.2f} \\\\'
)
print(f'PV CF: & {df_TX["HOPP.pv.CF"]/100:.2f} & {df_IN["HOPP.pv.CF"]/100:.2f} \\\\')

AEP_TX = df_TX["HOPP.wind.annual_energy"] + df_TX["HOPP.pv.annual_energy"]
AEP_IN = df_IN["HOPP.wind.annual_energy"] + df_IN["HOPP.pv.annual_energy"]

hybrid_CF_TX = AEP_TX / (
    8760 * (df_TX["HOPP.wind.rating_kw"] + df_TX["HOPP.pv.rating_kw"])
)
hybrid_CF_IN = AEP_IN / (
    8760 * (df_IN["HOPP.wind.rating_kw"] + df_IN["HOPP.pv.rating_kw"])
)

print(f"Hybrid CF: & {hybrid_CF_TX:.2f} & {hybrid_CF_IN:.2f} \\\\")
print(f"Hybrid AEP (GWh): & {AEP_TX/1e6:.0f} & {AEP_IN/1e6:.0f} \\\\")

LCOE_TX = (
    df_TX["HOPP.wind.LCOE"] * df_TX["HOPP.wind.annual_energy"]
    + df_TX["HOPP.pv.LCOE"] * df_TX["HOPP.pv.annual_energy"]
) / AEP_TX
LCOE_IN = (
    df_IN["HOPP.wind.LCOE"] * df_IN["HOPP.wind.annual_energy"]
    + df_IN["HOPP.pv.LCOE"] * df_IN["HOPP.pv.annual_energy"]
) / AEP_IN

print(
    f"Wind LCOE: & {df_TX['HOPP.wind.LCOE']:.2f} & {df_IN['HOPP.wind.LCOE']:.2f} \\\\"
)
print(f"PV LCOE: & {df_TX['HOPP.pv.LCOE']:.2f} & {df_IN['HOPP.pv.LCOE']:.2f} \\\\")
print(f"Hybrid LCOE: & {LCOE_TX:.2f} & {LCOE_IN:.2f} \\\\")
print(
    f"Annual $H_2$: & {df_TX['DGA.EL.H2_tot']:.2f} & {df_IN['DGA.EL.H2_tot']:.2f} \\\\"
)
print(
    f"Electrolyzer rating: & {df_TX['DGA.EL.P_EL_max']:.2f} & {df_IN['DGA.EL.P_EL_max']:.2f} \\\\"
)

fig.savefig(
    save_path / "generation_statistics/gen_stats_TX_IN.png", format="png", dpi=300
)
# plt.show()


hrs = np.linspace(0, 8760, 8760)


def plot_mean_std(profile, ax):
    factor = 1e-3
    ma, std = LPF_ma(factor * profile[:, 0], ma_widths[-1])
    ax.plot(hrs, ma, color="blue", label="TX 3-mo. avg.")
    ax.fill_between(
        np.linspace(0, 8760, 8760),
        ma - std,
        ma + std,
        zorder=0.95,
        color="blue",
        alpha=0.125,
    )
    ax.plot(hrs, ma - std, color="blue", linestyle="dashed", alpha=0.5)
    ax.plot(
        hrs, ma + std, color="blue", linestyle="dashed", alpha=0.5, label="std. dev."
    )

    ma, std = LPF_ma(factor * profile[:, 1], ma_widths[-1])
    ax.plot(hrs, ma, color="orange", label="IN 3-mo. avg.")
    ax.fill_between(
        np.linspace(0, 8760, 8760),
        ma - std,
        ma + std,
        zorder=0.95,
        color="orange",
        alpha=0.125,
    )
    ax.plot(hrs, ma - std, color="orange", linestyle="dashed", alpha=0.5)
    ax.plot(
        hrs, ma + std, color="orange", linestyle="dashed", alpha=0.5, label="std. dev."
    )


fig, ax = plt.subplots(1, 3, sharey="all", sharex="all", figsize=(7.2, 3))
plot_mean_std(wind_profiles, ax[0])
plot_mean_std(solar_profiles, ax[1])
plot_mean_std(gen_profiles, ax[2])
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
[]
