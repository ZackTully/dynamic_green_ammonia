# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PatchCollection
from pathlib import Path
import seaborn

# %%

line_kwargs = {}
subplot_kwargs = {"figsize": (15, 10)}
# subplot_kwargs = {"figsize": (398.3386/72, 236.07486/72), "dpi":500}

# %%
# gen_profiles = np.load(
# "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy"
# ).T
gen_profiles = np.load(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/hopp_sweep/H2_gen.npy"
).T
wind_profiles = np.load(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/hopp_sweep/wind_gen.npy"
).T
solar_profiles = np.load(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/hopp_sweep/solar_gen.npy"
).T
df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep.pkl")
df_full = pd.read_csv(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/DL_runs/full_sweep_main_df.csv"
)

lats = np.unique(df_all["lat"])

# df = df_all[df_all["lat"] == lats[0]]
df = df_all[df_all["lat"] == lats[1]]

splits = np.unique(df["split"])

RL = np.unique(df_all["rl"])
TD = np.unique(df_all["td"])

rl_BAT = RL[np.argmin(np.abs(RL - 0.2))]
td_BAT = TD[np.argmin(np.abs(TD - 0.6))]

# %%


def gp_smoothness(gen_profile):
    if mean_norm:
        smoothness = np.abs(gen_profile - np.roll(gen_profile, 1)) / np.mean(
            gen_profile
        )
    # smoothness = np.abs(gen_profile - np.roll(gen_profile, 1)) / np.max(gen_profile)
    else:
        smoothness = np.abs(gen_profile - np.roll(gen_profile, 1))
    smoothness = np.sum(smoothness)
    return smoothness


def gp_stats(gen_profile):
    locs = np.linspace(0, 100, 6)
    # locs = np.array([30, 50, 70])
    # locs = np.array([0, 30, 40, 50, 60, 70, 100])
    if mean_norm:
        percs = np.array([np.percentile(gen_profile, loc) for loc in locs]) / np.mean(
            gen_profile
        )
        gp_mean = np.mean(gen_profile)
        gp_max = np.max(gen_profile) / np.mean(gen_profile)
    else:
        percs = np.array([np.percentile(gen_profile, loc) for loc in locs])
        gp_mean = np.mean(gen_profile)
        gp_max = np.max(gen_profile)
    return percs, locs, gp_mean, gp_max


# for mean_norm in [False, True]:
for mean_norm in [False]:
    fig, ax = plt.subplots(
        3, len(np.unique(df_all["year"])), sharex="col", sharey="row", **subplot_kwargs
    )
    if len(np.unique(df_all["year"])) == 1:
        ax = np.array([[axis, None] for axis in ax])
    for j, year in enumerate(np.unique(df_all["year"])):
        splits = []
        percentiles = []
        smooths = []
        means = []
        maxs = []
        storage_caps = []

        # probably should separate years as well
        for i in range(np.shape(gen_profiles)[1]):
            if ((df_all[df_all["gen_ind"] == i]["lat"] == lats[0]).all()) & (
                (df_all[df_all["gen_ind"] == i]["year"] == year).all()
            ):
                split = np.unique(df_all[df_all["gen_ind"] == i]["split"])[0]
                sm = gp_smoothness(gen_profiles[:, i])
                percs, locs, gp_mean, gp_max = gp_stats(gen_profiles[:, i])

                BAT_storage = df_all[
                    (df_all["rl"] == rl_BAT)
                    & (df_all["td"] == td_BAT)
                    & (df_all["gen_ind"] == i)
                ]["storage_cap_kg"].iloc[0]
                splits.append(split)
                percentiles.append(percs)
                smooths.append(sm)
                means.append(gp_mean)
                maxs.append(gp_max)
                storage_caps.append(BAT_storage)

        splits = np.array(splits)
        reorder = np.argsort(splits)
        splits = splits[reorder]

        # splits = np.array(splits)[reorder]
        smooths = np.array(smooths)[reorder]
        percentiles = np.stack(percentiles)[reorder, :]
        means = np.array(means)[reorder]
        maxs = np.array(maxs)[reorder]
        storage_caps = np.array(storage_caps)[reorder]

        ax[0, j].plot(splits, smooths, color="black")
        ax[0, j].set_title(f"{year}")

        # ax[1,j].plot(splits, percentiles, color="blue")

        for i in range(np.shape(percentiles)[1]):
            if i == 0:
                continue
            ax[1, j].fill_between(
                splits,
                percentiles[:, i - 1],
                percentiles[:, i],
                color=cm.plasma(i / np.shape(percentiles)[1]),
            )
            # ax[1,j].fill_between(splits, percentiles[:,i-1]/means, percentiles[:,i]/means, color=cm.plasma(i/np.shape(percentiles)[1]))
            # ax[1,j].fill_between(splits, percentiles[:,i-1] / (means / means[0]), percentiles[:,i] / (means / means[0]), color=cm.plasma(i/np.shape(percentiles)[1]))

        if mean_norm:
            ax[1, j].plot(splits, means / means, linestyle="dotted", color="black")
        else:
            ax[1, j].plot(splits, means, linestyle="dotted", color="black")
        # ax[1,j].plot(splits, means /  (means / means[0]))
        ax[1, j].plot(splits, maxs, linestyle="dashed", color="black")
        # ax[1,j].set_title("percentiles")

        # ax[1,j].set_ylim([-1000, 20000])

        if mean_norm:
            ax[1, j].set_ylim([-0.25, 5])
            ax[2, j].plot(splits, storage_caps / means, color="purple")
        else:
            ax[2, j].plot(splits, storage_caps, color="purple")
        # ax[2,j].set_title("storage capacity \nnormalized by AEP")
        ax[2, j].set_xlabel("Wind capacity as \npercentage of 1GW")

    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0, 0].set_ylabel("smoothness")
    ax[1, 0].set_ylabel("gen. stats")
    if mean_norm:
        ax[2, 0].set_ylabel("Capacity/mean")
    else:
        ax[2, 0].set_ylabel("Capacity")

    # fig.suptitle(f"Normalized by mean: {mean_norm}")

# %%

year = 2012
splits = np.unique(df_all["split"])
# splits = splits[[0, 4, 9]]

fig, ax = plt.subplots(len(splits), 3, sharex="col", sharey="all", **subplot_kwargs)
# fig2, ax2 = plt.subplots(len(splits), 1, sharex='col', sharey='col', **subplot_kwargs)

for i in range(len(splits)):
    gen_ind = np.unique(
        df_all[
            (df_all["year"] == year)
            & (df_all["split"] == splits[i])
            & (df_all["lat"] == lats[0])
        ]["gen_ind"]
    )[0]
    gp = gen_profiles[:, gen_ind]
    gp_day = np.reshape(gp, (365, 24))
    gp_mean = np.mean(gp_day, axis=0)
    ax[i, 0].plot(gp_day.T, color="blue", linewidth=0.125, alpha=0.25)
    # ax[i, 0].set_ylabel(f"{splits[i]}% wind")
    ax[i, 0].set_ylabel("Generation")
    ax[i, 1].plot(gp_mean, color="black", linestyle="dashed")
    # ax[i, 1].text(0, ax[0, 0].get_ylim()[1] * 0.8, f"Total gen: {np.sum(gp):.2e}")
    ax[i, 1].text(0, np.max(gen_profiles) * 0.8, f"{splits[i]}%wind")

    seaborn.kdeplot(y=gp, ax=ax[i, 2], fill=True)
    # ax[i,2].yaxis.label_right()
    ax[i, 2].set_ylabel("")
    ax[i, 2].invert_xaxis()
    ax[i, 2].grid(True, axis="y", linewidth=0.25)
    ax[i, 1].grid(True, axis="y", linewidth=0.25)
    ax[i, 0].grid(True, axis="y", linewidth=0.25)


fig.subplots_adjust(hspace=0, wspace=0)
ax[0, 0].set_title("Daily generation profile")
ax[0, 1].set_title("Mean daily generation profile")
ax[0, 2].set_title("Yearly generation distribution")

# %%


gen_ind = 5
gp = gen_profiles[:, gen_ind]

df = df_full.loc[list(df_all["gen_ind"] == gen_ind)]

mins = df["H2_storage.min_demand"].to_numpy()
maxs = df["H2_storage.max_demand"].to_numpy()
caps = df["H2_storage.capacity_kg"].to_numpy()
reorder = np.argsort(mins)
mins = mins[reorder]
maxs = maxs[reorder]
caps = caps[reorder]


x, y = seaborn.kdeplot(gp).get_lines()[0].get_data()
dx = x[1] - x[0]

# %%

fig, ax = plt.subplots(3, 2, **subplot_kwargs)

ax[0, 1].plot(x, y)

n_bins = 50

ax[0, 0].hist(gp, bins=n_bins, density=True)
ax[0, 0].plot(x, y, color="black", alpha=0.5, linewidth=0.5)
ax[0, 0].set_ylim(ax[0, 1].get_ylim())

captured = []
bar_cap = []

ax[1, 1].plot(x, y)
n, bins, patches = ax[1, 0].hist(gp, bins=n_bins, density=True, alpha=0.25)
dbin = bins[1] - bins[0]
ax[1, 0].set_ylim(ax[1, 1].get_ylim())
ax[1, 0].set_xlim(ax[1, 1].get_xlim())

for i in range(len(mins)):
    center = (mins[i] + maxs[i]) / 2
    density_inds = np.where((mins[i] < x) & (x < maxs[i]))
    captured.append(np.sum(y[density_inds] * dx))

    # ax[1].plot([mins[i], center, maxs[i]], [0,0,0], 'k.')
    ax[1, 1].fill_between(
        x[density_inds],
        np.zeros(len(density_inds)),
        y[density_inds],
        color=cm.plasma(i / len(mins)),
    )

    bar_captured = 0

    bar_inds = np.where((mins[i] < bins) & (bins < maxs[i]))[0]
    if len(bar_inds) > 0:
        for j in range(len(bar_inds)):
            patches[bar_inds[j]].set_facecolor(cm.plasma(i / len(mins)))
            patches[bar_inds[j]].set_alpha(1)
            bar_captured += n[bar_inds[j]] * dbin

    bar_cap.append(bar_captured)


captured = np.array(captured)
bar_cap = np.array(bar_cap)
ax[2, 1].plot(1 - captured)
ax21 = ax[2, 1].twinx()
ax21.plot(caps, linestyle="dashed")

ax[2, 0].plot(1 - bar_cap)
ax20 = ax[2, 0].twinx()
ax20.plot(caps, linestyle="dashed")

# %%

df = df_all[
    (df_all["year"] == 2012)
    & (df_all["lat"] == lats[0])
    & (df_all["rl"] == rl_BAT)
    & (df_all["td"] == td_BAT)
]

# gen_inds = np.unique(df_all[(df_all["year"] == 2012) & (df_all["lat"] == lats[0])]["gen_ind"])
# splits = np.unique(df_all[(df_all["year"] == 2012) & (df_all["lat"] == lats[0])]["split"])
gen_inds = df["gen_ind"].to_numpy()
splits = df["split"].to_numpy()
reorder = np.argsort(splits)

gen_inds = gen_inds[reorder]
splits = splits[reorder]

gen_inds = gen_inds[::2]
splits = splits[::2]

fig, ax = plt.subplots(
    len(gen_inds), 4, sharey="col", sharex="col", dpi=400, **subplot_kwargs
)
ax1_twins = []

ax[0, 0].set_title("Probability density")
ax[-1, 0].set_xlabel("Generation")

ax[0, 1].set_title("Fraction of generation in operating range")
# ax[0, 1].invert_xaxis()
ax[-1, 1].set_xlabel("Turndown ratio")

fig.subplots_adjust(hspace=0)
storage_max = np.max(df_all["storage_cap_kg"])

ylim_top = []


def calc_pearson(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    r_xy = (np.sum((X - x_bar) * (Y - y_bar))) / (
        np.sqrt(np.sum((X - x_bar) ** 2)) * np.sqrt(np.sum((Y - y_bar) ** 2))
    )
    return r_xy


for k, gen_ind in enumerate(gen_inds):
    r_xy = calc_pearson(wind_profiles[:, gen_ind], solar_profiles[:, gen_ind])
    ax[k, 3].text(0, 0.9, f"{r_xy:.3f}")
    gp = gen_profiles[:, gen_ind]
    gp_sort = np.sort(gp)
    gp_sum = np.sum(gp)
    hrs = np.linspace(0, 8759, 8760)

    TD = np.linspace(0, 1, 500)
    plot_TDs = TD[:: int(len(TD) / 5)]

    g_bar = np.max(gp)
    mu = np.mean(gp)

    A = np.array([[1, -g_bar], [1, -mu]])
    b = np.array([0, mu])
    coeffs = np.linalg.inv(A) @ b

    a = coeffs[0]
    b = coeffs[1]

    d_max = lambda TD: a / (TD + b)
    d_min = lambda TD: TD * d_max(TD)

    # n_bins = 250
    # n, bins, patches = ax[k, 0].hist(gp, bins=n_bins, density=True, cumulative=True, alpha=0.25)
    # dbin = bins[1] - bins[0]
    percent_cap = np.zeros(len(TD))
    # ax[k,0].set_ylim([0, n[np.argsort(n)[-3]]])
    # ylim_top.append(n[np.argsort(n)[-7]])

    # ax[k,0].plot(gp_sort / np.linspace(1, gp_sort[-1], len(gp_sort)))
    # ax[k,0].plot(gp_sort[:-1], (np.roll(gp_sort, -1)-gp_sort)[:-1])
    # ax[k,0].plot(np.cumsum(gp_sort))

    dist = (1 / 8760) / (np.roll(gp_sort, -1) - gp_sort)

    ax[k, 0].plot(gp_sort[:-1], dist[:-1])
    ax[k, 0].set_ylim([0, 0.2])

    # ax[k,3].plot(gp_sort, hrs)

    for i in range(len(TD)):
        sorted_inds = np.where((gp_sort > d_min(TD[i])) & (gp_sort < d_max(TD[i])))[0]
        captured = 0
        # bar_inds = np.where((d_min(TD[i]) <= bins[0:-1]) & (bins[0:-1] < d_max(TD[i])))[
        #     0
        # ]
        # if len(bar_inds) > 0:
        #     for j in range(len(bar_inds)):
        #         if ((TD[i]* 100) % 5 == 0):
        #             patches[bar_inds[j]].set_facecolor(cm.plasma(i / len(TD)))
        #             patches[bar_inds[j]].set_alpha(1)
        #         captured += n[bar_inds[j]] * bins[j] * dbin / np.mean(gp)

        captured = np.sum(gp_sort[sorted_inds]) / gp_sum
        percent_cap[i] = captured

        ax[k, 1].plot(TD[i], captured, marker="o", color=cm.plasma(i / len(TD)))

        if TD[i] in plot_TDs:
            ax[k, 3].fill_between(
                gp_sort[sorted_inds],
                y1=np.zeros(len(sorted_inds)),
                y2=sorted_inds / 8760,
                color=cm.plasma(i / len(TD)),
            )

    # ax[k,0].set_ylabel("probability density")
    # ax[k,0].set_xlabel("generation")

    ax[k, 1].plot(TD, percent_cap, zorder=0.9, color="black", linewidth=0.5, alpha=0.5)

    axt = ax[k, 1].twinx()

    df_sim = df_all[
        (df_all["split"] == splits[k])
        & (df_all["rl"] == rl_BAT)
        & (df_all["lat"] == lats[0])
    ]
    axt.plot(df_sim["td"].to_numpy(), df_sim["storage_cap_kg"].to_numpy(), ".-")
    axt.set_ylim([0, storage_max])

    # axt.invert_xaxis()

    # ax[k,2].invert_xaxis()
    ax[k, 2].plot(TD, percent_cap)

    first_diff = (np.roll(percent_cap, -1) - percent_cap) / (np.roll(TD, -1) - TD)
    ax[k, 2].plot(TD[:-1], first_diff[:-1])

    # deriv_order = 1
    # ax[k, 1].plot(
    #     TD[:-deriv_order],
    #     (
    #         (percent_cap - np.roll(percent_cap, -deriv_order))
    #         / (TD[1] - TD[0]) ** deriv_order
    #     )[:-deriv_order],
    # )
    # deriv_order = 2
    # ax[k, 1].plot(
    #     TD[:-deriv_order],
    #     (
    #         (percent_cap - 2 * np.roll(percent_cap, -1) + np.roll(percent_cap, -2))
    #         / (TD[1] - TD[0]) ** deriv_order
    #     )[:-deriv_order],
    # )

    # tds = np.unique(df_td["td"])
    # # df_td[df_td["gen_ind"] == gen_ind]["storage_cap_kg"]
    # axt = ax[k,1].twinx()
    # axt.plot(tds, df_td[df_td["gen_ind"] == gen_ind]["storage_cap_kg"])

    ax[k, 1].text(1, 0.9, f"wind/pv split: {splits[k]} %")

    # ax[k,1].set_xlabel("Turndown ratio")
    # ax[k,1].set_ylabel("Fraction of generation in operating range")

    # fig.suptitle(df_all[df_all["gen_ind"] == gen_ind]["hopp_input"].iloc[0])
    print(df_all[df_all["gen_ind"] == gen_ind]["hopp_input"].iloc[0])

# ax[0, 1].set_ylim([-.1, 1.1])
# ax[0,1].set_ylim([-1, 1])
# ax[0, 0].set_ylim([0, np.mean(np.array(ylim_top))])

# %%

fake_data = False

df = df_all[
    (df_all["year"] == 2012)
    & (df_all["lat"] == lats[0])
    & (df_all["rl"] == rl_BAT)
    & (df_all["td"] == td_BAT)
]

if fake_data:
    gen_inds = np.arange(0, 5, 1)
    # splits = -1 * np.ones(len(gen_inds))
    splits = ["NA"] * len(gen_inds)
else:
    gen_inds = df["gen_ind"].to_numpy()
    splits = df["split"].to_numpy()
    reorder = np.argsort(splits)

    gen_inds = gen_inds[reorder]
    splits = splits[reorder]

    inds = np.linspace(0, len(gen_inds) - 1, 5).astype(int)

    # gen_inds = gen_inds[::2]
    # splits = splits[::2]
    gen_inds = gen_inds[inds]
    splits = splits[inds]

fig, ax = plt.subplots(
    len(gen_inds), 2, sharey="col", sharex="col", dpi=400, figsize=(9, 6)
)

ax[0, 0].set_title("Cumulative probability density")
ax[-1, 0].set_xlabel("Generation")

ax[0, 1].set_title("Fraction of gen. inside operating range")
ax[0, 1].invert_xaxis()
# ax[0,2].invert_xaxis()
ax[-1, 1].set_xlabel("Turndown ratio")


storage_max = np.max(
    df_all[(df_all["rl"] == RL[-2]) & (df_all["lat"] == lats[0])]["storage_cap_kg"]
)

ylim_top = []


def calc_pearson(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    r_xy = (np.sum((X - x_bar) * (Y - y_bar))) / (
        np.sqrt(np.sum((X - x_bar) ** 2)) * np.sqrt(np.sum((Y - y_bar) ** 2))
    )
    return r_xy


for k, gen_ind in enumerate(gen_inds):
    if fake_data:
        spread = 2 + (len(gen_inds) - k - 1) * 20
        gp = 100 * np.ones(8760) + np.linspace(-spread, spread, 8760)
        gp[0:5] = np.linspace(0, 1, 5)
        gp[-5:] = np.linspace(100 + spread, 200, 5)
    else:
        gp = wind_profiles[:, gen_ind] + solar_profiles[:, gen_ind]
        # gp = gen_profiles[:, gen_ind]
        r_xy = calc_pearson(wind_profiles[:, gen_ind], solar_profiles[:, gen_ind])
        ax[k, 1].text(1, 0.6, f"PC: {r_xy:.3f}")
        ax[k, 1].text(1, 0.4, f"var.: {np.mean(np.abs(gp - np.mean(gp))):.3f}")
        mean_cap = np.mean(df_all[(df_all["gen_ind"] == gen_ind)]["storage_cap_kg"])
        ax[k, 1].text(1, 0.2, f"MC: {mean_cap:.0f}")
        # ax[k,1].text(1, .2, f'cap.: {df_all[(df_all["td"] == 0.5) & (df_all["rl"] == 0.2) & (df_all["gen_ind"] == gen_ind)]["storage_cap_kg"].iloc[0]:.0f}')

    gp_sort = np.flip(np.sort(gp))
    gp_sum = np.sum(gp)
    hrs = np.linspace(0, 8759, 8760)

    TD = np.linspace(0, 1, 50)
    plot_TDs = TD[np.linspace(0, len(TD) - 1, 10).astype(int)]

    g_bar = np.max(gp)
    mu = np.mean(gp)

    A = np.array([[1, -g_bar], [1, -mu]])
    b = np.array([0, mu])
    coeffs = np.linalg.inv(A) @ b

    a = coeffs[0]
    b = coeffs[1]

    d_max = lambda TD: a / (TD + b)
    d_min = lambda TD: TD * d_max(TD)
    percent_cap = np.zeros(len(TD))
    # ax[k,0].plot(gp_sort, hrs/8760)

    for i in range(len(TD) - 1):
        sorted_inds = np.where((gp_sort > d_min(TD[i])) & (gp_sort < d_max(TD[i])))[0]
        captured = np.sum(gp_sort[sorted_inds]) / gp_sum
        # captured = np.sum(gp_sort[:sorted_inds[0]] - d_max(TD[i]) + d_min(TD[i])) + np.sum(gp_sort[sorted_inds[-1]:])
        percent_cap[i] = captured
        ax[k, 1].plot(TD[i], captured, marker="o", color=cm.plasma(i / len(TD)))
        if TD[i] in plot_TDs:
            ax[k, 0].fill_between(
                gp_sort[sorted_inds],
                y1=np.zeros(len(sorted_inds)),
                y2=sorted_inds / 8760,
                color=cm.plasma(i / len(TD)),
            )

    # sorted_inds = np.where((gp_sort > d_min(TD[0])) & (gp_sort < d_max(TD[0])))[0]
    # dy = (np.roll(sorted_inds, 1) - sorted_inds)[1:]
    # dx = (np.roll(gp_sort[sorted_inds], 1) - gp_sort[sorted_inds])[1:]

    # prob_dens = -dy/dx
    # prob_dens[prob_dens == np.inf] = 0
    # prob_dens[prob_dens == np.nan] = 0

    # ma_width = 800
    # ret = np.cumsum(prob_dens)
    # ret[ma_width:] = ret[ma_width:] - ret[:-ma_width]
    # prob_dens_ma = ret[ma_width-1:]/ma_width

    # ax[k,0].plot(gp_sort[sorted_inds][ma_width:], prob_dens_ma)

    ax[k, 1].plot(TD, percent_cap, zorder=0.9, color="black", linewidth=0.5, alpha=0.5)
    # if not fake_data:
    #     axt = ax[k,1].twinx()
    #     df_sim = df_all[(df_all["split"] == splits[k]) & (df_all["rl"] == RL[-2]) & (df_all["lat"] == lats[0])]
    #     axt.plot(df_sim["td"].to_numpy(), df_sim["storage_cap_kg"].to_numpy(), '.-')
    # axt.set_ylim([0, 1])
    # ax[k,2].plot(TD, percent_cap)

    # first_diff = (np.roll(percent_cap, - 1) - percent_cap) / (np.roll(TD, -1) - TD)
    # ax[k,2].plot(TD[:-1], first_diff[:-1])

    # ax[k, 2].plot(TD, percent_cap/TD)

    ax[k, 0].set_ylim([0, 1.15])
    # ax[k, 0].set_ylim([0, 20])
    ax[k, 0].set_yticks([0.25, 0.75], [], minor=True)
    ax[k, 0].set_yticks([0, 0.5, 1], [0, 0.5, 1], minor=False)
    # ax[k,0].spines[["top", "right"]].set_visible(False)
    ax[k, 0].spines[["top", "right"]].set_alpha(0.1)
    ax[k, 0].grid(True, axis="y", which="both", alpha=0.25)
    ax[k, 0].set_axisbelow(True)

    # ax[k,1].set_ylim([0, 1.15])
    ax[k, 1].set_yticks([0.25, 0.75], [], minor=True)
    ax[k, 1].set_yticks([0, 0.5, 1], [0, 0.5, 1], minor=False)
    ax[k, 1].spines[["top", "right"]].set_alpha(0.1)
    ax[k, 1].grid(True, axis="y", which="both", alpha=0.25)
    ax[k, 1].set_axisbelow(True)

    if not fake_data:
        ax[k, 1].text(1, 0.85, f"wind/pv split: {splits[k]} %")
    # print(df_all[df_all["gen_ind"] == gen_ind]["hopp_input"].iloc[0])

fig.tight_layout()

fig.subplots_adjust(hspace=0.075, top=0.9)

if fake_data:
    fig.suptitle("Example")
    fig.savefig(
        "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/TD_probability_example.png"
    )
else:
    # fig.suptitle(f"Texas, ramping limit: {np.unique(df['rl'])[0]:.2f}, turndown: {np.unique(df['td'])[0]:.2f}")
    fig.suptitle(f"Texas")
    fig.savefig(
        "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/TD_probability.png"
    )

# %%

gp = gen_profiles[:, 16]
t = np.linspace(0, 8759, len(gp))

sort_inds = np.flip(np.argsort(gp))

gp_sort = gp[sort_inds]

first_deriv = (-np.roll(gp_sort, -1) + gp_sort) / np.max(gp)

# plt.plot(t[sort_inds], gp, linewidth=.125)
# plt.plot(gp[sort_inds], t)
# plt.plot(np.linspace(0, np.max(gp), len(gp)), gp[sort_inds])

fig, ax = plt.subplots(1, 1)

ax.plot(np.linspace(0, np.max(gp), len(gp_sort) - 1), first_deriv[:-1])
ax.set_ylim([0, 0.005])


# %% Investigate similarity between different hybrid splits


df_HOPP = (
    df_full[
        [
            "HOPP.wind.capex",
            "HOPP.wind.opex",
            "HOPP.wind.rating_kw",
            "HOPP.wind.annual_energy",
            "HOPP.wind.CF",
            "HOPP.wind.LCOE",
            "HOPP.pv.capex",
            "HOPP.pv.opex",
            "HOPP.pv.rating_kw",
            "HOPP.pv.annual_energy",
            "HOPP.pv.CF",
            "HOPP.pv.LCOE",
            "HOPP.site.lat",
            "HOPP.site.lon",
            "HOPP.site.year",
        ]
    ]
    .drop_duplicates()
    .sort_values(by=["HOPP.site.lat", "HOPP.wind.rating_kw"])
)
x = np.linspace(0, len(df_HOPP) / 2, int(len(df_HOPP) / 2))

x = np.unique(df_all["split"])

fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex="col", sharey="row", dpi=400)
# fig.subplots_adjust(hspace=0)

ax[0, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.pv.rating_kw"],
    ".-",
    color="orange",
    label="PV",
)
ax[0, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.wind.rating_kw"],
    ".-",
    color="blue",
    label="Wind",
)
ax[0, 0].legend()
ax[0, 0].set_ylabel("Rating [kW]")
ax[1, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.wind.CF"],
    ".-",
    color="blue",
    label="PV",
)
ax[1, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.pv.CF"],
    ".-",
    color="orange",
    label="Wind",
)
ax[1, 0].legend(loc="right")
ax[1, 0].set_ylabel("CF [%]")
ax[2, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.pv.annual_energy"],
    ".-",
    color="orange",
    label="PV",
)
ax[2, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.wind.annual_energy"],
    ".-",
    color="blue",
    label="Wind",
)
ax[2, 0].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.wind.annual_energy"]
    + df_HOPP[df_HOPP["HOPP.site.lat"] == lats[0]]["HOPP.pv.annual_energy"],
    ".-",
    color="black",
    label="Hybrid",
)
ax[2, 0].legend(loc="right")
ax[2, 0].set_ylabel("Annual energy [kWh]")


ax[0, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.pv.rating_kw"],
    ".-",
    color="orange",
    label="PV",
)
ax[0, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.wind.rating_kw"],
    ".-",
    color="blue",
    label="Wind",
)
ax[0, 1].legend()
# ax[0,1].set_ylabel("Rating [kW]")
ax[1, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.wind.CF"],
    ".-",
    color="blue",
    label="Wind",
)
ax[1, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.pv.CF"],
    ".-",
    color="orange",
    label="PV",
)
ax[1, 1].legend()
# ax[1,1].set_ylabel("CF [%]")
ax[2, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.pv.annual_energy"],
    ".-",
    color="orange",
    label="PV",
)
ax[2, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.wind.annual_energy"],
    ".-",
    color="blue",
    label="Wind",
)
ax[2, 1].plot(
    x,
    df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.wind.annual_energy"]
    + df_HOPP[df_HOPP["HOPP.site.lat"] == lats[1]]["HOPP.pv.annual_energy"],
    ".-",
    color="black",
    label="Hybrid",
)
ax[2, 1].legend(loc="right")
# ax[2,1].set_ylabel("Energy [kWh]")


ax[2, 0].set_xlabel("Percent wind [%]")
ax[2, 1].set_xlabel("Percent wind [%]")

ax[0, 0].set_title("Texas")
ax[0, 1].set_title("Indiana")

fig.subplots_adjust(wspace=0.1)
fig.tight_layout()

fig.savefig(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/annual_energy.png"
)

# %% Pearon correlation coefficient


inds = (
    df_all[["lat", "gen_ind"]]
    .drop_duplicates()
    .sort_values(by=["lat", "gen_ind"])["gen_ind"]
    .to_numpy()
)
inds = inds[0:10][::2]
# inds = inds[10:]


X = wind_profiles[:, 3]
Y = solar_profiles[:, 3]
x_bar = np.mean(X)
y_bar = np.mean(Y)


def calc_pearson(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    r_xy = (np.sum((X - x_bar) * (Y - y_bar))) / (
        np.sqrt(np.sum((X - x_bar) ** 2)) * np.sqrt(np.sum((Y - y_bar) ** 2))
    )  # vanilla pearsons
    # r_xy = (np.sum((X - x_bar) * (Y - y_bar))) / np.sqrt(np.sum((X + Y - np.mean(X + Y))**2))\
    # r_xy = (np.sum((X - x_bar) * (Y - y_bar))) / len(X) # covariance
    return r_xy


complimentarity = np.zeros(len(inds))
fraction = np.zeros(len(inds))
other = np.zeros(len(inds))


fig, ax = plt.subplots(len(inds), 1, sharex="col", figsize=(12, 12))
fig.subplots_adjust(hspace=0)

for i, ind in enumerate(inds):
    X = wind_profiles[:, ind]
    Y = solar_profiles[:, ind]
    Z = X + Y

    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    z_bar = np.mean(Z)

    other[i] = np.sum(Z - z_bar)

    r_xy = calc_pearson(X, Y)
    complimentarity[i] = r_xy
    fraction[i] = np.sum(wind_profiles[:, ind]) / (
        np.sum(wind_profiles[:, ind]) + np.sum(solar_profiles[:, ind])
    )

    print(np.sum(gen_profiles[:, ind]))

    # if i == 5:
    if True:
        # ax.plot(wind_profiles[:,i] + solar_profiles[:,i], linewidth=.25, color="orange", alpha=.25)
        # ax.plot(solar_profiles[:,i], linewidth=.25, color="blue", alpha=.25)

        a_vals = np.linspace(np.min(Z), np.max(Z), 1000)
        z_range = np.max(Z) - np.min(Z)
        # a_vals = np.linspace(np.min(Z), z_bar + .3*z_range, 1000)
        var_a = np.zeros(len(a_vals))

        for j, a in enumerate(a_vals):
            # var_a[j] = 1/len(Z) * np.sum( (Z - a)**2 )

            # var_a[j] = np.mean((Z - a)**2) # MSE
            # var_a[j] = np.mean(np.abs(Z - a)) # 1-norm

            p = 2
            p_norm = (np.sum(np.abs(Z - a) ** p)) ** (1 / p)
            var_a[j] = p_norm  # p_norm

        ax[i].plot(a_vals, var_a, color=cm.plasma(i / len(inds)))
        ax[i].plot(
            a_vals[np.argmin(var_a)], var_a[np.argmin(var_a)], ".", color="orange"
        )
        print(f"{a_vals[np.argmin(var_a)]:.0f}, {np.max(Z)/2}")
        # ax.plot(z_bar, np.mean((Z - z_bar)**2), 'k.')
        ax[i].plot(z_bar, np.interp(z_bar, a_vals, var_a), "k.")

        axt = ax[i].twinx()
        z_inds = np.argsort(Z)
        z_sort = Z[z_inds]
        z_prob = np.cumsum(z_sort)
        deriv = np.zeros(len(z_prob))
        deriv_parts = 1

        for j in range(len(z_prob)):
            if j < deriv_parts:
                continue
            # deriv[j] = (z_prob[j+1] - z_prob[j-1])/( z_inds[j+1] - z_inds[j-1])

            # deriv[j] = 0.5 * (z_prob[j+1] - z_prob[j-1])/( z_inds[j+1] - z_inds[j-1]) + .5 * (z_prob[j+2] - z_prob[j-1])/( z_inds[j+2] - z_inds[j-1])

            der = 0
            for k in range(deriv_parts):
                der += (1 / deriv_parts) * (
                    (z_prob[j + k + 1] - z_prob[j - k - 1])
                    / (z_inds[j + k + 1] - z_inds[j - k - 1])
                )

            deriv[j] = der

            if j > (len(z_prob) - deriv_parts - 2):
                break

        # axt.plot(z_sort, deriv)
        # axt.set_ylim([-1000, 1000])
        axt.plot(Z[z_inds], np.cumsum(Z[z_inds]))

# ax.set_xlim([2e5, 3e5])

# fig, ax = plt.subplots(3, 1, figsize=(12,6))

# reorder = np.argsort(fraction)

# ax[0].plot(complimentarity[reorder])
# ax[1].plot(fraction[reorder])
# ax[1].set_xlabel("wind fraction")

# ax[2].plot(other[reorder])


# %%
