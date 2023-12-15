# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

gen_profiles = np.load(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy"
)

# gen_profiles = np.load("dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy")
df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep.pkl")


lats = np.unique(df_all["lat"])

# df = df_all[df_all["lat"] == lats[0]]
df = df_all[df_all["lat"] == lats[1]]


# %%

state_kwargs = {"color": "blue", "linewidth": 0.125, "alpha": 0.25}

# years = np.unique(df["year"])
RL = np.unique(df["rl"])
TD = np.unique(df["td"])


def get_soc(df):
    soc = np.stack(
        df[df["storage_cap_kg"] > 0]["storage_state"].to_numpy()
        / df[df["storage_cap_kg"] > 0]["storage_cap_kg"]
    )
    return soc


def parse_hoppp_input(string):
    strings = string[:-5].split("_")
    print(strings)
    year = int(strings[0][4:])
    lat = float(strings[1][3:])
    lon = float(strings[2][3:])
    split = int(strings[3][7:])
    return year, lat, lon, split


parse_hoppp_input(df.iloc[3]["hopp_input"])

storage_state = np.stack(df["storage_state"].to_numpy())
# soc = np.stack(df[df["storage_cap_kg"] > 0]["storage_state"].to_numpy() / df[df["storage_cap_kg"] > 0]["storage_cap_kg"])

# soc = []

# for i in range(len(df)):
# if df.iloc[i]["storage_cap_kg"] > 0:
# soc.append(df)


# %%


def plot_dist(data):
    fig, ax = plt.subplots(4, 1, figsize=(10, 15))
    ax[0].plot(gen_profiles.T, color="black", linewidth=0.125, alpha=0.125)

    mean_gen = np.mean(gen_profiles, axis=0)
    std_gen = np.std(gen_profiles, axis=0)
    std_up_gen = mean_gen + std_gen
    std_down_gen = mean_gen - std_gen

    ax[1].fill_between(
        np.linspace(0, len(gen_profiles.T), len(gen_profiles.T)),
        std_down_gen,
        std_up_gen,
        color="orange",
    )
    ax[1].plot(mean_gen, color="black")

    ax[2].plot(data, color="blue", linewidth=0.125, alpha=0.125)

    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    std_up = mean + std
    std_down = mean - std
    ax[3].fill_between(
        np.linspace(0, len(mean), len(mean)), std_down, std_up, color="orange"
    )
    ax[3].plot(mean, color="black")


plot_dist(storage_state.T)
# plot_dist(get_soc(df).T)

# %% do SVD on the data to find the underlying dimensions

data = storage_state.T

U, S, Vh = np.linalg.svd(data)

fig, ax = plt.subplots(1, 1)
ax.plot(np.cumsum(S / np.sum(S)))
ax.set_xlim([-5, 20])

# %%


trunc_dim = 1

S_trunc = np.copy(S)
S_trunc[trunc_dim:] = 0

data_trunc = np.dot(U[:, : len(S_trunc)] * S_trunc, Vh)

fig, ax = plt.subplots(3, 1, sharey="none", figsize=(15, 10))

ax[0].plot(data, **state_kwargs)
ax[1].plot(data_trunc, **state_kwargs)

for i in range(trunc_dim):
    ax[2].plot(U[:, i], color="black", linewidth=np.exp(-i / 10), alpha=np.exp(-i / 10))

print(f"{np.linalg.norm(data - data_trunc):e}")


# %% k means cluster storage state into groups

from sklearn.cluster import KMeans

errors = []

X = storage_state

clusters = np.concatenate(
    [np.arange(2, 15, 1), np.arange(16, 30, 2), np.arange(30, 350, 25)]
)

for n_clusters in clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    error = 0
    for i in range(n_clusters):
        data_inds = np.where(kmeans.labels_ == i)[0]
        for j in range(len(data_inds)):
            error += np.linalg.norm(kmeans.cluster_centers_[i] - data[:, data_inds[j]])

    # print(f"{error:e}")
    errors.append(error)

fig, ax = plt.subplots(1, 1)
ax.plot(errors, ".-")
# ax.set_yscale("log")
# ax.set_xscale("log")
ax.set_xlim(-2, 20)

# %%


n_clusters = 6

X = storage_state
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)


# fig, ax = plt.subplots(2, 1, figsize=(10,15))
# ax[0].plot(data, color="blue", linewidth=.125, alpha=.125)
# ax[0].plot(kmeans.cluster_centers_.T, color="black")

# mean = np.mean(data, axis=1)
# std = np.std(data, axis=1)
# std_up = mean + std
# std_down = mean - std
# ax[1].fill_between(np.linspace(0, len(mean), len(mean)), std_down, std_up, color="orange")
# ax[1].plot(mean, color="black")

if n_clusters < 20:
    fig1, ax1 = plt.subplots(
        n_clusters, 1, sharey="col", sharex="col", figsize=(10, 12)
    )
    fig2, ax2 = plt.subplots(
        n_clusters, 4, sharey="all", sharex="col", figsize=(10, 12)
    )

    data_max = np.max(data)

    for i in range(n_clusters):
        data_inds = np.where(kmeans.labels_ == i)[0]
        ax1[i].plot(data[:, data_inds], **state_kwargs)
        ax1[i].plot(kmeans.cluster_centers_[i].T, color="black")
        min_ind = np.argmin(kmeans.cluster_centers_[i])
        ax1[i].plot([min_ind, min_ind], [0, data_max], color="orange")

        df_cluster = df.iloc[list(data_inds)]
        rl_vals, rl_counts = np.unique(df_cluster["rl"], return_counts=True)
        td_vals, td_counts = np.unique(df_cluster["td"], return_counts=True)
        yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
        split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)

        ax2[i, 0].bar(rl_vals, rl_counts, width=1 / len(RL))
        # ax2[i,0].set_yscale("log")
        # ax2[i, 0].set_xscale("log")
        ax2[i, 1].bar(td_vals, td_counts, width=1 / len(TD))
        ax2[i, 2].bar(split_vals, split_counts, width=15)
        ax2[i, 3].bar(yr_vals, yr_counts)

        ax2[i, 0].set_ylabel(f"count, total={len(data_inds)}")

ax2[-1, 0].set_xlabel("rl")
ax2[-1, 1].set_xlabel("td")
ax2[-1, 2].set_xlabel("split")
ax2[-1, 3].set_xlabel("year")

fig2.subplots_adjust(wspace=0, hspace=0)
fig1.subplots_adjust(hspace=0)

# %%


years = np.unique(df["year"])
RL = np.unique(df["rl"])
TD = np.unique(df["td"])
split = np.unique(df["split"])


fig, ax = plt.subplots(len(years), 1, figsize=(10, 15))
for i in range(len(years)):
    state = np.stack(df[df["year"] == years[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_title(f"year: {years[i]}")


fig, ax = plt.subplots(len(RL), 1, figsize=(10, 15))
for i in range(len(RL)):
    state = np.stack(df[df["rl"] == RL[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_title(f"rl: {RL[i]:.5f}")


fig, ax = plt.subplots(len(TD), 1, figsize=(10, 15))
for i in range(len(TD)):
    state = np.stack(df[df["td"] == TD[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_title(f"td: {TD[i]:.2f}")

fig, ax = plt.subplots(len(split), 1, figsize=(10, 15))
for i in range(len(split)):
    state = np.stack(df[df["split"] == split[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_title(f"split: {split[i]}")


# %%
