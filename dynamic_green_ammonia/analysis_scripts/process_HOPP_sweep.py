# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.cluster import KMeans

# gen_profiles = np.load(
    # "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen_20231214.npy"
# ).T

gen_profiles = np.load("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy").T

# gen_profiles = np.load("dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy")
# df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep_20231214.pkl")

df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep.pkl")


lats = np.unique(df_all["lat"])

# df = df_all[df_all["lat"] == lats[0]]
df = df_all[df_all["lat"] == lats[1]]


state_kwargs = {"color": "blue", "linewidth": 0.125, "alpha": 0.25}

#%%

fig, ax = plt.subplots(2,1, figsize=(15, 10))
ax[0].plot(gen_profiles[:,0])
# 

# ax[0].set_xlim([100, 500])

sp = np.fft.fft(gen_profiles[:,0])
freq  = np.fft.fftfreq(np.shape(gen_profiles[:,0])[-1])
ax[1].plot(freq, np.abs(sp))
ax[1].set_ylim([-1e4, 1e6])
ax[1].set_xlim([0, .5])

max_inds = np.argsort(np.abs(sp))
num_inds = 20
ax[1].plot(freq[max_inds[-num_inds:]], [0]*num_inds, 'k.')


T_cutoff = 8760/(2**2) # cutoff period
f_cutoff = 1/T_cutoff
inds = np.where(np.abs(freq) < f_cutoff)[0]
print(len(inds))
sp_trunc = np.zeros(len(sp))
sp_trunc[inds] = sp[inds]
# sp_trunc = sp

# sp_trunc = np.zeros(len(sp))
# sp_trunc[max_inds[-num_inds:]] = sp[max_inds[-num_inds:]]

data_trunc = np.fft.ifft(sp_trunc)
ax[0].plot(data_trunc)

# ax[0].plot(np.cumsum(gen_profiles[:,0]))
# ax[0].plot(np.cumsum(np.roll(gen_profiles[:,0], 100)))
width = 1000
cs = np.cumsum(gen_profiles[:,0])
# ax[0].plot(np.linspace(width, 8760-width, int(8760-width)), (cs[width:] - cs[:-width])/width)
ax[0].plot(np.linspace(0, 8760, int(8760-width)), (cs[width:] - cs[:-width])/width)
# ax[0].plot(1/width * np.cumsum(gen_profiles[:,0] - np.roll(gen_profiles[:,0], width)) )

#%%


def LPF_ma(data):
    width = 500
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width])/width
    filtered_data = np.interp(np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760-width)), filtered_data)
    return filtered_data

def LPF(data):
    sp =  np.fft.fft(data)
    freq = np.fft.fftfreq(np.shape(data)[-1])
    T = 8760/(2**6)
    f = 1/T
    inds = np.where(np.abs(freq) < f)[0]
    sp_trunc = np.zeros(len(sp))
    sp_trunc[inds] = sp[inds]

    filtered_data = np.fft.ifft(sp_trunc)
    return filtered_data


gen_filtered = np.zeros(np.shape(gen_profiles))
for i in range(np.shape(gen_profiles)[1]):
    gen_filtered[:,i] = LPF_ma(gen_profiles[:,i])

# fig, ax = plt.subplots(2,1, figsize=(15, 10))
# ax[0].plot(gen_profiles, **state_kwargs)
# ax[1].plot(gen_filtered, color="blue", linewidth=.5)


#%% K means on gen profiles


n_clusters = 6

X = gen_filtered.T
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)

if n_clusters < 20:
    fig1, ax1 = plt.subplots(
        n_clusters, 1, sharey="col", sharex="col", figsize=(15, 10)
    )
    fig2, ax2 = plt.subplots(
        n_clusters, 4, sharey="all", sharex="col", figsize=(15, 10)
    )

    data_max = np.max(X)

    for i in range(n_clusters):
        data_inds = np.where(kmeans.labels_ == i)[0]
        ax1[i].plot(X[data_inds,:].T, color='blue', linewidth=.5, alpha=.5)
        ax1[i].plot(kmeans.cluster_centers_[i].T, color="black")
        min_ind = np.argmin(kmeans.cluster_centers_[i])
        # ax1[i].plot([min_ind, min_ind], [0, data_max], color="orange")

        df_cluster = df_all[df_all["gen_ind"].isin(data_inds)]
        gi_vals, gi_counts = np.unique(df_cluster["gen_ind"], return_counts=True)
        yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
        split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)

        ax2[i, 0].bar(gi_vals, gi_counts)
        ax2[i, 1].bar(yr_vals, yr_counts)
        ax2[i, 2].bar(split_vals, split_counts)

fig1.subplots_adjust(hspace=0)
fig1.suptitle("Generation profile clustered")

fig2.subplots_adjust(hspace=0, wspace=0)
ax2[-1,0].set_xlabel("generation indx")
ax2[-1,1].set_xlabel("year")
ax2[-1,2].set_xlabel("wind/pv split")
# %%



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


# %%


def plot_dist(data):
    fig, ax = plt.subplots(4, 1, sharex='col', figsize=(10, 10))
    ax[0].plot(gen_profiles, color="black", linewidth=0.125, alpha=0.125)
    ax[0].set_ylabel("all gen. profiles")

    mean_gen = np.mean(gen_profiles, axis=1)
    std_gen = np.std(gen_profiles, axis=1)
    std_up_gen = mean_gen + std_gen
    std_down_gen = mean_gen - std_gen

    ax[1].fill_between(
        np.linspace(0, len(gen_profiles), len(gen_profiles)),
        std_down_gen,
        std_up_gen,
        color="orange",
    )
    ax[1].plot(mean_gen, color="black")
    ax[1].set_ylabel("gen. stats")

    ax[2].plot(data, color="blue", linewidth=0.125, alpha=0.125)
    ax[2].set_ylabel("all storage profiles")

    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    std_up = mean + std
    std_down = mean - std
    ax[3].fill_between(
        np.linspace(0, len(mean), len(mean)), std_down, std_up, color="orange"
    )
    ax[3].plot(mean, color="black")
    ax[3].set_ylabel("storage stats")
    ax[3].set_ylim(ax[2].get_ylim())
    return fig, ax


fig, ax = plot_dist(storage_state.T)
fig.subplots_adjust(hspace=0)

# %% do SVD on the data to find the underlying dimensions

data = storage_state.T

U, S, Vh = np.linalg.svd(data)

fig, ax = plt.subplots(1, 1)
ax.plot(np.cumsum(S / np.sum(S)))
ax.set_xlim([-5, 20])
ax.set_title("Singular value cumulative fraction")

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
fig.suptitle(f"First {trunc_dim} singular values reconstruction")

# %% k means cluster storage state into groups



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
ax.set_xlabel("number of clusters")
ax.set_ylabel("Total error")

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
        ax1[i].text( min_ind+100, 0.9*data_max, f"min at {min_ind}")

        df_cluster = df.iloc[list(data_inds)]
        rl_vals, rl_counts = np.unique(df_cluster["rl"], return_counts=True)
        td_vals, td_counts = np.unique(df_cluster["td"], return_counts=True)
        yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
        split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)

        ax2[i, 0].bar(rl_vals, rl_counts, width=1 / len(RL))
        # ax2[i,0].set_yscale("log")
        # ax2[i, 0].set_xscale("log")
        ax2[i, 1].bar(td_vals, td_counts, width=1 / len(TD))
        ax2[i, 2].bar(split_vals, split_counts, width=8)
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


fig, ax = plt.subplots(len(years), 1, sharey="all", figsize=(10, 10))
for i in range(len(years)):
    state = np.stack(df[df["year"] == years[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_ylabel(f"year: {years[i]}")
fig.subplots_adjust(hspace=0)

fig, ax = plt.subplots(len(RL), 1, sharey="all", figsize=(10, 10))
for i in range(len(RL)):
    state = np.stack(df[df["rl"] == RL[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_ylabel(f"rl: {RL[i]:.5f}")
fig.subplots_adjust(hspace=0)

fig, ax = plt.subplots(len(TD), 1, sharey="all", figsize=(10, 10))
for i in range(len(TD)):
    state = np.stack(df[df["td"] == TD[i]]["storage_state"].to_numpy())
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_ylabel(f"td: {TD[i]:.2f}")
fig.subplots_adjust(hspace=0)

fig, ax = plt.subplots(len(split), 1, sharey="all", figsize=(10, 10))
for i in range(len(split)):
    state = np.stack(df[df["split"] == split[i]]["storage_state"].to_numpy())
    print(np.max(state))
    ax[i].plot(state.T, **state_kwargs)
    ax[i].set_ylabel(f"split: {split[i]}")
fig.subplots_adjust(hspace=0)

# %%
