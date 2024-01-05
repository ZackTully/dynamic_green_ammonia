# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.cluster import KMeans

# gen_profiles = np.load(
    # "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen_20231214.npy"
# ).T

# gen_profiles = np.load("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/H2_gen.npy").T
gen_profiles = np.load("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/data/HOPP_sweep/H2_gen_20231218.npy").T

# gen_profiles = np.load("dynamic_green_ammonia/data/HOPP_sweep/hopp_sweep_gen.npy")
# df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep_20231214.pkl")

# df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep.pkl")
df_all = pd.read_pickle(Path(__file__).parents[1] / "data/HOPP_sweep/hopp_sweep_20231218.pkl")


lats = np.unique(df_all["lat"])

# df = df_all[df_all["lat"] == lats[0]]
# df = df_all[df_all["lat"] == lats[1]]
df = df_all


state_kwargs = {"color": "blue", "linewidth": 0.25, "alpha": 0.5}

##%%

# fig, ax = plt.subplots(2,1, figsize=(15, 10))
# ax[0].plot(gen_profiles[:,0])
# # 

# # ax[0].set_xlim([100, 500])

# sp = np.fft.fft(gen_profiles[:,0])
# freq  = np.fft.fftfreq(np.shape(gen_profiles[:,0])[-1])
# ax[1].plot(freq, np.abs(sp))
# ax[1].set_ylim([-1e4, 1e6])
# ax[1].set_xlim([0, .5])

# max_inds = np.argsort(np.abs(sp))
# num_inds = 20
# ax[1].plot(freq[max_inds[-num_inds:]], [0]*num_inds, 'k.')


# T_cutoff = 8760/(2**2) # cutoff period
# f_cutoff = 1/T_cutoff
# inds = np.where(np.abs(freq) < f_cutoff)[0]
# print(len(inds))
# sp_trunc = np.zeros(len(sp))
# sp_trunc[inds] = sp[inds]
# # sp_trunc = sp

# # sp_trunc = np.zeros(len(sp))
# # sp_trunc[max_inds[-num_inds:]] = sp[max_inds[-num_inds:]]

# data_trunc = np.fft.ifft(sp_trunc)
# ax[0].plot(data_trunc)

# # ax[0].plot(np.cumsum(gen_profiles[:,0]))
# # ax[0].plot(np.cumsum(np.roll(gen_profiles[:,0], 100)))
# width = 1000
# cs = np.cumsum(gen_profiles[:,0])
# # ax[0].plot(np.linspace(width, 8760-width, int(8760-width)), (cs[width:] - cs[:-width])/width)
# ax[0].plot(np.linspace(0, 8760, int(8760-width)), (cs[width:] - cs[:-width])/width)
# # ax[0].plot(1/width * np.cumsum(gen_profiles[:,0] - np.roll(gen_profiles[:,0], width)) )

##%%

# def LPF(data):
#     sp =  np.fft.fft(data)
#     freq = np.fft.fftfreq(np.shape(data)[-1])
#     T = 8760/(2**6)
#     f = 1/T
#     inds = np.where(np.abs(freq) < f)[0]
#     sp_trunc = np.zeros(len(sp))
#     sp_trunc[inds] = sp[inds]

#     filtered_data = np.fft.ifft(sp_trunc)
#     return filtered_data

# fig, ax = plt.subplots(2,1, figsize=(15, 10))
# ax[0].plot(gen_profiles, **state_kwargs)
# ax[1].plot(gen_filtered, color="blue", linewidth=.5)


#%% K means on gen profiles


def LPF_ma(data):
    width = 24 * 30 * 3
    cs = np.cumsum(data)
    filtered_data = (cs[width:] - cs[:-width])/width
    filtered_data = np.interp(np.arange(0, 8760, 1), np.linspace(0, 8760, int(8760-width)), filtered_data)
    return filtered_data


gen_filtered = np.zeros(np.shape(gen_profiles))
for i in range(np.shape(gen_profiles)[1]):
    gen_filtered[:,i] = LPF_ma(gen_profiles[:,i])


#%%
    
# n_clusters = 4

# X = gen_filtered.T
# kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)

# if n_clusters < 20:
#     fig1, ax1 = plt.subplots(
#         n_clusters, 1, sharey="col", sharex="col", figsize=(15, 10)
#     )
#     fig2, ax2 = plt.subplots(
#         n_clusters, 4, sharey="all", sharex="col", figsize=(15, 10)
#     )

#     data_max = np.max(X)

#     for i in range(n_clusters):
#         data_inds = np.where(kmeans.labels_ == i)[0]
#         ax1[i].plot(X[data_inds,:].T, color='blue', linewidth=.5, alpha=.5)
#         ax1[i].plot(kmeans.cluster_centers_[i].T, color="black")
#         min_ind = np.argmin(kmeans.cluster_centers_[i])
#         # ax1[i].plot([min_ind, min_ind], [0, data_max], color="orange")

#         df_cluster = df_all[df_all["gen_ind"].isin(data_inds)]
#         gi_vals, gi_counts = np.unique(df_cluster["gen_ind"], return_counts=True)
#         yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
#         split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)

#         ax2[i, 0].bar(gi_vals, gi_counts)
#         ax2[i, 1].bar(yr_vals, yr_counts)
#         ax2[i, 2].bar(split_vals, split_counts)

# fig1.subplots_adjust(hspace=0)
# fig1.suptitle("Generation profiles filtered, clustered")

# fig2.subplots_adjust(hspace=0, wspace=0)
# ax2[-1,0].set_xlabel("generation indx")
# ax2[-1,1].set_xlabel("year")
# ax2[-1,2].set_xlabel("wind/pv split")

#%%
n_clusters = 4

fig = plt.figure(figsize=(12,6))
super_grid = fig.add_gridspec(1, 2)

subgrid_profile = super_grid[0].subgridspec(n_clusters, 1, hspace=0)
axs_prof = subgrid_profile.subplots()

subgrid_counts = super_grid[1].subgridspec(n_clusters, 5, hspace=0, wspace=0)
axs_count = subgrid_counts.subplots()

data = gen_filtered
X = data.T
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)



years = np.unique(df_all["year"])
RL = np.unique(df_all["rl"])
TD = np.unique(df_all["td"])
splits = np.unique(df_all["split"])
lats = np.unique(df_all["lat"])

data_max = np.max(data)
max_count = 0

for i in range(n_clusters):
    data_inds = np.where(kmeans.labels_ == i)[0]
    axs_prof[i].plot(data[:, data_inds], **state_kwargs)
    axs_prof[i].plot(kmeans.cluster_centers_[i].T, color="black")
    min_ind = np.argmin(kmeans.cluster_centers_[i])
    # axs_prof[i].plot([min_ind, min_ind], [0, data_max], color="orange")
    # axs_prof[i].text( min_ind+100, 0.9*data_max, f"min at {min_ind}")
    axs_prof[i].text(0, .9*data_max, f"{len(data_inds)} cases")
    axs_prof[i].set_ylim([-.05*data_max, 1.1*data_max])
    # axs_prof[i].set_yticks([])
    axs_prof[i].set_ylabel("Gen. [kg/hr]")

    # df_cluster = df.iloc[list(data_inds)]
    df_cluster = df_all[(df_all["gen_ind"].isin(data_inds)) & (df_all["rl"] == RL[1]) & (df_all["td"] == TD[1])]
    rl_vals, rl_counts = np.unique(df_cluster["rl"], return_counts=True)
    td_vals, td_counts = np.unique(df_cluster["td"], return_counts=True)
    yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
    split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)
    lat_vals, lat_counts = np.unique(df_cluster["lat"], return_counts=True)


    # rl_x = [np.where(RL == rlv)[0][0] for rlv in rl_vals]
    # td_x = [np.where(TD == tdv)[0][0] for tdv in td_vals]
    yr_x = [np.where(years == yrv)[0][0] for yrv in yr_vals]
    split_x = [np.where(splits == splitv)[0][0] for splitv in split_vals]
    lat_x = [np.where(lats == latv)[0][0] for latv in lat_vals]


    # axs_count[i,0].plot(df_cluster["storage_cap_kg"].to_numpy())

    # axs_count[i, 0].bar(rl_x, rl_counts)
    # axs_count[i, 1].bar(td_x, td_counts) #, width=0.9*np.min((TD - np.roll(TD, 1))[1:]))
    axs_count[i, 2].bar(split_x, split_counts) #, width=0.9*np.min((splits - np.roll(splits, 1))[1:]))
    axs_count[i, 3].bar(yr_x, yr_counts)
    axs_count[i, 4].bar(lat_x, lat_counts)


    # axs_count[i, 0].set_ylabel(f"count, total={len(data_inds)}")
    axs_count[i, 0].set_ylabel("count")

    max_count = np.max([max_count, np.max(np.concatenate([rl_counts, td_counts, yr_counts, split_counts, lat_counts]))])

for i in range(n_clusters):
    for j in range(5):
        axs_count[i, j].set_xticks([])
        if j > 0:
            axs_count[i, j].set_yticks([])
        axs_count[i, j].set_ylim([0, max_count*1.05])

    axs_count[i, 0].set_xlim([-1, len(RL)])
    axs_count[i, 1].set_xlim([-1, len(TD)])
    axs_count[i, 2].set_xlim([-1, len(splits)])
    axs_count[i, 3].set_xlim([-1, len(years)])
    axs_count[i, 4].set_xlim([-1, len(lats)])

axs_count[i,0].set_xticks(np.arange(0, len(RL), 1), RL, rotation=90)
axs_count[i,1].set_xticks(np.arange(0, len(TD), 1), TD, rotation=90)
axs_count[i,2].set_xticks(np.arange(0, len(splits), 1), splits, rotation=90)
axs_count[i,3].set_xticks(np.arange(0, len(years), 1), years, rotation=90)
axs_count[i,4].set_xticks(np.arange(0, len(lats), 1), ["TX", "IN"], rotation=90)


axs_prof[0].set_title("Generation clusters")
axs_count[0,0].set_title("Ramp limit")
axs_count[0,1].set_title("Turndown")
axs_count[0,2].set_title("Split")
axs_count[0,3].set_title("Year")
axs_count[0,4].set_title("Location")

fig.tight_layout()

fig.savefig("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/gen_clusters.png", format="png", dpi=400)



## %%

# # years = np.unique(df["year"])
# RL = np.unique(df["rl"])
# TD = np.unique(df["td"])


# def get_soc(df):
#     soc = np.stack(
#         df[df["storage_cap_kg"] > 0]["storage_state"].to_numpy()
#         / df[df["storage_cap_kg"] > 0]["storage_cap_kg"]
#     )
#     return soc


## %%

# RL = np.unique(df["rl"])
# TD = np.unique(df["td"])

# def parse_hoppp_input(string):
#     strings = string[:-5].split("_")
#     print(strings)
#     year = int(strings[0][4:])
#     lat = float(strings[1][3:])
#     lon = float(strings[2][3:])
#     split = int(strings[3][7:])
#     return year, lat, lon, split


# parse_hoppp_input(df.iloc[3]["hopp_input"])

# storage_state = np.stack(df["storage_state"].to_numpy())

# def plot_dist(data):
#     fig, ax = plt.subplots(4, 1, sharex='col', figsize=(10, 10))
#     ax[0].plot(gen_profiles, color="black", linewidth=0.125, alpha=0.125)
#     ax[0].set_ylabel("all gen. profiles")

#     mean_gen = np.mean(gen_profiles, axis=1)
#     std_gen = np.std(gen_profiles, axis=1)
#     std_up_gen = mean_gen + std_gen
#     std_down_gen = mean_gen - std_gen

#     ax[1].fill_between(
#         np.linspace(0, len(gen_profiles), len(gen_profiles)),
#         std_down_gen,
#         std_up_gen,
#         color="orange",
#     )
#     ax[1].plot(mean_gen, color="black")
#     ax[1].set_ylabel("gen. stats")

#     ax[2].plot(data, color="blue", linewidth=0.125, alpha=0.125)
#     ax[2].set_ylabel("all storage profiles")

#     mean = np.mean(data, axis=1)
#     std = np.std(data, axis=1)
#     std_up = mean + std
#     std_down = mean - std
#     ax[3].fill_between(
#         np.linspace(0, len(mean), len(mean)), std_down, std_up, color="orange"
#     )
#     ax[3].plot(mean, color="black")
#     ax[3].set_ylabel("storage stats")
#     ax[3].set_ylim(ax[2].get_ylim())
#     return fig, ax


# fig, ax = plot_dist(storage_state.T)
# fig.subplots_adjust(hspace=0)

# data = storage_state.T

## %% do SVD on the data to find the underlying dimensions

# U, S, Vh = np.linalg.svd(data)

# fig, ax = plt.subplots(1, 1)
# ax.plot(np.cumsum(S / np.sum(S)))
# ax.set_xlim([-5, 20])
# ax.set_title("Singular value cumulative fraction")

# # %%

# trunc_dim = 2

# S_trunc = np.copy(S)
# S_trunc[trunc_dim:] = 0
# data_trunc = np.dot(U[:, : len(S_trunc)] * S_trunc, Vh)

# fig, ax = plt.subplots(3, 1, sharey="none", figsize=(15, 10))
# ax[0].plot(data, **state_kwargs)
# ax[1].plot(data_trunc, **state_kwargs)

# for i in range(trunc_dim):
#     ax[2].plot(-U[:, i], color="black", linewidth=np.exp(-i / 10), alpha=np.exp(-i / 10))

# print(f"{np.linalg.norm(data - data_trunc):e}")
# fig.suptitle(f"First {trunc_dim} singular values reconstruction")

## %% k means cluster storage state into groups

# errors = []

# X = storage_state

# clusters = np.concatenate(
#     [np.arange(2, 15, 1), np.arange(16, 30, 2), np.arange(30, int(len(df)-1), 25)]
# )

# for n_clusters in clusters:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
#     error = 0
#     for i in range(n_clusters):
#         data_inds = np.where(kmeans.labels_ == i)[0]
#         for j in range(len(data_inds)):
#             error += np.linalg.norm(kmeans.cluster_centers_[i] - data[:, data_inds[j]])

#     # print(f"{error:e}")
#     errors.append(error)

# fig, ax = plt.subplots(1, 1)
# ax.plot(errors, ".-")
# ax.set_xlim(-2, 20)
# ax.set_xlabel("number of clusters")
# ax.set_ylabel("Total error")


#%%
n_clusters = 3

fig = plt.figure(figsize=(12,6))
super_grid = fig.add_gridspec(1, 2)

subgrid_profile = super_grid[0].subgridspec(n_clusters, 1, hspace=0)
axs_prof = subgrid_profile.subplots()

subgrid_counts = super_grid[1].subgridspec(n_clusters, 5, hspace=0, wspace=0)
axs_count = subgrid_counts.subplots()

data = np.stack(df["storage_state"].to_numpy()).T
X = data.T
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)

splits = np.sort(np.unique(df["split"]))

years = np.unique(df["year"])
RL = np.unique(df["rl"])
TD = np.unique(df["td"])
splits = np.unique(df["split"])
lats = np.unique(df_all["lat"])

data_max = np.max(data)
max_count = 0

for i in range(n_clusters):
    data_inds = np.where(kmeans.labels_ == i)[0]
    axs_prof[i].plot(data[:, data_inds], **state_kwargs)
    axs_prof[i].plot(kmeans.cluster_centers_[i].T, color="black")
    min_ind = np.argmin(kmeans.cluster_centers_[i])
    axs_prof[i].plot([min_ind, min_ind], [0, data_max], color="orange")
    axs_prof[i].text( min_ind+100, 0.9*data_max, f"min at {min_ind}")
    axs_prof[i].text(0, .9*data_max, f"{len(data_inds)} cases")
    # axs_prof[i].text(0, -.11*data_max, f"{len(data_inds)} profiles")
    # axs_prof[i].set_ylim([-.15*data_max, 1.1*data_max])
    # axs_prof[i].set_yticks([])
    axs_prof[i].set_ylabel("State [kg]")

    df_cluster = df.iloc[list(data_inds)]
    rl_vals, rl_counts = np.unique(df_cluster["rl"], return_counts=True)
    td_vals, td_counts = np.unique(df_cluster["td"], return_counts=True)
    yr_vals, yr_counts = np.unique(df_cluster["year"], return_counts=True)
    split_vals, split_counts = np.unique(df_cluster["split"], return_counts=True)
    lat_vals, lat_counts = np.unique(df_cluster["lat"], return_counts=True)


    rl_x = [np.where(RL == rlv)[0][0] for rlv in rl_vals]
    td_x = [np.where(TD == tdv)[0][0] for tdv in td_vals]
    yr_x = [np.where(years == yrv)[0][0] for yrv in yr_vals]
    split_x = [np.where(splits == splitv)[0][0] for splitv in split_vals]
    lat_x = [np.where(lats == latv)[0][0] for latv in lat_vals]
    

    axs_count[i, 0].bar(rl_x, rl_counts)
    axs_count[i, 1].bar(td_x, td_counts) #, width=0.9*np.min((TD - np.roll(TD, 1))[1:]))
    axs_count[i, 2].bar(split_x, split_counts) #, width=0.9*np.min((splits - np.roll(splits, 1))[1:]))
    axs_count[i, 3].bar(yr_x, yr_counts)
    axs_count[i, 4].bar(lat_x, lat_counts)


    # axs_count[i, 0].set_ylabel(f"count, total={len(data_inds)}")
    axs_count[i, 0].set_ylabel("count")

    max_count = np.max([max_count, np.max(np.concatenate([rl_counts, td_counts, yr_counts, split_counts, lat_counts]))])

for i in range(n_clusters):
    for j in range(5):
        axs_count[i, j].set_xticks([])
        if j > 0:
            axs_count[i, j].set_yticks([])
        axs_count[i, j].set_ylim([0, max_count*1.05])

    axs_count[i, 0].set_xlim([-1, len(RL)])
    axs_count[i, 1].set_xlim([-1, len(TD)])
    axs_count[i, 2].set_xlim([-1, len(splits)])
    axs_count[i, 3].set_xlim([-1, len(years)])
    axs_count[i, 4].set_xlim([-1, len(lats)])

axs_count[i,0].set_xticks(np.arange(0, len(RL), 1), RL, rotation=90)
axs_count[i,1].set_xticks(np.arange(0, len(TD), 1), TD, rotation=90)
axs_count[i,2].set_xticks(np.arange(0, len(splits), 1), splits, rotation=90)
axs_count[i,3].set_xticks(np.arange(0, len(years), 1), years, rotation=90)
axs_count[i,4].set_xticks(np.arange(0, len(lats), 1), ["TX", "IN"], rotation=90)

axs_prof[0].set_title("Storage state clusters")
axs_count[0,0].set_title("Ramp limit")
axs_count[0,1].set_title("Turndown")
axs_count[0,2].set_title("Split")
axs_count[0,3].set_title("Year")
axs_count[0,4].set_title("Location")

fig.tight_layout()

fig.savefig("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/state_clusters.png", format="png", dpi=400)

# %%

max_state = np.stack(df["storage_state"].to_numpy()).max()

years = np.unique(df["year"])
RL = np.unique(df["rl"])
TD = np.unique(df["td"])
split = np.unique(df["split"])
lats = np.unique(df["lat"])

plot_rows = np.max([len(years), len(RL), len(TD), len(split), len(lats)])

# use gridspec for subplot layout
fig = plt.figure(figsize=(12,6))
super_grid = fig.add_gridspec(1, 5, wspace=0)

sub_grid_yr = super_grid[0].subgridspec(plot_rows, 1, hspace=0)
axs = sub_grid_yr.subplots()
for i in range(len(years)):
    state = np.stack(df[df["year"] == years[i]]["storage_state"].to_numpy())
    axs[i].plot(state.T, **state_kwargs)
    axs[i].text(5000, 0.85 * max_state, f"year: {years[i]}")
    axs[i].set_ylim([-.05*max_state, 1.05 * max_state])

for i in range(len(axs)):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].spines[["top", "right", "bottom", "left"]].set_alpha(0.2)
axs[0].set_title("By year")

sub_grid_rl = super_grid[1].subgridspec(plot_rows, 1, hspace=0)    
axs = sub_grid_rl.subplots()
for i in range(len(RL)):
    state = np.stack(df[df["rl"] == RL[i]]["storage_state"].to_numpy())
    axs[i].plot(state.T, **state_kwargs)
    axs[i].text(5000, 0.85 * max_state, f"RL: {RL[i]}")
    axs[i].set_ylim([-.05*max_state, 1.05 * max_state])

for i in range(len(axs)):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].spines[["top", "right", "bottom", "left"]].set_alpha(0.2)
axs[0].set_title("By ramp limit")

sub_grid_td = super_grid[2].subgridspec(plot_rows, 1, hspace=0)    
axs = sub_grid_td.subplots()
for i in range(len(TD)):
    state = np.stack(df[df["td"] == TD[i]]["storage_state"].to_numpy())
    axs[i].plot(state.T, **state_kwargs)
    axs[i].text(5000, 0.85 * max_state, f"TD: {TD[i]}")
    axs[i].set_ylim([-.05*max_state, 1.05 * max_state])

for i in range(len(axs)):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].spines[["top", "right", "bottom", "left"]].set_alpha(0.2)
axs[0].set_title("By turndown ratio")

sub_grid_split = super_grid[3].subgridspec(plot_rows, 1, hspace=0)
axs = sub_grid_split.subplots()
for i in range(len(split)):
    state = np.stack(df[df["split"] == split[i]]["storage_state"].to_numpy())
    axs[i].plot(state.T, **state_kwargs)
    axs[i].text(5000, 0.85 * max_state, f"split: {split[i]}")
    axs[i].set_ylim([-.05*max_state, 1.05 * max_state])

for i in range(len(axs)):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].spines[["top", "right", "bottom", "left"]].set_alpha(0.2)
axs[0].set_title("By wind/pv split")

sub_grid_lat = super_grid[4].subgridspec(plot_rows, 1, hspace=0)
axs = sub_grid_lat.subplots()
locs = ["TX", "IN"]
for i in range(len(lats)):
    state = np.stack(df[df["lat"] == lats[i]]["storage_state"].to_numpy())
    axs[i].plot(state.T, **state_kwargs)
    axs[i].text(5000, 0.85 * max_state, f"loc: {locs[i]}")
    axs[i].set_ylim([-.05*max_state, 1.05 * max_state])

for i in range(len(axs)):
    axs[i].set_yticks([])
    axs[i].set_xticks([])
    axs[i].spines[["top", "right", "bottom", "left"]].set_alpha(0.2)
axs[0].set_title("By location")


fig.suptitle("All H2 storage state profiles")
fig.tight_layout()
fig.savefig("/Users/ztully/Documents/Green_Steel/HOPP_green_steel/hopp_scripts/dynamic_green_ammonia/dynamic_green_ammonia/plots/generation_statistics/all_gen.png", format="png", dpi=400)

#%%