# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# %%

max_gen = 10000
mean_gen = 2000

turndowns = np.linspace(0, 1, 100)


min_demands = np.zeros(len(turndowns))
max_demands = np.zeros(len(turndowns))
centers = np.zeros(len(turndowns))

for i, td in enumerate(turndowns):
    centers[i] = np.interp(td, [0, 1], [max_gen / 2, mean_gen])
    # center = np.mean(H2_gen)
    max_demands[i] = (2 / (td + 1)) * centers[i]
    min_demands[i] = td * max_demands[i]

    # if min_demands[i] > mean_gen:
    #     min_demands[i] = mean_gen
    #     max_demands[i] = min_demands[i] / td
    #     centers[i] = np.mean([min_demands[i], max_demands[i]])


fig, ax = plt.subplots(3, 1, sharex="col", figsize=(5, 6))


ax[0].plot(turndowns, min_demands, label="min")
ax[0].plot(turndowns, centers, label="center")
ax[0].plot(turndowns, max_demands, label="max")

ax0_xlim = ax[0].get_xlim()

ax[0].plot(
    [-1, 2],
    [mean_gen, mean_gen],
    alpha=0.5,
    linewidth=0.25,
    color="red",
    zorder=0.9,
    label="mean",
)
ax[0].legend()


min_der = (min_demands - np.roll(min_demands, 1)) / (len(turndowns))
center_der = (centers - np.roll(centers, 1)) / (len(turndowns))
max_der = (max_demands - np.roll(max_demands, 1)) / (len(turndowns))


ax[1].plot([-1, 2], [0, 0], alpha=0.5, linewidth=0.25, color="black")
ax[1].plot(turndowns[1:], min_der[1:], label="dmin / dTD")
ax[1].plot(turndowns[1:], center_der[1:], label="dcenter / dTD")
ax[1].plot(turndowns[1:], max_der[1:], label="dmax / dTD")
ax[1].set_xlim(ax0_xlim)
ax[1].legend()

ax[2].plot([-1, 2], [0, 0], alpha=0.5, linewidth=0.25, color="black")
ax[2].plot(turndowns, mean_gen - min_demands, label="mean - min")
ax[2].legend()
ax[2].set_xlabel("Turndown ratio")

fig.suptitle(f"Max generation: {max_gen}, Mean generation: {mean_gen}")

plt.show()


# %%

g_bar = 1
mu = 0.5
c = lambda TD, mu: (1 - TD) * g_bar / 2 + (TD) * mu
d_max = lambda TD, c: (2 / (TD + 1)) * c
d_min = lambda TD, d_max: TD * d_max
d_ubar = lambda TD, mu: d_min(TD, d_max(TD, c(TD, mu)))

TDs = np.linspace(0.9, 1, 100)
mus = np.linspace(0, g_bar, 20)

diff = np.zeros([len(TDs), len(mus)])
mins = np.zeros([len(TDs), len(mus)])
maxs = np.zeros([len(TDs), len(mus)])


for i in range(len(mus)):
    for j in range(len(TDs)):
        mins[j, i] = d_ubar(TDs[j], mus[i])
        maxs[j, i] = d_max(TDs[j], c(TDs[j], mus[i]))
        diff[j, i] = mus[i] - d_ubar(TDs[j], mus[i])


fig, ax = plt.subplots(2, 1, sharex="col")


for i in range(len(mus)):
    ax[0].plot(TDs, mins[:, i], color=cm.plasma(i / len(mus)))
    ax[0].plot(TDs, maxs[:, i], color=cm.plasma(i / len(mus)))
    ax[1].plot(TDs, diff[:, i], color=cm.plasma(i / len(mus)))

ax_xlim = ax[1].get_xlim()
ax[1].plot([-1, 2], [0, 0], color="black", alpha=0.5, linewidth=0.75, zorder=0.5)
ax[1].set_xlim(ax_xlim)

# %%

d_ubar = lambda TD, mu: TD * (2 / (1 + TD)) * ((1 - TD) * g_bar / 2 + TD * mu)

fig, ax = plt.subplots(2, 1, sharex="col")

TDs = np.linspace(0, 1, 100)
mus = np.linspace(0, 1, 1000)
crossover = np.zeros(len(TDs))

for i in range(len(TDs)):
    d_min = np.zeros(len(mus))
    for j in range(len(mus)):
        d_min[j] = d_ubar(TDs[i], mus[j])

    cross_idx = np.argmin(np.abs((d_min - mus)))
    crossover[i] = mus[cross_idx]

    if i % 20 == 0:
        label = {"label": f"TD:{TDs[i]:.2f}"}
    else:
        label = {}
    ax[0].plot(mus, d_min, color=cm.plasma(i / len(TDs)), **label)

ax[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
ax[1].set_xlabel("mu")
ax[0].set_ylabel("d_min")
ax[0].legend()

ax[1].plot(crossover, TDs, label="infeasible")
ax[1].set_ylabel("TD")

plt.show()
# %%
