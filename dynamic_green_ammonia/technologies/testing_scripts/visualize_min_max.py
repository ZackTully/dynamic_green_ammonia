# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import fsolve


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
factor = .5

c = lambda TD, mu: (1 - TD**factor) * g_bar / 2 + (TD**factor) * mu
d_max = lambda TD, c: (2 / (TD + 1)) * c
d_min = lambda TD, d_max: TD * d_max
d_ubar = lambda TD, mu: d_min(TD, d_max(TD, c(TD, mu)))

TDs = np.linspace(0, 1, 1000)
# mus = np.linspace(0, g_bar, 50)
# mus = np.linspace(.2, .8, 10)
mus = [.125]


fig, ax = plt.subplots(1,1)
ax.plot(TDs, TDs ** factor)

diff = np.zeros([len(TDs), len(mus)])
mins = np.zeros([len(TDs), len(mus)])
maxs = np.zeros([len(TDs), len(mus)])
centers = np.zeros([len(TDs), len(mus)])


for i in range(len(mus)):
    for j in range(len(TDs)):
        mins[j, i] = d_ubar(TDs[j], mus[i])
        maxs[j, i] = d_max(TDs[j], c(TDs[j], mus[i]))
        centers[j,i] = c(TDs[j], mus[i])
        diff[j, i] = mus[i] - d_ubar(TDs[j], mus[i])


fig, ax = plt.subplots(3, 1, sharex="col")



for i in range(len(mus)):
    ax[0].plot(TDs, mins[:, i], color=cm.plasma(i / len(mus)))
    ax[0].plot(TDs, maxs[:, i], color=cm.plasma(i / len(mus)))
    ax[0].plot(TDs, centers[:,i], color=cm.plasma(i/len(mus)))
    ax[1].plot(TDs, diff[:, i], color=cm.plasma(i / len(mus)))

ax_xlim = ax[1].get_xlim()
ax[1].plot([-1, 2], [0, 0], color="black", alpha=0.5, linewidth=0.75, zorder=0.5)
ax[1].set_xlim(ax_xlim)

# ax[1].set_ylim([-.1, .1])

i = 0


ax[2].plot(TDs, centers[:,i] - mus[i])
ax[2].plot(TDs, (1 - TDs) * maxs[:,i])
ax[2].set_ylim([0,.2])
# %%

d_ubar = lambda TD, mu: TD * (2 / (1 + TD)) * ((1 - TD) * g_bar / 2 + TD * mu)

d_max = lambda TD: a / (TD + b)
d_min = lambda TD: TD * d_max(TD)

def get_d_min(mu, TD):
    A = np.array([[1, -g_bar], [1, -mu]])
    b = np.array([0, mu])
    coeffs = np.linalg.inv(A) @ b

    a = coeffs[0]
    b = coeffs[1]

    d_max = lambda TD: a / (TD + b)
    d_min = lambda TD: TD * d_max(TD)

    return d_min(TD)

fig, ax = plt.subplots(2, 1, sharex="col", figsize=(10,10))

TDs = np.linspace(0.01, .99, 100)
mus = np.linspace(0, 1, 1000)
crossover = np.zeros(len(TDs))

for i in range(len(TDs)):
    d_mins = np.zeros(len(mus))
    for j in range(len(mus)):
        d_mins[j] = d_ubar(TDs[i], mus[j])
        # d_mins[j] = get_d_min(TDs[i], mus[j])

    cross_idx = np.argmin(np.abs((d_mins - mus)))
    crossover[i] = mus[cross_idx]

    if i % 20 == 0:
        label = {"label": f"TD:{TDs[i]:.2f}"}
    else:
        label = {}
    ax[0].plot(mus, d_mins, color=cm.plasma(i / len(TDs)), **label)

ax[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
ax[1].set_xlabel("mu")
ax[0].set_ylabel("d_min")
ax[0].legend()

ax[1].plot(crossover, TDs, label="infeasible")
ax[1].set_ylabel("TD")

plt.show()

# %%

mu = .5
g_bar = 1

d_min = lambda TD, mu:  TD**.9 * mu
d_max = lambda TD, mu: d_min(TD, mu) / TD

TDs = np.linspace(0.01, 0.99, 100)
mus = np.linspace(.4, .6, 5)

fig, ax = plt.subplots(1,1)

for i in range(len(mus)):
    d_mins = np.zeros(len(TDs))
    d_maxs = np.zeros(len(TDs))
    for j in range(len(TDs)):
        d_mins[j] = d_min(TDs[j], mus[i])
        d_maxs[j] = d_max(TDs[j], mus[i])

    ax.plot(TDs, d_mins, color=cm.plasma(i/len(mus)))
    ax.plot(TDs, d_maxs, color=cm.plasma(i/len(mus)))


#%%

g_bar = 1
mu = .25

fig, ax = plt.subplots(2, 1, sharex='col', figsize=(10,10))
ax[0].set_xlim([-.05, 1.05])
ax[0].hlines([g_bar, g_bar/2, mu], -1, 2, linestyle="dashed", color='black', linewidth=.75, alpha=.5)
ax[0].plot([0, 1], [g_bar, mu], linestyle='dashed', color='black', linewidth=.75)
ax[0].plot([0, 1], [0, mu], linestyle='dashed', color='black', linewidth=.75)

min_factor = .5
max_factor = .5

TDs = np.linspace(0, 1, 100)

d_max = np.zeros(len(TDs))
d_min = np.zeros(len(TDs))

for i in range(len(TDs)):
    d_max[i] = (TDs[i]**max_factor) * mu + (1 - TDs[i]**max_factor) * g_bar
    d_min[i] = (TDs[i]**min_factor) * mu + (1 - TDs[i]**min_factor) * 0

ax[0].plot(TDs, d_max)
ax[0].plot(TDs, d_min)
ax[0].plot(TDs, (d_min + d_max)/2)

# ax[1].plot(TDs, d_min/d_max)

func = lambda f, TD: mu * TD ** (f-1) - (mu - g_bar) * TD ** f - g_bar
factors = np.zeros(len(TDs))

for i in range(len(TDs)):

    factors[i] = fsolve(lambda f: func(f, TDs[i]), .5)

ax[1].plot(TDs, factors)
#%%

def plot_bounds(g_bar, mu, n_subplots):
    fig, ax = plt.subplots(n_subplots, 1, sharex='col', figsize=(10,10))
    ax[0].set_xlim([-.05, 1.05])
    ax[0].hlines([g_bar, g_bar/2, mu], -1, 2, linestyle="dashed", color='black', linewidth=.75, alpha=.5)
    ax[0].plot([0, 1], [g_bar, mu], linestyle='dashed', color='black', linewidth=.75)
    ax[0].plot([0, 1], [0, mu], linestyle='dashed', color='black', linewidth=.75)
    return fig, ax

#%%
g_bar = 1
mu = .25

fig, ax = plot_bounds(g_bar, mu, 2)

d_max = lambda TD: (1-TD) * g_bar + (TD) * mu
d_min = lambda TD: TD * d_max(TD)

TDs = np.linspace(0, 1, 100)

ax[0].plot(TDs, d_max(TDs))
ax[0].plot(TDs, d_min(TDs))

#%%

def d_min_from_d_max(TD, d_max):
    return d_max * TD

def d_max_from_d_min(TD, d_min):
    return d_min / TD

g_bar = 1
mu = .9


fig, ax = plot_bounds(g_bar, mu, 2)
TDs = np.linspace(0.0, 1, 100)

c = lambda TD: (mu - g_bar / 2) * TD + g_bar / 2

a = -1 * (mu - g_bar / 2)
b = 2 * (mu - g_bar  / 2)

c = lambda TD: a * TD**2 + b*TD + g_bar/2

# d_min = lambda TD: (2 / (1 + 1 / TD)) * c(TD)
# d_max = lambda TD: (2 / (1 + TD)) * c(TD)

# A = np.array([
#     [0, 0, 1],
#     [1, 1, 1],
#     [3, 2, 1]
# ])

# b = np.array([g_bar, mu, 0])

# coeffs = np.linalg.inv(A)@b

# a = coeffs[0]
# b = coeffs[1]
# c = coeffs[2]

# d_max = lambda TD: a * TD**2 + b * TD + c
# d_min = lambda TD: a * TD**3 + b * TD**2 + c*TD


A = np.array([[1, -g_bar], [1, -mu]])
b = np.array([0, mu])
coeffs = np.linalg.inv(A) @ b

a = coeffs[0]
b = coeffs[1]

d_max = lambda TD: a / (TD + b)
d_min = lambda TD: TD * d_max(TD)

# ax[0].plot(TDs, c(TDs))
ax[0].plot(TDs, mu / TDs, linestyle='dotted', color='red')
ax[0].plot(TDs, d_min(TDs), color='blue')
ax[0].plot(TDs, d_max(TDs), color='blue')
ax[0].set_ylim([-.05, 1.5])


ax[1].plot(TDs, mu-d_min(TDs))

#%%

def min_max(g_bar, mu, TD):

    A = np.array([[1, -g_bar], [1, -mu]])
    b = np.array([0, mu])
    coeffs = np.linalg.inv(A) @ b

    a = coeffs[0]
    b = coeffs[1]

    d_max = lambda TD: a / (TD + b)
    d_min = lambda TD: TD * d_max(TD)

    return d_min(TD), d_max(TD)

g_bar = 1
mu = np.linspace(0.01, .99, 5)
TD = np.linspace(0, 1, 100)

fig, ax = plt.subplots(1,1)

for i in range(len(mu)):
    d_mi, d_ma = min_max(g_bar, mu[i], TD)
    ax.plot(TD, d_mi, color=cm.plasma(i / len(mu)))
    ax.plot(TD, d_ma, color=cm.plasma(i / len(mu)))


def plot_bounds(g_bar, mu, ax):
    ax.set_xlim([-.05, 1.05])
    ax.hlines([g_bar, mu], -1, 2, linestyle="dashed", color='black', linewidth=.75, alpha=.5)
    ax.plot([0, 1], [g_bar, mu], linestyle='dashed', color='black', linewidth=.75)
    ax.plot([0, 1], [0, mu], linestyle='dashed', color='black', linewidth=.75)
    ax.text( 0, mu, "mean generation")
    ax.text(0, g_bar, "max generation")


mus = [0.05, 0.2, 0.8, 0.95]
fig, ax = plt.subplots(1, len(mus), sharex='row', sharey='row', figsize=(15, 5), dpi=400)



for i in range(len(mus)):
    plot_bounds(g_bar, mus[i], ax[i])
    d_mi, d_ma = min_max(g_bar, mus[i], TD)
    ax[i].plot(TD, d_mi)
    ax[i].plot(TD, d_ma)
    ax[i].set_xlabel("Turndown ratio")
    
ax[0].set_ylabel("Plant rating")

