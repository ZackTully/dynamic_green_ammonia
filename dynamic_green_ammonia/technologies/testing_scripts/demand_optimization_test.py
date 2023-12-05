# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle

from dynamic_green_ammonia.technologies.storage import DemandOptimization

# %%
run_opt = True

rl_realistic = 0.2
td_realistic = 0.6

gen = np.load("dynamic_green_ammonia/run_scripts/hybrid_gen.npy")

n_steps = len(gen)
# n_steps = 200
gen = gen[0:n_steps]

n_opts = 16

# ramp_lims = np.array([0, 0.00001, 0.0001, 0.001, 0.01, 0.99, 1])
# turndowns = np.array([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])

ramp_lims = np.concatenate([[0], np.logspace(-5, 0, n_opts - 1)])
turndowns = np.linspace(0, 1, n_opts)
# turndowns += 2e-1 * np.round(np.sin(np.linspace(0, 2 * np.pi, len(turndowns))), 6)

if run_opt:
    count = 0

    capacities = np.zeros([len(ramp_lims), len(turndowns)])
    initial_state = np.zeros([len(ramp_lims), len(turndowns)])

    for i in range(len(ramp_lims)):
        for j in range(len(turndowns)):
            rl = ramp_lims[i]
            td = turndowns[j]

            print(
                f"Ramp: {rl:.5f}, TD: {td:.2f}, {count} out of {len(ramp_lims) * len(turndowns)}"
            )
            count += 1

            center = np.interp(td, [0, 1], [np.max(gen) / 2, np.mean(gen)])
            # center = np.mean(gen)

            max_demand = (2 / (td + 1)) * center
            min_demand = td * max_demand
            ramp_lim = rl * max_demand

            if ramp_lim > (max_demand - min_demand):
                print("ramp lim is overly flexible")

            DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
            x, success = DO.optimize()
            if not success:
                print("Failed to converge")
                continue
            capacity = x[-2] - x[-1]
            capacities[i, j] = capacity
            initial_state[i, j] = x[n_steps]
            []

    np.save("dynamic_green_ammonia/technologies/demand_opt_capacities.npy", capacities)

else:
    capacities = np.load("dynamic_green_ammonia/technologies/demand_opt_capacities.npy")

fig, ax = plt.subplots(1, 2, sharey="row")

colors = np.linspace(
    np.array([10, 79, 191]) / 255, np.array([217, 235, 19]) / 255, len(ramp_lims)
)


for i in range(len(ramp_lims)):
    ax[0].plot(
        turndowns,
        capacities[i, :],
        color=np.flip(colors, axis=0)[i],
        marker=".",
        label=f"RL={ramp_lims[i]:.3f}",
    )
ax0_ylim = ax[0].get_ylim()


realistic_kwargs = {"color": "black", "linestyle": "dashed"}

ax[0].plot(
    [td_realistic, td_realistic],
    [ax0_ylim[0] - 1e6, ax0_ylim[1] + 1e6],
    **realistic_kwargs,
)
ax[0].set_ylim(ax0_ylim)

ax[0].plot(
    turndowns,
    [
        np.interp(rl_realistic, ramp_lims, capacities[:, i])
        for i in range(len(turndowns))
    ],
    **realistic_kwargs,
)

ax[0].legend()
ax[0].set_xlabel("turndown ratio")
ax[0].set_ylabel("Capacity")
ax[0].invert_xaxis()

for j in range(len(turndowns)):
    ax[1].plot(
        ramp_lims,
        capacities[:, j],
        color=colors[j],
        marker=".",
        label=f"TD={turndowns[j]:.3f}",
    )
ax1_ylims = ax[1].get_ylim()
ax[1].plot(
    [rl_realistic, rl_realistic],
    [ax1_ylims[0] - 1e6, ax1_ylims[1] * 1.5],
    **realistic_kwargs,
)
ax[1].plot(
    ramp_lims,
    [
        np.interp(td_realistic, turndowns, capacities[i, :])
        for i in range(len(ramp_lims))
    ],
    **realistic_kwargs,
)
ax[1].set_ylim(ax1_ylims)

ax[1].legend()
ax[1].set_xlabel("ramp limit")
ax[1].set_xscale("log")

plt.show()
[]


def make_surface_plot(data, zlabel):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
    RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
    surf = ax.plot_surface(RL, TD, data, cmap=cm.plasma)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

    ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
    ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
    ax.set_zticklabels([])
    ax.set_xlabel("ramp limit")
    ax.set_ylabel("turndown ratio")
    cbar.set_label(zlabel)
    ax.invert_xaxis()
    ax.invert_yaxis()


make_surface_plot(capacities, "Capacity [kg]")
# make_surface_plot(initial_state, "Initial state [kg]")
# make_surface_plot(initial_state / capacities, "Initial SOC")


data = capacities

# def make_contour_plot(data, zlabel):
fig, ax = plt.subplots(1, 1)
ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
RL, TD = np.meshgrid(ramp_lims_fake, turndowns)

n_levels = 15
# levels = np.linspace(np.min(data), np.max(data), n_levels)

curviness = 1
interp_locs = np.log(np.linspace(np.exp(0), np.exp(curviness), n_levels)) / curviness
levels = np.interp(interp_locs, [0, 1], [np.min(data), np.max(data)])

# color_kwargs = {"cmap": cm.plasma, "vmin": np.min(data), "vmax": np.max(data)}
color_kwargs = {"cmap": cm.plasma, "vmin": 1e5, "vmax": np.max(data)}

CSf = ax.contourf(RL, TD, data, alpha=1, levels=levels, **color_kwargs)
CS1 = ax.contour(RL, TD, data, levels=levels, **color_kwargs)

rect_kwargs = {"alpha": 0.5, "facecolor": "white"}
rect1 = Rectangle([0, 0], rl_realistic, 1, **rect_kwargs)
rect2 = Rectangle(
    [rl_realistic, td_realistic], 1 - rl_realistic, 1 - td_realistic, **rect_kwargs
)
ax.add_patch(rect1)
ax.add_patch(rect2)

ax.plot([rl_realistic, rl_realistic], [0, td_realistic], color="black")
ax.plot([rl_realistic, 1], [td_realistic, td_realistic], color="black")

# ax.contourf(RL_realistic, TD_realistic, data_realistic, alpha=1, **color_kwargs)

ax.clabel(CS1, CS1.levels, inline=True, colors="black")
cbar = fig.colorbar(CSf)
ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
ax.invert_xaxis()
ax.set_xlabel("ramp limit")
ax.set_ylabel("turndown ratio")

plt.show()
# make_contour_plot(capacities, "Capacity [kg]")


# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
# RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
# surf = ax.plot_surface(RL, TD, capacities, cmap=cm.plasma)
# cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

# ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
# ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
# ax.set_zticklabels([])
# ax.set_xlabel("ramp limit")
# ax.set_ylabel("turndown ratio")
# cbar.set_label("Capacity [kg]")
# ax.invert_xaxis()
# ax.invert_yaxis()

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# # ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
# # RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
# # surf = ax.plot_surface(RL, TD, initial_state * 4 / 3903988.512107127, cmap=cm.plasma)
# surf = ax.plot_surface(RL, TD, initial_state, cmap=cm.plasma)
# cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

# ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
# ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
# ax.set_zticklabels([])
# ax.set_xlabel("ramp limit")
# ax.set_ylabel("turndown ratio")
# cbar.set_label("Initial state [kg]")
# ax.invert_xaxis()
# ax.invert_yaxis()

# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
# # ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
# # RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
# surf = ax.plot_surface(RL, TD, initial_state / capacities, cmap=cm.plasma)
# cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

# ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
# ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
# ax.set_zticklabels([])
# ax.set_xlabel("ramp limit")
# ax.set_ylabel("turndown ratio")
# cbar.set_label("Initial SOC")
# ax.invert_xaxis()
# ax.invert_yaxis()


plt.show()

[]
