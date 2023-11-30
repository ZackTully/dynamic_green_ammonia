import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from dynamic_green_ammonia.technologies.storage import DemandOptimization

run_opt = True

gen = np.load("dynamic_green_ammonia/run_scripts/hybrid_gen.npy")

n_steps = len(gen)
# n_steps = 1000
gen = gen[0:n_steps]

n_opts = 4

# ramp_lims = np.array([0, 0.00001, 0.0001, 0.001, 0.01, 0.99, 1])
# turndowns = np.array([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])

ramp_lims = np.concatenate([[0], np.logspace(-6, 0, n_opts - 1)])
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

            # center = np.interp(td, [0, 1], [np.max(gen) / 2, np.mean(gen)])
            center = np.mean(gen)

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

    # np.save("dynamic_green_ammonia/technologies/demand_opt_capacities.npy", capacities)

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
ax[1].legend()
ax[1].set_xlabel("ramp limit")
ax[1].set_xscale("log")


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


def make_contour_plot(data, zlabel):
    fig, ax = plt.subplots(1, 1)
    ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
    RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
    ax.contourf(RL, TD, data)
    CS = ax.contour(RL, TD, data)
    ax.clabel(CS, CS.levels, inline=True, colors="black")

    []


make_contour_plot(capacities, "Capacity [kg]")


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
