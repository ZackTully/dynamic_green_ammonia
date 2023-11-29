import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from dynamic_green_ammonia.technologies.storage import DemandOptimization

run_opt = True

gen = np.load("dynamic_green_ammonia/run_scripts/hybrid_gen.npy")

n_steps = len(gen)
n_steps = 1000
gen = gen[0:n_steps]


# ramp_lims = np.array([0, 0.00001, 0.0001, 0.001, 0.01, 0.99, 1])
# turndowns = np.array([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])

ramp_lims = np.concatenate([[0], np.logspace(-6, 0, 7)])
turndowns = np.linspace(0, 1, 8)
# turndowns += 2e-1 * np.round(np.sin(np.linspace(0, 2 * np.pi, len(turndowns))), 6)

if run_opt:
    count = 0

    capacities = np.zeros([len(ramp_lims), len(turndowns)])

    for i in range(len(ramp_lims)):
        for j in range(len(turndowns)):
            rl = ramp_lims[i]
            td = turndowns[j]

            print(
                f"Ramp: {rl:.5f}, TD: {td:.2f}, {count} out of {len(ramp_lims) * len(turndowns)}"
            )
            count += 1

            max_demand = (2 / (td + 1)) * np.mean(gen)
            min_demand = td * max_demand
            ramp_lim = rl * max_demand

            DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
            x, success = DO.optimize()
            if not success:
                print("Failed to converge")
                continue
            capacity = x[-2] - x[-1]
            capacities[i, j] = capacity
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

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
ramp_lims_fake = np.linspace(0, 1, len(ramp_lims))
RL, TD = np.meshgrid(ramp_lims_fake, turndowns)
surf = ax.plot_surface(RL, TD, capacities, cmap=cm.plasma)
cbar = fig.colorbar(surf, shrink=0.5, aspect=7)

ax.set_xticks(ramp_lims_fake, np.flip(ramp_lims))
ax.set_yticks(turndowns, np.flip(np.round(turndowns, 2)))
ax.set_zticklabels([])
ax.set_xlabel("ramp limit")
ax.set_ylabel("turndown ratio")
cbar.set_label("Capacity [kg]")
ax.invert_xaxis()
ax.invert_yaxis()

plt.show()

[]
