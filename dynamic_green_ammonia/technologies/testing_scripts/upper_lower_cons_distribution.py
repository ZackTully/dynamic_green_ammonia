import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from dynamic_green_ammonia.technologies.storage import DemandOptimization

gen = np.load("dynamic_green_ammonia/run_scripts/hybrid_gen.npy")

fig, ax = plt.subplots(1, 1)

n, bins, patches = ax.hist(gen, bins=100, density=True)

td = 0.5
ramp_limit = 0.1

max_demand = (2 / (td + 1)) * np.mean(gen)
min_demand = td * max_demand
ramp_lim = ramp_limit * max_demand


ax.plot(
    [min_demand, max_demand], [np.max(n) / 2] * 2, color="black", linestyle="dashed"
)
ax.plot([0, min_demand], [np.max(n) / 2] * 2, color="black", linestyle="solid")

ax.plot(np.mean(gen), np.max(n) / 2, "r.", markersize=10)
ax.plot(min_demand, np.max(n) / 2, "k.", markersize=10)
ax.plot(max_demand, np.max(n) / 2, "k.", markersize=10)
ax.plot(0, np.max(n) / 2, "k.", markersize=10)

ax.set_xlabel("H2 generation [kg/hr]")
ax.set_ylabel("Probability density")


# With Zack approach
DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
x, success = DO.optimize()
capacity = x[-2] - x[-1]
print(f"Capacity (mean): {capacity:.2f}")

# With a slight shift downwards
shift = 500
DO = DemandOptimization(
    gen, ramp_limit * (max_demand + shift), min_demand + shift, max_demand + shift
)
x, success = DO.optimize()
capacity = x[-2] - x[-1]
print(f"Capacity (shifted): {capacity:.2f}")

plt.show()

# gen = gen[0:1000]
n_steps = len(gen)

ramp_limit = 0.01
TD_ratio = 0.5


n_ramps = 1
n_mins = 1

ramp_lims = np.exp(np.linspace(np.log(0.01), np.log(1), n_ramps))
TD_ratios = 1 - np.logspace(np.log10(0.99), -2, n_mins)


ramp_lims = [0.75]
TD_ratios = [0.01]

mean_caps = np.zeros([n_ramps, n_mins])
best_caps = np.zeros([n_ramps, n_mins])

count = 0

for i in range(len(ramp_lims)):
    for j in range(len(TD_ratios)):
        rl = ramp_lims[i]
        td = TD_ratios[j]

        print(f"Ramp: {rl:.2f}, TD: {td:.2f}, {count} out of {n_ramps * n_mins}")
        count += 1

        max_demand = (2 / (td + 1)) * np.mean(gen)
        min_demand = td * max_demand
        ramp_lim = ramp_limit * max_demand

        DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
        x, success = DO.optimize()
        capacity = x[-2] - x[-1]
        print(f"Capacity (mean): {capacity:.2f}")

        mean_caps[i, j] = capacity

        capture = np.zeros(len(bins))

        for k, mean in enumerate(bins):
            max_demand = (2 / (td + 1)) * mean
            min_demand = td * max_demand

            min_idx = np.argmin(np.abs(bins - min_demand))
            max_idx = np.argmin(np.abs(bins - max_demand))

            if min_idx != max_idx:
                capture[k] = np.sum(n[min_idx:max_idx]) / (
                    np.sum(n[0:min_idx]) + np.sum(n[max_idx:])
                )

        mean = bins[np.argmax(capture)]
        max_demand = (2 / (td + 1)) * mean
        min_demand = td * max_demand

        DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
        x, success = DO.optimize()
        if not success:
            print("best failed to converge")
            continue
        capacity = x[-2] - x[-1]

        print(f"Capacity (best): {capacity:.2f}")
        best_caps[i, j] = capacity


diff = best_caps - mean_caps

rls, tds = np.meshgrid(ramp_lims, TD_ratios)
diff_max = np.abs(diff).max()
fig, ax = plt.subplots(1, 1)
# c = ax.pcolormesh(rls, tds, diff, cmap="RdBu", vmin=-diff_max, vmax=diff_max)
# ax.axis([rl.min(), rl.max(), td.min(), td.max()])
# fig.colorbar(c, ax=ax)

# plt.show()

ax.plot(np.mean(gen), 100, "k.")
ax.plot(min_demand, 100, "k.")
ax.plot(max_demand, 100, "k.")

print(f"Capacity (mean): {capacity:.2f}")
# if not res.success:
#     print(res.message)
#     exit()

capture = np.zeros(len(bins))

for i, mean in enumerate(bins):
    max_demand = (2 / (TD_ratio + 1)) * mean
    min_demand = TD_ratio * max_demand

    min_idx = np.argmin(np.abs(bins - min_demand))
    max_idx = np.argmin(np.abs(bins - max_demand))

    if min_idx != max_idx:
        capture[i] = np.sum(n[min_idx:max_idx]) / (
            np.sum(n[0:min_idx]) + np.sum(n[max_idx:])
        )

mean = bins[np.argmax(capture)]
max_demand = (2 / (TD_ratio + 1)) * mean
min_demand = TD_ratio * max_demand


DO = DemandOptimization(gen, ramp_lim, min_demand, max_demand)
res = DO.optimize()
capacity = res.x[-2] - res.x[-1]


print(f"Capacity (best): {capacity:.2f}")


bin_diff = np.argmin(np.abs(bins - (max_demand - min_demand)))

counts = np.zeros(len(bins))

for i, bin in enumerate(bins):
    if (i + bin_diff) >= len(bins):
        break
    start_bin = bins[i]
    end_bin = bins[i + bin_diff]
    counts[i] = np.sum(n[i : i + bin_diff])


ax.plot(bins[1:], np.cumsum(n) / 87)
ax.plot(bins + bins[int(bin_diff / 2)], 100 * counts / np.max(counts))

ax.plot(bins, capture / np.max(capture) * 400)
# plt.show()

H2_demand = res.x[0:n_steps]
charge = res.x[n_steps:-2]
capacity = res.x[-2] - res.x[-1]

print(f"Capacity {capacity:.2f}")

fig, ax = plt.subplots(2, 1, sharex="col", figsize=[20, 10])


ax[0].plot(gen[0:n_steps], label="generation", linewidth=0.5)
ax[0].plot(H2_demand, label="demand", linewidth=2)
ax[0].plot(gen[0:n_steps] - H2_demand, label="chg/dchg", linewidth=0.5)
ax[0].hlines(
    [min_demand, np.mean(gen), max_demand], 0, n_steps, color="black", alpha=0.25
)

ax[0].legend()
ax[0].set_title("demand")
ax[1].plot(charge)
ax[1].set_title("charge")
fig.suptitle(f"Storage Capacity: {capacity:.2f}")
fig.savefig("dynamic_green_ammonia/plots/LCOA_heatmap.png", dpi=300, format="png")
plt.show()
[]
