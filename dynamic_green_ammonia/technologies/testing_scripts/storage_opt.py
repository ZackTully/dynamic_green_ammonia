import numpy as np
import pprint
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from pathlib import Path

print_matrices = False
rootdir = Path(__file__).parents[2]
H2_gen = np.load(rootdir / "data" / "hybrid_gen.npy")


N = 250

shift = 0
H2_gen = H2_gen[shift : N + shift]

# plant paramaters
td = 0.25
rl = 0.1 * (1 - td)
# These parameters cause the weird behavior
# td = 0.05
# rl = 0.05 * (1 - td)


center = np.interp(td, [0, 1], [np.max(H2_gen) / 2, np.mean(H2_gen)])
# center = np.mean(H2_gen)
max_demand = (2 / (td + 1)) * center
min_demand = td * max_demand
R = rl * max_demand


# H2_gen = np.linspace(0, 1.5 * max_demand, N) + 3 * np.random.random(N)

# x = [u_0, ... , u_N, x_0, ... , x_N, x_max, x_min]
u_min = min_demand
u_max = max_demand

# Cost vector
C = np.zeros(N + N + 2)
# C[N : N + N] = 1 * np.ones(N)
C[2 * N + 0] = 1  # highest storage state
C[2 * N + 1] = -1  # lowest storage state

# Upper and lower bounds
bound_l = np.concatenate(
    [
        [u_min] * N,  # demand lower bound
        [0] * N,  # storage state lower bound
        [None, 0],  # storage state max, min lower bound
    ]
)


bound_u = np.concatenate(
    [
        [u_max] * N,  # demand upper bound,
        [None] * N,  # storage state upper bound,
        [None, None],  # storage state max, min upper bound
    ]
)


# Positive demand ramp rate limit
Aub_ramp_pos = np.zeros([N, N + N + 2])
bub_ramp_pos = np.zeros(N)

# u[k+1] - u[k] <= R
# x[k+1] - x[k] <= R
for k in range(N):
    if (k + 1) == N:
        break
    Aub_ramp_pos[k, k + 1] = 1
    Aub_ramp_pos[k, k] = -1
    bub_ramp_pos[k] = R


# Negative demand ramp rate limit
Aub_ramp_neg = np.zeros([N, N + N + 2])
bub_ramp_neg = np.zeros(N)

# -u[k+1] + u[k] <= R
# -x[k+1] + x[k] <= R
for k in range(N):
    if (k + 1) == N:
        break
    Aub_ramp_neg[k, k + 1] = -1
    Aub_ramp_neg[k, k] = 1
    bub_ramp_neg[k] = R

factor = 1 / (R)

Aub_ramp_pos *= factor
bub_ramp_pos *= factor

Aub_ramp_neg *= factor
bub_ramp_neg *= factor

# x_max
Aub_xmax = np.zeros([N, N + N + 2])
bub_xmax = np.zeros(N)

# state[k] - state_max <= 0
# x[N+k] - x[N+N] <= 0
for k in range(N):
    Aub_xmax[k, N + k] = 1
    Aub_xmax[k, N + N] = -1
    bub_xmax[k] = 0


# x_min
Aub_xmin = np.zeros([N, N + N + 2])
bub_xmin = np.zeros(N)

# -state[k] + state_min <= 0
# -x[N+k] + x[N+N+1] <= 0
for k in range(N):
    Aub_xmin[k, N + k] = -1
    Aub_xmin[k, N + N + 1] = 1
    bub_xmin[k] = 0


# Storage "dynamics"
Aeq_dyn = np.zeros([N, N + N + 2])
beq_dyn = np.zeros(N)

# state[k+1] - state[k] + demand[k] = H2_gen[k]
# x[N+k+1] - x[N+k] + x[k] = beq_dyn[k]
for k in range(N):
    if (k + 1) == N:
        break
    Aeq_dyn[k, N + k + 1] = 1
    Aeq_dyn[k, N + k] = -1
    Aeq_dyn[k, k] = 1

    beq_dyn[k] = H2_gen[k]

# state[0] = state[N]
# -x[N+0] + x[N + N - 1] = 0
Aeq_dyn[N - 1, N] = -1
Aeq_dyn[N - 1, 2 * N - 1] = 1


A_ub = np.concatenate([Aub_ramp_pos, Aub_ramp_neg, Aub_xmax, Aub_xmin])
b_ub = np.concatenate([bub_ramp_pos, bub_ramp_neg, bub_xmax, bub_xmin])

# A_ub = np.flip(A_ub, axis=0)
# b_ub = np.flip(b_ub)

# A_ub *= 1 / (10 * R)
# b_ub *= 1 / (10 * R)

A_eq = Aeq_dyn
b_eq = beq_dyn

bounds = [(bound_l[i], bound_u[i]) for i, bl in enumerate(bound_l)]

if print_matrices:
    pprint.pprint("bound_l:")
    pprint.pprint(bound_l)
    pprint.pprint("bound_u:")
    pprint.pprint(bound_u)
    pprint.pprint("Aub_ramp_pos:")
    pprint.pprint(Aub_ramp_pos)
    pprint.pprint("bub_ramp_pos:")
    pprint.pprint(bub_ramp_pos)
    pprint.pprint("Aub_ramp_neg:")
    pprint.pprint(Aub_ramp_neg)
    pprint.pprint("bub_ramp_neg:")
    pprint.pprint(bub_ramp_neg)
    pprint.pprint("Aub_xmax:")
    pprint.pprint(Aub_xmax)
    pprint.pprint("bub_xmax:")
    pprint.pprint(bub_xmax)
    pprint.pprint("Aub_xmin:")
    pprint.pprint(Aub_xmin)
    pprint.pprint("bub_xmin:")
    pprint.pprint(bub_xmin)
    pprint.pprint("Aeq_dyn:")
    pprint.pprint(Aeq_dyn)
    pprint.pprint("beq_dyn:")
    pprint.pprint(beq_dyn)

res = linprog(
    c=C,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    bounds=bounds,
)
print(res.message)
print(f"Capacity: {(res.x[-2] - res.x[-1])}")
print(f"Integral: {np.sum(res.x[N:2*N])}")

time = np.linspace(0, N - 1, N)

fig, ax = plt.subplots(3, 1, sharex="col")
ax[0].hlines(
    [min_demand, max_demand], 0, time[-1], alpha=0.5, linewidth=0.5, color="black"
)

ax[0].plot(time, H2_gen)
ax[0].plot(time, res.x[0:N])
ax[1].plot(time, res.x[N : 2 * N])

ax[2].hlines([-R, R], 0, time[-1], alpha=0.5, linewidth=0.5, color="black")
ramp = res.x[0:N] - np.roll(res.x[0:N], 1)
ax[2].plot(time[1:], ramp[1:])


fig.suptitle(
    f"Capacity: {(res.x[-2] - res.x[-1]):.2f}, Integral: {np.sum(res.x[N:2*N]):.2f}"
)


fig, ax = plt.subplots(5, 1)
fig.suptitle("Residuals (slack)")

ax[0].plot(time, res.eqlin.residual)


ax[1].plot(time, res.ineqlin.residual[0:N])
ax[1].plot(time, res.ineqlin.residual[N : 2 * N])

ax[2].plot(time, res.ineqlin.residual[2 * N : 3 * N])
ax[2].plot(time, res.ineqlin.residual[3 * N : 4 * N])

ax[3].plot(time, res.upper.residual[0:N])
ax[3].plot(time, res.lower.residual[0:N])

ax[4].plot(time, res.upper.residual[N : 2 * N])
ax[4].plot(time, res.lower.residual[N : 2 * N])


ax[0].set_ylabel("dynamics")
ax[1].set_ylabel("ramp rate")
ax[2].set_ylabel("xmax/xmin")
ax[3].set_ylabel("demand bounds")
ax[4].set_ylabel("state bounds")

fig, ax = plt.subplots(5, 1)
fig.suptitle("Marginals (duals)")

ax[0].plot(time, res.eqlin.marginals)

ax[1].plot(time, res.ineqlin.marginals[0:N])
ax[1].plot(time, res.ineqlin.marginals[N : 2 * N])

ax[2].plot(time, res.ineqlin.marginals[2 * N : 3 * N])
ax[2].plot(time, res.ineqlin.marginals[3 * N : 4 * N])

ax[3].plot(time, res.upper.marginals[0:N])
ax[3].plot(time, res.lower.marginals[0:N])

ax[4].plot(time, res.upper.marginals[N : 2 * N])
ax[4].plot(time, res.lower.marginals[N : 2 * N])

ax[0].set_ylabel("dynamics")
ax[1].set_ylabel("ramp rate")
ax[2].set_ylabel("xmax/xmin")
ax[3].set_ylabel("demand bounds")
ax[4].set_ylabel("state bounds")

plt.show()

[]


# ===================================== from storage_opt.py ============================

rootdir = Path(__file__).parents[2]
H2_gen = np.load(rootdir / "data" / "hybrid_gen.npy")

n_steps = len(H2_gen)
n_steps = 100
time = np.linspace(0, n_steps - 1, n_steps)
shift = 0
H2_gen = H2_gen[shift : n_steps + shift]

rl = 0.9
td = 0.8


def run_opt(rl, td):
    max_demand = (2 / (td + 1)) * np.mean(H2_gen)
    min_demand = td * max_demand
    ramp_lim = rl * max_demand

    c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])

    A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
    b_ub = np.zeros([n_steps * 4])
    A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
    b_eq = np.zeros(n_steps + 1)

    # Generate constraint matrices
    A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]
    for i in range(n_steps):
        # Positive ramp rate constraint
        A_ub[i, [i + n_steps, -2]] = [1, -1]
        # Negative ramp rate constraint
        A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

        if i > 0:
            A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
            A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
        b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [
            ramp_lim,
            ramp_lim,
        ]

        b_eq[i] = H2_gen[i]
        if i == 0:
            A_eq[0, [0, n_steps]] = [1, 1]
            continue
        A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

    # bound_low = [min_demand] * n_steps + [None] * (n_steps + 2)
    bound_low = [min_demand] * n_steps + [0] * n_steps + [None] * 2
    bound_up = [max_demand] * n_steps + [None] * (n_steps + 2)
    bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
    # print(res.message)
    x = res.x
    success = res.success
    return min_demand, max_demand, res


min_demand, max_demand, res = run_opt(rl, td)


rl = 0.9

count = 0
turndown_0 = 0
td_prev = turndown_0
td_prev_success = 0
td_prev_failure = 1
td = 0.1

while True:
    min_demand, max_demand, res = run_opt(rl, td)
    if res.success:
        td_prev_success = td
        td += (td_prev_failure - td) / 2
    else:
        td_prev_failure = td
        td -= (td - td_prev_success) / 2
    if ((td_prev_failure - td_prev_success) < 1e-6) or (count > 100):
        break
    count += 1

min_demand, max_demand, res = run_opt(rl, td_prev_success)

print(f"turndown: {td_prev_success:.4f}")


fig, ax = plt.subplots(3, 1, sharex="col")

ax[0].hlines(
    [min_demand, max_demand], 0, time[-1], alpha=0.5, linewidth=0.5, color="black"
)

ax[0].plot(time, H2_gen)
ax[0].plot(time, res.x[0:n_steps])
# ax[0].plot([0, time[-1]], [min_demand, max_demand], label="limiting case")

ax[1].plot(time, res.x[n_steps : 2 * n_steps])
ax1_ylim = ax[1].get_ylim()
ax[1].set_ylim([ax1_ylim[0], 2 * ax1_ylim[1]])
# ax[1].plot(
#     time, np.cumsum(np.mean(H2_gen) - np.linspace(min_demand, max_demand, len(time)))
# )

# What is the limiting case? min_demand for the first half of the hydrogen storage, then max demand for the second half

idx_switch = np.argmin(np.abs(np.cumsum(H2_gen) - np.sum(H2_gen) / 2))
demand_limit = np.concatenate(
    [min_demand * np.ones(idx_switch), max_demand * np.ones(n_steps - idx_switch)]
)


# ax[1].plot(time, np.cumsum(H2_gen - demand_limit))

ax[1].plot(time, np.cumsum(H2_gen - min_demand))
ax[1].plot(
    time, np.cumsum(H2_gen - max_demand) - np.min(np.cumsum(H2_gen - max_demand))
)
fig.suptitle(f"Capacity: {(res.x[-2] - res.x[-1]):.2f}")

plt.show()

# TODO: Steady should always be a feasible solution if not optimal. Can I test the constraints on the steady solution signal for feasibility? Or better yet, can I start the optimization with an initial guess of the steady solution vector


[]
# == FIRST TRY DEVELOPING OPTIMIZATION ==
# ==vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv==

# c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])

# # c = np.concatenate([np.zeros(n_steps), [-1], np.zeros(n_steps - 2), [1, 1, -1]])

# A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
# b_ub = np.zeros([n_steps * 4])

# ramp_lim = 1 / (2) * max_diff

# for i in range(n_steps):
#     A_ub[i, [i + n_steps, -2]] = [1, -1]
#     A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

#     if i > 0:
#         A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
#         A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
#     b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [ramp_lim, ramp_lim]

# A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
# b_eq = np.zeros(n_steps + 1)

# # A_eq[-1, [n_steps + 1, 2 * n_steps]] = [1, -1]

# for i in range(n_steps):
#     b_eq[i] = gen[i]
#     if i == 0:
#         A_eq[0, [0, n_steps]] = [1, 1]
#         continue
#     A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

# A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]


# plant_min = 0.1
# plant_rating = np.mean(gen) * ((1 - plant_min) / 2 + 1)
# min_demand = plant_min * plant_rating
# max_demand = plant_rating

# # bound_low = np.concatenate([min_demand * np.ones(n_steps), [None] * (n_steps + 2)])
# # bound_up = np.concatenate([max_demand * np.ones(n_steps), [None] * (n_steps + 2)])

# bound_low = [min_demand] * n_steps + [None] * (n_steps + 2)
# bound_up = [max_demand] * n_steps + [None] * (n_steps + 2)

# bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

# x0 = np.concatenate([np.mean(gen) * np.ones(n_steps), np.zeros(n_steps), [1e3, -1e3]])
# res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)  # , x0=x0)

# if not res.success:
#     print(res.message)
#     exit()

# # print(res.x)
# print(np.max(np.abs(A_eq @ res.x - b_eq)))
# print(np.max(A_ub @ res.x - b_ub))

# demand = res.x[0:n_steps]
# charge = res.x[n_steps : 2 * n_steps]
# storage_size = res.x[-2] - res.x[-1]

# print(f"storage size: {storage_size:.2f} (kg)")

# fig, ax = plt.subplots(2, 1, sharex="col")

# ax[0].plot(gen[0:n_steps], label="generation", linewidth=0.5)
# ax[0].plot(demand, label="demand", linewidth=2)
# ax[0].plot(gen[0:n_steps] - demand, label="chg/dchg", linewidth=0.5)
# ax[0].hlines([min_demand, max_demand], 0, n_steps, color="black", alpha=0.25)

# ax[0].legend()
# ax[0].set_title("demand")
# ax[1].plot(charge)
# ax[1].set_title("charge")
# plt.show()

# []

## =============================== from upper_lower_cons_distribution.py ==============

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
