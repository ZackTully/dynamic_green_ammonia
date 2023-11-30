"""
Try 10 time steps

dem = [dem_0, ..., dem_10]
chg = [chg_0, ..., chg_10]
y_max
y_lim

x = [dem_0, ..., dem_10, chg_0, ..., chg_10, y_max, y_min]
C = [0    , ..., 0     , 0    , ..., 0,    , 1,   , -1   ]



"""


import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from pathlib import Path


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
