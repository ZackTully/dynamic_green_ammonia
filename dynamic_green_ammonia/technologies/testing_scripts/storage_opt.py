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
gen = np.load(rootdir / "data" / "hybrid_gen.npy")
max_diff = np.max(np.abs(gen - np.roll(gen, 1)))
n_steps = 4000


c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])

# c = np.concatenate([np.zeros(n_steps), [-1], np.zeros(n_steps - 2), [1, 1, -1]])

A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
b_ub = np.zeros([n_steps * 4])

ramp_lim = 1 / (2) * max_diff

for i in range(n_steps):
    A_ub[i, [i + n_steps, -2]] = [1, -1]
    A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

    if i > 0:
        A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
        A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
    b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [ramp_lim, ramp_lim]

A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
b_eq = np.zeros(n_steps + 1)

# A_eq[-1, [n_steps + 1, 2 * n_steps]] = [1, -1]

for i in range(n_steps):
    b_eq[i] = gen[i]
    if i == 0:
        A_eq[0, [0, n_steps]] = [1, 1]
        continue
    A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]


plant_min = 0.1
plant_rating = np.mean(gen) * ((1 - plant_min) / 2 + 1)
min_demand = plant_min * plant_rating
max_demand = plant_rating

# bound_low = np.concatenate([min_demand * np.ones(n_steps), [None] * (n_steps + 2)])
# bound_up = np.concatenate([max_demand * np.ones(n_steps), [None] * (n_steps + 2)])

bound_low = [min_demand] * n_steps + [None] * (n_steps + 2)
bound_up = [max_demand] * n_steps + [None] * (n_steps + 2)

bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

x0 = np.concatenate([np.mean(gen) * np.ones(n_steps), np.zeros(n_steps), [1e3, -1e3]])
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)  # , x0=x0)

if not res.success:
    print(res.message)
    exit()

# print(res.x)
print(np.max(np.abs(A_eq @ res.x - b_eq)))
print(np.max(A_ub @ res.x - b_ub))

demand = res.x[0:n_steps]
charge = res.x[n_steps : 2 * n_steps]
storage_size = res.x[-2] - res.x[-1]

print(f"storage size: {storage_size:.2f} (kg)")

fig, ax = plt.subplots(2, 1, sharex="col")

ax[0].plot(gen[0:n_steps], label="generation", linewidth=0.5)
ax[0].plot(demand, label="demand", linewidth=2)
ax[0].plot(gen[0:n_steps] - demand, label="chg/dchg", linewidth=0.5)
ax[0].hlines([min_demand, max_demand], 0, n_steps, color="black", alpha=0.25)

ax[0].legend()
ax[0].set_title("demand")
ax[1].plot(charge)
ax[1].set_title("charge")
plt.show()

[]
