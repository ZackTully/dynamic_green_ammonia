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
