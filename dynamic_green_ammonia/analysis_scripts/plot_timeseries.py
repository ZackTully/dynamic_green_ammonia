# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.lines import Line2D
from pathlib import Path
from copy import copy

from dynamic_green_ammonia.technologies.demand import DemandOptimization

# %%

style = "paper"

if style == "paper":
    plt.style.use(
        Path(__file__).parents[1] / "analysis_scripts" / "paper_figs.mplstyle"
    )
elif style == "pres":
    plt.style.use(Path(__file__).parents[1] / "analysis_scripts" / "pres_figs.mplstyle")


H2_gen = np.load(Path(__file__).parents[1] / "data" / "heatmap_runs" / "H2_gen.npy")
H2_gen = H2_gen[0, :]
# H2_gen = H2_gen[0:250]
N = len(H2_gen)
# time = np.linspace(0, N, N)

RL = [0.01, 0.99, 0.01, 0.99]
TD = [0.25, 0.25, 0.90, 0.90]
titles = ["Low flexibility", "High flexibility", "Low flexibility", "Low flexibility"]


# %%

count = 0


DOs = []
xs = []
ress = []
demands = []

d_maxs = []
d_mins = []

for i in range(3):
    rl = RL[count]
    td = TD[count]

    A = np.array([[1, -np.max(H2_gen)], [1, -np.mean(H2_gen)]])
    b = np.array([0, np.mean(H2_gen)])
    coeffs = np.linalg.inv(A) @ b
    d_max = coeffs[0] / (td + coeffs[1])
    d_min = td * coeffs[0] / (td + coeffs[1])

    d_maxs.append(d_max)
    d_mins.append(d_min)

    DO = DemandOptimization(H2_gen, rl * d_max, d_min, d_max)
    x, success, res = DO.optimize()

    demand = x[0:N]

    DOs.append(copy(DO))
    xs.append(copy(xs))
    ress.append(copy(res))
    demands.append(copy(demands))
    count += 1


# %%

count = 0
fig, ax = plt.subplots(2, 3, sharex="all", sharey="row", figsize=(7.2, 4), dpi=200)

t_start = 0
t_end = 100
# time = np.linspace(t_start, t_end, t_end - t_start)
time = np.arange(t_start, t_end, 1)


for i in range(3):
    rl = RL[count]
    td = TD[count]
    DO = DOs[i]
    x = xs[i]
    res = ress[i]
    demand = demands[i]
    d_min = d_mins[i]
    d_max = d_maxs[i]

    ax[0, count].hlines(
        [d_min, d_max],
        time[0],
        time[-1],
        linewidth=0.5,
        color="black",
        linestyle="dashed",
    )
    ax[0, count].plot(time, H2_gen[t_start:t_end], linewidth=0.5, color="blue")
    ax[0, count].plot(time, DO.demand[t_start:t_end], color="orange")
    # ax[count].set_title(
    #     f"{titles[count]},\nR: {rl:.2f}, T: {td:.2f},\n$f_R: {rl:.2f}$, $f_T: {1-td:.2f}$"
    # )
    ax[0, count].set_title(f"$f_R: {rl:.2f}$, $f_T: {1-td:.2f}$")
    ax[1, count].plot(time, DO.state[t_start:t_end])

    count += 1

fig.legend(
    handles=[
        Line2D(
            [0, 0],
            [0, 0],
            linewidth=0.5,
            color="black",
            linestyle="dashed",
            label="Bounds",
        ),
        Line2D([0, 0], [0, 0], color="orange", label="Demand"),
        Line2D([0, 0], [0, 0], color="blue", linewidth=0.5, label="Generation"),
    ],
    bbox_to_anchor=(0.535, 0.49),
    loc="center",
)
ax[0, 0].set_ylabel("Mass flow rate [kg/hr]")
ax[1, 0].set_ylabel("Storage state [kg]")
# ax[].set_ylabel("Mass flow rate [kg/hr]")
ax[1, 0].set_xlabel("time [hr]")
ax[1, 1].set_xlabel("time [hr]")
ax[1, 2].set_xlabel("time [hr]")

fig.tight_layout()

fig.savefig("dynamic_green_ammonia/plots/time_series.png", format="png", dpi=300)


plt.show()


[]

# %%

# %%
