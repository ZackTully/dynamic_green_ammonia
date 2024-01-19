import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.lines import Line2D
from pathlib import Path

style = "paper"

if style == "paper":
    plt.style.use(
        Path(__file__).parents[2] / "analysis_scripts" / "paper_figs.mplstyle"
    )
elif style == "pres":
    plt.style.use(Path(__file__).parent[2] / "analysis_scripts" / "pres_figs.mplstyle")

# plt.rcParams["text.usetex"] = True

from dynamic_green_ammonia.technologies.demand import DemandOptimization

H2_gen = np.load("dynamic_green_ammonia/run_scripts/hybrid_gen.npy")
H2_gen = H2_gen[0:250]
N = len(H2_gen)
time = np.linspace(0, N, N)

# fig, ax = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(8, 6))
fig, ax = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(7.2, 4))
# fig2, ax2 = plt.subplots(2, 2, sharex="all", sharey="all", figsize=(8, 8))

# RL = [0.0001, 0.0001, 0.5, 0.99]
# TD = [0.99, 0.01, 0.99, 1]
# RL = [0.01, 0.01, 0.99, 0.99]
# TD = [0.25, 0.90, 0.25, 0.90]
# titles = ["Low flexibility", "Low flexibility", "High flexibility", "Low flexibility"]
RL = [0.01, 0.99, 0.01, 0.99]
TD = [0.25, 0.25, 0.90, 0.90]
titles = ["Low flexibility", "High flexibility", "Low flexibility", "Low flexibility"]


count = 0


for i in range(2):
    for j in range(2):
        rl = RL[count]
        td = TD[count]

        A = np.array([[1, -np.max(H2_gen)], [1, -np.mean(H2_gen)]])
        b = np.array([0, np.mean(H2_gen)])
        coeffs = np.linalg.inv(A) @ b
        d_max = coeffs[0] / (td + coeffs[1])
        d_min = td * coeffs[0] / (td + coeffs[1])

        DO = DemandOptimization(H2_gen, rl * d_max, d_min, d_max)
        x, success, res = DO.optimize()

        demand = x[0:N]

        ax[i, j].hlines(
            [d_min, d_max],
            time[0],
            time[-1],
            linewidth=0.5,
            color="black",
            linestyle="dashed",
        )
        ax[i, j].plot(time, H2_gen, linewidth=0.5, color="blue")
        ax[i, j].plot(time, DO.demand, color="orange")
        ax[i, j].set_title(
            f"{titles[count]},\nR: {rl:.2f}, T: {td:.2f},\n$f_R: {rl:.2f}$, $f_T: {1-td:.2f}$"
        )

        # axt = ax[i, j].twinx()
        # axt.plot(time + 1.1 * N, DO.state)

        # ax[i, j].plot(time, x[N : 2 * N])

        # print(d_min, d_max)
        # print(res)

        # ax2[i, j].plot(time, DO.state)
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
ax[1, 0].set_ylabel("Mass flow rate [kg/hr]")
ax[1, 0].set_xlabel("time [hr]")
ax[1, 1].set_xlabel("time [hr]")

fig.tight_layout()

fig.savefig(
    "dynamic_green_ammonia/plots/flex_params_example.png", format="png", dpi=300
)

# fig.subplots_adjust(right=0.8)
plt.show()


[]
