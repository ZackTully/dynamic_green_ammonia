# %% imports

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (15, 8)


# %% define cost equation


def cost(a: float, b: float, c: float, cap):
    c_pkg = np.exp(a * (np.log(cap / 1e3)) ** 2 - b * np.log(cap / 1e3) + c)
    c_tot = c_pkg * cap
    return c_pkg, c_tot


n_a = 100
n_b = 100
n_c = 100
n_cap = 100

constant_a = 95
constant_b = 40
constant_c = 50

a = np.logspace(-3, -0.9, n_a)
b = np.linspace(0, 2, n_b)
c = np.linspace(5, 15, n_c)
caps = np.logspace(3, 7, n_cap)

color_start = [val / 255 for val in (28, 99, 214)]
color_end = [val / 255 for val in (217, 183, 13)]

# %%

c_pkg, c_tot = cost(a[-1], b[4], c[5], caps)

fig, ax = plt.subplots(2, 1, sharex="col")

ax[0].plot(caps, c_pkg)
ax[1].plot(caps, c_tot)

for axis in ax:
    axis.set_xscale("log")
    axis.set_yscale("log")

# %%%

fig, ax = plt.subplots(2, 1, sharex="col")
fig.suptitle(f"Sweeping a, b={b[constant_b]:.4f}, c={c[constant_c]:.4f}")

colors = np.linspace(color_start, color_end, n_a)

for i in range(len(a)):
    c_pkg, c_tot = cost(a[i], b[constant_b], c[constant_c], caps)
    ax[0].plot(caps, c_pkg, color=colors[i], label=f"a={i}")
    ax[1].plot(caps, c_tot, color=colors[i], label=f"a={i}")

    ax[0].plot(caps[np.argmin(c_pkg)], c_pkg[np.argmin(c_pkg)], "k.")
    ax[1].plot(caps[np.argmin(c_tot)], c_tot[np.argmin(c_tot)], "k.")

    if (i == 0) or (i == (len(a) - 1)):
        ax[0].text(caps[-1], c_pkg[-1], f"{a[i]:.4f}")
        ax[1].text(caps[-1], c_tot[-1], f"{a[i]:.4f}")

for axis in ax:
    axis.set_xscale("log")
    axis.set_yscale("log")

ax[1].set_xlabel("Storage Capacity [kg]")
ax[1].set_ylabel("Total cost [USD]")
ax[0].set_ylabel("per kg cost [USD]")

fig.savefig("../../plots/papadias_cost_scaling/a_sweep.png", format="png")

# %%
fig, ax = plt.subplots(2, 1, sharex="col")
fig.suptitle(f"Sweeping b, a={a[constant_a]:.4f}, c={c[constant_c]:.4f}")

colors = np.linspace(color_start, color_end, n_b)

flag = False

for i in range(len(b)):
    c_pkg, c_tot = cost(a[constant_a], b[i], c[constant_c], caps)

    if (c_tot[-1] < c_tot[0]) & (not flag):
        print(b[i])
        flag = True

    ax[0].plot(caps, c_pkg, color=colors[i], label=f"b={i}")
    ax[1].plot(caps, c_tot, color=colors[i], label=f"b={i}")

    ax[0].plot(caps[np.argmin(c_pkg)], c_pkg[np.argmin(c_pkg)], "k.")
    ax[1].plot(caps[np.argmin(c_tot)], c_tot[np.argmin(c_tot)], "k.")

    if (i == 0) or (i == (len(b) - 1)):
        ax[0].text(caps[-1], c_pkg[-1], f"{b[i]:.4f}")
        ax[1].text(caps[-1], c_tot[-1], f"{b[i]:.4f}")


for axis in ax:
    axis.set_xscale("log")
    axis.set_yscale("log")

ax[1].set_xlabel("Storage Capacity [kg]")
ax[1].set_ylabel("Total cost [USD]")
ax[0].set_ylabel("per kg cost [USD]")


fig.savefig("../../plots/papadias_cost_scaling/b_sweep.png", format="png")

# %%

fig, ax = plt.subplots(2, 1, sharex="col")
fig.suptitle(f"Sweeping c, a={a[constant_a]:.4f}, b={b[constant_b]:.4f}")

colors = np.linspace(color_start, color_end, n_c)

for i in range(len(a)):
    c_pkg, c_tot = cost(a[constant_a], b[constant_b], c[i], caps)
    ax[0].plot(caps, c_pkg, color=colors[i], label=f"c={i}")
    ax[1].plot(caps, c_tot, color=colors[i], label=f"c={i}")

    ax[0].plot(caps[np.argmin(c_pkg)], c_pkg[np.argmin(c_pkg)], "k.")
    ax[1].plot(caps[np.argmin(c_tot)], c_tot[np.argmin(c_tot)], "k.")

    if (i == 0) or (i == (len(c) - 1)):
        ax[0].text(caps[-1], c_pkg[-1], f"{c[i]:.4f}")
        ax[1].text(caps[-1], c_tot[-1], f"{c[i]:.4f}")

for axis in ax:
    axis.set_xscale("log")
    axis.set_yscale("log")

ax[1].set_xlabel("Storage Capacity [kg]")
ax[1].set_ylabel("Total cost [USD]")
ax[0].set_ylabel("per kg cost [USD]")

fig.savefig("../../plots/papadias_cost_scaling/c_sweep.png", format="png")

# %%

fig, ax = plt.subplots(2, 1, sharex="col")

# pipe
a_pipe = 0.001559
b_pipe = 0.035313
c_pipe = 4.5183

c_pkg_pipe, c_tot_pipe = cost(a=a_pipe, b=b_pipe, c=c_pipe, cap=caps)

# lined
a_lined = 0.092286
b_lined = 1.5565
c_lined = 8.4658

c_pkg_lined, c_tot_lined = cost(a=a_lined, b=b_lined, c=c_lined, cap=caps)

# salt
a_salt = 0.085863
b_salt = 1.5574
c_salt = 8.1606

c_pkg_salt, c_tot_salt = cost(a=a_salt, b=b_salt, c=c_salt, cap=caps)

ax[0].plot(caps, c_pkg_pipe, label="pipe")
ax[0].plot(caps, c_pkg_lined, label="lined")
ax[0].plot(caps, c_pkg_salt, label="salt")
ax[0].set_xscale("log")
# ax[0].set_yscale('log')
ax[0].legend()
ax[0].set_ylim([-10, 1000])

ax[1].plot(caps, c_tot_pipe, label="pipe")
ax[1].plot(caps, c_tot_lined, label="lined")
ax[1].plot(caps, c_tot_salt, label="salt")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].legend()


ax[0].plot(caps[np.argmin(c_pkg_pipe)], c_pkg_pipe[np.argmin(c_pkg_pipe)], "k.")
ax[1].plot(caps[np.argmin(c_tot_pipe)], c_tot_pipe[np.argmin(c_tot_pipe)], "k.")
ax[0].plot(caps[np.argmin(c_pkg_lined)], c_pkg_lined[np.argmin(c_pkg_lined)], "k.")
ax[1].plot(caps[np.argmin(c_tot_lined)], c_tot_lined[np.argmin(c_tot_lined)], "k.")
ax[0].plot(caps[np.argmin(c_pkg_salt)], c_pkg_salt[np.argmin(c_pkg_salt)], "k.")
ax[1].plot(caps[np.argmin(c_tot_salt)], c_tot_salt[np.argmin(c_tot_salt)], "k.")

ax[0].plot([500000, 500000], [0, ax[0].get_ylim()[1]])
ax[1].plot([500000, 500000], [0, ax[1].get_ylim()[1]])


ax[1].set_xlabel("Storage Capacity [kg]")
ax[1].set_ylabel("Total cost [USD]")
ax[0].set_ylabel("per kg cost [USD]")

fig.savefig("../../plots/papadias_cost_scaling/actual.png", format="png")
# %%

"""
Papadias lined rock cavern base storage size = 500,000 from Table 2
"""

def annual_cost_papa(a, b, c, m):
    """
    a,b,c: equation parameters
    m: storage capacity in tonnes of H2

    returns:
    C: annual cost in 2019 USD/kg-H2 stored
    """

    cost = np.exp(a * (np.log(m))**2 - b * np.log(m) + c)
    return cost


m = np.logspace(0, 4, 100)

params_pipe = (0.001559, 0.035313, 4.5183)
params_LRC = (0.092286, 1.5565, 8.4658)
params_salt = (0.085863, 1.5574, 8.1606)


pipe_pkg = annual_cost_papa(*params_pipe, m)
lined_pkg = annual_cost_papa(*params_LRC, m)


salt_pkg = annual_cost_papa(*params_salt, m)

fig, ax = plt.subplots(1, 2)

ax[0].plot(m, pipe_pkg)
ax[0].plot(m, lined_pkg)
ax[0].plot(m, salt_pkg)


def plot_dot1(params, m):
    ax[0].plot(m, annual_cost_papa(*params, m), "k.")

plot_dot1(params_LRC, 600)
plot_dot1(params_salt, 1000)
plot_dot1(params_LRC, 500)
plot_dot1(params_salt, 500)



mass = 500

def plot_dot2(params, m):
    ax[1].plot(m, m*annual_cost_papa(*params, m), "k.")

ax[1].plot(m, m*pipe_pkg)
ax[1].plot(m, m*lined_pkg)
ax[1].plot(m, m*salt_pkg)

plot_dot2(params_LRC, 500)
plot_dot2(params_salt, 500)

pipe_min = np.argmin(m*annual_cost_papa(*params_pipe, m))
lined_min = np.argmin(m*annual_cost_papa(*params_LRC, m))
salt_min = np.argmin(m*annual_cost_papa(*params_salt, m))

ax[1].plot(m[pipe_min], m[pipe_min] * annual_cost_papa(*params_pipe, m[pipe_min]), "k.")
ax[1].plot(m[lined_min], m[lined_min] * annual_cost_papa(*params_LRC, m[lined_min]), "k.")
ax[1].plot(m[salt_min], m[salt_min] * annual_cost_papa(*params_salt, m[salt_min]), "k.")



ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlim([1, 5000])
ax[0].set_ylim([1, 100])

ax[1].set_xscale("log")
ax[1].set_yscale("log")
# ax[0].set_xlim([1, 5000])
# ax[0].set_ylim([1, 100])



# ============================= old code down here ====================================
# =====================================================================================

from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage import (
    underground_pipe_storage,
)
from hopp.simulation.technologies.hydrogen.h2_storage.salt_cavern import salt_cavern
from hopp.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern import (
    lined_rock_cavern,
)


n_points = 100
# sweep through
# 1. H2_storage_kg (steady needs about 200000 kg )
base_storage = 2e5
H2_storage_kg = np.logspace(0, 7, n_points)

# 2. storage_duration_hrs
# 3. flow_kg_hr (steady max = 1577, min = -3919)
# 4. system_flow_rate (steady max = 1577, min = -3919, kg/hr or 94056 kg/day)


sweep_var = H2_storage_kg

PS_costs = np.zeros([n_points, 4])
LS_costs = np.zeros([n_points, 4])
SS_costs = np.zeros([n_points, 4])

plant_life = 25

for i in range(len(sweep_var)):
    storage_dict = {
        "H2_storage_kg": H2_storage_kg[i],
        # "storage_duration_hrs": 10,
        # "flow_rate_kg_hr": 10,
        "compressor_output_pressure": 100,
        "system_flow_rate": 1e5,
        "model": "papadias",
    }

    PS = underground_pipe_storage.UndergroundPipeStorage(storage_dict)
    PS_capex = np.sum(PS.pipe_storage_capex()[1:3])
    PS_capex_pu = PS.pipe_storage_capex()[0]
    PS_opex = PS.pipe_storage_opex()
    PS_costs[i, :] = [PS_capex + plant_life * PS_opex, PS_capex, PS_opex, PS_capex_pu]

    LS = lined_rock_cavern.LinedRockCavernStorage(storage_dict)
    LS_capex = np.sum(LS.lined_rock_cavern_capex()[1:3])
    LS_capex_pu = LS.lined_rock_cavern_capex()[0]
    LS_opex = LS.lined_rock_cavern_opex()
    LS_costs[i, :] = [LS_capex + plant_life * LS_opex, LS_capex, LS_opex, LS_capex_pu]

    SS = salt_cavern.SaltCavernStorage(storage_dict)
    SS_capex = np.sum(SS.salt_cavern_capex()[1:3])
    SS_capex_pu = SS.salt_cavern_capex()[0]
    SS_opex = SS.salt_cavern_opex()
    SS_costs[i, :] = [SS_capex + plant_life * SS_opex, SS_capex, SS_opex, SS_capex_pu]


# fig, ax = plt.subplots(3, 1, sharex="col")
# ax[0].set_xscale("log")

# ax[0].plot(sweep_var, PS_costs[:, 0])
# ax[0].plot(base_storage, np.interp(base_storage, sweep_var, PS_costs[:, 0]), "k.")
# ax[0].set_yscale("log")
# ax[0].set_title("")

# ax[1].plot(sweep_var, LS_costs[:, 0])
# ax[1].plot(base_storage, np.interp(base_storage, sweep_var, LS_costs[:, 0]), "k.")
# ax[1].set_yscale("log")

# ax[2].plot(sweep_var, SS_costs[:, 0])
# ax[2].plot(base_storage, np.interp(base_storage, sweep_var, SS_costs[:, 0]), "k.")
# ax[2].set_yscale("log")


fig, ax = plt.subplots(2, 1, sharex="col")
a = 0.0041617
b = 0.060369
c = 6.4581
caps = np.logspace(0, 9, 100)


log2_term = lambda cap: (a * np.log(cap / 1e3)) ** 2
log_term = lambda cap: b * np.log(cap / 1e3)
exp_guts = lambda cap: log2_term(cap) - log_term(cap) + c
c_pu = lambda cap: np.exp(exp_guts(cap))
c_tot = lambda cap: cap * c_pu(cap)


def dfunc_dcap(func, cap, dcap: float):
    return (func(cap + dcap) - func(cap)) / dcap


def plot_func(ax, func, x_vals):
    dcap = 10

    ax.plot(x_vals, func(x_vals))
    ax.set_xscale("log")
    ax.set_yscale("log")

    axt = ax.twinx()
    axt.plot(x_vals, dfunc_dcap(func, x_vals, dcap), color="red")
    axt.set_yscale("log")

    try:
        x_zero = np.interp(0, dfunc_dcap(func, x_vals), caps)
        axt.plot(x_zero, 0, "k.")
    except:
        print("tried and failed to find zero")
        pass


# plot_func(ax[0], log_term, caps)
# plot_func(ax[1], log2_term, caps)
# plot_func(ax[2], exp_guts, caps)
plot_func(ax[0], c_pu, caps)

ax[0].plot(H2_storage_kg, PS_costs[:, 3], linestyle="dashed", label="pipe")
ax[0].plot(H2_storage_kg, LS_costs[:, 3], linestyle="dashed", label="lined")
ax[0].plot(H2_storage_kg, SS_costs[:, 3], linestyle="dashed", label="salt")
ax[0].legend()

plot_func(ax[1], c_tot, caps)

ax[1].plot(H2_storage_kg, PS_costs[:, 0], linestyle="dashed", label="pipe")
ax[1].plot(H2_storage_kg, LS_costs[:, 0], linestyle="dashed", label="lined")
ax[1].plot(H2_storage_kg, SS_costs[:, 0], linestyle="dashed", label="salt")

ax[1].legend()

fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.125)

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()