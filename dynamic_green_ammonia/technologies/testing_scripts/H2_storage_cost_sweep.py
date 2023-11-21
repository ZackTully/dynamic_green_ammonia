import numpy as np
import matplotlib.pyplot as plt

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


[]
