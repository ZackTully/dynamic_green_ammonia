# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

from hopp.simulation import HoppInterface
from hopp.simulation.technologies.hydrogen.electrolysis import run_h2_PEM

from Dynamic_Load.technologies.storage import (
    SteadyStorage,
    DynamicAmmoniaStorage,
    DynamicSteelStorage,
)

from Dynamic_Load.technologies.chemical import HaberBosch, AirSeparationUnit, Electrolzyer
import importlib

# %%


# Hybrid generation data

run_HOPP = True

dir_path = Path(__file__).parents[1]
plant_life = 25

if run_HOPP:
    hopp_input = "inputs/hopp_input.yaml"
    h2_input = "inputs/electrolyzer_input.yaml"
    hi = HoppInterface(dir_path / hopp_input)
    hi.simulate(plant_life)
    HS_outputs = hi.system.hybrid_simulation_outputs()

    with open(dir_path / "outputs/hybrid_system_output.txt", "w") as file:
        file.write(json.dumps(HS_outputs, indent=4))

    simulation_length = 8760
    wind_generation = np.array(hi.system.wind.generation_profile[0:simulation_length])
    solar_generation = np.array(hi.system.pv.generation_profile[0:simulation_length])
    hybrid_generation = wind_generation + solar_generation
    np.save("hybrid_gen.npy", hybrid_generation)
else:
    hybrid_generation = np.load(dir_path / "data" / "hybrid_gen.npy")

time = np.arange(0, len(hybrid_generation), 1)  # hours

# %%
# Split power between electrolysis, air separation, and Haber-Bosch

dt = 3600

EL = Electrolzyer(dt, 100000)
ASU = AirSeparationUnit(dt, 100000)
HB = HaberBosch(dt, 100000)

# energy per kg NH3 from each component
# N2 + 3 H2 <-> 2 NH3

energypkg = np.array(
    [HB.kgpkg_H2 * EL.energypkg_H2, HB.kgpkg_N2 * ASU.energypkg_N2, HB.energypkg_NH3]
)
energypkg /= np.sum(energypkg)

P2EL, P2ASU, P2HB = np.atleast_2d(energypkg).T @ np.atleast_2d(hybrid_generation)

powers = np.zeros([len(hybrid_generation), 3])
chemicals = np.zeros([len(hybrid_generation), 3])
signals = np.zeros([len(hybrid_generation), 3])


for i in range(len(hybrid_generation)):
    H2, EL_reject = EL.step(P2EL[i])
    N2, ASU_reject = ASU.step(P2ASU[i])
    NH3, HB_reject = HB.step(H2, N2, P2HB[i])

    powers[i, :] = [P2EL[i], P2ASU[i], P2HB[i]]
    chemicals[i, :] = [H2, N2, NH3]

H2_tot, N2_tot, NH3_tot = np.sum(chemicals, axis=0)


fig, ax = plt.subplots(3, 1, sharex="col")
ax[0].plot(time, hybrid_generation, linestyle="dashed", color="black", label="P_gen")
ax[0].plot(time, P2EL, label="P_EL")
ax[0].plot(time, P2ASU, label="P_ASU")
ax[0].plot(time, P2HB, label="P_HB")
ax[0].legend()

ax[1].plot(time, chemicals[:, 0], label="H2")
ax[1].plot(time, chemicals[:, 1], label="N2")
ax[1].plot(time, chemicals[:, 2], label="NH3")
ax[1].legend()

ax[2].text(
    0.5,
    0.5,
    f"total energy: {np.sum(hybrid_generation)} [kWh]\nmax power: {np.max(hybrid_generation)} [kW]\ntotal ammonia: {np.sum(chemicals[:,2])} [kg]\nmax ammonia: {np.max(chemicals[:,2])} [kg/hr]",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[2].transAxes,
)

H2_gen = chemicals[:, 1]
power_to_industry = P2ASU + P2HB


# %%


SS = SteadyStorage(H2_gen=H2_gen, power_to_industry=power_to_industry)
SS.calc_storage()
SS.calc_financials()
print("=====================================================")
print("==                 Steady Storage                  ==")
print("=====================================================")
pprint.pprint(SS.summarize_financials())


DS = DynamicSteelStorage(H2_gen=H2_gen, power_to_industry=power_to_industry)
DS.calc_storage()
DS.calc_financials()
print("=====================================================")
print("==                  Steel Storage                  ==")
print("=====================================================")
pprint.pprint(DS.summarize_financials())


DA = DynamicAmmoniaStorage(
    H2_gen=H2_gen, power_to_industry=power_to_industry, ramp_lim=0.1, plant_min=0.1
)
DA.calc_storage()
DA.calc_financials()
print("=====================================================")
print("==                 Ammonia Storage                 ==")
print("=====================================================")
pprint.pprint(DA.summarize_financials())

fig, ax = plt.subplots(3, 3, sharex="col", sharey="col", figsize=(5.5, 3.25))
ax[0, 0].plot(SS.H2_chg)
ax[0, 1].plot(SS.H2_soc)
ax[0, 2].plot(SS.H2_gen + SS.H2_chg)
ax[0, 1].text(
    0.25, 0.8, f"storage: {SS.soc_max:.0f} [kg]", transform=ax[0, 1].transAxes
)


ax[1, 0].plot(DS.H2_chg)
ax[1, 1].plot(DS.H2_soc)
ax[1, 2].plot(DS.H2_gen + DS.H2_chg)
ax[1, 1].text(
    0.25, 0.8, f"storage: {DS.soc_max:.0f} [kg]", transform=ax[1, 1].transAxes
)


ax[2, 0].plot(DA.H2_chg)
ax[2, 1].plot(DA.H2_soc)
ax[2, 2].plot(DA.H2_gen + DA.H2_chg)
ax[2, 1].text(
    0.25, 0.8, f"storage: {DA.soc_max:.0f} [kg]", transform=ax[2, 1].transAxes
)

ax[2, 0].set_xlabel("time (hr)")
ax[2, 1].set_xlabel("time (hr)")
ax[2, 2].set_xlabel("time (hr)")

ax[0, 0].set_title("H2 charge/discharge (kg/hr)")
ax[0, 1].set_title("H2 SOC (kg)")
ax[0, 2].set_title("H2 demand (kg)")

ax[0, 0].set_ylabel("Steady")
ax[1, 0].set_ylabel("Dynamic Steel")
ax[2, 0].set_ylabel("Dynamic Ammonia")

fig.subplots_adjust(right=0.95, top=0.9, wspace=0.4)


# %%

n_points = 25
# ramp_lims = np.linspace(0, 1, n_points)
# ramp_lims = 1 - np.logspace(-1, -.001, n_points)
ramp_lims = np.exp(np.linspace(-10, 0, n_points))


soc_max = np.zeros(n_points)
storage_cost = np.zeros(n_points)


for i in range(len(ramp_lims)):
    DA = DynamicAmmoniaStorage(
        H2_gen=H2_gen,
        power_to_industry=power_to_industry,
        ramp_lim=ramp_lims[i],
        plant_min=0.1,
    )
    DA.calc_storage()
    DA.calc_financials()

    soc_max[i] = DA.soc_max

    storage_cost[i] = (DA.salt_capex[0] + DA.salt_opex) * soc_max[i]


fig, ax = plt.subplots(2, 1, sharex="col")


ax[0].plot(ramp_lims, soc_max, ".-")
ax[0].set_yscale("log")
ax[0].set_xscale("log")


# ax[0].set_xlabel("ramp limit")
ax[0].set_ylabel("H2 storage [kg]")

ax2 = ax[0].twinx()
ax2.plot(ramp_lims, storage_cost, ".-", color="orange")
ax2.set_yscale("log")
ax2.set_ylabel("salt cost")

ax[1].plot(ramp_lims, storage_cost / NH3)
ax[1].set_yscale("log")
# %%

plt.show()


# %%

if run_HOPP:
    wind_capex = hi.system.wind.cost_installed
    wind_opex = hi.system.wind.om_total_expense
    pv_capex = hi.system.pv.cost_installed
    pv_opex = hi.system.pv.om_total_expense

    H2_storage_capex = DA.salt_capex[1]
    H2_storage_opex = DA.salt_opex * np.ones(25)


[]
