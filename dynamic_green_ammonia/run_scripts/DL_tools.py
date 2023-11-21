from pathlib import Path
import numpy as np
import pandas as pd


from hopp.simulation import HoppInterface
from Dynamic_Load.technologies.storage import SteadyStorage, DynamicAmmoniaStorage

from Dynamic_Load.technologies.chemical import (
    HaberBosch,
    AirSeparationUnit,
    Electrolzyer,
)

def run_HOPP(hopp_input, plant_life):
    dir_path = Path(__file__).parents[1]

    hi = HoppInterface(dir_path / hopp_input)
    hi.simulate(plant_life)

    simulation_length = 8760
    wind_generation = np.array(hi.system.wind.generation_profile[0:simulation_length])
    solar_generation = np.array(hi.system.pv.generation_profile[0:simulation_length])
    hybrid_generation = wind_generation + solar_generation
    np.save("hybrid_gen.npy", hybrid_generation)
    return hi, hybrid_generation


def split_power(EL, ASU, HB, hybrid_generation):
    # energy per kg NH3 from each component
    # N2 + 3 H2 <-> 2 NH3

    # TODO: calculate actual HB and ASU profiles after calc_storage

    energypkg = np.array(
        [
            HB.kgpkg_H2 * EL.energypkg_H2,
            HB.kgpkg_N2 * ASU.energypkg_N2,
            HB.energypkg_NH3,
        ]
    )
    energypkg /= np.sum(energypkg)

    P2EL, P2ASU, P2HB = np.atleast_2d(energypkg).T @ np.atleast_2d(hybrid_generation)

    return P2EL, P2ASU, P2HB, energypkg


def calc_H2_gen(P2EL, EL):
    H2_gen = np.zeros(len(P2EL))

    for i in range(len(P2EL)):
        H2_gen[i], reject = EL.step(P2EL[i])

    return H2_gen


def calc_steady_storage(SS: SteadyStorage, siteinfo):
    SS.calc_storage()
    SS.calc_downstream_signals()
    SS.calc_H2_financials()
    SS.calc_electrolysis_financials()
    SS.calc_battery_financials(siteinfo)

    return SS


def calc_DL_storage(DA: DynamicAmmoniaStorage, siteinfo):
    # DA.calc_storage()
    DA.calc_storage_opt()
    DA.calc_downstream_signals()
    DA.calc_H2_financials()
    DA.calc_electrolysis_financials()
    DA.calc_battery_financials(siteinfo)

    return DA


def calc_chemicals(EL:Electrolzyer, ASU:AirSeparationUnit, HB:HaberBosch, P2EL, P2ASU, P2HB, H2_gen):
    powers = np.zeros([len(P2EL), 3])
    chemicals = np.zeros([len(P2EL), 3])
    signals = np.zeros([len(P2EL), 3])

    for i in range(len(P2EL)):
        # H2, EL_reject = EL.step(P2EL[i])
        N2, ASU_reject = ASU.step(P2ASU[i])
        NH3, HB_reject = HB.step(H2_gen[i], N2, P2HB[i])

        powers[i, :] = [P2EL[i], P2ASU[i], P2HB[i]]
        chemicals[i, :] = [H2_gen[i], N2, NH3]

    H2_tot, N2_tot, NH3_tot = np.sum(chemicals, axis=0)
    H2_max, N2_max, NH3_max = np.max(chemicals, axis=0)
    P_EL_max, P_ASU_max, P_HB_max = np.max(powers, axis=0)

    EL.calc_financials(P_EL_max, H2_max)
    ASU.calc_financials(P_ASU_max, N2_max)
    HB.calc_financials(P_HB_max, NH3_max)

    return EL, ASU, HB, powers, chemicals, [H2_tot, N2_tot, NH3_tot]


def calc_LCOA(LCOA_dict):
    pipe_blacklist = ["lined_capex", "lined_opex", "salt_capex", "salt_opex", "LT_NH3"]
    lined_blacklist = ["pipe_capex", "pipe_opex", "salt_capex", "salt_opex", "LT_NH3"]
    salt_blacklist = ["lined_capex", "lined_opex", "pipe_capex", "pipe_opex", "LT_NH3"]

    C_pipe = np.sum(
        [value for key, value in LCOA_dict.items() if key not in pipe_blacklist]
    )
    C_lined = np.sum(
        [value for key, value in LCOA_dict.items() if key not in lined_blacklist]
    )
    C_salt = np.sum(
        [value for key, value in LCOA_dict.items() if key not in salt_blacklist]
    )

    LCOA_pipe = C_pipe / LCOA_dict["LT_NH3"]
    LCOA_lined = C_lined / LCOA_dict["LT_NH3"]
    LCOA_salt = C_salt / LCOA_dict["LT_NH3"]

    return LCOA_pipe, LCOA_lined, LCOA_salt
