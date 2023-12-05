
"""
TODO:
1. Separate HOPP run so it only needs to be run once.
2. Make it so that the demand optimization can be run all at once. 
3. 

"""

# 1. import modules
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from multiprocess import Pool  # type: ignore
import time
from typing import Union

from hopp.simulation import HoppInterface
from hopp.simulation.technologies.wind.wind_plant import WindPlant
from hopp.simulation.technologies.pv.pv_plant import PVPlant
from hopp.simulation.technologies.battery.battery import Battery
from dynamic_green_ammonia.technologies.chemical import (
    HaberBosch,
    AirSeparationUnit,
    Electrolzyer,
)
from dynamic_green_ammonia.technologies.storage import (
    SteadyStorage,
    DynamicAmmoniaStorage,
)


class RunDL:
    hi: HoppInterface
    wind: WindPlant
    pv: PVPlant
    EL: Electrolzyer
    ASU: AirSeparationUnit
    HB: HaberBosch
    battery: Battery
    storage: Union[SteadyStorage, DynamicAmmoniaStorage]

    hybrid_generation: np.ndarray
    # site: HOPP SITE

    def __init__(
        self,
        hopp_input,
        ammonia_ramp_limit,
        ammonia_plant_turndown_ratio,
        dynamic_load=True,
    ):
        self.hopp_input = hopp_input

        self.plant_life = 30
        self.ammonia_ramp_limit = ammonia_ramp_limit
        self.ammonia_plant_turndown_ratio = ammonia_plant_turndown_ratio
        self.dynamic_load = dynamic_load

        self.LCOA_dict = {}
        self.main_dict = {}

    def run_HOPP(self):
        dir_path = Path(__file__).parents[1]

        self.hi = HoppInterface(dir_path / self.hopp_input)
        self.hi.simulate(self.plant_life)

        self.wind = self.hi.system.wind
        self.pv = self.hi.system.pv

        simulation_length = 8760
        wind_generation = np.array(self.wind.generation_profile[0:simulation_length])
        solar_generation = np.array(self.pv.generation_profile[0:simulation_length])
        self.hybrid_generation = wind_generation + solar_generation

    def split_power(self):
        # energy per kg NH3 from each component
        # N2 + 3 H2 <-> 2 NH3

        # TODO: calculate actual HB and ASU profiles after calc_storage

        self.energypkg = np.array(
            [
                self.HB.kgpkg_H2 * self.EL.energypkg_H2,
                self.HB.kgpkg_N2 * self.ASU.energypkg_N2,
                self.HB.energypkg_NH3,
            ]
        )
        self.energypkg /= np.sum(self.energypkg)

        self.P2EL, self.P2ASU, self.P2HB = np.atleast_2d(
            self.energypkg
        ).T @ np.atleast_2d(self.hybrid_generation)

    def calc_H2_gen(self):
        H2_gen = np.zeros(len(self.P2EL))

        for i in range(len(self.P2EL)):
            H2_gen[i], reject = self.EL.step(self.P2EL[i])

        self.H2_gen = H2_gen

    def calc_steady_storage(self, SS: SteadyStorage, siteinfo):
        SS.run(siteinfo)

        self.H2_storage = SS

    def calc_DL_storage(self, DA: DynamicAmmoniaStorage, siteinfo):
        # DA.calc_storage()
        DA.run(siteinfo)

        self.P2ASU, self.P2HB = np.atleast_2d(
            self.energypkg[1:] / np.sum(self.energypkg[1:])
        ).T @ np.atleast_2d(DA.P_out)

        self.H2_storage_out = DA.H2_out

        self.H2_storage = DA

    def calc_chemicals(self):
        powers = np.zeros([len(self.P2EL), 3])
        chemicals = np.zeros([len(self.P2EL), 3])
        signals = np.zeros([len(self.P2EL), 3])

        for i in range(len(self.P2EL)):
            # H2, EL_reject = EL.step(P2EL[i])
            N2, ASU_reject = self.ASU.step(self.P2ASU[i])
            NH3, HB_reject = self.HB.step(self.H2_storage_out[i], N2, self.P2HB[i])

            powers[i, :] = [self.P2EL[i], self.P2ASU[i], self.P2HB[i]]
            chemicals[i, :] = [self.H2_storage_out[i], N2, NH3]

        H2_tot, N2_tot, NH3_tot = np.sum(chemicals, axis=0)
        H2_max, N2_max, NH3_max = np.max(chemicals, axis=0)
        P_EL_max, P_ASU_max, P_HB_max = np.max(powers, axis=0)

        self.EL.calc_financials(P_EL_max, H2_max)
        self.ASU.calc_financials(P_ASU_max, N2_max)
        self.HB.calc_financials(P_HB_max, NH3_max)
        self.totals = [H2_tot, N2_tot, NH3_tot]
        self.chemicals = chemicals
        self.powers = powers

    def calc_LCOA(self):
        pipe_blacklist = [
            "lined_capex",
            "lined_opex",
            "salt_capex",
            "salt_opex",
            "LT_NH3",
        ]
        lined_blacklist = [
            "pipe_capex",
            "pipe_opex",
            "salt_capex",
            "salt_opex",
            "LT_NH3",
        ]
        salt_blacklist = [
            "lined_capex",
            "lined_opex",
            "pipe_capex",
            "pipe_opex",
            "LT_NH3",
        ]

        C_pipe = np.sum(
            [
                value
                for key, value in self.LCOA_dict.items()
                if key not in pipe_blacklist
            ]
        )
        C_lined = np.sum(
            [
                value
                for key, value in self.LCOA_dict.items()
                if key not in lined_blacklist
            ]
        )
        C_salt = np.sum(
            [
                value
                for key, value in self.LCOA_dict.items()
                if key not in salt_blacklist
            ]
        )

        self.LCOA_pipe = C_pipe / self.LCOA_dict["LT_NH3"]
        self.LCOA_lined = C_lined / self.LCOA_dict["LT_NH3"]
        self.LCOA_salt = C_salt / self.LCOA_dict["LT_NH3"]

    def build_LCOA_dict(self):
        self.LCOA_dict.update(
            {
                "wind_capex": self.wind.cost_installed,
                "wind_opex": np.sum(self.wind.om_total_expense),
                "pv_capex": self.pv.cost_installed,
                "pv_opex": np.sum(self.pv.om_total_expense),
                "pipe_capex": self.H2_storage.pipe_capex,
                "pipe_opex": self.H2_storage.pipe_opex,
                "lined_capex": self.H2_storage.lined_capex,
                "lined_opex": self.H2_storage.lined_opex,
                "salt_capex": self.H2_storage.salt_capex,
                "salt_opex": self.H2_storage.salt_opex,
                "EL_capex": self.H2_storage.EL_capex,
                "EL_opex": self.H2_storage.EL_opex,
                "battery_capex": self.H2_storage.battery_capex,
                "battery_opex": np.sum(self.H2_storage.battery_opex),
                "ASU_capex": self.ASU.capex,
                "ASU_opex": self.ASU.opex * self.plant_life,
                "HB_capex": self.HB.capex,
                "HB_opex": self.HB.opex * self.plant_life,
                "LT_NH3": self.LT_NH3,
            }
        )

    def build_main_dict(self):
        self.main_dict.update(
            {
                "storage_capacity_kg": self.H2_storage.H2_capacity_kg,
                "storage_flow_rate_kgphr": self.H2_storage.H2_chg_capacity,
                "storage_soc_f": self.H2_storage.H2_soc[-1],
                "plant_life": self.plant_life,
                "P_max_EL": np.max(self.P2EL),
                "H2_gen_max": np.max(self.H2_gen),
                "H2_max": np.max(self.chemicals[:, 0]),
                "P_EL_max": np.max(self.powers[:, 0]),
                "N2_max": np.max(self.chemicals[:, 1]),
                "P_ASU_max": np.max(self.powers[:, 1]),
                "NH3_max": np.max(self.chemicals[:, 2]),
                "P_HB_max": np.max(self.powers[:, 2]),
                "ramp_lim": self.ammonia_ramp_limit,
                "plant_min": self.ammonia_plant_turndown_ratio,
                "LCOA_pipe": self.LCOA_pipe,
                "LCOA_lined": self.LCOA_lined,
                "LCOA_salt": self.LCOA_salt,
                "wind_rating": self.wind.system_capacity_kw,
                "pv_rating": self.pv.system_capacity_kw,
                "EL_rating": self.EL.rating_elec,
                "H2_storage_rating": self.H2_storage.H2_capacity_kg,
                "battery_rating": self.H2_storage.P_capacity,
                "ASU_rating": self.ASU.rating_elec,
                "HB_rating": self.HB.rating_elec,
            }
        )
        self.main_dict.update(self.LCOA_dict)

    def run(self, ramp_lim=None, plant_min=None):
        if not (plant_min == None):
            self.ammonia_plant_turndown_ratio = plant_min
        if not (ramp_lim == None):
            self.ammonia_ramp_limit = ramp_lim

        # 2. inputs and outputs paths
        # 3. run HOPP
        # 4. collect generation data

        if not hasattr(self, "hi"):
            self.run_HOPP()

        # 5. split power
        dt = 3600  # [s]
        rating = 1e9

        self.EL = Electrolzyer(dt, rating)
        self.ASU = AirSeparationUnit(dt, rating)
        self.HB = HaberBosch(dt, rating)

        self.split_power()

        # 6 calculate hydrogen generation
        self.calc_H2_gen()

        # 7. calculate hydrogen storage
        # 8. calculate electricity storage

        if self.dynamic_load:
            DA = DynamicAmmoniaStorage(
                self.H2_gen,
                self.P2EL,
                self.P2ASU + self.P2HB,
                ramp_lim=self.ammonia_ramp_limit,
                plant_min=self.ammonia_plant_turndown_ratio,
            )
            self.calc_DL_storage(DA, self.hi.system.site)

        else:
            SS = SteadyStorage(self.H2_gen, self.P2EL, self.P2ASU + self.P2HB)
            self.calc_steady_storage(SS, self.hi.system.site)

        self.battery = self.H2_storage.battery

        self.H2_HB = self.H2_storage.H2_out
        self.P_HB = self.H2_storage.P_out * (
            self.energypkg[2] / np.sum(self.energypkg[1:3])
        )
        self.P_ASU = self.H2_storage.P_out * (
            self.energypkg[1] / np.sum(self.energypkg[1:3])
        )

        # 8.5 calculate ASU and HB operation
        self.calc_chemicals()

        # 9. calculate HOPP costs
        # 10. calculate storage costs
        # 11. calculate ammonia costs
        self.LT_NH3 = self.totals[2] * 1e-3 * self.plant_life  # [t]
        self.build_LCOA_dict()
        self.calc_LCOA()
        self.build_main_dict()
        # should be about 771 USD/t NH3 (Cesaro 2021)

        self.main_df = pd.DataFrame(self.main_dict, index=[0])

        []


def FlexibilityParameters(analysis="simple", n_ramps=1, n_tds=1):
    """Generate the ramp limits and turndown ratios for an analysis sweep

    inputs:
    analysis: either "simple" or "full sweep" which type of analysis the user is performing
    n_ramps: number of ramp limits to return
    n_tds: number of turndown ratios to return

    returns:
    ramp_lims: array of ramp_limits to simulate
    TD_ratios: array of turndown ratios to simulate
    """

    if analysis == "simple":
        # 5 cases

        ramp_lims = [1, 0.01, 0.99, 0]
        turndowns = [0, 0.01, 0.99, 1]

    elif analysis == "full_sweep":
        ramp_lims = np.concatenate([[0], np.logspace(-6, 0, 7)])
        turndowns = np.linspace(0, 1, 8)
        # if n_ramps > 1:
        #     ramp_lims = np.exp(np.linspace(np.log(0.01), np.log(1), n_ramps))
        # else:
        #     ramp_lims = [0.1]

        # if n_tds > 1:
        #     turndowns = 1 - np.logspace(np.log10(0.99), -2, n_tds)
        # else:
        #     turndowns = [0.25]

    return ramp_lims, turndowns
