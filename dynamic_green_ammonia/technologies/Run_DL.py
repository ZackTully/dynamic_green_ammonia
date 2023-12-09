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
import pprint


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
    DynamicAmmoniaStorage,
)
from dynamic_green_ammonia.technologies.demand import DemandOptimization


class RunDL:
    hi: HoppInterface
    wind: WindPlant
    pv: PVPlant
    EL: Electrolzyer
    ASU: AirSeparationUnit
    HB: HaberBosch
    battery: Battery
    storage: DynamicAmmoniaStorage

    hybrid_generation: np.ndarray
    # site: HOPP SITE

    def __init__(
        self,
        hopp_input,
        ammonia_ramp_limit,
        ammonia_plant_turndown_ratio,
    ):
        self.hopp_input = hopp_input

        self.plant_life = 30
        self.rl = ammonia_ramp_limit
        self.td = ammonia_plant_turndown_ratio

        self.LCOA_dict = {}
        self.main_dict = {}

        self.HB_sizing = "fraction"

    def re_init(self, hopp_input=None, ramp_lim=None, turndown=None):
        self.LCOA_dict = {}
        self.main_dict = {}

        if ((self.hopp_input != hopp_input) & (hopp_input is not None)) or (
            not hasattr(self, "hi")
        ):
            self.hi, self.hybrid_generation = self.run_HOPP(self.hopp_input)

        dt = 3600  # [s]
        rating = 1e9

        self.EL = Electrolzyer(dt, rating)
        self.ASU = AirSeparationUnit(dt, rating)
        self.HB = HaberBosch(dt, rating)

    def run_HOPP(self, hopp_input):
        dir_path = Path(__file__).parents[1]

        hi = HoppInterface(dir_path / hopp_input)
        hi.simulate(self.plant_life)

        self.wind = hi.system.wind
        self.pv = hi.system.pv

        simulation_length = 8760
        wind_generation = np.array(self.wind.generation_profile[0:simulation_length])
        solar_generation = np.array(self.pv.generation_profile[0:simulation_length])
        hybrid_generation = wind_generation + solar_generation

        return hi, hybrid_generation

    def write_hopp_main_dict(self):
        self.main_dict.update(
            {
                "HOPP": {
                    "wind": {
                        "capex": self.hi.system.wind.cost_installed,
                        "opex": np.sum(self.hi.system.wind.om_total_expense),
                        "rating_kw": self.hi.system.wind.system_capacity_kw,
                        "annual_energy": self.hi.system.wind.annual_energy_kwh,
                        "CF": self.hi.system.wind.capacity_factor,
                        "LCOE": self.hi.system.wind.levelized_cost_of_energy_real,
                    },
                    "pv": {
                        "capex": self.hi.system.pv.cost_installed,
                        "opex": np.sum(self.hi.system.pv.om_total_expense),
                        "rating_kw": self.hi.system.pv.system_capacity_kw,
                        "annual_energy": self.hi.system.pv.annual_energy_kwh,
                        "CF": self.hi.system.pv.capacity_factor,
                        "LCOE": self.hi.system.pv.levelized_cost_of_energy_real,
                    },
                    "site": {
                        "lat": self.hi.system.site.lat,
                        "lon": self.hi.system.site.lon,
                        "year": self.hi.system.site.year,
                    },
                }
            }
        )

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

        P2EL, P2ASU, P2HB = np.atleast_2d(self.energypkg).T @ np.atleast_2d(
            self.hybrid_generation
        )

        return P2EL, P2ASU, P2HB

    def calc_H2_gen(self):
        H2_gen = np.zeros(len(self.P2EL))
        reject = np.zeros(len(self.P2EL))

        for i in range(len(self.P2EL)):
            H2_gen[i], reject[i] = self.EL.step(self.P2EL[i])

        if np.sum(reject) > (1e-2 * np.sum(self.P2EL)):
            print(
                f"Electrolyzer rejected {np.sum(reject) / np.sum(self.P2EL) * 100:.2f} % of power"
            )

        self.main_dict.update(
            {
                "Electrolyzer": {
                    "H2_gen_max": np.max(H2_gen),
                    "H2_gen_tot": np.sum(H2_gen),
                    "Conv kWh/kg": np.sum(self.P2EL) / np.sum(H2_gen),
                    "mean_gen": np.mean(H2_gen),
                    "max_gen": np.max(H2_gen),
                }
            }
        )

        # TODO check how this H2 generation compares with the HOPP electrolyzer

        return H2_gen

    def calc_demand_profile(self):
        if self.HB_sizing == "mean":
            center = np.mean(self.H2_gen)
            max_demand = 2 / (1 + self.td)
            min_demand = self.td * max_demand
        elif self.HB_sizing == "linear interp":
            center = (
                np.mean(self.H2_gen) - np.max(self.H2_gen) / 2
            ) * self.td + np.max(self.H2_gen) / 2
            max_demand = 2 / (1 + self.td)
            min_demand = self.td * max_demand
        elif self.HB_sizing == "fraction":
            A = np.array([[1, -np.max(self.H2_gen)], [1, -np.mean(self.H2_gen)]])
            b = np.array([0, np.mean(self.H2_gen)])
            coeffs = np.linalg.inv(A) @ b
            max_demand = coeffs[0] / (self.td + coeffs[1])
            min_demand = self.td * coeffs[0] / (self.td + coeffs[1])

        ramp_lim = self.rl * max_demand

        self.DO = DemandOptimization(self.H2_gen, ramp_lim, min_demand, max_demand)
        x, success, res = self.DO.optimize()
        if success:
            N = len(self.H2_gen)
            H2_demand = x[0:N]
            H2_storage_state = x[N : 2 * N]
            H2_state_initial = x[N]
            H2_capacity = x[-2] - x[-1]

            self.main_dict.update(
                {
                    "H2 Storage": {
                        # "capacity_kg": H2_capacity,
                        "initial_state_kg": H2_state_initial,
                        "min_demand": min_demand,
                        "max_demand": max_demand,
                        "HB_sizing": self.HB_sizing,
                    }
                }
            )

            return H2_demand, H2_storage_state, H2_state_initial, H2_capacity
        else:
            print("Optimization failed")

    def calc_DL_storage(self, siteinfo):
        DA = DynamicAmmoniaStorage(
            self.H2_gen,
            self.H2_demand,
            self.P_demand,
            self.P2EL,
            self.P2ASU + self.P2HB,
        )

        # TODO correct the last input self.P2ASU + self.P2HB to be the power to the ammonia synthesis, not just the proportional to H2 values

        DA.calc_storage_requirements()
        DA.calc_downstream_signals()
        DA.calc_H2_financials()
        DA.calc_electrolysis_financials()
        DA.calc_battery_financials(siteinfo)

        self.P2ASU, self.P2HB = np.atleast_2d(
            self.energypkg[1:] / np.sum(self.energypkg[1:])
        ).T @ np.atleast_2d(DA.P_out)
        self.H2_storage_out = DA.H2_out

        self.H2_storage = DA

        self.main_dict.update(
            {
                "H2_storage": {
                    "capacity_kg": DA.H2_capacity_kg,
                    "max_chg_kgphr": np.max(DA.H2_chg),
                    "min_chg_kgphr": np.min(DA.H2_chg),
                    "financials": {
                        "pipe_capex": DA.pipe_capex,
                        "pipe_opex": DA.pipe_opex,
                        "lined_capex": DA.lined_capex,
                        "lined_opex": DA.lined_opex,
                        "salt_capex": DA.salt_capex,
                        "salt_opex": DA.salt_opex,
                    },
                },
                "Battery_storage": {
                    "capacity_kWh": DA.P_capacity,
                    "max_chg_kW": np.max(DA.P_chg),
                    "min_chg_kWr": np.min(DA.P_chg),
                    "financials": {
                        "battery_capex": DA.battery_capex,
                        "battery_opex": np.sum(DA.battery_opex),
                    },
                },
                "Electrolyzer": {
                    "HOPP_EL": {
                        "EL_capex": DA.EL_capex,
                        "EL_opex": DA.EL_opex,
                        "max_production": DA.H2_results[
                            "max_hydrogen_production [kg/hr]"
                        ],
                        "H2_annual_output": DA.H2_results["hydrogen_annual_output"],
                        "CF": DA.H2_results["cap_factor"],
                    },
                },
            }
        )

    def calc_chemicals(self):
        powers = np.zeros([len(self.P2EL), 3])
        chemicals = np.zeros([len(self.P2EL), 3])
        ASU_rejects = np.zeros([len(self.P2EL)])
        HB_rejects = np.zeros([len(self.P2EL), 3])

        for i in range(len(self.P2EL)):
            # H2, EL_reject = EL.step(P2EL[i])
            N2, ASU_rejects[i] = self.ASU.step(self.P2ASU[i])
            NH3, HB_rejects[i, :] = self.HB.step(
                self.H2_storage_out[i], N2, self.P2HB[i]
            )

            powers[i, :] = [self.P2EL[i], self.P2ASU[i], self.P2HB[i]]
            chemicals[i, :] = [self.H2_storage_out[i], N2, NH3]

        if np.sum(ASU_rejects) > (1e-2 * np.sum(self.P2ASU)):
            print(
                f"ASU rejected {np.sum(ASU_rejects) / np.sum(self.P2EL) * 100:.2f} % of power"
            )

        if (
            np.sum(HB_rejects, axis=0)
            > 1e-2
            * np.sum(
                np.stack([self.H2_storage_out, chemicals[:, 1], self.P2HB], axis=1),
                axis=0,
            )
        ).any():
            print(f"HB rejected some of its intputs")

        H2_tot, N2_tot, NH3_tot = np.sum(chemicals, axis=0)
        H2_max, N2_max, NH3_max = np.max(chemicals, axis=0)
        P_EL_max, P_ASU_max, P_HB_max = np.max(powers, axis=0)

        self.EL.calc_financials(P_EL_max, H2_max)
        self.ASU.calc_financials(P_ASU_max, N2_max)
        self.HB.calc_financials(P_HB_max, NH3_max)
        self.totals = [H2_tot, N2_tot, NH3_tot]
        self.chemicals = chemicals
        self.powers = powers

        self.main_dict.update(
            {
                "DGA": {
                    "EL": {
                        "H2_tot": H2_tot,
                        "H2_max": H2_max,
                        "P_EL_max": P_EL_max,
                        "rating_elec": self.EL.rating_elec,
                        "rating_H2": self.EL.rating_h2,
                        "capex_rated_power": self.EL.capex_rated_power,
                        "opex_rated_power": self.EL.opex_rated_power,  # per year
                        "capex_kgpday": self.EL.capex_kgpday,
                        "opex_kgpday": self.EL.opex_kgpday,  # per year
                    },
                    "ASU": {
                        "N2_tot": N2_tot,
                        "N2_max": N2_max,
                        "P_ASU_max": P_ASU_max,
                        "rating_elec": self.ASU.rating_elec,
                        "rating_NH3": self.ASU.rating_N2,
                        "capex": self.ASU.capex,
                        "opex": self.ASU.opex,  # per year
                    },
                    "HB": {
                        "NH3_tot": NH3_tot,
                        "NH3_max": NH3_max,
                        "P_HB_max": P_HB_max,
                        "rating_elec": self.HB.rating_elec,
                        "rating_NH3": self.HB.rating_NH3,
                        "capex": self.HB.capex,
                        "opex": self.HB.opex,  # per year
                    },
                }
            }
        )

    def calc_LCOA(self):
        self.build_LCOA_dict()
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
                "ramp_lim": self.rl,
                "plant_min": self.td,
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
        if not (plant_min is None):
            self.td = plant_min
        if not (ramp_lim is None):
            self.rl = ramp_lim

        self.re_init()
        self.write_hopp_main_dict()

        # 2. inputs and outputs paths
        # 3. run HOPP
        # 4. collect generation data
        # 5. split power
        self.P2EL, self.P2ASU, self.P2HB = self.split_power()
        self.P_gen = self.P2ASU + self.P2HB

        # 6 calculate hydrogen generation
        self.H2_gen = self.calc_H2_gen()

        # 7. calculate hydrogen storage
        # 8. calculate electricity storage

        (
            self.H2_demand,
            self.H2_storage_state,
            self.H2_state_initial,
            self.H2_capacity,
        ) = self.calc_demand_profile()
        (
            self.P_demand,
            self.P_storage_state,
            self.P_capacity,
        ) = self.DO.calc_proportional_demand(self.H2_gen, self.P_gen)

        self.calc_DL_storage(self.hi.system.site)

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
        self.calc_LCOA()
        # self.build_main_dict()
        # should be about 771 USD/t NH3 (Cesaro 2021)

        # self.main_df = self.build_outputs()

        # self.main_df = pd.DataFrame(self.main_dict, index=[0])

        # multikeys, values =  build_multiindex(self.main_dict)

        self.main_df = build_multiindex_df(self.main_dict)

        []

    def run_sweep(self, cases):
        pass


def FlexibilityParameters(analysis="simple", n_ramps=1, n_tds=1):
    """Generate the ramp limits and turndown ratios for an analysis sweep

    Args:
    analysis: either "simple" or "full sweep" which type of analysis the user is performing
    n_ramps: number of ramp limits to return
    n_tds: number of turndown ratios to return

    Returns:
    ramp_lims: array of ramp_limits to simulate
    TD_ratios: array of turndown ratios to simulate
    """

    if analysis == "testing":
        ramp_lims = [0.5]
        turndowns = [0.5]

    elif analysis == "simple":
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


def max_depth(d):
    if isinstance(d, dict):
        return 1 + max((max_depth(value) for value in d.values()), default=0)
    return 0


def build_multiindex(d):
    if isinstance(d, dict):
        ind_names = []
        ind_values = []
        for key in d.keys():
            index_names, index_values = build_multiindex(d[key])
            if index_names is None:
                ind_names.append((key,))
                ind_values.append(d[key])
            else:
                for i in range(len(index_names)):
                    index_names[i] = (key,) + index_names[i]

                ind_names.extend(index_names)
                ind_values.extend(index_values)

    else:
        return None, [d]

    return ind_names, ind_values


def build_multiindex_df(d):
    multikeys, values = build_multiindex(d)

    mi = pd.MultiIndex.from_tuples(multikeys)
    df = pd.DataFrame(data=[values], columns=mi)

    return df
