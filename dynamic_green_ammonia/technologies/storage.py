"""
Storage models for dynamic ammonia process and dynamic steel process

"""


import numpy as np
from scipy.optimize import linprog

from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage import (
    underground_pipe_storage,
)
from hopp.simulation.technologies.hydrogen.h2_storage.salt_cavern import salt_cavern
from hopp.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern import (
    lined_rock_cavern,
)
from hopp.simulation.technologies.battery import Battery, BatteryConfig
from hopp.simulation.technologies.hydrogen.electrolysis import (
    run_h2_PEM,
    PEM_costs_Singlitico_model,
)


class BaseStorage:
    """
    Base class for hydrogen storage models containing shared sizing and financial calculations
    """

    P_diff: np.ndarray
    P_chg: np.ndarray
    P_soc_max: float
    P_chg_max: np.floating
    P_dcg_max: np.floating

    H2_chg: np.ndarray
    H2_gen: np.ndarray
    H2_soc_max: float

    pipe_capex: float
    pipe_opex: float
    salt_capex: float
    salt_opex: float
    lined_capex: float
    lined_opex: float

    def __init__(self, H2_gen, power_to_el=0, power_to_industry=0):
        self.H2_gen = H2_gen
        self.P_gen = power_to_industry
        self.P_EL = power_to_el

    def calc_H2_demand(self):
        self.H2_demand = np.zeros(8760)
        self.P_demand = np.zeros(8760)

    def calc_storage_requirements(self):
        """Calculate the H2 and power storage capacity to reconcile gen and demand"""

        self.H2_chg = self.H2_demand - self.H2_gen
        self.H2_soc = np.cumsum(self.H2_chg)
        self.H2_capacity_kg = np.max(self.H2_soc) - np.min(self.H2_soc)
        self.H2_chg_capacity = np.max(self.H2_chg) - np.min(self.H2_chg)

        self.P_chg = self.P_demand - self.P_gen
        self.P_soc = np.cumsum(self.P_chg)
        self.P_capacity = np.max(self.P_soc) - np.min(self.P_soc)
        self.P_chg_capacity = np.max(self.P_chg) - np.min(self.P_chg)

    def calc_downstream_signals(self):
        """Calculate hydrogen and power timeseries coming out of storage"""
        self.H2_out = self.H2_gen + self.H2_chg
        self.P_out = self.P_gen + self.P_chg

    def calc_H2_financials(self):
        """Calculate buried pipe, lined cavern, and salt cavern capex and opex"""

        if (
            np.max(np.abs(self.H2_chg)) == 0
        ):  # If the industry does not need storage then costs will be 0
            self.pipe_capex = 0
            self.pipe_opex = 0
            self.salt_capex = 0
            self.salt_opex = 0
            self.lined_capex = 0
            self.lined_opex = 0
        else:
            self.storage_dict = {
                "H2_storage_kg": self.H2_capacity_kg,
                # "storage_duration_hrs": 10,
                "compressor_output_pressure": 100,
                "system_flow_rate": self.H2_chg_capacity,
                "model": "papadias",
            }

            self.pipe_storage = underground_pipe_storage.UndergroundPipeStorage(
                self.storage_dict
            )
            self.pipe_capex = self.pipe_storage.pipe_storage_capex()[1]
            self.pipe_opex = self.pipe_storage.pipe_storage_opex()

            self.salt_cavern = salt_cavern.SaltCavernStorage(self.storage_dict)
            self.salt_capex = self.salt_cavern.salt_cavern_capex()[1]
            self.salt_opex = self.salt_cavern.salt_cavern_opex()

            self.lined_cavern = lined_rock_cavern.LinedRockCavernStorage(
                self.storage_dict
            )
            self.lined_capex = self.lined_cavern.lined_rock_cavern_capex()[1]
            self.lined_opex = self.lined_cavern.lined_rock_cavern_capex()[1]

    def calc_battery_financials(self, siteinfo):
        """Instantiate a HOPP battery object and calculate its financials

        arguments:
        siteinfo: HOPP siteinfo object
        """
        config = BatteryConfig(
            tracking=True,
            system_capacity_kwh=self.P_capacity,
            system_capacity_kw=self.P_chg_capacity,
            chemistry="LFPGraphite",
        )

        self.battery = Battery(siteinfo, config)
        self.battery.simulate_financials(1e6, 25)
        self.battery_capex = self.battery.cost_installed
        self.battery_opex = self.battery.om_total_expense

    def calc_electrolysis_financials(self):
        """Instantiate a HOPP electrolyzer model and calculate its financials"""
        electrolyzer_size_mw = 100
        simulation_length = 8760  # 1 year
        use_degradation_penalty = True
        number_electrolyzer_stacks = 2
        grid_connection_scenario = "off-grid"
        EOL_eff_drop = 10
        pem_control_type = "basic"
        user_defined_pem_param_dictionary = {
            "Modify BOL Eff": False,
            "BOL Eff [kWh/kg-H2]": [],
            "Modify EOL Degradation Value": True,
            "EOL Rated Efficiency Drop": EOL_eff_drop,
        }

        run_h2_inputs = (
            self.P_EL,
            electrolyzer_size_mw,
            25,
            number_electrolyzer_stacks,
            [],
            pem_control_type,
            100,
            user_defined_pem_param_dictionary,
            use_degradation_penalty,
            grid_connection_scenario,
            [],
        )

        (
            h2_results,
            h2_timeseries,
            h2_summary,
            energy_input_to_electrolyzer,
        ) = run_h2_PEM.run_h2_PEM(*run_h2_inputs)
        PEM_cost = PEM_costs_Singlitico_model.PEMCostsSingliticoModel(0)
        PEM_capex, PEM_opex = PEM_cost.run(electrolyzer_size_mw * 1e-3, 600)
        self.EL_capex = PEM_capex * 1e6
        self.EL_opex = PEM_opex * 1e6

    def run(self, siteinfo):
        self.calc_H2_demand()
        self.calc_storage_requirements()
        self.calc_downstream_signals()
        self.calc_H2_financials()
        self.calc_electrolysis_financials()
        self.calc_battery_financials(siteinfo)


class SteadyStorage(BaseStorage):
    """Hydrogen storage for a steady industrial requirement"""

    def calc_H2_demand(self):
        """Calculate steady H2 demand profile and required storage"""

        self.P_ratio = np.sum(self.P_gen) / np.sum(self.H2_gen)

        self.H2_demand = np.mean(self.H2_gen)
        self.P_demand = self.P_ratio * self.H2_demand


class DynamicAmmoniaStorage(BaseStorage):
    """Hydrogen storage for a ramp-rate limited dynamic industrial requirement"""

    def __init__(
        self,
        H2_gen=0,
        power_to_el=0,
        power_to_industry=0,
        ramp_lim=0.1,
        plant_rating=None,
        plant_min=0.1,
        optimize=True,
    ):
        BaseStorage.__init__(
            self,
            H2_gen=H2_gen,
            power_to_el=power_to_el,
            power_to_industry=power_to_industry,
        )

        based_on_mean_generation = True  # base the plant on mean generation

        self.ramp_lim = ramp_lim
        self.optimize = optimize
        self.plant_min = plant_min

        if based_on_mean_generation:
            if plant_rating:
                self.plant_rating = plant_rating
            else:
                self.plant_rating = np.mean(self.H2_gen) * (
                    (1 - self.plant_min) / 2 + 1
                )
                # self.plant_rating = np.max(self.H2_gen)

            self.min_demand = self.plant_rating * self.plant_min
        else:  # best plant rating
            hist, bins = np.histogram(self.H2_gen, bins=100)

            capture = np.zeros(len(bins))

            for i, mean in enumerate(bins):
                max_demand = (2 / (self.plant_min + 1)) * mean
                min_demand = self.plant_min * max_demand

                min_idx = np.argmin(np.abs(bins - min_demand))
                max_idx = np.argmin(np.abs(bins - max_demand))

                if min_idx != max_idx:
                    capture[i] = np.sum(hist[min_idx:max_idx]) / (
                        np.sum(hist[0:min_idx]) + np.sum(hist[max_idx:])
                    )

            mean = bins[np.argmax(capture)]
            max_demand = (2 / (self.plant_min + 1)) * mean
            min_demand = self.plant_min * max_demand

            self.plant_rating = max_demand
            self.min_demand = min_demand

    def calc_H2_demand(self):
        """Calculate ramp rate H2 demand profile and required storage"""

        if self.optimize:
            self.calc_H2_demand_opt()
        else:
            self.P_ratio = np.sum(self.P_gen) / np.sum(
                self.H2_gen
            )  # ratio of power to hydrogen

            ramp_lim = self.ramp_lim * self.plant_rating
            H2_demand = np.zeros(len(self.H2_gen))
            P_demand = np.zeros(len(self.H2_gen))
            H2_SOC_loop = 0

            for i in range(len(self.H2_gen)):
                if i == 0:
                    H2_demand[i] = np.mean(self.H2_gen)
                    continue

                # constrain hydrogen demand within the ramp limits
                H2_delta = self.H2_gen[i] + H2_SOC_loop - H2_demand[i - 1]
                if np.abs(H2_delta) > ramp_lim:  # exceeding ramp limit
                    H2_dem = H2_demand[i - 1] + np.sign(H2_delta) * ramp_lim
                else:
                    extra_demand = ramp_lim - np.abs(H2_delta)
                    storage_correction = np.min([extra_demand, np.abs(H2_SOC_loop)])

                    storage_FB = np.sign(H2_SOC_loop) * storage_correction

                    H2_dem = self.H2_gen[i] + storage_FB

                # upper constraints on H2 demand
                H2_dem = np.min([self.plant_rating, H2_dem])
                # lower constraints on H2 demand
                H2_dem = np.max([self.min_demand, H2_dem])
                P_dem = self.P_ratio * H2_dem
                H2_demand[i] = H2_dem
                P_demand[i] = P_dem
                H2_SOC_loop += self.H2_gen[i] - H2_demand[i]

            self.P_demand = P_demand
            self.H2_demand = H2_demand
            self.calc_storage_requirements()

    def calc_H2_demand_opt(self):
        self.P_ratio = np.sum(self.P_gen) / np.sum(
            self.H2_gen
        )  # ratio of power to hydrogen

        ramp_lim = self.ramp_lim * self.plant_rating

        demand_opt = DemandOptimization(
            self.H2_gen, ramp_lim, self.min_demand, self.plant_rating
        )
        res = demand_opt.optimize()
        n_steps = len(self.H2_gen)
        self.H2_demand = res.x[0:n_steps]
        storage_size = res.x[-2] - res.x[-1]
        self.P_demand = self.P_ratio * self.H2_demand
        self.calc_storage_requirements()


class DemandOptimization:
    def __init__(self, H2_gen, ramp_lim, min_demand, max_demand):
        self.H2_gen = H2_gen
        self.ramp_lim = ramp_lim
        self.min_demand = min_demand
        self.max_demand = max_demand

    def optimize(self):
        n_steps = len(self.H2_gen)

        c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])
        # c = np.concatenate([np.zeros(n_steps), [-1], np.zeros(n_steps - 2), [1, 1, -1]])

        A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
        b_ub = np.zeros([n_steps * 4])
        A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
        b_eq = np.zeros(n_steps + 1)

        A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]
        for i in range(n_steps):
            A_ub[i, [i + n_steps, -2]] = [1, -1]
            A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

            if i > 0:
                A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
                A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
            b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [self.ramp_lim, self.ramp_lim]

            b_eq[i] = self.H2_gen[i]
            if i == 0:
                A_eq[0, [0, n_steps]] = [1, 1]
                continue
            A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

        bound_low = [self.min_demand] * n_steps + [None] * (n_steps + 2)
        bound_up = [self.max_demand] * n_steps + [None] * (n_steps + 2)

        bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
        return res
