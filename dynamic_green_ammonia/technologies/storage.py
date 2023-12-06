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

from dynamic_green_ammonia.technologies.demand import DemandOptimization


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

        based_on_mean_generation = False  # base the plant on mean generation

        self.ramp_lim = ramp_lim
        self.optimize = optimize
        self.plant_min = plant_min

        if based_on_mean_generation:
            print("Using mean generation to size the plant")
            if plant_rating:
                self.plant_rating = plant_rating
            else:
                self.plant_rating = np.mean(self.H2_gen) * (
                    (1 - self.plant_min) / 2 + 1
                )
                # self.plant_rating = np.max(self.H2_gen)

            self.min_demand = self.plant_rating * self.plant_min
        else:  # best plant rating
            # hist, bins = np.histogram(self.H2_gen, bins=100)

            # capture = np.zeros(len(bins))

            # for i, mean in enumerate(bins):
            #     max_demand = (2 / (self.plant_min + 1)) * mean
            #     min_demand = self.plant_min * max_demand

            #     min_idx = np.argmin(np.abs(bins - min_demand))
            #     max_idx = np.argmin(np.abs(bins - max_demand))

            #     if min_idx != max_idx:
            #         capture[i] = np.sum(hist[min_idx:max_idx]) / (
            #             np.sum(hist[0:min_idx]) + np.sum(hist[max_idx:])
            #         )

            # mean = bins[np.argmax(capture)]
            # max_demand = (2 / (self.plant_min + 1)) * mean
            # min_demand = self.plant_min * max_demand

            center = (
                self.plant_min * np.mean(self.H2_gen)
                + (1 - self.plant_min) * np.max(self.H2_gen) / 2
            )
            center = np.interp(
                self.plant_min, [0, 1], [np.max(self.H2_gen) / 2, np.mean(self.H2_gen)]
            )
            max_demand = center * (2 / (1 + self.plant_min))
            min_demand = self.plant_min * max_demand

            self.plant_rating = max_demand
            self.min_demand = min_demand

    def calc_H2_demand(self):
        """Calculate ramp rate H2 demand profile and required storage"""

        # if (self.ramp_lim == 0) or (self.plant_min == 1):
        #     BaseStorage.calc_H2_demand(self)
        if False:
            pass
        else:
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
        x, success = demand_opt.optimize()
        n_steps = len(self.H2_gen)
        self.H2_demand = x[0:n_steps]
        storage_size = x[-2] - x[-1]
        self.P_demand = self.P_ratio * self.H2_demand
        self.calc_storage_requirements()


# class DemandOptimization:
#     def __init__(self, H2_gen, ramp_lim, min_demand, max_demand):
#         self.optimization_version = 2
#         self.H2_gen = H2_gen
#         self.ramp_lim = ramp_lim
#         self.min_demand = min_demand
#         self.max_demand = max_demand

#     def optimize(self):
#         if self.min_demand >= self.max_demand:
#             # if False:
#             x, success = self.static_demand()
#         else:
#             if self.optimization_version == 1:
#                 x, success, res = self.optimize_v1()
#             elif self.optimization_version == 2:
#                 x, success, res = self.optimize_v2()

#         return x, success

#     def static_demand(self):
#         demand = self.max_demand * np.ones(len(self.H2_gen))
#         storage_state = np.cumsum(self.H2_gen - demand)
#         storage_state -= np.min(storage_state)
#         state_max = np.max(storage_state)
#         state_min = np.min(storage_state)
#         x = np.concatenate([demand, storage_state, [state_max, state_min]])
#         success = True
#         return x, success

#     def optimize_v1(self):
#         n_steps = len(self.H2_gen)

#         c = np.concatenate([np.zeros(2 * n_steps), [1, -1]])

#         A_ub = np.zeros([n_steps * 4, 2 * n_steps + 2])
#         b_ub = np.zeros([n_steps * 4])
#         A_eq = np.zeros([n_steps + 1, 2 * n_steps + 2])
#         b_eq = np.zeros(n_steps + 1)

#         A_eq[-1, [n_steps, 2 * n_steps - 1]] = [1, -1]
#         for i in range(n_steps):
#             A_ub[i, [i + n_steps, -2]] = [1, -1]
#             A_ub[i + n_steps, [i + n_steps, -1]] = [-1, 1]

#             if i > 0:
#                 A_ub[i + 2 * n_steps, [i, i - 1]] = [1, -1]
#                 A_ub[i + 3 * n_steps, [i, i - 1]] = [-1, 1]
#             b_ub[[i + 2 * n_steps, i + 3 * n_steps]] = [
#                 self.ramp_lim,
#                 self.ramp_lim,
#             ]

#             b_eq[i] = self.H2_gen[i]
#             if i == 0:
#                 A_eq[0, [0, n_steps]] = [1, 1]
#                 continue
#             A_eq[i, [i, i + n_steps - 1, i + n_steps]] = [1, -1, 1]

#         # bound_low = [self.min_demand] * n_steps + [None] * (n_steps + 2)
#         bound_low = [self.min_demand] * n_steps + [0] * n_steps + [None] * 2

#         bound_up = [self.max_demand] * n_steps + [None] * (n_steps + 2)

#         bounds = [(bound_low[i], bound_up[i]) for i, bl in enumerate(bound_low)]

#         res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
#         x = res.x
#         success = res.success

#         return x, success, res

#     def optimize_v2(self):
#         N = len(self.H2_gen)

#         # plant paramaters
#         # td = 0.5
#         # rl = 0.01 * (1 - td)
#         # td = 0.3813675880432129
#         # rl = 0.9

#         rl = self.ramp_lim

#         # center = np.interp(td, [0, 1], [np.max(self.H2_gen) / 2, np.mean(self.H2_gen)])
#         # # center = np.mean(H2_gen)
#         # max_demand = (2 / (td + 1)) * center
#         # min_demand = td * max_demand
#         min_demand = self.min_demand
#         max_demand = self.max_demand

#         R = rl * max_demand
#         R = self.ramp_lim

#         # H2_gen = np.linspace(0, 1.5 * max_demand, N) + 3 * np.random.random(N)

#         # x = [u_0, ... , u_N, x_0, ... , x_N, x_max, x_min]
#         u_min = min_demand
#         u_max = max_demand

#         # Cost vector
#         C = np.zeros(N + N + 2)
#         # C[N : N + N] = 1 * np.ones(N)
#         C[2 * N + 0] = 1  # highest storage state
#         C[2 * N + 1] = -1  # lowest storage state

#         # Upper and lower bounds
#         bound_l = np.concatenate(
#             [
#                 [u_min] * N,  # demand lower bound
#                 [0] * N,  # storage state lower bound
#                 [None, 0],  # storage state max, min lower bound
#             ]
#         )

#         bound_u = np.concatenate(
#             [
#                 [u_max] * N,  # demand upper bound,
#                 [None] * N,  # storage state upper bound,
#                 [None, None],  # storage state max, min upper bound
#             ]
#         )

#         # Positive demand ramp rate limit
#         Aub_ramp_pos = np.zeros([N, N + N + 2])
#         bub_ramp_pos = np.zeros(N)

#         # u[k+1] - u[k] <= R
#         # x[k+1] - x[k] <= R
#         for k in range(N):
#             if (k + 1) == N:
#                 break
#             Aub_ramp_pos[k, k + 1] = 1
#             Aub_ramp_pos[k, k] = -1
#             bub_ramp_pos[k] = R

#         # Negative demand ramp rate limit
#         Aub_ramp_neg = np.zeros([N, N + N + 2])
#         bub_ramp_neg = np.zeros(N)

#         # -u[k+1] + u[k] <= R
#         # -x[k+1] + x[k] <= R
#         for k in range(N):
#             if (k + 1) == N:
#                 break
#             Aub_ramp_neg[k, k + 1] = -1
#             Aub_ramp_neg[k, k] = 1
#             bub_ramp_neg[k] = R

#         # x_max
#         Aub_xmax = np.zeros([N, N + N + 2])
#         bub_xmax = np.zeros(N)

#         # state[k] - state_max <= 0
#         # x[N+k] - x[N+N] <= 0
#         for k in range(N):
#             Aub_xmax[k, N + k] = 1
#             Aub_xmax[k, N + N] = -1
#             bub_xmax[k] = 0

#         # x_min
#         Aub_xmin = np.zeros([N, N + N + 2])
#         bub_xmin = np.zeros(N)

#         # -state[k] + state_min <= 0
#         # -x[N+k] + x[N+N+1] <= 0
#         for k in range(N):
#             Aub_xmin[k, N + k] = -1
#             Aub_xmin[k, N + N + 1] = 1
#             bub_xmin[k] = 0

#         # Storage "dynamics"
#         Aeq_dyn = np.zeros([N, N + N + 2])
#         beq_dyn = np.zeros(N)

#         # state[k+1] - state[k] + demand[k] = H2_gen[k]
#         # x[N+k+1] - x[N+k] + x[k] = beq_dyn[k]
#         for k in range(N):
#             if (k + 1) == N:
#                 break
#             Aeq_dyn[k, N + k + 1] = 1
#             Aeq_dyn[k, N + k] = -1
#             Aeq_dyn[k, k] = 1

#             beq_dyn[k] = self.H2_gen[k]

#         # state[0] = state[N]
#         # -x[N+0] + x[N + N - 1] = 0
#         Aeq_dyn[N - 1, N] = -1
#         Aeq_dyn[N - 1, 2 * N - 1] = 1

#         A_ub = np.concatenate([Aub_ramp_pos, Aub_ramp_neg, Aub_xmax, Aub_xmin])
#         b_ub = np.concatenate([bub_ramp_pos, bub_ramp_neg, bub_xmax, bub_xmin])

#         A_eq = Aeq_dyn
#         b_eq = beq_dyn

#         bounds = [(bound_l[i], bound_u[i]) for i, bl in enumerate(bound_l)]

#         res = linprog(
#             c=C,
#             A_ub=A_ub,
#             b_ub=b_ub,
#             A_eq=A_eq,
#             b_eq=b_eq,
#             bounds=bounds,
#         )

#         return res.x, res.success, res
