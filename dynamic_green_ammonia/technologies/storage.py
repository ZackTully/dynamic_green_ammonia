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


class DynamicAmmoniaStorage:
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

    def __init__(self, H2_gen, H2_demand, P_demand, power_to_el=0, power_to_industry=0):
        self.H2_gen = H2_gen
        self.H2_demand = H2_demand
        self.P_demand = P_demand
        self.P_gen = power_to_industry
        self.P_EL = power_to_el

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
        number_electrolyzer_stacks = 10
        electrolyzer_size_mw = np.max(self.P_EL) / 1e3
        simulation_length = 8760  # 1 year
        use_degradation_penalty = True

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
            self.P_EL,  # generation timeseries
            electrolyzer_size_mw,  #
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
            self.H2_results,
            self.H2_timeseries,
            self.H2_summary,
            self.energy_input_to_electrolyzer,
        ) = run_h2_PEM.run_h2_PEM(*run_h2_inputs)
        PEM_cost = PEM_costs_Singlitico_model.PEMCostsSingliticoModel(0)
        PEM_capex, PEM_opex = PEM_cost.run(electrolyzer_size_mw * 1e-3, 600)
        self.EL_capex = PEM_capex * 1e6
        self.EL_opex = PEM_opex * 1e6
