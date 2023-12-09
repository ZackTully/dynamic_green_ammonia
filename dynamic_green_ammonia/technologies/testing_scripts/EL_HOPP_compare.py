# %% Imports

import numpy as np
import matplotlib.pyplot as plt

from dynamic_green_ammonia.technologies.Run_DL import RunDL

# %%
hopp_input = "inputs/hopp_input_IN.yaml"

td = 0.1
rl = 0.1
DL = RunDL(hopp_input, td, rl)

DL.run()

# %%

time = np.arange(0, 8760, 1)

DL.H2_storage.H2_results
DL.H2_storage.H2_summary
DL.H2_storage.H2_timeseries


fig, ax = plt.subplots(2, 1, sharex="col", figsize=(10, 10))

ax[0].plot(time, DL.H2_gen)
ax[0].plot(time, DL.H2_storage.H2_timeseries["hydrogen_hourly_production"])

ax[1].plot(time, DL.H2_storage.P_EL)
ax[1].plot(time, DL.H2_storage.energy_input_to_electrolyzer)


# %%

from hopp.simulation.technologies.hydrogen.electrolysis import (
    run_h2_PEM,
    PEM_costs_Singlitico_model,
)

P_EL = DL.H2_storage.P_EL

number_electrolyzer_stacks = 50
electrolyzer_size_mw = np.max(P_EL) / 1e3 
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
    P_EL,  # generation timeseries
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
    H2_results,
    H2_timeseries,
    H2_summary,
    energy_input_to_electrolyzer,
) = run_h2_PEM.run_h2_PEM(*run_h2_inputs)

PEM_cost = PEM_costs_Singlitico_model.PEMCostsSingliticoModel(0)
PEM_capex, PEM_opex = PEM_cost.run(electrolyzer_size_mw * 1e-3, 600)
plt.plot(H2_timeseries["hydrogen_hourly_production"])

# %%

plt.show()
