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

# %% Compare costs


print(f"HOPP EL capex: {PEM_capex:.2f}, HOPP EL opex: {PEM_opex:.2f}")
print(f'kgpday EL capex: {DL.main_dict["DGA"]["EL"]["capex_kgpday"]:.2f}, kgpday EL opex: {DL.main_dict["DGA"]["EL"]["opex_kgpday"]:.2f}')
print(f'kgpday EL capex: {DL.main_dict["DGA"]["EL"]["capex_rated_power"]:.2f}, kgpday EL opex: {DL.main_dict["DGA"]["EL"]["opex_rated_power"]:.2f}')


#%%

capex_p = lambda p_rated: 880 * p_rated ** 1
capex_h2 = lambda h2_rated: 1 * h2_rated ** 0.65
   
p = np.linspace(0, 1e6, 100) # 0 kW to 1 gW rated power
h2 = p / 53.5 * 24 # kW / (kWh / kg)


plt.plot(p, capex_p(p))
plt.plot(p, capex_h2(h2) )


# %%
