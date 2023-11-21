# %%
import numpy as np
import matplotlib.pyplot as plt

from hopp.simulation import HoppInterface
from hopp.tools.dispatch.plot_tools import (
    plot_battery_output,
    plot_battery_dispatch_error,
    plot_generation_profile,
)
from hopp.simulation.technologies.hydrogen.electrolysis import run_h2_PEM

# %%

hi = HoppInterface("./inputs/hopp_input.yaml")

# %%
plant_life = 30

hi.simulate(plant_life)


# site info
hi.system.site

# Generation profile
wind = np.array(hi.system.wind.generation_profile[0:8760])
solar = np.array(hi.system.pv.generation_profile[0:8760])
battery = np.array(hi.system.battery.generation_profile[0:8760])

fig, ax = plt.subplots(3, 1, sharex="col")

ax[0].plot(wind)
ax[1].plot(solar)
ax[2].plot(battery)


# %%


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

hybrid_plant = hi.system
solar_plant_power = np.array(hybrid_plant.pv.generation_profile[0:simulation_length])
wind_plant_power = np.array(hybrid_plant.wind.generation_profile[0:simulation_length])
hybrid_plant_generation_profile = solar_plant_power + wind_plant_power

power_to_H2 = 0.93144 * hybrid_plant_generation_profile
power_to_industry = 0.06855 * hybrid_plant_generation_profile

# (
#     h2_results,
#     H2_Timeseries,
#     H2_Summary,
#     energy_input_to_electrolyzer,
# ) = run_h2_PEM.run_h2_PEM(
#     power_to_H2,
#     electrolyzer_size_mw,
#     plant_life,
#     number_electrolyzer_stacks,
#     [],
#     pem_control_type,
#     100,
#     user_defined_pem_param_dictionary,
#     use_degradation_penalty,
#     grid_connection_scenario,
#     [],
# )

run_h2_inputs = (    power_to_H2,
    electrolyzer_size_mw,
    plant_life,
    number_electrolyzer_stacks,
    [],
    pem_control_type,
    100,
    user_defined_pem_param_dictionary,
    use_degradation_penalty,
    grid_connection_scenario,
    [],)

(
    h2_results,
    H2_Timeseries,
    H2_Summary,
    energy_input_to_electrolyzer,
) = run_h2_PEM.run_h2_PEM(*run_h2_inputs)   

# Total hydrogen output timeseries (kg-H2/hour)
hydrogen_production_kg_pr_hr = H2_Timeseries["hydrogen_hourly_production"]
# Rated/maximum hydrogen production from electrolysis system
max_h2_pr_h2 = h2_results["new_H2_Results"]["Rated BOL: H2 Production [kg/hr]"]
# x-values as hours of year
hours_of_year = np.arange(0, len(hydrogen_production_kg_pr_hr), 1)

H2_gen = H2_Timeseries["hydrogen_hourly_production"]

# %%

H2_avg = np.mean(H2_gen)
H2_diff = H2_gen - H2_avg
H2_storage = np.cumsum(H2_diff)

fig, ax = plt.subplots(3, 1, sharex="col")

ax[0].plot(H2_gen)
ax[0].plot([0, 8760], [H2_avg, H2_avg])
ax[1].plot(H2_gen - H2_diff)
ax[2].plot(H2_storage)

print(np.max(H2_storage))

# %%

from hopp.simulation.technologies.hydrogen.h2_storage.pipe_storage import (
    underground_pipe_storage,
)
from hopp.simulation.technologies.hydrogen.h2_storage.salt_cavern import salt_cavern
from hopp.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern import (
    lined_rock_cavern,
)

input_dict = {
    "H2_storage_kg": np.max(H2_storage),
    "storage_duration_hrs": 10,
    "compressor_output_pressure": 100,
    "system_flow_rate": 10,
    "model": "papadias",
}

pipes = underground_pipe_storage.UndergroundPipeStorage(input_dict)
pipe_capex = pipes.pipe_storage_capex()
pipe_opex = pipes.pipe_storage_opex()
print("pipe", pipe_capex)

lined = lined_rock_cavern.LinedRockCavernStorage(input_dict)
lined_capex = lined.lined_rock_cavern_capex()
lined_opex = lined.lined_rock_cavern_opex()
print("lined", lined_capex)

salt = salt_cavern.SaltCavernStorage(input_dict)
salt_capex = salt.salt_cavern_capex()
salt_opex = salt.salt_cavern_opex()
print("salt", salt_capex)

# %%

wind_installed_cost = hi.system.wind.total_installed_cost
solar_installed_cost = hi.system.pv.total_installed_cost
battery_installed_cost = hi.system.battery.total_installed_cost

# %%

fig, ax = plt.subplots(4, 1, sharex="col")

ax[0].plot(wind)
ax[1].plot(solar)
# ax[2].plot(battery)
ax[3].plot(H2_Timeseries["hydrogen_hourly_production"])

# %%
plot_battery_dispatch_error(hybrid_plant)
# %%
plot_battery_output(hybrid_plant)
# %%
plot_generation_profile(hybrid_plant)
