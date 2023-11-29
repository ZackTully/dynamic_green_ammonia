import numpy as np
import pandas as pd
from dynamic_green_ammonia.technologies.Run_DL import RunDL

hopp_inputs = ["inputs/hopp_input_IN.yaml", "inputs/hopp_input_TX.yaml"]

ramp_limits = np.array([0.01, 0.5])
TD_ratios = np.array([0.9, 0.1])

DL = RunDL(
    hopp_input=hopp_inputs[0],
    ammonia_ramp_limit=ramp_limits[0],
    ammonia_plant_turndown_ratio=TD_ratios[0],
    dynamic_load=True,
)

# DL.run(ramp_limits[0], TD_ratios[0])
dfs = []

count = 1
for rl in ramp_limits:
    for td in TD_ratios:
        DL.run(rl, td)
        dfs.append(DL.main_df.copy())
        print(
            f"Completed {count} out of {len(ramp_limits)*len(TD_ratios)}. Ramp rate: {rl:.3f}, turndown: {td:.3f}"
        )
        count += 1

columns = [
    "storage_capacity_kg",
    "storage_flow_rate_kgphr",
    "storage_soc_f",
    # "N2_max",
    # "P_ASU_max",
    "NH3_max",
    # "P_HB_max",
    "ramp_lim",
    "plant_min",
    "LCOA_pipe",
    "LCOA_lined",
    "LCOA_salt",
    "H2_storage_rating",
    "battery_rating",
    "ASU_rating",
    "HB_rating",
    # "pipe_capex",
    # "pipe_opex",
    # "lined_capex",
    # "lined_opex",
    # "salt_capex",
    # "salt_opex",
    # "battery_capex",
    # "battery_opex",
    # "ASU_capex",
    # "ASU_opex",
    # "HB_capex",
    # "HB_opex",
    "LT_NH3",
]
main_df = pd.concat(dfs)
[]
