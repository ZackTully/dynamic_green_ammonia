import numpy as np
import pandas as pd
import time

import sys
from pathlib import Path

from dynamic_green_ammonia.technologies.Run_DL import RunDL, FlexibilityParameters

t0 = time.time()

hopp_inputs = ["inputs/hopp_input_IN.yaml", "inputs/hopp_input_TX.yaml"]
count = 1

for hopp_input in hopp_inputs:
    DL = RunDL(
        hopp_input,
        ammonia_ramp_limit=0.1,
        ammonia_plant_turndown_ratio=0.1,
    )

    # analysis_type = "testing"
    analysis_type = "full_sweep"
    # analysis_type = "simple"

    ramp_lims, turndowns = FlexibilityParameters(
        analysis=analysis_type, n_ramps=8, n_tds=8
    )
    # ramp_lims, turndowns = FlexibilityParameters(
    #     analysis=analysis_type, n_ramps=3, n_tds=10
    # )

    dfs = []

    n_runs = len(ramp_lims) * len(turndowns) * len(hopp_inputs)

    t_prev = t0
    for i, rl in enumerate(ramp_lims):
        for j, pm in enumerate(turndowns):
            DL.run(ramp_lim=rl, plant_min=pm)
            print(
                f"Completed {count} out of {n_runs}. Ramp rate: {rl:.3f}, turndown: {pm:.3f}"
            )
            print(
                f"Took {time.time() - t_prev:.2f} seconds, maybe {( (time.time() - t0) * (n_runs / count) - (time.time() - t0)):.2f} more seconds"
            )

            t_prev = time.time()
            dfs.append(DL.main_df.copy())
            count += 1

    # import matplotlib.pyplot as plt
    # for i, zeros in enumerate(DL.storage_state_zeros):
    #     plt.plot(zeros+i)

    main_df = pd.concat(dfs)
    main_df.to_csv(
        f"dynamic_green_ammonia/data/DL_runs/{analysis_type}_main_df_{hopp_input.split('.')[0][-2:]}.csv"
    )

    print(f"{hopp_input.split('.')[0][-2:]} sweep took {time.time() - t0:.2f} seconds")
