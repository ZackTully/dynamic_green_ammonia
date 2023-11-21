import numpy as np
import pandas as pd
import time

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass

from Dynamic_Load.technologies.Run_DL import RunDL

t0 = time.time()

hopp_inputs = ["inputs/hopp_input_IN.yaml", "inputs/hopp_input_TX.yaml"]

for hopp_input in hopp_inputs:
    DL = RunDL(
        hopp_input,
        ammonia_ramp_limit=0.1,
        ammonia_plant_turndown_ratio=0.1,
        dynamic_load=True,
    )

    n_ramps = 5
    n_mins = 15

    if n_ramps > 1:
        # ramp_lims = np.linspace(0, 1, n_ramps)
        ramp_lims = np.exp(np.linspace(np.log(0.01), np.log(1), n_ramps))
    else:
        ramp_lims = [0.1]

    if n_mins > 1:
        # plant_mins = np.linspace(0.01, 0.99, n_mins)
        plant_mins = 1 - np.logspace(np.log10(0.99), -2, n_mins)
    else:
        plant_mins = [0.25]

    dfs = []

    t_prev = t0
    for i, rl in enumerate(ramp_lims):
        for j, pm in enumerate(plant_mins):
            DL.run(ramp_lim=rl, plant_min=pm)
            print(f"Completed ramp_lim: {rl:.3f}, plant_min: {pm:.3f}")
            print(f"Took {time.time() - t_prev:.2f} seconds")
            t_prev = time.time()
            dfs.append(DL.main_df.copy())

    main_df = pd.concat(dfs)
    main_df.to_csv(
        f"Dynamic_Load/data/DL_runs/main_df_{hopp_input.split('.')[0][-2:]}.csv"
    )

    print(f"Sweep took {time.time() - t0:.2f} seconds")
