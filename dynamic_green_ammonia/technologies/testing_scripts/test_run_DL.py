import numpy as np
import pandas as pd
import time
from multiprocess import Pool  # type: ignore
import sys
from pathlib import Path
import pprint


from dynamic_green_ammonia.technologies.Run_DL import RunDL


hopp_input = "inputs/hopp_input_IN.yaml"

# Completed 7 out of 64. Ramp rate: 0.000, turndown: 0.857
DL_1 = RunDL(hopp_input, ammonia_ramp_limit=0, ammonia_plant_turndown_ratio=0.857)
DL_1.run()


# DL_p05 = RunDL(hopp_input, ammonia_ramp_limit=0.25, ammonia_plant_turndown_ratio=0.05)
# DL_p25 = RunDL(hopp_input, ammonia_ramp_limit=0.25, ammonia_plant_turndown_ratio=0.25)

# DL_p05.run()
# DL_p25.run()

# pprint.pprint(DL_p05.main_df["storage_capacity_kg"])
# pprint.pprint(DL_p25.main_df["storage_capacity_kg"])
pprint.pprint(DL_1.main_dict)

[]
