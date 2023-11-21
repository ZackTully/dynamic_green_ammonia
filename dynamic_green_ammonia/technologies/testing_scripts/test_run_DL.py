import numpy as np
import pandas as pd
import time
from multiprocess import Pool  # type: ignore
import sys
from pathlib import Path
import pprint


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass

from Dynamic_Load.technologies.Run_DL import RunDL


hopp_input = "inputs/hopp_input.yaml"


DL_p05 = RunDL(hopp_input, ammonia_ramp_limit=0.25, ammonia_plant_turndown_ratio=0.05)
DL_p25 = RunDL(hopp_input, ammonia_ramp_limit=0.25, ammonia_plant_turndown_ratio=0.25)

DL_p05.run()
DL_p25.run()

pprint.pprint(DL_p05.main_df["storage_capacity_kg"])
pprint.pprint(DL_p25.main_df["storage_capacity_kg"])
[]
