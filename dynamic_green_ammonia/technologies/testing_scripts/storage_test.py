from pathlib import Path
import sys


file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import numpy as np
import matplotlib.pyplot as plt
from Dynamic_Load.technologies.storage import DynamicAmmoniaStorage

rootdir = Path(__file__).parents[2]

gen = np.load(rootdir / "data" / "hybrid_gen.npy")

P2EL = gen
H2_gen = gen / 55


DA = DynamicAmmoniaStorage(
    H2_gen, P2EL, np.zeros(len(P2EL)), ramp_lim=1 / 240, plant_min=0.99
)
DA.calc_storage()

fig, ax = plt.subplots(3, 1, sharex="col")

ax[0].plot(DA.H2_gen)

ax[0].plot(DA.H2_chg)
ax[0].plot(DA.H2_gen + DA.H2_chg)

ramp_lim = DA.ramp_lim * DA.plant_rating
ax[1].plot(DA.H2_soc)
ax[1].plot(np.arange(ramp_lim * 8760, 0, -ramp_lim))
ax[1].plot(np.arange(-ramp_lim * 8760, 0, ramp_lim))
ax[1].set_ylim([1.1 * np.min(DA.H2_soc), 1.1 * np.max(DA.H2_soc)])

# ax[2].plot(DA.H2_soc / np.linspace(1, 0.1, 8760))
ax[2].plot(np.cumsum(DA.H2_soc) / np.cumsum(np.arange(1, len(DA.H2_soc) + 1, 1)))

plt.show()

[]
