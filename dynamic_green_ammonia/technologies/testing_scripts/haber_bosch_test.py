import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[3]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import molmass
import numpy as np
import matplotlib.pyplot as plt

from Dynamic_Load.HOPP_DL.technologies.ammonia import HaberBosch

mm_H2 = molmass.Formula("H2").mass  # [g/mol] molar mass hydrogen
mm_N2 = molmass.Formula("N2").mass  # [g/mol] molar mass nitrogen
mm_NH3 = molmass.Formula("NH3").mass  # [g/mol] molar mass ammonia

# N2 + 3 H2 <-> 2 NH3

gpm_H2 = (3 / 2) * mm_H2  # g H2 in 1 mol NH3
gpm_N2 = (1 / 2) * mm_N2  # g N2 in 1 mol NH3

gpg_H2 = gpm_H2 / mm_NH3  # g H2 in 1 g NH3
gpg_N2 = gpm_N2 / mm_NH3  # g N2 in 1 g NH3

energy_H2 = 53.4  # kWh/kg H2
energy_N2 = 0.119  # kWh/kg N2
energy_NH3 = 0.6  # kWh/kg NH3

HB = HaberBosch()


ideal_inputs = np.array([HB.kgpkg_H2, HB.kgpkg_N2, HB.energypkg_NH3])
ideal_inputs = ideal_inputs / np.max(ideal_inputs)


t_start = 0
t_stop = 100
t_delta = 1

time = np.arange(t_start, t_stop, t_delta)

H2 = 10
N2 = 10
P = 10
gain_P = 0.5

signals = np.zeros([len(time), 4])

for i in range(len(time)):
    if i == 0:
        NH3, reject = HB.step(H2, N2, P)
        continue

    NH3, reject = HB.step(H2, N2, P)

    H2 += -gain_P * reject[0]
    N2 += -gain_P * reject[1]
    P += -gain_P * reject[2]

    signals[i, :] = [NH3, H2, N2, P]

fig, ax = plt.subplots(2, 1, sharex="col")

ax[0].plot(time, signals[:, 1:])
ax[1].plot(time, signals[:, 0])

plt.show()


[]
