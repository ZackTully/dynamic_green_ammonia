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

from Dynamic_Load.HOPP_DL.technologies.ammonia import AirSeparationUnit

t_start = 0
t_stop = 100
t_delta = 1

time = np.arange(t_start, t_stop, t_delta)

power = np.linspace(0, 2, len(time))


ASU = AirSeparationUnit()

signals = np.zeros([len(time), 2])

for i in range(len(time)):
    N2, reject = ASU.step(power[i])
    signals[i, 0] = N2
    signals[i, 1] = reject

fig, ax = plt.subplots(2, 1, sharex="col")
ax[0].plot(time, power)
ax[0].plot(time, signals[:, 1])
ax[0].plot(time, power - signals[:, 1])

ax[1].plot(time, signals[:, 0])

plt.show()


[]
