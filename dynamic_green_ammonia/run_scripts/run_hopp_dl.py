# %%

import matplotlib.pyplot as plt
import numpy as np
import os

from hopp.simulation.hopp import Hopp
from Dynamic_Load.technologies.chemical import Control
 #%%

fname = "input.yaml"
fpath = os.path.join(os.path.dirname(__file__), "inputs")

hp = Hopp.from_file(os.path.join(fpath, fname))

#%%

hp.simulate(1)
gen_profile = hp.system.generation_profile

wind_power = np.array(gen_profile["wind"])
pv_power = np.array(gen_profile["pv"])


fig, ax = plt.subplots(2, 1)
ax[0].plot(wind_power)
ax[1].plot(pv_power)

# plt.show()

#%%

dt = 1  # hour
time = np.arange(0, len(wind_power), dt)

ctrl = Control()

ctrl.wt.power = wind_power
ctrl.pv.power = pv_power

n_comps = len(ctrl.ci)

I = np.zeros([n_comps, n_comps, len(time)])  # information exchange
P = np.zeros([n_comps, n_comps, len(time)])  # power exchange
C = np.zeros([n_comps, n_comps, len(time)])  # chemical exchange

nh3 = np.zeros(len(time))

for i in range(len(time)):
    ctrl.step()
    I[:, :, i] = ctrl.I
    P[:, :, i] = ctrl.P
    C[:, :, i] = ctrl.C

fig, ax = plt.subplots(3, 1, sharex="col", dpi = 300)
ax[0].plot(time, wind_power + pv_power, label="HRES Power")
ax[0].legend()
ax[0].set_ylabel("kWh")
# ax[0].yaxis.tick_right()

ax[1].plot(time, P[ctrl.ci["el"], ctrl.ci["paa"], :], label="el")
ax[1].plot(time, P[ctrl.ci["asu"], ctrl.ci["paa"], :], label="asu")
ax[1].legend()
ax[1].set_ylabel("kWh")

ax[2].plot(time, C[ctrl.ci["out"], ctrl.ci["hb"], :], label="NH3")
ax[2].legend()
ax[2].set_ylabel("kg")
ax[2].set_xlabel("Time (hr)")
ax[2].set_xlim([0, 200])

fig.savefig("figures/timeseries.png", format="png")



plt.show()



#%%