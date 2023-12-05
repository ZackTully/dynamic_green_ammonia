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

from Dynamic_Load.HOPP_DL.technologies.ammonia import HaberBosch, AirSeparationUnit

HB = HaberBosch(dt=1, rating=3)
ASU = AirSeparationUnit(dt=1, rating=1)

t_start = 0
t_stop = 100
t_delta = 1

time = np.arange(t_start, t_stop, t_delta)  # [s]

P_avail = 10 * np.ones(len(time))  # [kW]
H2_avail = 5 * np.ones(len(time))  # [kg/s]

# P_avail = np.linspace(0.01, 10, len(time))
P_avail = np.concatenate([0.01 * np.ones(10), 3.5 * np.ones(90)])
H2_avail = np.linspace(0, 1 / 3600, len(time))


def ctrl(P_avail, H2_avail, error, P_split_prev):
    e_P_ASU = error[0]
    e_H2_HB = 1e1 * error[1]
    e_N2_HB = 1e1 * error[2]
    e_P_HB = error[3]

    # if there is P_HB error then send more power to ASU
    # if there is N2_HB error then send more power to HB

    P_split = P_split_prev + 0.01 * (e_P_HB - e_N2_HB)

    P2ASU = 1 / (1 + P_split) * P_avail
    P2HB = P_split / (1 + P_split) * P_avail
    H22HB = H2_avail

    return P2ASU, P2HB, H22HB


error = [1, 1, 1, 1]

chemicals = np.zeros([len(time), 2])
rejects = np.zeros([len(time), 4])
signals = np.zeros([len(time), 3])


P2ASU = 1
P2HB = 1

for i in range(len(time)):
    P2ASU, P2HB, H22HB = ctrl(P_avail[i], H2_avail[i], error, P2ASU / P2HB)

    N2, reject_ASU = ASU.step(P2ASU)

    NH3, reject_HB = HB.step(H22HB, N2, P2HB)

    chemicals[i, :] = [N2, NH3]
    rejects[i, :] = [reject_ASU, *reject_HB]
    error = [reject_ASU, *reject_HB]
    signals[i, :] = [P2ASU, P2HB, H22HB]


fig, ax = plt.subplots(3, 1, sharex="col")

# inputs
ax[0].plot(time, P_avail, label="power [kW]")
ax[0].plot(time, H2_avail, label="H2 [kg/s]")
ax[0].legend()

# outputs
ax[1].plot(time, H2_avail, linestyle="dotted", alpha=0.5, label="H2 [kg/s]")
ax[1].plot(time, chemicals[:, 0], linestyle="solid", label="N2 [kg/s]")
ax[1].plot(time, chemicals[:, 1], linestyle="solid", label="NH3 [kg/s]")
ax[1].legend()

# rejects
ax[2].plot(time, rejects[:, 0], label="P_ASU")
ax[2].plot(time, rejects[:, 1], label="H2_HB")
ax[2].plot(time, rejects[:, 2], label="N2_HB")
ax[2].plot(time, rejects[:, 3], label="P_HB")
ax[2].legend()


fig, ax = plt.subplots(2, 1, sharex="col")

# power
ax[0].plot(time, P_avail, linestyle="dotted", label="power [kW]")
ax[0].plot(time, signals[:, 0], linestyle="dashed", label="P_ASU [kW]")
ax[0].plot(time, signals[:, 1], linestyle="dashed", label="P_HB [kW]")
ax[0].plot(time, rejects[:, 0], linestyle="solid", label="reject ASU [kW]")
ax[0].plot(time, rejects[:, 3], linestyle="solid", label="reject HB [kW]")
ax[0].legend()


# chemicals
ax[1].plot(time, H2_avail, linestyle="dotted", label="H2 [kg/s]")
ax[1].plot(time, rejects[:, 1], linestyle="dashed", label="reject H2 [kg/s]")
ax[1].plot(time, rejects[:, 2], linestyle="dashed", label="reject N2 [kg/s]")
ax[1].plot(time, chemicals[:, 0], linestyle="solid", label="N2 [kg/s]")
ax[1].plot(time, chemicals[:, 1], linestyle="solid", label="NH3 [kg/s]")
ax[1].legend()

[]


# ========================== ASU test =================================================

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

# ===================================== HB Test ======================================

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
