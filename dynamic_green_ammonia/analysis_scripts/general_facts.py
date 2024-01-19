import numpy as np
import molmass
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


# Hydrogen
# https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
rho_h2 = 0.090  # [kg/m^3]
HHV_h2 = 39.4  # [kWh/kg]
LHV_h2 = 33.3  # [kWh/kg]

armijo_eta_LHV = 0.7  # 70%
armijo_eff = LHV_h2 / armijo_eta_LHV
# Armijo uses eta_LHV = 70% ---> 47.57 alkaline electrolyzer
# Capex = 450 USD/kW

# PEM
NL_eff = 53.4  # kWh / kg

# for [2020, 2030, 2040, 2050] alkaline
Fasihi_eta_LHV = np.array([0.733, 0.762, 0.792, 0.821])
Fasihi_eff = LHV_h2 / Fasihi_eta_LHV  # kWh/kg

wang_eta_LHV = np.array([0.68, 0.84])
wang_eff = LHV_h2 / wang_eta_LHV


# per 1 kg of NH3 produced
molmass_H2 = molmass.Formula("H2").mass  # g/mol
molmass_N2 = molmass.Formula("N2").mass  # g/mol
molmass_NH3 = molmass.Formula("NH3").mass  # g/mol

kgpkg_H2 = (3 / 2) * molmass_H2 / molmass_NH3  # kg H2 per kg NH3
kgpkg_N2 = (1 / 2) * molmass_N2 / molmass_NH3  # kg H2 per kg NH3

energypkg_NH3 = 0.6  # kWh/kg
energypkg_H2 = 53.4  # kWh/kg
energypkg_N2 = 0.119  # kWh/kg

engy_pkg = np.array([energypkg_H2 * kgpkg_H2, energypkg_N2 * kgpkg_N2, energypkg_NH3])
total_engy = np.sum(engy_pkg)  # kWh/kg
engy_frac = engy_pkg / total_engy
mass_frac = np.array([kgpkg_H2, kgpkg_N2, 0])

# per 1 kWh of electricity


# Sankey diagram
# sankey = Sankey()
# sankey.add()
# sankey.add()
# sankey.add()
# sankey.finish()
# Sankey(
#     flows=[0.25, 0.15, 0.60, -0.20, -0.15, -0.05, -0.50, -0.10],
#     labels=["", "", "", "First", "Second", "Third", "Fourth", "Fifth"],
#     orientations=[-1, 1, 0, 1, 1, 1, 0, -1],
# ).finish()

# Sankey(
#     flows=[1, -0.25, -0.5, -0.25],
#     labels=["", "H2", "N2", "NH3"],
#     orientations=[0, 1, 0, -1],
# ).finish()

# sk = Sankey()
# sk.add("Energy", [1, -0.5], [0, 0], ["energy", "H2"], trunklength=2)
# # sk.add("H2", 0.5, 0, trunklength=1)
# sk.finish()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
# flows = [0.25, 0.15, 0.60, -0.10, -0.05, -0.25, -0.15, -0.10, -0.35]
# sankey = Sankey(ax=ax, unit=None)
# sankey.add(flows=flows, label="one", orientations=[-1, 1, 0, 1, 1, 1, -1, -1, 0])
# sankey.add(
#     flows=[-0.25, 0.15, 0.1],
#     label="two",
#     orientations=[-1, -1, -1],
#     prior=0,
#     connect=(0, 0),
# )
# diagrams = sankey.finish()
# diagrams[-1].patch.set_hatch("/")
# plt.legend()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Two Systems")
# flows = [1, -0.25, -0.5, -0.25]
# sankey = Sankey(ax=ax)
# sankey.add(flows=flows, label="one", orientations=[0, -1, 0, 1], trunklength=5)
# sankey.add(flows=[0.25, 1, -1.25], orientations=[1, 0, 0], prior=0, connect=(1, 0))
# diagrams = sankey.finish()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# sankey = Sankey(ax=ax)
# sankey.add(flows=[10.4, -9.6, -0.2, -0.6], orientations=[0, 0, 0, 0], trunklength=20)
# sankey.add(flows=[9.6, -9.6], orientations=[0, 0], prior=0, connect=(1, 0))
# sankey.add(flows=[0.2, -0.2], orientations=[0,0], prior=0, connect=(2, 0))
# sankey.add(flows=[9.6, 0.2, 0.6], orientations=[0,0,0], prio)
# sankey.finish()

# import plotly.graph_objects as go

# fig = go.Figure(
#     data=[
#         go.Sankey(
#             node=dict(
#                 pad=30,
#                 thickness=5,
#                 line=dict(color="black", width=0.5),
#                 label=["Gen", "H2", "N2", "NH3"],
#                 color="blue",
#             ),
#             link=dict(
#                 source=[0, 0, 0, 1, 2],
#                 target=[1, 2, 3, 3, 3],
#                 value=[9.6, 0.2, 0.6, 9.6, 0.2],
#             ),
#         )
#     ]
# )

# # fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
# fig.show()

[]
