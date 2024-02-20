import numpy as np
import molmass
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


from pathlib import Path

style = "paper"

if style == "paper":
    plt.style.use(Path(__file__).parent / "paper_figs.mplstyle")
elif style == "pres":
    plt.style.use(Path(__file__).parent / "pres_figs.mplstyle")


save_path = Path(__file__).parents[1] / "plots"

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


# calculated energy per kg ammonia is 11.310324266229935 kWh/kg


# per 1 kWh of electricity

hbr_color = (210 / 255, 107 / 255, 102 / 255)
storage_color = (97 / 255, 15 / 255, 161 / 255)
H2_color = (210 / 255, 107 / 255, 102 / 255)
N2_color = (97 / 255, 6 / 255, 13 / 255)
P_color = (166 / 255, 161 / 255, 161 / 255)

H2_color = (210 / 255, 107 / 255, 102 / 255)
N2_color = (97 / 255, 6 / 255, 13 / 255)
P_color = (166 / 255, 161 / 255, 161 / 255)

EL_color = (244 / 255, 228 / 255, 84 / 255)
EL_color = "blue"
ASU_color = (189 / 255, 80 / 255, 119 / 255)

fig = plt.figure(figsize=(3.5, 3))

ax = fig.add_subplot(
    1,
    1,
    1,
    xticks=[],
    yticks=[],
)

ax.spines[["right", "top", "left", "bottom"]].set_visible(False)

sankey = Sankey(
    ax=ax,
    gap=0.5,
    radius=0.45,
    margin=0,
    shoulder=0.0,
    head_angle=120,
    offset=2.5,
    format=lambda x: f"{np.abs(x):.3f}",
    unit=None,
)
sankey.add(
    flows=[engy_pkg[0], engy_pkg[1], engy_pkg[2], -total_engy],
    orientations=[0, -1, 1, 0],
    pathlengths=[3, 1, 1, 4],
    labels=[
        f"H$_2$\n{mass_frac[0]:.3f} kg",
        f"N$_2$\n{mass_frac[1]:.3f} kg",
        None,
        "NH$_3$\n1 kg",
    ],
    facecolor=hbr_color,
    edgecolor=hbr_color,
)
#  f"Energy\n{total_engy:.3f} kWh"

sankey.add(
    flows=[-engy_pkg[0], engy_pkg[0]],
    orientations=[0, 0],
    prior=0,
    connect=(0, 0),
    pathlengths=[3, 3],
    labels=[
        # f"H$_2$\n{mass_frac[0]:.3f} kg",
        None,
        f"Electrolyzer\n{engy_pkg[0]:.3f} kWh",
    ],
    facecolor=EL_color,
    edgecolor=EL_color,
    
)
sankey.add(
    flows=[-engy_pkg[1], engy_pkg[1]],
    orientations=[0, 1],
    prior=0,
    connect=(1, 0),
    pathlengths=[1, 6],
    labels=[
        # f"N$_2$\n{mass_frac[1]:.3f} kg",
        None,
        f"ASU\n{engy_pkg[1]:.3f} kWh",
    ],
    facecolor=ASU_color,
    edgecolor=ASU_color,
)
sankey.add(
    flows=[-engy_pkg[2], engy_pkg[2]],
    orientations=[0, -1],
    prior=0,
    connect=(2, 0),
    pathlengths=[1, 6],
    labels=[None, f"BOS\n{engy_pkg[2]:.3f} kWh"],
    facecolor=P_color,
    edgecolor=P_color,
)


# ax.text(-15, 0, "test")

# sankey.add(flows=[9.6, 0.2, 0.6], orientations=[0,0,0], prio)
outputs = sankey.finish()

fig.savefig(save_path / "sankey.png", format="png")
[]

# fig = plt.figure(figsize=(3.5, 3))

# ax = fig.add_subplot(
#     1,
#     1,
#     1,
#     xticks=[],
#     yticks=[],
# )

# ax.spines[["right", "top", "left", "bottom"]].set_visible(False)

# sankey = Sankey(
#     ax=ax,
#     gap=0.5,
#     radius=0.45,
#     margin=0,
#     shoulder=0.0,
#     head_angle=120,
#     offset=2,
#     format=lambda x: f"{np.abs(x):.3f}",
#     unit=None,
# )
# sankey.add(
#     flows=[total_engy, -engy_pkg[2], -engy_pkg[1], -engy_pkg[0]],
#     orientations=[0, -1, 1, 0],
#     pathlengths=[5, 1, 1, 2],
#     labels=[f"Energy\n{total_engy:.3f} kWh", None, None, None],
#     facecolor=P_color,
#     edgecolor=P_color,
# )
# sankey.add(
#     flows=[engy_pkg[0], -engy_pkg[0]],
#     orientations=[0, 0],
#     prior=0,
#     connect=(3, 0),
#     pathlengths=[2, 2],
#     labels=[f"{engy_pkg[0]:.3f} kWh", f"H$_2$\n{mass_frac[0]:.3f} kg"],
#     facecolor=H2_color,
#     edgecolor=H2_color,
# )
# sankey.add(
#     flows=[engy_pkg[1], -engy_pkg[1]],
#     orientations=[-1, 0],
#     prior=0,
#     connect=(2, 0),
#     pathlengths=[1, 6],
#     labels=[f"{engy_pkg[1]:.3f} kWh", f"N$_2$\n{mass_frac[1]:.3f} kg"],
#     facecolor=N2_color,
#     edgecolor=N2_color,
# )
# sankey.add(
#     flows=[engy_pkg[2], -engy_pkg[2]],
#     orientations=[1, 0],
#     prior=0,
#     connect=(1, 0),
#     pathlengths=[1, 6],
#     labels=[None, f"BOS\n{engy_pkg[2]:.3f} kWh"],
#     facecolor=P_color,
#     edgecolor=P_color,
# )

# # sankey.add(flows=[9.6, 0.2, 0.6], orientations=[0,0,0], prio)
# outputs = sankey.finish()

# fig.savefig(save_path / "sankey.png", format="png")
[]

# def smooth_line(x0, y0, x1, y1):
#     # y = ax**3 + bx**2 + cx + d
#     # y' = 3ax**2 + 2bx + c + 0

#     A = np.array(
#         [
#             [x0**3, x0**2, x0, 1],  # y(x0) = y0
#             [x1**3, x1**2, x1, 1],  # y(x1) = y1
#             [3 * x0**2, 2 * x0, 1, 0],  # y'(x0) = 0
#             [3 * x1**2, 2 * x1, 1, 0],
#         ]
#     )  # y'(x1) = 0
#     b = np.array([y0, y1, 0, 0])
#     abcd = np.linalg.inv(A) @ b

#     x = np.linspace(x0, x1, 100)
#     y = abcd[0] * x**3 + abcd[1] * x**2 + abcd[2] * x + abcd[3]
#     return x, y


# fig, ax = plt.subplots(1, 1, figsize=[3.5, 3])


# x_pos = [0, 1, 2]
# left_points = np.array([[0, 0.6], [0.6, 0.8], [0.8, 10.4]])
# mid_points = np.array([[-0.2, 0.4], [0.6, 0.8], [1, 10.6]])
# right_points = left_points = np.array([[0, 0.6], [0.6, 0.8], [0.8, 10.4]])

# left_colors = ["blue", "blue", "blue"]
# right_colors = ["blue", "orange", "orange"]
# # power, nitrogen, hydrogen
# # leftleft_text = [f"{engy_pkg[0]} kWh", f"{engy_pkg[1]} kWh", f"{engy_pkg[2]} kWh"]
# # leftright_text = []
# # rightleft_text = []
# # rightright_text = []

# left_text = [
#     f"{engy_pkg[2]:.2f} kWh H2",
#     f"{engy_pkg[1]:.2f} kWh N2",
#     f"{engy_pkg[0]:.2f} kWh NH3",
# ]
# right_text = [f"", f"{mass_frac[1]:.2f} kg N2", f"{mass_frac[0]:.2f} kg H2"]

# outline_kwargs = dict(color="black", linewidth=0.25)

# ax.text(-0.05, 5, "10.18 kWh", ha="right")
# ax.text(2.05, 5, "1 kg NH3", ha="left")

# for i in range(np.shape(left_points)[0]):
#     # fill left to mid
#     bot_x, bot_y = smooth_line(x_pos[0], left_points[i, 0], x_pos[1], mid_points[i, 0])
#     top_x, top_y = smooth_line(x_pos[0], left_points[i, 1], x_pos[1], mid_points[i, 1])
#     ax.fill_between(bot_x, bot_y, top_y, color=left_colors[i])

#     ax.plot(bot_x, bot_y, **outline_kwargs)
#     ax.plot(top_x, top_y, **outline_kwargs)
#     ax.text(0.5 * (x_pos[0] + x_pos[1]), np.mean(left_points[i, :]), left_text[i])

#     # fill mid to right
#     bot_x, bot_y = smooth_line(x_pos[1], mid_points[i, 0], x_pos[2], right_points[i, 0])
#     top_x, top_y = smooth_line(x_pos[1], mid_points[i, 1], x_pos[2], right_points[i, 1])
#     ax.fill_between(bot_x, bot_y, top_y, color=right_colors[i])
#     ax.plot(bot_x, bot_y, **outline_kwargs)
#     ax.plot(top_x, top_y, **outline_kwargs)

#     ax.plot([x_pos[0], x_pos[0]], left_points[i, :], **outline_kwargs)
#     ax.plot([x_pos[1], x_pos[1]], mid_points[i, :], **outline_kwargs)
#     ax.plot([x_pos[2], x_pos[2]], right_points[i, :], **outline_kwargs)
#     ax.text(0.5 * (x_pos[1] + x_pos[2]), np.mean(left_points[i, :]), right_text[i])


# ax.spines[["left", "top", "right", "bottom"]].set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])


fig.savefig(save_path / "sankey.png", format="png")

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
