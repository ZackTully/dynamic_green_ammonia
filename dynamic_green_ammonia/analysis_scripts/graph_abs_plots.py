import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
from dynamic_green_ammonia.tools.file_management import FileMan


FM = FileMan()
FM.set_analysis_case("LCOA")

H2_gen, wind_gen, solar_gen = FM.load_gen_data()
hybrid_gen = wind_gen + solar_gen
df_all, df_full = FM.load_sweep_data()


lat = df_all.iloc[0]["lat"]
lon = df_all.iloc[0]["lon"]

resource_path = Path(
    "/Users/ztully/Documents/Green_Steel/HOPP_green_steel/HOPP/resource_files"
)
solar_files = os.listdir(resource_path / "solar")
solar_file = [sf for sf in solar_files if ((str(lat) in sf) & (str(lon) in sf))][0]
wind_files = os.listdir(resource_path / "wind")
wind_file = [
    wf
    for wf in wind_files
    if ((str(lat) in wf) & (str(lon) in wf) & ("60min_100m" in wf))
][0]

# solar_resource = np.genfromtxt(resource_path / "solar" / solar_file, delimiter=",")
solar_resource = pd.read_csv(resource_path / "solar" / solar_file, header=2)
wind_resource = pd.read_csv(resource_path / "wind" / wind_file, header=[2, 3, 4])

t_i = 626
t_f = 720
time = np.arange(t_i, t_f, 1)


def style_plot():
    fig = plt.figure(figsize=(1.5, 1), layout="constrained")
    ax = fig.add_subplot()

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t")

    return fig, ax

save_path = Path("dynamic_green_ammonia/plots/presentation_plots")

fig, ax = style_plot()
ax.plot(time, solar_resource["DNI"].iloc[t_i:t_f].to_numpy())
ax.set_ylabel("Irradience")
fig.savefig(save_path / "GA_irradience.png", format="png", dpi=300)


fig, ax = style_plot()
ax.plot(time, wind_resource["Speed"].iloc[t_i:t_f].to_numpy())
ax.set_ylabel("Wind speed")
fig.savefig(save_path / "GA_wind_speed.png", format="png", dpi=300)


fig, ax = style_plot()
ax.plot(time, wind_gen[0,t_i:t_f])
ax.set_ylabel("kW")
fig.savefig(save_path / "GA_wind_gen.png", format="png", dpi=300)

fig, ax = style_plot()
ax.plot(time, solar_gen[0,t_i:t_f])
ax.set_ylabel("kW")
fig.savefig(save_path / "GA_solar_gen.png", format="png", dpi=300)

fig, ax = style_plot()
ax.plot(time, hybrid_gen[0,t_i:t_f])
ax.set_ylabel("kW")
fig.savefig(save_path / "GA_hybrid_gen.png", format="png", dpi=300)


[]
