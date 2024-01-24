import numpy as np
import pandas as pd
from pathlib import Path


data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"

# ['year', 'lat', 'lon', 'split', 'rl', 'td', 'gen_ind', 'hopp_input','storage_cap_kg', 'storage_state']
df_all = pd.read_pickle(data_path / "hopp_sweep.pkl")
# ['Unnamed: 0', 'HOPP.wind.capex', 'HOPP.wind.opex',
#    'HOPP.wind.rating_kw', 'HOPP.wind.annual_energy', 'HOPP.wind.CF',
#    'HOPP.wind.LCOE', 'HOPP.pv.capex', 'HOPP.pv.opex', 'HOPP.pv.rating_kw',
#    'HOPP.pv.annual_energy', 'HOPP.pv.CF', 'HOPP.pv.LCOE', 'HOPP.site.lat',
#    'HOPP.site.lon', 'HOPP.site.year', 'Electrolyzer.HOPP_EL.EL_capex',
#    'Electrolyzer.HOPP_EL.EL_opex', 'Electrolyzer.HOPP_EL.max_production',
#    'Electrolyzer.HOPP_EL.H2_annual_output', 'Electrolyzer.HOPP_EL.CF',
#    'H2_storage.initial_state_kg', 'H2_storage.min_demand',
#    'H2_storage.max_demand', 'H2_storage.HB_sizing',
#    'H2_storage.min_state_index', 'H2_storage.capacity_kg',
#    'H2_storage.max_chg_kgphr', 'H2_storage.min_chg_kgphr',
#    'H2_storage.financials.pipe_capex', 'H2_storage.financials.pipe_opex',
#    'H2_storage.financials.lined_capex', 'H2_storage.financials.lined_opex',
#    'H2_storage.financials.salt_capex', 'H2_storage.financials.salt_opex',
#    'Battery_storage.capacity_kWh', 'Battery_storage.max_chg_kW',
#    'Battery_storage.min_chg_kWr',
#    'Battery_storage.financials.battery_capex',
#    'Battery_storage.financials.battery_opex', 'DGA.EL.H2_tot',
#    'DGA.EL.H2_max', 'DGA.EL.P_EL_max', 'DGA.EL.rating_elec',
#    'DGA.EL.rating_H2', 'DGA.EL.capex_rated_power',
#    'DGA.EL.opex_rated_power', 'DGA.EL.capex_kgpday', 'DGA.EL.opex_kgpday',
#    'DGA.ASU.N2_tot', 'DGA.ASU.N2_max', 'DGA.ASU.P_ASU_max',
#    'DGA.ASU.rating_elec', 'DGA.ASU.rating_NH3', 'DGA.ASU.capex',
#    'DGA.ASU.opex', 'DGA.HB.NH3_tot', 'DGA.HB.NH3_max', 'DGA.HB.P_HB_max',
#    'DGA.HB.rating_elec', 'DGA.HB.rating_NH3', 'DGA.HB.capex',
#    'DGA.HB.opex', 'run_params.ramp_lim', 'run_params.turndown',
#    'run_params.plant_life', 'LT_NH3']
df_full = pd.read_csv(data_path / "full_sweep_main_df.csv")
df_full.insert(1, "Case", len(df_full) * [""])

lats = np.unique(df_full["HOPP.site.lat"])
locs = ["TX", "IA"]
# find the two cases, inflexible and BAT
rl_bat = 0.2
td_bat = 0.6

ramp_lims = np.unique(df_full["run_params.ramp_lim"])
turndowns = np.unique(df_full["run_params.turndown"])

if (rl_bat in ramp_lims) and (td_bat in turndowns):
    df_bat = df_full[
        (df_full["run_params.ramp_lim"] == rl_bat)
        & (df_full["run_params.turndown"] == td_bat)
    ]
    # df_bat[df_bat["HOPP.site.lat"] == lats[0]].at["Case", f"{locs[0]}, BAT"]
    # df_bat[df_bat["HOPP.site.lat"] == lats[1]]["Case"] = f"{locs[1]}, BAT"
    df_bat.at[61, "Case"] = "TX, BAT"
    df_bat.at[160, "Case"] = "IA, BAT"

    df_inflexible = df_full[
        (df_full["run_params.ramp_lim"] == 0) & (df_full["run_params.turndown"] == 1)
    ]
    # df_inflexible[df_inflexible["HOPP.site.lat"] == lats[0]][
    #     "Case"
    # ] = f"{locs[0]}, inflexible"
    # df_inflexible[df_inflexible["HOPP.site.lat"] == lats[1]][
    #     "Case"
    # ] = f"{locs[1]}, inflexible"
    df_inflexible.at[10, "Case"] = "TX, inflexible"
    df_inflexible.at[109, "Case"] = "IA, inflexible"

    df_print = pd.concat([df_inflexible, df_bat])
else:
    pass

capacity_columns = [
    "Case",
    "HOPP.wind.rating_kw",
    "HOPP.pv.rating_kw",
    "DGA.EL.rating_elec",
    # "DGA.EL.rating_H2",
    "H2_storage.capacity_kg",
    "Battery_storage.capacity_kWh",
    "DGA.ASU.N2_max",
    "DGA.HB.rating_NH3",
]


cap_col_names = [
    "Case",
    "Wind [kW]",
    "Solar [kW]",
    "EL [kW]",
    "H2 storage [kg]",
    "Battery [kWh]",
    "ASU [kg/hr]",
    "HB [kg/hr]",
]

df_print_cap = df_print[capacity_columns]
df_print_cap.columns = cap_col_names
print(df_print_cap.to_latex(index=False, float_format="%.0f"))

[]
