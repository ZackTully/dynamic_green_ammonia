import numpy as np
import pandas as pd
from pathlib import Path
from dynamic_green_ammonia.tools.file_management import FileMan

FM = FileMan()
FM.set_analysis_case("cases_check")

df_all, df_full = FM.load_sweep_data()
df_full.insert(1, "Case", len(df_full) * [""])
cost_excel = pd.ExcelFile(FM.costs_path / "cost_models.xlsx")

loc_info = pd.read_csv(FM.input_path / "location_info.csv")

# data_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
# data_path = Path(__file__).parents[1] / "data" / "cases_check"
# df_all = pd.read_pickle(data_path / "hopp_sweep.pkl")
# df_full = pd.read_csv(data_path / "full_sweep_main_df.csv")
# df_full.insert(1, "Case", len(df_full) * [""])
# cost_excel = pd.ExcelFile(Path(__file__).parent / "cost_models" / "cost_models.xlsx")

lats = np.unique(df_full["HOPP.site.lat"])
locs = ["TX", "IA"]

plant_life = np.unique(df_full["run_params.plant_life"])[0]

# find the two cases, inflexible and BAT
rl_bat = 0.2
td_bat = 0.6

ramp_lims = np.unique(df_full["run_params.ramp_lim"])
turndowns = np.unique(df_full["run_params.turndown"])

tol = 1e-6
if np.min(np.abs(ramp_lims - rl_bat)) < tol:
    rl_bat = ramp_lims[np.argmin(np.abs(ramp_lims - rl_bat))]

if np.min(np.abs(turndowns - td_bat)) < tol:
    td_bat = turndowns[np.argmin(np.abs(turndowns - td_bat))]


bat_inds = df_full[
    (df_full["run_params.ramp_lim"] == rl_bat)
    & (df_full["run_params.turndown"] == td_bat)
].index
inf_inds = df_full[
    (df_full["run_params.ramp_lim"] == 0) & (df_full["run_params.turndown"] == 1)
].index

for i, bi in enumerate(bat_inds):
    lat_ind = np.where(loc_info["lat"] == df_full.loc[bi]["HOPP.site.lat"])[0][0]
    lon_ind = np.where(loc_info["lon"] == df_full.loc[bi]["HOPP.site.lon"])[0][0]
    # if lat_ind == lon_ind:
    df_full.at[
        bi, "Case"
    ] = f'{loc_info.loc[lon_ind]["loc"]} {loc_info.loc[lon_ind]["note"]} BAT'

for i, ii in enumerate(inf_inds):
    lat_ind = np.where(loc_info["lat"] == df_full.loc[ii]["HOPP.site.lat"])[0][0]
    lon_ind = np.where(loc_info["lon"] == df_full.loc[ii]["HOPP.site.lon"])[0][0]
    # if lat_ind == lon_ind:
    df_full.at[
        ii, "Case"
    ] = f'{loc_info.loc[lon_ind]["loc"]} {loc_info.loc[lon_ind]["note"]} INF'

if (rl_bat in ramp_lims) and (td_bat in turndowns):
    df_print = pd.concat([df_full.loc[bat_inds], df_full.loc[inf_inds]])
else:
    print(f"BAT not within {tol} of sweep")


state = df_print["Case"].str.split(" ", expand=True)[0].values
flexibility = [case[-1] for case in df_print["Case"].str.split(" ")]
note = [" ".join(case[1:-1]) for case in df_print["Case"].str.split(" ")]

df_print.insert(1, "Note", note)
df_print.insert(1, "Flex", flexibility)
df_print.insert(1, "State", state)


LT_NH3 = df_print["LT_NH3"]
LT_NH3.index = range(len(LT_NH3))

df_capex = df_print.filter(regex="Case|capex").drop(
    [
        # "Electrolyzer.HOPP_EL.EL_capex",
        "DGA.EL.capex_rated_power",
        "DGA.EL.capex_kgpday",
        "H2_storage.financials.lined_capex",
    ],
    axis=1,
)
df_opex = df_print.filter(regex="Case|opex").drop(
    [
        # "Electrolyzer.HOPP_EL.EL_opex",
        "DGA.EL.opex_rated_power",
        "DGA.EL.opex_kgpday",
        "H2_storage.financials.lined_opex",
    ],
    axis=1,
)

sweep_cost_pipe = df_capex.drop(["H2_storage.financials.salt_capex"], axis=1).sum(
    axis=1, numeric_only=True
) + df_opex.drop(["H2_storage.financials.salt_opex"], axis=1).sum(
    axis=1, numeric_only=True
)
sweep_LCOA_pipe = sweep_cost_pipe.to_numpy() / LT_NH3.to_numpy()

sweep_cost_cavern = df_capex.drop(["H2_storage.financials.pipe_capex"], axis=1).sum(
    axis=1, numeric_only=True
) + df_opex.drop(["H2_storage.financials.pipe_opex"], axis=1).sum(
    axis=1, numeric_only=True
)
sweep_LCOA_cavern = sweep_cost_cavern.to_numpy() / LT_NH3.to_numpy()

components = ["Wind", "PV", "EL", "Pipe", "Cavern", "Battery", "ASU", "HB"]

df_sweep = pd.concat([df_capex, df_opex.drop("Case", axis=1)], axis=1)
rename_dict = {}

for col in df_sweep.columns:
    if col == "Case":
        continue

    for comp in components:
        # if comp == "Solar":
        #     comp = "PV"
        if comp == "Cavern":
            comp = "Salt"
        if comp.lower() in col.lower():
            if comp == "Salt":
                comp = "Cavern"
            if "capex" in col:
                rename_dict.update({col: f"{comp} capex"})
            if "opex" in col:
                rename_dict.update({col: f"{comp} opex"})

df_sweep = df_sweep.rename(columns=rename_dict)
df_sweep.index = range(len(df_sweep))
df_print.index = range(len(df_sweep))
df_sweep.insert(
    len(df_sweep.columns),
    "Pipe LCOA",
    df_sweep.drop(["Cavern capex", "Cavern opex"], axis=1).sum(
        axis=1, numeric_only=True
    )
    / LT_NH3,
)
df_sweep.insert(
    len(df_sweep.columns),
    "Cavern LCOA",
    df_sweep.drop(["Pipe capex", "Pipe opex"], axis=1).sum(axis=1, numeric_only=True)
    / LT_NH3,
)
# re-order


# TODO: Include AEP and total amounts produced


capacity_columns = [
    "Case",
    "HOPP.wind.rating_kw",
    "HOPP.wind.annual_energy",
    "HOPP.pv.rating_kw",
    "HOPP.pv.annual_energy",
    "DGA.EL.rating_elec",
    # "DGA.EL.rating_H2",
    "H2_storage.capacity_kg",
    "H2_storage.capacity_kg",
    "Battery_storage.capacity_kWh",
    "DGA.ASU.N2_max",
    "DGA.ASU.rating_elec",
    "DGA.HB.rating_NH3",
    "DGA.HB.rating_elec",
]


cap_col_names = [
    "Case",
    "Wind [kW]",
    "Wind [kWh]",  # wind annual energy
    "PV [kW]",
    "PV [kWh]",  # solar annual energy
    "EL [kW]",
    "Pipe [kg]",
    "Cavern [kg]",
    "Battery [kWh]",
    "ASU [kg/hr]",
    "ASU [kW]",
    "HB [kg/hr]",
    "HB [kW]",
]

df_print_cap = df_print[capacity_columns]
df_print_cap.columns = cap_col_names


# Each method should print the cost of each model and also the LCOA


def run_costs(df: pd.DataFrame, cost_model):
    # for each row (case) in df:
    # send capacities to individual cost model
    # collect costs in the output df
    cost_list = []
    for row in range(len(df)):
        cost_list.append(cost_model(df.iloc[row]))

    costs = pd.DataFrame(cost_list)

    costs.insert(
        len(costs.columns),
        "Pipe LCOA",
        costs.drop(["Cavern capex", "Cavern opex"], axis=1).sum(
            axis=1, numeric_only=True
        )
        / LT_NH3,
    )
    costs.insert(
        len(costs.columns),
        "Cavern LCOA",
        costs.drop(["Pipe capex", "Pipe opex"], axis=1).sum(axis=1, numeric_only=True)
        / LT_NH3,
    )

    return costs  # , cavern_LCOA, pipe_LCOA


def compare_labels(case_labels, cost_labels):
    # cost_in_case has the union of cost and case
    cost_in_case = []

    for cost_lab in cost_labels:
        if cost_lab in case_labels:
            cost_in_case.append(cost_lab)  # should be exact string match

    # need to get these costs from the sweep values
    comp_not_in_cost_case = []
    for comp in components:
        add_comp = True
        for costcase in cost_in_case:
            if comp in costcase:
                add_comp = False
        if add_comp:
            comp_not_in_cost_case.append(comp)

    return cost_in_case, comp_not_in_cost_case


# NL 2018 cost model
def NL2018_cost(case: pd.Series):
    """
    [1] R. Nayak-Luke, R. Bañares-Alcántara, and I. Wilkinson, “'Green' Ammonia: Impact of Renewable Energy Intermittency on Plant Sizing and Levelized Cost of Ammonia,” Ind. Eng. Chem. Res., vol. 57, no. 43, pp. 14607-14616, Oct. 2018, doi: 10.1021/acs.iecr.8b02447.
    """

    costs = pd.read_excel(cost_excel, "NL2018", index_col=0)

    # https://www.exchangerates.org.uk/GBP-USD-spot-exchange-rates-history-2018.html
    GBP2USD = 1.3349  # 2018 USD per GBP

    def capex_opex(K, S, n, OF):
        capex = K * S**n  # GBP 2018
        opex = plant_life * OF * capex  # GBP/yr 2018
        return GBP2USD * capex, GBP2USD * opex

    out_dict = {}
    out_dict.update({"Case": case["Case"]})

    label_calc, label_get = compare_labels(case.index, costs.columns)
    for label in label_calc:
        capex, opex = capex_opex(
            costs[label].loc["K"],
            case[label],
            costs[label].loc["n"],
            costs[label].loc["opex"],
        )
        out_dict.update({f"{label.split('[')[0]}capex": capex})
        out_dict.update({f"{label.split('[')[0]}opex": opex})

    for label in label_get:
        if label == "Cavern":
            label = "salt"
        component = df_print[df_print["Case"] == case["Case"]].filter(like=label[1:])
        capex = component.filter(like="capex").iloc[0][0]
        opex = component.filter(like="opex").iloc[0][0]
        if label == "salt":
            label = "Cavern"
        out_dict.update({f"{label} capex": capex})
        out_dict.update({f"{label} opex": opex})

    out_series = pd.Series(out_dict)
    # if nan then use HOPP's calculated values from the sweep
    out_series = pd.Series(out_dict)

    return out_series


def Armijo2020(case: pd.Series):
    """[2] J. Armijo and C. Philibert, “Flexible production of green hydrogen and ammonia from variable solar and wind energy: Case study of Chile and Argentina,” International Journal of Hydrogen Energy, vol. 45, no. 3, pp. 1541-1558, Jan. 2020, doi: 10.1016/j.ijhydene.2019.11.028."""
    costs = pd.read_excel(cost_excel, "Armijo2020", index_col=0)
    # USD / unit

    def capex_opex(cap, op, size):
        capex = cap * size
        opex = plant_life * op * capex
        return capex, opex

    out_dict = {}
    out_dict.update({"Case": case["Case"]})

    label_calc, label_get = compare_labels(case.index, costs.columns)
    for label in label_calc:
        capex, opex = capex_opex(
            costs[label].loc["capex"], costs[label].loc["opex"], case[label]
        )
        out_dict.update({f"{label.split('[')[0]}capex": capex})
        out_dict.update({f"{label.split('[')[0]}opex": opex})

    for label in label_get:
        component = df_print[df_print["Case"] == case["Case"]].filter(like=label)
        capex = component.filter(like="capex").iloc[0][0]
        opex = component.filter(like="opex").iloc[0][0]
        out_dict.update({f"{label} capex": capex})
        out_dict.update({f"{label} opex": opex})

    out_series = pd.Series(out_dict)

    return out_series


def Fasihi2021(case: pd.Series):
    """[3] M. Fasihi, R. Weiss, J. Savolainen, and C. Breyer, “Global potential of green ammonia based on hybrid PV-wind power plants,” Applied Energy, vol. 294, p. 116170, Jul. 2021, doi: 10.1016/j.apenergy.2020.116170.="""

    costs = pd.read_excel(cost_excel, "Fasihi2021", index_col=0)

    def capex_opex(cap, op, size):
        capex = cap * size
        opex = plant_life * op * capex

        # https://www.xe.com/currencyconverter/convert/?Amount=1&From=EUR&To=USD
        euro2USD = 1.0830221
        capex = euro2USD * capex
        opex = euro2USD * opex

        return capex, opex

    out_dict = {}
    out_dict.update({"Case": case["Case"]})
    label_calc, label_get = compare_labels(case.index, costs.columns)
    for label in label_calc:
        capex, opex = capex_opex(
            costs[label].loc["capex"], costs[label].loc["opex"], case[label]
        )
        out_dict.update({f"{label.split('[')[0]}capex": capex})
        out_dict.update({f"{label.split('[')[0]}opex": opex})

    for label in label_get:
        component = df_print[df_print["Case"] == case["Case"]].filter(like=label)
        capex = component.filter(like="capex").iloc[0][0]
        opex = component.filter(like="opex").iloc[0][0]
        out_dict.update({f"{label} capex": capex})
        out_dict.update({f"{label} opex": opex})

    out_series = pd.Series(out_dict)

    return out_series


def Smith2024(case: pd.Series):
    """[4] C. Smith and L. Torrente-Murciano, “The importance of dynamic operation and renewable energy source on the economic feasibility of green ammonia,” Joule, vol. 8, no. 1, pp. 157-174, Jan. 2024, doi: 10.1016/j.joule.2023.12.002."""
    costs = pd.read_excel(cost_excel, "Smith2024", index_col=0)

    def capex_opex(cap, op, size, component):
        if component in ["PV [kW]", "Wind [kW]", "Battery [kWh]", "Pipe [kg]"]:
            capex = cap * size
            opex = op * size * plant_life
            capex = capex * 1.18
            opex = opex * 1.18
        elif component in ["EL [kW]"]:
            capex = cap * size
            opex = op * cap * plant_life
        elif component in ["ASU [kg/hr]"]:
            capex = 33000 * (size / 10) ** 0.61
            opex = 0
            capex = capex * 1.18
        return capex, opex

    out_dict = {}
    out_dict.update({"Case": case["Case"]})
    label_calc, label_get = compare_labels(case.index, costs.columns)
    for label in label_calc:
        capex, opex = capex_opex(
            costs[label].loc["capex"], costs[label].loc["opex"], case[label], label
        )
        out_dict.update({f"{label.split('[')[0]}capex": capex})
        out_dict.update({f"{label.split('[')[0]}opex": opex})

    for label in label_get:
        if label == "Cavern":
            label = "salt"
        component = df_print[df_print["Case"] == case["Case"]].filter(like=label[1:])
        capex = component.filter(like="capex").iloc[0][0]
        opex = component.filter(like="opex").iloc[0][0]
        if label == "salt":
            label = "Cavern"
        out_dict.update({f"{label} capex": capex})
        out_dict.update({f"{label} opex": opex})

    out_series = pd.Series(out_dict)

    return out_series


# Smith cost model

NL_costs = run_costs(df_print_cap, NL2018_cost)
AR_costs = run_costs(df_print_cap, Armijo2020)
FA_costs = run_costs(df_print_cap, Fasihi2021)
SM_costs = run_costs(df_print_cap, Smith2024)

col_order = NL_costs.columns
AR_costs = AR_costs[col_order]

df_sweep = df_sweep[col_order]

df_sweep.compare(NL_costs)


def perc_diff(x1, x2):
    # percent difference of x2 compared to x1
    return (x2 - x1) / (x1)


def percent_difference(df_sweep, df_comp):
    df_pd = df_sweep.copy()
    for col in df_sweep:
        if col == "Case":
            continue
        sweep_col = df_sweep.loc[:, col]
        comp_col = df_comp.loc[:, col]
        df_pd.loc[:, col] = 100 * perc_diff(sweep_col, comp_col)

    # sweep_tot = df_sweep.sum(axis=1, numeric_only=True)
    # comp_tot = df_comp.sum(axis=1, numeric_only=True)

    df_pd.insert(
        len(df_pd.columns),
        "Pipe Total",
        100
        * perc_diff(
            df_sweep.drop(["Cavern capex", "Cavern opex"], axis=1).sum(
                axis=1, numeric_only=True
            ),
            df_comp.drop(["Cavern capex", "Cavern opex"], axis=1).sum(
                axis=1, numeric_only=True
            ),
        ),
    )

    df_pd.insert(
        len(df_pd.columns),
        "Cavern total",
        100
        * perc_diff(
            df_sweep.drop(["Pipe capex", "Pipe opex"], axis=1).sum(
                axis=1, numeric_only=True
            ),
            df_comp.drop(["Pipe capex", "Pipe opex"], axis=1).sum(
                axis=1, numeric_only=True
            ),
        ),
    )

    return df_pd


pd_NL = percent_difference(df_sweep, NL_costs)
pd_AR = percent_difference(df_sweep, AR_costs)
pd_FA = percent_difference(df_sweep, FA_costs)
pd_SM = percent_difference(df_sweep, SM_costs)

# df_sweep.drop("Case", axis=1).divide(LT_NH3, axis=0)[["Wind capex",   "Wind opex",   "PV capex",   "PV opex",   "EL capex",   "EL opex"]]
# df_sweep.drop("Case", axis=1).divide(LT_NH3, axis=0)[["Wind capex",   "Wind opex",   "PV capex",   "PV opex",   "EL capex",   "EL opex"]].sum(axis=1)


# pd.concat([df_sweep, NL_costs, pd_NL, AR_costs, pd_AR]).to_csv(
#     "dynamic_green_ammonia/analysis_scripts/cost_models/comparison.csv"
# )

comp_cols = ["Pipe LCOA", "Cavern LCOA"]

pd.concat(
    [
        df_sweep["Case"],
        df_sweep[comp_cols],
        NL_costs[comp_cols],
        pd_NL[comp_cols],
        AR_costs[comp_cols],
        pd_AR[comp_cols],
        FA_costs[comp_cols],
        pd_FA[comp_cols],
        SM_costs[comp_cols],
        pd_SM[comp_cols],
    ],
    axis=1,
).to_csv("dynamic_green_ammonia/analysis_scripts/cost_models/comparison.csv")

case_index = df_sweep["Case"].str.split(" ", expand=True).sort_values(by=[0, 1]).index
sweep_side_by_side = np.reshape(
    df_sweep.loc[case_index]["Pipe LCOA"].to_numpy(), (2, int(len(df_sweep) / 4), 2)
)

# 1-26-2024 I should use the CF cases from ESG sweep going forward

df_sweep.loc[case_index].to_csv(
    "dynamic_green_ammonia/analysis_scripts/cost_models/ESG_locations.csv"
)
df_print.to_csv("dynamic_green_ammonia/analysis_scripts/cost_models/df_print.csv")


# Re-organize df print for output to table

"""

Case
LCOE
AEP
Wind CF
PV CF
Hybrid CF
Storage capacity
Storage max mfr
Battery capacity
HBR capacity
ASU capacity
NH3 produced
LCOA

"""


# state = df_sweep["Case"].str.split(" ", expand=True)[0].values
# flexibility = [case[-1] for case in df_sweep["Case"].str.split(" ")]
# note = [" ".join(case[1:-1]) for case in df_sweep["Case"].str.split(" ")]

# df_sweep.insert(1, "Note", note)
# df_sweep.insert(1, "Flex", flexibility)
# df_sweep.insert(1, "State", state)


df_table = pd.concat(
    [
        # df_print["Case"],
        df_print["State"],
        df_print["Flex"],
        df_print["Note"],
        pd.Series(df_print["HOPP.wind.CF"] * 1e-2, name="Wind CF"),
        pd.Series(df_print["HOPP.pv.CF"] * 1e-2, name="Solar CF"),
        pd.Series(
            (df_print["HOPP.wind.annual_energy"] + df_print["HOPP.pv.annual_energy"])
            / (
                (df_print["HOPP.wind.rating_kw"] + df_print["HOPP.pv.rating_kw"]) * 8760
            ),
            name="Hybrid CF",
        ),
        pd.Series(
            (df_print["HOPP.wind.annual_energy"] + df_print["HOPP.pv.annual_energy"])
            * 1e-9,
            name="AEP [TWh]",
        ),
        pd.Series(
            (
                df_print["HOPP.wind.LCOE"] * df_print["HOPP.wind.annual_energy"]
                + df_print["HOPP.pv.LCOE"] * df_print["HOPP.pv.annual_energy"]
            )
            / (df_print["HOPP.wind.annual_energy"] + df_print["HOPP.pv.annual_energy"]),
            name="LCOE [\\textcent/kWh]",
        ),
        pd.Series(df_print["H2_storage.capacity_kg"] / 1e3, name="Storage [t]"),
        pd.Series(df_print["DGA.HB.rating_NH3"] / 1e3, name="HB Rating [t/hr]"),
        pd.Series(df_print["LT_NH3"] / plant_life / 1e3, name="Ammonia [Mt/yr]"),
        pd.Series(df_sweep["Pipe LCOA"], name="LCOA [\$/t]"),
    ],
    axis=1,
)


# tx_inds = df_table.loc[df_table["Case"].str.startswith("TX")].index
# tx_flex = df_table.loc[tx_inds]["Case"].str.endswith("BAT")
# tx_inf = df_table.loc[tx_inds]["Case"].str.endswith("INF")


# tx_table = df_table.loc[tx_inds].loc[tx_inf]
# # tx_table.insert(
# #     loc=len(df_table.columns),
# #     column="Flex LCOA",
# #     value=df_table.loc[tx_inds].loc[tx_flex]["LCOA [\\$/t]"].values,
# # )

# ia_inds = df_table.loc[df_table["Case"].str.startswith("IA")].index
# ia_flex = df_table.loc[ia_inds]["Case"].str.endswith("BAT")
# ia_inf = df_table.loc[ia_inds]["Case"].str.endswith("INF")

# ia_table = df_table.loc[ia_inds].loc[ia_inf]
# # ia_table.insert(
# #     loc=len(df_table.columns),
# #     column="Flex LCOA",
# #     value=df_table.loc[ia_inds].loc[ia_flex]["LCOA [\\$/t]"].values,
# # )


# ia_table.index = np.arange(0, len(ia_table), 1)

# case_order = [
#     "Min LCOH INF",
#     "CF and storage INF",
#     "green steel sites INF",
#     "Min storage INF",
#     "Complimentarity INF",
# ]

# ia_index = []
# tx_index = []

# for co in case_order:
#     ia_index.append(ia_table[ia_table["Case"] == f"IA {co}"].index[0])
#     tx_index.append(tx_table[tx_table["Case"] == f"TX {co}"].index[0])


def print_latex(df, precision):
    s = df.style
    # s.hide(["Case"], axis=1)
    s.format(
        # {
        #     "Wind CF": "{:.2f}",
        #     "Solar CF": "{:.2f}",
        #     "Hybrid CF": "{:.2f}",
        #     "AEP": "{:.2f}",
        #     "LCOE": "{:.2f}",
        #     "Storage": "{:.0f}",
        #     "HB Rating": "{:.0f}",
        #     "Ammonia": "{:.0f}",
        #     "LCOA": "{:.0f}",
        #     "Flex LCOA": "{:.0f}",
        # },
        precision=precision,
        escape="latex",
        # index=False
    )
    s.hide(axis=0)
    print(s.to_latex())

    []


table_notes = ["PV CF", "Wind CF", "Complimentarity", "CF and storage"]
df_table = df_table[df_table["Note"].isin(table_notes)].sort_values(
    by=["State", "Flex", "Note"], ascending=[True, True, False]
)

note_rename = {
    "CF and storage": "CF+Variability",
}

wrong_note_index = df_table[df_table["Note"] == "CF and storage"].index
for ind in wrong_note_index:
    df_table.at[ind, "Note"] = "CF+Variability"


# for row in range(len(df_table)):
#     if df_table.iloc[row]["Note"] in note_rename.keys():
#         df_table.at[row, "Note"] = note_rename[df_table.iloc[row]["Note"]]


table1_cols = [
    "Note",
    "Wind CF",
    "Solar CF",
    # "Hybrid CF",
    "LCOE [\\textcent/kWh]",
    "AEP [TWh]",
    "Ammonia [Mt/yr]",
]

table2_cols = [
    "State",
    "Flex",
    "Note",
    "Storage [t]",
    "HB Rating [t/hr]",
    "LCOA [\\$/t]",
]


# print_latex(
# df_table[(df_table["State"] == "TX") & (df_table["Flex"] == "INF")][table1_cols], 2
# )

df_table1 = []


# make multiindex for columns

mi_columns = []
data = []


for col in df_table.columns:
    if col in ["State", "Flex", "Note"]:
        mi_columns.append((col, ""))
        data.append(df_table[df_table["State"] == "IA"][col].reset_index(drop=True))
    else:
        mi_columns.append((col, "IA"))
        data.append(df_table[df_table["State"] == "IA"][col].reset_index(drop=True))
        mi_columns.append((col, "TX"))
        data.append(df_table[df_table["State"] == "TX"][col].reset_index(drop=True))

df_table1 = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(mi_columns)).transpose()


# print_latex(df_table[(df_table["Flex"] == "INF")][table1_cols], 2)
print_latex(df_table1[df_table1["Flex"] == "INF"].drop("State", axis=1)[table1_cols], 2)


# inf_index = df_table[df_table["Flex"] == "INF"].index
# flex_index = df_table[df_table["Flex"] == "BAT"].index

# df_table2 = []
# for col in table2_cols:
#     if col in ["State", "Flex", "Note"]:
#         df_table2.append(
#             pd.Series(df_table.loc[inf_index][col].reset_index(drop=True), name=col)
#         )
#         continue

#     inf_vals = df_table.loc[inf_index][col].values.astype(int).astype(str)
#     flex_vals = df_table.loc[flex_index][col].values.astype(int).astype(str)
#     print_values = [f"{inf_vals[i]} / {flex_vals[i]}" for i, val in enumerate(inf_vals)]
#     df_table.loc[inf_index][col] = print_values
#     df_table2.append(pd.Series(print_values, name=col))

# df_table2 = pd.concat(df_table2, axis=1)


inf_index = df_table1[df_table1["Flex"] == "INF"].index
flex_index = df_table1[df_table1["Flex"] == "BAT"].index

dft2_mi = []

df_table2 = []
for col in table2_cols:
    if col in ["State", "Flex", "Note"]:
        df_table2.append(
            pd.Series(df_table1.loc[inf_index][col].reset_index(drop=True), name=col)
        )
        dft2_mi.append((col, ""))
        continue

    inf_vals_ia = df_table1.loc[inf_index][col]["IA"].values.astype(int).astype(str)
    flex_vals_ia = df_table1.loc[flex_index][col]["IA"].values.astype(int).astype(str)
    print_values = [
        f"{inf_vals_ia[i]} / {flex_vals_ia[i]}" for i, val in enumerate(inf_vals_ia)
    ]
    df_table1.loc[inf_index][col]["IA"] = print_values
    df_table2.append(pd.Series(print_values, name=(col, "IA")))
    dft2_mi.append((col, "IA"))

    inf_vals_tx = df_table1.loc[inf_index][col]["TX"].values.astype(int).astype(str)
    flex_vals_tx = df_table1.loc[flex_index][col]["TX"].values.astype(int).astype(str)
    print_values = [
        f"{inf_vals_tx[i]} / {flex_vals_tx[i]}" for i, val in enumerate(inf_vals_tx)
    ]
    df_table1.loc[inf_index][col]["TX"] = print_values
    df_table2.append(pd.Series(print_values, name=(col, "TX")))
    dft2_mi.append((col, "TX"))

# df_table2 = pd.concat(df_table2, axis=1)
df_table2 = pd.DataFrame(
    df_table2, index=pd.MultiIndex.from_tuples(dft2_mi)
).transpose()

print_latex(df_table2.drop(["State", "Flex"], axis=1), precision=0)
# print_latex(ia_table.loc[ia_index][table1_cols], 2)
# print_latex(ia_table.loc[ia_index][table2_cols], 0)

# ia_s = ia_table.loc[ia_index].style


# tuples = list(zip(*[state, flexibility, note]))
# index = pd.MultiIndex.from_tuples(tuples, names=["State", "Flex", "Note"])
# df_sweep.index = index
# df_sweep = df_sweep.sort_values(["State", "Flex", "Note"])

[]
