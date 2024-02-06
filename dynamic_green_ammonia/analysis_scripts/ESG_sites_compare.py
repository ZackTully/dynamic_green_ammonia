import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path(__file__).parents[1] / "data" / "ESG_sweep"

sites_tx = pd.read_csv(data_path / "Texas_sites.csv")
sites_tx.insert(1, "Case", len(sites_tx) * [""])


sites_ia = pd.read_csv(data_path / "Iowa_sites.csv")
sites_ia.insert(1, "Case", len(sites_ia) * [""])


TX_case = ["TX GS", "TX CF", "TX LCOH", "TX Storage"]
IA_case = ["IA GS", "IA CF", "IA LCOH", "IA Storage"]

TX_locs = np.array(
    [[34.196, -102.77], [30.735, -102.457], [32.257, -99.256], [36.069, -102.819]]
)
IA_locs = np.array(
    [[42.504, -90.692], [41.817, -94.882], [43.401, -91.220], [41.494, -92.834]]
)

tx_ind = []
ia_ind = []
for i in range(len(TX_locs)):
    tx_i = sites_tx[
        (sites_tx["latitude"] == TX_locs[i, 0])
        & (sites_tx["longitude"] == TX_locs[i, 1])
    ].index[0]
    tx_ind.append(tx_i)
    sites_tx.at[tx_i, "Case"] = TX_case[i]

    ia_i = sites_ia[
        (sites_ia["latitude"] == IA_locs[i, 0])
        & (sites_ia["longitude"] == IA_locs[i, 1])
    ].index[0]
    ia_ind.append(ia_i)
    sites_ia.at[ia_i, "Case"] = IA_case[i]

cases = pd.concat([sites_tx.loc[tx_ind], sites_ia.loc[ia_ind]])

keys = ["Case", "wind_size_mw", "solar_size_mw", "hydrogen_storage_size_kg", "success"]


sites_tx = sites_tx[sites_tx["success"] == False]
sites_ia = sites_ia[sites_ia["success"] == False]
# find cases that:
# 1. have good CF
# 2. have similar CF
# 3. have similar H2 storage / installed capacity

hybrid_cf_ia = (sites_ia["WIND: Capacity Factor"] + sites_ia["PV: Capacity Factor"]) / 2
hybrid_cf_tx = (sites_tx["WIND: Capacity Factor"] + sites_tx["PV: Capacity Factor"]) / 2

hybrid_cap_ia = sites_ia["wind_size_mw"] + sites_ia["solar_size_mw"]
hybrid_cap_tx = sites_tx["wind_size_mw"] + sites_tx["solar_size_mw"]

cap_storage_ia = sites_ia["hydrogen_storage_size_kg"] / hybrid_cap_ia
cap_storage_tx = sites_tx["hydrogen_storage_size_kg"] / hybrid_cap_tx


tol = 1e-3

min_cf = np.max([hybrid_cf_tx.min(), hybrid_cf_ia.min()])
max_cf = np.min([hybrid_cf_tx.max(), hybrid_cf_ia.max()])
# cf_step = (max_cf - min_cf) / 100
cf_step = 3e-3

index_ia = hybrid_cf_ia[hybrid_cf_ia.between(max_cf - cf_step, max_cf)].index
index_tx = hybrid_cf_tx[hybrid_cf_tx.between(max_cf - cf_step, max_cf)].index


max_common_st = np.min([cap_storage_ia[index_ia].max(), cap_storage_tx[index_tx].max()])

ind_ia = (
    cap_storage_ia[index_ia]
    .iloc[(cap_storage_ia[index_ia] - max_common_st).abs().argsort()]
    .index[0]
)
ind_tx = (
    cap_storage_tx[index_tx]
    .iloc[(cap_storage_tx[index_tx] - max_common_st).abs().argsort()]
    .index[0]
)

hybrid_cf_ia[ind_ia]
hybrid_cf_tx[ind_tx]
cap_storage_ia[ind_ia]
cap_storage_tx[ind_tx]

print(
    f"good CF similar storage IA: [{sites_ia.loc[ind_ia]['latitude']}, {sites_ia.loc[ind_ia]['longitude']}]"
)
print(
    f"good CF similar storage TX: [{sites_tx.loc[ind_tx]['latitude']}, {sites_tx.loc[ind_tx]['longitude']}]"
)

# (sites_ia["Life: Average Annual Hydrogen Produced [kg]"] / sites_ia["hydrogen_storage_size_kg"]).plot()


# other cases to try: best correlation coefficient

cc_ia = sites_ia.loc[sites_ia.sort_values(by="correlation_coeff").index[0]]
cc_tx = sites_tx.loc[sites_tx.sort_values(by="correlation_coeff").index[0]]

print(f"best complimentarity IA: [{cc_ia['latitude']}, {cc_ia['longitude']}]")
print(f"best complimentarity TX: [{cc_tx['latitude']}, {cc_tx['longitude']}]")


ia_dict = {}
ia_inds = []

tx_dict = {}
tx_inds = []
for col in sites_ia.columns:
    inds = sites_ia.sort_values(by=col).index[[0, -1]]
    ia_inds.append(inds.values)
    ia_dict.update({col: inds})

    inds = sites_tx.sort_values(by=col).index[[0, -1]]
    tx_inds.append(inds.values)
    tx_dict.update({col: inds})

ia_unique = np.unique(np.concatenate(ia_inds))
tx_unique = np.unique(np.concatenate(tx_inds))

ia_not = sites_ia.loc[ia_unique]
ia_not.insert(1, "min attr", [""] * len(ia_not))
ia_not.insert(1, "max attr", [""] * len(ia_not))

for col in ia_dict.keys():
    locs = ia_dict[col]
    ia_not.at[locs[0], "min attr"] = f"{ia_not.loc[locs[0]]['min attr']}, {col}"
    ia_not.at[locs[1], "max attr"] = f"{ia_not.loc[locs[0]]['max attr']}, {col}"


tx_not = sites_tx.loc[tx_unique]
tx_not.insert(0, "min attr", [""] * len(tx_not))
tx_not.insert(0, "max attr", [""] * len(tx_not))

for col in tx_dict.keys():
    locs = tx_dict[col]
    tx_not.at[locs[0], "min attr"] = f"{tx_not.loc[locs[0]]['min attr']}, {col}"
    tx_not.at[locs[1], "max attr"] = f"{tx_not.loc[locs[0]]['max attr']}, {col}"


[]
