import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_path = Path(__file__).parents[1] / "data" / "ESG_sweep"

sites_tx = pd.read_csv(data_path / "Texas_sites.csv")
sites_ia = pd.read_csv(data_path / "Iowa_sites.csv")


omit_columns = [
    "Unnamed: 0",
    "index",
    "latitude",
    "longitude",
    "state",
    "electrolyzer_size_mw",
    "success",
    "desal_size_kg_pr_sec",
    "LCOH: Advanced-max_policy-NoStorage",
    "LCOH: Moderate-no_policy-NoStorage",
    "LCOH: Moderate-max_policy-NoStorage",
    "LCOH: Conservative-no_policy-NoStorage",
    "LCOH: Conservative-max_policy-NoStorage",
    "LCOH: Advanced-no_policy-SaltCavern",
    "LCOH: Advanced-max_policy-SaltCavern",
    "LCOH: Moderate-no_policy-SaltCavern",
    "LCOH: Moderate-max_policy-SaltCavern",
    "LCOH: Conservative-no_policy-SaltCavern",
    "LCOH: Conservative-max_policy-SaltCavern",
    "LCOH: Advanced-no_policy-BuriedPipes",
    "LCOH: Advanced-max_policy-BuriedPipes",
    "LCOH: Moderate-no_policy-BuriedPipes",
    "LCOH: Moderate-max_policy-BuriedPipes",
    "LCOH: Conservative-no_policy-BuriedPipes",
    "LCOH: Conservative-max_policy-BuriedPipes",
]

df = sites_ia.drop(omit_columns, axis=1)

uppers = df.max(axis=0)
lowers = df.min(axis=0)

for col in df.columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

data = df.to_numpy()

errors = []
# for clusters in np.arange(1, 100, 4):
for clusters in [20]:
    kmeans = KMeans(n_clusters=clusters).fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    error = 0
    for i in range(len(labels)):
        error += np.linalg.norm((data[i, :] - centers[labels[i]]))

    errors.append(error)

    print(error)

fig, ax = plt.subplots(1, 1)
ax.plot(kmeans.cluster_centers_.T)
ax.set_xticks(np.arange(0, len(df.columns), 1), df.columns, rotation=90)
fig.tight_layout()


def uninterp(vec, high, low):
    return vec * high + (1 - vec) * low


reps = [uninterp(vec, uppers.values, lowers.values) for vec in centers]

rep_plants = pd.DataFrame(reps, columns=df.columns)


[]
