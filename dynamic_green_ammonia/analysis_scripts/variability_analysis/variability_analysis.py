import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin

from dynamic_green_ammonia.tools.file_management import FileMan

FM = FileMan()
FM.set_analysis_case("heat")



wind_gen = np.load(FM.data_path / "wind_gen.npy")
solar_gen = np.load(FM.data_path / "solar_gen.npy")
H2_gen = np.load(FM.data_path / "H2_gen.npy")
hybrid_gen = wind_gen + solar_gen

df_all = pd.read_csv(FM.data_path / "full_sweep_main_df.csv")


td = .5
d_max = 10
d_min = .1

profile = hybrid_gen[1,:]

sorted = np.sort(profile)

max_gen = np.max(profile)

n = 100
perc_in = np.zeros(n)

for i, d_max in enumerate(np.linspace(0.01 * max_gen, max_gen, n)):
    d_min = td * d_max
    perc_in[i] = np.sum(profile[np.argwhere((profile > d_min) & (profile < d_max))])


def calc_percent_in(d_max, td):
    
    d_min = td*d_max
    sum_in = np.sum(profile[np.argwhere((profile > d_min) & (profile < d_max))])
    total = np.sum(profile)
    return 1-(sum_in/total)

frac_out_opt = np.zeros(n)
frac_out_heur = np.zeros(n)
d_maxs_opt = np.zeros(n)
d_maxs_opt[-1] = 1e5
d_maxs_heur = np.zeros(n)
TD = np.linspace(0, 1, n)

for i, td in enumerate(TD):
    


    A = np.array([[1, -np.max(profile)], [1, -np.mean(profile)]])
    b = np.array([0, np.mean(profile)])
    coeffs = np.linalg.inv(A) @ b
    max_demand = coeffs[0] / (td + coeffs[1])
    min_demand = td * coeffs[0] / (td + coeffs[1])

    d_maxs_heur[i] = max_demand
    frac_out_heur[i] = calc_percent_in(max_demand, td)


    # d_maxs_opt[i], frac_out_opt[i], _, _, _ =fmin(calc_percent_in, d_maxs_opt[i-1], args=(td,), full_output=True)
    d_maxs_opt[i], frac_out_opt[i], _, _, _ =fmin(calc_percent_in, d_maxs_heur[i], args=(td,), full_output=True)
    
    if d_maxs_opt[i] < np.mean(profile):
        d_maxs_opt[i] = np.mean(profile)
    if td * d_maxs_opt[i] > np.mean(profile):
        d_maxs_opt[i] = np.mean(profile) / td

fig, ax = plt.subplots(2,1)
ax[0].plot(TD, d_maxs_heur)
ax[0].plot(TD, d_maxs_opt)
ax[0].plot(np.linspace(0, 1, len(profile)), sorted)

ax[1].plot(TD, frac_out_heur)
ax[1].plot(TD, frac_out_opt)

[]