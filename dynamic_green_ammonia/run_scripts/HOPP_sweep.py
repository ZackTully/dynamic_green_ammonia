import numpy as np
import yaml
from pathlib import Path
import os, shutil
import pprint
import time
import matplotlib.pyplot as plt
import pandas as pd

from dynamic_green_ammonia.technologies.Run_DL import RunDL, FlexibilityParameters


# save_path = Path(__file__).parents[1] / "data" / "heatmap_runs"
# save_path = Path(__file__).parents[1] / "data" / "LCOA_runs"
save_path = Path(__file__).parents[1] / "data" / "cases_check"

input_path = Path(__file__).parents[1] / "inputs"

generate_HOPP_files = True
run_HOPP = True

if generate_HOPP_files:
    folder = input_path / "HOPP_sweep_inputs"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isidir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print("failed to delete %s. Reason: %s" % (file_path, e))

    def update_dict(base_dict, year, note, loc, elev, hybrid_rating, split):
        save_name = f"year{year}_lat{loc[0]}_lon{loc[1]}_split0p{split*100:.0f}_{note}"

        wind_rating = split * hybrid_rating
        num_turbines = np.round(wind_rating / 5e3)
        turbine_rating = wind_rating / num_turbines
        pv_rating = (1 - split) * hybrid_rating

        save_dict = base_dict.copy()
        save_dict.update({"name": save_name})
        save_dict["site"]["data"].update(
            {
                "lat": float(loc[0]),
                "lon": float(loc[1]),
                "year": year,
                "elev": float(elev),
            }
        )
        save_dict["technologies"]["pv"].update({"system_capacity_kw": int(pv_rating)})
        save_dict["technologies"]["wind"].update(
            {
                "num_turbines": int(num_turbines),
                "turbine_rating_kw": float(turbine_rating),
            }
        )
        return save_dict, save_name

    # generate HOPP input files for sweep

    loc_info = pd.read_csv(input_path / "location_info.csv")

    data_path = Path(__file__).parent / "data" / "HOPP_sweep"

    run_type = str(save_path).split("/")[-1]

    if "cases" in run_type:
        print("Analysis type: CHECK CASES")

        ramp_lims = [0, 0.2]
        turndowns = [1, 0.6]
        rltd = [(ramp_lims[i], turndowns[i]) for i in range(len(ramp_lims))]

        # custom_index = [5, 7, 8, 9, 10, 11, 13, 14, 15, 16]
        # custom_index = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # run_cases = loc_info.loc[custom_index]

        run_cases = loc_info[loc_info["loc"].isin(["TX", "IA"])]

    else:
        if "heatmap" in run_type:
            print("Analysis type: HEATMAP")

            low = np.log10(1 / 8760)
            highest = np.log10(0.9)
            ramp_lims = np.logspace(low, -low - (-low - np.log10(0.2)) / 6 * 6, 8)
            turndowns = np.linspace(0.1, 0.9, 9)

            # Always include inflexible case and fully flexible case
            ramp_lims = np.concatenate([[0], ramp_lims, [1]])
            turndowns = np.concatenate([[0], turndowns, [1]])

            custom_index = [10, 16]
            run_cases = loc_info.loc[custom_index]

        elif "LCOA" in run_type:
            print("Analysis type: LCOA")

            ramp_lims = [0.2]
            turndowns = np.linspace(0, 1, 20)

        rltd = []  # tuples of (rl, td)
        for rl in ramp_lims:
            for td in turndowns:
                rltd.append((rl, td))

        # custom_index = [0, 5, 11, 10, 16]
        custom_index = [10, 16]
        run_cases = loc_info.loc[custom_index]

    # years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]
    years = [2012]

    # loc_type = [
    #     "green steel sites",
    #     "Max CF",
    #     "Min LCOH",
    #     "Min storage",
    #     "Complimentarity",
    #     "CF and storage",
    # ]
    # state_index = loc_info["loc"].isin(["IN", "TX", "IA"])

    locations = run_cases[["lat", "lon"]].to_numpy()
    elevations = run_cases["elevation"].to_numpy()

    hybrid_rating = 1e6  # [kW] 1 GW plant
    wind_pv_split = [0.5]

    base_file = open("dynamic_green_ammonia/inputs/BASEFILE_hopp_input.yaml", "r")
    base_dict = yaml.load(base_file, yaml.Loader)

    for year in years:
        for i in range(locations.shape[0]):
            for split in wind_pv_split:
                save_dict, save_name = update_dict(
                    base_dict,
                    year,
                    run_cases.iloc[i]["note"],
                    locations[i, :],
                    elevations[i],
                    hybrid_rating,
                    split,
                )
                save_file = open(
                    "".join(
                        [
                            "dynamic_green_ammonia/inputs/HOPP_sweep_inputs/",
                            save_name,
                            ".yaml",
                        ]
                    ),
                    "w",
                )
                yaml.dump(save_dict, save_file)

if run_HOPP:

    def progress(start, prev, curr, count, n_runs, message):
        print(message)
        print(f"completed {count} out of {n_runs} runs, took {curr- prev:.2f} seconds.")
        print(
            f"Predicting {(curr - start) * (n_runs / count) - (curr - start):.2f} more seconds."
        )
        print(
            "=======================================================================\n"
        )
        return curr

    hopp_files = os.listdir("dynamic_green_ammonia/inputs/HOPP_sweep_inputs")
    H2_profiles = []
    wind_profiles = []
    solar_profiles = []
    save_df = []
    dfs = []

    n_runs = len(hopp_files) * len(rltd)

    print(f"{n_runs} runs planned.")
    print(f"Save path is: {save_path}")
    # input("Press enter in terminal to continue")

    count = 1

    start = time.time()
    prev = start

    for i, hopp_file in enumerate(hopp_files):
        year = int(hopp_file.split("_")[0][4:])
        lat = float(hopp_file.split("_")[1][3:])
        lon = float(hopp_file.split("_")[2][3:])
        split = int(hopp_file.split("_")[3][7:])  # .split(".")[0][7:])
        hopp_input = (
            Path(__file__).parents[1] / "inputs" / "HOPP_sweep_inputs" / hopp_file
        )
        DL = RunDL(hopp_input, 0.5, 0.5)
        DL.re_init()
        DL.write_hopp_main_dict()

        DL.P2EL, DL.P2ASU, DL.P2HB = DL.split_power()
        DL.P_gen = DL.P2ASU + DL.P2HB
        DL.H2_gen = DL.calc_H2_gen()

        H2_profiles.append(DL.H2_gen)
        wind_profiles.append(DL.wind.generation_profile[0:8760])
        solar_profiles.append(DL.pv.generation_profile[0:8760])

        for flex_params in rltd:
            rl = flex_params[0]
            td = flex_params[1]

            DL.run(rl, td)
            dfs.append(DL.main_df.copy())

            df_dict = {
                "year": year,
                "lat": lat,
                "lon": lon,
                "split": split,
                "rl": rl,
                "td": td,
                "gen_ind": i,
                "hopp_input": hopp_file,
                "storage_cap_kg": DL.H2_capacity,
                "storage_state": [DL.H2_storage_state],
            }

            save_df.append(pd.DataFrame(df_dict))

            message = f"{hopp_file}, ramp lim: {rl:.4f}, turndown: {td:.2f}"
            prev = progress(start, prev, time.time(), count, n_runs, message)
            count += 1

    df = pd.concat(save_df)
    df.to_pickle(save_path / "hopp_sweep.pkl")

    main_df = pd.concat(dfs)
    main_df.to_csv(save_path / "full_sweep_main_df.csv")

    H2_profiles = np.stack(H2_profiles)
    wind_profiles = np.stack(wind_profiles)
    solar_profiles = np.stack(solar_profiles)
    np.save(save_path / "H2_gen.npy", H2_profiles)
    np.save(save_path / "wind_gen.npy", wind_profiles)
    np.save(save_path / "solar_gen.npy", solar_profiles)

    []
