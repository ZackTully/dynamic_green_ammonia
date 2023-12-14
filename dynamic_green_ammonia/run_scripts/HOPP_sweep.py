import numpy as np
import yaml
import pprint


def update_dict(base_dict, year, loc, hybrid_rating, split):
    save_name = f"year{year}_lat{loc[0]}_lon{loc[1]}_split0p{split*100:.0f}"

    wind_rating = split * hybrid_rating
    num_turbines = np.round(wind_rating / 5e3)
    turbine_rating = wind_rating / num_turbines

    pv_rating = (1 - split) * hybrid_rating

    save_dict = base_dict.copy()

    save_dict.update({"name": save_name})
    save_dict["site"]["data"].update({"lat": loc[0], "lon": loc[1], "year": year})
    save_dict["technologies"]["pv"].update({"system_capacity_kw": pv_rating})
    save_dict["technologies"]["wind"].update(
        {
            "num_turbines": int(num_turbines),
            "turbine_rating_k": float(turbine_rating),
        }
    )

    return save_dict, save_name


# generate HOPP input files for sweep

years = [2012]
locations = [[34.22, -102.75], [41.62, -87.22]]
hybrid_rating = 1e6  # [kW]
wind_pv_split = [0.5]

base_file = open("dynamic_green_ammonia/inputs/BASEFILE_hopp_input.yaml", "r")
base_dict = yaml.load(base_file, yaml.Loader)

# save_file = open("dynamic_green_ammonia/inputs/HOPP_sweep_inputs/test.yaml", "w")
# save_dict = base_dict.copy()
# save_dict, save_name = update_dict(
# base_dict, years[0], locations[0], hybrid_rating, wind_pv_split[0]
# )
# save_dict.update({"name": "Save test"})
# yaml.dump(save_dict, save_file)


# read_file = open("dynamic_green_ammonia/inputs/HOPP_sweep_inputs/test.yaml", "r")
# pprint.pprint(yaml.load(read_file, yaml.Loader))

for year in years:
    for location in locations:
        for split in wind_pv_split:
            save_dict, save_name = update_dict(
                base_dict, year, location, hybrid_rating, split
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
            []


[]
