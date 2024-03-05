from pathlib import Path
import pandas as pd
import numpy as np

class FileMan:
    def __init__(self):

        root = Path(__file__).parents[1]

        self.heatmap_path = root / "data" / "heatmap_runs"
        self.LCOA_path = root / "data" / "LCOA_runs"
        self.cases_path = root / "data" / "cases_check"

        self.costs_path = root / "analysis_scripts" / "cost_models"

        self.input_path = root / "inputs"
        self.plot_path = root / "plots"

    def set_analysis_case(self, analysis_case):
        if analysis_case == "heat":
            self.data_path = self.heatmap_path
        elif analysis_case == "LCOA":
            self.data_path = self.LCOA_path
        elif analysis_case == "cases_check":
            self.data_path = self.cases_path

    def load_sweep_data(self):
        df_all = pd.read_pickle(self.data_path / "hopp_sweep.pkl")
        df_full = pd.read_csv(self.data_path / "full_sweep_main_df.csv")
        return df_all, df_full

    def load_gen_data(self):
        gp = np.load(self.data_path / "H2_gen.npy")
        wp = np.load(self.data_path / "wind_gen.npy")
        sp = np.load(self.data_path / "solar_gen.npy")
        return gp, wp, sp


class LocMan:
    def __init__(self):
        self.FM = FileMan
        self.loc_info = pd.read_csv(self.FM.input_path / "location_info.csv")