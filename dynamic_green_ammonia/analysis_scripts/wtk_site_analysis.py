import pandas as pd
from pathlib import Path


wtk = pd.read_csv(Path(__file__).parents[1] / "data/wtk_metadata/wtk_site_metadata.csv")

[]