import yaml
from pathlib import Path

# Load config.yaml
CANCER_STATS_PROJECT = Path.cwd().parent 
CONFIG_PATH = CANCER_STATS_PROJECT/"config"/"config.yaml" 


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

raw_csv_path = (CANCER_STATS_PROJECT/config["raw_data"]["incidence_by_stage"]).resolve()

#All data and output locations are managed through config/data_paths.yaml:

print("Using raw data file:", raw_csv_path)

import pandas as pd

df = pd.read_csv(raw_csv_path)
print(df.head())
print(df.info())
print(df.describe())