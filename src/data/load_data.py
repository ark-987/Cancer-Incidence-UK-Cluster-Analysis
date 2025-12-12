import yaml
from pathlib import Path
import pandas as pd


def load_raw_data():
    """
    Loads the raw incidence-by-stage CSV using the path defined in config/config.yaml.

    Returns
    -------
    df : pandas.DataFrame
        The raw dataset loaded into a dataframe.
    """

    # Locate project root (one level above /src)
    CANCER_STATS_PROJECT = Path(__file__).resolve().parents[2]

    # Path to config.yaml
    CONFIG_PATH = CANCER_STATS_PROJECT / "config" / "config.yaml"

    # Load config file
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Get CSV path from config
    raw_csv_path = (CANCER_STATS_PROJECT / config["raw_data"]["incidence_by_stage"]).resolve()

    print("Loading raw data file:", raw_csv_path)

    # Load CSV into DataFrame
    df = pd.read_csv(raw_csv_path)

    return df


