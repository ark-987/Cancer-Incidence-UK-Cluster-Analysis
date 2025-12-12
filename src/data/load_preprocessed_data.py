import yaml
from pathlib import Path
import pandas as pd

def load_preprocessed_data():
    """
    Loads the preprocessed incidence-by-stage CSV using the path defined in config/config.yaml.
    """

    current = Path(__file__).resolve()

    # Walk upward to find project root
    for parent in current.parents:
        if (parent / "config" / "config.yaml").exists():
            project_root = parent
            break
    else:
        raise FileNotFoundError("config.yaml not found in any parent directory")

    # These MUST be defined before use
    CANCER_STATS_PROJECT = project_root
    CONFIG_PATH = project_root / "config" / "config.yaml"

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Correct path based on your config.yaml
    csv_rel_path = config["processed_data"]["incidence_by_stage"]

    # Build absolute CSV path
    preprocessed_csv_path = (CANCER_STATS_PROJECT / csv_rel_path).resolve()

    print("Loading preprocessed data file:", preprocessed_csv_path)

    # Load CSV
    df = pd.read_csv(preprocessed_csv_path)

    return df


