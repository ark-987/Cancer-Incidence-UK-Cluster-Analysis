from pathlib import Path
import yaml

config_file_pathg=r"C:\Users\arkha\cancer_stats_project\config\config.yaml"
df="df_stage_props"
dataset_key="incidence_by_stage"

def save_preprocessed_data(df, config_file_path, dataset_key):
    """
    Save a preprocessed dataframe to CSV using path from config.yaml.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed dataframe to save.
    config_file_path : str or Path
        Path to the config.yaml file.
    dataset_key : str
        Key inside config["processed_data"] pointing to the CSV path.
    """
    # make config file path is Path object
    config_file_path = Path(config_file_path)

    # Load config
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Build absolute path to save CSV
    project_root = config_file_path.parent.parent  # config is in <project_root>/config/
    output_path = (project_root / config["processed_data"][dataset_key]).resolve()

    # parent folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataframe
    df.to_csv(output_path, index=False)
    print("Saved preprocessed data to:", output_path)

