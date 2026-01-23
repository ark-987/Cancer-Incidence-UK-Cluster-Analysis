import pandas as pd
import yaml
import pytest
from src.data.save_preprocessed_data import save_preprocessed_data as save_df_with_config


def test_save_df_with_dataset_key(tmp_path):
    # Sample DataFrame
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4]
    })

    #Config file with dataset keys
    config_path = tmp_path / "config.yaml"
    config_data = {
        "processed_data": {
            "incidence_by_stage": "data/incidence.csv"
        }
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # Call function with dataset_key
    output_path = save_df_with_config(
        df,
        config_path,
        dataset_key="incidence_by_stage"
    )

    # Assertions
    assert output_path.exists()

    df_loaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(df, df_loaded)