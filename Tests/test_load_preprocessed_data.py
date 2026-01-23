import pytest
from load_preprocessed_data import load_preprocessed_data
import pandas as pd 

def test_load_preprocessed_data():
    # Call the function to load data
    df = load_preprocessed_data()
    
    # Assert that the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame), "The loaded data is not a pandas DataFrame"
    
    # Assert that the DataFrame is not empty
    assert not df.empty, "The loaded DataFrame is empty"
    
    # Optionally, check for expected columns (replace with actual expected columns)
    expected_columns = ['Stage', 'Incidence', 'Year']  # Example column names
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in DataFrame"
  
