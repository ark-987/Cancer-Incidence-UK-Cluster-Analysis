import pytest
import pandas as pd
from src.features.transformations import(
    convert_to_datetime,
    aggregate_stages,
    create_proportions,
    transform_stage_data,
)

@pytest.fixture
def sample_data():
        data = {
            'Date': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01',
                     '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01'],
            'Cancer Site': ['Lung', 'Lung', 'Lung', 'Lung',
                            'Breast', 'Breast', 'Breast', 'Breast'],
            'Stage': ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
                      'Stage I', 'Stage II', 'Stage III', 'Stage IV'],
            'Observed Count': [50, 30, 15, 5, 80, 10, 5, 5]
        }
        return pd.DataFrame(data)

def test_convert_to_datetime(sample_data):
        df_converted = convert_to_datetime(sample_data)
        assert 'Year' in df_converted.columns, "Year column not created"
        assert 'Month' in df_converted.columns, "Month column not created"
        assert df_converted.shape[0]==2, "Rows with valid dates not retained"
        assert pd.api.types.is_datetime64_any_dtype(df_converted['Date']), "Date column is not datetime type"

def test_aggregate_stages(sample_data):
        df_converted = convert_to_datetime(sample_data)
        df_grouped = aggregate_stages(df_converted)
        # Structure checks
        assert isinstance(df_grouped.index, pd.MultiIndex)
        assert 'Stage I' in df_grouped.columns

        # Correctness checks
        assert df_grouped.loc[(2020, 'Lung'), 'Stage I'] == 50
        assert df_grouped.loc[(2020, 'Breast'), 'Stage II'] == 10

def test_create_proportions(sample_data):
        df_converted = convert_to_datetime(sample_data)
        df_grouped = aggregate_stages(df_converted)
        df_proportions = create_proportions(df_grouped)

        # Structure checks
        assert isinstance(df_proportions.index, pd.MultiIndex)
        assert 'Stage I' in df_proportions.columns
          # All proportions between 0 and 1
        assert (df_proportions >= 0).all().all()
        assert (df_proportions <= 1).all().all()


# end-end pipeline sanity check

def test_transform_stage_data(sample_data):
        df_transformed = transform_stage_data(sample_data)

        assert isinstance(df_transformed, pd.DataFrame)
        assert not df_transformed.empty   