import pandas as pd
import numpy as np
import pytest

from src.models.KMeans_model_and_Evaluation import (
    prepare_data_for_clustering,
    fit_kmeans,
    compute_silhouette_score,
    cluster_stage_data,
    evaluate_k_values,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame with stage proportions."""
    return pd.DataFrame({
        'Stage I': [0.2, 0.1, 0.3, 0.25],
        'Stage II': [0.3, 0.4, 0.2, 0.25],
        'Stage III': [0.3, 0.3, 0.3, 0.25],
        'Stage IV': [0.2, 0.2, 0.2, 0.25],
    })


def test_prepare_data_default_columns(sample_df):
    X = prepare_data_for_clustering(sample_df, None)
    assert list(X.columns) == ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    assert X.shape == (4, 4)


def test_prepare_data_missing_columns_raises(sample_df):
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_data_for_clustering(sample_df, ['Stage I', 'Stage V'])


def test_fit_kmeans_returns_labels_and_model(sample_df):
    X = prepare_data_for_clustering(sample_df, None)
    n_clusters = 5

    model, labels = fit_kmeans(X, n_clusters=n_clusters) #fit_kmeans returns the fitted model object and the cluster labels (labels = which cluster each row belongs to (0 or 1))

    assert model.n_clusters == n_clusters #test that the clusters you got are valid for the chosen n_clusters
    assert len(labels) == len(X)
    assert labels.min() >= 0
    assert labels.max() < n_clusters # Ensure labels are within the correct range, so KMeans labels are integers starting at 0. The largest label must be < n_clusters



def test_fit_kmeans_is_deterministic(sample_df):  #testing that, for the same input and same random state, the same output os obtained 
    X = prepare_data_for_clustering(sample_df, None)
    _, labels1 = fit_kmeans(X, n_clusters=2, random_state=42) #KMeans returns the fitted model object and the cluster labels and we aren't concerned with the model object here, hence _
    _, labels2 = fit_kmeans(X, n_clusters=2, random_state=42)

    np.testing.assert_array_equal(labels1, labels2) #Every label in labels1 must match the label in labels2


def test_compute_silhouette_score_range(sample_df):
    X = prepare_data_for_clustering(sample_df, None)
    _, labels = fit_kmeans(X, n_clusters=2) #this test ignores model object as it is interested in evaluating the cluster assignments only
    score = compute_silhouette_score(X, labels)
    assert -1.0 <= score <= 1.0                 #silhouette score must be between -1 and 1



def test_cluster_stage_data_with_score(sample_df):
    # Call function with return_score=True (default)
    result = cluster_stage_data(sample_df, n_clusters=2)
    
    # Unpack safely
    df_clustered, score = result
    
    # Assertions
    assert isinstance(df_clustered, pd.DataFrame)           # type check
    assert 'Cluster' in df_clustered.columns               # column exists
    assert len(df_clustered) == len(sample_df)             # no rows lost
    assert score is not None                                # score returned
    assert -1.0 <= score <= 1.0                             # valid silhouette range


def test_cluster_stage_data_without_score(sample_df):
    # Call function with return_score=False
    df_clustered = cluster_stage_data(sample_df, n_clusters=2, return_score=False)
    
    # Assertions
    assert isinstance(df_clustered, pd.DataFrame)           # type check
    assert 'Cluster' in df_clustered.columns               # column exists
    assert len(df_clustered) == len(sample_df)             # no rows lost



def test_evaluate_k_values_returns_expected_keys(sample_df):
    scores = evaluate_k_values(sample_df, min_k=2, max_k=5) #range(min_k, max_k) → range(2, 5) → [2, 3, 4]
                                                            #So scores should be a dictionary with keys 2, 3, 4

    assert set(scores.keys()) == {2, 3, 4}          #tests that function returned silhouette scores for the correct cluster numbers.
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_evaluate_k_values_invalid_range(sample_df):
    scores = evaluate_k_values(sample_df, min_k=2, max_k=2)     #range(2, 2) → empty range → the loop never runs
                                                                #So the dictionary scores should be empty {}
    assert scores == {}                                         #tests that function handles edge cases where min_k >= max_k without crashing
