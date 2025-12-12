from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def perform_pca(X, n_components=2, random_state=42, return_model=False):
    """
    Perform PCA dimensionality reduction on a numeric feature matrix.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        The input feature matrix. Must contain only numeric columns.
    
    n_components : int, default=2
        Number of principal components to compute.
    
    random_state : int, default=42
        Random state for reproducibility.
    
    return_model : bool, default=False
        If True, return the fitted PCA model along with the transformed data.

    Returns
    -------
    X_pca : numpy.ndarray
        Array of shape (n_samples, n_components) containing the PCA-transformed data.
    
    pca_model : PCA object (optional)
        Returned only if return_model=True.
    """

    # Convert dataframe to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_numeric = X.values
    else:
        X_numeric = X

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_numeric)

    # Either return just the result, or result + model
    if return_model:
        return X_pca, pca
    else:
        return X_pca

