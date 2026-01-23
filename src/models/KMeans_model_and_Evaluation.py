from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def prepare_data_for_clustering(df, cluster_columns):
    """
    Extracts cluster columns and validates them.
    """
    if cluster_columns is None:
        cluster_columns = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    
    missing_cols = [col for col in cluster_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    X = df[cluster_columns].copy()
    return X


def fit_kmeans(X, n_clusters=2, random_state=42):
    """
    Fits KMeans and returns labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def compute_silhouette_score(X, labels):
    """
    Computes the silhouette score for given clustering.
    """
    return silhouette_score(X, labels) #The silhouette score measures how well the clusters are separated.If points are close to their own cluster and far from other clusters, score is high → near 1.
                                        #If points are on the boundary or wrongly assigned, score is low → near 0.
                                        #If clusters are overlapping badly, score can even be negative → near -1.


def cluster_stage_data(df, cluster_columns=None, n_clusters=2, return_score=True):
    """
    Orchestrates clustering and optional scoring.
    """
    X = prepare_data_for_clustering(df, cluster_columns)
    _, labels = fit_kmeans(X, n_clusters=n_clusters)
    
    df_copy = df.copy()
    df_copy['Cluster'] = labels

    score = None
    if return_score:
        score = compute_silhouette_score(X, labels)
    
    return (df_copy, score) if return_score else df_copy


def evaluate_k_values(df, cluster_columns=None, min_k=2, max_k=6):
    """
    Evaluate silhouette scores for multiple k values.
    """
    X = prepare_data_for_clustering(df, cluster_columns)
    scores = {}
    
    for k in range(min_k, max_k):
        _, labels = fit_kmeans(X, n_clusters=k)
        scores[k] = compute_silhouette_score(X, labels)
    
    return scores





