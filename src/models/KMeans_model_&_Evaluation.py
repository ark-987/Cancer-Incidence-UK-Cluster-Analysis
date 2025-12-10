from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_stage_data(df, cluster_columns=None, n_clusters=2, return_score=True):
    """
    Perform KMeans clustering on the specified stage columns and return the dataframe with cluster labels.
    """

    if cluster_columns is None:
        cluster_columns = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

    df_copy = df.copy()
    X = df_copy[cluster_columns].copy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df_copy['Cluster'] = clusters

    score = None
    if return_score:
        score = silhouette_score(X, clusters)

    return (df_copy, score) if return_score else df_copy



def evaluate_k_values(df, cluster_columns=None, min_k=2, max_k=6):
    """
    Test multiple values of k and compute silhouette scores.
    """

    if cluster_columns is None:
        cluster_columns = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

    X = df[cluster_columns].copy()
    scores = {}

    for k in range(min_k, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        scores[k] = silhouette_score(X, labels)

    return scores
