import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pca_clusters(pca_result, labels, names, title="KMeans Clusters (2D PCA Projection)"):
    """
    Plot a 2D PCA projection with cluster colours and centroid labels.

    Parameters
    ----------
    pca_result : numpy.ndarray
        Array of shape (n_samples, 2) containing 2D PCA coordinates.
    
    labels : array-like
        Cluster labels for each sample (e.g., output from KMeans).
    
    names : array-like
        Cancer site names corresponding to each row in pca_result.
        Typically: df.index.get_level_values('Cancer Site')
    
    title : str, default="KMeans Clusters (2D PCA Projection)"
        Title of the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    pca_result = np.asarray(pca_result)
    labels = np.asarray(labels)
    names = np.asarray(names)

    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of PCA points
    scatter = ax.scatter(
        pca_result[:, 0], 
        pca_result[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.7
    )

    # Loop through each cluster to compute centroids and label them
    for label in unique_labels:
        cluster_points = pca_result[labels == label]
        cluster_names = names[labels == label]

        # Compute centroid
        centroid = cluster_points.mean(axis=0)

        # Most common 3 cancer sites in this cluster
        name_counts = pd.Series(cluster_names).value_counts()
        top_names = name_counts.head(3).index.tolist()
        top_names_str = ", ".join(top_names)

        # Plot centroid
        ax.scatter(centroid[0], centroid[1], s=200, c='red', marker='X')

        # Label centroid
        ax.text(
            centroid[0], centroid[1],
            f"Cluster {label}\n{top_names_str}",
            fontsize=10,
            fontweight='bold',
            color='red',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.6)
        )

    # Add legend, labels, title
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return fig, ax
