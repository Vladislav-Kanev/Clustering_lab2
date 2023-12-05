from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


def hierarchical_clustering(distance_matrix, clusters=2, method='single'):
    """
    Perform hierarchical clustering with 2 clusters and plot the result.

    Parameters:
    - df: pandas DataFrame.
    - method: The linkage method ('single', 'complete', 'average', 'centroid', 'ward').

    Returns:
    - The clustered DataFrame.
    """

    condensed_distance = squareform(distance_matrix)

    linkage_matrix = linkage(condensed_distance, method=method)

    clusters = fcluster(linkage_matrix, clusters, criterion='maxclust')

    return clusters


def kmeans_clustering(df, clusters=2):
    """
    Perform K-means clustering

    Parameters:
    - df: pandas DataFrame.

    Returns:
    - The clustered DataFrame.
    """
    # Instantiate the KMeans model
    kmeans = KMeans(n_clusters=clusters, n_init='auto', random_state=42)

    # Fit the model to the data
    clusters_list = kmeans.fit_predict(df)

    return clusters_list


def em_clustering(df, clusters=2):
    """
    Perform EM clustering

    Parameters:
    - df: pandas DataFrame.

    Returns:

    Returns:
    - The clustered DataFrame.
    """
    # Instantiate the em model
    gmm = GaussianMixture(n_components=clusters, random_state=42)

    # Fit the model to the data
    clusters_list = gmm.fit_predict(df)

    return clusters_list


def spectral_clustering(df, clusters=2):
    """
    Perform Spectral Clustering clustering

    Parameters:
    - df: pandas DataFrame.

    Returns:

    Returns:
    - The clustered DataFrame.
    """
    # Instantiate the sc model
    sc = SpectralClustering(n_clusters=clusters,
                            affinity='precomputed', random_state=42)

    # Fit the model to the data
    clusters_list = sc.fit_predict(df)

    return clusters_list
