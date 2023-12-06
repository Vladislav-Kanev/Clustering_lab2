import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def hierarchical_clustering(data, distance_matrix, clusters=2, method='single'):
    """
    Perform hierarchical clustering with 2 clusters and plot the result.

    Parameters:
    - df: pandas DataFrame.
    - method: The linkage method ('single', 'complete', 'average', 'centroid', 'ward').

    Returns:
    - The clustered DataFrame.
    """

    clusters = AgglomerativeClustering(
        n_clusters=clusters, linkage='single', connectivity=distance_matrix).fit_predict(data)

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
    kmeans = KMeans(n_clusters=clusters,  # ожидаемое число кластеров
                    max_iter=300,  # максимальное число итераций K-means для одного запуска
                    random_state=42,
                    n_init="auto")

    # Fit the model to the data
    clusters_list = kmeans.fit_predict(np.array(df))

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
    gmm = GaussianMixture(n_components=clusters,
                          covariance_type="full",  # тип используемых ковариационных параметров
                          max_iter=300,
                          init_params="random_from_data",
                          random_state=42)

    # Fit the model to the data
    clusters_list = gmm.fit_predict(np.array(df))

    return clusters_list


def spectral_clustering(distance_matrix, clusters=2):
    """
    Perform Spectral Clustering clustering

    Parameters:
    - df: pandas DataFrame.

    Returns:

    Returns:
    - The clustered DataFrame.
    """
    # Instantiate the sc model
    sc = SpectralClustering(n_components=clusters,  # ожидаемое число кластеров
                            assign_labels='cluster_qr',  # стратегия назначения меток элементам в пространстве
                            affinity="precomputed",  # интерпретировать ли входные данные как заранее вычисленную матрицу сходства, где большие значения указывают на большее сходство между экземплярами
                            # случайное число для инициализации центроидов (centroid)
                            random_state=42,
                            )
    # Fit the model to the data
    clusters_list = sc.fit_predict(np.array(distance_matrix))

    return clusters_list
