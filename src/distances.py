import numpy as np
import pandas as pd
from scipy.spatial.distance import chebyshev, euclidean, cityblock, cosine


def chebyshev_distances(data: pd.DataFrame):
    distances = pd.DataFrame(index=range(len(data)), columns=range(len(data)))

    for i in range(len(data)):
        for j in range(i, len(data)):
            distances.at[i, j] = distances.at[j, i] = chebyshev(
                data.iloc[i], data.iloc[j])

    return distances


def euclidean_distances(data: pd.DataFrame):
    distances = pd.DataFrame(index=range(len(data)), columns=range(len(data)))

    for i in range(len(data)):
        for j in range(i, len(data)):
            distances.at[i, j] = distances.at[j, i] = euclidean(
                data.iloc[i], data.iloc[j])

    return distances


def cosine_distances(data: pd.DataFrame):
    distances = pd.DataFrame(index=range(len(data)), columns=range(len(data)))

    for i in range(len(data)):
        for j in range(i, len(data)):
            distances.at[i, j] = distances.at[j, i] = cosine(
                data.iloc[i], data.iloc[j])

    return distances


def convert_distance_to_similarity(distance_matrix: pd.DataFrame, sigma:float) -> pd.DataFrame:
    return pd.DataFrame(np.exp(-(distance_matrix.values.astype(float))**2/(2*sigma**2)))
