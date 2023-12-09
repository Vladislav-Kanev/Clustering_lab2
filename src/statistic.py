import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import rand_score, silhouette_score


def _process_dict(clasterisations):
    processed_clasterizartions = dict()
    for i in clasterisations:
        if (isinstance(clasterisations[i], dict)):
            for j in clasterisations[i]:
                key = f'{i}_{j}'
                processed_clasterizartions[key] = clasterisations[i][j]
        else:
            processed_clasterizartions[i] = clasterisations[i]

    return processed_clasterizartions


def calculate_rand_index(clasterisations, rand_function=rand_score):
    processed_clasterizartions = _process_dict(clasterisations)

    statistic = pd.DataFrame(
        columns=processed_clasterizartions.keys(), index=processed_clasterizartions.keys())
    for i in processed_clasterizartions.keys():
        for j in processed_clasterizartions.keys():
            statistic[i][j] = rand_function(
                processed_clasterizartions[i], processed_clasterizartions[j])

    return statistic


def _get_sets_of_classes(clusterization):
    partition = [set() for _ in range(len(np.unique(clusterization)))]
    for i in range(len(clusterization)):
        partition[clusterization[i]].add(i)
    return partition


def calculate_modularity(singularity_graphs, clusterizations, default_distance='euclidean'):
    processed_clasterizartions = dict()

    for i in clusterizations:
        if (isinstance(clusterizations[i], dict)):
            for j in clusterizations[i]:
                key = f'{i}_{j}'
                partition = _get_sets_of_classes(clusterizations[i][j])

                processed_clasterizartions[key] = nx.community.modularity(
                    singularity_graphs[j], partition)
        else:
            partition = _get_sets_of_classes(clusterizations[i])
            processed_clasterizartions[i] = nx.community.modularity(
                singularity_graphs[default_distance], partition)
    return pd.DataFrame(processed_clasterizartions.values(), index=processed_clasterizartions.keys(), columns=['modularity'])


def calculate_silhouette(data, clusterizations):
    processed_clasterizartions = _process_dict(clusterizations)

    silhouette = {}
    for method in processed_clasterizartions.keys():
        silhouette[method] = silhouette_score(
            data, processed_clasterizartions[method])

    score_df = pd.DataFrame.from_dict(silhouette, orient='index')
    score_df.columns = ['Silhouette score']
    return score_df
