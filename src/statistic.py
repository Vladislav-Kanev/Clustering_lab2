import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import rand_score


def _process_dict(clasterisations):
    processed_clast = dict()
    for i in clasterisations:
        if (isinstance(clasterisations[i], dict)):
            for j in clasterisations[i]:
                key = f'{i}_{j}'
                processed_clast[key] = clasterisations[i][j]
        else:
            processed_clast[i] = clasterisations[i]

    return processed_clast


def calculate_rand_index(clasterisations, rand_function=rand_score):
    processed_clast = _process_dict(clasterisations)

    statistic = pd.DataFrame(
        columns=processed_clast.keys(), index=processed_clast.keys())
    for i in processed_clast.keys():
        for j in processed_clast.keys():
            statistic[i][j] = rand_function(
                processed_clast[i], processed_clast[j])

    return statistic


def _get_sets_of_classes(clusterization):
    partition = [set() for _ in range(len(np.unique(clusterization)))]
    for i in range(len(clusterization)):
        partition[clusterization[i]].add(i)
    return partition


def calculate_modularity(singularity_graphs, clusterizations, default_distance='euclidean'):
    processed_clast = dict()

    for i in clusterizations:
        if (isinstance(clusterizations[i], dict)):
            for j in clusterizations[i]:
                key = f'{i}_{j}'
                partition = _get_sets_of_classes(clusterizations[i][j])

                processed_clast[key] = nx.community.modularity(
                    singularity_graphs[j], partition)
        else:
            partition = _get_sets_of_classes(clusterizations[i])
            processed_clast[i] = nx.community.modularity(
                singularity_graphs[default_distance], partition)
    return pd.DataFrame(processed_clast.values(), index=processed_clast.keys(), columns=['modularity'])
