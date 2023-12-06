import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def similarity_matrix_to_graph(similarity_matrix, threshold=0.5):
    """
    Convert a similarity matrix to a graph by thresholding the similarity values.

    Parameters:
    - similarity_matrix: pandas dataframe
    - threshold: The threshold value to create edges in the graph.

    Returns:
    - A NetworkX graph object.
    """
    G = nx.Graph()
    num_nodes = len(similarity_matrix)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    return G


def plot_graph(graph):
    """
    Plot the given graph using Matplotlib.

    Parameters:
    - graph: NetworkX graph object.
    """
    plt.figure(figsize=(10, 10))
    # You can use other layout algorithms based on your preference
    pos = nx.spring_layout(graph)

    # построение графа, pos - позиции вершин графа, width - ширина ребра
    nx.draw_networkx(graph, pos, width=1, with_labels=True)
    plt.show()


def get_graph_from_clasterisation(clasterisation):
    G = nx.Graph()

    for c in np.unique(clasterisation):
        nodes_in_class = np.where(clasterisation == c)[0]
        G.add_nodes_from(nodes_in_class)

    for c in np.unique(clasterisation):
        nodes_in_class = np.where(clasterisation == c)[0]
        edges = [(nodes_in_class[i], nodes_in_class[j]) for i in range(len(nodes_in_class))
                 for j in range(i + 1, len(nodes_in_class))]
        G.add_edges_from(edges)

    return G
