import numpy as np
import matplotlib.pyplot as plt

def draw_clusters(data, clusterizations, title=None):
    number_of_clusters = len(np.unique(clusterizations))
    colormap = plt.cm.get_cmap('winter', number_of_clusters)
    color = [colormap(value) for value in clusterizations]

    fig, ax = plt.subplots(4, 3, figsize=(24, 14))
    fig.suptitle(title)
    i = 0
    j = 0
    for column_x1 in data.columns:
        for column_x2 in data.columns:
            if column_x1 != column_x2:
                ax[i, j].scatter(data[column_x2], data[column_x1], c = color)
                if i == 0:
                    ax[i, j].set_title(f'column {column_x2}', fontweight='bold')
                if j == 0:
                    ax[i, j].set_ylabel(f'column {column_x1}', fontweight='bold')
                j += 1
                if j == 3:
                    j = 0
                    i+= 1

    plt.show()