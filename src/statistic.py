import pandas as pd
from sklearn.metrics import rand_score


def calculate_rand_index(clasterisations, rand_function=rand_score):
    processed_clast = dict()
    for i in clasterisations:
        if (isinstance(clasterisations[i], dict)):
            for j in clasterisations[i]:
                key = f'{i}_{j}'
                processed_clast[key] = clasterisations[i][j]
        else:
            processed_clast[i] = clasterisations[i]

    statistic = pd.DataFrame(
        columns=processed_clast.keys(), index=processed_clast.keys())
    for i in processed_clast.keys():
        for j in processed_clast.keys():
            statistic[i][j] = rand_function(
                processed_clast[i], processed_clast[j])

    return statistic
