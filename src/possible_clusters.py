from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def return_number_of_possible_clusters(data, k):
    model = KMeans(max_iter=300,  # максимальное число итераций K-means для одного запуска
                   random_state=42,
                   n_init="auto")

    visualizer = KElbowVisualizer(model, k=k)
    visualizer.fit(data)
    return visualizer.elbow_value_, visualizer.elbow_score_
