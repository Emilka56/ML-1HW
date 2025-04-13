

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from itertools import combinations

def find_optimal_clusters(data, max_k=15):
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # Преобразуем список в массив NumPy
    sse = np.array(sse)

    # Нормализуем SSE
    normalized_sse = (sse - min(sse)) / (max(sse) - min(sse))

    # Ищем точку, где наклон становится менее 0.05
    slopes = np.diff(normalized_sse)
    optimal_k = np.where(slopes > -0.05)[0][0] + 2 if len(np.where(slopes > -0.05)[0]) > 0 else np.argmin(
        np.abs(slopes)) + 1

    # Визуализация
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--',
                label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()

    return optimal_k

def distance(dot1, dot2):
    return np.sqrt(np.sum((dot1 - dot2) ** 2))


def find_centroids(data, num_of_clusters):
    centroids_indexes = np.random.choice(data.shape[0], num_of_clusters, replace=False)
    return data[centroids_indexes]


def visualisation(data, centroids, clusters, step):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=50, label='Точки')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Центроиды')
    plt.title(f'Шаг {step} (2D проекция)')
    plt.xlabel('Признак 1 (Длина чашелистика)')
    plt.ylabel('Признак 2 (Ширина чашелистика)')
    plt.legend()
    plt.show()


def update_centroids(data, clusters, num_of_centroids, old_centroids):
    centroids = []
    for i in range(num_of_centroids):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = old_centroids[i]
        centroids.append(new_centroid)

    return np.array(centroids)


def k_means(data, num_of_clusters, max_step=100):
    centroids = find_centroids(data, num_of_clusters)
    for i in range(max_step):
        clusters = []
        for point in data:
            distances = [distance(point, centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)

        visualisation(data, centroids, clusters, i)

        new_centroid = update_centroids(data, np.array(clusters), num_of_clusters, centroids)
        if np.all(centroids == new_centroid):
            print(f"алгоритм закончил работу на {i} шаге")
            break
        centroids = new_centroid

    return centroids, clusters


def plot_all_projections(data, clusters, centroids, feature_names):
    features = range(data.shape[1])

    for x_index, y_index in combinations(features, 2):
        plt.scatter(data[:, x_index], data[:, y_index], c=clusters, cmap='viridis', s=50, label='Точки')
        plt.scatter(centroids[:, x_index], centroids[:, y_index], c='red', marker='X', s=200, label='Центроиды')
        plt.title(f' проекция {feature_names[x_index]} vs {feature_names[y_index]}')
        plt.xlabel(feature_names[x_index])
        plt.ylabel(feature_names[y_index])
        plt.legend()
        plt.show()

def main():
    # Загрузка данных
    irises = load_iris()
    data = irises.data

    # Определение оптимального количества кластеров
    optimal_k = find_optimal_clusters(data)
    print(f"Оптимальное количество кластеров: {optimal_k}")

    centroids, clusters = k_means(data, optimal_k)
    feature_names = irises['feature_names']
    plot_all_projections(data, clusters, centroids, feature_names)


if __name__ == "__main__":
    main()