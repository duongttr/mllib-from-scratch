import numpy as np
import matplotlib.pyplot as plt
from utils.distance import euclidean
from matplotlib.pyplot import scatter, savefig
import matplotlib
from sklearn.cluster import KMeans as sk_KMeans
matplotlib.interactive(True)
import time


class KMeans:
    def __init__(self,
                 n_clusters: int = 9,
                 max_iter: int = 300,
                 tol: float = 1e-4) -> None:
        """

        """
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._centroids = None

    def fit(self, X: np.ndarray) -> None:
        """

        :param X:
        :return:
        """
        # Check constraints
        assert X.ndim == 2, Exception('Dim must be 2')
        # Initialize centroids
        self._centroids = X[np.random.choice(X.shape[0], size=self._n_clusters, replace=False)].astype(np.float32)
        labels = []
        previous_loss = 0
        for _ in range(self._max_iter):
            labels, loss = self.__predict(X)

            for i in range(self._n_clusters):
                related_points_idx = np.where(labels == i)[0]
                related_points = X[related_points_idx]
                self._centroids[i] = np.mean(related_points, axis=0)
            # self._centroids[np.arange(self._n_clusters)] = np.mean(X[np.where(labels == np.arange(self._n_clusters))], axis=0)
            previous_loss = loss
            # self.plot(X, labels)
            # if loss - previous_loss < self._tol:
            #     break
            # time.sleep(2)

    def plot(self, X: np.ndarray, labels: np.ndarray) -> None:
        """

        :param X:
        :param labels:
        :return:
        """
        plt.ion()
        plt.figure(figsize=(5, 5))
        colors = [list(np.random.choice(range(256), size=3, replace=False) / 255.0) for _ in range(self._n_clusters)]
        scatter(x=X[:, 0],
                y=X[:, 1],
                c=list(map(lambda x: colors[x], labels)))
        scatter(x=self._centroids[:, 0],
                y=self._centroids[:, 1],
                s=300,
                c=colors)
        savefig(r'D:\Coding\mllib-from-scratch\figs\fig.png')

    def __predict(self, X: np.ndarray) -> tuple:
        subtracted_matrix = X - self._centroids[:, np.newaxis]
        distances = np.sum(subtracted_matrix ** 2, axis=2) ** (1 / 2)
        return np.argmin(distances, axis=0), np.sum(np.min(distances, axis=0))


X = np.random.randint(0, 100, size=(100, 2))

kmeans = KMeans(n_clusters=3, max_iter=10000)
kmeans.fit(X)
print(kmeans._centroids)
sk_kmeans = sk_KMeans(n_clusters=3, max_iter=1000)
sk_kmeans.fit(X)
print(sk_kmeans.cluster_centers_)
