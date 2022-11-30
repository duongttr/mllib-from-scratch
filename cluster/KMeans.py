import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter


class KMeans:
    def __init__(self,
                 n_clusters: int = 9,
                 init: str = 'k-means++',
                 max_iter: int = 300,
                 tol: float = 1e-4) -> None:
        """Initialize attributes.

        :param n_clusters:
        :param init:
        :param max_iter:
        :param tol:
        """
        self._n_clusters = n_clusters
        assert init != 'k-means++' or init != 'random', Exception("Please specify 'k-means++' or 'random'")
        self._init = init
        self._max_iter = max_iter
        self._tol = tol
        self._centroids = None

    def fit(self, X: np.ndarray, visualized: bool = False) -> None:
        """Find the optimal centroid of each cluster.

        :param X: (np.ndarray)
            2-D array data
        :param bool visualized: (bool, default: False)
            Plot the final result on the coordinate plane only if the data is
            2-D array and have less than 4 features
        """
        # Check data in 2-D
        assert X.ndim == 2, Exception('Dim must be 2')
        # Initialize centroids
        if self._init == 'random':
            self._centroids = X[np.random.choice(X.shape[0], size=self._n_clusters, replace=False)].astype(np.float32)
        else:
            self._centroids = X[np.random.choice(X.shape[0], size=1, replace=False)].astype(np.float32)
            self.kmeanspp(X)
        labels = []
        previous_loss = 0
        for _ in range(self._max_iter):
            labels, loss = self.predict(X)
            for i in range(self._n_clusters):
                related_points_idx = np.where(labels == i)[0]
                related_points = X[related_points_idx]
                self._centroids[i] = np.mean(related_points, axis=0)
            # Check if it converges
            if loss - previous_loss < self._tol:
                previous_loss = loss
                break
            previous_loss = loss
        print(previous_loss)
        # Visualize
        if visualized:
            assert X.shape[1] < 4, Exception('Data is too complicated to be visualized')
            self.plot(X, labels)

    def kmeanspp(self, X: np.ndarray) -> None:
        """Choose initial cluster by statistics

        :param X: (np.ndarray) Array of datapoints
        """
        for _ in range(self._n_clusters - 1):
            distances = self.calc_distance(X, self._centroids)
            furthest_point = X[np.argmax(np.sum(distances, axis=0))]
            self._centroids = np.vstack((self._centroids, furthest_point))

    def plot(self, X: np.ndarray, labels: np.ndarray, loss: float) -> None:
        """Plot the datapoints and centroids on the coordinate plane

        :param X: (np.ndarray)
        :param labels: (np.ndarray)
        :param loss: (np.ndarray)
        """
        plt.figure(figsize=(5, 5))
        colors = [list(np.random.choice(range(256), size=3, replace=False) / 255.0) for _ in range(self._n_clusters)]
        scatter(x=X[:, 0],
                y=X[:, 1],
                c=list(map(lambda x: colors[x], labels)))
        scatter(x=self._centroids[:, 0],
                y=self._centroids[:, 1],
                s=300,
                c=colors)
        plt.show()

    def predict(self, X: np.ndarray) -> tuple:
        """Predict which centroid (label) that each datapoint belongs to, and calculate loss.

        :param X: (np.ndarray)
        :return: (tuple) (label vector, total loss)
        """
        distances = self.calc_distance(X, self._centroids)
        labels = np.argmin(distances, axis=0)
        loss = np.sum(np.min(distances, axis=0))    # Total distance from each datapoints to the nearest centroid
        return labels, loss

    @staticmethod
    def calc_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate the distance between each datapoints in X to each datapoints in Y.

        :param X: (np.ndarray) Array of datapoints
        :param Y: (np.ndarray) Array of datapoints
        :return: (np.ndarray) The distance matrix whose rows, columns present the datapoints in X, Y respectively
        """
        subtracted_matrix = X - Y[:, np.newaxis]
        return np.sum(subtracted_matrix ** 2, axis=2) ** (1 / 2)
