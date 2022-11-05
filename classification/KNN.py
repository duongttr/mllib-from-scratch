# KNN - K-Nearest Neighbor
from numbers import Number
from typing import Callable, List
import numpy as np
from utils.distance import *

class KNN:
    def __init__(self, k_neighbors: int,
                p: int = 2,
                custom_metric: Callable[[List[Number], List[Number]], Number] = None):

        assert k_neighbors > 0, "k-neighbors must be greater than 0"
        assert p > 0, "p must be greater than 0"

        self._k_neighbors = k_neighbors
        self._p = p
        self._custom_metric = custom_metric
    
    def fit(self, X: List[List[Number]], y: List[Number]):
        assert len(X) == len(y), "Datapoints and labels must be equal"
        self._X = np.array(X)
        self._y = np.array(y)

    def _get_max_count_label(self, x: List):
        unique, counts = np.unique(x, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict(self, X: List[List[Number]]):
        distances = []

        if self._custom_metric: # default metric
            for point in X:
                distances.append(np.apply_along_axis(minkowski, -1, self._X, point, self._p))
        else: # metric is callable
            for point in X:
                distances.append(np.apply_along_axis(self._custom_metric, -1, self._X, point))
        
        indices = np.argsort(distances, axis=-1)[:,:self._k_neighbors]

        labels = [self._y[idx] for idx in indices]
        return [self._get_max_count_label(label) for label in labels]