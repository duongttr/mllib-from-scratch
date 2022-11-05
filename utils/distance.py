from numbers import Number
from typing import List
import numpy as np

def minkowski(X: List[Number], Y: List[Number], p: int = 2):
    assert len(X) == len(Y), 'Lengths of two lists do not equal'
    assert p > 0, 'p must be greater than 0'

    diff = np.subtract(X,Y)
    return np.sum(diff ** p) ** (1/p)

def manhattan(X: List[Number], Y: List[Number]):
    return minkowski(X, Y, 1)

def euclidean(X: List[Number], Y: List[Number]):
    return minkowski(X, Y, 2)
