import numpy as np

def accuracy(y, y_pred):
    assert len(y) == len(y_pred), Exception('Length of y and y_pred must be equal')
    return np.sum(y == y_pred) / len(y)