import numpy as np
from numpy.linalg import pinv
from typing import List, Union


class LinearRegression:
    def __init__(self, random_init:bool = True) -> None:
        """Find the linear that fit with data given 
        :param
            random_init: bool
                set the permanent random or not 

        """
        if random_init: self = np.random.seed(99)
    
    def fit(self, x: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> np.ndarray:
        """Calculate the weight parameter with independent data (x) and dependent data(y)
        :param
            x: List[List], np.ndarray
                independent data
            y: List, np.ndarray
                dependent data
        :return
            ndarray: weight parameter(s) was calculated by x and y

        """
        x, y = np.array(x), np.array(y)
        assert np.ndim(x) == 2, Exception('ndim of x must be 2')
        assert np.ndim(y) == 1, Exception('ndim of y must be 1')
        theta = pinv(x.T@x) @ (x.T@y)
        self._theta = theta
        return self._theta

    def predict(self, x_pred: Union[List[List], np.ndarray]) -> np.ndarray:
        """Predict value from the data given (predict y from x)
        :param
            x_pred: List[List], np.ndarray
                independent data
        :return
            ndarray: predictive value calculate from x_pred and weight
        
        """
        x_pred = np.array(x_pred)
        assert np.ndim(x_pred) == 2, Exception('ndim of x must be 2')
        # assert self._theta, Exception("haven't fit data yet")
        y_pred = np.dot(x_pred, self._theta)
        return y_pred
