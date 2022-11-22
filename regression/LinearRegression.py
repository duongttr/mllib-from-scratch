import numpy as np
from numpy.linalg import pinv
from typing import List, Union


class LinearRegression:
    def __init__(self, random_init:bool = True) -> None:
        """Find the linear that fit with data given 
        :param
            random_init: bool
                set the permanent random or not
        :attribute
            _coef: np.ndarray
                the coefficient
            _inte: float
                the intercept
        """
        if random_init: np.random.seed(99)
        self._coef = None
        self._inte = None

    def fit(self, x: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> None:
        """Calculate the weight parameter with independent data (x) and dependent data(y)
        :param
            x: List[List], np.ndarray
                independent data
            y: List, np.ndarray
                dependent data
        """
        x, y = np.array(x), np.array(y)
        assert np.ndim(x) == 2, Exception('ndim of x must be 2')
        assert np.ndim(y) == 1, Exception('ndim of y must be 1')
        assert x.shape[0] == y.shape[-1], Exception('x and y must have same size')

        ones = np.ones([x.shape[0], 1])
        x = np.concatenate([ones, x], axis=1)
        theta = pinv(x.T@x) @ (x.T@y)
        
        self._inte, self._coef = theta[0], theta[1:]

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
        assert self._coef is not None and \
               self._inte is not None, Exception("haven't fit data yet")
        y_pred = np.dot(x_pred, self._coef) + self._inte
        return y_pred