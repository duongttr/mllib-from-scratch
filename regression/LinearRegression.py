import numpy as np
from numpy.linalg import pinv
from numpy.linalg import matrix_rank
from typing import List, Union


class LinearRegression:
    def __init__(self, random_state:bool = True) -> None:
        """Find the linear that fit with data given 
        :param
            random_init : bool
                set the permanent random or not
        :attribute
            coef_ : np.ndarray
                the coefficient
            intercept_ : float
                the intercept
            rank_ : int
                rank of dependence data (X)
        """

        if random_state: np.random.seed(99)
        self.coef_ = None
        self.intercept_ = None
        self.rank_ = None

    def fit(self, x: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> None:
        """Calculate the weight parameter with independent data (x) and dependent data(y)
        :param
            x : List[List], np.ndarray
                independent data
            y : List, np.ndarray
                dependent data
        """

        x, y = np.array(x), np.array(y)
        assert np.ndim(x) == 2, Exception('ndim of x must be 2')
        assert np.ndim(y) == 1, Exception('ndim of y must be 1')
        assert x.shape[0] == y.shape[-1], Exception('x and y must have same size')

        self.rank_ = matrix_rank(x)

        ones = np.ones([x.shape[0], 1])
        x = np.concatenate([ones, x], axis=1)
        theta = pinv(x.T@x) @ (x.T@y)
        
        self.intercept_, self.coef_ = theta[0], theta[1:]

    def predict(self, x_pred: Union[List[List], np.ndarray]) -> np.ndarray:
        """Predict value from the data given (predict y from x)
        :param
            x_pred : List[List], np.ndarray
                independent data, number of samples used in the fitting for the estimator

        :return
            ndarray : predictive value calculate from x_pred and weight
        """

        x_pred = np.array(x_pred)
        assert np.ndim(x_pred) == 2, Exception('ndim of x must be 2')
        assert self.coef_ is not None and \
                   self.intercept_ is not None, Exception("haven't fit data yet")
        return np.dot(x_pred, self.coef_) + self.intercept_

    def score(self, X: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> float:
        """Show the R2 score, which the coefficient of determination of the prediction
        :param
            X : List[List], np.ndarray
                independent data, number of samples used in the fitting for the estimator
            y : List, np.ndarray
                True value for x
        :return
            float : R2 score depend on self.predict(X) 
        """

        self.fit(X, y)
        y_pred = self.predict(X)
        u = np.sum((y - y_pred)**2)
        v = np.sum((y - np.mean(y))**2)
        return 1 - u/v