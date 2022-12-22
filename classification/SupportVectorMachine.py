import numpy as np
from typing import Union, List

class SVM:
    def __init__(self, learning_rate:float=0.01, epochs:int=100, lambda_:float=0.01) -> None:
        """
        Set parameters for SVM

        Parameters
        ----------
        lr : float
            Learning rate for training processing (fit method).
        epochs : int
            Iteration time in training processing
        lambda_ : float
            For regularization processing (L2 normalize).
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X: Union[List[List], np.ndarray], y: Union[List, np.ndarray]) -> None:
        """
        Training theta for find best margin between 2 features

        Parameters
        ----------
        X : Array-like of shape (n_samples, n_features)
            Input (data).
        y : Array-like of shape (n_features)
            Target (labels).

        Attributes
        ----------
        condition : bool
            Check if data on which side (-1 or 1)
        """
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception('ndim of X must be 2')
        assert np.ndim(y)==1, Exception('ndim of y must be 1')

        n_features = X.shape[1]

        y_ = np.where(y<=0, -1, 1)
        self.theta = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x_ in enumerate(X):
                condition = y_[idx]*(x_@self.theta - self.b) >= 1
                if condition:
                    self.theta -= self.lr*2*self.lambda_*self.theta
                else:
                    self.theta -= self.lr*(2*self.lambda_*self.theta - y_[idx]*x_)
                    self.b -= self.lr*y_[idx]

    def predict(self, X_test: Union[List[List], np.ndarray]):
        """
        Predict the side of input data

        Parameters
        ----------
        X_test : array-like of shape (n_samples_test, n_features_test)
            Input (test data).

        Attributes
        ----------
        approx : array-like of shape (n_samples)
            Approximate about the location in map.
        """
        X_test = np.array(X_test)
        assert np.ndim(X_test)==2, Exception('ndim of X_test must be 2')
        approx = X_test@self.theta - self.b
        return np.sign(approx)