import numpy as np
from typing import Union, List

class LogisticRegression:
    def __init__(self, random_state: int=None, save_best_option: bool=True) -> None:
        """
        Init attributes intercept and coefficient. Set the random state.

        Parameters
        ----------
        random_state : int
            Set the random state in seed. If not choice then auto random.
        save_best_theta_option : bool
            Option to choose the best theta in fit() method.
        
        Attributes
        ----------
        coef_ : np.ndarray
            Coefficient of the features in the decision function.
        intercept_ : float
            Intercept of the features in the decision function.
        """
        np.random.seed(random_state)

        self.save_best_choice = save_best_option
        
        self.intercept_ = None
        self.coef_ = None

    def __calculate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate logits, y prediction (sigmoid forward), derivative of logits, derivative of sigmoid.

        Parameters
        ----------
        X : np.ndarray
            Training vector.
        
        Returns
        -------
        logits : np.ndarray
            Linear forward of X and theta (weights).
        y_pred : np.ndarray
            Sigmoid forward of logits.
        dW : np.ndarray
            Derivative of logits (linear backward).
        dy_pred : np.ndarray
            Derivative of sigmoid_forward (sigmoid backward).
        """
        logits = X@self.theta
        y_pred = 1 / (1+np.exp(-logits))
        dW = X
        dy_pred = y_pred*(1-y_pred)
        return logits, y_pred, dW, dy_pred

    def __binary_cross_entropy(self, y: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Calculate the binary cross entropy loss and derivative of its
        
        Parameters
        ----------
        y : np.ndarray
            Vector label.
        y_pred : np.ndarray
            Vector predict from the sigmoid function.
        
        Returns
        -------
        mean_loss : float
            Number to measure the different between label and prediction.
        dy_loss : np.ndarray
            Derivation of loss vector.
        """
        len_arr = len(y)
        epsilon = 1e-9
        pos_y_pred = y_pred+epsilon
        neg_y_pred = 1-y_pred+epsilon
        loss = -(y*(np.log(pos_y_pred) + (1-y)*np.log(neg_y_pred)))
        mean_loss = sum(loss) / len_arr

        dy_loss = -(y/pos_y_pred - (1-y)/(neg_y_pred)) / len_arr
        return mean_loss, dy_loss

    def __shuffle(self, X: np.ndarray, y: np.ndarray, shape: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle the training vector and labels of them.
        
        Parameters
        ----------
        X : np.ndarray
            Old training vector.
        y : np.ndarray
            Old labels.

        Returns
        -------
        X : np.ndarray
            Shuffled training vector.
        y : np.ndarray
            Shuffled labels.
        """
        shuffle =  np.random.permutation(shape)
        return X[shuffle], y[shuffle]

    def fit(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], learning_rate: float=0.001, epoch: int=1000, batch: int=64) -> None:
        """
        Training to get weights fit with input and labels.
        
        Parameters
        ----------
        X : np.ndarray, List[List]
            Training matrix.
        y : np.ndarray, List
            Target vector relative to X.
        learning_rate : float
            Learning rate of each step training.
        epoch : int
            Iteration of training.
        batch : int
            Number of features get in each step.
        """
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception('the ndim of X must be 2')
        assert np.ndim(y)==1, Exception('ndim of y must be 1')
        assert X.shape[0]==y.shape[0], Exception('x and y must have the same size')

        self.batch = batch
        height, weight = X.shape
        self.theta = np.random.rand(weight, 1)
        loss_history = []
        min_loss = float('inf')
        for _ in range(epoch):
            X, y = self.__shuffle(X, y, height)
            loss_in_1batch = []
            for ith_batch in range(0, height, batch):
                _, y_pred, dW, dy_pred = self.__calculate(X[ith_batch:ith_batch+batch, :])
                loss, dy_loss = self.__binary_cross_entropy(y[ith_batch:ith_batch+batch, :], y_pred)

                loss_in_1batch.append(loss)
                grad = dy_loss*dy_pred*dW
                grad_mean = np.mean(grad, axis=0, keepdims=True).T
                self.theta -= learning_rate*grad_mean
            loss = sum(loss_in_1batch) / len(loss_in_1batch)
            loss_history.append(loss)
            if loss_history[-1] < min_loss:
                best_theta = self.theta
                min_loss = loss_history[-1]

        if self.save_best_choice: self.theta = best_theta

        self.intercept_, self.coef_ = self.theta[0], self.theta[1:]

    def predict(self, X: Union[np.ndarray, List[List]], threshold: float=0.5) -> np.ndarray:
        """
        Predict input value to suitable class.

        Parameters
        ----------
        X : np.ndarray, List[List]
            The data maxtrix for which we want to predict.
        threshold : float
            The threshold to seperate the input to which class.

        Returns
        -------
        predicted_value(s) : np.ndarray
            Predictive class for input value.
        """
        X = np.array(X)
        assert np.ndim(X)==2, Exception('ndim of X must be 2')

        _, y_pred, _, _ = self.__calculate(X)
        return np.where(y_pred <= threshold, 0, 1)

    def predict_proba(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        """
        Predict probability of class 1 in input value.

        Paremeters
        ----------
        X : np.ndarray, List
            The data matrix for which we want to predict.

        Returns
        -------
        predict_proba_value : np.ndarray
            Vector contain probability of correspond input value.
        """
        X = np.array(X)
        assert np.dim(X)==2, Exception('ndim of X must be 2')

        _, y_pred, _, _ = self.__calculate(X)
        return y_pred

    def score(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], threshold: float=0.5) -> float:
        """
        Return the mean accuracy on the given test data and labels
        
        Parameters
        ----------
        X : np.ndarray, List[List]
            Training matrix.
        y : np.ndarray, List
            Labels of training matrix.
        threshold : float
            The threshold to seperate the input to which class.
        
        Returns
        -------
        score: Float
            The mean accuracy of self.predict(X).
        """
        y_pred = self.predict(X, threshold=threshold)
        return np.mean(y==y_pred)
