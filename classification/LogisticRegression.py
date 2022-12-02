import numpy as np
from typing import Union, List

class LogisticRegression:
    def __init__(self, random_state: bool=True) -> None:
        """Classification data into 2 class
        :param
            random_init : bool
                set the permanent random or not
        :attributes
            coef_ : np.ndarray
                coefficient of the features in the decision function
            intercept_ : float
                intercept of the features in the decision function
        """
        if random_state: np.random.seed(99)
        
        self.intercept_ = None
        self.coef_ = None

    def __calculate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """calculate parameters
        :param
            X : np.ndarray
                training vector
        :return
            logits : np.ndarray
                linear forward of X and theta (weights)
            y_pred : np.ndarray
                sigmoid forward of logits
            dW : np.ndarray
                derivative of logits (linear backward)
            dy_pred : np.ndarray
                derivative of sigmoid_forward (sigmoid backward)
        """
        logits = X@self.theta
        y_pred = 1 / (1+np.exp(-logits))
        dW = X
        dy_pred = y_pred*(1-y_pred)
        return logits, y_pred, dW, dy_pred

    def __binary_cross_entropy(self, y: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate the binary cross entropy loss and derivative of its
        :param
            y : np.ndarray
                vector label
            y_pred : np.ndarray
                vector predict from the sigmoid function
        :return
            mean_loss : float
                number to measure the different between label and prediction 
            dy_loss : np.ndarray
                derivation of loss vector
        """
        epsilon = 1e-9
        pos_y_pred = y_pred+epsilon
        neg_y_pred = 1-y_pred+epsilon
        loss = -(y*(np.log(pos_y_pred) + (1-y)*np.log(neg_y_pred)))
        mean_loss = sum(loss) / self.batch

        dy_loss = -(y/pos_y_pred - (1-y)/(neg_y_pred)) / self.batch
        return mean_loss, dy_loss

    def __shuffle(self, X: np.ndarray, y: np.ndarray, shape: int) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle the training vector and labels of them
        :param
            X : np.ndarray
                old training vector
            y : np.ndarray
                old labels
        :return
            X : np.ndarray
                shuffled training vector
            y: np.ndarray
                shuffled labels
        """
        shuffle =  np.random.permutation(shape)
        return X[shuffle], y[shuffle]

    def fit(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], learning_rate: float=0.001, epoch: int=1000, batch: int=64) -> None:
        """training to get weights fit with input and labels
        :param
            X : np.ndarray, List[List]
                training matrix
            y : np.ndarray, List
                target vector relative to X
            learning_rate : float
                learning rate of each step training
            epoch : int
                iteration of training
            batch : int
                number of features get in each step
        """
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception('the ndim of X must be 2')
        # assert np.ndim(y)==1, Exception('ndim of y must be 1')
        assert X.shape[0]==y.shape[0], Exception('x and y must have the same size')

        self.batch = batch
        height, weight = X.shape
        self.theta = np.random.rand(weight, 1)
        loss_history = []
        min_loss = float('inf')
        for _ in range(epoch):
            X, y = self.__shuffle(X, y, height)
            for ith_batch in range(0, height, batch):
                _, y_pred, dW, dy_pred = self.__calculate(X[ith_batch:ith_batch+batch, :])
                loss, dy_loss = self.__binary_cross_entropy(y[ith_batch:ith_batch+batch, :], y_pred)

                loss_history.append(loss)

                grad = dy_loss*dy_pred*dW
                grad_mean = np.mean(grad, axis=0, keepdims=True).T
                self.theta -= learning_rate*grad_mean
                
            if loss_history[-1] < min_loss:
                best_theta = self.theta
                min_loss = loss_history[-1]
        choice = int(input(f"(1) Current theta: {loss_history[-1]}\n(2) Best theta: {min_loss}\nType the option you want (1, 2): "))
        if choice == 2:
            self.theta = best_theta

        self.intercept_, self.coef_ = self.theta[0], self.theta[1:]

    def predict(self, X: Union[np.ndarray, List[List]], threshold: float=0.5) -> np.ndarray:
        """Predict input value to suitable class
        :param
            X : np.ndarray, List[List]
                the data maxtrix for which we want to predict
            threshold : float
                the threshold to seperate the input to which class
        :return
            np.ndarray : predictive class for input value
        """
        X = np.array(X)
        assert np.ndim(X)==2, Exception('ndim of X must be 2')

        _, y_pred, _, _ = self.__calculate(X)
        return np.where(y_pred <= threshold, 0, 1)

    def predict_proba(self, X: Union[np.ndarray, List[List]]) -> np.ndarray:
        """Predict probability of class 1 in input value
        :param
            X : np.ndarray, List[List]
                the data matrix for which we want to predict
        :return
            np.ndarray
                vector contain probability of correspond input value
        """
        X = np.array(X)
        assert np.dim(X)==2, Exception('ndim of X must be 2')

        _, y_pred, _, _ = self.__calculate(X)
        return y_pred

    def score(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List], threshold: float=0.5) -> float:
        """Return the mean accuracy on the given test data and labels
        :param
            X : np.ndarray, List[List]
                training matrix
            y : np.ndarray, List
                labels of training matrix
            threshold : float
                the threshold to seperate the input to which class
        :return
            float : themMean accuracy of self.predict(X)
        """
        y_pred = self.predict(X, threshold=threshold)
        return np.mean(y==y_pred)