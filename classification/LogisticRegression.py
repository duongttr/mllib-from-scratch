import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils.metrics import accuracy

class LogisticRegression:
    def __init__(self, epsilon=1e-8, random_state: int = None) -> None:
        np.random.seed(random_state)
        self.epsilon = epsilon
        

    def linear_forward(self, X, theta):
        return X@theta

    def linear_backward(self, X):
        return X

    def sigmoid_forward(self, Z):
        return 1 / (1+np.exp(-Z))

    def sigmoid_backward(self, X):
        return X*(1-X)

    def binary_cross_entropy(self, y, y_pred, batch_size):
        y_pos = y_pred + self.epsilon
        y_neg = 1-y_pred + self.epsilon
        loss = -(y*np.log(y_pos) + (1-y)*np.log(1-y_neg))
        mean_loss = loss.sum() / batch_size

        dy_loss = -(y/(self.epsilon + y_pred) - (1-y)/(self.epsilon + 1-y_pred)) / batch_size
        return mean_loss, dy_loss

    def __shuffle_index(self, x):
        return np.random.permutation(x.shape[0])

    def fit(self, x, y, learning_rate, epochs, batch):
        self.theta = np.random.rand(x.shape[1], 1)
        self.loss_graph = []
        for ith_epoch in range(epochs):
            shuffle = self.__shuffle_index(x)
            x, y = x[shuffle], y[shuffle]
            for ith_batch in range(0, x.shape[0], batch):
                logits = self.linear_forward(x[ith_batch:ith_batch+batch, :], self.theta)
                y_pred = self.sigmoid_forward(logits)
                loss, dy_loss = self.binary_cross_entropy(y[ith_batch:ith_batch+batch, :], y_pred, batch)
                linear_backward = self.linear_backward(x[ith_batch:ith_batch+batch, :])
                sigmoid_backward = self.sigmoid_backward(y_pred)

                dtheta = dy_loss*sigmoid_backward*linear_backward
                dtheta_mean = np.mean(dtheta, axis=0, keepdims=True).T
                self.theta -= learning_rate*dtheta_mean

            if ith_epoch%10 == 0:
                print(f'Loss of {ith_epoch} is: {loss}')
            self.loss_graph.append(loss)

    def predict_proba(self, X):
        logits = self.linear_forward(X, self.theta)
        return self.sigmoid_forward(logits)
    
    def predict_labels(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.array(proba >= threshold, dtype='int')
        



                
N = 500
mu, sigma = 0, 1
x_0 = np.random.normal(mu, sigma, N)
x_1 = np.random.normal(mu + 100, sigma, N)

X = [x_0, x_1]
Y = [np.zeros_like(x_0), np.ones_like(x_1)]

X, Y = np.concatenate(X), np.concatenate(Y)


X, Y = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))

ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, X), axis=1)

epochs = 1000
lr = 0.00001
batch = 32

reg = LogisticRegression()
reg.fit(X, Y, lr, epochs, batch)

y_pred_label = reg.predict_labels(X, threshold=0.8)
y = Y
print(accuracy(y, y_pred_label))
        