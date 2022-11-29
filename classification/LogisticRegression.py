import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, random_state: bool = False) -> None:
        if random_state: np.random.seed(99)

    def linear_forward(self, X, theta):
        return X@theta

    def linear_backward(self, X):
        return X

    def sigmoid_forward(self, Z):
        return 1 / (1+np.exp(-Z))

    def sigmoid_backward(self, X):
        return X*(1-X)

    def binary_cross_entropy(self, y, y_pred, batch_size):
        y_pos = np.min(np.concatenate([y_pred + 1e-9, np.ones_like(y_pred)],axis=1))
        y_neg = np.min(np.concatenate([1-y_pred + 1e-9, np.ones_like(y_pred)],axis=1))
        loss = -(y*np.log(y_pos) + (1-y)*np.log(1-y_neg)) / batch_size

        dy_loss = -(y/y_pred - (1-y)/(1-y_pred))
        return loss, dy_loss

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
                dtheta = np.mean(dtheta)
                self.theta = self.theta - learning_rate*dtheta

            if ith_epoch%10 == 0:
                # print(f'loss of {ith_epoch} is: {loss}')
                self.loss_graph.append(loss)

    def predict(self, x_test):
        logits = self.linear_forward(x_test, self.theta)
        return self.sigmoid_forward(logits)

        



                
N = 500
mu, sigma = 0, 1
x_0 = np.random.normal(mu, sigma, N)
x_1 = np.random.normal(mu + 5, sigma*0.50, N)

X = [x_0, x_1]
Y = [np.zeros_like(x_0), np.ones_like(x_1)]

X, Y = np.concatenate(X), np.concatenate(Y)


X, Y = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))

ones = np.ones((X.shape[0], 1))
X = np.concatenate((ones, X), axis=1)

epochs = 1000
lr = 0.001
batch = 256

reg = LogisticRegression()
reg.fit(X, Y, lr, epochs, batch)
print(reg.loss_graph)

plt.plot(reg.loss_graph)
plt.show()

print(reg.predict(X[10:20]))
print(Y[10:20])

        