import numpy as np

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

    def binary_cross_entropy(self, y, y_pred):
        loss = -(y.T@np.log(y_pred) + (1-y.T)@np.log(1-y_pred))
        dy_loss = -(y/y_pred - (1-y)/(1-y_pred))
        return loss, dy_loss

    def __shuffle_index(self, x):
        return np.random.permutation(x.shape[0])

    def fit(self, x, y, learning_rate, epochs, batch):
        ones = np.ones((x.shape[0], 1))
        self.theta = np.random.rand(x.shape[1], 1)
        loss_graph = []
        for ith_epoch in range(epochs):
            shuffle = self.__shuffle_index(x)
            x, y = x[shuffle], y[shuffle]
            for ith_batch in range(0, x.shape[0], batch):
                logits = self.linear_forward(x[ith_batch:ith_batch+batch, :], self.theta)
                y_pred = self.sigmoid_forward(logits)

                loss, dy_loss = self.binary_cross_entropy(y[ith_batch:ith_batch+batch, :], y_pred)

                print(dy_loss.shape)
                a = self.linear_backward(x[ith_batch:ith_batch+batch, :]).shape
                b = self.sigmoid_backward(y_pred).shape
                print(a, b)

                dtheta = dy_loss*self.linear_backward(x[ith_batch:ith_batch+batch, :])*self.sigmoid_backward(y_pred)
                self.theta = self.theta - learning_rate*dtheta

            if ith_epoch%10 == 0:
                print(f'loss of {ith_epoch} is: {loss}')
                loss_graph.append(loss)

        


    # def train(self, X, y, theta, b, learning_rate, epochs):
    #     loss_graph = []
    #     for ith_epoch in range(epochs):
    #         shuffle_index = np.arange(X.shape[0])
    #         np.random.shuffle(shuffle_index)
            
    #         for i in shuffle_index:
    #             logits = self.linear_forward(X[i], theta, b)
    #             y_pred = self.sigmoid_forward(logits)

    #             loss, dy_pred = self.binary_cross_entropy(y[i], y_pred)

    #             dtheta = dy_pred * self.linear_backward(X[i])[0] * self.sigmoid_backward(y_pred)
    #             db = dy_pred * self.linear_backward(X[i])[1] * self.sigmoid_backward(y_pred)

    #             theta = theta - learning_rate*dtheta
    #             b = b - learning_rate*db
            
    #         if ith_epoch % 30 == 0:
    #             print('loss: ', ith_epoch, 'is' , float(loss))
    #             loss_graph.append(loss)

    #     return theta, b


                
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
Y = np.concatenate((ones, Y), axis=1)

epochs = 120
lr = 0.001
batch = 256

reg = LogisticRegression()
reg.fit(X, Y, lr, epochs, batch)

print(reg.theta)

        