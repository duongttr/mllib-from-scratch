import numpy as np

class naive_bayes:
    class GaussianNB:
        def fit(self, X, y):
            __num_shape, __num_features = X.shape
            self.__classes = np.unique(y)
            __num_class = len(self.__classes)

            self.__mean = np.zeros((__num_class, __num_features))
            self.__var = np.zeros((__num_class, __num_features))
            self.__prior = np.zeros(__num_class)

            for idx, value in enumerate(self.__classes):
                X_value = X[y==value]
                self.__mean[idx, :] = np.mean(X_value, axis=0)
                self.__var[idx, :] = np.var(X_value, axis=0)
                self.__prior[idx] = X_value.shape[0] / __num_class

        def predict(self, X):
            temp_y = []
            for x in X:
                temp_y_th = []
                for idx in range(len(self.__classes)):
                    mean = self.__mean[idx]
                    var = self.__var[idx]
                    gausiance = np.exp(-((x-mean)**2) / (2*var)) / (np.sqrt(2*np.pi*var))
                    gausiance_result = np.sum(np.log(gausiance))
                    prior = np.log(self.__prior[idx])
                    posterior = gausiance_result + prior
                    temp_y_th.append(posterior)
                temp_y.append(np.argmax(temp_y_th))
            return np.array(temp_y)