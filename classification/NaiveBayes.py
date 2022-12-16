import numpy as np
from typing import Union

class naive_bayes:
    class GaussianNB:
        def fit(self, X: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> None:
            """
            Fit Gaussian Naive Bayes according to X and y

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training vector, where `n_samples` and `n_features` is number of sample(s) and feature(s).
            y : array-like of shape (n_sample(s),)
                Target value(s).

            Attributes
            ----------
            mean : ndarray of shape (n_class, n_features)
                Mean for Gaussians in original set, where `n_class` and `n_features` is number of class(es) and feature(s).
            var : ndarray of shape (n_class, n_features)
                Variance for Gaussians in original set.
            prior : ndarray of shape (n_class)
                Prior for Gaussians in origianl set.
            """
            
            X, y = np.array(X), np.array(y)
            assert np.ndim(X)==2, Exception('ndim of x must be 1')
            assert np.ndim(y)==1, Exception('ndim of y must be 1')

            __num_samples, __num_features = X.shape
            self.__classes = np.unique(y)
            __num_class = len(self.__classes)

            self.mean = np.zeros((__num_class, __num_features))
            self.var = np.zeros((__num_class, __num_features))
            self.prior = np.zeros(__num_class)

            for idx, value in enumerate(self.__classes):
                X_value = X[y==value]
                self.mean[idx, :] = np.mean(X_value, axis=0)
                self.var[idx, :] = np.var(X_value, axis=0)
                self.prior[idx] = X_value.shape[0] / __num_class

        # def predict(self, X):
        #     temp_y = []
        #     for x in X:
        #         temp_y_th = []
        #         for idx in range(len(self.__classes)):
        #             mean = self.mean[idx]
        #             var = self.var[idx]
        #             gausiance = np.exp(-((x-mean)**2) / (2*var)) / (np.sqrt(2*np.pi*var))
        #             gausiance_result = np.sum(np.log(gausiance))
        #             prior = np.log(self.prior[idx])
        #             posterior = gausiance_result + prior
        #             temp_y_th.append(posterior)
        #         temp_y.append(np.argmax(temp_y_th))
        #     return np.array(temp_y)



        def predict(self, X):
            temp_y = []
            for x in X:
                temp_y_th = []
                for idx in range(len(self.__classes)):
                    mean = self.mean[idx]
                    var = self.var[idx]
                    gausiance = np.exp(-((x-mean)**2) / (2*var)) / (np.sqrt(2*np.pi*var))
                    gausiance_result = np.sum(np.log(gausiance))
                    prior = np.log(self.prior[idx])
                    posterior = gausiance_result + prior
                    temp_y_th.append(posterior)
                temp_y.append(np.argmax(temp_y_th))
            return np.array(temp_y)


        def print_test(self, X):
            print(f'X is {X} and shape is {X.shape}')
            print(f'mean is: {self.mean} and shape is {self.mean.shape}')
            print(f'var is: {self.var}')
            print(f'prior is {self.prior}')


    
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None
    )

    nb = naive_bayes.GaussianNB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print(y_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

    # nb.print_test(X)