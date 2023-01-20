# Linear Discriminant Analysis
import numpy as np
from typing import List, Union
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA:
    def __init__(self, n_components: int = None):
        """
        Parameters:
        - `n_components (int)`: number of components
        """
        if n_components is not None:
            assert n_components > 0, "Number of components must be greater than 0"
        self.n_components = n_components
        self.linear_discriminants = None
        
    
    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]):
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        if self.n_components is not None:
            assert self.n_components <= min(len(class_labels) - 1, n_features), "Number of components must be less than or equal to the minimum of the number of classes minus 1 and the number of features"
        else:
            self.n_components = min(len(class_labels) - 1, n_features)
            
        assert X.shape[0] == y.shape[0], "Number of samples must be equal to the number of labels"
        
        
        mean_overall = np.mean(X, axis=0)
        
        # ST = np.cov(X)
        SW = np.zeros((n_features, n_features)) # Within-class covariance matrix
        SB = np.zeros((n_features, n_features)) # Between-class covariance matrix
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c  = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot(X_c - mean_c)
            
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)
            
        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        evals, evecs = np.linalg.eigh(A)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.ld = evecs[:, :self.n_components]
        
    def transform(self, X: Union[List, np.ndarray]): 
        return X @ self.ld
    
    def fit_transform(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]):
        self.fit(X, y)
        return self.transform(self, X)