import numpy as np


class PCA:
    def __init__(self, n_components: int):
        """Principal Component Analysis (PCA)
        
        Parameters:
        -----------
        n_components (int): 
            number of components to keep
        """
        assert n_components > 0, "n_component mus be > 0"
        self._n_components = n_components
    
    def fit(self, X: np.ndarray, std_scaler: bool=True):
        """
        Fitting PCA
        
        Parameters:
        ----------
        X (np.ndarray):
            Inputs to fit PCA
            
        std_scaler (bool, default=True):
            If True, standardize the data before fitting PCA
            
        Attributes:
        ----------
        explained_variance (np.ndarray):
            The amount of variance explained by each of the selected components.
            
        explained_variance_ratio (np.ndarray):
            Percentage of variance explained by each of the selected components.
        
        components (np.ndarray):
            Principal axes in feature space, representing the directions of maximum variance in the data.
        """
        assert X.ndim == 2, "ndim must be == 2"
        assert self._n_components <= min(A.shape), "n_components = min(num_of_samples, num_of_features)"

        X_ = X.copy()
        if std_scaler:
            X_ = (X_ - X_.mean(axis=0)) / X_.std(axis=0)
        else:
            X_ = X_ - X_.mean(axis=0)
        
        X_cov = np.cov(X_.T)
        eigval, eigvec = np.linalg.eig(X_cov)

        assert self._n_components <= len(eigval), "n_components must be <= amount of eigenvalues"
        
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        
        self.explained_variance = eigval[:self._n_components]
        self.explained_variance_ratio = self.explained_variance / eigval.sum()
        self.components = eigvec[:,:self._n_components]
    
    def transform(self, X: np.ndarray):
        """
        Transform PCA after fitting
        
        Parameters:
        ----------
        X (np.ndarray):
            Inputs to transform PCA
        """
        assert X.ndim == 2 and X.shape[-1] == self.components.shape[0], "ndim <> 2 or shape of X is invalid"
        return np.dot(X, self.components)

    def fit_transform(self, X: np.ndarray, std_scaler: bool=True):
        """
        Fit and transform PCA
        
        Parameters:
        ----------
        X (np.ndarray):
            Inputs to fit and transform PCA
            
        std_scaler (bool, default=True):
            If True, standardize the data before fitting PCA
        """
        self.fit(X, std_scaler)
        return self.transform(X)