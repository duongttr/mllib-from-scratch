import numpy as np


class PCA:
    def __init__(self, n_components: int):
        assert n_components > 0, "n_component mus be > 0"
        self._n_components = n_components
    
    def fit(self, A: np.ndarray, std_scaler: bool=True):
        assert A.ndim == 2, "ndim must be == 2"
        assert self._n_components <= min(A.shape), "n_components = min(num_of_samples, num_of_features)"

        A_ = A.copy()
        if std_scaler:
            A_ = (A_ - A_.mean(axis=0)) / A_.std(axis=0)
        else:
            A_ = A_ - A_.mean(axis=0)
        
        A_cov = np.cov(A_.T)
        eigval, eigvec = np.linalg.eig(A_cov)

        assert self._n_components <= len(eigval), "n_components must be <= amount of eigenvalues"
        
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        
        self.explained_variance = eigval[:self._n_components]
        self.explained_variance_ratio = self.explained_variance / eigval.sum()
        self.components = eigvec[:,:self._n_components]
    
    def transform(self, A: np.ndarray):
        assert A.ndim == 2 and A.shape[-1] == self.components.shape[0], "ndim <> 2 or shape of A is invalid"
        return np.dot(A, self.components)

    def fit_transform(self, A: np.ndarray, std_scaler: bool=True):
        self.fit(A, std_scaler)
        return self.transform(A)