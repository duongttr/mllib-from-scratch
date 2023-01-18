import numpy as np
from typing import List, Union

class FactorAnalysis():
    def __init__(self, n_components:int=None, random_state:int=0, epochs:int=200, tol:float=1e-6):
        """
        Reduce dimension of data by find the lareitve of features
        
        Parameters
        ----------
        n_components : int
            The numbers of features want to have after factor analysis.
        random_state : int
            Seed in state.
        epochs : int
            Max iterations of training loop we want.
        tol : int
            Limit number about the different between old value and new value.
        """

        self.n_components = n_components
        np.random.seed(random_state)
        self.epochs = epochs
        self.tol = tol
    
    def fit(self, X: Union[List[List], np.ndarray]):
        """
        Training state and get the factor score

        Parameters
        ----------
        X : array-like shape (n_samples, n_features)
            Data want to reduce dimension

        Attributes
        ----------
        loading_ : array-like shape (n_samples, n_components)
            Factor score of traning
        """

        n_samples, n_features = X.shape
        
        X_centered = X - X.mean(axis=0)
        
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        if self.n_components is None:
            k = n_features
        else:
            k = self.n_components
        W = eigenvectors[:,:k]
        
        for i in range(self.epochs):
            F = X_centered @ W
            X_reconstructed = F @ W.T + X.mean(axis=0)
            
            residuals = X_centered - X_reconstructed
            
            sigma = np.mean(residuals ** 2)
            
            epsilon = 1e-6
            W_new = np.linalg.solve(F.T @ F + (sigma / (eigenvalues[:k]+epsilon)) * np.eye(k), F.T @ X_centered)
            
            if np.max(np.abs(W - W_new.T)) < self.tol:
                break
            W = W_new.T
        
        self.loading_ = W
        self.noise_variance_ = sigma
        self.components_ = W.T
        self.explained_variance_ = eigenvalues[:k]

    def transform(self, X: Union[List[List], np.ndarray]):
        """
        Apply dimensionality reduction to X using the model.
        """

        X_centered = X - X.mean(axis=0)
        F = X_centered @ self.loading_
        return F
    
    def fit_transform(self, X: Union[List[List], np.ndarray]):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data want to reduct dimension.
        
        Return
        ------
        X_newndarray array-like of shape (n_samples, n_components_)
            Transformed array.
        """

        self.fit(X)
        return self.transform(X)
