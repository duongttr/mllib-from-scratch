import numpy as np
from collections import Counter
from typing import Union, List

class Node:
    def __init__(self, features: int=None, threshold: int=None, left=None, right=None, value=None) -> None:
        """
        Initialize the node of tree.

        Parameters
        ----------
        features : int
            The feature index in node.
        threshold : int
            The threshold in this node.
        left : Root left
            The left Node conencted.
        right :Root right
            The right Node connected.
        value : int or float
            The value of Node.
        """
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        
        self.value = value

    def is_leaf_node(self):
        """
        Check if the present location is the leaf of tree
        """
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split: Union[int, float]=2, max_depth: int=100, n_features: Union[int, float]=None) -> None:
        """
        Initialize the condition needed of decision tree
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X: Union[np.ndarray, List[List]], y: Union[np.ndarray, List]) -> None:
        """
        Create the binary decision tree form from input X and target y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` and `n_features` is number of sample(s) of feature(s).
        y : array-like of shape (n_features)
            Target vector (labels).
        
        Attributes
        ----------
        root : Node
            The full tree from input and target.
        """
        X, y = np.array(X), np.array(y)
        assert np.ndim(X)==2, Exception("ndim of X must be 2")
        assert np.ndim(y)==1, Exception("ndim of y must be 1")
        if self.n_features:
            self.n_features = min(X.shape[1], self.n_features)
        else:
            self.n_features = X.shape[1]
        self.root = self.__grow_tree(X, y)

    def __grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_features)
            Target vector (labels).
        depth : int
            The present heigh of node according to tree (or the deeply node location in tree).

        Attributes
        ----------
        n_labels : array-like of shape (n_labels)
            The labels tag, where `n_labels` is unique of target vector y.
        feature_idxs : array-like of shape (n_features of node, n_features)
            The index of labels in node.
        best_feature : np.ndarray
            Optimzize feature index split.
        best_threshold : np.ndarray
            Optimzize threshold split.
        left_idxs : np.ndarray
            The index of node belower the threshold.
        right_idx : np.ndarray
            The index of node higher the threshold.
        left : Node left.
        right : Node right.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_node = self.__most_common_label(y)
            return Node(value=leaf_node)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self.__best_split(X, y, feature_idxs)

        left_idxs, right_idxs = self.__split(X[:, best_feature], best_threshold)
        left = self.__grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.__grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def __best_split(self, X: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray):
        """
        Compare the gain to get opimize value

        Parameters
        ----------
        X : np.ndarray
            The training vector of each node.
        y : np.ndarray
            The target vector of each node.
        feature_idxs : np.ndarray
            Index of features in node.

        Attributes
        ----------
        best_gain : int or float
            The temp value to check the highest optimize

        Return
        ------
        split_idx : int or float    
            The split index when gain is highest
        split_threshold
            The split threshold when gain is highest
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.__information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def __information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: Union[int, float]):
        """
        Calculate the information gain of decision tree to get value of two condition

        Parameters
        ----------
        y : np.ndarray
            Target vector.
        X_column : np.ndarray
            Comparing vector.
        threshold : int or float
            Threshold to divide left and right indexes
        
        Attributes
        ----------
        n_l : int
            len of samples in left side
        n_r : int
            len of samples in right side
        e_l : float
            entropy of left side
        e_r : float
            entropy of right side
        
        Return
        ------
        information_gain : float
            gnini score
        """
        parent_entropy = self.__entropy(y)

        left_idxs, right_idxs = self.__split(X_column, threshold)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.__entropy(y[left_idxs]), self.__entropy(y[right_idxs])
        children_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy - children_entropy


    def __split(self, X_column: np.ndarray, threshold: Union[int, float]):
        """
        Split the left side and right side from threshold

        Parameters
        ----------
        X_column : np.ndarray
            The samples
        threshold : int or float
            Threshold to devide left or right

        Return
        ------
        left_idxs : np.ndarray
            The index of samples in left side (below or equal than threshold)
        right_idxs : np.ndarray
            The index of samples in right side (high than therhold)
        """
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def __entropy(self, y: np.ndarray):
        """
        Calculate the entropy

        Parameters
        ----------
        y : np.ndarray
            The target want to know the apperent
        
        Attributes
        hist : np.ndarray
            The phrequency of each value in y
        ps : np.ndarray
            p(x)
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])

    def __most_common_label(self, y: np.ndarray):
        """
        Take the best phrequency of label

        Parameters
        ----------
        y : np.ndarray
            The target
        
        Return
        ------
        return the value of best phrequency
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X_test: Union[np.ndarray, List[List]]):
        """
        Predict class or regression value for X_test.

        Parameters
        ----------
        X_test : np.ndarray
            vector of testign
        
        Return
        ------
        Labels of testing through model
        """
        X_test = np.array(X_test)
        assert np.ndim(X_test)==2, Exception("ndim of X_test must be 2")
        return np.array([self.__traversal(x, self.root) for x in X_test])

    def __traversal(self, x: np.ndarray, node):
        """
        Traversal each node to find suitable position for X_train

        Parameters
        ----------
        x : int or float
            Each value of X_test
        """
        if node.is_leaf_node():
            return node.value
        if x[node.features] <= node.threshold:
            return self.__traversal(x, node.left)
        return self.__traversal(x, node.right)