<h1 align="center"> Machine Learning Library from Scratch with Python</h1>

## 👋 Introduction

In this project, I will "re-built from scratch" state-of-the-art algorithms in ML/DL, but using only Python language. I will use **NumPy library** for better performance on matrix calculation.

One thing you need to know that, I make this project **just for learning and understanding algorithms deeply**, not suitable to apply in real-world problems. I still recommend using many other SOTA libraries for building models.


## 👤 Contributors
**I would like to express my sincere thanks to the these wonderful people who have contributed to this library with me:**

<a href="https://github.com/AI-Coffee/mllib-from-scratch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI-Coffee/mllib-from-scratch" />
</a>


## 📝 TODO Lists

### A. Classification
1. [**Logistic Regression**](classification/LogisticRegression.py)
    - [x] Initialize algorithm.
    - [x] Write document-in-code.
2. Naive Bayes
3. [**K-Nearest Neighbors (KNN)**](classification/KNN.py)
    - [x] Initialize naive algorithm for KNN.
    - [ ] Write document-in-code.
    - [ ] Optimize alogirthm with better method of storing data: `BallTree` and  `KDTree`.
4. Decision Tree
5. Support Vector Machine (SVM)
6. Random Forest
7. Softmax Regression

### B. Regression
1. [**Linear Regression**](regression/LinearRegression.py)
      - [x] Update `fit()` and `predict()` method, add `intercept` and `coefficient` attribute.
      - [x] Write document-in-code.
2. Ridge Regression
3. Lasso Regression
4. Decision Tree for Regression
5. Random Forest for Regression
6. K-Nearest Neighbors for Regression
7. Support Vector Regression
8. Gaussian Regression
9. Polynomial Regression

### C. Clustering
1. [**K-Means**](cluster/KMeans.py)
    - [x] Initialize algorithm with naive brute-force method.
    - [x] Optimize initialized centroids method with `K-Means++`
    - [x] Write document-in-code
2. DBSCAN
3. Mean Shift
4. OPTICS
5. Spectral Clustering
6. Mixture of Gaussians
7. Affinity Propagation
8. Agglomerative Clustering
9. BIRCH

### D. Dimensionality reduction
1. [**Principal Components Analysis (PCA)**](decomposition/PCA.py)
    - [x] Initialize algorithm
2. Factor Analysis (FA)
3. Linear Discriminant Analysis (LDA)
4. Truncated SVD
5. Kernel PCA
6. t-Distributed Stochastic Neighbor Embedding (t-SNE)
7. Multidimensional Scaling (MDS)
8. Isomap