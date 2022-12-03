import numpy as np
from cluster.KMeans import KMeans as hm_kmeans
from sklearn.cluster import KMeans as sk_kmeans

X = np.array([[60, 50],
              [98, 34],
              [59, 20],
              [81,  6],
              [99, 98],
              [68, 12],
              [34, 83],
              [4, 89],
              [4, 72],
              [85, 21],
              [28, 35],
              [60, 12],
              [21, 38],
              [61,  2],
              [84,  8],
              [56, 37],
              [81, 23],
              [5, 18],
              [16, 57],
              [24, 65],
              [13, 13],
              [82, 92],
              [92, 42],
              [82, 52],
              [29, 36],
              [33, 42],
              [54, 35],
              [61, 59],
              [40, 73],
              [50, 51],
              [95, 67],
              [54, 88],
              [42, 18],
              [4,  5],
              [99, 27],
              [73, 94],
              [98, 73],
              [90, 51],
              [48, 44],
              [24, 88],
              [62, 11],
              [34, 99],
              [40, 46],
              [90, 43],
              [86, 72],
              [36, 45],
              [29, 36],
              [91, 76],
              [71, 37],
              [82, 70],
              [9, 11],
              [48,  9],
              [63, 69],
              [6, 83],
              [42, 11],
              [60, 23],
              [43, 28],
              [21, 15],
              [81, 85],
              [2, 45],
              [78, 67],
              [68, 27],
              [70, 83],
              [15, 88],
              [87, 90],
              [11, 89],
              [71, 91],
              [28, 48],
              [9, 72],
              [71, 70],
              [66, 83],
              [45,  9],
              [62, 99],
              [54, 32],
              [48, 11],
              [94, 55],
              [90, 17],
              [44, 23],
              [16, 97],
              [26, 71],
              [56,  9],
              [92, 15],
              [82, 96],
              [96, 56],
              [42,  9],
              [70, 32],
              [89,  2],
              [62, 46],
              [44,  8],
              [74, 76],
              [99, 47],
              [4, 79],
              [42, 51],
              [64, 71],
              [93, 54],
              [69, 31],
              [47, 74],
              [79,  6],
              [35, 96],
              [67,  5]])
# ---With K-Means++
kmeans = hm_kmeans(max_iter=100, n_clusters=3)
kmeans.fit(X, visualized=False)
print(kmeans._centroids)


# ---Without K-Means++
kmeans = hm_kmeans(init='random', max_iter=100, n_clusters=3)
kmeans.fit(X, visualized=False)
print(kmeans._centroids)


# ---With scikit-learn
def calc_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    subtracted_matrix = X - Y[:, np.newaxis]
    return np.sum(subtracted_matrix ** 2, axis=2) ** (1 / 2)


kmeans = sk_kmeans(max_iter=100, n_clusters=3)
kmeans.fit(X)
distances = calc_distance(X, kmeans.cluster_centers_)
loss = np.sum(np.min(distances, axis=0))
print(loss)
print(kmeans.cluster_centers_)
