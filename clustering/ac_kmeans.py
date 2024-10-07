import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Chooses random data points for initial centroids
def initialize_centroids(K, X):
    m, n = np.shape(X)
    centroids = np.empty((K,n))
    for i in range(K):
        centroids[i] = X[np.random.choice(range(m))]
    return centroids

#Euclidean distance between two vectors
def euclidean_distance(v1,v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))

#Find which points are closest to what centroid
def closest_centroid(K, centroids, x):
    distances = np.empty(K)
    for i in range(K):
        distances[i] = euclidean_distance(centroids[i], x)
    return np.argmin(distances)

def create_clusters(K, centroids, X):
    N, _ = np.shape(X)
    cluster_indexes = np.empty(N)
    for i in range(N):
        cluster_indexes[i] = closest_centroid(K, centroids, X[i])
    return cluster_indexes

def cluster_centroid_means(K, cluster_indexes, X):
    _, M = np.shape(X)
    centroids = np.empty((K,M))
    for i in range(K):
        points = X[cluster_indexes == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


#Access file and extract data
file = pd.read_csv('P2_CLUSTER2.csv')
X = file.values
N, M = np.shape(X)

#Data is all clean no need to clean data

#Visualize data before choosing K
plt.subplot(2,1,1)
plt.plot(X[:,0], X[:,1], "bo")
plt.show()

K = 2 #Choose K

#Code
centroids = initialize_centroids(K, X)
while(True):
    cluster_indexes = create_clusters(K, centroids, X)
    previous_centroids = centroids
    centroids = cluster_centroid_means(K, cluster_indexes, X)
    diff = previous_centroids - centroids
    if not diff.any():
        break

