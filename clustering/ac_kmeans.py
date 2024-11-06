import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Chooses random data points for initial centroids
def initialize_centroids(K, X):
    l, c = np.shape(X)
    centroids = np.empty((K,c))
    for i in range(K):
        centroids[i] = X[np.random.choice(range(l))]
    return centroids

#Euclidean distance between two vectors
def euclidean_distance(v1,v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))

#Calculate distance of a point to each centroid
def closest_centroid(K, centroids, x):
    distances = np.empty(K)
    for i in range(K):
        distances[i] = euclidean_distance(centroids[i], x)
    return np.argmin(distances)

#Create clusters using 'closest_centroid' function
def create_clusters(K, centroids, X):
    l, _ = np.shape(X)
    clusters = np.empty(l)
    for i in range(l):
        clusters[i] = closest_centroid(K, centroids, X[i])
    return clusters

#Calculate new centroids locations
def cluster_centroid_means(K, clusters, X):
    _, l = np.shape(X)
    centroids = np.empty((K,l))
    for i in range(K):
        points = X[clusters == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


#Access file and extract data
file = pd.read_csv('P2_CLUSTER4.csv')
X = file.values
c, l = np.shape(X)

#Data is all clean, no need to clean data

#Visualize data before choosing K
plt.subplot(1,1,1)
plt.plot(X[:,0], X[:,1], "bo")
plt.show()

#Choose K
K = 2

#Run code
centroids = initialize_centroids(K, X)
while(True):
    clusters = create_clusters(K, centroids, X)
    previous_centroids = centroids
    centroids = cluster_centroid_means(K, clusters, X)
    diff = previous_centroids - centroids
    if not diff.any():
        break

#Show clusters
for i in range(K):
    points = X[clusters == i]
    plt.scatter(points[:,0], points[:,1])
plt.plot(centroids[:][0],centroids[:][1],'r+')
plt.show()