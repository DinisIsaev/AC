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

#Access file and extract data
file = pd.read_csv('P2_CLUSTER1.csv')
X = file.values
N, M = np.shape(X)

#Data is all clean no need to clean data

#Visualize data before choosing K
plt.subplot(2,1,1)
plt.plot(X[:,0], X[:,1], "bo")
plt.show()

k = 1 #Choose K

#Code
initialize_centroids(k, X)
