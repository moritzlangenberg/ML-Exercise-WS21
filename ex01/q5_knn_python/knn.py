import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created
    N = len(samples)

    pos = np.arange(-5.0, 5.0, 0.1)
    den = np.zeros(len(pos))

    # case of 1 dimension, 
    for i in range(len(pos)):
        x = pos[i]
        # the distance to each point
        dist = np.abs(x-samples)
        # find indexes of k smallest values in dist
        idx = np.argpartition(dist, k)
        # volume is 2* dist to kth nearest point
        V = 2*dist[idx[k-1]]

        den[i] = k/(N*V)

    estDensity = np.stack((pos, den), axis=1)    
    return estDensity
