import numpy as np
import math

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    N = len(samples)

    pos = np.arange(-5.0, 5.0, 0.1)
    den = np.zeros(len(pos))

    # case of 1 dimension
    D = 1
    for i in range(len(pos)):
        x = pos[i]
        p = 0
        for xn in samples:
            p += (1/(((2*np.pi)**(D/2))*h)) * math.exp(-np.dot(x-xn,x-xn)/(2*h**2))
        den[i] = p/N

    estDensity = np.stack((pos, den), axis=1)    
    return estDensity
