
# import library numpy to optimize array calculations performance
import numpy as np

def ECDFFunc(Data):
    n = len(np.unique(Data))
    y = np.sort(np.unique(Data))
    x = np.arange(1,n+1)/n
    Ny = np.zeros(len(y)+1)
    Nx = np.zeros(len(y)+1)
    Ny[0] = y[0]
    Nx[0] = 0
    Ny [1:] = y
    Nx[1:] = x
    return Nx, Ny