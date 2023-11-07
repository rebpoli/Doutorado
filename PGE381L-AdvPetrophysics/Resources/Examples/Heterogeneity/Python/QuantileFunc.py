
# import library numpy to optimize array calculations performance
import numpy as np

def QuantileFunc(x,Q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(Q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))