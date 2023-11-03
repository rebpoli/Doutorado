
# import library numpy to optimize array calculations performance
import numpy as np

 
def PercentileFunc(x,P):
    n = len(x)
    y = np.sort(x)
    return(np.interp((P/100), np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))