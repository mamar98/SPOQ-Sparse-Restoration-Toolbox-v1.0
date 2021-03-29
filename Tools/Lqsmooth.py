import numpy as np
from numba import njit

@njit(cache=True)
def Lqsmooth(x,mu,q):
    """
    This function computes the smooth Lq norm of the vector x
    """
    lqx = (mu**q + np.sum(np.abs(x)**q))**(1/q)
    return lqx
