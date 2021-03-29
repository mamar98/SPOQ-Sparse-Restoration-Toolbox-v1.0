import numpy as np
from numba import njit

@njit(cache=True)
def Lpsmooth(x,alpha,p):
    """
    This function computes the smooth Lp norm of the vector x
    """
    Lp = (np.sum((x**2 + alpha**2) ** (p/2) -alpha**p))**(1/p)
    return Lp
