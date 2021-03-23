import numpy as np

def Lqsmooth(x,mu,q):
    """
    This function computes the smooth Lq norm of the vector x
    """
    lqx = (mu**q + np.sum(abs(x)**q))**(1/q)
    return lqx
