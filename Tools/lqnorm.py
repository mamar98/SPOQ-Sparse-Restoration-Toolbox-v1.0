from numba import njit

@njit(cache=True)
def lqnorm(X,Y,q):
    lq = ((X**2 + Y**2)**(1/q))
    return lq