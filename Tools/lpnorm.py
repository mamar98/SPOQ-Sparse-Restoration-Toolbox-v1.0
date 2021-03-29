from numba import njit

@njit(cache=True)
def lpnorm(X,Y,p):
    lp = (abs(X)**p+abs(Y)**p)
    return lp