from numba import njit

@njit(cache=True)
def Ind3(x,w):
    if x <= w:
        y = 1
    else:
        y = 0
    return y