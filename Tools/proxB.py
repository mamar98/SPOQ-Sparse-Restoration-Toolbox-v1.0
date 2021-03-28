import numpy as np
from numba import njit

@njit(cache=True)
def proxB(B,x,xhat,teta):
    """
    This function computes the proximity operators of f(x) = (teta/2) * ||y-x||_B^2
    """
    p = (x+teta*(B*xhat))/(1+teta*B)
    p_view = p.ravel()              # Ajout pour que njit fonctionne
    p_view[p_view<0] = 0
    return p