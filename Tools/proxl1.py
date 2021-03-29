import numpy as np
from numba import njit

@njit(cache=True)
def proxl1(x,w):
    """
    proximity operator of l1 norm: Thresholding y = max(abs(x)-w,0).*sign(x)
    """
    p = np.zeros(x.shape)
    # Pour que numba fonctionne
    x_view = x.ravel()
    p_view = p.ravel()
    p_view[x_view>w] = x_view[x_view>w] -w
    p_view[x_view<w] = x_view[x_view<w] +w
    p_view[p_view<0] = 0
    return p
