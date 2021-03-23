import numpy as np

def proxl1(x,w):
    """
    proximity operator of l1 norm: Thresholding y = max(abs(x)-w,0).*sign(x)
    """
    p = np.zeros(x.shape)
    p[x>w] = x[x>w] -w
    p[x<w] = x[x<w] +w
    p[p<0] = 0
    return p
