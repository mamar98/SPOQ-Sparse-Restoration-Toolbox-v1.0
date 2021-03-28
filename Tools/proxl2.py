import numpy as np

def proxl2(x,y,eta):
    """
    projection onto the l2 ball
    """
    t = x - y
    s = t * min(1, eta/np.linalg.norm(t))
    z = x + s - t
    return z
