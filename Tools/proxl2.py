import numpy as np

def proxl2(x,y,eta):
    """
    projection onto the l2 ball
    """
    t = x - y
    s = t * min(eta/np.linalg.norm(t,2),1)
    z = x + s -t
    return z
