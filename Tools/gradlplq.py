import numpy as np
from Tools.Lpsmooth import Lpsmooth
from Tools.Lqsmooth import Lqsmooth
from numba import njit

@njit(cache=True)
def gradlplq(x,alpha,beta,mu,p,q):
    """
    This function computes the gradient of smooth lp over lq function
    """
    lp = Lpsmooth(x,alpha,p)
    lq = Lqsmooth(x,mu,q)
    grad1 = x *((x**2 + alpha**2)**(p/2-1))/(lp**p + beta**p)
    grad2 = np.sign(x)*(np.abs(x))**(q -1) / (lq**q)
    gradphi = grad1 - grad2
    return gradphi