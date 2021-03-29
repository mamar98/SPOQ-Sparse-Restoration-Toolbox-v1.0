import numpy as np

from Tools.norm2 import norm2
from Tools.proxl1 import proxl1
from Tools.proxl2 import proxl2
from numba import njit

@njit(cache=True)
def pds(K,y,eta,nbiter):

    M, N = K.shape[0], K.shape[1]
    normK = norm2(K,N)
    tau = 1/normK
    # sigma = 0.9/(tau*(normK**2))   c'était écrit comme ça dans la version Matlab
    sigma = 0.9/normK
    ro = 1
    refspec = np.zeros((nbiter,1))
    xk_old = np.ones((N,1))
    uk_old = K @ xk_old
    prec = 1e-6

    for i in range(nbiter):
        xxk = proxl1(xk_old - tau * K.T @ uk_old, tau)

        zk = uk_old + sigma * K @ (2*xxk - xk_old)
        
        uuk = zk - sigma * proxl2(zk/sigma, y, eta)
        
        xk = xk_old + ro*(xxk - xk_old)
        uk = uk_old + ro*(uuk - uk_old)
        
        ex = np.linalg.norm(xk - xk_old)**2 / np.linalg.norm(xk)**2
        eu = np.linalg.norm(uk - uk_old)**2 / np.linalg.norm(uk)**2
        
        if ex < prec and eu < prec:
            break

        refspec[i] = ex
        xk_old = xk
        uk_old = uk    

    return xk, refspec