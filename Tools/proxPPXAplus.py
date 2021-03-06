import numpy as np

from Tools.proxB import proxB
from Tools.proxl2 import proxl2
from numba import njit

@njit(cache=True)
def proxPPXAplus(D, B, x, y, eta, J, prec):
    # This function computes the proximity operator 
    # using the PPXA+ algorithm

    N = D.shape[1]

    x1k_old = x.copy()
    x2k_old = D @ x1k_old

    A = np.linalg.inv(np.eye(N) + D.T @ D)
    zk_old = A @ (x1k_old + D.T @ x2k_old)

    teta = 1.9

    for j in range(J):
        y1k_old = proxB(B, x1k_old, x, teta)
        y2k_old = proxl2(x2k_old, y, eta)

        vk_old = A @ (y1k_old + D.T @ y2k_old)

        x1k = x1k_old + 2 * vk_old - zk_old - y1k_old
        x2k = x2k_old + D @ (2*vk_old - zk_old) - y2k_old
        zk = vk_old

        error = np.linalg.norm(zk - zk_old) ** 2

        if error < prec:
            print('PPXA stops at j = ', j)
            break

        x1k_old = x1k
        x2k_old = x2k
        zk_old = zk

    return zk, j
