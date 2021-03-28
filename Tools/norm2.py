import numpy as np

def norm2(K,N):
    nbiter = 50
    b = np.random.rand(N,1)
    for i in range(nbiter):
        tmp = K.T @ (K @ b)
        tmpnorm = np.linalg.norm(tmp)
        b = tmp/tmpnorm
    norm2K = np.linalg.norm(K @ b)
    return norm2K