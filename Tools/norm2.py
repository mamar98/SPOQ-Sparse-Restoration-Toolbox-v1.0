import numpy as np
from random import random

def norm2(K,N):
    nbiter = 50
    b = np.random.rand(N)
    for i in range(nbiter):
        tmp = K*(K*b)
        tmpnorm = np.linalg.norm(tmp, ord=2)
        b = tmp/tmpnorm
    norm2K = np.linalg.norm(K*b, ord = 2)
    return norm2K