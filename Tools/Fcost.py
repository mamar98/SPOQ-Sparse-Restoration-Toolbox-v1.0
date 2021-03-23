import numpy as np

def Fcost(x,alpha,beta,mu,p,q):
    lp = (np.sum((x** 2 + alpha**2)**(p/2))  - alpha**p) **(1/q)
    lq = (mu**q + np.sum(abs(x)**q))**(1/q)
    fcost = np.log(((lp**p + beta ** p) **(1/q)) / lq)
    return fcost