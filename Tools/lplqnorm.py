import numpy as np

def lplqnorm(x,alpha,beta,eta,p,q):
    smallnb = 1/(2**32)
    normx = np.zeros(x.shape)
    for i in range(0,len(x[0])):
        for j in range(0,len(x[1])):
            lp = ((max(smallnb, x[i,j]**2) + alpha**2) **(p/2) - alpha**p)**(1/p)
            #lp = ((x[i,j]**2 + alpha**2)**(p/2) - alpha**p)**(1/p)
            lpalpha = (lp**p +beta**p)**(1/p) 
            lq = (eta**q + abs(x[i,j])**q)**(1/q)
            normx[i,j] = lpalpha/lq
    normxx=normx
    return normxx