import numpy as np 
from time import time

from Tools.pds import pds
from Tools.Fcost import Fcost
from Tools.condlplq import condlplq
from Tools.gradlplq import gradlplq
from Tools.ComputeLipschitz import ComputeLipschitz
from Tools.proxPPXAplus import proxPPXAplus

def FB_PPXALpLq(K, y, p, q, metric, alpha, beta, eta, xi, nbiter, xtrue):
    # This function defines the Trust region algorihtm based on
    # Forward-Backward algorithm
    
    # Initialization
    N = K.shape[1]

    Bwhile = []
    Time  = []
    fcost = []
    mysnr = []

    xk_old, _ = pds(K,y,xi,10)
    mysnr.append( -10 * np.log10( np.sum((xk_old-xtrue)**2) / np.sum(xtrue**2)) )
    fcost.append( Fcost(xk_old, alpha, beta, eta, p, q) )

    gamma = 1
    prec = 1e-12
    J = 5000                                        # ppxa max iterations
    # bmax = 10                                     # maximum TR iterates
    
    L = ComputeLipschitz(alpha, beta, eta, p, q, N)
    
    # Algorithm   
    for k in range(nbiter):

        if (k%100==0):
            print('it={} : fcost {}'.format( k, fcost[k] ))
        
        start = time()

        # metric 0: Lip constant, 1: FBVM without TR, 2: FBVM-TR
        if metric == 0:      
            A = np.matmul(L,np.ones(N,1))
            B = A/gamma
            xxk = xk_old - (1/B) * gradlplq(xk_old, alpha, beta, eta, p, q)
            xk = proxPPXAplus(K, B, xxk, y, xi, J, prec)
        
        elif metric == 1: 
            A = condlplq(xk_old, alpha, beta, eta, p, q, 0)
            B = A/gamma
            xxk = xk_old - (1/B) * gradlplq(xk_old, alpha, beta, eta, p, q)
            xk = proxPPXAplus(K, B, xxk, y, xi, J, prec)
        
        elif metric ==2:     
                ro = np.sum( np.abs(xk_old**q) )**(1/q)
                bwhile = 0
                while 1 :
                    A = condlplq(xk_old, alpha, beta, eta, p, q, ro)
                    B = A/gamma
                    xxk = xk_old - (1/B) * gradlplq(xk_old, alpha, beta, eta, p, q)
                    xk = proxPPXAplus(K, B, xxk, y, xi, J, prec)[0]

                    if np.sum( np.abs(xk)**q )**(1/q) < ro: 
                        ro = ro/2
                        bwhile = bwhile + 1
                    else:
                        break

                Bwhile.append(bwhile)
            
        Time.append(time() - start)
        
        error = np.linalg.norm(xk-xk_old)**2 / np.linalg.norm(xk_old)**2
        mysnr.append( -10*np.log10(np.sum((xk-xtrue)**2) / np.sum(xtrue**2)) )
        fcost.append( Fcost(xk,alpha,beta,eta,p,q) )

        if error < prec:
            break
        
        xk_old = xk

    return xk, np.array(fcost), Bwhile, Time, np.array(mysnr)