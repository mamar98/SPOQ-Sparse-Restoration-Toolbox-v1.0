###########################################################################
## SPOQ: Smooth lp-Over-lq ratio
### This code implements the SPOQ regularization function presented in
### "SPOQ $\ell_p$-Over-$\ell_q$ Regularization for Sparse Signal: Recovery 
### applied to Mass Spectrometry"
### IEEE Transactions on Signal Processing, 2020, Volume 68, pages 6070--6084
### Afef Cherni, IEEE member, 
### Emilie Chouzenoux, IEEE Member,
### Laurent Duval, IEEE Member,
### Jean-Christophe Pesquet, IEEE Fellow
### https://arxiv.org/abs/2001.08496
### https://doi.org/10.1109/TSP.2020.3025731
###########################################################################

from time import time as current_time
import numpy as np
import matplotlib.pyplot as plt
from Tools.FB_PPXALpLq import FB_PPXALpLq

def Run_SPOQ_Recovery(K,y,p,q,alpha,beta,eta,xi,nbiter,xtrue):
    ## SPOQ recovery
    print('Running TR-VMFB algorithm on SPOQ penalty with p = ', str(p), ' and q = ', str(q))
    tic = current_time()
    xrec, fcost, Bwhile, time, mysnr = FB_PPXALpLq(K,y,p,q,2,alpha,beta,eta,xi,nbiter,xtrue)
    tf= tic - current_time()
    text=['Reconstruction in ', str(len(time)), ' iterations']
    print(text)
    text=['SNR = ', str(-10*np.log10(np.sum((xtrue-xrec)**2)/np.sum(xtrue**2)))]
    print(text)
    text=['Reconstruction time is ', str(sum(time)), 'seconds']
    print(text)
    print('_____________________________________________________')

    ## Results
    fig1 = plt.subplot(1, 1, 1)
    fig1.plot(xtrue, 'r-o')
    fig1.plot(xrec, 'b--')
    plt.tight_layout()
    fig1.grid()
    fig1.legend(["Original signal", "Estimated signal"])
    fig1.set_title("Reconstruction results")
    fig2 = plt.subplot(1, 1, 1)
    fig2.plot(np.insert(np.cumsum(time), 0, 0), mysnr, '-k', 'linewidth', 2)
    plt.tight_layout()
    fig1.grid()
    fig2.set_title("Algorithm convergence")
    fig2.set_xlabel("Time (s)")
    fig2.set_ylabel("SNR (dB)")
    fig2.legend(['TR-VMFB'])