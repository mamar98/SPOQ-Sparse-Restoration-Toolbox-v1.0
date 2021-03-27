import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Tools.lplqnorm import lplqnorm
from Tools.lqnorm import lqnorm
from Tools.lplqnorm import lplqnorm
from Tools.Ind3 import Ind3


def Display_SPOQ_Penalty_2D():
    x = np.arange(-1,1,0.005)
    x=np.transpose(x)
    y = np.arange(-1,1,0.005)
    y=np.transpose(y)
    [X,Y] = np.meshgrid(x,y)

    Xl0 = np.ones(X.shape)
    Yl0 = np.ones(Y.shape)
    Xl0[X==0] = 0
    Yl0[Y==0] = 0
    Zl0 = Xl0+Yl0
    Zl0 = Zl0/(np.nanmax(Zl0))


    ###################################################################################################
    #######                              l0.75/l2 smoothed quasi/norm ratio                     ####### 
    ###################################################################################################
    p = 0.75
    q = 2
    alpha = 7e-7
    beta = 3e-3
    eta = 0.1
    Zlplq =  lplqnorm(X,alpha,beta,eta,p,q) + lplqnorm(Y,alpha,beta,eta,p,q)
    Zlplq = Zlplq/np.nanmax(Zlplq)
    ZlplqSPOQ = np.log(Zlplq)
    ZlplqSPOQ = (ZlplqSPOQ - np.nanmin(ZlplqSPOQ))/(np.nanmax(ZlplqSPOQ)-np.nanmin(ZlplqSPOQ))
    Zlplq = np.nan_to_num(Zlplq) 
    ZlplqSPOQ = np.nan_to_num(ZlplqSPOQ)


    ###################################################################################################
    #######                              Figure                                                 ####### 
    ###################################################################################################
    fig = plt.figure(figsize=plt.figaspect(0.5)) # set up a figure twice as wide as it is tall
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X,Y,Zl0, cmap=cm.coolwarm)
    fig.colorbar(surf)
    ax1.set_title('$\ell_0$ count measure')
    #===============
    # Second subplot
    #===============
    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2= ax2.plot_surface(X,Y,Zlplq, cmap=cm.coolwarm)
    fig.colorbar(surf2)
    ax2.set_title('Smoothed $\ell_{3/4}$-over-$\ell_2$ quasinorm ratio')
    
    plt.show()

Display_SPOQ_Penalty_2D()
