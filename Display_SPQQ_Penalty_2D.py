import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------- Functions  -------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def lqnorm(X,Y,q):
    lq = ((X**2 + Y**2)**(1/q))
    return lq


def lpnorm(X,Y,p):
    lp = (abs(X)**p+abs(Y)**p)
    return lp


def lplqnorm(x,alpha,beta,eta,p,q):
    normx = np.zeros(x.shape)
    for i in range(0,len(x[0])):
        for j in range(0,len(x[1])):
            lp = ((x[i,j]**2 + alpha**2)**(p/2) - alpha**p)**(1/p)
            lpalpha = (lp**p +beta**p)**(1/p) 
            lq = (eta**q + abs(x[i,j])**q)**(1/q)
            normx[i,j] = lpalpha/lq
    normxx=normx
    return normxx


def Ind3(x,w):
    if x <= w:
        y = 1
    else:
        y = 0
    return y

def Display_SPOQ_Penalty_2D():
    x = np.arange(-1,1,0.005)
    x=np.transpose(x)
    y = np.arange(-1,1,0.005)
    y=np.transpose(y)
    [X,Y] = np.meshgrid(x,y)

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %----------------------------- l0 count measure  -------------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    Xl0 = np.ones(X.shape)
    Yl0 = np.ones(Y.shape)
    Xl0[X==0] = 0
    Yl0[Y==0] = 0
    Zl0 = Xl0+Yl0
    maxi=np.nanmax(Zl0)
    Zl0 = Zl0/maxi

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %----------------------------- l0.75/l2 smoothed quasi/norm ratio  -------------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
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

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %----------------------------- Figure  -------------------------------%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    fig = plt.figure(figsize=plt.figaspect(0.5)) # set up a figure twice as wide as it is tall
    #===============
    #  First subplot
    #===============
    # set up the axes for the first plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X,Y,Zl0, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax1.set_title('$\ell_0$ count measure')
    #===============
    # Second subplot
    #===============
    # set up the axes for the second plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2= ax2.plot_surface(X,Y,Zlplq, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)
    ax2.set_title('Smoothed $\ell_{3/4}$-over-$\ell_2$ quasinorm ratio')
  

Display_SPOQ_Penalty_2D()
plt.show()