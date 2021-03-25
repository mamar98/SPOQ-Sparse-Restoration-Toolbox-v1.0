"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SPOQ: Smooth lp-Over-lq ratio
%%% This code implements the SPOQ regularization function presented in
%%% "SPOQ $\ell_p$-Over-$\ell_q$ Regularization for Sparse Signal: Recovery 
%%% applied to Mass Spectrometry"
%%% IEEE Transactions on Signal Processing, 2020, Volume 68, pages 6070--6084
%%% Afef Cherni, IEEE member, 
%%% Emilie Chouzenoux, IEEE Member,
%%% Laurent Duval, IEEE Member,
%%% Jean-Christophe Pesquet, IEEE Fellow
%%% https://arxiv.org/abs/2001.08496
%%% https://doi.org/10.1109/TSP.2020.3025731
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Created by Laurent Duval 16/03/2021

%% Random location and amplitude peaks convolved with a finite support
%% "Gaussians" from binomial coefficients
"""

import numpy as np
import os
import sys
from random import randint, uniform
from scipy.linalg import pascal, toeplitz
from math import sqrt
from Present_SPOQ_Data_Information import Present_SPOQ_Data_Information
from Run_SPOQ_Recovery import Run_SPOQ_Recovery

sys.path.append("./Data/")
sys.path.append("./Tools/")

#INITIALISATION
print('_____________________________________________________')
print('...loading data ...')
nSample = 500
nPeak = 20 
peakWidth = 5

xtrue = np.zeros((nSample,1))
xtrueLocation=np.random.permutation(nSample)[:nPeak] #row vector containing nPeak unique integers selected randomly from 1 to nSample
xtrueAmplitude = np.random.rand(nPeak, 1) # vector of nPeak lenght with values beetween 0 and 1 
xtrue[xtrueLocation] = xtrueAmplitude

peakMatrix = pascal(peakWidth)
peakShape = np.diag(np.fliplr(peakMatrix))
peakShape = np.array(peakShape/np.sum(peakShape))
peakShapeFilled = np.concatenate((peakShape, np.zeros(nSample - peakWidth)))
peakShapeFilledtranspose = np.transpose(peakShapeFilled)

"""matlab
K = toeplitz([peakShapeFilled(1) fliplr(peakShapeFilled(2:end))], peakShapeFilled);
"""
#K = toeplitz(

y = K*xtrue

# ADD the gaussian noise with a standard deviation sigma
noise = np.array([uniform(0, 1) for i in range(nSample)])
sigma = 0.5 *np.max(y)/100
y = y + sigma*noise

# CHOOSE SPOQ PARAMETERS
xi = 1.1*sqrt(nSample)*sigma
eta = 2E-6
alpha = 7E-7
beta = 3E-2
p = 0.75
q = 2
nbiter=5000

# Verifiy/dispaly data information
Present_SPOQ_Data_Information(xtrue,K,y,noise)

#Run SPOQ Recovery
Run_SPOQ_Recovery(K,y,p,q,alpha,beta,eta,xi,nbiter,xtrue)


