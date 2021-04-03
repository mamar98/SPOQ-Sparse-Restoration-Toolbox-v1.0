import matplotlib.pyplot as plt
import numpy as np

from Present_SPOQ_Data_Information import Present_SPOQ_Data_Information
from Run_SPOQ_Recovery import Run_SPOQ_Recovery

plt.clf()


print('-------------------------')
print('... loading data...')

xtrue = input('Provide the path for the original sparse signal : ')
try:
    xtrue = np.loadtxt(xtrue)
except OSError:
    print("Le fichier que vous avez indiqué n'est pas bien indiqué, la fonction va s'arrêter.")

K = input('Provide the path for the measurement operator : ')
try:
    K = np.loadtxt(K)
except OSError:
    print("Le fichier que vous avez indiqué n'est pas bien indiqué, la fonction va s'arrêter.")
    
noise = input('Provide the path for the noise : ')
try:
    noise = np.loadtxt(noise)
except OSError:
    print("Le fichier que vous avez indiqué n'est pas bien indiqué, la fonction va s'arrêter.")

y = np.dot(K,xtrue)

sigma = (0.1*np.max(y))/100

y = y+(sigma*noise)

N = len(xtrue)
xi = 1.1*np.sqrt(N)*sigma
eta = 2e-06
alpha = 7e-7
beta = 3e-2
p = 0.75
q = 2
nbiter = 5000

Present_SPOQ_Data_Information(xtrue,K,y,N,noise) #Run la fonction en question

Run_SPOQ_Recovery(K,y,p,q,alpha,beta,eta,xi,nbiter,xtrue)"""
