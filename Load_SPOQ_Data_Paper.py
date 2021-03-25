import matplotlib.pyplot as plt
import os
import numpy as np

from Present_SPOQ_Data_Information import Present_SPOQ_Data_Information

plt.clf()

dir_path = os.path.dirname(os.path.realpath(__file__))
data = dir_path+'/Data'
tools = dir_path+'/Tools'

print('-------------------------')
print('... loading data...')

xtrue = np.loadtxt(data+'/x.txt')
K = np.loadtxt(data+'/K.txt')
y = np.dot(K,xtrue)

sigma = (0.1*np.max(y))/100

noise = np.loadtxt(data+'/noise.txt')

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

#Run_SPOQ_Recovery()
