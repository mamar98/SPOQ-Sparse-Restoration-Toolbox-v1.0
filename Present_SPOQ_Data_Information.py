import matplotlib.pyplot as plt
import numpy as np

def Present_SPOQ_Data_Information(xtrue,K,y,N,noise):
    print('The size of your original data is : ',xtrue.shape)
    print('The size of your observation operator is : ',K.shape)
    print('The size of your measurement data is : ',y.shape)

    sample = list(range(N))

    plt.figure(1,figsize=(20,20))

    #Xtrue
    plt.subplot(2,2,1)
    plt.plot(sample,xtrue,'-b')
    plt.ylim([-0.05*np.max(xtrue),1.1*max(xtrue)])
    plt.xlabel('Samples')
    plt.title('Original signal x')
    plt.grid(True)

    #K
    plt.subplot(2,2,2)
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.contourf(K)
    plt.gca().invert_yaxis()
    plt.title('Measurement matrix K')


    #Noise
    plt.subplot(2,2,3)
    plt.plot(sample,noise,'-k')
    plt.grid(True)
    plt.title('Noise')
    plt.xlabel('Samples')

    #Obs y
    plt.subplot(2,2,4)
    plt.plot(sample,y,'-r')
    plt.title('Observation y')
    plt.xlabel('Sample')
    plt.ylim(-0.05*np.max(y),1.1*np.max(y))
    plt.grid(True)
    plt.show()
