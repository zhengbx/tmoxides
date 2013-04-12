import numpy as np
import matplotlib.pyplot as plt
#import pylab

data = np.load("honeycombLn=12nImp=6l1.npy")
#data = np.load("dmet_resultLn=24nImp=61.npy")
length,numPara = data.shape

plt.figure(1)
plt.subplot(211)
plt.plot(data[:,0],data[:,4],'bo',markersize=8)
plt.subplot(212)
plt.plot(data[:,0],data[:,3],'bo',markersize=8)

plt.figure(2)
plt.plot(data[:,0],data[:,1])
#plt.figure(3)
#plt.plot(data[:,0],data[:,2])


data = np.load("dmet_resultLn=24nImp=62.npy")
length,numPara = data.shape

plt.figure(1)
plt.subplot(211)
plt.plot(data[:,0],data[:,4],'ro',markersize=8)
plt.subplot(212)
plt.plot(data[:,0],data[:,3],'ro',markersize=8)

plt.figure(2)
plt.plot(data[:,0],data[:,1])
#plt.figure(3)
#plt.plot(data[:,0],data[:,2])













plt.show()


