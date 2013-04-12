import pickle as p
import matplotlib.pyplot as plt

resultfile = open("honeycombl1Ln=12nImp=6.pickle", "r")
data = p.load(resultfile)

plt.figure(1)
plt.subplot(211)
plt.plot(data["U"],data["AFOrder"],'bo',markersize=8)
plt.subplot(212)
plt.plot(data["U"],data["Gap"],'bo',markersize=8)

plt.figure(2)
plt.plot(data["U"],data["E"])
#plt.figure(3)
#plt.plot(data[:,0],data[:,2])

"""
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
"""

plt.show()


