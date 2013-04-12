import pylab as pl
import numpy as np



filename = "iApbcLn=24dU=01nImp=4-.dat"
rawdata = pl.fromfile(filename, dtype=np.float, count=-1, sep=' ' )

pl.figure(1)# Energy
pl.plot(rawdata[::9],rawdata[1::9])

pl.figure(2)# double occupancy
pl.plot(rawdata[::9],rawdata[2::9])

pl.figure(3)# Mu
pl.plot(rawdata[::9],rawdata[3::9])

pl.figure(4)# HlGap
pl.plot(rawdata[::9],rawdata[4::9])

pl.figure(5) # AForder
pl.plot(rawdata[::9],rawdata[5::9])


data = np.load("honeycombLn=12nImp=4l1.npy")

pl.figure(1)# Energy
pl.plot(data[:,0],data[:,1]+1,'o',markersize=8)

pl.figure(3)# Mu
pl.plot(data[:,0],data[:,2],'o',markersize=8)

pl.figure(4)# HlGap
pl.plot(data[:,0],data[:,3],'o',markersize=8)

pl.figure(5) # AForder
pl.plot(data[:,0],data[:,4],'o',markersize=8)


pl.figure(1)# Energy
pl.legend(('old', 'lattice dmet'))
pl.xlabel('U')
pl.ylabel('Energy/2')

pl.figure(3)# Mu
pl.legend(('old', 'lattice dmet'))

pl.figure(4)# HlGap
pl.legend(('old', 'lattice dmet'))

pl.figure(5) # AForder
pl.legend(('old', 'lattice dmet'))

pl.show()
