import numpy as np
from rham_La2CuO4_origin import Hopping 

hopping2Cu5d = np.zeros((1331,34,34))
P2Cu5d = [i for i in range(34)]
P2Cu5d[1:4] = [3,1,2] 
P2Cu5d[6:9] = [8,6,7]

hopping2Cu1d = np.zeros((1331,26,26))
P2Cu1d = [i for i in range(26)]
P2Cu1d[0] = 3
P2Cu1d[1] = 8
P2Cu1d[2:26] = [i for i in range(10,34)]

hopping1Cu5d = np.zeros((1331,17,17))
P1Cu5d = [i for i in range(17)]
P1Cu5d[1:4] = [3,1,2]
P1Cu5d[5:11] = [i for i in range(10,16)]
P1Cu5d[11:17] = [i for i in range(22,28)]

for i in range(-5,6):
    for j in range(-5,6):
        for k in range(-5,6):
            index = (i+5)*121+(j+5)*11+k+5
            matrix = Hopping[(i,j,k)] 
            matrix = np.array(matrix).real
            #hoppingCu[index,:,:] = matrix.transpose()
            hopping2Cu5d[index,:,:] = matrix[:,P2Cu5d][P2Cu5d,:]
            hopping2Cu1d[index,:,:] = matrix[:,P2Cu1d][P2Cu1d,:]
            hopping1Cu5d[index,:,:] = matrix[:,P1Cu5d][P1Cu5d,:]



from rham_LaNiO3_origin import Hopping 

hoppingNi5d = np.zeros((1331,14,14))
PNi5d = [i for i in range(14)]
PNi5d[1:4] = [3,1,2] 

hoppingNi2d = np.zeros((1331,11,11))
PNi2d = [i for i in range(11)]
PNi2d[1] = 3
PNi2d[2:11] = [i for i in range(5,14)]

for i in range(-5,6):
    for j in range(-5,6):
        for k in range(-5,6):
            index = (i+5)*121+(j+5)*11+k+5
            matrix = Hopping[(i,j,k)] 
            matrix = np.array(matrix).real
            #matrix = matrix.transpose()
            hoppingNi5d[index,:,:] = matrix[:,PNi5d][PNi5d,:]
            hoppingNi2d[index,:,:] = matrix[:,PNi2d][PNi2d,:]

#print P2Cu5d
#print len(P2Cu5d)
#print P2Cu1d
#print len(P2Cu1d)
#print P1Cu5d
#print len(P1Cu5d)
#print PNi5d
#print len(PNi5d)
#print PNi2d
#print len(PNi2d)
#raise SystemExit
