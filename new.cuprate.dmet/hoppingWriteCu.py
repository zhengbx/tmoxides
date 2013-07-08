import numpy as np
from rham_La2CuO4_origin import Hopping 

hopping2Cu5d = np.zeros((1331,34,34))
hoppingCu5d = np.zeros((1331,34,34))
P2Cu5d = [i for i in range(34)]
P2Cu5d[1:4] = [3,1,2] 
P2Cu5d[6:9] = [8,6,7]

hopping2Cu1d = np.zeros((1331,26,26))
P2Cu1d = [i for i in range(26)]
P2Cu1d[0] = 3
P2Cu1d[1] = 8
P2Cu1d[2:26] = [i for i in range(10,34)]

hopping1Cu5d = np.zeros((2662,17,17))
P1Cu5d = range(17)
P1Cu5d[:5] = [5,6,7,8,9]
P1Cu5d[5:11] = range(16,22)
P1Cu5d[11:17] = range(28,34)
P1Cu5d0 = range(17)
P1Cu5d0[5:11] = range(10,16)
P1Cu5d0[11:17] = range(22,28)

hopping1Cu1d = np.zeros((2662,13,13))
P1Cu1d = range(13)
P1Cu1d[0] = 6
P1Cu1d[1:7] = range(16,22)
P1Cu1d[7:13] = range(28,34)
P1Cu1d0 = range(13)
P1Cu1d0[0] = 1
P1Cu1d0[1:7] = range(10,16)
P1Cu1d0[7:13] = range(22,28)


for i in range(-5,6):
    for j in range(-5,6):
        for k in range(-5,6):
            index = (i+5)*121+(j+5)*11+k+5
            matrix = Hopping[(i,j,k)] 
            matrix = np.array(matrix).real
            #hoppingCu[index,:,:] = matrix.transpose()
            hopping2Cu5d[index,:,:] = matrix[:,P2Cu5d][P2Cu5d,:]
            hopping2Cu1d[index,:,:] = matrix[:,P2Cu1d][P2Cu1d,:]
            #hopping1Cu5d[index,:,:] = matrix[:,P1Cu5d][P1Cu5d,:]


def rotateMatrix(hoppingCu5d0,orbital,edgeK,rangeK,rotColumn,direction,distance):
   for i in range(-5,6):
       for j in range(-5,6):
           k = edgeK
           if direction =="z":
              index = (i+5)*121+(j+5)*11+k+5
           elif direction == "y":
              index = (i+5)*121+(k+5)*11+j+5
           elif direction == "x":
              index = (k+5)*121+(j+5)*11+i+5
           else:
               raise Exception("Wrong direction")
           matrix =1.0* hoppingCu5d0[index,:,:]
           if rotColumn:
              matrix[:,orbital] = np.zeros((34,3))
           else:
              matrix[orbital,:] = np.zeros((3,34))
           hoppingCu5d[index,:,:] = matrix
           for k in rangeK:
              if direction =="z":
                 index = (i+5)*121+(j+5)*11+k+5
              elif direction == "y":
                 index = (i+5)*121+(k+5)*11+j+5
              elif direction == "x":
                 index = (k+5)*121+(j+5)*11+i+5
              else:
                  raise Exception("Wrong direction")
              matrix = 1.0* hoppingCu5d0[index,:,:]
              if rotColumn:
                 matrix[:,orbital] = hoppingCu5d0[index-distance,:,:][:,orbital]
              else:
                 matrix[orbital,:] = hoppingCu5d0[index+distance,:,:][orbital,:]
              hoppingCu5d[index,:,:] = matrix
   return hoppingCu5d

hoppingCu5d0 = 1.0*hopping2Cu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(25,28),-5,range(-4,6),True,'z',1)
hoppingCu5d0 = 1.0*hoppingCu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(16,19),5,range(-5,5),True,'y',-11)
hoppingCu5d0 = 1.0*hoppingCu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(19,22),5,range(-5,5),True,'x',-121)
hoppingCu5d0 = 1.0*hoppingCu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(25,28),5,range(-5,5),False,'z',1)
hoppingCu5d0 = 1.0*hoppingCu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(16,19),-5,range(-4,6),False,'y',-11)
hoppingCu5d0 = 1.0*hoppingCu5d
hoppingCu5d = rotateMatrix(hoppingCu5d0,range(19,22),-5,range(-4,6),False,'x',-121)

for i in range(-5,6):
    for j in range(-5,6):
        for k in range(-5,6):
           index = (i+5)*121+(j+5)*11+k+5
           matrix = hoppingCu5d[index,:,:]
           index0 = (i+5)*242+(j+5)*22+2*k+11
           hopping1Cu5d[index0-1,:,:] = matrix[P1Cu5d,:][:,P1Cu5d0]
           hopping1Cu5d[index0,:,:] = matrix[P1Cu5d,:][:,P1Cu5d]
           hopping1Cu1d[index0-1,:,:] = matrix[P1Cu1d,:][:,P1Cu1d0]
           hopping1Cu1d[index0,:,:] = matrix[P1Cu1d,:][:,P1Cu1d]

def checkHopping1Cu(hoppingCu5d):
   print hoppingCu5d.shape
   wrongindex = []
   for index in range(hoppingCu5d.shape[0]):
       matrix = hoppingCu5d[index,:,:]
       matrix0 = hoppingCu5d[-index-1,:,:]
       #diff = matrix[P1Cu5d,:][:,P1Cu5d0]-matrix0[P1Cu5d0,:][:,P1Cu5d].transpose() 
       #diff = matrix[P1Cu5d,:][:,P1Cu5d0]-matrix[P1Cu5d0,:][:,P1Cu5d] 
       diff = matrix[P1Cu5d,:][:,P1Cu5d]-matrix[P1Cu5d0,:][:,P1Cu5d0] 
       error = np.linalg.norm(diff)
       if error > 1.0e-4:
          print index, error
          wrongindex.append(index)
          #print diff.shape
          #print matrix.shape
       #diff = matrix[P1Cu5d,:][:,P1Cu5d] - matrix0[P1Cu5d,:][:,P1Cu5d].transpose()
       #error = np.linalg.norm(diff)
       #print index, error
       #if error > 1.0e-6:
       #    print index, error
       if index == 665:
           print index,error
           print matrix[0,22]
           print matrix[5,28]
           print matrix[0,25]
           print matrix[5,31]
           
           print matrix[25,25]
           print matrix[31,31]
           print matrix[10,13]
           print matrix[16,19]
           #print Hopping[(0,-4,0)][10][13]
           #print Hopping[(0,-5,0)][16][19]
           print matrix[0,0]
           print matrix[1,12]
           print matrix[2,2]
           print matrix[3,3]

           print P1Cu5d
           print P1Cu5d0
           #print diff
       #print matrix.shape
   #for i in range(-5,6):
   #    for j in range(-5,6):
   #        for k in range(-5,6):
   #            index = (i+5)*121+(j+5)*11+k+5
   #            if index in wrongindex:
   #                print i,j,k,index




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



if ( __name__ == "__main__" ):
   #checkHopping1Cu(hoppingCu5d)
   print P2Cu5d
   print len(P2Cu5d)
   print P2Cu1d
   print len(P2Cu1d)
   print P1Cu5d
   print len(P1Cu5d)
   print P1Cu1d
   print len(P1Cu1d)
   print PNi5d
   print len(PNi5d)
   print PNi2d
   print len(PNi2d)
   checkHopping1Cu(hoppingCu5d)
   #raise SystemExit
