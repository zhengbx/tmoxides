# This file is part of the lattice-dmet program. lattice-dmet is free
# software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software
# Foundation, version 3.
# 
# lattice-dmet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with bfint (LICENSE). If not, see http://www.gnu.org/licenses/
# 
# Authors:
#    Gerald Knizia, 2012

#from numpy import *
#from helpers import *
import numpy as np
import scipy.linalg as la
import itertools as it
from hoppingWriteCu import * 
#from hoppingCu import *
#from hoppingNi import *
   
# TODO: consider replacing UnitCell by just Cell
def key2str(x,y,z,orb1,orb2):
    return "(%.4f, %.4f, %.4f), %d, %d" %(x,y,z,orb1,orb2)

def is_square(M):
   return len(M.shape) == 2 and M.shape[0] == M.shape[1]

def MakeIntegerLattice(Size):
   """return an array of form np.product(Size) x len(Size),
   representing all integers (ijk..) with
      0 <= i < Size[0],
      0 <= j < Size[1],
      ...
   """
   iTs = []
   for kji in it.product(*[range(o) for o in Size][::-1]):
      ijk = kji[::-1] # i goes fastest, then j, then k
      iTs.append(ijk)
   return np.array(iTs)

class FSiteList(object):
   def __init__(self, SitesOrSiteTypes, Xyzs = None):
      """create a list of sites. This corresponds to (a part of) a basis set.
      Args:
         - SiteTypes: List of user defined objects (e.g., strings or own classes),
           identifying different types of sites.
           --OR--
           Sequence of tuples (SiteType,Xyz) with SiteType as above.
         - Xyzs: len(SiteTypes) x nDimR matrix defining the positions of the
           sites in real space. The real-space dimension (xyz dimension) nDimR
           can be chosen freely, but must be equal for all sites. Will usually
           be equal to the lattice dimension, but need non necessarily be.
      """
      if ( Xyzs is None ):
         # given a sequence of tuple -- unpack the input.
         Xyzs = np.array([o[1] for o in SitesOrSiteTypes])
         SiteTypes = [o[0] for o in SitesOrSiteTypes]
      else:
         SiteTypes = SitesOrSiteTypes

      self.SiteTypes = SiteTypes
      self.Types = self.SiteTypes
      if not isinstance(Xyzs, np.ndarray):
         self.Xyzs = np.array(Xyzs) # copy into array format
      else:
         self.Xyzs = Xyzs # reference the original array.
      assert(self.Xyzs.shape[0] == len(self.SiteTypes))

   def __len__(self):
      return len(self.SiteTypes)
   def __getitem__(self, i):
      if isinstance(i, slice):
         return FSiteList(self.SiteTypes[i], self.Xyzs[i])
      else:
         return (self.SiteTypes[i], self.Xyzs[i])
   def __iter__(self):
      for (Type,Xyz) in zip(self.SiteTypes, self.Xyzs):
         yield (Type,Xyz)
   def __add__(self, dXyz):
      """translate the sites by the given vector; return new sites."""
      return FSiteList(self.SiteTypes, self.Xyzs + dXyz)
   def __str__(self):
      L = []
      L.append("   Index " + (len(self.SiteTypes)*"%6s") % tuple(range(len(self.SiteTypes))))
      L.append("   Type  " + (len(self.SiteTypes)*"%6s") % tuple(self.SiteTypes))
      for ixyz in range(self.Xyzs.shape[1]):
         L.append("   Pos/%s " % "xyz"[ixyz] + (len(self.SiteTypes)*"%6s") % tuple(self.Xyzs[:,ixyz]))
      return "\n".join(L)

   def Repeat(self, Size, LatticeVectors):
      """ repeat the sites of *this Size[i] times in direction
      LatticeVectors[i,:]. Returns (iTs, Ts, NewSiteList), where iTs is an
      prod(size) x LatticeVectors.shape[0] integer array and Ts =
      dot(iTs,LatticeVectors)."""

      # make lists of lattice translations for the repetitions
      iTs = MakeIntegerLattice(Size)
      Ts = np.dot(iTs, LatticeVectors.T)
      
      #print ":Repeat  Size = %s\niTs = \n%s\nVecs =\n%s\nLattiveVectors:\n%s" % (Size,iTs.T,Ts.T, LatticeVectors)

      # make list of (SiteType, xyz) of all sites in the repeated cell.
      # These are simply the unit-cell sites translated by self.Ts
      SiteXyzs = []
      for T in Ts:
         for UcXyz in self.Xyzs:
            SiteXyzs.append(T + UcXyz)
      SiteTypes = len(Ts) * self.SiteTypes
      NewSites = FSiteList(SiteTypes, SiteXyzs)

      #print "Uc.Repeat %s:\niTs = %s\nTs=%s\n%s" % (Size,iTs,Ts,NewSites)
      return (iTs,Ts,NewSites)


class FLatticeModel(object):
   """abstract base class for a physical lattice model, defining
   both the lattice and the lattice Hamiltonian."""
   def __init__(self, UnitCell, LatticeVectors, ModelParams=None, MaxRangeT=float("inf"),
         SitesAreSpinOrbitals=False, EnergyFactor=1.):
      """Defines the site sof the lattice. You still need to provide
      a function for returning the actual matrix elements.

      Args:
         UnitCell: an list of tuples (SiteType,xyz) defining the sites in the unit
            cell. The type of SiteType is up to you (e.g., a string or an own
            class). xyz must be a numpy array with the same length as the
            lattice dimension.
         LatticeVectors: a D x R matrix, with D the lattice dimension, and R the
            real-space dimension (you could, for example, have two stacked 2d
            lattices in 3d real space, with lattices in xy direction and
            different z components). v[0,:] is the first lattice vector, v[1,:]
            the second, etc. This defines the lattice translations. The unit-
            cell is understood to be repeated in all directions an infinite
            number of times, by adding integer amounts ijk of v[ijk,:] to the
            xyz parameters of the unit cells.

            Note: Lattice translations are indexed with integer vectors (i,j,k),
            to differenciate them from the real-space coordinates (x,y,z).
         ModelParams: If given, defines the names of calculation parameters on
            which this Hamiltonian/Lattice depend (for information and
            consistency purposes)
         MaxRangeT: The range beyond t_ij matrix elements can be considered
            zero (optimization setting for culling at unit-cell level,
            unused atm).
         SitesAreSpinOrbitals: The given unit-cell does contain spin-orbital
            sites instead of spatial orbitals. Note that in this case it is
            requires that all input sites are such that alpha orbitals come
            on even sites and beta orbitals on odd sites!
         EnergyFactor: By default, energies/extensive quantities are normalized
            to the number of unit cells (=1.). This factor can be used, for
            example, to normalize them to the number of sites instead. All
            energies and numbers of electrons are multiplied by this factor.
      """
      assert(isinstance(LatticeVectors, np.ndarray))
      self.nDim = LatticeVectors.shape[1]

      assert(LatticeVectors.shape[1] >= LatticeVectors.shape[0])
      self.LatticeVectors = LatticeVectors
      self.UnitCell = FSiteList(UnitCell)

      assert(self.UnitCell.Xyzs.shape[1] == LatticeVectors.shape[1])

      self.MaxRangeT = MaxRangeT
      self.ModelParams = ModelParams
      self.EnergyFactor = EnergyFactor

      self.SitesAreSpinOrbitals = SitesAreSpinOrbitals
      if self.SitesAreSpinOrbitals:
         # since all sites must come with A and B indices, the total
         # number must be even.
         assert(len(self.UnitCell) % 2 == 0)
         for i in range(0,len(self.UnitCell),2):
            # A and B sites probably should be at the same place.
            assert(self.UnitCell.Xyzs[i] == self.UnitCell.Xyzs[i+1])

      # we'd expect real or complex floats here.
      #self.ScalarType = type(self.GetTij(UnitCell[0],UnitCell[0]))
      #assert(self.ScalarType in (float,complex,np.float64,np.complex128))

   def GetTij(self, SiteI, SiteJ):
      """Get the core Hamiltonian matrix elements t_ij, where
      SiteI = (SiteTypeI, XyzI) and SiteJ = (SiteTypeJ, XyzJ)"""
      raise Exception("FLatticeModel::GetTij must be implemented in derived classes!")

   def GetUi(self, SiteI):
      """return the size of the on-site Hubbard interaction U on site i.
      SiteI = (SiteTypeI, XyzI)"""
      raise Exception("FLatticeModel::GetUi must be implemented in derived classes!")

   def GetJi(self, SiteI, SiteJ):
      """return the size of the on-site Hubbard interaction J on site i.
      SiteI = (SiteTypeI, XyzI)"""
      raise Exception("FLatticeModel::GetJi must be implemented in derived classes!")

   def MakeTijMatrix(self, SitesR, SitesC):
      """return a len(SitesR) x len(SitesC) size of the core Hamiltonian
      matrix."""
      raise Exception("FLatticeModel::MakeTijMatrix must be implemented in derived classes!")

# what follows now are some example classes for simple lattice models which
# demonstrate how the lattice class is intented to be used. For making new
# classes, you probably want to define your own files.

class FHubbardModel_La2CuO4(FLatticeModel):
   """The infinite 3d Hubbard model for La_2CuO_4 (no La sites included, but 2d orb for each Cu and 3p orbs for each O)"""
   def __init__(self, t, U, J, Delta):
      UnitCell = []
      #fractional coords
      UnitCell.append( ("Cu1z2", np.array([0,  0,  0])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu1x2-y2", np.array([0,  0,  0])) ) #1d-orbital d_x2y2 
      UnitCell.append( ("Cu1xz", np.array([0,  0,  0])) ) #1d-orbital d_xz 
      UnitCell.append( ("Cu1yz", np.array([0,  0,  0])) ) #1d-orbital d_yz 
      UnitCell.append( ("Cu1xy", np.array([0,  0,  0])) ) #1d-orbital d_xy 

      UnitCell.append( ("Cu2z2", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2x2-y2", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_x2y2
      UnitCell.append( ("Cu2xz", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2yz", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2xy", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2

      UnitCell.append( ("O1z", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("O1x", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("O1y", np.array([0,  0.5,  0])) ) #p-orbital
    
      UnitCell.append( ("O2z", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("O2x", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("O2y", np.array([0.5,  0,  0])) ) #p-orbital
      
      UnitCell.append( ("O3z", np.array([0.5,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3x", np.array([0.5,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3y", np.array([0.5,  0,  0.5])) ) #p-orbital
      
      UnitCell.append( ("O4z", np.array([0,  0.5,  0.5])) ) #p-orbital
      UnitCell.append( ("O4x", np.array([0,  0.5,  0.5])) ) #p-orbital
      UnitCell.append( ("O4y", np.array([0,  0.5,  0.5])) ) #p-orbital

      UnitCell.append( ("O5z", np.array([0,  0, 0.1858])) ) #p-orbital
      UnitCell.append( ("O5x", np.array([0,  0, 0.1858])) ) #p-orbital
      UnitCell.append( ("O5y", np.array([0,  0, 0.1858])) ) #p-orbital
      
      UnitCell.append( ("O6z", np.array([0,  0, 0.8142]))  )#p-orbital
      UnitCell.append( ("O6x", np.array([0,  0, 0.8142]))  )#p-orbital
      UnitCell.append( ("O6y", np.array([0,  0, 0.8142]))  )#p-orbital
      
      UnitCell.append( ("O7z", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      UnitCell.append( ("O7x", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      UnitCell.append( ("O7y", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      
      UnitCell.append( ("O8z", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      UnitCell.append( ("O8x", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      UnitCell.append( ("O8y", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      
      self.t = t
      self.U = U
      self.J = J
      self.Delta = Delta

      LatticeVectors = np.zeros((3,3),int)
      LatticeVectors[0,:] = [1.,0.0,0.0]
      LatticeVectors[1,:] = [0.0,1.,0.0]
      LatticeVectors[2,:] = [0.0,0.0,1.]
      # FIXMEL MaxRangeT and EnergyFactor are not adapted to the model 
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t","Delta", "J"], MaxRangeT=1.8, EnergyFactor=1.0/2)

   def MakeTijMatrix(self, SitesR, SitesC):
      """return a len(SitesR) x len(SitesC) size of the core Hamiltonian
      matrix."""
      CoreH = np.zeros((len(SitesR), len(SitesC)))
      SiteType = {"Cu1z2": 0,
                  "Cu1x2-y2": 3,
                  "Cu1xz": 1,
                  "Cu1yz": 2,
                  "Cu1xy": 4,
                  "Cu2z2": 5,
                  "Cu2x2-y2": 8,
                  "Cu2xz": 6,
                  "Cu2yz": 7,
                  "Cu2xy": 9,
                  "O1z":  10,
                  "O1x":  11,
                  "O1y":  12,
                  "O2z":  13,
                  "O2x":  14,
                  "O2y":  15,
                  "O3z":  16,
                  "O3x":  17,
                  "O3y":  18,
                  "O4z":  19,
                  "O4x":  20,
                  "O4y":  21,
                  "O5z":  22,
                  "O5x":  23,
                  "O5y":  24,
                  "O6z":  25,
                  "O6x":  26,
                  "O6y":  27,
                  "O7z":  28,
                  "O7x":  29,
                  "O7y":  30,
                  "O8z":  31,
                  "O8x":  32,
                  "O8y":  33
      }
      nUSites = len(SiteType)
      for i in range(len(SitesR)/nUSites):
         for j in range(len(SitesC)/nUSites):
            dXyz = SitesR[i*nUSites][1]-SitesC[j*nUSites][1]
            dXyz = [int(dx) for dx in dXyz]
            if abs(dXyz[0]) < 6 and abs(dXyz[1]) < 6 and abs(dXyz[2]) < 6:
               index = (dXyz[0]+5)*121+(dXyz[1]+5)*11+dXyz[2]+5
               CoreH[i*nUSites:(i+1)*nUSites,j*nUSites:(j+1)*nUSites] = hopping2Cu5d[index,:,:]
               if index == 665:    
                  CoreH[i*nUSites:(i*nUSites+10),j*nUSites:(j*nUSites+10)] -= self.Delta*np.eye(10)
      #for (iSiteR, SiteR) in enumerate(SitesR):
      #   TypeR = SiteType[SiteR[0]]
      #   for (iSiteC, SiteC) in enumerate(SitesC):
      #      dXyz = SiteR[1] - SiteC[1]
      #      TypeC = SiteType[SiteC[0]]
      #      if ((dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC) in HoppingTCu:
      #         CoreH[iSiteR,iSiteC] = HoppingTCu[(dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC]
      #      if TypeC == TypeR and TypeC in range(10) and sum(abs(dXyz)) == 0.0:
      #         CoreH[iSiteR,iSiteC] -= self.Delta
      return CoreH
      
   def GetUi(self, (SiteTypeI,XyzI)):
     if (SiteTypeI[:2]=='Cu'): return self.U 
     else: return 0
   
   def GetJi(self, (SiteTypeI,XyzI), (SiteTypeJ, XyzJ)):
      dXyz = XyzJ - XyzI
      if ((SiteTypeI[:2]=='Cu') and (SiteTypeJ[:2] =='Cu') and sum(abs(dXyz)) == 0): return self.J 
      else: return 0


class FHubbardModel_La2CuO4_1Cu(FLatticeModel):
   """The infinite 3d Hubbard model for La_2CuO_4 (no La sites included, but 2d orb for each Cu and 3p orbs for each O)"""
   def __init__(self, t, U, J, Delta):
      UnitCell = []
      #fractional coords
      #UnitCell.append( ("Cu1z2", np.array([0,  0,  0])) ) #1d-orbital d_z2
      #UnitCell.append( ("Cu1x2-y2", np.array([0,  0,  0])) ) #1d-orbital d_x2y2 
      #UnitCell.append( ("Cu1xz", np.array([0,  0,  0])) ) #1d-orbital d_xz 
      #UnitCell.append( ("Cu1yz", np.array([0,  0,  0])) ) #1d-orbital d_yz 
      #UnitCell.append( ("Cu1xy", np.array([0,  0,  0])) ) #1d-orbital d_xy 

      UnitCell.append( ("Cu2z2", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2x2-y2", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_x2y2
      UnitCell.append( ("Cu2xz", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2yz", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
      UnitCell.append( ("Cu2xy", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2

      #UnitCell.append( ("O1z", np.array([0,  0.5,  0])) ) #p-orbital
      #UnitCell.append( ("O1x", np.array([0,  0.5,  0])) ) #p-orbital
      #UnitCell.append( ("O1y", np.array([0,  0.5,  0])) ) #p-orbital
    
      #UnitCell.append( ("O2z", np.array([0.5,  0,  0])) ) #p-orbital
      #UnitCell.append( ("O2x", np.array([0.5,  0,  0])) ) #p-orbital
      #UnitCell.append( ("O2y", np.array([0.5,  0,  0])) ) #p-orbital
      
      UnitCell.append( ("O3z", np.array([0.5,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3x", np.array([0.5,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3y", np.array([0.5,  0,  0.5])) ) #p-orbital
      
      UnitCell.append( ("O4z", np.array([0,  0.5,  0.5])) ) #p-orbital
      UnitCell.append( ("O4x", np.array([0,  0.5,  0.5])) ) #p-orbital
      UnitCell.append( ("O4y", np.array([0,  0.5,  0.5])) ) #p-orbital
      
      #UnitCell.append( ("O5z", np.array([0,  0, 0.1858])) ) #p-orbital
      #UnitCell.append( ("O5x", np.array([0,  0, 0.1858])) ) #p-orbital
      #UnitCell.append( ("O5y", np.array([0,  0, 0.1858])) ) #p-orbital
      
      #UnitCell.append( ("O6z", np.array([0,  0, 0.8142]))  )#p-orbital
      #UnitCell.append( ("O6x", np.array([0,  0, 0.8142]))  )#p-orbital
      #UnitCell.append( ("O6y", np.array([0,  0, 0.8142]))  )#p-orbital
      
      UnitCell.append( ("O7z", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      UnitCell.append( ("O7x", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      UnitCell.append( ("O7y", np.array([0.5,  0.5, 0.6858])) ) #p-orbital
      
      UnitCell.append( ("O8z", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      UnitCell.append( ("O8x", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      UnitCell.append( ("O8y", np.array([0.5,  0.5, 0.3142])) ) #p-orbital
      
      self.t = t
      self.U = U
      self.J = J
      self.Delta = Delta

      LatticeVectors = np.zeros((3,3),int)
      LatticeVectors[0,:] = [1.0,0.0,0.0]
      LatticeVectors[1,:] = [0.0,1.0,0.0]
      LatticeVectors[2,:] = [0.5,0.5,0.5]
      # FIXMEL MaxRangeT and EnergyFactor are not adapted to the model 
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t","Delta", "J"], MaxRangeT=1.8, EnergyFactor=1.0/1)

   def MakeTijMatrix(self, SitesR, SitesC):
      """return a len(SitesR) x len(SitesC) size of the core Hamiltonian
      matrix."""
      CoreH = np.zeros((len(SitesR), len(SitesC)))
      SiteType = {'Cu2z2': 0,
                  'Cu2x2-y2': 1,
                  'Cu2xz': 2,
                  'Cu2yz': 3,
                  'Cu2xy': 4,
                  'O3z':  5,
                  'O3x':  6,
                  'O3y':  7,
                  'O4z':  8,
                  'O4x':  9,
                  'O4y': 10,
                  'O7z': 11,
                  'O7x': 12,
                  'O7y': 13,
                  'O8z': 14,
                  'O8x': 15,
                  'O8y': 16
                  }
      nUSites = len(SiteType)
      for i in range(len(SitesR)/nUSites):
         for j in range(len(SitesC)/nUSites):
            dXyz = SitesR[i*nUSites][1]-SitesC[j*nUSites][1]
            dXyz = [int(dx) for dx in dXyz]
            if abs(dXyz[0]) < 6 and abs(dXyz[1]) < 6 and abs(dXyz[2]) < 6:
               index = (dXyz[0]+5)*121+(dXyz[1]+5)*11+dXyz[2]+5
               CoreH[i*nUSites:(i+1)*nUSites,j*nUSites:(j+1)*nUSites] = hopping1Cu5d[index,:,:]
               if index == 665:    
                  CoreH[i*nUSites:(i*nUSites+5),j*nUSites:(j*nUSites+5)] -= self.Delta*np.eye(5)
      #for (iSiteC, SiteC) in enumerate(SitesC):
      #   TypeC = SiteType[SiteC[0]]
      #   for (iSiteR, SiteR) in enumerate(SitesR):
      #      dXyz = SiteR[1] - SiteC[1]
      #      TypeR = SiteType[SiteR[0]]
      #      if ((dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC) in HoppingTCu:
      #         CoreH[iSiteR,iSiteC] = HoppingTCu[(dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC]
      #         #print"%s, %s: %s" %(iSiteR, iSiteC, CoreH[iSiteR,iSiteC])
      #         #print (dXyz[0], dXyz[1], dXyz[2]), SiteR[0], SiteC[0]
      #         #print"%s, %s: %s" %(iSiteR, iSiteC, CoreH[iSiteR,iSiteC])
      #         #print (dXyz[0], dXyz[1], dXyz[2]), SiteR[0], SiteC[0]
      #      if TypeC == TypeR and TypeC in [0, 1, 14, 15] and sum(abs(dXyz)) == 0.0:
      #         CoreH[iSiteR,iSiteC] -= self.Delta
      #        #print (dXyz[0], dXyz[1], dXyz[2]), SiteR[0], SiteC[0]
      return CoreH
      
   def GetUi(self, (SiteTypeI,XyzI)):
     if (SiteTypeI[:2]=='Cu'): return self.U 
     else: return 0
   
   def GetJi(self, (SiteTypeI,XyzI), (SiteTypeJ, XyzJ)):
      dXyz = XyzJ - XyzI
      if ((SiteTypeI[:2]=='Cu') and (SiteTypeJ[:2] =='Cu') and sum(abs(dXyz)) == 0): return self.J 
      else: return 0

class FHubbardModel_LaNiO3(FLatticeModel):
   """The infinite 3d Hubbard model for LaNiO_3 (no La sites included, but 2d orb for each Ni and 3p orbs for each O)"""
   def __init__(self, t, U, J, Delta):
      UnitCell = []
      #absolute coords
      UnitCell.append( ("Niz2", np.array([0,  0,  0])) ) #1d-orbital d_z2 
      UnitCell.append( ("Nix2-y2", np.array([0,  0,  0])) ) #1d-orbital d_x2-y2 
      UnitCell.append( ("Nixz", np.array([0,  0,  0])) ) #1d-orbital d_xz 
      UnitCell.append( ("Niyz", np.array([0,  0,  0])) ) #1d-orbital d_yz 
      UnitCell.append( ("Nixy", np.array([0,  0,  0])) ) #1d-orbital d_xy 

      UnitCell.append( ("O1z", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("O1x", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("O1y", np.array([0.5,  0,  0])) ) #p-orbital
    
      UnitCell.append( ("O2z", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("O2x", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("O2y", np.array([0,  0.5,  0])) ) #p-orbital
      
      UnitCell.append( ("O3z", np.array([0,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3x", np.array([0,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("O3y", np.array([0,  0,  0.5])) ) #p-orbital
      
      self.t = t
      self.U = U
      self.J = J
      self.Delta = Delta

      LatticeVectors = np.zeros((3,3), float)
      LatticeVectors[0,:] = [1.,0.0,0.0]
      LatticeVectors[1,:] = [0.0,1.,0.0]
      LatticeVectors[2,:] = [0.0,0.0,1.]
      # FIXMEL MaxRangeT and EnergyFactor are not adapted to the model 
      # MaxRangeT should be larger than sqrt(3)
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t","Delta", "J"], MaxRangeT=1.8, EnergyFactor=1.0/2)

   def MakeTijMatrix(self, SitesR, SitesC):
      """return a len(SitesR) x len(SitesC) size of the core Hamiltonian
      matrix."""
      CoreH = np.zeros((len(SitesR), len(SitesC)))
      
      SiteType = {'Niz2': 0,
                  'Nix2-y2': 1,
                  'Nixz': 2,
                  'Niyz': 3,
                  'Nixy': 4,
                  'O1z':  5,
                  'O1x':  6,
                  'O1y':  7,
                  'O2z':  8,
                  'O2x':  9,
                  'O2y':  10,
                  'O3z':  11,
                  'O3x':  12,
                  'O3y':  13,
                  }

      nUSites = len(SiteType)
      for i in range(len(SitesR)/nUSites):
         for j in range(len(SitesC)/nUSites):
            dXyz = SitesR[i*nUSites][1]-SitesC[j*nUSites][1]
            dXyz = [int(dx) for dx in dXyz]
            if abs(dXyz[0]) < 6 and abs(dXyz[1]) < 6 and abs(dXyz[2]) < 6:
               index = (dXyz[0]+5)*121+(dXyz[1]+5)*11+dXyz[2]+5
               CoreH[i*nUSites:(i+1)*nUSites,j*nUSites:(j+1)*nUSites] = hoppingNi5d[index,:,:]
               if index == 665:    
                  CoreH[i*nUSites:(i*nUSites+5),j*nUSites:(j*nUSites+5)] -= self.Delta*np.eye(5)
      #for (iSiteC, SiteC) in enumerate(SitesC):
      #   TypeC = SiteType[SiteC[0]]
      #   for (iSiteR, SiteR) in enumerate(SitesR):
      #      dXyz = SiteR[1] - SiteC[1]
      #      TypeR = SiteType[SiteR[0]]
      #      if ((dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC) in HoppingTNi:
      #         CoreH[iSiteR,iSiteC] = HoppingTNi[(dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC]
      #      if TypeC == TypeR and TypeC in [0, 1] and sum(abs(dXyz)) == 0.0:
      #         CoreH[iSiteR,iSiteC] -= self.Delta
      return CoreH
 
   def GetUi(self, (SiteTypeI,XyzI)):
      if (SiteTypeI[:2]=='Ni'): return self.U 
      else: return 0
   
   def GetJi(self, (SiteTypeI,XyzI), (SiteTypeJ, XyzJ)):
      dXyz = XyzJ - XyzI
      if ((SiteTypeI[:2]=='Ni') and (SiteTypeJ[:2] =='Ni') and sum(abs(dXyz)) == 0): return self.J
      else: return 0

def _TestLattices():
   L = 12
   nImp = 2
   Hub1d = FHubbardModel1d(nImp, t=1., U=4.)

   Sites = Hub1d.MakeSuperCellSites(np.arange(L/nImp).reshape((L/nImp,1)))
   CoreH = Hub1d.MakeTijMatrix(Sites,Sites)
   print CoreH


if __name__ == "__main__":
   _TestLattices()




