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
   
# TODO: consider replacing UnitCell by just Cell

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
      #CoreH = np.zeros((len(SitesR), len(SitesC)), self.ScalarType)
      CoreH = np.zeros((len(SitesR), len(SitesC)))
      
      #functions for the generation of hopping matrix input file
      def Cu1x2y2(XyzI):
         Xyzi = XyzI
         SiteTypeI = 'Cu11'
         return SiteTypeI, Xyzi
      def Cu1z2(XyzI):
         Xyzi = XyzI
         SiteTypeI = 'Cu12'
         return SiteTypeI, Xyzi
      def O1z(XyzI):
        Xyzi = XyzI+[0,  0.5,  0]
        SiteTypeI =  'O1z'
        return SiteTypeI, Xyzi
      def O1x(XyzI):
        Xyzi = XyzI+[0,  0.5,  0]
        SiteTypeI = 'O1x'
        return SiteTypeI, Xyzi
      def O1y(XyzI):
        Xyzi = XyzI+[0,  0.5,  0]
        SiteTypeI = 'O1y'
        return SiteTypeI, Xyzi
      def O2z(XyzI):
        Xyzi = XyzI+[0.5,  0,  0]
        SiteTypeI = 'O2z'
        return SiteTypeI, Xyzi
      def O2x(XyzI):
        Xyzi = XyzI+[0.5,  0,  0]
        SiteTypeI = 'O2x'
        return SiteTypeI, Xyzi
      def O2y(XyzI):
        Xyzi = XyzI+[0.5,  0,  0]
        SiteTypeI = 'O2y'
        return SiteTypeI, Xyzi
      def O3z(XyzI):
        Xyzi = XyzI+[0.5,  0,  0.5]
        SiteTypeI = 'O3z'
        return SiteTypeI, Xyzi
      def O3x(XyzI):
        Xyzi = XyzI+[0.5,  0,  0.5]
        SiteTypeI = 'O3x'
        return SiteTypeI, Xyzi
      def O3y(XyzI):
        Xyzi = XyzI+[0.5,  0,  0.5]
        SiteTypeI = 'O3y'
        return SiteTypeI, Xyzi
      def O4z(XyzI):
        Xyzi = XyzI+[0,  0.5,  0.5]
        SiteTypeI = 'O4z'
        return SiteTypeI, Xyzi
      def O4x(XyzI):
        Xyzi = XyzI+[0,  0.5,  0.5]
        SiteTypeI = 'O4x'
        return SiteTypeI, Xyzi
      def O4y(XyzI):
        Xyzi = XyzI+[0,  0.5,  0.5]
        SiteTypeI = 'O4y'
        return SiteTypeI, Xyzi
      def Cu2x2y2(XyzI):
        Xyzi = XyzI+[ 0.5,   0.5,  0.5]
        SiteTypeI = 'Cu21'
        return SiteTypeI, Xyzi
      def Cu2z2(XyzI):
        Xyzi = XyzI+[ 0.5,   0.5,  0.5]
        SiteTypeI = 'Cu22'
        return SiteTypeI, Xyzi
      def O5z(XyzI):
        Xyzi = XyzI+[0,  0, 0.1858]
        SiteTypeI = 'O5z'
        return SiteTypeI, Xyzi
      def O5x(XyzI):
        Xyzi = XyzI+[0,  0, 0.1858]
        SiteTypeI = 'O5x'
        return SiteTypeI, Xyzi
      def O5y(XyzI):
        Xyzi = XyzI+[0,  0, 0.1858]
        SiteTypeI = 'O5y'
        return SiteTypeI, Xyzi
      def O6z(XyzI):
        Xyzi = XyzI+[0,  0, 0.8142]
        SiteTypeI = 'O6z'
        return SiteTypeI, Xyzi
      def O6x(XyzI):
        Xyzi = XyzI+[0,  0, 0.8142]
        SiteTypeI = 'O6x'
        return SiteTypeI, Xyzi
      def O6y(XyzI):
        Xyzi = XyzI+[0,  0, 0.8142]
        SiteTypeI = 'O6y'
        return SiteTypeI, Xyzi
      def O7z(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.6858]
        SiteTypeI = 'O7z'
        return SiteTypeI, Xyzi
      def O7x(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.6858]
        SiteTypeI = 'O7x'
        return SiteTypeI, Xyzi
      def O7y(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.6858]
        SiteTypeI = 'O7y'
        return SiteTypeI, Xyzi
      def O8z(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.3142]
        SiteTypeI = 'O8z'
        return SiteTypeI, Xyzi
      def O8x(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.3142]
        SiteTypeI = 'O8x'
        return SiteTypeI, Xyzi
      def O8y(XyzI):
        Xyzi = XyzI+[0.5,  0.5, 0.3142]
        SiteTypeI = 'O8y'
        return SiteTypeI, Xyzi
      def NoMem(XyzI):
        Xyzi = XyzI
        SiteTypeI = 'Cu3'
        return SiteTypeI, Xyzi

      whichorb =  {  0: Cu1z2,
                     1: NoMem,
                     2: NoMem,
                     3: Cu1x2y2,
                     4: NoMem,
                     5: Cu2z2,
                     6: NoMem,
                     7: NoMem,
                     8: Cu2x2y2,
                     9: NoMem,
                    10: O1z,
                    11: O1x,
                    12: O1y,
                    13: O2z,
                    14: O2x,
                    15: O2y,
                    16: O3z,
                    17: O3x,
                    18: O3y,
                    19: O4z,
                    20: O4x,
                    21: O4y,
                    22: O5z,
                    23: O5x,
                    24: O5y,
                    25: O6z,
                    26: O6x,
                    27: O6y,
                    28: O7z,
                    29: O7x,
                    30: O7y,
                    31: O8z,
                    32: O8x,
                    33: O8y
                     }
     
      HoppingT = {}
      # this should be done only once and then written to file
      # print "read hopping data"
      import re
      f = open('rham_La2CuO4_sort.py','r')
      f2 = open('hoppingCu.py', 'a')
      for line in f:
         data = re.split(',|\n', line)
         XyzI = np.array([int(data[0]), int(data[1]), int(data[2])])
         XyzJ = np.array([ 0, 0, 0]) 
         if (whichorb[int(data[3])](XyzI)[0]!='Cu3' and whichorb[int(data[4])](XyzJ)[0]!='Cu3'):
            Xyzi=whichorb[int(data[3])](XyzI)[1]
            Xyzj=whichorb[int(data[4])](XyzJ)[1]
            dXyz = Xyzi - Xyzj
         HoppingT[(dXyz[0], dXyz[1], dXyz[2]), whichorb[int(data[3])](XyzI)[0],whichorb[int(data[4])](XyzJ)[0]] = data[5]
         f2.write("HoppingTCu[(%s, %s, %s), %s, %s]=%s \n" %(dXyz[0], dXyz[1], dXyz[2], whichorb[int(data[3])](XyzI)[0],whichorb[int(data[4])](XyzJ)[0], data[5]))
      f2.close()
      f.close()
      raise SystemExit
    
      SiteType = {'Cu1': 0,
                  'Cu2': 1,
                  'Oz':  2,
                  'Ox':  3,
                  'Oy':  4,
                  'Cu3': 5
                  }
      for (iSiteC, SiteC) in enumerate(SitesC):
         for (iSiteR, SiteR) in enumerate(SitesR):
            dXyz = SiteR[1] - SiteC[1]
            TypeR = SiteType[SiteR[0]]
            TypeC = SiteType[SiteC[0]]
            if (((dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC) in HoppingT):
               CoreH[iSiteR,iSiteC] = HoppingT[(dXyz[0], dXyz[1], dXyz[2]), TypeR, TypeC]
               print CoreH[iSiteR,iSiteC]
            else:
               CoreH[iSiteR,iSiteC] = 0.0
      return CoreH

# what follows now are some example classes for simple lattice models which
# demonstrate how the lattice class is intented to be used. For making new
# classes, you probably want to define your own files.

class FHubbardModel_La2CuO4(FLatticeModel):
   """The infinite 3d Hubbard model for La_2CuO_4 (no La sites included, but 2d orb for each Cu and 3p orbs for each O)"""
   def __init__(self, t, U, J, Delta):
      UnitCell = []
      #fractional coords
      UnitCell.append( ("Cu11", np.array([0,  0,  0])) ) #1d-orbital d_x2y2 
      UnitCell.append( ("Cu12", np.array([0,  0,  0])) ) #1d-orbital d_z2 
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
      
      UnitCell.append( ("Cu21", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_x2y2
      UnitCell.append( ("Cu22", np.array([ 0.5,   0.5,  0.5])) ) #1d-orbital d_z2
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
      LatticeVectors[0,:] = [3.8283,0.0,0.0]
      LatticeVectors[1,:] = [0.0,3.8283,0.0]
      LatticeVectors[2,:] = [0.0,0.0,13.1626]
      # FIXMEL MaxRangeT and EnergyFactor are not adapted to the model 
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t","Delta", "J"], MaxRangeT=1.5, EnergyFactor=1.0/6)

   def GetTij(self, (SiteTypeI,XyzI), (SiteTypeJ,XyzJ), iSiteI, iSiteJ):
      if (abs(XyzI[0])>6.): 
         self.t=0.0
         return self.t
      elif (abs(XyzI[1])>6.): 
         self.t=0.0
         return self.t
      elif (abs(XyzI[2])>6.): 
         self.t=0.0
         return self.t
      #Delta
      if ((abs(sum(XyzI-XyzJ)) == 0.0)):
        if ((SiteTypeI=='Ox') or (SiteTypeI=='Oy') or (SiteTypeI=='Oz')): 
          self.Delta = -1.3121
          #calculation of Delta (constant for all relevant sites)
          #Delta =0.0
          #for i in range(10, 34):
          #   self.Delta += HoppingCu[( 0, 0, 0,)][i][i]      
          #self.Delta = self.Delta/24 - 0.5*HoppingCu[( 0, 0, 0,)][3][3]- 0.5*HoppingCu[( 0, 0, 0,)][8][8]            
          return self.Delta
      i =0
      j =0
      whichOrbI = iSiteI % 28
      whichOrbJ = iSiteJ % 28
      def Cu1x2y2(XyzI):
         Xyzi = XyzI
         i = 3
         return Xyzi, i
      def Cu1z2(XyzI):
         Xyzi = XyzI
         i = 0
         return Xyzi, i
      def O1z(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =10
        return Xyzi, i
      def O1x(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =11
        return Xyzi, i
      def O1y(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =12
        return Xyzi, i
      def O2z(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =13
        return Xyzi, i
      def O2x(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =14
        return Xyzi, i
      def O2y(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =15
        return Xyzi, i
      def O3z(XyzI):
        Xyzi = XyzI-[0.5,  0,  0.5]
        i =16
        return Xyzi, i
      def O3x(XyzI):
        Xyzi = XyzI-[0.5,  0,  0.5]
        i =17
        return Xyzi, i
      def O3y(XyzI):
        Xyzi = XyzI-[0.5,  0,  0.5]
        i =18
        return Xyzi, i
      def O4z(XyzI):
        Xyzi = XyzI-[0,  0.5,  0.5]
        i =19
        return Xyzi, i
      def O4x(XyzI):
        Xyzi = XyzI-[0,  0.5,  0.5]
        i =20
        return Xyzi, i
      def O4y(XyzI):
        Xyzi = XyzI-[0,  0.5,  0.5]
        i =21
        return Xyzi, i
      def Cu2x2y2(XyzI):
        Xyzi = XyzI-[ 0.5,   0.5,  0.5]
        i =8
        return Xyzi, i
      def Cu2z2(XyzI):
        Xyzi = XyzI-[ 0.5,   0.5,  0.5]
        i =5
        return Xyzi, i
      def O5z(XyzI):
        Xyzi = XyzI-[0,  0, 0.1858]
        i =22
        return Xyzi, i
      def O5x(XyzI):
        Xyzi = XyzI-[0,  0, 0.1858]
        i =23
        return Xyzi, i
      def O5y(XyzI):
        Xyzi = XyzI-[0,  0, 0.1858]
        i =24
        return Xyzi, i
      def O6z(XyzI):
        Xyzi = XyzI-[0,  0, 0.8142]
        i =25
        return Xyzi, i
      def O6x(XyzI):
        Xyzi = XyzI-[0,  0, 0.8142]
        i =26
        return Xyzi, i
      def O6y(XyzI):
        Xyzi = XyzI-[0,  0, 0.8142]
        i =27
        return Xyzi, i
      def O7z(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.6858]
        i =28
        return Xyzi, i
      def O7x(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.6858]
        i =29
        return Xyzi, i
      def O7y(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.6858]
        i =30
        return Xyzi, i
      def O8z(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.3142]
        i =31
        return Xyzi, i
      def O8x(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.3142]
        i =32
        return Xyzi, i
      def O8y(XyzI):
        Xyzi = XyzI-[0.5,  0.5, 0.3142]
        i =33
        return Xyzi, i
      options = { 0: Cu1x2y2,
                  1: Cu1z2,
                  2: O1z,
                  3: O1x,
                  4: O1y,
                  5: O2z,
                  6: O2x,
                  7: O2y,
                  8: O3z,
                  9: O3x,
                 10: O3y,
                 11: O4z,
                 12: O4x,
                 13: O4y,
                 14: Cu2x2y2,
                 15: Cu2z2,
                 16: O5z,
                 17: O5x,
                 18: O5y,
                 19: O6z,
                 20: O6x,
                 21: O6y,
                 22: O7z,
                 23: O7x,
                 24: O7y,
                 25: O8z,
                 26: O8x,
                 27: O8y
      }
      (Xyzi, i)=options[whichOrbI](XyzI)
      if (abs(Xyzi[0]) >5.): 
         self.t =0.0
         return self.t
      if (abs(Xyzi[1]) >5.): 
         self.t =0.0
         return self.t
      if (abs(Xyzi[2]) >5.): 
         self.t =0.0
         return self.t
      (Xyzj, j)=options[whichOrbI](XyzJ)
      self.t = HoppingCu[(Xyzi[0], Xyzi[1], Xyzi[2])][i][j]
      return self.t
      
      
   def GetUi(self, (SiteTypeI,XyzI)):
      if (SiteTypeI=='Cu'): return self.U 
      else: return 0
   
   def GetJi(self, (SiteTypeI,XyzI), (SiteTypeJ, XyzJ)):
      dXyz = XyzJ - XyzI
      if ((SiteTypeI=='Cu') and (SiteTypeJ =='Cu') and sum(abs(dXyz)) == 0): return self.J 
      else: return 0

class FHubbardModel_LaNiO3(FLatticeModel):
   """The infinite 3d Hubbard model for LaNiO_3 (no La sites included, but 2d orb for each Ni and 3p orbs for each O)"""
   def __init__(self, t, U, J, Delta):
      UnitCell = []
      #absolute coords
      UnitCell.append( ("Ni1", np.array([0,  0,  0])) ) #1d-orbital d_x2y2 
      UnitCell.append( ("Ni2", np.array([0,  0,  0])) ) #1d-orbital d_z2 
      UnitCell.append( ("Oz", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("Ox", np.array([0.5,  0,  0])) ) #p-orbital
      UnitCell.append( ("Oy", np.array([0.5,  0,  0])) ) #p-orbital
    
      UnitCell.append( ("Oz", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("Ox", np.array([0,  0.5,  0])) ) #p-orbital
      UnitCell.append( ("Oy", np.array([0,  0.5,  0])) ) #p-orbital
      
      UnitCell.append( ("Oz", np.array([0,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("Ox", np.array([0,  0,  0.5])) ) #p-orbital
      UnitCell.append( ("Oy", np.array([0,  0,  0.5])) ) #p-orbital
      
      self.t = t
      self.U = U
      self.J = J
      self.Delta = Delta

      LatticeVectors = np.zeros((3,3), float)
      LatticeVectors[0,:] = [1.,0.0,0.0]
      LatticeVectors[1,:] = [0.0,1.,0.0]
      LatticeVectors[2,:] = [0.0,0.0,1.]
      # FIXMEL MaxRangeT and EnergyFactor are not adapted to the model 
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t","Delta", "J"], MaxRangeT=5, EnergyFactor=1.0/6)

   def GetTij(self, (SiteTypeI,XyzI), (SiteTypeJ,XyzJ), iSiteI, iSiteJ):
      if (abs(XyzI[0])>6.): 
         self.t=0.0
         return self.t
      elif (abs(XyzI[1])>6.): 
         self.t=0.0
         return self.t
      elif (abs(XyzI[2])>6.): 
         self.t=0.0
         return self.t
      #Delta
      if ((abs(sum(XyzI-XyzJ)) == 0.0)):
        if ((SiteTypeI=='Ox') or (SiteTypeI=='Oy') or (SiteTypeI=='Oz')): 
          #self.Delta = -1.3121
          #calculation of Delta (constant for all relevant sites)
          self.Delta =0.0
          for i in range(5, 13):
             self.Delta += HoppingNi[( 0, 0, 0,)][i][i]      
          self.Delta = self.Delta/8 - 0.5*HoppingNi[( 0, 0, 0,)][0][0]- 0.5*HoppingNi[( 0, 0, 0,)][3][3]            
          return self.Delta
      i =0
      j =0
      whichOrbI = iSiteI % 11
      whichOrbJ = iSiteJ % 11
      def Ni1x2y2(XyzI):
         Xyzi = XyzI
         i = 3
         return Xyzi, i
      def Ni1z2(XyzI):
         Xyzi = XyzI
         i = 0
         return Xyzi, i
      def O1z(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =5
        return Xyzi, i
      def O1x(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =6
        return Xyzi, i
      def O1y(XyzI):
        Xyzi = XyzI-[0.5,  0,  0]
        i =7
        return Xyzi, i
      def O2z(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =8
        return Xyzi, i
      def O2x(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =9
        return Xyzi, i
      def O2y(XyzI):
        Xyzi = XyzI-[0,  0.5,  0]
        i =10
        return Xyzi, i
      def O3z(XyzI):
        Xyzi = XyzI-[0,  0,  0.5]
        i =11
        return Xyzi, i
      def O3x(XyzI):
        Xyzi = XyzI-[0,  0,  0.5]
        i =12
        return Xyzi, i
      def O3y(XyzI):
        Xyzi = XyzI-[0,  0,  0.5]
        i =13
        return Xyzi, i
      options = { 0: Ni1x2y2,
                  1: Ni1z2,
                  2: O1z,
                  3: O1x,
                  4: O1y,
                  5: O2z,
                  6: O2x,
                  7: O2y,
                  8: O3z,
                  9: O3x,
                 10: O3y 
      }
      (Xyzi, i)=options[whichOrbI](XyzI)
      (Xyzj, j)=options[whichOrbJ](XyzJ)
      Xyzi = Xyzi - Xyzj
      if (abs(Xyzi[0]) >5.): 
         self.t =0.0
         return self.t
      if (abs(Xyzi[1]) >5.): 
         self.t =0.0
         return self.t
      if (abs(Xyzi[2]) >5.): 
         self.t =0.0
         return self.t
      self.t = HoppingNi[(Xyzi[0], Xyzi[1], Xyzi[2])][i][j]
      return self.t
      
      
   def GetUi(self, (SiteTypeI,XyzI)):
      if (SiteTypeI=='Ni'): return self.U 
      else: return 0
   
   def GetJi(self, (SiteTypeI,XyzI), (SiteTypeJ, XyzJ)):
      dXyz = XyzJ - XyzI
      if ((SiteTypeI=='Ni') and (SiteTypeJ =='Ni') and sum(abs(dXyz)) == 0): return self.J 
      else: return 0

class FSKModelCubic3d2o(FLatticeModel):
   """The infinite 3d Hubbard-Model with local Slater-Kanamori interactions within
   2 e_g orbitals
   """
   def __init__(self, t, U, Delta, J, Shift): # Delta=E_p-E_d Shift is the relative shift among d-orbitals
      UnitCell = []
      UnitCell.append( ("D1", np.array([0, 0, 0])) ) # two d orbitals
      UnitCell.append( ("D2", np.array([0, 0, 0])) )
      UnitCell.append( ("O", np.array([0.5, 0, 0])) )
      UnitCell.append( ("O", np.array([0, 0.5, 0])) )
      UnitCell.append( ("O", np.array([0, 0, 0.5])) )
      self.t = t
      self.U = U
      self.Delta = Delta
      self.J = J
      self.Shift = Shift

      LatticeVectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
      FLatticeModel.__init__(self, UnitCell, LatticeVectors, ["U", "t", "Delta", "J", "Shift"], MaxRangeT=1./5)

   def GetTij(self, (SiteTypeI,XyzI), (SiteTypeJ,XyzJ)):
      dXyz = XyzJ -XyzI
      if (sum(abs(dXyz)) == 0 and SiteTypeI == 'O'): return self.Delta
      elif (sum(abs(dXyz)) == 0 and SiteTypeI == 'D1'): return self.Shift[0]
      elif (sum(abs(dXyz)) == 0 and SiteTypeI == 'D2'): return self.Shift[1]
      # FIXME : within this framework, we cannot describe 2e part
      # maybe take some code from QM DMET

def _TestLattices():
   L = 12
   nImp = 2
   Hub1d = FHubbardModel1d(nImp, t=1., U=4.)

   Sites = Hub1d.MakeSuperCellSites(np.arange(L/nImp).reshape((L/nImp,1)))
   CoreH = Hub1d.MakeTijMatrix(Sites,Sites)
   print CoreH


if __name__ == "__main__":
   _TestLattices()




