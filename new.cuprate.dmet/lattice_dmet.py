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

from output import *
from textwrap import dedent
from meanfield import *
from copy import copy
from lattice_model import *
#from lattice_model import FLatticeModel
from fragments import FFragment, FDmetContext, FDmetParams
from jobs import FJob, FJobGroup, FDefaultParams, ToClass
import numpy as np
import pickle as p

Banner = """\
___________________________________________________________________

    L A T T I C E   D M E T                             [v20121221]
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""


class FParams1(FDefaultParams):
   def __init__(self, Model, SuperCell, LatticeWf, Fragments):
      #FDefaultParams.__init__(self, Model, SuperCell, LatticeWf, Fragments)
      FDefaultParams.__init__(self, LatticeWf)
      self.Model = Model
      self.SuperCell = SuperCell
      self.Fragments = Fragments

      self.MeanField.MaxIt = 40
      self.DMET.UseInt2e = False
      #self.DMET.DiisThr = 1e99
      #self.DMET.DiisStart = 0
   def __str__(self):
      return "U = %10.5f  J = %10.5f <n> = %10.5f" % (self.Model.U, self.Model.J,
      self.LatticeWf.nElec/(1.*np.prod(self.SuperCell.TotalSize)))

class FHub1dJob(FJob):
   def __init__(self, Params, InputJob = None):
      # Input job: nothing or job to take initial guess for
      # mean field/vloc/embedded wave functions from.
      assert(InputJob is None or isinstance(InputJob, FHub1dJob))
      FJob.__init__(self, Params, [InputJob])
      pass
   def Run(self, Log, InputJobs):
      P = self.Params

      StartingGuess = {}
      if InputJobs is not None:
         assert(len(InputJobs) == 1)
         GuessJob = InputJobs[0]
         StartingGuess = GuessJob.Results
      else:
         StartingGuess['fock'] = P.InitialGuessSpinBias
         if P.MeanField.MaxIt == 1 and P.InitialGuessSpinBias is not None:
            StartingGuess['vcor'] = np.diag(P.InitialGuessSpinBias)
      sc = FSuperCell(P.Model, OrbType=P.LatticeWf.OrbType, **P.SuperCell.__dict__)
      LatticeSystem = FLatticeSystem(WfDecl=P.LatticeWf, SuperCell=sc,
         InitialGuess=StartingGuess['fock'], Params=P.MeanField, Log=Log)
      #LatticeSystem.RunHf(Log=Log)
      DmetContext = FDmetContext(LatticeSystem, P.Fragments, P.DMET, Log)
      DmetResult = DmetContext.Run(Log, StartingGuess)
      # next step: how to organize DMET and the fragmentation,
      # and how to organize storage of results.
      # (need something for vcor, fock, and potentially CI vectors)
      # this.. is probably not it. Just to get stuff running.
      # Note in particular that this just keeps on accumulating Fock
      # matrices in memory which never get released until all job
      # objects are destroyed.
      self.Results = {
         'vcor': DmetResult.FullSystemVcor,
         'dmet': DmetResult,
         'fock': DmetResult.FullSystemFock,
         'Mu':  LatticeSystem.Mu,
         'Gap': LatticeSystem.Gap,
         'Rdm': LatticeSystem.RdmT[0].real,
         'nSites': len(sc.Sites), # number of sites in the super-cell
         'ErrVcor': DmetResult.dVc
      }
   def GetResultTable(self):
      P = self.Params
      R = self.Results
      Out = {}
      if P.LatticeWf.OrbType == "UHF":
         # in the UHF case, we get band gaps and chemical potentials
         # separately for alpha and beta spin. These are supplied as
         # tuples in LatticeSystem.Gap/.Mu. Unpack them.
         Out = {
            "Gap[A]": R["Gap"][0],
            "Gap[B]": R["Gap"][1],
            "Mu[A]": R["Mu"][0],
            "Mu[B]": R["Mu"][1],
            '<Sz>': (P.LatticeWf.nElecA() - P.LatticeWf.nElecB()) / (1.*R["nSites"]),
         }
      else:
         assert(P.LatticeWf.OrbType == "RHF")
         Out = {
            "Mu": R["Mu"],
            "Gap": R["Gap"],
         }
      Out.update({
         "U": P.Model.U,
         "J": P.Model.J,
         "Delta": P.Model.Delta,
         "Fragments": P.Fragments,
         "E/Site": R["dmet"].TotalEnergy,
         "<n>[DMET]": R["dmet"].TotalElec,
         '<n>': P.LatticeWf.nElec / (1.*R["nSites"]),
         "nSites": R["nSites"]*int(P.LatticeWf.OrbOcc())/2,
         'ErrVcor': R["ErrVcor"]
      })
      return Out, {
         "<Sz>": "average number of spin-up minus spin-down electrons per site",
         "<n>": "charge density per site at mean-field level (as input)",
         "<n>/DMET": "charge density per site at DMET level",
      }

def PrintTable(Log, Results, Desc = None, SortKeys = None):
   if SortKeys is None: SortKeys = []

   def CmpResults(a,b):
      for Key in SortKeys:
         ic = cmp(Key in a, Key in b)
         if ic != 0: return ic
         if Key not in a: continue # neither has this result. go on.
         ic = cmp(a[Key], b[Key])
         if ic != 0:
            return ic
      return 0
   Results.sort(CmpResults)

   # make a list of all keys, for the table captions.
   AllKeys = set([])
   for Result in Results:
      AllKeys |= set(Result.keys())
   def CmpSortKeys(a,b):
      ic = cmp(a in SortKeys, b in SortKeys)
      if ic != 0: return -ic # keys which we sort by go first.
      if a in SortKeys:
         # primary sorting keys go first.
         return cmp(SortKeys.index(a), SortKeys.index(b))
      else:
         # sort rest lexicographically, by name
         return cmp(a,b)
   AllKeys = list(AllKeys)
   AllKeys.sort(CmpSortKeys)

   def FmtV(v):
      if isinstance(v,float):
         return "%14.8f" % v
      return str(v).replace(" ","")
      # ^- replace: to simplify parsing the result tables. space only
      #             used as separator between columns.

   Lines = []
   for Result in Results:
      Line = []
      for Key in AllKeys:
         if Key not in Result:
            Line.append("")
         else:
            v = Result[Key]
            Line.append(FmtV(v))
      Lines.append(Line)

   Fmts = []
   for iCol in range(len(AllKeys)):
      Max = len(AllKeys[iCol])
      for Line in Lines:
         Max = max(Max, len(Line[iCol]))
      Fmts.append("{:^%is}" % Max)

   Caption = "  ".join(Fmt.format(Key) for (Key,Fmt) in zip(AllKeys,Fmts))
   Log(Caption + "\n" + "-" * len(Caption))
   for Line in Lines:
      Log("  ".join(Fmt.format(Val) for (Val,Fmt) in zip(Line,Fmts)))


   if Desc:
      Log()
      Log("Notes:")
      for (ItemName, ItemDesc) in Desc.items():
         Log("  {:18} {}", ItemName, ItemDesc)

def main():
   np.set_printoptions(precision=3,linewidth=10060,suppress=False,threshold=np.nan)
   Log = FOutputLog()
   Log(Banner)

   Jobs = []
   Ln = 8
   task = ['La2CuO4','LaNiO3','other'][1]
   nImps = [2]
   J = 0.5
   #read in input file
   #specify params: U, J,Delta, shift into ModelParam class
   #specify params: nel, nImp, ClusterSize in ScParam class
   for nImp in nImps:
      #Fragments = [('CI(2)',list(range(nImp)))]
      if task=='La2CuO4':
         assert(nImp % 4 == 0)
         ImpListOriginal = [0, 1, 14, 15]
         ImpList = []
         for j in range(nImp / 4):
            ImpList += [i + 28*j for i in ImpListOriginal]
         Fragments = [('FCI', ImpList)]
         FModelClass = FHubbardModel_La2CuO4
         ScParams = ToClass({'TotalSize': [Ln,Ln,Ln], 'PhaseShift': [-1,-1,-1], 'ClusterSize': [1,1,1]})
      elif task=='LaNiO3':
         assert(nImp % 2 == 0)
         ImpListOriginal = [0, 1]
         ImpList = []
         for j in range(nImp / 2):
           ImpList += [i + 11*j for i in ImpListOriginal]
         Fragments = [('FCI', ImpList)]
         FModelClass = FHubbardModel_LaNiO3
         ScParams = ToClass({'TotalSize': [Ln,Ln,Ln], 'PhaseShift': [-1,-1,-1], 'ClusterSize': [1,1,1]})
      else:
         FModelClass = FHubbardModel2dSquare
         ScParams = ToClass({'TotalSize': [Ln,Ln], 'PhaseShift': [-1,-1], 'ClusterSize': [3,2]})
         ModelParams = {'t': 1., "U": 1.0, "J": 1. }
      # make a super-cell in order to determine fillings with unique ground
      # states at tight-binding level.
      StartGuessNextU = None

      for U in [4.0]:
         StartGuess = StartGuessNextU # <- start with last U's half-filling result.
         StartGuessNextU = None
         for Delta in [2.]:
         #for Delta in [-1.3121]:         
            ModelParams = {'t': 1., "U": U, "J": J, "Delta": Delta }
            Model = FModelClass(**ModelParams)
            SuperCell = FSuperCell(Model, **ScParams.__dict__)
            #AllowedOccupations = SuperCell.CalcNonDegenerateOccupations(ThrDeg=1e-5)
            #Occs = AllowedOccupations[:len(AllowedOccupations)/2+1]
            #Occs = Occs[-1:] # <- only half filling
            #Occs = [13312]
            Occs = [Ln ** 3 * (2/2 + 9)]
            for Occ in reversed(Occs):
               sp = 0
               #sp = 1 # spin polarization.
               #iOccA = AllowedOccupations.index(Occ)
               LatticeWf = FWfDecl(nElecA=Occ,
                              nElecB=Occ,
                              OrbType="UHF") #_UHF/RHF
               #LatticeWf = FWfDecl(nElecA=AllowedOccupations[iOccA+sp],
               #               nElecB=AllowedOccupations[iOccA-sp],
               #               OrbType="UHF") #_UHF/RHF
               P = FParams1(
                   Model=Model,#ToClass(ModelParams),
                   SuperCell=ScParams,
                   LatticeWf=LatticeWf,
                   Fragments=Fragments)
                   #P.DMET.DiisStart = 0
                   #P.DMET.DiisThr = 1e-40
               P.DMET.MaxIt = 40
               P.MeanField.MaxIt = 1 # disable iterations.
               P.MeanField.DiisStart = 8
               P.InitialGuessSpinBias = None
               if 1:
                  # add some bias to make the system preferrably go into
                  # anti-ferromagnetic solutions. Otherwise we will always
                  # get RHF solutions via UHF, even if the UHF solution is
                  # lower.
                  bias = ModelParams["U"]/2
                  shift = 0.
                  shift = ModelParams["U"]/2 if P.MeanField.MaxIt == 1 else 0.
                  if P.LatticeWf.OrbType == "UHF":
                     P.InitialGuessSpinBias = np.zeros(2*len(SuperCell.UnitCell))
                     for (index, site) in enumerate(Fragments[0][1]):
                        if index % 2 == 0:
                           sign = 1
                        else:
                           sign = -1
                        P.InitialGuessSpinBias[site * 2] += shift + sign * bias
                        P.InitialGuessSpinBias[site * 2 + 1] += shift - sign * bias
                     #print P.InitialGuessSpinBias
                     #raise SystemExit
                  else:
                     P.InitialGuessSpinBias = np.zeros(1*len(SuperCell.UnitCell))
                     for site in Fragments[0][1]:
                        P.InitialGuessSpinBias[site] = shift
               Jobs.append(FHub1dJob(P, InputJob=StartGuess))
               #print Jobs
               StartGuess = Jobs[-1]
               if 0:
                  # enable this to propagate starting guesses from one U
                  # to another. Here disabled for the honeycomb hubbard case,
                  # because at some U between 3. and 4. the solution changes character.
                  if StartGuessNextU is None:
                     StartGuessNextU = Jobs[-1]

   def PrintJobResults(Log, JobsDone):
      with Log.Section("r%03x" % len(JobsDone), "RESULTS AFTER %i OF %i JOBS:" % (len(JobsDone), len(Jobs)), 1):
         AllResults = []
         for Job in JobsDone:
            Result, Desc = Job.GetResultTable()
            AllResults.append(Result)
         PrintTable(Log, AllResults, Desc, SortKeys=["U", "J", "Delta", "<n>", "Fragments", "ErrVcor"])
   JobGroup = FJobGroup(Jobs)
   JobGroup.Run(Log, PrintJobResults)
   # gathering output
   """
   length = len(Jobs)
   dataoutput = np.zeros((length,5),float)
   for i in range(length):
      dataoutput[i][0] = Jobs[i].Params.Model.U
      dataoutput[i][1] = Jobs[i].Results["dmet"].TotalEnergy
      chargeDensity=FmtRho('charge density',Jobs[i].Results["Rdm"],'_',Jobs[i].Params.LatticeWf.OrbType)
      spinDensity=FmtRho('spin density',Jobs[i].Results["Rdm"],'S',Jobs[i].Params.LatticeWf.OrbType)
      density1 = spinDensity[37:51]
      density2 = spinDensity[53:67]
      AForder = abs(float(density1)-float(density2))/2
      Mu = Jobs[i].Results["Mu"]
      Gap = Jobs[i].Results["Gap"]
      if (Jobs[i].Params.LatticeWf.OrbType=="UHF"):
         dataoutput[i][2] = Mu[0]
      else:
         dataoutput[i][2] = Mu
      if (Jobs[i].Params.LatticeWf.OrbType=="UHF"):
         dataoutput[i][3] = Gap[0]
      else:
         dataoutput[i][3] = Gap
      dataoutput[i][4] = AForder
   filename = 'honeycombLn='
   filename = filename+str(Ln)
   filename = filename+'nImp=4'
   print task
   if task=='honeycombl1':
      filename = filename+'l1'
   elif task=='honeycombl2':
      filename = filename+'l2'
   elif task=='honeycombo':
      filename = filename+'o'
   elif task=='honeycombx':
      filename = filename+'x'
   filename = filename+'.npy'
   np.save(filename,dataoutput)
   #np.save("dmet_resultLn=36nImp=61.npy",dataoutput)
   """
   dataoutput = {
      "U":[],
      "E":[],
      "Mu":[],
      "Gap":[],
      "AFOrder":[]
   }
   for Job in Jobs:
      dataoutput["U"].append(Job.Params.Model.U)
      dataoutput["E"].append(Job.Results["dmet"].TotalEnergy)
      if (Job.Params.LatticeWf.OrbType=="UHF"):
         dataoutput["Mu"].append(Job.Results["Mu"][0])
         dataoutput["Gap"].append(Job.Results["Gap"][0])
      else:
         dataoutput["Mu"].append(Job.Results["Mu"])
         dataoutput["Gap"].append(Job.Results["Gap"])
      #chargeDensity=FmtRho('charge density',Jobs[i].Results["Rdm"],'_',Jobs[i].Params.LatticeWf.OrbType)
      spinDensity=FmtRho('spin density',Job.Results["Rdm"],'S',Job.Params.LatticeWf.OrbType)
      density1 = spinDensity[37:51]
      density2 = spinDensity[53:67]
      dataoutput["AFOrder"].append(abs(float(density1)-float(density2))/2)

   filename = task
   filename = filename+"Ln="+str(Ln)
   filename = filename+'nImp='+str(nImp)
   filename = filename+'.pickle'
   resultfile = open(filename, 'w')
   p.dump(dataoutput, resultfile)
   resultfile.closed
   print "Result File:", filename

   #print length
   #a = array[4]
   #U = Jobs[0].Params.Model.U
   #E = Jobs[0].Results["dmet"].TotalEnergy
   #print "the jobs is:"
   #print Jobs[0].Params
   #print Jobs[0].Results["nSites"]
   #print Jobs[0].Results["Gap"]
   #print Jobs[0].Results["Mu"]
   #chargeDensity=FmtRho('charge density',Jobs[0].Results["Rdm"],'_',Jobs[0].Params.LatticeWf.OrbType)
   #spinDensity=FmtRho('spin density',Jobs[0].Results["Rdm"],'S',Jobs[0].Params.LatticeWf.OrbType)
   #print chargeDensity
   #print spinDensity
   #density = spinDensity[37:51]
   #print density
   #density = spinDensity[53:67]
   #print density
   #print len(chargeDensity)
   #print Jobs[0].Results["Rdm"]


main()

