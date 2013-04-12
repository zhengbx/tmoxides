/* This program is free software. It comes without any warranty, to the extent
 * permitted by applicable law. You may use it, redistribute it and/or modify
 * it, in whole or in part, provided that you do so at your own risk and do not
 * hold the developers or copyright holders liable for any claim, damages, or
 * other liabilities arising in connection with the software.
 * 
 * Developed by Gerald Knizia, 2010--2012.
 */

#include <stdlib.h> // only for rand()/RAND_MAX
#include <cctype> // for isalnum / isspace
#include <sys/time.h> // for gettimeofday()
#include <fstream>

#include "Fci.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

double GetTime() {
#ifdef NO_RT_LIB
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // ^- interesting trivia: CLOCK_MONOTONIC is not guaranteed
    //    to actually be monotonic.
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif // NO_RT_LIB
};

bool
    g_BosonicSigns = false;


struct FExtDetDesc;

struct FFciContext : public FFciData
{
    FOrbStringAdrTable
        AdrA, AdrB; // addressing for alpha and beta strings;
    FFciVectorPtr
        // current coefficient vector, residual vector, and
        // diagonal hamiltonian vector (i.e., Diag[I] = <I|H|I>)
        pCoeff, pResid, pDiagH;
    bool
        // if true, re-absorb the 1-electron part of the Hamiltonian
        // into the two-electron part instead of doing it explicitly.
        // Of course, this will only work if the 2e part is defined
        // in full (not for Hubbard systems etc).
        Absorb1e;

    typedef std::vector<FHamiltonianTermPtr>
        FHamiltonianTermList;
    FHamiltonianTermList
        HamiltonianTerms;

    TArray<double>
        // nOrb * nOrb matrices. ModCoreH contains the core hamiltonian (or
        // core Fock matrix in presence of frozen core orbitals), and
        // additionally the modifications of h0 due to the changed order
        // of the 2e-integrals (see Init() function).
        ModCoreH_A, ModCoreH_B,
        // nOrb x nOrb basis expressing the orbitals in which the calculations
        // are done in terms of the input orbitals. This matrix is supposed to
        // be unitary.
        BasisA, BasisB;
    FFciOptions
        &Options;
    FPSpace
        PSpace;

    TArray<double>
        Rdm1A, Rdm1B;
    int
        iPartialTrace;

    explicit FFciContext(FFciOptions &Options_)
        : Options(Options_)
    {
    };
    void Init(int ptrace, bool BaseOnly, FMemoryStack &Mem);
    void InitPSpace(FMemoryStack &Mem);
    void ChangeOrbitalBasis(FMemoryStack &Mem);
    void MakeDiagonalH(FFciVector &DiagH);
    void InvertPSpaceVector( double *pOut, double *pIn, double E, FMemoryStack &Mem );
    enum FUpdateOptions {
        UPDATE_DavidsonJacobi = 0x01,
        UPDATE_IncrementC = 0x02 // if not set, the new c vector will have *low* overlap with the old one.
    };
    void UpdateCiVector( FFciVector &r, FFciVector &c, FFciVector &DiagH, double Energy, double Shift, uint Flags, FMemoryStack &Mem );

    // transforms CoreH and Int2e into the orbital basis provided
    // in pNewOrbs, which is a nOrb * nOrb matrix containing the new
    // orbitals in terms of the old orbitals.
    void TransformIntegrals(double const *pNewOrbsA, double const *pNewOrbsB, FMemoryStack &Mem);
    void TransformIntegrals2e(double *pInt2e, double const *pNewOrbXX, double const *pNewOrbsYY, FMemoryStack &Mem);


    // make the (transition) density matrix
    //   pRdm[r + nOrb * s] = <cl|c^r_s|cr>
    void Make1Rdms(double *pRdmA, double *pRdmB, uint nOrb, FFciVector &cl, FFciVector &cr, FMemoryStack &Mem);

    // evaluates the (nOrb x nOrb) x (nOrb x nOrb) 2-RDM-like object:
    //     pRdm2[rs,tu] = <c|E^r_s E^t_u|c>
    // If SpinPhase == (-1), then instead of E-operators, the operators
    //    {\tilde E}^r_s = a^{rA} a_{sA} - a^{rB} a_{sB}
    // are applied (with a minus between the spin-orbital substitutions on
    // the rhs instead of plus!).
    void Make2RDM(double *pRdm2, FFciVector &c, double SpinPhase, FMemoryStack &Mem_);

    // returns error code
    int Run(FMemoryStack &Mem);
    // print time elapsed since since t0 = GetTime() was set.
    void PrintTimingT0(char const *p, double t0);
    // print time "t"
    void PrintTiming(char const *p, double t);
    void PrintResult(std::string const &s, double f, int iState);

public:
    double
        fEnergyFci,
        fEnergyPTrace;
private:
    std::string
        MethodName;
};


// keeps all data required to evaluate the energy of individual determinants.
struct FDiagonalHEvalData
{
    TArray<double>
        // JAA: (ii|jj), KAA: (ij|ij)
        JAA, KAA, KBB, JBB, JAB, KAB,
        // <i|ModCoreH|i>
        hbarA, hbarB;
    uint
        nOrb;
    void InitAB(double *pInt2e_AA, double *pInt2e_BB, double *pInt2e_AB,
            double *pCoreH_A, double *pCoreH_B, uint nOrb, uint nPairs, bool C1_Integrals);

    double operator ()(FOrbPat StrA, FOrbPat StrB) const;
};

static void GetDiagonalPairInts2e(TArray<double> &JAA, TArray<double> &KAA, double const *pInt2e, uint nPairs, uint nOrb)
{
    JAA.resize(nOrb * nOrb);
    KAA.resize(nOrb * nOrb);
    for ( uint i = 0; i < nOrb; ++ i )
        for ( uint j = 0; j <= i; ++ j ) {
            uint ij = i*(i+1)/2 + j;
            uint ii = i*(i+1)/2 + i;
            uint jj = j*(j+1)/2 + j;
            KAA[i + nOrb*j] = pInt2e[ij + nPairs * ij];
            KAA[j + nOrb*i] = pInt2e[ij + nPairs * ij];
            JAA[i + nOrb*j] = pInt2e[ii + nPairs * jj];
            JAA[j + nOrb*i] = pInt2e[jj + nPairs * ii];
        };
}

static void GetDiagonalPairInts1e(TArray<double> &hbar, double const *pKAA, double const *pCoreH, uint nOrb, bool C1_Integrals)
{
    if ( pCoreH == 0 )
        return hbar.clear();

    // [1], eq. (10)
    hbar.resize(nOrb);
    for ( uint i = 0; i < nOrb; ++ i ) {
//         hbar[i] = ModCoreH[i + nOrb * i];
        hbar[i] = pCoreH[i + nOrb * i];
        if ( !C1_Integrals ) {
            for ( uint k = 0; k < nOrb; ++ k )
                hbar[i] -= .5 * pKAA[i + nOrb * k];
        }
        // ^- that should be equivalent to taking ModCoreH and ignoring pKAA.
    }
}

void FDiagonalHEvalData::InitAB(double *pInt2e_AA, double *pInt2e_BB, double *pInt2e_AB,
    double *pCoreH_A, double *pCoreH_B, uint nOrb_, uint nPairs, bool C1_Integrals)
{
    nOrb = nOrb_;
    GetDiagonalPairInts2e(JAA, KAA, pInt2e_AA, nPairs, nOrb);
    GetDiagonalPairInts2e(JBB, KBB, pInt2e_BB, nPairs, nOrb);
    GetDiagonalPairInts2e(JAB, KAB, pInt2e_AB, nPairs, nOrb);
    GetDiagonalPairInts1e(hbarA, &KAA[0], pCoreH_A, nOrb, C1_Integrals);
    GetDiagonalPairInts1e(hbarB, &KBB[0], pCoreH_B, nOrb, C1_Integrals);
}

double FDiagonalHEvalData::operator ()(FOrbPat StrA, FOrbPat StrB) const
{
    assert( hbarA.empty() == hbarB.empty());
    double
        fElem = 0;
    // spin-unrestricted case. Different alpha/beta integrals.
    // iterate over active orbitals i and j.
    for ( uint i = 0; i < nOrb; ++ i ) {
        uint
            // alpha/beta occupations of orbitals i and j in |I>
            niA = (StrA >> i) & 1,
            niB = (StrB >> i) & 1;
        if ( niA + niB == 0 )
            continue;
        if ( !hbarA.empty() )
            fElem += niA * hbarA[i] + niB * hbarB[i];
        for ( uint j = 0; j < nOrb; ++ j ) {
            uint
                njA = (StrA >> j) & 1,
                njB = (StrB >> j) & 1,
                ij = i + nOrb * j;
            fElem += niA*njB*JAB[ij] + (niA*njA)*.5*JAA[ij] + (niB*njB)*.5*JBB[ij];
            fElem += .5*KAA[ij]*(niA*(1-njA)) + .5*KBB[ij]*(niB*(1-njB));
        }
    }
    return fElem;
}


void FFciContext::MakeDiagonalH(FFciVector &DiagH)
{
    MethodName = "FCI";

    FDiagonalHEvalData
        EvH;
    // make diagonal J/K integrals for forming diagonal hamiltonian elements.
    EvH.InitAB(pInt2e_AA, pInt2e_BB, pInt2e_AB, (Absorb1e? 0:pCoreH_A), (Absorb1e? 0:pCoreH_B), nOrb, nPairs, C1_Integrals);

    // [1], eq. (9)

    // iterate over determinants |I>.
    #pragma omp parallel for
    for ( uint64_t iStrB = 0; iStrB < AdrB.nStr(); ++ iStrB )
    for ( uint64_t iStrA = 0; iStrA < AdrA.nStr(); ++ iStrA ) {
        FOrbPat StrA = AdrA.MakePattern(iStrA);
        FOrbPat StrB = AdrB.MakePattern(iStrB);

        // remaining difference to molpro's fci code: here the
        // configurational energies are not averaged. I'm not quite
        // sure how to best do it at this moment, and the code will
        // probably work without it.
        DiagH(iStrA, iStrB) = EvH(StrA, StrB);
//         cout << format("DiagH[#A=%4i #B=%4i] = %16.10f\n") % iStrA % iStrB % DiagH(iStrA, iStrB);
//         cout << format("DiagH[%s] = %14.7f\n") % FmtDet(StrA,StrB,nOrb) % fElem;
    }
//     PrintMatrixGen(cout, DiagH.pData, 10,1,10,DiagH.nStrA, "DiagH: 10x10");
}

struct FEnergySortPred
{
    double const *pDiagH;
    bool operator () ( FAdr i, FAdr j ){
        return pDiagH[i] < pDiagH[j];
    }
};


void FFciContext::InitPSpace(FMemoryStack &Mem)
{
    uint
        nPSpaceMax = std::min((uint)Options.nPSpace, (uint)pCoeff->nData());
    FPSpace::FConfList
        Confs;
    if ( nPSpaceMax >= pCoeff->nData() ) {
        // take all configurations, in original order.
        for ( std::size_t i = 0; i < std::min((std::size_t)pCoeff->nData(), (std::size_t)nPSpaceMax); ++ i )
            Confs.push_back(i);
    } else {
        // take n configurations lowest in energy as p-space.
        TArray<FAdr>
            Idx;
        Idx.reserve(pDiagH->nData());
        for ( FAdr i = 0; i < pDiagH->nData(); ++ i )
            Idx.push_back(i);
        FEnergySortPred
            EPr;
        EPr.pDiagH = pDiagH->pData;
        std::sort(Idx.begin(), Idx.end(), EPr);
        for ( std::size_t i = 0; i < std::min(Idx.size(), (std::size_t)nPSpaceMax); ++ i )
            Confs.push_back(Idx[i]);
        if ( 0 ) {
            cout << format("pSpace: LastTaken = %16.8f  FirstLeft = %16.8f\n") % EPr.pDiagH[Confs.back()] % EPr.pDiagH[Idx[Confs.size()]];
            double fFirstLeft = EPr.pDiagH[Idx[Confs.size()]];
            while ( !Confs.empty() && std::abs(EPr.pDiagH[Confs.back()] - fFirstLeft) < 1e-7 )
                Confs.pop_back();

        }
    }
    bool
        Spatial = (IntClass == INTCLASS_Spatial);
    FHamiltonianData
        HamiltonianInfo(pInt2e_AA, pInt2e_BB, pInt2e_AB, nPairs,
            Absorb1e? 0 : &ModCoreH_A[0], (Absorb1e? 0 : (Spatial? &ModCoreH_A[0] : &ModCoreH_B[0])),
            nOrb, Absorb1e, Spatial);
    PSpace.Init(Confs, HamiltonianInfo, AdrA, AdrB, Mem);
//     PrintMatrixGen(std::cout, &PSpace.H[0], PSpace.nConfs(), 1, PSpace.nConfs(), PSpace.nConfs(), "p-Space Hamiltonian");
    if ( Options.FileName_PSpaceH != "" ) {
        cout << "*Writing p-space Hamiltonian to '" << Options.FileName_PSpaceH << "'" << std::endl;
        WriteMatrixToFile(Options.FileName_PSpaceH, "p-space Hamiltonian", &PSpace.H[0], PSpace.nConfs(), PSpace.nConfs());
        std::ofstream
            // append declaration of what is what.
            ostr(Options.FileName_PSpaceH.c_str(), std::ofstream::app | std::ofstream::out);
        ostr << "\n\n";
        ostr << "*Orbital basis: " << Options.DiagBasis << "\n";
        ostr << "*Determinants:  ";
        for ( FAdr ip = 0; ip < Confs.size(); ++ ip ) {
            FAdr iPat = PSpace.Confs[ip];
            ostr << " "
                 << FmtDet(PSpace.pAdrA->MakePattern(iPat % PSpace.pAdrA->nStr()),
                           PSpace.pAdrB->MakePattern(iPat / PSpace.pAdrA->nStr()),
                           nOrb);
        }
        ostr << "\n";
    }
}

enum FBasisChangeFlags {
    BASIS_Transpose = 0x01
};
// transform a nOrb x nOrb matrix to a new basis:
//    Out = OrbL^T * In * OrbR
// if BASIS_Transpose is set in Flags, OrbL and Orb^T are transposed.
void BasisChange2(double *pInOut, double const *pOrbL, double const *pOrbR, uint nOrb, FMemoryStack &Mem, uint Flags = 0)
{
    double
        *T1;
    Mem.Alloc(T1, nOrb*nOrb);

    if ( (Flags & BASIS_Transpose) == 0 ) {
        // T1 := CoreH * C
        Mxm(&T1[0],1,nOrb,  &pInOut[0],1,nOrb, pOrbR,1,nOrb,  nOrb,nOrb,nOrb);
        // CoreH := C^T * T1
        Mxm(&pInOut[0],1,nOrb,  pOrbL,nOrb,1,  &T1[0],1,nOrb, nOrb,nOrb,nOrb);
    } else {
        // T1 := CoreH * C
        Mxm(&T1[0],1,nOrb,  &pInOut[0],1,nOrb, pOrbR,nOrb,1,  nOrb,nOrb,nOrb);
        // CoreH := C^T * T1
        Mxm(&pInOut[0],1,nOrb,  pOrbL,1,nOrb,  &T1[0],1,nOrb, nOrb,nOrb,nOrb);
    };

    Mem.Free(T1);
};

void FFciContext::TransformIntegrals2e(double *pInt2e, double const *pNewOrbXX,
        double const *pNewOrbYY, FMemoryStack &Mem)
{
    // transform a set of 2e integrals into a new basis.

    // pNewOrbs: gives new orbitals in terms of old orbitals.
    //    XX: transformation for left side of (XX|YY)
    //    YY: transformation for right side of (XX|YY)
    double
        *T2;
    Mem.Alloc(T2, nOrb*nOrb);

    for ( uint iPair = 0; iPair < nPairs; ++ iPair ){
        // unpack integrals for (**|kl) into T2:
        //    T2[i,j] := (ij|kl)
        for ( uint i = 0; i < nOrb; ++ i )
            for ( uint j = 0; j <= i; ++ j ) {
                double t = pInt2e[(i*(i+1))/2+j + nPairs*iPair];
                T2[i + nOrb*j] = t;
                T2[j + nOrb*i] = t;
            }

        BasisChange2(T2, pNewOrbXX, pNewOrbXX, nOrb, Mem);

        // write back to (ij|kl).
        for ( uint i = 0; i < nOrb; ++ i )
            for ( uint j = 0; j <= i; ++ j )
                pInt2e[(i*(i+1))/2+j + nPairs*iPair] = T2[i + nOrb*j];
    };

    for ( uint iPair = 0; iPair < nPairs; ++ iPair ){
        // unpack integrals for (kl|**) into T2:
        //    T2[i,j] := (kl|ij)
        for ( uint i = 0; i < nOrb; ++ i )
            for ( uint j = 0; j <= i; ++ j ) {
                double t = pInt2e[iPair + nPairs * ((i*(i+1))/2+j)];
                T2[i + nOrb*j] = t;
                T2[j + nOrb*i] = t;
            }

        BasisChange2(T2, pNewOrbYY, pNewOrbYY, nOrb, Mem);

        // write back to (kl|ij).
        for ( uint i = 0; i < nOrb; ++ i )
            for ( uint j = 0; j <= i; ++ j )
                pInt2e[iPair + nPairs*((i*(i+1))/2+j)] = T2[i + nOrb*j];
    };

    Mem.Free(T2);
}


void FFciContext::TransformIntegrals(double const *pNewOrbA, double const *pNewOrbB,
        FMemoryStack &Mem)
{
    // pNewOrbs: gives new orbitals in terms of old orbitals.

    if ( IntClass == INTCLASS_Spatial ) {
        BasisChange2(&CoreH[0], pNewOrbA, pNewOrbA, nOrb, Mem);
        TransformIntegrals2e(&Int2e[0], pNewOrbA, pNewOrbA, Mem);
        if ( !Int2e_ImpPrj.empty() )
            TransformIntegrals2e(&Int2e_ImpPrj[0], pNewOrbA, pNewOrbA, Mem);
    } else {
        BasisChange2(pCoreH_A, pNewOrbA, pNewOrbA, nOrb, Mem);
        BasisChange2(pCoreH_B, pNewOrbB, pNewOrbB, nOrb, Mem);
        TransformIntegrals2e(pInt2e_AA, pNewOrbA, pNewOrbA, Mem);
        TransformIntegrals2e(pInt2e_BB, pNewOrbB, pNewOrbB, Mem);
        TransformIntegrals2e(pInt2e_AB, pNewOrbA, pNewOrbB, Mem);
//         Int2e_ImpPrj.clear(); // not supported.
        if ( !Int2e_ImpPrj_AA.empty() ) {
            TransformIntegrals2e(&Int2e_ImpPrj_AA[0], pNewOrbA, pNewOrbA, Mem);
            TransformIntegrals2e(&Int2e_ImpPrj_BB[0], pNewOrbB, pNewOrbB, Mem);
            TransformIntegrals2e(&Int2e_ImpPrj_AB[0], pNewOrbA, pNewOrbB, Mem);
        }
    }
}

static void SetIdentity(TArray<double> &Matrix, uint nOrb)
{
    Matrix.resize_and_clear(nOrb * nOrb);
    for ( uint i = 0; i < nOrb; ++ i )
        Matrix[i + nOrb * i] = 1.0;
};

static void AddCoulExch(TArray<double> &FockOp, uint nOrb, FInt2eData const &Int2eXX,
    FInt2eData const &Int2eXY, bool TransposeXY, uint nOccX, uint nOccY, double Factor)
{
    for ( uint m = 0; m < nOrb; ++ m )
        for ( uint n = 0; n < nOrb; ++ n ) {
            for ( uint i = 0; i < nOccX; ++ i )
                FockOp[m + nOrb * n] += Factor*(Int2eXX(m,n,i,i) - Int2eXX(m,i,n,i));
            for ( uint i = 0; i < nOccY; ++ i )
                if ( !TransposeXY )
                    FockOp[m + nOrb * n] += Factor*Int2eXY(m,n,i,i);
                else
                    FockOp[m + nOrb * n] += Factor*Int2eXY(i,i,m,n);
        }
};

static void ReadBasisFromFile(TArray<double> &Basis, uint nOrb, std::string const &FileName)
{
    Basis.resize_and_clear(nOrb * nOrb);

    assert(FileName[0] == '!');
    std::ifstream
        File(1 + FileName.c_str());
    // +1: that's a ! to indicate that we are supposed to load a file.
    if ( !File.good() )
        throw std::runtime_error("Failed to open basis input file '" + FileName + "'");
    while ( File.good() ) {
        double f;
        uint   i,j;
        File >> i >> j >> f;
        if ( i > nOrb || j > nOrb || i == 0 || j == 0 )
            throw std::runtime_error("Invalid index in basis input file '" + FileName + "'");
        Basis[(i-1) + nOrb*(j-1)] = f;
    }
};

void FFciContext::ChangeOrbitalBasis(FMemoryStack &Mem)
{
    if ( Options.DiagBasis == "Input" ) {
        // store an identity matrix as basis transformation.
        SetIdentity(BasisA, nOrb);
        SetIdentity(BasisB, nOrb);
    } else if ( Options.DiagBasis[0] == '!' ) {
        // basis is given as input. Read it from a file.
        if ( IntClass == INTCLASS_Spatial ) {
            ReadBasisFromFile(BasisA, nOrb, Options.DiagBasis);
            BasisB = BasisA;
        } else {
            ReadBasisFromFile(BasisA, nOrb, Options.DiagBasis + "_A");
            ReadBasisFromFile(BasisB, nOrb, Options.DiagBasis + "_B");
        }
        TransformIntegrals(&BasisA[0], &BasisB[0], Mem);
        cout << format(" Hamiltonian transformed into externally defined basis.\n") << std::endl;
    } else {
        // Form the Fock operator. This is (1) for checking
        // the 1e- code and (2) for getting a basis in which to diagonalize
        // the Hamiltonian.
        // Note: this only works if the input basis is given by
        //       MO orbitals (occupied first)
        bool CoreHBasis = (Options.DiagBasis == "CoreH");
        if ( !CoreHBasis && Options.DiagBasis != "Fock" )
            throw std::runtime_error("diagonalization basis not recognized: '" + Options.DiagBasis + "'.");
        uint
            nAlpha = (nElec + Ms2)/2,
            nBeta = (nElec - Ms2)/2;
        if ( IntClass == INTCLASS_Spatial )
        {
            if ( FockOp_A.size() != nOrb * nOrb ) {
                FockOp_A = CoreH;
                if ( !CoreHBasis ) {
                    AddCoulExch(FockOp_A, nOrb, Int2e, Int2e,0, nAlpha, nBeta, 0.5);
                    AddCoulExch(FockOp_A, nOrb, Int2e, Int2e,1, nBeta, nAlpha, 0.5);
                }
            }
        } else {
            if ( FockOp_A.size() != nOrb * nOrb || FockOp_B.size() != nOrb * nOrb ) {
                FockOp_A = CoreH_A;
                FockOp_B = CoreH_B;
                if ( !CoreHBasis ) {
                    AddCoulExch(FockOp_A, nOrb, Int2e_AA, Int2e_AB,0, nAlpha, nBeta, 1.0);
                    AddCoulExch(FockOp_B, nOrb, Int2e_BB, Int2e_AB,1, nBeta, nAlpha, 1.0);
                }
            }
        }
//         PrintMatrixGen(std::cout, &FockOpA[0], nOrb, 1, nOrb, nOrb, "Fock/CS");
        TArray<double>
            FockEw(nOrb);
        BasisA = FockOp_A;
        Diagonalize(&FockEw[0], &BasisA[0], nOrb, nOrb);
        if ( IntClass == INTCLASS_Spatial ) {
            BasisB = BasisA;
        } else {
            BasisB = FockOp_B;
            Diagonalize(&FockEw[0], &BasisB[0], nOrb, nOrb);
        }
        TransformIntegrals(&BasisA[0], &BasisB[0], Mem);
        cout << format(" Hamiltonian transformed into %s basis\n") % Options.DiagBasis << std::endl;
    }
}

static void CalcModCoreH(TArray<double> &ModCoreH, FInt2eData const &Int2e, uint nOrb, bool C1_Integrals)
{
    // absorb part of the two electron integrals into one-electron
    // operator. This modification results from
    //
    //    e^kl_ij = e^k_i c^j_l - \delta^l_i e^k_j
    //
    // Putting this into the Hamiltonian we arrive at:
    //
    //    H = h^i_k e^k_i + 1/2 W^ij_kl e^kl_ij
    //      = h^i_k e^k_i + 1/2 W^ij_kl [e^k_i c^j_l - \delta^l_i e^k_j]
    //      = h^i_k e^k_i + 1/2 W^ij_kl e^k_i c^j_l - 1/2 W^lj_kl e^k_j
    //      = [h^i_k - 1/2 W^li_kl] e^k_i + 1/2 W^ij_kl e^k_i c^j_l
    //        ^---- ModCoreH -----^             ^- (ik|jl)
    //
    // Note: This function calculates the 2e part only!
    assert(Int2e.nRows == (nOrb * (nOrb+1))/2);
    ModCoreH.resize_and_clear(nOrb * nOrb);
    if ( !C1_Integrals ) {
        for ( uint m = 0; m < nOrb; ++ m )
            for ( uint n = 0; n < nOrb; ++ n )
                for ( uint i = 0; i < nOrb; ++ i )
                    ModCoreH[m + nOrb * n] -= .5 * Int2e(m,i,n,i);
    }
};


static void Absorb1e_(FInt2eData &Int2e, uint nOrb, uint nElec,
        TArray<double> const &Int1eX, TArray<double> const &Int1eY)
{
    // absorb a one-electron operator into a two-electron operator.
    // This should work as long as the resulting 2e operator is only
    // applied to CI vectors with nElec electrons.
    uint
        nPairs = (nOrb * (nOrb+1))/2;
    assert(nPairs == Int2e.nRows);
    double
        fScale = 1./(nElec);
    for ( uint k = 0; k < nOrb; ++ k ) {
        uint kk = k*(k+1)/2 + k;
        for ( uint i = 0; i < nOrb; ++ i )
            for ( uint j = 0; j <= i; ++ j ) {
                uint ij = (i*(i+1))/2 + j;
                Int2e[kk + nPairs*ij] += fScale * Int1eY[j + nOrb*i];
                Int2e[ij + nPairs*kk] += fScale * Int1eX[j + nOrb*i];
            }
    };
};

static void FormImpPrj2e(FInt2eData &Int2e_ImpPrj, FInt2eData const &Int2e, uint nOrb, int iPartialTrace)
{
    assert(iPartialTrace > 0);
    if ( Int2e_ImpPrj.size() != Int2e.size() ) {
        // form 2e- integrals which have env/sys blocks multiplied by .5
        // and env/env blocks set to zero.
        uint n = static_cast<uint>(iPartialTrace);
        Int2e_ImpPrj = Int2e;

        for ( uint i = 0; i < nOrb; ++ i ) {
            for ( uint j = 0; j <= i; ++ j )
            {
                uint ij = (i*(i+1))/2+j;
                for ( uint k = 0; k < nOrb; ++ k )
                    for ( uint l = 0; l <= k; ++ l )
                    {
                        uint nsys = (int)(i<n) + (int)(j<n) + (int)(k<n) + (int)(l<n);
                        double f = 1.;
                        if ( nsys == 0 ) f = 0.0;
                        if ( nsys == 1 ) f = 0.25; // these don't do anything for 2e- in the hubbard model!
                        if ( nsys == 2 ) f = 0.50; // these don't do anything for 2e- in the hubbard model!
                        if ( nsys == 3 ) f = 0.75; // these don't do anything for 2e- in the hubbard model!
                        if ( nsys == 4 ) f = 1.0;

                        uint kl = (k*(k+1))/2+l;
                        Int2e_ImpPrj(ij,kl) = f * Int2e_ImpPrj(ij,kl);
                    }
            }
        }
    }
}


void FFciContext::Init(int ptrace, bool BaseOnly, FMemoryStack &Mem)
{
    if ( IntClass != INTCLASS_Spatial )
        // cannot use spin projection when A and B orbitals are different.
        Options.ProjectSpin = false;

    CheckSanity();
    if ( Options.DiagBasis != "Input" || Options.FileName_PSpaceH != "" )
        HubbardSys = false;
//     HubbardSys = false;
    Absorb1e = Options.TryAbsorb1e;
    if ( HubbardSys )
        Absorb1e = false;
    if ( HubbardSys ) {
        std::cout << "*Using Hubbard 2e operator:     U =";
        for ( uint i = 0; i < HubU.size(); ++ i ) std::cout << format(" %6.3f") % HubU[i];
        std::cout << std::endl;
    }

    iPartialTrace = ptrace;
    if ( iPartialTrace > 0 ) {
        assert(Absorb1e == true);
        if (IntClass == INTCLASS_Spatial) {
            FormImpPrj2e(Int2e_ImpPrj, Int2e, nOrb, iPartialTrace);
            CalcModCoreH(ModCoreH_A, Int2e_ImpPrj, nOrb, C1_Integrals);
            Absorb1e_(Int2e_ImpPrj, nOrb, nElec, ModCoreH_A, ModCoreH_A);
        } else {
            FormImpPrj2e(Int2e_ImpPrj_AA, Int2e_AA, nOrb, iPartialTrace);
            FormImpPrj2e(Int2e_ImpPrj_BB, Int2e_BB, nOrb, iPartialTrace);
            FormImpPrj2e(Int2e_ImpPrj_AB, Int2e_AB, nOrb, iPartialTrace);
            CalcModCoreH(ModCoreH_A, Int2e_ImpPrj_AA, nOrb, C1_Integrals);
            CalcModCoreH(ModCoreH_B, Int2e_ImpPrj_BB, nOrb, C1_Integrals);
            Absorb1e_(Int2e_ImpPrj_AA, nOrb, nElec, ModCoreH_A, ModCoreH_A);
            Absorb1e_(Int2e_ImpPrj_BB, nOrb, nElec, ModCoreH_B, ModCoreH_B);
            Absorb1e_(Int2e_ImpPrj_AB, nOrb, nElec, ModCoreH_A, ModCoreH_B);
        }
    }

    // transform integrals
    ChangeOrbitalBasis(Mem);

    if ( IntClass == INTCLASS_Spatial ) {
        CalcModCoreH(ModCoreH_A, Int2e, nOrb, C1_Integrals);
        Add(&ModCoreH_A[0], &CoreH[0], 1.0, nOrb*nOrb);

        if ( Absorb1e ) {
            Absorb1e_(Int2e, nOrb, nElec, ModCoreH_A, ModCoreH_A);
            HamiltonianTerms.push_back(new FHamiltonianTerm2e(nOrb, &Int2e[0]));
        } else {
            if ( !HubbardSys )
                HamiltonianTerms.push_back(new FHamiltonianTerm2e(nOrb, &Int2e[0]));
            else
                HamiltonianTerms.push_back(new FHamiltonianTerm2eHub(nOrb, &HubU[0]));
            HamiltonianTerms.push_back(new FHamiltonianTerm1e(nOrb, &ModCoreH_A[0]));
        }
    } else {
        CalcModCoreH(ModCoreH_A, Int2e_AA, nOrb, C1_Integrals);
        CalcModCoreH(ModCoreH_B, Int2e_BB, nOrb, C1_Integrals);
        Add(&ModCoreH_A[0], pCoreH_A, 1.0, nOrb*nOrb);
        Add(&ModCoreH_B[0], pCoreH_B, 1.0, nOrb*nOrb);

        if ( Absorb1e ) {
            Absorb1e_(Int2e_AA, nOrb, nElec, ModCoreH_A, ModCoreH_A);
            Absorb1e_(Int2e_BB, nOrb, nElec, ModCoreH_B, ModCoreH_B);
            Absorb1e_(Int2e_AB, nOrb, nElec, ModCoreH_A, ModCoreH_B);
            HamiltonianTerms.push_back(new FHamiltonianTerm2e(nOrb, pInt2e_AA, pInt2e_BB, pInt2e_AB));
        } else {
            if ( !HubbardSys )
                HamiltonianTerms.push_back(new FHamiltonianTerm2e(nOrb, pInt2e_AA, pInt2e_BB, pInt2e_AB));
            else
                HamiltonianTerms.push_back(new FHamiltonianTerm2eHub(nOrb, &HubU[0]));
            HamiltonianTerms.push_back(new FHamiltonianTerm1e(nOrb, &ModCoreH_A[0], &ModCoreH_B[0]));
        }
    }

    AdrA.Init((nElec + Ms2)/2, nOrb);
    AdrB.Init((nElec - Ms2)/2, nOrb);

    std::cout << format( " Number of alpha strings: %12i\n"
                         " Number of beta strings:  %12i\n"
                         " Number of %s  %12i  [%i mb/vec]\n")
        % AdrA.nStr()
        % AdrB.nStr()
        % (Options.BosonicSigns? "permanents:  " : "determinants:"  )
        % (AdrA.nStr() * AdrB.nStr())
        % (8*(AdrA.nStr() * AdrB.nStr())>>20)
        << std::endl;

    pCoeff = new FFciVector(nElec, nOrb, Ms2, Options.ProjectSpin);
    pResid = new FFciVector(nElec, nOrb, Ms2, Options.ProjectSpin);
    pDiagH = new FFciVector(nElec, nOrb, Ms2, false);

    if ( BaseOnly )
        return;

    double t0 = GetTime();
    MakeDiagonalH(*pDiagH);
    PrintTimingT0("diagonal coupling coefficients",t0);


    if ( 1 ) {
        // start vector: first determinant. If I'm not mistaken, this would
        // be a high spin determinant formed from the lowest orbitals.
        (*pCoeff)[0] = 1.0;
    } else {
        // put in some random stuff into the FCI vector.
        (*pCoeff)[0] = 1000.0;
        for ( uint64_t i = 0; i < pCoeff->nData(); ++ i ) {
            (*pCoeff)[i] += ((double)rand())/RAND_MAX;
        };
        pCoeff->Normalize();
    }


    if ( Options.nPSpace != 0 ) {
        t0 = GetTime();
        InitPSpace(Mem);
        if ( PSpace.nConfs() != 0 )
            PrintTimingT0("p-space Hamiltonian",t0);
    }

    if ( PSpace.nConfs() != 0 ) {
        t0 = GetTime();
        PSpace.MakeHEvs();
        uint
            iPSpaceRoot = Options.iPSpaceRoot;
        if ( iPSpaceRoot > PSpace.nConfs() )
            iPSpaceRoot = PSpace.nConfs() - 1;
        (*pCoeff)[0] = 0.0;
        for ( FAdr i = 0; i < PSpace.nConfs(); ++ i )
            (*pCoeff)[PSpace.Confs[i]] = PSpace.Ev[i + PSpace.nConfs() * iPSpaceRoot];
        PrintTimingT0("preconditioner and |c0>",t0);
    }
}

// form |out> := (H0 - E)^{-1} |in> where H0 is the pspace-Hamiltonian and pIn/pOut
// are vectors of the lenght of the p-space
void FFciContext::InvertPSpaceVector( double *pOut, double *pIn, double E, FMemoryStack &Mem )
{
    FAdr
        nConfP = PSpace.nConfs();
    if ( nConfP == 0 )
        return;
    double
        *pVec;
//     std::cout << boost::format("E0 says: %f") % E << std::endl;
//     PrintMatrixGen(std::cout, pIn, 1, 1, nConfP, 1, "|c0p>");
//     PrintMatrixGen(std::cout, &PSpace.Ew[0], 1, 1, nConfP, 1, "H0 eigenvalues");
    // transform pIn into H0's EV basis: Vec := Ev^T * In
    Mem.Alloc(pVec, nConfP);
    Mxm(pVec,1,nConfP,  &PSpace.Ev[0],nConfP,1,  pIn,1,nConfP,  nConfP,nConfP,1);
    // form (H0 - E)^{-1}.
    for ( uint i = 0; i < nConfP; ++ i )
        pVec[i] = pVec[i]/(PSpace.Ew[i] - E);
    // transform back to original basis: Out := Ev * Vec
    Mxm(pOut,1,nConfP,  &PSpace.Ev[0],1,nConfP,  pVec,1,nConfP,  nConfP,nConfP,1);
    Mem.Free(pVec);
};


void FFciContext::UpdateCiVector( FFciVector &r, FFciVector &c, FFciVector &DiagH, double Energy, double Shift, uint Flags, FMemoryStack &Mem )
{
    double
        fPrevC = 0.0; // factor of previous c.
    if ( 0 != (Flags & UPDATE_IncrementC) )
        fPrevC = 1.0;

    if ( 0 == (Flags & UPDATE_DavidsonJacobi) ) {
        // simple perturbative update.
        for ( uint64_t i = 0; i < c.nData(); ++ i )
            c[i] = fPrevC * c[i] - r[i]/(DiagH[i] - (Energy-Shift));
    } else {
        // perturbative Davidson-Jacobi including p-space preconditioner.
        // See Olsen and Joergensen in their "one billion configurations" paper
        // (eq. 6/7) and Sleijpen&Van der Horst p.274, eq. 17.
        // In practice I don't see it making much difference to the Davidson update.
        // (maybe it is only supposed to work if you have really good approximations
        //  to the inverse Hamiltonian)
        double
            Epsilon1 = 0,
            Epsilon1Denom = 0;
        double
            *pC0p, *pHC0p, *pHinvC0p, *pHinvHC0p;
        uint
            nConfP = PSpace.nConfs();
        Mem.Alloc(pC0p, nConfP);
        Mem.Alloc(pHC0p, nConfP);
        Mem.Alloc(pHinvC0p, nConfP);
        Mem.Alloc(pHinvHC0p, nConfP);
        #pragma omp parallel
        {
            double Epsilon1Denom_ = 0, Epsilon1_ = 0;
            #pragma omp for
            for ( uint64_t i = 0; i < pCoeff->nData(); ++ i ) {
                double
                    c0 = c[i],
                    r0 = r[i], // already includes the "- E0"
                    ifd = 1./(DiagH[i] - (Energy-Shift));
                FAdr
                    ip = PSpace.FindConf(i);
                if ( ip == AdrNotFound ) {
                    Epsilon1Denom_ += c0*c0*ifd;
                    Epsilon1_ += r0*c0*ifd;
                } else {
                    pC0p[ip] = c0;
                    pHC0p[ip] = r0;
                }
            }
            #pragma omp critical
            {
                Epsilon1Denom += Epsilon1Denom_;
                Epsilon1 += Epsilon1_;
            }
        }

    //             std::cout << boost::format("Dot(c0,r0) = %f") % (Dot(*pCoeff,*pResid) + Energy*Dot(*pCoeff,*pResid)) << std::endl;
        InvertPSpaceVector(pHinvC0p, pC0p, Energy-Shift, Mem);
    //             PrintMatrixGen(std::cout, pC0p, nConfP, 1, 1, nConfP, "|C0p>");
    //             PrintMatrixGen(std::cout, pHinvC0p, nConfP, 1, 1, nConfP, "(H0-E)^{-1} |C0p>");
        Epsilon1 += Dot(pHinvC0p, pHC0p, nConfP);
        Epsilon1Denom += Dot(pHinvC0p, pC0p, nConfP);
    //             std::cout << boost::format("Epsilon1 = %f  Epsilon1Denom = %f   1/2 = %f") % Epsilon1 % Epsilon1Denom % (Epsilon1/Epsilon1Denom) << std::endl;
        Epsilon1 /= Epsilon1Denom;

        #pragma omp parallel for
        for ( uint64_t i = 0; i < c.nData(); ++ i ) {
            double
                C1i =  -(r[i] - Epsilon1*c[i]);
            FAdr
                ip = PSpace.FindConf(i);
            if ( ip == AdrNotFound ) {
                double d = C1i/(DiagH[i] - (Energy-Shift));
                c[i] = fPrevC*c[i] + d;
            } else {
                pHC0p[ip] = C1i;
            }
        }
        InvertPSpaceVector(pHinvC0p, pHC0p, Energy-Shift, Mem);
        for ( FAdr ip = 0; ip < nConfP; ++ ip ) {
            FAdr ia = PSpace.Confs[ip];
            c[ia] = fPrevC*c[ia] + pHinvC0p[ip];
        }
    //             PrintMatrixGen(std::cout, pHC0p, nConfP, 1, 1, nConfP, "H0Cp");
    //             PrintMatrixGen(std::cout, pHinvC0p, nConfP, 1, 1, nConfP, "Hinv0Cp");
        Mem.Free(pC0p);
    }
};



int FFciContext::Run(FMemoryStack &Mem)
{
    void
        *pBaseOfMemory = Mem.Alloc(0);

    FSubspaceStates
        SubspaceData(Options.DiisDimension, Options.DiisBlockSize);

    if ( Options.ProjectSpin )
        std::cout << format("\n Spin projection enabled. Restricting search to vectors with S=%i/2.\n") % pCoeff->nSpin();

    std::cout << "\n ITER.       ENERGY      ENERGY CHANGE     VAR       TIME     DIIS" << std::endl;
    bool
        UseDiis = Options.DiisDimension >= 2,
        Converged = false;
    double
        LastEnergy = 0,
        Energy = 0,
        Var2 = 0,
        tStart = GetTime(), tMain = 0, tDiis = 0, tResid = 0, tRest = 0, tSpinProj = 0;
    tMain -= GetTime();
    for ( uint iIt = 0; iIt < Options.nMaxIt; ++ iIt ) {
        if ( Options.ProjectSpin ) {
            tSpinProj -= GetTime();
            pCoeff->ProjectSpin(Mem);
            tSpinProj += GetTime();

            double f = pCoeff->Norm();
            if ( iIt == 0 and f < 1e-4 ) {
                // bad: pspace hamiltonian initial guess or input fci vector
                //      landed at a root with the wrong spin. For a
                //      work-around we re-initialize c as a single-determiant
                //      WF. Otherwise we effectively just have a random input
                //      vector.
                std::cout << "*WARNING: initial |c> had wrong spin. Re-initialized to first determinant.\n";
                pCoeff->Clear();
                pCoeff->pData[0] = 1.;
                f = 1.;
            }
            Scale(pCoeff->pData, 1./f, pCoeff->nData());
        } else {
            pCoeff->Normalize();
        }

        // apply Hamiltonian: |r> := H |c>
        tResid -= GetTime();
        pResid->Clear();
        FHamiltonianTermList::iterator
            itTerm;
        _for_each(itTerm, HamiltonianTerms)
            (*itTerm)->Contract(*pResid, *pCoeff, 1.0, Mem);
        tResid += GetTime();

        // apply subspace conditioner (here: diagonalize H in iterative subspace)
        if ( UseDiis ) {
            tDiis -= GetTime();
            SubspaceData.Apply(pCoeff->pData, pResid->pData, pCoeff->nData(), PSpace, Mem);
            tDiis += GetTime();
            // energy should be in lowest subspace root.
            Energy = SubspaceData.Ew[0];
        } else {
            // calculate energy: E = <c|H|c> = <r|c>
            Energy = Dot(*pResid, *pCoeff);
        }

        // form actual residual (error vector):  |r> = (H - E) |c>
        Add(*pResid, *pCoeff, -Energy );
        Var2 = Dot(*pResid, *pResid);

        if ( iIt == 0 ) LastEnergy = Energy;
        std::cout << format("%4i    %14.8f %14.8f    %8.2e%10.2f  %2i%3i\n")
            % (1+iIt) % (Energy+fCoreEnergy) % (Energy-LastEnergy)% Var2
            % (GetTime() - tStart)
            % (SubspaceData.iThis+1) % SubspaceData.nDimUsed;

        Converged = Var2 < Options.ThrVar;
        if ( Converged || iIt == Options.nMaxIt-1 )
            break;
        LastEnergy = Energy;

        double
            Shift = 0.2;
//             Shift = Var2;
        uint
            UpdateFlags = UPDATE_DavidsonJacobi;
        if ( !UseDiis )
            // if there are subspace states, we need to try to keep the next
            // vector c as orthogonal as possible to those subspace states.
            // If not, we need to keep c as an approximation to the ground state.
            UpdateFlags |= UPDATE_IncrementC;
        UpdateCiVector(*pResid, *pCoeff, *pDiagH, Energy, Shift, UpdateFlags, Mem);
    }
    if ( Options.nMaxIt <= 1 )
        Converged = true; // special mode: just evaluate stuff with input vectors.
    pCoeff->Normalize();
    tMain += GetTime();
    tRest = tMain - tDiis - tResid - tSpinProj;
    if ( !Converged ) {
        cout << format("\n*WARNING: No convergence for root %i."
                       " Stopped at NIT: %i  DEN: %.2e  VAR: %.2e")
                % 0 % Options.nMaxIt % (Energy - LastEnergy) % Var2 << std::endl;
    }
    cout << "\n";
    PrintTiming("main loop", tMain);
    PrintTiming("DIIS", tDiis);
    PrintTiming("residual", tResid);
    if ( tSpinProj != 0 )
        PrintTiming("spin projection", tSpinProj);
    PrintTiming("rest", tRest);


    Energy += fCoreEnergy;

    std::cout << "\n";

    if ( Options.ThrPrintC <= 1.0 )
        pCoeff->Print(std::cout, "c0", Options.ThrPrintC);
    if ( 0 ) {
        pCoeff->Print(std::cout, "c0", 0.05);

        for ( uint iMax = 0; iMax < pCoeff->nStrA; ++ iMax ) {
            double cs = 0;
            for ( uint i = 0; i <= iMax; ++ i )
                for ( uint j = 0; j <= iMax; ++ j ) {
                    double c = (*pCoeff)(i,j);
                    cs += c*c;
                }
            std::cout << format("Accumulated weight of [strings up to %ix%i square (%7.2f%% of all)]: %12.5f")
                % iMax % iMax % (100.*(1+iMax)*(1+iMax)/pCoeff->nData()) % std::sqrt(cs)
                << std::endl;
        };
    }



    if ( 1 ) {
        FHamiltonianTerm1e
            CoreH(nOrb, pCoreH_A, (INTCLASS_Spatial == IntClass)? 0 : pCoreH_B);
        double
            Energy1e;
        pResid->Clear();
        CoreH.Contract(*pResid, *pCoeff, 1.0, Mem);
        Energy1e = Dot(*pResid, *pCoeff);
        PrintResult("<c|h|c>", Energy1e, 0);
    }

    if ( iPartialTrace > 0 ) {
        pResid->Clear();
        boost::intrusive_ptr<FHamiltonianTerm2e>
            pHptrace;
        if ( IntClass == INTCLASS_Spatial )
            pHptrace = new FHamiltonianTerm2e(nOrb, &Int2e_ImpPrj[0]);
        else
            pHptrace = new FHamiltonianTerm2e(nOrb, &Int2e_ImpPrj_AA[0], &Int2e_ImpPrj_BB[0], &Int2e_ImpPrj_AB[0]);
        pHptrace->Contract(*pResid, *pCoeff, 1.0, Mem);
        this->fEnergyPTrace = Dot(*pResid, *pCoeff);
        PrintResult("pTraceSys", fEnergyPTrace, 0);
    }


    PrintResult("ENERGY", Energy, 0);
    Mem.Free(pBaseOfMemory);
    fEnergyFci = Energy;

    return 1 - (int)Converged;
};

void FFciContext::PrintTiming(char const *p, double t)
{
    std::cout << format(" Time for %-35s%10.2f sec")
        % (std::string(p) + ":") % t << endl;
};

void FFciContext::PrintTimingT0(char const *p, double t0)
{
    PrintTiming(p, GetTime() - t0);
};

void FFciContext::PrintResult(std::string const &s, double v, int iState)
{
    std::stringstream
        Caption;
    if ( iState >= 0 ) {
        Caption << format("!%s STATE %i %s") % MethodName % (iState + 1) % s;
    } else {
        Caption << " " + s;
    }
    std::cout << format("%-23s%20.14f") % Caption.str() % v << std::endl;
};





// make the (transition) density matrix
//   pRdm[r + nOrb * s] = <cl|c^r_s|cr>
void FFciContext::Make1Rdms(double *pRdmA, double *pRdmB, uint nOrb, FFciVector &cl, FFciVector &cr, FMemoryStack &Mem)
{
    assert(compatible(cl,cr));
    assert(nOrb == cl.nOrb && nOrb == cr.nOrb);

    memset(pRdmA, 0, sizeof(double) * cl.nOrb * cr.nOrb);
    if ( pRdmB != 0 )
        memset(pRdmB, 0, sizeof(double) * cl.nOrb * cr.nOrb);
    else
        pRdmB = pRdmA;
    Add1RdmForSpin(pRdmA, &cl[0], &cr[0], cl.AdrA, cl.AdrB, 1, cl.nStrA, Mem);
    Add1RdmForSpin(pRdmB, &cl[0], &cr[0], cl.AdrB, cl.AdrA, cl.nStrA, 1, Mem);
    // PrintMatrixGen(std::cout, pRdm, nOrb, 1, nOrb, nOrb, "1-RDM from FCI");
};


// transpose a square matrix  M[iRow,iCol] = pM[iRow + nRowStride * iCol] in-place.
static void TransposeInplaceSqr(double *pM, std::size_t nRowStride, std::size_t nRows, std::size_t nCols )
{
    assert(nRows == nCols); // in-place transposition only possibly for square matrices.
    for ( std::size_t iRow = 0; iRow < nRows; ++ iRow )
        for ( std::size_t iCol = 0; iCol < iRow; ++ iCol )
            std::swap( pM[iRow + nRowStride * iCol], pM[iCol + nRowStride * iRow] );
};


// evaluates the (nOrb x nOrb) x (nOrb x nOrb) 2-RDM-like object:
//     pRdm2[rs,tu] = <c|E^r_s E^t_u|c>
// If SpinPhase == (-1), then instead of E-operators, the operators
//    {\tilde E}^r_s = a^{rA} a_{sA} - a^{rB} a_{sB}
// are applied (with a minus between the spin-orbital substitutions on
// the rhs instead of plus!).
void FFciContext::Make2RDM( double *pRdm2, FFciVector &c, double SpinPhase, FMemoryStack &Mem_ )
{
    assert(c.nOrb == nOrb);
    // we do:
    //   G[ij,kl] += c_I [<I|c^r\alpha c_s\alpha|K> + <I|c^r\beta c_s\beta|K>] *
    //                   [<K|c^t\alpha c_u\alpha|J> + <K|c^t\beta c_u\beta|J>] c_J
    //
    FAdr
        nTgtBlkK = 32,
        nTgtBlkKB = 32;
    uint
        nPairsN = nOrb * nOrb; // non-symmetirc pairs.
    FOrbStringAdrTable
        &AdrA = c.AdrA,
        &AdrB = c.AdrB;

    FMemoryStackArray MemStacks(Mem_);

    memset(pRdm2, 0, sizeof(pRdm2[0]) * nPairsN*nPairsN);

    // loop over blocks of alpha/beta strings of K.
    #pragma omp parallel for schedule(dynamic)
    for ( FAdr iBlockBegKB = 0; iBlockBegKB < AdrB.nStr(); iBlockBegKB += nTgtBlkKB ) {
        FMemoryStack &Mem = MemStacks.GetStackOfThread();

        // get memory for 2-RDM contribution.
        double
            *pRdm2Acc;
        Mem.ClearAlloc(pRdm2Acc, nPairsN * nPairsN);

        FAdr iBlockEndKB = std::min(iBlockBegKB + nTgtBlkKB, AdrB.nStr());
        uint nBlkB = iBlockEndKB - iBlockBegKB; // number of beta-strings in block
        FStrInfo *pInfoA, *pInfoB;

        // find beta determinant strings on |I> and |J> side
        // (both are equal) which connect to Ks within the current beta block.
        Mem.Alloc(pInfoB, nBlkB);
        for ( uint i = 0; i < nBlkB; ++ i ) {
            pInfoB[i].Str = AdrB.MakePattern(iBlockBegKB + i);
            FormStringSubstsForSpin(pInfoB[i].pSubst, pInfoB[i].nSubst, 0, AdrB, pInfoB[i].Str, Mem);
        }

        for ( FAdr iBlockBegKA = 0; iBlockBegKA < AdrA.nStr(); iBlockBegKA += nTgtBlkK )
        {
            FAdr iBlockEndKA = std::min(iBlockBegKA + nTgtBlkK, AdrA.nStr());
            uint nBlkA = iBlockEndKA - iBlockBegKA; // number of alpha-strings in block

            // find alpha determinant strings on |I>/|J> connecting to K.
            Mem.Alloc(pInfoA, nBlkA);
            for ( uint i = 0; i < nBlkA; ++ i ) {
                pInfoA[i].Str = AdrA.MakePattern(iBlockBegKA + i);
                FormStringSubstsForSpin(pInfoA[i].pSubst, pInfoA[i].nSubst, 0, AdrA, pInfoA[i].Str, Mem);
            }

            // get memory for storage of intermediate Dat[kl,K] block
            double
                *pInpK;
            Mem.ClearAlloc(pInpK, nPairsN * nBlkA * nBlkB);

            double
                csum = 0;

            // form Inp[kl,K\alpha K\beta] = <K\alpha|c^k\alpha c_l\alpha|J\alpha> * c[J\alpha,K\beta]
            // (note: includes symmetrization)
            BlockContractCc1_NoSym( pInpK, 1, nOrb, pInfoA, nBlkA, nBlkB, 1*nPairsN, nPairsN*nBlkA,
                AdrA, AdrB, &c(0,iBlockBegKB), 1, c.nStrA, &csum, 1.0);
//             PrintMatrixGen(std::cout, pInpK, nPairsN, 1, nBlkA*nBlkB, nPairsN, "Contrib to RDM2[Alpha]");
            // add Inp[kl,K\alpha K\beta] += SpinPhase * <K\beta|c^k\beta c_l\beta|J\beta> * c[K\alpha,J\beta]
            BlockContractCc1_NoSym( pInpK, 1, nOrb, pInfoB, nBlkB, nBlkA, nBlkA*nPairsN, 1*nPairsN,
                AdrB, AdrA, &c(iBlockBegKA,0), c.nStrA, 1, &csum, SpinPhase);

            if ( csum > ThrNegl ) { // <- mainly helpful for first iterations
                                  //    where c is nearly empty.
                // contract to 2-RDM:
                //     Rdm[rs,ut] += Dat[rs,K] Dat[ut,K].
                Mxm(pRdm2Acc, 1, nPairsN,
                    pInpK,  1, nPairsN,
                    pInpK, nPairsN, 1,
                    nPairsN, nBlkA * nBlkB, nPairsN, true);
                // (note: could use syrk for that.)
//                 PrintMatrixGen(std::cout, pRdm2Acc, nPairsN, 1, nPairsN, nPairsN, "Accumulated RDM2");

            }

            Mem.Free(pInfoA);
        };
        // add to output 2-RDM.
        #pragma omp critical
        Add(pRdm2, pRdm2Acc, 1.0, nPairsN*nPairsN);

        Mem.Free(pInfoB);
        Mem.Free(pRdm2Acc);
    }

    // in the above construction we calculated
    //     <E^s_r E^t_u>
    // instead of
    //     <E^r_s E^t_u>.
    // -> still need to transpose first two indices.
    #pragma omp parallel for
    for ( uint tu = 0; tu < nPairsN; ++ tu )
        TransposeInplaceSqr( &pRdm2[nPairsN*tu], nOrb, nOrb, nOrb );

#ifdef _DEBUG
    double e = 0;
    for ( uint l = 0; l < nOrb; ++ l )
        for ( uint k = 0; k < nOrb; ++ k )
            for ( uint j = 0; j < nOrb; ++ j )
                for ( uint i = 0; i < nOrb; ++ i )
                    e += .5*pRdm2[i+nOrb*j + nPairsN *(k + nOrb*l)] * Int2e(i,j,k,l);
    std::cout << format("!dbg: tr<2rdm x int2e> =  %.8f\n") % e;
    // ^- should give the FCI energy if Absorb1e is true.
#endif

};

static void TransformOrbMatrixSet( double *pMatrices, uint nOrb, uint nMatrixStride, uint nMatrices, double *pBasisT, FMemoryStack &Mem_ )
{
    // transform a number of nOrb x nOrb matrices C[rs,n] by transpose of
    // nOrb x nOrb matrix pBasisT:
    //    Out[n] = pBasisT In[n] pBasisT^T.
    FMemoryStackArray MemStacks(Mem_);
    #pragma omp parallel for
    for ( uint iMatrix = 0; iMatrix < nMatrices; ++ iMatrix ) {
        FMemoryStack &Mem = MemStacks.GetStackOfThread();
        double
            *pT1,
            *pMat = &pMatrices[nMatrixStride * iMatrix];
        Mem.Alloc(pT1, nOrb*nOrb);

        Mxm( &pT1[0],1,nOrb,  &pMat[0],1,nOrb,    &pBasisT[0],nOrb,1,  nOrb,nOrb,nOrb);
        Mxm(&pMat[0],1,nOrb,  &pBasisT[0],1,nOrb, &pT1[0],1,nOrb,  nOrb,nOrb,nOrb);

        Mem.Free(pT1);
    }
}


static void Transform2RDM( double *pRdm2, uint nOrb, double *pBasisT, FMemoryStack &Mem_ )
{
    // transform 2-RDM to input basis (i.e., by inverse of nOrb x nOrb matrix pBasisT).
    // (this assumes that pBasisT is unitary; we actually transform with the transpose.)
    uint
        nPairsN = nOrb * nOrb; // non-symmetirc pairs.
    TransformOrbMatrixSet( pRdm2, nOrb, nPairsN, nPairsN, pBasisT, Mem_ );
    TransposeInplaceSqr( pRdm2, nPairsN, nPairsN, nPairsN );
    TransformOrbMatrixSet( pRdm2, nOrb, nPairsN, nPairsN, pBasisT, Mem_ );
    TransposeInplaceSqr( pRdm2, nPairsN, nPairsN, nPairsN );
};






FFciOptions::FFciOptions()
    : DiisDimension(16), DiisBlockSize(1024), DiagBasis("Fock"), FileName_PSpaceH(""),
      ThrVar(1e-8), ThrPrintC(1.01), nPSpace(200), iPSpaceRoot(0), nMaxIt(2048),
      ProjectSpin(true), BosonicSigns(false), TryAbsorb1e(true)
{}


#ifndef FCI_NO_MAIN
int main(int argc, char *argv[]) {
    FFciOptions
        Options;

    int ptrace, nthreads, nWorkSpaceMb;
    std::string rdm1, rdm2, rdm2s, fci_vec, method, det_overlap, fci_fock;
    po::options_description options_desc("Options");
    options_desc.add_options()
        ("help",
            "print this help message")
        ("subspace-dimension,d", po::value<uint>(&Options.DiisDimension)->default_value(16),
            "maximum Davidson/DIIS subspace dimension")
        ("thr-var,v", po::value<double>(&Options.ThrVar)->default_value(1e-7),
            "residual threshold for convergence")
        ("spin-proj", po::value<bool>(&Options.ProjectSpin)->default_value(true),
            "if 1, the Ms2 value in the input file is also taken as desired total spin. "
            "Trial solutions will be projected onto this spin space. If 0, Ms2 only "
            "designates the spin's M quantum number, and fci will search for solutions "
            "with *any* spin S >= Ms2 (i.e., with Ms2=0 and spin-proj=0, you might get "
            "as ground state a triplet solution with M=0).")
        ("hc-bosons", po::value<bool>(&Options.BosonicSigns)->default_value(false),
            "if true, calculate wave function for hard-core bosons instead of fermions. "
            "That is, wave function is a linear combination of permanents (instead of determinants) "
            "but spin-orbitals still can be occupied at most once (experimental, incompatible with spin-proj). ")
        ("max-iter", po::value<uint>(&Options.nMaxIt)->default_value(2047),
            "maximum number of iterations")
        ("nthread", po::value<int>(&nthreads)->default_value(0),
            "number of OpenMP threads to use (0: use OMP_NUM_THREADS)")
        ("work-memory", po::value<int>(&nWorkSpaceMb)->default_value(20),
            "work space per thread for calculation intermediates, in MB. Increase this if you get 'stack size exceeded' errors.")
        ("diis-block-size", po::value<uint>(&Options.DiisBlockSize)->default_value(1024),
            "block size in kb for subspace vector blocks. if larger than vector size, subspace vectors are not stored on disk.")
        ("method",
            po::value< std::string>(&method)->default_value("FCI"),
            "electronic structure method to use (FCI, CI(n), CC(n), n=1,2,..; CC method work only via mrcc interface)")
        ("ptrace", po::value<int>(&ptrace)->default_value(-1),
            "if set, evaluate the partial trace of the energy over the first n orbitals.")
        ("pspace", po::value<uint>(&Options.nPSpace)->default_value(400),
            "if > 0, explicitly form Hamiltonian matrix <n|H|m> for nP determinants of lowest energy. Used in initial guess and conditioning")
        ("pspace-root", po::value<uint>(&Options.iPSpaceRoot)->default_value(0),
            "root (eigenvector) of the p-space Hamiltonian to use for forming the initial guess. Ground state is #0. If p-space contains all determinants, then this will result in an exact eigenstate (if you see iterations, combine with --spin-proj=0).")
        ("thr-print-c", po::value<double>(&Options.ThrPrintC)->default_value(1.01),
            "threshold for printing of CI vector. Only components with weight > thr are printed. With default threshold, nothing is printed.")
        ("basis",
            po::value<std::string>(&Options.DiagBasis)->default_value("Fock"),
            "orbital basis in which to update the CI vector: [CoreH, Fock, or Input]. "
            "Note: When using 'Fock', the first N input orbitals are assumed to be the "
            "occupied orbitals (as you would get, for example, from MO integrals "
            "after a RHF calculation). The FCI program does not do a HF calculation; 'Fock' "
            "will not work (well) if given non-MO integrals! In that case use --basis=CoreH.")
        ("save-rdm1", po::value<std::string>(&rdm1),
            "if given, store the 1-RDM (in the input basis) in the given file.")
        ("save-rdm2", po::value<std::string>(&rdm2),
            "if given, store the 2-RDM <E^r_s E^t_u> (in the input basis) in the given file.")
        ("save-rdm2s", po::value<std::string>(&rdm2s),
            "if given, store the spin-flipped 2-RDM <(e^rA_sA - e^rB_sB) (e^tA_sA - e^tB_sB)> in the given file.")
        ("save-pspace-h", po::value<std::string>(&Options.FileName_PSpaceH),
            "if given, store the p-space Hamiltonian in the given file name. That is the Hamiltonian "
            "in the space spanned by the p lowest energy determinants in the calculation basis (use with --basis=Input if you need it in the input basis).")
        ("fci-vec", po::value<std::string>(&fci_vec),
            "if given, store the final CI vector in the given file. If the file already exists, read the starting vector from this file.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options_desc), vm);
    po::notify(vm);

    if (nthreads != 0)
        omp_set_num_threads(nthreads);
    else
        nthreads = omp_get_max_threads();

    if (vm.count("help")) {
        cout << options_desc << "\n";
        return 1;
    }

    int
        iStatus = 0; // return value.

    Options.TryAbsorb1e = true;
//     cout << "FIXME: setting Absorb1e = false.\n";
//     Options.TryAbsorb1e = false;
//     if ( fci_vec != "" )
//         Options.TryAbsorb1e = false; // need to apply H on N-1 and N+1 configurations.

    std::string
        FileName = "/tmp/cgk/FCIDUMP";
    if (argc >= 2)
        FileName = std::string(argv[argc-1]);
    else {
        cout << options_desc << "\n";
        return 0;
    }

    Options.DiisBlockSize <<= 10;
    g_BosonicSigns = Options.BosonicSigns;
    if ( Options.BosonicSigns ) {
        cout <<   " WARNING: Don't know how to make CSFs for bosons."
                "\n          Spin projection disabled." << std::endl;
        Options.ProjectSpin = false;
    }
    ct::FMemoryStack2
        // get a block of memory for small dynamically allocated temporaries
        // of unknown compile time size.
        Mem(nthreads * (static_cast<std::size_t>(nWorkSpaceMb) << 20));
    FFciContext
        Context(Options);
    if ( method == "FCI" ) {
        Context.ReadDump(FileName);
        Context.Init(ptrace, det_overlap != "", Mem);
        if ( fci_vec != "" ) {
//             Options.ProjectSpin = 0;
            FILE *File = fopen(fci_vec.c_str(), "r");
            if ( File != 0 ) {
                std::size_t sz1, sz = Context.pCoeff->nData();
                sz1 = fread(Context.pCoeff->pData, sizeof(FScalar), sz, File);
                if ( sz1 != sz ) throw std::runtime_error("Failed to load initial FCI vector '" + fci_vec + "'. File exists but has wrong size.");
                fclose(File);
                Context.pCoeff->Normalize(); // should this be here?
                std::cout << format("\n*read starting FCI vector from file %s.") % fci_vec << std::endl;
            }
        }
        bool DefaultRun = true;
        if ( DefaultRun ) {
            iStatus = Context.Run(Mem);
            if ( fci_vec != "" ) {
                FILE *File = fopen(fci_vec.c_str(), "w");
                if ( File != 0 ) {
                    std::size_t sz1, sz = Context.pCoeff->nData();
                    sz1 = fwrite(Context.pCoeff->pData, sizeof(FScalar), sz, File);
                    if ( sz1 != sz ) throw std::runtime_error("Failed to write final FCI vector '" + fci_vec + "'. Disk full?");
                    fclose(File);
                    std::cout << format("*wrote FCI vector to file %s.") % fci_vec << std::endl;
                } else
                    throw std::runtime_error("Failed to open '" + fci_vec + "' for writing.");
            }
        }
    } else {
    }
    if ( rdm1 != "" || rdm2 != "" || rdm2s != "" || fci_fock != "" ) {
        uint nOrb = Context.nOrb;

        if ( rdm1 != "" ) {
            TArray<double>
                &Rdm1A = Context.Rdm1A,
                &Rdm1B = Context.Rdm1B;
            bool
                Spatial = Context.IntClass == INTCLASS_Spatial;
            if ( method == "FCI" ) {
                Rdm1A.resize(nOrb*nOrb, 0.0); Rdm1B.resize(nOrb*nOrb, 0.0);
                Context.Make1Rdms(&Rdm1A[0], (Spatial? 0 : &Rdm1B[0]), nOrb, *Context.pCoeff, *Context.pCoeff, Mem);
            } else
                if ( Rdm1A.empty() )
                    throw std::runtime_error("non-FCI method: expected 1-RDM to be already made at this point.");

            if ( Spatial ) {
                // transform 1-RDM back to original (input) basis.
                // note: this assumes that Context.Basis is a unitary matrix. it's supposed to be.
                BasisChange2(&Rdm1A[0], &Context.BasisA[0], &Context.BasisA[0], nOrb, Mem, BASIS_Transpose);
                WriteMatrixToFile(rdm1, "1-RDM <1|rs|1>", &Rdm1A[0], nOrb, nOrb);
                std::cout << format("*wrote 1-RDM to file %s.") % rdm1 << std::endl;
            } else {
                BasisChange2(&Rdm1A[0], &Context.BasisA[0], &Context.BasisA[0], nOrb, Mem, BASIS_Transpose);
                WriteMatrixToFile(rdm1 + ".A", "1-RDM <1|rs|1> (Alpha-subst only)", &Rdm1A[0], nOrb, nOrb);
                std::cout << format("*wrote alpha 1-RDM to file %s.A.") % rdm1 << std::endl;

                BasisChange2(&Rdm1B[0], &Context.BasisB[0], &Context.BasisB[0], nOrb, Mem, BASIS_Transpose);
                WriteMatrixToFile(rdm1 + ".B", "1-RDM <1|rs|1> (Beta-subst only)", &Rdm1B[0], nOrb, nOrb);
                std::cout << format("*wrote beta 1-RDM to file %s.B.") % rdm1 << std::endl;
            }
        }
        if ( rdm2 != "" || rdm2s != "" ) {
            if ( method != "FCI" ) throw std::runtime_error("sorry, 2-RDM atm only supported for FCI.");
            if ( Context.IntClass != INTCLASS_Spatial ) throw std::runtime_error("sorry, 2-RDM atm not supported for unrestricted orbitals.");
            uint nPairsN = nOrb * nOrb; // non-symmetric pairs.
            double *pRdm2;
            Mem.Alloc(pRdm2, nPairsN * nPairsN);

            if ( rdm2 != "" ) {
                Context.Make2RDM(pRdm2, *Context.pCoeff, 1.0, Mem);
                Transform2RDM(pRdm2, nOrb, &Context.BasisA[0], Mem);
                WriteMatrixToFile(rdm2, "2-RDM [rs,tu] = <1|E^r_s E^t_u|1>", &pRdm2[0], nPairsN, nPairsN);
                std::cout << format("*wrote 2-RDM to file %s.") % rdm2 << std::endl;
            }
            if ( rdm2s != "" ) {
                Context.Make2RDM(pRdm2, *Context.pCoeff, -1.0, Mem);
                Transform2RDM(pRdm2, nOrb, &Context.BasisA[0], Mem);
                WriteMatrixToFile(rdm2s, "2-RDM-S [rs,tu] = <1|(a^rA_sA-a^rB_sB) (a^tA_uA-a^tB_sB)|1>", &pRdm2[0], nPairsN, nPairsN);
                std::cout << format("*wrote spin-flipped 2-RDM to file %s.") % rdm2s << std::endl;
            }

            Mem.Free(pRdm2);
        }

    }
    return iStatus;
};
#else
    #include "CtFciInterface.inl.h"
#endif // FCI_NO_MAIN


// kate: indent-mode normal; indent-width 4;
