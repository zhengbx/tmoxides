/* This program is free software. It comes without any warranty, to the extent
 * permitted by applicable law. You may use it, redistribute it and/or modify
 * it, in whole or in part, provided that you do so at your own risk and do not
 * hold the developers or copyright holders liable for any claim, damages, or
 * other liabilities arising in connection with the software.
 * 
 * Developed by Gerald Knizia, 2010--2012.
 */

#ifndef FORTRAN_INT_H
#define FORTRAN_INT_H

#include <boost/cstdint.hpp>
//#include <stdint.h>
// ^- get those from C99 stdint.h (technically not C++ standard)
//    or from boost cstdint.hpp from outside.

// basic fortran integer type.
#ifdef _I4_
   typedef boost::int32_t FORTINT;
#else
//    typedef boost::int64_t FORTINT;
   typedef long FORTINT;
   // ^- (should default to 32bit for 32bit systems)
#endif

typedef FORTINT const
   &FINTARG;
typedef double const
   &FDBLARG;

// number of trailing underscores functions with FORTRAN signature.
#ifndef FORT_UNDERSCORES
   #define FORT_UNDERSCORES 1 /* most linux fortran compilers do it like that. */
#endif

// macro for defining C functions which are callable from FORTRAN side
// (and reverse direction):
// A function
//     void FORT_Extern(bla,BLA)(FORTINT &a, double &b);
// can be called as a function
//     subroutine bla(a, b)
//       integer :: a
//       double precision :: b
// from Fortran side.
#ifdef FORT_UPPERCASE
  #if FORT_UNDERSCORES == 0
    #define FORT_Extern(lowercasef,UPPERCASEF) UPPERCASEF
  #elif FORT_UNDERSCORES == 1
    #define FORT_Extern(lowercasef,UPPERCASEF) UPPERCASEF##_
  #elif FORT_UNDERSCORES == 2
    #define FORT_Extern(lowercasef,UPPERCASEF) UPPERCASEF##__
  #else
    #error "should define FORT_UNDERSCORES for fortran function signatures."
  #endif
#else
  #if FORT_UNDERSCORES == 0
    #define FORT_Extern(lowercasef,UPPERCASEF) lowercasef
  #elif FORT_UNDERSCORES == 1
    #define FORT_Extern(lowercasef,UPPERCASEF) lowercasef##_
  #elif FORT_UNDERSCORES == 2
    #define FORT_Extern(lowercasef,UPPERCASEF) lowercasef##__
  #else
    #error "should define FORT_UNDERSCORES for fortran function signatures."
  #endif
#endif

#endif /* FORTRAN_INT_H */
