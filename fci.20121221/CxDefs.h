/* This program is free software. It comes without any warranty, to the extent
 * permitted by applicable law. You may use it, redistribute it and/or modify
 * it, in whole or in part, provided that you do so at your own risk and do not
 * hold the developers or copyright holders liable for any claim, damages, or
 * other liabilities arising in connection with the software.
 * 
 * Developed by Gerald Knizia, 2010--2012.
 */

#ifndef CX_DEFS_H
#define CX_DEFS_H

#define AIC_NO_THROW throw()
#define CX_NO_THROW AIC_NO_THROW

// define macros for restricted pointer extensions. By defining a
// pointer as ``restricted'', we promise the compiler that
// the pointer is not aliased in the current scope

#ifdef __GNUC__ // g++
   #define AIC_RP __restrict__
#elif _MSC_VER // microsoft c++. intel c++ may also understand this syntax.
   #define AIC_RP __restrict
#elif __INTEL_COMPILER
   // compile with -restrict command line option (linux) or -Qrestrict (windows).
   #define AIC_RP restrict
#else
   #define AIC_RP
#endif

#define RESTRICT AIC_RP


#ifdef assert
    #undef assert
#endif
#ifdef assert_rt
   #undef assert_rt
#endif
void AicAssertFail( char const *pExpr, char const *pFile, int iLine );
#define assert_rt(x) if(x) {} else AicAssertFail(#x,__FILE__,__LINE__)

#ifdef _DEBUG
    #define assert(x) assert_rt(x)
#else
    #define assert(x) ((void) 0)
#endif



#endif // CX_DEFS_H
