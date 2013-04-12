/* This program is free software. It comes without any warranty, to the extent
 * permitted by applicable law. You may use it, redistribute it and/or modify
 * it, in whole or in part, provided that you do so at your own risk and do not
 * hold the developers or copyright holders liable for any claim, damages, or
 * other liabilities arising in connection with the software.
 * 
 * Developed by Gerald Knizia, 2010--2012.
 */

#ifndef _CX_TYPES_H
#define _CX_TYPES_H

#ifndef _for_each
    #define _for_each(it,con) for ( (it) = (con).begin(); (it) != (con).end(); ++(it) )
#endif

#define __STDC_CONSTANT_MACROS
// ^- ask stdint.h to include fixed-size literal macros (e.g., UINT64_C).
#include <boost/cstdint.hpp>

// gna... apparently, some older boost versions don't have the constant
// macros...
// We here guess something which will likely work on many linux compilers
// (which pretent to be g++ and accept its non-standard long long)
#ifndef UINT64_C
  #ifdef __GNUC__ // g++
     #define UINT64_C(x) x##ull
  #else
     #define UINT64_C(x) x##ul
  #endif
#endif

#ifndef UINT32_C
  #ifdef __GNUC__ // g++
     #define UINT32_C(x) x##u
  #else
     #define UINT32_C(x) x##u
  #endif
#endif

using boost::uint64_t;
using boost::uint32_t;
using boost::uint16_t;
using boost::uint8_t;
using boost::int64_t;
using boost::int32_t;
using boost::int16_t;
using boost::int8_t;
using std::size_t;
using std::ptrdiff_t;

typedef unsigned int
    uint;
typedef unsigned char
    uchar;
typedef unsigned int
    uint;

#include "CxDefs.h"
#define RESTRICT AIC_RP


namespace ct {
    struct FIntrusivePtrDest;
}

void intrusive_ptr_add_ref( ct::FIntrusivePtrDest const *pExpr );
void intrusive_ptr_release( ct::FIntrusivePtrDest const *pExpr );


namespace ct {
    /// A base class for reference counted objects. Classes derived from this can
    /// be used as target for boost::intrusive_ptr.
    struct FIntrusivePtrDest
    {
        FIntrusivePtrDest() : m_RefCount(0) {};
        inline virtual ~FIntrusivePtrDest() = 0;

        mutable int m_RefCount;
        friend void ::intrusive_ptr_add_ref( FIntrusivePtrDest const *Expr );
        friend void ::intrusive_ptr_release( FIntrusivePtrDest const *Expr );
    };

    inline FIntrusivePtrDest::~FIntrusivePtrDest()
    {
    };
} // namespace ct

inline void intrusive_ptr_add_ref( ct::FIntrusivePtrDest const *pExpr ) {
    pExpr->m_RefCount += 1;
}

inline void intrusive_ptr_release( ct::FIntrusivePtrDest const *pExpr ) {
    assert( pExpr->m_RefCount > 0 );
    pExpr->m_RefCount -= 1;
    if ( pExpr->m_RefCount == 0 )
        delete pExpr;
}


#endif // _CX_TYPES_H

// kate: space-indent on; tab-indent on; backspace-indent on; tab-width 4; indent-width 4; mixedindent off; indent-mode normal;
