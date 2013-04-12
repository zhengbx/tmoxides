/* This program is free software. It comes without any warranty, to the extent
 * permitted by applicable law. You may use it, redistribute it and/or modify
 * it, in whole or in part, provided that you do so at your own risk and do not
 * hold the developers or copyright holders liable for any claim, damages, or
 * other liabilities arising in connection with the software.
 * 
 * Developed by Gerald Knizia, 2010--2012.
 */

#include <string>
#include <iostream>
#include <algorithm>
#include <stdlib.h> // for malloc
#include <string.h> // for memset
#include <stdexcept>
#include <cmath>
using std::size_t;
using std::ptrdiff_t;
using std::cout;
using std::endl;

#include <boost/format.hpp>
using boost::format;

#include "CxOpenMpProxy.h"
#include "CxMemoryStack.h"
#include "CxPodArray.h"
using namespace ct;

extern bool g_BosonicSigns;

#include "CxTypes.h"
#include "CxAlgebra.h"
#include "CxStorageDevice.h"

#include "FciDump.h"
#include "FciVec.h"
#include "FciSubspace.h"
#include "FciOps.h"
#include "FciCsf.h"
