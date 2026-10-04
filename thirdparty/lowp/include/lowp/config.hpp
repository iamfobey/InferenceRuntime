#pragma once

#ifndef LOWP_HAS_F16C
    #if defined(__AVX2__) && (defined(_MSC_VER) || defined(__F16C__))
        #define LOWP_HAS_F16C 1
    #else
        #define LOWP_HAS_F16C 0
    #endif
#endif
