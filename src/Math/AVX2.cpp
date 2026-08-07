#if defined(__AVX2__) || (defined(_M_AMD64) && defined(___AVX2__)) || (defined(_MSC_VER) && defined(__AVX2__))
#define HAVE_SUPPORT_AVX2 1
#else
#define HAVE_SUPPORT_AVX2 0
#endif
