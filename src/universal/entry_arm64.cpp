/*
  幻神揭棋引擎 universal entry for arm64 / Android (ported from the 幻神 engine).
  Selects the armv8.2-a+dotprod slice when the CPU supports it.
*/
#include <stdint.h>
#include "slices.h"

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <sys/auxv.h>
    #ifndef HWCAP_ASIMDDP
        #define HWCAP_ASIMDDP (1 << 20)
    #endif
#endif

DEFINE_BUILD(armv8)
#ifdef HAS_armv8_dotprod
DEFINE_BUILD(armv8_dotprod)
#else
    #define entry_armv8_dotprod entry_armv8
#endif

static bool has_dotprod() {
#if defined(_WIN32)
    return IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE);
#else
    return getauxval(AT_HWCAP) & HWCAP_ASIMDDP;
#endif
}

int main(int argc, char* argv[]) {
    TRY_FORCED(armv8)
    TRY_FORCED(armv8_dotprod)

    return has_dotprod() ? entry_armv8_dotprod(argc, argv) : entry_armv8(argc, argv);
}
