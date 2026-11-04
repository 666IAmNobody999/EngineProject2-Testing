/*
  幻神揭棋引擎 universal entry for x86-64 (ported from the 幻神 engine).
  Selects the best ISA slice for the running CPU.
*/
#include <cpuid.h>
#include <stdint.h>
#include "slices.h"

// universal/build.sh passes -DHAS_<slice> for every slice it built; a missing
// slice falls back to the next lower one.
DEFINE_BUILD(x86_64)
#ifdef HAS_x86_64_sse41_popcnt
DEFINE_BUILD(x86_64_sse41_popcnt)
#else
    #define entry_x86_64_sse41_popcnt entry_x86_64
#endif
#ifdef HAS_x86_64_avx2
DEFINE_BUILD(x86_64_avx2)
#else
    #define entry_x86_64_avx2 entry_x86_64_sse41_popcnt
#endif
#ifdef HAS_x86_64_bmi2
DEFINE_BUILD(x86_64_bmi2)
#else
    #define entry_x86_64_bmi2 entry_x86_64_avx2
#endif
#ifdef HAS_x86_64_avx512
DEFINE_BUILD(x86_64_avx512)
#else
    #define entry_x86_64_avx512 entry_x86_64_bmi2
#endif

// AMD Excavator (family 15h) and Zen/Zen+/Zen2 (family 17h) implement pdep/pext
// in microcode, so the avx2 slice (magic bitboards) is faster there.
static bool has_slow_bmi2() {
    return __builtin_cpu_is("amd")
        && (__builtin_cpu_is("bdver4") || __builtin_cpu_is("znver1") || __builtin_cpu_is("znver2"));
}

int main(int argc, char* argv[]) {
    __builtin_cpu_init();

    TRY_FORCED(x86_64)
    TRY_FORCED(x86_64_sse41_popcnt)
    TRY_FORCED(x86_64_avx2)
    TRY_FORCED(x86_64_bmi2)
    TRY_FORCED(x86_64_avx512)

    if (!__builtin_cpu_supports("sse4.1") || !__builtin_cpu_supports("popcnt"))
        return entry_x86_64(argc, argv);

    if (!__builtin_cpu_supports("avx2"))
        return entry_x86_64_sse41_popcnt(argc, argv);

    if (!__builtin_cpu_supports("bmi2") || has_slow_bmi2())
        return entry_x86_64_avx2(argc, argv);

    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw"))
        return entry_x86_64_bmi2(argc, argv);

    return entry_x86_64_avx512(argc, argv);
}
