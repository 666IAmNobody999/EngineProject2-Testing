// Shared by universal/entry_*.cpp (ported from 幻神).
// universal/build.sh renames each slice's global main() to ui_main_<slice>.
// Each slice's static constructors were moved by universal/build.sh into a
// private section; they are run here only for the slice that is selected.
#pragma once
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
// PE/COFF: GNU ld sorts grouped sections ".ui_<x>$a" < "$m" (slice) < "$z".
// GCC/mingw .ctors tables are executed from the end towards the start.
    #define DEFINE_BUILD(x) \
        extern "C" int ui_main_##x(int argc, char* argv[]); \
        __attribute__((section(".ui_" #x "$a"), used)) static void (*ui_begin_##x[1])(void) = {nullptr}; \
        __attribute__((section(".ui_" #x "$z"), used)) static void (*ui_end_##x[1])(void) = {nullptr}; \
        static int entry_##x(int argc, char* argv[]) { \
            void (**volatile b)(void) = ui_begin_##x + 1; \
            void (**volatile e)(void) = ui_end_##x; \
            for (void (**p)(void) = e; p > b; ) { \
                --p; \
                if (*p && *p != (void (*)(void)) (intptr_t) -1) (*p)(); \
            } \
            return ui_main_##x(argc, argv); \
        }
#else
    #define DEFINE_BUILD(x) \
        extern "C" int ui_main_##x(int argc, char* argv[]); \
        extern "C" void (*__start_##x##_init[])(void) __attribute__((weak)); \
        extern "C" void (*__stop_##x##_init[])(void) __attribute__((weak)); \
        static int entry_##x(int argc, char* argv[]) { \
            for (void (**p)(void) = __start_##x##_init; p && p < __stop_##x##_init; ++p) \
                (*p)(); \
            return ui_main_##x(argc, argv); \
        }
#endif

// PIKAJIEQI_SLICE=<name> forces a slice (testing only, e.g. x86_64_avx2).
static inline bool forced_slice(const char* name) {
    const char* f = getenv("PIKAJIEQI_SLICE");
    return f && !strcmp(f, name);
}
#define TRY_FORCED(x) if (forced_slice(#x)) return entry_##x(argc, argv);
// (a forced slice that was not built maps to its fallback)
