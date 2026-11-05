/* 幻神揭棋引擎: embeds the default network (pikafish.nnue) into the binary.
   Assembled only when ABJ_EMBED_NET is set; ABJ_NET_PATH is the absolute path of the net file. */
#ifndef ABJ_NET_PATH
#error "ABJ_NET_PATH not defined"
#endif
#if defined(__APPLE__)
#define SYM(x) _##x
#else
#define SYM(x) x
#endif
#if defined(_WIN32)
    .section .rdata,"dr"
#elif defined(__APPLE__)
    .section __TEXT,__const
#else
    .section .rodata
#endif
    .balign 64
    .globl SYM(hs_embedded_net)
    .globl SYM(hs_embedded_net_end)
SYM(hs_embedded_net):
    .incbin ABJ_NET_PATH
SYM(hs_embedded_net_end):
    .byte 0
#if defined(__linux__) || defined(__ANDROID__)
    .section .note.GNU-stack,"",%progbits
#endif
