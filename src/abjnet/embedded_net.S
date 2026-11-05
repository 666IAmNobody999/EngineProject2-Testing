/*
  Embedded AB-JChess HNN net (pikafish.nnue = abjchess-20261030.nnue renamed).

  Assembled only when NNUE_EMBEDDING is defined, i.e. with the Makefile option
  embed=yes (the default when src/pikafish.nnue exists at build time). The
  .incbin path is relative to the make working directory (src/, or the slice
  directory of a universal build, which is a copy of src/).
  Adapted from 幻神 (EngineProject-Testing) nnue/embedded_nets.S.
*/

#if defined(NNUE_EMBEDDING) && defined(__ELF__)

    .section .rodata.pikafish_abjnet, "a", %progbits

    .balign 64
    .globl  pikafish_embedded_abjnet
    .globl  pikafish_embedded_abjnet_end
    .hidden pikafish_embedded_abjnet
    .hidden pikafish_embedded_abjnet_end
pikafish_embedded_abjnet:
    .incbin "pikafish.nnue"
pikafish_embedded_abjnet_end:

#elif defined(NNUE_EMBEDDING) && defined(_WIN32)

// Windows (PE/COFF): read-only data section; COFF has no .hidden. x86-64 and
// ARM64 Windows symbols carry no leading underscore, 32-bit x86 does.
    #if defined(__i386__)
        #define NET_SYM(x) _##x
    #else
        #define NET_SYM(x) x
    #endif

    .section .rdata$pikafish_abjnet, "dr"

    .balign 64
    .globl  NET_SYM(pikafish_embedded_abjnet)
    .globl  NET_SYM(pikafish_embedded_abjnet_end)
NET_SYM(pikafish_embedded_abjnet):
    .incbin "pikafish.nnue"
NET_SYM(pikafish_embedded_abjnet_end):

#endif

#if defined(__ELF__)
    .section .note.GNU-stack, "", %progbits
#endif
