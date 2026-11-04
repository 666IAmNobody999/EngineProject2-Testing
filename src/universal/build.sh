#!/usr/bin/env bash
# Universal (runtime-dispatched) build of 幻神揭棋引擎, ported from the 幻神 engine.
#
# Every ISA slice is compiled from the same sources with -DStockfish=Stockfish_<slice>
# (so all engine symbols are distinct), PGO-trained, partially linked into one
# relocatable object, and finally linked together with universal/entry_<cpu>.cpp,
# which picks the best slice for the running CPU at start-up.
#
# Usage (from src/):
#   universal/build.sh x86-64|arm64
# Environment:
#   COMP        gcc | mingw | clang | ndk        (default gcc)
#   PGO         yes | no                         (default yes)
#   RUN_PREFIX  emulator for the PGO bench, e.g. "sde -future --" or "qemu-aarch64 -cpu max"
#   JOBS        parallel make jobs               (default nproc)
#   OUT         output file name                 (default PikaJieQi-universal[.exe])
#   SLICES      override the slice list (space separated ARCH names)
#   LDFLAGS_EXTRA  extra flags for the final link (e.g. -static)
set -euo pipefail

TARGET=${1:?usage: universal/build.sh x86-64|arm64}
COMP=${COMP:-gcc}
PGO=${PGO:-yes}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 2)}
RUN_PREFIX=${RUN_PREFIX:-}
SRC=$(cd "$(dirname "$0")/.." && pwd)
WORK="$SRC/universal/work-$TARGET-$COMP"

case "$TARGET" in
  x86-64)
    SLICES=${SLICES:-"x86-64 x86-64-sse41-popcnt x86-64-avx2 x86-64-bmi2 x86-64-avx512"}
    ENTRY=entry_x86.cpp ;;
  arm64)
    SLICES=${SLICES:-"armv8 armv8-dotprod"}
    ENTRY=entry_arm64.cpp ;;
  *) echo "unknown target $TARGET" >&2; exit 1 ;;
esac

windows=no
if [ "$COMP" = mingw ] || [ "${OS:-}" = Windows_NT ]; then windows=yes; fi
if [ "$COMP" = ndk ]; then windows=no; fi
if [ $windows = yes ]; then EXT=.exe; else EXT=; fi
OUT=${OUT:-PikaJieQi-universal$EXT}

# Tools
case "$COMP" in
  ndk)
    case "$TARGET" in
      arm64)  CXX=aarch64-linux-android21-clang++ ;;
      x86-64) CXX=x86_64-linux-android21-clang++ ;;
    esac
    OBJCOPY=llvm-objcopy; NM=llvm-nm ;;
  clang)
    CXX=clang++; OBJCOPY=llvm-objcopy; NM=llvm-nm ;;
  mingw)
    if [ "${OS:-}" = Windows_NT ]; then CXX=g++; OBJCOPY=objcopy; NM=nm
    else CXX=x86_64-w64-mingw32-g++-posix; command -v $CXX >/dev/null || CXX=x86_64-w64-mingw32-g++
         OBJCOPY=x86_64-w64-mingw32-objcopy; NM=x86_64-w64-mingw32-nm; fi ;;
  *)
    CXX=g++; OBJCOPY=objcopy; NM=nm ;;
esac

rm -rf "$WORK"; mkdir -p "$WORK"
SLICE_OBJS=()
ENTRY_DEFS=()

for arch in $SLICES; do
  sfx=${arch//-/_}                 # x86-64-avx2 -> x86_64_avx2
  echo "=== slice $arch (namespace Stockfish_$sfx) ==="
  d="$WORK/$sfx"; mkdir -p "$d"
  (cd "$SRC" && tar --exclude='./universal/work-*' --exclude='*.o' -cf - .) | (cd "$d" && tar -xf -)
  export CXXFLAGS="-DStockfish=Stockfish_$sfx -DUNIVERSAL_BINARY"
  if [ "$PGO" = yes ]; then
    make -C "$d" -j"$JOBS" profile-build ARCH="$arch" COMP="$COMP" RUN_PREFIX="$RUN_PREFIX" >"$d.log" 2>&1 \
      || { tail -n 60 "$d.log"; exit 1; }
    # The Makefile pipes the PGO bench through tail, so check it really ran.
    grep -E "Nodes searched|Nodes/second" "$d.log" \
      || { echo "PGO bench of slice $arch did not run:" >&2; grep -A3 "Step 2/4" "$d.log" >&2; exit 1; }
  else
    make -C "$d" -j"$JOBS" build ARCH="$arch" COMP="$COMP" >"$d.log" 2>&1 || { tail -n 60 "$d.log"; exit 1; }
  fi
  o="$WORK/slice_$sfx.o"
  if [ $windows = yes ]; then
    # mingw: "ld -r" + LTO is broken for PE COMDATs, so (like 幻神) take the
    # native whole-program object that -save-temps left from the slice's
    # (PGO) link.  LTO already internalized everything except main and the
    # COMDAT std:: helpers, which are resolved to the first (baseline) slice.
    in="$d/PikaJieQi.ltrans0.ltrans.o"
    [ -f "$in" ] || { echo "missing $in" >&2; ls "$d" >&2; exit 1; }
    # '$' grouping sorts .ui_<sfx>$a < $m < $z inside one output section.
    $OBJCOPY --redefine-sym main="ui_main_$sfx" \
             --rename-section .ctors=".ui_$sfx\$m" \
             "$in" "$o"
  else
    # ELF: relocatable LTO link of this slice (PGO counts are inside the IR).
    make -C "$d" slice-object ARCH="$arch" COMP="$COMP" EXTRACXXFLAGS= >>"$d.log" 2>&1 \
      || { tail -n 60 "$d.log"; exit 1; }
    # Make every strong global definition local (Stockfish_<sfx>::*, the TUNE
    # globals, the C functions of miniz/zip) so slices never clash; weak/COMDAT
    # std:: definitions stay global and resolve to the first (baseline) slice.
    $NM --defined-only -g "$d/slice.o" | awk '$2 ~ /^[TDBRCSG]$/ && $3 != "main" {print $3}' > "$d/localize.txt"
    # Static constructors move to a private section run only by entry_*.cpp
    # for the selected slice.
    $OBJCOPY --localize-symbols="$d/localize.txt" --redefine-sym main="ui_main_$sfx" \
             --rename-section .init_array="${sfx}_init,alloc,load,data,contents" \
             "$d/slice.o" "$o"
  fi
  $NM --defined-only -g "$o" | awk '$3=="main"' | grep -q . && { echo "main still global in $o" >&2; exit 1; }
  $NM --defined-only -g "$o" | grep -q " ui_main_$sfx\$" || { echo "ui_main_$sfx missing in $o" >&2; exit 1; }
  SLICE_OBJS+=("$o")
  ENTRY_DEFS+=("-DHAS_$sfx")
done

echo "=== entry + final link ($OUT) ==="
case "$COMP" in ndk) EF="-stdlib=libc++ -fPIE";; *) EF="";; esac
FINAL_LD="-lm"
case "$COMP" in
  ndk)   FINAL_LD="-static-libstdc++ -static -lm" ;;
  mingw) FINAL_LD="-static -Wl,--allow-multiple-definition" ;;
  clang) FINAL_LD="-lpthread -latomic -lm" ;;
  gcc)   FINAL_LD="-lpthread -lm" ;;
esac
$CXX -O2 -std=c++17 $EF ${ENTRY_DEFS[@]} -c "$SRC/universal/$ENTRY" -o "$WORK/entry.o"
# Baseline slice first: identical weak/COMDAT (std::) definitions resolve to it.
$CXX -o "$SRC/$OUT" "$WORK/entry.o" "${SLICE_OBJS[@]}" $FINAL_LD ${LDFLAGS_EXTRA:-}
ls -l "$SRC/$OUT"
