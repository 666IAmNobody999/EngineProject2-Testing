#!/bin/bash
# build_engine.sh — Build 幻神揭棋引擎 (PikaJieQi) with PGO.
# The engine uses the classical evaluation (USE_NNUEEVAL is 0), so no network
# file is needed; the old pikafish.nnue download has been removed.
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/src"
make -j$(nproc) ARCH=${ARCH:-x86-64-bmi2} ${COMP:+COMP=$COMP} profile-build
make strip
cp PikaJieQi "$SCRIPT_DIR/pikajieqi-native"
chmod +x "$SCRIPT_DIR/pikajieqi-native"

echo "Engine built successfully"
ls -la "$SCRIPT_DIR/pikajieqi-native"
