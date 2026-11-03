#!/bin/bash
# build_engine.sh — Build PikaJieQi Enhanced + download NNUE
set -ex

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build engine
cd "$SCRIPT_DIR/src"
make -j$(nproc) ARCH=x86-64-sse41-popcnt build
cp PikaJieQi "$SCRIPT_DIR/pikajieqi-native"
chmod +x "$SCRIPT_DIR/pikajieqi-native"

# Download NNUE from official Pikafish Networks repo
if [ ! -f "$SCRIPT_DIR/pikafish.nnue" ]; then
    echo "Downloading pikafish.nnue..."
    curl -sL "https://github.com/official-pikafish/Networks/releases/download/master-net/pikafish.nnue" \
        -o "$SCRIPT_DIR/pikafish.nnue"
fi

echo "✅ Engine built successfully"
ls -la "$SCRIPT_DIR/pikajieqi-native" "$SCRIPT_DIR/pikafish.nnue"
