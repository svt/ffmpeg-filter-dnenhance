#!/usr/bin/env bash
# Build libdf.so / libdf.dylib (DeepFilterNet inference library) from upstream.
#
# Usage:
#   ./scripts/build-libdf.sh [<path-to-deepfilternet-checkout>]
#
# If no checkout path is given, clones DeepFilterNet to ./build/DeepFilterNet
# and builds there. The resulting shared library is copied into ./build/lib/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
LIB_OUT="$BUILD_DIR/lib"

DFN_REPO="${1:-$BUILD_DIR/DeepFilterNet}"

if [[ ! -d "$DFN_REPO" ]]; then
  mkdir -p "$BUILD_DIR"
  git clone --depth 1 https://github.com/Rikorose/DeepFilterNet.git "$DFN_REPO"
fi

cd "$DFN_REPO"
cargo build --lib --release -p deep_filter --features "capi,default-model,tract"

mkdir -p "$LIB_OUT"
case "$(uname -s)" in
  Darwin) cp target/release/libdf.dylib "$LIB_OUT/" ;;
  Linux)  cp target/release/libdf.so    "$LIB_OUT/" ;;
  *)      echo "Unsupported OS: $(uname -s)"; exit 1 ;;
esac

echo
echo "libdf built and copied to $LIB_OUT"
echo "At runtime, point FFmpeg at it with:"
echo "  LD_LIBRARY_PATH=$LIB_OUT ffmpeg ..."
echo "  (or DYLD_LIBRARY_PATH on macOS)"
