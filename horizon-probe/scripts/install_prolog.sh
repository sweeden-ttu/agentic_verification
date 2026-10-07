#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$ROOT/.tools"
mkdir -p "$TOOLS"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) PLAT=linux-64 ;;
  Linux-aarch64) PLAT=linux-aarch64 ;;
  Darwin-arm64) PLAT=osx-arm64 ;;
  Darwin-x86_64) PLAT=osx-64 ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)"; exit 1 ;;
esac
if [ ! -x "$TOOLS/bin/micromamba" ]; then
  curl -Ls "https://micro.mamba.pm/api/micromamba/$PLAT/latest" | tar -xj -C "$TOOLS" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$TOOLS/mamba"
PKGS="swi-prolog"
if [ "${WITH_LLAMA_CPP:-0}" = "1" ]; then PKGS="$PKGS llama.cpp"; fi
if [ -d "$TOOLS/env" ]; then
  "$TOOLS/bin/micromamba" install -y -q -p "$TOOLS/env" -c conda-forge $PKGS
else
  "$TOOLS/bin/micromamba" create -y -q -p "$TOOLS/env" -c conda-forge $PKGS
fi
"$TOOLS/env/bin/swipl" --version
if [ -x "$TOOLS/env/bin/llama-server" ]; then "$TOOLS/env/bin/llama-server" --version 2>&1 | head -1; fi
