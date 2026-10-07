#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
SIZES="${*:-E2B E4B}"
for s in $SIZES; do
  python3 - "$s" "$ROOT/models" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
size, out = sys.argv[1], sys.argv[2]
repo = f"google/gemma-4-{size}-it-qat-q4_0-gguf"
name = f"gemma-4-{size}_q4_0-it.gguf"
print(hf_hub_download(repo, name, local_dir=out))
EOF
done
