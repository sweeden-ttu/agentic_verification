#!/usr/bin/env bash
# Downloads the Kaggle dataset scottweeden/gemma4-evidently-examples into this folder.
# Needs Kaggle API credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY.
# On Kaggle itself, the dataset is already mounted at
# /kaggle/input/datasets/scottweeden/gemma4-evidently-examples; this copies it instead.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="/kaggle/input/datasets/scottweeden/gemma4-evidently-examples"
if [[ -d "$src" ]]; then
  cp -R "$src"/. "$here/data/" 2>/dev/null || { mkdir -p "$here/data"; cp -R "$src"/. "$here/data/"; }
else
  command -v kaggle >/dev/null || pip install -q kaggle
  mkdir -p "$here/data"
  kaggle datasets download -d scottweeden/gemma4-evidently-examples -p "$here/data" --unzip
fi
find "$here/data" -type f | sed "s#^$here/##" | sort
