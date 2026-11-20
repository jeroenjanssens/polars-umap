#!/usr/bin/env bash
# Export the marimo notebook as a self-contained WebAssembly app for the docs site.
# Run before `great-docs build`; the output is generated, not committed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=assets/notebooks/explore-umap
rm -rf "$OUT"
echo n | uv run --no-sync marimo export html-wasm notebooks/explore-umap.py \
  -o "$OUT" --mode run --no-sandbox
echo "exported $OUT"
