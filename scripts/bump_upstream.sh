#!/usr/bin/env bash
# Pin the umap/nndescent crates to the latest commit on apple/embedding-atlas main
# (or to the commit given as the first argument), then rebuild and run the tests.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=${1:-$(git ls-remote https://github.com/apple/embedding-atlas refs/heads/main | cut -f1)}
echo "Pinning embedding-atlas to $REV"
sed -i.bak -E "s/(embedding-atlas\", rev = \")[0-9a-f]+/\1$REV/g" Cargo.toml && rm Cargo.toml.bak
cargo update -p umap -p nndescent
cargo test --lib
uv run maturin develop --uv --profile dev-release
uv run pytest -n auto
uv run --group parity pytest -m parity
