# Contributing to polars-umap

Thanks for your interest! Bug reports, ideas and pull requests are welcome on
[GitHub](https://github.com/jeroenjanssens/polars-umap).

## Setup

You need [uv](https://docs.astral.sh/uv/) and a stable [Rust toolchain](https://rustup.rs).

```bash
git clone https://github.com/jeroenjanssens/polars-umap
cd polars-umap
uv sync                                             # creates .venv and builds the extension
uv run maturin develop --uv --profile dev-release  # faster rebuilds while developing
```

`uv sync` builds in release mode (fat LTO), which takes a few minutes. While you iterate on the
Rust code, `maturin develop --profile dev-release` is much faster.

## Tests and linting

```bash
uv run pytest -n auto                       # Python tests
uv run --group parity pytest -m parity      # compare with umap-learn
uv run pytest -m "slow or gpu"              # larger and GPU tests
uv run pytest --doctest-modules python/polars_umap
cargo test --lib                            # Rust tests

uv run ruff check . && uv run ruff format --check . && uv run mypy
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

CI runs all of these, and also tests the oldest supported Python and Polars versions.

## Benchmarks

```bash
uv run --group bench python benchmarks/run.py --quick   # smoke run
uv run --group bench python benchmarks/report.py        # regenerate the benchmarks page
```

See `user_guide/benchmarks.qmd` for the full methodology and the command that produced the
published results.

## Documentation

The site is built with [Great Docs](https://posit-dev.github.io/great-docs/) and
[Quarto](https://quarto.org), which you need to install separately.

```bash
bash scripts/export_notebook.sh            # export the marimo notebook to WebAssembly
QUARTO_PYTHON=.venv/bin/python uv run --group docs great-docs build
uv run --group docs great-docs preview
```

- The configuration is in `great-docs.yml`.
- User guide pages live in `user_guide/`, examples in `gallery/`, and the marimo notebook in
  `notebooks/`.
- The API reference is generated from the docstrings, which use the numpy style.
- Pages with Python code are executed during the build. Their outputs are cached in `_freeze/`,
  which is committed, so CI doesn't need to build the Rust extension. After changing a page,
  rebuild locally and commit the updated `_freeze/`.
- The marimo notebook can't run polars-umap in the browser. When you change the parameter grid,
  regenerate its data with `uv run python scripts/precompute_notebook_data.py`.

## Updating the upstream UMAP implementation

The `umap` and `nndescent` crates come from
[apple/embedding-atlas](https://github.com/apple/embedding-atlas), pinned to a commit in
`Cargo.toml`. To move to the latest commit and run all checks:

```bash
scripts/bump_upstream.sh
```

## Releasing

1. Update the version in `Cargo.toml` and add an entry to `CHANGELOG.md`.
2. Commit, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The release workflow builds the wheels, publishes them to PyPI through trusted publishing,
   checks that the published package installs, and creates a GitHub release.
