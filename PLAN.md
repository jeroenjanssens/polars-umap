# polars-umap: Research & Implementation Plan

Status: **approved 2026-11-20, implementation in progress**. See §11 for decisions and §12 for the TODO list.

---

## 1. Research: does this already exist?

**Short answer: no.** There is no Polars plugin that exposes UMAP as an expression or namespace.

| Where | Query | Result |
|---|---|---|
| PyPI | `polars-umap`, `polars_umap` | Name is free; no package |
| GitHub | "polars umap", "polars-umap", "polars plugin dimensionality reduction" | Only analysis notebooks that *use* Polars and umap-learn side by side; no plugin |
| crates.io | `umap` | Taken by an unrelated, abandoned 0.1.0 crate (2022, no repo). **Not** Apple's crate |
| polars-ds (`abstractqqq/polars_ds_extension`, ~660★, v0.14) | grep for umap/tsne | No UMAP/t-SNE. Has `pca`, `principal_components`, `singular_values`, kNN queries (`query_knn_ptwise`, …) |
| polars-ml (PyPI 0.2.0) | — | No summary or repo; no evidence of UMAP |

### Related Polars extensions (prior art and design references)

- **polars-ds**: the closest relative. A general data-science plugin with linear algebra, PCA, kNN, metrics and stats. It's a good reference for how to expose a "whole-column, multi-input" algorithm as an expression (`pl_plugin(..., changes_length=..., pass_name_to_apply=True)`). Built on `pyo3-polars 0.28` / `polars 0.55`.
- **polars-distance**: string and array distance metrics as expressions.
- **polars-xdt, polars-hash, polars-reverse-geocode, polars-business, polars-st**: the canonical small, focused plugins that follow the cookiecutter-polars-plugins layout (Marco Gorelli's template). These are our reference for project structure and best practices.
- **Related non-Polars Rust UMAPs on crates.io**: `umap-rs` (0.4.5), `fast-umap` (1.6.0, burn/GPU), `annembed` (0.1.6). We are told to use Apple's, and it is the best-ported option anyway: it ports umap-learn 0.5 and 0.6 faithfully and includes a PyNNDescent port.

### The upstream implementation (apple/embedding-atlas, `main` @ `a0d0f09`, 2026-11-20)

`packages/umap/` contains:
- `umap` crate: a `UmapBuilder` API, `Umap::builder(&Array2<f32>)`, `Umap::builder_shared(Arc<..>)`, and `Umap::builder_from_knn(indices, distances)`. It returns `UmapResult { embedding, knn_indices, knn_distances }`.
- `nndescent` crate: a PyNNDescent port with SIMD distances. Metrics: euclidean/l2, manhattan/l1, chebyshev, minkowski, cosine, correlation, hamming.
- Parameters: n_neighbors, n_components, min_dist, spread, metric, n_epochs, learning_rate, negative_sample_rate, repulsion_strength, local_connectivity, set_op_mix_ratio, random_state, init (`auto|spectral|tswspectral|random|recursive|pca|Custom(array)`), optimizer (`sgd|momentum|adam`), `compatibility_layout` (true = umap-learn 0.5 pipeline, false = 0.6), a, b, disconnection_distance, progress callback, and a `gpu` feature (wgpu).
- Deps: `ndarray 0.17`, `rayon 1.11`. MIT licensed. Edition 2021.

**Facts that shape the design:**
1. **No `transform()` for new points.** Upstream only does fit/fit_transform. That suits Polars expressions well, because they're stateless. We won't promise sklearn-style `fit` then `transform`.
2. **Not on crates.io**, and the `umap` name there is taken. We depend on it via **git, pinned to a commit `rev`**. That's fine for PyPI wheels, but it means we can't publish our own Rust crate to crates.io. Our sdist build also needs network access, or a vendored copy (see §7).
3. **Default SGD is Hogwild (racy) parallel SGD.** Results with `random_state` are not bit-reproducible across runs when multithreaded. The Adam optimizer uses "deterministic hashed negative sampling". Tests must take this into account (§5).
4. Upstream ships `umap-bench` binaries and a Python benchmark harness. Its quality metrics are trustworthiness and kNN recall, which we can reuse conceptually (§6).

---

## 2. Does a UMAP-specific extension make sense?

**Yes, as a focused plugin, with a few honest caveats.**

Arguments for:
- Embeddings now live in DataFrames. Polars has a first-class `Array(Float32, d)` dtype, and an LLM or sentence-transformer embedding column that you want to plot in 2D is one of the most common UMAP workflows.
- It fits the expression model surprisingly well. UMAP is a whole-column (non-elementwise) operation that keeps the row count. That makes `.over("group")` (UMAP per group, in parallel, in one query), lazy execution and `with_columns` work for free. Doing this with umap-learn means a pandas or numpy round trip and a Python loop per group.
- No numba. umap-learn pulls in numba and llvmlite, with slow import and JIT warm-up (often 5–15 s on first call) and frequent lag in Python-version support. A Rust wheel gives you instant start-up and a small dependency footprint.
- Zero-copy-ish input. The `Array(Float32)` buffer maps directly onto `ndarray::Array2<f32>`.

Arguments against, and how we handle them:
- *UMAP is model fitting, not a column transform.* There's no transform for unseen data upstream anyway. We document it as "fit_transform as an expression".
- *A general home (polars-ds) might be better.* polars-ds has no UMAP, and adding a git-pinned dependency with wgpu/ndarray 0.17 to it is a heavy ask. A focused plugin is easier to keep in sync with upstream. If it proves itself, upstreaming to polars-ds remains an option.
- *Scope creep.* We expose **UMAP plus NNDescent kNN**. kNN comes free since it's already a dependency and `UmapResult` returns it. Density clustering, which embedding-atlas also ships, is out of scope.

---

## 3. Proposed interface

Importing `polars_umap` registers a **`umap` namespace** on `Expr`, `DataFrame` and `LazyFrame`. Functional entry points are exported too, for typing-friendly use.

### 3.1 Expression level (core primitive)

```python
import polars as pl
import polars_umap as pu  # registers the `umap` namespace

# a) Single embedding column: Array(Float32|Float64, d) or List(numeric) with constant length
df.with_columns(
    xy=pl.col("embedding").umap.fit_transform(
        n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
)
# -> xy: Array(Float32, 2)

# b) Several scalar feature columns (like sklearn on a feature matrix)
df.with_columns(xy=pu.umap("sepal_length", "sepal_width", "petal_length", "petal_width"))

# c) Per group: one UMAP per group, in parallel, inside one query
df.with_columns(xy=pl.col("embedding").umap.fit_transform(random_state=0).over("cluster"))

# d) Unpack to scalar columns
df.with_columns(
    pl.col("embedding").umap.fit_transform().arr.to_struct(lambda i: f"umap_{i}")
).unnest(...)
```

- Output dtype: **`Array(Float32, n_components)`**. That's Polars-native, compact and matches the input dtype convention. A helper gives scalar columns (see 3.2).
- Inputs: Array or List of Float32 or Float64 (or ints, cast to f32), or ≥1 numeric columns stacked horizontally. f64 gets cast to f32, since upstream is f32-only.
- **Null policy** (`null_policy="error" | "ignore"`). The default `"error"` raises on null rows or NaN. `"ignore"` fits on the valid rows only and gives null output for invalid rows, so **row count is always preserved**.
- Errors raise `polars.exceptions.ComputeError` with clear messages: n_rows ≤ n_neighbors, ragged lists, unknown metric, n_components < 1, etc. Python validates params early where possible.

### 3.2 DataFrame / LazyFrame namespace (convenience)

```python
df.umap.fit_transform(
    "embedding",  # or a list of numeric columns / selectors
    n_components=2,
    output="columns",  # "columns" -> umap_0, umap_1 ; "array" -> single Array column
    prefix="umap_",
    **params,
)
lf.umap.fit_transform(...)  # same, lazy
```

### 3.3 kNN expressions (by-product of NNDescent)

```python
pl.col("embedding").umap.knn(k=15, metric="cosine")
# -> Struct{indices: Array(UInt32, k), distances: Array(Float32, k)}

# UMAP from a precomputed kNN graph (e.g. from polars-ds, FAISS, or a vector DB)
pu.umap_from_knn("knn_indices", "knn_distances", n_components=2)
```

### 3.4 Parameters (same names and defaults as umap-learn, so users can switch easily)

`n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0, metric="euclidean", n_epochs=None, learning_rate=1.0, init="auto", negative_sample_rate=5, repulsion_strength=1.0, local_connectivity=1.0, set_op_mix_ratio=1.0, a=None, b=None, random_state=None, optimizer=None, layout="0.5" | "0.6"` (mapped onto `compatibility_layout`), `disconnection_distance=None`, `verbose=False`, plus `n_threads=None` (§4.3).

`metric` is typed as `Literal[...]`. `init` accepts a string or a numpy array / column name for a custom init (phase 2).

### 3.5 Not in scope (documented)

- `transform` / `inverse_transform` of new data, because upstream doesn't support it.
- Supervised or semi-supervised UMAP (`y=`), densMAP, and sparse input, also because upstream doesn't support them.

---

## 4. Architecture

### 4.1 Layout (maturin mixed project, based on cookiecutter-polars-plugins)

```
polars-umap/
├── Cargo.toml              # cdylib "_internal"; deps: pyo3 (abi3-py310), pyo3-polars (derive), polars,
│                           #   serde, ndarray 0.17, rayon, umap & nndescent = { git, rev = "a0d0f09…" }
├── pyproject.toml          # build-backend = maturin; requires polars>=<min>; dependency-groups (dev, bench, docs)
├── rust-toolchain.toml     # pin stable
├── src/
│   ├── lib.rs              # pymodule, jemalloc/mimalloc global allocator
│   ├── expressions.rs      # #[polars_expr] umap_fit_transform, nndescent_knn, umap_from_knn
│   ├── convert.rs          # Series/Array/List/multi-col -> Array2<f32>; Array2 -> ArrayChunked; null masks
│   └── params.rs           # serde kwargs structs, validation, mapping to UmapBuilder
├── python/polars_umap/
│   ├── __init__.py         # umap(), umap_from_knn(), knn(); namespace registration
│   ├── _namespace.py       # @pl.api.register_{expr,dataframe,lazyframe}_namespace("umap")
│   ├── _typing.py          # Literal types for metric/init/optimizer
│   └── py.typed
├── tests/                  # pytest (+ cargo test for Rust units)
├── examples/               # scripts + marimo/Jupyter notebooks
├── benchmarks/             # see §6
├── docs/                   # mkdocs-material + mkdocstrings
└── .github/workflows/      # CI, wheels (maturin-action), release (trusted publishing)
```

### 4.2 Rust plugin functions

- `#[polars_expr(output_type_func_with_kwargs = umap_output_type)]`, because the output dtype `Array(Float32, n_components)` depends on the kwargs.
- Python side: `register_plugin_function(..., is_elementwise=False, changes_length=False, returns_scalar=False)`. Being non-elementwise makes Polars give us the whole column, or the whole group under `.over()`.
- Conversion: rechunk, then take the inner values buffer from `ArrayChunked` (contiguous f32), then `Array2::from_shape_vec((n, d), …)`, with no per-row work. Use `builder_shared(Arc<…>)` so NNDescent doesn't copy again. Lists get validated for constant width first.
- Output: `Array2<f32>` (row-major), then a flat `Float32Chunked`, then reshaped to `ArrayChunked`. Nulls get scattered back via a validity mask.
- Errors: `UmapError` maps to `PolarsError::ComputeError`.

### 4.3 Threading

The plugin `.so` statically links its own rayon, so its global pool is separate from Polars' pool. Under `.over()` with many groups, Polars runs groups in parallel and each group then spins up upstream rayon work, which risks oversubscription. Mitigation: use a lazily created plugin-local `rayon::ThreadPool`, sized by `n_threads` (default: `POLARS_MAX_THREADS` or the number of logical CPUs), and run `install()` on it. We'll benchmark the `.over()` case explicitly.

### 4.4 Version compatibility

Target `pyo3-polars 0.28` / `polars 0.55` (Rust), which works with the current Python Polars 2.x line. We'll decide the minimum supported Python Polars version empirically during phase 1 and test it in CI against both the minimum and the latest. abi3 wheels: py310+.

---

## 5. Testing strategy ("plenty of tests")

**Python (pytest, the main suite):**
- *Shape and dtype:* output length equals input length; dtype is `Array(Float32, k)` for k ∈ {1, 2, 3, 10}; works for Array, List, multiple columns, Float32/Float64/Int inputs.
- *Eager vs lazy:* `df.with_columns(...)` matches `lf.with_columns(...).collect()`. `df.umap` and `lf.umap` namespaces work. `select` and `with_columns` both work.
- *Groups:* `.over("g")` gives an independent UMAP per group. The result for group g equals UMAP on the filtered group, using the deterministic config.
- *Reproducibility:* with `random_state` plus the deterministic optimizer (Adam/0.6 layout) or `n_threads=1`, two runs give identical results. With Hogwild SGD we only assert "close in quality", per upstream semantics.
- *Quality (statistical, not exact):* on sklearn `make_blobs`, `load_digits` and Iris:
  - trustworthiness ≥ 0.9 (vs sklearn `trustworthiness`)
  - kNN-classifier accuracy in 2D ≥ a threshold
  - trustworthiness within ε of umap-learn on the same data (optional `parity` marker, needs `umap-learn` in a test group)
- *Metrics:* each metric runs, and cosine on L2-normalized data ≈ euclidean structure.
- *Nulls and NaN:* `error` raises; `ignore` keeps row positions and gives null output for null input.
- *Errors:* too few rows, n_neighbors ≥ n_rows, ragged lists, string column, unknown metric/init, n_components ≤ 0, empty frame.
- *kNN:* recall@k ≥ 0.95 vs exact brute-force kNN (sklearn `NearestNeighbors`); `umap_from_knn` with exact kNN gives a valid embedding.
- *Property-based (Hypothesis):* random shapes and dtypes never crash; output stays finite and preserves length.
- *Edge cases:* duplicate rows, constant column, n == n_neighbors + 1, high dimensions (d = 1024), single group in `.over()`, chunked input (`pl.concat(..., rechunk=False)`).
- *Typing:* `pyright` or `mypy` on the examples and public API.

**Rust (`cargo test`):** conversion round-trips (Array/List/multi-col, with and without nulls), kwargs deserialization, and the output-dtype function.

**Hygiene:** `ruff` (lint and format), `cargo fmt`, `cargo clippy -D warnings`, and pre-commit hooks. Tests run with `uv run pytest`. Slow and parity tests get markers so the default run stays under about 30 s.

---

## 6. Benchmark design

Goal: answer three questions.
1. How fast is polars-umap end to end compared to umap-learn?
2. How much overhead does the Polars layer add on top of the raw Rust crate?
3. Is the embedding quality equivalent?

**Contenders:**
- `polars-umap`: eager, lazy, and `.over()`
- `umap-learn 0.6`, run with both the 0.5 and 0.6 layouts, with numba JIT warm-up measured separately and excluded
- Upstream `umap-bench` native binary, which gives the overhead baseline
- Optional: cuML UMAP (GPU) and openTSNE, for context only

**Datasets:**
- Synthetic Gaussian blobs over a grid: n ∈ {1k, 10k, 100k, 1M}, d ∈ {16, 128, 768}
- MNIST 70k (784-d, euclidean) and Fashion-MNIST
- A real text-embedding set: e.g. 100k sentence-transformer vectors (384-d, cosine), cached as Parquet
- Group-by scenario: 1M rows in 100 or 1,000 groups. polars-umap `.over()` vs a Python loop over umap-learn

**Measurements:**
- Wall time: median of 5 runs after warm-up
- Per-stage time: kNN, graph, init, optimize, via the upstream progress callback, exposed through `verbose`
- Peak RSS: each run in a fresh subprocess, via `resource.getrusage` / `/usr/bin/time -l`
- Import / first-call latency: cold start, where numba hurts most
- Quality: trustworthiness and kNN recall on a fixed 10k subsample, the same metrics upstream uses

**Controls:** fixed `random_state`; the same params for every contender; thread counts pinned (`POLARS_MAX_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS`); machine specs and git revs recorded in the output JSON.

**Tooling:**
- `benchmarks/run.py`: a `uv run` CLI that writes a JSON-lines result file
- `benchmarks/report.py`: Markdown tables and plots (time vs n, quality vs time), committed to `docs/benchmarks.md`
- `pytest-benchmark` (or CodSpeed in CI) micro-benchmarks for conversion overhead and small-n end-to-end, to catch regressions on PRs

---

## 7. Packaging, CI, release

- **Build:** `maturin` via `uv` (`uv run maturin develop --uv`). abi3 wheels for linux x86_64/aarch64 (manylinux + musllinux), macOS arm64/x86_64 and Windows x64, built with `PyO3/maturin-action`.
- **Upstream pin:** `umap = { git = "https://github.com/apple/embedding-atlas", rev = "a0d0f09…" }`. Renovate or Dependabot opens PRs on upstream updates, and CI runs the parity tests on each bump.
  - sdist: either accept that it needs network access (git fetch), or run `cargo vendor` into the sdist. Proposal: vendor only the `umap` and `nndescent` crates under `vendor/` with attribution (MIT allows this). Decide in phase 5.
- **GPU (wgpu) feature:** off by default because of wheel size and driver issues. Possibly a separate `polars-umap-gpu` wheel later.
- **Release:** tag, then the GH Actions wheel matrix, then PyPI trusted publishing. Changelog via release-please or git-cliff.
- **Docs:** README quick start, mkdocs site with API reference, examples gallery and benchmark page.
- **License:** MIT, matching upstream, with upstream attribution in NOTICE.

---

## 8. Examples

1. `examples/quickstart.py`: Iris or digits, a `df.umap.fit_transform` one-liner, plotted with Altair or hvplot.
2. `examples/text_embeddings.py`: embed sentences, UMAP with cosine, scatter coloured by label.
3. `examples/per_group.py`: `.over("category")`, one embedding per category in one query.
4. `examples/lazy_parquet.py`: `scan_parquet` → `umap` → `sink_parquet`.
5. `examples/precomputed_knn.py`: kNN from polars-ds or exact, then `umap_from_knn`.
6. `examples/notebook.py` (marimo): interactive exploration of `n_neighbors` and `min_dist`.

---

## 9. Phased delivery

| Phase | Deliverable |
|---|---|
| 0 | Scaffold: maturin project, git dep compiles, empty plugin round-trip test, CI skeleton |
| 1 | `fit_transform` expression (Array/List/multi-col input, all params), Python namespaces, error mapping |
| 2 | Full test suite (§5), null policy, thread pool |
| 3 | `knn`, `umap_from_knn`, custom init |
| 4 | Benchmarks and report (§6) |
| 5 | Examples, docs, wheels and release pipeline, sdist/vendoring decision |
| later | GPU feature, upstreaming conversation with polars-ds |

---

## 10. Open questions for Jeroen

1. **Package name and namespace:** `polars-umap` with namespace `umap`? (The PyPI name is free.)
2. **Default output shape:** a single `Array(Float32, k)` column for expressions, with separate `umap_0…` columns only in the DataFrame helper? Or separate columns everywhere?
3. **Scope:** include NNDescent `knn` and `umap_from_knn` in v1, or UMAP only?
4. **Default layout:** follow upstream's default (umap-learn 0.5 SGD, racy) or default to the 0.6 pipeline (Adam, deterministic)?
5. **Support floor:** minimum Python (3.10?) and minimum Polars version.
6. **GPU:** worth pursuing at all?

---

## 11. Decisions (Jeroen, 2026-11-20)

1. Package `polars-umap`, import `polars_umap`, namespace `umap`. ✅
2. Expressions return `Array(Float32, k)`. The DataFrame helper can split this into `umap_0…` columns. ✅
3. v1 includes `knn` and `umap_from_knn`. ✅
4. **Default is the umap-learn 0.6 pipeline** (recursive init + Adam, deterministic). `layout="0.5"` is opt-in.
5. Support older Python and Polars versions within reason. The floor is set by what loads empirically: Python ≥ 3.10, Polars ≥ the oldest 1.x that accepts the plugin ABI.
6. GPU: yes. Cargo feature `gpu` (wgpu), exposed as `gpu=True`, with a silent fallback to CPU when no adapter is available.

## 12. TODO

Status as of 2026-11-20. `[x]` means done and verified, `[~]` means partially done (see notes).

### Phase 0: scaffold
- [x] git init, .gitignore, LICENSE (MIT), NOTICE (upstream attribution)
- [x] Cargo.toml (cdylib, pyo3 abi3-py39, pyo3-polars 0.28, polars 0.55, umap/nndescent git rev `a0d0f09`) and rust-toolchain.toml
- [x] pyproject.toml (maturin, uv dependency groups, ruff, mypy, pytest config)
- [x] Plugin compiles; `uv run maturin develop --uv --profile dev-release` works
  - Note: `PolarsAllocator` segfaults on import with Python Polars 2.1, so the plugin uses the system allocator. This is safe because exported arrays carry their own release callbacks.

### Phase 1: core
- [x] `convert.rs`: Array/List/multi-col → `Array2<f32>`, validity mask, `Array2` → ArrayChunked
- [x] Parameter kwargs (serde), validation, mapping to `UmapBuilder` (in `expressions.rs`)
- [x] `umap_fit_transform` expression (output type from kwargs)
- [x] `knn` expression → Struct{indices: Array(UInt32,k), distances: Array(Float32,k)}
- [x] `umap_from_knn` expression
- [x] Plugin-local rayon pools (`n_threads`); the global pool honours `POLARS_MAX_THREADS`
- [x] Python: `umap()`, `knn()`, `umap_from_knn()`, Expr/DataFrame/LazyFrame `umap` namespaces, Literal types, docstrings, py.typed, `_internal.pyi`
- [x] Null policy (`error` / `ignore`)
- [x] Custom init (numpy array, Series or expression)
- [x] Tiny inputs (0–2 rows) get a trivial embedding; `n_neighbors` is clamped to n−1, as in umap-learn

### Phase 2: tests (171 Python tests incl. parity/slow/gpu + 3 doctests + 6 Rust tests, ~15 s)
- [x] Shape, dtype and input variants (Array/List/columns/selectors/regex, int/float dtypes, chunked and sliced input)
- [x] Eager vs lazy, namespaces
- [x] `.over()` and `group_by().agg()`
- [x] Reproducibility: the 0.6 pipeline is bit-identical across runs and thread counts
- [x] Quality: trustworthiness, kNN accuracy, parity with umap-learn (`-m parity`)
- [x] Metrics, nulls/NaN/inf, errors, edge cases, degenerate data, d = 1024
- [x] kNN recall vs exact, `umap_from_knn` (with and without self, List input, custom init)
- [x] Hypothesis property tests
- [x] Rust unit tests (conversion)
- [x] Doctests

### Phase 3: GPU
- [x] `gpu` cargo feature, on by default in wheels; `gpu=` parameter; CPU fallback; tests (`-m gpu`)

### Phase 4: benchmarks
- [x] `benchmarks/run.py` + `_worker.py` (subprocess isolation, quality metrics, upstream binary, GPU)
- [x] `benchmarks/groups.py` (many groups)
- [x] `benchmarks/report.py` → docs/benchmarks.md + figures
- [x] pytest-benchmark micro-benchmarks (`benchmarks/test_micro.py`)
- [x] Benchmarks run locally (Apple Silicon); results in `benchmarks/results/`

### Phase 5: examples, docs, CI, release
- [x] examples (quickstart, text embeddings, per-group, lazy parquet, precomputed kNN, marimo notebook)
- [x] README, CHANGELOG
- [x] mkdocs site config (index, examples, API reference, benchmarks)
- [x] Compatibility tested locally: py3.9 + polars 1.0.0, py3.10 + 1.20/1.25, py3.11 + 1.30, py3.12 + 1.34/1.39, py3.13 + 2.0, py3.14 + 2.1
- [~] GitHub Actions written (lint, test matrix incl. oldest versions, parity, micro-benchmarks, wheels for linux/macos/windows, trusted publishing). **Not yet run**, because there's no remote repo yet.
- [x] sdist decision: no vendoring. The sdist builds against the pinned git rev, so building from source needs network access (documented). `scripts/bump_upstream.sh` updates the pin.

### Upstream bug found (not yet reported)
Hypothesis found an input where upstream's **recursive init** (the 0.6 default) returns an all-NaN embedding and umap-learn 0.6 doesn't:
`[[2,0],[1,2],[3,2]] + [[2,2]]*18`, with n_components=1, n_neighbors ∈ {2,5,15}, n_epochs=20, seed 0. Spectral, PCA and random init are fine.
polars-umap works around it by retrying with spectral init and raising a clear error if the result is still non-finite (`fit_checked` in `src/expressions.rs`). Regression test: `tests/test_params.py::test_heavily_duplicated_input_gives_finite_embedding`.
- [ ] Report to apple/embedding-atlas (needs Jeroen's OK)

### Open / for later
- [ ] Create the GitHub repo, push, and make CI green
- [ ] Configure the PyPI trusted publisher (environment `pypi`) and tag v0.1.0
- [ ] Publish docs (GitHub Pages)
- [ ] Consider asking upstream to publish the crates to crates.io (needs a name other than `umap`)
- [ ] Transform of new points once upstream supports it
