# Changelog

## 0.1.0 (2026-11-20)

First release.

- `umap()` expression and the `pl.Expr.umap.fit_transform()` namespace method. They take an
  `Array`/`List` embedding column or several numeric columns and return `Array(Float32, k)`.
- `df.umap.fit_transform()` / `lf.umap.fit_transform()`, which add `umap_0…` columns or one Array
  column.
- `knn()` / `.umap.knn()`: approximate nearest neighbors (NNDescent).
- `umap_from_knn()` / `.umap.from_knn()`: UMAP on a precomputed kNN graph.
- The umap-learn 0.6 pipeline is the default (deterministic). `layout="0.5"` gives the classic one.
- Support for `null_policy`, custom initial embeddings, `n_threads` and `gpu=True` (wgpu).
- Python ≥ 3.9, Polars ≥ 1.0.
- Built on apple/embedding-atlas @ `a0d0f09`.
