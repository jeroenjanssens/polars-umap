# polars-umap

**UMAP dimensionality reduction as a Polars plugin.**

```python
import polars as pl
import polars_umap  # registers the `umap` namespace

df.with_columns(xy=pl.col("embedding").umap.fit_transform(metric="cosine"))
```

polars-umap runs [UMAP](https://umap-learn.readthedocs.io) (Uniform Manifold Approximation and
Projection) directly inside Polars expressions. It is powered by the Rust ports of umap-learn and
PyNNDescent from Apple's [embedding-atlas](https://github.com/apple/embedding-atlas).

- **Native Polars**: works on `Array`/`List` embedding columns or plain numeric columns, in eager
  and lazy mode, and per group with `.over()` or `group_by().agg()`.
- **Fast**: 2–12× faster than umap-learn, with no JIT warm-up and optional GPU acceleration.
- **Faithful**: reproduces the umap-learn 0.6 pipeline by default, or the classic 0.5 pipeline.
  Embedding quality matches umap-learn.
- **Reproducible**: with a `random_state`, results are bit-identical regardless of the number of
  threads.
- **Small footprint**: one wheel that depends only on Polars. Supports Python ≥ 3.9 and
  Polars ≥ 1.0.

**Documentation: <https://jeroenjanssens.github.io/polars-umap/>**

## Installation

```bash
uv add polars-umap
# or
pip install polars-umap
```

## Quickstart

```python
import polars as pl
import polars_umap as pu
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
df = pl.DataFrame({"pixels": X, "digit": y})  # pixels: Array(Float64, 64)

# Expression API: returns an Array(Float32, 2) column
df.with_columns(xy=pl.col("pixels").umap.fit_transform(random_state=42))

# DataFrame API: adds umap_0 and umap_1 columns
df.umap.fit_transform("pixels", random_state=42)
```

![UMAP of the digits dataset](assets/images/digits.png)

## A taste of the API

```python
# Several numeric columns, stacked into a feature matrix (selectors work too)
pu.umap("sepal_length", "sepal_width", "petal_length", "petal_width")

# One UMAP per group, in a single query
pl.col("embedding").umap.fit_transform(random_state=0).over("category")

# Fit on the valid rows, return null for rows with null/NaN values
pl.col("embedding").umap.fit_transform(null_policy="ignore")

# Approximate nearest neighbors, and UMAP on a precomputed graph
pl.col("embedding").umap.knn(k=15, metric="cosine")
pu.umap_from_knn("indices", "distances")
```

The [user guide](https://jeroenjanssens.github.io/polars-umap/user-guide/quickstart.html)
covers these in depth, along with
[choosing parameters](https://jeroenjanssens.github.io/polars-umap/user-guide/choosing-parameters.html),
[reproducibility](https://jeroenjanssens.github.io/polars-umap/user-guide/reproducibility.html)
and [performance](https://jeroenjanssens.github.io/polars-umap/user-guide/performance.html).
The [examples](https://jeroenjanssens.github.io/polars-umap/gallery/) include a Quarto report and
an interactive marimo notebook.

## Benchmarks

![Fit time per dataset](assets/images/benchmark_fit_time_06.png)

- **2–12× faster than umap-learn**, with the same trustworthiness, kNN recall and label accuracy.
  This excludes umap-learn's 10–12 s numba warm-up.
- **No measurable overhead from Polars**: polars-umap is as fast as the raw upstream Rust binary.
- **`.over()` with 1,000 groups**: 18 s, against 58 s for a Python loop.

See the [benchmarks page](https://jeroenjanssens.github.io/polars-umap/user-guide/benchmarks.html)
for all results and the methodology.

## Limitations

The upstream implementation only fits, so there is no `transform()` for new points. Supervised UMAP,
densMAP and sparse input aren't supported either.

## Contributing

See [CONTRIBUTING.md](https://github.com/jeroenjanssens/polars-umap/blob/main/CONTRIBUTING.md) for how to build, test and benchmark polars-umap, and how
to build the documentation.

## Acknowledgements

The algorithms are implemented by Apple's
[embedding-atlas](https://github.com/apple/embedding-atlas) (MIT), which ports
[umap-learn](https://github.com/lmcinnes/umap) and
[PyNNDescent](https://github.com/lmcinnes/pynndescent) by Leland McInnes. Please cite the UMAP paper
if you use this in research:

> McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection
> for Dimension Reduction. arXiv:1802.03426.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
