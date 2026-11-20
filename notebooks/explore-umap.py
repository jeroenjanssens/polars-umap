# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20",
#     "polars>=1.0",
#     "polars-umap",
#     "altair>=5",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import importlib
    import importlib.util
    import io
    import sys

    import altair as alt
    import numpy as np
    import polars as pl

    # polars-umap is a compiled Rust extension. It runs when this notebook runs
    # on your machine, but not in the browser (WebAssembly), where the notebook
    # uses embeddings precomputed with polars-umap instead.
    IN_BROWSER = sys.platform == "emscripten"
    LIVE = not IN_BROWSER and importlib.util.find_spec("polars_umap") is not None
    return IN_BROWSER, LIVE, alt, importlib, io, np, pl


@app.cell(hide_code=True)
def _(LIVE, mo):
    mo.md(
        f"""
        # Exploring UMAP parameters

        Each point is one of 1,797 handwritten digits (8x8 pixels), embedded in 2D by
        [polars-umap](https://polars-umap.jeroenjanssens.com). Move the sliders to see how
        the two most important parameters shape the embedding:

        - **`n_neighbors`** sets how much of the neighborhood UMAP looks at. Small values
          focus on local detail, large values preserve more global structure.
        - **`min_dist`** sets how tightly points may be packed. Small values give tight,
          clumpy clusters, large values spread them out.

        {
            "_Computing live with polars-umap._"
            if LIVE
            else "_Running in the browser: showing embeddings precomputed with polars-umap "
            "(`random_state=42`). Run this notebook locally to compute them live._"
        }
        """
    )
    return


@app.cell
def _(mo):
    n_neighbors = mo.ui.slider(steps=[5, 15, 50, 200], value=15, label="`n_neighbors`")
    min_dist = mo.ui.slider(steps=[0.0, 0.1, 0.5, 0.99], value=0.1, label="`min_dist`")
    mo.hstack([n_neighbors, min_dist], justify="start", gap=2)
    return min_dist, n_neighbors


@app.cell(hide_code=True)
async def _(IN_BROWSER, LIVE, io, mo, np, pl):
    # Precomputed grid of embeddings, used when polars-umap is not available.
    if LIVE:
        grid = None
    else:
        _path = mo.notebook_location() / "public" / "digits-umap.npz"
        if IN_BROWSER:
            from pyodide.http import pyfetch

            _response = await pyfetch(str(_path))
            _source = io.BytesIO(await _response.bytes())
        else:
            _source = _path
        with np.load(_source) as _arrays:
            grid = pl.DataFrame({_name: _arrays[_name] for _name in _arrays.files})
    return (grid,)


@app.cell
def _(LIVE, grid, importlib, min_dist, n_neighbors, pl):
    if LIVE:
        importlib.import_module("polars_umap")  # registers the `umap` namespace
        _datasets = importlib.import_module("sklearn.datasets")
        _X, _y = _datasets.load_digits(return_X_y=True)
        points = (
            pl.DataFrame({"pixels": _X, "digit": _y})
            .umap.fit_transform(
                "pixels",
                n_neighbors=n_neighbors.value,
                min_dist=min_dist.value,
                random_state=42,
            )
            .select(
                pl.int_range(pl.len()).alias("row"),
                "digit",
                pl.col("umap_0").alias("x"),
                pl.col("umap_1").alias("y"),
            )
        )
    else:
        points = grid.filter(
            (pl.col("n_neighbors") == n_neighbors.value)
            & ((pl.col("min_dist") - min_dist.value).abs() < 1e-6)
        ).select("row", "digit", "x", "y")
    return (points,)


@app.cell
def _(alt, min_dist, n_neighbors, pl, points):
    _centers = points.group_by("digit").agg(pl.col("x", "y").median())
    _points = (
        alt.Chart(points)
        .mark_circle(size=14, opacity=0.55, color="#8d8b86")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            tooltip=[alt.Tooltip("digit:N", title="Digit"), alt.Tooltip("row:Q", title="Row")],
        )
    )
    _labels = (
        alt.Chart(_centers)
        .mark_text(fontSize=18, fontWeight="bold", color="#5f5d58")
        .encode(x="x:Q", y="y:Q", text="digit:N")
    )
    (_points + _labels).properties(
        width="container",
        height=460,
        title=f"Digits, n_neighbors={n_neighbors.value}, min_dist={min_dist.value}",
    ).configure_view(stroke=None).configure_title(anchor="start", color="#8d8b86")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        **Things to try**

        - `n_neighbors=5`: the clusters fall apart into small fragments, because each point
          only sees its closest neighbors.
        - `n_neighbors=200`: the digits that look alike (such as 3, 5 and 8) move closer
          together, because UMAP now sees more of the global structure.
        - `min_dist=0.99`: the clusters swell into soft blobs that nearly touch.

        Run this notebook on your own machine, with polars-umap computing live:

        ```bash
        uvx marimo edit --sandbox https://raw.githubusercontent.com/jeroenjanssens/polars-umap/main/notebooks/explore-umap.py
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
