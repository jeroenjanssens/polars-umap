"""Precompute the UMAP embeddings the marimo notebook uses in the browser.

polars-umap is a compiled Rust extension without a WebAssembly build, so the
notebook can't run it in Pyodide. It computes embeddings live when polars-umap
is installed and falls back to this grid of precomputed results otherwise.

    uv run python scripts/precompute_notebook_data.py
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.datasets import load_digits

import polars_umap  # noqa: F401

N_NEIGHBORS = [5, 15, 50, 200]
MIN_DIST = [0.0, 0.1, 0.5, 0.99]
OUT = Path(__file__).parent.parent / "notebooks" / "public" / "digits-umap.npz"

X, y = load_digits(return_X_y=True)
digits = pl.DataFrame({"pixels": X, "digit": y})

frames = []
for n_neighbors in N_NEIGHBORS:
    for min_dist in MIN_DIST:
        emb = digits.umap.fit_transform(
            "pixels", n_neighbors=n_neighbors, min_dist=min_dist, random_state=42
        )
        frames.append(
            emb.select(
                pl.lit(n_neighbors, pl.Int16).alias("n_neighbors"),
                pl.lit(min_dist, pl.Float32).alias("min_dist"),
                pl.int_range(pl.len(), dtype=pl.Int16).alias("row"),
                pl.col("digit").cast(pl.Int8),
                pl.col("umap_0").alias("x"),
                pl.col("umap_1").alias("y"),
            )
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
# A compressed NumPy archive: in the browser, Polars needs pyarrow to read Parquet,
# whereas NumPy reads this with the standard library alone.
grid = pl.concat(frames)
np.savez_compressed(OUT, **{name: grid[name].to_numpy() for name in grid.columns})
print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
