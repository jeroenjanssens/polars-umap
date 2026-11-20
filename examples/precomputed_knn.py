"""Nearest neighbors, and UMAP on a precomputed kNN graph.

uv run --group examples python examples/precomputed_knn.py
"""

import numpy as np
import polars as pl
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors

import polars_umap as pu

X, y = load_digits(return_X_y=True)
df = pl.DataFrame({"pixels": X, "digit": y})

# 1. Approximate neighbors with NNDescent (the search UMAP uses internally)
nn = df.with_columns(nn=pl.col("pixels").umap.knn(k=10, random_state=0))
print(nn.select("digit", "nn").head(3))

# How often is a neighbor the same digit?
same = (
    nn.with_row_index()
    .select("index", "digit", pl.col("nn").struct.field("indices").arr.to_list().alias("nbr"))
    .explode("nbr")
    .filter(pl.col("nbr") != pl.col("index"))
    .join(
        df.with_row_index().select(
            pl.col("index").alias("nbr"), pl.col("digit").alias("nbr_digit")
        ),
        on="nbr",
    )
    .select((pl.col("digit") == pl.col("nbr_digit")).mean())
    .item()
)
print(f"{same:.1%} of neighbors are the same digit")

# 2. UMAP on that graph: no need to recompute neighbors
graph = nn.select(pl.col("nn").struct.unnest())
emb = graph.select(xy=pu.umap_from_knn("indices", "distances", random_state=0))
print(emb.head(3))

# 3. ...or on a graph from somewhere else, e.g. an exact search
dist, idx = NearestNeighbors(n_neighbors=15).fit(X).kneighbors(X)
exact = pl.DataFrame({"idx": idx.astype(np.int64), "dist": dist.astype(np.float32)})
print(exact.select(xy=pl.col("idx").umap.from_knn("dist", random_state=0)).head(3))
