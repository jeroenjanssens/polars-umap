"""UMAP dimensionality reduction as a Polars plugin.

Importing this package registers a ``umap`` namespace on ``pl.Expr``,
``pl.DataFrame`` and ``pl.LazyFrame``:

>>> import polars as pl
>>> import polars_umap as pu
>>> df = pl.DataFrame({"x": range(100), "y": [i % 7 for i in range(100)]})
>>> df.umap.fit_transform("x", "y", random_state=0).columns
['x', 'y', 'umap_0', 'umap_1']

The algorithm is the Rust port of umap-learn and PyNNDescent from Apple's
`embedding-atlas <https://github.com/apple/embedding-atlas>`_.
"""

from __future__ import annotations

from polars_umap._functions import knn, umap, umap_from_knn
from polars_umap._internal import __version__

# Importing the namespace module registers the `umap` namespaces.
from polars_umap._namespace import (
    UmapDataFrameNamespace,
    UmapExprNamespace,
    UmapLazyFrameNamespace,
)

__all__ = [
    "UmapDataFrameNamespace",
    "UmapExprNamespace",
    "UmapLazyFrameNamespace",
    "__version__",
    "knn",
    "umap",
    "umap_from_knn",
]
