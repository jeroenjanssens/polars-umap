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

from polars_umap import _namespace  # noqa: F401  (registers the namespaces)
from polars_umap._functions import knn, umap, umap_from_knn
from polars_umap._internal import __version__

__all__ = ["__version__", "knn", "umap", "umap_from_knn"]
