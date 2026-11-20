"""Type aliases used in the public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    import polars as pl

Metric = Literal[
    "euclidean",
    "l2",
    "manhattan",
    "taxicab",
    "l1",
    "chebyshev",
    "infinity",
    "linfinity",
    "linfty",
    "linf",
    "minkowski",
    "cosine",
    "correlation",
    "hamming",
]
"""Distance metrics supported by the nearest neighbor search."""

InitMethod = Literal["auto", "spectral", "tswspectral", "random", "recursive", "pca"]
"""Named initialization methods. ``"auto"`` is ``"recursive"`` for the 0.6 layout and
``"spectral"`` for the 0.5 layout."""

Optimizer = Literal["sgd", "momentum", "adam"]
"""Layout optimizers. ``"momentum"`` and ``"adam"`` need ``layout="0.6"``."""

Layout = Literal["0.5", "0.6"]
"""Which umap-learn pipeline to reproduce."""

NullPolicy = Literal["error", "ignore"]
"""What to do with rows that are null or contain null, NaN or infinite values."""

Output = Literal["columns", "array"]
"""How the DataFrame helper returns the embedding."""

IntoExpr = Union["pl.Expr", str]
"""A column name or an expression."""
