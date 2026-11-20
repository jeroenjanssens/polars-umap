"""The ``umap`` namespace on ``pl.Expr``, ``pl.DataFrame`` and ``pl.LazyFrame``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

from polars_umap._functions import knn, umap, umap_from_knn
from polars_umap._utils import check_choice

if TYPE_CHECKING:
    from polars_umap._typing import IntoExpr, Output

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


@pl.api.register_expr_namespace("umap")
class UmapExprNamespace:
    """UMAP and nearest neighbor search on an expression.

    Available as ``pl.col(...).umap`` after ``import polars_umap``.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def fit_transform(self, **params: Any) -> pl.Expr:
        """Embed this column with UMAP. Takes the same parameters as :func:`polars_umap.umap`.

        Examples
        --------
        >>> import polars as pl
        >>> import polars_umap  # noqa: F401
        >>> df = pl.DataFrame({"v": [[float(i), float(i % 5), float(i % 3)] for i in range(50)]})
        >>> out = df.select(pl.col("v").umap.fit_transform(random_state=0))
        >>> out.schema["v"] == pl.Array(pl.Float32, 2)
        True
        """
        return umap(self._expr, **params)

    def knn(self, **params: Any) -> pl.Expr:
        """Approximate k nearest neighbors. Takes the same parameters as :func:`polars_umap.knn`."""
        return knn(self._expr, **params)

    def from_knn(self, distances: IntoExpr, **params: Any) -> pl.Expr:
        """UMAP on a precomputed kNN graph, with this expression as the indices.

        See :func:`polars_umap.umap_from_knn`.
        """
        return umap_from_knn(self._expr, distances, **params)


class _FrameNamespace:
    def __init__(self, frame: Any) -> None:
        self._frame = frame

    def fit_transform(
        self,
        *features: IntoExpr | list[IntoExpr],
        output: Output = "columns",
        name: str = "umap",
        prefix: str = "umap_",
        **params: Any,
    ) -> Any:
        """Add a UMAP embedding of ``features`` to the frame.

        Parameters
        ----------
        *features
            Column names, expressions or selectors; see :func:`polars_umap.umap`.
        output
            ``"columns"`` adds one ``Float32`` column per component, named
            ``{prefix}0``, ``{prefix}1``, and so on. ``"array"`` adds a single
            ``Array(Float32, n_components)`` column called ``name``.
        name
            Name of the embedding column when ``output="array"``.
        prefix
            Prefix of the component columns when ``output="columns"``.
        **params
            Passed on to :func:`polars_umap.umap`.

        Returns
        -------
        DataFrame or LazyFrame
            The input frame with the embedding added.
        """
        check_choice("output", output, ("columns", "array"))
        expr = umap(*features, **params)
        if output == "array":
            return self._frame.with_columns(expr.alias(name))
        n_components = params.get("n_components", 2)
        tmp = "__polars_umap_embedding__"
        return (
            self._frame.with_columns(expr.alias(tmp))
            .with_columns(pl.col(tmp).arr.get(i).alias(f"{prefix}{i}") for i in range(n_components))
            .drop(tmp)
        )

    def knn(
        self,
        *features: IntoExpr | list[IntoExpr],
        name: str = "knn",
        **params: Any,
    ) -> Any:
        """Add a ``Struct`` column ``name`` with the k nearest neighbors of each row.

        See :func:`polars_umap.knn` for the parameters.
        """
        return self._frame.with_columns(knn(*features, **params).alias(name))


@pl.api.register_dataframe_namespace("umap")
class UmapDataFrameNamespace(_FrameNamespace):
    """UMAP on a DataFrame. Available as ``df.umap`` after ``import polars_umap``."""

    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__(df)


@pl.api.register_lazyframe_namespace("umap")
class UmapLazyFrameNamespace(_FrameNamespace):
    """UMAP on a LazyFrame. Available as ``lf.umap`` after ``import polars_umap``."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        super().__init__(lf)
