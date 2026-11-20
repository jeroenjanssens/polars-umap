"""Functional API: expressions that run UMAP and nearest neighbor search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, get_args

import polars as pl

from polars_umap._typing import InitMethod, Layout, Metric, NullPolicy, Optimizer
from polars_umap._utils import (
    check_choice,
    check_float,
    check_int,
    check_optional_float,
    check_optional_int,
    parse_into_expr,
    register,
)

if TYPE_CHECKING:
    import numpy as np

    from polars_umap._typing import IntoExpr

    InitLike = Union[InitMethod, pl.Expr, pl.Series, np.ndarray[Any, Any]]

METRICS: tuple[str, ...] = get_args(Metric)
INIT_METHODS: tuple[str, ...] = get_args(InitMethod)
OPTIMIZERS: tuple[str, ...] = get_args(Optimizer)
LAYOUTS: tuple[str, ...] = get_args(Layout)
NULL_POLICIES: tuple[str, ...] = get_args(NullPolicy)

__all__ = ["knn", "umap", "umap_from_knn"]


def _init_to_expr(init: Any) -> tuple[str, pl.Expr | None]:
    """Split ``init`` into a method name and an optional custom init expression."""
    if isinstance(init, str):
        return check_choice("init", init, INIT_METHODS), None
    if isinstance(init, (pl.Expr, pl.Series)):
        return "custom", parse_into_expr(init)
    if hasattr(init, "__array__"):
        import numpy as np

        arr = np.asarray(init, dtype=np.float32)
        if arr.ndim != 2:
            msg = f"a custom init must be a 2D array, got shape {arr.shape}"
            raise ValueError(msg)
        return "custom", pl.lit(_array_to_series(arr))
    msg = (
        "init must be one of "
        f"{INIT_METHODS}, an expression, a Series or a 2D array, got {type(init).__name__}"
    )
    raise TypeError(msg)


def _array_to_series(arr: np.ndarray[Any, Any]) -> pl.Series:
    try:
        return pl.Series("init", arr)
    except (TypeError, ValueError):  # pragma: no cover - very old Polars versions
        return pl.Series("init", arr.tolist())


def _layout_kwargs(
    *,
    n_components: int,
    min_dist: float,
    spread: float,
    n_epochs: int | None,
    learning_rate: float,
    init: str,
    negative_sample_rate: int,
    repulsion_strength: float,
    local_connectivity: float,
    set_op_mix_ratio: float,
    a: float | None,
    b: float | None,
    random_state: int | None,
    optimizer: str | None,
    layout: str,
    disconnection_distance: float | None,
    verbose: bool,
    n_threads: int | None,
    gpu: bool,
) -> dict[str, Any]:
    if layout == "0.5" and optimizer in ("adam", "momentum"):
        msg = f'optimizer="{optimizer}" requires layout="0.6"; the 0.5 layout always uses "sgd"'
        raise ValueError(msg)
    return {
        "n_components": check_int("n_components", n_components, 1),
        "min_dist": check_float("min_dist", min_dist),
        "spread": check_float("spread", spread),
        "n_epochs": check_optional_int("n_epochs", n_epochs, 1),
        "learning_rate": check_float("learning_rate", learning_rate),
        "init": init,
        "negative_sample_rate": check_int("negative_sample_rate", negative_sample_rate, 0),
        "repulsion_strength": check_float("repulsion_strength", repulsion_strength),
        "local_connectivity": check_float("local_connectivity", local_connectivity),
        "set_op_mix_ratio": check_float("set_op_mix_ratio", set_op_mix_ratio),
        "a": check_optional_float("a", a),
        "b": check_optional_float("b", b),
        "random_state": check_optional_int("random_state", random_state, 0),
        "optimizer": None
        if optimizer is None
        else check_choice("optimizer", optimizer, OPTIMIZERS),
        "layout": check_choice("layout", layout, LAYOUTS),
        "disconnection_distance": check_optional_float(
            "disconnection_distance", disconnection_distance
        ),
        "verbose": bool(verbose),
        "n_threads": check_optional_int("n_threads", n_threads, 1),
        "gpu": bool(gpu),
    }


def umap(
    *features: IntoExpr | list[IntoExpr],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: Metric = "euclidean",
    n_epochs: int | None = None,
    learning_rate: float = 1.0,
    init: InitLike = "auto",
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    local_connectivity: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    random_state: int | None = None,
    optimizer: Optimizer | None = None,
    layout: Layout = "0.6",
    disconnection_distance: float | None = None,
    null_policy: NullPolicy = "error",
    n_threads: int | None = None,
    gpu: bool = False,
    verbose: bool = False,
) -> pl.Expr:
    """Embed rows into a low-dimensional space with UMAP.

    UMAP (Uniform Manifold Approximation and Projection) is fitted on all rows the
    expression sees and returns their embedding, so the output has as many rows as
    the input. Use ``.over(...)`` to fit one UMAP per group.

    Parameters
    ----------
    *features
        The input. Either a single ``Array`` or ``List`` column holding one vector per
        row (e.g. text embeddings), or one or more numeric columns, which are stacked
        into a feature matrix. Column names, expressions and selectors are accepted.
        Values are converted to ``Float32``.
    n_components
        Dimensionality of the embedding.
    n_neighbors
        Size of the local neighborhood. Larger values preserve more global
        structure. Clamped to the number of rows minus one.
    min_dist
        Minimum distance between embedded points. Smaller values give tighter
        clusters. Must be in ``[0, spread]``.
    spread
        Scale of the embedded points. Together with ``min_dist`` this determines
        how clustered the embedding is.
    metric
        Distance metric of the input space.
    n_epochs
        Number of optimization epochs. ``None`` picks a default based on the size
        of the data (more epochs for smaller data).
    learning_rate
        Initial learning rate of the layout optimization.
    init
        How to initialize the embedding: ``"auto"`` (``"recursive"`` for the 0.6
        layout, ``"spectral"`` for 0.5), ``"spectral"``, ``"tswspectral"``,
        ``"random"``, ``"recursive"`` or ``"pca"``. You can also pass an expression,
        Series or 2D array with one ``n_components``-dimensional starting point
        per row.
    negative_sample_rate
        Number of negative samples per positive sample during optimization.
    repulsion_strength
        Weight of the negative samples.
    local_connectivity
        Number of nearest neighbors assumed to be connected locally.
    set_op_mix_ratio
        Interpolation between fuzzy union (1.0) and fuzzy intersection (0.0) when
        symmetrizing the neighbor graph.
    a, b
        Parameters of the low-dimensional similarity curve. Derived from
        ``min_dist`` and ``spread`` when not given.
    random_state
        Seed for the random number generators. With the default 0.6 layout, the
        same seed gives the same embedding.
    optimizer
        ``"adam"``, ``"momentum"`` or ``"sgd"``. Defaults to ``"adam"`` for the 0.6
        layout and ``"sgd"`` for the 0.5 layout.
    layout
        ``"0.6"`` (default) reproduces the umap-learn 0.6 pipeline: recursive
        initialization and a deterministic Adam optimizer. ``"0.5"`` reproduces the
        classic umap-learn 0.5 pipeline: spectral initialization and Hogwild SGD,
        which is not bit-for-bit reproducible when multithreaded.
    disconnection_distance
        Input distances at or above this value disconnect points in the neighbor
        graph. Defaults to 2 for cosine and correlation and infinity otherwise.
    null_policy
        ``"error"`` raises when a row is null or contains null, NaN or infinite
        values. ``"ignore"`` fits UMAP on the remaining rows and returns null for
        the invalid ones.
    n_threads
        Number of threads. ``None`` uses a shared pool sized by
        ``RAYON_NUM_THREADS``, ``POLARS_MAX_THREADS`` or the number of CPUs.
    gpu
        Use WebGPU (Metal, Vulkan or DirectX 12) for the nearest neighbor search
        and layout optimization. Falls back to the CPU when no GPU is available.
    verbose
        Print progress to standard error.

    Returns
    -------
    pl.Expr
        An ``Array(Float32, n_components)`` expression. It is named after the first
        feature column.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_umap as pu
    >>> df = pl.DataFrame({"x": range(100), "y": [i % 7 for i in range(100)]})
    >>> out = df.select(pu.umap("x", "y", random_state=0).alias("xy"))
    >>> out.schema["xy"] == pl.Array(pl.Float32, 2)
    True
    """
    if not features:
        msg = "umap() needs at least one feature column"
        raise ValueError(msg)
    flat: list[IntoExpr] = []
    for f in features:
        flat.extend(f if isinstance(f, list) else [f])
    exprs = [parse_into_expr(f) for f in flat]

    init_name, init_expr = _init_to_expr(init)
    kwargs = _layout_kwargs(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init_name,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
        local_connectivity=local_connectivity,
        set_op_mix_ratio=set_op_mix_ratio,
        a=a,
        b=b,
        random_state=random_state,
        optimizer=optimizer,
        layout=layout,
        disconnection_distance=disconnection_distance,
        verbose=verbose,
        n_threads=n_threads,
        gpu=gpu,
    )
    kwargs["n_neighbors"] = check_int("n_neighbors", n_neighbors, 2)
    kwargs["metric"] = check_choice("metric", metric, METRICS)
    kwargs["null_policy"] = check_choice("null_policy", null_policy, NULL_POLICIES)
    kwargs["custom_init"] = init_expr is not None
    if init_expr is not None:
        exprs.append(init_expr)
    return register("umap_fit_transform", exprs, kwargs)


def umap_from_knn(
    indices: IntoExpr,
    distances: IntoExpr,
    *,
    n_components: int = 2,
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_epochs: int | None = None,
    learning_rate: float = 1.0,
    init: InitLike = "auto",
    negative_sample_rate: int = 5,
    repulsion_strength: float = 1.0,
    local_connectivity: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    a: float | None = None,
    b: float | None = None,
    random_state: int | None = None,
    optimizer: Optimizer | None = None,
    layout: Layout = "0.6",
    disconnection_distance: float | None = None,
    n_threads: int | None = None,
    gpu: bool = False,
    verbose: bool = False,
) -> pl.Expr:
    """Run UMAP on a precomputed k-nearest-neighbor graph.

    Use this when you already have neighbors, for example from :func:`knn`, an
    exact search, a vector database or FAISS. The original vectors are not needed.

    Parameters
    ----------
    indices
        ``Array`` or ``List`` column of integer row indices of each row's nearest
        neighbors, sorted by ascending distance. The row itself may be included
        (usually first, at distance 0) or left out. Every row must have the same
        number of neighbors.
    distances
        ``Array`` or ``List`` column with the matching distances.
    n_components, min_dist, spread, n_epochs, learning_rate, negative_sample_rate,
    repulsion_strength, local_connectivity, set_op_mix_ratio, a, b, random_state,
    optimizer, layout, disconnection_distance, n_threads, gpu, verbose
        See :func:`umap`.
    init
        See :func:`umap`. ``"pca"`` is not available because it needs the vectors.

    Returns
    -------
    pl.Expr
        An ``Array(Float32, n_components)`` expression.
    """
    init_name, init_expr = _init_to_expr(init)
    kwargs = _layout_kwargs(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init_name,
        negative_sample_rate=negative_sample_rate,
        repulsion_strength=repulsion_strength,
        local_connectivity=local_connectivity,
        set_op_mix_ratio=set_op_mix_ratio,
        a=a,
        b=b,
        random_state=random_state,
        optimizer=optimizer,
        layout=layout,
        disconnection_distance=disconnection_distance,
        verbose=verbose,
        n_threads=n_threads,
        gpu=gpu,
    )
    kwargs["custom_init"] = init_expr is not None
    exprs = [parse_into_expr(indices), parse_into_expr(distances)]
    if init_expr is not None:
        exprs.append(init_expr)
    return register("umap_from_knn", exprs, kwargs)


def knn(
    *features: IntoExpr | list[IntoExpr],
    k: int = 15,
    metric: Metric = "euclidean",
    random_state: int | None = None,
    null_policy: NullPolicy = "error",
    n_threads: int | None = None,
    gpu: bool = False,
    verbose: bool = False,
) -> pl.Expr:
    """Find the approximate k nearest neighbors of every row with NNDescent.

    This is the neighbor search UMAP uses internally, a Rust port of PyNNDescent.

    Parameters
    ----------
    *features
        See :func:`umap`.
    k
        Number of neighbors per row. As in PyNNDescent, the row itself counts as
        one of them: it is normally the first neighbor, at distance 0.
    metric
        Distance metric.
    random_state
        Seed for the random number generator.
    null_policy
        See :func:`umap`. Ignored rows get a null result and never appear as
        neighbors.
    n_threads, gpu, verbose
        See :func:`umap`.

    Returns
    -------
    pl.Expr
        A ``Struct`` expression with fields ``indices`` (``Array(UInt32, k)``, row
        numbers in the input) and ``distances`` (``Array(Float32, k)``), sorted by
        ascending distance. When there are fewer than ``k`` rows, the missing
        neighbors are null.
    """
    if not features:
        msg = "knn() needs at least one feature column"
        raise ValueError(msg)
    flat: list[IntoExpr] = []
    for f in features:
        flat.extend(f if isinstance(f, list) else [f])
    kwargs = {
        "k": check_int("k", k, 1),
        "metric": check_choice("metric", metric, METRICS),
        "random_state": check_optional_int("random_state", random_state, 0),
        "null_policy": check_choice("null_policy", null_policy, NULL_POLICIES),
        "n_threads": check_optional_int("n_threads", n_threads, 1),
        "gpu": bool(gpu),
        "verbose": bool(verbose),
    }
    return register("knn", [parse_into_expr(f) for f in flat], kwargs)
