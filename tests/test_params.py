"""Every parameter value is accepted and changes something sensible."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import to_numpy

import polars_umap as pu
from polars_umap._functions import INIT_METHODS, METRICS


def run(df: pl.DataFrame, **kw) -> np.ndarray:
    emb = to_numpy(df.select(pl.col("emb").umap.fit_transform(**kw))["emb"])
    assert emb.shape[0] == df.height
    assert np.isfinite(emb).all()
    return emb


@pytest.mark.parametrize("init", INIT_METHODS)
@pytest.mark.parametrize("layout", ["0.5", "0.6"])
def test_init_methods(small_df: pl.DataFrame, init: str, layout: str) -> None:
    run(small_df, init=init, layout=layout, random_state=0)


@pytest.mark.parametrize("metric", METRICS)
def test_metrics(small_df: pl.DataFrame, metric: str) -> None:
    df = small_df
    if metric == "hamming":
        df = df.with_columns(
            (pl.col("emb").arr.to_list().list.eval(pl.element() > 0)).cast(pl.List(pl.Float32))
        )
    run(df, metric=metric, random_state=0)


@pytest.mark.parametrize(
    ("optimizer", "layout"), [("sgd", "0.5"), ("sgd", "0.6"), ("adam", "0.6"), ("momentum", "0.6")]
)
def test_optimizers(small_df: pl.DataFrame, optimizer: str, layout: str) -> None:
    run(small_df, optimizer=optimizer, layout=layout, random_state=0)


def test_custom_init_numpy_series_and_expr(small_df: pl.DataFrame) -> None:
    init = np.random.default_rng(0).uniform(-5, 5, size=(small_df.height, 2)).astype(np.float32)
    a = run(small_df, init=init, random_state=0)
    b = run(small_df, init=pl.Series(init), random_state=0)
    c = run(small_df.with_columns(init=pl.Series(init)), init=pl.col("init"), random_state=0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_custom_init_is_used(small_df: pl.DataFrame) -> None:
    init = np.random.default_rng(0).uniform(-5, 5, size=(small_df.height, 2))
    a = run(small_df, init=init, n_epochs=1, random_state=0)
    b = run(small_df, init=init + 100, n_epochs=1, random_state=0)
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_epochs": 10},
        {"learning_rate": 0.5},
        {"negative_sample_rate": 2},
        {"repulsion_strength": 2.0},
        {"local_connectivity": 2.0},
        {"set_op_mix_ratio": 0.5},
        {"a": 1.5, "b": 0.9},
        {"spread": 2.0, "min_dist": 0.5},
        {"n_neighbors": 5},
        {"n_neighbors": 500},
        {"disconnection_distance": 3.0},
        {"verbose": True},
        {"min_dist": 0},
    ],
)
def test_parameters_change_the_result(small_df: pl.DataFrame, kwargs: dict) -> None:
    base = run(small_df, random_state=0)
    other = run(small_df, random_state=0, **kwargs)
    if kwargs not in ({"verbose": True}, {"n_neighbors": 500}, {"disconnection_distance": 3.0}):
        assert not np.array_equal(base, other)


def test_n_neighbors_is_clamped(small_df: pl.DataFrame) -> None:
    np.testing.assert_array_equal(
        run(small_df, n_neighbors=10_000, random_state=0),
        run(small_df, n_neighbors=small_df.height - 1, random_state=0),
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 10, 16, 17])
def test_tiny_inputs(n: int) -> None:
    X = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
    df = pl.DataFrame({"emb": pl.Series(X.tolist(), dtype=pl.Array(pl.Float32, 3))})
    out = df.select(pl.col("emb").umap.fit_transform(random_state=0))
    assert out.height == n
    assert np.isfinite(to_numpy(out["emb"]).reshape(n, 2)).all()
    assert df.select(pu.knn("emb", k=3)).height == n


def test_degenerate_data() -> None:
    dup = pl.DataFrame({"emb": np.ones((50, 4), dtype=np.float32)})
    assert np.isfinite(to_numpy(dup.select(pu.umap("emb", random_state=0)).to_series())).all()
    const = pl.DataFrame({"a": [1.0] * 50, "b": list(range(50))})
    assert np.isfinite(to_numpy(const.select(pu.umap("a", "b", random_state=0)).to_series())).all()


def test_high_dimensional() -> None:
    X = np.random.default_rng(0).normal(size=(200, 1024)).astype(np.float32)
    df = pl.DataFrame({"emb": X})
    assert np.isfinite(
        to_numpy(df.select(pu.umap("emb", metric="cosine", random_state=0)).to_series())
    ).all()


def test_heavily_duplicated_input_gives_finite_embedding() -> None:
    # Found by Hypothesis: upstream's recursive init yields NaN here (umap-learn doesn't).
    # polars-umap falls back to spectral init.
    rows = [[2, 0], [1, 2], [3, 2]] + [[2, 2]] * 18
    df = pl.DataFrame({"emb": rows}, schema={"emb": pl.Array(pl.Float32, 2)})
    for n_neighbors in (2, 5, 15):
        out = df.select(
            pu.umap("emb", n_components=1, n_neighbors=n_neighbors, random_state=0, n_epochs=20)
        )
        assert np.isfinite(to_numpy(out["emb"])).all()
