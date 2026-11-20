"""Property-based tests: random shapes and dtypes never crash and keep the row count."""

from __future__ import annotations

import numpy as np
import polars as pl
from _helpers import to_numpy
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import polars_umap as pu

finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False, width=32)


@st.composite
def matrices(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(0, 60))
    d = draw(st.integers(1, 6))
    return draw(arrays(np.float32, (n, d), elements=finite))


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    X=matrices(), k=st.integers(1, 4), n_neighbors=st.integers(2, 20), seed=st.integers(0, 2**32)
)
def test_umap_never_crashes(X: np.ndarray, k: int, n_neighbors: int, seed: int) -> None:
    df = pl.DataFrame({"emb": pl.Series(X.tolist(), dtype=pl.Array(pl.Float32, X.shape[1]))})
    out = df.select(
        pu.umap("emb", n_components=k, n_neighbors=n_neighbors, random_state=seed, n_epochs=20)
    )
    assert out.height == X.shape[0]
    assert out.schema["emb"] == pl.Array(pl.Float32, k)
    emb = to_numpy(out["emb"]).reshape(X.shape[0], k)
    assert np.isfinite(emb).all()


@settings(max_examples=40, deadline=None)
@given(X=matrices(), k=st.integers(1, 10), mask=st.lists(st.booleans(), min_size=60, max_size=60))
def test_knn_never_crashes_and_skips_nulls(X: np.ndarray, k: int, mask: list[bool]) -> None:
    rows = [row if keep else None for row, keep in zip(X.tolist(), mask)]
    df = pl.DataFrame({"emb": pl.Series(rows, dtype=pl.Array(pl.Float32, X.shape[1]))})
    out = df.select(pu.knn("emb", k=k, null_policy="ignore"))
    assert out.height == X.shape[0]
    s = out["emb"]
    assert s.null_count() == sum(1 for r in rows if r is None)
    valid = {i for i, r in enumerate(rows) if r is not None}
    for neighbors in s.struct.field("indices").drop_nulls().to_list():
        assert {i for i in neighbors if i is not None} <= valid
