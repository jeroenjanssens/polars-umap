"""null_policy: rows that are null or contain null, NaN or infinite values."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import to_numpy

import polars_umap as pu


@pytest.fixture
def dirty() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(60, 3)).tolist()
    rows[3] = None
    rows[10] = [1.0, None, 2.0]
    rows[20] = [1.0, float("nan"), 2.0]
    rows[30] = [float("inf"), 0.0, 0.0]
    return pl.DataFrame({"emb": rows}, schema={"emb": pl.Array(pl.Float64, 3)})


BAD = [3, 10, 20, 30]


def test_error_is_default(dirty: pl.DataFrame) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match=r"4 of 60 rows .*first at row 3"):
        dirty.select(pl.col("emb").umap.fit_transform())


def test_ignore_returns_null_for_bad_rows(dirty: pl.DataFrame) -> None:
    out = dirty.select(pl.col("emb").umap.fit_transform(null_policy="ignore", random_state=0))
    assert out.height == 60
    s = out["emb"]
    assert s.null_count() == 4
    assert s.is_null().arg_true().to_list() == BAD


def test_ignore_equals_fit_on_clean_rows(dirty: pl.DataFrame) -> None:
    out = dirty.select(pl.col("emb").umap.fit_transform(null_policy="ignore", random_state=0))
    clean = dirty.with_row_index().filter(~pl.col("index").is_in(BAD))
    expected = clean.select(pl.col("emb").umap.fit_transform(random_state=0))
    np.testing.assert_array_equal(to_numpy(out["emb"].drop_nulls()), to_numpy(expected["emb"]))


def test_ignore_with_list_input(dirty: pl.DataFrame) -> None:
    as_list = dirty.with_columns(pl.col("emb").arr.to_list())
    out = as_list.select(pl.col("emb").umap.fit_transform(null_policy="ignore", random_state=0))
    assert out["emb"].is_null().arg_true().to_list() == BAD


def test_ignore_with_multiple_columns() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0, 6.0] * 10, "b": list(range(60))})
    out = df.select(pu.umap("a", "b", null_policy="ignore", random_state=0))
    assert out.to_series().null_count() == 10


def test_all_null() -> None:
    df = pl.DataFrame({"emb": [None, None]}, schema={"emb": pl.Array(pl.Float32, 4)})
    out = df.select(pl.col("emb").umap.fit_transform(null_policy="ignore"))
    assert out["emb"].null_count() == 2


def test_knn_ignore_skips_bad_rows(dirty: pl.DataFrame) -> None:
    out = dirty.select(pl.col("emb").umap.knn(k=5, null_policy="ignore"))
    s = out["emb"]
    assert s.is_null().arg_true().to_list() == BAD
    idx = {i for row in s.struct.field("indices").drop_nulls().to_list() for i in row}
    assert idx.isdisjoint(BAD)
    with pytest.raises(pl.exceptions.ComputeError):
        dirty.select(pl.col("emb").umap.knn(k=5))


def test_custom_init_with_ignored_rows(dirty: pl.DataFrame) -> None:
    init = np.random.default_rng(0).normal(size=(60, 2))
    out = dirty.select(
        pl.col("emb").umap.fit_transform(init=init, null_policy="ignore", random_state=0)
    )
    assert out["emb"].null_count() == 4
