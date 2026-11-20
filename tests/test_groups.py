"""`.over()` and group_by: one UMAP per group."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import to_numpy


def test_over_equals_per_group_fit(blobs_df: pl.DataFrame) -> None:
    expr = pl.col("emb").umap.fit_transform(random_state=0)
    out = blobs_df.with_columns(xy=expr.over("label"))
    assert out.height == blobs_df.height
    for label in range(5):
        part = blobs_df.filter(pl.col("label") == label)
        expected = to_numpy(part.select(expr)["emb"])
        got = to_numpy(out.filter(pl.col("label") == label)["xy"])
        np.testing.assert_array_equal(got, expected)


def test_over_keeps_row_order(blobs_df: pl.DataFrame) -> None:
    shuffled = blobs_df.sample(fraction=1.0, shuffle=True, seed=0)
    out = shuffled.with_columns(xy=pl.col("emb").umap.fit_transform(random_state=0).over("label"))
    assert out["id"].to_list() == shuffled["id"].to_list()


def test_group_by_agg(blobs_df: pl.DataFrame) -> None:
    out = (
        blobs_df.group_by("label", maintain_order=True)
        .agg(pl.col("id"), pl.col("emb").umap.fit_transform(random_state=0).alias("xy"))
        .explode("id", "xy")
    )
    assert out.height == blobs_df.height
    assert out.schema["xy"] == pl.Array(pl.Float32, 2)


def test_many_small_groups() -> None:
    rng = np.random.default_rng(0)
    n = 2_000
    df = pl.DataFrame({"emb": rng.normal(size=(n, 4)).astype(np.float32), "g": np.arange(n) % 100})
    out = df.with_columns(
        xy=pl.col("emb").umap.fit_transform(random_state=0, n_epochs=50).over("g")
    )
    assert np.isfinite(to_numpy(out["xy"])).all()


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_tiny_groups(size: int) -> None:
    df = pl.DataFrame(
        {"emb": np.arange(size * 2, dtype=np.float32).reshape(size, 2), "g": [0] * size}
    )
    out = df.with_columns(xy=pl.col("emb").umap.fit_transform(random_state=0).over("g"))
    assert np.isfinite(to_numpy(out["xy"])).all()


def test_knn_over_uses_group_local_indices(blobs_df: pl.DataFrame) -> None:
    out = blobs_df.with_columns(nn=pl.col("emb").umap.knn(k=5).over("label"))
    counts = blobs_df.group_by("label").len()
    max_group = counts["len"].max()
    idx = np.asarray(out["nn"].struct.field("indices").to_list())
    assert idx.max() < max_group
