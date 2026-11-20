"""Micro-benchmarks (pytest-benchmark) to catch performance regressions in CI.

uv run --group bench pytest benchmarks/test_micro.py --benchmark-only
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_umap as pu

rng = np.random.default_rng(0)
X = rng.normal(size=(5_000, 64)).astype(np.float32)
DF_ARRAY = pl.DataFrame({"emb": X})
DF_LIST = DF_ARRAY.with_columns(pl.col("emb").arr.to_list())
DF_WIDE = pl.DataFrame(X, schema=[f"f{i}" for i in range(64)])


@pytest.mark.parametrize("df", [DF_ARRAY, DF_LIST], ids=["array", "list"])
def test_umap_5k(benchmark, df: pl.DataFrame) -> None:
    benchmark.pedantic(lambda: df.select(pu.umap("emb", random_state=0)), rounds=3, iterations=1)


def test_umap_5k_columns(benchmark) -> None:
    benchmark.pedantic(
        lambda: DF_WIDE.select(pu.umap(pl.all(), random_state=0)), rounds=3, iterations=1
    )


def test_knn_5k(benchmark) -> None:
    benchmark.pedantic(
        lambda: DF_ARRAY.select(pu.knn("emb", k=15, random_state=0)), rounds=3, iterations=1
    )


def test_tiny_groups(benchmark) -> None:
    df = DF_ARRAY.with_columns(g=pl.int_range(pl.len()) % 500)
    benchmark.pedantic(
        lambda: df.select(pu.umap("emb", random_state=0, n_epochs=20).over("g")),
        rounds=3,
        iterations=1,
    )
