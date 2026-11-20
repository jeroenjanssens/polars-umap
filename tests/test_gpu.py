"""GPU (WebGPU) path. Without an adapter the code falls back to the CPU, so these run everywhere."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import knn_accuracy, to_numpy


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["0.5", "0.6"])
def test_gpu_umap(digits, digits_df: pl.DataFrame, layout: str) -> None:
    _, y = digits
    out = digits_df.select(
        pl.col("emb").umap.fit_transform(gpu=True, layout=layout, random_state=0)
    )
    emb = to_numpy(out["emb"])
    assert np.isfinite(emb).all()
    assert knn_accuracy(emb, y) > 0.85


@pytest.mark.gpu
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_gpu_knn(digits_df: pl.DataFrame, metric: str) -> None:
    cpu = digits_df.select(pl.col("emb").umap.knn(k=10, metric=metric, random_state=0))
    gpu = digits_df.select(pl.col("emb").umap.knn(k=10, metric=metric, random_state=0, gpu=True))
    a = np.asarray(cpu["emb"].struct.field("indices").to_list())
    b = np.asarray(gpu["emb"].struct.field("indices").to_list())
    overlap = np.mean([len(set(x) & set(z)) / 10 for x, z in zip(a, b)])
    assert overlap > 0.9
