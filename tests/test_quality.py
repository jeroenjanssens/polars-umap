"""The embeddings must actually be good: neighborhoods preserved, clusters separated."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import knn_accuracy, to_numpy, trust


@pytest.mark.parametrize("layout", ["0.6", "0.5"])
def test_digits_quality(digits, digits_df: pl.DataFrame, layout: str) -> None:
    X, y = digits
    out = digits_df.select(pl.col("emb").umap.fit_transform(random_state=42, layout=layout))
    emb = to_numpy(out["emb"])
    assert trust(X, emb) > 0.95
    assert knn_accuracy(emb, y) > 0.9


def test_blobs_are_separated(blobs, blobs_df: pl.DataFrame) -> None:
    _, y = blobs
    emb = to_numpy(blobs_df.select(pl.col("emb").umap.fit_transform(random_state=0))["emb"])
    assert knn_accuracy(emb, y) > 0.99
    # Every cluster is far tighter than the distance between cluster centers.
    centers = np.stack([emb[y == c].mean(axis=0) for c in np.unique(y)])
    spread = max(np.linalg.norm(emb[y == c] - centers[c], axis=1).mean() for c in np.unique(y))
    gaps = [
        np.linalg.norm(centers[i] - centers[j])
        for i in range(len(centers))
        for j in range(i + 1, len(centers))
    ]
    assert min(gaps) > 2 * spread


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan", "correlation"])
def test_quality_per_metric(digits, digits_df: pl.DataFrame, metric: str) -> None:
    _, y = digits
    out = digits_df.select(pl.col("emb").umap.fit_transform(metric=metric, random_state=0))
    assert knn_accuracy(to_numpy(out["emb"]), y) > 0.85


def test_three_components_quality(digits, digits_df: pl.DataFrame) -> None:
    X, _ = digits
    emb = to_numpy(
        digits_df.select(pl.col("emb").umap.fit_transform(n_components=3, random_state=0))["emb"]
    )
    assert trust(X, emb) > 0.95


def test_min_dist_controls_tightness(blobs_df: pl.DataFrame) -> None:
    def mean_nn_dist(min_dist: float) -> float:
        emb = to_numpy(
            blobs_df.select(pl.col("emb").umap.fit_transform(min_dist=min_dist, random_state=0))[
                "emb"
            ]
        )
        d = np.linalg.norm(emb[:, None] - emb[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())

    assert mean_nn_dist(0.0) < mean_nn_dist(0.99)


@pytest.mark.slow
def test_larger_dataset_quality() -> None:
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=20_000, n_features=50, centers=10, random_state=0)
    df = pl.DataFrame({"emb": X.astype(np.float32)})
    emb = to_numpy(df.select(pl.col("emb").umap.fit_transform(random_state=0))["emb"])
    sub = np.random.default_rng(0).choice(len(X), 3000, replace=False)
    assert trust(X[sub], emb[sub]) > 0.9
    assert knn_accuracy(emb, y) > 0.99
