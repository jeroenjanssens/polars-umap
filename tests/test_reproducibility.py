"""Seeds, determinism and threading."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import knn_accuracy, to_numpy, trust


def run(df: pl.DataFrame, **kw) -> np.ndarray:
    return to_numpy(df.select(pl.col("emb").umap.fit_transform(**kw))["emb"])


def test_same_seed_same_result(blobs_df: pl.DataFrame) -> None:
    np.testing.assert_array_equal(run(blobs_df, random_state=7), run(blobs_df, random_state=7))


def test_different_seed_different_result(blobs_df: pl.DataFrame) -> None:
    assert not np.array_equal(run(blobs_df, random_state=1), run(blobs_df, random_state=2))


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_deterministic_across_thread_counts(blobs_df: pl.DataFrame, n_threads: int) -> None:
    ref = run(blobs_df, random_state=0)
    np.testing.assert_array_equal(run(blobs_df, random_state=0, n_threads=n_threads), ref)


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_modern_optimizers_are_deterministic(blobs_df: pl.DataFrame, optimizer: str) -> None:
    a = run(blobs_df, random_state=0, optimizer=optimizer)
    b = run(blobs_df, random_state=0, optimizer=optimizer)
    np.testing.assert_array_equal(a, b)


def test_legacy_layout_single_thread_is_deterministic(blobs_df: pl.DataFrame) -> None:
    a = run(blobs_df, random_state=0, layout="0.5", n_threads=1)
    b = run(blobs_df, random_state=0, layout="0.5", n_threads=1)
    np.testing.assert_array_equal(a, b)


def test_legacy_layout_multithreaded_is_equally_good(blobs, blobs_df: pl.DataFrame) -> None:
    # Hogwild SGD is racy, so only the quality is guaranteed, not the exact values.
    X, y = blobs
    for _ in range(2):
        emb = run(blobs_df, random_state=0, layout="0.5", n_threads=4)
        assert knn_accuracy(emb, y) > 0.99
        assert trust(X, emb) > 0.9


def test_unseeded_runs_work(blobs_df: pl.DataFrame) -> None:
    assert np.isfinite(run(blobs_df)).all()
