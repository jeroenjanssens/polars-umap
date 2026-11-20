"""Compare with umap-learn (optional: needs the `parity` dependency group)."""

from __future__ import annotations

import polars as pl
import pytest
from _helpers import knn_accuracy, to_numpy, trust

umap_learn = pytest.importorskip("umap")

pytestmark = [pytest.mark.parity, pytest.mark.filterwarnings("ignore::UserWarning")]


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_quality_close_to_umap_learn(digits, digits_df: pl.DataFrame, metric: str) -> None:
    X, y = digits
    ours = to_numpy(
        digits_df.select(pl.col("emb").umap.fit_transform(metric=metric, random_state=0))["emb"]
    )
    theirs = umap_learn.UMAP(metric=metric, random_state=0).fit_transform(X)
    assert trust(X, ours) > trust(X, theirs) - 0.02
    assert knn_accuracy(ours, y) > knn_accuracy(theirs, y) - 0.03


def test_same_cluster_structure_as_umap_learn(blobs, blobs_df: pl.DataFrame) -> None:
    X, y = blobs
    ours = to_numpy(blobs_df.select(pl.col("emb").umap.fit_transform(random_state=0))["emb"])
    theirs = umap_learn.UMAP(random_state=0).fit_transform(X)
    assert knn_accuracy(ours, y) == pytest.approx(knn_accuracy(theirs, y), abs=0.01)
