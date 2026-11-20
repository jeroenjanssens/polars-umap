"""NNDescent kNN and UMAP from a precomputed kNN graph."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _helpers import knn_accuracy, to_numpy
from sklearn.neighbors import NearestNeighbors

import polars_umap as pu


def knn_arrays(df: pl.DataFrame, col: str = "nn") -> tuple[np.ndarray, np.ndarray]:
    s = df[col]
    return (
        np.asarray(s.struct.field("indices").to_list()),
        np.asarray(s.struct.field("distances").to_list(), dtype=np.float32),
    )


def test_knn_schema(small_df: pl.DataFrame) -> None:
    out = small_df.select(nn=pl.col("emb").umap.knn(k=7))
    dtype = out.schema["nn"]
    assert isinstance(dtype, pl.Struct)
    fields = {f.name: f.dtype for f in dtype.fields}
    assert fields["indices"] == pl.Array(pl.UInt32, 7) or fields["indices"] == pl.Array(
        pl.UInt64, 7
    )
    assert fields["distances"] == pl.Array(pl.Float32, 7)


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
def test_knn_recall_vs_exact(digits, digits_df: pl.DataFrame, metric: str) -> None:
    X, _ = digits
    k = 10
    idx, dist = knn_arrays(
        digits_df.select(nn=pl.col("emb").umap.knn(k=k, metric=metric, random_state=0))
    )
    exact = (
        NearestNeighbors(n_neighbors=k, metric=metric).fit(X).kneighbors(X, return_distance=False)
    )
    recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(idx, exact)])
    assert recall > 0.95
    # Self is the first neighbor, distances are sorted.
    assert (idx[:, 0] == np.arange(len(X))).mean() > 0.99
    assert (np.diff(dist, axis=1) >= -1e-5).all()


def test_knn_distances_are_true_distances(digits, digits_df: pl.DataFrame) -> None:
    X, _ = digits
    idx, dist = knn_arrays(digits_df.select(nn=pl.col("emb").umap.knn(k=5, random_state=0)))
    expected = np.linalg.norm(X[:, None, :] - X[idx], axis=-1)
    np.testing.assert_allclose(dist, expected, rtol=1e-4, atol=1e-3)


def test_knn_fewer_rows_than_k() -> None:
    df = pl.DataFrame({"emb": np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)})
    out = df.select(nn=pl.col("emb").umap.knn(k=5))
    idx = out["nn"].struct.field("indices").to_list()
    assert idx[0][:3] == [0, 1, 2]
    assert idx[0][3:] == [None, None]
    dist = out["nn"].struct.field("distances").to_list()
    assert dist[2][:3] == pytest.approx([0.0, 2.0, 3.0])


def test_knn_deterministic(small_df: pl.DataFrame) -> None:
    a = small_df.select(nn=pu.knn("emb", k=5, random_state=0))
    b = small_df.select(nn=pu.knn("emb", k=5, random_state=0))
    assert a.equals(b)


def test_umap_from_knn_matches_quality(digits, digits_df: pl.DataFrame) -> None:
    _, y = digits
    nn = digits_df.select(nn=pl.col("emb").umap.knn(k=15, random_state=0)).unnest("nn")
    out = nn.select(xy=pu.umap_from_knn("indices", "distances", random_state=0))
    assert out.schema["xy"] == pl.Array(pl.Float32, 2)
    assert knn_accuracy(to_numpy(out["xy"]), y) > 0.9


def test_umap_from_exact_knn_lists(digits) -> None:
    X, y = digits
    dist, idx = NearestNeighbors(n_neighbors=15).fit(X).kneighbors(X)
    df = pl.DataFrame({"idx": idx.tolist(), "dist": dist.tolist()})
    out = df.select(pl.col("idx").umap.from_knn("dist", random_state=0, n_components=3))
    assert out.schema["idx"] == pl.Array(pl.Float32, 3)
    assert knn_accuracy(to_numpy(out["idx"]), y) > 0.9


def test_umap_from_knn_without_self(digits) -> None:
    X, y = digits
    dist, idx = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(X)
    df = pl.DataFrame({"idx": idx[:, 1:].tolist(), "dist": dist[:, 1:].tolist()})
    out = df.select(pu.umap_from_knn("idx", "dist", random_state=0))
    assert knn_accuracy(to_numpy(out["idx"]), y) > 0.9


def test_umap_from_knn_errors() -> None:
    df = pl.DataFrame(
        {"idx": [[0, 1], [1, 5], [2, 0]], "dist": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]}
    )
    with pytest.raises(pl.exceptions.ComputeError, match="in \\[0, 3\\)"):
        df.select(pu.umap_from_knn("idx", "dist"))
    df = pl.DataFrame({"idx": [[0, 1], [1, 0], [2, 0]], "dist": [[0.0], [0.0], [0.0]]})
    with pytest.raises(pl.exceptions.ComputeError, match="width"):
        df.select(pu.umap_from_knn("idx", "dist"))
    df = pl.DataFrame({"idx": [[0, 1], None, [2, 0]], "dist": [[0.0, 1.0]] * 3})
    with pytest.raises(pl.exceptions.ComputeError, match="null"):
        df.select(pu.umap_from_knn("idx", "dist"))
    with pytest.raises(pl.exceptions.ComputeError, match="pca"):
        pl.DataFrame({"idx": [[0, 1]] * 4, "dist": [[0.0, 1.0]] * 4}).select(
            pu.umap_from_knn("idx", "dist", init="pca")
        )


def test_umap_from_knn_custom_init(digits_df: pl.DataFrame) -> None:
    nn = digits_df.select(nn=pu.knn("emb", k=15, random_state=0)).unnest("nn")
    init = np.random.default_rng(0).uniform(-10, 10, size=(nn.height, 2))
    out = nn.select(pu.umap_from_knn("indices", "distances", init=init, random_state=0))
    assert np.isfinite(to_numpy(out.to_series())).all()
