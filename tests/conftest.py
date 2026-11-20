from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import load_digits, make_blobs

import polars_umap  # noqa: F401  (registers the namespaces)


@pytest.fixture(scope="session")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=600, n_features=16, centers=5, cluster_std=1.0, random_state=0)
    return X.astype(np.float32), y


@pytest.fixture(scope="session")
def blobs_df(blobs: tuple[np.ndarray, np.ndarray]) -> pl.DataFrame:
    X, y = blobs
    return pl.DataFrame({"id": np.arange(len(X)), "emb": X, "label": y})


@pytest.fixture(scope="session")
def digits() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_digits(return_X_y=True)
    return X.astype(np.float32), y


@pytest.fixture(scope="session")
def digits_df(digits: tuple[np.ndarray, np.ndarray]) -> pl.DataFrame:
    X, y = digits
    return pl.DataFrame({"emb": X, "label": y})


@pytest.fixture
def small_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    return pl.DataFrame({"emb": X, "g": np.arange(120) % 3})
