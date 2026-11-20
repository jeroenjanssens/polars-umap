"""Benchmark datasets, cached as Parquet files under ``benchmarks/data``.

Names:

- ``blobs-<n>-<d>``: Gaussian blobs with 20 centers, e.g. ``blobs-10k-128`` or ``blobs-1m-16``.
- ``digits``: scikit-learn digits (1,797 x 64).
- ``mnist``: MNIST, 70,000 x 784 (downloaded from OpenML on first use).
- ``fashion-mnist``: Fashion-MNIST, 70,000 x 784 (OpenML).
- ``parquet:<path>:<column>``: any Parquet file with an Array/List embedding column.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).parent / "data"
SUFFIX = {"k": 1_000, "m": 1_000_000}


def _count(text: str) -> int:
    m = re.fullmatch(r"(\d+)([km]?)", text)
    if not m:
        msg = f"bad count {text!r}"
        raise ValueError(msg)
    return int(m.group(1)) * SUFFIX.get(m.group(2), 1)


def _metric(name: str) -> str:
    return "euclidean"


def load(name: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return ``(X, labels, metric)`` for a dataset name."""
    if name.startswith("parquet:"):
        _, path, column = name.split(":", 2)
        df = pl.read_parquet(path, columns=[column])
        X = np.asarray(df[column].to_list(), dtype=np.float32)
        return X, None, "cosine"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{name}.parquet"
    if cache.exists():
        df = pl.read_parquet(cache)
        X = np.asarray(df["emb"].to_numpy(), dtype=np.float32).reshape(df.height, -1)
        return X, df["label"].to_numpy(), _metric(name)

    if m := re.fullmatch(r"blobs-(\w+)-(\d+)", name):
        from sklearn.datasets import make_blobs

        n, d = _count(m.group(1)), int(m.group(2))
        X, y = make_blobs(n_samples=n, n_features=d, centers=20, cluster_std=2.0, random_state=0)
    elif name == "digits":
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)
    elif name in ("mnist", "fashion-mnist"):
        from sklearn.datasets import fetch_openml

        openml = {"mnist": "mnist_784", "fashion-mnist": "Fashion-MNIST"}[name]
        X, y = fetch_openml(openml, version=1, return_X_y=True, as_frame=False, parser="liac-arff")
        y = y.astype(int)
    else:
        msg = f"unknown dataset {name!r}"
        raise ValueError(msg)

    X = np.ascontiguousarray(X, dtype=np.float32)
    pl.DataFrame({"emb": X, "label": np.asarray(y, dtype=np.int32)}).write_parquet(cache)
    return X, np.asarray(y), _metric(name)
