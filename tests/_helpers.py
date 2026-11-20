from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def to_numpy(s: pl.Series) -> np.ndarray:
    """Array column -> 2D numpy array (works on all supported Polars versions)."""
    return np.asarray(s.to_list(), dtype=np.float32)


def trust(X: np.ndarray, emb: np.ndarray, k: int = 10) -> float:
    return float(trustworthiness(X, emb, n_neighbors=k))


def knn_accuracy(emb: np.ndarray, y: np.ndarray) -> float:
    return float(cross_val_score(KNeighborsClassifier(5), emb, y, cv=3).mean())
