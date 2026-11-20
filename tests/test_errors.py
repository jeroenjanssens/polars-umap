"""Clear errors for bad input and bad parameters."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_umap as pu


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"n_components": 0}, ValueError, "n_components"),
        ({"n_components": 2.5}, TypeError, "n_components"),
        ({"n_neighbors": 1}, ValueError, "n_neighbors"),
        ({"n_epochs": 0}, ValueError, "n_epochs"),
        ({"metric": "nope"}, ValueError, "metric"),
        ({"init": "nope"}, ValueError, "init"),
        ({"init": 3}, TypeError, "init"),
        ({"init": np.zeros(5)}, ValueError, "2D"),
        ({"optimizer": "nope"}, ValueError, "optimizer"),
        ({"layout": "0.7"}, ValueError, "layout"),
        ({"layout": "0.5", "optimizer": "adam"}, ValueError, "requires layout"),
        ({"null_policy": "drop"}, ValueError, "null_policy"),
        ({"random_state": -1}, ValueError, "random_state"),
        ({"n_threads": 0}, ValueError, "n_threads"),
        ({"min_dist": "a"}, TypeError, "min_dist"),
        ({"n_components": True}, TypeError, "n_components"),
    ],
)
def test_parameter_validation(kwargs: dict, exc: type, match: str) -> None:
    with pytest.raises(exc, match=match):
        pu.umap("x", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_dist": 2.0}, "min_dist"),
        ({"spread": 0.0}, "spread|min_dist"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"set_op_mix_ratio": 1.5}, "set_op_mix_ratio"),
    ],
)
def test_runtime_parameter_validation(small_df: pl.DataFrame, kwargs: dict, match: str) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        small_df.select(pl.col("emb").umap.fit_transform(**kwargs))


def test_no_features() -> None:
    with pytest.raises(ValueError, match="at least one"):
        pu.umap()
    with pytest.raises(ValueError, match="at least one"):
        pu.knn()


def test_bad_feature_type() -> None:
    with pytest.raises(TypeError, match="column name"):
        pu.umap(42)  # type: ignore[arg-type]


def test_string_column() -> None:
    df = pl.DataFrame({"s": ["a", "b", "c"]})
    with pytest.raises(pl.exceptions.ComputeError, match="dtype"):
        df.select(pu.umap("s"))


def test_mixed_string_and_numeric_columns() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "s": ["a", "b", "c"]})
    with pytest.raises(pl.exceptions.ComputeError, match="numeric"):
        df.select(pu.umap("a", "s"))


def test_string_list_column() -> None:
    df = pl.DataFrame({"s": [["a"], ["b"]]})
    with pytest.raises(pl.exceptions.ComputeError, match="inner dtype"):
        df.select(pu.umap("s"))


def test_ragged_lists() -> None:
    df = pl.DataFrame({"emb": [[1.0, 2.0], [3.0], [4.0, 5.0]]})
    with pytest.raises(pl.exceptions.ComputeError, match="different lengths"):
        df.select(pl.col("emb").umap.fit_transform())


def test_empty_lists() -> None:
    df = pl.DataFrame({"emb": [[], []]}, schema={"emb": pl.List(pl.Float64)})
    with pytest.raises(pl.exceptions.ComputeError, match="empty"):
        df.select(pl.col("emb").umap.fit_transform())


def test_custom_init_wrong_width(small_df: pl.DataFrame) -> None:
    init = np.zeros((small_df.height, 3))
    with pytest.raises(pl.exceptions.ComputeError, match="width 2"):
        small_df.select(pl.col("emb").umap.fit_transform(init=init))


def test_custom_init_wrong_length(small_df: pl.DataFrame) -> None:
    init = np.zeros((small_df.height - 1, 2))
    with pytest.raises((pl.exceptions.ComputeError, pl.exceptions.ShapeError)):
        small_df.select(pl.col("emb").umap.fit_transform(init=init))


def test_custom_init_with_nan(small_df: pl.DataFrame) -> None:
    init = np.zeros((small_df.height, 2))
    init[0, 0] = np.nan
    with pytest.raises(pl.exceptions.ComputeError, match="init"):
        small_df.select(pl.col("emb").umap.fit_transform(init=init))
