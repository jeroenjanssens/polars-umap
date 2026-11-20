"""Shapes, dtypes, input variants and the namespaces."""

from __future__ import annotations

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
from _helpers import to_numpy

import polars_umap as pu


def test_version() -> None:
    assert isinstance(pu.__version__, str)
    assert pu.__version__.count(".") == 2


@pytest.mark.parametrize("n_components", [1, 2, 3, 5, 10])
def test_output_dtype_and_length(small_df: pl.DataFrame, n_components: int) -> None:
    out = small_df.select(
        pl.col("emb").umap.fit_transform(n_components=n_components, random_state=0)
    )
    assert out.schema["emb"] == pl.Array(pl.Float32, n_components)
    assert out.height == small_df.height
    emb = to_numpy(out["emb"])
    assert emb.shape == (small_df.height, n_components)
    assert np.isfinite(emb).all()


def test_schema_is_known_lazily(small_df: pl.DataFrame) -> None:
    lf = small_df.lazy().with_columns(xy=pl.col("emb").umap.fit_transform(n_components=3))
    schema = lf.collect_schema() if hasattr(lf, "collect_schema") else lf.schema
    assert schema["xy"] == pl.Array(pl.Float32, 3)


def test_output_is_named_after_first_input(small_df: pl.DataFrame) -> None:
    assert small_df.select(pl.col("emb").umap.fit_transform()).columns == ["emb"]
    assert small_df.select(pu.umap("emb").alias("xy")).columns == ["xy"]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt8])
def test_array_inner_dtypes(dtype: pl.DataType) -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 50, size=(80, 6))
    df = pl.DataFrame({"emb": X}).with_columns(pl.col("emb").cast(pl.Array(dtype, 6)))
    out = df.select(pl.col("emb").umap.fit_transform(random_state=0))
    assert out.schema["emb"] == pl.Array(pl.Float32, 2)


def test_list_input_matches_array_input(small_df: pl.DataFrame) -> None:
    as_list = small_df.with_columns(pl.col("emb").arr.to_list())
    assert isinstance(as_list.schema["emb"], pl.List)
    a = small_df.select(pl.col("emb").umap.fit_transform(random_state=1))
    b = as_list.select(pl.col("emb").umap.fit_transform(random_state=1))
    np.testing.assert_array_equal(to_numpy(a["emb"]), to_numpy(b["emb"]))


def test_columns_input_matches_array_input(small_df: pl.DataFrame) -> None:
    wide = small_df.select(pl.col("emb").arr.get(i).alias(f"f{i}") for i in range(8))
    a = small_df.select(pl.col("emb").umap.fit_transform(random_state=1))
    b = wide.select(pu.umap(*[f"f{i}" for i in range(8)], random_state=1))
    c = wide.select(pu.umap([f"f{i}" for i in range(8)], random_state=1))
    d = wide.select(pu.umap(cs.starts_with("f"), random_state=1))
    e = wide.select(pu.umap(pl.col("^f.*$"), random_state=1))
    for other in (b, c, d, e):
        np.testing.assert_array_equal(to_numpy(a["emb"]), to_numpy(other.to_series()))


def test_single_numeric_column() -> None:
    df = pl.DataFrame({"x": np.linspace(0, 10, 50)})
    out = df.select(pu.umap("x", random_state=0))
    assert out.schema["x"] == pl.Array(pl.Float32, 2)


def test_float64_and_float32_give_same_result(small_df: pl.DataFrame) -> None:
    f64 = small_df.with_columns(pl.col("emb").cast(pl.Array(pl.Float64, 8)))
    a = small_df.select(pl.col("emb").umap.fit_transform(random_state=3))
    b = f64.select(pl.col("emb").umap.fit_transform(random_state=3))
    np.testing.assert_array_equal(to_numpy(a["emb"]), to_numpy(b["emb"]))


def test_chunked_input(small_df: pl.DataFrame) -> None:
    chunked = pl.concat([small_df[:40], small_df[40:]], rechunk=False)
    assert chunked["emb"].n_chunks() == 2
    a = small_df.select(pl.col("emb").umap.fit_transform(random_state=0))
    b = chunked.select(pl.col("emb").umap.fit_transform(random_state=0))
    np.testing.assert_array_equal(to_numpy(a["emb"]), to_numpy(b["emb"]))


def test_sliced_input(small_df: pl.DataFrame) -> None:
    sliced = small_df.slice(10, 80)
    a = sliced.select(pl.col("emb").umap.fit_transform(random_state=0))
    b = pl.DataFrame({"emb": sliced["emb"].to_list()}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 8)).umap.fit_transform(random_state=0)
    )
    np.testing.assert_array_equal(to_numpy(a["emb"]), to_numpy(b["emb"]))


def test_eager_equals_lazy(small_df: pl.DataFrame) -> None:
    expr = pl.col("emb").umap.fit_transform(random_state=0).alias("xy")
    eager = small_df.with_columns(expr)
    lazy = small_df.lazy().with_columns(expr).collect()
    np.testing.assert_array_equal(to_numpy(eager["xy"]), to_numpy(lazy["xy"]))


def test_dataframe_namespace_columns(small_df: pl.DataFrame) -> None:
    out = small_df.umap.fit_transform("emb", n_components=3, random_state=0)
    assert out.columns == ["emb", "g", "umap_0", "umap_1", "umap_2"]
    assert all(out.schema[f"umap_{i}"] == pl.Float32 for i in range(3))
    ref = to_numpy(
        small_df.select(pl.col("emb").umap.fit_transform(n_components=3, random_state=0))["emb"]
    )
    np.testing.assert_array_equal(out.select("umap_0", "umap_1", "umap_2").to_numpy(), ref)


def test_dataframe_namespace_array_and_prefix(small_df: pl.DataFrame) -> None:
    out = small_df.umap.fit_transform("emb", output="array", name="xy", random_state=0)
    assert out.schema["xy"] == pl.Array(pl.Float32, 2)
    out = small_df.umap.fit_transform("emb", prefix="dim", random_state=0)
    assert out.columns[-2:] == ["dim0", "dim1"]
    with pytest.raises(ValueError, match="output"):
        small_df.umap.fit_transform("emb", output="nope")


def test_lazyframe_namespace(small_df: pl.DataFrame) -> None:
    lf = small_df.lazy().umap.fit_transform("emb", random_state=0)
    assert isinstance(lf, pl.LazyFrame)
    eager = small_df.umap.fit_transform("emb", random_state=0)
    assert lf.collect().equals(eager)


def test_frame_knn_namespace(small_df: pl.DataFrame) -> None:
    out = small_df.umap.knn("emb", k=4)
    assert isinstance(out.schema["knn"], pl.Struct)
    assert small_df.lazy().umap.knn("emb", k=4, name="nn").collect().columns[-1] == "nn"


def test_works_inside_select_with_other_exprs(small_df: pl.DataFrame) -> None:
    out = small_df.select(
        pl.col("g"),
        pl.col("emb").umap.fit_transform(random_state=0).alias("xy"),
        pl.col("emb").umap.fit_transform(n_components=3, random_state=0).alias("xyz"),
    )
    assert out.columns == ["g", "xy", "xyz"]


def test_after_filter(small_df: pl.DataFrame) -> None:
    out = small_df.lazy().filter(pl.col("g") == 1).with_columns(xy=pu.umap("emb")).collect()
    assert out.height == 40
