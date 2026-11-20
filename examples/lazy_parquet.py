"""Lazy pipeline: scan Parquet, filter, embed, write Parquet.

uv run --group examples python examples/lazy_parquet.py
"""

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import polars_umap  # noqa: F401

tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
n = 20_000
pl.DataFrame(
    {
        "id": np.arange(n),
        "lang": rng.choice(["en", "nl", "de"], n),
        "embedding": rng.normal(size=(n, 64)).astype(np.float32),
    }
).write_parquet(tmp / "embeddings.parquet")

query = (
    pl.scan_parquet(tmp / "embeddings.parquet")
    .filter(pl.col("lang") == "en")
    .umap.fit_transform("embedding", metric="cosine", random_state=0)
    .drop("embedding")
)
print(query.explain())
query.collect().write_parquet(tmp / "projected.parquet")
print(pl.read_parquet(tmp / "projected.parquet").head())
