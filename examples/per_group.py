"""One UMAP per group, in a single query, with `.over()`.

uv run --group examples python examples/per_group.py
"""

import time

import numpy as np
import polars as pl
from sklearn.datasets import make_blobs

import polars_umap  # noqa: F401

# 50 "tenants", each with its own embedding space of 2,000 vectors.
frames = []
for tenant in range(50):
    X, _ = make_blobs(n_samples=2_000, n_features=32, centers=5, random_state=tenant)
    frames.append(pl.DataFrame({"tenant": tenant, "embedding": X.astype(np.float32)}))
df = pl.concat(frames)
print(df.shape)

t0 = time.perf_counter()
out = df.with_columns(
    xy=pl.col("embedding").umap.fit_transform(random_state=0).over("tenant"),
)
print(f".over(): {time.perf_counter() - t0:.2f}s for {df['tenant'].n_unique()} UMAPs")

# Equivalent with group_by + agg (one list of embeddings per tenant)
per_tenant = df.group_by("tenant", maintain_order=True).agg(
    n=pl.len(), xy=pl.col("embedding").umap.fit_transform(random_state=0)
)
print(per_tenant.head(3))

# Every group is embedded independently: same result as filtering first
one = df.filter(pl.col("tenant") == 7).select(
    pl.col("embedding").umap.fit_transform(random_state=0)
)
assert out.filter(pl.col("tenant") == 7)["xy"].to_list() == one["embedding"].to_list()
print("per-group results match individual fits")
