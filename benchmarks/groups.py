"""Many-groups scenario: one UMAP per group.

Compares, for the same data:

- ``polars-umap .over()``: a single Polars query, groups run concurrently;
- ``polars-umap loop``: a Python loop calling polars-umap once per group;
- ``umap-learn loop``: the usual pandas/numpy workflow with a Python loop.

    uv run --group bench python benchmarks/groups.py --rows 200000 --groups 10,100,1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

import polars_umap  # noqa: F401

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--groups", default="10,100,1000")
    p.add_argument("--skip-umap-learn-above", type=int, default=100)
    p.add_argument("--out", default=str(HERE / "results" / "groups.jsonl"))
    args = p.parse_args()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(args.rows, args.dim)).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for n_groups in map(int, args.groups.split(",")):
        df = pl.DataFrame({"emb": X, "g": rng.integers(0, n_groups, args.rows)})
        results: dict[str, float] = {}

        t0 = time.perf_counter()
        df.with_columns(xy=pl.col("emb").umap.fit_transform(random_state=0).over("g"))
        results["polars-umap .over()"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _, part in df.group_by("g"):
            part.select(pl.col("emb").umap.fit_transform(random_state=0))
        results["polars-umap loop"] = time.perf_counter() - t0

        if n_groups <= args.skip_umap_learn_above:
            import umap

            umap.UMAP(n_epochs=5).fit_transform(X[:5_000])  # numba warm-up
            g = df["g"].to_numpy()
            t0 = time.perf_counter()
            for key in np.unique(g):
                umap.UMAP(compatibility_layout=False, optimizer="adam").fit_transform(X[g == key])
            results["umap-learn loop"] = time.perf_counter() - t0

        for name, secs in results.items():
            print(f"groups={n_groups:5}  {name:22} {secs:8.2f}s", flush=True)
            row = {
                "rows": args.rows,
                "dim": args.dim,
                "groups": n_groups,
                "method": name,
                "seconds": secs,
            }
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
