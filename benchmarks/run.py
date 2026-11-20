"""End-to-end UMAP benchmark: polars-umap vs umap-learn vs the raw upstream Rust crate.

Every (dataset, contender, layout, repeat) runs in a fresh subprocess, so timings
include no warm caches from other runs and peak memory is per run. Quality is
measured in this process from the saved embeddings.

Examples
--------
    uv run --group bench python benchmarks/run.py --quick
    uv run --group bench python benchmarks/run.py \\
        --datasets blobs-10k-128,blobs-100k-128,mnist \\
        --contenders polars-umap,polars-umap-gpu,umap-learn \\
        --layouts 0.6,0.5 --repeats 3

Results are appended as JSON lines to ``benchmarks/results/results.jsonl``. Use
``report.py`` to turn them into tables and plots.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from datasets import load  # noqa: E402

QUALITY_SAMPLE = 5_000


def quality(X: np.ndarray, y: np.ndarray | None, emb: np.ndarray, metric: str) -> dict:
    """Trustworthiness, kNN recall (as in the upstream benchmark) and label accuracy."""
    from sklearn.manifold import trustworthiness
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(len(X), QUALITY_SAMPLE), replace=False)
    Xs, Es = X[idx], emb[idx]
    k = 15
    hi = (
        NearestNeighbors(n_neighbors=k + 1, metric=metric)
        .fit(Xs)
        .kneighbors(Xs, return_distance=False)
    )
    lo = NearestNeighbors(n_neighbors=k + 1).fit(Es).kneighbors(Es, return_distance=False)
    recall = np.mean([len(set(a[1:]) & set(b[1:])) / k for a, b in zip(hi, lo)])
    out = {
        "trustworthiness": float(trustworthiness(Xs, Es, n_neighbors=k, metric=metric)),
        "knn_recall": float(recall),
    }
    if y is not None:
        out["label_knn_accuracy"] = float(
            cross_val_score(KNeighborsClassifier(5), Es, y[idx], cv=3).mean()
        )
    return out


def machine() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import polars as pl

        import polars_umap

        info["polars"] = pl.__version__
        info["polars_umap"] = polars_umap.__version__
    except ImportError:
        pass
    try:
        from importlib.metadata import version

        info["umap_learn"] = version("umap-learn")
    except Exception:
        pass
    if sys.platform == "darwin":
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        info["cpu"] = r.stdout.strip()
    return info


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", default="digits,blobs-10k-128,blobs-50k-128")
    p.add_argument("--contenders", default="polars-umap,umap-learn")
    p.add_argument("--layouts", default="0.6")
    p.add_argument("--metric", default=None, help="override the dataset's default metric")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--umap-learn-seed",
        type=int,
        default=None,
        help="seed umap-learn too (forces it to run single-threaded)",
    )
    p.add_argument("--upstream-bin", default=None, help="path to the upstream umap-bench binary")
    p.add_argument("--out", default=str(HERE / "results" / "results.jsonl"))
    p.add_argument("--quick", action="store_true", help="small smoke run")
    p.add_argument("--no-quality", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.datasets, args.repeats = "digits,blobs-10k-32", 1

    datasets = args.datasets.split(",")
    contenders = args.contenders.split(",")
    layouts = args.layouts.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = machine()
    run_id = time.strftime("%Y%m%dT%H%M%S")

    for name in datasets:
        X, y, metric = load(name)
        metric = args.metric or metric
        print(f"== {name}: {X.shape[0]:,} x {X.shape[1]} ({metric})", flush=True)
        for layout in layouts:
            for contender in contenders:
                for rep in range(args.repeats):
                    with tempfile.TemporaryDirectory() as tmp:
                        spec = {
                            "dataset": name,
                            "contender": contender,
                            "layout": layout,
                            "metric": metric,
                            "seed": args.seed,
                            "umap_learn_seed": args.umap_learn_seed,
                            "upstream_bin": args.upstream_bin,
                            "embedding_out": str(Path(tmp) / "emb.npy"),
                        }
                        t0 = time.perf_counter()
                        proc = subprocess.run(
                            [sys.executable, str(HERE / "_worker.py"), json.dumps(spec)],
                            capture_output=True,
                            text=True,
                        )
                        wall = time.perf_counter() - t0
                        if proc.returncode != 0:
                            print(
                                f"   {contender:16} layout={layout} FAILED\n{proc.stderr[-2000:]}"
                            )
                            break
                        r = json.loads(proc.stdout.strip().splitlines()[-1])
                        emb = np.load(spec["embedding_out"])
                    q = {} if args.no_quality or rep > 0 else quality(X, y, emb, metric)
                    row = {
                        "run_id": run_id,
                        "dataset": name,
                        "n": int(X.shape[0]),
                        "dim": int(X.shape[1]),
                        "metric": metric,
                        "contender": contender,
                        "layout": layout,
                        "repeat": rep,
                        "process_wall_s": wall,
                        **r,
                        **q,
                        "machine": info,
                    }
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    extra = "".join(f" {k}={v:.3f}" for k, v in q.items())
                    print(
                        f"   {contender:16} layout={layout} rep={rep} fit={r['fit_s']:.2f}s "
                        f"rss={r.get('peak_rss_mb', 0):.0f}MB{extra}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
