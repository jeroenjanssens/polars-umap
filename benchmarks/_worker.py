"""Run one benchmark configuration in a fresh process and print a JSON result.

Invoked by ``run.py``; not meant to be used directly.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from datasets import load


def peak_rss_mb(who: int = resource.RUSAGE_SELF) -> float:
    rss = resource.getrusage(who).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def run_polars_umap(X: np.ndarray, metric: str, layout: str, gpu: bool, seed: int) -> dict:
    t0 = time.perf_counter()
    import polars as pl

    import polars_umap  # noqa: F401

    import_s = time.perf_counter() - t0
    df = pl.DataFrame({"emb": X})
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    out = df.select(
        pl.col("emb").umap.fit_transform(metric=metric, layout=layout, gpu=gpu, random_state=seed)
    )
    fit_s = time.perf_counter() - t0
    emb = out["emb"].to_numpy().reshape(len(X), -1)
    return {"import_s": import_s, "fit_s": fit_s, "rss_before_mb": rss_before, "embedding": emb}


def run_umap_learn(X: np.ndarray, metric: str, layout: str, seed: int | None) -> dict:
    t0 = time.perf_counter()
    import umap

    import_s = time.perf_counter() - t0
    kw: dict = {"metric": metric}
    if layout == "0.6":
        kw.update(compatibility_layout=False, optimizer="adam")
    else:
        kw.update(compatibility_layout=True)
    # numba compiles on first use; measure that separately. More than 4,096 rows also
    # triggers the PyNNDescent code paths (smaller inputs use exact neighbors).
    t0 = time.perf_counter()
    umap.UMAP(**kw, n_epochs=5).fit_transform(
        np.random.default_rng(0).normal(size=(5_000, X.shape[1]))
    )
    jit_s = time.perf_counter() - t0
    # A seed forces umap-learn to run single-threaded, so only pass one if asked.
    if seed is not None:
        kw["random_state"] = seed
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    emb = umap.UMAP(**kw).fit_transform(X)
    fit_s = time.perf_counter() - t0
    return {
        "import_s": import_s,
        "jit_warmup_s": jit_s,
        "fit_s": fit_s,
        "rss_before_mb": rss_before,
        "embedding": emb,
    }


def run_upstream(
    X: np.ndarray, metric: str, layout: str, gpu: bool, seed: int, binary: str
) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "meta.json").write_text(json.dumps({"n_points": len(X), "dim": X.shape[1]}))
        X.astype("<f4").tofile(d / "data.bin")
        cmd = [binary, str(d), metric, "--seed", str(seed), "--output", str(d / "r.json")]
        cmd += ["--compatibility-layout", "true" if layout == "0.5" else "false"]
        if layout == "0.6":
            cmd += ["--optimizer", "adam"]
        if gpu:
            cmd.append("--gpu")
        cmd += ["--embedding-path", str(d / "emb.bin")]
        subprocess.run(cmd, check=True, capture_output=True)
        result = json.loads((d / "r.json").read_text())
        emb = np.fromfile(d / "emb.bin", dtype="<f4").reshape(len(X), -1)
    return {"fit_s": float(result["time_s"]), "knn_s": float(result["knn_s"]), "embedding": emb}


def main() -> None:
    spec = json.loads(sys.argv[1])
    X, _, default_metric = load(spec["dataset"])
    metric = spec.get("metric") or default_metric
    contender = spec["contender"]
    if contender == "polars-umap":
        r = run_polars_umap(X, metric, spec["layout"], False, spec["seed"])
    elif contender == "polars-umap-gpu":
        r = run_polars_umap(X, metric, spec["layout"], True, spec["seed"])
    elif contender == "umap-learn":
        r = run_umap_learn(X, metric, spec["layout"], spec.get("umap_learn_seed"))
    elif contender in ("upstream", "upstream-gpu"):
        r = run_upstream(
            X,
            metric,
            spec["layout"],
            contender == "upstream-gpu",
            spec["seed"],
            spec["upstream_bin"],
        )
    else:
        msg = f"unknown contender {contender!r}"
        raise ValueError(msg)
    emb = r.pop("embedding")
    np.save(spec["embedding_out"], np.asarray(emb, dtype=np.float32))
    # The upstream binary runs as a child process; count its memory, not ours.
    r["peak_rss_mb"] = peak_rss_mb(
        resource.RUSAGE_CHILDREN if contender.startswith("upstream") else resource.RUSAGE_SELF
    )
    r["threads"] = os.cpu_count()
    print(json.dumps(r))


if __name__ == "__main__":
    main()
