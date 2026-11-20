"""Turn benchmark JSON lines into ``user_guide/benchmarks.qmd`` and figures.

uv run --group bench python benchmarks/report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
ROOT = HERE.parent
CONTENDERS = ["polars-umap", "polars-umap-gpu", "upstream", "upstream-gpu", "umap-learn"]
LABELS = {
    "polars-umap": "polars-umap",
    "polars-umap-gpu": "polars-umap (GPU)",
    "upstream": "upstream Rust binary",
    "upstream-gpu": "upstream Rust binary (GPU)",
    "umap-learn": "umap-learn",
}
COLORS = {  # fixed slot per contender, so identity never depends on which ones ran
    "polars-umap": "#2a78d6",
    "polars-umap-gpu": "#eb6834",
    "upstream": "#1baf7a",
    "upstream-gpu": "#1baf7a",
    "umap-learn": "#eda100",
}


def load(path: Path) -> pl.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        r.pop("machine", None)
    return pl.DataFrame(rows, infer_schema_length=None)


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    quality_cols = [
        c for c in ("trustworthiness", "knn_recall", "label_knn_accuracy") if c in df.columns
    ]
    agg = df.group_by("dataset", "n", "dim", "metric", "layout", "contender").agg(
        fit_s=pl.col("fit_s").median(),
        fit_min=pl.col("fit_s").min(),
        fit_max=pl.col("fit_s").max(),
        runs=pl.len(),
        peak_rss_mb=pl.col("peak_rss_mb").median(),
        jit_warmup_s=pl.col("jit_warmup_s").median()
        if "jit_warmup_s" in df.columns
        else pl.lit(None),
        import_s=pl.col("import_s").median() if "import_s" in df.columns else pl.lit(None),
    )
    quality = df.group_by("dataset", "layout", "contender").agg(
        pl.col(c).drop_nulls().first() for c in quality_cols
    )
    agg = agg.join(quality, on=["dataset", "layout", "contender"], how="left")
    ref = agg.filter(pl.col("contender") == "umap-learn").select(
        "dataset", "layout", ref_s=pl.col("fit_s")
    )
    order = {c: i for i, c in enumerate(CONTENDERS)}
    return (
        agg.join(ref, on=["dataset", "layout"], how="left")
        .with_columns(speedup=pl.col("ref_s") / pl.col("fit_s"))
        .with_columns(_o=pl.col("contender").replace_strict(order, default=99))
        .sort("n", "dim", "dataset", "layout", "_o", descending=[False, False, False, True, False])
        .drop("_o")
    )


def fmt(x: float | None, spec: str) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def table(s: pl.DataFrame) -> str:
    lines = [
        "| Dataset | Rows × dims | Layout | Implementation | Fit time (s) | vs umap-learn | Peak RSS (MB) "
        "| Trustworthiness | kNN recall | Label kNN acc. |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s.iter_rows(named=True):
        spread = f" ({r['fit_min']:.2f}–{r['fit_max']:.2f})" if r["runs"] > 1 else ""
        lines.append(
            f"| {r['dataset']} | {r['n']:,} × {r['dim']} | {r['layout']} | {LABELS[r['contender']]} "
            f"| {r['fit_s']:.2f}{spread} | {fmt(r['speedup'], '.1f')}× | {fmt(r['peak_rss_mb'], ',.0f')} "
            f"| {fmt(r.get('trustworthiness'), '.3f')} | {fmt(r.get('knn_recall'), '.3f')} "
            f"| {fmt(r.get('label_knn_accuracy'), '.3f')} |"
        )
    return "\n".join(lines)


def figure(s: pl.DataFrame, layout: str, out: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    s = s.filter(pl.col("layout") == layout)
    datasets = s.select("dataset", "n", "dim").unique().sort("n", "dim")["dataset"].to_list()
    cols = min(3, len(datasets))
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 2.3 * rows + 0.9), facecolor=surface, squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, datasets):
        ax.set_visible(True)
        part = s.filter(pl.col("dataset") == name)
        part = part.with_columns(
            _o=pl.col("contender").replace_strict({c: i for i, c in enumerate(CONTENDERS)})
        )
        part = part.sort("_o", descending=True)
        y = list(range(part.height))
        ax.set_facecolor(surface)
        ax.barh(
            y,
            part["fit_s"],
            color=[COLORS[c] for c in part["contender"]],
            height=0.62,
            edgecolor=surface,
            linewidth=2,
        )
        ax.set_yticks(y, [LABELS[c] for c in part["contender"]], fontsize=8.5, color=ink2)
        xmax = part["fit_s"].max()
        for yi, v in zip(y, part["fit_s"]):
            ax.text(v + xmax * 0.02, yi, f"{v:.2f}s", va="center", fontsize=8, color=ink)
        ax.set_xlim(0, xmax * 1.25)
        n, d = part["n"][0], part["dim"][0]
        ax.set_title(
            f"{name}  ({n:,} × {d})", loc="left", fontsize=9.5, color=ink, fontweight="bold"
        )
        ax.tick_params(axis="x", colors=ink2, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.suptitle(
        f"UMAP fit time, umap-learn {layout} pipeline (median, lower is better)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=ink,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=surface)
    plt.close(fig)


def groups_table(path: Path) -> str:
    if not path.exists():
        return "_Not run._"
    g = pl.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    wide = g.pivot(
        on="method", index=["rows", "dim", "groups"], values="seconds", aggregate_function="median"
    )
    methods = [
        c
        for c in ("polars-umap .over()", "polars-umap loop", "umap-learn loop")
        if c in wide.columns
    ]
    lines = [
        "| Rows × dims | Groups | " + " | ".join(methods) + " |",
        "|---|---:|" + "---:|" * len(methods),
    ]
    for r in wide.sort("groups").iter_rows(named=True):
        cells = [fmt(r[m], ".2f") + ("s" if r[m] is not None else "—") for m in methods]
        lines.append(
            f"| {r['rows']:,} × {r['dim']} | {r['groups']:,} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    results = HERE / "results" / "results.jsonl"
    raw = load(results)
    machine = json.loads(results.read_text().splitlines()[0]).get("machine", {})
    s = summarize(raw)
    images = ROOT / "assets" / "images"
    images.mkdir(parents=True, exist_ok=True)
    layouts = sorted(s["layout"].unique().to_list(), reverse=True)
    for layout in layouts:
        figure(s, layout, images / f"benchmark_fit_time_{layout.replace('.', '')}.png")

    machine_md = "\n".join(f"- **{k}**: {v}" for k, v in machine.items())
    figs = "\n\n".join(
        f"![Fit time, {lay} pipeline](../assets/images/benchmark_fit_time_{lay.replace('.', '')}.png)"
        for lay in layouts
    )
    template = (HERE / "report_template.md").read_text()
    md = (
        template.replace("{{MACHINE}}", machine_md)
        .replace("{{FIGURES}}", figs)
        .replace("{{TABLE}}", table(s))
        .replace("{{GROUPS}}", groups_table(HERE / "results" / "groups.jsonl"))
    )
    out = ROOT / "user_guide" / "benchmarks.qmd"
    out.write_text(md)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
