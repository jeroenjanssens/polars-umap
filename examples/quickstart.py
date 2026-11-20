"""Quick start: embed the scikit-learn digits and plot them.

uv run --group examples python examples/quickstart.py
"""

import polars as pl
from _style import IMAGES, INK, NEUTRAL, SURFACE, bare, setup, title
from matplotlib import patheffects
from sklearn.datasets import load_digits

import polars_umap  # noqa: F401  (registers the `umap` namespace)

X, y = load_digits(return_X_y=True)
df = pl.DataFrame({"pixels": X, "digit": y})
print(df.head(3))

# Expression API: one Array(Float32, 2) column
df = df.with_columns(xy=pl.col("pixels").umap.fit_transform(random_state=42))
print(df.select("digit", "xy").head(3))

# DataFrame API: one column per component
df = df.umap.fit_transform("pixels", random_state=42)
print(df.select("digit", "umap_0", "umap_1").head(3))

# Plot: every digit forms its own cluster, labelled at its median position
fig, ax = setup()
ax.scatter(df["umap_0"], df["umap_1"], s=6, c=NEUTRAL, alpha=0.55, linewidths=0)
centers = df.group_by("digit").agg(pl.col("umap_0", "umap_1").median())
for digit, cx, cy in centers.iter_rows():
    ax.text(
        cx,
        cy,
        str(digit),
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=INK,
        path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)],
    )
bare(ax)
title(
    ax,
    "Handwritten digits, embedded with polars-umap",
    "1,797 images of 8x8 pixels, projected to 2D. Numbers mark each digit's cluster.",
)
fig.tight_layout()
fig.savefig(IMAGES / "digits.png", dpi=160, facecolor=SURFACE)
print("saved assets/images/digits.png")
