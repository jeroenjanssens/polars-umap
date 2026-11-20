"""Interactive exploration of n_neighbors and min_dist (marimo notebook).

uv run --group examples marimo edit examples/notebook.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl
    from sklearn.datasets import load_digits

    import polars_umap  # noqa: F401

    X, y = load_digits(return_X_y=True)
    df = pl.DataFrame({"pixels": X, "digit": y.astype(str)})
    return alt, df, mo, pl


@app.cell
def _(mo):
    n_neighbors = mo.ui.slider(2, 100, value=15, label="n_neighbors")
    min_dist = mo.ui.slider(0.0, 0.99, step=0.01, value=0.1, label="min_dist")
    layout = mo.ui.dropdown(["0.6", "0.5"], value="0.6", label="layout")
    mo.hstack([n_neighbors, min_dist, layout])
    return layout, min_dist, n_neighbors


@app.cell
def _(alt, df, layout, min_dist, n_neighbors):
    out = df.umap.fit_transform(
        "pixels",
        n_neighbors=n_neighbors.value,
        min_dist=min_dist.value,
        layout=layout.value,
        random_state=0,
    )
    base = alt.Chart(out.drop("pixels")).encode(
        x=alt.X("umap_0", axis=None), y=alt.Y("umap_1", axis=None)
    )
    points = base.mark_circle(size=12, opacity=0.5, color="#8d8b86").encode(tooltip=["digit"])
    labels = (
        alt.Chart(out.drop("pixels").group_by("digit").median())
        .mark_text(fontSize=16, fontWeight="bold", color="#0b0b0b")
        .encode(x="umap_0", y="umap_1", text="digit")
    )
    (points + labels).properties(width=600, height=500)
    return


if __name__ == "__main__":
    app.run()
