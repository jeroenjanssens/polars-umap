"""Text embeddings: TF-IDF + SVD vectors of newsgroup posts, projected with cosine UMAP.

Swap in any embedding model (sentence-transformers, OpenAI, ...) and the polars-umap part stays
the same: one Array column in, one Array column out.

    uv run --group examples python examples/text_embeddings.py
"""

import numpy as np
import polars as pl
from _style import IMAGES, INK_2, SERIES, SURFACE, bare, setup, title
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import polars_umap  # noqa: F401

CATEGORIES = ["rec.sport.hockey", "sci.space", "talk.religion.misc"]
news = fetch_20newsgroups(
    subset="all", categories=CATEGORIES, remove=("headers", "footers", "quotes")
)
tfidf = TfidfVectorizer(min_df=3, stop_words="english").fit_transform(news.data)
vectors = TruncatedSVD(n_components=100, random_state=0).fit_transform(tfidf).astype(np.float32)

# Very short posts get an all-zero vector, for which cosine distance is undefined.
# Make them null; null_policy="ignore" then leaves them out of the fit.
empty = np.linalg.norm(vectors, axis=1) == 0
df = pl.DataFrame(
    {
        "text": news.data,
        "topic": [CATEGORIES[i] for i in news.target],
        "embedding": vectors,  # Array(Float32, 100)
    }
).with_columns(pl.when(pl.Series(~empty)).then(pl.col("embedding")).alias("embedding"))

out = df.umap.fit_transform("embedding", metric="cosine", null_policy="ignore", random_state=0)
print(out.select("topic", "umap_0", "umap_1").head())

fig, ax = setup()
for color, topic in zip(SERIES, CATEGORIES):
    part = out.filter(pl.col("topic") == topic).drop_nulls("umap_0")
    ax.scatter(part["umap_0"], part["umap_1"], s=7, c=color, alpha=0.6, linewidths=0, label=topic)
leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1), markerscale=3, fontsize=9)
for text in leg.get_texts():
    text.set_color(INK_2)
bare(ax)
title(
    ax,
    "Newsgroup posts, cosine UMAP of TF-IDF vectors",
    f"{out.height:,} posts from three newsgroups",
)
fig.tight_layout()
fig.savefig(IMAGES / "newsgroups.png", dpi=160, facecolor=SURFACE)
print("saved docs/images/newsgroups.png")
