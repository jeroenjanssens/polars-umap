"""Shared matplotlib styling for the example figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

IMAGES = Path(__file__).parent.parent / "docs" / "images"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8d8b86"


def setup(figsize: tuple[float, float] = (7, 6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    return fig, ax


def bare(ax: plt.Axes) -> None:
    """Embedding axes carry no meaning: hide them."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def title(ax: plt.Axes, text: str, subtitle: str | None = None) -> None:
    ax.set_title(
        text, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22 if subtitle else 10
    )
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5)
