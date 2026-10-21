"""Figures as PNG: success rates with Wilson intervals, the dose-response, frame strips.

Matplotlib's object interface with the Agg canvas, no pyplot state. PNG files
are written without the "Software" text chunk, and matplotlib writes no date
into a PNG, so the same data gives the same bytes on the same installation.
Strips are plain pixel arrays written through imageio.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fleetlab.dashboard.data import DosePoint, Rate

__all__ = ["dose_figure", "frame_strip", "rate_figure"]

# One series per panel, so one hue; reference lines and bands in neutral grey.
SERIES = "#2a78d6"
REFERENCE = "#8a8984"
BAND = "#e5e4df"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#ecebe7"
STRIP_GAP = 4  # pixels of white between tiles and between the cameras' rows
STRIP_TILE = 240  # tiles larger than this are scaled down by an integer factor


def _style(ax: Axes, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("success", fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def _errorbars(ax: Axes, xs: Sequence[float], rates: Sequence[Rate]) -> None:
    centres = [r.rate for r in rates]
    # Wilson's bounds can miss 0% or 100% by a rounding error; errorbar refuses negatives.
    low = [max(0.0, r.rate - r.interval[0]) for r in rates]
    high = [max(0.0, r.interval[1] - r.rate) for r in rates]
    ax.errorbar(
        xs,
        centres,
        yerr=[low, high],
        fmt="o",
        color=SERIES,
        ecolor=SERIES,
        elinewidth=1.5,
        capsize=0,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )


def _save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=100, metadata={"Software": None}, facecolor="white")


def rate_figure(
    path: Path, groups: Sequence[tuple[Any, Rate]], overall: Rate, title: str, xlabel: str
) -> None:
    """Success per group with Wilson's 95% interval; the overall rate as a dashed line."""
    width = min(10.0, max(4.0, 0.45 * len(groups) + 2.0))
    fig = Figure(figsize=(width, 2.8), layout="constrained")
    ax = fig.subplots()
    xs = list(range(len(groups)))
    if overall.n:
        ax.axhline(overall.rate, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"all {100 * overall.rate:.0f}%",
            xy=(1.0, overall.rate),
            xycoords=("axes fraction", "data"),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
            ha="left",
        )
    _errorbars(ax, xs, [rate for _, rate in groups])
    ax.set_xticks(xs, [str(value) for value, _ in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _style(ax, title, xlabel)
    _save(fig, path)


def dose_figure(path: Path, points: Sequence[DosePoint]) -> None:
    """One panel per fault kind: success by level, the healthy episodes as a grey band."""
    healthy = next((p.rate for p in points if p.kind is None), None)
    kinds = sorted({p.kind for p in points if p.kind is not None})
    if not kinds:
        raise ValueError("no faulted calibration point")
    columns = min(3, len(kinds))
    rows = (len(kinds) + columns - 1) // columns
    fig = Figure(figsize=(3.6 * columns, 2.8 * rows), layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, columns, squeeze=False)).ravel()
    for ax, kind in zip(axes, kinds, strict=False):
        mine = [p for p in points if p.kind == kind]
        levels = [float(p.level or 0.0) for p in mine]
        if healthy is not None and healthy.n:
            low, high = healthy.interval
            ax.axhspan(low, high, color=BAND, linewidth=0)
            ax.axhline(healthy.rate, color=REFERENCE, linewidth=1)
            ax.annotate(
                f"healthy {100 * healthy.rate:.0f}%",
                xy=(0.0, healthy.rate),
                xycoords=("axes fraction", "data"),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=8,
                color=INK_SECONDARY,
                ha="left",
            )
        ax.plot(levels, [p.rate.rate for p in mine], color=SERIES, linewidth=1.5)
        _errorbars(ax, levels, [p.rate for p in mine])
        _style(ax, kind, "level")
    for ax in axes[len(kinds) :]:
        ax.set_visible(False)
    _save(fig, path)


def _shrink(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    factor = max(1, -(-max(image.shape[:2]) // STRIP_TILE))
    return image[::factor, ::factor]


def frame_strip(path: Path, rows: Sequence[Sequence[npt.NDArray[np.uint8]]]) -> None:
    """Tiles in a grid, one row per camera, white gaps between them; written as PNG."""
    if not rows or not rows[0]:
        raise ValueError("a strip needs at least one frame")
    tiles = [[_shrink(np.asarray(t, dtype=np.uint8)) for t in row] for row in rows]
    height = max(t.shape[0] for row in tiles for t in row)
    width = max(t.shape[1] for row in tiles for t in row)
    columns = max(len(row) for row in tiles)
    canvas = np.full(
        (
            len(tiles) * height + (len(tiles) - 1) * STRIP_GAP,
            columns * width + (columns - 1) * STRIP_GAP,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            y, x = r * (height + STRIP_GAP), c * (width + STRIP_GAP)
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile[..., :3]
    imageio.imwrite(path, canvas, format="PNG")  # type: ignore[call-overload]
