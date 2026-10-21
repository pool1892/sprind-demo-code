"""The exhibit's figures as PNG, in the dashboard's style (`fleetlab.dashboard.figures`).

Static images for the review site: one hue for a single series, the validated categorical order
for identities (sites, states), neutral grey for references, every number also in a table
beside the figure. A figure drawn from invented or toy records carries a large diagonal
mark saying so. Matplotlib's object interface with the Agg canvas; no date in the file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fleetlab.dashboard.data import Rate
from fleetlab.dashboard.figures import GRID, INK, INK_SECONDARY, REFERENCE, SERIES

__all__ = [
    "CATEGORICAL",
    "arms_figure",
    "pairs_figure",
    "probe_figure",
    "rate_dots",
    "series_by_group",
]

# The validated light-mode categorical order (dataviz palette, slots 1-5): blue, orange,
# aqua, yellow, magenta. Checked with validate_palette.js on 13 October 2026: all checks
# pass; contrast below 3:1 for slots 3-5, so every figure has its numbers in a table too.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKS = ("o", "s", "D", "^", "v")
WATERMARK = {"fixture": "FIXTURE: invented numbers", "toy": "TOY RUN: not robots"}


def _style(ax: Axes, title: str, xlabel: str, ylabel: str = "success") -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def _percent_axis(ax: Axes) -> None:
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])


def _mark(fig: Figure, origin: str) -> None:
    text = WATERMARK.get(origin)
    if text:
        fig.text(
            0.5,
            0.5,
            text,
            fontsize=26,
            color="#d03b3b",
            alpha=0.22,
            rotation=18,
            ha="center",
            va="center",
            weight="bold",
        )


def _save(fig: Figure, path: Path, origin: str) -> None:
    _mark(fig, origin)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, metadata={"Software": None}, facecolor="white")


def _bars(rate: Rate) -> tuple[float, float]:
    low, high = rate.interval
    return (max(0.0, rate.rate - low), max(0.0, high - rate.rate))


def rate_dots(
    path: Path,
    groups: Sequence[tuple[str, Rate]],
    title: str,
    xlabel: str,
    origin: str,
    overall: Rate | None = None,
    highlight: str | None = None,
) -> None:
    """One series: success per group with its 95% interval; the overall rate dashed.
    `highlight` draws that group in grey (a control or a baseline)."""
    width = min(10.0, max(4.0, 0.55 * len(groups) + 2.0))
    fig = Figure(figsize=(width, 2.9), layout="constrained")
    ax = fig.subplots()
    if overall is not None and overall.n:
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
    for x, (label, rate) in enumerate(groups):
        if not rate.n:
            continue
        low, high = _bars(rate)
        color = REFERENCE if label == highlight else SERIES
        ax.errorbar(
            [x],
            [rate.rate],
            yerr=[[low], [high]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=0,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{rate.successes}/{rate.n}",
            xy=(x, rate.rate),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
        )
    ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    _percent_axis(ax)
    _style(ax, title, xlabel)
    _save(fig, path, origin)


def series_by_group(
    path: Path,
    series: Mapping[str, Sequence[tuple[str, Rate]]],
    title: str,
    xlabel: str,
    origin: str,
) -> None:
    """Several series (sites over days, states over robots) in the fixed categorical order,
    each with its 95% intervals, slightly offset so the bars do not overlap; a legend and
    a direct label at each series' last point."""
    xs: list[str] = []
    for points in series.values():
        for x, _ in points:
            if x not in xs:
                xs.append(x)
    width = min(12.0, max(5.5, 0.6 * len(xs) + 4.0))
    fig = Figure(figsize=(width, 3.1), layout="constrained")
    ax = fig.subplots()
    names = list(series)
    spread = min(0.18, 0.7 / len(names)) if len(names) > 1 else 0.0
    for i, name in enumerate(names[: len(CATEGORICAL)]):
        color = CATEGORICAL[i]
        offset = (i - (len(names) - 1) / 2) * spread
        pts = [(xs.index(x) + offset, r) for x, r in series[name] if r.n]
        if not pts:
            continue
        ax.errorbar(
            [p for p, _ in pts],
            [r.rate for _, r in pts],
            yerr=[[_bars(r)[0] for _, r in pts], [_bars(r)[1] for _, r in pts]],
            fmt=MARKS[i],
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=name,
        )
    ax.set_xticks(range(len(xs)), xs, fontsize=8)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    _percent_axis(ax)
    _style(ax, title, xlabel)
    if len(names) > 1:
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path, origin)


def arms_figure(
    path: Path, panels: Sequence[tuple[str, Sequence[tuple[str, Rate]]]], origin: str
) -> None:
    """One small panel per experiment: each arm's success with its 95% interval; the
    control arm (first) in grey."""
    columns = min(3, max(1, len(panels)))
    rows = (len(panels) + columns - 1) // columns
    fig = Figure(figsize=(3.6 * columns, 2.9 * rows), layout="constrained")
    axes = fig.subplots(rows, columns, squeeze=False).ravel()
    for ax, (title, arms) in zip(axes, panels, strict=False):
        for x, (_, rate) in enumerate(arms):
            if not rate.n:
                continue
            low, high = _bars(rate)
            color = REFERENCE if x == 0 else SERIES
            ax.errorbar(
                [x],
                [rate.rate],
                yerr=[[low], [high]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=0,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            ax.annotate(
                f"{rate.successes}/{rate.n}",
                xy=(x, rate.rate),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=7,
                color=INK_SECONDARY,
            )
        ax.set_xticks(range(len(arms)), [label for label, _ in arms], fontsize=7)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        _percent_axis(ax)
        _style(ax, title, "arm")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    _save(fig, path, origin)


def pairs_figure(
    path: Path, pairs: Sequence[tuple[str, int, int]], title: str, origin: str
) -> None:
    """Attempts rescued (right, blue) and broken (left, orange) per pair of states."""
    fig = Figure(figsize=(7.0, 0.5 * len(pairs) + 1.3), layout="constrained")
    ax = fig.subplots()
    ys = list(range(len(pairs)))[::-1]
    rescued = [r for _, r, _ in pairs]
    broken = [-b for _, _, b in pairs]
    ax.barh(ys, rescued, color=CATEGORICAL[0], height=0.55, label="rescued")
    ax.barh(ys, broken, color=CATEGORICAL[1], height=0.55, label="broken")
    for y, (_, r, b) in zip(ys, pairs, strict=True):
        ax.annotate(
            f"{r} rescued",
            xy=(r, y),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
            va="center",
        )
        ax.annotate(
            f"{b} broken",
            xy=(-b, y),
            xytext=(-3, 0),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
            va="center",
            ha="right",
        )
    ax.axvline(0, color=REFERENCE, linewidth=1)
    ax.set_yticks(ys, [label for label, _, _ in pairs], fontsize=8)
    limit = max([1, *rescued, *[-b for b in broken]]) * 1.6
    ax.set_xlim(-limit, limit)
    _style(ax, title, "attempts (broken to the left, rescued to the right)", ylabel="")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _save(fig, path, origin)


ProbeRows = Sequence[Mapping[str, Any]]  # one probe episode's `probe_steps` rows, by step


def _first_moves(episodes: Sequence[ProbeRows]) -> str:
    """Rows from each commanded change to the first movement, per change, over episodes."""
    from fleetlab.loop.probes import step_lags

    per = [
        [e["first_movement_rows"] for e in step_lags([dict(r) for r in rows])] for rows in episodes
    ]
    changes = max((len(p) for p in per), default=0)
    parts = []
    for i in range(changes):
        values = ["never" if p[i] is None else str(p[i]) for p in per if i < len(p)]
        parts.append(values[0] if len(set(values)) == 1 else "/".join(values))
    return " | ".join(parts)


def probe_figure(
    path: Path,
    panels: Sequence[tuple[str, Sequence[ProbeRows]]],
    title: str,
    origin: str,
    axis: str = "x",
    reference: ProbeRows | None = None,
) -> None:
    """Commanded against measured hand position under a probe, one panel per robot or
    condition, side by side, on one scale: in each panel its probe episodes (`probe_steps`
    rows) as the offset of the measured hand from where it was in row 0, repetitions in the
    categorical order, over the commanded offset in dark grey; `reference`, one episode's
    rows (a healthy robot's), in light grey behind every panel. Millimetres against control
    rows; each panel's note gives the rows from each commanded change to the first movement
    (`loop.probes.step_lags`), one number where the repetitions agree."""
    index = "xyz".index(axis)

    def offsets(rows: ProbeRows) -> tuple[list[int], list[float], list[float]]:
        start = float(rows[0]["eef_pos"][index])
        steps = [int(r["step"]) for r in rows]
        measured = [(float(r["eef_pos"][index]) - start) * 1000 for r in rows]
        commanded = [
            float("nan")
            if r["target_pos"] is None
            else (float(r["target_pos"][index]) - start) * 1000
            for r in rows
        ]
        return steps, measured, commanded

    columns = min(4, max(1, len(panels)))
    rows_n = (len(panels) + columns - 1) // columns
    fig = Figure(figsize=(3.3 * columns, 2.6 * rows_n + 0.5), layout="constrained")
    axes = fig.subplots(rows_n, columns, squeeze=False, sharex=True, sharey=True).ravel()
    values: list[float] = []
    for _, episodes in panels:
        for rows in episodes:
            _, measured, commanded = offsets(rows)
            values += measured + [c for c in commanded if c == c]
    low, high = (min(values), max(values)) if values else (-1.0, 1.0)
    span = max(high - low, 1.0)
    for ax, (label, episodes) in zip(axes, panels, strict=False):
        if reference is not None:
            steps, measured, _ = offsets(reference)
            ax.plot(steps, measured, color=REFERENCE, linewidth=2, alpha=0.55, label="reference")
        if episodes:
            steps, _, commanded = offsets(episodes[0])
            ax.step(
                steps,
                commanded,
                where="post",
                color=INK_SECONDARY,
                linewidth=1.4,
                label="commanded",
            )
        for i, rows in enumerate(episodes[: len(CATEGORICAL)]):
            steps, measured, _ = offsets(rows)
            ax.plot(
                steps, measured, color=CATEGORICAL[i], linewidth=1.8, label=f"repetition {i + 1}"
            )
        ax.text(
            0.98,
            0.03,
            f"first move {_first_moves(episodes)} rows",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )
        _style(ax, label, "control row (50 ms)", ylabel=f"hand {axis} from start (mm)")
        ax.set_ylim(low - 0.45 * span, high + 0.12 * span)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1 : len(panels)]:
        for handle, text in zip(*ax.get_legend_handles_labels(), strict=True):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(6, len(labels)),
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(title, fontsize=10, color=INK)
    _save(fig, path, origin)
