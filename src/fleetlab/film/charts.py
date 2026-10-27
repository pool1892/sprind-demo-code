"""Charts for the story page, drawn at build time from the film's bound data.

Every chart is static HTML with inline SVG: no script, font or file from elsewhere, legible
on a phone and on a large screen. Text stays in HTML or in SVG at a fixed pixel size; only
the marks stretch (percentage coordinates, or a plot box drawn with non-scaling strokes), so
nothing shrinks to an unreadable size on a narrow screen. Each mark carries a `<title>` (the
browser shows it on hover), and each chart returns a plain table of its numbers beside it.

Colour follows the film: green filled for a completed job or a success, an open ring for a
failure (the shape carries the meaning, not the hue alone), indigo for the agent's
measurements, grey for what was commanded and for context.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chart",
    "DotRow",
    "Marks",
    "ProbeTrace",
    "Program",
    "Repair",
    "Result",
    "Robot",
    "dots",
    "exam",
    "fleet_bars",
    "fleet_result",
    "glance",
    "lags",
    "notebook",
    "probe",
    "repairs",
]


@dataclass
class Chart:
    """A chart's HTML, the table of its numbers (the chart's text equivalent), and the same
    numbers as plain values for the page's list of every number it shows."""

    html: str
    table: str
    values: list[dict[str, Any]] = field(default_factory=list)


def _e(value: object) -> str:
    return html.escape(str(value))


def _pct(k: int, n: int) -> float:
    return 100.0 * k / n if n else 0.0


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _table(head: Sequence[str], rows: Sequence[Sequence[object]], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    out = [f"<table{attr}><thead><tr>"]
    out += [f"<th>{_e(h)}</th>" for h in head]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ---- dots: one per try ------------------------------------------------------------------------


def dots(outcomes: Sequence[bool], titles: Sequence[str], *, pitch: int = 15) -> str:
    """A row of dots, one per try: filled when it succeeded, an open ring when it failed."""
    r = pitch * 0.34
    width, height = pitch * len(outcomes), pitch
    parts = [
        f'<svg class="dots" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'aria-hidden="true" focusable="false">'
    ]
    for i, (ok, title) in enumerate(zip(outcomes, titles, strict=True)):
        cx, cy = pitch * i + pitch / 2, pitch / 2
        if ok:
            parts.append(
                f'<circle class="ok" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}">'
                f"<title>{_e(title)}</title></circle>"
            )
        else:
            parts.append(
                f'<circle class="no" cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r - 1)}">'
                f"<title>{_e(title)}</title></circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _legend(*items: tuple[str, str]) -> str:
    keys = {
        "ok": '<svg width="12" height="12" aria-hidden="true"><circle class="ok" cx="6" cy="6" '
        'r="5"/></svg>',
        "no": '<svg width="12" height="12" aria-hidden="true"><circle class="no" cx="6" cy="6" '
        'r="4"/></svg>',
        "cell-ok": '<svg width="12" height="12" aria-hidden="true"><rect class="cell-ok" '
        'width="12" height="12" rx="2"/></svg>',
        "cell-new": '<svg width="12" height="12" aria-hidden="true"><rect class="cell-new" '
        'width="12" height="12" rx="2"/></svg>',
        "cell-no": '<svg width="12" height="12" aria-hidden="true"><rect class="cell-no" '
        'width="12" height="12" rx="2"/></svg>',
        "cell-lost": '<svg width="12" height="12" aria-hidden="true"><rect class="cell-lost" '
        'x="1" y="1" width="10" height="10" rx="2"/></svg>',
        "cmd": '<svg width="18" height="12" aria-hidden="true"><line class="cmd" x1="0" y1="6" '
        'x2="18" y2="6"/></svg>',
        "meas": '<svg width="18" height="12" aria-hidden="true"><line class="meas" x1="0" '
        'y1="6" x2="18" y2="6"/></svg>',
        "lag": '<svg width="14" height="12" aria-hidden="true"><rect class="lag" width="14" '
        'height="12"/></svg>',
        "ref": '<svg width="10" height="14" aria-hidden="true"><line class="ref" x1="5" y1="0" '
        'x2="5" y2="14"/></svg>',
        "bar": '<svg width="18" height="12" aria-hidden="true"><rect class="bar-fill" width="18" '
        'height="10" y="1" rx="2"/></svg>',
        "share": '<svg width="18" height="12" aria-hidden="true"><rect class="bar-ok" width="18" '
        'height="8" y="2" rx="4"/></svg>',
    }
    return (
        '<p class="legend">'
        + "".join(f'<span class="key">{keys[k]} {_e(text)}</span>' for k, text in items)
        + "</p>"
    )


# ---- experiment one: the exam and the lab notebook --------------------------------------------


@dataclass
class DotRow:
    label: str
    outcomes: list[bool]
    starts: list[int]
    role: str = ""  # "win" draws the row as the result


def exam(jobs: Sequence[tuple[str, list[DotRow], str]]) -> Chart:
    """The final exam: per job, the plain chain and the agent's job on the same locked layouts,
    one dot per layout, the same layout in the same column."""
    parts = ['<div class="exam">']
    rows_out: list[list[object]] = []
    values: list[dict[str, Any]] = []
    for title, rows, note in jobs:
        parts.append(f'<div class="exam-job"><p class="exam-title">{_e(title)}</p>')
        for row in rows:
            k, n = sum(row.outcomes), len(row.outcomes)
            values.append(
                {"job": title, "row": row.label, "k": k, "n": n, "layouts": row.starts,
                 "completed": row.outcomes}
            )  # fmt: skip
            titles = [
                f"{row.label}, final test layout {s}: {'completed' if ok else 'failed'}"
                for s, ok in zip(row.starts, row.outcomes, strict=True)
            ]
            parts.append(
                f'<div class="dot-row {_e(row.role)}">'
                f'<span class="row-label">{_e(row.label)}</span>'
                f"{dots(row.outcomes, titles)}"
                f'<span class="row-count"><strong>{k}</strong> of {n}</span></div>'
            )
        parts.append(f'<p class="row-note">{_e(note)}</p></div>')
        for i, start in enumerate(rows[0].starts):
            rows_out.append(
                [title, start] + ["completed" if r.outcomes[i] else "failed" for r in rows]
            )
    parts.append("</div>")
    parts.append(_legend(("ok", "completed"), ("no", "failed")))
    head = ["Job", "Final test layout"] + [r.label for r in jobs[0][1]]
    return Chart("".join(parts), _table(head, rows_out), values)


@dataclass
class Program:
    label: str
    outcomes: list[bool]
    starts: list[int]
    role: str  # plain, other, winner, loaded, confirm


def notebook(
    programs: Sequence[Program], *, muted: set[str] | None = None, summary: str = ""
) -> Chart:
    """The agent's lab notebook: one row per program in the order it ran them, one dot per
    practice layout; the confirmation on layouts never used before closes it. With `muted`
    roles and a `summary`, those rows stay out of the first view: it shows the rest, and the
    whole notebook, in order, opens beneath under `summary`."""
    muted = muted or set()

    def row_html(p: Program, dim: bool) -> str:
        k, n = sum(p.outcomes), len(p.outcomes)
        cls = ["dot-row", f"role-{p.role}"] + (["muted"] if dim else [])
        titles = [
            f"{p.label}, practice layout {s}: {'completed' if ok else 'failed'}"
            for s, ok in zip(p.starts, p.outcomes, strict=True)
        ]
        return (
            f'<div class="{" ".join(cls)}"><span class="row-label">{_e(p.label)}</span>'
            f"{dots(p.outcomes, titles)}"
            f'<span class="row-count"><strong>{k}</strong> of {n}</span></div>'
        )

    collapse = bool(muted and summary)
    shown = [p for p in programs if not (collapse and p.role in muted)]
    parts = ['<div class="notebook">']
    parts += [row_html(p, p.role in muted) for p in shown]
    parts.append("</div>")
    if collapse:
        parts.append(
            f'<details class="all-rows"><summary>{_e(summary)}</summary><div class="notebook">'
            + "".join(row_html(p, p.role in muted) for p in programs if p.role != "confirm")
            + "</div></details>"
        )
    parts.append(_legend(("ok", "job completed"), ("no", "failed")))
    rows: list[list[object]] = []
    values: list[dict[str, Any]] = []
    for p in programs:
        k, n = sum(p.outcomes), len(p.outcomes)
        rows.append([p.label, f"{k} of {n}", "".join("●" if o else "○" for o in p.outcomes)])
        values.append(
            {"program": p.label, "role": p.role, "k": k, "n": n, "layouts": p.starts,
             "completed": p.outcomes}
        )  # fmt: skip
    table = _table(["Program", "Completed", "Layout by layout"], rows)
    return Chart("".join(parts), table, values)


# ---- experiment two: the fleet, the probe, the repairs, the result -----------------------------


@dataclass
class Robot:
    site: str  # the site's label, e.g. "site 1"
    robot: str
    k: int
    n: int


def fleet_bars(robots: Sequence[Robot], ref: tuple[str, int, int]) -> Chart:
    """Success in the regular work, robot by robot and grouped by site, against a reference
    line at the healthy robots' rate."""
    ref_label, ref_k, ref_n = ref
    ref_x = _pct(ref_k, ref_n)
    parts = ['<div class="bars">']
    site = None
    for r in robots:
        if r.site != site:
            if site is not None:
                parts.append("</div>")
            parts.append(f'<div class="bar-group"><p class="bar-site">{_e(r.site)}</p>')
            site = r.site
        x = _pct(r.k, r.n)
        parts.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{_e(r.robot)}</span>'
            '<svg class="bar" width="100%" height="18" aria-hidden="true" focusable="false">'
            '<rect class="bar-track" x="0" y="3" width="100%" height="12" rx="3"/>'
            f'<rect class="bar-fill" x="0" y="3" width="{x:.2f}%" height="12" rx="3">'
            f"<title>{_e(r.robot)}, {_e(r.site)}: {r.k} of {r.n}</title></rect>"
            '<rect class="bar-fill" x="0" y="3" width="4" height="12"/>'
            f'<line class="ref" x1="{ref_x:.2f}%" y1="0" x2="{ref_x:.2f}%" y2="18"/>'
            "</svg>"
            f'<span class="bar-value"><strong>{r.k}</strong> of {r.n}</span>'
            "</div>"
        )
    parts.append("</div></div>")
    parts.append(
        _legend(
            ("bar", "successes in the regular work"),
            ("ref", f"{ref_label}: {ref_k} of {ref_n}"),
        )
    )
    rows = [[r.site, r.robot, f"{r.k} of {r.n}", f"{_pct(r.k, r.n):.0f}%"] for r in robots]
    values: list[dict[str, Any]] = [
        {"site": r.site, "robot": r.robot, "k": r.k, "n": r.n} for r in robots
    ]
    values.append({"reference": ref_label, "k": ref_k, "n": ref_n})
    return Chart("".join(parts), _table(["Site", "Robot", "Successes", "Share"], rows), values)


def lags(
    commanded: Sequence[float], measured: Sequence[float], *, moved_cm: float = 0.1
) -> list[tuple[int, int]]:
    """For each change of the commanded position: the control step it changed at, and how
    many control steps passed before the measured hand had moved 1 mm (`moved_cm`)."""
    out = []
    for i in range(1, len(commanded)):
        if commanded[i] == commanded[i - 1]:
            continue
        base = measured[i]
        lag = next(
            (j - i for j in range(i, len(measured)) if abs(measured[j] - base) >= moved_cm), None
        )
        if lag is None:
            raise ValueError(f"the hand never moved after the command at step {i}")
        out.append((i, lag))
    return out


@dataclass
class ProbeTrace:
    title: str  # e.g. "site 1 · r01"
    tag: str  # what the trace shows, in words
    commanded: list[float]
    measured: list[float]


def probe(
    traces: Sequence[ProbeTrace],
    step_s: float,
    *,
    axis_s: Sequence[float] = (),
    origin_cm: float = 0.0,
    tail_s: float = 1.5,
) -> Chart:
    """The agent's test motion: commanded against measured hand position over time, one panel
    per robot, each wait between a command and the hand's first movement shaded and timed.

    The time axis carries the ticks `axis_s` and ends at the first tick that lies `tail_s`
    after the last hand has settled (within 0.2 mm of where it ends), or at the recording's
    end; the flat remainder is not drawn. `origin_cm` is the commanded position the step
    starts from: a reference line beside the one at the step's far end."""
    n = max(len(t.commanded) for t in traces)
    settled = 0
    for t in traces:
        final = t.measured[-1]
        moving = [i for i, v in enumerate(t.measured) if abs(v - final) > 0.02]
        settled = max(settled, (moving[-1] + 1) if moving else 0)
    want = (settled + round(tail_s / step_s)) * step_s
    end_tick = next((a for a in sorted(axis_s) if a >= want - 1e-9), None)
    last = n - 1 if end_tick is None else min(n - 1, round(end_tick / step_s))
    lo = min(min(t.commanded[: last + 1] + t.measured[: last + 1]) for t in traces)
    hi = max(max(t.commanded[: last + 1] + t.measured[: last + 1]) for t in traces)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    seconds = last * step_s

    def x(i: float) -> float:
        return 1000.0 * i / last

    def y(v: float) -> float:
        return 100.0 * (hi - v) / (hi - lo)

    parts = ['<div class="probe">']
    rows: list[list[object]] = []
    values: list[dict[str, Any]] = []
    top = max(max(t.commanded) for t in traces)
    for t in traces:
        found = lags(t.commanded, t.measured)
        cmd = " ".join(f"{_fmt(x(i))},{_fmt(y(v))}" for i, v in enumerate(t.commanded[: last + 1]))
        meas = " ".join(f"{_fmt(x(i))},{_fmt(y(v))}" for i, v in enumerate(t.measured[: last + 1]))
        bands, notes = [], []
        for at, lag in found:
            w = max(x(at + lag) - x(at), 1.5)
            bands.append(
                f'<rect class="lag" x="{_fmt(x(at))}" y="0" width="{_fmt(w)}" height="100">'
                f"<title>{_e(t.title)}: moved {_fmt(lag * step_s)} s after the command</title>"
                "</rect>"
            )
            # the label sits left of the wait, half-way up, where neither line runs
            notes.append(
                f'<span class="lag-label" style="left:{100.0 * at / last:.2f}%">'
                f"{_fmt(lag * step_s)} s</span>"
            )
        row: list[object] = [t.title, t.tag]
        row += [
            f"{_fmt(lag * step_s)} s ({lag} control step{'' if lag == 1 else 's'})"
            for _, lag in found[:2]
        ]
        rows.append(row)
        values.append(
            {"robot": t.title, "shows": t.tag, "wait_steps": [lag for _, lag in found],
             "wait_s": [float(_fmt(lag * step_s)) for _, lag in found]}
        )  # fmt: skip
        grid = "".join(
            f'<line class="grid" x1="0" x2="1000" y1="{_fmt(y(v))}" y2="{_fmt(y(v))}"/>'
            for v in sorted({origin_cm, max(t.commanded)})
        )
        parts.append(
            '<div class="probe-panel">'
            f'<p class="probe-title"><strong>{_e(t.title)}</strong> <span>{_e(t.tag)}</span></p>'
            '<div class="probe-plot">'
            '<svg viewBox="0 0 1000 100" preserveAspectRatio="none" role="img" '
            f'aria-label="{_e(t.title)}: commanded and measured hand position">'
            f"{grid}{''.join(bands)}"
            f'<polyline class="cmd" points="{cmd}" vector-effect="non-scaling-stroke"/>'
            f'<polyline class="meas" points="{meas}" vector-effect="non-scaling-stroke"/>'
            "</svg>"
            f"{''.join(notes)}"
            "</div></div>"
        )
    ticks = []
    for a in sorted(axis_s):
        if a > seconds + 1e-9:
            break
        left = 100.0 * a / seconds
        cls = ' class="start"' if left < 4 else (' class="end"' if left > 96 else "")
        ticks.append(f'<span{cls} style="left:{left:.2f}%">{_fmt(a)} s</span>')
    parts.append(f'<div class="probe-axis" aria-hidden="true">{"".join(ticks)}</div>')
    parts.append("</div>")
    parts.append(
        _legend(
            ("cmd", f"commanded: a step of {_fmt(top - origin_cm)} cm out and back"),
            ("meas", "measured hand position"),
            ("lag", "wait before the hand moves"),
        )
    )
    table = _table(["Robot", "What it shows", "Wait, out", "Wait, back"], rows)
    return Chart("".join(parts), table, values)


@dataclass
class Repair:
    site: str
    change: str  # in plain words
    without: tuple[int, int]
    with_: tuple[int, int]
    rescued: int
    broken: int
    decision: str  # adopted, not adopted


def repairs(rows: Sequence[Repair]) -> Chart:
    """The tested repairs: each change against the same tasks and starting layouts without
    it."""
    out = [
        '<table class="repairs"><thead><tr><th scope="col">Repair tested</th>'
        '<th scope="col">Without it</th><th scope="col" class="dumbbell-col">'
        '<span class="sr">Change</span></th><th scope="col">With it</th>'
        '<th scope="col">Improved · worsened</th><th scope="col">Decision</th></tr></thead>'
        "<tbody>"
    ]
    site = None
    for r in rows:
        if r.site != site:
            out.append(
                f'<tr class="site-row"><th colspan="6" scope="rowgroup">{_e(r.site)}</th></tr>'
            )
            site = r.site
        a, b = _pct(*r.without), _pct(*r.with_)
        lo, hi = min(a, b), max(a, b)
        adopted = r.decision == "adopted"
        out.append(
            f'<tr class="{"adopted" if adopted else "declined"}">'
            f'<th scope="row">{_e(r.change)}</th>'
            f'<td class="num" data-label="without it">{r.without[0]} of {r.without[1]}</td>'
            '<td class="dumbbell-col"><svg class="dumbbell" width="100%" height="16" '
            'aria-hidden="true" focusable="false">'
            '<line class="track" x1="0" x2="100%" y1="8" y2="8"/>'
            f'<line class="span" x1="{lo:.2f}%" x2="{hi:.2f}%" y1="8" y2="8"/>'
            f'<circle class="before" cx="{a:.2f}%" cy="8" r="4"/>'
            f'<circle class="after" cx="{b:.2f}%" cy="8" r="5"/>'
            "</svg></td>"
            f'<td class="num" data-label="with it"><strong>{r.with_[0]} of {r.with_[1]}</strong>'
            "</td>"
            f'<td class="num" data-label="improved · worsened">{r.rescued} · {r.broken}</td>'
            f'<td class="decision">{_e(r.decision)}</td>'
            "</tr>"
        )
    out.append("</tbody></table>")
    values: list[dict[str, Any]] = [
        {"site": r.site, "change": r.change, "without": list(r.without), "with": list(r.with_),
         "rescued": r.rescued, "broken": r.broken, "decision": r.decision}
        for r in rows
    ]  # fmt: skip
    return Chart("".join(out), "", values)


def fleet_result(
    before: int, after: int, attempts: int, rescued: int, broken: int, labels: tuple[str, str]
) -> Chart:
    """The reserved attempts before and after, one square each, the same attempt in the same
    place in both grids, sorted by outcome: successes both times, then the broken, the
    rescued, and the attempts that failed both times."""
    both = before - broken
    neither = attempts - after - broken
    if both < 0 or neither < 0 or both + broken + rescued + neither != attempts:
        raise ValueError("the evaluation's counts do not add up")
    cols = 24 if attempts > 400 else 20
    rows_n = -(-attempts // cols)
    pitch, size = 10, 8
    width, height = cols * pitch - (pitch - size), rows_n * pitch - (pitch - size)
    order = ["both"] * both + ["broken"] * broken + ["rescued"] * rescued + ["neither"] * neither

    def grid(after_side: bool) -> str:
        cells = []
        for i, kind in enumerate(order):
            if after_side:
                cls = {"both": "cell-ok", "broken": "cell-lost", "rescued": "cell-new"}.get(
                    kind, "cell-no"
                )
            else:
                cls = "cell-ok" if kind in ("both", "broken") else "cell-no"
            cx, cy = (i % cols) * pitch, (i // cols) * pitch
            if cls == "cell-lost":
                cells.append(
                    f'<rect class="{cls}" x="{cx + 1}" y="{cy + 1}" width="{size - 2}" '
                    f'height="{size - 2}" rx="1.5"/>'
                )
            else:
                cells.append(
                    f'<rect class="{cls}" x="{cx}" y="{cy}" width="{size}" height="{size}" '
                    'rx="1.5"/>'
                )
        k = after if after_side else before
        return (
            f'<svg class="waffle" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{k} successes in {attempts} attempts">{"".join(cells)}</svg>'
        )

    parts = [
        '<div class="waffles">',
        f'<div class="waffle-side"><p class="waffle-label">{_e(labels[0])}</p>'
        f'<p class="waffle-count"><strong>{before}</strong> of {attempts}</p>{grid(False)}</div>',
        f'<div class="waffle-side"><p class="waffle-label">{_e(labels[1])}</p>'
        f'<p class="waffle-count"><strong>{after}</strong> of {attempts}</p>{grid(True)}</div>',
        "</div>",
        _legend(
            ("cell-ok", "success"),
            ("cell-new", f"improved: failed before, succeeded after ({rescued})"),
            ("cell-no", "failure"),
            ("cell-lost", f"worsened: succeeded before, failed after ({broken})"),
        ),
    ]
    table = _table(
        ["Attempts", "Success both times", "Improved", "Worsened", "Failed both times"],
        [[attempts, both, rescued, broken, neither]],
    )
    values: list[dict[str, Any]] = [
        {"attempts": attempts, "before": before, "after": after, "both": both,
         "rescued": rescued, "broken": broken, "neither": neither}
    ]  # fmt: skip
    return Chart("".join(parts), table, values)


# ---- the results at a glance ------------------------------------------------------------------
#
# One drawn image in two layouts, both in the page and one shown by the stylesheet: rows (one
# result per row, its marks at the right, drawn for the chart card's width on a wide screen) and
# stack (for phones). Unlike the charts above, the image scales as a whole, type included; each
# layout is drawn for the width range it is shown at, so its type stays near its drawn size.

_INK, _INK2, _INK3, _GREY = "#15171c", "#4f5562", "#5f6570", "#8b909a"
_OK, _BAD, _TRACK, _RULE = "#12a26c", "#e5484d", "#ece7dc", "#e6e0d3"
_FONT = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
# a character's advance in ems, close to Arial's (where a right-aligned number ends, and whether
# a line has room); San Francisco, the widest common system font, runs up to about 6% wider
_NARROW, _BROAD = frozenset("iljtfr.,:;'!|\u00b7 "), frozenset("mwMW")
_SLACK = 1.06


@dataclass
class Marks:
    """A row of marks under a result: a dot per locked layout, or a bar of k in n."""

    label: str
    outcomes: list[bool] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    k: int = 0
    n: int = 0

    @property
    def count(self) -> str:
        if self.outcomes:
            return f"{sum(self.outcomes)} of {len(self.outcomes)}"
        return f"{self.k} of {self.n}"


@dataclass
class Result:
    """One result: what it is, before and after, and the marks behind the two numbers."""

    kicker: str
    title: list[str]  # lines
    before: str
    after: str
    unit: str  # after the after number: "of 20", or ""
    caption: list[str]  # lines
    marks: list[Marks]


@dataclass
class _Type:
    kicker: float = 12.0
    title: float = 20.0
    title_lh: float = 25.0
    before: float = 40.0
    after: float = 72.0
    unit: float = 20.0
    caption: float = 15.0
    caption_lh: float = 21.0
    label: float = 13.0
    pitch: float = 14.4  # the drawn height of a row of marks
    bar: float = 10.0


def _width(text: str, size: float, weight: int = 400) -> float:
    ems = 0.0
    for ch in text:
        if ch.isdigit():
            ems += 0.56
        elif ch == "%":
            ems += 0.89
        elif ch in _NARROW:
            ems += 0.28
        elif ch in _BROAD:
            ems += 0.83
        else:
            ems += 0.67 if ch.isupper() else 0.54
    return ems * size * (1.0 if weight <= 500 else 1.04 if weight < 700 else 1.08)


def _room(width: float, room: float, what: str) -> None:
    """A line wider than its room, on the widest common system font, stops the build: a
    collision in the image is an error to fix in the words, not something to draw."""
    if width * _SLACK > room:
        raise ValueError(f"glance: {what!r} needs {width * _SLACK:.0f} of {room:.0f} units")


def _text(x: float, y: float, s: str, size: float, weight: int, fill: str, anchor: str = "",
          spacing: float = 0.0) -> str:  # fmt: skip
    a = f' text-anchor="{anchor}"' if anchor else ""
    ls = f' letter-spacing="{spacing:g}"' if spacing else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}" font-weight="{weight}" '
        f'fill="{fill}"{a}{ls}>{_e(s)}</text>'
    )


def _head(r: Result, x: float, y: float, w: float, t: _Type) -> tuple[str, float, float]:
    """Kicker, title, the before-and-after figure and its caption in the width w, from the
    kicker's baseline y; returns the drawing, its last baseline and the figure's middle."""
    kicker = r.kicker.upper()
    _room(_width(kicker, t.kicker, 700) + len(kicker) * t.kicker * 0.12, w, kicker)
    out = [_text(x, y, kicker, t.kicker, 700, _INK3, spacing=t.kicker * 0.12)]
    y += t.title * 1.55
    for line in r.title:
        _room(_width(line, t.title, 750), w, line)
        out.append(_text(x, y, line, t.title, 750, _INK))
        y += t.title_lh
    y += t.after * 0.98 - t.title_lh + 6
    # the before number in grey, then a drawn arrow at the middle of its figures, then the
    # after number; the arrow is drawn, not a glyph, so it sits at the same height everywhere
    end = x + _width(r.before, t.before, 650)
    gap = t.before * 0.3
    x1, x2 = end + gap, end + gap + t.before * 0.78
    ay, head = y - t.before * 0.36, t.before * 0.2
    out.append(_text(end, y, r.before, t.before, 650, _GREY, "end"))
    out.append(
        f'<path d="M{x1:.1f} {ay:.1f}H{x2:.1f}M{x2 - head:.1f} {ay - head:.1f}L{x2:.1f} '
        f'{ay:.1f}L{x2 - head:.1f} {ay + head:.1f}" fill="none" stroke="{_GREY}" '
        f'stroke-width="{max(2.0, t.before * 0.055):.1f}" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    )
    unit = (
        f'<tspan font-size="{t.unit:g}" font-weight="600" fill="{_INK2}" letter-spacing="0" '
        f'dx="{t.unit * 0.45:.1f}">{_e(r.unit)}</tspan>'
        if r.unit
        else ""
    )
    out.append(
        f'<text x="{x2 + gap:.1f}" y="{y:.1f}" font-size="{t.after:g}" font-weight="800" '
        f'fill="{_INK}" letter-spacing="{-t.after * 0.02:.1f}">{_e(r.after)}{unit}</text>'
    )
    figure = x2 + gap - x + _width(r.after, t.after, 800) * 0.98
    if r.unit:
        figure += t.unit * 0.45 + _width(r.unit, t.unit, 600)
    _room(figure, w, f"{r.before} to {r.after} {r.unit}".strip())
    mid = y - t.after * 0.36
    y += t.caption * 1.9
    for line in r.caption:
        _room(_width(line, t.caption), w, line)
        out.append(_text(x, y, line, t.caption, 400, _INK2))
        y += t.caption_lh
    return "".join(out), y - t.caption_lh, mid


def _marks(r: Result, x: float, y: float, w: float, t: _Type) -> tuple[str, float]:
    """The result's rows of marks, each under its label and count, from the top y; returns the
    drawing and its bottom."""
    out: list[str] = []
    for m in r.marks:
        _room(_width(m.label, t.label, 600) + 16 + _width(m.count, t.label, 600), w, m.label)
        out.append(_text(x, y + t.label, m.label, t.label, 600, _INK3))
        out.append(_text(x + w, y + t.label, m.count, t.label, 600, _INK3, "end"))
        cy = y + t.label * 1.75 + t.pitch * 0.4
        if m.outcomes:
            pitch = w / len(m.outcomes)
            radius = min(pitch * 0.36, 6.0)
            for i, ok in enumerate(m.outcomes):
                cx = x + pitch * (i + 0.5)
                verdict = "completed" if ok else "failed"
                tip = f"<title>{_e(m.label)}, final test layout {m.starts[i]}: {verdict}</title>"
                if ok:
                    out.append(
                        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{_OK}">'
                        f"{tip}</circle>"
                    )
                else:
                    out.append(
                        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius - 0.9:.1f}" '
                        f'fill="none" stroke="{_BAD}" stroke-width="1.7">{tip}</circle>'
                    )
        else:
            h, top = t.bar, cy - t.bar / 2
            tip = f"<title>{_e(m.label)}: {m.k} of {m.n} attempts succeeded</title>"
            out.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{w:.1f}" height="{h:g}" '
                f'rx="{h / 2:g}" fill="{_TRACK}"/>'
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{w * m.k / m.n:.1f}" height="{h:g}" '
                f'rx="{h / 2:g}" fill="{_OK}">{tip}</rect>'
            )
        y = cy + t.pitch * 0.5 + t.label * 0.9
    return "".join(out), y


def _image(w: float, h: float, body: str, layout: str, label: str) -> str:
    return (
        f'<svg class="glance-img glance-{layout}" viewBox="0 0 {w:g} {h:.0f}" role="img" '
        f'aria-label="{_e(label)}" font-family="{_FONT}">{body}</svg>'
    )


def _rule(x1: float, x2: float, y: float) -> str:
    return f'<line x1="{x1:g}" y1="{y:.1f}" x2="{x2:g}" y2="{y:.1f}" stroke="{_RULE}"/>'


def _rows(results: Sequence[Result], label: str) -> str:
    """One result per row, its marks at the right: drawn for the chart card's width on a wide
    screen, so it shows at its drawn size."""
    width, split = 726.0, 350.0
    t = _Type(kicker=13, before=36, after=64, unit=18, pitch=13.5)
    marks_h = len(results[0].marks) * (t.label * 1.75 + t.pitch * 0.9) + t.label * 0.9
    parts: list[str] = []
    y = 16.0
    for i, r in enumerate(results):
        if i:
            parts.append(_rule(0, width, y - 30))
        drawing, bottom, mid = _head(r, 0, y, split - 36, t)
        marks, marks_bottom = _marks(r, split, mid - marks_h / 2 + 4, width - split, t)
        parts += [drawing, marks]
        y = max(bottom, marks_bottom) + 56
    return _image(width, y - 46, "".join(parts), "rows", label)


def _stack(results: Sequence[Result], label: str) -> str:
    """The results one under another, for phones: drawn narrow, so that at 320 px its smallest
    type still renders at about 12 px."""
    width = 310.0
    t = _Type(kicker=13, title=18, title_lh=23, before=32, after=56, unit=16, caption_lh=20,
              label=14, pitch=14)  # fmt: skip
    parts: list[str] = []
    y = 16.0
    for i, r in enumerate(results):
        if i:
            parts.append(_rule(0, width, y - 30))
        drawing, bottom, _ = _head(r, 0, y, width, t)
        marks, marks_bottom = _marks(r, 0, bottom + 14, width, t)
        parts += [drawing, marks]
        y = marks_bottom + 50
    return _image(width, y - 42, "".join(parts), "stack", label)


def glance(results: Sequence[Result], legend: tuple[str, str, str]) -> Chart:
    """The results at a glance: per result its kicker and title, the number before and after
    the agent's work with a drawn arrow between them, and under them the marks those numbers
    count; one result per row on a wide screen, one under another on a phone. `legend` names a
    completed layout, a failed one and a bar's share."""
    if not results or any(not r.marks for r in results):
        raise ValueError("a glance needs results, each with its marks")
    label = "; ".join(
        f"{r.kicker}, {' '.join(r.title)}: from {r.before} to {r.after}"
        + (f" {r.unit}" if r.unit else "")
        + f", {' '.join(r.caption)}"
        for r in results
    )
    keys: list[tuple[str, str]] = []
    if any(m.outcomes for r in results for m in r.marks):
        keys += [("ok", legend[0]), ("no", legend[1])]
    if any(not m.outcomes for r in results for m in r.marks):
        keys.append(("share", legend[2]))
    html_out = (
        '<div class="glance">' + _rows(results, label) + _stack(results, label) + "</div>"
        + _legend(*keys)
    )  # fmt: skip
    rows: list[list[object]] = []
    values: list[dict[str, Any]] = []
    for r in results:
        name = f"{r.kicker}: {' '.join(r.title)}"
        for m in r.marks:
            rows.append([name, m.label, m.count])
        values.append(
            {"result": name, "before": r.before, "after": r.after, "unit": r.unit,
             "marks": [{"label": m.label, "count": m.count} for m in r.marks]}
        )  # fmt: skip
    return Chart(html_out, _table(["Result", "Marks", "Count"], rows), values)
