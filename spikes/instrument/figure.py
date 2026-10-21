"""The probe figure and table from `probe_shapes.py`'s results (no simulation).

    uv run --locked python spikes/instrument/figure.py PROBE_SHAPES.json FIGURE.png TABLE.md
    uv run --locked python spikes/instrument/figure.py --train TRAIN_SHAPES.json TABLE.md

Figure: one panel per condition, the commanded x offset (a step of +3 cm at row 10 and
back at row 50) and the measured x offset of the hand from its start position, the
three repetitions overlaid, with the healthy response as a muted reference.

Table: per condition, the response to each commanded step (first movement of 1 mm toward
it, half-way, 10 to 90 percent rise, all in control rows of 50 ms after the row the step
was issued in), where the hand stood against the commanded target at the end of each
segment (x and z, millimetres), and the alignment lag of `lag.sql` computed on the probe's
own rows. One number where the three repetitions agree, else all three.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REPEATS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1 to 3, validated all-pairs

ORDER = (
    "healthy",
    "remote_link_latency-17",
    "controller_overload-0.2",
    "arm_torque_loss-0.4",
    "controller_overload-0.1",
    "controller_overload-0.3",
    "arm_torque_loss-0.3",
    "arm_torque_loss-0.5",
    "camera_latency-24",
    "camera_pose_offset-6",
    "image_pipeline_release-crop20",
    "calibration_update",
)


def offsets(episode: dict[str, Any], axis: int) -> list[float]:
    eef = episode["series"]["eef"]
    return [(p[axis] - eef[0][axis]) * 1000 for p in eef]


def commanded(episode: dict[str, Any]) -> list[float | None]:
    tx = episode["series"]["target_x"]
    x0 = episode["series"]["eef"][0][0]
    return [None if t is None else (t - x0) * 1000 for t in tx]


def same(values: list[Any]) -> str:
    if all(v == values[0] for v in values):
        return "never" if values[0] is None else f"{values[0]:g}"
    return "/".join("never" if v is None else f"{v:g}" for v in values)


def table(results: dict[str, dict[str, Any]]) -> str:
    head = (
        "| Condition | Step out: first move, half-way, 10-90% rise (rows) "
        "| Step back: first move, half-way, 10-90% rise (rows) "
        "| Hand against target at end of step out: x, z (mm) "
        "| At end of probe: x, z (mm) | Alignment lag on the probe (rows) |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for name in ORDER:
        r = results[name]
        eps = r["episodes"]

        def edge(i: int, key: str, eps: list[dict[str, Any]] = eps) -> str:
            return same([e["edges"][i][key] for e in eps])

        def err(row: int, axis: int, eps: list[dict[str, Any]] = eps) -> str:
            # measured minus commanded target at `row` (the target of the command before it)
            vals = []
            for e in eps:
                target = commanded(e)[row - 1]
                assert target is not None
                measured = offsets(e, axis)[row]
                vals.append(round((measured - (target if axis == 0 else 0.0)), 1))
            return same(vals)

        lines.append(
            f"| {r['label']} "
            f"| {edge(0, 'first_movement_rows')}, {edge(0, 'half_way_rows')}, "
            f"{edge(0, 'rise_10_90_rows')} "
            f"| {edge(1, 'first_movement_rows')}, {edge(1, 'half_way_rows')}, "
            f"{edge(1, 'rise_10_90_rows')} "
            f"| {err(50, 0)}, {err(50, 2)} | {err(100, 0)}, {err(100, 2)} "
            f"| {same([e['alignment']['lag_rows'] for e in eps])} |"
        )
    return head + "\n".join(lines) + "\n"


def identical_to_healthy(results: dict[str, dict[str, Any]]) -> dict[str, bool]:
    base = [e["series"]["eef"] for e in results["healthy"]["episodes"]]
    return {
        name: [e["series"]["eef"] for e in r["episodes"]] == base
        for name, r in results.items()
        if name != "healthy"
    }


def figure(results: dict[str, dict[str, Any]], out: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    healthy = results["healthy"]["episodes"][0]
    rows = healthy["series"]["step"]
    for ax, name in zip(axes.flat, ORDER, strict=True):
        r = results[name]
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        cmd = commanded(r["episodes"][0])
        ax.step(
            rows,
            [c if c is not None else float("nan") for c in cmd],
            where="post",
            color=TEXT2,
            linewidth=1.5,
            label="commanded x",
        )
        if name != "healthy":
            ax.plot(
                rows,
                offsets(healthy, 0),
                color=AXIS,
                linewidth=2,
                solid_capstyle="round",
                label="healthy, measured",
            )
        series = [offsets(e, 0) for e in r["episodes"]]
        for i, values in enumerate(series):
            ax.plot(
                rows,
                values,
                color=REPEATS[i],
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=f"repetition {i + 1}, measured",
            )
        e1 = same([e["edges"][0]["first_movement_rows"] for e in r["episodes"]])
        e2 = same([e["edges"][1]["first_movement_rows"] for e in r["episodes"]])
        note = f"first move {e1} | {e2} rows"
        if all(s == series[0] for s in series):
            note += "\nrepetitions identical"
        ax.text(
            0.98,
            0.03,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=TEXT2,
        )
        ax.set_title(r["label"], fontsize=9.5, loc="left")
        ax.set_ylim(-40, 36)  # the band below -26 mm stays free for the note
        ax.set_yticks([-20, -10, 0, 10, 20, 30])
    for ax in axes[-1]:
        ax.set_xlabel("control row (50 ms each)")
    for ax in axes[:, 0]:
        ax.set_ylabel("hand x from start (mm)")
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Step probe on robot r01 (w10-v2): +3 cm on x at row 10, back at row 50; "
        "first move after each step in rows (step out | step back)",
        y=0.955,
        fontsize=10.5,
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=110, facecolor=SURFACE)


def train_table(results: list[dict[str, Any]]) -> str:
    """The step train (`probe_shapes.py --program train`): first movement after each of
    the four commanded steps, per repetition, and the alignment lag over the whole probe."""
    edges = [g["issued_at"] for g in results[0]["episodes"][0]["edges"]]
    head = (
        "| Condition | "
        + " | ".join(f"First move after the step at row {k} (rows)" for k in edges)
        + " | Alignment lag over the probe (rows) |\n|---|"
        + "---|" * (len(edges) + 1)
        + "\n"
    )
    lines = []
    for r in results:
        eps = r["episodes"]
        cells = [
            same([e["edges"][i]["first_movement_rows"] for e in eps]) for i in range(len(edges))
        ]
        lag = same([e["alignment"]["lag_rows"] for e in eps])
        lines.append(f"| {r['label']} | " + " | ".join(cells) + f" | {lag} |")
    return head + "\n".join(lines) + "\n"


def main() -> int:
    if sys.argv[1] == "--train":
        text = train_table(json.loads(Path(sys.argv[2]).read_text()))
        Path(sys.argv[3]).write_text(text)
        print(text)
        return 0
    source, png, md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    results = {r["condition"]: r for r in json.loads(source.read_text())}
    missing = [n for n in ORDER if n not in results]
    if missing:
        raise SystemExit(f"no results for {missing}")
    figure(results, png)
    text = table(results)
    same_as_healthy = identical_to_healthy(results)
    text += (
        "\nMeasured positions identical to the healthy probe, row for row: "
        + ", ".join(
            f"{results[n]['label']}: {'yes' if v else 'no'}" for n, v in same_as_healthy.items()
        )
        + ".\n"
    )
    text += (
        "chain_ok: "
        + ", ".join(f"{results[n]['label']}: {results[n]['chain_ok']}" for n in ORDER)
        + ".\n"
    )
    md.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
