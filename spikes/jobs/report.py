"""Tables and six-frame strips from the runner's results (light: CPU only).

    uv run --locked python spikes/jobs/report.py --runs DIR [DIR ...] --figures spikes/jobs/figures

Per job and variant: delivered of N (the host-fixed end check); per step, native
success reached during the step, native success at the end, and placed at the end
(native, robot clear of the step's object, the object at rest); held at the
switch; failure tags with counts; mean control steps and seconds per start. A
strip of six scene frames shows one run of the most common failure tag.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load(dirs: list[Path]) -> list[dict[str, Any]]:
    """Every run summary, the last one per (job, variant, start) winning, re-tagged from its
    trace with the runner's current `classify` (the runner's tag at run time is kept)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import classify

    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for d in dirs:
        for path in sorted(d.glob("*/*/s*/summary.json")):
            s = json.loads(path.read_text())
            s["dir"] = str(path.parent)
            s["tag_at_run"], s["reason_at_run"] = s["tag"], s["reason"]
            s["tag"], s["reason"] = classify(s, np.load(path.parent / "trace.npz"))
            runs[(s["job"], s["variant"], s["start"])] = s
    return list(runs.values())


def placed_at_end(run: dict[str, Any]) -> list[bool]:
    trace = np.load(Path(run["dir"]) / "trace.npz")
    return [bool(v) for v in trace["placed"][-1]]


def table(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    n_steps = len(runs[0]["steps"])
    out: dict[str, Any] = {
        "n": n,
        "starts": sorted(r["start"] for r in runs),
        "delivered": sum(r["delivered"] for r in runs),
        "steps": [],
        "tags": Counter(r["tag"] for r in runs),
        "control_steps_mean": float(np.mean([r["control_steps"] for r in runs])),
        "wall_s_mean": float(np.mean([r["timing"]["wall_s"] for r in runs])),
        "policy_call_ms_mean": 1000
        * float(np.mean([r["timing"]["policy_call_mean_s"] for r in runs])),
        "policy_calls_mean": float(np.mean([r["timing"]["policy_calls"] for r in runs])),
        "mismatches": sum(r["base_check_mismatches"] for r in runs),
    }
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import held_at_switch

    placed = [placed_at_end(r) for r in runs]
    traces = [np.load(Path(r["dir"]) / "trace.npz") for r in runs]
    for k in range(n_steps):
        recs = [r["steps"][k] for r in runs]
        # runs whose next instruction began after this step (held is measured there)
        switched = [
            (r, tr)
            for r, tr in zip(runs, traces, strict=True)
            if k + 1 < n_steps and r["steps"][k + 1].get("start_t") is not None
        ]
        out["steps"].append(
            {
                "stem": recs[0]["stem"],
                "native_reached": sum(x.get("native_first") is not None for x in recs),
                "native_end": sum(r["delivery"]["goals"][k] for r in runs),
                "placed_end": sum(p[k] for p in placed),
                "held_at_switch": sum(held_at_switch(r["steps"], tr, k + 1) for r, tr in switched),
                "switched": len(switched),
                "steps_mean": float(np.mean([x.get("steps", 0) for x in recs])),
            }
        )
    return out


def strip(run: dict[str, Any], path: Path) -> list[int]:
    """Six scene frames of one run: start, mid step 1, the switch, +20, the failure, the end."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    trace = np.load(Path(run["dir"]) / "trace.npz")
    native = trace["native"]
    total = run["control_steps"]
    s1 = run["steps"][0]
    switch = s1.get("end_t", total)
    fail = total
    for k in range(native.shape[1]):
        col = native[:, k]
        drops = [i + 1 for i in range(1, len(col)) if col[i - 1] and not col[i]]
        if drops and not run["delivery"]["goals"][k]:
            fail = min(fail, drops[-1] + 1)
    if fail == total and len(run["steps"]) > 1:
        later = run["steps"][1]
        if later.get("start_t") is not None:
            fail = (later["start_t"] + later.get("end_t", total)) // 2
    moments = [0, max(1, switch // 2), switch, min(total, switch + 20), fail, total]
    labels = ["start", "step 1", "switch", "switch+20", "failure", "end"]
    reader = imageio.get_reader(str(Path(run["dir"]) / "scene.mp4"))
    frames = {}
    for i, image in enumerate(reader):
        if i in moments:
            frames[i] = image
        if i > max(moments):
            break
    reader.close()
    tiles = []
    for t, label in zip(moments, labels, strict=True):
        image = frames.get(t, frames[max(k for k in frames if k <= t)])
        tile = Image.fromarray(np.ascontiguousarray(image[:, : image.shape[1] // 2])).resize(
            (240, 240)
        )
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, 240, 16], fill=(0, 0, 0))
        draw.text((4, 2), f"{label} t={t}", fill=(255, 255, 255))
        tiles.append(np.asarray(tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path, optimize=True)
    return moments


def write_csv(runs: list[dict[str, Any]], path: Path) -> None:
    """Every run: its start identity, per step native reached / native at end / placed at
    end / held when the next instruction began (1 or 0, empty for a step the job does not
    have), delivered, the tag and reason."""
    import csv
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import held_at_switch

    def key(r: dict[str, Any]) -> tuple[int, int, str, int]:
        job = r["job"]
        return ("SJK".index(job[0]), int(job[1:]), r["variant"], r["start"])

    fields = [
        "job",
        "variant",
        "start",
        "base_task",
        "base_bench_index",
        "init_state",
        "scene_seed",
        "noise_seed",
    ]
    for k in (1, 2, 3):
        fields += [
            f"step{k}_task",
            f"step{k}_outcome",
            f"step{k}_native_reached",
            f"step{k}_native_end",
            f"step{k}_placed_end",
            f"step{k}_held_at_switch",
        ]
    fields += ["delivered", "tag", "reason", "control_steps", "policy_calls", "wall_s"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(runs, key=key):
            row: dict[str, Any] = {k: r[k] for k in fields[:8]}
            placed = placed_at_end(r)
            trace = np.load(Path(r["dir"]) / "trace.npz")
            for k, rec in enumerate(r["steps"], start=1):
                row[f"step{k}_task"] = rec["task"]
                row[f"step{k}_outcome"] = rec["outcome"]
                row[f"step{k}_native_reached"] = int(rec.get("native_first") is not None)
                row[f"step{k}_native_end"] = int(r["delivery"]["goals"][k - 1])
                row[f"step{k}_placed_end"] = int(placed[k - 1])
                row[f"step{k}_held_at_switch"] = int(held_at_switch(r["steps"], trace, k))
            row.update(
                delivered=int(r["delivered"]),
                tag=r["tag"],
                reason=r["reason"],
                control_steps=r["control_steps"],
                policy_calls=r["timing"]["policy_calls"],
                wall_s=round(r["timing"]["wall_s"], 2),
            )
            w.writerow(row)


SHORT = {
    "put_the_cream_cheese_in_the_bowl": "cheese in bowl",
    "put_the_bowl_on_the_plate": "bowl on plate",
    "put_the_bowl_on_the_stove": "bowl on stove",
    "put_the_bowl_on_top_of_the_cabinet": "bowl on cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside": "bowl in top drawer",
    "push_the_plate_to_the_front_of_the_stove": "push plate",
    "turn_on_the_stove": "turn on stove",
    "open_the_middle_drawer_of_the_cabinet": "open middle drawer",
    "put_the_wine_bottle_on_the_rack": "wine on rack",
    "put_the_wine_bottle_on_top_of_the_cabinet": "wine on cabinet",
}


def flips(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each variant against naive on the starts both ran: rescued and broken starts."""
    by: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for r in runs:
        by[(r["job"], r["variant"])][r["start"]] = bool(r["delivered"])
    out = []

    def order(item: tuple[tuple[str, str], dict[int, bool]]) -> tuple[int, int, str]:
        (job, variant), _ = item
        return ("SJK".index(job[0]), int(job[1:]), variant)

    for (job, variant), res in sorted(by.items(), key=order):
        naive = by.get((job, "naive"))
        if variant == "naive" or naive is None:
            continue
        common = sorted(set(res) & set(naive))
        out.append(
            {
                "job": job,
                "variant": variant,
                "starts": len(common),
                "naive": sum(naive[s] for s in common),
                "variant_delivered": sum(res[s] for s in common),
                "rescued": [s for s in common if res[s] and not naive[s]],
                "broken": [s for s in common if naive[s] and not res[s]],
            }
        )
    return out


def throughput(runs: list[dict[str, Any]]) -> str:
    """Seconds per chained start and per policy call, with the machine's load, as Markdown."""
    chained = [r for r in runs if not r["job"].startswith("S")]
    wall = np.array([r["timing"]["wall_s"] for r in chained])
    calls = np.array([r["timing"]["policy_calls"] for r in chained])
    call_ms = 1000 * np.array([r["timing"]["policy_call_mean_s"] for r in chained])
    steps = np.array([r["control_steps"] for r in chained])
    sim_ms = 1000 * np.array([r["timing"]["sim_s"] / r["control_steps"] for r in chained])
    policy_share = sum(r["timing"]["policy_s"] for r in chained) / wall.sum()
    load = np.array([r["timing"]["load_start"] for r in chained])
    q = np.percentile
    return (
        "\n".join(
            [
                "| Quantity, over the chained runs | Value |",
                "|---|---|",
                f"| chained runs | {len(chained)} |",
                "| seconds per chained start (wall clock, video included) | "
                f"mean {wall.mean():.1f}, "
                f"median {np.median(wall):.1f}, 5th to 95th percentile {q(wall, 5):.1f} to "
                f"{q(wall, 95):.1f}, at most {wall.max():.1f} |",
                f"| milliseconds per policy call (batch 1, fp16_res32; a run's mean) | mean "
                f"{call_ms.mean():.0f}, median {np.median(call_ms):.0f}, 95th percentile "
                f"{q(call_ms, 95):.0f}, at most {call_ms.max():.0f} |",
                f"| policy calls per start | mean {calls.mean():.1f}, at most {calls.max()} |",
                f"| control steps per start | mean {steps.mean():.0f}, at most {steps.max()} |",
                f"| milliseconds of simulation per control step (physics, two cameras at 360 "
                f"pixels, privileged state, every goal) | mean {sim_ms.mean():.1f} |",
                f"| share of wall clock in policy calls | {policy_share:.0%} |",
                f"| one-minute load average at run starts | {load.min():.1f} to {load.max():.1f}, "
                f"median {np.median(load):.1f} |",
            ]
        )
        + "\n"
    )


def flips_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Job | Variant | Same starts | Naive delivered | Variant delivered | Rescued | Broken |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in rows:
        rescued = ", ".join(map(str, f["rescued"])) or "none"
        broken = ", ".join(map(str, f["broken"])) or "none"
        lines.append(
            f"| {f['job']} | {f['variant']} | {f['starts']} | {f['naive']} | "
            f"{f['variant_delivered']} | {len(f['rescued'])} ({rescued}) | "
            f"{len(f['broken'])} ({broken}) |"
        )
    return "\n".join(lines) + "\n"


def markdown(summary: dict[str, Any]) -> str:
    """One row per job and variant. Per step: native success reached during the step /
    native success at the end / placed at the end, each of N."""
    lines = [
        "| Job | Variant | N | Delivered | Per step: native reached / native at end / "
        "placed at end | Held at switch | Tags | Steps | s per start |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, t in summary.items():
        job, variant = key.split("/")
        steps = "; ".join(
            f"{SHORT.get(s['stem'], s['stem'])} {s['native_reached']} / {s['native_end']} / "
            f"{s['placed_end']}"
            for s in t["steps"]
        )
        held = ", ".join(f"{s['held_at_switch']}/{s['switched']}" for s in t["steps"][:-1]) or "-"
        tags = ", ".join(f"{k} {v}" for k, v in sorted(t["tags"].items(), key=lambda kv: -kv[1]))
        lines.append(
            f"| {job} | {variant} | {t['n']} | **{t['delivered']}** | {steps} | {held} | {tags} "
            f"| {t['control_steps_mean']:.0f} | {t['wall_s_mean']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path, help="one row per run: identity, outcomes, tag")
    parser.add_argument("--markdown", type=Path, help="the tables as Markdown")
    parser.add_argument(
        "--strips", nargs="*", default=["naive"], help="job/variant keys or variants to strip"
    )
    args = parser.parse_args()
    runs = load(args.runs)
    if args.csv is not None:
        write_csv(runs, args.csv)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["job"], r["variant"])].append(r)

    def order(key: tuple[str, str]) -> tuple[int, int, int, str]:
        job, variant = key
        return ("SJK".index(job[0]), int(job[1:]), 0 if variant == "naive" else 1, variant)

    summary = {}
    for key in sorted(groups, key=order):
        job, variant = key
        rs = sorted(groups[key], key=lambda r: r["start"])
        t = table(rs)
        summary[f"{job}/{variant}"] = t
        steps = "; ".join(
            f"{s['stem']}: native reached {s['native_reached']}, native at end {s['native_end']}, "
            f"placed at end {s['placed_end']}, held at switch {s['held_at_switch']}/{s['switched']}"
            for s in t["steps"]
        )
        tags = ", ".join(f"{k} {v}" for k, v in t["tags"].most_common())
        print(
            f"{job} {variant}: delivered {t['delivered']}/{t['n']} | {steps} | {tags} | "
            f"steps {t['control_steps_mean']:.0f} | {t['wall_s_mean']:.1f} s | "
            f"{t['policy_calls_mean']:.1f} calls x {t['policy_call_ms_mean']:.0f} ms | "
            f"mismatches {t['mismatches']}"
        )
        wanted = variant in args.strips or f"{job}/{variant}" in args.strips
        if args.figures is not None and not job.startswith("S") and wanted:
            failures = [r for r in rs if not r["delivered"]]
            if failures:
                common = Counter(r["tag"] for r in failures).most_common(1)[0][0]
                pick = next(r for r in failures if r["tag"] == common)
                moments = strip(pick, args.figures / f"strip_{job}_{variant}.png")
                t["strip"] = {
                    "start": pick["start"],
                    "tag": common,
                    "reason": pick["reason"],
                    "moments": moments,
                }
                print(f"  strip: s{pick['start']:02d} {common}: {pick['reason']} {moments}")
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1, default=dict))
    if args.markdown is not None:
        args.markdown.write_text(
            markdown(summary) + "\n" + flips_markdown(flips(runs)) + "\n" + throughput(runs)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
