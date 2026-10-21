"""The README's tables, from the campaign's result files.

    run.sh tables.py > tables.md      (reads ~/.cache/robots-probes/teleop-absolute/results/)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path.home() / ".cache" / "robots-probes" / "teleop-absolute" / "results"
KINDS = ("displaced", "dropped", "on_side", "closed_empty", "held", "awkward")
# spikes/teleop-expert, relative control, MuJoCo 3.8.1: successes / counted episodes, both suites
EARLIER = {"displaced": (197, 200), "dropped": (196, 199), "on_side": (198, 200), "closed_empty": (196, 198),
           "held": (198, 198), "awkward": (192, 200), "inverted_spatial": (0, 100), "init": (1000, 1000)}
SHORT = {"libero_spatial": "spatial", "libero_object": "object"}


def load(name: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_meta(name: str) -> dict[str, Any]:
    path = RESULTS / f"{name}.run.json"
    return json.loads(path.read_text()) if path.exists() else {}


def counted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not str(r["status"]).startswith("skipped")]


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "-"
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * max(0, centre - half):.1f} to {100 * min(1, centre + half):.1f}%"


def steps_text(rows: list[dict[str, Any]]) -> str:
    s = [r["steps"] for r in rows if r["success"]]
    if not s:
        return "-"
    return f"{int(np.median(s))} / {int(np.percentile(s, 90))} / {max(s)}"


def reasons(rows: list[dict[str, Any]]) -> str:
    c = Counter(r["reason"] for r in rows if not r["success"])
    return ", ".join(f"{k} {v}" for k, v in c.most_common())


def line(label: list[str], rows: list[dict[str, Any]], skipped: int | None = None, bold: bool = False) -> str:
    ok = [r for r in rows if r["success"]]
    first = sum(1 for r in ok if r["attempts"] == 1)
    cells = [f"{len(ok)}/{len(rows)}", str(first), steps_text(rows), reasons(rows)]
    if skipped is not None:
        cells.insert(1, str(skipped))
    if bold:
        label = [f"**{x}**" if x else x for x in label]
        cells[0] = f"**{cells[0]}**"
    return "| " + " | ".join(label + cells) + " |"


def table_init() -> None:
    rows = load("c_init_abs")
    if not rows:
        return
    meta = run_meta("c_init_abs")
    print(f"### From the initial states, absolute control (`c_init_abs`, {meta.get('wall_s', '?')} s)\n")
    print("| suite | task | success | first grasp | steps median / 90% / max | failures |")
    print("|---|---|---|---|---|---|")
    for suite in ("libero_spatial", "libero_object"):
        rs = [r for r in rows if r["suite"] == suite]
        for task in range(10):
            print(line([SHORT[suite], str(task)], [r for r in rs if r["task"] == task]))
        print(line([f"{SHORT[suite]} total", ""], rs, bold=True))
    print(line(["both suites", ""], rows, bold=True))
    ok = sum(r["success"] for r in rows)
    print(f"\n95% interval for both suites: {wilson(ok, len(rows))}. "
          f"Hold commands (gripper 0) sent in {sum(1 for r in rows if r.get('hold_commands'))} episodes. "
          f"LIBERO's own check turned true a median of "
          f"{int(np.median([r['steps'] - r['libero_success_step'] for r in rows if r['success'] and r['libero_success_step'] is not None]))}"
          " steps before the expert finished.\n")


def table_modes() -> None:
    absolute = {(r["suite"], r["task"], r["init"]): r for r in load("c_init_abs") if r["init"] < 10}
    relative = {(r["suite"], r["task"], r["init"]): r for r in load("c_init_rel")}
    if not relative:
        return
    print("### Absolute against relative control, same start states (`c_init_abs` states 0 to 9, `c_init_rel`)\n")
    print("| suite | absolute | relative | both succeed | only absolute | only relative | both fail "
          "| same step count | steps median abs / rel |")
    print("|---|---|---|---|---|---|---|---|---|")
    for suite in ("libero_spatial", "libero_object", None):
        keys = [k for k in relative if (suite is None or k[0] == suite) and k in absolute]
        a = [absolute[k]["success"] for k in keys]
        r = [relative[k]["success"] for k in keys]
        both = sum(x and y for x, y in zip(a, r, strict=True))
        oa = sum(x and not y for x, y in zip(a, r, strict=True))
        orl = sum(y and not x for x, y in zip(a, r, strict=True))
        neither = sum(not x and not y for x, y in zip(a, r, strict=True))
        same_steps = sum(absolute[k]["steps"] == relative[k]["steps"] for k in keys)
        med_a = int(np.median([absolute[k]["steps"] for k in keys if absolute[k]["success"]]))
        med_r = int(np.median([relative[k]["steps"] for k in keys if relative[k]["success"]]))
        name = SHORT.get(suite, "**both**") if suite else "**both**"
        print(f"| {name} | {sum(a)}/{len(keys)} | {sum(r)}/{len(keys)} | {both} | {oa} | {orl} | {neither} "
              f"| {same_steps} | {med_a} / {med_r} |")
    print()


def table_kinds() -> None:
    print("### Take-over from disturbed states, through a snapshot, absolute control\n")
    print("| start state | suite | success | could not be set up | first grasp | steps median / 90% / max "
          "| failures | earlier (relative, MuJoCo 3.8.1) |")
    print("|---|---|---|---|---|---|---|---|")
    total_ok = total_n = 0
    for kind in (*KINDS, "inverted"):
        rows = load(f"c_{kind}")
        if not rows:
            continue
        for suite in ("libero_spatial", "libero_object"):
            rs = [r for r in rows if r["suite"] == suite]
            if not rs:
                continue
            c = counted(rs)
            print(line([kind, SHORT[suite]], c, skipped=len(rs) - len(c)) + " |")
        c = counted(rows)
        if kind == "inverted":
            k, n = EARLIER["inverted_spatial"]
            earlier = f"{k}/{n} (spatial)"
        else:
            k, n = EARLIER[kind]
            earlier = f"{k}/{n}"
            total_ok += sum(r["success"] for r in c)
            total_n += len(c)
        print(line([kind, "both"], c, skipped=len(rows) - len(c), bold=True) + f" {earlier} |")
    print(f"\nSix kinds without the upside-down bowl: {total_ok} of {total_n} "
          f"({100 * total_ok / max(total_n, 1):.1f}%, 95% interval {wilson(total_ok, total_n)}); "
          "earlier 1177 of 1195 (98.5%).\n")
    rows = load("c_awkward")
    if rows:
        print("Awkward arm poses by kind:\n")
        print("| arm pose | success | first grasp | steps median / 90% / max | failures |")
        print("|---|---|---|---|---|")
        for pose in ("tilted", "stretched", "pressing", "folded", "twisted"):
            rs = [r for r in counted(rows) if r.get("setup", {}).get("kind") == pose]
            print(line([pose], rs))
        print()


def xvla_group(row: dict[str, Any]) -> str:
    faults = row.get("extra", {}).get("faults") or []
    if not faults:
        return "no fault"
    f = faults[0]
    if f["kind"] == "scene_camera_pan":
        return f"camera turned {int(f['degrees'])} deg"
    if f["kind"] == "action_delay":
        return f"actions {int(f['steps'])} steps late"
    return f["kind"]


def handover_state(row: dict[str, Any]) -> str:
    s = row.get("setup", {})
    if all(s.get("touching", [False, False])):
        return "object between the fingers"
    if s.get("object_to_target_m", 1.0) < 0.06:
        return "object at the target, not counted"
    if abs(s.get("object_lean_deg", 0.0)) > 60:
        return "object tipped over"
    return "object elsewhere"


def table_xvla() -> None:
    end = load("c_xvla_end")
    if end:
        c = counted(end)
        print("### Take-over from X-VLA's recorded failures, after their 280 steps (`c_xvla_end`)\n")
        print("| X-VLA's episode | handed over | recovered | first grasp | steps median / 90% / max | failures |")
        print("|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for r in c:
            groups[xvla_group(r)].append(r)
        for g in sorted(groups):
            rs = groups[g]
            ok = [r for r in rs if r["success"]]
            print(f"| {g} | {len(rs)} | {len(ok)} | {sum(r['attempts'] == 1 for r in ok)} | {steps_text(rs)} "
                  f"| {reasons(rs)} |")
        ok = [r for r in c if r["success"]]
        print(f"| **all** | **{len(c)}** | **{len(ok)}** | {sum(r['attempts'] == 1 for r in ok)} | {steps_text(c)} "
              f"| {reasons(c)} |")
        print(f"\n{len(end) - len(c)} skipped ({Counter(r['status'] for r in end if r not in c)}). "
              f"95% interval: {wilson(len(ok), len(c))}. Suites: {Counter(r['suite'] for r in c)}.\n")
        print("What X-VLA had left at the hand-over:\n")
        print("| state at hand-over | episodes | recovered | failures |")
        print("|---|---|---|---|")
        groups = defaultdict(list)
        for r in c:
            groups[handover_state(r)].append(r)
        for g in sorted(groups):
            rs = groups[g]
            print(f"| {g} | {len(rs)} | {sum(r['success'] for r in rs)} | {reasons(rs)} |")
        print()
    for name, label in (("c_xvla_mid", "absolute"), ("c_xvla_mid_rel", "relative")):
        mid = load(name)
        if not mid:
            continue
        print(f"### Take-over in the middle of X-VLA's motion, expert in {label} control (`{name}`)\n")
        print("| suite | handed over | could not be set up | recovered | first grasp | steps median / 90% / max "
              "| failures |")
        print("|---|---|---|---|---|---|---|")
        for suite in ("libero_spatial", "libero_object", None):
            rs = [r for r in mid if suite is None or r["suite"] == suite]
            c = counted(rs)
            ok = [r for r in c if r["success"]]
            print(f"| {SHORT.get(suite, 'both') if suite else '**both**'} | {len(c)} | {len(rs) - len(c)} "
                  f"| {len(ok)} | {sum(r['attempts'] == 1 for r in ok)} | {steps_text(c)} | {reasons(c)} |")
        held = [r for r in counted(mid) if all(r.get("setup", {}).get("touching", [False, False]))]
        print(f"\nOf these, {len(held)} were handed over with the object between the fingers "
              f"({sum(r['success'] for r in held)} recovered).\n")


def table_failures() -> None:
    print("### Every failed take-over and initial-state episode\n")
    print("| batch | suite | task | start | details at hand-over | expert's status | last events |")
    print("|---|---|---|---|---|---|---|")
    for path in sorted(RESULTS.glob("c_*.jsonl")):
        if path.stem == "c_inverted":
            continue
        for r in counted(load(path.stem)):
            if r["success"]:
                continue
            s = r.get("setup", {})
            detail = ", ".join(f"{k} {s[k]}" for k in ("kind", "object_to_target_m", "object_lean_deg",
                                                         "gripper_gap_mm") if k in s)
            start = r["xvla_episode"] + f"@{r['handover_step']}" if r["scenario"] == "xvla" else str(r["init"])
            ev = " / ".join(e.split(": ", 1)[1][:70] for e in r.get("events", [])[-3:])
            print(f"| {path.stem[2:]} | {SHORT[r['suite']]} | {r['task']} | {start} | {detail} | {r['status']} | {ev} |")
    print()


def table_speed() -> None:
    print("### Run times\n")
    print("| batch | episodes | wall time | episodes per minute (4 workers) | median s per episode in a worker "
          "| load average at the end |")
    print("|---|---|---|---|---|---|")
    for path in sorted(RESULTS.glob("c_*.run.json")):
        meta = json.loads(path.read_text())
        rows = load(path.name[: -len(".run.json")])
        w = [r["wall_s"] for r in rows if "wall_s" in r]
        print(f"| {meta['name']} | {meta['episodes']} | {meta['wall_s']:.0f} s | "
              f"{60 * meta['episodes'] / meta['wall_s']:.0f} | {np.median(w):.1f} | {' '.join(meta['load_avg'])} |")
    print()


def main() -> None:
    table_init()
    table_modes()
    table_kinds()
    table_xvla()
    table_failures()
    table_speed()
    restores = [r.get("restore_exact") for p in RESULTS.glob("c_*.jsonl") for r in load(p.stem) if "restore_exact" in r]
    print(f"Snapshot restores: {sum(bool(x) for x in restores)} of {len(restores)} gave back the saved state hash.")
    harness = [r for p in RESULTS.glob("c_*.jsonl") for r in load(p.stem) if r["status"] == "fail:harness_exception"]
    print(f"Harness exceptions: {len(harness)}.")


if __name__ == "__main__":
    main()
