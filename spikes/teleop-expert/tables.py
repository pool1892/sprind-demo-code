"""Markdown tables for README.md from the campaign's result files.

    run.sh tables.py            (reads ~/.cache/robots-probes/teleop/results/final_*.jsonl)
"""
import json
from collections import Counter, defaultdict
from statistics import median

from common import OUT

R = OUT / "results"
SCENARIOS = ["displaced", "dropped", "on_side", "inverted", "closed_empty", "held", "awkward"]
FAULTS = ["friction_0.5", "friction_0.1", "friction_0.02", "mass_10", "mass_50", "mass_200", "friction_0.1+mass_50"]
SUITES = ["libero_spatial", "libero_object"]


def load(name):
    p = R / f"{name}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0


def cell(rows):
    """'success/valid (first try), median steps' for a list of episodes."""
    valid = [r for r in rows if not r["status"].startswith("skipped")]
    ok = [r for r in valid if r["success"]]
    if not valid:
        return "-", "-", "-", "-"
    steps = [r["steps"] for r in ok]
    reasons = Counter(r["reason"] for r in valid if not r["success"])
    return (f"{len(ok)}/{len(valid)}", str(sum(1 for r in ok if r["attempts"] == 1)),
            f"{int(median(steps))} / {pct(steps, 0.9)} / {max(steps)}" if steps else "-",
            ", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "")


def table(header, lines):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for line in lines:
        print("| " + " | ".join(str(x) for x in line) + " |")
    print()


print("### From the initial states (final_init)\n")
rows = load("final_init")
lines = []
for s in SUITES:
    for t in range(10):
        rs = [r for r in rows if r["suite"] == s and r["task"] == t]
        if rs:
            lines.append([s.replace("libero_", ""), t, rs[0]["language"], *cell(rs)])
    lines.append([f"**{s.replace('libero_', '')} total**", "", "", *[f"**{c}**" for c in cell([r for r in rows if r['suite'] == s])]])
lines.append(["**both suites**", "", "", *[f"**{c}**" for c in cell(rows)]])
table(["suite", "task", "instruction", "success", "of which first grasp", "steps median / 90% / max", "failures"], lines)

print("### Take-over from disturbed states\n")
lines = []
for sc in SCENARIOS:
    rows = load(f"final_{sc}")
    skipped = sum(1 for r in rows if r["status"].startswith("skipped"))
    for s in SUITES:
        lines.append([sc, s.replace("libero_", ""), *cell([r for r in rows if r["suite"] == s])])
    lines.append([f"**{sc}**", "**both**", *[f"**{c}**" for c in cell(rows)][:3], f"{skipped} states could not be set up" if skipped else ""])
table(["start state", "suite", "success", "of which first grasp", "steps median / 90% / max", "failures"], lines)

print("### How the object actually lay after 'on_side' and 'inverted'\n")
lines = []
for sc in ("on_side", "inverted", "dropped"):
    rows = [r for r in load(f"final_{sc}") if not r["status"].startswith("skipped")]
    for s in SUITES:
        groups = defaultdict(list)
        for r in rows:
            if r["suite"] != s:
                continue
            lean = r["setup"].get("object_lean_deg", 0)
            groups["upright (< 30 deg)" if lean < 30 else "upside down (> 150 deg)" if lean > 150 else "on its side"].append(r)
        for k in ("upright (< 30 deg)", "on its side", "upside down (> 150 deg)"):
            if groups[k]:
                lines.append([sc, s.replace("libero_", ""), k, *cell(groups[k])[:1], cell(groups[k])[3]])
table(["start state", "suite", "object came to rest", "success", "failures"], lines)

print("### 'awkward' by kind of arm pose\n")
rows = [r for r in load("final_awkward") if not r["status"].startswith("skipped")]
lines = []
for k in ("tilted", "stretched", "pressing", "folded", "twisted"):
    rs = [r for r in rows if r["setup"].get("kind") == k]
    lines.append([k, *cell(rs)])
table(["arm pose", "success", "of which first grasp", "steps median / 90% / max", "failures"], lines)

print("### Faulted gripper and heavier object, from the initial states\n")
lines = [["none (first 10 states of final_init)", "", *cell([r for r in load("final_init") if r["init"] < 10])]]
for f in FAULTS:
    rows = load(f"final_fault_{f}")
    info = rows[0].get("fault_info", {}) if rows else {}
    masses = sorted(r["fault_info"].get("object_mass_g", 0) for r in rows if r.get("fault_info"))
    desc = []
    if "pad_friction" in info:
        desc.append(f"pad friction {info['pad_friction']} (normal 2.0)")
    if masses and masses[-1] > 0:
        desc.append(f"object mass {masses[0]:.0f} to {masses[-1]:.0f} g")
    for s in SUITES:
        lines.append([f, s.replace("libero_", ""), *cell([r for r in rows if r["suite"] == s])])
    lines.append([f"**{f}**", "; ".join(desc), *[f"**{c}**" for c in cell(rows)]])
table(["fault", "suite / meaning", "success", "of which first grasp", "steps median / 90% / max", "failures"], lines)

print("### Why take-overs failed (all kinds except the upside-down bowl)\n")


def cause(r):
    ev = " | ".join(r.get("events", []))
    if "robot0_link" in ev and "wooden_cabinet" in ev:
        return "forearm runs into the cabinet (object behind it)"
    if "no top-down grasp" in ev:
        return "planner found no collision-free grasp"
    if r["scenario"] == "awkward":
        return "arm kept stalling at its joint limits until steps or attempts ran out"
    return "repeated stalls on the way to the grasp until steps or attempts ran out"


handled = [r for sc in SCENARIOS for r in load(f"final_{sc}")
           if not r["status"].startswith("skipped") and not (sc == "inverted" and r["suite"] == "libero_spatial")]
fails = [r for r in handled if not r["success"]]
six = [r for r in handled if r["scenario"] != "inverted"]
print(f"Six kinds without 'inverted': {sum(r['success'] for r in six)} of {len(six)} "
      f"({100 * sum(r['success'] for r in six) / len(six):.1f}%). "
      f"With the upside-down grocery items: {sum(r['success'] for r in handled)} of {len(handled)}.\n")
table(["cause", "episodes", "where"],
      [[c, n, ", ".join(f"{k} {v}" for k, v in Counter(
          r["scenario"] + (f" ({r['setup']['kind']})" if r["setup"].get("kind") else "")
          for r in fails if cause(r) == c).most_common())]
       for c, n in Counter(cause(r) for r in fails).most_common()])

allrows = [r for n in ["final_init"] + [f"final_{s}" for s in SCENARIOS] for r in load(n)]
ok = [r for r in allrows if r["success"]]
print(f"All fault-free runs together: {len(ok)} successes; steps median {int(median(r['steps'] for r in ok))}, "
      f"90% within {pct([r['steps'] for r in ok], 0.9)}, max {max(r['steps'] for r in ok)}. "
      f"LIBERO's own success check turned true a median of "
      f"{int(median(r['steps'] - r['libero_success_step'] for r in ok if r.get('libero_success_step') is not None))} steps before the expert finished.")
