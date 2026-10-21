"""Tables of this spike, from the expert runs and the policy replays.

    .venv/bin/python spikes/success/analyze.py --expert NAME --replay NAME [--window 40]

Reads ~/.cache/robots-probes/success/{expert,replay}/NAME/ and prints Markdown tables;
the numbers also go to ~/.cache/robots-probes/success/analysis_<expert>_<replay>.json.
No simulator: this only reads the feature rows recorded by expert_tail.py and
replay_policy.py and applies the definitions of defs.py to them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defs  # noqa: E402

OUT = Path.home() / ".cache" / "robots-probes" / "success"
DEF_NAMES = list(defs.DEFINITIONS)
DEF_LABEL = {
    "a_libero": "(a) LIBERO",
    "b_released": "(b) released",
    "b_ungrasped": "(b'') not between both fingers",
    "b_open": "(b') gripper open",
    "c_rest_k10": "(c) released, at rest, K=10",
    "c_rest_k20": "(c) released, at rest, K=20",
    "d_placed": "(d) placed",
    "d_placed_k5": "(d) placed, K=5",
    "e_carried": "(e) carried by the target",
}
BRANCHES = ("expert", "open_now", "carry_off")
SUITE_SHORT = {"libero_spatial": "spatial", "libero_object": "object"}
OBJECT_ITEMS = ["alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup", "tomato sauce", "butter",
                "milk", "chocolate pudding", "orange juice"]
TASK_OBJECT = {**{("libero_spatial", t): "black bowl" for t in range(10)},
               **{("libero_object", t): x for t, x in enumerate(OBJECT_ITEMS)}}


def q(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else float("nan")


def fmt_steps(xs):
    if not len(xs):
        return "-"
    return f"{int(np.median(xs))} / {int(np.percentile(xs, 90))} / {int(np.max(xs))}"


def branch_rows(z, branch: str, s_l: int, tail: int) -> dict[str, np.ndarray]:
    """Feature rows from the state at s_L on: row 0 is the state at which LIBERO's check first held."""
    keys = [k.split("/", 1)[1] for k in z.files if k.startswith(branch + "/") and not k.endswith("/actions")]
    if not keys:
        return {}
    if branch == "expert":
        return {k: z[f"expert/{k}"][s_l : s_l + tail + 1] for k in keys}
    return {k: z[f"{branch}/{k}"][: tail + 1] for k in keys}


def evaluate(f: dict[str, np.ndarray], window: int) -> dict[str, dict]:
    out = {}
    for name, fn in defs.DEFINITIONS.items():
        ok = fn(f)
        w = ok[: window + 1]
        out[name] = {
            "fired": bool(w.any()),
            "offset": int(np.argmax(w)) if w.any() else None,
            "holds_at_window_end": bool(ok[min(window, len(ok) - 1)]),
        }
    # The end state, at the last recorded row (which may lie beyond the window): LIBERO's
    # check holds there and the robot does not touch the object. That is what the
    # definitions are compared against; it leaves out rest and support on purpose, so that
    # (c), (d) and (e) are not compared with themselves.
    end_placed = bool(defs.def_b(f)[-1])
    for name in out:
        out[name]["false_success"] = out[name]["fired"] and not end_placed
        out[name]["missed"] = end_placed and not out[name]["fired"]
    return out


def agreement_lines(items: list[tuple[str, dict]], cols: list[str]) -> tuple[list[str], dict]:
    """items: (column label, evaluate() result). Counts per definition and column."""
    lines = ["| definition | " + " | ".join(f"{c}: false success / missed" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    res = {}
    for dn in DEF_NAMES:
        cells = []
        for c in cols:
            evs = [ev for lab, ev in items if lab == c]
            fs = sum(ev[dn]["false_success"] for ev in evs)
            mi = sum(ev[dn]["missed"] for ev in evs)
            cells.append(f"{fs} / {mi}")
            res[f"{c}/{dn}"] = [fs, mi, len(evs)]
        lines.append(f"| {DEF_LABEL[dn]} | " + " | ".join(cells) + " |")
    return lines, res


# ------------------------------------------------------------------------------ expert
def expert_tables(name: str, window: int, long_window: int) -> tuple[list[str], dict, dict]:
    d = OUT / "expert" / name
    meta = json.loads((d / "run.json").read_text())
    tail = meta["tail"]
    rows = [json.loads(line) for line in open(d / "summary.jsonl")]
    lines: list[str] = []
    res = {"run": name, "episodes": len(rows)}
    errors = [r for r in rows if "error" in r or "job_error" in r]
    res["errors"] = len(errors)
    res["replica_mismatch_steps"] = sum(sum(r.get("replica_mismatch", {}).values()) for r in rows)
    res["replay_not_identical"] = sum(1 for r in rows for v in r.get("replay_ok", {}).values() if not v)
    res["no_libero_check"] = sum(1 for r in rows if r.get("s_l") is None and "error" not in r)

    per = []  # (row, branch -> eval at window, eval at long window, rows)
    rest_dz = defaultdict(list)
    at_check = defaultdict(list)
    for r in rows:
        if r.get("s_l") is None or "error" in r:
            continue
        z = np.load(d / f"{r['suite']}_t{r['task']}_i{r['init']}.npz")
        ev = {}
        for b in BRANCHES:
            f = branch_rows(z, b, r["s_l"], tail)
            if not f:
                continue
            ev[b] = (evaluate(f, window), evaluate(f, long_window), f)
        per.append((r, ev))
        fe = ev["expert"][2]
        if defs.def_d(fe)[-1]:
            rest_dz[(r["suite"], r["task"])].append(float(fe["dz"][-1]))
    rest = {k: float(np.median(v)) for k, v in rest_dz.items()}
    res["rest_dz"] = {f"{s}/{t}": v for (s, t), v in sorted(rest.items())}
    for r, ev in per:
        f = ev["expert"][2]
        at_check[r["suite"]].append(
            dict(touch=f["robot_touch"][0], tgt=f["target_touch"][0], gap=f["gap"][0], v=f["lin_speed"][0],
                 h=f["dz"][0] - rest.get((r["suite"], r["task"]), np.nan))
        )

    suites = sorted({r["suite"] for r in rows}, key=lambda s: list(SUITE_SHORT).index(s))
    n_suite = {s: sum(1 for r in rows if r["suite"] == s) for s in suites}

    # Table: success by definition and branch
    lines.append(f"Expert run `{name}`: {len(rows)} episodes, window {window} steps after LIBERO's check "
                 f"(in brackets: {long_window} steps).\n")
    hdr = "| definition | " + " | ".join(f"{SUITE_SHORT[s]}: {b}" for s in suites for b in BRANCHES) + " |"
    lines += [hdr, "|---|" + "---|" * (len(suites) * len(BRANCHES))]
    res["success"] = {}
    for dn in DEF_NAMES:
        cells = []
        for s in suites:
            for b in BRANCHES:
                k1 = sum(1 for r, ev in per if r["suite"] == s and b in ev and ev[b][0][dn]["fired"])
                k2 = sum(1 for r, ev in per if r["suite"] == s and b in ev and ev[b][1][dn]["fired"])
                cells.append(f"{k1}/{n_suite[s]} ({k2})")
                res["success"][f"{s}/{b}/{dn}"] = [k1, k2, n_suite[s]]
        lines.append(f"| {DEF_LABEL[dn]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Same, but the definition must still hold at the end of the window
    lines.append(f"Still true at the end of the {window}-step window (the object did not leave again):\n")
    lines += [hdr, "|---|" + "---|" * (len(suites) * len(BRANCHES))]
    res["holds_at_end"] = {}
    for dn in DEF_NAMES:
        cells = []
        for s in suites:
            for b in BRANCHES:
                k1 = sum(1 for r, ev in per if r["suite"] == s and b in ev and ev[b][0][dn]["holds_at_window_end"])
                cells.append(f"{k1}/{n_suite[s]}")
                res["holds_at_end"][f"{s}/{b}/{dn}"] = [k1, n_suite[s]]
        lines.append(f"| {DEF_LABEL[dn]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Extra steps after LIBERO's check, expert branch
    lines.append("Steps from LIBERO's check until the definition first holds, expert branch "
                 f"(median / 90% / max; episodes where it held within {long_window} steps):\n")
    lines += ["| definition | " + " | ".join(SUITE_SHORT[s] for s in suites) + " | both |", "|---|" + "---|" * (len(suites) + 1)]
    res["extra_steps"] = {}
    for dn in DEF_NAMES:
        cells, allx = [], []
        for s in suites:
            xs = [ev["expert"][1][dn]["offset"] for r, ev in per if r["suite"] == s and ev["expert"][1][dn]["fired"]]
            allx += xs
            cells.append(fmt_steps(xs))
            res["extra_steps"][f"{s}/{dn}"] = [q(xs, 50), q(xs, 90), float(max(xs)) if xs else None, len(xs)]
        res["extra_steps"][f"both/{dn}"] = [q(allx, 50), q(allx, 90), float(max(allx)) if allx else None, len(allx)]
        lines.append(f"| {DEF_LABEL[dn]} | " + " | ".join(cells) + f" | {fmt_steps(allx)} |")
    xs_done = [r["expert_done_at"] - r["s_l"] for r, _ in per if r.get("expert_done_at")]
    lines.append(f"| (the expert itself reports done) | | | {fmt_steps(xs_done)} |")
    lines.append("")

    # State at LIBERO's check
    lines.append("State at the step LIBERO's check first holds, expert:\n")
    lines += ["| suite | episodes | object touched by the robot | object touches the target | gripper opening, mm, median (min to max) "
              "| object above its resting height, cm, median / 90% / max | object speed, cm/s, median / 90% |",
              "|---|---|---|---|---|---|---|"]
    res["at_check_expert"] = {}
    for s in suites:
        a = at_check[s]
        touch = sum(x["touch"] > 0.5 for x in a)
        tgt = sum(x["tgt"] > 0.5 for x in a)
        gaps = np.array([x["gap"] for x in a]) * 1000
        h = np.array([x["h"] for x in a]) * 100
        v = np.array([x["v"] for x in a]) * 100
        lines.append(f"| {SUITE_SHORT[s]} | {len(a)} | {touch} | {tgt} | {np.median(gaps):.0f} ({gaps.min():.0f} to {gaps.max():.0f}) | "
                     f"{np.median(h):.1f} / {q(h, 90):.1f} / {h.max():.1f} | {np.median(v):.1f} / {q(v, 90):.1f} |")
        res["at_check_expert"][s] = dict(n=len(a), touched=touch, target_touch=tgt, gap_mm_median=float(np.median(gaps)),
                                         height_cm=[float(np.median(h)), q(h, 90), float(h.max())])
    lines.append("")

    # Agreement with the end state (object placed at the last row, `long_window` steps after the check)
    items = []
    for r, ev in per:
        for b in BRANCHES:
            if b in ev:
                items.append((f"{SUITE_SHORT[r['suite']]} {b}", ev[b][0]))
    cols = [f"{SUITE_SHORT[s]} {b}" for s in suites for b in BRANCHES]
    lines.append(f"Against the end state: an episode counts as placed if, {tail} steps after LIBERO's check, the check holds and the robot does not touch the object. "
                 f"'False success': the definition fired within {window} steps, but the object is not placed at the end. "
                 f"'Missed': placed at the end, but the definition did not fire within {window} steps.\n")
    l2, res["agreement"] = agreement_lines(items, cols)
    lines += l2 + [""]

    # Episodes that separate the definitions, expert branch
    sep = []
    for r, ev in per:
        fired = {dn: ev["expert"][0][dn]["fired"] for dn in DEF_NAMES}
        end = {dn: ev["expert"][0][dn]["holds_at_window_end"] for dn in DEF_NAMES}
        if len(set(fired.values())) > 1 or len(set(end.values())) > 1:
            sep.append((r, fired, end))
    res["separating_expert_branch"] = [
        dict(suite=r["suite"], task=r["task"], init=r["init"], s_l=r["s_l"], status=r["expert_status"],
             not_fired=[dn for dn, v in fi.items() if not v], not_at_end=[dn for dn, v in en.items() if not v])
        for r, fi, en in sep
    ]
    lines.append(f"Expert-branch episodes on which the definitions disagree within {window} steps: {len(sep)}\n")
    for x in res["separating_expert_branch"][:40]:
        lines.append(f"- {x['suite']} t{x['task']} i{x['init']} (s_L {x['s_l']}, {x['status']}): never fired "
                     f"{', '.join(x['not_fired']) or '-'}; not true at window end {', '.join(x['not_at_end']) or '-'}")
    lines.append("")

    # Which open_now episodes fail which definition (drops)
    drops = []
    for r, ev in per:
        if "open_now" in ev and not ev["open_now"][0]["d_placed"]["fired"]:
            drops.append(dict(suite=r["suite"], task=r["task"], init=r["init"],
                              libero_end=ev["open_now"][0]["a_libero"]["holds_at_window_end"]))
    res["open_now_not_placed"] = drops
    res["n_by_suite"] = n_suite

    # Per task: the object, its weight, where it rests, and the expert's state at the check
    lines.append("Per task, expert branch (height: object origin above its resting height at the check; "
                 "steps: from the check to (d), median):\n")
    lines += ["| suite | task | object | mass, g | episodes | touched at the check | height at the check, cm, median (max) | steps to (d) |",
              "|---|---|---|---|---|---|---|---|"]
    res["per_task"] = {}
    for s_name in suites:
        for task in sorted({r["task"] for r, _ in per if r["suite"] == s_name}):
            g = [(r, ev) for r, ev in per if r["suite"] == s_name and r["task"] == task]
            f0 = [ev["expert"][2] for _, ev in g]
            mass = 1000 * f0[0]["weight"][0] / 9.81
            h = np.array([f["dz"][0] - rest.get((s_name, task), np.nan) for f in f0]) * 100
            touched = sum(f["robot_touch"][0] > 0.5 for f in f0)
            offs = [ev["expert"][1]["d_placed"]["offset"] for _, ev in g if ev["expert"][1]["d_placed"]["fired"]]
            obj = TASK_OBJECT.get((s_name, task), "")
            lines.append(f"| {SUITE_SHORT[s_name]} | {task} | {obj} | {mass:.0f} | {len(g)} | {touched} | "
                         f"{np.median(h):.1f} ({h.max():.1f}) | {int(np.median(offs)) if offs else '-'} |")
            res["per_task"][f"{s_name}/{task}"] = dict(object=obj, mass_g=mass, n=len(g), touched=touched,
                                                       height_cm=[float(np.median(h)), float(h.max())],
                                                       steps_to_d=float(np.median(offs)) if offs else None)
    lines.append("")
    return lines, res, rest


# ------------------------------------------------------------------------------ replay
def replay_tables(name: str, rest: dict, window: int) -> tuple[list[str], dict]:
    d = OUT / "replay" / name
    rows = [json.loads(line) for line in open(d / "summary.jsonl")]
    meta = json.loads((d / "run.json").read_text())
    lines: list[str] = []
    res: dict = {"run": name, "episodes": len(rows), "errors": sum(1 for r in rows if "error" in r)}
    res["hash_not_identical"] = sum(1 for r in rows if not r.get("hash_match_all", False))
    res["replica_mismatch_steps"] = sum(r.get("replica_mismatch", 0) for r in rows)
    groups = defaultdict(list)
    for r in rows:
        if "error" in r or not r.get("hash_match_all"):
            continue
        z = np.load(d / f"{r['run']}__{r['episode_id']}.npz")
        n = int(z["n_rec"]) - 1
        f_all = {k: z[k] for k in z.files if k != "n_rec" and "/" not in k}
        chain = np.load(Path(meta["run"][[Path(x).name for x in meta["run"]].index(r["run"])]).expanduser()
                        / "truth" / "chain" / f"{r['episode_id']}.npz")
        g = chain["exe"][:, 6]
        at = {k: v[n] for k, v in f_all.items()}
        ev = {}
        for b in ("open_now", "open_lift"):
            tail = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(b + "/")}
            ev[b] = evaluate(tail, window)
        at_defs = {dn: bool(fn(f_all)[n]) for dn, fn in defs.DEFINITIONS.items()}
        policy = "SmolVLA" if r["control_mode"] == "relative" else "X-VLA"
        faults = ",".join(f["kind"].replace("scene_camera_pan", "camera").replace("action_delay", "delay") for f in r["faults"]) or "none"
        key = (policy, r["suite"], "healthy" if faults == "none" else "faulted")
        groups[key].append(dict(r=r, at=at, ev=ev, at_defs=at_defs, h=at["dz"] - rest.get((r["suite"], r["task"]), np.nan),
                                cmd_close=float(g[-1]) > 0, opened_last5=bool((g[-5:] < 0).any()), faults=faults))
    lines.append(f"Policy episodes replayed (`{name}`): {len(rows)} successful episodes, "
                 f"{len(rows) - res['hash_not_identical'] - res['errors']} reproduced bit for bit at every step.\n")
    lines += ["| policy | suite | episodes | runs | object touched by the robot | gripper still commanded closed | "
              "gripper opening, mm, median (10% to 90%) | object above its resting height, cm, median / 90% / max | "
              "object touches the target | (b) holds | (d) holds |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    res["at_check"] = {}
    order = sorted(groups, key=lambda k: (k[0] != "X-VLA", k[1] != "libero_spatial", k[2] != "healthy"))
    for key in order:
        g = groups[key]
        n = len(g)
        runs = sorted({x["r"]["run"] for x in g})
        touch = sum(x["at"]["robot_touch"] > 0.5 for x in g)
        closed = sum(x["cmd_close"] for x in g)
        gaps = np.array([x["at"]["gap"] for x in g]) * 1000
        h = np.array([x["h"] for x in g]) * 100
        tgt = sum(x["at"]["target_touch"] > 0.5 for x in g)
        b = sum(x["at_defs"]["b_released"] for x in g)
        dd = sum(x["at_defs"]["d_placed"] for x in g)
        lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} ({key[2]}) | {n} | {', '.join(runs)} | {touch} | {closed} | "
                     f"{np.median(gaps):.0f} ({q(gaps, 10):.0f} to {q(gaps, 90):.0f}) | "
                     f"{np.median(h):.1f} / {q(h, 90):.1f} / {h.max():.1f} | {tgt} | {b} | {dd} |")
        res["at_check"]["/".join(key)] = dict(n=n, runs=runs, touched=touch, cmd_closed=closed,
                                               gap_mm=[float(np.median(gaps)), q(gaps, 10), q(gaps, 90)],
                                               height_cm=[float(np.median(h)), q(h, 90), float(h.max())],
                                               target_touch=tgt, b=b, d=dd)
    lines.append("")
    lines.append(f"If the policy had let go at that moment ({window} steps; `open_now`: the arm keeps its last target "
                 "and the gripper opens; `open_lift`: the same, and after 10 steps the arm rises 10 cm):\n")
    lines += ["| policy | suite | episodes | branch | (a) LIBERO still true at the end | (b) released | (d) placed | (d, K=5) | (e) carried | "
              "(c) K=20 | steps to (d), median / 90% / max |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    res["tails"] = {}
    for key in order:
        g = groups[key]
        for b in ("open_now", "open_lift"):
            cnt = {dn: sum(x["ev"][b][dn]["fired"] for x in g) for dn in DEF_NAMES}
            a_end = sum(x["ev"][b]["a_libero"]["holds_at_window_end"] for x in g)
            offs = [x["ev"][b]["d_placed"]["offset"] for x in g if x["ev"][b]["d_placed"]["fired"]]
            lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} ({key[2]}) | {len(g)} | {b} | {a_end} | {cnt['b_released']} | "
                         f"{cnt['d_placed']} | {cnt['d_placed_k5']} | {cnt['e_carried']} | {cnt['c_rest_k20']} | {fmt_steps(offs)} |")
            res["tails"]["/".join(key) + "/" + b] = dict(
                n=len(g), libero_end=a_end, fired=cnt, d_offset=[q(offs, 50), q(offs, 90), max(offs) if offs else None],
                lost=[x["r"]["run"] + "/" + x["r"]["episode_id"] for x in g if not x["ev"][b]["a_libero"]["holds_at_window_end"]],
                not_placed=[x["r"]["run"] + "/" + x["r"]["episode_id"] for x in g if not x["ev"][b]["d_placed"]["fired"]])
    lines.append("")
    items = [(f"{k[0]} {SUITE_SHORT[k[1]]} {k[2]} {b}", x["ev"][b]) for k in order for x in groups[k] for b in ("open_now", "open_lift")]
    cols = [f"{k[0]} {SUITE_SHORT[k[1]]} {k[2]} open_lift" for k in order]
    lines.append(f"Against the end state of `open_lift` (LIBERO's check holds and the robot does not touch the object {window} steps after the check):\n")
    l2, res["agreement_open_lift"] = agreement_lines(items, cols)
    lines += l2 + [""]
    # What a tail after LIBERO's check costs in policy calls. X-VLA executes 30 actions per call, SmolVLA 10
    # (both as run in these fleets), so at the check (n - 1) mod chunk actions of the current chunk are used.
    lines.append("Extra policy calls per successful episode if the episode runs on after LIBERO's check: for a tail "
                 "of T steps, the actions left in the current chunk are used first. T from `open_lift` is the "
                 "steps until (d) holds when the policy lets go and lifts; 20 and 40 are fixed tails.\n")
    lines += ["| policy | suite | episodes | actions per call | calls per episode now (successes) | "
              "extra calls, T = steps to (d) in open_lift | T = 20 | T = 40 |", "|---|---|---|---|---|---|---|---|"]
    res["extra_calls"] = {}
    for key in order:
        g = groups[key]
        chunk = 30 if key[0] == "X-VLA" else 10
        def extra(tail_steps, n):
            left = chunk - ((n - 1) % chunk + 1)
            return -(-max(0, tail_steps - left) // chunk)
        now = np.mean([-(-x["r"]["n_steps"] // chunk) for x in g])
        e_d = np.mean([extra(x["ev"]["open_lift"]["d_placed"]["offset"] or 40, x["r"]["n_steps"]) for x in g])
        e20 = np.mean([extra(20, x["r"]["n_steps"]) for x in g])
        e40 = np.mean([extra(40, x["r"]["n_steps"]) for x in g])
        lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} ({key[2]}) | {len(g)} | {chunk} | {now:.2f} | {e_d:.2f} | {e20:.2f} | {e40:.2f} |")
        res["extra_calls"]["/".join(key)] = dict(chunk=chunk, now=float(now), to_d=float(e_d), t20=float(e20), t40=float(e40))
    lines.append("")

    lines.append("Per task, X-VLA healthy (height: object origin above its resting height at the check):\n")
    lines += ["| suite | task | object | episodes | touched at the check | gripper commanded closed | "
              "height at the check, cm, median (max) | robot's force on the object, N, median |",
              "|---|---|---|---|---|---|---|---|"]
    res["per_task_xvla"] = {}
    for key in order:
        if key[0] != "X-VLA" or key[2] != "healthy":
            continue
        for task in sorted({x["r"]["task"] for x in groups[key]}):
            g = [x for x in groups[key] if x["r"]["task"] == task]
            h = np.array([x["h"] for x in g]) * 100
            fz = np.array([x["at"]["robot_force"] for x in g])
            touched = sum(x["at"]["robot_touch"] > 0.5 for x in g)
            closed = sum(x["cmd_close"] for x in g)
            lines.append(f"| {SUITE_SHORT[key[1]]} | {task} | {TASK_OBJECT[(key[1], task)]} | {len(g)} | {touched} | {closed} | "
                         f"{np.median(h):.1f} ({h.max():.1f}) | {np.median(fz):.1f} |")
            res["per_task_xvla"][f"{key[1]}/{task}"] = dict(n=len(g), touched=touched, closed=closed,
                                                            height_cm=[float(np.median(h)), float(h.max())],
                                                            force_n=float(np.median(fz)))
    lines.append("")
    # the highest holds, for the videos
    tall = sorted((x for g in groups.values() for x in g), key=lambda x: -x["h"])[:8]
    res["highest_at_check"] = [dict(run=x["r"]["run"], episode=x["r"]["episode_id"], suite=x["r"]["suite"], task=x["r"]["task"],
                                    height_cm=round(100 * x["h"], 1), touched=bool(x["at"]["robot_touch"] > 0.5),
                                    open_now_libero_end=x["ev"]["open_now"]["a_libero"]["holds_at_window_end"]) for x in tall]
    return lines, res


def policy_tail_tables(name: str, window: int) -> tuple[list[str], dict]:
    """Policy runs that kept acting after LIBERO's check (fleet_tail.py), replayed with --tail 0.

    Every definition contains LIBERO's check, so an episode in which the check never held
    fails under all of them; the denominators are all episodes of the runs.
    """
    d = OUT / "replay" / name
    meta = json.loads((d / "run.json").read_text())
    rows = [json.loads(line) for line in open(d / "summary.jsonl")]
    totals: dict = defaultdict(int)
    for run in meta["run"]:
        for line in open(Path(run).expanduser() / "truth" / "episodes.jsonl"):
            e = json.loads(line)
            faults = ",".join(f["kind"] for f in e["faults"]) or "none"
            totals[(Path(run).name, e["suite"], faults)] += 1
    groups = defaultdict(list)
    bad = 0
    for r in rows:
        if "error" in r or not r.get("hash_match_all") or r.get("s_l") is None:
            bad += 1
            continue
        z = np.load(d / f"{r['run']}__{r['episode_id']}.npz")
        f = {k: z[k] for k in z.files if k != "n_rec" and "/" not in k}
        tail = {k: v[r["s_l"]:] for k, v in f.items()}
        ev = evaluate(tail, min(window, len(tail["libero"]) - 1))
        faults = ",".join(x["kind"] for x in r["faults"]) or "none"
        groups[(r["run"], r["suite"], faults)].append(dict(r=r, ev=ev, n_after=len(tail["libero"]) - 1))
    lines = [f"Policy runs with the policy acting on after LIBERO's check (`{name}`), window {window} steps; "
             f"{bad} episodes not reproduced, not replayable or without LIBERO's check.\n"]
    hdr = "| run | suite | faults | episodes | " + " | ".join(DEF_LABEL[dn] for dn in DEF_NAMES) + " |"
    lines += [hdr, "|---|---|---|---|" + "---|" * len(DEF_NAMES)]
    res: dict = {"run": name, "bad": bad, "groups": {}}
    for key in sorted(totals):
        g = groups.get(key, [])
        cnt = {dn: sum(x["ev"][dn]["fired"] for x in g) for dn in DEF_NAMES}
        lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} | {key[2]} | {totals[key]} | "
                     + " | ".join(str(cnt[dn]) for dn in DEF_NAMES) + " |")
        offs = {dn: [x["ev"][dn]["offset"] for x in g if x["ev"][dn]["fired"]] for dn in DEF_NAMES}
        res["groups"]["/".join(key)] = dict(
            n=totals[key], fired=cnt, offsets={dn: [q(v, 50), q(v, 90), max(v) if v else None] for dn, v in offs.items()},
            separating=[x["r"]["episode_id"] for x in g if len({x["ev"][dn]["fired"] for dn in DEF_NAMES}) > 1],
            tail_cut=[x["r"]["episode_id"] for x in g if x["n_after"] < window])
    lines.append("")
    lines.append("Steps from LIBERO's check until the definition first holds (median / 90% / max):\n")
    lines += ["| run | suite | faults | " + " | ".join(DEF_LABEL[dn] for dn in DEF_NAMES) + " |",
              "|---|---|---|" + "---|" * len(DEF_NAMES)]
    for key in sorted(totals):
        g = groups.get(key, [])
        cells = [fmt_steps([x["ev"][dn]["offset"] for x in g if x["ev"][dn]["fired"]]) for dn in DEF_NAMES]
        lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} | {key[2]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines, res


def policy_tail_summary(name: str, window: int, compare: dict[str, str]) -> tuple[list[str], dict]:
    """The X-VLA-side numbers from a fleet_tail.py run: does the policy let go, is the object placed, at what cost.

    Per run (one manifest): LIBERO successes; how many of them the policy commands open
    (gripper action < 0) within `window` steps after the check; how many end released (b),
    placed (d), placed K=5 and carried (e) within `window`; and the extra policy calls per
    LIBERO success under the engine rule recommended in the README: stop at the first
    step placed holds, else `window` steps after the check, never past the step limit.
    `compare` maps a tail run's name to the original run (same manifest and settings):
    the state at the check step must be the original's final state.
    """
    d = OUT / "replay" / name
    meta = json.loads((d / "run.json").read_text())
    run_dirs = {Path(r).expanduser().name: Path(r).expanduser() for r in meta["run"]}
    rows = [json.loads(line) for line in open(d / "summary.jsonl")]
    totals = {}
    for rn, rd in run_dirs.items():
        totals[rn] = sum(1 for _ in open(rd / "truth" / "episodes.jsonl"))
    chunk = 30
    per = defaultdict(list)
    for r in rows:
        if "error" in r or not r.get("hash_match_all") or r.get("s_l") is None:
            continue
        z = np.load(d / f"{r['run']}__{r['episode_id']}.npz")
        f = {k: z[k] for k in z.files if k != "n_rec" and "/" not in k}
        s_l = r["s_l"]
        tail = {k: v[s_l:] for k, v in f.items()}
        ev = evaluate(tail, min(window, len(tail["libero"]) - 1))
        exe = np.load(run_dirs[r["run"]] / "truth" / "chain" / f"{r['episode_id']}.npz")["exe"]
        g = exe[s_l : s_l + window, 6]  # actions executed from the check step on
        opened = bool((g < 0).any())
        open_at = int(np.argmax(g < 0)) if opened else None
        limit = r.get("max_steps") or 280
        end = s_l + ev["d_placed"]["offset"] if ev["d_placed"]["fired"] else min(s_l + window, limit)
        extra = -(-end // chunk) - (-(-s_l // chunk))
        per[r["run"]].append(dict(r=r, ev=ev, opened=opened, open_at=open_at, extra=extra,
                                  cut=len(tail["libero"]) - 1 < window))
    lines = [f"X-VLA acting on after LIBERO's check (`{name}`), window {window} steps:\n",
             "| run | episodes | LIBERO successes | commands open within the window | (b) released | (d) placed | (d) K=5 "
             "| (e) carried | LIBERO only | steps to (d), median / 90% / max | extra calls per LIBERO success | tail cut by the step limit |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res: dict = {"run": name, "window": window, "runs": {}}
    for rn in sorted(run_dirs):
        g = per.get(rn, [])
        n_l = len(g)
        cnt = {dn: sum(x["ev"][dn]["fired"] for x in g) for dn in DEF_NAMES}
        opened = sum(x["opened"] for x in g)
        offs = [x["ev"]["d_placed"]["offset"] for x in g if x["ev"]["d_placed"]["fired"]]
        extra = float(np.mean([x["extra"] for x in g])) if g else float("nan")
        cut = sum(x["cut"] for x in g)
        lines.append(f"| {rn} | {totals[rn]} | {n_l} | {opened} | {cnt['b_released']} | {cnt['d_placed']} | "
                     f"{cnt['d_placed_k5']} | {cnt['e_carried']} | {n_l - cnt['d_placed']} | {fmt_steps(offs)} | {extra:.2f} | {cut} |")
        res["runs"][rn] = dict(
            episodes=totals[rn], libero=n_l, opened=opened, fired=cnt, libero_only=n_l - cnt["d_placed"],
            d_steps=[q(offs, 50), q(offs, 90), max(offs) if offs else None], extra_calls=extra, cut=cut,
            not_placed=[x["r"]["episode_id"] for x in g if not x["ev"]["d_placed"]["fired"]],
            open_at=[q([x["open_at"] for x in g if x["opened"]], p) for p in (50, 90)])
        # by fault, for the mix
        by_f = defaultdict(list)
        for x in g:
            by_f[",".join(f["kind"] for f in x["r"]["faults"]) or "none"].append(x)
        if len(by_f) > 1:
            for fk, gg in sorted(by_f.items()):
                c2 = {dn: sum(x["ev"][dn]["fired"] for x in gg) for dn in DEF_NAMES}
                o2 = [x["ev"]["d_placed"]["offset"] for x in gg if x["ev"]["d_placed"]["fired"]]
                lines.append(f"| {rn}: {fk} | | {len(gg)} | {sum(x['opened'] for x in gg)} | {c2['b_released']} | "
                             f"{c2['d_placed']} | {c2['d_placed_k5']} | {c2['e_carried']} | {len(gg) - c2['d_placed']} | "
                             f"{fmt_steps(o2)} | {np.mean([x['extra'] for x in gg]):.2f} | {sum(x['cut'] for x in gg)} |")
                res["runs"][f"{rn}:{fk}"] = dict(libero=len(gg), fired=c2, opened=sum(x["opened"] for x in gg))
    lines.append("")
    if compare:
        lines.append("Same episodes as the original runs up to the check step (state hash at the check step "
                     "against the original's final state hash; outcome by LIBERO's check):\n")
        lines += ["| tail run | original | episodes compared | LIBERO success in both | only original | only tail run | "
                  "check step and state identical |", "|---|---|---|---|---|---|---|"]
        res["compare"] = {}
        for tail_run, orig in compare.items():
            od = Path(orig).expanduser()
            o = {json.loads(x)["episode_id"]: json.loads(x) for x in open(od / "truth" / "episodes.jsonl")}
            t = {json.loads(x)["episode_id"]: json.loads(x) for x in open(run_dirs[tail_run] / "truth" / "episodes.jsonl")}
            both = sum(1 for k in t if t[k]["success"] and o.get(k, {}).get("success"))
            only_o = sum(1 for k in t if not t[k]["success"] and o.get(k, {}).get("success"))
            only_t = sum(1 for k in t if t[k]["success"] and not o.get(k, {}).get("success"))
            same = 0
            for k in t:
                if t[k]["success"] and o.get(k, {}).get("success") and t[k].get("libero_success_step") == o[k]["n_steps"]:
                    h = np.load(run_dirs[tail_run] / "truth" / "chain" / f"{k}.npz")["state_hash"][o[k]["n_steps"]]
                    same += int(int(h) == int(o[k]["final_state_hash"]))
            lines.append(f"| {tail_run} | {od.name} | {len(t)} | {both} | {only_o} | {only_t} | {same} of {both} |")
            res["compare"][tail_run] = dict(original=od.name, n=len(t), both=both, only_original=only_o,
                                            only_tail=only_t, identical=same)
        lines.append("")
    return lines, res


def tilt_lines(replays: list[str], window: int) -> tuple[list[str], dict]:
    """The object's tilt (its starting up axis against the vertical, degrees) at the first step placed holds.

    For every replay made after the tilt feature was added: the scripted continuations of an
    ordinary replay (`open_now`, `open_lift`) and the recorded tail of a fleet_tail.py replay
    (rows from the check step on).
    """
    out: dict = {}
    for name in replays:
        d = OUT / "replay" / name
        for r in (json.loads(x) for x in open(d / "summary.jsonl")):
            if "error" in r or r.get("s_l") is None:
                continue
            z = np.load(d / f"{r['run']}__{r['episode_id']}.npz")
            policy = "SmolVLA" if r.get("control_mode") == "relative" else "X-VLA"
            parts = {b: {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(b + "/")} for b in ("open_now", "open_lift")}
            if not parts["open_now"]:
                parts = {"own": {k: z[k][r["s_l"]:] for k in z.files if k != "n_rec" and "/" not in k}}
            for b, f in parts.items():
                if "tilt" not in f:
                    continue
                ok = defs.def_d(f)[: window + 1]
                if ok.any():
                    key = (policy, r["suite"], b)
                    out.setdefault(key, []).append(float(f["tilt"][int(np.argmax(ok))]))
    lines = ["Tilt of the object at the first step placed holds (degrees from its starting up axis):\n",
             "| policy | suite | continuation | placed | median | 90% | max |", "|---|---|---|---|---|---|---|"]
    res = {}
    for key in sorted(out):
        v = np.array(out[key])
        lines.append(f"| {key[0]} | {SUITE_SHORT[key[1]]} | {key[2]} | {len(v)} | {np.median(v):.1f} | {q(v, 90):.1f} | {v.max():.1f} |")
        res["/".join(key)] = dict(n=len(v), median=float(np.median(v)), p90=q(v, 90), max=float(v.max()))
    lines.append("")
    return lines, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="")
    ap.add_argument("--replay", default="", help="a replay of ordinary policy runs (made with --tail 40)")
    ap.add_argument("--policy-run", default="", help="a replay (made with --tail 0) of runs made by fleet_tail.py")
    ap.add_argument("--tilt", action="append", default=[], help="replays with the tilt feature, for the tilt table")
    ap.add_argument("--compare", action="append", default=[],
                    help="TAILRUN=ORIGINAL_RUN_DIR: check that a fleet_tail.py run repeats the original up to the check")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--long-window", type=int, default=80)
    args = ap.parse_args()
    lines: list[str] = []
    res: dict = {"thresholds": dict(REST_LIN=defs.REST_LIN, REST_ANG=defs.REST_ANG, OPEN_GAP=defs.OPEN_GAP)}
    rest: dict = {}
    if args.expert:
        l1, res_e, rest = expert_tables(args.expert, args.window, args.long_window)
        lines += l1
        res["expert"] = res_e
    if args.replay:
        l2, res_r = replay_tables(args.replay, rest, args.window)
        lines += l2
        res["replay"] = res_r
    if args.policy_run:
        compare = dict(x.split("=", 1) for x in args.compare)
        l4, res_s = policy_tail_summary(args.policy_run, args.window, compare)
        l3, res_p = policy_tail_tables(args.policy_run, args.window)
        lines += l4 + l3
        res["policy_run"] = res_p
        res["policy_summary"] = res_s
    if args.tilt:
        l5, res["tilt"] = tilt_lines(args.tilt, args.window)
        lines += l5
    print("\n".join(lines))
    if args.expert:
        for k in ("errors", "replica_mismatch_steps", "replay_not_identical", "no_libero_check"):
            print(f"expert {k}: {res['expert'][k]}")
    if args.replay:
        for k in ("errors", "hash_not_identical", "replica_mismatch_steps"):
            print(f"replay {k}: {res['replay'][k]}")
    tag = "_".join(x for x in (args.expert, args.replay, args.policy_run, *args.tilt) if x)
    path = OUT / f"analysis_{tag}_w{args.window}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    print(f"# numbers -> {path}")


if __name__ == "__main__":
    main()
