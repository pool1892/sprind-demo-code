"""Tables from finished fleet runs. Reads only files; needs numpy.

    python analyze.py dose RUN_DIR...                 success by fault level
    python analyze.py compare A_DIR[,A_DIR..] B_DIR[,B_DIR..]   are episodes with the same identity bit-identical?
    python analyze.py attribute MAIN_DIR REPLAY_DIR   per episode: did the fault cause the failure?
    python analyze.py summary RUN_DIR                 speed, memory and bytes per episode of one run

Two episodes are "the same" when their identity agrees: robot, suite, task,
start state, scene seed, noise seed, batch size, row, step limit and fault set.
"Identical" means that every stored array is the same bytes: the hash of the
simulator state and of both camera images after every step, the robot state in
full precision, the commanded and the executed actions, the fixture layout,
and with them the length and the outcome.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDENT = ("robot", "suite", "task", "init", "scene_seed", "noise_seed", "rows", "row", "max_steps")
ARRAYS = ("fixtures", "state_hash", "img_hash", "proprio", "cmd", "exe")


def load(run_dirs) -> list[dict]:
    out = []
    for d in run_dirs:
        d = Path(d)
        for line in (d / "truth" / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            e["_dir"] = d
            out.append(e)
    return out


def fault_sig(e) -> tuple:
    sig = []
    for f in e.get("faults") or []:
        params = tuple(sorted((k, float(v)) for k, v in f.items() if k != "kind"))
        if any(v == 0 for _, v in params):
            continue
        sig.append((f["kind"], params))
    return tuple(sorted(sig))


def key(e) -> tuple:
    return tuple(e[k] for k in IDENT) + (fault_sig(e),)


def chain(e) -> dict:
    with np.load(e["_dir"] / "truth" / "chain" / f"{e['episode_id']}.npz") as z:
        return {k: z[k] for k in ARRAYS}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def first_diff(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0 if len(a) != len(b) else -1
    x = np.ascontiguousarray(a[:n]).reshape(n, -1).view(np.uint8).reshape(n, -1)
    y = np.ascontiguousarray(b[:n]).reshape(n, -1).view(np.uint8).reshape(n, -1)
    idx = np.flatnonzero((x != y).any(axis=1))
    if len(idx):
        return int(idx[0])
    return -1 if len(a) == len(b) else n


def cmd_dose(run_dirs):
    eps = load(run_dirs)
    by = defaultdict(list)
    for e in eps:
        by[(e["robot"], e["variant"])].append(e)

    def order(item):
        (robot, variant), _ = item
        return (robot, float(variant.split("=")[1]) if "=" in variant else 0.0)

    print("| Robot | Level | Successes | Rate | 95% interval | Mean steps | Successes by task 0 to 9 |")
    print("|---|---|---|---|---|---|---|")
    for (robot, variant), es in sorted(by.items(), key=order):
        k, n = sum(e["success"] for e in es), len(es)
        lo, hi = wilson(k, n)
        per_task = [sum(e["success"] for e in es if e["task"] == t) for t in range(10)]
        print(f"| {robot} | {variant} | {k}/{n} | {100 * k / n:.1f}% | {100 * lo:.0f} to {100 * hi:.0f}% | "
              f"{np.mean([e['n_steps'] for e in es]):.0f} | {' '.join(map(str, per_task))} |")


def compare(a_eps, b_eps):
    b_by = defaultdict(list)
    for e in b_eps:
        b_by[key(e)].append(e)
    rows = []
    for ea in a_eps:
        for eb in b_by.get(key(ea), []):
            ca, cb = chain(ea), chain(eb)
            diffs = {k: first_diff(ca[k], cb[k]) for k in ARRAYS}
            identical = all(v == -1 for v in diffs.values()) and ea["success"] == eb["success"] \
                and ea["n_steps"] == eb["n_steps"]
            first = min([v for v in diffs.values() if v >= 0], default=-1)
            gap = 0.0
            if not identical:
                n = min(len(ca["proprio"]), len(cb["proprio"]))
                gap = float(np.abs(ca["proprio"][:n, 18:21] - cb["proprio"][:n, 18:21]).max())
            rows.append({
                "a": ea["episode_id"], "b": eb["episode_id"], "robot": ea["robot"], "faults": bool(fault_sig(ea)),
                "identical": identical, "first_diff_step": first, "flip": ea["success"] != eb["success"],
                "which": [k for k, v in diffs.items() if v >= 0], "max_gripper_path_gap_m": gap,
                "a_workers": (ea["worker_pid"], ea["worker_life"]), "b_workers": (eb["worker_pid"], eb["worker_life"]),
            })
    return rows


def cmd_compare(a_dirs, b_dirs):
    rows = compare(load(a_dirs.split(",")), load(b_dirs.split(",")))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["robot"], "with its fault" if r["faults"] else "no fault")].append(r)
    print("| Robot | Fault set | Pairs | Identical | Outcome flips |")
    print("|---|---|---|---|---|")
    for g, rs in sorted(groups.items()):
        print(f"| {g[0]} | {g[1]} | {len(rs)} | {sum(r['identical'] for r in rs)} | {sum(r['flip'] for r in rs)} |")
    bad = [r for r in rows if not r["identical"]]
    for r in bad[:20]:
        print(f"  differs: {r['a']} vs {r['b']}: first at step {r['first_diff_step']} in {r['which']}, "
              f"gripper paths up to {100 * r['max_gripper_path_gap_m']:.2f} cm apart, flip={r['flip']}")
    print(f"total: {len(rows)} pairs, {len(rows) - len(bad)} identical, {sum(r['flip'] for r in rows)} outcome flips")
    return rows


def cmd_attribute(main_dir, replay_dir):
    main, rep = load([main_dir]), load([replay_dir])
    rep_by = {tuple(e[k] for k in IDENT): e for e in rep if not fault_sig(e)}
    table, counts = [], defaultdict(lambda: defaultdict(int))
    for e in main:
        r = rep_by[tuple(e[k] for k in IDENT)]
        if not fault_sig(e):
            cls = "healthy robot"
        elif not e["success"] and r["success"]:
            cls = "fault caused the failure"
        elif not e["success"] and not r["success"]:
            cls = "failed with and without the fault"
        elif e["success"] and r["success"]:
            cls = "succeeded with and without the fault"
        else:
            cls = "succeeded only with the fault"
        counts[e["robot"]][cls] += 1
        table.append({"episode_id": e["episode_id"], "robot": e["robot"], "task": e["task"], "init": e["init"],
                      "faults": e["faults"], "success": e["success"], "n_steps": e["n_steps"],
                      "success_without_fault": r["success"], "n_steps_without_fault": r["n_steps"], "class": cls})
    out = Path(main_dir) / "truth" / "attribution.json"
    out.write_text(json.dumps(table, indent=1))
    print("| Robot | As run | With the fault removed | Fault caused the failure | Failed either way | "
          "Succeeded either way | Succeeded only with the fault |")
    print("|---|---|---|---|---|---|---|")
    for robot in sorted(counts):
        es = [t for t in table if t["robot"] == robot]
        c = counts[robot]
        if c["healthy robot"]:  # no fault to remove: its replays are plain repeats
            cells = ["-"] * 4
        else:
            cells = [c["fault caused the failure"], c["failed with and without the fault"],
                     c["succeeded with and without the fault"], c["succeeded only with the fault"]]
        print(f"| {robot} | {sum(t['success'] for t in es)}/{len(es)} | "
              f"{sum(t['success_without_fault'] for t in es)}/{len(es)} | " + " | ".join(map(str, cells)) + " |")
    print(f"written: {out}")


def cmd_summary(run_dir):
    r = json.loads((Path(run_dir) / "run.json").read_text())
    p, b = r["peaks"], r["bytes_per_episode"]
    eps = load([run_dir])
    growth = []
    lives = defaultdict(list)
    for e in eps:
        lives[(e["worker_pid"])].append((e["t_end"], e["worker_pss_mib"]))
    for seq in lives.values():
        seq.sort()
        if len(seq) >= 4:
            growth.append((seq[-1][1] - seq[1][1]) / (len(seq) - 2))
    print(json.dumps({
        "episodes": r["episodes"], "successes": r["successes"], "total_s": round(r["total_s"], 1),
        "episodes_per_hour_overall": round(r["episodes_per_hour_overall"]),
        "episodes_per_hour_steady": round(r["episodes_per_hour_steady"]),
        "mean_steps": round(r["mean_steps"], 1), "mean_chunks": round(r["mean_chunks"], 2),
        "calls": r["inference_calls"], "s_per_call": round(r["inference_s_per_call"], 3),
        "mean_rows_in_use": round(r["mean_rows_filled"], 2), "server_busy": round(r["server_busy_fraction"], 3),
        "worker_time_share": {k: round(v, 3) for k, v in r["worker_time_share"].items()},
        "worker_ms_per_step": round(r["worker_ms_per_step"], 2), "reset_s": round(r["worker_reset_s_mean"], 3),
        "build_s": round(r["worker_build_s_mean"], 2), "write_s": round(r["worker_write_s_mean"], 3),
        "env_builds": r["env_builds"], "worker_processes": r["worker_processes"],
        "peak_process_memory_gib": round(p["pss_total_mib"] / 1024, 2),
        "peak_server_memory_gib": round(p["pss_server_mib"] / 1024, 2),
        "peak_worker_memory_gib": round(p["pss_worker_max_mib"] / 1024, 2),
        "peak_gpu_memory_these_processes_gib": round(p["gpu_own_total_mib"] / 1024, 2),
        "peak_gpu_memory_server_gib": round(p["gpu_server_mib"] / 1024, 2),
        "torch_peak_allocated_gib": round(r["torch_peak_allocated_gib"], 2),
        "peak_gpu_memory_machine_gib": round(p["gtt_gib"], 2), "gpu_memory_machine_at_start_gib": round(r["gtt_start_gib"], 2),
        "min_memory_available_gib": round(p["mem_available_min_gib"], 1),
        "load_start": r["load_start"], "load_end": r["load_end"],
        "bytes_per_episode": {k: round(v) for k, v in b.items()},
        "bytes_per_episode_log_total": round(b["steps_parquet"] + b["video_scene"] + b["video_wrist"]
                                             + r["bytes_episodes_table"] / r["episodes"]),
        "worker_memory_growth_mib_per_episode": [round(g, 1) for g in growth],
    }, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dose":
        cmd_dose(sys.argv[2:])
    elif cmd == "compare":
        cmd_compare(sys.argv[2], sys.argv[3])
    elif cmd == "attribute":
        cmd_attribute(sys.argv[2], sys.argv[3])
    elif cmd == "summary":
        cmd_summary(sys.argv[2])
    else:
        raise SystemExit(__doc__)
