"""Print results from eval_info.json and measure.json.

    python spikes/baseline/summarize.py [RUN_NAME ...]     one line per run (default: all runs)
    python spikes/baseline/summarize.py --suite NAME       add up the per-task runs NAME_t0 .. NAME_t9
                                                           written by run_suite.sh
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RUNS = Path.home() / ".cache/robots-probes/baseline/runs"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a success rate from k successes in n episodes."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(d: Path):
    info_p, meas_p = d / "eval_info.json", d / "measure.json"
    info = json.loads(info_p.read_text()) if info_p.exists() else None
    meas = json.loads(meas_p.read_text()) if meas_p.exists() else {}
    return info, meas


def row(d: Path) -> str:
    info, m = load(d)
    if info is None:
        return f"{d.name}: no eval_info.json (run failed or still running)"
    per = {}
    for t in info["per_task"]:
        per.setdefault(t["task_group"], {})[t["task_id"]] = sum(t["metrics"]["successes"])
    parts = []
    for g, tasks in per.items():
        ids = sorted(tasks)
        n = sum(len(t["metrics"]["successes"]) for t in info["per_task"] if t["task_group"] == g)
        parts.append(f"{g} {sum(tasks.values())}/{n} tasks{ids if len(ids) < 10 else ''} per-task {[tasks[i] for i in ids]}")
    perf = ""
    if m:
        own = m.get("own_gpu_peak_gib", {}).get("drm-memory-gtt")
        perf = (
            f" | {m['s_per_episode_rollout']} s/ep, {m['env_steps_per_s_rollout']} steps/s"
            f" (wall {m['wall_s']} s), model {m['model_calls']} calls x {m['model_ms_per_call']} ms,"
            f" sim {m['sim_step_s']} s, RSS {m['peak_rss_gib']} GiB,"
            f" GPU torch {m['torch_peak_reserved_gib']} GiB / process {own} GiB / system +{m['gtt_peak_delta_gib']} GiB"
        )
    return f"{d.name}: " + "; ".join(parts) + perf


def suite(name: str) -> None:
    per_task, n_per_task, missing = [], [], []
    tot = {"episodes": 0, "env_steps": 0, "rollout_s": 0.0, "wall_s": 0.0, "model_s": 0.0, "model_calls": 0,
           "sim_step_s": 0.0, "sim_reset_s": 0.0}
    peak = {"peak_rss_gib": 0.0, "torch_peak_reserved_gib": 0.0, "own_gtt": 0.0, "gtt_peak_gib": 0.0}
    group = "?"
    for t in range(10):
        info, m = load(RUNS / f"{name}_t{t}")
        if info is None:
            missing.append(t)
            per_task.append(None)
            n_per_task.append(0)
            continue
        s = info["per_task"][0]["metrics"]["successes"]
        group = info["per_task"][0]["task_group"]
        per_task.append(sum(s))
        n_per_task.append(len(s))
        for k in tot:
            tot[k] += m.get(k, 0)
        peak["peak_rss_gib"] = max(peak["peak_rss_gib"], m.get("peak_rss_gib", 0))
        peak["torch_peak_reserved_gib"] = max(peak["torch_peak_reserved_gib"], m.get("torch_peak_reserved_gib") or 0)
        peak["own_gtt"] = max(peak["own_gtt"], m.get("own_gpu_peak_gib", {}).get("drm-memory-gtt", 0))
        peak["gtt_peak_gib"] = max(peak["gtt_peak_gib"], m.get("gtt_peak_gib", 0))
    k = sum(x for x in per_task if x is not None)
    n = sum(n_per_task)
    lo, hi = wilson(k, n)
    print(f"{name}: {group} {k}/{n} = {100 * k / max(n, 1):.1f}%  (95% interval {100 * lo:.1f} to {100 * hi:.1f}%)")
    print(f"  per task (of {max(n_per_task) if n_per_task else 0}): {per_task}" + (f"   missing tasks: {missing}" if missing else ""))
    if tot["episodes"]:
        print(
            f"  {tot['rollout_s'] / tot['episodes']:.2f} s per episode in the rollout, {tot['wall_s'] / tot['episodes']:.2f} s wall incl. start-up;"
            f" {tot['env_steps'] / tot['rollout_s']:.1f} env steps/s; mean episode {tot['env_steps'] / tot['episodes']:.0f} steps;"
            f" model {tot['model_calls']} calls, {1000 * tot['model_s'] / max(tot['model_calls'], 1):.0f} ms each ({100 * tot['model_s'] / tot['rollout_s']:.0f}% of rollout time);"
            f" simulator steps {100 * tot['sim_step_s'] / tot['rollout_s']:.0f}%, resets {100 * tot['sim_reset_s'] / tot['rollout_s']:.0f}%"
        )
        print(
            f"  peak process memory {peak['peak_rss_gib']} GiB; peak GPU memory: PyTorch {peak['torch_peak_reserved_gib']} GiB,"
            f" whole process {peak['own_gtt']} GiB, whole machine {peak['gtt_peak_gib']} GiB; total wall time {tot['wall_s'] / 60:.1f} min"
        )


if len(sys.argv) > 2 and sys.argv[1] == "--suite":
    for nm in sys.argv[2:]:
        suite(nm)
else:
    names = sys.argv[1:] or sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    for nm in names:
        print(row(RUNS / nm))
