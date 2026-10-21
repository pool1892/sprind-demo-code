"""Batches of teleoperator episodes in up to four worker processes, one line per episode.

    run.sh evaluate.py --name NAME --scenario init|KIND [--suites a,b] [--tasks 0-9] [--inits 0-49]
                       [--mode absolute|relative] [--no-restore] [--seed-base 1000]
    run.sh evaluate.py --name NAME --scenario xvla --handovers FILE.jsonl

Rows go to ~/.cache/robots-probes/teleop-absolute/results/NAME.jsonl, with a
`NAME.run.json` beside it (command, start and end, code hashes, episode count).
Each worker process runs one task's episodes in one environment and is
replaced after every job, as in spikes/teleop-expert.

Scene seed of an episode from the stored start states: seed base + start state
index (1000 + i by default, the reference evaluation's seed 1000 counted on).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = Path.home() / ".cache" / "robots-probes" / "teleop-absolute"
CODE = ["teleop/actions.py", "teleop/expert.py", "teleop/grasp.py", "teleop/scene.py", "disturb.py",
        "episode.py", "snapshot.py", "xvla_record.py", "evaluate.py"]


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def code_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()[:12] for name in CODE}


def worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    """One job: several episodes of one task in one environment, in a fresh process."""
    import traceback

    from episode import EpisodeSpec, make_env, run_episode
    from xvla_record import load_episode, load_index

    rows = []
    try:
        env = make_env(job["suite"], job["task"])
    except Exception:  # noqa: BLE001
        return [dict(spec, success=False, status="fail:env_build", reason="env_build", steps=0, attempts=0,
                     error=traceback.format_exc()) for spec in job["specs"]]
    indexes: dict[str, Any] = {}
    for raw in job["specs"]:
        spec = EpisodeSpec(**raw)
        recording = None
        if spec.scenario == "xvla":
            run_dir = Path(spec.xvla_run)
            index = indexes.setdefault(spec.xvla_run, load_index(run_dir))
            recording = load_episode(run_dir, spec.xvla_episode, index)
        rows.append(run_episode(env, spec, recording))
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return rows


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    if args.scenario == "xvla":
        with open(args.handovers) as f:
            for line in f:
                h = json.loads(line)
                specs.append(dict(suite=h["suite"], task=h["task"], init=h["init"], scene_seed=h["scene_seed"],
                                  scenario="xvla", control_mode=args.mode, restore=not args.no_restore,
                                  xvla_run=h["run_dir"], xvla_episode=h["episode_id"],
                                  handover_step=h["handover_step"], tag=h.get("tag", ""),
                                  extra={k: h[k] for k in ("faults", "xvla_success") if k in h}))
        return specs
    for suite in args.suites.split(","):
        for task in parse_range(args.tasks):
            for init in parse_range(args.inits):
                specs.append(dict(suite=suite, task=task, init=init, scene_seed=args.seed_base + init,
                                  scenario=args.scenario, control_mode=args.mode,
                                  restore=not args.no_restore and args.scenario != "init"))
    return specs


def jobs_of(specs: list[dict[str, Any]], chunk: int) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in specs:
        by_task.setdefault((s["suite"], s["task"]), []).append(s)
    jobs = []
    for (suite, task), group in sorted(by_task.items()):
        for k in range(0, len(group), chunk):
            jobs.append(dict(suite=suite, task=task, specs=group[k : k + chunk]))
    return jobs


def summarize(rows: list[dict[str, Any]], by: tuple[str, ...] = ("suite",)) -> str:
    import numpy as np

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
    lines = []
    for key in sorted(groups, key=str):
        rs = [r for r in groups[key] if not str(r["status"]).startswith("skipped")]
        ok = [r for r in rs if r["success"]]
        steps = [r["steps"] for r in ok]
        reasons: dict[str, int] = {}
        for r in rs:
            if not r["success"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        lines.append(
            f"{' '.join(str(k) for k in key):28s} {len(ok):4d}/{len(rs):<4d} skipped {len(groups[key]) - len(rs):3d} "
            f"first-try {sum(1 for r in ok if r['attempts'] == 1):4d} "
            f"steps median {int(np.median(steps)) if steps else 0:4d} max {max(steps) if steps else 0:4d}  "
            + " ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scenario", default="init")
    ap.add_argument("--suites", default="libero_spatial,libero_object")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--mode", default="absolute", choices=("absolute", "relative"))
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--handovers", default=None, help="for --scenario xvla: jsonl from xvla_select.py")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=10, help="episodes per worker process")
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("at most four simulation processes in a light job")
    specs = build_specs(args)
    jobs = jobs_of(specs, args.chunk)
    out = OUT / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.jsonl"
    meta = dict(name=args.name, argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=len(specs), jobs=len(jobs), workers=args.workers, code=code_hashes())
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, open(path, "w") as f:
        for part in pool.imap_unordered(worker, jobs, chunksize=1):
            for r in part:
                f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            rows.extend(part)
    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1),
                load_avg=open("/proc/loadavg").read().split()[:3])
    (out / f"{args.name}.run.json").write_text(json.dumps(meta, indent=1))
    print(f"# {args.name}: {len(rows)} episodes in {time.time() - t0:.0f}s -> {path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
