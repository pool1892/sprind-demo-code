"""Batch evaluation of the scripted expert.

    run.sh evaluate.py --name NAME [--suites a,b] [--tasks 0-9] [--inits 0-9]
                       [--scenario init] [--fault none] [--workers 4] [--set key=value ...]

One row per episode goes to ~/.cache/robots-probes/teleop/results/NAME.jsonl and a
summary table is printed. Use summarize.py to print the table again later.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback

from common import OUT


def parse_range(text, n):
    if text == "all":
        return list(range(n))
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def worker(job):
    """One job is a handful of episodes of one task, run in a fresh process."""
    import numpy as np

    from common import make_env_bddl, reset_to_init
    from disturb import apply_fault, apply_scenario
    from runner import failure_reason, run_expert

    suite, tid, language, bddl = job["suite"], job["task"], job["language"], job["bddl"]
    scenario, fault, seed, name = job["scenario"], job["fault"], job["seed"], job["name"]
    rows = []
    try:
        env = make_env_bddl(bddl, renderer=bool(job["video_ids"]))
    except Exception:  # noqa: BLE001
        return [dict(suite=suite, task=tid, init=i, scenario=scenario, fault=fault, success=False,
                     status="fail:env_build", reason="env_build", steps=0, attempts=0, error=traceback.format_exc())
                for i, _ in job["inits"]]
    for iid, init_state in job["inits"]:
        t0 = time.time()
        row = dict(suite=suite, task=tid, init=iid, scenario=scenario, fault=fault, language=language)
        try:
            reset_to_init(env, init_state, seed=seed + iid)
            rng = np.random.default_rng([seed, tid, iid, sum(map(ord, suite + scenario))])
            row["fault_info"] = apply_fault(env, fault)
            setup = apply_scenario(env, scenario, rng)
            row["setup"] = setup
            if setup.get("invalid"):
                row.update(success=False, status="skipped:" + setup["invalid"], reason="setup_invalid", steps=0, attempts=0)
            else:
                video = None
                if iid in job["video_ids"]:
                    d = OUT / "videos" / name
                    d.mkdir(parents=True, exist_ok=True)
                    video = str(d / f"{suite}_t{tid}_i{iid}_{scenario}_{fault}.mp4")
                res = run_expert(env, video=video, expert_kwargs=job["expert_kwargs"])
                res["reason"] = failure_reason(res)
                row.update(res)
        except Exception:  # noqa: BLE001
            row.update(success=False, status="fail:harness_exception", reason="harness_exception", steps=0, attempts=0,
                       error=traceback.format_exc())
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return rows


def summarize(rows, by=("suite", "task")):
    import numpy as np

    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in by), []).append(r)
    lines = []
    for key in sorted(groups):
        rs = [r for r in groups[key] if not r["status"].startswith("skipped")]
        skipped = len(groups[key]) - len(rs)
        ok = [r for r in rs if r["success"]]
        steps = [r["steps"] for r in ok]
        reasons = {}
        for r in rs:
            if not r["success"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        first = sum(1 for r in ok if r["attempts"] == 1)
        lines.append(
            f"{' '.join(str(k) for k in key):28s} {len(ok):3d}/{len(rs):<3d} first-try {first:3d}  "
            f"steps median {int(np.median(steps)) if steps else 0:4d} max {max(steps) if steps else 0:4d}  "
            f"{'skipped ' + str(skipped) + '  ' if skipped else ''}"
            f"{' '.join(f'{k}:{v}' for k, v in sorted(reasons.items()))}"
        )
    rs = [r for r in rows if not r["status"].startswith("skipped")]
    ok = [r for r in rs if r["success"]]
    lines.append(f"{'TOTAL':28s} {len(ok):3d}/{len(rs):<3d} first-try {sum(1 for r in ok if r['attempts'] == 1):3d}  "
                 f"steps median {int(np.median([r['steps'] for r in ok])) if ok else 0:4d}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--suites", default="libero_spatial,libero_object")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--scenario", default="init")
    ap.add_argument("--fault", default="none")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=5, help="episodes per worker process")
    ap.add_argument("--videos", default="", help="init ids to record, e.g. 0,3")
    ap.add_argument("--set", action="append", default=[], help="expert option, e.g. max_steps=800")
    args = ap.parse_args()
    assert args.workers <= 4, "at most four simulator processes on this machine"

    expert_kwargs = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        expert_kwargs[k] = json.loads(v)
    video_ids = set(parse_range(args.videos, 50)) if args.videos else set()
    inits = parse_range(args.inits, 50)
    from common import bddl_path, load_init_states, suite_tasks

    jobs = []
    for suite_name in args.suites.split(","):
        suite, tasks = suite_tasks(suite_name)
        for tid in parse_range(args.tasks, 10):
            states = load_init_states(suite, tid)
            # Small jobs, one fresh process each: an environment process grows by 30 to 60 MB
            # per reset, and four of them have to stay under the 8 GB ceiling together.
            for k in range(0, len(inits), args.chunk):
                chunk = inits[k:k + args.chunk]
                jobs.append(dict(suite=suite_name, task=tid, language=tasks[tid].language, bddl=bddl_path(suite, tid),
                                 inits=[(i, states[i % len(states)]) for i in chunk], scenario=args.scenario,
                                 fault=args.fault, expert_kwargs=expert_kwargs, seed=args.seed,
                                 video_ids=video_ids, name=args.name))
    t0 = time.time()
    rows = []
    out = OUT / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.jsonl"
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool, open(path, "w") as f:
        for part in pool.imap_unordered(worker, jobs, chunksize=1):
            for r in part:
                f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            rows.extend(part)
    print(f"# {args.name}: scenario={args.scenario} fault={args.fault} expert={expert_kwargs} "
          f"episodes={len(rows)} wall={time.time()-t0:.0f}s -> {path}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
