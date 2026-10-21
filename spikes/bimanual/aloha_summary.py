"""Summarise aloha_eval.py runs and compare their hashes.

Usage: python aloha_summary.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR holds episodes.jsonl and meta.json. Prints success, timing and, for
every pair of runs, how many shared seeds have identical observation, action and
final-state hashes.
"""

import json
import statistics as st
import sys
from pathlib import Path


def load(d):
    eps = [json.loads(line) for line in open(Path(d) / "episodes.jsonl")]
    meta = json.loads((Path(d) / "meta.json").read_text())
    return eps, meta


def summary(d):
    eps, meta = load(d)
    n = len(eps)
    succ = sum(e["success"] for e in eps)
    calls = [c for e in eps for c in e["model_call_seconds"]]
    steps = sum(e["steps"] for e in eps)
    secs = [e["seconds"] for e in eps]
    print(f"== {d}  ({meta['task']}, threads {meta['threads']}, top_only {meta.get('top_only')})")
    print(f"   success {succ}/{n} = {100 * succ / n:.0f}%")
    print(f"   seconds per episode: mean {st.mean(secs):.2f}, median {st.median(secs):.2f}")
    print(f"   mean steps per episode {steps / n:.0f}")
    print(f"   env ms per step {1000 * sum(e['seconds_env'] for e in eps) / steps:.1f}")
    print(f"   policy ms per step (incl. queue pops) {1000 * sum(e['seconds_policy'] for e in eps) / steps:.2f}")
    print(f"   model calls {len(calls)}: median {1000 * st.median(calls):.0f} ms, "
          f"p90 {1000 * sorted(calls)[int(0.9 * len(calls))]:.0f} ms")
    print(f"   load avg start {meta['loadavg_start'][0]:.1f}, end {meta['loadavg_end'][0]:.1f}; "
          f"mujoco {meta['mujoco']}, torch {meta['torch']}")
    return eps


def compare(a_name, a, b_name, b):
    bi = {}
    for e in b:
        bi.setdefault(e["seed"], []).append(e)
    shared = same = 0
    diffs = []
    for e in a:
        for f in bi.get(e["seed"], []):
            shared += 1
            keys = ["first_obs_sha256", "obs_sha256", "act_sha256", "final_state_sha256", "success", "steps"]
            if all(e[k] == f[k] for k in keys):
                same += 1
            else:
                diffs.append((e["seed"], [k for k in keys if e[k] != f[k]]))
    print(f"   {a_name} vs {b_name}: {same}/{shared} episode pairs identical", end="")
    print(f"; differing: {diffs[:5]}" if diffs else "")


def main():
    runs = {d: summary(d) for d in sys.argv[1:]}
    task = {d: load(d)[1]["task"] for d in runs}
    names = list(runs)
    print("== hash comparison (runs of the same task)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if runs[a] and runs[b] and task[a] == task[b]:
                compare(a, runs[a], b, runs[b])


if __name__ == "__main__":
    main()
