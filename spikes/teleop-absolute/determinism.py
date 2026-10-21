"""Is a teleoperator episode exactly repeatable? Compares re-runs with the campaign's records.

    run.sh determinism.py [--per-file 12]

Every campaign row carries `chain`, a SHA-256 over every action the expert sent
and the complete simulator state after every step (MuJoCo's integration state,
the contact solver's warm start included), and the final state's hash. This
script draws a sample of campaign episodes (successes and failures from every
batch) and runs them again in new worker processes:

* `repeat`: the same episodes, in reverse order and in smaller groups, so each
  runs in another process after other episodes than in the campaign;
* `direct`: the take-overs without the snapshot: the expert takes over in the
  environment the set-up or X-VLA left, without saving, resetting elsewhere
  and restoring first.

Both must reproduce the campaign's chain bit for bit. Results go to
results/det_repeat.jsonl, results/det_direct.jsonl and results/det_summary.json.

    run.sh determinism.py --prefix recheck --per-file 4 --all-failures --repeat-only

checks a code change against the campaign instead: every failed episode plus a
sample of successes, `repeat` only, written under the prefix. It also counts
outcomes that changed, separately from chains that changed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate import OUT, jobs_of, worker

SPEC_FIELDS = ("suite", "task", "init", "scene_seed", "scenario", "control_mode", "restore", "xvla_run",
               "xvla_episode", "handover_step", "tag", "extra")
KEY_FIELDS = ("suite", "task", "init", "scene_seed", "scenario", "control_mode", "xvla_run", "xvla_episode",
              "handover_step")
COMPARED = ("chain", "chain_steps", "final_state_hash", "success", "steps")


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in KEY_FIELDS)


def load(name: str) -> list[dict[str, Any]]:
    path = OUT / "results" / f"{name}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def sample(per_file: int, all_failures: bool = False) -> list[dict[str, Any]]:
    rng = np.random.default_rng(11)
    picked = []
    for path in sorted((OUT / "results").glob("c_*.jsonl")):
        rows = [r for r in load(path.stem) if not str(r["status"]).startswith("skipped")]
        failed = [r for r in rows if not r["success"]]
        good = [r for r in rows if r["success"]]
        n_fail = len(failed) if all_failures else min(len(failed), max(2, per_file // 4))
        if all_failures:
            picked.extend(dict(r, source=path.stem) for r in failed)
            n_fail, failed = 0, []
        for group, n in ((failed, n_fail), (good, per_file - n_fail)):
            for i in rng.choice(len(group), size=min(n, len(group)), replace=False):
                picked.append(dict(group[int(i)], source=path.stem))
    return picked


def run(specs: list[dict[str, Any]], chunk: int, reverse: bool) -> list[dict[str, Any]]:
    jobs = jobs_of(specs, chunk)
    if reverse:
        jobs = [dict(j, specs=list(reversed(j["specs"]))) for j in reversed(jobs)]
    rows = []
    with mp.get_context("spawn").Pool(4, maxtasksperchild=1) as pool:
        for part in pool.imap_unordered(worker, jobs, chunksize=1):
            rows.extend(part)
    return rows


def compare(reference: list[dict[str, Any]], rerun: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {key(r): r for r in rerun}
    same, differ = 0, []
    for ref in reference:
        new = by_key.get(key(ref))
        diffs = [f for f in COMPARED if new is None or new.get(f) != ref.get(f)]
        if diffs:
            differ.append(dict(source=ref["source"], key=key(ref), fields=diffs))
        else:
            same += 1
    outcome_changed = [dict(source=r["source"], key=key(r), before=r["success"],
                            after=by_key.get(key(r), {}).get("success"))
                       for r in reference if by_key.get(key(r), {}).get("success") != r["success"]]
    return dict(compared=len(reference), identical=same, differing=differ, outcome_changed=outcome_changed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=12)
    ap.add_argument("--prefix", default="det")
    ap.add_argument("--all-failures", action="store_true")
    ap.add_argument("--repeat-only", action="store_true")
    args = ap.parse_args()
    picked = sample(args.per_file, args.all_failures)
    specs = [{k: r[k] for k in SPEC_FIELDS} for r in picked]
    t0 = time.time()
    repeat = run(specs, chunk=3, reverse=True)
    takeovers = [r for r in picked if r["scenario"] != "init"] if not args.repeat_only else []
    direct_specs = [dict({k: r[k] for k in SPEC_FIELDS}, restore=False) for r in takeovers]
    direct = run(direct_specs, chunk=3, reverse=False) if direct_specs else []
    summary: dict[str, Any] = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1))
    summary["repeat"] = compare(picked, repeat)
    summary["direct"] = compare(takeovers, direct)
    per_source: dict[str, list[int]] = {}
    rerun_by_key = {key(r): r for r in repeat}
    for r in picked:
        new = rerun_by_key.get(key(r), {})
        entry = per_source.setdefault(r["source"], [0, 0, 0])
        entry[0] += 1
        entry[1] += int(all(new.get(f) == r.get(f) for f in COMPARED))
        entry[2] += int(r["success"])
    summary["repeat_by_source"] = {k: dict(episodes=v[0], identical=v[1], successes=v[2])
                                   for k, v in per_source.items()}
    summary["actions_and_states_compared"] = int(sum(r.get("chain_steps", 0) for r in picked))
    out = OUT / "results"
    for name, rows in (("repeat", repeat), ("direct", direct)):
        with open(out / f"{args.prefix}_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    Path(out / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "repeat_by_source"}, indent=1, default=str))
    for k, v in summary["repeat_by_source"].items():
        print(k, v)


if __name__ == "__main__":
    main()
