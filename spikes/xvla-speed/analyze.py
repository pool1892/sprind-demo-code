"""Tables from finished fleet runs. Reads only files; needs numpy.

    python analyze.py summary RUN_DIR...        speed, rows in use and memory, one line per run
    python analyze.py compare A_DIR B_DIR       are episodes with the same identity bit-identical?
    python analyze.py pairs A_DIR B_DIR         the same episodes under two inference settings: successes side by side
    python analyze.py paired LABEL A_DIRS B_DIRS   the same, as one table row with an exact paired test, an interval
                                                for the difference and the smallest difference the sample could show
                                                (A is the 32-bit run or runs, B the 16-bit ones; comma-separated)
    python analyze.py bytask A_DIRS B_DIRS      the paired comparison task by task
    python analyze.py tasks RUN_DIR...          successes by task

Adapted from spikes/skeleton/analyze.py. Two episodes are "the same" when their
identity agrees: robot, suite, task, start state, scene seed, noise seed, step
limit, fault set, batch size and inference settings. The row is no longer part
of it. `compare --ignore-policy` leaves the inference settings out of the
identity (to compare with the skeleton's runs, which do not record them), and
`pairs` leaves out the batch size as well, since it compares outcomes, not bits.

"Identical" means that every stored array is the same bytes: the hash of the
simulator state and of both camera images after every step, the robot state in
full precision, the commanded and the executed actions, the fixture layout, and
with them the length and the outcome.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CORE = ("robot", "suite", "task", "init", "scene_seed", "noise_seed", "max_steps")
ARRAYS = ("fixtures", "state_hash", "img_hash", "proprio", "cmd", "exe")
RUNS = Path.home() / ".cache" / "robots-probes" / "xvla-speed" / "runs"


def run_dir(name: str) -> Path:
    p = Path(name)
    return p if p.exists() else RUNS / name


def load(run_dirs) -> list[dict]:
    out = []
    for d in run_dirs:
        d = run_dir(d)
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


def key(e, with_rows=True, with_policy=True) -> tuple:
    k = tuple(e[c] for c in CORE) + (fault_sig(e),)
    if with_rows:
        k += (e["rows"],)
    if with_policy:
        k += (e.get("policy"),)
    return k


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


def cmd_compare(a_dir, b_dir, ignore_policy=False):
    a_eps, b_eps = load(a_dir.split(",")), load(b_dir.split(","))
    b_by = defaultdict(list)
    for e in b_eps:
        b_by[key(e, with_policy=not ignore_policy)].append(e)
    rows = []
    for ea in a_eps:
        for eb in b_by.get(key(ea, with_policy=not ignore_policy), []):
            ca, cb = chain(ea), chain(eb)
            diffs = {k: first_diff(ca[k], cb[k]) for k in ARRAYS}
            identical = all(v == -1 for v in diffs.values()) and ea["success"] == eb["success"] \
                and ea["n_steps"] == eb["n_steps"]
            first = min([v for v in diffs.values() if v >= 0], default=-1)
            rows.append({"a": ea["episode_id"], "b": eb["episode_id"], "identical": identical, "first": first,
                         "flip": ea["success"] != eb["success"], "which": [k for k, v in diffs.items() if v >= 0],
                         "faults": bool(fault_sig(ea)),
                         "other_worker": (ea.get("worker"), ea.get("worker_episode")) != (eb.get("worker"), eb.get("worker_episode"))})
    bad = [r for r in rows if not r["identical"]]
    for r in bad[:10]:
        print(f"  differs: {r['a']} vs {r['b']}: first at step {r['first']} in {r['which']}, flip={r['flip']}")
    print(f"{a_dir} against {b_dir}: {len(rows)} pairs with the same identity "
          f"({sum(r['faults'] for r in rows)} with a fault, {sum(r['other_worker'] for r in rows)} run by another "
          f"worker slot or at another point of a worker's life), {len(rows) - len(bad)} identical, "
          f"{sum(r['flip'] for r in rows)} outcome flips")
    return rows


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided sign test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def cmd_pairs(a_dir, b_dir):
    a_eps, b_eps = load(a_dir.split(",")), load(b_dir.split(","))
    b_by = {key(e, False, False): e for e in b_eps}
    both = only_a = only_b = neither = same_len = 0
    steps_a, steps_b = [], []
    for ea in a_eps:
        eb = b_by.get(key(ea, False, False))
        if eb is None:
            continue
        sa, sb = ea["success"], eb["success"]
        both += sa and sb
        only_a += sa and not sb
        only_b += sb and not sa
        neither += (not sa) and (not sb)
        same_len += ea["n_steps"] == eb["n_steps"]
        steps_a.append(ea["n_steps"])
        steps_b.append(eb["n_steps"])
    n = both + only_a + only_b + neither
    ka, kb = both + only_a, both + only_b
    la, lb = wilson(ka, n), wilson(kb, n)
    print(f"{n} paired episodes. A = {a_dir}: {ka}/{n} ({100 * ka / n:.1f}%, 95% interval {100 * la[0]:.0f} to {100 * la[1]:.0f}%), "
          f"mean {np.mean(steps_a):.0f} steps. B = {b_dir}: {kb}/{n} ({100 * kb / n:.1f}%, {100 * lb[0]:.0f} to {100 * lb[1]:.0f}%), "
          f"mean {np.mean(steps_b):.0f} steps.")
    print(f"  both succeed {both}, only A {only_a}, only B {only_b}, neither {neither}; "
          f"exact two-sided p for a difference {mcnemar_p(only_a, only_b):.2f}; same length in {same_len} of {n}")


def log_binom_pmf(n: int, k: int, q: float) -> float:
    if q <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if q >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(q) + (n - k) * math.log(1.0 - q))


def binom_cdf(n: int, k: int, q: float) -> float:
    return min(1.0, sum(math.exp(log_binom_pmf(n, i, q)) for i in range(0, k + 1))) if k >= 0 else 0.0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for a binomial share, by bisection."""
    if n == 0:
        return (0.0, 1.0)

    def solve(target_fn, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if target_fn(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda q: 1.0 - binom_cdf(n, k - 1, q) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda q: binom_cdf(n, k, q) > alpha / 2)
    return (lower, upper)


def newcombe_paired(both: int, only_a: int, only_b: int, neither: int) -> tuple[float, float]:
    """95% interval for (B's success rate - A's) from paired outcomes.

    Newcombe's hybrid score interval for paired proportions (Statistics in Medicine 1998, method 10):
    Wilson intervals for the two rates, combined with the correlation between the paired outcomes.
    It stays sensible when few or no pairs differ.
    """
    n = both + only_a + only_b + neither
    ka, kb = both + only_a, both + only_b
    pa, pb = ka / n, kb / n
    la, ua = wilson(ka, n)
    lb, ub = wilson(kb, n)
    prod = ka * (n - ka) * kb * (n - kb)
    if prod == 0:
        phi = 0.0
    else:
        bq = both * neither - only_a * only_b
        c = bq - n / 2 if bq > n / 2 else (0.0 if bq >= 0 else bq)
        phi = c / math.sqrt(prod)
    theta = pb - pa
    delta = math.sqrt(max(0.0, (pb - lb) ** 2 - 2 * phi * (pb - lb) * (ua - pa) + (ua - pa) ** 2))
    eps = math.sqrt(max(0.0, (ub - pb) ** 2 - 2 * phi * (ub - pb) * (pa - la) + (pa - la) ** 2))
    return (theta - delta, theta + eps)


def reject_limit(m: int, alpha: float = 0.05) -> int:
    """Largest k such that a split of m differing pairs into k and m - k is significant (exact, two-sided); -1 if none."""
    k = -1
    while k + 1 <= m // 2 and mcnemar_p(k + 1, m - k - 1) <= alpha:
        k += 1
    return k


def power(n: int, flips: float, extra: float, alpha: float = 0.05) -> float:
    """Chance that the exact paired test at level alpha shows a difference, if format B truly fails `extra` more
    of the n episodes than A, on top of `flips` pairs that differ either way with equal chance."""
    p_a, p_b = flips / 2 + extra, flips / 2  # only A succeeds, only B succeeds
    psi = p_a + p_b
    if psi <= 0:
        return 0.0
    theta = p_a / psi
    total = 0.0
    for m in range(0, n + 1):
        w = math.exp(log_binom_pmf(n, m, psi))
        if w < 1e-12:
            continue
        k = reject_limit(m, alpha)
        if k < 0:
            continue
        # significant if "only A" is at most k or at least m - k
        total += w * (binom_cdf(m, k, theta) + 1.0 - binom_cdf(m, m - k - 1, theta))
    return total


def detectable(n: int, flips: float, want: float = 0.8) -> float:
    lo, hi = 0.0, 1.0 - flips
    for _ in range(40):
        mid = (lo + hi) / 2
        if power(n, flips, mid) < want:
            lo = mid
        else:
            hi = mid
    return hi


def cmd_paired(label, a_dir, b_dir, header=True):
    """One table row: A is the 32-bit run (or runs), B the 16-bit one, matched episode by episode."""
    a_eps, b_eps = load(a_dir.split(",")), load(b_dir.split(","))
    b_by = {key(e, False, False): e for e in b_eps}
    both = only_a = only_b = neither = same_len = 0
    steps_a, steps_b = [], []
    for ea in a_eps:
        eb = b_by.get(key(ea, False, False))
        if eb is None:
            continue
        assert ea["rows"] == eb["rows"], "the two formats must have run at the same batch size"
        sa, sb = ea["success"], eb["success"]
        both += sa and sb
        only_a += sa and not sb
        only_b += sb and not sa
        neither += (not sa) and (not sb)
        same_len += ea["n_steps"] == eb["n_steps"]
        steps_a.append(ea["n_steps"])
        steps_b.append(eb["n_steps"])
    n = both + only_a + only_b + neither
    m = only_a + only_b
    ka, kb = both + only_a, both + only_b
    # difference B - A in points, with an exact interval given the number of differing pairs
    diff = 100.0 * (only_b - only_a) / n
    lo_d, hi_d = (100.0 * x for x in newcombe_paired(both, only_a, only_b, neither))
    det = 100.0 * detectable(n, m / n)
    if header:
        print("| Case | Pairs | 32-bit | 16-bit | Both succeed | Only 32-bit | Only 16-bit | Neither | Exact paired test, p | "
              "16-bit minus 32-bit, points (95% interval) | Smallest loss this sample would have shown 4 times in 5 | "
              "Mean steps, 32 / 16 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(f"| {label} | {n} | {ka} ({100 * ka / n:.1f}%) | {kb} ({100 * kb / n:.1f}%) | {both} | {only_a} | {only_b} | "
          f"{neither} | {mcnemar_p(only_a, only_b):.2f} | {diff:+.1f} ({lo_d:+.1f} to {hi_d:+.1f}) | {det:.1f} points | "
          f"{np.mean(steps_a):.0f} / {np.mean(steps_b):.0f} |")
    return {"n": n, "a": ka, "b": kb, "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
            "p": mcnemar_p(only_a, only_b), "diff": diff, "ci": (lo_d, hi_d), "detectable": det}


def cmd_bytask(a_dir, b_dir):
    a_eps, b_eps = load(a_dir.split(",")), load(b_dir.split(","))
    b_by = {key(e, False, False): e for e in b_eps}
    rows: dict = defaultdict(lambda: [0, 0, 0, 0, 0])  # pairs, A, B, only A, only B
    for ea in a_eps:
        eb = b_by.get(key(ea, False, False))
        if eb is None:
            continue
        r = rows[ea["task"]]
        r[0] += 1
        r[1] += ea["success"]
        r[2] += eb["success"]
        r[3] += ea["success"] and not eb["success"]
        r[4] += eb["success"] and not ea["success"]
    print("| Task | " + " | ".join(str(t) for t in sorted(rows)) + " | All |")
    print("|---|" + "---|" * (len(rows) + 1))
    for name, i in (("Pairs", 0), ("32-bit successes", 1), ("16-bit successes", 2), ("Only 32-bit", 3), ("Only 16-bit", 4)):
        vals = [rows[t][i] for t in sorted(rows)]
        print(f"| {name} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")


def cmd_tasks(dirs):
    print("| Run | Successes | Rate | 95% interval | Mean steps | Chunks per episode | Successes by task 0 to 9 |")
    print("|---|---|---|---|---|---|---|")
    for d in dirs:
        es = load([d])
        k, n = sum(e["success"] for e in es), len(es)
        lo, hi = wilson(k, n)
        per = [sum(e["success"] for e in es if e["task"] == t) for t in range(10)]
        tot = [sum(1 for e in es if e["task"] == t) for t in range(10)]
        print(f"| `{Path(d).name}` | {k}/{n} | {100 * k / n:.1f}% | {100 * lo:.0f} to {100 * hi:.0f}% | "
              f"{np.mean([e['n_steps'] for e in es]):.0f} | {np.mean([e['chunks'] for e in es]):.1f} | "
              f"{' '.join(f'{a}/{b}' for a, b in zip(per, tot))} |")


def cmd_summary(dirs):
    print("| Run | Policy | Rows | Workers | Video | Episodes | Successes | Per hour, end to end | Steady state | "
          "Steps / chunks per episode | Call, ms | Rows in use (middle of run) | Server busy | Workers waiting | "
          "Process memory peak | GPU memory: these processes / machine | Least memory available | Load at start |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        r = json.loads((run_dir(d) / "run.json").read_text())
        p = r["peaks"]
        print(f"| `{run_dir(d).name}` | {r['policy']} | {r['rows']} | {r['workers']} | {'yes' if r['video'] else 'no'} | "
              f"{r['episodes']} | {r['successes']} | {r['episodes_per_hour_overall']:,.0f} | {r['episodes_per_hour_steady']:,.0f} | "
              f"{r['mean_steps']:.0f} / {r['mean_chunks']:.1f} | {1000 * r['inference_s_per_call']:.0f} | "
              f"{100 * r['share_of_rows_in_use']:.0f}% ({100 * r['share_of_rows_in_use_middle']:.0f}%) | "
              f"{100 * r['server_busy_fraction']:.0f}% | {100 * r['worker_time_share']['wait']:.0f}% | "
              f"{p['pss_total_mib'] / 1024:.1f} GiB | {p['gpu_own_total_mib'] / 1024:.1f} / {p['gtt_gib']:.1f} GiB | "
              f"{p['mem_available_min_gib']:.0f} GiB | {r['load_start'].split()[0]} |")


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "compare":
        ignore = "--ignore-policy" in args
        args = [a for a in args if a != "--ignore-policy"]
        cmd_compare(args[0], args[1], ignore)
    elif cmd == "pairs":
        cmd_pairs(args[0], args[1])
    elif cmd == "paired":
        # paired LABEL A B [LABEL A B ...]
        for i in range(0, len(args), 3):
            cmd_paired(args[i], args[i + 1], args[i + 2], header=i == 0)
    elif cmd == "bytask":
        cmd_bytask(args[0], args[1])
    elif cmd == "tasks":
        cmd_tasks(args)
    elif cmd == "summary":
        cmd_summary(args)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
