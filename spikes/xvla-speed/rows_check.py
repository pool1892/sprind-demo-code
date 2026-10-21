"""Does a row's chunk depend on where it sits in the batch, or on what the other rows hold?

    MIOPEN_DB=pinned tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/rows_check.py --tag fp16 --sizes 4 6 8 --opts '{"half": {...}, "benchmark": false}'

A "content" is one recorded observation with its instruction and its own starting
noise. For each batch size N the 80 recorded contents are laid out as base batches
B_0, B_1, ... of N rows each, and every content's chunk in its base batch is the
reference. Then, always at the same batch size:

  position   the base batch rotated by a few rows: every content sits in another
             row, with the same batch mates
  mates      row i of the batch holds row i of base batch k+i: every content keeps
             its row and gets other mates in every other row (for N above the
             number of base batches, in all but one or two)
  both       the two combined: another row and other mates
  padded     the first half of a base batch, the other rows filled with copies of
             row 0, which is how the fleet runner fills rows that nobody asked for

A comparison counts as identical only if all 30 x 20 numbers of the chunk are the
same bits. Last, the effect of the batch size itself: the first 16 contents at
sizes 1 to 16 against the chunk they get at the first size in --sizes.

Writes ~/.cache/robots-probes/xvla-speed/results/rows_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import xs_common as C
from policy_fast import bits_equal, load_obs

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8])
    ap.add_argument("--effect-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="xvla", choices=["xvla", "smolvla"])
    args = ap.parse_args()
    opts = json.loads(args.opts)
    obs, tasks = load_obs()
    n_obs = len(obs)
    if args.model == "smolvla":
        from policy_smolvla import Policy
    else:
        from policy_fast import Policy
    pol = Policy("cuda", opts=opts)
    report: dict = {"tag": args.tag, "model": args.model, "opts": opts, "miopen_db": C.MIOPEN_DB, "load_start": C.loadavg(),
                    "gtt_start_gib": C.gtt_gib(), "contents": n_obs, "by_size": {}}
    calls = 0

    call_s: dict[int, list] = {}

    def run(contents: list[int]) -> np.ndarray:
        nonlocal calls
        calls += 1
        batch = pol.batch([obs[c] for c in contents], [tasks[c] for c in contents])
        noise = torch.stack([pol.noise(9000 + c, 0) for c in contents])
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = pol.chunks_staged(batch, noise).numpy()
        call_s.setdefault(len(contents), []).append(time.perf_counter() - s)
        return out

    def tally(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        same = sum(bits_equal(a, b) for a, b in pairs)
        worst = max((float(np.abs(a.astype(np.float64) - b).max()) for a, b in pairs), default=0.0)
        return {"comparisons": len(pairs), "identical": same, "largest_abs_difference": worst}

    t0 = time.perf_counter()
    for n in args.sizes:
        k_base = n_obs // n
        base = [[k * n + i for i in range(n)] for k in range(k_base)]
        ref: dict[int, np.ndarray] = {}
        for b in base:
            out = run(b)
            for i, c in enumerate(b):
                ref[c] = out[i]
        res = {"base_batches": k_base}
        # the same batch once more: a call repeats
        again = run(base[0])
        res["same_batch_again"] = tally([(again[i], ref[c]) for i, c in enumerate(base[0])])
        # position: rotate each base batch by 1..n-1 rows
        pairs = []
        for k, b in enumerate(base):
            s = 1 + k % (n - 1) if n > 1 else 0
            rot = b[s:] + b[:s]
            out = run(rot)
            pairs += [(out[i], ref[c]) for i, c in enumerate(rot)]
        res["position"] = tally(pairs)
        # mates: row i from base batch k+i
        pairs = []
        for k in range(k_base):
            mix = [base[(k + i) % k_base][i] for i in range(n)]
            out = run(mix)
            pairs += [(out[i], ref[c]) for i, c in enumerate(mix)]
        res["mates"] = tally(pairs)
        # both: row i holds row (i + s) of base batch k+i
        pairs = []
        for k in range(k_base):
            s = 1 + k % (n - 1) if n > 1 else 0
            mix = [base[(k + i) % k_base][(i + s) % n] for i in range(n)]
            out = run(mix)
            pairs += [(out[i], ref[c]) for i, c in enumerate(mix)]
        res["both"] = tally(pairs)
        # padded: first half real, the rest copies of row 0
        pairs = []
        half = max(1, n // 2)
        for b in base:
            pad = b[:half] + [b[0]] * (n - half)
            out = run(pad)
            pairs += [(out[i], ref[c]) for i, c in enumerate(pad[:half])]
            pairs += [(out[i], ref[b[0]]) for i in range(half, n)]  # the copies themselves
        res["padded"] = tally(pairs)
        report["by_size"][n] = res
        print(f"batch size {n}: " + ", ".join(
            f"{k} {v['identical']}/{v['comparisons']}" + ("" if v["identical"] == v["comparisons"] else f" (up to {v['largest_abs_difference']:.1e})")
            for k, v in res.items() if isinstance(v, dict)), f"| {time.perf_counter() - t0:.0f} s, load {C.loadavg()}", flush=True)

    # the batch size itself
    n0 = args.sizes[0]
    ref0 = {}
    for k in range(0, 16, n0):
        b = [c for c in range(k, k + n0)]
        out = run(b)
        for i, c in enumerate(b):
            if c < 16:
                ref0[c] = out[i]
    effect = {}
    for n in args.effect_sizes:
        pairs = []
        for k in range(0, 16, n):
            b = [c % n_obs for c in range(k, k + n)]
            out = run(b)
            pairs += [(out[i], ref0[c]) for i, c in enumerate(b) if c in ref0]
        effect[n] = tally(pairs)
        effect[n]["mean_abs_difference"] = float(np.mean([np.abs(a.astype(np.float64) - b).mean() for a, b in pairs]))
    report["batch_size_effect"] = {"against_size": n0, "by_size": effect}
    print(f"batch size effect against size {n0}: " + ", ".join(
        f"{n}: {v['identical']}/{v['comparisons']} identical, up to {v['largest_abs_difference']:.1e}" for n, v in effect.items()), flush=True)
    report["calls"] = calls
    report["call_ms_median_by_size"] = {n: round(1000 * float(np.median(v[1:] or v)), 1) for n, v in sorted(call_s.items())}
    print("call time by batch size, ms:", report["call_ms_median_by_size"])
    report["load_end"] = C.loadavg()
    report["seconds"] = time.perf_counter() - t0
    (C.OUT / "results" / f"rows_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"wrote results/rows_{args.tag}.json; {calls} calls, load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
