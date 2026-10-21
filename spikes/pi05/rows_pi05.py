"""Question 3, policy alone: does a chunk repeat across processes, rows and batch mates?

    tools/heavy-run ~/.cache/robots-probes/pi05/venv/bin/python spikes/pi05/rows_pi05.py \
        --tag bf16 --sizes 4 8 --opts '{"dtype": "bfloat16"}' [--against TAG]

Adapted from spikes/xvla-speed/rows_check.py. A "content" is one of the 80
recorded observations with its instruction and its own starting noise (noise seed
9000 + content, chunk 0), passed to LeRobot's `predict_action_chunk(noise=...)`.
For each batch size N the contents are laid out as base batches of N rows, and
every content's chunk in its base batch is the reference. Then, always at the
same batch size:

  same_batch_again  the first base batch once more, in the same process
  position          the base batch rotated by a few rows: every content in another row, same mates
  mates             every content keeps its row and gets other mates in every other row
  both              the two combined
  padded            the first half of a base batch, the rest copies of row 0 (how the fleet fills rows)

and the batch size itself: the first 16 contents at sizes 1 to 16 against their
chunk at the first size in --sizes.

A comparison counts as identical only if all 50 x 7 numbers of the chunk are the
same bits. `--against TAG` compares this process's reference chunks with those
another process saved (`refs_<TAG>.npz`): the same contents, batch size and
rows, another process. Writes results/rows_<tag>.json and results/refs_<tag>.npz.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import p5_common as C
from policy_pi05 import Policy, bits_equal, load_obs

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--effect-sizes", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--against", default=None)
    ap.add_argument("--contents", type=int, default=80)
    args = ap.parse_args()
    opts = json.loads(args.opts)
    obs, tasks = load_obs(args.contents)
    n_obs = len(obs)
    pol = Policy("cuda", opts=opts)
    report: dict = {"tag": args.tag, "opts": opts, "miopen_db": C.MIOPEN_DB, "load_start": C.loadavg(),
                    "gtt_start_gib": C.gtt_gib(), "contents": n_obs, "by_size": {}, "n_linear_t": pol.n_linear_t,
                    "blas_library": pol.blas_library}
    calls = 0
    call_s: dict[int, list] = {}
    refs_out: dict[str, np.ndarray] = {}

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

    def tally(pairs) -> dict:
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
        refs_out[f"size{n}"] = np.stack([ref[c] for c in sorted(ref)])
        res = {"base_batches": k_base}
        again = run(base[0])
        res["same_batch_again"] = tally([(again[i], ref[c]) for i, c in enumerate(base[0])])
        pairs = []
        for k, b in enumerate(base):
            s = 1 + k % (n - 1) if n > 1 else 0
            rot = b[s:] + b[:s]
            out = run(rot)
            pairs += [(out[i], ref[c]) for i, c in enumerate(rot)]
        res["position"] = tally(pairs)
        pairs = []
        for k in range(k_base):
            mix = [base[(k + i) % k_base][i] for i in range(n)]
            out = run(mix)
            pairs += [(out[i], ref[c]) for i, c in enumerate(mix)]
        res["mates"] = tally(pairs)
        pairs = []
        for k in range(k_base):
            s = 1 + k % (n - 1) if n > 1 else 0
            mix = [base[(k + i) % k_base][(i + s) % n] for i in range(n)]
            out = run(mix)
            pairs += [(out[i], ref[c]) for i, c in enumerate(mix)]
        res["both"] = tally(pairs)
        pairs = []
        half = max(1, n // 2)
        for b in base:
            pad = b[:half] + [b[0]] * (n - half)
            out = run(pad)
            pairs += [(out[i], ref[c]) for i, c in enumerate(pad[:half])]
            pairs += [(out[i], ref[b[0]]) for i in range(half, n)]
        res["padded"] = tally(pairs)
        report["by_size"][n] = res
        print(f"batch size {n}: " + ", ".join(
            f"{k} {v['identical']}/{v['comparisons']}" + ("" if v["identical"] == v["comparisons"] else f" (up to {v['largest_abs_difference']:.1e})")
            for k, v in res.items() if isinstance(v, dict)), f"| {time.perf_counter() - t0:.0f} s, load {C.loadavg()}", flush=True)

    if args.effect_sizes:
        n0 = args.sizes[0]
        ref0 = {}
        for k in range(0, 16, n0):
            b = list(range(k, k + n0))
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
        report["batch_size_effect"] = {"against_size": n0, "by_size": effect}
        print(f"batch size effect against size {n0}: " + ", ".join(
            f"{n}: {v['identical']}/{v['comparisons']} identical, up to {v['largest_abs_difference']:.1e}"
            for n, v in effect.items()), flush=True)

    np.savez(C.OUT / "results" / f"refs_{args.tag}.npz", **refs_out)
    if args.against:
        other = np.load(C.OUT / "results" / f"refs_{args.against}.npz")
        cross = {}
        for k in refs_out:
            if k in other:
                a, b = refs_out[k], other[k]
                cross[k] = tally([(a[i], b[i]) for i in range(min(len(a), len(b)))])
        report["other_process"] = {"against": args.against, **cross}
        print("against another process", args.against, json.dumps(cross), flush=True)
    report["calls"] = calls
    report["call_ms_median_by_size"] = {n: round(1000 * float(np.median(v[1:] or v)), 1) for n, v in sorted(call_s.items())}
    report["load_end"] = C.loadavg()
    report["seconds"] = time.perf_counter() - t0
    (C.OUT / "results" / f"rows_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"wrote results/rows_{args.tag}.json; {calls} calls, {report['seconds']:.0f} s, load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
