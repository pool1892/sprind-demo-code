"""Time one way of running X-VLA and record its output, one fresh process per variant.

    tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python \
        spikes/xvla-speed/variants.py --suite today,today@2,hoist,t_fp16 [--batch 8]

    python spikes/xvla-speed/variants.py --table          # no GPU: compare what was recorded

A variant is a set of `Policy` options (see policy_fast.py) plus, optionally,
environment variables for MIOpen. `NAME@2` runs NAME again and stores it under
that name, which is how "repeats in a fresh process" is checked. The suite runs
its variants one after the other, each in its own process, so only one copy of the
model is ever on the GPU.

Per variant: the same 24 recorded observations as three batches of eight with
fixed noise; the first batch is timed stage by stage (median of --reps calls after
two warm-ups) and computed once more at the end to see whether a call repeats
inside the process. Stored: the 24 chunks and their environment actions.

Results: ~/.cache/robots-probes/xvla-speed/results/variants/NAME.{json,npz}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HALF_ALL = {"transformer": "fp16", "vision": "fp16", "encoder": "fp16"}
BF_ALL = {"transformer": "bf16", "vision": "bf16", "encoder": "bf16"}
ALL = ["vision", "encoder", "transformer"]

# name -> (Policy options, extra environment)
VARIANTS: dict[str, tuple[dict, dict]] = {
    "today": ({}, {}),
    # PyTorch settings
    "benchmark_off": ({"benchmark": False}, {}),
    "benchmark_off_empty_db": ({"benchmark": False}, {"MIOPEN_DB": "empty_immediate"}),
    "benchmark_off_no_winograd_db": ({"benchmark": False}, {"MIOPEN_DB": "no_winograd"}),  # reads the database the Winograd-free run stored
    "channels_last": ({"channels_last": True}, {}),
    "no_grad_only": ({}, {"XS_NO_INFERENCE_MODE": "1"}),
    "attn_manual": ({"attn": "manual"}, {}),
    "compile": ({"compile": "default"}, {}),
    "compile_reduce_overhead": ({"compile": "reduce-overhead"}, {}),
    # work moved out of the loop or out of the call
    "hoist": ({"hoist": True}, {}),
    "fast_post": ({"fast_post": True}, {}),
    "hoist_fast_post": ({"hoist": True, "fast_post": True}, {}),
    # MIOpen: which convolution algorithm, chosen how
    "miopen_empty_db": ({}, {"MIOPEN_DB": "empty_fast"}),               # heuristic choice, nothing stored yet
    "miopen_search": ({}, {"MIOPEN_DB": "tuned", "MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "3"}),  # measure every algorithm, tune, store
    "miopen_tuned": ({}, {"MIOPEN_DB": "tuned", "MIOPEN_FIND_MODE": "1"}),  # use what the search stored
    "miopen_tuned_fast": ({}, {"MIOPEN_DB": "tuned"}),                      # the same database, read in today's mode
    "miopen_no_winograd": ({}, {"MIOPEN_DB": "no_winograd", "MIOPEN_DEBUG_CONV_WINOGRAD": "0"}),  # forbid today's usual choice
    # 16-bit floats: automatic mixed precision (weights stay 32-bit) ...
    "amp_bf16": ({"amp": "bf16", "amp_parts": ALL}, {}),
    "amp_fp16": ({"amp": "fp16", "amp_parts": ALL}, {}),
    # ... and weights converted once
    "t_fp16": ({"half": {"transformer": "fp16"}}, {}),
    "t_bf16": ({"half": {"transformer": "bf16"}}, {}),
    "t_fp16_res32": ({"half": {"transformer": "fp16"}, "residual32": True}, {}),
    "t_bf16_res32": ({"half": {"transformer": "bf16"}, "residual32": True}, {}),
    "tv_fp16": ({"half": {"transformer": "fp16", "vision": "fp16"}}, {}),
    "all_fp16": ({"half": HALF_ALL}, {}),
    "all_bf16": ({"half": BF_ALL}, {}),
    "all_fp16_res32": ({"half": HALF_ALL, "residual32": True}, {}),
    "all_bf16_res32": ({"half": BF_ALL, "residual32": True}, {}),
    "tv_fp16_res32": ({"half": {"transformer": "fp16", "vision": "fp16"}, "residual32": True}, {}),
    "t_fp16_res32_hoist_fast_post": ({"half": {"transformer": "fp16"}, "residual32": True, "hoist": True, "fast_post": True}, {}),
    "all_fp16_res32_hoist_fast_post": ({"half": HALF_ALL, "residual32": True, "hoist": True, "fast_post": True}, {}),
    # the PyTorch settings again on top of the 16-bit model
    "all_fp16_benchmark_off": ({"half": HALF_ALL, "benchmark": False}, {}),
    "all_fp16_benchmark_off_empty_db": ({"half": HALF_ALL, "benchmark": False}, {"MIOPEN_DB": "empty_immediate16"}),
    "all_fp16_benchmark_off_no_winograd_db": ({"half": HALF_ALL, "benchmark": False}, {"MIOPEN_DB": "no_winograd16"}),
    "all_fp16_channels_last": ({"half": HALF_ALL, "channels_last": True}, {}),
    "all_fp16_attn_manual": ({"half": HALF_ALL, "attn": "manual"}, {}),
    "all_fp16_compile": ({"half": HALF_ALL, "compile": "default"}, {}),
    "all_fp16_hoist_fast_post": ({"half": HALF_ALL, "hoist": True, "fast_post": True}, {}),
    "all_fp16_empty_db": ({"half": HALF_ALL}, {"MIOPEN_DB": "empty_fast16"}),
    "all_fp16_no_winograd": ({"half": HALF_ALL}, {"MIOPEN_DB": "no_winograd16", "MIOPEN_DEBUG_CONV_WINOGRAD": "0"}),
    "all_fp16_search": ({"half": HALF_ALL}, {"MIOPEN_DB": "tuned16", "MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "3"}),
    "all_fp16_tuned": ({"half": HALF_ALL}, {"MIOPEN_DB": "tuned16", "MIOPEN_FIND_MODE": "1"}),
    # 16-bit with every linear layer's weight stored transposed (position-free, see rows_diag2.py)
    "all_fp16_lt": ({"half": HALF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False}, {"MIOPEN_DB": "pinned"}),
    "all_fp16_res32_lt": ({"half": HALF_ALL, "residual32": True, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False}, {"MIOPEN_DB": "pinned"}),
    "all_bf16_lt": ({"half": BF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False}, {"MIOPEN_DB": "pinned"}),
    "fp32_lt": ({"linear_t": True, "hoist": True, "fast_post": True, "benchmark": False}, {"MIOPEN_DB": "pinned"}),
    "fp32_pinned": ({"hoist": True, "fast_post": True, "benchmark": False}, {"MIOPEN_DB": "pinned"}),
    "all_fp16_lt_compile": ({"half": HALF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False,
                             "compile": "default"}, {"MIOPEN_DB": "pinned"}),
    # behaviour knobs, for the call time only (success rates come from the fleet runs)
    "steps_5": ({"steps": 5}, {}),
    "steps_20": ({"steps": 20}, {}),
    "all_fp16_lt_steps_5": ({"half": HALF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False, "steps": 5}, {"MIOPEN_DB": "pinned"}),
    "all_fp16_lt_steps_20": ({"half": HALF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False, "steps": 20}, {"MIOPEN_DB": "pinned"}),
    "all_fp16_lt_views_3": ({"half": HALF_ALL, "linear_t": True, "hoist": True, "fast_post": True, "benchmark": False, "views": 3}, {"MIOPEN_DB": "pinned"}),
    "views_3": ({"views": 3}, {}),
}

RESULTS = Path.home() / ".cache" / "robots-probes" / "xvla-speed" / "results" / "variants"


def run_one(name: str, batch_size: int, reps: int) -> None:
    opts, _env = VARIANTS[name.split("@")[0]]
    import numpy as np

    import xs_common as C
    from policy_fast import Policy, bits_equal, load_obs

    import torch

    obs, tasks = load_obs()
    rep: dict = {"name": name, "opts": opts, "batch": batch_size, "load_start": C.loadavg(),
                 "gtt_start_gib": C.gtt_gib(), "miopen_db": C.MIOPEN_DB,
                 "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
                 "miopen_find_enforce": os.environ.get("MIOPEN_FIND_ENFORCE")}
    t0 = time.perf_counter()
    pol = Policy("cuda", opts=opts)
    rep["policy_load_s"] = pol.load_s
    ctx = torch.no_grad if os.environ.get("XS_NO_INFERENCE_MODE") else torch.inference_mode

    def sync():
        torch.cuda.synchronize()

    def call(rows, timers=None):
        ob, tk = [obs[r] for r in rows], [tasks[r] for r in rows]
        batch = pol.batch(ob, tk)
        noise = torch.stack([pol.noise(7000 + r, 0) for r in rows])
        sync()
        marks = [time.perf_counter()]
        with ctx():
            inputs = pol.model_inputs(batch)
            sync(); marks.append(time.perf_counter())
            feats = pol.vision(inputs)
            sync(); marks.append(time.perf_counter())
            enc = pol.encode(inputs, feats)
            sync(); marks.append(time.perf_counter())
            action = pol.sample(inputs, enc, noise)
            sync(); marks.append(time.perf_counter())
            chunk = pol.finish(action)
        sync(); marks.append(time.perf_counter())
        acts = pol.to_env_actions(chunk)
        sync(); marks.append(time.perf_counter())
        if timers is not None:
            timers.append(np.diff(marks))
        return chunk.numpy(), acts

    rows0 = list(range(batch_size))
    s = time.perf_counter()
    first_chunk, _ = call(rows0)
    rep["first_call_s"] = time.perf_counter() - s  # includes kernel compilation and any algorithm search
    call(rows0)
    timers: list = []
    for _ in range(reps):
        call(rows0, timers)
    med = np.median(np.array(timers), axis=0)
    names = ["resize", "vision", "encoder", "loop", "to_cpu", "to_env_actions"]
    rep["ms"] = {k: round(1000 * float(v), 2) for k, v in zip(names, med)}
    rep["ms"]["call"] = round(1000 * float(np.sum(med)), 1)
    rep["ms_call_min_max"] = [round(1000 * float(np.sum(t)), 1) for t in (min(timers, key=np.sum), max(timers, key=np.sum))]
    chunks, acts = [], []
    for k in range(3):
        c, a = call([k * batch_size + i for i in range(batch_size)])
        chunks.append(c)
        acts.append(a)
    again, _ = call(rows0)
    rep["same_call_again_identical"] = bits_equal(again, chunks[0])
    rep["first_call_identical_to_later"] = bits_equal(first_chunk, chunks[0])
    chunks, acts = np.concatenate(chunks), np.concatenate(acts)
    rep["finite"] = bool(np.isfinite(chunks).all())
    rep["torch_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rep["own_gpu_mib"] = round(C.gpu_mib_of())
    rep["pss_mib"] = round(C.pss_mib())
    rep["load_end"] = C.loadavg()
    rep["total_s"] = time.perf_counter() - t0
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS / f"{name}.npz", chunks=chunks, acts=acts)
    (RESULTS / f"{name}.json").write_text(json.dumps(rep, indent=1))
    print(f"{name}: call {rep['ms']['call']} ms {rep['ms']} first call {rep['first_call_s']:.1f} s, "
          f"repeats {rep['same_call_again_identical']}, finite {rep['finite']}, torch peak {rep['torch_peak_gib']} GiB, "
          f"load {rep['load_start']} -> {rep['load_end']}", flush=True)


def table(ref_name: str = "today") -> None:
    import numpy as np

    ref = np.load(RESULTS / f"{ref_name}.npz")
    ref_ms = json.loads((RESULTS / f"{ref_name}.json").read_text())["ms"]
    print(f"{'variant':34s} {'call ms':>8s} {'x':>5s} {'vision':>7s} {'enc':>6s} {'loop':>7s} {'post':>5s} "
          f"{'vs today':>14s} {'max|d|':>9s} {'xyz mm':>7s} {'rot rad':>8s} {'grip':>4s} {'again':>5s} {'peak':>5s}")
    for f in sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime):
        r = json.loads(f.read_text())
        d = np.load(RESULTS / f"{r['name']}.npz")
        same = d["chunks"].tobytes() == ref["chunks"].tobytes() and d["acts"].tobytes() == ref["acts"].tobytes()
        dmax = float(np.abs(d["chunks"].astype(np.float64) - ref["chunks"]).max()) if d["chunks"].shape == ref["chunks"].shape else float("nan")
        n = min(d["acts"].shape[1], ref["acts"].shape[1])
        da = np.abs(d["acts"][:, :n].astype(np.float64) - ref["acts"][:, :n])
        ms = r["ms"]
        print(f"{r['name']:34s} {ms['call']:8.1f} {ref_ms['call'] / ms['call']:5.2f} {ms['vision']:7.1f} {ms['encoder']:6.1f} "
              f"{ms['loop']:7.1f} {ms['to_env_actions']:5.1f} {'bit-identical' if same else 'differs':>14s} {dmax:9.2e} "
              f"{1000 * da[..., :3].max():7.3f} {da[..., 3:6].max():8.4f} {int((da[..., 6] > 0).sum()):4d} "
              f"{str(r['same_call_again_identical'])[:1]:>5s} {r['torch_peak_gib']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", help="comma-separated variant names; NAME@2 repeats NAME under that name")
    ap.add_argument("--one")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--ref", default="today")
    args = ap.parse_args()
    if args.table:
        table(args.ref)
    elif args.one:
        run_one(args.one, args.batch, args.reps)
    else:
        failed = []
        for name in args.suite.split(","):
            _opts, extra = VARIANTS[name.split("@")[0]]
            # variants that name no database use the copy of the shared one, as the first measurements did
            env = {**os.environ, "MIOPEN_DB": os.environ.get("MIOPEN_DB", "today"), **extra}
            r = subprocess.run([sys.executable, __file__, "--one", name, "--batch", str(args.batch),
                                "--reps", str(args.reps)], env=env)
            if r.returncode != 0:
                failed.append((name, r.returncode))
                print(f"{name}: FAILED with exit status {r.returncode}", flush=True)
        if failed:
            sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
