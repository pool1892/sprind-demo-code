"""Time per SmolVLA inference call against batch size, in fp32 and bfloat16.

One call = one action chunk (50 actions) for every environment in the batch.
Times are wall-clock with the GPU queue flushed before and after. GPU memory is
read from the kernel's system-wide counter (GTT), sampled during the calls, so
it includes whatever else is on the GPU; the baseline before loading is saved.

    tools/heavy-run spikes/throughput/run.sh infer_bench.py
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import numpy as np
import torch

import common
import policy_setup as ps


class GttPeak:
    """Samples system-wide GPU memory in the background and keeps the maximum."""

    def __init__(self):
        self.peak = common.gtt_gib()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.wait(0.02):
            self.peak = max(self.peak, common.gtt_gib())

    def reset(self):
        self.peak = common.gtt_gib()

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--amp", nargs="+", default=["fp32", "bf16"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    ap.add_argument("--cameras", type=int, default=2, help="3 adds a copy of the first camera as camera3")
    ap.add_argument("--gtt-stop", type=float, default=16.0, help="stop growing the batch above this GTT use (GiB)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device

    obs, task = ps.sample_obs(args.res)
    gtt_base = common.gtt_gib()
    cpu0, t0 = common.cpu_seconds(), time.perf_counter()
    pol = ps.Policy(dev)
    load = {"load_s": pol.load_s, "gtt_base_gib": gtt_base, "gtt_after_load_gib": common.gtt_gib(),
            "pss_mib": common.proc_mem_mib().get("Pss"),
            "params_m": sum(p.numel() for p in pol.policy.parameters()) / 1e6}
    print(f"policy loaded in {pol.load_s:.1f} s, {load['params_m']:.0f} M parameters, "
          f"GTT {gtt_base:.2f} -> {load['gtt_after_load_gib']:.2f} GiB, process Pss {load['pss_mib']:.0f} MiB, "
          f"load {common.loadavg()}", flush=True)

    # Split the call: time spent in the denoising loop, the rest is the image/language prefix pass.
    model = pol.policy.model
    den = {"t": 0.0, "n": 0}
    orig = model.denoise_step

    def timed_denoise(*a, **k):
        ps.sync(dev)
        s = time.perf_counter()
        out = orig(*a, **k)
        ps.sync(dev)
        den["t"] += time.perf_counter() - s
        den["n"] += 1
        return out

    model.denoise_step = timed_denoise

    peak = GttPeak()
    rows, ref = [], {}
    # Batch size outside, precision inside: the two precisions are then measured back to back
    # under the same machine load, which varies a lot here because other jobs share the CPU.
    for b in args.batches:
        if common.gtt_gib() > args.gtt_stop:
            print(f"stop at batch {b}: GTT {common.gtt_gib():.1f} GiB above {args.gtt_stop}")
            break
        for amp in args.amp:
            stacked = ps.stack_obs([obs] * b)
            t_pre = []
            for _ in range(3):
                s = time.perf_counter()
                batch = pol.preprocess(stacked, [task] * b)
                ps.sync(dev)
                t_pre.append(time.perf_counter() - s)
            if args.cameras == 3:
                batch["observation.images.camera3"] = batch["observation.images.camera1"].clone()
            noise = torch.randn(b, pol.cfg.chunk_size, pol.cfg.max_action_dim,
                                generator=torch.Generator().manual_seed(0)).to(dev)
            pol.chunks(batch, amp, noise)  # warm-up: kernel selection for this shape
            ps.sync(dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            peak.reset()
            times, dens = [], []
            for _ in range(args.reps):
                den["t"] = 0.0
                ps.sync(dev)
                s = time.perf_counter()
                out = pol.chunks(batch, amp, noise)
                ps.sync(dev)
                times.append(time.perf_counter() - s)
                dens.append(den["t"])
            s = time.perf_counter()
            acts = pol.to_env_actions(out)
            t_post = time.perf_counter() - s
            out_cpu = out.float().cpu()
            if amp == "fp32":
                ref[b] = out_cpu
            row = {
                "amp": amp, "batch": b, "cameras": args.cameras,
                "call_s": statistics.median(times), "call_s_min": min(times), "call_s_max": max(times),
                "denoise_s": statistics.median(dens), "prefix_s": statistics.median(times) - statistics.median(dens),
                "per_env_ms": 1000 * statistics.median(times) / b,
                "preprocess_s": statistics.median(t_pre), "postprocess_s": t_post,
                "gtt_peak_gib": peak.peak,
                "torch_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0,
                "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30 if dev == "cuda" else 0.0,
                "max_abs_diff_vs_fp32": float((out_cpu - ref[b]).abs().max()) if b in ref else None,
                "action_shape": list(acts.shape), "loadavg": common.loadavg(),
            }
            rows.append(row)
            print(f"{amp} batch {b:3d}: {row['call_s'] * 1000:8.0f} ms/call (min {row['call_s_min'] * 1000:.0f}, "
                  f"max {row['call_s_max'] * 1000:.0f}; prefix {row['prefix_s'] * 1000:.0f}, "
                  f"denoise {row['denoise_s'] * 1000:.0f}) = {row['per_env_ms']:6.1f} ms per env; "
                  f"pre {row['preprocess_s'] * 1000:.0f} ms, post {row['postprocess_s'] * 1000:.0f} ms; "
                  f"GTT peak {row['gtt_peak_gib']:.2f} GiB, torch peak {row['torch_peak_gib']:.2f} GiB; "
                  f"diff vs fp32 {row['max_abs_diff_vs_fp32']}; load {row['loadavg'].split()[0]}", flush=True)
    wall = time.perf_counter() - t0
    summary = {"cpu_s_per_wall_s": (common.cpu_seconds() - cpu0) / wall}
    print(f"process CPU use over the whole run: {summary['cpu_s_per_wall_s']:.2f} cores")
    peak.stop()
    print("saved", common.save(f"infer_bench_{dev}_{args.cameras}cam{args.tag}.json",
                               {"args": vars(args), "load": load, "rows": rows, "summary": summary}))


if __name__ == "__main__":
    main()
