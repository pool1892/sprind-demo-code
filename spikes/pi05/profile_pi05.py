"""Question 1: does the checkpoint load and run here, how long does a call take, and where does the time go?

    tools/heavy-run ~/.cache/robots-probes/pi05/venv/bin/python spikes/pi05/profile_pi05.py \
        --tag bf16 --opts '{"dtype": "bfloat16"}' [--batches 1 4 8] [--reps 5] [--pin]

One process, one number format. Loads the policy (time, process memory, GPU
memory), then for each batch size takes the first B of the 80 recorded
observations (spikes/xvla-speed/make_obs.py), each with its own noise, and:

- times whole calls as the fleet makes them: batch preparation on the CPU
  (`Policy.batch`), the chunk (`chunks_staged`, LeRobot's predict_action_chunk)
  and the conversion to environment actions; median of --reps after one warm-up;
- times the stages of one call (`chunks_timed`): image resize and scaling,
  SigLIP on the three images, the language model over the prefix (which fills
  the key-value cache), the denoising loop of the action expert, the copy back;
- checks that the timed call gives the same bits as LeRobot's own call, and that
  a call repeated in the same process gives the same bits;
- saves the chunks of the first 8 contents, so that another process (another
  number format) can be compared with them (`--against TAG`).

`--pin` first runs the vision tower with cudnn.benchmark on into the MIOpen
database named by MIOPEN_DB (default "pinned") for every batch size given, and
makes the database read-only; every later run reads it with benchmark off. The
convolution is SigLIP's patch embedding, in 32-bit floats in both formats.

Writes ~/.cache/robots-probes/pi05/results/profile_<tag>.json and chunks_<tag>.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path

import numpy as np

import p5_common as C
from policy_pi05 import Policy, bits_equal, load_obs

import torch  # noqa: E402


def vm(field: str) -> float:
    """A field of /proc/self/status in GiB (VmRSS, VmHWM)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 2**20
    return -1.0


def memory() -> dict:
    return {"rss_gib": round(vm("VmRSS"), 2), "rss_peak_gib": round(vm("VmHWM"), 2),
            "pss_gib": round(C.pss_mib() / 1024, 2), "gpu_process_gib": round(C.gpu_mib_of() / 1024, 2),
            "torch_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "gtt_machine_gib": round(C.gtt_gib(), 2), "mem_available_gib": round(C.mem_available_gib(), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--against", default=None, help="tag of another run whose saved chunks to compare with")
    args = ap.parse_args()
    opts = json.loads(args.opts)
    (C.OUT / "results").mkdir(parents=True, exist_ok=True)
    obs, tasks = load_obs()
    report: dict = {"tag": args.tag, "opts": opts, "miopen_db": C.MIOPEN_DB, "load_start": C.loadavg(),
                    "gtt_start_gib": round(C.gtt_gib(), 2), "before_load": None}
    torch.cuda.init()
    report["before_load"] = memory()
    if args.pin:
        db = Path(os.environ["MIOPEN_USER_DB_PATH"])
        db.mkdir(parents=True, exist_ok=True)
        for f in db.glob("*.txt"):
            f.chmod(f.stat().st_mode | stat.S_IWUSR)
        opts = {**opts, "benchmark": True}
    t0 = time.perf_counter()
    pol = Policy("cuda", opts=opts)
    report["load_s"] = round(time.perf_counter() - t0, 1)
    report["after_load"] = memory()
    report["parameters"] = pol.n_params
    report["parameters_by_dtype"] = pol.params_by_dtype
    report["n_linear_t"] = pol.n_linear_t
    print(f"loaded in {report['load_s']} s: {pol.n_params / 1e9:.2f} B parameters {pol.params_by_dtype}; "
          f"{report['after_load']}", flush=True)

    if args.pin:
        pin = []
        for b in args.batches:
            batch = pol.batch(obs[:b], tasks[:b])
            with torch.inference_mode():
                images, _ = pol.policy._preprocess_images(dict(batch))
                torch.cuda.synchronize()
                s = time.perf_counter()
                pol.policy.model.paligemma_with_expert.embed_image(images[0])
                torch.cuda.synchronize()
                first = time.perf_counter() - s
                s = time.perf_counter()
                pol.policy.model.paligemma_with_expert.embed_image(images[0])
                torch.cuda.synchronize()
            pin.append({"rows": b, "first_s": round(first, 2), "second_ms": round(1000 * (time.perf_counter() - s), 1)})
            print("pinned", pin[-1], flush=True)
        for f in db.glob("*.txt"):
            f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        report["pin"] = pin
        report["pin_files"] = {f.name: f.stat().st_size for f in sorted(db.glob("*"))}
        torch.backends.cudnn.benchmark = False

    report["by_batch"] = {}
    saved = {}
    for b in args.batches:
        contents = list(range(b))
        noise = torch.stack([pol.noise(9000 + c, 0) for c in contents])
        whole, prep, chunk_s, post_s, outs = [], [], [], [], []
        for r in range(args.reps + 1):
            torch.cuda.synchronize()
            s0 = time.perf_counter()
            batch = pol.batch([obs[c] for c in contents], [tasks[c] for c in contents])
            torch.cuda.synchronize()
            s1 = time.perf_counter()
            ch = pol.chunks_staged(batch, noise)
            s2 = time.perf_counter()
            acts = pol.to_env_actions(ch)
            s3 = time.perf_counter()
            if r == 0:
                first_call_s = s3 - s0
                continue
            whole.append(s3 - s0)
            prep.append(s1 - s0)
            chunk_s.append(s2 - s1)
            post_s.append(s3 - s2)
            outs.append(ch.numpy())
        repeat_identical = all(bits_equal(o, outs[0]) for o in outs)
        stages: dict[str, list] = {}
        timed_out = None
        if pol.opts["model"] == "pi05":
            for r in range(max(3, args.reps // 2)):
                t: dict = {}
                timed_out = pol.chunks_timed(batch, noise, t).numpy()
                for k, v in t.items():
                    stages.setdefault(k, []).append(v)
        med = lambda xs: round(1000 * float(np.median(xs)), 1)  # noqa: E731
        res = {
            "first_call_s": round(first_call_s, 2),
            "call_ms_median": med(whole), "call_ms_all": [round(1000 * x, 1) for x in whole],
            "prep_ms_median": med(prep), "chunk_ms_median": med(chunk_s), "to_env_actions_ms_median": med(post_s),
            "per_row_ms": round(med(whole) / b, 1),
            "repeat_identical": repeat_identical,
            "stages_ms_median": {k: (med(v) if k not in ("prefix_tokens", "prefix_tokens_real_mean") else v[0])
                                 for k, v in stages.items()},
            "timed_equals_lerobot_call": None if timed_out is None else bits_equal(timed_out, outs[0]),
            "memory_after": memory(),
            "action_range": [float(acts.min()), float(acts.max())],
            "finite": bool(np.isfinite(outs[0]).all()),
        }
        report["by_batch"][b] = res
        if b == 8 or b == max(args.batches):
            saved = {"chunks": outs[0], "contents": np.array(contents), "env_actions": acts}
        print(f"batch {b}: call {res['call_ms_median']} ms ({res['per_row_ms']} per row; prep {res['prep_ms_median']}, "
              f"chunk {res['chunk_ms_median']}, to actions {res['to_env_actions_ms_median']}), first {res['first_call_s']} s, "
              f"stages {res['stages_ms_median']}, repeat identical {repeat_identical}, timed == LeRobot "
              f"{res['timed_equals_lerobot_call']}, {res['memory_after']}", flush=True)

    np.savez(C.OUT / "results" / f"chunks_{args.tag}.npz", **saved)
    if args.against:
        other = np.load(C.OUT / "results" / f"chunks_{args.against}.npz")
        n = min(len(other["contents"]), len(saved["contents"]))
        a, o = saved["chunks"][:n].astype(np.float64), other["chunks"][:n].astype(np.float64)
        ea, eo = saved["env_actions"][:n].astype(np.float64), other["env_actions"][:n].astype(np.float64)
        report["against"] = {
            "tag": args.against, "contents": n,
            "chunk_max_abs_difference_normalised": float(np.abs(a - o).max()),
            # relative control: the first three action numbers are position steps, scaled by robosuite
            "env_action_max_abs_difference": [float(x) for x in np.abs(ea - eo).max(axis=(0, 1))],
            "env_action_mean_abs_difference": [float(x) for x in np.abs(ea - eo).mean(axis=(0, 1))],
            "gripper_sign_flips": int((np.sign(ea[..., 6]) != np.sign(eo[..., 6])).sum()),
        }
        print("against", args.against, json.dumps(report["against"]), flush=True)
    report["peak"] = memory()
    report["load_end"] = C.loadavg()
    (C.OUT / "results" / f"profile_{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(f"wrote results/profile_{args.tag}.json; load {report['load_start']} -> {report['load_end']}")


if __name__ == "__main__":
    main()
