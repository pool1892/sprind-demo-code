"""Question 4 and 5: the policy alone, on recorded observations.

    python policy_probe.py --obs RECORDING --out NAME [--amp fp32,bf16,fp16] [--k 8]

RECORDING is a closed_loop.py output made with --save-obs, or the file written
by make_obs.py (observations straight from the simulator). For each precision
mode the script checks
  A  the same observation and the same noise, three calls in a row;
  B  where the default noise comes from and whether it depends on batch size;
  C1 a row's output when the other rows of the batch are replaced (batch size fixed);
  C3 a row's output when its position in the batch changes as well (batch size fixed);
  C2 a row's output at batch sizes 2, 3, 4, 8 and 10 against the row alone;
and stores the outputs, so that a second run of this script (a fresh process)
can be compared with the first by compare_probe.py.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import common as C
from policy_lib import Policy, bits_equal


def load_obs(name: str):
    z = np.load(C.OUT / f"{name}.npz")
    n = len(z["obs_image"])
    obs = [
        {
            "pixels": {"image": z["obs_image"][i], "image2": z["obs_image2"][i]},
            "robot_state": {
                "eef": {"pos": z["obs_eef_pos"][i], "quat": z["obs_eef_quat"][i], "mat": z["obs_eef_mat"][i]},
                "gripper": {"qpos": z["obs_gripper_qpos"][i], "qvel": z["obs_gripper_qvel"][i]},
                "joints": {"pos": z["obs_joint_pos"][i], "vel": z["obs_joint_vel"][i]},
            },
        }
        for i in range(n)
    ]
    return obs, [str(t) for t in z["obs_task"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", required=True)
    p.add_argument("--quick", action="store_true", help="a handful of calls only, for debugging")
    p.add_argument("--reference-only", action="store_true", help="only the outputs compare_probe.py needs")
    p.add_argument("--out", required=True)
    p.add_argument("--amp", default="fp32,bf16,fp16")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    obs, tasks = load_obs(args.obs)
    # spread the picks over the recording so that tasks and episode phases are mixed
    pick = np.linspace(0, len(obs) - 1, num=min(len(obs), 16)).round().astype(int)
    obs = [obs[i] for i in pick]
    tasks = [tasks[i] for i in pick]
    n, k = len(obs), min(args.k, len(obs))
    print(f"{n} observations, {len(set(tasks))} distinct task texts")

    t0 = time.time()
    pol = Policy(device=args.device)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"policy loaded in {time.time() - t0:.1f} s", flush=True)
    noise = [pol.noise(7000 + i, 0) for i in range(n)]
    out = {}

    def run(rows, amp):
        b = pol.batch([obs[i] for i in rows], [tasks[i] for i in rows])
        return pol.chunks(b, torch.stack([noise[i] for i in rows]), amp=amp).numpy()

    def report(label, pairs):
        """pairs: list of (candidate, reference) arrays."""
        same = sum(bits_equal(a, b) for a, b in pairs)
        worst = max(float(np.abs(a - b).max()) for a, b in pairs)
        print(f"  {label}: {same}/{len(pairs)} bit-identical, largest difference {worst:.3e}", flush=True)

    # B: where the default noise comes from
    if args.device == "cuda":
        torch.manual_seed(1000)
        n1 = pol.policy.model.sample_noise((1, *pol.noise_shape), "cuda").cpu().numpy()
        torch.manual_seed(1000)
        n1b = pol.policy.model.sample_noise((1, *pol.noise_shape), "cuda").cpu().numpy()
        torch.manual_seed(1000)
        n4 = pol.policy.model.sample_noise((4, *pol.noise_shape), "cuda").cpu().numpy()
        n_next = pol.policy.model.sample_noise((1, *pol.noise_shape), "cuda").cpu().numpy()
        print(
            "default noise (torch's global GPU generator, one draw for the whole batch): "
            f"same seed and same shape repeats: {bits_equal(n1, n1b)}; "
            f"row 0 of a batch of 4 equals the batch-of-1 draw: {bits_equal(n4[:1], n1)}; "
            f"the next call repeats the first: {bits_equal(n_next, n1)}"
        )
        out["default_noise_b1"] = n1

    for amp in args.amp.split(","):
        if args.device != "cuda" and amp != "fp32":
            continue
        t1 = time.time()
        print(f"precision {amp}")
        alone = [run([i], amp)[0] for i in range(4 if args.quick else n)]
        out[f"{amp}_alone"] = np.stack(alone)
        # A: three calls in a row
        if args.reference_only:
            out[f"{amp}_b4_base"] = run(list(range(4)), amp)
            print(f"  reference outputs stored ({time.time() - t1:.1f} s)")
            continue
        again = [run([0], amp)[0] for _ in range(1 if args.quick else 3)]
        report("A  same input three more times", [(a, alone[0]) for a in again])
        # C1 and C3: fixed batch size; replace the other rows and move the tested row
        rng = np.random.default_rng(0)
        for bsz in (4,) if args.quick else (4, 8, 10):
            if bsz > n:
                continue
            same_pos, moved, vs_alone = [], [], []
            for i in range(1 if args.quick else min(4, k)):
                others = [j for j in range(n) if j != i]
                base_rows = [i, *rng.choice(others, bsz - 1, replace=False).tolist()]
                base = run(base_rows, amp)[0]
                vs_alone.append((base, alone[i]))
                if i == 0 and bsz == 4:
                    out[f"{amp}_b4_row0"] = base
                # other rows replaced, tested row stays first
                rows = [i, *rng.choice(others, bsz - 1, replace=False).tolist()]
                same_pos.append((run(rows, amp)[0], base))
                # other rows replaced and tested row moved to the middle and to the end
                for pos in (bsz // 2, bsz - 1):
                    rows = rng.choice(others, bsz - 1, replace=False).tolist()
                    rows.insert(pos, i)
                    moved.append((run(rows, amp)[pos], base))
            report(f"C1 batch of {bsz}: other rows replaced, same position", same_pos)
            report(f"C3 batch of {bsz}: other rows replaced and position moved", moved)
            report(f"   batch of {bsz} against the row alone", vs_alone)
        # C2: batch size
        for b in (2,) if args.quick else (2, 3, 4, 8, 10):
            if b > n:
                continue
            pairs = []
            for i in range(1 if args.quick else k):
                rows = [i] + [j for j in range(n) if j != i][: b - 1]
                pairs.append((run(rows, amp)[0], alone[i]))
            report(f"C2 batch of {b} against the row alone", pairs)
        # C2b: the same observation repeated in every row
        for b in () if args.quick else (2, 4, 8):
            r = run([0] * b, amp)
            report(f"C2 the same row repeated {b} times against the row alone", [(x, alone[0]) for x in r])
        out[f"{amp}_b4_base"] = run(list(range(4)), amp)
        scale = float(np.abs(np.stack(alone)).mean())
        print(f"  (mean absolute value of the outputs: {scale:.3f}; {time.time() - t1:.1f} s)")

    np.savez_compressed(C.OUT / f"{args.out}.npz", **out)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
