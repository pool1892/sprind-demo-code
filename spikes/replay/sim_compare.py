"""Compare recordings made by sim_trace.py, bit for bit.

    python sim_compare.py A[:repeat] B[:repeat] [...]

The first argument is the reference. Reports, per pair, whether the simulator
state, the proprioceptive observation and each camera stream are identical, the
first step at which they differ, and how large the difference is.
"""

from __future__ import annotations

import sys

import numpy as np

import common as C


def load(spec: str):
    name, _, rep = spec.partition(":")
    rep = int(rep) if rep else 0
    z = np.load(C.OUT / f"{name}.npz")
    return {k: z[f"{k}_{rep}"] for k in ("state", "proprio", "success", *C.CAMERAS)}


def describe_images(a: np.ndarray, b: np.ndarray) -> str:
    n = min(len(a), len(b))
    d = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))
    frames = int((d.reshape(n, -1).max(axis=1) > 0).sum())
    if frames == 0:
        return "identical"
    per_frame = (d.reshape(n, -1) > 0).mean(axis=1)
    return (
        f"DIFFER in {frames}/{n} frames, first at step {int(np.flatnonzero(per_frame)[0])}; "
        f"largest difference {int(d.max())} of 255, on average {100 * per_frame[per_frame > 0].mean():.3f}% "
        f"of the values in a differing frame (worst frame {100 * per_frame.max():.3f}%), "
        f"mean absolute difference over all values {d.mean():.5f}"
    )


def main():
    ref_spec, *others = sys.argv[1:]
    ref = load(ref_spec)
    for spec in others:
        o = load(spec)
        print(f"{ref_spec} vs {spec}")
        for k in ("state", "proprio"):
            n = min(len(ref[k]), len(o[k]))
            same = len(ref[k]) == len(o[k]) and np.array_equal(
                ref[k].view(np.uint64), o[k].view(np.uint64)
            )
            if same:
                print(f"  {k}: identical over {n - 1} steps")
            else:
                i = C.first_diff(ref[k], o[k])
                d = np.abs(ref[k][:n] - o[k][:n]).max(axis=1)
                print(
                    f"  {k}: DIFFER, first at step {i}, max abs difference there {d[i]:.3e}, "
                    f"at the end {d[-1]:.3e}, largest {d.max():.3e}"
                )
        for k in C.CAMERAS:
            print(f"  {k}: {describe_images(ref[k], o[k])}")
        print(f"  outcome: {bool(ref['success'].any())} vs {bool(o['success'].any())}")


if __name__ == "__main__":
    main()
