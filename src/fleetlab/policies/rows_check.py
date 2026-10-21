"""The rows check: does a row's chunk depend on its place in the batch or on the other rows?

A policy may declare `row_independent = True` only if this check passes
(contracts, section 6; replay rule 3). Ported from spikes/xvla-speed/
rows_check.py. A "content" is one observation with its instruction and its
own noise. For a batch size N the contents are laid out as base batches of N
rows, and every content's chunk in its base batch is the reference. Then, at
the same batch size:

- `again`: the same base batch once more (a call repeats);
- `position`: each base batch rotated, every content in another row with the
  same mates;
- `mates`: row i taken from base batch k + i, every content in its row with
  other mates;
- `padded`: the first half of a base batch, the rest copies of row 0, which is
  how the server fills rows nobody asked for.

A comparison counts only if every number of the chunk has the same bits.

    tools/heavy-run uv run --locked python -m fleetlab.policies.rows_check --precision fp16_res32
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from fleetlab.contracts.env import Observation
from fleetlab.contracts.policy import NoiseSample, Policy


def _noise(policy: Policy, content: int) -> NoiseSample:
    assert policy.noise_shape is not None and policy.noise_dtype is not None
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([7, content])))
    values = generator.standard_normal(policy.noise_shape, dtype=np.float32)
    return NoiseSample(noise_seed=content, chunk_index=0, values=values.astype(policy.noise_dtype))


def rows_check(
    policy: Policy, contents: Sequence[tuple[Observation, str]], batch_size: int
) -> dict[str, Any]:
    """Run the check at one batch size; returns the tallies and `independent`."""
    n_contents = len(contents)
    if n_contents < 2 * batch_size:
        raise ValueError(f"need at least {2 * batch_size} contents, have {n_contents}")
    noise = [_noise(policy, c) for c in range(n_contents)]

    def call(rows: list[int]) -> list[np.ndarray[Any, Any]]:
        chunks = policy.act(
            [contents[c][0] for c in rows],
            [contents[c][1] for c in rows],
            [noise[c] for c in rows],
            policy.spec,
        )
        return [np.ascontiguousarray(chunk.actions) for chunk in chunks]

    def tally(pairs: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]) -> dict[str, Any]:
        same = sum(a.tobytes() == b.tobytes() for a, b in pairs)
        worst = max(float(np.abs(a.astype(np.float64) - b).max()) for a, b in pairs)
        return {"comparisons": len(pairs), "identical": same, "largest_difference": worst}

    t0 = time.perf_counter()
    k = n_contents // batch_size
    base = [[b * batch_size + i for i in range(batch_size)] for b in range(k)]
    reference: dict[int, np.ndarray[Any, Any]] = {}
    for rows in base:
        for c, chunk in zip(rows, call(rows), strict=True):
            reference[c] = chunk
    result: dict[str, Any] = {"batch_size": batch_size, "contents": n_contents}
    result["again"] = tally(list(zip(call(base[0]), [reference[c] for c in base[0]], strict=True)))
    pairs = []
    for b, rows in enumerate(base):
        s = 1 + b % (batch_size - 1) if batch_size > 1 else 0
        turned = rows[s:] + rows[:s]
        pairs += [(chunk, reference[c]) for c, chunk in zip(turned, call(turned), strict=True)]
    result["position"] = tally(pairs)
    pairs = []
    for b in range(k):
        mixed = [base[(b + i) % k][i] for i in range(batch_size)]
        pairs += [(chunk, reference[c]) for c, chunk in zip(mixed, call(mixed), strict=True)]
    result["mates"] = tally(pairs)
    pairs = []
    half = max(1, batch_size // 2)
    for rows in base:
        padded = rows[:half] + [rows[0]] * (batch_size - half)
        pairs += [(chunk, reference[c]) for c, chunk in zip(padded, call(padded), strict=True)]
    result["padded"] = tally(pairs)
    result["independent"] = all(
        result[name]["identical"] == result[name]["comparisons"]
        for name in ("again", "position", "mates", "padded")
    )
    result["seconds"] = time.perf_counter() - t0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.policies.rows_check")
    parser.add_argument("--precision", default="fp16_res32")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--contents", type=int, default=16)
    parser.add_argument("--miopen", default="pinned")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.engine import miopen
    from fleetlab.policies.xvla import XVLAPolicy, sample_observations

    miopen.activate(args.miopen)
    db = {b: miopen.require(b, args.precision, args.miopen) for b in args.batch_sizes}
    contents = sample_observations(args.contents)
    policy = XVLAPolicy()
    policy.load("cuda", args.precision)
    report = {
        "precision": args.precision,
        "miopen": db,
        "by_size": [rows_check(policy, contents, b) for b in args.batch_sizes],
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
