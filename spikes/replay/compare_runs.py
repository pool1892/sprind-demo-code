"""Compare closed-loop recordings made by closed_loop.py, episode by episode.

    python compare_runs.py A[:repeat] B[:repeat] [--quiet]

Reports for every episode whether actions, simulator states and images are
bit-identical, the first step where they differ, how far the end-effector
positions drift apart, and whether the outcome flipped.
"""

from __future__ import annotations

import sys

import numpy as np

import common as C


def load(spec: str):
    name, _, rep = spec.partition(":")
    z = np.load(C.OUT / f"{name}.npz")
    return z, f"r{int(rep) if rep else 0}_"


def compare(a_spec: str, b_spec: str, quiet: bool = False) -> dict:
    (za, pa), (zb, pb) = load(a_spec), load(b_spec)
    # episodes are matched by (task, initial state, scene seed); the two recordings may hold
    # them in a different order, and the second may hold only some of them
    where_b = {tuple(zb[f"{pb}e{j}_spec"][:3]): j for j in range(int(zb["n_episodes"]))}
    pairs = [
        (i, where_b[tuple(za[f"{pa}e{i}_spec"][:3])])
        for i in range(int(za["n_episodes"]))
        if tuple(za[f"{pa}e{i}_spec"][:3]) in where_b
    ]
    n = len(pairs)
    exact = flips = 0
    first_div, rows, mm1, cm1, final, succ_a, succ_b = [], [], [], [], [], [], []
    for i, j in pairs:
        ka, kb = f"{pa}e{i}_", f"{pb}e{j}_"
        A, B = za[ka + "actions"], zb[kb + "actions"]
        SA, SB = za[ka + "states"], zb[kb + "states"]
        IA, IB = za[ka + "imgs"], zb[kb + "imgs"]
        sa, sb = bool(za[ka + "success"]), bool(zb[kb + "success"])
        same = (
            A.shape == B.shape
            and A.tobytes() == B.tobytes()
            and SA.tobytes() == SB.tobytes()
            and IA.tobytes() == IB.tobytes()
        )
        exact += same
        flips += sa != sb
        succ_a.append(sa)
        succ_b.append(sb)
        if same:
            rows.append(f"  episode {i} {za[ka + 'spec'][:2]}: identical, {len(A)} steps, success {sa}")
            continue
        m = min(len(A), len(B))
        da = C.first_diff(A[:m], B[:m])
        ds = C.first_diff(SA[: m + 1], SB[: m + 1])
        di = C.first_diff(IA[: m + 1], IB[: m + 1])
        pA, pB = za[ka + "proprio"][: m + 1, :3], zb[kb + "proprio"][: m + 1, :3]
        dist = np.linalg.norm(pA - pB, axis=1)
        chunk_d = [
            float(np.abs(x - y).max())
            for x, y in zip(za[ka + "chunks"], zb[kb + "chunks"], strict=False)
        ]
        first_div.append(da if da >= 0 else m)

        def when(th):
            idx = np.flatnonzero(dist > th)
            return int(idx[0]) if len(idx) else None

        mm1.append(when(1e-3))
        cm1.append(when(1e-2))
        final.append(1000 * dist.max())

        rows.append(
            f"  episode {i} {za[ka + 'spec'][:2]}: DIFFER, steps {len(A)} vs {len(B)}, success {sa} vs {sb}; "
            f"first difference: action step {da}, state step {ds}, image step {di}; "
            f"per-chunk max action difference {['%.1e' % c for c in chunk_d]}; "
            f"gripper position apart by >1 mm at step {when(1e-3)}, >1 cm at step {when(1e-2)}, "
            f"largest {1000 * dist.max():.2f} mm"
        )
    if not quiet:
        print("\n".join(rows))
    print(
        f"{a_spec} vs {b_spec}: {exact}/{n} episodes bit-identical, {flips}/{n} outcomes flipped; "
        f"successes {sum(succ_a)}/{n} vs {sum(succ_b)}/{n}"
    )
    if first_div:
        k = len(first_div)
        r1 = [x for x in mm1 if x is not None]
        r2 = [x for x in cm1 if x is not None]
        print(
            f"  of the {k} differing episodes: first differing action at step median {int(np.median(first_div))} "
            f"(min {min(first_div)}, max {max(first_div)}); gripper paths more than 1 mm apart in {len(r1)}/{k} "
            f"(median from step {int(np.median(r1)) if r1 else '-'}), more than 1 cm apart in {len(r2)}/{k} "
            f"(median from step {int(np.median(r2)) if r2 else '-'}); largest separation median "
            f"{np.median(final):.1f} mm, max {max(final):.1f} mm"
        )
    return {"n": n, "exact": exact, "flips": flips}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare(args[0], args[1], quiet="--quiet" in sys.argv)
