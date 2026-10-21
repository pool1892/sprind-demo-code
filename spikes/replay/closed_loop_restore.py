"""Question 3 with the policy in the loop: branch an episode from a saved instant.

    python closed_loop_restore.py --out NAME [--tasks 0,1] [--k 37,130] [--batching alone]

Each episode is run once from start to end (the reference). On the way a
snapshot is taken after step k: the simulator snapshot of snapshot.py plus the
policy-side state, which in this loop is only the actions still queued from
the current chunk and the index of the next chunk (it selects the noise).
Afterwards the simulator is reset to a different episode, the snapshot is put
back and the episode is continued. The continuation must equal the reference.
"""

from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

import common as C
import engine as E
import snapshot as S
from closed_loop import specs_for
from policy_lib import Policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", default="0,1")
    p.add_argument("--inits", default="0")
    p.add_argument("--k", default="37,130")
    p.add_argument("--batching", default="alone")
    p.add_argument("--amp", default="fp32")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    specs = specs_for([int(x) for x in args.tasks.split(",")], [int(x) for x in args.inits.split(",")])
    ks = [int(x) for x in args.k.split(",")]

    policy = Policy(device=args.device)
    pool = E.EnvPool()
    saved = {}
    for spec in specs:
        t0 = time.time()
        snaps = {}

        def grab(r, snaps=snaps):
            if r.t in ks:
                snaps[r.t] = {
                    "sim": S.take(r.env, r.obs),
                    "queue": [a.copy() for a in r.queue],
                    "chunk_index": r.chunk_index,
                    "t": r.t,
                }

        ref = E.rollout(policy, [spec], pool, batching=args.batching, amp=args.amp, before_step=grab)[0]
        print(f"episode task {spec.task} init {spec.init}: {ref.t} steps, success {ref.success} ({time.time() - t0:.1f} s)")
        for k, snap in snaps.items():
            env = pool.take(spec)
            other = E.Spec(task=spec.task, init=(spec.init + 7) % 50, seed=spec.seed + 13, noise_seed=0)
            E.reset_episode(env, other)  # a different scene and a fresh controller
            for a in C.synthetic_actions(15, seed=k):
                env.step(a)  # and some unrelated motion
            run = E.Run(spec=spec, env=env)
            run.obs = S.restore(env, snap["sim"])
            run.queue = [a.copy() for a in snap["queue"]]
            run.chunk_index = snap["chunk_index"]
            run.t = snap["t"]
            E.record(run)
            E.advance(policy, [run], batching=args.batching, amp=args.amp)
            pool.give(spec, env)
            same = (
                np.stack(run.actions).tobytes() == np.stack(ref.actions[k:]).tobytes()
                and np.stack(run.states).tobytes() == np.stack(ref.states[k:]).tobytes()
                and np.stack(run.imgs).tobytes() == np.stack(ref.imgs[k:]).tobytes()
                and run.success == ref.success
            )
            n_inf = len(run.chunks)
            print(
                f"  restored after step {k} ({len(snap['queue'])} queued actions, {n_inf} later inferences): "
                f"continuation {'IDENTICAL' if same else 'DIFFERS'} over {run.t - k} steps, "
                f"success {run.success} vs {ref.success}",
                flush=True,
            )
            saved[(spec.task, spec.init, k)] = same
        if spec == specs[0]:
            with open(C.OUT / f"{args.out}_snapshot.pkl", "wb") as f:
                pickle.dump({"snaps": snaps, "spec": spec, "ref_actions": np.stack(ref.actions),
                             "ref_states": np.stack(ref.states), "ref_imgs": np.stack(ref.imgs),
                             "ref_success": ref.success}, f)
    print(f"{sum(saved.values())}/{len(saved)} restored continuations identical")
    pool.close()


if __name__ == "__main__":
    main()
