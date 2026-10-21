"""What does it cost to keep several environments in one process?

lerobot-eval steps ten environments one after another in a single process.
This steps K environments the same way, without a policy, and reports the time
per simulated step and its parts. K = 1 is the reference.

--mixed checks the images instead: robosuite makes a rendering context current
only when it creates it, so with several environments in one process every
environment draws inside whichever context was created last. The check puts
two different scenes into one process and draws the first one twice from the
same state, once as it would be drawn in a step and once after switching to
its own context.

    tools/heavy-run --light spikes/throughput/run.sh multi_env_one_process.py --k 1 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import common


def run(k: int, steps: int, res: int, threads: int) -> dict:
    import profile_step

    if threads:
        import torch

        torch.set_num_threads(threads)
    envs = [common.make_env(task_id=0, res=res, episode_index=i) for i in range(k)]
    for i, e in enumerate(envs):
        e.reset(seed=i)
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.3, 0.3, size=(steps, 7)).astype(np.float32)
    acts[:, 6] = np.sign(acts[:, 6])
    profile_step.T.clear()
    profile_step.N.clear()
    cpu0, t0 = common.cpu_seconds(), time.perf_counter()
    for a in acts:
        for e in envs:
            e.step(a)
    wall = time.perf_counter() - t0
    T, n = profile_step.T, k * steps
    row = {"envs_in_process": k, "ms_per_env_step": 1000 * wall / n,
           "physics_ms": 1000 * (T["physics.step1"] + T["physics.step2"]) / n,
           "controller_ms": 1000 * T["controller"] / n,
           "render_ms": 1000 * sum(v for key, v in T.items() if key.startswith("render.")
                                   and key not in ("render.draw", "render.readback")) / n,
           "cpu_cores": (common.cpu_seconds() - cpu0) / wall, "pss_mib": common.proc_mem_mib().get("Pss"),
           "threads": threads, "loadavg": common.loadavg()}
    for e in envs:
        e.close()
    return row


def mixed(res: int) -> dict:
    """Two different scenes in one process: is the first one still drawn correctly?

    Same process, same simulation state, same camera; the only thing that changes
    between the two pictures is which rendering context is current.
    """
    e0 = common.make_env(task_id=0, res=res, suite_name="libero_spatial")
    e1 = common.make_env(task_id=0, res=res, suite_name="libero_10")
    e0.reset(seed=0)
    e1.reset(seed=0)  # the context of e1 is now the current one, as after any vector-env reset
    sim0 = e0._env.env.sim

    def shot():
        return sim0.render(camera_name="agentview", width=res, height=res).copy()

    foreign = shot()  # what e0.step() would return now
    sim0._render_context_offscreen.gl_ctx.make_current()
    own = shot()  # e0 drawn inside its own context
    diff = np.abs(own.astype(np.int16) - foreign.astype(np.int16))
    try:
        import imageio.v2 as iio

        iio.imwrite(str(common.OUT / "glctx_compare.png"), np.concatenate([own[::-1], foreign[::-1]], axis=1))
    except Exception as exc:  # the picture is a convenience, the numbers are the result
        print("could not write the comparison picture:", exc)
    e0.close()
    e1.close()
    return {"mean_abs_pixel_diff": float(diff.mean()),
            "share_of_pixels_differing_by_more_than_10": float((diff.max(axis=-1) > 10).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--res", type=int, default=common.DEFAULT_RES)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import profile_step

    profile_step.instrument()
    payload: dict = {"args": vars(args)}
    if args.mixed:
        d = payload["mixed"] = mixed(args.res)
        print(f"same state drawn in its own context and in the other environment's context: mean absolute "
              f"pixel difference {d['mean_abs_pixel_diff']:.1f} of 255, "
              f"{100 * d['share_of_pixels_differing_by_more_than_10']:.0f}% of pixels differ by more than 10; "
              f"picture: {common.OUT / 'glctx_compare.png'}")
    else:
        rows = [run(k, args.steps, args.res, args.threads) for k in args.k]
        payload["rows"] = rows
        for r in rows:
            print(f"{r['envs_in_process']:2d} envs in one process: {r['ms_per_env_step']:.2f} ms per env step "
                  f"(physics {r['physics_ms']:.2f}, controller {r['controller_ms']:.2f}, render {r['render_ms']:.2f}); "
                  f"process uses {r['cpu_cores']:.2f} cores, Pss {r['pss_mib']:.0f} MiB; load {r['loadavg'].split()[0]}")
    print("saved", common.save(f"multi_env_one_process{'_mixed' if args.mixed else ''}{args.tag}.json", payload))


if __name__ == "__main__":
    main()
