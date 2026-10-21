"""Where does one LIBERO environment step spend its time?

One environment, one process, no policy. The robosuite and LIBERO methods that
make up a step are wrapped with timers at class level, so the numbers are
inclusive wall-clock times per call; the table printed at the end subtracts the
nested parts from their parents.

    tools/heavy-run --light spikes/throughput/run.sh profile_step.py --res 360 256 128
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

import common

T: dict[str, float] = defaultdict(float)
N: dict[str, int] = defaultdict(int)


def wrap(cls, name: str, key):
    orig = getattr(cls, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            label = key(*a, **k) if callable(key) else key
            T[label] += time.perf_counter() - t0
            N[label] += 1

    setattr(cls, name, timed)


def instrument():
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.environments.base import MujocoEnv
    from robosuite.environments.robot_env import RobotEnv
    from robosuite.utils.binding_utils import MjRenderContext, MjSim

    from lerobot.envs.libero import LiberoEnv

    wrap(MjSim, "step1", "physics.step1")
    wrap(MjSim, "step2", "physics.step2")
    wrap(MjSim, "forward", "physics.forward")
    wrap(RobotEnv, "_pre_action", "controller")
    wrap(MujocoEnv, "_update_observables", "observables")
    wrap(MjSim, "render", lambda self, width=None, height=None, *, camera_name=None, **k: f"render.{camera_name}")
    wrap(MjRenderContext, "render", "render.draw")
    wrap(MjRenderContext, "read_pixels", "render.readback")
    wrap(MujocoEnv, "_post_action", "post_action")
    wrap(MujocoEnv, "_get_observations", "get_observations")
    for cls in set(TASK_MAPPING.values()):  # each problem class defines its own check
        if "_check_success" in cls.__dict__:
            wrap(cls, "_check_success", "check_success")
    wrap(ControlEnv, "step", "libero.step")
    wrap(LiberoEnv, "_format_raw_obs", "format_obs")
    wrap(LiberoEnv, "step", "wrapper.step")
    wrap(LiberoEnv, "reset", "wrapper.reset")


def run(res: int, steps: int, hard_reset: bool, resets: int, task_id: int) -> dict:
    T.clear()
    N.clear()
    rng = np.random.default_rng(0)

    mem_stages = {"after imports": common.proc_mem_mib().get("Pss", 0)}
    t0 = time.perf_counter()
    env = common.make_env(task_id=task_id, res=res, hard_reset=hard_reset)
    env.reset(seed=0)
    t_first = time.perf_counter() - t0
    mem_stages["after build + first reset"] = common.proc_mem_mib().get("Pss", 0)

    # Time later resets separately: these are what every further episode pays.
    reset_times = []
    for i in range(resets):
        t0 = time.perf_counter()
        env.reset(seed=i + 1)
        reset_times.append(time.perf_counter() - t0)
        mem_stages[f"after reset {i + 1}"] = common.proc_mem_mib().get("Pss", 0)

    T.clear()
    N.clear()
    # Small random end-effector deltas: the arm moves, contacts happen now and then.
    actions = rng.uniform(-0.3, 0.3, size=(steps, 7)).astype(np.float32)
    actions[:, 6] = np.sign(actions[:, 6])
    t0 = time.perf_counter()
    for a in actions:
        env.step(a)
    total = time.perf_counter() - t0
    sim = env._env.env.sim
    substeps = int(env._env.env.control_timestep / env._env.env.model_timestep)
    shape = {"nq": sim.model.nq, "nv": sim.model.nv, "nbody": sim.model.nbody, "ngeom": sim.model.ngeom,
             "physics_dt": env._env.env.model_timestep, "control_dt": env._env.env.control_timestep,
             "substeps": substeps}
    mem = common.proc_mem_mib()
    mem_stages["after steps"] = mem.get("Pss", 0)
    env.close()

    ms = {k: 1000 * v / steps for k, v in T.items()}  # milliseconds per environment step
    render_total = sum(v for k, v in ms.items() if k.startswith("render.") and k not in ("render.draw", "render.readback"))
    rows = {
        "physics (mj_step1 + mj_step2, all substeps)": ms["physics.step1"] + ms["physics.step2"],
        "controller (robot.control, all substeps)": ms["controller"],
        "render agentview": ms.get("render.agentview", 0.0),
        "render robot0_eye_in_hand": ms.get("render.robot0_eye_in_hand", 0.0),
        "  of which GPU draw (mjr_render)": ms.get("render.draw", 0.0),
        "  of which pixel readback (mjr_readPixels)": ms.get("render.readback", 0.0),
        "other observables (sensors, all substeps)": ms["observables"] - render_total,
        "post_action + LIBERO object states": ms["post_action"],
        "success check (called three times per step)": ms["check_success"],
        "collect observations": ms["get_observations"],
        "LeRobot wrapper formatting": ms["format_obs"],
    }
    accounted = sum(v for k, v in rows.items() if not k.startswith("  "))
    rows["unaccounted (Python between the parts)"] = 1000 * total / steps - accounted
    return {
        "res": res, "hard_reset": hard_reset, "steps": steps, "model": shape,
        "ms_per_step": 1000 * total / steps, "steps_per_s": steps / total,
        "rows_ms": rows, "calls_per_step": {k: N[k] / steps for k in N},
        "first_build_and_reset_s": t_first, "reset_s": reset_times, "mem_mib": mem,
        "mem_stages_pss_mib": mem_stages,
        "loadavg": common.loadavg(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[360])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--resets", type=int, default=3)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--soft", action="store_true", help="hard_reset=False")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    instrument()
    results = []
    for res in args.res:
        r = run(res, args.steps, not args.soft, args.resets, args.task)
        results.append(r)
        print(f"\n== {res}x{res}, two cameras, hard_reset={r['hard_reset']}: "
              f"{r['ms_per_step']:.2f} ms per step = {r['steps_per_s']:.0f} steps/s  (load {r['loadavg']})")
        print(f"   model {r['model']}")
        for k, v in r["rows_ms"].items():
            print(f"   {k:48s} {v:7.3f} ms  {100 * v / r['ms_per_step']:5.1f}%")
        print(f"   first build + reset {r['first_build_and_reset_s']:.2f} s; later resets "
              f"{', '.join(f'{x:.2f}' for x in r['reset_s'])} s")
        print("   memory (Pss, MiB): " + "; ".join(f"{k} {v:.0f}" for k, v in r["mem_stages_pss_mib"].items()))
    path = common.save(f"profile_step{args.tag}.json", {"results": results})
    print("\nsaved", path)


if __name__ == "__main__":
    main()
