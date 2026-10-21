"""Side check for question 2 and 3: which instant does a delivered observation show?

robosuite samples each sensor on its own clock inside the 25 physics substeps
of a control step. This script plays the synthetic action script and, after
every control step, compares the delivered observation with
  (a) a re-read of the sensors without touching the simulator, and
  (b) a re-read after mj_forward, i.e. of the true end-of-step state.
The simulator and sensor state is snapshotted before and restored after each
comparison, so the episode itself is not disturbed.
"""

from __future__ import annotations

import copy

import numpy as np

import common as C
import engine as E
import snapshot as S


def main():
    env = C.make_env(observation_height=360, observation_width=360)
    obs = E.reset_episode(env, E.Spec(task=0, init=0, seed=1000, noise_seed=0))
    e = C.rs_env_of(env)
    cam = e._observables["agentview_image"]
    acts = C.synthetic_actions()
    rows, ages = [], []
    for a in acts:
        obs, *_ = env.step(a)
        obs = copy.deepcopy(obs)
        clock = cam._time_since_last_sample
        snap = S.take(env, obs)
        a_obs = copy.deepcopy(env._format_raw_obs(e._get_observations(force_update=True)))
        S.restore(env, snap)
        C.sim_of(env).forward()
        b_obs = copy.deepcopy(env._format_raw_obs(e._get_observations(force_update=True)))
        S.restore(env, snap)

        def cmp(o):
            img = np.abs(o["pixels"]["image"].astype(int) - obs["pixels"]["image"].astype(int))
            q = np.abs(o["robot_state"]["joints"]["pos"] - obs["robot_state"]["joints"]["pos"]).max()
            return (img > 0).mean(), int(img.max()), q

        # how old is the delivered reading? joint displacement divided by joint speed, joint by joint
        dq = a_obs["robot_state"]["joints"]["pos"] - obs["robot_state"]["joints"]["pos"]
        v = obs["robot_state"]["joints"]["vel"]
        fast = np.abs(v) > 0.05
        ages.extend((dq[fast] / v[fast]).tolist())
        rows.append((clock, *cmp(a_obs), *cmp(b_obs)))
    r = np.array(rows)
    print(f"steps: {len(r)}")
    print(f"camera clock after a control step: min {r[:, 0].min():.18f} max {r[:, 0].max():.18f} "
          f"distinct values {len(np.unique(r[:, 0]))}")
    for name, off in (("re-read without mj_forward", 1), ("re-read after mj_forward (true end of step)", 4)):
        same_img = int((r[:, off] == 0).sum())
        same_q = int((r[:, off + 2] == 0).sum())
        print(
            f"{name}: image identical to the delivered one in {same_img}/{len(r)} steps "
            f"(when different: on average {100 * r[r[:, off] > 0, off].mean() if same_img < len(r) else 0:.2f}% of values, "
            f"largest {int(r[:, off + 1].max())}/255); joint positions identical in {same_q}/{len(r)} steps "
            f"(largest difference {r[:, off + 2].max():.2e} rad)"
        )
    ages = 1000 * np.array(ages)
    print(
        f"age of the delivered joint reading, from displacement over speed: median {np.median(ages):.2f} ms, "
        f"5th to 95th percentile {np.percentile(ages, 5):.2f} to {np.percentile(ages, 95):.2f} ms "
        f"(one physics substep is {1000 * e.model_timestep:.0f} ms, a control step {1000 * e.control_timestep:.0f} ms)"
    )
    env.close()


if __name__ == "__main__":
    main()
