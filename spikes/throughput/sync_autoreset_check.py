"""Does lerobot-eval keep simulating an environment after its episode has ended?

Builds two environments the way lerobot-eval does with
--eval.use_async_envs=false, lets the first one report "finished" after five
steps (as a success would), and runs LeRobot's rollout pattern for 30
iterations with a do-nothing action. Then it prints how many steps and resets
each environment actually executed. No policy, no GPU model.

    tools/heavy-run --light spikes/throughput/run.sh sync_autoreset_check.py
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

import common


def main():
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    steps, resets, reset_s = Counter(), Counter(), Counter()
    orig_step, orig_reset = LiberoEnv.step, LiberoEnv.reset

    def step(self, action):
        obs, reward, terminated, truncated, info = orig_step(self, action)
        steps[self.episode_index] += 1
        if self.episode_index == 0 and steps[0] == 5:
            terminated = True  # stand-in for "task succeeded"
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        t0 = time.perf_counter()
        out = orig_reset(self, seed=seed, **kwargs)
        resets[self.episode_index] += 1
        reset_s[self.episode_index] += time.perf_counter() - t0
        return out

    LiberoEnv.step, LiberoEnv.reset = step, reset

    cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[0])
    env = make_env(cfg, n_envs=2, use_async_envs=False)["libero_spatial"][0]
    env.reset(seed=[1000, 1001], options={NEW_ROLLOUT_OPTION: True})
    action = np.tile(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32), (2, 1))
    done = np.array([False, False])
    iterations = 30
    for _ in range(iterations):  # lerobot_eval.rollout: step everything until all are done or max_steps
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated | done
    print(f"type of the vector env: {type(env).__name__}")
    for i in (0, 1):
        print(f"environment {i}: {steps[i]} simulated steps in {iterations} loop iterations, "
              f"{resets[i]} resets taking {reset_s[i]:.2f} s in total"
              + (" (reported finished after its 5th step)" if i == 0 else ""))
    print("saved", common.save("sync_autoreset_check.json",
                               {"iterations": iterations, "steps": dict(steps), "resets": dict(resets),
                                "reset_s": dict(reset_s), "loadavg": common.loadavg()}))
    env.close()


if __name__ == "__main__":
    main()
