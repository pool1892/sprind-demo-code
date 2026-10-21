"""What the reference launcher adds, on LeRobot's real environments but without a model.

`tests/test_reference_eval.py` checks the same on stand-ins. The evaluation
itself needs a model: `make smoke` and the earlier reference runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sim_helpers import SEED

from fleetlab import reference_eval

pytestmark = pytest.mark.sim


@pytest.mark.usefixtures("lerobot_on_robosuite_15")
def test_start_states_are_read_from_lerobots_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.envs import libero
    from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, FreezeAfterEpisodeEnd

    monkeypatch.setattr(libero, "_make_env_fns", libero._make_env_fns)  # put back afterwards
    reference_eval.freeze_finished_environments()  # the launcher's default
    config = LeRobotLiberoConfig(
        task="libero_spatial", task_ids=[0], observation_height=64, observation_width=64
    )
    # Built the way lerobot-eval builds them: one vector environment per task.
    env = make_env(config, n_envs=2, use_async_envs=False)["libero_spatial"][0]
    assert all(isinstance(single, FreezeAfterEpisodeEnd) for single in env.envs)

    def rollout(env: Any, seeds: list[int]) -> None:
        env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})  # how lerobot-eval begins one

    def eval_one(env: Any) -> dict[str, Any]:
        lerobot_eval.rollout(env, seeds=[SEED, SEED + 1])
        lerobot_eval.rollout(env, seeds=[SEED + 2, SEED + 3])
        return {"successes": [False] * 4}

    lerobot_eval = SimpleNamespace(rollout=rollout, eval_one=eval_one)
    reference_eval.record_start_states(lerobot_eval)
    metrics = lerobot_eval.eval_one(env)
    env.close()

    # Read before each round's reset. Two environments begin with states 0 and 1, and a
    # reset moves each on by as many states as there are environments.
    assert metrics["init_states"] == [0, 1, 2, 3]
