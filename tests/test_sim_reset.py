"""A reset is repeatable: the seed and the stored start state decide the scene.

The fixes are in the vendored LIBERO, `third_party/libero-reset.patch`.
"""

from __future__ import annotations

import numpy as np
import pytest
from sim_helpers import IMAGE_SIZE, SEED, fingerprint, in_fresh_process

from fleetlab.sim.libero_env import LiberoEnv

pytestmark = pytest.mark.sim

# Two scenes with furniture, whose place LIBERO draws at every reset, and with a drawer
# whose opening is drawn from the same random numbers just before.
TASKS = [("libero_spatial", 4), ("libero_10", 3)]


def start_of_episode(suite: str, task: int, hard_reset: bool, history: bool) -> dict[str, str]:
    """Reset to start state 0 with SEED and fingerprint the result.

    With `history` the process has first built, used and closed another scene, and has
    played part of another episode in this one: other seed, other start state, the arm
    moved and the gripper closed.
    """
    if history:
        other = LiberoEnv("libero_spatial", 7, image_size=IMAGE_SIZE)
        other.reset(init_state=5, seed=7)
        other.close()
    env = LiberoEnv(suite, task, image_size=IMAGE_SIZE, hard_reset=hard_reset)
    if history:
        env.reset(init_state=7, seed=4242)
        for _ in range(30):
            env.step(np.array([0.4, 0.3, -0.6, 0.0, 0.0, 0.2, 1.0], dtype=np.float32))
    result = fingerprint(env, env.reset(init_state=0, seed=SEED))
    env.close()
    return result


@pytest.mark.parametrize(("suite", "task"), TASKS)
def test_an_episode_starts_the_same_whatever_came_before(suite: str, task: int) -> None:
    reference = in_fresh_process(start_of_episode, suite, task, True, False)

    for hard_reset, history in [(True, True), (False, True), (False, False)]:
        again = in_fresh_process(start_of_episode, suite, task, hard_reset, history)
        assert again == reference, f"hard_reset={hard_reset}, history={history}"


def test_the_seed_decides_where_the_furniture_stands() -> None:
    # Guards the test above: it would also pass if the furniture never moved.
    env = LiberoEnv("libero_spatial", 4, image_size=IMAGE_SIZE, hard_reset=False)
    one = fingerprint(env, env.reset(init_state=0, seed=SEED))
    other = fingerprint(env, env.reset(init_state=0, seed=SEED + 1))

    assert one["fixtures"] != other["fixtures"]
    assert one["image"] != other["image"]

    # A reset without a seed, as Gymnasium makes it for a finished environment in the
    # middle of an evaluation, lets the random numbers run on instead of restarting them
    # from the clock. A whole run of lerobot-eval then still repeats.
    np.random.seed(SEED)
    env._env.seed(None)
    continued = np.random.random()
    env.close()
    np.random.seed(SEED)
    assert continued == np.random.random()
