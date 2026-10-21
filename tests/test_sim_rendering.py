"""Two different scenes alive in one process each render their own images.

robosuite never switches back to an environment's rendering context, so the
adapter does. Without that, the environment that was not built or reset last
is drawn with the other scene's meshes and textures, and nothing fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from sim_helpers import IMAGE_SIZE, SEED, images

from fleetlab.sim.libero_env import NO_OP_ACTION, LiberoEnv

pytestmark = pytest.mark.sim


def first_step(env: LiberoEnv) -> dict[str, str]:
    """The images after a reset and one step. Repeatable, see test_sim_reset.py."""
    env.reset(init_state=0, seed=SEED)
    observation, _, _ = env.step(np.array(NO_OP_ACTION, dtype=np.float32))
    return images(observation)


# A soft reset keeps the rendering context, a hard reset makes a new one and leaves it
# current. Either way the other environment then has to switch back to its own.
@pytest.mark.parametrize("hard_reset", [False, True])
def test_two_scenes_in_one_process_render_their_own_images(hard_reset: bool) -> None:
    kitchen = LiberoEnv("libero_10", 3, image_size=IMAGE_SIZE, hard_reset=hard_reset)
    kitchen_alone = first_step(kitchen)  # the only rendering context in the process
    table = LiberoEnv("libero_spatial", 0, image_size=IMAGE_SIZE, hard_reset=hard_reset)
    table_built_last = first_step(table)  # its context is the current one in any case
    assert kitchen_alone != table_built_last

    assert first_step(kitchen) == kitchen_alone
    assert first_step(table) == table_built_last

    # The same with the other environment reset in between, not only built.
    kitchen.reset(init_state=0, seed=SEED)
    table.reset(init_state=0, seed=SEED)
    observation, _, _ = kitchen.step(np.array(NO_OP_ACTION, dtype=np.float32))
    assert images(observation) == kitchen_alone

    table.close()
    assert first_step(kitchen) == kitchen_alone  # closing one leaves the other intact
    kitchen.close()
