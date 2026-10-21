"""Scenes start the way the training data shows them, which depends on MuJoCo.

LIBERO's stored start states are not at rest: some objects are stored above
or inside the surface they belong on, and the physics settles them during the
reset. MuJoCo 3.8.1 settles four tasks differently from 3.3.7. Those two
versions are the ones measured here; LeRobot issue 4390 places the change at
3.4. The tests check both ways in which a start goes wrong, by where the
objects end up and not by the version number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sim_helpers import SEED

from fleetlab.sim.libero_env import LiberoEnv

pytestmark = pytest.mark.sim


def test_the_bowl_on_the_ramekin_starts_inside_it() -> None:
    # "pick up the black bowl on the ramekin and place it on the plate"
    env = LiberoEnv("libero_spatial", 5, image_size=64)
    env.reset(init_state=0, seed=SEED)
    data = env._env.sim.data._data
    bowl = data.body("akita_black_bowl_1_main").xpos
    ramekin = data.body("glazed_rim_porcelain_ramekin_1_main").xpos
    off_centre = float(np.linalg.norm((bowl - ramekin)[:2]))
    env.close()

    # MuJoCo 3.3.7: 1.45 cm, the bowl sits in the ramekin. MuJoCo 3.8.1: 3.7 cm, it hangs
    # tilted on the rim, and SmolVLA solved the task in 1 of 50 episodes instead of 18 of 30.
    assert off_centre < 0.02


# Height of the body's origin in metres with MuJoCo 3.3.7, where all three stand still.
AT_REST = {"ketchup_1_main": 0.5092, "orange_juice_1_main": 0.5065, "milk_1_main": 0.5065}


def test_the_bottles_beside_the_basket_start_at_rest() -> None:
    # "put both the alphabet soup and the tomato sauce in the basket". The ketchup, the
    # orange juice and the milk are stored about 3 cm inside the floor.
    import mujoco

    env = LiberoEnv("libero_10", 0, image_size=64)
    env.reset(init_state=0, seed=SEED)
    model, data = env._env.sim.model._model, env._env.sim.data._data

    def height_and_speed(name: str) -> tuple[Any, Any]:
        velocity = np.zeros(6)  # rotational, then linear
        body = model.body(name).id
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return data.xpos[body][2], np.linalg.norm(velocity[3:])

    found = {name: height_and_speed(name) for name in AT_REST}
    env.close()

    # MuJoCo 3.8.1 throws them out of the floor: after the reset they are 0.39 to 0.54 m
    # higher and still moving at 1.3 to 1.6 m/s, so the first camera image misses them.
    for name, (height, speed) in found.items():
        assert abs(height - AT_REST[name]) < 0.01, name
        assert speed < 0.01, name
