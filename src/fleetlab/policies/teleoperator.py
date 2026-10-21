"""The scripted teleoperator behind `contracts.policy.InWorkerPolicy`.

An in-worker policy: it reads the live simulator through the environment's
handle every step and never goes through the inference server. The engine
builds one per episode or continuation; the expert itself
(`fleetlab.policies.teleop.ScriptedExpert`) is built at the first step, from
the scene as it is then, which for a hand-over is the restored snapshot (the
restore ends with `mj_forward`, so the scene it reads is the right one).

- `done` and `status` are the expert's: done when it has placed the object or
  given up; after that it holds the present pose with its last gripper
  command, never the adapter's open no-op. The engine ends a continuation on
  `done`.
- The expert's only randomness, the subsampling of large point clouds in its
  grasp planner, is seeded from the identity's noise seed, so a take-over
  replays.
- It may command the gripper with 0, "hold the opening".
- The human model on top (hand-over delay, slower pace, fumbling) is
  `fleetlab.policies.teleop.human`; this is the expert alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.policy import Action7, ActionChunk, ControlMode

TELEOPERATOR_SPEC = PolicySpec(
    name="scripted-teleoperator",
    revision="teleop-absolute-v1",
    precision="fp32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=1,
    sampling_steps=None,
    image_views=1,
)
EXPERT_STEPS = 600  # the expert's own budget, as measured in its spike


class Teleoperator:
    """The scripted expert as an in-worker policy."""

    spec = TELEOPERATOR_SPEC
    chunk_length = 1
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = True
    image_views = 1

    @property
    def commands_in_world_frame(self) -> bool:
        return True

    def __init__(self) -> None:
        self.done = False
        self.status = "not started"
        self._expert: Any = None
        self._seed = 0

    def load(self, device: str, precision: str) -> None:
        pass

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        self._seed = identity.noise_seed % 2**32

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        raise NotImplementedError("an in-worker policy")

    def act_one(self, observation: Any, env: Any) -> Action7:
        if self._expert is None:
            from fleetlab.policies.teleop import ScriptedExpert

            mode = ControlMode(env.spec.control_mode)
            self._expert = ScriptedExpert(
                env.handle().env, control_mode=mode.value, max_steps=EXPERT_STEPS, seed=self._seed
            )
        action = np.asarray(self._expert.act(), np.float32)
        self.done = bool(self._expert.done)
        self.status = str(self._expert.status)
        return action

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        raise NotImplementedError("an in-worker policy")
