"""The probe player: an in-worker policy that plays a probe's program (`contracts.probes`).

A probe episode's identity carries its program (`EpisodeIdentity.probe`) and names this
player as its policy (`PROBE_POLICY_NAME`). The engine builds one player per episode
from the run's `teleoperator` factory, as for the scripted teleoperator (contracts,
section 6), and asks it for one command per control step.

At its first step the player reads the end effector's position and orientation from
the observation, the robot's own report after every observation hook, and keeps them
as the probe's start. Step k then commands `start + offset[k]` with the start
orientation and the gripper command of `program_trajectory`. The commands are in the
frame the robot reports in, the frame the policy's absolute commands are in
(`commands_in_world_frame = False`), and they go through every action hook in force,
like the policy's commands: what the robot does with them is the controller's, the
hardware's and the command path's doing. In relative control the player sends the
saturated unit command toward its target (the controller's 5 cm per unit).

The player is deterministic: its commands depend on the program and on the episode's
own observations, so a probe episode replays from its reset bit for bit, like any
episode with its chain. It never reads the simulator's hidden state. After the
program's last step it reports `done`.

`NoServerPolicy` stands in for the server policy of a probe run: every job runs this
player in the worker, so nothing is loaded and no batch is ever served.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.policy import Action7, ActionChunk, ControlMode
from fleetlab.contracts.probes import Trajectory, program_trajectory

__all__ = [
    "PLAYER_FACTORY",
    "PROBE_POLICY_NAME",
    "RELATIVE_STEP_M",
    "SERVER_FACTORY",
    "NoServerPolicy",
    "ProbePlayer",
    "probe_spec",
]

PROBE_POLICY_NAME = "probe-player"
PLAYER_FACTORY = "fleetlab.engine.probe_player:ProbePlayer"
SERVER_FACTORY = "fleetlab.engine.probe_player:NoServerPolicy"
# Metres per unit of a relative position command: the operational-space controller's
# output limit as LIBERO configures it (`fleetlab.policies.teleop.actions.POS_STEP`).
RELATIVE_STEP_M = 0.05


def probe_spec(control_mode: ControlMode, precision: str = "fp32") -> PolicySpec:
    """The nominal policy spec of a probe episode: the player, in the robot's control mode."""
    return PolicySpec(
        name=PROBE_POLICY_NAME,
        revision="1",
        precision=precision,
        control_mode=control_mode,
        actions_executed_per_chunk=1,
        sampling_steps=None,
        image_views=1,
    )


class ProbePlayer:
    """In-worker: plays the program of the episode's identity, one command per step."""

    spec = probe_spec(ControlMode.absolute)
    chunk_length = 1
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = True
    # Targets are built on the position the robot reports: the policy's frame.
    commands_in_world_frame = False
    image_views = 1

    def __init__(self) -> None:
        self.done = False
        self.status = "not started"
        self._trajectory: Trajectory | None = None
        self._start: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._rotation: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._k = 0

    def load(self, device: str, precision: str) -> None:
        pass

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        if identity.probe is None:
            raise ValueError(f"{identity.episode_id} is not a probe episode: no program")
        if step != 0:
            raise ValueError("a probe episode plays from its reset, never from a snapshot")
        self._trajectory = program_trajectory(identity.probe.segments)
        self._start = None
        self._rotation = None
        self._k = 0
        self.done = False
        self.status = "playing"

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        raise NotImplementedError("an in-worker policy")

    def act_one(self, observation: Any, env: Any) -> Action7:
        if self._trajectory is None:
            raise ValueError("begin_episode first")
        eef = observation["robot_state"]["eef"]
        position = np.asarray(eef["pos"], np.float64)
        if self._start is None:
            self._start = position.copy()
            self._rotation = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
        assert self._rotation is not None
        last = len(self._trajectory.segment) - 1
        k = min(self._k, last)
        target = self._start + self._trajectory.offset[k]
        action = np.zeros(7, np.float64)
        if env.spec.control_mode is ControlMode.relative:
            action[:3] = np.clip((target - position) / RELATIVE_STEP_M, -1.0, 1.0)
        else:
            action[:3] = target
            action[3:6] = self._rotation
        action[6] = float(self._trajectory.gripper[k])
        self._k += 1
        if self._k > last:
            self.done, self.status = True, "played"
        return np.asarray(action, np.float32)

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        raise NotImplementedError("an in-worker policy")


class NoServerPolicy:
    """The server policy of a run whose every job is a probe: loads nothing, serves nothing."""

    spec = probe_spec(ControlMode.absolute)
    chunk_length = 1
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = False
    image_views = 1

    def load(self, device: str, precision: str) -> None:
        self.spec = probe_spec(ControlMode.absolute, precision)

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        raise RuntimeError("a probe run serves no policy calls")

    def act_one(self, observation: Any, env: Any) -> Action7:
        raise NotImplementedError("a server policy")

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        raise NotImplementedError("a server policy")
