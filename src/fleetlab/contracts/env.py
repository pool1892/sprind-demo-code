"""The environment adapter (docs/design/contracts.md, section 5).

The simulator sits behind `SimEnv`, so a second simulator runs through the same
rollout. The environment owns the reset sequence (`RESET_SEQUENCE`) and its
fixed moments; the rollout owns the action queue and the step limit, and the
environment never resets by itself (replay rule 4). Version 1 runs soft resets
only: a hard reset rebuilds the model inside `reset()` and wipes a cell hook.

`Observation` is the dictionary `fleetlab.sim.libero_env.format_observation`
returns (camera images under LeRobot's names, the robot state), plus the task
text and the step number, which the adapter adds. Images are stored as
rendered (rotated by 180 degrees in LIBERO); a policy adapter flips or resizes.

Choices this module makes where the note is silent:

- `EnvSnapshot` is opaque bytes plus the delivered observation: the bytes hold
  the lean set of section 5, in the adapter's own encoding; `format` names the
  adapter and its version so that a restore can refuse a foreign snapshot.
- `CellHandle.upload_texture` takes the MuJoCo texture id.

`new_episode` of every active fault runs before the cell hooks, which work on
the state it returns (section 7).

**The success check**. The fleet's check is *placed*: LIBERO's
check holds, no collision geometry of the robot touches the object, and the
object is at rest (`PLACED_THRESHOLDS`). Its verdict carries `done`, which
ends the episode: success with reason "placed" when placed holds; no success
with reason "LIBERO only" when `tail_steps` have passed since LIBERO's check
first held without placed; the rollout itself ends an episode at the step
limit with reason "step limit" (`VERDICT_REASONS`). A run configured with
LIBERO's own check (the reference mode the gate-2 pass mark compares with)
ends on it with reason "LIBERO's check"; an attempt the supervisor ended has
reason "stopped" or "handed over", and a handed-over episode's visible outcome
is `handed_over` (`contracts.log.Outcome`). The check is the engine's
to implement (it needs the adapter); the contracts fix its inputs
(`PrivilegedState`, `TaskGoal`), its verdict and its thresholds. Its state,
LIBERO's first-fire step or none, is part of a snapshot
(`SuccessCheck.snapshot`, carried by `contracts.rollout.Snapshot`): without it
a restore inside the tail restarts it and the continuation ends on another
step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypedDict, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import Field

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.policy import ControlMode

if TYPE_CHECKING:
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.policy import Action7


class Camera(StrEnum):
    """The two cameras of a robot, as named in the log, the videos and `get_frames`."""

    scene = "scene"
    wrist = "wrist"


# The key of each camera's image in `Observation["pixels"]` (LeRobot's names,
# kept so that LeRobot's preprocessing applies unchanged).
PIXEL_KEYS: dict[Camera, str] = {Camera.scene: "image", Camera.wrist: "image2"}

# What undoes a cell hook's change of the simulator model.
Undo = Callable[[], None]


class EndEffectorState(TypedDict):
    pos: npt.NDArray[np.float64]  # position, 3
    quat: npt.NDArray[np.float64]  # orientation as a quaternion, 4
    mat: npt.NDArray[np.float64]  # the controller's reference orientation, 3x3


class GripperState(TypedDict):
    qpos: npt.NDArray[np.float64]  # finger positions, 2
    qvel: npt.NDArray[np.float64]  # finger velocities, 2


class JointState(TypedDict):
    pos: npt.NDArray[np.float64]  # joint positions, 7
    vel: npt.NDArray[np.float64]  # joint velocities, 7


class RobotState(TypedDict):
    eef: EndEffectorState
    gripper: GripperState
    joints: JointState


class Observation(TypedDict):
    """What a step delivers: the images as rendered, the robot state, the task, the step."""

    pixels: dict[str, npt.NDArray[np.uint8]]  # keyed by the values of PIXEL_KEYS
    robot_state: RobotState
    task_description: str
    step: int


class EnvSpec(ContractModel):
    """How an environment is built. The control mode is set from the identity at reset.

    `image_size` is required: robosuite fixes the renderer's size when the
    environment is built, so `reset` refuses an identity with another size.
    """

    suite: str
    task_index: int = Field(ge=0)
    image_size: int = Field(ge=1)
    control_mode: ControlMode
    hard_reset: bool = False  # version 1: soft resets only, hard resets wipe cell hooks


class StepResult(NamedTuple):
    """One step's result: the observation, whether the episode is over, the configured
    check's verdict, and the simulator's own success predicate.

    `done` is the verdict's `done` (before the data hook, which the rollout
    applies). `native_success` is LIBERO's `check_success()` at this step,
    whatever check is configured, so that the truth side records LIBERO's
    first-fire step in every mode.
    """

    observation: Observation
    done: bool
    verdict: SuccessVerdict
    native_success: bool


@dataclass(frozen=True)
class EnvSnapshot:
    """The lean set the replay and teleoperator spikes measured as necessary to continue
    bit for bit.

    `state` holds, in the adapter's encoding: the MuJoCo state
    (`mjSTATE_INTEGRATION`) with the solver warm start; the gripper
    controller's `goal_qvel` and `current_action`; the positions and
    orientations of the fixture bodies only, as the reset placed them (never
    every body, and never the controller ranges, which the reset's cell hooks
    set: a restore must not carry a cell-hook fault back into an episode that
    removed it); the per-sensor clocks. The arm
    controller's goal is not saved: it is rebuilt from the next action in both
    control modes (replay spike for relative, teleop-absolute for absolute).
    `observation` is the observation delivered at that step: a restore hands it
    back instead of rendering a new one, because a re-render differs by one
    substep. The success check's state travels in `contracts.rollout.Snapshot`.
    """

    format: str
    state: bytes
    observation: Observation
    step: int


# The fleet check's thresholds, pre-registered (the success spike):
# the object is at rest when its linear speed is below `linear_m_s` and its
# angular speed below `angular_rad_s`; the check gives up `tail_steps` steps
# after LIBERO's check first held without placed (provisional).
PLACED_THRESHOLDS: dict[str, float] = {"linear_m_s": 0.01, "angular_rad_s": 0.15, "tail_steps": 60}

# The reasons a verdict that ends an episode gives (the `verdicts.true_reason` values).
REASON_PLACED = "placed"
REASON_LIBERO_ONLY = "LIBERO only"
REASON_STEP_LIMIT = "step limit"
REASON_LIBERO_CHECK = "LIBERO's check"  # a success by LIBERO's own check, the reference mode
REASON_STOPPED = "stopped"  # the supervisor stopped the attempt
REASON_HANDED_OVER = "handed over"  # the supervisor handed the attempt to a teleoperator
VERDICT_REASONS: tuple[str, ...] = (
    REASON_PLACED,
    REASON_LIBERO_ONLY,
    REASON_STEP_LIMIT,
    REASON_LIBERO_CHECK,
    REASON_STOPPED,
    REASON_HANDED_OVER,
)


@dataclass(frozen=True)
class TaskGoal:
    """The task's goal as its BDDL file states it: a predicate over two bodies.

    For example `On(akita_black_bowl_1, plate_1)`: `predicate` "On",
    `object_body` the body that has to move, `target_body` the one it has to
    reach. The success check finds both in `PrivilegedState` by these names.
    """

    predicate: str
    object_body: str
    target_body: str


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator internals for the success check and event detection, never the visible log.

    `object_poses` maps each object body to its position (3, world) and
    quaternion (4). `object_velocities` maps the same bodies to the six `qvel`
    values of their free joint: linear velocity in the world frame, angular
    velocity in the body frame. `contacts` lists every pair of bodies in
    contact that involves an object, and must include every robot body (arm
    links and gripper parts) against the objects, not only object against
    object: placed requires that no robot geometry touches the object.
    `libero_success` is LIBERO's own check (`env.check_success()`) at this step.
    """

    object_poses: dict[str, npt.NDArray[np.float64]]
    object_velocities: dict[str, npt.NDArray[np.float64]]
    libero_success: bool
    contacts: list[tuple[str, str]] = field(default_factory=list)
    gripper_opening: float = 0.0  # metres between the fingers


@dataclass(frozen=True)
class SuccessVerdict:
    """A success check's answer for one step: success, whether the episode is over, why.

    `done` ends the episode. A verdict with `success` is always `done`
    (reason "placed"); `done` without success has reason "LIBERO only" when
    the check gave up after its tail. A verdict that is not done carries no
    success, and its reason says what the check is waiting for.
    """

    success: bool
    done: bool
    reason: str

    def __post_init__(self) -> None:
        if self.success and not self.done:
            raise ValueError("a successful verdict ends the episode: success needs done")


@runtime_checkable
class SuccessCheck(Protocol):
    """A pluggable, stateful success check.

    The rollout calls `reset` with the task's goal (`SimEnv.goal`, naming the
    object and the target) at the start of an episode and `update` after
    every step, applies the same check to an original and its replays, and
    ends the episode on a `done` verdict as the data hook returns it, or at the
    step limit. `tail_steps` is the number of steps after LIBERO's check first
    held before the check gives up ("LIBERO only"). `snapshot` and `restore`
    carry the check's state (LIBERO's first-fire step or none) through a
    rollout snapshot, so a continuation ends on the same step.
    """

    tail_steps: int  # steps after LIBERO's first fire before the check gives up

    def reset(self, goal: TaskGoal) -> None: ...

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict: ...

    def snapshot(self) -> bytes: ...

    def restore(self, blob: bytes) -> None: ...


@dataclass(frozen=True)
class CellHandle:
    """What cell hooks and in-worker policies reach into: the live simulator.

    `model` and `data` are MuJoCo's, `env` the robosuite environment, and
    `upload_texture(texture_id)` pushes a changed texture to the renderer (a
    cell hook that changes a texture calls it, and again on undo).
    """

    model: Any
    data: Any
    env: Any
    upload_texture: Callable[[int], None]


@runtime_checkable
class CellHook(Protocol):
    """A change of the live simulator at the reset's cell-hook moment.

    Gets the handle and returns how to undo the change, or None if it changed
    nothing. The environment undoes all cell hooks at the end of the episode.
    """

    def __call__(self, handle: CellHandle) -> Undo | None: ...


class ResetStep(NamedTuple):
    name: str
    description: str


# The reset sequence, owned by the environment, in this order (skeleton, hook point 1).
RESET_SEQUENCE: tuple[ResetStep, ...] = (
    ResetStep("reuse", "reuse the simulator built before (soft reset; hard resets unsupported)"),
    ResetStep("new_episode", "the rollout gives every active fault a fresh state"),
    ResetStep(
        "apply_cell_hooks",
        "apply the cell hooks through the CellHandle: faults that change the live model "
        "(camera pose, physics parameters, object batch, textures with their upload)",
    ),
    ResetStep("seed_and_reset", "set the scene seed directly before the reset, then reset"),
    ResetStep("set_start_state", "set the task's recorded start state"),
    ResetStep(
        "settle",
        "run SETTLE_STEPS no-op steps so that objects come to rest; they bypass the "
        "observation and action hooks",
    ),
    ResetStep("set_control_mode", "set the episode's control mode on the arm controller"),
    ResetStep("first_observation", "deliver the first observation"),
)


@runtime_checkable
class SimEnv(Protocol):
    """One task's simulated work cell, rendered off-screen.

    Cell hooks are undone at the end of the episode, and a test proves it: the
    fixture hash of a healthy episode after a faulted one equals that of a
    healthy episode in a fresh process.
    """

    spec: EnvSpec
    task_description: str
    max_steps: int
    goal: TaskGoal  # read from the task's BDDL goal

    def reset(self, identity: EpisodeIdentity, cell_hooks: Sequence[CellHook]) -> Observation:
        """Run `RESET_SEQUENCE`; the control mode comes from the identity.

        Raises ValueError for an identity whose `image_size` differs from
        `spec.image_size`: the renderer's size is fixed when the environment is built.
        """
        ...

    def step(self, action: Action7) -> StepResult: ...

    def state_hash(self) -> bytes:
        """The chain's state hash: `contracts.hashing.hash_digest` of
        `mj_getState(mjSTATE_INTEGRATION)` as float64, 8 raw bytes."""
        ...

    def fixture_hash(self) -> bytes:
        """The chain's fixture hash, 8 raw bytes: over every model array a cell hook can
        change (body poses, camera poses, geometry friction, masses, lights, actuator
        ranges and texture data), so that a cell hook left behind by a faulted episode
        shows. The chain file stores it, as hex, after the reset or the restore."""
        ...

    def snapshot(self) -> EnvSnapshot: ...

    def restore(self, snapshot: EnvSnapshot) -> Observation:
        """Put the snapshot back and return its saved observation; never renders.

        Ends with `mj_forward`: once the state is set, MuJoCo's body and site
        positions and its contacts still belong to the state before the
        restore, and the privileged state, the cell hooks and an in-worker
        policy would read the wrong scene. The call changes nothing in the
        physics that follows (teleop-absolute: 12 of 12 continuations identical).
        """
        ...

    def privileged_state(self) -> PrivilegedState:
        """Object poses and velocities, contacts including every robot body against the
        objects, gripper opening, LIBERO's own check: for the success check and events."""
        ...

    def handle(self) -> CellHandle: ...

    def set_success_check(self, check: SuccessCheck) -> None: ...

    def close(self) -> None: ...
