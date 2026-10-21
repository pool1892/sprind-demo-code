"""A toy work cell, toy policies and toy faults: the engine's tests without simulator or GPU.

`ToyEnv` implements `contracts.env.SimEnv` with a few lines of deterministic
physics: an end effector that moves toward commanded poses, a gripper, one
object to carry to a target, a scene drawn from the scene seed, and 16x16
"camera images" computed from the state and the toy model's camera offset.
`ToyPolicy` is a server policy whose chunks depend on the observation and the
noise and on nothing else (row-independent by construction); it counts its
calls. `ToyTeleoperator` is an in-worker policy that reads the true object
position through the environment handle. The toy faults exercise every hook
point: a cell hook, an observation hook, an action hook (a delay line), a
policy input hook and a data hook.

Nothing here imports torch, mujoco or LIBERO.
"""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from fleetlab.contracts.env import (
    PIXEL_KEYS,
    REASON_PLACED,
    Camera,
    CellHandle,
    CellHook,
    EnvSnapshot,
    EnvSpec,
    Observation,
    PrivilegedState,
    StepResult,
    SuccessCheck,
    SuccessVerdict,
    TaskGoal,
    Undo,
)
from fleetlab.contracts.faults import (
    FaultSpec,
    FaultState,
    FaultTypeBase,
    TruthMapping,
)
from fleetlab.contracts.hashing import hash_digest
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec
from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.policy import Action7, ActionChunk, ControlMode, NoiseSample
from fleetlab.contracts.vocabulary import FaultClass, Verb
from fleetlab.faults import FaultRegistry

TOY_SUITE = "toy"
IMAGE = 16
FORMAT = "toy-env/1"


@dataclass
class ToyModel:
    """The toy simulator's model: what cell hooks change."""

    camera_shift: int = 0
    friction: float = 1.0


@dataclass
class ToyData:
    """The toy simulator's state."""

    eef: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    gripper: float = -1.0
    obj: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    held: bool = False
    closed_empty: bool = False
    goal: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))  # absolute target
    fixtures: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))


class ToyCheck:
    """Success when the object rests within 5 cm of the target, gripper open, for `hold` steps.

    Stateful like the fleet's check: it counts the steps the condition has held.
    """

    tail_steps = 0
    hold = 2

    def __init__(self) -> None:
        self.count = 0

    def reset(self, goal: TaskGoal) -> None:
        self.count = 0

    def snapshot(self) -> bytes:
        return pickle.dumps(self.count)

    def restore(self, blob: bytes) -> None:
        self.count = pickle.loads(blob)

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict:
        distance = float(
            np.linalg.norm(state.object_poses["object"][:3] - state.object_poses["target"][:3])
        )
        resting = distance < 0.05 and state.gripper_opening > 0.0
        self.count = self.count + 1 if resting else 0
        if self.count >= self.hold:
            return SuccessVerdict(True, True, REASON_PLACED)
        return SuccessVerdict(False, False, f"object {distance:.3f} m from the target")


class ToyEnv:
    """A deterministic work cell that implements `contracts.env.SimEnv`."""

    TASKS: ClassVar[list[str]] = ["carry the block to the plate", "put the cube on the mat"]

    def __init__(self, suite: str, task_index: int, image_size: int) -> None:
        if suite != TOY_SUITE:
            raise ValueError(f"the toy cell has suite {TOY_SUITE!r}, not {suite!r}")
        self.spec = EnvSpec(
            suite=suite,
            task_index=task_index,
            image_size=image_size,
            control_mode=ControlMode.absolute,
        )
        self.task_description = self.TASKS[task_index % len(self.TASKS)]
        self.max_steps = 60
        self.goal = TaskGoal("On", "object", "target")
        self.model = ToyModel()
        self.data = ToyData()
        self._check: SuccessCheck = ToyCheck()
        self._undo: list[Undo] = []
        self._step = 0
        self._obs: Observation | None = None
        self.last_verdict = SuccessVerdict(False, False, "")
        self.native_success = False
        self.resets = 0
        self.closed = False

    # ---- SimEnv ------------------------------------------------------------------------------

    def reset(self, identity: EpisodeIdentity, cell_hooks: Sequence[CellHook]) -> Observation:
        if (identity.suite, identity.task_index) != (self.spec.suite, self.spec.task_index):
            raise ValueError("this environment was built for another task")
        self.end_episode()
        self.spec = self.spec.model_copy(update={"control_mode": identity.policy.control_mode})
        for hook in cell_hooks:
            undo = hook(self.handle())
            if undo is not None:
                self._undo.append(undo)
        rng = np.random.RandomState(identity.scene_seed % 2**32)
        self.data = ToyData()
        self.data.fixtures = rng.uniform(-0.2, 0.2, 3)
        start = np.random.RandomState(identity.start_state).uniform(-0.1, 0.1, 3)
        start[2] = 0.0
        self.data.obj = np.array([0.3, 0.0, 0.0]) + start
        self.data.goal = self.data.eef.copy()
        self._step = 0
        self._check.reset(self.goal)
        self.resets += 1
        self._obs = self._observe()
        return self._obs

    def step(self, action: Action7) -> StepResult:
        a = np.asarray(action, np.float64)
        d = self.data
        if self.spec.control_mode is ControlMode.absolute:
            d.goal = a[:3].copy()
        else:
            d.goal = d.eef + 0.05 * a[:3]
        d.eef = d.eef + 0.3 * (d.goal - d.eef) / self.model.friction
        d.gripper = 1.0 if a[6] > 0 else -1.0
        if d.gripper > 0 and not d.closed_empty and np.linalg.norm(d.eef - d.obj) < 0.06:
            d.held = True
        d.closed_empty = d.gripper > 0 and not d.held
        if d.gripper < 0:
            d.held = False
        if d.held:
            d.obj = d.eef.copy()
        self._step += 1
        self._obs = self._observe()
        self.native_success = bool(np.linalg.norm(d.obj - self._target()) < 0.05 and not d.held)
        self.last_verdict = self._check.update(self.privileged_state(), self._step)
        return StepResult(self._obs, self.last_verdict.done, self.last_verdict, self.native_success)

    def snapshot(self) -> EnvSnapshot:
        assert self._obs is not None
        state = pickle.dumps((self.data, self._step))
        return EnvSnapshot(format=FORMAT, state=state, observation=self._obs, step=self._step)

    def restore(self, snapshot: EnvSnapshot) -> Observation:
        if snapshot.format != FORMAT:
            raise ValueError(f"not a toy snapshot: {snapshot.format}")
        self.data, self._step = pickle.loads(snapshot.state)
        self._obs = snapshot.observation
        return snapshot.observation

    def privileged_state(self) -> PrivilegedState:
        d = self.data
        return PrivilegedState(
            object_poses={
                "object": np.concatenate([d.obj, [1.0, 0, 0, 0]]),
                "target": np.concatenate([self._target(), [1.0, 0, 0, 0]]),
            },
            object_velocities={"object": np.zeros(6), "target": np.zeros(6)},
            libero_success=self.native_success,
            contacts=[("gripper0_finger", "object")] if d.held else [],
            gripper_opening=0.08 if d.gripper < 0 else 0.0,
        )

    def handle(self) -> CellHandle:
        return CellHandle(model=self.model, data=self.data, env=self, upload_texture=lambda _: None)

    def set_success_check(self, check: SuccessCheck) -> None:
        self._check = check

    def close(self) -> None:
        self.end_episode()
        self.closed = True

    # ---- for the chain -----------------------------------------------------------------------

    def state_hash(self) -> bytes:
        d = self.data
        values = np.concatenate([d.eef, [d.gripper], d.obj, [float(d.held)], d.goal])
        return hash_digest(np.ascontiguousarray(values).tobytes())

    def fixture_hash(self) -> bytes:
        """The fixtures and everything a toy cell hook changes (the camera, the friction)."""
        cell = np.array([self.model.camera_shift, self.model.friction], np.float64)
        return hash_digest(np.concatenate([self.data.fixtures, cell]).tobytes())

    def end_episode(self) -> None:
        """Undo the cell hooks, last first."""
        while self._undo:
            self._undo.pop()()

    # ---- inside ------------------------------------------------------------------------------

    def _target(self) -> npt.NDArray[np.float64]:
        return np.asarray(np.array([-0.3, 0.2, 0.0]) + 0.1 * self.data.fixtures, np.float64)

    def _observe(self) -> Observation:
        d = self.data
        size = self.spec.image_size
        scene = np.zeros((size, size, 3), np.uint8)
        for k, p in enumerate((d.eef, d.obj, self._target())):
            x = int(np.clip((p[0] + 0.5) * size, 0, size - 1))
            y = int(np.clip((p[1] + 0.5) * size, 0, size - 1))
            x = (x + self.model.camera_shift) % size
            scene[y, x, k] = 255
        wrist = np.full((size, size, 3), int(127 + 100 * d.gripper), np.uint8)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        return {
            "pixels": {PIXEL_KEYS[Camera.scene]: scene, PIXEL_KEYS[Camera.wrist]: wrist},
            "robot_state": {
                "eef": {"pos": d.eef.copy(), "quat": quat, "mat": np.eye(3)},
                "gripper": {
                    "qpos": np.array([0.04, -0.04])
                    * (1.0 if d.gripper < 0 else 0.5 if d.held else 0.0),
                    "qvel": np.zeros(2),
                },
                "joints": {"pos": np.concatenate([d.eef, d.obj, [d.gripper]]), "vel": np.zeros(7)},
            },
            "task_description": self.task_description,
            "step": self._step,
        }


TOY_SPEC = PolicySpec(
    name="toy",
    revision="1",
    precision="fp32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=5,
    sampling_steps=2,
    image_views=2,
)


def _find(image: npt.NDArray[np.uint8], channel: int) -> npt.NDArray[np.float64]:
    ys, xs = np.nonzero(image[:, :, channel])
    size = image.shape[0]
    if not len(xs):
        return np.zeros(3)
    return np.array([xs[0] / size - 0.5, ys[0] / size - 0.5, 0.0])


class ToyPolicy:
    """A server policy for the toy cell: reads object and target from the scene image."""

    chunk_length = 8
    needs_noise = True
    noise_shape: tuple[int, ...] | None = (8, 3)
    noise_dtype: str | None = "float32"
    row_independent = True
    in_worker = False
    image_views = 2

    def __init__(self) -> None:
        self.spec = TOY_SPEC
        self.calls: list[int] = []  # rows per call
        self.specs_seen: list[PolicySpec] = []
        self.loaded = ""

    def load(self, device: str, precision: str) -> None:
        self.loaded = f"{device}/{precision}"
        self.spec = self.spec.model_copy(update={"precision": precision})

    def act(
        self,
        batch: Sequence[Observation],
        instructions: Sequence[str],
        noise: Sequence[NoiseSample | None],
        spec: PolicySpec,
    ) -> list[ActionChunk]:
        self.calls.append(len(batch))
        self.specs_seen.append(spec)
        out = []
        scale = 0.002 * (spec.sampling_steps or 1)
        for obs, sample in zip(batch, noise, strict=True):
            scene = obs["pixels"][PIXEL_KEYS[Camera.scene]]
            eef = np.asarray(obs["robot_state"]["eef"]["pos"])
            finger = obs["robot_state"]["gripper"]["qpos"][0]
            held = 0.01 < finger < 0.03
            obj, target = _find(scene, 1), _find(scene, 2)
            goal = target if held else obj
            close = held or np.linalg.norm(eef[:2] - obj[:2]) < 0.05
            actions = np.zeros((self.chunk_length, 7), np.float32)
            for k in range(self.chunk_length):
                frac = (k + 1) / self.chunk_length
                pos = eef + frac * (goal - eef)
                actions[k, :3] = pos
                actions[k, 6] = 1.0 if close else -1.0
                if held and np.linalg.norm(eef[:2] - target[:2]) < 0.05:
                    actions[k, 6] = -1.0
                if not held and finger < 0.01:
                    actions[k, 6] = -1.0  # closed on nothing: open again
            if sample is not None:
                actions[:, :3] += scale * np.asarray(sample.values, np.float32)
            out.append(
                ActionChunk(
                    chunk_index=0 if sample is None else sample.chunk_index, actions=actions
                )
            )
        return out

    def act_one(self, observation: Observation, env: Any) -> Action7:
        raise NotImplementedError("a server policy")

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        return [np.asarray(a, np.float32) for a in chunk.actions]


TELEOP_SPEC = PolicySpec(
    name="toy-teleop",
    revision="1",
    precision="fp32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=1,
    sampling_steps=None,
    image_views=1,
)


class ToyTeleoperator:
    """An in-worker policy: reads the true object and target through the handle."""

    spec = TELEOP_SPEC
    chunk_length = 1
    needs_noise = False
    noise_shape: tuple[int, ...] | None = None
    noise_dtype: str | None = None
    row_independent = True
    in_worker = True
    commands_in_world_frame = True
    image_views = 1

    def __init__(self) -> None:
        self.began: list[tuple[str, int]] = []
        self.done = False
        self.status = "starting"

    def load(self, device: str, precision: str) -> None:
        pass

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        self.began.append((identity.episode_id, step))

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[ActionChunk]:
        raise NotImplementedError("an in-worker policy")

    def act_one(self, observation: Observation, env: Any) -> Action7:
        handle = env.handle()
        data: ToyData = handle.data
        target = env._target()
        a = np.zeros(7, np.float32)
        if self.done:
            a[:3], a[6] = data.eef, 0.0  # a hold: where it is, the gripper as it is
            return a
        if not data.held and np.linalg.norm(data.obj - target) < 0.03:
            self.done, self.status = True, "placed"
            a[:3], a[6] = data.eef, 0.0
            return a
        self.status = "carrying" if data.held else "reaching"
        if not data.held:
            a[:3] = data.obj
            a[6] = 1.0 if np.linalg.norm(data.eef - data.obj) < 0.03 else -1.0
        else:
            a[:3] = target
            a[6] = -1.0 if np.linalg.norm(data.eef - target) < 0.02 else 1.0
        return a

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Action7]:
        raise NotImplementedError("an in-worker policy")


# ---- toy faults --------------------------------------------------------------------------------


def _int(value: object) -> int:
    if not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return int(value)


class _Shift(BaseModel):
    pixels: int


class _Steps(BaseModel):
    steps: int


class _Detector(BaseModel):
    mode: str


def _component(verb: Verb) -> TruthMapping:
    from fleetlab.contracts.faults import Direction
    from fleetlab.contracts.vocabulary import TargetKind

    return TruthMapping(
        target_kind=TargetKind.component,
        fixing_verbs=frozenset({verb}),
        scopes=frozenset({ScopeKind.robot}),
        writes_change=False,
        directions=frozenset({Direction.degraded}),
    )


def _config() -> TruthMapping:
    from fleetlab.contracts.faults import Direction
    from fleetlab.contracts.vocabulary import TargetKind

    return TruthMapping(
        target_kind=TargetKind.config_key,
        fixing_verbs=frozenset({Verb.set, Verb.revert}),
        scopes=frozenset({ScopeKind.robot, ScopeKind.site, ScopeKind.fleet}),
        writes_change=True,
        directions=frozenset({Direction.misconfigured}),
    )


class ToyCameraShift(FaultTypeBase):
    """Cell hook: moves the toy scene camera by a few pixels; undone after the episode."""

    kind = "toy_camera_shift"
    fault_class = FaultClass.sensor
    Params = _Shift
    dose_levels = (1.0, 2.0)
    truth = _component(Verb.recalibrate)

    def cell_hook(self, handle: CellHandle, state: FaultState) -> Undo | None:
        model: ToyModel = handle.model
        before = model.camera_shift
        model.camera_shift = before + _int(state.spec.params["pixels"])

        def undo() -> None:
            model.camera_shift = before

        return undo


@dataclass
class DelayState(FaultState):
    line: deque[Action7] = field(default_factory=deque)


class ToyDelay(FaultTypeBase):
    """Action hook: each command is executed `steps` control steps late (a delay line)."""

    kind = "toy_delay"
    fault_class = FaultClass.software
    Params = _Steps
    dose_levels = (2.0, 4.0)
    truth = _config()

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return DelayState(spec=spec, episode_id=identity.episode_id)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, DelayState)
        state.line.append(np.asarray(action, np.float32).copy())
        if len(state.line) > _int(state.spec.params["steps"]):
            return state.line.popleft()
        return state.line[0]


@dataclass
class NoiseState(FaultState):
    calls: int = 0


class ToyImageNoise(FaultTypeBase):
    """Observation hook: a stateful pattern added to the wrist image (counts its calls)."""

    kind = "toy_image_noise"
    fault_class = FaultClass.sensor
    Params = _Steps
    dose_levels = (1.0,)
    truth = _component(Verb.clean)

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return NoiseState(spec=spec, episode_id=identity.episode_id)

    def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
        assert isinstance(state, NoiseState)
        state.calls += 1
        pixels = dict(observation["pixels"])
        wrist = pixels[PIXEL_KEYS[Camera.wrist]].copy()
        wrist[0, 0, 0] = state.calls % 256
        pixels[PIXEL_KEYS[Camera.wrist]] = wrist
        out = dict(observation)
        out["pixels"] = pixels
        return out  # type: ignore[return-value]


class ToyKnob(FaultTypeBase):
    """Policy input hook: changes the sampling steps the policy runs with."""

    kind = "toy_knob"
    fault_class = FaultClass.configuration
    Params = _Steps
    dose_levels = (5.0,)
    truth = _config()

    def policy_input_hook(self, policy_spec: PolicySpec, state: FaultState) -> PolicySpec:
        return policy_spec.model_copy(update={"sampling_steps": _int(state.spec.params["steps"])})


class ToyDetector(FaultTypeBase):
    """Data hook: the success detector misses a success ("miss") or reports a failure as a
    success ("false"). It may flip only a final verdict (contracts, section 5)."""

    kind = "toy_detector"
    fault_class = FaultClass.data
    Params = _Detector
    dose_levels = (1.0,)
    truth = _component(Verb.service)

    def data_hook(self, verdict: SuccessVerdict, state: FaultState) -> SuccessVerdict:
        assert isinstance(state, DetectorState)
        state.step += 1
        if not verdict.done:
            return verdict
        mode = state.spec.params.get("mode")
        if mode == "miss" and verdict.success:
            return SuccessVerdict(False, True, "detector missed it")
        if mode == "false" and not verdict.success:
            return SuccessVerdict(True, True, "detector saw a success")
        return verdict

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return DetectorState(spec=spec, episode_id=identity.episode_id)


@dataclass
class DetectorState(FaultState):
    step: int = 0


TOY_FAULTS = FaultRegistry()
for _type in (ToyCameraShift(), ToyDelay(), ToyImageNoise(), ToyKnob(), ToyDetector()):
    TOY_FAULTS.add(_type)
