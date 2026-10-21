"""What the fault types share: exact undo of model changes, per-episode random draws,
truth-mapping builders, the LIBERO names the cell hooks reach for, and the
calibration points.

Nothing here registers a fault type. The cell hooks import MuJoCo only when
they run, so importing a fault module costs nothing in a process without a
simulator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, JsonValue

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.env import Camera
from fleetlab.contracts.faults import Direction, FaultSpec, Target, TruthMapping
from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, TargetKind, Verb

if TYPE_CHECKING:
    from fleetlab.contracts.env import CellHandle, Observation
    from fleetlab.contracts.identity import EpisodeIdentity

# ---------------------------------------------------------------------------
# Names in LIBERO's scenes (robosuite 1.5, Panda with its default gripper).

SCENE_CAMERA = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"
# The arm's seven torque motors (robosuite's Panda), shoulder first.
ARM_MOTORS = tuple(f"robot0_torq_j{i}" for i in range(1, 8))
PAD_GEOMS = ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")
# Finger 1 closes toward the low end of its range, finger 2 toward the high end.
FINGER_ACTUATORS = (
    "gripper0_right_gripper_finger_joint1",
    "gripper0_right_gripper_finger_joint2",
)
# The materials whose textures make up each surface a backdrop change can repaint.
SURFACE_MATERIALS: dict[str, tuple[str, ...]] = {
    "walls": ("walls_mat",),
    "floor": ("floorplane",),
    "table": ("table_texture",),
}
CAMERA_NAMES = {Component.scene_camera: SCENE_CAMERA, Component.wrist_camera: WRIST_CAMERA}
CAMERAS = {Component.scene_camera: Camera.scene, Component.wrist_camera: Camera.wrist}


def config_key_of(spec: FaultSpec, allowed: set[ConfigKey]) -> ConfigKey:
    """The configuration key a fault targets; raises unless it is one of `allowed`."""
    key = spec.target.config_key
    if key not in allowed:
        raise ValueError(
            f"fault {spec.fault_id} ({spec.kind}) targets {key}, expected one of "
            f"{sorted(k.value for k in allowed)}"
        )
    return key


def camera_of(spec: FaultSpec) -> Component:
    """The camera a fault targets; raises for any other component."""
    component = spec.target.component
    if component not in CAMERA_NAMES:
        raise ValueError(f"fault {spec.fault_id} ({spec.kind}) targets {component}, not a camera")
    return component


# ---------------------------------------------------------------------------
# Params models


class FaultParams(BaseModel):
    """Base of every fault type's `Params`: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoParams(FaultParams):
    """A fault type whose level says everything."""


class ValueParams(FaultParams):
    """A changed value, as the two generic change kinds and the configuration knobs carry it."""

    value: ConfigValue


# ---------------------------------------------------------------------------
# Truth mappings, one builder per kind of target. The contracts check the rest.

DEGRADED = frozenset({Direction.degraded, Direction.drifting})
CONFIGURED = frozenset({Direction.misconfigured, Direction.changed})
SOFTWARE = frozenset({Direction.changed})
SITE = frozenset({Direction.changed, Direction.drifting})


def component_truth(verbs: set[Verb], directions: frozenset[Direction] = DEGRADED) -> TruthMapping:
    """A fault on a component of one robot (component faults are robot-scoped only)."""
    return TruthMapping(
        target_kind=TargetKind.component,
        fixing_verbs=frozenset(verbs),
        scopes=frozenset({ScopeKind.robot}),
        writes_change=False,
        directions=directions,
    )


def config_truth(directions: frozenset[Direction] = CONFIGURED) -> TruthMapping:
    """A fault that writes a `Change` of its configuration key: set it right, or revert it."""
    return TruthMapping(
        target_kind=TargetKind.config_key,
        fixing_verbs=frozenset({Verb.set, Verb.revert}),
        scopes=frozenset(ScopeKind),
        writes_change=True,
        directions=directions,
    )


def condition_truth(directions: frozenset[Direction] = SITE) -> TruthMapping:
    """A fault on a condition of one site; changing the condition back removes it."""
    return TruthMapping(
        target_kind=TargetKind.condition_key,
        fixing_verbs=frozenset({Verb.change_condition}),
        scopes=frozenset({ScopeKind.site}),
        writes_change=False,
        directions=directions,
    )


# ---------------------------------------------------------------------------
# Calibration points


@dataclass(frozen=True)
class DosePoint:
    """One setting the dose-response calibration runs: a level with its params and target.

    `target` names the thing the fault acts on (a component, configuration key
    or condition, by its vocabulary value); `nominal` is the value a
    configuration or condition fault replaces. `companions` names calibration
    points of other types, as (kind, label), that run in the same episodes: a
    data fault that shows only on failed attempts is measured on top of a
    fault that makes them (`success_false_positive`).
    """

    label: str
    level: float
    params: dict[str, JsonValue] = field(default_factory=dict)
    target: str = ""
    nominal: ConfigValue = None
    companions: tuple[tuple[str, str], ...] = ()


def default_points(
    levels: Sequence[float],
    target: str,
    params: dict[str, JsonValue] | None = None,
    nominal: ConfigValue = None,
    unit: str = "",
) -> tuple[DosePoint, ...]:
    """Calibration points for the non-zero levels, all with the same params and target."""
    return tuple(
        DosePoint(f"{level:g}{unit}", float(level), dict(params or {}), target, nominal)
        for level in levels
        if level > 0
    )


def target_of(scope: str, thing: str, kind: TargetKind) -> Target:
    """A `Target` from a scope and the vocabulary value of the thing it names."""
    if kind is TargetKind.component:
        return Target(scope=scope, component=Component(thing))
    if kind is TargetKind.config_key:
        return Target(scope=scope, config_key=ConfigKey(thing))
    return Target(scope=scope, condition_key=ConditionKey(thing))


# ---------------------------------------------------------------------------
# Per-episode random draws, independent of the policy's noise stream.

_SCHEME = "fleetlab.faults.v1"


def _digest(identity: EpisodeIdentity, spec: FaultSpec, purpose: str) -> bytes:
    text = f"{_SCHEME}\0{purpose}\0{spec.kind}\0{spec.fault_id}\0{identity.noise_seed}"
    return hashlib.sha256(text.encode()).digest()


def episode_rng(identity: EpisodeIdentity, spec: FaultSpec, purpose: str) -> np.random.Generator:
    """A generator for one fault in one episode: the same identity and fault give the same draws.

    Seeded from SHA-256 over the episode's noise seed, the fault's kind and id
    and a purpose, so it never shares a stream with the policy's noise
    (`noise_for`) or with another fault.
    """
    entropy = int.from_bytes(_digest(identity, spec, purpose)[:16], "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def episode_uniform(identity: EpisodeIdentity, spec: FaultSpec, purpose: str) -> float:
    """One number in [0, 1) for one fault in one episode, from the same hash."""
    return int.from_bytes(_digest(identity, spec, purpose)[:8], "big") / 2.0**64


# ---------------------------------------------------------------------------
# Exact undo of changes to the live model.


class ModelPatch:
    """Changes to arrays of the live MuJoCo model, each saved before it is made.

    `set(obj, name, index, value)` saves a copy of `getattr(obj, name)[index]`
    the first time that entry is touched and then writes the value. `undo()`
    writes every saved copy back, last change first, and re-uploads the
    textures that were touched, so that the model, and what the renderer
    holds of it, is bit for bit what it was.
    """

    def __init__(self, upload_texture: Callable[[int], None] | None = None) -> None:
        self._saved: list[tuple[Any, str, Any, npt.NDArray[Any]]] = []
        self._seen: set[tuple[int, str, str]] = set()
        self._textures: list[int] = []
        self._upload = upload_texture

    def set(self, obj: Any, name: str, index: Any, value: Any) -> None:
        array = getattr(obj, name)
        key = (id(obj), name, repr(index))
        if key not in self._seen:
            self._seen.add(key)
            self._saved.append((obj, name, index, np.array(array[index], copy=True)))
        array[index] = value

    def texture(self, tex_id: int) -> None:
        """Record that a texture changed: upload it now, and again after the undo."""
        if self._upload is None:
            raise RuntimeError("a texture changed but the handle cannot upload textures")
        if tex_id not in self._textures:
            self._textures.append(tex_id)
        self._upload(tex_id)

    @property
    def empty(self) -> bool:
        return not self._saved

    def undo(self) -> None:
        for obj, name, index, saved in reversed(self._saved):
            getattr(obj, name)[index] = saved
        self._saved.clear()
        self._seen.clear()
        if self._upload is not None:
            for tex_id in self._textures:
                self._upload(tex_id)
        self._textures.clear()


def mj_id(model: Any, kind: str, name: str) -> int:
    """The id of a named MuJoCo object; raises KeyError when the scene has none."""
    import mujoco

    obj = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "material": mujoco.mjtObj.mjOBJ_MATERIAL,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
    }[kind]
    found = int(mujoco.mj_name2id(model, obj, name))
    if found < 0:
        raise KeyError(f"the scene has no {kind} named {name!r}")
    return found


def subtree(model: Any, root: int) -> list[int]:
    """The bodies below `root`, `root` included, in id order."""
    out = []
    for body in range(int(model.nbody)):
        b = body
        while b not in (root, 0):
            b = int(model.body_parentid[b])
        if b == root:
            out.append(body)
    return out


# ---------------------------------------------------------------------------
# Rotations, in numpy only (scipy is not a dependency of the package).


def quat_mul(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The product of two quaternions (w, x, y, z), as MuJoCo's `mju_mulQuat`."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def axis_angle_quat(axis: Sequence[float], radians: float) -> npt.NDArray[np.float64]:
    """The quaternion (w, x, y, z) of a turn about a unit axis, as MuJoCo's `mju_axisAngle2Quat`."""
    half = radians / 2.0
    return np.array([np.cos(half), *(np.sin(half) * np.asarray(axis, dtype=np.float64))])


def rotvec_to_matrix(rotvec: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Rodrigues' formula: a rotation vector (axis times angle) as a 3x3 matrix."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    out: npt.NDArray[np.float64] = (
        np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)
    )
    return out


def matrix_to_rotvec(matrix: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """A 3x3 rotation matrix as a rotation vector, the inverse of `rotvec_to_matrix`."""
    r = np.asarray(matrix, dtype=np.float64)
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:  # near half a turn the antisymmetric part vanishes
        diag = np.sqrt(np.clip((np.diag(r) + 1.0) / 2.0, 0.0, None))
        axis = diag.copy()
        i = int(np.argmax(diag))
        for j in range(3):
            if j != i:
                axis[j] = np.copysign(diag[j], r[i, j] + r[j, i])
        half_turn: npt.NDArray[np.float64] = axis / np.linalg.norm(axis) * angle
        return half_turn
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (
        2.0 * np.sin(angle)
    )
    out: npt.NDArray[np.float64] = axis * angle
    return out


# ---------------------------------------------------------------------------
# Observations are never changed in place: a hook returns a new dictionary.


def with_image(observation: Observation, key: str, image: npt.NDArray[np.uint8]) -> Observation:
    """A copy of the observation with one camera image replaced."""
    pixels = {**observation["pixels"], key: image}
    return {**observation, "pixels": pixels}


def with_eef_pos(observation: Observation, pos: npt.NDArray[np.float64]) -> Observation:
    """A copy of the observation with the end effector's reported position replaced."""
    state = observation["robot_state"]
    eef = {**state["eef"], "pos": pos}
    return {**observation, "robot_state": {**state, "eef": eef}}  # type: ignore[typeddict-item]


def handle_env(handle: CellHandle) -> Any:
    """The robosuite environment behind a handle (LIBERO's wrapper holds it as `env`)."""
    env = handle.env
    inner = getattr(env, "env", None)
    return inner if inner is not None and hasattr(inner, "robots") else env
