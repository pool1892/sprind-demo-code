"""The LIBERO work cell behind `contracts.env.SimEnv` (docs/design/contracts.md, section 5).

Built on `fleetlab.sim.libero_env.LiberoEnv`, which owns the repeatable reset,
the control mode and the rendering context. What this adds:

- **The reset sequence with cell hooks.** Soft resets only (version 1): the
  simulator is reused, so a cell hook's change of the live MuJoCo model
  survives the reset and is undone by the environment at the end of the
  episode (`end_episode`, called by the rollout, or by the next reset or the
  close at the latest). Order: undo what is left, apply the cell hooks through
  the `CellHandle`, seed and reset, set the recorded start state, settle
  (`SETTLE_STEPS` no-op steps that bypass every hook), set the episode's
  control mode, first observation. The control mode comes from the identity's
  policy spec (the rollout hands in the effective spec); the image size is
  fixed when the renderer is built, so an identity with another size is
  refused.
- **Snapshot and restore**, the lean set (contracts, section 5): MuJoCo's
  integration state with the solver's warm start; the fixture bodies' poses
  as the reset placed them (those rows only); the gripper controllers'
  `goal_qvel` and the gripper models' integrated command; every sensor's
  sampling clock and last value; robosuite's step counter and clock; the
  observation delivered at that step, which a restore hands back instead of
  rendering (a re-render differs by one substep). Nothing a cell hook sets is
  in it (other bodies, controller ranges), so a restore cannot carry a fault
  into an episode that removed it. The restore ends with `mj_forward`.
- **Privileged state** for success checks and event detection: object poses
  and free-joint velocities by object name, contacts by body name (an
  object's bodies under the object's name, so that the task's `goal` finds
  them; every robot body included), the gripper opening, LIBERO's own check.
  A check that reads nothing before LIBERO's check holds
  (`needs_contacts_only_when_libero`) gets an empty state until then.
- **The task's goal** (`goal`, a `TaskGoal`) from the BDDL file's goal
  predicate; for "In", the target is the object that owns the region.
- **Success checks.** `libero_success.FleetCheck` is the fleet's check
  ("placed"); `LiberoCheck` is LIBERO's own, which the reference
  evaluation and every published LIBERO rate use. Both plug in the same way.
  `native_success` always holds LIBERO's predicate for the step, so the
  rollout records LIBERO's first-fire step whatever check is configured.
- **The chain's hashes**: `state_hash` over `mj_getState(INTEGRATION)` as
  float64 and `fixture_hash` over every model array a cell hook can change
  (`CELL_ARRAYS` and a hash per texture), with the contracts' hash.

One rendering context at a time (replay rule 6): every call that renders or
frees makes this environment's context current first.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import (
    REASON_LIBERO_CHECK,
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
from fleetlab.contracts.hashing import hash_digest
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.policy import Action7, ControlMode
from fleetlab.sim.libero_env import LiberoEnv, format_observation

FORMAT = "fleetlab.libero/1"
# The model arrays a cell hook can change (with the textures): the fixture hash covers them.
CELL_ARRAYS = (
    "body_pos",
    "body_quat",
    "body_mass",
    "body_inertia",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "geom_friction",
    "geom_size",
    "geom_rgba",
    "mat_rgba",
    "light_pos",
    "light_dir",
    "light_diffuse",
    "light_ambient",
    "light_specular",
    "light_active",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_gainprm",
    "actuator_biasprm",
)
PLAIN = (int, float, bool, str, type(None), np.integer, np.floating, np.bool_)
BUFFERS = ("DeltaBuffer", "RingBuffer")


class LiberoCheck:
    """LIBERO's own success check, the one the reference evaluation ends on: done when it holds."""

    tail_steps = 0
    needs_contacts_only_when_libero = True  # reads `libero_success` alone

    def reset(self, goal: TaskGoal) -> None:
        pass

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict:
        if state.libero_success:
            return SuccessVerdict(True, True, REASON_LIBERO_CHECK)
        return SuccessVerdict(False, False, "LIBERO's check has not held")

    def snapshot(self) -> bytes:
        return b""

    def restore(self, blob: bytes) -> None:
        pass


@dataclass(frozen=True)
class _Index:
    """Names and ids of one scene, built once: which body belongs to which object."""

    body_names: dict[int, str]  # every body: its object's name, or its MuJoCo name
    object_bodies: dict[str, int]  # object (and fixture) name -> root body id
    object_dofs: dict[str, int]  # object name -> first qvel index of its free joint
    goal: TaskGoal


class LiberoSimEnv:
    """One LIBERO task's work cell, rendered off-screen; implements `contracts.env.SimEnv`."""

    def __init__(self, suite: str, task_index: int, image_size: int) -> None:
        self._inner = LiberoEnv(
            suite, task_index, image_size=image_size, control_mode="absolute", hard_reset=False
        )
        self.spec = EnvSpec(
            suite=suite,
            task_index=task_index,
            image_size=image_size,
            control_mode=ControlMode.absolute,
            hard_reset=False,
        )
        # The rollout owns the step limit (replay rule 4), and a continuation may run beyond
        # LIBERO's horizon of 1,000 steps: robosuite never declares the episode done itself.
        self._rs.ignore_done = True
        self.task_description = self._inner.task_description
        self.max_steps = self._inner.max_steps
        self._check: SuccessCheck = LiberoCheck()
        self._undo: list[Undo] = []
        self._obs: Observation | None = None
        self._step = 0
        self.last_verdict = SuccessVerdict(False, False, "")
        self.native_success = False
        self._scene: _Index | None = None
        self._textures: list[bytes] | None = None  # a hash per texture, for the fixture hash
        # The task's goal from its BDDL file: predicate, the object to move, its target.
        self.goal: TaskGoal = self._index().goal

    # ---- the parts of robosuite this adapter reaches into ----------------------------------------

    @property
    def _rs(self) -> Any:
        return self._inner._env.env  # robosuite's environment inside LIBERO's wrapper

    @property
    def _sim(self) -> Any:
        return self._inner._env.sim

    # ---- SimEnv ------------------------------------------------------------------------------

    def reset(self, identity: EpisodeIdentity, cell_hooks: Sequence[CellHook]) -> Observation:
        if (identity.suite, identity.task_index) != (self.spec.suite, self.spec.task_index):
            raise ValueError(
                f"built for {self.spec.suite} task {self.spec.task_index}, "
                f"asked for {identity.suite} task {identity.task_index}"
            )
        if identity.image_size != self.spec.image_size:
            raise ValueError(
                f"the renderer was built for {self.spec.image_size} pixels, "
                f"the identity asks for {identity.image_size}"
            )
        mode = ControlMode(identity.policy.control_mode)
        self.end_episode()
        self._inner._make_current()
        for hook in cell_hooks:  # the simulator exists; nothing of this episode is rendered yet
            undo = hook(self.handle())
            if undo is not None:
                self._undo.append(undo)
        self._inner._relative = mode is ControlMode.relative
        self.spec = self.spec.model_copy(update={"control_mode": mode})
        raw_obs = self._inner.reset(identity.start_state, identity.scene_seed)
        self._check.reset(self.goal)
        self._step = 0
        self.native_success = False
        self.last_verdict = SuccessVerdict(False, False, "")
        self._obs = self._wrap(raw_obs)
        return self._obs

    def step(self, action: Action7) -> StepResult:
        self._inner._make_current()
        raw, _, _, _ = self._inner._env.step(np.asarray(action))
        self._step += 1
        self._obs = self._wrap(format_observation(raw, self._rs.robots[0]))
        libero = self._libero_success()
        if not libero and getattr(self._check, "needs_contacts_only_when_libero", False):
            # Before LIBERO's check holds, these checks read nothing else: skip the rest,
            # which costs 0.2 to 0.5 ms per step (spikes/success).
            state = PrivilegedState(object_poses={}, object_velocities={}, libero_success=False)
        else:
            state = self.privileged_state(libero)
        self.native_success = state.libero_success
        self.last_verdict = self._check.update(state, self._step)
        return StepResult(self._obs, self.last_verdict.done, self.last_verdict, self.native_success)

    def snapshot(self) -> EnvSnapshot:
        """The lean set (contracts, section 5), and nothing a cell hook sets.

        MuJoCo's integration state with the solver's warm start; the fixture
        bodies' poses as the reset placed them (only those rows: never every
        body, which would carry a cell hook's changes); the gripper controllers'
        `goal_qvel` and the gripper models' integrated `current_action` (never
        the controllers' ranges, which the reset rebuilds from the cell's
        actuators); every sensor's sampling clock and last value; robosuite's
        step counter and clock; the observation delivered at this step.
        """
        if self._obs is None:
            raise RuntimeError("no episode to snapshot")
        rs = self._rs
        model = self._sim.model._model
        fixtures = self._fixture_bodies()
        robot = rs.robots[0]
        state = {
            "mujoco": self._mj_state(),
            "fixtures": fixtures,
            "fixture_pos": model.body_pos[fixtures].copy(),
            "fixture_quat": model.body_quat[fixtures].copy(),
            "goal_qvel": {
                name: _copy(getattr(part, "goal_qvel", None))
                for name, part in robot.composite_controller.part_controllers.items()
                if "gripper" in name
            },
            "current_action": {
                arm: np.array(gripper.current_action, copy=True)
                for arm, gripper in robot.gripper.items()
            },
            "observables": {name: _snap_obj(o) for name, o in rs._observables.items()},
            "obs_cache": {k: np.array(v, copy=True) for k, v in rs._obs_cache.items()},
            "clock": (rs.timestep, rs.cur_time, rs.done),
            "step": self._step,
            "native_success": self.native_success,
        }
        return EnvSnapshot(
            format=FORMAT,
            state=pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),
            observation=copy.deepcopy(self._obs),
            step=self._step,
        )

    def restore(self, snapshot: EnvSnapshot) -> Observation:
        if snapshot.format != FORMAT:
            raise ValueError(f"not a snapshot of this adapter: {snapshot.format}")
        import mujoco

        self._inner._make_current()
        state = pickle.loads(snapshot.state)
        rs = self._rs
        model = self._sim.model._model
        fixtures = state["fixtures"]
        if list(fixtures) != list(self._fixture_bodies()):
            raise ValueError("the snapshot was taken in another scene")
        model.body_pos[fixtures] = state["fixture_pos"]
        model.body_quat[fixtures] = state["fixture_quat"]
        robot = rs.robots[0]
        for name, part in robot.composite_controller.part_controllers.items():
            if name in state["goal_qvel"]:
                part.goal_qvel = _copy(state["goal_qvel"][name])
        for arm, gripper in robot.gripper.items():
            gripper.current_action = np.array(state["current_action"][arm], copy=True)
        for name, observable in rs._observables.items():
            if name in state["observables"]:
                _restore_obj(observable, state["observables"][name], inplace=False)
        rs._obs_cache = {k: np.array(v, copy=True) for k, v in state["obs_cache"].items()}
        rs.timestep, rs.cur_time, rs.done = state["clock"]
        data = self._sim.data._data
        mujoco.mj_setState(
            model,
            data,
            np.asarray(state["mujoco"], np.float64),
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        # Positions, contacts and the other derived quantities still belong to the state before
        # the restore until MuJoCo recomputes them; an in-worker policy or the success check
        # reads them. It does not change the physics (spikes/teleop-absolute: 12 of 12
        # continuations identical with and without; the restore test here checks the chain).
        mujoco.mj_forward(model, data)
        self._step = state["step"]
        self.native_success = state["native_success"]
        self._obs = copy.deepcopy(snapshot.observation)
        return self._obs

    def _fixture_bodies(self) -> list[int]:
        """The root bodies of the fixtures, whose poses the reset draws (the furniture)."""
        rs = self._rs
        return sorted(int(rs.obj_body_id[name]) for name in rs.fixtures_dict)

    def privileged_state(self, libero: bool | None = None) -> PrivilegedState:
        """Object poses and free-joint velocities by object name, contacts by body name (an
        object's bodies under the object's name, every robot body included), the gripper
        opening, LIBERO's own check."""
        scene = self._index()
        data = self._sim.data._data
        poses = {
            name: np.concatenate([data.xpos[body], data.xquat[body]])
            for name, body in scene.object_bodies.items()
        }
        velocities = {
            name: np.array(data.qvel[dof : dof + 6], np.float64)
            for name, dof in scene.object_dofs.items()
        }
        qpos = self._obs["robot_state"]["gripper"]["qpos"] if self._obs is not None else np.zeros(2)
        return PrivilegedState(
            object_poses=poses,
            object_velocities=velocities,
            libero_success=self._libero_success() if libero is None else libero,
            contacts=self._contacts(scene),
            gripper_opening=float(qpos[0] - qpos[1]),
        )

    def _libero_success(self) -> bool:
        return bool(self._inner._env.check_success())

    def handle(self) -> CellHandle:
        sim = self._sim
        return CellHandle(
            model=sim.model._model,
            data=sim.data._data,
            env=self._rs,
            upload_texture=self._upload_texture,
        )

    def set_success_check(self, check: SuccessCheck) -> None:
        self._check = check

    def close(self) -> None:
        self.end_episode()
        self._inner.close()

    # ---- beyond the protocol: the chain's hashes, the end of an episode ---------------------

    def state_hash(self) -> bytes:
        return hash_digest(np.ascontiguousarray(self._mj_state()).tobytes())

    def fixture_hash(self) -> bytes:
        """Over every model array a cell hook can change, so a hook left behind shows.

        Body poses, masses and inertias; camera poses and fields of view;
        geometry friction, sizes and colours; material colours; lights;
        actuator ranges and gains; and the textures, by a hash per texture kept
        up to date by `upload_texture` (hashing all 190 MB of texture data takes
        0.3 s; a texture changed without an upload never reaches the images).
        """
        model = self._sim.model._model
        parts = [np.ascontiguousarray(getattr(model, name)).tobytes() for name in CELL_ARRAYS]
        parts.append(b"".join(self._texture_hashes()))
        return hash_digest(b"".join(parts))

    def end_episode(self) -> None:
        """Undo the cell hooks, last applied first (each undo repeats its texture upload)."""
        if self._undo:
            self._inner._make_current()
        while self._undo:
            self._undo.pop()()

    # ---- inside ------------------------------------------------------------------------------

    def _wrap(self, raw: dict[str, Any]) -> Observation:
        obs = dict(raw)
        obs["task_description"] = self.task_description
        obs["step"] = self._step
        return obs  # type: ignore[return-value]

    def _mj_state(self) -> npt.NDArray[np.float64]:
        import mujoco

        model, data = self._sim.model._model, self._sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        out = np.empty(mujoco.mj_stateSize(model, spec), np.float64)
        mujoco.mj_getState(model, data, out, spec)
        return out

    def _upload_texture(self, texture_id: int) -> None:
        self._inner._make_current()
        self._sim._render_context_offscreen.upload_texture(texture_id)
        if self._textures is not None:
            self._textures[texture_id] = self._texture_hash(texture_id)

    def _texture_hashes(self) -> list[bytes]:
        if self._textures is None:
            model = self._sim.model._model
            self._textures = [self._texture_hash(t) for t in range(model.ntex)]
        return self._textures

    def _texture_hash(self, texture_id: int) -> bytes:
        model = self._sim.model._model
        start = int(model.tex_adr[texture_id])
        size = int(model.tex_width[texture_id] * model.tex_height[texture_id])
        size *= int(model.tex_nchannel[texture_id])
        return hash_digest(np.ascontiguousarray(model.tex_data[start : start + size]).tobytes())

    def _contacts(self, scene: _Index) -> list[tuple[str, str]]:
        """Every pair of bodies in contact that involves an object, sorted, each pair once."""
        model, data = self._sim.model._model, self._sim.data._data
        objects = set(scene.object_bodies)
        pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            a = scene.body_names[int(model.geom_bodyid[c.geom1])]
            b = scene.body_names[int(model.geom_bodyid[c.geom2])]
            if a != b and (a in objects or b in objects):
                pairs.add((a, b) if a < b else (b, a))
        return sorted(pairs)

    def _index(self) -> _Index:
        if self._scene is None:
            self._scene = build_index(self._rs)
        return self._scene


def build_index(rs: Any) -> _Index:
    """Which bodies make up each LIBERO object, its free joint, and the task's goal."""
    import mujoco

    model = rs.sim.model._model
    roots: dict[str, int] = {name: int(body) for name, body in rs.obj_body_id.items()}
    owner = {body: name for name, body in roots.items()}
    names: dict[int, str] = {}
    for body in range(model.nbody):
        walk = body
        while walk not in owner and walk != 0:
            walk = int(model.body_parentid[walk])
        names[body] = (
            owner.get(walk)
            or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            or f"body{body}"
        )
    dofs = {}
    for name, obj in rs.objects_dict.items():
        joints = list(getattr(obj, "joints", []) or [])
        if not joints:
            continue
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joints[-1])
        if joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            dofs[name] = int(model.jnt_dofadr[joint])
    goal = rs.parsed_problem["goal_state"]
    if len(goal) != 1:
        raise NotImplementedError(f"a goal of {len(goal)} predicates: {goal}")
    predicate, obj_name, target_name = goal[0]
    target = target_name
    sites = getattr(rs, "object_sites_dict", {})
    if target_name in sites:  # "In": the target is a region of an object
        target = sites[target_name].parent_name
    return _Index(
        body_names=names,
        object_bodies=roots,
        object_dofs=dofs,
        goal=TaskGoal(predicate=predicate.capitalize(), object_body=obj_name, target_body=target),
    )


def make_check() -> SuccessCheck:
    return LiberoCheck()


# ---- the generic snapshot of robosuite's Python-side state (spikes/replay/snapshot.py) ------


def _copy(value: Any) -> Any:
    return np.array(value, copy=True) if isinstance(value, np.ndarray) else value


def _snap_value(value: Any, depth: int = 0) -> Any:
    if isinstance(value, np.ndarray):
        return ("nd", value.copy()) if value.dtype != object else None
    if isinstance(value, PLAIN):
        return ("pl", value)
    if type(value).__name__ in BUFFERS:
        return ("ob", _snap_obj(value))
    if depth < 3 and isinstance(value, list | tuple):
        items = [_snap_value(x, depth + 1) for x in value]
        if any(i is None for i in items):
            return None
        return ("li" if isinstance(value, list) else "tu", items)
    if depth < 3 and isinstance(value, dict):
        entries = {k: _snap_value(x, depth + 1) for k, x in value.items()}
        return None if any(i is None for i in entries.values()) else ("di", entries)
    return None


def _snap_obj(obj: Any) -> dict[str, Any]:
    return {k: s for k, v in vars(obj).items() if (s := _snap_value(v)) is not None}


def _restore_value(current: Any, saved: Any, inplace: bool) -> Any:
    """The value to store for a saved entry. Arrays are overwritten in place where they are,
    so that arrays robosuite's controllers share stay shared; sensor values are replaced,
    because observations already handed out refer to them."""
    kind = saved[0]
    if kind == "ob":
        _restore_obj(current, saved[1], inplace)
        return current
    if kind == "nd":
        array = saved[1]
        if (
            inplace
            and isinstance(current, np.ndarray)
            and current.shape == array.shape
            and current.dtype == array.dtype
            and current.flags.writeable
        ):
            current[...] = array
            return current
        return array.copy()
    if kind == "pl":
        return saved[1]
    if kind in ("li", "tu"):
        old = (
            list(current)
            if isinstance(current, list | tuple) and len(current) == len(saved[1])
            else [None] * len(saved[1])
        )
        items = [_restore_value(c, x, inplace) for c, x in zip(old, saved[1], strict=True)]
        return items if kind == "li" else tuple(items)
    if kind == "di":
        target = current if isinstance(current, dict) else {}
        for key in [k for k in target if k not in saved[1]]:
            del target[key]
        for key, x in saved[1].items():
            target[key] = _restore_value(target.get(key), x, inplace)
        return target
    raise ValueError(kind)


def _restore_obj(obj: Any, saved: dict[str, Any], inplace: bool = True) -> None:
    for key, entry in saved.items():
        setattr(obj, key, _restore_value(vars(obj).get(key), entry, inplace))


CellHookFn = Callable[[CellHandle], Undo | None]
