"""One episode of the scripted teleoperator in fleetlab's LIBERO adapter, from start to verdict.

An episode is described by an `EpisodeSpec` and run by `run_episode`. Three
kinds of start:

* `init`: the expert works from one of LIBERO's stored start states.
* a disturbed kind (`displaced`, `dropped`, `on_side`, `inverted`,
  `closed_empty`, `held`, `awkward`): `disturb.apply_scenario` turns the start
  state into the kind of state a failing policy leaves behind.
* `xvla`: an X-VLA episode recorded by the walking skeleton is played again
  from its recorded actions up to the hand-over step, with its state hash
  checked against the recording after every step (`xvla_record`).

For the last two the hand-over goes through a snapshot, as the rollout engine
will do it: the complete simulator state is saved (`snapshot.take`), the
environment is reset to another start state with another scene seed and moved,
the snapshot is put back, and the restored state's hash must equal the saved
one. Then the arm controller is switched to the expert's control mode, MuJoCo's
derived quantities are recomputed (`mj_forward`, so that positions and
contacts belong to the restored state; it does not change the physics, see the
README), and a fresh expert takes over.

Success is the strict outcome of spikes/teleop-expert: LIBERO's own check is
true after the expert has let go and backed off and the scene has had 15 more
steps to settle, and no finger touches the object.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fleetlab  # noqa: F401  (torch before mujoco)
import numpy as np

import snapshot
from disturb import KINDS, apply_scenario, describe
from fleetlab.sim.libero_env import LiberoEnv, camera_frame, set_relative_control
from teleop.actions import OPEN, ControlMode, controller_mode
from teleop.expert import ScriptedExpert
from xvla_record import RecordedEpisode, h64, mj_state

OUT = Path.home() / ".cache" / "robots-probes" / "teleop-absolute"
# robosuite ends an episode after `horizon` steps and then refuses to step. A take-over
# plays a set-up or a policy's 280 steps first, then up to 600 of the expert's.
HORIZON = 5000
SETTLE_AFTER = 15
EXPERT_MAX_STEPS = 600
MEASURE_IMAGE_SIZE = 64  # the expert does not look at the images; physics does not depend on their size


@dataclass
class EpisodeSpec:
    suite: str
    task: int
    init: int
    scene_seed: int
    scenario: str  # "init", one of disturb.KINDS, or "xvla"
    control_mode: ControlMode = "absolute"  # the expert's; set-up and X-VLA always run in absolute
    restore: bool = True  # hand over through snapshot, reset elsewhere, restore (not for "init")
    xvla_run: str | None = None
    xvla_episode: str | None = None
    handover_step: int | None = None  # for "xvla": steps of the recording played before the hand-over
    tag: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Chain:
    """Running hash over (action, simulator state after the step) of every step."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self.steps = 0

    def add(self, action: Any, state: np.ndarray) -> None:
        self._h.update(np.ascontiguousarray(action, dtype=np.float64).tobytes())
        self._h.update(np.ascontiguousarray(state).tobytes())
        self.steps += 1

    def digest(self) -> str:
        return self._h.hexdigest()[:16]


def make_env(suite: str, task: int, image_size: int = MEASURE_IMAGE_SIZE) -> LiberoEnv:
    """The adapter in absolute control with soft resets; per episode the mode is set again."""
    env = LiberoEnv(suite, task, image_size=image_size, control_mode="absolute", hard_reset=False)
    env._env.env.horizon = HORIZON
    return env


def set_mode(env: LiberoEnv, mode: ControlMode) -> None:
    for robot in env._env.robots:
        set_relative_control(robot, mode == "relative")


def forward(env: LiberoEnv) -> None:
    env._env.sim.forward()


def scramble(env: LiberoEnv, spec: EpisodeSpec) -> None:
    """Put the environment somewhere else: another start state, another scene seed, a few steps."""
    env.reset(init_state=(spec.init + 25) % 50, seed=spec.scene_seed + 7919)
    expert = ScriptedExpert(env._env, control_mode="absolute")
    for _ in range(3):
        env.step(expert.hold(OPEN))


class Recorder:
    """Video frames: both cameras side by side, a coloured bar on top (orange before, blue after the hand-over)."""

    def __init__(self, every: int) -> None:
        self.every = every
        self.frames: list[np.ndarray] = []

    def grab(self, obs: Any, step: int, phase: str) -> None:
        if step % self.every:
            return
        frame = camera_frame(obs).copy()
        colour = (255, 140, 0) if phase == "before" else (40, 110, 255)
        frame[: max(4, frame.shape[0] // 40)] = colour
        self.frames.append(frame)


def run_expert(env: LiberoEnv, mode: ControlMode, chain: Chain, recorder: Recorder | None = None,
               obs: Any = None) -> dict[str, Any]:
    forward(env)
    expert = ScriptedExpert(env._env.env, control_mode=mode, max_steps=EXPERT_MAX_STEPS)
    if recorder is not None and obs is not None:
        recorder.grab(obs, 0, "after")
    steps, error = 0, None
    while not expert.done:
        try:
            action = expert.act()
            obs, _, _ = env.step(action)
        except Exception as ex:  # noqa: BLE001  (simulation instability and the like)
            error = f"{type(ex).__name__}: {ex}"
            break
        chain.add(action, mj_state(env))
        steps += 1
        if recorder is not None:
            recorder.grab(obs, steps, "after")
    if error is None:
        for i in range(SETTLE_AFTER):
            action = expert.hold(OPEN)
            obs, _, _ = env.step(action)
            chain.add(action, mj_state(env))
            if recorder is not None:
                recorder.grab(obs, steps + i + 1, "after")
    touching = any(expert.sc.fingers_touching(expert.obj))
    success = bool(env._env.check_success()) and not touching and error is None
    status = expert.status if error is None else "fail:exception"
    if success and status != "success":
        status = "success_late"  # satisfied although the script had given up
    if not success and status == "success":
        status = "fail:unstable_after_release"
    result = dict(
        success=success,
        status=status,
        steps=steps,
        attempts=expert.attempt,
        libero_success_step=expert.first_success_step,
        hold_commands=expert.hold_commands,
        events=[f"{t}: {msg}" for t, msg in expert.events],
        error=error,
    )
    result["reason"] = failure_reason(result)
    return result


def failure_reason(result: dict[str, Any]) -> str:
    """One short label per failed run, from the status and the event log (as in spikes/teleop-expert)."""
    if result["success"]:
        return ""
    ev = " | ".join(result["events"])
    for key, label in [
        ("no top-down grasp", "no_grasp_found"),
        ("object_off_table", "object_off_table"),
        ("exception", "sim_exception"),
    ]:
        if key in ev or key in result["status"]:
            return label
    counts = {
        "grasp_missed": ev.count("grasp missed"),
        "slipped_in_lift": ev.count("slipped out while lifting"),
        "dropped_in_transit": ev.count("dropped in transit") + ev.count("dropped while"),
        "unreachable": ev.count("could not reach"),
        "placed_wrong": ev.count("not satisfied"),
    }
    top = max(counts, key=lambda k: counts[k])
    return top if counts[top] else result["status"].replace("fail:", "")


def _replay_xvla(env: LiberoEnv, rec: RecordedEpisode, steps: int, recorder: Recorder | None) -> tuple[bool, Any]:
    """Play the recording's first `steps` actions; False if any state hash differs from the recording."""
    if h64(mj_state(env)) != rec.state_hashes[0]:
        return False, None
    obs = None
    for t in range(steps):
        obs, _, _ = env.step(rec.actions[t])
        if h64(mj_state(env)) != rec.state_hashes[t + 1]:
            return False, obs
        if recorder is not None:
            recorder.grab(obs, t + 1, "before")
    return True, obs


def run_episode(env: LiberoEnv, spec: EpisodeSpec, recording: RecordedEpisode | None = None,
                recorder: Recorder | None = None) -> dict[str, Any]:
    """Run one episode in `env` (built for spec.suite and spec.task) and return its record."""
    t0 = time.time()
    row: dict[str, Any] = asdict(spec)
    chain = Chain()
    try:
        set_up_mode: ControlMode = spec.control_mode if spec.scenario == "init" else "absolute"
        obs = env.reset(init_state=spec.init, seed=spec.scene_seed)
        set_mode(env, set_up_mode)
        row["start_state_hash"] = int(h64(mj_state(env)))
        if recorder is not None:
            recorder.grab(obs, 0, "before" if spec.scenario != "init" else "after")
        if spec.scenario != "init":
            probe = ScriptedExpert(env._env, control_mode="absolute")  # for describing the hand-over state
            up_body = probe.sc.body_mat(probe.obj).T @ np.array([0.0, 0.0, 1.0])
            start_xy = probe.sc.body_pos(probe.obj)[:2].copy()
            if spec.scenario in KINDS:
                rng = np.random.default_rng(
                    [spec.scene_seed, spec.task, spec.init, sum(map(ord, spec.suite + spec.scenario))]
                )
                setup = apply_scenario(env, spec.scenario, rng, control_mode="absolute")
            elif spec.scenario == "xvla":
                assert recording is not None and spec.handover_step is not None
                ok, last = _replay_xvla(env, recording, spec.handover_step, recorder)
                setup = {"scenario": "xvla", "setup_steps": spec.handover_step, "xvla_success": recording.success,
                         "xvla_steps": len(recording.actions), "xvla_faults": recording.faults}
                if not ok:
                    setup["invalid"] = "replay of the recording differs from the recorded state hashes"
                obs = last if last is not None else obs
            else:
                raise ValueError(f"unknown scenario {spec.scenario!r}")
            if "invalid" not in setup:
                probe = ScriptedExpert(env._env, control_mode="absolute")
                setup.update(describe(probe, up_body, start_xy))
                obj_xy = probe.sc.body_pos(probe.obj)[:2]
                setup["object_to_target_m"] = round(float(np.linalg.norm(obj_xy - probe._goal_xy())), 3)
                setup["arm_joints"] = [round(float(v), 2) for v in probe.sc.d.qpos[probe.sc.arm_qpos_idx]]
                if spec.scenario == "xvla" and setup["already_success"] and not any(setup["touching"]):
                    setup["invalid"] = "task already satisfied at the hand-over"
            row["setup"] = setup
            if setup.get("invalid"):
                row.update(success=False, status="skipped:" + setup["invalid"], reason="setup_invalid",
                           steps=0, attempts=0)
                row["wall_s"] = round(time.time() - t0, 2)
                return row
            row["handover_state_hash"] = int(h64(mj_state(env)))
            row["handover_controller"] = controller_mode(env._env.robots[0].part_controllers["right"])
            if spec.restore:
                snap = snapshot.take(env)
                scramble(env, spec)
                row["scrambled_state_hash"] = int(h64(mj_state(env)))
                snapshot.restore(env, snap)
                row["restored_state_hash"] = int(h64(mj_state(env)))
                row["restore_exact"] = row["restored_state_hash"] == row["handover_state_hash"]
                if not row["restore_exact"]:
                    raise RuntimeError("the restored simulator state differs from the snapshot")
            set_mode(env, spec.control_mode)
        result = run_expert(env, spec.control_mode, chain, recorder, obs)
        row.update(result)
    except Exception:  # noqa: BLE001
        row.update(success=False, status="fail:harness_exception", reason="harness_exception", steps=0,
                   attempts=0, error=traceback.format_exc())
    row["chain"] = chain.digest()
    row["chain_steps"] = chain.steps
    row["final_state_hash"] = int(h64(mj_state(env)))
    row["wall_s"] = round(time.time() - t0, 2)
    return row
