"""Environment and policy adapters, the live layer, rollout jobs and world files.

docs/design/contracts.md, sections 5, 6, 9, 10 and 14.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import numpy as np
import pytest
from pydantic import ValidationError

from fleetlab.contracts import env, live, policy, rollout, world
from fleetlab.contracts.env import (
    PIXEL_KEYS,
    PLACED_THRESHOLDS,
    REASON_LIBERO_ONLY,
    REASON_PLACED,
    RESET_SEQUENCE,
    VERDICT_REASONS,
    EndEffectorState,
    EnvSpec,
    GripperState,
    JointState,
    Observation,
    PrivilegedState,
    RobotState,
    SuccessCheck,
    SuccessVerdict,
    TaskGoal,
)
from fleetlab.contracts.faults import (
    Direction,
    Drift,
    FaultSpec,
    Schedule,
    Target,
    check_data_hook,
)
from fleetlab.contracts.hashing import hash_bytes, hash_digest, hash_json
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.live import Decision, DecisionRecord, StepTelemetry, hold_action
from fleetlab.contracts.log import Outcome, allowed_columns
from fleetlab.contracts.policy import ActionChunk, ControlMode, NoiseSample, check_policy_flags
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SnapshotRef, chain_sample
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.sim import libero_env

SPEC = PolicySpec(
    name="xvla-libero",
    revision="12e8783e",
    precision="float32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=30,
    sampling_steps=10,
    image_views=3,
)
FAULT = FaultSpec(
    fault_id="f01",
    kind="action_delay",
    params={"steps": 3},
    level=3.0,
    target=Target(scope="s2", config_key=ConfigKey.control_action_delay_steps),
    schedule=Schedule(onset_day=0),
    nominal=0,
)


def identity(faults: list[FaultSpec] | None = None) -> EpisodeIdentity:
    scene, noise = seeds_for(1, "r03", 2, 0)
    return EpisodeIdentity(
        world_id="w01",
        episode_id="e000001",
        robot_id="r03",
        site_id="s2",
        day=2,
        episode_index=0,
        suite="libero_spatial",
        task_index=0,
        start_state=0,
        scene_seed=scene,
        noise_seed=noise,
        policy=SPEC,
        batch_size=4,
        row=None,
        image_size=360,
        max_steps=280,
        faults=faults if faults is not None else [FAULT],
        software=SoftwareVersion(
            lock_hash="a", libero_patch_set="p", mujoco_version="3.3.7", miopen_db_hash="m"
        ),
    )


# Environment.


def test_observation_matches_what_the_libero_adapter_formats() -> None:
    controller = SimpleNamespace(ref_ori_mat=np.eye(3))
    fake_robot = SimpleNamespace(part_controllers={"right": controller})
    raw = {
        "agentview_image": np.zeros((4, 4, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_gripper_qvel": np.zeros(2),
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
    }
    formatted = libero_env.format_observation(raw, fake_robot)
    added = {"task_description", "step"}  # the adapter adds these two
    assert set(formatted) | added == set(get_type_hints(Observation))
    state = formatted["robot_state"]
    assert set(state) == set(get_type_hints(RobotState))
    assert set(state["eef"]) == set(get_type_hints(EndEffectorState))
    assert set(state["gripper"]) == set(get_type_hints(GripperState))
    assert set(state["joints"]) == set(get_type_hints(JointState))
    assert set(formatted["pixels"]) == set(PIXEL_KEYS.values()) == set(libero_env.CAMERAS.values())


def test_the_reset_sequence_has_its_fixed_moments() -> None:
    names = [step.name for step in RESET_SEQUENCE]
    assert names == [
        "reuse",
        "new_episode",
        "apply_cell_hooks",
        "seed_and_reset",
        "set_start_state",
        "settle",
        "set_control_mode",
        "first_observation",
    ]
    spec = EnvSpec(
        suite="libero_spatial", task_index=0, image_size=360, control_mode=ControlMode.absolute
    )
    assert spec.hard_reset is False
    with pytest.raises(ValidationError):  # the renderer's size is fixed at construction
        EnvSpec.model_validate(
            {"suite": "libero_spatial", "task_index": 0, "control_mode": "absolute"}
        )


GOAL = TaskGoal(predicate="On", object_body="bowl_1", target_body="plate_1")
ROBOT_BODIES = frozenset({"gripper0_left_finger", "gripper0_right_finger", "robot0_link7"})


class Placed:
    """A toy of the fleet's check, to exercise the contract: the engine ships the real one.

    Placed when LIBERO's check holds, no robot body touches the object and the
    object is at rest; gives up `tail_steps` after LIBERO's first fire.
    """

    def __init__(self, tail_steps: int) -> None:
        self.tail_steps = tail_steps
        self.first_fire: int | None = None

    def reset(self, goal: TaskGoal) -> None:
        assert goal == GOAL
        self.first_fire = None

    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict:
        if state.libero_success and self.first_fire is None:
            self.first_fire = step
        body = GOAL.object_body
        touched = any(body in pair and set(pair) & ROBOT_BODIES for pair in state.contacts)
        velocity = state.object_velocities[body]
        at_rest = (
            np.linalg.norm(velocity[:3]) < PLACED_THRESHOLDS["linear_m_s"]
            and np.linalg.norm(velocity[3:]) < PLACED_THRESHOLDS["angular_rad_s"]
        )
        if state.libero_success and not touched and at_rest:
            return SuccessVerdict(True, True, REASON_PLACED)
        if self.first_fire is not None and step - self.first_fire >= self.tail_steps:
            return SuccessVerdict(False, True, REASON_LIBERO_ONLY)
        return SuccessVerdict(False, False, "not placed")

    def snapshot(self) -> bytes:
        return json.dumps(self.first_fire).encode()

    def restore(self, blob: bytes) -> None:
        self.first_fire = json.loads(blob)


def state(libero: bool, held: bool, speed: float = 0.0) -> PrivilegedState:
    contacts = [("bowl_1", "plate_1")] + ([("gripper0_left_finger", "bowl_1")] if held else [])
    return PrivilegedState(
        object_poses={"bowl_1": np.zeros(7)},
        object_velocities={"bowl_1": np.array([speed, 0, 0, 0, 0, 0], dtype=np.float64)},
        libero_success=libero,
        contacts=contacts,
    )


def test_the_success_check_ends_the_episode_with_done() -> None:
    check = Placed(tail_steps=3)
    assert isinstance(check, SuccessCheck)
    # Held above the plate: LIBERO fires, placed does not; the check gives up after the tail.
    held = [check.update(state(libero=True, held=True), step) for step in range(10, 14)]
    assert [(v.success, v.done) for v in held] == [(False, False)] * 3 + [(False, True)]
    assert held[-1].reason == REASON_LIBERO_ONLY
    # Released, but still moving: not yet; at rest: placed.
    check.reset(GOAL)
    assert not check.update(state(libero=True, held=False, speed=0.05), 20).done
    placed = check.update(state(libero=True, held=False), 21)
    assert (placed.success, placed.done, placed.reason) == (True, True, REASON_PLACED)


def test_the_check_s_state_survives_a_snapshot() -> None:
    check = Placed(tail_steps=3)
    check.update(state(libero=True, held=True), 10)  # LIBERO's first fire
    saved = check.snapshot()
    check.reset(GOAL)
    check.restore(saved)  # a restore inside the tail continues it instead of restarting it
    assert check.update(state(libero=True, held=True), 13).done


def test_the_verdict_and_its_constants() -> None:
    with pytest.raises(ValueError, match="success needs done"):
        SuccessVerdict(True, False, REASON_PLACED)
    assert VERDICT_REASONS == (
        "placed",
        "LIBERO only",
        "step limit",
        "LIBERO's check",
        "stopped",
        "handed over",
    )
    assert [o.value for o in Outcome] == ["success", "failure", "handed_over"]
    assert PLACED_THRESHOLDS == {"linear_m_s": 0.01, "angular_rad_s": 0.15, "tail_steps": 60}
    assert "goal" in env.SimEnv.__annotations__
    assert {f.name for f in dataclasses.fields(PrivilegedState)} >= {
        "object_poses",
        "object_velocities",
        "contacts",
        "gripper_opening",
        "libero_success",
    }


def test_a_data_hook_flips_only_a_done_verdict() -> None:
    done = SuccessVerdict(True, True, REASON_PLACED)
    check_data_hook(done, SuccessVerdict(False, True, "the detector missed it"))
    waiting = SuccessVerdict(False, False, "not placed")
    check_data_hook(waiting, waiting)
    with pytest.raises(ValueError):  # a success needs done, which the hook may not set
        check_data_hook(waiting, SuccessVerdict(True, True, REASON_PLACED))
    with pytest.raises(ValueError, match="done"):
        check_data_hook(done, SuccessVerdict(False, False, "kept running"))


def test_a_restore_ends_with_mj_forward_and_the_arm_goal_is_not_saved() -> None:
    restore_doc = env.SimEnv.restore.__doc__ or ""
    assert "mj_forward" in restore_doc and "never renders" in restore_doc
    snapshot_doc = " ".join((env.EnvSnapshot.__doc__ or "").split())
    assert "arm controller's goal is not saved" in snapshot_doc
    assert "goal_qvel" in snapshot_doc and "current_action" in snapshot_doc


def test_the_environment_protocol_has_the_note_s_methods() -> None:
    for name in (
        "reset",
        "step",
        "snapshot",
        "restore",
        "privileged_state",
        "handle",
        "set_success_check",
        "close",
    ):
        assert callable(getattr(env.SimEnv, name))
    assert {f.name for f in dataclasses.fields(env.CellHandle)} == {
        "model",
        "data",
        "env",
        "upload_texture",
    }


# Policies.


class ServerPolicy:
    spec = SPEC
    chunk_length = 30
    needs_noise = True
    noise_shape: tuple[int, ...] | None = (30, 20)
    noise_dtype: str | None = "float32"
    row_independent = True
    in_worker = False
    image_views = 3

    def load(self, device: str, precision: str) -> None:
        pass

    def act(
        self,
        batch: Sequence[Observation],
        instructions: Sequence[str],
        noise: Sequence[NoiseSample | None],
        spec: PolicySpec,
    ) -> list[ActionChunk]:
        assert spec.sampling_steps is not None  # the group's effective spec drives the loop
        return [ActionChunk(0, np.zeros((30, 20), np.float32)) for _ in batch]

    def act_one(self, observation: Observation, env: Any) -> Any:
        raise NotImplementedError

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Any]:
        return [row[:7] for row in chunk.actions]


class Teleoperator(ServerPolicy):
    chunk_length = 1
    needs_noise = False
    noise_shape = None
    noise_dtype = None
    in_worker = True
    done = False
    status = "approaching the bowl"

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        self.identity, self.start = identity, step


def test_every_policy_declares_the_replay_rule_flags() -> None:
    for p in (ServerPolicy(), Teleoperator()):
        assert isinstance(p, policy.Policy)
        check_policy_flags(p)
    broken = ServerPolicy()
    broken.noise_shape = None
    with pytest.raises(ValueError, match="noise_shape"):
        check_policy_flags(broken)
    no_flag = SimpleNamespace(needs_noise=False, noise_shape=None, noise_dtype=None, chunk_length=1)
    with pytest.raises(ValueError, match="row_independent"):
        check_policy_flags(no_flag)
    chunky = Teleoperator()
    chunky.chunk_length = 30
    with pytest.raises(ValueError, match="one step at a time"):
        check_policy_flags(chunky)


def test_an_in_worker_policy_reports_done_and_status() -> None:
    assert isinstance(Teleoperator(), policy.InWorkerPolicy)
    assert not isinstance(ServerPolicy(), policy.InWorkerPolicy)

    class Silent(ServerPolicy):  # in the worker, but says nothing about its progress
        chunk_length = 1
        needs_noise = False
        noise_shape = None
        noise_dtype = None
        in_worker = True

    with pytest.raises(ValueError, match="exposes done") as caught:
        check_policy_flags(Silent())
    assert "exposes status" in str(caught.value) and "begin_episode" in str(caught.value)
    teleop = Teleoperator()
    teleop.begin_episode(identity(), 40)  # a continuation from a snapshot at step 40
    assert teleop.start == 40


def test_server_policies_receive_the_group_s_effective_spec() -> None:
    effective = SPEC.model_copy(update={"sampling_steps": 5})
    chunks = ServerPolicy().act([], [], [], effective)
    assert chunks == []


def test_the_environment_exposes_the_chain_s_hashes() -> None:
    for name in ("state_hash", "fixture_hash"):
        assert callable(getattr(env.SimEnv, name))
    fixture_doc = " ".join((env.SimEnv.fixture_hash.__doc__ or "").split())
    for covered in (
        "body poses",
        "camera poses",
        "friction",
        "masses",
        "lights",
        "actuator",
        "texture",
    ):
        assert covered in fixture_doc
    assert list(env.StepResult._fields) == ["observation", "done", "verdict", "native_success"]
    assert hash_digest(b"x").hex() == hash_bytes(b"x") and len(hash_digest(b"x")) == 8


# Live layer.


def test_holding_still_depends_on_the_control_mode() -> None:
    last = np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.0, 1.0], np.float32)  # gripper closed
    assert np.array_equal(hold_action(ControlMode.absolute, last), last)
    relative = hold_action(ControlMode.relative, last)
    assert np.array_equal(relative, np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32))
    assert relative[-1] != libero_env.NO_OP_ACTION[-1]  # the no-op would open the gripper


def test_stage_five_decisions_and_the_trace() -> None:
    assert [d.value for d in Decision] == [
        "continue",
        "retry",
        "stop",
        "hand_over",
        "recover",
        "degrade",
    ]
    assert live.STAGE5_DECISIONS == ("recover", "degrade")
    record = DecisionRecord(
        step=40, decision=Decision.hand_over, hold_steps=12, decision_ms=9500.0, model_ms=4100.0
    )
    assert record.decision is Decision.hand_over


def test_telemetry_is_what_the_visible_steps_table_holds() -> None:
    fields = {f.name for f in dataclasses.fields(StepTelemetry)}
    assert fields <= set(allowed_columns("steps"))


# Rollout.


def test_jobs_fit_their_mode() -> None:
    ep = identity()
    run = Job(identity=ep)
    assert run.key == "e000001" and run.effective_faults == [FAULT]
    replay = Job(identity=ep, mode=JobMode.replay, remove=["f01"])
    assert replay.effective_faults == [] and replay.key != run.key
    assert replay.key != Job(identity=ep, mode=JobMode.replay).key
    snapshot = SnapshotRef(episode_id="e000001", step=40, path="truth/snapshots/e000001/40.bin")
    Job(identity=ep, mode=JobMode.continue_, from_snapshot=snapshot)
    # A run job names where it saves snapshots; no other mode does.
    assert Job(identity=ep, snapshot_steps=[40, 120]).key == "e000001"
    for steps, mode in (([40], JobMode.replay), ([40, 40], JobMode.run), ([280], JobMode.run)):
        with pytest.raises(ValidationError):
            Job(
                identity=ep,
                mode=mode,
                remove=["f01"] if mode is JobMode.replay else [],
                snapshot_steps=steps,
            )
    trace = [
        DecisionRecord(
            step=40, decision=Decision.retry, hold_steps=10, decision_ms=1.0, model_ms=1.0
        )
    ]
    Job(identity=ep, mode=JobMode.scripted, decision_trace=trace)
    # A continuation has its own step budget; the key changes only when one is set.
    long = Job(identity=ep, mode=JobMode.continue_, from_snapshot=snapshot, max_steps=650)
    plain = Job(identity=ep, mode=JobMode.continue_, from_snapshot=snapshot)
    assert long.step_budget == 650 and plain.step_budget == ep.max_steps
    assert long.key != plain.key
    differs = plain.model_dump(
        mode="json", include={"remove", "add", "from_snapshot", "decision_trace"}
    )
    assert plain.key == f"e000001.continue.{hash_json(differs)}"
    Job(identity=ep, mode=JobMode.scripted, decision_trace=trace, max_steps=650)
    for bad in (
        {"mode": JobMode.run, "max_steps": 650},
        {"mode": JobMode.replay, "remove": ["f01"], "max_steps": 650},
        {"mode": JobMode.continue_, "from_snapshot": snapshot, "max_steps": 0},
        {"mode": JobMode.run, "remove": ["f01"]},
        {"mode": JobMode.replay, "remove": ["f07"]},
        {"mode": JobMode.continue_},
        {"mode": JobMode.scripted},
        {"mode": JobMode.replay, "decision_trace": trace},
        {"mode": JobMode.replay, "add": [FAULT]},
    ):
        with pytest.raises(ValidationError):
            Job(identity=ep, **bad)


def test_the_snapshot_holds_rollout_state_and_no_controller_goal() -> None:
    names = {f.name for f in dataclasses.fields(rollout.Snapshot)}
    assert names == {
        "env",
        "queued_actions",
        "chunk_index",
        "step",
        "fault_states",
        "success_check",
        "last_commanded",
        "remaining_active_steps",
        "retry_used",
        "degraded",
    }


def test_the_chain_sample_is_every_tenth_and_at_least_twenty() -> None:
    ids = [f"e{n:06d}" for n in range(1000)]
    assert len(chain_sample(ids)) == 100
    assert len(chain_sample(ids[:100])) == 20
    assert chain_sample(ids[:15]) == ids[:15]


def test_the_run_layout_names_every_file(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path)
    assert layout.table("episodes") == tmp_path / "log/episodes.parquet"
    assert layout.table("steps", "e000001") == tmp_path / "log/steps/e000001.parquet"
    assert layout.video("e000001", env.Camera.wrist) == tmp_path / "log/video/e000001.wrist.mp4"
    with pytest.raises(ValueError):
        layout.table("steps")
    job = Job(identity=identity())
    replay = Job(identity=identity(), mode=JobMode.replay, remove=["f01"])
    assert layout.chain(job) == tmp_path / "truth/chain/e000001.npz"
    assert layout.chain(replay) == tmp_path / f"truth/chain/{replay.key}.npz"  # its own key
    assert layout.table("chain", job.key) == layout.chain(job)
    # A continuation spawned from a snapshot is persisted next to it, and found again.
    snap = layout.snapshot_file("e000001", job.key, 40)
    assert snap == tmp_path / "truth/snapshots/e000001/e000001.40.pkl"
    path = layout.continuation_file("e000001", job.key, 40)
    assert path == tmp_path / "truth/snapshots/e000001/e000001.40.continue.json"
    ref = SnapshotRef(episode_id="e000001", step=40, path=str(snap.relative_to(tmp_path)))
    spawned = Job(identity=identity(), mode=JobMode.continue_, from_snapshot=ref, max_steps=600)
    path.parent.mkdir(parents=True)
    path.write_text(spawned.model_dump_json())
    assert layout.continuation_files() == [path]
    assert layout.continuations() == [spawned]
    assert not layout.is_done(job)
    layout.chain(job).parent.mkdir(parents=True)
    layout.chain(job).write_bytes(b"")
    assert not layout.is_done(job)  # the steps file is missing
    layout.table("steps", "e000001").parent.mkdir(parents=True)
    layout.table("steps", "e000001").write_bytes(b"")
    assert layout.is_done(job)


# Worlds.


def world_spec(**changes: Any) -> world.WorldSpec:
    drift = FaultSpec(
        fault_id="f02",
        kind="gripper_wear",
        params={},
        level=0.2,
        target=Target(scope="r03", component=Component.gripper),
        schedule=Schedule(onset_day=0, drift=Drift(shape="linear", rate=0.01)),
    )
    fields: dict[str, Any] = {
        "world_id": "w01",
        "seed": 11,
        "size": "small",
        "sites": 3,
        "robots_per_site": 4,
        "days": 30,
        "windows": world.Windows(
            observed=world.DayRange(first=0, last=13),
            experiment=world.DayRange(first=14, last=15),
            scoring=world.DayRange(first=16, last=29),
        ),
        "fault_schedule": [FAULT, drift],
        "patterns": world.Patterns(drifts=["f02"], decoys=["f01"]),
        "policy": SPEC,
        "tasks": [("libero_spatial", t) for t in range(10)],
        "episodes_per_robot_day": 20,
    }
    fields.update(changes)
    return world.WorldSpec(**fields)


def test_a_world_spec_checks_its_windows_and_patterns() -> None:
    spec = world_spec()
    assert spec.patterns.excluded_from_replay() == {"f01"}
    assert world.WorldSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError, match="follow each other"):
        world.Windows(
            observed=world.DayRange(first=0, last=14),
            experiment=world.DayRange(first=14, last=15),
            scoring=world.DayRange(first=16, last=29),
        )
    with pytest.raises(ValidationError, match="not in the schedule"):
        world_spec(patterns=world.Patterns(decoys=["f09"]))
    with pytest.raises(ValidationError, match="no drift"):
        world_spec(patterns=world.Patterns(drifts=["f01"]))
    with pytest.raises(ValidationError, match="ends after"):
        world_spec(days=20)


def test_world_files_are_sealed_by_their_hash(tmp_path: Path) -> None:
    for name in world.WORLD_FILES:
        (tmp_path / name).write_text(f"{name} contents\n")
    first = world.seal_world(tmp_path)
    assert len(first) == 64 and world.verify_world(tmp_path)
    assert world.hash_world_files(tmp_path) == first
    (tmp_path / "manifest.jsonl").write_text("manifest.jsonl contents, changed\n")
    assert not world.verify_world(tmp_path)
    (tmp_path / "model.json").unlink()
    with pytest.raises(FileNotFoundError):
        world.hash_world_files(tmp_path)


def test_diagnosis_directions_are_one_list() -> None:
    assert [d.value for d in Direction] == ["degraded", "misconfigured", "changed", "drifting"]
