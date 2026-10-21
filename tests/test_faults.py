"""The fault types of fleetlab.faults on a toy handle (docs/design/faults.md, contracts section 7).

Hook order and per-episode state, exact undo of every cell hook, truth
mappings that fit the vocabulary and the experimental design, change-log
entries in the fleet's own words, the generic change kinds reproducing the
faults they stand for, and what the observation hooks do to an image.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pytest
from fault_toys import (
    changed_arrays,
    identity,
    model_arrays,
    observation,
    spec_for,
    toy_handle,
)

import fleetlab.faults as faults
from fleetlab.contracts.actions import ADDED_CHANGE_KINDS, Action, resolve
from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.faults import (
    HOOK_ORDER,
    STEP_ORDER,
    Direction,
    FaultSpec,
    FaultState,
    FaultTypeBase,
    Schedule,
    Target,
    check_data_hook,
)
from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.model import ChangeAuthor
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.vocabulary import (
    VERBS,
    Component,
    ConditionKey,
    ConfigKey,
    FaultClass,
    TargetKind,
    Verb,
)
from fleetlab.faults._common import DosePoint, matrix_to_rotvec, rotvec_to_matrix
from fleetlab.faults.records import change_entry, condition_value, health_shift
from fleetlab.faults.success_detector_error import FALSE_REASON, MISSED_REASON

KINDS = faults.load_all()
WAITING = SuccessVerdict(False, False, "waiting")
TYPED = [k for k in KINDS if k not in ADDED_CHANGE_KINDS]
HOOKS = ("cell_hook", "observation_hook", "action_hook", "policy_input_hook", "data_hook")


def points(kind: str) -> tuple[DosePoint, ...]:
    calibration: tuple[DosePoint, ...] = getattr(faults.get(kind), "calibration", ())
    return calibration


ALL_POINTS = [(kind, p) for kind in TYPED for p in points(kind)]
POINT_IDS = [f"{kind}:{p.label}" for kind, p in ALL_POINTS]


def overridden(kind: str) -> set[str]:
    cls = type(faults.get(kind))
    return {h for h in HOOKS if getattr(cls, h) is not getattr(FaultTypeBase, h)}


# --------------------------------------------------------------------------- registry


def test_the_catalogue_is_registered() -> None:
    assert set(KINDS) == {
        "camera_pose_offset",
        "camera_degradation",
        "action_latency",
        "policy_knob",
        "control_mode_change",
        "software_release",
        "gripper_wear",
        "object_batch_variant",
        "furniture_shift",
        "illumination_change",
        "backdrop_change",
        "success_detector_error",
        "config_change",
        "condition_change",
        # stage 3
        "lens_contamination",
        "camera_latency",
        "gripper_force_loss",
        "gripper_sticky_release",
        "arm_torque_loss",
        "controller_overload",
        "calibration_update",
        "image_pipeline_release",
        "table_height_change",
        "light_color_shift",
        "success_false_positive",
        # the loop's lasting condition (docs/design/loop.md, section 7)
        "remote_link_latency",
    }
    for kind in TYPED:
        fault_type = faults.get(kind)
        assert fault_type.dose_levels[0] == 0.0, kind
        assert list(fault_type.dose_levels) == sorted(fault_type.dose_levels), kind
        assert points(kind), f"{kind} has no calibration points"


# Each type with its class, target and fixing verbs, from the experimental design,
# adapted where the contracts' rules say otherwise (docs/design/faults.md).
EXPECTED = {
    "camera_pose_offset": (FaultClass.sensor, TargetKind.component, {"recalibrate", "replace"}),
    "camera_degradation": (FaultClass.sensor, TargetKind.component, {"clean", "replace"}),
    "action_latency": (FaultClass.configuration, TargetKind.config_key, {"set", "revert"}),
    "policy_knob": (FaultClass.configuration, TargetKind.config_key, {"set", "revert"}),
    "control_mode_change": (FaultClass.configuration, TargetKind.config_key, {"set", "revert"}),
    "software_release": (FaultClass.software, TargetKind.config_key, {"set", "revert"}),
    "gripper_wear": (FaultClass.physical, TargetKind.component, {"replace", "service"}),
    "object_batch_variant": (
        FaultClass.physical,
        TargetKind.condition_key,
        {"change_condition"},
    ),
    "furniture_shift": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "illumination_change": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "backdrop_change": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "success_detector_error": (FaultClass.data, TargetKind.component, {"service", "replace"}),
    "config_change": (FaultClass.configuration, TargetKind.config_key, {"set", "revert"}),
    "condition_change": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "lens_contamination": (FaultClass.sensor, TargetKind.component, {"clean", "replace"}),
    "camera_latency": (FaultClass.sensor, TargetKind.component, {"service", "replace"}),
    "gripper_force_loss": (FaultClass.physical, TargetKind.component, {"replace", "service"}),
    "gripper_sticky_release": (
        FaultClass.physical,
        TargetKind.component,
        {"clean", "service", "replace"},
    ),
    "arm_torque_loss": (FaultClass.physical, TargetKind.component, {"service", "replace"}),
    "controller_overload": (FaultClass.physical, TargetKind.component, {"service", "replace"}),
    "calibration_update": (FaultClass.configuration, TargetKind.config_key, {"set", "revert"}),
    "image_pipeline_release": (FaultClass.software, TargetKind.config_key, {"set", "revert"}),
    "table_height_change": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "light_color_shift": (FaultClass.site, TargetKind.condition_key, {"change_condition"}),
    "success_false_positive": (FaultClass.data, TargetKind.component, {"service", "replace"}),
    # A lasting condition: no verb removes it.
    "remote_link_latency": (FaultClass.site, TargetKind.component, set()),
}


@pytest.mark.parametrize("kind", KINDS)
def test_truth_mappings_fit_the_vocabulary_and_the_design(kind: str) -> None:
    fault_type = faults.get(kind)
    truth = fault_type.truth
    fault_class, target_kind, verbs = EXPECTED[kind]
    assert fault_type.fault_class is fault_class
    assert truth.target_kind is target_kind
    assert {v.value for v in truth.fixing_verbs} == verbs
    assert all(VERBS[v].target_kind is target_kind for v in truth.fixing_verbs)
    assert truth.writes_change == (fault_class in {FaultClass.configuration, FaultClass.software})
    if target_kind is TargetKind.component:
        assert truth.scopes == {ScopeKind.robot}
    if target_kind is TargetKind.condition_key:
        assert truth.scopes == {ScopeKind.site}
    accepted = {
        FaultClass.physical: Direction.degraded,
        FaultClass.sensor: Direction.degraded,
        FaultClass.configuration: Direction.misconfigured,
        FaultClass.software: Direction.changed,
        FaultClass.site: Direction.changed,
        FaultClass.data: Direction.degraded,
    }
    assert accepted[fault_class] in truth.directions


@pytest.mark.parametrize(("kind", "point"), ALL_POINTS, ids=POINT_IDS)
def test_every_calibration_point_is_a_valid_fault_its_verbs_remove(
    kind: str, point: DosePoint
) -> None:
    spec = spec_for(kind, point)
    faults.REGISTRY.check(spec)
    episode = identity([spec])
    truth = faults.truth_table()
    for verb in faults.get(kind).truth.fixing_verbs:
        value: Any = None
        if verb is Verb.set or verb is Verb.change_condition:
            value = spec.nominal
        target = spec.target
        if verb is Verb.change_condition:
            target = Target(scope="s1", condition_key=spec.target.condition_key)
        result = resolve(Action(verb=verb, target=target, value=value), episode, [spec], truth)
        assert result.removed == {spec.fault_id}, verb
        assert not result.added


@pytest.mark.parametrize("kind", TYPED)
def test_a_type_implements_only_the_hooks_it_needs(kind: str) -> None:
    expected = {
        "camera_pose_offset": {"cell_hook"},
        "camera_degradation": {"observation_hook"},
        "action_latency": {"action_hook"},
        "policy_knob": {"policy_input_hook"},
        "control_mode_change": {"policy_input_hook"},
        "software_release": set(HOOKS),  # forwards to what it carries
        "gripper_wear": {"cell_hook"},
        "object_batch_variant": {"cell_hook"},
        "furniture_shift": {"cell_hook", "observation_hook", "action_hook"},
        "illumination_change": {"cell_hook"},
        "backdrop_change": {"cell_hook"},
        "success_detector_error": {"data_hook"},
        "lens_contamination": {"observation_hook"},
        "camera_latency": {"observation_hook"},
        "gripper_force_loss": {"cell_hook"},
        "gripper_sticky_release": {"action_hook"},
        "arm_torque_loss": {"cell_hook"},
        "controller_overload": {"action_hook"},
        "calibration_update": set(),  # inert: no mechanism at all
        "image_pipeline_release": {"observation_hook"},
        "table_height_change": {"cell_hook", "observation_hook", "action_hook"},
        "light_color_shift": {"cell_hook"},
        "success_false_positive": {"observation_hook", "data_hook"},  # it only listens
        "remote_link_latency": {"action_hook"},
    }
    assert overridden(kind) == expected[kind]


# --------------------------------------------------------------------------- an episode


class Recorder:
    """Runs one toy episode through the hooks in the fixed order and records what happens."""

    def __init__(self, specs: list[FaultSpec], control_mode: ControlMode = ControlMode.absolute):
        self.identity = identity(specs, control_mode=control_mode)
        self.handle, self.uploads = toy_handle()
        self.types = [faults.get(s.kind) for s in specs]
        self.states: list[FaultState] = []

    def start(self) -> Any:
        # new_episode first, then the cell hooks, which work on the states it returned.
        self.states = [
            t.new_episode(s, self.identity)
            for t, s in zip(self.types, self.identity.faults, strict=True)
        ]
        undos = [
            t.cell_hook(self.handle, st) for t, st in zip(self.types, self.states, strict=True)
        ]
        policy = self.identity.policy
        for t, st in zip(self.types, self.states, strict=True):
            policy = t.policy_input_hook(policy, st)
        self.policy = policy
        return [u for u in undos if u is not None], policy

    def step(self, obs: Any, command: Any, verdict: SuccessVerdict) -> tuple[Any, Any, Any]:
        for t, st in zip(self.types, self.states, strict=True):
            obs = t.observation_hook(obs, st)
        executed = command
        from fleetlab.faults.command import action_context

        with action_context(self.policy.control_mode):
            for t, st in zip(self.types, self.states, strict=True):
                executed = t.action_hook(executed, st)
        for t, st in zip(self.types, self.states, strict=True):
            before, verdict = verdict, t.data_hook(verdict, st)
            check_data_hook(before, verdict)
        return obs, executed, verdict


def test_the_rollout_order_is_the_contracts_one() -> None:
    assert STEP_ORDER == ("observation_hook", "policy", "action_hook", "simulator", "data_hook")
    assert [p.name for p in HOOK_ORDER][:3] == ["new_episode", "cell_hook", "policy_input_hook"]


def delay_spec(steps: float) -> FaultSpec:
    return FaultSpec(
        fault_id="f01",
        kind="action_latency",
        level=steps,
        target=Target(scope="r01", config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=0),
        nominal=0,
    )


def test_the_delay_line_is_episode_state_and_replays_from_a_snapshot() -> None:
    commands = [np.full(7, float(i), dtype=np.float32) for i in range(12)]
    run = Recorder([delay_spec(3)])
    run.start()
    executed = [run.step(observation(i), c, WAITING)[1] for i, c in enumerate(commands)]
    assert [float(a[0]) for a in executed] == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    # A new episode starts with an empty line, whatever the last one left.
    run.start()
    assert float(run.step(observation(), commands[5], WAITING)[1][0]) == 5
    # Snapshot after 6 steps, then both branches continue alike.
    run.start()
    fault_type = run.types[0]
    for i in range(6):
        run.step(observation(i), commands[i], WAITING)
    blob = fault_type.snapshot(run.states[0])
    rest = [run.step(observation(i), commands[i], WAITING)[1] for i in range(6, 12)]
    run.states = [fault_type.restore(blob)]
    again = [run.step(observation(i), commands[i], WAITING)[1] for i in range(6, 12)]
    assert [a.tobytes() for a in rest] == [a.tobytes() for a in again]


def degradation(mode: str, level: float, **params: Any) -> FaultSpec:
    return FaultSpec(
        fault_id="f02",
        kind="camera_degradation",
        params={"mode": mode, **params},
        level=level,
        target=Target(scope="r01", component=Component.scene_camera),
        schedule=Schedule(onset_day=0),
    )


def test_sensor_noise_is_drawn_from_the_episode_and_replays() -> None:
    spec = degradation("noise", 10.0)
    fault_type = faults.get("camera_degradation")

    def images(index: int, steps: int = 3) -> list[bytes]:
        state = fault_type.new_episode(spec, identity([spec], index=index))
        return [
            fault_type.observation_hook(observation(i), state)["pixels"]["image"].tobytes()
            for i in range(steps)
        ]

    assert images(0) == images(0)
    assert images(0) != images(1)
    assert len(set(images(0))) == 3  # fresh noise every step
    state = fault_type.new_episode(spec, identity([spec]))
    fault_type.observation_hook(observation(0), state)
    blob = fault_type.snapshot(state)
    after = fault_type.observation_hook(observation(1), state)["pixels"]["image"]
    restored = fault_type.restore(blob)
    assert (
        fault_type.observation_hook(observation(1), restored)["pixels"]["image"].tobytes()
        == after.tobytes()
    )


def test_observation_hooks_degrade_the_image_as_intended() -> None:
    fault_type = faults.get("camera_degradation")
    obs = observation(size=360)
    original = obs["pixels"]["image"].copy()

    def apply(spec: FaultSpec) -> np.ndarray[Any, Any]:
        state = fault_type.new_episode(spec, identity([spec]))
        out = fault_type.observation_hook(obs, state)
        assert np.array_equal(obs["pixels"]["image"], original)  # never in place
        assert out["pixels"]["image2"] is obs["pixels"]["image2"]  # the other camera untouched
        image: np.ndarray[Any, Any] = out["pixels"]["image"]
        assert image.dtype == np.uint8 and image.shape == original.shape
        return image

    def sharpness(image: np.ndarray[Any, Any]) -> float:
        g = image.astype(np.float64).mean(axis=-1)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(lap.var())

    blurred = [sharpness(apply(degradation("blur", s))) for s in (1.0, 2.0, 4.0)]
    assert sharpness(original) > blurred[0] > blurred[1] > blurred[2]
    assert (
        abs(apply(degradation("blur", 2.0)).mean() - original.mean()) < 1.0
    )  # blur keeps the mean

    noisy = apply(degradation("noise", 10.0)).astype(np.float64) - original
    assert 9.0 < noisy.std() < 11.0 and abs(noisy.mean()) < 0.5

    darker = apply(degradation("exposure", 1.0, darker=True)).astype(np.float64)
    assert abs(darker.mean() / original.mean() - 0.5) < 0.01
    brighter = apply(degradation("exposure", 1.0)).astype(np.float64)
    assert brighter.max() == 255 and brighter.mean() > 1.5 * original.mean()


def test_the_detector_gets_a_share_of_outcomes_wrong_nested_by_level() -> None:
    fault_type = faults.get("success_detector_error")
    placed = SuccessVerdict(True, True, "placed")
    gave_up = SuccessVerdict(False, True, "LIBERO only")

    def wrong(level: float, mode: str = "missed") -> set[int]:
        spec = FaultSpec(
            fault_id="f03",
            kind="success_detector_error",
            params={"mode": mode},
            level=level,
            target=Target(scope="r01", component=Component.controller),
            schedule=Schedule(onset_day=0),
        )
        out = set()
        for index in range(2000):
            state = fault_type.new_episode(spec, identity([spec], index=index))
            for verdict in (placed, gave_up, WAITING):
                after = fault_type.data_hook(verdict, state)
                check_data_hook(verdict, after)  # success changes on a done verdict only
            if not fault_type.data_hook(placed, state).success:
                out.add(index)
            assert fault_type.data_hook(WAITING, state) is WAITING
        return out

    low, high = wrong(0.1), wrong(0.3)
    assert 150 < len(low) < 250 and 520 < len(high) < 680
    assert low <= high
    assert wrong(0.3, "false_success") == set()  # successes stay successes
    false_spec = FaultSpec(
        fault_id="f03",
        kind="success_detector_error",
        params={"mode": "false_success"},
        level=0.3,
        target=Target(scope="r01", component=Component.controller),
        schedule=Schedule(onset_day=0),
    )
    flips = sum(
        fault_type.data_hook(
            gave_up, fault_type.new_episode(false_spec, identity([false_spec], index=i))
        ).success
        for i in range(2000)
    )
    assert 520 < flips < 680
    spec = FaultSpec(
        fault_id="f03",
        kind="success_detector_error",
        params={"mode": "both"},
        level=1.0,
        target=Target(scope="r01", component=Component.controller),
        schedule=Schedule(onset_day=0),
    )
    state = fault_type.new_episode(spec, identity([spec]))
    assert fault_type.data_hook(placed, state) == SuccessVerdict(False, True, MISSED_REASON)
    assert fault_type.data_hook(gave_up, state) == SuccessVerdict(True, True, FALSE_REASON)


def test_a_policy_knob_changes_the_effective_spec_only() -> None:
    point = points("policy_knob")[0]
    spec = spec_for("policy_knob", point)
    run = Recorder([spec])
    _, policy = run.start()
    assert policy.sampling_steps == 5
    assert policy.model_dump(exclude={"sampling_steps"}) == run.identity.policy.model_dump(
        exclude={"sampling_steps"}
    )
    bad = spec.model_copy(update={"params": {"value": "five"}})
    with pytest.raises(ValueError, match="takes a int"):
        faults.get("policy_knob").new_episode(bad, identity([spec]))


def test_a_switched_controller_is_the_effective_specs_control_mode() -> None:
    spec = spec_for("control_mode_change", points("control_mode_change")[0])
    run = Recorder([spec])
    _, policy = run.start()
    assert policy.control_mode is ControlMode.relative
    nominal = run.identity.policy
    assert policy.model_dump(exclude={"control_mode"}) == nominal.model_dump(
        exclude={"control_mode"}
    )
    # Commands pass unchanged: the switched controller reads them, not a hook.
    command = np.array([0.5, -2.0, 1.1, 0.2, 0.0, 0.0, 1.0], dtype=np.float32)
    assert run.step(observation(2), command, WAITING)[1] is command
    with pytest.raises(ValueError):
        faults.get("control_mode_change").new_episode(
            spec.model_copy(update={"params": {"value": "sideways"}}), run.identity
        )


def test_rotation_vectors_round_trip() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        v = rng.normal(size=3)
        v *= rng.uniform(0, np.pi - 1e-3) / np.linalg.norm(v)
        assert np.allclose(matrix_to_rotvec(rotvec_to_matrix(v)), v, atol=1e-8)
    assert np.allclose(rotvec_to_matrix(np.array([0.0, 0.0, np.pi])), np.diag([-1.0, -1.0, 1.0]))
    half = matrix_to_rotvec(np.diag([-1.0, -1.0, 1.0]))
    assert np.allclose(np.abs(half), [0, 0, np.pi])


def test_a_shifted_table_is_seen_and_reached_in_the_robots_own_frame() -> None:
    point = points("furniture_shift")[1]  # 2 cm to the robot's left
    spec = spec_for("furniture_shift", point)
    run = Recorder([spec])
    undos, _ = run.start()
    model = run.handle.model
    base = int(np.flatnonzero([model.body(i).name == "robot0_base" for i in range(model.nbody)])[0])
    assert np.allclose(model.body_pos[base], [-0.66, -0.02, 0.912])
    obs = observation(1)
    command = np.array([0.1, 0.2, 1.0, 0, 0, 0, -1], dtype=np.float32)
    hooked, executed, _ = run.step(obs, command, WAITING)
    assert np.allclose(
        hooked["robot_state"]["eef"]["pos"] - obs["robot_state"]["eef"]["pos"], [0, 0.02, 0]
    )
    assert np.allclose(executed - command, [0, -0.02, 0, 0, 0, 0, 0], atol=1e-7)
    for undo in reversed(undos):
        undo()
    relative = Recorder([spec], control_mode=ControlMode.relative)
    relative.start()
    assert relative.step(obs, command, WAITING)[1] is command


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
@pytest.mark.parametrize("mode_first", [False, True])
@pytest.mark.parametrize("mode", [ControlMode.relative, ControlMode.absolute])
def test_placement_commands_follow_effective_mode_and_command_frame(
    kind: str, mode_first: bool, mode: ControlMode
) -> None:
    from fleetlab.faults.command import action_context

    shift = spec_for("furniture_shift", points("furniture_shift")[1])
    if kind == "table_height_change":
        shift = spec_for(kind, points(kind)[1])
    elif kind == "condition_change":
        shift = shift.model_copy(
            update={"kind": kind, "level": 1.0, "params": {"value": [0, 2, -2]}}
        )
    switch = spec_for("control_mode_change", points("control_mode_change")[0]).model_copy(
        update={"fault_id": "f02", "params": {"value": mode.value}}
    )
    run = Recorder(
        [switch, shift] if mode_first else [shift, switch],
        control_mode=ControlMode.absolute if mode is ControlMode.relative else ControlMode.relative,
    )
    run.start()
    assert run.policy.control_mode is mode
    command = np.array([0.1, 0.2, 1.0, 0, 0, 0, -1], np.float32)
    original = command.copy()
    executed = run.step(observation(1), command, WAITING)[1]
    if mode is ControlMode.relative:
        assert executed is command
    else:
        offset = np.asarray(condition_value(shift)) / 100
        if len(offset) == 2:
            offset = np.append(offset, 0)
        expected = command.copy()
        expected[:3] -= offset.astype(np.float32)
        assert np.array_equal(executed, expected)
    # Snapshot-restored states cannot retain the nominal mode or a producer's frame.
    for ft, state in zip(run.types, run.states, strict=True):
        restored = ft.restore(ft.snapshot(state))
        with action_context(mode, world_frame=True):
            assert ft.action_hook(command, restored) is command
    assert np.array_equal(command, original)


# --------------------------------------------------------------------------- cell hooks


# The release forwards every hook to what it carries; its own test is below.
CELL_POINTS = [
    (k, p) for k, p in ALL_POINTS if "cell_hook" in overridden(k) and k != "software_release"
]


@pytest.mark.parametrize(
    ("kind", "point"), CELL_POINTS, ids=[f"{k}:{p.label}" for k, p in CELL_POINTS]
)
def test_every_cell_hook_changes_the_model_and_undoes_exactly(kind: str, point: DosePoint) -> None:
    spec = spec_for(kind, point)
    run = Recorder([spec])
    before = model_arrays(run.handle.model)
    undos, _ = run.start()
    assert undos, "a cell hook at a non-zero level changes the model"
    changed = changed_arrays(before, run.handle.model)
    assert changed
    uploads = list(run.uploads.calls)
    for undo in reversed(undos):
        undo()
    assert changed_arrays(before, run.handle.model) == []
    # A texture is uploaded when it changes and again when it is put back.
    if "tex_data" in changed:
        assert uploads and run.uploads.calls == uploads + uploads
    else:
        assert not run.uploads.calls


def test_cell_hooks_change_what_they_should() -> None:
    def changed(kind: str, index: int = 0) -> list[str]:
        run = Recorder([spec_for(kind, points(kind)[index])])
        before = model_arrays(run.handle.model)
        run.start()
        return changed_arrays(before, run.handle.model)

    assert changed("camera_pose_offset") == ["cam_quat"]
    assert changed("gripper_wear") == ["actuator_ctrlrange", "geom_friction"]
    assert changed("object_batch_variant") == [
        "body_inertia",
        "body_invweight0",
        "body_mass",
        "body_subtreemass",
        "dof_M0",
        "dof_invweight0",
    ]
    assert changed("object_batch_variant", 3) == ["geom_friction", "geom_priority"]
    assert changed("furniture_shift") == ["body_pos", "cam_pos"]
    assert changed("illumination_change") == [
        "light_diffuse",
        "light_specular",
        "vis.headlight.ambient",
        "vis.headlight.diffuse",
        "vis.headlight.specular",
    ]  # the toy's lights have no ambient term to scale
    assert changed("backdrop_change") == ["tex_data"]
    assert changed("gripper_force_loss") == [
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_gainprm",
    ]
    assert changed("arm_torque_loss") == ["actuator_gainprm"]
    assert changed("table_height_change") == ["body_pos", "cam_pos"]
    assert changed("light_color_shift") == [
        "light_diffuse",
        "light_specular",
        "vis.headlight.ambient",
        "vis.headlight.diffuse",
        "vis.headlight.specular",
    ]


def test_gripper_wear_leaves_a_gap_and_takes_grip() -> None:
    import mujoco

    spec = spec_for("gripper_wear", points("gripper_wear")[1])  # level 0.5
    run = Recorder([spec])
    model = run.handle.model
    run.start()
    one = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper0_right_gripper_finger_joint1"
    )
    two = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper0_right_gripper_finger_joint2"
    )
    assert np.allclose(model.actuator_ctrlrange[one], [0.00075, 0.04])
    assert np.allclose(model.actuator_ctrlrange[two], [-0.04, -0.00075])
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper0_right_finger1_pad_collision")
    assert np.isclose(model.geom_friction[pad][0], 2.0 * 0.75)
    shifts = [
        health_shift(spec_for("gripper_wear", p))["grip_force_loss"] for p in points("gripper_wear")
    ]
    assert shifts == sorted(shifts) and shifts[-1] > 0.5


def test_heavier_objects_scale_the_solver_constants_with_their_mass() -> None:
    import mujoco

    spec = spec_for("object_batch_variant", points("object_batch_variant")[0])  # 2 doublings
    run = Recorder([spec])
    model = run.handle.model
    reference = mujoco.MjModel.from_xml_string(model_xml_with_mass(model, factor=4.0))
    mujoco.mj_setConst(reference, mujoco.MjData(reference))
    run.start()
    for name in (
        "body_mass",
        "body_inertia",
        "body_invweight0",
        "body_subtreemass",
        "dof_invweight0",
        "dof_M0",
    ):
        assert np.allclose(getattr(model, name), getattr(reference, name), rtol=1e-9, atol=1e-12), (
            name
        )


def model_xml_with_mass(model: Any, factor: float) -> str:
    from fault_toys import TOY_XML

    return TOY_XML.replace('mass="0.0056"', f'mass="{0.0056 * factor!r}"').replace(
        'mass="0.0096"', f'mass="{0.0096 * factor!r}"'
    )


# --------------------------------------------------------------------------- records


@pytest.mark.parametrize(("kind", "point"), ALL_POINTS, ids=POINT_IDS)
def test_change_entries_come_from_configuration_faults_in_the_fleets_words(
    kind: str, point: DosePoint
) -> None:
    spec = spec_for(kind, point)
    entry = change_entry(spec, day=4, author=ChangeAuthor.vendor)
    fault_type = faults.get(kind)
    if fault_type.fault_class not in {FaultClass.configuration, FaultClass.software}:
        assert entry is None
        return
    assert entry is not None
    assert (entry.day, entry.scope, entry.key, entry.old) == (
        4,
        spec.target.scope,
        spec.target.config_key,
        spec.nominal,
    )
    assert entry.new != entry.old
    text = json.dumps(entry.model_dump(mode="json")).lower()
    for word in (*KINDS, "fault", "f01", "decoy", "injected"):
        assert word not in text, word
    if kind == "software_release":
        assert entry.key is ConfigKey.software_version and entry.new == "2.5.0"


def test_condition_values_are_in_the_conditions_units() -> None:
    def value(kind: str, index: int = 0) -> Any:
        return condition_value(spec_for(kind, points(kind)[index]))

    assert value("illumination_change") == -1.0
    assert value("illumination_change", 2) == 1.0
    assert (
        value("furniture_shift", 1)[1] == pytest.approx(2.0)
        and abs(value("furniture_shift", 1)[0]) < 1e-12
    )
    assert value("object_batch_variant") == [2.0, 0.0]
    assert value("object_batch_variant", 3) == [0.0, 2.0]
    assert value("backdrop_change") == 0.5
    camera = spec_for("camera_pose_offset", points("camera_pose_offset")[0])
    with pytest.raises(TypeError):
        condition_value(camera)
    assert health_shift(camera)["calibration_residual"] == pytest.approx(
        434.6 * np.tan(np.deg2rad(3)), rel=1e-3
    )
    assert health_shift(spec_for("policy_knob", points("policy_knob")[0])) == {}


# --------------------------------------------------------------------------- the generic kinds


def generic(spec: FaultSpec, value: Any) -> FaultSpec:
    """The change an action adds that puts the same value in force."""
    target = spec.target
    kind = "config_change" if target.kind is TargetKind.config_key else "condition_change"
    if kind == "condition_change":
        target = Target(scope="s1", condition_key=target.condition_key)
    return FaultSpec(
        fault_id="f02",
        kind=kind,
        params={"value": value},
        level=1.0,
        target=target,
        schedule=Schedule(onset_day=0),
        nominal=spec.nominal,
    )


def trace(spec: FaultSpec, steps: int = 25) -> tuple[list[str], list[bytes], list[bytes], str]:
    """Everything a fault does in a toy episode: model changes, observations, actions, policy."""
    run = Recorder([spec])
    before = model_arrays(run.handle.model)
    undos, policy = run.start()
    after = model_arrays(run.handle.model)
    model = [
        f"{k}:{after[k].tobytes().hex()}"
        for k in sorted(before)
        if before[k].tobytes() != after[k].tobytes()
    ]
    observations, actions = [], []
    for i in range(steps):
        command = np.array([0.1 * i, -0.2, 1.0, 0.1, 0.0, -0.1, 1.0], dtype=np.float32)
        obs, executed, _ = run.step(observation(i), command, WAITING)
        observations.append(pickle.dumps(obs["robot_state"]) + obs["pixels"]["image"].tobytes())
        actions.append(np.asarray(executed).tobytes())
    for undo in reversed(undos):
        undo()
    return model, observations, actions, policy.model_dump_json()


SET_TO_FAULT = [
    (k, p)
    for k, p in ALL_POINTS
    if k in {"action_latency", "policy_knob", "control_mode_change", "calibration_update"}
]


@pytest.mark.parametrize(
    ("kind", "point"), SET_TO_FAULT, ids=[f"{k}:{p.label}" for k, p in SET_TO_FAULT]
)
def test_a_set_to_a_faults_value_reproduces_the_fault(kind: str, point: DosePoint) -> None:
    spec = spec_for(kind, point)
    new: Any = faults.get(kind)
    assert trace(generic(spec, new.new_value(spec))) == trace(spec)


CHANGE_TO_FAULT = [
    (k, p) for k, p in ALL_POINTS if faults.get(k).truth.target_kind is TargetKind.condition_key
]


@pytest.mark.parametrize(
    ("kind", "point"), CHANGE_TO_FAULT, ids=[f"{k}:{p.label}" for k, p in CHANGE_TO_FAULT]
)
def test_a_condition_change_to_a_faults_value_reproduces_the_fault(
    kind: str, point: DosePoint
) -> None:
    spec = spec_for(kind, point)
    if spec.params and kind == "backdrop_change":
        pytest.skip("the condition's value carries the blend only; other surfaces are the fault's")
    if spec.params.get("prioritise"):
        pytest.skip("the control point has no condition value of its own")
    assert trace(generic(spec, condition_value(spec))) == trace(spec)


def test_the_generic_kinds_do_nothing_at_the_nominal_value_or_without_a_mechanism() -> None:
    configs: tuple[tuple[ConfigKey, Any], ...] = (
        (ConfigKey.control_action_delay_steps, 0),
        (ConfigKey.policy_sampling_steps, 10),
        (ConfigKey.camera_scene_calibration, [0.0, 0.0, 0.0]),
        (ConfigKey.software_version, "2.4.1"),
    )
    for key, nominal in configs:
        base = FaultSpec(
            fault_id="f01",
            kind="config_change",
            params={"value": nominal},
            level=1.0,
            target=Target(scope="r01", config_key=key),
            schedule=Schedule(onset_day=0),
            nominal=nominal,
        )
        healthy = trace(base)
        assert healthy[0] == [] and healthy[3] == identity().policy.model_dump_json()
        if key in (ConfigKey.camera_scene_calibration, ConfigKey.software_version):
            other = [1.0, 2.0, 3.0] if key is ConfigKey.camera_scene_calibration else "9.9.9"
            assert trace(base.model_copy(update={"params": {"value": other}})) == healthy
    conditions: tuple[tuple[ConditionKey, Any], ...] = (
        (ConditionKey.lighting, 0.0),
        (ConditionKey.placement, [0.0, 0.0]),
    )
    for condition, nominal in conditions:
        base = FaultSpec(
            fault_id="f01",
            kind="condition_change",
            params={"value": nominal},
            level=1.0,
            target=Target(scope="s1", condition_key=condition),
            schedule=Schedule(onset_day=0),
            nominal=nominal,
        )
        assert trace(base)[0] == []


def test_a_release_carrying_a_fault_is_that_fault() -> None:
    release = spec_for("software_release", points("software_release")[0])
    delay = delay_spec(17)
    assert trace(release, steps=40) == trace(delay, steps=40)
    with pytest.raises(ValueError, match="cannot carry"):
        faults.get("software_release").Params.model_validate(
            {
                "version": "2.5.0",
                "carries": [{"kind": "config_change", "level": 1, "target": "control.mode"}],
            }
        )
    with pytest.raises(ValueError):
        faults.get("software_release").Params.model_validate(
            {
                "version": "delay-17",
                "carries": [
                    {
                        "kind": "action_latency",
                        "level": 17,
                        "target": "control.action_delay_steps",
                        "nominal": 0,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "carried",
    [
        {"kind": "no_such_fault", "level": 1, "target": "control.mode", "nominal": "absolute"},
        {"kind": "action_latency", "level": 17, "target": "control.action_delay_steps"},
        {"kind": "action_latency", "level": 17, "target": "gripper", "nominal": 0},
        {"kind": "camera_degradation", "level": 2, "target": "scene_camera", "params": {"x": 1}},
    ],
)
def test_a_release_refuses_a_bad_carried_fault_when_it_is_made(carried: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        faults.get("software_release").Params.model_validate(
            {"version": "2.5.0", "carries": [carried]}
        )


def test_health_shifts_use_each_cameras_optics() -> None:
    scene = spec_for("camera_pose_offset", points("camera_pose_offset")[0])  # 3 deg
    wrist = scene.model_copy(
        update={"target": Target(scope="r01", component=Component.wrist_camera)}
    )
    ratio = (
        health_shift(scene)["calibration_residual"] / health_shift(wrist)["calibration_residual"]
    )
    assert ratio == pytest.approx(np.tan(np.deg2rad(37.5)) / np.tan(np.deg2rad(22.5)), rel=1e-9)


def test_fault_states_are_fresh_per_episode_and_snapshot_round_trips() -> None:
    for kind, point in ALL_POINTS:
        spec = spec_for(kind, point)
        fault_type = faults.get(kind)
        a = fault_type.new_episode(spec, identity([spec]))
        b = fault_type.new_episode(spec, identity([spec]))
        assert a is not b
        restored = fault_type.restore(fault_type.snapshot(a))
        assert type(restored) is type(a) and restored.spec == spec
