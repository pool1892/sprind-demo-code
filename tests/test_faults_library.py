"""The fault types added in stage 3 on the toy handle and synthetic observations.

Each kind's mechanism does what its docstring says, exactly and repeatably:
the dirty lens blurs and hazes most where the smear lies, the late camera
hands out old frames, the weak gripper and the weak arm scale their
actuators, the sticky gripper holds its fingers shut for the stated steps,
the overloaded controller falls further behind with every missed cycle, the
calibration record changes nothing, the image pipeline swaps, crops and
compresses, the lower table moves the robot up and keeps its frame, the
lamps change colour at about the same brightness, and the false-positive
detector flips only failed placements. The registry-wide checks (truth
mappings, exact undo of every cell hook, change entries, the generic kinds
reproducing every condition point) are in `test_faults.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fault_toys import identity, model_arrays, observation, spec_for, toy_handle

import fleetlab.faults as faults
from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.faults import FaultSpec, FaultState, Schedule, Target
from fleetlab.contracts.model import ChangeAuthor
from fleetlab.contracts.vocabulary import Component, ConfigKey
from fleetlab.faults._common import DosePoint
from fleetlab.faults.light_color_shift import COOL_TINT, WARM_TINT
from fleetlab.faults.records import change_entry, condition_value, health_shift

WAITING = SuccessVerdict(False, False, "waiting")


def points(kind: str) -> tuple[DosePoint, ...]:
    calibration: tuple[DosePoint, ...] = getattr(faults.get(kind), "calibration", ())
    return calibration


def spec(kind: str, index: int = 0, **update: Any) -> FaultSpec:
    out = spec_for(kind, points(kind)[index])
    return out.model_copy(update=update) if update else out


def start(fault: FaultSpec, index: int = 0) -> tuple[Any, FaultState]:
    fault_type = faults.get(fault.kind)
    in_force = [fault] if fault.is_fault else []  # an identity lists no fault at level zero
    return fault_type, fault_type.new_episode(fault, identity(in_force, index=index))


# --------------------------------------------------------------------------- lens_contamination


def sharpness(image: Any) -> float:
    g = np.asarray(image, dtype=np.float64).mean(axis=-1)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def test_a_dirty_lens_blurs_and_hazes_most_where_the_smear_lies() -> None:
    obs = observation(size=120)
    original = obs["pixels"]["image"].copy()

    def through(level: float) -> Any:
        fault_type, state = start(spec("lens_contamination", level=level))
        out = fault_type.observation_hook(obs, state)
        assert np.array_equal(obs["pixels"]["image"], original)  # never in place
        assert out["pixels"]["image2"] is obs["pixels"]["image2"]  # the wrist camera is clean
        return out["pixels"]["image"]

    images = [through(level) for level in (0.25, 0.5, 1.0)]
    values = [sharpness(original)] + [sharpness(image) for image in images]
    assert values == sorted(values, reverse=True)  # dirtier is softer
    centre, corner = (slice(45, 75), slice(45, 75)), (slice(0, 20), slice(0, 20))
    full = images[-1].astype(np.float64)
    loss_centre = sharpness(full[centre]) / sharpness(original[centre])
    loss_corner = sharpness(full[corner]) / sharpness(original[corner])
    assert loss_centre < 0.5 * loss_corner  # the smear is densest in the middle
    assert full.mean() > original.mean()  # scattered light washes the image toward a bright haze
    zero = faults.get("lens_contamination")
    state = zero.new_episode(spec("lens_contamination", level=0.0), identity())
    assert np.array_equal(zero.observation_hook(obs, state)["pixels"]["image"], original)


def test_a_dirty_lens_is_the_same_every_step() -> None:
    fault_type, state = start(spec("lens_contamination", 3))
    a = fault_type.observation_hook(observation(2), state)["pixels"]["image"]
    b = fault_type.observation_hook(observation(2), state)["pixels"]["image"]
    assert a.tobytes() == b.tobytes()
    with pytest.raises(ValueError):
        start(spec("lens_contamination", level=1.5))


def test_the_wrist_lens_point_dirties_the_wrist_image() -> None:
    fault_type, state = start(spec("lens_contamination", -1))  # wrist, level 1
    obs = observation(size=96)
    out = fault_type.observation_hook(obs, state)
    assert out["pixels"]["image"] is obs["pixels"]["image"]
    assert not np.array_equal(out["pixels"]["image2"], obs["pixels"]["image2"])


# --------------------------------------------------------------------------- camera_latency


def frame(step: int) -> Any:
    obs = observation(step, size=16)
    image = np.full((16, 16, 3), step, dtype=np.uint8)
    return {**obs, "pixels": {**obs["pixels"], "image": image}}


def test_a_late_camera_hands_out_frames_that_many_steps_old() -> None:
    fault_type, state = start(spec("camera_latency", 0))  # 6 steps, scene camera
    seen = [
        int(fault_type.observation_hook(frame(t), state)["pixels"]["image"][0, 0, 0])
        for t in range(11)
    ]
    assert seen == [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    obs = frame(9)
    out = fault_type.observation_hook(obs, state)
    assert out["pixels"]["image2"] is obs["pixels"]["image2"]  # the other camera is on time
    assert out["robot_state"] is obs["robot_state"]  # and so are the joints


def test_a_late_camera_replays_from_a_snapshot() -> None:
    fault_type, state = start(spec("camera_latency", 1))  # 12 steps
    for t in range(20):
        fault_type.observation_hook(frame(t), state)
    blob = fault_type.snapshot(state)
    rest = [fault_type.observation_hook(frame(t), state)["pixels"]["image"] for t in range(20, 30)]
    again = fault_type.restore(blob)
    redo = [fault_type.observation_hook(frame(t), again)["pixels"]["image"] for t in range(20, 30)]
    assert [a.tobytes() for a in rest] == [b.tobytes() for b in redo]
    with pytest.raises(ValueError, match="whole number"):
        start(spec("camera_latency", level=2.5))


# --------------------------------------------------------------------------- gripper_force_loss


def actuator(model: Any, name: str) -> int:
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def test_a_weak_gripper_has_softer_fingers_and_a_lower_force_limit() -> None:
    fault = spec("gripper_force_loss", 2)  # 0.75 lost
    handle, _ = toy_handle()
    model = handle.model
    before = model_arrays(model)
    fault_type, state = start(fault)
    undo = fault_type.cell_hook(handle, state)
    for name in ("gripper0_right_gripper_finger_joint1", "gripper0_right_gripper_finger_joint2"):
        a = actuator(model, name)
        assert model.actuator_gainprm[a][0] == pytest.approx(250.0)
        assert model.actuator_biasprm[a][1] == pytest.approx(-250.0)
        assert np.allclose(model.actuator_forcerange[a], [-5.0, 5.0])
    motor = actuator(model, "robot0_torq_j1")
    assert model.actuator_gainprm[motor][0] == 1.0  # the arm is untouched
    assert undo is not None
    undo()
    assert all(before[k].tobytes() == v.tobytes() for k, v in model_arrays(model).items())
    assert health_shift(fault) == {"grip_force_loss": 0.75}
    with pytest.raises(ValueError):
        start(spec("gripper_force_loss", level=1.0))


# --------------------------------------------------------------------------- gripper_sticky_release


def grip(value: float) -> Any:
    return np.array([0.1, 0.0, 1.0, 0.0, 0.0, 0.0, value], dtype=np.float32)


def test_a_sticky_gripper_opens_that_many_steps_late() -> None:
    fault_type, state = start(spec("gripper_sticky_release", 0))  # 5 steps
    commands = [-1.0] * 3 + [1.0] * 4 + [-1.0] * 8 + [1.0] + [-1.0] * 5
    executed = [float(fault_type.action_hook(grip(c), state)[6]) for c in commands]
    assert executed == [-1.0] * 3 + [1.0] * 4 + [1.0] * 5 + [-1.0] * 3 + [1.0] + [1.0] * 5
    command = grip(-1.0)
    out = fault_type.action_hook(command, state)  # the hold has run out: unchanged
    assert out is command
    held = fault_type.action_hook(grip(0.6), state)
    assert float(held[6]) == pytest.approx(0.6)
    late = fault_type.action_hook(grip(-1.0), state)
    assert float(late[6]) == pytest.approx(0.6)  # the last closing value is held
    assert np.array_equal(late[:6], grip(-1.0)[:6])  # the arm moves on as commanded


# --------------------------------------------------------------------------- arm_torque_loss


def test_a_weak_arm_scales_every_motor_and_nothing_else() -> None:
    fault = spec("arm_torque_loss", 1)  # 0.2 lost
    handle, _ = toy_handle()
    model = handle.model
    fault_type, state = start(fault)
    undo = fault_type.cell_hook(handle, state)
    assert undo is not None
    for i in range(1, 8):
        assert model.actuator_gainprm[actuator(model, f"robot0_torq_j{i}")][0] == pytest.approx(0.8)
    finger = actuator(model, "gripper0_right_gripper_finger_joint1")
    assert model.actuator_gainprm[finger][0] == 1000.0
    undo()
    assert model.actuator_gainprm[actuator(model, "robot0_torq_j4")][0] == 1.0
    assert health_shift(fault) == {}


# --------------------------------------------------------------------------- controller_overload


def overload(level: float) -> FaultSpec:
    return FaultSpec(
        fault_id="f04",
        kind="controller_overload",
        level=level,
        target=Target(scope="r01", component=Component.controller),
        schedule=Schedule(onset_day=0),
    )


def lags(level: float, steps: int = 200, index: int = 0) -> list[int]:
    """How many steps behind its command each executed command is."""
    fault_type, state = start(overload(level), index)
    out = []
    for t in range(steps):
        executed = fault_type.action_hook(np.full(7, float(t), dtype=np.float32), state)
        out.append(t - int(executed[0]))
    return out


def test_an_overloaded_controller_falls_further_behind_with_every_miss() -> None:
    assert lags(0.0) == [0] * 200
    lag = lags(0.1)
    assert lag == sorted(lag)  # it never catches up
    assert lag[0] == 0 and 10 <= lag[-1] <= 35  # about 0.1 per step
    assert lags(0.1) == lag  # the misses are the episode's own draws
    assert lags(0.1, index=1) != lag
    low = lags(0.05)
    assert all(a <= b for a, b in zip(low, lag, strict=True))  # nested by level


def test_an_overloaded_controller_replays_from_a_snapshot() -> None:
    fault_type, state = start(overload(0.2))
    for t in range(30):
        fault_type.action_hook(np.full(7, float(t), dtype=np.float32), state)
    blob = fault_type.snapshot(state)
    rest = [fault_type.action_hook(np.full(7, float(t), np.float32), state) for t in range(30, 60)]
    again = fault_type.restore(blob)
    redo = [fault_type.action_hook(np.full(7, float(t), np.float32), again) for t in range(30, 60)]
    assert [a.tobytes() for a in rest] == [b.tobytes() for b in redo]
    with pytest.raises(ValueError):
        start(overload(0.6))


# --------------------------------------------------------------------------- calibration_update


def test_a_rewritten_calibration_record_is_a_change_and_nothing_else() -> None:
    fault = spec("calibration_update")
    entry = change_entry(fault, day=3, author=ChangeAuthor.auto)
    assert entry is not None
    assert entry.key is ConfigKey.camera_scene_calibration
    assert entry.old == [0.66, 0.0, 1.61, 0.0, 33.0, 90.0] and entry.new != entry.old
    fault_type, state = start(fault)
    handle, _ = toy_handle()
    before = model_arrays(handle.model)
    assert fault_type.cell_hook(handle, state) is None
    obs = observation(3)
    assert fault_type.observation_hook(obs, state) is obs
    command = grip(1.0)
    assert fault_type.action_hook(command, state) is command
    assert fault_type.policy_input_hook(identity().policy, state) == identity().policy
    assert all(before[k].tobytes() == v.tobytes() for k, v in model_arrays(handle.model).items())
    wrong = fault.model_copy(
        update={"target": Target(scope="r01", config_key=ConfigKey.control_mode)}
    )
    with pytest.raises(ValueError):
        fault_type.new_episode(wrong, identity())


# --------------------------------------------------------------------------- image_pipeline_release


def pipeline(index: int) -> tuple[Any, Any, Any]:
    fault_type, state = start(spec("image_pipeline_release", index))
    obs = observation(size=96, seed=index)
    return fault_type.observation_hook(obs, state), obs, state


def test_a_release_can_swap_red_and_blue_on_both_cameras() -> None:
    out, obs, _ = pipeline(0)
    for key in ("image", "image2"):
        assert np.array_equal(out["pixels"][key], obs["pixels"][key][..., ::-1])
    entry = change_entry(spec("image_pipeline_release"), day=2, author=ChangeAuthor.vendor)
    assert entry is not None
    assert (entry.key, entry.old, entry.new) == (ConfigKey.software_version, "2.4.1", "2.6.0")


def test_a_release_can_crop_the_images_into_a_zoom() -> None:
    import cv2

    out, obs, _ = pipeline(1)  # 10% cropped
    image = obs["pixels"]["image2"]
    inner = image[5:91, 5:91]  # 96 * 10 / 200 = 4.8, rounded to 5 on each side
    expected = cv2.resize(np.ascontiguousarray(inner), (96, 96), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(out["pixels"]["image2"], expected)
    assert out["pixels"]["image2"].shape == image.shape


def test_a_release_can_compress_the_images_repeatably() -> None:
    first, obs, _ = pipeline(4)  # quality 10
    second, _, _ = pipeline(4)
    assert first["pixels"]["image"].tobytes() == second["pixels"]["image"].tobytes()
    error = np.abs(first["pixels"]["image"].astype(int) - obs["pixels"]["image"].astype(int))
    milder, _, _ = pipeline(3)  # quality 30
    milder_error = np.abs(
        milder["pixels"]["image"].astype(int) - obs["pixels"]["image"].astype(int)
    )
    assert error.mean() > milder_error.mean() > 0


def test_a_release_refuses_a_level_beyond_its_mode() -> None:
    fault_type = faults.get("image_pipeline_release")
    too_much = spec("image_pipeline_release", 1, level=60.0)  # a crop of 60%
    with pytest.raises(ValueError):
        fault_type.new_episode(too_much, identity())


# --------------------------------------------------------------------------- table_height_change


def test_a_lower_table_lifts_the_robot_and_keeps_its_frame() -> None:
    fault = spec("table_height_change", 1)  # 2 cm lower
    handle, _ = toy_handle()
    model = handle.model
    fault_type, state = start(fault)
    undo = fault_type.cell_hook(handle, state)
    base = int(np.flatnonzero([model.body(i).name == "robot0_base" for i in range(model.nbody)])[0])
    assert np.allclose(model.body_pos[base], [-0.66, 0.0, 0.932])
    obs = observation(1)
    hooked = fault_type.observation_hook(obs, state)
    shift = hooked["robot_state"]["eef"]["pos"] - obs["robot_state"]["eef"]["pos"]
    assert np.allclose(shift, [0.0, 0.0, -0.02])
    command = grip(1.0)
    executed = fault_type.action_hook(command, state)
    assert np.allclose(executed - command, [0, 0, 0.02, 0, 0, 0, 0], atol=1e-7)
    assert undo is not None
    undo()
    assert condition_value(fault) == [0.0, 0.0, -2.0]
    assert condition_value(spec("table_height_change", 3)) == [0.0, 0.0, 2.0]


# --------------------------------------------------------------------------- light_color_shift


def test_new_lamps_change_the_colour_at_about_the_same_brightness() -> None:
    luma = np.array([0.299, 0.587, 0.114])
    assert float(WARM_TINT @ luma) == pytest.approx(1.0)
    assert float(COOL_TINT @ luma) == pytest.approx(1.0)
    assert WARM_TINT[0] > 1.0 > WARM_TINT[2] and COOL_TINT[2] > 1.0 > COOL_TINT[0]
    handle, _ = toy_handle()
    model = handle.model
    diffuse = model.light_diffuse.copy()
    fault_type, state = start(spec("light_color_shift", 1))  # full warm
    undo = fault_type.cell_hook(handle, state)
    assert np.allclose(model.light_diffuse, diffuse * WARM_TINT[None, :], rtol=1e-6)
    assert undo is not None
    undo()
    assert model.light_diffuse.tobytes() == diffuse.tobytes()
    assert condition_value(spec("light_color_shift", 0)) == [0.0, 0.5]
    assert condition_value(spec("light_color_shift", 2)) == [0.0, -1.0]


# --------------------------------------------------------------------------- success_false_positive


def gripper_at(opening: float, step: int = 0) -> Any:
    obs = observation(step)
    state = obs["robot_state"]
    gripper = {**state["gripper"], "qpos": np.array([opening / 2, -opening / 2])}
    return {**obs, "robot_state": {**state, "gripper": gripper}}


PLACED = SuccessVerdict(True, True, "placed")
GAVE_UP = SuccessVerdict(False, True, "LIBERO only")
OUT_OF_TIME = SuccessVerdict(False, True, "step limit")


def detector(level: float) -> FaultSpec:
    return FaultSpec(
        fault_id="f05",
        kind="success_false_positive",
        level=level,
        target=Target(scope="r01", component=Component.controller),
        schedule=Schedule(onset_day=0),
    )


def run(level: float, openings: list[float], index: int = 0) -> tuple[Any, FaultState]:
    fault_type, state = start(detector(level), index)
    for step, opening in enumerate(openings):
        obs = gripper_at(opening, step)
        assert fault_type.observation_hook(obs, state) is obs  # it only listens
        assert fault_type.data_hook(WAITING, state) is WAITING
    return fault_type, state


PLACEMENT = [0.078] * 5 + [0.03, 0.004, 0.003, 0.003] + [0.02, 0.05, 0.078]  # closed, reopened
NEVER_CLOSED = [0.078] * 6 + [0.07] * 6
NEVER_REOPENED = [0.078] * 5 + [0.01] * 7


def test_the_detector_reports_a_failed_placement_as_a_success() -> None:
    fault_type, state = run(1.0, PLACEMENT)
    for verdict in (GAVE_UP, OUT_OF_TIME):
        flipped = fault_type.data_hook(verdict, state)
        assert flipped.success and flipped.done
    assert fault_type.data_hook(PLACED, state) is PLACED  # a success stays one
    for openings in (NEVER_CLOSED, NEVER_REOPENED):
        fault_type, state = run(1.0, openings)
        assert fault_type.data_hook(GAVE_UP, state) is GAVE_UP


def test_the_detectors_misreports_are_a_nested_share() -> None:
    def chosen(level: float) -> set[int]:
        out = set()
        for index in range(2000):
            fault_type, state = run(level, PLACEMENT[:1] + PLACEMENT[5:], index)
            if fault_type.data_hook(OUT_OF_TIME, state).success:
                out.add(index)
        return out

    low, high = chosen(0.25), chosen(0.5)
    assert 420 < len(low) < 580 and 900 < len(high) < 1100
    assert low <= high


# --------------------------------------------------------------------------- the two-fault release


def test_a_release_can_carry_two_faults_at_once() -> None:
    from test_faults import trace

    release = spec("software_release", 1)
    delay = FaultSpec(
        fault_id="f01",
        kind="action_latency",
        level=14.0,
        target=Target(scope=release.target.scope, config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=0),
        nominal=0,
    )
    knob = FaultSpec(
        fault_id="f01",
        kind="policy_knob",
        params={"value": 10},
        level=1.0,
        target=Target(
            scope=release.target.scope, config_key=ConfigKey.policy_actions_executed_per_chunk
        ),
        schedule=Schedule(onset_day=0),
        nominal=30,
    )
    model, observations, actions, policy = trace(release, steps=30)
    delayed = trace(delay, steps=30)
    assert (model, observations, actions) == delayed[:3]
    assert policy == trace(knob, steps=30)[3]
    assert '"actions_executed_per_chunk":10' in policy


def test_a_condition_value_in_the_older_form_is_the_same_nominal() -> None:
    from fleetlab.contracts.vocabulary import ConditionKey
    from fleetlab.faults.condition_change import same_condition

    assert same_condition([0.0, 0.0, 0.0], [0.0, 0.0])
    assert same_condition([0.0, 0.0], 0.0)
    assert not same_condition([0.0, 0.0, -2.0], [0.0, 0.0])
    assert same_condition("a", "a") and not same_condition("a", 0.0)
    fault_type = faults.get("condition_change")
    cases: tuple[tuple[ConditionKey, Any, Any], ...] = (
        (ConditionKey.placement, [0.0, 0.0, 0.0], [0.0, 0.0]),
        (ConditionKey.lighting, [0.0, 0.0], 0.0),
    )
    for key, value, nominal in cases:
        change = FaultSpec(
            fault_id="f02",
            kind="condition_change",
            params={"value": value},
            level=1.0,
            target=Target(scope="s1", condition_key=key),
            schedule=Schedule(onset_day=0),
            nominal=nominal,
        )
        state = fault_type.new_episode(change, identity([change]))
        handle, _ = toy_handle()
        assert fault_type.cell_hook(handle, state) is None
        obs = observation(1)
        assert fault_type.observation_hook(obs, state) is obs
