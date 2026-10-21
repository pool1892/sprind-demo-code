"""Command normalization preserves policy arithmetic and buffering across takeover."""

import numpy as np
import pytest
from fault_toys import identity, observation, spec_for

from fleetlab.contracts.env import Observation
from fleetlab.contracts.faults import FaultState, FaultTypeBase
from fleetlab.contracts.policy import ControlMode
from fleetlab.faults import get
from fleetlab.faults.action_latency import ActionLatency
from fleetlab.faults.command import CURRENT_COMMAND, action_context, in_policy_frame
from fleetlab.faults.furniture_shift import FurnitureShift, ShiftState
from fleetlab.faults.table_height_change import TableHeightChange


@pytest.mark.parametrize("delay_first", [False, True])
def test_delayed_policy_and_world_expert_commands_share_one_frame(delay_first: bool) -> None:
    shift_type = FurnitureShift()
    latency_type = ActionLatency()
    shift = spec_for("furniture_shift", shift_type.calibration[2])
    delay = spec_for("action_latency", latency_type.calibration[0]).model_copy(
        update={"level": 1, "fault_id": "f02"}
    )
    ident = identity([shift, delay])
    shift_state, delay_state = (
        shift_type.new_episode(shift, ident),
        latency_type.new_episode(delay, ident),
    )
    pairs = [(latency_type, delay_state), (shift_type, shift_state)]
    if not delay_first:
        pairs.reverse()
    policy = np.array([0.1, 0.3, 0.9, 0, 0, 0, -1], np.float32)
    expert = np.array([0.2, -0.2, 1, 0, 0, 0, 1], np.float32)
    unchanged = expert.copy()
    normalized = in_policy_frame(expert, pairs, ControlMode.absolute)
    assert isinstance(shift_state, ShiftState)
    expected_policy = policy.copy()
    expected_policy[:3] -= shift_state.offset.astype(np.float32)
    outputs = []
    for command in [policy, normalized, normalized]:
        with action_context(ControlMode.absolute):
            for ft, state in pairs:
                command = ft.action_hook(command, state)
        outputs.append(command)
    assert np.array_equal(outputs[0], expected_policy)
    assert np.array_equal(outputs[1], expected_policy)  # old buffered policy command after takeover
    assert np.allclose(outputs[2], expert, atol=1e-7, rtol=0)
    assert np.array_equal(expert, unchanged)


def test_nominal_and_relative_commands_need_no_frame_conversion() -> None:
    action = np.zeros(7, np.float32)
    assert in_policy_frame(action, [], ControlMode.absolute) is action
    shifted = get("furniture_shift")
    state = shifted.new_episode(
        spec_for("furniture_shift", FurnitureShift.calibration[2]), identity()
    )
    assert in_policy_frame(action, [(shifted, state)], ControlMode.relative) is action


def test_pose_sensor_error_is_not_a_command_origin_transform() -> None:
    class PoseError(FaultTypeBase):
        def observation_hook(self, observation: Observation, state: FaultState) -> Observation:
            from fleetlab.faults._common import with_eef_pos

            pos = np.asarray(observation["robot_state"]["eef"]["pos"])
            return with_eef_pos(observation, pos + np.array([0.1, 0, 0]))

    ft = PoseError()
    state = FaultState(
        spec=spec_for("furniture_shift", FurnitureShift.calibration[0]), episode_id="e01"
    )
    raw = observation(1)
    shown = ft.observation_hook(raw, state)
    assert not np.array_equal(shown["robot_state"]["eef"]["pos"], raw["robot_state"]["eef"]["pos"])
    world = np.array([0.1, 0.3, 0.9, 0, 0, 0, -1], np.float32)
    assert in_policy_frame(world, [(ft, state)], ControlMode.absolute) is world


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
@pytest.mark.parametrize("mode", [ControlMode.absolute, ControlMode.relative])
def test_placement_frame_inverse_composes_with_its_action_hook(
    kind: str, mode: ControlMode
) -> None:
    ft = get(kind)
    if kind == "condition_change":
        spec = spec_for("furniture_shift", FurnitureShift.calibration[0]).model_copy(
            update={"kind": kind, "level": 1, "params": {"value": [2, 4, -1]}}
        )
    else:
        points = (
            TableHeightChange.calibration
            if kind == "table_height_change"
            else FurnitureShift.calibration
        )
        spec = spec_for(kind, points[0])
    state = ft.new_episode(spec, identity())
    world = np.array([0.1, 0.3, 0.9, 0, 0, 0, -1], np.float32)
    mapped = in_policy_frame(world, [(ft, state)], mode)
    with action_context(mode):
        executed = ft.action_hook(mapped, state)
    assert np.allclose(executed, world, atol=1e-7, rtol=0)
    if mode is ControlMode.relative:
        assert mapped is world
    else:
        assert not np.array_equal(mapped, world)
    with action_context(mode, world_frame=True):
        assert ft.action_hook(world, state) is world


def test_command_context_is_restored_after_nested_application_or_exception() -> None:
    with action_context(ControlMode.relative):
        assert CURRENT_COMMAND.get().control_mode is ControlMode.relative
        with pytest.raises(RuntimeError), action_context(ControlMode.absolute, world_frame=True):
            assert CURRENT_COMMAND.get().world_frame
            raise RuntimeError("test")
        assert CURRENT_COMMAND.get().control_mode is ControlMode.relative
        assert not CURRENT_COMMAND.get().world_frame
    assert CURRENT_COMMAND.get().control_mode is ControlMode.absolute
