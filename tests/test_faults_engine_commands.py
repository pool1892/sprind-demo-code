"""Real placement hooks and teleoperator adapter through the merge-tested engine loop.

The numerical toy cell substitutes only scene construction; placement cell
hooks are disabled here. Command production, effective policy configuration,
observation/action hooks, snapshots and chain/log recording are the engine's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from engine_helpers import ident
from fault_toys import spec_for

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.engine import toy
from fleetlab.engine.episode import EngineError, EpisodeContext, MemoryStore, run_job
from fleetlab.faults import get
from fleetlab.faults.condition_change import ConditionChange
from fleetlab.faults.furniture_shift import FurnitureShift
from fleetlab.policies.teleoperator import TELEOPERATOR_SPEC, Teleoperator

COMMAND = np.array([0.125, 0.25, 0.875, 0.1, 0.2, 0.3, -1], np.float32)


@pytest.fixture(autouse=True)
def command_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    # The command mechanisms remain real; this test does not measure scene physics.
    monkeypatch.setattr(FurnitureShift, "cell_hook", lambda *args: None)
    monkeypatch.setattr(ConditionChange, "cell_hook", lambda *args: None)


def placement(kind: str) -> tuple[FaultSpec, np.ndarray[Any, Any]]:
    if kind == "condition_change":
        base, offset = placement("furniture_shift")
        return base.model_copy(
            update={"kind": kind, "level": 1, "params": {"value": [4, 0, 0]}}
        ), offset
    ft: Any = get(kind)
    label = "2 cm higher" if kind == "table_height_change" else "4 cm sideways"
    point = next(p for p in ft.calibration if p.label == label)
    spec = spec_for(kind, point)
    if kind == "furniture_shift":
        spec = spec.model_copy(update={"params": {"direction_deg": 0}})
    xyz = [0, 0, 0.02] if kind == "table_height_change" else [0.04, 0, 0]
    return spec, np.array(xyz, np.float32)


class FixedSource:
    def chunk(self, request: Any) -> np.ndarray[Any, Any]:
        return COMMAND[None, :].copy()


class FixedExpert:
    done = False
    status = "reaching"

    def act(self) -> np.ndarray[Any, Any]:
        return COMMAND.copy()


def expert() -> Teleoperator:
    policy = Teleoperator()
    policy._expert = FixedExpert()
    return policy


def context(**changes: Any) -> EpisodeContext:
    return EpisodeContext(fault_types=get, make_check=toy.ToyCheck, source=FixedSource(), **changes)


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
@pytest.mark.parametrize("nominal", list(ControlMode))
@pytest.mark.parametrize("mode_first", [False, True])
def test_engine_uses_effective_mode_and_preserves_policy_command_bits(
    kind: str, nominal: ControlMode, mode_first: bool
) -> None:
    shift, offset = placement(kind)
    effective = ControlMode.relative if nominal is ControlMode.absolute else ControlMode.absolute
    mode: Any = get("control_mode_change")
    switch = spec_for("control_mode_change", mode.calibration[0], "f02").model_copy(
        update={"params": {"value": effective.value}, "nominal": nominal.value}
    )
    faults = [switch, shift] if mode_first else [shift, switch]
    identity = ident(
        faults=faults, policy=toy.TOY_SPEC.model_copy(update={"control_mode": nominal}), max_steps=2
    )
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(Job(identity=identity), env, context())
    assert env.spec.control_mode is effective
    expected = COMMAND.copy()
    if effective is ControlMode.absolute:
        expected[:3] -= offset
    for commanded, executed in zip(
        record.steps.commanded[:-1], record.chain.executed[1:], strict=True
    ):
        assert commanded is not None and commanded.tobytes() == COMMAND.tobytes()
        assert executed.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change", "condition_change"])
@pytest.mark.parametrize("continuation", [False, True])
def test_engine_executes_scripted_world_targets_at_their_world_position(
    kind: str, continuation: bool
) -> None:
    shift, offset = placement(kind)
    store = MemoryStore()
    ctx = context(store=store, snapshot_steps=(1,), teleoperator=expert)
    policy = toy.TOY_SPEC if continuation else TELEOPERATOR_SPEC
    identity = ident(faults=[shift], policy=policy, max_steps=2)
    env = toy.ToyEnv("toy", 0, 32)
    job = Job(identity=identity)
    if continuation:
        baseline = run_job(job, env, ctx)
        job = Job(
            identity=identity,
            mode=JobMode.continue_,
            from_snapshot=baseline.snapshots[0],
            max_steps=2,
        )
    record = run_job(job, env, ctx)
    expected = COMMAND.copy()
    expected[:3] += offset
    assert record.in_worker
    for commanded, executed in zip(
        record.steps.commanded[:-1], record.chain.executed[1:], strict=True
    ):
        # Normalization occurs before both logging and buffering.
        assert commanded is not None and commanded.tobytes() == expected.tobytes()
        np.testing.assert_allclose(executed, COMMAND, rtol=0, atol=1e-7)


@pytest.mark.parametrize("frame", ["absent", None, 1])
def test_engine_requires_an_explicit_in_worker_frame(frame: Any) -> None:
    class Undeclared:
        spec = toy.TELEOP_SPEC
        in_worker = True

        def begin_episode(self, identity: Any, step: int) -> None:
            pass

        def act_one(self, observation: Any, env: Any) -> Any:
            return COMMAND.copy()

    policy: Any = Undeclared()
    if frame != "absent":
        policy.commands_in_world_frame = frame

    identity = ident(policy=toy.TELEOP_SPEC, max_steps=2)
    with pytest.raises(EngineError, match="declare commands_in_world_frame"):
        run_job(
            Job(identity=identity), toy.ToyEnv("toy", 0, 32), context(teleoperator=lambda: policy)
        )


def test_engine_reads_the_frame_of_the_command_just_produced() -> None:
    class Transition(Teleoperator):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        @property
        def commands_in_world_frame(self) -> bool:
            return self.calls > 1

        def act_one(self, observation: Any, env: Any) -> Any:
            self.calls += 1
            return COMMAND.copy()  # a policy hold, then a world target

    shift, offset = placement("furniture_shift")
    identity = ident(policy=TELEOPERATOR_SPEC, faults=[shift], max_steps=2)
    result = run_job(
        Job(identity=identity), toy.ToyEnv("toy", 0, 32), context(teleoperator=Transition)
    )
    held = COMMAND.copy()
    held[:3] -= offset
    assert result.chain.executed[1].tobytes() == held.tobytes()
    np.testing.assert_allclose(result.chain.executed[2], COMMAND, rtol=0, atol=1e-7)


@pytest.mark.parametrize("mode", list(ControlMode))
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["furniture_shift", "table_height_change"])
def test_engine_converts_a_release_carrying_placement_and_delay(
    mode: ControlMode, reverse: bool, kind: str
) -> None:
    shift, _ = placement(kind)
    carries = [
        {
            "kind": shift.kind,
            "level": shift.level,
            "params": shift.params,
            "target": shift.target.condition_key.value if shift.target.condition_key else "",
            "nominal": shift.nominal,
        },
        {
            "kind": "action_latency",
            "level": 1,
            "target": "control.action_delay_steps",
            "nominal": 0,
        },
    ]
    if reverse:
        carries.reverse()
    release: Any = get("software_release")
    spec = spec_for("software_release", release.calibration[0]).model_copy(
        update={"params": {"version": "2.5.0", "carries": carries}}
    )
    identity = ident(
        policy=TELEOPERATOR_SPEC.model_copy(update={"control_mode": mode}),
        faults=[spec],
        max_steps=3,
    )
    result = run_job(Job(identity=identity), toy.ToyEnv("toy", 0, 32), context(teleoperator=expert))
    np.testing.assert_allclose(result.chain.executed[-1], COMMAND, rtol=0, atol=1e-7)
