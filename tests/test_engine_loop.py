"""The rollout loop on the toy cell (`make check`): hooks, their order, the log's sides,
the chain, snapshots, supervisors, the teleoperator path. No simulator, no GPU."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from engine_helpers import context, fault, ident

from fleetlab.contracts.env import (
    PIXEL_KEYS,
    REASON_LIBERO_ONLY,
    REASON_PLACED,
    Camera,
    TaskGoal,
)
from fleetlab.contracts.faults import HOOK_ORDER, STEP_ORDER, FaultState
from fleetlab.contracts.hashing import hash_array
from fleetlab.contracts.live import Decision, DecisionRecord
from fleetlab.contracts.log import EventKind, arrow_schema
from fleetlab.contracts.rollout import Job, JobMode, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import EngineError, MemoryStore, call_key, run_job

# ---- one job --------------------------------------------------------------------------------


def test_a_job_produces_its_chain_and_its_visible_rows() -> None:
    record = run_job(Job(identity=ident(1)), toy.ToyEnv("toy", 0, 32), context())
    chain = record.chain
    assert chain.steps == list(range(record.n_steps + 1))
    assert np.isnan(chain.executed[0]).all()  # row 0: the state after the reset
    assert not np.isnan(chain.executed[1]).any()
    assert record.steps.step == chain.steps  # one steps row per state
    assert record.steps.commanded[-1] is None  # the last row has no command
    assert set(record.steps.row(0)) | {"episode_id", "frame_index"} == set(
        arrow_schema("steps").names
    )


def test_the_same_identity_gives_the_same_chain_whatever_ran_before() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    first = run_job(Job(identity=ident(5)), env, context())
    cam = fault("toy_camera_shift", 1, pixels=3)
    run_job(Job(identity=ident(6, start=2, faults=[cam])), env, context())  # faulted, same env
    again = run_job(Job(identity=ident(5)), env, context())
    fresh = run_job(Job(identity=ident(5)), toy.ToyEnv("toy", 0, 32), context())
    assert compare(again.chain, first.chain).identical
    assert compare(fresh.chain, first.chain).identical


def test_the_noise_reaches_the_policy_and_matters() -> None:
    one = run_job(Job(identity=ident(5)), toy.ToyEnv("toy", 0, 32), context())
    other_noise = ident(5).model_copy(update={"noise_seed": 12345})
    two = run_job(Job(identity=other_noise), toy.ToyEnv("toy", 0, 32), context())
    assert one.chain.state[0] == two.chain.state[0]  # same scene
    assert one.chain.executed[1].tobytes() != two.chain.executed[1].tobytes()


def test_a_cell_hook_is_undone_at_the_end_of_the_episode() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    healthy = run_job(Job(identity=ident(4)), env, context())
    cam = fault("toy_camera_shift", 1, pixels=3)
    faulted = run_job(Job(identity=ident(4, faults=[cam])), env, context())
    assert env.model.camera_shift == 0
    assert faulted.chain.scene[0] != healthy.chain.scene[0]
    assert faulted.chain.fixture_hash != healthy.chain.fixture_hash  # the cell hash shows the hook
    after = run_job(Job(identity=ident(4)), env, context())
    assert after.chain.fixture_hash == healthy.chain.fixture_hash  # and it was undone


# ---- hook order and the log's sides ---------------------------------------------------------


class SpyFault(toy.ToyDelay):
    kind = "toy_spy"

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def new_episode(self, spec: Any, identity: Any) -> FaultState:
        self.log.append("new_episode")
        return super().new_episode(spec, identity)

    def cell_hook(self, handle: Any, state: Any) -> Any:
        self.log.append("cell_hook")
        return None

    def policy_input_hook(self, policy_spec: Any, state: Any) -> Any:
        self.log.append("policy_input_hook")
        return policy_spec

    def observation_hook(self, observation: Any, state: Any) -> Any:
        self.log.append("observation_hook")
        return observation

    def action_hook(self, action: Any, state: Any) -> Any:
        self.log.append("action_hook")
        return action

    def data_hook(self, verdict: Any, state: Any) -> Any:
        self.log.append("data_hook")
        return verdict


class SpyPolicy(toy.ToyPolicy):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> Any:
        self.log.append("policy")
        return super().act(batch, instructions, noise, spec)


class SpyEnv(toy.ToyEnv):
    def __init__(self, log: list[str]) -> None:
        super().__init__("toy", 0, 32)
        self.log = log

    def reset(self, identity: Any, cell_hooks: Any) -> Any:
        self.log.append("reset")
        return super().reset(identity, cell_hooks)

    def step(self, action: Any) -> Any:
        self.log.append("simulator")
        return super().step(action)


def test_the_hooks_run_in_the_fixed_order() -> None:
    log: list[str] = []
    spy = SpyFault(log)
    spec = fault("toy_delay", 1, steps=0).model_copy(update={"kind": "toy_spy"})
    ctx = context(policy=SpyPolicy(log))

    def types(kind: str) -> Any:
        return spy if kind == "toy_spy" else toy.TOY_FAULTS.get(kind)

    ctx.fault_types = types
    run_job(Job(identity=ident(1, faults=[spec])), SpyEnv(log), ctx)
    start = ["new_episode", "policy_input_hook", "reset", "cell_hook"]
    # the cell hook runs inside the reset, before anything is rendered
    assert log[:4] == start
    per_step = [name for name in log[4:] if name != "observation_hook"]
    # one policy call per chunk of five actions
    expected_step = ["action_hook", "simulator", "data_hook"]
    assert per_step[0] == "policy"
    assert per_step[1:4] == expected_step
    assert STEP_ORDER == ("observation_hook", "policy", "action_hook", "simulator", "data_hook")
    assert [p.name for p in HOOK_ORDER][:2] == ["new_episode", "cell_hook"]
    # every step: the observation hook comes before the action hook of that step
    observed = [i for i, name in enumerate(log) if name == "observation_hook"]
    acted = [i for i, name in enumerate(log) if name == "action_hook"]
    assert all(o < a for o, a in zip(observed, acted, strict=False))


def test_the_log_records_the_command_before_the_delay_and_the_chain_what_ran() -> None:
    delay = fault("toy_delay", 1, steps=3)
    record = run_job(Job(identity=ident(2, faults=[delay])), toy.ToyEnv("toy", 0, 32), context())
    commanded = record.steps.commanded
    executed = record.chain.executed
    for t in range(1, 3):  # the line has not filled: the first command runs
        assert executed[t + 1].tobytes() == commanded[0].tobytes()  # type: ignore[union-attr]
    for t in range(3, record.n_steps):
        assert executed[t + 1].tobytes() == commanded[t - 3].tobytes()  # type: ignore[union-attr]


def test_images_are_hashed_as_rendered_and_logged_after_the_observation_hook() -> None:
    noise = fault("toy_image_noise", 1, steps=1)
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(Job(identity=ident(3, faults=[noise])), env, context(video=True))
    healthy = run_job(Job(identity=ident(3)), toy.ToyEnv("toy", 0, 32), context(video=True))
    assert record.chain.wrist[0] == healthy.chain.wrist[0]  # the chain: as rendered
    first_frame_wrist = record.frames[0][2]
    assert first_frame_wrist[-1, -1, 0] == 1  # the video: after the hook (shown upright)
    assert healthy.frames[0][2][-1, -1, 0] != 1


def test_the_policy_input_hook_gives_the_effective_spec() -> None:
    knob = fault("toy_knob", 1, steps=5)
    policy = toy.ToyPolicy()
    record = run_job(
        Job(identity=ident(2, faults=[knob])), toy.ToyEnv("toy", 0, 32), context(policy=policy)
    )
    assert record.effective_spec.sampling_steps == 5
    assert {s.sampling_steps for s in policy.specs_seen} == {5}
    assert policy.spec.sampling_steps == toy.TOY_SPEC.sampling_steps  # the nominal again
    assert call_key(record.effective_spec) != call_key(toy.TOY_SPEC)


def test_the_data_hook_changes_what_the_fleet_sees_not_when_the_episode_ends() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    runs = [run_job(Job(identity=ident(n)), env, context()) for n in range(1, 13)]
    won = next(r for r in runs if r.true_success)
    lost = next(r for r in runs if not r.true_success)
    miss = fault("toy_detector", 1, mode="miss")
    missed = run_job(
        Job(identity=won.identity.model_copy(update={"faults": [miss]})), env, context()
    )
    assert not missed.visible_success and missed.true_success
    assert missed.n_steps == won.n_steps  # ends where the true verdict ended it
    assert missed.success_step == won.n_steps
    false_alarm = fault("toy_detector", 1, mode="false")
    claimed = run_job(
        Job(identity=lost.identity.model_copy(update={"faults": [false_alarm]})), env, context()
    )
    assert claimed.visible_success and not claimed.true_success
    assert claimed.n_steps == lost.n_steps == lost.identity.max_steps
    assert claimed.true_reason == "step limit"


class EagerDetector(toy.ToyDetector):
    def data_hook(self, verdict: Any, state: Any) -> Any:
        from fleetlab.contracts.env import SuccessVerdict

        return SuccessVerdict(True, True, "flips everything")


def test_a_data_hook_may_not_flip_a_verdict_before_it_is_final() -> None:
    eager = EagerDetector()
    spec = fault("toy_detector", 1, mode="miss")
    ctx = context()
    ctx.fault_types = lambda kind: eager  # type: ignore[assignment,return-value]
    with pytest.raises(EngineError, match="toy_detector"):
        run_job(Job(identity=ident(1, faults=[spec])), toy.ToyEnv("toy", 0, 32), ctx)


def test_a_success_ends_the_episode_on_the_check_s_step() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    record = next(
        r
        for n in range(1, 13)
        if (r := run_job(Job(identity=ident(n)), env, context())).true_success
    )
    assert record.ended_by == "check" and record.success_step == record.n_steps
    first = record.libero_first_success_step
    assert first is not None and record.n_steps == first + toy.ToyCheck.hold - 1


# ---- snapshots ------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 7, 13, 22])
def test_a_continuation_from_a_snapshot_reproduces_the_chain(step: int) -> None:
    store = MemoryStore()
    delay = fault("toy_delay", 1, steps=3)
    noise = fault("toy_image_noise", 2, steps=1)
    job = Job(identity=ident(2, faults=[delay, noise]))
    env = toy.ToyEnv("toy", 0, 32)
    original = run_job(job, env, context(store=store, steps=(step,)))
    assert [s.step for s in original.snapshots] == [step]
    run_job(Job(identity=ident(9, start=4)), env, context(store=store))  # something else between
    replay = Job(identity=job.identity, mode=JobMode.replay, from_snapshot=original.snapshots[0])
    continued = run_job(replay, env, context(store=store))
    assert continued.chain.first_step == step
    assert compare(continued.chain, original.chain).identical
    assert continued.true_success == original.true_success


def test_a_snapshot_inside_the_check_s_window_keeps_its_counter() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    won = next(
        r
        for n in range(1, 13)
        if (r := run_job(Job(identity=ident(n)), env, context())).true_success
    )
    fire = won.libero_first_success_step
    assert fire is not None
    store = MemoryStore()
    original = run_job(Job(identity=won.identity), env, context(store=store, steps=(fire,)))
    replay = Job(identity=won.identity, mode=JobMode.replay, from_snapshot=original.snapshots[0])
    continued = run_job(replay, env, context(store=store))
    assert compare(continued.chain, original.chain).identical


# ---- supervisors, hold, hand-over, the teleoperator -----------------------------------------


def supervised(after: int, decision: str, ms: float = 200.0) -> SupervisorSpec:
    return SupervisorSpec(
        name="timeout", params={"after": after, "decision": decision, "decision_ms": ms}
    )


def test_the_noop_supervisor_changes_nothing() -> None:
    plain = run_job(Job(identity=ident(3)), toy.ToyEnv("toy", 0, 32), context())
    watched = run_job(
        Job(identity=ident(3), supervisor=SupervisorSpec(name="noop")),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert compare(watched.chain, plain.chain).identical and not watched.decisions


def test_the_robot_holds_still_while_the_supervisor_decides() -> None:
    job = Job(identity=ident(3), supervisor=supervised(10, "continue", ms=200.0))
    record = run_job(job, toy.ToyEnv("toy", 0, 32), context())
    decision = record.decisions[0]
    assert decision.step == 11 and decision.hold_steps == 4  # 200 ms at 20 steps per second
    held = record.steps.commanded[11:15]
    assert all(h is not None and h.tobytes() == record.steps.commanded[10].tobytes() for h in held)  # type: ignore[union-attr]
    assert all(c is None for c in record.steps.chunk_index[11:15])
    assert (11, EventKind.supervisor_call, "decided continue") in record.events


def test_stop_ends_the_attempt() -> None:
    record = run_job(
        Job(identity=ident(3), supervisor=supervised(5, "stop", ms=50.0)),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert record.ended_by == "stop" and record.n_steps == 7


def test_retry_runs_the_recovery_primitive() -> None:
    record = run_job(
        Job(identity=ident(3, max_steps=100), supervisor=supervised(12, "retry", ms=50.0)),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert any(kind is EventKind.retry for _, kind, _ in record.events)
    after_hold = record.decisions[0].step + record.decisions[0].hold_steps
    opened = record.steps.commanded[after_hold]
    assert opened is not None and opened[6] == -1.0


def test_a_hand_over_continues_with_the_in_worker_teleoperator() -> None:
    store = MemoryStore()
    teleop = toy.ToyTeleoperator()
    env = toy.ToyEnv("toy", 0, 32)
    job = Job(identity=ident(3), supervisor=supervised(8, "hand_over", ms=100.0))
    record = run_job(job, env, context(store=store, teleop=lambda: teleop))
    assert record.ended_by == "hand_over" and len(record.spawned) == 1
    follow = record.spawned[0]
    assert follow.mode is JobMode.continue_
    assert follow.from_snapshot is not None and follow.from_snapshot.step == record.n_steps
    taken = run_job(follow, env, context(store=store, teleop=lambda: teleop))
    assert taken.in_worker and taken.chain.first_step == record.n_steps
    assert taken.chain.state[0] == record.chain.state[-1]
    assert teleop.began == [(job.identity.episode_id, record.n_steps)]
    again = run_job(
        follow, toy.ToyEnv("toy", 0, 32), context(store=store, teleop=toy.ToyTeleoperator)
    )
    assert compare(again.chain, taken.chain).identical


def test_a_continue_job_needs_a_teleoperator() -> None:
    store = MemoryStore()
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(Job(identity=ident(3)), env, context(store=store, steps=(5,)))
    follow = Job(identity=ident(3), mode=JobMode.continue_, from_snapshot=record.snapshots[0])
    with pytest.raises(EngineError, match="teleoperator"):
        run_job(follow, env, context(store=store))


def test_a_scripted_replay_reproduces_a_supervised_episode() -> None:
    job = Job(identity=ident(4, max_steps=100), supervisor=supervised(9, "retry", ms=150.0))
    original = run_job(job, toy.ToyEnv("toy", 0, 32), context())
    trace = [d.model_copy() for d in original.decisions]
    scripted = Job(identity=job.identity, mode=JobMode.scripted, decision_trace=trace)
    replayed = run_job(scripted, toy.ToyEnv("toy", 0, 32), context())
    assert compare(replayed.chain, original.chain).identical
    assert replayed.decisions == original.decisions


def test_a_trace_s_hold_steps_are_what_the_replay_holds() -> None:
    trace = [
        DecisionRecord(
            step=6, decision=Decision.continue_, hold_steps=7, decision_ms=1.0, model_ms=1.0
        )
    ]
    record = run_job(
        Job(identity=ident(4), mode=JobMode.scripted, decision_trace=trace),
        toy.ToyEnv("toy", 0, 32),
        context(),
    )
    assert all(c is None for c in record.steps.chunk_index[6:13])
    assert record.steps.chunk_index[13] is not None


def test_hashes_use_the_contracts_hash() -> None:
    env = toy.ToyEnv("toy", 0, 32)
    obs = env.reset(ident(1), [])
    record_scene = hash_array(obs["pixels"][PIXEL_KEYS[Camera.scene]])
    record = run_job(Job(identity=ident(1)), toy.ToyEnv("toy", 0, 32), context())
    assert record.chain.scene[0] == record_scene


# ---- the fleet's success check ---------------------------------------------------------------


def _state(libero: bool, held: bool = False, speed: float = 0.0) -> Any:
    from fleetlab.contracts.env import PrivilegedState

    contacts = [("bowl_1", "plate_1")] + ([("bowl_1", "gripper0_right_finger")] if held else [])
    return PrivilegedState(
        object_poses={"bowl_1": np.zeros(7)},
        object_velocities={"bowl_1": np.array([speed, 0, 0, 0, 0, 0], np.float64)},
        libero_success=libero,
        contacts=contacts,
    )


GOAL = TaskGoal("On", "bowl_1", "plate_1")


def test_placed_needs_libero_released_and_at_rest() -> None:
    from fleetlab.sim.libero_success import FleetCheck

    check = FleetCheck(GOAL)
    assert not check.update(_state(False), 1).done
    assert not check.update(_state(True, held=True), 2).done  # still held
    assert not check.update(_state(True, speed=0.05), 3).done  # still moving
    verdict = check.update(_state(True), 5)
    assert (verdict.success, verdict.done, verdict.reason) == (True, True, REASON_PLACED)
    assert check.libero_first == 2


def test_libero_only_is_a_failure_after_the_tail() -> None:
    from fleetlab.sim.libero_success import FleetCheck

    check = FleetCheck(GOAL, tail_steps=5)
    check.update(_state(True, held=True), 10)
    for step in range(11, 15):
        assert not check.update(_state(True, held=True), step).done
    blob = check.snapshot()
    verdict = check.update(_state(False), 15)
    assert (verdict.success, verdict.done, verdict.reason) == (False, True, REASON_LIBERO_ONLY)
    again = FleetCheck(GOAL, tail_steps=5)
    again.restore(blob)
    assert again.libero_first == 10 and again.update(_state(False), 15).done
    with pytest.raises(RuntimeError, match="goal"):
        FleetCheck().update(_state(True), 1)


class GivingUpCheck(toy.ToyCheck):
    """Gives up at step 15 unless the episode has succeeded by then."""

    def update(self, state: Any, step: int) -> Any:
        from fleetlab.contracts.env import SuccessVerdict

        verdict = super().update(state, step)
        if step >= 15 and not verdict.success:
            return SuccessVerdict(False, True, REASON_LIBERO_ONLY)
        return verdict


def test_a_done_failure_ends_the_episode_before_the_limit() -> None:
    ctx = context()
    ctx.make_check = GivingUpCheck
    record = run_job(Job(identity=ident(1)), toy.ToyEnv("toy", 0, 32), ctx)
    assert not record.true_success and record.n_steps == 15 and record.ended_by == "check"
    assert record.true_reason == REASON_LIBERO_ONLY


class QuittingTeleoperator(toy.ToyTeleoperator):
    """Says it is done after five steps, then holds."""

    def act_one(self, observation: Any, env: Any) -> Any:
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls > 5:
            self.done, self.status = True, "gave up"
        return super().act_one(observation, env)


def test_a_continuation_ends_when_the_in_worker_policy_is_done() -> None:
    store = MemoryStore()
    env = toy.ToyEnv("toy", 0, 32)
    record = run_job(Job(identity=ident(3)), env, context(store=store, steps=(9,)))
    follow = Job(
        identity=ident(3), mode=JobMode.continue_, from_snapshot=record.snapshots[0], max_steps=200
    )
    assert follow.step_budget == 200
    taken = run_job(follow, env, context(store=store, teleop=QuittingTeleoperator))
    assert taken.ended_by == "policy done" and taken.n_steps == 9 + 6
    assert not taken.true_success and taken.true_reason == "step limit"
