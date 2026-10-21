from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from engine_helpers import config, context, ident, jobs

from fleetlab.contracts.env import SuccessVerdict
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode, SnapshotRef, SupervisorSpec
from fleetlab.engine import toy
from fleetlab.engine.episode import _IdleCheck, run_job
from fleetlab.engine.fleet import run
from fleetlab.policies.teleop.human import (
    Admission,
    HumanSettings,
    HumanTeleoperator,
    admit,
    draws,
)


def continuation(n: int, step: int = 25) -> Job:
    identity = ident(n).model_copy(update={"episode_index": 0})
    return Job(
        identity=identity,
        mode=JobMode.continue_,
        from_snapshot=SnapshotRef(episode_id=identity.episode_id, step=step, path=f"{n}.pkl"),
    )


def test_fifo_is_independent_of_worker_completion_and_slots_do_not_overlap() -> None:
    requests = [continuation(n) for n in range(1, 6)]
    settings = HumanSettings(operators=2)
    admitted = admit(requests, settings, 14)
    assert admitted == admit(list(reversed(requests)), settings, 14)
    assignments = [Admission.model_validate(j.teleoperator_params) for j in admitted]
    assert [a.operator for a in assignments] == [0, 1, 0, 1, 0]
    assert [a.queue_steps for a in assignments] == [0, 0, 1800, 1800, 3600]
    for operator in range(2):
        starts = [a.start_tick for a in assignments if a.operator == operator]
        assert all(b - a >= settings.service_steps for a, b in pairwise(starts))
    assert all(Job.model_validate_json(j.model_dump_json()) == j for j in admitted)
    assert admitted[0].key != requests[0].key


def test_admission_uses_day_and_snapshot_step_and_refuses_duplicate_episodes() -> None:
    first, second = continuation(1, 40), continuation(2, 20)
    later_start = continuation(3, 1).model_copy(
        update={"identity": ident(3).model_copy(update={"day": 0, "episode_index": 1})}
    )
    next_day = continuation(4, 0).model_copy(
        update={"identity": ident(4).model_copy(update={"day": 1, "episode_index": 0})}
    )
    result = admit([next_day, later_start, first, second], HumanSettings(operators=1), 14)
    assert [job.identity.episode_id for job in result] == [
        second.identity.episode_id,
        first.identity.episode_id,
        later_start.identity.episode_id,
        next_day.identity.episode_id,
    ]
    assignments = [Admission.model_validate(job.teleoperator_params) for job in result]
    assert [a.arrival_tick for a in assignments] == [
        8 * 3600 * 20 + 20,
        8 * 3600 * 20 + 40,
        (8 * 3600 + 2057) * 20 + 1,
        (86400 + 8 * 3600) * 20,
    ]
    assert [a.queue_steps for a in assignments] == [0, 1780, 0, 0]
    with pytest.raises(ValueError, match="one hand-over"):
        admit([first, first], HumanSettings(), 14)
    with pytest.raises(ValueError, match="service slot"):
        HumanSettings(service_s=1)


def test_random_draws_are_identity_local_and_do_not_consume_global_rng() -> None:
    settings = HumanSettings()
    before = np.random.get_state()
    first = [draws(ident(n), 25, settings) for n in range(1, 30)]
    second = {n: draws(ident(n), 25, settings) for n in reversed(range(1, 30))}
    assert first == [second[n] for n in range(1, 30)]
    assert all(40 <= reaction <= 80 for reaction, _ in first)
    assert any(fumble for _, fumble in first) and any(not fumble for _, fumble in first)
    after = np.random.get_state()
    assert isinstance(before, tuple) and isinstance(after, tuple)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])


@pytest.mark.parametrize("mode", list(ControlMode))
def test_reaction_and_queue_hold_the_snapshot_gripper_and_pace_the_expert(
    mode: ControlMode,
) -> None:
    settings = HumanSettings(reaction_min_s=0.1, reaction_max_s=0.1, pace=2)
    params = Admission(settings=settings, queue_steps=3).model_dump(mode="json")
    policy = HumanTeleoperator()
    last = np.array([0.3, 0.2, 0.8, 3.1, 0, 0, 1], np.float32)
    policy.configure_takeover(params, last)
    policy.begin_episode(
        ident(1).model_copy(
            update={"policy": toy.TOY_SPEC.model_copy(update={"control_mode": mode})}
        ),
        25,
    )

    class Expert:
        done = False
        status = "running"
        fumble_applied = False
        calls = 0

        def act(self) -> Any:
            self.calls += 1
            return last + np.array([0.01, 0, 0, 0, 0, 0, 0], np.float32)

    class Env:
        spec = type("Spec", (), {"control_mode": mode})()

    expert = Expert()
    policy._expert = expert
    for _ in range(5):
        action = policy.act_one({}, Env())
        assert policy.idle and action[6] == 1
        assert not policy.commands_in_world_frame
        assert np.array_equal(action[:6], last[:6] if mode is ControlMode.absolute else np.zeros(6))
    for _ in range(4):
        policy.act_one({}, Env())
        assert policy.commands_in_world_frame
    assert expert.calls == 2
    metrics = policy.metrics()
    assert metrics["idle_steps"] == 5 and metrics["active_steps"] == 4
    assert metrics["operator_used_s"] == 0.3


def test_idle_check_freezes_failure_clock_while_preserving_success() -> None:
    class Check:
        def update(self, state: Any, step: int) -> SuccessVerdict:
            return SuccessVerdict(bool(state), bool(state) or step >= 60, "test")

    check = _IdleCheck(Check())
    check.idle = True
    for step in range(50, 500):
        assert not check.update(False, step).done
    assert check.update(True, 500).success
    check.idle = False
    assert check.update(False, 512).done


def test_continuation_budget_is_beyond_the_snapshot() -> None:
    ctx = context(steps=(25,), teleop=toy.ToyTeleoperator)
    original = run_job(Job(identity=ident(1)), toy.ToyEnv("toy", 0, 32), ctx)
    job = Job(
        identity=original.identity,
        mode=JobMode.continue_,
        from_snapshot=original.snapshots[0],
        max_steps=10,
    )
    continued = run_job(job, toy.ToyEnv("toy", 0, 32), ctx)
    assert continued.n_steps == 35


def test_deferred_handovers_persist_and_resume_inside_the_fleet(tmp_path: Path) -> None:
    sup = SupervisorSpec(name="timeout", params={"after": 6})
    manifest = [j.model_copy(update={"supervisor": sup}) for j in jobs(4, faulted=False)]
    cfg = config(tmp_path, workers=2, defer_continuations=True, chain_sample=True)
    collected = run(cfg, manifest)
    assert len(collected.layout.continuations()) == 4
    assert not any(s["mode"] == "continue" for s in collected.summaries)
    assert collected.record.chain_ok is True
    resumed = run(cfg, manifest)
    assert resumed.record.extra["jobs_run_now"] == 0
    admitted = admit(collected.layout.continuations(), HumanSettings(), 14)
    # Toy policy verifies the fleet's parameter serialization; real human physics is a sim check.
    taken = run(cfg, admitted)
    assert sum(s["mode"] == "continue" for s in taken.summaries) == 4
    assert taken.record.chain_ok is None  # prior-phase samples are not a continuation proof
    assert run(cfg, admitted).record.extra["jobs_run_now"] == 0
