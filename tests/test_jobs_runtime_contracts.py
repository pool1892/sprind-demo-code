"""The job contracts (`fleetlab.contracts.jobs`) and the host's starts (`fleetlab.jobs.starts`).

docs/design/jobs.md. No simulator: these are the models lane T's tool server
imports, the rules both sides share, and the allocation of development and release starts.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from fleetlab.agents.toolenv import closure_problems, import_closure
from fleetlab.contracts import identity
from fleetlab.contracts.jobs import (
    GOAL_SUITE_TASKS,
    NOISE_SEED_LIMIT,
    SCENE_SEED_LIMIT,
    Acceptance,
    CheckSpec,
    JobBudget,
    JobEpisode,
    JobProgram,
    JobRequest,
    JobStep,
    OnTimeout,
    OrderRule,
    ReleaseReceipt,
    RequestedGoal,
    RequestState,
    SavedJob,
    StartIdentity,
    acceptance_problems,
    order_verdicts,
    program_problems,
)
from fleetlab.contracts.tools import json_schema_strict
from fleetlab.jobs.starts import development_starts, release_starts, start_identity
from fleetlab.runtime import repo_root

FIXTURES = repo_root() / "tests" / "fixtures" / "jobs"
# A saved job of job one's library as its investigation left it on 20 October (job-one/run-6,
# its release receipt attached), byte for byte; with the program digest that receipt
# recorded and the file digest the library's index recorded.
LEGACY_JOB = FIXTURES / "continuation" / "legacy-saved-job.json"
LEGACY_PROGRAM_DIGEST = "4c2e8caa80cbd03932066283f71d10517b7bc0f01ab6943ef730d26b7ffcd6d7"
LEGACY_FILE_DIGEST = "bdc1993c990b384d1856632d6b1b0dc6a14e114a295da41fd0c4bfaa94a1a058"

POLICY: dict[str, Any] = {
    "kind": "policy",
    "instruction": "put the cream cheese in the bowl",
    "until": {"goal": 0, "steps": None},
    "max_steps": 300,
    "segment": None,
    "steps": None,
}


def motion(primitive: str, steps: int, axis: str | None = None, amp: float | None = None) -> Any:
    segment = {
        "primitive": primitive,
        "axis": axis,
        "amplitude_m": amp,
        "duration_steps": steps,
        "period_steps": None,
    }
    return {
        **POLICY,
        "kind": "motion",
        "instruction": None,
        "until": None,
        "max_steps": None,
        "segment": segment,
    }


def program(*steps: Any) -> JobProgram:
    return JobProgram.model_validate(
        {"name": "cheese-then-plate", "purpose": "test", "steps": list(steps), "notes": ""}
    )


def state(**update: Any) -> RequestState:
    base: dict[str, Any] = {
        "request_text": "the cheese in the bowl, the bowl on the plate",
        "requested_goals": [
            {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
            {"predicate": "On", "objects": ["akita_black_bowl_1", "plate_1"]},
        ],
        "base_task": 6,
        "order_rules": [{"first": 0, "then": 1}],
        "budget": {
            "starts_per_request": 20,
            "requests_per_session": 4,
            "sessions": 6,
            "starts_per_investigation": 400,
        },
        "acceptance": None,
        "message_log": [],
    }
    return RequestState.model_validate({**base, **update})


def test_the_seed_limits_agree_with_the_identity_contract() -> None:
    assert SCENE_SEED_LIMIT == identity.SCENE_SEED_LIMIT
    assert NOISE_SEED_LIMIT == identity.NOISE_SEED_LIMIT


def test_the_tool_server_may_import_the_job_contracts() -> None:
    # Nothing that holds truth, and nothing the tool server's bundle does not hold already.
    closure = import_closure("fleetlab.contracts.jobs")
    assert closure_problems(closure) == []
    assert set(closure) - {"fleetlab.contracts.jobs"} <= set(import_closure())


def test_the_task_numbers_are_the_goal_suite_s_tasks_info_order() -> None:
    info = repo_root() / "third_party/libero/libero/libero/bddl_files/libero_goal/tasks_info.txt"
    stems = [line.rsplit("/", 1)[-1].removesuffix(".bddl") for line in info.read_text().split()]
    assert tuple(stems) == GOAL_SUITE_TASKS
    assert GOAL_SUITE_TASKS[6] == "put_the_cream_cheese_in_the_bowl"
    assert GOAL_SUITE_TASKS[4] == "put_the_bowl_on_the_plate"


def test_a_step_takes_the_fields_of_its_kind_and_null_for_the_rest() -> None:
    JobStep.model_validate(POLICY)
    JobStep.model_validate(motion("ramp", 20, "z", 0.05))
    with pytest.raises(ValidationError, match="takes no segment"):
        JobStep.model_validate({**POLICY, "segment": motion("hold", 5)["segment"]})
    with pytest.raises(ValidationError, match="needs instruction"):
        JobStep.model_validate({**POLICY, "instruction": None})
    with pytest.raises(ValidationError, match="one; the other null"):
        JobStep.model_validate({**POLICY, "until": {"goal": 0, "steps": 10}})
    with pytest.raises(ValidationError, match="exceeds max_steps"):
        JobStep.model_validate({**POLICY, "until": {"goal": None, "steps": 50}, "max_steps": 40})
    with pytest.raises(ValidationError):
        JobStep.model_validate({**POLICY, "max_steps": 301})
    wait = {**motion("hold", 1), "kind": "wait", "segment": None, "steps": 100}
    JobStep.model_validate(wait)
    with pytest.raises(ValidationError):
        JobStep.model_validate({**wait, "steps": 101})


def test_a_motion_step_keeps_the_probe_bounds() -> None:
    with pytest.raises(ValidationError, match=r"at most 0\.05"):
        JobStep.model_validate(motion("step", 10, "z", 0.08))
    with pytest.raises(ValidationError, match=r"m/s"):  # 5 cm in 2 steps: 0.5 m/s is the limit
        JobStep.model_validate(motion("ramp", 1, "z", 0.05))
    JobStep.model_validate(motion("ramp", 2, "z", 0.05))


def test_on_timeout_is_stop_when_absent_and_part_of_the_digest() -> None:
    absent = JobStep.model_validate(POLICY)
    assert absent.on_timeout is OnTimeout.stop
    explicit = JobStep.model_validate({**POLICY, "on_timeout": "stop"})
    retry = JobStep.model_validate({**POLICY, "on_timeout": "continue"})
    assert program(absent).digest == program(explicit).digest
    assert program(retry).digest != program(explicit).digest
    with pytest.raises(ValidationError, match="takes no on_timeout"):
        JobStep.model_validate({**motion("hold", 5), "on_timeout": "continue"})


def test_the_digest_is_sha256_over_the_canonical_dump() -> None:
    import hashlib

    from fleetlab.contracts.hashing import canonical_json

    p = program(POLICY, motion("open_gripper", 10))
    expected = hashlib.sha256(canonical_json(p.model_dump(mode="json")).encode()).hexdigest()
    assert p.digest == expected and len(p.digest) == 64
    renamed = p.model_copy(update={"notes": "another note"})
    assert renamed.digest != p.digest
    assert JobProgram.model_validate_json(p.model_dump_json()).digest == p.digest


def test_a_program_has_one_to_eight_steps() -> None:
    with pytest.raises(ValidationError):
        program()
    program(*[POLICY] * 8)
    with pytest.raises(ValidationError):
        program(*[POLICY] * 9)


def test_a_request_refuses_release_starts() -> None:
    p = program(POLICY)
    JobRequest(program=p, starts=development_starts(6, 20, 10), record_video=False)
    with pytest.raises(ValidationError, match="reserved for release runs"):
        JobRequest(program=p, starts=[start_identity(6, 30)], record_video=False)
    with pytest.raises(ValidationError, match="reserved for release runs"):
        JobRequest.model_validate(
            {
                "program": p.model_dump(),
                "starts": [start_identity(6, 49).model_dump()],
                "record_video": True,
            }
        )
    with pytest.raises(ValidationError, match="each start once"):
        JobRequest(program=p, starts=[start_identity(6, 1)] * 2, record_video=False)
    with pytest.raises(ValidationError, match="one base task"):
        JobRequest(
            program=p, starts=[start_identity(6, 1), start_identity(4, 2)], record_video=False
        )
    with pytest.raises(ValidationError):
        StartIdentity(base_task=6, init_state=50, scene_seed=0, noise_seed=0)


def test_starts_are_allocated_disjointly_with_the_spike_s_seeds() -> None:
    from fleetlab.engine import manifests, software
    from fleetlab.policies.xvla import nominal_spec

    dev = development_starts(6, 30)
    release = release_starts(6, 20)
    assert [s.init_state for s in dev] == list(range(30))
    assert [s.init_state for s in release] == list(range(30, 50))
    assert all(s.is_release for s in release) and not any(s.is_release for s in dev)
    with pytest.raises(ValueError):
        development_starts(6, 2, 29)
    with pytest.raises(ValueError):
        release_starts(6, 21)
    for start in (0, 13, 29):
        # The spike's runner (spikes/jobs/runner.py, start_identity) draws them so.
        spike = manifests.identity(
            world_seed=20261020,
            episode_number=start + 1,
            robot=1,
            site=1,
            day=0,
            episode_index=start,
            suite="libero_goal",
            task_index=6,
            start_state=start,
            policy=nominal_spec("fp16_res32"),
            batch_size=1,
            software=software.current("none"),
            max_steps=700,
        )
        mine = start_identity(6, start)
        assert (mine.scene_seed, mine.noise_seed) == (spike.scene_seed, spike.noise_seed)
    assert start_identity(4, 3).scene_seed == start_identity(6, 3).scene_seed


def test_goals_are_checked_for_arity_and_rules_for_their_goals() -> None:
    RequestedGoal.model_validate({"predicate": "Turnon", "objects": ["flat_stove_1"]})
    with pytest.raises(ValidationError, match="takes 2"):
        RequestedGoal.model_validate({"predicate": "On", "objects": ["plate_1"]})
    with pytest.raises(ValidationError, match="takes 1"):
        RequestedGoal.model_validate({"predicate": "Open", "objects": ["a", "b"]})
    with pytest.raises(ValidationError, match="beyond the 2 requested"):
        state(order_rules=[{"first": 0, "then": 2}])
    with pytest.raises(ValidationError, match="two different goals"):
        OrderRule(first=1, then=1)
    with pytest.raises(ValidationError):
        JobBudget(
            starts_per_request=21, requests_per_session=4, sessions=6, starts_per_investigation=400
        )
    assert program_problems(program({**POLICY, "until": {"goal": 2, "steps": None}}), state())


def test_an_investigation_has_at_most_twelve_sessions_and_six_stay_valid() -> None:
    def budget(sessions: int) -> JobBudget:
        return JobBudget(
            starts_per_request=20,
            requests_per_session=4,
            sessions=sessions,
            starts_per_investigation=400,
        )

    assert budget(12).sessions == 12
    assert state().budget.sessions == 6  # the requests written so far
    for name in ("job-one.json", "job-two.json"):
        RequestState.model_validate_json((FIXTURES / name).read_text())
    with pytest.raises(ValidationError):
        budget(13)


def test_a_legacy_saved_job_keeps_its_bytes_and_its_digests() -> None:
    from fleetlab.jobs.library import digest_of

    text = LEGACY_JOB.read_text()
    job = SavedJob.model_validate_json(text)
    assert job.allocation_profile is None and job.program.actions_per_chunk is None
    dumped = job.model_dump(mode="json")
    # The library's own file format (`library._save`), so the same bytes.
    assert json.dumps(dumped, indent=1, sort_keys=True) + "\n" == text
    assert digest_of(dumped) == LEGACY_FILE_DIGEST
    assert job.receipt is not None
    assert job.program.digest == job.receipt.program_digest == LEGACY_PROGRAM_DIGEST
    assert "allocation_profile" not in job.model_dump_json()
    assert "actions_per_chunk" not in job.model_dump_json()


def test_actions_per_chunk_is_left_out_when_null_and_another_program_when_set() -> None:
    legacy = json.loads(LEGACY_JOB.read_text())["program"]
    absent = JobProgram.model_validate(legacy)
    null = JobProgram.model_validate({**legacy, "actions_per_chunk": None})
    whole = JobProgram.model_validate({**legacy, "actions_per_chunk": 30})
    assert absent.actions_per_chunk is None and null.actions_per_chunk is None
    assert absent.digest == null.digest == LEGACY_PROGRAM_DIGEST
    assert "actions_per_chunk" not in null.model_dump(mode="json")
    # The whole chunk named explicitly is another program text, and so another digest,
    # though a runner plays it as the legacy program.
    assert whole.model_dump(mode="json")["actions_per_chunk"] == 30
    assert whole.digest != LEGACY_PROGRAM_DIGEST
    assert JobProgram.model_validate_json(whole.model_dump_json()).digest == whole.digest
    assert JobProgram.model_validate({**legacy, "actions_per_chunk": 5}).actions_per_chunk == 5
    for outside in (0, 4, 31):
        with pytest.raises(ValidationError):
            JobProgram.model_validate({**legacy, "actions_per_chunk": outside})
    schema = json_schema_strict(JobProgram)
    assert "actions_per_chunk" in schema["required"]
    assert {"type": "null"} in schema["properties"]["actions_per_chunk"]["anyOf"]


def test_the_allocation_profile_is_left_out_when_null_and_one_of_the_host_s() -> None:
    for name in ("job-one.json", "job-two.json"):
        text = (FIXTURES / name).read_text()
        legacy = RequestState.model_validate_json(text)
        assert legacy.allocation_profile is None
        assert legacy.model_dump(mode="json") == json.loads(text)
    pooled = state(allocation_profile="pool-v1")
    assert pooled.model_dump(mode="json")["allocation_profile"] == "pool-v1"
    assert RequestState.model_validate_json(pooled.model_dump_json()) == pooled
    with pytest.raises(ValidationError):
        state(allocation_profile="pool-v2")
    saved = json.loads(LEGACY_JOB.read_text())
    job = SavedJob.model_validate({**saved, "allocation_profile": "pool-v1"})
    assert job.model_dump(mode="json")["allocation_profile"] == "pool-v1"
    with pytest.raises(ValidationError):
        SavedJob.model_validate({**saved, "allocation_profile": "legacy"})
    for model in (RequestState, SavedJob):
        assert "allocation_profile" in json_schema_strict(model)["required"]


def test_the_order_rule_reads_the_goals_first_times() -> None:
    rule = [OrderRule(first=0, then=1)]
    assert order_verdicts(rule, [92, 167])[0].kept
    assert order_verdicts(rule, [92, 92])[0].kept  # at the same step is not before
    assert not order_verdicts(rule, [92, 40])[0].kept
    assert not order_verdicts(rule, [None, 40])[0].kept
    assert order_verdicts(rule, [92, None])[0].kept
    assert order_verdicts(rule, [None, None])[0].kept


def test_the_lead_approves_twenty_starts_and_four_fifths() -> None:
    assert acceptance_problems(Acceptance(N=20, K=16)) == []
    assert acceptance_problems(Acceptance(N=20, K=20)) == []
    assert len(acceptance_problems(Acceptance(N=10, K=8))) == 1  # too few starts
    assert len(acceptance_problems(Acceptance(N=20, K=15))) == 1  # too few deliveries
    with pytest.raises(ValidationError):
        Acceptance(N=21, K=20)  # more than the release starts
    with pytest.raises(ValidationError):
        Acceptance(N=20, K=21)


def test_a_check_declares_exactly_the_parameters_its_sql_uses() -> None:
    sql = "SELECT * FROM job_episodes WHERE job_id = $job AND init_state < $below"
    params = [{"name": "job", "type": "str"}, {"name": "below", "type": "int"}]

    def spec(parameters: Any, text: str = sql) -> CheckSpec:
        return CheckSpec.model_validate(
            {
                "name": "early-starts",
                "purpose": "p",
                "parameters": parameters,
                "sql": text,
                "expects": "e",
            }
        )

    spec(params)
    with pytest.raises(ValidationError, match="undeclared"):
        spec(params[:1])
    with pytest.raises(ValidationError, match="never uses"):
        spec(params, "SELECT 1")


@pytest.mark.parametrize(
    "model",
    [
        JobProgram,
        JobRequest,
        CheckSpec,
        Acceptance,
        RequestState,
        JobEpisode,
        ReleaseReceipt,
        SavedJob,
    ],
)
def test_the_models_compile_to_strict_mode(model: Any) -> None:
    schema = json_schema_strict(model)
    assert schema["type"] == "object"


def test_a_home_step_takes_a_budget_and_on_timeout() -> None:
    home = {
        **POLICY,
        "kind": "home",
        "instruction": None,
        "until": None,
        "max_steps": None,
        "duration_steps": 120,
    }
    step = JobStep.model_validate(home)
    assert step.on_timeout is OnTimeout.stop and step.budget == 120
    retry = JobStep.model_validate({**home, "on_timeout": "continue"})
    assert retry.on_timeout is OnTimeout.continue_
    with pytest.raises(ValidationError, match="needs duration_steps"):
        JobStep.model_validate({**home, "duration_steps": None})
    with pytest.raises(ValidationError):
        JobStep.model_validate({**home, "duration_steps": 201})
    with pytest.raises(ValidationError, match="takes no duration_steps"):
        JobStep.model_validate({**POLICY, "duration_steps": 10})
    with pytest.raises(ValidationError, match="takes no instruction"):
        JobStep.model_validate({**home, "instruction": "go home"})
    # Steps of the other kinds dump without the field: their programs keep their digests.
    assert "duration_steps" not in JobStep.model_validate(POLICY).model_dump(mode="json")
    assert step.model_dump(mode="json")["duration_steps"] == 120
    assert program(POLICY, home).digest != program(POLICY).digest
    items = json_schema_strict(JobProgram)["properties"]["steps"]["items"]
    assert "duration_steps" in items["required"]


def _receipt(**update: Any) -> dict[str, Any]:
    """A consistent receipt: 12 of 20 delivered against K 16, 7 for the naive chain."""
    starts = release_starts(6, 20)
    delivered = [True] * 12 + [False] * 8
    naive = [True] * 6 + [False] * 13 + [True]
    pairs = [
        {
            "start": s.model_dump(),
            "episode_id": f"e{101 + i:06d}",
            "delivered": d,
            "order_kept": True,
            "naive_episode_id": f"e{201 + i:06d}",
            "naive_delivered": q,
        }
        for i, (s, d, q) in enumerate(zip(starts, delivered, naive, strict=True))
    ]
    base: dict[str, Any] = {
        "job_name": "cheese-then-plate",
        "program_digest": "a" * 64,
        "naive_digest": "b" * 64,
        "release_starts": [s.model_dump() for s in starts],
        "N": 20,
        "K": 16,
        "delivered": 12,
        "naive_delivered": 7,
        "rescued": 6,
        "broken": 1,
        "order_violations": 0,
        "passed": False,
        "released": False,
        "pairs": pairs,
        "run_dir": "x",
        "starts_fresh": True,
    }
    return {**base, **update}


def test_a_receipt_s_qualification_follows_from_its_counts() -> None:
    ReleaseReceipt.model_validate(_receipt())
    with pytest.raises(ValidationError, match="at least K of the N"):
        ReleaseReceipt.model_validate(_receipt(passed=True))
    with pytest.raises(ValidationError, match="released needs"):
        ReleaseReceipt.model_validate(_receipt(released=True))
    with pytest.raises(ValidationError, match="passed needs the approved K"):
        ReleaseReceipt.model_validate(_receipt(K=None))
    ReleaseReceipt.model_validate(_receipt(K=None, passed=None))


def test_a_receipt_releases_only_an_approvable_condition_on_fresh_starts() -> None:
    all_delivered = _receipt()
    for pair in all_delivered["pairs"]:
        pair["delivered"] = True
    all_delivered.update(delivered=20, rescued=13, broken=0, passed=True, released=True)
    ReleaseReceipt.model_validate(all_delivered)
    with pytest.raises(ValidationError, match="released needs"):
        ReleaseReceipt.model_validate({**all_delivered, "starts_fresh": False})
    with pytest.raises(ValidationError, match="released needs"):
        ReleaseReceipt.model_validate({**all_delivered, "starts_fresh": None})
    one = {**_receipt(), "N": 1, "K": 0, "release_starts": all_delivered["release_starts"][:1]}
    one["pairs"] = all_delivered["pairs"][:1]
    one.update(delivered=1, naive_delivered=1, rescued=0, broken=0, passed=True, released=True)
    with pytest.raises(ValidationError, match="not an acceptance the lead approves"):
        ReleaseReceipt.model_validate(one)


def test_a_receipt_pairs_each_release_start_once_in_order() -> None:
    repeated = _receipt()
    repeated["pairs"] = [repeated["pairs"][0]] * 20
    repeated["release_starts"] = [repeated["release_starts"][0]] * 20
    repeated.update(delivered=20, naive_delivered=20, rescued=0, broken=0)
    with pytest.raises(ValidationError, match="appears once"):
        ReleaseReceipt.model_validate(repeated)
    swapped = _receipt()
    swapped["release_starts"] = list(reversed(swapped["release_starts"]))
    with pytest.raises(ValidationError, match="in their order"):
        ReleaseReceipt.model_validate(swapped)
    twice = _receipt()
    twice["pairs"][1] = {**twice["pairs"][1], "episode_id": "e000101"}
    with pytest.raises(ValidationError, match="its own"):
        ReleaseReceipt.model_validate(twice)
    development = _receipt()
    development["release_starts"][0] = start_identity(6, 3).model_dump()
    development["pairs"][0] = {
        **development["pairs"][0],
        "start": start_identity(6, 3).model_dump(),
    }
    with pytest.raises(ValidationError, match="reserved release starts"):
        ReleaseReceipt.model_validate(development)
