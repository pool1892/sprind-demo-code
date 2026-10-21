"""Numbered investigations of a job line: fresh pools under the host's
allocation profile, the exposure of evaluation starts, the host's pool assignment, and the
continuation requests. No simulator: the toy runner, a scripted agent and the real tool
server, as in tests/test_jobs_tools_driver.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fleetlab.agents.sessions import AgentSpec
from fleetlab.agents.toolenv import project_tool_env
from fleetlab.contracts.jobs import RequestState, SavedJob
from fleetlab.jobs import library
from fleetlab.jobs.driver import Investigation, ToyRunner, investigate, toy_agent, toy_skills
from fleetlab.jobs.prior import ASSIGNMENT_FORMAT, PreflightError
from fleetlab.jobs.starts import (
    JOB_SEED,
    release_starts_for,
    seed_origin,
    start_for,
    start_identity,
    starts_for,
)
from fleetlab.runtime import repo_root
from fleetlab.toolserver.jobs import lead_reply

FIXTURES = repo_root() / "tests" / "fixtures" / "jobs" / "continuation"

REQUEST: dict[str, Any] = {
    "request_text": "The cream cheese into the bowl, then the bowl onto the plate.",
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
# A replacement job on its own pool under pool-v1, with the continuation's budget.
POOLED: dict[str, Any] = {
    **REQUEST,
    "base_task": 2,
    "allocation_profile": "pool-v1",
    "budget": {**REQUEST["budget"], "sessions": 12},
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    return project_tool_env(tmp_path / "env")


def write_request(path: Path, data: dict[str, Any]) -> tuple[Path, RequestState]:
    path.write_text(json.dumps(data))
    return path, RequestState.model_validate(data)


def how(state: RequestState, root: Path, env: Any, k: int = 16, **over: Any) -> Investigation:
    return Investigation(
        agent=AgentSpec("codex", "scripted"),
        runner=ToyRunner(state),
        scripted=toy_agent(root / "scripted", state, k=k),
        placeholder_task=True,
        env=env,
        policy_skills=toy_skills(),
        **over,
    )


def assignment(path: Path, request: Path | None, base_task: int = 2, **entry: Any) -> Path:
    """The coordinator's assignment record (`fleetlab.coordinator-cohorts/1`)."""
    digest = hashlib.sha256(request.read_bytes()).hexdigest() if request else None
    record = {
        "format": ASSIGNMENT_FORMAT,
        "historical": [
            {"base_task": 6, "allocation_profile": None, "status": "evaluation-exposed"}
        ],
        "assignments": [
            {
                "line": "replacement",
                "cohort": "replacement-01",
                "base_task": base_task,
                "allocation_profile": "pool-v1",
                "status": "reserved",
                "request_sha256": digest,
                **entry,
            },
            {
                "line": "cabinet",
                "cohort": "cabinet-01",
                "base_task": 0,
                "allocation_profile": "pool-v1",
                "status": "reserved",
                "request_sha256": None,
            },
        ],
    }
    path.write_text(json.dumps(record))
    return path


# ---- the allocation profiles ----------------------------------------------------------------


def test_the_legacy_profile_keeps_every_start_and_pool_v1_gives_each_pool_its_own_seeds() -> None:
    for task in range(10):
        for k in (0, 17, 30, 49):
            assert start_for(task, k, profile=None) == start_identity(task, k)
    assert seed_origin(6, None) == JOB_SEED and seed_origin(2, "pool-v1") == JOB_SEED + 3
    scene = {s.scene_seed for s in starts_for(6, range(50), profile=None)}
    noise = {s.noise_seed for s in starts_for(6, range(50), profile=None)}
    for task in range(10):
        pooled = starts_for(task, range(50), profile="pool-v1")
        assert len({s.scene_seed for s in pooled}) == len({s.noise_seed for s in pooled}) == 50
        assert not {s.scene_seed for s in pooled} & scene
        assert not {s.noise_seed for s in pooled} & noise
        scene |= {s.scene_seed for s in pooled}
        noise |= {s.noise_seed for s in pooled}
    with pytest.raises(ValueError, match="unknown allocation profile"):
        start_for(2, 3, profile="pool-v2")


def test_the_legacy_rule_gives_the_identities_job_one_s_release_recorded() -> None:
    # Exposure lines written before the identities were recorded are read under the legacy
    # profile; job one's real receipt (job-one/run-6) recorded the identities it ran from.
    saved = SavedJob.model_validate_json((FIXTURES / "legacy-saved-job.json").read_text())
    assert saved.receipt is not None
    assert saved.receipt.release_starts == starts_for(6, range(30, 50), profile=None)
    legacy_line = {"event": "started", "base_task": 6, "init_states": list(range(30, 50))}
    assert library.exposure_identities(legacy_line) == saved.receipt.release_starts


def test_the_continuation_requests_ask_for_nine_in_ten_and_force_no_save() -> None:
    for name, pool in (("cabinet-01.json", 0), ("plate-04.json", 1)):
        text = (FIXTURES / name).read_text()
        state = RequestState.model_validate_json(text)
        assert (state.base_task, state.allocation_profile) == (pool, "pool-v1")
        assert state.budget.sessions == 12 and state.budget.starts_per_investigation == 400
        assert state.acceptance is None and state.message_log == []
        assert "at least 18 deliveries in a 20-start evaluation" in state.request_text
        assert "Do not save merely because one development batch" in state.request_text
        assert "save it: the reserved starts decide" not in state.request_text  # the old clause
    # The lead launched at nine in ten: 18 of 20 approved, 17 refused.
    assert lead_reply(20, 18, 20, (9, 10))[0] and not lead_reply(20, 17, 20, (9, 10))[0]


# ---- a pooled investigation -----------------------------------------------------------------


def test_a_pooled_investigation_runs_on_its_pool_with_twelve_sessions(
    tmp_path: Path, env: Any
) -> None:
    request, state = write_request(tmp_path / "pooled.json", POOLED)
    record_file = assignment(tmp_path / "cohorts.json", request)
    plan = how(
        state,
        tmp_path,
        env,
        k=18,
        lead_fraction=(9, 10),
        assignment=record_file,
        cohort="replacement-01",
    )
    out = tmp_path / "replacement-01"
    record = investigate(request, out, plan)
    assert record["outcome"] == "done", record.get("ended_because")
    assert record["max_sessions"] == 12 and record["allocation_profile"] == "pool-v1"
    checked = record["preflight"]
    assert checked["assignment"]["cohort"] == "replacement-01"
    assert checked["evaluation_starts"] == list(range(30, 50))
    # Development starts and the release come from pool 2 under pool-v1.
    batch = json.loads((out / "runs" / "session-01" / "batch.json").read_text())
    starts = batch["jobs"][0]["request"]["starts"]
    pooled = starts_for(2, range(10), profile="pool-v1")
    assert starts == [s.model_dump(mode="json") for s in pooled]
    assert batch["state"]["allocation_profile"] == "pool-v1"
    receipt = record["release"]["receipt"]
    assert receipt["release_starts"] == [
        s.model_dump(mode="json") for s in release_starts_for(2, 20, profile="pool-v1")
    ]
    assert (receipt["K"], receipt["delivered"], receipt["released"]) == (18, 18, True)
    saved = SavedJob.model_validate_json((out / "library/jobs/loaded-carry.json").read_text())
    assert (saved.base_task, saved.allocation_profile) == (2, "pool-v1")
    # The attempt's exposure names its full identities, written before it ran.
    [started] = [e for e in library.release_exposure(out / "library") if e["event"] == "started"]
    assert started["allocation_profile"] == "pool-v1"
    assert library.exposure_identities(started) == release_starts_for(2, 20, profile="pool-v1")
    # The agent never sees the profile.
    mounted = (Path(record["analysis"]) / "results" / "operator_state.json").read_text()
    assert "allocation_profile" not in mounted and "pool-v1" not in mounted


def test_a_legacy_request_keeps_its_starts_and_needs_no_assignment(
    tmp_path: Path, env: Any
) -> None:
    request, state = write_request(tmp_path / "legacy.json", REQUEST)
    out = tmp_path / "legacy"
    record = investigate(request, out, how(state, tmp_path, env))
    assert record["outcome"] == "done" and record["allocation_profile"] is None
    assert record["preflight"]["assignment"] is None and record["max_sessions"] == 6
    batch = json.loads((out / "runs" / "session-01" / "batch.json").read_text())
    assert "allocation_profile" not in batch["state"]
    starts = batch["jobs"][0]["request"]["starts"]
    assert starts == [start_identity(6, k).model_dump(mode="json") for k in range(10)]
    saved = json.loads((out / "library/jobs/loaded-carry.json").read_text())
    assert "allocation_profile" not in saved
    # A later request evaluated on the same pool is refused before anything runs: its
    # evaluation starts were examined, for another request's goals as much as for these.
    other = {**REQUEST, "request_text": "Another job on the same starts."}
    again, again_state = write_request(tmp_path / "again.json", other)
    with pytest.raises(PreflightError, match="were examined before"):
        investigate(
            again,
            tmp_path / "again",
            how(again_state, tmp_path / "b", env, library=out / "library"),
        )
    assert not (tmp_path / "again").exists()


# ---- the preflight --------------------------------------------------------------------------


def refused(tmp_path: Path, env: Any, data: dict[str, Any], name: str, **over: Any) -> str:
    request, state = write_request(tmp_path / f"{name}.json", data)
    out = tmp_path / name
    with pytest.raises(PreflightError) as caught:
        investigate(request, out, how(state, tmp_path / f"s-{name}", env, **over))
    assert not out.exists()  # nothing written, nothing run
    return str(caught.value)


def test_the_preflight_refuses_a_pool_the_assignment_does_not_give(
    tmp_path: Path, env: Any
) -> None:
    assert "assignment record" in refused(tmp_path, env, POOLED, "none")
    request = tmp_path / "pooled.json"
    request.write_text(json.dumps(POOLED))
    wrong = assignment(tmp_path / "wrong-cohorts.json", request, base_task=3)
    assert "holds pool 3" in refused(
        tmp_path, env, POOLED, "wrong", assignment=wrong, cohort="replacement-01"
    )
    other = assignment(tmp_path / "other-cohorts.json", None, request_sha256="0" * 64)
    assert "not the one assigned" in refused(
        tmp_path, env, POOLED, "other", assignment=other, cohort="replacement-01"
    )
    shared = assignment(tmp_path / "shared-cohorts.json", None, base_task=0)
    on_zero = {**POOLED, "base_task": 0}
    assert "assigned to line 'replacement' as well" in refused(
        tmp_path, env, on_zero, "shared", assignment=shared, cohort="cabinet-01"
    )
    retired = assignment(tmp_path / "retired-cohorts.json", None, status="evaluation-exposed")
    assert "not one of" in refused(
        tmp_path, env, POOLED, "retired", assignment=retired, cohort="replacement-01"
    )
    on_six = {**POOLED, "base_task": 6}
    historical = assignment(tmp_path / "historical-cohorts.json", None, base_task=6)
    assert "pool 6 is historical" in refused(
        tmp_path, env, on_six, "historical", assignment=historical, cohort="replacement-01"
    )


def _saved(base_task: int, profile: str | None) -> SavedJob:
    return SavedJob.model_validate(
        {
            "name": "earlier-job",
            "program": {
                "name": "earlier",
                "purpose": "p",
                "steps": [
                    {
                        "kind": "wait",
                        "instruction": None,
                        "until": None,
                        "max_steps": None,
                        "segment": None,
                        "steps": 5,
                    }
                ],
                "notes": "",
            },
            "requested_goals": REQUEST["requested_goals"],
            "base_task": base_task,
            "order_rules": [{"first": 0, "then": 1}],
            "acceptance": None,
            "saved_from": [],
            "receipt": None,
            "allocation_profile": profile,
        }
    )


def test_an_interrupted_attempt_exposes_its_starts_and_a_pool_has_one_mapping(
    tmp_path: Path, env: Any
) -> None:
    # An attempt that started on pool 2 and never finished (no "finished" line).
    lib = tmp_path / "earlier-library"
    job = _saved(2, "pool-v1")
    library.record_release_exposure(
        lib, job, release_starts_for(2, 5, profile="pool-v1"), tmp_path / "killed"
    )
    request = tmp_path / "pooled.json"
    request.write_text(json.dumps(POOLED))
    cohorts = assignment(tmp_path / "cohorts.json", request)
    text = refused(
        tmp_path, env, POOLED, "pooled", assignment=cohorts, cohort="replacement-01", library=lib
    )
    assert "evaluation starts [30, 31, 32, 33, 34] of pool 2 were examined before" in text
    # Pool 6's starts were examined under the legacy profile: pool-v1 would give the same
    # public starts other seeds, which the history cannot tell apart.
    legacy_lib = tmp_path / "legacy-library"
    library.record_release_exposure(
        legacy_lib, _saved(6, None), release_starts_for(6, 20, profile=None), tmp_path / "old"
    )
    on_six = {**POOLED, "base_task": 6}
    six = tmp_path / "six.json"
    six.write_text(json.dumps(on_six))
    unhistoric = assignment(tmp_path / "six-cohorts.json", six, base_task=6)
    data = json.loads(unhistoric.read_text())
    data["historical"] = []
    unhistoric.write_text(json.dumps(data))
    text = refused(
        tmp_path,
        env,
        on_six,
        "six",
        assignment=unhistoric,
        cohort="replacement-01",
        library=legacy_lib,
    )
    assert "start 6/30 means two identities" in text


# ---- imported history: three earlier investigations and a numbered continuation ------------


def _data(result: Any) -> Any:
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    if result.is_error:
        raise AssertionError(f"tool error: {text}")
    return json.loads(text)


def _error(result: Any) -> str:
    assert result.is_error, result.content
    return "\n".join(p["text"] for p in result.content if p.get("type") == "text")


CHECK: dict[str, Any] = {
    "name": "goal-undone",
    "purpose": "episodes in which a requested goal held and was lost by the end",
    "parameters": [{"name": "goal", "type": "int"}],
    "sql": "SELECT job_id, count(*) AS undone FROM job_episodes WHERE NOT "
    "goals_held_at_end[$goal + 1] GROUP BY job_id ORDER BY job_id",
    "expects": "no rows once the transition keeps the goal; the verdict is the count",
}
GOAL0 = [{"name": "goal", "value": "0"}]


def _approve(box: Any, k: int = 18) -> None:
    assert _data(box.call("submit_acceptance_proposal", {"N": 20, "K": k, "reason": "need"}))[
        "approved"
    ]


def _submit(box: Any, program: dict[str, Any], starts: list[int]) -> str:
    receipt = _data(
        box.call("submit_job_program", {"program": program, "starts": starts, "record_video": True})
    )
    assert receipt["accepted"], receipt
    return str(receipt["job_id"])


def _programs(state: RequestState) -> dict[str, dict[str, Any]]:
    from fleetlab.jobs.driver import toy_programs

    return toy_programs(state)


def _source_one_steps(state: RequestState) -> list[Any]:
    """An earlier plate investigation whose final evaluation fails: the naive chain saved."""
    programs = _programs(state)

    def first(box: Any) -> str:
        _approve(box)
        assert _submit(box, programs["naive"], [0, 1, 2, 3, 4]) == "j0001"
        assert _submit(box, programs["release"], [0, 1, 2, 3, 4]) == "j0002"
        box.call("memory_write", {"notes": "naive j0001; the tables say the verdict is mixed"})
        return "waiting"

    def second(box: Any) -> str:
        written = _data(box.call("write_check", {"check": CHECK, "check_with": GOAL0}))
        assert written["accepted"], written
        _data(box.call("run_check", {"name": "goal-undone", "job_id": "j0001", "params": GOAL0}))
        saved = _data(
            box.call(
                "save_job",
                {"name": "plate-naive", "job_id": "j0001", "purpose": "p", "check": "goal-undone"},
            )
        )
        assert saved["accepted"], saved
        return "waiting"

    def third(box: Any) -> str:
        account = "Released plate-naive; the evaluation verdict: it missed the condition."
        done = _data(box.call("submit_job_account", {"job": "plate-naive", "account": account}))
        assert done["accepted"], done
        return "done"

    return [first, second, third]


def _source_two_steps(state: RequestState) -> list[Any]:
    """An earlier investigation whose first batch is broken off after its first job."""
    programs = _programs(state)

    def first(box: Any) -> str:
        _approve(box)
        _submit(box, programs["naive"], [0, 1, 2])
        _submit(box, programs["release"], [0, 1, 2])
        return "waiting"

    return [first]


def _source_three_steps(state: RequestState) -> list[Any]:
    """An earlier investigation of another request whose release is broken off."""
    programs = _programs(state)

    def first(box: Any) -> str:
        _approve(box)
        _submit(box, programs["release"], [0, 1, 2])
        return "waiting"

    def second(box: Any) -> str:
        _data(box.call("run_check", {"name": "goal-undone", "job_id": "j0001", "params": GOAL0}))
        saved = _data(
            box.call(
                "save_job",
                {
                    "name": "stove-release",
                    "job_id": "j0001",
                    "purpose": "p",
                    "check": "goal-undone",
                },
            )
        )
        assert saved["accepted"], saved
        return "waiting"

    return [first, second]


class _FirstJobThenInterrupted(ToyRunner):
    """The toy runner, stopped by a signal after the batch's first job finished."""

    def run(self, batch: Any, out: Path) -> Any:
        import dataclasses

        from fleetlab.jobs.driver import Interrupted

        super().run(dataclasses.replace(batch, jobs=batch.jobs[:1]), out)
        raise Interrupted("signal 15")


class _ReleaseInterrupted(ToyRunner):
    """The toy runner, stopped by a signal once the release attempt is recorded."""

    def release(self, job: str, library_root: Path, n: int, k: int | None, out: Path) -> Any:
        from fleetlab.jobs.driver import Interrupted, load_saved_job

        saved = load_saved_job(library_root, job)
        starts = release_starts_for(saved.base_task, n, profile=saved.allocation_profile)
        out.mkdir(parents=True)
        library.record_release_exposure(library_root, saved, starts, out / "run")
        raise Interrupted("signal 15")


def _scripted(root: Path, steps: list[Any]) -> Any:
    from fleetlab.loop.scripted import ScriptedAgent

    return ScriptedAgent(steps, run_root=root)


def _investigation(
    tmp_path: Path,
    env: Any,
    name: str,
    data: dict[str, Any],
    steps: Any,
    runner: Any = None,
    **over: Any,
) -> tuple[Path, dict[str, Any] | None]:
    from fleetlab.jobs.driver import Interrupted

    request, state = write_request(tmp_path / f"{name}.json", data)
    plan = Investigation(
        agent=AgentSpec("codex", "scripted"),
        runner=runner(state, videos=True) if runner else ToyRunner(state, videos=True),
        scripted=_scripted(tmp_path / f"{name}-agent", steps(state)),
        placeholder_task=True,
        env=env,
        policy_skills=toy_skills(),
        lead_fraction=(9, 10),
        video_all=True,
        **over,
    )
    out = tmp_path / name
    try:
        return out, investigate(request, out, plan)
    except Interrupted:
        return out, None


def _tree_digests(*roots: Path) -> dict[str, str]:
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _break_order(run_dir: Path, job_id: str) -> None:
    """A recorded run of `job_id` that broke the order rule (its first episode's verdict)."""
    import pyarrow.parquet as pq

    path = run_dir / "log" / "job_episodes.parquet"
    rows = pq.read_table(path).to_pylist()
    first = next(r for r in rows if r["job_id"] == job_id)
    first["order_kept"] = False
    pq.write_table(__import__("pyarrow").Table.from_pylist(rows, schema=pq.read_schema(path)), path)


def test_a_numbered_investigation_continues_from_its_predecessors_public_records(
    tmp_path: Path, env: Any
) -> None:
    from fleetlab.toolserver.jobs import HISTORY_FILE

    # Three earlier investigations: a failed final evaluation; a batch broken off after its
    # first job; another request's investigation whose release was broken off.
    one, one_record = _investigation(tmp_path, env, "plate-01", REQUEST, _source_one_steps)
    assert one_record is not None and one_record["outcome"] == "done"
    assert one_record["release"]["receipt"]["delivered"] == 6  # of 20: the condition is missed
    _break_order(one / "runs" / "session-01" / "run", "j0002")  # a violation of that program
    two_request = {**REQUEST, "base_task": 4}
    two, two_record = _investigation(
        tmp_path, env, "plate-02", two_request, _source_two_steps, _FirstJobThenInterrupted
    )
    assert two_record is None
    stove = {
        **REQUEST,
        "base_task": 5,
        "requested_goals": [
            {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
            {"predicate": "On", "objects": ["akita_black_bowl_1", "flat_stove_1_cook_region"]},
        ],
    }

    def three_steps(state: RequestState) -> list[Any]:
        steps = _source_three_steps(state)

        def write_then(box: Any) -> str:
            assert _data(box.call("write_check", {"check": CHECK, "check_with": GOAL0}))["accepted"]
            return str(steps[0](box))

        return [write_then, steps[1]]

    three, three_record = _investigation(
        tmp_path, env, "stove-01", stove, three_steps, _ReleaseInterrupted
    )
    assert three_record is None
    _break_order(three / "runs" / "session-01" / "run", "j0001")  # the same program, elsewhere
    before = _tree_digests(one, two, three)

    # The continuation: plate's own pool under pool-v1, the earlier library mounted.
    pooled = {**POOLED, "base_task": 1}
    request = tmp_path / "plate-04.json"
    request.write_text(json.dumps(pooled))
    cohorts = assignment(tmp_path / "plate-cohorts.json", request, base_task=1, cohort="plate-04")
    programs = _programs(RequestState.model_validate(pooled))
    variant = {**programs["release"], "notes": "the release program again, a new candidate"}
    seen: dict[str, Any] = {}

    def first(box: Any) -> str:
        state = _data(box.call("operator_state", {}))
        assert state["history"]["investigations"] == 3
        assert "pool-v1" not in json.dumps(state) and "allocation_profile" not in json.dumps(state)
        results = _data(box.call("job_results", {}))
        imported = {r["job_id"]: r for r in results if r["status"] == "imported"}
        seen["imported"] = imported
        assert all(r["starts_charged"] == 0 and r["session"] == 0 for r in imported.values())
        # The same id twice in the sources (a batch's j0001 and the release's j0001), apart.
        assert (imported["j1001"]["part"], imported["j1001"]["source_job_id"]) == (
            "development",
            "j0001",
        )
        assert (imported["j1098"]["part"], imported["j1098"]["source_job_id"]) == (
            "release-candidate",
            "j0001",
        )
        assert imported["j1099"]["part"] == "release-naive"
        assert imported["j1101"]["source_status"] == "interrupted"
        assert "j1102" not in imported  # its job never finished: nothing of it is shown
        assert imported["j1201"]["requested_goals"][1].endswith("flat_stove_1_cook_region")
        assert imported["j1098"]["source_episode_ids"][0] == "e000101"
        assert imported["j1098"]["episode_ids"][0] == "e109801"
        counted = _data(
            box.call("query", {"sql": "SELECT count(*) FROM job_episodes WHERE job_id = 'j1098'"})
        )
        assert counted["rows"] == [[20]]
        frames = box.call("get_frames", {"episode_id": "e109801", "steps": [0], "camera": "scene"})
        assert not frames.is_error and any(p["type"] == "image" for p in frames.content)
        checked = _data(
            box.call("run_check", {"name": "goal-undone", "job_id": "j1098", "params": GOAL0})
        )
        assert checked["rows"] and checked["rows"][0][0] == "j1098"
        history = _data(box.call("job_library", {}))["history"]
        evaluation = history["investigations"][0]["final_evaluation"]
        assert (evaluation["delivered"], evaluation["condition_met"]) == (6, False)
        assert history["investigations"][2]["final_evaluation"] is None  # broken off unrun
        assert "verdict" in history["investigations"][0]["notes"]
        assert "plate-naive" in history["job_names"] and "goal-undone" in history["check_names"]
        refused = _data(
            box.call(
                "save_job",
                {"name": "plate-again", "job_id": "j1098", "purpose": "p", "check": None},
            )
        )
        assert not refused["accepted"] and "imported from investigation 1" in refused["problems"][0]
        _approve(box)
        seen["release"] = _submit(box, programs["release"], list(range(10)))
        seen["variant"] = _submit(box, variant, list(range(10)))
        assert (seen["release"], seen["variant"]) == ("j0001", "j0002")
        return "waiting"

    def second(box: Any) -> str:
        budget = _data(box.call("operator_state", {}))["budget"]
        assert budget["starts_used"] == 20  # this investigation's own, nothing imported
        for name, words in (
            ("plate-naive", "the installation has a job named"),  # the mounted library's
            ("stove-release", "earlier investigation used"),  # only investigation 3's
        ):
            taken = _data(
                box.call(
                    "save_job", {"name": name, "job_id": "j0002", "purpose": "p", "check": None}
                )
            )
            assert not taken["accepted"] and words in " ".join(taken["problems"]), taken
        again = _data(box.call("write_check", {"check": CHECK, "check_with": GOAL0}))
        assert not again["accepted"]
        _data(box.call("run_check", {"name": "goal-undone", "job_id": "j0001", "params": GOAL0}))
        _data(box.call("run_check", {"name": "goal-undone", "job_id": "j0002", "params": GOAL0}))
        # The release program broke the rule in investigation 1, for these goals: refused.
        # Investigation 3's violation of the same program was for another request.
        inherited = _data(
            box.call(
                "save_job",
                {
                    "name": "plate-release",
                    "job_id": "j0001",
                    "purpose": "p",
                    "check": "goal-undone",
                },
            )
        )
        problems = " ".join(inherited["problems"])
        assert not inherited["accepted"] and "e100201" in problems and "e120101" not in problems
        saved = _data(
            box.call(
                "save_job",
                {
                    "name": "plate-candidate",
                    "job_id": "j0002",
                    "purpose": "p",
                    "check": "goal-undone",
                },
            )
        )
        assert saved["accepted"], saved
        return "waiting"

    def third(box: Any) -> str:
        release = _data(box.call("operator_state", {}))["release"]
        seen["release_counts"] = release
        account = f"plate-candidate delivered {release['delivered']} of {release['N']}."
        assert _data(
            box.call("submit_job_account", {"job": "plate-candidate", "account": account})
        )["accepted"]
        return "done"

    out, record = _investigation(
        tmp_path,
        env,
        "plate-04",
        pooled,
        lambda state: [first, second, third],
        library=one / "library",
        prior_runs=(one, two, three),
        assignment=cohorts,
        cohort="plate-04",
    )
    assert record is not None and record["outcome"] == "done", record and record.get(
        "ended_because"
    )
    # Investigation 1: two batches' jobs (5 starts each) and its evaluation (20 and 20);
    # investigations 2 and 3: the one job each that finished (3 starts each).
    assert (record["prior"]["jobs"], record["prior"]["episodes"]) == (6, 56)
    # The evaluation ran on plate's pool under pool-v1, its own identities.
    receipt = record["release"]["receipt"]
    assert receipt["release_starts"] == [
        s.model_dump(mode="json") for s in release_starts_for(1, 20, profile="pool-v1")
    ]
    assert (receipt["delivered"], receipt["released"]) == (18, True)
    # Charged: this investigation's 20 starts.
    assert record["costs"]["starts_charged"] == 20
    # The sources are as they were, byte for byte.
    assert _tree_digests(one, two, three) == before
    # Nothing raw reached the agent: no run record, truth, receipt or host file in any
    # export or the results.
    analysis = Path(record["analysis"])
    for root in [analysis / "results", *(out / "exports").iterdir()]:
        names = {p.name for p in root.rglob("*")}
        assert not names & {"run.json", "episodes.jsonl", "receipt.json", "job_run.json",
                            "registry.json", "batch.json", "truth"}  # fmt: skip
        assert not list(root.rglob("*.npz"))
    assert (analysis / "results" / HISTORY_FILE).read_bytes() == (
        out / "prior" / HISTORY_FILE
    ).read_bytes()
    # The interrupted release's exposure is carried on: pool 5's evaluation starts stay
    # examined for every later request.
    carried = library.exposed_identities(out / "library")
    assert set(release_starts_for(5, 20, profile=None)) <= carried
    assert set(release_starts_for(6, 20, profile=None)) <= carried
    assert set(release_starts_for(1, 20, profile="pool-v1")) <= carried
    on_five = {**REQUEST, "base_task": 5, "request_text": "Again on pool five."}
    with pytest.raises(PreflightError, match="were examined before"):
        _investigation(tmp_path, env, "again", on_five, _source_two_steps, library=out / "library")
    seen["out"], seen["record"] = out, record
    _the_tools_refuse_history_that_does_not_fit(tmp_path, out, record)
    _the_launch_audit_accepts_the_history(tmp_path, env, out, record)


def _box(out: Path, record: dict[str, Any], export: Path) -> Any:
    from fleetlab.jobs.driver import JOB_SESSION_TOOLS
    from fleetlab.toolserver.server import Limits, Paths, ToolBox

    analysis = Path(record["analysis"])
    log = out / "probe-logs" / f"{export.name}.jsonl"
    log.parent.mkdir(exist_ok=True)
    return ToolBox(
        Paths(
            export=export,
            memory=analysis / "memory",
            outbox=analysis / "outbox",
            results=analysis / "results",
            log=log,
        ),
        JOB_SESSION_TOOLS,
        session=9,
        limits=Limits(),
    )


def _the_tools_refuse_history_that_does_not_fit(
    tmp_path: Path, out: Path, record: dict[str, Any]
) -> None:
    """A row of no request and no import, or an imported job missing a row: refused whole."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    last = Path(record["sessions"][-1]["export"])
    box = _box(out, record, last)
    assert any(r["status"] == "imported" for r in _data(box.call("job_results", {})))
    box.close()
    for change in ("extra", "missing"):
        copy = tmp_path / f"export-{change}"
        shutil.copytree(last, copy)
        path = copy / "log" / "job_episodes.parquet"
        rows = pq.read_table(path).to_pylist()
        if change == "extra":
            stray = dict(next(r for r in rows if r["job_id"] == "j1001"))
            rows.append({**stray, "job_id": "j1050", "episode_id": "e105001"})
        else:
            rows = [r for r in rows if r["episode_id"] != "e109801"]
        pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
        box = _box(out, record, copy)
        message = _error(box.call("job_results", {}))
        box.close()
        assert "failed public validation" in message, (change, message)


def _the_launch_audit_accepts_the_history(
    tmp_path: Path, env: Any, out: Path, record: dict[str, Any]
) -> None:
    """The mounts of a fourth session, with the imported history, pass `run_agent`'s audit."""
    from fleetlab.agents.audit import AuditContext, launch_audit
    from fleetlab.agents.sessions import Analysis
    from fleetlab.jobs.driver import JOB_SESSION_TOOLS, SYSTEM, installation, task_text_for
    from fleetlab.toolserver.jobs import job_tool_texts

    analysis = Analysis(Path(record["analysis"]))
    install = installation(out / "installation")
    access = analysis.tool_access(
        env,
        Path(record["sessions"][-1]["export"]),
        4,
        tools=JOB_SESSION_TOOLS,
        audit=AuditContext((install.run,), install.fleet),
    )
    state = RequestState.model_validate({**POOLED, "base_task": 1})
    box = tmp_path / "audit-box"
    box.mkdir()
    history = {"investigations": 3, "jobs": 6, "episodes": 56}
    texts = {
        "system": SYSTEM,
        "prompt": task_text_for(False)(state, 4, True, history),
        "schema": "{}",
        **job_tool_texts(state.budget),
    }
    assert launch_audit(access, box, texts)["passed"]


def test_the_history_s_own_words_pass_and_its_secrets_and_keys_stay_strict(
    tmp_path: Path,
) -> None:
    """agents.audit._HISTORY_PROSE: the earlier investigations' prose in results/history.json
    is scanned with the agent-text rules, which still refuse a canary, a seed and a hidden
    identifier there; every other key and value of the file stays strict."""
    import copy

    import pyarrow.parquet as pq
    from log_toy import write_toy_run

    from fleetlab.agents.audit import audit_state
    from fleetlab.contracts.log import SCHEMA
    from fleetlab.log._files import read_canary

    run = tmp_path / "toy-run"
    fleet = write_toy_run(run)
    canary = read_canary(run)
    assert canary is not None
    identities = pq.read_table(run / SCHEMA["identities"].path).to_pylist()
    seed = next(v for k, v in identities[0].items() if k.endswith("seed") and v)
    program = {
        "name": "verdict-check-chain",
        "purpose": "the verdict of a chained job",
        "steps": [{"kind": "policy", "instruction": "rate the tables", "until": None}],
        "notes": "the tables give the verdict",
    }
    history: dict[str, Any] = {
        "format": "fleetlab.job-history/1",
        "investigations": [
            {
                "investigation": 1,
                "requested_goals": ["On cream_cheese_1 akita_black_bowl_1"],
                "order_rules": [],
                "base_task": 6,
                "outcome": "done",
                "sessions": 3,
                "final_evaluation": {"job": "verdict-job", "N": 20, "K": 18, "delivered": 6},
                "notes": "Read the tables: the verdict is a miss, the rate is low.",
                "account": {"job": "verdict-job", "words": "The verdict: missed the condition."},
            }
        ],
        "imported": [
            {
                "job_id": "j1001",
                "investigation": 1,
                "part": "development",
                "name": "verdict-check-chain",
                "purpose": "the verdict of the tables",
                "episodes": [{"episode_id": "e100101", "source_episode_id": "e000101"}],
            }
        ],
        "programs": [{"program_digest": "a" * 64, "program": program}],
        "checks": [
            {
                "investigation": 1,
                "check": {
                    "name": "verdict-check",
                    "purpose": "the verdict per job",
                    "parameters": [{"name": "rate", "type": "int"}],
                    "sql": "SELECT 1 AS verdict FROM job_episodes WHERE control_steps > $rate",
                    "expects": "the verdict is one",
                },
            }
        ],
        "job_names": ["verdict-job"],
        "check_names": ["verdict-check"],
    }
    results = tmp_path / "results"
    results.mkdir()

    def hits(data: dict[str, Any]) -> list[str]:
        (results / "history.json").write_text(json.dumps(data))
        return [h.where for h in audit_state(results, (run,), fleet, "results").hits]

    assert hits(history) == []
    for leak in (canary, str(seed), "toy_camera_wobble"):
        for where in ("notes", "sql", "account", "names"):
            leaked = copy.deepcopy(history)
            if where == "notes":
                leaked["investigations"][0]["notes"] += f" {leak}"
            elif where == "sql":
                leaked["checks"][0]["check"]["sql"] += f" -- {leak}"
            elif where == "account":
                leaked["investigations"][0]["account"]["words"] += f" {leak}"
            else:
                leaked["job_names"].append(f"x {leak}")
            assert hits(leaked), (leak, where)
    # The same agent words at an unlisted path, or as a key, are strict.
    unlisted = copy.deepcopy(history)
    unlisted["investigations"][0]["outcome"] = "the verdict"
    assert hits(unlisted)
    keyed = copy.deepcopy(history)
    keyed["investigations"][0]["role"] = "development"
    assert hits(keyed)


class _BrokenOffInFirstSession:
    """The agent runtime, stopped by a signal in the first session: no batch has run."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from fleetlab.jobs.driver import Interrupted

        raise Interrupted("signal 15")


def test_exposure_stays_transitive_through_an_investigation_broken_off_before_any_batch(
    tmp_path: Path, env: Any
) -> None:
    """A review of 08f8704: an investigation broken off before it froze a library kept
    its ancestors' examined identities nowhere a successor imported them."""
    from fleetlab.jobs.driver import Interrupted
    from fleetlab.jobs.prior import preflight, read_prior

    # Generation 1: its final evaluation examined pool 6's twenty evaluation starts.
    one, record = _investigation(tmp_path, env, "plate-01", REQUEST, _source_one_steps)
    assert record is not None and record["release"]["receipt"]["N"] == 20
    examined = set(release_starts_for(6, 20, profile=None))
    # Generation 2 on pool 1, importing generation 1 or mounting its library, broken off in
    # its first session.
    pooled = {**POOLED, "base_task": 1}
    seconds = []
    for name, over in (("import", {"prior_runs": (one,)}), ("mount", {"library": one / "library"})):
        request = tmp_path / f"two-{name}.json"
        request.write_text(json.dumps(pooled))
        cohorts = assignment(tmp_path / f"two-{name}-cohorts.json", request, base_task=1)
        state = RequestState.model_validate(pooled)
        plan = Investigation(
            agent=AgentSpec("codex", "scripted"),
            runner=ToyRunner(state),
            scripted=_BrokenOffInFirstSession(),
            placeholder_task=True,
            env=env,
            policy_skills=toy_skills(),
            assignment=cohorts,
            cohort="replacement-01",
            **over,
        )
        out = tmp_path / f"two-{name}"
        with pytest.raises(Interrupted):
            investigate(request, out, plan)
        broken_off = json.loads((out / "job_run.json").read_text())
        assert broken_off["outcome"] == "incomplete" and broken_off["batches"] == []
        assert not (out / "library").exists() and not (out / "release").exists()
        seconds.append(out)
    # Generation 3 imports generation 2 alone: the twenty identities stay examined.
    six = tmp_path / "six.json"
    six.write_text(json.dumps({**REQUEST, "request_text": "Pool six again."}))
    fresh = {**POOLED, "base_task": 2}
    fresh_request = tmp_path / "fresh.json"
    fresh_request.write_text(json.dumps(fresh))
    fresh_cohorts = assignment(tmp_path / "fresh-cohorts.json", fresh_request, base_task=2)
    for two in seconds:
        before = _tree_digests(two)
        prior = read_prior([two])
        assert examined <= set(prior.exposure.identities), two.name
        with pytest.raises(PreflightError, match="were examined before"):
            preflight(
                RequestState.model_validate_json(six.read_text()),
                six,
                exposure=prior.exposure,
                seen=prior.identities,
            )
        checked = preflight(
            RequestState.model_validate(fresh),
            fresh_request,
            exposure=prior.exposure,
            seen=prior.identities,
            assignment=fresh_cohorts,
            cohort="replacement-01",
        )
        assert checked["identities_examined_before"] >= 20
        # And carried on again: generation 3's own carried exposure holds them.
        written = prior.write(tmp_path / f"three-from-{two.name}")
        assert examined <= library.exposed_identities(written["exposure"])
        assert _tree_digests(two) == before
