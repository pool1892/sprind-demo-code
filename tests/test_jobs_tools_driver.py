"""The job driver on CPU fixtures (fleetlab.jobs.driver): a scripted agent drives a toy job
through the tools, sessions, batches, the shift lead, the library and the release run; job
two mounts job one's library; an interrupted investigation stays incomplete; and the
launch audit accepts a job session's mounts. The tool server is the real one, the runner
the toy (no simulator), the agent a script (no model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fleetlab.agents.run import AgentResult
from fleetlab.agents.sessions import AgentSpec
from fleetlab.agents.toolenv import project_tool_env
from fleetlab.contracts.jobs import RequestState
from fleetlab.jobs.driver import (
    JOB_SESSION_TOOLS,
    SYSTEM,
    Investigation,
    ToyRunner,
    investigate,
    placeholder_task_text,
    task_text_for,
    toy_agent,
    toy_skills,
)
from fleetlab.toolserver.files import read_lines

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
# Job two draws its starts from another pool: job one's release examined pool 6's
# evaluation starts, which no later request may be evaluated on (jobs.prior.preflight).
JOB_TWO = {
    **REQUEST,
    "request_text": "The cream cheese into the bowl, then the bowl onto the stove.",
    "requested_goals": [
        {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
        {"predicate": "On", "objects": ["akita_black_bowl_1", "flat_stove_1_cook_region"]},
    ],
    "base_task": 4,
}


def write_request(path: Path, data: dict[str, Any]) -> tuple[Path, RequestState]:
    path.write_text(json.dumps(data))
    return path, RequestState.model_validate(data)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    return project_tool_env(tmp_path / "env")


def how(state: RequestState, root: Path, env: Any, **over: Any) -> Investigation:
    return Investigation(
        agent=AgentSpec("codex", "scripted"),
        runner=ToyRunner(state),
        scripted=toy_agent(root / "scripted", state),
        placeholder_task=True,
        env=env,
        policy_skills=toy_skills(),
        **over,
    )


def tool_log(root: Path) -> list[dict[str, Any]]:
    return [r for p in sorted(root.rglob("tool_calls.jsonl")) for r in read_lines(p)]


def test_a_scripted_agent_drives_a_toy_job_to_its_release(tmp_path: Path, env: Any) -> None:
    request, state = write_request(tmp_path / "request.json", REQUEST)
    out = tmp_path / "job-one"
    record = investigate(request, out, how(state, tmp_path, env))
    assert record["outcome"] == "done", record.get("ended_because")
    assert json.loads((out / "job_run.json").read_text())["outcome"] == "done"
    assert [s["state"] for s in record["sessions"]] == ["waiting", "waiting", "waiting", "done"]
    assert [s["queued"] for s in record["sessions"]] == [["j0001"], ["j0002", "j0003"], [], []]
    assert record["release"]["after_session"] == 3  # the session that saved the job
    assert record["acceptance"] == {"N": 20, "K": 16}
    log = [(m["author"], m["text"][:9]) for m in record["message_log"]]
    assert log[:5] == [("lead", "The cream"), ("agent", "Proposal:"), ("lead", "Not yet: "),
                       ("agent", "Proposal:"), ("lead", "Approved:")]  # fmt: skip
    assert log[-1] == ("agent", "Account, ")
    # Every tool of the investigation was used, and the reverse order was refused by rule.
    calls = tool_log(tmp_path / "scripted")
    used = {c["tool"] for c in calls}
    assert set(JOB_SESSION_TOOLS) - used <= {"get_frames", "list_skills", "read_skill",
                                             "memory_read"}  # fmt: skip
    refusals = [
        c for c in calls if c["tool"] == "save_job" and '"accepted":false' in c["result"][0]
    ]
    assert len(refusals) == 1 and "broke the process rule" in refusals[0]["result"][0]
    # The library, the release and the costs.
    assert record["library"]["jobs"] == ["loaded-carry"]
    assert record["library"]["checks"] == ["goal-undone"]
    assert record["library"]["skills"] == ["job-goal-undone"]
    receipt = record["release"]["receipt"]
    assert (receipt["N"], receipt["K"], receipt["delivered"], receipt["naive_delivered"]) == (
        20, 16, 18, 6,
    )  # fmt: skip
    assert receipt["rescued"] == 12 and receipt["broken"] == 0
    assert receipt["passed"] and receipt["released"]
    saved = json.loads((out / "library" / "jobs" / "loaded-carry.json").read_text())
    assert saved["receipt"]["delivered"] == 18 and saved["saved_from"] == ["j0002"]
    assert saved["check"] == "goal-undone" and len(saved["check_digest"]) == 64
    # Lane J's release recorded its exposure, carried into the final library.
    exposure = [
        json.loads(line)
        for line in (out / "library" / "release_exposure.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in exposure] == ["started", "finished"]
    assert record["lead_fraction"] == "4/5"
    costs = record["costs"]
    assert (costs["sessions"], costs["starts_charged"], costs["release_starts"]) == (4, 25, 40)
    assert record["released"] is True
    assert costs["tool_calls"] == len(calls)
    # Review finding 8: simulated starts apart from charged ones, and no queue-inclusive
    # time passed off as the runner's (the toy runner reports none).
    assert (costs["starts_simulated"], costs["starts_unfinished"]) == (25, 0)
    assert costs["batch_elapsed_seconds"] > 0 and costs["runner_seconds"] is None
    assert "simulation_wall_seconds" not in costs
    # Lane F: the film finds every run directory in the record.
    dirs = record["run_dirs"]
    assert [d["session"] for d in dirs["development"]] == [1, 2]
    assert dirs["release"] == str(out / "release" / "run")


def test_a_bar_below_four_fifths_carries_through_the_library_and_the_release(
    tmp_path: Path, env: Any
) -> None:
    """`--lead-fraction 2/3`: the lead approves 14 of 20, the saved job and its receipt carry
    the bar and validate against it, and the release is judged on K = 14."""
    from fleetlab.contracts.jobs import SavedJob

    request, state = write_request(tmp_path / "request.json", REQUEST)
    plan = how(state, tmp_path, env, lead_fraction=(2, 3))
    plan.scripted = toy_agent(tmp_path / "scripted-23", state, k=14)
    out = tmp_path / "job-one"
    record = investigate(request, out, plan)
    assert record["outcome"] == "done", record.get("ended_because")
    assert record["lead_fraction"] == "2/3" and record["acceptance"] == {"N": 20, "K": 14}
    receipt = record["release"]["receipt"]
    assert (receipt["K"], receipt["lead_fraction"], receipt["passed"]) == (14, [2, 3], True)
    saved = SavedJob.model_validate_json((out / "library/jobs/loaded-carry.json").read_text())
    assert saved.lead_fraction == [2, 3] and saved.acceptance is not None
    assert saved.acceptance.K == 14 and saved.receipt is not None
    with pytest.raises(ValueError, match="lead approves"):
        SavedJob.model_validate({**saved.model_dump(mode="json"), "lead_fraction": [4, 5],
                                 "receipt": None})  # fmt: skip


def test_job_two_mounts_job_one_s_library_and_reuses_its_check(tmp_path: Path, env: Any) -> None:
    request, state = write_request(tmp_path / "one.json", REQUEST)
    investigate(request, tmp_path / "job-one", how(state, tmp_path / "a", env))
    library = tmp_path / "job-one" / "library"
    request, state = write_request(tmp_path / "two.json", JOB_TWO)
    record = investigate(
        request, tmp_path / "job-two", how(state, tmp_path / "b", env, library=library)
    )
    assert record["outcome"] == "done"
    assert record["library_mounted"]["entries"] == {"jobs": ["loaded-carry"],
                                                    "checks": ["goal-undone"]}  # fmt: skip
    assert record["library_mounted"]["learned"]["source"] == str(library / "skills")
    calls = tool_log(tmp_path / "b" / "scripted")
    assert not [c for c in calls if c["tool"] == "write_check"]  # the saved check is reused
    ran = [c for c in calls if c["tool"] == "run_check" and c["ok"]]
    assert ran and all(c["arguments"]["name"] == "goal-undone" for c in ran)
    listed = [c for c in calls if c["tool"] == "list_skills"]
    assert not listed or "job-goal-undone" in listed[0]["result"][0]
    assert sorted(p.stem for p in (tmp_path / "job-two" / "library" / "jobs").glob("*.json")) == [
        "loaded-carry", "loaded-carry-2",
    ]  # fmt: skip
    assert record["release"]["job"] == "loaded-carry-2"


class LimitedAgent:
    """The scripted agent, with a rate limit on the `at`-th call."""

    def __init__(self, inner: Any, at: int, root: Path) -> None:
        self.inner, self.at, self.root, self.calls = inner, at, root, 0

    def __call__(self, backend: str, model: str, system: str, prompt: str, **kw: Any) -> Any:
        self.calls += 1
        if self.calls != self.at:
            return self.inner(backend, model, system, prompt, **kw)
        run_dir = self.root / f"limited-{self.calls}"
        (run_dir / "box" / "toolbox").mkdir(parents=True)
        if kw.get("on_start") is not None:
            kw["on_start"](run_dir)
        return AgentResult(False, None, {}, 0.1, error="rate_limit", error_detail="429",
                           run_dir=str(run_dir))  # fmt: skip


def test_an_interrupted_investigation_stays_incomplete(tmp_path: Path, env: Any) -> None:
    request, state = write_request(tmp_path / "request.json", REQUEST)
    plan = how(state, tmp_path, env)
    plan.scripted = LimitedAgent(plan.scripted, 2, tmp_path / "limited")
    out = tmp_path / "job"
    record = investigate(request, out, plan)
    assert record["outcome"] == "incomplete"
    assert "rate_limit" in record["ended_because"] and "not resumed" in record["ended_because"]
    assert "release" not in record and "library" not in record
    assert [s["state"] for s in record["sessions"]] == ["waiting", "failed"]
    assert len(record["batches"]) == 1  # session 1's job ran; nothing after the limit
    with pytest.raises(FileExistsError):
        investigate(request, out, how(state, tmp_path / "again", env))


class TimeoutAfterWork:
    """The scripted agent, whose `at`-th call does its work and then times out."""

    def __init__(self, inner: Any, at: int) -> None:
        self.inner, self.at, self.calls = inner, at, 0

    def __call__(self, backend: str, model: str, system: str, prompt: str, **kw: Any) -> Any:
        self.calls += 1
        res = self.inner(backend, model, system, prompt, **kw)
        if self.calls != self.at:
            return res
        return AgentResult(False, None, res.usage, res.seconds, error="timeout",
                           error_detail="timed out", tool_log=res.tool_log,
                           run_dir=res.run_dir)  # fmt: skip


def test_a_call_lost_after_queued_jobs_keeps_them_and_ends_incomplete(
    tmp_path: Path, env: Any
) -> None:
    """Review finding 1: the runtime replaced a call lost after `j0001` was accepted,
    restored the outbox and gave the replacement a fresh budget. The queued job is accepted
    work: no replacement, the job and its charge stand, the investigation is incomplete."""
    request, state = write_request(tmp_path / "request.json", REQUEST)
    plan = how(state, tmp_path, env)
    agent = TimeoutAfterWork(plan.scripted, 1)
    plan.scripted = agent
    record = investigate(request, tmp_path / "job", plan)
    assert agent.calls == 1  # no replacement call
    assert record["outcome"] == "incomplete"
    assert "timeout after accepted work (j0001)" in record["ended_because"]
    first = record["sessions"][0]
    assert first["queued"] == ["j0001"] and first["runtime_state"] == "waiting"
    outbox = Path(record["analysis"]) / "outbox" / "job_requests.jsonl"
    assert [r["job_id"] for r in read_lines(outbox)] == ["j0001"]
    assert record["batches"] == [] and "release" not in record


class InterruptedRunner:
    """The toy runner, stopped by a signal while its first batch waits for the queue."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def run(self, batch: Any, out: Path) -> Any:
        from fleetlab.jobs.driver import Interrupted

        raise Interrupted("signal 15")

    def release(self, *args: Any) -> Any:
        return self.inner.release(*args)


def test_an_interruption_while_waiting_ends_incomplete_with_costs(tmp_path: Path, env: Any) -> None:
    """Review finding 7: a stopped driver left `running` and no costs. It now ends
    incomplete, with the waiting batch marked interrupted and the costs so far."""
    from fleetlab.jobs.driver import Interrupted

    request, state = write_request(tmp_path / "request.json", REQUEST)
    plan = how(state, tmp_path, env)
    plan.runner = InterruptedRunner(plan.runner)
    out = tmp_path / "job"
    with pytest.raises(Interrupted):
        investigate(request, out, plan)
    record = json.loads((out / "job_run.json").read_text())
    assert record["outcome"] == "incomplete" and "not resumed" in record["ended_because"]
    assert [b["status"] for b in record["batches"]] == ["interrupted"]
    assert record["costs"]["sessions"] == 1 and record["costs"]["starts_simulated"] == 0


def test_the_model_minutes_end_the_investigation_incomplete(tmp_path: Path, env: Any) -> None:
    """Review finding 9: the 90 model minutes were not enforced. With a budget the
    first session uses up, no second session starts."""
    request, state = write_request(tmp_path / "request.json", REQUEST)
    record = investigate(request, tmp_path / "job", how(state, tmp_path, env, model_minutes=1e-9))
    assert record["outcome"] == "incomplete" and "model minutes are used" in record["ended_because"]
    assert len(record["sessions"]) == 1


def test_video_for_every_development_episode_when_asked(tmp_path: Path, env: Any) -> None:
    """Lane F: a real run records every development episode's video (`--video-all`, on by
    default on the command line) whatever the request asked."""
    from fleetlab.jobs.driver import parse

    assert parse(["--request", "r", "--out", "o"]).video_all is True
    request, state = write_request(tmp_path / "request.json", REQUEST)
    record = investigate(request, tmp_path / "job", how(state, tmp_path, env, video_all=True))
    assert all(j["record_video"] for b in record["batches"] for j in b["jobs"])


def test_the_runner_s_own_timings_are_read_from_its_log(tmp_path: Path) -> None:
    from fleetlab.jobs.driver import runner_timings

    log = tmp_path / "runner.log"
    log.write_text(
        "heavy-run: another heavy job holds the lock, waiting\n"
        "runner: policy xvla-libero ready in 22.5 s\n"
        "j0001 e000101 start 00: not delivered rea@92 t=392 order=kept wall=5.3s calls=14 (x)\n"
        "j0001 e000102 start 01: not delivered rea@94 t=411 order=kept wall=4.7s calls=15 (y)\n"
        "runner: {'done': 1, 'refused': 0, 'failed': 0} in 36 s\n"
    )
    timings = runner_timings(log)
    assert timings["waited_for_lock"] and timings["policy_load_seconds"] == 22.5
    assert timings["runner_seconds"] == 36.0 and timings["episode_seconds"] == [5.3, 4.7]


def launch_receipt(record: dict[str, Any], state: RequestState, env: Any, box: Path) -> Any:
    """`run_agent`'s launch audit on the mounts a fourth session of `record` would get."""
    from fleetlab.agents.audit import AuditContext, launch_audit
    from fleetlab.agents.sessions import Analysis
    from fleetlab.jobs.driver import installation
    from fleetlab.toolserver.jobs import job_tool_texts

    analysis = Analysis(Path(record["analysis"]))
    install = installation(Path(record["analysis"]).parents[1] / "installation")
    access = analysis.tool_access(
        env,
        Path(record["sessions"][-1]["export"]),
        4,
        tools=JOB_SESSION_TOOLS,
        audit=AuditContext((install.run,), install.fleet),
    )
    box.mkdir()
    texts = {
        "system": SYSTEM,
        "prompt": task_text_for(False)(state, 4, record["library_mounted"] is not None),
        "schema": "{}",
        **job_tool_texts(state.budget),
    }
    return launch_audit(access, box, texts)


def test_the_launch_audit_accepts_job_sessions_mounts(tmp_path: Path, env: Any) -> None:
    """What `run_agent` checks before a real model starts, on the mounts of a toy
    investigation's next session (its export with job tables, outbox and results, lane K's
    job skills and task text), for job one and for job two with job one's library and
    learned skill mounted."""
    import shutil

    from fleetlab.skill_library import skill_dirs

    skills = tmp_path / "job-skills"
    for source in skill_dirs("jobs"):
        shutil.copytree(source, skills / source.name)
    env = project_tool_env(tmp_path / "env-jobs", skills=skills)
    request, state = write_request(tmp_path / "one.json", REQUEST)
    one = investigate(request, tmp_path / "job-one", how(state, tmp_path / "a", env))
    assert launch_receipt(one, state, env, tmp_path / "box-one")["passed"]
    request, state = write_request(tmp_path / "two.json", JOB_TWO)
    library = tmp_path / "job-one" / "library"
    two = investigate(
        request, tmp_path / "job-two", how(state, tmp_path / "b", env, library=library)
    )
    assert two["library_mounted"]["learned"] is not None
    assert launch_receipt(two, state, env, tmp_path / "box-two")["passed"]


def test_the_driver_gives_lane_k_s_task_text_per_session() -> None:
    state = RequestState.model_validate(REQUEST)
    text = task_text_for(False)
    first = text(state, 1, False)
    assert (
        REQUEST["request_text"] in first and "goal 0: On cream_cheese_1 akita_black_bowl_1" in first
    )
    assert text(state, 2, True).startswith("This is session 2 of the investigation.")
    assert "An earlier job." in text(state, 2, True) and "An earlier job." not in first
    assert placeholder_task_text(state, 1, False) != first


def test_the_task_text_names_every_job_skill_and_the_installation_skill() -> None:
    from fleetlab.jobs.task_text import JOB_SKILLS
    from fleetlab.skill_library import skill_dirs

    text = task_text_for(False)(RequestState.model_validate(REQUEST), 1, False)
    assert "jobs-installation" in JOB_SKILLS
    assert JOB_SKILLS.index("jobs-traces") < JOB_SKILLS.index("jobs-installation")
    assert JOB_SKILLS.index("jobs-installation") < JOB_SKILLS.index("jobs-release")
    assert all(name in text for name in JOB_SKILLS)
    assert "jobs-installation" in {d.name for d in skill_dirs("jobs")}


def test_the_lead_fraction_is_a_launch_parameter() -> None:
    import argparse

    from fleetlab.jobs.driver import lead_fraction, parse

    assert lead_fraction("2/3") == (2, 3)
    for bad in ("3/2", "0/5", "two thirds"):
        with pytest.raises(argparse.ArgumentTypeError):
            lead_fraction(bad)
    args = parse(["--request", "r.json", "--out", "o", "--lead-fraction", "2/3"])
    assert args.lead_fraction == (2, 3)
    assert parse(["--request", "r.json", "--out", "o"]).lead_fraction == (4, 5)
