"""The job tools of the tool server and the operator's state (docs/design/jobs.md;
fleetlab.toolserver.jobs, fleetlab.jobs.operator).

The export is the installation's record with the toy runner's job tables
(fleetlab.jobs.driver.ToyRunner); the tool server is the real one, called in process.
No model, no simulator."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.jobs import JobRequest, RequestState
from fleetlab.contracts.tools import JOB_TOOLS, TOOLS, json_schema_strict
from fleetlab.jobs.driver import (
    JOB_SESSION_TOOLS,
    JobBatch,
    ToyRunner,
    _batch,
    _record_outcomes,
    installation,
    job_export,
    toy_programs,
    toy_skills,
)
from fleetlab.jobs.operator import ShiftLead, load_request, publish, sync_session
from fleetlab.toolserver.files import append_line, read_lines
from fleetlab.toolserver.jobs import (
    ACCEPTANCE_FILE,
    JOB_REQUESTS_FILE,
    JOB_RESULTS_FILE,
    OPERATOR_FILE,
    RELEASE_FILE,
    SAVED_JOBS_FILE,
    JobOutcome,
    lead_reply,
    rule_text,
)
from fleetlab.toolserver.server import Limits, Paths, ToolBox, default_tool_names, tool_description

REQUEST: dict[str, Any] = {
    "request_text": "The cream cheese into the bowl, then the bowl onto the plate.",
    "requested_goals": [
        {"predicate": "On", "objects": ["cream_cheese_1", "akita_black_bowl_1"]},
        {"predicate": "On", "objects": ["akita_black_bowl_1", "plate_1"]},
    ],
    "base_task": 6,
    "order_rules": [{"first": 0, "then": 1}],
    "budget": {
        "starts_per_request": 10,
        "requests_per_session": 2,
        "sessions": 4,
        "starts_per_investigation": 25,
    },
    "acceptance": None,
    "message_log": [],
}


def state_of(**budget: int) -> RequestState:
    data = json.loads(json.dumps(REQUEST))
    data["budget"].update(budget)
    return RequestState.model_validate(data)


class Job:
    """One analysis' directories over a changing export, with ToolBoxes by session."""

    def __init__(self, root: Path, state: RequestState) -> None:
        self.root = root
        self.state = state
        self.install = installation(root / "installation")
        for part in ("memory", "outbox", "results"):
            (root / part).mkdir()
        publish(root / "results", state, toy_skills())
        self.runs: list[Path] = []
        self.export = job_export(self.install, [], root / "exports" / "e00")
        self.runner = ToyRunner(state)
        self.boxes: list[ToolBox] = []

    def box(self, session: int, tools: tuple[str, ...] = JOB_SESSION_TOOLS) -> ToolBox:
        paths = Paths(
            export=self.export,
            memory=self.root / "memory",
            outbox=self.root / "outbox",
            results=self.root / "results",
            log=self.root / "logs" / f"session-{session:02d}" / "tool_calls.jsonl",
        )
        assert paths.log is not None
        paths.log.parent.mkdir(parents=True, exist_ok=True)
        box = ToolBox(paths, tools, session=session, limits=Limits())
        self.boxes.append(box)
        return box

    def run_pending(self, session: int) -> None:
        """What the driver does after a session: take its acceptance exchange into the
        host's request, run its queue, record, re-export."""
        self.state, _ = sync_session(self.state, self.root / "outbox", session)
        publish(self.root / "results", self.state, toy_skills())
        done = {r["job_id"] for r in read_lines(self.root / "results" / JOB_RESULTS_FILE)}
        pending = [
            r for r in read_lines(self.root / "outbox" / JOB_REQUESTS_FILE)
            if r["job_id"] not in done
        ]  # fmt: skip
        batch = _batch(pending, self.state, session)
        reply = self.runner.run(batch, self.root / "runs" / f"s{session:02d}")

        class Analysis:
            outbox = self.root / "outbox"
            results = self.root / "results"

        _record_outcomes(Analysis, batch, reply)
        assert reply.run_dir is not None
        self.runs.append(reply.run_dir)
        self.export = job_export(self.install, self.runs, self.root / "exports" / f"e{session:02d}")

    def close(self) -> None:
        for box in self.boxes:
            box.close()


@pytest.fixture
def job(tmp_path: Path) -> Iterator[Job]:
    made = Job(tmp_path / "job", state_of())
    yield made
    made.close()


def call(box: ToolBox, tool: str, arguments: dict[str, Any]) -> Any:
    result = box.call(tool, arguments)
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    if result.is_error:
        raise AssertionError(f"tool error: {text}")
    return json.loads(text)


def error(box: ToolBox, tool: str, arguments: dict[str, Any]) -> str:
    result = box.call(tool, arguments)
    assert result.is_error, result.content
    return "\n".join(p["text"] for p in result.content if p.get("type") == "text")


def approve(box: ToolBox) -> None:
    assert call(box, "submit_acceptance_proposal", {"N": 20, "K": 16, "reason": "r"})["approved"]


def submit(box: ToolBox, name: str = "naive", starts: list[int] | None = None) -> Any:
    programs = toy_programs(state_of())
    return call(
        box,
        "submit_job_program",
        {"program": programs[name], "starts": starts or list(range(5)), "record_video": False},
    )


# The tool contracts.


def test_every_job_tool_has_a_strict_schema_and_is_opt_in() -> None:
    specs = {s.name: s for s in TOOLS}
    for name in JOB_TOOLS:
        json_schema_strict(specs[name].arguments)
        assert name not in default_tool_names()
    assert set(JOB_TOOLS) <= set(JOB_SESSION_TOOLS)


def test_the_descriptions_state_the_request_s_budget(job: Job) -> None:
    box = job.box(1)
    text = {t.name: t.description for t in box.mcp_tools()}["submit_job_program"]
    assert "at most 10 starts a request, 2 requests a session, 25 starts" in text
    assert "0 to 29" in text
    spec = next(s for s in TOOLS if s.name == "submit_job_program")
    assert "at most 20 starts a request" in tool_description(spec, Limits())
    # The program's action horizon: its bounds and the legacy default, no value advised.
    assert "actions_per_chunk" in text and "5 to 30; null executes the whole chunk (30)" in text


# The scripted shift lead.


@pytest.mark.parametrize(
    ("n", "k", "approved", "words"),
    [
        (20, 16, True, "Approved"),
        (20, 20, True, "Approved"),
        (19, 19, False, "at least 20 release starts"),
        (20, 15, False, "at least 16 deliveries of 20"),
        (21, 21, False, "at most 20"),
        (20, 21, False, "cannot exceed N"),
    ],
)
def test_the_lead_approves_n_of_at_least_20_and_k_of_four_fifths(
    n: int, k: int, approved: bool, words: str
) -> None:
    ok, reply = lead_reply(n, k, 20)
    assert ok is approved
    assert words in reply


def test_the_lead_s_bar_is_a_launch_parameter_and_never_shown(tmp_path: Path) -> None:
    assert lead_reply(20, 14, 20, (2, 3))[0] and not lead_reply(20, 14, 20)[0]
    ok, reply = lead_reply(20, 13, 20, (2, 3))
    assert not ok and "at least 14 deliveries of 20" in reply
    made = Job(tmp_path / "two-thirds", state_of())
    publish(made.root / "results", made.state, toy_skills(), (2, 3))
    box = made.box(1)
    approved = call(box, "submit_acceptance_proposal", {"N": 20, "K": 14, "reason": "r"})
    assert approved["approved"]
    shown = json.dumps(call(box, "operator_state", {}))
    assert "lead_bar" not in shown and "numerator" not in shown
    state, problems = sync_session(made.state, made.root / "outbox", 1, ShiftLead(fraction=(2, 3)))
    assert problems == [] and state.acceptance is not None and state.acceptance.K == 14
    made.close()


def test_the_host_s_allocation_profile_is_never_mounted(tmp_path: Path) -> None:
    pooled = RequestState.model_validate({**REQUEST, "allocation_profile": "pool-v1"})
    made = Job(tmp_path / "pooled", pooled)
    mounted = (made.root / "results" / OPERATOR_FILE).read_text()
    assert "allocation_profile" not in mounted and "pool-v1" not in mounted
    shown = json.dumps(call(made.box(1), "operator_state", {}))
    assert "allocation_profile" not in shown and "pool-v1" not in shown
    made.close()


def test_a_proposal_is_answered_at_once_and_an_approval_stands(job: Job) -> None:
    box = job.box(1)
    low = call(box, "submit_acceptance_proposal", {"N": 10, "K": 10, "reason": "quick"})
    assert low["accepted"] and not low["approved"] and "at least 20" in low["reply"]
    approve(box)
    again = call(box, "submit_acceptance_proposal", {"N": 20, "K": 20, "reason": "stricter"})
    assert not again["accepted"] and "does not change" in again["problems"][0]
    state = call(box, "operator_state", {})
    assert state["acceptance"] == {"N": 20, "K": 16}
    authors = [m["author"] for m in state["message_log"]]
    assert authors == ["agent", "lead", "agent", "lead"]
    assert (
        "goal 1 (On akita_black_bowl_1 plate_1) may not first hold before goal 0"
        in (state["order_rules"][0]["in_words"])
    )


def test_the_host_decides_the_proposal_again_and_keeps_the_log(job: Job) -> None:
    box = job.box(1)
    call(box, "submit_acceptance_proposal", {"N": 10, "K": 8, "reason": "few"})
    approve(box)
    # An edited reply in the outbox does not stand: the host decides with the same rule.
    path = job.root / "outbox" / ACCEPTANCE_FILE
    lines = list(read_lines(path))
    lines[0].update(approved=True, reply="fine")
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    state, problems = sync_session(job.state, job.root / "outbox", 1, ShiftLead())
    assert state.acceptance is not None and (state.acceptance.N, state.acceptance.K) == (20, 16)
    assert problems == ["acceptance-1: the outbox's reply differs from the lead's"]
    assert [m.author for m in state.message_log] == ["agent", "lead", "agent", "lead"]


def test_a_request_file_with_an_acceptance_is_refused(tmp_path: Path) -> None:
    data = json.loads(json.dumps(REQUEST))
    data["acceptance"] = {"N": 20, "K": 16}
    (tmp_path / "r.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="acceptance is the agent's proposal"):
        load_request(tmp_path / "r.json")
    data["acceptance"] = None
    (tmp_path / "r.json").write_text(json.dumps(data))
    loaded = load_request(tmp_path / "r.json")
    assert [m.author for m in loaded.message_log] == ["lead"]


# Programs and the budget.


def test_no_program_is_tested_before_the_acceptance_is_approved(job: Job) -> None:
    box = job.box(1)
    receipt = submit(box)
    assert not receipt["accepted"]
    assert "before testing any program" in receipt["problems"][0]
    assert receipt["starts"] == 0


def test_release_starts_and_repeated_starts_are_refused(job: Job) -> None:
    box = job.box(1)
    approve(box)
    receipt = submit(box, starts=[1, 2, 30])
    assert not receipt["accepted"] and "[30] are not development starts" in receipt["problems"][0]
    assert not submit(box, starts=[1, 1])["accepted"]


def test_a_goal_index_beyond_the_request_is_refused(job: Job) -> None:
    box = job.box(1)
    approve(box)
    program = toy_programs(state_of())["naive"]
    program["steps"][1]["until"]["goal"] = 2
    receipt = call(
        box, "submit_job_program", {"program": program, "starts": [0], "record_video": False}
    )
    assert not receipt["accepted"] and "until.goal 2" in " ".join(receipt["problems"])


def test_the_budget_holds_from_the_first_session(job: Job) -> None:
    box = job.box(1)
    approve(box)
    assert not submit(box, starts=list(range(11)))["accepted"]  # 10 a request
    first = submit(box, starts=list(range(10)))
    assert first["accepted"] and first["job_id"] == "j0001"
    assert (first["starts_used"], first["starts_left"]) == (10, 15)
    second = submit(box, "release", starts=list(range(10)))
    assert second["accepted"]
    third = submit(box, starts=[0])
    assert not third["accepted"] and "at most 2 requests a session" in third["problems"][0]
    job.run_pending(1)
    box = job.box(2)
    over = submit(box, starts=list(range(6)))
    assert not over["accepted"] and "5 of the investigation's 25 are left" in over["problems"][0]
    assert submit(box, starts=list(range(5)))["accepted"]


def test_results_show_in_the_next_session_with_their_deliveries(job: Job) -> None:
    box = job.box(1)
    approve(box)
    submit(box, starts=list(range(10)))
    submit(box, "release", starts=list(range(10)))
    queued = call(box, "job_results", {})
    assert [r["status"] for r in queued] == ["queued", "queued"]
    job.run_pending(1)
    results = call(job.box(2), "job_results", {})
    assert [r["status"] for r in results] == ["done", "done"]
    naive, release = results
    assert (naive["delivered"], release["delivered"]) == (3, 9)  # the toy's physics
    assert naive["order_rule_violations"] == 0
    assert naive["episode_ids"][0] == "e000101" and release["episode_ids"][-1] == "e000210"
    rows = call(
        job.box(2), "query", {"sql": "SELECT count(*) AS n FROM job_episodes WHERE delivered"}
    )
    assert rows["rows"] == [[12]]


def test_an_outcome_of_the_current_session_or_off_its_rows_is_refused(job: Job) -> None:
    box = job.box(1)
    approve(box)
    submit(box)
    job.run_pending(1)
    # Visible only in a later session.
    assert "failed public validation" in error(job.box(1), "job_results", {})
    # An outcome that names episodes the export does not hold for its job.
    path = job.root / "results" / JOB_RESULTS_FILE
    outcome = JobOutcome.model_validate(next(read_lines(path)))
    forged = outcome.model_copy(update={"episode_ids": ["e000102", "e000103", "e000104",
                                                        "e000105", "e000199"]})  # fmt: skip
    path.write_text(forged.model_dump_json() + "\n")
    assert "failed public validation" in error(job.box(2), "job_results", {})


def test_a_refused_job_is_free(job: Job) -> None:
    box = job.box(1)
    approve(box)
    receipt = submit(box, starts=list(range(10)))
    append_line(
        job.root / "results" / JOB_RESULTS_FILE,
        {
            "job_id": receipt["job_id"],
            "status": "refused",
            "episode_ids": [],
            "message": "The runner could not run this program; its starts were returned.",
            "program_digest": "",
            "problems": ["step 0: an instruction the runner refused"],
        },
    )
    later = job.box(2)
    status = call(later, "operator_state", {})["budget"]
    assert status["starts_used"] == 0 and status["starts_left"] == 25
    result = call(later, "job_results", {})[0]
    assert result["status"] == "refused" and result["starts_charged"] == 0
    assert result["problems"] == ["step 0: an instruction the runner refused"]


# Saving a job, checks, the account.


def test_save_job_refuses_a_job_that_broke_the_rule_with_the_rule_s_text(job: Job) -> None:
    box = job.box(1)
    approve(box)
    reverse = submit(box, "reverse")
    release = submit(box, "release")
    early = call(
        box,
        "save_job",
        {"name": "carry", "job_id": release["job_id"], "purpose": "p", "check": None},
    )
    assert not early["accepted"] and "has not run" in early["problems"][0]
    job.run_pending(1)
    box = job.box(2)
    refused = call(
        box,
        "save_job",
        {"name": "carry", "job_id": reverse["job_id"], "purpose": "p", "check": None},
    )
    assert not refused["accepted"]
    assert "broke the process rule" in refused["problems"][0]
    assert rule_text(job.state, job.state.order_rules[0]) in refused["problems"][0]
    saved = call(
        box,
        "save_job",
        {"name": "carry", "job_id": release["job_id"], "purpose": "p", "check": None},
    )
    assert saved["accepted"] and saved["program_digest"] == release["program_digest"]
    library = call(box, "job_library", {})
    assert [j["name"] for j in library["this_investigation"]["jobs"]] == ["carry"]


CHECK = {
    "name": "undone-at-end",
    "purpose": "runs of a job in which a goal held and no longer holds at the end",
    "parameters": [{"name": "goal", "type": "int"}],
    "sql": "SELECT e.job_id, count(*) AS n FROM job_episodes e WHERE "
    "list_contains(list_transform(e.goals_held_at_end, x -> x), false) AND EXISTS "
    "(SELECT 1 FROM goal_steps g WHERE g.episode_id = e.episode_id AND g.goal = $goal "
    "AND g.holds) GROUP BY e.job_id ORDER BY e.job_id",
    "expects": "none for a job that keeps its earlier goals",
}


def test_a_check_is_run_once_when_written_and_again_by_name(job: Job) -> None:
    box = job.box(1)
    approve(box)
    naive = submit(box, starts=list(range(10)))
    release = submit(box, "release", starts=list(range(10)))
    job.run_pending(1)
    box = job.box(2)
    missing = call(box, "write_check", {"check": CHECK, "check_with": []})
    assert not missing["accepted"] and "check_with" in missing["problems"][0]
    bad = dict(CHECK, sql="DELETE FROM job_episodes WHERE $goal = 0")
    refused = call(
        box, "write_check", {"check": bad, "check_with": [{"name": "goal", "value": "0"}]}
    )
    assert not refused["accepted"] and "one SELECT" in refused["problems"][0]
    written = call(
        box, "write_check", {"check": CHECK, "check_with": [{"name": "goal", "value": "0"}]}
    )
    assert written["accepted"]
    everything = call(box, "run_check", {"name": "undone-at-end", "job_id": None,
                                         "params": [{"name": "goal", "value": "0"}]})  # fmt: skip
    assert everything["rows"] == [[naive["job_id"], 6]]  # the toy undoes goal 0 on 6 of 10
    one = call(box, "run_check", {"name": "undone-at-end", "job_id": release["job_id"],
                                  "params": [{"name": "goal", "value": "0"}]})  # fmt: skip
    assert one["rows"] == [] and one["job_id"] == release["job_id"]
    naive_only = call(box, "run_check", {"name": "undone-at-end", "job_id": naive["job_id"],
                                         "params": [{"name": "goal", "value": "0"}]})  # fmt: skip
    assert naive_only["rows"] == [[naive["job_id"], 6]]
    assert "takes the parameters" in error(
        box, "run_check", {"name": "undone-at-end", "job_id": None, "params": []}
    )
    assert "no check named" in error(
        box, "run_check", {"name": "nothing-here", "job_id": None, "params": []}
    )


def test_a_saved_job_keeps_the_version_of_its_check_that_ran_on_it(job: Job) -> None:
    """Review finding 5: save_job checked only that the check's name existed. The save now
    needs a run_check of that version on the job, keeps the version's digest, and the check
    cannot be rewritten under a saved job."""
    from fleetlab.contracts.jobs import CheckSpec
    from fleetlab.toolserver.jobs import check_digest

    box = job.box(1)
    approve(box)
    ran = submit(box, "release")
    job.run_pending(1)
    box = job.box(2)
    goal = [{"name": "goal", "value": "0"}]
    assert call(box, "write_check", {"check": CHECK, "check_with": goal})["accepted"]
    args = {"name": "carry", "job_id": ran["job_id"], "purpose": "p", "check": CHECK["name"]}
    untested = call(box, "save_job", args)
    assert not untested["accepted"] and "has not been run on" in untested["problems"][0]
    call(box, "run_check", {"name": CHECK["name"], "job_id": ran["job_id"], "params": goal})
    changed = dict(CHECK, expects="a different expectation")
    assert call(box, "write_check", {"check": changed, "check_with": goal})["accepted"]
    assert not call(box, "save_job", args)["accepted"]  # the version run on it was replaced
    call(box, "run_check", {"name": CHECK["name"], "job_id": ran["job_id"], "params": goal})
    assert call(box, "save_job", args)["accepted"]
    saved = next(read_lines(job.root / "outbox" / SAVED_JOBS_FILE))
    assert saved["check_digest"] == check_digest(CheckSpec.model_validate(changed))
    again = call(box, "write_check", {"check": CHECK, "check_with": goal})
    assert not again["accepted"] and "a saved job names" in again["problems"][0]


def test_an_outcome_of_another_program_than_queued_is_refused(job: Job) -> None:
    """Review finding 6: a runner's digest of 64 zeroes still showed results and allowed a
    save. The runner's digest must be the queued program's."""
    box = job.box(1)
    approve(box)
    ran = submit(box, "release")
    job.run_pending(1)
    path = job.root / "results" / JOB_RESULTS_FILE
    outcome = JobOutcome.model_validate(next(read_lines(path)))
    path.write_text(outcome.model_copy(update={"program_digest": "0" * 64}).model_dump_json())
    box = job.box(2)
    assert "failed public validation" in error(box, "job_results", {})
    args = {"name": "carry", "job_id": ran["job_id"], "purpose": "p", "check": None}
    assert "failed public validation" in error(box, "save_job", args)


def test_a_check_with_an_undeclared_parameter_is_refused_by_the_model(job: Job) -> None:
    box = job.box(1)
    bad = dict(CHECK, parameters=[])
    assert "invalid arguments" in error(box, "write_check", {"check": bad, "check_with": []})


def test_the_account_follows_the_release_of_the_job_saved_last(job: Job) -> None:
    box = job.box(1)
    approve(box)
    release = submit(box, "release")
    early = call(box, "submit_job_account", {"job": "carry", "account": "It works."})
    assert not early["accepted"]
    assert any("nothing was released" in p for p in early["problems"])
    assert any("queued j0001" in p for p in early["problems"])
    queued_save = call(
        box,
        "save_job",
        {"name": "carry", "job_id": release["job_id"], "purpose": "p", "check": None},
    )
    assert not queued_save["accepted"]
    job.run_pending(1)
    box = job.box(2)
    unknown = call(box, "save_job", {"name": "carry", "job_id": release["job_id"],
                                     "purpose": "p", "check": "no-such-check"})  # fmt: skip
    assert not unknown["accepted"] and "no check named" in unknown["problems"][0]
    saved = call(
        box,
        "save_job",
        {"name": "carry", "job_id": release["job_id"], "purpose": "p", "check": None},
    )
    assert saved["accepted"]
    assert call(box, "operator_state", {})["release_pending"] == "carry"
    pending = call(box, "submit_job_account", {"job": "carry", "account": "It works."})
    assert not pending["accepted"] and "released after this session" in pending["problems"][0]
    more = submit(box, starts=[1])
    assert not more["accepted"] and "released after it" in more["problems"][0]
    # The host's release, with its counts in the results.
    counts = {"job": "carry", "after_session": 2, "N": 20, "K": 16, "delivered": 18,
              "naive_delivered": 6, "program_only": 12, "naive_only": 0, "order_violations": 0,
              "condition_met": True, "released": True, "program_digest": ""}  # fmt: skip
    (job.root / "results" / RELEASE_FILE).write_text(json.dumps(counts))
    box = job.box(3)
    assert call(box, "operator_state", {})["release"]["delivered"] == 18
    again = call(
        box,
        "save_job",
        {"name": "carry", "job_id": release["job_id"], "purpose": "p", "check": None},
    )
    assert not again["accepted"] and "result is final" in " ".join(again["problems"])
    assert not submit(box, starts=[2])["accepted"]
    wrong = call(box, "submit_job_account", {"job": None, "account": "It works."})
    assert not wrong["accepted"] and "the released job is 'carry'" in wrong["problems"][0]
    done = call(box, "submit_job_account", {"job": "carry", "account": "Delivered 18 of 20."})
    assert done["accepted"] and done["receipt_id"] == "job_account-1"
    after = submit(box, starts=[1])
    assert not after["accepted"] and "account is submitted" in after["problems"][0]


def test_stopping_without_a_release_is_an_account_with_no_job(job: Job) -> None:
    box = job.box(1)
    approve(box)
    stopped = call(box, "submit_job_account", {"job": None, "account": "Nothing was ready."})
    assert stopped["accepted"]


def test_no_job_is_saved_after_the_account(job: Job) -> None:
    """Review finding 4: a save after `submit_job_account` released a job behind an account
    that predated it. The account ends the investigation: no save after it."""
    box = job.box(1)
    approve(box)
    ran = submit(box, "release")
    job.run_pending(1)
    box = job.box(2)
    assert call(box, "submit_job_account", {"job": None, "account": "Nothing was ready."})[
        "accepted"
    ]
    late = call(
        box, "save_job", {"name": "carry", "job_id": ran["job_id"], "purpose": "p", "check": None}
    )
    assert not late["accepted"] and "account is submitted" in " ".join(late["problems"])


def test_only_an_account_after_the_release_ends_the_investigation() -> None:
    from fleetlab.jobs.driver import account_follows

    def account(session: int, job: str | None) -> dict[str, Any]:
        return {"session": session, "answer": {"job": job, "account": "a"}}

    released = {"release": {"job": "carry", "after_session": 2}}
    assert account_follows(account(3, "carry"), 3, released)
    assert not account_follows(account(2, None), 3, released)  # an earlier session's
    assert not account_follows(account(2, "carry"), 2, released)  # before the release ran
    assert not account_follows(account(3, None), 3, released)  # not naming the release
    assert account_follows(account(2, None), 2, {})
    assert not account_follows(account(2, "carry"), 2, {})
    assert not account_follows(None, 2, {})


def test_a_long_account_is_kept_whole_and_shortened_in_the_message_log(job: Job) -> None:
    """Review finding 3: a 2,000-character account (JobAccount allows 6,000) overflowed the
    message log's 2,000 and crashed the host's sync. The log keeps a bounded form; the
    account's own record keeps it whole."""
    from fleetlab.jobs.operator import account_of

    box = job.box(1)
    approve(box)
    text = "x" * 6000
    assert call(box, "submit_job_account", {"job": None, "account": text})["accepted"]
    job.run_pending(1)
    last = job.state.message_log[-1]
    assert last.author == "agent" and len(last.text) <= 2000
    assert last.text.startswith("Account, nothing released: xxx") and "full account" in last.text
    kept = account_of(job.root / "outbox")
    assert kept is not None and kept["answer"]["account"] == text


def test_operator_state_lists_the_skills_and_the_delivery_thresholds(job: Job) -> None:
    state = call(job.box(1), "operator_state", {})
    assert [s["task"] for s in state["policy_skills"]] == list(range(10))
    assert state["policy_skills"][6]["instruction"] == "put the cream cheese in the bowl"
    assert state["delivery_thresholds"]["linear_speed_below_m_s"] == pytest.approx(0.01)
    assert state["base_task"]["task"] == "put_the_cream_cheese_in_the_bowl"
    assert state["release"] is None and state["release_pending"] is None


def test_a_mounted_library_is_listed_and_its_check_runs(job: Job) -> None:
    library = job.root / "results" / "library"
    library.mkdir()
    (library / "saved_checks.json").write_text(json.dumps([CHECK]))
    box = job.box(1)
    approve(box)
    submit(box)
    job.run_pending(1)
    box = job.box(2)
    listed = call(box, "job_library", {})
    assert [c["name"] for c in listed["installation"]["checks"]] == ["undone-at-end"]
    ran = call(box, "run_check", {"name": "undone-at-end", "job_id": None,
                                  "params": [{"name": "goal", "value": "0"}]})  # fmt: skip
    assert ran["check"] == "undone-at-end"
    taken = call(
        box, "write_check", {"check": CHECK, "check_with": [{"name": "goal", "value": "0"}]}
    )
    assert not taken["accepted"] and "the installation has a check" in taken["problems"][0]


# Frames of job episodes.


def test_get_frames_reads_a_job_episode_s_video_by_control_step(job: Job) -> None:
    import imageio.v2 as imageio
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    box = job.box(1)
    approve(box)
    submit(box, starts=[0])
    job.run_pending(1)
    run = job.runs[0]
    episode = "e000101"
    # A scene video of four frames, one every four control steps.
    video = run / "log" / "video" / f"{episode}.scene.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video, fps=5, macro_block_size=1)
    for value in (0, 80, 160, 240):
        writer.append_data(np.full((16, 16, 3), value, np.uint8))
    writer.close()
    episodes = pq.read_table(run / "log" / "job_episodes.parquet").to_pylist()
    episodes[0]["video_scene"] = f"video/{episode}.scene.mp4"
    schema = pq.read_schema(run / "log" / "job_episodes.parquet")
    pq.write_table(
        pa.Table.from_pylist(episodes, schema=schema), run / "log" / "job_episodes.parquet"
    )
    steps_file = run / "log" / "robot_steps" / f"{episode}.parquet"
    steps_schema = pq.read_schema(steps_file)
    template = pq.read_table(steps_file).to_pylist()[0]
    rows = [dict(template, step=s, frame_index=s // 4 if s % 4 == 0 else None) for s in range(13)]
    pq.write_table(pa.Table.from_pylist(rows, schema=steps_schema), steps_file)
    job.export = job_export(job.install, job.runs, job.root / "exports" / "frames")
    box = job.box(2)
    result = box.call("get_frames", {"episode_id": episode, "steps": [0, 9], "camera": "scene"})
    assert not result.is_error, result.content
    texts = [p["text"] for p in result.content if p["type"] == "text"]
    assert texts[0].endswith("control step 0: its frame (video frame 0)")
    assert texts[1].endswith(
        "control step 9: no frame at this step; the nearest earlier frame, at step 8 "
        "(video frame 2)"
    )
    described = {t.name: t.description for t in box.mcp_tools()}["get_frames"]
    assert "nearest earlier frame" in described
    assert sum(p["type"] == "image" for p in result.content) == 2
    wrist = box.call("get_frames", {"episode_id": episode, "steps": [0], "camera": "wrist"})
    assert wrist.is_error and "no wrist video" in wrist.content[0]["text"]


def test_a_batch_carries_the_request_s_state_and_development_starts() -> None:
    state = state_of()
    programs = toy_programs(state)
    record = {
        "job_id": "j0003",
        "session": 1,
        "program_digest": "",
        "request": {"program": programs["naive"], "starts": [4, 5], "record_video": True},
    }
    from fleetlab.contracts.jobs import JobProgram

    record["program_digest"] = JobProgram.model_validate(programs["naive"]).digest
    batch = _batch([record], state, 1)
    assert isinstance(batch, JobBatch)
    data = batch.to_json()
    assert data["format"] == "fleetlab.job-batch/1" and data["state"]["base_task"] == 6
    request = JobRequest.model_validate(data["jobs"][0]["request"])
    assert [s.init_state for s in request.starts] == [4, 5] and request.record_video
    record["program_digest"] = "0" * 64
    with pytest.raises(ValueError, match="changed since it was queued"):
        _batch([record], state, 1)
