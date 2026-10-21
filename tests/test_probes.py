"""Probes the agent composes (contracts.probes; docs/design/loop.md, section 10): the program's
one interpretation, its public bounds, and the tool server's request_probe and probe_results
with the attempt budget they share with experiments.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from log_toy import TASK_CLASSES, write_toy_run
from pydantic import ValidationError

from fleetlab.contracts.probes import (
    PROBE_BOUNDS,
    ProbeOutcome,
    ProbeProgram,
    ProbeReceipt,
    ProbeRequest,
    ProbeSegment,
    program_trajectory,
)
from fleetlab.contracts.tools import json_schema_strict, strict_schema_problems
from fleetlab.log import export_runs
from fleetlab.toolserver.files import EXPERIMENTS_FILE, PROBES_FILE, append_line
from fleetlab.toolserver.server import Limits, Paths, ToolBox, tool_names


def seg(primitive: str, duration: int, axis: str | None = None, amplitude: float | None = None,
        period: int | None = None) -> ProbeSegment:  # fmt: skip
    return ProbeSegment(
        primitive=primitive,  # type: ignore[arg-type]
        axis=axis,  # type: ignore[arg-type]
        amplitude_m=amplitude,
        duration_steps=duration,
        period_steps=period,
    )


def test_the_program_plays_its_primitives_in_order() -> None:
    segments = [
        seg("hold", 3),
        seg("step", 4, "x", 0.02),
        seg("ramp", 4, "z", 0.04),
        seg("sine", 8, "y", 0.01, 8),
        seg("close_gripper", 2),
        seg("open_gripper", 1),
    ]
    t = program_trajectory(segments)
    assert len(t.segment) == ProbeProgram(segments=segments).steps == 22
    assert t.first_step == (0, 3, 7, 11, 19, 21)
    assert np.all(t.offset[:3] == 0.0)
    assert np.all(t.offset[3:7, 0] == 0.02)  # the step jumps at its first step and stays
    assert np.allclose(t.offset[7:11, 2], [0.01, 0.02, 0.03, 0.04])  # the ramp ends on target
    assert np.allclose(t.offset[11:19, 1], 0.01 * np.sin(2 * np.pi * np.arange(8) / 8))
    assert np.all(t.offset[19:, 0] == 0.02) and np.all(t.offset[19:, 2] == 0.04)
    assert np.all(t.offset[19:, 1] == 0.0)  # a sine ends where it began
    assert list(t.gripper) == [-1.0] * 19 + [1.0, 1.0, -1.0]  # the gripper starts open
    assert t.base[2] == (0.02, 0.0, 0.0)


def raw(primitive: str, duration: int, axis: str | None = None, amplitude: float | None = None,
        period: int | None = None) -> dict[str, Any]:  # fmt: skip
    return {
        "primitive": primitive,
        "axis": axis,
        "amplitude_m": amplitude,
        "duration_steps": duration,
        "period_steps": period,
    }


@pytest.mark.parametrize(
    ("segments", "words"),
    [
        ([raw("step", 5, "x", 0.08)], "amplitude_m is at most"),
        ([raw("step", 5, "x", 0.04), raw("sine", 20, "x", 0.03, 40)], "reaches"),
        ([raw("ramp", 1, "z", 0.05)], "m/s"),
        ([raw("sine", 20, "z", 0.05, 4)], "m/s"),
        ([raw("sine", 20, "z", 0.001, 2)], "period_steps"),
        ([raw("hold", 200), raw("hold", 200), raw("hold", 1)], "at most 400"),
        ([raw("hold", 5, "x", None)], "takes no axis"),
        ([raw("step", 5, "x", None)], "needs an axis and an amplitude_m"),
        ([raw("hold", 5, None, None, 10)], "takes no period_steps"),
    ],
)
def test_a_program_outside_the_public_bounds_is_refused(
    segments: list[dict[str, Any]], words: str
) -> None:
    with pytest.raises(ValidationError, match=words):
        ProbeProgram.model_validate({"segments": segments})


def test_a_bad_segment_is_refused_on_its_own() -> None:
    with pytest.raises(ValidationError, match="needs an axis"):
        seg("sine", 10, None, None, 10)
    with pytest.raises(ValidationError):
        seg("hold", PROBE_BOUNDS.max_segment_steps + 1)


def request(**over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": "step-response-x",
        "purpose": "how quickly the hand follows a commanded step",
        "expectation": "the hand reaches the new target within a few tenths of a second",
        "robots": ["r01", "r02"],
        "repeats": 2,
        "segments": [
            seg("hold", 10).model_dump(mode="json"),
            seg("step", 30, "x", 0.03).model_dump(mode="json"),
            seg("hold", 20).model_dump(mode="json"),
        ],
    }
    out.update(over)
    return out


def test_the_request_has_a_strict_schema_and_costs_one_attempt_per_episode() -> None:
    assert strict_schema_problems(json_schema_strict(ProbeRequest)) == []
    parsed = ProbeRequest.model_validate(request())
    assert parsed.attempts == 4 and parsed.program.steps == 60
    with pytest.raises(ValidationError, match="each robot once"):
        ProbeRequest.model_validate(request(robots=["r01", "r01"]))
    with pytest.raises(ValidationError):
        ProbeRequest.model_validate(request(name="Has Spaces"))
    with pytest.raises(ValidationError):
        ProbeRequest.model_validate(request(repeats=PROBE_BOUNDS.max_repeats + 1))


def outcome(status: str, message: str, episodes: tuple[str, ...] = (),
            executed: int = 0, reserved: int = 0) -> ProbeOutcome:  # fmt: skip
    return ProbeOutcome(
        probe_id="p0001",
        status=status,  # type: ignore[arg-type]
        episode_ids=list(episodes),
        message=message,
        attempts_executed=executed,
        attempts_reserved=reserved,
    )


def test_an_outcome_carries_only_ids_counts_and_a_fixed_message() -> None:
    outcome("done", "done", ("e000101",), 1)
    # A broken-off run: known executed work and the reserved, unknown rest, kept apart.
    assert outcome("failed", "execution_failed", (), 1, 3).charged == 4
    with pytest.raises(ValidationError):
        outcome("done", "done", ("e000101",), 1, 1)  # only a failed probe reserves
    with pytest.raises(ValidationError):
        outcome("refused", "it broke")
    with pytest.raises(ValidationError):
        outcome("done", "done")
    with pytest.raises(ValidationError):
        outcome("done", "done", ("e000101",), 2)  # done executed exactly its episodes
    with pytest.raises(ValidationError):
        outcome("refused", "probe_not_runnable", (), 1)  # refused before anything ran


# The tool server.


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("probes")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    return base / "export"


def paths(root: Path, export: Path) -> Paths:
    for part in ("memory", "outbox", "results"):
        (root / part).mkdir(parents=True, exist_ok=True)
    return Paths(
        export=export,
        memory=root / "memory",
        outbox=root / "outbox",
        results=root / "results",
        log=root / "calls.jsonl",
    )


@pytest.fixture
def box(tmp_path: Path, export: Path) -> Iterator[ToolBox]:
    toolbox = ToolBox(
        paths(tmp_path, export),
        tool_names(),  # every tool, the opt-in probes included
        session=2,
        limits=Limits(max_experiment_episodes=18, budget_first_session=2),
    )
    yield toolbox
    toolbox.close()


def answer(result: Any) -> Any:
    assert not result.is_error, result.content
    return json.loads("\n".join(p["text"] for p in result.content if p["type"] == "text"))


def test_probe_tools_are_offered_only_when_named(tmp_path: Path, export: Path) -> None:
    default = ToolBox(paths(tmp_path, export))
    assert "request_probe" not in default.enabled and "probe_results" not in default.enabled
    assert default.call("request_probe", request()).is_error
    default.close()


def test_a_probe_is_charged_to_the_budget_apart_from_task_attempts(box: ToolBox) -> None:
    # A task experiment of this round: two arms of three attempts.
    outbox = box.paths.outbox
    arms = {"arms": [{"actions": []}, {"actions": []}], "robots": ["r01"], "paired_starts": True,
            "power_note": "n"}  # fmt: skip
    append_line(outbox / EXPERIMENTS_FILE, {"experiment_id": "x0001", "session": 2,
                "request": {**arms, "episodes_per_arm": 3}})  # fmt: skip
    receipt = ProbeReceipt.model_validate(answer(box.call("request_probe", request())))
    assert receipt.accepted and receipt.probe_id == "p0001"
    assert (receipt.attempts, receipt.steps_per_episode) == (4, 60)
    assert (receipt.task_attempts_used, receipt.probe_attempts_used) == (6, 4)
    assert receipt.attempts_left == 18 - 10
    status = answer(box.call("experiment_budget", {}))
    assert status["attempts_used"] == 10 and status["attempts_left"] == 8
    assert status["task_attempts_used"] == 6 and status["probe_attempts_used"] == 4
    assert status["probes_used"] == 1 and status["probes"] == box.limits.max_probe_requests
    # Over the budget: refused, free, and named.
    refused = ProbeReceipt.model_validate(
        answer(box.call("request_probe", request(repeats=5, robots=["r01", "r02"])))
    )
    assert not refused.accepted and refused.attempts == 0 and refused.probe_id == ""
    assert "this probe needs 10 attempts" in refused.problems[0]
    # An experiment request sees the probe's charge too.
    over = answer(box.call("request_experiment", {**arms, "episodes_per_arm": 6}))
    assert not over["accepted"] and "needs 12" in " ".join(over["problems"])


def test_a_probe_on_an_unknown_robot_is_refused(box: ToolBox) -> None:
    refused = answer(box.call("request_probe", request(robots=["r09"])))
    assert not refused["accepted"] and "no robot 'r09'" in refused["problems"][0]
    assert not (box.paths.outbox / PROBES_FILE).exists()


def test_probe_results_show_the_agent_s_own_definition_while_queued(box: ToolBox) -> None:
    assert answer(box.call("request_probe", request()))["accepted"]
    queued = answer(box.call("probe_results", {}))
    assert [r["status"] for r in queued] == ["queued"]
    assert queued[0]["name"] == "step-response-x" and queued[0]["episode_ids"] == []
    assert queued[0]["purpose"].startswith("how quickly")


# Outcomes are bound to their request and to the export's probe rows, like experiment
# results (an early review of b38534e).

FOUR = [("e000901", "r01", 0), ("e000902", "r01", 1), ("e000903", "r02", 0), ("e000904", "r02", 1)]


def probe_export(root: Path, export: Path, rows: list[tuple[str, str, str, int]],
                 steps: set[str]) -> Path:  # fmt: skip
    """A copy of the export with probe rows: (episode, probe, robot, repeat), and two
    probe_steps rows for each episode in `steps`."""
    from fleetlab.contracts.log import SCHEMA
    from fleetlab.log._rows import to_arrow

    out = root / "export-with-probes"
    shutil.copytree(export, out)
    episodes = [
        {
            "probe_id": p,
            "episode_id": e,
            "robot_id": r,
            "site_id": "s1",
            "day": 2,
            "repeat": k,
            "steps": 1,
        }
        for e, p, r, k in rows
    ]
    pq.write_table(to_arrow("probe_episodes", episodes), out / SCHEMA["probe_episodes"].path)
    for e in steps:
        step_rows = [
            {
                "episode_id": e,
                "step": k,
                "time_s": k / 20,
                "segment": 0 if k == 0 else None,
                "target_pos": [0.0, 0.0, 0.0] if k == 0 else None,
                "gripper_command": -1.0 if k == 0 else None,
                "eef_pos": [0.0, 0.0, 0.0],
                "eef_quat": [1.0, 0.0, 0.0, 0.0],
                "joint_pos": [0.0] * 7,
                "joint_vel": [0.0] * 7,
                "gripper_opening": 0.08,
            }
            for k in range(2)
        ]
        path = out / SCHEMA["probe_steps"].path.format(episode_id=e)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(to_arrow("probe_steps", step_rows), path)
    return out


def results_box(root: Path, export: Path, outcomes: list[dict[str, Any]],
                session_of_probe: int = 1) -> ToolBox:  # fmt: skip
    """A session-2 box whose outbox holds p0001 (r01 and r02, two repeats) from
    `session_of_probe` and whose results hold `outcomes`."""
    box = ToolBox(paths(root, export), tool_names(), session=2, limits=Limits())
    record = {"probe_id": "p0001", "session": session_of_probe, "request": request()}
    append_line(root / "outbox" / PROBES_FILE, record)
    for outcome in outcomes:
        append_line(root / "results" / PROBES_FILE, outcome)
    return box


def done(*episodes: str) -> dict[str, Any]:
    return outcome("done", "done", episodes, len(episodes)).model_dump(mode="json")


def test_a_done_outcome_bound_to_its_request_and_rows_is_shown(
    tmp_path: Path, export: Path
) -> None:
    rows = [(e, "p0001", r, k) for e, r, k in FOUR]
    probed = probe_export(tmp_path, export, rows, {e for e, _, _ in FOUR})
    box = results_box(tmp_path, probed, [done(*(e for e, _, _ in FOUR))])
    shown = answer(box.call("probe_results", {}))
    assert shown[0]["status"] == "done" and len(shown[0]["episode_ids"]) == 4
    assert answer(box.call("experiment_budget", {}))["probe_attempts_used"] == 4
    box.close()


@pytest.mark.parametrize(
    ("case", "rows", "steps", "outcomes"),
    [
        # Two completed episodes for a request of four attempts.
        ("count", FOUR[:2], {"e000901", "e000902"}, [done("e000901", "e000902")]),
        # An episode the export does not hold.
        ("missing row", FOUR[:3], {e for e, _, _ in FOUR}, [done(*(e for e, _, _ in FOUR))]),
        # An episode without telemetry.
        ("no steps", FOUR, {"e000901", "e000902", "e000903"}, [done(*(e for e, _, _ in FOUR))]),
        # Rows on a robot the request did not name.
        ("robot", [*FOUR[:3], ("e000904", "r03", 1)], {e for e, _, _ in FOUR},
         [done(*(e for e, _, _ in FOUR))]),
        # Probe rows no outcome names.
        ("unnamed rows", FOUR, {e for e, _, _ in FOUR}, []),
        # Contradictory duplicates: refused and done for one probe.
        ("contradiction", FOUR, {e for e, _, _ in FOUR},
         [{"probe_id": "p0001", "status": "refused", "episode_ids": [],
           "message": "probe_not_runnable", "attempts_executed": 0, "attempts_reserved": 0},
          done(*(e for e, _, _ in FOUR))]),
        # A refused probe whose episodes are exported anyway.
        ("refused with rows", FOUR, {e for e, _, _ in FOUR},
         [{"probe_id": "p0001", "status": "refused", "episode_ids": [],
           "message": "probe_not_runnable", "attempts_executed": 0, "attempts_reserved": 0}]),
        # A failed probe that claims more attempts than it could have run.
        ("too many", [], set(),
         [{"probe_id": "p0001", "status": "failed", "episode_ids": [],
           "message": "execution_failed", "attempts_executed": 5, "attempts_reserved": 0}]),
        # A failed probe that refunds part of its request.
        ("refund", [], set(),
         [{"probe_id": "p0001", "status": "failed", "episode_ids": [],
           "message": "execution_failed", "attempts_executed": 1, "attempts_reserved": 1}]),
    ],
)  # fmt: skip
def test_an_outcome_that_does_not_match_is_refused_as_a_whole(
    tmp_path: Path,
    export: Path,
    case: str,
    rows: list[tuple[str, str, int]],
    steps: set[str],
    outcomes: list[dict[str, Any]],
) -> None:
    probed = probe_export(tmp_path, export, [(e, "p0001", r, k) for e, r, k in rows], steps)
    box = results_box(tmp_path, probed, outcomes)
    for tool in ("probe_results", "experiment_budget"):  # never a refund beside a completion
        result = box.call(tool, {})
        assert result.is_error, (case, tool)
        assert "probe results failed public validation" in result.content[0]["text"]
    assert box.call("request_probe", request(robots=["r01"], repeats=1)).is_error
    box.close()


def test_an_outcome_is_visible_only_in_a_later_session(tmp_path: Path, export: Path) -> None:
    rows = [(e, "p0001", r, k) for e, r, k in FOUR]
    probed = probe_export(tmp_path, export, rows, {e for e, _, _ in FOUR})
    box = results_box(tmp_path, probed, [done(*(e for e, _, _ in FOUR))], session_of_probe=2)
    assert box.call("probe_results", {}).is_error
    box.close()


def test_an_earlier_probe_without_an_outcome_is_refused(tmp_path: Path, export: Path) -> None:
    box = results_box(tmp_path, export, [])
    assert box.call("probe_results", {}).is_error
    box.close()


def test_a_probe_refused_before_it_ran_is_free(tmp_path: Path, export: Path) -> None:
    refused = outcome("refused", "probe_not_runnable").model_dump(mode="json")
    box = results_box(tmp_path, export, [refused])
    budget = answer(box.call("experiment_budget", {}))
    assert budget["probe_attempts_used"] == 0 and budget["probes_used"] == 0
    shown = answer(box.call("probe_results", {}))
    assert shown[0]["status"] == "refused" and shown[0]["episode_ids"] == []
    assert shown[0]["attempts_charged"] == 0 and "budget was returned" in shown[0]["message"]
    box.close()


def test_a_failed_probe_keeps_its_request_charged_executed_and_reserved_apart(
    tmp_path: Path, export: Path
) -> None:
    failed = outcome("failed", "execution_failed", (), 1, 3).model_dump(mode="json")
    box = results_box(tmp_path, export, [failed])
    budget = answer(box.call("experiment_budget", {}))
    assert budget["probe_attempts_used"] == 4 and budget["probes_used"] == 1
    shown = answer(box.call("probe_results", {}))
    assert shown[0]["status"] == "failed" and shown[0]["episode_ids"] == []
    assert (shown[0]["attempts_charged"], shown[0]["attempts_executed"]) == (4, 1)
    assert shown[0]["attempts_reserved"] == 3 and "stay charged" in shown[0]["message"]
    box.close()


def test_an_outcome_s_message_matches_its_status() -> None:
    with pytest.raises(ValidationError):
        outcome("failed", "done")
    with pytest.raises(ValidationError):
        outcome("refused", "execution_failed")
    with pytest.raises(ValidationError):
        outcome("done", "done", ("e000901", "e000901"), 2)


def test_the_request_limit_counts_probe_requests(box: ToolBox) -> None:
    small = request(robots=["r01"], repeats=1)
    for _ in range(box.limits.max_probe_requests):
        assert answer(box.call("request_probe", small))["accepted"]
    last = answer(box.call("request_probe", small))
    assert not last["accepted"] and "probe requests" in last["problems"][0]


def test_the_tool_description_states_the_bounds(box: ToolBox) -> None:
    described = {t.name: t.description for t in box.mcp_tools()}
    text = described["request_probe"]
    assert "0.05 m" in text and "0.5 m/s" in text and "400 steps" in text
    assert "one attempt of the experiment budget" in text
