"""Probes on the engine (docs/design/loop.md, section 10), on CPU: the probe player in the toy
cell, through the loop's probe runs, export, round engine and round sessions, with a lasting
command-path condition at one site of the toy fleet (`fleetlab.loop.probe_toy`).

What is pinned here: the timing convention of `probe_steps` (contracts.log), that a
command-path condition shows as lag in the measured response and nowhere as a value, that
the probe tables pass the export's leak audit (and that the audit reads them), that a run
which breaks off charges what it started and shows nothing, and that a scripted agent can
compose a probe, read it back with `query` across two sessions of a round, and keep its
definition in a learned skill for a later investigation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fleetlab.agents.audit import AuditContext
from fleetlab.agents.sessions import AgentSpec, Analysis
from fleetlab.agents.toolenv import project_tool_env
from fleetlab.contracts.log import SCHEMA
from fleetlab.contracts.probes import ProbeRequest, QueuedProbe
from fleetlab.log import LogReader
from fleetlab.log.leaks import audit_export
from fleetlab.loop.driver import AttemptBudget
from fleetlab.loop.experiments import RoundEngine
from fleetlab.loop.export import export_round
from fleetlab.loop.layout import LoopLayout, layout_for_world
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.probe_toy import LINK_ROBOTS, write_link_toy_world
from fleetlab.loop.probes import FAILED_FILE, probe_root, probe_runs, run_probes
from fleetlab.loop.rounds import ROUND_TOOLS, run_round_sessions
from fleetlab.loop.scripted import ScriptedAgent
from fleetlab.loop.state import Ledger
from fleetlab.loop.world import LoopWorld

HOLD, MOVE_AT, AMPLITUDE = 5, 5, 0.03
CLEAN, LINKED = "r01", LINK_ROBOTS[1]  # r04: the link alone (r03 also has a toy delay)


def step_request(robots: list[str], repeats: int = 1, name: str = "step-lag-x") -> dict[str, Any]:
    return {
        "name": name,
        "purpose": "how long after a commanded step the hand starts to follow it",
        "expectation": "the hand starts within a step or two on a robot with a direct link",
        "robots": robots,
        "repeats": repeats,
        "segments": [
            {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": HOLD,
             "period_steps": None},
            {"primitive": "step", "axis": "x", "amplitude_m": AMPLITUDE, "duration_steps": 15,
             "period_steps": None},
            {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 5,
             "period_steps": None},
        ],
    }  # fmt: skip


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[LoopWorld, Any, LoopLayout]:
    root = tmp_path_factory.mktemp("link-toy")
    world, setup = write_link_toy_world(root / "world-root")
    return world, setup, layout_for_world(world.spec)


def queued(probe_id: str, request: dict[str, Any], session: int = 1) -> QueuedProbe:
    return QueuedProbe(
        probe_id=probe_id, session=session, request=ProbeRequest.model_validate(request)
    )


def first_movement(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(the first row whose command differs from row 0's position, the first row whose
    measured position differs from row 0's)."""
    x0 = rows[0]["eef_pos"][0]
    commanded = next(
        r["step"] for r in rows if r["target_pos"] and abs(r["target_pos"][0] - x0) > 1e-6
    )
    moved = next(r["step"] for r in rows if abs(r["eef_pos"][0] - x0) > 1e-9)
    return commanded, moved  # fmt: skip


def probe_rows(export: Path, robot: str) -> list[dict[str, Any]]:
    with LogReader(export) as reader:
        rows: list[dict[str, Any]] = reader.query(
            "SELECT s.* FROM probe_steps s JOIN probe_episodes e USING (episode_id) "
            "WHERE e.robot_id = ? ORDER BY s.step",
            [robot],
        ).to_pylist()
    return rows


def test_the_timing_convention_on_a_run_without_delay(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    world, setup, layout = toy
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")
    outcomes = run_probes(
        world, layout, 1, ledger, paths, setup, "a1", 1, [queued("p0001", step_request([CLEAN]))]
    )
    assert [(o.status, o.attempts_executed) for o in outcomes] == [("done", 1)]
    export = export_round(world, layout, 1, ledger, paths, tmp_path / "export", analyses=["a1"])
    rows = probe_rows(export, CLEAN)
    # Row k: measured, then command k; the last row (step 25) has no command.
    assert [r["step"] for r in rows] == list(range(26))
    assert [r["time_s"] for r in rows] == [k / 20 for k in range(26)]
    assert rows[-1]["target_pos"] is None and rows[-1]["segment"] is None
    assert [r["segment"] for r in rows[:-1]] == [0] * 5 + [1] * 15 + [2] * 5
    commanded, moved = first_movement(rows)
    # The step's command is issued at row 5 (the segment's first step); with nothing on the
    # command path it shows first in the measurement of row 6.
    assert commanded == MOVE_AT
    assert moved == commanded + 1
    assert rows[commanded]["target_pos"][0] == pytest.approx(rows[0]["eef_pos"][0] + AMPLITUDE)
    with LogReader(export) as reader:
        segments = reader.query("SELECT * FROM probe_segments ORDER BY segment").to_pylist()
        episodes = reader.query("SELECT * FROM probe_episodes").to_pylist()
        regular = reader.query("SELECT episode_id FROM episodes").column("episode_id").to_pylist()
    assert [s["first_step"] for s in segments] == [0, 5, 20]
    assert episodes[0]["probe_id"] == "p0001" and episodes[0]["steps"] == 25
    assert episodes[0]["episode_id"] not in regular  # a probe is no task attempt
    dictionary = (export / "data_dictionary.md").read_text()
    assert "## probe_steps" in dictionary and "row k + 1" in dictionary
    # The run proved its chains: the probe episode replays bit for bit.
    run = probe_root(paths, 1, "a1") / "session-01"
    assert json.loads((run / "run.json").read_text())["chain_ok"] is True


def test_a_command_path_condition_shows_as_lag_and_as_nothing_else(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    world, setup, layout = toy
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")
    request = step_request([CLEAN, LINKED], repeats=2)
    outcomes = run_probes(
        world, layout, 1, ledger, paths, setup, "a1", 1, [queued("p0001", request)]
    )
    assert outcomes[0].status == "done" and outcomes[0].attempts_executed == 4
    # The export's own leak audit runs here, over the probe tables and the probe runs' truth.
    export = export_round(world, layout, 1, ledger, paths, tmp_path / "export", analyses=["a1"])
    lag = {}
    for robot in (CLEAN, LINKED):
        rows = probe_rows(export, robot)
        commanded, moved = first_movement(rows[: len(rows) // 2])  # the first repeat
        lag[robot] = moved - commanded
    assert lag[CLEAN] == 1
    assert lag[LINKED] > lag[CLEAN] + 1  # measured behaviour, larger on the linked robot
    # The tables hold the declared columns only, and nothing names the condition.
    names = {f.name for f in pq.read_schema(export / SCHEMA["probe_episodes"].path)}
    assert names == {c.name for c in SCHEMA["probe_episodes"].columns}
    text = " ".join(
        str(pq.read_table(p).to_pylist()) for p in (export / "log").rglob("probe_*.parquet")
    )
    assert "remote" not in text and "latency" not in text and "delay" not in text


def test_the_audit_reads_the_probe_tables(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    world, setup, layout = toy
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")
    probes = [queued("p0001", step_request([LINKED]))]
    run_probes(world, layout, 1, ledger, paths, setup, "a1", 1, probes)
    export = export_round(world, layout, 1, ledger, paths, tmp_path / "export", analyses=["a1"])
    # A planted truth term in a probe table is caught.
    doctored = tmp_path / "doctored"
    shutil.copytree(export, doctored)
    path = doctored / SCHEMA["probe_segments"].path
    data = pq.read_table(path).to_pylist()
    data[0]["axis"] = "remote_link_latency"
    pq.write_table(pa.Table.from_pylist(data, schema=pq.read_schema(path)), path)
    runs = [world.observed_run, *probe_runs(paths, 1, ["a1"])]
    report = audit_export(doctored, runs, fleet=world.model)
    assert not report.clean
    assert any("probe_segments" in hit.where for hit in report.hits)


def test_a_run_that_breaks_off_charges_what_it_started_and_shows_nothing(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import fleet

    world, setup, layout = toy
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")
    real = fleet.run

    def breaking(config: Any, jobs: Any) -> Any:
        real(config, list(jobs)[:2])  # two of four probe episodes, then the run is lost
        raise RuntimeError("worker lost")

    monkeypatch.setattr(fleet, "run", breaking)
    request = step_request([CLEAN, "r02"], repeats=2)
    outcomes = run_probes(
        world, layout, 1, ledger, paths, setup, "a1", 1, [queued("p0001", request)]
    )
    assert len(outcomes) == 1
    failed = outcomes[0]
    assert (failed.status, failed.message, failed.episode_ids) == ("failed", "execution_failed", [])
    # Two left records (known executed work); the other two cannot be confirmed either way:
    # reserved, still charged, never counted as executed.
    assert (failed.attempts_executed, failed.attempts_reserved) == (2, 2)
    run = probe_root(paths, 1, "a1") / "session-01"
    assert (run / FAILED_FILE).is_file() and probe_runs(paths, 1, ["a1"]) == []
    monkeypatch.setattr(fleet, "run", real)
    # Reported failed once, it stays failed: nothing completes it later or exports it.
    again = run_probes(world, layout, 1, ledger, paths, setup, "a1", 1, [queued("p0001", request)])
    assert again == outcomes
    export = export_round(world, layout, 1, ledger, paths, tmp_path / "export", analyses=["a1"])
    assert not (export / SCHEMA["probe_episodes"].path).exists()


def test_a_worker_lost_mid_motion_before_any_record_keeps_the_attempt_charged(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost worker that was playing a probe leaves no record (the sink writes after the
    episode): the runner cannot tell it from one that never started, so the attempt is
    reserved and charged, never refunded and never called executed."""
    from fleetlab.engine import fleet

    world, setup, layout = toy
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")

    def lost(config: Any, jobs: Any) -> Any:
        raise RuntimeError("worker lost mid-motion, nothing written")

    monkeypatch.setattr(fleet, "run", lost)
    probes = [queued("p0001", step_request([LINKED]))]
    (failed,) = run_probes(world, layout, 1, ledger, paths, setup, "a1", 1, probes)
    assert (failed.status, failed.attempts_executed, failed.attempts_reserved) == ("failed", 0, 1)
    assert failed.charged == 1


# A round of the driver: a probe in the first session, its results in the second.

LAG_SQL = """
WITH s AS (
  SELECT e.robot_id, p.episode_id, p.step, p.eef_pos[1] AS x, p.target_pos[1] AS tx
  FROM probe_steps p JOIN probe_episodes e USING (episode_id)
  WHERE e.probe_id = $probe AND e.repeat = 0),
start AS (SELECT episode_id, x AS x0 FROM s WHERE step = 0),
cmd AS (SELECT episode_id, min(step) AS k FROM s JOIN start USING (episode_id)
        WHERE abs(tx - x0) > 0.001 GROUP BY episode_id),
moved AS (SELECT episode_id, min(step) AS m FROM s JOIN start USING (episode_id)
          WHERE abs(x - x0) > 0.000001 GROUP BY episode_id)
SELECT DISTINCT s.robot_id, moved.m - cmd.k AS lag_steps
FROM cmd JOIN moved USING (episode_id) JOIN s USING (episode_id)
ORDER BY s.robot_id
"""


def _data(result: Any) -> Any:
    text = "\n".join(p["text"] for p in result.content if p.get("type") == "text")
    assert not result.is_error, text
    return json.loads(text)


def probe_session(box: Any) -> str:
    receipt = _data(box.call("request_probe", step_request([CLEAN, LINKED])))
    assert receipt["accepted"], receipt["problems"]
    assert (receipt["attempts"], receipt["probe_attempts_used"]) == (2, 2)
    _data(box.call("memory_write", {"notes": f"probe {receipt['probe_id']}: step lag"}))
    return "waiting"


def reading_session(box: Any) -> str:
    notes = _data(box.call("memory_read", {}))["notes"]
    probe_id = re.search(r"probe (p[0-9]+)", notes)[1]  # type: ignore[index]
    results = _data(box.call("probe_results", {}))
    assert [(r["probe_id"], r["status"]) for r in results] == [(probe_id, "done")]
    budget = _data(box.call("experiment_budget", {}))
    assert (budget["probe_attempts_used"], budget["task_attempts_used"]) == (2, 0)
    sql = LAG_SQL.replace("$probe", f"'{probe_id}'")
    lag = {row[0]: row[1] for row in _data(box.call("query", {"sql": sql}))["rows"]}
    assert lag[LINKED] > lag[CLEAN]
    definition = {k: v for k, v in step_request([]).items() if k != "robots"}
    kept = _data(
        box.call(
            "write_learned_skill",
            {
                "name": "command-lag-probe",
                "description": "measure how far a robot's motion trails its commands",
                "applies_when": "robots of one site fail more often in motions that need "
                "timely corrections, and their records show nothing that changed",
                "parameters": [
                    {"name": "probe", "meaning": "the id of the step probe defined below"},
                    {"name": "robot", "meaning": "a robot under suspicion"},
                ],
                "evidence": [
                    {
                        "claim": "steps from a commanded step to the first measured movement",
                        "sql": LAG_SQL,
                        "result": f"one step on a robot elsewhere, {lag[LINKED]} at the site "
                        "under suspicion",
                    }
                ],
                "alternatives": [
                    {
                        "explanation": "a slow arm controller rather than late commands",
                        "how_to_tell": "a slow controller rounds the response; late commands "
                        "shift it whole",
                    }
                ],
                "next_experiment": "Request this probe with request_probe on {robot} and on "
                "a robot of another site, then run the evidence query with its id: "
                + json.dumps(definition),
                "check_with": [{"name": "probe", "value": probe_id}],
            },
        )
    )
    assert kept["accepted"], kept["problems"]
    decision = {
        "adopt": [],
        "replace": [],
        "undo": [],
        "predicted_set_gain": {"low": 0.0, "mid": 0.0, "high": 0.0},
        "expectations": ["the probe's lag stays as measured"],
        "stop": False,
        "ranked_candidates": [],
    }
    assert _data(box.call("submit_decision", decision))["accepted"]
    return "done"


@pytest.fixture
def agent_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    yield tmp_path


def test_a_round_composes_a_probe_reads_it_back_and_keeps_its_definition(
    agent_root: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    world, setup, layout = toy
    tmp_path = agent_root
    ledger, paths = Ledger.empty(world.world_id), LoopPaths(tmp_path / "loop")
    analysis = Analysis.create(world.world_id, base=tmp_path / "analyses")
    export = export_round(world, layout, 1, ledger, paths, tmp_path / "export-0")
    engine = RoundEngine(world, layout, 1, ledger, paths, setup)
    agent = ScriptedAgent([probe_session, reading_session], run_root=tmp_path / "scripted")
    budget = AttemptBudget(attempts=40)
    days = layout.round(1).experiment
    with mock.patch("fleetlab.agents.sessions.run_agent", agent):
        outcome = run_round_sessions(
            analysis,
            export,
            AgentSpec("claude", "scripted"),
            "system",
            "the task",
            project_tool_env(tmp_path / "env"),
            engine,
            round_number=1,
            limits=budget.limits(first_session=1),
            attempts=budget.attempts,
            max_sessions=3,
            audit=AuditContext((), None, None, (days.first, days.last)),
            run_root=tmp_path / "runs",
            tools=(*ROUND_TOOLS, "request_probe", "probe_results"),
        )
    assert [s.state for s in outcome.sessions] == ["waiting", "done"]
    assert outcome.decision is not None and outcome.stopped is None
    # The round driver recorded the probe's outcome (so the second session could read it).
    recorded = analysis.probe_outcomes()
    assert [(o.status, o.attempts_executed) for o in recorded.values()] == [("done", 2)]
    # The learned skill carries the probe's definition: a later investigation can request it
    # again on its own robots.
    learned = [
        json.loads(line) for line in (analysis.outbox / "learned.jsonl").read_text().splitlines()
    ]
    text = learned[-1]["draft"]["next_experiment"]
    carried = json.loads(text[text.index('{"name"') :])
    ProbeRequest.model_validate({**carried, "robots": ["r02"]})
    assert carried["segments"] == step_request([])["segments"]


def test_the_probe_check_reads_each_commanded_step_s_lag(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    from fleetlab.loop.probes import probe_check

    world, setup, layout = toy
    request = step_request([CLEAN, LINKED])
    request["segments"].append(
        {"primitive": "step", "axis": "x", "amplitude_m": 0.0, "duration_steps": 15,
         "period_steps": None}
    )  # fmt: skip
    summary = probe_check(
        world,
        layout,
        Ledger.empty(world.world_id),
        LoopPaths(tmp_path / "loop"),
        setup,
        "probe-check",
        request,
        tmp_path / "export",
    )
    assert [o["status"] for o in summary["outcomes"]] == ["done"]
    by_robot = {e["robot_id"]: e["edges"] for e in summary["episodes"]}
    assert [edge["issued_at"] for edge in by_robot[CLEAN]] == [5, 25]  # out and back
    assert [edge["first_movement_rows"] for edge in by_robot[CLEAN]] == [1, 1]
    linked = [edge["first_movement_rows"] for edge in by_robot[LINKED]]
    assert all(lag > 2 for lag in linked)
    assert "target_pos" in summary["columns"]["probe_steps"]


# The opt-in probe path of an engagement (Engagement.probes, `--probes`).


def test_an_engagement_offers_probes_only_when_switched_on(tmp_path: Path) -> None:
    from fleetlab.loop.driver import run_rounds
    from fleetlab.loop.probes import PROBE_TASK, with_probes
    from fleetlab.loop.rounds import round_task
    from fleetlab.loop.scripted import decide_step
    from fleetlab.toolserver.decisions import PublicLedger

    plain = round_task(
        round_number=1,
        rounds=2,
        operation=[(1, (0, 1))],
        experiment_days=(2, 2),
        budget_text="40 robot attempts",
        ledger=PublicLedger(round=1, changes=[]),
    )
    assert "request_probe" not in plain and "probe" not in plain.lower()
    assert PROBE_TASK in with_probes(plain) and "learned skill" in with_probes(plain)
    assert "learned skill" not in with_probes(plain, learning=False)

    previous = os.environ.get("ROBOTS_AGENT_ROOT")
    os.environ["ROBOTS_AGENT_ROOT"] = str(tmp_path / "agent-root")
    try:
        world, setup = write_link_toy_world(tmp_path / "world-root")
        agent = ScriptedAgent(
            [probe_session, reading_session, decide_step([], stop=True, alternatives=0)],
            run_root=tmp_path / "scripted",
        )
        record = run_rounds(
            world,
            tmp_path / "out",
            agent=AgentSpec("codex", "scripted"),
            budget=AttemptBudget(attempts=40, per_arm=4),
            max_sessions=3,
            scripted=agent,
            heavy=False,
            setup=setup,
            env=project_tool_env(tmp_path / "env"),
            probes=True,
        )
    finally:
        if previous is None:
            os.environ.pop("ROBOTS_AGENT_ROOT", None)
        else:
            os.environ["ROBOTS_AGENT_ROOT"] = previous
    first = record["rounds"][0]
    assert PROBE_TASK in first["task"]
    assert [s["state"] for s in first["sessions"]] == ["waiting", "done"]
    assert "probe_results" in agent.seen[1]["prompt"]  # the continuation names probes
    # The engagement record: the probe-only session's charged attempts, apart from tasks.
    costs = first["costs"]
    assert costs["probe_requests"] == 1 and costs["probe_attempts"] == 2
    assert (costs["probe_attempts_executed"], costs["probe_attempts_reserved"]) == (2, 0)
    assert costs["probes"][0]["status"] == "done" and costs["arm_attempts"] == 0
    assert record["rounds"][1]["costs"]["probe_attempts"] == 0  # round 2 requested none
    # The simulation summary: the probe run's episodes and replays, apart from the arms.
    simulated = record["simulated"]
    assert simulated["probes"]["run_jobs"] == 2 and simulated["experiments"]["run_jobs"] == 0
    assert simulated["probes"]["simulated_confirmed"] >= 2
    # The learned skill that carries the probe passed the host's checks and was frozen.
    learned = record["learned"]
    assert learned["kept"] == ["command-lag-probe"] and learned["library"] is not None, learned
    kept = Path(learned["library"]) / "skills" / "command-lag-probe" / "SKILL.md"
    assert '"segments"' in kept.read_text() and "probe_steps" in kept.read_text()


def test_the_probe_check_of_round_two_runs_under_the_retained_state(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    """Round 2's probe under a ledger that took back r03's own command delay: its export
    holds round 1's probe and round 2's (the round padding of `RoundEngine.analyses`), and
    the measured response on r03 is quicker in round 2, while the lasting link stays."""
    from fleetlab.contracts.actions import Action
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import ConfigKey, Verb
    from fleetlab.loop.operate import operate
    from fleetlab.loop.probes import probe_check

    world, setup, layout = toy
    paths = LoopPaths(tmp_path / "loop")
    original = Ledger.empty(world.world_id)
    request = step_request(["r03"])
    first = probe_check(world, layout, original, paths, setup, "pc", request, tmp_path / "export-1")
    revert = Action(
        verb=Verb.revert,
        target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
        value=None,
    )
    retained = original.adopt(1, layout.round(2).ops.first, [revert])
    operate(world, layout, 2, retained, paths, setup)
    second = probe_check(
        world,
        layout,
        retained,
        paths,
        setup,
        "pc",
        request,
        tmp_path / "export-2",
        round_number=2,
        earlier=["pc"],
    )
    by_probe = {e["probe_id"]: e for e in second["episodes"]}
    assert sorted(by_probe) == ["p0001", "p0002"]  # round 1's probe stays in the history
    assert by_probe["p0002"]["day"] in range(
        layout.round(2).experiment.first, layout.round(2).experiment.last + 1
    )
    before = first["episodes"][0]["edges"][0]["first_movement_rows"]
    after = by_probe["p0002"]["edges"][0]["first_movement_rows"]
    assert by_probe["p0001"]["edges"][0]["first_movement_rows"] == before
    assert 1 < after < before  # the delay the ledger took back is gone, the link is not


def test_a_probe_only_queue_is_costed_apart_in_the_heavy_queue_record(
    tmp_path: Path, toy: tuple[LoopWorld, Any, LoopLayout]
) -> None:
    """`HeavyRoundEngine` with its heavy step run in process: the queue record names the
    probe, its planned and charged attempts and the probe run's simulations apart from
    the arms', the queue log line counts the probe attempts, and `queue_costs` and
    `WorldLoop.simulated` report them apart."""
    from fleetlab.loop.costs import probe_costs, queue_costs
    from fleetlab.loop.driver import _simulated_text
    from fleetlab.loop.world_loop import HeavySteps, WorldLoop

    world, setup, layout = toy
    log = tmp_path / "queue.log"

    class InProcess(HeavySteps):
        def run(
            self, step: str, purpose: str, attempts: str, args: Any, simulated: Any = None
        ) -> Any:
            assert step == "queue"
            given = dict(zip(args[::2], args[1::2], strict=True))
            engine = RoundEngine(
                world,
                layout,
                int(given["--round"]),
                Ledger.load(given["--ledger"]),
                loop.paths,
                setup,
                analysis_id=given["--analysis"],
            )
            reply = engine.run_queue(Path(given["--queue"]), None)
            Path(given["--reply"]).write_text(reply.model_dump_json())
            counted: dict[str, Any] = dict(simulated())
            record: dict[str, Any] = {"step": step, "simulated": counted}
            self.records.append(record)
            log.write_text(_simulated_text(counted) + "\n")
            return record

    heavy = InProcess(world.root, tmp_path / "loop", tmp_path / "loop" / "setup.json")
    loop = WorldLoop.create(world, layout, tmp_path / "loop", setup, heavy)
    analysis = Analysis.create(world.world_id, base=tmp_path / "analyses")
    ledger = loop.empty_ledger()
    export = loop.export(1, ledger, analysis.analysis_id)
    engine = loop.engine(1, ledger, analysis.analysis_id)
    agent = ScriptedAgent([probe_session, reading_session], run_root=tmp_path / "scripted")
    budget = AttemptBudget(attempts=40)
    days = layout.round(1).experiment
    with mock.patch("fleetlab.agents.sessions.run_agent", agent):
        outcome = run_round_sessions(
            analysis, export, AgentSpec("claude", "scripted"), "system", "the task",
            project_tool_env(tmp_path / "env"), engine, round_number=1,
            limits=budget.limits(first_session=1), attempts=budget.attempts, max_sessions=3,
            audit=AuditContext((), None, None, (days.first, days.last)),
            run_root=tmp_path / "runs", tools=(*ROUND_TOOLS, "request_probe", "probe_results"),
        )  # fmt: skip
    assert [s.state for s in outcome.sessions] == ["waiting", "done"]
    (record,) = engine.records
    assert record["experiments"] == [] and record["planned_attempts"] == 0
    assert record["probes"] == ["p0001"] and record["planned_probe_attempts"] == 2
    assert record["charged_probe_attempts"] == 2 and record["delivered_attempts"] == 0
    assert record["simulated"]["run_jobs"] == 0 and record["simulated_probes"]["run_jobs"] == 2
    assert "probes:2(probe-attempts)" in log.read_text()
    summary = queue_costs(engine.records)
    assert summary["simulated_arm_attempts"] == 0 and summary["simulated_probe_attempts"] == 2
    assert summary["simulated_probe_jobs"] >= 2
    assert loop.simulated()["probes"]["run_jobs"] == 2
    assert probe_costs(analysis)["probe_attempts"] == 2


def test_the_queue_log_text_counts_probe_simulations_as_the_json_does() -> None:
    from fleetlab.loop.driver import _simulated_text

    arms = {"run_jobs": 0, "replays": 0, "failed_confirmed": 0, "unconfirmed": 0}
    probes = {"run_jobs": 2, "replays": 2, "failed_confirmed": 1, "partial_jobs": 0,
              "unconfirmed": 0, "simulated_confirmed": 5}  # fmt: skip
    text = _simulated_text({**arms, "probes": probes})
    part = text.split(";probes:", 1)[1]
    assert part == "2(probe-attempts)+2(chain-sample-replays)+1(failed-simulations)"
    assert sum(int(n) for n in re.findall(r"(\d+)\(", part)) == probes["simulated_confirmed"]
    # Failed-only probe jobs do not disappear; partial and unknown work shows where present.
    failed_only = {**probes, "run_jobs": 0, "replays": 0, "simulated_confirmed": 1}
    assert _simulated_text({**arms, "probes": failed_only}).endswith(
        ";probes:0(probe-attempts)+0(chain-sample-replays)+1(failed-simulations)"
    )
    broken = {**probes, "partial_jobs": 1, "unconfirmed": 2}
    assert _simulated_text({**arms, "probes": broken}).endswith(
        "+1(failed-simulations)+1(partial)+2(unconfirmed-assignments)"
    )
    assert ";probes:" not in _simulated_text({**arms, "probes": dict.fromkeys(probes, 0)})


def test_the_round_continuation_names_probes_only_with_the_switch() -> None:
    from fleetlab.loop.rounds import round_continuation

    plain = round_continuation(1, 2, "the task")
    assert "probe" not in plain and "experiment_results" in plain
    probed = round_continuation(1, 2, "the task", probes=True)
    assert "probe_results" in probed and "experiment_results" in probed
    assert probed.endswith("The task:\nthe task")
