"""Two rounds of the loop on the toy world, end to end on the CPU (the engine's toy cell):
round 1's export and arms, the adoption, round 2's operation under the ledger, round 2's
arms from the current state, the exports, and the reserved evaluation."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from fleetlab.agents.sessions import ExperimentQueue, QueuedExperiment
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.model import Fleet, effective_config
from fleetlab.contracts.tools import ExperimentArm, ExperimentRequest
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.log import LogReader
from fleetlab.loop.evaluate import evaluate, reproduces_as_run
from fleetlab.loop.experiments import RoundEngine
from fleetlab.loop.export import export_round
from fleetlab.loop.layout import layout_for_world
from fleetlab.loop.operate import operate
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.state import Ledger
from fleetlab.loop.toy import write_toy_world

DELAY = ConfigKey.control_action_delay_steps


def recalibrate(robot_id: str) -> Action:
    return Action(
        verb=Verb.recalibrate, target=Target(scope=robot_id, component=Component.scene_camera)
    )


def revert(scope: str) -> Action:
    return Action(verb=Verb.revert, target=Target(scope=scope, config_key=DELAY))


def queue(
    export: Path, analysis: str, experiment: str, arms: list[list[Action]], robots: list[str]
) -> ExperimentQueue:
    request = ExperimentRequest(
        arms=[ExperimentArm(actions=a) for a in arms],
        robots=robots,
        episodes_per_arm=4,
        paired_starts=True,
        power_note="toy",
    )
    return ExperimentQueue(
        world_id="w01",
        analysis_id=analysis,
        session=1,
        created_at=datetime.datetime(2026, 10, 13).isoformat(),
        export=str(export),
        experiments=[QueuedExperiment(experiment_id=experiment, session=1, request=request)],
    )


def identities(run: Path) -> dict[str, EpisodeIdentity]:
    rows = pq.read_table(run / "truth" / "identities.parquet").to_pylist()
    return {r["episode_id"]: EpisodeIdentity.model_validate_json(r["identity_json"]) for r in rows}


def episodes(export: Path) -> list[dict[str, Any]]:
    with LogReader(export) as reader:
        rows: list[dict[str, Any]] = reader.query(
            "select episode_id, robot_id, day, experiment_id, outcome from episodes"
        ).to_pylist()
    return rows


@pytest.fixture(scope="module")
def loop(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("loop")
    world, setup = write_toy_world(root / "world-root")
    layout = layout_for_world(world.spec)
    paths = LoopPaths(root / "loop")
    paths.root.mkdir(parents=True)
    paths.layout.write_text(layout.model_dump_json(indent=1))
    l0 = Ledger.empty(world.world_id)
    first = export_round(world, layout, 1, l0, paths, paths.exports(1) / "start")
    engine1 = RoundEngine(world, layout, 1, l0, paths, setup)
    q1 = queue(first, "w01-r1", "x0001", [[], [recalibrate("r02")]], ["r02"])
    q1_file = root / "q1.json"
    q1_file.write_text(q1.model_dump_json())
    reply1 = engine1.run_queue(q1_file, q1)
    l1 = l0.adopt(1, layout.adoption_day(1), adopt=[recalibrate("r02"), revert("r03")])
    ops2 = operate(world, layout, 2, l1, paths, setup)
    second = export_round(
        world, layout, 2, l1, paths, paths.exports(2) / "start", analyses=["w01-r1"]
    )
    engine2 = RoundEngine(world, layout, 2, l1, paths, setup, earlier=["w01-r1"])
    q2 = queue(second, "w01-r2", "x0002", [[], [revert("r03")]], ["r03"])
    q2_file = root / "q2.json"
    q2_file.write_text(q2.model_dump_json())
    reply2 = engine2.run_queue(q2_file, q2)
    summary = evaluate(world, layout, {"original": l0, "round1": l1}, paths, setup)
    inside = LoopPaths(root / "inside")
    evaluate(world, layout, {"original": l0}, inside, setup, days=[1])
    return {
        "world": world,
        "layout": layout,
        "paths": paths,
        "l1": l1,
        "first": first,
        "reply1": reply1,
        "ops2": ops2,
        "second": second,
        "reply2": reply2,
        "summary": summary,
        "inside": inside,
    }


def test_round_two_operates_the_worlds_own_work_under_the_ledger(loop: dict[str, Any]) -> None:
    world, ops2 = loop["world"], loop["ops2"]
    ran = identities(ops2)
    as_run = {i.episode_id: i for i in world.identities if i.day == 3}
    assert set(ran) == set(as_run)
    for episode_id, identity in ran.items():
        original = as_run[episode_id]
        same = ("robot_id", "day", "episode_index", "task_index", "start_state", "scene_seed")
        assert all(getattr(identity, k) == getattr(original, k) for k in same)
        kinds = {f.kind for f in identity.faults}
        if identity.robot_id == "r02":
            assert "toy_camera_shift" not in kinds and any(
                f.kind == "toy_camera_shift" for f in original.faults
            )
        if identity.robot_id == "r03":
            assert "toy_delay" not in kinds
    # The records: the ledger's own entries, as the fleet's.
    public = Fleet.model_validate(
        __import__("json").loads((ops2 / "log" / "description.json").read_text())["fleet"]
    )
    assert effective_config(public, "r03")[DELAY] == 0
    ops = [c for c in public.change_log if c.author.value == "ops" and c.day == 3]
    assert [(c.scope, c.key, c.old, c.new) for c in ops] == [("r03", DELAY, 3, 0)]
    assert any(m.day == 3 and m.robot_id == "r02" for m in public.maintenance_log)
    assert public.site("s1").conditions[ConditionKey.clutter] == 0.0


def test_arms_run_from_the_current_state(loop: dict[str, Any]) -> None:
    paths = loop["paths"]
    reply1, reply2 = loop["reply1"], loop["reply2"]
    assert [r.status for r in reply1.results] == ["done"]
    assert [r.status for r in reply2.results] == ["done"]
    arms1 = identities(paths.experiments(1, "w01-r1") / "session-01")
    arms2 = identities(paths.experiments(2, "w01-r2") / "session-01")
    assert {i.day for i in arms1.values()} == {2} and {i.day for i in arms2.values()} == {4}
    assert min(arms2) > max(arms1)  # round 2 allocates after round 1
    control, treatment = (set(a.episode_ids) for a in reply2.results[0].arms)
    # Round 2's control arm is the current state: L1's revert of r03's delay in force.
    assert all(not arms2[e].faults for e in control)
    # The treatment's revert undoes the retained revert: the world's delay comes back.
    assert all([f.kind for f in arms2[e].faults] == ["toy_delay"] for e in treatment)
    control1, fixed1 = (set(a.episode_ids) for a in reply1.results[0].arms)
    assert all(any(f.kind == "toy_camera_shift" for f in arms1[e].faults) for e in control1)
    assert all([f.kind for f in arms1[e].faults] == ["toy_knob"] for e in fixed1)  # s1's decoy


def test_exports_hold_the_history_up_to_the_round(loop: dict[str, Any]) -> None:
    first, second, reply2 = loop["first"], loop["second"], loop["reply2"]
    days_first = {r["day"] for r in episodes(first)}
    assert days_first == {0, 1}
    rows = episodes(second)
    assert {r["day"] for r in rows} == {0, 1, 2, 3}
    assert {r["experiment_id"] for r in rows if r["day"] == 2} == {"x0001"}
    assert all(r["experiment_id"] is None for r in rows if r["day"] != 2)
    latest = episodes(reply2.export)
    assert {r["day"] for r in latest} == {0, 1, 2, 3, 4}
    assert {r["experiment_id"] for r in latest if r["day"] == 4} == {"x0002"}

    # Round 1's export holds the world's records up to its last day; round 2's (a review of
    # cc1ff68: a round-1 adoption that writes a change and a maintenance entry must not
    # block it) holds those and round 2's, the adoption among them as the fleet's own.
    def records(export: Path) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        with LogReader(export) as reader:
            log = reader.query("select day, scope, key, author from change_log").to_pylist()
            upkeep = reader.query("select day, robot_id, verb from maintenance").to_pylist()
        return [tuple(r.values()) for r in log], [tuple(r.values()) for r in upkeep]

    round1_log = [
        (0, "r03", DELAY.value, "vendor"),
        (1, "s1", ConfigKey.policy_sampling_steps.value, "ops"),
    ]
    round1_upkeep = [(1, "r04", "clean")]
    assert records(first) == (round1_log, round1_upkeep)
    log, upkeep = records(second)
    assert log == [*round1_log, (3, "r03", DELAY.value, "ops")]
    assert upkeep == [*round1_upkeep, (3, "r01", "service"), (3, "r02", "recalibrate")]
    assert records(reply2.export) == (log, upkeep)
    # Nothing of the evaluation in any export.
    panel = {
        i["episode_id"]
        for i in pq.read_table(loop["paths"].evaluation / "attempts.parquet").to_pylist()
    }
    assert not panel & {r["episode_id"] for r in latest}


def test_evaluation_compares_states_on_the_same_attempts(loop: dict[str, Any]) -> None:
    summary, world = loop["summary"], loop["world"]
    assert summary.days == [5] and summary.attempts == 16
    names = [s.name for s in summary.states]
    assert names == ["original", "round1"]
    original, round1 = summary.states
    assert original.changes == 0 and round1.changes == 2
    pair = summary.pairs[0]
    assert (pair.first, pair.second) == ("original", "round1")
    assert pair.net == round1.successes - original.successes
    assert pair.rescued + pair.broken <= 16
    attempts = pq.read_table(loop["paths"].evaluation / "attempts.parquet").to_pylist()
    assert len(attempts) == 32
    assert {a["episode_id"] for a in attempts if a["state"] == "original"} == {
        a["episode_id"] for a in attempts if a["state"] == "round1"
    }
    exported = {i.episode_id for i in world.identities}
    assert not exported & {a["episode_id"] for a in attempts}


def test_original_state_reproduces_the_as_run_outcomes(loop: dict[str, Any]) -> None:
    result = reproduces_as_run(loop["world"], loop["inside"].evaluation)
    assert result["compared"] == 16 and result["differ"] == 0


def test_round_engine_places_analyses_by_round() -> None:
    """The real run on w10-v2 found it: with no analysis in round 1, round 2's own
    analysis was taken for round 1's, and the reply's export lacked round 2's arms."""
    from unittest import mock

    parts: Any = mock.Mock()
    engine = RoundEngine(parts, parts, 2, parts, parts, parts)
    assert engine.analyses("a2") == ["", "a2"]
    engine.earlier = ["a1"]
    assert engine.analyses("a2") == ["a1", "a2"]
    engine.earlier = ["a1", "a0"]
    with pytest.raises(ValueError, match="earlier rounds"):
        engine.analyses("a2")
