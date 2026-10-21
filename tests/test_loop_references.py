"""The two reference methods on the common reserved evaluation, on the toy world (CPU): each
reads round 1's export, its ranked list is cut at the final number of changes by the
one-look plan's convention, and the resulting state is evaluated on the same panel beside
the agent's. The methods run as they are."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.loop.evaluate import evaluate
from fleetlab.loop.export import export_round
from fleetlab.loop.layout import layout_for_world
from fleetlab.loop.paths import LoopPaths
from fleetlab.loop.references import (
    REFERENCES,
    reference_costs,
    reference_ledger,
    run_reference,
    run_references,
)
from fleetlab.loop.state import Ledger, changes_in_force, one_look_ledger
from fleetlab.loop.toy import write_toy_world


@pytest.fixture(scope="module")
def setting(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("references")
    world, setup = write_toy_world(root / "world-root")
    layout = layout_for_world(world.spec)
    paths = LoopPaths(root / "loop")
    paths.root.mkdir(parents=True)
    l0 = Ledger.empty(world.world_id)
    first = export_round(world, layout, 1, l0, paths, paths.exports(1) / "start")
    runs = run_references(first, paths.root / "references")
    # The agent's final state: two changes in force (one replaced in round 2).
    recalibrate = Action(
        verb=Verb.recalibrate, target=Target(scope="r02", component=Component.scene_camera)
    )
    revert = Action(
        verb=Verb.revert,
        target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
    )
    final = l0.adopt(1, layout.adoption_day(1), adopt=[recalibrate]).adopt(
        2, layout.adoption_day(2), adopt=[revert]
    )
    return {
        "world": world,
        "setup": setup,
        "layout": layout,
        "paths": paths,
        "first": first,
        "runs": {r.state: r for r in runs},
        "final": final,
    }


def test_each_reference_reads_round_one_and_becomes_a_state(setting: dict[str, Any]) -> None:
    runs, world, layout = setting["runs"], setting["world"], setting["layout"]
    assert set(runs) == set(REFERENCES)
    rule = runs["build_rule"]
    assert rule.ok and rule.ranked  # the toy's site change (s1's sampling steps), reverted
    assert all(a.verb is Verb.revert for a in rule.ranked)
    k = changes_in_force(setting["final"])
    assert k == 2
    for run in runs.values():
        record = json.loads(
            (setting["paths"].root / "references" / run.state / "reference.json").read_text()
        )
        assert record["status"] == ("done" if run.ok else "failed")
        assert record["input_sha256"]["log/episodes.parquet"]
        if not run.ok:
            with pytest.raises(ValueError, match="missing, not zero"):
                reference_ledger(run, world.world_id, k, layout.adoption_day(1))
            continue
        ledger = reference_ledger(run, world.world_id, k, layout.adoption_day(1))
        # The one-look plan's convention, exactly.
        assert ledger == one_look_ledger(
            world.world_id, list(run.ranked), k, layout.adoption_day(1)
        )
        assert len(ledger.entries) <= k and ledger.version == 1
        assert all(e.adopted_day == layout.adoption_day(1) for e in ledger.entries)
        costs = reference_costs(run, ledger)
        assert costs["sessions"] == 0 and costs["usage_complete"] is True
        assert costs["experiments"]["arm_attempts"] == 0
        assert costs["compute_seconds_status"] == "measured" and costs["compute_seconds"] >= 0
        assert costs["api_equivalent_usd"]["status"] == "unavailable"


def test_references_are_evaluated_beside_the_agents_states(setting: dict[str, Any]) -> None:
    world, layout, paths, setup = (setting[k] for k in ("world", "layout", "paths", "setup"))
    k = changes_in_force(setting["final"])
    day = layout.adoption_day(1)
    states = {"original": Ledger.empty(world.world_id), "final": setting["final"]}
    for run in setting["runs"].values():
        if run.ok:
            states[run.state] = reference_ledger(run, world.world_id, k, day)
    summary = evaluate(world, layout, states, paths, setup, per_robot_day=2)
    assert [s.name for s in summary.states] == list(states)
    assert {s.attempts for s in summary.states} == {summary.attempts}
    assert summary.attempts == len(world.truth.fleet.robots()) * 2


def test_a_failed_reference_is_recorded_missing(tmp_path: Path) -> None:
    broken = tmp_path / "not-an-export"
    broken.mkdir()
    run = run_reference("classical", broken, tmp_path / "out")
    assert not run.ok and run.ranked == () and run.compute_seconds is None
    record = json.loads((tmp_path / "out" / "reference.json").read_text())
    assert record["status"] == "failed" and record["error"]
    assert record["compute_seconds"] is None and record["seconds_until_failure"] >= 0
    costs = reference_costs(run)
    assert costs["compute_seconds"] is None and costs["compute_seconds_status"] == "unknown"
    with pytest.raises(ValueError, match="no reference"):
        run_reference("random", broken, tmp_path / "other")
