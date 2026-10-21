"""Synthetic statistical and isolation checks, not evidence from a physical fleet."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fleetlab.baselines.classical import analyse, run
from fleetlab.baselines.classical.analysis import change_comparisons, change_points
from fleetlab.baselines.classical.data import Row, VisibleData, load_export
from fleetlab.baselines.classical.features import Design
from fleetlab.baselines.classical.regression import fit_survival, predict
from fleetlab.contracts.actions import COSTS, catalogue, catalogue_rows, costs_rows, fits_budget
from fleetlab.contracts.faults import Target
from fleetlab.contracts.model import (
    Change,
    ChangeAuthor,
    ComponentState,
    ConfigVersion,
    Fleet,
    Robot,
    Site,
    describe_for_agents,
)
from fleetlab.contracts.tools import Diagnosis, RankedFixSet
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb

DELAY = ConfigKey.control_action_delay_steps


def synthetic(*, harmful: bool = True, coincident: bool = False) -> VisibleData:
    """Task imbalance, a recorded regression and a good robot with a bad health check."""
    robots = [
        Robot(
            id=f"r{i:02}",
            site_id="s1" if i <= 3 else "s2",
            components={c: ComponentState(component=c, installed_day=-100) for c in Component},
        )
        for i in range(1, 7)
    ]
    changes = [Change(day=2, scope="r02", key=DELAY, old=0, new=20, author=ChangeAuthor.ops)]
    if coincident:
        changes.append(
            Change(
                day=2,
                scope="r02",
                key=ConfigKey.policy_sampling_steps,
                old=10,
                new=5,
                author=ChangeAuthor.ops,
            )
        )
    robots[1] = robots[1].model_copy(
        update={
            "config": ConfigVersion(
                version=len(changes), since_day=2, values={c.key: c.new for c in changes}
            )
        }
    )
    fleet = Fleet(
        id="w09",
        sites=[Site(id="s1", robots=robots[:3]), Site(id="s2", robots=robots[3:])],
        config=ConfigVersion(version=0, values={DELAY: 0, ConfigKey.policy_sampling_steps: 10}),
        change_log=changes,
    )
    episodes: list[Row] = []
    health: list[Row] = []
    for robot in robots:
        for day in range(6):
            for task in ("easy", "hard"):
                n = 30 if (task == "hard") == (robot.site_id == "s1") else 10
                rate = 0.9 if task == "easy" else 0.6
                if harmful and robot.id == "r02" and day >= 2:
                    rate *= 0.0
                for index in range(n):
                    episodes.append(
                        dict(
                            episode_id=f"e{len(episodes) + 1:06}",
                            robot_id=robot.id,
                            site_id=robot.site_id,
                            day=day,
                            task_text=task,
                            outcome="success" if index < round(n * rate) else "failure",
                            experiment_id=None,
                            started_at=time(9, 0),
                        )
                    )
            for component in (Component.scene_camera, Component.wrist_camera):
                health.append(
                    dict(
                        day=day,
                        robot_id=robot.id,
                        component=component.value,
                        metric="calibration_residual",
                        value=float(80 if robot.id == "r06" else 20) + day % 2,
                    )
                )
    return VisibleData(
        fleet,
        episodes,
        health,
        [c.model_dump(mode="json") for c in changes],
        [],
        catalogue(fleet),
        dict(COSTS),
        {},
    )


def write_export(root: Path, data: VisibleData) -> None:
    log = root / "log"
    log.mkdir(parents=True)
    (root / "data_dictionary.md").write_text("Synthetic fixture: visible tables only.\n")
    (log / "description.json").write_text(json.dumps(describe_for_agents(data.fleet)))
    rows = {
        "episodes": data.episodes,
        "health": data.health,
        "change_log": [
            {**r, "old": json.dumps(r["old"]), "new": json.dumps(r["new"])} for r in data.changes
        ],
        "events": [],
        "maintenance": [],
        "costs": costs_rows(),
        "catalogue": catalogue_rows(data.fleet),
    }
    for name, records in rows.items():
        from fleetlab.log._rows import table_schema

        schema = table_schema(name) if not records else None
        pq.write_table(pa.Table.from_pylist(records, schema=schema), log / f"{name}.parquet")


def test_recovers_change_and_rejects_health_decoy() -> None:
    data = synthetic()
    result = analyse(data, bootstraps=20)
    days = {r["episode_id"]: r["day"] for r in data.episodes}
    cause = next(
        c for c in result.diagnosis.causes if c.target == Target(scope="r02", config_key=DELAY)
    )
    assert all(e.episode_id is not None and days[e.episode_id] >= 2 for e in cause.evidence)
    first = result.fix_set.actions[0]
    assert first.action.verb == Verb.revert
    assert first.action.target == Target(scope="r02", config_key=DELAY)
    assert first.expected_gain > 8  # affected robot's task-adjusted gain in fleet points
    assert first.uncertainty.low > 0
    assert all(c.target.scope != "r06" for c in result.diagnosis.causes)
    assert len(result.fix_set.actions) == 10
    assert all(fits_budget(f) for f in result.fix_sets)
    assert [f.budget.amount for f in result.fix_sets] == [3, 6, 4, 8]
    assert len(result.evidence["validation_log_loss"]) == 6


def test_task_mix_is_not_a_configuration_fault() -> None:
    result = analyse(synthetic(harmful=False), bootstraps=20)
    matching = [
        a
        for a in result.evidence["actions"]
        if a["action"]["target"]["scope"] == "r02" and a["action"]["target"]["config_key"] == DELAY
    ]
    assert abs(matching[0]["gain_points"]) < 1
    assert not result.diagnosis.causes
    assert abs(change_comparisons(synthetic(harmful=False))[0]["adjusted_change_points"]) < 1e-8


def test_coincident_changes_are_explicitly_unidentified() -> None:
    result = analyse(synthetic(coincident=True), bootstraps=20)
    assert result.evidence["exposure_aliases"]
    records = [
        r
        for r in result.evidence["actions"]
        if r["aliases"] and r["action"]["target"]["config_key"]
    ]
    assert len(records) == 2
    assert all(r["low"] <= 0 < r["high"] for r in records)
    assert all(r["diagnosis_confidence"] <= 0.5 for r in records)
    assert not result.diagnosis.causes


def test_multiplicative_model_recovers_masked_repair() -> None:
    # Four equal-sized cells: healthy, either single impairment, both together.
    x, y = [], []
    for left, right, rate in [(0, 0, 0.9), (1, 0, 0.45), (0, 1, 0.3), (1, 1, 0.15)]:
        for i in range(200):
            x.append([1, left, right])
            y.append(float(i < round(200 * rate)))
    design = np.asarray(x, dtype=float)
    beta = fit_survival(design, np.asarray(y), np.ones(3) * 0.001, 1)
    fitted = predict(
        np.asarray([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1.0]]), beta, 1, "multiplicative"
    )
    np.testing.assert_allclose(fitted, [0.9, 0.45, 0.3, 0.15], atol=0.003)


def test_history_and_scope_precedence() -> None:
    data = synthetic()
    design = Design(data)
    assert design.configuration("r02", 1)[DELAY] == 0
    assert design.configuration("r02", 2)[DELAY] == 20
    assert design.configuration("r01", 5)[DELAY] == 0
    action = next(
        a
        for a in design.candidates()
        if a.verb == Verb.revert and a.target == Target(scope="r02", config_key=DELAY)
    )
    assert design.configuration("r02", 8, action)[DELAY] == 0
    # A fleet intervention may not overwrite an explicit robot override.
    changes = [
        dict(day=1, scope="fleet", key=DELAY.value, old=0, new=10, author="ops"),
        *data.changes,
    ]
    other = Design(replace(data, changes=changes))
    fleet_action = action.model_copy(update={"target": Target(scope="fleet", config_key=DELAY)})
    assert other.configuration("r02", 8, fleet_action)[DELAY] == 20


def test_visible_export_only_and_answer_roundtrip(tmp_path: Path) -> None:
    export = tmp_path / "export"
    data = synthetic()
    write_export(export, data)
    # Experiment episodes are deliberately impossible evidence and must be ignored.
    exp = {**data.episodes[0], "episode_id": "e999999", "day": 9, "experiment_id": "x0001"}
    pq.write_table(pa.Table.from_pylist([*data.episodes, exp]), export / "log/episodes.parquet")
    (tmp_path / "truth").mkdir()
    (tmp_path / "truth/canary.json").write_text("not even valid JSON; reading is a bug")
    first = run(export, bootstraps=20)
    (tmp_path / "truth/canary.json").write_text("different hidden truth")
    second = run(export, bootstraps=20)
    assert first.fix_set == second.fix_set
    assert first.diagnosis == second.diagnosis
    assert first.evidence["observed_episodes"] == len(data.episodes)
    first.write(tmp_path / "answer")
    assert (
        RankedFixSet.model_validate_json((tmp_path / "answer/outbox/fix_set.json").read_text())
        == first.fix_set
    )
    assert (
        Diagnosis.model_validate_json((tmp_path / "answer/outbox/diagnosis.json").read_text())
        == first.diagnosis
    )
    with pytest.raises(FileExistsError):
        first.write(tmp_path / "answer")


def test_rejects_future_data_and_path_escape(tmp_path: Path) -> None:
    export = tmp_path / "export"
    data = synthetic()
    write_export(export, data)
    data.episodes[0]["day"] = 8
    pq.write_table(pa.Table.from_pylist(data.episodes), export / "log/episodes.parquet")
    with pytest.raises(ValueError, match="observed window"):
        load_export(export)
    path = export / "log/episodes.parquet"
    outside = tmp_path / "outside.parquet"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_export(export)


def test_runtime_imports_do_not_reach_hidden_modules() -> None:
    package = Path(__file__).parents[1] / "src/fleetlab/baselines/classical"
    forbidden = (
        "fleetlab.oracle",
        "fleetlab.worlds",
        "fleetlab.scoring",
        "fleetlab.faults",
        "fleetlab.engine",
        "fleetlab.policies",
    )
    for path in package.glob("*.py"):
        if path.name in {"evaluate.py", "driver.py"}:
            continue  # explicit offline evaluation is not imported by runtime
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not any(m.startswith(forbidden) for m in modules), path


def test_change_point_scan_adjusts_tasks_and_matches_logged_date() -> None:
    shifts = change_points(synthetic())
    target = next(s for s in shifts if s["scope"] == "r02")
    assert target["detected"] and target["day"] == 2
    assert target["direction"] == "decline"
    assert target["coincident_changes"] == [DELAY]
    assert all(not s["detected"] for s in change_points(synthetic(harmful=False)))


def test_late_break_is_not_assigned_to_an_earlier_edit() -> None:
    data = synthetic()
    key = ConfigKey.software_version
    late = Change(day=4, scope="r02", key=key, old="v0", new="v1", author=ChangeAuthor.ops)
    robots = data.fleet.robots()
    robots[1] = robots[1].model_copy(
        update={"config": ConfigVersion(version=2, since_day=4, values={DELAY: 20, key: "v1"})}
    )
    fleet = data.fleet.model_copy(
        update={
            "sites": [
                data.fleet.sites[0].model_copy(update={"robots": robots[:3]}),
                data.fleet.sites[1],
            ],
            "config": ConfigVersion(version=0, values={**data.fleet.config.values, key: "v0"}),
            "change_log": [*data.fleet.change_log, late],
        }
    )
    counts: dict[tuple[int, str], int] = {}
    for row in data.episodes:
        if row["robot_id"] != "r02":
            continue
        group = row["day"], row["task_text"]
        index = counts.get(group, 0)
        counts[group] = index + 1
        successes = 9 if row["task_text"] == "easy" else 18
        row["outcome"] = "success" if row["day"] < 4 and index < successes else "failure"
    result = analyse(
        replace(data, fleet=fleet, changes=[*data.changes, late.model_dump(mode="json")]),
        bootstraps=200,
    )
    targets = [c.target for c in result.diagnosis.causes]
    assert Target(scope="r02", config_key=key) in targets
    assert Target(scope="r02", config_key=DELAY) not in targets


def test_unobserved_restoration_value_is_not_treated_as_reference() -> None:
    data = synthetic()
    # All observed days contain the new value. The old value has never been
    # observed anywhere; another robot's zero is not evidence about this value.
    changes = [{**data.changes[0], "day": 0, "old": 123}]
    result = analyse(replace(data, changes=changes), bootstraps=20)
    row = next(
        r
        for r in result.evidence["actions"]
        if r["action"]["target"]["scope"] == "r02" and r["action"]["target"]["config_key"] == DELAY
    )
    assert row["gain_points"] == 0
    assert row["diagnosis_confidence"] == 0
    assert row["low"] < 0 < row["high"]
    assert row["unobserved_restoration"]
