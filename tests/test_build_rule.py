"""The build rule: what it names, what it reads, and that the scorer never sees it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_scoring import NOMINAL, inputs
from test_scoring import toy_fleet as scored_fleet
from test_scoring_gen3 import certify_fixture, classical_source, simple_source
from test_scoring_round6 import fake_scoring_world

from fleetlab.agents.sessions import Analysis
from fleetlab.baselines import build_rule as rule
from fleetlab.baselines.build_rule import Entry, build_rule, evaluate, main, nominations, suite_of
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, catalogue_rows
from fleetlab.contracts.faults import Target
from fleetlab.contracts.log import visible_schema
from fleetlab.contracts.model import (
    ComponentState,
    ConfigVersion,
    Fleet,
    Robot,
    Site,
    describe_for_agents,
)
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.log.reader import LogReader
from fleetlab.oracle import world as oracle_world
from fleetlab.scoring import __main__ as command
from fleetlab.scoring.baselines import read_baseline_fix_set
from fleetlab.scoring.comparisons import BASELINE_METHODS

# Independent literals: a change of the module's wording or task lists must fail here.
EXPECTED_WORDING = (
    "One revert for every scope and key that has an entry in the change log made at a site, "
    "then one for every scope and key with an entry made for the whole fleet, each group in "
    "the order of its earliest entry (day, site number, key). Then, for every site at which "
    "every task run is among the ten sentences of one suite and for which no site entry was "
    "read, the action that replaces its batch of objects by the site's recorded batch, by "
    "site number. An entry whose scope or key the vocabulary does not hold is left out; an "
    "action the catalogue does not hold is skipped and the next moves up. The first three "
    "actions are the answer at three and at six fixes; at four and eight hours, those of the "
    "three that fit, in order. No episode outcome, no health reading and no video is read."
)
SPATIAL = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
]
OBJECT = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
]
DELAY = "control.action_delay_steps"
VERSION = "software.version"
CONFIG = {key: value for key, value in NOMINAL.items() if isinstance(key, ConfigKey)}
CONDITIONS = {key: value for key, value in NOMINAL.items() if isinstance(key, ConditionKey)}
FORBIDDEN = (
    "fleetlab.engine",
    "fleetlab.oracle",
    "fleetlab.worlds",
    "fleetlab.faults",
    "fleetlab.scoring",
    "fleetlab.sim",
)
# The build of every composed world: two sites of one suite, the third runs both.
BUILD = {"s1": SPATIAL, "s2": [*SPATIAL[:6], *OBJECT[:5]], "s3": SPATIAL}


def table(export: Path, name: str, rows: list[dict[str, Any]]) -> None:
    spec = visible_schema()[name]
    schema = pa.schema([pa.field(c.name, c.type) for c in spec.columns])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), export / spec.path)


def three_sites() -> Fleet:
    def site(number: int) -> Site:
        robot = Robot(
            id=f"r{number:02}",
            site_id=f"s{number}",
            components={c: ComponentState(component=c, installed_day=0) for c in Component},
        )
        return Site(
            id=f"s{number}",
            robots=[robot],
            conditions={**CONDITIONS, ConditionKey.object_batch: [0.0, float(number)]},
        )

    return Fleet(
        id="w01", sites=[site(n) for n in (1, 2, 3)], config=ConfigVersion(version=0, values=CONFIG)
    )


def write_export(
    export: Path,
    changes: list[tuple[int, str, str]],
    ran: dict[str, list[str]],
    *,
    fleet: Fleet | None = None,
    outcome: str = "failure",
    drop: tuple[str, str, str] | None = None,
) -> Path:
    """A synthetic public export with the tables the rule reads and those it must not."""
    fleet = fleet or three_sites()
    (export / "log").mkdir(parents=True)
    (export / "data_dictionary.md").write_text("synthetic visible export")
    (export / "log/description.json").write_text(json.dumps(describe_for_agents(fleet)))
    rows = catalogue_rows(fleet, ["put"])
    if drop is not None:
        verb, scope, thing = drop
        rows = [
            r
            for r in rows
            if (r["verb"], r["scope"], r["config_key"] or r["condition_key"]) != drop
        ]
        assert len(rows) + 1 == len(catalogue_rows(fleet, ["put"])), (verb, scope, thing)
    table(export, "catalogue", rows)
    table(
        export,
        "change_log",
        [dict(day=d, scope=s, key=k, old="0", new="17", author="ops") for d, s, k in changes],
    )
    robots = {site.id: site.robots[0].id for site in fleet.sites}
    episodes = [
        dict(
            episode_id=f"e{number:06}",
            robot_id=robots.get(site, "r99"),
            site_id=site,
            day=number % 5,
            task_text=task,
            steps=200,
            outcome=outcome,
            duration_s=10.0 if outcome == "failure" else 4.0,
            video_scene=f"video/e{number:06}.scene.mp4",
        )
        for number, (site, task) in enumerate(
            ((site, task) for site, tasks in ran.items() for task in tasks), start=1
        )
    ]
    table(export, "episodes", episodes)
    table(
        export,
        "health",
        [
            dict(
                day=1,
                robot_id=robot,
                component="gripper",
                metric="grip_force",
                value=50.0 if outcome == "failure" else 5.0,
            )
            for robot in robots.values()
        ],
    )
    return export


def named(root: Path, budget: str = "count-3") -> list[tuple[str, str, str]]:
    raw = json.loads((root / f"fix_sets/{budget}.json").read_text())
    return [
        (
            a["verb"],
            a["target"]["scope"],
            a["target"]["config_key"] or a["target"]["condition_key"],
        )
        for a in raw["actions"]
    ]


def test_the_wording_the_cap_and_the_task_lists_are_frozen() -> None:
    assert rule.WORDING == EXPECTED_WORDING
    assert (rule.METHOD, rule.RULE_VERSION, rule.MAX_ACTIONS) == ("build_rule", "build-rule-v1", 3)
    assert dict(rule.SUITES) == {
        "libero_spatial": frozenset(SPATIAL),
        "libero_object": frozenset(OBJECT),
    }
    assert rule.INPUTS == (
        "log/description.json",
        "log/catalogue.parquet",
        "log/change_log.parquet",
        "log/episodes.parquet",
    )


def test_suite_of_one_suite_only() -> None:
    assert suite_of(SPATIAL)[0] == "libero_spatial"
    assert suite_of(OBJECT[:1])[0] == "libero_object"
    assert suite_of([*SPATIAL, OBJECT[3]])[0] is None
    assert suite_of([])[0] is None
    # An unknown task is no evidence of a single suite, however alike it reads.
    assert suite_of([*SPATIAL, "pick up the black bowl and place it on the plate"])[0] is None


def test_nominations_follow_the_three_steps_in_order() -> None:
    suites = {"s3": "libero_spatial", "s1": "libero_spatial", "s2": None}
    changes = [
        Entry(4, "fleet", VERSION),
        Entry(0, "fleet", VERSION),
        Entry(0, "r06", DELAY),
        Entry(1, "s1", DELAY),
        Entry(3, "r10", DELAY),
    ]
    got = nominations(changes, suites)
    assert [(n.step, n.verb.value, n.target.scope, n.target.thing) for n in got] == [
        ("site change", "revert", "s1", ConfigKey.control_action_delay_steps),
        ("fleet change", "revert", "fleet", ConfigKey.software_version),
        ("single-suite batch", "change_condition", "s3", ConditionKey.object_batch),
    ]
    # A revert's reason names the earliest entry of its scope and key and says how many
    # there are: at the fleet that is the harmless patch of day 0, not the later release.
    assert [n.because for n in got] == [
        f"change log: s1, {DELAY}, entries: 1, the earliest on day 1",
        f"change log: fleet, {VERSION}, entries: 2, the earliest on day 0",
        "every task run at s3 is of libero_spatial, and no site entry was read for it",
    ]
    # No robot's entry is ever named, and the order of the rows handed in does not matter.
    assert nominations(list(reversed(changes)), dict(reversed(suites.items()))) == got


def test_nominations_order_within_a_step_and_site_ids_by_number() -> None:
    suites = {f"s{n}": "libero_object" for n in (10, 2, 1, 3)}
    changes = [
        Entry(2, "s3", DELAY),
        Entry(1, "s10", DELAY),
        Entry(1, "s2", "policy.sampling_steps"),
        Entry(1, "s2", DELAY),
        Entry(5, "s2", DELAY),
        Entry(1, "fleet", "control.mode"),
        Entry(0, "fleet", VERSION),
        Entry(3, "fleet", VERSION),
    ]
    got = [(n.target.scope, n.target.thing) for n in nominations(changes, suites)]
    assert got == [
        ("s2", ConfigKey.control_action_delay_steps),
        ("s2", ConfigKey.policy_sampling_steps),
        ("s10", ConfigKey.control_action_delay_steps),
        ("s3", ConfigKey.control_action_delay_steps),
        ("fleet", ConfigKey.software_version),
        ("fleet", ConfigKey.control_mode),
        ("s1", ConditionKey.object_batch),
    ]
    unchanged = {"s10": "libero_object", "s2": "libero_spatial", "s1": None}
    assert [n.target.scope for n in nominations([], unchanged)] == ["s2", "s10"]


def test_nominations_without_a_site_change_and_with_one_at_the_mixed_site() -> None:
    suites = {"s1": "libero_spatial", "s2": None, "s3": "libero_spatial"}
    fleet = Entry(0, "fleet", VERSION)
    assert [(n.verb, n.target.scope) for n in nominations([fleet], suites)] == [
        (Verb.revert, "fleet"),
        (Verb.change_condition, "s1"),
        (Verb.change_condition, "s3"),
    ]
    mixed = nominations([fleet, Entry(1, "s2", DELAY)], suites)
    assert [(n.verb, n.target.scope) for n in mixed] == [
        (Verb.revert, "s2"),
        (Verb.revert, "fleet"),
        (Verb.change_condition, "s1"),
        (Verb.change_condition, "s3"),
    ]
    assert nominations([], {"s1": None, "s2": None}) == []
    with pytest.raises(ValueError):
        nominations([Entry(0, "s1", "no.such.key")], suites)


def test_fix_sets_of_the_usual_build_at_every_budget(tmp_path: Path) -> None:
    export = write_export(
        tmp_path / "export",
        [(0, "fleet", VERSION), (0, "r02", DELAY), (1, "s1", DELAY), (4, "fleet", VERSION)],
        BUILD,
    )
    out = tmp_path / "rule"
    assert main(["--export", str(export), "--out", str(out)]) == 0
    three = [
        ("revert", "s1", DELAY),
        ("revert", "fleet", VERSION),
        ("change_condition", "s3", "object_batch"),
    ]
    assert named(out, "count-3") == named(out, "count-6") == named(out, "cost-8") == three
    # Four hours hold the two reverts (0.1 h each) and not the batch (4 h) on top.
    assert named(out, "cost-4") == three[:2]
    for budget in SCORING_BUDGETS:
        fix = read_baseline_fix_set(out, budget)
        assert len(fix.actions) <= 3
    batch = read_baseline_fix_set(out, SCORING_BUDGETS[0]).actions[2]
    assert batch.value == [0.0, 3.0]  # the site's own recorded batch, not another site's
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["method"] == "build_rule" and meta["rule"] == EXPECTED_WORDING
    assert meta["rule_version"] == "build-rule-v1" and meta["baseline_of_the_experiment"] is False
    assert [n["step"] for n in meta["named"]] == [
        "site change",
        "fleet change",
        "single-suite batch",
    ]
    assert meta["skipped"] == [] and meta["beyond_the_cap"] == []
    assert {s: v["suite"] for s, v in meta["sites"].items()} == {
        "s1": "libero_spatial",
        "s2": None,
        "s3": "libero_spatial",
    }
    assert meta["sites"]["s1"]["change_at_site"] and not meta["sites"]["s3"]["change_at_site"]
    assert sorted(meta["input_sha256"]) == sorted(rule.INPUTS)
    assert meta["explicit_fix_sets_authoritative"] and meta["format_padding_never_selected"]
    ranked = json.loads((out / "outbox/fix_set.json").read_text())["actions"]
    assert len(ranked) == 10
    assert [r["rationale"].startswith("FORMAT PADDING ONLY") for r in ranked] == [False] * 3 + [
        True
    ] * 7


def test_the_first_named_action_stands_first(tmp_path: Path) -> None:
    export = write_export(tmp_path / "export", [(0, "fleet", VERSION), (1, "s3", DELAY)], BUILD)
    answer = build_rule(export)
    first = answer.fix_sets()["count-3"].actions[0]
    assert first == Action(
        verb=Verb.revert,
        target=Target(scope="s3", config_key=ConfigKey.control_action_delay_steps),
    )
    assert [r.action for r in answer.ranked.actions][:3] == answer.fix_sets()["count-3"].actions


def test_the_cap_is_three_and_what_falls_beyond_is_recorded(tmp_path: Path) -> None:
    export = write_export(tmp_path / "export", [(0, "fleet", VERSION), (2, "s2", DELAY)], BUILD)
    out = tmp_path / "rule"
    assert main(["--export", str(export), "--out", str(out)]) == 0
    assert named(out) == [
        ("revert", "s2", DELAY),
        ("revert", "fleet", VERSION),
        ("change_condition", "s1", "object_batch"),
    ]
    meta = json.loads((out / "metadata.json").read_text())
    assert [(n["verb"], n["target"]["scope"]) for n in meta["beyond_the_cap"]] == [
        ("change_condition", "s3")
    ]
    # Eight hours hold one batch beside the reverts, and the list has no second one to try.
    assert named(out, "cost-8") == named(out)


@pytest.mark.parametrize(
    "drop, expected",
    [
        (("revert", "s1", DELAY), [("revert", "fleet", VERSION), ("change_condition", "s3")]),
        (("change_condition", "s3", "object_batch"), [("revert", "s1"), ("revert", "fleet")]),
    ],
)
def test_an_action_the_catalogue_lacks_is_skipped_and_said(
    tmp_path: Path, drop: tuple[str, str, str], expected: list[tuple[str, ...]]
) -> None:
    export = write_export(
        tmp_path / "export", [(0, "fleet", VERSION), (1, "s1", DELAY)], BUILD, drop=drop
    )
    answer = build_rule(export)
    got = [(a.verb.value, a.target.scope) for a in answer.fix_sets()["count-3"].actions]
    assert got == [e[:2] for e in expected]
    skipped = answer.metadata["skipped"]
    assert [(s["verb"], s["target"]["scope"]) for s in skipped] == [drop[:2]]
    assert "catalogue does not hold" in skipped[0]["reason"]
    # The site keeps its change: a skipped revert does not bring the batch to that site.
    assert ("change_condition", "s1") not in got


def test_a_batch_that_is_not_recorded_cannot_be_named(tmp_path: Path) -> None:
    fleet = three_sites()
    sites = [
        s if s.id != "s3" else s.model_copy(update={"conditions": {ConditionKey.clutter: 0.0}})
        for s in fleet.sites
    ]
    export = write_export(
        tmp_path / "export",
        [(1, "s1", DELAY)],
        BUILD,
        fleet=fleet.model_copy(update={"sites": sites}),
    )
    answer = build_rule(export)
    assert [a.target.scope for a in answer.fix_sets()["count-3"].actions] == ["s1"]
    assert [s["target"]["scope"] for s in answer.metadata["skipped"]] == ["s3"]


def test_an_entry_outside_the_vocabulary_is_left_unread_and_said(tmp_path: Path) -> None:
    export = write_export(
        tmp_path / "export",
        [(0, "s1", "no.such.key"), (1, "s3", DELAY), (2, "depot", DELAY)],
        {**BUILD, "s9": OBJECT},
    )
    answer = build_rule(export)
    got = [(a.verb.value, a.target.scope) for a in answer.fix_sets()["count-3"].actions]
    assert got == [("revert", "s3"), ("change_condition", "s1")]
    assert [(e["scope"], e["key"]) for e in answer.metadata["unread_change_entries"]] == [
        ("s1", "no.such.key"),
        ("depot", DELAY),
    ]
    assert answer.metadata["sites_with_episodes_and_no_description"] == ["s9"]


def revert(scope: str, key: str = DELAY) -> str:
    return f"{scope}:{key}"


def batch(site: str) -> str:
    return f"{site}:object_batch"


F0, F4 = (0, "fleet", VERSION), (4, "fleet", VERSION)
FLEET = revert("fleet", VERSION)
MIXED = [*SPATIAL, *OBJECT]
# The fix sets of the first delivery (f5b3150) on constructed exports, taken from that code:
# the change log, the tasks per site, a catalogue row left out, the three actions, and how
# many of them the four and the eight hours hold. No later change of the module may move one.
PINNED: dict[str, tuple[list[tuple[int, str, str]], dict[str, list[str]], Any, list[str], int, int]]
PINNED = {
    "the usual build": (
        [F0, (0, "r02", DELAY), (1, "s1", DELAY), F4],
        BUILD,
        None,
        [revert("s1"), FLEET, batch("s3")],
        2,
        3,
    ),
    # Nothing stands at one fix. Eight hours hold one batch: 0.1 + 4 + 4 is more than 8.
    "no site entry": (
        [F0, F4, (0, "r02", DELAY)],
        BUILD,
        None,
        [FLEET, batch("s1"), batch("s3")],
        1,
        2,
    ),
    "two site entries at two sites": (
        [F0, (2, "s1", DELAY), (1, "s3", DELAY)],
        BUILD,
        None,
        [revert("s3"), revert("s1"), FLEET],
        3,
        3,
    ),
    "one site, two keys": (
        [F0, (1, "s1", DELAY), (1, "s1", "policy.sampling_steps")],
        BUILD,
        None,
        [revert("s1"), revert("s1", "policy.sampling_steps"), FLEET],
        3,
        3,
    ),
    # Two entries of one scope and key are one revert.
    "one site, one key twice": (
        [F0, F4, (1, "s1", DELAY), (3, "s1", DELAY)],
        BUILD,
        None,
        [revert("s1"), FLEET, batch("s3")],
        2,
        3,
    ),
    "three site entries": (
        [F0, (1, "s1", DELAY), (2, "s2", DELAY), (3, "s3", DELAY)],
        BUILD,
        None,
        [revert("s1"), revert("s2"), revert("s3")],
        3,
        3,
    ),
    "site entry at the mixed site": (
        [F0, F4, (1, "s2", DELAY)],
        BUILD,
        None,
        [revert("s2"), FLEET, batch("s1")],
        2,
        3,
    ),
    "two fleet keys": (
        [F0, (2, "fleet", "control.mode"), (1, "s1", DELAY)],
        BUILD,
        None,
        [revert("s1"), FLEET, revert("fleet", "control.mode")],
        3,
        3,
    ),
    "two fleet keys, no site entry": (
        [F0, (2, "fleet", "control.mode")],
        BUILD,
        None,
        [FLEET, revert("fleet", "control.mode"), batch("s1")],
        2,
        3,
    ),
    # An entry that is left out is no site entry read: s1 keeps its batch.
    "a key outside the vocabulary at s1": (
        [F0, (1, "s1", "no.such.key")],
        BUILD,
        None,
        [FLEET, batch("s1"), batch("s3")],
        1,
        2,
    ),
    "an entry at a site not described": (
        [F0, (1, "s9", DELAY)],
        BUILD,
        None,
        [FLEET, batch("s1"), batch("s3")],
        1,
        2,
    ),
    # One task that is none of the twenty sentences ends the site's batch.
    "one task of neither suite at s3": (
        [F0, (1, "s1", DELAY)],
        {**BUILD, "s3": [*SPATIAL, "wipe the table"]},
        None,
        [revert("s1"), FLEET],
        2,
        2,
    ),
    "three sites of one suite each": (
        [F0],
        {"s1": SPATIAL, "s2": OBJECT, "s3": SPATIAL},
        None,
        [FLEET, batch("s1"), batch("s2")],
        1,
        2,
    ),
    # A skipped action takes none of the three places: the next moves up.
    "the catalogue lacks the fleet revert": (
        [F0, (1, "s2", DELAY)],
        BUILD,
        ("revert", "fleet", VERSION),
        [revert("s2"), batch("s1"), batch("s3")],
        1,
        2,
    ),
    "no change log, every site mixed": (
        [],
        {"s1": MIXED, "s2": MIXED, "s3": MIXED},
        None,
        [],
        0,
        0,
    ),
    # A batch alone is exactly four hours.
    "robots' entries only": (
        [(0, "r01", DELAY), (1, "r02", VERSION)],
        BUILD,
        None,
        [batch("s1"), batch("s3")],
        1,
        2,
    ),
    # Days are numbers: day 2 comes before day 10.
    "a day of two digits": (
        [F0, (10, "s1", DELAY), (2, "s3", DELAY)],
        BUILD,
        None,
        [revert("s3"), revert("s1"), FLEET],
        3,
        3,
    ),
}


@pytest.mark.parametrize("name", list(PINNED))
def test_fix_sets_are_those_of_the_first_delivery(tmp_path: Path, name: str) -> None:
    changes, ran, drop, three, four_hours, eight_hours = PINNED[name]
    export = write_export(tmp_path / "export", changes, ran, drop=drop)
    got = {
        label: [
            f"{a.target.scope}:{a.target.config_key or a.target.condition_key}" for a in f.actions
        ]
        for label, f in build_rule(export).fix_sets().items()
    }
    assert got == {
        "count-3": three,
        "count-6": three,
        "cost-4": three[:four_hours],
        "cost-8": three[:eight_hours],
    }


def test_reads_no_outcome_no_health_and_no_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changes = [(0, "fleet", VERSION), (1, "s1", DELAY)]
    failing = write_export(tmp_path / "a/export", changes, BUILD, outcome="failure")
    working = write_export(tmp_path / "b/export", changes, BUILD, outcome="success")
    (failing.parent / "fault_table.json").write_text('{"canary": "hidden-truth-canary"}')
    asked: list[str] = []
    original = LogReader.query

    def query(self: LogReader, sql: str, params: Any = None) -> pa.Table:
        asked.append(sql)
        return original(self, sql, params)

    monkeypatch.setattr(LogReader, "query", query)
    first, second = build_rule(failing), build_rule(working)
    assert sorted(set(asked)) == [
        "SELECT * FROM catalogue",
        "SELECT DISTINCT site_id, task_text FROM episodes",
        "SELECT day, scope, key FROM change_log",
    ]
    assert first.ranked == second.ranked and first.fix_sets() == second.fix_sets()
    differing = {"export", "input_sha256"}
    assert {k for k in first.metadata if first.metadata[k] != second.metadata[k]} == differing
    hashes = first.metadata["input_sha256"], second.metadata["input_sha256"]
    assert {k for k in hashes[0] if hashes[0][k] != hashes[1][k]} == {"log/episodes.parquet"}
    assert "canary" not in json.dumps(first.metadata)


def test_command_refuses_what_it_must_not_write_or_read(tmp_path: Path) -> None:
    export = write_export(tmp_path / "world/export", [(1, "s1", DELAY)], BUILD)
    out = tmp_path / "rule"
    assert main(["--export", str(export), "--out", str(out)]) == 0
    for arguments in (
        ["--export", str(export), "--out", str(out)],  # nothing is overwritten
        ["--export", str(export), "--out", str(export / "rule")],  # never inside the export
        ["--export", str(export.parent), "--out", str(tmp_path / "x")],  # a world root
        ["--export", str(export), "--world-root", str(tmp_path), "--out", str(tmp_path / "y")],
        ["--evaluate", str(out), "--out", str(tmp_path / "z.json")],  # no world named
        ["--out", str(tmp_path / "v")],
    ):
        with pytest.raises(SystemExit) as stop:
            main(arguments)
        assert stop.value.code == 2
    assert not (export / "rule").exists() and sorted(p.name for p in export.iterdir()) == [
        "data_dictionary.md",
        "log",
    ]


def test_selection_loads_no_module_that_holds_truth(tmp_path: Path) -> None:
    export = write_export(tmp_path / "export", [(1, "s1", DELAY)], BUILD)
    out = tmp_path / "rule"
    code = f"""
import sys
from importlib.abc import MetaPathFinder
forbidden = {FORBIDDEN!r}
class RefuseTruth(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(forbidden), fullname
sys.meta_path.insert(0, RefuseTruth())
from fleetlab.baselines.build_rule import main
status = main(['--export', {str(export)!r}, '--out', {str(out)!r}])
assert not [n for n in sys.modules if n.startswith(forbidden)]
raise SystemExit(status)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert named(out) == [("revert", "s1", DELAY), ("change_condition", "s3", "object_batch")]


def test_the_rule_is_none_of_the_baselines() -> None:
    import fleetlab.baselines as package

    source = Path(package.__file__).parent
    assert "build_rule" not in package.__all__
    assert "build_rule" not in (source / "__main__.py").read_text()
    makefile = (source.parents[2] / "Makefile").read_text()
    assert "build_rule" not in makefile and "build-rule" not in makefile
    # The scorer's fixed opponents are named in its own code, and the rule is not among them.
    assert rule.EVALUATION_NAME not in BASELINE_METHODS
    assert all(rule.METHOD not in methods for methods in BASELINE_METHODS.values())


# ---- the evaluation, on the scorer tests' constructed world -------------------------------


def rule_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Analysis, Any]:
    """The scorer tests' world with a real export in it; the evaluation's loaders are the
    scorer's own, so the same stand-ins serve both."""
    root, analysis, oracle_report = fake_scoring_world(tmp_path, monkeypatch)
    for module, names in (
        ("fleetlab.oracle.world", ("load_world", "world_ready", "recorded_sources", "all_runs")),
        ("fleetlab.oracle.steps", ("Oracle",)),
        ("fleetlab.oracle.table", ("build_table",)),
        ("fleetlab.oracle.report", ("report_json",)),
    ):
        for name in names:
            monkeypatch.setattr(f"{module}.{name}", getattr(command, name))
    shutil.rmtree(root / "export")
    fleet = scored_fleet()
    sites = [s.model_copy(update={"conditions": CONDITIONS}) for s in fleet.sites]
    write_export(
        root / "export",
        [(0, "fleet", VERSION), (3, "s2", DELAY)],
        {"s1": SPATIAL, "s2": [*SPATIAL, *OBJECT]},
        fleet=fleet.model_copy(
            update={"sites": sites, "config": ConfigVersion(version=0, values=CONFIG)}
        ),
    )
    certify_fixture(root / "export", "w01")
    return root, analysis, oracle_report


def plain_list(rule_dir: Path, to: Path) -> Path:
    """The same four fix sets with no record beside them: an ordinary list to the scorer."""
    shutil.copytree(rule_dir / "fix_sets", to / "fix_sets")
    return to


def test_evaluation_values_the_fix_sets_as_the_scorer_values_a_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    assert named(out) == [
        ("revert", "s2", DELAY),
        ("revert", "fleet", VERSION),
        ("change_condition", "s1", "object_batch"),
    ]
    file = tmp_path / "evaluation/rule-w01.json"
    assert main(["--evaluate", str(out), "--world-root", str(root), "--out", str(file)]) == 0
    result = json.loads(file.read_text())
    assert result == json.loads(json.dumps(evaluate(root, out)))
    # By hand (tests/test_scoring.py): reverting the delay at s2 rescues two of ten episodes,
    # and the oracle's best three fixes gain four.
    at = {b["label"]: b for b in result["budgets"]}
    assert list(at) == ["k = 3", "k = 6", "4 h", "8 h"]
    assert at["k = 3"]["rule"]["gain"] == {"rescued": 2, "broken": 0, "net": 2}
    assert at["k = 3"]["rule"]["points"] == 20.0 and at["k = 3"]["achievable"]["points"] == 40.0
    assert at["k = 3"]["rule"]["share"]["value"] == 0.5
    assert len(at["4 h"]["rule"]["actions"]) == 2 and at["4 h"]["rule"]["hours"] == 0.2
    assert result["method"] == "build_rule" and result["world_id"] == "w01"
    assert result["rule"] == EXPECTED_WORDING and result["rule_version"] == "build-rule-v1"
    assert result["export"] == str(root / "export") and "computed again" in result["valued"]
    assert result["export_receipt_problem"] is None and "claim table" in result["claims"]
    assert result["scoring_episodes"] == 10 and result["needs_fresh_replay"] == {}
    assert sorted(result["fix_set_sha256"]) == [
        f"fix_sets/{name}.json" for name in ("cost-4", "cost-8", "count-3", "count-6")
    ]
    # The same four fix sets, handed to the scorer as an ordinary list, are valued alike.
    plain = plain_list(out, tmp_path / "plain")
    report = command.score_world(root, analysis.root, baselines={"some other list": plain})
    for scored in report.claim3.budgets:
        mine = at[scored.label]
        assert mine["rule"] == scored.baselines["some other list"].model_dump(mode="json")
        assert mine["achievable"] == scored.achievable.model_dump(mode="json")


def test_evaluation_is_withheld_where_the_scorer_would_withhold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, oracle_report = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    oracle_report.replay_verification = {}
    result = evaluate(root, out)
    assert all(b["rule"]["points"] is None and b["rule"]["missing"] for b in result["budgets"])
    assert all(b["achievable"]["points"] is None for b in result["budgets"])


def test_evaluation_values_only_what_the_rule_names_from_the_worlds_own_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    whole = evaluate(root, out)["budgets"]
    # A copy of the export elsewhere is the same bytes: the rule names the same from it.
    copy = shutil.copytree(root / "export", tmp_path / "copy/export")
    elsewhere = tmp_path / "rule-from-copy"
    assert main(["--export", str(copy), "--out", str(elsewhere)]) == 0
    assert evaluate(root, elsewhere)["budgets"] == whole
    # 1. Any one of the four fix sets cut by hand to its first action, and a fifth laid beside.
    edited = []
    for label in ("count-3", "count-6", "cost-4", "cost-8"):
        cut = shutil.copytree(out, tmp_path / f"cut-{label}")
        raw = json.loads((cut / f"fix_sets/{label}.json").read_text())
        assert len(raw["actions"]) > 1
        (cut / f"fix_sets/{label}.json").write_text(
            json.dumps({**raw, "actions": raw["actions"][:1]})
        )
        edited.append(cut)
    fifth = shutil.copytree(out, tmp_path / "fifth")
    shutil.copy(fifth / "fix_sets/count-3.json", fifth / "fix_sets/count-9.json")
    for directory in (*edited, fifth):
        with pytest.raises(ValueError, match="not what the rule names"):
            evaluate(root, directory)
    # 2. Another version of the rule, or another wording, over the very same fix sets.
    for field, value in (("rule_version", "build-rule-v0"), ("rule", "another wording")):
        other = shutil.copytree(out, tmp_path / f"other-{field}")
        meta = json.loads((other / "metadata.json").read_text())
        (other / "metadata.json").write_text(json.dumps({**meta, field: value}))
        with pytest.raises(ValueError, match="another version or wording"):
            evaluate(root, other)
    # 3. The rule run on another export that lies under the world's root.
    old = shutil.copytree(root / "export", root / "export-old")
    table(old, "change_log", [dict(day=3, scope="s1", key=DELAY, old="0", new="17", author="ops")])
    stale = tmp_path / "stale"
    assert main(["--export", str(old), "--out", str(stale)]) == 0
    assert named(stale)[0] == ("revert", "s1", DELAY) and named(out)[0] == ("revert", "s2", DELAY)
    with pytest.raises(ValueError, match="not what the rule names"):
        evaluate(root, stale)
    # Fix sets under another method's name are not the rule's.
    other = shutil.copytree(out, tmp_path / "not-the-rule")
    meta = json.loads((other / "metadata.json").read_text())
    (other / "metadata.json").write_text(json.dumps({**meta, "method": "worst_health"}))
    with pytest.raises(ValueError, match="not an output of the build rule"):
        evaluate(root, other)
    # The command says so, exits with 2 and writes nothing.
    for directory in (edited[0], tmp_path / "other-rule", stale):
        with pytest.raises(SystemExit) as stop:
            main(
                [
                    "--evaluate",
                    str(directory),
                    "--world-root",
                    str(root),
                    "--out",
                    str(tmp_path / "e"),
                ]
            )
        assert stop.value.code == 2 and not (tmp_path / "e").exists()
    # The world's export changes after the rule read it: the directory is no longer its answer.
    assert evaluate(root, out)["budgets"] == whole
    table(
        root / "export",
        "change_log",
        [dict(day=3, scope="s1", key=DELAY, old="0", new="17", author="ops")],
    )
    with pytest.raises(ValueError, match="not what the rule names"):
        evaluate(root, out)
    assert evaluate(root, stale)["budgets"] != whole


def test_evaluation_values_the_scoring_episodes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, oracle_report = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    # The table holds ten episodes; let the scoring window be the five at s2, all replayed.
    world = oracle_world.load_world(root)  # the scorer tests' stand-in, see rule_world
    at_s2 = [i for i in world.identities if i.site_id == "s2"]
    assert len(at_s2) == 5 and len(world.identities) == 10
    monkeypatch.setattr(world, "scoring", lambda: at_s2)
    oracle_report.replay_verification["coverage"][0].update(attempts=5, attempts_executed_twice=5)
    result = evaluate(root, out)
    assert result["scoring_episodes"] == 5
    three = result["budgets"][0]
    assert three["rule"]["gain"] == {"rescued": 2, "broken": 0, "net": 2}
    assert three["rule"]["points"] == 40.0  # two of five, not two of ten


def test_evaluation_requires_a_ready_world_and_its_own_oracle_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, oracle_report = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    assert evaluate(root, out)["export_receipt_problem"] is None
    # The oracle report of another world is refused.
    own_hash = oracle_report.world_hash
    oracle_report.world_hash = "f" * 64
    with pytest.raises(ValueError, match="another world hash"):
        evaluate(root, out)
    oracle_report.world_hash = own_hash
    # A missing receipt of the export check is recorded where the world still counts as ready.
    (root / "export-checks/check-fixture.json").unlink()
    problem = evaluate(root, out)["export_receipt_problem"]
    assert problem is not None and "certify" in problem
    # A world that is not ready is refused, as the scorer refuses it.
    monkeypatch.setattr(
        "fleetlab.oracle.world.world_ready", lambda _: "runner.json: ready is False"
    )
    with pytest.raises(ValueError, match="ready is False"):
        evaluate(root, out)
    with pytest.raises(SystemExit) as stop:
        main(["--evaluate", str(out), "--world-root", str(root), "--out", str(tmp_path / "e")])
    assert stop.value.code == 2 and not (tmp_path / "e").exists()


def test_no_path_into_the_rules_directory_is_read_by_the_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "baselines/build_rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    # All the directory holds: the padded list, the four fix sets and the record. No list
    # lies anywhere without the record beside it; a baseline's budgets/ is left out.
    inside = sorted(out.rglob("*"))
    assert [str(p.relative_to(out)) for p in inside] == [
        "fix_sets",
        "fix_sets/cost-4.json",
        "fix_sets/cost-8.json",
        "fix_sets/count-3.json",
        "fix_sets/count-6.json",
        "metadata.json",
        "outbox",
        "outbox/fix_set.json",
    ]
    for name in ("build rule", "dramatic videos"):
        for path in (out, *inside):
            with pytest.raises(ValueError, match="not a baseline of the experiment"):
                command.score_world(root, analysis.root, baselines={name: path})
    # The two paths the review found accepted: a per-budget list and its file, which the
    # scorer cut with the padding into six, nine and ten actions. They do not exist.
    for found in (out / "budgets/count-6", out / "budgets/count-6/outbox/fix_set.json"):
        assert not found.exists() and not (out / "budgets").exists()
        with pytest.raises((ValueError, OSError)):
            command.score_world(root, analysis.root, baselines={"build rule": found})


def test_the_scorer_refuses_the_rules_directory_under_any_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = rule_world(tmp_path, monkeypatch)
    out = tmp_path / "baselines/build_rule"
    assert main(["--export", str(root / "export"), "--out", str(out)]) == 0
    handed = (out, out / "fix_sets", out / "fix_sets/count-3.json", out / "outbox/fix_set.json")
    for name in ("build rule", "worst health metric", "some other list"):
        for path in handed:
            with pytest.raises(ValueError, match="not a baseline of the experiment"):
                command.score_world(root, analysis.root, baselines={name: path})
    arguments = ["--world-root", str(root), "--analysis", str(analysis.root)]
    with pytest.raises(ValueError, match="not a baseline of the experiment"):
        command.main([*arguments, "--baseline", f"build rule={out}", "--out", str(tmp_path / "s")])
    assert not (tmp_path / "s").exists()
    # What is refused is the source's own word, not the fix sets: without it they are a list.
    plain = plain_list(out, tmp_path / "plain")
    report = command.score_world(root, analysis.root, baselines={"some other list": plain})
    assert report.claim3.budgets[0].baselines["some other list"].points == 20.0
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["baseline_of_the_experiment"] is False


def test_scorer_output_is_the_same_bytes_with_and_without_the_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = rule_world(tmp_path, monkeypatch)
    export = root / "export"
    baselines = {
        "worst health metric": simple_source(tmp_path / "baselines", export),
        "classical statistics": classical_source(tmp_path / "classical", export),
    }
    ranked = inputs().fix_set
    assert ranked is not None
    for path in baselines.values():
        (path / "outbox/fix_set.json").write_text(ranked.model_dump_json())
    arguments = ["--world-root", str(root), "--analysis", str(analysis.root)]
    for name, path in baselines.items():
        arguments += ["--baseline", f"{name}={path}"]

    def scored(out: Path) -> tuple[bytes, bytes]:
        assert command.main([*arguments, "--out", str(out)]) == 0
        return (out / "report.json").read_bytes(), (out / "report.md").read_bytes()

    before = scored(tmp_path / "score-before")
    # The rule's directory and its evaluation, everywhere a careless run could leave them:
    # beside the baselines of the same producer run, in the world's root, in its oracle
    # directory and beside the analysis.
    first = tmp_path / "baselines/build_rule"
    assert main(["--export", str(export), "--out", str(first)]) == 0
    places = [first, root / "build_rule", root / "oracle/build_rule", analysis.root.parent / "rule"]
    for place in places[1:]:
        shutil.copytree(first, place)
    for place in places:
        file = place.with_name(place.name + "-evaluation.json")
        assert main(["--evaluate", str(first), "--world-root", str(root), "--out", str(file)]) == 0
        assert json.loads(file.read_text())["budgets"][0]["rule"]["points"] == 20.0
    after = scored(tmp_path / "score-after")
    assert after == before
    for text in after:
        assert b"build_rule" not in text and b"build rule" not in text
    # The margin and the best baseline come from the fixed opponents alone.
    report = json.loads(after[0])
    assert set(report["claim3"]["budgets"][0]["baselines"]) >= set(baselines)
    assert not any("rule" in name for name in report["claim3"]["budgets"][0]["baselines"])
