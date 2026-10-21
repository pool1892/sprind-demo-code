"""The exhibit (fleetlab.loop.exhibit): pages from a run's records, provenance, origin marks,
bounds that stay bounds, the working table and the paired film. CPU only: invented
records from `fleetlab.loop.exhibit.fixtures`, small synthetic videos."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from fleetlab.dashboard.data import Rate
from fleetlab.loop.exhibit import build, derive, load_run
from fleetlab.loop.exhibit.fixtures import (
    CAMERA,
    aborted_probe_costs,
    completed_probe_costs,
    failed_probe_costs,
    write_fixture,
)
from fleetlab.loop.exhibit.records import (
    FIXTURE_MARKER,
    Arm,
    Experiment,
    Sources,
    load_investigation,
    load_truth,
)


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fixture(tmp_path_factory.mktemp("exhibit") / "run", videos=True)


@pytest.fixture(scope="module")
def bundle(fixture_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    view = load_run(fixture_run.parents[1] / "exhibit.json")
    return build(view, tmp_path_factory.mktemp("bundle") / "b", generated="test")


def _pages(bundle: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(bundle.glob("*.md"))}


def test_bundle_has_its_pages_and_every_source_is_in_the_provenance(bundle: Path) -> None:
    pages = _pages(bundle)
    assert set(pages) == {"index.md", "round-1.md", "round-2.md", "provenance.md"}
    provenance = json.loads((bundle / "provenance.json").read_text())
    items = {i["item"]: i for i in provenance["items"]}
    cited = {
        m
        for text in pages.values()
        for m in re.findall(r"provenance\.md#user-content-([\w-]+)", text)
    }
    assert cited and cited <= set(items)
    for item in items.values():
        assert item["files"], item["item"]
        assert set(item["files"]) <= set(provenance["files"])
        assert item["derivation"]
    for record in provenance["files"].values():
        assert Path(record["path"]).is_file()
        assert len(record["sha256"]) == 64


def test_every_page_and_table_built_from_a_fixture_says_so(bundle: Path) -> None:
    for name, text in _pages(bundle).items():
        if name == "provenance.md":
            assert "origin: fixture" in text
            continue
        assert "**FIXTURE.**" in text, name
        captions = re.findall(r"^\*(?:Table|Figure) [\d.]+\. (.*)$", text, re.M)
        assert captions, name
        assert all(c.startswith("**[FIXTURE]**") for c in captions), name


def test_fleet_table_counts_match_the_export(fixture_run: Path, bundle: Path) -> None:
    rows = pq.read_table(
        fixture_run.parents[1] / "rounds" / "r1" / "export" / "log" / "episodes.parquet"
    ).to_pylist()
    ops = [r for r in rows if r["experiment_id"] is None and r["day"] in (0, 1)]
    ok = sum(r["outcome"] == "success" for r in ops)
    page = (bundle / "round-1.md").read_text()
    assert f"| all |  | {ok}/{len(ops)} = " in page


def test_working_table_has_confidence_before_and_after_and_a_withdrawal(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    table = derive.working_table(view, view.rounds[0])
    by_target = {row[0].split(",")[0]: row for row in table.rows}
    kept = by_target["scene_camera at r02"]
    dropped = by_target["scene_camera at r01"]
    assert (kept[4], kept[5], kept[6]) == ("0.50", "0.85", "claimed in the final diagnosis")
    assert (dropped[4], dropped[5]) == ("0.30", "not stated")
    assert dropped[6].startswith("withdrawn")
    assert kept[1] == "session 1 (diagnosis)"


def test_experiment_ledger_reads_paired_arms(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    experiment = view.rounds[0].experiments[0]
    results = {r.arm.index: r for r in derive.arm_results(experiment)}
    assert results[1].paired and results[1].rescued == 6 and results[1].broken == 0
    assert results[1].verdict == "better than control"
    assert results[2].verdict == "no difference detected"
    table = derive.experiment_table(view, view.rounds[0])
    assert any("adopted (L1)" in row[-1] for row in table.rows)
    assert any(row[-1] == "not adopted" for row in table.rows)


def test_unpaired_arms_get_an_interval_and_missing_outcomes_are_counted() -> None:
    def rows(outcomes: list[str | None], task: str) -> list[dict[str, Any] | None]:
        return [None if o is None else {"outcome": o, "task_text": task} for o in outcomes]

    control = Arm(0, [], ["a", "b", "c", "d"], rows(["failure"] * 4, "t"))
    arm = Arm(1, [CAMERA("r01")], ["e", "f", "g", "h"], rows(["success"] * 3 + [None], "u"))
    experiment = Experiment(
        "x0001", 1, 1, {"robots": ["r01"], "paired_starts": True}, "done", "", [control, arm], []
    )
    (result,) = derive.arm_results(experiment)
    assert not result.paired  # different tasks and a missing row: not read as pairs
    assert result.missing == 1
    text = derive.result_text(result)
    assert "[" in text and "without a recorded outcome" in text


def test_costs_keep_unknowns_and_lower_bounds() -> None:
    complete = {
        "model_starts": 2,
        "starts_without_usage": 0,
        "model_seconds": 120.0,
        "tokens": {"input_tokens": 10, "cached_input_tokens": 5},
    }
    partial = dict(complete, starts_without_usage=1)
    none = dict(complete, starts_without_usage=2, tokens={"input_tokens": 0})
    assert derive._tokens_text(complete, "input_tokens", "cached_input_tokens") == "10 (5)"
    assert derive._tokens_text(partial, "input_tokens").startswith("at least 10")
    assert derive._tokens_text(none, "input_tokens").startswith("not recorded")
    assert derive._tokens_text({}, "input_tokens").startswith("not recorded")
    assert derive._seconds_text(partial) == "at least 2.0 min"
    assert derive._seconds_text(none).startswith("not recorded")
    assert derive._seconds_text({}) == "not recorded"
    flagged = dict(complete, starts_without_usage=None, usage_complete=False)
    assert derive._tokens_text(flagged, "input_tokens").startswith("at least")


def test_a_nested_lower_bound_price_stays_a_lower_bound() -> None:
    bounded = {"status": "bounded_api_equivalent", "low_usd": 1.25, "high_usd": 2.0}
    once = {"status": "lower_bound", "reason": "r", "partial": bounded}
    twice = {"status": "lower_bound", "reason": "r", "partial": once}
    assert derive.usd_text(bounded).startswith("1.25 to 2.00 USD")
    assert derive.usd_text(once).startswith("at least 1.25 USD")
    assert derive.usd_text(twice).startswith("at least 1.25 USD")
    empty = {"status": "lower_bound", "reason": "r", "partial": {"status": "unavailable"}}
    assert "0.00" not in derive.usd_text(empty) and derive.usd_text(empty).startswith(
        "not recorded"
    )
    assert derive.usd_text(None) == "not recorded"
    assert derive.usd_text({"status": "unavailable", "reason": "no price"}).startswith(
        "not available"
    )


def test_reuse_table_marks_the_incomplete_arm(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    view.reuse = json.loads((fixture_run.parents[1] / "reuse" / "comparison.json").read_text())
    view.reuse_files = []
    table = derive.reuse_table(view)
    assert table is not None
    rows = {r[0]: r for r in table.rows}
    assert rows["model time (s)"][1].startswith("at least")
    assert not rows["model time (s)"][2].startswith("at least")
    assert rows["model time (s)"][3].startswith("no difference")
    assert rows["price equivalent"][1].startswith("at least 12.50 USD")
    assert rows["experiment attempts simulated"][3] == "-60"


def test_film_is_made_from_the_first_rescued_pair(bundle: Path) -> None:
    page = (bundle / "round-1.md").read_text()
    assert "clips/r1-clip-x0001-arm1.gif" in page
    assert (bundle / "clips" / "r1-clip-x0001-arm1.gif").stat().st_size > 0
    assert (bundle / "clips" / "r1-clip-x0001-arm1.strip.png").is_file()
    assert (bundle / "clips" / "r1-clip-x0001-arm1.before.mp4").is_file()
    assert "the first paired position (2 of 8)" in page


def test_reserved_panel_film_shows_a_rescued_and_a_broken_attempt(bundle: Path) -> None:
    index = (bundle / "index.md").read_text()
    for kind in ("rescued", "broken"):
        assert f"clips/panel-clip-{kind}.gif" in index
        assert (bundle / "clips" / f"panel-clip-{kind}.strip.png").is_file()
    assert "the first reserved attempt by episode id rescued" in index


def test_replacements_undos_and_retired_entries_are_shown(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    rnd = view.rounds[1]
    assert rnd.decision is not None and rnd.ledger_after is not None
    rnd.decision["replace"] = [
        {"entry_id": "L1", "action": CAMERA("r04"), "rationale": "fixture: replace"}
    ]
    rnd.decision["undo"] = [{"entry_id": "L9", "rationale": "fixture: undo"}]
    tables = derive.decision_tables(view, rnd)
    kinds = [row[0] for row in tables[0].rows]
    assert kinds == ["adopt", "replace L1", "undo L9"]
    assert "forecast, not a measurement" in tables[0].caption
    assert "The agent also decided to stop." in tables[0].caption


def test_learned_skills_follow_their_recorded_source(bundle: Path) -> None:
    round_1 = (bundle / "round-1.md").read_text()
    round_2 = (bundle / "round-2.md").read_text()
    index = (bundle / "index.md").read_text()
    # Frozen with no round but session 4: the session map puts it in round 2 only.
    assert "fixture-one-robot-misses" not in round_1.split("## The procedure")[1]
    assert "**fixture-one-robot-misses**" in round_2
    assert "(round 2; session 4; checked and frozen;" in round_2
    assert "session 0" not in round_1 + round_2 + index
    # Another investigation's skill and one without a source stay off the round pages.
    assert "fixture-carried-skill" not in round_1 + round_2
    assert "**fixture-carried-skill**" in index and "carried from library cccccccccccccccc" in index
    assert (
        "**fixture-skill-without-source** **[FIXTURE]** (round not recorded; session not " in index
    )
    assert "Drafts the host's check refused and did not keep: fixture-refused-draft" in index
    assert "host-side reason" not in index


def test_one_look_plan_reports_its_counts_and_no_reason(bundle: Path) -> None:
    index = (bundle / "index.md").read_text()
    assert (
        "Requested cutoff: 2. Actions listed: 2. Adoptable among them: 2. Applied in the plan: 1."
        in index
    )
    assert "shorter" not in index and "pad" not in index


def test_headline_uses_the_cost_tables_rules(fixture_run: Path, bundle: Path) -> None:
    index = (bundle / "index.md").read_text()
    # Two rounds of 24 completed attempts and one confirmed failed simulation each.
    assert (
        "| [E] | robot executions of experiments and probes: attempts run, plus any execution "
        "that failed and was run again (all rounds) | 50 task executions; probe executions "
        "not recorded |" in index
    )
    assert "| [T] | minutes of model time (all rounds) | 20.0 |" in index
    view = load_run(fixture_run)
    view.rounds[0].costs["usage_complete"] = False
    view.rounds[0].costs["starts_without_usage"] = None
    assert derive.model_minutes(view.rounds) == "at least 20.0"
    assert derive._seconds_text(view.rounds[0].costs).startswith("at least")


def test_simulation_failures_and_missing_timing_stay_apart() -> None:
    failed_queue: dict[str, Any] = {
        "state": "failed",
        "simulated": {
            "run_jobs": 4,
            "failed_confirmed": 2,
            "partial_jobs": 1,
            "unconfirmed": 3,
            "replays": 1,
        },
    }
    done_queue: dict[str, Any] = {"state": "done", "timing": {}, "simulated": {"run_jobs": 6}}
    costs: dict[str, Any] = {"arm_attempts": 10, "queues": [failed_queue, done_queue]}
    sim = derive.simulation(costs)
    assert (sim.failed, sim.failed_queues, sim.unconfirmed, sim.replays) == (3, 1, 3, 1)
    assert sim.seconds is None
    assert sim.completed == 10  # the queues' run jobs, 4 + 6
    timed = dict(
        costs, queues=[dict(failed_queue, timing={"simulation_seconds": 60.0}), done_queue]
    )
    assert derive.simulation(timed).seconds == 60.0 and derive.simulation(timed).partial_timing


def test_a_failed_queue_counts_its_finished_jobs_without_a_delivered_result(
    fixture_run: Path,
) -> None:
    # The queue failed before it delivered a result: no arm attempts in the results, but
    # 4 finished run jobs and 1 confirmed failed simulation are recorded.
    costs: dict[str, Any] = {
        "arm_attempts": 0,
        "experiments": [{"experiment_id": "x0001", "status": "pending", "attempts": 0}],
        "queues": [
            {
                "state": "failed",
                "simulated": {
                    "run_jobs": 4,
                    "failed_confirmed": 1,
                    "partial_jobs": 0,
                    "unconfirmed": 0,
                    "replays": 0,
                },
            }
        ],
    }
    sim = derive.simulation(costs)
    assert (sim.completed, sim.failed, sim.failed_queues) == (4, 1, 1)
    view = load_run(fixture_run)
    view.rounds[0].costs = costs
    assert derive.experiment_attempts(view.rounds[:1]) == "5"
    legacy = {
        "arm_attempts": 0,
        "queues": [
            {
                "state": "done",
                "simulated_jobs": 9,
                "simulated_run_jobs": 8,
                "partial_jobs": 1,
                "planned_attempts": 9,
                "delivered_attempts": 8,
            }
        ],
    }
    assert derive.simulation(legacy).completed == 8
    assert derive.simulation({"arm_attempts": 3}).completed == 3


def _queue_with(part: dict[str, Any]) -> dict[str, Any]:
    """A round cost record holding only a probe (no experiment arms), as lane probe writes."""
    part = dict(part)
    queue: dict[str, Any] = {"state": "done", "simulated": {"run_jobs": 0, "replays": 0}}
    queue.update(part.pop("queue"))
    return {"arm_attempts": 0, "experiments": [], **part, "queues": [queue]}


def test_a_completed_probe_is_counted_apart_from_task_attempts() -> None:
    costs = _queue_with(completed_probe_costs(1))
    probe = derive.probe_simulation(costs)
    assert (probe.executed, probe.replays, probe.failed, probe.reserved) == (6, 6, 0, 0)
    assert (probe.unconfirmed, probe.charged, probe.requests, probe.refused) == (0, 6, 1, 0)
    rows = derive._probe_rows(costs)
    assert rows["probe attempts executed"] == "6"
    assert rows["probe chain-check replays"] == "6"
    assert rows["probe executions (failed ones included)"] == "6"
    sim = derive.simulation(costs)
    assert (sim.completed, sim.replays) == (0, 0)  # the task side stays its own


def test_a_failed_probe_keeps_its_known_execution_and_its_reservation() -> None:
    costs = _queue_with(failed_probe_costs(3))
    probe = derive.probe_simulation(costs)
    assert (probe.executed, probe.failed, probe.unconfirmed, probe.reserved) == (1, 1, 2, 4)
    rows = derive._probe_rows(costs)
    assert rows["probe executions (failed ones included)"] == "at least 2"
    assert rows["probe attempts reserved (a failed probe's rest, unknown)"] == "4"
    assert rows["probe simulations not confirmed"] == "2"
    # Without the engine's probe block, the outcome's known execution and reservation.
    bare = {k: v for k, v in costs.items() if k != "queues"}
    assert derive.probe_simulation(bare).executed == 2
    assert derive.probe_attempts_of(bare) == "at least 2"


def test_an_aborted_queue_leaves_its_probe_request_unresolved(fixture_run: Path) -> None:
    costs = _queue_with(aborted_probe_costs(1))
    costs["queues"][0]["state"] = "failed"
    probe = derive.probe_simulation(costs)
    assert (probe.executed, probe.failed, probe.unconfirmed, probe.reserved) == (1, 0, 0, 0)
    assert (probe.unresolved, probe.unresolved_charged, probe.failed_queues) == (1, 6, 1)
    assert derive.probe_attempts_of(costs) == "at least 1"
    rows = derive._probe_rows(costs)
    assert rows["probe requests unresolved (outcome unknown)"] == (
        "1 request unresolved: 6 charged, outcome unknown (the queue run failed before its reply)"
    )
    assert rows["probe attempts executed"] == "1"  # the known one, nothing invented
    view = load_run(fixture_run)
    view.rounds[0].costs = costs
    assert derive.robot_attempts(view.rounds[:1]) == "at least 1 (0 task, at least 1 probe)"
    # Without the queue's probe list or outcomes, the cost record's status decides alone.
    alone = {k: v for k, v in costs.items() if k != "queues"}
    assert derive.probe_simulation(alone).unresolved == 1
    assert derive.probe_attempts_of(alone).startswith("at least")


def test_a_round_without_experiments_and_the_second_public_ledger_format() -> None:
    # A round that requested nothing simulated nothing: exact zeros, not a lower bound.
    sim = derive.simulation({"experiments": [], "requests": 0, "arm_attempts": 0, "queues": []})
    assert (sim.completed, sim.failed, sim.unconfirmed, sim.replays) == (0, 0, 0, 0)
    assert sim.seconds is None
    # The public ledger's second format keeps its entries under "changes".
    ledger = {"format": "fleetlab.public-ledger/2", "changes": [{"entry_id": "L1"}]}
    assert derive.ledger_entries(ledger) == [{"entry_id": "L1"}]
    assert derive.ledger_entries({"entries": [{"entry_id": "L2"}]}) == [{"entry_id": "L2"}]
    assert derive.ledger_entries(None) == []


def test_probes_in_the_bundle_and_the_headline(tmp_path: Path) -> None:
    engagement = write_fixture(tmp_path / "run", probes=True, reuse=False)
    view = load_run(engagement.parents[1] / "exhibit.json")
    assert derive.robot_attempts(view.rounds) == "at least 58 (50 task, at least 8 probe)"
    out = build(view, tmp_path / "bundle", generated="test", clips=False)
    index = (out / "index.md").read_text()
    assert "at least 58 (50 task, at least 8 probe)" in index
    assert "probes requested: 2 (0 refused)" in index
    assert "| probe executions (failed ones included) | 6 | at least 2 |  |" in index
    assert "| probe chain-check replays | 6 | 0 |" in index


def test_a_record_from_before_probes_says_so(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    assert not derive.probe_simulation(view.rounds[0].costs).recorded
    assert derive.robot_attempts(view.rounds) == "50 task executions; probe executions not recorded"
    column = derive._cost_column(view, view.rounds[0])
    assert column["probe attempts"].startswith("not recorded")


def test_every_simulation_by_purpose_from_the_driver_record(bundle: Path) -> None:
    index = (bundle / "index.md").read_text()
    # Operation 4 robots x 4 per day x 2 days (round 2 only), experiments 2 x 24, evaluation
    # 4 robots x 8 x 5 states; replays 3 + 4 + 5.
    assert "| the fleet's regular work after round 1 | 32 | 3 | 0 | 0 |" in index
    assert "| all | 240 | 12 | 0 | 0 |" in index


def test_an_opening_from_the_annotations_and_the_procedures_heading(
    fixture_run: Path, tmp_path: Path
) -> None:
    view = load_run(fixture_run)
    view.annotations = {
        "opening": {"title": "In short", "by": "the lead", "paragraphs": ["One.", "Two  two."]}
    }
    out = build(view, tmp_path / "b", generated="test", clips=False)
    index = (out / "index.md").read_text()
    head = index.split("*Table 1.")[0]
    assert "## In short\n\nOne.\n\nTwo two.\n\n*Written for this page by the lead." in head
    # Without a reuse comparison the section does not say the procedure was used again.
    assert "## The procedures the agent wrote" in index
    assert "used again" not in index


def test_the_forecast_is_shown_beside_the_last_day_for_context(
    fixture_run: Path, tmp_path: Path
) -> None:
    view = load_run(fixture_run)
    table = derive.next_round_check(view, view.rounds[0])
    assert table is not None
    pooled, last = table.rows[0], table.rows[1]
    assert "pooled" in pooled[1] and pooled[5] == "shown beside the last day, for context"
    assert last[1] == "last day of regular work before the decision (round 1, day 1)"
    assert last[5] == "+6.00 (+2.00 to +10.00)"
    assert table.headers[5] == "agent's forecast gain over the current fleet state"
    ops = view.rounds[0].ops_rows
    day1 = [r for r in ops if r["day"] == 1]
    ok = sum(r["outcome"] == "success" for r in day1)
    assert last[2].startswith(f"{ok}/{len(day1)} = ")
    assert "for context" in table.caption and "other reference periods" in table.caption
    assert "not the reserved evidence" in table.caption
    assert "stated over" not in table.caption + " ".join(" ".join(r) for r in table.rows)
    # A run's own explanation of its forecast's reference periods goes in as a note.
    view.annotations = {
        "by": "the lead",
        "notes": [{"page": "round-1", "section": "next", "text": "per-component rates"}],
    }
    out = build(view, tmp_path / "b", generated="test", clips=False)
    section = (out / "round-1.md").read_text().split("### What the next round")[1]
    assert section.index("per-component rates") < section.index("*Table")


def test_plain_names_on_the_pages_identifiers_in_provenance(
    fixture_run: Path, tmp_path: Path
) -> None:
    view = load_run(fixture_run)
    view.annotations = {"names": {"world": "the test fleet", "agent": "the analysis agent"}}
    view.label = "a test run"  # the title is the manifest's label, chosen by its author
    out = build(view, tmp_path / "b", generated="test", clips=False)
    pages = (out / "index.md").read_text() + (out / "round-1.md").read_text()
    assert "Fleet: the test fleet. Agent: the analysis agent." in pages
    assert "The fleet (the test fleet) did its regular work" in pages
    assert "w99" not in pages and "fixture-model" not in pages
    provenance = (out / "provenance.md").read_text()
    assert "Exact identifiers of this run: world w99; model fixture-model" in provenance


def test_a_manifest_hub_id_becomes_the_summary_front_matter(
    fixture_run: Path, bundle: Path, tmp_path: Path
) -> None:
    manifest = json.loads((fixture_run.parents[1] / "exhibit.json").read_text())
    manifest["hub_id"] = "exampleid1"
    path = fixture_run.parents[1] / "exhibit-with-id.json"
    path.write_text(json.dumps(manifest))
    out = build(load_run(path), tmp_path / "b", generated="test", clips=False)
    lines = (out / "index.md").read_text().splitlines()
    assert lines[:2] == ["---", "id: exampleid1"]
    assert lines[2].startswith('title: "') and lines[3] == "---"
    assert "id:" not in (out / "round-1.md").read_text().split("---")[1]
    # Without an id, none is written.
    assert not any(
        line.startswith("id:") for line in (bundle / "index.md").read_text().splitlines()[:4]
    )
    manifest["hub_id"] = "bad id\ntitle: x"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hub_id"):
        load_run(path)


def test_a_plain_search_comparator_uses_its_own_cost_receipt(tmp_path: Path) -> None:
    engagement = write_fixture(tmp_path / "run", comparators=True, reuse=False)
    view = load_run(engagement.parents[1] / "exhibit.json")
    table = derive.comparator_table(view)
    assert table is not None
    agent, search, pending = table.rows
    # The receipt's normalized costs, not the engagement's raw ones (2 scripted starts).
    assert search[0] == "plain search (told the site and the setting)"
    assert search[1] == "adopted: set control.latency_compensation_steps at s1 to 15"
    assert search[5:11] == [
        "144 (+21 replays)",
        "2 (2 scripted)",
        "0",
        "0.0 min",
        "15",
        "0.00 to 0.00 USD (API-equivalent bound, not a bill); no model was called",
    ]
    assert pending[2] == "**[PENDING]** evaluation queued; to be filled once it finishes"
    assert agent[2] == "2" and agent[5] == "50 (+4 replays)"
    assert agent[0] == "the analysis agent"
    record = str((engagement.parents[1] / "plain-search" / "plain_search.json").resolve())
    assert record in table.files
    breakdown = derive.comparator_breakdown(view)
    assert breakdown is not None
    assert breakdown.headers[-1] == "plain search (told the site and the setting)"
    r02 = next(r for r in breakdown.rows if r[1] == "r02")
    assert r02[4] == "8/8"  # the searched robot all rescued in this fixture
    out = build(view, tmp_path / "b", generated="test", clips=False)
    index = (out / "index.md").read_text()
    assert "## Beside other methods on the same reserved attempts" in index
    provenance = json.loads((out / "provenance.json").read_text())
    assert record in provenance["files"]


def test_incidents_sites_executions_and_the_library_fingerprint(
    fixture_run: Path, tmp_path: Path
) -> None:
    view = load_run(fixture_run)
    run_dir = tmp_path / "queue-run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "extra": {
                    "incidents": [
                        {
                            "why": "exit code -6",
                            "jobs": ["e002386"],
                            "assignment_status": {"e002386": "execution_started"},
                            "recovered": True,
                            "replay_checks": {"e002386": {"checked": True, "verdict": "identical"}},
                        }
                    ]
                }
            }
        )
    )
    queue = view.rounds[1].costs["queues"][0]
    queue["simulated"] = dict(queue["simulated"], incidents=1, run_dir=str(run_dir))
    queue["session"] = 4
    table = derive.incidents_table(view)
    assert table is not None
    assert table.rows == [
        [
            "2",
            "4",
            "e002386",
            "worker stopped (exit code -6)",
            "e002386: execution_started",
            "yes, run again",
            "e002386: identical",
        ]
    ]
    assert str((run_dir / "run.json").resolve()) in table.files
    assert "attributes no cause" in table.caption
    sites = derive.panel_sites(view)
    assert sites is not None and [r[0] for r in sites.rows] == ["s1", "s2"]
    assert all(cell.endswith("/16") for row in sites.rows for cell in row[1:])
    simulated = derive.simulated_table(view)
    assert simulated is not None and simulated.rows[-1][-1] == "252"
    out = build(view, tmp_path / "b", generated="test", clips=False)
    index = (out / "index.md").read_text()
    assert "## Where the change happened, by site" in index
    assert "*Table" in index.split("## Where the change happened, by site")[1].split("##")[0]
    assert "fingerprint f1x700000000… (full: `f1x7" in index
    assert "1 kept, 1 refused." in index
    assert "Incidents in the simulations" in index


def _with_comparator(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    engagement = write_fixture(tmp_path / "run", comparators=True, reuse=False)
    manifest_file = engagement.parents[1] / "exhibit.json"
    return manifest_file, json.loads(manifest_file.read_text())


def test_comparisons_need_recorded_matching_panel_fingerprints(tmp_path: Path) -> None:
    manifest_file, _ = _with_comparator(tmp_path)
    view = load_run(manifest_file)
    assert view.evaluation is not None
    search = view.comparators[0]
    assert search.record is not None
    # Both fingerprints missing: never equal, nothing compared.
    view.evaluation.summary["panel_sha256"] = ""
    search.record["evaluation"]["panel_sha256"] = ""
    table = derive.comparator_table(view)
    assert table is not None
    assert table.rows[1][2] == "not comparable: the agent's evaluation records no panel fingerprint"
    assert derive.comparator_breakdown(view) is None
    # The agent's recorded, the method's missing.
    view.evaluation.summary["panel_sha256"] = "a" * 64
    table = derive.comparator_table(view)
    assert table is not None
    assert (
        table.rows[1][2] == "not comparable: this method's evaluation records no panel fingerprint"
    )
    assert derive.comparator_breakdown(view) is None
    # Both recorded but different.
    search.record["evaluation"]["panel_sha256"] = "b" * 64
    table = derive.comparator_table(view)
    assert table is not None
    assert table.rows[1][2] == "not comparable: measured on a different panel"


def test_attempts_that_do_not_belong_to_the_method_are_refused(tmp_path: Path) -> None:
    manifest_file, manifest = _with_comparator(tmp_path)
    # The engagement's own attempts substituted under the search's receipt.
    manifest["comparators"][0]["attempts"] = "evaluation/attempts.parquet"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="do not belong to its evaluation"):
        load_run(manifest_file)


def test_comparator_costs_keep_unknowns_and_lower_bounds() -> None:
    costs: dict[str, Any] = {
        "arm_attempts": 10,
        "queues": [
            {
                "state": "done",
                "simulated": {
                    "run_jobs": 10,
                    "replays": 2,
                    "failed_confirmed": 1,
                    "partial_jobs": 0,
                    "unconfirmed": 3,
                },
            }
        ],
        "model_seconds": 0.0,
    }
    cells = derive._cost_cells(costs)
    assert cells[0] == "at least 11 (+2 replays)"
    assert cells[1] == "not recorded"  # sessions missing
    assert cells[2] == "not recorded"  # model starts missing
    assert cells[4] == "not recorded"  # tool calls missing


def test_a_lasting_condition_reads_as_such_with_the_adaptation_beside_it(
    fixture_run: Path,
) -> None:
    from fleetlab.loop.exhibit.records import TruthView

    view = load_run(fixture_run)
    lasting = {
        "fault_id": "f9",
        "kind": "remote_link_latency",
        "target": {
            "scope": "r01",
            "component": "controller",
            "config_key": None,
            "condition_key": None,
        },
        "schedule": {"onset_day": 0},
    }
    removable = {
        "fault_id": "f1",
        "kind": "gripper_wear",
        "target": {
            "scope": "r02",
            "component": "scene_camera",
            "config_key": None,
            "condition_key": None,
        },
        "schedule": {"onset_day": 0},
    }
    view.truth = TruthView("w99", fixture_run, [lasting, removable], None, None, [])
    final = view.rounds[-1].ledger_after
    assert final is not None
    setting: dict[str, Any] = {
        "verb": "set",
        "target": {
            "scope": "s1",
            "component": None,
            "config_key": "control.latency_compensation_steps",
            "condition_key": None,
        },
        "value": 18,
    }
    target: dict[str, Any] = dict(setting["target"])
    elsewhere = dict(setting, target={**target, "scope": "s2"})
    final["entries"] = [
        *final["entries"],
        {"entry_id": "L8", "round": 1, "action": setting, "status": "active"},
        {"entry_id": "L9", "round": 1, "action": elsewhere, "status": "active"},
    ]
    assert derive.lasting("remote_link_latency") and not derive.lasting("gripper_wear")
    table = derive.truth_table(view)
    assert table is not None
    row = table.rows[0]
    assert row[4] == "lasting: no removing action exists; no setting in force at this site"
    assert row[5] == (
        "lasting: no removing action exists; actions at this site from an earlier round, still "
        "in force: set control.latency_compensation_steps at s1 to 18 (L8, round 1)"
    )
    assert "L9" not in row[4]  # another site's setting is not listed
    assert table.rows[1][4].startswith("named")  # a removable fault keeps the matching


def test_a_setting_undone_in_round_2_is_not_shown_in_force(fixture_run: Path) -> None:
    from fleetlab.loop.exhibit.records import TruthView

    view = load_run(fixture_run)
    lasting = {
        "fault_id": "f9",
        "kind": "remote_link_latency",
        "target": {
            "scope": "r01",
            "component": "controller",
            "config_key": None,
            "condition_key": None,
        },
        "schedule": {"onset_day": 0},
    }
    view.truth = TruthView("w99", fixture_run, [lasting], None, None, [])
    setting: dict[str, Any] = {
        "verb": "set",
        "target": {
            "scope": "s1",
            "component": None,
            "config_key": "control.latency_compensation_steps",
            "condition_key": None,
        },
        "value": 18,
    }
    first, second = view.rounds[0].ledger_after, view.rounds[1].ledger_after
    assert first is not None and second is not None
    first["entries"] = [{"entry_id": "L8", "round": 1, "action": setting, "status": "active"}]
    second["entries"] = [{"entry_id": "L8", "round": 1, "action": setting, "status": "undone"}]
    table = derive.truth_table(view)
    assert table is not None
    round_1, round_2 = table.rows[0][4], table.rows[0][5]
    assert round_1.endswith(
        "actions at this site: set control.latency_compensation_steps at s1 to 18 (L8, round 1)"
    )
    assert "in force" not in round_2
    assert round_2.endswith(
        "undone at this site in this round: set "
        "control.latency_compensation_steps at s1 to 18 (L8, round 1)"
    )


def test_zero_counts_are_valid_and_absent_counts_are_not() -> None:
    from fleetlab.loop.exhibit.records import attempts_problems

    def record(successes: int | None, attempts: int | None) -> dict[str, Any]:
        state = {"name": "original", "successes": successes, "attempts": attempts}
        return {"evaluation": {"states": [state]}}

    rows = [{"state": "original", "success": False}, {"state": "original", "success": False}]
    nowhere = Path("/nonexistent/attempts.parquet")
    assert attempts_problems(record(0, 2), rows, nowhere) == []  # 0 of 2 is a valid state
    assert attempts_problems(record(0, 0), [], nowhere) == []  # so is no attempt at all
    assert attempts_problems(record(None, 2), rows, nowhere)  # an absent count is not zero


def test_the_site_caption_claims_only_an_unchanged_count(fixture_run: Path) -> None:
    view = load_run(fixture_run)
    sites = derive.panel_sites(view)
    assert sites is not None
    assert "only its success count is unchanged" in sites.caption
    assert "was not changed" not in sites.caption


def test_the_panel_beside_comparator_attempts_is_in_the_provenance(tmp_path: Path) -> None:
    manifest_file, _ = _with_comparator(tmp_path)
    search = manifest_file.parent / "plain-search"
    summary = json.loads((search / "plain_search.json").read_text())
    panel = search / "evaluation" / "panel.json"
    panel.write_text('{"panel": "fixture"}')
    summary["evaluation"]["panel_sha256"] = hashlib.sha256(panel.read_bytes()).hexdigest()
    (search / "plain_search.json").write_text(json.dumps(summary))
    view = load_run(manifest_file)
    assert str(panel.resolve()) in view.sources.files
    assert str(panel.resolve()) in view.comparators[0].files


def test_summary_fills_the_headline_and_says_what_is_missing(bundle: Path) -> None:
    index = (bundle / "index.md").read_text()
    assert "| [A] | reserved attempts, as at the start |" in index
    assert "| [X] | reuse: experiment attempts without the procedure | 120 |" in index
    assert "| reference: fixed-procedure |" in index
    assert "Reference methods on the same attempts: reference: fixed-procedure" in index
    assert "the planted truth" in index.split("## What this bundle does not show")[1]
    assert "Possible negative transfer" not in index


def test_investigation_format_shows_the_ranked_list_with_labels(tmp_path: Path) -> None:
    engagement = write_fixture(tmp_path / "run", rounds=(1,), reuse=False, learned=False)
    record = json.loads(engagement.read_text())
    analysis = Path(record["analysis"])
    submissions = analysis / "outbox" / "submissions.jsonl"
    lines = [json.loads(x) for x in submissions.read_text().splitlines()]
    lines = [x for x in lines if x["kind"] != "decision"]
    ranked = [
        {
            "action": CAMERA(f"r0{i}"),
            "expected_gain": float(5 - i),
            "uncertainty": {"low": 0.0, "high": 6.0},
            "rationale": "fixture",
        }
        for i in range(1, 5)
    ]
    lines.append(
        {"kind": "fix_set", "session": 2, "receipt_id": "fix_set-1", "answer": {"actions": ranked}}
    )
    submissions.write_text("".join(json.dumps(x) + "\n" for x in lines))
    run = tmp_path / "run" / "investigation.json"
    run.write_text(
        json.dumps(
            {
                "kind": "investigation",
                "label": "old format",
                "world_id": "w99",
                "agent": {"backend": "codex", "model": "fixture-model"},
                "analysis": str(analysis),
                "sessions": [{"number": 1}, {"number": 2}],
                "costs": {
                    "sessions": 2,
                    "model_starts": 2,
                    "tokens": {"input_tokens": 3},
                    "queues": [
                        {
                            "state": "done",
                            "simulated_jobs": 30,
                            "simulated_run_jobs": 24,
                            "partial_jobs": 1,
                            "timing": {"simulation_seconds": 60},
                        }
                    ],
                },
            }
        )
    )
    view = load_investigation(run)
    view.annotations = {
        "by": "a reviewer",
        "labels": [{"section": "ranked", "rank": 3, "label": "padding"}],
    }
    (rnd,) = view.rounds
    assert rnd.decision is None and rnd.fix_set is not None
    (table,) = derive.decision_tables(view, rnd)
    assert [r[4] for r in table.rows] == ["", "", "padding", ""]
    column = derive._cost_column(view, rnd)
    assert column["task chain-check replays"] == "6"
    assert column["failed task simulations"] == "1"
    out = build(view, tmp_path / "bundle", generated="test", clips=False)
    page = (out / "round-1.md").read_text()
    assert "This older format has no adoption" in page
    assert "ranked 1 of 4 (no adoption in this format)" in page


def test_origin_comes_from_the_fixture_marker(tmp_path: Path) -> None:
    sources = Sources()
    real = tmp_path / "real" / "a.json"
    fake = tmp_path / "fake" / "deep" / "b.json"
    for path in (real, fake):
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    (tmp_path / "fake" / FIXTURE_MARKER).write_text("x")
    assert sources.origin_of(real) == "real"
    assert sources.origin_of(fake) == "fixture"
    assert sources.origin_of(real, "toy") == "toy"
    sources.json(real, "r")
    sources.json(fake, "f")
    assert sources.origin([str(real.resolve())]) == "real"
    assert sources.origin([str(real.resolve()), str(fake.resolve())]) == "fixture"


def test_truth_is_refused_for_sealed_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        load_truth(Sources(), tmp_path / "robots-unseen-x", "w99")


def test_rate_figures_render(tmp_path: Path) -> None:
    from fleetlab.loop.exhibit import figures

    figures.rate_dots(
        tmp_path / "a.png", [("r01", Rate(10, 7)), ("r02", Rate(0, 0))], "t", "x", "toy"
    )
    figures.pairs_figure(tmp_path / "b.png", [("a to b", 3, 1)], "t", "real")
    figures.series_by_group(
        tmp_path / "c.png", {"s1": [("day 0", Rate(4, 2))]}, "t", "x", "fixture"
    )
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.png", "b.png", "c.png"))


def test_a_bundle_directory_that_is_not_one_is_not_cleared(
    tmp_path: Path, fixture_run: Path
) -> None:
    other = tmp_path / "precious"
    other.mkdir()
    (other / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        build(load_run(fixture_run), other, clips=False)
    assert (other / "keep.txt").is_file()
