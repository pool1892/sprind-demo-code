"""Round 8: what the review of round 7 found open before the test worlds are drawn.

The private recipe is produced by code and every field of it is pinned (finding F1);
admission is the interval of entry 20 alone, and the onset days of the interaction's
members and of the plain faults are round 6's again (F3, F2); the single-place breaks
that the tests of round 7 let through (F4); the two damaged sources the import passed
(F5). Synthetic tests isolate a guard and say nothing about the simulator.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from fleetlab.worlds import calibration, compose, curves, estimate
from fleetlab.worlds.recipe import (
    TEST_WORLD_JOBS,
    TEST_WORLD_SOURCES,
    WorldRecipe,
    external_recipe,
    planned_recipe,
    private_recipe,
    stated_fields,
)

SEEDS = compose.ASSESSMENT_SEEDS
CALIBRATION = Path.home() / ".cache/robots-probes/worlds-v4"
WORLDS = Path.home() / ".cache/robots-probes/worlds"

# ---- 1. the private recipe ---------------------------------------------------------------

# Every field a test world's recipe file states, and its value. The three identity fields
# are the function's arguments; the seed is absent, creation draws it. A change to any
# setting of the template shows here before a world is drawn from it.
PRIVATE_RECIPE: dict[str, Any] = {
    "name": "T1",
    "world_id": "w51",
    "public_fleet_id": "w07",
    "legacy_identity": False,
    "kind": "test",
    "size": "small",
    "sites": 3,
    "robots_per_site": 4,
    "days": 10,
    "episodes_per_robot_day": 14,
    "windows": {
        "observed": {"first": 0, "last": 5},
        "experiment": {"first": 6, "last": 7},
        "scoring": {"first": 8, "last": 9},
    },
    "tasks": [["libero_spatial", i] for i in range(10)] + [["libero_object", i] for i in range(10)],
    "site_suites": [["s1", "libero_spatial"], ["s3", "libero_spatial"]],
    "interaction_single_loss_points": 10.0,
    "band": [0.6, 0.75],
    "aim": [0.66, 0.695],
    "scoring_floor": 0.5,
    "robot_success_floor": 0.3,
    "decoy_margin": 0.02,
    "decoy_margin_of": "fleet",
    "retained_error": "generation-4",
    "calibration_jobs": [
        "20261010-task-cells-r6",
        "20261010-task-cells-r6c",
        "20261010-task-cells-r6d",
        "20261011-task-cells-r7a",
        "20261011-task-cells-r7b",
        "20261011-task-cells-r7c",
    ],
    "search_revision": "r8",
    "admission": "interval",
    "warnable_drift": True,
    "estimator": "measured-v3",
    "calibration_sources": [
        "fleetlab/faults/curves.json sha256 6c0402890bbe635c",
        "fleetlab/worlds/curves_supplement.json sha256 0fec9a3e17435b69",
        "task_refinement sha256 4d970aba2f5a8550",
    ],
    "sealed_hash": None,
    "total_failures": 1,
    "min_clean_robots": 0,
    "repeats": 3,
    "worn_robots": [1, 2],
    "settle_levels_first": False,
    "pairs_by_loss": False,
    "clear_interaction_site": False,
    "members_alone_confirm": False,
    "provisional": False,
    "plain": 4,
    "small": 0,
    "confounders": 1,
    "confounder_companion": "calibration_record",
    "decoys": 1,
    "interactions": 1,
    "interaction_scope": "robot",
    "additive_pairs": 1,
    "fleet_fault": True,
    "fleet_image_blur": 3,
    "growing_drift": False,
    "extension_types": True,
    "drifts": 1,
    "config_share": None,
    "strong": False,
    "min_detectable": 6,
    "margin": 0,
    "attempts": 400,
    "note": "test worlds' template: the planned difficulty on measured contexts, "
    "entry 20's settings, six named calibration jobs; no unseen fault types",
}
# What the function adds to the template; every other field is the template's own.
ADDED = ("name", "kind", "public_fleet_id", "calibration_jobs", "calibration_sources", "note")


def test_private_recipe_states_every_field_and_each_is_pinned() -> None:
    produced = private_recipe("w51", "w07", name="T1")
    assert set(produced) == set(WorldRecipe.model_fields) - {"seed"}
    assert produced == PRIVATE_RECIPE
    assert json.loads(json.dumps(produced)) == produced
    # The template itself, field by field: what the assessment composes.
    template = stated_fields(planned_recipe("planned", world_id="w51", seed=SEEDS[0]))
    assert template == {
        **PRIVATE_RECIPE,
        "name": "planned",
        "kind": "development",
        "public_fleet_id": "w01",
        "calibration_jobs": None,
        "calibration_sources": None,
        "note": "unsealed candidate; measured contexts, robot-window floor and "
        "scoring-removal resampling; no guarantee of feasibility",
    }
    assert {key for key in produced if produced[key] != template[key]} == set(ADDED)
    # Another place differs by its identity alone.
    other = private_recipe("w52", "w08", name="T2")
    assert {key for key in produced if produced[key] != other[key]} == {
        "name",
        "world_id",
        "public_fleet_id",
    }


def test_private_recipe_stands_on_the_six_jobs_cells_and_composes_as_the_template() -> None:
    assert curves.measured_sources(TEST_WORLD_JOBS) == TEST_WORLD_SOURCES
    assert len(TEST_WORLD_JOBS) == len(set(TEST_WORLD_JOBS)) == 6
    produced = private_recipe("w99", "w07", name="T1")
    recipe = WorldRecipe.model_validate({**produced, "seed": SEEDS[6]})
    named = compose.Composer(recipe)
    assert named.cells() == curves.measured_cells_of(TEST_WORLD_JOBS)
    # Fewer jobs are other cells under the same pinned digest: refused.
    for jobs in (TEST_WORLD_JOBS[:-1], None):
        with pytest.raises(ValueError, match="sealed recipe calibration changed"):
            compose.Composer(recipe.model_copy(update={"calibration_jobs": jobs}))
    # On a public seed the produced recipe accepts the template's draft with its seal.
    template = compose.Composer(planned_recipe("planned", world_id="w99", seed=SEEDS[6]))
    seals = []
    for composer in (named, template):
        draft = composer.draft(46)
        problems, expected = composer.check(draft, composer.settle(draft, 46))
        assert problems == []
        seals.append(expected["forecast_seal"])
    assert seals[0] == seals[1]


def test_external_recipe_refuses_a_file_that_leaves_a_field_out(tmp_path: Path) -> None:
    produced = private_recipe("w51", "w07", name="T1")
    path = tmp_path / "private-recipe.json"
    path.write_text(json.dumps(produced))
    loaded = external_recipe(path)
    assert stated_fields(loaded) == produced and loaded.seed >= 1 << 62
    for field in produced:
        path.write_text(json.dumps({k: v for k, v in produced.items() if k != field}))
        with pytest.raises(ValueError, match=r"test or reserve|state every field"):
            external_recipe(path)
    # Stated, but not fixing its cells: the record "as it is" moves with every import.
    path.write_text(json.dumps({**produced, "calibration_jobs": None}))
    with pytest.raises(ValueError, match="name the calibration jobs"):
        external_recipe(path)
    # Why it matters: left out, these would have been the older rules (the review
    # composed three other worlds that way), and the model still accepts the omission.
    for field in ("retained_error", "search_revision", "aim", "admission"):
        assert WorldRecipe.model_fields[field].default != getattr(loaded, field)


# ---- 2. the two decisions ----------------------------------------------------------------


def test_admission_is_the_interval_and_the_target_refuses_no_draft() -> None:
    """Seed 6, draft 46: forecast 70.6%, above the target, interval 64.6 to 74.7%."""
    recipe = planned_recipe("planned", world_id="w99", seed=SEEDS[6])
    assert recipe.admission == "interval" and recipe.retained_error == "generation-4"
    composer = compose.Composer(recipe)
    draft = composer.draft(46)
    schedule = composer.settle(draft, 46)
    problems, expected = composer.check(draft, schedule)
    assert problems == []
    low, high = expected["observed_forecast"]["logged"]["prediction_interval"]
    assert recipe.band[0] <= low and high <= recipe.band[1]
    assert expected["observed_logged_success"] > recipe.aim[1]
    assert expected["observed_forecast"]["model_error_points"] == [-2.1, 2.1]

    def refusals(**changes: Any) -> list[str]:
        other = compose.Composer(recipe.model_copy(update=changes))
        return other.check(draft, schedule)[0]

    # The rule every world sealed so far was composed under still refuses it there.
    assert any("expected observed success" in p for p in refusals(admission="target"))
    # No target refuses: one far from the forecast changes nothing.
    assert refusals(aim=(0.61, 0.62)) == []
    # The interval does, at either end: one point less room in the band.
    for band in ((round(low + 0.01, 4), 0.75), (0.60, round(high - 0.01, 4))):
        narrow = refusals(band=band)
        assert len(narrow) == 1 and "observed prediction interval" in narrow[0]
        # And without the 2.1 points the same draft has that room again.
        assert refusals(retained_error="none", band=band) == []
    # The ends of the band themselves are inside it.
    assert refusals(band=(low, high)) == []


def test_sealed_worlds_keep_the_rules_they_were_composed_under() -> None:
    from fleetlab.worlds.recipe import DEVELOPMENT

    for key in ("w48", "w49"):
        sealed = DEVELOPMENT[key]
        assert (sealed.admission, sealed.search_revision) == ("target", "r6")
        # Their stored records do not carry the new field and must still compare equal.
        assert "admission" not in sealed.model_dump(mode="json")
    for name in ("D2", "D3", "planned"):
        candidate = planned_recipe(name, world_id="w99", seed=SEEDS[0])
        assert (candidate.admission, candidate.search_revision) == ("interval", "r8")
    with pytest.raises(ValueError, match="measured-v3 settings"):
        WorldRecipe(
            name="x", world_id="w99", seed=SEEDS[0], estimator="task-v2", admission="interval"
        )


def test_assessment_counts_a_draft_above_the_target() -> None:
    result = compose.assess(("planned",), seeds=[SEEDS[6]], attempts=47)["planned"]
    row = result["rows"][0]
    assert result["accepted"] == 1 and row["attempt"] == 46
    assert row["observed"] > 0.695 and 0.60 <= row["interval"][0] < row["interval"][1] <= 0.75
    refused = compose.assess(("planned",), seeds=[SEEDS[6]], attempts=46)["planned"]
    assert refused["accepted"] == 0
    assert not any(
        "expected observed success" in key for key in refused["refused_attempts_by_rule"]
    )
    assert any("prediction interval" in key for key in refused["refused_attempts_by_rule"])


def _bare(**changes: Any) -> WorldRecipe:
    return planned_recipe("D2", world_id="w99", seed=SEEDS[0]).model_copy(
        update={
            "fleet_fault": False,
            "decoys": 0,
            "additive_pairs": 0,
            "drifts": 0,
            "confounders": 0,
            "plain": 0,
            "small": 0,
            **changes,
        }
    )


PAIR = (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"))


def test_interaction_members_keep_round6_days(monkeypatch: pytest.MonkeyPatch) -> None:
    assert planned_recipe("planned", world_id="w99", seed=SEEDS[0]).search_revision == "r8"
    c = compose.Composer(_bare())
    monkeypatch.setattr(c, "_joint_pairs", lambda: [PAIR])
    monkeypatch.setattr(c, "pair_scope_choices", lambda *args: ["r01"])
    tried = []

    def from_day_two(schedule: list[Any], *args: Any) -> bool:
        tried.append(schedule[-1].schedule.onset_day)
        return bool(tried[-1] == 2)

    # Revision r7 moved the second member to day 2 here; r8 passes the carrier over.
    monkeypatch.setattr(c, "supported", from_day_two)
    draft = c.draft(0)
    assert draft.stopped == "no interaction carrier left on measured contexts above the floor"
    assert tried == [1]
    monkeypatch.setattr(c, "supported", lambda *args, **kw: True)
    draft = c.draft(0)
    assert [f.schedule.onset_day for f in draft.fixed] == [0, 1]


def test_interaction_passes_over_a_carrier_whose_day_one_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = compose.Composer(_bare(decoys=1))
    monkeypatch.setattr(c, "_joint_pairs", lambda: [PAIR])
    monkeypatch.setattr(c, "pair_scope_choices", lambda *args: ["r01", "r05"])
    monkeypatch.setattr(c, "supported", lambda *args, **kw: True)
    backdrop = ("backdrop_change", "1 walls and floor")
    monkeypatch.setattr(
        c, "decoy_choices", lambda site, onset: [backdrop] if (site, onset) == ("s1", 1) else []
    )
    for attempt in range(8):
        draft = c.draft(attempt)
        decoy, first, second = draft.fixed
        assert (decoy.target.scope, decoy.schedule.onset_day) == ("s1", 1)
        # r01 is at s1, where the decoy starts on day 1: the members go to r05, on 0 and 1.
        assert first.target.scope == second.target.scope == "r05"
        assert (first.schedule.onset_day, second.schedule.onset_day) == (0, 1)
        assert not c.co_onsets(c.schedule(draft), draft.patterns)


def _plain_composer(revision: str, monkeypatch: pytest.MonkeyPatch) -> compose.Composer:
    from test_worlds_task_guards import task_recipe

    c = compose.Composer(
        task_recipe(
            estimator="measured-v3",
            search_revision=revision,
            robot_success_floor=0.3,
            aim=(0.66, 0.695),
        )
    )
    c.plain_menu = [("action_latency", "robot")]
    c.effect_levels["action_latency"] = ("8 steps", "17 steps", "20 steps")
    c._used, c._totals, c._heavy, c._clear_sites = {}, 0, 0, set()

    # The middle dose is nearest the target on every day, most of all on day 2; the
    # strongest is below the floor.
    def forecast(schedule: list[Any], *args: Any) -> float:
        fault = schedule[-1]
        day = fault.schedule.onset_day
        return float({8: 0.9, 17: 0.6775 + 0.01 * abs(day - 2), 20: 0.1}[fault.level])

    monkeypatch.setattr(c, "observed_logged", forecast)
    monkeypatch.setattr(c, "supported", lambda schedule, *args, **kw: forecast(schedule) >= 0.33)
    monkeypatch.setattr(c, "stands_alone", lambda *args: True)
    return c


def _draw(c: compose.Composer, stream: int, draft: compose.Draft) -> compose.Plain | None:
    from fleetlab.worlds import base

    c._used, c._totals, c._heavy = {}, 0, 0
    return c._draw_plain(
        base.rng(c.recipe.seed, "round-8-days", stream),
        compose.Occupancy(c.robots),
        {robot for robot, _ in c.robots if robot != "r01"},
        "any",
        "f09",
        lambda: "4.3.0",
        draft=draft,
    )


def test_plain_fault_keeps_the_day_round6_draws_and_ranks_only_its_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days: dict[str, list[int]] = {}
    for revision in ("r6", "r7", "r8"):
        c = _plain_composer(revision, monkeypatch)
        drawn = [_draw(c, stream, compose.Draft()) for stream in range(12)]
        assert all(p is not None and p.scope == "r01" for p in drawn)
        days[revision] = [p.onset for p in drawn if p is not None]
        if revision != "r6":  # the level nearest the target, whatever was drawn
            assert {p.labels[p.level] for p in drawn if p is not None} == {"17 steps"}
            assert {p.labels for p in drawn if p is not None} == {
                ("8 steps", "17 steps", "20 steps")
            }
    # Round 6 draws a day between 0 and 4; r8 keeps exactly that day, stream by stream.
    assert days["r8"] == days["r6"]
    assert len(set(days["r6"])) >= 4 and set(days["r6"]) <= set(range(5))
    # Revision r7 ranked the day by the forecast as well and took day 2 every time.
    assert set(days["r7"]) == {2}


def test_plain_fault_takes_a_free_day_when_its_drawn_day_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _plain_composer("r8", monkeypatch)
    batch = ("object_batch_variant", "4 mass doublings")
    taken = compose.Draft(
        fixed=[compose.make_fault(f"f0{day}", *batch, "s1", day) for day in (1, 2, 4)]
    )
    free = [p.onset for stream in range(12) if (p := _draw(c, stream, taken)) is not None]
    drawn = [p.onset for stream in range(12) if (p := _draw(c, stream, compose.Draft()))]
    assert len(free) == len(drawn) == 12
    # A drawn day that is free stands; one on which the site's faults start is replaced.
    assert set(free) == {0, 3}
    assert all(a == b for a, b in zip(free, drawn, strict=True) if b in (0, 3))
    assert any(b in (1, 2, 4) for b in drawn)


def _settled(
    revision: str,
    draft: compose.Draft,
    forecast: Any,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool = True,
) -> list[Any]:
    from test_worlds_task_guards import task_recipe

    c = compose.Composer(
        task_recipe(
            estimator="measured-v3",
            search_revision=revision,
            aim=(0.66, 0.69),
            robot_success_floor=0.3,
        )
    )
    monkeypatch.setattr(c, "observed_logged", lambda schedule, ctx: forecast(schedule))
    monkeypatch.setattr(c, "supported", lambda *args, **kw: supported)
    return c.settle_measured(draft, 0)


def _one_plain(role: str, onset: int, first: int, last: int) -> compose.Draft:
    plain = compose.Plain(
        "action_latency", "r01", ("8 steps", "17 steps"), 1, onset, last, "f01", None, first
    )
    return compose.Draft(plains=[plain], roles={"f01": role})


def test_settling_moves_a_plain_fault_one_day_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # Day 0 is on target, day 2 is nearer than day 3, and day 1 lies between them, worse.
    by_day = {0: 0.675, 1: 0.90, 2: 0.70, 3: 0.80, 4: 0.82}

    def forecast(schedule: list[Any]) -> float:
        return by_day[schedule[0].schedule.onset_day] if schedule[0].level == 17 else 0.95

    for revision, day in (("r7", 0), ("r8", 2)):
        result = _settled(revision, _one_plain("plain", 3, 0, 4), forecast, monkeypatch)
        assert (result[0].schedule.onset_day, result[0].level) == (day, 17), revision


def test_settling_keeps_the_confounder_and_the_interaction_on_their_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The confounder: day 1 would be on target, and so would the stronger level.
    def by_day(schedule: list[Any]) -> float:
        return {1: 0.675, 2: 0.80, 3: 0.90}[schedule[0].schedule.onset_day]

    for revision, day in (("r7", 1), ("r8", 2)):
        result = _settled(revision, _one_plain("confounder", 2, 1, 3), by_day, monkeypatch)
        assert result[0].schedule.onset_day == day, revision

    def by_level(schedule: list[Any]) -> float:
        return 0.675 if schedule[0].level == 8 else 0.80

    result = _settled("r8", _one_plain("confounder", 2, 2, 2), by_level, monkeypatch)
    assert (result[0].level, result[0].schedule.onset_day) == (8, 2)

    # An interaction member: a later day would be on target.
    def later(schedule: list[Any]) -> float:
        return float(0.60 + 0.04 * schedule[0].schedule.onset_day)

    for revision, day in (("r7", 2), ("r8", 1)):
        member = compose.make_fault("f01", *PAIR[1], "r01", 1)
        draft = compose.Draft(fixed=[member], roles={"f01": "interaction"})
        assert _settled(revision, draft, later, monkeypatch)[0].schedule.onset_day == day
    # The batch's day still moves in r8, as in r7 (tests/test_worlds_round7.py).
    batch = compose.make_fault("f01", "object_batch_variant", "4 mass doublings", "s1", 1)
    draft = compose.Draft(fixed=[batch], roles={"f01": "additive_pair"})
    assert _settled("r8", draft, later, monkeypatch)[0].schedule.onset_day == 2


def test_template_drafts_carry_the_interaction_on_days_zero_and_one() -> None:
    """Real cells: every complete draft of the template, after settling."""
    recipe = planned_recipe("planned", world_id="w99", seed=SEEDS[1])
    c = compose.Composer(recipe)
    complete = 0
    for attempt in range(30, 42):
        draft = c.draft(attempt)
        if draft.stopped or not draft.patterns.interactions:
            continue
        confounder = {
            p.fault_id: p.onset for p in draft.plains if draft.roles[p.fault_id] != "plain"
        }
        days = {f.fault_id: f.schedule.onset_day for f in c.settle(draft, attempt)}
        first, second = draft.patterns.interactions[0]
        assert (days[first], days[second]) == (0, 1)
        assert all(days[fid] == day and 1 <= day <= 3 for fid, day in confounder.items())
        complete += 1
    assert complete >= 6


# ---- 3. the breaks the tests of round 7 let through --------------------------------------


@pytest.mark.parametrize("revision", ["r7", "r8"])
def test_settling_refuses_a_move_onto_a_taken_day(
    revision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_day = {0: 0.99, 1: 0.95, 2: 0.675, 3: 0.80, 4: 0.90}

    def forecast(schedule: list[Any]) -> float:
        return by_day[next(f for f in schedule if f.fault_id == "f02").schedule.onset_day]

    def draft(blocked: bool) -> compose.Draft:
        # A fault at the robot's site that starts on day 2; settling cannot move it.
        blocker = compose.make_fault(
            "f01", "object_batch_variant", "4 mass doublings", "s1" if blocked else "s2", 2
        )
        plain = compose.Plain("action_latency", "r01", ("17 steps",), 0, 3, 4, "f02")
        return compose.Draft(
            fixed=[blocker], plains=[plain], roles={"f01": "small", "f02": "plain"}
        )

    def day(result: list[Any]) -> int:
        return int(next(f for f in result if f.fault_id == "f02").schedule.onset_day)

    assert day(_settled(revision, draft(False), forecast, monkeypatch)) == 2
    assert day(_settled(revision, draft(True), forecast, monkeypatch)) == 3


@pytest.mark.parametrize("revision", ["r7", "r8"])
def test_settling_refuses_a_move_off_measured_ground(
    revision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forecast(schedule: list[Any]) -> float:
        return 0.675 if schedule[0].level == 8 else 0.80

    moved = _settled(revision, _one_plain("plain", 2, 0, 4), forecast, monkeypatch)
    assert moved[0].level == 8
    # The same move, where the floor, a cell or a decoy check fails: it is not made.
    kept = _settled(revision, _one_plain("plain", 2, 0, 4), forecast, monkeypatch, supported=False)
    assert (kept[0].level, kept[0].schedule.onset_day) == (17, 2)


def test_interaction_carrier_is_judged_on_its_overlap_not_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = planned_recipe("D3", world_id="w99", seed=SEEDS[1])
    # Alone, every carrier would lie below the floor.
    monkeypatch.setattr(
        estimate, "robot_window_success", lambda *args, **kw: [{"true": 0.0, "logged": 0.0}]
    )
    for revision, kept in (("r6", False), ("r7", True), ("r8", True)):
        c = compose.Composer(recipe.model_copy(update={"search_revision": revision}))
        carriers = [c.task_pair_robot_choices(pair) for pair in c._joint_pairs()]
        assert any(carriers) is kept, revision


def test_recompose_record_refuses_a_stored_seal_that_does_not_recompose() -> None:
    from test_worlds_generation4 import TEXTS, sealed

    from fleetlab.contracts.identity import Task
    from fleetlab.worlds import world

    w, seal = sealed("w49")
    record = json.loads(json.dumps(world.generator_record(w)))
    assert record["expected"]["forecast_seal"] == seal["forecast_seal"]
    assert record["recipe"]["sealed_hash"] == seal["hash"]
    texts = {Task(r["suite"], r["task_index"]): r["text"] for r in TEXTS}
    record["expected"]["forecast_seal"] = "0" * 64
    with pytest.raises(ValueError, match="stored forecast seal does not recompose"):
        world.recompose_record(record, texts)


# ---- 4. the import and damaged sources ---------------------------------------------------


def _copy_run(source: Path, target: Path, drop: str | None = None) -> None:
    """A run's record and three tables; `drop` removes one attempt from all of them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.log.runfiles import table_path

    target.mkdir(parents=True)
    shutil.copyfile(source / "run.json", target / "run.json")
    for name in ("identities", "verdicts", "episodes"):
        data = pq.read_table(table_path(source, name))
        if drop is not None:
            ids = (
                [
                    json.loads(text)["episode_id"]
                    for text in data.column("identity_json").to_pylist()
                ]
                if name == "identities"
                else data.column("episode_id").to_pylist()
            )
            assert drop in ids
            data = data.filter(pa.array([episode != drop for episode in ids]))
        destination = table_path(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data, destination)


def _attempts(record: dict[str, Any]) -> int:
    return sum(int(cell["n"]) for cell in record["cells"])


def _folded(root: Path, drop: str | None = None) -> Path:
    source = WORLDS / "w47-opaque-v1"
    if not (source / "files/manifest.jsonl").is_file():
        pytest.skip("the folded world is not installed on this host")
    (root / "files").mkdir(parents=True)
    for name in ("spec.json", "model.json", "description.json", "manifest.jsonl", "hash"):
        shutil.copyfile(source / "files" / name, root / "files" / name)
    for window in ("observed", "later"):
        run = f"w47-{window}"
        _copy_run(source / "runs" / run, root / "runs" / run, drop if window == "later" else None)
    return root


def test_folded_world_with_an_attempt_removed_everywhere_is_not_imported(tmp_path: Path) -> None:
    whole = _folded(tmp_path / "whole")
    assert _attempts(calibration.refinement_record([whole])) == 1680
    manifest = (whole / "files/manifest.jsonl").read_text().splitlines()
    last = json.loads(manifest[-1])["identity"]["episode_id"]
    # Identity, verdict and outcome gone alike: the run's tables agree with one another.
    short = _folded(tmp_path / "short", drop=last)
    with pytest.raises(ValueError, match="folded world w47 differs from its own manifest"):
        calibration.refinement_record([short])


STAGES = ("pair-batch-2", "pair-batch-2-refine")


def _job(root: Path, selected: Any = None, drop: str | None = None) -> Path:
    """Job 2 of round 7, one stage of 100 and its continuation of 40, as a scratch copy."""
    source = CALIBRATION / "20261011-task-cells-r7b"
    if not (source / "plan.json").is_file():
        pytest.skip("archived round-7 manifests are not installed on this host")
    plan = json.loads((source / "plan.json").read_text())
    root.mkdir(parents=True)
    (root / "plan.json").write_text(
        json.dumps(
            {"order": list(STAGES), "stages": {s: plan["stages"][s] for s in STAGES}, "assays": []}
        )
    )
    for stage in STAGES:
        shutil.copyfile(source / f"{stage}.jsonl", root / f"{stage}.jsonl")
        refine = "parent" in plan["stages"][stage]
        _copy_run(source / stage / "run", root / stage / "run", drop if refine else None)
    text = (source / f"{STAGES[1]}.selected.jsonl").read_text()
    (root / f"{STAGES[1]}.selected.jsonl").write_text(selected(text) if selected else text)
    return root


def test_continuation_one_attempt_short_of_the_stop_rule_is_not_imported(tmp_path: Path) -> None:
    whole = calibration.refinement_record([], _job(tmp_path / "whole"))
    assert _attempts(whole) == 140 and whole["skipped_stages"] == []
    source = CALIBRATION / "20261011-task-cells-r7b"
    lines = (source / f"{STAGES[1]}.selected.jsonl").read_text().splitlines()
    assert len(lines) == 40
    last = json.loads(lines[-1])["identity"]["episode_id"]
    # The selected list and the run hold 39 of the 40 attempts the rule selects; every
    # one of the 39 is a sealed identity and the run is complete against the list.
    short = _job(tmp_path / "short", lambda text: "\n".join(lines[:-1]) + "\n", drop=last)
    with pytest.raises(ValueError, match=r"differs from the stop rule: pair-batch-2-refine"):
        calibration.refinement_record([], short)


def test_continuation_with_an_identity_outside_the_sealed_manifest_is_not_imported(
    tmp_path: Path,
) -> None:
    def foreign(text: str) -> str:
        lines = text.splitlines()
        row = json.loads(lines[0])
        row["identity"]["scene_seed"] += 1
        lines[0] = json.dumps(row, separators=(",", ":"))
        return "\n".join(lines) + "\n"

    root = _job(tmp_path / "foreign", foreign)
    with pytest.raises(ValueError, match="selected manifest differs from sealed identities"):
        calibration.refinement_record([], root)
