"""Round 7 contracts and preregistered calibration evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from fleetlab.contracts.rollout import Job
from fleetlab.worlds import calibration, supplement


def test_round7_wear_plan_seals_six_contexts_and_fresh_continuations() -> None:
    root = Path.home() / ".cache/robots-probes/worlds-v4/20261011-task-cells-r7a"
    if not (root / "plan.json").is_file():
        pytest.skip("archived round-7 manifests are not installed on this host")
    sealed = json.loads(Path(__file__).with_name("worlds_r7a_plan.json").read_text())
    assert json.loads((root / "plan.json").read_text()) == sealed
    assert sealed["budget_seconds"] == 10800
    assert len(sealed["order"]) == 12
    expected = {
        tuple(sorted((("gripper_wear", f"{wear} wear"), calibration._M4, *cover)))
        for wear in ("0.25", "0.75", "1")
        for cover in ((), (calibration._BLUR3,))
    }
    seen_contexts = set()
    for name, suite, kind, members in calibration.ROUND7_WEAR_STAGES:
        assert suite == "libero_spatial" and kind == "contexts"
        seen_contexts.add(tuple(sorted(members)))
        signatures: list[tuple[Any, ...]] = []
        for stage, per_task in ((name, 10), (f"{name}-refine", 40)):
            payload = (root / f"{stage}.jsonl").read_bytes()
            text = payload.decode()
            assert hashlib.sha256(payload).hexdigest() == sealed["stages"][stage]["sha256"]
            identities = [Job.model_validate_json(line).identity for line in text.splitlines()]
            assert Counter(i.task_index for i in identities) == dict.fromkeys(range(10), per_task)
            assert {calibration.mechanism_members(i.faults) for i in identities} == {
                tuple(sorted(members))
            }
            signatures.extend(supplement._task_signature(i) for i in identities)
        assert len(signatures) == len(set(signatures)) == 500
    assert seen_contexts == expected


def test_decoy_uses_scoring_tasks_in_fast_and_final_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    from fleetlab.contracts.identity import Task
    from fleetlab.contracts.world import Patterns
    from fleetlab.worlds import compose, curves
    from fleetlab.worlds.recipe import planned_recipe

    c = compose.Composer(planned_recipe("D3", world_id="w99", seed=compose.ASSESSMENT_SEEDS[0]))
    c.slots = tuple(replace(s, task=Task(curves.SPATIAL, 0 if s.day < 6 else 1)) for s in c.slots)
    f = compose.make_fault("f99", "backdrop_change", "1 walls and floor", "s1", 0)
    assay: dict[str, Any] = {
        "suite": curves.SPATIAL,
        "members": [[f.kind, "1 walls and floor"]],
        "tasks": [
            {"task_index": 0, "n": 10000, "paired_counts": [5000, 800, 0, 4200]},
            {"task_index": 1, "n": 10000, "paired_counts": [5000, 0, 1200, 3800]},
        ],
    }
    monkeypatch.setattr(calibration, "decoy_assays", lambda *args: [assay])
    assert calibration.decoy_mix_interval(assay["tasks"], {0: 6, 1: 4}, 0.06)["flat"]
    assert not c.decoys_flat([f], [f.fault_id])
    assert not calibration.context_decoy_confirmed(f, [f], list(c.slots), c.decoy_margin(f))
    problems, _ = c.check(compose.Draft(fixed=[f], patterns=Patterns(decoys=[f.fault_id])), [f])
    assert "decoy lacks replay-confirmed flatness with its covering faults" in problems

    # The explanatory bound must not certify the same decoy on pooled days either.
    # Its earliest onset is day 1: five helpful days cancel four harmful ones
    # only when pooled. Isolate it with consistent cells and no other patterns.
    c.recipe = c.recipe.model_copy(update={"interactions": 0, "additive_pairs": 0, "drifts": 0})
    assay["tasks"][1]["paired_counts"] = [5000, 0, 1000, 4000]
    table = {
        (curves.SPATIAL, task, members): curves.MeasuredCell(10000, successes, successes, ())
        for task, healthy, changed in ((0, 5000, 5800), (1, 6000, 5000))
        for members, successes in (((), healthy), (((f.kind, "1 walls and floor"),), changed))
    }
    monkeypatch.setattr(c, "cells", lambda: table)
    monkeypatch.setattr(compose, "Composer", lambda *args: c)
    monkeypatch.setattr(
        calibration, "decoy_assays", lambda kind, *args: [assay] if kind == f.kind else []
    )
    assert compose.forecast_bound(c.recipe)["bound"] is None


def test_world_context_and_record_keep_named_cells_and_error() -> None:
    from test_worlds_generation4 import sealed

    from fleetlab.worlds import curves, world
    from fleetlab.worlds.recipe import WorldRecipe

    for key in ("w48", "w49"):
        w, seal = sealed(key)
        ctx = world.context(w.composition, w.calibration)
        assert ctx.measured == curves.measured_cells_of(w.recipe.calibration_jobs)
        assert ctx.retained_error == w.recipe.retained_error == "none"
        record = world.generator_record(w)
        assert record["curves"] == list(curves.measured_sources(w.recipe.calibration_jobs))
        for name, row in seal["forecast_logged"].items():
            assert record["predictions"][name]["logged"]["prediction_interval"] == row["interval"]
        again = world.generate(
            WorldRecipe.model_validate(record["recipe"]), w.software, w.task_texts
        )
        assert world.world_hash(again) == seal["hash"]
        assert again.composition.expected["forecast_seal"] == seal["forecast_seal"]


@pytest.mark.parametrize(
    "chain,outcomes,message",
    [(True, {}, "incomplete"), (False, {"e1": object()}, "chain")],
)
def test_import_rejects_a_missing_outcome_and_a_failed_chain(
    monkeypatch: pytest.MonkeyPatch, chain: bool, outcomes: dict[str, Any], message: str
) -> None:
    from types import SimpleNamespace

    from fleetlab import dose_response
    from fleetlab.log import runfiles

    monkeypatch.setattr(
        runfiles, "read_identities", lambda path: [SimpleNamespace(episode_id="e1")]
    )
    monkeypatch.setattr(
        dose_response, "read_run", lambda path: SimpleNamespace(chain_ok=chain, outcomes=outcomes)
    )
    supplement._task_rows.cache_clear()
    with pytest.raises(ValueError, match=message):
        supplement._task_rows("/synthetic-damaged-run")
    supplement._task_rows.cache_clear()


@pytest.mark.parametrize("damage", ["truncate", "reorder", "line_endings", "outcomes"])
def test_import_rejects_changed_stage_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    source = Path.home() / ".cache/robots-probes/worlds-v4/20261010-task-cells-r6c"
    if not (source / "plan.json").is_file():
        pytest.skip("archived refinement manifests are not installed on this host")
    plan = json.loads((source / "plan.json").read_text())
    name = "topups-1"
    root = tmp_path / "damaged-stage"
    root.mkdir()
    (root / "plan.json").write_text(
        json.dumps({"order": [name], "stages": {name: plan["stages"][name]}, "assays": []})
    )
    lines = (source / f"{name}.jsonl").read_text().splitlines()
    changed = lines[:-1] if damage == "truncate" else lines[::-1] if damage == "reorder" else lines
    ending = "\r\n" if damage == "line_endings" else "\n"
    (root / f"{name}.jsonl").write_bytes((ending.join(changed) + ending).encode())
    last = Job.model_validate_json(lines[-1]).identity.episode_id
    rows = [
        row
        for row in supplement._task_rows(str(source / name / "run"))
        if damage not in ("truncate", "outcomes") or row[0].episode_id != last
    ]
    monkeypatch.setattr(supplement, "_task_rows", lambda p: rows if str(root) in p else [])
    message = (
        r"incomplete.*stage"
        if damage == "outcomes"
        else r"manifest.*sealed|manifest.*preregistered"
    )
    with pytest.raises(ValueError, match=message):
        calibration.refinement_record([], root)


def test_check_names_either_window_count_as_forecast(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_worlds_generation4 import sealed

    from fleetlab.worlds import __main__ as cli
    from fleetlab.worlds import world

    w, _ = sealed("w49")
    assert len(w.detect.observed_counted(False)) == 5
    monkeypatch.setattr(world, "generate", lambda *args: w)
    assert cli.main(["check", "w49", "--fake-software"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["cause_forecast"]["count"] == 7
    assert report["cause_forecast"]["windows"] == ["observed", "scoring"]
    assert report["cause_forecast"]["measured"] is False


def test_template_names_generation4_error_and_keeps_development_settings() -> None:
    from fleetlab.worlds import estimate
    from fleetlab.worlds.recipe import planned_recipe

    template = planned_recipe("planned", world_id="w99", seed=(1 << 62) + 1)
    assert template.retained_error == "generation-4"
    assert template.robot_success_floor == 0.30
    assert template.decoy_margin == 0.02 and template.decoy_margin_of == "fleet"
    assert template.fleet_fault and template.band == (0.60, 0.75)
    assert estimate.RETAINED_ERRORS[template.retained_error] == {
        name: (-0.021, 0.021) for name in ("observed", "experiment", "scoring")
    }
    for name in ("D2", "D3"):
        dev = planned_recipe(name, world_id="w99", seed=(1 << 62) + 1)
        assert dev.retained_error == "none"
        assert dev.aim == pytest.approx((0.640, 0.715))


@pytest.mark.parametrize("key", ["w48", "w49"])
def test_original_stored_record_recomposes_without_resealing(key: str) -> None:
    from test_worlds_generation4 import TEXTS

    from fleetlab.contracts.identity import Task
    from fleetlab.worlds import world

    path = Path.home() / f".cache/robots-probes/worlds/{key}/files/generator.json"
    if not path.is_file():
        pytest.skip("original world records are not installed on this host")
    before = path.read_bytes()
    record = json.loads(before)
    texts = {Task(r["suite"], r["task_index"]): r["text"] for r in TEXTS}
    rebuilt = world.recompose_record(record, texts)
    assert world.world_hash(rebuilt) == record["recipe"]["sealed_hash"]
    assert rebuilt.composition.expected["forecast_seal"] == record["expected"]["forecast_seal"]
    assert path.read_bytes() == before
    damaged = json.loads(before)
    damaged["recipe"]["calibration_sources"][-1] = "task_refinement sha256 0000000000000000"
    with pytest.raises(ValueError, match="sealed recipe calibration changed"):
        world.recompose_record(damaged, texts)


def test_drafts_avoid_same_day_overlapping_onsets(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.worlds import compose
    from fleetlab.worlds.recipe import planned_recipe

    c = compose.Composer(planned_recipe("D2", world_id="w99", seed=compose.ASSESSMENT_SEEDS[0]))
    # Synthetic feasibility isolates structural drafting; no claim about measured cells.
    monkeypatch.setattr(c, "supported", lambda *args, **kw: True)
    monkeypatch.setattr(c, "observed_logged", lambda *args: 0.8)
    monkeypatch.setattr(c, "measured_drift_rate", lambda *args: 0.1)
    monkeypatch.setattr(c, "pair_scope_choices", lambda pair: [r for r, _ in c.robots])
    for attempt in range(20):
        draft = c.draft(attempt)
        assert not c.co_onsets(c.schedule(draft), draft.patterns), attempt


def test_plain_draft_targets_forecast_across_levels_and_onsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_worlds_task_guards import task_recipe

    from fleetlab.worlds import base, compose

    c = compose.Composer(
        task_recipe(search_revision="r7", robot_success_floor=0.3, aim=(0.66, 0.695))
    )
    c.plain_menu = [("action_latency", "robot")]
    c.effect_levels["action_latency"] = ("8 steps", "17 steps", "20 steps")
    c._used, c._totals, c._heavy, c._clear_sites = {}, 0, 0, set()

    # The middle dose is on target only from day 2. The strongest dose is below
    # the floor here; retaining its label lets a later overlap/onset reconsider it.
    def forecast(schedule: list[Any], *args: Any) -> float:
        fault = schedule[-1]
        return float({8: 0.9, 17: 0.64 + 0.02 * fault.schedule.onset_day, 20: 0.1}[fault.level])

    monkeypatch.setattr(c, "observed_logged", forecast)
    monkeypatch.setattr(c, "supported", lambda schedule, *args: forecast(schedule) >= 0.33)
    plain = c._draw_plain(
        base.rng(c.recipe.seed, "target-test"),
        compose.Occupancy(c.robots),
        {robot for robot, _ in c.robots if robot != "r01"},
        "any",
        "f01",
        lambda: "4.3.0",
        draft=compose.Draft(),
    )
    assert plain is not None
    assert (plain.labels[plain.level], plain.onset) == ("17 steps", 2)
    assert plain.labels == ("8 steps", "17 steps", "20 steps")


def test_interaction_draft_tries_later_supported_onsets(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.worlds import compose
    from fleetlab.worlds.recipe import planned_recipe

    recipe = planned_recipe("D2", world_id="w99", seed=compose.ASSESSMENT_SEEDS[0]).model_copy(
        update={
            "search_revision": "r7",  # r8 keeps round 6's day 1 (tests/test_worlds_round8.py)
            "fleet_fault": False,
            "decoys": 0,
            "additive_pairs": 0,
            "drifts": 0,
            "confounders": 0,
            "plain": 0,
            "small": 0,
        }
    )
    c = compose.Composer(recipe)
    pair = (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk"))
    monkeypatch.setattr(c, "_joint_pairs", lambda: [pair])
    monkeypatch.setattr(c, "pair_scope_choices", lambda *args: ["r01"])
    # A synthetic observed-window floor passes at day 2, not the first free day.
    # Actual admission must still examine the whole overlapping schedule.
    tried = []

    def supported(schedule: list[Any], *args: Any) -> bool:
        day = schedule[-1].schedule.onset_day
        tried.append(day)
        return bool(day == 2)

    monkeypatch.setattr(c, "supported", supported)
    draft = c.draft(0)
    assert not draft.stopped
    assert draft.patterns.interactions
    assert draft.fixed[-1].schedule.onset_day == 2
    assert tried == [1, 2]


def test_plain_prefilter_keeps_a_rescuing_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_worlds_measured import cell
    from test_worlds_task_guards import task_recipe

    from fleetlab.contracts.identity import Task
    from fleetlab.worlds import base, compose, curves

    delay, wear = calibration._D17, calibration._W50
    table = {
        (curves.SPATIAL, 0, tuple(sorted(members))): cell(rate)
        for members, rate in (((), 0.98), ((delay,), 0.1), ((wear,), 0.9), ((delay, wear), 0.7))
    }
    monkeypatch.setattr(curves, "measured_cells", lambda: table)
    c = compose.Composer(
        task_recipe(
            estimator="measured-v3",
            search_revision="r7",
            robot_success_floor=0.3,
            tasks=(Task(curves.SPATIAL, 0),),
        )
    )
    c.plain_menu = [("action_latency", "robot")]
    c.effect_levels["action_latency"] = ("17 steps",)
    c._used, c._totals, c._heavy, c._clear_sites = {}, 0, 0, set()
    draft = compose.Draft(fixed=[compose.make_fault("f01", *wear, "r01", 0)])
    assert not c.stands_alone("action_latency", ("17 steps",), "r01", draft)
    faulted = {r for r, _ in c.robots if r != "r01"}
    plain = c._draw_plain(
        base.rng(c.recipe.seed, "rescue-test"),
        compose.Occupancy(c.robots),
        faulted,
        "any",
        "f02",
        lambda: "4.3.0",
        draft=draft,
    )
    assert plain is not None
    draft.plains.append(plain)
    assert c.supported(c.schedule(draft), draft)


def test_settling_crosses_a_two_move_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_worlds_task_guards import task_recipe

    from fleetlab.worlds import compose

    c = compose.Composer(
        task_recipe(
            estimator="measured-v3", search_revision="r7", aim=(0.66, 0.69), robot_success_floor=0.3
        )
    )
    draft = compose.Draft(
        plains=[
            compose.Plain("action_latency", robot, ("8 steps", "17 steps"), 0, 0, 0, fid)
            for robot, fid in (("r01", "f01"), ("r02", "f02"))
        ]
    )
    rates: dict[tuple[float, ...], float] = {
        (8, 8): 0.8,
        (17, 8): 0.95,
        (8, 17): 0.45,
        (17, 17): 0.68,
    }
    monkeypatch.setattr(
        c, "observed_logged", lambda schedule, ctx: rates[tuple(f.level for f in schedule)]
    )
    monkeypatch.setattr(c, "supported", lambda *args, **kw: True)
    result = c.settle_measured(draft, 0)
    assert rates[tuple(f.level for f in result)] == 0.68


def test_settling_moves_a_pattern_onset(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_worlds_task_guards import task_recipe

    from fleetlab.worlds import compose

    c = compose.Composer(
        task_recipe(estimator="measured-v3", search_revision="r7", aim=(0.66, 0.69))
    )
    fault = compose.make_fault("f01", "object_batch_variant", "4 mass doublings", "s1", 1)
    draft = compose.Draft(fixed=[fault], roles={"f01": "additive_pair"})
    monkeypatch.setattr(
        c, "observed_logged", lambda schedule, ctx: 0.60 + 0.04 * schedule[0].schedule.onset_day
    )
    monkeypatch.setattr(c, "supported", lambda *args, **kw: True)
    result = c.settle_measured(draft, 0)
    assert result[0].schedule.onset_day == 2


def _copy_run_tables(source: Path, target: Path, *, drop_outcome: bool = False) -> None:
    import shutil

    import pyarrow.parquet as pq

    from fleetlab.log.runfiles import table_path

    if not (source / "run.json").is_file():
        pytest.skip("archived measurement runs are not installed on this host")
    target.mkdir(parents=True)
    shutil.copyfile(source / "run.json", target / "run.json")
    for name in ("identities", "verdicts", "episodes"):
        destination = table_path(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = pq.read_table(table_path(source, name))
        if drop_outcome and name == "episodes":
            data = data.slice(1)
        pq.write_table(data, destination)


def test_folded_world_with_missing_outcome_is_not_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "folded"
    (root / "files").mkdir(parents=True)
    (root / "files/spec.json").write_text(json.dumps({"world_id": "w46"}))
    source = Path.home() / ".cache/robots-probes/worlds/w46-opaque-v1/runs/w46-observed"
    damaged = root / "runs/w46-observed"
    _copy_run_tables(source, damaged, drop_outcome=True)
    original = supplement._task_rows
    monkeypatch.setattr(
        supplement, "_task_rows", lambda p: original(p) if p == str(damaged) else []
    )
    with pytest.raises(ValueError, match=r"incomplete.*outcomes"):
        calibration.refinement_record([root])


def test_stage_without_passing_chain_contributes_no_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    source = Path.home() / ".cache/robots-probes/worlds-v4/20261010-task-cells-r6c"
    root = tmp_path / "stage-chain"
    name = "topups-1"
    damaged = root / name / "run"
    _copy_run_tables(source / name / "run", damaged)
    run = json.loads((damaged / "run.json").read_text())
    run["chain_ok"] = False
    (damaged / "run.json").write_text(json.dumps(run))
    plan = json.loads((source / "plan.json").read_text())
    (root / "plan.json").write_text(
        json.dumps({"order": [name], "stages": {name: plan["stages"][name]}, "assays": []})
    )
    shutil.copyfile(source / f"{name}.jsonl", root / f"{name}.jsonl")
    original = supplement._task_rows
    monkeypatch.setattr(
        supplement, "_task_rows", lambda p: original(p) if p == str(damaged) else []
    )
    record = calibration.refinement_record([], root)
    assert record["cells"] == []
    assert record["skipped_stages"] == [name]


def test_generation4_interval_moves_both_ends_outward_without_moving_point() -> None:
    from dataclasses import replace

    from test_worlds_measured import context

    from fleetlab.worlds import curves, estimate

    ctx = context({(): 0.68}, 1000)
    episodes = [estimate.Episode(curves.SPATIAL, (), 0)] * 1008
    for window in ("observed", "experiment", "scoring"):
        base = estimate.prediction(episodes, replace(ctx, retained_error="none"), window=window)
        kept = estimate.prediction(
            episodes, replace(ctx, retained_error="generation-4"), window=window
        )
        assert kept["model_error_points"] == [-2.1, 2.1]
        for outcome in ("true", "logged"):
            assert kept[outcome]["expected"] == base[outcome]["expected"]
            assert (
                kept[outcome]["conditional_prediction_interval"]
                == base[outcome]["conditional_prediction_interval"]
            )
            low, high = base[outcome]["prediction_interval"]
            assert kept[outcome]["prediction_interval"] == pytest.approx(
                [low - 0.021, high + 0.021], abs=0.00011
            )
