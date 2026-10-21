"""Hand-counted edge rules for development-world scoring, pooling and costs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_scoring import (
    CAMERA,
    WEAR,
    act,
    cause,
    episode,
    inputs,
    oracle_table,
)

from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.actions import SCORING_BUDGETS, Budget, FixSet
from fleetlab.contracts.faults import Direction, Drift
from fleetlab.contracts.oracle import Gain, OracleRow, OracleTable, fault_list_hash
from fleetlab.contracts.tools import Diagnosis, RetrainIndicator, Warning
from fleetlab.contracts.vocabulary import Verb
from fleetlab.oracle.world import Cause as OracleCause
from fleetlab.scoring import score
from fleetlab.scoring.accounting import analysis_ledgers, score_accounting
from fleetlab.scoring.foresight import score_foresight
from fleetlab.scoring.loading import baseline_answers
from fleetlab.scoring.models import Metric
from fleetlab.scoring.pooling import pool as score_pool
from fleetlab.scoring.pooling import pool_markdown
from fleetlab.scoring.splits import score_splits
from fleetlab.scoring.thresholds import PASS_MARKS


def pool(reports: list[Any], *, expected_worlds: list[str]) -> dict[str, Any]:
    """Toy comparisons use an explicit fixture roster, never infer its world list."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        roster = Path(directory) / "toy-roster.json"
        hashes = {r.world_id: r.metadata.get("world_hash") for r in reports}
        roster.write_text(
            json.dumps(
                {
                    "worlds": [
                        {"world_id": world, "world_hash": hashes.get(world)}
                        for world in expected_worlds
                    ]
                }
            )
        )
        return score_pool(reports, roster=roster)


def diagnosis(**kwargs: Any) -> Diagnosis:
    return Diagnosis(
        **{"causes": [], "warnings": [], "retrain": [], "mode_summaries": [], **kwargs}
    )


def test_empty_explicit_answer_is_a_measured_floor() -> None:
    report = score(inputs(diagnosis=diagnosis(), fix_set=None, decision_actions=[]))
    assert report.claim1.recall.value == 0
    for budget in report.claim3.budgets:
        assert budget.agent.gain == Gain()
        assert budget.agent.share.value == 0
    assert report.headline.rows[0].share.value == 0
    assert {p["metric"]: p["status"] for p in report.pass_marks}["claim1.recall"] == "missed"
    assert report.errors["missed_causes"]


def test_no_missing_draw_is_conditioned_away() -> None:
    supplied = {
        "budget": "k = 3",
        "draws": 200,
        "valued": 199,
        "skipped_fresh_replay": 1,
        "missing": "one missing draw",
        "bounds": {"p95_points": [0, 20]},
    }
    result = score(inputs(oracle_chance=[supplied])).claim3.budgets[0]
    assert result.random.valued == 199 and result.random.draws == 200
    assert result.random.mean_points.value is None
    assert result.chance_bounds == {"p95_points": [0, 20]}
    assert result.exact_chance is None


def test_pair_is_one_detectable_cause_and_needs_both_members() -> None:
    unit = OracleCause("f01+f04", ("f01", "f04"), ("effective",), True, ("scoring",), 4, True)
    one = diagnosis(causes=[cause(CAMERA.target, Direction.degraded, 0.9)])
    partial = score(inputs(diagnosis=one, cause_units=(unit,)))
    assert partial.claim1.detectable == 1 and partial.claim1.named_cut == 3
    assert partial.claim1.recall.value == 0
    assert partial.claim1.hits == 0 and partial.claim1.small == 1
    both = diagnosis(causes=[*one.causes, cause(WEAR.target, Direction.degraded, 0.9)])
    found = score(inputs(diagnosis=both, cause_units=(unit,)))
    assert found.claim1.recall.value == 1
    assert found.claim1.units[0]["found"]
    assert found.claim1.precision.value == 1


def test_additive_pair_interaction_assertions_are_feedback_not_truth_hits() -> None:
    answer = diagnosis(
        causes=[
            cause(CAMERA.target, Direction.degraded, 0.9, "An interaction with the other component")
        ]
    )
    report = score(inputs(diagnosis=answer, additive_pairs=[("f01", "f04")]))
    assert report.remarks[0]["matched"] == "an additive pair reported as an interaction"


def test_frozen_marks_are_immutable_and_missing_live_is_explicit() -> None:
    assert dict(PASS_MARKS) == {
        "claim1.recall": 0.8,
        "claim1.decoy_rejection": 1,
        "claim2.spearman": 0.7,
        "claim3.share_k3": 0.5,
        "claim3.margin_points": 3,
        "claim3.above_chance": 0,
        "claim4.relative_rescue_improvement": 0.2,
    }
    with pytest.raises(TypeError):
        PASS_MARKS["claim1.recall"] = 0  # type: ignore[index]
    report = score(inputs())
    assert (
        next(p for p in report.pass_marks if p["metric"].startswith("claim4"))["status"]
        == "unavailable"
    )


def test_warning_crossing_is_strict_and_uses_issue_day_not_forecast_day() -> None:
    identities = [episode(n, "r01" if n <= 20 else "r02", [], day=9) for n in range(1, 41)]
    # r01: 11/20 fail; whole fleet: 20/40 fail. Exactly five points above: no crossing.
    rows = [
        OracleRow(
            episode_id=i.episode_id,
            active_faults=fault_list_hash(()),
            success=not (i.episode_index <= 11 or 21 <= i.episode_index <= 29),
        )
        for i in identities
    ]
    table = OracleTable(identities, rows)
    drift = CAMERA.model_copy(
        update={
            "schedule": CAMERA.schedule.model_copy(update={"drift": Drift(shape="linear", rate=1)})
        }
    )
    warning = Warning(scope="r01", metric="failure_rate", expected_day=10, confidence=0.1)
    args = inputs(
        drift_faults=[drift], as_run=table, issued_day=6, diagnosis=diagnosis(warnings=[warning])
    )
    first = score_foresight(args)["warnings"][0]
    assert first["crossing_day"] is None
    # One extra local failure is 57.5% vs 52.5%, still exactly five points.
    # Replace one remote failure with success: local .55, fleet .475, crossing.
    changed = [
        r.model_copy(update={"success": True}) if r.episode_id == "e000029" else r for r in rows
    ]
    result = score_foresight(replace(args, as_run=OracleTable(identities, changed)))["warnings"][0]
    assert result["old_wording_crossing_day"] == 9
    assert result["logged_rate_measure"]["state"] == "later crossing, warned"
    assert result["logged_rate_measure"]["lead_days"] == 3
    assert result["logged_rate_measure"]["forecast_errors"][0]["error_days"] == 1
    assert result["crossing_day"] is None and result["lead_days"] is None
    logged_late = score_foresight(
        replace(args, as_run=OracleTable(identities, changed), issued_day=9)
    )["warnings"][0]["logged_rate_measure"]
    assert logged_late["state"] == "already crossed"
    assert logged_late["noticed"] is True and logged_late["lead_days"] is None
    logged_unwarned = score_foresight(
        replace(args, as_run=OracleTable(identities, changed), diagnosis=diagnosis())
    )["warnings"][0]["logged_rate_measure"]
    assert logged_unwarned["state"] == "later crossing, not warned"
    assert logged_unwarned["before_crossing"] is False
    assert logged_unwarned["lead_days"] is None
    result = score_foresight(replace(args, drift_crossing_days={"f01": 9}))["warnings"][0]
    assert result["crossing_day"] == 9 and result["lead_days"] == 3
    assert result["before_crossing"] is True
    late = score_foresight(replace(args, drift_crossing_days={"f01": 9}, issued_day=9))["warnings"][
        0
    ]
    assert late["before_crossing"] is None and late["lead_days"] is None
    assert late["state"] == "already crossed"
    gap = score_foresight(replace(args, as_run=OracleTable(identities, rows[1:])))["warnings"][0]
    assert (
        gap["crossing_day"] is None
        and gap["old_wording_missing"] == "daily as-run outcomes incomplete"
    )


def test_retraining_precision_requires_every_healthy_row_and_deduplicates() -> None:
    table = oracle_table()
    examples = ["e000001", "e000001", "e000003"]
    indicator = RetrainIndicator(
        failure_mode="mode",
        episode_ids=examples,
        why_no_action_fixes_it="Policy fails on a healthy robot",
        data_wanted="examples",
    )
    result = score_foresight(inputs(diagnosis=diagnosis(retrain=[indicator])))["retrain"][0]
    assert result["examples"] == 2
    assert result["precision"] is None  # all-removed row including the decoy is missing
    identities = [episode(1, "r01", []), episode(2, "r02", [])]
    table = OracleTable(
        identities,
        [
            OracleRow(
                episode_id=i.episode_id,
                active_faults=fault_list_hash(()),
                success=i.episode_index == 2,
            )
            for i in identities
        ],
    )
    indicator = indicator.model_copy(update={"episode_ids": ["e000001", "e000002", "e000001"]})
    result = score_foresight(inputs(oracle=table, diagnosis=diagnosis(retrain=[indicator])))[
        "retrain"
    ][0]
    assert result["precision"] == 0.5 and result["healthy_failures"] == 1
    # Examples may come from the observed window: use any publicly checked healthy row.
    full = score_foresight(
        inputs(
            oracle=table.restricted(identities[:1]),
            as_run=table,
            diagnosis=diagnosis(retrain=[indicator]),
        )
    )["retrain"][0]
    assert full["precision"] == 0.5 and full["missing_examples"] == []
    outside = indicator.model_copy(update={"episode_ids": ["e000001", "e999999"]})
    absent = score_foresight(inputs(oracle=table, diagnosis=diagnosis(retrain=[outside])))[
        "retrain"
    ][0]
    assert absent["precision"] is None and absent["missing_examples"] == ["e999999"]


def test_baselines_share_the_same_budgets_and_missing_budget_cannot_pass() -> None:
    target = act(Verb.recalibrate, "r01", component=CAMERA.target.component)
    report = score(inputs(baseline_rankings={"health": [target]}))
    assert all(b.baselines["health"].gain == Gain.of(2, 1) for b in report.claim3.budgets)
    fixed = FixSet(actions=[], budget=Budget(kind="count", amount=3))
    partial = score(inputs(baseline_fix_sets={"vision": {"k = 3": fixed}}))
    assert partial.claim3.budgets[0].baselines["vision"].gain == Gain()
    assert partial.claim3.budgets[1].baselines["vision"].gain is None


def test_baseline_file_reader_checks_ranked_and_fixed_formats(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(FixSet(actions=[], budget=Budget(kind="count", amount=3)).model_dump_json())
    ranking, fixed = baseline_answers({"vision": path})
    assert not ranking and fixed["vision"]["k = 3"].actions == []
    from test_scoring import FIX_SET

    path.write_text(FIX_SET.model_dump_json())
    ranking, fixed = baseline_answers({"health": path})
    assert len(ranking["health"]) == 10 and not fixed


def test_explicit_baseline_cost_set_is_not_silently_recut() -> None:
    fixes = FixSet(
        actions=[
            act(Verb.replace_, "r01", component=CAMERA.target.component),
            act(Verb.replace_, "r02", component=CAMERA.target.component),
            act(Verb.replace_, "r03", component=CAMERA.target.component),
        ],
        budget=Budget(kind="cost", amount=4),
    )
    with pytest.raises(ValueError, match="exceeds or mislabels"):
        score(inputs(baseline_fix_sets={"baseline": {"4 h": fixes}}))


def test_known_unseen_uses_type_names_and_separates_mixed_cohorts() -> None:
    split = score_splits(inputs(), {"action_latency"})
    assert split["diagnosis"]["unseen"]["detectable"] == 1
    assert split["diagnosis"]["known"]["detectable"] == 1
    assert split["decision_cohorts"]["unseen_including_mixed"]["episodes"] > 0
    empty = score_splits(inputs(), set())
    assert empty["diagnosis"]["unseen"]["recall"] is None
    assert empty["decision_cohorts"]["unseen_including_mixed"]["episodes"] == 0


def test_pool_uses_world_mean_not_episode_weighting_and_separates_repeats() -> None:
    first, second = score(inputs()), score(inputs())
    second.world_id = "w02"
    second.scoring_episodes = 10000
    first.claim1.recall = Metric.of(0.4)
    second.claim1.recall = Metric.of(1)
    first.metadata = {
        "method": "agent",
        "condition": "logs alone",
        "repeat": 1,
        "run_labels_verified": True,
    }
    second.metadata = dict(first.metadata)
    repeated = first.model_copy(deep=True)
    repeated.metadata["repeat"] = 2
    repeated.claim1.recall = Metric.of(0.9)
    pooled = pool([first, second, repeated], expected_worlds=["w01", "w02"])
    row = next(r for r in pooled["pooled"] if r["metric"] == "recall" and r["repeat"] == 1)
    assert row["mean"] == 0.7 and row["minimum"] == 0.4 and row["maximum"] == 1
    repeat2 = next(r for r in pooled["pooled"] if r["metric"] == "recall" and r["repeat"] == 2)
    assert repeat2["mean"] == 0.9 and not repeat2["complete"]
    assert len(pooled["required_cells"]) == 8
    assert "with experiments, repeat 2" in pool_markdown(pooled)
    with pytest.raises(ValueError, match="duplicate analysis cell"):
        pool([first, first], expected_worlds=["w01"])


def test_pool_includes_type_cohorts_counts_and_unavailable_empty_split() -> None:
    args = inputs()
    first, second = score(args), score(args)
    first.splits = score_splits(args, set())
    second.splits = score_splits(args, set())
    first.metadata = {
        "method": "agent",
        "condition": "logs alone",
        "repeat": 1,
        "run_labels_verified": True,
    }
    second.metadata = dict(first.metadata)
    second.world_id = "w02"
    second.claim3.budgets[0].agent.gain = Gain.of(8, 2)
    pooled = pool([first, second], expected_worlds=["w01", "w02"])
    rescued = next(
        r for r in pooled["pooled"] if r["metric"] == "rescued" and r["budget"] == "k = 3"
    )
    first_gain = first.claim3.budgets[0].agent.gain
    assert first_gain is not None
    assert rescued["mean"] == (first_gain.rescued + 8) / 2
    unseen = next(
        r
        for r in pooled["pooled"]
        if r["metric"] == "unseen_including_mixed share" and r["budget"] == "k = 3"
    )
    assert unseen["mean"] is None and unseen["worlds_valued"] == 0
    assert unseen["worlds_expected"] == 2
    text = pool_markdown(pooled)
    assert "known_only world_points" in text and "known spearman" in text
    assert "confounder resolution" in text


def test_cost_ledger_counts_retries_cached_tokens_and_model_tiers(tmp_path: Path) -> None:
    analysis = Analysis.create("w01", base=tmp_path / "answers")
    run1, run2 = tmp_path / "call1", tmp_path / "call2"
    (analysis.root / "sessions/session-01.json").write_text(
        json.dumps({"run_dir": str(run2), "attempts": ["timeout", None]})
    )
    rows = [
        {
            "run_dir": str(run),
            "label": analysis.analysis_id + "/session-01",
            "backend": "codex",
            "model": "exact-model",
            "service_tier": "fast",
            "seconds": seconds,
            "ok": ok,
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 250,
                "output_tokens": 100,
                "reasoning_tokens": 50,
            },
        }
        for run, seconds, ok in ((run1, 3, False), (run2, 7, True))
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in rows))
    prices = {
        "exact-model": {
            "input": 2,
            "cached_input": 0.2,
            "output": 10,
            "source": "constructed test schedule",
            "as_of": "2026-10-09",
        }
    }
    result = score_accounting(analysis, 100, ledger, prices)
    tier = result["tiers"][0]
    assert result["ledger_calls"] == 2 and result["missing"] is None
    assert tier["failed_calls"] == 1 and tier["wall_seconds"] == 10
    assert tier["api_price_equivalent_usd"] == pytest.approx(0.0051)
    assert tier["per_thousand_attempts"]["wall_seconds"] == 100
    assert tier["per_thousand_attempts"]["input_tokens"] == 20000
    assert score_accounting(analysis, 100, ledger)["tiers"][0]["api_price_equivalent_usd"] is None
    ledger.write_text(json.dumps(rows[0]))
    incomplete = score_accounting(analysis, 100, ledger, prices)
    assert incomplete["missing"] is not None
    assert incomplete["tiers"][0]["per_thousand_attempts"]["input_tokens"] is None


def test_accounting_follows_recorded_run_roots_and_requires_session_receipts(
    tmp_path: Path,
) -> None:
    analysis = Analysis.create("w01", base=tmp_path / "results")
    paths = []
    for number, seconds in ((1, 3), (2, 7)):
        root = tmp_path / f"runtime{number}"
        root.mkdir()
        run = root / "call"
        (analysis.root / f"sessions/session-{number:02d}.json").write_text(
            json.dumps({"run_dir": str(run), "attempts": [None]})
        )
        ledger = root / "ledger.jsonl"
        ledger.write_text(
            json.dumps(
                {
                    "run_dir": str(run),
                    "label": f"{analysis.analysis_id}/session-{number:02d}",
                    "backend": "codex",
                    "model": "fixture-model",
                    "seconds": seconds,
                    "ok": True,
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "output_tokens": 10,
                        "reasoning_tokens": 0,
                        "cost_usd_list_price": 0.01,
                    },
                }
            )
        )
        paths.append(ledger)
    discovered = analysis_ledgers(analysis)
    assert discovered == paths
    result = score_accounting(analysis, 100, discovered)
    assert result["missing"] is None and result["ledger_calls"] == 2
    assert result["tiers"][0]["per_thousand_attempts"]["wall_seconds"] == 100
    assert result["tiers"][0]["per_thousand_attempts"]["input_tokens"] == 2000
    assert analysis_ledgers(analysis, paths[0]) == paths[:1]
    for file in (analysis.root / "sessions").glob("*.json"):
        file.unlink()  # fixture only, not a saved measurement
    incomplete = score_accounting(analysis, 100, discovered)
    assert incomplete["missing"] == "runtime session records absent"
    assert incomplete["tiers"][0]["per_thousand_attempts"]["wall_seconds"] is None


def test_authoritative_budget_files_override_padded_rankings(tmp_path: Path) -> None:
    from test_scoring import FIX_SET

    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox/fix_set.json").write_text(FIX_SET.model_dump_json())
    (tmp_path / "fix_sets").mkdir()
    for budget in SCORING_BUDGETS:
        name = f"{budget.kind}-{budget.amount:g}"
        (tmp_path / "fix_sets" / (name + ".json")).write_text(
            FixSet(actions=[], budget=budget).model_dump_json()
        )
    ranking, fixed = baseline_answers({"vision": tmp_path})
    assert not ranking
    assert len(fixed["vision"]) == 4
    assert all(f.actions == [] for f in fixed["vision"].values())


def test_partial_rank_correlation_cannot_meet_the_frozen_mark() -> None:
    report = score(inputs())
    assert report.claim2.spearman.value is not None
    assert not report.claim2.complete
    mark = next(p for p in report.pass_marks if p["metric"] == "claim2.spearman")
    assert mark["status"] == "unavailable" and "partial" in mark["missing"]


def test_denied_interaction_is_not_a_false_positive_assertion() -> None:
    answer = diagnosis(
        causes=[cause(CAMERA.target, Direction.degraded, 0.9, "There is no interaction")]
    )
    report = score(inputs(diagnosis=answer, additive_pairs=[("f01", "f04")]))
    assert not report.errors["false_positives"]


def test_cost_does_not_lose_failed_attempts_if_last_call_is_present(tmp_path: Path) -> None:
    analysis = Analysis.create("w01", base=tmp_path / "answers")
    run = tmp_path / "last-call"
    (analysis.root / "sessions/session-01.json").write_text(
        json.dumps({"number": 1, "run_dir": str(run), "attempts": ["timeout", None]})
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "run_dir": str(run),
                "label": analysis.analysis_id + "/session-01",
                "model": "m",
                "backend": "codex",
                "seconds": 1,
                "ok": True,
                "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
            }
        )
    )
    scored = score_accounting(analysis, 10, ledger)
    assert scored["missing"] == "failed retry calls absent from ledger"
    assert scored["tiers"][0]["per_thousand_attempts"]["wall_seconds"] is None


def test_generator_unit_does_not_hide_wrong_direction_feedback() -> None:
    unit = OracleCause("f01", ("f01",), ("effective",), True, ("scoring",), 4, True)
    answer = diagnosis(causes=[cause(CAMERA.target, Direction.changed, 0.9)])
    report = score(inputs(diagnosis=answer, cause_units=(unit,)))
    assert report.claim1.causes[0].verdict == "wrong direction"
    assert len(report.errors["wrong_directions"]) == 1
    assert report.claim1.recall.value == 0


def comparison_report(world: str = "w01") -> Any:
    """Hand-valued comparison: 100 possible net rescues, a 10-point chance tail."""
    from fleetlab.scoring.comparisons import BASELINE_METHODS, REQUIRED_BASELINES
    from fleetlab.scoring.decisions import headline
    from fleetlab.scoring.models import Achievable, FixSetValue
    from fleetlab.scoring.thresholds import pass_marks

    report = score(inputs(oracle_chance=[]))
    report.world_id = world
    report.scoring_episodes = 100
    report.metadata = {
        "method": "agent",
        "condition": "logs alone",
        "repeat": 1,
        "run_labels_verified": True,
    }
    at = report.claim3.budgets[0]
    at.world_id = world
    at.achievable = Achievable(gain=Gain.of(100, 0), points=100)
    at.agent = FixSetValue(gain=Gain.of(90, 0), points=90, share=Metric.of(0.9))
    at.baselines = {
        name: FixSetValue(gain=Gain.of(40, 0), points=40, share=Metric.of(0.4))
        for name in REQUIRED_BASELINES
    }
    at.baseline_sources = {
        name: {
            "source": f"/constructed/{name}",
            "binding": "constructed producer output",
            "record": {"method": methods[0], "repeat": 1, "world_id": world},
        }
        for name, methods in BASELINE_METHODS.items()
        if name in REQUIRED_BASELINES
    }
    at.no_data = FixSetValue(missing="no no-data ranking supplied")
    at.exact_chance = {
        "histogram": [[0, 94], [10, 6]],
        "sets": 100,
        "unvalued_sets": 0,
        "mean_points": 0.6,
        "p5_points": 0,
        "p95_points": 10,
    }
    at.chance_missing = None
    # These empirical figures deliberately disagree with the population floor.
    at.random.p95_points = Metric.of(0)
    at.random.p95_share = Metric.of(0)
    report.pass_marks = pass_marks(report)
    report.headline = headline(report.claim3)
    return report


def mark(report: Any, name: str) -> dict[str, Any]:
    from fleetlab.scoring.thresholds import pass_marks

    return next(p for p in pass_marks(report) if p["metric"] == name)


def test_all_required_baselines_and_first_model_repeats_are_mandatory() -> None:
    from fleetlab.scoring.comparisons import REQUIRED_BASELINES
    from fleetlab.scoring.decisions import headline
    from fleetlab.scoring.models import FixSetValue

    report = comparison_report()
    at = report.claim3.budgets[0]
    at.baselines["classical statistics"] = FixSetValue(
        gain=Gain.of(90, 0), points=90, share=Metric.of(0.9)
    )
    assert mark(report, "claim3.margin_points")["value"] == 0
    assert mark(report, "claim3.margin_points")["status"] == "missed"
    assert headline(report.claim3).ratio == 1
    for name in REQUIRED_BASELINES:
        saved = at.baselines.pop(name)
        missing = mark(report, "claim3.margin_points")
        assert missing["value"] is None and name in missing["missing"]
        assert headline(report.claim3).ratio is None
        at.baselines[name] = saved
        at.baselines[name] = FixSetValue(missing="missing replay")
        assert mark(report, "claim3.margin_points")["status"] == "unavailable"
        assert headline(report.claim3).ratio is None
        at.baselines[name] = saved
    at.baselines["no-data ranker repeat 1"] = FixSetValue(
        gain=Gain.of(95, 0), points=95, share=Metric.of(0.95)
    )
    at.baselines["no-data ranker repeat 2"] = FixSetValue(
        gain=Gain.of(100, 0), points=100, share=Metric.of(1)
    )
    at.baseline_sources["no-data ranker repeat 1"] = at.baseline_sources["no-data ranker"]
    at.baseline_sources["no-data ranker repeat 2"] = {
        "source": "/constructed/prior2",
        "record": {"method": "no_data", "repeat": 2},
    }
    assert mark(report, "claim3.margin_points")["value"] == -5
    head = headline(report.claim3)
    assert head.ratio == pytest.approx(90 / 95)
    assert {r.method for r in head.rows} >= {"no-data ranker repeat 1", "no-data ranker repeat 2"}
    at.baselines.pop("no-data ranker repeat 1")
    at.baselines.pop("no-data ranker")
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None


def test_marks_use_exact_upper_chance_strictly_and_k3_inclusively() -> None:
    from fleetlab.scoring.decisions import headline

    report = comparison_report()
    at = report.claim3.budgets[0]
    at.agent.points = 10
    at.agent.gain = Gain.of(10, 0)
    assert mark(report, "claim3.above_chance")["status"] == "missed"
    at.agent.points = 11
    at.agent.gain = Gain.of(11, 0)
    assert mark(report, "claim3.above_chance")["status"] == "met"
    at.exact_chance = {
        "histogram": [[5, 90]],
        "sets": 100,
        "unvalued_sets": 10,
        "net_low": 5,
        "net_high": 15,
        "p95_points": None,
        "bounds": {"p95_points": [5, 15]},
    }
    assert mark(report, "claim3.above_chance")["status"] == "missed"
    at.agent.points = 43
    at.agent.gain = Gain.of(43, 0)
    assert mark(report, "claim3.margin_points")["status"] == "met"  # exactly three
    at.agent.share = Metric.of(0.5)
    report.claim3.budgets[1].agent.share = Metric.of(0)
    assert mark(report, "claim3.share_k3")["status"] == "met"  # equality at k3
    report.claim1.recall = Metric.of(0.8)
    report.claim1.decoy_rejection = Metric.of(1)
    report.claim2.spearman = Metric.of(0.7)
    report.claim2.complete = True
    for name in ("claim1.recall", "claim1.decoy_rejection", "claim2.spearman"):
        assert mark(report, name)["status"] == "met"
    at.exact_chance = None
    at.chance_missing = "produce oracle report with COMMAND"
    assert mark(report, "claim3.above_chance")["value"] is None
    assert "COMMAND" in mark(report, "claim3.above_chance")["missing"]
    assert headline(report.claim3).ratio is None
    # Best baseline, not the weakest, decides the headline's chance comparison.
    at.exact_chance = {"histogram": [[40, 100]], "sets": 100, "unvalued_sets": 0, "p95_points": 40}
    assert headline(report.claim3).ratio is None


def test_zero_achievable_share_is_undefined_even_for_zero_or_positive_net() -> None:
    from fleetlab.scoring.decisions import share_of
    from fleetlab.scoring.models import Achievable

    for net in (0, 1, -1):
        share = share_of(net, Achievable(gain=Gain()), None)
        assert share.value is None and "zero" in str(share.missing)


def test_cause_dedup_targetless_scopes_confidence_and_inert() -> None:
    from test_scoring import DECOY, INERT, LATENCY

    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import Component

    hit = cause(CAMERA.target, Direction.degraded, 0.9)
    wrong = cause(Target(scope="r02", component=Component.arm), Direction.degraded, 0.9)
    answer = diagnosis(causes=[hit, hit, hit, wrong, wrong])
    result = score(inputs(diagnosis=answer)).claim1
    assert result.hits == 1 and result.false_positives == 1 and result.precision.value == 0.5
    assert result.named == 2
    vague = cause(Target(scope=CAMERA.target.scope), Direction.degraded, 0.9)
    result = score(inputs(diagnosis=diagnosis(causes=[hit, vague]))).claim1
    assert result.vague == 1 and result.precision.value == 0.5
    assert result.precision_without_vague.value == 1
    assert result.false_positives == 1 and result.targetless == 1
    assert result.precision_with_vague.value == 0.5
    for target in (DECOY.target, INERT.target):
        result = score(
            inputs(
                diagnosis=diagnosis(
                    causes=[cause(Target(scope=target.scope), Direction.changed, 0.9)]
                )
            )
        ).claim1
        assert result.vague == 0 and result.false_positives == 1
    result = score(
        inputs(diagnosis=diagnosis(causes=[cause(INERT.target, Direction.changed, 0.9)]))
    ).claim1
    assert result.false_positives == 1
    both = diagnosis(
        causes=[
            cause(LATENCY.target, Direction.changed, 0.9),
            cause(INERT.target, Direction.changed, 0.9),
        ]
    )
    assert (
        score(inputs(diagnosis=both, confounders=[("f02", "f05")])).claim1.confounders[0].resolved
        is False
    )
    candidates = [
        cause(Target(scope=f"r0{i}", component=Component.arm), Direction.degraded, 0.9)
        for i in range(2, 6)
    ]
    result = score(
        inputs(
            diagnosis=diagnosis(causes=[*candidates, cause(LATENCY.target, Direction.changed, 0.5)])
        )
    ).claim1
    assert result.named == 5 and result.recall.value == 0.5


def test_curve_truth_disagreements_are_descriptive_beside_measured_membership() -> None:
    from fleetlab.scoring import report_markdown

    unit = OracleCause("f01+f04", ("f01", "f04"), ("effective",), True, ("scoring",), 4, True)
    report = score(inputs(cause_units=(unit,)))
    assert report.claim1_curve is not None
    assert report.claim1_curve.truth_disagreements == ["f01", "f02"]
    assert report.claim1_curve.recall.provisional and report.claim1_curve.precision.provisional
    assert report.claim1.truth_disagreements == []
    assert not report.claim1.recall.provisional and not report.claim1.precision.provisional
    text = report_markdown(report)
    assert "generator lists disagree for f01, f02" in text
    assert "provisional and descriptive only" in text


def test_retraining_requires_all_indicators_and_counts_only_healthy_failures() -> None:
    identities = [episode(n, "r01", []) for n in range(1, 5)]
    rows = [
        OracleRow(
            episode_id=i.episode_id, active_faults=fault_list_hash(()), success=i.episode_index != 1
        )
        for i in identities[:3]
    ]

    def indicator(ids: list[str]) -> RetrainIndicator:
        return RetrainIndicator(
            failure_mode="m", episode_ids=ids, why_no_action_fixes_it="x", data_wanted="y"
        )

    table = OracleTable(identities, rows)
    good = indicator(["e000001", "e000002", "e000003"])
    out = score_foresight(inputs(as_run=table, diagnosis=diagnosis(retrain=[good])))
    assert out["retrain"][0]["precision"] == pytest.approx(1 / 3)
    incomplete = indicator(["e000001", "e000004"])
    out = score_foresight(inputs(as_run=table, diagnosis=diagnosis(retrain=[good, incomplete])))
    assert all(r["precision"] is None for r in out["retrain"])
    assert out["retrain"][1]["healthy_failures"] == 1
    assert out["missing_examples"] == ["e000004"]
    assert "--request report.json" in out["replay_request_command"]


def test_warning_states_use_drift_truth_and_expose_false_alarms() -> None:
    drift = CAMERA.model_copy(
        update={
            "schedule": CAMERA.schedule.model_copy(update={"drift": Drift(shape="linear", rate=1)})
        }
    )
    warning = Warning(scope="r01", metric="x", expected_day=12, confidence=0.5)
    false_alarm = warning.model_copy(update={"scope": "r02"})
    args = inputs(
        drift_faults=[drift], issued_day=6, diagnosis=diagnosis(warnings=[warning, false_alarm])
    )
    out = score_foresight(args)
    assert out["warnings"][0]["state"] == "unavailable"
    assert out["false_alarm_count"] == 1
    for day, state, lead in (
        (5, "already crossed", None),
        (6, "already crossed", None),
        (9, "later crossing, warned", 3),
        (None, "never crossed", None),
    ):
        out = score_foresight(replace(args, drift_crossing_days={"f01": day}))
        row = out["warnings"][0]
        assert row["state"] == state and row["lead_days"] == lead
        if day is not None:
            assert row["forecast_errors"][0]["error_days"] == 12 - day
    out = score_foresight(
        replace(args, drift_crossing_days={"f01": 9}, diagnosis=diagnosis(warnings=[false_alarm]))
    )
    assert out["warnings"][0]["state"] == "later crossing, not warned"
    assert out["false_alarm_count"] == 1
    assert score_foresight(replace(args, drift_faults=[]))["false_alarm_count"] == 2
    issued_early = score_foresight(replace(args, issued_day=4, drift_crossing_days={"f01": 5}))
    assert issued_early["warnings"][0]["state"] == "later crossing, warned"
    assert issued_early["warnings"][0]["lead_days"] == 1


def test_pooled_ratio_best_per_world_and_exact_convolution_floor() -> None:
    from fleetlab.scoring.pooled_chance import pooled_floor
    from fleetlab.scoring.thresholds import pass_marks

    first, second = comparison_report(), comparison_report("w02")
    for report, name in ((first, "worst health metric"), (second, "classical statistics")):
        report.claim3.budgets[0].baselines[name].share = Metric.of(0.8)
        report.claim3.budgets[0].baselines[name].points = 80
        report.pass_marks = pass_marks(report)
    result = pool([first, second], expected_worlds=["w01", "w02"])
    ratio = result["headline_ratios"][0]
    assert ratio["ratio"] == pytest.approx(0.9 / 0.8)
    assert ratio["best_on_average"]["ratio"] == pytest.approx(0.9 / 0.6)
    # 94% zero and 6% .1 each: mean .05 with probability .1128, .1 with .0036.
    # The pooled p95 is .05; averaging per-world p95s would incorrectly yield .1.
    floor = ratio["pooled_chance"]
    assert floor["p95_low"] <= 0.05 <= floor["p95_high"]
    assert floor["p95_high"] - floor["p95_low"] <= 0.000251
    assert floor["percentile_error_share"] < 0.001
    for report in (first, second):
        for baseline in report.claim3.budgets[0].baselines.values():
            baseline.share = Metric.of(0.01)
    assert (
        pool([first, second], expected_worlds=["w01", "w02"])["headline_ratios"][0]["ratio"] is None
    )
    at = second.claim3.budgets[0]
    at.baselines.pop("dramatic videos")
    assert (
        pool([first, second], expected_worlds=["w01", "w02"])["headline_ratios"][0]["ratio"] is None
    )
    at.exact_chance = None
    assert pooled_floor({"w02": at})["missing_worlds"] == ["w02"]


def test_pooled_marks_read_mean_chance_everywhere_and_name_missing_world() -> None:
    from fleetlab.scoring.thresholds import pass_marks

    a, b = comparison_report(), comparison_report("w02")
    a.claim1.recall, b.claim1.recall = Metric.of(0.7), Metric.of(0.9)
    b.claim3.budgets[0].agent.points = 5  # below chance despite the high mean
    b.claim3.budgets[0].agent.gain = Gain.of(5, 0)
    a.pass_marks, b.pass_marks = pass_marks(a), pass_marks(b)
    result = pool([a, b], expected_worlds=["w01", "w02"])
    marks = {r["metric"]: r for r in result["pooled_pass_marks"]}
    assert marks["claim1.recall"]["status"] == "met"
    assert marks["claim1.recall"]["worlds_met"] == 1
    assert marks["claim3.above_chance"]["status"] == "missed"
    b.claim2.spearman = Metric.absent("constant predictions")
    b.pass_marks = pass_marks(b)
    marks = {
        r["metric"]: r for r in pool([a, b], expected_worlds=["w01", "w02"])["pooled_pass_marks"]
    }
    assert marks["claim2.spearman"]["status"] == "unavailable"
    assert "w02" in marks["claim2.spearman"]["missing_worlds"]


def test_run_record_refuses_condition_repeat_and_day_mismatches(tmp_path: Path) -> None:
    from fleetlab.scoring.loading import check_run_record

    analysis = Analysis.create("w01", base=tmp_path)
    (analysis.root / "run.json").write_text(
        json.dumps({"condition": "logs", "repeat": 2, "issued_day": 5, "world_hash": "h"})
    )
    assert check_run_record(analysis.root, "logs alone", 2, 5, {}, "h")[0] == 5
    for condition, repeat, day in (
        ("with experiments", 2, 5),
        ("logs alone", 1, 5),
        ("logs alone", 2, 6),
    ):
        with pytest.raises(ValueError, match="differs"):
            check_run_record(analysis.root, condition, repeat, day, {}, "h")


def test_missing_session_ledger_entry_with_matching_other_call_is_unavailable(
    tmp_path: Path,
) -> None:
    analysis = Analysis.create("w01", base=tmp_path / "answers")
    for n in (1, 2):
        (analysis.root / f"sessions/session-{n:02d}.json").write_text(
            json.dumps({"number": n, "run_dir": str(tmp_path / f"run{n}")})
        )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "run_dir": str(tmp_path / "run1"),
                "label": analysis.analysis_id + "/session-01",
                "backend": "codex",
                "model": "m",
                "seconds": 1,
                "ok": True,
                "usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1},
            }
        )
    )
    result = score_accounting(analysis, 100, ledger)
    assert result["missing"] == "session run directories absent from ledger"
    assert result["tiers"][0]["per_thousand_attempts"]["wall_seconds"] is None


def test_every_missing_healthy_example_is_a_native_fresh_request(tmp_path: Path) -> None:
    from fleetlab.oracle.fresh import fresh_jobs, read_requests

    observed = episode(901, "r01", [CAMERA], day=3)
    healthy = OracleTable(
        [observed],
        [
            OracleRow(
                episode_id=observed.episode_id,
                active_faults=fault_list_hash([CAMERA]),
                success=False,
            )
        ],
    )
    indicator = RetrainIndicator(
        failure_mode="m",
        episode_ids=[observed.episode_id],
        why_no_action_fixes_it="x",
        data_wanted="y",
    )
    report = score(inputs(as_run=healthy, diagnosis=diagnosis(retrain=[indicator])))
    request = next(n for n in report.needs_fresh_replay if n.episode_id == observed.episode_id)
    assert request.faults == [] and request.needed_for == ["claim 5: retraining indicator 0"]
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    native = read_requests([path])
    cited = [n for n in native if n.episode_id == observed.episode_id]
    jobs, refused = fresh_jobs({observed.episode_id: observed}, cited)
    assert not refused and len(jobs) == 1 and jobs[0].remove == [CAMERA.fault_id]
    assert jobs[0].identity.day == 3  # the observed window, not scoring


def test_old_wording_does_not_infer_a_drift_before_onset() -> None:
    drift = CAMERA.model_copy(
        update={
            "schedule": CAMERA.schedule.model_copy(
                update={"onset_day": 10, "drift": Drift(shape="linear", rate=1)}
            )
        }
    )
    identities = [episode(1, "r01", [], day=9), episode(2, "r02", [], day=9)]
    rows = [
        OracleRow(
            episode_id=i.episode_id, active_faults=fault_list_hash(()), success=i.robot_id != "r01"
        )
        for i in identities
    ]
    result = score_foresight(inputs(drift_faults=[drift], as_run=OracleTable(identities, rows)))
    assert result["warnings"][0]["old_wording_crossing_day"] is None


def test_headline_unvalued_share_and_pooled_unvalued_required_baseline() -> None:
    from fleetlab.scoring.decisions import headline

    report = comparison_report()
    report.claim3.budgets[0].baselines["classical statistics"].share = Metric.absent("zero ceiling")
    assert headline(report.claim3).ratio is None
    assert pool([report], expected_worlds=["w01"])["headline_ratios"][0]["ratio"] is None


@pytest.mark.parametrize("variant", ["malformed", "nodiag", "nofix", "nothing", "lost"])
def test_unreadable_submission_scores_at_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    from types import SimpleNamespace

    from test_scoring import TRUTH, current, fault_table, full_catalogue

    from fleetlab.scoring import __main__ as command

    analysis = Analysis.create("w01", base=tmp_path)
    (analysis.root / "run.json").write_text(
        json.dumps({"condition": "logs", "repeat": 1, "world_hash": "h"})
    )
    base = inputs()
    assert base.diagnosis is not None and base.fix_set is not None
    if variant == "malformed":
        (analysis.outbox / "diagnosis.json").write_text("{broken")
        (analysis.outbox / "fix_set.json").write_text('{"actions": []}')
    if variant == "nodiag":
        (analysis.outbox / "fix_set.json").write_text(base.fix_set.model_dump_json())
    if variant == "nofix":
        (analysis.outbox / "diagnosis.json").write_text(base.diagnosis.model_dump_json())
    if variant == "lost":
        (analysis.outbox / "diagnosis.json").write_text(base.diagnosis.model_dump_json())
        (analysis.outbox / "fix_set.json").write_text(base.fix_set.model_dump_json())
        (analysis.root / "run.json").write_text(
            json.dumps(
                {
                    "condition": "logs",
                    "repeat": 1,
                    "world_hash": "h",
                    "lost_twice": True,
                }
            )
        )
    world = SimpleNamespace(
        fault_table=fault_table(),
        world_id="w01",
        world_hash="h",
        excluded=frozenset(),
        identities=base.oracle.identities,
        additive_pairs=(),
        causes=(),
        spec=SimpleNamespace(
            seed=11,
            windows=base.fault_table.windows,
            fault_schedule=[],
            patterns=SimpleNamespace(confounders=[]),
        ),
        scoring=lambda: base.oracle.identities,
        truth=lambda: TRUTH,
        current=lambda: current,
        catalogue=full_catalogue,
        window=lambda _: base.oracle.identities,
    )
    monkeypatch.setattr(command, "load_world", lambda _: world)
    monkeypatch.setattr(command, "world_ready", lambda _: None)
    monkeypatch.setattr(command, "read_measured", lambda *_: base.measured_causes)
    monkeypatch.setattr(command, "all_runs", lambda *_: [])
    monkeypatch.setattr(
        command, "build_table", lambda *_: SimpleNamespace(table=base.oracle, problems=lambda: [])
    )
    # This answer-readability test isolates the oracle gate; round6 covers it end to end.
    monkeypatch.setattr(command, "replay_problem", lambda *_: None)
    report = command.score_world(tmp_path / "world", analysis.root)
    if variant != "nofix":
        assert report.claim1.recall.value == 0
        assert mark(report, "claim1.recall")["status"] == "missed"
        assert mark(report, "claim1.decoy_rejection")["status"] != "met"
    else:
        assert report.claim1.recall == score(base).claim1.recall
    if variant != "nodiag":
        assert all(b.agent.gain == Gain() for b in report.claim3.budgets)
        assert report.claim3.budgets[0].agent.share.value == 0
    else:
        assert report.claim3.budgets[0].agent.gain == score(base).claim3.budgets[0].agent.gain
    if variant in ("malformed", "nothing", "lost"):
        assert any("unreadable" in n for n in report.notes)


def test_scalar_cli_messages_do_not_hide_structured_refused_calls(tmp_path: Path) -> None:
    from fleetlab.integration.chain import refused_calls

    events = [
        "status",
        {"message": "progress"},
        {"message": {"content": "text"}},
        {
            "message": {
                "content": [
                    {"type": "tool_use", "name": "mcp__query", "id": "c", "input": {"sql": "q"}}
                ]
            }
        },
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c",
                        "content": "exceeds maximum allowed tokens",
                    }
                ]
            }
        },
    ]
    (tmp_path / "stdout.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    assert refused_calls(tmp_path) == [("query", json.dumps({"sql": "q"}, sort_keys=True))]


def test_model_baseline_repeat_cannot_be_relabelled(tmp_path: Path) -> None:
    (tmp_path / "fix_sets").mkdir()
    (tmp_path / "fix_sets/count-3.json").write_text(
        FixSet(actions=[], budget=Budget(kind="count", amount=3)).model_dump_json()
    )
    (tmp_path / "metadata.json").write_text(json.dumps({"repeat": 2}))
    with pytest.raises(ValueError, match="repeat differs"):
        baseline_answers({"no-data ranker": tmp_path})
    _, sets = baseline_answers({"no-data ranker repeat 2": tmp_path})
    assert "k = 3" in sets["no-data ranker repeat 2"]


def test_pool_retains_provisional_world_fault_caveats_without_changing_marks() -> None:
    from fleetlab.scoring.models import Metric

    first, second = comparison_report(), comparison_report("w02")
    first.claim1.recall = Metric.of(0.8)
    second.claim1.recall = Metric(value=1.0, provisional=True)
    second.claim1.precision = Metric(value=0.75, provisional=True)
    second.claim1.precision_without_vague = Metric(value=1.0, provisional=True)
    second.claim1.truth_disagreements = ["f01"]
    data = pool([first, second], expected_worlds=["w01", "w02"])
    recall = next(r for r in data["pooled"] if r["metric"] == "recall")
    assert recall["mean"] == 0.9 and recall["minimum"] == 0.8 and recall["maximum"] == 1.0
    assert recall["provisional"] is True and recall["truth_disagreements"] == {"w02": ["f01"]}
    for metric in ("precision", "older precision"):
        row = next(r for r in data["pooled"] if r["metric"] == metric)
        assert row["provisional"] is True and row["truth_disagreements"] == {"w02": ["f01"]}
    mark = next(r for r in data["pooled_pass_marks"] if r["metric"] == "claim1.recall")
    assert mark["status"] == "met" and mark["value"] == 0.9 and mark["threshold"] == 0.8
    assert mark["provisional"] is True and mark["truth_disagreements"] == {"w02": ["f01"]}
    assert next(p for p in second.pass_marks if p["metric"] == "claim1.recall")["provisional"]
    md = pool_markdown(data)
    assert "Provisional diagnosis metrics" in md and "faults f01" in md
    assert "met (provisional)" in md
    assert "Exact count-budget p95 (points)" in md
    assert "Cost-budget p95 confidence interval" in md


def test_required_values_follow_source_method_and_first_repeat(tmp_path: Path) -> None:
    from fleetlab.scoring.comparisons import BASELINE_METHODS, REQUIRED_BASELINES
    from fleetlab.scoring.decisions import headline
    from fleetlab.scoring.loading import baseline_records

    sources = {}
    export = tmp_path / "world/export"
    (export / "log").mkdir(parents=True)
    description = export / "log/description.json"
    description.write_text(json.dumps({"fleet": {"id": "w01"}}))
    (export / "log/episodes.parquet").write_bytes(b"constructed observed outcomes")
    from test_scoring_gen3 import certify_fixture

    certify_fixture(export)
    producer = tmp_path / "producer"
    manifest: dict[str, Any] = {
        "export": str(export),
        "methods": {},
        "model_repeats": [{"repeat": 1, "methods": {}}],
    }
    for name, methods in BASELINE_METHODS.items():
        if name not in REQUIRED_BASELINES:
            continue
        root = producer / methods[0]
        (root / "fix_sets").mkdir(parents=True)
        (root / "fix_sets/count-3.json").write_text(
            FixSet(actions=[], budget=Budget(kind="count", amount=3)).model_dump_json()
        )
        metadata: dict[str, Any] = {"method": methods[0]}
        if name in ("no-data ranker", "dramatic videos"):
            metadata["repeat"] = 1
        (root / "metadata.json").write_text(json.dumps(metadata))
        (root / "outbox").mkdir()
        (root / "outbox/fix_set.json").write_text('{"actions": []}')
        if name == "classical statistics":
            from fleetlab.scoring.provenance import file_hash

            (root / "evidence.json").write_text(
                json.dumps(
                    {
                        "method": methods[0],
                        "world_id": "w01",
                        "input_sha256": {
                            "log/description.json": file_hash(description),
                            "log/episodes.parquet": file_hash(export / "log/episodes.parquet"),
                        },
                    }
                )
            )
        else:
            relative = f"{methods[0]}/outbox/fix_set.json"
            manifest["methods"][methods[0]] = relative
            if name in ("no-data ranker", "dramatic videos"):
                manifest["model_repeats"][0]["methods"][methods[0]] = relative
        sources[name] = root
    (producer / "manifest.json").write_text(json.dumps(manifest))
    report = comparison_report()
    at = report.claim3.budgets[0]
    at.baseline_sources = baseline_records(sources, world_id="w01", world_root=tmp_path / "world")
    assert headline(report.claim3).ratio == pytest.approx(2.25)
    # Every way to reach a real producer still reads that producer's own record.
    video = sources["dramatic videos"]
    (video / "metadata.json").write_text(json.dumps({"method": "dramatic_videos", "repeat": 2}))
    for path in (video, video / "fix_sets", video / "fix_sets/count-3.json"):
        with pytest.raises(ValueError, match="repeat differs"):
            baseline_answers({"dramatic videos": path})
    with pytest.raises(ValueError, match="share a source"):
        baseline_answers(
            {
                "worst health metric": sources["worst health metric"],
                "no-data ranker": sources["worst health metric"],
            }
        )
    with pytest.raises(ValueError, match="method differs"):
        baseline_answers({"dramatic videos": sources["worst health metric"]})
    with pytest.raises(ValueError, match="share a source"):
        link = tmp_path / "alias"
        link.symlink_to(sources["worst health metric"], target_is_directory=True)
        baseline_records({"worst health metric": sources["worst health metric"], "alias": link})
    # A copy without metadata remains visible, but cannot decide a comparison.
    copied = tmp_path / "copy"
    copied.mkdir()
    (copied / "count-3.json").write_bytes((video / "fix_sets/count-3.json").read_bytes())
    at.baseline_sources["dramatic videos"] = baseline_records({"dramatic videos": copied})[
        "dramatic videos"
    ]
    assert baseline_answers({"dramatic videos": copied})[1]["dramatic videos"]["k = 3"]
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None
    at.baseline_sources["dramatic videos"] = {
        "source": str(video),
        "record": {"method": "dramatic_videos", "repeat": 2},
    }
    assert headline(report.claim3).ratio is None
    at.baseline_sources["dramatic videos"] = {
        "source": str(video),
        "record": {"method": "dramatic_videos"},
    }
    assert headline(report.claim3).ratio is None


def test_core_comparison_requires_method_record_and_unshared_source() -> None:
    from fleetlab.scoring.decisions import headline

    report = comparison_report()
    at = report.claim3.budgets[0]
    at.baseline_sources["classical statistics"]["record"].pop("method")
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None
    at.baseline_sources["classical statistics"]["record"]["method"] = "classical-binomial-v2"
    at.baseline_sources["classical statistics"]["source"] = at.baseline_sources[
        "worst health metric"
    ]["source"]
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None


def test_absent_baseline_is_not_described_as_unvalued() -> None:
    report = score(inputs())
    missing = mark(report, "claim3.margin_points")["missing"]
    assert "dramatic videos (absent)" in missing
    assert "(unvalued)" not in missing


def test_pool_requires_expected_roster_and_refuses_unverified_labels(tmp_path: Path) -> None:
    from fleetlab.scoring import __main__ as command

    report = comparison_report()
    with pytest.raises(TypeError, match="roster"):
        score_pool([report])  # type: ignore[call-arg]
    with pytest.raises(SystemExit) as exc:
        command.main(["--reports", "not-read.json", "--out", str(tmp_path / "out")])
    assert exc.value.code == 2 and not (tmp_path / "out").exists()
    result = pool([report], expected_worlds=["w01", "w02"])
    assert result["worlds"] == ["w01", "w02"]
    assert result["per_world"]["w02"] == []
    assert all(p["status"] == "unavailable" for p in result["pooled_pass_marks"])
    assert all(
        p["worlds_expected"] == 2 and "w02" in p["missing_worlds"]
        for p in result["pooled_pass_marks"]
    )
    ratio = result["headline_ratios"][0]
    assert ratio["ratio"] is None and ratio["best_on_average"]["ratio"] is None
    assert ratio["pooled_chance"]["p95_high"] is None and ratio["missing_worlds"] == ["w02"]
    assert "w02" in ratio["verdict"]
    report.notes.append("caller labels are unverified")
    with pytest.raises(ValueError, match="unverified"):
        pool([report], expected_worlds=["w01"])
    report.notes.clear()
    report.metadata.pop("run_labels_verified")
    with pytest.raises(ValueError, match="unverified"):
        pool([report], expected_worlds=["w01"])
    from fleetlab.scoring.loading import check_run_record

    analysis = Analysis.create("w01", base=tmp_path)
    with pytest.raises(ValueError, match="unverified"):
        check_run_record(analysis.root, "logs alone", 1, 5, {}, "h")


def test_pooled_mark_misses_when_mean_misses_despite_one_passing_world() -> None:
    first, second = comparison_report(), comparison_report("w02")
    first.claim1.recall, second.claim1.recall = Metric.of(0.1), Metric.of(0.9)
    result = pool([first, second], expected_worlds=["w01", "w02"])
    row = next(p for p in result["pooled_pass_marks"] if p["metric"] == "claim1.recall")
    assert row["value"] == 0.5 and row["worlds_met"] == 1 and row["status"] == "missed"


@pytest.mark.parametrize("defect", ["no_histogram", "no_total", "no_unknown", "wrong_total"])
def test_incomplete_exact_set_accounting_cannot_decide_world_or_pool(defect: str) -> None:
    from fleetlab.scoring.decisions import headline
    from fleetlab.scoring.pooled_chance import pooled_floor

    report = comparison_report()
    at = report.claim3.budgets[0]
    assert at.exact_chance is not None
    if defect == "wrong_total":
        at.exact_chance["sets"] += 1
    else:
        at.exact_chance.pop(
            {"no_histogram": "histogram", "no_total": "sets", "no_unknown": "unvalued_sets"}[defect]
        )
    row = mark(report, "claim3.above_chance")
    assert row["status"] == "unavailable" and at.chance_command in row["missing"]
    assert headline(report.claim3).ratio is None
    assert pooled_floor({"w01": at})["p95_high"] is None


def test_pooled_floor_covers_hand_counted_quantile_and_rounding() -> None:
    from fleetlab.scoring.models import Achievable
    from fleetlab.scoring.pooled_chance import pooled_floor

    at = comparison_report().claim3.budgets[0]
    at.achievable = Achievable(gain=Gain.of(3, 0), points=3)
    at.exact_chance = {"histogram": [[0, 90], [1, 10]], "sets": 100, "unvalued_sets": 0}
    result = pooled_floor({"w01": at})
    # Ordered positions 94 and 95 of this 100-set population are both 1/3.
    # Position 89 is zero: its 90th percentile is only .1/3.
    true_p95 = 1 / 3
    assert result["p95_low"] <= true_p95 <= result["p95_high"]
    assert result["p95_high"] - result["p95_low"] <= 0.000251
    at.exact_chance = {
        "histogram": [[0, 90]],
        "sets": 100,
        "unvalued_sets": 10,
        "net_low": -3,
        "net_high": 3,
    }
    unknown = pooled_floor({"w01": at})
    # All ten unknown sets can occupy the top tail; dropping them gives zero.
    assert unknown["p95_low"] == 0 and unknown["p95_high"] >= 1


def test_pooled_ratio_uses_upper_endpoint_of_unvalued_population() -> None:
    report = comparison_report()
    at = report.claim3.budgets[0]
    at.exact_chance = {
        "histogram": [[0, 90]],
        "sets": 100,
        "unvalued_sets": 10,
        "net_low": 0,
        "net_high": 80,
        "p95_points": None,
        "bounds": {"p95_points": [0, 80]},
    }
    result = pool([report], expected_worlds=["w01"])["headline_ratios"][0]
    assert result["pooled_chance"]["p95_low"] == 0
    assert result["pooled_chance"]["p95_high"] >= 0.8
    assert result["baseline_mean_share"] == 0.4
    assert result["ratio"] is None and result["best_on_average"]["ratio"] is None


def test_roster_identity_hash_and_world_hashes_are_reported_and_enforced(tmp_path: Path) -> None:
    import hashlib

    from fleetlab.scoring.roster import read_roster

    report = comparison_report()
    digest = "a" * 64
    report.metadata["world_hash"] = digest
    roster = tmp_path / "frozen-roster.json"
    raw = json.dumps(
        {
            "worlds": [
                {"world_id": "w01", "world_hash": digest},
                {"world_id": "w02", "world_hash": "b" * 64},
            ]
        }
    ).encode()
    roster.write_bytes(raw)
    result = score_pool([report], roster=roster)
    assert result["roster"]["path"] == str(roster)
    assert result["roster"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert str(roster) in pool_markdown(result) and hashlib.sha256(
        raw
    ).hexdigest() in pool_markdown(result)
    assert all(mark["status"] == "unavailable" for mark in result["pooled_pass_marks"])
    report.metadata["world_hash"] = "c" * 64
    with pytest.raises(ValueError, match="world hash differs"):
        score_pool([report], roster=roster)
    with pytest.raises(ValueError, match="roster file"):
        score_pool([report], roster=["w01"])  # type: ignore[arg-type]
    from fleetlab.scoring import __main__ as command

    with pytest.raises(SystemExit) as exc:
        command.main(
            [
                "--reports",
                "not-read.json",
                "--expected-worlds",
                "w01",
                "--out",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 2 and not (tmp_path / "out").exists()
    roster.write_text(json.dumps({"worlds": [{"world_id": "w01"}]}))
    with pytest.raises(ValueError, match="omitted world_hash"):
        read_roster(roster)


def test_producer_manifest_binds_world_repeat_and_output_directory(tmp_path: Path) -> None:
    from fleetlab.scoring.loading import baseline_records

    export = tmp_path / "world/export"
    (export / "log").mkdir(parents=True)
    description = export / "log/description.json"
    description.write_text(json.dumps({"fleet": {"id": "w01"}}))
    from test_scoring_gen3 import certify_fixture

    certify_fixture(export)
    producer = tmp_path / "producer"
    method = producer / "no_data"
    (method / "outbox").mkdir(parents=True)
    (method / "outbox/fix_set.json").write_text('{"actions": []}')
    (method / "fix_sets").mkdir()
    (method / "fix_sets/count-3.json").write_text(
        FixSet(actions=[], budget=Budget(kind="count", amount=3)).model_dump_json()
    )
    manifest: dict[str, Any] = {
        "export": str(export),
        "methods": {"no_data": "no_data/outbox/fix_set.json"},
        "model_repeats": [{"repeat": 1, "methods": {"no_data": "no_data/outbox/fix_set.json"}}],
    }
    record = producer / "manifest.json"
    record.write_text(json.dumps(manifest))
    kwargs: dict[str, Any] = {"world_id": "w01", "world_root": tmp_path / "world"}
    bound = baseline_records({"no-data ranker": method}, **kwargs)["no-data ranker"]
    assert bound["binding"] and bound["record"] == {
        "world_id": "w01",
        "method": "no_data",
        "repeat": 1,
    }
    assert bound["producer_record"] == str(record)
    assert "fix_sets/count-3.json" in bound["scored_file_sha256"]
    # A record for this directory cannot authorize an arbitrary file beside it.
    unrelated = method / "other.json"
    unrelated.write_text('{"actions": []}')
    with pytest.raises(ValueError, match="not a producer-recorded output"):
        baseline_records({"no-data ranker": unrelated}, **kwargs)
    outside = tmp_path / "outside.json"
    outside.write_bytes((method / "fix_sets/count-3.json").read_bytes())
    link = method / "fix_sets/cost-4.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="escape the producer directory"):
        baseline_records({"no-data ranker": method}, **kwargs)
    link.unlink()  # constructed fixture only
    with pytest.raises(ValueError, match="another world"):
        baseline_records({"no-data ranker": method}, world_id="w02", world_root=tmp_path / "world")
    manifest["model_repeats"][0]["repeat"] = 2
    record.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="repeat differs"):
        baseline_records({"no-data ranker": method}, **kwargs)
    manifest["model_repeats"][0]["repeat"] = 1
    manifest["methods"]["no_data"] = "elsewhere/outbox/fix_set.json"
    manifest["model_repeats"][0]["methods"]["no_data"] = "elsewhere/outbox/fix_set.json"
    record.write_text(json.dumps(manifest))
    assert (
        "binding_missing"
        in baseline_records({"no-data ranker": method}, **kwargs)["no-data ranker"]
    )
    (method / "metadata.json").write_text(json.dumps({"method": "no_data", "repeat": 1}))
    report = comparison_report()
    report.claim3.budgets[0].baseline_sources["no-data ranker"] = baseline_records(
        {"no-data ranker": method}, **kwargs
    )["no-data ranker"]
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"


def test_classical_producer_record_checks_world_and_recorded_input_hashes(tmp_path: Path) -> None:
    from fleetlab.scoring.loading import baseline_records
    from fleetlab.scoring.provenance import file_hash

    export = tmp_path / "world/export"
    (export / "log").mkdir(parents=True)
    description = export / "log/description.json"
    description.write_text(json.dumps({"fleet": {"id": "w01"}}))
    (export / "log/episodes.parquet").write_bytes(b"constructed observed outcomes")
    from test_scoring_gen3 import certify_fixture

    certify_fixture(export)
    method = tmp_path / "classical"
    (method / "outbox").mkdir(parents=True)
    (method / "outbox/fix_set.json").write_text('{"actions": []}')
    evidence = method / "evidence.json"
    raw: dict[str, Any] = {
        "method": "classical-binomial-v2",
        "world_id": "w01",
        "input_sha256": {
            "log/description.json": file_hash(description),
            "log/episodes.parquet": file_hash(export / "log/episodes.parquet"),
        },
    }
    evidence.write_text(json.dumps(raw))
    bound = baseline_records(
        {"classical statistics": method}, world_id="w01", world_root=tmp_path / "world"
    )["classical statistics"]
    assert bound["binding"] and bound["producer_record"] == str(evidence)
    with pytest.raises(ValueError, match="another world"):
        baseline_records(
            {"classical statistics": method}, world_id="w02", world_root=tmp_path / "world"
        )
    raw["input_sha256"]["log/description.json"] = "0" * 64
    evidence.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="input hashes"):
        baseline_records(
            {"classical statistics": method}, world_id="w01", world_root=tmp_path / "world"
        )
