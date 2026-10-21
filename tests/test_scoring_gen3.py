"""Constructed third-generation bindings and measured memberships, no simulation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_scoring import DECOY, cause, inputs
from test_scoring_stage4 import comparison_report, diagnosis, mark, pool

from fleetlab.baselines._seed import _export_digest, default_seed
from fleetlab.contracts.faults import Direction
from fleetlab.contracts.oracle import Gain
from fleetlab.log.leaks import audit_identifier, export_digest
from fleetlab.oracle.causes import MeasuredCauses
from fleetlab.scoring import report_markdown, score
from fleetlab.scoring.comparisons import EXPERIMENTAL_BASELINE
from fleetlab.scoring.decisions import headline
from fleetlab.scoring.loading import baseline_answers, baseline_records
from fleetlab.scoring.measured import read_measured
from fleetlab.scoring.models import FixSetValue, Metric
from fleetlab.scoring.provenance import file_hash


def certify_fixture(export: Path, world_id: str = "w01") -> None:
    """A labelled synthetic readiness receipt, not a real leak audit."""
    root = export.parent
    (root / "runner.json").write_text(json.dumps({"world_id": world_id}))
    receipts = root / "export-checks"
    receipts.mkdir(exist_ok=True)
    (receipts / "check-fixture.json").write_text(
        json.dumps(
            {
                "clean": True,
                "export": str(export.resolve()),
                "export_digest": export_digest(export),
                "check_id": audit_identifier(()),
                "terms": [],
                "terms_digest": hashlib.sha256(json.dumps([]).encode()).hexdigest(),
                "synthetic": True,
            }
        )
    )


def opaque_export(root: Path, world_id: str, outcomes: bytes) -> Path:
    export = root / "export"
    (export / "log").mkdir(parents=True)
    (export / "log/description.json").write_text(json.dumps({"fleet": {"id": "w01"}}))
    (export / "log/episodes.parquet").write_bytes(outcomes)
    (export / "empty").mkdir()
    (root / "files").mkdir()
    (root / "files/hash").write_text(hashlib.sha256(outcomes).hexdigest())
    certify_fixture(export, world_id)
    return export


def simple_source(root: Path, export: Path) -> Path:
    source = root / "worst_health"
    (source / "outbox").mkdir(parents=True)
    (source / "outbox/fix_set.json").write_text('{"actions": []}')
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "export": str(export),
                "methods": {"worst_health": "worst_health/outbox/fix_set.json"},
            }
        )
    )
    return source


def classical_source(root: Path, export: Path, experimental: bool = False) -> Path:
    (root / "outbox").mkdir(parents=True)
    (root / "outbox/fix_set.json").write_text('{"actions": []}')
    hashes = {str(p.relative_to(export)): file_hash(p) for p in export.rglob("*") if p.is_file()}
    raw: dict[str, Any] = {
        "method": "classical-binomial-v2",
        "world_id": "w01",
        "input_sha256": hashes,
    }
    if experimental:
        raw.update(
            condition="experiments",
            requests=1,
            completed_requests=1,
            request_sha256="a" * 64,
            result_sha256="b" * 64,
            experiment_input_sha256=hashes,
        )
    (root / "evidence.json").write_text(json.dumps(raw))
    return root


@pytest.mark.parametrize("kind", ["simple", "classical", "experiments"])
def test_opaque_name_binds_genuine_sources_but_not_another_world(tmp_path: Path, kind: str) -> None:
    export = opaque_export(tmp_path / "w47", "w47", b"own observed outcomes")
    other = opaque_export(tmp_path / "w43", "w43", b"other observed outcomes")
    if kind == "simple":
        name = "worst health metric"
        own = simple_source(tmp_path / "own-producer", export)
        foreign = simple_source(tmp_path / "foreign-producer", other)
    else:
        experimental = kind == "experiments"
        name = EXPERIMENTAL_BASELINE if experimental else "classical statistics"
        own = classical_source(tmp_path / "own-producer", export, experimental)
        foreign = classical_source(tmp_path / "foreign-producer", other, experimental)
    kwargs: dict[str, Any] = {"world_id": "w47", "world_root": export.parent}
    bound = baseline_records({name: own}, **kwargs)[name]
    assert bound["record"]["world_id"] == "w47"
    assert bound["public_world_id"] == "w01"
    assert bound["export_digest"] == export_digest(export)
    assert bound["binding"]
    with pytest.raises(ValueError, match=r"another world|input hashes"):
        baseline_records({name: foreign}, **kwargs)


def test_classical_description_only_cannot_bind_a_shared_public_name(tmp_path: Path) -> None:
    export = opaque_export(tmp_path / "w47", "w47", b"own outcomes")
    source = classical_source(tmp_path / "producer", export)
    raw = json.loads((source / "evidence.json").read_text())
    raw["input_sha256"].pop("log/episodes.parquet")
    (source / "evidence.json").write_text(json.dumps(raw))
    record = baseline_records(
        {"classical statistics": source}, world_id="w47", world_root=export.parent
    )["classical statistics"]
    assert "episodes" in record["binding_missing"] and "binding" not in record


def test_seed_binds_metadata_and_bytes_without_chance_values(tmp_path: Path) -> None:
    export = opaque_export(tmp_path / "w47", "w47", b"opaque seed test")
    assert _export_digest(export) == export_digest(export)
    oracle = export.parent / "oracle"
    oracle.mkdir()
    chance = oracle / "oracle.json"
    raw = {
        "world_id": "w47",
        "world_hash": (export.parent / "files/hash").read_text(),
        "seed": 197,
        "chance": "intentionally not a chance table",
    }
    chance.write_text(json.dumps(raw))
    assert default_seed(export) == 197
    raw["world_id"] = "w43"
    chance.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different world"):
        default_seed(export)
    raw["world_id"] = "w47"
    raw["world_hash"] = "b" * 64
    chance.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different world"):
        default_seed(export)
    raw["world_hash"] = (export.parent / "files/hash").read_text()
    chance.write_text(json.dumps(raw))
    (export / "log/episodes.parquet").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="readiness receipt"):
        default_seed(export)


def test_measured_decoy_promotion_is_a_hit_and_curves_do_not_decide() -> None:
    base = inputs()
    assert base.measured_causes is not None
    raw = base.measured_causes.model_dump()
    item = next(f for f in raw["faults"] if f["item"] == DECOY.fault_id)
    item.update(measured_class="cause", qualified_by=["scoring"])
    item["windows"]["scoring"].update(points=2, net=2, rescued=2)
    raw["causes"].append(
        {"unit": DECOY.fault_id, "fault_ids": [DECOY.fault_id], "qualified_by": ["scoring"]}
    )
    raw["cause_count"] += 1
    raw["no_decoy_left"] = True
    answer = diagnosis(causes=[cause(DECOY.target, Direction.changed, 0.9)])
    report = score(inputs(diagnosis=answer, measured_causes=MeasuredCauses.model_validate(raw)))
    assert report.claim1.detectable == 3 and report.claim1.recall.value == pytest.approx(1 / 3)
    assert report.claim1.hits == 1 and report.claim1.false_positives == 0
    assert report.claim1.precision.value == 1
    assert report.claim1.decoy_rejection.value is None
    assert mark(report, "claim1.decoy_rejection")["status"] == "unavailable"
    assert report.claim1_curve is not None and report.claim1_curve.false_positives == 1
    assert report.claim1_curve.recall.value == 0
    text = report_markdown(report)
    assert "Measured (decides marks)" in text and "Curves (descriptive only)" in text
    # The oracle's below-threshold classification stays a decoy, even when named.
    item.update(measured_class="decoy", qualified_by=[])
    item["windows"]["scoring"].update(points=1, net=1, rescued=1)
    raw["causes"].pop()
    raw["cause_count"] -= 1
    raw["no_decoy_left"] = False
    low = score(inputs(diagnosis=answer, measured_causes=MeasuredCauses.model_validate(raw)))
    assert low.claim1.hits == 0 and low.claim1.false_positives == 1
    assert low.claim1.decoy_rejection.value == 0


@pytest.mark.parametrize("missing", ["absent", "incomplete", "problems"])
def test_measured_membership_unavailable_never_falls_back_to_curves(missing: str) -> None:
    base = inputs()
    assert base.measured_causes is not None
    record = (
        None
        if missing == "absent"
        else base.measured_causes.model_copy(
            update={
                "complete": missing != "incomplete",
                "problems": ["gap"] if missing == "problems" else [],
            }
        )
    )
    command = "produce it with: uv run --locked python -m fleetlab.oracle causes WORLD_ROOT"
    report = score(inputs(measured_causes=record, measured_missing=command))
    for metric in (report.claim1.recall, report.claim1.precision, report.claim1.decoy_rejection):
        assert metric.value is None and command in str(metric.missing)
    assert report.claim1_curve is not None and report.claim1_curve.recall.value is not None
    assert mark(report, "claim1.recall")["status"] == "unavailable"
    assert command in report_markdown(report)


def test_measured_record_identity_rule_and_singleton_accounting(tmp_path: Path) -> None:
    base = inputs()
    assert base.measured_causes is not None
    path = tmp_path / "measured_causes.json"
    assert read_measured(path, "w01", None) is None
    path.write_text(base.measured_causes.model_dump_json())
    assert read_measured(path, "w01", None) == base.measured_causes
    for world_id, world_hash in (("w43", None), ("w01", "a" * 64)):
        with pytest.raises(ValueError, match="another world"):
            read_measured(path, world_id, world_hash)
    raw = base.measured_causes.model_dump()
    raw["causes"].pop()
    raw["cause_count"] -= 1
    with pytest.raises(ValueError, match="singleton list"):
        score(inputs(measured_causes=MeasuredCauses.model_validate(raw)))
    raw = base.measured_causes.model_dump()
    raw["threshold_points"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="frozen design"):
        read_measured(path, "w01", None)


def experiments_report(world: str = "w01", extra: float | None = None) -> Any:
    report = comparison_report(world)
    report.metadata["condition"] = "with experiments"
    for at in report.claim3.budgets:
        at.condition = "with experiments"
    at = report.claim3.budgets[0]
    if extra is not None:
        at.baselines[EXPERIMENTAL_BASELINE] = FixSetValue(
            gain=Gain.of(int(extra), 0), points=extra, share=Metric.of(extra / 100)
        )
        at.baseline_sources[EXPERIMENTAL_BASELINE] = {
            "source": "/constructed/experimental-classical",
            "binding": "constructed evidence",
            "record": {"method": "classical-binomial-v2", "world_id": world},
        }
    report.headline = headline(report.claim3)
    return report


def test_experiments_requires_its_own_baseline_in_world_and_pool() -> None:
    missing = experiments_report()
    assert mark(missing, "claim3.margin_points")["status"] == "unavailable"
    assert EXPERIMENTAL_BASELINE in mark(missing, "claim3.margin_points")["missing"]
    assert missing.headline.ratio is None
    complete = experiments_report("w43", 95)
    pooled = pool([missing, complete], expected_worlds=["w01", "w43"])
    assert pooled["headline_ratios"][0]["ratio"] is None
    assert (
        next(p for p in pooled["pooled_pass_marks"] if p["metric"] == "claim3.margin_points")[
            "status"
        ]
        == "unavailable"
    )
    assert comparison_report().headline.ratio == 2.25  # logs still use the four logs opponents


@pytest.mark.parametrize(
    "extra, best, margin, ratio",
    [
        (95, EXPERIMENTAL_BASELINE, -5, 90 / 95),
        (20, "no-data ranker", 50, 2.25),
    ],
)
def test_experiments_compares_the_stronger_opponent_and_shows_both(
    extra: float, best: str, margin: float, ratio: float
) -> None:
    report = experiments_report(extra=extra)
    assert report.headline.best_baseline == best
    assert report.headline.ratio == pytest.approx(ratio)
    assert mark(report, "claim3.margin_points")["value"] == margin
    assert report.headline.opponents["best logs-only"]["points"] == 40
    assert report.headline.opponents[EXPERIMENTAL_BASELINE]["points"] == extra
    assert "Comparison opponent" in report_markdown(report)
    pooled = pool([report], expected_worlds=["w01"])
    assert pooled["headline_ratios"][0]["ratio"] == pytest.approx(ratio)


def test_experimental_label_cannot_accept_logs_evidence_and_reverse(tmp_path: Path) -> None:
    export = opaque_export(tmp_path / "w47", "w47", b"own")
    logs = classical_source(tmp_path / "logs", export)
    exp = classical_source(tmp_path / "experiments", export, True)
    kwargs: dict[str, Any] = {"world_id": "w47", "world_root": export.parent}
    bound = baseline_records({EXPERIMENTAL_BASELINE: logs}, **kwargs)[EXPERIMENTAL_BASELINE]
    assert "binding_missing" in bound and "binding" not in bound
    with pytest.raises(ValueError, match="logs-only"):
        baseline_records({"classical statistics": exp}, **kwargs)


def test_two_lost_baseline_starts_override_stale_answer_and_cannot_decide_comparisons(
    tmp_path: Path,
) -> None:
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox/fix_set.json").write_text('{"actions": []}')
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "method": "no_data",
                "repeat": 1,
                "attempts": json.loads(
                    (Path(__file__).parent / "data/scoring/runtime-records.json").read_text()
                )["lost"]["attempts"],
            }
        )
    )
    rankings, sets = baseline_answers({"no-data ranker": tmp_path})
    assert rankings == {} and sets == {}
    record = baseline_records({"no-data ranker": tmp_path})["no-data ranker"]
    assert record["lost_call"]
    report = comparison_report()
    report.claim3.budgets[0].baseline_sources["no-data ranker"] = record
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None
    assert "both model call starts lost" in headline(report.claim3).verdict
