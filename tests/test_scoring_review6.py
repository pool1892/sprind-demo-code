"""Remaining surviving review mutations, without model calls or simulation."""

import json
from pathlib import Path

import pytest
from test_scoring import CAMERA, cause, inputs
from test_scoring_gen3 import classical_source, opaque_export
from test_scoring_round6 import RUNTIME
from test_scoring_stage4 import comparison_report, diagnosis, mark

from fleetlab.contracts.faults import Direction, Target
from fleetlab.contracts.oracle import Gain
from fleetlab.scoring import score, validation
from fleetlab.scoring.decisions import headline
from fleetlab.scoring.loading import baseline_records
from fleetlab.scoring.models import Metric


def test_classical_public_name_must_match_selected_export(tmp_path: Path) -> None:
    export = opaque_export(tmp_path / "world", "w01", b"own")
    source = classical_source(tmp_path / "producer", export)
    raw = json.loads((source / "evidence.json").read_text())
    raw["world_id"] = "another-public-name"
    (source / "evidence.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="another world"):
        baseline_records({"classical statistics": source}, world_id="w01", world_root=export.parent)


def test_measured_zero_effect_does_not_borrow_curve_vagueness() -> None:
    from fleetlab.oracle.causes import MeasuredCauses

    base = inputs()
    assert base.measured_causes is not None
    raw = base.measured_causes.model_dump()
    item = next(i for i in raw["faults"] if i["item"] == CAMERA.fault_id)
    item.update(measured_class="small", qualified_by=[])
    for window in item["windows"].values():
        window.update(points=0, net=0, rescued=0, broken=0)
    raw["causes"] = [u for u in raw["causes"] if CAMERA.fault_id not in u["fault_ids"]]
    raw["cause_count"] = len(raw["causes"])
    report = score(
        inputs(
            measured_causes=MeasuredCauses.model_validate(raw),
            diagnosis=diagnosis(
                causes=[cause(Target(scope=CAMERA.target.scope), Direction.degraded, 0.9)]
            ),
        )
    )
    assert report.claim1.causes[0].verdict == "no fault"
    assert report.claim1_curve is not None
    assert report.claim1_curve.causes[0].verdict == "vague"


@pytest.mark.parametrize("defect", ["incomplete", "problems"])
def test_constructed_ceiling_requires_complete_measured_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    from types import SimpleNamespace

    base = inputs()
    assert base.measured_causes is not None
    world = SimpleNamespace(
        world_id="w01",
        world_hash="h",
        fault_table=base.fault_table,
        spec=SimpleNamespace(windows=base.fault_table.windows),
    )
    measured = base.measured_causes.model_copy(
        update={
            "complete": defect != "incomplete",
            "problems": ["gap"] if defect == "problems" else [],
        }
    )
    monkeypatch.setattr(validation, "load_world", lambda _: world)
    monkeypatch.setattr(validation, "read_measured", lambda *_: measured)
    with pytest.raises(ValueError, match="complete oracle measured causes"):
        validation.constructed_answer(tmp_path / "world", tmp_path / "answers", empty=True)


def test_headline_shows_strongest_logs_opponent() -> None:
    report = comparison_report()
    at = report.claim3.budgets[0]
    for name, net in zip(at.baselines, (40, 30, 20, 50), strict=True):
        value = at.baselines[name]
        value.gain = Gain.of(net, 0)
        value.points = net
        value.share = Metric.of(net / 100)
    result = headline(report.claim3)
    assert result.opponents["best logs-only"] == {
        "name": "classical statistics",
        "points": 50,
        "share": 0.5,
    }
    assert result.best_baseline == "classical statistics"


def test_runtime_attempt_receipts_cannot_conflict(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps(RUNTIME["lost"]))
    (tmp_path / "evidence.json").write_text(json.dumps(RUNTIME["recovered"]))
    with pytest.raises(ValueError, match="conflicting baseline attempts"):
        baseline_records({"no-data ranker": tmp_path})


def test_exact_margin_uses_net_despite_float_subtraction() -> None:
    report = comparison_report()
    at = report.claim3.budgets[0]
    report.scoring_episodes = 300
    at.agent.gain = Gain.of(16, 0)
    at.agent.points = 100.0 * 16 / 300
    for value in at.baselines.values():
        value.gain = Gain.of(7, 0)
        value.points = 100.0 * 7 / 300
    assert at.agent.points - 100.0 * 7 / 300 < 3
    result = mark(report, "claim3.margin_points")
    assert result["value"] == 3 and result["status"] == "met"
