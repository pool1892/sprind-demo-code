"""Round-5 breaks: real runtime record shapes and constructed scoring inputs."""

import json
import shlex
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_scoring import CAMERA, DECOY, cause, inputs
from test_scoring_gen3 import certify_fixture, classical_source, opaque_export, simple_source
from test_scoring_stage4 import comparison_report, diagnosis, mark, pool

from fleetlab.agents.sessions import Analysis
from fleetlab.baselines._seed import default_seed
from fleetlab.contracts.faults import Direction
from fleetlab.contracts.oracle import Gain
from fleetlab.oracle.causes import MeasuredCauses
from fleetlab.oracle.world import Cause as OracleCause
from fleetlab.scoring import __main__ as command
from fleetlab.scoring import report_markdown, score
from fleetlab.scoring.comparisons import EXPERIMENTAL_BASELINE
from fleetlab.scoring.decisions import headline
from fleetlab.scoring.loading import baseline_answers, baseline_records
from fleetlab.scoring.lost import analysis_lost, exhausted_call
from fleetlab.scoring.measured import read_measured
from fleetlab.scoring.splits import score_splits

RUNTIME = json.loads((Path(__file__).parent / "data/scoring/runtime-records.json").read_text())


@pytest.mark.parametrize(
    "record, expected", [("lost", True), ("single", False), ("recovered", False)]
)
def test_runtime_attempt_records(record: str, expected: bool) -> None:
    assert exhausted_call(RUNTIME[record]) is expected
    # A single lost start is not exhaustion either.
    assert not exhausted_call({"attempts": RUNTIME["lost"]["attempts"][:1]})


def test_runtime_session_and_run_override_stale_outbox(tmp_path: Path) -> None:
    analysis = Analysis.create("w47", base=tmp_path)
    (analysis.outbox / "fix_set.json").write_text('{"actions": []}')
    (analysis.root / "sessions/session-01.json").write_text(json.dumps(RUNTIME["session"]))
    assert analysis_lost(analysis.root)
    (analysis.root / "sessions/session-01.json").unlink()
    (analysis.root / "run.json").write_text(json.dumps(RUNTIME["analysis_run"]))
    assert analysis_lost(analysis.root)


@pytest.mark.parametrize("form", ["directory", "file"])
def test_lost_manifest_without_method_directory_is_missing(tmp_path: Path, form: str) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"methods": {}, "missing": {"dramatic_videos": RUNTIME["lost"]}})
    )
    path = tmp_path / "dramatic_videos"
    if form == "file":
        path /= "outbox/fix_set.json"
    paths = {"dramatic videos": path}
    assert baseline_answers(paths) == ({}, {})
    receipt = baseline_records(paths)["dramatic videos"]
    assert receipt["lost_call"] and receipt["attempts"] == RUNTIME["lost"]["attempts"]
    report = comparison_report()
    report.claim3.budgets[0].baseline_sources["dramatic videos"] = receipt
    assert mark(report, "claim3.margin_points")["status"] == "unavailable"
    assert headline(report.claim3).ratio is None


def test_lost_metadata_and_decision_layer_discard_stale_value(tmp_path: Path) -> None:
    (tmp_path / "outbox").mkdir()
    (tmp_path / "outbox/fix_set.json").write_text('{"actions": []}')
    (tmp_path / "metadata.json").write_text(json.dumps(RUNTIME["lost"]))
    receipt = baseline_records({"no-data ranker": tmp_path})["no-data ranker"]
    assert baseline_answers({"no-data ranker": tmp_path}) == ({}, {})
    report = score(
        inputs(
            baseline_rankings={"no-data ranker": []}, baseline_sources={"no-data ranker": receipt}
        )
    )
    value = report.claim3.budgets[0].baselines["no-data ranker"]
    assert value.gain is None and value.points is None and "lost" in str(value.missing)


def authoritative_source(root: Path) -> Path:
    (root / "outbox").mkdir(parents=True)
    (root / "fix_sets").mkdir()
    (root / "metadata.json").write_text(json.dumps({"explicit_fix_sets_authoritative": True}))
    ranked = inputs().fix_set
    assert ranked is not None
    (root / "outbox/fix_set.json").write_text(ranked.model_dump_json())
    for k in (3, 6):
        (root / f"fix_sets/count-{k}.json").write_text(
            json.dumps({"actions": [], "budget": {"kind": "count", "amount": k}})
        )
    return root


def test_authoritative_producer_refuses_ranked_path_and_normalizes_budget_paths(
    tmp_path: Path,
) -> None:
    root = authoritative_source(tmp_path / "video")
    expected = baseline_answers({"dramatic videos": root})
    for path in (root / "fix_sets", root / "fix_sets/count-3.json", root / "fix_sets/count-6.json"):
        assert baseline_answers({"dramatic videos": path}) == expected
    for path in (root / "outbox", root / "outbox/fix_set.json"):
        with pytest.raises(ValueError, match="authoritative"):
            baseline_answers({"dramatic videos": path})
    shutil.rmtree(root / "fix_sets")
    with pytest.raises(ValueError, match="authoritative"):
        baseline_answers({"dramatic videos": root})


def fake_scoring_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Analysis, Any]:
    """Real scorer CLI/loading/valuation, mocked oracle/world I/O; no simulation."""
    base = inputs()
    assert base.measured_causes is not None
    assert base.diagnosis is not None and base.fix_set is not None
    export = opaque_export(tmp_path / "world", "w01", b"scored outcomes")
    world_hash = (export.parent / "files/hash").read_text()
    world = SimpleNamespace(
        fault_table=base.fault_table,
        world_id="w01",
        world_hash=world_hash,
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
        truth=lambda: base.truth,
        current=lambda: base.current,
        catalogue=lambda: [],
        window=lambda _: base.oracle.identities,
    )
    monkeypatch.setattr(command, "load_world", lambda _: world)
    monkeypatch.setattr(command, "world_ready", lambda _: None)
    monkeypatch.setattr(command, "recorded_sources", lambda *_: [])
    monkeypatch.setattr(command, "all_runs", lambda *_: [])
    monkeypatch.setattr(
        command,
        "Oracle",
        lambda *_: SimpleNamespace(verifier=lambda: SimpleNamespace(validate=lambda: None)),
    )
    monkeypatch.setattr(
        command, "build_table", lambda *_: SimpleNamespace(table=base.oracle, problems=lambda: [])
    )
    oracle = export.parent / "oracle"
    oracle.mkdir()
    (oracle / "oracle.json").write_text("{}")
    measured = base.measured_causes.model_copy(update={"world_hash": world_hash})
    (oracle / "measured_causes.json").write_text(measured.model_dump_json())
    replay = {
        "coverage": [
            {
                "window": "scoring",
                "attempts": len(base.oracle.identities),
                "attempts_executed_twice": len(base.oracle.identities),
                "missing": [],
            }
        ]
    }
    oracle_report = SimpleNamespace(
        world_hash=world_hash, patterns=[], chance=[], replay_verification=replay
    )
    monkeypatch.setattr(command, "report_json", lambda _: oracle_report)
    analysis = Analysis.create("w01", base=tmp_path / "analyses")
    (analysis.root / "run.json").write_text(
        json.dumps(
            {"world_hash": world_hash, "export": str(export), "condition": "logs", "repeat": 1}
        )
    )
    (analysis.outbox / "diagnosis.json").write_text(base.diagnosis.model_dump_json())
    (analysis.outbox / "fix_set.json").write_text(base.fix_set.model_dump_json())
    return export.parent, analysis, oracle_report


@pytest.mark.parametrize(
    "gap", ["legacy", "partial", "wrong_count", "missing_list", "duplicate", "malformed"]
)
def test_score_world_blocks_unverified_claims_and_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap: str
) -> None:
    root, analysis, oracle = fake_scoring_world(tmp_path, monkeypatch)
    recorded_run = root / "recorded-run"
    monkeypatch.setattr(command, "all_runs", lambda *_: [recorded_run])
    row = oracle.replay_verification["coverage"][0]
    if gap == "legacy":
        oracle.replay_verification = {}
    elif gap == "partial":
        row["attempts_executed_twice"] -= 1
    elif gap == "wrong_count":
        row["attempts"] += 1
    elif gap == "missing_list":
        row["missing"] = ["e000001"]
    elif gap == "malformed":
        oracle.replay_verification["coverage"] = None
    else:
        oracle.replay_verification["coverage"].append(dict(row))
    report = command.score_world(root, analysis.root)
    assert report.claim1.recall.value is not None
    assert report.claim2.spearman.value is None and not report.claim2.complete
    assert all(
        b.agent.points is None and b.achievable.points is None for b in report.claim3.budgets
    )
    assert report.headline.ratio is None and not report.headline.rows
    for metric in (
        "claim2.spearman",
        "claim3.share_k3",
        "claim3.margin_points",
        "claim3.above_chance",
    ):
        m = mark(report, metric)
        assert m["status"] == "unavailable" and "fleetlab.oracle verify" in m["missing"]
    assert "fleetlab.oracle verify" in report_markdown(report)
    advice = shlex.split(report.metadata["replay_verification_missing"])
    destinations = [advice[i + 1] for i, arg in enumerate(advice) if arg == "--oracle-dir"]
    assert destinations == [str(root / "oracle-verified-NEW")] * 2
    assert "--no-truth-table" in advice
    assert [advice[i + 1] for i, arg in enumerate(advice) if arg == "--runs"] == [
        str(recorded_run)
    ] * 2
    pooled = pool([report], expected_worlds=["w01"])
    assert pooled["headline_ratios"][0]["ratio"] is None
    assert all(
        m["status"] == "unavailable"
        for m in pooled["pooled_pass_marks"]
        if m["metric"].startswith(("claim2.", "claim3."))
    )


def test_cli_passes_experiment_condition_and_measured_world_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = fake_scoring_world(tmp_path, monkeypatch)
    raw = json.loads((analysis.root / "run.json").read_text())
    raw["condition"] = "experiments"
    (analysis.root / "run.json").write_text(json.dumps(raw))
    out = tmp_path / "scored"
    assert (
        command.main(
            [
                "--world-root",
                str(root),
                "--analysis",
                str(analysis.root),
                "--condition",
                "with experiments",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "report.json").read_text())
    assert all(b["condition"] == "with experiments" for b in report["claim3"]["budgets"])
    assert EXPERIMENTAL_BASELINE in report["claim3"]["budgets"][0]["baselines"]
    assert "measured causes unavailable" not in str(report["claim1"]["missing"])
    assert report["claim1"]["recall"]["value"] is not None


def test_cli_lost_analysis_is_no_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, analysis, _ = fake_scoring_world(tmp_path, monkeypatch)
    raw = json.loads((analysis.root / "run.json").read_text())
    raw.update(RUNTIME["analysis_run"])
    (analysis.root / "run.json").write_text(json.dumps(raw))
    report = command.score_world(root, analysis.root)
    assert report.claim1.recall.value == 0
    assert all(b.agent.gain == Gain() for b in report.claim3.budgets)
    assert any("both model call starts lost" in n for n in report.notes)


def alternate_export(export: Path) -> Path:
    old = export.parent / "export-checks/check-fixture.json"
    old.rename(old.with_name("check-original.json"))
    other = export.with_name("export-review")
    shutil.copytree(export, other)
    (other / "log/episodes.parquet").write_bytes(b"another certified snapshot")
    certify_fixture(other, "w01")
    return other


def test_same_alternate_certified_bytes_bind_analysis_and_both_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = fake_scoring_world(tmp_path, monkeypatch)
    other = alternate_export(root / "export")
    raw = json.loads((analysis.root / "run.json").read_text())
    raw["export"] = str(other)
    (analysis.root / "run.json").write_text(json.dumps(raw))
    baselines = {
        "worst health metric": simple_source(tmp_path / "health", other),
        "classical statistics": classical_source(tmp_path / "classical", other),
    }
    ranked = inputs().fix_set
    assert ranked is not None
    for path in baselines.values():
        (path / "outbox/fix_set.json").write_text(ranked.model_dump_json())
    report = command.score_world(root, analysis.root, baselines=baselines)
    records = report.claim3.budgets[0].baseline_sources
    assert all(r.get("binding") and r["export"] == str(other) for r in records.values())
    assert report.metadata["analysis_export"] == str(other)
    for name in baselines:
        wrong = dict(baselines)
        wrong[name] = (
            simple_source(tmp_path / "badhealth", root / "export")
            if name == "worst health metric"
            else classical_source(tmp_path / "badclassical", root / "export")
        )
        (wrong[name] / "outbox/fix_set.json").write_text(ranked.model_dump_json())
        with pytest.raises(ValueError, match=r"another world|input hashes"):
            command.score_world(root, analysis.root, baselines=wrong)


def test_analysis_foreign_export_and_missing_hash_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, analysis, _ = fake_scoring_world(tmp_path, monkeypatch)
    raw = json.loads((analysis.root / "run.json").read_text())
    raw.pop("world_hash")
    (analysis.root / "run.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="world_hash"):
        command.score_world(root, analysis.root)
    raw["world_hash"] = (root / "files/hash").read_text()
    raw["export"] = str(opaque_export(tmp_path / "foreign", "w43", b"foreign"))
    (analysis.root / "run.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"export|another world"):
        command.score_world(root, analysis.root)


def test_nets_decide_exact_chance_tie_and_three_point_margin() -> None:
    report = comparison_report()
    report.scoring_episodes = 336
    at = report.claim3.budgets[0]
    at.exact_chance = {
        "histogram": [[0, 94], [12, 6]],
        "sets": 100,
        "unvalued_sets": 0,
        "p95_points": 3.571,
    }
    at.agent.gain = Gain.of(12, 0)
    at.agent.points = 100 * 12 / 336
    assert mark(report, "claim3.above_chance")["status"] == "missed"
    assert mark(report, "claim3.above_chance")["value"] == 0
    # 58 - 55 is 2.9999999999999996 points over 100 attempts.
    report.scoring_episodes = 100
    at.agent.gain = Gain.of(58, 0)
    at.agent.points = 100 * 58 / 100
    for value in at.baselines.values():
        value.gain = Gain.of(55, 0)
        value.points = 100 * 55 / 100
    assert mark(report, "claim3.margin_points")["status"] == "met"
    assert mark(report, "claim3.margin_points")["value"] == 3


@pytest.mark.parametrize(
    "field, value",
    [
        ("condition", "logs"),
        ("completed_requests", 0),
        ("requests", 9),
        ("requests", -1),
        ("requests", True),
        ("request_sha256", ""),
        ("result_sha256", ""),
        ("experiment_input_sha256", {}),
    ],
)
def test_experimental_producer_contract_fields(tmp_path: Path, field: str, value: Any) -> None:
    export = opaque_export(tmp_path / "world", "w01", b"own")
    source = classical_source(tmp_path / "producer", export, True)
    raw = json.loads((source / "evidence.json").read_text())
    raw[field] = value
    if field == "requests":
        raw["completed_requests"] = value
    (source / "evidence.json").write_text(json.dumps(raw))
    record = baseline_records(
        {EXPERIMENTAL_BASELINE: source}, world_id="w01", world_root=export.parent
    )[EXPERIMENTAL_BASELINE]
    assert "binding" not in record


@pytest.mark.parametrize("defect", ["unclean", "another_path", "uncertified"])
def test_seed_and_scorer_require_correct_clean_receipt(tmp_path: Path, defect: str) -> None:
    export = opaque_export(tmp_path / "world", "w01", b"own")
    source = simple_source(tmp_path / "producer", export)
    oracle = export.parent / "oracle"
    oracle.mkdir()
    (oracle / "oracle.json").write_text(
        json.dumps(
            {
                "seed": 13,
                "world_id": "w01",
                "world_hash": (export.parent / "files/hash").read_text(),
            }
        )
    )
    receipt = export.parent / "export-checks/check-fixture.json"
    raw = json.loads(receipt.read_text())
    if defect == "unclean":
        raw["clean"] = False
    elif defect == "another_path":
        raw["export"] = str(export.with_name("another-export"))
    else:
        raw["check_id"] = "outdated"
    receipt.write_text(json.dumps(raw))
    if defect != "uncertified":
        with pytest.raises(ValueError, match="readiness receipt"):
            default_seed(export)
    with pytest.raises(ValueError, match="certified"):
        baseline_records({"worst health metric": source}, world_id="w01", world_root=export.parent)


def test_measured_units_and_splits_ignore_curve_membership() -> None:
    base = inputs()
    assert base.measured_causes is not None
    raw = base.measured_causes.model_dump()
    item = next(i for i in raw["faults"] if i["item"] == DECOY.fault_id)
    item.update(measured_class="cause", qualified_by=["scoring"])
    item["windows"]["scoring"].update(points=2, net=2, rescued=2)
    raw["causes"].append(
        {"unit": DECOY.fault_id, "fault_ids": [DECOY.fault_id], "qualified_by": ["scoring"]}
    )
    raw["cause_count"] += 1
    raw["no_decoy_left"] = True
    curve = OracleCause(
        CAMERA.fault_id, (CAMERA.fault_id,), ("effective",), True, ("scoring",), 9, True
    )
    data = replace(
        base,
        measured_causes=MeasuredCauses.model_validate(raw),
        cause_units=(curve,),
        diagnosis=diagnosis(causes=[cause(DECOY.target, Direction.changed, 0.9)]),
    )
    report = score(data)
    assert report.claim1.detectable == 3 and report.claim1.recall.value == pytest.approx(1 / 3)
    split = score_splits(data, {DECOY.kind})["diagnosis"]["unseen"]
    assert split["detectable"] == 1 and split["found"] == 1 and split["recall"] == 1
    assert score_splits(inputs(diagnosis=None), set())["diagnosis"]["known"]["decoy_rejection"] == 0


def test_pool_refuses_comparison_condition_relabelling() -> None:
    report = comparison_report()
    report.metadata["condition"] = "with experiments"
    with pytest.raises(ValueError, match="comparison condition"):
        pool([report], expected_worlds=["w01"])


@pytest.mark.parametrize("field, value", [("rule", "wrong"), ("min_observed_carriers", 19)])
def test_measured_rule_and_minimum_are_frozen(tmp_path: Path, field: str, value: Any) -> None:
    measured = inputs().measured_causes
    assert measured is not None
    raw = measured.model_dump()
    raw[field] = value
    path = tmp_path / "measured.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="frozen design"):
        read_measured(path, "w01", None)


@pytest.mark.parametrize("defect", ["count", "fault_set"])
def test_measured_counts_and_fault_roster_are_checked(defect: str) -> None:
    measured = inputs().measured_causes
    assert measured is not None
    raw = measured.model_dump()
    if defect == "count":
        raw["cause_count"] += 1
    else:
        raw["faults"].pop()
    with pytest.raises(ValueError, match=r"cause_count|faults differ"):
        score(inputs(measured_causes=MeasuredCauses.model_validate(raw)))
