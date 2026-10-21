"""Score one schema-checked analysis or pool existing score records."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.actions import FixSet
from fleetlab.integration.chain import tool_log
from fleetlab.log.leaks import export_digest
from fleetlab.oracle.report import report_json
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.table import build_table
from fleetlab.oracle.world import OracleLayout, all_runs, load_world, recorded_sources, world_ready
from fleetlab.scoring import ScoringInputs, report_markdown, score
from fleetlab.scoring.accounting import analysis_ledgers, score_accounting
from fleetlab.scoring.loading import baseline_answers, baseline_records, check_run_record
from fleetlab.scoring.lost import analysis_lost
from fleetlab.scoring.measured import read_measured
from fleetlab.scoring.models import ScoreReport
from fleetlab.scoring.provenance import analysis_export
from fleetlab.scoring.splits import score_splits
from fleetlab.scoring.verification import replay_problem

CONDITIONS = ("logs alone", "with experiments", "with memory", "knowledge ablation")


def score_world(
    root: Path,
    analysis_root: Path,
    *,
    issued_day: int | None = None,
    condition: str = "logs alone",
    repeat: int = 1,
    method: str = "agent",
    baselines: dict[str, Path] | None = None,
    unseen: set[str] | None = None,
    ledger: Path | None = None,
    prices: dict[str, Any] | None = None,
    attempts_analysed: int | None = None,
    extra_runs: Sequence[Path] = (),
    oracle_dir: Path | None = None,
) -> ScoreReport:
    world = load_world(root)
    if world.fault_table is None:
        raise ValueError("world lacks the hidden fault table")
    if problem := world_ready(root):
        raise ValueError(problem)
    analysis = Analysis(analysis_root)
    if analysis.world_id != world.world_id:
        raise ValueError("analysis and world ids differ")
    if condition == "logs alone" and analysis.requests():
        raise ValueError("analysis requested experiments: specify its actual condition")
    layout = OracleLayout(oracle_dir or root / "oracle", world.world_id)
    source_runs = all_runs(root, layout, [*extra_runs, *recorded_sources(layout, world.world_hash)])
    verifier = Oracle(root, world, layout).verifier()
    verifier.validate()
    built = build_table(world.identities, world.excluded, source_runs, world.world_hash, verifier)
    rankings, sets = baseline_answers(baselines or {})
    checkpoint, record_notes = check_run_record(
        analysis_root,
        condition,
        repeat,
        issued_day,
        {
            "logs alone": world.spec.windows.observed.last,
            "with experiments": world.spec.windows.experiment.last,
        },
        world.world_hash,
    )
    export, binding_missing = analysis_export(analysis_root, root, world.world_id)
    fixed = None
    answer_error = None
    try:
        if analysis_lost(analysis_root):
            raise ValueError("both model call starts lost")
        diagnosis = analysis.diagnosis()
        fixed_path = analysis.outbox / "fix_set.json"
        if fixed_path.is_file() and "budget" in json.loads(fixed_path.read_text()):
            fixed = FixSet.model_validate_json(fixed_path.read_text())
            if fixed.actions:
                raise ValueError("analysis requires a ranked list or an empty FixSet")
        ranking = None if fixed is not None else analysis.fix_set()
        if diagnosis is None and ranking is None:
            raise ValueError("no answer submitted")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        answer_error = "unreadable answer scores as no answer: " + str(exc)
        diagnosis = None
        ranking = None
    additive = list(world.additive_pairs)
    oracle_chance = []
    replay_verification: dict[str, Any] = {}
    if layout.report.is_file():
        oracle_report = report_json(layout.report)
        if oracle_report.world_hash != world.world_hash:
            raise ValueError("oracle pattern report belongs to another world hash")
        additive = [
            (p.items[0], p.items[1])
            for p in oracle_report.patterns
            if p.pattern == "additive_pair" and len(p.items) == 2
        ]
        oracle_chance = [c.model_dump(mode="json") for c in oracle_report.chance]
        replay_verification = oracle_report.replay_verification
    # Existing oracle results remain records. Verify and report into a fresh
    # destination, carrying the actual recorded replay sources with it.
    verified_oracle = layout.root.with_name(layout.root.name + "-verified-NEW")
    replay_sources = [arg for path in source_runs for arg in ("--runs", str(path))]
    verify_command = shlex.join(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "fleetlab.oracle",
            "verify",
            str(root),
            "--oracle-dir",
            str(verified_oracle),
            "--window",
            "scoring",
            *replay_sources,
        ]
    )
    verify_command += " && " + shlex.join(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "fleetlab.oracle",
            "report",
            str(root),
            "--oracle-dir",
            str(verified_oracle),
            "--no-truth-table",
            *replay_sources,
        ]
    )
    replay_missing = replay_problem(replay_verification, len(world.scoring()), verify_command)
    inputs = ScoringInputs(
        fault_table=world.fault_table,
        condition=condition,
        replay_missing=replay_missing,
        analysis_binding_missing=binding_missing,
        measured_causes=read_measured(
            layout.root / "measured_causes.json", world.world_id, world.world_hash
        ),
        measured_missing="measured causes unavailable; produce them with: "
        + shlex.join(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "fleetlab.oracle",
                "causes",
                str(root),
                "--oracle-dir",
                str(layout.root),
            ]
        ),
        oracle=built.table.restricted(world.scoring()),
        truth=world.truth(),
        current=world.current(),
        diagnosis=diagnosis,
        fix_set=ranking,
        decision_actions=[] if fixed is not None or answer_error is not None else None,
        tool_log=tool_log(analysis_root),
        catalogue=world.catalogue(),
        seed=world.spec.seed,
        confounders=list(world.spec.patterns.confounders),
        as_run=built.table,
        issued_day=checkpoint,
        drift_faults=[f for f in world.spec.fault_schedule if f.schedule.drift is not None],
        baseline_rankings=rankings,
        baseline_fix_sets=sets,
        baseline_sources=baseline_records(
            baselines or {},
            world_id=world.world_id,
            world_root=root if export is not None else None,
            export=export,
        ),
        cause_units=world.causes,
        additive_pairs=additive,
        oracle_chance=oracle_chance,
        chance_command=shlex.join(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "fleetlab.oracle",
                "report",
                str(root),
                "--oracle-dir",
                "NEW_ORACLE_DIR",
                *(arg for path in source_runs for arg in ("--runs", str(path))),
            ]
        ),
        drift_crossing_days={
            f.fault_id: f.model_dump()["drift_crossing_day"]
            for f in world.fault_table.faults
            if "drift_crossing_day" in f.model_fields_set
        },
    )
    report = score(inputs)
    attempts = {i.episode_id for i in world.window("observed")}
    for result in analysis.results_by_id().values():
        for arm in result.arms:
            attempts.update(arm.episode_ids)
    denominator = len(attempts) if attempts_analysed is None else attempts_analysed
    report.claim6 = score_accounting(
        analysis, denominator, analysis_ledgers(analysis, ledger), prices
    )
    report.claim6["denominator_basis"] = (
        "observed attempts plus distinct delivered experiment attempts"
        if attempts_analysed is None
        else "explicit caller denominator"
    )
    report.splits = score_splits(inputs, unseen or set())
    report.metadata = {
        "analysis_id": analysis.analysis_id,
        "analysis_root": str(analysis_root.resolve()),
        "world_root": str(root.resolve()),
        "world_hash": world.world_hash,
        "condition": condition,
        "run_labels_verified": True,
        "repeat": repeat,
        "primary": repeat == 1,
        "method": method,
        "oracle_report": str(layout.report),
        "replay_verification": replay_verification,
        "replay_verification_missing": replay_missing,
        "analysis_export": None if export is None else str(export),
        "analysis_export_digest": None if export is None else export_digest(export),
        "analysis_binding_missing": binding_missing,
        "baseline_sources": {
            name: str(path.expanduser().resolve()) for name, path in (baselines or {}).items()
        },
    }
    report.notes.extend([*built.problems(), *record_notes])
    if answer_error:
        report.notes.append(answer_error)
    return report


def write_score(report: ScoreReport, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (out / "report.md").write_text(report_markdown(report))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-root", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--reports", nargs="+", type=Path, help="pool existing report.json files")
    parser.add_argument("--world-roster", type=Path, help="JSON roster with world IDs and hashes")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--issued-day", type=int)
    parser.add_argument("--condition", choices=CONDITIONS, default="logs alone")
    parser.add_argument("--repeat", choices=(1, 2), type=int, default=1)
    parser.add_argument("--method", default="agent")
    parser.add_argument(
        "--baseline", action="append", default=[], help="NAME=PATH to ranking or budget FixSets"
    )
    parser.add_argument(
        "--unseen-types", type=Path, help="JSON list of exact fault type names; default empty"
    )
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--prices", type=Path, help="dated, sourced per-million-token USD schedule")
    parser.add_argument("--attempts-analysed", type=int)
    parser.add_argument("--extra-run", action="append", type=Path, default=[])
    parser.add_argument("--oracle-dir", type=Path, help="new oracle output root, if supplied")
    args = parser.parse_args(argv)
    args.out = args.out.expanduser()
    if args.issued_day is not None and args.issued_day < 0:
        parser.error("--issued-day must be non-negative")
    if args.reports:
        if args.world_root or args.analysis:
            parser.error("--reports is separate from single-analysis scoring")
        if args.world_roster is None:
            parser.error("--world-roster is required with --reports; typed world lists are refused")
        from fleetlab.scoring.pooling import pool, pool_markdown

        reports = [
            ScoreReport.model_validate_json(p.expanduser().read_text()) for p in args.reports
        ]
        pooled = pool(reports, roster=args.world_roster)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "report.json").write_text(json.dumps(pooled, indent=2) + "\n")
        (args.out / "report.md").write_text(pool_markdown(pooled))
    else:
        if args.world_root is None or args.analysis is None:
            parser.error("--world-root and --analysis are required")
        baselines = {}
        for value in args.baseline:
            name, separator, path = value.partition("=")
            if not separator or not name or not path or name in baselines:
                parser.error("baselines are unique NAME=PATH assignments")
            baselines[name] = Path(path)
        unseen = (
            []
            if args.unseen_types is None
            else json.loads(args.unseen_types.expanduser().read_text())
        )
        if not isinstance(unseen, list) or not all(isinstance(t, str) for t in unseen):
            parser.error("--unseen-types is a JSON list of exact type names")
        if args.attempts_analysed is not None and args.attempts_analysed <= 0:
            parser.error("--attempts-analysed must be positive")
        prices = None if args.prices is None else json.loads(args.prices.expanduser().read_text())
        report = score_world(
            args.world_root.expanduser(),
            args.analysis.expanduser(),
            issued_day=args.issued_day,
            condition=args.condition,
            repeat=args.repeat,
            method=args.method,
            baselines=baselines,
            unseen=set(unseen),
            ledger=None if args.ledger is None else args.ledger.expanduser(),
            prices=prices,
            attempts_analysed=args.attempts_analysed,
            extra_runs=args.extra_run,
            oracle_dir=None if args.oracle_dir is None else args.oracle_dir.expanduser(),
        )
        write_score(report, args.out)
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
