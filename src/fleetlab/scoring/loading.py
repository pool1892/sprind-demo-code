"""Read supplied baseline answers without inventing unavailable methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import SCORING_BUDGETS, Action, FixSet
from fleetlab.contracts.tools import RankedFixSet
from fleetlab.scoring.comparisons import BASELINE_METHODS, MODEL_BASELINES
from fleetlab.scoring.decisions import budget_label
from fleetlab.scoring.lost import exhausted_call, run_records
from fleetlab.scoring.provenance import file_hash, producer_binding


def check_run_record(
    analysis_root: Path,
    condition: str,
    repeat: int,
    issued_day: int | None,
    checkpoints: dict[str, int],
    world_hash: str | None,
) -> tuple[int | None, list[str]]:
    """Refuse relabelled runs; old records derive their day from their condition."""
    notes = []
    located = run_records(analysis_root)
    records = [raw for _, raw in located]
    if any(path.name == "run.json" and not raw.get("world_hash") for path, raw in located):
        raise ValueError("analysis run.json lacks world_hash")
    aliases = {"logs": "logs alone", "experiments": "with experiments"}
    recorded_condition = False
    recorded_repeat = False
    day = issued_day
    for record in records:
        if record.get("world_hash") not in (None, world_hash):
            raise ValueError("analysis run and world hashes differ")
        if "condition" in record:
            actual = aliases.get(record["condition"], record["condition"])
            if actual != condition:
                raise ValueError("condition differs from the analysis run record")
            recorded_condition = True
        if "repeat" in record:
            if record["repeat"] != repeat:
                raise ValueError("repeat differs from the analysis run record")
            recorded_repeat = True
        expected_day = record.get("issued_day")
        if expected_day is None and "condition" in record:
            actual = aliases.get(record["condition"], record["condition"])
            expected_day = checkpoints.get(actual)
        if expected_day is not None:
            if day is not None and day != expected_day:
                raise ValueError("issue day differs from the analysis run record")
            day = expected_day
    if not recorded_condition or not recorded_repeat:
        raise ValueError("run record lacks condition or repeat; unverified labels refused")
    if day is None:
        day = checkpoints.get(condition)
        notes.append("run record lacks issue day; window checkpoint used descriptively")
    return day, notes


def baseline_records(
    paths: dict[str, Path],
    *,
    world_id: str | None = None,
    world_root: Path | None = None,
    export: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Keep producer records with the values; copies without records stay unverified."""
    records = {}
    sources: dict[Path, str] = {}
    for name, original in paths.items():
        path = original.expanduser().resolve()
        root = path.parent if path.suffix == ".json" else path
        if root.name in ("fix_sets", "outbox"):
            root = root.parent
        if root in sources:
            raise ValueError(f"baselines {sources[root]} and {name} share a source: {root}")
        sources[root] = name
        method, _, suffix = name.partition(" repeat ")
        record: dict[str, Any] = {}
        record_paths = []
        for file in ("metadata.json", "evidence.json", "analysis.json", "run.json"):
            record_path = root / file
            if not record_path.is_file():
                continue
            raw = json.loads(record_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"invalid baseline record: {record_path}")
            # A source may say of itself that it is no baseline (the build rule does).
            if raw.get("baseline_of_the_experiment") is False:
                raise ValueError(
                    f"baseline {name}: its source says it is not a baseline of the experiment: "
                    f"{record_path}"
                )
            for key in (
                "method",
                "repeat",
                "attempts",
                "rerun",
                "lost_twice",
                "explicit_fix_sets_authoritative",
            ):
                if key in raw:
                    if key in record and record[key] != raw[key]:
                        raise ValueError(f"conflicting baseline {key} records: {root}")
                    record[key] = raw[key]
            record_paths.append(str(record_path))
        if method in BASELINE_METHODS:
            if "method" in record and record["method"] not in BASELINE_METHODS[method]:
                raise ValueError(f"baseline {name} method differs from its source record")
            if method in MODEL_BASELINES:
                expected = 2 if suffix == "2" else 1
                if "repeat" in record and record["repeat"] != expected:
                    raise ValueError(f"baseline {name} repeat differs from its run record")
        # A lost method has no directory. Its only receipt is manifest.missing.
        for parent in list(root.parents)[:3]:
            manifest = parent / "manifest.json"
            if not manifest.is_file():
                continue
            raw = json.loads(manifest.read_text())
            expected = 2 if suffix == "2" else 1
            entries = [(parent, raw.get("missing", {}))] if expected == 1 else []
            entries += [
                (parent / "repeats" / f"repeat-{expected}", item.get("missing", {}))
                for item in raw.get("model_repeats", [])
                if item.get("repeat") == expected
            ]
            for base, missing in entries:
                for key, failure in missing.items():
                    if key in BASELINE_METHODS.get(method, ()) and root == base / key:
                        record.update(failure)
                        record_paths.append(str(manifest))
        lost = exhausted_call(record)
        starts = record.get("attempts")
        binding: dict[str, Any] = (
            producer_binding(name, root, world_id=world_id, world_root=world_root, export=export)
            if not lost
            else {"binding_missing": "both model call starts lost"}
        )
        if "record" in binding:
            produced = binding["record"]
            for key in ("method", "repeat"):
                if key in record and key in produced and record[key] != produced[key]:
                    raise ValueError(f"baseline {name} metadata differs from its producer {key}")
            record = {**record, **produced}
        files = sorted((root / "fix_sets").glob("*.json"))
        outputs = [*files, root / "outbox/fix_set.json"]
        if any(not file.resolve().is_relative_to(root) for file in outputs):
            raise ValueError("baseline fix sets escape the producer directory")
        if "record" in binding and path not in (
            root,
            root / "fix_sets",
            root / "outbox",
            *outputs,
        ):
            raise ValueError("baseline answer is not a producer-recorded output")
        records[name] = {
            **binding,
            "source": str(root),
            "record": record,
            "record_paths": record_paths,
            "lost_call": lost,
            "attempts": starts,
            "scored_file_sha256": {
                str(p.relative_to(root)): file_hash(p) for p in outputs if p.is_file()
            },
        }
    return records


def baseline_answers(
    paths: dict[str, Path],
) -> tuple[dict[str, list[Action]], dict[str, dict[str, FixSet]]]:
    rankings: dict[str, list[Action]] = {}
    fixed: dict[str, dict[str, FixSet]] = {}
    records = baseline_records(paths)
    for name, original in paths.items():
        if records[name]["lost_call"]:
            continue
        path = original.expanduser().resolve()
        root = Path(records[name]["source"])
        authoritative = records[name]["record"].get("explicit_fix_sets_authoritative") is True
        if authoritative and path in (root / "outbox", root / "outbox/fix_set.json"):
            raise ValueError("ranked answer refused: producer fix_sets are authoritative")
        if authoritative and not (root / "fix_sets").is_dir():
            raise ValueError("authoritative producer fix_sets are absent")
        if (root / "fix_sets").is_dir():
            path = root
            sets = {}
            for file in sorted((path / "fix_sets").glob("*.json")):
                fix = FixSet.model_validate_json(file.read_text())
                if fix.budget not in SCORING_BUDGETS:
                    raise ValueError(f"not a scoring budget: {fix.budget}")
                key = budget_label(fix.budget)
                if key in sets:
                    raise ValueError(f"duplicate baseline budget {name}: {key}")
                sets[key] = fix
            if not sets:
                raise ValueError(f"no authoritative budget records at {path / 'fix_sets'}")
            fixed[name] = sets
            continue
        if path.is_dir():
            for relative in ("outbox/fix_set.json", "fix_set.json", "ranked_fix_set.json"):
                if (path / relative).is_file():
                    path = path / relative
                    break
            else:
                # A directory of budget-specific FixSet records.
                files = sorted(path.glob("*.json"))
                sets = {}
                for file in files:
                    fix = FixSet.model_validate_json(file.read_text())
                    if fix.budget not in SCORING_BUDGETS:
                        raise ValueError(f"not a scoring budget: {fix.budget}")
                    key = budget_label(fix.budget)
                    if key in sets:
                        raise ValueError(f"duplicate baseline budget {name}: {key}")
                    sets[key] = fix
                if not sets:
                    raise ValueError(f"no baseline answer at {original}")
                fixed[name] = sets
                continue
        raw = json.loads(path.read_text())
        if "budget" in raw:
            fix = FixSet.model_validate(raw)
            if fix.budget not in SCORING_BUDGETS:
                raise ValueError(f"not a scoring budget: {fix.budget}")
            fixed[name] = {budget_label(fix.budget): fix}
        else:
            rankings[name] = [r.action for r in RankedFixSet.model_validate(raw).actions]
    return rankings, fixed
