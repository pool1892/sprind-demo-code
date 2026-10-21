"""Bindings to the records written by the baseline producers, not caller labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fleetlab.contracts.log import DESCRIPTION_FILE
from fleetlab.log.leaks import export_check_problem, export_digest
from fleetlab.scoring.comparisons import BASELINE_METHODS, MODEL_BASELINES
from fleetlab.scoring.lost import run_records
from fleetlab.worlds.layout import WorldPaths


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export_world(export: Path) -> str:
    return str(json.loads((export / DESCRIPTION_FILE).read_text())["fleet"]["id"])


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("producer record path escapes its directory")
    return path


def certified_export(
    world_root: Path, world_id: str, export: Path | None = None
) -> tuple[Path, str]:
    """Readiness certifies these bytes; a public fleet name is not an identity."""
    world_root = world_root.expanduser().resolve()
    export = (export or world_root / "export").expanduser().resolve()
    if export.parent != world_root:
        raise ValueError("analysis/producer export is outside the scored world")
    if WorldPaths.from_root(world_root).world_id != world_id:
        raise ValueError("producer context belongs to another world")
    if problem := export_check_problem(export, receipt_dir=world_root / "export-checks"):
        raise ValueError("scored world export is not certified: " + problem)
    return export, export_digest(export)


def analysis_export(root: Path, world_root: Path, world_id: str) -> tuple[Path | None, str | None]:
    """Choose the runtime's export, including older certified snapshots of this world."""
    paths = {
        Path(raw["export"]).expanduser().resolve()
        for _, raw in run_records(root)
        if raw.get("export")
    }
    if len(paths) > 1:
        raise ValueError("conflicting analysis export records")
    if not paths:
        return None, "analysis run record lacks a certified export binding"
    export, _ = certified_export(world_root, world_id, paths.pop())
    return export, None


def producer_binding(
    name: str,
    root: Path,
    *,
    world_id: str | None,
    world_root: Path | None,
    export: Path | None = None,
) -> dict[str, Any]:
    """Use manifest output pointers or the classical evidence's own output directory.

    Neither current producer hashes its fix sets. These are directory bindings;
    the hashes captured here are receipts, not invented producer attestations.
    """
    method, _, suffix = name.partition(" repeat ")
    if method not in BASELINE_METHODS or world_id is None or world_root is None:
        return {"binding_missing": "producer world context not supplied"}
    world_root = world_root.expanduser().resolve()
    if method in ("classical statistics", "classical statistics with experiments"):
        record_path = root / "evidence.json"
        if not record_path.is_file():
            return {"binding_missing": "producer evidence.json absent"}
        raw = json.loads(record_path.read_text())
        export, digest = certified_export(world_root, world_id, export)
        if raw.get("world_id") != _export_world(export):
            raise ValueError("classical producer record belongs to another world")
        experimental = method == "classical statistics with experiments"
        if experimental and (
            raw.get("condition") != "experiments"
            or type(raw.get("requests")) is not int
            or not 0 <= raw["requests"] <= 8
            or raw.get("completed_requests") != raw["requests"]
            or not raw.get("request_sha256")
            or not raw.get("result_sha256")
        ):
            return {"binding_missing": "classical experiments record incomplete or absent"}
        if not experimental and raw.get("condition") == "experiments":
            raise ValueError("experiment baseline cannot stand in for logs-only classical")
        hashes = raw.get("input_sha256")
        if (
            not isinstance(hashes, dict)
            or not hashes.get(str(DESCRIPTION_FILE))
            or not hashes.get("log/episodes.parquet")
        ):
            return {
                "binding_missing": "classical evidence lacks input_sha256/description or episodes"
            }
        if not all(
            _inside(export, key).is_file() and file_hash(_inside(export, key)) == value
            for key, value in hashes.items()
        ):
            raise ValueError("classical producer input hashes match no export of the scored world")
        if experimental and (
            not isinstance(raw.get("experiment_input_sha256"), dict)
            or raw["experiment_input_sha256"].get(str(DESCRIPTION_FILE))
            != hashes[str(DESCRIPTION_FILE)]
        ):
            return {"binding_missing": "classical experiment description differs from logs"}
        if not (root / "outbox/fix_set.json").is_file():
            return {"binding_missing": "classical producer output outbox/fix_set.json absent"}
        return {
            "record": {"method": raw.get("method"), "world_id": world_id},
            "public_world_id": raw["world_id"],
            "producer_record": str(record_path),
            "producer_record_sha256": file_hash(record_path),
            "export": str(export),
            "export_digest": digest,
            "binding": "evidence directory; inputs match certified export bytes",
            "producer_fix_set_hashes": "absent in evidence.json",
        }
    # The producer writes repeat-2 methods two directory levels below its output root.
    for parent in list(root.parents)[:3]:
        record_path = parent / "manifest.json"
        if not record_path.is_file():
            continue
        raw = json.loads(record_path.read_text())
        entries = [(key, path, None) for key, path in raw.get("methods", {}).items()]
        entries += [
            (key, path, repeat.get("repeat"))
            for repeat in raw.get("model_repeats", [])
            for key, path in repeat.get("methods", {}).items()
        ]
        candidates = [
            (key, repeat)
            for key, path, repeat in entries
            if _inside(parent, path) == root / "outbox/fix_set.json"
        ]
        if not candidates:
            continue
        if any(key not in BASELINE_METHODS[method] for key, _ in candidates):
            raise ValueError(f"baseline {name} method differs from its producer manifest")
        actual_method = candidates[0][0]
        repeats = {repeat for _, repeat in candidates if repeat is not None}
        if len(repeats) > 1:
            raise ValueError("producer manifest has conflicting repeats for the same output")
        repeat = next(iter(repeats), None)
        expected = 2 if suffix == "2" else 1
        if method in MODEL_BASELINES and repeat is not None and repeat != expected:
            raise ValueError(f"baseline {name} repeat differs from its producer manifest")
        producer_export = Path(raw["export"]).expanduser().resolve()
        _, digest = certified_export(world_root, world_id, export)
        if export_digest(producer_export) != digest:
            raise ValueError("baseline producer export belongs to another world")
        if not (root / "outbox/fix_set.json").is_file():
            return {"binding_missing": "producer manifest output absent"}
        return {
            "record": {"method": actual_method, "repeat": repeat, "world_id": world_id},
            "producer_record": str(record_path),
            "producer_record_sha256": file_hash(record_path),
            "export": str(producer_export),
            "export_digest": digest,
            "public_world_id": _export_world(producer_export),
            "binding": "manifest output pointer; certified export bytes; sibling fix_sets",
            "producer_fix_set_hashes": "absent in manifest.json",
        }
    return {"binding_missing": "producer manifest naming these outputs absent"}
