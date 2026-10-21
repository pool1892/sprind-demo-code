"""Seed-only chance metadata reader; standard library only, no oracle imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _export_digest(root: Path) -> str:
    """The export receipt's names/bytes digest, implemented without truth imports."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("seed export must be a regular directory")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("seed export contains a symbolic link")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(b"directory\0" if path.is_dir() else b"file\0")
        if path.is_file():
            content = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(block)
            digest.update(content.digest())
        elif not path.is_dir():
            raise ValueError("seed export has an unsupported entry")
    return digest.hexdigest()


def default_seed(export: Path, chance_table: Path | None = None) -> int:
    """Use seed and binding metadata only; never inspect a chance valuation."""
    export = export.expanduser().resolve()
    path = (chance_table or export.parent / "oracle/oracle.json").expanduser().resolve()
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read chance-table seed; pass --chance-table or --seed") from exc
    if not isinstance(record, dict):
        raise ValueError("chance-table seed record must be an object")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("chance-table seed must be a non-negative integer")
    root = export.parent
    try:
        world_id = json.loads((root / "runner.json").read_text())["world_id"]
        hashes = [root / part / "hash" for part in ("files", "world")]
        world_hash = next(p.read_text().strip() for p in hashes if p.is_file())
        if record.get("world_id") != world_id or record.get("world_hash") != world_hash:
            raise ValueError("chance-table seed belongs to a different world")
        digest = _export_digest(export)
        receipts = (
            json.loads(p.read_text()) for p in (root / "export-checks").glob("check-*.json")
        )
        if not any(
            r.get("clean") is True
            and r.get("export") == str(export)
            and r.get("export_digest") == digest
            for r in receipts
        ):
            raise ValueError("seed export has no matching readiness receipt")
    except (OSError, KeyError, StopIteration) as exc:
        raise ValueError("seed world binding unavailable; pass --chance-table or --seed") from exc
    return seed
