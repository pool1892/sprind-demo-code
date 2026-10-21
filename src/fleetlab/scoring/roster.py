"""Read the complete comparison roster and identify its exact bytes in every pool."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def read_roster(path: Path) -> tuple[dict[str, str | None], dict[str, Any]]:
    if not isinstance(path, Path):
        raise ValueError("pool requires a roster file, not a typed list of worlds")
    path = path.expanduser().resolve()
    data = path.read_bytes()
    raw = json.loads(data)
    if not isinstance(raw, dict) or not isinstance(raw.get("worlds"), list) or not raw["worlds"]:
        raise ValueError("roster needs a nonempty worlds list")
    worlds: dict[str, str | None] = {}
    for row in raw["worlds"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("world_id"), str)
            or not row["world_id"]
        ):
            raise ValueError("roster entries need world_id and world_hash")
        world = row["world_id"]
        if world in worlds or "world_hash" not in row:
            raise ValueError("roster has duplicate worlds or an omitted world_hash")
        digest = row["world_hash"]
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("world_hash must be a SHA-256 digest or explicit null")
        worlds[world] = digest
    return worlds, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "worlds": raw["worlds"],
    }
