"""The pinned convolution database (replay rule 7).

X-VLA's vision tower has eight kinds of convolution and MIOpen several
algorithms for each; which one runs decides the last bits of the result
(xvla-speed section 4: 1.7e-6 in 32 bits, 1e-3 in 16 bits, a millimetre of
commanded position). With `torch.backends.cudnn.benchmark = True` every
process times the algorithms again and takes the fastest of the moment; with
it off PyTorch takes the fastest one stored in MIOpen's database and writes
nothing, and with nothing stored it falls back to a slower default with yet
other bits. So:

- `pin` searches once per (batch size, precision) on a quiet machine, into a
  database of the project's own (`MIOPEN_USER_DB_PATH`,
  `MIOPEN_CUSTOM_CACHE_DIR`), and makes the database files read-only. It
  writes `pinned.json`: what was searched and the hash of every file.
- `activate` points this process at that database and switches benchmarking
  off. It must run before the first convolution.
- `require` refuses a (batch size, precision) the database does not hold, and
  a database whose files no longer match their recorded hashes: MIOpen would
  silently fall back to another algorithm.
- The database hash is part of `SoftwareVersion` (`miopen_db_hash`).

The search is keyed by the number of images in a call (two real views per
row; the empty views are masked out) and by the number format of the vision
tower, which is float16 in both 16-bit settings. Where it lives: below
`FLEETLAB_MIOPEN_DIR`, by default `<data>/miopen/<name>`. Exactness holds per
machine, so the database is machine-local and never committed.

    tools/heavy-run uv run --locked python -m fleetlab.engine.miopen pin \\
        --batch-sizes 1 4 8 --precisions fp32 fp16_res32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from fleetlab.runtime import data_dir

DEFAULT_NAME = "pinned"
MANIFEST = "pinned.json"


class UnpinnedError(RuntimeError):
    """The convolution database does not hold what a run needs, or has changed."""


def root(name: str = DEFAULT_NAME) -> Path:
    override = os.environ.get("FLEETLAB_MIOPEN_DIR")
    return Path(override) / name if override else data_dir() / "miopen" / name


def vision_format(precision: str) -> str:
    """The number format the vision tower computes in, which is what MIOpen's search sees."""
    return "fp32" if precision == "fp32" else "fp16"


def activate(name: str = DEFAULT_NAME) -> Path:
    """Point MIOpen at the pinned database and switch benchmarking off, in this process.

    Sets the environment variables (they must be set before MIOpen is first
    used; children inherit them) and `torch.backends.cudnn.benchmark = False`
    if torch is loaded.
    """
    base = root(name)
    os.environ["MIOPEN_USER_DB_PATH"] = str(base / "db")
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(base / "cache")
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    (base / "cache").mkdir(parents=True, exist_ok=True)
    import sys

    if "torch" in sys.modules:
        import torch

        torch.backends.cudnn.benchmark = False
    return base


def file_hashes(base: Path) -> dict[str, str]:
    db = base / "db"
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(db.glob("*"))
        if f.is_file()
    }


def database_hash(base: Path) -> str:
    """One hash over the database files (names and contents), 16 hexadecimal digits."""
    files = file_hashes(base)
    text = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def manifest(base: Path) -> dict[str, Any]:
    path = base / MANIFEST
    if not path.is_file():
        raise UnpinnedError(f"no pinned convolution database at {base} (run `miopen pin`)")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def require(batch_size: int, precision: str, name: str = DEFAULT_NAME) -> str:
    """The database hash, if the pinned database holds this (batch size, precision); else raise."""
    base = root(name)
    record = manifest(base)
    if file_hashes(base) != record["files"]:
        raise UnpinnedError(f"the database at {base} differs from what was pinned")
    held = {(e["batch_size"], e["vision_format"]) for e in record["entries"]}
    if (batch_size, vision_format(precision)) not in held:
        raise UnpinnedError(
            f"batch size {batch_size} in {precision} ({vision_format(precision)} vision tower) "
            f"is not in the pinned database {base}; it holds {sorted(held)}"
        )
    return str(record["hash"])


REAL_VIEWS = 2  # the scene and the wrist camera; X-VLA's other views are empty and masked


def images_per_call(batch_size: int, image_views: int) -> int:
    """How many images the vision tower convolves in one call: what the database is keyed by.

    Each row brings its real views (at most the two cameras); the empty views
    are masked out before the vision tower.
    """
    return batch_size * min(image_views, REAL_VIEWS)


def require_images(n_images: int, precision: str, name: str = DEFAULT_NAME) -> str:
    """The database hash, if a call of `n_images` images in `precision` is pinned; else raise.

    Replay rule 7 per call: a configuration fault that changes the image views
    changes the images per call and needs its own pinned entry.
    """
    base = root(name)
    record = manifest(base)
    if file_hashes(base) != record["files"]:
        raise UnpinnedError(f"the database at {base} differs from what was pinned")
    fmt = vision_format(precision)
    held = {REAL_VIEWS * e["batch_size"] for e in record["entries"] if e["vision_format"] == fmt}
    if n_images not in held:
        raise UnpinnedError(
            f"{n_images} images per call in {fmt} are not in the pinned database {base}; "
            f"it holds {sorted(held)}"
        )
    return str(record["hash"])


def pin(batch_sizes: list[int], precisions: list[str], name: str = DEFAULT_NAME) -> dict[str, Any]:
    """Search the convolution algorithms for these settings and pin the database."""
    base = root(name)
    db = base / "db"
    db.mkdir(parents=True, exist_ok=True)
    for f in db.glob("*"):
        f.chmod(f.stat().st_mode | stat.S_IWUSR)  # a database pinned before is extended
    os.environ["MIOPEN_USER_DB_PATH"] = str(db)
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(base / "cache")
    import torch

    from fleetlab.policies.xvla import XVLAPolicy, sample_observations

    old = json.loads((base / MANIFEST).read_text()) if (base / MANIFEST).is_file() else {}
    entries = {(e["batch_size"], e["vision_format"]) for e in old.get("entries", [])}
    searched = list(old.get("searched", []))
    observations = sample_observations(max(batch_sizes))
    formats = sorted({vision_format(p) for p in precisions})
    for fmt in formats:
        policy = XVLAPolicy()
        policy.load("cuda", "fp32" if fmt == "fp32" else "fp16_res32", benchmark=True)
        for b in batch_sizes:
            seconds = policy.time_vision(observations[:b])
            entries.add((b, fmt))
            searched.append({"batch_size": b, "vision_format": fmt, "search_s": round(seconds, 1)})
            print(searched[-1], flush=True)
        del policy
        torch.cuda.empty_cache()
    for f in db.glob("*"):
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    record = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [{"batch_size": b, "vision_format": f} for b, f in sorted(entries)],
        "searched": searched,
        "files": file_hashes(base),
        "hash": database_hash(base),
        "loadavg": Path("/proc/loadavg").read_text().split()[:3],
    }
    (base / MANIFEST).write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine.miopen")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pin", help="search and pin the convolution algorithms (heavy)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16_res32"])
    p.add_argument("--name", default=DEFAULT_NAME)
    s = sub.add_parser("show", help="what the pinned database holds")
    s.add_argument("--name", default=DEFAULT_NAME)
    args = parser.parse_args()
    from fleetlab.runtime import configure_process

    configure_process()
    if args.command == "pin":
        record = pin(args.batch_sizes, args.precisions, args.name)
    else:
        record = manifest(root(args.name))
    print(json.dumps({k: record[k] for k in ("entries", "hash")}, indent=1))


if __name__ == "__main__":
    main()
