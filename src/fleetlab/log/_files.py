"""Files of a run directory: atomic writes through a staging directory, Parquet, the canary.

Every file of a run is written under a temporary name in `<run_dir>/.staging/`
and renamed into place, so a reader never sees half a file and a killed writer
leaves its debris in one directory that the next writer clears. Parquet is
written with zstd; the same rows give the same bytes. Parquet files are
written without Arrow's stored schema: pyarrow 25 cannot read back a null in a
fixed-size list column when that schema is present ("Expected all lists to be
of size=7"), and the last row of an episode has no commanded action. Readers
cast to the declared types (`_rows.to_arrow`); DuckDB reads the lists as lists.
Hidden tables keep the stored schema, because it carries their canary; they
have no list columns.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "CANARY_FILE",
    "CANARY_METADATA_KEY",
    "MERGED_FILE",
    "PARTS_DIR",
    "STAGING_DIR",
    "clear_staging",
    "plant_canary",
    "read_canary",
    "staged",
    "write_parquet",
    "write_text",
]

STAGING_DIR = ".staging"
PARTS_DIR = "truth/parts"
MERGED_FILE = "truth/parts/_merged.json"
CANARY_FILE = "truth/canary.txt"
CANARY_METADATA_KEY = b"fleetlab.canary"
LOCK_FILE = ".writer.lock"


def run_root(path: Path) -> Path:
    """The run directory a file of the run lies in (the parent of its `log` or `truth`)."""
    for parent in path.parents:
        if (parent / "run.json").is_file() or (parent / STAGING_DIR).is_dir():
            return parent
    raise FileNotFoundError(f"{path} is not inside a run directory")


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def staged(final: Path, root: Path) -> Iterator[Path]:
    """A temporary path to write `final` to; renamed into place when the block ends cleanly.

    The temporary name keeps the file's suffix, so writers that pick a format by
    suffix (imageio) still do. The file is flushed to disk before the rename and
    the directory after it, so that the order of a job's files (the chain last)
    holds on disk as well, also across a power loss.
    """
    staging = root / STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"{uuid.uuid4().hex}.{final.name}"
    try:
        yield tmp
        final.parent.mkdir(parents=True, exist_ok=True)
        _fsync(tmp)
        os.replace(tmp, final)
        _fsync(final.parent)
    finally:
        tmp.unlink(missing_ok=True)


def clear_staging(root: Path) -> int:
    """Delete what a killed writer left in the staging directory; returns how many files."""
    staging = root / STAGING_DIR
    if not staging.is_dir():
        return 0
    count = sum(1 for p in staging.rglob("*") if p.is_file() or p.is_symlink())
    shutil.rmtree(staging)
    return count


def write_parquet(
    data: pa.Table, final: Path, root: Path, metadata: dict[bytes, bytes] | None = None
) -> int:
    """Write a table as Parquet with zstd, atomically; returns the size in bytes."""
    if metadata:
        data = data.replace_schema_metadata({**(data.schema.metadata or {}), **metadata})
    with staged(final, root) as tmp:
        # Key-value metadata (the canary of a hidden table) is written with the stored schema.
        pq.write_table(data, tmp, compression="zstd", store_schema=bool(metadata))
    return final.stat().st_size


def write_text(text: str, final: Path, root: Path) -> int:
    with staged(final, root) as tmp:
        tmp.write_text(text)
    return final.stat().st_size


def plant_canary(root: Path) -> str:
    """The run's canary token, planted in `truth/canary.txt` on first use.

    A random marker that exists only on the hidden side of the run: in that
    file, in the metadata of every hidden Parquet table and in every staged
    part. Finding it anywhere an agent can read means something hidden got out.
    """
    existing = read_canary(root)
    if existing is not None:
        return existing
    token = f"CANARY-{secrets.token_hex(12)}"
    write_text(token + "\n", root / CANARY_FILE, root)
    return token


def read_canary(root: Path) -> str | None:
    path = root / CANARY_FILE
    if not path.is_file():
        return None
    token = path.read_text().strip()
    return token or None
