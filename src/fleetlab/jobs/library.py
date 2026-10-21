"""The installation's job library: saved jobs, checks and probe recipes, and checks run.

    LIB/jobs/<name>.json      a `SavedJob`: the frozen program with what it must achieve
    LIB/checks/<name>.json    a `CheckSpec`: one SELECT over the job tables, with parameters
    LIB/recipes/<name>.json   a probe recipe (story B, lane P's format), kept as given
    LIB/index.json            a `LibraryIndex`: kind, name, purpose and digest of each file
    LIB/release_exposure.jsonl  every release attempt, written before it runs (append only)

Every file is written atomically and the index is rebuilt after every save, from the files
themselves. A file's digest is the SHA-256 of its canonical JSON. Saving under an existing
name with another digest needs `replace=True`; saving the same content again changes
nothing.

The release exposure log keeps the reserved release starts honest. A release attempt is
written to it, with its job, program digest, request (base task and requested goals),
initial states and full start identities (seeds included) under its allocation profile,
before anything of it runs, and its end afterwards; nothing ever removes a line. A start
that any earlier attempt examined is no longer fresh, whatever request or program it ran
for, whether the attempt finished, and whether the job was saved again since
(`exposed_identities`; `exposed_release_starts` is the older test by request and initial
state). An entry written before the identities were recorded is read under the legacy
profile, the only one that existed then (`exposure_identities`).

`run_check` runs a check read-only over a job run's tree (or an export of several) the way
the tool server's `query` does: the tables copied into DuckDB's memory, file access off,
the configuration locked, one SELECT and nothing else; the parameters are bound by name
(`$name`), typed as the check declares them. There are no Python checks.

Nothing here loads the simulator.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetlab.contracts.hashing import canonical_json
from fleetlab.contracts.jobs import (
    JOB_NAME_PATTERN,
    CheckParameterType,
    CheckSpec,
    LibraryEntry,
    LibraryIndex,
    LibraryKind,
    ReleaseReceipt,
    SavedJob,
    StartIdentity,
)
from fleetlab.jobs.starts import start_for, starts_for

__all__ = [
    "EXPOSURE_FILE",
    "CheckError",
    "LibraryError",
    "carry_exposure",
    "digest_of",
    "exposed_identities",
    "exposed_release_starts",
    "exposure_identities",
    "finish_release_exposure",
    "list_entries",
    "load_check",
    "load_index",
    "load_job",
    "load_recipe",
    "parse_params",
    "record_release_exposure",
    "release_exposure",
    "run_check",
    "run_saved_check",
    "save_check",
    "save_job",
    "save_recipe",
]

INDEX_FILE = "index.json"
EXPOSURE_FILE = "release_exposure.jsonl"
_DIRS = {LibraryKind.job: "jobs", LibraryKind.check: "checks", LibraryKind.recipe: "recipes"}


class LibraryError(ValueError):
    """A save the library refuses, or an entry it does not hold."""


class CheckError(ValueError):
    """A check that cannot run: not one SELECT, or parameters that do not fit."""


def digest_of(value: Any) -> str:
    """SHA-256 (hex) of a JSON value's canonical form."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _path(lib: Path, kind: LibraryKind, name: str) -> Path:
    import re

    if not re.fullmatch(JOB_NAME_PATTERN, name):
        raise LibraryError(f"{name!r} is not a library name ({JOB_NAME_PATTERN})")
    return Path(lib) / _DIRS[kind] / f"{name}.json"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save(lib: Path, kind: LibraryKind, name: str, value: Any, replace: bool) -> Path:
    path = _path(lib, kind, name)
    digest = digest_of(value)
    if path.is_file():
        old = digest_of(json.loads(path.read_text()))
        if old == digest:
            return path
        if not replace:
            raise LibraryError(
                f"the library holds another {kind.value} named {name!r} (digest {old[:12]}); "
                "save it under a new name, or replace it explicitly"
            )
    _atomic_write(path, json.dumps(value, indent=1, sort_keys=True) + "\n")
    _rebuild_index(Path(lib))
    return path


def _purpose(kind: LibraryKind, data: Mapping[str, Any]) -> str:
    if kind is LibraryKind.job:
        return str(data["program"]["purpose"])
    return str(data.get("purpose", ""))


def _rebuild_index(lib: Path) -> LibraryIndex:
    entries = []
    for kind, directory in _DIRS.items():
        for path in sorted((lib / directory).glob("*.json")):
            data = json.loads(path.read_text())
            entries.append(
                LibraryEntry(
                    kind=kind,
                    name=path.stem,
                    purpose=_purpose(kind, data),
                    digest=digest_of(data),
                )
            )
    index = LibraryIndex(entries=entries)
    _atomic_write(lib / INDEX_FILE, index.model_dump_json(indent=1) + "\n")
    return index


def save_job(lib: Path, job: SavedJob, *, replace: bool = False) -> Path:
    """Save a job under its name; returns the file."""
    return _save(lib, LibraryKind.job, job.name, job.model_dump(mode="json"), replace)


def load_job(lib: Path, name: str) -> SavedJob:
    path = _path(lib, LibraryKind.job, name)
    if not path.is_file():
        raise LibraryError(f"the library has no job {name!r}")
    return SavedJob.model_validate_json(path.read_text())


def save_check(lib: Path, spec: CheckSpec, *, replace: bool = False) -> Path:
    """Save a check under its name; returns the file."""
    return _save(lib, LibraryKind.check, spec.name, spec.model_dump(mode="json"), replace)


def load_check(lib: Path, name: str) -> CheckSpec:
    path = _path(lib, LibraryKind.check, name)
    if not path.is_file():
        raise LibraryError(f"the library has no check {name!r}")
    return CheckSpec.model_validate_json(path.read_text())


def save_recipe(
    lib: Path, name: str, purpose: str, data: Mapping[str, Any], *, replace: bool = False
) -> Path:
    """Save a probe recipe (story B) as given, with its purpose; returns the file."""
    value = {**dict(data), "name": name, "purpose": purpose}
    json.loads(canonical_json(value))  # JSON only
    return _save(lib, LibraryKind.recipe, name, value, replace)


def load_recipe(lib: Path, name: str) -> dict[str, Any]:
    path = _path(lib, LibraryKind.recipe, name)
    if not path.is_file():
        raise LibraryError(f"the library has no recipe {name!r}")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def load_index(lib: Path) -> LibraryIndex:
    """The library's index; rebuilt from the files when it is missing."""
    path = Path(lib) / INDEX_FILE
    if not path.is_file():
        return _rebuild_index(Path(lib)) if Path(lib).is_dir() else LibraryIndex(entries=[])
    return LibraryIndex.model_validate_json(path.read_text())


def list_entries(lib: Path, kind: LibraryKind | None = None) -> list[LibraryEntry]:
    return [e for e in load_index(lib).entries if kind is None or e.kind is kind]


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _typed(name: str, kind: CheckParameterType, value: Any) -> Any:
    try:
        if kind is CheckParameterType.int_:
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if kind is CheckParameterType.float_:
            return float(value)
        if kind is CheckParameterType.bool_:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE | _FALSE:
                return text in _TRUE
            raise ValueError
        return str(value)
    except (TypeError, ValueError):
        raise CheckError(f"parameter {name}: {value!r} is not a {kind.value}") from None


def parse_params(spec: CheckSpec, pairs: Sequence[str]) -> dict[str, Any]:
    """`k=v` arguments of the command line as the check's typed parameters."""
    given: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise CheckError(f"a parameter is name=value, not {pair!r}")
        given[name.strip()] = value
    return typed_params(spec, given)


def typed_params(spec: CheckSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    declared = {p.name: p.type for p in spec.parameters}
    unknown = sorted(set(values) - set(declared))
    missing = sorted(set(declared) - set(values))
    if unknown:
        raise CheckError(f"{spec.name} takes no parameter {', '.join(unknown)}")
    if missing:
        raise CheckError(f"{spec.name} needs {', '.join(missing)}")
    return {name: _typed(name, declared[name], values[name]) for name in declared}


def run_check(
    spec: CheckSpec, run_dir: Path, params: Mapping[str, Any] | None = None, *, max_rows: int = 500
) -> list[dict[str, Any]]:
    """Run a check over a job run's tree, read-only, and return its rows (at most `max_rows`)."""
    import duckdb

    from fleetlab.log import LogReader

    values = typed_params(spec, params or {})
    with LogReader(run_dir, materialize=True, lock=True) as reader:
        con = reader.con
        statements = con.extract_statements(spec.sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise CheckError(f"{spec.name}: a check is one SELECT")
        cursor = con.execute(spec.sql, values) if values else con.execute(spec.sql)
        columns = [c[0] for c in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows) if cursor.description else []
    return [dict(zip(columns, row, strict=True)) for row in rows]


def run_saved_check(
    lib: Path, name: str, run_dir: Path, params: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Section 4's `run_check(name, run, params) -> rows`: a saved check by its name."""
    return run_check(load_check(lib, name), run_dir, params)


def _append_durably(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _request_key(job: SavedJob) -> dict[str, Any]:
    return {"base_task": job.base_task, "goals": [g.text for g in job.requested_goals]}


def record_release_exposure(
    lib: Path, job: SavedJob, starts: Sequence[StartIdentity], run_dir: Path
) -> str:
    """Write a release attempt to the exposure log before it runs; returns its id.

    The starts are the job's own: its base task's, derived under its allocation profile.
    """
    if not all(s.is_release for s in starts):
        raise LibraryError("only release starts are recorded as exposed")
    profile = job.allocation_profile
    foreign = [
        s.init_state for s in starts if s != start_for(job.base_task, s.init_state, profile=profile)
    ]
    if foreign:
        raise LibraryError(
            f"starts {foreign} are not the job's: its allocation gives other identities"
        )
    attempt = uuid.uuid4().hex
    _append_durably(
        Path(lib) / EXPOSURE_FILE,
        {
            "attempt": attempt,
            "event": "started",
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "job_name": job.name,
            "program_digest": job.program.digest,
            **_request_key(job),
            "init_states": [s.init_state for s in starts],
            "allocation_profile": profile,
            "starts": [s.model_dump(mode="json") for s in starts],
            "run_dir": str(run_dir),
        },
    )
    return attempt


def finish_release_exposure(lib: Path, attempt: str, receipt: ReleaseReceipt) -> None:
    """Write the end of a recorded release attempt, with its receipt's verdict."""
    _append_durably(
        Path(lib) / EXPOSURE_FILE,
        {
            "attempt": attempt,
            "event": "finished",
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "receipt_digest": digest_of(receipt.model_dump(mode="json")),
            "passed": receipt.passed,
            "released": receipt.released,
        },
    )


def release_exposure(lib: Path) -> list[dict[str, Any]]:
    """The exposure log, every line, in order."""
    path = Path(lib) / EXPOSURE_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exposed_release_starts(lib: Path, job: SavedJob) -> set[int]:
    """The release initial states that earlier attempts examined for this job's request."""
    key = _request_key(job)
    out: set[int] = set()
    for entry in release_exposure(lib):
        if entry.get("event") != "started":
            continue
        if {"base_task": entry.get("base_task"), "goals": entry.get("goals")} == key:
            out.update(int(k) for k in entry.get("init_states", []))
    return out


def exposure_identities(entry: Mapping[str, Any]) -> list[StartIdentity]:
    """The full start identities one "started" line of an exposure log examined.

    A line records them since 20 October (`starts`, with its `allocation_profile`); an
    older line has its base task and initial states only, and is read under the legacy
    profile, the one that gave every start then. A line whose identities disagree with its
    own initial states is refused.
    """
    if entry.get("event") != "started":
        return []
    states = [int(k) for k in entry.get("init_states", [])]
    if "starts" not in entry:
        return starts_for(int(entry["base_task"]), states, profile=None)
    found = [StartIdentity.model_validate(s) for s in entry["starts"]]
    if [s.init_state for s in found] != states or any(
        s.base_task != entry.get("base_task") for s in found
    ):
        raise LibraryError(f"exposure attempt {entry.get('attempt')}: starts and states differ")
    return found


def exposed_identities(lib: Path) -> set[StartIdentity]:
    """Every start identity any earlier release attempt of the library examined, for any
    request, finished or not."""
    return {s for entry in release_exposure(lib) for s in exposure_identities(entry)}


def carry_exposure(source: Path, target: Path) -> int:
    """Append the lines of `source`'s exposure log that `target`'s does not hold yet, in
    their order and byte for byte; returns how many. Nothing is removed or rewritten."""
    path = Path(source) / EXPOSURE_FILE
    if not path.is_file():
        return 0
    into = Path(target) / EXPOSURE_FILE
    held = set(into.read_text().splitlines()) if into.is_file() else set()
    new = [line for line in path.read_text().splitlines() if line.strip() and line not in held]
    if new:
        into.parent.mkdir(parents=True, exist_ok=True)
        with into.open("a") as f:
            f.write("".join(line + "\n" for line in new))
            f.flush()
            os.fsync(f.fileno())
    return len(new)
