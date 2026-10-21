"""Rows of the episode log: checked against `contracts.log.SCHEMA` and turned into Arrow.

A row is a mapping from column name to value. A row for a table of the
visible tree may only carry the columns on that table's allow-list
(`contracts.log.allowed_columns`); a column the table declares as hidden, or
one it does not declare at all, is refused, and so is a row that lacks a
column. Types are checked by Arrow's conversion to the declared type.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from fleetlab.contracts.log import SCHEMA, ColumnSpec, Visibility, allowed_columns

__all__ = [
    "EPISODE_TABLES",
    "PER_EPISODE_TABLES",
    "RUN_TABLES",
    "RowError",
    "Rows",
    "check_keys",
    "decode_part",
    "encode_part",
    "file_columns",
    "sort_table",
    "table_columns",
    "table_schema",
    "to_arrow",
]

type Rows = Iterable[Mapping[str, Any]] | pa.Table

# Additive truth-only metadata. All original timing/decision columns remain required.
DECISION_DEFAULTS: dict[str, Any] = {
    "rationale": "",
    "call_status": "ok",
    "error": None,
    "retry_budget_steps": None,
}


class RowError(ValueError):
    """A row that does not fit its table in `contracts.log.SCHEMA`."""


def _has_episode_id(table: str) -> bool:
    return any(c.name == "episode_id" for c in SCHEMA[table].columns)


# The tables a run writes; an export's own tables (`TableSpec.source`) are not among them.
_RUN_SCHEMA = tuple(t for t in SCHEMA.values() if t.source == "run")
# Tables stored as one file per episode (or per job): written by the episode writer.
PER_EPISODE_TABLES: tuple[str, ...] = tuple(t.name for t in _RUN_SCHEMA if t.file_key is not None)
# Run-level tables whose rows belong to one episode: the episode writer stages its rows,
# `RunWriter.finalize` builds the table from the finished jobs.
EPISODE_TABLES: tuple[str, ...] = tuple(
    t.name for t in _RUN_SCHEMA if t.file_key is None and _has_episode_id(t.name)
)
# Run-level tables that belong to no episode (health, logs, gains): written whole.
RUN_TABLES: tuple[str, ...] = tuple(
    t.name for t in _RUN_SCHEMA if t.file_key is None and not _has_episode_id(t.name)
)

# The order rows are stored in, so that a table does not depend on the order jobs finished in.
_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "episodes": ("episode_id",),
    "events": ("episode_id", "step", "kind", "detail"),
    "identities": ("episode_id",),
    "faults_active": ("episode_id", "fault_id"),
    "effective_policy": ("episode_id",),
    "verdicts": ("episode_id",),
    "replays": ("episode_id", "active_faults", "success"),
    "decisions": ("episode_id", "step", "decision"),
    "job_outcomes": ("episode_id", "job_key"),
    "probe_episodes": ("probe_id", "episode_id"),
    "probe_segments": ("probe_id", "segment"),
}


def table_columns(table: str) -> tuple[ColumnSpec, ...]:
    """The row columns a table is written with: the allow-list for the visible tree, all
    for truth. A per-file column (`ColumnSpec.per_file`) is not a row column."""
    spec = SCHEMA[table]
    if spec.tree == "log":
        allowed = set(allowed_columns(table))
        return tuple(c for c in spec.columns if c.name in allowed and not c.per_file)
    return tuple(c for c in spec.columns if c.visibility is Visibility.hidden and not c.per_file)


def file_columns(table: str) -> tuple[ColumnSpec, ...]:
    """The per-file columns of a table: one value per file (the chain's fixture hash)."""
    return tuple(c for c in SCHEMA[table].columns if c.per_file)


def table_schema(table: str) -> pa.Schema:
    """The Arrow schema a table is written with."""
    return pa.schema([pa.field(c.name, c.type) for c in table_columns(table)])


def check_keys(table: str, keys: Iterable[str]) -> None:
    """Raise RowError unless `keys` are exactly the columns `table` is written with."""
    if table not in SCHEMA:
        raise RowError(f"no table {table!r} in contracts.log.SCHEMA")
    names = [c.name for c in table_columns(table)]
    given = list(keys)
    problems: list[str] = []
    declared = {c.name: c for c in SCHEMA[table].columns}
    for key in given:
        if key in names:
            continue
        if key in declared:
            problems.append(
                f"column {key!r} is tagged {declared[key].visibility.value} and table {table} "
                "is visible: it is not on the allow-list"
            )
        else:
            problems.append(f"table {table} has no column {key!r}")
    missing = [n for n in names if n not in given]
    if missing:
        problems.append(f"row for {table} lacks {missing}")
    if problems:
        raise RowError("; ".join(problems))


def to_arrow(table: str, rows: Rows) -> pa.Table:
    """Check rows against the schema and convert them to an Arrow table of the declared types."""
    schema = table_schema(table)
    if isinstance(rows, pa.Table):
        if table == "decisions":
            for name, value in DECISION_DEFAULTS.items():
                if name not in rows.column_names:
                    rows = rows.append_column(
                        name, pa.array([value] * rows.num_rows, schema.field(name).type)
                    )
        check_keys(table, rows.column_names)
        try:
            return rows.select(schema.names).cast(schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as error:
            raise RowError(f"{table}: {error}") from error
    listed = [dict(r) for r in rows]
    if table == "decisions":
        for row in listed:
            for name, value in DECISION_DEFAULTS.items():
                row.setdefault(name, value)
    for row in listed:
        check_keys(table, row)
    try:
        return pa.Table.from_pylist(listed, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as error:
        raise RowError(f"{table}: {error}") from error


def sort_table(table: str, data: pa.Table) -> pa.Table:
    """Rows in their stored order (by episode, then step), independent of arrival order."""
    keys = _SORT_KEYS.get(table)
    if not keys or data.num_rows < 2:
        return data
    return data.sort_by([(k, "ascending") for k in keys])


def vector(value: Any, length: int) -> list[float] | None:
    """A fixed-length vector as a list of floats; None stays None."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,):
        raise RowError(f"expected {length} values, got shape {np.shape(value)}")
    return [float(x) for x in array]


def action(value: Any) -> list[float] | None:
    """A 7-vector action as a list, or None when there is none (None or all NaN)."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (7,):
        raise RowError(f"an action has 7 values, got shape {np.shape(value)}")
    if bool(np.all(np.isnan(array))):
        return None
    return [float(x) for x in array]


# Staged rows are kept as JSON between an episode's end and `finalize`. Python's JSON
# round-trips floats exactly; a time of day is the one type it lacks.


def _default(value: Any) -> Any:
    if isinstance(value, datetime.time):
        return {"__time__": value.isoformat()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot stage a {type(value).__name__}")


def _hook(obj: dict[str, Any]) -> Any:
    if set(obj) == {"__time__"}:
        return datetime.time.fromisoformat(obj["__time__"])
    return obj


def encode_part(part: Mapping[str, Any]) -> str:
    return json.dumps(part, default=_default, sort_keys=True)


def decode_part(text: str) -> dict[str, Any]:
    value = json.loads(text, object_hook=_hook)
    if not isinstance(value, dict):
        raise RowError("a staged part is a JSON object")
    return value


def pylist(data: pa.Table) -> list[dict[str, Any]]:
    """An Arrow table as rows of Python values."""
    rows: list[dict[str, Any]] = data.to_pylist()
    return rows


def column_names(table: str) -> Sequence[str]:
    return [c.name for c in table_columns(table)]
