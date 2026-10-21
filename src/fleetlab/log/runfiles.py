"""Reading a whole run directory, hidden tables included: for the oracle, tests and the dashboard.

Never for an agent: these functions read `truth/`. The agents' reader is
`fleetlab.log.reader.LogReader` over an export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.log import SCHEMA, type_name
from fleetlab.contracts.oracle import OracleRow, fault_list_hash
from fleetlab.contracts.rollout import RunLayout
from fleetlab.log._rows import table_schema, to_arrow
from fleetlab.log.reader import LogReader

__all__ = [
    "full_data_dictionary",
    "open_run",
    "oracle_rows",
    "read_chain",
    "read_identities",
    "read_table",
    "table_path",
]


def table_path(run_dir: Path | str, table: str, key: str | None = None) -> Path:
    """The file of a table; per-episode tables (steps, chain) need the episode id or job key."""
    return RunLayout(Path(run_dir)).table(table, key)


def read_chain(run_dir: Path | str, key: str) -> dict[str, npt.NDArray[Any]]:
    """A job's chain as stored: one array per column."""
    with np.load(table_path(run_dir, "chain", key)) as data:
        return {name: data[name] for name in data.files}


def read_table(run_dir: Path | str, table: str, key: str | None = None) -> pa.Table:
    """Any table of the schema as written, hidden tables included, without file metadata."""
    if SCHEMA[table].storage == "npz":
        if key is None:
            raise ValueError(f"table {table} is stored one file per job: give the job key")
        arrays = read_chain(run_dir, key)
        schema = table_schema(table)
        columns = []
        for name, kind in zip(schema.names, schema.types, strict=True):
            array = arrays[name]
            if pa.types.is_fixed_size_list(kind):
                flat = pa.array(array.reshape(-1), type=kind.value_type)
                columns.append(pa.FixedSizeListArray.from_arrays(flat, kind.list_size))
            else:
                columns.append(pa.array(array.tolist(), type=kind))
        return pa.Table.from_arrays(columns, schema=schema)
    data: pa.Table = pq.read_table(table_path(run_dir, table, key))
    return to_arrow(table, data.replace_schema_metadata(None))


def read_identities(run_dir: Path | str) -> list[EpisodeIdentity]:
    """The identities of a run's episodes, in episode order."""
    data = read_table(run_dir, "identities")
    return [
        EpisodeIdentity.model_validate_json(text)
        for text in data.column("identity_json").to_pylist()
    ]


def oracle_rows(run_dir: Path | str) -> list[OracleRow]:
    """The run's outcomes by effective fault list: each episode as run, then each replay.

    An episode as run counts with its true verdict (before the data hook)
    under its full fault list; a replay with the outcome it recorded.
    """
    identities = {i.episode_id: i for i in read_identities(run_dir)}
    rows = [
        OracleRow(
            episode_id=v["episode_id"],
            active_faults=fault_list_hash(identities[v["episode_id"]].faults),
            success=v["true_success"],
        )
        for v in read_table(run_dir, "verdicts").to_pylist()
    ]
    rows += [OracleRow.model_validate(r) for r in read_table(run_dir, "replays").to_pylist()]
    return rows


def open_run(run_dir: Path | str, *, materialize: bool = False) -> LogReader:
    """A DuckDB reader over a run with views of its hidden Parquet tables as well (dashboard)."""
    reader = LogReader(run_dir, materialize=materialize)
    root = reader.root
    kind = "TABLE" if materialize else "VIEW"
    for name, spec in SCHEMA.items():
        if spec.tree != "truth" or spec.storage != "parquet" or spec.file_key is not None:
            continue
        path = root / spec.path
        if name == "decisions" and path.is_file():
            # Normalize additive truth columns in legacy runs without rewriting the record.
            reader.con.register("_decisions", read_table(root, name))
            reader.con.execute("CREATE TABLE decisions AS SELECT * FROM _decisions")
            reader.con.unregister("_decisions")
        elif path.is_file():
            source = "'" + str(path).replace("'", "''") + "'"
            reader.con.execute(f"CREATE {kind} {name} AS SELECT * FROM read_parquet({source})")
        else:
            reader.con.register("_empty", table_schema(name).empty_table())
            reader.con.execute(f"CREATE TABLE {name} AS SELECT * FROM _empty")
            reader.con.unregister("_empty")
        reader.tables = (*reader.tables, name)
    return reader


def full_data_dictionary() -> str:
    """Every table and column of the schema with its visibility: for people, never for agents.

    The agents' dictionary (`fleetlab.log.dictionary.data_dictionary`) lists the
    visible columns only.
    """
    lines = ["# Episode log: every table and column", ""]
    for table in SCHEMA.values():
        lines += [
            f"## {table.name} ({table.tree})",
            "",
            f"`{table.path}`: {table.description}.",
            "",
        ]
        lines += ["| column | type | visibility | clock | description |", "|---|---|---|---|---|"]
        lines += [
            f"| {c.name} | {type_name(c.type)} | {c.visibility.value} | {c.clock or ''} "
            f"| {c.description} |"
            for c in table.columns
        ]
        lines.append("")
    return "\n".join(lines)
