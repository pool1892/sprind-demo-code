"""The public surface: everything an analysis agent is shown, numbers included.

The export (its description, its data dictionary, every cell of every table),
the tools' names, descriptions and argument schemas, every file of the mounted
skills, and the audit's own briefs. The scanner reads it first
(`Scanner.learn_public`): what of the truth inventory it shows cannot convict a
session that merely repeats it.

Small tables are rendered as the text the tools return (rows as JSON objects
and as the `columns`/`rows` answer of `query`), so that a name standing beside
a value in a tool answer stands beside it here too. Tables with one file per
episode (`steps`) hold millions of numbers; those are kept as typed columns and
compared by value, which asks the same question without printing them.

`declared_names` are the field names the contracts declare public (visible
columns, visible fields of the system model, keys of the tools' requests).
The scan takes a hidden value beside such a name for the public field's own
value; it takes them from the contracts, never from the export as found.

Not part of it: pixels. A frame or a video is not text, and the scan cannot
judge what an image shows.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.audit.scan import Scanner, fold, words_of
from fleetlab.contracts._base import Visibility, field_visibility
from fleetlab.contracts.log import visible_schema
from fleetlab.contracts.model import MODEL_CLASSES
from fleetlab.contracts.tools import TOOLS, json_schema_strict
from fleetlab.toolserver.server import Limits, tool_description

__all__ = ["Column", "Surface", "declared_names", "public_surface", "surface_hits", "tool_texts"]

# A table with more rows than this is compared as typed columns instead of text.
TEXT_ROWS = 50_000
ROWS_PER_TEXT = 400
TEXT_SUFFIXES = frozenset({".md", ".json", ".txt", ".csv", ".jsonl"})


@dataclass
class Column:
    label: str  # where a person finds it: "steps.time_s"
    name: str
    integers: np.ndarray[Any, Any] | None = None
    floats: np.ndarray[Any, Any] | None = None
    singles: np.ndarray[Any, Any] | None = None  # 32-bit floats, compared as such


@dataclass
class Surface:
    texts: list[tuple[str, str]] = field(default_factory=list)
    words: set[str] = field(default_factory=set)
    request_words: set[str] = field(default_factory=set)  # keys of the tools' argument schemas
    # Names the contracts declare public: visible columns, visible fields, request keys.
    schema_names: set[str] = field(default_factory=set)
    columns: list[Column] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, label: str, text: str) -> None:
        if text:
            self.texts.append((label, text))
            self.words |= words_of(text)

    def summary(self) -> dict[str, Any]:
        return {
            "texts": len(self.texts),
            "characters": sum(len(t) for _, t in self.texts),
            "typed_columns": len(self.columns),
            "values_in_typed_columns": int(
                sum(
                    len(a)
                    for c in self.columns
                    for a in (c.integers, c.floats, c.singles)
                    if a is not None
                )
            ),
            "words": len(self.words),
            "problems": self.problems,
        }


def tool_texts() -> dict[str, str]:
    """What an agent reads of each tool: name, description and argument schema."""
    limits = Limits()
    return {
        f"tool {spec.name}": json.dumps(
            {
                "name": spec.name,
                "description": tool_description(spec, limits),
                "inputSchema": json_schema_strict(spec.arguments),
            }
        )
        for spec in TOOLS
    }


def schema_keys() -> set[str]:
    """The property names of the tools' argument schemas, folded: what every request spells."""
    keys: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                keys.update(fold(name) for name in properties)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for spec in TOOLS:
        walk(json_schema_strict(spec.arguments))
    return keys


def declared_names() -> set[str]:
    """The field names the contracts declare public, folded: the visible columns of the
    log, the visible fields of the system model, and the keys of the tools' requests.

    Taken from the declarations and not from the export as it was found: a column the
    export holds without the contract declaring it visible is not one of these, so a
    hidden value beside it stays unresolved.
    """
    names = schema_keys()
    for table in visible_schema().values():
        names.update(fold(column.name) for column in table.columns)
    for model in MODEL_CLASSES:
        names.update(
            fold(name)
            for name in model.model_fields
            if field_visibility(model, name) is Visibility.visible
        )
    return names


def _flat(array: pa.ChunkedArray | pa.Array) -> pa.Array:
    data = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    while pa.types.is_list(data.type) or pa.types.is_fixed_size_list(data.type):
        data = data.flatten()
    return data.drop_null()


def _columns(
    table: pa.Table, name: str, into: dict[str, dict[str, list[np.ndarray[Any, Any]]]]
) -> list[str]:
    """Collect the numbers of `table` per column; returns the strings of its text columns."""
    strings: list[str] = []
    for column in table.schema.names:
        flat = _flat(table.column(column))
        kind = flat.type
        store = into.setdefault(f"{name}.{column}", {"integers": [], "floats": [], "singles": []})
        if pa.types.is_integer(kind):
            store["integers"].append(flat.to_numpy(zero_copy_only=False).astype(np.int64))
        elif pa.types.is_float32(kind):
            store["singles"].append(flat.to_numpy(zero_copy_only=False).astype(np.float32))
        elif pa.types.is_floating(kind) or pa.types.is_decimal(kind):
            store["floats"].append(flat.to_numpy(zero_copy_only=False).astype(np.float64))
        elif pa.types.is_string(kind) or pa.types.is_large_string(kind):
            strings.extend(set(flat.to_pylist()))
    return strings


def _table_texts(name: str, table: pa.Table) -> Iterable[tuple[str, str]]:
    rows = table.to_pylist()
    names = table.schema.names
    for start in range(0, len(rows), ROWS_PER_TEXT):
        part = rows[start : start + ROWS_PER_TEXT]
        label = f"table {name} rows {start}-{start + len(part) - 1}"
        yield label, "\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in part)
        answer = {
            "columns": names,
            "rows": [[r[n] for n in names] for r in part],
            "truncated": False,
        }
        yield label + " (as a query answer)", json.dumps(answer, default=str, ensure_ascii=False)


def public_surface(
    export: Path,
    skills: Path | None,
    briefs: Mapping[str, str],
    extra: Mapping[str, str] | None = None,
) -> Surface:
    surface = Surface(request_words=schema_keys(), schema_names=declared_names())
    log = export / "log"
    for path in (export / "data_dictionary.md", log / "description.json"):
        if path.is_file():
            surface.add(f"export {path.relative_to(export)}", path.read_text(errors="replace"))
        else:
            surface.problems.append(f"missing in the export: {path.relative_to(export)}")
    description = log / "description.json"
    if description.is_file():
        try:  # as `describe` prints it: compact, and with spaces
            data = json.loads(description.read_text())
            surface.add("export description, compact", json.dumps(data, separators=(",", ":")))
            surface.add("export description, spaced", json.dumps(data, ensure_ascii=False))
        except ValueError as error:
            surface.problems.append(f"description.json: {error}")
    numbers: dict[str, dict[str, list[np.ndarray[Any, Any]]]] = {}
    for path in sorted(log.rglob("*.parquet")) if log.is_dir() else []:
        relative = path.relative_to(log)
        name = relative.parts[0] if len(relative.parts) > 1 else path.stem
        try:
            table = pq.read_table(path)
        except Exception as error:  # an export file that is not a table is a finding of its own
            surface.problems.append(f"{relative}: {error}")
            continue
        surface.words |= {fold(n) for n in table.schema.names} | words_of(
            " ".join(table.schema.names)
        )
        strings = _columns(table, name, numbers)
        bulk = len(relative.parts) > 1 or table.num_rows > TEXT_ROWS
        if bulk:
            surface.add(
                f"table {name}: text cells of {relative}", "\n".join(sorted(map(str, strings)))
            )
            if relative == sorted(p.relative_to(log) for p in path.parent.glob("*.parquet"))[0]:
                for label, text in _table_texts(
                    f"{name} (first file)", table.slice(0, ROWS_PER_TEXT)
                ):
                    surface.add(label, text)
        else:
            for label, text in _table_texts(name, table):
                surface.add(label, text)
    for label, arrays in numbers.items():
        column = Column(label, label.split(".", 1)[1])
        for key in ("integers", "floats", "singles"):
            if arrays[key]:
                values = np.unique(np.concatenate(arrays[key]))
                setattr(column, key, values[np.isfinite(values)] if key != "integers" else values)
        surface.columns.append(column)
    # Every other file of the export that is text (a video is not).
    known = (export / "data_dictionary.md", description)
    for path in sorted(export.rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path not in known:
            surface.add(f"export {path.relative_to(export)}", path.read_text(errors="replace"))
    surface.add(
        "export file names",
        "\n".join(p.relative_to(export).as_posix() for p in sorted(export.rglob("*"))),
    )
    for label, text in tool_texts().items():
        surface.add(label, text)
    if skills is not None:
        for path in sorted(skills.rglob("*")):
            if path.is_file():
                surface.add(f"skill {path.relative_to(skills)}", path.read_text(errors="replace"))
        surface.add(
            "skill file names",
            "\n".join(p.relative_to(skills).as_posix() for p in sorted(skills.rglob("*"))),
        )
    for label, text in {**briefs, **(extra or {})}.items():
        surface.add(label, text)
    return surface


def surface_hits(scanner: Scanner, surface: Surface) -> list[dict[str, str]]:
    """The scan of the public surface itself. Must be empty: part (a) of the self-test."""
    hits = []
    for label, text in surface.texts:
        for cand, bare in scanner.scan(text):
            hits.append(
                {
                    "kind": cand.fact.kind,
                    "value": cand.fact.value,
                    "where": label,
                    "counted": "alone" if bare else f"beside {cand.binder!r}",
                    "text": cand.context,
                }
            )
    # The typed columns: no value that still counts alone may equal a cell.
    known = dict(scanner.public_values)
    pairs = dict(scanner.public_pairs)
    try:
        scanner.public_values, scanner.public_pairs = {}, {}
        scanner._learn_columns(surface.columns)
        for (key, exactness), sources in scanner.public_values.items():
            if (key, exactness) not in known:
                hits.append(
                    {
                        "kind": key[0],
                        "value": key[1],
                        "where": ", ".join(sorted(sources)),
                        "counted": "alone",
                        "text": "",
                    }
                )
    finally:
        scanner.public_values, scanner.public_pairs = known, pairs
    return hits
