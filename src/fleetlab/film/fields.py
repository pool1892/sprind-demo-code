"""Numbers in the narration: named placeholders bound to fields of recorded runs.

The script never carries a number of its own. It names one, `{release_delivered}`, and
`site/narration/fields.json` (`fleetlab.film-fields/1`) says what each name means and which
field of which record holds it:

    {"format": "fleetlab.film-fields/1",
     "records": {"job_one_release": "what the record is and where a run writes it", ...},
     "fields": {
       "release_delivered": {"record": "job_one_release", "field": "delivered",
                             "meaning": "release starts the frozen program delivered on"},
       "dev_delivered": {"record": "job_one_dev_episodes", "count_true": "delivered",
                         "where": {"job_id": "j0001"}, "meaning": ...},
       "dev_starts": {"record": "job_one_dev_episodes", "count_rows": true, ...},
       "job_one_model_minutes": {"record": "job_one", "field": "costs.model_seconds",
                                 "scale": 0.016666667, "format": "{:.0f}", ...}}}

A cut names the files that stand for the records (`fleetlab.film-records/1`):

    {"format": "fleetlab.film-records/1",
     "records": {"job_one": "~/.cache/.../job_run.json",
                 "job_one_dev_episodes": ["~/.cache/.../log/job_episodes.parquet"]}}

A `field` is a dotted path into a JSON record (list indexes as numbers, `key=value` for
the first object of a list with that value); `count_true`
counts the rows of a table (Parquet or JSON lines, several files joined) whose column is
true, `count_rows` counts the rows and `count_distinct` the distinct values of a column
(with a `field`, the rows are the list at that field of a JSON record); `where` keeps rows
whose columns equal the values.
Binding writes the script with every bound name replaced by its value and `bindings.json`
beside it: each value with its record, the field and the SHA-256 of the file it was read
from. A name the script uses that `fields.json` does not define is an error; a name whose
record the cut does not provide, or whose field is missing, stays in the text as
`{name}` and is listed as unbound, so a rehearsal can show it, and a final cut and a voice
recording refuse it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from dataclasses import field as _default
from pathlib import Path
from typing import Any

__all__ = [
    "BINDINGS_FORMAT",
    "FIELDS_FORMAT",
    "RECORDS_FORMAT",
    "Binding",
    "FieldError",
    "FieldSpec",
    "bind",
    "bind_script",
    "load_fields",
    "load_records",
    "placeholders",
]

FIELDS_FORMAT = "fleetlab.film-fields/1"
RECORDS_FORMAT = "fleetlab.film-records/1"
BINDINGS_FORMAT = "fleetlab.film-bindings/1"
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


class FieldError(ValueError):
    pass


def placeholders(text: str) -> list[str]:
    """The placeholder names in a text, in order of first appearance."""
    seen: list[str] = []
    for name in PLACEHOLDER.findall(text):
        if name not in seen:
            seen.append(name)
    return seen


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FieldSpec:
    name: str
    record: str
    meaning: str
    field: str | None = None  # a dotted path into a JSON record
    count_true: str | None = None  # count the rows whose column is true
    count_rows: bool = False  # count the rows
    count_distinct: str | None = None  # count the distinct values of a column
    where: dict[str, Any] = _default(default_factory=dict)
    scale: float | None = None
    format: str | None = None  # a Python format for the (scaled) value, e.g. "{:.0f}"

    @staticmethod
    def parse(name: str, data: object) -> FieldSpec:
        where = f"field {name}"
        if not isinstance(data, dict):
            raise FieldError(f"{where}: an object expected")
        unknown = set(data) - {
            "record", "meaning", "field", "count_true", "count_rows", "count_distinct",
            "where", "scale", "format", "pending",
        }  # fmt: skip
        if unknown:
            raise FieldError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
        record, meaning = data.get("record"), data.get("meaning")
        if not isinstance(record, str) or not record:
            raise FieldError(f"{where}: names its record")
        if not isinstance(meaning, str) or not meaning:
            raise FieldError(f"{where}: says what the number means")
        modes = [k for k in ("field", "count_true", "count_rows", "count_distinct") if data.get(k)]
        if sum(bool(data.get(k)) for k in ("count_true", "count_rows", "count_distinct")) > 1:
            raise FieldError(f"{where}: one of count_true, count_rows or count_distinct")
        if not modes and not data.get("pending"):
            raise FieldError(f"{where}: field, count_true, count_rows, or pending (why not yet)")
        filters = data.get("where") or {}
        if not isinstance(filters, dict):
            raise FieldError(f"{where}.where: an object expected")
        scale = data.get("scale")
        if scale is not None and (not isinstance(scale, int | float) or isinstance(scale, bool)):
            raise FieldError(f"{where}.scale: a number expected")
        fmt = data.get("format")
        if fmt is not None and not isinstance(fmt, str):
            raise FieldError(f"{where}.format: text expected")
        return FieldSpec(
            name=name,
            record=record,
            meaning=meaning,
            field=data.get("field") or None,
            count_true=data.get("count_true") or None,
            count_rows=bool(data.get("count_rows")),
            count_distinct=data.get("count_distinct") or None,
            where=dict(filters),
            scale=None if scale is None else float(scale),
            format=fmt,
        )

    @property
    def pending(self) -> bool:
        return (
            self.field is None
            and self.count_true is None
            and not self.count_rows
            and self.count_distinct is None
        )

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"record": self.record, "meaning": self.meaning}
        if self.field is not None:
            out["field"] = self.field
        if self.count_true is not None:
            out["count_true"] = self.count_true
        if self.count_rows:
            out["count_rows"] = True
        if self.count_distinct is not None:
            out["count_distinct"] = self.count_distinct
        if self.where:
            out["where"] = self.where
        return out


def load_fields(path: Path) -> dict[str, FieldSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != FIELDS_FORMAT:
        raise FieldError(f"{path} is not a {FIELDS_FORMAT} file")
    raw = data.get("fields")
    if not isinstance(raw, dict):
        raise FieldError(f"{path}: fields is an object of name to field")
    return {str(name): FieldSpec.parse(str(name), spec) for name, spec in raw.items()}


def load_records(path: Path) -> dict[str, list[Path]]:
    """The files standing for each record name; a relative path is relative to the file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != RECORDS_FORMAT:
        raise FieldError(f"{path} is not a {RECORDS_FORMAT} file")
    raw = data.get("records")
    if not isinstance(raw, dict):
        raise FieldError(f"{path}: records is an object of name to path")
    out: dict[str, list[Path]] = {}
    for name, value in raw.items():
        items = value if isinstance(value, list) else [value]
        if not items or not all(isinstance(v, str) and v for v in items):
            raise FieldError(f"{path}: record {name} is a path or a list of paths")
        paths = []
        for v in items:
            p = Path(v).expanduser()
            paths.append(p if p.is_absolute() else path.parent / p)
        out[str(name)] = paths
    return out


def _json_field(data: Any, dotted: str) -> Any:
    """A dotted path: object keys, list indexes, and `key=value` for the first object of a
    list whose `key` is `value` (`episodes.robot_id=r11.edges.0.first_movement_rows`)."""
    value = data
    for part in dotted.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, list) and "=" in part:
            key, _, wanted = part.partition("=")
            match = [v for v in value if isinstance(v, dict) and str(v.get(key)) == wanted]
            if not match:
                raise KeyError(part)
            value = match[0]
        else:
            raise KeyError(part)
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@dataclass
class Binding:
    values: dict[str, dict[str, Any]]  # name -> value, its source and meaning
    unbound: dict[str, str]  # name -> why it has no value
    files: dict[str, str]  # path -> SHA-256 of every record file read

    def text(self, text: str) -> str:
        """The text with every bound name replaced; unbound names stay as `{name}`."""

        def sub(match: re.Match[str]) -> str:
            name = match.group(1)
            bound = self.values.get(name)
            return match.group(0) if bound is None else str(bound["value"])

        return PLACEHOLDER.sub(sub, text)


def _value(spec: FieldSpec, files: list[Path], read: dict[str, str]) -> tuple[Any, str | None]:
    """The value and, when there is none, why."""
    if spec.pending:
        return None, "pending: the record format is not fixed yet"
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        return None, f"{', '.join(missing)} does not exist"
    for p in files:
        read[str(p)] = _sha256(p)
    counting = spec.count_rows or spec.count_true is not None or spec.count_distinct is not None
    value: Any = None
    if spec.field is not None:
        if len(files) != 1:
            return None, f"a field is read from one JSON record, not {len(files)} files"
        try:
            value = _json_field(json.loads(files[0].read_text(encoding="utf-8")), spec.field)
        except KeyError as error:
            return None, f"{files[0]} has no {spec.field} (at {error.args[0]!r})"
        except json.JSONDecodeError as error:
            return None, f"{files[0]} is not JSON ({error})"
    if counting:
        if spec.field is not None:  # the rows are the list at the field of a JSON record
            if not isinstance(value, list):
                return None, f"{spec.field} in {files[0]} is not a list"
            if not all(isinstance(r, dict) for r in value):
                if spec.where or not spec.count_rows:
                    return None, f"{spec.field} in {files[0]} is not a list of rows"
                value = [{"item": r} for r in value]  # a list of names: count its items
            rows = list(value)
        else:
            rows = [r for p in files for r in _rows(p)]
        rows = [r for r in rows if all(r.get(k) == v for k, v in spec.where.items())]
        if spec.count_rows:
            value = len(rows)
        elif spec.count_distinct is not None:
            value = len({json.dumps(r.get(spec.count_distinct), sort_keys=True) for r in rows})
        else:
            value = sum(1 for r in rows if r.get(spec.count_true) is True)
    if value is None:
        return None, f"{spec.field} is null in the record"
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None, f"{spec.field} is {type(value).__name__}, not a number or a word"
    if spec.scale is not None:
        if isinstance(value, str):
            return None, f"{spec.field} is text and cannot be scaled"
        value = value * spec.scale
    if spec.format is not None:
        value = spec.format.format(value)
    elif isinstance(value, float):
        value = f"{value:g}"
    return value, None


def bind(text: str, fields: dict[str, FieldSpec], records: dict[str, list[Path]]) -> Binding:
    """Bind every placeholder the script's beats speak; a name `fields` does not define is an
    error. The script's preamble and the editor's notes are not read aloud and not bound."""
    from fleetlab.film.narration import parse_script

    names: list[str] = []
    for beat in parse_script(text):
        names += [n for n in placeholders(beat.text) if n not in names]
    undefined = [n for n in names if n not in fields]
    if undefined:
        raise FieldError(
            "the script uses placeholder(s) no field defines: "
            + ", ".join("{" + n + "}" for n in undefined)
        )
    values: dict[str, dict[str, Any]] = {}
    unbound: dict[str, str] = {}
    read: dict[str, str] = {}
    for name in names:
        spec = fields[name]
        files = records.get(spec.record)
        if files is None:
            unbound[name] = f"this cut provides no record {spec.record}"
            continue
        value, why = _value(spec, files, read)
        if why is not None:
            unbound[name] = why
            continue
        values[name] = {
            "value": value,
            **spec.describe(),
            "files": [{"path": str(p), "sha256": read[str(p)]} for p in files],
        }
    return Binding(values, unbound, read)


def bind_script(
    script: Path, fields_path: Path, records_path: Path | None, out: Path
) -> tuple[Path, dict[str, Any]]:
    """Write `out/script.md`, the script with its placeholders bound, and `out/bindings.json`."""
    text = script.read_text(encoding="utf-8")
    fields = load_fields(fields_path)
    records = load_records(records_path) if records_path is not None else {}
    binding = bind(text, fields, records)
    out.mkdir(parents=True, exist_ok=True)
    bound = out / "script.md"
    bound.write_text(binding.text(text), encoding="utf-8")
    report = {
        "format": BINDINGS_FORMAT,
        "script": {"path": str(script), "sha256": _sha256(script)},
        "fields": {"path": str(fields_path), "sha256": _sha256(fields_path)},
        "records": (
            {"path": str(records_path), "sha256": _sha256(records_path)}
            if records_path is not None
            else None
        ),
        "bound_script_sha256": _sha256(bound),
        "values": binding.values,
        "unbound": binding.unbound,
        "files_read": binding.files,
    }
    (out / "bindings.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bound, report
