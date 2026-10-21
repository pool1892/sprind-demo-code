"""The word search over an export and the canary check, for the leak audit.

    report = audit_export(export_dir, run_dirs, fleet=fleet,
                          texts={"prompt": PROMPT}, code={"tool server": source})
    assert report.clean, report.summary()

Two lists of terms. `truth_terms` is what may not appear in the data an agent
reads: a few generic words about how the data was made (`GENERIC_TERMS`),
the truth-only schema and actual world/run-truth/oracle names, identity seeds,
fault identifiers, the fault kinds of the registry
(`fleetlab.faults.load_all()`) and of the runs (`faults_active`), the names of
the robots' hidden properties (from the trait producers even without a full fleet),
and each run's canary token. `TruthTerms` carries two matching classes: names
from hidden injection/execution contracts remain flexible, while report-record
keys have exact spelling and are checked only as names or labels. Public schema
fields are excluded even when a hidden record also uses them.
`code_terms` is the
narrower list for source code mounted into the sandbox (the tool server and
the `contracts` it imports): only what is specific to a world, the fault kinds,
property names and canaries, because the contracts' source names the hidden
tables by design. The two kinds of change an action can add
(`contracts.actions.ADDED_CHANGE_KINDS`) are the
known exception in code: the tool server imports them with `actions`. The generic
field name `traits` is also allowed in contract code, while staying forbidden in data.

The search (`leak_search`) covers file/directory names, decoded Parquet fields,
metadata and values, and supported text files. Opaque binary retains exact generic
terms of at least five characters; flexible names need eight letters/digits and
ASCII separators to limit incidental compressed-video matches. Decoded names have
the flexible class's case/separator forms, including camelCase. Fault kinds use identifier forms,
not spaced prose. Unsupported files/columns and symbolic links are hits. MP4 pixels
are outside the word search. Visible vocabulary and column names mask coincident
substrings; the description's vocabulary block is skipped only when it equals the
contract's own. Standalone callers with unregistered kinds must pass `kinds=`;
`audit_export` supplies every kind found in its world truth runs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import decimal
import hashlib
import inspect
import io
import json
import math
import re
import struct
import textwrap
import time
import uuid
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from fleetlab.contracts._base import field_visibility, visible_dump
from fleetlab.contracts.log import DESCRIPTION_FILE, SCHEMA, VIDEO_PATH, Visibility
from fleetlab.contracts.model import MODEL_CLASSES, Fleet, describe_for_agents
from fleetlab.contracts.vocabulary import is_vocabulary, vocabulary_json
from fleetlab.log._files import CANARY_METADATA_KEY, read_canary

__all__ = [
    "GENERIC_TERMS",
    "LeakHit",
    "LeakReport",
    "TruthTerms",
    "audit_export",
    "audit_identifier",
    "canary_problems",
    "classify_truth_terms",
    "code_terms",
    "export_digest",
    "hidden_field_terms",
    "leak_search",
    "matching_export_check",
    "search_texts",
    "truth_terms",
    "write_export_check",
]

# Words about how the data was made. Short ones are searched in text only.
GENERIC_TERMS: tuple[str, ...] = (
    "fault",
    "truth",
    "oracle",
    "replay",
    "inject",
    "canary",
    "severity",
    "decoy",
    "confounder",
    "counterfactual",
    "seed",
)
MIN_BYTES_TERM = 5
# Deliberate prose policy for established truth labels; discovering a key alone
# never makes a new single word forbidden in prose. Ordinary drift, level, outcome
# and interaction remain name-only, including when they occur in truth records.
_PROSE_TRUTH_NAMES = frozenset(
    {
        "nominal",
        "params",
        "schedule",
        "generator",
        "haze",
        "decoy",
        "inert",
        "confounder",
        "superadditive",
        "additive",
        "detectable",
        "planted",
        "injected",
        "counterfactual",
        "oracle",
        "verdict",
    }
)
# Flexible names need more evidence in opaque binary than in decoded text.
MIN_RAW_NAME_LETTERS = 8
_TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".txt",
    ".jsonl",
    ".csv",
    ".py",
    ".yaml",
    ".yml",
    ".log",
    ".html",
    ".toml",
}
# These words have ordinary fleet meanings. Still compare them as names and labels.
# Historical reserved labels. Newly discovered single words follow this same
# name-only policy automatically; only the explicit prose terms below are word terms.
_LABEL_ONLY_TERMS = frozenset(
    [
        "add",
        "added",
        "aim",
        "attempt",
        "attempts",
        "band",
        "brighter",
        "bytes",
        "broken",
        "calibration",
        "cameras",
        "carries",
        "centre",
        "classes",
        "cooler",
        "counts",
        "curves",
        "darker",
        "days",
        "decision",
        "done",
        "directions",
        "emphasis",
        "ended",
        "expected",
        "experiment",
        "export",
        "extra",
        "favoured",
        "fingers",
        "files",
        "first",
        "floor",
        "friction",
        "frames",
        "item",
        "joint",
        "jobs",
        "last",
        "logged",
        "margin",
        "mass",
        "measured",
        "mode",
        "motion",
        "negative",
        "net",
        "neutral",
        "observed",
        "ok",
        "out",
        "pair",
        "path",
        "plain",
        "policy",
        "precision",
        "radius",
        "raised",
        "rate",
        "ready",
        "remove",
        "removed",
        "rescued",
        "revision",
        "robot",
        "role",
        "roles",
        "row",
        "rule",
        "run",
        "runs",
        "scoring",
        "seconds",
        "shape",
        "share",
        "size",
        "site",
        "small",
        "sources",
        "strong",
        "step_s",
        "started",
        "success",
        "summary",
        "suite",
        "supervisor",
        "surfaces",
        "tables",
        "tasks",
        "text",
        "tint",
        "true",
        "unit",
        "video",
        "what",
        "when",
        "weights",
        "windows",
    ]
)


@dataclass(frozen=True)
class LeakHit:
    where: str
    term: str
    context: str


@dataclass
class LeakReport:
    """The terms searched, how much was searched, and every hit; clean when there is none."""

    terms: tuple[str, ...]
    searched: dict[str, int] = field(default_factory=dict)
    hits: list[LeakHit] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.hits

    def summary(self) -> str:
        shown = ["(a canary token)" if t.startswith("canary-") else t for t in self.terms]
        lines = [
            f"searched for {len(self.terms)} terms: {', '.join(shown)}",
            "in " + ", ".join(f"{v} {k}" for k, v in self.searched.items()),
            f"hits: {len(self.hits)}",
        ]
        lines += [f"  {h.where}: {h.term!r} in {h.context!r}" for h in self.hits]
        return "\n".join(lines)


class TruthTerms(list[str]):
    """A list-compatible vocabulary carrying the provenance of record names.

    Keep this object when calling ``leak_search``. Converting it to a plain list
    deliberately loses its matching policy, just as passing arbitrary terms does.
    """

    def __init__(
        self,
        terms: Iterable[str],
        *,
        structural: Iterable[str],
        labels: Iterable[str],
        planted: Iterable[str],
        spellings: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        super().__init__(_clean(terms))
        self.structural = frozenset(_clean(structural)) & set(self)
        self.labels = frozenset(_clean(labels)) & set(self)
        self.planted = frozenset(_clean(planted)) & set(self)
        self.spellings = {t: frozenset((spellings or {}).get(t, (t,))) for t in self.structural}


def _planted_field_names() -> set[str]:
    """Names defined by the hidden data and injection contracts, not report keys.

    Fault parameters and mappings describe planted mechanisms. Identity/job
    fields and hidden log columns describe their execution evidence. In contrast,
    a dictionary emitted by a report producer only defines record structure.
    """
    from fleetlab.contracts import faults, identity, rollout

    names = _hidden_columns() | _hidden_model_fields()
    names.update(t.name for t in SCHEMA.values() if t.tree == "truth")
    for module in (faults, identity, rollout):
        for model in _defined_models(module):
            names.update(_schema_names(model))
    import fleetlab.faults

    for kind in fleetlab.faults.REGISTRY.kinds():
        fault = fleetlab.faults.REGISTRY.get(kind)
        names.update(_schema_names(fault.Params))
        names.update(_schema_names(type(fault.truth)))
    return names - _visible_names()


def _defined_models(module: Any) -> set[type[BaseModel]]:
    """Discover declarations in a defining module, including future record models."""
    return {
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == module.__name__
    }


def _record_structure_names() -> set[str]:
    """Structure explicitly established by execution schemas or report producers.

    Recipes prescribe planted worlds, rather than reporting their structure, so
    their fields do not establish this exemption. An undeclared name stays strict.
    """
    from fleetlab.contracts import world as world_contract
    from fleetlab.engine.experiments import WorldTruth
    from fleetlab.worlds import base, compose, detect, summary, truth, world

    names = _planted_field_names() - _hidden_model_fields()
    names.update(_LABEL_ONLY_TERMS)
    names.update(_record_keys(world.generator_record))
    names.update(_record_keys(base.Emphasis.as_json))
    names.update(_record_keys(detect))
    names.update(_record_keys(summary))
    names.update(_record_keys(compose))
    names.update(_future_truth_names())
    models = _defined_models(world_contract) | _defined_models(truth)
    models.add(WorldTruth)
    for model in models:
        names.update(_schema_names(model))
    return set(_clean(names))


def _agent_planted_names(fleet: Fleet | None, kinds: Iterable[str]) -> set[str]:
    """Mechanism/role names and hidden fleet properties, not execution records.

    Pattern labels come from the role contract and assignments to the composer's
    role mapping. Job/identity fields, log columns and fault-parameter schemas
    describe truth records; those keys cannot establish a leak in agent prose.
    """
    from fleetlab.worlds import compose, truth

    roles = set(get_args(truth.Role))
    tree = ast.parse(inspect.getsource(compose))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "roles"
            for target in node.targets
        ):
            roles.update(
                value.value
                for value in ast.walk(node.value)
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            )
    return set(_clean([*_hidden_model_fields(), *_trait_terms(fleet), *kinds, *roles]))


def _recipe_field_names() -> set[str]:
    """Declared recipe configuration is not a mechanism name in agent prose.

    Keep these fields strict on our texts. Their defining schemas only establish
    that they are known configuration keys, rather than unknown planted names.
    """
    from fleetlab.worlds import recipe

    return set(_clean(name for model in _defined_models(recipe) for name in _schema_names(model)))


@cache
def _vocabulary_names() -> tuple[str, ...]:
    """Every name of the visible vocabulary, longest first, lower case."""
    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            names.update(str(k) for k in value)
            for key, item in value.items():
                if key == "name" and isinstance(item, str):
                    names.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(vocabulary_json())
    return tuple(sorted((n.lower() for n in names if len(n) > 2), key=lambda n: (-len(n), n)))


@cache
def _mask_text() -> re.Pattern[str]:
    """Occurrences of vocabulary names; a term found inside one does not count."""
    names = {
        *_vocabulary_names(),
        *(c.name for t in SCHEMA.values() for c in t.columns if c.visibility is Visibility.visible),
    }
    return re.compile("|".join(re.escape(n) for n in sorted(names, key=lambda n: (-len(n), n))))


@cache
def _mask_bytes() -> re.Pattern[bytes]:
    return re.compile(_mask_text().pattern.encode())


def _hidden_columns() -> set[str]:
    visible = {
        c.name for t in SCHEMA.values() for c in t.columns if c.visibility is Visibility.visible
    }
    hidden = {
        c.name for t in SCHEMA.values() for c in t.columns if c.visibility is Visibility.hidden
    }
    return hidden - visible


def hidden_column_terms() -> list[str]:
    """Compound hidden column names, suitable for callers that search ordinary prose.

    Kept compatible with the skill validator. The export audit derives all names from
    `_hidden_columns`, with context-aware matching for ordinary single-word fields.
    """
    visible = set(_clean(_visible_names()))
    return sorted(name for name in _hidden_columns() if "_" in name and name not in visible)


def _hidden_model_fields() -> set[str]:
    return {
        name
        for model in MODEL_CLASSES
        for name in model.model_fields
        if field_visibility(model, name) is Visibility.hidden
    }


def _schema_names(model: type[BaseModel]) -> set[str]:
    """Serialized field names, including nested models and aliases, from their definitions."""
    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            names.update(value.get("properties", {}))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(model.model_json_schema(mode="serialization"))
    return names


def _record_keys(producer: Any) -> set[str]:
    """Keys of the generator's untyped truth record, read from its defining code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(producer)))
    names = {
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    names.update(
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    )
    return names


def _future_truth_names() -> set[str]:
    """Serialized oracle/scoring declarations, before any reader writes its files."""
    package = Path(__file__).parents[1]
    names: set[str] = set()
    for path in [*(package / "oracle").glob("*.py"), package / "scoring/models.py"]:
        tree = ast.parse(path.read_text())
        # Keyword fields fed into local record writers, rather than arbitrary
        # numerical-library or command-line options. The writers define their
        # **fields dictionary update; no second list of their emitted keys.
        record_writers = {
            function.name
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and function.args.kwarg is not None
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and any(
                    isinstance(argument, ast.Name) and argument.id == function.args.kwarg.arg
                    for argument in call.args
                )
                for call in ast.walk(function)
            )
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                names.update(
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
            elif isinstance(node, ast.ClassDef):
                names.update(
                    field.target.id
                    for field in node.body
                    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in record_writers
            ):
                names.update(keyword.arg for keyword in node.keywords if keyword.arg is not None)
    return names


def hidden_field_terms() -> list[str]:
    """Truth-only names from schemas, never a duplicate list of field names.

    Imports stay inside the audit: the mounted log reader must not import generators,
    the fault registry or the experiment engine. Shared visible fields are not secrets.
    Ordinary single-word field names are searched as labels, not as words in prose.
    """
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.rollout import Job
    from fleetlab.contracts.world import WorldSpec
    from fleetlab.engine.experiments import WorldTruth
    from fleetlab.worlds import base, detect, summary, world
    from fleetlab.worlds.recipe import WorldRecipe
    from fleetlab.worlds.truth import FaultTable

    visible = _visible_names()
    hidden = _hidden_model_fields()
    hidden.update(_planted_field_names())
    hidden.update(t.name for t in SCHEMA.values() if t.tree == "truth")
    hidden.update(_record_keys(world.generator_record))
    hidden.update(_record_keys(base.Emphasis.as_json))
    hidden.update(_record_keys(detect))
    hidden.update(_record_keys(summary))  # truth summary, not runner bookkeeping
    hidden.update(_future_truth_names())
    for truth_model in (
        EpisodeIdentity,
        Job,
        WorldSpec,
        WorldTruth,
        FaultTable,
        WorldRecipe,
    ):
        hidden.update(_schema_names(truth_model))
    import fleetlab.faults

    for kind in fleetlab.faults.REGISTRY.kinds():
        hidden.update(_schema_names(fleetlab.faults.REGISTRY.get(kind).Params))
        hidden.update(_schema_names(type(fleetlab.faults.REGISTRY.get(kind).truth)))
    public = set(_clean(visible))
    return sorted(n for n in hidden | _hidden_columns() if _clean([n])[0] not in public)


def _registry_kinds() -> list[str]:
    # Imported here, not at the top: fleetlab.faults is never mounted into a sandbox, and
    # importing this package must not need it.
    import fleetlab.faults

    return list(fleetlab.faults.load_all())


def _run_kinds(run_dir: Path) -> list[str]:
    path = run_dir / SCHEMA["faults_active"].path
    if not path.is_file():
        return []
    kinds: list[str] = pq.read_table(path, columns=["kind"]).column("kind").unique().to_pylist()
    return [k for k in kinds if k]


def _trait_terms(fleet: Fleet | None) -> list[str]:
    # Read the producers' dictionaries, not a copied list or the visible fleet. A newly
    # added key is covered even if the caller passes no fleet or a trait-free one.
    from fleetlab.worlds import base, generate

    keys = {
        *base.TRAIT_NAMES,
        *generate.TRAIT_NAMES,
        *base.traits_for(0, "r01"),
        *generate.traits_for(0, "r01"),
    }
    if fleet is not None:
        keys.update(key for robot in fleet.robots() for key in robot.traits)
    return sorted(keys)


def _known_code_exceptions() -> set[str]:
    from fleetlab.contracts.actions import ADDED_CHANGE_KINDS

    return set(ADDED_CHANGE_KINDS)


def _clean(terms: Iterable[str]) -> list[str]:
    def normalize(term: str) -> str:
        term = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", term.strip())
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", term).lower()

    return sorted({normalize(t) for t in terms if t and t.strip()})


def _visible_names(fleet: Fleet | None = None) -> set[str]:
    from fleetlab.agents.sessions import SessionReport
    from fleetlab.contracts import tools
    from fleetlab.contracts.log import EventKind, Outcome
    from fleetlab.contracts.skills import SkillManifest
    from fleetlab.toolserver.server import ToolBox

    names = {
        c.name for t in SCHEMA.values() for c in t.columns if c.visibility is Visibility.visible
    }
    names.update(_vocabulary_names())
    names.update(t.name for t in SCHEMA.values() if t.tree == "log")
    names.update(Path(VIDEO_PATH).parts[:-1])
    names.update(k.value for enum in (EventKind, Outcome) for k in enum)
    names.update(describe_for_agents(Fleet(id="w00", sites=[])))
    for model in MODEL_CLASSES:
        names.update(
            n for n in model.model_fields if field_visibility(model, n) is Visibility.visible
        )
    for name in tools.__all__:
        model = getattr(tools, name)
        if isinstance(model, type) and model is not BaseModel and issubclass(model, BaseModel):
            names.update(_schema_names(model))
            names.update(_clean([name]))  # declared public concepts, such as FixSet
    for tool in tools.TOOLS:
        names.update(_schema_names(tool.arguments))
    # The analysis session's declared answer schema is public too. Read its
    # definition, so a newly added answer field cannot become a truth term.
    names.update(_schema_names(SessionReport))
    names.update(_schema_names(SkillManifest))
    # The SQL reply's untyped public schema is defined by its producer.
    names.update(_record_keys(ToolBox._query))
    if fleet is not None:

        def keys(value: Any) -> None:
            if isinstance(value, dict):
                names.update(str(k) for k in value)
                for item in value.values():
                    keys(item)
            elif isinstance(value, list):
                for item in value:
                    keys(item)

        keys(describe_for_agents(fleet))
    return names


def _world_roots(run_dirs: Iterable[Path | str]) -> set[Path]:
    return {
        root
        for run in map(Path, run_dirs)
        for root in [run.parent.parent]
        if any((root / name / "spec.json").is_file() for name in ("files", "world"))
    }


def _truth_runs(run_dirs: Iterable[Path | str]) -> list[Path]:
    runs = set(map(Path, run_dirs))
    for root in _world_roots(runs):
        runs.update(p.parent for p in (root / "runs").glob("*/truth") if p.is_dir())
        runs.update(p.parent for p in (root / "oracle").rglob("truth") if p.is_dir())
    return sorted(runs)


def _truth_sources(run_dirs: Iterable[Path | str]) -> list[Path]:
    """Run truth and the world's host-only files, independent of their writer module."""
    sources: set[Path] = set()
    runs = list(map(Path, run_dirs))
    truth_runs = _truth_runs(runs)
    for run in truth_runs:
        sources.add(run / "truth")
    for root in _world_roots(runs):
        # Oracle reports are hidden, but replay runs still have a visible log
        # contract. Read each run's truth/record once, not its log or staging.
        sources.update(
            p
            for p in (root / "oracle").rglob("*")
            if p.is_file() and not any(p.is_relative_to(run) for run in truth_runs)
        )
        for name in ("files", "world"):
            if (root / name / "spec.json").is_file():
                sources.add(root / name)
    return sorted(
        {
            p
            for source in sources
            for p in (source.rglob("*") if source.is_dir() else [source])
            if p.is_file()
        }
    )


def _file_truth_names(run_dirs: Iterable[Path | str], fleet: Fleet | None = None) -> set[str]:
    names: set[str] = set()

    def keys(value: Any) -> None:
        if isinstance(value, dict):
            names.update(str(k) for k in value)
            for item in value.values():
                keys(item)
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            names.add(value[0])
            keys(value[1])
        elif isinstance(value, list | tuple):
            for item in value:
                keys(item)
        elif isinstance(value, bytes):
            keys(_decode_text(value))
        elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            with suppress(ValueError):
                keys(json.loads(value))

    def fields(dtype: pa.DataType) -> None:
        if isinstance(dtype, pa.BaseExtensionType):
            fields(dtype.storage_type)
        elif pa.types.is_struct(dtype) or pa.types.is_union(dtype):
            for child in dtype:
                names.add(child.name)
                names.update(_decode_text(k) for k in (child.metadata or {}))
                fields(child.type)
        elif (
            pa.types.is_dictionary(dtype)
            or pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
            or pa.types.is_list_view(dtype)
            or pa.types.is_large_list_view(dtype)
        ):
            fields(dtype.value_type)
        elif pa.types.is_map(dtype):
            fields(dtype.key_type)
            fields(dtype.item_type)

    for path in _truth_sources(run_dirs):
        if path.suffix.lower() == ".parquet":
            table = pq.read_table(path)
            names.update(
                _decode_text(k) for k in (table.schema.metadata or {}) if k != b"ARROW:schema"
            )
            for col in table.schema:
                names.add(col.name)
                names.update(_decode_text(k) for k in (col.metadata or {}))
                fields(col.type)
            for column in table.columns:
                if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
                    for value in (
                        column.unique().to_pylist()
                        if pa.types.is_string(column.type)
                        else column.to_pylist()
                    ):
                        keys(value)
        elif path.suffix.lower() in {".json", ".jsonl"}:
            text = path.read_text()
            if path.suffix.lower() == ".jsonl":
                for line in text.splitlines():
                    if line.strip():
                        keys(json.loads(line))
            else:
                keys(json.loads(text))
        elif path.suffix.lower() == ".npz":
            with zipfile.ZipFile(path) as packed:
                names.update(Path(n).stem for n in packed.namelist())
        elif path.suffix.lower() in {".csv", ".tsv"}:
            with path.open() as handle:
                names.update(
                    next(
                        csv.reader(
                            handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ","
                        ),
                        [],
                    )
                )
    # Numeric keys and neutral identifiers are visible contract values, not secret names.
    from fleetlab.contracts.ids import IdKind, is_neutral

    public = _visible_names(fleet)
    return {
        n
        for n in names
        if n not in public and not n.isdecimal() and not any(is_neutral(k, n) for k in IdKind)
    }


def _secret_values(run_dirs: Iterable[Path | str]) -> tuple[set[int], set[str]]:
    seeds: set[int] = set()
    fault_ids: set[str] = set()
    for run in _truth_runs(run_dirs):
        path = run / SCHEMA["identities"].path
        if path.is_file():
            data = pq.read_table(path)
            for name in data.column_names:
                if name.endswith("seed"):
                    seeds.update(int(v) for v in data[name].unique().to_pylist() if v is not None)
        path = run / SCHEMA["faults_active"].path
        if path.is_file():
            fault_ids.update(
                v
                for v in pq.read_table(path, columns=["fault_id"])["fault_id"].unique().to_pylist()
                if v
            )
    return seeds, fault_ids


def _identity_names(run_dirs: Iterable[Path | str]) -> set[str]:
    """Task identifiers are structural names, also used in public task prose.

    Select string columns belonging to the Task contract, so a new task-name
    field is covered without a second list. Neutral public ids are not selected.
    """
    from fleetlab.contracts.identity import Task

    fields = set(Task._fields)
    columns = [
        c.name
        for c in SCHEMA["identities"].columns
        if c.name in fields and c.visibility is Visibility.hidden and pa.types.is_string(c.type)
    ]
    names: set[str] = set()
    for run in _truth_runs(run_dirs):
        path = run / SCHEMA["identities"].path
        if columns and path.is_file():
            present = [c for c in columns if c in pq.read_schema(path).names]
            for column in pq.read_table(path, columns=present).columns:
                names.update(v for v in column.unique().to_pylist() if v)
    return names


def truth_terms(
    run_dirs: Iterable[Path | str] = (), *, fleet: Fleet | None = None, extra: Iterable[str] = ()
) -> TruthTerms:
    """Hidden names and values, with code-defined matching classes attached."""
    runs = _truth_runs(run_dirs)
    extra = list(extra)
    seeds, fault_ids = _secret_values(runs)
    identity_names = _identity_names(runs)
    terms: list[str] = [
        *GENERIC_TERMS,
        *_PROSE_TRUTH_NAMES,
        *(_LABEL_ONLY_TERMS - _visible_names(fleet)),
        *hidden_field_terms(),
        *_file_truth_names(runs, fleet),
        *_registry_kinds(),
        *map(str, seeds),
        *fault_ids,
        *identity_names,
    ]
    for run in runs:
        terms += _run_kinds(run)
        token = read_canary(run)
        if token:
            terms.append(token)
    return classify_truth_terms(
        [*terms, *_trait_terms(fleet), *extra],
        fleet=fleet,
        kinds={*_registry_kinds(), *(k for run in runs for k in _run_kinds(run))},
        fault_ids=fault_ids,
        identity_names=identity_names,
        extra=extra,
    )


def classify_truth_terms(
    terms: Iterable[str],
    *,
    fleet: Fleet | None = None,
    kinds: Iterable[str] | None = None,
    fault_ids: Iterable[str] = (),
    identity_names: Iterable[str] = (),
    extra: Iterable[str] = (),
) -> TruthTerms:
    """Attach current code-defined policy to a collected or frozen vocabulary.

    This lets a caller retain a world's exact inputs without repeating discovery.
    Public fields are removed. Only code-declared record structure is lenient;
    unknown names retain planted-name matching until a definition says otherwise.
    """
    kinds = set(_registry_kinds() if kinds is None else kinds)
    extra = set(extra)
    collected = [*terms, *identity_names]
    terms = _clean(collected)
    public = set(_clean(_visible_names(fleet)))
    terms = [t for t in terms if t not in public]
    prose = {*GENERIC_TERMS, *_PROSE_TRUTH_NAMES, "traits"}
    planted = set(_clean([*_planted_field_names(), *_trait_terms(fleet), *kinds]))
    mechanisms = _agent_planted_names(fleet, kinds)
    record_names = _record_structure_names() | set(_clean(identity_names))
    unknown = set(terms) - record_names - planted - set(_clean(prose))
    flexible = planted | unknown | mechanisms | set(_clean([*prose, *fault_ids, *extra]))
    flexible.update(t for t in terms if t.isdecimal() or t.startswith("canary-"))
    structural = set(terms) - flexible
    labels = structural | {t for t in terms if t.isalpha() and t not in prose}
    spellings: dict[str, set[str]] = {}
    for original in collected:
        normalized = _clean([original])
        if normalized and normalized[0] in structural:
            spellings.setdefault(normalized[0], set()).add(original.strip().lower())
    return TruthTerms(
        terms,
        structural=structural,
        labels=labels - set(extra),
        # Public configuration keys show these identifiers after their namespace
        # (control.sampling_steps). An agent may use the same visible leaf name.
        planted=(mechanisms | (unknown - _recipe_field_names())) - _agent_public_names(fleet),
        spellings=spellings,
    )


def _agent_public_names(fleet: Fleet | None = None) -> set[str]:
    return set(_clean(n.rsplit(".", 1)[-1] for n in _visible_names(fleet))) | set(
        _clean(_known_code_exceptions())
    )


def code_terms(
    run_dirs: Iterable[Path | str] = (), *, fleet: Fleet | None = None, extra: Iterable[str] = ()
) -> list[str]:
    """The terms for source code in the sandbox: what is specific to a world only."""
    runs = _truth_runs(run_dirs)
    seeds, fault_ids = _secret_values(runs)
    terms: list[str] = [*_registry_kinds(), *map(str, seeds), *fault_ids]
    for run in runs:
        terms += _run_kinds(run)
        token = read_canary(run)
        if token:
            terms.append(token)
    exceptions = _known_code_exceptions()
    return [t for t in _clean([*terms, *_trait_terms(fleet), *extra]) if t not in exceptions]


def _guard(term: str) -> str:
    """A generic truth word cannot start inside another word (default is not fault)."""
    return "(?<![a-z])" if term.isalpha() else ""


def _first_outside(
    pattern: re.Pattern[Any], data: Any, mask: re.Pattern[Any] | None
) -> tuple[int, int] | None:
    """The first match of `pattern` that does not lie inside a name of the visible vocabulary."""
    spans: list[tuple[int, int]] | None = None
    for match in pattern.finditer(data):
        if mask is None:
            return match.start(), match.end()
        if spans is None:
            spans = [(m.start(), m.end()) for m in mask.finditer(data)]
        if not any(a <= match.start() and match.end() <= b for a, b in spans):
            return match.start(), match.end()
    return None


class _Search:
    def __init__(
        self,
        terms: Iterable[str],
        field_terms: Iterable[str] = (),
        *,
        surface: Literal["ours", "agent"],
        kinds: Iterable[str] = (),
        seeds: Iterable[int] = (),
        fault_ids: Iterable[str] = (),
    ) -> None:
        if surface not in ("ours", "agent"):
            raise ValueError("surface must be 'ours' or 'agent'")
        self.surface = surface
        self.structural = set(terms.structural) if isinstance(terms, TruthTerms) else set()
        spellings = terms.spellings if isinstance(terms, TruthTerms) else {}
        labels = terms.labels if isinstance(terms, TruthTerms) else ()
        self.terms = tuple(_clean(terms))
        fault_ids = tuple(fault_ids)
        self.fields = {*field_terms, *labels}
        self.kinds = set(_clean(kinds))
        self.fields -= self.kinds
        # A class describes multipart spellings. It cannot turn an ordinary
        # single word (even a new kind/property) into evidence in prose.
        self.single_names = {
            t
            for t in self.terms
            if re.fullmatch(r"[a-z0-9]+", t)
            and not t.isdecimal()
            and t not in {*GENERIC_TERMS, *_PROSE_TRUTH_NAMES, "traits", *_clean(fault_ids)}
        }
        if surface == "ours":
            self.fields.update(self.single_names)
        self.seeds = set(seeds)
        self.float_seeds: dict[float, set[int]] = {}
        for seed in self.seeds:
            for format_ in ("d", "f", "e"):
                with suppress(OverflowError, struct.error):
                    rounded = struct.unpack(format_, struct.pack(format_, float(seed)))[0]
                    if math.isfinite(rounded):
                        self.float_seeds.setdefault(rounded, set()).add(seed)
        # Countable IDs collide with an agent's own finding numbers. Exempt
        # only that world's one-letter/two-digit IDs on its authored surface.
        self.fault_ids = {
            fault_id
            for fault_id in fault_ids
            if surface == "ours" or re.fullmatch(r"[A-Za-z][0-9]{2}", fault_id) is None
        }
        if surface == "agent":
            # Only the code-defined planted/hidden names belong to this
            # tripwire. Ordinary words and record structure are not evidence
            # of a leak in an agent's own diagnosis, even as keys or labels.
            self.fields.clear()
            planted = (
                set(terms.planted)
                if isinstance(terms, TruthTerms)
                else set(
                    classify_truth_terms(
                        self.terms, kinds={*_registry_kinds(), *self.kinds}
                    ).planted
                )
            )
            planted = set(planted) - _agent_public_names()
            agent_terms = {
                t
                for t in self.terms
                if t.startswith("canary-")
                or t in self.fault_ids
                or (t in planted and len([p for p in re.split(r"[_.]+", t) if p]) > 1)
            }
            # A world fault id is checked even if its spelling coincides with a
            # public name, or the caller omitted it from the name vocabulary.
            agent_terms |= self.fault_ids
        else:
            agent_terms = set(self.terms)
        seed_terms = {str(s) for s in self.seeds}
        self.seed_pattern = (
            re.compile(r"(?<!\d)(?:" + "|".join(map(str, sorted(self.seeds))) + r")(?!\d)")
            if self.seeds
            else None
        )

        def pattern(term: str, *, raw: bool = False) -> str:
            if surface == "agent":
                if term.startswith("canary-"):
                    return "(?i:" + re.escape(term) + ")"
                parts = re.split(r"[_.]+", term)
                literal = re.escape(term)
                if term not in self.fault_ids:
                    literal = r"[_.]".join(map(re.escape, parts))
                exact = "(?i:" + literal + ")"
                if len(parts) > 1 and all(parts) and term not in self.fault_ids:
                    # Keep case until matching: gripperWear is an identifier;
                    # gripperwear, gripper wear and gripper-wear are natural forms.
                    camel = (
                        r"(?=[a-zA-Z0-9]*(?:[a-z0-9][A-Z]|[A-Z][A-Z][a-z]))(?i:"
                        + re.escape(parts[0])
                        + ")"
                        + "".join(
                            re.escape(p[0].upper()) + "(?i:" + re.escape(p[1:]) + ")"
                            for p in parts[1:]
                        )
                    )
                    exact = "(?:" + exact + "|" + camel + ")"
                # Literal identifiers also count inside aliases, prefixes,
                # suffixes and plural spellings. Natural forms remain excluded.
                return exact
            if term.startswith("canary-"):
                return re.escape(term)
            if term in self.fault_ids:
                return r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])"
            if term in self.structural:
                literal = "|".join(map(re.escape, sorted(spellings.get(term, (term,)))))
                return r"(?<![a-z0-9_])(?:" + literal + r")(?![a-z0-9_])"
            parts = re.split(r"[^a-z0-9]+", term)
            if not all(parts):
                return re.escape(term)
            separator = r"(?:_|[^\w\s])*" if term in self.kinds else r"[\W_]*"
            if raw:
                # Opaque compressed bytes are not Unicode text separators.
                punctuation = r"\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e"
                whitespace = "" if term in self.kinds else r"\x09-\x0d\x20"
                separator = "[" + whitespace + punctuation + "]*"
            joined = separator.join(re.escape(p) for p in parts)
            if len(parts) > 1:
                # Main's literal spelling is a substring match. Only forms that
                # tolerate separators/joining need boundaries (as_run != has run).
                return r"(?:" + re.escape(term) + r"|(?<![a-z0-9])" + joined + r"(?![a-z0-9]))"
            if term in GENERIC_TERMS or term == "traits":
                return _guard(term) + joined
            return r"(?<![a-z0-9_])" + joined + r"(?![a-z0-9_])"

        self.names = {t: re.compile(pattern(t)) for t in sorted(agent_terms) if t not in seed_terms}
        self.anchors = {t: max(re.split(r"[^a-z0-9]+", t.lower()), key=len) for t in self.names}

        def text_pattern(term: str) -> str:
            if surface == "ours" and term in self.single_names:
                literal = "(?i:" + re.escape(term) + ")"
                # A prose phrase ending in a colon is not a record label:
                # "from the gripper and the frames: approach" is ordinary text.
                # Unquoted colon labels start a line or a delimited record item.
                return (
                    r"(?<![a-zA-Z0-9_])"
                    + literal
                    + r"[\"']?\s*="
                    + r"|(?m:(?:^|[\n{\[;,])[ \t]*[\"'`]?"
                    + literal
                    + r"[\"'`]?\s*:)"
                    + r"|`"
                    + literal
                    + r"`"
                    + r"|(?<![a-zA-Z0-9_])"
                    + literal
                    + r"(?=\.[a-zA-Z_])"
                    + r"|(?<=[a-zA-Z0-9]\.)"
                    + literal
                    + r"(?![a-zA-Z0-9_])"
                )
            return self.names[term].pattern + (r"[\"']?\s*[:=]" if term in self.fields else "")

        self.text = {t: re.compile(text_pattern(t)) for t in self.names}
        self.raw = {
            t: re.compile(
                (
                    _guard(t) + re.escape(t)
                    if t in GENERIC_TERMS and surface == "ours"
                    else pattern(t, raw=True)
                ).encode()
            )
            for t in self.names
            if t not in self.fields
            and (
                len(t) >= MIN_BYTES_TERM
                if t in GENERIC_TERMS
                else sum(c.isalnum() for c in t) >= MIN_RAW_NAME_LETTERS
            )
        }
        self.report = LeakReport(self.terms)
        self.searched = {"files": 0, "bytes": 0, "names": 0, "text values": 0, "texts": 0}
        self.report.searched = self.searched

    def scan_text(self, where: str, text: str, mask: bool = True, *, name: bool = False) -> None:
        lowered = text.lower()
        data = text if self.surface == "agent" else lowered
        mask_pattern = _mask_text()
        if self.surface == "agent":
            mask_pattern = re.compile(mask_pattern.pattern, re.IGNORECASE)
        if self.seed_pattern is not None:
            for seed_match in self.seed_pattern.finditer(lowered):
                self.report.hits.append(
                    LeakHit(
                        where,
                        seed_match.group(),
                        lowered[max(0, seed_match.start() - 30) : seed_match.end() + 30],
                    )
                )
            # Exact seed values in ordinary numeric notation: separators,
            # base prefixes and scientific notation. Decimal avoids rounding
            # large identity seeds into unrelated neighboring integers.
            numeric = (
                r"0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|"
                r"[0-9]+(?:[,_][0-9]+)*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
            )
            for numeric_match in re.finditer(r"(?<![0-9])(?:" + numeric + r")(?![0-9])", text):
                token = numeric_match.group().replace(",", "").replace("_", "")
                try:
                    number = (
                        decimal.Decimal(int(token, 0))
                        if token.lower().startswith(("0x", "0b", "0o"))
                        else decimal.Decimal(token)
                    )
                except (ValueError, decimal.InvalidOperation):
                    continue
                if number in self.seeds and not numeric_match.group().isdecimal():
                    self.report.hits.append(LeakHit(where, str(int(number)), numeric_match.group()))
        for term, pattern in (self.names if name else self.text).items():
            if self.anchors[term] not in lowered:
                continue
            match = _first_outside(
                pattern,
                data,
                mask_pattern
                if mask
                and not (name and term == lowered)
                and not (
                    self.surface == "agent"
                    and (term in self.fault_ids or term.startswith("canary-"))
                )
                else None,
            )
            if match is not None:
                start = max(0, match[0] - 30)
                self.report.hits.append(LeakHit(where, term, data[start : match[1] + 30]))

    def scan_bytes(self, where: str, data: bytes) -> None:
        lowered = data.lower()
        matched_data = data if self.surface == "agent" else lowered
        mask_pattern = _mask_bytes()
        if self.surface == "agent":
            mask_pattern = re.compile(mask_pattern.pattern, re.IGNORECASE)
        for byte_seed_match in re.finditer(rb"(?<![0-9])[0-9]{5,20}(?![0-9])", data):
            if int(byte_seed_match.group()) in self.seeds:
                self.report.hits.append(
                    LeakHit(where, byte_seed_match.group().decode(), repr(byte_seed_match.group()))
                )
        for term, pattern in self.raw.items():
            if self.anchors[term].encode() not in lowered:
                continue
            match = _first_outside(
                pattern,
                matched_data,
                None
                if self.surface == "agent"
                and (term in self.fault_ids or term.startswith("canary-"))
                else mask_pattern,
            )
            if match is not None:
                start = max(0, match[0] - 20)
                context = repr(lowered[start : match[1] + 20])
                self.report.hits.append(LeakHit(where, term, context))

    def scan_json(
        self, where: str, value: Any, path: str = "", *, name_value: bool = False
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                self.searched["text values"] += 1
                self.scan_text(f"{where} key {path}/{key}", str(key), name=True)
                self.scan_json(where, item, f"{path}/{key}", name_value=name_value)
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
            self.scan_text(f"{where} map key {path}", value[0], name=True)
            self.scan_json(where, value[1], f"{path}/{value[0]}", name_value=name_value)
        elif isinstance(value, list | tuple):
            for i, item in enumerate(value):
                self.scan_json(where, item, f"{path}/{i}", name_value=name_value)
        elif isinstance(value, str):
            self.searched["text values"] += 1
            self.scan_text(f"{where} at {path}", value, name=name_value)
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                except ValueError:
                    return
                self.scan_json(where, decoded, path + " (decoded JSON)", name_value=name_value)
        elif isinstance(value, bytes):
            try:
                self.scan_json(where, _decode_text(value).rstrip("\0"), path, name_value=name_value)
            except UnicodeError:
                self.report.hits.append(
                    LeakHit(
                        f"{where} at {path}",
                        "(undecodable value)",
                        "binary text is not UTF-8/UTF-16",
                    )
                )
        elif isinstance(value, int | float) and not isinstance(value, bool):
            self.scan_number(f"{where} at {path}", value)

    def scan_number(self, where: str, value: int | float) -> None:
        seeds = (
            self.float_seeds.get(value, set()) if isinstance(value, float) else {value} & self.seeds
        )
        for seed in sorted(seeds):
            self.report.hits.append(LeakHit(where, str(seed), str(value)))

    def scan_parquet(self, where: str, path: Path) -> None:
        data = pq.read_table(path)
        for column in data.column_names:
            self.scan_text(f"column of {where}.{column}", column, name=True)
        self.scan_metadata(f"metadata of {where}", data.schema.metadata)
        self.scan_text(f"writer of {where}", str(pq.ParquetFile(path).metadata.created_by))
        for arrow_field in data.schema:
            self.scan_field(where, arrow_field)
        for name in data.column_names:
            column = data.column(name)
            dtype = column.type
            if not self.decodable(dtype):
                self.report.hits.append(
                    LeakHit(f"column of {where}.{name}", "(undecodable column)", str(dtype))
                )
            elif pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
                for value in column.unique().to_pylist():
                    if value is not None:
                        self.scan_number(f"value of {where}.{name}", value)
            elif (
                pa.types.is_string(dtype)
                or pa.types.is_large_string(dtype)
                or pa.types.is_string_view(dtype)
            ):
                for value in column.unique().to_pylist():
                    if value is not None:
                        self.searched["text values"] += 1
                        self.scan_text(
                            f"value of {where}.{name}", value, name=name in {"metric", "key"}
                        )
                        if value.lstrip().startswith(("{", "[")):
                            try:
                                decoded = json.loads(value)
                            except ValueError:
                                continue  # valid text need not be a JSON document
                            else:
                                self.scan_json(
                                    f"decoded value of {where}.{name}",
                                    decoded,
                                    name_value=name in {"metric", "key"},
                                )
            elif not (
                pa.types.is_floating(dtype)
                or pa.types.is_boolean(dtype)
                or pa.types.is_temporal(dtype)
                or pa.types.is_decimal(dtype)
                or pa.types.is_null(dtype)
            ):
                # A word in a compressed list/struct/dictionary is invisible in raw bytes.
                for i, value in enumerate(column.to_pylist()):
                    self.scan_json(
                        f"value of {where}.{name} row {i}",
                        value,
                        name_value=name in {"metric", "key"},
                    )

    def scan_metadata(self, where: str, metadata: Mapping[bytes, bytes] | None) -> None:
        for key, value in (metadata or {}).items():
            if key == b"ARROW:schema":
                continue  # its decoded schema and fields are checked separately
            try:
                self.scan_text(f"{where} key", _decode_text(key), name=True)
                self.scan_text(f"{where} value", _decode_text(value), name=True)
                self.scan_json(where, _decode_text(value))
            except UnicodeError:
                self.report.hits.append(LeakHit(where, "(undecodable metadata)", repr(key)))

    def scan_field(self, where: str, field: pa.Field, prefix: str = "") -> None:
        name = f"{prefix}.{field.name}" if prefix else field.name
        self.scan_text(f"field of {where}.{name}", field.name, name=True)
        self.scan_metadata(f"field metadata of {where}.{name}", field.metadata)
        dtype = field.type
        if pa.types.is_struct(dtype) or pa.types.is_union(dtype):
            for child in dtype:
                self.scan_field(where, child, name)
        elif (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
            or pa.types.is_list_view(dtype)
            or pa.types.is_large_list_view(dtype)
        ):
            self.scan_field(where, dtype.value_field, name)
        elif pa.types.is_map(dtype):
            self.scan_field(where, dtype.key_field, name)
            self.scan_field(where, dtype.item_field, name)

    @staticmethod
    def decodable(dtype: pa.DataType) -> bool:
        if isinstance(dtype, pa.BaseExtensionType):
            return dtype == pa.json_() and _Search.decodable(dtype.storage_type)
        if (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_string_view(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_fixed_size_binary(dtype)
            or pa.types.is_binary_view(dtype)
        ):
            return True
        if pa.types.is_dictionary(dtype):
            return _Search.decodable(dtype.value_type)
        if pa.types.is_struct(dtype) or pa.types.is_union(dtype):
            return all(_Search.decodable(f.type) for f in dtype)
        if (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
            or pa.types.is_list_view(dtype)
            or pa.types.is_large_list_view(dtype)
        ):
            return _Search.decodable(dtype.value_type)
        if pa.types.is_map(dtype):
            return _Search.decodable(dtype.key_type) and _Search.decodable(dtype.item_type)
        return bool(
            pa.types.is_integer(dtype)
            or pa.types.is_floating(dtype)
            or pa.types.is_boolean(dtype)
            or pa.types.is_temporal(dtype)
            or pa.types.is_decimal(dtype)
            or pa.types.is_null(dtype)
        )


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def _description(search: _Search, where: str, text: str) -> None:
    data = json.loads(text)
    if isinstance(data, dict) and is_vocabulary(data.get("vocabulary")):
        data = {k: v for k, v in data.items() if k != "vocabulary"}
    if isinstance(data, dict) and where == DESCRIPTION_FILE and search.surface == "ours":
        for key in sorted(data.keys() - _record_keys(describe_for_agents)):
            search.report.hits.append(
                LeakHit(f"{where} key /{key}", key, "undeclared public description field")
            )
        if "fleet" in data:
            try:
                fleet = Fleet.model_validate(data["fleet"])
            except ValidationError as error:
                for problem in error.errors():
                    loc = problem["loc"]
                    pointer = "/fleet/" + "/".join(map(str, loc))
                    term = str(loc[-1]) if loc else "(invalid public description)"
                    search.report.hits.append(
                        LeakHit(f"{where} at {pointer}", term, problem["msg"])
                    )
            else:
                # Valid public maps (health/configuration readings) retain their
                # declared arbitrary keys. Hidden model fields never serialize.
                def visibility(value: Any, public: Any, path: str) -> None:
                    if isinstance(value, dict) and isinstance(public, dict):
                        for key, item in value.items():
                            pointer = f"{path}/{key}"
                            if key not in public:
                                search.report.hits.append(
                                    LeakHit(f"{where} key {pointer}", key, "hidden model field")
                                )
                            else:
                                visibility(item, public[key], pointer)
                    elif isinstance(value, list) and isinstance(public, list):
                        for i, (item, declared) in enumerate(zip(value, public, strict=True)):
                            visibility(item, declared, f"{path}/{i}")

                visibility(data["fleet"], visible_dump(fleet), "/fleet")
    search.scan_json(where, data)


def leak_search(
    root: Path | str,
    terms: Iterable[str],
    texts: Mapping[str, str] | None = None,
    *,
    agent_texts: Mapping[str, str] | None = None,
    field_terms: Iterable[str] = (),
    kinds: Iterable[str] | None = None,
    seeds: Iterable[int] = (),
    fault_ids: Iterable[str] = (),
) -> LeakReport:
    """Search a tree strictly; only separately sourced agent_texts are lenient."""
    root = Path(root)
    terms = terms if isinstance(terms, TruthTerms) else tuple(terms)
    field_terms, seeds, fault_ids = tuple(field_terms), tuple(seeds), tuple(fault_ids)
    kinds = tuple(_registry_kinds() if kinds is None else kinds)
    search = _Search(
        terms,
        field_terms,
        surface="ours",
        kinds=kinds,
        seeds=seeds,
        fault_ids=fault_ids,
    )
    if not root.is_dir() or root.is_symlink():
        search.report.hits.append(
            LeakHit(str(root), "(unreadable export)", "missing directory or symbolic link")
        )
        return search.report
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        search.searched["names"] += 1
        search.scan_text(f"name {relative}", path.name, name=True)
        if path.is_symlink():
            search.report.hits.append(LeakHit(relative, "(symbolic link)", str(path.readlink())))
            continue
        if not path.is_file():
            continue
        search.searched["files"] += 1
        try:
            data = path.read_bytes()
            search.searched["bytes"] += len(data)
            suffix = path.suffix.lower()
            if suffix not in _TEXT_SUFFIXES:
                search.scan_bytes(f"bytes of {relative}", data)
            if suffix == ".parquet":
                search.scan_parquet(relative, path)
            elif suffix == ".mp4":
                if data[4:8] != b"ftyp":
                    raise ValueError("not an MP4 container")
            elif suffix in {".gz", ".zip", ".npz", ".feather", ".arrow"}:
                raise ValueError("unsupported packed/binary file")
            else:
                text = _decode_text(data)
                if relative.endswith(DESCRIPTION_FILE.removeprefix("log/")) and suffix == ".json":
                    _description(search, relative, text)
                elif suffix == ".json":
                    search.scan_json(relative, json.loads(text))
                elif suffix == ".jsonl":
                    for i, line in enumerate(text.splitlines()):
                        if line.strip():
                            search.scan_json(f"{relative} line {i + 1}", json.loads(line))
                elif suffix == ".csv":
                    for i, row in enumerate(csv.reader(io.StringIO(text))):
                        for j, value in enumerate(row):
                            search.scan_text(f"{relative} row {i} column {j}", value, name=i == 0)
                else:
                    search.scan_text(relative, text)
        except (OSError, ValueError, pa.ArrowException, UnicodeError) as error:
            search.report.hits.append(LeakHit(relative, "(undecodable file)", str(error)))
    additional = search_texts(
        terms,
        texts,
        agent_texts=agent_texts,
        field_terms=field_terms,
        kinds=kinds,
        seeds=seeds,
        fault_ids=fault_ids,
    )
    _combine_reports(search.report, additional)
    return search.report


def _combine_reports(report: LeakReport, other: LeakReport) -> None:
    report.hits.extend(other.hits)
    for key, value in other.searched.items():
        report.searched[key] = report.searched.get(key, 0) + value


def search_texts(
    terms: Iterable[str],
    texts: Mapping[str, str] | None = None,
    *,
    agent_texts: Mapping[str, str] | None = None,
    field_terms: Iterable[str] = (),
    kinds: Iterable[str] | None = None,
    seeds: Iterable[int] = (),
    fault_ids: Iterable[str] = (),
) -> LeakReport:
    """Scan our texts and separately named agent-written texts, without a tree.

    Each mapping key must identify the source (file and field where available).
    No surface selector can relax an export tree or any mounted source code.
    """
    terms, field_terms = (
        list(terms) if not isinstance(terms, TruthTerms) else terms,
        tuple(field_terms),
    )
    kinds = tuple(_registry_kinds() if kinds is None else kinds)
    seeds, fault_ids = tuple(seeds), tuple(fault_ids)
    report = LeakReport(tuple(terms))
    surfaces: tuple[tuple[Literal["ours", "agent"], Mapping[str, str] | None], ...] = (
        ("ours", texts),
        ("agent", agent_texts),
    )
    for surface, sources in surfaces:
        if not sources:
            continue
        search = _Search(
            terms,
            field_terms,
            surface=surface,
            kinds=kinds,
            seeds=seeds,
            fault_ids=fault_ids,
        )
        for where, text in sources.items():
            if not isinstance(where, str) or not where.strip():
                raise ValueError("each text must name its source")
            search.searched["texts"] += 1
            try:
                if where.endswith(".json"):
                    search.scan_json(where, json.loads(text))
                elif where.endswith(".jsonl"):
                    for index, line in enumerate(text.splitlines(), 1):
                        if line.strip():
                            search.scan_json(f"{where} line {index}", json.loads(line))
                else:
                    search.scan_text(where, text)
            except ValueError:
                # A malformed supplied document is still text to the tripwire.
                search.scan_text(where, text)
        _combine_reports(report, search.report)
    return report


def canary_problems(run_dir: Path | str) -> list[str]:
    """Why a run's canary could not show a leak: missing, or absent from a hidden table."""
    root = Path(run_dir)
    token = read_canary(root)
    if token is None:
        return [f"{root}: no canary planted (open the run with RunWriter)"]
    problems = []
    for table in SCHEMA.values():
        path = root / table.path
        if table.tree != "truth" or table.storage != "parquet" or not path.is_file():
            continue
        metadata = pq.read_schema(path).metadata or {}
        if metadata.get(CANARY_METADATA_KEY) != token.encode():
            problems.append(f"{root}: {table.path} does not carry the canary")
    return problems


def audit_export(
    export_dir: Path | str,
    run_dirs: Sequence[Path | str],
    *,
    fleet: Fleet | None = None,
    texts: Mapping[str, str] | None = None,
    agent_texts: Mapping[str, str] | None = None,
    code: Mapping[str, str] | None = None,
    extra_terms: Iterable[str] = (),
) -> LeakReport:
    """The pre-export check: truth terms and canaries over the export and the given texts.

    The tree and `texts` are strict (prompts, answer schemas); `agent_texts`
    names only agent-written prose separately, with its source. `code`, the
    source mounted into the sandbox, with `code_terms` only. Raises ValueError
    when a run's canary is not planted, because the check would then prove nothing.
    """
    problems = [p for run in run_dirs for p in canary_problems(run)]
    if problems:
        raise ValueError("the canary check cannot run:\n" + "\n".join(problems))
    extra = list(extra_terms)
    # Single words such as "row", "first" and "small" can be ordinary fleet prose.
    # Still reject them as JSON keys, columns, filenames and labels like "small=true".
    kinds = {*_registry_kinds(), *(k for run in _truth_runs(run_dirs) for k in _run_kinds(run))}
    terms = truth_terms(run_dirs, fleet=fleet, extra=extra)
    prose = {*GENERIC_TERMS, *_PROSE_TRUTH_NAMES, "traits"}
    fields = {t for t in terms if t.isalpha() and t not in prose} | _LABEL_ONLY_TERMS
    fields -= prose
    seeds, fault_ids = _secret_values(run_dirs)
    report = leak_search(
        export_dir,
        terms,
        texts,
        agent_texts=agent_texts,
        field_terms=fields - set(extra),
        kinds=kinds,
        seeds=seeds,
        fault_ids=fault_ids,
    )
    if code:
        narrow_terms = code_terms(run_dirs, fleet=fleet, extra=())
        narrow = _Search(
            narrow_terms,
            surface="ours",
            kinds=kinds,
            seeds=seeds,
            fault_ids=fault_ids,
        )
        for where, text in code.items():
            narrow.searched["texts"] += 1
            narrow.scan_text(f"code {where}", text, mask=False)
        report.hits += narrow.report.hits
        report.searched["code texts"] = narrow.searched["texts"]
    return report


def export_digest(root: Path | str) -> str:
    """Content digest of names, directories and file bytes; never follow symlinks."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{root}: not a regular export directory")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{path}: symbolic link")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(b"directory\0" if path.is_dir() else b"file\0")
        if path.is_file():
            content = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    content.update(block)
            digest.update(content.digest())
        elif not path.is_dir():
            raise ValueError(f"{path}: unsupported filesystem entry")
    return digest.hexdigest()


def audit_identifier(run_dirs: Sequence[Path | str], fleet: Fleet | None = None) -> str:
    """Current code-defined vocabulary and checker; never inspect a mutable world."""
    from fleetlab.contracts import _base, ids, log, model, tools, vocabulary
    from fleetlab.log import export

    terms = truth_terms()
    digest = hashlib.sha256(
        json.dumps(
            {
                "terms": terms,
                "structural": sorted(terms.structural),
                "labels": sorted(terms.labels),
                "planted": sorted(terms.planted),
                "spellings": {k: sorted(v) for k, v in sorted(terms.spellings.items())},
                "public": sorted(_visible_names()),
            }
        ).encode()
    )
    digest.update(Path(__file__).read_bytes())
    for module in (
        export,
        log,
        _base,
        ids,
        model,
        tools,
        vocabulary,
    ):
        assert module is not None and module.__file__ is not None
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def write_export_check(
    export_dir: Path,
    run_dirs: Sequence[Path | str],
    *,
    fleet: Fleet | None,
    receipt_dir: Path,
    publish_as: Path | None = None,
) -> dict[str, Any]:
    """Check a tree and append a receipt; a staged tree may be renamed to publish_as.

    The digest uses relative names and bytes, so that rename preserves what was
    checked. A reader still requires the published tree to match the receipt.
    Existing exports and receipts are never modified.
    """
    from dataclasses import asdict

    from fleetlab.log.export import export_problems

    started = time.perf_counter()
    before = export_digest(export_dir)
    identifier = audit_identifier(run_dirs, fleet)
    # Check the exact tree between digests. Atomic directory publication preserves
    # these relative names and bytes, and readiness compares the published digest.
    report = audit_export(export_dir, run_dirs, fleet=fleet)
    problems = export_problems(export_dir)
    after = export_digest(export_dir)
    if after != before or identifier != audit_identifier(run_dirs, fleet):
        problems.append("export or checker vocabulary changed during the check")
    receipt = {
        "export": str((publish_as or export_dir).resolve()),
        "export_digest": after,
        "check_id": identifier,
        "checked_at": time.time(),
        "clean": report.clean and not problems,
        "problems": problems,
        "hits": [asdict(hit) for hit in report.hits],
        "searched": report.searched,
        "term_count": len(report.terms),
        "terms": report.terms,
        "terms_digest": hashlib.sha256(json.dumps(report.terms).encode()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    receipt_dir.mkdir(parents=True, exist_ok=True)
    target = receipt_dir / f"check-{time.time_ns()}-{uuid.uuid4().hex}.json"
    with target.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return {**receipt, "receipt": str(target)}


def matching_export_check(
    export_dir: Path, run_dirs: Sequence[Path | str], *, fleet: Fleet | None, receipt_dir: Path
) -> bool:
    """A saved flag alone is never evidence; require current bytes and check identity."""
    return export_check_problem(export_dir, receipt_dir=receipt_dir) is None


def export_check_problem(export_dir: Path, *, receipt_dir: Path) -> str | None:
    """Explain a stale/missing receipt, without discovering mutable reader outputs."""
    import shlex

    command = (
        f"uv run --locked python -m fleetlab.log certify {shlex.quote(str(receipt_dir.parent))}"
    )
    if export_dir != receipt_dir.parent / "export":
        command += f" --export {shlex.quote(str(export_dir))}"
    reason = "no clean receipt from the current checker for these export bytes"
    try:
        identifier = audit_identifier(())
        digest = export_digest(export_dir)
        for path in sorted(receipt_dir.glob("check-*.json"), reverse=True):
            if path.is_symlink():
                continue
            receipt = json.loads(path.read_text())
            if (
                receipt.get("clean") is True
                and receipt.get("export") == str(export_dir.resolve())
                and receipt.get("export_digest") == digest
                and receipt.get("check_id") == identifier
                and isinstance(receipt.get("terms"), list)
                and receipt.get("terms_digest")
                == hashlib.sha256(json.dumps(receipt["terms"]).encode()).hexdigest()
            ):
                return None
    except (OSError, ValueError, pa.ArrowException, zipfile.BadZipFile) as error:
        reason = f"cannot validate the export receipt: {error}"
    return f"{reason}; certify without rewriting stored records: {command}"


def main(argv: Sequence[str] | None = None) -> int:
    """Append a certification receipt for a stored export, without running a world."""

    parser = argparse.ArgumentParser(prog="python -m fleetlab.log")
    sub = parser.add_subparsers(dest="command", required=True)
    certify = sub.add_parser("certify", help="check a stored export and append a host receipt")
    certify.add_argument("world", type=Path)
    certify.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    root = args.world.resolve()
    from fleetlab.worlds.layout import WorldPaths

    paths = WorldPaths.from_root(root)
    files = root / "files" if (root / "files").is_dir() else root / "world"
    fleet = Fleet.model_validate_json((files / "model.json").read_text())

    run = paths.observed
    receipt = write_export_check(
        (args.export or root / "export").resolve(),
        [run],
        fleet=fleet,
        receipt_dir=root / "export-checks",
    )
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["clean"] else 1
