"""Shared by the skill tests: the library, its helpers, the toy exports, the planted words.

The helpers are loaded from their files without writing bytecode, so that no
`__pycache__` appears in a skill directory (the runtime mounts every file of a
skill into the sandbox).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.vocabulary import vocabulary_json
from fleetlab.log import export_runs
from fleetlab.log.leaks import code_terms, hidden_column_terms
from fleetlab.skill_library import ANALYSIS_SKILLS, SUPERVISOR_SKILLS

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
V1_SKILLS = ANALYSIS_SKILLS + SUPERVISOR_SKILLS
MAX_ROWS = 200  # what the tool server's `query` returns at most (its default limit)

_BLOCK = re.compile(r"```sql\n(-- helper: (?P<name>[a-z_]+)\n.*?)```", re.DOTALL)

# Fault kinds of the design's catalogue that the fault library renamed; the registry's
# own kinds come from `registry_kinds`. Plain words such as "lighting" are visible
# vocabulary (condition keys) and are not planted.
DESIGN_KINDS = ("action_delay", "software_version")
# Measured values of the stage-1 calibration (dose-response points, nominal rates):
# general skills must state shapes, never these numbers.
PLANTED_VALUES = ("89.5", "54.5", "72.5", "98.5", "96.0")


def registry_kinds() -> list[str]:
    """Every fault kind of the registry except the two kinds an action adds."""
    import fleetlab.faults
    from fleetlab.contracts.actions import ADDED_CHANGE_KINDS

    fleetlab.faults.load_all()
    return sorted(set(fleetlab.faults.kinds()) - set(ADDED_CHANGE_KINDS))


def helper_path(skill: str) -> Path:
    return SKILLS / skill / f"{skill.replace('-', '_')}_helpers.py"


@cache
def helpers(skill: str) -> ModuleType:
    """A skill's helper module, loaded from its file without writing bytecode."""
    path = helper_path(skill)
    spec = importlib.util.spec_from_file_location(f"skill_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def sql_blocks(skill: str) -> dict[str, str]:
    """The SQL blocks of a skill's SKILL.md by name."""
    text = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    return {m["name"]: m[1] for m in _BLOCK.finditer(text)}


@dataclass(frozen=True)
class Exports:
    """The two toy worlds the skill tests run on: the skills' fleet and the log's toy."""

    fleet_run: Path
    fleet_export: Path
    fleet: Fleet
    toy_run: Path
    toy_export: Path
    toy_fleet: Fleet


def build_exports(base: Path) -> Exports:
    import log_toy
    import skills_fleet

    fleet_run, fleet_export = skills_fleet.write_fleet_export(base / "fleet")
    toy_fleet = log_toy.write_toy_run(base / "toy" / "run")
    export_runs(
        [base / "toy" / "run"],
        base / "toy" / "export",
        fleet=toy_fleet,
        task_classes=log_toy.TASK_CLASSES,
    )
    return Exports(
        fleet_run,
        fleet_export,
        skills_fleet.fleet(),
        base / "toy" / "run",
        base / "toy" / "export",
        toy_fleet,
    )


def _vocabulary_names() -> set[str]:
    names: set[str] = set()
    for entries in vocabulary_json().values():
        if isinstance(entries, list):
            names.update(str(entry["name"]) for entry in entries)
    return names


def planted_vocabulary(exports: Exports) -> list[str]:
    """The truth vocabulary a skill must not contain.

    What the runtime's source audit searches mounted files for (fault kinds of
    the registry and of the runs, trait names, canary tokens), the fault kinds
    of the design, the hidden columns' compound names (except those inside a
    visible vocabulary name, as the leak search does), and calibration values.
    """
    terms = set(
        code_terms(
            [exports.fleet_run, exports.toy_run],
            fleet=exports.fleet,
            extra=sorted(exports.toy_fleet.robots()[0].traits),
        )
    )
    visible = _vocabulary_names()
    terms |= {t for t in hidden_column_terms() if not any(t in name for name in visible)}
    terms |= set(registry_kinds()) | set(DESIGN_KINDS) | set(PLANTED_VALUES)
    return sorted(terms)


def example_ids() -> list[str]:
    """One identifier of every kind, as a world would print it."""
    return [neutral_id(kind, 7) for kind in IdKind]
