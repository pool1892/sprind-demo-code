"""The toy world of the checks and the measured tier: `tests/log_toy.py`'s run, export, a skill.

Needs a checkout (it imports `log_toy` from the repository's `tests/`); the
runtime itself does not.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import fleetlab
from fleetlab.contracts.model import Fleet
from fleetlab.log import export_runs
from fleetlab.log.export import export_problems

__all__ = ["TOY_SKILL", "ToyWorld", "toy_world", "write_toy_skill"]

TESTS = Path(fleetlab.__file__).resolve().parents[2] / "tests"

# A general procedure (contracts, section 13): no identifier, no value of a world.
TOY_SKILL = """---
name: triage
description: Find the robots whose success rate departs from the rest of the fleet.
when_to_use: At the start of an analysis, before any hypothesis.
tools_used: [describe, query, memory_write]
inputs:
  - the visible tables
outputs: [a short list of robots that stand out, with their success rates]
version: "1"
---
# Triage

1. Read the data dictionary once with describe (part "dictionary", include_logs
   false).
2. Count episodes and successes per robot: `select robot_id, count(*) as n,
   avg((outcome = 'success')::int) as rate from episodes group by robot_id order by rate`.
3. A robot stands out when its rate is lower than every other robot's. Note
   the rates and the robot in your memory with memory_write.
"""


def write_toy_skill(skills_root: Path) -> Path:
    skill = skills_root / "triage"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(TOY_SKILL)
    return skill


@dataclass(frozen=True)
class ToyWorld:
    base: Path
    run_dir: Path  # the run, with its truth side
    export: Path  # what the tool server mounts
    fleet: Fleet
    skills: Path  # a skill library with the one toy skill


def toy_world(base: Path, *, fresh: bool = False) -> ToyWorld:
    """Write the toy run and export it under `base` (once; again with `fresh`)."""
    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    log_toy = importlib.import_module("log_toy")
    TASK_CLASSES, toy_fleet, write_toy_run = (
        log_toy.TASK_CLASSES,
        log_toy.toy_fleet,
        log_toy.write_toy_run,
    )
    if fresh and base.exists():
        shutil.rmtree(base)
    run_dir, export, skills = base / "run", base / "export", base / "skills"
    # Rebuilt when missing or when the schema moved on since it was written.
    if not (export / "log").is_dir() or export_problems(export):
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        fleet = write_toy_run(run_dir)
        export_runs([run_dir], export, fleet=fleet, task_classes=TASK_CLASSES)
    if not (skills / "triage" / "SKILL.md").is_file():
        write_toy_skill(skills)
    return ToyWorld(base, run_dir, export, toy_fleet(), skills)
