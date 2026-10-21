"""Visible inputs, seed-only chance metadata and stage-2 answer records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import SCORING_BUDGETS, Action, FixSet
from fleetlab.contracts.faults import Target
from fleetlab.contracts.log import DESCRIPTION_FILE
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.tools import Interval, RankedAction, RankedFixSet
from fleetlab.contracts.vocabulary import Component, Verb
from fleetlab.log.reader import LogReader

WORDING_VERSION = "baselines-v4"


def validate_export(export: Path) -> None:
    """Refuse run/world roots and symlinks before any visible table is queried."""
    if not export.is_dir() or not (export / "data_dictionary.md").is_file():
        raise ValueError("--export must be an export with data_dictionary.md, not a run directory")
    if any(p.name not in {"log", "data_dictionary.md"} for p in export.iterdir()):
        raise ValueError("--export contains files outside the export allowlist")
    if any(p.is_symlink() for p in export.rglob("*")):
        raise ValueError("--export must not contain symbolic links")


def catalogue_templates(export: Path) -> list[Action]:
    """All offered verb/target pairs, without recorded configuration values."""
    with LogReader(export) as reader:
        rows = reader.query("SELECT * FROM catalogue").to_pylist()
    return [
        Action(
            verb=row["verb"],
            target=Target(
                scope=row["scope"],
                component=row["component"],
                config_key=row["config_key"],
                condition_key=row["condition_key"],
            ),
            # Task-class entries are static catalogue values, not configuration readings.
            value=row["value"] if row["verb"] == "route_to_teleop" else None,
        )
        for row in rows
    ]


COMPONENT_FIXES = {
    Component.scene_camera: Verb.recalibrate,
    Component.wrist_camera: Verb.recalibrate,
    Component.gripper: Verb.service,
    Component.arm: Verb.service,
    Component.controller: Verb.service,
}


def budget_label(kind: str, amount: float) -> str:
    return f"{kind}-{amount:g}"


@dataclass
class BaselineAnswer:
    """The canonical ranking and optional rankings for individual budget conditions."""

    method: str
    ranked: RankedFixSet
    by_budget: dict[str, RankedFixSet] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    explicit_fix_sets: dict[str, FixSet] = field(default_factory=dict)

    def fix_sets(self) -> dict[str, FixSet]:
        return {
            **{
                budget_label(b.kind, b.amount): self.by_budget.get(
                    budget_label(b.kind, b.amount), self.ranked
                ).as_fix_set(b)
                for b in SCORING_BUDGETS
            },
            **self.explicit_fix_sets,
        }


def ranking(actions: Sequence[Action], rationales: Sequence[str] | None = None) -> RankedFixSet:
    """Ordinal methods do not estimate percentage-point gains; zero is a labelled placeholder."""
    return RankedFixSet(
        actions=[
            RankedAction(
                action=a,
                expected_gain=0.0,
                uncertainty=Interval(low=-100.0, high=100.0),
                rationale=(
                    rationales[i] if rationales else "Ordinal baseline; gain not estimated."
                ),
            )
            for i, a in enumerate(actions)
        ]
    )


def static_description(export: Path) -> dict[str, Any]:
    """An allowlist: topology and vocabulary, never health, dates, settings or logs."""
    raw = json.loads((export / DESCRIPTION_FILE).read_text())
    return {
        "sites": [
            {
                "id": s["id"],
                "robots": [
                    {"id": r["id"], "components": sorted(r["components"])} for r in s["robots"]
                ],
            }
            for s in raw["fleet"]["sites"]
        ],
        "vocabulary": raw["vocabulary"],
        "costs": raw["costs"],
    }


def visible_fleet(export: Path) -> Fleet:
    return Fleet.model_validate(json.loads((export / DESCRIPTION_FILE).read_text())["fleet"])


def export_catalogue(export: Path) -> list[Action]:
    """Fill the visible catalogue by the stage-2 operator rule, without importing truth code."""
    fleet = visible_fleet(export)
    with LogReader(export) as reader:
        rows = reader.query("SELECT * FROM catalogue").to_pylist()
    entries = []
    for row in rows:
        target = Target(
            scope=row["scope"],
            component=row["component"],
            config_key=row["config_key"],
            condition_key=row["condition_key"],
        )
        action = Action(verb=row["verb"], target=target, value=row["value"])
        if not action.is_complete:
            if action.verb == Verb.set:
                assert target.config_key is not None
                value = fleet.config.values[target.config_key]
            elif action.verb == Verb.change_condition:
                assert target.condition_key is not None
                recorded = fleet.site(target.scope).conditions
                if target.condition_key not in recorded:
                    continue
                value = recorded[target.condition_key]
            elif action.verb == Verb.request_training_data:
                value = "any failure mode"
            else:
                continue
            action = Action(verb=action.verb, target=target, value=value)
        entries.append(action)
    if len(entries) < 10:
        raise ValueError("the stage-2 answer format needs at least ten catalogue actions")
    if len({a.model_dump_json() for a in entries}) != len(entries):
        raise ValueError("duplicate catalogue actions")
    return entries


def component_actions(export: Path) -> list[Action]:
    fleet = visible_fleet(export)
    allowed = {a.model_dump_json() for a in export_catalogue(export)}
    return [
        a
        for r in sorted(fleet.robots(), key=lambda r: r.id)
        for c, verb in COMPONENT_FIXES.items()
        if c in r.components
        and (a := Action(verb=verb, target=Target(scope=r.id, component=c))).model_dump_json()
        in allowed
    ]


def write_answer(root: Path, answer: BaselineAnswer) -> None:
    """Write only new files. Analysis(root).fix_set() reads outbox/fix_set.json."""
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "outbox/fix_set.json", answer.ranked.model_dump(mode="json"))
    for budget in SCORING_BUDGETS:
        label = budget_label(budget.kind, budget.amount)
        selected = answer.by_budget.get(label, answer.ranked)
        write_json(root / f"budgets/{label}/outbox/fix_set.json", selected.model_dump(mode="json"))
        write_json(
            root / f"fix_sets/{label}.json", answer.fix_sets()[label].model_dump(mode="json")
        )
    write_json(
        root / "metadata.json",
        {
            "method": answer.method,
            "wording_version": WORDING_VERSION,
            "expected_gain": "not estimated; zero placeholder, interval is full possible range",
            **answer.metadata,
        },
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
