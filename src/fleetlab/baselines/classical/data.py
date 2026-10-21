"""Read only the named, visible export files; never discover a world root."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fleetlab.contracts.actions import Action, CostKey
from fleetlab.contracts.faults import Target
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.vocabulary import Component, TargetKind, Verb

Row = dict[str, Any]


def visible_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"export file escapes its root: {relative}")
    if not path.is_file():
        raise ValueError(f"missing visible export file: {relative}")
    return path


@dataclass(frozen=True)
class VisibleData:
    fleet: Fleet
    episodes: list[Row]
    health: list[Row]
    changes: list[Row]
    events: list[Row]
    actions: list[Action]
    costs: dict[CostKey, float | None]
    hashes: dict[str, str]
    maintenance: list[Row] = field(default_factory=list)


def load_export(root: Path, catalogue: Path | None = None) -> VisibleData:
    """The export root contains log/ and data_dictionary.md, not runs/ or truth/.

    A separately supplied catalogue is an explicit visible input. Only regular,
    observed episodes (days 0..5) enter a fit. Later ordinary rows are refused,
    not silently admitted. Experiment rows are excluded, including their events.
    """
    root = root.expanduser().resolve()
    hashes: dict[str, str] = {}

    def path(name: str) -> Path:
        found = visible_file(root, name)
        hashes[name] = hashlib.sha256(found.read_bytes()).hexdigest()
        return found

    path("data_dictionary.md")
    raw = json.loads(path("log/description.json").read_text())
    fleet = Fleet.model_validate(raw["fleet"])
    if any(r.traits for r in fleet.robots()):
        raise ValueError("description contains hidden robot traits")
    episodes = pq.read_table(path("log/episodes.parquet")).to_pylist()
    episodes = [r for r in episodes if not r.get("experiment_id")]
    if not episodes or any(not 0 <= r["day"] <= 5 for r in episodes):
        raise ValueError("expected nonempty observed window, days 0 through 5 only")
    if len({r["episode_id"] for r in episodes}) != len(episodes):
        raise ValueError("duplicate episode ids")
    for r in episodes:
        if r["outcome"] not in {"success", "failure", "handed_over"}:
            raise ValueError("unknown episode outcome")
        if fleet.robot(r["robot_id"]).site_id != r["site_id"]:
            raise ValueError("episode robot/site disagrees with description")
    health = pq.read_table(path("log/health.parquet")).to_pylist()
    changes = pq.read_table(path("log/change_log.parquet")).to_pylist()
    maintenance = pq.read_table(path("log/maintenance.parquet")).to_pylist()
    dated = [*health, *changes, *maintenance]
    if any(not 0 <= r["day"] <= 5 for r in dated):
        raise ValueError("non-observed health, change or maintenance rows")
    versions = [fleet.config, *(s.config for s in fleet.sites)]
    versions.extend(r.config for r in fleet.robots())
    if (
        any(v.since_day > 5 for v in versions)
        or any(c.day > 5 for c in fleet.change_log)
        or any(m.day > 5 for m in fleet.maintenance_log)
    ):
        raise ValueError("description includes a future configuration/log")
    for r in changes:
        r["old"], r["new"] = json.loads(r["old"]), json.loads(r["new"])
    # Parquet is the authoritative observed change log, not a second copy in JSON.
    episode_ids = {r["episode_id"] for r in episodes}
    events = pq.read_table(path("log/events.parquet")).to_pylist()
    events = [r for r in events if r["episode_id"] in episode_ids]
    if catalogue is None:
        catalogue = path("log/catalogue.parquet")
    else:
        catalogue = catalogue.expanduser().resolve()
        hashes["supplied_catalogue"] = hashlib.sha256(catalogue.read_bytes()).hexdigest()
    actions = []
    for r in pq.read_table(catalogue).to_pylist():
        actions.append(
            Action(
                verb=r["verb"],
                target=Target(
                    scope=r["scope"],
                    component=r["component"],
                    config_key=r["config_key"],
                    condition_key=r["condition_key"],
                ),
                value=r["value"],
            )
        )
    costs: dict[CostKey, float | None] = {}
    for r in pq.read_table(path("log/costs.parquet")).to_pylist():
        named = r["target"]
        kind = (
            None
            if named is None
            else (Component(named) if named in Component._value2member_map_ else TargetKind(named))
        )
        costs[(Verb(r["verb"]), kind)] = r["hours"]
    return VisibleData(
        fleet,
        sorted(episodes, key=lambda r: r["episode_id"]),
        health,
        sorted(changes, key=lambda r: r["day"]),
        events,
        actions,
        costs,
        hashes,
        maintenance,
    )
