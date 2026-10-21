"""A finished world as the oracle reads it, and where the oracle writes.

The world root (agreed between the world generator and the oracle; the miniature fleet's
layout, `fleetlab.integration.chain.ChainPaths`, is one):

```
<world root>/
  files/          spec.json, model.json, description.json, manifest.jsonl, hash (contracts
                  section 9) and on the truth side beside them fault_table.json,
                  world_truth.json, task_texts.json (the miniature fleet: world/)
  runs/<name>/    the engine's run directories of the world as run, each with run.json
  runner.json     the fleet runner's record; "ready": true once the runs are final
  oracle/         the oracle's own (`OracleLayout`), never written by anyone else
```

What is read: the spec (windows, patterns: decoys, confounders, interactions, drifts),
the manifest's "run" jobs (each identity with the faults in force on its day), the
hidden fault table (`fault_table.json`: each fault's role) and the detectability list
(`detectability.json`, the world generator's causes: a fault, or the faults an
interaction joins, each detectable or not), the fleet at the end of the world
(`model.json`, for `resolve`'s current values), the observed fleet
(`world_truth.json`'s `fleet`, for the filled catalogue) and the task texts. Everything
but the spec and the manifest is optional: what is missing is reported where it is
needed. Nothing here writes into the world directory or the runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import TruthTable
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.ids import FaultId
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey
from fleetlab.contracts.world import HASH_FILE, Patterns, WorldSpec
from fleetlab.worlds.truth import FAULT_TABLE_FILE, FaultTable, FaultTruth

if TYPE_CHECKING:
    from fleetlab.worlds.generate import World

WORLD_SUBDIRS = ("files", "world")
RUNS_SUBDIR = "runs"
RUNNER_FILE = "runner.json"
ORACLE_SUBDIR = "oracle"
DEFAULT_REGISTRY = "fleetlab.faults:REGISTRY"


class WorldError(ValueError):
    """The world directory lacks what the oracle needs, or contradicts itself."""


@dataclass(frozen=True)
class OracleWorld:
    """The parts of a world the oracle reads."""

    spec: WorldSpec
    identities: tuple[EpisodeIdentity, ...]  # every as-run episode, in manifest order
    fault_table: FaultTable | None = None
    causes: tuple[Cause, ...] = ()  # the generator's detectability list, if it wrote one
    fleet: Fleet | None = None  # the model at the end of the world (truth)
    observed_fleet: Fleet | None = None  # the records as the observed window ends
    task_classes: tuple[str, ...] = ()
    world_hash: str | None = None
    world_dir: Path | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    # The fault registry whose truth mappings resolve actions ("module:NAME"); the toy
    # world of the tests names its own.
    registry: str = DEFAULT_REGISTRY
    # Read the truth-only extension without editing the shared contract in this lane.
    # Once Patterns includes additive_pairs, the loader reads it from there as well.
    additive_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def world_id(self) -> str:
        return self.spec.world_id

    def window(self, name: str) -> list[EpisodeIdentity]:
        days = getattr(self.spec.windows, name)
        return [i for i in self.identities if i.day in days]

    def scoring(self) -> list[EpisodeIdentity]:
        """The scoring window's episodes, in manifest order."""
        return self.window("scoring")

    def by_id(self) -> dict[str, EpisodeIdentity]:
        return {i.episode_id: i for i in self.identities}

    @property
    def excluded(self) -> frozenset[str]:
        """Faults the contract's power set leaves in force: decoys and inert confounder entries."""
        return self.spec.patterns.excluded_from_replay()

    @property
    def decoys(self) -> frozenset[str]:
        return frozenset(self.spec.patterns.decoys)

    @property
    def inert(self) -> frozenset[str]:
        return frozenset(inert for _, inert in self.spec.patterns.confounders)

    def truth(self) -> TruthTable:
        """Every registered fault kind's truth mapping (the registry's, `fleetlab.faults`)."""
        import importlib

        module, _, name = self.registry.partition(":")
        registry = getattr(importlib.import_module(module), name)
        table: TruthTable = registry.truth_table()
        return table

    def current(
        self,
    ) -> Callable[[EpisodeIdentity], Mapping[ConfigKey | ConditionKey, ConfigValue]] | None:
        """`resolve`'s current values from the fleet at the end of the world, if known."""
        from fleetlab.oracle.valuation import current_values

        return None if self.fleet is None else current_values(self.fleet)

    def catalogue(self) -> list[Action]:
        """The filled catalogue from the observed fleet (else the final one); empty if neither."""
        from fleetlab.oracle.valuation import filled_catalogue

        fleet = self.observed_fleet or self.fleet
        return [] if fleet is None else filled_catalogue(fleet, list(self.task_classes))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _manifest(path: Path) -> list[EpisodeIdentity]:
    out: list[EpisodeIdentity] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        job = Job.model_validate_json(line)
        if job.mode is not JobMode.run:
            raise WorldError(f"{path}:{number}: a world's manifest holds run jobs, not {job.mode}")
        out.append(job.identity)
    ids = [i.episode_id for i in out]
    if len(ids) != len(set(ids)):
        raise WorldError(f"{path}: an episode id appears twice")
    return out


DETECTABILITY_FILE = "detectability.json"


@dataclass(frozen=True)
class Cause:
    """One cause of the generator's detectability list (experimental design, section 5)."""

    unit: str  # "f03", or "f01+f02" for the faults an interaction joins
    fault_ids: tuple[str, ...]
    roles: tuple[str, ...]
    detectable: bool
    qualified_by: tuple[str, ...]
    expected_points: float | None  # the generator's expected effect over the scoring window
    counts: bool | None  # whether it counts for the gate's sizing rule
    data: bool = False  # a data fault: it changes the logged outcome, not what the robot does


def _causes(raw: Any) -> tuple[Cause, ...]:
    out: list[Cause] = []
    for c in raw.get("causes", []) if isinstance(raw, dict) else []:
        expected = c.get("expected_effect_points") or {}
        out.append(
            Cause(
                unit=str(c["unit"]),
                fault_ids=tuple(str(f) for f in c["fault_ids"]),
                roles=tuple(str(r) for r in c.get("roles", [])),
                detectable=bool(c.get("detectable")),
                qualified_by=tuple(str(w) for w in c.get("qualified_by", [])),
                expected_points=expected.get("scoring") if isinstance(expected, dict) else None,
                counts=c.get("counts_for_gate"),
                data=bool(c.get("data_fault", False)),
            )
        )
    return tuple(out)


def world_dir_of(root: Path) -> Path:
    """The directory holding the world files: `root/files` (the world generator's),
    `root/world` (the miniature fleet's), or `root` itself."""
    root = Path(root)
    for sub in WORLD_SUBDIRS:
        if (root / sub / "spec.json").is_file():
            return root / sub
    if (root / "spec.json").is_file():
        return root
    raise WorldError(f"{root}: no world files (spec.json) in it or in {', '.join(WORLD_SUBDIRS)}")


def world_ready(root: Path) -> str | None:
    """Why the world's runs are not ready for the oracle, or None: the fleet runner
    writes `runner.json` with `"ready": true` once both runs are finalized
    and proven; a root without it (the miniature fleet) is taken as ready."""
    from fleetlab.worlds.layout import WorldPaths

    try:
        paths = WorldPaths.from_root(root)
    except ValueError as error:
        return str(error)
    marker = paths.record
    if not marker.is_file():
        return None
    ready = paths.ready
    return None if ready is True else f"{marker}: ready is {ready!r}"


def load_world(root: Path | str, registry: str = DEFAULT_REGISTRY) -> OracleWorld:
    """Read a world from its root (with `world/`) or from its world directory itself."""
    world_dir = world_dir_of(Path(root).expanduser())
    raw_spec = _read_json(world_dir / "spec.json")
    additive_pairs = TypeAdapter(list[tuple[FaultId, FaultId]]).validate_python(
        raw_spec.get("patterns", {}).get("additive_pairs", [])
    )
    if "additive_pairs" not in Patterns.model_fields:
        raw_spec.get("patterns", {}).pop("additive_pairs", None)
    spec = WorldSpec.model_validate(raw_spec)
    scheduled = {f.fault_id for f in spec.fault_schedule}
    if any(a == b or not {a, b} <= scheduled for a, b in additive_pairs):
        raise WorldError("additive_pairs must name two distinct scheduled faults")
    identities = _manifest(world_dir / "manifest.jsonl")
    notes: list[str] = []
    fault_table = None
    table_path = world_dir / FAULT_TABLE_FILE
    if table_path.is_file():
        raw_table = _read_json(table_path)
        # The parallel generator adds this curve annotation. Measured membership
        # uses neither its drift forecast nor its crossing day.
        if "drift_crossing_day" not in FaultTruth.model_fields:
            for row in raw_table.get("faults", []):
                row.pop("drift_crossing_day", None)
        fault_table = FaultTable.model_validate(raw_table)
    else:
        notes.append(f"no {FAULT_TABLE_FILE}: the roles come from the spec alone")
    causes: tuple[Cause, ...] = ()
    if (world_dir / DETECTABILITY_FILE).is_file():
        causes = _causes(_read_json(world_dir / DETECTABILITY_FILE))
    elif fault_table is None:
        notes.append("no detectability list: the detectability check is skipped")
    fleet = None
    if (world_dir / "model.json").is_file():
        fleet = Fleet.model_validate_json((world_dir / "model.json").read_text())
    else:
        notes.append("no model.json: actions that set a key no fault touches cannot be resolved")
    observed = None
    truth_file = world_dir / "world_truth.json"
    if truth_file.is_file():
        raw_truth = _read_json(truth_file)
        if isinstance(raw_truth, dict) and "fleet" in raw_truth:
            observed = Fleet.model_validate(raw_truth["fleet"])
    if observed is None:
        notes.append("no observed fleet (world_truth.json): the catalogue is the final fleet's")
    texts: tuple[str, ...] = ()
    texts_file = world_dir / "task_texts.json"
    if texts_file.is_file():
        by_task = {
            (t["suite"], int(t["task_index"])): str(t["text"]) for t in _read_json(texts_file)
        }
        lacking = [t for t in spec.tasks if (t.suite, t.task_index) not in by_task]
        if lacking:
            notes.append(f"task_texts.json lacks {len(lacking)} of the world's tasks")
        else:
            texts = tuple(by_task[(t.suite, t.task_index)] for t in spec.tasks)
    stored = world_dir / HASH_FILE
    return OracleWorld(
        spec=spec,
        identities=tuple(identities),
        fault_table=fault_table,
        causes=causes,
        fleet=fleet,
        observed_fleet=observed,
        task_classes=texts,
        world_hash=stored.read_text().strip() if stored.is_file() else None,
        world_dir=world_dir,
        notes=tuple(notes),
        registry=registry,
        additive_pairs=tuple(additive_pairs),
    )


def from_generated(world: World, world_dir: Path | None = None) -> OracleWorld:
    """The oracle's view of a world generated in memory (`fleetlab.worlds.generate`)."""
    stored = None if world_dir is None else Path(world_dir) / HASH_FILE
    return OracleWorld(
        spec=world.spec,
        identities=tuple(j.identity for j in world.jobs),
        fault_table=world.fault_table,
        fleet=world.fleet,
        observed_fleet=world.observed_fleet,
        task_classes=tuple(world.task_classes),
        world_hash=stored.read_text().strip() if stored is not None and stored.is_file() else None,
        world_dir=None if world_dir is None else Path(world_dir),
    )


def world_runs(root: Path | str) -> list[Path]:
    """The run directories under `root/runs` that hold a run.json, sorted by name."""
    runs = Path(root).expanduser() / RUNS_SUBDIR
    if not runs.is_dir():
        return []
    return sorted(p for p in runs.iterdir() if (p / "run.json").is_file())


@dataclass(frozen=True)
class OracleLayout:
    """Where the oracle writes, under `<world root>/oracle` unless told otherwise; the
    replays' run is `oracle/runs/<world>-oracle` (agreed with the fleet runner)."""

    root: Path
    world_id: str = "world"

    @property
    def plan(self) -> Path:
        return self.root / "plan.json"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs.jsonl"

    @property
    def run(self) -> Path:
        """The engine run of the planned replays."""
        return self.root / RUNS_SUBDIR / f"{self.world_id}-oracle"

    @property
    def report(self) -> Path:
        return self.root / "oracle.json"

    @property
    def summary(self) -> Path:
        return self.root / "summary.md"

    @property
    def episodes(self) -> Path:
        return self.root / "episodes.parquet"

    @property
    def steps(self) -> Path:
        """When each step started and ended (light and heavy), for the README's timings."""
        return self.root / "steps.json"

    def fresh(self, name: str) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"a fresh replay set's name is a plain word, not {name!r}")
        return self.root / "fresh" / name

    def fresh_runs(self) -> list[Path]:
        base = self.root / "fresh"
        if not base.is_dir():
            return []
        return sorted(p / "run" for p in base.iterdir() if (p / "run" / "run.json").is_file())


def oracle_runs(layout: OracleLayout) -> list[Path]:
    """The oracle's own run directories that exist: the planned replays, then every fresh set."""
    own = [layout.run] if (layout.run / "run.json").is_file() else []
    verification = layout.root / "verification" / "run"
    causes = layout.root / "causes" / "run"
    return (
        own
        + layout.fresh_runs()
        + ([verification] if (verification / "run.json").is_file() else [])
        + ([causes] if (causes / "run.json").is_file() else [])
    )


def all_runs(root: Path | str, layout: OracleLayout, extra: Sequence[Path] = ()) -> list[Path]:
    """Every run whose rows the table reads: the world's, the oracle's own, and `extra`."""
    seen: list[Path] = []
    for path in [*world_runs(root), *oracle_runs(layout), *(Path(p) for p in extra)]:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.append(resolved)
    return seen


def recorded_sources(layout: OracleLayout, world_hash: str | None) -> list[Path]:
    """Explicit sources persist across oracle and scorer invocations."""
    sources: list[Path] = []
    for name in ("sources.json", "oracle.json"):
        path = layout.root / name
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        if record.get("world_hash") != world_hash:
            raise ValueError(f"{path}: sources belong to another world")
        sources.extend(Path(r if isinstance(r, str) else r["run_dir"]) for r in record["runs"])
    return sources
