"""What the dashboard reads from a run directory or an export, and the tables it derives.

A *source* is a run directory (`log/`, `truth/`, `run.json`) or an export
(`log/` only). The visible side is read through `fleetlab.log.LogReader`; the
hidden side, only when the page asks for it, through `fleetlab.log.runfiles`.
Everything here is a pure function of the files: no clock, no randomness, and
every collection is sorted before it is returned.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.log import DESCRIPTION_FILE
from fleetlab.contracts.oracle import OracleTable, fault_effect
from fleetlab.contracts.rollout import RunLayout, RunRecord
from fleetlab.dose_response import WORLD_ID as CALIBRATION_WORLD
from fleetlab.dose_response import wilson
from fleetlab.log import LogReader, RunStatus, run_status
from fleetlab.log.runfiles import oracle_rows, read_identities, read_table, table_path

__all__ = [
    "CALIBRATION_WORLD",
    "TRUTH_KINDS",
    "DosePoint",
    "FaultGain",
    "FaultRow",
    "Rate",
    "Source",
    "Truth",
    "WorldRegistry",
    "dose_points",
    "faults_by",
    "load_source",
    "load_truth",
    "rates_by",
    "sample_episodes",
]

# World kinds whose hidden side the dashboard may show (plan: development worlds only;
# the smaller worlds of the ladder and the engine's probes are development material too).
TRUTH_KINDS = frozenset({"probe", "smoke", "nightly", "miniature", "calibration", "development"})
# World kinds nobody on the development side looks at before the test (plan, honesty rule 3).
SEALED_KINDS = frozenset({"test", "reserve"})
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Rate:
    """Successes out of n, with Wilson's 95% interval."""

    n: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.successes, self.n)

    def text(self) -> str:
        if self.n == 0:
            return "no episodes"
        low, high = self.interval
        return f"{100 * self.rate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"

    @staticmethod
    def of(outcomes: Iterable[bool]) -> Rate:
        values = list(outcomes)
        return Rate(n=len(values), successes=sum(values))


@dataclass(frozen=True)
class WorldRegistry:
    """`dashboard/worlds.json`: the kind of each world id the dashboard knows."""

    kinds: dict[str, str]
    notes: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def load(path: Path | None) -> WorldRegistry:
        if path is None or not path.is_file():
            return WorldRegistry({})
        data = json.loads(path.read_text())
        kinds = {w: str(v["kind"]) for w, v in data.items()}
        notes = {w: str(v.get("note", "")) for w, v in data.items()}
        return WorldRegistry(kinds, notes)

    def kind(self, world_id: str) -> str:
        return self.kinds.get(world_id, "unregistered")

    def registered(self, world_id: str) -> bool:
        return world_id in self.kinds

    def truth_allowed(self, world_id: str) -> bool:
        return self.kind(world_id) in TRUTH_KINDS

    def sealed(self, world_id: str) -> bool:
        return self.kind(world_id) in SEALED_KINDS


@dataclass(frozen=True)
class Source:
    """One run directory or export, visible side, with its state."""

    run_id: str
    root: Path
    is_run: bool  # a run directory with truth/; False for an export
    episodes: list[dict[str, Any]]  # the visible episodes rows, by episode id
    world_ids: tuple[str, ...]
    status: RunStatus | None
    record: RunRecord | None

    @property
    def overall(self) -> Rate:
        return Rate.of(e["outcome"] == "success" for e in self.episodes)

    @property
    def state(self) -> str:
        if not self.is_run:
            return "export"
        assert self.status is not None
        if self.status.finalized:
            return "finalized"
        status = self.status
        merged = not status.partial and set(status.done) == set(status.merged)
        if merged and not status.staging_files and status.missing:
            missing = (
                ", ".join(status.missing)
                if len(status.missing) <= 3
                else f"{len(status.missing)} run-level tables"
            )
            return (
                f"complete ({len(status.done)} jobs, all merged), but without {missing}: "
                "finalized by an older log version, or the table was added since"
            )
        parts = [f"{len(status.done)} jobs done"]
        if status.partial:
            parts.append(f"{len(status.partial)} partial")
        if set(status.done) != set(status.merged):
            parts.append(f"{len(set(status.done) - set(status.merged))} not merged")
        if 0 < len(status.missing) <= 3:
            parts.append("missing " + ", ".join(status.missing))
        elif status.missing:
            parts.append(f"{len(status.missing)} run-level tables missing")
        if status.staging_files:
            parts.append(f"{status.staging_files} files in staging")
        return "not finalized (" + "; ".join(parts) + ")"

    @property
    def world(self) -> str:
        return ", ".join(self.world_ids) if self.world_ids else "unknown"

    def reader(self) -> LogReader:
        return LogReader(self.root)


def check_run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id):
        raise ValueError(
            f"run id {run_id!r} (the directory's name) is not a plain name: "
            "letters, digits, '.', '_' and '-', at most 64"
        )
    return run_id


def _world_ids(root: Path, is_run: bool) -> tuple[str, ...]:
    """The worlds of a run's episodes (its identities), or of an export (its description)."""
    if is_run and table_path(root, "identities").is_file():
        return tuple(sorted({i.world_id for i in read_identities(root)}))
    description = root / DESCRIPTION_FILE
    if description.is_file():
        fleet_id = json.loads(description.read_text()).get("fleet", {}).get("id")
        if fleet_id:
            return (str(fleet_id),)
    return ()


def load_source(root: Path | str, run_id: str | None = None) -> Source:
    """Read the visible side and the state of a run directory or an export."""
    path = Path(root).resolve()
    if not (path / "log").is_dir():
        raise FileNotFoundError(f"{path} has no log/ directory: not a run or an export")
    is_run = (path / "truth").is_dir()
    with LogReader(path) as reader:
        episodes: list[dict[str, Any]] = reader.query(
            "SELECT * FROM episodes ORDER BY episode_id"
        ).to_pylist()
    status = run_status(path) if is_run else None
    record = None
    layout = RunLayout(path)
    if is_run and layout.run_record.is_file():
        record = RunRecord.model_validate_json(layout.run_record.read_text())
    return Source(
        run_id=check_run_id(run_id or path.name),
        root=path,
        is_run=is_run,
        episodes=episodes,
        world_ids=_world_ids(path, is_run),
        status=status,
        record=record,
    )


def rates_by(
    episodes: Sequence[dict[str, Any]], key: str | Callable[[dict[str, Any]], Any]
) -> list[tuple[Any, Rate]]:
    """Success by a column of `episodes` (or a function of a row), sorted by its value."""
    get = (lambda e: e[key]) if isinstance(key, str) else key
    groups: dict[Any, list[bool]] = defaultdict(list)
    for e in episodes:
        groups[get(e)].append(e["outcome"] == "success")
    return [(value, Rate.of(groups[value])) for value in sorted(groups, key=_sort_key)]


def _sort_key(value: Any) -> tuple[int, Any]:
    return (0, value) if isinstance(value, int | float) else (1, str(value))


def sample_episodes(episodes: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """A deterministic sample of episodes with a scene video: failures and successes alike.

    Half (rounded up) are failures, the rest successes, each picked evenly
    spaced through its list in episode order, so that the sample spreads over
    robots and days as the episode numbering does; if one kind runs short the
    other fills in. Returned in episode order.
    """
    with_video = [e for e in episodes if e.get("video_scene")]
    failures = [e for e in with_video if e["outcome"] != "success"]
    successes = [e for e in with_video if e["outcome"] == "success"]
    want_failures = min(len(failures), (count + 1) // 2)
    want_successes = min(len(successes), count - want_failures)
    want_failures = min(len(failures), count - want_successes)
    chosen = _evenly(failures, want_failures) + _evenly(successes, want_successes)
    return sorted(chosen, key=lambda e: str(e["episode_id"]))


def _evenly(items: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    if k == 1:
        return [items[0]]
    n = len(items)
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


# ---------------------------------------------------------------------------- the hidden side


@dataclass(frozen=True)
class FaultRow:
    """A fault as the run applied it, with the visible outcome of the episodes it was active in."""

    fault_id: str
    kind: str
    level: float  # the lowest level in any episode
    level_high: float  # the highest: a drifting fault's level changes by day
    target: str
    onset_day: int
    episodes: Rate  # the visible outcome of the episodes it was active in

    @property
    def levels(self) -> str:
        if self.level == self.level_high:
            return f"{self.level:g}"
        return f"{self.level:g} to {self.level_high:g}"

    @property
    def label(self) -> str:
        return f"{self.fault_id} {self.kind} at {self.levels}"


@dataclass(frozen=True)
class FaultGain:
    """The oracle's gain of removing one fault wherever it is active, over the replayed episodes."""

    fault_id: str
    active: int  # episodes as run with the fault active
    covered: int  # of those, the episodes with a replay without it
    rescued: int
    broken: int

    @property
    def net(self) -> int:
        return self.rescued - self.broken


@dataclass(frozen=True)
class DosePoint:
    """A calibration point: a fault kind at a level (None for the healthy episodes)."""

    kind: str | None
    level: float | None
    rate: Rate
    runs: str = ""  # the other runs a healthy point was pooled from; empty for this run's own


@dataclass(frozen=True)
class Truth:
    """The hidden side of a run, as far as the dashboard shows it."""

    identities: dict[str, EpisodeIdentity]
    faults_active: list[dict[str, Any]]  # rows of truth/faults_active, by episode and fault
    faults: list[FaultRow]
    true_success: dict[str, bool]  # the fleet's check before the data hook, by episode
    oracle: list[dict[str, Any]]  # rows of truth/oracle.parquet as written
    gains: list[FaultGain]
    oracle_problem: str | None  # why the per-fault gains could not be computed, if so


def _rows(root: Path, table: str) -> list[dict[str, Any]]:
    if not table_path(root, table).is_file():
        return []
    rows: list[dict[str, Any]] = read_table(root, table).to_pylist()
    return rows


def _target(row: dict[str, Any]) -> str:
    parts = [row["target_scope"]]
    parts += [
        row[k]
        for k in ("target_component", "target_config_key", "target_condition_key")
        if row.get(k)
    ]
    return " ".join(str(p) for p in parts)


def load_truth(source: Source, *, gains: bool = True) -> Truth:
    """Read the hidden side of a run directory. Never for an agent.

    A table that does not exist yet (a run in progress) reads as empty; the
    oracle's gains then say what is missing. `gains=False` skips them.
    """
    if not source.is_run:
        raise ValueError(f"{source.run_id} is an export: it has no hidden side")
    root = source.root
    identities = {}
    if table_path(root, "identities").is_file():
        identities = {i.episode_id: i for i in read_identities(root)}
    visible = {e["episode_id"]: e for e in source.episodes}
    active = sorted(
        _rows(root, "faults_active"), key=lambda r: (str(r["episode_id"]), str(r["fault_id"]))
    )
    by_fault: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in active:
        by_fault[row["fault_id"]].append(row)
    faults = []
    for fault_id in sorted(by_fault):
        rows = by_fault[fault_id]
        first = rows[0]
        levels = sorted({float(r["level"]) for r in rows})
        outcomes = [
            visible[r["episode_id"]]["outcome"] == "success"
            for r in rows
            if r["episode_id"] in visible
        ]
        faults.append(
            FaultRow(
                fault_id=fault_id,
                kind=str(first["kind"]),
                level=levels[0],
                level_high=levels[-1],
                target=_target(first),
                onset_day=int(first["onset_day"]),
                episodes=Rate.of(outcomes),
            )
        )
    verdicts = {r["episode_id"]: bool(r["true_success"]) for r in _rows(root, "verdicts")}
    oracle = sorted(_rows(root, "oracle"), key=lambda r: (str(r["item_kind"]), str(r["item"])))
    fault_gains, problem = _fault_gains(root, identities, verdicts) if gains else ([], None)
    return Truth(
        identities=identities,
        faults_active=active,
        faults=faults,
        true_success=verdicts,
        oracle=oracle,
        gains=fault_gains,
        oracle_problem=problem,
    )


def _fault_gains(
    root: Path, identities: dict[str, EpisodeIdentity], verdicts: dict[str, bool]
) -> tuple[list[FaultGain], str | None]:
    """Per fault, the gain of removing it, over the episodes whose replay without it exists."""
    missing = [
        t for t in ("identities", "verdicts", "replays") if not table_path(root, t).is_file()
    ]
    if missing:
        return [], "missing " + ", ".join(f"truth/{t}.parquet" for t in missing)
    as_run = [identities[e] for e in sorted(verdicts) if e in identities]
    try:
        table = OracleTable(as_run, oracle_rows(root))
    except (ValueError, KeyError) as error:
        return [], f"the oracle rows do not form a table: {error}"
    gains = []
    for fault_id in table.fault_ids():
        active = [i for i in as_run if fault_id in i.fault_ids]
        covered = [
            i
            for i in active
            if table.success(i.episode_id, [f for f in i.faults if f.fault_id != fault_id])
            is not None
        ]
        gain = fault_effect(fault_id, table.restricted(covered))
        gains.append(
            FaultGain(
                fault_id=fault_id,
                active=len(active),
                covered=len(covered),
                rescued=gain.rescued,
                broken=gain.broken,
            )
        )
    return gains, None


def faults_by(truth: Truth, episodes: Sequence[dict[str, Any]], key: str) -> dict[Any, list[str]]:
    """The faults active in the episodes of each value of a visible column (robot, site, day)."""
    value_of = {e["episode_id"]: e[key] for e in episodes}
    labels = {f.fault_id: f.label for f in truth.faults}
    out: dict[Any, set[str]] = defaultdict(set)
    for row in truth.faults_active:
        if row["episode_id"] in value_of:
            out[value_of[row["episode_id"]]].add(row["fault_id"])
    return {value: [labels[f] for f in sorted(ids)] for value, ids in out.items()}


def dose_points(truth: Truth, episodes: Sequence[dict[str, Any]]) -> list[DosePoint]:
    """Success by fault kind and level, by the true outcome; episodes without faults are healthy.

    For a calibration run, where each episode carries at most one fault. An
    episode with several faults is left out.
    """
    groups: dict[tuple[str | None, float | None], list[bool]] = defaultdict(list)
    for e in episodes:
        identity = truth.identities.get(e["episode_id"])
        if identity is None or len(identity.faults) > 1:
            continue
        success = truth.true_success.get(e["episode_id"], e["outcome"] == "success")
        if identity.faults:
            fault = identity.faults[0]
            groups[(fault.kind, float(fault.level))].append(success)
        else:
            groups[(None, None)].append(success)
    keys = sorted(groups, key=lambda k: ("" if k[0] is None else k[0], k[1] or 0.0))
    return [DosePoint(kind, level, Rate.of(groups[(kind, level)])) for kind, level in keys]
