"""Realised cause membership and its scoring-first, observed-window replay plan.

The curves are a comparison list, never evidence of a realised removal gain.
This module does no simulation. Every point uses the full window denominator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import OracleTable, ReplayNeed
from fleetlab.contracts.rollout import JobMode, RunRecord
from fleetlab.contracts.vocabulary import FaultClass
from fleetlab.oracle.world import OracleWorld
from fleetlab.worlds.truth import DETECT_EPISODES, DETECT_POINTS, FaultTruth

if TYPE_CHECKING:
    from fleetlab.oracle.table import TableBuild

CAUSES_FILE = "measured_causes.json"
CLASS_RULE = "realised-removal-either-window-v1"


def logged_outcomes(build: TableBuild) -> dict[str, tuple[bool, bool]]:
    """True and visible outcomes of the actual run, never of scratch run jobs."""
    from fleetlab.log.runfiles import read_table

    result: dict[str, tuple[bool, bool]] = {}
    for source in build.runs:
        originals = [e for e in source.executions if e.job.mode is JobMode.run]
        if not originals:
            continue
        record = RunRecord.model_validate_json((source.run_dir / "run.json").read_text())
        details: Any = record.extra.get("jobs_detail", [])
        shown = {
            str(d["key"]): bool(d["visible_success"])
            for d in details
            if isinstance(d, dict) and "visible_success" in d
        }
        if any(e.job.key not in shown for e in originals):
            # Early runs may not retain the summary field.
            shown.update(
                {
                    str(r["episode_id"]): r["outcome"] == "success"
                    for r in read_table(Path(source.run_dir), "episodes").to_pylist()
                }
            )
        for execution in originals:
            if execution.job.key in shown:
                result.setdefault(
                    execution.job.identity.episode_id, (execution.success, shown[execution.job.key])
                )
    return result


class Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["measured", "missing", "not_needed", "not_applicable"]
    episodes: int
    carrying: int
    known: int = 0
    missing: int = 0
    rescued: int | None = None
    broken: int | None = None
    net: int | None = None
    points: float | None = None
    metric: str = "net_successes"
    reason: str | None = None


class MeasuredItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str
    fault_ids: list[str]
    pair_kind: str | None = None
    planted_role: str
    curve_class: str
    measured_class: Literal["cause", "small", "decoy", "inert"] | None = None
    data: bool = False
    identifiable: bool
    observed_carriers: int
    qualified_by: list[str] = Field(default_factory=list)
    windows: dict[str, Measurement]


class MeasuredCauses(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    world_id: str
    world_hash: str | None
    rule: str = CLASS_RULE
    threshold_points: float = DETECT_POINTS
    min_observed_carriers: int = DETECT_EPISODES
    episodes_per_window: dict[str, int]
    faults: list[MeasuredItem]
    pairs: list[MeasuredItem]
    causes: list[dict[str, object]]
    cause_count: int
    curve_causes: list[str]
    no_decoy_left: bool
    complete: bool
    problems: list[str]


def _carriers(identities: Sequence[EpisodeIdentity], ids: Sequence[str]) -> int:
    return sum(bool(set(ids) & i.fault_ids) for i in identities)


def _unmeasured(
    identities: Sequence[EpisodeIdentity],
    ids: Sequence[str],
    reason: str,
    status: Literal["not_needed", "not_applicable"] = "not_needed",
) -> Measurement:
    return Measurement(
        status=status, episodes=len(identities), carrying=_carriers(identities, ids), reason=reason
    )


def removal_measurement(
    table: OracleTable, identities: Sequence[EpisodeIdentity], ids: Sequence[str]
) -> Measurement:
    rescued = broken = known = missing = 0
    gone = set(ids)
    for identity in identities:
        if not gone & identity.fault_ids:
            continue
        before = table.success(identity.episode_id, identity.faults)
        after = table.success(
            identity.episode_id, [f for f in identity.faults if f.fault_id not in gone]
        )
        if before is None or after is None:
            missing += 1
        else:
            known += 1
            rescued += int(after and not before)
            broken += int(before and not after)
    net = rescued - broken
    return Measurement(
        status="missing" if missing else "measured",
        episodes=len(identities),
        carrying=known + missing,
        known=known,
        missing=missing,
        rescued=rescued,
        broken=broken,
        net=None if missing else net,
        points=None if missing else 100 * net / len(identities) if identities else 0.0,
        reason="removal or baseline outcomes missing" if missing else None,
    )


def _data_measurement(
    identities: Sequence[EpisodeIdentity],
    fault: FaultTruth,
    records: Mapping[str, tuple[bool, bool]],
    data_ids: set[str],
) -> Measurement:
    rescued = broken = known = missing = 0
    ambiguous = False
    for identity in identities:
        if fault.fault_id not in identity.fault_ids:
            continue
        recorded = records.get(identity.episode_id)
        # Today's records retain true/logged outcomes, not each data hook's verdict.
        if recorded is None or len(identity.fault_ids & data_ids) != 1:
            ambiguous |= len(identity.fault_ids & data_ids) > 1
            missing += 1
            continue
        true, logged = recorded
        known += 1
        rescued += int(true and not logged)
        broken += int(logged and not true)
    return Measurement(
        status="missing" if missing else "measured",
        episodes=len(identities),
        carrying=known + missing,
        known=known,
        missing=missing,
        rescued=rescued,
        broken=broken,
        net=None if missing else rescued - broken,
        points=None
        if missing
        else 100 * (rescued + broken) / len(identities)
        if identities
        else 0.0,
        metric="misreported_outcomes",
        reason="coactive data hooks cannot be attributed from today's records"
        if ambiguous
        else "true/logged as-run outcomes missing"
        if missing
        else None,
    )


def _qualify(item: MeasuredItem) -> None:
    if item.identifiable and item.observed_carriers >= DETECT_EPISODES:
        item.qualified_by = [
            name
            for name, m in item.windows.items()
            if m.points is not None
            and m.points >= DETECT_POINTS
            and (item.planted_role not in ("decoy", "inert") or name == "scoring")
        ]
    if item.qualified_by:
        item.measured_class = "cause"
    elif any(m.status == "missing" for m in item.windows.values()):
        item.measured_class = None
    else:
        item.measured_class = (
            "decoy"
            if item.planted_role == "decoy"
            and item.windows["scoring"].points is not None
            and item.windows["scoring"].points < DETECT_POINTS
            else "inert"
            if item.planted_role == "inert"
            else "small"
        )


def measure_causes(
    world: OracleWorld,
    table: OracleTable,
    logged_outcomes: Mapping[str, tuple[bool, bool]] | None = None,
) -> MeasuredCauses:
    """Measure standalone faults first, then eligible interaction pairs.

    The table must include observed identities as well as scoring identities.
    Unmeasured observed rows are explicit; they never become zeroes.
    """
    scoring, observed = world.scoring(), world.window("observed")
    faults: list[MeasuredItem] = []
    problems: list[str] = []
    truths = [] if world.fault_table is None else world.fault_table.faults
    data_ids = {f.fault_id for f in truths if f.fault_class is FaultClass.data}
    for fault in truths:
        data = fault.fault_id in data_ids
        ids = [fault.fault_id]
        m = (
            _data_measurement(scoring, fault, logged_outcomes or {}, data_ids)
            if data
            else removal_measurement(table, scoring, ids)
        )
        if data:
            o = _data_measurement(observed, fault, logged_outcomes or {}, data_ids)
        elif fault.role == "effective" and m.points is not None and m.points < DETECT_POINTS:
            o = removal_measurement(table, observed, ids)
        else:
            o = _unmeasured(
                observed,
                ids,
                "scoring pending"
                if m.points is None
                else "scoring qualifies or planted decoy/inert",
            )
        item = MeasuredItem(
            item=fault.fault_id,
            fault_ids=ids,
            planted_role=fault.role,
            curve_class="cause"
            if fault.detectable
            else "small"
            if fault.role == "effective"
            else fault.role,
            data=data,
            identifiable=fault.identifiable,
            observed_carriers=fault.episodes.get("observed", 0),
            windows={"scoring": m, "observed": o},
        )
        _qualify(item)
        faults.append(item)
    if world.fault_table is None:
        problems.append("fault_table.json required for measured cause eligibility")
    by_id = {f.item: f for f in faults}
    curve_pairs = {c.unit: c.detectable for c in world.causes if len(c.fault_ids) > 1}
    pairs: list[MeasuredItem] = []
    marked = {tuple(sorted(p)) for p in world.spec.patterns.interactions}
    additive = {tuple(sorted(p)) for p in world.additive_pairs}
    additive.update(tuple(sorted(p)) for p in getattr(world.spec.patterns, "additive_pairs", ()))
    for ids_tuple in sorted(marked | additive):
        ids = list(ids_tuple)
        members = [by_id.get(f) for f in ids]
        interaction = ids_tuple in marked
        pending = any(m is None or m.measured_class is None for m in members)
        single_cause = any(m is not None and m.measured_class == "cause" for m in members)
        reason = (
            "additive pairs have no joint cause treatment"
            if not interaction
            else "standalone member qualifies"
            if single_cause
            else "standalone measurements pending"
        )
        s = _unmeasured(scoring, ids, reason, "not_applicable")
        o = _unmeasured(observed, ids, reason, "not_applicable")
        if interaction and not single_cause and not pending:
            s = removal_measurement(table, scoring, ids)
            o = (
                removal_measurement(table, observed, ids)
                if s.points is not None and s.points < DETECT_POINTS
                else _unmeasured(observed, ids, "scoring pair qualifies or pending")
            )
        item = MeasuredItem(
            item="+".join(ids),
            fault_ids=ids,
            pair_kind="interaction" if interaction else "additive_pair",
            planted_role="pair",
            curve_class="cause" if curve_pairs.get("+".join(ids), False) else "small",
            identifiable=all(m is not None and m.identifiable for m in members),
            observed_carriers=sum(set(ids) <= i.fault_ids for i in observed),
            windows={"scoring": s, "observed": o},
        )
        _qualify(item)
        pairs.append(item)
    for item in [*faults, *pairs]:
        for name, measurement in item.windows.items():
            if measurement.status == "missing":
                problems.append(f"{item.item} {name}: {measurement.reason} ({measurement.missing})")
    cause_list: list[dict[str, object]] = [
        {"unit": f.item, "fault_ids": f.fault_ids, "qualified_by": f.qualified_by}
        for f in [*faults, *pairs]
        if f.measured_class == "cause"
    ]
    curve_list = (
        [c.unit for c in world.causes if c.detectable]
        if world.causes
        else [f.fault_id for f in truths if f.detectable]
    )
    return MeasuredCauses(
        world_id=world.world_id,
        world_hash=world.world_hash,
        episodes_per_window={"scoring": len(scoring), "observed": len(observed)},
        faults=faults,
        pairs=pairs,
        causes=cause_list,
        cause_count=len(cause_list),
        curve_causes=curve_list,
        no_decoy_left=not any(f.measured_class == "decoy" for f in faults),
        complete=not problems,
        problems=problems,
    )


def observed_requests(
    world: OracleWorld,
    measured: MeasuredCauses,
    table: OracleTable,
) -> list[ReplayNeed]:
    """The next stage only: singles first; joint rows once singles are decided."""
    from fleetlab.oracle.run import RunRefused

    scoring_missing = [f.item for f in measured.faults if f.windows["scoring"].points is None]
    if scoring_missing or world.fault_table is None:
        raise RunRefused(
            f"scoring measurements required before observed planning: {scoring_missing}"
        )
    requests: list[ReplayNeed] = []
    for item in [*measured.faults, *measured.pairs]:
        observed = item.windows["observed"]
        if item.data or observed.status != "missing":
            continue
        gone = set(item.fault_ids)
        for identity in world.window("observed"):
            if not gone & identity.fault_ids:
                continue
            remaining = tuple(f for f in identity.faults if f.fault_id not in gone)
            if table.success(identity.episode_id, remaining) is None:
                requests.append(ReplayNeed(identity.episode_id, remaining))
    return list({(r.episode_id, tuple(f.fault_id for f in r.faults)): r for r in requests}.values())


def required_observed_episodes(world: OracleWorld, measured: MeasuredCauses) -> list[str]:
    """Coverage includes reused removal rows, not only newly requested ones."""
    ids = {
        f
        for item in [*measured.faults, *measured.pairs]
        if not item.data and item.windows["observed"].status in ("measured", "missing")
        for f in item.fault_ids
    }
    return sorted(i.episode_id for i in world.window("observed") if i.fault_ids & ids)


def curve_observed_estimate(world: OracleWorld) -> dict[str, object]:
    """An explicitly provisional count from curves, before scoring replays exist."""
    candidates = (
        []
        if world.fault_table is None
        else [
            f
            for f in world.fault_table.faults
            if f.role == "effective"
            and f.fault_class is not FaultClass.data
            and f.expected_loss_points.get("scoring", 0) < DETECT_POINTS
        ]
    )
    ids = {f.fault_id for f in candidates}
    return {
        "basis": "curve estimates only; measured scoring gains decide the actual plan",
        "faults": sorted(ids),
        "removal_replays": sum(f.episodes.get("observed", 0) for f in candidates),
        "attempts_to_execute_twice": sum(bool(i.fault_ids & ids) for i in world.window("observed")),
        "joint_replays": "unknown until standalone gains in both windows exist",
    }
