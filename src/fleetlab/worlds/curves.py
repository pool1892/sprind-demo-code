"""Suite-specific calibration for the world generator.

The library's curves.json, including its supplement, is authoritative over matching
legacy points in worlds/curves_supplement.json. Joint success can be predicted from a
measured joint alone. Interaction qualification additionally requires the same-attempt
single counts and a paired interaction interval; unpaired legacy singles are ignored.

Ratios use healthy successes as denominator for difficulty estimation. Reported effect
and interaction percentage points instead use all paired attempts. Releases prefer
their own measured suite point, then a measured carried joint, before multiplying
independent members. Interpolation between measured levels is a modelling assumption,
not evidence of a slope below a fault's measured knee.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from importlib import resources
from typing import Any

from fleetlab.contracts.faults import FaultSpec

SPATIAL = "libero_spatial"
OBJECT = "libero_object"
SUPPLEMENT_FILE = "curves_supplement.json"
# Healthy success by the fleet's check where no healthy run of the suite is at hand:
# the experimental design (placed: 195/200 spatial, 196/200 object).
FALLBACK_HEALTHY = {SPATIAL: 195 / 200, OBJECT: 196 / 200}


@dataclass(frozen=True)
class CurvePoint:
    """One measured point on one suite."""

    kind: str
    label: str
    level: float
    suite: str
    n: int
    successes: float  # measured counts are integers; uncertainty draws may be fractional
    # the analysed outcome (logged for the data class, else placed)
    logged_successes: float
    healthy_successes: float
    role: str  # "effect", "decoy" or "small_cause"
    difference: tuple[float, float, float]  # faulted minus healthy, with its 95% interval
    run: str

    @property
    def ratio(self) -> float:
        return self.successes / self.healthy_successes if self.healthy_successes else 1.0

    @property
    def logged_ratio(self) -> float:
        return self.logged_successes / self.healthy_successes if self.healthy_successes else 1.0


@dataclass(frozen=True)
class JointPoint:
    """An interaction pair measured together: both faults in the same episodes."""

    members: tuple[tuple[str, str], tuple[str, str]]  # (kind, label) of each member
    suite: str
    n: int
    successes: float
    healthy_successes: float
    role: str
    difference: tuple[float, float, float]
    run: str
    singles: tuple[float, float] | None = None  # each member's ratio on the same episodes
    interaction: tuple[float, float, float] | None = None
    single_successes: tuple[int, int] | None = None

    @property
    def paired(self) -> bool:
        return self.single_successes is not None and self.interaction is not None

    @property
    def interaction_role(self) -> str:
        if not self.paired or self.interaction is None:
            return "unmeasured"
        if self.interaction[2] < 0:
            return "interaction"
        return "sub-additive" if self.interaction[1] > 0 else "additive"

    @property
    def ratio(self) -> float:
        return self.successes / self.healthy_successes if self.healthy_successes else 1.0


@dataclass(frozen=True)
class Ratio:
    """What a fault does to an episode's success on one suite, as multipliers."""

    true: float
    logged: float
    source: str  # the point(s) it came from, for the fault table
    transferred: bool  # a libero_spatial point applied to another suite
    interpolated: bool
    effect: bool  # every point it rests on is an effect beyond noise


def _source_run(raw: dict[str, Any]) -> str:
    """Prefer explicit provenance when several calibration stages are all named `run`."""
    paths = raw.get("run_dirs")
    return (
        " + ".join(str(path) for path in paths)
        if paths
        else str(raw.get("run_id", raw.get("run", "")))
    )


def _point(kind: str, raw: dict[str, Any], suite: str) -> CurvePoint:
    return CurvePoint(
        kind=kind,
        label=str(raw["label"]),
        level=float(raw["level"]),
        suite=suite,
        n=int(raw["n"]),
        successes=int(raw["successes"]),
        logged_successes=int(raw.get("logged_successes", raw["successes"])),
        healthy_successes=int(raw["healthy_successes"]),
        role=str(raw["role"]),
        difference=(
            float(raw["difference"][0]),
            float(raw["difference"][1]),
            float(raw["difference"][2]),
        ),
        run=_source_run(raw),
    )


def _supplement_text() -> str | None:
    path = resources.files("fleetlab.worlds").joinpath(SUPPLEMENT_FILE)
    return path.read_text() if path.is_file() else None


@dataclass(frozen=True)
class Curves:
    """Every measured point by (kind, suite, label), the joint points, the healthy rates."""

    points: dict[tuple[str, str, str], CurvePoint]
    joints: tuple[JointPoint, ...]
    healthy: dict[str, float]  # suite -> healthy success by the fleet's check
    sources: tuple[str, ...]
    healthy_counts: dict[str, tuple[int, int]] = field(default_factory=dict)

    tasks: dict[tuple[str, int], Curves] = field(default_factory=dict, compare=False)

    def for_task(self, suite: str, task_index: int | None) -> Curves:
        """No suite-average fallback for an unmeasured task."""
        if task_index is None or (suite, task_index) not in self.tasks:
            raise KeyError(f"no task calibration for {suite}/{task_index}")
        return self.tasks[suite, task_index]

    # -- lookup ---------------------------------------------------------------------------

    def suites_measured(self, kind: str) -> set[str]:
        return {s for (k, s, _) in self.points if k == kind}

    def point(self, kind: str, label: str, suite: str) -> tuple[CurvePoint, bool]:
        """The point on `suite`, or the `libero_spatial` one (transferred = True)."""
        found = self.points.get((kind, suite, label))
        if found is not None:
            return found, False
        return self.points[(kind, SPATIAL, label)], suite != SPATIAL

    def joint(self, a: tuple[str, str], b: tuple[str, str], suite: str) -> tuple[JointPoint, bool]:
        """The joint point of a pair on `suite`, or the `libero_spatial` one."""
        want = {a, b}
        for wanted_suite in (suite, SPATIAL):
            for joint in self.joints:
                if set(joint.members) == want and joint.suite == wanted_suite:
                    return joint, wanted_suite != suite
        raise KeyError(f"no joint point for {a} with {b}")

    def has_joint(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        return any(set(j.members) == {a, b} for j in self.joints)

    # -- ratios ---------------------------------------------------------------------------

    def ratio(self, fault: FaultSpec, suite: str) -> Ratio:
        """The multiplier of a fault at its level on a suite (a release: what it carries)."""
        if fault.kind == "software_release":
            return self._release(fault, suite)
        family = [
            p
            for p in family_points(fault)
            if any(
                (fault.kind, use_suite, p.label) in self.points for use_suite in (suite, SPATIAL)
            )
        ]
        if not family:
            raise KeyError(f"fault {fault.fault_id}: {fault.kind} has no calibrated family")
        exact = [p for p in family if p.level == fault.level]
        if exact:
            point, moved = self.point(fault.kind, exact[0].label, suite)
            true = self._true(point, fault)
            return Ratio(
                true=true,
                logged=point.logged_ratio if _is_data(fault) else true,
                source=f"'{point.label}' on {point.suite}",
                transferred=moved,
                interpolated=False,
                effect=point.role == "effect",
            )
        top = max(p.level for p in family)
        if fault.level > top:
            raise KeyError(
                f"fault {fault.fault_id}: {fault.kind} at {fault.level} is above the highest "
                f"calibrated level {top}"
            )
        below = [p for p in family if p.level < fault.level]
        above = min((p for p in family if p.level > fault.level), key=lambda p: p.level)
        hi, moved_hi = self.point(fault.kind, above.label, suite)
        if below:
            low_dose = max(below, key=lambda p: p.level)
            lo, moved_lo = self.point(fault.kind, low_dose.label, suite)
            x0, r0, l0 = lo.level, self._true(lo, fault), lo.logged_ratio
            source_points = [lo, hi]
        else:
            x0, r0, l0 = 0.0, 1.0, 1.0
            moved_lo = False
            source_points = [hi]
        w = (fault.level - x0) / (hi.level - x0)
        true = r0 + w * (self._true(hi, fault) - r0)
        logged = l0 + w * (hi.logged_ratio - l0)
        return Ratio(
            true=true,
            logged=logged if _is_data(fault) else true,
            source=" to ".join(f"'{p.label}'" for p in source_points) + " (interpolated)",
            transferred=moved_lo or moved_hi,
            interpolated=True,
            effect=all(p.role == "effect" for p in source_points),
        )

    def _true(self, point: CurvePoint, fault: FaultSpec) -> float:
        return 1.0 if _is_data(fault) else point.ratio

    def _release(self, fault: FaultSpec, suite: str) -> Ratio:
        import fleetlab.faults as faults
        from fleetlab.faults.software_release import ReleaseParams, carried_spec

        params = ReleaseParams.model_validate(fault.params)
        for dose in getattr(faults.get("software_release"), "calibration", ()):
            measured = self.points.get(("software_release", suite, dose.label))
            if (
                measured is not None
                and ReleaseParams.model_validate(dose.params).carries == params.carries
            ):
                return Ratio(
                    measured.ratio,
                    measured.logged_ratio,
                    f"measured release '{dose.label}' on {suite}",
                    False,
                    False,
                    measured.role == "effect",
                )
        inner_faults = [
            carried_spec(c, fault.target.scope, fault.schedule, fault.fault_id)
            for c in params.carries
        ]
        # Versions are deployment labels. Match mechanisms and calibrated levels,
        # including a release renumbered by the composer.
        if len(inner_faults) == 2:
            keys = [(f.kind, calibrated_label(f)) for f in inner_faults]
            if all(label is not None for _, label in keys):
                a, b = ((k, str(label)) for k, label in keys)
                if self.has_joint(a, b):
                    joint, moved = self.joint(a, b, suite)
                    return Ratio(
                        joint.ratio,
                        joint.ratio,
                        f"measured joint {joint.run}",
                        moved,
                        False,
                        joint.role == "effect",
                    )
        true = logged = 1.0
        sources: list[str] = []
        moved = interpolated = False
        effect = True
        for carried in params.carries:
            inner = carried_spec(carried, fault.target.scope, fault.schedule, fault.fault_id)
            part = self.ratio(inner, suite)
            true *= part.true
            logged *= part.logged
            sources.append(part.source)
            moved |= part.transferred
            interpolated |= part.interpolated
            effect &= part.effect
        return Ratio(
            true, logged, "release carrying " + ", ".join(sources), moved, interpolated, effect
        )


def _is_data(fault: FaultSpec) -> bool:
    import fleetlab.faults as faults
    from fleetlab.contracts.vocabulary import FaultClass

    return faults.get(fault.kind).fault_class is FaultClass.data


def family_points(fault: FaultSpec) -> list[Any]:
    """The fault type's calibration points of the same family as `fault` (same params,
    target and nominal), as `DosePoint`s."""
    import fleetlab.faults as faults

    thing = fault.target.thing
    out = []
    from fleetlab.worlds.calibration import extra_doses

    for p in (*getattr(faults.get(fault.kind), "calibration", ()), *extra_doses(fault.kind)):
        parameters = dict(fault.params)
        calibrated = dict(p.params)
        software = fault.kind == "image_pipeline_release"
        if software:
            parameters.pop("version", None)
            calibrated.pop("version", None)
        nominal_same = p.nominal == fault.nominal
        if fault.target.condition_key is not None:
            # Zero-valued scalar/short vector and the extension's padded vector
            # describe the same nominal scene (condition_change uses that rule).
            def zeros(value: Any) -> bool:
                values = value if isinstance(value, list) else [value]
                return bool(values) and all(v == 0 for v in values)

            nominal_same |= zeros(p.nominal) and zeros(fault.nominal)
        same = (
            calibrated == parameters
            and thing is not None
            and p.target == thing.value
            and (software or nominal_same)
        )
        if same:
            out.append(p)
    return out


def calibrated_label(fault: FaultSpec) -> str | None:
    """The label of the calibration point a fault equals, or None (between points)."""
    for p in family_points(fault):
        if p.level == fault.level:
            return str(p.label)
    return None


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def has_supplement() -> bool:
    return _supplement_text() is not None


@cache
def load() -> Curves:
    """The curves of both sources, read once per process."""
    from fleetlab.faults.records import measured

    base_text = resources.files("fleetlab.faults").joinpath("curves.json").read_text()
    base = json.loads(base_text)
    points: dict[tuple[str, str, str], CurvePoint] = {}
    for kind in base["types"]:
        for raw in measured(kind)["points"]:
            p = _point(kind, raw, SPATIAL)
            points[(kind, SPATIAL, p.label)] = p
    healthy = dict(FALLBACK_HEALTHY)
    healthy[SPATIAL] = base["healthy"]["successes"] / base["healthy"]["n"]
    healthy_n = {SPATIAL: int(base["healthy"]["n"])}
    healthy_counts = {SPATIAL: (int(base["healthy"]["successes"]), healthy_n[SPATIAL])}
    joints: dict[tuple[frozenset[tuple[str, str]], str], JointPoint] = {}
    sources = [f"fleetlab/faults/curves.json sha256 {_digest(base_text)}"]
    text = _supplement_text()
    supplements = []
    if text is not None:
        supplements.append(json.loads(text))
        legacy = json.loads(text)
        legacy.pop("task_refinement", None)
        legacy_text = json.dumps(legacy, indent=1) + "\n"
        sources.append(f"fleetlab/worlds/{SUPPLEMENT_FILE} sha256 {_digest(legacy_text)}")
    # The fault library's newer, paired measurements supersede legacy joints.
    supplements.append(base.get("supplement", {}))
    for sup in supplements:
        for suite, rate in sup.get("healthy", {}).items():
            if int(rate["n"]) > healthy_n.get(suite, 0):
                healthy[suite] = float(rate["successes"]) / float(rate["n"])
                healthy_n[suite] = int(rate["n"])
                healthy_counts[suite] = (int(rate["successes"]), int(rate["n"]))
        for raw in sup.get("points", []):
            p = _point(raw["kind"], raw, raw["suite"])
            points[(p.kind, p.suite, p.label)] = p
        for raw in sup.get("joints", []):
            members = tuple((m["kind"], m["label"]) for m in raw["members"])
            assert len(members) == 2
            counts = raw.get("single_successes")
            interaction = raw.get("interaction")
            paired = counts is not None and interaction is not None
            # Legacy `singles` alone are not evidence of same-attempt measurements.
            singles = (
                None if not paired else tuple(float(n) / raw["healthy_successes"] for n in counts)
            )
            joint = JointPoint(
                members=(members[0], members[1]),
                suite=raw["suite"],
                n=int(raw["n"]),
                successes=int(raw["successes"]),
                healthy_successes=int(raw["healthy_successes"]),
                role=str(raw["role"]),
                difference=(
                    float(raw["difference"][0]),
                    float(raw["difference"][1]),
                    float(raw["difference"][2]),
                ),
                run=_source_run(raw),
                singles=None if singles is None else (singles[0], singles[1]),
                single_successes=None if not paired else (int(counts[0]), int(counts[1])),
                interaction=None
                if not paired
                else (float(interaction[0]), float(interaction[1]), float(interaction[2])),
            )
            joints[(frozenset(members), joint.suite)] = joint
    out = Curves(points, tuple(joints.values()), healthy, tuple(sources), healthy_counts)
    if text is not None:
        _load_tasks(out, json.loads(text).get("task_calibration", {}))
    return out


def _load_tasks(curves: Curves, record: dict[str, Any]) -> None:
    """Materialized task counts from the exact published cohorts, never a local cache."""
    for suite, baseline in record.get("healthy", {}).items():
        for cell in baseline["tasks"]:
            task = int(cell["task_index"])
            count, n = int(cell["successes"]), int(cell["n"])
            curves.tasks[suite, task] = Curves(
                {}, (), {suite: count / n}, curves.sources, {suite: (count, n)}
            )
    for row in record.get("points", []):
        suite = row["suite"]
        members = tuple((str(k), str(label)) for k, label in row["members"])
        for cell in row["tasks"]:
            target = curves.tasks[suite, int(cell["task_index"])]
            if cell["n"] <= 0 or cell["healthy_successes"] <= 0:
                continue
            fields = {k: cell[k] for k in ("n", "successes", "healthy_successes")}
            if len(members) == 1:
                key = members[0][0], suite, members[0][1]
                point = curves.points[key]
                target.points[key] = replace(
                    point, **fields, logged_successes=cell["logged_successes"]
                )
            else:
                joint, moved = curves.joint(members[0], members[1], suite)
                if moved:
                    continue
                singles = cell.get("single_successes")
                interaction = cell.get("interaction")
                # Member order in the cells is canonical; do not swap A and B counts.
                updated = replace(
                    joint,
                    **fields,
                    members=(members[0], members[1]),
                    singles=None
                    if singles is None
                    else tuple(v / cell["healthy_successes"] for v in singles),
                    single_successes=None
                    if singles is None
                    else (int(singles[0]), int(singles[1])),
                    interaction=None if interaction is None else tuple(interaction),
                )
                object.__setattr__(target, "joints", (*target.joints, updated))


# A context cell is an absolute outcome probability for a task with *all* the
# listed mechanisms active. It is not a ratio and has no independence fallback.
CellKey = tuple[str, int, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class MeasuredCell:
    n: int
    successes: int
    logged_successes: int
    sources: tuple[str, ...]

    def rate(self, logged: bool = False) -> float:
        return (self.logged_successes if logged else self.successes) / self.n

    @property
    def sufficient(self) -> bool:
        return self.n >= 10 and all(
            not 0.15 <= self.rate(logged) <= 0.85 or self.n >= 50 for logged in (False, True)
        )


def refinement(jobs: tuple[str, ...] | None = None) -> dict[str, Any]:
    """The refinement record: whole, or as it stands with only the named calibration jobs.

    A cohort belongs to a job when its run directory lies under the calibration root of
    that name; the cohorts of the two calibration worlds belong to none and always count.
    A sealed recipe names its jobs (`WorldRecipe.calibration_jobs`), so that a job imported
    after its seal leaves its cells, its comparisons and its pinned digest as they were.
    """
    raw: dict[str, Any] = json.loads(_supplement_text() or "{}").get("task_refinement", {})
    if jobs is None:
        return raw
    roots = [str(x) for x in raw.get("calibration_roots", [])]
    if "calibration_root" in raw:
        roots.append(str(raw["calibration_root"]))
    names = {root.rstrip("/").rsplit("/", 1)[-1]: root.rstrip("/") for root in roots}
    if not set(jobs) <= set(names) or len(set(jobs)) != len(jobs):
        raise ValueError(f"calibration jobs {sorted(set(jobs) - set(names))} are not imported")
    left_out = [root for name, root in names.items() if name not in jobs]

    def counted(path: str) -> bool:
        return not any(path == root or path.startswith(root + "/") for root in left_out)

    cells = []
    for row in raw.get("cells", []):
        sources = {path: part for path, part in row["sources"].items() if counted(path)}
        if sources:
            totals = {
                field: sum(part[field] for part in sources.values())
                for field in ("n", "successes", "logged_successes")
            }
            cells.append({**row, **totals, "sources": sources})
    return {
        "calibration_jobs": list(jobs),
        "cells": cells,
        "decoy_assays": [
            assay
            for assay in raw.get("decoy_assays", [])
            if all(counted(path) for path in assay["sources"])
        ],
    }


@cache
def measured_cells() -> dict[CellKey, MeasuredCell]:
    """Versioned task evidence; historical suite/task recipes retain their own cells."""
    return _cells(None)


@cache
def measured_cells_of(jobs: tuple[str, ...]) -> dict[CellKey, MeasuredCell]:
    """The cells as they stand with only the named calibration jobs (`refinement`)."""
    return _cells(jobs)


def _cells(jobs: tuple[str, ...] | None) -> dict[CellKey, MeasuredCell]:
    table = baseline_cells()
    for row in refinement(jobs).get("cells", []):
        key = row["suite"], int(row["task_index"]), tuple(sorted(tuple(m) for m in row["members"]))
        old = table.get(key, MeasuredCell(0, 0, 0, ()))
        # Refinement cohorts are disjoint from the baseline and from one another;
        # the materializer verifies physical identities before publishing counts.
        table[key] = MeasuredCell(
            old.n + row["n"],
            old.successes + row["successes"],
            old.logged_successes + row["logged_successes"],
            tuple(dict.fromkeys((*old.sources, *row["sources"]))),
        )
    return table


def baseline_cells() -> dict[CellKey, MeasuredCell]:
    """The cells of the task calibration alone, before any refinement cohort."""
    text = _supplement_text()
    if text is None:
        raise ValueError("task calibration is absent")
    raw = json.loads(text)
    table: dict[CellKey, MeasuredCell] = {}
    baseline = raw["task_calibration"]
    for suite, point in baseline["healthy"].items():
        for cell in point["tasks"]:
            table[suite, int(cell["task_index"]), ()] = MeasuredCell(
                cell["n"], cell["successes"], cell["successes"], tuple(point["run_dirs"])
            )
    for point in baseline["points"]:
        members = tuple(sorted(tuple(m) for m in point["members"]))
        for cell in point["tasks"]:
            key = point["suite"], int(cell["task_index"]), members
            import fleetlab.faults as faults
            from fleetlab.contracts.vocabulary import FaultClass

            true = cell["successes"]
            if len(members) == 1 and faults.get(members[0][0]).fault_class is FaultClass.data:
                true = cell["healthy_successes"]
            table[key] = MeasuredCell(
                cell["n"], true, cell["logged_successes"], tuple(point["run_dirs"])
            )
    return table


def measured_sources(jobs: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """The additional record is pinned separately; historical seals cannot absorb it.

    With `jobs`, the digest is that of the record as of those calibration jobs.
    """
    raw = refinement(jobs)
    return (*load().sources, f"task_refinement sha256 {_digest(json.dumps(raw, sort_keys=True))}")


def context_weights(
    faults: Sequence[FaultSpec],
    suite: str,
    task: int | None,
    cells: dict[CellKey, MeasuredCell],
    *,
    require_precision: bool = True,
) -> dict[CellKey, float]:
    """Interpolate a measured context grid, refusing every missing corner/subset.

    Linear interpolation is the recorded dose-response model. Multiplication is
    used only for interpolation weights, never to combine probabilities of faults.
    """
    import itertools

    from fleetlab.faults.software_release import ReleaseParams, carried_spec

    if task is None:
        raise KeyError("scheduled task is required for measured contexts")
    expanded: list[FaultSpec] = []
    for f in faults:
        if f.kind == "software_release":
            expanded.extend(
                carried_spec(c, f.target.scope, f.schedule, f.fault_id)
                for c in ReleaseParams.model_validate(f.params).carries
            )
        elif f.kind != "calibration_update":
            expanded.append(f)
    if any(calibrated_label(f) is None and f.level != 0 for f in expanded):
        from fleetlab.worlds.calibration import mechanism_members

        direct = suite, task, mechanism_members(expanded)
        measured = cells.get(direct)
        if measured is not None and (not require_precision or measured.sufficient):
            return {direct: 1.0}
    choices: list[list[tuple[tuple[str, str] | None, float]]] = []
    for f in expanded:
        if f.level == 0:
            continue
        exact = calibrated_label(f)
        if exact is not None:
            choices.append([((f.kind, exact), 1.0)])
            continue
        family = sorted(family_points(f), key=lambda p: p.level)
        others = [g for g in expanded if g is not f and g.level != 0]
        if all(calibrated_label(g) is not None for g in others):
            fixed = [(g.kind, str(calibrated_label(g))) for g in others]
            # A continuously drifting dose uses knots measured in THIS context.
            # A solo quarter-wear knot does not imply a quarter-wear+blur knot.
            # Keep thin measured knots here so precision cannot be evaded by
            # choosing a coarser interval around a known but uncertain point.
            family = [
                p
                for p in family
                if (suite, task, tuple(sorted([*fixed, (f.kind, str(p.label))]))) in cells
            ]
            if not family:
                raise KeyError(f"unmeasured task context: {suite}, task {task}, {fixed}, {f.kind}")
        levels = [(0.0, None), *((float(p.level), (f.kind, str(p.label))) for p in family)]
        lower = next((x for x in reversed(levels) if x[0] <= f.level), None)
        upper = next((x for x in levels if x[0] >= f.level), None)
        if lower is None or upper is None or lower[0] == upper[0]:
            raise KeyError(f"dose outside measured grid: {f.kind} {f.level}")
        share = (f.level - lower[0]) / (upper[0] - lower[0])
        choices.append([(lower[1], 1 - share), (upper[1], share)])
    weights: dict[CellKey, float] = {}
    for corner in itertools.product(*choices):
        key = suite, task, tuple(sorted(member for member, _ in corner if member is not None))
        weight = math.prod(w for _, w in corner)
        if weight <= 0:
            continue
        cell = cells.get(key)
        if cell is None:
            raise KeyError(f"unmeasured task context: {key}")
        if require_precision and not cell.sufficient:
            raise KeyError(f"thin task context ({cell.n} attempts): {key}")
        weights[key] = weights.get(key, 0) + weight
    return weights
