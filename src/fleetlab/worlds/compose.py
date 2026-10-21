"""Seeded composition from measured curves, with paired pattern qualification.

Each attempt draws the configuration interaction, a separately labelled additive
wear/mass pair, a drift, a confounder and suite-aware decoys. Plain faults are drawn
from calibrated levels; stage-3 extensions enter the menu only when measured on both
suites. Moving plain levels and onsets settles observed success into the recipe's aim.

Acceptance requires enough detectable causes, measured paired interaction evidence on
the scoring task mix, additive support for the marked additive pair, a detectable drift
and confounder, the configuration/software share, the scoring floor and the four-fault
replay cap. Occupancy keeps mechanisms from conflicting; overlapping onsets after day
zero must be designed pairs. The first accepted attempt is the world. No simulator is
called during composition, and these predictions do not replace oracle confirmation.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, cast

import numpy as np
from pydantic import BaseModel

from fleetlab.contracts.faults import Drift, FaultSpec, Schedule
from fleetlab.contracts.ids import FLEET_SCOPE, IdKind, is_neutral, neutral_id, scope_covers
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey, FaultClass
from fleetlab.contracts.world import DayRange, Patterns, WorldSpec
from fleetlab.worlds import base, curves, estimate
from fleetlab.worlds.recipe import WorldRecipe

AIM_STEPS = 40
MAX_EFFECTIVE = 4  # the oracle's cap on effective faults per episode (contracts section 9)
MIN_HEALTHY_ROBOTS = 3
ROBOT_LAST_ONSET = 4  # a robot fault from day 4 still has 28 observed episodes (>= 20)
SITE_LAST_ONSET = 5
RELEASE_BASE = "4.3.0"

# The calibration labels the composer uses, per kind. Effect levels are the curves'
# points with an effect beyond noise; the decoys and the inert change are the confirmed
# decoy levels and a flat change.
# The families of levels a plain fault may take; `effect_levels` keeps the levels whose
# point is an effect beyond noise on at least one suite (a level flat on libero_spatial
# may act on libero_object, and the reverse), weakest first.
LEVEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "camera_pose_offset": ("3 deg scene pan", "6 deg scene pan", "9 deg scene pan"),
    "camera_degradation": ("1 px blur", "2 px blur", "3 px blur", "4 px blur"),
    "action_latency": ("8 steps", "14 steps", "17 steps", "20 steps"),
    "control_mode_change": ("relative",),
    "gripper_wear": ("0.5 wear", "0.75 wear", "1 wear"),
    "object_batch_variant": ("2 mass doublings", "4 mass doublings", "6 mass doublings"),
    "furniture_shift": ("1 cm sideways", "2 cm sideways", "4 cm sideways"),
    "success_detector_error": ("0.2 missed", "0.4 missed"),
}


@functools.cache
def covers(scope: str, robot_id: str, site_id: str) -> bool:
    """`scope_covers`, remembered: the search asks it a million times for a few answers."""
    return scope_covers(scope, robot_id, site_id)


def faults_in_force(
    schedule: Sequence[FaultSpec], robot_id: str, site_id: str, day: int
) -> list[FaultSpec]:
    """`base.in_force` for the measured search (the same faults at the same levels)."""
    out = []
    for fault in schedule:
        if not (fault.schedule.active_on(day) and covers(fault.target.scope, robot_id, site_id)):
            continue
        if fault.schedule.drift is None:
            if fault.level > 0.0:
                out.append(fault)
            continue
        level = base.day_level(fault, day)
        if level <= 0.0:
            continue
        out.append(fault if level == fault.level else fault.model_copy(update={"level": level}))
    return out


def effect_levels(the_curves: curves.Curves) -> dict[str, tuple[str, ...]]:
    """Per kind, the levels of its family that are an effect on some suite."""
    out: dict[str, tuple[str, ...]] = {}
    for kind, labels in LEVEL_FAMILIES.items():
        kept = tuple(
            label
            for label in labels
            if any(
                p.role == "effect"
                for (k, _, lb), p in the_curves.points.items()
                if k == kind and lb == label
            )
        )
        if kept:
            out[kind] = kept
    return out


DECOYS = (("illumination_change", "2 stops dimmer"), ("backdrop_change", "1 walls and floor"))
INERT = ("policy_knob", "5 sampling steps")
# Releases carry one effect point of a configuration type (a release reproduces the
# carried fault's episodes bit for bit, docs/design/faults.md section 6).
RELEASE_CARRIES: tuple[tuple[str, str], ...] = (
    ("action_latency", "14 steps"),
    ("action_latency", "17 steps"),
    ("action_latency", "20 steps"),
)
# Small causes: levels the curves found flat but did not confirm flat, measured on both
# suites (on libero_object the image and timing faults cost far more than on
# libero_spatial, so a level measured on one suite only may be no small cause at all).
SMALL_CAUSES: tuple[tuple[str, str, str], ...] = (  # kind, label, scope kind
    ("gripper_wear", "0.25 wear", "robot"),
    ("illumination_change", "1 stops dimmer", "site"),
)
FLEET_SMALL_RELEASE = ("policy_knob", "10 actions per chunk")
# Both singles and the joint must share their calibration attempts.
INTERACTION_PAIRS = ((("action_latency", "14 steps"), ("policy_knob", "10 actions per chunk")),)
ADDITIVE_PAIR = (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings"))
INERT_CHOICES = (INERT, ("calibration_update", "scene calibration rewritten"))
EXTENSION_KINDS = (
    "lens_contamination",
    "camera_latency",
    "gripper_force_loss",
    "gripper_sticky_release",
    "arm_torque_loss",
    "controller_overload",
    "image_pipeline_release",
    "table_height_change",
)
# A joint ratio assumed for a pair not yet measured, so that a world can be composed on
# the CPU before the supplement's run; a world composed on it says so and is composed
# again once the pair is measured.
ASSUMED_JOINT_RATIO = 0.70

# Faults that end every episode of a robot in failure: trivially detectable, so a world at
# the planned difficulty has few of them (docs/design/faults.md section 6: they suit D1).
# A single image view (policy_knob, 13 of 100) would be one too, but X-VLA then makes 4
# images per call at batch size 4, which the machine's pinned convolution database does
# not hold (2, 8 and 16), and the engine refuses such a job (replay rule 7); it is left out.
TOTAL_FAILURE = frozenset({"control_mode_change"})
# Plain faults that cannot cost two points of the fleet alone at robot scope (a twelfth
# of the fleet): D1 leaves them out.
WEAK_ALONE = frozenset(
    {
        ("gripper_wear", "robot"),
        ("camera_degradation", "robot"),
        ("success_detector_error", "robot"),
    }
)

# Site-wide faults that cost a third of the fleet most of its libero_object episodes (the
# delay and the table shift cost 93 and 68 points there): at most one per world, or
# the band leaves the other faults too small to detect.
HEAVY_AT_SITE = frozenset({"action_latency", "furniture_shift", "control_mode_change"})
MAX_HEAVY_AT_SITE = 1

# Plain fault menu: (kind, scope kind).
PLAIN_MENU: tuple[tuple[str, str], ...] = (
    ("camera_pose_offset", "robot"),
    ("camera_degradation", "robot"),
    ("gripper_wear", "robot"),
    ("success_detector_error", "robot"),
    ("action_latency", "robot"),
    ("control_mode_change", "robot"),
    ("software_release", "robot"),
    ("action_latency", "site"),
    ("software_release", "site"),
    ("object_batch_variant", "site"),
    ("furniture_shift", "site"),
)
CONFIG_CLASSES = frozenset({FaultClass.configuration, FaultClass.software})


def fault_class(kind: str) -> FaultClass:
    import fleetlab.faults as faults

    return faults.get(kind).fault_class


def dose_point(kind: str, label: str) -> Any:
    import fleetlab.faults as faults
    from fleetlab.worlds.calibration import extra_doses

    for p in (*getattr(faults.get(kind), "calibration", ()), *extra_doses(kind)):
        if p.label == label:
            return p
    raise KeyError(f"{kind} has no calibration point {label!r}")


def make_fault(
    fault_id: str,
    kind: str,
    label: str,
    scope: str,
    onset: int,
    *,
    nominal_version: str | None = None,
    version: str | None = None,
    drift: Drift | None = None,
    level: float | None = None,
) -> FaultSpec:
    """A fault at a calibration point (or, for a release, carrying one) at a scope."""
    import fleetlab.faults as faults
    from fleetlab.faults._common import target_of

    if kind == "software_release":
        if "|" in label:
            carries = []
            for member in label.split(" + "):
                carried_kind, carried_label = member.split("|")
                p = dose_point(carried_kind, carried_label)
                carries.append(
                    {
                        "kind": carried_kind,
                        "level": p.level,
                        "params": dict(p.params),
                        "target": p.target,
                        "nominal": p.nominal,
                    }
                )
        else:
            carries = dose_point(kind, label).params["carries"]
        params: dict[str, Any] = {"version": version, "carries": carries}
        truth = faults.get(kind).truth
        spec = FaultSpec(
            fault_id=fault_id,
            kind=kind,
            params=params,
            level=1.0,
            target=target_of(scope, ConfigKey.software_version.value, truth.target_kind),
            schedule=Schedule(onset_day=onset),
            nominal=nominal_version,
        )
    else:
        p = dose_point(kind, label)
        truth = faults.get(kind).truth
        params = dict(p.params)
        nominal = p.nominal
        if version is not None and "version" in params:
            # A release of its own kind (image_pipeline_release) under the fleet's version
            # numbers; its calibration point keeps its label (`curves.family_points`).
            params["version"] = version
            nominal = nominal_version if nominal_version is not None else nominal
        spec = FaultSpec(
            fault_id=fault_id,
            kind=kind,
            params=params,
            level=float(p.level if level is None else level),
            target=target_of(scope, p.target, truth.target_kind),
            schedule=Schedule(onset_day=onset, drift=drift),
            nominal=nominal,
        )
    faults.REGISTRY.check(spec)
    return spec


@dataclass
class Plain:
    """A plain fault while the composer adjusts it: its kind, scope, level index, onset."""

    kind: str
    scope: str
    labels: tuple[str, ...]  # the levels it may take, weakest first
    level: int
    onset: int
    last_onset: int
    fault_id: str = ""
    version: str | None = None  # a release's version
    first_onset: int = 0


@dataclass
class Draft:
    """One attempt's faults before the levels are settled."""

    fixed: list[FaultSpec] = field(default_factory=list)  # patterns and small causes
    plains: list[Plain] = field(default_factory=list)
    patterns: Patterns = field(default_factory=Patterns)
    small: set[str] = field(default_factory=set)
    roles: dict[str, str] = field(default_factory=dict)  # fault id -> pattern or "plain"
    assumed_joint: bool = False
    stopped: str = ""  # where the measured search gave an attempt up, for the refusal count


class Occupancy:
    """What each fault takes, so that no two faults act on the same thing for one robot."""

    def __init__(self, robots: Sequence[tuple[str, str]]) -> None:
        self.robots = list(robots)
        self.taken: list[tuple[str, str]] = []  # (scope, thing)

    def _covered(self, scope: str) -> set[str]:
        return {r for r, s in self.robots if covers(scope, r, s)}

    def free(self, scope: str, things: Sequence[str]) -> bool:
        mine = self._covered(scope)
        for other_scope, thing in self.taken:
            if thing in things and mine & self._covered(other_scope):
                return False
        return True

    def take(self, scope: str, things: Sequence[str]) -> None:
        self.taken.extend((scope, t) for t in things)


def things_of(kind: str, label: str) -> list[str]:
    """What a fault acts on, for the occupancy: its component, key or condition (a
    release its version and what it carries)."""
    if kind == "software_release":
        if "|" in label:
            return [
                ConfigKey.software_version.value,
                *(
                    thing
                    for member in label.split(" + ")
                    for thing in things_of(*member.split("|"))
                ),
            ]
        return [
            ConfigKey.software_version.value,
            *(str(c["target"]) for c in dose_point(kind, label).params["carries"]),
        ]
    p = dose_point(kind, label)
    target = str(p.target)
    return [
        {
            ConfigKey.camera_scene_calibration.value: "scene_camera",
            ConfigKey.camera_wrist_calibration.value: "wrist_camera",
        }.get(target, target)
    ]


def calibration_value(
    seed: int, scope: str, day: int, key: str, prior: Sequence[float]
) -> list[float]:
    """One distribution for benign and planted inert calibration records."""
    rng = base.rng(seed, "calibration-value", scope, day, key)
    return [
        round(float(v + rng.normal(0.0, 0.002 if i < 3 else 0.3)), 4) for i, v in enumerate(prior)
    ]


@dataclass(frozen=True)
class Composition:
    """A composed world: the spec, and what the composer knows about its faults."""

    recipe: WorldRecipe
    spec: WorldSpec
    attempt: int
    roles: dict[
        str, str
    ]  # fault id -> "confounder", "inert", "decoy", "interaction", "drift", "plain", "small"
    small: frozenset[str]
    assumed_joint: bool
    expected: dict[str, Any]
    refusals: dict[str, int] = field(default_factory=dict)  # why earlier attempts failed
    # Private draw handles -> final opaque identities. Never part of the visible fleet.
    # Keeping the handles until assembly preserves earlier seeded author draws.
    fault_id_map: dict[str, str] = field(default_factory=dict)


def opaque_fault_ids(seed: int, handles: Sequence[str]) -> dict[str, str]:
    """Independent, domain-separated pseudorandom tokens; no type/scope/order input.

    Handles are private draft draw keys, not final ids or positions in the accepted
    schedule. A permutation of that schedule leaves each token unchanged. The world
    seed is secret outside development records; knowing another token reveals no key.
    """
    result: dict[str, str] = {}
    for handle in handles:
        for retry in itertools.count():
            message = f"fleetlab/fault-id/v1\0{handle}\0{retry}".encode()
            token = (
                "fx" + hmac.new(seed.to_bytes(8, "big"), message, hashlib.sha256).hexdigest()[:32]
            )
            if is_neutral(IdKind.fault, token):
                result[handle] = token
                break
    if len(set(result.values())) != len(result):
        raise ValueError("opaque fault identity collision")
    return result


def rename_fault_ids[T](value: T, mapping: dict[str, str]) -> T:
    """Rename exact fault references, including pair labels, without changing order.

    Used only on truth-side generator objects. Types, container ordering and every
    non-id value survive; Pydantic objects are revalidated with the new references.
    """
    if not mapping:
        return value
    pattern = re.compile(
        r"(?<![a-zA-Z0-9_])(?:" + "|".join(map(re.escape, mapping)) + r")(?![a-zA-Z0-9_])"
    )

    def walk(item: Any) -> Any:
        if isinstance(item, str):
            return pattern.sub(lambda match: mapping[match[0]], item)
        if isinstance(item, BaseModel):
            return type(item).model_validate(walk(item.model_dump()))
        if is_dataclass(item) and not isinstance(item, type):
            return replace(item, **{f.name: walk(getattr(item, f.name)) for f in fields(item)})
        if isinstance(item, dict):
            return {walk(k): walk(v) for k, v in item.items()}
        if isinstance(item, (list, tuple, set, frozenset)):
            return type(item)(walk(v) for v in item)
        return item

    return cast(T, walk(value))


def identify(comp: Composition) -> Composition:
    """Assign final identities after acceptance, leaving historical worlds unchanged."""
    historical = {
        ("w41", 41),
        ("w42", 42),
        ("w43", 43),
        ("w44", 5540438229502422191),
        ("w45", 6689360473052309405),
    }
    if (comp.recipe.world_id, comp.recipe.seed) in historical:
        return comp
    mapping = opaque_fault_ids(comp.recipe.seed, [f.fault_id for f in comp.spec.fault_schedule])
    final = replace(rename_fault_ids(comp, mapping), fault_id_map=mapping)
    if "window_forecasts" in final.expected:
        # The seal covers the forecast as it is published, under the final identities.
        seal = forecast_seal(
            final.expected["window_forecasts"], final.expected["measured_cause_forecast"]
        )
        final = replace(final, expected={**final.expected, "forecast_seal": seal})
    return final


class Composer:
    def __init__(self, recipe: WorldRecipe, the_curves: curves.Curves | None = None) -> None:
        self.recipe = recipe
        self.curves = the_curves or curves.load()
        if (
            recipe.calibration_sources is not None
            and (
                self._measured_sources()
                if recipe.estimator == "measured-v3"
                else self.curves.sources
            )
            != recipe.calibration_sources
        ):
            raise ValueError("sealed recipe calibration changed; create a new recipe and world")
        self.robots = base.all_robots(recipe.sites, recipe.robots_per_site)
        self.sites = base.site_ids(recipe.sites)
        self.traits = {r: base.traits_for(recipe.seed, r) for r, _ in self.robots}
        self.slots = base.slots(
            recipe.seed,
            recipe.sites,
            recipe.robots_per_site,
            recipe.days,
            recipe.episodes_per_robot_day,
            recipe.tasks,
            recipe.site_suites,
        )
        self.day_mix: dict[tuple[str, str, int], dict[str, int]] = {}
        for slot in self.slots:
            mix = self.day_mix.setdefault((slot.robot_id, slot.site_id, slot.day), {})
            mix[slot.suite] = mix.get(slot.suite, 0) + 1
        self.drift_rates: dict[tuple[str, str], float | None] = {}
        self._pair_choices: dict[tuple[tuple[str, str], tuple[str, str]], list[str]] = {}
        self.site_of = dict(self.robots)
        self.effect_levels = effect_levels(self.curves)
        self.plain_menu = list(PLAIN_MENU)
        if recipe.extension_types:
            import fleetlab.faults as faults

            for kind in EXTENSION_KINDS:
                labels = tuple(
                    p.label
                    for p in getattr(faults.get(kind), "calibration", ())
                    if all(
                        (kind, suite, p.label) in self.curves.points
                        for suite in (curves.SPATIAL, curves.OBJECT)
                    )
                    and any(
                        self.curves.points[(kind, suite, p.label)].role == "effect"
                        for suite in (curves.SPATIAL, curves.OBJECT)
                    )
                )
                if labels:
                    self.effect_levels[kind] = labels
                    scope = "site" if fault_class(kind) is FaultClass.site else "robot"
                    self.plain_menu.append((kind, scope))
        self.spatial_share = {
            e.site_id: (e.share if e.favoured == curves.SPATIAL else 1.0 - e.share)
            for e in base.emphasis(recipe.seed, recipe.sites, recipe.tasks, recipe.site_suites)
        }
        self.windows = {
            "observed": recipe.windows.observed,
            "experiment": recipe.windows.experiment,
            "scoring": recipe.windows.scoring,
        }
        # The measured-context search: one table of context lookups for the composer's
        # whole life (its cells never change), and the slots grouped by robot and day.
        self._cells = self.cells()
        self._weights: dict[tuple[Any, ...], Any] = {}
        self._grouped: dict[tuple[str, str], list[dict[tuple[str, int], int]]] = {}
        self._grouped_slots: object = None
        self._decoy_choices: dict[tuple[str, int, int], list[tuple[str, str]]] = {}
        self._alone: dict[tuple[Any, ...], bool] = {}
        self._margins: dict[tuple[str, int, int], float] = {}

    # -- the recipe's settings ---------------------------------------------------------------

    def cells(self) -> dict[curves.CellKey, curves.MeasuredCell] | None:
        """The measured cells this recipe stands on: the record, or as of its named jobs."""
        if self.recipe.estimator != "measured-v3":
            return None
        jobs = self.recipe.calibration_jobs
        return curves.measured_cells() if jobs is None else curves.measured_cells_of(jobs)

    def _measured_sources(self) -> tuple[str, ...]:
        jobs = self.recipe.calibration_jobs
        return curves.measured_sources() if jobs is None else curves.measured_sources(jobs)

    def decoy_margin(self, decoy: FaultSpec) -> float:
        """The loss on its carriers that the decoy rule must exclude for this decoy.

        "carriers": the recipe's margin as it is. "fleet": the margin is fleet success
        over the scoring window (design note, entry 15: below two points there a planted
        decoy remains a decoy), so on the carriers it is the margin divided by their
        share of that window's attempts: six points for a decoy on one of three sites.
        """
        recipe = self.recipe
        if recipe.decoy_margin_of == "carriers":
            return recipe.decoy_margin
        key = (decoy.target.scope, decoy.schedule.onset_day, id(self.slots))
        if key not in self._margins:
            scoring = [slot for slot in self.slots if slot.day in recipe.windows.scoring]
            carried = sum(
                1
                for slot in scoring
                if decoy.schedule.onset_day <= slot.day
                and covers(decoy.target.scope, slot.robot_id, slot.site_id)
            )
            self._margins[key] = (
                recipe.decoy_margin * len(scoring) / carried if carried else math.inf
            )
        return self._margins[key]

    # -- drawing ---------------------------------------------------------------------------

    def _weighted_robot(
        self, r: np.random.Generator, candidates: Sequence[str], trait: str, spatial: bool = False
    ) -> str:
        """A robot drawn by a trait; with `spatial`, also by its site's share of
        `libero_spatial` episodes (gripper wear and heavier batches act on its thin bowls
        and barely on `libero_object`'s thick objects, as the supplement measured)."""
        values = np.asarray([float(self.traits[c][trait]) for c in candidates]) + 1e-6
        if spatial:
            values = values * np.asarray([self.spatial_share[self.site_of[c]] for c in candidates])
        return candidates[int(r.choice(len(candidates), p=values / values.sum()))]

    def _spatial_site(self, r: np.random.Generator, candidates: Sequence[str]) -> str:
        """A site drawn by its share of `libero_spatial` episodes, squared."""
        values = np.asarray([self.spatial_share[s] ** 2 for s in candidates]) + 1e-9
        return candidates[int(r.choice(len(candidates), p=values / values.sum()))]

    def _free_day(
        self, r: np.random.Generator, d: Draft, scope: str, low: int, high: int
    ) -> int | None:
        """A day in [low, high] on which no fault drafted so far starts at a scope sharing a
        robot with `scope` (day 0 is always free); None if there is none."""
        mine = {x for x, s in self.robots if covers(scope, x, s)}
        taken = {
            f.schedule.onset_day
            for f in d.fixed
            if mine & {x for x, s in self.robots if covers(f.target.scope, x, s)}
        }
        taken |= {
            p.onset for p in d.plains if mine & {x for x, s in self.robots if covers(p.scope, x, s)}
        }
        days = [x for x in range(low, high + 1) if x == 0 or x not in taken]
        return days[int(r.integers(len(days)))] if days else None

    def _joint_pairs(self) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        candidates = dict.fromkeys((*INTERACTION_PAIRS, *(j.members for j in self.curves.joints)))
        return [
            pair
            for pair in candidates
            if self.curves.has_joint(*pair)
            and any(
                j.paired and j.interaction_role == "interaction"
                for j in self.curves.joints
                if set(j.members) == set(pair)
            )
        ]

    def pair_robot_choices(self, pair: tuple[tuple[str, str], tuple[str, str]]) -> list[str]:
        """Require paired support and the dose criteria on a carrier's actual task mix."""
        if self.recipe.estimator != "suite-v1":
            return self.task_pair_robot_choices(pair)
        if (
            self.recipe.robot_success_floor is None
            and self.recipe.interaction_single_loss_points is None
        ):
            return [robot for robot, _ in self.robots]
        out = []
        for robot, _ in self.robots:
            slots = [s for s in self.slots if s.robot_id == robot and s.day >= 1]
            suites = {s.suite for s in slots}
            points = {suite: self.curves.joint(*pair, suite) for suite in suites}
            if any(moved or not j.paired for j, moved in points.values()):
                continue
            upper: dict[str, float] = {}
            losses: dict[str, dict[tuple[str, str], float]] = {}
            for suite, (joint, _) in points.items():
                assert joint.interaction is not None and joint.single_successes is not None
                upper[suite] = joint.interaction[2]
                losses[suite] = {
                    member: 100 * (joint.healthy_successes - success) / joint.n
                    for member, success in zip(joint.members, joint.single_successes, strict=True)
                }
            limit = self.recipe.interaction_single_loss_points
            floor = self.recipe.robot_success_floor
            if limit is not None and any(
                upper[suite] >= 0
                or any(loss > limit for loss in losses[suite].values())
                or (
                    floor is not None
                    and self.curves.healthy[suite] * points[suite][0].ratio < floor
                )
                for suite in suites
            ):
                continue  # the preregistered confirmation permits eligible suites only
            interval = sum(upper[s.suite] for s in slots) / len(slots)
            if interval >= 0:
                continue
            if limit is not None and any(
                sum(losses[s.suite][member] for s in slots) / len(slots) > limit for member in pair
            ):
                continue
            floor = self.recipe.robot_success_floor
            if floor is not None and any(
                sum(
                    self.curves.healthy[s.suite] * points[s.suite][0].ratio
                    for s in slots
                    if s.day == day
                )
                / sum(s.day == day for s in slots)
                < floor
                for day in sorted({s.day for s in slots})
            ):
                continue
            out.append(robot)
        return out

    def task_pair_robot_choices(self, pair: tuple[tuple[str, str], tuple[str, str]]) -> list[str]:
        """Screen the joint and singles on each carrier's actual scheduled tasks."""
        if pair in self._pair_choices:
            return list(self._pair_choices[pair])
        out = []
        for robot, _ in self.robots:
            active = [s for s in self.slots if s.robot_id == robot and s.day >= 1]
            try:
                points = [
                    self.curves.for_task(s.suite, s.task.task_index).joint(*pair, s.suite)[0]
                    for s in active
                ]
            except KeyError:
                continue
            if not points or any(not j.paired for j in points):
                continue
            upper = sum(j.interaction[2] for j in points if j.interaction is not None) / len(points)
            if upper >= 0:
                continue
            limit = self.recipe.interaction_single_loss_points
            if limit is not None and any(
                sum(
                    100
                    * (
                        j.healthy_successes
                        - dict(zip(j.members, j.single_successes or (), strict=True))[member]
                    )
                    / j.n
                    for j in points
                )
                / len(points)
                > limit
                for member in pair
            ):
                continue
            if self.recipe.estimator == "measured-v3" and self.recipe.search_revision != "r6":
                # Paired interaction evidence qualifies the mechanism. The floor and
                # precise cells belong to its actual overlap, checked during placement;
                # another mechanism can rescue a pair that fails the standalone floor.
                out.append(robot)
                continue
            faults = [make_fault("f01", *pair[0], robot, 0), make_fault("f02", *pair[1], robot, 1)]
            measured = self.cells()
            ctx = estimate.Context(
                self.curves,
                frozenset(),
                frozenset(),
                (("f01", "f02"),),
                task_aware=True,
                measured=measured,
                weights_cache=self._shared_weights(measured),
            )
            floor = self.recipe.robot_success_floor
            try:
                carrier_slots = [slot for slot in self.slots if slot.robot_id == robot]
                if ctx.measured is not None:
                    # A supported joint alone is not sufficient for counting its
                    # members: both single-removal contexts must also be measured
                    # on every carried task, including tasks absent before onset.
                    episodes = [
                        estimate.Episode(
                            slot.suite,
                            tuple(base.in_force(faults, slot.robot_id, slot.site_id, slot.day)),
                            slot.task.task_index,
                        )
                        for slot in carrier_slots
                    ]
                    for removed in (("f01",), ("f02",)):
                        estimate.measurement_weights(episodes, ctx, removed)
                rows = estimate.robot_window_success(
                    faults,
                    carrier_slots,
                    ctx,
                    self.recipe.windows,
                )
            except KeyError:
                continue
            if floor is not None and any(
                min(row["true"], row["logged"]) < floor + ctx.success_margin for row in rows
            ):
                continue
            out.append(robot)
        self._pair_choices[pair] = out
        return list(out)

    def pair_scope_choices(self, pair: tuple[tuple[str, str], tuple[str, str]]) -> list[str]:
        """A moderate pair may cover a site, with every carrier independently eligible."""
        robots = self.pair_robot_choices(pair)
        if self.recipe.interaction_scope == "robot":
            return robots
        return [
            site
            for site in self.sites
            if all(robot in robots for robot, here in self.robots if here == site)
        ]

    def decoy_choices(self, scope: str, onset: int) -> list[tuple[str, str]]:
        """Only measured, confirmed-flat levels on every suite that would carry them."""
        key = (scope, onset, id(self.slots))
        if key not in self._decoy_choices:
            self._decoy_choices[key] = self._decoy_choices_now(scope, onset)
        return list(self._decoy_choices[key])

    def _decoy_choices_now(self, scope: str, onset: int) -> list[tuple[str, str]]:
        suites = {
            s.suite for s in self.slots if s.day >= onset and covers(scope, s.robot_id, s.site_id)
        }
        return [
            (kind, label)
            for kind, label in (*DECOYS, ("light_color_shift", "1 warmer"))
            if suites
            and all(
                (point := self.curves.points.get((kind, suite, label))) is not None
                and point.role == "decoy"
                and point.n >= 1000
                for suite in suites
            )
        ]

    def draft(self, attempt: int) -> Draft:
        if self.recipe.estimator == "measured-v3":
            return self.draft_measured(attempt)
        recipe = self.recipe
        r = base.rng(recipe.seed, "compose", attempt)
        occ = Occupancy(self.robots)
        d = Draft()
        counter = itertools.count(1)

        def new_id() -> str:
            return neutral_id(IdKind.fault, next(counter))

        robot_faulted: set[str] = set()
        self._totals = 0
        self._heavy = 0
        self._clear_sites: set[str] = set()
        self._used: dict[tuple[str, str], int] = {}  # plain faults drawn per (kind, scope kind)
        has_fleet_release = recipe.config_share is not None and recipe.small > 0
        releases = itertools.count(0)

        def version_for() -> str:
            major, minor, patch = (int(x) for x in RELEASE_BASE.split("."))
            return f"{major}.{minor + next(releases)}.{patch}"

        # A measured configuration interaction on one robot or site. Every carrier
        # must satisfy the preregistered single, interaction and floor criteria.
        if recipe.interactions:
            pairs = self._joint_pairs()
            if not pairs:
                return d
            feasible = {pair: self.pair_scope_choices(pair) for pair in pairs}
            pairs = [pair for pair in pairs if feasible[pair]]
            if not pairs:
                return d
            pair = pairs[int(r.integers(len(pairs)))]
            if recipe.interaction_scope == "site":
                scope = self._spatial_site(r, feasible[pair])
            else:
                sites = [
                    s for s in self.sites if any(self.site_of[rid] == s for rid in feasible[pair])
                ]
                site = self._spatial_site(r, sites)
                here = [rid for rid in feasible[pair] if self.site_of[rid] == site]
                scope = here[int(r.integers(len(here)))]
            ids = []
            for i, (kind, label) in enumerate(pair):
                fid = new_id()
                d.fixed.append(make_fault(fid, kind, label, scope, i))
                d.roles[fid] = "interaction"
                occ.take(scope, things_of(kind, label))
                ids.append(fid)
            d.patterns = d.patterns.model_copy(update={"interactions": [(ids[0], ids[1])]})
            if recipe.interaction_scope == "robot":
                robot_faulted.add(scope)

        # The additive example has identical visible records to an interaction
        # made from these faults; only the hidden pattern label differs.
        if recipe.additive_pairs:
            wear, batch = ADDITIVE_PAIR
            site = self._spatial_site(r, self.sites)
            here = [rid for rid, s in self.robots if s == site and rid not in robot_faulted]
            robot = self._weighted_robot(r, here, base.TRAIT_PADS, spatial=True)
            batch_id, wear_id = new_id(), new_id()
            onset = self._free_day(r, d, site, 1, 2)
            if onset is None:
                return d
            d.fixed += [
                make_fault(batch_id, *batch, site, onset),
                make_fault(wear_id, *wear, robot, 0),
            ]
            occ.take(site, things_of(*batch))
            occ.take(robot, things_of(*wear))
            d.roles.update({batch_id: "additive_pair", wear_id: "additive_pair"})
            d.patterns = d.patterns.model_copy(update={"additive_pairs": [(wear_id, batch_id)]})
            robot_faulted.add(robot)

        if recipe.fleet_fault:
            fid = new_id()
            # A measured image-pipeline defect is carried by a logged release.
            # Delay/chunk settings cannot also be fleet-wide: the interaction owns them.
            packed = f"camera_degradation|{recipe.fleet_image_blur} px blur"
            candidate = make_fault(
                fid,
                "software_release",
                packed,
                FLEET_SCOPE,
                0,
                nominal_version=base.BASE_VERSION,
                version=version_for(),
            )
            suites = {task.suite for task in recipe.tasks}
            measured = all(not self.curves.ratio(candidate, suite).transferred for suite in suites)
            affected = [self.curves.ratio(candidate, suite).effect for suite in suites]
            if not measured or not (
                any(affected) if recipe.estimator == "measured-v3" else all(affected)
            ):
                return d
            if not occ.free(FLEET_SCOPE, things_of("software_release", packed)):
                return d
            onset = self._free_day(r, d, FLEET_SCOPE, 2, 5)
            if onset is None:
                return d
            d.fixed.append(candidate.model_copy(update={"schedule": Schedule(onset_day=onset)}))
            d.roles[fid] = "fleet"
            occ.take(FLEET_SCOPE, things_of("software_release", packed))

        # 1b. A linear physical drift with a health precursor. A fleet image
        # release occupies every scene camera, so use a different measured sensor.
        if recipe.drifts:
            kind, label, component, maximum = (
                ("gripper_wear", "1 wear", "gripper", 1.0)
                if recipe.fleet_fault or recipe.estimator == "measured-v3"
                else ("camera_pose_offset", "3 deg scene pan", "scene_camera", 9.0)
            )
            free = [
                x for x, _ in self.robots if occ.free(x, [component]) and x not in robot_faulted
            ]
            if not free:
                return d
            robot = self._weighted_robot(r, free, base.TRAIT_MOUNT)
            start = maximum / 100 if recipe.require_warnable_drift else maximum / 9
            end = float(r.uniform(maximum * 0.85, maximum))
            rate = (end - start) / (recipe.days - 1)
            if recipe.require_warnable_drift:
                cache_key = (robot, kind)
                if cache_key not in self.drift_rates:
                    candidates = []
                    for endpoint in np.linspace(start, maximum, 160)[1:]:
                        candidate = make_fault(
                            "f99",
                            kind,
                            label,
                            robot,
                            0,
                            level=start,
                            drift=Drift(
                                shape="linear", rate=(float(endpoint) - start) / (recipe.days - 1)
                            ),
                        )
                        try:
                            crossing = estimate.drift_crossing_day(
                                candidate, self.slots, self.context(d)
                            )
                        except KeyError:
                            continue
                        if crossing in recipe.windows.scoring and (
                            recipe.estimator == "suite-v1"
                            or drift_precursor(candidate, recipe, crossing) is not None
                        ):
                            candidates.append((float(endpoint) - start) / (recipe.days - 1))
                    self.drift_rates[cache_key] = candidates[-1] if candidates else None
                selected_rate = self.drift_rates[cache_key]
                if selected_rate is None:
                    return d
                rate = selected_rate
            drift_id = new_id()
            d.fixed.append(
                make_fault(
                    drift_id,
                    kind,
                    label,
                    robot,
                    0,
                    drift=Drift(shape="linear", rate=rate),
                    level=start,
                )
            )
            occ.take(robot, [component])
            robot_faulted.add(robot)
            d.roles[drift_id] = "drift"
            d.patterns = d.patterns.model_copy(update={"drifts": [drift_id]})

        if not self.floor_holds(d.fixed, d):
            return d

        # 1c. the confounder: an effective fault and an inert change on the same day
        if recipe.confounders:
            # D3's effective fault is a configuration or software change itself (two changes
            # on one day); otherwise it writes no change, so the inert one is all the log shows.
            want = (
                "config"
                if recipe.config_share is not None
                else "any"
                if recipe.estimator == "measured-v3"
                else "other"
            )
            inert = INERT_CHOICES[int(r.integers(len(INERT_CHOICES)))]
            if recipe.confounder_companion is not None:
                inert = INERT_CHOICES[recipe.confounder_companion == "calibration_record"]
            if recipe.estimator == "measured-v3":
                inert = (
                    "calibration_update",
                    "wrist calibration rewritten"
                    if recipe.fleet_fault
                    else "scene calibration rewritten",
                )
            elif recipe.fleet_fault:
                if recipe.confounder_companion == "calibration_record":
                    return d
                inert = INERT_CHOICES[0]
            for _ in range(50):
                inert_free = [x for x, _ in self.robots if occ.free(x, things_of(*inert))]
                if not inert_free:
                    break
                plain = self._draw_plain(
                    r, occ, robot_faulted, want, new_id(), version_for, onset_range=(1, 3), draft=d
                )
                if plain is None:
                    continue
                inert_scope = plain.scope
                if inert[0] == "calibration_update":
                    # Benign calibration records are robot-scoped. A site's real
                    # fault can coincide with such a record on one of its robots.
                    covered = [
                        robot
                        for robot, site in self.robots
                        if covers(plain.scope, robot, site) and occ.free(robot, things_of(*inert))
                    ]
                    if not covered:
                        continue
                    inert_scope = covered[int(r.integers(len(covered)))]
                if not occ.free(inert_scope, things_of(*inert)):
                    continue
                d.plains.append(plain)
                d.roles[plain.fault_id] = "confounder"
                inert_id = new_id()
                # the inert change follows the effective fault's onset (set in `settle`)
                d.fixed.append(make_fault(inert_id, *inert, inert_scope, plain.onset))
                occ.take(inert_scope, things_of(*inert))
                d.roles[inert_id] = "inert"
                d.patterns = d.patterns.model_copy(
                    update={"confounders": [(plain.fault_id, inert_id)]}
                )
                break

        # 1d. Decoys on sites whose complete task mix has confirmed-flat calibration.
        decoy_sites = list(r.permutation(self.sites))
        if recipe.site_suites:
            decoy_sites = [s for s in decoy_sites if self.decoy_choices(str(s), 4)]
        if recipe.decoys and not decoy_sites:
            return d
        decoys = []
        for k in range(recipe.decoys):
            site = str(decoy_sites[k % len(decoy_sites)])
            onset = self._free_day(r, d, site, 1, 4)
            if onset is None:
                continue
            choices = [
                (kind, label)
                for kind, label in self.decoy_choices(site, onset)
                if occ.free(site, things_of(kind, label))
            ]
            if not choices:
                continue
            index = k % len(choices) if recipe.name == "D1" else int(r.integers(len(choices)))
            kind, label = choices[index]
            decoy_id = new_id()
            d.fixed.append(make_fault(decoy_id, kind, label, site, onset))
            occ.take(site, things_of(kind, label))
            d.roles[decoy_id] = "decoy"
            decoys.append(decoy_id)
        d.patterns = d.patterns.model_copy(update={"decoys": decoys})

        # 2. the plain faults; with a share of configuration and software faults, as many
        # of them from those classes as the share needs (the small causes count as such)
        config_needed = 0
        if recipe.config_share is not None:
            counted = [f for f in d.fixed if d.roles.get(f.fault_id) != "decoy"]
            so_far = sum(1 for f in counted if fault_class(f.kind) in CONFIG_CLASSES)
            so_far += sum(1 for p in d.plains if fault_class(p.kind) in CONFIG_CLASSES)
            total = len(counted) + len(d.plains) + recipe.plain + recipe.small
            config_needed = math.ceil(recipe.config_share * total - 1e-9) - so_far - recipe.small
        tries = 0
        drawn = 0
        while drawn < recipe.plain:
            tries += 1
            if tries > 200:
                break
            if recipe.config_share is None:
                want = "any"
            else:
                want = "config" if drawn < config_needed else "other"
            plain = self._draw_plain(r, occ, robot_faulted, want, "", version_for, draft=d)
            if plain is None:
                continue
            plain.fault_id = new_id()
            d.plains.append(plain)
            d.roles[plain.fault_id] = "plain"
            drawn += 1

        # 3. the small causes
        for k in range(recipe.small):
            if k == 0 and has_fleet_release:
                kind, label = FLEET_SMALL_RELEASE
                packed = f"{kind}|{label}"
                if occ.free(FLEET_SCOPE, things_of("software_release", packed)):
                    onset = 1 + int(r.integers(3))
                    fid = new_id()
                    d.fixed.append(
                        make_fault(
                            fid,
                            "software_release",
                            packed,
                            FLEET_SCOPE,
                            onset,
                            nominal_version=base.BASE_VERSION,  # set in `schedule`
                            version=version_for(),
                        )
                    )
                    occ.take(FLEET_SCOPE, things_of("software_release", packed))
                    d.roles[fid] = "small"
                    d.small.add(fid)
                    continue
            for _ in range(30):
                kind, label, scope_kind = SMALL_CAUSES[int(r.integers(len(SMALL_CAUSES)))]
                share = recipe.config_share
                if share is not None and fault_class(kind) not in CONFIG_CLASSES:
                    continue  # the share counts the small causes as configuration faults
                if scope_kind == "robot":
                    scope = self.robots[int(r.integers(len(self.robots)))][0]
                else:
                    scope = self.sites[int(r.integers(len(self.sites)))]
                if not occ.free(scope, things_of(kind, label)):
                    continue
                onset = self._free_day(r, d, scope, 0, 3)
                if onset is None:
                    continue
                fid = new_id()
                d.fixed.append(make_fault(fid, kind, label, scope, onset))
                occ.take(scope, things_of(kind, label))
                d.roles[fid] = "small"
                d.small.add(fid)
                break
        return d

    def draft_measured(self, attempt: int) -> Draft:
        """One attempt of the measured-context search.

        The rules are `check`'s. This only keeps the search on measured ground: the
        fleet release and the site-wide patterns are placed first, and every step takes,
        in an order drawn from the attempt's generator, the first candidate whose
        contexts and single removals are measured, whose robots stay above the floor and
        under which every decoy stays flat (`supported`). The older draft drew each
        pattern's place without looking and found out at the end.
        """
        recipe = self.recipe
        r = base.rng(recipe.seed, "compose-measured", attempt)
        occ = Occupancy(self.robots)
        d = Draft()
        issued = [0]

        def peek(ahead: int = 1) -> str:
            return neutral_id(IdKind.fault, issued[0] + ahead)

        def take(count: int = 1) -> None:
            issued[0] += count

        robot_faulted: set[str] = set()
        self._totals = 0
        self._heavy = 0
        self._clear_sites = set()
        self._used = {}
        releases = itertools.count(0)

        def version_for() -> str:
            major, minor, patch = (int(x) for x in RELEASE_BASE.split("."))
            return f"{major}.{minor + next(releases)}.{patch}"

        def shuffled(items: Sequence[Any]) -> list[Any]:
            return [items[int(i)] for i in r.permutation(len(items))]

        def by_weight(items: Sequence[str], weights: Sequence[float]) -> list[str]:
            """A drawn order in which heavier items tend to come first."""
            keys = [float(r.random()) ** (1.0 / max(float(w), 1e-9)) for w in weights]
            return [item for _, item in sorted(zip(keys, items, strict=True), reverse=True)]

        def fits(extra: Sequence[FaultSpec], scope: str, decoys: Sequence[str] = ()) -> bool:
            return self.supported([*self.schedule(d), *extra], d, scope, decoys)

        def day_free(scope: str, day: int) -> bool:
            """No fault drafted so far starts that day at a scope sharing a robot (day 0 is)."""
            if day == 0:
                return True
            mine = {x for x, s in self.robots if covers(scope, x, s)}
            starts = [(f.target.scope, f.schedule.onset_day) for f in d.fixed]
            starts += [(p.scope, p.onset) for p in d.plains]
            return not any(
                onset == day and mine & {x for x, s in self.robots if covers(other, x, s)}
                for other, onset in starts
            )

        # 1. The fleet-wide release: it meets everything, so it comes first.
        if recipe.fleet_fault:
            packed = f"camera_degradation|{recipe.fleet_image_blur} px blur"
            things = things_of("software_release", packed)
            version = version_for()
            suites = {task.suite for task in recipe.tasks}
            for onset in shuffled(range(2, 6)):
                candidate = make_fault(
                    peek(),
                    "software_release",
                    packed,
                    FLEET_SCOPE,
                    int(onset),
                    nominal_version=base.BASE_VERSION,
                    version=version,
                )
                ratios = [self.curves.ratio(candidate, suite) for suite in suites]
                if any(x.transferred for x in ratios) or not any(x.effect for x in ratios):
                    d.stopped = "the fleet release has no measured effect on any suite"
                    return d
                if occ.free(FLEET_SCOPE, things) and fits([candidate], FLEET_SCOPE):
                    take()
                    d.fixed.append(candidate)
                    d.roles[candidate.fault_id] = "fleet"
                    occ.take(FLEET_SCOPE, things)
                    break
            else:
                d.stopped = "the fleet release has an unmeasured or thin cell on a scheduled task"
                return d

        # 2. The decoys: a site, a kind and a day on which the decoy is flat under what
        # covers its carriers so far. Whatever comes later must keep it flat.
        placed: list[str] = []
        options = [
            (site, kind, label, onset)
            for site in self.sites
            for onset in range(1, 5)
            for kind, label in self.decoy_choices(site, onset)
        ]
        for site, kind, label, onset in shuffled(options):
            if len(placed) >= recipe.decoys:
                break
            if not occ.free(site, things_of(kind, label)) or not day_free(site, onset):
                continue
            candidate = make_fault(peek(), kind, label, site, onset)
            if fits([candidate], site, [candidate.fault_id]):
                take()
                d.fixed.append(candidate)
                occ.take(site, things_of(kind, label))
                d.roles[candidate.fault_id] = "decoy"
                placed.append(candidate.fault_id)
                d.patterns = d.patterns.model_copy(update={"decoys": list(placed)})
        if len(placed) < recipe.decoys:
            d.stopped = "no site on which a decoy is confirmed flat under what covers it"
            return d

        # 3. The additive pair: the batch at a site, the worn pads on one of its robots.
        if recipe.additive_pairs:
            wear, batch = ADDITIVE_PAIR
            found = None
            for site in by_weight(self.sites, [self.spatial_share[x] ** 2 for x in self.sites]):
                if found or not occ.free(site, things_of(*batch)):
                    continue
                here = [
                    rid
                    for rid, x in self.robots
                    if x == site and rid not in robot_faulted and occ.free(rid, things_of(*wear))
                ]
                order = by_weight(here, [float(self.traits[c][base.TRAIT_PADS]) for c in here])
                for onset, robot in itertools.product(shuffled([1, 2]), order):
                    if not day_free(site, onset):
                        continue
                    pair_faults = [
                        make_fault(peek(1), *batch, site, onset),
                        make_fault(peek(2), *wear, robot, 0),
                    ]
                    if fits(pair_faults, site):
                        found = (site, robot, pair_faults)
                        break
            if found is None:
                d.stopped = "no site left where the additive pair meets only measured contexts"
                return d
            site, robot, pair_faults = found
            take(2)
            d.fixed += pair_faults
            occ.take(site, things_of(*batch))
            occ.take(robot, things_of(*wear))
            batch_id, wear_id = (f.fault_id for f in pair_faults)
            d.roles.update({batch_id: "additive_pair", wear_id: "additive_pair"})
            d.patterns = d.patterns.model_copy(update={"additive_pairs": [(wear_id, batch_id)]})
            robot_faulted.add(robot)

        # 4. The interaction, on a carrier the preflight passed and the draft still supports.
        if recipe.interactions:
            chosen = None
            for pair in shuffled(self._joint_pairs()):
                scopes = [x for x in self.pair_scope_choices(pair) if x not in robot_faulted]
                weights = [self.spatial_share[self.site_of.get(x, x)] ** 2 for x in scopes]
                for scope in by_weight(scopes, weights):
                    if not all(occ.free(scope, things_of(k, lb)) for k, lb in pair):
                        continue
                    available = [1]
                    if recipe.search_revision == "r7":
                        available = [day for day in range(1, 4) if day_free(scope, day)]
                    elif recipe.search_revision == "r8":
                        # Round 6's days stand (0 and 1). A carrier on which another fault
                        # starts on day 1 is passed over instead of moving the member.
                        available = [1] if day_free(scope, 1) else []
                    for second_day in available:
                        members = [
                            make_fault(peek(i + 1), kind, label, scope, 0 if i == 0 else second_day)
                            for i, (kind, label) in enumerate(pair)
                        ]
                        if fits(members, scope):
                            chosen = (pair, scope, members)
                            break
                    if chosen:
                        break
                if chosen:
                    break
            if chosen is None:
                d.stopped = "no interaction carrier left on measured contexts above the floor"
                return d
            pair, scope, members = chosen
            take(2)
            d.fixed += members
            for (kind, label), member in zip(pair, members, strict=True):
                d.roles[member.fault_id] = "interaction"
                occ.take(scope, things_of(kind, label))
            d.patterns = d.patterns.model_copy(
                update={"interactions": [(members[0].fault_id, members[1].fault_id)]}
            )
            if scope in self.site_of:
                robot_faulted.add(scope)

        # 5. The drift: worn pads growing linearly, crossing its own five points in the
        # scoring window, with a health reading that moves before.
        if recipe.drifts:
            kind, label, component, maximum = "gripper_wear", "1 wear", "gripper", 1.0
            free = [
                x for x, _ in self.robots if occ.free(x, [component]) and x not in robot_faulted
            ]
            order = by_weight(free, [float(self.traits[c][base.TRAIT_MOUNT]) for c in free])
            for robot in order:
                rate = self.measured_drift_rate(robot, kind, label, maximum)
                if rate is None:
                    continue
                candidate = make_fault(
                    peek(),
                    kind,
                    label,
                    robot,
                    0,
                    drift=Drift(shape="linear", rate=rate),
                    level=maximum / 100,
                )
                if fits([candidate], robot):
                    take()
                    d.fixed.append(candidate)
                    occ.take(robot, [component])
                    robot_faulted.add(robot)
                    d.roles[candidate.fault_id] = "drift"
                    d.patterns = d.patterns.model_copy(update={"drifts": [candidate.fault_id]})
                    break
            else:
                d.stopped = "no robot left for a drift crossing in the scoring window"
                return d

        # 6. The confounder: an effective fault and an inert calibration record that day.
        if recipe.confounders:
            want = "config" if recipe.config_share is not None else "any"
            inert = (
                "calibration_update",
                "wrist calibration rewritten"
                if recipe.fleet_fault
                else "scene calibration rewritten",
            )
            for _ in range(50):
                plain = self._draw_plain(
                    r, occ, robot_faulted, want, "", version_for, onset_range=(1, 3), draft=d
                )
                if plain is None:
                    continue
                covered = [
                    robot
                    for robot, site in self.robots
                    if covers(plain.scope, robot, site) and occ.free(robot, things_of(*inert))
                ]
                if not covered:
                    continue
                plain.fault_id = peek(1)
                inert_fault = make_fault(
                    peek(2), *inert, covered[int(r.integers(len(covered)))], plain.onset
                )
                take(2)
                d.plains.append(plain)
                d.roles[plain.fault_id] = "confounder"
                d.fixed.append(inert_fault)
                occ.take(inert_fault.target.scope, things_of(*inert))
                d.roles[inert_fault.fault_id] = "inert"
                d.patterns = d.patterns.model_copy(
                    update={"confounders": [(plain.fault_id, inert_fault.fault_id)]}
                )
                break

        # 7. The plain faults, as many from configuration and software as the share needs.
        config_needed = 0
        if recipe.config_share is not None:
            counted = [f for f in d.fixed if d.roles.get(f.fault_id) != "decoy"]
            so_far = sum(1 for f in counted if fault_class(f.kind) in CONFIG_CLASSES)
            so_far += sum(1 for p in d.plains if fault_class(p.kind) in CONFIG_CLASSES)
            total = len(counted) + len(d.plains) + recipe.plain + recipe.small
            config_needed = math.ceil(recipe.config_share * total - 1e-9) - so_far - recipe.small
        drawn = misses = 0
        for _ in range(200):
            if drawn >= recipe.plain or misses >= 60:
                break
            if recipe.config_share is None:
                want = "any"
            else:
                want = "config" if drawn < config_needed else "other"
            plain = self._draw_plain(r, occ, robot_faulted, want, "", version_for, draft=d)
            if plain is None:
                misses += 1
                continue
            misses = 0
            plain.fault_id = peek()
            take()
            d.plains.append(plain)
            d.roles[plain.fault_id] = "plain"
            drawn += 1
        if drawn < recipe.plain:
            d.stopped = f"only {drawn} of {recipe.plain} plain faults fit measured contexts"

        # 8. Small causes, where a recipe asks for separate ones.
        for _ in range(recipe.small):
            for _ in range(30):
                kind, label, scope_kind = SMALL_CAUSES[int(r.integers(len(SMALL_CAUSES)))]
                if recipe.config_share is not None and fault_class(kind) not in CONFIG_CLASSES:
                    continue
                scopes = [x for x, _ in self.robots] if scope_kind == "robot" else self.sites
                scope = scopes[int(r.integers(len(scopes)))]
                onset = int(r.integers(0, 4))
                if not occ.free(scope, things_of(kind, label)) or not day_free(scope, onset):
                    continue
                candidate = make_fault(peek(), kind, label, scope, onset)
                if fits([candidate], scope):
                    take()
                    d.fixed.append(candidate)
                    occ.take(scope, things_of(kind, label))
                    d.roles[candidate.fault_id] = "small"
                    d.small.add(candidate.fault_id)
                    break
        return d

    def measured_drift_rate(
        self, robot: str, kind: str, label: str, maximum: float
    ) -> float | None:
        """A steep linear drift on a robot whose own crossing day lies in the scoring
        window and whose health reading shows before; None where the search finds none.

        The older search tried all 159 endpoints and kept the steepest that qualifies.
        This one tries every eighth from the steepest down and then the seven above the
        first that qualifies: a robot's daily task mix makes the crossing day uneven in
        the rate, so it cannot be halved, and a qualifying stretch narrower than eight
        endpoints may be missed (the robot then gets no drift).
        """
        key = (robot, kind)
        if key in self.drift_rates:
            return self.drift_rates[key]
        recipe = self.recipe
        start = maximum / 100
        ends = np.linspace(start, maximum, 160)[1:]
        own = [slot for slot in self.slots if slot.robot_id == robot]
        ctx = self.context(Draft())

        def rate_at(index: int) -> float | None:
            rate = (float(ends[index]) - start) / (recipe.days - 1)
            fault = make_fault(
                "f99", kind, label, robot, 0, level=start, drift=Drift(shape="linear", rate=rate)
            )
            day = estimate.drift_crossing_day(fault, own, ctx)
            if day in recipe.windows.scoring and drift_precursor(fault, recipe, day) is not None:
                return rate
            return None

        found: float | None = None
        try:
            for index in range(len(ends) - 1, -1, -8):
                found = rate_at(index)
                if found is not None:
                    for above in range(min(index + 7, len(ends) - 1), index, -1):
                        steeper = rate_at(above)
                        if steeper is not None:
                            found = steeper
                            break
                    break
        except KeyError:
            found = None  # a knot of the wear curve is unmeasured or thin on this robot's tasks
        self.drift_rates[key] = found
        return found

    def stands_alone(
        self, kind: str, labels: Sequence[str], scope: str, draft: Draft | None
    ) -> bool:
        """Whether some level of a plain fault at a scope is on measured ground by itself
        (with the fleet release, where the draft has one), from its earliest day.

        A filter for the measured search, asked once per composer: a placement that
        fails alone is not tried in every draft again. Whether it fits beside the other
        faults of a draft is `supported`'s to say.
        """
        fleet = [f for f in (draft.fixed if draft else []) if f.target.scope == FLEET_SCOPE]
        key = (kind, tuple(labels), scope, tuple(estimate.mechanism_signature(f) for f in fleet))
        if key not in self._alone:
            onset = 0 if scope in self.site_of else 1
            release = kind == "software_release"
            self._alone[key] = any(
                self.supported(
                    [
                        *fleet,
                        make_fault(
                            "f99999",
                            kind,
                            label,
                            scope,
                            onset,
                            nominal_version=base.BASE_VERSION if release else None,
                            version=RELEASE_BASE if release else None,
                        ),
                    ],
                    Draft(),
                    scope,
                )
                for label in labels
            )
        return self._alone[key]

    def _draw_plain(
        self,
        r: np.random.Generator,
        occ: Occupancy,
        robot_faulted: set[str],
        want: str,
        fault_id: str,
        version_for: Callable[[], str],
        onset_range: tuple[int, int] | None = None,
        *,
        draft: Draft | None = None,
    ) -> Plain | None:
        menu = [
            (k, s)
            for k, s in self.plain_menu
            if want == "any" or (fault_class(k) in CONFIG_CLASSES) == (want == "config")
        ]
        if self.recipe.strong:  # clear faults only: each can reach the detectable size alone
            menu = [(k, s) for k, s in menu if (k, s) not in WEAK_ALONE]
        if onset_range is not None:  # a confounder's effective fault changes what robots do
            menu = [(k, s) for k, s in menu if fault_class(k) is not FaultClass.data]
        if not menu:
            return None
        kind, scope_kind = menu[int(r.integers(len(menu)))]
        # A release counts as what it carries: against that kind's repeats and, when it
        # carries a total failure, against the world's budget of those.
        labels: tuple[str, ...]
        if kind == "software_release":
            carries = list(RELEASE_CARRIES)
            if self.recipe.robot_success_floor is not None:
                # The measured image defect can arrive in a release at any scope.
                carries.extend(
                    ("camera_degradation", label)
                    for label in self.effect_levels.get("camera_degradation", ())
                )
            carried = carries[int(r.integers(len(carries)))]
            labels = (f"{carried[0]}|{carried[1]}",)
            mechanism = carried[0]
        else:
            if kind not in self.effect_levels:
                return None
            labels = self.effect_levels[kind]
            mechanism = kind
        cap = self.recipe.repeats
        if self._used.get((mechanism, scope_kind), 0) >= cap:
            return None
        total = mechanism in TOTAL_FAILURE
        if total and self.recipe.robot_success_floor is not None:
            return None
        if total and not self.recipe.strong and self._totals >= self.recipe.total_failures:
            return None
        searching = self.recipe.estimator == "measured-v3" and draft is not None
        if scope_kind == "robot":
            free = [x for x, _ in self.robots if x not in robot_faulted]
            if searching and self.recipe.search_revision == "r6":
                free = [x for x in free if self.stands_alone(kind, labels, x, draft)]
            if kind in ("camera_pose_offset", "camera_degradation"):
                if not free:
                    return None
                scope = self._weighted_robot(r, free, base.TRAIT_MOUNT)
            elif kind == "gripper_wear" and free:
                scope = self._weighted_robot(r, free, base.TRAIT_PADS, spatial=True)
            elif free:
                scope = free[int(r.integers(len(free)))]
            else:
                return None
            last = ROBOT_LAST_ONSET
        else:
            open_sites = [x for x in self.sites if x not in self._clear_sites] or self.sites
            if searching and self.recipe.search_revision == "r6":
                open_sites = [x for x in open_sites if self.stands_alone(kind, labels, x, draft)]
                if not open_sites:
                    return None
            if kind == "object_batch_variant":
                scope = self._spatial_site(r, open_sites)
            else:
                scope = open_sites[int(r.integers(len(open_sites)))]
            last = SITE_LAST_ONSET
        if not occ.free(scope, things_of(kind, labels[0])):
            return None
        heavy = scope_kind == "site" and mechanism in HEAVY_AT_SITE
        if heavy and self._heavy >= MAX_HEAVY_AT_SITE:
            return None
        # A site-wide fault at the planned difficulty starts after day 0, so the observed
        # window holds a before and an after.
        first = 1 if scope_kind == "site" and not self.recipe.strong else 0
        if self.recipe.strong:
            level = len(labels) - 1
            onset = int(r.integers(0, 2))
        else:
            level = int(r.integers(len(labels)))
            onset = int(r.integers(first, last + 1))
        if onset_range is not None:
            onset = int(r.integers(onset_range[0], onset_range[1] + 1))
            if searching and draft is not None:
                # The confounder's day is fixed: take one on which nothing else starts
                # at an overlapping scope, instead of finding the clash at the end.
                free_day = self._free_day(r, draft, scope, onset_range[0], onset_range[1])
                if free_day is None:
                    return None
                onset = free_day
            first, last = onset, onset
        elif searching and draft is not None and self.recipe.search_revision == "r7":
            free_day = self._free_day(r, draft, scope, first, last)
            if free_day is None:
                return None
            onset = free_day
        elif searching and draft is not None and self.recipe.search_revision == "r8":
            # The day is the one drawn above, as round 6 drew it. Only a day on which
            # another fault starts at an overlapping scope is replaced, by a drawn free
            # one (round 6 did that when it settled).
            drawn = Plain(kind, scope, labels, level, onset, last, fault_id, None, first)
            if not self._onset_free(draft, drawn, onset):
                free_day = self._free_day(r, draft, scope, first, last)
                if free_day is None:
                    return None
                onset = free_day
        if self.recipe.robot_success_floor is not None and draft is not None:
            schedule = self.schedule(draft)
            if self.recipe.search_revision != "r6":
                proposal = Plain(kind, scope, labels, level, onset, last, fault_id, None, first)
                days: Sequence[int] = range(first, last + 1)
                if onset_range is not None:
                    days = range(onset_range[0], onset_range[1] + 1)
                if self.recipe.search_revision == "r8":
                    # Levels are ranked toward the forecast; the day is not (decision of
                    # 2026-10-11: nobody decided that faults should start earlier).
                    days = [onset]
                ctx = self.context(draft)
                middle = sum(self.recipe.aim) / 2
                ranked = []
                for index, label in enumerate(labels):
                    for day in days:
                        if not self._onset_free(draft, proposal, day):
                            continue
                        candidate = make_fault(
                            fault_id or "f99999",
                            kind,
                            label,
                            scope,
                            day,
                            nominal_version=base.BASE_VERSION
                            if kind == "software_release"
                            else None,
                            version=RELEASE_BASE if kind == "software_release" else None,
                        )
                        trial = [*schedule, candidate]
                        try:
                            forecast = self.observed_logged(trial, ctx)
                        except KeyError:
                            continue
                        ranked.append((abs(forecast - middle), index, day, candidate))
                for _, index, day, candidate in sorted(ranked, key=lambda item: item[:3]):
                    if self.supported([*schedule, candidate], draft, scope):
                        level, onset = index, day
                        if onset_range is not None:
                            first = last = onset
                        break
                else:
                    return None
                # Keep the whole measured level menu. A level that fails at this
                # onset may be supported at another onset during settling.
            else:
                feasible = []
                for label in labels:
                    candidate = make_fault(
                        fault_id or "f99999",
                        kind,
                        label,
                        scope,
                        onset,
                        nominal_version=base.BASE_VERSION if kind == "software_release" else None,
                        version=RELEASE_BASE if kind == "software_release" else None,
                    )
                    if self.supported([*schedule, candidate], draft, scope):
                        feasible.append(label)
                if not feasible:
                    return None
                labels = tuple(feasible)
                level = min(level, len(labels) - 1)
        occ.take(scope, things_of(kind, labels[0]))
        if scope_kind == "robot":
            robot_faulted.add(scope)
        self._used[(mechanism, scope_kind)] = self._used.get((mechanism, scope_kind), 0) + 1
        if total:
            self._totals += 1
        if heavy:
            self._heavy += 1
        version = version_for() if kind == "software_release" else None
        return Plain(kind, scope, labels, level, onset, last, fault_id, version, first)

    # -- settling the levels -------------------------------------------------------------------

    def schedule(self, d: Draft) -> list[FaultSpec]:
        """The draft's faults at their current levels and onsets; the inert entry at its
        effective fault's onset; releases numbered by onset, each replacing the version
        the fleet ran that day."""
        out = list(d.fixed)
        for p in d.plains:
            out.append(
                make_fault(
                    p.fault_id,
                    p.kind,
                    p.labels[p.level],
                    p.scope,
                    p.onset,
                    nominal_version=base.BASE_VERSION if p.kind == "software_release" else None,
                    version=p.version,
                )
            )
        by_id = {f.fault_id: i for i, f in enumerate(out)}
        for effective, inert in d.patterns.confounders:
            onset = next(p.onset for p in d.plains if p.fault_id == effective)
            i = by_id[inert]
            out[i] = out[i].model_copy(
                update={"schedule": out[i].schedule.model_copy(update={"onset_day": onset})}
            )
        for i, fault in enumerate(out):
            if fault.kind == "calibration_update":
                assert isinstance(fault.nominal, list) and fault.target.config_key is not None
                value = calibration_value(
                    self.recipe.seed,
                    fault.target.scope,
                    fault.schedule.onset_day,
                    fault.target.config_key.value,
                    [float(v) for v in fault.nominal],
                )
                out[i] = fault.model_copy(update={"params": {**fault.params, "value": value}})
        timeline = base.software_timeline(self.recipe.seed, self.recipe.days, out)
        releases = sorted(
            (f for f in out if f.target.config_key is ConfigKey.software_version),
            key=lambda f: (f.schedule.onset_day, f.fault_id),
        )
        major, minor, patch = (int(x) for x in RELEASE_BASE.split("."))
        for k, release in enumerate(releases):
            params = {**release.params, "version": f"{major}.{minor + k}.{patch}"}
            nominal = base.fleet_version_on(timeline, release.schedule.onset_day)
            out[by_id[release.fault_id]] = release.model_copy(
                update={"params": params, "nominal": nominal}
            )
        if self.recipe.extension_types:
            shapes = {ConditionKey.placement: [0.0, 0.0, 0.0], ConditionKey.lighting: [0.0, 0.0]}
            out = [
                f.model_copy(update={"nominal": shapes[f.target.condition_key]})
                if f.target.condition_key in shapes
                else f
                for f in out
            ]
        return sorted(out, key=lambda f: f.fault_id)

    def context(self, d: Draft) -> estimate.Context:
        neutral = frozenset({*d.patterns.decoys, *(inert for _, inert in d.patterns.confounders)})
        the_curves = self.curves
        if d.assumed_joint:
            the_curves = with_assumed_joints(self.curves, d, self)
        measured = self.cells()
        return estimate.Context(
            curves=the_curves,
            neutral=neutral,
            small=frozenset(d.small),
            pairs=tuple([*d.patterns.interactions, *d.patterns.additive_pairs]),
            task_aware=self.recipe.estimator != "suite-v1",
            measured=measured,
            retained_error=self.recipe.retained_error,
            weights_cache=self._shared_weights(measured),
        )

    def _shared_weights(self, measured: object) -> dict[tuple[Any, ...], Any] | None:
        """The composer's lookup table, only for the cells it was built on."""
        return self._weights if measured is not None and measured is self._cells else None

    def robot_days(self) -> dict[tuple[str, str], list[dict[tuple[str, int], int]]]:
        """Per robot and day, how often each task is scheduled (regrouped if slots change)."""
        if self._grouped_slots is not self.slots:
            grouped: dict[tuple[str, str], list[dict[tuple[str, int], int]]] = {}
            for slot in self.slots:
                days = grouped.setdefault((slot.robot_id, slot.site_id), [])
                while len(days) <= slot.day:
                    days.append({})
                key = (slot.suite, slot.task.task_index)
                days[slot.day][key] = days[slot.day].get(key, 0) + 1
            self._grouped, self._grouped_slots = grouped, self.slots
        return self._grouped

    def episodes(self, schedule: Sequence[FaultSpec], days: range) -> list[estimate.Episode]:
        return [
            estimate.Episode(
                s.suite,
                tuple(base.in_force(schedule, s.robot_id, s.site_id, s.day)),
                s.task.task_index,
            )
            for s in self.slots
            if s.day in days
        ]

    def observed_success(self, schedule: Sequence[FaultSpec], ctx: estimate.Context) -> float:
        obs = self.recipe.windows.observed
        eps = self.episodes(schedule, range(obs.first, obs.last + 1))
        return estimate.mean_success(eps, ctx, logged=True)

    def floor_holds(
        self, schedule: Sequence[FaultSpec], draft: Draft, scope: str | None = None
    ) -> bool:
        """Require measured contexts and the window floor while choosing doses.

        An unsupported candidate is skipped locally, rather than abandoning a
        draft before trying another calibrated level. Final acceptance repeats
        the evidence check and reports its missing cell explicitly.
        """
        floor = self.recipe.robot_success_floor
        ctx = self.context(draft)
        if ctx.measured is not None:
            return self._measured_floor(schedule, ctx, scope)
        if floor is None:
            return True
        if ctx.task_aware:
            return all(
                min(row["true"], row["logged"]) >= floor + ctx.success_margin
                for row in estimate.robot_window_success(
                    schedule, self.slots, ctx, self.recipe.windows
                )
                if scope is None or covers(scope, row["robot_id"], self.site_of[row["robot_id"]])
            )
        for (robot, site, day), mix in self.day_mix.items():
            if scope is not None and not covers(scope, robot, site):
                continue
            active = base.in_force(schedule, robot, site, day)
            for logged in (False, True):
                success = sum(
                    n * estimate.probability(active, suite, ctx, logged) for suite, n in mix.items()
                ) / sum(mix.values())
                if success < floor:
                    return False
        return True

    def _measured_floor(
        self, schedule: Sequence[FaultSpec], ctx: estimate.Context, scope: str | None
    ) -> bool:
        """Every context and every single removal measured, each robot window above the floor.

        The same two conditions as before, read robot by robot and day by day: the faults
        in force are the same for all of a robot's attempts of a day.
        """
        assert ctx.measured is not None
        cells = ctx.measured
        floor = self.recipe.robot_success_floor
        need = None if floor is None else floor + ctx.success_margin
        try:
            for (robot, site), days in self.robot_days().items():
                if scope is not None and not covers(scope, robot, site):
                    continue
                totals = {name: [0.0, 0.0, 0.0] for name in self.windows}
                for day, tasks in enumerate(days):
                    if not tasks:
                        continue
                    active = faults_in_force(schedule, robot, site, day)
                    marks = tuple(estimate.mechanism_signature(f) for f in active)
                    removals = [
                        ([f for f in active if f is not gone], marks[:i] + marks[i + 1 :])
                        for i, gone in enumerate(active)
                    ]
                    window = next(
                        (name for name, span in self.windows.items() if day in span), None
                    )
                    for (suite, task), count in tasks.items():
                        weights = ctx.weights(active, suite, task, marks)
                        for kept, kept_marks in removals:
                            ctx.weights(kept, suite, task, kept_marks)
                        if need is not None and window is not None:
                            row = totals[window]
                            row[0] += count * sum(w * cells[k].rate() for k, w in weights.items())
                            row[1] += count * sum(
                                w * cells[k].rate(True) for k, w in weights.items()
                            )
                            row[2] += count
                if need is not None and any(
                    n and min(true, logged) / n < need for true, logged, n in totals.values()
                ):
                    return False
        except KeyError:
            return False
        return True

    def decoys_flat(
        self,
        schedule: Sequence[FaultSpec],
        decoys: Sequence[str],
        scope: str | None = None,
        *,
        scoring: bool = True,
    ) -> bool:
        """Each of these decoys flat in every context that covers its carriers.

        `calibration.context_decoy_confirmed` asked robot by robot and day by day, for
        the search; `check` asks the original again on the finished schedule. With a
        `scope`, only decoys whose carriers that scope can reach are asked.
        """
        from fleetlab.worlds import calibration

        wanted = set(decoys)
        for decoy in schedule:
            if decoy.fault_id not in wanted:
                continue
            label = curves.calibrated_label(decoy)
            if label is None:
                return False
            mixes: dict[tuple[str, tuple[tuple[str, str], ...]], dict[int, int]] = {}
            touched = scope is None
            for (robot, site), days in self.robot_days().items():
                if not decoy.target.covers(robot, site):
                    continue
                touched = touched or covers(str(scope), robot, site)
                for day, tasks in enumerate(days):
                    if (scoring and day not in self.recipe.windows.scoring) or not tasks:
                        continue
                    active = faults_in_force(schedule, robot, site, day)
                    if not any(f.fault_id == decoy.fault_id for f in active):
                        continue
                    try:
                        members = calibration.mechanism_members(active)
                    except KeyError:
                        return False
                    for (suite, task), count in tasks.items():
                        mix = mixes.setdefault((suite, members), {})
                        mix[task] = mix.get(task, 0) + count
            if touched and not calibration.decoy_flat(
                decoy.kind, label, mixes, self.decoy_margin(decoy), self.recipe.calibration_jobs
            ):
                return False
        return True

    def supported(
        self,
        schedule: Sequence[FaultSpec],
        draft: Draft,
        scope: str | None = None,
        decoys: Sequence[str] = (),
    ) -> bool:
        """Whether a (partial) schedule stands on measured ground, for the search.

        The floor and, with measured contexts, every context and single removal of the
        robots in `scope`, and every decoy of the draft (and `decoys`, for a candidate
        not yet in it) flat under what covers its carriers. `check` decides acceptance;
        this keeps the search from proposing what `check` must refuse.
        """
        if not self.floor_holds(schedule, draft, scope):
            return False
        if self.recipe.estimator != "measured-v3":
            return True
        return self.decoys_flat(
            schedule,
            [*draft.patterns.decoys, *decoys],
            scope,
            scoring=self.recipe.search_revision != "r6",
        )

    def permitted_move(self, draft: Draft, plain: Plain, move: str) -> bool:
        """A proposed settling move must preserve the floor with every fault present."""
        if self.recipe.robot_success_floor is None:
            return True
        level, onset = plain.level, plain.onset
        plain.level += {"up": 1, "down": -1}.get(move, 0)
        plain.onset += {"earlier": -1, "later": 1}.get(move, 0)
        try:
            if self.recipe.estimator == "measured-v3":
                # A move changes only the contexts of the robots the fault reaches.
                return self.supported(self.schedule(draft), draft, plain.scope)
            return self.floor_holds(self.schedule(draft), draft)
        finally:
            plain.level, plain.onset = level, onset

    def observed_logged(self, schedule: Sequence[FaultSpec], ctx: estimate.Context) -> float:
        """`observed_success` for the measured search, read robot by robot and day by day."""
        if ctx.measured is None:
            return self.observed_success(schedule, ctx)
        observed = self.recipe.windows.observed
        total = attempts = 0.0
        for (robot, site), days in self.robot_days().items():
            for day, tasks in enumerate(days):
                if day not in observed or not tasks:
                    continue
                active = faults_in_force(schedule, robot, site, day)
                marks = tuple(estimate.mechanism_signature(f) for f in active)
                for (suite, task), count in tasks.items():
                    weights = ctx.weights(active, suite, task, marks)
                    total += count * sum(w * ctx.measured[k].rate(True) for k, w in weights.items())
                    attempts += count
        return total / attempts

    def settle_measured(self, d: Draft, attempt: int) -> list[FaultSpec]:
        """Move plain faults' levels and onsets toward the middle of the aim.

        The same moves as `settle`. Each step takes the supported move that brings the
        observed forecast closest to the aim's middle, and stops inside the aim or when
        no move brings it closer: the aim of a measured recipe is under two points wide,
        and a move drawn at random steps over it.
        """
        if self.recipe.search_revision != "r6":
            return self.settle_contexts(d)
        r = base.rng(self.recipe.seed, "settle-measured", attempt)
        lo, hi = self.recipe.aim
        middle = (lo + hi) / 2
        ctx = self.context(d)
        for p in d.plains:
            if d.roles.get(p.fault_id) == "confounder" or self._onset_free(d, p, p.onset):
                continue
            days = [x for x in range(p.first_onset, p.last_onset + 1) if x != p.onset]
            for day in [days[int(i)] for i in r.permutation(len(days))]:
                if self._onset_free(d, p, day):
                    p.onset = day
                    break
        shift = {"up": (1, 0), "down": (-1, 0), "earlier": (0, -1), "later": (0, 1)}
        try:
            for _ in range(AIM_STEPS):
                value = self.observed_logged(self.schedule(d), ctx)
                if lo <= value <= hi:
                    break
                options: list[tuple[Plain, str]] = []
                for p in d.plains:
                    free = d.roles.get(p.fault_id) != "confounder"
                    if p.level + 1 < len(p.labels):
                        options.append((p, "up"))
                    if p.level > 0:
                        options.append((p, "down"))
                    if free and p.onset > p.first_onset and self._onset_free(d, p, p.onset - 1):
                        options.append((p, "earlier"))
                    if free and p.onset < p.last_onset and self._onset_free(d, p, p.onset + 1):
                        options.append((p, "later"))
                ranked: list[tuple[float, int, Plain, str]] = []
                for index in r.permutation(len(options)):
                    p, move = options[int(index)]
                    level, onset = p.level, p.onset
                    p.level, p.onset = level + shift[move][0], onset + shift[move][1]
                    try:
                        moved = self.observed_logged(self.schedule(d), ctx)
                    except KeyError:
                        moved = None
                    finally:
                        p.level, p.onset = level, onset
                    if moved is not None and abs(moved - middle) < abs(value - middle):
                        ranked.append((abs(moved - middle), len(ranked), p, move))
                for _, _, p, move in sorted(ranked, key=lambda row: row[:2]):
                    if self.permitted_move(d, p, move):
                        p.level += shift[move][0]
                        p.onset += shift[move][1]
                        break
                else:
                    break
        except KeyError:
            pass  # an unmeasured context: `check` names it
        return self.schedule(d)

    def settle_contexts(self, draft: Draft) -> list[FaultSpec]:
        """Search full level/onset moves, pattern onsets, then pairs of moves.

        Rank on the raw cell forecast, just as the recipe's target is defined.
        Every accepted move checks the actual overlap's cells, floor, decoys and
        onset collisions. Pairs can cross a plateau where neither single helps.
        """
        lo, hi = self.recipe.aim
        middle = (lo + hi) / 2
        ctx = self.context(draft)
        # (plain or fixed, index, level, onset); only distinct entries can move together.
        Move = tuple[str, int, int, int]

        def moved(moves: Sequence[Move]) -> Draft:
            result = replace(
                draft, plains=[replace(p) for p in draft.plains], fixed=list(draft.fixed)
            )
            for kind, index, level, onset in moves:
                if kind == "plain":
                    result.plains[index].level = level
                    result.plains[index].onset = onset
                else:
                    fault = result.fixed[index]
                    result.fixed[index] = fault.model_copy(
                        update={"schedule": fault.schedule.model_copy(update={"onset_day": onset})}
                    )
            return result

        # Revision r8 moves a plain fault's day as round 6 did: one day earlier or later,
        # never the confounder's, and never the interaction's members. Levels still move
        # freely, and so do the days of the release, the decoy and the batch.
        drawn_days = self.recipe.search_revision == "r8"

        def options() -> list[Move]:
            choices: list[Move] = []
            for index, p in enumerate(draft.plains):
                confounder = draft.roles.get(p.fault_id) == "confounder"
                if drawn_days:
                    steps = [] if confounder else [p.onset - 1, p.onset + 1]
                    choices.extend(
                        ("plain", index, level, p.onset)
                        for level in range(len(p.labels))
                        if level != p.level
                    )
                    choices.extend(
                        ("plain", index, p.level, day)
                        for day in steps
                        if p.first_onset <= day <= p.last_onset
                    )
                    continue
                days = range(1, 4) if confounder else range(p.first_onset, p.last_onset + 1)
                choices.extend(
                    ("plain", index, level, day)
                    for level in range(len(p.labels))
                    for day in days
                    if (level, day) != (p.level, p.onset)
                )
            for index, fault in enumerate(draft.fixed):
                role = draft.roles.get(fault.fault_id)
                if role == "fleet":
                    days = range(2, 6)
                elif role == "decoy":
                    days = range(1, 5)
                elif role == "interaction":
                    if drawn_days:
                        continue
                    days = range(0, 4)
                elif role == "additive_pair" and fault.target.scope in self.sites:
                    days = range(1, 4)
                else:
                    continue
                choices.extend(
                    ("fixed", index, 0, day) for day in days if day != fault.schedule.onset_day
                )
            return choices

        def value_of(candidate: Draft) -> float | None:
            try:
                return self.observed_logged(self.schedule(candidate), ctx)
            except KeyError:
                return None

        def accepts(candidate: Draft) -> bool:
            schedule = self.schedule(candidate)
            return not self.co_onsets(schedule, candidate.patterns) and self.supported(
                schedule, candidate
            )

        for _ in range(AIM_STEPS):
            value = value_of(draft)
            if value is None or lo <= value <= hi:
                break
            distance = abs(value - middle)
            singles = []
            for move in options():
                candidate = moved([move])
                forecast = value_of(candidate)
                if forecast is not None:
                    singles.append((abs(forecast - middle), move, forecast))
            chosen = None
            for error, move, _ in sorted(singles):
                if error + 1e-10 >= distance:
                    break
                candidate = moved([move])
                if accepts(candidate):
                    chosen = candidate
                    break
            if chosen is None:
                # Rank by additive change only to order exploration; acceptance always
                # recomputes the actual joint context, never a product or additive proxy.
                pairs = sorted(
                    (abs(a[2] + b[2] - value - middle), a[1], b[1])
                    for a, b in itertools.combinations(singles, 2)
                    if a[1][:2] != b[1][:2]
                )
                for _, first, second in pairs[:64]:
                    candidate = moved([first, second])
                    forecast = value_of(candidate)
                    if (
                        forecast is not None
                        and abs(forecast - middle) + 1e-10 < distance
                        and accepts(candidate)
                    ):
                        chosen = candidate
                        break
            if chosen is None:
                break
            draft.fixed, draft.plains = chosen.fixed, chosen.plains
        return self.schedule(draft)

    def settle(self, d: Draft, attempt: int) -> list[FaultSpec]:
        """Move plain faults' levels and onsets until the expected success is in the aim."""
        if self.recipe.estimator == "measured-v3":
            return self.settle_measured(d, attempt)
        r = base.rng(self.recipe.seed, "settle", attempt)
        lo, hi = self.recipe.aim
        ctx = self.context(d)
        movable = [p for p in d.plains]
        # First move plain faults off days on which an overlapping fault starts.
        for p in movable:
            if d.roles.get(p.fault_id) == "confounder" or self._onset_free(d, p, p.onset):
                continue
            days = [x for x in range(p.first_onset, p.last_onset + 1) if x != p.onset]
            for day in [days[int(i)] for i in r.permutation(len(days))]:
                if self._onset_free(d, p, day):
                    p.onset = day
                    break
        for _ in range(AIM_STEPS):
            schedule = self.schedule(d)
            value = self.observed_success(schedule, ctx)
            if lo <= value <= hi:
                break
            harder = value > hi
            options: list[tuple[Plain, str]] = []
            for p in movable:
                if harder:
                    if p.level + 1 < len(p.labels):
                        options.append((p, "up"))
                    if (
                        p.onset > p.first_onset
                        and d.roles.get(p.fault_id) != "confounder"
                        and self._onset_free(d, p, p.onset - 1)
                    ):
                        options.append((p, "earlier"))
                else:
                    if p.level > 0:
                        options.append((p, "down"))
                    if (
                        p.onset < p.last_onset
                        and d.roles.get(p.fault_id) != "confounder"
                        and self._onset_free(d, p, p.onset + 1)
                    ):
                        options.append((p, "later"))
            options = [o for o in options if self.permitted_move(d, *o)]
            if not options:
                break
            # Levels first when easing (a later onset only moves the loss into the scoring
            # window), onsets first when hardening (an earlier onset raises the observed
            # window's loss more than the scoring window's).
            pool = options
            if self.recipe.settle_levels_first:
                preferred = ("down",) if not harder else ("earlier",)
                pool = [o for o in options if o[1] in preferred] or options
            p, move = pool[int(r.integers(len(pool)))]
            if move == "up":
                p.level += 1
            elif move == "down":
                p.level -= 1
            elif move == "earlier":
                p.onset -= 1
            else:
                p.onset += 1
        return self.schedule(d)

    # -- the whole -----------------------------------------------------------------------------

    def spec(self, schedule: Sequence[FaultSpec], patterns: Patterns) -> WorldSpec:
        from fleetlab.worlds.generate import policy_spec_nominal

        recipe = self.recipe
        return WorldSpec(
            world_id=recipe.world_id,
            seed=recipe.seed,
            size=recipe.size,
            sites=recipe.sites,
            robots_per_site=recipe.robots_per_site,
            days=recipe.days,
            windows=recipe.windows,
            fault_schedule=list(schedule),
            patterns=patterns,
            policy=policy_spec_nominal(),
            tasks=list(recipe.tasks),
            episodes_per_robot_day=recipe.episodes_per_robot_day,
        )

    def _onset_free(self, d: Draft, plain: Plain, day: int) -> bool:
        """Whether no other fault overlapping the plain fault's scope starts on `day`."""
        if day == 0:
            return True
        mine = {r for r, s in self.robots if covers(plain.scope, r, s)}
        others = [(f.target.scope, f.schedule.onset_day) for f in d.fixed]
        others += [(p.scope, p.onset) for p in d.plains if p is not plain]
        inert = {i for e, i in d.patterns.confounders if e == plain.fault_id}
        for scope, onset in others:
            if onset != day:
                continue
            if any(scope == f.target.scope and f.fault_id in inert for f in d.fixed):
                continue
            if mine & {r for r, s in self.robots if covers(scope, r, s)}:
                return False
        return True

    def co_onsets(self, schedule: Sequence[FaultSpec], patterns: Patterns) -> list[str]:
        """Pairs of faults (decoys included) that start on the same day, after day 0, at
        scopes sharing a robot, other than the designed pairs: an accidental confounder."""
        designed = {
            frozenset(p)
            for p in (*patterns.confounders, *patterns.interactions, *patterns.additive_pairs)
        }
        out = []
        for a, b in itertools.combinations(schedule, 2):
            day = a.schedule.onset_day
            if day == 0 or day != b.schedule.onset_day:
                continue
            if frozenset({a.fault_id, b.fault_id}) in designed:
                continue
            ra = {r for r, s in self.robots if covers(a.target.scope, r, s)}
            rb = {r for r, s in self.robots if covers(b.target.scope, r, s)}
            if ra & rb:
                out.append(f"{a.fault_id}+{b.fault_id}")
        return out

    def check(self, d: Draft, schedule: Sequence[FaultSpec]) -> tuple[list[str], dict[str, Any]]:
        """Why an attempt is refused (empty: accepted), and its expected numbers.

        The cheap rules on the schedule's structure come first and refuse at once; the
        estimates (observed and scoring success, then the detectability list) after them.
        """
        from fleetlab.worlds import detect

        recipe = self.recipe
        ctx = self.context(d)
        if ctx.measured is not None:
            try:
                all_episodes = self.episodes(schedule, range(recipe.days))
                estimate.measurement_weights(all_episodes, ctx)
                # Every candidate removal must have its complete measured context.
                for fault in schedule:
                    estimate.measurement_weights(all_episodes, ctx, [fault.fault_id])
            except KeyError as error:
                return [str(error)], {}
        neutral = ctx.neutral
        problems: list[str] = []
        if any(
            f.kind == "calibration_update" and f.target.scope not in self.site_of for f in schedule
        ):
            problems.append("inert calibration record lacks a benign robot-scope counterpart")
        if recipe.config_share is not None:
            counted = [f for f in schedule if f.fault_id not in d.patterns.decoys]
            config = sum(1 for f in counted if fault_class(f.kind) in CONFIG_CLASSES)
            share = config / max(1, len(counted))
            if share + 1e-9 < recipe.config_share:
                problems.append(f"configuration and software share {share:.2f}")
        # Faults at one scope each: some at a site, some at a robot (the interaction's batch
        # is a site-scoped fault too); decoys and inert entries do not count.
        acting = [f for f in schedule if f.fault_id not in neutral]
        if recipe.fleet_fault and not any(f.target.scope == FLEET_SCOPE for f in acting):
            problems.append("the requested fleet-wide fault is missing")
        if len(d.patterns.drifts) < recipe.drifts:
            problems.append("the requested drift is missing")
        sites = sum(1 for f in acting if f.target.scope.startswith("s"))
        robots = sum(1 for f in acting if f.target.scope.startswith("r"))
        if recipe.plain >= 3 and (sites < 1 or robots < 2):
            problems.append(f"plain faults at {sites} sites and {robots} robots")
        robot_scoped = {
            f.target.scope
            for f in schedule
            if f.target.scope.startswith("r") and f.fault_id not in neutral
        }
        healthy = len(self.robots) - len(robot_scoped)
        if healthy < MIN_HEALTHY_ROBOTS:
            problems.append(f"only {healthy} robots without a robot-scoped fault")
        observed = range(recipe.windows.observed.first, recipe.windows.observed.last + 1)
        clean = [
            r
            for r, s in self.robots
            if not any(
                f.fault_id not in neutral and f.fault_id not in d.small
                for day in observed
                for f in base.in_force(schedule, r, s, day)
            )
        ]
        if len(clean) < recipe.min_clean_robots:
            problems.append(f"only {len(clean)} robots free of faults in the observed window")
        clash = self.co_onsets(schedule, d.patterns)
        if clash:
            problems.append(f"faults {clash} start on the same day at overlapping scopes")
        worst = 0
        for s in self.slots:
            n = sum(
                1
                for f in base.in_force(schedule, s.robot_id, s.site_id, s.day)
                if f.fault_id not in neutral
            )
            worst = max(worst, n)
        if worst > MAX_EFFECTIVE:
            problems.append(f"an episode carries {worst} effective faults")
        for fault in schedule:
            if fault.fault_id not in d.patterns.decoys:
                continue
            label = curves.calibrated_label(fault)
            suites = {
                slot.suite
                for slot in self.slots
                if fault.schedule.onset_day <= slot.day
                and fault.target.covers(slot.robot_id, slot.site_id)
            }
            if not all(
                (point := self.curves.points.get((fault.kind, suite, label or ""))) is not None
                and point.role == "decoy"
                and point.n >= 1000
                for suite in suites
            ):
                problems.append("decoy lacks confirmed flat calibration on a carrying suite")
            if ctx.task_aware and not decoy_context_confirmed(
                fault,
                schedule,
                self.slots,
                measured=ctx.measured is not None,
                margin=self.decoy_margin(fault),
                jobs=recipe.calibration_jobs,
                scoring_days=recipe.windows.scoring,
            ):
                problems.append("decoy lacks replay-confirmed flatness with its covering faults")
        if recipe.robot_success_floor is not None:
            floor_rows = (
                estimate.robot_window_success(schedule, self.slots, ctx, recipe.windows)
                if ctx.task_aware
                else estimate.robot_day_success(schedule, self.slots, ctx)
            )
            for row in floor_rows:
                if min(float(row["true"]), float(row["logged"])) < recipe.robot_success_floor + (
                    ctx.success_margin if ctx.task_aware else 0
                ):
                    problems.append(
                        f"robot {row['robot_id']} "
                        f"{'window' if ctx.task_aware else 'day'} "
                        f"{row.get('window', row.get('day'))} below success floor "
                        f"{recipe.robot_success_floor}"
                    )
                    break
        for fault in acting:
            if (
                fault.target.scope == FLEET_SCOPE
                and ctx.measured is None
                and not all(
                    self.curves.ratio(fault, suite).effect
                    and not self.curves.ratio(fault, suite).transferred
                    for suite in {slot.suite for slot in self.slots}
                )
            ):
                problems.append("fleet-wide fault lacks a measured effect on each suite")
        for fault in schedule:
            if fault.fault_id not in d.patterns.drifts:
                continue
            drift = fault.schedule.drift
            if (
                drift is None
                or drift.shape != "linear"
                or fault.schedule.onset_day != 0
                or drift.rate <= 0
            ):
                problems.append("drift must grow linearly over the whole world")
            if recipe.require_warnable_drift:
                if (
                    estimate.drift_crossing_day(fault, self.slots, ctx)
                    not in recipe.windows.scoring
                ):
                    problems.append("drift's own crossing day is outside the scoring window")
                from fleetlab.faults.records import health_shift

                if (
                    ctx.task_aware
                    and drift_precursor(
                        fault, recipe, estimate.drift_crossing_day(fault, self.slots, ctx)
                    )
                    is None
                ):
                    problems.append("drift lacks a visible health precursor above daily noise")
                if not any(
                    value != 0
                    for value in health_shift(
                        fault.model_copy(
                            update={"level": base.day_level(fault, recipe.windows.observed.last)}
                        )
                    ).values()
                ):
                    problems.append("drift lacks a visible health precursor")
        if problems:
            return problems, {}
        value = self.observed_success(schedule, ctx)
        forecast: dict[str, Any] | None = None
        if recipe.admission == "interval":
            # Entry 20 and nothing more: the forecast's own 95% interval, with the error
            # the recipe retains, inside the band. The recipe's target guides the search
            # and refuses no draft. Asked here, before the causes, because it is cheap.
            forecast = estimate.measured_prediction(
                self.episodes(schedule, observed), ctx, band=recipe.band
            )
            interval = forecast["logged"]["prediction_interval"]
            if interval[0] < recipe.band[0] or interval[1] > recipe.band[1]:
                problems.append(f"observed prediction interval {interval} outside {recipe.band}")
        else:
            lo, hi = recipe.aim
            if ctx.task_aware:
                lo = max(lo, recipe.band[0] + ctx.success_margin)
                hi = min(hi, recipe.band[1] - ctx.success_margin)
            if not lo <= value <= hi:
                problems.append(f"expected observed success {value:.3f} outside {recipe.aim}")
        scoring_days = range(recipe.windows.scoring.first, recipe.windows.scoring.last + 1)
        scoring = estimate.mean_success(self.episodes(schedule, scoring_days), ctx, logged=True)
        if scoring < recipe.scoring_floor + (ctx.success_margin if ctx.task_aware else 0):
            problems.append(f"expected scoring success {scoring:.3f} below the floor")
        if problems:
            return problems, {}
        table = detect.detectability(
            schedule,
            d.patterns,
            d.small,
            self.slots,
            recipe.windows,
            ctx,
            members_alone=recipe.members_alone_confirm,
        )
        units = table.counted(recipe.provisional)
        need = recipe.min_detectable + recipe.margin
        observed_units = table.observed_counted(recipe.provisional)
        if ctx.task_aware:
            observed_units = [
                u for u in observed_units if u.effect["observed"] >= 2 + ctx.cause_margin_points
            ]
        if ctx.measured is None and len(observed_units) < need:
            problems.append(f"{len(observed_units)} observed detectable causes, {need} wanted")
        counted_ids = {fid for u in units for fid in u.fault_ids}
        for effective, _ in d.patterns.confounders:
            if effective not in counted_ids:
                problems.append("the confounder's effective fault is not detectable")
        for fid in d.patterns.drifts:
            if fid not in counted_ids and not recipe.require_warnable_drift:
                problems.append("the drift is not detectable")
        for a, b in d.patterns.interactions:
            if a not in counted_ids and b not in counted_ids:
                problems.append("the interaction is not detectable")
        if recipe.interaction_single_loss_points is not None:
            for pair in table.pairs:
                if pair["kind"] == "interaction" and any(
                    pair["loss_points"].get(fid, float("inf"))
                    > recipe.interaction_single_loss_points
                    for fid in pair["pair"]
                ):
                    problems.append("interaction singles cost more than the recipe allows")
        if not table.pairs_superadditive:
            problems.append("an interaction pair lacks paired super-additive support")
        if any(not p["additive"] for p in table.pairs if p["kind"] == "additive_pair"):
            problems.append("an additive pair lacks paired additive support")
        expected: dict[str, Any] = {
            "observed_logged_success": round(value, 4),
            "scoring_logged_success": round(scoring, 4),
            "detectable_units": len(units),
            "observed_detectable_units": len(observed_units),
            "worst_effective_per_episode": worst,
            "healthy_robots": healthy,
            "fault_free_robots": len(clean),
            "provisional": recipe.provisional,
        }
        if ctx.measured is not None:
            if forecast is None:
                forecast = estimate.measured_prediction(
                    self.episodes(schedule, observed), ctx, band=recipe.band
                )
                interval = forecast["logged"]["prediction_interval"]
                if interval[0] < recipe.band[0] or interval[1] > recipe.band[1]:
                    problems.append(
                        f"observed prediction interval {interval} outside {recipe.band}"
                    )
            causes = estimate.measured_causes(
                self.episodes(schedule, scoring_days),
                ctx,
                d.patterns.interactions,
                minimum=recipe.min_detectable,
            )
            if causes["probability_at_least_minimum"] < 0.95:
                problems.append(
                    f"scoring cause probability {causes['probability_at_least_minimum']:.4f}"
                    " below .95"
                )
            for fault in acting:
                if fault.target.scope == FLEET_SCOPE:
                    evidence = causes["faults"].get(fault.fault_id)
                    if evidence is None or evidence["interval_points"][0] <= 0:
                        problems.append(
                            "fleet fault lacks a positive measured task-mix removal interval"
                        )
            expected["measured_cause_forecast"] = causes
            expected["observed_forecast"] = forecast
            expected["lowest_robot_window"] = min(
                estimate.robot_window_success(schedule, self.slots, ctx, recipe.windows),
                key=lambda row: min(row["true"], row["logged"]),
            )
            if not problems:
                # What the composition forecasts, fixed here before anything runs: every
                # window with its interval under the recipe's retained error, and the
                # measured causes. The seal is the digest of exactly these numbers.
                windows = {"observed": forecast}
                for name, days in (
                    ("experiment", recipe.windows.experiment),
                    ("scoring", recipe.windows.scoring),
                ):
                    windows[name] = estimate.measured_prediction(
                        self.episodes(schedule, range(days.first, days.last + 1)),
                        ctx,
                        band=recipe.band,
                        window=name,
                    )
                expected["window_forecasts"] = windows
                expected["forecast_seal"] = forecast_seal(windows, causes)
        return problems, expected


def forecast_seal(windows: dict[str, Any], causes: dict[str, Any]) -> str:
    """The digest that seals a composition's forecast: its windows and its measured causes."""
    import hashlib
    import json

    text = json.dumps({"windows": windows, "causes": causes}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def decoy_context_confirmed(
    decoy: FaultSpec,
    schedule: Sequence[FaultSpec],
    slots: Sequence[base.Slot],
    *,
    measured: bool = False,
    margin: float | None = None,
    jobs: tuple[str, ...] | None = None,
    scoring_days: DayRange | None = None,
) -> bool:
    """Current decoy measurements are solo. Unmeasured combinations cannot be certified.

    Harmless calibration records have no mechanism; every other companion, including
    another decoy or a carried software fault, requires a joint replay measurement.
    There are no such decoy-context measurements in the current record.
    """
    if measured:
        from fleetlab.worlds.calibration import DECOY_MARGIN, context_decoy_confirmed

        return context_decoy_confirmed(
            decoy,
            schedule,
            slots,
            DECOY_MARGIN if margin is None else margin,
            jobs,
            scoring_days=scoring_days,
        )
    for slot in slots:
        active = base.in_force(schedule, slot.robot_id, slot.site_id, slot.day)
        if decoy.fault_id in {f.fault_id for f in active} and any(
            f.fault_id != decoy.fault_id and f.kind != "calibration_update" for f in active
        ):
            return False
    return True


def drift_precursor(
    fault: FaultSpec, recipe: WorldRecipe, crossing: int | None
) -> dict[str, Any] | None:
    """At least two observed readings before crossing, each shifted by >=2 noise SD.

    Use the conservative baseline for a multiplicative force loss. Include the
    metric's benign slope dispersion as well as its daily reading noise.
    """
    from fleetlab.faults.records import health_shift
    from fleetlab.worlds.health import METRICS

    if crossing is None:
        return None
    for component, metric, span, sd, (mode, slope) in METRICS:
        if component is not fault.target.component or sd <= 0:
            continue
        evidence = []
        for day in range(
            recipe.windows.observed.first, min(recipe.windows.observed.last + 1, crossing)
        ):
            shift = health_shift(fault.model_copy(update={"level": base.day_level(fault, day)}))
            size = abs(shift.get(metric, 0.0))
            if metric == "grip_force":
                size = abs(shift.get("grip_force_loss", 0.0)) * max(0, span[0] - day * slope)
            noise = math.hypot(sd, day * slope) if mode in ("normal", "down") else sd
            if size >= 2 * noise:
                evidence.append(day)
        if len(evidence) >= 2:
            return {"metric": metric, "days": evidence, "minimum_noise_sd": 2}
    return None


def with_assumed_joints(the_curves: curves.Curves, d: Draft, composer: Composer) -> curves.Curves:
    """The curves with an assumed joint point for every unmeasured interaction pair."""
    joints = list(the_curves.joints)
    seen = {(j.members, j.suite) for j in joints}
    from fleetlab.worlds.curves import JointPoint, calibrated_label

    faults = {f.fault_id: f for f in d.fixed}
    for a, b in d.patterns.interactions:
        ka = (faults[a].kind, calibrated_label(faults[a]) or "")
        kb = (faults[b].kind, calibrated_label(faults[b]) or "")
        members = (ka, kb)
        for suite in (curves.SPATIAL, curves.OBJECT):
            if the_curves.has_joint(ka, kb) or (members, suite) in seen:
                continue
            n = 100
            healthy = 98
            joints.append(
                JointPoint(
                    members=members,
                    suite=suite,
                    n=n,
                    successes=round(ASSUMED_JOINT_RATIO * healthy),
                    healthy_successes=healthy,
                    role="assumed",
                    difference=(ASSUMED_JOINT_RATIO - 1.0, 0.0, 0.0),
                    run="assumed",
                )
            )
            seen.add((members, suite))
    return curves.Curves(
        the_curves.points,
        tuple(joints),
        the_curves.healthy,
        the_curves.sources,
        the_curves.healthy_counts,
    )


class NoComposition(ValueError):
    """No attempt met every rule; `reasons` counts what refused them."""

    def __init__(self, message: str, reasons: dict[str, int]) -> None:
        super().__init__(message)
        self.reasons = dict(reasons)


def compose(recipe: WorldRecipe, the_curves: curves.Curves | None = None) -> Composition:
    """The world spec of a recipe: the first attempt that meets every rule."""
    # model_copy(update=...) is used for recipe variants and skips Pydantic
    # validation. Recheck the public-id and seed boundary before composing.
    recipe = WorldRecipe.model_validate(recipe.model_dump())
    composer = Composer(recipe, the_curves)
    if recipe.interactions:
        suites = {task.suite for task in recipe.tasks}
        supported = []
        for pair in composer._joint_pairs():
            joints = [composer.curves.joint(*pair, suite) for suite in suites]
            if all(joint.paired and not moved for joint, moved in joints):
                supported.append(pair)
        if not supported:
            raise ValueError(
                "interaction requires same-attempt paired measurements on every task suite"
            )
    if (
        recipe.estimator != "measured-v3"
        and recipe.interactions
        and recipe.robot_success_floor is not None
    ):
        best = max(
            composer.curves.healthy[suite] * joint.ratio
            for pair in supported
            for suite in {t.suite for t in recipe.tasks}
            for joint, _ in [composer.curves.joint(*pair, suite)]
        )
        if best < recipe.robot_success_floor:
            raise ValueError(
                f"measured interaction success is at most {best:.4f}, below robot-day floor "
                f"{recipe.robot_success_floor}; a milder measured pair is required"
            )
    if (
        recipe.estimator == "measured-v3"
        and recipe.interactions
        and not any(composer.pair_scope_choices(pair) for pair in supported)
    ):
        raise ValueError(
            "no interaction carrier meets task precision, paired support and robot-window floor"
        )
    placements = {
        (site, tuple(things_of(kind, label)))
        for site in composer.sites
        for kind, label in composer.decoy_choices(site, 4)
    }
    if len(placements) < recipe.decoys:
        raise ValueError("decoys have no confirmed-flat placement on the scheduled task suites")
    reasons: dict[str, int] = {}
    for attempt in range(recipe.attempts):
        try:
            d = composer.draft(attempt)
        except KeyError as error:
            reason = str(error)[:150]
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        plains = sum(1 for p in d.plains if d.roles.get(p.fault_id) == "plain")
        if d.stopped or plains < recipe.plain or len(d.patterns.decoys) < recipe.decoys:
            why = d.stopped or "draft incomplete"
            reasons[why] = reasons.get(why, 0) + 1
            continue
        if recipe.interactions and not d.patterns.interactions:
            reasons["no measured interaction"] = reasons.get("no measured interaction", 0) + 1
            continue
        if recipe.additive_pairs and not d.patterns.additive_pairs:
            reasons["no additive pair"] = reasons.get("no additive pair", 0) + 1
            continue
        if recipe.confounders and not d.patterns.confounders:
            reasons["no confounder"] = reasons.get("no confounder", 0) + 1
            continue
        try:
            schedule = composer.settle(d, attempt)
            problems, expected = composer.check(d, schedule)
        except KeyError as error:
            reason = str(error)[:150]
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if problems:
            for p in problems:
                key = re.sub(r"f[0-9]{2}|[0-9]+(\.[0-9]+)?", "N", p)[:70]
                reasons[key] = reasons.get(key, 0) + 1
            continue
        roles = dict(d.roles)
        spec = composer.spec(schedule, d.patterns)
        return identify(
            Composition(
                recipe=recipe,
                spec=spec,
                attempt=attempt,
                roles=roles,
                small=frozenset(d.small),
                assumed_joint=d.assumed_joint,
                expected=expected,
                refusals=dict(sorted(reasons.items())),
            )
        )
    raise NoComposition(
        f"{recipe.name}: no composition in {recipe.attempts} attempts; refusals: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])),
        reasons,
    )


# The twenty seeds on which round 6 asks each recipe to compose (its point 1): fixed by
# the first builder before any of them was tried, public, and never a world's seed.
ASSESSMENT_SEEDS = tuple((1 << 62) + 1100000 + k for k in range(20))


def assess(
    names: Sequence[str] = ("D2", "D3", "planned"),
    seeds: Sequence[int] = ASSESSMENT_SEEDS,
    attempts: int | None = None,
) -> dict[str, Any]:
    """On how many seeds each fourth-generation recipe composes, and what refuses the rest.

    Per recipe: the seeds accepted, and for the others how many attempts each rule
    refused (a seed refused before its first attempt counts once, under the reason
    given). CPU only; `python -m fleetlab.worlds.compose` prints it.
    """
    from fleetlab.worlds.recipe import planned_recipe

    out: dict[str, Any] = {}
    for name in names:
        rows: list[dict[str, Any]] = []
        refused: dict[str, int] = {}
        for index, seed in enumerate(seeds):
            recipe = planned_recipe(name, world_id="w99", seed=seed)  # type: ignore[arg-type]
            if attempts is not None:
                recipe = recipe.model_copy(update={"attempts": attempts})
            row: dict[str, Any] = {"seed_index": index, "accepted": False}
            try:
                comp = compose(recipe)
            except NoComposition as error:
                row["refusals"] = error.reasons
            except (ValueError, KeyError) as error:
                row["refusals"] = {str(error): 1}
                row["before_any_attempt"] = True
            else:
                forecast = comp.expected.get("observed_forecast", {}).get("logged", {})
                causes = comp.expected.get("measured_cause_forecast", {})
                faults = {f.fault_id: f for f in comp.spec.fault_schedule}
                top = sorted(
                    causes.get("faults", {}).items(),
                    key=lambda item: (-item[1]["gain_points"], item[0]),
                )[:3]
                row.update(
                    accepted=True,
                    attempt=comp.attempt,
                    observed=comp.expected.get("observed_logged_success"),
                    interval=forecast.get("prediction_interval"),
                    six_causes=causes.get("probability_at_least_minimum"),
                    forecast_causes=causes.get("count"),
                    largest_fixes=[
                        {
                            "kind": faults[fid].kind,
                            "scope": faults[fid].target.scope,
                            "gain_points": gain["gain_points"],
                        }
                        for fid, gain in top
                    ],
                )
            for reason, count in row.get("refusals", {}).items():
                refused[reason] = refused.get(reason, 0) + count
            rows.append(row)
        out[name] = {
            "seeds": len(rows),
            "accepted": sum(1 for row in rows if row["accepted"]),
            "distinct_top_two": len(
                {
                    tuple(sorted((f["kind"], f["scope"]) for f in row["largest_fixes"][:2]))
                    for row in rows
                    if row["accepted"]
                }
            ),
            "refused_attempts_by_rule": dict(sorted(refused.items(), key=lambda kv: -kv[1])),
            "rows": rows,
        }
    return out


def forecast_bound(recipe: WorldRecipe, floor: float | None = None) -> dict[str, Any]:
    """A bound from below on the observed forecast of ANY composition of a measured recipe.

    Each robot outside the decoy's site ends in the measured context that lowers its
    observed forecast most, among the contexts that are precise on every task it runs
    and keep its experiment and scoring windows at the floor with its reserve; on each
    observed attempt it counts at the lowest precise rate among that context and the
    contexts of fewer of its faults, whatever the onsets (a fault may raise success:
    the blur rescues attempts the interaction pair breaks). The four robots of the
    decoy's site all end in one context in which the decoy rule confirms a decoy on
    that site's tasks from some day on (alone, under the fleet release, or under a
    site-wide fault with a same-attempt comparison), again above the floor; a site
    without such a context cannot take the decoy. One robot (or, for a site-wide
    interaction, one site) carries the interaction pair, another robot the additive
    pair, a third the drift, counted at its lowest rate without the drift less five
    points (the drift's own harm stays under five points before the scoring window, by
    rule).

    Everything else a composition must respect only raises the forecast: one fault per
    thing, site-wide faults reaching every robot of a site, the days on which faults
    may start, the menu of fault types, the decoy being confirmed on its site. So no
    composition under these rules can forecast less; the search ends far above it.
    `floor` replaces the recipe's, to ask what another floor would allow.
    """
    composer = Composer(recipe)
    cells = composer.cells() or {}
    need = (recipe.robot_success_floor if floor is None else floor) or 0.0
    need += composer.context(Draft()).success_margin
    fleet = (
        (("camera_degradation", f"{recipe.fleet_image_blur} px blur"),)
        if recipe.fleet_fault
        else ()
    )
    by_context: dict[tuple[str, tuple[tuple[str, str], ...]], dict[int, curves.MeasuredCell]] = {}
    for (suite, task, members), cell in cells.items():
        if not any(label.startswith("dose:") for _, label in members):
            by_context.setdefault((suite, members), {})[task] = cell
    contexts = sorted({members for _, members in by_context if set(fleet) <= set(members)})
    pairs = composer._joint_pairs() if recipe.interactions else []
    decoy_kinds = (*DECOYS, ("light_color_shift", "1 warmer"))

    def rates(
        members: tuple[tuple[str, str], ...], mix: dict[tuple[str, int], int]
    ) -> tuple[float, float] | None:
        true = logged = 0.0
        for (suite, task), count in mix.items():
            cell = by_context.get((suite, members), {}).get(task)
            if cell is None or not cell.sufficient:
                return None
            true += count * cell.rate()
            logged += count * cell.rate(True)
        total = sum(mix.values())
        return true / total, logged / total

    def lowest_observed(
        members: tuple[tuple[str, str], ...], mix: dict[tuple[str, int], int]
    ) -> float:
        """Each observed attempt at the lowest precise logged rate among the context and
        the contexts of fewer of its faults (nothing at all among them)."""
        parts = [
            subset
            for size in range(len(members) + 1)
            for subset in itertools.combinations(members, size)
        ]
        total = 0.0
        for (suite, task), count in mix.items():
            found = [
                cell.rate(True)
                for subset in parts
                if (cell := by_context.get((suite, subset), {}).get(task)) is not None
                and cell.sufficient
            ]
            total += count * min(found)
        return total / sum(mix.values())

    def admitted(
        members: tuple[tuple[str, str], ...], mixes: dict[str, dict[tuple[str, int], int]]
    ) -> bool:
        """Precise on every task of the robot, and at the floor in both later windows."""
        if rates(members, mixes["observed"]) is None:
            return False
        later = [rates(members, mixes[name]) for name in ("experiment", "scoring")]
        return not any(x is None or min(x) < need for x in later)

    lowest: dict[str, dict[str, float]] = {}
    window_mix: dict[str, dict[str, dict[tuple[str, int], int]]] = {}
    for (robot, _), days in composer.robot_days().items():
        mixes: dict[str, dict[tuple[str, int], int]] = {name: {} for name in composer.windows}
        for day, tasks in enumerate(days):
            for name, span in composer.windows.items():
                if day in span:
                    for key, count in tasks.items():
                        mixes[name][key] = mixes[name].get(key, 0) + count
        window_mix[robot] = mixes
        if rates((), mixes["observed"]) is None or rates(fleet, mixes["observed"]) is None:
            raise KeyError(f"robot {robot}: no precise cells without further faults")
        plain = lowest_observed(fleet, mixes["observed"])
        row = {"free": plain, "drift": plain - 0.05}
        for members in contexts:
            if not admitted(members, mixes):
                continue
            observed = lowest_observed(members, mixes["observed"])
            row["free"] = min(row["free"], observed)
            if any(set(pair) <= set(members) for pair in pairs):
                row["pair"] = min(row.get("pair", 1.0), observed)
            if set(ADDITIVE_PAIR) <= set(members):
                row["additive"] = min(row.get("additive", 1.0), observed)
        lowest[robot] = row

    def decoy_site_total(site: str) -> float | None:
        """The lowest sum over the site's robots in a context that confirms a decoy there."""
        from fleetlab.worlds import calibration

        here = [robot for robot, x in composer.robots if x == site]
        found: float | None = None
        for kind, label in decoy_kinds:
            for assay in calibration.decoy_assays(kind, label, recipe.calibration_jobs):
                members = tuple(sorted((str(k), str(lb)) for k, lb in assay["members"]))
                if not set(fleet) <= set(members):
                    continue
                if not all(admitted(members, window_mix[robot]) for robot in here):
                    continue
                confirmed = False
                for start in range(1, recipe.windows.observed.last + 1):
                    mix: dict[tuple[str, tuple[tuple[str, str], ...]], dict[int, int]] = {}
                    for robot in here:
                        for day, tasks in enumerate(composer.robot_days()[robot, site]):
                            if day < start or day not in recipe.windows.scoring:
                                continue
                            for (suite, task), count in tasks.items():
                                cell = mix.setdefault((suite, members), {})
                                cell[task] = cell.get(task, 0) + count
                    margin = composer.decoy_margin(make_fault("f99", kind, label, site, start))
                    if calibration.decoy_flat(kind, label, mix, margin, recipe.calibration_jobs):
                        confirmed = True
                        break
                if not confirmed:
                    continue
                total = sum(lowest_observed(members, window_mix[r]["observed"]) for r in here)
                found = total if found is None else min(found, total)
        return found

    best: tuple[float, str] | None = None
    for decoy_site in composer.sites if recipe.decoys else [""]:
        clean_site = decoy_site if recipe.decoys else None
        clean_total = decoy_site_total(decoy_site) if recipe.decoys else 0.0
        if clean_total is None:
            continue
        free = [robot for robot, site in composer.robots if site != clean_site]
        total = clean_total + sum(lowest[robot]["free"] for robot in free)
        extras: list[float] = []
        carriers: list[tuple[frozenset[str], float]] = [(frozenset(), 0.0)]
        if pairs and recipe.interaction_scope == "site":
            carriers = [
                (
                    frozenset(r for r, s in composer.robots if s == site),
                    sum(
                        lowest[r]["pair"] - lowest[r]["free"]
                        for r, s in composer.robots
                        if s == site
                    ),
                )
                for site in composer.sites
                if site != clean_site
                and all("pair" in lowest[r] for r, s in composer.robots if s == site)
            ]
        elif pairs:
            carriers = [
                (frozenset({r}), lowest[r]["pair"] - lowest[r]["free"])
                for r in free
                if "pair" in lowest[r]
            ]
        for taken, cost in carriers:
            rest = [r for r in free if r not in taken]
            worn = [
                (r, lowest[r]["additive"] - lowest[r]["free"])
                for r in rest
                if "additive" in lowest[r]
            ]
            if not recipe.additive_pairs:
                worn = [("", 0.0)]
            for worn_robot, worn_cost in worn:
                drifting = [lowest[r]["drift"] - lowest[r]["free"] for r in rest if r != worn_robot]
                if recipe.drifts and not drifting:
                    continue
                extras.append(cost + worn_cost + (min(drifting) if recipe.drifts else 0.0))
        if (recipe.interactions or recipe.additive_pairs or recipe.drifts) and not extras:
            continue
        value = (total + (min(extras) if extras else 0.0)) / len(composer.robots)
        if best is None or value < best[0]:
            best = (value, decoy_site)
    if best is None:
        return {
            "bound": None,
            "reason": "no site takes a confirmed decoy and leaves the other patterns a place",
        }
    return {
        "bound": round(best[0], 4),
        "decoy_site": best[1],
        "floor_with_reserve": round(need, 4),
        "aim": list(recipe.aim),
    }


if __name__ == "__main__":
    import json
    import sys

    result = assess(sys.argv[1:] or ("D2", "D3", "planned"))
    for outcome in result.values():
        for line in outcome["rows"]:
            line.pop("refusals", None)
    print(json.dumps(result, indent=1))
