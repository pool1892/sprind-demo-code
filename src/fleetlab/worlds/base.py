"""What the version-1 composer and generator share: seeded draws, the fleet's shape, the
episode slots with each site's task emphasis, the software timeline, a fault's daily level.

Every draw comes from `rng(seed, *parts)`, a generator seeded by SHA-256 over this
module's scheme, the world seed and the draw's name, never from a global generator, so
the composer and the generator see the same draws in any process and any order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import Task
from fleetlab.contracts.ids import FLEET_SCOPE, IdKind, neutral_id, scope_covers

SCHEME = "fleetlab.worlds.v1"
GENERATOR = "fleetlab.worlds v1"

START_STATES = 50  # recorded start states per LIBERO task
IMAGE_SIZE = 360
MAX_STEPS = 280  # libero_spatial and libero_object
BATCH_SIZE = 4

# Hidden traits (contracts section 3): constants per robot that decide where faults fall.
TRAIT_MOUNT = "camera_mount_tolerance_mm"
TRAIT_PADS = "gripper_pad_age_days"
TRAIT_FIRMWARE = "controller_board_revision"
TRAIT_NAMES = (TRAIT_MOUNT, TRAIT_PADS, TRAIT_FIRMWARE)

# A site's favoured suite gets this share of its episodes, drawn per site.
EMPHASIS_RANGE = (0.6, 0.8)
TASK_CONCENTRATION = 3.0  # Dirichlet concentration of a site's weights over a suite's tasks

BASE_VERSION = "4.2.0"


def rng(seed: int, *parts: object) -> np.random.Generator:
    """A generator seeded from the world seed and the draw's name; pure."""
    text = json.dumps([SCHEME, seed, *[str(p) for p in parts]], separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def site_ids(sites: int) -> list[str]:
    return [neutral_id(IdKind.site, s + 1) for s in range(sites)]


def robot_ids(robots_per_site: int, site_index: int) -> list[str]:
    first = site_index * robots_per_site + 1
    return [neutral_id(IdKind.robot, first + i) for i in range(robots_per_site)]


def all_robots(sites: int, robots_per_site: int) -> list[tuple[str, str]]:
    """(robot, site) of every robot, site by site."""
    return [(r, s) for i, s in enumerate(site_ids(sites)) for r in robot_ids(robots_per_site, i)]


def traits_for(seed: int, robot_id: str) -> dict[str, float | int]:
    """A robot's hidden traits, drawn from the seed."""
    r = rng(seed, "traits", robot_id)
    return {
        TRAIT_MOUNT: round(float(r.uniform(0.1, 1.0)), 3),
        TRAIT_PADS: round(float(r.uniform(0.0, 300.0)), 1),
        TRAIT_FIRMWARE: int(r.integers(1, 4)),
    }


def suites_of(tasks: Sequence[Task]) -> list[str]:
    seen: list[str] = []
    for task in tasks:
        if task.suite not in seen:
            seen.append(task.suite)
    return seen


@dataclass(frozen=True)
class Emphasis:
    """A site's task emphasis: its favoured suite, that suite's share, task weights."""

    site_id: str
    favoured: str
    share: float
    weights: dict[str, tuple[float, ...]]  # suite -> weight per task of the suite, in order

    def as_json(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "favoured": self.favoured,
            "share": self.share,
            "weights": {s: list(w) for s, w in self.weights.items()},
        }


def emphasis(
    seed: int,
    sites: int,
    tasks: Sequence[Task],
    site_suites: tuple[tuple[str, str], ...] = (),
) -> list[Emphasis]:
    """Each site's task emphasis: sites alternate their favoured suite from a drawn start."""
    suites = suites_of(tasks)
    start = int(rng(seed, "emphasis-start").integers(len(suites)))
    out = []
    for i, site in enumerate(site_ids(sites)):
        r = rng(seed, "emphasis", site)
        favoured = suites[(start + i) % len(suites)]
        share = round(float(r.uniform(*EMPHASIS_RANGE)), 2) if len(suites) > 1 else 1.0
        weights = {}
        for suite in suites:
            n = sum(1 for t in tasks if t.suite == suite)
            w = r.dirichlet([TASK_CONCENTRATION] * n)
            weights[suite] = tuple(round(float(x), 3) for x in w)
        if site in dict(site_suites):
            favoured, share = dict(site_suites)[site], 1.0
        out.append(Emphasis(site, favoured, share, weights))
    return out


@dataclass(frozen=True)
class Slot:
    """One as-run episode before it has an identity."""

    robot_id: str
    site_id: str
    day: int
    index: int  # the robot's episode number within the day
    task: Task
    start_state: int

    @property
    def suite(self) -> str:
        return self.task.suite


@cache
def _slots_cached(
    seed: int,
    sites: int,
    robots_per_site: int,
    days: int,
    per_day: int,
    tasks: tuple[Task, ...],
    site_suites: tuple[tuple[str, str], ...] = (),
) -> tuple[Slot, ...]:
    suites = suites_of(tasks)
    marks = {e.site_id: e for e in emphasis(seed, sites, tasks, site_suites)}
    out = []
    for day in range(days):
        for robot, site in all_robots(sites, robots_per_site):
            mark = marks[site]
            for index in range(per_day):
                r = rng(seed, "task", robot, day, index)
                if len(suites) > 1:
                    others = [s for s in suites if s != mark.favoured]
                    suite = (
                        mark.favoured
                        if r.random() < mark.share
                        else others[int(r.integers(len(others)))]
                    )
                else:
                    suite = suites[0]
                choices = [t for t in tasks if t.suite == suite]
                weights = np.asarray(mark.weights[suite], np.float64)
                task = choices[int(r.choice(len(choices), p=weights / weights.sum()))]
                start = int(rng(seed, "start", robot, day, index).integers(START_STATES))
                out.append(Slot(robot, site, day, index, task, start))
    return tuple(out)


def slots(
    seed: int,
    sites: int,
    robots_per_site: int,
    days: int,
    per_day: int,
    tasks: Sequence[Task],
    site_suites: tuple[tuple[str, str], ...] = (),
) -> tuple[Slot, ...]:
    """Every as-run episode of the world: day by day, robot by robot, in index order."""
    return _slots_cached(seed, sites, robots_per_site, days, per_day, tuple(tasks), site_suites)


def day_level(fault: FaultSpec, day: int) -> float:
    """A fault's level on a day: its level, or for a drift its level moved by the rate.

    Linear: `level + rate * (day - onset)`. Step: `level + rate * floor((day - onset) / 2)`
    (a step every second day). Rounded to four decimals, never below zero.
    """
    drift = fault.schedule.drift
    if drift is None:
        return fault.level
    days = max(0, day - fault.schedule.onset_day)
    moved = drift.rate * (days if drift.shape == "linear" else days // 2)
    return round(max(0.0, fault.level + moved), 4)


def in_force(
    schedule: Sequence[FaultSpec], robot_id: str, site_id: str, day: int
) -> list[FaultSpec]:
    """The faults in force on a robot on a day, each at its level of that day.

    The composer keeps faults on one configuration key from overlapping in scope, and
    benign changes never touch a key with a mechanism, so no fault is shadowed here
    (the generator checks this against `contracts.faults.active_faults`).
    """
    out = []
    for fault in schedule:
        if not (
            fault.schedule.active_on(day) and scope_covers(fault.target.scope, robot_id, site_id)
        ):
            continue
        level = day_level(fault, day)
        if level <= 0.0:
            continue
        out.append(fault if level == fault.level else fault.model_copy(update={"level": level}))
    return out


def next_patch(version: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def software_timeline(
    seed: int, days: int, schedule: Sequence[FaultSpec]
) -> list[tuple[int, str, str]]:
    """The benign fleet-wide software versions: (day, old, new).

    A patch release on day 0, and a second patch on a day drawn from days 2 to 6 on which
    no fault starts (a version change on a fault's onset day would be a second, undesigned
    confounder), unless a fleet-wide release fault exists (whose version must stay the
    fleet's). Neither carries anything: a version without a release fault behind it does
    nothing.
    """
    first = next_patch(BASE_VERSION)
    out = [(0, BASE_VERSION, first)]
    fleet_release = any(
        f.kind == "software_release" and f.target.scope == FLEET_SCOPE for f in schedule
    )
    busy = {f.schedule.onset_day for f in schedule}
    free = [d for d in range(2, min(7, days)) if d not in busy]
    if not fleet_release and free:
        day = free[int(rng(seed, "software-patch").integers(len(free)))]
        out.append((day, first, next_patch(first)))
    return out


def fleet_version_on(timeline: Sequence[tuple[int, str, str]], day: int) -> str:
    version = BASE_VERSION
    for d, _, new in timeline:
        if d <= day:
            version = new
    return version


def is_fleet(scope: str) -> bool:
    return scope == FLEET_SCOPE
