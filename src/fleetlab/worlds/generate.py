"""The version-0 world generator: a seeded, minimal fleet with its history, its fault
schedule's episodes and its hidden fault table (stage 3's generator
replaces it).

`generate(spec, software)` is a pure function of the `WorldSpec` (contracts section 9)
and the software version the jobs are made for: the same arguments give the same
world, byte for byte, in any process. Every random draw comes from `rng(seed, *parts)`,
a generator seeded by SHA-256 over the world seed and the draw's name, never from a
global generator; the episodes' scene and noise seeds come from
`contracts.identity.seeds_for`.

What the generator makes from the spec:

- **The fleet** (`contracts.model.Fleet`): sites `s1..`, robots `r01..` numbered site
  by site, each with the five components (installed and last serviced on drawn days
  before day 0) and two hidden traits (`TRAIT_NAMES`), the fleet-level configuration
  holding `NOMINAL_CONFIG`, every site's conditions at their nominal values.
- **The fleet's records, under the one rule of contracts section 3**: each fault of
  the schedule whose type writes a `Change` adds one on its onset day, at its scope,
  with its true old and new value and an author drawn like a benign change's, and the
  model holds the new value at that scope from that day; benign entries come on top
  (`BENIGN_RULES`): a fleet-wide software version on day 0, a recalibration record of
  one robot's scene camera, three maintenance entries, all in the observed window and
  none that would fix or reveal a fault (no fixing verb on a faulted component, no
  calibration record on a robot with a camera fault).
- **Health readings** per robot, component and day: the two
  cameras' calibration residual in pixels, the gripper's grip force and cycle count,
  the arm's joint torque and the controller's load, each with a per-robot offset and
  day-to-day noise; a fault moves its component's reading by
  `fleetlab.faults.records.health_shift` from its onset.
- **The episodes**: `episodes_per_robot_day` per robot and day, each with a task drawn
  from the spec's tasks with the site's emphasis (`SITE_EMPHASIS`), a start state in
  `0..START_STATES-1`, the seeds of `seeds_for`, the faults in force that day
  (`contracts.faults.active_faults`), and a neutral episode id from a seeded
  permutation, so that an id says nothing about robot or day. Each becomes a "run" job
  with video.
- **The hidden fault table** (`worlds.truth.FaultTable`): per fault its role, the
  accepted directions, the episodes carrying it per window, the expected loss of fleet
  success per window from the calibrated points (`worlds.calibration`), and the
  detectability (`worlds.detect`).

The world files (`write_world`): the contracts' four hashed files (`spec.json`,
`model.json` with the traits, `description.json` as the agents receive it, cut to the
observed window, `manifest.jsonl`) and their `hash`, plus truth-side files outside the
hash: `fault_table.json`, `health.jsonl`, `task_texts.json` and `world_truth.json`
(the engine's input for experiments).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import Direction, FaultSpec, TruthTable, active_faults
from fleetlab.contracts.identity import (
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    Task,
    seeds_for,
)
from fleetlab.contracts.ids import FLEET_SCOPE, IdKind, neutral_id, scope_covers
from fleetlab.contracts.model import (
    Change,
    ChangeAuthor,
    ComponentState,
    ConfigVersion,
    Fleet,
    Maintenance,
    Robot,
    Site,
    describe_for_agents,
)
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, TargetKind, Verb
from fleetlab.contracts.world import WorldSpec, seal_world
from fleetlab.worlds import calibration
from fleetlab.worlds.truth import (
    DETECT_EPISODES,
    DETECT_POINTS,
    FAULT_TABLE_FILE,
    FaultTable,
    FaultTruth,
    Role,
    WindowName,
)

GENERATOR = "fleetlab.worlds v0"
_SCHEME = "fleetlab.worlds.v0"

START_STATES = 50  # recorded start states per LIBERO task
IMAGE_SIZE = 360
MAX_STEPS = 280  # libero_spatial and libero_object
TRAIT_NAMES = ("camera_mount_tolerance_mm", "gripper_pad_age_days")
# Odd sites lean toward the first suite of the spec's tasks, even sites toward the second,
# with this weight for the favoured suite.
SITE_EMPHASIS = 2.0 / 3.0

NOMINAL_CONFIG: dict[ConfigKey, ConfigValue] = {
    ConfigKey.policy_sampling_steps: 10,
    ConfigKey.policy_actions_executed_per_chunk: 30,
    ConfigKey.policy_image_views: 4,
    ConfigKey.policy_precision: "fp16_res32",
    ConfigKey.control_mode: "absolute",
    ConfigKey.control_action_delay_steps: 0,
    # The poses the fleet believes: position in metres, roll, pitch and yaw in degrees.
    ConfigKey.camera_scene_calibration: [0.66, 0.0, 1.61, 0.0, 33.0, 90.0],
    ConfigKey.camera_wrist_calibration: [0.0, 0.0, 0.05, 0.0, 0.0, 180.0],
    ConfigKey.software_version: "4.2.0",
}
NOMINAL_CONDITIONS: dict[ConditionKey, ConfigValue] = {
    ConditionKey.lighting: 0.0,
    ConditionKey.placement: [0.0, 0.0],
    ConditionKey.object_batch: [0.0, 0.0],
    ConditionKey.clutter: 0.0,
}

BENIGN_RULES = (
    "a fleet-wide software version change on day 0 (patch release)",
    "a recalibration record of one robot's scene camera in the observed window, never on a "
    "robot with a camera fault",
    "three maintenance entries in the observed window: cleaning a camera or the gripper, or "
    "servicing the arm or the controller, never a verb that fixes a fault on that component",
)
_MAINTENANCE_CHOICES: tuple[tuple[Component, Verb, str], ...] = (
    (Component.gripper, Verb.clean, "pads cleaned"),
    (Component.wrist_camera, Verb.clean, "lens cleaned"),
    (Component.scene_camera, Verb.clean, "lens cleaned"),
    (Component.arm, Verb.service, "scheduled service"),
    (Component.controller, Verb.service, "fan and filters"),
)

# The health model: per metric the component, the reading's
# baseline range across robots, the day-to-day noise and a daily trend. The cameras'
# residual has day-to-day noise comparable to a 3 degree offset (the residual of the
# scene camera turned 3 degrees, about 23 px), so that a reading is a clue, not a label;
# a residual is the size of an error, so the reading is the absolute value.
RESIDUAL_NOISE_PX = 20.0
HEALTH_METRICS: tuple[tuple[Component, str, tuple[float, float], float, float], ...] = (
    (Component.scene_camera, "calibration_residual", (10.0, 35.0), RESIDUAL_NOISE_PX, 0.0),
    (Component.wrist_camera, "calibration_residual", (10.0, 35.0), RESIDUAL_NOISE_PX, 0.0),
    (Component.gripper, "grip_force", (55.0, 65.0), 4.0, 0.0),
    (Component.gripper, "cycle_count", (20_000.0, 80_000.0), 0.0, 0.0),
    (Component.arm, "joint_torque_rms", (10.0, 14.0), 0.8, 0.0),
    (Component.controller, "cpu_load", (0.35, 0.6), 0.04, 0.0),
)


def rng(seed: int, *parts: object) -> np.random.Generator:
    """A generator seeded from the world seed and the draw's name; pure."""
    text = json.dumps([_SCHEME, seed, *[str(p) for p in parts]], separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def traits_for(seed: int, robot_id: str) -> dict[str, float]:
    """A robot's hidden traits: constants drawn from the seed (contracts section 3)."""
    r = rng(seed, "traits", robot_id)
    return {
        TRAIT_NAMES[0]: round(float(r.uniform(0.1, 1.0)), 3),
        TRAIT_NAMES[1]: round(float(r.uniform(0.0, 300.0)), 1),
    }


def site_ids(spec: WorldSpec) -> list[str]:
    return [neutral_id(IdKind.site, s + 1) for s in range(spec.sites)]


def robot_ids(spec: WorldSpec, site_index: int) -> list[str]:
    """The robots of the site with 0-based index `site_index`, numbered site by site."""
    first = site_index * spec.robots_per_site + 1
    return [neutral_id(IdKind.robot, first + i) for i in range(spec.robots_per_site)]


@dataclass(frozen=True)
class World:
    """A generated world: everything `write_world` stores, in memory."""

    spec: WorldSpec
    software: SoftwareVersion
    fleet: Fleet  # the model at the end of the world, traits included (truth)
    observed_fleet: Fleet  # the records as the fleet holds them when the observed window ends
    jobs: list[Job]  # one run job per episode, every window
    health: list[dict[str, Any]]  # rows of the health table, every day
    task_texts: dict[Task, str]
    fault_table: FaultTable

    def identities(self, window: WindowName | None = None) -> list[EpisodeIdentity]:
        days = None if window is None else getattr(self.spec.windows, window)
        return [j.identity for j in self.jobs if days is None or j.identity.day in days]

    def jobs_in(self, window: WindowName) -> list[Job]:
        days = getattr(self.spec.windows, window)
        return [j for j in self.jobs if j.identity.day in days]

    def health_in(self, window: WindowName) -> list[dict[str, Any]]:
        days = getattr(self.spec.windows, window)
        return [row for row in self.health if row["day"] in days]

    @property
    def task_classes(self) -> list[str]:
        return [self.task_texts[t] for t in self.spec.tasks]


# ---- the fleet and its records ------------------------------------------------------------


def _config_changes(spec: WorldSpec, fleet_robots: list[str]) -> list[Change]:
    """The changes the faults write, then the benign ones; ordered by day and scope."""
    from fleetlab.faults.records import change_entry

    authors = list(ChangeAuthor)
    changes: list[Change] = []
    for fault in spec.fault_schedule:
        author = authors[int(rng(spec.seed, "author", fault.fault_id).integers(len(authors)))]
        entry = change_entry(fault, fault.schedule.onset_day, author)
        if entry is not None:
            changes.append(entry)
    # Benign: a patch release fleet-wide on day 0.
    old = str(NOMINAL_CONFIG[ConfigKey.software_version])
    major, minor, patch = (int(x) for x in old.split("."))
    author = authors[int(rng(spec.seed, "author", "software").integers(len(authors)))]
    changes.append(
        Change(
            day=0,
            scope=FLEET_SCOPE,
            key=ConfigKey.software_version,
            old=old,
            new=f"{major}.{minor}.{patch + 1}",
            author=author,
        )
    )
    # Benign: one robot's scene camera recalibrated (the record of the believed pose).
    camera_faulted = {
        f.target.scope
        for f in spec.fault_schedule
        if f.target.component in (Component.scene_camera, Component.wrist_camera)
    }
    candidates = [r for r in fleet_robots if r not in camera_faulted]
    r = rng(spec.seed, "benign-calibration")
    robot = candidates[int(r.integers(len(candidates)))]
    day = int(r.integers(spec.windows.observed.first, spec.windows.observed.last + 1))
    believed = NOMINAL_CONFIG[ConfigKey.camera_scene_calibration]
    assert isinstance(believed, list)
    pose = [float(v) for v in believed]
    nudged = [
        round(float(v + r.normal(0.0, 0.002 if i < 3 else 0.3)), 4) for i, v in enumerate(pose)
    ]
    changes.append(
        Change(
            day=day,
            scope=robot,
            key=ConfigKey.camera_scene_calibration,
            old=pose,
            new=nudged,
            author=ChangeAuthor.auto,
        )
    )
    return sorted(changes, key=lambda c: (c.day, _scope_order(c.scope), c.key.value))


def _scope_order(scope: str) -> tuple[int, str]:
    return (0 if scope == FLEET_SCOPE else 1 if scope.startswith("s") else 2, scope)


def _maintenance(spec: WorldSpec, fleet_robots: list[str], truth: TruthTable) -> list[Maintenance]:
    r = rng(spec.seed, "maintenance")
    entries: list[Maintenance] = []
    while len(entries) < 3:
        robot = fleet_robots[int(r.integers(len(fleet_robots)))]
        component, verb, note = _MAINTENANCE_CHOICES[int(r.integers(len(_MAINTENANCE_CHOICES)))]
        day = int(r.integers(spec.windows.observed.first, spec.windows.observed.last + 1))
        fixes = any(
            f.target.component is component
            and f.target.scope == robot
            and verb in truth[f.kind].fixing_verbs
            for f in spec.fault_schedule
        )
        if fixes or any(e.robot_id == robot and e.component is component for e in entries):
            continue
        entries.append(
            Maintenance(day=day, robot_id=robot, component=component, verb=verb, note=note)
        )
    return sorted(entries, key=lambda e: (e.day, e.robot_id, e.component.value))


def nominal_config(spec: WorldSpec) -> dict[ConfigKey, ConfigValue]:
    """The fleet's configuration before any change: `NOMINAL_CONFIG`, and the keys added later
    that the spec's fleet has, at their nominal values (`WorldSpec.later_settings`)."""
    from fleetlab.contracts.vocabulary import LATER_CONFIG_NOMINAL

    return {**NOMINAL_CONFIG, **{k: LATER_CONFIG_NOMINAL[k] for k in spec.later_settings}}


def _apply_changes(
    changes: Sequence[Change],
    until_day: int,
    base: dict[ConfigKey, ConfigValue] | None = None,
) -> tuple[ConfigVersion, dict[str, ConfigVersion]]:
    """The fleet's configuration and every narrower scope's, after the changes up to a day,
    from `base` (default `NOMINAL_CONFIG`)."""
    fleet_values: dict[ConfigKey, ConfigValue] = dict(NOMINAL_CONFIG if base is None else base)
    fleet_version, fleet_since = 1, 0
    scoped: dict[str, tuple[dict[ConfigKey, ConfigValue], int, int]] = {}
    for change in changes:
        if change.day > until_day:
            continue
        if change.scope == FLEET_SCOPE:
            fleet_values[change.key] = change.new
            fleet_version += 1
            fleet_since = change.day
        else:
            values, version, _ = scoped.get(change.scope, ({}, 0, 0))
            scoped[change.scope] = ({**values, change.key: change.new}, version + 1, change.day)
    fleet_config = ConfigVersion(version=fleet_version, values=fleet_values, since_day=fleet_since)
    return fleet_config, {
        scope: ConfigVersion(version=version, values=values, since_day=since)
        for scope, (values, version, since) in scoped.items()
    }


def _health_rows(spec: WorldSpec, fleet_robots: list[str]) -> list[dict[str, Any]]:
    from fleetlab.faults.records import health_shift

    rows: list[dict[str, Any]] = []
    for robot in fleet_robots:
        base = rng(spec.seed, "health-base", robot)
        offsets = [float(base.uniform(*span)) for _, _, span, _, _ in HEALTH_METRICS]
        for day in range(spec.days):
            noise = rng(spec.seed, "health-noise", robot, day)
            for (component, metric, _, sd, trend), offset in zip(
                HEALTH_METRICS, offsets, strict=True
            ):
                value = offset + trend * day + (float(noise.normal(0.0, sd)) if sd else 0.0)
                if metric == "cycle_count":
                    value = offset + day * spec.episodes_per_robot_day
                if metric == "calibration_residual":
                    value = abs(value)  # an error's size, before any fault's shift
                for fault in spec.fault_schedule:
                    on = fault.schedule.active_on(day) and fault.target.scope == robot
                    if on and fault.target.component is component:
                        shifts = health_shift(fault)
                        if metric in shifts:  # the camera's residual, in pixels
                            value += shifts[metric]
                        if metric == "grip_force" and "grip_force_loss" in shifts:
                            value *= 1.0 - shifts["grip_force_loss"]  # a share of the force
                value = max(0.0, value)
                rows.append(
                    {
                        "day": day,
                        "robot_id": robot,
                        "component": component.value,
                        "metric": metric,
                        "value": round(value, 3 if metric == "cpu_load" else 2),
                    }
                )
    return rows


def _fleet(
    spec: WorldSpec,
    changes: Sequence[Change],
    maintenance: Sequence[Maintenance],
    health: Sequence[dict[str, Any]],
    until_day: int,
    with_traits: bool,
) -> Fleet:
    """The fleet's model as it stands at the end of `until_day` (logs cut to that day)."""
    from fleetlab.faults.records import site_notes

    fleet_config, scoped = _apply_changes(changes, until_day, nominal_config(spec))
    latest: dict[tuple[str, str], dict[str, float]] = {}
    for row in health:
        if row["day"] == until_day:
            latest.setdefault((row["robot_id"], row["component"]), {})[row["metric"]] = row["value"]
    sites: list[Site] = []
    for s, site_id in enumerate(site_ids(spec)):
        robots: list[Robot] = []
        for robot_id in robot_ids(spec, s):
            draw = rng(spec.seed, "components", robot_id)
            components: dict[Component, ComponentState] = {}
            for component in Component:
                installed = -int(draw.integers(30, 400))
                serviced: int | None = -int(draw.integers(1, 60))
                done = [
                    m.day
                    for m in maintenance
                    if m.robot_id == robot_id and m.component is component and m.day <= until_day
                ]
                if done:
                    serviced = max(done)
                components[component] = ComponentState(
                    component=component,
                    installed_day=installed,
                    last_service_day=serviced,
                    health=latest.get((robot_id, component.value), {}),
                )
            robots.append(
                Robot(
                    id=robot_id,
                    site_id=site_id,
                    components=components,
                    config=scoped.get(robot_id, ConfigVersion(version=0)),
                    traits=dict(traits_for(spec.seed, robot_id)) if with_traits else {},
                )
            )
        sites.append(
            Site(
                id=site_id,
                robots=robots,
                conditions=dict(NOMINAL_CONDITIONS),
                config=scoped.get(site_id, ConfigVersion(version=0)),
                notes=site_notes(spec.fault_schedule, site_id, [r.id for r in robots]),
            )
        )
    return Fleet(
        id=spec.world_id,
        sites=sites,
        config=fleet_config,
        change_log=[c for c in changes if c.day <= until_day],
        maintenance_log=[m for m in maintenance if m.day <= until_day],
    )


# ---- the episodes ---------------------------------------------------------------------------


def _suites(spec: WorldSpec) -> list[str]:
    seen: list[str] = []
    for task in spec.tasks:
        if task.suite not in seen:
            seen.append(task.suite)
    return seen


def draw_task(spec: WorldSpec, site_index: int, robot_id: str, day: int, index: int) -> Task:
    """An episode's task: the site's favoured suite with weight `SITE_EMPHASIS`."""
    suites = _suites(spec)
    r = rng(spec.seed, "task", robot_id, day, index)
    if len(suites) > 1:
        favoured = suites[site_index % len(suites)]
        others = [s for s in suites if s != favoured]
        suite = favoured if r.random() < SITE_EMPHASIS else others[int(r.integers(len(others)))]
    else:
        suite = suites[0]
    choices = [t for t in spec.tasks if t.suite == suite]
    return choices[int(r.integers(len(choices)))]


def _episode_numbers(spec: WorldSpec, count: int) -> list[int]:
    """A seeded permutation of 1..count: episode numbers that say nothing about the episode."""
    order = rng(spec.seed, "episode-ids").permutation(count)
    return [int(n) + 1 for n in order]


def _jobs(spec: WorldSpec, fleets: Sequence[Fleet], software: SoftwareVersion) -> list[Job]:
    """One run job per episode; `fleets[day]` is the model as it stands that day, whose
    configuration decides which wider-scoped configuration faults a robot's own setting
    shadows (`active_faults`)."""
    n = spec.episodes_per_robot_day
    slots = [
        (day, s, robot_id, index)
        for day in range(spec.days)
        for s in range(spec.sites)
        for robot_id in robot_ids(spec, s)
        for index in range(n)
    ]
    numbers = _episode_numbers(spec, len(slots))
    jobs: list[Job] = []
    for (day, s, robot_id, index), number in zip(slots, numbers, strict=True):
        task = draw_task(spec, s, robot_id, day, index)
        start = int(rng(spec.seed, "start", robot_id, day, index).integers(START_STATES))
        scene, noise = seeds_for(spec.seed, robot_id, day, index)
        identity = EpisodeIdentity(
            world_id=spec.world_id,
            episode_id=neutral_id(IdKind.episode, number),
            robot_id=robot_id,
            site_id=neutral_id(IdKind.site, s + 1),
            day=day,
            episode_index=index,
            suite=task.suite,
            task_index=task.task_index,
            start_state=start,
            scene_seed=scene,
            noise_seed=noise,
            policy=spec.policy,
            batch_size=4,
            row=None,
            image_size=IMAGE_SIZE,
            max_steps=MAX_STEPS,
            faults=active_faults(spec.fault_schedule, fleets[day], robot_id, day),
            software=software,
        )
        jobs.append(Job(identity=identity, record_video=True))
    return jobs


# ---- the hidden fault table ---------------------------------------------------------------


def calibrated_label(fault: FaultSpec) -> str:
    """The label of the calibration point a fault equals (kind, level, params, thing, nominal)."""
    import fleetlab.faults as faults

    points = getattr(faults.get(fault.kind), "calibration", ())
    thing = fault.target.thing
    for p in points:
        same = (
            p.level == fault.level
            and dict(p.params) == dict(fault.params)
            and thing is not None
            and p.target == thing.value
            and p.nominal == fault.nominal
        )
        if same:
            return str(p.label)
    raise KeyError(
        f"fault {fault.fault_id} ({fault.kind}, level {fault.level}) is no calibration point"
    )


def _fault_table(spec: WorldSpec, jobs: Sequence[Job], truth: TruthTable) -> FaultTable:
    import fleetlab.faults as faults

    windows: dict[WindowName, Any] = {
        "observed": spec.windows.observed,
        "experiment": spec.windows.experiment,
        "scoring": spec.windows.scoring,
    }
    per_window = {
        name: sum(1 for j in jobs if j.identity.day in days) for name, days in windows.items()
    }
    decoys = set(spec.patterns.decoys)
    inert = {i for _, i in spec.patterns.confounders}
    drifts = set(spec.patterns.drifts)
    rows: list[FaultTruth] = []
    for fault in spec.fault_schedule:
        point = calibration.point(fault.kind, calibrated_label(fault))
        role: Role = (
            "decoy"
            if fault.fault_id in decoys
            else "inert"
            if fault.fault_id in inert
            else "effective"
        )
        episodes = {
            name: sum(
                1 for j in jobs if j.identity.day in days and fault.fault_id in j.identity.fault_ids
            )
            for name, days in windows.items()
        }
        loss = {
            name: round(100.0 * episodes[name] * point.loss_points / 100.0 / per_window[name], 3)
            if per_window[name]
            else 0.0
            for name in windows
        }
        qualified: list[WindowName] = [
            w for w in ("observed", "scoring") if role == "effective" and loss[w] >= DETECT_POINTS
        ]
        identifiable = (
            True  # robots, sites and keys all appear in the visible data (no trait scope)
        )
        detectable = (
            role == "effective"
            and bool(qualified)
            and identifiable
            and episodes["observed"] >= DETECT_EPISODES
        )
        mapping = truth[fault.kind]
        # "drifting" only for a fault with a drift schedule.
        directions = set(mapping.directions) - {Direction.drifting}
        if fault.fault_id in drifts or fault.schedule.drift is not None:
            directions.add(Direction.drifting)
        rows.append(
            FaultTruth(
                fault_id=fault.fault_id,
                kind=fault.kind,
                fault_class=faults.get(fault.kind).fault_class,
                target=fault.target,
                role=role,
                onset_day=fault.schedule.onset_day,
                level=fault.level,
                directions=sorted(directions, key=lambda d: d.value),
                fixing_verbs=sorted(mapping.fixing_verbs, key=lambda v: v.value),
                episodes=episodes,
                expected_loss_points=loss,
                identifiable=identifiable,
                qualified_by=qualified,
                detectable=detectable,
                small=role == "effective" and not detectable,
                calibration=calibration.describe(point),
            )
        )
    rule = (
        f"detectable: role effective, expected loss of fleet success at least {DETECT_POINTS:g} "
        f"points over the observed or the scoring window from the calibrated point, target "
        f"identifiable in the visible data, at least {DETECT_EPISODES} observed episodes carry it "
        "(experiment.md section 5)"
    )
    return FaultTable(
        world_id=spec.world_id,
        windows=spec.windows,
        episodes_per_window=per_window,
        rule=rule,
        faults=rows,
    )


# ---- the whole world --------------------------------------------------------------------------


def task_texts(tasks: Sequence[Task]) -> dict[Task, str]:
    """The language of each LIBERO task (imports LIBERO's benchmark, not the simulator)."""
    import fleetlab  # noqa: F401  (torch before anything that imports mujoco)
    from fleetlab.runtime import configure_process

    configure_process()
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    out: dict[Task, str] = {}
    for task in tasks:
        out[Task(task.suite, task.task_index)] = (
            suites[task.suite]().get_task(task.task_index).language
        )
    return out


def check_spec(spec: WorldSpec, truth: TruthTable) -> None:
    """Refuse a spec the version-0 generator cannot make faithfully."""
    import fleetlab.faults as faults

    for fault in spec.fault_schedule:
        faults.REGISTRY.check(fault)
        if fault.schedule.drift is not None:
            raise ValueError("generator v0 makes no drift")
        if fault.target.kind is TargetKind.config_key and fault.target.config_key in (
            ConfigKey.software_version,
            ConfigKey.camera_scene_calibration,
        ):
            raise ValueError("generator v0 keeps software.version and the scene calibration benign")
        if fault.target.scope != FLEET_SCOPE and not any(
            scope_covers(fault.target.scope, r, s)
            for i, s in enumerate(site_ids(spec))
            for r in robot_ids(spec, i)
        ):
            raise ValueError(
                f"fault {fault.fault_id} targets {fault.target.scope}, not in the fleet"
            )
    if spec.policy.precision != NOMINAL_CONFIG[ConfigKey.policy_precision]:
        raise ValueError("the fleet's nominal configuration is X-VLA in fp16_res32")
    for kind in {f.kind for f in spec.fault_schedule}:
        if kind not in truth:
            raise ValueError(f"no truth mapping for {kind}")


def generate(
    spec: WorldSpec, software: SoftwareVersion, texts: dict[Task, str] | None = None
) -> World:
    """The world of a spec, for jobs made for `software`; a pure function of both.

    `texts` gives the tasks' language when the caller has it (tests); otherwise it is
    read from LIBERO's benchmark.
    """
    import fleetlab.faults as faults

    truth = faults.REGISTRY.truth_table()
    check_spec(spec, truth)
    robots = [r for s in range(spec.sites) for r in robot_ids(spec, s)]
    changes = _config_changes(spec, robots)
    maintenance = _maintenance(spec, robots, truth)
    health = _health_rows(spec, robots)
    last = spec.days - 1
    fleet = _fleet(spec, changes, maintenance, health, last, with_traits=True)
    observed = _fleet(
        spec, changes, maintenance, health, spec.windows.observed.last, with_traits=False
    )
    daily = [
        _fleet(spec, changes, maintenance, health, d, with_traits=False) for d in range(spec.days)
    ]
    jobs = _jobs(spec, daily, software)
    return World(
        spec=spec,
        software=software,
        fleet=fleet,
        observed_fleet=observed,
        jobs=jobs,
        health=health,
        task_texts=texts if texts is not None else task_texts(spec.tasks),
        fault_table=_fault_table(spec, jobs, truth),
    )


def _dump(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=False) + "\n"


def world_truth(world: World, base_runs: Sequence[str] = ()) -> dict[str, Any]:
    """The engine's input for the experiments of an analysis (`WorldTruth`).

    `fleet` is the model as the experiment window starts, traits dropped, and
    `first_free_episode` the first episode number the as-run world leaves free.
    """
    spec = world.spec
    return {
        "world_id": spec.world_id,
        "world_seed": spec.seed,
        "fleet": world.observed_fleet.model_dump(mode="json"),
        "windows": spec.windows.model_dump(mode="json"),
        "fault_schedule": [f.model_dump(mode="json") for f in spec.fault_schedule],
        "fault_levels": {},
        "episodes_per_robot_day": spec.episodes_per_robot_day,
        "tasks": [list(t) for t in spec.tasks],
        "task_classes": world.task_classes,
        "start_states": START_STATES,
        "policy": spec.policy.model_dump(mode="json"),
        "image_size": IMAGE_SIZE,
        "base_runs": list(base_runs),
        "first_free_episode": len(world.jobs) + 1,
    }


def write_world(world: World, directory: Path | str, base_runs: Sequence[str] = ()) -> str:
    """Write the world files and the truth-side files; return the world hash.

    The four hashed files are the contracts' (section 9); the description is the one
    the agents receive, with the logs cut to the observed window.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(world.spec.model_dump_json(indent=1) + "\n")
    (root / "model.json").write_text(world.fleet.model_dump_json(indent=1) + "\n")
    description = describe_for_agents(world.observed_fleet, True, world.task_classes)
    (root / "description.json").write_text(_dump(description))
    (root / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in world.jobs))
    value = seal_world(root)
    write_truth_files(world, root, base_runs)
    return value


def write_truth_files(world: World, directory: Path | str, base_runs: Sequence[str] = ()) -> None:
    """The truth-side files beside the hashed ones: fault table, health readings, task texts,
    the engine's `WorldTruth` and the generator's record. Derived, so rewritten freely; a
    file whose content is unchanged is left alone (another process may be reading it)."""
    from fleetlab.engine.experiments import WorldTruth

    root = Path(directory)
    texts = [
        {"suite": t.suite, "task_index": t.task_index, "text": world.task_texts[t]}
        for t in world.spec.tasks
    ]
    truth = WorldTruth.model_validate(world_truth(world, base_runs))  # the engine's own check
    files = {
        FAULT_TABLE_FILE: world.fault_table.model_dump_json(indent=1) + "\n",
        "health.jsonl": "".join(json.dumps(r) + "\n" for r in world.health),
        "task_texts.json": _dump(texts),
        "world_truth.json": truth.model_dump_json(indent=1) + "\n",
        "generator.json": _dump(
            {"generator": GENERATOR, "software": world.software.model_dump(mode="json")}
        ),
    }
    for name, text in files.items():
        path = root / name
        if not path.is_file() or path.read_text() != text:
            path.write_text(text)


def policy_spec_nominal() -> PolicySpec:
    """X-VLA's nominal spec in the fleet's precision (imports the adapter module only)."""
    from fleetlab.policies.xvla import nominal_spec

    return nominal_spec(str(NOMINAL_CONFIG[ConfigKey.policy_precision]))
