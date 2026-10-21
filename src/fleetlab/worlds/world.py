"""The version-1 world generator: a pure function of the recipe.

    world = generate(recipe, software)            # compose, then build
    value = write_world(world, directory, base_runs)

`generate` composes the fault schedule (`fleetlab.worlds.compose`) and builds the world
from the resulting `WorldSpec` (`build`), both pure: the same recipe, software and
curves give the same world, byte for byte, in any process. What `build` makes:

- **the fleet** (`contracts.model.Fleet`): sites `s1..`, robots `r01..` numbered site by
  site, each with its five components (installed and last serviced on drawn days
  before day 0) and three hidden traits (`base.TRAIT_NAMES`, which decided where the
  faults fell), the fleet-level configuration holding the nominal values, every site's
  conditions as the fleet believes them (the simulator's cell; a site fault changes the
  real condition and leaves the believed value);
- **the records, under the one rule of contracts section 3**: each configuration or
  software fault writes its `Change` on its onset day at its scope, with its true old
  and new value and an author drawn like a benign change's; benign entries come on top
  and are never a giveaway: the fleet's software patches (`base.software_timeline`),
  calibration records of cameras that carry no pose fault, and maintenance entries
  (cleaning cameras and grippers, servicing arms and controllers, recalibrating arms),
  about one per robot, never a verb that would fix a fault of that robot's component;
- **the health checks** (`fleetlab.worlds.health`), daily, with noise and benign drift;
- **the episodes**: per robot and day `episodes_per_robot_day` slots (`base.slots`:
  the site's task emphasis, a start state, the seeds of `seeds_for`), the faults in
  force that day at that day's level (a drift moves), checked against
  `contracts.faults.active_faults` on the day's model, and a neutral episode id from a
  seeded permutation; each a "run" job with video;
- **the truth**: the hidden fault table in the scorer's schema (`worlds.truth`), the
  detectability list (`worlds.detect`) with the expected
  success per window, and the engine's `WorldTruth` (drift levels per day included).

The world files: the contracts' four hashed files (`spec.json`, `model.json` with the
traits, `description.json` as the agents receive it, cut to the observed window,
`manifest.jsonl`) and their `hash`, plus the truth side outside the hash
(`fault_table.json`, `detectability.json`, `health.jsonl`, `task_texts.json`,
`world_truth.json`, `generator.json`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.faults import Direction, active_faults
from fleetlab.contracts.identity import EpisodeIdentity, SoftwareVersion, Task, seeds_for
from fleetlab.contracts.ids import FLEET_SCOPE, IdKind, neutral_id
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
from fleetlab.contracts.vocabulary import Component, ConfigKey, Verb
from fleetlab.contracts.world import WORLD_FILES, WorldSpec, seal_world
from fleetlab.worlds import base, curves, detect, estimate
from fleetlab.worlds import health as health_model
from fleetlab.worlds.compose import (
    Composition,
    calibration_value,
    compose,
    rename_fault_ids,
    with_assumed_joints,
)
from fleetlab.worlds.generate import (
    NOMINAL_CONDITIONS,
    NOMINAL_CONFIG,
    World,
    _apply_changes,
    nominal_config,
    task_texts,
)
from fleetlab.worlds.recipe import WorldRecipe
from fleetlab.worlds.truth import (
    FAULT_TABLE_FILE,
    FaultTable,
    FaultTruth,
    Role,
    WindowName,
)

DETECTABILITY_FILE = "detectability.json"
WINDOWS: tuple[WindowName, ...] = ("observed", "experiment", "scoring")

# Benign maintenance: (component, verb, note); none fixes a fault of a component that
# carries one on that robot (checked against the fault types' fixing verbs).
MAINTENANCE_CHOICES: tuple[tuple[Component, Verb, str], ...] = (
    (Component.gripper, Verb.clean, "pads cleaned"),
    (Component.wrist_camera, Verb.clean, "lens cleaned"),
    (Component.scene_camera, Verb.clean, "lens cleaned"),
    (Component.arm, Verb.service, "scheduled service"),
    (Component.controller, Verb.service, "fan and filters"),
    (Component.arm, Verb.recalibrate, "joint offsets checked"),
)
MAINTENANCE_PER_ROBOT = 0.8  # mean entries per robot over the world's days
CALIBRATION_RECORDS = (2, 4)  # benign calibration records per world, at least and at most


@dataclass(frozen=True)
class WorldV1(World):
    """A version-1 world: the version-0 `World` (which the engine, the oracle and the
    integration read) with the recipe, the composition and the detectability list."""

    recipe: WorldRecipe
    composition: Composition
    detect: detect.DetectTable
    emphasis: tuple[base.Emphasis, ...]
    curves_sources: tuple[str, ...]
    calibration: curves.Curves

    @cached_property
    def predictions(self) -> dict[str, Any]:
        ctx = context(self.composition, self.calibration)
        return {
            name: estimate.prediction(
                [estimate.Episode.of(job.identity) for job in self.jobs_in(name)],
                ctx,
                draws=2048 if ctx.measured is not None else 512,
                window=name,
            )
            for name in WINDOWS
        }


# ---- the records -----------------------------------------------------------------------------


def _author(seed: int, *parts: object) -> ChangeAuthor:
    authors = list(ChangeAuthor)
    return authors[int(base.rng(seed, "author", *parts).integers(len(authors)))]


def changes(spec: WorldSpec) -> list[Change]:
    """The change log: the faults' changes, the software patches, the calibration records."""
    from fleetlab.faults.records import change_entry

    out: list[Change] = []
    for fault in spec.fault_schedule:
        author = (
            _author(
                spec.seed,
                "calibration",
                fault.target.scope,
                fault.schedule.onset_day,
                fault.target.config_key,
            )
            if fault.kind == "calibration_update"
            else _author(spec.seed, fault.fault_id)
        )
        entry = change_entry(fault, fault.schedule.onset_day, author)
        if entry is not None:
            out.append(entry)
    for day, old, new in base.software_timeline(spec.seed, spec.days, spec.fault_schedule):
        out.append(
            Change(
                day=day,
                scope=FLEET_SCOPE,
                key=ConfigKey.software_version,
                old=old,
                new=new,
                author=_author(spec.seed, "software", day),
            )
        )
    out += _calibration_records(spec)
    return sorted(out, key=lambda c: (c.day, _scope_order(c.scope), c.key.value, str(c.new)))


def _scope_order(scope: str) -> tuple[int, str]:
    return (0 if scope == FLEET_SCOPE else 1 if scope.startswith("s") else 2, scope)


def _calibration_records(spec: WorldSpec) -> list[Change]:
    """Benign records of a camera's believed pose being renewed, on cameras
    that carry no pose fault at any time."""
    r = base.rng(spec.seed, "calibration-records")
    posed = {
        (f.target.scope, f.target.component)
        for f in spec.fault_schedule
        if f.kind in ("camera_pose_offset", "camera_degradation")
    }
    keys = {
        Component.scene_camera: ConfigKey.camera_scene_calibration,
        Component.wrist_camera: ConfigKey.camera_wrist_calibration,
    }
    from fleetlab.contracts.ids import scope_covers

    robots = [rid for rid, _ in base.all_robots(spec.sites, spec.robots_per_site)]
    for fault in spec.fault_schedule:
        if fault.kind == "calibration_update":
            for robot, site in base.all_robots(spec.sites, spec.robots_per_site):
                if scope_covers(fault.target.scope, robot, site):
                    posed.update(
                        (robot, camera)
                        for camera, key in keys.items()
                        if key is fault.target.config_key
                    )
    count = int(r.integers(CALIBRATION_RECORDS[0], CALIBRATION_RECORDS[1] + 1))
    out: list[Change] = []
    believed: dict[tuple[str, ConfigKey], list[float]] = {}
    for _ in range(count * 4):
        if len(out) >= count:
            break
        robot = robots[int(r.integers(len(robots)))]
        camera = list(keys)[int(r.integers(len(keys)))]
        day = int(r.integers(0, spec.days))
        taken = any(c.scope == robot and c.day == day for c in out)
        if (robot, camera) in posed or taken or day in _onsets(spec, robot):
            continue
        key = keys[camera]
        nominal = NOMINAL_CONFIG[key]
        assert isinstance(nominal, list)
        prior = believed.get((robot, key), [float(v) for v in nominal])
        new = calibration_value(spec.seed, robot, day, key.value, prior)
        believed[(robot, key)] = new
        out.append(
            Change(
                day=day,
                scope=robot,
                key=key,
                old=prior,
                new=new,
                author=_author(spec.seed, "calibration", robot, day, key),
            )
        )
    # A later record of the same camera starts from the earlier one's pose.
    out.sort(key=lambda c: (c.day, c.scope))
    poses: dict[tuple[str, ConfigKey], ConfigValue] = {}
    fixed = []
    for c in out:
        old: ConfigValue = poses.get((c.scope, c.key), NOMINAL_CONFIG[c.key])
        assert isinstance(old, list)
        new = calibration_value(spec.seed, c.scope, c.day, c.key.value, [float(v) for v in old])
        fixed.append(c.model_copy(update={"old": old, "new": new}))
        poses[(c.scope, c.key)] = new
    return fixed


def _onsets(spec: WorldSpec, robot: str) -> set[int]:
    """The days on which a fault covering the robot starts (no benign record lands there:
    it would be a second, undesigned confounder)."""
    from fleetlab.contracts.ids import scope_covers

    site = dict(base.all_robots(spec.sites, spec.robots_per_site))[robot]
    return {
        f.schedule.onset_day
        for f in spec.fault_schedule
        if f.schedule.onset_day > 0 and scope_covers(f.target.scope, robot, site)
    }


def maintenance(spec: WorldSpec) -> list[Maintenance]:
    """Benign maintenance entries, about `MAINTENANCE_PER_ROBOT` per robot."""
    import fleetlab.faults as faults

    truth = faults.REGISTRY.truth_table()
    robots = [rid for rid, _ in base.all_robots(spec.sites, spec.robots_per_site)]
    entries: list[Maintenance] = []
    for robot in robots:
        r = base.rng(spec.seed, "maintenance", robot)
        for _ in range(int(r.poisson(MAINTENANCE_PER_ROBOT))):
            component, verb, note = MAINTENANCE_CHOICES[int(r.integers(len(MAINTENANCE_CHOICES)))]
            day = int(r.integers(0, spec.days))
            fixes = any(
                f.target.component is component
                and f.target.scope == robot
                and verb in truth[f.kind].fixing_verbs
                for f in spec.fault_schedule
            )
            taken = any(e.robot_id == robot and e.day == day for e in entries)
            if fixes or taken or day in _onsets(spec, robot):
                continue
            entries.append(
                Maintenance(day=day, robot_id=robot, component=component, verb=verb, note=note)
            )
    return sorted(entries, key=lambda e: (e.day, e.robot_id, e.component.value))


def _fleet(
    spec: WorldSpec,
    log: Sequence[Change],
    upkeep: Sequence[Maintenance],
    readings: Sequence[dict[str, Any]],
    until_day: int,
    with_traits: bool,
    extended_conditions: bool = False,
    public_fleet_id: str | None = None,
) -> Fleet:
    """The fleet's model as it stands at the end of `until_day`, logs cut to that day."""
    from fleetlab.faults.records import site_notes

    fleet_config, scoped = _apply_changes(log, until_day, nominal_config(spec))
    latest: dict[tuple[str, str], dict[str, float]] = {}
    for row in readings:
        if row["day"] == until_day:
            latest.setdefault((row["robot_id"], row["component"]), {})[row["metric"]] = row["value"]
    conditions = dict(NOMINAL_CONDITIONS)
    if extended_conditions:
        # The extension's vector-valued conditions are exposed uniformly at all
        # sites, independent of which site carries a fault of that type.
        from fleetlab.contracts.vocabulary import ConditionKey

        conditions[ConditionKey.placement] = [0.0, 0.0, 0.0]
        conditions[ConditionKey.lighting] = [0.0, 0.0]
    sites: list[Site] = []
    for s, site_id in enumerate(base.site_ids(spec.sites)):
        robots: list[Robot] = []
        for robot_id in base.robot_ids(spec.robots_per_site, s):
            draw = base.rng(spec.seed, "components", robot_id)
            components: dict[Component, ComponentState] = {}
            for component in Component:
                installed = -int(draw.integers(30, 600))
                serviced: int | None = -int(draw.integers(1, 90))
                done = [
                    m.day
                    for m in upkeep
                    if m.robot_id == robot_id
                    and m.component is component
                    and m.day <= until_day
                    and m.verb in (Verb.service, Verb.replace_)
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
                    traits=dict(base.traits_for(spec.seed, robot_id)) if with_traits else {},
                )
            )
        sites.append(
            Site(
                id=site_id,
                robots=robots,
                conditions=dict(conditions),
                config=scoped.get(site_id, ConfigVersion(version=0)),
                notes=site_notes(spec.fault_schedule, site_id, [r.id for r in robots]),
            )
        )
    return Fleet(
        id=public_fleet_id or spec.world_id,
        sites=sites,
        config=fleet_config,
        change_log=[c for c in log if c.day <= until_day],
        maintenance_log=[m for m in upkeep if m.day <= until_day],
    )


# ---- the episodes ----------------------------------------------------------------------------


def jobs(
    spec: WorldSpec,
    daily: Sequence[Fleet],
    software: SoftwareVersion,
    slot_plan: Sequence[base.Slot] | None = None,
) -> list[Job]:
    """One run job per as-run episode, its faults at the day's level."""
    slots = (
        slot_plan
        if slot_plan is not None
        else base.slots(
            spec.seed,
            spec.sites,
            spec.robots_per_site,
            spec.days,
            spec.episodes_per_robot_day,
            spec.tasks,
        )
    )
    order = base.rng(spec.seed, "episode-ids").permutation(len(slots))
    out: list[Job] = []
    schedules = {
        day: [f.model_copy(update={"level": base.day_level(f, day)}) for f in spec.fault_schedule]
        for day in range(spec.days)
    }
    for slot, number in zip(slots, order, strict=True):
        faults = base.in_force(spec.fault_schedule, slot.robot_id, slot.site_id, slot.day)
        contract = active_faults(schedules[slot.day], daily[slot.day], slot.robot_id, slot.day)
        if {f.fault_id for f in faults} != {f.fault_id for f in contract}:
            raise ValueError(
                f"{slot.robot_id} day {slot.day}: faults {sorted(f.fault_id for f in faults)} "
                f"against the contracts' {sorted(f.fault_id for f in contract)} (a shadowed fault)"
            )
        scene, noise = seeds_for(spec.seed, slot.robot_id, slot.day, slot.index)
        identity = EpisodeIdentity(
            world_id=spec.world_id,
            episode_id=neutral_id(IdKind.episode, int(number) + 1),
            robot_id=slot.robot_id,
            site_id=slot.site_id,
            day=slot.day,
            episode_index=slot.index,
            suite=slot.task.suite,
            task_index=slot.task.task_index,
            start_state=slot.start_state,
            scene_seed=scene,
            noise_seed=noise,
            policy=spec.policy,
            batch_size=base.BATCH_SIZE,
            row=None,
            image_size=base.IMAGE_SIZE,
            max_steps=base.MAX_STEPS,
            faults=faults,
            software=software,
        )
        out.append(Job(identity=identity, record_video=True))
    return out


def episode_slots(all_jobs: Sequence[Job]) -> tuple[base.Slot, ...]:
    """Use the actual sealed task schedule for drift and robot-day accounting."""
    return tuple(
        base.Slot(
            i.robot_id,
            i.site_id,
            i.day,
            i.episode_index,
            Task(i.suite, i.task_index),
            i.start_state,
        )
        for job in all_jobs
        for i in [job.identity]
    )


# ---- the truth -------------------------------------------------------------------------------


def context(comp: Composition, the_curves: curves.Curves) -> estimate.Context:
    spec = comp.spec
    use = the_curves
    if comp.assumed_joint:
        from fleetlab.worlds.compose import Composer, Draft

        draft = Draft(fixed=list(spec.fault_schedule), patterns=spec.patterns)
        use = with_assumed_joints(the_curves, draft, Composer(comp.recipe, the_curves))
    return estimate.Context(
        curves=use,
        neutral=frozenset(spec.patterns.excluded_from_replay()),
        small=comp.small,
        pairs=tuple([*spec.patterns.interactions, *spec.patterns.additive_pairs]),
        task_aware=comp.recipe.estimator != "suite-v1",
        measured=(
            curves.measured_cells()
            if comp.recipe.calibration_jobs is None
            else curves.measured_cells_of(comp.recipe.calibration_jobs)
        )
        if comp.recipe.estimator == "measured-v3"
        else None,
        retained_error=comp.recipe.retained_error,
    )


def fault_table(
    spec: WorldSpec,
    all_jobs: Sequence[Job],
    table: detect.DetectTable,
    ctx: estimate.Context,
    by_window: dict[str, list[estimate.Episode]],
) -> FaultTable:
    """The hidden fault table in the scorer's schema: one row per fault of the schedule.

    A single fault's expected loss and detectability are its own (removing it alone);
    the detectability of an interaction as a whole is in `detectability.json`.
    """
    import fleetlab.faults as faults

    truth = faults.REGISTRY.truth_table()
    per_window = {w: len(by_window[w]) for w in WINDOWS}
    decoys = set(spec.patterns.decoys)
    inert = {i for _, i in spec.patterns.confounders}
    rows: list[FaultTruth] = []
    sources = {u.fault_ids[0]: u for u in table.units if len(u.fault_ids) == 1}
    slots = episode_slots(all_jobs)
    for fault in spec.fault_schedule:
        fid = fault.fault_id
        role: Role = "decoy" if fid in decoys else "inert" if fid in inert else "effective"
        unit = sources.get(fid)
        episodes = {w: estimate.carriers(by_window[w], [fid]) for w in WINDOWS}
        loss = {w: round(unit.effect[w], 3) if unit else 0.0 for w in WINDOWS}
        qualified: list[WindowName] = [
            w for w in ("observed", "scoring") if unit is not None and w in unit.qualified_by
        ]
        detectable = unit is not None and unit.detectable
        mapping = truth[fault.kind]
        directions = set(mapping.directions) - {Direction.drifting}
        if fault.schedule.drift is not None:
            directions.add(Direction.drifting)
        unit = sources.get(fid)
        if unit is not None:
            note = "; ".join(unit.sources)
        else:
            note = f"{role}: neutrality is suite-specific; see detectability.json"
        rows.append(
            FaultTruth(
                fault_id=fid,
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
                identifiable=True,
                qualified_by=qualified,
                detectable=detectable,
                small=role == "effective" and not detectable,
                calibration=note,
                drift_crossing_day=estimate.drift_crossing_day(fault, slots, ctx),
            )
        )
    return FaultTable(
        world_id=spec.world_id,
        windows=spec.windows,
        episodes_per_window=per_window,
        rule=detect.RULE + " (this table: each fault alone; causes in detectability.json)",
        faults=rows,
    )


# ---- the whole world -------------------------------------------------------------------------


def generate(
    recipe: WorldRecipe,
    software: SoftwareVersion,
    texts: dict[Task, str] | None = None,
    the_curves: curves.Curves | None = None,
) -> WorldV1:
    """The world of a recipe for jobs made for `software`; pure."""
    use = the_curves or curves.load()
    world = build(compose(recipe, use), software, texts, use)
    if recipe.sealed_hash is not None and world_hash(world) != recipe.sealed_hash:
        raise ValueError("recipe no longer composes to its sealed hash")
    return world


def recompose_record(record: dict[str, Any], texts: dict[Task, str]) -> WorldV1:
    """Recompose a stored contract without writing to or resealing its record.

    The two round-6c records accidentally saved a whole-record digest instead of
    their recipe's named-job digest. Recognize only those exact published recipes
    and erroneous digests; every other mismatch still fails closed. The original
    world hash and forecast seal must both reproduce after this metadata repair.
    """
    from fleetlab.worlds.recipe import DEVELOPMENT

    recipe = WorldRecipe.model_validate(record["recipe"])
    old = {
        "w48": "task_refinement sha256 3372111d1251ec11",
        "w49": "task_refinement sha256 fac0ac6c160805cf",
    }
    canonical = DEVELOPMENT.get(recipe.world_id)
    if canonical is not None and recipe.world_id in old and recipe.calibration_sources:
        corrected = recipe.model_copy(update={"calibration_sources": canonical.calibration_sources})
        if (
            corrected == canonical
            and recipe.calibration_sources
            == (*canonical.calibration_sources[:-1], old[recipe.world_id])  # type: ignore[index]
        ):
            recipe = corrected
    result = generate(recipe, SoftwareVersion.model_validate(record["software"]), texts)
    seal = record.get("expected", {}).get("forecast_seal")
    if seal is not None and result.composition.expected.get("forecast_seal") != seal:
        raise ValueError("stored forecast seal does not recompose")
    return result


def assert_visible_ids_absent(world: WorldV1, handles: set[str]) -> None:
    """Both opaque identities and private counted draw handles stay out of visible text."""
    import re

    text = json.dumps(describe_for_agents(world.fleet, True, world.task_classes))
    text += json.dumps(world.health)
    if any(
        re.search(r"(?<![a-zA-Z0-9])" + re.escape(h) + r"(?![a-zA-Z0-9])", text) for h in handles
    ):
        raise ValueError("private fault identity leaked into visible world text")


def build(
    comp: Composition,
    software: SoftwareVersion,
    texts: dict[Task, str] | None = None,
    the_curves: curves.Curves | None = None,
) -> WorldV1:
    """The world of a composed spec."""
    import fleetlab.faults as faults

    use = the_curves or curves.load()
    if comp.fault_id_map:
        # Assemble against private draw handles first. They seed change-log authors
        # and order the truth calculations; changing identities must not redraw the
        # fleet or alter floating-point evaluation order. Only final references change.
        original = replace(comp, fault_id_map={})
        original = rename_fault_ids(original, {v: k for k, v in comp.fault_id_map.items()})
        world = build(original, software, texts, use)
        assert_visible_ids_absent(world, {*comp.fault_id_map, *comp.fault_id_map.values()})
        return replace(
            world,
            spec=comp.spec,
            composition=comp,
            jobs=rename_fault_ids(world.jobs, comp.fault_id_map),
            fault_table=rename_fault_ids(world.fault_table, comp.fault_id_map),
            detect=rename_fault_ids(world.detect, comp.fault_id_map),
        )
    spec = comp.spec
    for fault in spec.fault_schedule:
        faults.REGISTRY.check(fault)
    robots = base.all_robots(spec.sites, spec.robots_per_site)
    log = changes(spec)
    upkeep = maintenance(spec)
    readings = health_model.rows(
        spec.seed, robots, spec.days, spec.episodes_per_robot_day, spec.fault_schedule, log, upkeep
    )
    last = spec.days - 1
    fleet = _fleet(
        spec,
        log,
        upkeep,
        readings,
        last,
        with_traits=True,
        extended_conditions=comp.recipe.extension_types,
        public_fleet_id=comp.recipe.public_fleet_id,
    )
    observed = _fleet(
        spec,
        log,
        upkeep,
        readings,
        spec.windows.observed.last,
        with_traits=False,
        extended_conditions=comp.recipe.extension_types,
        public_fleet_id=comp.recipe.public_fleet_id,
    )
    daily = [
        _fleet(
            spec,
            log,
            upkeep,
            readings,
            d,
            with_traits=False,
            extended_conditions=comp.recipe.extension_types,
            public_fleet_id=comp.recipe.public_fleet_id,
        )
        for d in range(spec.days)
    ]
    slot_plan = base.slots(
        spec.seed,
        spec.sites,
        spec.robots_per_site,
        spec.days,
        spec.episodes_per_robot_day,
        spec.tasks,
        comp.recipe.site_suites,
    )
    all_jobs = jobs(spec, daily, software, slot_plan)
    ctx = context(comp, use)
    by_window: dict[str, list[estimate.Episode]] = {w: [] for w in WINDOWS}
    ranges = {
        "observed": spec.windows.observed,
        "experiment": spec.windows.experiment,
        "scoring": spec.windows.scoring,
    }
    for job in all_jobs:
        for name, days in ranges.items():
            if job.identity.day in days:
                by_window[name].append(estimate.Episode.of(job.identity))
    slots = episode_slots(all_jobs)
    table = detect.detectability(
        spec.fault_schedule,
        spec.patterns,
        comp.small,
        slots,
        spec.windows,
        ctx,
        roles=comp.roles,
        by_window=by_window,
        members_alone=comp.recipe.members_alone_confirm,
    )
    return WorldV1(
        spec=spec,
        software=software,
        fleet=fleet,
        observed_fleet=observed,
        jobs=all_jobs,
        health=readings,
        task_texts=texts if texts is not None else task_texts(spec.tasks),
        fault_table=fault_table(spec, all_jobs, table, ctx, by_window),
        recipe=comp.recipe,
        composition=comp,
        detect=table,
        emphasis=tuple(base.emphasis(spec.seed, spec.sites, spec.tasks, comp.recipe.site_suites)),
        curves_sources=curves.measured_sources(comp.recipe.calibration_jobs)
        if comp.recipe.estimator == "measured-v3"
        else use.sources,
        calibration=use,
    )


def _dump(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=False) + "\n"


def drift_levels(world: World) -> dict[str, dict[int, float]]:
    """Each drifting fault's level per day it is in force (the engine's `fault_levels`)."""
    out: dict[str, dict[int, float]] = {}
    for fault in world.spec.fault_schedule:
        if fault.schedule.drift is None:
            continue
        out[fault.fault_id] = {
            day: base.day_level(fault, day)
            for day in range(world.spec.days)
            if fault.schedule.active_on(day)
        }
    return out


def world_truth(world: WorldV1, base_runs: Sequence[str] = ()) -> dict[str, Any]:
    """The engine's input for the experiments of an analysis (`WorldTruth`)."""
    spec = world.spec
    return {
        "world_id": spec.world_id,
        "world_seed": spec.seed,
        "fleet": world.observed_fleet.model_dump(mode="json"),
        "windows": spec.windows.model_dump(mode="json"),
        "fault_schedule": [f.model_dump(mode="json") for f in spec.fault_schedule],
        "fault_levels": {
            k: {str(d): v for d, v in levels.items()} for k, levels in drift_levels(world).items()
        },
        "episodes_per_robot_day": spec.episodes_per_robot_day,
        "tasks": [list(t) for t in spec.tasks],
        **(
            {
                "robot_tasks": {
                    robot: [
                        list(t)
                        for t in spec.tasks
                        if t.suite == dict(world.recipe.site_suites)[site]
                    ]
                    for robot, site in base.all_robots(spec.sites, spec.robots_per_site)
                    if site in dict(world.recipe.site_suites)
                }
            }
            if world.recipe.site_suites
            else {}
        ),
        "task_classes": world.task_classes,
        "start_states": base.START_STATES,
        "policy": spec.policy.model_dump(mode="json"),
        "image_size": base.IMAGE_SIZE,
        "max_steps": {suite: base.MAX_STEPS for suite in base.suites_of(spec.tasks)},
        "base_runs": list(base_runs),
        "first_free_episode": len(world.jobs) + 1,
    }


def generator_record(world: WorldV1) -> dict[str, Any]:
    from fleetlab.worlds.summary import composition_report

    comp = world.composition
    return {
        "generator": base.GENERATOR,
        "recipe": {
            **world.recipe.model_dump(mode="json"),
            "sealed_hash": world_hash(world),
            "calibration_sources": list(world.curves_sources),
        },
        "attempt": comp.attempt,
        "roles": comp.roles,
        "small_causes": sorted(comp.small),
        "assumed_joint": comp.assumed_joint,
        "provisional": world.recipe.provisional,
        "expected": comp.expected,
        "predictions": world.predictions,
        "predictions_sha256": hashlib.sha256(_dump(world.predictions).encode()).hexdigest(),
        "composition": composition_report(world),
        "refusals_before_acceptance": comp.refusals,
        "emphasis": [e.as_json() for e in world.emphasis],
        "curves": list(world.curves_sources),
        "software": world.software.model_dump(mode="json"),
        "traits": {r.id: dict(r.traits) for r in world.fleet.robots()},
    }


def write_world(world: WorldV1, directory: Path | str, base_runs: Sequence[str] = ()) -> str:
    """Write the world files and the truth-side files; return the world hash."""
    files = serialized_world(world)
    predicted_hash = world_hash(world)
    if world.recipe.sealed_hash is not None and predicted_hash != world.recipe.sealed_hash:
        raise ValueError(f"recipe no longer composes to its sealed hash: {predicted_hash}")
    root = Path(directory)
    if world.recipe.kind in ("test", "reserve"):
        from fleetlab.worlds.recipe import private_path

        private_path(root)
        if root.is_dir():
            for entry in root.iterdir():
                private_path(entry)
    if (root / "hash").is_file() and (root / "hash").read_text().strip() != predicted_hash:
        raise FileExistsError("refusing to replace a different sealed world")
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    value = seal_world(root)
    write_truth_files(world, root, base_runs)
    return value


def serialized_world(world: WorldV1) -> dict[str, bytes]:
    """The exact sealed payload, also usable before writing anything."""
    description = describe_for_agents(world.observed_fleet, True, world.task_classes)
    return {
        name: text.encode()
        for name, text in {
            "spec.json": world.spec.model_dump_json(indent=1) + "\n",
            "model.json": world.fleet.model_dump_json(indent=1) + "\n",
            "description.json": _dump(description),
            "manifest.jsonl": "".join(j.model_dump_json() + "\n" for j in world.jobs),
        }.items()
    }


def world_hash(world: WorldV1) -> str:
    files = serialized_world(world)
    digest = hashlib.sha256()
    for name in WORLD_FILES:
        data = files[name]
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def write_truth_files(world: WorldV1, directory: Path | str, base_runs: Sequence[str] = ()) -> None:
    """The truth-side files beside the hashed ones; a file whose content is unchanged is
    left alone (another process may be reading it)."""
    from fleetlab.engine.experiments import WorldTruth

    root = Path(directory)
    texts = [
        {"suite": t.suite, "task_index": t.task_index, "text": world.task_texts[t]}
        for t in world.spec.tasks
    ]
    truth = WorldTruth.model_validate(world_truth(world, base_runs))
    files = {
        FAULT_TABLE_FILE: world.fault_table.model_dump_json(indent=1) + "\n",
        DETECTABILITY_FILE: _dump(world.detect.as_json()),
        "health.jsonl": "".join(json.dumps(r) + "\n" for r in world.health),
        "task_texts.json": _dump(texts),
        "world_truth.json": truth.model_dump_json(indent=1) + "\n",
        "generator.json": _dump(generator_record(world)),
    }
    for name, text in files.items():
        path = root / name
        if not path.is_file() or path.read_text() != text:
            path.write_text(text)
