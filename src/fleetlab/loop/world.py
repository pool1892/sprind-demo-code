"""A stored world as the loop reads it (truth side; never mounted into a sandbox).

    world = LoopWorld.load("~/.cache/robots-probes/worlds/w62")
    state = world.state_on("r05", 8, ledger)            # faults and values under the ledger
    identities = world.regular_identities([8, 9], software, ledger)
    panel = world.panel_identities([12, 13], software)  # the reserved evaluation panel

What it reads: the world files (`files/` or the miniature fleet's `world/`: spec, model,
manifest, `world_truth.json`, task texts, health, generator record) and the stored runs
(`runs/<wid>-observed`, `runs/<wid>-later`). Nothing is written into the world.

How a world extends past its stored days (loop.md, section 1; the lane's judgment calls):

- **Faults.** On a day within the world, the faults in force are the world's own
  (`WorldTruth.faults_on`, which the as-run manifest agrees with). Past the last stored
  day a fault without an end stays in force and a drift holds the level of the last
  stored day; a fault with an end is in force while its schedule says so.
- **Regular work** on a stored day is the world's own as-run identity (the manifest:
  episode id, task, start state, seeds), with the current software and the faults under
  the ledger. Past the horizon there is no regular work in a two-round layout; minting it
  is refused rather than invented.
- **The reserved panel** (`panel_identities`): per robot, per evaluation day,
  `episodes_per_robot_day` attempts, numbered 0, 1, ... like a day's regular work, with
  the seeds of `contracts.identity.seeds_for` for that robot, day and index, and the
  task and start state drawn the way the world's generator draws its regular work (the
  version-1 slot rule with the site emphasis of `generator.json`, or the version-0 rule
  of the miniature fleet). Days past the horizon have never been run, so no identity of
  the panel has been exported. Episode ids come from a block of their own
  (`PANEL_ID_BASE` above the world's first free number), never used by experiments.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import FaultSpec, TruthTable
from fleetlab.contracts.identity import (
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    Task,
    seeds_for,
)
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.model import ConfigVersion, Fleet
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import COMPONENT_VERBS, Component, ConditionKey, ConfigKey, Verb
from fleetlab.contracts.world import WorldSpec
from fleetlab.engine.experiments import WorldTruth
from fleetlab.loop.state import (
    Ledger,
    PolicyOf,
    RobotDay,
    RobotState,
    apply_arm,
    believed_fleet,
    ledger_records,
    materialize,
)

__all__ = ["DEFAULT_REGISTRY", "PANEL_ID_BASE", "LoopWorld"]

DEFAULT_REGISTRY = "fleetlab.faults:REGISTRY"
# The reserved panel's episode ids start this far above the world's first free number.
PANEL_ID_BASE = 500_000


@dataclass(frozen=True)
class LoopWorld:
    """A stored world: its files, its as-run identities, its stored runs."""

    root: Path
    files: Path
    spec: WorldSpec
    truth: WorldTruth
    identities: tuple[EpisodeIdentity, ...]
    model: Fleet  # the model at the end of the world, traits included (audit vocabulary)
    task_classes: tuple[str, ...]
    health: tuple[dict[str, Any], ...]
    generator: dict[str, Any] | None
    registry: str = DEFAULT_REGISTRY
    runs: tuple[Path, ...] = ()
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, root: Path | str, registry: str = DEFAULT_REGISTRY) -> LoopWorld:
        from fleetlab.oracle.world import world_dir_of, world_runs

        root = Path(root).expanduser().resolve()
        files = world_dir_of(root)
        spec = WorldSpec.model_validate_json((files / "spec.json").read_text())
        truth = WorldTruth.load(files / "world_truth.json")
        identities = tuple(
            Job.model_validate_json(line).identity
            for line in (files / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
        model = Fleet.model_validate_json((files / "model.json").read_text())
        texts_file = files / "task_texts.json"
        if texts_file.is_file():
            by_task = {
                (t["suite"], int(t["task_index"])): str(t["text"])
                for t in json.loads(texts_file.read_text())
            }
            classes = tuple(by_task[(t.suite, t.task_index)] for t in spec.tasks)
        else:
            classes = tuple(truth.task_classes)
        health_file = files / "health.jsonl"
        health = (
            tuple(json.loads(line) for line in health_file.read_text().splitlines() if line.strip())
            if health_file.is_file()
            else ()
        )
        generator_file = files / "generator.json"
        generator = json.loads(generator_file.read_text()) if generator_file.is_file() else None
        runs = tuple(world_runs(root))
        world = cls(
            root=root,
            files=files,
            spec=spec,
            truth=truth,
            identities=identities,
            model=model,
            task_classes=classes,
            health=health,
            generator=generator,
            registry=registry,
            runs=runs,
        )
        world.check()
        return world

    # ---- what the world is ------------------------------------------------------------------

    @property
    def world_id(self) -> str:
        return self.spec.world_id

    @property
    def days(self) -> int:
        """The stored horizon: days 0 .. days - 1."""
        return self.spec.days

    @property
    def observed_run(self) -> Path:
        """The world's stored run of the observed window (round 1's operation)."""
        for run in self.runs:
            if run.name.endswith("-observed"):
                return run
        raise FileNotFoundError(f"{self.root}: no observed run")

    @property
    def events_record(self) -> dict[str, Any] | None:
        """What the world runner did for the observed run's derived events (`runner.json`,
        step `observed-events`), or None when the world's runs never got them (the
        miniature fleet predates them)."""
        record = self.root / "runner.json"
        if not record.is_file():
            return None
        step = (json.loads(record.read_text()).get("steps") or {}).get("observed-events")
        return step if isinstance(step, dict) and step.get("done") else None

    def event_setup(self, setup: Any) -> Any:
        """`setup` with the derived events the world's own runs carry: none if they carry
        none, the telemetry ones, and the contact replay only if the world's had it. The
        loop's runs record the same kinds of events as the history they extend."""
        from dataclasses import replace as replace_setup

        record = self.events_record
        if record is None:
            return replace_setup(setup, events=False)
        return replace_setup(setup, contacts=setup.contacts and "contact_replay" in record)

    def check(self) -> None:
        """Refuse a world the loop's rules do not cover: a configuration fault whose
        scope and key carry another recorded change after its onset (revert would then
        mean two things)."""
        from fleetlab.faults.records import change_entry  # noqa: F401  (registry loaded)

        for fault in self.spec.fault_schedule:
            key = fault.target.config_key
            if key is None:
                continue
            later = [
                c
                for c in self.model.change_log
                if c.scope == fault.target.scope
                and c.key == key
                and c.day > fault.schedule.onset_day
            ]
            if later:
                raise ValueError(
                    f"{self.world_id}: a recorded change of {key.value} at "
                    f"{fault.target.scope} follows a configuration fault there"
                )

    def truth_table(self) -> TruthTable:
        from fleetlab.engine.experiments import _truth_table

        cached = self._cache.get("truth")
        if cached is None:
            cached = self._cache["truth"] = _truth_table(self.registry)
        table: TruthTable = cached
        return table

    # ---- faults and values ------------------------------------------------------------------

    def world_faults_on(self, robot_id: str, day: int) -> list[FaultSpec]:
        """Every fault of the world in force on a robot on a day, each at the level of that
        day, a configuration fault a narrower scope shadows included (the layered state
        decides which applies). Past the horizon a drift holds its last stored level."""
        site_id = self.truth.fleet.robot(robot_id).site_id
        level_day = min(day, self.days - 1)
        out = []
        for fault in self.truth.fault_schedule:
            if not (fault.schedule.active_on(day) and fault.target.covers(robot_id, site_id)):
                continue
            level = self.truth.fault_levels.get(fault.fault_id, {}).get(level_day)
            moved = fault if level is None else fault.model_copy(update={"level": level})
            if moved.is_fault:
                out.append(moved)
        return out

    def layers_of(self, robot_id: str) -> dict[str, dict[ConfigKey, ConfigValue]]:
        """The world model's own configuration values at the fleet, the robot's site and
        the robot (the model as the experiment window starts, as the as-run faults were
        derived from it)."""
        fleet = self.truth.fleet
        robot = fleet.robot(robot_id)
        return {
            "fleet": dict(fleet.config.values),
            robot.site_id: dict(fleet.site(robot.site_id).config.values),
            robot.id: dict(robot.config.values),
        }

    def policy_of(self, robot_id: str, day: int) -> PolicyOf:
        """The engine's effective policy for the robot on the day under a list of faults:
        the registry's policy input hooks in the list's order
        (`fleetlab.engine.episode.effective_spec`), as an attempt would run them."""
        return hook_chain(self.registry, self.probe(robot_id, day))

    def probe(self, robot_id: str, day: int) -> EpisodeIdentity:
        """An identity of the robot on the day with no fault, to run hooks on (never run)."""
        robot = self.truth.fleet.robot(robot_id)
        update = {"robot_id": robot_id, "site_id": robot.site_id, "day": day, "faults": []}
        return self.identities[0].model_copy(update=update)

    def faults_on(self, robot_id: str, day: int) -> list[FaultSpec]:
        """The world's own faults in force for a robot on a day (no ledger): the as-run
        faults on a stored day; past the horizon a drift holds its last stored level."""
        return list(self.state_on(robot_id, day, Ledger.empty(self.world_id)).faults)

    def values_on(self, robot_id: str) -> dict[ConditionKey | ConfigKey, ConfigValue]:
        """The robot's configuration and its site's conditions as the world holds them
        when the experiment window starts (`resolve`'s current values before any step)."""
        return self.truth.current(robot_id)

    def state_on(
        self,
        robot_id: str,
        day: int,
        ledger: Ledger,
        arm: Sequence[Action] = (),
    ) -> RobotState:
        """The robot's effective state on a day under the ledger, with an arm's actions as
        a delta on top (the one interpretation operation, experiments and the evaluation
        share)."""
        key = (robot_id, day, ledger.model_dump_json())
        cache = self._cache.setdefault("states", {})
        state = cache.get(key)
        if state is None:
            site_id = self.truth.fleet.robot(robot_id).site_id
            state = materialize(
                self.world_faults_on(robot_id, day),
                RobotDay(robot_id, site_id, day),
                ledger.active_steps(day),
                self.truth_table(),
                self.values_on(robot_id),
                self.layers_of(robot_id),
                self.policy_of(robot_id, day),
            )
            cache[key] = state
        return apply_arm(state, arm, self.truth_table()) if arm else state

    # ---- the records ------------------------------------------------------------------------

    def fleet_on(self, day: int) -> Fleet:
        """The world's own records as they stand at the end of `day`, without the ledger:
        the observed fleet rolled forward by the recorded changes and maintenance after the
        observed window, with that day's health readings on the components."""
        observed = self.spec.windows.observed.last
        if day < observed:
            raise ValueError(f"the loop's records start at the observed window's end ({observed})")
        cached = self._cache.setdefault("fleet_on", {}).get(day)
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        fleet = self.truth.fleet
        if day > observed:
            snapshot = _snapshot([r for r in self.health if r["day"] == day])
            fleet = _roll_forward(fleet, self.model, observed, day, snapshot)
        self._cache["fleet_on"][day] = fleet
        return fleet

    def public_fleet(self, day: int, ledger: Ledger) -> Fleet:
        """The fleet's model at the end of `day` after the ledger, as the fleet believes it
        (the description and logs a round's export carries). The components' latest
        health readings are that day's readings under the ledger (`health_rows`), the same
        numbers the health table holds."""
        if not ledger.entries:
            return self.fleet_on(day)
        believed = believed_fleet(self.fleet_on, ledger, day)
        return _with_health(believed, _snapshot(self.health_rows([day], ledger)))

    def health_rows(self, days: Sequence[int], ledger: Ledger) -> list[dict[str, Any]]:
        """The daily health readings of `days` under the ledger.

        Without a component entry in the ledger and inside the horizon, the world's stored
        readings. Otherwise the world's own health model again (version 1:
        `worlds.health.rows`; version 0: the miniature fleet's), with each component fault
        a ledger entry removed ended on that entry's adoption day, and the ledger's records
        among those that take benign drift back. The readings of every stored day before
        the first adoption must come out as stored, or the call refuses. A world without
        health readings (the toy) has none.
        """
        wanted = sorted(set(days))
        if not self.health or not wanted:
            return []
        components = [e for e in ledger.entries if e.action.verb in COMPONENT_VERBS]
        if not components and wanted[-1] < self.days:
            return _sorted_health([r for r in self.health if r["day"] in wanted])
        until = wanted[-1]
        schedule = self._health_schedule(ledger)
        changes, upkeep = list(self.model.change_log), list(self.model.maintenance_log)
        if ledger.entries:
            extra, extra_upkeep, _ = ledger_records(self.fleet_on, ledger, until)
            changes += extra
            upkeep += extra_upkeep
        rows = self._health_model(until + 1, schedule, changes, upkeep)
        first = min((e.adopted_day for e in ledger.entries), default=until + 1)
        stored = {_health_key(r): r["value"] for r in self.health}
        drifted = [
            k
            for r in rows
            if r["day"] < min(first, self.days) and stored.get(k := _health_key(r)) != r["value"]
        ]
        if drifted:
            raise ValueError(
                f"{self.world_id}: the health model no longer gives the stored readings "
                f"({len(drifted)} differ, first {drifted[0]})"
            )
        return _sorted_health([r for r in rows if r["day"] in wanted])

    def _health_schedule(self, ledger: Ledger) -> list[FaultSpec]:
        """The world's schedule with each component fault an entry fixed ended on the
        entry's adoption day (component faults are robot-scoped, one per robot)."""
        truth = self.truth_table()
        out = []
        for fault in self.spec.fault_schedule:
            end: int | None = None
            for e in ledger.entries:
                action = e.action
                if (
                    action.verb in COMPONENT_VERBS
                    and fault.target.component is not None
                    and action.target.component is fault.target.component
                    and action.target.scope == fault.target.scope
                    and action.verb in truth[fault.kind].fixing_verbs
                    and fault.schedule.onset_day <= e.adopted_day
                ):
                    end = e.adopted_day if end is None else min(end, e.adopted_day)
            if end is None:
                out.append(fault)
            elif end > fault.schedule.onset_day:
                length = end - fault.schedule.onset_day
                if fault.schedule.duration_days is not None:
                    length = min(length, fault.schedule.duration_days)
                schedule = fault.schedule.model_copy(update={"duration_days": length})
                out.append(fault.model_copy(update={"schedule": schedule}))
        return out

    def _health_model(
        self,
        days: int,
        schedule: Sequence[FaultSpec],
        changes: Sequence[Any],
        upkeep: Sequence[Any],
    ) -> list[dict[str, Any]]:
        from fleetlab.worlds import base, generate, health

        spec = self.spec
        if self.generator_version == "v1":
            robots = base.all_robots(spec.sites, spec.robots_per_site)
            return health.rows(
                spec.seed, robots, days, spec.episodes_per_robot_day, schedule, changes, upkeep
            )
        longer = spec.model_copy(update={"fault_schedule": list(schedule), "days": days})
        return generate._health_rows(longer, [r.id for r in self.truth.fleet.robots()])

    @property
    def generator_version(self) -> str:
        """ "v1" for the world generator's version-1 worlds, else "v0" (the miniature fleet's
        and the toy's rule)."""
        name = (self.generator or {}).get("generator", "")
        return "v1" if str(name).endswith("v1") else "v0"

    # ---- identities -------------------------------------------------------------------------

    def regular_identities(
        self, days: Sequence[int], software: SoftwareVersion, ledger: Ledger
    ) -> list[EpisodeIdentity]:
        """The world's regular work on `days`, as run (manifest order), with `software` and
        each attempt's faults materialized under the ledger. Refuses a day past the horizon."""
        wanted = set(days)
        past = sorted(d for d in wanted if d >= self.days)
        if past:
            raise ValueError(
                f"{self.world_id} has no regular work after day {self.days - 1} (asked {past})"
            )
        states: dict[tuple[str, int], list[FaultSpec]] = {}
        out = []
        for identity in self.identities:
            if identity.day not in wanted:
                continue
            key = (identity.robot_id, identity.day)
            if key not in states:
                states[key] = list(self.state_on(identity.robot_id, identity.day, ledger).faults)
            out.append(_with(identity, software, states[key]))
        return out

    def panel_identities(
        self,
        days: Sequence[int],
        software: SoftwareVersion,
        batch_size: int | None = None,
        per_robot_day: int | None = None,
    ) -> list[EpisodeIdentity]:
        """The reserved evaluation panel on `days`, faults those of the original state
        (`evaluate` materializes them per state). A day inside the horizon reuses the
        world's as-run identities of that day; a day past it is minted by the world's own
        slot rule, with ids from the panel's own block (`panel_episode_id`).
        `per_robot_day` keeps the first so many attempts of each robot and day (default:
        the world's whole day)."""
        batch = batch_size or (self.identities[0].batch_size if self.identities else 4)
        per_day = self.truth.episodes_per_robot_day
        keep = per_day if per_robot_day is None else min(per_robot_day, per_day)
        out: list[EpisodeIdentity] = []
        number = 0
        for day in sorted(set(days)):
            if day < self.days:
                out += [
                    _with(i, software, i.faults).model_copy(update={"batch_size": batch})
                    for i in self.identities
                    if i.day == day and i.episode_index < keep
                ]
                continue
            for robot in self.truth.fleet.robots():
                faults = self.faults_on(robot.id, day)
                for index in range(keep):
                    task, start = self.slot_tasks(robot.id, day, index)
                    scene, noise = seeds_for(self.truth.world_seed, robot.id, day, index)
                    out.append(
                        EpisodeIdentity(
                            world_id=self.world_id,
                            episode_id=panel_episode_id(self, number),
                            robot_id=robot.id,
                            site_id=robot.site_id,
                            day=day,
                            episode_index=index,
                            suite=task.suite,
                            task_index=task.task_index,
                            start_state=start,
                            scene_seed=scene,
                            noise_seed=noise,
                            policy=self.truth.policy,
                            batch_size=batch,
                            row=None,
                            image_size=self.truth.image_size,
                            max_steps=self.truth.steps_for(task.suite),
                            faults=faults,
                            software=software,
                        )
                    )
                    number += 1
        return out

    def slot_tasks(self, robot_id: str, day: int, index: int) -> tuple[Task, int]:
        """The task and start state the world's own generator gives a robot's attempt
        `index` on `day` (version 1: `worlds.base.slots`; version 0: `worlds.generate`)."""
        from fleetlab.worlds import base, generate

        spec = self.spec
        site_id = self.truth.fleet.robot(robot_id).site_id
        if self.generator_version == "v1":
            recipe = (self.generator or {}).get("recipe", {})
            suites = tuple((str(a), str(b)) for a, b in (recipe.get("site_suites") or ()))
            slots = base.slots(
                spec.seed,
                spec.sites,
                spec.robots_per_site,
                day + 1,
                spec.episodes_per_robot_day,
                spec.tasks,
                suites,
            )
            for slot in slots[-spec.sites * spec.robots_per_site * spec.episodes_per_robot_day :]:
                if (slot.robot_id, slot.day, slot.index) == (robot_id, day, index):
                    return slot.task, slot.start_state
            raise KeyError(f"no slot for {robot_id} day {day} index {index}")
        site_index = int(site_id[1:]) - 1
        task = generate.draw_task(spec, site_index, robot_id, day, index)
        start = int(
            generate.rng(spec.seed, "start", robot_id, day, index).integers(generate.START_STATES)
        )
        return task, start


def _snapshot(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """Readings by (robot, component): what a component's `health` shows for their day."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        out.setdefault((row["robot_id"], row["component"]), {})[row["metric"]] = row["value"]
    return out


def _with_health(fleet: Fleet, snapshot: dict[tuple[str, str], dict[str, float]]) -> Fleet:
    """The fleet with its components' latest readings replaced where `snapshot` has them."""
    if not snapshot:
        return fleet
    sites = []
    for site in fleet.sites:
        robots = []
        for robot in site.robots:
            components = {
                c: (
                    state.model_copy(update={"health": snapshot[(robot.id, c.value)]})
                    if (robot.id, c.value) in snapshot
                    else state
                )
                for c, state in robot.components.items()
            }
            robots.append(robot.model_copy(update={"components": components}))
        sites.append(site.model_copy(update={"robots": robots}))
    return fleet.model_copy(update={"sites": sites})


def hook_chain(registry: str, probe: EpisodeIdentity) -> PolicyOf:
    """The effective policy of `probe` under a list of faults, by the registry's own
    policy input hooks in the list's order (what the engine's attempt computes)."""
    from fleetlab.engine.episode import effective_spec
    from fleetlab.engine.factories import resolve as resolve_path

    types = resolve_path(registry)

    def run(faults: Sequence[FaultSpec]) -> PolicySpec:
        pairs = []
        for fault in faults:
            fault_type = types.get(fault.kind)
            pairs.append((fault_type, fault_type.new_episode(fault, probe)))
        return effective_spec(probe, pairs)

    return run


def _sorted_health(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Health rows in the order the world runner writes them: day, robot, component, metric."""
    return sorted(rows, key=_health_key)


def _health_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["day"], row["robot_id"], row["component"], row["metric"])


def _with(
    identity: EpisodeIdentity, software: SoftwareVersion, faults: Sequence[FaultSpec]
) -> EpisodeIdentity:
    """An as-run identity with other software and another fault list, validated again."""
    data = identity.model_dump()
    data["software"] = software.model_dump()
    data["faults"] = [f.model_dump() for f in faults]
    return EpisodeIdentity.model_validate(data)


def _roll_forward(
    fleet: Fleet,
    model: Fleet,
    after_day: int,
    until_day: int,
    health: dict[tuple[str, str], dict[str, float]],
) -> Fleet:
    """`fleet` (records up to `after_day`) with `model`'s records of the days after it up
    to `until_day` applied, and `health` as the components' latest readings."""
    changes = [c for c in model.change_log if after_day < c.day <= until_day]
    upkeep = [m for m in model.maintenance_log if after_day < m.day <= until_day]

    def bump(config: ConfigVersion, scope: str) -> ConfigVersion:
        mine = [c for c in changes if c.scope == scope]
        if not mine:
            return config
        values = dict(config.values)
        for c in sorted(mine, key=lambda c: c.day):
            values[c.key] = c.new
        return ConfigVersion(
            version=config.version + len(mine),
            values=values,
            since_day=max(c.day for c in mine),
        )

    serviced: dict[tuple[str, Component], int] = {}
    for m in sorted(upkeep, key=lambda m: m.day):
        if m.verb in (Verb.service, Verb.replace_):
            serviced[(m.robot_id, m.component)] = m.day
    sites = []
    for site in fleet.sites:
        robots = []
        for robot in site.robots:
            components = {}
            for component, state in robot.components.items():
                update: dict[str, Any] = {}
                if (robot.id, component) in serviced:
                    update["last_service_day"] = serviced[(robot.id, component)]
                readings = health.get((robot.id, component.value))
                if readings is not None:
                    update["health"] = readings
                components[component] = state.model_copy(update=update) if update else state
            robots.append(
                robot.model_copy(
                    update={"config": bump(robot.config, robot.id), "components": components}
                )
            )
        sites.append(
            site.model_copy(update={"robots": robots, "config": bump(site.config, site.id)})
        )
    return fleet.model_copy(
        update={
            "sites": sites,
            "config": bump(fleet.config, "fleet"),
            "change_log": [*fleet.change_log, *sorted(changes, key=lambda c: c.day)],
            "maintenance_log": [*fleet.maintenance_log, *sorted(upkeep, key=lambda m: m.day)],
        }
    )


def panel_episode_id(world: LoopWorld, number: int) -> str:
    """The id of the reserved panel's attempt `number` (0-based)."""
    return neutral_id(IdKind.episode, world.truth.first_free_episode + PANEL_ID_BASE + number)


__all__ += ["ledger_records", "panel_episode_id", "seeds_for"]
