"""A toy world with a lasting command-path condition at one site: the probe lane's CPU fixture.

    world, setup = write_link_toy_world(root, link_steps)

The loop's toy world (`fleetlab.loop.toy`: two sites, four robots, the toy cell, policy
and faults) with one more fault per robot of site `s2`: the lasting remote-link condition
(`remote_link_latency`, the real fault type, on each robot's controller, in force from day
0 without an end, fixed by no verb), and `s2`'s public note that its robots are driven
over a remote link (`SITE_NOTE`), as the world generators write it. Nothing else names the
condition; its size is the fixture's own `link_steps`, unrelated to any development or
demonstration fleet. The registry is the toy oracle's plus that one type
(`LINK_REGISTRY`), for the engine's workers and for the loop's world alike.

Nothing here imports torch, mujoco or LIBERO.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fleetlab.contracts.faults import FaultSpec, Schedule, Target, active_faults
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.rollout import Job, RunRecord
from fleetlab.contracts.vocabulary import Component
from fleetlab.contracts.world import WorldSpec, seal_world
from fleetlab.engine import toy
from fleetlab.faults import FaultRegistry
from fleetlab.faults.remote_link_latency import SITE_NOTE, RemoteLinkLatency
from fleetlab.loop import toy as toy_world
from fleetlab.loop.operate import EngineSetup
from fleetlab.loop.world import LoopWorld
from fleetlab.oracle.toy import TOY_ORACLE_FAULTS

__all__ = ["LINK_REGISTRY", "LINK_ROBOTS", "LINK_SITE", "link_faults", "write_link_toy_world"]

LINK_REGISTRY = "fleetlab.loop.probe_toy:LINK_FAULTS"
LINK_SITE = "s2"
LINK_ROBOTS = ("r03", "r04")

LINK_FAULTS = FaultRegistry()
for _kind in TOY_ORACLE_FAULTS.kinds():
    LINK_FAULTS.add(TOY_ORACLE_FAULTS.get(_kind))
LINK_FAULTS.add(RemoteLinkLatency())


def link_faults(link_steps: int) -> list[FaultSpec]:
    """One lasting link fault per robot of `LINK_SITE`, on its controller, from day 0."""
    return [
        FaultSpec(
            fault_id=f"f{11 + n:02d}",
            kind="remote_link_latency",
            params={},
            level=float(link_steps),
            target=Target(scope=robot_id, component=Component.controller),
            schedule=Schedule(onset_day=0),
        )
        for n, robot_id in enumerate(LINK_ROBOTS)
    ]


def _schedule(link_steps: int) -> list[FaultSpec]:
    return [*toy_world.faults(), *link_faults(link_steps)]


def _fleet(until_day: int) -> Fleet:
    """The toy fleet's records, with the link site's operator note."""
    fleet = toy_world.fleet(until_day)
    sites = [
        s.model_copy(update={"notes": [SITE_NOTE]}) if s.id == LINK_SITE else s for s in fleet.sites
    ]
    return fleet.model_copy(update={"sites": sites})


def _spec(link_steps: int) -> WorldSpec:
    return toy_world.spec().model_copy(update={"fault_schedule": _schedule(link_steps)})


def _jobs(software: object, link_steps: int) -> list[Job]:
    """The toy world's run jobs (`fleetlab.loop.toy.jobs`) with the link faults in force."""
    out = []
    for job in toy_world.jobs(software):
        identity = job.identity
        faults = active_faults(
            _schedule(link_steps), _fleet(identity.day), identity.robot_id, identity.day
        )
        out.append(job.model_copy(update={"identity": _with_faults(identity, faults)}))
    return out


def _with_faults(identity: EpisodeIdentity, faults: list[FaultSpec]) -> EpisodeIdentity:
    return EpisodeIdentity.model_validate({**identity.model_dump(), "faults": faults})


# The fixture's own link: long enough to show in the toy cell's motion.
LINK_STEPS = 4


def write_link_toy_world(
    root: Path | str, link_steps: int = LINK_STEPS, workers: int = 2
) -> tuple[LoopWorld, EngineSetup]:
    """Write the link toy world under `root` and run its observed window; resumable."""
    from fleetlab.engine import software as soft
    from fleetlab.engine.fleet import run
    from fleetlab.log import RunWriter

    root = Path(root)
    files = root / "world"
    files.mkdir(parents=True, exist_ok=True)
    software = soft.current("none")
    manifest = _jobs(software, link_steps)
    world_spec = _spec(link_steps)
    observed_run = root / "runs" / f"{toy_world.WORLD_ID}-observed"
    (files / "spec.json").write_text(world_spec.model_dump_json(indent=1) + "\n")
    (files / "model.json").write_text(_fleet(toy_world.DAYS - 1).model_dump_json(indent=1) + "\n")
    (files / "description.json").write_text("{}\n")
    (files / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in manifest))
    seal_world(files)
    truth = {
        "world_id": toy_world.WORLD_ID,
        "world_seed": toy_world.SEED,
        "fleet": _fleet(world_spec.windows.observed.last).model_dump(mode="json"),
        "windows": world_spec.windows.model_dump(mode="json"),
        "fault_schedule": [f.model_dump(mode="json") for f in _schedule(link_steps)],
        "fault_levels": {},
        "episodes_per_robot_day": toy_world.PER_DAY,
        "tasks": [list(t) for t in toy_world.TASKS],
        "task_classes": toy_world.TASK_CLASSES,
        "start_states": 50,
        "policy": toy.TOY_SPEC.model_dump(mode="json"),
        "image_size": toy_world.IMAGE,
        "max_steps": {toy.TOY_SUITE: toy_world.MAX_STEPS},
        "base_runs": [str(observed_run)],
        "first_free_episode": len(manifest) + 1,
    }
    (files / "world_truth.json").write_text(json.dumps(truth, indent=1) + "\n")
    (files / "task_texts.json").write_text(
        json.dumps(
            [
                {"suite": t.suite, "task_index": t.task_index, "text": text}
                for t, text in zip(toy_world.TASKS, toy_world.TASK_CLASSES, strict=True)
            ]
        )
        + "\n"
    )
    (files / "generator.json").write_text(json.dumps({"generator": "fleetlab.worlds v0"}) + "\n")
    base = toy_world.toy_setup(workers)
    setup = replace(base, config=replace(base.config, fault_registry=LINK_REGISTRY))
    observed = [j for j in manifest if j.identity.day in world_spec.windows.observed]
    if not (observed_run / "log" / "description.json").is_file():
        run(setup.for_run(observed_run, toy_world.PER_DAY), observed)
        record = RunRecord.model_validate_json((observed_run / "run.json").read_text())
        with RunWriter(observed_run, record, episodes_per_robot_day=toy_world.PER_DAY) as writer:
            writer.write_model(_fleet(world_spec.windows.observed.last), toy_world.TASK_CLASSES)
            writer.write_table("health", [])
            writer.finalize()
    return LoopWorld.load(root, registry=LINK_REGISTRY), setup
