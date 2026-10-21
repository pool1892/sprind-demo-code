"""Which catalogue action removes which lookalike fault? (lane P, step 5; no simulation)

    uv run --locked python spikes/instrument/catalogue_check.py OUT.json

Every condition of `probe_shapes.py` is planted on r01's identity as there, and each
candidate action is resolved against it by `contracts.actions.resolve_all`, the function
the loop uses for an adopted action and for an experiment arm (`LoopWorld.state_on`,
`fleetlab.adaptation.identity`). Prints, per fault, the actions that remove it, the ones
that leave it in force, and what each costs (`contracts.actions.COSTS`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from fleetlab.runtime import configure_process

    configure_process()
    from probe_shapes import CONDITIONS, PROGRAMS, ROBOT, WORLD, jobs_for

    import fleetlab  # noqa: F401  (torch before mujoco)
    import fleetlab.faults as faults
    from fleetlab.contracts.actions import Action, cost_of, resolve_all
    from fleetlab.contracts.faults import Target
    from fleetlab.contracts.vocabulary import LATER_CONFIG_NOMINAL, Component, ConfigKey, Verb
    from fleetlab.engine import software as soft
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.world import LoopWorld

    world = LoopWorld.load(WORLD)
    layout = layout_for_world(world.spec)
    site = world.truth.fleet.robot(ROBOT).site_id

    def component(verb: Verb, part: Component) -> Action:
        return Action(verb=verb, target=Target(scope=ROBOT, component=part))

    actions: dict[str, Action] = {}
    for part in (Component.controller, Component.arm, Component.scene_camera):
        for verb in (Verb.service, Verb.replace_, Verb.recalibrate):
            if verb is Verb.recalibrate and part is Component.controller:
                continue  # not in the vocabulary: recalibrate applies to cameras and the arm
            actions[f"{verb.value} {part.value}"] = component(verb, part)
    actions["set control.latency_compensation_steps = 17 at the site"] = Action(
        verb=Verb.set,
        target=Target(scope=site, config_key=ConfigKey.control_latency_compensation_steps),
        value=17,
    )
    for key in (ConfigKey.software_version, ConfigKey.camera_scene_calibration):
        actions[f"revert {key.value} at the robot"] = Action(
            verb=Verb.revert, target=Target(scope=ROBOT, config_key=key)
        )
    # w10-v2 predates the compensation setting: give it the key's nominal, as a fleet
    # that lists it in `later_settings` records it.
    current = {**world.truth.current(ROBOT), **LATER_CONFIG_NOMINAL}
    out: dict[str, Any] = {}
    for index, condition in enumerate(CONDITIONS):
        if condition.kind is None:
            continue
        (job, *_) = jobs_for(
            world, layout, index, condition, soft.current("none"), PROGRAMS["step"]
        )
        identity = job.identity
        removes, leaves = [], []
        for name, action in actions.items():
            result = resolve_all([action], identity, identity.faults, faults.truth_table(), current)
            hours = cost_of(action)
            entry = f"{name} ({hours:g} h)" if hours is not None else name
            (removes if result.removed else leaves).append(entry)
        out[condition.name] = {"removed_by": removes, "left_in_force_by": leaves}
        print(f"{condition.name}: removed by {removes}", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
