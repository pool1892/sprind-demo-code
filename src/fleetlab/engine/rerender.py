"""Render a recorded episode again without its policy, to tell which rendering was the odd one.

A chain holds, per step, the state hash, both cameras' image hashes and the
executed action. When a chain-sample replay differs from its run only in an
image, the question is which of the two renderings was the odd one. This
module answers it without the model: it resets the environment as the
engine does (the faults' fresh states, the effective policy spec, the cell
hooks), steps it with the chain's executed actions, and hashes every frame
again, `passes` times in one environment. The state hash must equal the
chain's at every step, which shows that the trajectory is the same; the
image hashes are then further renderings of the same scenes. The action
hooks are not run (the chain holds the actions after them), and observation
and data hooks do not touch the rendered images. One simulator process and
no model: a light job.

    tools/heavy-run --light uv run --locked python -m fleetlab.engine.rerender \\
        RUN_DIR KEY [--against KEY ...] [--passes N]

KEY is a chain's file name without `.npz` (a job's key). A chain that
continues from a snapshot (`first_step` > 0) is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fleetlab.contracts.rollout import RunLayout

LIBERO_ENV = "fleetlab.sim.libero_sim:LiberoSimEnv"
FLEET_CHECK = "fleetlab.sim.libero_success:FleetCheck"
FAULTS = "fleetlab.faults:REGISTRY"


def rerender(
    run_dir: Path,
    key: str,
    *,
    against: Sequence[str] = (),
    passes: int = 3,
    env_factory: str = LIBERO_ENV,
    check_factory: str = FLEET_CHECK,
    fault_registry: str = FAULTS,
) -> dict[str, Any]:
    """Render the episode of chain `key` again `passes` times; compare with it and `against`."""
    import importlib

    from fleetlab.engine.chain import Chain
    from fleetlab.engine.episode import _bound_cell_hook, effective_spec, image_hashes
    from fleetlab.engine.factories import build, resolve
    from fleetlab.log.runfiles import read_identities

    layout = RunLayout(run_dir)
    chains = {k: Chain.load(layout.table("chain", k)) for k in (key, *against)}
    reference = chains[key]
    if reference.first_step != 0:
        raise ValueError(f"{key} continues from step {reference.first_step}: not supported")
    episode_id = key.split(".")[0]
    identity = {i.episode_id: i for i in read_identities(run_dir)}[episode_id]
    module = importlib.import_module(fault_registry.partition(":")[0])
    if callable(getattr(module, "load_all", None)):
        module.load_all()
    registry = resolve(fault_registry)
    make_check = resolve(check_factory)
    env = build(env_factory, identity.suite, identity.task_index, identity.image_size)
    passes_out = []
    try:
        for p in range(passes):
            t0 = time.perf_counter()
            types = [registry.get(f.kind) for f in identity.faults]
            faults = identity.faults
            pairs = [(ft, ft.new_episode(f, identity)) for ft, f in zip(types, faults, strict=True)]
            spec = effective_spec(identity, pairs)
            env_identity = (
                identity
                if spec == identity.policy
                else identity.model_copy(update={"policy": spec})
            )
            env.set_success_check(make_check())
            obs = env.reset(env_identity, [_bound_cell_hook(ft, st) for ft, st in pairs])
            frames = [(env.state_hash().hex(), *image_hashes(obs))]
            for t in range(1, len(reference.steps)):
                result = env.step(np.asarray(reference.executed[t], np.float32))
                frames.append((env.state_hash().hex(), *image_hashes(result.observation)))
            row: dict[str, Any] = {"pass": p, "seconds": round(time.perf_counter() - t0, 2)}
            for name, chain in chains.items():
                n = min(len(chain.steps), len(frames))
                row[name] = {
                    what: [s for s in range(n) if frames[s][i] != getattr(chain, what)[s]]
                    for i, what in enumerate(("state", "scene", "wrist"))
                }
            passes_out.append(row)
    finally:
        env.close()
    return {"run": str(run_dir), "key": key, "against": list(against), "passes": passes_out}


def odd_one(result: dict[str, Any]) -> dict[str, Any]:
    """Per compared chain: the steps where every new rendering agrees with each other and not
    with that chain (its odd frames), and whether the new renderings agreed among themselves."""
    out: dict[str, Any] = {}
    names = [result["key"], *result["against"]]
    for name in names:
        for what in ("state", "scene", "wrist"):
            sets = [set(p[name][what]) for p in result["passes"]]
            out.setdefault(name, {})[what] = {
                "in_every_pass": sorted(set.intersection(*sets)) if sets else [],
                "in_some_pass": sorted(set.union(*sets)) if sets else [],
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.engine.rerender")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("key")
    parser.add_argument("--against", nargs="*", default=[])
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--env-factory", default=LIBERO_ENV)
    parser.add_argument("--check-factory", default=FLEET_CHECK)
    parser.add_argument("--fault-registry", default=FAULTS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)

    result = rerender(
        args.run_dir,
        args.key,
        against=args.against,
        passes=args.passes,
        env_factory=args.env_factory,
        check_factory=args.check_factory,
        fault_registry=args.fault_registry,
    )
    result["odd"] = odd_one(result)
    text = json.dumps(result, indent=1)
    if args.json_out is not None:
        args.json_out.write_text(text)
    else:
        print(text)
    return 0


def odd_rendering(
    result: dict[str, Any], run_key: str, replay_key: str, frame: tuple[int, str]
) -> tuple[str | None, str]:
    """Which rendering of `frame` (step, camera) was the odd one: "run", "replay" or None.

    Named only when every new pass followed the run's states, every new pass
    agrees with one of the two at that frame and not with the other, and the
    new passes differ from that one nowhere else.
    """
    step, camera = frame
    odd = odd_one(result)
    for name in (run_key, replay_key):
        if odd[name]["state"]["in_some_pass"]:
            return None, f"the re-rendering left {name}'s states"
    run_bad = odd[run_key][camera]["in_every_pass"] == [step]
    replay_bad = odd[replay_key][camera]["in_every_pass"] == [step]
    other = "scene" if camera == "wrist" else "wrist"
    clean = {
        name: not odd[name][other]["in_some_pass"]
        and odd[name][camera]["in_some_pass"] in ([], [step])
        for name in (run_key, replay_key)
    }
    if replay_bad and not odd[run_key][camera]["in_some_pass"] and clean[replay_key]:
        return "replay", ""
    if run_bad and not odd[replay_key][camera]["in_some_pass"] and clean[run_key]:
        return "run", ""
    return None, "the re-renderings did not agree with one rendering throughout"


if __name__ == "__main__":
    sys.exit(main())
