"""Dry run of the runner's paths without the model (light): a stub policy holds still and,
on its first call, moves the cream cheese into the bowl, so step 1 of J1 reaches its goal,
the switch (and in `release` the transition) runs, and step 2 times out.

    tools/heavy-run --light uv run --locked python spikes/jobs/dry_run.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import fleetlab  # noqa: F401  first: torch before mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goals_check
import runner


class Stub:
    """Holds the commanded pose; the first call puts the cheese 6 cm above the bowl."""

    def __init__(self, env: Any) -> None:
        from fleetlab.policies.xvla import nominal_spec

        self.env, self.spec, self.calls = env, nominal_spec(runner.PRECISION), 0

    def act(self, batch: Any, instructions: Any, noise: Any, spec: Any) -> list[Any]:
        from fleetlab.contracts.policy import ActionChunk

        if self.calls == 0:
            bowl = self.env.privileged_state().object_poses["akita_black_bowl_1"][:3]
            goals_check.move(self.env, "cream_cheese_1", bowl + np.array([0.0, 0.0, 0.06]))
        self.calls += 1
        a = goals_check.hold(batch[0])
        return [ActionChunk(chunk_index=n.chunk_index, actions=np.tile(a, (30, 1))) for n in noise]

    def convert(self, chunk: Any, control_mode: Any) -> list[np.ndarray]:
        return [row for row in chunk.actions]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from fleetlab.engine import software

    cat = runner.Catalogue()
    soft = software.current("none")
    env = runner.make_env(cat.bench[cat.stems[6]])
    goals = [cat.goal(n, env._rs) for n in runner.JOBS["J1"]]
    for variant in runner.VARIANTS:
        stub = Stub(env)
        s = runner.Run(env, stub, soft, goals, "J1", variant, 0, args.out / variant).run(True)
        print(
            variant,
            s["tag"],
            s["reason"],
            [(r["outcome"], r.get("end_t"), r.get("held")) for r in s["steps"]],
            s["control_steps"],
            s["base_check_mismatches"],
        )
        print(json.dumps(s["delivery"]))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
