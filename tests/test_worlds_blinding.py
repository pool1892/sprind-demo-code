"""Generated identities through a real export, with synthetic telemetry (no simulator).

The attacker is given only that export and the source bundle mounted for agents.
Truth is read by the verifier after the attack, never passed into its subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from log_toy import episode_data, fixture_hash_of
from test_worlds_v1 import SOFTWARE, TEXTS, example_curves, recipes

from fleetlab.agents.toolenv import build_bundle
from fleetlab.contracts.env import REASON_PLACED, REASON_STEP_LIMIT
from fleetlab.contracts.identity import WORLD_SEED_LIMIT, seeds_for
from fleetlab.contracts.rollout import RunRecord
from fleetlab.log import EpisodeWriter, RunWriter, export_runs
from fleetlab.worlds.compose import compose
from fleetlab.worlds.recipe import DEVELOPMENT, WorldRecipe, new_world_seed
from fleetlab.worlds.world import generate

# This is the seed-leak attack: fleet-id suffix -> world seed; visible clocks
# and daily episode count -> episode index; the public algorithm -> both seeds.
# It receives no truth path, recipe, generator module or expected seed.
ATTACK = r"""
import sys
sys.path.insert(0, sys.argv[1])
import json
from pathlib import Path
import pyarrow.parquet as pq
# The hardened batch-4 bundle omits the identity module. Grant the attacker
# the public seed algorithm anyway, so a missing import cannot make this pass.
import hashlib

def seeds_for(seed, robot, day, index):
    fields = json.dumps([seed, robot, day, index], separators=(",", ":"))
    def digest(purpose):
        return hashlib.sha256(f"fleetlab.seeds.v1\0{purpose}\0{fields}".encode()).digest()
    scene = int.from_bytes(digest("scene")[:4], "big")
    noise = int.from_bytes(digest("noise")[:8], "big") >> 1
    return scene, noise

root = Path(sys.argv[2])
fleet = json.loads((root / "log/description.json").read_text())["fleet"]
rows = pq.read_table(root / "log/episodes.parquet").to_pylist()
groups = {}
for row in rows:
    groups.setdefault((row["robot_id"], row["day"]), []).append(row)
n = max(map(len, groups.values()))
clocks = {(8 * 3600 + i * 8 * 3600 // n): i for i in range(n)}
guess = int(fleet["id"][1:])
result = []
for row in rows:
    t = row["started_at"]
    index = clocks[t.hour * 3600 + t.minute * 60 + t.second]
    result.append({"episode_id": row["episode_id"],
                   "seeds": seeds_for(guess, row["robot_id"], row["day"], index)})
print(json.dumps({"guessed_world_seed": guess, "episodes": result}))
"""


def attack(export: Path, bundle: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-I", "-c", ATTACK, str(bundle), str(export)],
        text=True,
        capture_output=True,
        check=True,
    )
    value: dict[str, Any] = json.loads(result.stdout)
    return value


def test_new_seeds_fit_every_stored_type_and_small_seeds_are_refused() -> None:
    for _ in range(20):
        seed = new_world_seed()
        assert 1 << 62 <= seed < WORLD_SEED_LIMIT
        # Identity hashes preserve the full integer; neither input nor derived
        # values overflow the signed-int64 identity storage columns.
        scene, noise = seeds_for(seed, "r01", 0, 0)
        assert 0 <= scene < 2**32 and 0 <= noise < 2**63
    with pytest.raises(ValueError, match="62 random bits"):
        WorldRecipe(name="new", world_id="w99", seed=99)
    with pytest.raises(ValueError, match="truth world id"):
        WorldRecipe(name="new", world_id="w01")
    with pytest.raises(ValueError, match="encode"):
        WorldRecipe(name="new", world_id="w99", seed=2**62, public_fleet_id=f"w{2**62}")
    with pytest.raises(ValueError, match="signed int64"):
        seeds_for(2**63, "r01", 0, 0)


def test_copying_a_recipe_cannot_bypass_blinding_validation() -> None:
    with pytest.raises(ValueError, match="historical D1/w41"):
        WorldRecipe(name="new", world_id="w99", seed=99, legacy_identity=True)
    with pytest.raises(ValueError, match="historical D1/w41"):
        compose(DEVELOPMENT["D1"].model_copy(update={"world_id": "w99"}))
    with pytest.raises(ValueError, match="62 random bits"):
        compose(DEVELOPMENT["D2"].model_copy(update={"seed": 44}))


@pytest.mark.parametrize("name", ["D2", "D3"])
def test_visible_ids_and_clocks_do_not_recover_hidden_streams(name: str, tmp_path: Path) -> None:
    world = generate(recipes()[name], SOFTWARE, TEXTS, example_curves())
    run_dir, export = tmp_path / "run", tmp_path / "export"
    # Two full robot-days exercise clock inference and permuted episode ids.
    jobs = [
        j.model_copy(update={"record_video": False})
        for j in world.jobs
        if j.identity.day == 0 and j.identity.robot_id in ("r01", "r02")
    ]
    record = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[])
    with RunWriter(run_dir, record, episodes_per_robot_day=14) as writer:
        writer.write_model(world.observed_fleet, world.task_classes)
        for job in jobs:
            data = episode_data(job)
            episode = EpisodeWriter(
                run_dir, job, task_text=TEXTS[job.identity.task], episodes_per_robot_day=14
            )
            for telemetry in data.steps:
                episode.add_step(telemetry)
            for step, state, images, executed in data.chain:
                episode.add_chain(step, state, images, executed)
            episode.set_fixture_hash(fixture_hash_of(job))
            episode.set_verdict(
                data.success,
                REASON_PLACED if data.success else REASON_STEP_LIMIT,
                2,
                data.steps[-1].step if data.success else None,
                placed_step=1 if data.success else None,
            )
            episode.finish(outcome=data.success)
        writer.finalize()
    export_runs([run_dir], export, fleet=world.observed_fleet, task_classes=world.task_classes)
    bundle = build_bundle(tmp_path / "bundle")
    assert not (bundle / "fleetlab/worlds/recipe.py").exists()
    for source in bundle.rglob("*.py"):
        assert str(world.spec.seed) not in source.read_text(), source
    result = attack(export, bundle)
    truth = {j.identity.episode_id: j.identity for j in jobs}
    assert result["guessed_world_seed"] != world.spec.seed
    assert len(result["episodes"]) == len(truth) == 28
    for row in result["episodes"]:
        ident = truth[row["episode_id"]]
        assert row["seeds"][0] != ident.scene_seed
        assert row["seeds"][1] != ident.noise_seed
    forbidden = [world.spec.world_id, str(world.spec.seed)]
    for path in export.rglob("*"):
        assert all(term not in str(path.relative_to(export)) for term in forbidden)
        if path.suffix == ".parquet":
            content = str(pq.read_table(path).to_pylist())
        elif path.is_file():
            content = path.read_text()
        else:
            continue
        assert all(term not in content for term in forbidden), path
    # Positive control: the same attack recovers every seed if the old coupling
    # is restored. This prevents an inert/miswired attack from passing the test.
    public_seed = result["guessed_world_seed"]
    for row in result["episodes"]:
        ident = truth[row["episode_id"]]
        assert tuple(row["seeds"]) == seeds_for(
            public_seed, ident.robot_id, ident.day, ident.episode_index
        )
