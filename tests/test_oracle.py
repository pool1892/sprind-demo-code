"""The oracle (src/fleetlab/oracle): planning, the table from job outcomes, the gains with
their bootstrap, the chance level, the detectability check and fresh replays, on a toy
table counted by hand, on the miniature fleet's fixtures, and end to end through the
engine's toy cell on the CPU.

The toy table: one site, three robots, twelve scoring episodes of day 2. r01 carries the
decoy f03 only; r02 a camera fault f01 and the decoy; r03 a delay f02 and the decoy.
Outcomes by hand (one episode is 100/12 points):

- e000001..4 (r01, [f03]): as run 1, 1, 1, 0; without f03 the same.
- e000005..8 (r02, [f01, f03]): as run 0, 0, 1, 0; without f01: 1, 1, 1, 0 (two
  rescued); without f03: unchanged; without both: 1, 1, 1, 1 (e000008 needs both gone).
- e000009..12 (r03, [f02, f03]): as run 0, 1, 0, 0; without f02: 1, 0, 1, 0 (two
  rescued, one broken); without f03: unchanged; without both: 1, 0, 1, 0.

So f01 nets +2, f02 +1, f03 0 (f03 with f01 gone breaks nothing and rescues e000008);
the pair f01, f03 has the interaction +1; achievable at k = 1 is f01's +2 and at k = 2
f01 and f02 together, +3.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, Budget, FixSet, cut
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.identity import (
    EpisodeIdentity,
    PolicySpec,
    SoftwareVersion,
    Task,
    seeds_for,
)
from fleetlab.contracts.oracle import (
    Gain,
    MissingReplays,
    OracleRow,
    OracleTable,
    achievable_gain,
    fault_effect,
    fault_list_hash,
    interaction,
    value_fix_set,
)
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, Verb
from fleetlab.contracts.world import DayRange, Patterns, Windows, WorldSpec
from fleetlab.faults import truth_table
from fleetlab.oracle.chance import chance_intervals
from fleetlab.oracle.detect import check_detectability, misses
from fleetlab.oracle.fresh import (
    fresh_jobs,
    fresh_proofs,
    labelled_fix_sets,
    needs_of_fix_sets,
    needs_of_oracle,
)
from fleetlab.oracle.gains import Gains
from fleetlab.oracle.plan import (
    PlanChanged,
    check_jobs,
    episode_replays,
    estimate_from_spec,
    plan_replays,
    plan_world,
    write_plan,
)
from fleetlab.oracle.valuation import best_removal, cheapest_fix, removal_options
from fleetlab.oracle.world import OracleLayout, OracleWorld
from fleetlab.worlds.truth import FaultTable

TRUTH = truth_table()
SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
POLICY = PolicySpec(
    name="xvla-libero",
    revision="12e8783e",
    precision="fp16_res32",
    control_mode=ControlMode.absolute,
    actions_executed_per_chunk=30,
    sampling_steps=10,
    image_views=4,
)
SEED = 7
ROBOTS = ["r01"] * 4 + ["r02"] * 4 + ["r03"] * 4

CAMERA = FaultSpec(
    fault_id="f01",
    kind="camera_pose_offset",
    level=6.0,
    target=Target(scope="r02", component=Component.scene_camera),
    schedule=Schedule(onset_day=0),
)
DELAY = FaultSpec(
    fault_id="f02",
    kind="action_latency",
    level=17.0,
    target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
    schedule=Schedule(onset_day=0),
    nominal=0,
)
DECOY = FaultSpec(
    fault_id="f03",
    kind="policy_knob",
    params={"value": 5},
    level=1.0,
    target=Target(scope="s1", config_key=ConfigKey.policy_sampling_steps),
    schedule=Schedule(onset_day=1),
    nominal=10,
)
FAULTS = {"r01": [DECOY], "r02": [CAMERA, DECOY], "r03": [DELAY, DECOY]}


def episode(number: int, robot: str, day: int = 2) -> EpisodeIdentity:
    index = (number - 1) % 4
    scene, noise = seeds_for(SEED, robot, day, index)
    faults = [f for f in FAULTS[robot] if f.schedule.active_on(day)]
    return EpisodeIdentity(
        world_id="w98",
        episode_id=f"e{number:06d}",
        robot_id=robot,
        site_id="s1",
        day=day,
        episode_index=index,
        suite="libero_spatial",
        task_index=number % 10,
        start_state=number % 50,
        scene_seed=scene,
        noise_seed=noise,
        policy=POLICY,
        batch_size=4,
        row=None,
        image_size=360,
        max_steps=280,
        faults=faults,
        software=SOFTWARE,
    )


SCORING = [episode(n, robot) for n, robot in enumerate(ROBOTS, 1)]
EARLIER = [
    episode(100 * (d + 1) + n, robot, d) for d in (0, 1) for n, robot in enumerate(ROBOTS, 1)
]

# Outcomes by removal set (module docstring); a missing entry means "as run".
AS_RUN = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 0, 10: 1, 11: 0, 12: 0}
WITHOUT: dict[frozenset[str], dict[int, int]] = {
    frozenset({"f01"}): {5: 1, 6: 1},
    frozenset({"f01", "f03"}): {5: 1, 6: 1, 8: 1},
    frozenset({"f02"}): {9: 1, 10: 0, 11: 1},
    frozenset({"f02", "f03"}): {9: 1, 10: 0, 11: 1},
}


def outcome(number: int, remove: frozenset[str]) -> bool:
    return bool(WITHOUT.get(remove, {}).get(number, AS_RUN[number]))


def row(identity: EpisodeIdentity, kept: list[FaultSpec], success: bool) -> OracleRow:
    return OracleRow(
        episode_id=identity.episode_id, active_faults=fault_list_hash(kept), success=success
    )


def rows(skip: frozenset[str] | None = None) -> list[OracleRow]:
    """As run, and every planned replay's outcome (the confirmation replays included)."""
    out = []
    for identity in SCORING:
        number = int(identity.episode_id[1:])
        out.append(row(identity, identity.faults, bool(AS_RUN[number])))
        plan = episode_replays(identity, {"f03"})
        for remove in [*plan.contract, *plan.confirmation]:
            if remove == skip:
                continue
            kept = [f for f in identity.faults if f.fault_id not in remove]
            out.append(row(identity, kept, outcome(number, remove)))
    return out


def toy_table(skip: frozenset[str] | None = None) -> OracleTable:
    return OracleTable(SCORING, rows(skip), excluded={"f03"})


def world_spec(schedule: list[FaultSpec] | None = None, **patterns: Any) -> WorldSpec:
    return WorldSpec(
        world_id="w98",
        seed=SEED,
        size="small",
        sites=1,
        robots_per_site=3,
        days=3,
        windows=Windows(
            observed=DayRange(first=0, last=0),
            experiment=DayRange(first=1, last=1),
            scoring=DayRange(first=2, last=2),
        ),
        fault_schedule=schedule or [CAMERA, DELAY, DECOY],
        patterns=Patterns(decoys=["f03"], **patterns),
        policy=POLICY,
        tasks=[Task("libero_spatial", 0)],
        episodes_per_robot_day=4,
    )


def toy_world() -> OracleWorld:
    return OracleWorld(spec=world_spec(), identities=tuple(EARLIER + SCORING))


# ---- planning ----------------------------------------------------------------------------------


def test_the_power_set_and_its_confirmation() -> None:
    plan = episode_replays(SCORING[4], {"f03"})  # r02: f01 toggled, the decoy f03 held
    assert plan.toggled == ("f01",) and plan.excluded == ("f03",)
    assert plan.contract == (frozenset({"f01"}),)
    assert set(plan.confirmation) == {frozenset({"f01", "f03"}), frozenset({"f03"})}
    bare = episode_replays(SCORING[4], {"f03"}, confirm=False)
    assert bare.confirmation == ()
    two = SCORING[4].model_copy(
        update={
            "faults": [
                CAMERA,
                DELAY.model_copy(
                    update={
                        "target": Target(
                            scope="r02", config_key=ConfigKey.control_action_delay_steps
                        )
                    }
                ),
            ]
        }
    )
    both = episode_replays(two, ())
    assert len(both.contract) == 3 and both.confirmation == ()


def test_the_cap_and_the_counts() -> None:
    extra = [
        FaultSpec(
            fault_id=f"f1{i}",
            kind="gripper_wear",
            level=0.5,
            params={"friction_scale": 0.8},
            target=Target(scope="r02", component=Component.gripper),
            schedule=Schedule(onset_day=0),
        )
        for i in range(4)
    ]
    crowded = SCORING[5].model_copy(update={"faults": [CAMERA, *extra, DECOY]})
    plan = plan_replays([SCORING[0], crowded, SCORING[8]], {"f03"}, proof=True)
    assert plan.over_cap == [crowded.episode_id]
    assert plan.contract == 0 + 0 + 1 and plan.confirmation == 1 + 0 + 2
    assert len(plan.proof) == 3 and all(not j.remove for j in plan.proof)
    full = plan_replays(SCORING, {"f03"})
    assert (full.contract, full.confirmation) == (8, 4 + 8 + 8)
    assert full.toggled_histogram() == {0: 4, 1: 8}
    assert full.counts()["per_scoring_episode"] == round(28 / 12, 2)


def test_a_manifest_that_does_not_belong_to_the_world_is_caught() -> None:
    world = toy_world()
    plan = plan_world(world)
    assert check_jobs(world, plan.manifest()) == []
    job = plan.jobs[0]
    moved = job.identity.model_copy(update={"start_state": job.identity.start_state + 1})
    reseeded = job.identity.model_copy(update={"noise_seed": job.identity.noise_seed + 1})
    bad = [
        Job(identity=moved, mode=JobMode.replay, remove=job.remove),
        job,
        job,
        Job(identity=EARLIER[0], mode=JobMode.replay),
    ]
    problems = check_jobs(world, bad)
    assert any("differs from the world's manifest" in p for p in problems)
    assert any("twice" in p for p in problems)
    assert any("not a scoring episode" in p for p in problems)
    seeded_world = OracleWorld(
        spec=world.spec, identities=tuple([*EARLIER, reseeded, *SCORING[1:]])
    )
    assert any(
        "seeds_for" in p
        for p in check_jobs(seeded_world, [Job(identity=reseeded, mode=JobMode.replay)])
    )


def test_the_manifest_on_disk_is_resumable(tmp_path: Path) -> None:
    world = toy_world()
    layout = OracleLayout(tmp_path / "oracle")
    plan = plan_world(world)
    assert write_plan(layout, world, plan) is True
    assert write_plan(layout, world, plan) is False  # the same plan: nothing written
    record = json.loads(layout.plan.read_text())
    assert record["replays"] == 28 and record["proof"] == 12 and record["decoys"] == ["f03"]
    other = plan_world(world, confirm=False)
    with pytest.raises(PlanChanged):
        write_plan(layout, world, other)
    assert write_plan(layout, world, other, replace=True) is True


def test_the_estimate_from_a_spec() -> None:
    estimate = estimate_from_spec(world_spec())
    assert estimate["scoring_episodes"] == 12
    assert estimate["contract"] == 8 and estimate["replays"] == 28 and estimate["proof"] == 12
    assert estimate_from_spec(world_spec(), confirm=False)["replays"] == 8


# ---- gains -------------------------------------------------------------------------------------


def test_the_gains_are_the_contract_s() -> None:
    table = toy_table()
    gains = Gains(table, boot=200, seed=1)
    for fault_id, net in (("f01", 2), ("f02", 1), ("f03", 0)):
        mine = gains.fault_effect(fault_id)
        assert mine.net == net == fault_effect(fault_id, table).net
        assert (mine.rescued, mine.broken) == (
            fault_effect(fault_id, table).rescued,
            fault_effect(fault_id, table).broken,
        )
        assert mine.low is not None and mine.high is not None
        assert mine.low <= mine.points <= mine.high  # type: ignore[operator]
    assert gains.fault_effect("f02").broken == 1
    pair = gains.interaction("f01", "f03", marked=True)
    assert pair.net == interaction("f01", "f03", table) == 1
    for k in (1, 2, 3):
        mine = gains.achievable_count(k)
        theirs = achievable_gain(k, table)
        assert (mine.rescued, mine.broken, mine.net) == (theirs.rescued, theirs.broken, theirs.net)
    assert gains.achievable_count(1).detail["faults"] == ["f01"]
    assert gains.achievable_count(2).detail["faults"] == ["f01", "f02"]


def test_the_achievable_gain_under_hours_is_the_scorer_s() -> None:
    table = toy_table()
    gains = Gains(table, boot=0)
    specs = {"f01": CAMERA, "f02": DELAY, "f03": DECOY}
    options, needs = removal_options(table)
    assert needs == []
    costs = {}
    for fault_id in ("f01", "f02"):
        found = cheapest_fix(specs[fault_id], TRUTH, {**_costs()})
        assert found is not None
        costs[fault_id] = found[1]
    for hours in (0.1, 0.5, 4.0):
        mine = gains.achievable_cost(hours, specs, TRUTH, _costs())
        theirs, chosen = best_removal(options, costs, hours)
        assert mine.net == theirs.net and tuple(mine.detail["faults"]) == chosen
    assert gains.achievable_cost(0.1, specs, TRUTH, _costs()).detail["faults"] == ["f02"]


def _costs() -> dict[Any, Any]:
    from fleetlab.contracts.actions import COSTS

    return dict(COSTS)


def test_the_bootstrap_is_seeded_and_holds_the_point() -> None:
    a = Gains(toy_table(), boot=300, seed=4).fault_effect("f01")
    b = Gains(toy_table(), boot=300, seed=4).fault_effect("f01")
    c = Gains(toy_table(), boot=300, seed=5).fault_effect("f01")
    assert (a.low, a.high) == (b.low, b.high)
    assert a.low is not None and c.low is not None
    best = Gains(toy_table(), boot=300, seed=4).achievable_count(2)
    assert best.low is not None and best.high is not None and best.low <= best.points <= best.high  # type: ignore[operator]
    none = Gains(toy_table(), boot=0).fault_effect("f01")
    assert none.low is None and none.net == 2


def test_every_fault_removed_per_episode() -> None:
    estimate, episodes = Gains(toy_table(), boot=0).all_faults()
    assert (estimate.rescued, estimate.broken, estimate.net) == (5, 1, 4)
    by_id = {r["episode_id"]: r for r in episodes}
    assert by_id["e000004"]["all_removed_success"] is False  # r01's own failure
    assert by_id["e000008"]["all_removed_success"] is True
    assert by_id["e000012"]["all_removed_success"] is False
    assert sum(r["all_removed_success"] is False for r in episodes) == 3


def test_a_missing_lookup_is_reported_not_guessed() -> None:
    table = toy_table(skip=frozenset({"f01", "f03"}))
    gains = Gains(table, boot=50)
    assert gains.fault_effect("f01").net == 2  # does not need the joint row
    pair = gains.interaction("f01", "f03", marked=False)
    assert pair.net is None and pair.missing and "4 episodes" in pair.missing
    assert gains.achievable_count(2).missing is None  # the decoy is not a candidate
    estimate, _ = gains.all_faults()
    assert estimate.net is None and estimate.missing
    needs = needs_of_oracle(table)
    assert {n.episode_id for n in needs.unique()} == {s.episode_id for s in SCORING[4:8]}


def test_the_chance_level_counts_draws_that_need_a_fresh_replay() -> None:
    entries = [
        Action(verb=Verb.recalibrate, target=Target(scope="r02", component=Component.scene_camera)),
        Action(
            verb=Verb.revert,
            target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
        ),
        Action(verb=Verb.clean, target=Target(scope="r01", component=Component.gripper)),
        Action(
            verb=Verb.set,
            target=Target(scope="r01", config_key=ConfigKey.control_action_delay_steps),
            value=5,
        ),
    ]
    current: dict[ConfigKey | ConditionKey, ConfigValue] = {ConfigKey.control_action_delay_steps: 0}
    chance = chance_intervals(
        toy_table(),
        TRUTH,
        lambda identity: current,
        entries,
        seed=3,
        achievable={"k = 3": 3},
        draws=40,
    )
    assert [c.budget for c in chance] == ["k = 3", "k = 6", "4 h", "8 h"]
    first = chance[0]
    assert first.draws == 40 and first.valued + first.skipped_fresh_replay == 40
    assert first.skipped_fresh_replay > 0  # the set to 5 adds a change
    assert first.mean_share is None and first.p95_share is None
    assert first.bounds is not None and first.missing is not None


# ---- detectability -----------------------------------------------------------------------------


def test_the_detectability_check() -> None:
    gains = Gains(toy_table(), boot=100)
    effects = {f: gains.fault_effect(f) for f in ("f01", "f02", "f03")}
    world = toy_world()
    table = _fault_table(world, detectable=("f01", "f02"))
    rows_ = {r.item: r for r in check_detectability(table, effects)}
    assert rows_["f01"].verdict == "ok"  # +2 of 12 = 16.7 points
    assert rows_["f02"].verdict == "ok"  # +1 of 12 = 8.3 points
    assert rows_["f03"].verdict == "decoy"
    weak = {**effects, "f02": effects["f02"].model_copy(update={"points": 1.0})}
    found = misses(check_detectability(table, weak))
    assert [r.item for r in found] == ["f02"] and found[0].verdict == "miss"
    small = _fault_table(world, detectable=("f01",))
    assert {r.item: r.verdict for r in check_detectability(small, effects)}["f02"] == "stronger"
    from fleetlab.oracle.world import Cause

    causes = [
        Cause(
            "f01+f02", ("f01", "f02"), ("interaction", "interaction"), True, ("scoring",), 3.0, True
        ),
        Cause("f05", ("f05",), ("plain",), True, ("observed",), 2.5, False),
    ]
    joint = {"f01+f02": gains.estimate("f01+f02", "cause", *gains.removal({"f01", "f02"}), 8)}
    rows_ = {r.item: r for r in check_detectability(table, effects, causes, joint)}
    assert rows_["f01+f02"].verdict == "ok" and rows_["f01+f02"].oracle_points == 25.0
    assert rows_["f05"].verdict == "absent"  # no scoring episode carries it
    assert "f01" not in rows_ and rows_["f03"].verdict == "decoy"  # the causes stand for faults
    assert [r.item for r in misses(list(rows_.values()))] == ["f05"]


def _fault_table(world: OracleWorld, detectable: tuple[str, ...]) -> FaultTable:
    from fleetlab.oracle import toy

    table = toy.fault_table([Job(identity=i) for i in world.identities], detectable=detectable)
    by_id = {f.fault_id: f for f in world.spec.fault_schedule}
    return table.model_copy(
        update={
            "faults": [
                f.model_copy(
                    update={"kind": by_id[f.fault_id].kind, "target": by_id[f.fault_id].target}
                )
                for f in table.faults
            ]
        }
    )


# ---- fresh replays -----------------------------------------------------------------------------


def test_an_added_change_needs_a_fresh_replay_then_is_a_lookup() -> None:
    table = toy_table()
    half = Action(
        verb=Verb.set,
        target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
        value=8,
    )
    fix_set = FixSet(actions=[half], budget=Budget(kind="count", amount=1))
    with pytest.raises(MissingReplays) as caught:
        value_fix_set(fix_set, table, TRUTH)
    assert {n.episode_id for n in caught.value.needs} == {s.episode_id for s in SCORING[8:]}
    needs = needs_of_fix_sets([("half", fix_set)], table, TRUTH, None)
    scoring = {s.episode_id: s for s in SCORING}
    jobs, refused = fresh_jobs(scoring, needs.unique())
    assert refused == [] and len(jobs) == 4
    for job in jobs:
        assert job.remove == ["f02"] and len(job.add) == 1
        assert job.add[0].kind == "config_change" and job.add[0].params == {"value": 8}
    assert len(fresh_proofs(jobs)) == 4
    # The fresh replays ran: delay 8 rescues e000009 and e000011, as removing it would.
    for job in jobs:
        number = int(job.identity.episode_id[1:])
        table.add(row(job.identity, job.effective_faults, number in (9, 10, 11)))
    assert value_fix_set(fix_set, table, TRUTH) == Gain.of(2, 0)


def test_an_episode_above_the_cap_needs_fresh_replays() -> None:
    extra = [
        FaultSpec(
            fault_id=f"f1{i}",
            kind="gripper_wear",
            level=0.5,
            params={"friction_scale": 0.8},
            target=Target(scope="r02", component=Component.gripper),
            schedule=Schedule(onset_day=0),
        )
        for i in range(4)
    ]
    crowded = SCORING[5].model_copy(update={"faults": [CAMERA, *extra, DECOY]})
    table = OracleTable([crowded], [row(crowded, crowded.faults, False)], excluded={"f03"})
    assert table.over_cap() == [crowded.episode_id]
    fix = FixSet(
        actions=[
            Action(
                verb=Verb.recalibrate, target=Target(scope="r02", component=Component.scene_camera)
            )
        ],
        budget=Budget(kind="count", amount=1),
    )
    with pytest.raises(MissingReplays):
        value_fix_set(fix, table, TRUTH)
    jobs, _ = fresh_jobs(
        {crowded.episode_id: crowded}, needs_of_fix_sets([("x", fix)], table, TRUTH, None).unique()
    )
    assert len(jobs) == 1 and jobs[0].remove == ["f01"] and jobs[0].add == []


def test_a_missing_as_run_row_is_not_a_fresh_replay() -> None:
    from fleetlab.contracts.oracle import ReplayNeed

    jobs, refused = fresh_jobs(
        {SCORING[4].episode_id: SCORING[4]},
        [ReplayNeed(SCORING[4].episode_id, tuple(SCORING[4].faults))],
    )
    assert jobs == [] and "has no row" in refused[0]


def test_a_ranked_list_stands_for_its_actions_and_its_cuts() -> None:
    ranked = {
        "actions": [
            {
                "action": Action(
                    verb=Verb.recalibrate,
                    target=Target(scope="r02", component=Component.scene_camera),
                ).model_dump(mode="json"),
                "expected_gain": 1.0,
                "uncertainty": {"low": 0.0, "high": 2.0},
                "rationale": "x",
            },
            {
                "action": Action(
                    verb=Verb.revert,
                    target=Target(scope="r03", config_key=ConfigKey.control_action_delay_steps),
                ).model_dump(mode="json"),
                "expected_gain": 0.5,
                "uncertainty": {"low": 0.0, "high": 1.0},
                "rationale": "y",
            },
        ]
    }
    labelled = labelled_fix_sets("fix_set.json", ranked)
    labels = [label for label, _ in labelled]
    assert labels[:2] == ["fix_set.json: ranked action 1", "fix_set.json: ranked action 2"]
    assert len(labelled) == 2 + len(SCORING_BUDGETS)


# ---- the miniature fleet's fixtures --------------------------------------------------------------

MINI = Path.home() / ".cache" / "robots-fixtures" / "mini-v1"
needs_mini = pytest.mark.skipif(
    not (MINI / "world" / "manifest.jsonl").is_file(), reason="mini-v1 fixtures not on this machine"
)


@pytest.fixture(scope="module")
def mini() -> Any:
    from fleetlab.oracle.steps import Oracle

    return Oracle.open(MINI, oracle_dir=Path("/nonexistent-oracle-dir"), proof=False)


@needs_mini
def test_mini_the_plan_is_the_integration_s(mini: Any) -> None:
    from fleetlab.log.runfiles import read_table

    plan = mini.plan()
    assert (plan.contract, plan.confirmation, plan.over_cap) == (65, 65, [])
    assert plan.replays == 130 and len(plan.episodes) == 104
    assert check_jobs(mini.world, plan.jobs) == []
    ran = read_table(MINI / "runs" / "w10-later", "job_outcomes").to_pylist()
    replayed = {r["job_key"] for r in ran if r["mode"] == "replay" and r["removed"] != "[]"}
    assert {j.key for j in plan.jobs} == replayed
    removed = Counter(tuple(j.remove) for j in plan.jobs)
    assert removed[("f02",)] == 52 and removed[("f03",)] == 52
    with_proof = plan_world(mini.world)
    assert len(with_proof.proof) == 21 and check_jobs(mini.world, with_proof.manifest()) == []
    estimate = estimate_from_spec(mini.world.spec)
    assert estimate["replays"] == 130 and estimate["scoring_episodes"] == 104


@needs_mini
def test_mini_the_table_from_job_outcomes(mini: Any) -> None:
    build = mini.table()
    assert build.rows_held == 234 and build.problems() == []
    table = build.table
    assert len(table.identities) == 104
    gains = Gains(table, boot=200, seed=10)
    effects = {f: gains.fault_effect(f) for f in ("f01", "f02", "f03")}
    assert [(e.rescued, e.broken, e.net) for e in effects.values()] == [
        (8, 1, 7),
        (41, 0, 41),
        (0, 1, -1),
    ]
    assert effects["f03"].zero_within_noise  # the decoy, confirmed by replay
    assert gains.interaction("f01", "f03", marked=False).net == 1
    assert [gains.achievable_count(k).net for k in (1, 2, 3, 6)] == [41, 48, 48, 48]


@needs_mini
def test_mini_the_two_fix_sets_reproduce_the_integration_s_gains(mini: Any) -> None:
    from fleetlab.contracts.tools import RankedFixSet
    from fleetlab.oracle.valuation import value_or_missing

    table = mini.table().table
    truth, current = mini.world.truth(), mini.world.current()
    for condition in ("logs", "experiments"):
        report = json.loads((MINI / "score" / condition / "report.json").read_text())
        step = json.loads((MINI / "steps" / f"analysis-{condition}.json").read_text())
        analysis = MINI / "analyses" / step["analysis_id"] / "outbox" / "fix_set.json"
        ranked = RankedFixSet.model_validate_json(analysis.read_text())
        actions = [r.action for r in ranked.actions]
        for budget, scored in zip(SCORING_BUDGETS, report["claim3"]["budgets"], strict=True):
            fix_set = cut(actions, budget)
            assert fix_set.model_dump(mode="json")["actions"] == scored["agent"]["actions"]
            valued = value_or_missing(fix_set, table, truth, current)
            assert valued.gain is not None
            assert valued.gain.model_dump() == scored["agent"]["gain"]
            assert valued.gain.net == 48
        for single, scored in zip(actions, report["claim2"]["actions"], strict=True):
            if scored["true_points"] is None:
                continue
            fix = FixSet(actions=[single], budget=Budget(kind="count", amount=1))
            gain = value_fix_set(fix, table, truth, current)
            assert 100.0 * gain.net / 104 == pytest.approx(scored["true_points"])


@needs_mini
def test_mini_chance_values_every_draw_with_mechanism_equivalence(mini: Any) -> None:
    table = mini.table().table
    chance = chance_intervals(
        table,
        mini.world.truth(),
        mini.world.current(),
        mini.world.catalogue(),
        seed=mini.world.spec.seed,
        achievable={"k = 3": 48, "k = 6": 48, "4 h": 48, "8 h": 48},
    )
    # The stored scorer demanded inert-addition replays. Recompute independently
    # through uncached contract resolution rather than compare its historical gaps.
    expected: list[list[int]] = [[] for _ in SCORING_BUDGETS]
    pool = [a for a in mini.world.catalogue() if a.is_complete]
    rng = np.random.default_rng(mini.world.spec.seed)
    for _ in range(200):
        shuffled = [pool[int(i)] for i in rng.permutation(len(pool))]
        for index, budget in enumerate(SCORING_BUDGETS):
            expected[index].append(
                value_fix_set(
                    cut(shuffled, budget), table, mini.world.truth(), mini.world.current()
                ).net
            )
    for mine, nets in zip(chance, expected, strict=True):
        assert mine.draws == mine.valued == 200
        assert mine.skipped_fresh_replay == mine.skipped_error == 0
        assert mine.missing is None and mine.bounds is None
        pts = 100 * np.array(nets) / len(table.identities)
        assert mine.mean_points == round(float(pts.mean()), 3)
        assert mine.p95_points == round(float(np.percentile(pts, 95)), 3)


@needs_mini
def test_mini_the_report(mini: Any, tmp_path: Path) -> None:
    from fleetlab.oracle.report import compute_report, parquet_rows, summary_markdown

    report, episodes = compute_report(mini.world, mini.table(), mini.plan(), boot=200, draws=50)
    assert report.rows == 234 and report.scoring_episodes == 104
    assert [p.verdict for p in report.patterns] == ["confirmed"]  # the decoy
    assert [d.verdict for d in report.detectability] == ["ok", "ok", "decoy"]
    assert report.achievable_at("k=3") is not None and report.achievable_at("k=3").net == 48  # type: ignore[union-attr]
    assert {e.item for e in report.achievable} == {
        "k=1",
        "k=2",
        "k=3",
        "k=4",
        "k=5",
        "k=6",
        "4 h",
        "8 h",
    }
    assert report.all_faults.net == 48 and report.still_failing == 2 and len(episodes) == 104
    items = {(r["item"], r["item_kind"]) for r in parquet_rows(report)}
    assert ("f02", "fault_effect") in items and ("all faults", "fix_set") in items
    assert "| f02 | effective | 52 | 41 | 0 | 41 | +39.42 |" in summary_markdown(report)


# ---- end to end through the engine's toy cell ----------------------------------------------------


def test_the_toy_world_end_to_end(tmp_path: Path) -> None:
    from fleetlab.engine import software
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.oracle import toy
    from fleetlab.oracle.report import report_json
    from fleetlab.oracle.steps import Oracle

    root = tmp_path / "w99"
    manifest = toy.write_world(root, software.current("none"))
    config = RunConfig(
        out=str(root / "runs" / "toy"),
        batch_size=4,
        episodes_per_robot_day=4,
        **toy.engine_overrides(),
    )
    assert run(config, manifest).record.chain_ok is True
    oracle = Oracle.open(root, registry=toy.REGISTRY)
    plan, written, _ = oracle.step_plan()
    assert written and (plan.contract, plan.confirmation, len(plan.proof)) == (8, 20, 12)
    world_table = root / "runs" / "toy" / "truth" / "oracle.parquet"  # the engine's, empty
    before = world_table.read_bytes()
    early = oracle.step_report(boot=0, draws=0)  # before the replays: no run is written
    assert early.fault_effects[0].missing and world_table.read_bytes() == before
    info = oracle.step_run(here=True, overrides=toy.engine_overrides())
    assert info["chain_ok"] is True and info["proof"]["identical"] == 12
    record = json.loads((oracle.layout.run / "run.json").read_text())
    assert record["chain_ok"] is True and len(record["chain_sample"]) == 12
    again = oracle.step_run(here=True, overrides=toy.engine_overrides())
    assert again["jobs_run_now"] == 0  # resumable: nothing left to run

    build = oracle.table()
    assert build.problems() == [] and build.rows_held == 12 + 28
    table = build.table
    report = oracle.step_report(boot=100, draws=20)
    for effect in report.fault_effects:
        assert effect.net == fault_effect(effect.item, table).net
    assert report.effect("f03") is not None and report.effect("f03").net == 0  # type: ignore[union-attr]
    assert report_json(oracle.layout.report) == report
    assert (oracle.layout.run / "truth" / "oracle.parquet").is_file()
    assert oracle.layout.summary.read_text().startswith("# Oracle of world w99")

    # A set to a non-nominal value on a healthy robot adds a change: a fresh replay.
    slow = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.control_action_delay_steps),
        value=25,
    )
    fix_set = FixSet(actions=[slow], budget=Budget(kind="count", amount=1))
    truth, current = oracle.world.truth(), oracle.world.current()
    with pytest.raises(MissingReplays):
        value_fix_set(fix_set, table, truth, current)
    record_ = oracle.step_fresh_plan("slow", [("slow", fix_set)])
    assert record_["jobs"] == 4 and record_["proof"] == 4 and record_["refused"] == []
    fresh = oracle.step_fresh_run("slow", here=True, overrides=toy.engine_overrides())
    assert fresh["chain_ok"] is True
    values = oracle.step_fresh_value("slow")
    assert values[0]["label"] == "slow" and "net" in values[0], values
    gain = value_fix_set(fix_set, oracle.table().table, truth, current)
    assert values[0]["net"] == gain.net
    as_run = [j for j in manifest if j.identity.robot_id == "r01" and j.identity.day == 2]
    assert gain.broken <= len(as_run)
    assert np.isfinite(values[0]["points"])
    again_ = oracle.step_fresh_plan("slow", [("slow", fix_set)])  # after the run: kept
    assert again_["jobs"] == 4 and again_["proof"] == 4

    # Provenance: a re-plan over replays that ran is refused; a run whose manifest changed
    # or that was made for another world gives no rows; a failed proof is not run again.
    with pytest.raises(PlanChanged):
        write_plan(
            oracle.layout, oracle.world, plan_world(oracle.world, confirm="none"), replace=True
        )
    stored = oracle.layout.jobs.read_text()
    oracle.layout.jobs.write_text(stored.replace('"start_state":', '"start_state": ', 1))
    refused = {str(r.run_dir): r.refused for r in oracle.table().runs}
    assert "changed after the run" in str(refused[str(oracle.layout.run)])
    oracle.layout.jobs.write_text(stored)
    other = Oracle.open(root, registry=toy.REGISTRY)
    other.world = OracleWorld(**{**other.world.__dict__, "world_hash": "0" * 64})
    assert "made for world hash" in str({r.refused for r in other.table().runs})
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.log import RunWriter
    from fleetlab.oracle.run import RunRefused

    run_record = RunRecord.model_validate_json((oracle.layout.run / "run.json").read_text())
    with RunWriter(oracle.layout.run, run_record, episodes_per_robot_day=4) as writer:
        writer.update_record(chain_ok=False)
    with pytest.raises(RunRefused, match="the proof failed"):
        oracle.step_run(here=True, overrides=toy.engine_overrides())


def test_the_patterns_the_oracle_confirms() -> None:
    from fleetlab.contracts.faults import Drift
    from fleetlab.oracle.gains import Estimate
    from fleetlab.oracle.report import pattern_checks

    inert = DELAY.model_copy(
        update={
            "fault_id": "f04",
            "target": Target(scope="r01", config_key=ConfigKey.camera_scene_calibration),
            "nominal": [0.0],
            "schedule": Schedule(onset_day=1),
        }
    )
    drifting = CAMERA.model_copy(
        update={"schedule": Schedule(onset_day=0, drift=Drift(shape="linear", rate=1.0))}
    )
    spec = world_spec(
        [drifting, DELAY, DECOY, inert],
        interactions=[("f01", "f02")],
        confounders=[("f03", "f04")],
        drifts=["f01"],
    )
    world = OracleWorld(spec=spec, identities=tuple(SCORING))

    def effect(
        item: str, net: int, low: float, high: float, kind: str = "fault_effect", **detail: Any
    ) -> Estimate:
        return Estimate(
            item=item,
            item_kind=kind,
            rescued=max(net, 0),
            broken=max(-net, 0),
            net=net,
            points=net * 1.0,
            low=low,
            high=high,
            episodes=4,
            detail=detail,
        )

    parts = {
        "fault_effects": [
            effect("f01", 3, 1.0, 5.0),
            effect("f02", 1, -1.0, 3.0),
            effect("f03", 3, 1.0, 5.0),
            effect("f04", 0, -1.0, 1.0),
        ],
        "interactions": [
            effect("f01+f02", -2, -4.0, -0.5, "interaction", joint_net=2, alone_net=[2, 2])
        ],
        "by_day": [
            effect("f01@8", 1, 0.0, 2.0, "fault_effect_day", fault_id="f01", day=8),
            effect("f01@9", 3, 1.0, 5.0, "fault_effect_day", fault_id="f01", day=9),
        ],
    }
    checks = {c.pattern: c for c in pattern_checks(world, parts)}
    assert checks["decoy"].verdict == "not confirmed"  # f03 here gains 3 points: not a decoy
    assert checks["confounder"].verdict == "confirmed"  # f04 inert, f03 effective
    assert checks["interaction"].verdict == "confirmed"
    assert checks["drift"].verdict == "confirmed" and "day 9" in checks["drift"].text
    weak = {
        **parts,
        "interactions": [
            effect("f01+f02", -1, -2.0, 0.5, "interaction", joint_net=3, alone_net=[2, 2])
        ],
    }
    assert {c.pattern: c for c in pattern_checks(world, weak)}[
        "interaction"
    ].verdict == "not confirmed"


@needs_mini
def test_mini_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from fleetlab.oracle.__main__ import main

    out = tmp_path / "oracle"
    assert main(["plan", str(MINI), "--oracle-dir", str(out)]) == 0
    record = json.loads((out / "plan.json").read_text())
    assert (record["contract"], record["confirmation"], record["proof"]) == (65, 65, 21)
    assert main(["plan", str(MINI), "--oracle-dir", str(out)]) == 0  # the same plan again
    assert main(["check", str(MINI), "--oracle-dir", str(out)]) == 0
    assert main(["estimate", str(MINI / "world" / "spec.json")]) == 0
    assert '"replays": 130' in capsys.readouterr().out
    args = ["report", str(MINI), "--oracle-dir", str(out), "--no-truth-table"]
    # Legacy gains remain readable, but this stored run has no complete baseline
    # verification ledger, so its new measured-cause record is incomplete.
    assert main([*args, "--boot", "50", "--draws", "20"]) == 2
    report = json.loads((out / "oracle.json").read_text())
    assert not report["measured_causes"]["complete"]
    assert any("baselines not executed twice" in p for p in report["problems"])
    assert (out / "summary.md").is_file() and (out / "episodes.parquet").is_file()
    assert not (out / "runs").exists()  # nothing ran, nothing was written into a run


def test_the_confirmation_modes() -> None:
    from fleetlab.oracle.plan import confirmations

    single = episode_replays(SCORING[4], {"f03"}, confirm="single")  # r02: f01 toggled, f03 held
    assert set(single.confirmation) == {frozenset({"f03"}), frozenset({"f01", "f03"})}
    alone = episode_replays(SCORING[0], {"f03"}, confirm="single")  # r01: only the decoy
    assert alone.confirmation == (frozenset({"f03"}),)
    assert episode_replays(SCORING[4], {"f03"}, confirm="none").confirmation == ()
    for mode in ("full", "single", "none"):
        plan = plan_replays(SCORING, {"f03"}, confirm=mode)
        estimate = estimate_from_spec(world_spec(), confirm=mode)
        assert estimate["replays"] == plan.replays, mode
        by_shape = sum(
            confirmations(len(e.toggled), len(e.excluded), plan.confirm) for e in plan.episodes
        )
        assert by_shape == plan.confirmation, mode
    with pytest.raises(ValueError):
        episode_replays(SCORING[0], {"f03"}, confirm="some")


def test_the_worker_count_fits_the_memory_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.engine import fleet
    from fleetlab.oracle.run import auto_workers

    for available, workers in ((40.0, 10), (24.4, 5), (17.0, 2), (9.0, 1)):
        monkeypatch.setattr(fleet, "mem_available_gib", lambda a=available: a)
        assert auto_workers() == workers, available
        config = fleet.RunConfig(out="")
        need = workers * config.worker_gib + config.server_gib
        assert available - need >= config.mem_floor_gib or workers == 1


def test_a_growing_manifest_survives_an_interrupted_extension(tmp_path: Path) -> None:
    from fleetlab.contracts.rollout import RunRecord
    from fleetlab.engine import software
    from fleetlab.engine.fleet import RunConfig, run
    from fleetlab.log import RunWriter
    from fleetlab.oracle import toy
    from fleetlab.oracle.run import proof_failed, run_here
    from fleetlab.oracle.table import build_table

    root = tmp_path / "w99"
    manifest = toy.write_world(root, software.current("none"))
    world_run = root / "runs" / "toy"
    config = RunConfig(
        out=str(world_run), batch_size=4, episodes_per_robot_day=4, **toy.engine_overrides()
    )
    assert run(config, manifest).record.chain_ok is True
    scoring = [j.identity for j in manifest if j.identity.day == 2]
    r01 = [i for i in scoring if i.robot_id == "r01"]
    first = [Job(identity=i, mode=JobMode.replay, remove=["f03"]) for i in r01[:2]]
    more = [Job(identity=i, mode=JobMode.replay, remove=["f03"]) for i in r01[2:]]
    proof = [Job(identity=r01[0], mode=JobMode.replay)]
    fresh, listing = tmp_path / "fresh-run", tmp_path / "fresh-manifest.jsonl"
    args = dict(
        manifest_file=listing, world_hash=None, extend=True, overrides=toy.engine_overrides()
    )
    assert run_here(fresh, [*first, *proof], [world_run], 4, **args)["chain_ok"] is True  # type: ignore[arg-type]
    record = json.loads((fresh / "run.json").read_text())
    assert not Path(record["extra"]["oracle"]["manifest"]).is_absolute()

    # The manifest grew, then the run stopped before the engine (the memory guard, a kill).
    grown = [*first, *proof, *more]
    listing.write_text("".join(j.model_dump_json() + "\n" for j in grown))
    build = build_table(scoring, {"f03"}, [world_run, fresh])
    rows = {str(r.run_dir): r for r in build.runs}
    assert rows[str(fresh)].refused is None and rows[str(fresh)].replays == 2
    again = run_here(fresh, grown, [world_run], 4, **args)  # type: ignore[arg-type]
    assert again["chain_ok"] is True and again["jobs_run_now"] == 2
    build = build_table(scoring, {"f03"}, [world_run, fresh])
    assert {str(r.run_dir): r.replays for r in build.runs}[str(fresh)] == 4

    # A proof that failed only for want of an as-run chain may be made again.
    done = RunRecord.model_validate_json((fresh / "run.json").read_text())
    missing = {
        "compared": 1,
        "identical": 0,
        "details": [
            {"episode_id": "x", "identical": False, "what": "no as-run chain in the world's runs"}
        ],
    }
    with RunWriter(fresh, done, episodes_per_robot_day=4) as writer:
        writer.update_record(chain_ok=False, extra={"chain_proof": missing})
    assert proof_failed(fresh, grown) is None
    differs = {
        **missing,
        "details": [{"episode_id": "x", "identical": False, "what": "state_hash"}],
    }
    with RunWriter(fresh, done, episodes_per_robot_day=4) as writer:
        writer.update_record(chain_ok=False, extra={"chain_proof": differs})
    assert proof_failed(fresh, grown) is not None


def test_data_and_observed_only_causes_are_listed_not_missed() -> None:
    from fleetlab.oracle.world import Cause

    gains = Gains(toy_table(), boot=0)
    effects = {f: gains.fault_effect(f) for f in ("f01", "f02", "f03")}
    weak = {**effects, "f02": effects["f02"].model_copy(update={"points": 0.5})}
    causes = [
        Cause("f01", ("f01",), ("plain",), True, ("observed",), 1.0, True, data=True),
        Cause("f02", ("f02",), ("plain",), True, ("observed",), 1.0, True),
    ]
    rows_ = {r.item: r for r in check_detectability(None, weak, causes)}
    assert rows_["f01"].verdict == "data" and rows_["f02"].verdict == "observed-only"
    assert misses(list(rows_.values())) == []
    scored = [Cause("f02", ("f02",), ("plain",), True, ("scoring",), 1.0, True)]
    assert [r.verdict for r in misses(check_detectability(None, weak, scored))] == ["miss"]


def test_a_small_drift_rise_is_weak() -> None:
    from fleetlab.contracts.faults import Drift
    from fleetlab.oracle.gains import Estimate
    from fleetlab.oracle.report import pattern_checks

    drifting = CAMERA.model_copy(
        update={"schedule": Schedule(onset_day=0, drift=Drift(shape="linear", rate=1.0))}
    )
    world = OracleWorld(
        spec=world_spec([drifting, DELAY, DECOY], drifts=["f01"]), identities=tuple(SCORING)
    )

    def day(number: int, net: int) -> Estimate:
        return Estimate(
            item=f"f01@{number}",
            item_kind="fault_effect_day",
            rescued=net,
            broken=0,
            net=net,
            episodes=14,
            detail={"fault_id": "f01", "day": number},
        )

    def verdict(first: int, last: int) -> str:
        parts = {"fault_effects": [], "interactions": [], "by_day": [day(8, first), day(9, last)]}
        return next(c.verdict for c in pattern_checks(world, parts) if c.pattern == "drift")

    assert (
        verdict(4, 10) == "confirmed"
        and verdict(4, 5) == "weak"
        and verdict(5, 4) == "not confirmed"
    )
