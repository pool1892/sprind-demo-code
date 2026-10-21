"""Exact catalogue counts against explicitly enumerated sets and permutations."""

import itertools
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_oracle import CAMERA, DECOY, DELAY, SCORING, toy_table, toy_world

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import Action, Budget, FixSet, cut
from fleetlab.contracts.faults import Target
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import value_fix_set
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey, Verb
from fleetlab.oracle.exact import exact_chance
from fleetlab.oracle.fresh import chance_fix_sets, needs_of_fix_sets
from fleetlab.oracle.steps import Oracle
from fleetlab.oracle.valuation import DrawValuer, random_fix_sets
from fleetlab.oracle.world import OracleLayout


def entries() -> list[Action]:
    return [
        Action(verb=Verb.recalibrate, target=CAMERA.target),
        Action(verb=Verb.replace_, target=CAMERA.target),
        Action(verb=Verb.revert, target=DELAY.target),
        Action(verb=Verb.request_training_data, target=Target(scope="fleet"), value="unspecified"),
    ]


@pytest.mark.parametrize(
    "kind,amount", [("count", 1), ("count", 2), ("count", 3), ("count", 6), ("cost", 0.6)]
)
def test_exact_distribution_matches_brute_force(
    kind: Literal["count", "cost"], amount: float
) -> None:
    table, truth = toy_table(), toy_world().truth()
    budget = Budget(kind=kind, amount=amount)
    pool = entries()
    if kind == "count":
        sets = [
            FixSet(actions=list(a), budget=budget)
            for a in itertools.combinations(pool, min(int(amount), len(pool)))
        ]
    else:
        sets = [cut(a, budget) for a in itertools.permutations(pool)]
    nets = [value_fix_set(fs, table, truth).net for fs in sets]
    result = exact_chance(DrawValuer(table, truth, None), pool, budget)
    assert result.missing is None and result.sets == len(nets)
    assert sum(weight for _, weight in result.histogram) == len(nets)
    assert result.unvalued_sets == 0 and result.bounds is None
    assert result.mean_points == round(100 * np.mean(nets) / len(table.identities), 3)
    assert [result.p5_points, result.p95_points] == [
        round(float(n), 3)
        for n in np.percentile(100 * np.array(nets) / len(table.identities), [5, 95])
    ]


def test_exact_never_drops_sets_with_missing_joint_outcomes() -> None:
    table = toy_table()
    table._rows.clear()
    result = exact_chance(
        DrawValuer(table, toy_world().truth(), None), entries(), Budget(kind="count", amount=2)
    )
    assert result.missing is None and result.mean_points is None and result.sets == 6
    assert result.unvalued_sets == 6 and result.histogram == []
    assert result.bounds


def test_only_mechanism_additions_require_replay_rows() -> None:
    table, truth = toy_table(), toy_world().truth()

    def current(identity: EpisodeIdentity) -> dict[ConfigKey | ConditionKey, ConfigValue]:
        return {ConfigKey.software_version: "old", ConfigKey.control_action_delay_steps: 0}

    raw = DrawValuer(table, truth, current)
    inert = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.software_version),
        value="new",
    )
    physical = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.control_action_delay_steps),
        value=20,
    )
    inert_result = exact_chance(raw, [inert], Budget(kind="count", amount=1))
    assert raw.effects(inert) == {}
    assert inert_result.mean_points == 0 and inert_result.histogram == [(0, 1)]
    assert inert_result.unvalued_sets == 0
    result = exact_chance(raw, [physical], Budget(kind="count", amount=1))
    assert raw.effects(physical) and result.mean_points is None
    assert result.unvalued_sets == 1 and result.bounds


def test_chance_plan_collects_the_same_seeded_joint_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, table = toy_world(), toy_table()
    oracle = Oracle(tmp_path, world, OracleLayout(tmp_path))
    from fleetlab.oracle.table import TableBuild

    monkeypatch.setattr(Oracle, "table", lambda self: TableBuild(table, []))
    monkeypatch.setattr(type(world), "catalogue", lambda self: entries())
    planned = chance_fix_sets(entries(), world.spec.seed, draws=3)
    from fleetlab.contracts.oracle import fault_list_hash

    del table._rows[SCORING[4].episode_id, fault_list_hash([DECOY])]
    expected = needs_of_fix_sets(planned, table, world.truth(), world.current()).unique()
    record = oracle.step_fresh_plan("chance", [], chance=True, draws=3)
    assert expected and record["jobs"] > 0
    assert record["chance"] and record["chance_draws"] == 3
    assert record["chance_seed"] == world.spec.seed
    assert len(record["fix_sets"]) == 12
    assert record["fix_sets"] == [
        {"label": label, **fs.model_dump(mode="json")} for label, fs in planned
    ]
    assert sum(record["needs"].values()) >= len(expected)
    from fleetlab.contracts.oracle import fault_list_hash
    from fleetlab.oracle.plan import read_manifest

    jobs = read_manifest(oracle.layout.fresh("chance") / "jobs.jsonl")
    needed_keys = {(n.episode_id, fault_list_hash(n.faults)) for n in expected}
    assert needed_keys <= {
        (j.identity.episode_id, fault_list_hash(j.effective_faults)) for j in jobs
    }
    draws = list(
        random_fix_sets(
            entries(),
            [
                Budget(kind="count", amount=3),
                Budget(kind="count", amount=6),
                Budget(kind="cost", amount=4),
                Budget(kind="cost", amount=8),
            ],
            seed=world.spec.seed,
            draws=3,
        )
    )
    assert [fs for _, fs in planned] == [fs for sets in draws for fs in sets]
