"""The lasting condition and the settings that adapt to it (docs/design/loop.md, section 7):
the condition stays under every action, the compensation acts on it, compensation 0 changes
no byte, the site's public description says the condition exists and nothing of its size.
No simulator, no GPU: the toy cell carries the engine's part."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from engine_helpers import ident

import fleetlab.faults as faults
from fleetlab.adaptation import (
    COMPENSATION,
    NOMINAL,
    PER_CHUNK,
    PHASES,
    Cell,
    analyse,
    cell_of,
    condition,
    plan,
    setting_actions,
)
from fleetlab.contracts._base import visible_dump
from fleetlab.contracts.actions import Action, catalogue, resolve, resolve_all
from fleetlab.contracts.env import PIXEL_KEYS, Camera
from fleetlab.contracts.faults import FaultSpec, Schedule, Target, active_faults
from fleetlab.contracts.identity import PolicySpec, SoftwareVersion, Task
from fleetlab.contracts.model import (
    ComponentState,
    ConfigVersion,
    Fleet,
    Robot,
    Site,
    describe_for_agents,
)
from fleetlab.contracts.policy import ActionChunk, ControlMode
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import (
    LATER_CONFIG_NOMINAL,
    Component,
    ConditionKey,
    ConfigKey,
    Verb,
    describe,
    is_vocabulary,
    vocabulary_json,
)
from fleetlab.dose_response import Outcome, RunOutcomes
from fleetlab.engine import toy
from fleetlab.engine.chain import compare
from fleetlab.engine.episode import ChunkRequest, DirectSource, EpisodeContext, call_policy, run_job
from fleetlab.engine.software import current
from fleetlab.faults.records import measured, site_notes
from fleetlab.faults.remote_link_latency import SITE_NOTE
from fleetlab.policies.xvla import nominal_spec
from fleetlab.worlds.generate import NOMINAL_CONDITIONS, NOMINAL_CONFIG

# A fleet composed with the setting (`WorldSpec.later_settings`): its fleet default.
WITH_KEY: dict[ConfigKey, Any] = {**NOMINAL_CONFIG, **LATER_CONFIG_NOMINAL}

KIND = "remote_link_latency"


# ---- the knob: compensation 0 changes no byte ----------------------------------------------


def test_compensation_zero_is_left_out_of_every_serialized_spec() -> None:
    spec = nominal_spec()
    assert "latency_compensation_steps" not in spec.model_dump()
    assert "latency_compensation_steps" not in spec.model_dump_json()
    shifted = spec.model_copy(update={"latency_compensation_steps": 14})
    assert shifted.model_dump()["latency_compensation_steps"] == 14
    assert PolicySpec.model_validate_json(shifted.model_dump_json()) == shifted
    assert PolicySpec.model_validate(spec.model_dump()) == spec
    with pytest.raises(ValueError):
        spec.model_copy(update={"latency_compensation_steps": -1}).model_validate(
            {**spec.model_dump(), "latency_compensation_steps": -1}
        )


class _Numbered:
    """A server policy whose chunk is its own action numbers: action j has every entry j."""

    chunk_length = 8

    def act(self, batch: Any, instructions: Any, noise: Any, spec: PolicySpec) -> Any:
        actions = np.repeat(np.arange(self.chunk_length, dtype=np.float32)[:, None], 7, axis=1)
        return [ActionChunk(chunk_index=0, actions=actions) for _ in batch]

    def convert(self, chunk: ActionChunk, control_mode: ControlMode) -> list[Any]:
        return [np.asarray(a, np.float32) for a in chunk.actions]


def _request(executed: int, skip: int, chunk_index: int = 1) -> ChunkRequest:
    spec = toy.TOY_SPEC.model_copy(
        update={"actions_executed_per_chunk": executed, "latency_compensation_steps": skip}
    )
    pixels = {PIXEL_KEYS[c]: np.zeros((4, 4, 3), np.uint8) for c in Camera}
    return ChunkRequest(
        key="e1",
        observation={"pixels": pixels},  # type: ignore[typeddict-item]
        instruction="",
        noise=None,
        spec=spec,
        row=None,
        batch_size=1,
        chunk_index=chunk_index,
    )


@pytest.mark.parametrize(
    ("executed", "skip", "first"),
    [
        (5, 0, [0, 1, 2, 3, 4]),
        (5, 3, [3, 4, 5, 6, 7]),
        (5, 6, [6, 7]),
        (5, 20, [7]),
        (8, 0, list(range(8))),
    ],
)
def test_the_cut_skips_the_stale_head_and_keeps_at_least_one_action(
    executed: int, skip: int, first: list[int]
) -> None:
    (actions,) = call_policy(_Numbered(), [_request(executed, skip)])  # type: ignore[arg-type]
    assert [int(a[0]) for a in actions] == first
    # an attempt's first chunk: the robot waits for it, nothing in it is stale
    (opening,) = call_policy(_Numbered(), [_request(executed, skip, 0)])  # type: ignore[arg-type]
    assert [int(a[0]) for a in opening] == list(range(min(executed, 8)))


# ---- the condition: no action removes it ----------------------------------------------------


def _fleet() -> Fleet:
    """Two sites of two robots; s2's robots are driven over the remote link."""
    components = {c: ComponentState(component=c, installed_day=-100) for c in Component}

    def site(number: int) -> Site:
        robots = [
            Robot(id=f"r0{2 * number - 1 + i}", site_id=f"s{number}", components=components)
            for i in range(2)
        ]
        notes = [SITE_NOTE] if number == 2 else []
        return Site(
            id=f"s{number}", robots=robots, conditions=dict(NOMINAL_CONDITIONS), notes=notes
        )

    return Fleet(
        id="w01", sites=[site(1), site(2)], config=ConfigVersion(version=1, values=WITH_KEY)
    )


def _filled(action: Action) -> Action:
    """A catalogue entry with a value an operator could give it."""
    if action.verb is Verb.set:
        key = action.target.config_key
        values: dict[ConfigKey | None, Any] = {
            ConfigKey.policy_sampling_steps: 5,
            ConfigKey.policy_actions_executed_per_chunk: 10,
            ConfigKey.policy_image_views: 3,
            ConfigKey.policy_precision: "fp16_res32",
            ConfigKey.control_mode: "relative",
            ConfigKey.control_action_delay_steps: 0,
            ConfigKey.control_latency_compensation_steps: 14,
            ConfigKey.camera_scene_calibration: [0.66, 0.0, 1.61, 0.0, 33.0, 90.0],
            ConfigKey.camera_wrist_calibration: [0.0, 0.0, 0.05, 0.0, 0.0, 180.0],
            ConfigKey.software_version: "4.2.1",
        }
        return action.model_copy(update={"value": values[key]})
    if action.verb is Verb.change_condition:
        conditions: dict[ConditionKey | None, Any] = {
            ConditionKey.lighting: 0.0,
            ConditionKey.placement: [0.0, 0.0],
            ConditionKey.object_batch: [0.0, 0.0],
            ConditionKey.clutter: 0.0,
        }
        return action.model_copy(update={"value": conditions[action.target.condition_key]})
    if action.verb in (Verb.route_to_teleop, Verb.request_training_data):
        return action.model_copy(update={"value": "anything"})
    return action


def test_every_catalogue_action_leaves_the_condition_in_force() -> None:
    """A review's must-fix (loop.md section 7): `set` or `revert` of the delay key, or any
    catalogue action at the site, leaves the condition in force; a planted delay on the
    configuration key beside it is removed by the same verbs as before."""
    fleet = _fleet()
    truth = faults.truth_table()
    now: dict[ConfigKey | ConditionKey, Any] = {}
    for config_key, value in WITH_KEY.items():
        now[config_key] = value
    for condition_key, value in NOMINAL_CONDITIONS.items():
        now[condition_key] = value
    planted = FaultSpec(
        fault_id="f09",
        kind="action_latency",
        level=5.0,
        target=Target(scope="s2", config_key=ConfigKey.control_action_delay_steps),
        schedule=Schedule(onset_day=0),
        nominal=0,
    )
    for robot in ("r03", "r04"):
        lasting = condition(14, robot).model_copy(update={"fault_id": f"f0{robot[-1]}"})
        episode = ident(1, robot=int(robot[1:])).model_copy(
            update={"site_id": "s2", "faults": [lasting, planted]}
        )
        actions = [_filled(a) for a in catalogue(fleet, ["a task"])]
        assert len(actions) > 100
        for action in actions:
            result = resolve(action, episode, episode.faults, truth, now)
            assert lasting.fault_id not in result.removed, action
        # all of them at once, and the delay key's own verbs remove only the planted delay
        together = resolve_all(actions, episode, episode.faults, truth, now)
        assert lasting.fault_id not in together.removed
        revert = Action(
            verb=Verb.revert,
            target=Target(scope="s2", config_key=ConfigKey.control_action_delay_steps),
        )
        assert resolve(revert, episode, episode.faults, truth).removed == {"f09"}
        # what stays in force after the operator's settings, as the loop resolves a day
        added = list(together.added)
        left = [f for f in episode.faults if f.fault_id not in together.removed] + added
        settled = fleet.model_copy(
            update={
                "sites": [
                    fleet.sites[0],
                    fleet.sites[1].model_copy(
                        update={
                            "config": ConfigVersion(
                                version=2, values={ConfigKey.control_action_delay_steps: 0}
                            )
                        }
                    ),
                ]
            }
        )
        assert lasting in active_faults(left, settled, robot, 1)
    assert not faults.get(KIND).truth.fixing_verbs


def test_a_fleet_composed_before_the_key_does_not_have_the_setting() -> None:
    """A later key is offered, listed and described only by a fleet whose configuration
    records its fleet default, so a stored fleet keeps its catalogue and description and no
    action names a value its description lacks (the build rule's and the oracle's filled
    catalogues read that value). A key held only at a site or robot is refused."""
    new = _fleet()
    old = new.model_copy(update={"config": ConfigVersion(version=1, values=NOMINAL_CONFIG)})

    def keys(fleet: Fleet) -> set[Any]:
        return {a.target.config_key for a in catalogue(fleet) if a.target.config_key}

    assert COMPENSATION in keys(new)
    assert keys(old) == set(ConfigKey) - {COMPENSATION}
    assert len(catalogue(new)) - len(catalogue(old)) == 2 * (1 + 2 + 4)  # set, revert x scopes
    listed = {k["name"] for k in describe_for_agents(new)["vocabulary"]["config_keys"]}
    assert COMPENSATION.value in listed
    unlisted = {k["name"] for k in describe_for_agents(old)["vocabulary"]["config_keys"]}
    assert unlisted == listed - {COMPENSATION.value}
    assert describe_for_agents(old)["vocabulary"] == vocabulary_json(frozenset({COMPENSATION}))
    assert is_vocabulary(describe_for_agents(old)["vocabulary"])
    assert is_vocabulary(describe_for_agents(new)["vocabulary"])
    # a site may override the fleet default, never hold the key alone
    site = new.sites[1].model_copy(
        update={"config": ConfigVersion(version=1, values={COMPENSATION: 14})}
    )
    Fleet.model_validate(new.model_copy(update={"sites": [new.sites[0], site]}).model_dump())
    alone = old.model_copy(update={"sites": [old.sites[0], site]}).model_dump()
    with pytest.raises(ValueError, match="without the fleet default"):
        Fleet.model_validate(alone)


def test_a_spec_opts_in_and_every_other_world_keeps_its_bytes() -> None:
    from fleetlab.contracts.world import WorldSpec
    from fleetlab.worlds.generate import generate
    from fleetlab.worlds.mini import TASKS, mini_spec

    software = SoftwareVersion(
        lock_hash="0" * 16,
        libero_patch_set="1" * 16,
        mujoco_version="3.3.7",
        miopen_db_hash="2" * 16,
    )
    texts = {Task(t.suite, t.task_index): f"task {t.suite} {t.task_index}" for t in TASKS}
    spec = mini_spec()
    assert "later_settings" not in spec.model_dump_json()
    plain = generate(spec, software, texts)
    assert COMPENSATION not in plain.fleet.config.values
    opted = spec.model_copy(update={"later_settings": [COMPENSATION]})
    assert WorldSpec.model_validate_json(opted.model_dump_json()) == opted
    world = generate(opted, software, texts)
    assert world.fleet.config.values[COMPENSATION] == 0
    assert COMPENSATION in {a.target.config_key for a in catalogue(world.fleet)}
    with pytest.raises(ValueError):
        WorldSpec.model_validate({**spec.model_dump(), "later_settings": ["control.mode"]})


def test_the_vocabulary_offers_the_knob_and_names_no_condition() -> None:
    text = json.dumps(vocabulary_json())
    assert ConfigKey.control_latency_compensation_steps.value in text
    assert KIND not in text
    assert "late" in describe(ConfigKey.control_latency_compensation_steps)
    assert {ConfigKey.control_latency_compensation_steps: 0} == LATER_CONFIG_NOMINAL
    assert ConfigKey.control_latency_compensation_steps not in NOMINAL_CONFIG
    assert NOMINAL == {COMPENSATION: 0, PER_CHUNK: 30}


# ---- the compensation acts on the condition, in the rollout loop ---------------------------


def _toy_context() -> EpisodeContext:
    policy = toy.ToyPolicy()
    return EpisodeContext(
        fault_types=faults.get,
        make_check=toy.ToyCheck,
        source=DirectSource(policy),
        noise_shape=policy.noise_shape,
        noise_dtype=policy.noise_dtype,
    )


def _toy_episode(delay: int, compensation: int) -> Any:
    episode = ident(3, max_steps=40)
    lasting = [condition(delay)] if delay else []
    episode = episode.model_copy(update={"faults": lasting})
    actions = (
        [
            Action(
                verb=Verb.set,
                target=Target(scope="s1", config_key=COMPENSATION),
                value=compensation,
            )
        ]
        if compensation
        else []
    )
    added = resolve_all(actions, episode, episode.faults, faults.truth_table(), NOMINAL).added
    episode = episode.model_copy(update={"faults": [*lasting, *added]})
    return run_job(Job(identity=episode), toy.ToyEnv("toy", 0, 32), _toy_context())


def test_compensation_changes_what_the_robot_does_under_the_condition() -> None:
    plain = _toy_episode(3, 0)
    compensated = _toy_episode(3, 3)
    assert compensated.effective_spec.latency_compensation_steps == 3
    assert plain.effective_spec.latency_compensation_steps == 0
    assert {f.kind for f in compensated.effective_faults} == {KIND, "config_change"}
    commanded = [c for c in plain.steps.commanded if c is not None]
    shifted = [c for c in compensated.steps.commanded if c is not None]
    # the first chunk (five of the toy's eight actions) is the same: the warm-up is too
    first = toy.TOY_SPEC.actions_executed_per_chunk
    assert all(
        np.array_equal(a, b) for a, b in zip(commanded[:first], shifted[:first], strict=True)
    )
    assert not all(np.array_equal(a, b) for a, b in zip(commanded, shifted, strict=False))
    assert np.array_equal(
        plain.chain.executed[1 : 1 + first], compensated.chain.executed[1 : 1 + first]
    )
    # the condition's delay line: the robot executes each command three steps late
    executed = compensated.chain.executed[1:]  # the chain's first row is the reset
    assert np.array_equal(executed[3], shifted[0])
    assert np.array_equal(executed[10], shifted[7])


def test_the_condition_is_the_calibrated_delay_line_and_compensation_zero_is_no_change() -> None:
    lasting = _toy_episode(3, 0)
    healthy = _toy_episode(0, 0)
    planted = ident(3, max_steps=40)
    planted = planted.model_copy(
        update={
            "faults": [
                FaultSpec(
                    fault_id="f01",
                    kind="action_latency",
                    level=3.0,
                    target=Target(scope="r01", config_key=ConfigKey.control_action_delay_steps),
                    schedule=Schedule(onset_day=0),
                    nominal=0,
                )
            ]
        }
    )
    same = run_job(Job(identity=planted), toy.ToyEnv("toy", 0, 32), _toy_context())
    assert compare(lasting.chain, same.chain).identical
    assert not compare(lasting.chain, healthy.chain).identical
    assert measured(KIND) == measured("action_latency")


# ---- what the operator knows -----------------------------------------------------------------


def test_the_site_note_says_the_condition_exists_and_nothing_of_its_size() -> None:
    schedule = [condition(14, "r03"), condition(14, "r04").model_copy(update={"fault_id": "f02"})]
    assert site_notes(schedule, "s2", ["r03", "r04"]) == [SITE_NOTE]
    assert site_notes(schedule, "s1", ["r01", "r02"]) == []
    assert not any(ch.isdigit() for ch in SITE_NOTE)
    assert KIND not in SITE_NOTE and "step" not in SITE_NOTE and "late" not in SITE_NOTE
    fleet = _fleet()
    description = describe_for_agents(fleet)
    sites = description["fleet"]["sites"]
    assert "notes" not in sites[0]
    assert sites[1]["notes"] == [SITE_NOTE]
    assert "notes" not in fleet.sites[0].model_dump()
    assert "notes" not in visible_dump(fleet.sites[0])


# ---- the measurement's plan and tables -------------------------------------------------------


def test_the_plan_pairs_attempts_across_cells_and_keeps_the_condition() -> None:
    jobs = plan("p1", nominal_spec(), current("none"))
    assert len(jobs) == sum(c.episodes for c in PHASES["p1"])
    by_cell: dict[str, dict[int, Any]] = {}
    for job in jobs:
        cell, index = cell_of("p1", job.identity.episode_id)
        by_cell.setdefault(cell.name, {})[index] = job.identity
        kinds = [f.kind for f in job.identity.faults]
        assert kinds.count(KIND) == (1 if cell.delay else 0)
        assert kinds.count("config_change") == len(cell.setting)
    ids = [j.identity.episode_id for j in jobs]
    assert len(ids) == len(set(ids))
    a, b = by_cell["spatial-d17-c0"][7], by_cell["spatial-d17-c17"][7]
    assert (a.scene_seed, a.noise_seed, a.task_index, a.start_state, a.suite) == (
        b.scene_seed,
        b.noise_seed,
        b.task_index,
        b.start_state,
        b.suite,
    )
    assert (
        setting_actions(Cell("libero_spatial", 17, ((COMPENSATION, 17),)))[0].target.scope == "s1"
    )


def test_the_tables_pair_each_setting_with_its_default() -> None:
    outcomes = {}
    for job in plan("p1", nominal_spec(), current("none")):
        cell, index = cell_of("p1", job.identity.episode_id)
        if cell.delay == 0:
            success = True
        elif cell.is_default:
            success = index % 2 == 0
        else:
            success = index % 4 != 3
        outcomes[job.identity.episode_id] = Outcome(
            episode_id=job.identity.episode_id,
            true_success=success,
            logged_success=success,
            libero_success=success,
            steps=100,
        )
    first = RunOutcomes("toy", True, outcomes)
    rows = {r.cell: r for r in analyse({"p1": first})}
    full = rows["spatial-d17-c17"]
    assert (full.n, full.successes, full.against) == (60, 45, "p1/spatial-d17-c0")
    assert (full.rescued, full.broken) == (15, 0)
    assert full.difference is not None and full.difference[1] > 0
    assert rows["spatial-d17-c0"].difference is None
    # a later phase's settings are paired with p1's default cells
    later = {}
    for job in plan("p2", nominal_spec(), current("none")):
        cell, index = cell_of("p2", job.identity.episode_id)
        success = cell.delay == 0 or index % 3 != 0
        later[job.identity.episode_id] = Outcome(
            episode_id=job.identity.episode_id,
            true_success=success,
            logged_success=success,
            libero_success=success,
            steps=90,
        )
    both = analyse({"p1": first, "p2": RunOutcomes("toy2", True, later)})
    again = next(r for r in both if r.phase == "p2" and r.cell == "spatial-d17-c8")
    assert (again.against, again.pairs, again.successes) == ("p1/spatial-d17-c0", 60, 40)
    # default succeeds on even attempts, the setting off multiples of three
    assert (again.rescued, again.broken) == (20, 10)
