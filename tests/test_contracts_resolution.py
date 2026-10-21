"""Faults, actions and the oracle against a toy truth with three fault types: a camera pose
offset, an action delay and gripper wear.

docs/design/contracts.md, sections 7, 8 and 9: per-episode resolution with the
exact-scope rule and an added change, healthy targets left alone, signed gains
including an episode a fault rescues.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from fleetlab.contracts.actions import (
    CONFIG_CHANGE_KIND,
    COSTS,
    NO_REPLAY_VERBS,
    SCORING_BUDGETS,
    Action,
    Budget,
    FixSet,
    catalogue,
    cost_key,
    cost_of,
    costs_rows,
    cut,
    fits_budget,
    resolve,
    resolve_all,
    total_cost,
)
from fleetlab.contracts.faults import (
    FAULT_HOOKS,
    HOOK_ORDER,
    STEP_ORDER,
    Direction,
    FaultSpec,
    FaultState,
    FaultTypeBase,
    LogSide,
    Schedule,
    Target,
    TruthMapping,
    active_faults,
    check_fault_type,
)
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.ids import ScopeKind
from fleetlab.contracts.model import ComponentState, ConfigVersion, Fleet, Robot, Site
from fleetlab.contracts.oracle import (
    Gain,
    MissingReplays,
    OracleRow,
    OracleTable,
    achievable_gain,
    fault_effect,
    fault_list_hash,
    interaction,
    power_set_replays,
    replays_needed,
    value_fix_set,
)
from fleetlab.contracts.policy import Action7, ControlMode
from fleetlab.contracts.vocabulary import (
    Component,
    ConditionKey,
    ConfigKey,
    FaultClass,
    TargetKind,
    Verb,
)
from fleetlab.faults import FaultRegistry

DELAY = ConfigKey.control_action_delay_steps

# The three toy fault types.


class DegreesParams(BaseModel):
    degrees: float


class StepsParams(BaseModel):
    steps: int


class WearParams(BaseModel):
    friction_scale: float


class CameraPoseOffset(FaultTypeBase):
    kind = "camera_pose_offset"
    fault_class = FaultClass.sensor
    Params = DegreesParams
    dose_levels = (0.0, 3.0, 6.0, 9.0)
    truth = TruthMapping(
        target_kind=TargetKind.component,
        fixing_verbs=frozenset({Verb.recalibrate}),
        scopes=frozenset({ScopeKind.robot}),
        writes_change=False,
        directions=frozenset({Direction.degraded, Direction.changed}),
    )


@dataclass
class DelayState(FaultState):
    line: deque[Action7] = field(default_factory=deque)


class ActionDelay(FaultTypeBase):
    kind = "action_delay"
    fault_class = FaultClass.configuration
    Params = StepsParams
    dose_levels = (0.0, 5.0, 10.0)
    truth = TruthMapping(
        target_kind=TargetKind.config_key,
        fixing_verbs=frozenset({Verb.revert, Verb.set}),
        scopes=frozenset(ScopeKind),
        writes_change=True,
        directions=frozenset({Direction.misconfigured}),
    )

    def new_episode(self, spec: FaultSpec, identity: EpisodeIdentity) -> FaultState:
        return DelayState(spec=spec, episode_id=identity.episode_id)

    def action_hook(self, action: Action7, state: FaultState) -> Action7:
        assert isinstance(state, DelayState)
        state.line.append(action)
        steps = int(state.spec.level)
        return state.line.popleft() if len(state.line) > steps else state.line[0]


class GripperWear(FaultTypeBase):
    kind = "gripper_wear"
    fault_class = FaultClass.physical
    Params = WearParams
    dose_levels = (0.0, 0.5, 1.0)
    truth = TruthMapping(
        target_kind=TargetKind.component,
        fixing_verbs=frozenset({Verb.replace_, Verb.service}),
        scopes=frozenset({ScopeKind.robot}),
        writes_change=False,
        directions=frozenset({Direction.degraded, Direction.drifting}),
    )


def registry() -> FaultRegistry:
    reg = FaultRegistry()
    for cls in (CameraPoseOffset, ActionDelay, GripperWear):
        reg.register(cls.kind)(cls)
    return reg


TRUTH = registry().truth_table()

# The toy world: s1 with r01, r02; s2 with r03, r04.


def toy_fleet() -> Fleet:
    def robot(robot_id: str, site_id: str) -> Robot:
        components = {c: ComponentState(component=c, installed_day=0) for c in Component}
        return Robot(id=robot_id, site_id=site_id, components=components)

    return Fleet(
        id="w01",
        sites=[
            Site(id="s1", robots=[robot("r01", "s1"), robot("r02", "s1")]),
            Site(id="s2", robots=[robot("r03", "s2"), robot("r04", "s2")]),
        ],
    )


CAMERA = FaultSpec(
    fault_id="f01",
    kind="camera_pose_offset",
    params={"degrees": 6.0},
    level=6.0,
    target=Target(scope="r01", component=Component.scene_camera),
    schedule=Schedule(onset_day=2),
)
DELAY_S2 = FaultSpec(
    fault_id="f02",
    kind="action_delay",
    params={"steps": 3},
    level=3.0,
    target=Target(scope="s2", config_key=DELAY),
    schedule=Schedule(onset_day=2),
    nominal=0,
)
WEAR = FaultSpec(
    fault_id="f03",
    kind="gripper_wear",
    params={"friction_scale": 0.5},
    level=0.5,
    target=Target(scope="r03", component=Component.gripper),
    schedule=Schedule(onset_day=0),
)
SITE_OF = {"r01": "s1", "r02": "s1", "r03": "s2", "r04": "s2"}


def episode(number: int, robot_id: str, faults: list[FaultSpec], day: int = 5) -> EpisodeIdentity:
    scene, noise = seeds_for(7, robot_id, day, number)
    return EpisodeIdentity(
        world_id="w01",
        episode_id=f"e{number:06d}",
        robot_id=robot_id,
        site_id=SITE_OF[robot_id],
        day=day,
        episode_index=number,
        suite="libero_spatial",
        task_index=number % 10,
        start_state=number,
        scene_seed=scene,
        noise_seed=noise,
        policy=PolicySpec(
            name="xvla-libero",
            revision="12e8783e",
            precision="float32",
            control_mode=ControlMode.absolute,
            actions_executed_per_chunk=30,
            sampling_steps=10,
            image_views=3,
        ),
        batch_size=4,
        row=None,
        image_size=360,
        max_steps=280,
        faults=faults,
        software=SoftwareVersion(
            lock_hash="a", libero_patch_set="p", mujoco_version="3.3.7", miopen_db_hash="m"
        ),
    )


def act(verb: Verb, scope: str, value: Any = None, **thing: Any) -> Action:
    return Action(verb=verb, target=Target(scope=scope, **thing), value=value)


def removed(action: Action, ep: EpisodeIdentity, current: Any = None) -> set[str]:
    return set(resolve(action, ep, ep.faults, TRUTH, current).removed)


# Fault types and the registry.


def test_the_registry_holds_the_three_types_by_kind() -> None:
    reg = registry()
    assert reg.kinds() == ("action_delay", "camera_pose_offset", "gripper_wear")
    assert reg.get("action_delay").truth.writes_change
    reg.check(DELAY_S2)
    with pytest.raises(KeyError, match="registered: action_delay"):
        reg.get("lens_smear")
    with pytest.raises(ValueError, match="already registered"):

        class Impostor(GripperWear):
            pass

        reg.add(Impostor())
    with pytest.raises(ValidationError):
        reg.check(CAMERA.model_copy(update={"params": {"degrees": "a lot"}}))


def test_a_fault_type_must_implement_the_protocol() -> None:
    class NoTruth(FaultTypeBase):
        kind = "no_truth"
        fault_class = FaultClass.data
        Params = WearParams
        dose_levels = (0.0,)

    with pytest.raises(TypeError):
        check_fault_type(NoTruth())
    with pytest.raises(TypeError):
        check_fault_type(object())


def test_truth_mappings_are_consistent_with_the_verbs() -> None:
    with pytest.raises(ValidationError, match="do not act on"):
        TruthMapping(
            target_kind=TargetKind.component,
            fixing_verbs=frozenset({Verb.revert}),
            scopes=frozenset({ScopeKind.robot}),
            writes_change=False,
            directions=frozenset({Direction.degraded}),
        )
    with pytest.raises(ValidationError, match="writes one"):
        TruthMapping(
            target_kind=TargetKind.config_key,
            fixing_verbs=frozenset({Verb.revert}),
            scopes=frozenset({ScopeKind.site}),
            writes_change=False,
            directions=frozenset({Direction.misconfigured}),
        )
    with pytest.raises(ValueError, match="allows"):
        TRUTH["camera_pose_offset"].check(
            CAMERA.model_copy(
                update={"target": Target(scope="s1", component=Component.scene_camera)}
            )
        )


def mapping(kind: TargetKind, verbs: set[Verb], scopes: set[ScopeKind], writes: bool) -> None:
    TruthMapping(
        target_kind=kind,
        fixing_verbs=frozenset(verbs),
        scopes=frozenset(scopes),
        writes_change=writes,
        directions=frozenset({Direction.misconfigured}),
    )


def test_truth_mappings_follow_the_rules_of_section_7() -> None:
    config, condition, component = (
        TargetKind.config_key,
        TargetKind.condition_key,
        (TargetKind.component),
    )
    with pytest.raises(ValidationError, match="set and revert"):
        mapping(config, {Verb.revert}, {ScopeKind.site}, True)
    with pytest.raises(ValidationError, match="set and revert"):
        mapping(config, {Verb.set}, {ScopeKind.site}, True)
    mapping(config, set(), {ScopeKind.site}, True)  # a failure mode no action fixes
    with pytest.raises(ValidationError, match="change_condition"):
        mapping(condition, set(), {ScopeKind.site}, False)
    mapping(condition, {Verb.change_condition}, {ScopeKind.site}, False)
    with pytest.raises(ValidationError, match="robot-scoped"):
        mapping(component, {Verb.replace_}, {ScopeKind.robot, ScopeKind.site}, False)


def test_a_component_fault_is_fixable_by_a_verb_that_applies_to_it() -> None:
    # recalibrate does not apply to the gripper, so a gripper fault fixed only by it is wrong.
    with pytest.raises(ValueError, match="applies to that component"):
        TRUTH["camera_pose_offset"].check(
            CAMERA.model_copy(update={"target": Target(scope="r01", component=Component.gripper)})
        )
    TRUTH["gripper_wear"].check(WEAR)


def test_writes_change_follows_the_fault_class() -> None:
    class QuietDelay(ActionDelay):
        truth = ActionDelay.truth.model_copy(update={"writes_change": False})

    class LoudCamera(CameraPoseOffset):
        fault_class = FaultClass.software

    for wrong in (QuietDelay(), LoudCamera()):
        with pytest.raises(ValueError, match="writes"):
            check_fault_type(wrong)
    for right in (ActionDelay(), CameraPoseOffset(), GripperWear()):
        check_fault_type(right)


def test_configuration_and_condition_faults_carry_their_nominal_value() -> None:
    with pytest.raises(ValidationError, match="nominal"):
        DELAY_S2.model_validate({**DELAY_S2.model_dump(), "nominal": None})
    with pytest.raises(ValidationError, match="no nominal"):
        CAMERA.model_validate({**CAMERA.model_dump(), "nominal": 0.0})


def test_a_fault_type_implements_only_the_hooks_it_needs() -> None:
    ep = episode(1, "r03", [DELAY_S2])
    delay = ActionDelay()
    state = delay.new_episode(DELAY_S2, ep)
    assert isinstance(state, DelayState)
    commands = [np.full(7, i, dtype=np.float32) for i in range(6)]
    executed = [delay.action_hook(c, state)[0] for c in commands]
    assert executed == [0, 0, 0, 0, 1, 2]  # the first command until the line has filled
    fresh = delay.new_episode(DELAY_S2, ep)  # a new episode starts with an empty line
    assert isinstance(fresh, DelayState) and not fresh.line
    restored = delay.restore(delay.snapshot(state))
    assert isinstance(restored, DelayState) and len(restored.line) == len(state.line)
    camera = CameraPoseOffset()  # the defaults change nothing
    cam_state = camera.new_episode(CAMERA, ep)
    assert camera.cell_hook(None, cam_state) is None  # type: ignore[arg-type]
    assert camera.action_hook(commands[3], cam_state) is commands[3]


def test_the_hook_order_is_fixed_and_says_where_the_log_listens() -> None:
    assert STEP_ORDER == ("observation_hook", "policy", "action_hook", "simulator", "data_hook")
    assert set(FAULT_HOOKS) == {
        "new_episode",
        "cell_hook",
        "policy_input_hook",
        "observation_hook",
        "action_hook",
        "data_hook",
    }
    data_hook = next(point for point in HOOK_ORDER if point.name == "data_hook")
    assert "ends on the verdict after the hook" in data_hook.when
    assert "verdicts" in data_hook.log_note
    assert "only on a done verdict" in data_hook.when
    sides = {point.name: point.log_side for point in HOOK_ORDER}
    assert sides["action_hook"] is LogSide.before  # the log records the commanded action
    assert sides["observation_hook"] is LogSide.after  # and the image the policy saw


def test_active_faults_follow_scope_schedule_and_level() -> None:
    zero = DELAY_S2.model_copy(update={"fault_id": "f09", "level": 0.0})
    later = WEAR.model_copy(update={"fault_id": "f08", "schedule": Schedule(onset_day=9)})
    faults = [CAMERA, DELAY_S2, WEAR, zero, later]
    fleet = toy_fleet()
    assert active_faults(faults, fleet, "r03", day=5) == [DELAY_S2, WEAR]
    assert active_faults(faults, fleet, "r01", day=1) == []
    assert active_faults(faults, fleet, "r03", day=9) == [DELAY_S2, WEAR, later]


def test_a_robot_s_own_setting_shadows_a_wider_configuration_fault() -> None:
    fleet = toy_fleet()
    s2 = fleet.site("s2")
    r03 = s2.robots[0].model_copy(update={"config": ConfigVersion(version=1, values={DELAY: 0})})
    fleet = fleet.model_copy(
        update={"sites": [fleet.sites[0], s2.model_copy(update={"robots": [r03, s2.robots[1]]})]}
    )
    assert active_faults([DELAY_S2], fleet, "r03", day=5) == []  # its own value wins
    assert active_faults([DELAY_S2], fleet, "r04", day=5) == [DELAY_S2]
    at_fleet = DELAY_S2.model_copy(
        update={"fault_id": "f05", "target": Target(scope="fleet", config_key=DELAY)}
    )
    with_site_value = toy_fleet()
    with_site_value = with_site_value.model_copy(
        update={
            "sites": [
                with_site_value.sites[0].model_copy(
                    update={"config": ConfigVersion(version=1, values={DELAY: 0})}
                ),
                with_site_value.sites[1],
            ]
        }
    )
    assert active_faults([at_fleet], with_site_value, "r01", day=5) == []  # the site's value wins
    assert active_faults([at_fleet], with_site_value, "r03", day=5) == [at_fleet]


# Actions.


def test_component_verbs_remove_the_fault_on_that_robot_only() -> None:
    ep1 = episode(1, "r01", [CAMERA])
    recalibrate = act(Verb.recalibrate, "r01", component=Component.scene_camera)
    assert removed(recalibrate, ep1) == {"f01"}
    assert removed(act(Verb.recalibrate, "r01", component=Component.wrist_camera), ep1) == set()
    assert removed(act(Verb.clean, "r01", component=Component.scene_camera), ep1) == set()
    assert removed(act(Verb.recalibrate, "r02", component=Component.scene_camera), ep1) == set()
    healthy = episode(2, "r02", [])
    assert resolve(recalibrate, healthy, healthy.faults, TRUTH).is_no_change


def test_a_bad_batch_at_a_site_is_one_fault_per_robot() -> None:
    # Component faults are robot-scoped only, so k fixes are always k actions.
    with pytest.raises(ValueError, match="allows"):
        TRUTH["gripper_wear"].check(
            WEAR.model_copy(update={"target": Target(scope="s2", component=Component.gripper)})
        )
    on_r04 = WEAR.model_copy(
        update={"fault_id": "f04", "target": Target(scope="r04", component=Component.gripper)}
    )
    ep3, ep4 = episode(3, "r03", [WEAR]), episode(4, "r04", [on_r04])
    replace = act(Verb.replace_, "r03", component=Component.gripper)
    assert removed(replace, ep3) == {"f03"}
    assert removed(replace, ep4) == set()


def test_revert_follows_the_exact_scope_rule() -> None:
    on_r03, on_r04 = episode(3, "r03", [DELAY_S2]), episode(4, "r04", [DELAY_S2])
    at_site = act(Verb.revert, "s2", config_key=DELAY)
    assert removed(at_site, on_r03) == removed(at_site, on_r04) == {"f02"}
    assert removed(act(Verb.revert, "r03", config_key=DELAY), on_r03) == set()
    assert removed(act(Verb.revert, "fleet", config_key=DELAY), on_r03) == set()


def test_set_to_nominal_overrides_a_wider_fault_without_a_change() -> None:
    on_r03, on_r04 = episode(3, "r03", [DELAY_S2]), episode(4, "r04", [DELAY_S2])
    at_robot = act(Verb.set, "r03", 0, config_key=DELAY)
    result = resolve(at_robot, on_r03, on_r03.faults, TRUTH)
    assert result.removed == {"f02"} and result.added == ()
    assert resolve(at_robot, on_r04, on_r04.faults, TRUTH).is_no_change  # r04 not covered
    assert removed(act(Verb.set, "s2", 0, config_key=DELAY), on_r04) == {"f02"}
    # A fleet-wide value does not reach a robot whose site sets its own.
    assert resolve(
        act(Verb.set, "fleet", 0, config_key=DELAY), on_r03, on_r03.faults, TRUTH
    ).is_no_change


def test_set_to_another_value_adds_a_change_that_needs_a_fresh_replay() -> None:
    on_r03 = episode(3, "r03", [DELAY_S2])
    result = resolve(act(Verb.set, "r03", 1, config_key=DELAY), on_r03, on_r03.faults, TRUTH)
    assert result.removed == {"f02"} and result.needs_fresh_replay
    (change,) = result.added
    assert change.kind == CONFIG_CHANGE_KIND and change.params == {"value": 1}
    assert change.target == Target(scope="r03", config_key=DELAY) and change.nominal == 0
    assert change.fault_id.startswith("f9") and change.schedule.onset_day == on_r03.day


def test_set_on_a_healthy_key_changes_only_a_different_value() -> None:
    healthy = episode(1, "r01", [])
    set_five = act(Verb.set, "r01", 5, config_key=DELAY)
    assert resolve(set_five, healthy, [], TRUTH, {DELAY: 5}).is_no_change
    added = resolve(set_five, healthy, [], TRUTH, {DELAY: 0})
    assert added.needs_fresh_replay and added.added[0].nominal == 0
    with pytest.raises(ValueError, match="needs its current value"):
        resolve(set_five, healthy, [], TRUTH)


def test_change_condition_on_a_healthy_site_adds_a_change() -> None:
    healthy = episode(1, "r01", [])
    dim = act(Verb.change_condition, "s1", "dim", condition_key=ConditionKey.lighting)
    assert resolve(dim, healthy, [], TRUTH, {ConditionKey.lighting: "dim"}).is_no_change
    assert resolve(dim, healthy, [], TRUTH, {ConditionKey.lighting: "bright"}).needs_fresh_replay


def test_the_verbs_without_a_replay_change_nothing() -> None:
    ep = episode(1, "r01", [CAMERA])
    teleop = act(Verb.route_to_teleop, "fleet", "put the bowl on the plate")
    data = act(Verb.request_training_data, "fleet", "grasps slip on worn pads")
    for action in (teleop, data):
        assert not action.has_replay
        assert resolve(action, ep, ep.faults, TRUTH).is_no_change


def test_actions_resolve_together_and_refuse_foreign_faults() -> None:
    ep = episode(3, "r03", [DELAY_S2, WEAR])
    both = resolve_all(
        [
            act(Verb.revert, "s2", config_key=DELAY),
            act(Verb.service, "r03", component=Component.gripper),
        ],
        ep,
        ep.faults,
        TRUTH,
    )
    assert both.removed == {"f02", "f03"} and both.added == ()
    with pytest.raises(ValueError, match="not active"):
        resolve(act(Verb.revert, "s2", config_key=DELAY), ep, [CAMERA], TRUTH)
    with pytest.raises(ValueError, match="without a value"):
        resolve(act(Verb.set, "s2", config_key=DELAY), ep, ep.faults, TRUTH)


def test_actions_refuse_targets_and_scopes_their_verb_does_not_take() -> None:
    with pytest.raises(ValidationError, match="takes component"):
        act(Verb.recalibrate, "r01", config_key=DELAY)
    with pytest.raises(ValidationError, match="applies at scope"):
        act(Verb.recalibrate, "s1", component=Component.arm)
    with pytest.raises(ValidationError, match="applies at scope"):
        act(Verb.change_condition, "r01", "dim", condition_key=ConditionKey.lighting)
    with pytest.raises(ValidationError, match="takes no value"):
        act(Verb.revert, "s1", 3, config_key=DELAY)
    with pytest.raises(ValidationError, match="at most one"):
        Target(scope="r01", component=Component.arm, config_key=DELAY)


def test_the_catalogue_lists_every_valid_pair() -> None:
    f = toy_fleet()
    entries = catalogue(f)
    # 16 component actions x 4 robots (recalibrate 3, clean 3, replace 5, service 5),
    # 2 verbs x 9 keys x 7 scopes, 4 conditions x 2 sites, route_to_teleop and
    # request_training_data once each.
    assert len(entries) == 64 + 126 + 8 + 1 + 1
    on_r01 = {(a.verb, a.target.component) for a in entries if a.target.scope == "r01"}
    assert (Verb.recalibrate, Component.gripper) not in on_r01
    assert (Verb.clean, Component.arm) not in on_r01
    assert (Verb.clean, Component.gripper) in on_r01
    assert len({a.model_dump_json() for a in entries}) == len(entries)
    with_tasks = catalogue(f, task_classes=[f"task {i}" for i in range(10)])
    assert len(with_tasks) == len(entries) + 9


def test_a_small_world_has_the_catalogue_size_of_the_experimental_design() -> None:
    def robot(n: int, site: str) -> Robot:
        components = {c: ComponentState(component=c, installed_day=0) for c in Component}
        return Robot(id=f"r{n:02d}", site_id=site, components=components)

    sites = [
        Site(id=f"s{s}", robots=[robot(4 * s + i, f"s{s}") for i in range(4)]) for s in range(3)
    ]
    entries = catalogue(Fleet(id="w01", sites=sites), task_classes=[f"t{i}" for i in range(10)])
    assert len(entries) == 192 + 288 + 12 + 10 + 1  # about 490


def test_fix_sets_keep_their_budget() -> None:
    fix = act(Verb.recalibrate, "r01", component=Component.scene_camera)
    revert = act(Verb.revert, "s2", config_key=DELAY)
    with pytest.raises(ValidationError, match="exceed"):
        FixSet(actions=[fix, revert], budget=Budget(kind="count", amount=1))
    with pytest.raises(ValidationError, match="once"):
        FixSet(actions=[fix, fix], budget=Budget(kind="count", amount=3))
    with pytest.raises(ValidationError, match="without a value"):
        FixSet(
            actions=[act(Verb.set, "s2", config_key=DELAY)], budget=Budget(kind="count", amount=3)
        )
    with pytest.raises(ValidationError, match="whole number"):
        Budget(kind="count", amount=2.5)
    cheap = FixSet(actions=[fix, revert], budget=Budget(kind="cost", amount=0.6))
    assert total_cost(cheap.actions).hours == pytest.approx(0.6)
    assert fits_budget(cheap)
    assert not fits_budget(cheap.model_copy(update={"budget": Budget(kind="cost", amount=0.5)}))
    with pytest.raises(ValidationError, match="applies to"):
        act(Verb.recalibrate, "r01", component=Component.gripper)


def test_the_costs_are_the_experimental_design_s() -> None:
    def hours(verb: Verb, **thing: Any) -> float | None:
        scope = "fleet" if verb in NO_REPLAY_VERBS else "s1" if "condition_key" in thing else "r01"
        value: str | int | None = None
        if verb in NO_REPLAY_VERBS or verb is Verb.change_condition:
            value = "x"
        elif verb is Verb.set:
            value = 1
        return cost_of(act(verb, scope, value, **thing))

    assert hours(Verb.recalibrate, component=Component.arm) == 0.5
    assert hours(Verb.clean, component=Component.gripper) == 0.25
    assert hours(Verb.service, component=Component.controller) == 3
    assert hours(Verb.replace_, component=Component.wrist_camera) == 8
    assert hours(Verb.replace_, component=Component.arm) == 16
    assert hours(Verb.set, config_key=DELAY) == hours(Verb.revert, config_key=DELAY) == 0.1
    assert hours(Verb.change_condition, condition_key=ConditionKey.lighting) == 4
    assert hours(Verb.route_to_teleop) == 0
    assert hours(Verb.request_training_data) is None  # not costed: claim 5
    # Every catalogue entry has a cost, and the table rows say the same.
    assert all(cost_key(a) in COSTS for a in catalogue(toy_fleet()))
    assert {(r["verb"], r["target"]): r["hours"] for r in costs_rows()}[("replace", "arm")] == 16


def ranked_actions() -> list[Action]:
    return [
        act(Verb.replace_, "r01", component=Component.arm),  # 16 hours
        act(Verb.recalibrate, "r01", component=Component.scene_camera),  # 0.5
        act(Verb.replace_, "r02", component=Component.gripper),  # 8
        act(Verb.revert, "s2", config_key=DELAY),  # 0.1
        act(Verb.recalibrate, "r01", component=Component.scene_camera),  # again: skipped
        act(Verb.service, "r03", component=Component.gripper),  # 3
        act(Verb.request_training_data, "fleet", "slips on worn pads"),  # not costed
        act(Verb.clean, "r04", component=Component.wrist_camera),  # 0.25
    ]


def test_cut_walks_down_the_list_and_skips_what_no_longer_fits() -> None:
    ranked = ranked_actions()
    three = cut(ranked, Budget(kind="count", amount=3))
    assert three.actions == ranked[:3]
    six = cut(ranked, Budget(kind="count", amount=6))
    assert six.actions == [*ranked[:4], ranked[5], ranked[6]]  # the repeat is skipped
    four_hours = cut(ranked, Budget(kind="cost", amount=4))
    # 16 and 8 never fit; 0.5 + 0.1 + 3 = 3.6, the untimed request fits, 0.25 still fits.
    assert four_hours.actions == [ranked[1], ranked[3], ranked[5], ranked[6], ranked[7]]
    assert fits_budget(four_hours)
    eight_hours = cut(ranked, Budget(kind="cost", amount=8))
    assert eight_hours.actions == [ranked[1], ranked[3], ranked[5], ranked[6], ranked[7]]
    assert [b.amount for b in SCORING_BUDGETS] == [3, 6, 4, 8]
    with pytest.raises(ValueError, match="without a value"):
        cut([act(Verb.set, "s2", config_key=DELAY)], Budget(kind="count", amount=3))


# The oracle.

# Each episode's outcome as a function of the faults left in it.
OUTCOMES: dict[str, dict[frozenset[str], bool]] = {
    # f01 breaks e1 and e3; on e2 it rescues the episode (gate 1 saw that in 2 of 40).
    "e000001": {frozenset({"f01"}): False, frozenset(): True},
    "e000002": {frozenset({"f01"}): True, frozenset(): False},
    "e000003": {frozenset({"f01"}): False, frozenset(): True},
    # f02 and f03 interact on e4: either alone is enough to fail.
    "e000004": {
        frozenset({"f02", "f03"}): False,
        frozenset({"f02"}): False,
        frozenset({"f03"}): False,
        frozenset(): True,
    },
    "e000005": {frozenset({"f02"}): False, frozenset(): True},
    "e000006": {frozenset(): True},  # healthy
}


def scoring_episodes() -> list[EpisodeIdentity]:
    return [
        episode(1, "r01", [CAMERA]),
        episode(2, "r01", [CAMERA]),
        episode(3, "r01", [CAMERA]),
        episode(4, "r03", [DELAY_S2, WEAR]),
        episode(5, "r04", [DELAY_S2]),
        episode(6, "r02", []),
    ]


def oracle_table(skip: tuple[str, frozenset[str]] | None = None) -> OracleTable:
    identities = scoring_episodes()
    rows = []
    for ep in identities:
        lists = [tuple(ep.faults), *(power_set_replays(ep) or [])]
        for faults in lists:
            kept = frozenset(f.fault_id for f in faults)
            if skip == (ep.episode_id, kept):
                continue
            rows.append(
                OracleRow(
                    episode_id=ep.episode_id,
                    active_faults=fault_list_hash(faults),
                    success=OUTCOMES[ep.episode_id][kept],
                )
            )
    return OracleTable(identities, rows)


def fix_set(*actions: Action) -> FixSet:
    return FixSet(actions=list(actions), budget=Budget(kind="count", amount=6))


def test_a_fault_effect_is_signed_and_counts_the_episode_it_rescues() -> None:
    table = oracle_table()
    assert fault_effect("f01", table) == Gain(rescued=2, broken=1, net=1)
    assert fault_effect("f02", table) == Gain.of(1, 0)  # e4 still fails on f03
    assert fault_effect("f03", table) == Gain.of(0, 0)


def test_a_fix_set_is_valued_per_episode_by_lookup() -> None:
    table = oracle_table()
    recalibrate = act(Verb.recalibrate, "r01", component=Component.scene_camera)
    assert value_fix_set(fix_set(recalibrate), table, TRUTH) == Gain.of(2, 1)
    revert = act(Verb.revert, "s2", config_key=DELAY)
    assert value_fix_set(fix_set(revert), table, TRUTH) == Gain.of(1, 0)
    replace = act(Verb.replace_, "r03", component=Component.gripper)
    assert value_fix_set(fix_set(revert, replace), table, TRUTH) == Gain.of(2, 0)
    everything = fix_set(recalibrate, revert, replace)
    assert value_fix_set(everything, table, TRUTH) == Gain.of(4, 1)


def test_actions_on_healthy_targets_need_no_simulation() -> None:
    table = oracle_table()
    healthy = fix_set(
        act(Verb.recalibrate, "r02", component=Component.scene_camera),
        act(Verb.revert, "s1", config_key=DELAY),
        act(Verb.route_to_teleop, "fleet", "pick up the black bowl"),
    )
    assert replays_needed(healthy, table, TRUTH) == []
    assert value_fix_set(healthy, table, TRUTH) == Gain()


def test_an_added_change_or_a_missing_row_is_reported_not_guessed() -> None:
    table = oracle_table()
    added = fix_set(act(Verb.set, "r04", 1, config_key=DELAY))
    needs = replays_needed(added, table, TRUTH)
    assert [n.episode_id for n in needs] == ["e000005"]
    assert needs[0].faults[0].kind == CONFIG_CHANGE_KIND
    with pytest.raises(MissingReplays) as caught:
        value_fix_set(added, table, TRUTH)
    assert caught.value.needs == needs
    gap = oracle_table(skip=("e000005", frozenset()))
    with pytest.raises(MissingReplays, match="e000005"):
        value_fix_set(fix_set(act(Verb.revert, "s2", config_key=DELAY)), gap, TRUTH)


def test_an_added_change_is_a_lookup_once_its_row_is_in_the_table() -> None:
    # Run the fresh replay, add its row, and the same call values the set.
    table = oracle_table()
    added = fix_set(act(Verb.set, "r04", 1, config_key=DELAY))
    (need,) = replays_needed(added, table, TRUTH)
    table.add(
        OracleRow(
            episode_id=need.episode_id, active_faults=fault_list_hash(need.faults), success=True
        )
    )
    assert replays_needed(added, table, TRUTH) == []
    assert value_fix_set(added, table, TRUTH) == Gain.of(1, 0)  # e5 failed with the delay


def test_interaction_is_the_joint_effect_beyond_the_sum() -> None:
    table = oracle_table()
    assert interaction("f02", "f03", table) == 1  # together 1, apart 0 and 0
    assert interaction("f01", "f02", table) == 0  # never active together


def test_achievable_gain_is_the_best_set_of_k_fixes() -> None:
    table = oracle_table()
    assert achievable_gain(0, table) == Gain()
    assert achievable_gain(1, table).net == 1
    assert achievable_gain(2, table).net == 2
    assert achievable_gain(3, table) == Gain.of(4, 1)


def test_the_power_set_covers_every_subset_up_to_the_cap() -> None:
    ep = episode(4, "r03", [DELAY_S2, WEAR])
    lists = power_set_replays(ep)
    assert lists is not None
    assert sorted(sorted(f.fault_id for f in faults) for faults in lists) == [[], ["f02"], ["f03"]]
    # An excluded fault (a decoy) stays in force and is not toggled.
    only = power_set_replays(ep, excluded={"f03"})
    assert only is not None and [[f.fault_id for f in faults] for faults in only] == [["f03"]]
    many = [
        CAMERA.model_copy(
            update={
                "fault_id": f"f1{i}",
                "target": Target(scope="r03", component=Component.scene_camera),
            }
        )
        for i in range(5)
    ]
    crowded = episode(9, "r03", many)
    assert power_set_replays(crowded) is None
    assert OracleTable([crowded]).over_cap() == ["e000009"]


def test_the_oracle_holds_only_its_own_episodes() -> None:
    table = oracle_table()
    stranger = OracleRow(episode_id="e000099", active_faults=fault_list_hash([]), success=True)
    with pytest.raises(ValueError, match="not a scoring episode"):
        table.add(stranger)


def test_replays_of_one_fault_list_must_agree() -> None:
    table = oracle_table()
    ep = scoring_episodes()[0]
    with pytest.raises(ValueError, match="disagree"):
        table.add(
            OracleRow(
                episode_id=ep.episode_id, active_faults=fault_list_hash(ep.faults), success=True
            )
        )
    assert fault_list_hash([CAMERA, DELAY_S2]) == fault_list_hash([DELAY_S2, CAMERA])
