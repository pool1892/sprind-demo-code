"""The scorer against a toy world counted by hand (src/fleetlab/scoring).

Two sites with two robots each, ten scoring episodes and five faults: a camera
pose offset on r01 (detectable), an action latency at s2 (detectable), a worn
gripper on r04 (small), a policy knob at s1 (decoy) and a policy knob on r03
(the inert entry of a confounder pair). The oracle replays the power set of the
three toggled faults; the decoy and the inert entry stay in force, and no row
removes them, so an action that does needs a fresh replay.

Gains by hand (ten episodes, so one episode is ten points): removing f01
rescues e000001 and e000002 and breaks e000003 (net 1); removing f02 rescues
e000005 and e000007 (net 2); removing f04 alone changes nothing; f02 and f04
together also rescue e000008 (net 3); all three net 4, five rescued and one
broken. achievable_gain is 2, 3 and 4 at k = 1, 2 and 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr

from fleetlab.contracts._base import ConfigValue
from fleetlab.contracts.actions import (
    CONFIG_CHANGE_KIND,
    SCORING_BUDGETS,
    Action,
    Budget,
    FixSet,
    catalogue,
)
from fleetlab.contracts.faults import Direction, FaultSpec, Schedule, Target
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion, seeds_for
from fleetlab.contracts.model import ComponentState, Fleet, Robot, Site
from fleetlab.contracts.oracle import (
    Gain,
    MissingReplays,
    OracleRow,
    OracleTable,
    achievable_gain,
    fault_list_hash,
    power_set_replays,
    replays_needed,
    value_fix_set,
)
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.tools import (
    Cause,
    Diagnosis,
    EvidenceRef,
    Interval,
    RankedAction,
    RankedFixSet,
)
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey, FaultClass, Verb
from fleetlab.contracts.world import DayRange, Windows
from fleetlab.faults import truth_table
from fleetlab.scoring import (
    ScoreReport,
    ScoringInputs,
    replays_needed_for,
    report_markdown,
    score,
    trivial_no_data_ranker,
)
from fleetlab.scoring.decisions import headline, random_draws
from fleetlab.scoring.diagnosis import named_flags
from fleetlab.scoring.models import (
    Achievable,
    BudgetScore,
    DecisionScore,
    FixSetValue,
    Metric,
    RandomBaseline,
)
from fleetlab.scoring.ranking import average_ranks, slope, spearman
from fleetlab.scoring.trace import float_number, score_traceability, text_numbers
from fleetlab.scoring.valuation import (
    FRESH,
    DrawValuer,
    best_removal,
    cheapest_fix,
    removal_options,
    value_or_missing,
)
from fleetlab.worlds.truth import FaultTable, FaultTruth, Role

TRUTH = truth_table()
DELAY = ConfigKey.control_action_delay_steps
STEPS = ConfigKey.policy_sampling_steps
CHUNK = ConfigKey.policy_actions_executed_per_chunk
SITE_OF = {"r01": "s1", "r02": "s1", "r03": "s2", "r04": "s2"}

# The toy world.


def toy_fleet() -> Fleet:
    def robot(robot_id: str) -> Robot:
        components = {c: ComponentState(component=c, installed_day=0) for c in Component}
        return Robot(id=robot_id, site_id=SITE_OF[robot_id], components=components)

    return Fleet(
        id="w01",
        sites=[
            Site(id="s1", robots=[robot("r01"), robot("r02")]),
            Site(id="s2", robots=[robot("r03"), robot("r04")]),
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
LATENCY = FaultSpec(
    fault_id="f02",
    kind="action_latency",
    level=17.0,
    target=Target(scope="s2", config_key=DELAY),
    schedule=Schedule(onset_day=3),
    nominal=0,
)
DECOY = FaultSpec(
    fault_id="f03",
    kind="policy_knob",
    params={"value": 5},
    level=1.0,
    target=Target(scope="s1", config_key=STEPS),
    schedule=Schedule(onset_day=1),
    nominal=10,
)
WEAR = FaultSpec(
    fault_id="f04",
    kind="gripper_wear",
    params={"friction_scale": 0.8},
    level=0.2,
    target=Target(scope="r04", component=Component.gripper),
    schedule=Schedule(onset_day=0),
)
INERT = FaultSpec(
    fault_id="f05",
    kind="policy_knob",
    params={"value": 10},
    level=1.0,
    target=Target(scope="r03", config_key=CHUNK),
    schedule=Schedule(onset_day=3),
    nominal=30,
)
EXCLUDED = {"f03", "f05"}

NOMINAL: dict[ConfigKey | ConditionKey, ConfigValue] = {
    ConfigKey.policy_sampling_steps: 10,
    ConfigKey.policy_actions_executed_per_chunk: 30,
    ConfigKey.policy_image_views: 3,
    ConfigKey.policy_precision: "float32",
    ConfigKey.control_mode: "absolute",
    ConfigKey.control_action_delay_steps: 0,
    ConfigKey.camera_scene_calibration: [0.0, 0.0, 0.0],
    ConfigKey.camera_wrist_calibration: [0.0, 0.0, 0.0],
    ConfigKey.software_version: "1.0",
    ConditionKey.lighting: "standard",
    ConditionKey.placement: "nominal",
    ConditionKey.object_batch: "a",
    ConditionKey.clutter: "none",
}


def current(identity: EpisodeIdentity) -> dict[ConfigKey | ConditionKey, ConfigValue]:
    return dict(NOMINAL)


def episode(number: int, robot_id: str, faults: list[FaultSpec], day: int = 9) -> EpisodeIdentity:
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


def scoring_episodes() -> list[EpisodeIdentity]:
    return [
        episode(1, "r01", [CAMERA, DECOY]),
        episode(2, "r01", [CAMERA, DECOY]),
        episode(3, "r01", [CAMERA, DECOY]),
        episode(4, "r02", [DECOY]),
        episode(5, "r03", [LATENCY, INERT]),
        episode(6, "r03", [LATENCY, INERT]),
        episode(7, "r04", [LATENCY, WEAR]),
        episode(8, "r04", [LATENCY, WEAR]),
        episode(9, "r02", [DECOY]),
        episode(10, "r04", [LATENCY, WEAR]),
    ]


# Success per episode by the toggled faults still in force.
OUTCOMES: dict[str, dict[frozenset[str], bool]] = {
    "e000001": {frozenset({"f01"}): False, frozenset(): True},
    "e000002": {frozenset({"f01"}): False, frozenset(): True},
    "e000003": {frozenset({"f01"}): True, frozenset(): False},  # the offset rescues it
    "e000004": {frozenset(): True},
    "e000005": {frozenset({"f02"}): False, frozenset(): True},
    "e000006": {frozenset({"f02"}): False, frozenset(): False},
    "e000007": {
        frozenset({"f02", "f04"}): False,
        frozenset({"f02"}): False,
        frozenset({"f04"}): True,
        frozenset(): True,
    },
    "e000008": {
        frozenset({"f02", "f04"}): False,
        frozenset({"f02"}): False,
        frozenset({"f04"}): False,
        frozenset(): True,
    },
    "e000009": {frozenset(): True},
    "e000010": {
        frozenset({"f02", "f04"}): True,
        frozenset({"f02"}): True,
        frozenset({"f04"}): True,
        frozenset(): True,
    },
}


def oracle_table() -> OracleTable:
    identities = scoring_episodes()
    rows = []
    for ep in identities:
        for faults in [tuple(ep.faults), *(power_set_replays(ep, EXCLUDED) or [])]:
            kept = frozenset(f.fault_id for f in faults) - EXCLUDED
            rows.append(
                OracleRow(
                    episode_id=ep.episode_id,
                    active_faults=fault_list_hash(faults),
                    success=OUTCOMES[ep.episode_id][kept],
                )
            )
    return OracleTable(identities, rows, excluded=EXCLUDED)


def truth_row(spec: FaultSpec, role: Role, detectable: bool, small: bool) -> FaultTruth:
    mapping = TRUTH[spec.kind]
    return FaultTruth(
        fault_id=spec.fault_id,
        kind=spec.kind,
        fault_class=FaultClass.configuration
        if spec.target.config_key is not None
        else FaultClass.sensor,
        target=spec.target,
        role=role,
        onset_day=spec.schedule.onset_day,
        level=spec.level,
        directions=sorted(mapping.directions),
        fixing_verbs=sorted(mapping.fixing_verbs),
        episodes={"observed": 100, "experiment": 30, "scoring": 10},
        expected_loss_points={"observed": 3.0, "experiment": 3.0, "scoring": 3.0},
        identifiable=True,
        qualified_by=["observed"] if detectable else [],
        detectable=detectable,
        small=small,
        calibration="toy",
    )


def fault_table() -> FaultTable:
    return FaultTable(
        world_id="w01",
        windows=Windows(
            observed=DayRange(first=1, last=6),
            experiment=DayRange(first=7, last=8),
            scoring=DayRange(first=9, last=10),
        ),
        episodes_per_window={"observed": 60, "experiment": 20, "scoring": 10},
        rule="toy",
        faults=[
            truth_row(CAMERA, "effective", True, False),
            truth_row(LATENCY, "effective", True, False),
            truth_row(DECOY, "decoy", False, False),
            truth_row(WEAR, "effective", False, True),
            truth_row(INERT, "inert", False, False),
        ],
    )


def act(verb: Verb, scope: str, value: Any = None, **thing: Any) -> Action:
    return Action(verb=verb, target=Target(scope=scope, **thing), value=value)


def evidence() -> list[EvidenceRef]:
    return [EvidenceRef(episode_id="e000001", step=None, table=None, query=None)]


def cause(target: Target, direction: Direction, confidence: float, mode: str = "") -> Cause:
    return Cause(
        target=target,
        direction=direction,
        confidence=confidence,
        evidence=evidence(),
        failure_mode=mode or "misses the grasp",
    )


DIAGNOSIS = Diagnosis(
    causes=[
        cause(
            Target(scope="r01", component=Component.scene_camera),
            Direction.degraded,
            0.9,
            "grasps land 1.5 cm off on 12 of 30 episodes",
        ),  # hit
        cause(Target(scope="s2", config_key=DELAY), Direction.degraded, 0.8),  # wrong direction
        cause(Target(scope="s1", config_key=STEPS), Direction.changed, 0.7),  # decoy
        cause(Target(scope="s2"), Direction.changed, 0.6),  # vague
        cause(Target(scope="r04", component=Component.gripper), Direction.degraded, 0.55),  # small
        cause(Target(scope="r02", component=Component.wrist_camera), Direction.degraded, 0.52),
        cause(Target(scope="r03", config_key=CHUNK), Direction.changed, 0.3),  # inert, not named
        cause(Target(scope="r03", config_key=DELAY), Direction.misconfigured, 0.2),  # wrong scope
    ],
    mode_summaries=["s2 fell to 62.3% while s1 held 0.95"],
    warnings=[],
    retrain=[],
)


def ranked(action: Action, expected: float, low: float, high: float, why: str = "") -> RankedAction:
    return RankedAction(
        action=action,
        expected_gain=expected,
        uncertainty=Interval(low=low, high=high),
        rationale=why or "see the query",
    )


FIX_SET = RankedFixSet(
    actions=[
        ranked(act(Verb.revert, "s2", config_key=DELAY), 20.0, 10.0, 30.0, "costs 0.1 h"),
        ranked(
            act(Verb.recalibrate, "r01", component=Component.scene_camera),
            15.0,
            12.0,
            20.0,
            "gain 15 points",
        ),
        ranked(
            act(Verb.service, "r04", component=Component.gripper),
            5.0,
            0.0,
            10.0,
            "99.9 percent sure",
        ),
        ranked(act(Verb.route_to_teleop, "fleet", "pick up the black bowl"), 3.0, 0.0, 6.0),
        ranked(act(Verb.recalibrate, "r02", component=Component.scene_camera), 1.0, 0.0, 2.0),
        ranked(act(Verb.set, "r03", 0, config_key=DELAY), 8.0, 5.0, 12.0),
        ranked(act(Verb.revert, "s1", config_key=STEPS), 2.0, 0.0, 4.0),  # the decoy: no rows
        ranked(act(Verb.clean, "r03", component=Component.wrist_camera), 0.5, 0.0, 1.0),
        ranked(act(Verb.set, "r04", 5, config_key=DELAY), 0.4, 0.0, 1.0),  # adds a change
        ranked(act(Verb.service, "r01", component=Component.arm), 0.2, 0.1, 0.5),
    ]
)


def full_catalogue() -> list[Action]:
    """The toy fleet's catalogue with every value filled: sets and conditions to their
    nominal values, one class of work, one failure mode."""
    out = []
    for entry in catalogue(toy_fleet(), task_classes=["pick up the black bowl"]):
        if entry.is_complete:
            out.append(entry)
        elif entry.verb is Verb.request_training_data:
            out.append(entry.model_copy(update={"value": "grasp slips"}))
        else:
            key = entry.target.config_key or entry.target.condition_key
            assert key is not None
            out.append(entry.model_copy(update={"value": NOMINAL[key]}))
    return out


TOOL_LOG: list[dict[str, Any]] = [
    {
        "seq": 1,
        "session": 0,
        "tool": "describe",
        "arguments": {"include_logs": True},
        "ok": True,
        "error": None,
        "result": "costs: recalibrate 0.5 h, clean 0.25 h, service 3 h, revert 0.1 h",
        "seconds": 0.1,
    },
    {
        "seq": 2,
        "session": 0,
        "tool": "query",
        "arguments": {"sql": "SELECT site_id, avg(success) FROM episodes GROUP BY 1"},
        "ok": True,
        "error": None,
        "result": ['[{"site_id": "s1", "rate": 0.95}, {"site_id": "s2", "rate": 0.6234}]'],
        "seconds": 0.2,
    },
    {
        "seq": 3,
        "session": 0,
        "tool": "query",
        "arguments": {"sql": "SELECT 15.0 AS gain"},
        "ok": True,
        "error": None,
        "result": '[{"gain": 15.0}]',
        "seconds": 0.1,
    },
    {
        "seq": 4,
        "session": 0,
        "tool": "query",
        "arguments": {"sql": "SELECT x FROM t"},
        "ok": False,
        "error": "Binder Error",
        "result": ["Binder Error: column 99.9 not found"],
        "seconds": 0.1,
    },
    {
        "seq": 5,
        "session": 1,
        "tool": "memory_read",
        "arguments": {},
        "ok": True,
        "error": None,
        "result": "notes: the delay is worth 8 points; 12 of 30 episodes miss",
        "seconds": 0.0,
    },
    {
        "seq": 6,
        "session": 1,
        "tool": "submit_fix_set",
        "arguments": FIX_SET.model_dump(mode="json"),
        "ok": True,
        "error": None,
        "result": '{"accepted": true, "receipt_id": "x0001", "problems": []}',
        "seconds": 0.0,
    },
]


def inputs(**changes: Any) -> ScoringInputs:
    base: dict[str, Any] = {
        "fault_table": fault_table(),
        "oracle": oracle_table(),
        "truth": TRUTH,
        "current": current,
        "diagnosis": DIAGNOSIS,
        "fix_set": FIX_SET,
        "tool_log": TOOL_LOG,
        "catalogue": full_catalogue(),
        "no_data_ranked": trivial_no_data_ranker(toy_fleet(), full_catalogue()),
        "seed": 11,
        "draws": 200,
    }
    base.update(changes)
    if "measured_causes" not in changes:
        base["measured_causes"] = measured_fixture(base["fault_table"], base.get("cause_units", ()))
    # Oracle-owned statistics are input to the scorer. No duplicate aggregation.
    if "oracle_chance" not in changes:
        from fleetlab.oracle.chance import chance_intervals

        base["oracle_chance"] = [
            c.model_dump(mode="json")
            for c in chance_intervals(
                base["oracle"],
                base["truth"],
                base["current"],
                base["catalogue"],
                seed=base["seed"],
                draws=base["draws"],
                achievable={},
            )
        ]
    return ScoringInputs(**base)


def measured_fixture(table: FaultTable, units: Any = ()) -> Any:
    """Explicitly constructed membership, not measured robot evidence."""
    from fleetlab.oracle.causes import MeasuredCauses

    counted = [
        {"unit": u.unit, "fault_ids": list(u.fault_ids), "qualified_by": list(u.qualified_by)}
        for u in units
        if u.detectable
    ]
    if not units:
        counted = [
            {"unit": f.fault_id, "fault_ids": [f.fault_id], "qualified_by": ["scoring"]}
            for f in table.faults
            if f.detectable and f.role == "effective"
        ]
    singles = {u["fault_ids"][0] for u in counted if len(u["fault_ids"]) == 1}
    faults = []
    for f in table.faults:
        cls = "cause" if f.fault_id in singles else "small" if f.role == "effective" else f.role
        points = 3 if cls == "cause" else 0
        window = {
            "status": "measured",
            "episodes": 100,
            "carrying": 20,
            "known": 20,
            "net": points,
            "rescued": points,
            "broken": 0,
            "points": points,
        }
        faults.append(
            {
                "item": f.fault_id,
                "fault_ids": [f.fault_id],
                "planted_role": f.role,
                "curve_class": "cause" if f.detectable else f.role,
                "measured_class": cls,
                "identifiable": True,
                "observed_carriers": 20,
                "qualified_by": ["scoring"] if cls == "cause" else [],
                "windows": {"scoring": window, "observed": window},
            }
        )
    return MeasuredCauses.model_validate(
        {
            "world_id": table.world_id,
            "world_hash": None,
            "episodes_per_window": {"observed": 100, "scoring": 100},
            "faults": faults,
            "pairs": [],
            "causes": counted,
            "cause_count": len(counted),
            "curve_causes": [f.fault_id for f in table.faults if f.detectable],
            "no_decoy_left": not any(f.role == "decoy" for f in table.faults),
            "complete": True,
            "problems": [],
        }
    )


@pytest.fixture(scope="module")
def report() -> ScoreReport:
    return score(inputs())


# The toy's own arithmetic, through the contract.


def test_the_toy_oracle_gives_the_hand_counted_gains() -> None:
    table = oracle_table()
    assert [achievable_gain(k, table).net for k in (1, 2, 3)] == [2, 3, 4]
    everything = FixSet(
        actions=[FIX_SET.actions[i].action for i in range(3)],
        budget=Budget(kind="count", amount=3),
    )
    assert value_fix_set(everything, table, TRUTH, current) == Gain.of(5, 1)


# Claim 1.


def test_claim_1_counts_hits_wrong_directions_vague_small_and_decoys(report: ScoreReport) -> None:
    c1 = report.claim1
    assert c1.detectable == 2 and c1.named_cut == 4
    verdicts = [(c.verdict, c.named) for c in c1.causes]
    assert verdicts == [
        ("hit", True),
        ("wrong direction", True),
        ("decoy", True),
        ("vague", True),
        ("small", True),
        ("no fault", True),
        ("inert", False),
        ("no fault", False),  # the key at r03, while the fault sits at s2
    ]
    assert c1.named == 6
    assert c1.recall.value == pytest.approx(0.5)  # f01 found; f02 only in the wrong direction
    assert (c1.hits, c1.false_positives) == (1, 3)
    assert c1.precision.value == pytest.approx(1 / 4)
    assert c1.precision_without_vague.value == pytest.approx(1 / 3)
    assert (c1.wrong_direction, c1.small, c1.vague) == (1, 1, 1)
    assert c1.decoy_rejection.value == 0.0
    found = {f.fault_id: f.found for f in c1.faults}
    assert found == {"f01": True, "f02": False, "f03": False, "f04": False, "f05": False}
    f02 = next(f for f in c1.faults if f.fault_id == "f02")
    assert f02.wrong_direction_by == [1]


def test_claim_1_reports_the_inert_entry_and_resolves_a_known_pair(report: ScoreReport) -> None:
    [alone] = report.claim1.confounders
    assert (alone.inert, alone.inert_named, alone.effective, alone.resolved) == (
        "f05",
        False,
        None,
        None,
    )
    assert "does not carry" in report.claim1.confounder_note
    paired = score(inputs(confounders=[("f02", "f05")])).claim1.confounders
    assert [(c.effective, c.effective_found, c.resolved) for c in paired] == [("f02", True, True)]


def test_named_is_the_confidence_or_the_top_n_plus_two_ties_in_list_order() -> None:
    def low(confidence: float) -> Cause:
        return cause(Target(scope="r01"), Direction.degraded, confidence)

    causes = [low(0.4), low(0.1), low(0.4), low(0.4), low(0.6)]
    assert named_flags(causes, 0) == [True, False, False, False, True]
    assert named_flags(causes, 1) == [True, False, True, False, True]


def test_decoys_rejected_when_no_named_cause_matches_them() -> None:
    clean = Diagnosis(
        causes=[c for i, c in enumerate(DIAGNOSIS.causes) if i != 2],
        mode_summaries=[],
        warnings=[],
        retrain=[],
    )
    c1 = score(inputs(diagnosis=clean, fix_set=None, catalogue=[])).claim1
    assert c1.decoy_rejection.value == 1.0
    assert c1.precision.value == pytest.approx(1 / 3)
    assert c1.precision_without_vague.value == pytest.approx(1 / 2)


# Claim 2.


def test_average_ranks_and_spearman_on_known_examples() -> None:
    assert average_ranks([10.0, 20.0, 20.0, 5.0]) == [2.0, 3.5, 3.5, 1.0]
    x, y = [1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 6.0, 7.0, 8.0, 7.0]
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    assert spearman(x, y) == pytest.approx(0.8207826816681233)
    assert spearman(x, [3.0] * 5) is None
    assert slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]) == pytest.approx(2.0)


def test_claim_2_values_each_ranked_action_alone(report: ScoreReport) -> None:
    c2 = report.claim2
    true = {a.rank: a.true_points for a in c2.actions}
    assert true == {
        1: 20.0,
        2: 10.0,
        3: 0.0,
        4: None,  # route_to_teleop: claim 5
        5: 0.0,
        6: 10.0,  # set to nominal on r03 overrides the site fault there: e000005
        7: None,  # the decoy's removal was never replayed
        8: 0.0,
        9: None,  # a non-nominal value adds a change
        10: 0.0,
    }
    missing = {a.rank: a.missing for a in c2.actions if a.missing}
    assert "claim 5" in (missing[4] or "")
    assert (missing[7] or "").startswith("MissingReplays: 5 episodes need a fresh replay")
    assert (missing[9] or "").startswith("MissingReplays: 3 episodes need a fresh replay")
    expected = [20.0, 15.0, 5.0, 1.0, 8.0, 0.5, 0.2]
    observed = [20.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    assert c2.valued == 7
    assert c2.spearman.value == pytest.approx(spearmanr(expected, observed).statistic)
    assert c2.spearman.value == pytest.approx((22.5 / 28) ** 0.5)
    assert c2.mae_points.value == pytest.approx(13.7 / 7)
    assert c2.slope.value == pytest.approx(np.polyfit(expected, observed, 1)[0])
    assert c2.coverage.value == pytest.approx(5 / 7)  # ranks 2 and 10 fall outside
    assert c2.top_k_share["3"].value == pytest.approx(1.0)
    assert c2.top_k_share["6"].value == pytest.approx(1.0)


# Claim 3.


def test_claim_3_values_the_cuts_and_catches_missing_replays(report: ScoreReport) -> None:
    by = {s.label: s for s in report.claim3.budgets}
    assert list(by) == ["k = 3", "k = 6", "4 h", "8 h"]
    k3 = by["k = 3"]
    assert k3.agent.gain == Gain.of(5, 1) and k3.agent.points == pytest.approx(40.0)
    assert k3.achievable.gain == Gain.of(5, 1)
    assert k3.agent.share.value == pytest.approx(1.0)
    assert by["k = 6"].agent.gain == Gain.of(5, 1)
    # Both cost cuts reach the decoy's revert and the change on r04.
    for label in ("4 h", "8 h"):
        agent = by[label].agent
        assert agent.gain is None and (agent.missing or "").startswith("MissingReplays")
        assert agent.share.value is None
    assert by["4 h"].agent.hours == pytest.approx(3.9)
    assert len(by["8 h"].agent.actions) == 10
    # The no-data ranker recalibrates scene cameras first: r01's fix and two healthy ones.
    assert k3.no_data.gain == Gain.of(2, 1)
    assert k3.no_data.share.value == pytest.approx(0.25)


def test_the_achievable_gain_under_a_cost_budget_enumerates_the_cheapest_fixes(
    report: ScoreReport,
) -> None:
    table = oracle_table()
    fixes = {
        f.fault_id: cheapest_fix(f, TRUTH, dict(inputs().costs)) for f in (CAMERA, LATENCY, WEAR)
    }
    assert {k: (v[0].verb, v[1]) for k, v in fixes.items() if v} == {
        "f01": (Verb.recalibrate, 0.5),
        "f02": (Verb.set, 0.1),
        "f04": (Verb.service, 3.0),
    }
    options, missing = removal_options(table)
    assert missing == []
    costs = {"f01": 0.5, "f02": 0.1, "f04": 3.0}
    assert best_removal(options, costs, 0.6) == (Gain.of(4, 1), ("f01", "f02"))
    assert best_removal(options, costs, 0.4) == (Gain.of(2, 0), ("f02",))
    assert best_removal(options, costs, 4.0)[0] == Gain.of(5, 1)
    for k in (1, 2, 3):  # with unit costs it is achievable_gain(k)
        unit = dict.fromkeys(costs, 1.0)
        assert best_removal(options, unit, k)[0] == achievable_gain(k, table)
    four = next(s for s in report.claim3.budgets if s.label == "4 h").achievable
    assert four.faults == ["f01", "f02", "f04"] and four.hours == pytest.approx(3.6)
    assert four.gain == Gain.of(5, 1)


def random_fix_sets(count: int) -> list[FixSet]:
    """Fix sets drawn from the catalogue and three actions that add a change."""
    changes = [
        act(Verb.set, "r04", 5, config_key=DELAY),
        act(Verb.set, "s1", 2, config_key=ConfigKey.policy_image_views),
        act(Verb.change_condition, "s2", "dim", condition_key=ConditionKey.lighting),
    ]
    entries = full_catalogue() + changes * 20
    rng = np.random.default_rng(3)
    out = []
    for _ in range(count):
        picked = {
            entries[int(i)].model_dump_json(): entries[int(i)]
            for i in rng.choice(len(entries), size=int(rng.integers(1, 8)), replace=False)
        }
        out.append(FixSet(actions=list(picked.values()), budget=Budget(kind="count", amount=8)))
    return out


def test_the_draw_valuer_agrees_with_value_fix_set() -> None:
    table = oracle_table()
    valuer = DrawValuer(table, TRUTH, current)
    seen = set()
    fix_sets = random_fix_sets(200)
    for fix_set in fix_sets:
        fast = valuer.value(fix_set.actions)
        try:
            slow: Gain | str = value_fix_set(fix_set, table, TRUTH, current)
        except MissingReplays:
            slow = FRESH
        assert fast == slow
        seen.add(fast if isinstance(fast, str) else "gain" if fast.net == 0 else "net")
    assert seen == {"fresh", "gain", "net"}
    # Once the fresh replays are in the table, both look them up, added changes too.
    filled = oracle_table()
    for fix_set in fix_sets:
        for need in replays_needed(fix_set, filled, TRUTH, current):
            filled.add(
                OracleRow(
                    episode_id=need.episode_id,
                    active_faults=fault_list_hash(need.faults),
                    success=need.episode_id < "e000006",
                )
            )
    valuer = DrawValuer(filled, TRUTH, current)
    for fix_set in fix_sets:
        valued = value_or_missing(fix_set, filled, TRUTH, current)
        assert valued.gain is not None and valuer.value(fix_set.actions) == valued.gain
    # Without `current`, a set of an untouched key cannot be resolved: an error, not a gain.
    blind = DrawValuer(table, TRUTH, None)
    image_views = act(Verb.set, "r02", 3, config_key=ConfigKey.policy_image_views)
    assert isinstance(blind.value([image_views]), str) and blind.value([image_views]) != FRESH


def test_random_fixes_are_reproducible_under_the_seed(report: ScoreReport) -> None:
    again = score(inputs())
    assert again.claim3 == report.claim3
    draws = random_draws(inputs(draws=20), SCORING_BUDGETS)
    other = random_draws(inputs(draws=20, seed=12), SCORING_BUDGETS)
    assert draws == random_draws(inputs(draws=20), SCORING_BUDGETS)
    assert draws != other
    for s in report.claim3.budgets:
        r = s.random
        assert r.draws == 200 and r.valued + r.skipped_fresh_replay + r.skipped_error == 200
        assert r.valued > 0
        if r.valued < r.draws:
            assert r.missing is not None and "unavailable" in r.missing
            assert r.mean_points.value is None and r.p95_points.value is None
        else:
            assert r.p5_points.value is not None and r.p95_points.value is not None
            assert r.mean_points.value is not None
            assert r.p5_points.value <= r.p95_points.value
    # Cost cuts of a shuffled catalogue reach the decoy's key often enough to be skipped.
    assert any(s.random.skipped_fresh_replay for s in report.claim3.budgets)


def test_the_headline_gives_a_ratio_only_above_chance() -> None:
    from fleetlab.scoring.comparisons import BASELINE_METHODS, REQUIRED_BASELINES

    def budget(no_data_share: float, p95: float) -> DecisionScore:
        achievable = Achievable(gain=Gain.of(100, 0), points=40.0)
        return DecisionScore(
            budgets=[
                BudgetScore(
                    budget=Budget(kind="count", amount=3),
                    label="k = 3",
                    world_id="w01",
                    achievable=achievable,
                    agent=FixSetValue(gain=Gain.of(75, 0), points=30.0, share=Metric.of(0.75)),
                    no_data=FixSetValue(points=10.0, share=Metric.of(no_data_share)),
                    exact_chance={
                        "p95_points": p95 * 40,
                        "histogram": [[0, 94], [int(p95 * 100), 6]],
                        "sets": 100,
                        "unvalued_sets": 0,
                    },
                    baseline_sources={
                        name: {
                            "source": f"/constructed/{name}",
                            "record": {"method": methods[0], "repeat": 1, "world_id": "w01"},
                            "binding": "constructed producer output",
                        }
                        for name, methods in BASELINE_METHODS.items()
                        if name in REQUIRED_BASELINES
                    },
                    baselines={
                        name: FixSetValue(
                            gain=Gain.of(25, 0), points=10, share=Metric.of(no_data_share)
                        )
                        for name in (
                            "no-data ranker",
                            "worst health metric",
                            "dramatic videos",
                            "classical statistics",
                        )
                    },
                    random=RandomBaseline(
                        draws=200,
                        valued=200,
                        mean_points=Metric.of(1.0),
                        mean_share=Metric.of(0.03),
                        p95_share=Metric.of(p95),
                    ),
                )
            ]
        )

    above = headline(budget(0.25, 0.1))
    assert above.ratio == pytest.approx(3.0) and above.best_baseline == "no-data ranker"
    within = headline(budget(0.25, 0.25))
    assert within.ratio is None
    assert within.verdict == "the baselines recover no more than chance"
    assert [r.method for r in above.rows] == [
        "agent",
        "random fixes (mean of 200 draws)",
        "no-data ranker",
        "worst health metric",
        "dramatic videos",
        "classical statistics",
    ]


# Missing replays, and scoring again once they are run.


def test_missing_replays_are_listed_and_scoring_again_fills_them(report: ScoreReport) -> None:
    needs = replays_needed_for(inputs())
    listed = {(n.episode_id, tuple(n.fault_ids)) for n in report.needs_fresh_replay}
    assert {(n.episode_id, tuple(sorted(f.fault_id for f in n.faults))) for n in needs} == listed
    # The decoy's removal on the five s1 episodes, with and without the camera fix.
    assert ("e000004", ()) in listed and ("e000001", ()) in listed
    assert ("e000001", ("f01",)) in listed
    changes = [n for n in needs if any(f.kind == CONFIG_CHANGE_KIND for f in n.faults)]
    assert sorted({n.episode_id for n in changes}) == ["e000007", "e000008", "e000010"]
    record = next(r for r in report.needs_fresh_replay if r.episode_id == "e000007")
    assert "claim 2: ranked action 9" in record.needed_for

    table = oracle_table()
    for need in needs:  # the caller runs them; here every fresh replay succeeds
        table.add(
            OracleRow(
                episode_id=need.episode_id,
                active_faults=fault_list_hash(need.faults),
                success=True,
            )
        )
    again = score(inputs(oracle=table))
    assert again.needs_fresh_replay == []
    assert replays_needed_for(inputs(oracle=table)) == []
    ranks = {a.rank: a.true_points for a in again.claim2.actions}
    assert ranks[7] == pytest.approx(20.0)  # e000001 and e000002 succeed without the decoy
    assert all(s.agent.gain is not None for s in again.claim3.budgets)


# Traceability.


def test_numbers_are_read_from_text_without_ids_and_words() -> None:
    text = (
        "r03 and s2 at e000123, f01, day3, x0001, v1.2.3: 12.3 points, 1,234 episodes, "
        "-4.5 in a range 1-6, 2x faster, the 3rd, 62%, 12h"
    )
    found = [(n.text, n.value, n.decimals) for n in text_numbers("t", text)]
    assert found == [
        ("12.3", 12.3, 1),
        ("1,234", 1234.0, 0),
        ("-4.5", -4.5, 1),
        ("1", 1.0, 0),
        ("6", 6.0, 0),
        ("62", 62.0, 0),
        ("12", 12.0, 0),
    ]
    assert float_number("f", 12.27).decimals == 2
    assert float_number("f", 2.0).decimals == 0
    assert float_number("f", 1e-05).decimals == 5
    assert float_number("f", 0.1234567891).decimals == 6


def test_traceability_matches_at_the_written_precision_and_flags_the_rest() -> None:
    trace = score_traceability(DIAGNOSIS, FIX_SET, TOOL_LOG)
    status = {(n.where, n.text): n.status for n in trace.numbers}
    rationale = "fix_set.actions[{}].rationale".format
    assert status[(rationale(0), "0.1")] == "traced"  # the cost table
    assert status[(rationale(1), "15")] == "echo"  # only from SELECT 15.0
    assert status[(rationale(2), "99.9")] == "untraced"  # only in a failed call
    assert status[("fix_set.actions[5].expected_gain", "8.0")] == "untraced"  # only in memory
    assert status[("fix_set.actions[0].expected_gain", "20.0")] == "untraced"
    assert status[("fix_set.actions[1].uncertainty.low", "12.0")] == "untraced"
    assert status[("fix_set.actions[9].uncertainty.high", "0.5")] == "traced"
    summary = "diagnosis.mode_summaries[0]"
    assert status[(summary, "62.3")] == "traced"  # 0.6234 in a list of text parts, divided by 100
    assert status[(summary, "0.95")] == "traced"
    mode = "diagnosis.causes[0].failure_mode"
    assert status[(mode, "1.5")] == "untraced"
    assert status[(mode, "12")] == "untraced"  # memory only
    assert status[(mode, "30")] == "untraced"
    source = next(n.source for n in trace.numbers if n.where == summary and n.text == "62.3")
    assert source == "session 0 seq 2 (query), /100"
    assert trace.traced + trace.echo + trace.untraced + trace.small == len(trace.numbers)
    assert trace.calls == 6 and trace.calls_searched == 3
    assert [j.where for j in trace.not_checked][:2] == [
        "diagnosis.causes[0].confidence",
        "diagnosis.causes[1].confidence",
    ]


# The report.


def test_the_report_renders_as_markdown_and_round_trips_as_json(report: ScoreReport) -> None:
    text = report_markdown(report)
    for heading in (
        "# Score of world w01",
        "## Headline: share of the achievable gain at k = 3",
        "## Claim 1: diagnosis",
        "## Claim 2: expected against true gains",
        "## Claim 3: fix sets at the scoring budgets",
        "## Traceability of the answer's numbers",
        "## Replays the scoring waits for",
    ):
        assert heading in text
    assert "MissingReplays: 5 episodes need a fresh replay" in text
    assert "recalibrate r01 scene_camera" in text
    assert ScoreReport.model_validate_json(report.model_dump_json()) == report


def test_a_missing_answer_finds_no_causes_and_chooses_no_fixes() -> None:
    empty = score(inputs(diagnosis=None, fix_set=None))
    assert empty.claim1.missing == "no diagnosis submitted"
    assert empty.claim2.missing == "no ranked fix set submitted"
    k3 = empty.claim3.budgets[0]
    assert empty.claim1.recall.value == 0
    assert empty.claim1.decoy_rejection.value == 0
    assert k3.agent.gain == Gain() and k3.agent.share.value == 0
    assert empty.headline.rows[0].share.value == 0
    assert empty.headline.ratio is None
    assert empty.traceability.missing == "no answer submitted"
    text = report_markdown(empty)
    assert "no diagnosis submitted; no causes found." in text
    nothing = score(inputs(catalogue=[], no_data_ranked=None))
    assert nothing.claim3.budgets[0].random.missing == "no catalogue"
    assert "required baselines missing" in nothing.headline.verdict
    report_markdown(nothing)


def test_the_no_data_ranker_follows_the_prior_and_reads_no_logs() -> None:
    ranked_list = trivial_no_data_ranker(toy_fleet(), full_catalogue())
    head = [(a.verb, a.target.scope, a.target.component) for a in ranked_list[:6]]
    assert head == [
        (Verb.recalibrate, "r01", Component.scene_camera),
        (Verb.recalibrate, "r02", Component.scene_camera),
        (Verb.recalibrate, "r03", Component.scene_camera),
        (Verb.recalibrate, "r04", Component.scene_camera),
        (Verb.recalibrate, "r01", Component.wrist_camera),
        (Verb.recalibrate, "r02", Component.wrist_camera),
    ]
    assert len(ranked_list) == len(full_catalogue())
    assert len({a.model_dump_json() for a in ranked_list}) == len(ranked_list)


def test_traceability_reads_only_delivered_data_and_sets_small_numbers_apart() -> None:
    skill = {
        "seq": 7,
        "session": 1,
        "tool": "read_skill",
        "arguments": {"name": "scope"},
        "ok": True,
        "error": None,
        "result": "use z = 1.96 and a gain of 20.0",
        "seconds": 0.0,
    }
    refused = {**TOOL_LOG[1], "seq": 8, "result": ['{"gain": 12.0}'], "delivered": False}
    trace = score_traceability(DIAGNOSIS, FIX_SET, [*TOOL_LOG, skill, refused])
    status = {(n.where, n.text): n.status for n in trace.numbers}
    assert status[("fix_set.actions[0].expected_gain", "20.0")] == "untraced"  # skill text only
    assert status[("fix_set.actions[1].uncertainty.low", "12.0")] == "untraced"  # not delivered
    assert trace.calls_searched == 3
    small = [n for n in trace.numbers if n.status == "small"]
    assert all(abs(n.value) < 10 and "." not in n.text for n in small)
    assert all(not n.where.endswith(("expected_gain", ".low", ".high")) for n in small)
