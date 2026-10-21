"""The measured-context search of round 6c: it changes how places are found, not the rules.

Each faster check is compared with the slot-by-slot original it replaces, on the real
cells; the search itself is asked to propose only what those checks support. Cells that
are thin only for want of attempts count as fifty here, so that the tests do not depend
on which top-ups have been imported.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from fleetlab.contracts.faults import Drift, FaultSpec
from fleetlab.contracts.ids import FLEET_SCOPE, scope_covers
from fleetlab.worlds import base, calibration, compose, curves, estimate
from fleetlab.worlds.compose import Composer, Draft, faults_in_force, make_fault
from fleetlab.worlds.recipe import planned_recipe

SEED = (1 << 62) + 1100000


@pytest.fixture
def filled(monkeypatch: pytest.MonkeyPatch) -> dict[curves.CellKey, curves.MeasuredCell]:
    real = curves.measured_cells()
    table = {
        key: cell
        if cell.sufficient or cell.n < 10
        else curves.MeasuredCell(
            50,
            round(50 * cell.successes / cell.n),
            round(50 * cell.logged_successes / cell.n),
            cell.sources,
        )
        for key, cell in real.items()
    }
    monkeypatch.setattr(curves, "measured_cells", lambda: table)
    return table


def schedules() -> dict[str, list[FaultSpec]]:
    delay = make_fault("f01", "action_latency", "17 steps", "r01", 2)
    pan = make_fault("f02", "camera_pose_offset", "6 deg scene pan", "r01", 0)
    batch = make_fault("f03", "object_batch_variant", "4 mass doublings", "s3", 1)
    worn = make_fault("f04", "gripper_wear", "0.5 wear", "r09", 0)
    pair = [
        make_fault("f05", "action_latency", "8 steps", "r02", 0),
        make_fault("f06", "policy_knob", "15 actions per chunk", "r02", 1),
    ]
    drift = make_fault(
        "f07",
        "gripper_wear",
        "1 wear",
        "r03",
        0,
        level=0.01,
        drift=Drift(shape="linear", rate=0.09),
    )
    release = make_fault(
        "f08",
        "software_release",
        "camera_degradation|3 px blur",
        FLEET_SCOPE,
        3,
        nominal_version="4.2.0",
        version="4.3.0",
    )
    site_delay = make_fault("f09", "action_latency", "17 steps", "s1", 2)
    return {
        "nothing": [],
        "one delay": [delay],
        "an unmeasured overlap": [delay, pan],
        "the additive pair": [batch, worn],
        "the interaction": pair,
        "a drift": [drift],
        "a release over a delay": [release, delay],
        "a site-wide delay": [site_delay],
        "several": [batch, worn, *pair, drift, delay],
    }


def reference_floor(
    comp: Composer, schedule: list[FaultSpec], ctx: estimate.Context, scope: str | None
) -> bool:
    """The check as it stood before the search read it robot by robot and day by day."""
    assert ctx.measured is not None
    try:
        for slot in comp.slots:
            if scope is not None and not scope_covers(scope, slot.robot_id, slot.site_id):
                continue
            active = base.in_force(schedule, slot.robot_id, slot.site_id, slot.day)
            for removed in (None, *(f.fault_id for f in active)):
                curves.context_weights(
                    [f for f in active if f.fault_id != removed],
                    slot.suite,
                    slot.task.task_index,
                    ctx.measured,
                )
    except KeyError:
        return False
    floor = comp.recipe.robot_success_floor
    assert floor is not None
    # The original averaged every robot's windows and then kept the rows in scope, which
    # raised when a robot outside the scope had an unmeasured context. Only the robots
    # in scope are asked here, as the search does.
    inside = [
        slot
        for slot in comp.slots
        if scope is None or scope_covers(scope, slot.robot_id, slot.site_id)
    ]
    rows = estimate.robot_window_success(
        schedule, inside, replace(ctx, weights_cache=None), comp.recipe.windows
    )
    return all(min(row["true"], row["logged"]) >= floor + ctx.success_margin for row in rows)


def test_the_grouped_floor_check_agrees_with_the_slot_by_slot_one(filled: Any) -> None:
    comp = Composer(planned_recipe("D3", world_id="w99", seed=SEED))
    ctx = comp.context(Draft())
    assert ctx.weights_cache is not None
    seen = set()
    for name, schedule in schedules().items():
        for robot, site in comp.robots:
            for day in range(comp.recipe.days):
                assert faults_in_force(schedule, robot, site, day) == base.in_force(
                    schedule, robot, site, day
                ), name
        for scope in (None, "r01", "r02", "r03", "s1", "s3"):
            fast = comp.floor_holds(schedule, Draft(), scope)
            assert fast is reference_floor(comp, schedule, ctx, scope), (name, scope)
            seen.add(fast)
    assert seen == {True, False}  # both answers occur, or the comparison shows nothing
    assert not comp.floor_holds(schedules()["an unmeasured overlap"], Draft(), "r01")
    assert comp.floor_holds([], Draft())


def test_the_shared_lookup_gives_the_same_cells_and_keeps_refusing(filled: Any) -> None:
    comp = Composer(planned_recipe("D3", world_id="w99", seed=SEED))
    shared = comp.context(Draft())
    plain = replace(shared, weights_cache=None)
    cases = schedules()
    for faults in (cases["nothing"], cases["one delay"], cases["the interaction"]):
        for task in range(10):
            for _ in range(2):  # the second answer comes from the table
                assert shared.weights(faults, curves.SPATIAL, task) == plain.weights(
                    faults, curves.SPATIAL, task
                )
    overlap = cases["an unmeasured overlap"]
    for _ in range(2):
        with pytest.raises(KeyError, match="unmeasured task context"):
            shared.weights(overlap, curves.SPATIAL, 0)
    # A fault's identity, scope and onset do not enter: the cells are keyed by mechanism.
    twin = make_fault("f77", "action_latency", "17 steps", "s2", 4)
    assert estimate.mechanism_signature(twin) == estimate.mechanism_signature(cases["one delay"][0])
    other = make_fault("f01", "action_latency", "14 steps", "r01", 2)
    assert estimate.mechanism_signature(other) != estimate.mechanism_signature(
        cases["one delay"][0]
    )


@pytest.mark.parametrize(("margin_of", "on_a_site"), (("carriers", 0.02), ("fleet", 0.06)))
def test_the_grouped_decoy_check_agrees_with_the_slot_by_slot_one(
    filled: Any, margin_of: str, on_a_site: float
) -> None:
    recipe = planned_recipe("D3", world_id="w99", seed=SEED)
    comp = Composer(recipe.model_copy(update={"decoy_margin_of": margin_of}))
    answers = set()
    for kind, label in compose.DECOYS:
        for site in ("s1", "s3"):
            decoy = make_fault("f20", kind, label, site, 2)
            delay = make_fault("f21", "action_latency", "17 steps", "r01", 1)
            batch = make_fault("f22", "object_batch_variant", "4 mass doublings", site, 3)
            margin = comp.decoy_margin(decoy)
            assert margin == pytest.approx(on_a_site)
            for others in ([], [delay], [batch]):
                schedule = [decoy, *others]
                fast = comp.decoys_flat(schedule, [decoy.fault_id])
                slow = calibration.context_decoy_confirmed(
                    decoy, schedule, list(comp.slots), margin
                )
                assert fast is slow, (kind, site, [f.kind for f in others])
                answers.add(fast)
            # A candidate whose scope cannot reach the decoy's carriers leaves it unasked.
            elsewhere = "r05" if site != "s2" else "r01"
            assert comp.decoys_flat([decoy, batch], [decoy.fault_id], elsewhere)
            assert not comp.decoys_flat([decoy, batch], [decoy.fault_id], site)
    assert answers == {True, False}


def test_the_measured_draft_proposes_only_what_stands_on_measured_ground(filled: Any) -> None:
    """Whatever an attempt returns, finished or given up, every context of it is measured,
    every robot window is above the floor and every decoy is flat under what covers it.

    The older draft drew the batch's site, the decoy's site and the drift's robot without
    looking: with it this fails on the first attempts.
    """
    for name in ("D2", "D3"):
        assert name in ("D2", "D3")
        recipe = planned_recipe(name, world_id="w99", seed=SEED + 5).model_copy(
            update={"interaction_scope": "robot", "plain": 3}
        )
        comp = Composer(recipe)
        placed = 0
        for attempt in range(12):
            d = comp.draft(attempt)
            schedule = comp.schedule(d)
            placed += len(schedule)
            assert comp.supported(schedule, d), (name, attempt, d.stopped)
            for fault in schedule:
                if fault.fault_id in d.patterns.decoys:
                    assert calibration.context_decoy_confirmed(
                        fault, schedule, list(comp.slots), comp.decoy_margin(fault)
                    )
            ids = [f.fault_id for f in schedule]
            assert len(ids) == len(set(ids)) and all(ids)
        if name == "D3":
            assert placed > 0  # attempts that place nothing would show nothing


@pytest.mark.parametrize("repeats", (1, 3))
def test_plain_faults_of_one_mechanism_at_one_kind_of_scope_keep_the_recipes_cap(
    filled: Any, repeats: int
) -> None:
    """`repeats`: one in the third generation's D2, three in D3 and, from the fourth
    generation on, in the recipes with a fleet-wide release. A release counts as what
    it carries; the confounder's fault is drawn from the same menu and counts too."""
    from collections import Counter

    recipe = planned_recipe("D3", world_id="w99", seed=SEED).model_copy(update={"repeats": repeats})
    comp = Composer(recipe)
    most = 0
    for attempt in range(30):
        counts: Counter[tuple[str, str]] = Counter()
        for plain in comp.draft(attempt).plains:
            mechanism = plain.kind
            if plain.kind == "software_release":
                mechanism = plain.labels[0].split("|")[0]
            counts[mechanism, "robot" if plain.scope.startswith("r") else "site"] += 1
        assert all(count <= repeats for count in counts.values()), (attempt, counts)
        most = max([most, *counts.values()])
    assert most == repeats  # the cap is reached, so it is the cap that holds the count


def test_refusals_say_where_the_search_stopped(filled: Any) -> None:
    recipe = planned_recipe("D3", world_id="w99", seed=SEED).model_copy(
        update={"attempts": 6, "plain": 30}
    )  # deliberately exceeds the available physical slots, even after search improves
    with pytest.raises(ValueError, match="no composition in 6 attempts") as refused:
        compose.compose(recipe)
    message = str(refused.value)
    assert "draft incomplete" not in message
    assert any(
        reason in message
        for reason in (
            "plain faults fit measured contexts",
            "no interaction carrier",
            "no site on which a decoy",
            "no robot left for a drift",
            "no site left where the additive pair",
            "expected observed success",
        )
    )


def test_a_measured_drift_crosses_in_the_scoring_window_with_a_precursor(filled: Any) -> None:
    recipe = planned_recipe("D3", world_id="w99", seed=SEED)
    comp = Composer(recipe)
    ctx = comp.context(Draft())
    found = 0
    for robot, _ in comp.robots[:6]:
        rate = comp.measured_drift_rate(robot, "gripper_wear", "1 wear", 1.0)
        assert comp.measured_drift_rate(robot, "gripper_wear", "1 wear", 1.0) == rate  # kept
        if rate is None:
            continue
        found += 1
        fault = make_fault(
            "f99",
            "gripper_wear",
            "1 wear",
            robot,
            0,
            level=0.01,
            drift=Drift(shape="linear", rate=rate),
        )
        day = estimate.drift_crossing_day(fault, comp.slots, ctx)
        assert day in recipe.windows.scoring
        assert compose.drift_precursor(fault, recipe, day) is not None
        assert 0.01 + rate * (recipe.days - 1) <= 1.0 + 1e-9
    assert found > 0


def test_no_draft_forecasts_less_than_the_bound(filled: Any) -> None:
    """`forecast_bound` is from below: whatever the search drafts lies at or above it."""
    drafted = 0
    for seed in (SEED, SEED + 5):
        recipe = planned_recipe("D3", world_id="w99", seed=seed).model_copy(
            update={"interaction_scope": "robot", "plain": 3}
        )
        bound = compose.forecast_bound(recipe)
        assert bound["bound"] is not None and 0.3 < bound["bound"] < 1.0
        lower = compose.forecast_bound(recipe, floor=0.2)["bound"]
        assert lower is not None and lower <= bound["bound"]  # a lower floor allows no less
        # The third generation's floor and its margin on the carriers allow no less either.
        third = recipe.model_copy(
            update={"robot_success_floor": 0.40, "decoy_margin_of": "carriers"}
        )
        stricter = compose.forecast_bound(third)["bound"]
        assert stricter is not None and stricter >= bound["bound"] - 1e-9
        assert compose.forecast_bound(third, floor=0.30)["bound"] <= stricter + 1e-9
        comp = Composer(recipe)
        for attempt in range(25):
            d = comp.draft(attempt)
            if d.stopped or not d.patterns.confounders:
                continue
            schedule = comp.settle(d, attempt)
            assert comp.observed_logged(schedule, comp.context(d)) >= bound["bound"] - 1e-9
            drafted += 1
    assert drafted > 0


def test_the_assessment_counts_every_attempt_under_the_rule_that_refused_it(filled: Any) -> None:
    result = compose.assess(("D3",), compose.ASSESSMENT_SEEDS[:2], attempts=5)
    outcome = result["D3"]
    assert outcome["seeds"] == 2 and len(outcome["rows"]) == 2
    assert outcome["accepted"] == sum(1 for row in outcome["rows"] if row["accepted"])
    for row in outcome["rows"]:
        if row["accepted"]:
            continue
        expected = 1 if row.get("before_any_attempt") else 5
        assert sum(row["refusals"].values()) >= expected
        assert "draft incomplete" not in row["refusals"]
    assert len(compose.ASSESSMENT_SEEDS) == 20 and len(set(compose.ASSESSMENT_SEEDS)) == 20
    assert all(seed >= 1 << 62 for seed in compose.ASSESSMENT_SEEDS)


def test_the_bound_needs_a_site_that_takes_a_confirmed_decoy(
    filled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decoy's site counts only in a context that confirms a decoy on its tasks."""
    recipe = planned_recipe("D3", world_id="w99", seed=SEED).model_copy(
        update={"interaction_scope": "robot"}
    )
    monkeypatch.setattr(calibration, "decoy_flat", lambda kind, label, mixes, *rest: False)
    refused = compose.forecast_bound(recipe)
    assert refused["bound"] is None and "decoy" in refused["reason"]
    monkeypatch.setattr(calibration, "decoy_flat", lambda kind, label, mixes, *rest: bool(mixes))
    anywhere = compose.forecast_bound(recipe)["bound"]
    without = compose.forecast_bound(recipe.model_copy(update={"decoys": 0}))["bound"]
    assert anywhere is not None and without is not None
    assert without <= anywhere + 1e-9  # no decoy to place can only allow less
