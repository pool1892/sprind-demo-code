"""Conditional gains intervals on synthetic binary observations only.

Run the actual published SQL. SciPy supplies independent Wilson bounds; the
explicit coefficient vectors below encode which observations each fallback uses.
These arithmetic regressions do not establish empirical coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from scipy.stats import binomtest, norm
from skills_support import helpers

from_logs = helpers("sensitivity-from-logs")


@dataclass(frozen=True)
class Cell:
    robot: str
    task: str | None
    successes: int
    n: int
    day: int = 0


@dataclass(frozen=True)
class Result:
    output: dict[str, Any]
    raw: dict[str, Any]

    @property
    def gap(self) -> tuple[float, float, float]:
        point = self.raw["expected_rate"] - self.raw["rate"]
        return point, point - self.raw["low_distance"], point + self.raw["high_distance"]

    @property
    def change(self) -> tuple[float, float, float]:
        point = self.gap[0] - self.raw["gap_before"]
        return (
            point,
            point - self.raw["change_low_distance"],
            point + self.raw["change_high_distance"],
        )


def run_gains(
    cells: list[Cell],
    *,
    from_day: int = 0,
    attribution: object = 1.0,
    suspects: bool = False,
    **params: object,
) -> Result:
    observations = [
        {
            "episode_id": f"e{j}-{i}",
            "robot_id": cell.robot,
            "site_id": "site",
            "day": cell.day,
            "task_text": cell.task,
            "outcome": "success" if i < cell.successes else "failure",
            "experiment_id": None,
        }
        for j, cell in enumerate(cells)
        for i in range(cell.n)
    ]
    candidates = [("action", "target", from_day, attribution)]
    if suspects:
        candidates.append(("other action", "suspect", 0, 1.0))
    sql = from_logs.fill(from_logs.blocks()["gains"], params, candidates)
    with duckdb.connect() as con:
        con.register("episodes", pa.Table.from_pylist(observations))
        outputs = con.execute(sql).to_arrow_table().to_pylist()
        # Also inspect unrounded endpoints: final output intentionally retains its
        # existing precision (three decimal places for gaps, two for fleet points).
        raw_sql = sql.rsplit("\nSELECT i.label", 1)[0] + "\nSELECT * FROM intervals"
        raw = con.execute(raw_sql).to_arrow_table().to_pylist()
    return Result(
        next(r for r in outputs if r["label"] == "action"),
        next(r for r in raw if r["label"] == "action"),
    )


def mover(terms: Sequence[tuple[float, int, int]], z: float = 1.96) -> tuple[float, float, float]:
    """Reference arithmetic on hand-specified, already combined cell coefficients."""
    point = math.fsum(c * s / n for c, s, n in terms)
    lower_sq, upper_sq = [], []
    for coefficient, successes, n in terms:
        rate = successes / n
        bounds = binomtest(successes, n).proportion_ci(
            confidence_level=2 * norm.cdf(z) - 1, method="wilson"
        )
        down, up = rate - bounds.low, bounds.high - rate
        if coefficient < 0:
            down, up = up, down
        lower_sq.append((coefficient * down) ** 2)
        upper_sq.append((coefficient * up) ** 2)
    return point, point - math.sqrt(math.fsum(lower_sq)), point + math.sqrt(math.fsum(upper_sq))


def check_interval(
    result: Result, expected: tuple[float, float, float], *, change: bool = False
) -> None:
    actual = result.change if change else result.gap
    assert actual == pytest.approx(expected, abs=1e-12)
    point, low, high = actual
    assert low <= point <= high
    scale = 100 * result.raw["share"] * result.raw["attributable"]
    keys = (
        ("gain_points_change", "gain_change_low", "gain_change_high")
        if change
        else ("gain_points", "gain_low", "gain_high")
    )
    assert result.output[keys[0]] == pytest.approx(point * scale, abs=0.005000001)
    assert low * scale - 0.010000001 <= result.output[keys[1]] <= low * scale + 1e-12
    assert high * scale - 1e-12 <= result.output[keys[2]] <= high * scale + 0.010000001
    assert result.output[keys[1]] <= result.output[keys[0]] <= result.output[keys[2]]
    if not change:
        assert result.output["gap"] == pytest.approx(point, abs=0.000500001)
        assert low - 0.001000001 <= result.output["gap_low"] <= low + 1e-12
        assert high - 1e-12 <= result.output["gap_high"] <= high + 0.001000001


def test_extreme_outcomes_have_one_sided_uncertainty_and_complement_symmetry() -> None:
    positive = run_gains([Cell("target", "A", 0, 10), Cell("control", "A", 90, 90)])
    negative = run_gains([Cell("target", "A", 10, 10), Cell("control", "A", 0, 90)])
    expected = mover([(-1, 0, 10), (1, 90, 90)])
    assert expected == pytest.approx((1, 0.719457, 1), abs=1e-6)
    check_interval(positive, expected)
    check_interval(negative, (-expected[0], -expected[2], -expected[1]))
    assert positive.output["gain_points"] == 10
    assert positive.output["gain_low"] == 7.19
    # A floating endpoint just above the integer may consume one outward unit.
    assert 10 <= positive.output["gain_high"] <= 10.01


def test_fractional_exposure_limits_remain_inside_reported_interval() -> None:
    positive = run_gains([Cell("target", "A", 0, 2), Cell("control", "A", 5, 5)])
    negative = run_gains([Cell("target", "A", 2, 2), Cell("control", "A", 0, 5)])
    expected = mover([(-1, 0, 2), (1, 5, 5)])
    check_interval(positive, expected)
    check_interval(negative, (-expected[0], -expected[2], -expected[1]))
    exact_exposure = 200 / 7
    assert positive.output["gain_high"] == 28.58
    assert negative.output["gain_low"] == -28.58
    assert positive.output["gain_high"] >= exact_exposure
    assert negative.output["gain_low"] <= -exact_exposure


def test_unused_task_controls_do_not_narrow_the_gap_interval() -> None:
    cells = [Cell("target", "A", 0, 100), Cell("control", "A", 1, 2)]
    original = run_gains(cells)
    added = run_gains([*cells, Cell("control", "B", 500, 1000)])
    expected = mover([(-1, 0, 100), (1, 1, 2)])
    assert expected == pytest.approx((0.5, 0.092844, 0.905471), abs=1e-6)
    check_interval(original, expected)
    check_interval(added, expected)
    assert original.raw["share"] != added.raw["share"]
    assert (original.output["n_comparison"], added.output["n_comparison"]) == (2, 1002)


def test_unequal_task_weights_and_custom_z_scale_only_at_final_output() -> None:
    cells = [
        Cell("target", "A", 1, 4),
        Cell("target", "B", 9, 12),
        Cell("control", "A", 3, 10),
        Cell("control", "B", 2, 4),
        Cell("control", "C", 23, 30),
    ]
    terms = [(-0.25, 1, 4), (-0.75, 9, 12), (0.25, 3, 10), (0.75, 2, 4)]
    result = run_gains(cells, attribution=0.37, z=1.5)
    check_interval(result, mover(terms, z=1.5))
    assert result.output["gap"] == -0.175
    assert result.raw["share"] == 16 / 60
    zero = run_gains(cells, attribution=0)
    check_interval(zero, mover(terms))
    assert (zero.output["gain_points"], zero.output["gain_low"], zero.output["gain_high"]) == (
        0,
        0,
        0,
    )


@pytest.mark.parametrize(
    ("controls", "terms", "expected_rate", "n_comparison"),
    [
        pytest.param(
            [
                Cell("control", "A", 6, 10),
                Cell("control", "B", 1, 10),
                Cell("suspect", "A", 10, 10),
            ],
            [(1.0, 6, 10)],
            0.6,
            20,
            id="clean-task",
        ),
        pytest.param(
            [Cell("control", "B", 2, 4), Cell("suspect", "A", 0, 10)],
            [(1.0, 2, 4)],
            0.5,
            4,
            id="clean-pool-before-rest-task",
        ),
        pytest.param(
            [Cell("suspect", "A", 1, 2), Cell("suspect", "B", 0, 7)],
            [(1.0, 1, 2)],
            0.5,
            0,
            id="rest-task",
        ),
        pytest.param(
            [Cell("suspect", "B", 3, 4), Cell("suspect", "C", 1, 2)],
            [(4 / 6, 3, 4), (2 / 6, 1, 2)],
            4 / 6,
            0,
            id="rest-pool",
        ),
    ],
)
def test_each_fallback_preserves_points_and_uses_its_actual_controls(
    controls: list[Cell],
    terms: list[tuple[float, int, int]],
    expected_rate: float,
    n_comparison: int,
) -> None:
    result = run_gains([Cell("target", "A", 0, 8), *controls], suspects=True)
    assert result.raw["expected_rate"] == pytest.approx(expected_rate)
    assert result.raw["n_comparison"] == n_comparison
    check_interval(result, mover([(-1, 0, 8), *terms]))
    # The same fallback also supplies pre-period points and uncertainty.
    changed = run_gains(
        [
            Cell("target", "A", 4, 8, day=0),
            Cell("target", "A", 0, 8, day=1),
            *controls,
            *(replace(c, day=1) for c in controls),
        ],
        suspects=True,
        from_day=1,
    )
    assert changed.raw["gap_before"] == pytest.approx(expected_rate - 0.5)
    check_interval(
        changed,
        mover([(-1, 0, 8), *terms, (1, 4, 8), *((-c, s, n) for c, s, n in terms)]),
        change=True,
    )


@pytest.mark.parametrize("suspects", [False, True], ids=["clean", "rest"])
def test_shared_fallback_pool_coefficients_are_added_before_squaring(suspects: bool) -> None:
    control = "suspect" if suspects else "control"
    cells = [Cell("target", "A", 0, 10), Cell("target", "B", 2, 30), Cell(control, "C", 2, 5)]
    result = run_gains(cells, suspects=suspects)
    expected = mover([(-0.25, 0, 10), (-0.75, 2, 30), (1.0, 2, 5)])
    check_interval(result, expected)
    # Treating the same five controls as two independent samples is falsely narrower.
    duplicated = mover([(-0.25, 0, 10), (-0.75, 2, 30), (0.25, 2, 5), (0.75, 2, 5)])
    assert expected[1] < duplicated[1] and expected[2] > duplicated[2]


@pytest.mark.parametrize("suspects", [False, True], ids=["clean", "rest"])
def test_task_and_pooled_fallback_overlap_is_combined(suspects: bool) -> None:
    control = "suspect" if suspects else "control"
    cells = [
        Cell("target", "A", 1, 10),
        Cell("target", "B", 3, 30),
        Cell(control, "A", 4, 8),
        Cell(control, "C", 1, 2),
    ]
    result = run_gains(cells, suspects=suspects)
    # A's eight controls serve both A directly and B through the ten-control pool.
    check_interval(result, mover([(-0.25, 1, 10), (-0.75, 3, 30), (0.85, 4, 8), (0.15, 1, 2)]))


def test_including_suspects_preserves_the_original_comparison_policy() -> None:
    cells = [Cell("target", "A", 1, 10), Cell("control", "A", 1, 2), Cell("suspect", "A", 0, 8)]
    clean = run_gains(cells, suspects=True)
    included = run_gains(cells, suspects=True, exclude_suspects=False)
    check_interval(clean, mover([(-1, 1, 10), (1, 1, 2)]))
    # With exclusion disabled, all ten outside observations form one clean task cell.
    check_interval(included, mover([(-1, 1, 10), (1, 1, 10)]))
    assert (clean.output["n_comparison"], included.output["n_comparison"]) == (2, 10)


def test_swapping_scope_and_comparison_reverses_the_gap_bounds() -> None:
    cells = [Cell("target", "A", 1, 7), Cell("control", "A", 8, 13)]
    original = run_gains(cells)
    swapped = run_gains(
        [replace(c, robot="control" if c.robot == "target" else "target") for c in cells]
    )
    point, low, high = original.gap
    check_interval(swapped, (-point, -high, -low))


def test_sparse_pre_controls_contribute_uncertainty_to_change() -> None:
    cells = [
        Cell("target", "A", 5, 10, day=0),
        Cell("control", "A", 1, 1, day=0),
        Cell("target", "A", 1, 10, day=1),
        Cell("control", "A", 9, 10, day=1),
    ]
    sparse = run_gains(cells, from_day=1)
    terms = [(-1, 1, 10), (1, 9, 10), (1, 5, 10), (-1, 1, 1)]
    check_interval(sparse, mover(terms), change=True)
    assert sparse.raw["gap_before"] == 0.5
    assert sparse.change[0] == pytest.approx(0.3)
    dense = run_gains(
        [replace(c, successes=100, n=100) if c.n == 1 else c for c in cells], from_day=1
    )
    check_interval(dense, mover([*terms[:-1], (-1, 100, 100)]), change=True)
    assert sparse.change[0] == dense.change[0]
    assert sparse.change[2] > dense.change[2]  # uncertainty in the reversed pre comparator


def test_change_uses_each_periods_task_weights_and_complements() -> None:
    cells = [
        Cell("target", "A", 1, 4, day=1),
        Cell("target", "B", 9, 12, day=1),
        Cell("control", "A", 3, 10, day=1),
        Cell("control", "B", 2, 4, day=1),
        Cell("target", "A", 2, 12, day=0),
        Cell("target", "B", 0, 4, day=0),
        Cell("control", "A", 1, 2, day=0),
        Cell("control", "B", 9, 10, day=0),
    ]
    result = run_gains(cells, from_day=1, attribution=0.43, project_to_day=3)
    post_terms = [(-0.25, 1, 4), (-0.75, 9, 12), (0.25, 3, 10), (0.75, 2, 4)]
    reversed_pre = [(0.75, 2, 12), (0.25, 0, 4), (-0.75, 1, 2), (-0.25, 9, 10)]
    check_interval(result, mover(post_terms))
    expected = mover([*post_terms, *reversed_pre])
    check_interval(result, expected, change=True)
    complemented = run_gains([replace(c, successes=c.n - c.successes) for c in cells], from_day=1)
    check_interval(complemented, (-expected[0], -expected[2], -expected[1]), change=True)


def test_drift_scenario_is_bounded_without_changing_observed_point() -> None:
    cells = [Cell("target", "A", successes, 10, day=day) for day, successes in enumerate((8, 4, 0))]
    cells += [Cell("control", "A", 10, 10, day=day) for day in range(3)]
    result = run_gains(cells, attribution=0.4, project_to_day=3)
    check_interval(result, mover([(-1, 12, 30), (1, 30, 30)]))
    assert result.output["gain_points"] == 12
    assert result.output["slope_per_day"] == 0.4
    assert result.output["raw_projected_gain_points"] == 28
    assert result.output["projected_gain_points"] == 20
    assert result.output["projection_clipped"] is True
    assert result.output["projection_observed_days"] == 3
    assert result.output["projection_first_day"] == 0
    assert result.output["projection_last_day"] == 2


@pytest.mark.parametrize(
    ("successes", "raw", "bounded"), [((10, 6, 2), 40, 30), ((0, 4, 8), -30, -20)]
)
def test_projection_bounds_depend_on_comparator_rate(
    successes: tuple[int, int, int], raw: float, bounded: float
) -> None:
    cells = [Cell("target", "A", n, 10, day=day) for day, n in enumerate(successes)]
    cells += [Cell("control", "A", 6, 10, day=day) for day in range(3)]
    result = run_gains(cells, project_to_day=3)
    assert result.output["raw_projected_gain_points"] == raw
    assert result.output["projected_gain_points"] == bounded
    assert result.output["projection_clipped"] is True
    zero = run_gains(cells, attribution=0, project_to_day=3)
    assert zero.output["raw_projected_gain_points"] == 0
    assert zero.output["projected_gain_points"] == 0
    assert zero.output["projection_clipped"] is True


@pytest.mark.parametrize("day", [-1, None, "NaN", "Infinity", "-Infinity"])
def test_disabled_or_nonfinite_projection_is_unavailable(day: object) -> None:
    cells = [Cell("target", "A", 1, 10, day=d) for d in range(2)]
    cells += [Cell("control", "A", 9, 10, day=d) for d in range(2)]
    result = run_gains(cells, project_to_day=day)
    for key in ("raw_projected_gain_points", "projected_gain_points", "projection_clipped"):
        assert result.output[key] is None


def test_finite_day_that_overflows_gain_is_unavailable() -> None:
    cells = [Cell("target", "A", n, 10, day=d) for d, n in enumerate((8, 4))]
    cells += [Cell("control", "A", 10, 10, day=d) for d in range(2)]
    result = run_gains(cells, project_to_day=1e308)
    for key in ("raw_projected_gain_points", "projected_gain_points", "projection_clipped"):
        assert result.output[key] is None


@pytest.mark.parametrize("controls", [False, True])
def test_single_day_or_missing_comparator_has_no_projection(controls: bool) -> None:
    cells = [Cell("target", "A", 1, 10)]
    if controls:
        cells.append(Cell("control", "A", 9, 10))
    else:
        cells.append(Cell("target", "A", 0, 10, day=1))
    result = run_gains(cells, project_to_day=3)
    assert result.output["projection_observed_days"] == int(controls)
    for key in ("raw_projected_gain_points", "projected_gain_points", "projection_clipped"):
        assert result.output[key] is None


@pytest.mark.parametrize("missing", ["pre-period", "pre-controls", "post-controls", "all-controls"])
def test_missing_support_is_unavailable_even_with_zero_attribution(missing: str) -> None:
    cells = [Cell("target", "A", 0, 10, day=1)]
    if missing != "pre-period":
        cells.append(Cell("target", "A", 1, 10, day=0))
    if missing not in ("pre-period", "pre-controls", "all-controls"):
        cells.append(Cell("control", "A", 1, 2, day=0))
    if missing not in ("post-controls", "all-controls"):
        cells.append(Cell("control", "A", 1, 2, day=1))
    result = run_gains(cells, from_day=1, attribution=0)
    for key in ("gain_points_change", "gain_change_low", "gain_change_high"):
        assert result.output[key] is None
    assert result.raw["change_low_distance"] is None
    assert result.raw["change_high_distance"] is None
    if missing in ("post-controls", "all-controls"):
        for key in ("gap", "gap_low", "gap_high", "gain_points", "gain_low", "gain_high"):
            assert result.output[key] is None
        assert result.raw["low_distance"] is None
        assert result.raw["high_distance"] is None
    else:
        check_interval(result, mover([(-1, 0, 10), (1, 1, 2)]))


@pytest.mark.parametrize("z", [0, -1.96, None, "NaN", "Infinity", "-Infinity"])
def test_invalid_z_is_rejected_in_the_sql(z: object) -> None:
    with pytest.raises(duckdb.InvalidInputException, match="z must be positive and finite"):
        run_gains([Cell("target", "A", 0, 10), Cell("control", "A", 1, 2)], z=z)


@pytest.mark.parametrize("attribution", [-0.1, 1.1, None, "NaN", "Infinity", "-Infinity"])
def test_invalid_attribution_is_rejected_in_the_sql(attribution: object) -> None:
    with pytest.raises(duckdb.InvalidInputException, match=r"attributable must be in \[0, 1\]"):
        run_gains([Cell("target", "A", 0, 10), Cell("control", "A", 1, 2)], attribution=attribution)
