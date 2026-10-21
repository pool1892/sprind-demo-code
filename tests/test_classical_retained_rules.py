"""Focused controls for previously surviving removals of retained choices."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from classical_planted import planted
from test_classical_baseline import write_export
from test_classical_statistics import noisy, set_rates

from fleetlab.baselines.classical import analyse
from fleetlab.baselines.classical.analysis import named_actions
from fleetlab.baselines.classical.data import load_export
from fleetlab.baselines.classical.experiments import plan
from fleetlab.baselines.classical.features import Design
from fleetlab.baselines.classical.health import _trajectory, health_series
from fleetlab.baselines.classical.regression import Regression, fit_survival, predict, regress
from fleetlab.baselines.classical.selection import JointPrediction
from fleetlab.baselines.classical.signals import outcome_check
from fleetlab.contracts.model import ConfigVersion
from fleetlab.contracts.vocabulary import ConfigKey


def test_one_naming_floor_and_multiplicity_screen() -> None:
    template = dict(
        changed_features=["exposure"], latent_candidate=False, aliases=[], signature=None
    )
    rows = [
        dict(template, gain_points=0.49, diagnosis_confidence=1.0),
        dict(template, gain_points=0.51, diagnosis_confidence=0.94),
    ]
    assert named_actions(rows) == {1}
    # A lone 6% tail among 20 hypotheses does not pass BH; promoting every
    # experiment or screening only the previously named ones would pass it.
    rows.extend(dict(template, gain_points=1.0, diagnosis_confidence=0.1) for _ in range(19))
    assert named_actions(rows) == set()
    for r in rows:
        r["estimate_source"] = "experiments"
    assert named_actions(rows) == set()


def test_noise_free_parameter_draws_still_have_future_panel_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.baselines.classical import analysis

    original = regress

    def fixed_draws(*args: Any, **kwargs: Any) -> Regression:
        fitted = original(*args, **kwargs)
        return replace(fitted, draws=np.tile(fitted.beta, (kwargs["bootstraps"], 1)))

    monkeypatch.setattr(analysis, "regress", fixed_draws)
    data = noisy()
    set_rates(data, "r02", 2, 0.3)
    answer = analyse(data, bootstraps=20)
    rec = next(
        r
        for r in answer.evidence["actions"]
        if r["action"]["target"]["scope"] == "r02" and r["action"]["target"]["config_key"]
    )
    assert rec["prediction_radius"] > 0.5
    assert rec["high"] >= rec["gain_points"] + rec["prediction_radius"] - 1e-9
    assert rec["low"] <= rec["gain_points"] - rec["prediction_radius"] + 1e-9
    for r in answer.evidence["actions"]:
        scope = r["action"]["target"]["scope"]
        if scope.startswith("r"):
            assert -100 / 6 - 1e-10 <= r["low"] <= r["high"] <= 100 / 6 + 1e-10


def test_inherited_initial_value_and_day_five_persist_in_forecast() -> None:
    data = planted(0.1, 3, 5109)
    sites = [
        s.model_copy(
            update={
                "robots": [
                    r.model_copy(
                        update={
                            "config": ConfigVersion(
                                version=0, values={ConfigKey.software_version: "v0"}
                            )
                        }
                    )
                    for r in s.robots
                ]
            }
        )
        for s in data.fleet.sites
    ]
    data = replace(data, fleet=data.fleet.model_copy(update={"sites": sites}))
    design = Design(data)
    assert design.configuration("r01", 5)[str(ConfigKey.software_version)] == "v1"
    row = data.episodes[0]
    future = design.matrix([row], future=True)
    assert future[0, design.names.index("day:5")] == 1
    constant = [
        i
        for i, n in enumerate(design.names)
        if n.startswith("config:") and np.ptp(design.x[:, i]) == 0
    ]
    assert constant and np.all(design.x[:, constant] == 0)
    assert np.all(future[:, constant] == 0)


def test_beneficial_release_is_not_forced_to_be_harmful_by_reference_coding() -> None:
    data = planted(-0.12, 3, 1702)
    design = Design(data)
    y = np.array([r["outcome"] == "success" for r in data.episodes], dtype=float)
    beta = fit_survival(design.x, y, design.penalty, design.nuisance)
    action = next(a for a in design.candidates() if a.target.scope == "fleet")
    before = design.matrix(data.episodes, future=True)
    after = design.matrix(data.episodes, future=True, action=action)
    assert (
        np.mean(
            predict(after, beta, design.nuisance, "multiplicative")
            - predict(before, beta, design.nuisance, "multiplicative")
        )
        < -0.04
    )


def test_health_noise_deadband_suppresses_healthy_measurement_variation() -> None:
    series = health_series(noisy(774))
    exposures = [h.exposure(r, 0) for h in series for r in h.robots]
    assert sum(v == 0 for v in exposures) >= 10
    assert max(exposures) < 0.7


def test_suspect_short_failures_keep_peer_error_probability() -> None:
    data = noisy()
    for r in data.episodes:
        if r["robot_id"] == "r05" and r["outcome"] == "success":
            r["outcome"] = "failure"
    y, checks = outcome_check(data)
    ids = set(checks[0]["episode_ids"])
    repaired = y[[r["episode_id"] in ids for r in data.episodes]]
    assert np.all((repaired > 0.90) & (repaired < 1.0))


@pytest.mark.parametrize("requests,per_arm", [(0, 60), (9, 60), (8, 0), (8, 61)])
def test_public_budget_rejected_before_io(tmp_path: Path, requests: int, per_arm: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        plan(
            tmp_path / "absent",
            None,  # type: ignore[arg-type]
            tmp_path / "analysis",
            requests=requests,
            episodes_per_arm=per_arm,
        )


def test_hidden_robot_traits_are_rejected_even_in_an_otherwise_valid_export(tmp_path: Path) -> None:
    write_export(tmp_path, noisy())
    path = tmp_path / "log/description.json"
    description = json.loads(path.read_text())
    description["fleet"]["sites"][0]["robots"][0]["traits"] = {"unobserved": 0.1}
    path.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="hidden robot traits"):
        load_export(tmp_path)


def test_second_cut_is_exact_conditional_on_first_three() -> None:
    rng = np.random.default_rng(173)
    for _ in range(12):
        x = np.ones((8, 5))
        beta = rng.normal(size=5)
        repaired = [x + rng.normal(size=x.shape) for _ in range(10)]
        joint = JointPrediction(x, beta, 5, "logistic", repaired, np.full(8, 1 / 8))
        order, _ = joint.order([1.0] * 10, [str(i) for i in range(10)])
        first = tuple(order[:3])
        best = max(
            joint.gain((*first, *c))
            for c in combinations([i for i in range(10) if i not in first], 3)
        )
        assert joint.gain(tuple(order[:6])) == pytest.approx(best)


def test_searching_for_a_short_health_break_pays_a_noise_penalty() -> None:
    # A moderate noisy transition does not support pinpointing its date. A
    # large persistent jump does, under the same pooled measurement noise.
    rng = np.random.default_rng(95)
    noise = rng.normal(0, 0.15, 6)
    fits = []
    for jump in (2.5, 12.0):
        values = 20 + jump * (np.arange(6) >= 3) + noise
        fits.append(
            _trajectory([dict(day=d, value=float(v)) for d, v in enumerate(values)], 1.0, ())
        )
    assert fits[0].kind == "linear"
    assert fits[1].kind == "step" and fits[1].onset == 3


def test_noisy_near_complete_repair_cannot_exceed_affected_population() -> None:
    data = noisy(410)
    rng = np.random.default_rng(411)
    for row in data.episodes:
        success = rng.random() < (0.002 if row["robot_id"] == "r02" and row["day"] >= 2 else 0.998)
        row["outcome"] = "success" if success else "failure"
        row["steps"] = int(rng.integers(90, 230)) if success else 280
    answer = analyse(data, bootstraps=20)
    rec = next(
        r
        for r in answer.evidence["actions"]
        if r["action"]["target"]["scope"] == "r02" and r["action"]["target"]["config_key"]
    )
    assert rec["gain_points"] > 15
    assert rec["gain_points"] + rec["prediction_radius"] > 100 / 6
    assert rec["high"] == pytest.approx(100 / 6, abs=1e-12)
