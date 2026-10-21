"""CPU checks of the predeclared paired calibration; no simulator or model."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import SOFTWARE

from fleetlab import dose_response as dr
from fleetlab.worlds import supplement as sup
from fleetlab.worlds.curves import OBJECT, SPATIAL
from fleetlab.worlds.generate import policy_spec_nominal


def test_mild_plan_pairs_every_arm_and_keeps_confirmation_independent() -> None:
    screen = sup.mild_plan(policy_spec_nominal(), SOFTWARE)
    confirm = sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15))
    assert len(screen) == 2000 and len(confirm) == 1600
    assert sup.MILD_PAIRS == ((8, 10), (8, 15), (10, 15), (10, 20))
    signatures: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for job in screen + confirm:
        i = job.identity
        signatures[(i.suite, i.episode_index)].append(
            i.model_dump(exclude={"episode_id", "faults"})
        )
        assert i.world_id == "w94" and i.batch_size == 4
    assert all(all(s == values[0] for s in values) for values in signatures.values())
    assert {j.identity.episode_index for j in screen}.isdisjoint(
        j.identity.episode_index for j in confirm
    )
    for jobs in (screen, confirm):
        assert len({j.identity.episode_id for j in jobs}) == len(jobs)
        assert len({j.identity.task_index for j in jobs}) == 10


def joint(pair: tuple[int, int], suite: str, success: float) -> dict[str, Any]:
    return {
        "dose": pair,
        "suite": suite,
        "corrected_joint_success": success,
        "interaction": [-0.3, -0.4, -0.2],
        "effects": {"effect_a": [-0.04, -0.08, 0], "effect_b": [-0.04, -0.08, 0]},
    }


def test_mild_selection_is_predeclared_and_does_not_relax_any_boundary() -> None:
    a = joint((8, 10), SPATIAL, 0.60)
    b = joint((8, 15), SPATIAL, 0.65)
    c = joint((8, 15), OBJECT, 0.67)
    assert sup.mild_select({"joints": [a, b, c]}) == (8, 15)  # both suites before closeness
    assert sup.mild_select({"joints": [a, b]}) == (8, 10)
    assert sup.mild_select({"joints": []}) is None
    assert not sup.mild_eligible({**a, "corrected_joint_success": 0.399}, 0.40)
    assert not sup.mild_eligible({**a, "interaction": [-0.1, -0.2, 0]}, 0.40)
    assert not sup.mild_eligible({**a, "effects": {"effect_a": [-0.101], "effect_b": [0]}}, 0.40)
    assert sup.mild_eligible({**a, "corrected_joint_success": 0.40}, 0.40)


@pytest.mark.parametrize("breakage", ["none", "identity", "missing", "chain"])
def test_mild_analysis_requires_exact_complete_replayed_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breakage: str
) -> None:
    from fleetlab.log import runfiles

    jobs = sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15))
    (tmp_path / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    identities = [j.identity for j in jobs]
    outcomes = {}
    for i in identities:
        success = not (len(i.faults) == 2 and i.episode_index % 10 < 4)
        outcomes[i.episode_id] = dr.Outcome(i.episode_id, success, success, success, 100)
    if breakage == "identity":
        identities[0] = identities[0].model_copy(update={"scene_seed": 0})
    if breakage == "missing":
        outcomes.pop(next(iter(outcomes)))
    run = dr.RunOutcomes("test-only", breakage != "chain", outcomes)
    monkeypatch.setattr(dr, "read_run", lambda path: run)
    monkeypatch.setattr(runfiles, "read_identities", lambda path: identities)
    if breakage != "none":
        with pytest.raises(ValueError):
            sup.mild_statistics(tmp_path, (8, 15))
        return
    result = sup.mild_statistics(tmp_path, (8, 15))
    assert len(result["joints"]) == 2 and len(result["points"]) == 4
    for measured in result["joints"]:
        assert measured["n"] == 200
        assert measured["single_successes"] == [200, 200]
        assert measured["successes"] == 120
        assert measured["interaction"][0] == -0.4
        assert measured["interaction"][2] < 0
    assert not result["selection_data_pooled"]


def test_mild_driver_refuses_extra_workers_or_time(tmp_path: Path) -> None:
    for workers, seconds in [(7, 100), (0, 100), (6, 10801), (6, 0)]:
        with pytest.raises(ValueError):
            sup.mild_driver(tmp_path, workers, seconds)


def test_blur_fallback_preserves_the_independent_healthy_identities(tmp_path: Path) -> None:
    import json

    jobs = sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15))
    (tmp_path / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    document = {"stage": "confirmation", "chain_ok": True, "joints": [joint((8, 15), SPATIAL, 0.6)]}
    (tmp_path / "analysis.json").write_text(json.dumps(document))
    blur = sup.blur_plan(tmp_path)
    controls = [j for j in jobs if not j.identity.faults]
    assert len(blur) == 400
    for measured, control in zip(blur, controls, strict=True):
        assert measured.identity.model_dump(exclude={"episode_id", "faults"}) == (
            control.identity.model_dump(exclude={"episode_id", "faults"})
        )
        assert len(measured.identity.faults) == 1
        assert measured.identity.faults[0].kind == "camera_degradation"
        assert measured.identity.faults[0].level == 3
    for field, value in (("stage", "screen"), ("chain_ok", False), ("joints", [])):
        (tmp_path / "analysis.json").write_text(json.dumps({**document, field: value}))
        with pytest.raises(ValueError, match="independently confirmed"):
            sup.blur_plan(tmp_path)


@pytest.mark.parametrize("breakage", ["none", "manifest", "identity", "missing", "healthy-chain"])
def test_blur_analysis_uses_exact_replayed_healthy_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breakage: str
) -> None:
    import json

    from fleetlab.log import runfiles

    confirmation, root = tmp_path / "confirmation", tmp_path / "blur"
    confirmation.mkdir()
    root.mkdir()
    controls = sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15))
    (confirmation / "manifest.jsonl").write_text(
        "".join(j.model_dump_json() + "\n" for j in controls)
    )
    (confirmation / "analysis.json").write_text(
        json.dumps(
            {"stage": "confirmation", "chain_ok": True, "joints": [joint((8, 15), SPATIAL, 0.6)]}
        )
    )
    planned = sup.blur_plan(confirmation)
    (root / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in planned))
    outcomes = {}
    for job in planned:
        i = job.identity
        ok = i.episode_index % 10 < 8
        outcomes[i.episode_id] = dr.Outcome(i.episode_id, ok, ok, ok, 100)
    healthy = {
        j.identity.episode_id: dr.Outcome(j.identity.episode_id, True, True, True, 100)
        for j in controls
        if not j.identity.faults
    }
    if breakage == "missing":
        outcomes.pop(next(iter(outcomes)))
    identities = [j.identity for j in planned]
    if breakage == "identity":
        identities[0] = identities[0].model_copy(update={"noise_seed": 0})
    if breakage == "manifest":
        (root / "manifest.jsonl").write_text("")
    monkeypatch.setattr(
        dr,
        "read_run",
        lambda path: (
            dr.RunOutcomes("blur", True, outcomes)
            if path == root / "run"
            else dr.RunOutcomes("healthy", breakage != "healthy-chain", healthy)
        ),
    )
    monkeypatch.setattr(
        runfiles,
        "read_identities",
        lambda path: identities if path == root / "run" else [j.identity for j in controls],
    )
    if breakage != "none":
        with pytest.raises(ValueError):
            sup.blur_statistics(root, confirmation)
        return
    result = sup.blur_statistics(root, confirmation)
    assert result["effective_on_both_suites"]
    assert len(result["points"]) == 2
    assert all(
        p["n"] == 200
        and p["successes"] == 160
        and p["healthy_successes"] == 200
        and p["difference"][2] < 0
        for p in result["points"]
    )


def test_resource_confirmation_is_independent_and_never_truncates_a_started_stage(
    tmp_path: Path,
) -> None:
    import json

    screen = sup.mild_plan(policy_spec_nominal(), SOFTWARE)
    reduced = sup.mild_plan(
        policy_spec_nominal(), SOFTWARE, pair=(8, 15), confirmation_episodes=100
    )
    assert len(reduced) == 800
    assert {j.identity.episode_index for j in reduced} == set(range(100, 200))
    assert {j.identity.episode_index for j in reduced}.isdisjoint(
        j.identity.episode_index for j in screen
    )
    with pytest.raises(ValueError, match="100 or 200"):
        sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15), confirmation_episodes=150)
    receipt = {
        "state": "insufficient remaining budget for confirmation",
        "stages": [{"stage": "screen"}],
    }
    (tmp_path / "execution-01.json").write_text(json.dumps(receipt))
    (tmp_path / "selection.json").write_text(json.dumps({"pair": [8, 15]}))
    sup.validate_resource_confirmation(tmp_path, (8, 15))
    with pytest.raises(ValueError, match="same selection"):
        sup.validate_resource_confirmation(tmp_path, (10, 15))
    (tmp_path / "confirmation").mkdir()
    with pytest.raises(ValueError, match="unstarted"):
        sup.validate_resource_confirmation(tmp_path, (8, 15))


def test_reduced_confirmation_and_blur_use_their_actual_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from fleetlab.log import runfiles

    jobs = sup.mild_plan(policy_spec_nominal(), SOFTWARE, pair=(8, 15), confirmation_episodes=100)
    (tmp_path / "manifest.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    outcomes = {}
    for j in jobs:
        i = j.identity
        ok = not (len(i.faults) == 2 and i.episode_index % 10 < 4)
        outcomes[i.episode_id] = dr.Outcome(i.episode_id, ok, ok, ok, 100)
    monkeypatch.setattr(dr, "read_run", lambda path: dr.RunOutcomes("test", True, outcomes))
    monkeypatch.setattr(runfiles, "read_identities", lambda path: [j.identity for j in jobs])
    result = sup.mild_statistics(tmp_path, (8, 15), confirmation_episodes=100)
    assert result["n_per_arm"] == 100 and result["stage"] == "confirmation"
    assert all(p["n"] == 100 for p in result["points"] + result["joints"])
    (tmp_path / "analysis.json").write_text(json.dumps(result))
    assert len(sup.blur_plan(tmp_path)) == 200
    with pytest.raises(ValueError, match="incomplete or unequal"):
        sup.mild_statistics(tmp_path, (8, 15))  # cannot claim 200 attempts from these 100
