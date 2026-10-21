"""Deferred commands' data guards and receipts, without running a GPU or a model."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import ident
from test_engine_ladder import night

from fleetlab.contracts.rollout import Job
from fleetlab.engine import manifests
from fleetlab.live.vlm.deferred import check_sample, compare_reference, night_settings
from fleetlab.live.vlm.measure import dump


@pytest.mark.parametrize("corruption", ["hash", "counts", "ids"])
def test_deferred_sample_is_the_frozen_paired_cohort(tmp_path: Path, corruption: str) -> None:
    jobs = [Job(identity=ident(n)) for n in range(1, 81)]
    manifests.write(tmp_path / "sample.jsonl", jobs)
    selection: dict[str, Any] = {
        "sample_sha256": hashlib.sha256((tmp_path / "sample.jsonl").read_bytes()).hexdigest(),
        "episodes": {j.identity.episode_id: {"success": i >= 60} for i, j in enumerate(jobs)},
    }
    dump(tmp_path / "selection.json", selection)
    check_sample(tmp_path)
    if corruption == "hash":
        selection["sample_sha256"] = "different"
    elif corruption == "counts":
        selection["episodes"]["e000001"]["success"] = True
    else:
        selection["episodes"]["e999999"] = selection["episodes"].pop("e000001")
    dump(tmp_path / "selection.json", selection)
    with pytest.raises(ValueError, match="unchanged frozen"):
        check_sample(tmp_path)


@pytest.mark.parametrize("failure", [None, "world", "canary", "proof", "date"])
def test_reference_receipt_keeps_every_moved_key_and_refuses_unsupervised_drift(
    tmp_path: Path, failure: str | None
) -> None:
    before, after = tmp_path / "before", tmp_path / "after"
    old = night("unchanged")
    old["canary"] = {"compared": 10, "identical": 10}
    for world in ("deterministic", "randomized"):
        old[world]["runs"]["live"] = dict(old[world]["runs"]["world"])
        old[world]["digests"]["live"] = {"e000001": "old", "e000001.scripted.old": "proof"}
        dump(before / world / "files/spec.json", {"episodes_per_robot_day": 2})
    dump(before / "scores.json", old)
    original_bytes = (before / "scores.json").read_bytes()
    assert night_settings(before) == ("2026-10-08", 2)
    new = copy.deepcopy(old)
    for world in ("deterministic", "randomized"):
        new[world]["digests"]["live"] = {"e000001": "new", "e000001.scripted.new": "proof"}
    if failure == "world":
        new["randomized"]["digests"]["world"]["e000001"] = "drift"
    elif failure == "canary":
        new["canary"]["identical"] = 9
    elif failure == "proof":
        new["deterministic"]["runs"]["live"]["chain_ok"] = False
    elif failure == "date":
        new["date"] = "2026-10-09"
    dump(after / "scores.json", new)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            compare_reference(before, after)
    else:
        receipt = compare_reference(before, after)
        changed = receipt["comparisons"]["deterministic/live"]["changed"]
        assert changed == {
            "e000001": {"before": "old", "after": "new"},
            "e000001.scripted.old": {"before": "proof", "after": None},
            "e000001.scripted.new": {"before": None, "after": "proof"},
        }
        assert receipt["only_live_chains_changed"]
        assert json.loads((after / "reference-transition.json").read_text()) == receipt
        assert "**different**" in (after / "reference-comparison.md").read_text()
    assert (before / "scores.json").read_bytes() == original_bytes


def test_dry_start_validates_real_settings_and_stops_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live.vlm import measure

    sample = tmp_path / "sample"
    sample.mkdir()
    manifests.write(sample / "sample.jsonl", [Job(identity=ident(1))])
    dump(sample / "selection.json", {"episodes_per_robot_day": 2, "episodes": {}})
    settings = tmp_path / "settings.json"
    dump(settings, {"model": "gpt-6.1-sol", "effort": "medium"})
    gates = []
    monkeypatch.setattr(measure, "gate", lambda route: gates.append(route))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("dry start loaded the GPU or served an operator")

    monkeypatch.setattr(measure, "run", forbidden)
    monkeypatch.setattr(measure, "human_pass", forbidden)
    out = tmp_path / "dry"
    measure.measure(
        sample, out, "vlm", "codex", 2, None, "xhigh", settings_file=settings, dry_start=True
    )
    assert gates == [["codex"]]
    receipt = json.loads((out / "dry-start.json").read_text())
    params = receipt["metadata"]["supervisor"]["params"]
    assert params["model"] == "gpt-6.1-sol" and params["effort"] == "medium"
    assert params["stop_file"] == str(out / "route-stopped.json")
    assert not (out / "measurement.json").exists()
    assert (out / "manifest.jsonl").exists()


def test_preflight_refuses_an_incompatible_pinned_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine import miopen
    from fleetlab.live.vlm.deferred import check_software

    manifests.write(tmp_path / "sample.jsonl", [Job(identity=ident(1))])
    monkeypatch.setattr(miopen, "require", lambda *args: "none")
    assert check_software(tmp_path)["miopen_hash"] == "none"
    monkeypatch.setattr(miopen, "require", lambda *args: "changed")
    with pytest.raises(ValueError, match="do not match"):
        check_software(tmp_path)
