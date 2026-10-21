"""Offline agreement retains failures in its denominator and measures wall time."""

from pathlib import Path
from typing import Any

import pytest

from fleetlab.live.vlm.offline import VARIANTS, summary


def test_offline_agreement_is_not_success_and_invalid_answers_are_not_agreement() -> None:
    rows = [
        dict(
            model=model,
            effort=effort,
            decision_ms=ms,
            original="retry",
            decision=decision,
            valid=valid,
        )
        for model, effort in VARIANTS
        for ms, decision, valid in [
            (20000, "retry", True),
            (10000, "hand_over", True),
            (30000, "retry", False),
        ]
    ]
    for row in summary(rows):
        assert row["calls"] == 3 and row["valid"] == 2 and row["agreement"] == 1
        assert row["decision_s_median"] == 20 and row["decision_s_total"] == 60
        assert row["transitions"] == {"retry->retry": 2, "retry->hand_over": 1}


def test_offline_never_calls_when_the_live_gate_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from fleetlab.live.vlm import offline

    monkeypatch.setattr(offline, "inputs", lambda *args: [])
    calls: list[tuple[Any, ...]] = []

    def refuse(backends: list[str]) -> None:
        assert backends == ["codex"]
        raise RuntimeError("gate closed")

    monkeypatch.setattr(offline, "gate", refuse)
    monkeypatch.setattr(offline, "run_agent", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="gate closed"):
        offline.measure(tmp_path, tmp_path, tmp_path / "out")
    assert not calls


def test_stored_instructions_cannot_silently_change(tmp_path: Path) -> None:
    import json

    from fleetlab.live.vlm.offline import inputs

    root = tmp_path / "call"
    root.mkdir()
    (root / "supervisor.json").write_text("{}")
    (root / "prompt.json").write_text(json.dumps({"system": "different instructions"}))
    with pytest.raises(ValueError, match="reviewed live supervisor"):
        inputs(tmp_path, tmp_path)


def test_offline_call_preserves_prompt_images_and_each_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from PIL import Image
    from test_vlm_supervisor import result

    from fleetlab.live.vlm import offline
    from fleetlab.live.vlm.supervisor import Answer

    image_paths = []
    image_bytes = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"image-{index:02d}.png"
        Image.new("RGB", (3, 2), color).save(path)
        image_paths.append(str(path))
        image_bytes.append(path.read_bytes())
    prompt = {"system": "exact fixture system\n", "prompt": '{"options": {"retry": "allowed"}}\n'}
    row = {
        "source": "stored-call-fixture",
        "prompt": prompt,
        "images": image_paths,
        "supervisor": {"answer": {"decision": "retry"}},
    }
    monkeypatch.setattr(offline, "inputs", lambda *args: [row] * 92)
    gates = []
    verified = []
    calls = []
    monkeypatch.setattr(offline, "gate", lambda backends: gates.append(backends))
    monkeypatch.setattr(offline, "require_verified", lambda *args: verified.append(args))

    class FixtureBatch:
        stopped = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def map(self, call: Any, tasks: Any) -> None:
            for index, model, effort in tasks:
                calls.append((index, model, effort))
                call((index, model, effort))

    def caller(*args: Any, **kwargs: Any) -> Any:
        _, model, effort = calls[-1]
        assert args == ("codex", model, prompt["system"], prompt["prompt"])
        assert kwargs["effort"] == effort
        assert kwargs["schema"] is Answer
        assert kwargs["auth"] == "subscription" and kwargs["service_tier"] == "fast"
        assert [p.name for p in kwargs["images"]] == ["image-00.png", "image-01.png"]
        assert [p.read_bytes() for p in kwargs["images"]] == image_bytes
        return result(model=model)

    monkeypatch.setattr(offline, "Batch", FixtureBatch)
    monkeypatch.setattr(offline, "run_agent", caller)
    out = tmp_path / "out"
    offline.measure(tmp_path, tmp_path, out)
    assert gates == [["codex"]] and len(verified) == 276
    assert len(calls) == 276
    assert {(model, effort) for _, model, effort in calls} == set(offline.VARIANTS)
    report = json.loads((out / "summary.json").read_text())
    assert report["complete"] and not report["stopped"]
    assert all(r["calls"] == r["valid"] == r["agreement"] == 92 for r in report["table"])
