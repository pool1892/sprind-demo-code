"""Integration of the live harness and the real VLM adapter.

No subscription call: the real VLM adapter receives a fixed, schema-valid model
response. Only its caller and the toy expert are fixtures; engine, queue and
claim-4 scoring are real. `--light` is required.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import config, fault, ident

from fleetlab.agents.run import AgentResult
from fleetlab.contracts.rollout import Job, SupervisorSpec
from fleetlab.engine import manifests, toy
from fleetlab.engine.fleet import run_in_process
from fleetlab.live.vlm.measure import dump
from fleetlab.live.vlm.supervisor import VLMSettings, VLMSupervisor, make_vlm
from fleetlab.policies.teleop.human import HumanSettings

pass_run = pytest.importorskip("fleetlab.live.pass_run")
scoring = pytest.importorskip("fleetlab.live.scoring")
counterfactual = pytest.importorskip("fleetlab.live.counterfactual")


class ToyExpert:
    fumble_applied = False

    def __init__(self, env: Any, **kwargs: Any) -> None:
        self.env = env
        self.policy = toy.ToyTeleoperator()
        self.done = False
        self.status = "running"

    def act(self) -> Any:
        action = self.policy.act_one(self.env._observe(), self.env)
        self.done, self.status = self.policy.done, self.policy.status
        return action


def fixture_factory(name: str, params: Any) -> Any:
    first_step = 200
    if params.get("stagger") and Path(params["payloads"]).exists():
        first_step += 2 * len(json.loads(Path(params["payloads"]).read_text()))
    settings = VLMSettings(first_step=first_step, cooldown_steps=70)
    parts = make_vlm(vars(settings))
    action = params["answer"]

    def call(*args: Any, **kwargs: Any) -> AgentResult:
        payload = json.loads(args[3])
        if params.get("payloads"):
            path = Path(params["payloads"])
            previous = json.loads(path.read_text()) if path.exists() else []
            dump(path, [*previous, payload])
        if action == "failure":
            return AgentResult(ok=False, result=None, usage={}, seconds=0, error="timeout")
        assert action in payload["options"]
        return AgentResult(
            ok=True,
            result={"decision": action, "rationale": "toy caller fixture"},
            usage={},
            seconds=0,
            meta={"phases": {"ready_s": 0, "answer_s": 0}},
        )

    return replace(parts, supervisor=VLMSupervisor(settings, caller=call))


def setup_world(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    world = root / "world"
    jobs = [
        Job(
            identity=ident(
                i,
                robot=i,
                max_steps=280,
                faults=[fault("toy_camera_shift", i, robot=i, pixels=8)] if i <= 6 else [],
            ).model_copy(update={"episode_index": 0})
        )
        for i in range(1, 8)
    ]
    dump(world / "runner.json", {"ready": True})
    dump(
        world / "files/spec.json",
        {
            "world_id": "w03",
            "windows": {"scoring": {"first": 0, "last": 0}},
            "episodes_per_robot_day": 100,
        },
    )
    manifests.write(world / "files/manifest.jsonl", jobs)
    cfg = config(world / "runs/as-run", episodes_per_robot_day=100, workers=1)
    run_in_process(cfg, jobs)
    # The numerical toy world is a locally constructed test fixture, not a
    # development export. Cohort still checks its real outcomes and chain proof.
    monkeypatch.setattr(
        pass_run,
        "require_ready",
        lambda _: {
            "files": world / "files",
            "spec": json.loads((world / "files/spec.json").read_text()),
        },
    )
    monkeypatch.setattr(pass_run, "run", run_in_process)
    monkeypatch.setattr(counterfactual, "run", run_in_process)
    monkeypatch.setattr("fleetlab.policies.teleop.ScriptedExpert", ToyExpert)
    return world, cfg


def test_vlm_rule_queue_and_three_cost_ratios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(os.environ.get("SUPERVISOR_HARNESS_OUT", str(tmp_path)))
    # These are toy fixture measurements, not a claim about the working tree.
    # Pin every pass to the same synthetic revision; ambient tracker writes and
    # uncommitted development edits must not change scorer acceptance.
    fixture_code = {"commit": "supervisor-harness-fixture-v1", "dirty": False}
    monkeypatch.setattr(pass_run, "code_version", lambda: dict(fixture_code))
    world, cfg = setup_world(root, monkeypatch)
    human = HumanSettings(
        operators=2,
        reaction_min_s=0,
        reaction_max_s=0,
        pace=1,
        fumble_probability=0,
        expert_steps=50,
        service_s=5,
    )
    paths = []
    for ratio in (1, 2, 4):
        out = root / f"supervisor-{ratio}"
        record = pass_run.execute(
            world,
            out,
            "supervisor",
            human,
            supervisor=SupervisorSpec(name="vlm", params={"answer": "hand_over"}),
            factory="test_supervisor_harness:fixture_factory",
            config=cfg,
            cost_ratio=ratio,
        )
        assert record["evidence"]["proposals"]["chain_ok"]
        assert all(e["model_s"] == 0 for e in record["episodes"])
        events = record["queue_events"]
        assert sum(e["handover"] for e in record["episodes"]) == 6
        assert max(e["waiting"] for e in events) > 0
        assert max(e["queue_s"] for e in record["episodes"]) > 0
        paths.append(out / "record.json")
    out = root / "timeout"
    pass_run.execute(world, out, "timeout", human, after=200, config=cfg)
    paths.append(out / "record.json")
    result = scoring.score_files(paths, root / "scores", resamples=30)
    assert len(result["rows"]) == 4
    timeout = next(row for row in result["rows"] if row["rule"] == "timeout")
    assert timeout["rescued_with_person"] == 6
    assert timeout["operator_used_s"] == pytest.approx(6.7)
    assert timeout["queue_idle_s"] == 30
    assert timeout["subscription_decision_s"] == 0
    for row in result["rows"]:
        assert row["cohort_episodes"] == 7 and row["would_fail_as_run"] == 6
        assert row["successes_lost"] == row["rescued_without_person"] == 0
        if row["rule"] != "supervisor":
            continue
        # Real adapter overhead is charged even though the fixture's model timer
        # is zero; these must not collapse into one cost in the harness scorer.
        assert row["subscription_decision_s"] > 0 and row["model_response_s"] == 0
        for key in (
            "rescued_with_person",
            "handovers",
            "cohort_successes",
            "operator_used_s",
            "queue_idle_s",
        ):
            assert row[key] == timeout[key]
    dump(root / "receipt.json", {"records": paths, "scores": result})


@pytest.mark.parametrize("answer", ["continue", "failure"])
def test_harness_scores_terminal_and_failed_call_handovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    world, cfg = setup_world(tmp_path, monkeypatch)
    record = pass_run.execute(
        world,
        tmp_path / answer,
        "supervisor",
        HumanSettings(),
        supervisor=SupervisorSpec(name="vlm", params={"answer": answer}),
        factory="test_supervisor_harness:fixture_factory",
        config=cfg,
    )
    assert record["evidence"]["proposals"]["chain_ok"]
    failures = [e for e in record["episodes"] if not e["as_run"]]
    assert len(failures) == 6 and all(e["handover"] for e in failures)
    for episode in failures:
        calls, decisions = episode["calls"], episode["decisions"]
        assert episode["actual_calls"] == len(calls) > 0
        assert [(c["step"], c["decision"]) for c in calls] == [
            (d["step"], d["decision"]) for d in decisions if d["call_status"] != "rule"
        ]
        assert all(0 <= c["risk"] <= 1 for c in calls)
        assert episode["subscription_s"] > 0
        assert record["hold_control"]["record"]
        if answer == "continue":
            assert episode["rule_decisions"] == 1
            assert decisions[-1]["call_status"] == "rule"
            assert decisions[-1]["decision"] == "hand_over"
            assert decisions[-1]["hold_steps"] == 0
            assert episode["model_s"] == 0
        else:
            assert episode["rule_decisions"] == 0
            assert calls[0]["call_status"] == decisions[0]["call_status"] == "failed"
            assert "timeout" in calls[0]["error"]
            assert calls[0]["model_ms"] is decisions[0]["model_ms"] is None
            assert episode["model_s"] is None and decisions[0]["hold_steps"] >= 1


def test_harness_queue_counts_reach_actual_model_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = setup_world(tmp_path, monkeypatch)
    payloads = tmp_path / "payloads.json"
    record = pass_run.execute(
        world,
        tmp_path / "queued",
        "supervisor",
        HumanSettings(operators=2),
        supervisor=SupervisorSpec(
            name="vlm", params={"answer": "hand_over", "payloads": str(payloads), "stagger": True}
        ),
        factory="test_supervisor_harness:fixture_factory",
        config=cfg,
    )
    inputs = json.loads(payloads.read_text())
    calls = [c for e in record["episodes"] for c in e["calls"]]
    assert len(inputs) == len(calls) == 6
    assert [p["operator_queue"] for p in inputs] == [
        {"waiting": c["waiting"], "free_operators": c["free_operators"]} for c in calls
    ]
    assert inputs[0]["operator_queue"] == {"waiting": 0, "free_operators": 2}
    assert any(p["operator_queue"] == {"waiting": 1, "free_operators": 0} for p in inputs)
    assert len({json.dumps(p["operator_queue"], sort_keys=True) for p in inputs}) > 1
