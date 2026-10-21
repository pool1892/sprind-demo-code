from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from engine_helpers import config, ident
from test_teleop_measure import table

from fleetlab.contracts.live import (
    Decision,
    DecisionRecord,
    OperatorQueue,
    SupervisorContext,
    SupervisorDecision,
)
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, SnapshotRef, SupervisorSpec
from fleetlab.engine import manifests
from fleetlab.engine.episode import SupervisorParts
from fleetlab.engine.fleet import run, run_in_process
from fleetlab.engine.supervisors import AfterSteps
from fleetlab.engine.toy import ToyCheck
from fleetlab.live import pass_cli
from fleetlab.live.pass_run import cohort, economic_proposal, execute, frozen, wrap
from fleetlab.live.rules import CostRule, QueuePolicy, gate_requests, make
from fleetlab.live.scoring import compare_records, frontier, score_files, summarize
from fleetlab.policies.teleop.human import HumanSettings
from fleetlab.policies.teleop.measure import dump


@pytest.fixture(autouse=True)
def fixture_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructed fixtures exercise cohort validation without impersonating checked exports.
    from fleetlab.live import pass_run

    monkeypatch.setattr(
        pass_run,
        "require_ready",
        lambda world: {
            "files": world / "files",
            "spec": json.loads((world / "files/spec.json").read_text()),
        },
    )


def source(world: Path, jobs: list[Job], successes: int) -> None:
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
    dump(world / "runs/as-run/run.json", {"chain_ok": True})
    table(
        world / "runs/as-run/truth/job_outcomes.parquet",
        [
            {"episode_id": j.key, "job_key": j.key, "mode": "run", "success": i < successes}
            for i, j in enumerate(jobs)
        ],
    )


def test_cohort_is_all_failures_and_50_successes_with_window_weights(tmp_path: Path) -> None:
    jobs = [Job(identity=ident(i + 1)) for i in range(77)]
    source(tmp_path, jobs, 70)
    chosen, meta = cohort(tmp_path)
    assert len(chosen) == 57
    assert {j.key for j in jobs[70:]} <= {j.key for j in chosen}
    assert meta["window_successes"] == 70 and meta["window_failures"] == 7
    assert sum(r["weight"] for r in meta["episodes"]) == pytest.approx(77)
    assert cohort(tmp_path)[1] == meta
    dump(tmp_path / "runs/as-run/run.json", {"chain_ok": False})
    with pytest.raises(ValueError, match="chain sample"):
        cohort(tmp_path)


def test_cohort_refuses_missing_and_duplicate_truth(tmp_path: Path) -> None:
    jobs = [Job(identity=ident(1)), Job(identity=ident(2))]
    source(tmp_path, jobs, 1)
    table(
        tmp_path / "runs/as-run/truth/job_outcomes.parquet",
        [
            {"episode_id": jobs[0].key, "mode": "run", "success": True},
        ],
    )
    with pytest.raises(ValueError, match="no denominator"):
        cohort(tmp_path)
    source(tmp_path, jobs, 1)
    dump(tmp_path / "runs/duplicate/run.json", {"chain_ok": True})
    table(
        tmp_path / "runs/duplicate/truth/job_outcomes.parquet",
        [
            {"episode_id": jobs[0].key, "mode": "run", "success": True},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        cohort(tmp_path)


def test_matched_random_does_not_require_custom_supervisor_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = tmp_path / "world"
    source(world, [Job(identity=ident(1))], 0)
    spec = tmp_path / "supervisor.json"
    spec.write_text(json.dumps({"name": "custom-only", "params": {}}))
    calls: list[tuple[str, Path, dict[str, Any]]] = []

    def custom_boundary(
        world: Path, out: Path, rule: str, settings: HumanSettings, **options: Any
    ) -> dict[str, Any]:
        if rule == "random" and options["factory"] == "custom_only:make":
            raise ValueError("custom factory only supports its own supervisor spec")
        calls.append((rule, out, options))
        return {}

    monkeypatch.setattr(pass_cli, "execute", custom_boundary)
    monkeypatch.setattr(pass_cli, "score_files", lambda *args: {})
    monkeypatch.setattr(pass_cli, "draw", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "live-pass",
            "sweep",
            "--world",
            str(world),
            "--out",
            str(tmp_path / "sweep"),
            "--rule",
            "supervisor",
            "--supervisor",
            str(spec),
            "--factory",
            "custom_only:make",
            "--ratios",
            "2",
            "--matched-random",
        ],
    )
    pass_cli.main()
    assert [call[0] for call in calls] == ["supervisor", "random"]
    assert calls[0][2]["factory"] == "custom_only:make"
    assert calls[1][2]["factory"] == "fleetlab.engine.supervisors:make"
    assert calls[1][2]["supervisor"] is None
    assert calls[1][2]["reference"] == calls[0][1]


def request(n: int) -> Job:
    identity = ident(n, robot=n).model_copy(update={"episode_index": 0})
    return Job(
        identity=identity,
        mode=JobMode.continue_,
        from_snapshot=SnapshotRef(
            episode_id=identity.episode_id,
            step=25,
            path=f"{identity.episode_id}.25.pkl",
        ),
    )


def test_cost_gate_prices_real_queue_and_only_reserves_accepted_requests() -> None:
    jobs = [request(i) for i in range(1, 4)]
    settings = HumanSettings(operators=1)
    risks = {j.identity.episode_id: 0.8 for j in jobs}
    admitted, denied, events = gate_requests(
        jobs[::-1], risks, settings, 14, QueuePolicy(max_wait_s=0)
    )
    assert [j.identity.episode_id for j in admitted] == [jobs[0].identity.episode_id]
    assert len(denied) == 2
    assert events[1]["free_operators"] == 0 and events[1]["proposed_queue_s"] == 90
    assert events[2]["waiting"] == 0  # rejected request never reserves or joins the queue
    queued, _, queued_events = gate_requests(jobs, risks, settings, 14, QueuePolicy())
    assert len(queued) == 3 and queued_events[2]["waiting"] == 1
    assert queued[2].teleoperator_params is not None
    assert queued[2].teleoperator_params["queue_steps"] == 3600
    assert gate_requests(jobs, risks, settings, 14, QueuePolicy())[2] == queued_events


def test_cost_gate_requires_autonomous_exhaustion_and_strict_inequality() -> None:
    context = SupervisorContext(
        {}, [], 0.5, "", [Decision.continue_, Decision.hand_over], OperatorQueue(0, 2)
    )
    assert CostRule(2).decide(context).decision == Decision.continue_
    assert CostRule(4).decide(context).decision == Decision.hand_over
    with pytest.raises(ValueError, match="autonomous"):
        CostRule(4).decide(SupervisorContext({}, [], 0.8, "", list(Decision)))


def test_economic_proposal_has_no_classifier_cutoff_and_prices_avoidable_fraction(
    tmp_path: Path,
) -> None:
    from test_live_model import constant_model

    assert economic_proposal(CostRule(1)) is None
    assert economic_proposal(CostRule(4, avoidable_fraction=0)) is None
    assert economic_proposal(CostRule(4, avoidable_fraction=0.25)) is None
    partial = economic_proposal(CostRule(4, avoidable_fraction=0.5))
    assert partial is not None
    partial_threshold = partial.params["threshold"]
    assert isinstance(partial_threshold, float) and partial_threshold > 0.5
    spec = economic_proposal(CostRule(4))
    assert spec is not None
    threshold = spec.params["threshold"]
    assert isinstance(threshold, float) and 0.25 < threshold < 0.3
    spec = spec.model_copy(
        update={"params": {**spec.params, "model": constant_model(0.3).to_dict()}}
    )
    job = wrap(Job(identity=ident(1)), spec, tmp_path, "fleetlab.engine.supervisors:make")
    result = run_in_process(
        config(
            tmp_path / "fleet", supervisors="fleetlab.live.rules:make", defer_continuations=True
        ),
        [job],
    )
    assert result.summaries[0]["ended_by"] == "hand_over"
    context = SupervisorContext({}, [], 0.3, "", [Decision.continue_, Decision.hand_over])
    assert CostRule(4).decide(context).decision is Decision.hand_over


def row(n: int, **changes: Any) -> dict[str, Any]:
    return {
        "episode_id": f"e{n:06}",
        "robot": f"r{n:02}",
        "as_run": False,
        "success": True,
        "weight": 1,
        "snapshot_without": False,
        "operator_s": 10,
        "reserved_s": 90,
        "queue_s": 0,
        "subscription_s": 1,
        "model_s": 0.2,
        "decision_hold_s": 1,
        "request": True,
        "handover": True,
        "decisions": [],
        "status": "placed",
        "task": "toy/0",
        "human": {"active_steps": 10, "fumble_draw": False, "fumble_applied": False},
        **changes,
    }


def record(rows: list[dict[str, Any]], rule: str = "timeout") -> dict[str, Any]:
    return {
        "population": {"version": 1},
        "settings": HumanSettings().model_dump(mode="json"),
        "rule": rule,
        "cost_ratio": 2,
        "after": 174,
        "config": {"code": {"commit": "synthetic-test-fixture", "dirty": False}},
        "episodes": rows,
    }


def test_scoring_retains_both_denominators_and_expands_success_sample(tmp_path: Path) -> None:
    rec = record(
        [
            row(1),
            row(2, operator_s=0),
            row(3, as_run=True, success=False, snapshot_without=True, weight=4),
        ]
    )
    score = summarize(rec)
    assert score["rescued_without_person"] == score["rescued_with_person"] == 1
    assert score["successes_lost"] == score["successes_lost_snapshot"] == 1
    assert score["estimated_window_episodes"] == 6
    assert score["estimated_window_success"] == pytest.approx(2 / 6)
    assert score["operator_used_s"] == 20 and score["estimated_window_operator_s"] == 50
    dump(tmp_path / "record.json", rec)
    result = score_files([tmp_path / "record.json"], tmp_path / "score")
    assert all(result["rows"][0][key] == value for key, value in score.items())
    assert len(result["rows"][0]["estimated_window_net_interval95"]) == 2
    assert "Counterfactual" in (tmp_path / "score/table.md").read_text()
    assessed = summarize(
        record(
            [
                row(
                    n,
                    as_run=True,
                    request=False,
                    handover=False,
                    operator_s=0,
                    decisions=[{"decision": action}],
                )
                for n, action in ((1, "continue"), (2, "retry"))
            ],
            "supervisor",
        )
    )
    assert assessed["healthy_assessments"] == 2
    assert assessed["false_alarms"] == 1 and assessed["false_alarm_share"] == 0.5
    assert assessed["false_requests"] == assessed["false_handovers"] == 0


def test_equal_time_frontier_reselects_best_rule_and_never_extrapolates() -> None:
    points = [
        {"estimated_window_operator_s": 100, "estimated_window_net_gain": 10, "label": "timeout"},
        {"estimated_window_operator_s": 200, "estimated_window_net_gain": 8, "label": "random"},
    ]
    best = frontier(points, 50)
    assert best is not None and best["net"] == 5
    assert frontier(points, 201) is None
    base = record([row(1), row(2, success=False)])
    candidate = record([row(1), row(2)], "supervisor")
    comparison = compare_records(candidate, [base], resamples=30)
    assert comparison["relative_gain"] == 1
    assert comparison["interval"]["valid_relative"] > 0
    assert comparison == compare_records(candidate, [base], resamples=30)
    candidate["settings"]["operators"] = 6
    with pytest.raises(ValueError, match="identical human"):
        compare_records(candidate, [base])
    candidate["settings"]["operators"] = 2
    candidate["config"]["code"]["commit"] = "different-test-fixture"
    with pytest.raises(ValueError, match="code revision"):
        compare_records(candidate, [base])
    candidate["config"]["code"]["dirty"] = True
    with pytest.raises(ValueError, match="clean committed"):
        compare_records(candidate, [base])


def test_fixed_adapter_through_one_interface_and_pinned_resume(tmp_path: Path) -> None:
    from test_live_model import constant_model

    spec = SupervisorSpec(
        name="risk",
        params={
            "model": constant_model().to_dict(),
            "threshold": 0.5,
            "decision": "stop",
            "decision_ms": 200,
        },
    )
    job = wrap(Job(identity=ident(1)), spec, tmp_path, "fleetlab.engine.supervisors:make")
    assert job.supervisor is not None
    parts = make(job.supervisor.name, job.supervisor.params)
    assert parts.threshold == 0.5
    run_in_process(config(tmp_path / "fleet", supervisors="fleetlab.live.rules:make"), [job])
    events = json.loads((tmp_path / "decisions/e000001.json").read_text())
    assert events[0]["decision"] == "stop"
    assert events[0]["risk"] == pytest.approx(0.7)
    frozen(tmp_path / "pin.json", {"ratio": 2})
    with pytest.raises(ValueError, match="changed"):
        frozen(tmp_path / "pin.json", {"ratio": 4})


class TimedSupervisor:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(Decision.stop, "fixture measured routes", 250, 75)


def latency_factory(name: str, params: Any) -> SupervisorParts:
    return SupervisorParts(AfterSteps(4), TimedSupervisor())


class AssessmentRisk:
    def update(self, telemetry: Any) -> float:
        return {2: 0.7, 5: 0.9}.get(telemetry.step + 1, 0.0)


class AssessmentSupervisor:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        if context.telemetry_window[-1].step + 1 == 2:
            return SupervisorDecision(Decision.continue_, "healthy assessment", 50, 0)
        return SupervisorDecision(Decision.stop, "fixture intervention", 150, 0)


def assessment_factory(name: str, params: Any) -> SupervisorParts:
    return SupervisorParts(AssessmentRisk(), AssessmentSupervisor())


def test_latency_routes_charge_physical_idle_and_keep_both_original_timings(tmp_path: Path) -> None:
    for route, hold in (("subscription", 5), ("model", 2)):
        root = tmp_path / route
        spec = SupervisorSpec(name="fixture")
        job = wrap(Job(identity=ident(1)), spec, root, "test_live_pass:latency_factory", route)
        result = run_in_process(
            config(root / "fleet", supervisors="fleetlab.live.rules:make"), [job]
        )
        detail = result.summaries[0]
        events = json.loads((root / "decisions/e000001.json").read_text())
        assert events[0]["decision_ms"] == 250 and events[0]["model_ms"] == 75
        assert events[0]["step"] == 5
        assert detail["steps"] == 5 + hold
        assert detail["success"] is False


def test_autonomous_stop_has_exact_snapshot_counterfactual_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live import counterfactual, pass_run

    world = tmp_path / "world"
    jobs = [Job(identity=ident(i)) for i in range(1, 4)]
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
    cfg = config(world / "runs/as-run", episodes_per_robot_day=100)
    run_in_process(cfg, jobs)
    monkeypatch.setattr(pass_run, "run", run_in_process)
    monkeypatch.setattr(counterfactual, "run", run_in_process)
    spec = SupervisorSpec(name="fixture")
    factory = "test_live_pass:assessment_factory"
    result = execute(
        world,
        tmp_path / "pass",
        "supervisor",
        HumanSettings(),
        supervisor=spec,
        config=cfg,
        factory=factory,
    )
    from dataclasses import replace

    control = [
        Job(
            identity=j.identity,
            mode=JobMode.scripted,
            decision_trace=[
                DecisionRecord(
                    step=2, decision=Decision.continue_, hold_steps=1, decision_ms=50, model_ms=0
                ),
                DecisionRecord(
                    step=5, decision=Decision.continue_, hold_steps=3, decision_ms=150, model_ms=0
                ),
            ],
        )
        for j in jobs
    ]
    expected = run_in_process(
        replace(cfg, out=str(tmp_path / "expected"), chain_sample=False), control
    )
    expected_success = {r["episode_id"]: r["success"] for r in expected.summaries}
    assert all(
        e["snapshot_without"] == expected_success[e["episode_id"]] for e in result["episodes"]
    )
    assert all(e["interventions"][0]["prefix_chain_identical"] for e in result["episodes"])
    assert all(len(e["interventions"]) == 1 for e in result["episodes"])
    assert all(e["interventions"][0]["decision_index"] == 1 for e in result["episodes"])
    assert all(e["snapshot_step"] == 8 for e in result["episodes"])
    assert (
        execute(
            world,
            tmp_path / "pass",
            "supervisor",
            HumanSettings(),
            supervisor=spec,
            config=cfg,
            factory=factory,
        )
        == result
    )
    broken = RunLayout(tmp_path / "pass/autonomous-counters/1")
    from fleetlab.engine.chain import Chain

    chain = Chain.load(broken.chain(jobs[0]))
    chain.state[0] = "broken"
    chain.save(broken.chain(jobs[0]))
    with pytest.raises(ValueError, match="prefix differs"):
        execute(
            world,
            tmp_path / "pass",
            "supervisor",
            HumanSettings(),
            supervisor=spec,
            config=cfg,
            factory=factory,
        )


@pytest.mark.sim
def test_real_pass_queue_holds_snapshots_and_resume(tmp_path: Path) -> None:
    from fleetlab.engine import software
    from fleetlab.engine.scripted import SWAY_SPEC

    world = tmp_path / "world"
    jobs = [
        Job(
            identity=manifests.identity(
                world_seed=604,
                episode_number=i,
                robot=i,
                site=1,
                day=0,
                episode_index=0,
                suite="libero_spatial",
                task_index=0,
                start_state=2,
                policy=SWAY_SPEC,
                batch_size=1,
                software=software.current("none"),
                image_size=64,
                max_steps=60,
            )
        )
        for i in range(1, 4)
    ]
    dump(world / "runner.json", {"ready": True})
    dump(
        world / "files/spec.json",
        {
            "world_id": "w604",
            "windows": {"scoring": {"first": 0, "last": 0}},
            "episodes_per_robot_day": 100,
        },
    )
    manifests.write(world / "files/manifest.jsonl", jobs)
    cfg = config(
        world / "runs/as-run",
        batch_size=1,
        workers=2,
        env_factory="fleetlab.sim.libero_sim:LiberoSimEnv",
        policy_factory="fleetlab.engine.scripted:SwayPolicy",
        check_factory="fleetlab.sim.libero_success:FleetCheck",
        fault_registry="fleetlab.faults:REGISTRY",
        episodes_per_robot_day=100,
    )
    run(cfg, jobs)
    settings = HumanSettings(operators=2, reaction_min_s=0, reaction_max_s=0, fumble_probability=0)
    result = execute(world, tmp_path / "pass", "timeout", settings, after=20, config=cfg)
    assert len(result["episodes"]) == 3
    assert result["evidence"]["proposals"]["chain_ok"] is True
    assert max(e["human"]["queue_steps"] for e in result["episodes"]) == 1800
    assert all(e["snapshot_without"] is not None for e in result["episodes"])
    assert execute(world, tmp_path / "pass", "timeout", settings, after=20, config=cfg) == result

    # The same real LIBERO states also exercise an autonomous decision counter.
    stop = SupervisorSpec(
        name="timeout", params={"after": 20, "decision": "stop", "decision_ms": 150}
    )
    stopped = execute(
        world, tmp_path / "stop-pass", "supervisor", settings, supervisor=stop, config=cfg
    )
    assert all(e["interventions"][0]["prefix_chain_identical"] for e in stopped["episodes"])
    assert all(e["snapshot_step"] == 24 for e in stopped["episodes"])
    assert all(e["snapshot_without"] is False for e in stopped["episodes"])


class ToyExpert:
    fumble_applied = False

    def __init__(self, env: Any, **kwargs: Any) -> None:
        from fleetlab.engine.toy import ToyTeleoperator

        self.env, self.policy = env, ToyTeleoperator()
        self.done, self.status = False, "running"

    def act(self) -> Any:
        action = self.policy.act_one(self.env._observe(), self.env)
        self.done, self.status = self.policy.done, self.policy.status
        return action


def live_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    from engine_helpers import fault

    from fleetlab.live import counterfactual, pass_run

    jobs = [
        Job(
            identity=ident(
                i,
                robot=i,
                max_steps=70,
                faults=[fault("toy_camera_shift", i, robot=i, pixels=8)] if i <= 4 else [],
            ).model_copy(update={"episode_index": 0})
        )
        for i in range(1, 7)
    ]
    world = tmp_path / "world"
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
    monkeypatch.setattr(pass_run, "run", run_in_process)
    monkeypatch.setattr(counterfactual, "run", run_in_process)
    monkeypatch.setattr(pass_run, "code_version", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr("fleetlab.policies.teleop.ScriptedExpert", ToyExpert)
    return world, cfg


def human_settings(operators: int = 2) -> HumanSettings:
    return HumanSettings(
        operators=operators,
        reaction_min_s=0,
        reaction_max_s=0,
        pace=1,
        expert_steps=50,
        service_s=5,
        fumble_probability=0,
    )


def test_random_real_pass_selection_counts_timing_reference_and_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.contracts.identity import noise_for

    world, cfg = live_world(tmp_path, monkeypatch)
    human = human_settings()
    reference = tmp_path / "timeout"
    baseline = execute(world, reference, "timeout", human, after=63, config=cfg)
    assert sum(e["handover"] for e in baseline["episodes"]) == 4
    # Refused requests must not become the count that random matches.
    baseline["episodes"][-1]["request"] = True
    dump(reference / "record.json", baseline)
    result = execute(world, tmp_path / "random", "random", human, reference=reference, config=cfg)
    selected, _ = cohort(world)
    expected = {
        j.key
        for j in sorted(
            selected, key=lambda j: float(noise_for(j.identity, 0, (1,), sample=651)[0])
        )[:4]
    }
    assert {e["episode_id"] for e in result["episodes"] if e["handover"]} == expected
    assert all(e["decisions"][0]["step"] == 1 for e in result["episodes"] if e["handover"])
    assert any(not e["success"] for e in result["episodes"])
    assert result == execute(
        world, tmp_path / "random", "random", human, reference=reference, config=cfg
    )
    for e in result["episodes"]:
        assert e["operator_s"] == e["human"].get("operator_used_s", 0)
        assert e["reserved_s"] == e["human"].get("operator_reserved_s", 0)
        assert (
            e["queue_s"]
            == min(e["human"].get("idle_steps", 0), e["human"].get("queue_steps", 0)) / 20
        )
    assert any(e["queue_s"] > 0 for e in result["episodes"])
    assert summarize(result)["queue_idle_s"] == sum(e["queue_s"] for e in result["episodes"])
    assert all(e["operator_s"] < e["reserved_s"] for e in result["episodes"] if e["handover"])
    for field, value in (("population", {}), ("settings", {"operators": 8})):
        wrong = {**baseline, field: value}
        dump(reference / "record.json", wrong)
        with pytest.raises(ValueError, match="same cohort"):
            execute(
                world, tmp_path / f"wrong-{field}", "random", human, reference=reference, config=cfg
            )
    dump(reference / "record.json", baseline)
    original = __import__("fleetlab.live.pass_run", fromlist=["gate_requests"]).gate_requests

    def incomplete(*args: Any, **kwargs: Any) -> Any:
        admitted, denied, events = original(*args, **kwargs)
        return admitted[:-1], denied, events

    monkeypatch.setattr("fleetlab.live.pass_run.gate_requests", incomplete)
    with pytest.raises(ValueError, match="reference hand-over count"):
        execute(world, tmp_path / "incomplete", "random", human, reference=reference, config=cfg)


class QueueSupervisor:
    def __init__(self, params: Any) -> None:
        self.params = params

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        path = Path(self.params["observations"])
        observations = json.loads(path.read_text()) if path.exists() else []
        observations.append(
            [
                context.operator_queue.waiting,
                context.operator_queue.free_operators,
                context.task_text,
                context.telemetry_window[-1].step + 1,
            ]
        )
        dump(path, observations)
        return SupervisorDecision(Decision.hand_over, "fixture request", 0, 0)


def queue_factory(name: str, params: Any) -> SupervisorParts:
    after = params.get("after", 5)
    path = Path(params["observations"])
    if params.get("stagger") and path.exists():
        after += 2 * len(json.loads(path.read_text()))
    return SupervisorParts(AfterSteps(after), QueueSupervisor(params))


def test_requests_reach_queue_at_all_ratios_and_supervisor_sees_known_occupancy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    for ratio in (1, 2, 4):
        observations = tmp_path / f"observations-{ratio}.json"
        result = execute(
            world,
            tmp_path / f"ratio-{ratio}",
            "supervisor",
            human_settings(1),
            cost_ratio=ratio,
            config=cfg,
            factory="test_live_pass:queue_factory",
            supervisor=SupervisorSpec(
                name="fixture", params={"observations": str(observations), "stagger": True}
            ),
        )
        assert all(e["handover"] for e in result["episodes"])
        contexts = json.loads(observations.read_text())
        assert contexts[0][:2] == [0, 1]
        assert any(c[:2] == [1, 0] for c in contexts)
        assert all(f"ratio: {float(ratio)}" in c[2] or f"ratio: {ratio}" in c[2] for c in contexts)
        assert result["hold_control"]["record"]
        control = json.loads(Path(result["hold_control"]["record"]).read_text())
        assert all(
            not e["request"] and not e["handover"] and e["actual_calls"] == 0
            for e in control["episodes"]
        )
        assert all(d["decision"] == "continue" for e in control["episodes"] for d in e["decisions"])
        assert (
            summarize(result)["hold_control_net"] == summarize(control)["estimated_window_net_gain"]
        )


def test_queue_refusal_keeps_the_rule_until_a_later_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    observations = tmp_path / "observations.json"
    result = execute(
        world,
        tmp_path / "refused",
        "supervisor",
        human_settings(1),
        config=cfg,
        max_wait_s=0,
        factory="test_live_pass:queue_factory",
        supervisor=SupervisorSpec(name="fixture", params={"observations": str(observations)}),
    )
    refused = [e for e in result["episodes"] if e["refused_requests"]]
    assert refused and all(len(e["calls"]) > 1 for e in refused)
    assert any(e["success"] for e in refused)  # physical policy result, not stopped proposal
    assert any(e["success"] and e["operator_s"] == 0 and not e["handover"] for e in refused)
    assert all(
        any(c.get("refused") and c["decision"] == "continue" for c in e["calls"]) for e in refused
    )


class FallbackSupervisor:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        from fleetlab.contracts.live import SupervisorCallFailure

        if self.answer in {"failure", "skipped"}:
            raise SupervisorCallFailure(
                "fixture timeout", 75, None, called=self.answer != "skipped"
            )
        return SupervisorDecision(Decision(self.answer), "fixture answer", 75, 20)


def fallback_factory(name: str, params: Any) -> SupervisorParts:
    return SupervisorParts(
        AfterSteps(5),
        FallbackSupervisor(params["answer"]),
        should_call=lambda c: not c.previous_decisions,
        failure_fallback=Decision.hand_over,
    )


def repeating_failure_factory(name: str, params: Any) -> SupervisorParts:
    return SupervisorParts(
        AfterSteps(5), FallbackSupervisor("failure"), failure_fallback=Decision.hand_over
    )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="open issue: read-only engine hardcodes failed-call fallback to hand_over",
)
def test_failed_call_queue_refusal_retains_supervision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    try:
        result = execute(
            world,
            tmp_path / "failed-refusal",
            "supervisor",
            human_settings(1),
            config=cfg,
            max_wait_s=0,
            factory="test_live_pass:repeating_failure_factory",
            supervisor=SupervisorSpec(name="fixture", params={"answer": "failure"}),
        )
    except ValueError as error:
        assert str(error) == "queue refusal escaped the retained supervisor"
        raise
    refused = [e for e in result["episodes"] if e["refused_requests"]]
    assert refused
    # A later request may be accepted; only the refused call must continue.
    assert all(e["operator_s"] == 0 for e in refused if not e["handover"])
    assert any(len(e["calls"]) > 1 for e in refused)
    for episode in refused:
        for call in (c for c in episode["calls"] if c.get("refused")):
            decision = next(d for d in episode["decisions"] if d["step"] == call["step"])
            assert call["requested"] and call["decision"] == decision["decision"] == "continue"
            assert call["call_status"] == decision["call_status"] == "failed"
            assert call["model_ms"] is decision["model_ms"] is None
            assert decision["decision_ms"] == 75 and decision["hold_steps"] == 2
    requests = manifests.read(tmp_path / "failed-refusal/requests.jsonl")
    assert len(requests) == len({j.identity.episode_id for j in requests})
    assert any(e["success"] for e in refused)


@pytest.mark.parametrize("answer", ["continue", "degrade", "failure"])
def test_delayed_or_failed_call_handovers_complete_and_keep_canonical_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    result = execute(
        world,
        tmp_path / answer,
        "supervisor",
        human_settings(),
        config=cfg,
        factory="test_live_pass:fallback_factory",
        supervisor=SupervisorSpec(name="fixture", params={"answer": answer}),
    )
    assert any(e["handover"] for e in result["episodes"])
    assert all(e["actual_calls"] == 1 for e in result["episodes"])
    if answer == "failure":
        assert all(e["model_s"] is None for e in result["episodes"])
        assert all(
            e["calls"][0]["risk"] == 1 and e["calls"][0]["error"] == "fixture timeout"
            for e in result["episodes"]
        )
        assert summarize(result)["model_response_s"] is None
    else:
        failed = [e for e in result["episodes"] if not e["as_run"]]
        assert all(e["rule_decisions"] == 1 for e in failed)
        assert all(e["decisions"][-1]["call_status"] == "rule" for e in failed)


def test_skipped_route_cannot_be_scored_as_a_model_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="skipped model route"):
        execute(
            world,
            tmp_path / "skipped",
            "supervisor",
            human_settings(),
            config=cfg,
            factory="test_live_pass:fallback_factory",
            supervisor=SupervisorSpec(name="fixture", params={"answer": "skipped"}),
        )


def test_net_mark_penalizes_broken_successes_and_uses_window_operator_time() -> None:
    base = record([row(1), row(2, success=False), row(3, as_run=True, weight=4)], "decision")
    candidate = record([row(1), row(2), row(3, as_run=True, weight=4)], "supervisor")
    assert summarize(base)["rescued_as_run"] == 1
    assert summarize(base)["successes_lost"] == 0
    good = compare_records(candidate, [base], resamples=40)
    assert good["candidate_net"] == 2 and good["relative_gain"] == 1
    candidate["episodes"][-1]["success"] = False
    broken = compare_records(candidate, [base], resamples=40)
    assert broken["candidate_net"] == -2 and broken["relative_gain"] == -3
    assert not broken["point_meets_20_percent"]
    weighted = record(
        [row(1, operator_s=0), row(2, operator_s=0), row(3, as_run=True, weight=4, operator_s=5)],
        "supervisor",
    )
    measured = compare_records(weighted, [base], resamples=40)
    assert measured["budget_s"] == 20 and measured["best_simple_rule"]["net"] == pytest.approx(
        1 / 3
    )
    assert measured["relative_gain"] == pytest.approx(5)


def test_pass_mark_at_twenty_percent_and_bootstrap_really_resamples() -> None:
    baselines = record([row(i, success=i <= 10) for i in range(1, 21)])
    candidate = record([row(i, success=i <= 11) for i in range(1, 21)], "supervisor")
    result = compare_records(candidate, [baselines], resamples=50)
    assert result["relative_gain"] == pytest.approx(0.1)
    assert result["point_meets_20_percent"] is False
    interval = result["difference_interval95"]
    assert interval[0] != interval[1]
    candidate["episodes"][11]["success"] = True
    assert compare_records(candidate, [baselines], resamples=50)["point_meets_20_percent"]


def test_hold_control_sets_the_zero_and_losses_and_rescues_stay_visible() -> None:
    candidate = record([row(1), row(2)], "supervisor")
    candidate["hold_control"] = {
        "episodes": [
            {"episode_id": "e000001", "success": True},
            {"episode_id": "e000002", "success": False},
        ]
    }
    base = record([row(1), row(2, success=False)])
    comparison = compare_records(candidate, [base], resamples=20)
    assert comparison["candidate_control_zero"] == 1
    assert comparison["candidate_net"] == 2
    assert comparison["candidate_net_above_control"] == 1
    assert comparison["relative_gain"] == 1
    assert comparison["point_meets_20_percent"]


def test_comparison_checks_pairing_simple_rules_and_names_unavailable_time(tmp_path: Path) -> None:
    candidate = record([row(1)], "supervisor")
    with pytest.raises(ValueError, match="simple rule"):
        compare_records(candidate, [record([row(1)], "supervisor")])
    with pytest.raises(ValueError, match="pairing"):
        compare_records(candidate, [record([row(2)])])
    candidate["episodes"][0]["operator_s"] = 11
    result = compare_records(candidate, [record([row(1)])])
    assert result["status"] == "outside_measured_baseline_time"
    assert "exceeds" in result["reason"] and "relative_gain" not in result
    dump(tmp_path / "base.json", record([row(1)]))
    dump(tmp_path / "candidate.json", candidate)
    score_files([tmp_path / "base.json", tmp_path / "candidate.json"], tmp_path / "scores")
    assert "no extrapolation" in (tmp_path / "scores/table.md").read_text()


def test_published_mark_uses_same_supported_resamples_for_both_nets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.live import scoring

    monkeypatch.setattr(scoring, "resample_indices", lambda *args: iter([[0, 0], [1, 1]]))
    candidate = record([row(1, success=False, operator_s=1), row(2, operator_s=3)], "supervisor")
    base = record([row(1, operator_s=2), row(2, success=False, operator_s=2)])
    result = compare_records(candidate, [base], resamples=2)
    assert result["point_meets_20_percent"] is False
    # Second draw cannot compare at equal time (6s candidate, 4s baseline).
    assert result["candidate_net_interval95"] == [0, 0]
    assert result["baseline_net_interval95"] == [1, 1]
    assert result["difference_interval95"] == [-1, -1]
    assert result["interval"]["valid_difference"] == 1


def test_published_mark_and_hold_control_intervals_are_separate(tmp_path: Path) -> None:
    candidate = record([row(1), row(2)], "supervisor")
    base = record([row(1), row(2, success=False)])
    holds = record([row(1, success=False, operator_s=0), row(2, operator_s=0)], "holds")
    candidate["hold_control"] = {"episodes": holds["episodes"]}
    original = compare_records(candidate, [base], resamples=40)
    holds["episodes"][0]["success"] = True
    changed = compare_records(candidate, [base], resamples=40)
    for key in ("candidate_net", "best_simple_rule", "relative_gain", "point_meets_20_percent"):
        assert changed[key] == original[key]
    assert changed["candidate_control_zero"] != original["candidate_control_zero"]
    paths = []
    for name, rec in (("supervisor", candidate), ("timeout", base), ("holds", holds)):
        path = tmp_path / f"{name}.json"
        dump(path, rec)
        paths.append(path)
    scores = score_files(paths, tmp_path / "scores", resamples=40)
    assert len(scores["comparisons"]) == 1
    assert all(r["estimated_window_net_interval95"] is not None for r in scores["rows"])
    assert (
        scores["rows"][0]["hold_control_net_interval95"]
        == scores["rows"][2]["estimated_window_net_interval95"]
    )
    unavailable = compare_records(
        candidate, [record([row(1, success=False), row(2, success=False)])]
    )
    assert unavailable["status"] == "nonpositive_baseline_net"
    assert unavailable["point_meets_20_percent"] is unavailable["relative_gain"] is None
    assert unavailable["candidate_net"] == 2 and unavailable["best_simple_rule"]["net"] == 0


VARYING_ANSWER = "hand_over"
VARYING_MS = 100.0


class VaryingSupervisor:
    def decide(self, context: SupervisorContext) -> SupervisorDecision:
        return SupervisorDecision(Decision(VARYING_ANSWER), "varying fixture", VARYING_MS, 50)


def varying_factory(name: str, params: Any) -> SupervisorParts:
    return SupervisorParts(
        AfterSteps(10), VaryingSupervisor(), should_call=lambda c: not c.previous_decisions
    )


@pytest.mark.parametrize("answer", ["continue", "hand_over"])
def test_resume_discards_abandoned_requests_and_sample_jobs_with_varying_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    from fleetlab.engine.records import LogSink

    world, cfg = live_world(tmp_path, monkeypatch)
    selected, _ = cohort(world)
    original = LogSink.write
    for cut_index in range(len(selected)):
        out = tmp_path / f"cut-{cut_index}"
        monkeypatch.setattr("test_live_pass.VARYING_ANSWER", "hand_over")
        monkeypatch.setattr("test_live_pass.VARYING_MS", 100.0)
        cut_key = selected[cut_index].key

        def interrupt(self: Any, rec: Any, text: str, key: str = cut_key) -> Any:
            if rec.job.mode is JobMode.run and rec.job.key == key:
                raise KeyboardInterrupt("fixture crash before attempt commit")
            return original(self, rec, text)

        monkeypatch.setattr(LogSink, "write", interrupt)
        options = {
            "factory": "test_live_pass:varying_factory",
            "config": cfg,
            "supervisor": SupervisorSpec(name="fixture"),
        }
        with pytest.raises(KeyboardInterrupt):
            execute(world, out, "supervisor", human_settings(), **options)
        monkeypatch.setattr(LogSink, "write", original)
        monkeypatch.setattr("test_live_pass.VARYING_ANSWER", answer)
        monkeypatch.setattr("test_live_pass.VARYING_MS", 300.0)
        result = execute(world, out, "supervisor", human_settings(), **options)
        requests = manifests.read(out / "requests.jsonl")
        ids = [j.identity.episode_id for j in requests]
        assert len(ids) == len(set(ids))
        assert (cut_key in ids) == (answer == "hand_over")
        restarted = next(e for e in result["episodes"] if e["episode_id"] == cut_key)
        assert restarted["calls"][0]["decision_ms"] == 300
        assert len(restarted["calls"]) == 1
        assert result["evidence"]["proposals"]["chain_ok"]
        assert execute(world, out, "supervisor", human_settings(), **options) == result


def test_cohort_applies_window_and_ignores_replay_rows(tmp_path: Path) -> None:
    jobs = [
        Job(identity=ident(i).model_copy(update={"day": day}))
        for i, day in ((1, 0), (2, 0), (3, 1))
    ]
    source(tmp_path, jobs, 1)
    extra = [{"episode_id": j.key, "mode": "replay", "success": False} for j in jobs]
    original = [{"episode_id": j.key, "mode": "run", "success": i == 0} for i, j in enumerate(jobs)]
    table(tmp_path / "runs/as-run/truth/job_outcomes.parquet", original + extra)
    selected, meta = cohort(tmp_path)
    assert [j.key for j in selected] == [j.key for j in jobs[:2]]
    assert meta["window_episodes"] == 2 and meta["window_failures"] == 1


def test_no_supervisor_verifies_failed_as_run_outcomes_and_replay_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live import pass_run

    world, cfg = live_world(tmp_path, monkeypatch)
    result = execute(world, tmp_path / "none", "none", human_settings(), config=cfg)
    assert any(not e["success"] for e in result["episodes"])
    assert all(e["success"] == e["as_run"] for e in result["episodes"])
    real_cohort = pass_run.cohort

    def false_labels(world: Path) -> Any:
        selected, population = real_cohort(world)
        population["episodes"][0]["as_run"] = not population["episodes"][0]["as_run"]
        return selected, population

    monkeypatch.setattr(pass_run, "cohort", false_labels)
    with pytest.raises(ValueError, match="changed the as-run"):
        execute(world, tmp_path / "wrong-label", "none", human_settings(), config=cfg)
    monkeypatch.setattr(pass_run, "cohort", real_cohort)

    def bad_sample(config: Any, jobs: Any) -> Any:
        result = run_in_process(config, jobs)
        result.record = result.record.model_copy(update={"chain_ok": False})
        return result

    monkeypatch.setattr(pass_run, "run", bad_sample)
    with pytest.raises(ValueError, match="failed supervised replay sample"):
        execute(world, tmp_path / "bad-sample", "none", human_settings(), config=cfg)


def test_queue_orders_by_time_not_name_and_observes_slot_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.live import rules

    early = request(3).model_copy(
        update={
            "from_snapshot": SnapshotRef.model_validate(request(3).from_snapshot).model_copy(
                update={"step": 0}
            )
        }
    )
    late = request(1).model_copy(
        update={
            "from_snapshot": SnapshotRef.model_validate(request(1).from_snapshot).model_copy(
                update={"step": 1800}
            )
        }
    )
    queued = request(2).model_copy(
        update={
            "from_snapshot": SnapshotRef.model_validate(request(2).from_snapshot).model_copy(
                update={"step": 1}
            )
        }
    )
    _jobs, denied, events = gate_requests(
        [late, queued, early], {}, HumanSettings(operators=1), 14, None
    )
    assert not denied and [e["episode_id"] for e in events] == [
        early.identity.episode_id,
        queued.identity.episode_id,
        late.identity.episode_id,
    ]
    assert events[0]["free_operators"] == 1
    assert events[1]["free_operators"] == 0 and events[1]["waiting"] == 0
    assert events[2]["free_operators"] == 0 and events[2]["waiting"] == 0
    # With no waiter, a slot that ends exactly at this arrival is free.
    assert (
        gate_requests([early, late], {}, HumanSettings(operators=1), 14, None)[2][-1][
            "free_operators"
        ]
        == 1
    )
    from fleetlab.policies.teleop.human import admit

    native = admit

    def disagree(*args: Any, **kwargs: Any) -> Any:
        jobs = native(*args, **kwargs)
        assert jobs[0].teleoperator_params is not None
        jobs[0].teleoperator_params["queue_steps"] = 1
        return jobs

    monkeypatch.setattr(rules, "admit", disagree)
    with pytest.raises(ValueError, match="admission disagree"):
        gate_requests([early], {}, HumanSettings(), 14, None)


def test_economic_ratios_share_one_recorded_telemetry_run_and_skip_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live import pass_run
    from fleetlab.live.model import TelemetryRiskEstimator

    world, cfg = live_world(tmp_path, monkeypatch)
    calls = []
    original = TelemetryRiskEstimator.update

    def counted(self: Any, telemetry: Any) -> float:
        calls.append(telemetry.step)
        return original(self, telemetry)

    monkeypatch.setattr(TelemetryRiskEstimator, "update", counted)
    tape = pass_run.economic_tape(world, tmp_path / "tape", [1, 2, 4])
    assert tape["points"]["1"] == {}
    source_steps = sum(
        r["steps"]
        for r in __import__("fleetlab.policies.teleop.measure", fromlist=["read_rows"]).read_rows(
            world / "runs/as-run/truth/job_outcomes.parquet"
        )
        if r["mode"] == "run"
    )
    assert len(calls) == source_steps
    assert pass_run.economic_tape(world, tmp_path / "tape", [1, 2, 4]) == tape
    assert len(calls) == source_steps
    for ratio in (2, 4):
        result = execute(
            world,
            tmp_path / f"economic-{ratio}",
            "decision",
            human_settings(),
            config=cfg,
            cost_ratio=ratio,
            economic_points=tape["points"][str(ratio)],
            economic_tape_digest=pass_run.digest(tmp_path / "tape/economic-tape.json"),
        )
        for e in result["episodes"]:
            point = tape["points"][str(ratio)].get(e["episode_id"])
            assert e["request"] == (point is not None)
            assert e["actual_calls"] == 0
            if point is not None:
                assert e["calls"][0]["step"] == point["step"]
                assert e["calls"][0]["risk"] * ratio > 1
    assert len(calls) == source_steps


def test_economic_sweep_does_not_spend_a_ratio_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, _ = live_world(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(pass_cli, "execute", lambda *args, **kw: calls.append(kw["cost_ratio"]))
    monkeypatch.setattr(pass_cli, "score_files", lambda *args: {})
    monkeypatch.setattr(pass_cli, "draw", lambda *args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "live-pass",
            "sweep",
            "--world",
            str(world),
            "--out",
            str(tmp_path / "sweep"),
            "--rule",
            "decision",
        ],
    )
    pass_cli.main()
    assert calls == [2, 4]


def test_expected_cost_includes_the_declared_wait_when_called_directly() -> None:
    context = SupervisorContext({}, [], 0.8, "", [Decision.continue_, Decision.hand_over])
    assert CostRule(2, wait_s=90, idle_cost=1).decide(context).decision is Decision.continue_
    assert CostRule(2, wait_s=0, idle_cost=1).decide(context).decision is Decision.hand_over


def test_record_chooses_first_applied_counter_and_keeps_requests_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live import pass_run

    world, cfg = live_world(tmp_path, monkeypatch)

    def counters(root: Path, config: Any, jobs: Any, decisions: Any) -> Any:
        return {
            j.key: [
                {
                    "decision_index": i,
                    "decision": "stop",
                    "call_step": 5,
                    "snapshot_step": 8,
                    "without": without,
                    "applied": True,
                }
                for i, without in enumerate((False, True))
            ]
            for j in jobs
        }

    monkeypatch.setattr(pass_run, "capture", counters)
    result = execute(
        world,
        tmp_path / "counters",
        "supervisor",
        human_settings(),
        config=cfg,
        factory="test_live_pass:assessment_factory",
        supervisor=SupervisorSpec(name="fixture"),
    )
    assert all(e["snapshot_without"] is False for e in result["episodes"])
    reference = execute(
        world, tmp_path / "timeout", "timeout", human_settings(), after=63, config=cfg
    )
    raw = manifests.read(tmp_path / "timeout/requests.jsonl")
    duplicate = tmp_path / "timeout/fleet/truth/snapshots/e000001/stray.continue.json"
    dump(duplicate, raw[0].model_dump(mode="json"))
    assert (
        execute(world, tmp_path / "timeout", "timeout", human_settings(), after=63, config=cfg)
        == reference
    )


class SettlingCheck(ToyCheck):
    """Constructed check that settles during a pending decision's physical hold."""

    def update(self, state: Any, step: int) -> Any:
        from fleetlab.contracts.env import SuccessVerdict

        self.count += 1
        return SuccessVerdict(self.count >= 8, self.count >= 8, "placed")


def test_completion_during_hold_clips_time_and_leaves_primitive_unapplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.live import counterfactual, pass_run

    world = tmp_path / "world"
    job = Job(identity=ident(1))
    dump(
        world / "files/spec.json",
        {
            "world_id": "w03",
            "windows": {"scoring": {"first": 0, "last": 0}},
            "episodes_per_robot_day": 100,
        },
    )
    manifests.write(world / "files/manifest.jsonl", [job])
    cfg = config(
        world / "runs/as-run",
        episodes_per_robot_day=100,
        check_factory="test_live_pass:SettlingCheck",
    )
    baseline = run_in_process(cfg, [job])
    assert baseline.summaries[0]["success"] and baseline.summaries[0]["steps"] == 8
    monkeypatch.setattr(pass_run, "run", run_in_process)
    monkeypatch.setattr(counterfactual, "run", run_in_process)
    result = execute(
        world,
        tmp_path / "settling",
        "supervisor",
        human_settings(),
        config=cfg,
        factory="test_live_pass:latency_factory",
        supervisor=SupervisorSpec(name="fixture"),
    )
    e = result["episodes"][0]
    assert e["success"] and e["decisions"][0]["hold_steps"] == 5
    assert e["decision_hold_s"] == pytest.approx(3 / 20)
    assert e["interventions"][0]["applied"] is False
    assert e["snapshot_without"] is None


def queue_predicate_factory(name: str, params: Any) -> SupervisorParts:
    from dataclasses import replace

    return replace(
        queue_factory(name, params),
        should_call=lambda context: context.operator_queue.free_operators > 0,
    )


def test_call_predicate_sees_the_same_known_queue_as_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, cfg = live_world(tmp_path, monkeypatch)
    observations = tmp_path / "predicate-observations.json"
    result = execute(
        world,
        tmp_path / "predicate",
        "supervisor",
        human_settings(1),
        config=cfg,
        factory="test_live_pass:queue_predicate_factory",
        supervisor=SupervisorSpec(
            name="fixture", params={"observations": str(observations), "stagger": True}
        ),
    )
    assert sum(e["actual_calls"] for e in result["episodes"]) == 1
    assert sum(e["handover"] for e in result["episodes"]) == 1
    assert json.loads(observations.read_text())[0][:2] == [0, 1]


def test_queue_refusal_uses_the_selected_physical_latency_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_live_model import telemetry

    from fleetlab.live.rules import QueueView, RecordingSupervisor

    prior = request(1).model_copy(
        update={
            "from_snapshot": SnapshotRef.model_validate(request(1).from_snapshot).model_copy(
                update={"step": 0}
            )
        }
    )
    monkeypatch.setattr(RunLayout, "continuations", lambda _layout: [prior])

    class RequestWithTimers:
        def decide(self, context: SupervisorContext) -> SupervisorDecision:
            return SupervisorDecision(Decision.hand_over, "two measured timers", 250, 50)

    for route, expected in (("subscription", Decision.hand_over), ("model", Decision.continue_)):
        path = tmp_path / f"{route}.json"
        supervisor = RecordingSupervisor(
            RequestWithTimers(),
            path,
            route,
            QueueView(
                RunLayout(tmp_path / "fleet"),
                Job(identity=request(2).identity),
                HumanSettings(
                    operators=1,
                    service_s=0.5,
                    expert_steps=1,
                    pace=1,
                    reaction_min_s=0,
                    reaction_max_s=0,
                ),
                14,
                QueuePolicy(max_wait_s=0),
            ),
        )
        answer = supervisor.decide(
            SupervisorContext({}, [telemetry(4)], 1, "", [Decision.continue_, Decision.hand_over])
        )
        assert answer.decision is expected
        event = json.loads(path.read_text())[0]
        assert event["requested"] and event["refused"] == (route == "model")
        assert event["decision_ms"] == 250 and event["model_ms"] == 50


def test_frozen_launcher_reads_the_committed_profile_from_its_own_checkout(tmp_path: Path) -> None:
    import os
    import subprocess

    repo = tmp_path / "caller"
    live = repo / "src/fleetlab/live"
    live.mkdir(parents=True)
    (live / "run_grid.sh").write_text(
        (Path(__file__).resolve().parents[1] / "src/fleetlab/live/run_grid.sh").read_text()
    )
    (live / "grid.sh").write_text(
        'set -eu\ntest "$3" = "$PWD/src/fleetlab/live/profile.json"\ntest "$(cat "$3")" = frozen\n'
    )
    (live / "profile.json").write_text("frozen\n")
    (repo / "Makefile").write_text(".PHONY: setup\nsetup:\n\ttrue\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Live launcher fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "frozen fixture",
        ],
        cwd=repo,
        check=True,
    )
    (live / "profile.json").write_text("caller changed after commit\n")
    world = tmp_path / "world"
    world.mkdir()
    completed = subprocess.run(
        [
            "bash",
            str(live / "run_grid.sh"),
            "HEAD",
            str(world),
            str(tmp_path / "result"),
            "src/fleetlab/live/profile.json",
            "fixture:make",
        ],
        cwd=repo,
        env={**os.environ, "GRID_DRY_START": "1"},
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_grid_preflight_prepares_both_boundaries_without_gpu_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.engine import miopen, software
    from fleetlab.live import preflight
    from fleetlab.live.vlm import supervisor as vlm

    world, _cfg = live_world(tmp_path, monkeypatch)
    selected, _ = cohort(world)
    monkeypatch.setattr(miopen, "require", lambda *args: "fixture-database")
    monkeypatch.setattr(software, "current", lambda *args: selected[0].identity.software)
    monkeypatch.setattr(preflight, "code_version", lambda: {"commit": "fixture", "dirty": False})
    gates = []
    monkeypatch.setattr(preflight, "gate", lambda backends: gates.append(backends))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("dry start attempted simulation or a provider call")

    monkeypatch.setattr("fleetlab.live.pass_run.run", forbidden)
    monkeypatch.setattr(vlm, "run_agent", forbidden)
    profile = tmp_path / "profile.json"
    dump(profile, {"model": "gpt-6-astra", "effort": "low", "first_step": 5, "max_calls": 1})
    out = tmp_path / "dry"
    receipt = preflight.prepare(
        world, out, profile, "fleetlab.engine.supervisors:make", 2, "development", True
    )
    assert gates == [["codex"]]
    assert receipt["base_attempts"] == 17 * len(selected)
    assert receipt["expected_model_calls"] == 3 * len(selected)
    assert receipt["memory"]["required_available_gib"] == pytest.approx(16.2)
    gpu = json.loads((out / "gpu-boundary/dry-start.json").read_text())
    model = json.loads((out / "model-boundary/dry-start.json").read_text())
    assert gpu["config"]["workers"] == 2 and model["config"]["workers"] == 1
    call = json.loads((out / "model-boundary/call-boundary.json").read_text())
    assert call["model"] == "gpt-6-astra" and call["effort"] == "low"
    assert "synthetic" in call["input_scope"] and call["images"] == 2
    assert not list(out.rglob("record.json"))
    assert not list(out.rglob("run.json"))


def test_cli_loads_the_actual_revision_profile() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "src/fleetlab/live/vlm/profiles/one_early_look.json"
    )
    spec = pass_cli.load_supervisor(path)
    assert spec.name == "vlm"
    assert spec.params["model"] == "gpt-6-astra" and spec.params["effort"] == "low"
    assert spec.params["first_step"] == 120 and spec.params["max_calls"] == 1


@pytest.mark.parametrize("cohort_size", [140, 141])
def test_test_grid_preflight_retains_full_cohort_above_decided_budget_estimate(
    tmp_path: Path, cohort_size: int
) -> None:
    import os
    import shlex
    import subprocess
    import sys

    # This isolates budgeting after readiness; the frozen w42 dry start checks readiness.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\nset -eu\nshift 2\n"
        'if [[ $1 == python && $2 == -m ]]; then cat "$COHORT_FIXTURE";\n'
        f'else exec {shlex.quote(sys.executable)} "${{@:2}}"; fi\n'
    )
    uv.chmod(0o755)
    preview = tmp_path / "preview.json"
    dump(preview, {"window_failures": cohort_size - 50, "sampled_successes": 50})
    world = tmp_path / "world"
    world.mkdir()
    script = Path(__file__).resolve().parents[1] / "src/fleetlab/live/grid.sh"
    completed = subprocess.run(
        ["bash", str(script), str(world), str(tmp_path / "out"), "profile.json", "fixture:make"],
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "GRID_MODE": "test",
            "GRID_DRY_START": "1",
            "COHORT_FIXTURE": str(preview),
        },
        text=True,
        capture_output=True,
    )
    assert f"10 full-cohort passes, at most {10 * cohort_size} attempts" in completed.stdout
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DRY START" in completed.stdout
    if cohort_size > 140:
        assert "Budget estimate exceeded openly; full cohort retained" in completed.stdout


def test_hold_control_has_a_verifiable_immutable_source_after_interrupted_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.engine.records import LogSink
    from fleetlab.live import pass_run

    world, cfg = live_world(tmp_path, monkeypatch)
    out = tmp_path / "interrupted-control"
    options = {
        "factory": "test_live_pass:varying_factory",
        "config": cfg,
        "supervisor": SupervisorSpec(name="fixture"),
    }
    monkeypatch.setattr("test_live_pass.VARYING_ANSWER", "hand_over")
    monkeypatch.setattr("test_live_pass.VARYING_MS", 100.0)
    original_write = LogSink.write

    def interrupt_write(self: Any, rec: Any, text: str) -> Any:
        raise KeyboardInterrupt("fixture crash inside hold control")

    def interrupt_control(config: Any, jobs: Any) -> Any:
        if "hold-control" in Path(config.out).parts:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(LogSink, "write", interrupt_write)
                return run_in_process(config, jobs)
        return run_in_process(config, jobs)

    monkeypatch.setattr(pass_run, "run", interrupt_control)
    with pytest.raises(KeyboardInterrupt, match="inside hold control"):
        execute(world, out, "supervisor", human_settings(), **options)
    source_path = out / "hold-source/record.json"
    initial_source = source_path.read_bytes()
    monkeypatch.setattr(LogSink, "write", original_write)
    monkeypatch.setattr(pass_run, "run", run_in_process)
    monkeypatch.setattr("test_live_pass.VARYING_ANSWER", "continue")
    monkeypatch.setattr("test_live_pass.VARYING_MS", 400.0)
    result = execute(world, out, "supervisor", human_settings(), **options)
    assert source_path.read_bytes() == initial_source
    control = json.loads(Path(result["hold_control"]["record"]).read_text())
    assert result["hold_control"]["source_record"] == str(source_path.resolve())
    assert (
        control["config"]["reference_sha256"]
        == result["hold_control"]["source_sha256"]
        == pass_run.digest(source_path)
    )
    source = json.loads(initial_source)
    assert [e["decisions"] for e in source["episodes"]] == [
        e["decisions"] for e in result["episodes"]
    ]
    assert all(e["actual_calls"] == 0 and not e["handover"] for e in control["episodes"])
    assert all(e["calls"][0]["decision_ms"] == 100 for e in result["episodes"])
    assert execute(world, out, "supervisor", human_settings(), **options) == result


def test_figure_renders_a_supervisor_without_deciding_holds_or_control(tmp_path: Path) -> None:
    from fleetlab.live.figure import draw

    summary = summarize(record([row(1), row(2, as_run=True)], "supervisor"))
    assert summary["hold_control_net"] == 0.0
    summary["estimated_window_model_s"] = None
    table_path = tmp_path / "table.json"
    dump(table_path, {"rows": [summary]})
    output = tmp_path / "without-control.png"
    draw(table_path, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_grid_scores_a_no_hold_supervisor_without_missing_control_files(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    # Dispatch-only fixture: the actual random pass and rendering have separate tests.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, subprocess, sys\n"
        "args = sys.argv[4:]\n"
        "if args[0] == '-':\n"
        "    raise SystemExit(subprocess.call([sys.executable, *args], stdin=sys.stdin))\n"
        "if args[1] != 'fleetlab.live.pass_cli':\n"
        "    raise SystemExit(0)\n"
        "args = args[2:]\n"
        "if args[0] == 'cohort':\n"
        "    print(json.dumps({'window_failures': 2, 'sampled_successes': 50}))\n"
        "    raise SystemExit(0)\n"
        "out = pathlib.Path(args[args.index('--out') + 1])\n"
        "if args[0] == 'score':\n"
        "    records = args[1:args.index('--out')]\n"
        "    assert len(records) == 7, records\n"
        "    assert all(pathlib.Path(p).is_file() for p in records), records\n"
        "    assert not any('hold-control' in p for p in records), records\n"
        "    raise SystemExit(0)\n"
        "destinations = [out]\n"
        "if args[0] == 'sweep':\n"
        "    rule = args[args.index('--rule') + 1]\n"
        "    flag, prefix = ('--timeouts', 'after-') if rule == 'timeout' "
        "else ('--ratios', 'cost_ratio-')\n"
        "    start = args.index(flag) + 1\n"
        "    stop = next(i for i in range(start, len(args)) if args[i].startswith('--'))\n"
        "    destinations = [out / (prefix + r) for r in args[start:stop]]\n"
        "for destination in destinations:\n"
        "    destination.mkdir(parents=True, exist_ok=True)\n"
        "    (destination / 'record.json').write_text('{}')\n"
    )
    uv.chmod(0o755)
    world = tmp_path / "world"
    world.mkdir()
    script = Path(__file__).resolve().parents[1] / "src/fleetlab/live/grid.sh"
    completed = subprocess.run(
        ["bash", str(script), str(world), str(tmp_path / "out"), "profile.json", "fixture:make"],
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "GRID_MODE": "test",
            "GRID_DRY_START": "0",
        },
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Stored table and figure" in completed.stdout
