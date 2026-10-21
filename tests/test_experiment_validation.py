"""Public preflight and engine fault isolation without simulation or model calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from engine_helpers import SOFTWARE, config
from log_toy import TASK_CLASSES, write_toy_run
from test_engine_experiments import queue, recalibrate, request, toy_world

from fleetlab.contracts.actions import Action
from fleetlab.contracts.faults import Target
from fleetlab.contracts.hashing import hash_json
from fleetlab.contracts.vocabulary import ConfigKey, Verb
from fleetlab.engine.episode import EngineError
from fleetlab.engine.experiments import plan_queue, queue_dirs, run_queue
from fleetlab.log import export_runs


def test_bad_requests_do_not_allocate_jobs_or_suppress_valid_requests(tmp_path: Path) -> None:
    world = toy_world([], 1)
    invalid = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.control_mode),
        value="sideways",
    )
    q = queue(
        tmp_path,
        1,
        ("x0001", request(["r01"], [], [invalid], n=2)),
        ("x0002", request(["r01"], [], [], n=2)),
        ("x0003", request(["r01"], [], [recalibrate("r01")], n=2)),
    )
    allocations: dict[str, object] = {}
    planned = plan_queue(q, world, allocations, software=SOFTWARE, batch_size=4, truth={})
    assert [r.status for r in planned.results] == ["refused", "refused", "done"]
    assert [r.message for r in planned.results[:2]] == ["invalid_action_value", "identical_arms"]
    assert len(planned.jobs) == 4
    assert {j.identity.experiment_id for j in planned.jobs} == {"x0003"}
    assert allocations["next_episode"] == 5


def test_machine_failure_stops_without_a_reply_and_preserves_the_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fleet = write_toy_run(tmp_path / "observed")
    export = tmp_path / "export"
    export_runs([tmp_path / "observed"], export, fleet=fleet, task_classes=TASK_CLASSES)
    world = toy_world([str(tmp_path / "observed")], 100)
    q = queue(export, 1, ("x0001", request(["r01"], [], [recalibrate("r01")], n=1)))
    path = tmp_path / "queue.json"
    path.write_text(q.model_dump_json())
    monkeypatch.setattr("fleetlab.engine.fleet.prepare", lambda cfg: SOFTWARE)
    monkeypatch.setattr("fleetlab.engine.experiments._truth_table", lambda name: {})
    execute = Mock(side_effect=EngineError("private_trait and hidden mechanism"))
    monkeypatch.setattr("fleetlab.engine.fleet.run", execute)
    monkeypatch.setattr("fleetlab.engine.experiments.export_for", lambda *a: export)
    timings: dict[str, float] = {}
    with pytest.raises(EngineError, match="private_trait"):
        run_queue(
            path, world, tmp_path / "engine", config=config(tmp_path / "unused"), timings=timings
        )
    assert execute.call_count == 1
    plan = next((tmp_path / "engine").rglob("*.plan.json"))
    assert plan.is_file() and timings["simulation_seconds"] >= 0
    stored = plan.read_bytes()
    # Retrying the same queue uses the saved allocation, rather than refusing
    # the whole queue or minting a new set of arm episodes.
    execute.side_effect = None
    reply = run_queue(path, world, tmp_path / "engine", config=config(tmp_path / "unused"))
    assert execute.call_count == 2 and plan.read_bytes() == stored
    assert reply.results[0].status == "done"
    assert "private_trait" not in reply.model_dump_json()


def test_a_cached_plan_cannot_bypass_public_request_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = tmp_path / "observed"
    fleet = write_toy_run(observed)
    export = tmp_path / "export"
    export_runs([observed], export, fleet=fleet, task_classes=TASK_CLASSES)
    world = toy_world([str(observed)], 100)
    valid = queue(export, 1, ("x0001", request(["r01"], [], [recalibrate("r01")], n=1)))
    planned = plan_queue(valid, world, {}, software=SOFTWARE, batch_size=4, truth={})
    invalid = Action(
        verb=Verb.set,
        target=Target(scope="r01", config_key=ConfigKey.control_mode),
        value="sideways",
    )
    legacy = queue(export, 1, ("x0001", request(["r01"], [], [invalid], n=1)))
    path = tmp_path / "queue.json"
    path.write_text(legacy.model_dump_json())
    engine = tmp_path / "engine"
    run_dir, _ = queue_dirs(engine, legacy.analysis_id, legacy.session)
    run_dir.parent.mkdir(parents=True)
    (run_dir.parent / f"{run_dir.name}.plan.json").write_text(
        json.dumps(
            {
                "queue_sha256": hash_json(legacy.model_dump(mode="json")),
                "jobs": [j.model_dump(mode="json") for j in planned.jobs],
                "results": [r.model_dump(mode="json") for r in planned.results],
            }
        )
    )
    monkeypatch.setattr("fleetlab.engine.fleet.prepare", lambda cfg: SOFTWARE)
    execute = Mock(side_effect=AssertionError("invalid cached request must not run"))
    monkeypatch.setattr("fleetlab.engine.fleet.run", execute)
    reply = run_queue(path, world, engine, config=config(tmp_path / "unused"), export=False)
    assert not execute.called
    assert reply.results[0].status == "refused"
    assert reply.results[0].message == "invalid_action_value"


@pytest.mark.parametrize(
    "key,value,valid",
    [
        ("lighting", 0.0, True),
        ("lighting", -2.0, True),
        ("lighting", 2.0, True),
        ("lighting", [0.0], True),
        ("lighting", [-1.0, 0.5], True),
        ("lighting", [], False),
        ("lighting", [1.0, 2.0, 3.0], False),
        ("object_batch", [0.0, 0.0], True),
        ("object_batch", [-0.5, 1.0], True),
        ("object_batch", 1.0, False),
        ("object_batch", [1.0], False),
        ("object_batch", [1.0, 2.0, 3.0], False),
        ("placement", [0.0, 0.0], True),
        ("placement", [-1.0, 2.0], True),
        ("placement", [1.0, 2.0, -3.0], True),
        ("placement", 4.0, False),
        ("placement", [0.0], False),
        ("placement", [0.0, 1.0, 2.0, 3.0], False),
        ("clutter", 0.0, True),
        ("clutter", 1.0, True),
        ("clutter", 0.5, True),
        ("clutter", [0.5], True),
        ("clutter", -0.1, False),
        ("clutter", 5.0, False),
        ("clutter", [0.0, 0.0], False),
    ],
)
def test_condition_domains_match_registered_hook(key: str, value: object, valid: bool) -> None:
    """Application semantics only: no cell, simulator, calibration or model runs."""
    from engine_helpers import ident

    import fleetlab.faults
    from fleetlab.contracts.actions import CONDITION_CHANGE_KIND, resolve
    from fleetlab.contracts.vocabulary import ConditionKey
    from fleetlab.toolserver.action_validation import value_problem

    condition = ConditionKey(key)
    action = Action.model_validate(
        {
            "verb": "change_condition",
            "target": {"scope": "s1", "condition_key": key},
            "value": value,
        }
    )
    assert (value_problem(action) is None) is valid
    identity = ident()
    # A distinct placeholder current value forces dispatch even for the public
    # zero value; nominal equality must not bypass shape validation in the test.
    resolution = resolve(action, identity, [], {}, {condition: "test current value"})
    assert len(resolution.added) == 1
    registered = fleetlab.faults.REGISTRY.get(CONDITION_CHANGE_KIND)
    hook = registered() if isinstance(registered, type) else registered
    if valid:
        state = hook.new_episode(resolution.added[0], identity)
        assert state.key == condition  # type: ignore[union-attr]
    else:
        with pytest.raises(ValueError):
            hook.new_episode(resolution.added[0], identity)


def test_one_request_resolver_failure_does_not_refuse_its_neighbour(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from fleetlab.engine import experiments

    original = experiments._mint

    def mint(*args: object, **kwargs: object) -> object:
        if args[1] == "x0001":
            raise ValueError("private request-specific detail")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(experiments, "_mint", mint)
    q = queue(
        tmp_path,
        1,
        ("x0001", request(["r01"], [], [recalibrate("r01")], n=1)),
        ("x0002", request(["r01"], [], [recalibrate("r01")], n=1)),
    )
    planned = plan_queue(q, toy_world([], 1), {}, software=SOFTWARE, batch_size=4, truth={})
    assert [r.status for r in planned.results] == ["refused", "done"]
    assert planned.results[0].message == "request_not_runnable"
    assert {j.identity.experiment_id for j in planned.jobs} == {"x0002"}
    assert "private request-specific detail" in caplog.text
    assert all("private" not in r.model_dump_json() for r in planned.results)
