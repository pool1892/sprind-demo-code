"""Visible-only inputs, faithful budget cuts, bounded calls and duplicate video nominations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import TASK_CLASSES, frame_value, toy_fleet, write_toy_run
from PIL import Image

from fleetlab.agents.run import AgentResult
from fleetlab.agents.sessions import Analysis
from fleetlab.baselines._seed import default_seed
from fleetlab.baselines.common import (
    component_actions,
    export_catalogue,
    static_description,
    write_answer,
)
from fleetlab.baselines.models import (
    BaselineModelError,
    model_call,
    no_data_ranker,
    validate_actions,
)
from fleetlab.baselines.simple import fixed_prior, random_fixes, worst_health
from fleetlab.baselines.video import dramatic_failures, dramatic_videos, filmstrip
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, Budget, catalogue_rows, fits_budget
from fleetlab.contracts.env import Camera
from fleetlab.contracts.log import visible_schema
from fleetlab.contracts.model import ConfigVersion, describe_for_agents
from fleetlab.contracts.vocabulary import Component, ConditionKey, ConfigKey
from fleetlab.log import export_runs
from fleetlab.log.reader import LogReader
from fleetlab.scoring.baselines import read_baseline_fix_set

# Independent literals: mutating the implementation's constants must fail.
EXPECTED_SCORING_RULE = (
    "Neither changes what the robots do in the scoring window, so their expected gain "
    "in fleet success is zero; they belong at the bottom of the list, and in the "
    "diagnosis as a retrain indicator."
)
EXPECTED_SET_RULE = (
    "A set writes the recorded standard value of that key at that scope, so it "
    "changes something only where the current value at that scope differs from "
    "the standard, and at fleet scope it changes nothing on a robot or site that "
    "carries a value of its own."
)


def table(export: Path, name: str, rows: list[dict[str, Any]], episode: str = "") -> None:
    spec = visible_schema()[name]
    path = export / spec.path.format(episode_id=episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(c.name, c.type) for c in spec.columns])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.fixture
def export(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    (root / "log").mkdir(parents=True)
    fleet = toy_fleet().model_copy(
        update={
            "config": ConfigVersion(
                version=0,
                values={
                    ConfigKey.policy_sampling_steps: 11,
                    ConfigKey.policy_actions_executed_per_chunk: 29,
                    ConfigKey.policy_image_views: 4,
                    ConfigKey.policy_precision: "fp16_res32",
                    ConfigKey.control_mode: "absolute",
                    ConfigKey.control_action_delay_steps: 2,
                    ConfigKey.control_latency_compensation_steps: 0,
                    ConfigKey.camera_scene_calibration: [1.5, 2.0, 3.0],
                    ConfigKey.camera_wrist_calibration: [4.5, 5.0, 6.0],
                    ConfigKey.software_version: "4.7.3",
                },
            ),
        }
    )
    for i, site in enumerate(fleet.sites):
        site.conditions.update(
            {
                ConditionKey.lighting: [0.25 + i, -0.1],
                ConditionKey.placement: [1.0 + i, 2.0, 3.0],
                ConditionKey.object_batch: [0.5, 0.75 + i],
                ConditionKey.clutter: 0.25 + 0.1 * i,
            }
        )
    chance = root.parent / "oracle/oracle.json"
    chance.parent.mkdir()
    chance.write_text(
        json.dumps(
            {"world_id": fleet.id, "world_hash": "a" * 64, "seed": 7, "chance": "secret-gain"}
        )
    )
    fleet.sites[0].robots[0].components[Component.scene_camera].health.update(canary_health=991)
    (root / "log/description.json").write_text(json.dumps(describe_for_agents(fleet)))
    (root / "data_dictionary.md").write_text("synthetic visible export")
    table(root, "catalogue", catalogue_rows(fleet, ["put"]))
    health = []
    for robot in ("r01", "r02"):
        for component, metric, value in (
            ("scene_camera", "calibration_residual", 1.0),
            ("wrist_camera", "calibration_residual", 1.0),
            ("gripper", "grip_force", 50.0),
            ("gripper", "cycle_count", 100.0),
            ("arm", "joint_torque_rms", 5.0),
            ("controller", "cpu_load", 0.5),
        ):
            for day in (1, 2, 3):
                health.append(
                    dict(day=day, robot_id=robot, component=component, metric=metric, value=value)
                )
    table(root, "health", health)
    # Hidden siblings have a canary; none of the baseline methods reads them.
    (root.parent / "fault_table.json").write_text('{"canary": "hidden-truth-canary"}')
    episodes = []
    for i, (outcome, duration, distance) in enumerate(
        [
            ("failure", 10.0, 2.0),
            ("failure", 10.0, 5.0),
            ("failure", 9.0, 99.0),
            ("success", 100.0, 100.0),
        ],
        1,
    ):
        episode = f"e{i:06}"
        episodes.append(
            dict(
                episode_id=episode,
                robot_id="r01",
                site_id="s1",
                day=1,
                outcome=outcome,
                duration_s=duration,
                steps=2,
                task_text="place",
                video_scene=f"video/{episode}.scene.mp4",
                video_wrist=None,
            )
        )
        table(
            root,
            "steps",
            [
                dict(episode_id=episode, step=j, eef_pos=[j * distance, 0.0, 0.0], frame_index=j)
                for j in range(2)
            ],
            episode,
        )
    table(root, "episodes", episodes)
    from test_scoring_gen3 import certify_fixture

    certify_fixture(root, fleet.id)
    (root.parent / "files").mkdir()
    (root.parent / "files/hash").write_text("a" * 64)
    return root


def fake_rank(*args: Any, **kwargs: Any) -> AgentResult:
    assert args[:2] == ("codex", "gpt-6-astra")
    assert kwargs["tools"] is None and kwargs["auth"] == "subscription"
    assert kwargs["effort"] == "xhigh" and kwargs["service_tier"] == "fast"
    return AgentResult(
        ok=True,
        result={"choices": [{"index": i, "rationale": "prior only"} for i in range(10)]},
        usage={},
        seconds=0.0,
    )


def test_random_uniform_permutations_match_stage2_seed_rule(export: Path) -> None:
    pool = export_catalogue(export)
    draws = random_fixes(export, seed=123, draws=200)
    generator = np.random.default_rng(123)
    for answer in draws:
        expected = [pool[int(i)] for i in generator.permutation(len(pool))]
        assert [r.action for r in answer.ranked.actions] == expected
        for budget in SCORING_BUDGETS:
            fix = answer.fix_sets()[f"{budget.kind}-{budget.amount:g}"]
            assert fits_budget(fix)
            if budget.kind == "count":
                assert fix.actions == expected[: int(budget.amount)]
    assert draws[0].ranked == random_fixes(export, seed=123, draws=1)[0].ranked
    assert draws[0].ranked != random_fixes(export, seed=124, draws=1)[0].ranked


def test_catalogue_filling_does_not_filter_difficult_actions(export: Path) -> None:
    pool = export_catalogue(export)
    with LogReader(export) as reader:
        original = reader.query("SELECT * FROM catalogue").to_pylist()
    assert len(pool) == len(original)
    assert all(a.is_complete for a in pool)
    assert any(a.verb.value == "set" for a in pool)
    assert any(a.verb.value == "request_training_data" for a in pool)
    fleet = json.loads((export / "log/description.json").read_text())["fleet"]
    assert all(
        a.value == fleet["config"]["values"][a.target.config_key]
        for a in pool
        if a.verb.value == "set"
    )
    sites = {s["id"]: s for s in fleet["sites"]}
    assert all(
        a.value == sites[a.target.scope]["conditions"][a.target.condition_key]
        for a in pool
        if a.verb.value == "change_condition"
    )
    assert all(
        a.value == "any failure mode" for a in pool if a.verb.value == "request_training_data"
    )


def test_no_data_prompt_has_only_allowlisted_structure(export: Path) -> None:
    calls = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(args[3])
        payload = json.loads(args[3].split("\n", 1)[1])
        text = json.dumps(payload)
        assert set(payload) == {"system", "catalogue"}
        assert set(payload["system"]) == {"sites", "vocabulary", "costs"}
        for site in payload["system"]["sites"]:
            assert set(site) == {"id", "robots"}
            for robot in site["robots"]:
                assert set(robot) == {"id", "components"}
        for forbidden in (
            "health",
            "installed_day",
            "last_service_day",
            "change_log",
            "since_day",
            "canary",
        ):
            assert forbidden not in text
        assert all("value" not in a for a in payload["catalogue"] if a["verb"] != "route_to_teleop")
        assert "simple baseline" not in args[2] + args[3]
        assert kwargs["images"] == ()
        return fake_rank(*args, **kwargs)

    answer = no_data_ranker(export, caller=caller)
    assert len(calls) == 1 and answer.method == "no_data"
    assert len(answer.ranked.actions) == 10
    assert "health" not in json.dumps(static_description(export))
    # Changes to actual fleet/site/robot settings must not change any prompt byte.
    path = export / "log/description.json"
    raw = json.loads(path.read_text())
    for config in [
        raw["fleet"]["config"],
        *[s["config"] for s in raw["fleet"]["sites"]],
        *[r["config"] for s in raw["fleet"]["sites"] for r in s["robots"]],
    ]:
        config["values"].update(
            {
                "control.mode": "secret-setting-canary",
                "control.action_delay_steps": 87654321,
            }
        )
        config["version"] = 87654322
    for site in raw["fleet"]["sites"]:
        site["conditions"] = {key: "secret-condition-canary" for key in site["conditions"]}
    path.write_text(json.dumps(raw))
    no_data_ranker(export, caller=caller)
    assert calls[0] == calls[1]
    assert "secret-setting-canary" not in calls[1] and "87654321" not in calls[1]
    assert "secret-condition-canary" not in calls[1]


def test_no_data_fills_recorded_values_and_explains_scoring(export: Path) -> None:
    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        assert EXPECTED_SCORING_RULE in args[3]
        assert EXPECTED_SET_RULE in args[3]
        assert "robot over site over fleet" in args[3]
        assert "Setting an unchanged value changes nothing" in args[3]
        assert "do not choose or return a value" in args[3]
        payload = json.loads(args[3].split("\n", 1)[1])
        chosen = [a for a in payload["catalogue"] if a["verb"] == "set"][:10]
        return AgentResult(
            ok=True,
            result={
                "choices": [dict(index=a["index"], rationale="standard setting") for a in chosen]
            },
            usage={},
            seconds=0,
        )

    answer = no_data_ranker(export, caller=caller)
    standards = json.loads((export / "log/description.json").read_text())["fleet"]["config"][
        "values"
    ]
    assert all(
        r.action.value == standards[r.action.target.config_key] for r in answer.ranked.actions
    )
    assert len({r.action.model_dump_json() for r in answer.ranked.actions}) == 10


@pytest.mark.parametrize("value", [None, "proposed prose", 17])
def test_model_cannot_supply_any_value(export: Path, value: Any) -> None:
    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        result = fake_rank(*args, **kwargs)
        result.result["choices"][0]["value"] = value
        return result

    with pytest.raises(ValueError, match="Extra inputs"):
        no_data_ranker(export, caller=caller)


def test_teleoperation_task_class_cannot_change(export: Path) -> None:
    pool = export_catalogue(export)
    route = next(a for a in pool if a.verb.value == "route_to_teleop")
    validate_actions([route], pool)
    with pytest.raises(ValueError, match="task class"):
        validate_actions([Action(verb=route.verb, target=route.target, value="other task")], pool)


@pytest.mark.parametrize("verb", ["set", "change_condition"])
def test_unapplicable_recorded_values_fail_before_answer_is_accepted(
    export: Path, verb: str
) -> None:
    path = export / "log/description.json"
    raw = json.loads(path.read_text())
    raw["fleet"]["config"]["values"]["policy.sampling_steps"] = "bad steps"
    raw["fleet"]["sites"][0]["conditions"]["lighting"] = "prose condition"
    path.write_text(json.dumps(raw))

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        payload = json.loads(args[3].split("\n", 1)[1])
        chosen = next(
            a["index"]
            for a in payload["catalogue"]
            if a["verb"] == verb
            and (
                a["target"]["config_key"] == "policy.sampling_steps"
                if verb == "set"
                else a["target"]["condition_key"] == "lighting" and a["target"]["scope"] == "s1"
            )
        )
        indices = [chosen, *[i for i in range(10) if i != chosen]][:10]
        return AgentResult(
            ok=True,
            result={"choices": [dict(index=i, rationale="prior") for i in indices]},
            usage={},
            seconds=0,
        )

    with pytest.raises(ValueError, match="cannot be applied"):
        no_data_ranker(export, caller=caller)


def test_fallback_labelled_and_stage2_prior_kept(export: Path) -> None:
    answer = fixed_prior(export)
    assert answer.metadata["fallback"] is True
    assert answer.ranked.actions[0].action.target.scope == "r01"
    assert answer.ranked.actions[0].action.target.component == Component.scene_camera


def test_bad_grip_force_and_camera_units_and_health_ties(export: Path) -> None:
    rows = pq.read_table(export / "log/health.parquet").to_pylist()
    for row in rows:
        if row["robot_id"] == "r02" and row["metric"] == "grip_force":
            row["value"] = 1.0
        if row["robot_id"] == "r02" and row["component"] == "scene_camera":
            row["value"] = 100.0
    table(export, "health", rows)
    answer = worst_health(export)
    first = answer.ranked.actions[0].action
    assert (first.target.scope, first.target.component) == ("r02", Component.scene_camera)
    assert first.verb.value == "recalibrate"
    assert (
        len({(a.action.target.scope, a.action.target.component) for a in answer.ranked.actions})
        == 10
    )
    # Huge cycle-count units on both robots are normal; a bad grip force still rises.
    assert any(
        a.action.target.scope == "r02" and a.action.target.component == Component.gripper
        for a in answer.ranked.actions[:3]
    )
    gripper_fix = next(
        a.action
        for a in answer.ranked.actions
        if a.action.target.scope == "r02" and a.action.target.component == Component.gripper
    )
    assert gripper_fix.verb.value == "service"  # section 2: grip force reflects pad wear
    assert all(fits_budget(f) for f in answer.fix_sets().values())
    assert answer.ranked == worst_health(export).ranked


def test_missing_health_after_measured_and_unknown_health_refused(export: Path) -> None:
    table(
        export,
        "health",
        [dict(day=1, robot_id="r02", component="gripper", metric="grip_force", value=2.0)],
    )
    assert worst_health(export).ranked.actions[0].action.target.scope == "r02"
    table(
        export,
        "health",
        [dict(day=1, robot_id="r02", component="gripper", metric="unsupported_metric", value=2.0)],
    )
    with pytest.raises(ValueError, match="unknown health"):
        worst_health(export)


def test_duration_then_motion_then_id_never_select_success(export: Path) -> None:
    selected = dramatic_failures(export)
    assert [r["episode_id"] for r in selected] == ["e000002", "e000001", "e000003"]
    assert selected[0]["motion_m"] == 5.0


def test_repeated_video_targets_do_not_add_padding_to_any_fix_set(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def filmstrip(_reader: Any, _clip: Any, target: Path) -> None:
        Image.new("RGB", (16, 16)).save(target)

    monkeypatch.setattr("fleetlab.baselines.video.filmstrip", filmstrip)

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        payload = json.loads(args[3].split("\n", 1)[1])
        assert "health" not in json.dumps(payload)
        assert "motion_m" not in json.dumps(payload)
        assert len(kwargs["images"]) == len(payload["clips"])
        calls.append(payload)
        return AgentResult(
            ok=True,
            result={
                "nominations": [
                    {"episode_id": c["episode_id"], "index": 0, "rationale": "visually camera"}
                    for c in payload["clips"]
                ]
            },
            usage={},
            seconds=0.0,
        )

    answer = dramatic_videos(export, tmp_path / "images", caller=caller)
    assert len(calls) == 1
    assert len(answer.ranked.actions) == 10
    assert all(len(f.actions) == 1 for f in answer.fix_sets().values())
    assert "FORMAT PADDING ONLY" in answer.ranked.actions[1].rationale
    root = tmp_path / "answer"
    write_answer(root, answer)
    assert Analysis(root).fix_set() == answer.ranked
    for budget in SCORING_BUDGETS:
        fix = read_baseline_fix_set(root, budget)
        assert len(fix.actions) == 1 and fits_budget(fix)
    with pytest.raises(FileExistsError):
        write_answer(root, answer)


@pytest.mark.parametrize("error", ["auth", "rate_limit", "timeout", "invalid_output"])
def test_model_failure_reruns_only_lost_runtime_answer(export: Path, error: str) -> None:
    calls = []

    def failed(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(args)
        return AgentResult(ok=False, result=None, usage={}, seconds=0.0, error=error)

    with pytest.raises(BaselineModelError) as exc:
        no_data_ranker(export, caller=failed)
    assert len(calls) == (2 if error == "timeout" else 1)
    assert len(exc.value.attempts) == len(calls)


def test_model_repeated_or_bad_indices_fail_without_retry(export: Path) -> None:
    def repeated(*args: Any, **kwargs: Any) -> AgentResult:
        return AgentResult(
            ok=True,
            result={"choices": [{"index": 0, "rationale": "duplicate"} for _ in range(10)]},
            usage={},
            seconds=0.0,
        )

    with pytest.raises(ValueError, match="repeated"):
        no_data_ranker(export, caller=repeated)


def test_budget_reader_rejects_wrong_budget(export: Path, tmp_path: Path) -> None:
    root = tmp_path / "answer"
    write_answer(root, worst_health(export))
    path = root / "fix_sets/cost-4.json"
    raw = json.loads(path.read_text())
    raw["budget"]["amount"] = 8
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not match"):
        read_baseline_fix_set(root, Budget(kind="cost", amount=4))
    assert len(component_actions(export)) == 10


@pytest.mark.parametrize("error", ["auth", "rate_limit", "timeout", "invalid_output", "semantic"])
def test_command_records_missing_and_continues_only_ordinary_failures(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    from fleetlab.baselines.__main__ import main

    root = tmp_path / "answers"
    monkeypatch.setattr(
        "sys.argv",
        ["baselines", "--export", str(export), "--out", str(root), "--seed", "7", "--draws", "1"],
    )
    calls = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        if error == "semantic":
            raise ValueError("repeated indices")
        raise BaselineModelError(
            AgentResult(ok=False, result=None, usage={}, seconds=0, error=error)
        )

    def vision(*args: Any, **kwargs: Any) -> Any:
        calls.append("vision")
        answer = worst_health(export)
        answer.method = "dramatic_videos"
        return answer

    monkeypatch.setattr("fleetlab.baselines.__main__.no_data_ranker", fail)
    monkeypatch.setattr("fleetlab.baselines.__main__.dramatic_videos", vision)
    assert main() == 2
    record = json.loads((root / "manifest.json").read_text())
    assert record["missing"]["no_data"]["error"] == (
        "invalid_output" if error == "semantic" else error
    )
    assert "fixed_prior_fallback" in record["methods"] and "no_data" not in record["methods"]
    if error in {"auth", "rate_limit"}:
        assert record["stopped"]["error"] == error
        assert record["missing"]["dramatic_videos"]["error"] == "not_attempted"
        assert calls == []
    else:
        assert "stopped" not in record and calls == ["vision", "vision"]
        assert "dramatic_videos" in record["methods"]


def test_command_seed_defaults_to_chance_record_and_refuses_run_root(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.baselines.__main__ import main

    path = export / "log/description.json"
    raw = json.loads(path.read_text())
    raw["fleet"]["id"] = "w99"
    path.write_text(json.dumps(raw))
    (export.parent / "oracle/oracle.json").write_text(
        json.dumps({"world_id": "w99", "world_hash": "a" * 64, "seed": 197})
    )
    from test_scoring_gen3 import certify_fixture

    certify_fixture(export, "w99")
    root = tmp_path / "answers"
    monkeypatch.setattr(
        "sys.argv",
        ["baselines", "--export", str(export), "--out", str(root), "--draws", "1", "--offline"],
    )
    assert main() == 0
    assert json.loads((root / "manifest.json").read_text())["seed"] == 197
    assert json.loads((root / "random/draw-000/metadata.json").read_text())["seed"] == 197
    assert default_seed(export) == 197
    (export / "run.json").write_text("{}")
    refused = tmp_path / "refused"
    monkeypatch.setattr(
        "sys.argv", ["baselines", "--export", str(export), "--out", str(refused), "--seed", "42"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2 and not refused.exists()


def test_unknown_world_requires_explicit_seed(export: Path) -> None:
    assert default_seed(export) == 7  # w01 was generated with seed 7, not 1.
    (export.parent / "oracle/oracle.json").unlink()
    with pytest.raises(ValueError, match="pass --chance-table or --seed"):
        default_seed(export)


def test_health_uses_temporal_median_and_pools_both_cameras(export: Path) -> None:
    rows: list[dict[str, Any]] = []
    for robot, component, values in (
        ("r01", "scene_camera", [1, 1, 1000]),
        ("r01", "wrist_camera", [4, 4, 4]),
        ("r02", "scene_camera", [3, 3, 3]),
        ("r02", "wrist_camera", [2, 2, 2]),
    ):
        rows.extend(
            dict(day=i, robot_id=robot, component=component, metric="calibration_residual", value=v)
            for i, v in enumerate(values, 1)
        )
    table(export, "health", rows)
    answer = worst_health(export)
    assert [
        (r.action.target.scope, r.action.target.component) for r in answer.ranked.actions[:2]
    ] == [("r01", Component.wrist_camera), ("r02", Component.scene_camera)]
    assert all(r.action.verb.value == "recalibrate" for r in answer.ranked.actions[:2])
    scores = {(r["robot"], r["component"]): r["severity"] for r in answer.metadata["scores"]}
    assert scores["r01", "scene_camera"] == pytest.approx(-1.5 / 1.4826)
    assert scores["r01", "wrist_camera"] == pytest.approx(1.5 / 1.4826)


def test_health_benign_metrics_compete_and_component_uses_worst_metric(export: Path) -> None:
    rows: list[dict[str, Any]] = []
    for robot, force, cycles, cpu in [("r01", 10, 100, 0.1), ("r02", 50, 1000, 0.9)]:
        for comp, metric, value in [
            ("gripper", "grip_force", force),
            ("gripper", "cycle_count", cycles),
            ("controller", "cpu_load", cpu),
        ]:
            rows.extend(
                dict(day=d, robot_id=robot, component=comp, metric=metric, value=value)
                for d in (1, 2, 3)
            )
    table(export, "health", rows)
    answer = worst_health(export)
    # The gripper's bad force is not averaged away by its better cycle count.
    assert [
        (r.action.target.scope, r.action.target.component) for r in answer.ranked.actions[:3]
    ] == [("r01", Component.gripper), ("r02", Component.controller), ("r02", Component.gripper)]
    assert all(
        r["severity"] == pytest.approx(1 / 1.4826)
        for r in answer.metadata["scores"]
        if r["component"] == "gripper"
    )


def test_identical_health_ties_break_by_robot_then_component(export: Path) -> None:
    answer = worst_health(export)
    assert [
        (r.action.target.scope, str(r.action.target.component)) for r in answer.ranked.actions
    ] == [
        (robot, component)
        for robot in ("r01", "r02")
        for component in ("arm", "controller", "gripper", "scene_camera", "wrist_camera")
    ]
    assert all(r["severity"] == 0 for r in answer.metadata["scores"])


def test_real_filmstrip_decodes_both_cameras_and_covers_start_to_end(tmp_path: Path) -> None:
    fleet = write_toy_run(tmp_path / "run")
    root = tmp_path / "export"
    export_runs([tmp_path / "run"], root, fleet=fleet, task_classes=TASK_CLASSES)
    with LogReader(root) as reader:
        clip = reader.query("SELECT * FROM episodes WHERE episode_id = 'e000001'").to_pylist()[0]
        target = tmp_path / "strip.png"
        filmstrip(reader, clip, target)
    sheet = np.asarray(Image.open(target))
    assert sheet.shape == (1748, 768, 3)
    for ci, camera in enumerate(Camera):
        for i, step in [(0, 0), (15, 4)]:
            x, y = i % 4 * 192 + 8, 36 + (ci * 4 + i // 4) * 214 + 8
            assert float(sheet[y, x].mean()) == pytest.approx(frame_value(1, step, camera), abs=3)
    with pytest.raises(FileExistsError), LogReader(root) as reader:
        filmstrip(reader, clip, target)


@pytest.mark.parametrize("invalid", [None, "other_robot", "order"])
def test_vision_uses_each_budget_and_all_filmed_robot_or_site_verbs(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str | None,
) -> None:
    with LogReader(export) as reader:
        episodes = reader.query("SELECT * FROM episodes").to_pylist()
    # Six failures, with a second filmed robot, to test scope validation in the union.
    for i in (5, 6, 7):
        episode = f"e{i:06}"
        episodes.append(
            dict(episodes[0], episode_id=episode, robot_id="r02", site_id="s2", duration_s=8.0)
        )
        table(
            export,
            "steps",
            [
                dict(episode_id=episode, step=j, eef_pos=[j, 0.0, 0.0], frame_index=j)
                for j in range(2)
            ],
            episode,
        )
    table(export, "episodes", episodes)
    monkeypatch.setattr(
        "fleetlab.baselines.video.filmstrip",
        lambda reader, clip, target: Image.new("RGB", (16, 16)).save(target),
    )
    calls = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        payload = json.loads(args[3].split("\n", 1)[1])
        calls.append(payload)
        assert "simple baseline" not in args[2] + args[3]
        assert "repeated targets are expected" not in args[3]
        assert EXPECTED_SCORING_RULE in args[3]
        assert EXPECTED_SET_RULE in args[3]
        assert len(kwargs["images"]) == len(payload["clips"])
        catalogue = payload["catalogue"]
        assert {a["verb"] for a in catalogue if a["target"]["component"]} >= {
            "recalibrate",
            "clean",
            "replace",
            "service",
        }
        assert any(a["verb"] == "change_condition" for a in catalogue)
        assert any(a["verb"] == "revert" and a["target"]["config_key"] for a in catalogue)
        assert all(a["target"]["scope"] != "fleet" for a in catalogue)
        nominations = []
        for i, clip in enumerate(payload["clips"]):
            # First three clips repeat one site action; later clips add robot actions.
            scope = clip["site_id"] if i < 3 else clip["robot_id"]
            if invalid == "other_robot" and len(payload["clips"]) == 6:
                scope = "r02" if clip["robot_id"] == "r01" else "r01"
            action = next(
                a
                for a in catalogue
                if a["verb"] == "set"
                and a["target"]["scope"] == scope
                and a["target"]["config_key"] == "control.mode"
            )
            nominations.append(
                dict(
                    episode_id=clip["episode_id"],
                    index=action["index"],
                    rationale="visual hypothesis",
                )
            )
        if invalid == "order":
            nominations.reverse()
        return AgentResult(ok=True, result={"nominations": nominations}, usage={}, seconds=0)

    if invalid:
        with pytest.raises(
            ValueError, match="different robot or site" if invalid == "other_robot" else "in order"
        ):
            dramatic_videos(export, tmp_path / "images", caller=caller)
        return
    answer = dramatic_videos(export, tmp_path / "images", caller=caller)
    assert [len(c["clips"]) for c in calls] == [6]
    assert [a.target.scope for a in answer.fix_sets()["count-3"].actions] == ["s1"]
    assert [a.target.scope for a in answer.fix_sets()["count-6"].actions] == ["s1", "r02"]
    assert all(a.value == "absolute" for f in answer.fix_sets().values() for a in f.actions)


@pytest.mark.parametrize("seed", [True, -1, "41", None])
def test_seed_metadata_must_match_world_and_be_integer(export: Path, seed: Any) -> None:
    path = export.parent / "oracle/oracle.json"
    path.write_text(json.dumps({"world_id": "w01", "seed": seed}))
    with pytest.raises(ValueError, match="non-negative integer"):
        default_seed(export)
    path.write_text(json.dumps({"world_id": "w99", "seed": 41}))
    with pytest.raises(ValueError, match="different world"):
        default_seed(export)


def test_repeat_two_never_replaces_invalid_first(
    export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.baselines.__main__ import main

    root = tmp_path / "answers"
    monkeypatch.setattr(
        "sys.argv", ["baselines", "--export", str(export), "--out", str(root), "--draws", "1"]
    )
    calls = []

    def prior(*args: Any, **kwargs: Any) -> Any:
        calls.append("prior")
        if calls.count("prior") == 1:
            raise ValueError("invalid first answer")
        return no_data_ranker(export, caller=fake_rank)

    def vision(*args: Any, **kwargs: Any) -> Any:
        calls.append("vision")
        answer = worst_health(export)
        answer.method = "dramatic_videos"
        return answer

    monkeypatch.setattr("fleetlab.baselines.__main__.no_data_ranker", prior)
    monkeypatch.setattr("fleetlab.baselines.__main__.dramatic_videos", vision)
    assert main() == 2
    assert calls == ["prior", "vision", "prior", "vision"]
    record = json.loads((root / "manifest.json").read_text())
    assert "no_data" not in record["methods"] and not (root / "no_data").exists()
    repeats = record["model_repeats"]
    assert [r["scored"] for r in repeats] == [True, False]
    assert repeats[0]["missing"]["no_data"]["error"] == "invalid_output"
    assert repeats[1]["methods"]["no_data"] == "repeats/repeat-2/no_data/outbox/fix_set.json"
    assert (
        json.loads((root / "repeats/repeat-2/no_data/metadata.json").read_text())["scored"] is False
    )


@pytest.mark.parametrize("error", ["timeout", "overloaded", "cli_error"])
def test_lost_answer_rerun_preserves_conditions_and_attempts(error: str, tmp_path: Path) -> None:
    calls = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append((args, kwargs))
        if len(calls) == 1:
            return AgentResult(
                ok=False,
                result=None,
                usage={},
                seconds=600,
                error=error,
                error_detail="transport error: connection reset",
                run_dir="first",
            )
        return AgentResult(ok=True, result={"answer": 17}, usage={}, seconds=20, run_dir="rerun")

    raw, meta = model_call(
        "same prompt", images=[tmp_path / "image.png"], caller=caller, label="same repeat"
    )
    assert calls[0] == calls[1]
    assert raw == {"answer": 17} and meta["rerun"] is True and meta["seconds"] == 620
    assert [a["run_dir"] for a in meta["attempts"]] == ["first", "rerun"]
    assert [a["answer_present"] for a in meta["attempts"]] == [False, True]


@pytest.mark.parametrize(
    ("error", "result", "text", "detail"),
    [
        ("timeout", {"invalid": "answer"}, "", ""),
        ("timeout", None, "invalid prose", ""),
        ("malformed_output", None, "", "no answer"),
        ("cli_error", None, "", "executable not found"),
        ("auth", None, "", "connection reset"),
        ("rate_limit", None, "", "transport error"),
        ("tool_server", None, "", "failed to initialize"),  # no tools in a baseline call
    ],
)
def test_failed_answer_or_nontransport_error_is_never_rerun(
    error: str,
    result: Any,
    text: str,
    detail: str,
) -> None:
    calls = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(args)
        return AgentResult(
            ok=False,
            result=result,
            usage={},
            seconds=0,
            error=error,
            text=text,
            error_detail=detail,
        )

    with pytest.raises(BaselineModelError):
        model_call("prompt", caller=caller, label="test")
    assert len(calls) == 1


def test_recorded_rerun_has_only_one_remaining_start() -> None:
    original = AgentResult(
        ok=False, result=None, usage={}, seconds=600, error="timeout", run_dir="original"
    )
    calls = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        calls.append(args)
        return AgentResult(
            ok=False, result=None, usage={}, seconds=600, error="timeout", run_dir="rerun"
        )

    with pytest.raises(BaselineModelError) as exc:
        model_call("prompt", caller=caller, label="test", previous_attempt=original)
    assert len(calls) == 1
    assert [a["run_dir"] for a in exc.value.attempts] == ["original", "rerun"]
    original.error = "malformed_output"
    with pytest.raises(ValueError, match="lost runtime answer"):
        model_call("prompt", caller=caller, label="test", previous_attempt=original)
    assert len(calls) == 1


def test_video_prompt_hides_recorded_values_and_overrides(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fleetlab.baselines.video.filmstrip",
        lambda reader, clip, target: Image.new("RGB", (16, 16)).save(target),
    )
    prompts = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        prompts.append(args[3])
        payload = json.loads(args[3].split("\n", 1)[1])
        assert set(payload["system"]) == {"sites", "vocabulary", "costs"}
        for site in payload["system"]["sites"]:
            assert set(site) == {"id", "robots"}
            assert all(set(r) == {"id", "components"} for r in site["robots"])
        assert all("value" not in a for a in payload["catalogue"] if a["verb"] != "route_to_teleop")
        return AgentResult(
            ok=True,
            result={
                "nominations": [
                    dict(episode_id=c["episode_id"], index=0, rationale="visual")
                    for c in payload["clips"]
                ]
            },
            usage={},
            seconds=0,
        )

    dramatic_videos(export, tmp_path / "first-images", caller=caller)
    path = export / "log/description.json"
    raw = json.loads(path.read_text())
    for config in [
        raw["fleet"]["config"],
        *[s["config"] for s in raw["fleet"]["sites"]],
        *[r["config"] for s in raw["fleet"]["sites"] for r in s["robots"]],
    ]:
        config["values"]["control.mode"] = "secret-video-config-canary"
        config["version"] = 987654321
    for site in raw["fleet"]["sites"]:
        site["conditions"] = {key: "secret-video-condition-canary" for key in site["conditions"]}
    path.write_text(json.dumps(raw))
    dramatic_videos(export, tmp_path / "second-images", caller=caller)
    assert prompts[0] == prompts[1]
    assert "canary" not in prompts[1] and "987654321" not in prompts[1]


@pytest.mark.parametrize("clutter", [-0.001, 1.001, "invalid recorded condition"])
def test_video_rejects_unapplicable_recorded_condition(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clutter: Any,
) -> None:
    path = export / "log/description.json"
    raw = json.loads(path.read_text())
    raw["fleet"]["sites"][0]["conditions"]["clutter"] = clutter
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(
        "fleetlab.baselines.video.filmstrip",
        lambda reader, clip, target: Image.new("RGB", (16, 16)).save(target),
    )

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        payload = json.loads(args[3].split("\n", 1)[1])
        action = next(a for a in payload["catalogue"] if a["target"]["condition_key"] == "clutter")
        return AgentResult(
            ok=True,
            result={
                "nominations": [
                    dict(episode_id=c["episode_id"], index=action["index"], rationale="visual")
                    for c in payload["clips"]
                ]
            },
            usage={},
            seconds=0,
        )

    with pytest.raises(ValueError, match="cannot be applied"):
        dramatic_videos(export, tmp_path / "images", caller=caller)


@pytest.mark.parametrize("error", ["timeout", "invalid_output"])
def test_only_second_repeat_missing_exits_two_and_keeps_first(
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    from fleetlab.baselines.__main__ import main

    root = tmp_path / "answers"
    monkeypatch.setattr(
        "sys.argv", ["baselines", "--export", str(export), "--out", str(root), "--draws", "1"]
    )
    starts = []

    def caller(*args: Any, **kwargs: Any) -> AgentResult:
        starts.append(args)
        if len(starts) == 1:
            return fake_rank(*args, **kwargs)
        return AgentResult(ok=False, result=None, usage={}, seconds=0, error=error)

    monkeypatch.setattr(
        "fleetlab.baselines.__main__.no_data_ranker",
        lambda *args, **kwargs: no_data_ranker(export, caller=caller),
    )

    def vision(*args: Any, **kwargs: Any) -> Any:
        answer = worst_health(export)
        answer.method = "dramatic_videos"
        return answer

    monkeypatch.setattr("fleetlab.baselines.__main__.dramatic_videos", vision)
    assert main() == 2
    assert len(starts) == (3 if error == "timeout" else 2)
    record = json.loads((root / "manifest.json").read_text())
    assert record["missing"] == {} and "no_data" in record["methods"]
    assert (root / "no_data/outbox/fix_set.json").is_file()
    assert not (root / "repeats/repeat-2/no_data").exists()
    second = record["model_repeats"][1]
    assert second["missing"]["no_data"]["error"] == error
    assert second["comparison_available"] is False
    assert second["margin"] is None and second["ratio"] is None
