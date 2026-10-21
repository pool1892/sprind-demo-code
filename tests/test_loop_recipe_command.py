"""The callable check of story B (lane P): the probe recipe a learned skill
carries, run by a command on named robots (`loop.probes.recipe_check`, the `--library` mode
of `python -m fleetlab.loop.probes`), on the probe lane's toy world with the remote link at
one site (`loop.probe_toy`; CPU, no LIBERO). The skill is frozen into a library the way the
host freezes one (`loop.learned.freeze`); the agent's side is the learned-skill tests'.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.learned import LearnedSkillDraft
from fleetlab.loop.learned import RECIPES_DIR, FrozenLibrary, Source, freeze
from fleetlab.loop.probe_toy import LINK_ROBOTS, write_link_toy_world
from fleetlab.loop.probes import (
    FIGURE_FILE,
    HOST_DIR,
    HOST_FILE,
    RESULT_FILE,
    RESULT_PAGE,
    main,
    recipe_check,
)
from fleetlab.loop.world import LoopWorld
from fleetlab.toolserver.files import LEARNED_FILE, append_line

CLEAN, LINKED = "r01", LINK_ROBOTS[1]  # r04: the link alone
SKILL = "command-lag-probe"
FIRST_MOVES = (
    "WITH s AS (SELECT e.robot_id, e.repeat, st.step, st.eef_pos[1] AS x, "
    "first_value(st.eef_pos[1]) OVER (PARTITION BY st.episode_id ORDER BY st.step) AS x_start "
    "FROM probe_steps AS st JOIN probe_episodes AS e USING (episode_id) "
    "WHERE e.probe_id = $probe AND e.robot_id IN ($robot, $reference)) "
    "SELECT robot_id, repeat, min(step) FILTER (WHERE abs(x - x_start) > 0.001) AS first_moved "
    "FROM s GROUP BY robot_id, repeat ORDER BY robot_id, repeat"
)
STEP: list[dict[str, Any]] = [
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 5,
     "period_steps": None},
    {"primitive": "step", "axis": "x", "amplitude_m": 0.03, "duration_steps": 15,
     "period_steps": None},
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 5,
     "period_steps": None},
]  # fmt: skip


def draft(**changes: Any) -> LearnedSkillDraft:
    data: dict[str, Any] = {
        "name": SKILL,
        "description": "Measure how long a robot's hand takes to follow a commanded step, "
        "against a robot elsewhere that does well.",
        "applies_when": "Robots at one place fail more often in motions that need timely "
        "corrections, and their motion records show the hand trailing its commands.",
        "parameters": [
            {"name": "probe", "meaning": "the id of the step probe below, once it has run"},
            {"name": "robot", "meaning": "a robot under suspicion"},
            {"name": "reference", "meaning": "a robot elsewhere that does well"},
        ],
        "evidence": [
            {
                "claim": "the first row in which the hand of {robot} and of {reference} moved",
                "sql": FIRST_MOVES,
                "result": "{robot} moved some rows after {reference} in every repetition",
            }
        ],
        "alternatives": [
            {
                "explanation": "a sluggish arm rather than commands that arrive late",
                "how_to_tell": "late commands shift the whole response; a weak arm starts on "
                "time and falls short",
            }
        ],
        "next_experiment": "Run the probe below on {robot} and {reference}, then the evidence "
        "query with its id.",
        "check_with": [],
        "probe": {
            "name": "step-lag-x",
            "purpose": "how long the hand of {robot} takes to follow a commanded step",
            "expectation": "a robot whose commands arrive at once moves in the next row",
            "robot_parameters": ["robot", "reference"],
            "repeats": 2,
            "segments": STEP,
        },
    }
    return LearnedSkillDraft.model_validate({**data, **changes})


def frozen(world: LoopWorld, root: Path, kept: LearnedSkillDraft) -> FrozenLibrary:
    """A library frozen from one analysis whose outbox holds `kept`, as the host freezes it."""
    analysis = Analysis.create(world.world_id, base=root / "analyses")
    record = {"receipt_id": "learned-1", "session": 1, "draft": kept.model_dump(mode="json")}
    append_line(analysis.outbox / LEARNED_FILE, record)
    source = Source(analysis.root, "the toy fleet", 1, tuple(world.runs), world.model)
    report = freeze([source], root / "libraries", kept_on="2026-10-19")
    assert report.library is not None, report.refused
    return report.library


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[LoopWorld, Any, FrozenLibrary]:
    root = tmp_path_factory.mktemp("recipe-toy")
    world, setup = write_link_toy_world(root / "world-root")
    return world, setup, frozen(world, root, draft())


def test_the_recipe_runs_on_named_robots_and_its_sql_reads_the_probe(
    toy: tuple[LoopWorld, Any, FrozenLibrary], tmp_path: Path
) -> None:
    world, setup, library = toy
    out = tmp_path / "check"
    result = recipe_check(world, setup, library.root, SKILL, [LINKED, CLEAN], out)
    assert result["outcome"]["status"] == "done" and result["probe_id"] == "p0001"
    assert result["robots"] == {"robot": LINKED, "reference": CLEAN}
    assert result["request"]["robots"] == [LINKED, CLEAN] and result["request"]["repeats"] == 2
    assert sorted((e["robot_id"], e["repeat"]) for e in result["episodes"]) == [
        (CLEAN, 0),
        (CLEAN, 1),
        (LINKED, 0),
        (LINKED, 1),
    ]
    lags = {
        (e["robot_id"], e["repeat"]): e["edges"][0]["first_movement_rows"]
        for e in result["episodes"]
    }
    assert lags[(CLEAN, 0)] == lags[(CLEAN, 1)] == 1
    assert lags[(LINKED, 0)] > 2 and lags[(LINKED, 1)] > 2
    (evidence,) = result["evidence"]
    assert evidence["status"] == "done", evidence
    assert evidence["parameters"] == {"probe": "p0001", "robot": LINKED, "reference": CLEAN}
    first = {(row[0], row[1]): row[2] for row in evidence["rows"]}
    assert first[(LINKED, 0)] > first[(CLEAN, 0)] and first[(LINKED, 1)] > first[(CLEAN, 1)]
    page = (out / RESULT_PAGE).read_text()
    assert f"# Probe recipe `{SKILL}` on {LINKED}, {CLEAN}" in page
    assert f"![commanded against measured]({FIGURE_FILE})" in page and "## Evidence" in page
    assert (out / FIGURE_FILE).stat().st_size > 10_000
    assert json.loads((out / RESULT_FILE).read_text())["evidence"] == result["evidence"]
    # What a person reads names the fleet by its public id; the world's id, the paths and the
    # probe runs stay under host/ (in the toy world the two ids happen to be the same).
    assert result["fleet_id"] == world.truth.fleet.id and "world_id" not in result
    assert f"of fleet `{world.truth.fleet.id}`" in page
    provenance = json.loads((out / HOST_DIR / HOST_FILE).read_text())
    assert provenance["world_id"] == world.world_id
    assert Path(provenance["probe_runs"]).is_relative_to(out / HOST_DIR)
    assert str(out / HOST_DIR) not in (out / RESULT_FILE).read_text()


def test_the_recipe_command_refuses_what_it_cannot_run(
    toy: tuple[LoopWorld, Any, FrozenLibrary], tmp_path: Path
) -> None:
    world, setup, library = toy
    with pytest.raises(ValueError, match="takes 2 robots"):
        recipe_check(world, setup, library.root, SKILL, [LINKED], tmp_path / "one")
    plain = frozen(world, tmp_path, draft(name="no-probe-here", probe=None))
    with pytest.raises(ValueError, match="carries no probe"):
        recipe_check(world, setup, plain.root, "no-probe-here", [LINKED, CLEAN], tmp_path / "two")
    copy = tmp_path / "copy" / library.root.name
    shutil.copytree(library.root, copy)
    tampered = copy / RECIPES_DIR / f"{SKILL}.json"
    os.chmod(tampered, 0o644)
    tampered.write_text(tampered.read_text().replace('"repeats": 2', '"repeats": 5'))
    with pytest.raises(ValueError, match="recipe of"):
        recipe_check(world, setup, tampered.parents[1], SKILL, [LINKED, CLEAN], tmp_path / "x")
    # A used output directory would answer a new request with an earlier run's measurements.
    used = tmp_path / "used"
    used.mkdir()
    (used / "index.md").write_text("an earlier check\n")
    with pytest.raises(ValueError, match="not empty"):
        recipe_check(world, setup, library.root, SKILL, [LINKED, CLEAN], used)


def test_the_recipe_runs_only_if_the_skill_passes_the_audit_against_this_fleet(
    toy: tuple[LoopWorld, Any, FrozenLibrary], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import audit

    world, setup, library = toy
    seen: list[tuple[Path, tuple[Path, ...], Any]] = []

    def refuse(skill: Path, runs: tuple[Path, ...], fleet: Any) -> list[str]:
        seen.append((skill, runs, fleet))
        return [f"{skill.name}: a term of this fleet"]

    monkeypatch.setattr(audit, "audit_learned_skill", refuse)
    out = tmp_path / "check"
    with pytest.raises(ValueError, match="may not be used on this fleet"):
        recipe_check(world, setup, library.root, SKILL, [LINKED, CLEAN], out)
    # Audited against the destination's own runs and its full fleet, before anything ran.
    (skill, runs, fleet) = seen[0]
    assert skill == library.skills / SKILL and runs == tuple(world.runs) and fleet == world.model
    assert not out.exists()


def test_the_command_line_needs_the_skill_and_the_robots(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--world", str(tmp_path), "--out", str(tmp_path / "o"), "--library", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["--world", str(tmp_path), "--out", str(tmp_path / "o")])


def test_the_plan_s_form_of_the_command_forwards_to_the_recipe_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.loop import __main__ as loop_main
    from fleetlab.loop import probes

    seen: list[list[str]] = []

    def forwarded(argv: list[str]) -> int:
        seen.append(list(argv))
        return 0

    monkeypatch.setattr(probes, "main", forwarded)
    recipe = tmp_path / "lib" / RECIPES_DIR / f"{SKILL}.json"
    world, out = str(tmp_path / "w"), str(tmp_path / "o")
    argv = ["check", str(recipe), "--world", world, "--robots", "r05, r01", "--out", out]
    assert loop_main.main([*argv, "--param", "day=3"]) == 0
    assert seen == [
        [
            "--library",
            str((tmp_path / "lib").resolve()),
            "--skill",
            SKILL,
            "--world",
            world,
            "--out",
            out,
            "--workers",
            "4",
            "--robot",
            "r05",
            "--robot",
            "r01",
            "--param",
            "day=3",
        ]
    ]
    with pytest.raises(SystemExit):
        loop_main.main(["check", str(tmp_path / "x.json"), "--world", world, "--robots", "r05",
                        "--out", out])  # fmt: skip


class _Built(Exception):
    """Raised by the stand-in tool environment: the engagement stops where it would build it."""


def test_an_engagement_with_probes_mounts_the_base_skill_in_its_default_environment(
    toy: tuple[LoopWorld, Any, FrozenLibrary], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.agents import toolenv
    from fleetlab.agents.sessions import AgentSpec
    from fleetlab.loop.driver import AttemptBudget, run_rounds
    from fleetlab.loop.probes import PROBE_SKILLS, probe_tool_env_args
    from fleetlab.skill_library import ANALYSIS_SKILLS, library_dir

    assert probe_tool_env_args(False) == {"role": "analysis"}
    args = probe_tool_env_args(True)
    assert args["skills"] == library_dir()
    assert list(args["skill_names"]) == [*ANALYSIS_SKILLS, *PROBE_SKILLS]
    mounted = toolenv.build_skills(tmp_path / "mount", library_dir(), args["skill_names"])
    assert sorted(p.name for p in mounted.iterdir()) == sorted([*ANALYSIS_SKILLS, *PROBE_SKILLS])
    assert (mounted / "instrument-step-response" / "SKILL.md").is_file()
    # run_rounds without an explicit environment builds the default one: with probes the
    # analysis skills and the base skill on step responses, without probes the analysis role.
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        calls.append(kwargs)
        raise _Built

    monkeypatch.setattr(toolenv, "build_tool_env", capture)
    world, setup, _ = toy
    for probes in (True, False):
        with pytest.raises(_Built):
            run_rounds(
                world,
                tmp_path / f"engagement-{probes}",
                agent=AgentSpec("codex", "scripted"),
                budget=AttemptBudget(attempts=40, per_arm=4),
                max_sessions=1,
                heavy=False,
                setup=setup,
                probes=probes,
            )
    assert calls == [probe_tool_env_args(True), {"role": "analysis"}]
