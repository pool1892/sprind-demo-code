"""Retained engineering learning on CPU fixtures: write, validate, refuse, freeze, mount, compare.

docs/design/loop.md, section 9; src/fleetlab/contracts/learned.py, src/fleetlab/loop/.
The source investigation runs on the skills' toy fleet (w02, a step at one
site from one day on), the later case on the log's toy fleet (w01). The agents are
scripted: they prove the mechanics, not that a model learns.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import skills_fleet
from log_toy import TASK_CLASSES, run_jobs, write_toy_run
from toolenv_fixed import fixed_digest

from fleetlab.agents import checks
from fleetlab.agents import run as runmod
from fleetlab.agents.audit import AuditContext, tree_digest
from fleetlab.agents.checks import GateError
from fleetlab.agents.run import AgentResult, ToolAccess, call_log
from fleetlab.agents.sandbox import BOX_LEARNED, ToolMounts, tool_server_command
from fleetlab.agents.sessions import AgentSpec, Analysis, EngineReply, ExperimentQueue, run_analysis
from fleetlab.agents.toolenv import ToolEnv, project_tool_env
from fleetlab.agents.toy import write_toy_skill
from fleetlab.contracts.learned import (
    LearnedSkillDraft,
    draft_problems,
    render_learned_skill,
)
from fleetlab.contracts.model import Fleet
from fleetlab.contracts.skills import load_skill
from fleetlab.contracts.tools import ArmResult as ToolArmResult
from fleetlab.contracts.tools import ExperimentResult
from fleetlab.log import export_runs
from fleetlab.log._files import read_canary
from fleetlab.loop.learned import (
    LEARNING_INSTRUCTIONS,
    FrozenLibrary,
    LearnedRefused,
    Source,
    freeze,
    stored_drafts,
    with_learning,
)
from fleetlab.loop.reuse import ArmSetup, Budget, LaterCase, ReuseError, compare
from fleetlab.toolserver.files import LEARNED_FILE, append_line
from fleetlab.toolserver.server import Paths, ToolBox

SKILL: dict[str, Any] = {
    "name": "site-step-after-control-timing",
    "description": (
        "Find a site whose robots all lose success from one day on, and tell a control "
        "timing setting apart from other causes shared by the site."
    ),
    "applies_when": (
        "Every robot of one site drops together from one day on while the other sites "
        "hold, and no single robot stands out."
    ),
    "parameters": [
        {"name": "site_id", "meaning": "the site whose robots dropped together"},
        {"name": "from_day", "meaning": "the first day of the drop"},
    ],
    "evidence": [
        {
            "claim": "success at {site_id} before and after day {from_day}, against the rest",
            "sql": (
                "SELECT site_id = $site_id AS at_site, day >= $from_day AS after_step, "
                "count(*) AS n, avg(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS rate "
                "FROM episodes WHERE experiment_id IS NULL GROUP BY 1, 2 ORDER BY 1, 2"
            ),
            "result": "at {site_id} the rate fell by about half from day {from_day}; "
            "the other site held",
        }
    ],
    "alternatives": [
        {
            "explanation": "a software update logged on the same day at the same site",
            "how_to_tell": "revert one of the two alone in an experiment; the log cannot "
            "separate entries made together",
        }
    ],
    "next_experiment": (
        "At {site_id}, two arms on the same robots and start states: as it is, and with the "
        "control timing setting reverted; 30 attempts each."
    ),
    "check_with": [{"name": "site_id", "value": "s2"}, {"name": "from_day", "value": "2"}],
}


def skill(**changes: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(SKILL)), **changes}


@dataclass(frozen=True)
class Worlds:
    base: Path
    source_run: Path
    source_export: Path
    source_fleet: Fleet
    later_run: Path
    later_export: Path
    later_after: Path  # the later export with e000002 and e000004 as the arms of x0001
    later_fleet: Fleet
    env: ToolEnv


@pytest.fixture(scope="module")
def worlds(tmp_path_factory: pytest.TempPathFactory) -> Worlds:
    base = tmp_path_factory.mktemp("learned")
    source_run, source_export = skills_fleet.write_fleet_export(base / "source")
    later = base / "later"
    later_fleet = write_toy_run(later / "run")
    export_runs([later / "run"], later / "export", fleet=later_fleet, task_classes=TASK_CLASSES)
    arms = []
    for job in run_jobs():
        if job.identity.episode_id in ("e000002", "e000004"):
            job = job.model_copy(
                update={"identity": job.identity.model_copy(update={"experiment_id": "x0001"})}
            )
        arms.append(job)
    write_toy_run(later / "run-after", arms)
    export_runs(
        [later / "run-after"], later / "after", fleet=later_fleet, task_classes=TASK_CLASSES
    )
    write_toy_skill(base / "skills")
    env = fixed_digest(project_tool_env(base / "env", base / "skills"))
    return Worlds(
        base,
        source_run,
        source_export,
        skills_fleet.fleet(),
        later / "run",
        later / "export",
        later / "after",
        later_fleet,
        env,
    )


def toolbox(analysis: Analysis, export: Path, env: ToolEnv, session: int = 1) -> ToolBox:
    return ToolBox(
        Paths(
            export=export,
            memory=analysis.memory,
            outbox=analysis.outbox,
            results=analysis.results,
            skills=env.skills,
            learned=analysis.learned,
        ),
        session=session,
    )


def receipt(result: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result.content[0]["text"])
    return data


def source_analysis(worlds: Worlds, root: Path, **changes: Any) -> Analysis:
    analysis = Analysis.create("w02", base=root / "source-analyses")
    box = toolbox(analysis, worlds.source_export, worlds.env)
    try:
        answer = receipt(box.call("write_learned_skill", skill(**changes)))
    finally:
        box.close()
    assert answer["accepted"], answer["problems"]
    return analysis


def source(worlds: Worlds, analysis: Analysis, label: str = "fleet A") -> Source:
    return Source(analysis.root, label, 1, (worlds.source_run,), worlds.source_fleet)


# The draft and its rendering.


def test_a_rendered_draft_is_a_skill_with_the_learned_fields() -> None:
    draft = LearnedSkillDraft.model_validate(skill(check_with=[]))
    assert draft_problems(draft) == []
    text = render_learned_skill(draft, "learned on fleet A, round 1")
    assert text.startswith("---\nname: site-step-after-control-timing\n")
    for section in ("Applies when", "Evidence", "Competing explanations", "Next experiment"):
        assert f"\n## {section}\n" in text
    assert "```sql\n-- helper: evidence-1\nSELECT site_id = $site_id" in text
    assert render_learned_skill(draft, "learned on fleet A, round 1") == text  # deterministic


def test_the_rendered_skill_loads_with_its_manifest(tmp_path: Path) -> None:
    draft = LearnedSkillDraft.model_validate(skill(check_with=[]))
    directory = tmp_path / draft.name
    directory.mkdir()
    (directory / "SKILL.md").write_text(render_learned_skill(draft, "learned on fleet A"))
    manifest = load_skill(directory).manifest
    assert manifest.origin == "learned"
    assert manifest.provenance == "learned on fleet A"
    assert manifest.applies_when == " ".join(SKILL["applies_when"].split())
    assert manifest.tools_used == ["query", "request_experiment"]
    assert manifest.inputs[0].startswith("site_id: ")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"applies_when": "When site s2 drops."}, "'s2' has the form of a site identifier"),
        (
            {"next_experiment": "Request it on r04 and r05."},
            "'r04' has the form of a robot identifier",
        ),
        (
            {"evidence": [{**SKILL["evidence"][0], "result": "see e000123"}]},
            "'e000123' has the form of a episode identifier",
        ),
        (
            {"evidence": [{**SKILL["evidence"][0], "sql": "SELECT s1.day FROM episodes s1"}]},
            "or rename the SQL alias",
        ),
        (
            {"evidence": [{**SKILL["evidence"][0], "sql": "SELECT $robot FROM episodes"}]},
            "parameters ['robot'] are not declared",
        ),
        ({"applies_when": "From {day} on."}, "placeholders ['day'] are not declared"),
        ({"description": " "}, "description: empty"),
        ({"check_with": [{"name": "robot", "value": "r01"}]}, "'robot' is not a declared"),
    ],
)
def test_draft_problems_explain_what_to_change(changes: dict[str, Any], message: str) -> None:
    problems = draft_problems(LearnedSkillDraft.model_validate(skill(**changes)))
    assert any(message in p for p in problems), problems


# The tool.


def test_write_learned_skill_keeps_the_draft_without_its_check_values(
    worlds: Worlds, tmp_path: Path
) -> None:
    analysis = Analysis.create("w02", base=tmp_path)
    box = toolbox(analysis, worlds.source_export, worlds.env)
    try:
        first = receipt(box.call("write_learned_skill", skill()))
        assert first["accepted"] and first["receipt_id"] == "learned-1"
        assert "## Provenance\n\nadded by the host" in first["skill_md"]
        updated = receipt(box.call("write_learned_skill", skill(description="Revised.")))
        assert updated["accepted"] and updated["receipt_id"] == "learned-2"
    finally:
        box.close()
    stored = stored_drafts(analysis.root)
    assert list(stored) == [SKILL["name"]]
    assert stored[SKILL["name"]].draft.description == "Revised."
    assert stored[SKILL["name"]].draft.check_with == []
    assert "s2" not in (analysis.outbox / LEARNED_FILE).read_text()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"applies_when": "Site s2 drops from day 2."}, "site identifier"),
        (
            {"evidence": [{**SKILL["evidence"][0], "sql": "DELETE FROM episodes"}]},
            "one SELECT",
        ),
        (
            {"evidence": [{**SKILL["evidence"][0], "sql": "SELECT nothing FROM episodes"}]},
            "failed on this fleet",
        ),
        ({"check_with": [{"name": "site_id", "value": "s2"}]}, "give a value of this fleet"),
        ({"name": "triage"}, "is a base skill"),
    ],
)
def test_write_learned_skill_refuses_with_helpful_problems(
    worlds: Worlds, tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    analysis = Analysis.create("w02", base=tmp_path)
    box = toolbox(analysis, worlds.source_export, worlds.env)
    try:
        answer = receipt(box.call("write_learned_skill", skill(**changes)))
    finally:
        box.close()
    assert not answer["accepted"]
    assert any(message in p for p in answer["problems"]), answer["problems"]
    assert not (analysis.outbox / LEARNED_FILE).exists()


def test_at_most_three_learned_skills_per_analysis(worlds: Worlds, tmp_path: Path) -> None:
    analysis = Analysis.create("w02", base=tmp_path)
    box = toolbox(analysis, worlds.source_export, worlds.env)
    try:
        for i in range(3):
            assert receipt(box.call("write_learned_skill", skill(name=f"lesson-{i}")))["accepted"]
        fourth = receipt(box.call("write_learned_skill", skill(name="lesson-3")))
        again = receipt(box.call("write_learned_skill", skill(name="lesson-0")))
    finally:
        box.close()
    assert not fourth["accepted"] and "at most 3 learned skills" in fourth["problems"][0]
    assert again["accepted"]


# The host: check and freeze.


def test_freeze_checks_and_digests_a_library(worlds: Worlds, tmp_path: Path) -> None:
    analysis = source_analysis(worlds, tmp_path)
    report = freeze([source(worlds, analysis)], tmp_path / "libraries", kept_on="2026-10-14")
    library = report.library
    assert library is not None and report.kept == [SKILL["name"]] and not report.refused
    assert library.root.name == f"learned-{library.digest[:16]}"
    assert tree_digest(library.skills) == library.digest
    assert [p.name for p in library.skills.iterdir()] == [SKILL["name"]]
    text = (library.skills / SKILL["name"] / "SKILL.md").read_text()
    assert "learned by the analysis agent on fleet A, round 1, session 1" in text
    assert "w02" not in text
    record = library.record
    assert record["skills"][0]["source"]["world_id"] == "w02"  # the host's record only
    again = freeze([source(worlds, analysis)], tmp_path / "libraries", kept_on="2026-10-14")
    assert again.library is not None and again.library.root == library.root
    os.chmod(library.skills / SKILL["name"] / "SKILL.md", 0o644)
    (library.skills / SKILL["name"] / "SKILL.md").write_text(text + "changed")
    with pytest.raises(ValueError, match="changed since"):
        FrozenLibrary.load(library.root)


def _plant(analysis: Analysis, **changes: Any) -> None:
    """A stored draft as an older or broken tool server might have kept it."""
    draft = LearnedSkillDraft.model_validate(skill(check_with=[], **changes))
    append_line(
        analysis.outbox / LEARNED_FILE,
        {"receipt_id": "learned-9", "session": 1, "draft": draft.model_dump()},
    )


@pytest.mark.parametrize("planted", ["canary", "kind", "trait", "identifier"])
def test_freeze_refuses_a_planted_truth_term_or_identifier(
    worlds: Worlds, tmp_path: Path, planted: str
) -> None:
    analysis = Analysis.create("w02", base=tmp_path)
    term = {
        "canary": str(read_canary(worlds.source_run)),
        "kind": skills_fleet.KINDS[0],
        "trait": skills_fleet.TRAIT,
        "identifier": "r02",
    }[planted]
    _plant(analysis, applies_when=f"When the records show {term} at one site.")
    with pytest.raises(LearnedRefused):
        freeze([source(worlds, analysis)], tmp_path / "libraries")
    report = freeze([source(worlds, analysis)], tmp_path / "libraries", drop_refused=True)
    assert report.library is None and list(report.refused) == [SKILL["name"]]
    assert not list((tmp_path / "libraries").iterdir())  # nothing half-written


def test_a_fleet_label_may_not_be_an_identifier(worlds: Worlds, tmp_path: Path) -> None:
    analysis = source_analysis(worlds, tmp_path)
    with pytest.raises(ValueError, match="fleet label"):
        freeze([source(worlds, analysis, label="w02")], tmp_path / "libraries")


def test_a_later_library_carries_skills_forward_and_replaces_by_name(
    worlds: Worlds, tmp_path: Path
) -> None:
    first = source_analysis(worlds, tmp_path)
    one = freeze([source(worlds, first)], tmp_path / "lib", kept_on="2026-10-14").library
    assert one is not None
    second = source_analysis(worlds, tmp_path, name="second-lesson")
    two = freeze(
        [source(worlds, second, "fleet B")], tmp_path / "lib", previous=one, kept_on="2026-10-15"
    )
    assert two.library is not None and two.carried == [SKILL["name"]]
    assert two.library.names == sorted([SKILL["name"], "second-lesson"])
    assert (two.library.skills / SKILL["name"] / "SKILL.md").read_bytes() == (
        one.skills / SKILL["name"] / "SKILL.md"
    ).read_bytes()
    third = source_analysis(worlds, tmp_path, description="Revised on fleet C.")
    three = freeze(
        [source(worlds, third, "fleet C")], tmp_path / "lib", previous=two.library,
        kept_on="2026-10-16",
    )  # fmt: skip
    assert three.library is not None and three.carried == ["second-lesson"]
    entry = next(s for s in three.library.record["skills"] if s["name"] == SKILL["name"])
    assert "fleet A" in entry["replaces"] and "fleet C" in entry["provenance"]


# Mounting.


@pytest.fixture
def library(worlds: Worlds, tmp_path: Path) -> FrozenLibrary:
    analysis = source_analysis(worlds, tmp_path)
    frozen = freeze([source(worlds, analysis)], tmp_path / "libraries", kept_on="2026-10-14")
    assert frozen.library is not None
    return frozen.library


def test_a_later_analysis_lists_the_library_beside_the_unchanged_base_skills(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary
) -> None:
    base_before = tree_digest(worlds.env.skills)  # type: ignore[arg-type]
    plain = Analysis.create("w01", base=tmp_path / "later", analysis_id="plain")
    given = Analysis.create(
        "w01", base=tmp_path / "later", analysis_id="given", learned=library.skills
    )
    assert plain.learned is None
    assert given.learned is not None and tree_digest(given.learned) == library.digest
    assert given.meta["learned"]["digest"] == library.digest
    listings = {}
    for analysis in (plain, given):
        box = toolbox(analysis, worlds.later_export, worlds.env)
        try:
            listings[analysis.analysis_id] = json.loads(
                box.call("list_skills", {}).content[0]["text"]
            )
            if analysis is given:
                text = box.call("read_skill", {"name": SKILL["name"]}).content[0]["text"]
                assert "## Provenance\n\nlearned by the analysis agent on fleet A" in text
        finally:
            box.close()
    assert [m["name"] for m in listings["plain"]] == ["triage"]
    assert listings["given"][0] == listings["plain"][0]  # the base skill, byte for byte the same
    learned = listings["given"][1]
    assert learned["name"] == SKILL["name"] and learned["origin"] == "learned"
    assert learned["provenance"].startswith("learned by the analysis agent on fleet A, round 1")
    assert "origin" not in listings["plain"][0]
    assert tree_digest(worlds.env.skills) == base_before  # type: ignore[arg-type]
    access = given.tool_access(worlds.env, worlds.later_export, 1)
    assert access.learned == given.learned
    mounts = ToolMounts(
        env=worlds.env,
        export=worlds.later_export,
        memory=given.memory,
        outbox=given.outbox,
        results=given.results,
        log_dir=tmp_path,
        learned=access.learned,
    )
    command = tool_server_command(mounts, session=1, enabled=None, max_experiment_episodes=None)
    assert command[command.index("--learned") + 1] == BOX_LEARNED
    assert ("ro-bind", given.learned, BOX_LEARNED) in mounts.pairs()


def test_a_learned_skill_cannot_shadow_a_base_skill(worlds: Worlds, tmp_path: Path) -> None:
    learned = tmp_path / "learned"
    shutil.copytree(worlds.env.skills, learned)  # type: ignore[arg-type]
    paths = Paths(
        export=worlds.later_export,
        memory=tmp_path,
        outbox=tmp_path,
        results=tmp_path,
        skills=worlds.env.skills,
        learned=learned,
    )
    box = ToolBox(paths)
    try:
        listed = json.loads(box.call("list_skills", {}).content[0]["text"])
    finally:
        box.close()
    assert [m["name"] for m in listed] == ["triage"]


# The launch audit of a session that mounts a library, and of an outbox with a draft.


class ReachedLaunch(Exception):
    pass


@pytest.fixture
def launch(
    worlds: Worlds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., ToolAccess]:
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", str(tmp_path / "agent-root"))
    for name in ("source", "python", "site"):
        (tmp_path / name).mkdir()
    (tmp_path / "source" / "module.py").write_text("value = 1\n")
    env = ToolEnv(tmp_path / "python", tmp_path / "site", tmp_path / "source", worlds.env.skills)

    def prepare(*args: Any, **kwargs: Any) -> runmod.PreparedCall:
        box = runmod.box_dir(args[4])
        (box / "cwd").mkdir(parents=True)
        (box / "toolbox").mkdir()
        return runmod.PreparedCall([], {}, args[3], args[4], None, 0)

    def execute(*args: Any, **kwargs: Any) -> None:
        assert json.loads((args[3] / "audit.json").read_text())["passed"]
        raise ReachedLaunch

    monkeypatch.setattr(runmod, "prepare_call", prepare)
    monkeypatch.setattr(runmod, "_execute", execute)
    monkeypatch.setattr(checks, "require_verified", lambda *args: {"passed": True})

    def make(learned: Path | None = None) -> ToolAccess:
        analysis = Analysis.create(
            "w01", base=tmp_path / "analyses", analysis_id=None, learned=learned
        )
        return analysis.tool_access(
            env,
            worlds.later_export,
            1,
            audit=AuditContext((worlds.later_run,), worlds.later_fleet),
        )

    return make


def test_the_launch_audit_passes_a_clean_library_and_records_its_digest(
    launch: Callable[..., ToolAccess], library: FrozenLibrary
) -> None:
    access = launch(library.skills)
    with pytest.raises(ReachedLaunch):
        runmod.run_agent("claude", "model", "System", "Task", tools=access)
    runs = sorted((access.memory.parent.parent.parent / "agent-root").rglob("audit.json"))
    receipt_file = json.loads(runs[-1].read_text())
    assert "learned" in receipt_file["manifest"]


@pytest.mark.parametrize("contamination", ["canary", "identifier", "extra file"])
def test_the_launch_audit_refuses_a_contaminated_library(
    worlds: Worlds, launch: Callable[..., ToolAccess], library: FrozenLibrary, contamination: str
) -> None:
    access = launch(library.skills)
    assert access.learned is not None
    path = access.learned / SKILL["name"] / "SKILL.md"
    if contamination == "extra file":
        (access.learned / SKILL["name"] / "notes.txt").write_text("ordinary")
    else:
        token = str(read_canary(worlds.later_run)) if contamination == "canary" else "r02"
        os.chmod(path, 0o644)
        path.write_text(path.read_text() + f"\nSee {token}.\n")
    with pytest.raises(GateError, match="learned-library audit"):
        runmod.run_agent("claude", "model", "System", "Task", tools=access)


@pytest.mark.parametrize(
    ("prose", "passes"),
    [
        ("Look at the camera wobble and the mount tolerance mm when the site drops.", True),
        ("Look for toy_camera_wobble when the site drops.", False),
    ],
)
def test_a_draft_in_the_outbox_is_agent_prose_at_the_next_launch(
    launch: Callable[..., ToolAccess], prose: str, passes: bool
) -> None:
    access = launch()
    draft = LearnedSkillDraft.model_validate(skill(check_with=[], applies_when=prose))
    append_line(
        access.outbox / LEARNED_FILE,
        {"receipt_id": "learned-1", "session": 1, "draft": draft.model_dump()},
    )
    if passes:
        with pytest.raises(ReachedLaunch):
            runmod.run_agent("claude", "model", "System", "Task", tools=access)
    else:
        with pytest.raises(GateError, match="outbox audit"):
            runmod.run_agent("claude", "model", "System", "Task", tools=access)


def test_the_learning_texts_pass_the_strict_audit(worlds: Worlds, tmp_path: Path) -> None:
    from fleetlab.agents.checks import tool_texts
    from fleetlab.log import audit_export

    draft = LearnedSkillDraft.model_validate(skill(check_with=[]))
    texts = {
        "learning": LEARNING_INSTRUCTIONS,
        "tool": tool_texts()["tool write_learned_skill"],
        "template": render_learned_skill(draft, "learned on fleet A, round 1"),
    }
    (tmp_path / "empty").mkdir()
    for runs, fleet in (
        ((worlds.source_run,), worlds.source_fleet),
        ((worlds.later_run,), worlds.later_fleet),
    ):
        report = audit_export(tmp_path / "empty", runs, fleet=fleet, texts=texts)
        assert report.clean, report.summary()
    assert with_learning("Task.").endswith(LEARNING_INSTRUCTIONS + "\n")


# The reuse comparison with a scripted agent.


class OneExperimentEngine:
    """Answers x0001 with two one-attempt arms, from the export where they carry its id."""

    def __init__(self, export: Path) -> None:
        self.export = export
        self.queues: list[ExperimentQueue] = []

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        self.queues.append(queue)
        results = [
            ExperimentResult(
                experiment_id=q.experiment_id,
                status="done",
                arms=[
                    ToolArmResult(arm_index=0, episode_ids=["e000002"]),
                    ToolArmResult(arm_index=1, episode_ids=["e000004"]),
                ],
                message="done",
            )
            for q in queue.experiments
        ]
        return EngineReply(results=results, export=self.export)


REQUEST = {
    "arms": [
        {"actions": []},
        {
            "actions": [
                {
                    "verb": "recalibrate",
                    "target": {
                        "scope": "r02",
                        "component": "scene_camera",
                        "config_key": None,
                        "condition_key": None,
                    },
                    "value": None,
                }
            ]
        },
    ],
    "robots": ["r02"],
    "episodes_per_arm": 1,
    "paired_starts": True,
    "power_note": "one attempt per arm: a check of the mechanics",
}
DIAGNOSIS = {"causes": [], "mode_summaries": ["a step at one site"], "warnings": [], "retrain": []}


def scripted_agent(worlds: Worlds) -> Callable[..., AgentResult]:
    """With a learned skill it reads it, runs its query and answers at once; without one it
    triages more widely and asks for an experiment first."""

    def agent(
        *args: Any, tools: ToolAccess, run_root: Path | None = None, **kw: Any
    ) -> AgentResult:
        run_dir = (run_root or worlds.base) / f"call-{tools.session}-{os.urandom(4).hex()}"
        log = call_log(run_dir)
        log.parent.mkdir(parents=True)
        box = ToolBox(
            Paths(
                export=tools.export,
                memory=tools.memory,
                outbox=tools.outbox,
                results=tools.results,
                skills=tools.env.skills,
                log=log,
                learned=tools.learned,
            ),
            session=tools.session,
        )
        try:
            listed = json.loads(box.call("list_skills", {}).content[0]["text"])
            learned = [m for m in listed if m.get("origin") == "learned"]
            if learned:
                text = box.call("read_skill", {"name": learned[0]["name"]}).content[0]["text"]
                sql = text.split("```sql\n", 1)[1].split("```", 1)[0]
                box.call(
                    "query", {"sql": sql.replace("$site_id", "'s2'").replace("$from_day", "1")}
                )
                box.call("submit_diagnosis", DIAGNOSIS)
                report = {"status": "done", "summary": "the learned skill applied"}
            elif tools.session == 1:
                for column in ("site_id", "robot_id", "day", "task_text"):
                    box.call(
                        "query", {"sql": f"SELECT {column}, count(*) FROM episodes GROUP BY 1"}
                    )
                box.call("request_experiment", REQUEST)
                report = {"status": "waiting", "summary": "asked"}
            else:
                box.call("experiment_results", {})
                box.call("submit_diagnosis", DIAGNOSIS)
                report = {"status": "done", "summary": "submitted"}
        finally:
            box.close()
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return AgentResult(
            ok=True,
            result=report,
            usage={"input_tokens": 100 * len(calls), "output_tokens": 10 * len(calls)},
            seconds=0.5 * len(calls),
            tool_log=calls,
            run_dir=str(run_dir),
        )

    return agent


def later_case(worlds: Worlds) -> LaterCase:
    return LaterCase("w01", worlds.later_export, "SYSTEM", with_learning("TASK"))


def test_the_reuse_comparison_runs_two_fresh_identical_arms(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", scripted_agent(worlds))
    engines = {arm: OneExperimentEngine(worlds.later_after) for arm in ("without", "with")}
    scores = {"without": 0.5, "with": 0.5}

    def score(analysis: Analysis, extra: Any) -> dict[str, Any]:
        arm = "with" if analysis.learned is not None else "without"
        return {"score": scores[arm], "diagnosis": analysis.diagnosis() is not None}

    comparison = compare(
        later_case(worlds),
        AgentSpec("claude", "opus"),
        worlds.env,
        library,
        tmp_path / "reuse",
        budget=Budget(max_sessions=3),
        engine_for=lambda arm: engines[arm],
        score=score,
    )
    without, with_ = comparison.arms["without"], comparison.arms["with"]
    # Identical inputs except the library, and fresh analyses that share nothing.
    for key in ("export", "base_skills", "tool_env", "agent", "budget", "task_sha256"):
        assert without.inputs[key] == with_.inputs[key], key
    assert without.inputs["library"] is None and without.inputs["mounted_library"] is None
    assert with_.inputs["library"] == with_.inputs["mounted_library"] == library.digest
    assert without.inputs["export"] == tree_digest(worlds.later_export)
    assert Path(without.analysis) != Path(with_.analysis)
    assert not stored_drafts(Path(with_.analysis)) and not stored_drafts(Path(without.analysis))
    # What each spent.
    assert without.costs["sessions"] == 2 and with_.costs["sessions"] == 1
    assert without.costs["experiments"]["requested"] == 1
    assert without.costs["experiments"]["attempts_simulated"] == 2
    assert with_.costs["experiments"]["requested"] == 0
    assert with_.costs["learned_skill_reads"] == {SKILL["name"]: 1}
    assert without.costs["learned_skill_reads"] == {}
    assert with_.costs["by_tool"]["read_skill"] == 1
    assert without.costs["tool_calls"] == without.costs["logged_calls"] == 9
    assert with_.costs["tool_calls"] == 4
    assert len(engines["without"].queues) == 1 and not engines["with"].queues
    differences = comparison.differences
    assert differences["sessions"]["with_minus_without"] == -1
    assert differences["attempts_simulated"]["with_minus_without"] == -2
    assert differences["score"]["with_minus_without"] == 0
    assert comparison.negative_transfer == []
    written = json.loads((tmp_path / "reuse" / "comparison.json").read_text())
    assert written["arms"]["with"]["inputs"]["library"] == library.digest
    assert "| sessions | 2.0 | 1.0 | -1.0 |" in (tmp_path / "reuse" / "comparison.md").read_text()
    inputs = json.loads((tmp_path / "reuse" / "inputs.json").read_text())
    assert set(inputs) == {"without", "with"}


def test_negative_transfer_is_reported(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", scripted_agent(worlds))
    comparison = compare(
        later_case(worlds),
        AgentSpec("claude", "opus"),
        worlds.env,
        library,
        tmp_path / "reuse",
        engine_for=lambda arm: OneExperimentEngine(worlds.later_after),
        score=lambda analysis, extra: {"score": 0.4 if analysis.learned is not None else 0.6},
    )
    assert any("scored" in note for note in comparison.negative_transfer)


def test_the_harness_refuses_arms_that_differ_or_an_export_that_changes(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary
) -> None:
    export = tmp_path / "export"
    shutil.copytree(worlds.later_export, export)
    case = replace(later_case(worlds), export=export)

    def runner(setup: ArmSetup) -> dict[str, Any]:
        (export / "log" / "extra.txt").write_text(setup.arm)
        return {}

    with pytest.raises(ReuseError, match="export changed"):
        compare(
            case, AgentSpec("claude", "opus"), worlds.env, library, tmp_path / "a", runner=runner
        )
    with pytest.raises(FileExistsError):
        compare(case, AgentSpec("claude", "opus"), worlds.env, library, tmp_path / "a")


def test_the_with_arm_gets_only_the_library_from_its_source(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary
) -> None:
    from fleetlab.loop.reuse import prepare_arms

    arms = prepare_arms(
        later_case(worlds), AgentSpec("claude", "opus"), worlds.env, library, tmp_path, Budget()
    )
    with_ = arms["with"][0]
    assert sorted(p.name for p in with_.root.iterdir()) == sorted(
        ["analysis.json", "learned", "memory", "outbox", "queues", "results", "sessions"]
    )
    for part in ("memory", "outbox", "results", "queues", "sessions"):
        assert not any((with_.root / part).iterdir())


def test_run_analysis_with_a_library_mounts_it_in_every_session(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path | None] = []

    def agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        seen.append(tools.learned)
        box = ToolBox(
            Paths(
                export=tools.export,
                memory=tools.memory,
                outbox=tools.outbox,
                results=tools.results,
            ),
            session=tools.session,
        )
        try:
            if tools.session == 1:
                box.call("request_experiment", REQUEST)
                report = {"status": "waiting", "summary": ""}
            else:
                box.call("submit_diagnosis", DIAGNOSIS)
                report = {"status": "done", "summary": ""}
        finally:
            box.close()
        return AgentResult(ok=True, result=report, usage={}, seconds=0.0)

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", agent)
    analysis = Analysis.create("w01", base=tmp_path, learned=library.skills)
    record = run_analysis(
        analysis,
        worlds.later_export,
        AgentSpec("claude", "opus"),
        "S",
        "T",
        worlds.env,
        OneExperimentEngine(worlds.later_after),
    )
    assert [s.state for s in record.sessions] == ["waiting", "done"]
    assert seen == [analysis.learned, analysis.learned] and analysis.learned is not None


bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")


@bwrap
def test_the_library_reaches_the_tool_server_in_its_sandbox(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary
) -> None:
    import subprocess

    analysis = Analysis.create("w01", base=tmp_path / "analyses", learned=library.skills)
    (tmp_path / "log").mkdir()
    mounts = ToolMounts(
        env=worlds.env,
        export=worlds.later_export,
        memory=analysis.memory,
        outbox=analysis.outbox,
        results=analysis.results,
        log_dir=tmp_path / "log",
        learned=analysis.learned,
    )
    command = tool_server_command(
        mounts, session=1, enabled=None, max_experiment_episodes=None, nested=False
    )
    later_skill = skill(
        name="later-lesson",
        check_with=[{"name": "site_id", "value": "s2"}, {"name": "from_day", "value": "1"}],
    )
    messages: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "list_skills", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "read_skill", "arguments": {"name": SKILL["name"]}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "write_learned_skill", "arguments": later_skill}},
    ]  # fmt: skip
    done = subprocess.run(
        command,
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin"},
    )
    answers = {a["id"]: a for a in map(json.loads, done.stdout.splitlines()) if "id" in a}
    listed = json.loads(answers[2]["result"]["content"][0]["text"])
    assert [m["name"] for m in listed] == ["triage", SKILL["name"]], done.stderr
    assert "learned by the analysis agent" in answers[3]["result"]["content"][0]["text"]
    assert json.loads(answers[4]["result"]["content"][0]["text"])["accepted"]
    assert list(stored_drafts(analysis.root)) == ["later-lesson"]


# Repairs after a review of 25565f6: costs over every start, ids per comparison, resume.


def _start(root: Path, name: str, calls: int, usage: dict[str, Any], ok: bool) -> dict[str, Any]:
    log = call_log(root / name)
    log.parent.mkdir(parents=True)
    log.write_text(
        "".join(json.dumps({"seq": i + 1, "tool": "query"}) + "\n" for i in range(calls))
    )
    return {
        "ok": ok,
        "error": None if ok else "timeout",
        "detail": "",
        "run_dir": str(root / name),
        "seconds": 10.0 if ok else 4.0,
        "usage": usage,
        "runtime_meta": {},
        "answer_present": ok,
    }


def test_arm_costs_sum_every_start_of_a_retried_session(tmp_path: Path) -> None:
    """A lost first start and its replacement: the record's own usage is the second start's
    only (200/40 tokens, $0.2); the arm's cost is both (300/60, $0.3, 3 + 4 tool calls)."""
    from dataclasses import asdict

    from fleetlab.agents.sessions import SessionRecord
    from fleetlab.loop.reuse import arm_costs

    analysis = Analysis.create("w01", base=tmp_path)
    usage = [
        {"input_tokens": 100, "output_tokens": 20, "cost_usd_list_price": 0.1},
        {"input_tokens": 200, "output_tokens": 40, "cost_usd_list_price": 0.2},
    ]
    first = _start(tmp_path, "start-1", 3, usage[0], ok=False)
    second = _start(tmp_path, "start-2", 4, usage[1], ok=True)
    record = SessionRecord(
        number=1,
        state="done",
        export=str(tmp_path),
        run_dir=second["run_dir"],
        ok=True,
        error=None,
        seconds=14.0,
        usage=second["usage"],
        queued=[],
        submitted=["diagnosis-1"],
        report={"status": "done", "summary": ""},
        tool_calls=4,
        attempts=[first, second],
        rerun=True,
    )
    (analysis.root / "sessions" / "session-01.json").write_text(json.dumps(asdict(record)))
    costs = arm_costs(analysis, "claude-opus-5-5")
    assert costs["tokens"]["input_tokens"] == 300 and costs["tokens"]["output_tokens"] == 60
    assert costs["cost_usd_list_price"] == pytest.approx(0.3)
    assert costs["tool_calls"] == 7 and costs["logged_calls"] == 7
    assert costs["starts"] == 2 and costs["failed_starts"] == 1
    assert costs["model_seconds"] == 14.0


class KeyedEngine:
    """Keeps a plan per world, analysis and session, as the queue engine does, and refuses a
    saved plan that another queue wrote; can fail once after saving its plan."""

    def __init__(self, world_dir: Path, export: Path, *, fail_once: bool = False) -> None:
        self.world_dir, self.export, self.fail_once = world_dir, export, fail_once
        self.calls: list[str] = []

    def run_queue(self, queue_file: Path, queue: ExperimentQueue) -> EngineReply:
        import hashlib

        self.calls.append(queue.analysis_id)
        plan = (
            self.world_dir
            / "experiments"
            / queue.analysis_id
            / f"session-{queue.session:02d}.plan.json"
        )
        digest = hashlib.sha256(queue.model_dump_json().encode()).hexdigest()
        if plan.is_file():
            if json.loads(plan.read_text())["queue_sha256"] != digest:
                raise ValueError("saved experiment plan does not match this queue")
        else:
            plan.parent.mkdir(parents=True, exist_ok=True)
            plan.write_text(json.dumps({"queue_sha256": digest}))
        if self.fail_once:
            self.fail_once = False
            raise KeyboardInterrupt
        return OneExperimentEngine(self.export).run_queue(queue_file, queue)


def test_two_comparisons_on_one_world_keep_their_own_analysis_ids(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", scripted_agent(worlds))
    engine = KeyedEngine(tmp_path / "world", worlds.later_after)
    ids = []
    for name in ("first", "second"):
        comparison = compare(
            later_case(worlds), AgentSpec("claude", "opus"), worlds.env, library,
            tmp_path / name, engine_for=lambda arm: engine,
        )  # fmt: skip
        state = json.loads((tmp_path / name / "reuse.json").read_text())
        ids.append(state["analysis_ids"])
        assert Path(comparison.arms["without"].analysis).name == state["analysis_ids"]["without"]
        assert comparison.arms["without"].costs["experiments"]["attempts_simulated"] == 2
    assert ids[0]["without"] != ids[1]["without"] and ids[0]["with"] != ids[1]["with"]
    assert engine.calls == [ids[0]["without"], ids[1]["without"]]


def test_an_interrupted_comparison_resumes_with_the_same_ids_and_plan(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With tuple-bearing inputs (the audit's experiment days, the budget's tools), which
    inputs.json stores as lists: an unchanged case resumes, a changed budget does not."""
    from fleetlab.toolserver.server import tool_names

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", scripted_agent(worlds))
    engine = KeyedEngine(tmp_path / "world", worlds.later_after, fail_once=True)
    out = tmp_path / "reuse"
    case = replace(
        later_case(worlds),
        audit=AuditContext((worlds.later_run,), worlds.later_fleet, None, (0, 2)),
    )
    budget = Budget(max_sessions=3, tools=tool_names())
    assert isinstance(budget.tools, tuple) and case.audit is not None
    arguments = (case, AgentSpec("claude", "opus"), worlds.env, library, out)
    with pytest.raises(KeyboardInterrupt):
        compare(*arguments, budget=budget, engine_for=lambda arm: engine)
    state = json.loads((out / "reuse.json").read_text())
    without = Analysis(out / "analyses" / state["analysis_ids"]["without"])
    assert without.pending() and not (out / "arms" / "without.json").exists()
    recorded = json.loads((out / "inputs.json").read_text())
    assert recorded["without"]["audit"]["experiment_days"] == [0, 2]
    with pytest.raises(FileExistsError):
        compare(*arguments, budget=budget, engine_for=lambda arm: engine)
    changed = replace(budget, max_sessions=4)
    with pytest.raises(ReuseError, match="changed"):
        compare(*arguments, budget=changed, engine_for=lambda arm: engine, resume=True)
    comparison = compare(*arguments, budget=budget, engine_for=lambda arm: engine, resume=True)
    assert json.loads((out / "reuse.json").read_text()) == state
    assert engine.calls == [state["analysis_ids"]["without"]] * 2  # the same saved plan, reused
    arm = comparison.arms["without"]
    assert arm.runner["resumed"] is True and arm.runner["final_state"] == "done"
    assert arm.costs["sessions"] == 2 and arm.costs["experiments"]["done"] == 1
    assert comparison.arms["with"].runner["resumed"] is False
    again = compare(*arguments, budget=budget, engine_for=lambda arm: engine, resume=True)
    assert engine.calls == [state["analysis_ids"]["without"]] * 2  # finished arms are not rerun
    assert again.differences == comparison.differences


def test_a_start_without_usage_makes_usage_deltas_unavailable(
    worlds: Worlds, tmp_path: Path, library: FrozenLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One arm's start has no usage record: its tokens, seconds and price are lower bounds,
    and no difference of them is reported; counts that do not depend on usage still are."""
    scripted = scripted_agent(worlds)

    def agent(*args: Any, tools: ToolAccess, **kwargs: Any) -> AgentResult:
        result = scripted(*args, tools=tools, **kwargs)
        if tools.learned is not None:
            result.usage = {}
        return result

    monkeypatch.setattr("fleetlab.agents.sessions.run_agent", agent)
    comparison = compare(
        later_case(worlds),
        AgentSpec("codex", "gpt-6.1-sol"),
        worlds.env,
        library,
        tmp_path / "reuse",
        engine_for=lambda arm: OneExperimentEngine(worlds.later_after),
    )
    with_, without = comparison.arms["with"].costs, comparison.arms["without"].costs
    assert without["usage_complete"] and without["usage_note"] is None
    assert not with_["usage_complete"] and with_["starts_without_usage"] == 1
    assert "1 of 1 model starts have no usage record" in with_["usage_note"]
    assert with_["api_equivalent_usd"]["status"] == "lower_bound"
    assert without["api_equivalent_usd"]["status"] == "bounded_api_equivalent"
    for key in ("input_tokens", "output_tokens", "model_seconds"):
        row = comparison.differences[key]
        assert row["with_minus_without"] is None and row["status"] == "unavailable", key
        assert "incomplete in ['with']" in row["reason"]
    assert comparison.differences["tool_calls"]["with_minus_without"] == -5
    assert comparison.differences["tool_calls"]["status"] == "complete"
    text = (tmp_path / "reuse" / "comparison.md").read_text()
    assert "| input_tokens | 900.0 | 0.0 | unavailable (incomplete in ['with']" in text
