"""The skill library, version 1 (docs/design/contracts.md, section 13): format, tools, generality.

Every skill loads with `contracts.skills.load_skill`, names only tools of
`contracts.tools.TOOLS`, and passes `validate_skill_general` with a planted
truth vocabulary; a planted identifier or term makes it fail. Every SQL block
of every skill runs, as written, on both toy exports and stays within what
the tool server's `query` returns.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from skills_support import (
    MAX_ROWS,
    SKILLS,
    V1_SKILLS,
    Exports,
    example_ids,
    helper_path,
    helpers,
    planted_vocabulary,
    registry_kinds,
    sql_blocks,
)

from fleetlab.contracts.skills import SKILL_FILE, load_skill, validate_skill_general
from fleetlab.contracts.tools import TOOLS
from fleetlab.jobs import task_text
from fleetlab.log import LogReader
from fleetlab.skill_library import (
    ANALYSIS_SKILLS,
    JOB_SKILLS,
    KNOWLEDGE_BEGIN,
    KNOWLEDGE_END,
    SUPERVISOR_SKILLS,
    library_dir,
    skill_dirs,
    strip_fault_class_knowledge,
)

TOOL_NAMES = {tool.name for tool in TOOLS}
SQL_SKILLS = (
    "triage",
    "episode-cards",
    "failure-modes",
    "scope",
    "sensitivity-from-logs",
    "sensitivity-by-experiment",
    "rank-actions",
)


@pytest.mark.parametrize(
    "text",
    [
        KNOWLEDGE_BEGIN,
        KNOWLEDGE_END,
        f"{KNOWLEDGE_BEGIN}\n{KNOWLEDGE_BEGIN}\n{KNOWLEDGE_END}",
        f"prefix {KNOWLEDGE_BEGIN}\n{KNOWLEDGE_END}",
    ],
)
def test_ablation_refuses_malformed_passage_markers(text: str) -> None:
    with pytest.raises(ValueError, match="fault-class knowledge"):
        strip_fault_class_knowledge(text)


# Lane P's story-B skill (demo/instrument), which the demo merges beside the job skills; the
# roster admits it with or without that branch merged.
INSTRUMENT_SKILLS = ("instrument-step-response",)


def test_the_library_holds_exactly_the_version_1_and_the_demo_skills() -> None:
    found = sorted(p.parent.name for p in SKILLS.glob(f"*/{SKILL_FILE}"))
    assert set(V1_SKILLS + JOB_SKILLS) <= set(found)
    assert set(found) <= set(V1_SKILLS + JOB_SKILLS + INSTRUMENT_SKILLS)
    assert len(found) == len(set(found))
    assert [d.name for d in skill_dirs("jobs")] == list(JOB_SKILLS)
    assert JOB_SKILLS == task_text.JOB_SKILLS  # the role mounts what the task text names
    assert library_dir() == SKILLS
    assert skill_dirs("analysis") == [SKILLS / name for name in ANALYSIS_SKILLS]
    assert [d.name for d in skill_dirs("supervisor", "/x")] == list(SUPERVISOR_SKILLS)
    with pytest.raises(KeyError):
        skill_dirs("teleoperator")


@pytest.mark.parametrize("name", V1_SKILLS)
def test_every_skill_loads_and_its_manifest_is_complete(name: str) -> None:
    skill = load_skill(SKILLS / name)
    manifest = skill.manifest
    assert manifest.name == name
    # Individual procedures can be revised without changing the library's roster.
    assert manifest.version.strip()
    analysis_versions = {
        "triage": "2",
        "episode-cards": "4",
        "failure-modes": "3",
        "scope": "4",
        "sensitivity-from-logs": "8",
        "sensitivity-by-experiment": "5",
        "rank-actions": "6",
        "write-diagnosis": "7",
    }
    assert manifest.version == ("2" if name in SUPERVISOR_SKILLS else analysis_versions[name])
    assert manifest.description.strip() and manifest.when_to_use.strip()
    assert manifest.inputs and manifest.outputs
    assert all(item.strip() for item in manifest.inputs + manifest.outputs)
    assert len(skill.body.splitlines()) > 10, "a skill is a procedure, not a stub"


@pytest.mark.parametrize("name", JOB_SKILLS)
def test_every_job_skill_loads_and_works_through_the_tools(name: str) -> None:
    manifest = load_skill(SKILLS / name).manifest
    assert manifest.name == name and manifest.version.strip()
    assert manifest.tools_used and set(manifest.tools_used) <= TOOL_NAMES


@pytest.mark.parametrize("name", JOB_SKILLS)
def test_every_job_skill_is_general(name: str, skill_exports: Exports) -> None:
    assert validate_skill_general(SKILLS / name, planted_vocabulary(skill_exports)) == []


@pytest.mark.parametrize("name", V1_SKILLS)
def test_tools_used_name_existing_tools(name: str) -> None:
    manifest = load_skill(SKILLS / name).manifest
    assert set(manifest.tools_used) <= TOOL_NAMES
    if name in ANALYSIS_SKILLS:
        assert manifest.tools_used, "an analysis skill works through the tools"
    else:
        assert manifest.tools_used == [], "the supervisor has no tool server"


@pytest.mark.parametrize("name", SQL_SKILLS)
def test_a_skill_that_runs_sql_says_query_and_has_its_runner(name: str) -> None:
    manifest = load_skill(SKILLS / name).manifest
    assert "query" in manifest.tools_used
    assert sql_blocks(name), "a skill whose helper runs SQL has SQL blocks"
    assert helper_path(name).is_file()
    assert set(helpers(name).blocks()) == set(sql_blocks(name))


def test_skill_files_are_plain_text_and_helpers_only() -> None:
    for path in SKILLS.rglob("*"):
        if path.is_dir():
            assert path.name != "__pycache__", f"{path} must not exist (bytecode is mounted too)"
            continue
        assert not path.is_symlink()
        if path.parent == SKILLS:
            assert path.name in {"README.md", "ROUNDS.md"}
            continue
        assert path.name == SKILL_FILE or path == helper_path(path.parent.name), path
        path.read_text(encoding="utf-8")


def test_the_runner_is_the_same_in_every_helper() -> None:
    def runner(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return text[text.index("# --- runner") : text.index("# --- end runner")]

    reference = runner(helper_path("triage"))
    for name in SQL_SKILLS:
        assert runner(helper_path(name)) == reference, name


def test_the_planted_vocabulary_is_the_real_one(skill_exports: Exports) -> None:
    terms = planted_vocabulary(skill_exports)
    assert "toy_pad_slip" in terms and "toy_camera_wobble" in terms  # kinds of the runs
    assert "pad_batch_code" in terms and "mount_tolerance_mm" in terms  # trait names
    assert any(t.startswith("canary-") for t in terms)  # the runs' canary tokens
    assert "gripper_wear" in terms and "true_success" in terms and "89.5" in terms
    assert len(registry_kinds()) >= 10 and set(registry_kinds()) <= set(terms)
    assert "policy.sampling_steps" not in terms and "sampling_steps" not in terms


@pytest.mark.parametrize("name", V1_SKILLS)
def test_every_skill_is_general(name: str, skill_exports: Exports) -> None:
    problems = validate_skill_general(SKILLS / name, planted_vocabulary(skill_exports))
    assert problems == []


def test_the_readme_is_general_too(skill_exports: Exports, tmp_path: Path) -> None:
    # The README is not a skill, but it is read by people who write skills: same rule.
    skill = tmp_path / "readme"
    skill.mkdir()
    shutil.copy(SKILLS / "README.md", skill / "notes.md")
    shutil.copy(SKILLS / "triage" / SKILL_FILE, skill / SKILL_FILE)
    text = (skill / SKILL_FILE).read_text().replace("name: triage", "name: readme")
    (skill / SKILL_FILE).write_text(text)
    assert validate_skill_general(skill, planted_vocabulary(skill_exports)) == []
    readme = (SKILLS / "README.md").read_text()
    assert all(f"`{name}`" in readme for name in V1_SKILLS)


@pytest.mark.parametrize("planted", ["identifier", "term", "value"])
def test_the_generality_check_fails_on_a_planted_word(
    planted: str, skill_exports: Exports, tmp_path: Path
) -> None:
    copy = tmp_path / "triage"
    shutil.copytree(SKILLS / "triage", copy)
    terms = planted_vocabulary(skill_exports)
    assert validate_skill_general(copy, terms) == []
    word = {"identifier": example_ids()[2], "term": "toy_pad_slip", "value": "54.5"}[planted]
    with (copy / SKILL_FILE).open("a") as handle:
        handle.write(f"\nIn this world, look at {word} first.\n")
    problems = validate_skill_general(copy, terms)
    assert len(problems) == 1 and word in problems[0]


def test_every_identifier_kind_is_caught_in_a_helper(tmp_path: Path) -> None:
    copy = tmp_path / "triage"
    shutil.copytree(SKILLS / "triage", copy)
    helper = copy / helper_path("triage").name
    with helper.open("a") as handle:
        handle.writelines(f"# {identifier}\n" for identifier in example_ids())
    problems = validate_skill_general(copy)
    assert len(problems) == len(example_ids())


@pytest.mark.parametrize("which", ["fleet", "toy", "mini"])
@pytest.mark.parametrize("name", SQL_SKILLS)
def test_every_block_runs_as_written_on_a_locked_reader(
    name: str, which: str, skill_exports: Exports, request: pytest.FixtureRequest
) -> None:
    if which == "mini":  # the miniature fleet's real export, where its fixtures exist
        export = request.getfixturevalue("mini_export")
    else:
        export = skill_exports.fleet_export if which == "fleet" else skill_exports.toy_export
    with LogReader(export, materialize=True, lock=True) as reader:  # the tool server's way
        for block, sql in sql_blocks(name).items():
            result = reader.query(sql)
            assert result.num_rows <= MAX_ROWS, f"{name}/{block}"
            if which != "toy":
                assert result.num_rows > 0, f"{name}/{block} returned nothing"


def test_every_block_is_one_select_statement() -> None:
    import duckdb

    con = duckdb.connect()
    for name in SQL_SKILLS:
        for block, sql in sql_blocks(name).items():
            statements = con.extract_statements(sql)
            assert len(statements) == 1, f"{name}/{block}"
            assert statements[0].type == duckdb.StatementType.SELECT, f"{name}/{block}"
            first = sql.splitlines()[1]
            # Some blocks take only input rows, with no scalar parameters.
            assert first == "WITH" or (
                first.startswith("WITH ") and " params AS (SELECT " in first
            ), f"{name}/{block}"


def test_every_input_shows_the_form_of_its_rows() -> None:
    for name in SQL_SKILLS:
        module = helpers(name)
        for block, sql in sql_blocks(name).items():
            markers = [line for line in sql.splitlines() if line.startswith("-- input: ")]
            assert len(markers) == len(module.inputs_of(sql)), f"{name}/{block}"
            assert sql.count("-- end input") == len(markers), f"{name}/{block}"
            for line in markers:
                head = line.removeprefix("-- input: ").split(" AS ")[0]
                assert f"{head} AS (VALUES (" in line, f"{name}/{block}: {line}"


def test_supervisor_skills_have_python_helpers_only() -> None:
    for name in SUPERVISOR_SKILLS:
        assert sql_blocks(name) == {}
        assert helper_path(name).is_file()
