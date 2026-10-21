"""The skill format: front matter, loading, and the check that a skill stays general.

docs/design/contracts.md, section 13.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetlab.contracts.skills import load_skill, parse_front_matter, validate_skill_general

TRIAGE = """---
name: triage
description: Find the robots and days where the fleet's success rate departs from the rest.
when_to_use: At the start of an analysis, before any hypothesis.
tools_used: [query, describe]
inputs:
  - the visible tables
outputs: [a short list of suspect robots and days]
version: "1"
---
# Triage

1. Query the success rate per robot and per day: `select robot_id, day, ...`.
2. Compare each robot with its site; a camera that moved shows as misplaced grasps.
"""


def write_skill(root: Path, name: str, text: str, helper: str | None = None) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(text)
    if helper is not None:
        (directory / "helper.py").write_text(helper)
    return directory


def test_a_skill_loads_with_its_manifest(tmp_path: Path) -> None:
    skill = load_skill(write_skill(tmp_path, "triage", TRIAGE, helper="print('rates')\n"))
    assert skill.manifest.name == "triage"
    assert skill.manifest.tools_used == ["query", "describe"]
    assert skill.manifest.inputs == ["the visible tables"]
    assert skill.manifest.version == "1"
    assert skill.body.startswith("# Triage")
    assert [p.name for p in skill.helpers] == ["helper.py"]


def test_a_general_skill_passes(tmp_path: Path) -> None:
    directory = write_skill(tmp_path, "triage", TRIAGE)
    assert validate_skill_general(directory, ["camera_pose_offset", "action_delay"]) == []


def test_a_planted_robot_id_is_caught(tmp_path: Path) -> None:
    planted = TRIAGE.replace("Compare each robot", "Start with r07, then compare each robot")
    problems = validate_skill_general(write_skill(tmp_path, "triage", planted))
    assert problems == ["SKILL.md:14: identifier 'r07'"]


@pytest.mark.parametrize("identifier", ["e000123", "s2", "w01", "x0001", "f03"])
def test_every_kind_of_identifier_is_caught_in_helpers_too(tmp_path: Path, identifier: str) -> None:
    directory = write_skill(tmp_path, "triage", TRIAGE, helper=f"WATCH = '{identifier}'\n")
    assert validate_skill_general(directory) == [f"helper.py:1: identifier {identifier!r}"]


def test_terms_of_the_truth_vocabulary_are_caught(tmp_path: Path) -> None:
    planted = TRIAGE.replace("a camera that moved", "a Camera_Pose_Offset of 6 degrees")
    problems = validate_skill_general(
        write_skill(tmp_path, "triage", planted), ["camera_pose_offset"]
    )
    assert problems == ["SKILL.md:14: forbidden term 'camera_pose_offset'"]


def test_an_invalid_manifest_is_a_problem(tmp_path: Path) -> None:
    wrong_dir = write_skill(tmp_path, "scope", TRIAGE)
    assert any("directory named 'scope'" in p for p in validate_skill_general(wrong_dir))
    unknown_tool = TRIAGE.replace("[query, describe]", "[query, read_truth]")
    bad = write_skill(tmp_path / "x", "triage", unknown_tool)
    assert any("read_truth" in p for p in validate_skill_general(bad))
    with pytest.raises(ValueError, match="front matter"):
        parse_front_matter("# no front matter\n")
    with pytest.raises(ValueError, match="not a front matter line"):
        parse_front_matter("---\nname: triage\n{oops}\n---\n")
