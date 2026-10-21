"""The probe a learned skill carries (lane P): the contract
(`contracts.learned.LearnedProbe`, the draft's optional `probe`), the tool server's check
of it on the fleet it is written on, and the recipe the freeze keeps beside the library
(`loop.learned`, `recipes/<name>.json`). A draft without a probe validates, dumps and
renders exactly as before the field existed. A draft with one carries a query that analyses
the probe's measurements (`probe_steps`), not merely one that mentions its id.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_loop_learned import SKILL, Worlds, receipt, skill, source, toolbox
from test_loop_learned import worlds as learned_worlds

from fleetlab.agents.audit import audit_learned_library
from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.learned import (
    LearnedProbe,
    LearnedSkillDraft,
    draft_problems,
    evidence_sql,
    probe_block,
    reads_probe,
    render_learned_skill,
)
from fleetlab.contracts.tools import json_schema_strict
from fleetlab.loop.learned import RECIPES_DIR, FrozenLibrary, freeze
from fleetlab.toolserver.files import LEARNED_FILE, append_line

worlds = learned_worlds  # the learned tests' toy fleet and export, as a fixture of this module

STEP: list[dict[str, Any]] = [
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 5,
     "period_steps": None},
    {"primitive": "step", "axis": "x", "amplitude_m": 0.03, "duration_steps": 15,
     "period_steps": None},
    {"primitive": "hold", "axis": None, "amplitude_m": None, "duration_steps": 5,
     "period_steps": None},
]  # fmt: skip
PROBE: dict[str, Any] = {
    "name": "step-lag-x",
    "purpose": "how long the hand of {robot} takes to answer a commanded step, against {reference}",
    "expectation": "a robot whose commands arrive at once moves in the next row",
    "robot_parameters": ["robot", "reference"],
    "repeats": 2,
    "segments": STEP,
}
# A probe analysis: the hand's largest distance from its commanded target per repetition.
PROBE_SQL = (
    "SELECT e.robot_id, e.repeat, round(max(abs(st.eef_pos[1] - st.target_pos[1])) * 1000, 1) "
    "AS worst_x_error_mm FROM probe_steps AS st JOIN probe_episodes AS e USING (episode_id) "
    "WHERE e.probe_id = $probe AND e.robot_id IN ($robot, $reference) AND st.target_pos IS "
    "NOT NULL GROUP BY e.robot_id, e.repeat ORDER BY e.robot_id, e.repeat"
)
# Mentions the probe's id, analyses nothing of it: refused as a probe's evidence.
EPISODES_SQL = (
    "SELECT $probe AS probe_id, robot_id, count(*) AS n FROM episodes "
    "WHERE robot_id IN ($robot, $reference) GROUP BY robot_id ORDER BY robot_id"
)


def checks(**values: str) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in values.items()]


def probe_skill(**changes: Any) -> dict[str, Any]:
    base = skill(
        name="command-lag-probe",
        parameters=[
            {"name": "probe", "meaning": "the id of the step probe below, once it has run"},
            {"name": "robot", "meaning": "a robot under suspicion"},
            {"name": "reference", "meaning": "a robot elsewhere that does well"},
        ],
        evidence=[
            {
                "claim": "the hand's largest distance from its target on {robot} and "
                "{reference}, per repetition of the step probe",
                "sql": PROBE_SQL,
                "result": "{robot} trailed its targets far more than {reference}",
            }
        ],
        next_experiment="Run the probe below on {robot} and {reference}, then the evidence "
        "query with its id.",
        check_with=checks(probe="p0001", robot="r04", reference="r01"),
        probe=json.loads(json.dumps(PROBE)),
    )
    return {**base, **changes}


# The contract.


def test_a_draft_without_a_probe_dumps_and_renders_as_before() -> None:
    draft = LearnedSkillDraft.model_validate(skill())
    assert draft.probe is None and "probe" not in draft.model_dump()
    # Digests of the dump and the rendering made by the code before the field existed.
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    assert digest == "8d268dba2abed2b76cbca7d69f4ddddae02e7db8be4818ddb4dd6fe52e3324fc"
    for provenance, expected in (
        (None, "169a47f9ebfc0374990f375551f815d75295171645842cad6500005fcb4f1e0a"),
        (
            "learned on fleet A, round 1",
            "eb0437338562849b0d8e5ff709896e55e84cccf90fd24bbabc9becd1b376fe93",
        ),
    ):
        text = render_learned_skill(draft, provenance)
        assert hashlib.sha256(text.encode()).hexdigest() == expected
        assert probe_block(text) is None
    assert draft_problems(draft) == []


def test_a_probe_draft_validates_renders_its_probe_and_makes_the_request() -> None:
    draft = LearnedSkillDraft.model_validate(probe_skill())
    assert draft_problems(draft) == []
    text = render_learned_skill(draft, "learned on fleet A")
    assert "tools_used: [query, request_experiment, request_probe]" in text
    assert "\n### The probe\n" in text and "\n## Provenance\n" in text
    assert draft.probe is not None
    assert probe_block(text) == draft.probe.model_dump(mode="json")
    assert LearnedProbe.model_validate(probe_block(text)) == draft.probe
    assert evidence_sql(text) == [("evidence-1", PROBE_SQL)]
    request = draft.probe.request(["r05", "r01"])
    assert request.robots == ["r05", "r01"] and request.repeats == 2
    assert request.purpose == (
        "how long the hand of r05 takes to answer a commanded step, against r01"
    )
    assert request.attempts == 4 and request.program.steps == 25
    with pytest.raises(ValueError, match="takes 2 robots"):
        draft.probe.request(["r05"])
    assert "probe" in draft.model_dump() and LearnedSkillDraft.model_validate(draft.model_dump())


def test_a_probe_analysis_reads_the_measurements_with_the_probe_s_id() -> None:
    assert reads_probe(PROBE_SQL)
    assert not reads_probe(EPISODES_SQL)  # takes $probe, reads no measurement
    assert not reads_probe("SELECT * FROM probe_steps")  # reads them, for no probe in particular
    assert not reads_probe("SELECT * FROM probe_steps_copy WHERE x = $probe")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"probe": {**PROBE, "robot_parameters": ["robot", "other"]}}, "not declared"),
        ({"probe": {**PROBE, "robot_parameters": ["probe", "robot"]}}, "the probe's own id"),
        (
            {
                "evidence": [{"claim": "c", "sql": "SELECT count(*) FROM episodes", "result": "r"}],
                "check_with": [],
            },
            "no evidence SQL analyses its measurements",
        ),
        (
            {
                "evidence": [
                    {
                        "claim": "the episodes of {robot} and {reference}",
                        "sql": EPISODES_SQL,
                        "result": "r",
                    }
                ]
            },
            "no evidence SQL analyses its measurements",
        ),
        ({"probe": {**PROBE, "purpose": "the step on {site_id}"}}, "placeholders ['site_id']"),
        ({"probe": {**PROBE, "expectation": "r05 moves late"}}, "identifier"),
    ],
)
def test_draft_problems_explain_what_is_wrong_with_the_probe(
    changes: dict[str, Any], message: str
) -> None:
    draft = LearnedSkillDraft.model_validate(probe_skill(**changes))
    assert any(message in p for p in draft_problems(draft)), draft_problems(draft)


@pytest.mark.parametrize(
    "probe",
    [
        {**PROBE, "segments": [{**STEP[1], "amplitude_m": 0.06}]},  # beyond 5 cm
        {**PROBE, "robot_parameters": ["robot", "robot"]},
        {**PROBE, "repeats": 6},
        {**PROBE, "robot_parameters": []},
    ],
)
def test_a_probe_outside_the_public_bounds_does_not_validate(probe: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        LearnedSkillDraft.model_validate(probe_skill(probe=probe))


def test_the_tool_schema_stays_strict_with_a_nullable_probe() -> None:
    schema = json_schema_strict(LearnedSkillDraft)
    assert "probe" in schema["required"]
    assert {"type": "null"} in schema["properties"]["probe"]["anyOf"]


# The tool server, on an export that holds a probe's tables.


@pytest.fixture(scope="module")
def probe_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The probe lane's toy world (the link at site s2) after one step probe on r01 and r04:
    an export with `probe_episodes` and `probe_steps`."""
    from fleetlab.loop.layout import layout_for_world
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.probe_toy import write_link_toy_world
    from fleetlab.loop.probes import probe_check
    from fleetlab.loop.state import Ledger

    root = tmp_path_factory.mktemp("probe-export")
    world, setup = write_link_toy_world(root / "world-root")
    request = LearnedProbe.model_validate(PROBE).request(["r04", "r01"])
    summary = probe_check(
        world,
        layout_for_world(world.spec),
        Ledger.empty(world.world_id),
        LoopPaths(root / "loop"),
        setup,
        "probe-check",
        request.model_dump(mode="json"),
        root / "export",
    )
    assert [o["status"] for o in summary["outcomes"]] == ["done"]
    return Path(summary["export"])


def test_write_learned_skill_keeps_a_probe_checked_on_this_fleet(
    worlds: Worlds, probe_export: Path, tmp_path: Path
) -> None:
    analysis = Analysis.create("w01", base=tmp_path / "analyses")
    box = toolbox(analysis, probe_export, worlds.env)
    try:
        answer = receipt(box.call("write_learned_skill", probe_skill()))
        assert answer["accepted"], answer["problems"]
        assert "```json" in answer["skill_md"]
        short = probe_skill(check_with=checks(probe="p0001", robot="r04"))
        missing = receipt(box.call("write_learned_skill", short))
        stranger = probe_skill(check_with=checks(probe="p0001", robot="r04", reference="r99"))
        unknown = receipt(box.call("write_learned_skill", stranger))
        idle = probe_skill(
            evidence=[{"claim": "the episodes of {robot} and {reference}", "sql": EPISODES_SQL,
                       "result": "r"}]
        )  # fmt: skip
        mentions = receipt(box.call("write_learned_skill", idle))
    finally:
        box.close()
    record = json.loads((analysis.outbox / LEARNED_FILE).read_text().splitlines()[0])
    assert record["draft"]["probe"] == PROBE and record["draft"]["check_with"] == []
    assert not missing["accepted"]
    assert any("reference" in p and "check_with" in p for p in missing["problems"]), missing
    assert not unknown["accepted"] and any("'r99'" in p for p in unknown["problems"]), unknown
    assert not mentions["accepted"]
    assert any("analyses its measurements" in p for p in mentions["problems"]), mentions


# The freeze.


def outbox_analysis(root: Path, **changes: Any) -> Analysis:
    """An analysis whose outbox holds one accepted probe draft (as the tool server keeps it)."""
    analysis = Analysis.create("w02", base=root / "source-analyses")
    draft = LearnedSkillDraft.model_validate({**probe_skill(**changes), "check_with": []})
    assert draft_problems(draft) == [], draft_problems(draft)
    record = {"receipt_id": "learned-1", "session": 1, "draft": draft.model_dump(mode="json")}
    append_line(analysis.outbox / LEARNED_FILE, record)
    return analysis


def test_the_freeze_keeps_the_probe_as_a_recipe_beside_the_mounted_skills(
    worlds: Worlds, tmp_path: Path
) -> None:
    analysis = outbox_analysis(tmp_path)
    report = freeze([source(worlds, analysis)], tmp_path / "libraries", kept_on="2026-10-19")
    library = report.library
    assert library is not None and report.kept == ["command-lag-probe"]
    # The mounted part holds the SKILL.md only, so the launch audit's rule holds.
    assert [p.name for p in (library.skills / "command-lag-probe").iterdir()] == ["SKILL.md"]
    assert audit_learned_library(library.skills, (worlds.source_run,), worlds.source_fleet) == []
    path = library.root / RECIPES_DIR / "command-lag-probe.json"
    assert path.is_file() and not os.access(path, os.W_OK)
    recipe = library.recipe("command-lag-probe")
    assert recipe == LearnedProbe.model_validate(PROBE)
    entry = library.record["skills"][0]
    assert entry["recipe"]["path"] == f"{RECIPES_DIR}/command-lag-probe.json"
    text = (library.skills / "command-lag-probe" / "SKILL.md").read_text()
    assert probe_block(text) == recipe.model_dump(mode="json")  # the agent reads the same probe
    os.chmod(path, 0o644)
    path.write_text(path.read_text().replace('"repeats": 2', '"repeats": 3'))
    with pytest.raises(ValueError, match="recipe of command-lag-probe changed"):
        FrozenLibrary.load(library.root)


def test_a_skill_without_a_probe_has_no_recipe_and_carrying_keeps_or_drops_it(
    worlds: Worlds, tmp_path: Path
) -> None:
    from test_loop_learned import source_analysis

    plain = source_analysis(worlds, tmp_path)
    one = freeze([source(worlds, plain)], tmp_path / "lib", kept_on="2026-10-19").library
    assert one is not None and one.recipe(SKILL["name"]) is None
    assert not (one.root / RECIPES_DIR).exists()
    with pytest.raises(KeyError):
        one.recipe("no-such-skill")
    probed = outbox_analysis(tmp_path)
    two = freeze(
        [source(worlds, probed, "fleet B")], tmp_path / "lib", previous=one, kept_on="2026-10-20"
    ).library
    assert two is not None and two.recipe("command-lag-probe") is not None
    third = outbox_analysis(tmp_path, name="other-lesson", probe=None)
    three = freeze(
        [source(worlds, third, "fleet C")], tmp_path / "lib", previous=two, kept_on="2026-10-21"
    ).library
    assert three is not None
    assert three.recipe("command-lag-probe") == LearnedProbe.model_validate(PROBE)  # carried
    assert three.recipe("other-lesson") is None
    replaced = outbox_analysis(tmp_path, probe=None)
    four = freeze(
        [source(worlds, replaced, "fleet D")], tmp_path / "lib", previous=three,
        kept_on="2026-10-22",
    ).library  # fmt: skip
    assert four is not None and four.recipe("command-lag-probe") is None
    assert not (four.root / RECIPES_DIR / "command-lag-probe.json").exists()
    FrozenLibrary.load(four.root)
