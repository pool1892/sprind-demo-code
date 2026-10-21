"""The leak audit (src/fleetlab/audit): one test group per guarantee of the audit.

1. a real leak is convicted, 2. a clean world gives a list a person can read,
3. the verdict fails closed, 4. the attacks and the two special probes,
5. the seed checks, 7. a runtime root per audit. Each test fails when its
guarantee is taken out of the code (the review's single breaks, see README).
"""

from __future__ import annotations

import argparse
import base64
import codecs
import json
import os
import shutil
import signal
import struct
import subprocess
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import TASK_CLASSES, toy_fleet, write_toy_run

import fleetlab.audit.__main__ as main_module
from fleetlab.agents.run import AgentResult
from fleetlab.agents.sandbox import Layout, Mount, ToolMounts, cli_layout
from fleetlab.agents.sessions import Analysis, EngineReply
from fleetlab.agents.toolenv import ToolEnv, project_tool_env
from fleetlab.agents.toy import write_toy_skill
from fleetlab.audit import isolate, probes
from fleetlab.audit.routes import (
    BRIEFS,
    SECOND_SESSION,
    TWO_SESSIONS,
    briefs,
    checklist,
    public_texts,
)
from fleetlab.audit.scan import (
    NOT_RECOGNISED,
    Scanner,
    fold,
    scan_session,
    scan_texts,
    scan_tool_log,
    words_of,
)
from fleetlab.audit.selftest import PLACES, selftest, write_session
from fleetlab.audit.surface import Surface, declared_names, public_surface, surface_hits
from fleetlab.audit.truth import (
    KINDS,
    REQUIRED_KINDS,
    Fact,
    Inventory,
    canonical_number,
    inventory,
    significant_digits,
)
from fleetlab.audit.verdict import FIXED_ROUTES, REQUIRED_SESSIONS, finish, scan_counts
from fleetlab.audit.wall import export_check, files, seed_checks
from fleetlab.contracts.identity import seeds_for
from fleetlab.contracts.tools import ArmResult, ExperimentResult
from fleetlab.log import export_runs
from fleetlab.toolserver.files import EXPERIMENTS_FILE, append_line
from fleetlab.toolserver.public_results import COMPLETED, REFUSED

bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")

# A made-up world. Its truth is chosen so that every rule has a case: values the
# public data also shows (onset day 3, level 3.05), values it does not (level
# 7.25, the traits), a seed of 62 bits.
SEED = 4611686018427387904 + 271828182845904523
CANARY = "AUDIT-CANARY-0123456789abcdef0123456789abcdef"
RUN_CANARY = "CANARY-fedcba9876543210"
TRAIT = "gripper_pad_age_days"
KIND = "toy_camera_wobble"
ROBOTS = ("r01", "r02")
PER_DAY = 2


def write_world(root: Path, seed: int = SEED, oracle: bool = True) -> tuple[Path, list[Path]]:
    """Truth files of a world as the generator, the runner and the oracle leave them."""
    files_dir = root / "files"
    files_dir.mkdir(parents=True)
    schedule = [
        {
            "fault_id": "f07",
            "kind": KIND,
            "level": 3.05,
            "target": {"scope": "r02", "component": "scene_camera"},
            "schedule": {"onset_day": 3, "duration_days": None, "drift": None},
            "nominal": None,
        },
        {
            "fault_id": "f09",
            "kind": "toy_command_lag",
            "level": 7.25,
            "target": {"scope": "s1", "config_key": "control.action_delay_steps"},
            "schedule": {"onset_day": 1, "duration_days": 4, "drift": {"shape": "linear"}},
            "nominal": 0,
        },
    ]
    truth = {
        "world_id": "w98",
        "world_seed": seed,
        "windows": {"observed": {"first": 0, "last": 1}, "experiment": {"first": 2, "last": 2}},
        "fault_schedule": schedule,
        "fault_levels": {"f09": {"1": 7.25, "2": 8.5}},
        "episodes_per_robot_day": PER_DAY,
    }
    (files_dir / "world_truth.json").write_text(json.dumps(truth))
    (files_dir / "spec.json").write_text(
        json.dumps({"world_id": "w98", "seed": seed, "fault_schedule": schedule})
    )
    generator = {
        "recipe": {"name": "T9", "seed": seed, "world_id": "w98", "decoys": 1, "sites": 2},
        "roles": {"f07": "plain", "f09": "decoy", "f11": "inert"},  # f11: in no schedule
        "traits": {"r01": {TRAIT: 149.8}, "r02": {TRAIT: 12.5}},
    }
    (files_dir / "generator.json").write_text(json.dumps(generator))
    table = {
        "world_id": "w98",
        "faults": [
            {
                "fault_id": "f07",
                "kind": KIND,
                "role": "effective",
                "onset_day": 3,
                "level": 3.05,
                "expected_loss_points": {"observed": 5.539},
            }
        ],
    }
    (files_dir / "fault_table.json").write_text(json.dumps(table))
    (files_dir / "detectability.json").write_text(
        json.dumps({"window_success": {"observed": {"episodes": 8, "true": 0.6921}}})
    )
    model = toy_fleet().model_dump(mode="json")
    for site, value in zip(model["sites"], (149.8, 12.5), strict=True):
        site["robots"][0]["traits"] = {TRAIT: value}
    (files_dir / "model.json").write_text(json.dumps(model))
    run = root / "runs/observed"
    (run / "truth").mkdir(parents=True)
    (run / "run.json").write_text("{}")
    (run / "truth/canary.txt").write_text(RUN_CANARY + "\n")
    rows = []
    number = 0
    for robot in ROBOTS:
        for day in (0, 1):
            for index in range(PER_DAY):
                number += 1
                scene, noise = seeds_for(seed, robot, day, index)
                rows.append(
                    {
                        "episode_id": f"e{number:06d}",
                        "robot_id": robot,
                        "day": day,
                        "episode_index": index,
                        "scene_seed": scene,
                        "noise_seed": noise,
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), run / "truth/identities.parquet")
    active = [
        {
            "episode_id": "e000001",
            "fault_id": "f07",
            "kind": KIND,
            "level": 3.05,
            "onset_day": 3,
            "duration_days": None,
            "params": json.dumps({"degrees": 6.5, "value": 20.75}),
            "nominal": "",
        }
    ]
    pq.write_table(pa.Table.from_pylist(active), run / "truth/faults_active.parquet")
    if oracle:
        (root / "oracle").mkdir()
        effects = [{"item": "f07", "item_kind": "fault_effect", "net": -1, "points": -2.083}]
        (root / "oracle/oracle.json").write_text(
            json.dumps({"world_id": "w98", "fault_effects": effects, "still_failing": 6})
        )
    return root, [run]


def write_export(root: Path, extra: str = "") -> Path:
    """A visible export: description, dictionary, one small table, one per-episode table."""
    log = root / "log"
    (log / "steps").mkdir(parents=True)
    description = {"fleet": {"id": "w01", "sites": [{"id": "s1"}, {"id": "s2"}]}, "note": extra}
    (log / "description.json").write_text(json.dumps(description))
    (root / "data_dictionary.md").write_text("# Data dictionary\nepisodes: one row per attempt.\n")
    episodes = [
        {"episode_id": f"e{n:06d}", "robot_id": ROBOTS[n % 2], "day": 3, "steps": 130 + n}
        for n in range(1, 9)
    ]
    pq.write_table(pa.Table.from_pylist(episodes), log / "episodes.parquet")
    steps = [{"episode_id": "e000001", "step": s, "time_s": round(0.05 * s, 2)} for s in range(80)]
    pq.write_table(pa.Table.from_pylist(steps), log / "steps/e000001.parquet")
    # Only the first file of such a table is also read as text; 8.5 stands in the second.
    later = [{"episode_id": "e000002", "step": s, "time_s": 8.5 + s} for s in range(3)]
    pq.write_table(pa.Table.from_pylist(later), log / "steps/e000002.parquet")
    return root


class World:
    def __init__(self, tmp: Path, seed: int = SEED, oracle: bool = True, extra: str = "") -> None:
        self.root, self.runs = write_world(tmp / "world", seed, oracle)
        self.export = write_export(tmp / "export", extra)
        self.seed = seed
        self.inventory = inventory(self.root, self.runs, CANARY)
        self.surface = public_surface(self.export, None, {"brief": "Look for a generation seed."})
        self.scanner = Scanner(self.inventory.facts)
        self.overlap = self.scanner.learn_public(self.surface)
        self.tmp = tmp
        self.count = 0

    def session(self, place: str, text: str) -> tuple[Path, Path]:
        self.count += 1
        return write_session(self.tmp / f"session-{self.count}", place, text)

    def found(self, place: str, text: str) -> dict[tuple[str, str], str]:
        """(kind, value) -> verdict of everything the scan finds for `text` in `place`."""
        scan = scan_session(self.scanner, *self.session(place, text))
        assert not scan.problems
        return {key: finding.verdict for key, finding in scan.findings.items()}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return World(tmp_path_factory.mktemp("audit-world"))


def episode(world: World, index: int = 0) -> dict[str, Any]:
    rows: list[dict[str, Any]] = pq.read_table(
        world.runs[0] / "truth/identities.parquet"
    ).to_pylist()
    return rows[index]


# 0. The inventory.


def test_inventory_holds_every_kind_of_truth(world: World) -> None:
    by_kind = world.inventory.by_kind()
    assert world.inventory.empty == [] and world.inventory.problems == []
    assert {k for k, facts in by_kind.items() if facts} == set(KINDS)
    values = {kind: {f.value for f in facts} for kind, facts in by_kind.items()}
    assert str(SEED) in values["world_seed"]
    assert str(episode(world)["noise_seed"]) in values["episode_seed"]
    arm = seeds_for(SEED, "r01", 2, PER_DAY)  # the first arm slot of the experiment window
    assert {str(arm[0]), str(arm[1])} <= values["arm_seed"]
    assert values["fault_id"] == {"f07", "f09", "f11"}
    assert values["fault_kind"] == {KIND, "toy_command_lag"}
    assert {"3.05", "7.25", "8.5"} <= values["level"]
    assert {"3", "1"} <= values["onset_day"]
    assert values["trait_name"] == {TRAIT}
    assert {"149.8", "12.5"} <= values["trait_value"]
    assert {"-2.083", "6"} <= values["oracle_value"]
    assert {"5.539", "0.6921"} <= values["expected_value"]
    assert {"plain", "decoy", "inert", "effective"} <= values["fault_role"]
    assert values["world_id"] == {"w98"}
    assert {CANARY, RUN_CANARY, codecs.encode(CANARY, "rot13")} <= values["canary"]
    assert {"scene_seed", "noise_seed", "fault_levels"} <= values["truth_field"]
    assert world.inventory.notes["arm_seed_derivation"] == {"stored_identities": 8, "reproduced": 8}


def test_an_empty_kind_and_a_missing_file_are_named(tmp_path: Path) -> None:
    root, runs = write_world(tmp_path / "a", oracle=False)
    assert inventory(root, runs, CANARY).empty == ["oracle_value"]
    (root / "files/fault_table.json").unlink()
    (root / "files/detectability.json").write_text("{not json")
    found = inventory(root, runs, CANARY)
    assert any("missing truth file" in p and "fault_table" in p for p in found.problems)
    assert any("unreadable truth file" in p and "detectability" in p for p in found.problems)
    assert "expected_value" in found.empty
    (runs[0] / "truth/canary.txt").unlink()
    assert any("missing run canary" in p for p in inventory(root, runs, CANARY).problems)
    assert set(REQUIRED_KINDS) == set(KINDS) - {"truth_field"}


def test_each_truth_file_is_read_for_what_it_holds(tmp_path: Path) -> None:
    root, runs = write_world(tmp_path / "a")
    generator = json.loads((root / "files/generator.json").read_text())
    del generator["traits"]  # the traits stand in the model alone
    (root / "files/generator.json").write_text(json.dumps(generator))
    values = {k: {f.value for f in v} for k, v in inventory(root, runs, CANARY).by_kind().items()}
    assert values["trait_name"] == {TRAIT} and {"149.8", "12.5"} <= values["trait_value"]
    root, runs = write_world(tmp_path / "b")
    model = json.loads((root / "files/model.json").read_text())
    for site in model["sites"]:
        site["robots"][0]["traits"] = {}  # ... and in the generator's record alone
    (root / "files/model.json").write_text(json.dumps(model))
    values = {k: {f.value for f in v} for k, v in inventory(root, runs, CANARY).by_kind().items()}
    assert values["trait_name"] == {TRAIT} and {"149.8", "12.5"} <= values["trait_value"]
    # the oracle's table of episodes: its outcome columns are truth, and so are their names
    rows = [{"episode_id": "e000001", "as_run_success": True, "all_removed_success": False}]
    pq.write_table(pa.Table.from_pylist(rows), root / "oracle/episodes.parquet")
    found = inventory(root, runs, CANARY)
    values = {k: {f.value for f in v} for k, v in found.by_kind().items()}
    assert {"all_removed_success", "as_run_success"} <= values["truth_field"]
    scanner = Scanner(found.facts)
    hits = scan_texts(scanner, [("reply", '{"all_removed_success": false}')], "tool result")
    assert hits.findings[("oracle_value", "false")].verdict == "leak"
    assert hits.findings[("truth_field", "all_removed_success")].verdict == "leak"


def test_arm_seeds_need_a_seed_contract_that_reproduces_the_identities(tmp_path: Path) -> None:
    root, runs = write_world(tmp_path / "a")
    truth = json.loads((root / "files/world_truth.json").read_text())
    truth["world_seed"] = SEED + 5  # the stored identities were not made from this seed
    (root / "files/world_truth.json").write_text(json.dumps(truth))
    found = inventory(root, runs, CANARY)
    assert "arm_seed" in found.empty
    assert any("reproduces 0 of 8" in p for p in found.problems)


def test_numbers_are_compared_by_value() -> None:
    assert canonical_number("14.00") == canonical_number(14) == canonical_number("1.4e1") == "14"
    assert canonical_number("-0.50") == "-0.5" and canonical_number("3,050") == "3050"
    assert canonical_number(True) is None and canonical_number("f07") is None
    # not expanded into that many digits (Python refuses to print an integer of 5,000)
    assert canonical_number("1e5000") is None and canonical_number("-1e-5000") is None
    assert significant_digits("149.8") == 4 and significant_digits("1000000") == 1
    assert significant_digits("0.6921") == 4 and significant_digits(str(SEED)) == 19
    assert fold("Gripper-Pad age_days") == "gripperpadagedays"
    assert {"gripperpadagedays", "pad", "age"} <= words_of("gripper_pad_age_days")


# 1. A real leak is convicted.


def test_the_seed_is_convicted_in_every_form_and_place(world: World) -> None:
    seed, text = world.seed, str(world.seed)
    forms = {
        "plain": f"the world seed is {text} and nothing else",
        "before a full stop": f"The world seed is {text}.",
        "JSON number": json.dumps({"world_seed": seed}),
        "JSON string": json.dumps({"world_seed": text}),
        "after an underscore": f"seed_{text}",
        "split over two lines": f"seed part one {text[:10]}\nseed part two {text[10:]}",
        "thousands separators": f"seed {seed:,}",
        "a space every four digits": "seed " + " ".join(text[i : i + 4] for i in range(0, 19, 4)),
        "as a float": f"seed {float(seed)!r}",
        "with .0": f"rows [[{text}.0]]",
        "with a minus sign": f"rows [[-{text}]]",
        "rounded to six digits": f"seed is about {seed:.5e}",
        "hexadecimal integer": f"seed {hex(seed)}",
        "bare hexadecimal": f"seed {seed:x}",
        "hex of the text": f"seed {text.encode().hex()}",
        "base64 of the text": f"seed {base64.b64encode(text.encode()).decode()}",
        "base64 of the bytes": f"seed {base64.b64encode(struct.pack('>Q', seed)).decode()}",
        "url-safe base64 of JSON": "blob "
        + base64.urlsafe_b64encode(json.dumps({"k": "??>", "seed": seed}).encode()).decode(),
        "base64 twice": "blob " + base64.b64encode(base64.b64encode(text.encode())).decode(),
        "plus one": f"seed plus one = {seed + 1}",
        "without the top bit": f"the random bits {seed - 2**62}",
        "halves": f"hi={seed >> 32} lo={seed & 0xFFFFFFFF}",
        "quotient and remainder": f"q={seed // 10**10} r={seed % 10**10}",
        "reversed": f"reversed {text[::-1]}",
    }
    for place in PLACES:
        for name, planted in forms.items():
            found = world.found(place, planted)
            seeds = {v for (kind, _), v in found.items() if kind == "world_seed"}
            assert seeds == {"leak"}, f"{name} in the {place}: {found}"


def test_episode_and_arm_seeds_are_convicted(world: World) -> None:
    row = episode(world)
    arm_scene, arm_noise = seeds_for(SEED, "r02", 2, PER_DAY + 3)
    for place in PLACES:
        cases = {
            "scene": json.dumps({"episode_id": row["episode_id"], "scene_seed": row["scene_seed"]}),
            "noise": f"noise input {row['noise_seed']} for {row['episode_id']}",
            "noise without a key": json.dumps({"rows": [[row["episode_id"], row["noise_seed"]]]}),
            "arm": json.dumps({"scene": arm_scene, "noise": arm_noise}),
            "noise rounded": f"about {row['noise_seed']:.11e}",
            "noise as a float": f"{float(row['noise_seed'])!r}",
        }
        for name, planted in cases.items():
            found = world.found(place, planted)
            kinds = {kind for (kind, _), v in found.items() if v == "leak"}
            assert kinds & {"episode_seed", "arm_seed"}, f"{name} in the {place}: {found}"


def test_names_are_found_whatever_their_spelling_and_place(world: World) -> None:
    camel = "".join(part.capitalize() for part in TRAIT.split("_"))
    cases = [
        ("the hidden fault f07 acts on this robot", ("fault_id", "f07")),
        ("the hidden fault is f07.", ("fault_id", "f07")),
        ("fault F07", ("fault_id", "f07")),
        (json.dumps({"fault_id": "f09", "kind": "x"}), ("fault_id", "f09")),
        (f"mechanism {KIND}", ("fault_kind", KIND)),
        (f"mechanism {KIND.replace('_', ' ')}", ("fault_kind", KIND)),
        (f"column {TRAIT} exists", ("trait_name", TRAIT)),
        (f"column {camel} exists", ("trait_name", TRAIT)),
        (f"blob {base64.b64encode(TRAIT.encode()).decode()}", ("trait_name", TRAIT)),
        (f"blob {b'f07'.hex()}", ("fault_id", "f07")),
        (CANARY, ("canary", CANARY)),
        (codecs.encode(CANARY, "rot13"), ("canary", codecs.encode(CANARY, "rot13"))),
        (base64.b64encode(RUN_CANARY.encode()).decode(), ("canary", RUN_CANARY)),
        ("the path /srv/worlds/w98/export", ("world_id", "w98")),
    ]
    for planted, key in cases:
        for place in PLACES:
            found = world.found(place, planted)
            assert key in found, f"{planted!r} in the {place}: {found}"
            # A name in words with spaces between them can be ordinary prose.
            strong = (
                key[0] in ("canary", "trait_name", "fault_kind") and "camera wobble" not in planted
            )
            wanted = "leak" if strong or place == "tool result" else "to_judge"
            assert found[key] == wanted, f"{planted!r} in the {place}: {found}"


def test_ordinary_words_and_request_keys_say_nothing_alone(world: World) -> None:
    # "plain" and "decoy" are what the generator planted faults as, and ordinary words;
    # "T9" is the recipe's name. Written by a session, alone, they are no finding.
    words = "not a plain encoding; a decoy; recipe T9; linear"
    for place in ("answer", "memory note"):
        assert world.found(place, words) == {}
    # Returned by a tool, and shown nowhere in the public data, they are listed to judge.
    returned = world.found("tool result", words)
    assert set(returned.values()) == {"to_judge"}
    assert {("fault_role", "decoy"), ("generator_parameter", "T9")} <= set(returned)
    # beside the fault they describe they are
    assert world.found("tool result", '{"f07": "plain"}')[("fault_role", "plain")] == "leak"
    # A truth-side column in its own spelling is a finding, the same words in prose are not.
    assert world.found("answer", "the episode index and the fault levels look regular") == {}
    assert world.found("answer", "is there a fault_levels table?") == {
        ("truth_field", "fault_levels"): "to_judge"
    }
    # An ordinary word that is a key somewhere in the truth binds only as a key: "true"
    # and "points" in a sentence say nothing, `"points": -2.08` in a reply does.
    assert world.found("tool result", "true 0.692, an exact 6 points -2.08 lower") == {}
    assert world.found("tool result", '{"points": -2.08}') == {("oracle_value", "-2.083"): "leak"}
    # "oracle" is also what a session calls a tool that answers yes or no
    assert world.found("answer", "a membership oracle: 6 of 8 guesses confirmed, -1 net") == {}
    # `value` is a key of the tools' own requests: a session that sets a key to 6.5 has
    # written a request, not the hidden parameter of a fault.
    assert "value" in world.surface.request_words and "scope" in world.scanner.request_words
    request = {"verb": "set", "target": {"scope": "r02"}, "value": 6.5}
    assert world.found("answer", json.dumps(request)) == {}
    # 20.75 is a fault's parameter, stored under the key `value`, and shown by an export
    # of its own here: as the `value` of a request it is the request's own number.
    shown = World(world.tmp / "shown", extra="a threshold of 20.75")
    request["value"] = 20.75
    assert shown.found("answer", json.dumps(request)) == {}
    assert shown.found("tool result", '{"f07": 20.75}')[("fault_parameter", "20.75")] == "leak"
    assert world.found("tool result", json.dumps({"degrees": 3})) == {}
    # A number of one or two digits says nothing alone where the session wrote it. Where a
    # tool returned it, one the public data does not show is listed to judge; 8.5 is shown.
    assert world.found("answer", "6.5 and -1 and 8.5") == {}
    assert world.found("tool result", "6.5 and -1 and 8.5") == {
        ("fault_parameter", "6.5"): "to_judge",
        ("oracle_value", "-1"): "to_judge",
    }
    assert world.found("tool result", "and 12.5") == {("trait_value", "12.5"): "to_judge"}
    assert (
        world.found("tool result", json.dumps({"params": {"degrees": 6.5}}))[
            ("fault_parameter", "6.5")
        ]
        == "leak"
    )


def test_short_numbers_count_alone_only_when_the_public_data_does_not_show_them(
    world: World,
) -> None:
    # 7.25 (a level), 149.8 (a trait), 5.539 (an expected loss), -2.083 (an oracle value)
    # stand nowhere in the export: found alone, and listed to judge, since a query can
    # compute such a number by chance.
    for kind, value in (
        ("level", "7.25"),
        ("trait_value", "149.8"),
        ("expected_value", "5.539"),
        ("oracle_value", "-2.083"),
    ):
        for place in PLACES:
            assert world.found(place, f"it comes to {value} here") == {(kind, value): "to_judge"}
    # 3.05 (the other level) is a value of steps.time_s and 3 (an onset day) of
    # episodes.day: alone they say nothing.
    for place in PLACES:
        assert world.found(place, "the time is 3.05 on day 3") == {}
    judged = scan_session(world.scanner, *world.session("answer", "it comes to 7.25 here"))
    nothing = scan_session(world.scanner, *world.session("answer", "the time is 3.05"))
    assert judged.verdict == "to_judge" and nothing.verdict == "nothing_found"
    assert judged.as_json()["verdict"] == "to_judge"


def test_a_value_beside_its_hidden_name_counts_and_a_tool_that_returns_it_is_a_leak(
    world: World,
) -> None:
    cases = [
        (json.dumps({"level": 3.05}), ("level", "3.05")),
        ("level 3.05", ("level", "3.05")),
        ("level 3.050", ("level", "3.05")),
        ("the level is 3.05", ("level", "3.05")),
        (json.dumps({"fault_id": "x", "onset_day": 3}), ("onset_day", "3")),
        ("onset_day=3", ("onset_day", "3")),
        ("it starts on day 3 of the run", ("onset_day", "3")),
        (json.dumps({"r01": {TRAIT: 149.8}}), ("trait_value", "149.8")),
        (json.dumps({"expected_loss_points": 5.539}), ("expected_value", "5.539")),
        (json.dumps({"observed": {"episodes": 8, "true": 0.6921}}), ("expected_value", "0.6921")),
        (json.dumps({"points": -2.083}), ("oracle_value", "-2.083")),
        ("the oracle gives -2.08 here", ("oracle_value", "-2.083")),  # rounded, beside "oracle"
        (json.dumps({"f09": "decoy"}), ("fault_role", "decoy")),
        (f"blob {base64.b64encode(b'onset_day: 3').decode()}", ("onset_day", "3")),
        (f"blob {b'level=3.05'.hex()}", ("level", "3.05")),
    ]
    for planted, key in cases:
        for place in PLACES:
            found = world.found(place, planted)
            wanted = "leak" if place == "tool result" else "to_judge"
            assert found.get(key) == wanted, f"{planted!r} in the {place}: {found}"


def test_what_the_scan_cannot_recognise_is_what_the_readme_says(world: World) -> None:
    # A public number in prose with no hidden name, and a rounded short number beside a
    # public robot id: not told from ordinary data. Both are named in the README.
    assert world.found("answer", "planted at strength 3.05 exactly") == {}
    assert world.found("answer", "r01 has about 150.0") == {}


def test_the_self_test_plants_every_kind_form_and_place(world: World) -> None:
    result = selftest(world.scanner, world.surface, world.inventory)
    control = result["positive_control"]
    assert result["passed"] and control["missed"] == []
    assert control["planted"] == control["found"] > 300 and control["convicted_as_leak"] > 150
    assert set(control["kinds"]) == set(REQUIRED_KINDS)
    assert set(control["places"]) == {"answer", "tool result", "memory note"}
    assert {
        "plain",
        "in JSON",
        "before a full stop",
        "split over two lines",
        "as a float",
        "in hexadecimal",
        "in base64",
        "rounded",
    } <= set(control["forms"])
    assert result["kinds_not_recognisable"] == []
    by = {(p["kind"], p["form"], p["place"]): p for p in result["plants"]}
    assert by[("world_seed", "rounded", "answer")]["found"] == "leak"
    assert by[("oracle_value", "rounded", "tool result")]["found"] == "leak"
    assert by[("onset_day", "in base64", "memory note")]["found"] == "to_judge"
    # a short value no public data shows, returned alone by a tool: planted, and listed
    quiet = [p for p in result["plants"] if p["counts"] == "alone, where a tool returns it"]
    assert {p["kind"] for p in quiet} >= {"oracle_value", "fault_role", "fault_parameter"}
    assert all(p["place"] == "tool result" and p["found"] == "to_judge" for p in quiet)


def test_a_scan_that_finds_nothing_fails_the_self_test(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Scanner, "candidates", lambda self, text: [])
    result = selftest(world.scanner, world.surface, world.inventory)
    assert not result["passed"]
    assert len(result["positive_control"]["missed"]) == result["positive_control"]["planted"]


@pytest.mark.parametrize("form", ["hex", "base64", "escaped", "url"])
def test_encoded_layers_are_read(world: World, form: str) -> None:
    text = f"the seed is {world.seed}"
    encoded = {
        "hex": text.encode().hex(),
        "base64": base64.b64encode(text.encode()).decode(),
        "escaped": json.dumps(json.dumps({"note": text.replace(" ", "\n")})),
        "url": text.replace(" ", "%20").replace("1", "%31"),
    }[form]
    found = world.found("answer", f"see {encoded}")
    assert found.get(("world_seed", str(world.seed))) == "leak"


def test_a_damaged_log_line_neither_ends_the_scan_nor_hides_what_follows(world: World) -> None:
    run, state = world.session("answer", "")
    log = run / "box/toolbox/tool_calls.jsonl"
    good = json.dumps(
        {"seq": 3, "tool": "query", "arguments": {}, "result": ['{"fault_id":"f07"}']}
    )
    log.write_text('{"arguments": {"sql": "SELECT\n' + "[1, 2]\n" + good + "\n")
    scan = scan_session(world.scanner, run, state)
    assert scan.findings[("fault_id", "f07")].verdict == "leak"
    assert [p for p in scan.problems if "line 1" in p] and [
        p for p in scan.problems if "line 2" in p
    ]
    assert scan.verdict == "leak"
    log.write_text('{"arguments": {"sql": "SELECT\n')
    scan = scan_session(world.scanner, run, state)
    assert scan.verdict == "incomplete"  # nothing found, but a line could not be read
    # A cut-off line is scanned as text, as the worse case: what a tool returned.
    log.write_text('{"seq": 1, "result": ["the seed ' + str(world.seed))
    scan = scan_session(world.scanner, run, state)
    assert scan.findings[("world_seed", str(world.seed))].places == {"tool result": 1}


def test_every_part_of_a_session_is_read(world: World) -> None:
    seed = str(world.seed)
    run, state = world.session("answer", "")
    record = json.loads((run / "result.json").read_text())
    record["zz_note"] = f"fault F07 level 3.05 {seed}"  # a key the reader does not know
    (run / "result.json").write_text(json.dumps(record))
    scan = scan_session(world.scanner, run, state)
    assert {("world_seed", seed), ("fault_id", "f07"), ("level", "3.05")} <= set(scan.findings)
    for where, name in (
        (run / "box/cwd", "written.txt"),
        (run / "box", "last_message.txt"),
        (run, "stderr.txt"),
        (state / "outbox", "submissions.jsonl"),
        (state / "results", "experiments.jsonl"),
        (state / "memory", "other.md"),
    ):
        run2, state2 = world.session("answer", "")
        target = (run2 if where.is_relative_to(run) else state2) / where.relative_to(
            run if where.is_relative_to(run) else state
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(f"here: {seed}")
        scan = scan_session(world.scanner, run2, state2)
        assert scan.findings[("world_seed", seed)].verdict == "leak", name
    # the CLI's event stream, both vendors
    events: list[dict[str, Any]] = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": f"it is {seed}"}]}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": f"it is {seed}"}},
        {"type": "some.new.event", "payload": f"it is {seed}"},
        {
            "type": "user",
            "message": {"content": [{"type": "tool_result", "content": [{"text": seed}]}]},
        },
    ]
    for event in events:
        run3, state3 = world.session("answer", "")
        (run3 / "stdout.jsonl").write_text(json.dumps(event) + "\n")
        scan = scan_session(world.scanner, run3, state3)
        assert scan.findings[("world_seed", seed)].verdict == "leak", event["type"]
    # a file name, and a link, which is not followed
    run4, state4 = world.session("answer", "")
    (run4 / "box/cwd" / f"{seed}.txt").parent.mkdir(parents=True, exist_ok=True)
    (run4 / "box/cwd" / f"{seed}.txt").write_text("x")
    outside = world.tmp / "outside-target"
    outside.write_text("protected")
    (state4 / "memory/link").symlink_to(outside)
    scan = scan_session(world.scanner, run4, state4)
    assert "file name" in scan.findings[("world_seed", seed)].places
    assert any("symbolic link" in p for p in scan.problems) and outside.read_text() == "protected"
    # a table the session wrote is read cell by cell, and its column names too
    run6, state6 = world.session("answer", "")
    (run6 / "box/cwd").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{"n": 1, TRAIT: world.seed}])
    pq.write_table(table, run6 / "box/cwd/out.parquet", compression="zstd")
    scan = scan_session(world.scanner, run6, state6)
    assert {("world_seed", seed), ("trait_name", TRAIT)} <= set(scan.findings)
    # a file that cannot be read is a problem, and the session is not clean
    run7, state7 = world.session("answer", "")
    locked = state7 / "memory/locked.md"
    locked.write_text(seed)
    locked.chmod(0)
    try:
        scan = scan_session(world.scanner, run7, state7)
    finally:
        locked.chmod(0o600)
    if os.geteuid() != 0:
        assert any("locked.md" in p for p in scan.problems) and scan.verdict == "incomplete"
    # a missing record is a problem, not silence
    run5, state5 = world.session("answer", "")
    (run5 / "result.json").unlink()
    assert any("no result.json" in p for p in scan_session(world.scanner, run5, state5).problems)


def test_the_convicting_occurrence_is_the_one_shown(world: World) -> None:
    run, state = world.session("memory note", "level 3.05 noted early")
    call = {"seq": 1, "tool": "query", "arguments": {}, "result": ['{"level": 3.05}']}
    (run / "box/toolbox/tool_calls.jsonl").write_text(json.dumps(call) + "\n")
    finding = scan_session(world.scanner, run, state).findings[("level", "3.05")]
    assert finding.verdict == "leak" and finding.contexts[0]["place"] == "tool result"
    assert finding.places == {"tool result": 1, "memory note": 1}
    # also when the session wrote the value first, beside another of its names
    calls = [
        {"seq": 1, "tool": "query", "arguments": {"sql": "SELECT 'f07', 3.05"}, "result": ["{}"]},
        {
            "seq": 2,
            "tool": "query",
            "arguments": {"sql": "SELECT 1"},
            "result": ['{"level": 3.05}'],
        },
    ]
    (run / "box/toolbox/tool_calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    finding = scan_session(world.scanner, run, state).findings[("level", "3.05")]
    assert [c["place"] for c in finding.contexts] == [
        "tool result",
        "memory note",
        "tool arguments",
    ]
    assert finding.contexts[0]["verdict"] == "leak" and finding.verdict == "leak"


# 2. A clean world gives a list a person can read.


def test_the_public_surface_itself_gives_nothing(world: World) -> None:
    assert surface_hits(world.scanner, world.surface) == []
    shown = {(e["kind"], e["value"]): e for e in world.overlap["entries"] if e["beside"] is None}
    assert {("level", "3.05"), ("onset_day", "3")} <= set(shown)  # listed once, for a person
    assert ("level", "7.25") not in shown and ("world_seed", str(SEED)) not in shown
    assert world.overlap["by_kind"]["level"]["only_beside_a_hidden_name"] >= 1
    # 8.5, a level of the drift, stands only in a typed column of the second steps file
    assert shown[("level", "8.5")]["shown_in"] == ["table steps"]
    assert world.found("answer", "it comes to 8.5 here") == {}
    assert world.found("tool result", '{"level": 8.5}') == {("level", "8.5"): "leak"}
    summary = world.surface.summary()
    assert summary["typed_columns"] >= 6 and summary["values_in_typed_columns"] > 80


def test_without_learning_the_public_surface_it_would_be_flagged(world: World) -> None:
    unlearned = Scanner(world.inventory.facts)
    assert surface_hits(unlearned, world.surface)
    result = selftest(unlearned, world.surface, world.inventory)
    assert not result["passed"] and result["public_surface"]["count"] > 0


def test_a_session_that_repeats_public_data_is_not_flagged(world: World) -> None:
    rows = [
        json.dumps(r, default=str)
        for r in pq.read_table(world.export / "log/episodes.parquet").to_pylist()
    ]
    steps = pq.read_table(world.export / "log/steps/e000001.parquet").to_pylist()
    answer = {"columns": ["step", "time_s"], "rows": [[r["step"], r["time_s"]] for r in steps]}
    text = (
        "\n".join(rows) + json.dumps(answer) + (world.export / "log/description.json").read_text()
    )
    for place in PLACES:
        assert world.found(place, text) == {}


def test_a_truth_value_in_the_export_leaves_the_inventory_and_is_listed(tmp_path: Path) -> None:
    world = World(tmp_path, extra="the loss was 5.539 points; trait gripper_pad_age_days")
    shown = {(e["kind"], e["value"]): e for e in world.overlap["entries"] if e["beside"] is None}
    assert shown[("expected_value", "5.539")]["shown_in"] == ["export log/description.json"]
    assert ("trait_name", TRAIT) in shown
    assert surface_hits(world.scanner, world.surface) == []
    assert world.found("answer", "it comes to 5.539 here") == {}
    found = world.found("tool result", json.dumps({"expected_loss_points": 5.539}))
    assert found[("expected_value", "5.539")] == "leak"


def test_one_finding_per_fact_and_session(world: World) -> None:
    seed = str(world.seed)
    run, state = world.session("answer", f"the seed {seed}, again {seed}")
    (state / "memory/notes.md").write_text(f"seed {seed}")
    call = {"seq": 1, "tool": "query", "arguments": {}, "result": [seed, seed]}
    (run / "box/toolbox/tool_calls.jsonl").write_text(json.dumps(call) + "\n")
    scan = scan_session(world.scanner, run, state)
    assert list(scan.findings) == [("world_seed", seed)]
    finding = scan.findings[("world_seed", seed)]
    assert finding.places == {"answer": 2, "tool result": 2, "memory note": 1}
    assert len(finding.contexts) == 3
    data = scan.as_json(listed=0)
    assert data["findings"] == [] and data["findings_total"] == 1
    assert data["findings_by_kind"] == {"world_seed": {"leak": 1, "to_judge": 0}}


def test_the_hosts_own_measurements_are_not_session_output(world: World) -> None:
    run, state = world.session("answer", "")
    record = json.loads((run / "result.json").read_text())
    record.update(seconds=7.25, usage={"input_tokens": 149.8}, meta={"api_ms": 5.539})
    record["run_dir"] = "/srv/worlds/w98/calls"
    (run / "result.json").write_text(json.dumps(record))
    call = {
        "seq": 1,
        "tool": "query",
        "arguments": {},
        "result": ["{}"],
        "seconds": 7.25,
        "at": 149.8,
    }
    (run / "box/toolbox/tool_calls.jsonl").write_text(json.dumps(call) + "\n")
    events = [
        {"type": "system", "subtype": "init", "cwd": "/srv/w98", "n": 7.25},
        {"type": "rate_limit_event", "rate_limit_info": {"utilization": 0.6921}},
        {"type": "turn.completed", "usage": {"input_tokens": 5.539}},
        {"type": "result", "result": "", "duration_ms": 149.8, "total_cost_usd": 7.25},
    ]
    (run / "stdout.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    for name in ("argv.json", "prompt.json"):
        (run / name).write_text(json.dumps([str(world.seed), "w98"]))
    (state / "analysis.json").write_text(json.dumps({"world_id": "w98"}))
    scan = scan_session(world.scanner, run, state)
    assert scan.findings == {} and scan.problems == []
    assert {
        f["read_as"] for f in scan.files if f["path"].endswith(("argv.json", "analysis.json"))
    } == {"host file, not session output"}


def test_an_echo_of_what_the_session_wrote_is_not_what_a_tool_returned(world: World) -> None:
    run, state = world.session("answer", "")
    log = run / "box/toolbox/tool_calls.jsonl"

    def calls(
        *items: tuple[dict[str, Any], str], names: bool = False
    ) -> dict[tuple[str, str], str]:
        """What the scan finds; findings of truth-side names only where `names` asks."""
        lines = [
            json.dumps({"seq": n, "tool": "query", "arguments": a, "result": [r]})
            for n, (a, r) in enumerate(items, 1)
        ]
        log.write_text("\n".join(lines) + "\n")
        scan = scan_session(world.scanner, run, state)
        return {
            k: f"{f.verdict} {sorted(f.places)}"
            for k, f in scan.findings.items()
            if names == (k[0] == "truth_field")
        }

    guess = {"sql": "SELECT 3 AS onset_day"}
    echoed = calls((guess, '{"columns": ["onset_day"], "rows": [[3]]}'))
    assert echoed == {("onset_day", "3"): "to_judge ['tool arguments']"}
    # the same reply without the session having written it: the tool returned it
    assert calls(({"sql": "SELECT * FROM t"}, '{"onset_day": 3}')) == {
        ("onset_day", "3"): "leak ['tool result']"
    }
    # a bare 3 in an earlier query is not the same evidence
    assert calls(({"sql": "SELECT 3"}, "{}"), ({"sql": "SELECT * FROM t"}, '{"onset_day": 3}')) == {
        ("onset_day", "3"): "leak ['tool result']"
    }
    # a truth-side column name the session tried, echoed in the error
    tried = ({"sql": "SELECT scene_seed FROM episodes"}, 'no column "scene_seed"')
    assert calls(tried, names=True) == {}
    assert calls(({"sql": "SELECT 1"}, 'did you mean "scene_seed"?'), names=True) == {
        ("truth_field", "scene_seed"): "leak ['tool result']"
    }
    # a seed the session wrote into a query is a leak where it wrote it
    seed = str(world.seed)
    assert calls(({"sql": f"SELECT {seed}"}, f'{{"rows": [[{seed}]]}}')) == {
        ("world_seed", seed): "leak ['tool arguments']"
    }


def tool_log(world: World, *calls: dict[str, Any]) -> Any:
    """The scan of a session whose tool log holds `calls`."""
    run, state = world.session("answer", "")
    lines = [json.dumps({"seq": n, "ok": True, **call}) for n, call in enumerate(calls, 1)]
    (run / "box/toolbox/tool_calls.jsonl").write_text("\n".join(lines) + "\n")
    return scan_session(world.scanner, run, state)


def test_an_earlier_guess_does_not_excuse_a_tool_that_later_returns_the_value(
    world: World,
) -> None:
    # Review 2, point 5: only the request a reply answers makes the reply an echo.
    guess: dict[str, Any] = {"tool": "query", "arguments": {"sql": "SELECT 3 AS onset_day"}}
    guess["result"] = ['{"columns": ["onset_day"], "rows": [[3]]}']
    scan = tool_log(world, guess)
    assert scan.findings[("onset_day", "3")].verdict == "to_judge"
    assert sorted(scan.findings[("onset_day", "3")].places) == ["tool arguments"]
    told = {"tool": "describe", "arguments": {"part": "fleet"}, "result": ['{"onset_day": 3}']}
    for order in ([guess, told], [told, guess]):
        finding = tool_log(world, *order).findings[("onset_day", "3")]
        assert finding.verdict == "leak" and finding.contexts[0]["place"] == "tool result"
        assert "(describe)" in finding.contexts[0]["where"]
    # the same guess in the same words by another call is not this call's request either
    again = {"tool": "query", "arguments": {"sql": "SELECT 1"}, "result": guess["result"]}
    assert tool_log(world, guess, again).findings[("onset_day", "3")].verdict == "leak"


def test_a_guess_written_to_memory_does_not_excuse_what_a_tool_returns_later(
    world: World,
) -> None:
    # Review 3, case 3: 42 is an oracle value no public data shows. The session writes
    # "my guess: 42" into memory; later experiment_results returns 42.
    fact = Fact("oracle_value", "42", True, ("replay_score", "oracle"), "synthetic")
    scanner = Scanner([fact])
    scanner.learn_public(Surface())

    def scanned(*calls: dict[str, Any], note: str = "") -> Any:
        run, state = world.session("memory note", note)
        lines = [json.dumps({"seq": n, "ok": True, **call}) for n, call in enumerate(calls, 1)]
        (run / "box/toolbox/tool_calls.jsonl").write_text("\n".join(lines) + "\n")
        return scan_session(scanner, run, state)

    guess = {"tool": "memory_write", "arguments": {"text": "my guess: 42"}, "result": ["stored"]}
    reply = {"tool": "experiment_results", "arguments": {}, "result": ["42"]}
    scan = scanned(guess, reply)
    assert scan.problems == [] and scan.findings[fact.key].verdict == "to_judge"
    assert scan.findings[fact.key].places == {"tool result": 1}
    record = clean_record()
    _route("bulk-data", 1, scan=scan.as_json(), tool_call_count=2)(record)
    finish(record)
    assert record["verdict"] == "to_judge" and record["found"] == {"leak": 0, "to_judge": 1}
    # ... whatever the session wrote before, in any place it can write, and whichever
    # tool returns it afterwards
    asked = {"tool": "request_experiment", "arguments": {"power_note": "42"}, "result": ["ok"]}
    sent = {"tool": "submit_diagnosis", "arguments": {"summary": "42?"}, "result": ["ok"]}
    for wrote in (guess, asked, sent):
        for tool in ("experiment_results", "memory_read", "query", "describe"):
            scan = scanned(wrote, {**reply, "tool": tool})
            assert scan.findings[fact.key].verdict == "to_judge", (wrote["tool"], tool)
    for tool in ("experiment_results", "memory_read"):
        scan = scanned({**reply, "tool": tool}, note="my guess: 42")  # an earlier session's note
        assert scan.findings[fact.key].places == {"tool result": 1}, tool
    # beside its hidden name it is what a tool returned beside its hidden name
    told = {"tool": "memory_read", "arguments": {}, "result": ['{"replay_score": 42}']}
    named = {"tool": "memory_write", "arguments": {"text": "replay_score 42?"}, "result": ["ok"]}
    assert scanned(named, told).findings[fact.key].verdict == "leak"
    # the echo that is left is the request of the same call
    same = {"tool": "query", "arguments": {"sql": "SELECT 42 AS replay_score"}}
    scan = scanned({**same, "result": ['{"columns": ["replay_score"], "rows": [[42]]}']})
    assert scan.findings[fact.key].places == {"tool arguments": 1}


def test_a_short_hidden_number_a_tool_returns_alone_is_listed_to_judge() -> None:
    # Review 2, point 5: 42 is an oracle value no public data shows. A tool returned it.
    fact = Fact("oracle_value", "42", True, ("replay_score", "oracle"), "oracle/oracle.json")
    word = Fact("fault_role", "decoy", False, ("f07",), "files/generator.json")
    scanner = Scanner([fact, word])
    overlap = scanner.learn_public(Surface())
    assert overlap["entries"] == [] and scanner.bare(fact) and not scanner.alone(fact)

    def returned(text: str, place: str = "tool result") -> dict[tuple[str, str], str]:
        scan = scan_texts(scanner, [("x", text)], place)
        return {key: f"{f.verdict}: {f.reason}" for key, f in scan.findings.items()}

    found = returned("[[42]]")
    assert list(found) == [("oracle_value", "42")] and found[fact.key].startswith("to_judge")
    assert "returned this number of one or two digits alone" in found[fact.key]
    assert returned("a decoy")[word.key].startswith("to_judge: a tool returned this ordinary word")
    assert returned('{"replay_score": 42}')[fact.key].startswith("leak")
    for place in ("experiment reply", "sandbox output"):
        assert returned("42", place)[fact.key].startswith("to_judge"), place
    # written by the session itself it is not recognised, and the record's scope says so
    for place in ("answer", "model text", "memory note", "tool arguments", "file"):
        assert returned("it is 42, a decoy", place) == {}, place
    # where the public data shows 42 it is one of thousands, and listed as that
    shown = Surface()
    shown.add("table steps rows 0-99", '{"step": 42}')
    overlap = scanner.learn_public(shown)
    assert [e["status"] for e in overlap["entries"]] == ["says nothing alone"]
    assert returned("[[42]]") == {} and returned('{"replay_score": 42}')[fact.key][:4] == "leak"


def test_a_short_value_among_the_bytes_of_a_payload_is_no_finding(world: World) -> None:
    # A frame comes back as some hundred thousand characters of base64. "T9" and "decoy"
    # between two signs are in there by chance; a seed or a canary is not.
    noise = "QmFzZTY0" * 20
    assert world.found("tool result", "it says /T9+decoy/ here") == {
        ("generator_parameter", "T9"): "to_judge",
        ("fault_role", "decoy"): "to_judge",
    }
    frame: dict[str, Any] = {"type": "image", "source": {"type": "base64"}}
    frame["source"]["data"] = f"{noise}/T9+decoy/{noise}"
    assert world.found("tool result", json.dumps(frame)) == {}
    frame["source"]["data"] = f"{noise}/{world.seed}+{noise}"
    assert world.found("tool result", json.dumps(frame)) == {
        ("world_seed", str(world.seed)): "leak"
    }
    frame["source"]["data"] = f"{noise}/f07+{noise}"
    assert world.found("tool result", json.dumps(frame)) == {}
    # a run that is shorter is text, and so is what stands outside the run
    short = {"path": f"{noise[:150]}/decoy/x", "note": f"{noise}{noise} decoy"}
    assert world.found("tool result", json.dumps(short["path"])) == {
        ("fault_role", "decoy"): "to_judge"
    }
    assert world.found("tool result", json.dumps(short["note"])) == {
        ("fault_role", "decoy"): "to_judge"
    }
    # ... also in a long text: 147 characters of a path are not a payload
    path = "the file " * 30 + f"{noise[:120]}/decoy/{noise[:20]}"
    assert world.found("tool result", path) == {("fault_role", "decoy"): "to_judge"}


def test_a_tool_s_error_text_is_read_as_what_the_tool_returned(world: World) -> None:
    # Review 2, point 7: an error is returned to the session like a result is.
    failed = {"tool": "query", "arguments": {"sql": "SELECT 1"}, "ok": False, "result": None}
    scan = tool_log(world, {**failed, "error": f"no such value: {world.seed}"})
    assert scan.findings[("world_seed", str(world.seed))].places == {"tool result": 1}
    scan = tool_log(world, {**failed, "error": {"message": "hint", "detail": {"level": 7.25}}})
    assert scan.findings[("level", "7.25")].verdict == "leak"
    assert tool_log(world, {**failed, "error": "no such column"}).findings == {}


def stream(world: World, *events: dict[str, Any]) -> Any:
    """The scan of a session whose CLI stream holds `events`."""
    run, state = world.session("answer", "")
    (run / "stdout.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return scan_session(world.scanner, run, state)


def test_what_a_tool_returned_in_codex_s_stream_is_read_as_returned(world: World) -> None:
    # Review 2, point 7: the text of an MCP response, its other fields, and its error.
    def call(**fields: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"id": "item_3", "type": "mcp_tool_call", "tool": "query"}
        item.update(server="robots", arguments={"sql": "SELECT 1"}, status="completed", **fields)
        return {"type": "item.completed", "item": item}

    text = [{"type": "text", "text": '{"level": 7.25}'}]
    scan = stream(world, call(result={"content": text, "structured_content": None}))
    finding = scan.findings[("level", "7.25")]
    assert finding.verdict == "leak" and finding.places == {"tool result": 1}
    scan = stream(world, call(result={"content": [], "structured_content": {"level": 7.25}}))
    assert scan.findings[("level", "7.25")].verdict == "leak"
    scan = stream(world, call(result=None, error={"message": f"seed {world.seed}"}))
    assert scan.findings[("world_seed", str(world.seed))].places == {"tool result": 1}
    # what the request held is an echo here as well, for this call only
    asked = call(result={"content": [{"type": "text", "text": '{"onset_day": 3}'}]})
    asked["item"]["arguments"] = {"sql": "SELECT 3 AS onset_day"}
    assert stream(world, asked).findings[("onset_day", "3")].verdict == "to_judge"
    other = call(result={"content": [{"type": "text", "text": '{"onset_day": 3}'}]})
    assert stream(world, asked, other).findings[("onset_day", "3")].verdict == "leak"
    assert scan.as_json()["records_read"]["events in the stream"] == 1


def test_a_tool_result_in_claude_s_stream_answers_the_call_with_its_id(world: World) -> None:
    def use(number: int, sql: str) -> dict[str, Any]:
        block = {"type": "tool_use", "id": f"toolu_{number}", "name": "mcp__robots__query"}
        return {"type": "assistant", "message": {"content": [{**block, "input": {"sql": sql}}]}}

    def result(number: int, text: str) -> dict[str, Any]:
        block = {"type": "tool_result", "tool_use_id": f"toolu_{number}"}
        content = [{"type": "text", "text": text}]
        return {"type": "user", "message": {"content": [{**block, "content": content}]}}

    reply = '{"columns": ["onset_day"], "rows": [[3]]}'
    scan = stream(world, use(1, "SELECT 3 AS onset_day"), result(1, reply))
    assert scan.findings[("onset_day", "3")].verdict == "to_judge"
    # two calls in flight: each result is held against its own request
    events = [
        use(1, "SELECT 3 AS onset_day"),
        use(2, "SELECT 1"),
        result(2, reply),
        result(1, reply),
    ]
    finding = stream(world, *events).findings[("onset_day", "3")]
    assert finding.verdict == "leak" and finding.places == {"tool arguments": 1, "tool result": 1}
    # a result that answers no call the stream holds is nobody's echo
    assert stream(world, result(9, reply)).findings[("onset_day", "3")].verdict == "leak"
    # what the session wrote into memory before is not the request of the call that reads
    write = use(1, "x")
    write["message"]["content"][0].update(name="mcp__robots__memory_write", input={"text": reply})
    read = use(2, "x")
    read["message"]["content"][0].update(name="mcp__robots__memory_read", input={})
    scan = stream(world, write, read, result(2, reply))
    assert scan.findings[("onset_day", "3")].verdict == "leak"


# Review 2, point 4: a record that holds nothing is not a session that was scanned.


@pytest.mark.parametrize(
    ("name", "content", "words"),
    [
        ("result.json", "{}", "the result record holds no answer and no status"),
        ("result.json", '""', "not the runtime's result record"),
        ("result.json", "", "not the runtime's result record"),
        ("result.json", '{"ok": true}', "holds no answer and no status"),
        ("result.json", '{"result": {"summary": ""}}', "holds no answer and no status"),
        ("stdout.jsonl", "", "the CLI's event stream holds no event"),
        ("stdout.jsonl", "\n\n", "the CLI's event stream holds no event"),
        ("stdout.jsonl", '""\n', "is not an event"),
    ],
)
def test_a_transcript_that_holds_nothing_was_not_scanned(
    world: World, name: str, content: str, words: str
) -> None:
    run, state = world.session("answer", "nothing obtained")
    assert scan_session(world.scanner, run, state).verdict == "nothing_found"
    (run / name).write_text(content)
    scan = scan_session(world.scanner, run, state)
    assert any(words in problem for problem in scan.problems), scan.problems
    assert scan.verdict == "incomplete"
    record = clean_record()
    _route("bulk-data", 1, scan={**scan.as_json(), "records_read": {"tool calls in the log": 3}})(
        record
    )
    finish(record)
    assert record["verdict"] == "incomplete"


def test_records_that_are_empty_objects_are_not_a_scanned_session(world: World) -> None:
    # Review 3, case 2: `{}` for the tool call and for the event, a null answer, and the
    # counts of calls agree with each other.
    run, state = world.session("answer", "")
    good = {name: (run / name).read_text() for name in ("result.json", "stdout.jsonl")}
    good["box/toolbox/tool_calls.jsonl"] = (run / "box/toolbox/tool_calls.jsonl").read_text()
    empty = {
        "result.json": json.dumps({"ok": True, "result": None, "tool_log": [{}]}),
        "stdout.jsonl": "{}\n",
        "box/toolbox/tool_calls.jsonl": "{}\n",
    }
    for name, content in empty.items():
        (run / name).write_text(content)
    scan = scan_session(world.scanner, run, state)
    assert scan.verdict == "incomplete"
    assert not scan.records.get("tool calls in the log")
    assert not scan.records.get("events in the stream")
    for words in (
        "tool_calls.jsonl: line 1 is not a tool call record",
        "stdout.jsonl: line 1 is not an event",
        "the CLI's event stream holds no event",
        "result.json: the result record holds no answer",
        "result.json tool_log: line 1 is not a tool call record",
    ):
        assert any(words in problem for problem in scan.problems), (words, scan.problems)
    record = clean_record()
    calls = scan.records.get("tool calls in the log", 1)  # as the session record would say
    _route("bulk-data", 1, scan=scan.as_json(), tool_call_count=calls)(record)
    finish(record)
    assert record["verdict"] == "incomplete"
    # each of the three alone is enough
    for name, content in empty.items():
        for other, text in good.items():
            (run / other).write_text(text)
        (run / name).write_text(content)
        assert scan_session(world.scanner, run, state).verdict == "incomplete", name
    for other, text in good.items():
        (run / other).write_text(text)
    assert scan_session(world.scanner, run, state).problems == []
    # a call needs a tool and what came back, an event its type, a failure its reason
    log = run / "box/toolbox/tool_calls.jsonl"
    for call in ({"tool": "query"}, {"result": ["x"]}, {"tool": "", "result": []}, {"tool": 3}):
        log.write_text(json.dumps(call) + "\n")
        assert scan_session(world.scanner, run, state).problems, call
    log.write_text(good["box/toolbox/tool_calls.jsonl"])
    for event in ({"item": {"type": "agent_message"}}, {"type": ""}, {"type": None}):
        (run / "stdout.jsonl").write_text(json.dumps(event) + "\n")
        assert scan_session(world.scanner, run, state).problems, event
    (run / "stdout.jsonl").write_text(good["stdout.jsonl"])
    for answer, fine in (
        ({"ok": True, "result": None, "text": ""}, False),
        ({"ok": True, "result": None, "text": "nothing obtained"}, True),
        ({"ok": False, "result": None, "error": "malformed_output"}, True),
        ({"ok": False, "result": None, "error": None}, False),
        ({"ok": "yes", "result": {"summary": ""}}, False),
    ):
        (run / "result.json").write_text(json.dumps(answer))
        assert (scan_session(world.scanner, run, state).problems == []) == fine, answer


def test_the_scan_says_what_the_records_held(world: World) -> None:
    run, state = world.session("answer", "nothing obtained")
    call = {"seq": 1, "tool": "query", "arguments": {"sql": "SELECT 1"}, "ok": True, "result": []}
    record = json.loads((run / "result.json").read_text())

    def scanned(logged: int, listed: int | None) -> Any:
        (run / "box/toolbox/tool_calls.jsonl").write_text((json.dumps(call) + "\n") * logged)
        stored_record = dict(record) if listed is None else {**record, "tool_log": [call] * listed}
        (run / "result.json").write_text(json.dumps(stored_record))
        return scan_session(world.scanner, run, state)

    scan = scanned(2, 2)
    assert scan.problems == [] and scan.as_json()["records_read"] == {
        "events in the stream": 1,
        "tool calls in the log": 2,
        "tool calls in the result record": 2,
    }
    assert scanned(2, None).problems == []  # a record without the runtime's copy
    # an emptied tool log beside a result record that lists calls
    assert scanned(0, 2).problems == [
        "run: the result record lists 2 tool calls, the tool log holds 0"
    ]
    assert scanned(3, 2).problems == [
        "run: the result record lists 2 tool calls, the tool log holds 3"
    ]
    # a second file of that name, wherever the session put it, counts on top
    (run / "box/cwd").mkdir()
    (run / "box/cwd/tool_calls.jsonl").write_text(json.dumps(call) + "\n")
    assert scanned(2, 2).problems == [
        "run: the result record lists 2 tool calls, the tool log holds 3"
    ]
    # and the verdict holds the number the scan read against the number the session made
    record_ = clean_record()
    _route("bulk-data", 1, scan=scan.as_json(), tool_call_count=2)(record_)
    finish(record_)
    assert record_["verdict"] == "no_leak_detected"
    _route("bulk-data", 1, tool_call_count=5)(record_)
    finish(record_)
    assert record_["blockers"] == [
        "the session bulk-data/1 made 5 tool calls and the scan read 2 in its tool log"
    ]


def test_a_stored_route_s_comparison_is_made_again_from_what_it_derived(
    world: World, tmp_path: Path
) -> None:
    row = dict(episode(world))
    (tmp_path / "probe").mkdir()
    assert probes.compare_again("identifier", tmp_path, world.runs) is None  # it left nothing
    assert probes.compare_again("identifier-query", tmp_path, world.runs) is None
    (tmp_path / "obtained.json").write_text(json.dumps({"derived": [row, {**row, "day": 9}]}))
    again = probes.compare_again("identifier", tmp_path, world.runs)
    assert again is not None and again["verdict"] == "leak" and again["compared_episodes"] == 2
    through = {"episode_id": row["episode_id"]}
    through.update(computed_a=row["scene_seed"], computed_b=row["noise_seed"])
    (tmp_path / "probe/obtained.json").write_text(json.dumps([through]))
    again = probes.compare_again("identifier-query", tmp_path, world.runs)
    assert again is not None and again["matched"] == {"scene_seed": 1, "noise_seed": 1}
    through.update(computed_a=row["noise_seed"], computed_b=row["scene_seed"])
    (tmp_path / "probe/obtained.json").write_text(json.dumps([through]))
    again = probes.compare_again("identifier-query", tmp_path, world.runs)
    assert again is not None and again["verdict"] == "no_leak_detected"
    for damaged in ("{}", '{"derived": 3}', '{"derived": [3]}', "not JSON"):
        (tmp_path / "obtained.json").write_text(damaged)
        assert probes.compare_again("identifier", tmp_path, world.runs) is None, damaged


def test_a_route_that_compared_no_seed_with_its_true_value_has_not_passed(world: World) -> None:
    truth = probes._identities(world.runs)
    row = dict(episode(world))
    keys = {"scene_seed": "scene_seed", "noise_seed": "noise_seed"}

    def compared(found: list[dict[str, Any]], known: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        probes._compare(record, found, known, keys)
        return record

    # seeds derived for episodes the stored identities do not hold: right or wrong, no
    # seed was held against its true value
    other = compared([{**row, "episode_id": "e999999"}], truth)
    assert other["verdict"] == "nothing_compared" and "not_applicable" not in other
    assert other["compared_episodes"] == 0 and other["attempted_episodes"] == 1
    assert "none of the 1 episodes" in other["reason"]
    assert other["matched"] == {"scene_seed": 0, "noise_seed": 0}
    right = compared([row, {**row, "episode_id": "e999999"}], truth)
    assert right["verdict"] == "leak" and right["compared_episodes"] == 1
    assert right["matched"] == {"scene_seed": 1, "noise_seed": 1}
    wrong = compared([{**row, "scene_seed": 1, "noise_seed": 2}], truth)
    assert wrong["verdict"] == "no_leak_detected" and wrong["compared_episodes"] == 1
    nothing = compared([], truth)
    assert nothing["verdict"] == "nothing_compared" and nothing["not_applicable"]
    assert compared([row], {})["reason"] == "no stored identities to compare with"
    for record in (other, nothing):
        clean = clean_record()
        _route("identifier-derived-seeds", None, **record)(clean)
        finish(clean)
        assert (clean["verdict"] == "no_leak_detected") == bool(record.get("not_applicable"))


# Review 2, point 2: what the public surface shows of the truth is public by a rule
# that is safe, or because a person accepted it, on record. Everything else is open.


def by_entry(overlap: dict[str, Any], status: str | None = None) -> dict[Any, dict[str, Any]]:
    return {
        (e["kind"], e["value"], e["beside"]): e
        for e in overlap["entries"]
        if status is None or e["status"] == status
    }


def test_a_hidden_value_disclosed_in_a_brief_does_not_become_public_and_disappear(
    world: World,
) -> None:
    brief = {"brief": "Check whether expected_loss_points: 5.539 can be had from the tools."}
    told = public_surface(world.export, None, brief)
    scanner = Scanner(world.inventory.facts)
    overlap = scanner.learn_public(told)
    open_ones = by_entry(overlap, "unresolved")
    beside = open_ones[("expected_value", "5.539", "expectedlosspoints")]
    assert beside["verdict"] == "leak" and beside["shown_in"] == ["brief"]
    assert open_ones[("expected_value", "5.539", None)]["verdict"] == "to_judge"
    assert open_ones[("truth_field", "expected_loss_points", None)]["verdict"] == "to_judge"
    # the scan's self-test still passes, and a session that repeats the brief is not flagged:
    assert surface_hits(scanner, told) == [] and selftest(scanner, told, world.inventory)["passed"]
    assert scanner.scan('{"expected_loss_points": 5.539}') == []
    # ... so it is the audit that must not be clean
    record = clean_record()
    record["public_surface"]["overlaps"] = main_module.overlap_record(
        main_module.Evidence(world.inventory, told, scanner, overlap)
    )
    finish(record)
    assert record["verdict"] == "leak" and record["found"]["leak"] >= 1
    assert [i for i in record["leaks"] if i["route"] == "public-surface" and i["value"] == "5.539"]
    # without the disclosure none of the three is there
    assert not [key for key in by_entry(world.overlap) if "5.539" in key or "expected" in key[1]]


def test_public_values_are_innocent_by_rule_only_where_the_rule_is_safe(tmp_path: Path) -> None:
    world = World(tmp_path, extra="since_day: 3, and onset_day: 3; fault_levels; r02 plain")
    entries = by_entry(world.overlap)
    # a number of one or two digits without its name: the public data is full of them
    assert entries[("onset_day", "3", None)]["status"] == "says nothing alone"
    assert entries[("fault_role", "plain", None)]["status"] == "says nothing alone"
    # beside a field the contracts declare public, it is that field's own value
    assert {"sinceday", "episodeid", "sql"} <= world.surface.schema_names == declared_names()
    assert entries[("onset_day", "3", "sinceday")]["status"] == "beside a public field of that name"
    # beside its hidden name it is the hidden value, wherever it stands
    hidden = entries[("onset_day", "3", "onsetday")]
    assert (hidden["status"], hidden["verdict"]) == ("unresolved", "leak")
    assert "onsetday" not in declared_names()
    # a number of three digits that a table also holds: for a person
    level = entries[("level", "3.05", None)]
    assert (level["status"], level["verdict"], level["shown_in"]) == (
        "unresolved",
        "to_judge",
        ["table steps"],
    )
    # a truth-side name: public where the export's own data shows it ...
    assert entries[("truth_field", "fault_levels", None)]["status"] == "a public name as well"
    # ... not there at all where the public surface has only its words ...
    facts = world.inventory.facts

    def shown(texts: dict[str, str], kind: str, value: str) -> dict[str, Any]:
        surface = public_surface(world.export, None, texts)
        return by_entry(Scanner(facts).learn_public(surface))[(kind, value, None)]

    words = shown({"brief": "The fault schedule and levels."}, "truth_field", "fault_schedule")
    assert (words["status"], words["shown_in"]) == ("its words, not the name", ["brief"])
    # ... and open where only a brief or a skill shows it in its own spelling
    asked = {"brief": "is there a fault_schedule table?"}
    name = shown(asked, "truth_field", "fault_schedule")
    assert (name["status"], name["verdict"], name["shown_in"]) == (
        "unresolved",
        "to_judge",
        ["brief"],
    )
    name = shown(
        {**asked, "skill x/SKILL.md": "A fault schedule."}, "truth_field", "fault_schedule"
    )
    assert (name["status"], name["shown_in"]) == ("unresolved", ["brief"])
    # a fault's kind in words is for a person; in its own spelling it convicts
    kind = shown({"brief": "Look for a toy camera wobble."}, "fault_kind", KIND)
    assert (kind["status"], kind["verdict"]) == ("unresolved", "to_judge")
    kind = shown({"brief": f"Look for {KIND}."}, "fault_kind", KIND)
    assert (kind["status"], kind["verdict"]) == ("unresolved", "leak")
    assert world.overlap["by_status"] == {
        status: sum(e["status"] == status for e in world.overlap["entries"])
        for status in sorted({e["status"] for e in world.overlap["entries"]})
    }


def test_a_column_the_export_holds_by_mistake_resolves_nothing(tmp_path: Path) -> None:
    world = World(tmp_path)
    path = world.export / "log/episodes.parquet"
    table = pq.read_table(path)
    leaked = table.append_column("onset_day", pa.array([3] * table.num_rows, pa.int64()))
    pq.write_table(leaked, path)
    surface = public_surface(world.export, None, {})
    overlap = Scanner(world.inventory.facts).learn_public(surface)
    entry = by_entry(overlap)[("onset_day", "3", "onsetday")]
    assert (entry["status"], entry["verdict"]) == ("unresolved", "leak")
    assert entry["shown_in"] == ["table episodes"]
    # the name itself the export shows, and a name holds no value; its values convict
    assert (
        by_entry(overlap)[("truth_field", "onset_day", None)]["status"] == "a public name as well"
    )


def test_an_acceptance_of_a_rounded_value_does_not_cover_the_exact_one() -> None:
    # Review 3, case 1. A brief shows the expected loss rounded beside its hidden key, and
    # a person accepts that. Later the brief shows the exact value in the same place.
    fact = Fact("expected_value", "5.539", True, ("expected_loss_points",), "synthetic")
    scanner = Scanner([fact])
    before = Surface()
    before.add("brief", "expected_loss_points: 5.5")
    proposal = scanner.learn_public(before)["to_accept"]
    assert [(e["beside"], e["form"]) for e in proposal] == [("expectedlosspoints", "rounded")]
    accept = [{**e, "accepted_by": "a reviewer", "reason": "rounded only"} for e in proposal]
    assert scanner.learn_public(before, accept)["unresolved"] == []  # what it names
    after = Surface()
    after.add("brief", "expected_loss_points: 5.539")
    overlap = scanner.learn_public(after, accept)
    pair = by_entry(overlap)[("expected_value", "5.539", "expectedlosspoints")]
    assert (pair["form"], pair["status"], pair["verdict"]) == ("exact", "unresolved", "leak")
    assert overlap["by_status"] == {"unresolved": 2}
    record = clean_record()
    record["public_surface"]["overlaps"] = overlap
    finish(record)
    assert record["verdict"] == "leak" and record["found"] == {"leak": 1, "to_judge": 1}
    # the other way round an acceptance of the exact value does not cover a rounded one
    exact = [{**e, "accepted_by": "a reviewer", "reason": "x"} for e in overlap["to_accept"]]
    assert sorted(e["form"] for e in exact) == ["exact", "exact"]
    assert scanner.learn_public(after, exact)["unresolved"] == []
    assert [e["form"] for e in scanner.learn_public(before, exact)["unresolved"]] == ["rounded"]
    # An entry that does not say which form, or that names something the audit does not
    # know, is refused: it resolves nothing, and the refusal blocks.
    for change in ({"form": None}, {"form": "any"}, {"only_in_session": 2}):
        entry = {**accept[0], **change}
        if change.get("form", "") is None:
            del entry["form"]
        refused = scanner.learn_public(before, [entry])
        assert len(refused["unresolved"]) == 1 and len(refused["acceptance_problems"]) == 1, change
    unbound = {k: v for k, v in accept[0].items() if k != "beside"}
    assert len(scanner.learn_public(before, [unbound])["acceptance_problems"]) == 1
    # the same for a name: its words accepted are not the name accepted
    kind = Scanner([Fact("fault_kind", KIND, False, ("f07",), "synthetic")])
    words = Surface()
    words.add("brief", "Look for a toy camera wobble.")
    proposal = kind.learn_public(words)["to_accept"]
    assert [e["form"] for e in proposal] == ["in words"]
    signed = [{**e, "accepted_by": "a reviewer", "reason": "prose"} for e in proposal]
    assert kind.learn_public(words, signed)["unresolved"] == []
    spelled = Surface()
    spelled.add("brief", f"Look for {KIND}.")
    (name,) = kind.learn_public(spelled, signed)["unresolved"]
    assert (name["form"], name["verdict"]) == ("exact", "leak")


def test_an_acceptance_resolves_exactly_what_a_person_named(world: World, tmp_path: Path) -> None:
    open_ones = world.overlap["unresolved"]
    proposal = world.overlap["to_accept"]
    assert open_ones and sum(len(e["values"]) for e in proposal) == len(open_ones)
    assert all(e["accepted_by"] == "" and e["reason"] == "" for e in proposal)
    scanner = Scanner(world.inventory.facts)
    # the proposal as it is written out, with nobody's name: nothing is accepted
    overlap = scanner.learn_public(world.surface, proposal)
    assert len(overlap["unresolved"]) == len(open_ones)
    assert len(overlap["acceptance_problems"]) == len(proposal)
    for missing in ("accepted_by", "reason"):
        partly = [
            {**e, "accepted_by": "cs", "reason": "step times", missing: " "} for e in proposal
        ]
        assert len(scanner.learn_public(world.surface, partly)["unresolved"]) == len(open_ones)
    signed = [{**e, "accepted_by": "cs", "reason": "step times of the export"} for e in proposal]
    overlap = scanner.learn_public(world.surface, signed)
    assert overlap["unresolved"] == [] and overlap["acceptance_problems"] == []
    assert overlap["by_status"]["accepted"] == len(open_ones) and overlap["to_accept"] == []
    accepted = by_entry(overlap, "accepted")[("level", "3.05", None)]
    assert accepted["accepted_by"] == "cs" and accepted["reason"] == "step times of the export"
    # shown in a place the acceptance does not name: open again
    narrowed = [{**e, "shown_in": e["shown_in"][1:]} for e in signed]
    assert len(scanner.learn_public(world.surface, narrowed)["unresolved"]) == len(open_ones)
    told = public_surface(world.export, None, {"brief": "a level of 3.05 was planted"})
    reopened = by_entry(scanner.learn_public(told, signed), "unresolved")
    assert reopened[("level", "3.05", None)]["shown_in"] == ["brief", "table steps"]
    # the value beside a hidden name is another entry and needs its own acceptance
    assert ("level", "3.05", "level") in reopened
    # another kind or another value is not what was accepted
    other = [{**e, "kind": "trait_value"} for e in signed]
    assert len(scanner.learn_public(world.surface, other)["unresolved"]) == len(open_ones)
    # a seed or a canary is never public, whoever signs
    seeded = World(tmp_path, extra=f"generated from {SEED}")
    entry = {"kind": "world_seed", "values": [str(SEED)], "beside": None, "reason": "x"}
    entry.update(shown_in=["export log/description.json"], accepted_by="cs", form="exact")
    overlap = Scanner(seeded.inventory.facts).learn_public(seeded.surface, [entry, "junk"])
    assert overlap["acceptance_problems"] == [
        "acceptance 1: a world_seed is never public",
        "acceptance 2: needs kind, values, beside, form, shown_in, accepted_by and reason",
    ]
    seed = by_entry(overlap, "unresolved")[("world_seed", str(SEED), None)]
    assert seed["verdict"] == "leak"
    # the file a person hands in
    assert main_module.acceptances(None) == []
    path = accept_file(tmp_path / "accept.json", world.overlap)
    assert main_module.acceptances(path) == json.loads(path.read_text())["accept"]
    path.write_text(json.dumps(signed))
    assert main_module.acceptances(path) == signed
    path.write_text(json.dumps({"accepted": signed}))
    with pytest.raises(ValueError, match="holds a list under 'accept'"):
        main_module.acceptances(path)


def test_a_weak_seed_counts_only_beside_a_word_for_it(tmp_path: Path) -> None:
    world = World(tmp_path, seed=41)  # the first development world's kind of seed
    assert world.found("answer", "there are 41 episodes") == {}
    for text in ("generation number 41", "the seed = 41", json.dumps({"world_seed": 41})):
        assert world.found("answer", text)[("world_seed", "41")] == "leak", text


def test_scripted_outputs_are_scanned_as_returned(world: World) -> None:
    row = episode(world)
    scan = scan_texts(world.scanner, [("obtained.json", json.dumps([row]))], "sandbox output")
    assert {f.verdict for f in scan.findings.values()} == {"leak"}
    assert {k[0] for k in scan.findings} == {"episode_seed", "truth_field"}
    assert scan_tool_log(world.scanner, world.tmp / "absent.jsonl").problems


# 3. The verdict fails closed.


def scan_record(
    findings: list[dict[str, Any]] | None = None,
    problems: list[str] | None = None,
    counts: dict[str, dict[str, int]] | None = None,
    calls: int = 3,
) -> dict[str, Any]:
    """A scan record as `SessionScan.as_json` writes it; `counts` where they exceed the list."""
    listed = findings or []
    if counts is None:
        counts = {}
        for item in listed:
            entry = counts.setdefault(item["kind"], {"leak": 0, "to_judge": 0})
            entry[item["verdict"]] += 1
    leak = sum(c["leak"] for c in counts.values())
    judge = sum(c["to_judge"] for c in counts.values())
    verdict = (
        "leak" if leak else "incomplete" if problems else "to_judge" if judge else "nothing_found"
    )
    return {
        "verdict": verdict,
        "findings_total": leak + judge,
        "findings": listed,
        "findings_by_kind": counts,
        "problems": problems or [],
        "records_read": {"tool calls in the log": calls},
    }


def clean_record() -> dict[str, Any]:
    """A record with everything a clean verdict needs."""
    routes: list[dict[str, Any]] = [
        {
            "route": name,
            "verdict": "no_leak_detected",
            "compared_episodes": 8,
            "scan": scan_record(),
        }
        for name in FIXED_ROUTES
    ]
    routes += [
        {
            "route": name,
            "session": number,
            "ok": True,
            "cut_off": False,
            "tool_call_count": 3,
            "scan": scan_record(),
        }
        for name, number in REQUIRED_SESSIONS
    ]
    control = {"planted": 400, "found": 400, "convicted_as_leak": 250, "missed": []}
    return {
        "stopped": None,
        "routes": routes,
        "truth": {"facts": 100, "empty_kinds": [], "problems": []},
        "public_surface": {
            "scanned": {"texts": 40, "problems": []},
            "overlaps": {"by_status": {}, "unresolved": [], "acceptance_problems": []},
        },
        "selftest": {
            "passed": True,
            "public_surface": {"count": 0},
            "positive_control": control,
            "kinds_not_recognisable": [],
        },
        "controls": {"live": {"passed": True}},
        "wall": {
            "passed": True,
            "checks": [],
            "export_contract": {"passed": True},
            "seed_checks": [{"name": "the world seed stands in no mounted file", "passed": True}],
            "generator_import": {"passed": True},
        },
        "isolate": {
            "passed": True,
            "exec_calls_made": 5,
            "exec_calls_in_the_cli_event_stream": 5,
        },
        "experiment": {
            "status": "completed",
            "replied": True,
            "completed": True,
            "reply_read_by_second_session": True,
            "new_export_contract": {"passed": True},
        },
    }


def test_a_record_with_everything_is_clean() -> None:
    record = clean_record()
    finish(record)
    assert record["verdict"] == "no_leak_detected" and record["complete"]
    assert record["blockers"] == [] and record["leaks"] == [] and record["to_judge"] == []
    assert record["found"] == {"leak": 0, "to_judge": 0}
    # the verdict says in its own record what it does not cover
    assert record["scope"]["the scan does not recognise"] == list(NOT_RECOGNISED)
    assert "without any of its hidden names" in " ".join(NOT_RECOGNISED)


def _set(path: str, value: Any) -> Any:
    def change(record: dict[str, Any]) -> None:
        target = record
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        if value is _DROP:
            del target[keys[-1]]
        else:
            target[keys[-1]] = value

    return change


def _route(name: str, session: int | None, **changes: Any) -> Any:
    def change(record: dict[str, Any]) -> None:
        for route in record["routes"]:
            if route["route"] == name and route.get("session") == session:
                route.update(changes)

    return change


def _drop_route(name: str, session: int | None) -> Any:
    def change(record: dict[str, Any]) -> None:
        record["routes"] = [
            r for r in record["routes"] if (r["route"], r.get("session")) != (name, session)
        ]

    return change


_DROP = object()
BLOCKED = {
    "stopped": (_set("stopped", "KeyError: 'world_id'"), "the audit stopped"),
    "no inventory": (_set("truth", _DROP), "no truth inventory"),
    "an empty kind": (_set("truth.empty_kinds", ["oracle_value"]), "kind oracle_value"),
    "a truth file unread": (_set("truth.problems", ["missing truth file: x"]), "could not be read"),
    "no self-test": (_set("selftest", _DROP), "self-test did not run"),
    "self-test failed": (_set("selftest.passed", False), "self-test did not pass"),
    "surface gives findings": (_set("selftest.public_surface.count", 3), "public surface"),
    "a plant missed": (_set("selftest.positive_control.missed", [{"kind": "level"}]), "planted"),
    "nothing planted": (_set("selftest.positive_control.planted", 0), "planted"),
    "nothing convicted": (_set("selftest.positive_control.convicted_as_leak", 0), "convicted"),
    "a kind unrecognisable": (
        _set("selftest.kinds_not_recognisable", ["world_id"]),
        "cannot be recognised",
    ),
    "no live control": (_set("controls", {}), "live control"),
    "live control failed": (_set("controls.live.passed", False), "live control"),
    "no wall": (_set("wall", _DROP), "wall was not checked"),
    "wall failed": (_set("wall.passed", False), "wall checks failed"),
    "generator importable": (_set("wall.generator_import.passed", False), "world generator"),
    "generator unchecked": (_set("wall.generator_import", _DROP), "world generator"),
    "no seed checks": (_set("wall.seed_checks", []), "seed checks did not run"),
    "no isolate probe": (_set("isolate", _DROP), "exec isolate"),
    "isolate reached out": (
        _set("isolate", {"passed": False, "reached_beyond_computation": {"the marker file": 1}}),
        "exec isolate, reached: the marker file",
    ),
    "isolate report incomplete": (
        _set("isolate", {"passed": False, "problems": ["93 attempts without a readable outcome"]}),
        "exec isolate, 93 attempts",
    ),
    "exec calls unrecorded": (
        _set("isolate.exec_calls_in_the_cli_event_stream", 0),
        "event stream",
    ),
    "fewer exec calls recorded than made": (
        _set("isolate.exec_calls_in_the_cli_event_stream", 4),
        "event stream",
    ),
    "exec calls not counted": (_set("isolate.exec_calls_made", _DROP), "event stream"),
    "the export beyond the contract": (
        _set("wall.export_contract", {"passed": False, "problems": ["log/x.parquet: not a file"]}),
        "log contract does not declare",
    ),
    "the export unchecked": (_set("wall.export_contract", _DROP), "log contract"),
    "the surface unread in part": (
        _set("public_surface.scanned.problems", ["missing in the export: data_dictionary.md"]),
        "could not be read in full",
    ),
    "the overlap unrecorded": (_set("public_surface", _DROP), "was not recorded"),
    "the overlap without its list": (
        _set("public_surface.overlaps", {"by_status": {}}),
        "was not recorded",
    ),
    "an acceptance that does not hold": (
        _set("public_surface.overlaps.acceptance_problems", ["acceptance 1: needs a reason"]),
        "acceptance of a public value",
    ),
    "a fixed route missing": (
        _drop_route("sandbox-interior", None),
        "sandbox-interior did not run",
    ),
    "a fixed route unfinished": (
        _route("identifier-seeds-through-query", None, verdict="incomplete", error="timeout"),
        "did not finish",
    ),
    "the memory route compared nothing": (
        _route("memory-across-sessions", None, verdict="nothing_compared", reason="no note"),
        "compared nothing",
    ),
    "a session missing": (_drop_route("bulk-data", 1), "bulk-data/1 did not run"),
    "the second session missing": (_drop_route(TWO_SESSIONS, 2), f"{TWO_SESSIONS}/2 did not run"),
    "the checklist missing": (_drop_route("fixed-checklist", 1), "fixed-checklist/1 did not run"),
    "a session failed": (_route("skills-tools", 1, ok=False, error="refusal"), "failed: refusal"),
    "a session cut off": (_route("files-sandbox", 1, cut_off=True), "cut off"),
    "a session without a tool call": (
        _route("files-sandbox", 1, tool_call_count=0),
        "files-sandbox/1 made no tool call",
    ),
    "a session whose tool calls the scan did not read": (
        _route("files-sandbox", 1, scan=scan_record(calls=0)),
        "made 3 tool calls and the scan read 0",
    ),
    "a scan with problems": (
        _route("files-sandbox", 1, scan=scan_record(problems=["line 3"])),
        "scan of files-sandbox/1 is not evidence: line 3",
    ),
    "a session unscanned": (_route("bulk-data", 1, scan=None), "scan of bulk-data/1"),
    "a scan record without counts": (
        _route("bulk-data", 1, scan={"verdict": "nothing_found", "findings": [], "problems": []}),
        "scan of bulk-data/1 is not evidence: the scan record has no counts",
    ),
    "a scan record without a verdict": (
        _route("bulk-data", 1, scan={**scan_record(), "verdict": None}),
        "the scan record's verdict is None",
    ),
    "a scan record that does not say whether it had problems": (
        _route("bulk-data", 1, scan={**scan_record(), "problems": None}),
        "does not say whether it had problems",
    ),
    "a scan that says incomplete and names nothing": (
        _route("bulk-data", 1, scan={**scan_record(), "verdict": "incomplete"}),
        "says incomplete and names no problem",
    ),
    "a fixed route's scan with problems": (
        _route("sandbox-interior", None, scan=scan_record(problems=["seen.json: cut off"])),
        "scan of sandbox-interior is not evidence: seen.json: cut off",
    ),
    "a fixed route that compared no seed": (
        _route("identifier-derived-seeds", None, compared_episodes=0),
        "does not say that it compared",
    ),
    "no experiment step": (_set("experiment", _DROP), "no experiment ran as it was requested"),
    "no runner": (
        _set("experiment", {"status": "no experiment runner was handed to the audit"}),
        "no experiment runner",
    ),
    "an experiment that was refused": (
        _set("experiment.completed", False),
        "no experiment ran as it was requested",
    ),
    "the reply not read": (
        _set("experiment.reply_read_by_second_session", False),
        "did not read the reply",
    ),
    "the export after the experiment beyond the contract": (
        _set("experiment.new_export_contract", {"passed": False, "problems": ["x"]}),
        "export after the experiment",
    ),
    "the export after the experiment unchecked": (
        _set("experiment.new_export_contract", _DROP),
        "export after the experiment",
    ),
}


@pytest.mark.parametrize("case", sorted(BLOCKED))
def test_each_missing_piece_blocks_a_clean_verdict(case: str) -> None:
    change, words = BLOCKED[case]
    record = clean_record()
    change(record)
    finish(record)
    assert record["verdict"] == "incomplete" and not record["complete"]
    assert any(words in blocker for blocker in record["blockers"]), record["blockers"]


def test_a_route_that_cannot_apply_is_listed_and_neither_passed_nor_failed() -> None:
    record = clean_record()
    _route(
        "identifier-derived-seeds",
        None,
        verdict="nothing_compared",
        not_applicable=True,
        reason="x",
    )(record)
    finish(record)
    assert record["routes_that_compared_nothing"] == ["identifier-derived-seeds"]
    assert record["verdict"] == "no_leak_detected"


def test_findings_and_leaks_decide_before_and_after_the_blockers() -> None:
    judge_item = {"kind": "level", "value": "7.25", "verdict": "to_judge"}
    leak_item = {"kind": "world_seed", "value": "1", "verdict": "leak"}
    record = clean_record()
    kinds = {"level": {"leak": 0, "to_judge": 1}, "episode_seed": {"leak": 2016, "to_judge": 0}}
    _route("bulk-data", 1, scan=scan_record([judge_item], counts=kinds))(record)
    finish(record)
    # everything found is counted, also what a record does not list
    assert record["found"] == {"leak": 2016, "to_judge": 1} and record["blockers"] == []
    record = clean_record()
    _route("bulk-data", 1, scan=scan_record([judge_item]))(record)
    finish(record)
    assert record["verdict"] == "to_judge" and record["complete"]
    assert record["to_judge"] == [{"route": "bulk-data", "session": 1, **judge_item}]
    _set("experiment.reply_read_by_second_session", False)(record)
    finish(record)
    assert record["verdict"] == "incomplete"  # a blocker outranks an item to judge
    _route("sandbox-interior", None, scan=scan_record([leak_item]))(record)
    finish(record)
    assert record["verdict"] == "leak"  # and a leak outranks everything
    for change in (
        _route("identifier-derived-seeds", None, verdict="leak"),
        _route(TWO_SESSIONS, 2, scan=scan_record([leak_item])),
        _set("wall.seed_checks", [{"name": "no seed in the public surface", "passed": False}]),
        lambda r: r["routes"].append(
            {"route": "experiment-export", "verdict": "leak", "what": "x"}
        ),
    ):
        record = clean_record()
        change(record)
        finish(record)
        assert record["verdict"] == "leak" and record["leaks"], record["blockers"]
        assert record["found"]["leak"] == 1


# Review 2, point 1: what a record lists never decides. With `--findings-listed 0` a
# scan lists nothing and still counts everything.


@pytest.mark.parametrize("listed", [0, 1, 200])
def test_the_verdict_does_not_depend_on_how_many_findings_are_listed(
    world: World, listed: int
) -> None:
    run, state = world.session("answer", f"the seed is {world.seed}; level 7.25 perhaps")
    scan = scan_session(world.scanner, run, state).as_json(listed=listed)
    assert len(scan["findings"]) == min(listed, 2) and scan["findings_total"] == 2
    record = clean_record()
    _route("bulk-data", 1, scan={**scan, "records_read": {"tool calls in the log": 3}})(record)
    finish(record)
    assert record["verdict"] == "leak" and record["found"] == {"leak": 1, "to_judge": 1}
    assert len(record["leaks"]) == min(listed, 1)  # the listing follows the limit, only it
    # the same for an item to judge alone
    run, state = world.session("answer", "level 7.25 perhaps")
    scan = scan_session(world.scanner, run, state).as_json(listed=listed)
    record = clean_record()
    _route("bulk-data", 1, scan={**scan, "records_read": {"tool calls in the log": 3}})(record)
    finish(record)
    assert record["verdict"] == "to_judge" and record["found"] == {"leak": 0, "to_judge": 1}


def test_a_scan_record_counts_as_the_worse_of_what_it_says() -> None:
    leak_item = {"kind": "world_seed", "value": "1", "verdict": "leak"}
    # the scan's own verdict, with nothing counted and nothing listed
    found, faults = scan_counts({**scan_record(), "verdict": "leak"})
    assert found == {"leak": 1, "to_judge": 0} and "says leak and counts none" in faults[0]
    found, faults = scan_counts({**scan_record(), "verdict": "to_judge"})
    assert found == {"leak": 0, "to_judge": 1} and faults
    # more listed than counted, counts that do not add up, a verdict below the counts
    found, faults = scan_counts({**scan_record(), "findings": [leak_item], "verdict": "leak"})
    assert found["leak"] == 1 and "lists more findings" in faults[0]
    _, faults = scan_counts({**scan_record([leak_item]), "findings_total": 5})
    assert faults == ["the scan's counts do not add up to its total"]
    _, faults = scan_counts({**scan_record([leak_item]), "findings_total": None})
    assert faults == ["the scan record has no total"]
    found, faults = scan_counts({**scan_record([leak_item]), "verdict": "nothing_found"})
    assert found["leak"] == 1 and faults == ["the scan says nothing found and counts findings"]
    assert scan_counts(None) == ({"leak": 0, "to_judge": 0}, ["there is no scan"])
    assert scan_counts(scan_record([leak_item])) == ({"leak": 1, "to_judge": 0}, [])
    for broken in ("leak", "to_judge", "nothing_found"):
        record = clean_record()
        _route("skills-tools", 1, scan={**scan_record(), "verdict": broken, "findings_total": 9})(
            record
        )
        finish(record)
        assert record["verdict"] != "no_leak_detected", broken


# Review 2, point 4: a route's own status counts, with or without a finding to show for it.


@pytest.mark.parametrize("name", FIXED_ROUTES)
@pytest.mark.parametrize("status", ["leak", "to_judge"])
def test_a_fixed_route_s_own_status_is_never_dropped(name: str, status: str) -> None:
    record = clean_record()
    _route(name, None, verdict=status, what="read in the sandbox")(record)
    finish(record)
    assert record["verdict"] == status and record["found"][status] == 1
    items = record["leaks"] if status == "leak" else record["to_judge"]
    assert items == [
        {
            "route": name,
            "session": None,
            "what": f"read in the sandbox: its own result is {status.replace('_', ' ')}",
        }
    ]
    # with findings of its own scan, those are what is counted and listed
    item = {"kind": "truth_field", "value": "world_truth", "verdict": status}
    _route(name, None, scan=scan_record([item]))(record)
    finish(record)
    listed = record["leaks" if status == "leak" else "to_judge"]
    assert record["found"][status] == 1 and [i.get("kind") for i in listed] == ["truth_field"]


def test_an_unresolved_public_value_blocks_and_counts() -> None:
    open_judge = {"kind": "level", "value": "3.05", "beside": None, "verdict": "to_judge"}
    open_leak = {"kind": "expected_value", "value": "5.539", "beside": "expectedlosspoints"}
    open_leak["verdict"] = "leak"
    record = clean_record()
    record["public_surface"]["overlaps"]["unresolved"] = [open_judge]
    finish(record)
    assert record["verdict"] == "incomplete" and record["found"] == {"leak": 0, "to_judge": 1}
    assert "1 values of the truth that the public surface shows" in record["blockers"][0]
    assert record["to_judge"] == [{"route": "public-surface", **open_judge}]
    record["public_surface"]["overlaps"]["unresolved"].append(open_leak)
    finish(record)
    assert record["verdict"] == "leak"
    assert record["leaks"] == [{"route": "public-surface", **open_leak}]
    # what the export after an experiment shows in addition counts once
    record = clean_record()
    record["public_surface"]["overlaps"]["unresolved"] = [open_judge]
    record["public_surface"]["overlaps_after_experiment"] = {"unresolved": [open_judge, open_leak]}
    finish(record)
    assert record["found"] == {"leak": 1, "to_judge": 1} and record["verdict"] == "leak"
    later = {"unresolved": [], "acceptance_problems": ["acceptance 2: needs a reason"]}
    record = clean_record()
    record["public_surface"]["overlaps_after_experiment"] = later
    finish(record)
    assert record["verdict"] == "incomplete" and "acceptance 2" in record["blockers"][0]


def audit_args(world: World, out: Path, **changes: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "world": world.root,
        "export": world.export,
        "output": out,
        "static_only": False,
        "engine": None,
        "only": None,
        "budget_seconds": 5400,
        "session_seconds": 900,
        "tool_call_limit": 40,
        "findings_listed": 200,
        "codex_model": "gpt-6-astra",
        # what a person accepted of the toy export's own numbers (the `fake` fixture)
        "accept": ACCEPTED.get("file"),
    }
    values.update(changes)
    return argparse.Namespace(**values)


ACCEPTED: dict[str, Path] = {}


def accept_file(path: Path, overlap: dict[str, Any], **changes: Any) -> Path:
    """An acceptance file as a person fills in `to_accept` of `public-overlap.json`."""
    entries = [
        {**entry, "accepted_by": "a test", "reason": "the toy export's own numbers", **changes}
        for entry in overlap["to_accept"]
    ]
    path.write_text(json.dumps({"accept": entries}))
    return path


def stored(out: Path) -> dict[str, Any]:
    (path,) = out.glob("*/record.json")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def test_the_record_carries_a_verdict_before_anything_runs(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def look(args: Any, record: dict[str, Any], out: Path) -> None:
        seen.update(json.loads((out / "record.json").read_text()))
        raise SystemExit(3)  # the worst case: the program is leaving

    monkeypatch.setattr(main_module, "audit", look)
    with pytest.raises(SystemExit):
        main_module.run(audit_args(world, tmp_path / "records"))
    assert seen["verdict"] == "incomplete" and "has not finished" in seen["stopped"]
    after = stored(tmp_path / "records")
    assert after["verdict"] == "incomplete" and after["stopped"].startswith("SystemExit")
    assert after["phase"] == "finished" and "the audit stopped" in after["blockers"][0]


def test_a_kill_that_can_be_caught_ends_in_a_record_that_says_so(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(main_module, "audit", lambda *a: os.kill(os.getpid(), signal.SIGTERM))
    with pytest.raises(SystemExit, match="terminated by signal 15"):
        main_module.run(audit_args(world, tmp_path / "records"))
    record = stored(tmp_path / "records")
    assert record["verdict"] == "incomplete" and "terminated by signal 15" in record["stopped"]
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.parametrize("error", [KeyError("world_id"), RuntimeError("boom"), OSError("gone")])
def test_any_error_ends_in_a_record_with_a_verdict(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: Any) -> None:
        raise error

    monkeypatch.setattr(main_module, "audit", fail)
    _, record = main_module.run(audit_args(world, tmp_path / "records"))
    assert record["verdict"] == "incomplete" and type(error).__name__ in record["stopped"]
    assert "traceback" in record and stored(tmp_path / "records")["verdict"] == "incomplete"


@pytest.mark.parametrize(
    ("verdict", "code"), [("no_leak_detected", 0), ("to_judge", 1), ("incomplete", 1), ("leak", 1)]
)
def test_only_a_clean_verdict_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str, code: int
) -> None:
    record: dict[str, Any] = {"verdict": verdict, "leaks": [], "to_judge": [], "blockers": ["x"]}
    record.update(stopped=None, found={"leak": 0, "to_judge": 0})
    record["scope"] = {"the verdict covers": "x", "the scan does not recognise": ["y"]}
    monkeypatch.setattr(main_module, "run", lambda args: (tmp_path, record))
    judged: dict[str, Any] = {**record, "scan_verdict": "nothing_found", "findings": {}}
    judged["selftest"] = {"passed": True}
    judged["public_surface"] = {"overlaps": {"by_status": {}}}
    monkeypatch.setattr(main_module, "rejudge", lambda *a: (tmp_path, judged))
    mask = os.umask(0o022)
    try:
        for arguments in (
            ["--world", str(tmp_path), "--export", str(tmp_path)],
            ["--rejudge", "x"],
        ):
            monkeypatch.setattr("sys.argv", ["audit", *arguments])
            assert main_module.main() == code, arguments
        # a record of the first format has the scan's verdict only, and that is never
        # the audit's: nothing found by the scan alone does not exit zero
        del judged["verdict"]
        assert main_module.main() == 1
        judged["scan_verdict"] = "to_judge"
        assert main_module.main() == 1
    finally:
        os.umask(mask)


# 4. The attacks.


def test_the_briefs_name_every_surface_and_bound_the_calls(tmp_path: Path) -> None:
    outside, repo = tmp_path / "outside/notes.txt", tmp_path / "repo"
    prompts = dict(briefs(outside, repo, 37, "AUDIT-NOTE-1"))
    assert tuple(prompts) == BRIEFS and TWO_SESSIONS in prompts
    text = " ".join(prompts.values())
    for surface in (
        "steps table",
        "events table",
        "get_frames from at least twelve different episodes",
        "video file names",
        "read_skill",
        "helper queries",
        "/opt/tools/src",
        "/opt/tools/site",
        "/proc/self/environ",
        "/proc/self/mountinfo",
        "/proc/1/environ",
        "getenv",
        "/memory, /outbox, /results and /log",
        "error messages from every tool",
        "clocks and timing",
        "memory_write",
        "submit_diagnosis",
        "one valid experiment with two arms of ten attempts",
        "AUDIT-NOTE-1",
        str(outside),
    ):
        assert surface in text, surface
    assert all("at most 37 tool calls" in prompt for prompt in prompts.values())
    assert "memory" in SECOND_SESSION and "experiment_results" in SECOND_SESSION
    steps = checklist(outside, repo)
    for step in ("read_text", "/proc/self/mountinfo", "FROM steps", "get_frames", "memory_read"):
        assert step in steps
    assert "getOwnPropertyNames(globalThis)" in steps and "ALL_TOOLS" in steps
    assert steps.count("'../../outside'") == 2  # get_frames and read_skill, out of their trees
    assert len([line for line in steps.splitlines() if line[:1].isdigit()]) == 25
    # A brief hands out no world: its path would carry the truth-side name.
    public = public_texts(outside, repo, 37, "AUDIT-NOTE-1")
    assert set(public) >= {f"brief {name}" for name in BRIEFS} | {"brief fixed-checklist"}
    assert "worlds/" not in " ".join(public.values())


class FakeRuntime:
    """Stands in for everything of an audit that needs a sandbox, a CLI or a model."""

    def __init__(self, world: World, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.world, self.tmp, self.calls = world, tmp, []  # type: ignore[var-annotated]
        self.answers: dict[tuple[str, int], str] = {}
        self.errors: dict[tuple[str, int], str] = {}
        self.request = True
        self.write_marker = True
        self.read_results = True
        self.results_ok = True  # whether the second session's experiment_results call succeeds
        self.tool_calls = True  # whether a session calls a tool at all
        monkeypatch.setenv("ROBOTS_AGENT_ROOT", "set again by every audit")  # restored after
        monkeypatch.setenv("ROBOTS_LEAK_AUDIT_ROOTS", str(tmp / "roots"))
        (tmp / "roots").mkdir(mode=0o700)
        python = tmp / "python"
        for part in ("bin", "site", "src", "skills"):
            (python / part).mkdir(parents=True, exist_ok=True)
        (python / "bin/python3.12").write_bytes(b"not an interpreter")
        self.env = ToolEnv(python, python / "site", python / "src", python / "skills")
        m = main_module
        monkeypatch.setattr(m, "build_tool_env", lambda **kw: self.env)
        monkeypatch.setattr(m, "check_wall", lambda *a: {"passed": True, "checks": []})
        # the toy export is not laid out as the log contract wants; `export_check` has
        # its own test, on an export the engine's exporter wrote
        monkeypatch.setattr(m, "export_check", lambda *a: {"passed": True})
        monkeypatch.setattr(m, "generator_import", lambda *a: {"passed": True})
        fixed = {"verdict": "no_leak_detected", "compared_episodes": 8}
        monkeypatch.setattr(
            m, "identifier_probe", lambda *a: {"route": "identifier-derived-seeds", **fixed}
        )
        monkeypatch.setattr(
            m, "query_probe", lambda *a: {"route": "identifier-seeds-through-query", **fixed}
        )
        monkeypatch.setattr(m, "interior_probe", lambda *a: {"route": "sandbox-interior", **fixed})
        monkeypatch.setattr(
            m, "memory_probe", lambda *a: {"route": "memory-across-sessions", **fixed}
        )
        monkeypatch.setattr(m, "live_control", lambda *a: {"passed": True})
        probed = {"passed": True, "exec_calls_made": 5, "exec_calls_in_the_cli_event_stream": 5}
        probed["login_path"] = {"in_place": True}  # type: ignore[assignment]
        monkeypatch.setattr(m, "isolate_probe", lambda *a: dict(probed))
        gate = {"claude": {"passed": True}, "codex": {"passed": True}}
        monkeypatch.setattr(m, "subscription_gate", lambda *a: (gate, False))
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess([], 0, "fixture-revision", ""),
        )
        monkeypatch.setattr(m, "run_agent", self.agent)

    def agent(self, backend: str, model: str, system: str, prompt: str, **kw: Any) -> AgentResult:
        label = kw["label"].split("/")
        name, session = label[1], int(label[2])
        self.calls.append((backend, name, session, kw["timeout"], prompt))
        run = kw["run_root"] / f"{name}-{session}"
        access = kw["tools"]
        memory = access.memory
        answer = self.answers.get((name, session), "nothing obtained")
        (run / "box/toolbox").mkdir(parents=True)
        (run / "argv.json").write_text(
            json.dumps(Layout(mounts=[Mount("bind", str(memory), "/memory")]).argv())
        )
        (run / "prompt.json").write_text(json.dumps({"system": system, "prompt": prompt}))
        log: list[dict[str, Any]] = []
        if self.tool_calls:
            look = {"seq": 1, "tool": "describe", "arguments": {"part": "costs"}, "ok": True}
            log.append({**look, "error": None, "result": ["[]"]})
        if name == TWO_SESSIONS and session == 1:
            marker = prompt.split("the line ")[1].split(".")[0]
            (memory / "notes.md").write_text(f"tried things; {marker if self.write_marker else ''}")
            if self.request:
                request = experiment_request(10)
                append_line(
                    access.outbox / EXPERIMENTS_FILE,
                    {"experiment_id": "x0001", "session": 1, "request": request},
                )
        if name == TWO_SESSIONS and session == 2 and self.read_results:
            results = access.results / "experiments.jsonl"
            shown = (
                [json.loads(line) for line in results.read_text().splitlines()]
                if results.exists()
                else []
            )
            queued = [{"experiment_id": "x0001", "status": "queued", "arms": [], "message": ""}]
            log.append(
                {
                    "seq": 2,
                    "tool": "experiment_results",
                    "arguments": {},
                    "ok": self.results_ok,
                    "result": [json.dumps(shown or queued)],
                }
            )
        (run / "box/toolbox/tool_calls.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in log)
        )
        result = {"attempts": [], "summary": answer}
        error = self.errors.get((name, session))
        res = AgentResult(
            ok=error is None,
            result=result,
            usage={},
            seconds=10.0,
            error=error,
            tool_log=log,
            run_dir=str(run),
            meta={"timed_out": error == "timeout"},
        )
        stored_result = {"ok": res.ok, "result": result, "text": "", "tool_log": log}
        (run / "result.json").write_text(json.dumps(stored_result))
        (run / "stdout.jsonl").write_text(json.dumps({"type": "result", "result": ""}) + "\n")
        return res


def experiment_request(episodes: int) -> dict[str, Any]:
    target = {"scope": "r02", "component": "scene_camera", "config_key": None}
    action = {"verb": "recalibrate", "target": {**target, "condition_key": None}, "value": None}
    return {
        "arms": [{"actions": []}, {"actions": [action]}],
        "robots": ["r02"],
        "episodes_per_arm": episodes,
        "paired_starts": True,
        "power_note": "audit",
    }


class StandInEngine:
    """A runner handed to the audit: runs the queue's experiment as it was asked for, in
    a new export that holds the arm episodes under the experiment's id.

    `ENGINES["how"]` makes it answer otherwise: "refused", "one arm short", "one episode
    short", "an episode twice", "not in the export", "another experiment's episodes".
    """

    def __init__(self, context: Any, message: str | None = None) -> None:
        # Without a message of its own it answers as the tool server's public contract
        # wants: "done" for an experiment that ran, a fixed code for a refusal.
        self.context, self.message, self.queues = context, message, []  # type: ignore[var-annotated]
        self.how = ENGINES.get("how", "as requested")

    def run_queue(self, queue_file: Path, queue: Any) -> EngineReply:
        self.queues.append(queue)
        export = self.context.out / "export"
        shutil.copytree(self.context.export, export)
        path = export / "log/episodes.parquet"
        rows = [{**row, "experiment_id": ""} for row in pq.read_table(path).to_pylist()]
        results = []
        number = 900000
        for q in queue.experiments:
            arms = []
            for index in range(len(q.request.arms)):
                episodes = []
                for _ in range(q.request.episodes_per_arm):
                    number += 1
                    episodes.append(f"e{number}")
                    tagged = "x0999" if self.how.startswith("another") else q.experiment_id
                    rows.append({**rows[0], "episode_id": episodes[-1], "experiment_id": tagged})
                arms.append(ArmResult(arm_index=index, episode_ids=episodes))
            if self.how == "one arm short":
                arms.pop()
            elif self.how == "one episode short":
                arms[0].episode_ids.pop()
            elif self.how == "an episode twice":
                arms[0].episode_ids[0] = arms[0].episode_ids[1]
            elif self.how == "not in the export":
                rows = [row for row in rows if row["episode_id"] != arms[1].episode_ids[0]]
            refused = self.how == "refused"
            results.append(
                ExperimentResult(
                    experiment_id=q.experiment_id,
                    status="refused" if refused else "done",
                    arms=[] if refused else arms,
                    message=self.message or (REFUSED if refused else COMPLETED),
                )
            )
        pq.write_table(pa.Table.from_pylist(rows), path)
        return EngineReply(results=results, export=export)


ENGINES: dict[str, Any] = {}


def stand_in(context: Any) -> StandInEngine:
    ENGINES["last"] = StandInEngine(context, ENGINES.get("message"))
    return ENGINES["last"]  # type: ignore[no-any-return]


def stand_in_with_news(context: Any) -> Any:
    """A runner whose new export shows the number 7.25, which the first export does not."""

    class WithNews(StandInEngine):
        def run_queue(self, queue_file: Path, queue: Any) -> EngineReply:
            reply = super().run_queue(queue_file, queue)
            description = reply.export / "log/description.json"
            data = json.loads(description.read_text())
            data["note"] = "arm episodes ran at 7.25 Hz"
            description.write_text(json.dumps(data))
            return reply

    return WithNews(context)


def failing(context: Any) -> Any:
    class Broken:
        def run_queue(self, *args: Any) -> Any:
            raise RuntimeError("no GPU today")

    return Broken()


@pytest.fixture
def fake(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRuntime]:
    ENGINES.clear()
    ACCEPTED["file"] = accept_file(tmp_path / "accepted.json", world.overlap)
    yield FakeRuntime(world, tmp_path, monkeypatch)
    ACCEPTED.clear()


def test_a_full_audit_with_a_runner_handed_in_is_clean_and_ordered(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    out, record = main_module.run(args)
    assert record["verdict"] == "no_leak_detected", record["blockers"]
    assert record["stopped"] is None and record["complete"]
    assert [(c[0], c[1], c[2]) for c in fake.calls] == [
        ("claude", "files-sandbox", 1),
        ("claude", "metadata-errors-timing", 1),
        ("claude", "bulk-data", 1),
        ("claude", "skills-tools", 1),
        ("claude", TWO_SESSIONS, 1),
        ("claude", TWO_SESSIONS, 2),
        ("codex", "fixed-checklist", 1),
    ]
    assert all(call[3] == 900 for call in fake.calls)  # the limit of one session
    assert fake.calls[5][4] == SECOND_SESSION.format(limit=40)
    assert record["model_wall_seconds"] == 70.0
    experiment = record["experiment"]
    assert experiment["status"] == "completed" and experiment["completed"]
    assert experiment["reply_read_by_second_session"]
    assert not experiment["refusal_read_by_second_session"]
    assert experiment["requested"] == experiment["run"] == 1
    assert experiment["request"] == {"experiment_id": "x0001", "arms": 2, "episodes_per_arm": 10}
    assert [len(arm["episode_ids"]) for arm in experiment["result"]["arms"]] == [10, 10]
    assert len(ENGINES["last"].queues[0].experiments) == 1
    assert ENGINES["last"].context.world == world.root.resolve()
    assert record["limits"]["tool_calls_instructed_per_brief"] == 40
    assert record["limits"]["claude_opus_sessions"] == 6
    sessions = {(r["route"], r.get("session")): r for r in record["routes"]}
    assert sessions[(TWO_SESSIONS, 1)]["note_marker_in_memory"]
    assert sessions[(TWO_SESSIONS, 2)]["tool_calls_by_tool"] == {
        "describe": 1,
        "experiment_results": 1,
    }
    for name in ("truth-inventory.json", "public-overlap.json", "selftest.json", "record.json"):
        assert (out / name).is_file()
    # what the toy export shows of the toy truth was accepted by a person, on record
    overlaps = record["public_surface"]["overlaps"]
    assert overlaps["unresolved"] == [] and overlaps["by_status"]["accepted"] >= 1
    assert {e["accepted_by"] for e in overlaps["accepted"]} == {"a test"}
    assert record["acceptance_file"] == str(ACCEPTED["file"])
    assert (out / "public-overlap-after-experiment.json").is_file()
    assert record["public_surface"]["overlaps_after_experiment"]["unresolved"] == []
    assert record["selftest"]["passed"] and "plants" not in record["selftest"]
    assert record["controls"]["planted_sessions"]["convicted_as_leak"] > 0
    assert record["truth"]["by_kind"]["world_seed"] >= 1
    # the reply and the new export were read with a scanner of their own
    assert (out / "truth-inventory-after-experiment.json").is_file()
    assert all(r["passed"] for r in experiment["new_export_seed_checks"])


def test_seeds_a_fixed_route_derived_are_scanned_once_its_comparison_matched(
    world: World, tmp_path: Path, fake: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = episode(world)

    def probe(verdict: str) -> Any:
        def run(mounts: Any, out: Path, runs: Any) -> dict[str, Any]:
            out.write_text(json.dumps({"derived": [row]}))
            return {"route": "identifier-derived-seeds", "verdict": verdict, "evidence": str(out)}

        return run

    monkeypatch.setattr(main_module, "identifier_probe", probe("leak"))
    _, record = main_module.run(audit_args(world, tmp_path / "a", static_only=True))
    found = [i.get("kind") for i in record["leaks"] if i["route"] == "identifier-derived-seeds"]
    assert found == ["episode_seed", "episode_seed"]  # both seeds, from the scan
    assert record["verdict"] == "leak" and record["found"]["leak"] == 2
    # Without a match the derived numbers are guesses; the probe's own keys are no finding.
    monkeypatch.setattr(main_module, "identifier_probe", probe("no_leak_detected"))
    out, record = main_module.run(audit_args(world, tmp_path / "b", static_only=True))
    assert record["leaks"] == [] and "scan" not in record["routes"][0]
    # A re-judge makes the comparison again from the seeds the route left. Here they are
    # the true ones, whatever the stored verdict says.
    _, judged = main_module.rejudge(out, tmp_path / "rejudged", 200)
    route = judged["routes"][0]
    assert (route["stored_verdict"], route["verdict"]) == ("no_leak_detected", "leak")
    assert route["compared_episodes"] == 1 and route["matched"]["noise_seed"] == 1
    assert judged["verdict"] == "leak" and judged["found"]["leak"] == 2
    # A stored record that does not say how many seeds were compared blocks; the
    # re-judge counts them from what the route left, and that it compared is on record.
    row = {**row, "scene_seed": 1, "noise_seed": 2}
    out, record = main_module.run(audit_args(world, tmp_path / "c", static_only=True))
    said = "identifier-derived-seeds says nothing matched and does not say that it compared"
    assert [b for b in record["blockers"] if said in b]
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-c", 200)
    assert judged["routes"][0]["verdict"] == "no_leak_detected"
    assert judged["routes"][0]["compared_episodes"] == 1 and judged["found"]["leak"] == 0
    assert not [b for b in judged["blockers"] if said in b]
    assert len(judged["blockers"]) == len(record["blockers"]) - 1


def test_a_hidden_value_in_the_experiment_reply_is_convicted(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    ENGINES["message"] = f"refused: fault f07 ({KIND}) is not active before day 3"
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    _, record = main_module.run(args)
    assert record["verdict"] == "leak"
    found = {(i["kind"], i["value"]): i for i in record["leaks"] if i["route"] == TWO_SESSIONS}
    assert found[("fault_kind", KIND)]["places"]["experiment reply"] == 1
    assert found[("onset_day", "3")]["verdict"] == "leak"  # "... before day 3", beside the fault
    assert found[("fault_id", "f07")]["session"] == 2
    # Such a text is no message of the public reply contract: the reply was not handed
    # to the second session, it is kept in the record, and it is no experiment that ran.
    experiment = record["experiment"]
    assert experiment["status"] == "the runner's reply breaks the public reply contract"
    assert experiment["problems"] == ["result has no allow-listed public status/message"]
    assert not experiment.get("replied") and not experiment.get("completed")
    assert experiment["reply_kept_back"][0]["message"] == ENGINES["message"]
    assert not experiment["reply_read_by_second_session"]
    assert not experiment["refusal_read_by_second_session"]
    assert [b for b in record["blockers"] if b.startswith("no experiment ran as it was requested")]


@pytest.mark.parametrize(
    ("how", "message", "status"),
    [
        ("refused", "fault f07 is not active before day 3", "breaks the public reply contract"),
        ("one arm short", "fault f07 is not active before day 3", "not the experiment requested"),
    ],
)
def test_a_reply_that_is_not_handed_on_is_scanned_as_a_reply_all_the_same(
    world: World, tmp_path: Path, fake: FakeRuntime, how: str, message: str, status: str
) -> None:
    # A refusal in the runner's own words, and a reply that is not what was asked for:
    # neither reaches the second session, and what either says is read by the scan.
    ENGINES["how"], ENGINES["message"] = how, message
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    out, record = main_module.run(args)
    experiment = record["experiment"]
    assert status in experiment["status"] and not experiment.get("replied")
    assert experiment["reply_kept_back"][0]["message"] == message
    sessions = {(r["route"], r.get("session")): r for r in record["routes"]}
    assert sessions[(TWO_SESSIONS, 1)]["scan"]["verdict"] == "nothing_found"
    assert sessions[(TWO_SESSIONS, 2)]["tool_calls_by_tool"]["experiment_results"] == 1
    found = {(i["kind"], i["value"]): i for i in record["leaks"]}
    # the fault, its onset day, and the 3 as its level of 3.05 rounded, each beside f07
    assert record["verdict"] == "leak"
    assert set(found) == {("fault_id", "f07"), ("onset_day", "3"), ("level", "3.05")}
    assert all(i["places"] == {"experiment reply": 1} and i["session"] == 2 for i in found.values())
    # and again when the stored record is judged again
    _, judged = main_module.rejudge(out, tmp_path / "rejudged", 200)
    assert judged["verdict"] == "leak" and judged["found"] == record["found"]
    # a refusal by one of the contract's codes says nothing and is handed on
    ENGINES.update(how="refused", message=None)
    _, record = main_module.run(audit_args(world, tmp_path / "b", engine="test_audit:stand_in"))
    assert record["verdict"] == "incomplete" and record["found"] == {"leak": 0, "to_judge": 0}
    assert record["experiment"]["result"]["message"] == REFUSED
    assert "reply_kept_back" not in record["experiment"]


def test_the_second_session_is_read_with_the_export_it_saw(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    # 7.25 is a hidden level and stands nowhere in the first export. A session that
    # writes it before the experiment has it from somewhere; after a reply whose export
    # shows it, the same words repeat public data.
    fake.answers[(TWO_SESSIONS, 2)] = "the new episodes ran at 7.25"
    args = audit_args(world, tmp_path / "a", engine="test_audit:stand_in_with_news")
    out, record = main_module.run(args)
    sessions = {(r["route"], r.get("session")): r for r in record["routes"]}
    assert sessions[(TWO_SESSIONS, 2)]["scan"]["verdict"] == "nothing_found"
    # ... and that the new export shows a hidden level is itself what has to be judged
    later = record["public_surface"]["overlaps_after_experiment"]["unresolved"]
    assert [(e["kind"], e["value"], e["shown_in"]) for e in later] == [
        ("level", "7.25", ["export log/description.json"])
    ]
    assert record["verdict"] == "incomplete" and record["found"] == {"leak": 0, "to_judge": 1}
    assert [i["route"] for i in record["to_judge"]] == ["public-surface"]
    _, judged = main_module.rejudge(out, tmp_path / "rejudged", 200)
    assert judged["verdict"] == "incomplete" and judged["findings"]["to_judge"] == 0
    assert judged["found"] == record["found"] and judged["blockers"] == record["blockers"]
    # the same sentence in the first session, which saw the first export
    del fake.answers[(TWO_SESSIONS, 2)]
    fake.answers[(TWO_SESSIONS, 1)] = "the new episodes ran at 7.25"
    out, record = main_module.run(audit_args(world, tmp_path / "b", engine="test_audit:stand_in"))
    assert record["verdict"] == "to_judge"
    assert [(i["route"], i["session"], i["value"]) for i in record["to_judge"]] == [
        (TWO_SESSIONS, 1, "7.25")
    ]


def test_a_reply_nobody_read_and_a_note_nobody_wrote_are_recorded_as_such(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    fake.read_results = False  # the runner replied, the second session did not look
    fake.write_marker = False
    _, record = main_module.run(audit_args(world, tmp_path / "a", engine="test_audit:stand_in"))
    assert record["experiment"]["status"] == "completed"
    assert not record["experiment"]["reply_read_by_second_session"]
    assert record["verdict"] == "incomplete"
    assert record["blockers"] == [
        "the second session did not read the reply of the experiment that ran"
    ]
    sessions = {(r["route"], r.get("session")): r for r in record["routes"]}
    assert not sessions[(TWO_SESSIONS, 1)]["note_marker_in_memory"]


# Review 2, point 3: an experiment counts when it ran as it was asked for and the second
# session read that result. A refusal is a reply, and no experiment.


@pytest.mark.parametrize(
    ("how", "words"),
    [
        ("refused", "the runner's result is 'refused', not 'done'"),
        ("one arm short", "the reply has arms [0], 2 were requested"),
        ("one episode short", "arm 0 has 9 episodes, 10 were requested"),
        ("an episode twice", "an episode stands in the reply twice"),
        ("not in the export", "e900011 is not in the reply's export"),
        ("another experiment's episodes", "e900001 carries experiment_id 'x0999' in the export"),
    ],
)
def test_an_experiment_that_did_not_run_as_requested_is_not_one_that_ran(
    world: World, tmp_path: Path, fake: FakeRuntime, how: str, words: str
) -> None:
    ENGINES["how"] = how
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    _, record = main_module.run(args)
    experiment = record["experiment"]
    assert not experiment.get("completed") and not experiment["reply_read_by_second_session"]
    assert record["verdict"] == "incomplete" and len(record["blockers"]) == 1
    assert record["blockers"][0].startswith("no experiment ran as it was requested: ")
    assert words in str(experiment.get("problems") or experiment["status"])
    assert (TWO_SESSIONS, 2) in {(c[1], c[2]) for c in fake.calls}  # the second session ran
    if how == "refused":
        # the refusal reached the second session and was read: recorded, and kept apart
        assert experiment["replied"] and experiment["refusal_read_by_second_session"]
        assert experiment["result"]["status"] == "refused"
        assert words in record["blockers"][0]
    else:
        assert not experiment.get("replied") and "result" not in experiment
        assert experiment["status"] == "the runner's reply is not the experiment requested"
        assert not experiment["refusal_read_by_second_session"]


def test_a_result_call_that_failed_did_not_read_the_reply(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    fake.results_ok = False  # the second session called experiment_results; the call failed
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    _, record = main_module.run(args)
    assert record["experiment"]["completed"]
    assert not record["experiment"]["reply_read_by_second_session"]
    assert record["verdict"] == "incomplete"
    assert record["blockers"] == [
        "the second session did not read the reply of the experiment that ran"
    ]


def test_the_reply_is_held_against_the_request_and_the_export(tmp_path: Path) -> None:
    analysis = Analysis.create("w98", base=tmp_path, analysis_id="a1")
    line = {"experiment_id": "x0001", "session": 1, "request": experiment_request(2)}
    append_line(analysis.outbox / EXPERIMENTS_FILE, line)
    _, queue = analysis.write_queue(1, tmp_path)  # type: ignore[misc]
    (wanted,) = queue.experiments
    export = tmp_path / "export"
    (export / "log").mkdir(parents=True)

    def reply(rows: dict[str, str], *arms: list[str], status: Any = "done") -> EngineReply:
        table = [{"episode_id": e, "experiment_id": x} for e, x in rows.items()]
        pq.write_table(pa.Table.from_pylist(table), export / "log/episodes.parquet")
        made = [ArmResult(arm_index=i, episode_ids=ids) for i, ids in enumerate(arms)]
        result = ExperimentResult(
            experiment_id="x0001",
            status=status,
            arms=made,
            message="m",
        )
        return EngineReply(results=[result], export=export)

    ran = {f"e90000{n}": "x0001" for n in range(1, 5)}
    arms = (["e900001", "e900002"], ["e900003", "e900004"])
    problems = main_module.completion_problems
    assert problems(reply({"e000001": "", **ran}, *arms), wanted) == []
    assert problems(reply(ran, *arms, status="refused"), wanted) == [
        "the runner's result is 'refused', not 'done': m"
    ]
    assert "'queued'" in problems(reply(ran, status="queued"), wanted)[0]
    assert problems(reply(ran, arms[0]), wanted) == ["the reply has arms [0], 2 were requested"]
    assert problems(reply(ran, arms[0], ["e900003"]), wanted) == [
        "arm 1 has 1 episodes, 2 were requested"
    ]
    assert problems(reply(ran, arms[0], arms[0]), wanted) == [
        "an episode stands in the reply twice"
    ]
    assert problems(reply({**ran, "e900004": "x0002"}, *arms), wanted) == [
        "e900004 carries experiment_id 'x0002' in the export"
    ]
    del ran["e900002"]
    assert problems(reply(ran, *arms), wanted) == ["e900002 is not in the reply's export"]
    # an export without the table, and a reply for another request
    answer = reply(ran, *arms)
    (export / "log/episodes.parquet").unlink()
    assert "no readable episodes table" in problems(answer, wanted)[0]
    other = wanted.model_copy(update={"experiment_id": "x0002"})
    assert problems(answer, other) == ["the reply holds no result for x0002"]


def test_sessions_that_try_nothing_are_not_attacks_that_found_nothing(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    fake.tool_calls = False
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    _, record = main_module.run(args)
    assert record["verdict"] == "incomplete"
    quiet = [b for b in record["blockers"] if "made no tool call: it tried nothing" in b]
    assert len(quiet) == 6  # every session but the second, which read the experiment's result


def test_without_an_acceptance_the_public_overlap_blocks_and_with_one_it_is_on_record(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    # Review 2, point 2, through the whole audit: the toy export's step times hold the
    # level 3.05. Nobody accepted that: the audit is not clean and says what is open.
    args = audit_args(world, tmp_path / "open", engine="test_audit:stand_in", accept=None)
    out, record = main_module.run(args)
    assert record["verdict"] == "incomplete" and record["acceptance_file"] is None
    assert record["found"] == {"leak": 0, "to_judge": 1} and len(record["blockers"]) == 1
    assert "1 values of the truth that the public surface shows" in record["blockers"][0]
    (item,) = record["to_judge"]
    assert (item["route"], item["kind"], item["value"]) == ("public-surface", "level", "3.05")
    listed = json.loads((out / "public-overlap.json").read_text())
    assert listed["to_accept"] == [
        {
            "kind": "level",
            "values": ["3.05"],
            "beside": None,
            "form": "exact",
            "shown_in": ["table steps"],
            "accepted_by": "",
            "reason": "",
        }
    ]
    # an acceptance that names nobody is a problem of its own, and resolves nothing
    unsigned = accept_file(tmp_path / "unsigned.json", world.overlap, accepted_by="")
    args = audit_args(world, tmp_path / "unsigned", engine="test_audit:stand_in", accept=unsigned)
    _, record = main_module.run(args)
    assert record["verdict"] == "incomplete" and len(record["blockers"]) == 2
    assert "acceptance 1: needs kind, values" in record["blockers"][1]
    # the stored record is judged again with the acceptance it was given, or with another
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-open", 200)
    assert judged["verdict"] == "incomplete" and judged["blockers"] == [
        b for b in record["blockers"] if "acceptance" not in b
    ]
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-accepted", 200, ACCEPTED["file"])
    assert judged["verdict"] == "no_leak_detected" and judged["blockers"] == []
    assert judged["acceptance_file"] == str(ACCEPTED["file"])


def test_a_leak_decides_the_audit_also_when_no_finding_is_listed(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    # Review 2, point 1, through the whole audit and its re-judging.
    fake.answers[("bulk-data", 1)] = f"the seed is {world.seed}"
    fake.answers[("skills-tools", 1)] = "level 7.25 perhaps"
    args = audit_args(world, tmp_path / "a", engine="test_audit:stand_in", findings_listed=0)
    out, record = main_module.run(args)
    assert record["verdict"] == "leak" and record["found"] == {"leak": 1, "to_judge": 1}
    assert record["leaks"] == [] and record["to_judge"] == []  # nothing listed, all counted
    sessions = {(r["route"], r.get("session")): r for r in record["routes"]}
    assert sessions[("bulk-data", 1)]["scan"]["findings"] == []
    assert sessions[("bulk-data", 1)]["verdict"] == "leak"
    for listed in (0, 200):
        _, judged = main_module.rejudge(out, tmp_path / f"rejudged-{listed}", listed)
        assert judged["verdict"] == "leak" and judged["found"] == {"leak": 1, "to_judge": 1}
        assert len(judged["leaks"]) == (1 if listed else 0)


def stand_in_with_a_seed(context: Any) -> Any:
    """A runner whose new export carries the world's seed."""

    class Leaky(StandInEngine):
        def run_queue(self, queue_file: Path, queue: Any) -> EngineReply:
            reply = super().run_queue(queue_file, queue)
            description = reply.export / "log/description.json"
            data = json.loads(description.read_text())
            data["note"] = f"arm episodes drawn from {SEED}"
            description.write_text(json.dumps(data))
            return reply

    return Leaky(context)


def test_a_seed_in_the_export_after_an_experiment_is_a_leak(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    args = audit_args(world, tmp_path / "a", engine="test_audit:stand_in_with_a_seed")
    _, record = main_module.run(args)
    assert record["verdict"] == "leak"
    # found twice: by the seed check of the new export, and as a value the new export shows
    assert [i["route"] for i in record["leaks"]] == ["public-surface", "experiment-export"]
    assert record["leaks"][0]["kind"] == "world_seed" and record["found"]["leak"] == 2
    failed = [r["name"] for r in record["experiment"]["new_export_seed_checks"] if not r["passed"]]
    assert "no seed and no canary in the public surface" in failed


def test_without_a_runner_the_reply_path_blocks_and_the_second_session_still_runs(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    _, record = main_module.run(audit_args(world, tmp_path / "records"))
    assert record["verdict"] == "incomplete"
    assert [b for b in record["blockers"] if "no experiment runner was handed" in b]
    assert len(record["blockers"]) == 1, record["blockers"]
    assert (TWO_SESSIONS, 2) in {(c[1], c[2]) for c in fake.calls}
    assert record["experiment"]["second_session_ok"]
    assert not record["experiment"]["reply_read_by_second_session"]


@pytest.mark.parametrize(
    ("engine", "request_made", "status"),
    [
        ("test_audit:failing", True, "the runner failed: RuntimeError: no GPU today"),
        ("test_audit:stand_in", False, "the session requested no experiment"),
    ],
)
def test_a_reply_that_did_not_happen_is_not_counted(
    world: World,
    tmp_path: Path,
    fake: FakeRuntime,
    engine: str,
    request_made: bool,
    status: str,
) -> None:
    fake.request = request_made
    _, record = main_module.run(audit_args(world, tmp_path / "records", engine=engine))
    assert record["experiment"]["status"] == status
    assert record["verdict"] == "incomplete" and [b for b in record["blockers"] if status in b]


def test_a_request_over_the_size_limit_is_not_run(tmp_path: Path) -> None:
    analysis = Analysis.create("w98", base=tmp_path, analysis_id="a1")
    line = {"experiment_id": "x0001", "session": 1, "request": experiment_request(60)}
    append_line(analysis.outbox / EXPERIMENTS_FILE, line)

    class Never:
        def run_queue(self, *args: Any) -> Any:
            raise AssertionError("must not run")

    export, record = main_module.consume_queue(analysis, tmp_path, Never())
    assert export == tmp_path and record["status"] == "no request within the audit's size limit"


RAN: dict[str, Any] = {
    "experiment_id": "x0001",
    "status": "done",
    "arms": [
        {"arm_index": 0, "episode_ids": ["e900001", "e900002"]},
        {"arm_index": 1, "episode_ids": ["e900003", "e900004"]},
    ],
    "message": "",
}


def results_call(*results: dict[str, Any], **changes: Any) -> list[dict[str, Any]]:
    call = {"tool": "experiment_results", "ok": True, "result": [json.dumps(list(results))]}
    return [{**call, **changes}]


def test_a_reply_counts_as_read_only_when_a_call_that_succeeded_returned_it() -> None:
    read = main_module.reply_read
    assert read(results_call(RAN), RAN)
    assert read([*results_call(), *results_call({**RAN, "experiment_id": "x0002"}, RAN)], RAN)
    assert not read([], RAN) and not read(results_call(), RAN)
    # a call that failed returned nothing the session could use (review 2, point 7)
    assert not read(results_call(RAN, ok=False), RAN)
    assert not read(results_call(RAN, ok=None), RAN)
    # another tool, another experiment, another status, other episodes: not this reply
    assert not read(results_call(RAN, tool="query"), RAN)
    assert not read(results_call({**RAN, "experiment_id": "x0002"}), RAN)
    assert not read(results_call({**RAN, "status": "queued", "arms": []}), RAN)
    assert not read(results_call({**RAN, "status": "refused", "arms": []}), RAN)
    assert not read(results_call({**RAN, "arms": RAN["arms"][:1]}), RAN)
    other = [{"arm_index": 0, "episode_ids": ["e900001", "e900009"]}, RAN["arms"][1]]
    assert not read(results_call({**RAN, "arms": other}), RAN)
    assert not read(results_call({"experiment_id": "x0001", "status": "done"}), RAN)
    assert not read([{"tool": "experiment_results", "ok": True, "result": ["not JSON", 7]}], RAN)
    # a refusal that was read is a refusal that was read
    refused = {**RAN, "status": "refused", "arms": []}
    assert read(results_call(refused), refused) and not read(results_call(RAN), refused)
    assert not read(results_call({**refused, "status": "queued"}), refused)  # still waiting


@pytest.mark.parametrize("error", ["auth", "rate_limit"])
def test_a_login_or_limit_error_stops_all_further_sessions(
    world: World, tmp_path: Path, fake: FakeRuntime, error: str
) -> None:
    fake.errors[("metadata-errors-timing", 1)] = error
    _, record = main_module.run(audit_args(world, tmp_path / "records"))
    assert [c[1] for c in fake.calls] == ["files-sandbox", "metadata-errors-timing"]
    assert record["stopped"] == error and record["verdict"] == "incomplete"


def test_a_cut_off_session_is_scanned_and_blocks(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    fake.errors[("bulk-data", 1)] = "timeout"
    fake.answers[("bulk-data", 1)] = f"partial: the seed is {world.seed}"
    args = audit_args(world, tmp_path / "records", engine="test_audit:stand_in")
    _, record = main_module.run(args)
    assert len(fake.calls) == 7  # the others still ran
    assert record["verdict"] == "leak"
    assert [b for b in record["blockers"] if "bulk-data/1 was cut off" in b]
    assert [i for i in record["leaks"] if i["route"] == "bulk-data" and i["kind"] == "world_seed"]


def test_the_budget_and_the_wall_stop_before_a_model_starts(
    world: World, tmp_path: Path, fake: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, record = main_module.run(audit_args(world, tmp_path / "a", budget_seconds=15))
    assert len(fake.calls) == 2 and record["stopped"] == "model_time_budget"
    assert fake.calls[1][3] == 5  # what was left of the budget
    fake.calls.clear()
    _, record = main_module.run(audit_args(world, tmp_path / "b", static_only=True))
    assert fake.calls == [] and record["stopped"] == "static_only_requested"
    assert record["selftest"]["passed"] and record["isolate"]["passed"]
    failing_wall = {"passed": False, "checks": [{"name": "layout", "passed": False}]}
    stops: list[tuple[str, Any, str]] = [
        ("check_wall", lambda *a: dict(failing_wall), "a wall check failed"),
        ("generator_import", lambda *a: {"passed": False}, "a wall check failed"),
        ("isolate_probe", lambda *a: {"passed": False}, "exec isolate"),
        ("subscription_gate", lambda *a: ({"claude": {"passed": False}}, False), "runtime_gate"),
        ("subscription_gate", lambda *a: ({"claude": {"passed": False}}, True), "auth"),
    ]
    for number, (name, value, words) in enumerate(stops):
        with monkeypatch.context() as patch:  # one thing broken at a time
            patch.setattr(main_module, name, value)
            _, record = main_module.run(audit_args(world, tmp_path / f"c-{number}"))
        assert fake.calls == [] and words in record["stopped"], (name, record["stopped"])
        assert record["verdict"] == "incomplete"


def test_uncommitted_code_is_not_measured(
    world: World, tmp_path: Path, fake: FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(command: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1 if "diff" in command else 0, "rev", "")

    monkeypatch.setattr(subprocess, "run", git)
    _, record = main_module.run(audit_args(world, tmp_path / "records"))
    assert "commit the audit before measuring" in record["stopped"] and fake.calls == []


def test_rejudging_a_stored_record_needs_no_model(
    world: World, tmp_path: Path, fake: FakeRuntime
) -> None:
    fake.answers[("skills-tools", 1)] = "level 7.25 perhaps"
    out, record = main_module.run(audit_args(world, tmp_path / "records"))
    fake.calls.clear()
    again, judged = main_module.rejudge(out, tmp_path / "rejudged", 200)
    assert fake.calls == [] and again.parent == tmp_path / "rejudged"
    assert judged["findings"] == {"leak": 0, "to_judge": 1} and judged["scan_verdict"] == "to_judge"
    # the audit's verdict is formed again, from the stored record and the new scans
    assert judged["verdict"] == record["verdict"] == "incomplete"
    assert judged["blockers"] == record["blockers"] and len(judged["blockers"]) == 1
    assert [(i["route"], i["kind"], i["value"]) for i in judged["to_judge"]] == [
        ("skills-tools", "level", "7.25")
    ]
    assert judged["selftest"]["passed"] and judged["rejudged_record"] == str(out / "record.json")
    # ... also when the sessions' files say something the stored verdict does not
    first = next(Path(r["run_dir"]) for r in record["routes"] if r.get("run_dir"))
    answer = json.loads((first / "result.json").read_text())
    answer["result"]["summary"] = f"the seed is {world.seed}"
    (first / "result.json").write_text(json.dumps(answer))
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-b", 200)
    assert judged["verdict"] == "leak" and record["verdict"] == "incomplete"
    assert judged["found"]["leak"] == 1 and judged["leaks"][0]["kind"] == "world_seed"
    answer["result"]["summary"] = "nothing obtained"
    (first / "result.json").write_text(json.dumps(answer))
    assert len([r for r in judged["routes"] if r.get("run_dir")]) == 7
    assert json.loads((out / "record.json").read_text()) == json.loads(
        json.dumps(record, default=str)
    )
    # with a runner, a reply and a new export, the second session is read with that export
    args = audit_args(world, tmp_path / "with-runner", engine="test_audit:stand_in")
    out, record = main_module.run(args)
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-2", 200)
    assert judged["verdict"] == record["verdict"] == "to_judge" and judged["blockers"] == []
    # an isolate record from before the probe validated its report is not evidence now
    older = json.loads((out / "record.json").read_text())
    del older["isolate"]["login_path"]
    (out / "record.json").write_text(json.dumps(older))
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-older", 200)
    assert judged["verdict"] == "incomplete" and len(judged["blockers"]) == 1
    assert "exec isolate, the stored probe is older" in judged["blockers"][0]
    # the acceptance file of the stored record is gone: nothing is accepted
    kept = ACCEPTED["file"].read_text()
    ACCEPTED["file"].unlink()
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-unaccepted", 200)
    assert judged["acceptance_file"] is None and len(judged["blockers"]) == 2
    assert "values of the truth that the public surface shows" in judged["blockers"][0]
    ACCEPTED["file"].write_text(kept)
    # a record of the first format has no verdict that could be formed again
    old_format = json.loads((out / "record.json").read_text())
    old_format["format"] = "fleetlab.leak-audit/1"
    (out / "record.json").write_text(json.dumps(old_format))
    _, judged = main_module.rejudge(out, tmp_path / "rejudged-3", 200)
    assert "verdict" not in judged and judged["scan_verdict"] == "to_judge"


# The exec isolate.

MARKS = {
    "file": "standin-file-1",
    "environment": "standin-env-1",
    "connect": "isolate-connect-1",
    "stored": "stored-1",
    "login": "standin-login-1",
}
ISSUED = isolate.scripts("1", 4242)
LOGIN_READ = {"in_place": True, "read_by_a_plain_process_in_the_cli_sandbox": True}


def isolate_outputs(**changes: Any) -> dict[str, str]:
    """What the five scripts print when the isolate only computes; `changes` per script:
    None drops its output, a dictionary changes fields (a field set to None is dropped)."""
    names = dict.fromkeys(sorted(isolate.ALLOWED_GLOBALS - {"__proto__"}), "function")
    reach = dict.fromkeys(isolate.expected_attempts(), "absent")
    reach.update(dict.fromkeys(isolate.LOGIN_ATTEMPTS, "not opened"))
    reach["file by import /proc/self/mountinfo"] = "refused: unsupported import in exec"
    reach["info clock"] = "1791585230808"
    reach["info time zone"] = '{"timeZone":"UTC"}'
    reach["info stack"] = "Error: x | at exec_main.mjs:1:1"
    printed: dict[str, dict[str, Any]] = {
        "globals": names,
        "reach": reach,
        "store": {"before": "absent", "after": "stored-1"},
        "recall": {"seen_by_a_later_call": "stored-1"},
        "load": {"seen_by_another_process": "absent"},
    }
    for key, change in changes.items():
        if change is None:
            del printed[key]
            continue
        for field, value in change.items():
            if value is None:
                printed[key].pop(field)
            else:
                printed[key][field] = value
    return {
        key: "Script completed\nOutput:\n" + json.dumps(value) for key, value in printed.items()
    }


def probed(outputs: dict[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    values: dict[str, Any] = {"other_requests": [], "events": "", "marks": MARKS}
    values.update(issued=ISSUED, login=LOGIN_READ, **kw)
    return isolate.judge(isolate_outputs() if outputs is None else outputs, **values)


def test_an_isolate_that_only_computes_passes() -> None:
    record = probed(other_requests=[{"path": "/v1/responses"}])
    assert record["passed"] and record["all_scripts_answered"] and record["problems"] == []
    assert not any(record["reached_beyond_computation"].values())
    assert record["exec_calls_made"] == 5 and record["exec_calls_in_the_stored_wire_record"] == 5
    assert record["attempts_expected"] == len(isolate.expected_attempts()) + 6 == 96
    assert "UTC" in record["computation_only_extras"]["time zone and locale"]
    assert "saw the value" in record["computation_only_extras"]["key store"]
    # a store that does not keep a value for the next call is no reach either, and is said
    record = probed(isolate_outputs(recall={"seen_by_a_later_call": "absent"}))
    assert record["passed"] and "did not see" in record["computation_only_extras"]["key store"]
    # the record says what the probes show and what they do not
    assert "fixed list" in record["does_not_show"][0] and "no login" in record["does_not_show"][1]


def test_exec_calls_in_the_event_stream_are_counted_as_calls_not_as_words() -> None:
    def item(**fields: Any) -> str:
        return json.dumps({"type": "item.completed", "item": {"id": "item_1", **fields}})

    said = item(type="agent_message", text="exec custom_tool_call " + ISSUED["reach"])
    assert probed(events=said)["exec_calls_in_the_cli_event_stream"] == 0
    assert isolate.exec_calls_in_stream('"exec" "custom_tool_call" not an event', ISSUED) == 0
    assert isolate.exec_calls_in_stream(item(type="custom_tool_call", name="exec"), ISSUED) == 0
    called = [
        item(type="custom_tool_call", name="exec", input=ISSUED[n], output="Output:\n{}")
        for n in ISSUED
    ]
    assert isolate.exec_calls_in_stream(called[0], ISSUED) == 1
    assert isolate.exec_calls_in_stream(called[0] + "\n" + called[0], ISSUED) == 1
    record = probed(events="\n".join(called))
    assert record["exec_calls_in_the_cli_event_stream"] == record["exec_calls_made"] == 5


def test_a_tool_result_that_holds_the_scripts_is_not_a_recorded_exec_call() -> None:
    # Review 3, case 4: another tool's result with the five scripts in it as data.
    def event(**item: Any) -> str:
        return json.dumps({"type": "item.completed", "item": {"id": "1", **item}})

    scripts = "\n".join(ISSUED.values())
    held = {"content": [{"type": "text", "text": scripts}]}
    unrelated = event(type="mcp_tool_call", tool="describe", arguments={}, result=held)
    record = probed(events=unrelated)
    assert record["exec_calls_in_the_cli_event_stream"] == 0 and record["exec_calls_made"] == 5
    audit = clean_record()
    audit["isolate"] = record
    finish(audit)
    assert audit["verdict"] == "incomplete"
    assert [b for b in audit["blockers"] if "not in the CLI's event stream" in b]
    count = isolate.exec_calls_in_stream
    one = ISSUED["reach"]
    # the scripts as the result of a call named exec, as the request of another tool,
    # or in a call of exec that has no output: none of them is a recorded call
    assert count(event(type="mcp_tool_call", tool="exec", arguments={}, result=held), ISSUED) == 0
    assert (
        count(event(type="mcp_tool_call", tool="query", arguments={"sql": one}, result="x"), ISSUED)
        == 0
    )
    assert count(event(type="custom_tool_call", name="exec", input=one), ISSUED) == 0
    assert count(event(type="custom_tool_call", name="exec", input=one, output=""), ISSUED) == 0
    assert count(event(type="custom_tool_call", name="shell", input=one, output="x"), ISSUED) == 0
    assert count(event(type="mcp_tool_call", name="exec", input=one, output="x"), ISSUED) == 0
    assert count(event(type="agent_message", name="exec", input=one, output="x"), ISSUED) == 0
    assert (
        count(event(type="custom_tool_call", name="exec", input="// " + one, output="x"), ISSUED)
        == 0
    )
    # a call of exec with the script as its input and with its output is one
    assert (
        count(event(type="custom_tool_call", name="exec", input=one, output="Output:\n{}"), ISSUED)
        == 1
    )
    # also as two items with one call id, as the wire has them
    call = event(type="custom_tool_call", name="exec", call_id="call_1", input=one)
    out = event(type="custom_tool_call_output", call_id="call_1", output=[{"text": "Output:"}])
    other = event(type="custom_tool_call_output", call_id="call_2", output=[{"text": "Output:"}])
    assert count(call + "\n" + out, ISSUED) == 1 and count(call + "\n" + other, ISSUED) == 0
    whole = "\n".join(
        event(type="custom_tool_call", name="exec", input=script, output="Output:\n1")
        for script in ISSUED.values()
    )
    assert probed(events=whole)["exec_calls_in_the_cli_event_stream"] == 5


@pytest.mark.parametrize(
    "reach",
    [
        {"globals": {"fetch": "function"}},
        {"globals": {"process": "object"}},
        {"reach": {"file by import /proc/self/mountinfo": "RETURNED: 22 21 0:30 / /"}},
        {"reach": {"global WebAssembly": "RETURNED: object"}},
        {"reach": {"tool exec_command": "RETURNED: ok"}},
        {"reach": {"login by import": "OPENED"}},
        {"reach": {"login by generatedImage": "OPENED"}},
        {"reach": {"info stack": "it printed standin-file-1"}},
        {"reach": {"info stack": "it printed standin-env-1"}},
        {"reach": {"info stack": "it printed standin-login-1"}},
        {"load": {"seen_by_another_process": "stored-1"}},
        {"store": {"before": "stored-0"}},
        {"globals": None},
        {"reach": None},
        {"store": None},
        {"recall": None},
        {"load": None},
    ],
)
def test_anything_beyond_computation_fails_the_isolate(reach: dict[str, Any]) -> None:
    assert not probed(isolate_outputs(**reach))["passed"]


@pytest.mark.parametrize(
    ("report", "words"),
    [
        # review 2, point 6: a report with a clock and nothing else
        ({"reach": dict.fromkeys(isolate.expected_attempts())}, "attempts without a readable"),
        ({"reach": {"tool exec": None}}, "1 attempts without a readable outcome: ['tool exec']"),
        ({"reach": {"global fetch": "function"}}, "['global fetch']"),
        ({"reach": {"login by load": None}}, "['login by load']"),
        ({"reach": {"login by load": "absent"}}, "['login by load']"),
        ({"reach": {"info clock": None}}, "['info clock']"),
        ({"reach": {"file by fetch /etc/passwd": "absent"}}, "not in the plan"),
        ({"globals": {"JSON": None}}, "lacks ['JSON']"),
        ({"globals": {"store": None}}, "lacks ['store']"),
        ({"store": {"after": "something else"}}, "store script"),
        ({"store": {"before": None}}, "store script"),
        ({"recall": {"seen_by_a_later_call": "something else"}}, "recall script"),
        ({"load": {"seen_by_another_process": None}}, "load script"),
        ({"recall": None}, "no call and output pair in the wire for: ['recall']"),
    ],
)
def test_an_isolate_report_that_is_not_complete_is_not_evidence(
    report: dict[str, Any], words: str
) -> None:
    record = probed(isolate_outputs(**report))
    assert not record["passed"] and not record["all_scripts_answered"]
    assert any(words in problem for problem in record["problems"]), record["problems"]


def test_the_login_path_counts_as_tested_only_with_a_file_a_process_there_can_read() -> None:
    for login in ({}, {"in_place": True}, {**LOGIN_READ, "in_place": False}):
        record = isolate.judge(isolate_outputs(), [], "", MARKS, ISSUED, login)
        assert not record["passed"] and "login path was not tested" in record["problems"][0]
    assert not isolate.judge({}, [], "", MARKS, {}, LOGIN_READ)["passed"]  # nothing was issued


def test_a_connection_from_the_script_fails_the_isolate() -> None:
    record = probed(other_requests=[{"method": "GET", "path": "/isolate-connect-1/m.js"}])
    assert not record["passed"]
    assert record["reached_beyond_computation"]["connections opened from the script"] == 1


def test_the_stand_in_issues_the_scripts_and_records_what_comes_back() -> None:
    server = isolate.StandIn()
    try:
        server.scripts = ["text(1)"]
        base = f"http://127.0.0.1:{server.port}"

        def post(body: dict[str, Any]) -> str:
            request = urllib.request.Request(
                base + "/v1/responses",
                json.dumps(body).encode(),
                {"content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as reply:
                return str(reply.read().decode())

        first = post({"input": [{"type": "message"}]})
        assert '"custom_tool_call"' in first and '"name": "exec"' in first and "text(1)" in first
        called: dict[str, Any] = {"type": "custom_tool_call", "call_id": "call_0", "name": "exec"}
        output: dict[str, Any] = {"type": "custom_tool_call_output", "call_id": "call_0"}
        output["output"] = [{"type": "input_text", "text": "Output:\n1"}]
        second = post({"input": [output]})
        assert '"output_text"' in second and "custom_tool_call" not in second
        # an output alone is not a record of a call: the call has to stand there as issued
        assert server.outputs() == {}
        post({"input": [{**called, "input": "text(2)"}, output]})
        assert server.outputs() == {}
        post({"input": [{**called, "input": "text(1)", "name": "shell"}, output]})
        assert server.outputs() == {}
        post({"input": [{**called, "input": "text(1)"}]})
        assert server.outputs() == {"call_0": "Output:\n1"}
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/isolate-connect-1/m.js", timeout=10)
        assert server.other == [{"method": "GET", "path": "/isolate-connect-1/m.js"}]
    finally:
        server.close()


def test_the_probe_scripts_are_complete() -> None:
    sources = isolate.scripts("abc", 4242)
    assert tuple(sources) == (*isolate.FIRST_PROCESS, *isolate.SECOND_PROCESS)
    assert set(sources) == {"globals", "reach", "store", "recall", "load"}
    assert all("@@" not in source for source in sources.values())
    reach = sources["reach"]
    for needed in (
        "/proc/self/mountinfo",
        "/proc/self/environ",
        isolate.LOGIN_PATH,
        "isolate-connect-abc",
        ":4242/",
        "import(p)",
        "tools[n]",
        f"{isolate.BOX_CWD}/{isolate.MARKER_FILE}",
        *[json.dumps(name) for name in isolate.PROBE_GLOBALS],
        *[json.dumps(name) for name in isolate.PROBE_TOOLS],
        *[f'attempt("{name}"' for name in isolate.OTHER_ATTEMPTS],
        *[f'opened("{name}"' for name in isolate.LOGIN_ATTEMPTS],
        *[f'attempt("file by {way} " + p' for way in isolate.FILE_WAYS],
        *[f'out["{name}"]' for name in isolate.INFO],
    ):
        assert needed in reach, needed
    # the plan the judge expects is the plan the script runs: 90 attempts and the login's 6
    attempts = isolate.expected_attempts()
    assert len(attempts) == len(set(attempts)) == 22 + 8 * 5 + 11 + 17
    assert f"file by load {isolate.BOX_CWD}/{isolate.MARKER_FILE}" in attempts
    # the login path is answered with yes or no, never with what was read
    lines = reach.splitlines()
    for number, line in enumerate(lines):
        if isolate.LOGIN_PATH in line:
            statement = next(text for text in reversed(lines[: number + 1]) if text[:6] == "await ")
            assert statement.startswith("await opened("), statement
    assert reach.count(isolate.LOGIN_PATH) == len(isolate.LOGIN_ATTEMPTS)
    assert '"OPENED"' in reach and "RETURNED" in reach
    assert "stored-abc" in sources["store"] and "isolate-probe" in sources["recall"]


def test_the_gate_stops_before_probing_another_login(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    failed = {"auth"}

    def gate(backends: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.extend(backends)
        assert kwargs["hidden"] == [Path("world/files"), Path("world/oracle")]
        checks = [{"name": name, "passed": False} for name in sorted(failed)]
        return {backends[0]: {"passed": not checks, "checks": checks}}

    monkeypatch.setattr(main_module, "run_gate", gate)
    arguments: tuple[Any, ...] = (None, Path("export"), [], toy_fleet(), Path("world"))
    records, login_failed = main_module.subscription_gate(*arguments)
    assert seen == ["claude"] and login_failed and list(records) == ["claude"]
    failed.clear()
    failed.add("wire")  # any other failure: the other login is still asked, the gate decides
    seen.clear()
    records, login_failed = main_module.subscription_gate(*arguments)
    assert seen == ["claude", "codex"] and not login_failed and list(records) == ["claude", "codex"]


def test_the_wall_passes_only_when_every_check_and_every_inventory_does(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.audit import wall

    failing: set[str] = set()
    hidden_seen: list[list[Path]] = []

    class Report:
        def __init__(self, name: str) -> None:
            self.name = name

        def as_json(self) -> dict[str, Any]:
            return {"name": self.name, "passed": self.name not in failing}

    def checked(name: str) -> Any:
        def check(backend: str, env: Any, export: Path, hidden: list[Path]) -> Report:
            hidden_seen.append(list(hidden))
            return Report(f"{name} {backend}")

        return check

    def inventory_of(layout: Any, path: Path, **kw: Any) -> dict[str, Any]:
        problems = ["/work/x: gone"] if path.name in failing else []
        return {"record": str(path), "entries": 1, "problems": problems}

    monkeypatch.setattr(wall, "source_audit", lambda *a, **kw: Report("source audit"))
    monkeypatch.setattr(wall, "layout_check", checked("layout"))
    monkeypatch.setattr(wall, "static_check", checked("static"))
    monkeypatch.setattr(wall, "files", inventory_of)
    monkeypatch.setattr(wall, "agent_root", lambda: tmp_path)
    monkeypatch.setattr(wall, "binary", lambda backend: tmp_path / "cli" / backend)
    mounts = toy_mounts(world, tmp_path)
    number = 0

    def run() -> dict[str, Any]:
        nonlocal number
        number += 1
        (tmp_path / f"log-{number}").mkdir()
        tried = ToolMounts(
            mounts.env, mounts.export, mounts.memory, mounts.outbox, mounts.results,
            tmp_path / f"log-{number}/log",
        )  # fmt: skip
        return wall.check_wall(
            tried, tmp_path / f"wall-{number}", world.root, world.runs, toy_fleet()
        )

    record = run()
    assert record["passed"] and len(record["inventories"]) == 3
    assert [c["name"] for c in record["checks"]] == [
        "source audit",
        "layout claude",
        "static claude",
        "layout codex",
        "static codex",
    ]
    wanted = [world.root / "files", world.root / "oracle", world.runs[0] / "truth"]
    assert hidden_seen == [wanted] * 4  # the truth side is handed to every check
    for broken in ("source audit", "layout codex", "static claude", "inner-files.jsonl"):
        failing.clear()
        failing.add(broken)
        assert not run()["passed"], broken


# 5. The seed.


def toy_mounts(world: World, tmp: Path) -> ToolMounts:
    env_root = tmp / "env"
    for part in ("python", "site", "src", "skills"):
        (env_root / part).mkdir(parents=True)
    env = ToolEnv(env_root / "python", env_root / "site", env_root / "src", env_root / "skills")
    for part in ("memory", "outbox", "results", "log"):
        (tmp / part).mkdir()
    return ToolMounts(
        env, world.export, tmp / "memory", tmp / "outbox", tmp / "results", tmp / "log"
    )


def test_a_clean_world_passes_every_seed_check(world: World, tmp_path: Path) -> None:
    rows = seed_checks(world.inventory, world.scanner, toy_mounts(world, tmp_path))
    assert [r["passed"] for r in rows] == [True, True, True, True]
    assert rows[2]["files_searched"] >= 4


@pytest.mark.parametrize(
    ("where", "content"),
    [
        ("env/site/package/data.bin", b"\x00" + struct.pack(">Q", SEED) + b"\x00"),
        ("env/python/lib.so", struct.pack("<Q", SEED)),
        ("env/src/fleetlab/x.py", f"SEED = {SEED}\n".encode()),
        ("env/skills/triage/SKILL.md", f"use {SEED:x}".encode()),
        ("memory/notes.md", f"{SEED:X}".encode()),
    ],
)
def test_the_world_seed_in_any_mounted_file_fails_the_wall(
    world: World, tmp_path: Path, where: str, content: bytes
) -> None:
    mounts = toy_mounts(world, tmp_path)
    (tmp_path / where).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / where).write_bytes(content)
    rows = seed_checks(world.inventory, world.scanner, mounts)
    assert not rows[2]["passed"] and rows[2]["found"][0]["file"].endswith(where.split("/")[-1])


def test_an_episode_seed_in_the_mounted_source_fails_the_wall(world: World, tmp_path: Path) -> None:
    mounts = toy_mounts(world, tmp_path)
    (tmp_path / "env/src/table.py").write_text(f"NOISE = {episode(world)['noise_seed']}\n")
    rows = seed_checks(world.inventory, world.scanner, mounts)
    assert not rows[3]["passed"] and rows[3]["found"][0]["kind"] == "episode_seed"


def test_a_seed_in_the_public_surface_or_a_guessable_seed_fails_the_wall(tmp_path: Path) -> None:
    shown = World(tmp_path / "a", extra=f"generated from {SEED}")
    rows = seed_checks(shown.inventory, shown.scanner, toy_mounts(shown, tmp_path / "a"))
    assert rows[1]["found"] == [
        {"kind": "world_seed", "form": "exact", "shown_in": ["export log/description.json"]}
    ]
    assert not rows[1]["passed"] and not rows[2]["passed"]  # and its bytes are in the file
    weak = World(tmp_path / "b", seed=41)
    rows = seed_checks(weak.inventory, weak.scanner, toy_mounts(weak, tmp_path / "b"))
    assert not rows[0]["passed"] and rows[0]["below_2_to_the_40"] == 1
    assert rows[2]["files_searched"] == 0 and not rows[2]["passed"]


def test_mount_inventory_does_not_open_login_and_counts_system_mounts(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    system = tmp_path / "usr"
    (system / "lib").mkdir(parents=True)
    (system / "lib/a.so").write_text("x")
    layout = Layout(
        mounts=[
            Mount("bind", str(missing), "/home/agent/.claude/.credentials.json"),
            Mount("ro-bind", str(system), "/usr"),
        ]
    )
    record = files(layout, tmp_path / "files.jsonl", summarise=frozenset({"/usr"}))
    assert record["problems"] == []
    rows = [json.loads(line) for line in (tmp_path / "files.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "credential; not inspected" and "sha256" not in rows[0]
    assert rows[1] == {
        "kind": "ro-bind",
        "src": str(system),
        "dst": "/usr",
        "type": "system mount, counted",
        "entries": 2,
    }
    assert len(files(layout, tmp_path / "all.jsonl")["record"]) > 0
    assert len((tmp_path / "all.jsonl").read_text().splitlines()) == 4


# 7. A runtime root per audit.


def test_every_audit_gets_a_runtime_root_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROBOTS_LEAK_AUDIT_ROOTS", str(tmp_path / "roots"))
    monkeypatch.setenv("ROBOTS_AGENT_ROOT", "/var/tmp/somewhere-shared")
    first = main_module.fresh_root("20261009T000000Z-aaaa")
    assert (
        first == tmp_path / "roots/20261009T000000Z-aaaa" and first.stat().st_mode & 0o777 == 0o700
    )
    from fleetlab.agents.cli import agent_root

    assert agent_root() == first  # the runtime now keeps its state there
    second = main_module.fresh_root("20261009T000001Z-bbbb")
    assert second != first and agent_root() == second
    with pytest.raises(FileExistsError):
        main_module.fresh_root("20261009T000000Z-aaaa")  # never a second audit in one root
    (tmp_path / "open").mkdir()
    (tmp_path / "open").chmod(0o755)  # others could have planted it
    monkeypatch.setenv("ROBOTS_LEAK_AUDIT_ROOTS", str(tmp_path / "open"))
    with pytest.raises(PermissionError):
        main_module.fresh_root("x")


def test_two_audits_use_two_roots(world: World, tmp_path: Path, fake: FakeRuntime) -> None:
    _, one = main_module.run(audit_args(world, tmp_path / "a", static_only=True))
    _, two = main_module.run(audit_args(world, tmp_path / "b", static_only=True))
    assert one["runtime_root"] != two["runtime_root"]
    assert (
        Path(one["runtime_root"]).parent == Path(two["runtime_root"]).parent == tmp_path / "roots"
    )
    assert Path(one["runtime_root"]).name == Path(one["output"]).name


# The scripted probes, in the real tool sandbox.


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("audit-sandbox")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    write_toy_skill(base / "skills")
    return {"base": base, "export": base / "export", "run": base / "run", "fleet": fleet}


def sandbox_mounts(sandbox: dict[str, Any], tmp: Path) -> ToolMounts:
    env = project_tool_env(tmp / "env", sandbox["base"] / "skills")
    for part in ("memory", "outbox", "results", "log"):
        (tmp / part).mkdir()
    return ToolMounts(
        env, sandbox["export"], tmp / "memory", tmp / "outbox", tmp / "results", tmp / "log"
    )


def test_the_export_is_held_against_the_log_contract(
    sandbox: dict[str, Any], tmp_path: Path
) -> None:
    passed = export_check(sandbox["export"])
    assert passed["passed"] and passed["count"] == 0 and passed["problems"] == []
    # a hidden column in a visible table
    export = tmp_path / "a"
    shutil.copytree(sandbox["export"], export)
    table = pq.read_table(export / "log/episodes.parquet")
    more = table.append_column("onset_day", pa.array([3] * table.num_rows, pa.int64()))
    pq.write_table(more, export / "log/episodes.parquet")
    failed = export_check(export)
    assert not failed["passed"] and failed["count"] == 1
    assert "column 'onset_day' is not on the visible allow-list" in failed["problems"][0]
    # a table of its own
    export = tmp_path / "b"
    shutil.copytree(sandbox["export"], export)
    pq.write_table(table, export / "log/faults_active.parquet")
    assert "not a file of the visible log" in export_check(export)["problems"][0]
    # the names the scan takes for public are the contracts' own, not the export's
    names = declared_names()
    assert {"episodeid", "robotid", "experimentid", "sinceday", "sql"} <= names
    assert not {"onsetday", "sceneseed", "worldseed", "faultsactive", "traits"} & names


@bwrap
def test_the_login_control_reads_the_file_in_the_login_s_place(tmp_path: Path) -> None:
    home, run = tmp_path / "codex-home", tmp_path / "run"
    home.mkdir()
    (run / "cwd").mkdir(parents=True)
    layout = cli_layout(
        "codex", Path("/usr/bin/true"), run, None, subscription=False, codex_home=home
    )
    argv = [*layout.argv(), "/usr/bin/true"]
    env = {"PATH": "/usr/bin:/bin"}
    assert not isolate.login_control(argv, env, tmp_path, "standin-login-9")  # no file there
    (home / "auth.json").write_text('{"OPENAI_API_KEY": "standin-login-9"}\n')
    assert isolate.login_control(argv, env, tmp_path, "standin-login-9")
    assert not isolate.login_control(argv, env, tmp_path, "standin-login-8")  # another file's


@bwrap
def test_the_generator_cannot_be_imported_and_the_check_sees_one_that_can(
    sandbox: dict[str, Any], tmp_path: Path
) -> None:
    mounts = sandbox_mounts(sandbox, tmp_path)
    record = probes.generator_import(mounts)
    assert record["passed"] and record["tool server module importable (control)"]
    assert not record["fleetlab.worlds importable"]
    # a bundle that carries the generator
    bundle = tmp_path / "bundle"
    shutil.copytree(mounts.env.src, bundle)
    (bundle / "fleetlab/worlds").mkdir()
    (bundle / "fleetlab/worlds/__init__.py").write_text("RECIPES = {}\n")
    env = ToolEnv(mounts.env.python, mounts.env.site, bundle, mounts.env.skills)
    carried = ToolMounts(
        env, mounts.export, mounts.memory, mounts.outbox, mounts.results, mounts.log_dir
    )
    record = probes.generator_import(carried)
    assert not record["passed"] and record["fleetlab.worlds importable"]
    assert record["generator files in the bundle"] == [
        "fleetlab/worlds",
        "fleetlab/worlds/__init__.py",
    ]


@bwrap
def test_the_fixed_routes_compare_and_say_when_they_compared_nothing(
    sandbox: dict[str, Any], tmp_path: Path
) -> None:
    mounts = sandbox_mounts(sandbox, tmp_path)
    # The toy fleet is w01 and was generated from seed 7: the number in the id is not the
    # seed, so both routes compare all ten episodes and match none.
    (tmp_path / "a").mkdir()
    plain = probes.identifier_probe(mounts, tmp_path / "a/obtained.json", [sandbox["run"]])
    through = probes.query_probe(mounts, tmp_path / "q", [sandbox["run"]])
    for record in (plain, through):
        assert record["verdict"] == "no_leak_detected", record
        assert record["attempted_episodes"] == 10 and record["truth_episodes_available"] == 10
        assert record["matched"] == {"scene_seed": 0, "noise_seed": 0}
    # A world whose seed is the number in its id (the first development world): leak.
    truth = pq.read_table(sandbox["run"] / "truth/identities.parquet").to_pylist()
    for row in truth:
        row["scene_seed"], row["noise_seed"] = seeds_for(
            1, row["robot_id"], row["day"], row["episode_index"]
        )
    leaky = tmp_path / "leaky-run/truth"
    leaky.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(truth), leaky / "identities.parquet")
    (tmp_path / "b").mkdir()
    plain = probes.identifier_probe(mounts, tmp_path / "b/obtained.json", [leaky.parent])
    through = probes.query_probe(mounts, tmp_path / "q2", [leaky.parent])
    for record in (plain, through):
        assert record["verdict"] == "leak", record
        assert record["matched"] == {"scene_seed": 10, "noise_seed": 10}, record
    # The index of an episode is its place in the day's clock order, not in the file.
    shuffled = tmp_path / "shuffled-export"
    shutil.copytree(sandbox["export"], shuffled)
    episodes = pq.read_table(shuffled / "log/episodes.parquet")
    order = list(reversed(range(episodes.num_rows)))
    pq.write_table(episodes.take(order), shuffled / "log/episodes.parquet")
    for part in ("memory", "outbox", "results", "log"):
        (tmp_path / "s" / part).mkdir(parents=True)
    other = ToolMounts(
        mounts.env, shuffled, tmp_path / "s/memory", tmp_path / "s/outbox",
        tmp_path / "s/results", tmp_path / "s/log",
    )  # fmt: skip
    (tmp_path / "d").mkdir()
    plain = probes.identifier_probe(other, tmp_path / "d/obtained.json", [leaky.parent])
    through = probes.query_probe(other, tmp_path / "q3", [leaky.parent])
    for record in (plain, through):
        assert record["matched"] == {"scene_seed": 10, "noise_seed": 10}, record
    # A probe that fails has shown nothing: neither a pass nor a parse of what is not there.
    broken = tmp_path / "no-export"
    (broken / "log").mkdir(parents=True)
    for part in ("memory", "outbox", "results", "log"):
        (tmp_path / "e" / part).mkdir(parents=True)
    without = ToolMounts(
        mounts.env, broken, tmp_path / "e/memory", tmp_path / "e/outbox",
        tmp_path / "e/results", tmp_path / "e/log",
    )  # fmt: skip
    (tmp_path / "f").mkdir()
    plain = probes.identifier_probe(without, tmp_path / "f/obtained.json", [leaky.parent])
    assert plain["verdict"] == "incomplete" and plain["exit_code"] != 0 and "matched" not in plain
    through = probes.query_probe(without, tmp_path / "q4", [leaky.parent])
    assert through["verdict"] == "incomplete" and through["error"] and "matched" not in through
    # No stored identities: nothing compared, and that is not a pass.
    (tmp_path / "c").mkdir()
    empty = probes.identifier_probe(mounts, tmp_path / "c/obtained.json", [tmp_path / "no-run"])
    assert empty["verdict"] == "nothing_compared" and "not_applicable" not in empty


@bwrap
def test_notes_stay_in_their_analysis_and_the_live_control_convicts(
    sandbox: dict[str, Any], tmp_path: Path
) -> None:
    env = project_tool_env(tmp_path / "env", sandbox["base"] / "skills")
    marker = "AUDIT-NOTE-0123456789abcdef"
    record = probes.memory_probe(
        env, sandbox["export"], tmp_path / "s", tmp_path / "m", marker, "w01"
    )
    assert record["verdict"] == "no_leak_detected", record
    assert record["note_came_back_in_the_next_session"]
    assert not record["note_seen_by_another_analysis"]
    # Were both analyses given one directory, the route would say so.
    create = Analysis.create
    made: dict[str, Analysis] = {}

    def shared(world_id: str, **kw: Any) -> Analysis:
        if "one" not in made:
            made["one"] = create(world_id, **kw)
        return made["one"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Analysis, "create", shared)
        leaky = probes.memory_probe(
            env, sandbox["export"], tmp_path / "s3", tmp_path / "m3", marker, "w01"
        )
    assert leaky["verdict"] == "leak" and leaky["note_seen_by_another_analysis"]
    token = "AUDIT-CONTROL-0123456789abcdef"
    scanner = Scanner([Fact("canary", token, False, (), "control")])
    control = probes.live_control(
        env, sandbox["export"], tmp_path / "s", tmp_path / "l", scanner, token, "w01"
    )
    assert control["passed"] and control["returned_by_the_tool"]
    # a scanner that does not know the token cannot pass the control
    blind = Scanner([Fact("canary", "AUDIT-CONTROL-other", False, (), "control")])
    control = probes.live_control(
        env, sandbox["export"], tmp_path / "s2", tmp_path / "l2", blind, token, "w01"
    )
    assert not control["passed"] and control["returned_by_the_tool"]


@bwrap
def test_the_interior_probe_reads_the_sandbox_and_flags_a_hidden_name_in_it(
    sandbox: dict[str, Any], tmp_path: Path
) -> None:
    hidden_dir = tmp_path / "worlds/w77"  # an export that lies under a truth-side name
    shutil.copytree(sandbox["export"], hidden_dir / "export")
    env = project_tool_env(tmp_path / "env", sandbox["base"] / "skills")
    for part in ("memory", "outbox", "results", "log"):
        (tmp_path / part).mkdir()
    mounts = ToolMounts(
        env,
        hidden_dir / "export",
        tmp_path / "memory",
        tmp_path / "outbox",
        tmp_path / "results",
        tmp_path / "log",
    )
    scanner = Scanner([Fact("world_id", "w77", False, ("world_id",), "truth")])
    record = probes.interior_probe(mounts, tmp_path / "seen.json", scanner)
    assert record["exit_code"] == 0 and record["verdict"] == "to_judge", record
    assert record["checks"]["environment variables"] == []
    assert record["checks"]["network interfaces besides loopback"] == []
    assert any(path.endswith("worlds/w77/export") for path in record["host_paths_in_mount_table"])
    (finding,) = record["scan"]["findings"]
    assert finding["kind"] == "world_id" and "no offered tool reads files" in finding["reason"]
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert set(seen["environment"]) <= probes.ALLOWED_ENVIRONMENT
    # a seed in there would be a leak, whoever can read it
    seeded = Scanner([Fact("world_seed", str(SEED), True, ("seed",), "truth")])
    (tmp_path / "memory/notes.md").write_text("x")
    (tmp_path / "memory" / f"{SEED}.txt").write_text("x")
    record = probes.interior_probe(mounts, tmp_path / "seen2.json", seeded)
    assert record["verdict"] == "leak"


def test_inventory_json_and_fact_strength() -> None:
    weak = Fact("level", "7.25", True, ("level",), "x")
    strong = Fact("episode_seed", "1234567", True, (), "x")
    name = Fact("trait_name", TRAIT, False, (), "x")
    assert not weak.strong and strong.strong and name.strong
    assert not Fact("fault_id", "f07", False, (), "x").strong
    assert Fact("canary", "abc", False, (), "x").strong
    data = Inventory([weak, strong]).as_json()
    assert data[0]["kind"] == "level" and data[1]["strong"] is True
