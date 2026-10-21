"""Round 9: a draw leaves a record, and the search cannot change unnoticed.

The first part was written before anything else in the round, on the code as round 8
left it, and pins what the search composes (decision 3: the search is frozen): the
world of the produced recipe on two public seeds, and the five single-place breaks the
review of round 8 found surviving. The parts after it cover the recipe file, the
import's check of a folded world's files and, in its final form of round 9c, the ledger
of draws: the seed derived from a public beacon and recorded before it is composed, a
draw killed or interrupted and repeated, the lock, the chain, places in order, entry
16's check as a command, and the recipe-file command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_worlds_round8 import _attempts, _draw, _job, _one_plain, _plain_composer, _settled

from fleetlab.worlds import calibration, compose, curves
from fleetlab.worlds import ledger as book
from fleetlab.worlds.recipe import (
    TEST_WORLD_SOURCES,
    WorldRecipe,
    code_recipe,
    external_recipe,
    private_recipe,
    recipe_hash,
    stated_hash,
)

SEEDS = compose.ASSESSMENT_SEEDS
OTHER = tuple((1 << 62) + 7700000 + k for k in range(20))  # the review of round 7's seeds
WORLDS = Path.home() / ".cache/robots-probes/worlds"

# ---- 3. the search cannot change unnoticed -----------------------------------------------

# Public seeds on which the produced recipe composes within seconds: the accepted draft,
# the world's hash and the forecast seal, under the world id w97. A change to the search,
# to a cell or to a setting of the recipe that moves a composition moves one of these.
PINNED = [
    (
        SEEDS[10],
        6,
        "b72feef4b5c7a04815774bb57a3144d9db2414f4df60470318081c2bc195793c",
        "65dc118384669d6282b5d692a76b24d6cefe424bf25c5a1adc1bd976b3b45a26",
    ),
    (
        OTHER[13],
        2,
        "7c735b6febbb9eb3e502a8571ab7bb4149632cdd77498cd0f5523affa782a3ee",
        "10f20f1a161811f92999c6b533588c24c7246d23a3da3f9dd11909bcc139eb6a",
    ),
]


def _texts() -> dict[Any, str]:
    from test_worlds_generation4 import TEXTS

    from fleetlab.contracts.identity import Task

    return {Task(row["suite"], row["task_index"]): row["text"] for row in TEXTS}


def _software() -> Any:
    from test_worlds_generation4 import SEALS

    from fleetlab.contracts.identity import SoftwareVersion

    return SoftwareVersion.model_validate(SEALS["software"])


@pytest.mark.parametrize(("seed", "draft", "digest", "seal"), PINNED)
def test_the_produced_recipe_composes_the_pinned_world(
    seed: int, draft: int, digest: str, seal: str
) -> None:
    from fleetlab.worlds.world import generate, world_hash

    recipe = WorldRecipe.model_validate({**private_recipe("w97", "w07", name="T1"), "seed": seed})
    world = generate(recipe, _software(), _texts())
    assert world.composition.attempt == draft
    assert world_hash(world) == digest
    assert world.composition.expected["forecast_seal"] == seal


BATCH = ("object_batch_variant", "4 mass doublings")


def test_a_taken_drawn_day_is_replaced_by_a_drawn_free_day_not_the_earliest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _plain_composer("r8", monkeypatch)
    # Faults at the robot's site start on days 1 and 2; days 0, 3 and 4 are free.
    taken = compose.Draft(fixed=[compose.make_fault(f"f0{d}", *BATCH, "s1", d) for d in (1, 2)])
    replaced = []
    for stream in range(40):
        drawn, placed = _draw(c, stream, compose.Draft()), _draw(c, stream, taken)
        assert drawn is not None and placed is not None
        if drawn.onset in (1, 2):
            replaced.append(placed.onset)
        else:
            assert placed.onset == drawn.onset
    # Every free day is drawn as a replacement. The earliest alone (day 0) would start a
    # quarter of all single-robot faults on day 0, which decision 2 of round 8 forbids.
    assert len(replaced) >= 8 and set(replaced) == {0, 3, 4}


def test_a_site_wide_fault_is_never_replaced_onto_day_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _plain_composer("r8", monkeypatch)
    c.plain_menu = [("action_latency", "site")]
    # Both sites have a fault starting on each of days 2 to 5: day 1 is the one free day
    # of a site-wide fault's range. Day 0 is free too, and outside that range.
    taken = compose.Draft(
        fixed=[
            compose.make_fault(f"f{10 * s + d}", *BATCH, f"s{s}", d)
            for s in (1, 2)
            for d in (2, 3, 4, 5)
        ]
    )
    drawn_days = set()
    for stream in range(20):
        drawn, placed = _draw(c, stream, compose.Draft()), _draw(c, stream, taken)
        assert drawn is not None and placed is not None
        assert placed.scope in ("s1", "s2") and placed.onset == 1
        drawn_days.add(drawn.onset)
    assert drawn_days <= {1, 2, 3, 4, 5} and len(drawn_days) >= 4


def test_settling_moves_a_plain_fault_later_as_well_as_earlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forecast(days: dict[int, float]) -> Any:
        return lambda schedule: days[schedule[0].schedule.onset_day]

    # Too much loss on day 2: one day later is on target.
    later = _settled(
        "r8",
        _one_plain("plain", 2, 0, 4),
        forecast({0: 0.40, 1: 0.50, 2: 0.60, 3: 0.675, 4: 0.80}),
        monkeypatch,
    )
    assert later[0].schedule.onset_day == 3
    # Too little: one day earlier is.
    earlier = _settled(
        "r8",
        _one_plain("plain", 2, 0, 4),
        forecast({0: 0.50, 1: 0.675, 2: 0.80, 3: 0.90, 4: 0.95}),
        monkeypatch,
    )
    assert earlier[0].schedule.onset_day == 1


def _folded_world(
    root: Path, edit: Any = None, *, reseal: bool = True, drop: str | None = None
) -> Path:
    """A scratch copy of the folded world w47 with all of its sealed files and both runs."""
    import shutil

    from test_worlds_round8 import _copy_run

    from fleetlab.contracts.world import HASH_FILE, WORLD_FILES, seal_world

    source = WORLDS / "w47-opaque-v1"
    if not (source / "files/manifest.jsonl").is_file():
        pytest.skip("the folded world is not installed on this host")
    (root / "files").mkdir(parents=True)
    for name in (*WORLD_FILES, HASH_FILE):
        shutil.copyfile(source / "files" / name, root / "files" / name)
    for window in ("observed", "later"):
        run = f"w47-{window}"
        _copy_run(source / "runs" / run, root / "runs" / run, drop if window == "later" else None)
    if edit is not None:
        edit(root / "files")
        if reseal:
            seal_world(root / "files")  # the files agree with their hash again
    return root


def test_folded_world_is_compared_with_its_manifest_by_identity_not_by_episode_id(
    tmp_path: Path,
) -> None:
    def another_scene(files: Path) -> None:
        lines = (files / "manifest.jsonl").read_text().splitlines()
        row = json.loads(lines[-1])
        row["identity"]["scene_seed"] += 1
        lines[-1] = json.dumps(row, separators=(",", ":"))
        (files / "manifest.jsonl").write_text("\n".join(lines) + "\n")

    # Same episode ids in manifest and runs; one attempt ran another scene than planned.
    changed = _folded_world(tmp_path / "changed", another_scene)
    with pytest.raises(ValueError, match="folded world w47 differs from its own manifest"):
        calibration.refinement_record([changed])


def test_continuation_whose_parent_cannot_be_read_is_skipped_with_it(tmp_path: Path) -> None:
    root = _job(tmp_path / "parent-chain")
    run = root / "pair-batch-2" / "run" / "run.json"
    record = json.loads(run.read_text())
    record["chain_ok"] = False
    run.write_text(json.dumps(record))
    # The continuation's own run is complete and its chain passes; its rule cannot be
    # rechecked without the parent's outcomes, so it contributes no cells either.
    imported = calibration.refinement_record([], root)
    assert imported["cells"] == []
    assert imported["skipped_stages"] == ["pair-batch-2", "pair-batch-2-refine"]


# ---- 4. the import: a folded world's manifest against its hash ---------------------------


def test_folded_world_whose_manifest_lost_the_attempt_too_is_not_imported(tmp_path: Path) -> None:
    source = (WORLDS / "w47-opaque-v1/files/manifest.jsonl").read_text().splitlines()
    last = json.loads(source[-1])["identity"]["episode_id"]

    def without_last(files: Path) -> None:
        (files / "manifest.jsonl").write_text("\n".join(source[:-1]) + "\n")

    # The attempt is gone from the runs and from the manifest alike: runs and manifest
    # agree, 1,679 attempts each. Only the hash stored beside the manifest still knows.
    short = _folded_world(tmp_path / "short", without_last, reseal=False, drop=last)
    with pytest.raises(ValueError, match="folded world w47: its files differ from their"):
        calibration.refinement_record([short])
    whole = _folded_world(tmp_path / "whole")
    assert _attempts(calibration.refinement_record([whole])) == 1680
    # Without a readable hash file nothing says that the files are the world's own.
    (whole / "files" / "hash").unlink()
    with pytest.raises(ValueError, match="folded world w47: its files differ from their"):
        calibration.refinement_record([whole])


# ---- 2. the recipe file is the code's recipe ---------------------------------------------

RECIPE_HASH = "d64334423b024b4461ffa13e06b26da0d4c424ec479609f76d55f4501699bb86"


def test_recipe_hash_is_pinned_and_leaves_out_the_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fleetlab.worlds import __main__ as cli

    assert recipe_hash() == RECIPE_HASH
    # The same for every place; another kind of world is another recipe.
    for world_id, fleet_id, name in (("w97", "w07", "T1"), ("w51", "w33", "T5")):
        assert stated_hash(private_recipe(world_id, fleet_id, name=name)) == RECIPE_HASH
    assert recipe_hash("reserve") != RECIPE_HASH
    # Any rule changed is another hash.
    stated = private_recipe("w97", "w07", name="T1")
    for field, value in (("attempts", 800), ("robot_success_floor", 0.25), ("note", "x")):
        assert stated_hash({**stated, field: value}) != RECIPE_HASH
    assert cli.main(["recipe-hash"]) == 0
    assert capsys.readouterr().out.strip() == RECIPE_HASH


def test_recipe_file_states_no_null_and_is_created_only_as_the_codes_recipe(
    tmp_path: Path,
) -> None:
    stated = private_recipe("w97", "w07", name="T1")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(stated))
    assert code_recipe(path) == stated
    # `null` where the code states a value: no floor, no pinned digest. Refused on reading.
    stated_values = [field for field, value in stated.items() if value is not None]
    assert {"robot_success_floor", "calibration_sources", "confounder_companion"} <= set(
        stated_values
    )
    for field in stated_values:
        path.write_text(json.dumps({**stated, field: None}))
        with pytest.raises(ValueError):
            external_recipe(path)
        with pytest.raises(ValueError):
            code_recipe(path)
    # Any other valid value is another recipe under the same name: creation refuses it,
    # although the model accepts it.
    for field, value in (
        ("robot_success_floor", 0.25),
        ("retained_error", "none"),
        ("search_revision", "r7"),
        ("admission", "target"),
        ("attempts", 4000),
        ("calibration_jobs", stated["calibration_jobs"][:-1]),
        ("seed", SEEDS[0]),
    ):
        path.write_text(json.dumps({**stated, field: value}))
        with pytest.raises(ValueError, match="differs from the code's recipe"):
            code_recipe(path)
    WorldRecipe.model_validate({**stated, "robot_success_floor": 0.25, "seed": SEEDS[0]})


# ---- 1. the ledger of draws (rounds 9, 9b and 9c) ------------------------------------------

KEY = book.KEY
# A made-up beacon string of the one form a ledger opens with (no network: the round and
# its randomness are invented) and the seeds that follow from it for the test recipe.
# The first two compose within seconds.
RANDOMNESS = "cae23969aac88637047e5e7c803a821dbb53594d7cd0c06987632281cbfa6ffb"
BEACON = f"drand:default:1032:{RANDOMNESS}"
T1_FIRST, T1_FIRST_DRAFT = 8192029344934392712, 17
T1_SECOND, T1_SECOND_DRAFT = 9218722357417287245, 9
T2_FIRST = 5685832324810067171
CLEAN = {"commit": "c" * 40, "clean": True}


def test_the_seed_is_derived_from_the_beacon_and_pinned() -> None:
    import hashlib

    assert book.derived_seed(BEACON, RECIPE_HASH, "T1", 1) == T1_FIRST
    assert book.derived_seed(BEACON, RECIPE_HASH, "T1", 2) == T1_SECOND
    assert book.derived_seed(BEACON, RECIPE_HASH, "T2", 1) == T2_FIRST
    assert book.derived_seed(BEACON, recipe_hash("reserve"), "R1", 1) == 6356369290057135455
    # The README's worked example, with a hash tool: 2^62 + (SHA-256 mod 2^62).
    example = "drand:default:1000:" + "00112233445566778899aabbccddeeff" * 2
    digest = hashlib.sha256(f"{example}|{RECIPE_HASH}|T1|1".encode()).hexdigest()
    assert digest == "c4a12292ae816e3a975e0d8ae0103f657229bc771401d05cd58d2f3a937e2a24"
    assert book.derived_seed(example, RECIPE_HASH, "T1", 1) == 6164635393569204772
    assert 2**62 + int(digest, 16) % 2**62 == 6164635393569204772
    # The format seeds have: 62 derived bits above a set bit 62; a model takes it.
    seeds = {book.derived_seed(BEACON, RECIPE_HASH, f"T{k}", n) for k in (1, 2, 3) for n in (1, 2)}
    assert len(seeds) == 6 and all(2**62 <= seed < 2**63 for seed in seeds)
    WorldRecipe.model_validate({**private_recipe("w97", "w07", name="T1"), "seed": T1_FIRST})
    # Every part moves it.
    assert (
        len(
            {
                T1_FIRST,
                book.derived_seed(BEACON.upper(), RECIPE_HASH, "T1", 1),
                book.derived_seed(BEACON, RECIPE_HASH[::-1], "T1", 1),
            }
        )
        == 3
    )


def test_a_beacon_has_one_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from fleetlab.worlds import __main__ as cli

    def opened(name: str, beacon: str) -> int:
        path = tmp_path / name / "ledger.jsonl"
        return int(cli.main(["ledger", "open", "--ledger", str(path), "--beacon", beacon]))

    # The review's seven writings of one public value: one opens a ledger, six do not.
    assert opened("as-announced", BEACON) == 0
    others = {
        "capital-digits": f"drand:default:1032:{RANDOMNESS.upper()}",
        "a-note-after": f"{BEACON} (fetched from two relays)",
        "capital-chain": f"DRAND:DEFAULT:1032:{RANDOMNESS}",
        "leading-zero": f"drand:default:01032:{RANDOMNESS}",
        "randomness-alone": RANDOMNESS,
        "a-blank-inside": f"drand:default:1032: {RANDOMNESS}",
        "blanks-around": f" {BEACON} ",
        "a-short-digest": f"drand:default:1032:{RANDOMNESS[:-1]}",
        "round-zero": f"drand:default:0:{RANDOMNESS}",
        "another-chain": f"drand:quicknet:1032:{RANDOMNESS}",
    }
    for name, beacon in others.items():
        capsys.readouterr()
        assert opened(name, beacon) == 2, name
        assert "drand:default:ROUND:RANDOMNESS" in capsys.readouterr().err
        assert not (tmp_path / name / "ledger.jsonl").exists()
        with pytest.raises(ValueError, match="drand:default:ROUND:RANDOMNESS"):
            book.open_ledger(tmp_path / name / "ledger.jsonl", beacon)
    # A ledger made throughout under another writing does not read either.
    path = tmp_path / "as-announced" / "ledger.jsonl"
    path.write_text(path.read_text().replace(RANDOMNESS, RANDOMNESS.upper()))
    with pytest.raises(book.BrokenLedger, match="not written drand:default:ROUND:RANDOMNESS"):
        book.Ledger.read(path)


class _Creation:
    """The command line on scratch paths, with an opened ledger and a tree that counts as clean."""

    def __init__(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, opened: bool = True
    ) -> None:
        from fleetlab.worlds import __main__ as cli
        from fleetlab.worlds import recipe as recipes
        from fleetlab.worlds import runner
        from fleetlab.worlds import world as world_module

        self.cli, self.world, self.runner, self.monkeypatch = cli, world_module, runner, monkeypatch
        self.tmp = tmp_path
        self.ledger = tmp_path / "truth" / "ledger.jsonl"
        monkeypatch.setattr(runner, "engine_software", _software)
        monkeypatch.setattr(world_module, "task_texts", lambda tasks: _texts())
        monkeypatch.setattr(book, "code_state", lambda: dict(CLEAN))

        def no_seed_is_drawn() -> int:
            raise AssertionError("a test or reserve world draws no seed")

        monkeypatch.setattr(recipes, "new_world_seed", no_seed_is_drawn)
        if opened:
            book.open_ledger(self.ledger, BEACON)

    def recipe(self, world_id: str = "w97", kind: str = "test", **changes: Any) -> Path:
        path = self.tmp / f"recipe-{world_id}-{kind}-{len(changes)}.json"
        stated = private_recipe(world_id, "w07", name="T1", kind=kind)  # type: ignore[arg-type]
        path.write_text(json.dumps({**stated, **changes}))
        return path

    def root(self, world_id: str = "w97") -> Path:
        return self.tmp / f"root-{world_id}"

    def generate(
        self, world_id: str = "w97", place: str = "T1", kind: str = "test", **paths: Any
    ) -> int:
        args = {
            "--recipe": self.recipe(world_id, kind),
            "--root": self.root(world_id),
            "--ledger": self.ledger,
            "--place": place,
            **paths,
        }
        flat = [str(x) for pair in args.items() if pair[1] is not None for x in pair]
        return int(self.cli.main(["generate", *flat]))

    def command(self, *words: Any) -> int:
        rest = [str(word) for word in words[1:]]
        return int(self.cli.main(["ledger", str(words[0]), "--ledger", str(self.ledger), *rest]))

    def lines(self) -> list[dict[str, Any]]:
        return book.Ledger.read(self.ledger).lines

    def kinds(self) -> list[str]:
        return [str(line[KEY]) for line in self.lines()]

    def fail_with(self, error: BaseException) -> None:
        def generate(*args: Any, **kw: Any) -> Any:
            raise error

        self.monkeypatch.setattr(self.world, "generate", generate)

    def ran(self, world_id: str, successes: int, episodes: int = 1008, **more: Any) -> None:
        """A runner's record written by hand: the world ran, and this is its one number."""
        record = {
            "world_id": world_id,
            "ready": True,
            "steps": {name: {"done": True} for name in self.runner.STEPS},
            "summary": {
                "measured": {"observed": {"logged": {"ok": successes, "n": episodes}}},
                "causes": ["f07 action_latency s1: the truth beside the number"],
            },
            **more,
        }
        record["world_id"] = record.pop("named", world_id)
        exported = record.pop("exported", successes)
        (self.root(world_id) / "runner.json").write_text(json.dumps(record))
        _export(self.root(world_id), exported, episodes)

    def ready(self, answer: str | None = None) -> list[Any]:
        """Stand in for the runner's own readiness test, and note what it was asked."""
        asked: list[Any] = []

        def readiness_problem(record: dict[str, Any], paths: Any) -> str | None:
            asked.append((record.get("world_id"), paths.root))
            return answer

        self.monkeypatch.setattr(self.runner, "readiness_problem", readiness_problem)
        return asked


def _export(root: Path, successes: int, episodes: int) -> None:
    """The export's public table of the observed run, written by hand: outcomes only."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    outcomes = ["success"] * successes + ["failure"] * (episodes - successes)
    path = root / "export" / "log" / "episodes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"outcome": outcomes}), path)


def _by_hand(path: Path, draws: list[tuple[str, str, str]], kind: str = "test") -> book.Ledger:
    """A ledger written by hand through `append`: (place, world id, state) per draw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    ledger = book.open_ledger(path, BEACON)
    numbers: dict[str, int] = {}
    for place, world_id, state in draws:
        n = numbers[place] = numbers.get(place, 0) + 1
        root = f"/gone/{world_id}-{place}-{n}"
        draw = ledger.append(
            {
                KEY: "drawn",
                "place": place,
                "world_id": world_id,
                "n": n,
                "seed": book.derived_seed(BEACON, RECIPE_HASH, place, n),
                "public_fleet_id": "w07",
                "name": "T1",
                "kind": kind,
                "recipe_file_sha256": "f" * 64,
                "recipe_hash": RECIPE_HASH,
                "band": [0.6, 0.75],
                "code": dict(CLEAN),
                "record": list(TEST_WORLD_SOURCES),
                "software": _software().model_dump(mode="json"),
                "root": root,
            }
        )
        base = {"place": place, "world_id": world_id, "draw": draw}
        if state == "open":
            continue
        if state == "failed":
            ledger.append({KEY: "failed", **base, "error": "KeyboardInterrupt"})
        elif state == "no_composition":
            ledger.append({KEY: "no_composition", **base, "drafts": 400, "refusals": {}})
        else:
            ledger.append(
                {KEY: "composed", **base, "draft": 1, "world_hash": "h", "forecast_seal": "s"}
            )
            if state in ("kept", "discarded"):
                ok = 700 if state == "kept" else 800
                number = {"successes": ok, "episodes": 1000, "rate": ok / 1000}
                ledger.append({KEY: state, **base, **number, "band": [0.6, 0.75], "root": root})
    return ledger


def _rechained(path: Path, edit: Any) -> None:
    """Edit a ledger's lines and give every line the hash of the line before it again."""
    import hashlib

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    rows = [{**json.loads(text), "_was": digest(text)} for text in path.read_text().splitlines()]
    texts: list[str] = []
    renamed: dict[str, str] = {}
    for row in edit(rows) or rows:
        row = dict(row)
        was = row.pop("_was")
        row["prev"] = digest(texts[-1]) if texts else book.FIRST
        if "draw" in row:
            row["draw"] = renamed.get(row["draw"], row["draw"])
        texts.append(json.dumps(row, separators=(",", ":")))
        renamed.setdefault(was, digest(texts[-1]))
    path.write_text("\n".join(texts) + "\n")


def test_the_seed_is_in_the_ledger_before_it_is_composed_and_each_outcome_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    made = _Creation(tmp_path, monkeypatch, opened=False)
    # A ledger is opened once, with the beacon; without it nothing is drawn.
    assert made.generate() == 2
    assert "(ledger)" in capsys.readouterr().err and not made.ledger.exists()
    assert made.command("open", "--beacon", "made-up beacon, round 9c, number 01") == 2
    assert made.command("open") == 2 and not made.ledger.exists()
    assert made.command("open", "--beacon", BEACON) == 0
    out = capsys.readouterr().out
    (first,) = made.lines()
    assert (first[KEY], first["beacon"], first["prev"]) == ("beacon", BEACON, book.FIRST)
    assert json.loads(out.splitlines()[-1])["lines"] == 1
    assert made.command("open", "--beacon", BEACON) == 2 and len(made.lines()) == 1
    capsys.readouterr()

    # A seed that does not compose: exit 3 and its own line; two more lines, no world.
    made.fail_with(compose.NoComposition("none", {"only N of N plain faults fit": 400}))
    assert made.generate() == 3
    err = capsys.readouterr().err
    assert "No composition within the recipe's drafts" in err and "Draw again" in err
    _, drawn, outcome = made.lines()
    assert (drawn[KEY], drawn["n"], drawn["seed"]) == ("drawn", 1, T1_FIRST)
    assert not {"draft", "world_hash", "forecast_seal", "refusals"} & set(drawn)
    assert (outcome[KEY], outcome["drafts"]) == ("no_composition", 400)
    assert outcome["refusals"] == {"only N of N plain faults fit": 400} and "seed" not in outcome
    second = made.ledger.read_text().splitlines()[1]
    assert outcome["draw"] == outcome["prev"] == hashlib.sha256(second.encode()).hexdigest()
    assert not (made.root() / "files").exists()

    # The place's second draw: when composing starts its line is on the disk already.
    monkeypatch.undo()
    made = _Creation(tmp_path, monkeypatch, opened=False)
    real, seen = made.world.generate, []

    def generate(recipe: Any, *args: Any, **kw: Any) -> Any:
        texts = made.ledger.read_text().splitlines()
        last = json.loads(texts[-1])
        seen.append((last[KEY], last.get("seed") == recipe.seed, len(texts)))
        return real(recipe, *args, **kw)

    monkeypatch.setattr(made.world, "generate", generate)
    assert made.generate() == 0
    shown_now = capsys.readouterr()
    assert seen == [("drawn", True, 4)]  # the seed's line was there when composing began
    digest = (made.root() / "files/hash").read_text().strip()
    printed = shown_now.out[shown_now.out.index("{") :]
    assert json.loads(printed) == {"hash": digest, "private": True}
    drawn, outcome = made.lines()[3:]
    assert (drawn["place"], drawn["world_id"], drawn["n"], drawn["seed"]) == (
        "T1",
        "w97",
        2,
        T1_SECOND,
    )
    assert drawn["recipe_hash"] == RECIPE_HASH and drawn["record"] == list(TEST_WORLD_SOURCES)
    recipe_file = made.recipe()
    assert drawn["recipe_file_sha256"] == hashlib.sha256(recipe_file.read_bytes()).hexdigest()
    assert drawn["code"] == CLEAN and drawn["time"].endswith("Z")
    assert drawn["software"] == _software().model_dump(mode="json")
    assert drawn["root"] == str(made.root())
    assert (outcome[KEY], outcome["draft"], outcome["world_hash"]) == (
        "composed",
        T1_SECOND_DRAFT,
        digest,
    )
    assert str(T1_SECOND) not in shown_now.out + shown_now.err

    # The same command on the root that holds the candidate: no line, no byte changed,
    # and nothing printed that shows the seed.
    before = {p.name: p.read_bytes() for p in (made.root() / "files").iterdir()}
    assert made.generate() == 0
    again = capsys.readouterr()
    assert len(made.lines()) == 5
    assert {p.name: p.read_bytes() for p in (made.root() / "files").iterdir()} == before
    assert json.loads(again.out[again.out.index("{") :]) == {"hash": digest, "private": True}
    for secret in (str(T1_SECOND), str(T1_FIRST), outcome["forecast_seal"], '"seed"'):
        assert secret not in again.out + again.err
    # A second world for the place is refused.
    assert made.generate("w98") == 2
    err = capsys.readouterr().err
    assert "before any seed was drawn (place)" in err and "standing candidate" in err
    assert len(made.lines()) == 5 and not made.root("w98").exists()

    # The summary: counts and candidates per place; the head: lines and hashes.
    assert made.command("summary") == 0
    text = capsys.readouterr().out
    shown = json.loads(text)
    assert shown["places"] == {
        "T1": {
            "draws": 2,
            "open": 0,
            "no_composition": 1,
            "failed_attempts": 0,
            "candidates": [{"world_id": "w97", "state": "standing"}],
        }
    }
    texts = made.ledger.read_text().splitlines()
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    assert shown["beacon"] == BEACON  # public, and the one thing a reader compares
    assert (shown["lines"], shown["head"], shown["chain"], bool(shown["beacon"])) == (
        5,
        hashes[-1],
        "intact",
        True,
    )
    for secret in (str(T1_FIRST), str(T1_SECOND), digest, outcome["forecast_seal"], '"seed"'):
        assert secret not in text
    assert made.command("head") == 0
    assert json.loads(capsys.readouterr().out) == {"lines": 5, "head": hashes[-1]}
    # The hash at an earlier line, to compare an earlier posting with the ledger as it stands.
    assert made.command("head", "--line", 3) == 0
    assert json.loads(capsys.readouterr().out) == {
        "lines": 5,
        "head": hashes[-1],
        "line": 3,
        "hash": hashes[2],
    }
    assert made.command("head", "--line", 6) == 2 and made.command("head", "--line", 0) == 2


def test_every_line_is_synced_to_the_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    synced: list[int] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    ledger = book.open_ledger(tmp_path / "l.jsonl", BEACON)
    assert len(synced) == 2  # the file and its directory
    _by_hand(tmp_path / "l.jsonl", [("T1", "w97", "no_composition")])
    assert len(synced) == 2 + 3 * 2 and ledger.head()[0] == 1


CHILD = """
import json, os, secrets, signal, sys, time
from pathlib import Path
import fleetlab  # noqa: F401
from fleetlab.contracts.identity import SoftwareVersion, Task
from fleetlab.worlds import __main__ as cli, ledger as book, runner, world

mode, tests, recipe, root, ledger = sys.argv[1:]
seals = json.loads(Path(tests, "worlds_generation4_seals.json").read_text())
rows = json.loads(Path(tests, "worlds_round3_seals.json").read_text())["task_texts"]
texts = {Task(r["suite"], r["task_index"]): r["text"] for r in rows}
runner.engine_software = lambda: SoftwareVersion.model_validate(seals["software"])
world.task_texts = lambda tasks: texts
book.code_state = lambda: {"commit": "c" * 40, "clean": True}

def die(*args, **kw):
    os.kill(os.getpid(), signal.SIGKILL)

def interrupt(*args, **kw):
    # A terminal delivers Ctrl-C; a test started in the background may have inherited
    # an ignored SIGINT, so the default is put back before the signal is sent.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(30)

if mode == "kill-before-composing":
    world.generate = die  # the seed is recorded; nothing is composed
elif mode == "interrupt-while-composing":
    world.generate = interrupt
else:
    real = book.Ledger.append

    def append(self, row):
        if row[book.KEY] not in ("beacon", "drawn"):
            die()  # the world's files are written; the outcome line never is
        return real(self, row)

    book.Ledger.append = append
args = ["generate", "--recipe", recipe, "--root", root, "--ledger", ledger, "--place", "T1"]
raise SystemExit(cli.main(args))
"""


def _child(made: _Creation, mode: str) -> int:
    import subprocess
    import sys

    args = [mode, Path(__file__).parent, made.recipe(), made.root(), made.ledger]
    done = subprocess.run(
        [sys.executable, "-c", CHILD, *map(str, args)], capture_output=True, text=True, check=False
    )
    return done.returncode


def test_a_draw_killed_between_its_two_lines_is_repeated_with_its_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    # Killed for real after the seed's line, before anything was composed.
    assert _child(made, "kill-before-composing") == -9
    assert made.kinds() == ["beacon", "drawn"] and made.lines()[1]["seed"] == T1_FIRST
    assert not made.root().exists()
    assert made.command("summary") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["open_draws"] == 1 and shown["places"]["T1"]["open"] == 1
    assert shown["places"]["T1"]["candidates"] == [{"world_id": "w97", "state": "open"}]
    # While it is open no draw is taken under another recipe or root ...
    assert made.generate("w98", "T1") == 2
    assert "before any seed was drawn (ledger)" in capsys.readouterr().err
    elsewhere: dict[str, Any] = {"--root": tmp_path / "elsewhere"}
    assert made.generate(**elsewhere) == 2 and "(ledger)" in capsys.readouterr().err
    # ... nor from another commit, with other software, or from a tree that is not clean.
    with monkeypatch.context() as patch:
        patch.setattr(book, "code_state", lambda: {"commit": "d" * 40, "clean": True})
        assert made.generate() == 2 and "(ledger)" in capsys.readouterr().err
    with monkeypatch.context() as patch:
        other = _software().model_copy(update={"lock_hash": "0" * 16})
        patch.setattr(made.runner, "engine_software", lambda: other)
        assert made.generate() == 2 and "(ledger)" in capsys.readouterr().err
    with monkeypatch.context() as patch:
        patch.setattr(book, "code_state", lambda: {**CLEAN, "clean": False})
        assert made.generate() == 2 and "(code)" in capsys.readouterr().err
    assert made.kinds() == ["beacon", "drawn"] and not (tmp_path / "elsewhere").exists()
    # The same command composes the recorded seed and writes its outcome.
    assert made.generate() == 0
    capsys.readouterr()
    outcome = made.lines()[2]
    assert made.kinds() == ["beacon", "drawn", "composed"]
    assert (outcome[KEY], outcome["draft"]) == ("composed", T1_FIRST_DRAFT)
    digest = (made.root() / "files/hash").read_text().strip()
    assert outcome["world_hash"] == digest

    # Killed for real after the world's files were written, before the outcome's line.
    again = _Creation(tmp_path / "second", monkeypatch, opened=False)
    again.tmp.mkdir()
    book.open_ledger(again.ledger, BEACON)
    assert _child(again, "kill-after-the-files") == -9
    assert again.kinds() == ["beacon", "drawn"]
    files = {p.name: p.read_bytes() for p in (again.root() / "files").iterdir()}
    # Files that differ from what the seed composes are an error and are not overwritten;
    # the error closes nothing, and the same command repeats the same seed.
    description = again.root() / "files/description.json"
    description.write_bytes(files["description.json"] + b" ")
    assert again.generate() == 2
    err = capsys.readouterr().err
    assert "failed after the draw (FileExistsError)" in err and "same command again" in err
    assert description.read_bytes() == files["description.json"] + b" "
    assert again.kinds() == ["beacon", "drawn", "failed"]
    description.write_bytes(files["description.json"])
    assert again.generate() == 0
    assert again.kinds() == ["beacon", "drawn", "failed", "composed"]
    assert again.lines()[3]["draw"] == again.lines()[2]["draw"]
    assert {p.name: p.read_bytes() for p in (again.root() / "files").iterdir()} == files


def test_an_interrupt_or_an_error_closes_no_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    # Ctrl-C while composing, for real: the attempt is recorded as failed, the draw is open.
    assert _child(made, "interrupt-while-composing") != 0
    assert made.kinds() == ["beacon", "drawn", "failed"]
    assert made.lines()[2]["error"] == "KeyboardInterrupt" and made.lines()[1]["seed"] == T1_FIRST
    # A foreign file under the root makes the command fail: again nothing is closed.
    (made.root() / "files").mkdir(parents=True)
    (made.root() / "files/hash").write_text("a foreign file\n")
    assert made.generate() == 2
    err = capsys.readouterr().err
    assert "failed after the draw (FileExistsError)" in err and "same command again" in err
    assert made.kinds() == ["beacon", "drawn", "failed", "failed"]
    # Another error, and still no second seed for the place, nor a draw anywhere else.
    made.fail_with(KeyError("f07 action_latency r05 seed and level"))
    assert made.generate() == 2
    err = capsys.readouterr().err
    assert "failed after the draw (KeyError)" in err and "action_latency" not in err
    assert made.generate("w98", "T2") == 2 and "(ledger)" in capsys.readouterr().err
    assert made.kinds() == ["beacon", "drawn", "failed", "failed", "failed"]
    assert made.command("summary") == 0
    here = json.loads(capsys.readouterr().out)["places"]["T1"]
    assert (here["draws"], here["open"], here["failed_attempts"]) == (1, 1, 3)
    # verify composes the open draw too and says that it composes; an open draw fails it.
    monkeypatch.undo()
    made = _Creation(tmp_path, monkeypatch, opened=False)
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    counts = json.loads(text[text.index("{") :])
    assert (counts["draws"], counts["open"], counts["open_compose"]) == (1, 1, 1)
    # The root cleared: the same command repeats the recorded seed, which composes.
    (made.root() / "files/hash").unlink()
    assert made.generate() == 0
    assert made.kinds() == ["beacon", "drawn", "failed", "failed", "failed", "composed"]
    assert made.lines()[5]["draft"] == T1_FIRST_DRAFT
    capsys.readouterr()
    assert made.command("verify") == 0
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{") :])["repeated"] == 1


def test_everything_that_needs_no_seed_is_checked_before_one_is_drawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    made.fail_with(AssertionError("no world may be composed"))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /unneeded")
    ten = [("T1", "w97", "no_composition")] * 10
    three = [("T1", f"w9{k}", "discarded") for k in range(3)]
    kept = [("T1", "w90", "kept")]
    cases: list[tuple[str, dict[str, Any], list[tuple[str, str, str]] | None, str]] = [
        ("arguments", {"--ledger": None}, None, "needs --ledger and --place"),
        ("arguments", {"--place": None}, None, "needs --ledger and --place"),
        ("path", {"--root": checkout / "root"}, None, "outside every checkout"),
        ("path", {"--ledger": checkout / "ledger.jsonl"}, None, "outside every checkout"),
        ("path", {"--ledger": tmp_path / "root-w97" / "ledger.jsonl"}, None, "apart"),
        ("recipe", {"--recipe": made.recipe(attempts=4000)}, None, "not the running code's"),
        ("recipe", {"--recipe": made.recipe(robot_success_floor=None)}, None, "not the running"),
        ("recipe", {"--recipe": tmp_path / "absent.json"}, None, "not the running code's"),
        ("place", {"--place": "T1b"}, [], "a test world's place is T1 to T5"),
        ("place", {"--place": "R1"}, [], "a test world's place is T1 to T5"),
        ("place", {"--place": "T2"}, [], "the place before it is not filled"),
        ("place", {"--place": "T3"}, kept, "the place before it is not filled"),
        ("place", {}, [("T1", "w96", "standing")], "standing candidate"),
        ("place", {}, kept, "the place is filled"),
        ("place", {}, [("T1", "w97", "discarded")], "was a candidate already"),
        ("place", {"--place": "T2"}, [("T1", "w97", "kept")], "under another place"),
        ("place", {}, ten, "10 draws in a row without a composition"),
        (
            "place",
            {"--place": "T2"},
            [("T1", "w90", "kept"), *[("T2", "w97", "no_composition")] * 10],
            "10 draws in a row without a composition",
        ),
        ("place", {}, three, "entry 16's limit"),
        ("ledger", {}, [("T1", "w96", "open")], "an open draw was made under another"),
        ("ledger", {"--place": "T2"}, [("T1", "w96", "failed")], "a draw is open"),
    ]
    for kind, paths, draws, words in cases:
        if draws is not None:
            _by_hand(made.ledger, draws)
        before = made.ledger.read_bytes()
        capsys.readouterr()
        assert made.generate(**paths) == 2, words
        err = capsys.readouterr().err
        assert f"before any seed was drawn ({kind})" in err and words in err, err
        assert not (tmp_path / "root-w97").exists()
        assert not (checkout / "root").exists() and not (checkout / "ledger.jsonl").exists()
        assert made.ledger.read_bytes() == before
    book.open_ledger(tmp_path / "fresh" / "ledger.jsonl", BEACON)
    made.ledger = tmp_path / "fresh" / "ledger.jsonl"
    before = made.ledger.read_bytes()
    # The record's digest is not the one the recipe pins: refused before the draw as well.
    with monkeypatch.context() as patch:
        patch.setattr(curves, "measured_sources", lambda jobs=None: ("another record",))
        assert made.generate() == 2
        assert "before any seed was drawn (calibration)" in capsys.readouterr().err
    # Code whose commit cannot be read, and a tree that is not clean.
    with monkeypatch.context() as patch:
        patch.setattr(book, "code_state", lambda: (_ for _ in ()).throw(ValueError("no git")))
        assert made.generate() == 2
        assert "before any seed was drawn (code)" in capsys.readouterr().err
    with monkeypatch.context() as patch:
        patch.setattr(book, "code_state", lambda: {**CLEAN, "clean": False})
        assert made.generate() == 2
        err = capsys.readouterr().err
        assert "before any seed was drawn (code)" in err and "not clean" in err
    assert made.ledger.read_bytes() == before and not made.root().exists()


def test_the_codes_state_is_read_from_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from fleetlab import runtime

    def git(*args: str) -> str:
        identity = ["-c", "user.name=scratch", "-c", "user.email=scratch@example.invalid"]
        done = subprocess.run(
            ["git", *identity, *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    monkeypatch.setattr(runtime, "repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="commit cannot be read"):
        book.code_state()
    git("init", "-q", ".")
    (tmp_path / "file").write_text("one\n")
    git("add", "file")
    git("commit", "-q", "-m", "one")
    assert book.code_state() == {"commit": git("rev-parse", "HEAD"), "clean": True}
    (tmp_path / "file").write_text("two\n")
    assert book.code_state() == {"commit": git("rev-parse", "HEAD"), "clean": False}
    git("commit", "-q", "-am", "two")
    (tmp_path / "untracked").write_text("x")
    assert book.code_state()["clean"] is False


def test_one_draw_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    made.fail_with(AssertionError("no world may be composed"))
    _by_hand(made.ledger, [("T1", "w96", "standing")])
    before = made.ledger.read_bytes()
    with book.locked(made.ledger):
        # A second command that writes fails at once, and says why.
        assert made.generate("w98", "T2") == 2
        err = capsys.readouterr().err
        assert "before any seed was drawn (ledger)" in err and "in use" in err
        assert made.command("check", "--place", "T1") == 2
        err = capsys.readouterr().err
        assert "(ledger)" in err and "in use" in err
        with pytest.raises(book.LedgerBusy), book.locked(made.ledger):
            pass
        # Reading is not held up.
        assert made.command("summary") == 0 and made.command("head") == 0
    assert made.ledger.read_bytes() == before
    with book.locked(made.ledger):
        pass


def test_a_changed_removed_or_cut_line_breaks_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    made = _Creation(tmp_path, monkeypatch)
    made.fail_with(AssertionError("no world may be composed"))
    ledger = _by_hand(made.ledger, [("T1", "w96", "no_composition"), ("T1", "w96", "standing")])
    whole = made.ledger.read_text().splitlines()
    assert [line["prev"] for line in ledger.lines] == [
        book.FIRST,
        *(hashlib.sha256(text.encode()).hexdigest() for text in whole[:-1]),
    ]
    assert ledger.head() == (5, hashlib.sha256(whole[-1].encode()).hexdigest())
    changed = whole[2].replace('"drafts":400', '"drafts":40')
    broken = [
        ("line 3 draws although a draw is open", [*whole[:2], *whole[3:]]),  # a line removed
        ("line 4 does not name the line before it", [*whole[:2], changed, *whole[3:]]),  # changed
        ("line 1 is a beacon line that is not the first, or a first", whole[1:]),  # front cut
        ("line 1 is a beacon line that is not the first, or a first", whole[3:]),
        ("line 3 is not a line of this ledger", [*whole[:2], "{}", *whole[2:]]),
        ("the last line is cut off", whole),
    ]
    for problem, texts in broken:
        text = "\n".join(texts) + "\n"
        if "cut off" in problem:
            text = text[:-9]
        made.ledger.write_text(text)
        capsys.readouterr()
        assert made.generate("w98", "T2") == 2
        assert "before any seed was drawn (ledger)" in capsys.readouterr().err
        for action in (("check", "--place", "T1"), ("verify",), ("head",)):
            assert made.command(*action) == 2, (problem, action)
            err = capsys.readouterr().err
            assert "Refused (ledger)" in err and problem in err, (problem, action, err)
        # The summary still answers, says what is wrong, exits 1 and shows no seed.
        assert made.command("summary") == 1
        shown = capsys.readouterr().out
        assert problem in json.loads(shown)["chain"] and '"seed"' not in shown
        assert made.ledger.read_text() == text
    made.ledger.write_text("\n".join(whole) + "\n")
    assert made.command("summary") == 0 and made.command("head") == 0


def test_a_ledger_edited_and_chained_again_is_refused_line_by_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    made.fail_with(compose.NoComposition("none", {}))
    draws = [("T1", "w95", "no_composition"), ("T1", "w96", "discarded"), ("T1", "w97", "kept")]

    def edit(change: Any) -> Any:
        def apply(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return change(rows) or rows

        return apply

    def to_kept(rows: list[dict[str, Any]]) -> None:
        rows[5][KEY] = "kept"  # 80.0% turned into kept

    def other_root(rows: list[dict[str, Any]]) -> None:
        rows[5]["root"] = "/another/directory"

    def other_seed(rows: list[dict[str, Any]]) -> None:
        rows[3]["seed"] = OTHER[13]  # a seed handed in

    def other_number(rows: list[dict[str, Any]]) -> None:
        rows[3]["n"] = 3

    def other_place(rows: list[dict[str, Any]]) -> None:
        rows[4]["place"] = "T2"  # an outcome moved to another place

    def moved_draw(rows: list[dict[str, Any]]) -> None:
        for row in rows[3:6]:
            row["place"] = "T2"  # a whole draw moved: its seed is another place's

    def two_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [*rows[:3], {**rows[2], "drafts": 399}, *rows[3:]]

    def two_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [*rows, dict(rows[-1])]

    def no_beacon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[1:]

    def other_rate(rows: list[dict[str, Any]]) -> None:
        rows[5]["rate"] = 0.7

    def a_field_gone(rows: list[dict[str, Any]]) -> None:
        del rows[5]["episodes"]  # the round-9b summary ended in a KeyError here

    def no_draft(rows: list[dict[str, Any]]) -> None:
        del rows[4]["draft"]

    def unclean(rows: list[dict[str, Any]]) -> None:
        rows[3]["code"] = {**rows[3]["code"], "clean": False}

    def other_commit(rows: list[dict[str, Any]]) -> None:
        rows[3]["code"] = {**rows[3]["code"], "commit": "d" * 40}

    def out_of_order(rows: list[dict[str, Any]]) -> None:
        for row in rows[6:9]:
            row["place"] = "T2"  # T2 drawn while T1's candidate was discarded, not kept
        rows[6].update(n=1, seed=book.derived_seed(BEACON, RECIPE_HASH, "T2", 1))

    refused = {
        unclean: "was drawn from a tree that is not clean",
        other_commit: "names another commit than the ledger's other draws",
        out_of_order: "draws for a place while the place before it is not kept",
        to_kept: "a decision or a rate that its own number does not give",
        other_rate: "a decision or a rate that its own number does not give",
        other_root: "a count, a root or a band that is not its draw's",
        other_seed: "a seed that is not the derived one",
        other_number: "a number that is not the place's next draw",
        other_place: "a place or a world id that is not its draw's",
        moved_draw: "a number that is not the place's next draw",
        two_outcomes: "follows an outcome of its draw",
        two_checks: "checks a draw that has no candidate, or a second time",
        no_beacon: "a first line without the beacon",
        a_field_gone: "lacks a field of a `discarded` line",
        no_draft: "lacks a field of a `composed` line",
    }
    for change, problem in refused.items():
        _by_hand(made.ledger, draws)
        _rechained(made.ledger, edit(change))
        capsys.readouterr()
        assert made.command("verify") == 2, problem
        err = capsys.readouterr().err
        assert "Refused (ledger)" in err and problem in err, (problem, err)
        assert made.generate("w98", "T2") == 2 and "(ledger)" in capsys.readouterr().err
        # The summary answers without a traceback and without a seed.
        assert made.command("summary") == 1
        shown = capsys.readouterr().out
        assert problem in json.loads(shown)["chain"] and '"seed"' not in shown
    # What `read` cannot know, `verify` asks the recipe: a band widened in a `drawn` line
    # (and in its check) no longer is the recipe's; a checked root that is gone counts.
    _by_hand(made.ledger, draws)
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    counts = json.loads(text[text.index("{") :])
    assert (counts["repeated"], counts["checks"], counts["checks_differ"]) == (1, 2, 2)

    def wider(rows: list[dict[str, Any]]) -> None:
        rows[3]["band"] = rows[5]["band"] = [0.6, 0.81]
        rows[5][KEY] = "kept"

    _rechained(made.ledger, edit(wider))
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{") :])["recipe_differs"] == 1


def test_three_ways_to_choose_a_seed_end_in_the_same_seeds_or_a_refused_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fleetlab.worlds import recipe as recipes

    made = _Creation(tmp_path, monkeypatch)
    made.fail_with(compose.NoComposition("none", {}))
    # (a) The tail cut away: three draws, two of them removed again. The place's next
    # draws have the same numbers and so the same seeds.
    for _ in range(3):
        assert made.generate() == 3
    whole = made.ledger.read_text().splitlines()
    seeds = [line["seed"] for line in made.lines() if line[KEY] == "drawn"]
    assert seeds[:2] == [T1_FIRST, T1_SECOND] and len(set(seeds)) == 3
    made.ledger.write_text("\n".join(whole[:3]) + "\n")
    assert made.generate() == 3 and made.generate() == 3
    assert [line["seed"] for line in made.lines() if line[KEY] == "drawn"] == seeds
    # (b) A second ledger: with the same beacon it holds the same seeds.
    second = _Creation(tmp_path / "second", monkeypatch, opened=False)
    second.tmp.mkdir()
    book.open_ledger(second.ledger, BEACON)
    second.fail_with(compose.NoComposition("none", {}))
    assert second.generate() == 3 and second.lines()[1]["seed"] == T1_FIRST
    # ... and with another beacon it is another ledger, which its first line says.
    third = tmp_path / "third" / "ledger.jsonl"
    book.open_ledger(third, f"drand:default:1033:{RANDOMNESS}")
    second.ledger = third
    assert second.generate() == 3
    assert second.lines()[0]["beacon"] != BEACON and second.lines()[1]["seed"] != T1_FIRST
    # (c) A seed handed in. The command takes none from the random source ...
    monkeypatch.setattr(recipes, "new_world_seed", lambda: OTHER[13])
    monkeypatch.setattr("secrets.randbits", lambda bits: OTHER[13] - (1 << 62))
    made.ledger = tmp_path / "fourth" / "ledger.jsonl"
    book.open_ledger(made.ledger, BEACON)
    assert made.generate() == 3 and made.lines()[1]["seed"] == T1_FIRST
    # ... and a wrapper that replaces the derivation leaves a ledger nobody accepts.
    real = book.derived_seed
    with monkeypatch.context() as patch:
        patch.setattr(
            book, "derived_seed", lambda *parts: OTHER[13] if parts[3] == 2 else real(*parts)
        )
        assert made.generate() == 3
    assert json.loads(made.ledger.read_text().splitlines()[3])["seed"] == OTHER[13]
    capsys.readouterr()
    assert made.command("verify") == 2
    assert "a seed that is not the derived one" in capsys.readouterr().err
    assert made.generate() == 2 and made.command("summary") == 1


def test_places_in_order_and_the_stop_for_the_whole_draw(tmp_path: Path) -> None:
    def refusal(
        draws: list[tuple[str, str, str]], place: str, world_id: str, kind: str = "test"
    ) -> str:
        ledger = _by_hand(tmp_path / "l.jsonl", draws, kind)
        return str(book.refusal(ledger.draws(), kind, place, world_id))

    # Test worlds: T1 to T5 and nothing else, each after the one before it was kept.
    assert refusal([], "T1", "w97") == "None"
    for place in ("T0", "T6", "T1b", "t1", "R1", ""):
        assert "T1 to T5" in refusal([], place, "w97")
    assert "not filled" in refusal([], "T2", "w97")
    assert "not filled" in refusal([("T1", "w97", "standing")], "T2", "w98")
    assert "not filled" in refusal([("T1", "w97", "discarded")], "T2", "w98")
    kept = [("T1", "w97", "kept")]
    assert refusal(kept, "T2", "w98") == "None" and "not filled" in refusal(kept, "T3", "w98")
    assert "filled" in refusal(kept, "T1", "w98")
    # Reserve worlds: R1, R2 and so on, in order like the test places; no upper end.
    assert refusal([], "R1", "w97", "reserve") == "None"
    assert "not filled" in refusal([], "R7", "w97", "reserve")  # the number chooses no seed
    assert "not filled" in refusal([], "R2", "w97", "reserve")
    first = [("R1", "w97", "kept")]
    assert refusal(first, "R2", "w98", "reserve") == "None"
    assert "not filled" in refusal(first, "R3", "w98", "reserve")
    assert "R1, R2 and so on" in refusal([], "T1", "w97", "reserve")
    assert "R1, R2 and so on" in refusal([], "R0", "w97", "reserve")
    # Ten draws in a row without a composition for one place stop every place: at a first
    # place, and at a later one while the place before it is kept.
    nine = [("R1", "w97", "no_composition")] * 9
    assert refusal(nine, "R1", "w97", "reserve") == "None"
    ten = [*nine, ("R1", "w97", "no_composition")]
    assert "10 draws" in refusal(ten, "R1", "w97", "reserve")
    assert "10 draws" in refusal(ten, "R2", "w98", "reserve")
    later = [("R1", "w90", "kept"), *[("R2", "w97", "no_composition")] * 10]
    assert refusal(later[:10], "R2", "w97", "reserve") == "None"
    assert "10 draws" in refusal(later, "R2", "w97", "reserve")
    assert "10 draws" in refusal(later, "R3", "w98", "reserve")
    # A composition ends the run; a discarded candidate frees the place under another id.
    run = [*nine, ("R1", "w97", "discarded")]
    assert refusal(run, "R1", "w98", "reserve") == "None"
    assert "candidate already" in refusal(run, "R1", "w97", "reserve")
    # Entry 16's stop: three discarded for one place, or six in all, and no place draws.
    three = [("R1", f"w8{k}", "discarded") for k in range(3)]
    assert refusal(three[:2], "R1", "w99", "reserve") == "None"
    assert "entry 16" in refusal(three, "R2", "w99", "reserve")
    assert "entry 16" in refusal(three, "R1", "w99", "reserve")
    six = [
        (f"R{k + 1}", f"w{80 + 3 * k + j}", state)
        for k in range(3)
        for j, state in enumerate(("discarded", "discarded", "kept"))
    ][:8]
    assert refusal(six[:7], "R3", "w99", "reserve") == "None"
    assert "entry 16" in refusal(six, "R3", "w99", "reserve")
    assert "entry 16" in refusal(six, "R4", "w99", "reserve")
    stopped = book.summary(_by_hand(tmp_path / "l.jsonl", six, "reserve"))
    assert stopped["discarded_in_all"] == 6 and "entry 16" in stopped["stopped"]
    # An open draw holds up every place, also one that only carries a failed attempt.
    assert "a draw is open" in refusal([("T1", "w97", "open")], "T1", "w98")
    assert "a draw is open" in refusal([("R1", "w97", "failed")], "R2", "w98", "reserve")


def test_entry_16s_check_reads_the_number_and_records_the_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    made = _Creation(tmp_path, monkeypatch)
    assert made.generate() == 0
    digest = made.lines()[2]["world_hash"]
    capsys.readouterr()

    def refused(kind: str, words: str, *action: Any) -> None:
        assert made.command("check", *(action or ("--place", "T1"))) == 2
        err = capsys.readouterr().err
        assert f"Refused ({kind})" in err and words in err, err
        assert made.kinds() == ["beacon", "drawn", "composed"]

    # No record; a record that says ready over a root that never ran: the runner's own
    # test is asked, and it says no.
    refused("world", "not this world's")
    made.ran("w97", 700)
    refused("world", "not ready by the runner's own test")
    asked = made.ready("the export has no receipt")
    refused("world", "not ready by the runner's own test")
    assert asked == [("w97", made.root())]
    made.ready(None)
    made.ran("w97", 700, ready=False)
    refused("world", "not ready by the runner's own test")
    made.ran("w97", 700, named="w98")
    refused("world", "not this world's")  # a record that names another world
    made.ran("w97", 800)
    # A copy of the candidate beside it, with a record of its own, is not the draw's root.
    copy = tmp_path / "copy"
    shutil.copytree(made.root(), copy)
    (copy / "runner.json").write_text(
        (made.root() / "runner.json").read_text().replace("800", "650")
    )
    refused("world", "not the one the draw was made into", "--place", "T1", "--root", copy)
    # The record says 650 where the export's table counts 800: the check counts for itself.
    made.ran("w97", 650, exported=800)
    refused("world", "not what the export's table counts")
    (made.root() / "export" / "log" / "episodes.parquet").unlink()
    refused("world", "not what the export's table counts")
    made.ran("w97", 800)
    refused("place", "no standing candidate", "--place", "T2")
    with pytest.raises(SystemExit):  # no number is typed
        made.command("discard", "--place", "T1", "--world-id", "w97", "--observed", "0.78")
    capsys.readouterr()

    # Outside the band: discarded, exit 4, one JSON line and nothing else of the world.
    assert made.command("check", "--place", "T1", "--root", made.root()) == 4
    out = capsys.readouterr().out
    shown = {"place": "T1", "world_id": "w97", "successes": 800, "episodes": 1008}
    shown.update(rate=round(800 / 1008, 6), band=[0.6, 0.75], decision="discarded")
    assert json.loads(out) == shown and len(out.splitlines()) == 1
    assert "action_latency" not in out and str(T1_FIRST) not in out and digest not in out
    line = made.lines()[3]
    assert (line[KEY], line["successes"], line["root"]) == ("discarded", 800, str(made.root()))
    assert line["draw"] == book.Ledger.read(made.ledger).draws()[0]["hash"]
    # The place draws again under another id and root: its second seed.
    assert made.generate() == 2 and "(place)" in capsys.readouterr().err
    assert made.generate("w98", "T2") == 2 and "not filled" in capsys.readouterr().err
    assert made.generate("w98") == 0
    assert (made.lines()[4]["n"], made.lines()[4]["seed"]) == (2, T1_SECOND)
    made.ran("w98", 700)
    capsys.readouterr()
    # Inside the band: kept, exit 0. Checked once; the place is filled; the next may draw.
    assert made.command("check", "--place", "T1") == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "kept"
    assert made.command("check", "--place", "T1") == 2
    assert "Refused (place)" in capsys.readouterr().err
    assert made.generate("w99") == 2 and "the place is filled" in capsys.readouterr().err
    made.fail_with(compose.NoComposition("none", {}))
    assert made.generate("w99", "T2") == 3 and made.lines()[-2]["seed"] == T2_FIRST
    capsys.readouterr()

    # The summary: each place's candidates in order, with the number once checked.
    assert made.command("summary") == 0
    text = capsys.readouterr().out
    places = json.loads(text)["places"]
    assert places["T1"]["candidates"] == [
        {
            "world_id": "w97",
            "state": "discarded",
            "successes": 800,
            "episodes": 1008,
            "rate": 0.793651,
        },
        {"world_id": "w98", "state": "kept", "successes": 700, "episodes": 1008, "rate": 0.694444},
    ]
    assert (places["T2"]["draws"], places["T2"]["no_composition"]) == (1, 1)
    for secret in (str(T1_FIRST), str(T1_SECOND), digest, '"draft"', '"seed"', "forecast_seal"):
        assert secret not in text


def test_the_band_includes_its_ends_and_the_check_reads_only_a_sound_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.worlds import runner

    monkeypatch.setattr(runner, "readiness_problem", lambda record, paths: None)

    def decide(successes: Any, episodes: Any = 1000, stored: str = "h") -> str:
        ledger = _by_hand(tmp_path / "l.jsonl", [("T1", "w90", "standing")])
        root = Path(ledger.lines[1]["root"].replace("/gone", str(tmp_path)))
        _rechained(ledger.path, lambda rows: rows[1].update(root=str(root)))
        (root / "files").mkdir(parents=True, exist_ok=True)
        recipe = {"world_id": "w90", "seed": T1_FIRST}
        (root / "files/generator.json").write_text(json.dumps({"recipe": recipe}))
        (root / "files/hash").write_text(stored + "\n")
        logged = {"ok": successes, "n": episodes}
        record = {
            "world_id": "w90",
            "ready": True,
            "summary": {"measured": {"observed": {"logged": logged}}},
        }
        (root / "runner.json").write_text(json.dumps(record))
        if isinstance(successes, int) and isinstance(episodes, int) and 0 <= successes <= episodes:
            _export(root, successes, episodes)
        return str(book.check(book.Ledger.read(ledger.path), "T1")["decision"])

    assert [decide(k) for k in (599, 600, 750, 751)] == ["discarded", "kept", "kept", "discarded"]
    for successes, episodes in ((700.0, 1000), (True, 1), (5, 0), (11, 10), (None, 1000)):
        with pytest.raises(ValueError, match="lacks the observed window"):
            decide(successes, episodes)
    # Another hash under the draw's root is not the place's candidate.
    with pytest.raises(ValueError, match="does not hold the place's candidate"):
        decide(700, stored="another")


def test_ledger_verify_composes_each_recorded_seed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made = _Creation(tmp_path, monkeypatch)
    made.ready(None)
    assert made.generate() == 0
    made.ran("w97", 700)
    assert made.command("check", "--place", "T1") == 0
    real = made.world.generate

    def generate(recipe: Any, *args: Any, **kw: Any) -> Any:
        if recipe.seed == T2_FIRST:
            raise compose.NoComposition("none", {})
        return real(recipe, *args, **kw)

    monkeypatch.setattr(made.world, "generate", generate)
    assert made.generate("w98", "T2") == 3
    capsys.readouterr()
    assert made.command("verify") == 0
    text = capsys.readouterr().out
    counts = json.loads(text[text.index("{") :])
    assert counts == {
        "draws": 2,
        "repeated": 2,
        "differ": 0,
        "recipe_differs": 0,
        "open": 0,
        "open_compose": 0,
        "checks": 1,
        "checks_repeated": 1,
        "checks_differ": 0,
    }
    assert str(T1_FIRST) not in text and made.lines()[2]["world_hash"] not in text

    # From a tree that is not clean, or from another commit than the ledger's: refused.
    for state in ({**CLEAN, "clean": False}, {"commit": "d" * 40, "clean": True}):
        with monkeypatch.context() as patch:
            patch.setattr(book, "code_state", lambda state=state: dict(state))
            assert made.command("verify") == 2
            out = capsys.readouterr()
            assert "Refused (code)" in out.err and "clean checkout" in out.err and not out.out
    # A record changed together with nothing else: the export still counts 700.
    made.ran("w97", 650, exported=700)
    assert made.command("verify") == 1
    capsys.readouterr()

    # The runner's record no longer holds the recorded number; the root gone; the root's
    # world another: each counted, exit 1.
    made.ran("w97", 701)
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{") :])["checks_differ"] == 1
    made.ran("w97", 700)
    hash_file = made.root() / "files/hash"
    stored = hash_file.read_text()
    hash_file.write_text("another world\n")
    assert made.command("verify") == 1
    capsys.readouterr()
    hash_file.write_text(stored)
    made.root().rename(tmp_path / "moved")
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{") :])["checks_differ"] == 1
    (tmp_path / "moved").rename(made.root())
    assert made.command("verify") == 0
    capsys.readouterr()

    # A draw recorded as composing that does not, and one recorded as not composing that
    # does: counted, exit 1.
    monkeypatch.setattr(
        made.world,
        "generate",
        lambda *a, **k: (_ for _ in ()).throw(compose.NoComposition("none", {})),
    )
    assert made.command("verify") == 1
    text = capsys.readouterr().out
    counts = json.loads(text[text.index("{") :])
    assert (counts["repeated"], counts["differ"]) == (1, 1)

    # The record under the recipe has changed: its own outcome, and nothing is composed.
    made.fail_with(AssertionError("nothing may be composed"))
    monkeypatch.setattr(curves, "measured_sources", lambda jobs=None: ("another record",))
    assert made.command("verify") == 2
    out = capsys.readouterr()
    assert "Refused (record)" in out.err and "nothing was composed" in out.err and not out.out


def test_recipe_file_is_written_by_a_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fleetlab.worlds import __main__ as cli

    def write(path: Path, world_id: str = "w97", fleet: str = "w07", *more: str) -> int:
        identity = ["--world-id", world_id, "--public-fleet-id", fleet, "--name", "T1"]
        return int(cli.main(["recipe-file", *identity, "--out", str(path), *more]))

    path = tmp_path / "T1.json"
    assert write(path) == 0 and capsys.readouterr().out == ""
    assert json.loads(path.read_text()) == private_recipe("w97", "w07", name="T1")
    assert code_recipe(path) == private_recipe("w97", "w07", name="T1")
    reserve = tmp_path / "R1.json"
    assert write(reserve, "w97", "w07", "--kind", "reserve") == 0
    assert json.loads(reserve.read_text())["kind"] == "reserve"
    # A file that exists is not replaced; a checkout, a missing directory and an identity
    # the recipe does not take are refused, and nothing is written.
    before = path.read_bytes()
    assert write(path, "w98") == 2 and path.read_bytes() == before
    assert "Refused (path)" in capsys.readouterr().err
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /unneeded")
    for target, world_id, fleet in (
        (checkout / "T1.json", "w97", "w07"),
        (tmp_path / "absent" / "T1.json", "w97", "w07"),
        (tmp_path / "same.json", "w97", "w97"),
        (tmp_path / "bad.json", "T1", "w07"),
    ):
        assert write(target, world_id, fleet) == 2 and not target.exists()
        assert "Refused (path or identity)" in capsys.readouterr().err
