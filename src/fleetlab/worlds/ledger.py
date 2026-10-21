"""The ledger of draws: every seed of a test or reserve world, and what became of it.

Design note, entry 16: the five test worlds are reproducible and every candidate is
reported, the discarded ones too. The ledger is a JSON-lines file on the truth side,
outside every checkout and outside the world's root.

**The seed is derived, not drawn.** A ledger cannot bind its operator: a tail cut away, a
second ledger and a seed handed in all leave nothing in it (review of rounds 9 and 9b,
F2). So the operator gets nothing to choose. The first line fixes a public random string,
the beacon, and every seed follows from it:

    seed = 2^62 + (SHA-256("BEACON|RECIPE_HASH|PLACE|N") mod 2^62)

with the four parts joined by "|" as UTF-8 text and N, the number of the place's draw
counted from 1, in decimal. A cut tail, a second ledger or a wrapper changes no seed.

The lines:

    beacon          the first line: the public random string
    drawn           written and synced BEFORE the seed is composed: place, n, seed, world
                    id, the recipe file's SHA-256, the recipe's hash, the band, the code's
                    commit, the record's digest, the engine's software, the root
    composed        the outcome that closes a draw, naming its `drawn` line: the accepted
    no_composition  draft's number, the world's hash and the forecast seal; or the number
                    of drafts and what refused them
    failed          an attempt that ended in an error or an interrupt: it closes nothing
    kept            entry 16's check of a candidate that ran, naming its `drawn` line:
    discarded       successes, episodes and rate of the fleet's logged success over the
                    observed window, read from the runner's record, and the decision

Only `composed` and `no_composition` close a draw. A draw that is open, after a kill, an
interrupt or an error, is repeated with its seed by the same command, and no draw is
taken anywhere while one is open. Every line carries `prev`, the SHA-256 of the line
before it (of its bytes without the newline; the first line carries 64 zeros). `read`
accepts a line only if it follows from the lines before it: the chain, the fields of its
kind, the derived seed, the draw's number, one outcome and one check per draw, the
check's decision from its own number. Commands that write hold an exclusive lock.

The rules: the first seed that composes is the place's candidate; a seed that does not
compose is replaced by the next draw; a kept candidate fills its place; a test world's
places are T1 to T5 in that order, a reserve world's `R` and a number; entry 16's stop
(three discarded candidates for a place, six in all) and ten draws in a row without a
composition for one place stop the draw for every place.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fleetlab.worlds.recipe import private_path

KEY = "line"
BEACON = "beacon"
DRAWN = "drawn"
FAILED = "failed"
OUTCOMES = ("composed", "no_composition")
CHECKS = ("kept", "discarded")
FIRST = "0" * 64
# The one writing of a beacon: the drand network's default chain, the round in decimal
# without leading zeros, and the 64 lower-case hexadecimal characters of its randomness.
# Nothing before, between or after: the same value written otherwise would be other seeds.
BEACON_FORM = re.compile(r"drand:default:[1-9][0-9]*:[0-9a-f]{64}")
TEST_PLACES = ("T1", "T2", "T3", "T4", "T5")
RESERVE_PLACE = re.compile(r"R[1-9][0-9]*")
MAX_DRAWS_WITHOUT_COMPOSITION = 10
# Entry 16, "when to stop": candidates outside the band, for one place and in all.
MAX_DISCARDED_PER_PLACE = 3
MAX_DISCARDED_IN_ALL = 6
# What an open draw must still find as it was recorded, to be repeated (and the commit).
RESUMED_ON = ("world_id", "recipe_file_sha256", "recipe_hash", "software", "root")
# The fields every line of a kind carries, with their types: `read` accepts nothing else.
FIELDS: dict[str, dict[str, type | tuple[type, ...]]] = {
    BEACON: {"beacon": str},
    DRAWN: {
        "place": str,
        "world_id": str,
        "n": int,
        "seed": int,
        "public_fleet_id": str,
        "name": str,
        "kind": str,
        "recipe_file_sha256": str,
        "recipe_hash": str,
        "band": list,
        "code": dict,
        "record": list,
        "software": dict,
        "root": str,
    },
    FAILED: {"place": str, "world_id": str, "draw": str, "error": str},
    "composed": {
        "place": str,
        "world_id": str,
        "draw": str,
        "draft": int,
        "world_hash": str,
        "forecast_seal": str,
    },
    "no_composition": {"place": str, "world_id": str, "draw": str, "drafts": int, "refusals": dict},
    **{
        name: {
            "place": str,
            "world_id": str,
            "draw": str,
            "successes": int,
            "episodes": int,
            "rate": (int, float),
            "band": list,
            "root": str,
        }
        for name in CHECKS
    },
}


class LedgerBusy(ValueError):
    """Another command holds the ledger's lock."""


class BrokenLedger(ValueError):
    """A line does not follow from the lines before it, or the last line is cut off."""


class RecordChanged(ValueError):
    """The calibration record is not the one a recorded draw was made on."""


def derived_seed(beacon: str, recipe_hash: str, place: str, n: int) -> int:
    """The seed of a place's n-th draw: 2^62 + (SHA-256("beacon|hash|place|n") mod 2^62)."""
    text = f"{beacon}|{recipe_hash}|{place}|{n}"
    return (1 << 62) | (int(hashlib.sha256(text.encode()).hexdigest(), 16) % (1 << 62))


def place_fits(kind: str, place: str) -> bool:
    """T1 to T5 for a test world; R1, R2 and so on for a reserve world."""
    if kind == "test":
        return place in TEST_PLACES
    return kind == "reserve" and RESERVE_PLACE.fullmatch(place) is not None


def place_before(place: str) -> str | None:
    """The place that has to be kept before this one draws; None for a first place."""
    number = int(place[1:])
    return None if number == 1 else f"{place[0]}{number - 1}"


def inside(band: Sequence[float], successes: int, episodes: int) -> bool:
    """Entry 16's rule: the rate inside the band, both ends included."""
    return bool(band[0] <= successes / episodes <= band[1])


def ledger_path(path: Path, root: Path | None = None) -> Path:
    """The ledger's resolved path: outside every checkout and outside the world's root."""
    resolved = private_path(path)
    if root is not None and resolved.is_relative_to(root.expanduser().resolve()):
        raise ValueError("the ledger must lie outside the world's root")
    if resolved.is_dir():
        raise ValueError("the ledger is a file")
    return resolved


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """An exclusive lock for a command that writes; a second one fails at once.

    The lock is on the ledger's directory, so that no file is made for it.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path.parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LedgerBusy("the ledger is in use by another command") from None
        yield
    finally:
        os.close(handle)


def line_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_state() -> dict[str, Any]:
    """The checkout's commit and whether its tree was clean: which code made the draw."""
    import subprocess

    from fleetlab.runtime import repo_root

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=False
        )
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("the code's commit cannot be read; creation runs from a git checkout")
    return {"commit": commit, "clean": not git("status", "--porcelain")}


class Ledger:
    """The lines of a ledger file, each verified against the lines before it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.texts: list[str] = []
        self.lines: list[dict[str, Any]] = []
        self.problem: str | None = None
        self._draws: dict[str, dict[str, Any]] = {}

    @property
    def beacon(self) -> str | None:
        return str(self.lines[0][BEACON]) if self.lines else None

    @property
    def commit(self) -> str | None:
        """The one commit every draw of this ledger was made from; None before a draw."""
        first = next(iter(self._draws.values()), None)
        return None if first is None else str(first["drawn"]["code"]["commit"])

    def _wrong(self, row: Any) -> str | None:
        """Why a line does not follow from the lines so far; None if it does."""
        fields = FIELDS.get(str(row.get(KEY))) if isinstance(row, dict) else None
        if fields is None or not isinstance(row.get("time"), str):
            return "is not a line of this ledger"
        if any(
            not isinstance(row.get(key), kind) or isinstance(row[key], bool)
            for key, kind in fields.items()
        ):
            return f"lacks a field of a `{row[KEY]}` line"
        kind = row[KEY]
        if (kind == BEACON) != (not self.lines):
            return "is a beacon line that is not the first, or a first line without the beacon"
        if kind == BEACON:
            written = BEACON_FORM.fullmatch(row[BEACON]) is not None
            return None if written else "has a beacon not written drand:default:ROUND:RANDOMNESS"
        draws = list(self._draws.values())
        if kind == DRAWN:
            if any(draw["outcome"] is None for draw in draws):
                return "draws although a draw is open"
            # Nothing a command could not have written: a clean tree, one commit for the
            # whole ledger, and a place only after the place before it was kept.
            commit = row["code"].get("commit")
            if row["code"].get("clean") is not True or not isinstance(commit, str):
                return "was drawn from a tree that is not clean"
            if self.commit not in (None, commit):
                return "names another commit than the ledger's other draws"
            before = sum(1 for draw in draws if draw["drawn"]["place"] == row["place"])
            if row["n"] != before + 1:
                return "has a number that is not the place's next draw"
            if row["seed"] != derived_seed(
                str(self.beacon), row["recipe_hash"], row["place"], row["n"]
            ):
                return "has a seed that is not the derived one"
            band = row["band"]
            sound = len(band) == 2 and all(isinstance(x, (int, float)) for x in band)
            if not sound or not place_fits(row["kind"], row["place"]):
                return "has a band or a place that no recipe has"
            earlier = place_before(row["place"])
            if earlier is not None and earlier not in {
                draw["drawn"]["place"] for draw in draws if state_of(draw) == "kept"
            }:
                return "draws for a place while the place before it is not kept"
            return None
        draw = self._draws.get(row["draw"])
        if draw is None:
            return "names a draw that is not in the ledger"
        if (row["place"], row["world_id"]) != (draw["drawn"]["place"], draw["drawn"]["world_id"]):
            return "has a place or a world id that is not its draw's"
        if kind in (FAILED, *OUTCOMES):
            return None if draw["outcome"] is None else "follows an outcome of its draw"
        if draw["outcome"] is None or draw["outcome"][KEY] != "composed" or draw["check"]:
            return "checks a draw that has no candidate, or a second time"
        whole = 0 <= row["successes"] <= row["episodes"] and row["episodes"] >= 1
        if (
            not whole
            or row["root"] != draw["drawn"]["root"]
            or row["band"] != draw["drawn"]["band"]
        ):
            return "has a count, a root or a band that is not its draw's"
        decision = "kept" if inside(row["band"], row["successes"], row["episodes"]) else "discarded"
        if kind != decision or row["rate"] != round(row["successes"] / row["episodes"], 6):
            return "has a decision or a rate that its own number does not give"
        return None

    def _take(self, text: str, row: dict[str, Any]) -> None:
        self.texts.append(text)
        self.lines.append(row)
        if row[KEY] == DRAWN:
            own = line_hash(text)
            self._draws[own] = {
                "drawn": row,
                "hash": own,
                "outcome": None,
                "check": None,
                "failed": 0,
            }
        elif row[KEY] == FAILED:
            self._draws[row["draw"]]["failed"] += 1
        elif row[KEY] != BEACON:
            self._draws[row["draw"]]["outcome" if row[KEY] in OUTCOMES else "check"] = row

    @classmethod
    def read(cls, path: Path, *, strict: bool = True) -> Ledger:
        """Read and verify. A broken ledger raises, or with `strict=False` is cut at the
        last line that holds and says what is wrong in `problem`."""
        ledger = cls(path)
        if not path.is_file():
            return ledger
        data = path.read_text()
        texts = data.splitlines()
        if data and not data.endswith("\n"):
            ledger.problem = "the last line is cut off"
            texts = texts[:-1]
        for number, text in enumerate(texts, 1):
            try:
                row = json.loads(text)
            except ValueError:
                row = None
            wrong = ledger._wrong(row)
            if wrong is None and row.get("prev") != ledger.head()[1]:
                wrong = "does not name the line before it: the chain is broken"
            if wrong is not None:
                ledger.problem = f"line {number} {wrong}"
                break
            ledger._take(text, row)
        if ledger.problem is not None and strict:
            raise BrokenLedger(ledger.problem)
        return ledger

    def head(self) -> tuple[int, str]:
        """The number of lines and the last line's hash (64 zeros for an empty ledger)."""
        return len(self.texts), line_hash(self.texts[-1]) if self.texts else FIRST

    def append(self, row: dict[str, Any]) -> str:
        """Append one line naming the line before it, sync it, and return its hash.

        The line must follow from the lines before it, as `read` will ask."""
        if self.problem is not None:
            raise BrokenLedger(self.problem)
        row = {**row, "time": now(), "prev": self.head()[1]}
        wrong = self._wrong(row)
        if wrong is not None:
            raise BrokenLedger(f"a new line {wrong}")
        text = json.dumps(row, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self._take(text, row)
        return line_hash(text)

    def draws(self) -> list[dict[str, Any]]:
        """Every draw in order: its `drawn` line, that line's hash, its outcome, its
        check and the number of its failed attempts."""
        return list(self._draws.values())


def open_ledger(path: Path, beacon: str) -> Ledger:
    """Begin a ledger that does not exist yet with the line that fixes its beacon."""
    if path.exists() or path.is_symlink():
        raise FileExistsError("the ledger exists; a ledger is opened once")
    if BEACON_FORM.fullmatch(beacon) is None:
        raise ValueError(
            "a beacon is written drand:default:ROUND:RANDOMNESS, the round in decimal "
            "without leading zeros, the randomness as 64 lower-case hexadecimal characters"
        )
    ledger = Ledger(path)
    ledger.append({KEY: BEACON, BEACON: beacon})
    return ledger


def state_of(draw: dict[str, Any]) -> str:
    """open, no_composition, standing (composed, not checked), kept or discarded."""
    if draw["outcome"] is None:
        return "open"
    if draw["outcome"][KEY] != "composed":
        return "no_composition"
    return "standing" if draw["check"] is None else str(draw["check"][KEY])


def open_draw(draws: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next((draw for draw in draws if state_of(draw) == "open"), None)


def standing(draws: Sequence[dict[str, Any]], place: str) -> dict[str, Any] | None:
    """The place's candidate that entry 16's check has not judged yet, if there is one."""
    for draw in draws:
        if draw["drawn"]["place"] == place and state_of(draw) == "standing":
            return draw
    return None


def stop(draws: Sequence[dict[str, Any]]) -> str | None:
    """Why no draw is taken for any place: entry 16's stop, or ten draws without composition."""
    discarded: dict[str, int] = {}
    runs: dict[str, int] = {}  # draws without a composition since the place's last candidate
    for draw in draws:
        place, name = draw["drawn"]["place"], state_of(draw)
        discarded[place] = discarded.get(place, 0) + (name == "discarded")
        runs[place] = runs.get(place, 0) + 1 if name == "no_composition" else 0
    if any(count >= MAX_DRAWS_WITHOUT_COMPOSITION for count in runs.values()):
        return (
            f"{MAX_DRAWS_WITHOUT_COMPOSITION} draws in a row without a composition for a "
            "place; no draw is taken for any place and the matter goes to a human"
        )
    if (
        any(count >= MAX_DISCARDED_PER_PLACE for count in discarded.values())
        or sum(discarded.values()) >= MAX_DISCARDED_IN_ALL
    ):
        return (
            "entry 16's limit of candidates outside the band is reached; no further "
            "world is drawn and the matter goes to a human"
        )
    return None


def refusal(draws: Sequence[dict[str, Any]], kind: str, place: str, world_id: str) -> str | None:
    """Why the place may not take a new draw under this world id now; None if it may.

    The reasons name no seed and no world: they are printed.
    """
    if not place_fits(kind, place):
        return "a test world's place is T1 to T5, a reserve world's R1, R2 and so on"
    states = [state_of(draw) for draw in draws if draw["drawn"]["place"] == place]
    kept = {draw["drawn"]["place"] for draw in draws if state_of(draw) == "kept"}
    if open_draw(draws) is not None:
        return "a draw is open; it is repeated with its seed before any other draw"
    if place in kept:
        return "the place is filled: its candidate was kept"
    if "standing" in states:
        return "the place has a standing candidate"
    stopped = stop(draws)
    if stopped is not None:
        return stopped
    before = place_before(place)
    if before is not None and before not in kept:
        return "the place before it is not filled; places are drawn in order"
    if any(d["drawn"]["world_id"] == world_id and d["drawn"]["place"] != place for d in draws):
        return "the ledger holds this world id under another place"
    if any(
        d["drawn"]["world_id"] == world_id and d["check"] is not None
        for d in draws
        if d["drawn"]["place"] == place
    ):
        return "this world id was a candidate already; a new draw needs another id and root"
    return None


def summary(ledger: Ledger) -> dict[str, Any]:
    """Per place the counts and its candidates in order; what the orchestrating session reads.

    A candidate shows its world id, its state and, once checked, successes, episodes and
    rate. No seed, no draft and no hash of a world.
    """
    draws = ledger.draws()
    places: dict[str, Any] = {}
    for draw in draws:
        here = places.setdefault(
            draw["drawn"]["place"],
            {"draws": 0, "open": 0, "no_composition": 0, "failed_attempts": 0, "candidates": []},
        )
        name = state_of(draw)
        here["draws"] += 1
        here["failed_attempts"] += draw["failed"]
        if name == "no_composition":
            here["no_composition"] += 1
            continue
        here["open"] += name == "open"
        candidate = {"world_id": draw["drawn"]["world_id"], "state": name}
        if draw["check"] is not None:
            candidate.update({k: draw["check"][k] for k in ("successes", "episodes", "rate")})
        here["candidates"].append(candidate)
    lines, last = ledger.head()
    return {
        "lines": lines,
        "head": last,
        "chain": ledger.problem or "intact",
        "beacon": ledger.beacon,
        "places": places,
        "open_draws": sum(1 for draw in draws if state_of(draw) == "open"),
        "discarded_in_all": sum(1 for draw in draws if state_of(draw) == "discarded"),
        "stopped": stop(draws),
        "limits": {
            "draws_without_composition_in_a_row": MAX_DRAWS_WITHOUT_COMPOSITION,
            "discarded_per_place": MAX_DISCARDED_PER_PLACE,
            "discarded_in_all": MAX_DISCARDED_IN_ALL,
        },
    }


def observed_count(root: Path, world_id: str) -> tuple[int, int]:
    """Successes and episodes of the fleet's logged success over the observed window.

    From the runner's record of the world at `root`, and only when the runner's own
    readiness test passes on that root (every step done, both chains held, the export
    checked by the current code) and the record says so. Nothing else is returned.
    """
    from fleetlab.worlds import runner
    from fleetlab.worlds.layout import WorldPaths

    paths = WorldPaths(root, world_id)
    record = paths.read_record()
    if record.get("world_id") != world_id:
        raise ValueError("the runner's record is not this world's")
    if record.get("ready") is not True or runner.readiness_problem(record, paths) is not None:
        raise ValueError("the world at the root is not ready by the runner's own test")
    try:
        logged = record["summary"]["measured"]["observed"]["logged"]
        successes, episodes = logged["ok"], logged["n"]
    except (KeyError, TypeError):
        raise ValueError("the runner's summary lacks the observed window") from None
    whole = isinstance(successes, int) and isinstance(episodes, int)
    if not whole or isinstance(successes, bool) or not 0 <= successes <= episodes or episodes < 1:
        raise ValueError("the runner's summary lacks the observed window")
    if export_count(paths.export) != (successes, episodes):
        raise ValueError("the runner's summary is not what the export's table counts")
    return successes, episodes


def export_count(export: Path) -> tuple[int, int] | None:
    """Successes and episodes counted from the world's export, the public table of the
    observed run that an analysis agent sees; None if the table cannot be read."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(export / "log" / "episodes.parquet", columns=["outcome"])
    except (OSError, pa.ArrowException):
        return None
    outcomes = table.column("outcome").to_pylist()
    return sum(1 for outcome in outcomes if outcome == "success"), len(outcomes)


def stored_candidate(root: Path, draw: dict[str, Any]) -> bool:
    """Whether the world under `root` is the draw's: its id, its seed and its hash."""
    try:
        recipe = json.loads((root / "files" / "generator.json").read_text())["recipe"]
        stored = (root / "files" / "hash").read_text().strip()
    except (OSError, ValueError, KeyError):
        return False
    drawn, outcome = draw["drawn"], draw["outcome"] or {}
    return (recipe.get("world_id"), recipe.get("seed"), stored) == (
        drawn["world_id"],
        drawn["seed"],
        outcome.get("world_hash"),
    )


def check(ledger: Ledger, place: str, root: Path | None = None) -> dict[str, Any]:
    """Entry 16's check of the place's standing candidate: read the number, decide, record.

    The root is the draw's own, as its `drawn` line records it; another one is refused.
    Inside the band (both ends included) the candidate is kept and fills the place;
    outside it is discarded. Returns what is printed: no seed, nothing of a composition.
    """
    draw = standing(ledger.draws(), place)
    if draw is None:
        raise LookupError("the place has no standing candidate without a check")
    drawn = draw["drawn"]
    own = Path(drawn["root"])
    if root is not None and root.expanduser().resolve() != own:
        raise ValueError("the root is not the one the draw was made into")
    if not stored_candidate(own, draw):
        raise ValueError("the draw's root does not hold the place's candidate")
    successes, episodes = observed_count(own, drawn["world_id"])
    decision = "kept" if inside(drawn["band"], successes, episodes) else "discarded"
    shown = {
        "place": place,
        "world_id": drawn["world_id"],
        "successes": successes,
        "episodes": episodes,
        "rate": round(successes / episodes, 6),
        "band": drawn["band"],
    }
    ledger.append({KEY: decision, **shown, "draw": draw["hash"], "root": str(own)})
    return {**shown, "decision": decision}


def verify(ledger: Ledger) -> dict[str, int]:
    """Compose every recorded seed again and count whether what the ledger says repeats.

    `read` has already recomputed every seed from the beacon, every draw's number, and
    every check's decision from its own number. Here the recipe is rebuilt from each
    draw's identity by the running code: its hash and its band must be the recorded
    ones. Every draw is composed, an open one too (also one that carries failed
    attempts): a closed draw repeats when draft, world hash and forecast seal are the
    recorded ones, or when it again has no composition. A checked candidate repeats
    when its root still holds it and the runner's record still holds the recorded
    number; a root that is gone counts as different. If the calibration record is not
    the one a draw was made on, nothing is composed at all (`RecordChanged`). Counts
    only: nothing of a composition is returned.
    """
    from fleetlab.contracts.identity import SoftwareVersion
    from fleetlab.worlds import curves, world
    from fleetlab.worlds.compose import NoComposition
    from fleetlab.worlds.recipe import WorldRecipe, private_recipe, stated_hash

    draws = ledger.draws()
    recipes: list[dict[str, Any] | None] = []
    for draw in draws:
        drawn = draw["drawn"]
        stated = private_recipe(
            drawn["world_id"], drawn["public_fleet_id"], name=drawn["name"], kind=drawn["kind"]
        )
        same_rules = stated_hash(stated) == drawn["recipe_hash"] and stated["band"] == drawn["band"]
        jobs = tuple(stated["calibration_jobs"])
        if same_rules and list(curves.measured_sources(jobs)) != drawn["record"]:
            raise RecordChanged("the calibration record is not the one the draws were made on")
        recipes.append(stated if same_rules else None)
    names = ("draws", "repeated", "differ", "recipe_differs", "open", "open_compose")
    counts = dict.fromkeys((*names, "checks", "checks_repeated", "checks_differ"), 0)
    for draw, rules in zip(draws, recipes, strict=True):
        drawn, outcome = draw["drawn"], draw["outcome"]
        counts["draws"] += 1
        if rules is None:
            counts["recipe_differs"] += 1
            continue
        recipe = WorldRecipe.model_validate({**rules, "seed": drawn["seed"]})
        try:
            again = world.generate(recipe, SoftwareVersion.model_validate(drawn["software"]))
        except NoComposition:
            again = None
        if outcome is None:
            counts["open"] += 1
            counts["open_compose"] += again is not None
            continue
        if again is None:
            same = outcome[KEY] == "no_composition"
        else:
            same = outcome[KEY] == "composed" and (
                again.composition.attempt,
                world.world_hash(again),
                again.composition.expected.get("forecast_seal"),
            ) == (outcome["draft"], outcome["world_hash"], outcome["forecast_seal"])
        counts["repeated" if same else "differ"] += 1
    for draw in draws:
        checked = draw["check"]
        if checked is None:
            continue
        counts["checks"] += 1
        root = Path(checked["root"])
        try:
            found = (
                observed_count(root, checked["world_id"]) if stored_candidate(root, draw) else None
            )
        except ValueError:
            found = None
        same = found == (checked["successes"], checked["episodes"])
        counts["checks_repeated" if same else "checks_differ"] += 1
    return counts
