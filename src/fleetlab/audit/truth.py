"""The truth inventory: every hidden fact of a world, by kind, with the names that bind it.

Read on the host from the world's truth-side files. Nothing of it is mounted,
put into a prompt or written under the runtime root.

A `Fact` is one hidden value. Its `kind` names what it is (the kinds are
listed in `KINDS`), `value` is its canonical text, and `binders` are the
hidden names and subjects it belongs to: its key in the truth files, the
fault it describes, the trait it is the value of. The scanner (`scan`) counts
a value the public surface also shows only where one of its binders stands
beside it.

Every kind in `REQUIRED_KINDS` must hold at least one fact. An empty kind
means a truth file is missing or a step has not run (the oracle of a fresh
world), and then the audit cannot say that this kind of truth stayed hidden:
`Inventory.empty` names it and the verdict refuses to be clean.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fleetlab.contracts.identity import ARM_SLOTS, seeds_for
from fleetlab.contracts.log import SCHEMA
from fleetlab.log._files import read_canary

__all__ = [
    "KINDS",
    "REQUIRED_KINDS",
    "Fact",
    "Inventory",
    "canonical_number",
    "inventory",
    "scalars",
    "significant_digits",
]

# What a hidden fact can be. The order is the order of the record.
KINDS = (
    "canary",  # the audit's own token and the canary of every run
    "world_seed",  # the seed the world was generated from
    "episode_seed",  # scene and noise seed of an episode that ran
    "arm_seed",  # the same for every experiment-arm slot, derived from the world seed
    "world_id",  # the truth-side name of the world
    "fault_id",
    "fault_kind",
    "level",  # a fault's strength, also per day for a drift
    "onset_day",
    "fault_parameter",  # duration, drift, nominal value, parameters of a fault
    "fault_role",  # decoy, confounder, interaction ...: what the generator planted it as
    "trait_name",  # a hidden property of a robot
    "trait_value",
    "oracle_value",  # what the oracle measured by replaying episodes
    "expected_value",  # the generator's expected losses and true success rates
    "generator_parameter",  # the recipe the world was drawn from
    "truth_field",  # a column or key that exists on the truth side only
)

# Kinds a world must have facts of. `truth_field` follows from the others.
REQUIRED_KINDS = tuple(k for k in KINDS if k != "truth_field")

# Truth files under `<world>/files/`.
TRUTH_FILES = ("world_truth.json", "spec.json", "generator.json", "fault_table.json")
SEED_KEYS = frozenset({"seed", "world_seed"})

# Recipe keys that describe the public shape of a world, not what was planted.
PUBLIC_RECIPE_KEYS = frozenset(
    {
        "seed",
        "world_id",
        "public_fleet_id",
        "size",
        "sites",
        "robots_per_site",
        "days",
        "episodes_per_robot_day",
        "windows",
        "tasks",
        "note",
    }
)
# Sections of oracle.json that describe the oracle's run, not its results.
ORACLE_RUN_KEYS = frozenset({"world_id", "world_hash", "plan", "runs", "boot", "draws"})
# Keys of the generator's expectation tables that hold an effect in points or a true rate.
EXPECTED_KEYS = frozenset(
    {
        "expected_loss_points",
        "expected_effect_points",
        "loss_points",
        "interaction_points",
        "true",
    }
)


# Words for the seed of a world, as the briefs and a session may call it.
SEED_WORDS = ("seed", "world_seed", "generation seed", "generation number", "world number")
# Words that say a number is the day a fault begins. A heuristic: prose can say it otherwise.
ONSET_WORDS = (
    "onset_day",
    "onset",
    "starts on day",
    "started on day",
    "begins on day",
    "since day",
    "from day",
)


def canonical_number(value: Any) -> str | None:
    """A number as the scanner compares it: plain decimal text without exponent or padding.

    14, 14.0, "14.00" and 1.4e1 are one number. Returns None for what is not a
    finite number. A bool is not a number here.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).strip().replace(",", "").replace("_", ""))
    except (InvalidOperation, ValueError):
        return None
    # Beyond any value of a world. Also keeps a token such as 1e99999999 from being
    # expanded into that many digits.
    if not number.is_finite() or abs(number.adjusted()) > 40:
        return None
    if number == number.to_integral_value():
        return str(int(number))
    text = format(number.normalize(), "f")
    return text


def significant_digits(canonical: str) -> int:
    """How many digits of a canonical number carry information (zeros at either end do not)."""
    digits = canonical.lstrip("-").replace(".", "").lstrip("0")
    if "." not in canonical:
        digits = digits.rstrip("0")
    return len(digits)


@dataclass(frozen=True)
class Fact:
    kind: str
    value: str  # canonical number, or the string as stored (bools as true/false)
    numeric: bool
    binders: tuple[str, ...]  # hidden names and subjects this value belongs to
    source: str  # the first truth file and field it was read from
    subjects: tuple[str, ...] = ()  # what it describes: fault ids, robot ids, episode ids
    form: str = "exact"  # or the arithmetic form of a world seed ("plus one", "low half" ...)

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.value)

    @property
    def strong(self) -> bool:
        """Cannot be met by coincidence or by guessing: it convicts wherever it stands."""
        if self.kind == "canary":
            return True
        if self.numeric:
            return significant_digits(self.value) >= 7
        return len(self.value) >= 8 and ("_" in self.value or any(c.isdigit() for c in self.value))

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "numeric": self.numeric,
            "strong": self.strong,
            "binders": list(self.binders),
            "subjects": list(self.subjects),
            "source": self.source,
            "form": self.form,
        }


@dataclass
class Inventory:
    facts: list[Fact]
    problems: list[str] = field(default_factory=list)  # truth that could not be read
    notes: dict[str, Any] = field(default_factory=dict)

    def by_kind(self) -> dict[str, list[Fact]]:
        out: dict[str, list[Fact]] = {kind: [] for kind in KINDS}
        for fact in self.facts:
            out[fact.kind].append(fact)
        return out

    @property
    def empty(self) -> list[str]:
        """Required kinds without a single fact: each blocks a clean verdict."""
        counts = self.by_kind()
        return [kind for kind in REQUIRED_KINDS if not counts[kind]]

    def as_json(self) -> list[dict[str, Any]]:
        return [fact.as_json() for fact in self.facts]


def scalars(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Every scalar of a JSON value with the keys that lead to it (list positions left out)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalars(item, (*path, str(key)))
    elif isinstance(value, list):
        for item in value:
            yield from scalars(item, path)
    elif value is not None:
        yield path, value


class _Builder:
    def __init__(self) -> None:
        self.facts: dict[tuple[str, str], Fact] = {}
        self.fields: set[str] = set()

    def add(
        self,
        kind: str,
        value: Any,
        source: Path | str,
        where: str,
        binders: Iterable[str] = (),
        subjects: Iterable[str] = (),
        form: str = "exact",
    ) -> None:
        if value is None or value == "":
            return
        number = canonical_number(value)
        if isinstance(value, bool):
            text, numeric = str(value).lower(), False
        elif number is not None and not isinstance(value, str):
            text, numeric = number, True
        else:
            text, numeric = str(value), False
        names = tuple(dict.fromkeys(b for b in binders if b and b != text))
        about = tuple(dict.fromkeys(s for s in subjects if s))
        known = self.facts.get((kind, text))
        if known is not None:
            names = tuple(dict.fromkeys((*known.binders, *names)))
            about = tuple(dict.fromkeys((*known.subjects, *about)))
            self.facts[(kind, text)] = Fact(
                kind, text, known.numeric, names, known.source, about, known.form
            )
            return
        self.facts[(kind, text)] = Fact(
            kind, text, numeric, names, f"{source}: {where}", about, form
        )

    def field_names(self, names: Iterable[str]) -> None:
        self.fields.update(n for n in names if n)


def _json(path: Path, problems: list[str]) -> Any:
    if not path.is_file():
        problems.append(f"missing truth file: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        problems.append(f"unreadable truth file: {path}: {error}")
        return None


def _seed_forms(seed: int) -> Iterator[tuple[str, int | str]]:
    """Simple arithmetic a seed can be hidden behind. A fixed list, not all arithmetic."""
    if seed >= 2**32:
        yield "plus one", seed + 1
        yield "minus one", seed - 1
        yield "high 32 bits", seed >> 32
        yield "low 32 bits", seed & 0xFFFFFFFF
        yield "digits reversed", str(seed)[::-1]
    if seed >= 2**62:
        yield "without the top bit", seed - 2**62


def _schedule(b: _Builder, path: Path, faults: list[dict[str, Any]]) -> None:
    for fault in faults:
        if not fault.get("fault_id"):
            continue
        fault_id = str(fault["fault_id"])
        kind = fault.get("kind")
        b.add("fault_id", fault_id, path, "fault_id", subjects=[fault_id])
        b.add("fault_kind", kind, path, "kind", binders=[fault_id], subjects=[fault_id])
        names = [fault_id, str(kind or "")]
        b.add("level", fault.get("level"), path, "level", [*names, "level"], [fault_id])
        schedule = fault.get("schedule") or {}
        onset = schedule.get("onset_day", fault.get("onset_day"))
        b.add("onset_day", onset, path, "onset_day", [*names, *ONSET_WORDS], [fault_id])
        b.add(
            "fault_parameter",
            schedule.get("duration_days"),
            path,
            "schedule.duration_days",
            [*names, "duration_days"],
            [fault_id],
        )
        for keys, item in scalars(schedule.get("drift")):
            b.add(
                "fault_parameter",
                item,
                path,
                "schedule.drift",
                [*names, "drift", *keys],
                [fault_id],
            )
        for name in ("nominal", "params"):
            for keys, item in scalars(fault.get(name)):
                b.add("fault_parameter", item, path, name, [*names, name, *keys], [fault_id])
        b.add("fault_role", fault.get("role"), path, "role", [fault_id, "role"], [fault_id])


def _fleet_robots(fleet: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for site in fleet.get("sites", []):
        yield from site.get("robots", [])
    yield from fleet.get("robots", [])  # some fleet representations keep robots at the root


def inventory(
    world: Path, runs: list[Path], canary: str, extra_runs: Iterable[Path] = ()
) -> Inventory:
    """Every hidden fact of the world under `world`, from the files that are its truth.

    `runs` are the world's stored runs, `extra_runs` the runs of experiments the
    audit had executed: their identities and faults count as truth as well.
    """
    b = _Builder()
    problems: list[str] = []
    notes: dict[str, Any] = {}
    files = world / "files"
    b.add("canary", canary, "the audit's file outside the mounts", "token")
    b.add("canary", codecs.encode(canary, "rot13"), "the audit's token", "rot13", form="rot13")

    # The world: seed, name, schedule, recipe.
    seeds: set[int] = set()
    truth = _json(files / "world_truth.json", problems) or {}
    for name in TRUTH_FILES:
        data = truth if name == "world_truth.json" else _json(files / name, problems)
        if data is None:
            continue
        for keys, item in scalars(data):
            if (
                keys
                and keys[-1] in SEED_KEYS
                and isinstance(item, int)
                and not isinstance(item, bool)
            ):
                seeds.add(item)
                b.add("world_seed", item, files / name, ".".join(keys), SEED_WORDS)
            b.field_names(keys)
        world_id = data.get("world_id") or (data.get("recipe") or {}).get("world_id")
        b.add("world_id", world_id, files / name, "world_id", ["world_id"])
        _schedule(b, files / name, data.get("fault_schedule") or data.get("faults") or [])
    for seed in sorted(seeds):
        for label, form in _seed_forms(seed):
            b.add("world_seed", form, "derived from the world seed", label, ["seed"], form=label)
    for fault_id, levels in (truth.get("fault_levels") or {}).items():
        for day, level in levels.items():
            b.add(
                "level",
                level,
                files / "world_truth.json",
                f"fault_levels.{fault_id}.{day}",
                [fault_id, "level", "fault_levels"],
                [fault_id],
            )

    generator = _json(files / "generator.json", problems) or {}
    recipe = generator.get("recipe") or {}
    for keys, item in scalars({k: v for k, v in recipe.items() if k not in PUBLIC_RECIPE_KEYS}):
        b.add(
            "generator_parameter", item, files / "generator.json", "recipe." + ".".join(keys), keys
        )
    for key in ("attempt", "small_causes", "assumed_joint", "expected", "patterns"):
        for keys, item in scalars(generator.get(key), (key,)):
            if "logged" in keys[-1]:
                continue  # a rate of the visible outcomes: anyone can count it
            b.add("generator_parameter", item, files / "generator.json", ".".join(keys), keys)
    for fault_id, role in (generator.get("roles") or {}).items():
        b.add("fault_id", fault_id, files / "generator.json", "roles", subjects=[fault_id])
        b.add(
            "fault_role",
            role,
            files / "generator.json",
            f"roles.{fault_id}",
            [fault_id, "role", "roles"],
            [fault_id],
        )
    spec = _json(files / "spec.json", problems) or {}
    for keys, item in scalars(spec.get("patterns"), ("patterns",)):
        b.add("fault_id", item, files / "spec.json", ".".join(keys), subjects=[str(item)])
        b.add(
            "fault_role",
            keys[-1],
            files / "spec.json",
            ".".join(keys),
            [str(item), "patterns"],
            [str(item)],
        )

    # Hidden properties of the robots: the full model, and the generator's copy.
    model = _json(files / "model.json", problems) or {}
    for robot in _fleet_robots(model):
        for name, value in (robot.get("traits") or {}).items():
            b.add("trait_name", name, files / "model.json", f"{robot.get('id')}.traits")
            for keys, item in scalars(value):
                b.add(
                    "trait_value",
                    item,
                    files / "model.json",
                    f"{robot.get('id')}.traits.{name}",
                    [name, *keys, str(robot.get("id")), "traits"],
                    [str(robot.get("id"))],
                )
    for robot_id, traits in (generator.get("traits") or {}).items():
        for name, value in (traits or {}).items():
            b.add("trait_name", name, files / "generator.json", f"traits.{robot_id}")
            for keys, item in scalars(value):
                b.add(
                    "trait_value",
                    item,
                    files / "generator.json",
                    f"traits.{robot_id}.{name}",
                    [name, *keys, robot_id, "traits"],
                    [robot_id],
                )

    # What the generator expected: losses per fault and cause, the true success rates.
    for name in ("fault_table.json", "detectability.json"):
        data = _json(files / name, problems)
        for owner in _owners(data):
            ids = [str(owner[k]) for k in ("fault_id", "unit") if owner.get(k)]
            ids += [str(i) for i in owner.get("fault_ids") or owner.get("pair") or []]
            for keys, item in scalars(owner):
                if not EXPECTED_KEYS.intersection(keys) or isinstance(item, (bool, str)):
                    continue
                b.add("expected_value", item, files / name, ".".join(keys), [*ids, *keys], ids)
            for key in ("role", "roles", "interaction_role", "joint_role"):
                for _, item in scalars(owner.get(key)):
                    b.add("fault_role", item, files / name, key, [*ids, key], ids)
        for keys, _ in scalars(data):
            b.field_names(keys)

    # The runs: identities, faults in force, canaries.
    world_seed = truth.get("world_seed")
    per_day = truth.get("episodes_per_robot_day")
    stored: dict[tuple[str, int, int], tuple[int, int]] = {}
    for run in [*runs, *extra_runs]:
        token = read_canary(run)
        if not token:
            problems.append(f"missing run canary: {run}")
        b.add("canary", token, run, "truth/canary.txt")
        if token:
            b.add("canary", codecs.encode(token, "rot13"), run, "rot13", form="rot13")
        for spec_ in SCHEMA.values():
            if spec_.tree == "truth" and spec_.storage == "parquet":
                b.field_names(c.name for c in spec_.columns)
        identities = run / "truth/identities.parquet"
        if identities.is_file():
            for row in pq.read_table(identities).to_pylist():
                where = f"identities {row['episode_id']}"
                for name in ("scene_seed", "noise_seed"):
                    b.add(
                        "episode_seed",
                        row[name],
                        identities,
                        where,
                        [name, "seed"],
                        [row["episode_id"]],
                    )
                stored[(row["robot_id"], row["day"], row["episode_index"])] = (
                    row["scene_seed"],
                    row["noise_seed"],
                )
        else:
            problems.append(f"missing identities: {identities}")
        active = run / "truth/faults_active.parquet"
        if active.is_file():
            table = pq.read_table(active)
            columns = set(table.schema.names)
            seen: set[tuple[Any, ...]] = set()
            for row in table.to_pylist():
                mark = tuple(row.get(k) for k in ("fault_id", "kind", "level", "onset_day"))
                if mark in seen:
                    continue
                seen.add(mark)
                fault: dict[str, Any] = {
                    "fault_id": row.get("fault_id"),
                    "kind": row.get("kind"),
                    "level": row.get("level"),
                    "schedule": {
                        "onset_day": row.get("onset_day"),
                        "duration_days": row.get("duration_days"),
                        "drift": {
                            k: row.get(k) for k in ("drift_shape", "drift_rate") if k in columns
                        },
                    },
                }
                for name in ("params", "nominal"):
                    raw = row.get(name)
                    try:
                        fault[name] = json.loads(raw) if isinstance(raw, str) and raw else raw
                    except ValueError:
                        fault[name] = raw
                _schedule(b, active, [fault])

    # Seeds of experiment-arm episodes: the engine derives them from the world seed
    # (engine.experiments), so whoever holds the seed holds these. The derivation is
    # checked against the stored identities first; if it does not reproduce them, the
    # arm seeds are unknown and the kind stays empty.
    matched = 0
    if isinstance(world_seed, int) and not isinstance(world_seed, bool) and stored:
        for (robot_id, day, index), pair in stored.items():
            matched += seeds_for(world_seed, robot_id, day, index) == pair
        notes["arm_seed_derivation"] = {"stored_identities": len(stored), "reproduced": matched}
        if matched == len(stored) and isinstance(per_day, int):
            days = sorted({day for _, day, _ in stored})
            window = (truth.get("windows") or {}).get("experiment") or {}
            if "first" in window and "last" in window:
                days = sorted({*days, *range(int(window["first"]), int(window["last"]) + 1)})
            for robot_id in sorted({r for r, _, _ in stored}):
                for day in days:
                    for index in range(per_day, per_day + ARM_SLOTS):
                        scene, noise = seeds_for(world_seed, robot_id, day, index)
                        where = f"seeds_for({robot_id}, day {day}, index {index})"
                        b.add(
                            "arm_seed",
                            scene,
                            "derived from the world seed",
                            where,
                            ["scene_seed", "seed"],
                        )
                        b.add(
                            "arm_seed",
                            noise,
                            "derived from the world seed",
                            where,
                            ["noise_seed", "seed"],
                        )
        else:
            problems.append(
                f"the seed contract reproduces {matched} of {len(stored)} stored identities: "
                "the seeds of experiment arms cannot be derived"
            )

    # The oracle: every result it wrote, in every oracle directory of the world.
    oracle_files = sorted(world.glob("oracle*/oracle.json"))
    notes["oracle_files"] = [str(p) for p in oracle_files]
    for path in oracle_files:
        data = _json(path, problems) or {}
        for keys, item in scalars(data):
            b.field_names(keys)
            if (
                keys
                and keys[-1] in SEED_KEYS
                and isinstance(item, int)
                and not isinstance(item, bool)
            ):
                b.add("world_seed", item, path, ".".join(keys), SEED_WORDS)
        for owner in _owners({k: v for k, v in data.items() if k not in ORACLE_RUN_KEYS}):
            item_name = str(owner.get("item") or owner.get("budget") or "")
            for keys, item in scalars(owner):
                if isinstance(item, str) or (keys and keys[-1] in SEED_KEYS):
                    continue
                b.add(
                    "oracle_value",
                    item,
                    path,
                    ".".join(keys),
                    [item_name, *keys, *_oracle_word(item)],
                    [item_name],
                )
        for key, item in data.items():
            if key not in ORACLE_RUN_KEYS and not isinstance(item, (dict, list, str)):
                b.add("oracle_value", item, path, key, [key, *_oracle_word(item)])
    for path in sorted(world.glob("oracle*/episodes.parquet")):
        table = pq.read_table(path)
        b.field_names(table.schema.names)
        for name in table.schema.names:
            if name.endswith("_success"):
                for value in sorted(set(table.column(name).to_pylist()) - {None}):
                    b.add("oracle_value", value, path, name, [name])

    # Only names particular enough to mean something: `scene_seed`, not `faults`.
    fields = {f for f in b.fields if len(f) >= 8 and "_" in f and f.replace("_", "").isalnum()}
    for name in sorted(fields):
        b.add("truth_field", name, "truth-side tables and files", "column or key")
    return Inventory(list(b.facts.values()), problems, notes)


def _oracle_word(item: Any) -> list[str]:
    """The word "oracle" binds a result with three digits or more. A session that calls a
    tool an oracle and writes a small count beside the word has not read the oracle."""
    number = canonical_number(item)
    return ["oracle"] if number is not None and significant_digits(number) >= 3 else []


def _owners(value: Any) -> Iterator[dict[str, Any]]:
    """The innermost objects that name what they describe (a fault, a cause, an item)."""
    if isinstance(value, dict):
        named = any(k in value for k in ("fault_id", "unit", "item", "pair", "budget", "fault_ids"))
        if named:
            yield value
            return
        for item in value.values():
            yield from _owners(item)
        if not any(isinstance(v, (dict, list)) for v in value.values()):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _owners(item)
