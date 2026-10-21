"""Runs the SQL blocks of episode-cards' SKILL.md on an export, through `LogReader`.

The analysis agent runs the same blocks itself, through the tool server's
`query`, so that every number it reports is in the tool log. This file is the
reference runner: the tests drive it against a toy export, a person can run
it from the command line, and a tool server could call it. The blocks live
in SKILL.md only, so the text the agent reads and the code the tests run
cannot drift apart.

    python episode_cards_helpers.py EXPORT_DIR profile --rows '[["<episode id>"]]'
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fleetlab.log import LogReader

# --- runner: identical in every skill's helpers (tests/test_skills_library.py checks it)
SKILL = Path(__file__).resolve().with_name("SKILL.md")
_BLOCK = re.compile(r"```sql\n(-- helper: (?P<name>[a-z_]+)\n.*?)```", re.DOTALL)
_PARAM = re.compile(r"('(?:[^']|'')*'|-?[0-9][0-9.]*|NULL|TRUE|FALSE) AS ([a-z_]+)")
_INPUT = re.compile(r"^-- input: (?P<head>(?P<name>[a-z_]+)\([a-z_, ]*\))")

Rows = Iterable[Sequence[object]]


def blocks() -> dict[str, str]:
    """The SQL blocks of SKILL.md by name."""
    return {m["name"]: m[1] for m in _BLOCK.finditer(SKILL.read_text(encoding="utf-8"))}


def inputs_of(sql: str) -> list[str]:
    """The names of a block's inputs, in their order."""
    return [m["name"] for line in sql.splitlines() if (m := _INPUT.match(line))]


def literal(value: object) -> str:
    """A Python value as a SQL literal: text in single quotes, numbers bare, NULL for None."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def fill(
    sql: str,
    params: Mapping[str, object] | None = None,
    rows: Rows | None = None,
    inputs: Mapping[str, Rows] | None = None,
) -> str:
    """The block with its `params` line changed and input rows in place of the defaults.

    `rows` go to the block's first input, `inputs` to inputs by name; an input
    given neither keeps its default.
    """
    given = dict(inputs or {})
    names = inputs_of(sql)
    if rows is not None:
        if not names:
            raise KeyError("the block takes no input rows")
        given[names[0]] = rows
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise KeyError(f"no inputs {unknown}; the block has {names}")
    out: list[str] = []
    skipping = False
    for line in sql.splitlines():
        if line.startswith("WITH ") and " params AS (SELECT " in line and params:
            current = {name: value for value, name in _PARAM.findall(line)}
            missing = sorted(set(params) - set(current))
            if missing:
                raise KeyError(f"no parameters {missing}; the block has {sorted(current)}")
            merged = {**current, **{k: literal(v) for k, v in params.items()}}
            head = line[: line.index("params AS (SELECT ")]
            line = head + "params AS (SELECT " + ", ".join(f"{v} AS {k}" for k, v in merged.items())
            line += "),"
        match = _INPUT.match(line)
        if match and match["name"] in given:
            values = [", ".join(literal(v) for v in row) for row in given[match["name"]]]
            if not values:
                raise ValueError(f"input {match['name']} needs at least one row")
            out += [line, match["head"] + " AS (VALUES"]
            out += [f"  ({v})," for v in values[:-1]] + [f"  ({values[-1]})", "),"]
            skipping = True
            continue
        if line.startswith("-- end input"):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def run(
    reader: LogReader,
    name: str,
    params: Mapping[str, object] | None = None,
    rows: Rows | None = None,
    inputs: Mapping[str, Rows] | None = None,
) -> list[dict[str, Any]]:
    """Run the block `name` on the reader and return its rows as dictionaries."""
    sql = fill(blocks()[name], params, rows, inputs)
    result: list[dict[str, Any]] = reader.query(sql).to_pylist()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """`EXPORT BLOCK [--params JSON] [--rows JSON] [--inputs JSON]`: run a block, print rows."""
    import argparse

    from fleetlab.log import LogReader

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("export")
    parser.add_argument("block", choices=sorted(blocks()))
    parser.add_argument("--params", default="{}", help="parameter values as a JSON object")
    parser.add_argument("--rows", default=None, help="the first input's rows, a JSON list of lists")
    parser.add_argument("--inputs", default="{}", help="rows by input name, a JSON object")
    parser.add_argument("--sql", action="store_true", help="print the filled SQL, run nothing")
    args = parser.parse_args(argv)
    rows = None if args.rows is None else json.loads(args.rows)
    sql = fill(blocks()[args.block], json.loads(args.params), rows, json.loads(args.inputs))
    if args.sql:
        print(sql)
        return 0
    with LogReader(args.export, materialize=True, lock=True) as reader:
        for row in reader.query(sql).to_pylist():
            print(json.dumps(row, default=str))
    return 0


# --- end runner


def profile(
    reader: LogReader, episode_ids: Sequence[str], **params: object
) -> list[dict[str, Any]]:
    """One row per episode: grasp, collapse, release, stillness, reversals, frames to look at."""
    return run(reader, "profile", params, [(episode,) for episode in episode_ids])


if __name__ == "__main__":
    sys.exit(main())
