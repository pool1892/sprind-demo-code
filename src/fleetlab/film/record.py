"""The record page behind the film: a Markdown page whose numbers are bound to recorded runs.

`site/record/record.md` is written like the narration (`site/narration/script.md`): it
carries no number of its own. Every number is a placeholder, `{release_delivered}`, defined
in `site/record/fields.json` (`fleetlab.film-fields/1`, lane F's format and binding rules,
`fleetlab.film.fields`) as a field of a named record. A cut's records file
(`fleetlab.film-records/1`) names the files that stand for those records.

Some numbers need a derivation before they can be bound: a 95% Wilson interval from a count
and its runs, rescued and broken recounted from a release receipt's pairs, a count over a
list or a table, an agent's message picked from a message log. `fields.json` lists them
under `derived` (lane F's loader reads only `fields`, so the key is additive); `derive`
computes them into one JSON record, `derived.json`, which the page's fields read like any
other record (`"record": "derived"`). Each derived value names its inputs and their SHA-256.

A passage the runs must decide is marked in the page as a gap, `[gap: what goes here and
from which record]`, or as alternatives, `[one of: ... | ...]`. Editor's notes are HTML
comments and never reach the bound page. Three tables are generated from the binding itself:
`{provenance_records}` (every record file by public identifier and SHA-256),
`{provenance_numbers}` (every number on the page with its meaning and source) and
`{provenance_clips}` (every clip of the cut's film, from its provenance).

    python -m fleetlab.film.record bind PAGE --records FILE --out DIR [--fields FILE] [--final]
    python -m fleetlab.film.record derive --records FILE --out FILE [--fields FILE]
    python -m fleetlab.film.record site BUNDLE --records FILE --out DIR [--site-out DIR] [--final]

`bind` writes `DIR/page/index.md` (the bound page: the site's record page and, unchanged,
the Markdown to publish for review), `DIR/derived.json` and `DIR/bindings.json`. In a draft,
an unbound number shows as **[unbound: name]**, a gap stays as written, and a note under the
title says how many of each remain. `--final` refuses both (exit 4). `site` binds the page
and then builds the site of the bundle (`fleetlab.film-site/1`) with that page as its record
(`python -m fleetlab.film site`, lane F's builder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fleetlab.film.fields import (
    PLACEHOLDER,
    FieldError,
    FieldSpec,
    load_fields,
    load_records,
)
from fleetlab.film.fields import _value as field_value
from fleetlab.film.public import public_ref

__all__ = [
    "DERIVED_FORMAT",
    "GENERATED",
    "MIN_RUNS_FOR_INTERVAL",
    "PageBinding",
    "bind_page",
    "bind_record",
    "derive",
    "gaps",
    "main",
    "proportion",
    "safe_ref",
    "wilson",
]

DERIVED_FORMAT = "fleetlab.film-record-derived/1"
RECORD_BINDINGS_FORMAT = "fleetlab.film-record-bindings/1"
DERIVED_RECORD = "derived"
MIN_RUNS_FOR_INTERVAL = 20  # a count resting on at least 20 runs gets an interval
Z95 = 1.959963984540054
GENERATED = ("provenance_records", "provenance_numbers", "provenance_clips")
SITE_ROOT = Path(__file__).resolve().parents[3] / "site"
_COMMENT = re.compile(r"<!--.*?-->\n?", re.DOTALL)
_GAP = re.compile(r"\[(?:gap|one of):[^\]]*\]")
_FRONT = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


class DeriveError(ValueError):
    """A derived value cannot be computed from the records it names."""


# ---- statistics ----------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of k successes in n runs (95% at the default z)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"a count of {k} in {n} runs")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(k: int, n: int) -> dict[str, Any]:
    """A literal count with its percentage and, from 20 runs on, its 95% Wilson interval.

    `interval` is the phrase the page appends after "k of n": " (85%; 95% interval 64% to
    95%)" from 20 runs on, and the empty string below that, where the page gives the count
    alone."""
    out: dict[str, Any] = {"k": k, "n": n, "pct": f"{100 * k / n:.0f}%"}
    if n >= MIN_RUNS_FOR_INTERVAL:
        low, high = wilson(k, n)
        out.update(
            low_pct=f"{100 * low:.0f}%",
            high_pct=f"{100 * high:.0f}%",
            interval=f" ({out['pct']}; 95% interval {100 * low:.0f}% to {100 * high:.0f}%)",
        )
    else:
        out["interval"] = ""
    return out


# ---- reading records -----------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_values(data: Any, path: str) -> list[Any]:
    """Every value at a dotted path. A part `*` takes every item of a list (or every value of
    an object); a part `key=value` keeps the items of a list (or the objects) whose `key`
    equals `value` as text, so selections chain; a number takes that item of a list. Missing
    parts give no values."""
    current = [data]
    for part in path.split(".") if path else []:
        following: list[Any] = []
        for value in current:
            if part == "*":
                if isinstance(value, list):
                    following += value
                elif isinstance(value, dict):
                    following += list(value.values())
            elif "=" in part:
                key, _, wanted = part.partition("=")
                items = value if isinstance(value, list) else [value]
                following += [v for v in items if isinstance(v, dict) and str(v.get(key)) == wanted]
            elif isinstance(value, dict) and part in value:
                following.append(value[part])
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                following.append(value[int(part)])
        current = following
    return current


def _one(data: Any, path: str, where: str) -> Any:
    found = path_values(data, path)
    if not found:
        raise DeriveError(f"{where} has nothing at {path}")
    return found[0]


def _table_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_table(path).to_pylist())
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@dataclass
class _Reader:
    records: dict[str, list[Path]]
    read: dict[str, str]
    derived: dict[str, Any] = field(default_factory=dict)  # the values derived so far
    words: dict[str, str] = field(default_factory=dict)  # actions in plain words

    def files(self, record: str) -> list[Path]:
        files = self.records.get(record)
        if files is None:
            raise DeriveError(f"this cut provides no record {record}")
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise DeriveError(f"{', '.join(missing)} does not exist")
        for p in files:
            self.read.setdefault(str(p), _sha256(p))
        return files

    def json(self, record: str) -> Any:
        files = self.files(record)
        if len(files) != 1:
            raise DeriveError(f"record {record} is one JSON file, not {len(files)}")
        return json.loads(files[0].read_text(encoding="utf-8"))

    def jsons(self, record: str) -> list[tuple[Path, Any]]:
        return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in self.files(record)]

    def rows(self, record: str) -> list[dict[str, Any]]:
        return [r for p in self.files(record) for r in _table_rows(p)]

    def text(self, record: str) -> str:
        files = self.files(record)
        if len(files) != 1:
            raise DeriveError(f"record {record} is one text file, not {len(files)}")
        return files[0].read_text(encoding="utf-8")

    def resolve(self, value: Any, where: str) -> Any:
        """A filter's value, or `{"derived": name, "key": k}`: a value derived earlier."""
        if isinstance(value, dict) and "derived" in value:
            entry = self.derived.get(str(value["derived"]))
            if entry is None:
                raise DeriveError(f"{where} needs derived {value['derived']}, not available")
            found = path_values(entry, str(value.get("key", "value")))
            if not found:
                raise DeriveError(f"{where}: derived {value['derived']} has no {value.get('key')}")
            return found[0]
        return value


def _filtered(rows: list[Any], spec: dict[str, Any], reader: _Reader | None = None) -> list[Any]:
    def resolved(filters: dict[str, Any]) -> dict[str, Any]:
        if reader is None:
            return dict(filters)
        return {k: reader.resolve(v, "a filter") for k, v in filters.items()}

    where = resolved(spec.get("where") or {})
    where_not = resolved(spec.get("where_not") or {})
    return [
        r
        for r in rows
        if isinstance(r, dict)
        and all(r.get(k) == v for k, v in where.items())
        and all(r.get(k) != v for k, v in where_not.items())
    ]


def _items(reader: _Reader, spec: dict[str, Any]) -> list[Any]:
    """The items a count or a join reads: a table's rows, or the values at a JSON path."""
    record = str(spec["record"])
    if spec.get("table"):
        rows: list[Any] = reader.rows(record)
    else:
        data = reader.json(record)
        path = str(spec.get("path", ""))
        rows = path_values(data, path)
        if not rows and not path_values(data, path.split(".*")[0]):
            raise DeriveError(f"record {record} has nothing at {path}")
        if len(rows) == 1 and isinstance(rows[0], list):
            rows = rows[0]
    if spec.get("where") or spec.get("where_not"):
        rows = _filtered(rows, spec, reader)
    if spec.get("column"):
        column = str(spec["column"])
        rows = [
            (path_values(r, column) or [None])[0] if "." in column else r.get(column)
            for r in rows
            if isinstance(r, dict)
        ]
    return rows


def _number(reader: _Reader, spec: dict[str, Any], where: str) -> int:
    kind = spec.get("kind", "value")
    if kind == "count":
        return len(_items(reader, spec))
    if kind == "value":
        value = _one(reader.json(str(spec["record"])), str(spec["path"]), where)
        if isinstance(value, bool) or not isinstance(value, int):
            raise DeriveError(f"{where}: {spec['path']} is not a whole number")
        return value
    raise DeriveError(f"{where}: a count is a value or a count, not {kind}")


def _action_words(action: dict[str, Any], templates: dict[str, str] | None = None) -> str:
    """An action in words: a template of `fields.json`'s `action_words` for its verb and
    what it acts on ("set:control.latency_compensation_steps"), else verb, key and scope."""
    target = action.get("target") or {}
    what = target.get("config_key") or target.get("component") or target.get("condition_key")
    template = (templates or {}).get(f"{action.get('verb')}:{what}")
    if template:
        return template.format(scope=target.get("scope"), value=action.get("value"), what=what)
    words = f"{action.get('verb')} {what or ''}".strip()
    scope = target.get("scope")
    if scope:
        words += f" at {scope}"
    if action.get("value") is not None:
        words += f" to {action['value']}"
    return words


def _derive_one(reader: _Reader, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec.get("kind")
    where = f"derived {name}"
    if kind == "proportion":
        k = _number(reader, spec["k"], where)
        n = _number(reader, spec["n"], where)
        if n <= 0:
            raise DeriveError(f"{where}: no runs to count")
        return proportion(k, n)
    if kind in ("value", "count"):
        return {"value": _number(reader, spec, where)}
    if kind == "distinct":
        values = _items(reader, spec)
        return {"value": len({json.dumps(v, sort_keys=True, default=str) for v in values})}
    if kind == "text":
        found = [
            v
            for v in path_values(reader.json(str(spec["record"])), str(spec["path"]))
            if v is not None and not isinstance(v, dict | list)
        ]
        prefix = spec.get("startswith")
        if prefix:
            found = [v for v in found if str(v).startswith(str(prefix))]
        if not found:
            raise DeriveError(f"{where}: no text at {spec['path']}")
        return {"value": str(found[-1] if spec.get("pick") == "last" else found[0])}
    if kind == "digest":
        value = str(_one(reader.json(str(spec["record"])), str(spec["path"]), where))
        return {"value": value[: int(spec.get("chars", 12))]}
    if kind == "join":
        items = [str(i) for i in _items(reader, spec) if not isinstance(i, dict | list)]
        if not items:
            raise DeriveError(f"{where}: nothing to list at {spec.get('path')}")
        if spec.get("code"):
            items = [f"`{i}`" for i in items]
        return {"value": str(spec.get("sep", "; ")).join(items)}
    if kind == "pairs":
        receipt = reader.json(str(spec["record"]))
        pairs = receipt.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise DeriveError(f"{where}: the receipt has no pairs")
        counts = {
            "n": len(pairs),
            "rescued": sum(bool(p["delivered"] and not p["naive_delivered"]) for p in pairs),
            "broken": sum(bool(p["naive_delivered"] and not p["delivered"]) for p in pairs),
            "both": sum(bool(p["delivered"] and p["naive_delivered"]) for p in pairs),
            "neither": sum(bool(not p["delivered"] and not p["naive_delivered"]) for p in pairs),
        }
        for key in ("rescued", "broken"):
            if key in receipt and receipt[key] != counts[key]:
                raise DeriveError(f"{where}: the receipt says {key} {receipt[key]}, its pairs "
                                  f"{counts[key]}")  # fmt: skip
        return counts
    if kind == "verdict":  # the receipt's judgement against the shift lead's bar, in words
        receipt = reader.json(str(spec["record"]))
        fresh = " on starts no program had run on" if receipt.get("starts_fresh") else ""
        if receipt.get("released"):
            words = f"met the shift lead's bar{fresh} and was released"
        elif receipt.get("passed"):
            words = f"met the shift lead's bar{fresh}; the installation kept it unreleased"
        elif receipt.get("passed") is False:
            words = f"ran{fresh} and did not meet the shift lead's bar"
        else:
            words = "had no bar of this size to be judged against"
        return {"value": words}
    if kind == "sha256":  # a record file's own checksum, as the records table lists it
        files = reader.files(str(spec["record"]))
        if len(files) != 1:
            raise DeriveError(f"{where}: record {spec['record']} is one file, not {len(files)}")
        return {"value": reader.read[str(files[0])][: int(spec.get("chars", 12))]}
    if kind == "actions":
        found = path_values(reader.json(str(spec["record"])), str(spec["path"]))
        actions = found[0] if len(found) == 1 and isinstance(found[0], list) else found
        if not found:
            raise DeriveError(f"{where}: nothing at {spec['path']}")
        adopted = [
            _action_words(a.get("action", a), reader.words) for a in actions if isinstance(a, dict)
        ]
        return {"value": "; ".join(adopted) if adopted else "no change", "count": len(adopted)}
    if kind == "evaluation":
        data = reader.json(str(spec["record"]))
        state = _one(data, f"{spec.get('path', 'evaluation.summary')}.states.name={spec['state']}",
                     where)  # fmt: skip
        result = proportion(int(state["successes"]), int(state["attempts"]))
        recorded = state.get("interval")
        if recorded is not None:
            low, high = wilson(int(state["successes"]), int(state["attempts"]))
            if abs(low - recorded[0]) > 1e-6 or abs(high - recorded[1]) > 1e-6:
                raise DeriveError(f"{where}: the record's interval {recorded} is not Wilson's")
        return {**result, "changes": state.get("changes")}
    if kind == "evaluation_pair":
        data = reader.json(str(spec["record"]))
        pairs = path_values(data, f"{spec.get('path', 'evaluation.summary')}.pairs")
        pairs = pairs[0] if pairs and isinstance(pairs[0], list) else pairs
        for pair in pairs:
            if pair.get("first") == spec["first"] and pair.get("second") == spec["second"]:
                return {"rescued": int(pair["rescued"]), "broken": int(pair["broken"])}
        raise DeriveError(f"{where}: no pair {spec['first']} to {spec['second']}")
    if kind == "rerenders":
        reports = reader.jsons(str(spec["record"]))
        accepted = [r for _, r in reports if r.get("accepted") and r.get("verdict_matches")]
        return {
            "reports": len(reports),
            "accepted": len(accepted),
            "hashes_compared": sum(int(r.get("hashes_compared") or 0) for _, r in reports),
            "mismatches": sum(int(r.get("hash_mismatches") or 0) for _, r in reports),
        }
    if kind == "film_clips":
        film = reader.json(str(spec["record"]))
        clips = [
            c
            for b in film.get("beats", [])
            if b.get("id") == spec["beat"]
            for s in b.get("shots", [])
            for c in s.get("clips", [])
        ]
        if not clips:
            raise DeriveError(f"{where}: the film shows no clip in beat {spec['beat']}")
        wanted = str(spec.get("outcome", "delivered"))
        return {"clips": len(clips), "outcome": sum(c.get("outcome") == wanted for c in clips)}
    if kind == "recipe":
        check = reader.json(str(spec["record"]))
        out: dict[str, Any] = {}
        for role, robot in (check.get("robots") or {}).items():
            episodes = [e for e in check.get("episodes", []) if e.get("robot_id") == robot]
            if not episodes:
                raise DeriveError(f"{where}: no probe episode of {robot}")
            firsts = sorted({tuple(x["first_movement_rows"] for x in e["edges"]) for e in episodes})
            halves = sorted({tuple(x["half_way_rows"] for x in e["edges"]) for e in episodes})
            out[role] = {
                "robot": robot,
                "repeats": len(episodes),
                "first_move": " / ".join(" and ".join(str(v) for v in f) for f in firsts),
                "half_way": " / ".join(" and ".join(str(v) for v in h) for h in halves),
            }
        if not out:
            raise DeriveError(f"{where}: the recipe check names no robots")
        return out
    if kind == "noted_site":
        description = reader.json(str(spec["record"]))
        noted = [
            site
            for site in path_values(description, "fleet.sites.*")
            if isinstance(site, dict)
            and any(str(spec.get("note", "")) in str(n) for n in site.get("notes") or [])
        ]
        if len(noted) != 1:
            raise DeriveError(f"{where}: {len(noted)} sites carry the note, not one")
        return {"value": str(noted[0]["id"]), "note": str(noted[0]["notes"][0]),
                "robots": len(noted[0].get("robots") or [])}  # fmt: skip
    if kind == "quote":
        if spec.get("path"):
            texts = [str(v) for v in path_values(reader.json(str(spec["record"])), spec["path"])]
        else:
            texts = [reader.text(str(spec["record"]))]
        for text in texts:
            match = re.search(str(spec["regex"]), text)
            if match:  # a quote across lines of a Markdown file reads as one line
                quoted = str(match.group(1) if match.groups() else match.group(0))
                return {"value": re.sub(r"\s+", " ", quoted).strip()}
        raise DeriveError(f"{where}: the record does not say {spec['regex']!r}")
    if kind in ("lowest", "highest"):
        rows = _filtered(reader.rows(str(spec["record"])), spec, reader)
        by: dict[str, list[int]] = {}
        for r in rows:
            tally = by.setdefault(str(r[spec["group"]]), [0, 0])
            tally[0] += r.get(spec["column"]) == spec["success"]
            tally[1] += 1
        if not by:
            raise DeriveError(f"{where}: no rows to rank")
        if kind == "lowest":
            name, (k, n) = min(by.items(), key=lambda item: (item[1][0] / item[1][1], item[0]))
        else:  # the highest share; of equal shares, the first group by name
            name, (k, n) = min(by.items(), key=lambda item: (-item[1][0] / item[1][1], item[0]))
        return {"value": name, **proportion(k, n)}
    if kind == "probe":
        from fleetlab.loop.probes import step_lags  # the probe route's own reading

        episodes = [
            e for e in reader.rows(str(spec["episodes"])) if e.get("probe_id") == spec["probe"]
        ]
        steps: dict[str, list[dict[str, Any]]] = {}
        for p in reader.files(str(spec["steps"])):
            for row in _table_rows(p):
                steps.setdefault(str(row["episode_id"]), []).append(row)
        robots: dict[str, dict[str, Any]] = {}
        for e in sorted(episodes, key=lambda e: (e["robot_id"], e["repeat"])):
            rows = sorted(steps.get(str(e["episode_id"]), []), key=lambda r: r["step"])
            if not rows:
                raise DeriveError(f"{where}: no probe steps for {e['episode_id']}")
            edges = step_lags(rows)
            entry = robots.setdefault(
                e["robot_id"], {"site": e["site_id"], "first": [], "half": []}
            )
            entry["first"].append(tuple(x["first_movement_rows"] for x in edges))
            entry["half"].append(tuple(x["half_way_rows"] for x in edges))
        if not robots:
            raise DeriveError(f"{where}: no episode of probe {spec['probe']}")
        out = {}
        for robot, entry in robots.items():
            first = _per_edge(entry["first"])
            half = _per_edge(entry["half"])
            out[robot] = {"site": entry["site"], "repeats": len(entry["first"]),
                          "first_move": first, "half_way": half}  # fmt: skip
        out["summary"] = "; ".join(
            f"{robot} (site {e['site']}): first moved after {e['first_move']} control steps, "
            f"half-way after {e['half_way']}"
            for robot, e in robots.items()
            for e in [out[robot]]
        )
        return out
    if kind == "breakdown":
        rows = reader.rows(str(spec["record"]))
        states = [str(s) for s in spec["states"]]
        lines = ["| " + " | ".join([str(spec.get("title", "Attempts")), *spec["labels"]]) + " |",
                 "|---" * (len(states) + 1) + "|"]  # fmt: skip
        for group in spec["groups"]:
            chosen = _filtered(rows, group, reader)
            label = str(group["label"]).format(**{
                k: reader.resolve(v, where) for k, v in (group.get("where") or {}).items()
            })  # fmt: skip
            cells = []
            for state in states:
                mine = [r for r in chosen if r.get("state") == state]
                if not mine:
                    raise DeriveError(f"{where}: no attempts of {state} for {label}")
                k = sum(r.get(spec["column"]) == spec["success"] for r in mine)
                cells.append(f"{k} of {len(mine)}{proportion(k, len(mine))['interval']}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return {"value": "\n".join(lines)}
    if kind == "experiments_table":
        requests = {str(r["experiment_id"]): r for r in reader.rows(str(spec["requests"]))}
        results = {
            str(r["experiment_id"]): r
            for r in reader.rows(str(spec["results"]))
            if r.get("status") == "done"
        }
        outcome = {str(r["episode_id"]): r.get("outcome") == "success"
                   for r in reader.rows(str(spec["episodes"]))}  # fmt: skip
        lines = ["| Experiment | Change tested | Without it | With it | Rescued | Broken |",
                 "|---|---|---|---|---|---|"]  # fmt: skip
        for experiment in spec["experiments"]:
            asked_for, got = requests.get(experiment), results.get(experiment)
            if asked_for is None or got is None:
                raise DeriveError(f"{where}: experiment {experiment} has no finished result")
            arms = {int(a["arm_index"]): a["episode_ids"] for a in got["arms"]}
            asked = asked_for["request"]["arms"]
            control = [outcome[e] for e in arms[0]]
            for index in sorted(arms):
                if index == 0:
                    continue
                treated = [outcome[e] for e in arms[index]]
                if len(treated) != len(control):
                    raise DeriveError(f"{where}: {experiment} arm {index} is not paired")
                words = "; ".join(_action_words(a, reader.words) for a in asked[index]["actions"])
                k0, k1 = sum(control), sum(treated)
                rescued = sum(t and not c for c, t in zip(control, treated, strict=True))
                broken = sum(c and not t for c, t in zip(control, treated, strict=True))
                lines.append(
                    f"| {experiment} | {words} | {k0} of {len(control)}"
                    f"{proportion(k0, len(control))['interval']} | {k1} of {len(treated)}"
                    f"{proportion(k1, len(treated))['interval']} | {rescued} | {broken} |"
                )
        return {"value": "\n".join(lines)}
    if kind == "action_costs":
        costs = reader.rows(str(spec["costs"]))
        found = path_values(reader.json(str(spec["record"])), str(spec["path"]))
        actions = found[0] if len(found) == 1 and isinstance(found[0], list) else found
        hours: list[float] = []
        for item in actions:
            action = item.get("action", item)
            target = action.get("target") or {}
            acts_on: str | None
            if target.get("config_key"):
                acts_on = "config_key"
            elif target.get("condition_key"):
                acts_on = "condition_key"
            else:
                acts_on = target.get("component")
            priced = [
                c
                for c in costs
                if c.get("verb") == action.get("verb") and c.get("target") == acts_on
            ]
            if len(priced) != 1 or priced[0].get("hours") is None:
                raise DeriveError(f"{where}: no recorded cost for {_action_words(action)}")
            hours.append(float(priced[0]["hours"]))
        if not hours:
            raise DeriveError(f"{where}: no actions")
        each = [f"{h:g}" if h != int(h) else f"{h:.1f}" for h in hours]
        listed = each[0] if len(each) == 1 else ", ".join(each[:-1]) + f" and {each[-1]}"
        return {"value": listed, "total": f"{sum(hours):.1f}", "count": len(hours)}
    if kind == "jobs_table":
        names = {
            str(j["job_id"]): str(j["program"])
            for j in path_values(reader.json(str(spec["record"])), "batches.*.jobs.*")
            if isinstance(j, dict)
        }
        per_job: dict[str, list[dict[str, Any]]] = {}
        for row in reader.rows(str(spec["episodes"])):
            per_job.setdefault(str(row["job_id"]), []).append(row)
        if not per_job:
            raise DeriveError(f"{where}: no episodes")
        lines = ["| Program run | Program | Development starts | Delivered |", "|---|---|---|---|"]
        for job in sorted(per_job):
            rows = per_job[job]
            inits = sorted(int(r["start"]["init_state"]) for r in rows)
            k = sum(bool(r["delivered"]) for r in rows)
            starts = (
                f"{inits[0]} to {inits[-1]}"
                if inits == list(range(inits[0], inits[-1] + 1))
                else ", ".join(str(i) for i in inits)
            )
            lines.append(f"| {job} | {names.get(job, 'not in the record')} | {starts} | "
                         f"{k} of {len(rows)}{proportion(k, len(rows))['interval']} |")  # fmt: skip
        return {"value": "\n".join(lines)}
    if kind == "difference":
        return {"value": _number(reader, spec["a"], where) - _number(reader, spec["b"], where)}
    if kind == "start_range":
        if spec.get("rows_path"):  # the rows are a list in a JSON record
            found = path_values(reader.json(str(spec["record"])), str(spec["rows_path"]))
            items = found[0] if len(found) == 1 and isinstance(found[0], list) else found
            rows = _filtered([r for r in items if isinstance(r, dict)], spec, reader)
        else:
            rows = _filtered(reader.rows(str(spec["record"])), spec, reader)
        inits = sorted(
            {int(_one(r, str(spec.get("path", "start.init_state")), where)) for r in rows}
        )
        if not inits:
            raise DeriveError(f"{where}: no episodes")
        return {"value": f"{inits[0]} to {inits[-1]}", "count": len(inits)}
    if kind == "added":
        before_path = str(spec.get("before_path", spec["path"]))
        before = str(_one(reader.json(str(spec["before"])), before_path, where)).strip()
        after = str(_one(reader.json(str(spec["record"])), str(spec["path"]), where)).strip()
        if not after.startswith(before) or after == before:
            raise DeriveError(f"{where}: the later text does not extend the earlier one")
        return {"value": after[len(before) :].strip()}
    raise DeriveError(f"{where}: unknown kind {kind!r}")


def _per_edge(readings: list[tuple[Any, ...]]) -> str:
    """Readings per commanded step over the repetitions: "18 and 18" when every repetition
    agrees, else each step's values over the repetitions, "4, 12 and 5 then 37, 45 and 39"."""
    if len(set(readings)) == 1:
        return " and ".join(str(v) for v in readings[0])
    per_edge = list(zip(*readings, strict=True))
    return " then ".join(", ".join(str(v) for v in edge[:-1]) + f" and {edge[-1]}"
                         for edge in per_edge)  # fmt: skip


def _inputs(spec: dict[str, Any]) -> list[str]:
    keys = ("record", "episodes", "steps", "before", "requests", "results", "costs")
    names = [str(spec[k]) for k in keys if k in spec]
    for key in ("k", "n"):
        if isinstance(spec.get(key), dict):
            names += _inputs(spec[key])
    return list(dict.fromkeys(names))


def derive(
    spec: dict[str, Any], records: dict[str, list[Path]], words: dict[str, str] | None = None
) -> dict[str, Any]:
    """Every derived value `spec` names that the records allow; the others with the reason.
    `words` puts actions in plain words (`fields.json`'s `action_words`)."""
    values: dict[str, Any] = {}
    reader = _Reader(records, {}, values, dict(words or {}))
    unavailable: dict[str, str] = {}
    for name, item in spec.items():
        if not isinstance(item, dict):
            raise FieldError(f"derived {name}: an object expected")
        try:
            result = _derive_one(reader, name, item)
        except DeriveError as error:
            unavailable[name] = str(error)
            continue
        except (KeyError, TypeError, ValueError) as error:
            unavailable[name] = f"derived {name}: the record does not have the form ({error!r})"
            continue
        values[name] = {
            **result,
            "how": str(item.get("how") or item.get("kind")),
            "inputs": [
                {"record": r, "id": public_ref(p), "sha256": reader.read[str(p)]}
                for r in _inputs(item)
                for p in records.get(r, [])
                if str(p) in reader.read
            ],
        }
    return {
        "format": DERIVED_FORMAT,
        "values": values,
        "unavailable": unavailable,
        "files_read": dict(reader.read),
    }


# ---- binding the page ----------------------------------------------------------------------


def gaps(text: str) -> list[str]:
    """The passages a page still leaves to the runs: `[gap: ...]` and `[one of: ...]`."""
    return _GAP.findall(_COMMENT.sub("", text))


@dataclass
class PageBinding:
    text: str  # the bound page
    values: dict[str, dict[str, Any]]
    unbound: dict[str, str]
    gaps: list[str]
    files: dict[str, str]  # every record file read, with its SHA-256


def _safe_id(ref: str, forbidden: Sequence[str]) -> str:
    """An identifier `public_ref` already made, made safe the same way."""
    parts = ref.split("/")
    out = []
    for part in parts:
        if any(re.search(p, part) for p in forbidden):
            label, colon, rest = part.partition(":") if ":" in part else ("", "", part)
            out.append(f"{label}{colon}run#{hashlib.sha256(rest.encode()).hexdigest()[:8]}")
        else:
            out.append(part)
    return "/".join(out)


def safe_ref(path: str | Path, forbidden: Sequence[str] = ()) -> str:
    """The public identifier of a record, with every part of it that would show a forbidden
    name (a hidden world id in a run folder's name) replaced by a digest of that part. The
    page is still checked against the same patterns after binding."""
    return _safe_id(public_ref(path), forbidden)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _records_table(records: dict[str, list[Path]], meanings: dict[str, str],
                   read: dict[str, str], forbidden: Sequence[str] = ()) -> str:  # fmt: skip
    rows = ["| Record | What it is | Identifier | SHA-256 (first 12) |", "|---|---|---|---|"]
    for name, files in records.items():
        if name == DERIVED_RECORD:
            continue
        for p in files:
            digest = read.get(str(p)) or (_sha256(p) if p.is_file() else "")
            rows.append(
                f"| {_cell(name)} | {_cell(meanings.get(name, ''))} | `{safe_ref(p, forbidden)}` | "
                f"`{digest[:12] or 'missing'}` |"
            )
    if len(rows) == 2:
        rows.append("| none | this draft is bound to no record yet | | |")
    return "\n".join(rows)


def _numbers_table(values: dict[str, dict[str, Any]], unbound: dict[str, str],
                   specs: dict[str, FieldSpec], derived: dict[str, Any],
                   forbidden: Sequence[str] = ()) -> str:  # fmt: skip
    rows = ["| Number | Value | Meaning | Source |", "|---|---|---|---|"]
    for name, spec in specs.items():
        if name not in values and name not in unbound:
            continue
        if name in values:
            value = values[name]["value"]
            if spec.record == DERIVED_RECORD:
                key = (spec.field or "").split(".")[0]
                entry = derived.get("values", {}).get(key, {})
                sources = ", ".join(
                    f"`{_safe_id(i['id'], forbidden)}` (`{i['sha256'][:12]}`)"
                    for i in entry.get("inputs", [])
                )
                source = f"{entry.get('how', 'derived')} from {sources or 'the records'}"
            else:
                files = values[name].get("files") or []
                where = spec.field or spec.count_true or "rows"
                source = f"{where} of " + ", ".join(
                    f"`{safe_ref(f['path'], forbidden)}` (`{f['sha256'][:12]}`)" for f in files
                )
        else:
            value, source = "unbound", unbound[name]
            source = "not in the records of this build" if "/" in source else source
        rows.append(f"| {_cell(name)} | {_cell(value)} | {_cell(spec.meaning)} | {_cell(source)} |")
    return "\n".join(rows)


def _clips_table(records: dict[str, list[Path]], forbidden: Sequence[str] = ()) -> str:
    rows = [
        "| Beat | What the clip shows | Run | Episode | Outcome | Video | SHA-256 (first 12) |",
        "|---|---|---|---|---|---|---|",
    ]
    files = records.get("film") or []
    film = json.loads(files[0].read_text(encoding="utf-8")) if files and files[0].is_file() else {}
    for beat in film.get("beats", []):
        for shot in beat.get("shots", []):
            for clip in shot.get("clips", []):
                run = safe_ref(clip["run"], forbidden) if clip.get("run") else "placeholder: no run"
                state = clip.get("rerender") or {}
                video = "verified re-render" if state.get("used") else "original"
                if clip.get("placeholder"):
                    video = "placeholder"
                digest = str(clip.get("original_sha256") or clip.get("video_sha256") or "")
                rows.append(
                    f"| {_cell(beat.get('title', beat.get('id')))} | {_cell(clip.get('label', ''))}"
                    f" | `{run}` | `{_cell(clip.get('episode', ''))}` | "
                    f"{_cell(clip.get('outcome') or '')} | {video} | `{digest[:12]}` |"
                )
    if len(rows) == 2:
        rows.append("| none | this draft names no film yet | | | | | |")
    return "\n".join(rows)


def bind_page(
    text: str,
    fields: dict[str, FieldSpec],
    records: dict[str, list[Path]],
    derived: dict[str, Any],
    meanings: dict[str, str] | None = None,
    derived_path: Path | None = None,
    forbidden: Sequence[str] = (),
) -> PageBinding:
    """Bind every placeholder of the page (outside editor's notes); a name neither
    `fields` nor `GENERATED` defines is an error."""
    body = _COMMENT.sub("", text)
    names: list[str] = []
    for name in PLACEHOLDER.findall(body):
        if name not in names:
            names.append(name)
    undefined = [n for n in names if n not in fields and n not in GENERATED]
    if undefined:
        raise FieldError(
            "the page uses placeholder(s) no field defines: "
            + ", ".join("{" + n + "}" for n in undefined)
        )
    all_records = dict(records)
    if derived_path is not None:
        all_records[DERIVED_RECORD] = [derived_path]
    values: dict[str, dict[str, Any]] = {}
    unbound: dict[str, str] = {}
    read: dict[str, str] = dict(derived.get("files_read", {}))
    for name in names:
        if name in GENERATED:
            continue
        spec = fields[name]
        if spec.record == DERIVED_RECORD:
            key = (spec.field or "").split(".")[0]
            if key in derived.get("unavailable", {}):
                unbound[name] = derived["unavailable"][key]
                continue
            if key not in derived.get("values", {}):
                unbound[name] = f"no derived value {key}"
                continue
        files = all_records.get(spec.record)
        if files is None:
            unbound[name] = f"this cut provides no record {spec.record}"
            continue
        reading = spec
        if spec.record == DERIVED_RECORD:  # derived.json keeps its values under `values`
            reading = replace(spec, field=f"values.{spec.field}")
        value, why = field_value(reading, files, read)
        if why is not None:
            unbound[name] = why
            continue
        values[name] = {
            "value": value,
            **spec.describe(),
            "files": [{"path": str(p), "sha256": read[str(p)]} for p in files],
        }
    tables = {
        "provenance_records": _records_table(records, meanings or {}, read, forbidden),
        "provenance_numbers": _numbers_table(values, unbound, fields, derived, forbidden),
        "provenance_clips": _clips_table(records, forbidden),
    }

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in tables:
            return tables[name]
        if name in values:
            return str(values[name]["value"])
        return f"**[unbound: {name}]**"

    bound = PLACEHOLDER.sub(sub, body)
    for pattern in forbidden:
        hit = re.search(pattern, bound)
        if hit:
            raise FieldError(f"the bound page shows {hit.group(0)!r}, which it must never show")
    open_gaps = gaps(bound)
    if unbound or open_gaps:
        note = (
            f"> **Draft.** {len(unbound)} number(s) on this page are not yet bound to a "
            f"recorded run and show as **[unbound: name]**; {len(open_gaps)} passage(s) "
            "marked [gap: ...] or [one of: ...] wait for the runs that decide them."
        )
        bound = _after_title(bound, note)
    return PageBinding(bound, values, unbound, open_gaps, read)


def _after_title(text: str, note: str) -> str:
    """The note as the first paragraph after the page's title (after front matter)."""
    front = _FRONT.match(text)
    head, rest = (text[: front.end()], text[front.end() :]) if front else ("", text)
    title = re.search(r"^# .*\n", rest, flags=re.MULTILINE)
    if title is None:
        return head + note + "\n\n" + rest
    return head + rest[: title.end()] + "\n" + note + "\n" + rest[title.end() :]


def bind_record(
    page: Path, fields_path: Path, records_path: Path | None, out: Path
) -> dict[str, Any]:
    """Write `out/derived.json`, `out/page/index.md` and `out/bindings.json`; the report."""
    data = json.loads(fields_path.read_text(encoding="utf-8"))
    fields = load_fields(fields_path)
    records = load_records(records_path) if records_path is not None else {}
    out.mkdir(parents=True, exist_ok=True)
    derived = derive(data.get("derived") or {}, records, data.get("action_words"))
    derived_path = out / "derived.json"
    derived_path.write_text(json.dumps(derived, indent=1, ensure_ascii=False) + "\n")
    # the records table names each record in plain words (`titles`), its place in `records`
    meanings = {str(k): str(v) for k, v in (data.get("titles") or {}).items()}
    binding = bind_page(
        page.read_text(encoding="utf-8"),
        fields,
        records,
        derived,
        meanings,
        derived_path,
        [str(p) for p in data.get("forbidden") or []],
    )
    (out / "page").mkdir(exist_ok=True)
    bound = out / "page" / "index.md"
    bound.write_text(binding.text, encoding="utf-8")
    report = {
        "format": RECORD_BINDINGS_FORMAT,
        "page": {"path": str(page), "sha256": _sha256(page)},
        "fields": {"path": str(fields_path), "sha256": _sha256(fields_path)},
        "records": (
            {"path": str(records_path), "sha256": _sha256(records_path)}
            if records_path is not None
            else None
        ),
        "derived": {"path": str(derived_path), "sha256": _sha256(derived_path)},
        "bound_page": {"path": str(bound), "sha256": _sha256(bound)},
        "values": binding.values,
        "unbound": binding.unbound,
        "gaps": binding.gaps,
        "files_read": binding.files,
    }
    (out / "bindings.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


# ---- the command line ----------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    for name, value in report["values"].items():
        print(f"bound   {{{name}}} = {value['value']}")
    for name, why in report["unbound"].items():
        print(f"unbound {{{name}}}: {why}", file=sys.stderr)
    for gap in report["gaps"]:
        print(f"gap     {gap[:100]}", file=sys.stderr)
    print(
        f"wrote {report['bound_page']['path']}: {len(report['values'])} bound, "
        f"{len(report['unbound'])} unbound, {len(report['gaps'])} gap(s)"
    )


def _cmd_bind(args: argparse.Namespace) -> int:
    records = args.records.expanduser() if args.records else None
    report = bind_record(args.page, args.fields, records, args.out.expanduser())
    _print_report(report)
    return 4 if args.final and (report["unbound"] or report["gaps"]) else 0


def _cmd_derive(args: argparse.Namespace) -> int:
    data = json.loads(args.fields.read_text(encoding="utf-8"))
    records = load_records(args.records.expanduser())
    derived = derive(data.get("derived") or {}, records, data.get("action_words"))
    out: Path = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derived, indent=1, ensure_ascii=False) + "\n")
    for name, why in derived["unavailable"].items():
        print(f"unavailable {name}: {why}", file=sys.stderr)
    print(f"wrote {out}: {len(derived['values'])} derived, {len(derived['unavailable'])} not")
    return 0


def _cmd_site(args: argparse.Namespace) -> int:
    from fleetlab.film import website

    out: Path = args.out.expanduser()
    bundle: Path = args.bundle.expanduser()
    spec = json.loads(bundle.read_text(encoding="utf-8"))
    page = Path(str(spec.get("record_source") or SITE_ROOT / "record" / "record.md"))
    page = page if page.is_absolute() else bundle.parent / page
    records = args.records.expanduser() if args.records else None
    report = bind_record(page, args.fields, records, out / "record")
    _print_report(report)
    if args.final and (report["unbound"] or report["gaps"]):
        print("refused: a final site needs every number bound and no gap", file=sys.stderr)
        return 4
    film = Path(str(spec.get("film", "film"))).expanduser()
    spec["film"] = str(film if film.is_absolute() else bundle.parent / film)
    for item in spec.get("artifacts") or []:
        path = Path(str(item["path"])).expanduser()
        item["path"] = str(path if path.is_absolute() else bundle.parent / path)
    spec["record"] = "page"
    spec.pop("record_source", None)
    site_json = out / "record" / "site.json"
    site_json.write_text(json.dumps(spec, indent=1) + "\n")
    site_out = args.site_out.expanduser() if args.site_out else out / "site"
    result = website.build_site(site_json, site_out)
    print(f"wrote {result.out} ({result.files} files, {result.bytes / 1e6:.1f} MB)")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 1 if result.errors else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetlab.film.record", description=__doc__.splitlines()[0]
    )
    default_fields = SITE_ROOT / "record" / "fields.json"
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bind", help="bind the record page's numbers to the cut's records")
    p.add_argument("page", type=Path)
    p.add_argument("--records", type=Path, help="the cut's fleetlab.film-records/1 file")
    p.add_argument("--fields", type=Path, default=default_fields)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--final", action="store_true", help="fail on an unbound number or a gap")
    p.set_defaults(run=_cmd_bind)
    p = sub.add_parser("derive", help="only the derived values, as one JSON record")
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--fields", type=Path, default=default_fields)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_cmd_derive)
    p = sub.add_parser("site", help="bind the page, then build the bundle's site with it")
    p.add_argument("bundle", type=Path)
    p.add_argument("--records", type=Path)
    p.add_argument("--fields", type=Path, default=default_fields)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--final", action="store_true", help="refuse an unbound number or a gap")
    p.add_argument("--site-out", type=Path, help="where the site goes (default: OUT/site)")
    p.set_defaults(run=_cmd_site)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
