"""Before-and-after pairs chosen by rule, never by looking.

`python -m fleetlab.film pairs --before RUN --after RUN` matches the episodes of two runs on
the same starts and reports, by start:

- the counts: both delivered, rescued (failed before, delivered after), broken (the
  reverse), both failed, and starts only one side ran;
- the first rescued pair and the first broken pair, by start (base task, then initial
  state), as manifest clips ready to paste into a `pair` shot.

A run is a jobs run (`episodes.jsonl`, one `JobEpisode` a line, its start identity under
`start`; `fleetlab.jobs.tables`) or a spike batch (`results.jsonl` with `job`, `variant`,
`start`, the identity fields and episode folders `<job>/<variant>/sNN`);
`--before-where`/`--after-where` (`column=value`, repeatable) select rows, for example
`variant=naive` against `variant=release` in the same batch. Starts match on the whole
start identity, all four fields (`base_task`, `init_state`, `scene_seed`, `noise_seed`); an
episode that lacks one stops the command, since two runs with other seeds are other starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleetlab.film.sources import sha256

__all__ = ["Episode", "PairReport", "load_side", "pair"]

IDENTITY = ("base_task", "init_state", "scene_seed", "noise_seed")


@dataclass(frozen=True)
class Episode:
    run: Path
    episode: str
    delivered: bool
    identity: dict[str, Any]


@dataclass(frozen=True)
class PairReport:
    keys: tuple[str, ...]
    both: int
    rescued: list[tuple[Episode, Episode]]
    broken: list[tuple[Episode, Episode]]
    neither: int
    unmatched_before: int
    unmatched_after: int
    files: dict[str, str]

    def summary(self) -> dict[str, Any]:
        def clip(e: Episode, label: str) -> dict[str, Any]:
            return {"run": str(e.run), "episode": e.episode, "label": label}

        def first(pairs: list[tuple[Episode, Episode]]) -> list[dict[str, Any]] | None:
            if not pairs:
                return None
            before, after = pairs[0]
            return [clip(before, "before"), clip(after, "after")]

        return {
            "matched_on": list(self.keys),
            "starts": self.both + len(self.rescued) + len(self.broken) + self.neither,
            "both_delivered": self.both,
            "rescued": len(self.rescued),
            "broken": len(self.broken),
            "neither": self.neither,
            "only_before": self.unmatched_before,
            "only_after": self.unmatched_after,
            "first_rescued": first(self.rescued),
            "first_broken": first(self.broken),
            "rule": "the first start, by base task and initial state, where the episodes differ",
            "files": self.files,
        }


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_where(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"{item!r}: column=value expected")
        out[key] = _value(value)
    return out


def _identity(fields: dict[str, Any], where: str) -> dict[str, Any]:
    missing = [k for k in IDENTITY if fields.get(k) is None]
    if missing:
        raise ValueError(f"{where}: no {', '.join(missing)} in its start identity")
    return {k: fields[k] for k in IDENTITY}


def load_side(run: Path, where: dict[str, Any]) -> tuple[list[Episode], dict[str, str]]:
    run = run.expanduser()
    files: dict[str, str] = {}
    episodes: list[Episode] = []
    jobs = run / "episodes.jsonl"  # the jobs runner's JobEpisode records
    results = run / "results.jsonl"  # the spike's batch results
    if jobs.is_file():
        files[str(jobs)] = sha256(jobs)
        for line in jobs.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(row.get(k) == v for k, v in where.items()):
                continue
            name = f"{run}: episode {row.get('episode_id')}"
            identity = _identity(dict(row.get("start") or {}), name)
            episodes.append(
                Episode(run, str(row["episode_id"]), bool(row.get("delivered")), identity)
            )
        return episodes, files
    if not results.is_file():
        raise ValueError(f"{run}: neither a jobs run with episodes.jsonl nor a spike batch")
    files[str(results)] = sha256(results)
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(row.get(k) == v for k, v in where.items()):
            continue
        folder = f"{row['job']}/{row['variant']}/s{int(row['start']):02d}"
        identity = _identity(row, f"{run}: {folder}")
        episodes.append(Episode(run, folder, bool(row.get("delivered")), identity))
    return episodes, files


def pair(before: list[Episode], after: list[Episode], files: dict[str, str]) -> PairReport:
    if not before or not after:
        raise ValueError("a side has no episodes")
    keys = IDENTITY

    def index(side: list[Episode], name: str) -> dict[tuple[Any, ...], Episode]:
        out: dict[tuple[Any, ...], Episode] = {}
        for e in side:
            key = tuple(e.identity[k] for k in keys)
            if key in out:
                raise ValueError(f"the {name} side has two episodes on the start {key}")
            out[key] = e
        return out

    b, a = index(before, "before"), index(after, "after")
    both = neither = 0
    rescued: list[tuple[Episode, Episode]] = []
    broken: list[tuple[Episode, Episode]] = []
    for key in sorted(set(b) & set(a)):
        x, y = b[key], a[key]
        if x.delivered and y.delivered:
            both += 1
        elif not x.delivered and y.delivered:
            rescued.append((x, y))
        elif x.delivered and not y.delivered:
            broken.append((x, y))
        else:
            neither += 1
    return PairReport(
        keys, both, rescued, broken, neither, len(set(b) - set(a)), len(set(a) - set(b)), files
    )
