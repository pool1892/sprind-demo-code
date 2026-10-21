"""A world's page: a world root of the version-1 generator on one page, both runs merged.

    worlds/<world id>/index.md      the page, with its PNG figures

A world root (`fleetlab.worlds.layout.WorldPaths`) holds the world files under
`files/`, the engine run of the observed window (`runs/<id>-observed`), the run
of the experiment and scoring windows (`runs/<id>-later`), and the runner's
record (`runner.json`). The page reads them as follows.

**Visible side**, always: the world's shape from `files/spec.json` (sites,
robots per site, days, windows, episodes per robot and day, the number of
tasks; only these fields, never its fault schedule or patterns); each run's
state and its visible `log/` tree through `fleetlab.log.LogReader` (episodes,
health, change and maintenance logs, events); the runner's steps and its
`ready` flag. Health readings come from each run's `log/health.parquet`, or,
for a run whose records are not written yet, from `files/health.jsonl` cut to
that run's days.

**Truth side**, only with `--truth` and only for worlds registered as
development material: the fault schedule and the patterns (the whole of
`files/spec.json`), the roles, recipe and expectations of `generator.json`,
the detectability list (`detectability.json`), the runner's summary, the
runs' `truth/identities` and `truth/verdicts`, and the expected success of
each episode recomputed from the calibrated curves (`fleetlab.worlds.estimate`).

Pure functions of the files, like the run pages: rows are sorted, figures are
written without a date or software chunk, and the only clock value is the
generated-at line.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fleetlab.dashboard.data import Rate, Source, check_run_id, load_source, rates_by
from fleetlab.dashboard.figures import (
    GRID,
    INK,
    INK_SECONDARY,
    REFERENCE,
    _errorbars,
    _save,
    _style,
)
from fleetlab.dashboard.pages import link, table
from fleetlab.log import LogReader
from fleetlab.log.runfiles import read_identities, read_table, table_path
from fleetlab.worlds.layout import RUN_WINDOWS, WorldPaths

if TYPE_CHECKING:
    from fleetlab.contracts.faults import FaultSpec
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.contracts.world import WorldSpec

__all__ = [
    "WORLD_TRUTH_NOTE",
    "FaultComparison",
    "Shape",
    "WorldRun",
    "WorldSource",
    "WorldTruth",
    "load_world",
    "load_world_truth",
    "write_world_page",
]

WINDOWS = ("observed", "experiment", "scoring")
# The reference palette's categorical slots in their fixed order (dataviz skill): one per
# robot of a site, never cycled; a ninth robot of one site is drawn in grey.
CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
WORLD_TRUTH_NOTE = (
    "> **Truth: the hidden side.** Everything marked *truth* on this page comes from the "
    "world's generator files (the fault schedule and patterns of `files/spec.json`, "
    "`generator.json`, `detectability.json`), the runner's summary and the runs' `truth/` "
    "tables, which the agents never see. It is shown because the page was built with "
    "`--truth`, which the dashboard allows only for worlds registered as development "
    "material in `dashboard/worlds.json`."
)


# ---------------------------------------------------------------------------- the visible side


@dataclass(frozen=True)
class Shape:
    """A world's size and windows, from the visible fields of `files/spec.json`."""

    sites: int
    robots_per_site: int
    days: int
    episodes_per_robot_day: int
    tasks: int
    windows: tuple[tuple[str, int, int], ...]  # (name, first day, last day), in order

    @staticmethod
    def of(spec: dict[str, Any]) -> Shape:
        windows = spec["windows"]
        return Shape(
            sites=int(spec["sites"]),
            robots_per_site=int(spec["robots_per_site"]),
            days=int(spec["days"]),
            episodes_per_robot_day=int(spec["episodes_per_robot_day"]),
            tasks=len(spec.get("tasks", [])),
            windows=tuple((w, int(windows[w]["first"]), int(windows[w]["last"])) for w in WINDOWS),
        )

    def window_of(self, day: int) -> str:
        for name, first, last in self.windows:
            if first <= day <= last:
                return name
        return "outside the windows"

    def days_of(self, run: str) -> list[int]:
        """The days of a run (`observed` or `later`), from its windows."""
        names = RUN_WINDOWS[run]
        return [
            d for name, first, last in self.windows if name in names for d in range(first, last + 1)
        ]

    def site_of(self, robot_id: str) -> str:
        """The site of a robot by the generator's numbering (robots numbered site by site);
        used only for a robot without episodes."""
        digits = robot_id.lstrip("r")
        if not digits.isdigit():
            return "unknown site"
        return f"s{(int(digits) - 1) // self.robots_per_site + 1}"

    def planned(self, first: int, last: int) -> int:
        robots = self.sites * self.robots_per_site
        return robots * self.episodes_per_robot_day * (last - first + 1)


@dataclass(frozen=True)
class WorldRun:
    """One of a world's two runs, read as the run pages read it, or why it could not be."""

    name: str  # "observed" or "later"
    directory: Path
    source: Source | None
    note: str = ""  # why there is no source

    @property
    def run_id(self) -> str:
        return self.directory.name


@dataclass(frozen=True)
class WorldSource:
    """A world root, visible side."""

    world_id: str
    root: Path
    shape: Shape
    runs: tuple[WorldRun, ...]
    runner: dict[str, Any]  # runner.json; its summary is shown on the truth side only
    health_jsonl: Path

    @property
    def ready(self) -> bool:
        from fleetlab.worlds.runner import is_ready

        return is_ready(self.runner, WorldPaths.from_root(self.root))

    @property
    def episodes(self) -> list[dict[str, Any]]:
        """Both runs' episodes, each with its window and run added."""
        out = []
        for run in self.runs:
            if run.source is None:
                continue
            for e in run.source.episodes:
                out.append({**e, "window": self.shape.window_of(int(e["day"])), "run": run.name})
        return out

    @property
    def run_world_ids(self) -> tuple[str, ...]:
        return tuple(sorted({w for r in self.runs if r.source for w in r.source.world_ids}))

    @property
    def overall(self) -> Rate:
        return Rate.of(e["outcome"] == "success" for e in self.episodes)


def load_world(root: Path | str) -> WorldSource:
    """Read the visible side of a world root: its shape, its runs and the runner's record."""
    path = Path(root).expanduser().resolve()
    spec_file = path / "files" / "spec.json"
    if not spec_file.is_file():
        raise FileNotFoundError(
            f"{path} has no files/spec.json: not a world root; "
            f"identity records are {spec_file} and {path / 'runner.json'}"
        )
    paths = WorldPaths.from_root(path)
    spec = json.loads(spec_file.read_text())
    world_id = check_run_id(paths.world_id)
    runs = []
    for name in RUN_WINDOWS:
        directory = paths.run(name)
        if not directory.exists():
            runs.append(WorldRun(name, directory, None, "not started"))
            continue
        if not (directory / "log").is_dir():
            runs.append(WorldRun(name, directory, None, "started, no log/ yet"))
            continue
        try:
            runs.append(WorldRun(name, directory, load_source(directory)))
        except Exception as error:  # a run being written: say so, keep the page
            runs.append(WorldRun(name, directory, None, f"not readable: {type(error).__name__}"))
    try:
        runner = paths.read_record()
    except (OSError, ValueError) as error:
        runner = {"unreadable": f"{type(error).__name__}"}
    return WorldSource(
        world_id=world_id,
        root=path,
        shape=Shape.of(spec),
        runs=tuple(runs),
        runner=runner,
        health_jsonl=paths.files / "health.jsonl",
    )


def _log_rows(run: WorldRun, sql: str) -> list[dict[str, Any]]:
    if run.source is None:
        return []
    with LogReader(run.source.root) as reader:
        rows: list[dict[str, Any]] = reader.query(sql).to_pylist()
    return rows


@dataclass(frozen=True)
class Health:
    rows: list[dict[str, Any]]  # day, robot_id, component, metric, value
    notes: list[str]


def read_health(world: WorldSource) -> Health:
    """Health readings of each run's days: the run's table, or the world's file cut to them."""
    rows: list[dict[str, Any]] = []
    notes = []
    fallback: list[dict[str, Any]] | None = None
    for run in world.runs:
        if run.source is None:
            continue
        mine = _log_rows(run, "SELECT day, robot_id, component, metric, value FROM health")
        days = world.shape.days_of(run.name)
        if mine:
            notes.append(f"days {_days(days)} from the {run.name} run's `log/health.parquet`")
        else:
            if fallback is None:
                fallback = []
                if world.health_jsonl.is_file():
                    fallback = [
                        json.loads(line)
                        for line in world.health_jsonl.read_text().splitlines()
                        if line.strip()
                    ]
            keep = ("day", "robot_id", "component", "metric", "value")  # the visible columns
            mine = [{k: r[k] for k in keep} for r in fallback if int(r["day"]) in days]
            where = "`files/health.jsonl`" if mine else "nowhere yet"
            notes.append(
                f"days {_days(days)} from {where}: the {run.name} run's records are not written"
            )
        rows += mine
    rows.sort(key=lambda r: (r["component"], r["metric"], r["robot_id"], int(r["day"])))
    return Health(rows, notes)


def _days(days: Sequence[int]) -> str:
    if not days:
        return "none"
    return f"{min(days)} to {max(days)}" if len(days) > 1 else str(days[0])


# ---------------------------------------------------------------------------- the truth side


@dataclass(frozen=True)
class FaultComparison:
    """Logged success of the episodes carrying a fault, beside the generator's expectation."""

    label: str  # a fault id, "no fault" or "all"
    role: str
    kind: str
    run: str
    measured: Rate
    expected: float  # share, the mean of the episodes' expected logged success


@dataclass(frozen=True)
class WorldTruth:
    """The hidden side of a world, as far as its page shows it."""

    spec: WorldSpec
    generator: dict[str, Any]
    detect: dict[str, Any]
    summary: dict[str, Any] | None
    true_success: dict[str, bool]  # by episode, both runs
    comparisons: list[FaultComparison] = field(default_factory=list)
    comparison_note: str = ""  # what the comparison left out, or why it is missing

    @property
    def roles(self) -> dict[str, str]:
        roles: dict[str, str] = self.generator.get("roles", {})
        return roles


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _verdicts(root: Path) -> dict[str, bool]:
    if not table_path(root, "verdicts").is_file():
        return {}
    rows = read_table(root, "verdicts").to_pylist()
    return {str(r["episode_id"]): bool(r["true_success"]) for r in rows}


def load_world_truth(world: WorldSource) -> WorldTruth:
    """Read the hidden side of a world root. Never for an agent."""
    from fleetlab.contracts.world import WorldSpec

    files = world.root / "files"
    spec = WorldSpec.model_validate_json((files / "spec.json").read_text())
    generator = _json(files / "generator.json")
    detect = _json(files / "detectability.json")
    true_success: dict[str, bool] = {}
    identities: dict[str, EpisodeIdentity] = {}
    for run in world.runs:
        if run.source is None or not run.source.is_run:
            continue
        true_success.update(_verdicts(run.source.root))
        if table_path(run.source.root, "identities").is_file():
            identities.update({i.episode_id: i for i in read_identities(run.source.root)})
    summary = world.runner.get("summary")
    truth = WorldTruth(
        spec=spec,
        generator=generator,
        detect=detect,
        summary=summary if isinstance(summary, dict) else None,
        true_success=true_success,
    )
    comparisons, note = _comparisons(world, truth, identities)
    return replace(truth, comparisons=comparisons, comparison_note=note)


def _context(truth: WorldTruth) -> Any:
    """The generator's estimate context, rebuilt from the world's files."""
    from fleetlab.worlds import curves
    from fleetlab.worlds.compose import Composition
    from fleetlab.worlds.recipe import WorldRecipe
    from fleetlab.worlds.world import context

    g = truth.generator
    composition = Composition(
        recipe=WorldRecipe.model_validate(g["recipe"]),
        spec=truth.spec,
        attempt=int(g.get("attempt", 0)),
        roles=dict(g.get("roles", {})),
        small=frozenset(g.get("small_causes", [])),
        assumed_joint=bool(g.get("assumed_joint", False)),
        expected=dict(g.get("expected", {})),
    )
    return context(composition, curves.load())


def _comparisons(
    world: WorldSource,
    truth: WorldTruth,
    identities: dict[str, EpisodeIdentity],
) -> tuple[list[FaultComparison], str]:
    """Per fault and run: the logged success of the episodes carrying it against the mean of
    their expected logged success (`fleetlab.worlds.estimate.probability`)."""
    from fleetlab.worlds import curves, estimate

    if not truth.generator.get("recipe"):
        return [], "not computed: files/generator.json is missing"
    try:
        ctx = _context(truth)
    except Exception as error:  # the page stays, the comparison says why it is missing
        return [], f"not computed: the generator's context could not be rebuilt ({error})"
    notes = []
    if list(truth.generator.get("curves", [])) != list(curves.load().sources):
        notes.append(
            "the curves installed now ("
            + ", ".join(curves.load().sources)
            + ") are not the ones the world was generated with ("
            + ", ".join(truth.generator.get("curves", []))
            + "), so the expectation differs from the generator's"
        )
    outcome = {e["episode_id"]: (e["outcome"] == "success", e["run"]) for e in world.episodes}
    rows: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)
    left_out = 0
    failed: set[str] = set()
    for episode_id, (success, run) in sorted(outcome.items()):
        identity = identities.get(episode_id)
        if identity is None or identity.world_id != world.world_id:
            left_out += 1
            continue
        try:
            p = estimate.probability(identity.faults, identity.suite, ctx, logged=True)
        except (KeyError, ValueError) as error:
            failed.add(str(error))
            left_out += 1
            continue
        labels = {f.fault_id for f in identity.faults} or {"no fault"}
        for label in (*labels, "all"):
            rows[(label, run)].append((success, p))
    if left_out:
        notes.append(
            f"{left_out} episodes of the runs are left out: no identity of world "
            f"{world.world_id}, or no expectation"
            + (f" ({'; '.join(sorted(failed)[:3])})" if failed else "")
        )
    kinds = {f.fault_id: f.kind for f in truth.spec.fault_schedule}
    order = [f.fault_id for f in truth.spec.fault_schedule] + ["no fault", "all"]
    out = []
    for label in order:
        for run in RUN_WINDOWS:
            values = rows.get((label, run))
            if not values:
                continue
            out.append(
                FaultComparison(
                    label=label,
                    role=truth.roles.get(label, ""),
                    kind=kinds.get(label, ""),
                    run=run,
                    measured=Rate.of(s for s, _ in values),
                    expected=sum(p for _, p in values) / len(values),
                )
            )
    return out, "; ".join(notes)


# ---------------------------------------------------------------------------- figures


def _boundaries(ax: Axes, shape: Shape) -> None:
    for _, first, _ in shape.windows[1:]:
        ax.axvline(first - 0.5, color=REFERENCE, linewidth=0.8, linestyle=(0, (1, 2)))


def _panels(count: int, height: float) -> tuple[Figure, list[Axes]]:
    count = max(1, count)
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    fig = Figure(figsize=(3.6 * columns, height * rows), layout="constrained")
    grid = fig.subplots(rows, columns, squeeze=False, sharey=True)
    axes: list[Axes] = list(np.asarray(grid).ravel())
    for ax in axes[count:]:
        ax.set_visible(False)
    return fig, axes[:count]


def site_day_figure(
    path: Path, episodes: Sequence[dict[str, Any]], shape: Shape, overall: Rate
) -> None:
    """One panel per site: success by day with Wilson's interval, the fleet's rate dashed."""
    sites = sorted({str(e["site_id"]) for e in episodes})
    fig, axes = _panels(len(sites), 2.6)
    for ax, site in zip(axes, sites, strict=True):
        _boundaries(ax, shape)
        if overall.n:
            ax.axhline(overall.rate, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        points = rates_by([e for e in episodes if e["site_id"] == site], "day")
        _errorbars(ax, [float(d) for d, _ in points], [r for _, r in points])
        ax.set_xticks(range(shape.days), [str(d) for d in range(shape.days)])
        ax.set_xlim(-0.6, shape.days - 0.4)
        _style(ax, site, "day")
    _save(fig, path)


def _plain_style(ax: Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("day", fontsize=9, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def health_figure(
    path: Path,
    rows: Sequence[dict[str, Any]],
    sites: dict[str, str],
    shape: Shape,
    title: str,
) -> None:
    """One panel per site, one line per robot (in the palette's fixed order), shared y."""
    by_robot: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_robot[str(r["robot_id"])].append((int(r["day"]), float(r["value"])))
    site_ids = sorted({sites.get(robot, "unknown site") for robot in by_robot})
    fig, axes = _panels(len(site_ids), 3.0)
    for ax, site in zip(axes, site_ids, strict=True):
        _boundaries(ax, shape)
        robots = sorted(r for r in by_robot if sites.get(r, "unknown site") == site)
        for slot, robot in enumerate(robots):
            points = sorted(by_robot[robot])
            color = CATEGORICAL[slot] if slot < len(CATEGORICAL) else REFERENCE
            ax.plot(
                [d for d, _ in points],
                [v for _, v in points],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=robot,
            )
        ax.set_xticks(range(shape.days), [str(d) for d in range(shape.days)])
        ax.set_xlim(-0.6, shape.days - 0.4)
        _plain_style(ax, site)
        if ax is axes[0]:
            ax.set_ylabel(title.split(": ")[-1], fontsize=9, color=INK_SECONDARY)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(4, max(1, len(robots))),
            fontsize=7,
            frameon=False,
            labelcolor=INK_SECONDARY,
        )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    _save(fig, path)


# ---------------------------------------------------------------------------- the page


def _reading(value: float, sign: bool = False) -> str:
    """A health reading: four significant digits, whole numbers from a thousand up."""
    if abs(value) >= 1000:
        return f"{value:+.0f}" if sign else f"{value:.0f}"
    return f"{value:+.4g}" if sign else f"{value:.4g}"


def _yes(value: Any) -> str:
    return "yes" if value else "no"


def _number(value: Any) -> str:
    if isinstance(value, bool):
        return _yes(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list | dict):
        return json.dumps(value, separators=(", ", ": "))
    return "" if value is None else str(value)


def _percent(share: float) -> str:
    return "" if math.isnan(share) else f"{100 * share:.1f}%"


def _window_rates(episodes: Sequence[dict[str, Any]], key: str) -> dict[Any, dict[str, Rate]]:
    out: dict[Any, dict[str, Rate]] = defaultdict(dict)
    for name in WINDOWS:
        for value, rate in rates_by([e for e in episodes if e["window"] == name], key):
            out[value][name] = rate
    return out


def _by(episodes: Sequence[dict[str, Any]], key: str, label: str, with_site: bool = False) -> str:
    windows = _window_rates(episodes, key)
    headers = [label, *(["site"] if with_site else []), "episodes", "successes", "success [95%]"]
    headers += [f"{w} [95%]" for w in WINDOWS]
    rows = []
    for value, rate in rates_by(episodes, key):
        row: list[Any] = [value]
        if with_site:
            row.append(next(e["site_id"] for e in episodes if e[key] == value))
        row += [rate.n, rate.successes, rate.text()]
        row += [windows[value][w].text() if w in windows[value] else "" for w in WINDOWS]
        rows.append(row)
    return table(headers, rows)


def _site_day(episodes: Sequence[dict[str, Any]], shape: Shape) -> str:
    sites = sorted({str(e["site_id"]) for e in episodes})
    counts: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for e in episodes:
        counts[(str(e["site_id"]), int(e["day"]))].append(e["outcome"] == "success")
    rows = []
    for site in sites:
        cells = []
        for day in range(shape.days):
            values = counts.get((site, day), [])
            cells.append(f"{sum(values)}/{len(values)}" if values else "")
        rows.append([site, *cells])
    return table(["site", *[f"day {d}" for d in range(shape.days)]], rows)


def _runs_section(world: WorldSource, run_links: dict[str, str]) -> list[str]:
    rows: list[list[Any]] = []
    for run in world.runs:
        days = world.shape.days_of(run.name)
        windows = ", ".join(RUN_WINDOWS[run.name])
        name = run_links.get(run.name) or run.run_id
        s = run.source
        if s is None:
            rows.append([name, windows, _days(days), run.note, "", "", ""])
            continue
        chain = ""
        if s.record is not None:
            chain = {None: "not replayed", True: "ok", False: "**broken**"}[s.record.chain_ok]
        rows.append([name, windows, _days(days), s.state, len(s.episodes), s.overall.text(), chain])
    out = [
        table(
            ["run", "windows", "days", "state", "episodes", "success [95%]", "hash chains"], rows
        ),
        "",
    ]
    steps = world.runner.get("steps") or {}
    if isinstance(steps, dict) and steps:
        step_rows = []
        for step, record in steps.items():
            record = record if isinstance(record, dict) else {}
            attempts = [a for a in record.get("attempts", []) if isinstance(a, dict)]
            seconds = sum(float(a.get("seconds") or 0.0) for a in attempts)
            step_rows.append(
                [
                    step,
                    _yes(record.get("done")),
                    len(attempts),
                    f"{seconds:.1f}" if attempts else "",
                ]
            )
        out += [
            "The runner's steps (`runner.json`):",
            "",
            table(["step", "done", "attempts", "seconds"], step_rows),
            "",
        ]
    elif "unreadable" in world.runner:
        out += [f"`runner.json` could not be read ({world.runner['unreadable']}).", ""]
    else:
        out += ["No `runner.json` yet.", ""]
    return out


def _records(
    world: WorldSource, table_name: str, columns: list[str], headers: list[str], order: str
) -> list[str]:
    observed = next((r for r in world.runs if r.name == "observed"), None)
    later = next((r for r in world.runs if r.name == "later"), None)
    sql = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY {order}"
    last_observed = world.shape.windows[0][2]
    out = []
    if observed is None or observed.source is None:
        out += ["The observed run is not readable yet, so neither are its records.", ""]
    else:
        rows = _log_rows(observed, sql)
        if rows:
            out += [table(headers, [[r[c] for c in columns] for r in rows]), ""]
        else:
            out += ["The observed run's records have no entry.", ""]
    if later is not None and later.source is not None:
        added = [r for r in _log_rows(later, sql) if int(r["day"]) > last_observed]
        out += [
            f"The later run's records extend these through day {world.shape.days - 1}; "
            f"they add {len(added)} entries after day {last_observed}" + (":" if added else "."),
            "",
        ]
        if added:
            out += [table(headers, [[r[c] for c in columns] for r in added]), ""]
    return out


def _events(world: WorldSource) -> list[str]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    episodes: dict[str, set[str]] = defaultdict(set)
    for run in world.runs:
        rows = _log_rows(
            run,
            "SELECT ev.kind, ev.episode_id, ep.day FROM events ev "
            "JOIN episodes ep USING (episode_id)",
        )
        for r in rows:
            counts[str(r["kind"])][world.shape.window_of(int(r["day"]))] += 1
            episodes[str(r["kind"])].add(str(r["episode_id"]))
    if not counts:
        return ["The runs' events tables have no rows.", ""]
    columns = [*WINDOWS]
    if any("outside the windows" in c for c in counts.values()):
        columns.append("outside the windows")
    table_rows: list[list[Any]] = [
        [
            kind,
            *[counts[kind].get(w, 0) for w in columns],
            sum(counts[kind].values()),
            len(episodes[kind]),
        ]
        for kind in sorted(counts)
    ]
    return [table(["kind", *columns, "events", "episodes with one"], table_rows), ""]


def _health(world: WorldSource, page: Path, health: Health) -> list[str]:
    """The health section, with a figure per component and metric written into the page."""
    sources = "; ".join(health.notes) or "no run is readable yet"
    out = [f"Readings by component and metric over the days; {sources}.", ""]
    if not health.rows:
        return [*out, "No reading.", ""]
    sites = {r: world.shape.site_of(r) for r in {str(row["robot_id"]) for row in health.rows}}
    sites.update({str(e["robot_id"]): str(e["site_id"]) for e in world.episodes})
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in health.rows:
        groups[(str(r["component"]), str(r["metric"]))].append(r)
    for (component, metric), rows in sorted(groups.items()):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"health-{component}-{metric}") + ".png"
        health_figure(page / name, rows, sites, world.shape, f"{component}: {metric}")
        by_robot: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in rows:
            by_robot[str(r["robot_id"])].append((int(r["day"]), float(r["value"])))
        summary = []
        for robot in sorted(by_robot):
            points = sorted(by_robot[robot])
            values = [v for _, v in points]
            summary.append(
                [
                    robot,
                    sites.get(robot, ""),
                    len(points),
                    f"{_reading(values[0])} (day {points[0][0]})",
                    f"{_reading(values[-1])} (day {points[-1][0]})",
                    _reading(values[-1] - values[0], sign=True),
                    _reading(min(values)),
                    _reading(max(values)),
                ]
            )
        out += [
            f"### {component}: {metric}",
            "",
            f"![Health: {component} {metric}]({name})",
            "",
            table(
                ["robot", "site", "readings", "first", "last", "change", "lowest", "highest"],
                summary,
            ),
            "",
        ]
    return out


# ---- the truth sections


def _target(fault: FaultSpec) -> str:
    t = fault.target
    parts = [p for p in (t.component, t.config_key, t.condition_key) if p is not None]
    return " ".join(str(getattr(p, "value", p)) for p in parts)


def _level(fault: FaultSpec, days: int) -> str:
    from fleetlab.worlds.base import day_level

    drift = fault.schedule.drift
    if drift is None:
        return f"{fault.level:g}"
    active = [d for d in range(days) if fault.schedule.active_on(d)]
    if not active:
        return f"{fault.level:g}"
    first, last = active[0], active[-1]
    return (
        f"{day_level(fault, first):g} on day {first} to {day_level(fault, last):g} on day {last} "
        f"({drift.shape} drift, rate {drift.rate:g})"
    )


def _schedule(truth: WorldTruth) -> list[str]:
    spec = truth.spec
    rows = []
    for f in spec.fault_schedule:
        duration = f.schedule.duration_days
        rows.append(
            [
                f.fault_id,
                truth.roles.get(f.fault_id, ""),
                f.kind,
                f.target.scope,
                _target(f),
                _level(f, spec.days),
                f.schedule.onset_day,
                "to the end" if duration is None else f"{duration} days",
            ]
        )
    out = [
        "### Fault schedule (truth)",
        "",
        "Roles from `generator.json`; the level of a drift over the days it is in force.",
        "",
        table(["fault", "role", "kind", "scope", "target", "level", "onset day", "in force"], rows)
        if rows
        else "The schedule is empty.",
        "",
    ]
    p = spec.patterns
    patterns = [
        ("confounders (effective, inert)", ", ".join(f"{a} with {b}" for a, b in p.confounders)),
        ("decoys", ", ".join(p.decoys)),
        ("interactions", ", ".join(f"{a} with {b}" for a, b in p.interactions)),
        ("drifts", ", ".join(p.drifts)),
    ]
    out += ["### Patterns (truth)", ""]
    out += [f"- **{name}**: {value or 'none'}" for name, value in patterns]
    small = truth.generator.get("small_causes") or []
    out += [f"- **small causes**: {', '.join(small) or 'none'}", ""]
    return out


def _effects(cause: dict[str, Any]) -> str:
    effect = cause.get("expected_effect_points", {})
    return " / ".join(f"{float(effect.get(w, 0.0)):.2f}" for w in WINDOWS)


def _detectability(truth: WorldTruth) -> list[str]:
    detect = truth.detect
    out = ["### Detectability (truth)", ""]
    if not detect:
        return [*out, "`files/detectability.json` is missing.", ""]
    causes = detect.get("causes", [])
    counted = sum(1 for c in causes if c.get("counts_for_gate"))
    by_rule = sum(1 for c in causes if c.get("detectable"))
    wanted = truth.generator.get("recipe", {}).get("min_detectable", 6)
    rows = [
        [
            c.get("unit", ""),
            ", ".join(c.get("kinds", [])),
            ", ".join(c.get("roles", [])),
            ", ".join(c.get("scopes", [])),
            _effects(c),
            c.get("episodes", {}).get("observed", ""),
            ", ".join(c.get("qualified_by", [])),
            _yes(c.get("detectable")),
            _yes(c.get("confirmed")),
            _yes(c.get("counts_for_gate")),
            _yes(c.get("transferred_from_spatial")),
            _yes(c.get("assumed_joint")),
        ]
        for c in causes
    ]
    meets = "meets" if counted >= wanted else "**misses**"
    out += [
        f"**{counted} causes count for gate 3's sizing rule** (detectable and confirmed; "
        f"{by_rule} are detectable by the rule alone); the rule asks for at least {wanted}, "
        f"so the world {meets} it.",
        "",
        f"The rule: {detect.get('rule', '')}.",
        "",
        table(
            [
                "cause",
                "kinds",
                "roles",
                "scopes",
                "expected effect, points (observed / experiment / scoring)",
                "observed episodes",
                "qualified by",
                "detectable",
                "confirmed",
                "counts for gate 3",
                "transferred from spatial",
                "assumed joint",
            ],
            rows,
        )
        if rows
        else "No cause.",
        "",
    ]
    neutral = detect.get("neutral", [])
    if neutral:
        out += [
            "Neutral faults, zero in expectation (truth):",
            "",
            table(
                ["fault", "kind", "role", "scope", "observed episodes"],
                [
                    [n["fault_id"], n["kind"], n["role"], n["scope"], n["episodes"].get("observed")]
                    for n in neutral
                ],
            ),
            "",
        ]
    pairs = detect.get("interaction_pairs", [])
    if pairs:
        rows = []
        for p in pairs:
            a, b = p["pair"]
            loss = p.get("loss_points", {})
            rows.append(
                [
                    f"{a} with {b}",
                    p.get("suite", ""),
                    _number(loss.get(a)),
                    _number(loss.get(b)),
                    _number(loss.get("joint")),
                    _yes(p.get("superadditive")),
                    p.get("joint_role", ""),
                ]
            )
        out += [
            "Interaction pairs, loss in points of the calibrated curves (truth):",
            "",
            table(
                [
                    "pair",
                    "suite",
                    "first alone",
                    "second alone",
                    "joint",
                    "superadditive",
                    "joint point",
                ],
                rows,
            ),
            "",
        ]
    return out


def _windows_truth(world: WorldSource, truth: WorldTruth) -> list[str]:
    expected = truth.detect.get("window_success", {})
    episodes = world.episodes
    rows = []
    for name, first, last in world.shape.windows:
        mine = [e for e in episodes if e["window"] == name]
        logged = Rate.of(e["outcome"] == "success" for e in mine)
        known = [
            truth.true_success[e["episode_id"]]
            for e in mine
            if e["episode_id"] in truth.true_success
        ]
        true = Rate.of(known)
        want = expected.get(name, {})
        exp_logged = want.get("logged")
        rows.append(
            [
                name,
                _days(range(first, last + 1)),
                want.get("episodes", ""),
                "" if exp_logged is None else _percent(float(exp_logged)),
                "" if want.get("true") is None else _percent(float(want["true"])),
                logged.n,
                logged.text(),
                true.text() + ("" if true.n == logged.n else f" ({true.n} with a verdict)"),
                ""
                if exp_logged is None or not logged.n
                else f"{100 * (logged.rate - float(exp_logged)):+.1f}",
            ]
        )
    return [
        "### Success per window: expected and measured (truth)",
        "",
        "Expected from `detectability.json` (the calibrated curves over the as-run episodes of "
        "the window), measured from the runs: logged is the fleet's check after the data hook, "
        "true the verdict before it.",
        "",
        table(
            [
                "window",
                "days",
                "episodes planned",
                "expected logged",
                "expected true",
                "episodes run",
                "measured logged [95%]",
                "measured true [95%]",
                "logged, measured minus expected (points)",
            ],
            rows,
        ),
        "",
    ]


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        out = []
        for k, v in value.items():
            out += _flatten(f"{prefix}.{k}" if prefix else str(k), v)
        return out
    return [(prefix, value)]


def _summary(truth: WorldTruth) -> list[str]:
    out = ["### The runner's summary: measured against expected, gate-3 marks (truth)", ""]
    if truth.summary is None:
        return [*out, "`runner.json` has no summary yet (the runner's last step).", ""]
    rows = [[key, _number(value)] for key, value in _flatten("", truth.summary)]
    gate = [r for r in rows if "gate" in r[0].lower()]
    if gate:
        out += ["Gate-3 marks:", "", table(["mark", "value"], gate), "", "Everything:", ""]
    return [*out, table(["field", "value"], rows), ""]


def _generator(truth: WorldTruth) -> list[str]:
    g = truth.generator
    if not g:
        return ["### The generator (truth)", "", "`files/generator.json` is missing.", ""]
    recipe = g.get("recipe", {})
    facts = [
        ("generator", g.get("generator", "")),
        (
            "recipe",
            f"{recipe.get('name', '')}, seed {recipe.get('seed', '')}: {recipe.get('note', '')}",
        ),
        (
            "band and aim (observed logged success)",
            f"{_number(recipe.get('band'))}, aim {_number(recipe.get('aim'))}",
        ),
        ("attempt", g.get("attempt", "")),
        ("assumed joint point", _yes(g.get("assumed_joint"))),
        ("curves", ", ".join(g.get("curves", []))),
    ]
    facts += [(f"expected {k}", _number(v)) for k, v in (g.get("expected") or {}).items()]
    return ["### The generator (truth)", "", *[f"- **{k}**: {v}" for k, v in facts], ""]


def _comparison(truth: WorldTruth) -> list[str]:
    out = [
        "### Per fault: logged success of its episodes against the expectation (truth)",
        "",
        "A comparison of rates over the same episodes, **not an effect estimate**: for each "
        "fault, the episodes of each run that carry it, their logged success, and the mean of "
        "their expected logged success from the calibrated curves "
        "(`fleetlab.worlds.estimate.probability`, the generator's model: healthy rate of the "
        "suite times each fault's ratio). Those episodes carry other faults too, and both "
        "columns include them.",
        "",
    ]
    if truth.comparison_note:
        out += [f"Note: {truth.comparison_note}.", ""]
    if not truth.comparisons:
        return [*out, "No episode to compare.", ""]
    rows = [
        [
            c.label,
            c.role,
            c.kind,
            c.run,
            c.measured.n,
            c.measured.text(),
            _percent(c.expected),
            f"{100 * (c.measured.rate - c.expected):+.1f}",
        ]
        for c in truth.comparisons
    ]
    return [
        *out,
        table(
            [
                "fault",
                "role",
                "kind",
                "run",
                "episodes",
                "measured logged [95%]",
                "expected logged",
                "measured minus expected (points)",
            ],
            rows,
        ),
        "",
    ]


def _truth_section(world: WorldSource, truth: WorldTruth) -> list[str]:
    out = [
        "## Truth",
        "",
        "From the world's generator files and the runs' `truth/` tables; see the note at the "
        "top of the page.",
        "",
    ]
    out += _schedule(truth)
    out += _detectability(truth)
    out += _windows_truth(world, truth)
    out += _summary(truth)
    out += _comparison(truth)
    out += _generator(truth)
    return out


def write_world_page(
    world: WorldSource,
    page: Path,
    *,
    kind: str,
    note: str,
    truth: WorldTruth | None,
    run_links: dict[str, str],
    progress_url: str,
    generated: str,
    front: str = "",
) -> None:
    """Write the world's figures and `index.md` into the (fresh) page directory."""
    shape = world.shape
    episodes = world.episodes
    overall = world.overall
    windows = "; ".join(f"{name} days {first} to {last}" for name, first, last in shape.windows)
    facts: list[tuple[str, str]] = [
        ("root", f"`{world.root}`"),
        ("world", f"{world.world_id} ({kind}" + (f": {note})" if note else ")")),
        (
            "shape",
            f"{shape.sites} sites, {shape.robots_per_site} robots per site, {shape.days} days, "
            f"{shape.episodes_per_robot_day} episodes per robot and day, {shape.tasks} tasks",
        ),
        ("windows", windows),
        ("runner", "ready" if world.ready else "not ready"),
        ("episodes", f"{len(episodes)} of {shape.planned(0, shape.days - 1)} planned"),
        ("success [95%]", f"{overall.text()}, {overall.successes} of {overall.n}"),
    ]
    others = sorted(set(world.run_world_ids) - {world.world_id})
    if others:
        facts.append(
            (
                "note",
                f"the runs' episodes belong to world {', '.join(world.run_world_ids)}, "
                f"not only to {world.world_id}; they are shown as they are",
            )
        )
    if progress_url:
        facts.append(("progress page", link("dashboard progress", progress_url)))
    out = [
        f"# Dashboard: world {world.world_id}",
        "",
        f"{generated} Everything else on this page is a function of the world root's files.",
        "",
        *[f"- **{name}**: {value}" for name, value in facts],
        "",
    ]
    if truth is not None:
        out += [WORLD_TRUTH_NOTE, ""]
    out += ["## Runs", "", *_runs_section(world, run_links)]
    out += [
        "## Outcomes",
        "",
        "Both runs together, success by the logged outcome (the fleet's check, what the agents "
        "see), with Wilson's 95% interval in brackets.",
        "",
    ]
    if not episodes:
        out += ["No episode is logged yet.", ""]
    else:
        rows = []
        for name, first, last in shape.windows:
            mine = [e for e in episodes if e["window"] == name]
            rate = Rate.of(e["outcome"] == "success" for e in mine)
            run = ", ".join(sorted({e["run"] for e in mine}))
            planned = shape.planned(first, last)
            rows.append(
                [
                    name,
                    _days(range(first, last + 1)),
                    run,
                    planned,
                    rate.n,
                    rate.successes,
                    rate.text(),
                ]
            )
        out += [
            "### By window",
            "",
            table(
                [
                    "window",
                    "days",
                    "run",
                    "episodes planned",
                    "episodes",
                    "successes",
                    "success [95%]",
                ],
                rows,
            ),
            "",
            "### By site",
            "",
            _by(episodes, "site_id", "site"),
            "",
            "### By robot",
            "",
            _by(episodes, "robot_id", "robot", with_site=True),
            "",
        ]
        day_rows = [
            [d, shape.window_of(int(d)), r.n, r.successes, r.text()]
            for d, r in rates_by(episodes, "day")
        ]
        site_day_figure(page / "success-by-day-and-site.png", episodes, shape, overall)
        out += [
            "### By day",
            "",
            table(["day", "window", "episodes", "successes", "success [95%]"], day_rows),
            "",
            "### By site and day",
            "",
            "![Success by day, one panel per site](success-by-day-and-site.png)",
            "",
            "Successes of episodes per site and day; in the figure, the dashed line is the "
            "world's rate over both runs and the dotted lines separate the windows ("
            + windows
            + ").",
            "",
            _site_day(episodes, shape),
            "",
        ]
    health = _health(world, page, read_health(world))
    out += ["## Health checks", "", *health]
    out += [
        "## Change log",
        "",
        "As the observed run's records hold it (cut to the observed window).",
        "",
    ]
    out += _records(
        world,
        "change_log",
        ["day", "scope", "key", "old", "new", "author"],
        ["day", "scope", "key", "old", "new", "author"],
        "day, scope, key, new",
    )
    out += ["## Maintenance log", "", "As the observed run's records hold it.", ""]
    out += _records(
        world,
        "maintenance",
        ["day", "robot_id", "component", "verb", "note"],
        ["day", "robot", "component", "verb", "note"],
        "day, robot_id, component",
    )
    out += ["## Events", "", "Events per kind and window, both runs.", "", *_events(world)]
    if truth is not None:
        out += _truth_section(world, truth)
    (page / "index.md").write_text(front + "\n".join(out).rstrip() + "\n")
