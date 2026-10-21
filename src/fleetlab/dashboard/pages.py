"""The Markdown of the run page and the progress page (a world's page: `world_page`).

Pure text: these functions get what the build read and computed and return the
page. The only wall-clock value is the generated-at line, passed in.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from fleetlab.dashboard.data import (
    DosePoint,
    FaultGain,
    Source,
    Truth,
    WorldRegistry,
    faults_by,
    rates_by,
)
from fleetlab.dashboard.sources import Bead, Progress, QueueJob

__all__ = ["GENERATED_PREFIX", "EpisodeCard", "WorldRow", "progress_page", "run_page", "table"]

GENERATED_PREFIX = "_Generated at "
TRUTH_NOTE = (
    "> **Truth: the hidden side.** Everything marked *truth* on this page comes from the "
    "run's `truth/` tables, which the agents never see. It is shown because the page was "
    "built with `--truth`, which the dashboard allows only for worlds registered as "
    "development material in `dashboard/worlds.json`."
)


def cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def link(text: str, url: str) -> str:
    return f"[{text}]({url})" if url else text


@dataclass(frozen=True)
class EpisodeCard:
    """A sampled episode as its page section shows it."""

    row: dict[str, Any]
    strip: str  # file name of the frame strip
    strip_steps: tuple[int, ...]
    cameras: tuple[str, ...]  # the strip's rows, top to bottom
    video: str = ""  # file name of the scene video in the bundle, if copied
    video_note: str = ""  # why there is no video, if there is none
    faults: tuple[str, ...] = ()  # truth: the active faults
    verdict: str = ""  # truth: the true verdict


@dataclass(frozen=True)
class RunPageData:
    source: Source
    world_kind: str
    world_note: str
    figures: dict[str, str]  # figure key to file name
    cards: list[EpisodeCard]
    dose: list[DosePoint] = field(default_factory=list)
    truth: Truth | None = None
    progress_url: str = ""


def _chain(source: Source) -> str:
    record = source.record
    if record is None:
        return "no run record"
    n = len(record.chain_sample)
    if record.chain_ok is None:
        return f"{n} episodes in the sample, not replayed yet"
    transients = len(chain_checks(source).get("transients") or [])
    if record.chain_ok and transients:
        return (
            f"the sample of {n} episodes reproduced its states and actions; "
            f"{transients} render transient (below)"
        )
    if record.chain_ok:
        return f"the sample of {n} episodes reproduced its hash chains"
    why = chain_checks(source).get("broken_by") or ""
    return f"**broken**: the sample of {n} episodes did not reproduce" + (
        f" ({why})" if why else ""
    )


def chain_checks(source: Source) -> dict[str, Any]:
    """`run.json`'s `extra.chain_checks` (identical, render transients, broken), or {}."""
    if source.record is None:
        return {}
    value = source.record.extra.get("chain_checks")
    return value if isinstance(value, dict) else {}


def _transient_lines(source: Source) -> list[str]:
    """The run's render transients as a warning (amber), one line each
    (docs/design/contracts.md section 4)."""
    out = []
    for t in chain_checks(source).get("transients") or []:
        if not isinstance(t, dict):
            continue
        odd = {"run": "the run's", "replay": "the sample replay's"}.get(str(t.get("odd")), "an")
        out.append(
            f"> Render transient: episode {t.get('episode_id')}, step {t.get('step')}, "
            f"{t.get('camera')} camera; {odd} rendering was the odd one. State and executed "
            "action were identical at every step and nothing consumed the frame, so the run "
            "keeps its chain verdict."
        )
    if not out:
        return []
    return ["> [!WARNING]", *[line for item in out for line in (item, ">")][:-1], ""]


def _software(source: Source) -> str:
    if source.record is None:
        return ""
    s = source.record.software
    return (
        f"lock {s.lock_hash}, LIBERO patches {s.libero_patch_set}, "
        f"MuJoCo {s.mujoco_version}, contracts {s.contracts_version}"
    )


def _outcome_section(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    figure: str | None,
    alt: str,
) -> list[str]:
    out = [f"### {title}", "", table(headers, rows), ""]
    if figure:
        out += [f"![{alt}]({figure})", ""]
    return out


def _grouped(
    data: RunPageData, key: str, label: str, extra: dict[str, str] | None = None
) -> tuple[list[str], list[list[Any]]]:
    episodes = data.source.episodes
    headers = [label, *(extra or {}).values(), "episodes", "successes", "success [95%]"]
    truth_faults = faults_by(data.truth, episodes, key) if data.truth else None
    if truth_faults is not None:
        headers.append("active faults (truth)")
    rows = []
    for value, rate in rates_by(episodes, key):
        row: list[Any] = [value]
        for column in extra or {}:
            row.append(next(e[column] for e in episodes if e[key] == value))
        row += [rate.n, rate.successes, rate.text()]
        if truth_faults is not None:
            row.append("; ".join(truth_faults.get(value, [])) or "none")
        rows.append(row)
    return headers, rows


def _card(card: EpisodeCard) -> list[str]:
    e = card.row
    out = [
        f"### {e['episode_id']}: {e['robot_id']} at {e['site_id']}, day {e['day']}, {e['outcome']}",
        "",
        f"{e['task_text']}; {e['steps']} steps, {e['duration_s']:g} s simulated, "
        f"started at {e['started_at']}.",
        "",
    ]
    if card.strip:
        out += [
            f"![Frames of {e['episode_id']}]({card.strip})",
            "",
            "Frames at steps "
            + ", ".join(str(s) for s in card.strip_steps)
            + "; "
            + ", ".join(f"{name} camera" for name in card.cameras)
            + (" from top to bottom." if len(card.cameras) > 1 else "."),
            "",
        ]
    if card.video:
        out += [
            f'<video src="{card.video}" controls preload="metadata" width="360"></video>',
            "",
            f"[Scene video]({card.video})",
            "",
        ]
    if card.video_note:
        out += [card.video_note, ""]
    if card.faults or card.verdict:
        faults = "; ".join(card.faults) or "none"
        out += [f"*Truth*: active faults {faults}; true verdict {card.verdict}.", ""]
    return out


def _dose_section(points: list[DosePoint], figure: str | None) -> list[str]:
    rows = [
        [
            p.kind or ("healthy" + (f" (runs {p.runs})" if p.runs else "")),
            "" if p.level is None else f"{p.level:g}",
            p.rate.n,
            p.rate.successes,
            p.rate.text(),
        ]
        for p in points
    ]
    out = [
        "## Dose-response (truth)",
        "",
        "> *Truth*, shown without `--truth`: the world is registered as the calibration, "
        "whose fault kinds and levels are its published design (docs/design/faults.md). "
        "They are read from the run's identities, and success is the true outcome (the "
        "fleet's check before the data hook).",
        "",
        "Success by fault kind and level; the grey band is the healthy episodes' rate and "
        "interval.",
        "",
        table(["kind", "level", "episodes", "successes", "success [95%]"], rows),
        "",
    ]
    if figure:
        out += [f"![Dose-response]({figure})", ""]
    return out


def _gains(gains: list[FaultGain]) -> str:
    rows = [[g.fault_id, g.active, g.covered, g.rescued, g.broken, g.net] for g in gains]
    return table(
        ["fault", "episodes active", "with a replay without it", "rescued", "broken", "net"],
        rows,
    )


def _truth_section(data: RunPageData) -> list[str]:
    truth = data.truth
    assert truth is not None
    episodes = data.source.episodes
    out = [
        "## Truth",
        "",
        "From the run's `truth/` tables; see the note at the top of the page.",
        "",
    ]
    out += ["### Active faults (truth)", ""]
    if truth.faults:
        rows = [
            [
                f.fault_id,
                f.kind,
                f.levels,
                f.target,
                f.onset_day,
                f.episodes.n,
                f.episodes.text(),
            ]
            for f in truth.faults
        ]
        out += [
            table(
                [
                    "fault",
                    "kind",
                    "level",
                    "target",
                    "onset day",
                    "episodes",
                    "logged success of those [95%]",
                ],
                rows,
            ),
            "",
        ]
    else:
        out += ["No fault was active in this run.", ""]
    out += [
        "### Oracle: the gain of removing each fault (truth)",
        "",
        "Over the episodes as run in which the fault was active and that were replayed "
        "without it: rescued turned from failure to success, broken the other way "
        "(`contracts.oracle.fault_effect`).",
        "",
    ]
    if truth.oracle_problem:
        out += [f"Not computed: {truth.oracle_problem}.", ""]
    elif truth.gains:
        out += [_gains(truth.gains), ""]
    else:
        out += ["No fault to remove.", ""]
    out += ["### Oracle gains as written (truth/oracle.parquet)", ""]
    if truth.oracle:
        rows = [
            [r["item"], r["item_kind"], r["rescued"], r["broken"], r["net"]] for r in truth.oracle
        ]
        out += [table(["item", "kind", "rescued", "broken", "net"], rows), ""]
    else:
        out += ["The table is empty.", ""]
    agree = {(a, b): 0 for a in (True, False) for b in (True, False)}
    missing = 0
    for e in episodes:
        true = truth.true_success.get(e["episode_id"])
        if true is None:
            missing += 1
            continue
        agree[(e["outcome"] == "success", true)] += 1
    out += [
        "### Logged outcome against the true verdict (truth)",
        "",
        table(
            ["logged", "true success", "true failure"],
            [
                ["success", agree[(True, True)], agree[(True, False)]],
                ["failure", agree[(False, True)], agree[(False, False)]],
            ],
        ),
        "",
    ]
    if missing:
        out += [f"{missing} episodes have no verdict.", ""]
    return out


def run_page(data: RunPageData, generated: str) -> str:
    source = data.source
    overall = source.overall
    world = f"{source.world} ({data.world_kind}" + (
        f": {data.world_note})" if data.world_note else ")"
    )
    facts: list[tuple[str, str]] = [
        ("directory", f"`{source.root}`"),
        ("world", world),
        ("state", source.state),
        ("episodes", str(len(source.episodes))),
        ("success [95%]", f"{overall.text()}, {overall.successes} of {overall.n}"),
    ]
    if source.is_run:
        facts.append(("hash chains", _chain(source)))
        facts.append(("software", _software(source)))
    if data.progress_url:
        facts.append(("progress page", link("dashboard progress", data.progress_url)))
    out = [
        f"# Dashboard: run {source.run_id}",
        "",
        f"{generated} Everything else on this page is a function of the run's files.",
        "",
        *[f"- **{name}**: {value}" for name, value in facts],
        "",
    ]
    if source.is_run:
        out += _transient_lines(source)
    if data.truth is not None:
        out += [TRUTH_NOTE, ""]
    out += [
        "## Outcomes",
        "",
        "Success by the logged outcome (the fleet's check, what the agents see), with "
        "Wilson's 95% interval in brackets.",
        "",
    ]
    if not source.episodes:
        out += ["No episode is logged yet.", ""]
    else:
        headers, rows = _grouped(data, "robot_id", "robot", {"site_id": "site"})
        out += _outcome_section(
            "By robot", headers, rows, data.figures.get("robot"), "Success by robot"
        )
        headers, rows = _grouped(data, "site_id", "site")
        out += _outcome_section("By site", headers, rows, None, "")
        headers, rows = _grouped(data, "day", "day")
        out += _outcome_section("By day", headers, rows, data.figures.get("day"), "Success by day")
        task_rows = [
            [t, r.n, r.successes, r.text()] for t, r in rates_by(source.episodes, "task_text")
        ]
        out += _outcome_section(
            "By task", ["task", "episodes", "successes", "success [95%]"], task_rows, None, ""
        )
    if data.dose:
        out += _dose_section(data.dose, data.figures.get("dose"))
    out += ["## Episodes", ""]
    with_video = sum(1 for e in source.episodes if e.get("video_scene"))
    if data.cards:
        out += [
            f"{len(data.cards)} of the {with_video} episodes with video: failures and "
            "successes, each picked evenly spaced in episode order.",
            "",
        ]
        for card in data.cards:
            out += _card(card)
    elif with_video:
        out += [f"{with_video} episodes have a video; none was sampled (`--sample`).", ""]
    else:
        out += ["No episode of this run has a video.", ""]
    if data.truth is not None:
        out += _truth_section(data)
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------- progress page


@dataclass(frozen=True)
class RunRow:
    """A run on the progress page: from this build, or known from the manifest."""

    run_id: str
    url: str
    source: Source | None  # None when the directory is gone or unreadable
    note: str = ""  # why there is no source


def _transient_rate(runs: list[RunRow]) -> str:
    """Render transients per million compared frames, over the runs that recorded them."""
    frames = transients = counted = 0
    for run in runs:
        if run.source is None:
            continue
        checks = chain_checks(run.source)
        if "compared_frames" not in checks:
            continue
        counted += 1
        frames += int(checks.get("compared_frames") or 0)
        transients += len(checks.get("transients") or [])
    if not frames:
        return (
            "No run of this page recorded its compared frames yet (newer engine runs record them)."
        )
    rate = transients / frames * 1e6
    return (
        f"{transients} render transient{'' if transients == 1 else 's'} in {frames:,} frames "
        "compared by the chain samples of "
        f"{counted} runs: {rate:.2f} per million frames. A rising rate means a GPU going bad "
        "(docs/design/contracts.md section 4)."
    )


def _runs_table(runs: list[RunRow], registry: WorldRegistry) -> str:
    rows = []
    for run in runs:
        s = run.source
        if s is None:
            rows.append([link(run.run_id, run.url), "", "", run.note, "", "", ""])
            continue
        kinds = ", ".join(registry.kind(w) for w in s.world_ids) or "unknown"
        chain = ""
        if s.record is not None:
            chain = {None: "not replayed", True: "ok", False: "**broken**"}[s.record.chain_ok]
            if s.record.chain_ok and chain_checks(s).get("transients"):
                chain = "ok, 1 render transient"
        rows.append(
            [
                link(run.run_id, run.url),
                s.world,
                kinds,
                s.state,
                str(len(s.episodes)),
                s.overall.text(),
                chain,
            ]
        )
    return table(
        ["run", "world", "world kind", "state", "episodes", "success [95%]", "hash chains"], rows
    )


def _worlds_table(runs: list[RunRow], registry: WorldRegistry) -> str:
    worlds: dict[str, list[RunRow]] = {}
    for run in runs:
        if run.source is None:
            continue
        for w in run.source.world_ids or ("unknown",):
            worlds.setdefault(w, []).append(run)
    rows = []
    for w in sorted(worlds):
        members = worlds[w]
        episodes = sum(len(r.source.episodes) for r in members if r.source is not None)
        rows.append(
            [
                w,
                registry.kind(w),
                ", ".join(r.run_id for r in members),
                episodes,
                registry.notes.get(w, ""),
            ]
        )
    return table(["world", "kind", "runs", "episodes", "note"], rows)


@dataclass(frozen=True)
class WorldRow:
    """A world page on the progress page: from this build, or known from the manifest."""

    world_id: str
    url: str  # the world page's hub URL, if published
    kind: str
    ready: str = ""
    runs: str = ""  # the world's runs, linked to their pages where those exist
    episodes: str = ""
    success: str = ""
    root: str = ""
    note: str = ""  # built in this build but not published yet, or not readable


def _world_pages(worlds: Sequence[WorldRow]) -> list[str]:
    rows = [
        [
            link(w.world_id, w.url),
            w.kind,
            w.ready,
            w.runs,
            w.episodes,
            w.success,
            f"`{w.root}`" if w.root else "",
            w.note,
        ]
        for w in sorted(worlds, key=lambda w: w.world_id)
    ]
    return [
        table(
            ["world page", "kind", "runner", "runs", "episodes", "success [95%]", "root", "note"],
            rows,
        ),
        "",
        "One page per world root, both of its runs merged (`--world-roots`).",
        "",
    ]


def _queue(queue: list[QueueJob] | str) -> list[str]:
    if isinstance(queue, str):
        return [f"Not read: {queue}.", ""]
    if not queue:
        return ["No heavy job is running or waiting.", ""]
    rows = [[j.state, j.pid, f"`{j.command}`"] for j in queue]
    return [
        table(["state", "process", "command"], rows),
        "",
        "Waiting jobs are listed by process id; the lock does not serve them in that order.",
        "",
    ]


def _beads(rows: Sequence[Bead], gate: bool = False) -> str:
    if gate:
        return table(
            ["bead", "status", "closed", "title"],
            [[b.id, b.status, b.closed_on, b.title] for b in rows],
        )
    return table(
        ["bead", "priority", "type", "title"],
        [[b.id, f"P{b.priority}", b.issue_type, b.title] for b in rows],
    )


def _gate_number(bead: Bead) -> int:
    match = re.match(r"^Gate (\d+)", bead.title)
    return int(match.group(1)) if match else 0


def _demote(markdown: str) -> str:
    """Headings of an included note two levels down, outside code fences."""
    out, fenced = [], False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and re.match(r"^#{1,4} ", line):
            line = "##" + line
        out.append(line)
    return "\n".join(out)


def progress_page(
    runs: list[RunRow],
    registry: WorldRegistry,
    progress: Progress,
    generated: str,
    worlds: Sequence[WorldRow] = (),
) -> str:
    out = [
        "# Dashboard: progress",
        "",
        f"{generated} Runs, beads and the queue are read at that moment.",
        "",
        "## Runs",
        "",
    ]
    if runs:
        out += [
            _runs_table(runs, registry),
            "",
            "Runs of this build and those published before (the dashboard's manifest); "
            "success by the logged outcome with Wilson's 95% interval.",
            "",
        ]
    else:
        out += ["No run is known to the dashboard yet.", ""]
    if worlds:
        out += ["## World pages", "", *_world_pages(worlds)]
    out += ["## Worlds", "", _worlds_table(runs, registry), ""] if runs else []
    if runs:
        out += ["## Render transients", "", _transient_rate(runs), ""]
    out += ["## Queue (tools/heavy-run)", "", *_queue(progress.queue)]
    out += ["## Gates", ""]
    if isinstance(progress.beads, str):
        out += [f"Not read: {progress.beads}.", ""]
    else:
        gates = sorted((b for b in progress.beads if b.is_gate), key=_gate_number)
        out += [_beads(gates, gate=True), ""]
    out += ["## Waiting for a human", ""]
    if isinstance(progress.human, str):
        out += [f"Not read: {progress.human}.", ""]
    elif progress.human:
        human = sorted(progress.human, key=lambda b: (b.priority, b.id))
        out += [_beads(human), ""]
    else:
        out += ["Nothing.", ""]
    if isinstance(progress.beads, list):
        open_beads = sorted(
            (b for b in progress.beads if b.status != "closed" and not b.is_gate),
            key=lambda b: (b.priority, b.id),
        )
        doing = [b for b in open_beads if b.status == "in_progress"]
        waiting = [b for b in open_beads if b.status != "in_progress"]
        out += ["## Beads in progress", "", _beads(doing) if doing else "None.", ""]
        out += ["## Open beads", "", _beads(waiting) if waiting else "None.", ""]
    out += ["## Last nightly comparison", ""]
    if progress.nightly is None:
        out += ["None given (`--nightly FILE`); the nightly worlds come with stage 3.", ""]
    else:
        name, text = progress.nightly
        out += [f"From `{name}`:", "", _demote(text).strip(), ""]
    return "\n".join(out).rstrip() + "\n"
