"""The bundle an observer reads: one page per round, a summary, film and provenance.

    bundle/
      index.md          the summary: reserved panel, rescued and broken, cost, film
      round-<N>.md      one page per round
      provenance.md     every table, figure and clip with the files it comes from
      provenance.json   the same, with every file's SHA-256
      figures/*.png     clips/*.{gif,png,mp4}

The Markdown renders on the review site as a bundle (`hub check` before publishing). Prose on the
pages is fixed text around the records: it names what each number is and how it was
derived, and makes no claim of its own. Anything drawn from invented (fixture) or toy
records is marked on the page, in its caption and on its figure.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fleetlab.dashboard.data import Rate
from fleetlab.loop.exhibit import derive, figures
from fleetlab.loop.exhibit.clips import ClipPair, RenderedClip, first_rescued, render_pair
from fleetlab.loop.exhibit.derive import Item, Table, action_text, arm_text, state_label
from fleetlab.loop.exhibit.records import LearnedView, Origin, RoundView, RunView

__all__ = ["BANNERS", "build"]

BANNERS: dict[str, str] = {
    "fixture": (
        "**FIXTURE.** Invented numbers in the real record format, written to show the layout. "
        "Not a measurement; nothing here says anything about the robots."
    ),
    "toy": (
        "**TOY RUN.** The real loop pipeline on a CPU toy world with a scripted agent, made "
        "to exercise the records. The numbers say nothing about robots or about an agent."
    ),
}
MARK = {"fixture": "**[FIXTURE]** ", "toy": "**[TOY]** ", "real": ""}
MARKER_FILE = ".exhibit"


@dataclass
class Page:
    name: str
    lines: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)


@dataclass
class Book:
    view: RunView
    out: Path
    generated: str
    clips: bool
    items: list[Item] = field(default_factory=list)
    numbers: dict[str, int] = field(default_factory=dict)

    def register(self, item: Item) -> Item:
        if any(i.item == item.item for i in self.items):
            raise ValueError(f"two items named {item.item}")
        self.items.append(item)
        return item

    def number(self, page: str) -> int:
        self.numbers[page] = self.numbers.get(page, 0) + 1
        return self.numbers[page]


# ---- Markdown pieces ----------------------------------------------------------------------


def _anchor(item: str) -> str:
    """A link to an item on the provenance page. The review site's sanitizer prefixes element ids
    with `user-content-`, so the fragment carries the prefix; elsewhere the link still
    opens the page."""
    return f"provenance.md#user-content-{item}"


def cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def table_md(book: Book, page: Page, table: Table, label: str) -> None:
    book.register(table)
    page.add(
        f"*{label}. {MARK[table.origin]}{table.title}.* {table.caption} "
        f"(Source: [{table.item}]({_anchor(table.item)}).)",
        "",
        "| " + " | ".join(cell(h) for h in table.headers) + " |",
        "|" + "|".join("---" for _ in table.headers) + "|",
    )
    for row in table.rows:
        page.add("| " + " | ".join(cell(c) for c in row) + " |")
    page.add("")


def figure_md(book: Book, page: Page, item: Item, path: str, caption: str, label: str) -> None:
    book.register(item)
    page.add(
        f"![{cell(item.title)}]({path})",
        "",
        f"*{label}. {MARK[item.origin]}{item.title}.* {caption} "
        f"(Source: [{item.item}]({_anchor(item.item)}).)",
        "",
    )


def fence(text: str, info: str = "text") -> list[str]:
    longest = max((len(run) for run in _runs(text, "`")), default=0)
    mark = "`" * max(3, longest + 1)
    return [f"{mark}{info}", text.rstrip("\n"), mark]


def _runs(text: str, char: str) -> Iterable[str]:
    run = ""
    for c in text:
        if c == char:
            run += c
        elif run:
            yield run
            run = ""
    if run:
        yield run


def details(summary: str, body: Sequence[str]) -> list[str]:
    return ["<details>", f"<summary>{cell(summary)}</summary>", "", *body, "", "</details>", ""]


def opening(book: Book) -> list[str]:
    """The annotations' plain-language opening for the summary page (`opening`: a title,
    who wrote it, paragraphs), set apart from what the generator derives."""
    found = book.view.annotations.get("opening")
    if not isinstance(found, dict) or not found.get("paragraphs"):
        return []
    lines = [f"## {cell(found.get('title', 'In short'))}", ""]
    for paragraph in found["paragraphs"]:
        lines += [" ".join(str(paragraph).split()), ""]
    lines += [
        f"*Written for this page by {cell(found.get('by', 'a reviewer'))}. Its numbers are "
        "taken or summed from the tables below and their records.*",
        "",
    ]
    return lines


def annotations_for(book: Book, page: str, section: str) -> list[str]:
    ann = book.view.annotations
    by = ann.get("by", "a reviewer")
    out = []
    for note in ann.get("notes", []):
        if note.get("page") == page and note.get("section") == section:
            label = cell(ann.get("note_label", "Note"))
            out += [f"> **{label}** ({cell(by)}): {cell(note['text'])}", ""]
    return out


def banner(book: Book, page: Page, origins: Iterable[Origin]) -> None:
    shown = set()
    for origin in sorted(set(origins), key=lambda o: ("real", "toy", "fixture").index(o)):
        if origin in BANNERS and origin not in shown:
            page.add(f"> {BANNERS[origin]}", "")
            shown.add(origin)
    caption = book.view.annotations.get("run_caption")
    if caption:
        page.add(
            f"> **About this run** ({cell(book.view.annotations.get('by', 'a reviewer'))}): "
            f"{cell(caption)}",
            "",
        )


def generated_line(book: Book) -> str:
    return (
        f"*Generated by `fleetlab.loop.exhibit` on {book.generated} from the run's records. "
        "Every number on this page comes from a record named in the "
        "[provenance list](provenance.md).*"
    )


# ---- round pages --------------------------------------------------------------------------


def world_name(view: RunView) -> str:
    """The fleet's name for the reader: the annotations' `names.world` where given (plain
    words), else its identifier. Provenance always keeps the identifiers."""
    names = view.annotations.get("names") or {}
    return str(names.get("world") or f"world {view.world_id}")


def agent_name(view: RunView) -> str:
    names = view.annotations.get("names") or {}
    if names.get("agent"):
        return str(names["agent"])
    agent = view.agent
    text = f"agent {agent.get('model')}"
    if agent.get("effort"):
        text += f" at effort {agent['effort']}"
    return text + (f" ({agent.get('backend')})" if agent.get("backend") else "")


def identifiers(view: RunView) -> str:
    agent = view.agent
    parts = [f"world {view.world_id}"] + [
        f"{k} {agent[k]}" for k in ("model", "effort", "backend", "service_tier") if agent.get(k)
    ]
    return "; ".join(parts)


def _round_intro(view: RunView, rnd: RoundView) -> str:
    robots = sorted({str(r["robot_id"]) for r in rnd.ops_rows})
    sites = sorted({str(r["site_id"]) for r in rnd.ops_rows})
    first, last = rnd.operation_days
    experiments = len(rnd.experiments)
    attempts = sum(len(a.episode_ids) for e in rnd.experiments for a in e.arms)
    text = (
        f"The fleet ({world_name(view)}) did its regular work on "
        f"{derive.days_text(first, last)}: {len(rnd.ops_rows)} attempts by {len(robots)} "
        f"robots at {len(sites)} sites. The {agent_name(view)} then worked in "
        f"{len(rnd.sessions)} session{'s' if len(rnd.sessions) != 1 else ''}, requested "
        f"{experiments} "
        f"experiment{'s' if experiments != 1 else ''} ({attempts} attempts on the robots), "
    )
    if rnd.decision is not None:
        d = rnd.decision
        text += (
            f"and decided: {len(d.get('adopt', []))} adopted, {len(d.get('replace', []))} "
            f"replaced, {len(d.get('undo', []))} undone" + (", and stop." if d.get("stop") else ".")
        )
    elif rnd.fix_set is not None:
        text += (
            "and submitted a diagnosis and a ranked list of actions. This older format has no "
            "adoption: nothing was changed on the fleet."
        )
    else:
        text += "and left no decision" + (f" ({rnd.stopped})." if rnd.stopped else ".")
    return text


def _fleet_section(book: Book, page: Page, rnd: RoundView) -> None:
    view = book.view
    name = page.name
    page.add("## The fleet before the decision", "")
    by_robot = derive.fleet_by_robot(view, rnd)
    groups: dict[str, list[bool]] = {}
    for row in rnd.ops_rows:
        groups.setdefault(str(row["robot_id"]), []).append(row.get("outcome") == "success")
    overall = Rate.of(row.get("outcome") == "success" for row in rnd.ops_rows)
    path = f"figures/r{rnd.number}-fleet-robots.png"
    figures.rate_dots(
        book.out / path,
        [(robot, Rate.of(v)) for robot, v in sorted(groups.items())],
        f"Round {rnd.number}: success by robot, {derive.days_text(*rnd.operation_days)}",
        "robot",
        by_robot.origin,
        overall=overall,
    )
    figure_md(
        book,
        page,
        Item(
            f"r{rnd.number}-fig-robots",
            "Success by robot",
            by_robot.files,
            by_robot.derivation,
            by_robot.origin,
        ),
        path,
        "Each dot is a robot's success rate over the round's regular work, with its 95% "
        "interval; the dashed line is the whole fleet.",
        f"Figure {rnd.number}.{book.number(name + '-fig')}",
    )
    site_days = derive.fleet_by_site_day(view, rnd)
    series: dict[str, list[tuple[str, Rate]]] = {}
    for row in rnd.ops_rows:
        series.setdefault(str(row["site_id"]), [])
    for site in series:
        days = sorted({int(r["day"]) for r in rnd.ops_rows})
        series[site] = [
            (
                f"day {d}",
                Rate.of(
                    r.get("outcome") == "success"
                    for r in rnd.ops_rows
                    if r["site_id"] == site and int(r["day"]) == d
                ),
            )
            for d in days
        ]
    path = f"figures/r{rnd.number}-fleet-sites.png"
    figures.series_by_group(
        book.out / path,
        dict(sorted(series.items())),
        f"Round {rnd.number}: success by site and day",
        "day",
        site_days.origin,
    )
    figure_md(
        book,
        page,
        Item(
            f"r{rnd.number}-fig-sites",
            "Success by site and day",
            site_days.files,
            site_days.derivation,
            site_days.origin,
        ),
        path,
        "Each site's success rate on each day of regular work, with 95% intervals.",
        f"Figure {rnd.number}.{book.number(name + '-fig')}",
    )
    table_md(book, page, by_robot, f"Table {rnd.number}.{book.number(name)}")
    table_md(book, page, site_days, f"Table {rnd.number}.{book.number(name)}")
    lines, item = derive.weak_spots(view, rnd)
    book.register(item)
    page.add(
        f"**Where it is weak** {MARK[item.origin]}(rule: {item.derivation}; source "
        f"[{item.item}]({_anchor(item.item)})):",
        "",
    )
    page.add(*(f"- {line}" for line in lines) if lines else ["- nothing falls below by this rule"])
    page.add("")


def _working_section(book: Book, page: Page, rnd: RoundView) -> None:
    page.add("## The agent's working table", "")
    page.add(*annotations_for(book, page.name, "working"))
    if not rnd.diagnoses and not rnd.experiments:
        page.add("The agent submitted no diagnosis and no experiment in this round.", "")
    else:
        table_md(
            book,
            page,
            derive.working_table(book.view, rnd),
            f"Table {rnd.number}.{book.number(page.name)}",
        )
    summaries = derive.diagnosis_summaries(rnd)
    if summaries:
        session = rnd.diagnoses[-1][0]
        page.add(
            f"The agent's own summaries of the failure modes, verbatim from its last diagnosis "
            f"(session {session}):",
            "",
        )
        page.add(*(f"- {cell(s)}" for s in summaries))
        page.add("")
    if rnd.notes:
        page.add(
            "The agent's notes, verbatim, as they stood after each session (its only memory "
            "between sessions):",
            "",
        )
        for when, text in rnd.notes:
            page.add(*details(f"Notes {when}", fence(text)))


def _experiment_section(book: Book, page: Page, rnd: RoundView) -> None:
    page.add("## The experiment ledger", "")
    page.add(*annotations_for(book, page.name, "experiments"))
    if not rnd.experiments:
        page.add("The agent requested no experiment in this round.", "")
        return
    table = derive.experiment_table(book.view, rnd)
    panels = []
    for experiment in rnd.experiments:
        results = derive.arm_results(experiment)
        if not results:
            continue
        control = results[0].control_rate
        arms = [("control", control)] + [(f"arm {r.arm.index}", r.rate) for r in results]
        panels.append((f"{experiment.experiment_id} on {', '.join(experiment.robots)}", arms))
    if panels:
        path = f"figures/r{rnd.number}-experiments.png"
        figures.arms_figure(book.out / path, panels, table.origin)
        figure_md(
            book,
            page,
            Item(
                f"r{rnd.number}-fig-experiments",
                "Experiment arms",
                table.files,
                table.derivation,
                table.origin,
            ),
            path,
            "Each panel is one experiment: the control arm in grey, each changed arm in blue, "
            "with 95% intervals. The arms are named in the table below.",
            f"Figure {rnd.number}.{book.number(page.name + '-fig')}",
        )
    table_md(book, page, table, f"Table {rnd.number}.{book.number(page.name)}")
    page.add("Each request's question and power calculation, verbatim:", "")
    for experiment in rnd.experiments:
        named = "; ".join(f"arm {a.index}: {arm_text(a)}" for a in experiment.arms)
        page.add(
            f"- **{experiment.experiment_id}** (session {experiment.session}; {named}): "
            f"{cell(experiment.request.get('power_note', ''))}"
        )
    page.add("")


def _decision_section(book: Book, page: Page, rnd: RoundView) -> None:
    page.add("## The decision", "")
    page.add(*annotations_for(book, page.name, "decision"))
    tables = derive.decision_tables(book.view, rnd)
    if not tables:
        page.add(
            "No decision is in the records of this round"
            + (f": {rnd.stopped}." if rnd.stopped else "."),
            "",
        )
        return
    for table in tables:
        table_md(book, page, table, f"Table {rnd.number}.{book.number(page.name)}")
    if rnd.decision is not None:
        expectations = rnd.decision.get("expectations", [])
        if expectations:
            page.add("What the agent said the next round's records should show, verbatim:", "")
            page.add(*(f"- {cell(e)}" for e in expectations))
            page.add("")
        retired = [
            e for e in derive.ledger_entries(rnd.ledger_after) if e.get("status") != "active"
        ]
        if retired:
            page.add("Ledger entries no longer in force after this decision:", "")
            page.add(
                *(
                    f"- {e['entry_id']} ({action_text(e['action'])}): {e['status']}"
                    + (f", by {e['replaced_by']}" if e.get("replaced_by") else "")
                    for e in retired
                )
            )
            page.add("")
    check = derive.next_round_check(book.view, rnd)
    if check is not None:
        page.add("### What the next round's records showed", "")
        page.add(*annotations_for(book, page.name, "next"))
        table_md(book, page, check, f"Table {rnd.number}.{book.number(page.name)}")


def _learned_section(book: Book, page: Page, rnd: RoundView) -> None:
    page.add("## The procedure the agent wrote", "")
    if not rnd.learned:
        page.add("No diagnostic procedure in the records belongs to this round.", "")
        return
    for skill in rnd.learned:
        _skill_md(book, page, skill, f"r{rnd.number}-learned-{skill.name}")


def _skill_md(book: Book, page: Page, skill: LearnedView, item_id: str) -> None:
    item = Item(
        item_id,
        f"Learned procedure {skill.name}",
        skill.files,
        "the frozen library's SKILL.md, verbatim; round and session from the library "
        "record's source of the skill"
        if skill.frozen
        else "the agent's last accepted draft (outbox/learned.jsonl), rendered by "
        "contracts.learned.render_learned_skill without changes; its session from the "
        "draft's record, its round from the engagement's sessions",
        book.view.sources.origin(skill.files),
    )
    book.register(item)
    where = [
        f"round {skill.round}" if skill.round is not None else "round not recorded",
        f"session {skill.session}" if skill.session is not None else "session not recorded",
    ]
    state = "checked and frozen" if skill.frozen else "as drafted, before the host's check"
    if skill.carried_from:
        state += f"; carried from library {skill.carried_from[:16]}"
    page.add(
        f"**{skill.name}** {MARK[item.origin]}({'; '.join(where)}; {state}; source "
        f"[{item.item}]({_anchor(item.item)})):",
        "",
    )
    if skill.provenance:
        page.add(f"The host's provenance line: {cell(skill.provenance)}", "")
    page.add(*fence(skill.text, "markdown"), "")


def _round_clips(book: Book, rnd: RoundView) -> list[ClipPair]:
    pairs = []
    for experiment in rnd.experiments:
        for result in derive.arm_results(experiment):
            if result.verdict != "better than control" or not result.paired:
                continue
            i = first_rescued(result.control.rows, result.arm.rows)
            if i is None:
                continue
            before_row, after_row = result.control.rows[i], result.arm.rows[i]
            before_export, after_export = result.control.exports[i], result.arm.exports[i]
            if (
                before_row is None
                or after_row is None
                or before_export is None
                or after_export is None
            ):
                continue
            pairs.append(
                ClipPair(
                    item=f"r{rnd.number}-clip-{experiment.experiment_id}-arm{result.arm.index}",
                    before_log=before_export / "log",
                    before_row=before_row,
                    after_log=after_export / "log",
                    after_row=after_row,
                    before_label="before (control)",
                    after_label=f"after (arm {result.arm.index})",
                    rule=(
                        f"experiment {experiment.experiment_id}, arm {result.arm.index} "
                        f"({arm_text(result.arm)}): the first paired position ({i + 1} of "
                        f"{len(result.arm.rows)}) where the control attempt failed and the "
                        "arm's attempt succeeded on the same task"
                    ),
                    files=tuple(experiment.files),
                )
            )
    return pairs[:2]


def _clips_section(book: Book, page: Page, pairs: Sequence[ClipPair], empty: str) -> None:
    page.add("## Film", "")
    if not book.clips:
        page.add("Film was left out of this build.", "")
        return
    if not pairs:
        page.add(empty, "")
        return
    for pair in pairs:
        stem = pair.item
        rendered = render_pair(pair, book.out / "clips", stem)
        origin = book.view.sources.origin(pair.files)
        item = Item(pair.item, "Paired film", list(pair.files), pair.rule, origin)
        book.register(item)
        _clip_md(page, rendered, item)


def _clip_md(page: Page, rendered: RenderedClip, item: Item) -> None:
    pair = rendered.pair
    page.add(
        f"**{MARK[item.origin]}{cell(pair.before_row.get('task_text', ''))}** "
        f"(robot {pair.before_row.get('robot_id')}, {_days(pair)}). Chosen by rule: "
        f"{cell(pair.rule)}. "
        f"(Source: [{item.item}]({_anchor(item.item)}).)",
        "",
    )
    if rendered.gif:
        page.add(f"![Before and after, moving]({'clips/' + rendered.gif})", "")
    if rendered.strip:
        page.add(
            f"![Before and after, six frames each]({'clips/' + rendered.strip})",
            "",
            f"*Top row: {pair.before_label}, episode {pair.before_row.get('episode_id')}; "
            f"bottom row: {pair.after_label}, episode {pair.after_row.get('episode_id')}. Six "
            "evenly spaced frames of the scene camera from the start of each attempt to its "
            "end; the time is above each frame.*",
            "",
        )
    if rendered.videos:
        page.add(
            "Videos: " + ", ".join(f"[{v.split('.')[-2]}](clips/{v})" for v in rendered.videos),
            "",
        )
    for problem in rendered.problems:
        page.add(f"- Not shown: {cell(problem)}")
    if rendered.problems:
        page.add("")


def _days(pair: ClipPair) -> str:
    a, b = pair.before_row.get("day"), pair.after_row.get("day")
    return f"day {a}" if a == b else f"days {a} and {b}"


def round_page(book: Book, rnd: RoundView) -> Page:
    view = book.view
    page = Page(f"round-{rnd.number}")
    page.add("---", f'title: "{cell(view.label)}: round {rnd.number}"', "---")
    page.add(f"# Round {rnd.number} of {cell(view.label)}", "")
    banner(book, page, [rnd.origin])
    page.add(generated_line(book), "", "[Back to the summary](index.md)", "")
    page.add(*annotations_for(book, page.name, "top"))
    page.add(cell(_round_intro(view, rnd)), "")
    _fleet_section(book, page, rnd)
    _working_section(book, page, rnd)
    _experiment_section(book, page, rnd)
    _decision_section(book, page, rnd)
    _learned_section(book, page, rnd)
    _clips_section(
        book,
        page,
        _round_clips(book, rnd),
        "No experiment arm in this round both did better than its control and had a "
        "rescued pair with video, so there is no paired film from this round.",
    )
    page.add("## What this round cost", "")
    table_md(
        book,
        page,
        derive.cost_table(view, [rnd], evaluation=False),
        f"Table {rnd.number}.{book.number(page.name)}",
    )
    return page


# ---- the summary --------------------------------------------------------------------------


def _headline(book: Book) -> Table:
    view = book.view
    ev = view.evaluation
    states = {str(s["name"]): s for s in (ev.summary.get("states", []) if ev else [])}
    rounds = sorted(
        (n for n in states if n.startswith("round-")), key=lambda n: int(n.split("-")[1])
    )
    final = rounds[-1] if rounds else None

    def rate(name: str | None) -> str:
        if name is None or name not in states:
            return "not in the records"
        s = states[name]
        low, high = s["interval"]
        return f"{100 * s['rate']:.1f}% [{100 * low:.1f}, {100 * high:.1f}] ({name})"

    # The same rules as the cost tables (derive.experiment_attempts, derive.model_minutes).
    run = [e for r in view.rounds for e in r.experiments if e.status != "refused"]
    done = [e for e in run if e.status == "done"]
    attempts = derive.robot_attempts(view.rounds)
    probes = [derive.probe_simulation(r.costs) for r in view.rounds]
    if all(p.recorded and p.requests is not None for p in probes):
        probe_text = (
            f"; probes requested: {sum(p.requests or 0 for p in probes)} "
            f"({sum(p.refused or 0 for p in probes)} refused)"
        )
    else:
        probe_text = "; probes: not recorded"
    minutes = derive.model_minutes(view.rounds)
    reuse = (view.reuse or {}).get("differences", {})

    def reuse_value(key: str, arm: str) -> str:
        value = (reuse.get(key) or {}).get(arm)
        return "not in the records" if value is None else f"{value:g}"

    rows = [
        ["[A]", "reserved attempts, as at the start", rate("original")],
        ["[C]", "reserved attempts, after the last round", rate(final)],
        ["[B]", "reserved attempts, plan fixed after one look", rate("one-look")],
        [
            "[K]",
            "changes in force after the last round",
            str(states[final]["changes"]) if final else "not in the records",
        ],
        [
            "[N]",
            "experiments requested and not refused (all rounds)",
            f"{len(run)} ({len(done)} done){probe_text}",
        ],
        [
            "[E]",
            "robot executions of experiments and probes: attempts run, plus any execution "
            "that failed and was run again (all rounds)",
            attempts,
        ],
        ["[T]", "minutes of model time (all rounds)", minutes],
        [
            "[X]",
            "reuse: experiment attempts without the procedure",
            reuse_value("attempts_simulated", "without"),
        ],
        [
            "[Y]",
            "reuse: experiment attempts with the procedure",
            reuse_value("attempts_simulated", "with"),
        ],
        ["[R0]", "reuse: outcome score without the procedure", reuse_value("score", "without")],
        ["[R1]", "reuse: outcome score with the procedure", reuse_value("score", "with")],
    ]
    files = [
        f for r in view.rounds for f in r.cost_files + [x for e in r.experiments for x in e.files]
    ]
    files += (ev.files if ev else []) + view.reuse_files
    return Table(
        item="headline",
        title="The numbers the plan's headline asks for",
        files=sorted(set(files)),
        derivation="as in the tables below: panel states from summary.json; experiments by "
        "their recorded status; probes by the round cost records; attempts (task and probe "
        "apart) and model time by the cost tables' rules (completed plus failed "
        "simulations, 'at least' where usage, simulations or a probe's reserved rest are "
        "incomplete); "
        "reuse differences from comparison.json",
        origin=view.sources.origin(files),
        headers=["placeholder", "what", "value"],
        rows=rows,
        caption="The bracketed placeholders of the plan's headline, filled from the records "
        "where they exist. 'Not in the records' means this run did not produce it.",
    )


def _evaluation_clips(book: Book) -> list[ClipPair]:
    ev = book.view.evaluation
    if ev is None or not ev.attempts:
        return []
    names = [str(s["name"]) for s in derive.ordered_states(ev.summary)]
    rounds = [n for n in names if n.startswith("round-")]
    if "original" not in names or not rounds:
        return []
    final = rounds[-1]
    outcome = {(str(r["state"]), str(r["episode_id"])): bool(r["success"]) for r in ev.attempts}
    ids = sorted({str(r["episode_id"]) for r in ev.attempts})
    picks = []
    for kind, want in (("rescued", (False, True)), ("broken", (True, False))):
        for eid in ids:
            if (outcome.get(("original", eid)), outcome.get((final, eid))) == want:
                picks.append((kind, eid))
                break
    pairs = []
    for kind, eid in picks:
        rows = {}
        for state in ("original", final):
            table = ev.root / state / "log" / "episodes.parquet"
            if not table.is_file():
                break
            found = [
                r
                for r in book.view.sources.parquet(table, "evaluation state episodes", ev.origin)
                if r["episode_id"] == eid
            ]
            if found:
                rows[state] = (found[0], table)
        if len(rows) != 2 or not all(r.get("video_scene") for r, _ in rows.values()):
            continue  # an evaluation run that kept no video gives no film
        files = tuple(
            sorted({*ev.files, str(rows["original"][1].resolve()), str(rows[final][1].resolve())})
        )
        pairs.append(
            ClipPair(
                item=f"panel-clip-{kind}",
                before_log=ev.root / "original" / "log",
                before_row=rows["original"][0],
                after_log=ev.root / final / "log",
                after_row=rows[final][0],
                before_label="before (original)",
                after_label=f"after ({final})",
                rule=(
                    f"the first reserved attempt by episode id {kind} between the original "
                    f"state and {state_label(final)}"
                ),
                files=files,
            )
        )
    return pairs


def summary_page(book: Book) -> Page:
    view = book.view
    page = Page("index")
    # The review site's identity first, as `hub publish` writes it, when the manifest gives one.
    page.add("---", *([f"id: {view.hub_id}"] if view.hub_id else []))
    page.add(f'title: "{cell(view.label)}: what the loop did"', "---")
    page.add(f"# {cell(view.label)}: what the loop did and what it cost", "")
    origins: list[Origin] = [r.origin for r in view.rounds]
    if view.evaluation is not None:
        origins.append(view.sources.origin(view.evaluation.files))
    if view.reuse is not None:
        origins.append(view.sources.origin(view.reuse_files))
    banner(book, page, origins)
    page.add(generated_line(book), "")
    page.add(*opening(book))
    page.add(*annotations_for(book, "index", "top"))
    page.add(
        cell(
            f"Fleet: {world_name(view)}. Agent: {agent_name(view)}. Rounds in the records: "
            + ", ".join(str(r.number) for r in view.rounds)
            + (f". The run ended: {view.ended_because}." if view.ended_because else ".")
        ),
        "",
    )
    page.add("Pages of this bundle:", "")
    for rnd in view.rounds:
        page.add(
            f"- [Round {rnd.number}](round-{rnd.number}.md) {MARK[rnd.origin].strip()}".rstrip()
        )
    page.add("- [Provenance: every number's source](provenance.md)", "")
    table_md(book, page, _headline(book), "Table 1")
    missing: list[str] = []

    page.add("## The result on the reserved attempts", "")
    page.add(*annotations_for(book, "index", "panel"))
    states = derive.panel_states(view)
    if states is None:
        page.add(
            "No reserved evaluation is in the records of this run: there is no result on "
            "reserved attempts to show."
            + (
                f" The engagement record says it was skipped: {cell(view.evaluation_skipped)}."
                if view.evaluation_skipped
                else ""
            ),
            "",
        )
        missing.append("the reserved evaluation (states, rescued and broken, by robot)")
    else:
        path = "figures/panel-states.png"
        figures.rate_dots(
            book.out / path,
            [(name, r) for name, r in derive.by_state_rates(view)],
            "Reserved attempts: success by state of the fleet",
            "state",
            states.origin,
            highlight="original",
        )
        figure_md(
            book,
            page,
            Item(
                "fig-panel-states",
                "Success by state",
                states.files,
                states.derivation,
                states.origin,
            ),
            path,
            "The same reserved attempts under each state of the fleet, with 95% intervals; the "
            "original state in grey.",
            f"Figure {book.number('index-fig')}",
        )
        table_md(book, page, states, f"Table {book.number('index') + 1}")
        look = derive.one_look_text(view)
        if look is not None:
            text, item = look
            book.register(item)
            page.add(
                f"{MARK[item.origin]}{text} (Source: [{item.item}]({_anchor(item.item)}).)", ""
            )
        references = [
            str(s["name"])
            for s in view.evaluation.summary.get("states", [])  # type: ignore[union-attr]
            if not str(s["name"]).startswith("round-") and s["name"] not in ("original", "one-look")
        ]
        if references:
            page.add(
                "Reference methods on the same attempts: "
                + "; ".join(
                    derive.state_label(r, view.annotations.get("state_labels")) for r in references
                )
                + ". What each one was told and what it spent come from its own records; "
                "they are not on this page unless a table above shows them.",
                "",
            )
        page.add("## Attempts rescued and broken", "")
        pairs = derive.panel_pairs(view)
        if pairs is not None:
            pair_rows = [
                (f"{p['first']} to {p['second']}", int(p["rescued"]), int(p["broken"]))
                for p in view.evaluation.summary.get("pairs", [])  # type: ignore[union-attr]
            ]
            path = "figures/panel-pairs.png"
            figures.pairs_figure(
                book.out / path, pair_rows, "Rescued and broken attempts", pairs.origin
            )
            figure_md(
                book,
                page,
                Item(
                    "fig-panel-pairs",
                    "Rescued and broken",
                    pairs.files,
                    pairs.derivation,
                    pairs.origin,
                ),
                path,
                "For each pair of states: attempts the second state rescued (blue, right) and "
                "broke (orange, left) on the same reserved attempts.",
                f"Figure {book.number('index-fig')}",
            )
            table_md(book, page, pairs, f"Table {book.number('index') + 1}")
        sites = derive.panel_sites(view)
        if sites is not None:
            page.add("## Where the change happened, by site", "")
            page.add(*annotations_for(book, "index", "sites"))
            table_md(book, page, sites, f"Table {book.number('index') + 1}")
        robots = derive.panel_robots(view)
        if robots is not None:
            page.add("## Where the change happened, by robot", "")
            by_state = derive.robots_by_state(view)
            path = "figures/panel-robots.png"
            order = [str(s["name"]) for s in derive.ordered_states(view.evaluation.summary)]  # type: ignore[union-attr]
            figures.series_by_group(
                book.out / path,
                {
                    state_label(n, view.annotations.get("state_labels")): sorted(
                        by_state[n].items()
                    )
                    for n in order
                    if n in by_state
                },
                "Reserved attempts by robot and state",
                "robot",
                robots.origin,
            )
            figure_md(
                book,
                page,
                Item(
                    "fig-panel-robots",
                    "By robot and state",
                    robots.files,
                    robots.derivation,
                    robots.origin,
                ),
                path,
                "Each robot's success on the reserved attempts under each state, with 95% "
                "intervals.",
                f"Figure {book.number('index-fig')}",
            )
            table_md(book, page, robots, f"Table {book.number('index') + 1}")

    comparators = derive.comparator_table(view)
    if comparators is not None:
        page.add("## Beside other methods on the same reserved attempts", "")
        page.add(*annotations_for(book, "index", "comparators"))
        table_md(book, page, comparators, f"Table {book.number('index') + 1}")
        breakdown = derive.comparator_breakdown(view)
        if breakdown is not None:
            table_md(book, page, breakdown, f"Table {book.number('index') + 1}")
    page.add("## What was actually wrong", "")
    page.add(*annotations_for(book, "index", "truth"))
    truth = derive.truth_table(view)
    if truth is None:
        page.add(
            "The planted truth is not shown: no development world was given for it, or the "
            "world's truth may not be shown.",
            "",
        )
        missing.append("the planted truth")
    else:
        table_md(book, page, truth, f"Table {book.number('index') + 1}")

    page.add(
        "## The agent's procedure, used again"
        if view.reuse is not None
        else "## The procedures the agent wrote",
        "",
    )
    reuse = derive.reuse_table(view)
    page.add(*annotations_for(book, "index", "procedures"))
    record = view.learned_record or {}
    if record.get("digest"):
        page.add(
            f"The host checked the procedures and froze them into a library with the "
            f"fingerprint {str(record['digest'])[:12]}… (full: `{record['digest']}`): "
            f"{len(record.get('kept') or [])} kept, {len(record.get('refused') or {})} "
            "refused.",
            "",
        )
    learned = [(r.number, s) for r in view.rounds for s in r.learned]
    if learned:
        page.add(
            "Procedures the agent wrote: "
            + ", ".join(f"[{s.name}](round-{n}.md) (round {n})" for n, s in learned)
            + " (verbatim on the round pages).",
            "",
        )
    if record.get("refused"):
        page.add(
            "Drafts the host's check refused and did not keep: "
            + ", ".join(sorted(str(n) for n in record["refused"]))
            + " (the reasons are host-side and not shown).",
            "",
        )
    for k, skill in enumerate(view.unplaced_learned, 1):
        if k == 1:
            page.add(
                "Procedures in the records that no round of this run can be tied to (no "
                "round or session recorded, or written by another investigation), verbatim:",
                "",
            )
        _skill_md(book, page, skill, f"learned-unplaced-{skill.name}")
    if reuse is None:
        page.add("No reuse comparison is in the records of this run.", "")
        missing.append("the reuse comparison")
    else:
        table_md(book, page, reuse, f"Table {book.number('index') + 1}")
        notes = derive.reuse_notes(view)
        if notes:
            page.add("Possible negative transfer, as the comparison recorded it:", "")
            page.add(*(f"- {cell(n)}" for n in notes))
            page.add("")

    page.add("## What it cost", "")
    page.add(*annotations_for(book, "index", "costs"))
    table_md(book, page, derive.cost_table(view), f"Table {book.number('index') + 1}")
    simulated = derive.simulated_table(view)
    if simulated is not None:
        table_md(book, page, simulated, f"Table {book.number('index') + 1}")
    incidents = derive.incidents_table(view)
    if incidents is not None:
        table_md(book, page, incidents, f"Table {book.number('index') + 1}")
    film = _evaluation_clips(book)
    _clips_section(
        book,
        page,
        film,
        "No reserved attempt with video was rescued or broken between the original and the "
        "final state, or the evaluation kept no video, so there is no film from the reserved "
        "attempts.",
    )
    if not film and view.evaluation is not None:
        missing.append("film from the reserved attempts")
    page.add("## What this bundle does not show", "")
    always = [
        "anything the agent did not record: its reasoning between tool calls is not in these "
        "records",
        "whether an experiment changed the decision: the pages show what the decision did "
        "with each tested change, not what it would have been without the test",
    ]
    page.add(*(f"- {x}" for x in [*missing, *always]))
    page.add("")
    return page


# ---- provenance ---------------------------------------------------------------------------


def provenance_page(book: Book) -> tuple[Page, dict[str, Any]]:
    view = book.view
    page = Page("provenance")
    page.add("---", f'title: "{cell(view.label)}: provenance"', "---")
    page.add(f"# Where every number of {cell(view.label)} comes from", "")
    page.add(
        generated_line(book),
        "",
        "Each table, figure and film on these pages has an entry here: the files it was "
        "derived from and how. Files are listed with their SHA-256, so a changed record "
        "shows. Origin is *real* (a run of the loop), *toy* (the real pipeline on a CPU toy "
        "world with a scripted agent) or *fixture* (invented numbers in the real format).",
        "",
        f"Exact identifiers of this run: {cell(identifiers(view))}.",
        "",
    )
    short = {key: f"F{i + 1}" for i, key in enumerate(sorted(view.sources.files))}
    page.add("## Items", "")
    for item in book.items:
        page.add(
            f'<a id="{item.item}"></a>',
            "",
            f"**{item.item}**: {cell(item.title)} (origin: {item.origin}). Derived as: "
            f"{cell(item.derivation)}. Files: "
            + ", ".join(f"[{short[f]}](#{short[f].lower()})" for f in item.files if f in short),
            "",
        )
    page.add("## Files", "")
    page.add("| file | role | origin | bytes | SHA-256 | path |", "|---|---|---|---|---|---|")
    for key in sorted(view.sources.files):
        record = view.sources.files[key]
        page.add(
            f'| <a id="{short[key].lower()}"></a>{short[key]} | {cell(record.role)} | '
            f"{record.origin} | {record.bytes} | `{record.sha256[:16]}` | `{cell(key)}` |"
        )
    page.add("")
    items = [
        {k: v for k, v in asdict(i).items() if k not in ("headers", "rows")} for i in book.items
    ]
    data: dict[str, Any] = {
        "format": "fleetlab.exhibit-provenance/1",
        "generated": book.generated,
        "label": view.label,
        "world_id": view.world_id,
        "items": items,
        "files": {k: asdict(v) for k, v in sorted(view.sources.files.items())},
    }
    return page, data


# ---- the build ----------------------------------------------------------------------------


def _prepare(out: Path) -> None:
    if out.exists():
        if not (out / MARKER_FILE).is_file():
            raise FileExistsError(
                f"{out} exists and is not an exhibit bundle; refusing to clear it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER_FILE).write_text("an exhibit bundle; rebuilt in full by fleetlab.loop.exhibit\n")


def build(view: RunView, out: Path, *, clips: bool = True, generated: str | None = None) -> Path:
    """Write the bundle for `view` into `out` (cleared first if it is an earlier bundle)."""
    out = Path(out).resolve()
    _prepare(out)
    when = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    book = Book(view=view, out=out, generated=when, clips=clips)
    pages = [round_page(book, r) for r in view.rounds]
    pages.insert(0, summary_page(book))
    provenance, data = provenance_page(book)
    pages.append(provenance)
    for page in pages:
        (out / f"{page.name}.md").write_text("\n".join(page.lines).rstrip() + "\n")
    (out / "provenance.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    return out
