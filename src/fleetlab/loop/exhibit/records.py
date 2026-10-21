"""What the exhibit reads: a finished loop run's records, each file registered for provenance.

Three kinds of input, all read as the files the runtime wrote, never recomputed by a model:

- an **engagement** (`engagement.json`, lane driver's `run_engagement`): rounds with their
  operation, export, sessions, costs, decision and ledger; the evaluation's summary; the
  analysis directory with the agent's submissions, requests, results and notes;
- an older **investigation** (`run.json` of kind `investigation`): one round, no round
  decision, a ranked fix set instead (the miniature runs of 13 October);
- a **manifest** (`exhibit.json`) that assembles a run from parts of either, with the
  evaluation, the reuse comparison, a learned library, the world for the planted truth and
  reviewers' annotations.

Every file read goes through `Sources`, which keeps its path, SHA-256 and origin: `real`,
`toy` (the real pipeline on a CPU toy world with a scripted agent) or `fixture` (invented
numbers in the real format, under a directory holding a `FIXTURE` marker file). Pages
carry the origin of what they show. Recorded absolute paths that moved can be mapped with
`relocate` (old prefix to new prefix).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

__all__ = [
    "FIXTURE_MARKER",
    "Arm",
    "EvaluationView",
    "Experiment",
    "LearnedView",
    "Origin",
    "RoundView",
    "RunView",
    "Sources",
    "TruthView",
    "load_engagement",
    "load_investigation",
    "load_manifest",
    "load_run",
]

Origin = Literal["real", "toy", "fixture"]
FIXTURE_MARKER = "FIXTURE"  # a file of this name marks its directory tree as invented
_RANK: dict[str, int] = {"real": 0, "toy": 1, "fixture": 2}


def worst(origins: Iterable[Origin]) -> Origin:
    """The least real of some origins (a table mixing real and fixture rows is a fixture)."""
    out: Origin = "real"
    for origin in origins:
        if _RANK[origin] > _RANK[out]:
            out = origin
    return out


@dataclass
class FileRecord:
    path: str
    sha256: str
    bytes: int
    role: str
    origin: Origin


@dataclass
class Sources:
    """Every file the exhibit read, with its digest and origin; the base of provenance."""

    relocate: dict[str, str] = field(default_factory=dict)
    default_origin: Origin = "real"
    files: dict[str, FileRecord] = field(default_factory=dict)
    _tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    def path(self, recorded: str | Path) -> Path:
        """A recorded path, mapped by `relocate` when it moved."""
        text = str(recorded)
        for old, new in self.relocate.items():
            if text == old or text.startswith(old.rstrip("/") + "/"):
                text = new.rstrip("/") + text[len(old.rstrip("/")) :]
                break
        return Path(text)

    def origin_of(self, path: Path, origin: Origin | None = None) -> Origin:
        for parent in [path, *path.parents]:
            if (parent / FIXTURE_MARKER).is_file():
                return "fixture"
        return origin if origin is not None else self.default_origin

    def register(self, path: Path, role: str, origin: Origin | None = None) -> str:
        path = path.resolve()
        key = str(path)
        if key not in self.files:
            data = path.read_bytes()
            self.files[key] = FileRecord(
                key,
                hashlib.sha256(data).hexdigest(),
                len(data),
                role,
                self.origin_of(path, origin),
            )
        return key

    def json(self, path: Path, role: str, origin: Origin | None = None) -> Any:
        self.register(path, role, origin)
        return json.loads(path.read_text())

    def lines(self, path: Path, role: str, origin: Origin | None = None) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        self.register(path, role, origin)
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def text(self, path: Path, role: str, origin: Origin | None = None) -> str | None:
        if not path.is_file():
            return None
        self.register(path, role, origin)
        return path.read_text()

    def parquet(self, path: Path, role: str, origin: Origin | None = None) -> list[dict[str, Any]]:
        key = str(path.resolve())
        if key not in self._tables:
            self.register(path, role, origin)
            rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
            self._tables[key] = rows
        return self._tables[key]

    def origin(self, keys: Iterable[str]) -> Origin:
        return worst(self.files[k].origin for k in keys if k in self.files)


# ---- the normalized view ------------------------------------------------------------------


@dataclass
class Arm:
    index: int
    actions: list[dict[str, Any]]
    episode_ids: list[str]
    rows: list[dict[str, Any] | None]  # the episodes rows, None where no export holds one
    exports: list[Path | None] = field(default_factory=list)  # the export holding each row

    @property
    def outcomes(self) -> list[bool | None]:
        return [None if r is None else r.get("outcome") == "success" for r in self.rows]


@dataclass
class Experiment:
    experiment_id: str
    round: int
    session: int
    request: dict[str, Any]
    status: str  # done, refused, queued, or "no result" when none was recorded
    message: str
    arms: list[Arm]
    files: list[str]

    @property
    def robots(self) -> list[str]:
        return [str(r) for r in self.request.get("robots", [])]

    @property
    def paired(self) -> bool:
        return bool(self.request.get("paired_starts"))


@dataclass
class LearnedView:
    """A learned skill as the exhibit shows it. `round` and `session` are None where the
    records do not say them; a skill is never put on every round, nor given session 0."""

    name: str
    round: int | None
    session: int | None
    text: str  # the SKILL.md, verbatim
    frozen: bool  # from a frozen library (else rendered from the agent's stored draft)
    files: list[str]
    analysis: str | None = None  # the analysis that wrote it (a frozen skill's source)
    provenance: str | None = None  # the host's provenance line of a frozen skill
    carried_from: str | None = None  # the earlier library a frozen skill was carried from


@dataclass
class RoundView:
    number: int
    origin: Origin
    operation_days: tuple[int, int]
    ops_rows: list[dict[str, Any]]  # the round's regular work (no experiment id, its days)
    ops_files: list[str]
    sessions: list[int]
    diagnoses: list[tuple[int, dict[str, Any]]]  # (session, Diagnosis) in submission order
    experiments: list[Experiment]
    decision: dict[str, Any] | None  # RoundDecision, None in an older investigation
    fix_set: dict[str, Any] | None  # an older investigation's ranked list
    ledger_after: dict[str, Any] | None  # the public ledger after the decision
    notes: list[tuple[str, str]]  # (when, text): the agent's notes, verbatim
    costs: dict[str, Any]
    cost_files: list[str]
    decision_files: list[str]
    stopped: str | None
    learned: list[LearnedView] = field(default_factory=list)
    session_records: list[dict[str, Any]] = field(default_factory=list)
    analysis: Path | None = None  # the analysis directory the round's sessions wrote


@dataclass
class EvaluationView:
    root: Path
    summary: dict[str, Any]
    attempts: list[dict[str, Any]]  # attempts.parquet rows (state, episode, robot, success)
    files: list[str]
    one_look_list: list[dict[str, Any]] | None
    one_look_ledger: dict[str, Any] | None
    one_look_files: list[str]
    seconds: float | None
    origin: Origin = "real"  # the origin of the run it belongs to (state videos inherit it)


@dataclass
class ComparatorView:
    """Another method measured on the same reserved attempts (the plan's comparisons): a
    plain search (`plain_search.json`: its search, choice, adopted actions, its own
    evaluation and its normalized cost receipt), or a method whose evaluation is not
    there yet (`pending`, shown as a marked placeholder)."""

    name: str
    label: str  # what the method is and what it was told, in plain words
    kind: str  # plain_search or pending
    status: str  # for a pending one: why it is not shown yet
    record: dict[str, Any] | None
    files: list[str]
    attempts: list[dict[str, Any]] = field(default_factory=list)  # its evaluation's rows

    @property
    def evaluation(self) -> dict[str, Any] | None:
        found = (self.record or {}).get("evaluation")
        return found if isinstance(found, dict) else None

    @property
    def costs(self) -> dict[str, Any] | None:
        found = (self.record or {}).get("costs")
        return found if isinstance(found, dict) else None


@dataclass
class TruthView:
    world_id: str
    root: Path
    faults: list[dict[str, Any]]
    oracle_gains: list[dict[str, Any]] | None
    oracle_scoring_episodes: int | None
    files: list[str]
    kind: str | None = None  # the registry's kind of the world (dashboard/worlds.json)


@dataclass
class RunView:
    label: str
    world_id: str
    kind: str  # engagement or investigation
    agent: dict[str, Any]
    budget: dict[str, Any]
    rounds: list[RoundView]
    sources: Sources
    record_files: list[str]
    ended_because: str | None = None
    evaluation: EvaluationView | None = None
    reuse: dict[str, Any] | None = None
    reuse_files: list[str] = field(default_factory=list)
    truth: TruthView | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
    annotation_files: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    # Learned skills the records tie to no round of this run (round not recorded, or
    # written by another investigation and carried in a library): shown on the summary.
    unplaced_learned: list[LearnedView] = field(default_factory=list)
    learned_record: dict[str, Any] | None = None  # the engagement's `learned` entry
    # The driver's count of every simulation the loop ran, by purpose (`driver.json`
    # beside the engagement directory: operation, experiments, probes, evaluation).
    simulated_totals: dict[str, Any] | None = None
    simulated_files: list[str] = field(default_factory=list)
    # The review-site artifact this bundle was published as (the manifest's `hub_id`): written as
    # the summary page's front-matter `id:` so that every re-render keeps the identity.
    hub_id: str | None = None
    comparators: list[ComparatorView] = field(default_factory=list)
    learned_files: list[str] = field(default_factory=list)
    one_look_counts: dict[str, Any] | None = None  # the engagement's `one_look` entry
    evaluation_skipped: str | None = None
    outcome: str | None = None

    @property
    def origin(self) -> Origin:
        parts: list[Origin] = [r.origin for r in self.rounds]
        if self.evaluation is not None:
            parts.append(self.sources.origin(self.evaluation.files))
        if self.reuse is not None:
            parts.append(self.sources.origin(self.reuse_files))
        return worst(parts)


# ---- episodes in exports ------------------------------------------------------------------


class Episodes:
    """The episodes rows of every export the run names, by episode id (latest export wins)."""

    def __init__(self, sources: Sources, exports: Sequence[Path], origin: Origin) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        self.where: dict[str, Path] = {}
        self.file_of: dict[str, str] = {}
        for export in exports:
            table = export / "log" / "episodes.parquet"
            if not table.is_file():
                continue
            rows = sources.parquet(table, "episodes of an export", origin)
            key = str(table.resolve())
            for row in rows:
                eid = str(row["episode_id"])
                # A later export wins, unless only the earlier one holds the episode's video.
                if (
                    eid in self.where
                    and not self._has_video(export, row)
                    and self._has_video(self.where[eid], self.by_id[eid])
                ):
                    continue
                self.by_id[eid] = row
                self.where[eid] = export
                self.file_of[eid] = key

    @staticmethod
    def _has_video(export: Path, row: Mapping[str, Any]) -> bool:
        video = row.get("video_scene")
        return bool(video) and (export / "log" / str(video)).is_file()

    def rows(self, ids: Sequence[str]) -> list[dict[str, Any] | None]:
        return [self.by_id.get(i) for i in ids]

    def files(self, ids: Iterable[str]) -> list[str]:
        return sorted({self.file_of[i] for i in ids if i in self.file_of})


def _unique(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen and p.exists():
            seen.add(key)
            out.append(p)
    return out


def _session_files(analysis: Path) -> dict[int, Path]:
    out = {}
    for path in sorted((analysis / "sessions").glob("session-*.json")):
        try:
            out[int(path.stem.split("-")[1])] = path
        except (IndexError, ValueError):
            continue
    return out


def _exports_of(
    sources: Sources, analysis: Path, extra: Sequence[Path], origin: Origin
) -> list[Path]:
    """Every export the analysis' sessions and queues name, oldest first, then `extra`."""
    found: list[Path] = list(extra)
    for number, path in sorted(_session_files(analysis).items()):
        record = sources.json(path, "session record", origin)
        if record.get("export"):
            found.append(sources.path(record["export"]))
        del number
    for reply in sorted((analysis / "queues").glob("session-*.reply.json")):
        data = sources.json(reply, "queue reply", origin)
        if data.get("export"):
            found.append(sources.path(data["export"]))
    return _unique(found)


def _experiments(
    sources: Sources,
    analysis: Path,
    episodes: Episodes,
    round_of: Mapping[int, int],
    origin: Origin,
) -> list[Experiment]:
    requests_file = analysis / "outbox" / "experiments.jsonl"
    results_file = analysis / "results" / "experiments.jsonl"
    requests = sources.lines(requests_file, "experiment requests", origin)
    results = {
        str(r["experiment_id"]): r
        for r in sources.lines(results_file, "experiment results", origin)
    }
    out = []
    for queued in requests:
        eid = str(queued["experiment_id"])
        session = int(queued.get("session", 0))
        request = queued["request"]
        result = results.get(eid)
        arms_out = []
        by_index = {int(a["arm_index"]): a for a in (result or {}).get("arms", [])}
        for index, arm in enumerate(request.get("arms", [])):
            ids = [str(e) for e in by_index.get(index, {}).get("episode_ids", [])]
            arms_out.append(
                Arm(
                    index,
                    list(arm.get("actions", [])),
                    ids,
                    episodes.rows(ids),
                    [episodes.where.get(i) for i in ids],
                )
            )
        files = [str(requests_file.resolve())]
        if result is not None:
            files.append(str(results_file.resolve()))
        files += episodes.files(i for a in arms_out for i in a.episode_ids)
        out.append(
            Experiment(
                experiment_id=eid,
                round=round_of.get(session, 0),
                session=session,
                request=request,
                status=str(result["status"]) if result else "no result",
                message=str(result.get("message", "")) if result else "",
                arms=arms_out,
                files=files,
            )
        )
    return out


def _notes(
    sources: Sources, analysis: Path, sessions: Sequence[int], last: bool, origin: Origin
) -> list[tuple[str, str]]:
    """The agent's notes as they stood after each of `sessions`: the snapshot taken before
    the next session, or the final notes after the analysis' last session."""
    out = []
    for number in sessions:
        before_next = analysis / "session-state" / f"session-{number + 1:02d}" / "before"
        text = sources.text(before_next / "memory" / "notes.md", "agent notes", origin)
        if text is None and last and number == max(sessions):
            text = sources.text(analysis / "memory" / "notes.md", "agent notes", origin)
        if text is not None:
            out.append((f"after session {number}", text))
    return out


def _learned(
    sources: Sources, analysis: Path, round_of: Mapping[int, int], origin: Origin
) -> list[LearnedView]:
    from fleetlab.contracts.learned import LearnedSkillDraft, render_learned_skill
    from fleetlab.toolserver.files import LEARNED_FILE

    path = analysis / "outbox" / LEARNED_FILE
    latest: dict[str, tuple[int | None, LearnedSkillDraft]] = {}
    for record in sources.lines(path, "learned skill drafts", origin):
        draft = LearnedSkillDraft.model_validate(record["draft"])
        session = record.get("session")
        latest[draft.name] = (int(session) if isinstance(session, int) else None, draft)
    out = []
    for name, (session, draft) in latest.items():
        out.append(
            LearnedView(
                name=name,
                round=round_of.get(session) if session is not None else None,
                session=session,
                text=render_learned_skill(draft),
                frozen=False,
                files=[str(path.resolve())],
                analysis=str(analysis.resolve()),
            )
        )
    return out


def _place(view: RunView, skills: Sequence[LearnedView], selected: Sequence[int] | None) -> None:
    """`attach_learned`, leaving out skills whose recorded round is not among the rounds
    this view was loaded for (a manifest loads an engagement one round at a time)."""
    if selected is not None:
        skills = [s for s in skills if s.round is None or s.round in selected]
    attach_learned(view, skills)


def _library(sources: Sources, root: Path, origin: Origin) -> list[LearnedView]:
    """A frozen learned library's skills, verbatim (`library.json` and
    `skills/<name>/SKILL.md`), with the provenance the freeze recorded under each skill's
    `source` (analysis, round, session; lane learn's `Source.round_of`). What the record
    does not hold stays None."""
    record_file = root / "library.json"
    record = sources.json(record_file, "learned library record", origin)
    out = []
    for skill in record.get("skills", []):
        name = str(skill["name"])
        path = root / "skills" / name / "SKILL.md"
        text = sources.text(path, "learned skill, frozen", origin) or ""
        source = skill.get("source") or {}
        round_ = source.get("round")
        session = source.get("session")
        analysis = source.get("analysis")
        out.append(
            LearnedView(
                name=name,
                round=int(round_) if isinstance(round_, int) else None,
                session=int(session) if isinstance(session, int) else None,
                text=text,
                frozen=True,
                files=[str(record_file.resolve()), str(path.resolve())],
                analysis=str(sources.path(analysis).resolve()) if analysis else None,
                provenance=skill.get("provenance"),
                carried_from=skill.get("carried_from"),
            )
        )
    return out


def attach_learned(view: RunView, skills: Sequence[LearnedView]) -> None:
    """Put each skill on the page of the round that wrote it: a skill of one of this
    run's analyses goes to its recorded round, else to the round whose sessions include
    its recorded session. A skill the records tie to no round of this run (no round, no
    session, or another investigation's) goes to `view.unplaced_learned`. A frozen skill
    replaces the draft of the same name."""
    for skill in skills:
        ours = [
            r
            for r in view.rounds
            if skill.analysis is None
            or (r.analysis is not None and str(r.analysis.resolve()) == skill.analysis)
        ]
        number = None
        if skill.analysis is not None and ours and skill.carried_from is None:
            if skill.round is not None and any(r.number == skill.round for r in ours):
                number = skill.round
            elif skill.session is not None:
                number = next((r.number for r in ours if skill.session in r.sessions), None)
        for rnd in view.rounds:
            rnd.learned = [x for x in rnd.learned if x.name != skill.name]
        view.unplaced_learned = [x for x in view.unplaced_learned if x.name != skill.name]
        if number is None:
            view.unplaced_learned.append(skill)
        else:
            skill.round = number
            next(r for r in view.rounds if r.number == number).learned.append(skill)


def _diagnoses(
    sources: Sources, analysis: Path, origin: Origin
) -> tuple[list[dict[str, Any]], str]:
    path = analysis / "outbox" / "submissions.jsonl"
    return sources.lines(path, "agent submissions", origin), str(path.resolve())


def _session_records(
    sources: Sources, analysis: Path, numbers: Sequence[int], origin: Origin
) -> list[dict[str, Any]]:
    files = _session_files(analysis)
    return [sources.json(files[n], "session record", origin) for n in numbers if n in files]


def _ops(
    sources: Sources, export: Path, days: tuple[int, int] | None, origin: Origin
) -> tuple[list[dict[str, Any]], list[str], tuple[int, int]]:
    table = export / "log" / "episodes.parquet"
    rows = sources.parquet(table, "episodes of the round's export", origin)
    regular = [r for r in rows if r.get("experiment_id") is None]
    if days is None:
        found = sorted({int(r["day"]) for r in regular})
        days = (found[0], found[-1]) if found else (0, -1)
    first, last = days
    ops = [r for r in regular if first <= int(r["day"]) <= last]
    return sorted(ops, key=lambda r: str(r["episode_id"])), [str(table.resolve())], days


def _starts_without_usage(records: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for record in records:
        for start in record.get("attempts") or []:
            meta = start.get("runtime_meta") or {}
            if not start.get("usage") or meta.get("interrupted_without_result"):
                count += 1
    return count


# ---- loaders ------------------------------------------------------------------------------


def _agent_origin(agent: Mapping[str, Any]) -> Origin:
    return "toy" if str(agent.get("model")) == "scripted" else "real"


def load_engagement(
    path: Path | str,
    *,
    sources: Sources | None = None,
    rounds: Sequence[int] | None = None,
) -> RunView:
    """An engagement record (`engagement.json`) with everything it names."""
    sources = sources if sources is not None else Sources()
    path = Path(path).resolve()
    record = sources.json(path, "engagement record")
    origin = sources.origin_of(path, _agent_origin(record.get("agent", {})))
    sources.files[str(path)].origin = origin
    analysis = sources.path(record["analysis"])
    round_of: dict[int, int] = {}
    for entry in record.get("rounds", []):
        for s in entry.get("sessions", []):
            round_of[int(s["number"])] = int(entry["round"])
    round_exports = [sources.path(e["export"]["path"]) for e in record["rounds"] if e.get("export")]
    episodes = Episodes(sources, _exports_of(sources, analysis, round_exports, origin), origin)
    experiments = _experiments(sources, analysis, episodes, round_of, origin)
    submissions, submissions_file = _diagnoses(sources, analysis, origin)
    learned = _learned(sources, analysis, round_of, origin)
    last_round = max((int(e["round"]) for e in record["rounds"]), default=0)
    views = []
    for entry in record.get("rounds", []):
        number = int(entry["round"])
        if rounds is not None and number not in rounds:
            continue
        sessions = [int(s["number"]) for s in entry.get("sessions", [])]
        export = sources.path(entry["export"]["path"])
        days = tuple(entry["operation"]["days"])
        ops, ops_files, days_ = _ops(sources, export, (int(days[0]), int(days[1])), origin)
        views.append(
            RoundView(
                number=number,
                origin=origin,
                operation_days=days_,
                ops_rows=ops,
                ops_files=ops_files,
                sessions=sessions,
                diagnoses=[
                    (int(s["session"]), s["answer"])
                    for s in submissions
                    if s.get("kind") == "diagnosis" and int(s.get("session", 0)) in sessions
                ],
                experiments=[e for e in experiments if e.round == number],
                decision=entry.get("decision"),
                fix_set=None,
                ledger_after=entry.get("ledger"),
                notes=_notes(sources, analysis, sessions, number == last_round, origin),
                costs=dict(entry.get("costs") or {}),
                cost_files=[str(path)],
                decision_files=[str(path), submissions_file],
                stopped=entry.get("stopped"),
                session_records=_session_records(sources, analysis, sessions, origin),
                analysis=analysis,
            )
        )
    view = RunView(
        label=str(record.get("label", "engagement")),
        world_id=str(record.get("world_id", "")),
        kind="engagement",
        agent=dict(record.get("agent", {})),
        budget=dict(record.get("budget", {})),
        rounds=views,
        sources=sources,
        record_files=[str(path)],
        ended_because=record.get("ended_because"),
        started_at=record.get("started_at"),
        ended_at=record.get("ended_at"),
    )
    _place(view, learned, rounds)
    learned_record = record.get("learned")
    if isinstance(learned_record, dict):
        view.learned_record = learned_record
        view.learned_files = [str(path)]
        if learned_record.get("library"):
            library = sources.path(learned_record["library"])
            if (library / "library.json").is_file():
                frozen = _library(sources, library, origin)
                # The driver's own record of each kept skill's round, where it has one.
                recorded = learned_record.get("rounds") or {}
                for skill in frozen:
                    value = recorded.get(skill.name)
                    if (
                        skill.round is None
                        and isinstance(value, int)
                        and skill.analysis == str(analysis.resolve())
                    ):
                        skill.round = value
                _place(view, frozen, rounds)
    if isinstance(record.get("one_look"), dict):
        view.one_look_counts = dict(record["one_look"])
    driver_file = path.parent.parent / "driver.json"
    if driver_file.is_file():
        driver = sources.json(driver_file, "driver record (simulations by purpose)", origin)
        if isinstance(driver.get("simulated"), dict):
            view.simulated_totals = dict(driver["simulated"])
            view.simulated_files = [str(driver_file.resolve())]
    view.outcome = record.get("outcome")
    evaluation = record.get("evaluation") or {}
    if evaluation.get("skipped"):
        view.evaluation_skipped = str(evaluation["skipped"])
    if evaluation.get("path"):
        eval_dir = sources.path(evaluation["path"])
        if (eval_dir / "summary.json").is_file():
            view.evaluation = load_evaluation(
                sources,
                eval_dir,
                one_look_list=path.parent / "one-look-list.json",
                seconds=evaluation.get("seconds"),
                origin=origin,
            )
    return view


def load_investigation(path: Path | str, *, sources: Sources | None = None) -> RunView:
    """An older single-round investigation (`run.json`, kind `investigation`) as round 1:
    no round decision; its ranked fix set and diagnosis are shown as submitted."""
    sources = sources if sources is not None else Sources()
    path = Path(path).resolve()
    record = sources.json(path, "investigation record")
    if record.get("kind") != "investigation":
        raise ValueError(f"{path}: not an investigation record")
    origin = sources.origin_of(path, _agent_origin(record.get("agent", {})))
    analysis = sources.path(record["analysis"])
    sessions = [int(s["number"]) for s in record.get("sessions", [])]
    session_records = _session_records(sources, analysis, sessions, origin)
    first_export = (
        sources.path(session_records[0]["export"]) if session_records else path.parent / "export"
    )
    episodes = Episodes(sources, _exports_of(sources, analysis, [first_export], origin), origin)
    round_of = dict.fromkeys(sessions, 1)
    submissions, submissions_file = _diagnoses(sources, analysis, origin)
    fix_sets = [s["answer"] for s in submissions if s.get("kind") == "fix_set"]
    ops, ops_files, days = _ops(sources, first_export, None, origin)
    costs = dict(record.get("costs") or {})
    costs.setdefault("starts_without_usage", _starts_without_usage(session_records))
    costs.setdefault("per_session", [])
    round_view = RoundView(
        number=1,
        origin=origin,
        operation_days=days,
        ops_rows=ops,
        ops_files=ops_files,
        sessions=sessions,
        diagnoses=[
            (int(s["session"]), s["answer"]) for s in submissions if s.get("kind") == "diagnosis"
        ],
        experiments=_experiments(sources, analysis, episodes, round_of, origin),
        decision=None,
        fix_set=fix_sets[-1] if fix_sets else None,
        ledger_after=None,
        notes=_notes(sources, analysis, sessions, True, origin),
        costs=costs,
        cost_files=[str(path)],
        decision_files=[submissions_file],
        stopped=None,
        session_records=session_records,
        analysis=analysis,
    )
    view = RunView(
        label=str(record.get("label", "investigation")),
        world_id=str(record.get("world_id") or _world_of(analysis, sources, origin)),
        kind="investigation",
        agent=dict(record.get("agent", {})),
        budget=dict(record.get("budget", {})),
        rounds=[round_view],
        sources=sources,
        record_files=[str(path)],
        ended_because=None,
        started_at=record.get("started_at"),
        ended_at=record.get("ended_at"),
    )
    attach_learned(view, _learned(sources, analysis, round_of, origin))
    return view


def _world_of(analysis: Path, sources: Sources, origin: Origin) -> str:
    meta = analysis / "analysis.json"
    if meta.is_file():
        return str(sources.json(meta, "analysis record", origin).get("world_id", ""))
    return ""


def load_evaluation(
    sources: Sources,
    root: Path,
    *,
    one_look_list: Path | None = None,
    seconds: float | None = None,
    origin: Origin | None = None,
) -> EvaluationView:
    """Lane state's reserved evaluation: `summary.json`, `attempts.parquet`, the one-look
    plan's list and ledger when present."""
    summary = sources.json(root / "summary.json", "evaluation summary", origin)
    files = [str((root / "summary.json").resolve())]
    attempts: list[dict[str, Any]] = []
    if (root / "attempts.parquet").is_file():
        attempts = sources.parquet(root / "attempts.parquet", "evaluation attempts", origin)
        files.append(str((root / "attempts.parquet").resolve()))
    if (root / "panel.json").is_file():
        sources.register(root / "panel.json", "evaluation panel", origin)
        files.append(str((root / "panel.json").resolve()))
    look: list[dict[str, Any]] | None = None
    look_files: list[str] = []
    if one_look_list is not None and one_look_list.is_file():
        look = sources.json(one_look_list, "one-look list (frozen after round 1)", origin)
        look_files.append(str(one_look_list.resolve()))
    ledger = None
    ledger_file = root / "one-look.ledger.json"
    if ledger_file.is_file():
        ledger = sources.json(ledger_file, "one-look ledger", origin)
        look_files.append(str(ledger_file.resolve()))
    return EvaluationView(
        root, summary, attempts, files, look, ledger, look_files, seconds, origin or "real"
    )


def load_truth(sources: Sources, root: Path, world_id: str) -> TruthView:
    """The planted faults of a development world (its `world_truth.json`) and, when the
    world keeps one, its oracle's per-fault gains. Callers check that the world's truth may
    be shown (`fleetlab.dashboard.data.WorldRegistry.truth_allowed`)."""
    lowered = str(root).lower()
    if "unseen" in lowered or "held" in lowered.split("/")[-1]:
        raise ValueError(f"{root}: refusing to read a sealed world's truth")
    files_dir = root / "world" if (root / "world").is_dir() else root / "files"
    truth = sources.json(files_dir / "world_truth.json", "planted truth (world)")
    if str(truth.get("world_id")) != world_id:
        raise ValueError(f"{root} holds world {truth.get('world_id')}, not {world_id}")
    files = [str((files_dir / "world_truth.json").resolve())]
    gains = None
    scoring = None
    oracle = root / "oracle" / "oracle.json"
    if oracle.is_file():
        data = sources.json(oracle, "oracle of the world's stored scoring window")
        gains = list(data.get("gains", []))
        scoring = data.get("scoring_episodes")
        files.append(str(oracle.resolve()))
    return TruthView(world_id, root, list(truth.get("fault_schedule", [])), gains, scoring, files)


def load_run(path: Path | str, *, sources: Sources | None = None) -> RunView:
    """An engagement, an investigation or a manifest, by what the file holds."""
    path = Path(path)
    if path.is_dir():
        for name in ("exhibit.json", "engagement.json", "run.json"):
            if (path / name).is_file():
                path = path / name
                break
    data = json.loads(path.read_text())
    if data.get("format") == MANIFEST_FORMAT:
        return load_manifest(path, sources=sources)
    if data.get("kind") == "engagement":
        return load_engagement(path, sources=sources)
    if data.get("kind") == "investigation":
        return load_investigation(path, sources=sources)
    raise ValueError(f"{path}: not an engagement, investigation or exhibit manifest")


MANIFEST_FORMAT = "fleetlab.exhibit-manifest/1"


def attempts_problems(
    record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    attempts_file: Path,
    sources: Sources | None = None,
) -> list[str]:
    """Why a comparator's attempts table does not belong to its record's evaluation: every
    state the summary lists must have the summary's attempts and successes in the table;
    the table's states must be the summary's; its panel file, where one lies beside the
    table, must have the summary's fingerprint. Empty when they agree."""
    summary = record.get("evaluation") or {}
    problems = []
    states = {str(s["name"]): s for s in summary.get("states", [])}
    found: dict[str, list[bool]] = {}
    for row in attempts:
        found.setdefault(str(row.get("state")), []).append(bool(row.get("success")))
    # A state with no attempt has no row; any other summary state must be in the table.
    expected = {n for n, st in states.items() if st.get("attempts") != 0}
    if not expected <= set(found) or not set(found) <= set(states):
        problems.append(f"states {sorted(found)} in the table, {sorted(states)} in the summary")
    for name, state in states.items():
        values = found.get(name, [])
        recorded_n, recorded_ok = state.get("attempts"), state.get("successes")
        if (
            recorded_n is None
            or recorded_ok is None
            or len(values) != int(recorded_n)
            or sum(values) != int(recorded_ok)
        ):
            problems.append(
                f"state {name}: {sum(values)}/{len(values)} in the table, "
                f"{state.get('successes')}/{state.get('attempts')} in the summary"
            )
    panel_file = attempts_file.parent / "panel.json"
    if panel_file.is_file() and summary.get("panel_sha256"):
        if sources is not None:
            # Read through Sources, so the validation input is listed in the provenance.
            digest = sources.files[sources.register(panel_file, "plain search panel")].sha256
        else:
            digest = hashlib.sha256(panel_file.read_bytes()).hexdigest()
        if digest != summary["panel_sha256"]:
            problems.append("the panel beside the table has another fingerprint")
    return problems


def load_manifest(path: Path | str, *, sources: Sources | None = None) -> RunView:
    """A run assembled from parts (`exhibit.json`):

    ```
    {"format": "fleetlab.exhibit-manifest/1", "label": "...",
     "relocate": {"/old/prefix": "/new/prefix"},
     "rounds": [{"investigation": "run.json"} | {"engagement": "engagement.json", "round": 2}],
     "evaluation": "evaluation dir", "one_look_list": "one-look-list.json",
     "reuse": "comparison.json", "library": "learned library dir",
     "truth_world": "world root", "annotations": "annotations.json",
     "hub_id": "the review-site artifact id once published (written as index.md's front matter)"}
    ```

    Relative paths are relative to the manifest. Each part keeps its own origin.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    sources = sources if sources is not None else Sources()
    sources.relocate = {**data.get("relocate", {}), **sources.relocate}
    sources.register(path, "exhibit manifest")
    base = path.parent

    def at(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    view: RunView | None = None
    for part in data.get("rounds", []):
        if "investigation" in part:
            piece = load_investigation(at(part["investigation"]), sources=sources)
        else:
            piece = load_engagement(
                at(part["engagement"]), sources=sources, rounds=[int(part["round"])]
            )
        if view is None:
            view = piece
        else:
            view.rounds += piece.rounds
            view.record_files += piece.record_files
            view.ended_because = piece.ended_because or view.ended_because
            if piece.evaluation is not None:
                view.evaluation = piece.evaluation
            placed = {(x.name, x.analysis) for r in view.rounds for x in r.learned}
            seen = {(x.name, x.analysis) for x in view.unplaced_learned}
            view.unplaced_learned += [
                x
                for x in piece.unplaced_learned
                if (x.name, x.analysis) not in placed and (x.name, x.analysis) not in seen
            ]
            view.unplaced_learned = [
                x for x in view.unplaced_learned if (x.name, x.analysis) not in placed
            ]
            for key in (
                "learned_record",
                "one_look_counts",
                "evaluation_skipped",
                "outcome",
                "simulated_totals",
            ):
                if getattr(piece, key) is not None:
                    setattr(view, key, getattr(piece, key))
            view.learned_files += piece.learned_files
            view.simulated_files = sorted({*view.simulated_files, *piece.simulated_files})
    if view is None:
        raise ValueError(f"{path}: a manifest names at least one round")
    view.label = str(data.get("label", view.label))
    if data.get("hub_id") is not None:
        hub_id = str(data["hub_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", hub_id):
            raise ValueError(f"{path}: hub_id {hub_id!r} is not a review-site artifact id")
        view.hub_id = hub_id
    view.record_files.insert(0, str(path))
    if data.get("evaluation"):
        look = at(data["one_look_list"]) if data.get("one_look_list") else None
        view.evaluation = load_evaluation(sources, at(data["evaluation"]), one_look_list=look)
    if data.get("reuse"):
        reuse_file = at(data["reuse"])
        view.reuse = sources.json(reuse_file, "reuse comparison")
        view.reuse_files = [str(reuse_file)]
    if data.get("library"):
        attach_learned(view, _library(sources, at(data["library"]), "real"))
    for entry in data.get("comparators") or []:
        kind = str(entry.get("kind", "pending"))
        record = None
        files: list[str] = []
        attempts: list[dict[str, Any]] = []
        if kind == "plain_search":
            record_file = at(entry["record"])
            record = sources.json(record_file, "plain search record (choice, evaluation, costs)")
            files = [str(record_file)]
            if entry.get("attempts"):
                attempts_file = at(entry["attempts"])
                attempts = sources.parquet(attempts_file, "plain search evaluation attempts")
                files.append(str(attempts_file))
                problems = attempts_problems(record, attempts, attempts_file, sources)
                if (attempts_file.parent / "panel.json").is_file():
                    files.append(str((attempts_file.parent / "panel.json").resolve()))
                if problems:
                    raise ValueError(
                        f"{path}: comparator {entry['name']}: its attempts do not belong to its "
                        "evaluation: " + "; ".join(problems)
                    )
        elif kind != "pending":
            raise ValueError(f"{path}: comparator kind {kind!r} is not known")
        view.comparators.append(
            ComparatorView(
                name=str(entry["name"]),
                label=str(entry.get("label", entry["name"])),
                kind=kind,
                status=str(entry.get("status", "")),
                record=record,
                files=files,
                attempts=attempts,
            )
        )
    if data.get("truth_world"):
        view.truth = load_truth(sources, at(data["truth_world"]), view.world_id)
    if data.get("annotations"):
        ann_file = at(data["annotations"])
        view.annotations = sources.json(ann_file, "reviewers' annotations")
        view.annotation_files = [str(ann_file)]
    return view
