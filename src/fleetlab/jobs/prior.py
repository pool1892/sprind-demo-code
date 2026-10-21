"""What an investigation may start from: its pool, its exposure and its predecessors.

Host side only (it derives seeds, `jobs.starts`). Before the first session the driver
checks the request's allocation against everything earlier attempts examined
(`preflight`):

- **The pool.** A request with an allocation profile names its pool (its base task) in
  the host's assignment record, read from an explicit file (`--assignment`, format
  `fleetlab.coordinator-cohorts/1`, the coordinator's): the cohort's base task and profile
  are the request's, its status is `reserved` or `active`, its request digest (when the
  record names one) is the request file's, and no other job line and no historical entry
  holds the same pool. A job line keeps its pool across numbered investigations until an
  evaluation is attempted on it; the record says so, the exposure below enforces it.
- **The mapping.** Every public start, a base task and an initial state, means one full
  identity across everything the investigation can see and run: the identities examined
  before (each read under its own recorded profile) and the request's own allocation for
  its pool, initial states 0 to 49. A pair that two of them give different seeds is
  refused.
- **The exposure.** None of the request's evaluation identities (its pool's initial states
  30 to 49 under its profile) may have been examined by any earlier attempt, for any
  request, finished or not.

The examined identities come from the release exposure logs (`jobs.library`) of the
libraries the investigation carries, the mounted library's and its predecessors', and from
the imported episodes that ran on evaluation starts.

**The import** (`read_prior`, `--prior-run DIR`, repeatable). A numbered investigation
starts from the public records its predecessors left: for each earlier investigation
directory (its `job_run.json`), the completed episodes of its development batches and of
its final evaluation (the release run: the saved program, j0001 there, and the naive chain,
j0002), interrupted batches included with their status; an episode the run's
`job_episodes` table does not hold never finished and is not imported. Imported are their
five job tables and original videos, their programs, the checks they wrote or saved, their
notes and accounts, the names they gave their jobs and checks, and their exposure lines;
never a run record, `truth/`, a batch file, the host's own records or anything outside the
investigation's directory. Every episode is checked against its run's own record: its job's
program digest, its place among its job's starts, and its start, which must be the one the
run's own allocation profile gives (the legacy rule when the run recorded none); a release
receipt's pairs against the run's episodes; an exposure line's identities against its own
profile, and a legacy line's, when it names the investigation's evaluation run, against the
receipt. The exposure an investigation carried from its start (`prior/exposure`, its
mounted library's lines and its own predecessors') is read too, so that what its ancestors
examined stays examined even when it was broken off before it froze a library.

Investigation i (in the order given, at most 89) gets job numbers 1000 + 100 (i - 1) + n:
its development job jNNNN keeps n = NN (at most 97), its final evaluation's program 98 and
its naive chain 99; an episode keeps its last two digits, its place in its job
(`toolserver.jobs.FIRST_IMPORTED_JOB`). Every row of all five tables, the file names and the
video columns get the new ids; nothing else of a row changes. `Prior.write` writes the
remapped tree (`history/`, a job-run tree the export merges like a batch's), the host's
registry (`registry.json`: sources, file digests, every job and episode with its own id and
full start identity), the carried exposure lines (`exposure/`) and the public history the
driver mounts (`toolserver.jobs.PublicHistory`), which it checks against the tree as the tools
will; and it verifies that every source file it read is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.env import Camera
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.jobs import (
    DEVELOPMENT_INIT_STATES,
    RELEASE_INIT_STATES,
    CheckSpec,
    JobEpisode,
    JobProgram,
    ReleaseReceipt,
    RequestState,
    StartIdentity,
)
from fleetlab.jobs import library, tables
from fleetlab.jobs.starts import check_profile, release_starts_for, start_for, starts_for

__all__ = [
    "ASSIGNMENT_FORMAT",
    "Exposure",
    "ImportRefused",
    "PreflightError",
    "Prior",
    "assignment_problems",
    "evaluation_identities",
    "exposure_of",
    "mapping_problems",
    "preflight",
    "read_prior",
]

ASSIGNMENT_FORMAT = "fleetlab.coordinator-cohorts/1"
OPEN_STATUSES = ("reserved", "active")


class PreflightError(ValueError):
    """An investigation the host refuses to start, with every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass
class Exposure:
    """Start identities examined before, each with where it was recorded."""

    identities: dict[StartIdentity, list[str]] = field(default_factory=dict)

    def add(self, identity: StartIdentity, where: str) -> None:
        places = self.identities.setdefault(identity, [])
        if where not in places:
            places.append(where)

    def update(self, other: Exposure) -> None:
        for identity, places in other.identities.items():
            for where in places:
                self.add(identity, where)


def exposure_of(libraries: Iterable[Path]) -> Exposure:
    """The identities the release exposure logs of these libraries record, every started
    attempt of any request, finished or not; older lines read under the legacy profile."""
    out = Exposure()
    for lib in libraries:
        for number, entry in enumerate(library.release_exposure(Path(lib)), 1):
            for identity in library.exposure_identities(entry):
                out.add(identity, f"{Path(lib) / library.EXPOSURE_FILE} line {number}")
    return out


def evaluation_identities(state: RequestState) -> list[StartIdentity]:
    """The request's evaluation starts: its pool's initial states 30 to 49, its profile."""
    return release_starts_for(
        state.base_task, len(RELEASE_INIT_STATES), profile=state.allocation_profile
    )


def mapping_problems(state: RequestState, seen: Iterable[StartIdentity]) -> list[str]:
    """Public starts that two identities claim: among those seen, and between them and the
    request's own allocation of its pool (initial states 0 to 49)."""
    claimed: dict[tuple[int, int], StartIdentity] = {}
    problems = []
    own = starts_for(
        state.base_task,
        range(DEVELOPMENT_INIT_STATES.start, RELEASE_INIT_STATES.stop),
        profile=state.allocation_profile,
    )
    for identity in [*own, *seen]:
        pair = (identity.base_task, identity.init_state)
        other = claimed.setdefault(pair, identity)
        if other != identity:
            problems.append(
                f"start {pair[0]}/{pair[1]} means two identities (seeds {other.scene_seed} "
                f"and {identity.scene_seed}): another allocation profile used this pool before"
            )
    return sorted(set(problems))


def assignment_problems(
    state: RequestState, request_file: Path, assignment: Path | None, cohort: str | None
) -> tuple[list[str], dict[str, Any] | None]:
    """The request against the host's assignment record; returns the problems and the
    cohort's entry with the record's digest."""
    if assignment is None:
        if state.allocation_profile is None:
            return [], None
        return [
            "a request with an allocation profile names its pool in the host's assignment "
            "record (--assignment and --cohort)"
        ], None
    if cohort is None:
        return ["--assignment needs --cohort, the cohort this investigation runs"], None
    data = json.loads(Path(assignment).read_text())
    if data.get("format") != ASSIGNMENT_FORMAT:
        return [f"{assignment}: not an assignment record ({ASSIGNMENT_FORMAT})"], None
    entries = [a for a in data.get("assignments", []) if a.get("cohort") == cohort]
    if len(entries) != 1:
        return [f"{assignment}: {len(entries)} assignments of cohort {cohort!r}, not one"], None
    entry = entries[0]
    problems = []
    if entry.get("base_task") != state.base_task:
        problems.append(
            f"cohort {cohort} holds pool {entry.get('base_task')}; the request's is "
            f"{state.base_task}"
        )
    if entry.get("allocation_profile") != state.allocation_profile:
        problems.append(
            f"cohort {cohort}'s profile is {entry.get('allocation_profile')!r}; the request's "
            f"is {state.allocation_profile!r}"
        )
    if entry.get("status") not in OPEN_STATUSES:
        problems.append(f"cohort {cohort} is {entry.get('status')!r}, not one of {OPEN_STATUSES}")
    digest = hashlib.sha256(Path(request_file).read_bytes()).hexdigest()
    if entry.get("request_sha256") not in (None, digest):
        problems.append(f"the request file is not the one assigned to cohort {cohort}")
    for other in data.get("assignments", []):
        if (
            other is not entry
            and other.get("base_task") == state.base_task
            and other.get("line") != entry.get("line")
        ):
            problems.append(
                f"pool {state.base_task} is assigned to line {other.get('line')!r} as well"
            )
    for past in data.get("historical", []):
        if past.get("base_task") == state.base_task:
            problems.append(f"pool {state.base_task} is historical ({past.get('status')})")
    record = {
        **entry,
        "record": str(assignment),
        "record_sha256": hashlib.sha256(Path(assignment).read_bytes()).hexdigest(),
        "request_file_sha256": digest,
    }
    return problems, record


def preflight(
    state: RequestState,
    request_file: Path,
    *,
    exposure: Exposure,
    seen: Iterable[StartIdentity] = (),
    assignment: Path | None = None,
    cohort: str | None = None,
) -> dict[str, Any]:
    """The checks before an investigation starts (the module's docstring); raises
    `PreflightError` with every problem, or returns what was checked for the record.

    `exposure` is what earlier attempts examined; `seen` are further identities the
    investigation shows or carries (imported episodes)."""
    check_profile(state.allocation_profile)
    problems, cohort_entry = assignment_problems(state, request_file, assignment, cohort)
    shown = [*exposure.identities, *seen]
    problems += mapping_problems(state, shown)
    evaluation = evaluation_identities(state)
    examined = sorted(
        {s.init_state for s in evaluation if s in exposure.identities},
    )
    if examined:
        where = sorted({w for s in evaluation for w in exposure.identities.get(s, [])})
        problems.append(
            f"the evaluation starts {examined} of pool {state.base_task} were examined before "
            f"({'; '.join(where[:3])}{' ...' if len(where) > 3 else ''}): this pool is not fresh"
        )
    if problems:
        raise PreflightError(problems)
    return {
        "base_task": state.base_task,
        "allocation_profile": state.allocation_profile,
        "evaluation_starts": [s.init_state for s in evaluation],
        "identities_examined_before": len(exposure.identities),
        "identities_shown": len(set(shown)),
        "assignment": cohort_entry,
    }


# ---- the import of earlier investigations ---------------------------------------------------

RECORD_FILE = "job_run.json"  # an investigation's record (`jobs.driver.RECORD_FILE`)
# Where an investigation keeps, from before its first session, every exposure line it
# carries: those of its mounted library and of its predecessors (`jobs.driver.investigate`).
CARRIED_EXPOSURE = "prior/exposure"
RECEIPT_FILE = "receipt.json"
PROGRAM_JOB, NAIVE_JOB = "j0001", "j0002"  # the release run's jobs (`jobs.__main__`)
BLOCK = 100
MAX_DEVELOPMENT_JOB = 97
PROGRAM_SLOT, NAIVE_SLOT = 98, 99
MAX_SOURCES = 89  # 1000 + 100 * 89 + 99 = 9999: episode ids keep six digits
PER_EPISODE = ("object_steps", "goal_steps", "robot_steps")


class ImportRefused(ValueError):
    """Earlier investigations the host will not import, with every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass
class _Episode:
    old: str
    new: str
    start: StartIdentity
    row: dict[str, Any]  # the run's job_episodes row
    events: list[dict[str, Any]]  # its job_step_events rows


@dataclass
class _Job:
    new: str
    investigation: int
    old: str
    part: str
    status: str
    session: int | None
    program: JobProgram
    state: RequestState  # the request the run recorded
    run: Path
    episodes: list[_Episode]


@dataclass
class Prior:
    """Earlier investigations read and checked; nothing is written until `write`."""

    sources: list[dict[str, Any]]
    jobs: list[_Job]
    investigations: list[dict[str, Any]]  # `toolserver.jobs.ImportedInvestigation` fields
    checks: list[dict[str, Any]]
    job_names: list[str]
    check_names: list[str]
    exposure: Exposure
    exposure_lines: list[str]
    files: dict[str, str]  # every source file read, by path: its SHA-256

    @property
    def identities(self) -> list[StartIdentity]:
        """Every start the imported episodes ran from and every examined identity."""
        return [e.start for j in self.jobs for e in j.episodes] + list(self.exposure.identities)

    def public(self) -> Any:
        """The history as mounted (`toolserver.jobs.PublicHistory`)."""
        from fleetlab.toolserver.jobs import PublicHistory
        from fleetlab.toolserver.jobs import goal_text as text_of

        programs: dict[str, dict[str, Any]] = {}
        imported = []
        for job in self.jobs:
            programs.setdefault(
                job.program.digest,
                {
                    "program_digest": job.program.digest,
                    "program": job.program.model_dump(mode="json"),
                },
            )
            imported.append(
                {
                    "job_id": job.new,
                    "investigation": job.investigation,
                    "source_job_id": job.old,
                    "part": job.part,
                    "source_status": job.status,
                    "source_session": job.session,
                    "name": job.program.name,
                    "purpose": job.program.purpose,
                    "program_digest": job.program.digest,
                    "base_task": job.state.base_task,
                    "requested_goals": [text_of(g) for g in job.state.requested_goals],
                    "order_rules": [
                        {"earlier_goal": r.first, "later_goal": r.then}
                        for r in job.state.order_rules
                    ],
                    "episodes": [
                        {
                            "episode_id": e.new,
                            "source_episode_id": e.old,
                            "init_state": e.start.init_state,
                        }
                        for e in job.episodes
                    ],
                }
            )
        return PublicHistory.model_validate(
            {
                "format": "fleetlab.job-history/1",
                "investigations": self.investigations,
                "imported": imported,
                "programs": list(programs.values()),
                "checks": self.checks,
                "job_names": self.job_names,
                "check_names": self.check_names,
            }
        )

    def write(self, out: Path) -> dict[str, Any]:
        """The remapped tree, the registry, the carried exposure lines and the public
        history under `out` (new); checked as the tools check it, and the sources unchanged."""
        from fleetlab.toolserver.jobs import history_problems

        out.mkdir(parents=True, exist_ok=False)
        tree = out / "history"
        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for job in self.jobs:
            for episode in job.episodes:
                row, moved = _remap(job, episode, tree)
                rows.append(row)
                events.extend(moved)
        if rows:
            tables.write_run_tables(tree, rows, events)
        exposure = out / "exposure"
        exposure.mkdir()
        if self.exposure_lines:
            text = "".join(line + "\n" for line in self.exposure_lines)
            (exposure / library.EXPOSURE_FILE).write_text(text)
        public = self.public()
        written = {
            r["episode_id"]: {
                k: r[k] for k in ("job_id", "program_digest", "base_task", "init_state")
            }
            for r in rows
        }
        problems = history_problems(public, written)
        if problems:
            raise ImportRefused(problems)
        public_text = json.dumps(public.model_dump(mode="json"), indent=1, ensure_ascii=False)
        (out / "history.json").write_text(public_text + "\n")
        registry = {
            "format": "fleetlab.job-history-registry/1",
            "sources": self.sources,
            "jobs": [
                {
                    "job_id": job.new,
                    "investigation": job.investigation,
                    "source_job_id": job.old,
                    "part": job.part,
                    "source_status": job.status,
                    "source_session": job.session,
                    "program_digest": job.program.digest,
                    "base_task": job.state.base_task,
                    "allocation_profile": job.state.allocation_profile,
                    "run": str(job.run),
                    "episodes": [
                        {
                            "episode_id": e.new,
                            "source_episode_id": e.old,
                            "start": e.start.model_dump(mode="json"),
                        }
                        for e in job.episodes
                    ],
                }
                for job in self.jobs
            ],
            "exposure": [
                {"start": identity.model_dump(mode="json"), "where": where}
                for identity, where in self.exposure.identities.items()
            ],
            "files": self.files,
            "public_sha256": hashlib.sha256((public_text + "\n").encode()).hexdigest(),
        }
        registry_text = json.dumps(registry, indent=1, sort_keys=True) + "\n"
        (out / "registry.json").write_text(registry_text)
        changed = sorted(
            path for path, digest in self.files.items() if _sha256(Path(path)) != digest
        )
        if changed:
            raise ImportRefused([f"{path} changed while it was imported" for path in changed])
        return {
            "tree": tree if rows else None,
            "exposure": exposure,
            "public": public,
            "registry": out / "registry.json",
            "registry_sha256": hashlib.sha256(registry_text.encode()).hexdigest(),
            "jobs": len(self.jobs),
            "episodes": len(rows),
        }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _remap(job: _Job, episode: _Episode, tree: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One episode's files and rows under its new ids; every other value as it was."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.log._files import write_parquet
    from fleetlab.log._rows import to_arrow

    for name in PER_EPISODE:
        data = pq.read_table(tables.table_path(job.run, name, episode.old))
        if set(data["episode_id"].to_pylist()) != {episode.old}:
            raise ImportRefused([f"{job.run}: {name} of {episode.old} holds other episodes"])
        column = data.schema.get_field_index("episode_id")
        data = data.set_column(
            column, "episode_id", pa.array([episode.new] * data.num_rows, pa.string())
        )
        write_parquet(to_arrow(name, data), tables.table_path(tree, name, episode.new), tree)
    row = {**episode.row, "job_id": job.new, "episode_id": episode.new}
    for camera in Camera:
        key = f"video_{camera.value}"
        if not episode.row.get(key):
            row[key] = None
            continue
        source = job.run / "log" / tables.video_relative(episode.old, camera)
        target = tree / "log" / tables.video_relative(episode.new, camera)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        row[key] = tables.video_relative(episode.new, camera)
    return row, [{**event, "episode_id": episode.new} for event in episode.events]


class _Reader:
    """The files of the sources, each read once and its digest kept."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def bytes(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.files[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path: Path) -> Any:
        return json.loads(self.bytes(path))

    def lines(self, path: Path) -> list[str]:
        return [line for line in self.bytes(path).decode().splitlines() if line.strip()]

    def table(self, path: Path) -> list[dict[str, Any]]:
        import io

        import pyarrow.parquet as pq

        return list(pq.read_table(io.BytesIO(self.bytes(path))).to_pylist())

    def note(self, path: Path) -> None:
        self.bytes(path)


def _inside(root: Path, path: Path | str | None, default: Path) -> Path:
    """A path the record names, when it lies inside the investigation's directory (a moved
    directory keeps its own layout); otherwise the layout's default."""
    if path:
        found = Path(path).resolve()
        if found.is_relative_to(root):
            return found
    return default


def _goals(state: RequestState) -> tuple[list[str], list[tuple[int, int]]]:
    return [g.text for g in state.requested_goals], [(r.first, r.then) for r in state.order_rules]


def read_prior(sources: Sequence[Path]) -> Prior:
    """The public records of earlier investigations, read and checked (the module's
    docstring); raises `ImportRefused` with every problem."""
    roots = [Path(s).expanduser().resolve() for s in sources]
    problems = []
    if len(roots) > MAX_SOURCES:
        problems.append(f"at most {MAX_SOURCES} earlier investigations, not {len(roots)}")
    if len(set(roots)) != len(roots):
        problems.append("an earlier investigation is named twice")
    if problems:
        raise ImportRefused(problems)
    reader = _Reader()
    prior = Prior([], [], [], [], [], [], Exposure(), [], reader.files)
    seen_checks: set[tuple[str, str]] = set()
    for number, root in enumerate(roots, 1):
        _read_source(reader, prior, number, root, seen_checks)
    prior.job_names = sorted(set(prior.job_names))
    prior.check_names = sorted(set(prior.check_names))
    return prior


def _read_source(
    reader: _Reader, prior: Prior, number: int, root: Path, seen_checks: set[tuple[str, str]]
) -> None:
    record_path = root / RECORD_FILE
    if not record_path.is_file():
        raise ImportRefused([f"{root}: no {RECORD_FILE}, not an investigation"])
    record = reader.json(record_path)
    if record.get("kind") != "job-investigation":
        raise ImportRefused([f"{root}: {RECORD_FILE} is not a job investigation's record"])
    block = 1000 + BLOCK * (number - 1)
    runs: list[tuple[Path, str, int | None, str, dict[str, str]]] = []
    for batch in record.get("batches", []):
        session = int(batch["session"])
        default = root / "runs" / f"session-{session:02d}" / "run"
        path = _inside(root, batch.get("run_dir"), default)
        if not (path / tables.RUN_RECORD).is_file():
            continue  # nothing ran: refused, or broken off before the runner wrote anything
        status = batch.get("status") if batch.get("status") in ("done", "failed") else "interrupted"
        outcomes = {str(o["job_id"]): str(o["status"]) for o in batch.get("outcomes", [])}
        runs.append((path, "development", session, status, outcomes))
    release = record.get("release")
    attempt = record.get("release_attempt") or {}
    default = root / "release" / "run" / "run"
    evaluation = _inside(root, Path(release["run_dir"]) / "run" if release else None, default)
    finished = release is not None and (evaluation / RECEIPT_FILE).is_file()
    if (evaluation / tables.RUN_RECORD).is_file():
        after = (release or attempt).get("after_session")
        after_session = None if after is None else int(after)
        state_of_run = "done" if finished else "interrupted"
        runs.append((evaluation, "evaluation", after_session, state_of_run, {}))
    contract: tuple[list[str], list[tuple[int, int]]] | None = None
    base_task = None
    jobs: list[_Job] = []
    for path, part, run_session, status, outcomes in runs:
        found, state = _read_run(reader, number, block, path, part, run_session, status, outcomes)
        if contract is None:
            contract, base_task = _goals(state), state.base_task
        elif _goals(state) != contract:
            raise ImportRefused([f"{path}: another request's goals than its investigation's"])
        jobs += found
    analyses = sorted(p for p in (root / "analyses").glob("*") if p.is_dir())
    only = analyses[0] if len(analyses) == 1 else root / "-"
    analysis = _inside(root, record.get("analysis"), only)
    public_state = analysis / "results" / "operator_state.json"
    if contract is None:
        if not public_state.is_file():
            raise ImportRefused([f"{root}: no run and no request to import"])
        from fleetlab.toolserver.jobs import request_from_public

        state = request_from_public(reader.json(public_state))
        contract, base_task = _goals(state), state.base_task
    assert base_task is not None
    evaluation_summary = None
    program_job = next((j for j in jobs if j.part == "release-candidate"), None)
    naive_job = next((j for j in jobs if j.part == "release-naive"), None)
    if program_job is not None or naive_job is not None:
        evaluation_summary = _evaluation(
            reader, evaluation, release, attempt, program_job, naive_job, finished
        )
    notes_file = analysis / "memory" / "notes.md"
    account_file = analysis / "outbox" / "job_account.json"
    account = reader.json(account_file) if account_file.is_file() else None
    goals, rules = contract
    prior.investigations.append(
        {
            "investigation": number,
            "requested_goals": goals,
            "order_rules": [{"earlier_goal": a, "later_goal": b} for a, b in rules],
            "base_task": base_task,
            "outcome": str(record.get("outcome", "unknown")),
            "sessions": len(record.get("sessions", [])),
            "final_evaluation": evaluation_summary,
            "notes": reader.bytes(notes_file).decode() if notes_file.is_file() else None,
            "account": None
            if account is None
            else {"job": account.get("job"), "words": str(account.get("account", ""))},
        }
    )
    libraries = [root / "library", root / "release" / "library"]
    _read_names_and_checks(reader, prior, number, analysis, libraries, seen_checks)
    # The exposure the investigation carried from the start (`CARRIED_EXPOSURE`: its
    # mounted library's and its own predecessors'), beside what its libraries recorded, so
    # that exposure stays transitive through an investigation broken off before it froze
    # a library.
    for lib in (*libraries, root / CARRIED_EXPOSURE):
        _read_exposure(reader, prior, lib, evaluation, finished)
    for job in jobs:
        for episode in job.episodes:
            if episode.start.is_release:
                prior.exposure.add(episode.start, f"imported episode {episode.new}")
    prior.jobs += jobs
    prior.sources.append(
        {
            "investigation": number,
            "path": str(root),
            "label": record.get("label"),
            "outcome": record.get("outcome"),
            "record_sha256": reader.files[str(record_path)],
            "runs": [
                {"path": str(path), "part": part, "session": session, "status": status}
                for path, part, session, status, _ in runs
            ],
        }
    )


def _read_run(
    reader: _Reader,
    number: int,
    block: int,
    path: Path,
    part: str,
    session: int | None,
    status: str,
    outcomes: dict[str, str],
) -> tuple[list[_Job], RequestState]:
    """The completed episodes of one run directory, checked against its own record."""
    run_record = reader.json(path / tables.RUN_RECORD)
    state = RequestState.model_validate(run_record["state"])
    profile = check_profile(state.allocation_profile)
    entries = {str(j["job_id"]): j for j in run_record.get("jobs", [])}
    episodes_file = path / tables.EPISODES_FILE
    recorded = (
        {e.episode_id: e for e in map(JobEpisode.model_validate_json, reader.lines(episodes_file))}
        if episodes_file.is_file()
        else {}
    )
    table = tables.table_path(path, "job_episodes")
    rows = reader.table(table) if table.is_file() else []
    events_file = tables.table_path(path, "job_step_events")
    events: dict[str, list[dict[str, Any]]] = {}
    for event in reader.table(events_file) if events_file.is_file() else []:
        events.setdefault(str(event["episode_id"]), []).append(event)
    problems = []
    by_job: dict[str, list[_Episode]] = {}
    for row in rows:
        old, job_id = str(row["episode_id"]), str(row["job_id"])
        entry = entries.get(job_id)
        if entry is None:
            problems.append(f"{path}: {old} belongs to {job_id}, which the run did not record")
            continue
        program = JobProgram.model_validate(entry["program"])
        if not program.digest == entry["program_digest"] == row["program_digest"]:
            problems.append(f"{path}: {old} ran another program than {job_id}'s")
        starts = [StartIdentity.model_validate(s) for s in entry["starts"]]
        place = int(old[1:]) - int(entry["first_episode"])
        offset = int(old[1:]) - int(job_id[1:]) * 100
        if not 0 <= place < len(starts) or not 1 <= offset <= 99:
            problems.append(f"{path}: {old} is not one of {job_id}'s starts")
            continue
        start = starts[place]
        if (row["base_task"], row["init_state"]) != (start.base_task, start.init_state):
            problems.append(f"{path}: {old}'s row names another start than its job's")
        if start != start_for(state.base_task, start.init_state, profile=profile):
            problems.append(f"{path}: {old}'s start is not the one its run's allocation gives")
        mine = recorded.get(old)
        if mine is not None and (mine.job_id, mine.program_digest, mine.start) != (
            job_id,
            program.digest,
            start,
        ):
            problems.append(f"{path}: {old}'s record differs from its row")
        for name in PER_EPISODE:
            if not tables.table_path(path, name, old).is_file():
                problems.append(f"{path}: {old} has no {name} file")
        for camera in Camera:
            if row.get(f"video_{camera.value}"):
                video = path / "log" / tables.video_relative(old, camera)
                if row[f"video_{camera.value}"] != tables.video_relative(old, camera):
                    problems.append(f"{path}: {old}'s {camera.value} video is named otherwise")
                elif not video.is_file():
                    problems.append(f"{path}: {old}'s {camera.value} video is missing")
                else:
                    reader.note(video)
        for name in PER_EPISODE:
            if tables.table_path(path, name, old).is_file():
                reader.note(tables.table_path(path, name, old))
        if part == "development":
            n = int(job_id[1:])
            if not 1 <= n <= MAX_DEVELOPMENT_JOB:
                problems.append(f"{path}: {job_id} is beyond the {MAX_DEVELOPMENT_JOB} jobs kept")
                continue
            new_job = block + n
        elif job_id in (PROGRAM_JOB, NAIVE_JOB):
            new_job = block + (PROGRAM_SLOT if job_id == PROGRAM_JOB else NAIVE_SLOT)
        else:
            problems.append(f"{path}: a release run has no job {job_id}")
            continue
        new = neutral_id(IdKind.episode, new_job * 100 + offset)
        by_job.setdefault(job_id, []).append(
            _Episode(old, new, start, dict(row), events.get(old, []))
        )
    if problems:
        raise ImportRefused(problems)
    jobs = []
    for job_id, found in sorted(by_job.items()):
        entry = entries[job_id]
        if part == "development":
            name, new_job = "development", block + int(job_id[1:])
            outcome = outcomes.get(job_id)
            job_status = outcome if outcome in ("done", "failed") else status
        else:
            name = "release-candidate" if job_id == PROGRAM_JOB else "release-naive"
            new_job = block + (PROGRAM_SLOT if job_id == PROGRAM_JOB else NAIVE_SLOT)
            job_status = status
        jobs.append(
            _Job(
                new=f"j{new_job:04d}",
                investigation=number,
                old=job_id,
                part=name,
                status=job_status,
                session=session,
                program=JobProgram.model_validate(entry["program"]),
                state=state,
                run=path,
                episodes=sorted(found, key=lambda e: e.new),
            )
        )
    return jobs, state


def _evaluation(
    reader: _Reader,
    path: Path,
    release: dict[str, Any] | None,
    attempt: dict[str, Any],
    program_job: _Job | None,
    naive_job: _Job | None,
    finished: bool,
) -> dict[str, Any]:
    """An investigation's final evaluation: the receipt's counts once it finished, checked
    against the run's episodes."""
    out: dict[str, Any] = {
        "job": str((release or attempt).get("job") or ""),
        "status": "done" if finished else "interrupted",
        "N": None,
        "K": None,
        "delivered": None,
        "naive_delivered": None,
        "condition_met": None,
        "released": None,
        "program_job_id": program_job.new if program_job else None,
        "naive_job_id": naive_job.new if naive_job else None,
    }
    if not finished:
        return out
    receipt = ReleaseReceipt.model_validate(reader.json(path / RECEIPT_FILE))
    ours = {e.old: e for e in (program_job.episodes if program_job else [])}
    theirs = {e.old: e for e in (naive_job.episodes if naive_job else [])}
    for pair in receipt.pairs:
        if pair.episode_id not in ours or pair.naive_episode_id not in theirs:
            raise ImportRefused([f"{path}: the receipt pairs episodes the run does not hold"])
        if (
            ours[pair.episode_id].start != pair.start
            or theirs[pair.naive_episode_id].start != pair.start
        ):
            raise ImportRefused([f"{path}: the receipt's starts differ from the run's"])
    out.update(
        job=receipt.job_name,
        N=receipt.N,
        K=receipt.K,
        delivered=receipt.delivered,
        naive_delivered=receipt.naive_delivered,
        condition_met=receipt.passed,
        released=receipt.released,
    )
    return out


def _read_names_and_checks(
    reader: _Reader,
    prior: Prior,
    number: int,
    analysis: Path,
    libraries: Sequence[Path],
    seen: set[tuple[str, str]],
) -> None:
    """The job and check names an investigation used, and its checks, each once."""
    checks: list[CheckSpec] = []
    outbox = analysis / "outbox"
    if (outbox / "saved_jobs.jsonl").is_file():
        prior.job_names += [
            json.loads(x)["name"] for x in reader.lines(outbox / "saved_jobs.jsonl")
        ]
    if (outbox / "checks.jsonl").is_file():
        checks += [
            CheckSpec.model_validate(json.loads(x)["check"])
            for x in reader.lines(outbox / "checks.jsonl")
        ]
    for lib in libraries:
        for path in sorted((lib / "jobs").glob("*.json")):
            reader.note(path)
            prior.job_names.append(path.stem)
        for path in sorted((lib / "checks").glob("*.json")):
            checks.append(CheckSpec.model_validate(reader.json(path)))
    for check in checks:
        key = (check.name, library.digest_of(check.model_dump(mode="json")))
        prior.check_names.append(check.name)
        if key not in seen:
            seen.add(key)
            prior.checks.append({"investigation": number, "check": check.model_dump(mode="json")})


def _read_exposure(
    reader: _Reader, prior: Prior, lib: Path, evaluation: Path, finished: bool
) -> None:
    """A library's exposure lines, kept byte for byte, each read under its own profile; a
    legacy line of this investigation's evaluation run checked against its receipt."""
    path = lib / library.EXPOSURE_FILE
    if not path.is_file():
        return
    receipt = (
        ReleaseReceipt.model_validate(reader.json(evaluation / RECEIPT_FILE)) if finished else None
    )
    for number, line in enumerate(reader.lines(path), 1):
        if line not in prior.exposure_lines:
            prior.exposure_lines.append(line)
        entry = json.loads(line)
        if entry.get("event") != "started":
            continue
        identities = library.exposure_identities(entry)
        if "starts" in entry:
            profile = check_profile(entry.get("allocation_profile"))
            if any(s != start_for(s.base_task, s.init_state, profile=profile) for s in identities):
                raise ImportRefused([f"{path} line {number}: starts its profile does not give"])
        elif (
            receipt is not None
            and _inside(evaluation, entry.get("run_dir"), Path("/")) == evaluation
        ):
            if identities != receipt.release_starts[: len(identities)]:
                raise ImportRefused([f"{path} line {number}: the legacy rule misses the receipt"])
        for identity in identities:
            prior.exposure.add(identity, f"{path} line {number}")
