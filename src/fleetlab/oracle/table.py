"""The oracle table: every outcome the runs recorded for the scoring episodes, by episode
and effective fault list (contracts section 9), read from `truth/job_outcomes.parquet`.

A row comes from a job of mode "run" (the episode as run, its true verdict before the
data hook, under its full fault list) or "replay" from the reset (its faults minus
`removed`, plus `added`). Replays from a snapshot, continuations and scripted replays
are not oracle rows: they change the episode from a step on, not from the reset.

Checks, each refusing the row or the run with a reason:

- only runs whose chain sample held (`run.json` `chain_ok` true; the oracle's own
  replay-only runs get it from `fleetlab.oracle.proof`) and that are finalized as far
  as the job tables go (nothing partial, every finished job merged; a run-level table
  added to the schema after the run, such as the visible `catalogue`, does not matter
  here) give rows;
- a "run" row's identity, read from the run's `identities` table, equals the world's;
- a "replay" row's job key is the key of the job rebuilt from the world's identity,
  its `removed` and its `added`: that ties the row to the episode and to what was
  removed and added, not to the identity (the key does not hash it). The oracle's own
  runs carry their provenance (`run.json` `extra.oracle`: the world hash and the
  manifest they were made from, with its SHA-256), so for them the job of every row is
  looked up in that manifest and its identity must equal the world's bit for bit; a run
  made for another world hash, or whose manifest changed or is gone, is refused. A
  world run's replays (the miniature fleet keeps its oracle's replays in `w10-later`)
  require a matching independently recorded identity in its identities table too;
- two rows of one effective fault list that disagree need the shared four-execution
  classifier. With no settlement context, or an unresolved difference, the build stops.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.identity import EpisodeIdentity
from fleetlab.contracts.oracle import OracleRow, OracleTable, fault_list_hash
from fleetlab.contracts.rollout import Job, JobMode, RunRecord
from fleetlab.oracle.mechanism import MechanismTable

if TYPE_CHECKING:
    from fleetlab.oracle.verification import Execution, ReplayVerifier

ORACLE_MODES = frozenset({JobMode.run.value, JobMode.replay.value})


class ReplayNotExact(RuntimeError):
    """Two outcomes of one episode under one fault list disagree."""


@dataclass
class RunRows:
    """What one run gave the table."""

    run_dir: Path
    used: bool = False
    rows: int = 0  # rows of scoring episodes added
    as_run: int = 0
    replays: int = 0  # replays that remove or add something
    unmodified: int = 0  # unmodified replays (a chain sample's, a proof's)
    refused: str | None = None  # why the run gives no rows
    problems: list[str] = field(default_factory=list)  # rows refused, with the reason
    sources: dict[tuple[str, str], Execution] = field(default_factory=dict)
    executions: list[Execution] = field(default_factory=list)


@dataclass
class TableBuild:
    """The table and where its rows came from."""

    table: OracleTable
    runs: list[RunRows]
    keys: set[tuple[str, str]] = field(default_factory=set)  # distinct (episode, list hash)
    verification: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_held(self) -> int:
        return len(self.keys)

    def problems(self) -> list[str]:
        out = [f"{r.run_dir}: {r.refused}" for r in self.runs if r.refused]
        return out + [f"{r.run_dir}: {p}" for r in self.runs for p in r.problems]


def run_refusal(run_dir: Path) -> str | None:
    """Why a run gives no oracle rows, or None if it does."""
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        return "no run.json"
    record = RunRecord.model_validate_json(record_path.read_text())
    if record.chain_ok is not True:
        return (
            f"chain_ok is {record.chain_ok!r}: the run gives no oracle rows (contracts section 4)"
        )
    if not (run_dir / "truth" / "job_outcomes.parquet").is_file():
        return "no truth/job_outcomes.parquet"
    if not jobs_merged(run_dir):
        return "not finalized: jobs ended after the last finalize"
    return None


def jobs_merged(run_dir: Path) -> bool:
    """Nothing partial and every finished job in the run-level tables (module docstring)."""
    from fleetlab.log import run_status

    status = run_status(run_dir)
    return not (status.partial or status.staging_files) and set(status.done) == set(status.merged)


def effective(
    identity: EpisodeIdentity, removed: Sequence[str], added: Sequence[FaultSpec]
) -> list[FaultSpec]:
    gone = set(removed)
    return [f for f in identity.faults if f.fault_id not in gone] + list(added)


def own_manifest(run_dir: Path, world_hash: str | None) -> tuple[dict[str, Job] | None, str | None]:
    """The jobs of an oracle-owned run by key (from the manifest its provenance names), or
    None for a run without provenance (a world run); and why the run is refused, if so."""
    from fleetlab.oracle.run import recorded_manifest, recorded_text

    record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    provenance, path = recorded_manifest(record, run_dir)
    if provenance is None:
        return None, None
    made_for = provenance.get("world_hash")
    if world_hash is not None and made_for is not None and made_for != world_hash:
        return None, f"made for world hash {made_for}, not {world_hash}"
    if not path.is_file():
        return None, f"its manifest {path} is gone"
    text = recorded_text(provenance, path.read_text())
    if text is None:
        return None, f"its manifest {path} changed after the run"
    jobs = [Job.model_validate_json(line) for line in text.splitlines() if line.strip()]
    return {j.key: j for j in jobs}, None


def rows_of_run(
    run_dir: Path, scoring: dict[str, EpisodeIdentity], world_hash: str | None = None
) -> tuple[list[OracleRow], RunRows]:
    """The oracle rows a run holds for the scoring episodes, checked (module docstring)."""
    from fleetlab.log.runfiles import read_identities, read_table

    info = RunRows(run_dir=run_dir)
    info.refused = run_refusal(run_dir)
    if info.refused:
        return [], info
    own, refused = own_manifest(run_dir, world_hash)
    if refused:
        info.refused = refused
        return [], info
    ran = {i.episode_id: i for i in read_identities(run_dir)}
    rows: list[OracleRow] = []
    for record in read_table(run_dir, "job_outcomes").to_pylist():
        identity = scoring.get(str(record["episode_id"]))
        mode = str(record["mode"])
        if identity is None or mode not in ORACLE_MODES or record["from_snapshot_step"] is not None:
            continue
        key = str(record["job_key"])
        removed = [str(f) for f in json.loads(record["removed"] or "[]")]
        added = [FaultSpec.model_validate(f) for f in json.loads(record["added"] or "[]")]
        if mode == JobMode.run.value:
            if ran.get(identity.episode_id) != identity:
                info.problems.append(f"{key}: the as-run identity differs from the world's")
                continue
            info.as_run += 1
            job = Job(identity=identity)
        else:
            try:
                job = Job(identity=identity, mode=JobMode.replay, remove=removed, add=added)
            except ValueError as exc:
                info.problems.append(f"{key}: not a replay of the world's identity: {exc}")
                continue
            if job.key != key:
                info.problems.append(f"{key}: another removal or addition than recorded")
                continue
            if own is not None:
                ran_as = own.get(key)
                if ran_as is None or ran_as.identity != identity:
                    info.problems.append(f"{key}: its manifest's identity differs from the world's")
                    continue
            elif ran.get(identity.episode_id) != identity:
                # A world run's replay: its episode as run there must be the world's.
                info.problems.append(f"{key}: the run's identity of the episode differs")
                continue
            if removed or added:
                info.replays += 1
            else:
                info.unmodified += 1
        faults = effective(identity, removed, added)
        from fleetlab.oracle.verification import Execution

        info.sources[(identity.episode_id, fault_list_hash(faults))] = Execution(
            run_dir, job, bool(record["success"])
        )
        info.executions.append(info.sources[(identity.episode_id, fault_list_hash(faults))])
        rows.append(
            OracleRow(
                episode_id=identity.episode_id,
                active_faults=fault_list_hash(faults),
                success=bool(record["success"]),
            )
        )
    info.used = True
    info.rows = len(rows)
    return rows, info


def build_table(
    scoring: Iterable[EpisodeIdentity],
    excluded: Iterable[str],
    run_dirs: Iterable[Path],
    world_hash: str | None = None,
    verifier: ReplayVerifier | None = None,
) -> TableBuild:
    """The oracle table of these scoring episodes from every run that gives rows; an
    oracle-owned run made for another world hash is refused."""
    identities = list(scoring)
    by_id = {i.episode_id: i for i in identities}
    table = MechanismTable(identities, excluded=excluded)
    build = TableBuild(table=table, runs=[])
    known: dict[tuple[str, str], tuple[OracleRow, Execution | None]] = {}
    for run_dir in run_dirs:
        rows, info = rows_of_run(Path(run_dir), by_id, world_hash)
        build.runs.append(info)
        sources: dict[tuple[str, str, bool], deque[Execution]] = defaultdict(deque)
        for execution in info.executions:
            sources[
                (
                    execution.job.identity.episode_id,
                    fault_list_hash(execution.job.effective_faults),
                    execution.success,
                )
            ].append(execution)
        for row in rows:
            matching = sources[(row.episode_id, row.active_faults, row.success)]
            source = matching.popleft() if matching else None
            key = (row.episode_id, row.active_faults)
            if verifier is not None and source is not None:
                row = row.model_copy(update={"success": verifier.canonical(source)})
            previous = known.get(key)
            if previous is not None and previous[0].success != row.success:
                if verifier is None or previous[1] is None or source is None:
                    raise ReplayNotExact(f"{run_dir}: disagreeing rows for {key}")
                result = verifier.settle(previous[1], source)
                row = row.model_copy(update={"success": result["success"]})
            known[key] = (row, source)
            build.keys.add(key)
    for row, _ in known.values():
        try:
            table.add(row)
        except ValueError as exc:
            raise ReplayNotExact(str(exc)) from exc
    if verifier is not None:
        build.verification = verifier.summary()
    return build
