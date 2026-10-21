"""Running the oracle's replays through the engine: one heavy job, resumable, proven.

`run_here` runs a manifest of replays (`fleetlab.engine.fleet.run`) into one run
directory and then proves it against the world's as-run chains
(`fleetlab.oracle.proof`); it is what the heavy process executes. `run_heavy` starts that
process under `tools/heavy-run`, so that the heavy lock is held only while the policy is
loaded and the replays run, and waits for it. A run directory whose jobs are all done,
finalized and proven is not started again.

The engine settings come from the world: the batch size and precision of its identities
(the engine refuses jobs of another), the world's episodes per robot and day (the visible
start times, which replays do not write but the engine checks), ten workers (`0`: as many
as the engine's memory guard admits when the heavy job starts), no video,
the pinned convolution database at `~/.cache/robots-probes/engine/miopen` unless
`FLEETLAB_MIOPEN_DIR` says otherwise. The engine's own chain sample is off: a replay-only
run has no "run" job to sample, and the proof replays in the manifest take its place.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fleetlab.contracts.rollout import Job, RunLayout, RunRecord
from fleetlab.oracle.proof import PROVENANCE

if TYPE_CHECKING:
    from fleetlab.oracle.verification import Execution, ReplayVerifier

REPO = Path(__file__).resolve().parents[3]
HEAVY_RUN = REPO / "tools" / "heavy-run"
MIOPEN_DIR = Path.home() / ".cache" / "robots-probes" / "engine" / "miopen"
WORKERS = 10
ORIGIN_FILE = "oracle-origin.json"


def note(path: Path, step: str, **fields: Any) -> None:
    """Merge fields into one step's record in `steps.json`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault(step, {}).update(fields)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=1, default=str) + "\n")
    temporary.replace(path)


def run_complete(run_dir: Path, jobs: Sequence[Job]) -> bool:
    """Every job done, the run finalized and its proof held."""
    from fleetlab.oracle.table import jobs_merged

    if not (run_dir / "run.json").is_file():
        return False
    layout = RunLayout(run_dir)
    if not all(layout.is_done(j) for j in jobs):
        return False
    record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
    return jobs_merged(run_dir) and record.chain_ok is True


def engine_config(
    jobs: Sequence[Job],
    run_dir: Path,
    episodes_per_robot_day: int,
    *,
    workers: int = WORKERS,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """The engine's `RunConfig` for these replays (module docstring)."""
    from fleetlab.engine.fleet import RunConfig

    if not jobs:
        raise ValueError("no jobs to run")
    first = jobs[0].identity
    fields: dict[str, Any] = {
        "out": str(run_dir),
        "batch_size": first.batch_size,
        "workers": workers,
        "precision": first.policy.precision,
        "episodes_per_robot_day": episodes_per_robot_day,
        "video": False,
        "chain_sample": False,
    }
    fields.update(overrides or {})
    return RunConfig(**fields)


class RunRefused(RuntimeError):
    """The run directory holds another manifest's replays, or its proof failed."""


def manifest_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# A proof detail whose failure says nothing about exactness: proving again may pass.
REPROVABLE = ("no as-run chain in the world's runs", "the proof replay did not run")


def proof_failed(
    run_dir: Path, jobs: Sequence[Job], *, allow_adjudication: bool = False
) -> str | None:
    """Why a run whose jobs are all done is not proven, or None.

    A proof that found a chain that differs is final (replay is not exact, running again
    changes nothing), so the run is refused. A proof that failed only because an as-run
    chain was not there (a world run not final yet) or a proof replay had not run is
    not: the run may be proven again.
    """
    path = run_dir / "run.json"
    if not path.is_file() or not all(RunLayout(run_dir).is_done(j) for j in jobs):
        return None
    record = RunRecord.model_validate_json(path.read_text())
    if record.chain_ok is not False:
        return None
    proof = record.extra.get("chain_proof")
    details = proof.get("details") if isinstance(proof, dict) else None
    listed = details if isinstance(details, list) else []
    failures = [d for d in listed if isinstance(d, dict) and not d.get("identical")]
    if allow_adjudication and failures and not record.extra.get("replay_refusal"):
        # A pre-rule proof may have stopped at an action-caused state difference.
        # Classify its existing files; do not execute its completed jobs again.
        return None
    if failures and all(d.get("what") in REPROVABLE for d in failures):
        return None
    return f"{run_dir}: the proof failed: {proof_summary(record)}"


def recorded_manifest(record: RunRecord, run_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    """A run's recorded provenance and the manifest file it names (a path relative to the
    run directory, or an absolute one as the first runs recorded it)."""
    provenance = record.extra.get(PROVENANCE)
    if not isinstance(provenance, dict):
        return None, run_dir
    named = Path(str(provenance.get("manifest")))
    return provenance, named if named.is_absolute() else (run_dir / named).resolve()


def recorded_text(provenance: dict[str, Any], stored: str) -> str | None:
    """The manifest text the provenance was recorded for: `stored` itself, or its first
    `jobs` lines when a growing manifest was extended and the run stopped before its new
    proof; None if neither matches."""
    want = provenance.get("manifest_sha256")
    if manifest_sha(stored) == want:
        return stored
    jobs = provenance.get("jobs")
    if isinstance(jobs, int):
        head = "".join(stored.splitlines(keepends=True)[:jobs])
        if manifest_sha(head) == want:
            return head
    return None


def check_provenance(
    run_dir: Path,
    manifest_file: Path,
    stored: str,
    extend: bool,
    world_hash: str | None = None,
) -> None:
    """Refuse a run directory whose recorded manifest is not the one stored in
    `manifest_file` (a re-plan over old replays would otherwise mix them in by job key,
    which does not hash the identity); with `extend`, a stored manifest that grew from
    the recorded one passes."""
    path = run_dir / "run.json"
    provenance = None
    if path.is_file():
        provenance, _ = recorded_manifest(RunRecord.model_validate_json(path.read_text()), run_dir)
    if provenance is None:
        origin = run_dir / ORIGIN_FILE
        if origin.is_file():
            provenance = json.loads(origin.read_text())
            if not isinstance(provenance, dict):
                raise RunRefused(f"{origin}: invalid origin record")
        elif path.is_file() or (run_dir.is_dir() and any(run_dir.iterdir())):
            raise RunRefused(
                f"{run_dir}: no independently recorded origin; use another run directory"
            )
        else:
            return
    made_for = provenance.get("world_hash")
    if world_hash is not None and made_for is not None and world_hash != made_for:
        raise RunRefused(f"{run_dir}: made for world hash {made_for}, not {world_hash}")
    found = recorded_text(provenance, stored)
    if found is None or (found != stored and not extend):
        raise RunRefused(
            f"{run_dir} holds the replays of another manifest than {manifest_file}; "
            "use another oracle directory or fresh set"
        )


def auto_workers(most: int = WORKERS) -> int:
    """As many workers as the engine's memory guard admits now (`fleet.run`: the memory
    available less the server and the workers must stay above the floor), at most `most`
    and at least one; a heavy job computes it when it holds the lock, because the memory
    other processes leave free changes while it queues."""
    from fleetlab.engine.fleet import RunConfig, mem_available_gib

    config = RunConfig(out="")
    room = mem_available_gib() - config.mem_floor_gib - config.server_gib
    return max(1, min(most, int(room / config.worker_gib)))


def run_here(
    run_dir: Path,
    manifest: Sequence[Job],
    world_runs: Sequence[Path],
    episodes_per_robot_day: int,
    *,
    manifest_file: Path,
    world_hash: str | None,
    extend: bool = False,
    workers: int = WORKERS,
    overrides: Mapping[str, Any] | None = None,
    verifier: ReplayVerifier | None = None,
    proof_replays: Sequence[Execution] = (),
) -> dict[str, Any]:
    """Run a manifest of replays in this process, then prove the run with its unmodified
    replays (the proofs); the record's fields of interest.

    `manifest_file` holds the manifest (one JSON line per job). It is written if absent
    and must hold exactly this manifest otherwise, or with `extend` a beginning of it
    (a run that grows, as the miniature fleet's single fresh run does). Its SHA-256 and
    the world hash become the run's provenance, against which the table checks every row.
    """
    os.environ.setdefault("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR))
    from fleetlab.engine.fleet import run
    from fleetlab.oracle.proof import prove_run

    text = "".join(j.model_dump_json() + "\n" for j in manifest)
    provenance = {
        "world_hash": world_hash,
        "manifest": os.path.relpath(manifest_file.resolve(), run_dir.resolve()),
        "manifest_sha256": manifest_sha(text),
        "jobs": len(manifest),
    }
    if manifest_file.is_file():
        stored = manifest_file.read_text()
        if stored != text and not (extend and text.startswith(stored)):
            raise RunRefused(f"{manifest_file} holds another manifest than this run's")
        check_provenance(run_dir, manifest_file, stored, extend, world_hash)
    elif run_dir.is_dir():
        check_provenance(run_dir, manifest_file, "", extend, world_hash)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_file.with_name(f".{manifest_file.name}.tmp")
    temporary.write_text(text)
    temporary.replace(manifest_file)
    proofs = [j for j in manifest if not (j.remove or j.add)]
    if proof_replays:
        from fleetlab.oracle.table import rows_of_run

        for execution in proof_replays:
            _, checked = rows_of_run(
                execution.run,
                {execution.job.identity.episode_id: execution.job.identity},
                world_hash,
            )
            if checked.refused or checked.problems or execution not in checked.executions:
                raise RunRefused("the reused proof is not an independently proven execution")
    proof_locations = {e.job.key: e.run for e in proof_replays}
    proofs.extend(e.job for e in proof_replays if e.job.key not in {j.key for j in proofs})
    failed = proof_failed(run_dir, manifest, allow_adjudication=verifier is not None)
    if failed:
        raise RunRefused(failed)
    record_path = run_dir / "run.json"
    if run_complete(run_dir, manifest) and record_path.is_file():
        done = RunRecord.model_validate_json(record_path.read_text())
        recorded_provenance, _ = recorded_manifest(done, run_dir)
        if recorded_provenance is not None and recorded_provenance.get(
            "manifest_sha256"
        ) == manifest_sha(text):
            # Done and proven for exactly this manifest: nothing to run, nothing to prove.
            return {
                "jobs": len(manifest),
                "jobs_run_now": 0,
                "seconds": 0.0,
                "throughput": None,
                "chain_ok": done.chain_ok,
                "proof": proof_summary(done),
            }
    started = time.time()
    # Persist the execution identity BEFORE the engine can create reusable job
    # files. An interruption before prove_run must not leave adoptable rows.
    run_dir.mkdir(parents=True, exist_ok=True)
    origin = run_dir / ORIGIN_FILE
    temporary_origin = origin.with_name(f".{origin.name}.tmp")
    temporary_origin.write_text(json.dumps(provenance, indent=1) + "\n")
    temporary_origin.replace(origin)
    from fleetlab.oracle.table import jobs_merged

    physically_complete = (
        record_path.is_file()
        and all(RunLayout(run_dir).is_done(j) for j in manifest)
        and jobs_merged(run_dir)
    )
    if not physically_complete:
        config = engine_config(
            manifest,
            run_dir,
            episodes_per_robot_day,
            workers=workers if workers > 0 else auto_workers(),
            overrides=overrides,
        )
        result = run(config, manifest)
        extra = result.record.extra
    else:
        extra = {}
    ran = time.time() - started
    if verifier is not None:
        verifier.refresh(run_dir)
    if verifier is None:
        record = prove_run(
            run_dir,
            proofs,
            world_runs,
            episodes_per_robot_day,
            provenance,
            proof_locations=proof_locations,
        )
    else:
        try:
            record = prove_run(
                run_dir,
                proofs,
                world_runs,
                episodes_per_robot_day,
                provenance,
                verifier.proof,
                proof_locations,
            )
            verifier.save()
        except RunRefused as exc:
            from fleetlab.log import RunWriter

            failed_record = RunRecord.model_validate_json((run_dir / "run.json").read_text())
            with RunWriter(
                run_dir, failed_record, episodes_per_robot_day=episodes_per_robot_day
            ) as writer:
                writer.update_record(chain_ok=False, extra={"replay_refusal": str(exc)})
            raise
    if record.chain_ok is not True:
        raise RunRefused(f"{run_dir}: the proof failed: {proof_summary(record)}")
    return {
        "jobs": len(manifest),
        "jobs_run_now": extra.get("jobs_run_now", 0),
        "seconds": round(ran, 1),
        "throughput": extra.get("throughput"),
        "chain_ok": record.chain_ok,
        "proof": proof_summary(record),
    }


def execute_third(
    folder: Path,
    job: Job,
    episodes_per_robot_day: int,
    world_hash: str | None,
    workers: int,
    overrides: Mapping[str, Any] | None,
) -> Any:
    """A further execution of its actual job kind, resumable without recursive proofs."""
    from fleetlab.engine.fleet import run
    from fleetlab.oracle.verification import Execution, outcome

    manifest = folder / "jobs.jsonl"
    run_dir = folder / "run"
    text = job.model_dump_json() + "\n"
    check_provenance(run_dir, manifest, text, False, world_hash)
    provenance = {
        "world_hash": world_hash,
        "manifest": "../jobs.jsonl",
        "manifest_sha256": manifest_sha(text),
        "jobs": 1,
    }
    folder.mkdir(parents=True, exist_ok=True)
    if manifest.is_file() and manifest.read_text() != text:
        raise RunRefused("the third execution's manifest changed")
    manifest.write_text(text)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ORIGIN_FILE).write_text(json.dumps(provenance) + "\n")
    from fleetlab.oracle.table import jobs_merged

    if not RunLayout(run_dir).is_done(job) or not jobs_merged(run_dir):
        run(
            engine_config(
                [job],
                run_dir,
                episodes_per_robot_day,
                workers=workers if workers > 0 else auto_workers(),
                overrides=overrides,
            ),
            [job],
        )
    if not (run_dir / "truth/job_outcomes.parquet").is_file():
        raise RunRefused("third execution did not finalize")
    return Execution(run_dir, job, outcome(run_dir, job))


def proof_summary(record: RunRecord) -> dict[str, Any]:
    """The run record's proof without its per-episode details."""
    proof = record.extra.get("chain_proof")
    return {k: v for k, v in proof.items() if k != "details"} if isinstance(proof, dict) else {}


def run_heavy(arguments: Sequence[str], steps: Path, step: str) -> int:
    """Start `python -m fleetlab.oracle <arguments> --here` under tools/heavy-run and wait;
    the queue time (from the call to the step's own start) goes into `steps`."""
    invoked = time.time()
    note(steps, step, invoked=invoked)
    command = [
        str(HEAVY_RUN),
        "uv",
        "run",
        "--locked",
        "python",
        "-m",
        "fleetlab.oracle",
        *arguments,
        "--here",
    ]
    env = {
        **os.environ,
        "FLEETLAB_MIOPEN_DIR": os.environ.get("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR)),
    }
    status = subprocess.run(command, cwd=REPO, env=env, check=False).returncode
    record = json.loads(steps.read_text()).get(step, {}) if steps.is_file() else {}
    started = record.get("started")
    note(
        steps,
        step,
        exit_status=status,
        queue_s=round(started - invoked, 1) if started else None,
        returned=time.time(),
    )
    return status
