"""The proof of a replay-only run (contracts section 4): its unmodified replays against the
chains of the world as run.

The engine proves a run by replaying a sample of its own "run" jobs inside it. A run of
the oracle's replays has no "run" job, so the engine samples nothing and leaves
`chain_ok` empty. The oracle's plan adds the sample itself (`plan.proof_jobs`: every
tenth scoring episode, at least 20, replayed unmodified), and this module compares each
of those chains with the as-run chain of the same episode in the world's runs, step by
step (`fleetlab.engine.chain.compare`: state, both images and the executed action at
every step, and the fixture hash). The result goes into the run's `run.json`:
`chain_sample` the episodes compared, `chain_ok` whether all were identical, and
`extra.chain_proof` the details. Oracle steps supply the shared recorded-action
four-execution classifier: two further executions settle an action-first divergence under
the fixed event ceiling. An unresolved or non-action-first difference invalidates it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetlab.contracts.rollout import Job, JobMode, RunLayout, RunRecord

PROVENANCE = "oracle"  # the key of the run's provenance in run.json's extra


def as_run_chain(episode_id: str, world_runs: Sequence[Path]) -> Path | None:
    """The chain of an episode as run, in the first proven world run that holds it."""
    for run_dir in world_runs:
        record = Path(run_dir) / "run.json"
        if not record.is_file():
            continue
        if RunRecord.model_validate_json(record.read_text()).chain_ok is not True:
            continue
        path = RunLayout(Path(run_dir)).table("chain", episode_id)
        if path.is_file():
            return path
    return None


def compare_proofs(
    run_dir: Path,
    proofs: Sequence[Job],
    world_runs: Sequence[Path],
    adjudicate: Callable[[Job, Path, Path], dict[str, Any]] | None = None,
    proof_locations: Mapping[str, Path] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Each proof replay's chain against the as-run chain; (all identical, details)."""
    from fleetlab.engine.chain import Chain, compare

    layout = RunLayout(Path(run_dir))
    details: list[dict[str, Any]] = []
    ok = bool(proofs)
    for job in proofs:
        if job.mode is not JobMode.replay or job.remove or job.add or job.from_snapshot:
            raise ValueError(f"{job.key}: a proof is an unmodified replay from the reset")
        episode = job.identity.episode_id
        original = as_run_chain(episode, world_runs)
        replay = RunLayout((proof_locations or {}).get(job.key, layout.root)).chain(job)
        if original is None:
            same, what = False, "no as-run chain in the world's runs"
        elif not replay.is_file():
            same, what = False, "the proof replay did not run"
        else:
            if adjudicate is not None:
                detail = adjudicate(job, original, replay)
                details.append({**detail, "against": str(original)})
                ok = ok and detail["accepted"]
                continue
            result = compare(Chain.load(replay), Chain.load(original))
            same, what = result.identical, result.what
        ok = ok and same
        details.append(
            {
                "episode_id": episode,
                "identical": same,
                "what": what,
                "against": None if original is None else str(original),
            }
        )
    return ok, details


def prove_run(
    run_dir: Path,
    proofs: Sequence[Job],
    world_runs: Sequence[Path],
    episodes_per_robot_day: int,
    provenance: dict[str, Any] | None = None,
    adjudicate: Callable[[Job, Path, Path], dict[str, Any]] | None = None,
    proof_locations: Mapping[str, Path] | None = None,
) -> RunRecord:
    """Compare the proofs and record the verdict in the run's `run.json`, with the run's
    provenance (`extra.oracle`: the world hash and the manifest the run was made from,
    `fleetlab.oracle.table` checks every row against it)."""
    from fleetlab.log import RunWriter

    ok, details = compare_proofs(run_dir, proofs, world_runs, adjudicate, proof_locations)
    record = RunRecord.model_validate_json((Path(run_dir) / "run.json").read_text())
    extra: dict[str, Any] = {
        "chain_proof": {
            "rule": "unmodified replays against the as-run chains (fleetlab.oracle.proof)",
            "compared": len(details),
            "identical": sum(1 for d in details if d["identical"]),
            "details": details,
        }
    }
    if provenance is not None:
        extra[PROVENANCE] = provenance
    with RunWriter(run_dir, record, episodes_per_robot_day=episodes_per_robot_day) as writer:
        return writer.update_record(
            chain_sample=[d["episode_id"] for d in details], chain_ok=ok, extra=extra
        )
