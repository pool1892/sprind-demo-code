"""The chain's last steps: the fresh replays the scorer needs, the score, the fixtures.

- **fresh**: the scorer lists every replay the oracle table lacks for the agent's ranked
  actions, its fix sets and the no-data ranker's (`scoring.replays_needed_for`): actions
  that add a change (`set` to a value no fault had) are covered by no replayed subset
  (contracts section 8). Those become "replay" jobs from the reset that remove and add
  faults (`fresh/jobs.jsonl`), run in `runs/w10-fresh` under `tools/heavy-run`. A replay
  the engine would refuse (a change of the precision, which is per run, or of the image
  views, whose convolution database is not pinned) is left out and listed.
- **score**: the oracle table from the later and the fresh runs, the scorer's inputs,
  `score/report.json` and `score/report.md`.
- **fixtures**: the chain's files copied to `~/.cache/robots-fixtures/<version>/` with the
  manifest `fixtures/<version>.json` in the repository; the version is
  `fleetlab.worlds.mini.VERSION` unless the chain is told another.

The random baseline draws from the catalogue with every value filled (`filled_catalogue`):
`set` writes the fleet-wide value of its key, so that a site or robot runs the fleet's
standard setting (at fleet scope it changes nothing); `change_condition` keeps the
site's recorded condition; `request_training_data` names no particular mode. A rule an
operator could follow from the visible records alone. An entry that adds a change (a
`set` where the scope's own value differs from the fleet's without a fault behind it:
on the miniature fleet the scene camera's calibration record of one robot) needs a
fresh replay, so the scorer skips the draws that hold it and counts them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fleetlab.contracts.rollout import Job
from fleetlab.oracle.fresh import REFUSED_KEYS
from fleetlab.oracle.fresh import fresh_jobs as oracle_fresh_jobs
from fleetlab.oracle.valuation import current_values, filled_catalogue

if TYPE_CHECKING:
    from fleetlab.contracts.oracle import ReplayNeed
    from fleetlab.integration.chain import ChainPaths
    from fleetlab.scoring import ScoringInputs
    from fleetlab.worlds.generate import World

__all__ = [
    "DRAWS",
    "REFUSED_KEYS",
    "current_values",
    "filled_catalogue",
    "fresh_jobs",
    "scoring_inputs",
    "step_fixtures",
    "step_fresh",
    "step_score",
]

DRAWS = 200


# ---- inputs -----------------------------------------------------------------------------------
# The filled catalogue and the resolver's current values live in fleetlab.oracle.valuation.


def scoring_inputs(world: World, run_dirs: Sequence[Path], analysis: Path | None) -> ScoringInputs:
    """The scorer's inputs for one analysis (None: the baselines alone)."""
    import fleetlab.faults as faults
    from fleetlab.integration.chain import tool_log
    from fleetlab.integration.oracle import oracle_table
    from fleetlab.scoring import ScoringInputs, trivial_no_data_ranker

    diagnosis = fix_set = None
    calls: list[dict[str, Any]] = []
    if analysis is not None:
        from fleetlab.agents.sessions import Analysis

        record = Analysis(analysis)
        diagnosis, fix_set, calls = record.diagnosis(), record.fix_set(), tool_log(analysis)
    entries = filled_catalogue(world.observed_fleet, world.task_classes)
    return ScoringInputs(
        fault_table=world.fault_table,
        oracle=oracle_table(world, run_dirs),
        truth=faults.REGISTRY.truth_table(),
        current=current_values(world.fleet),
        diagnosis=diagnosis,
        fix_set=fix_set,
        tool_log=calls,
        catalogue=entries,
        no_data_ranked=trivial_no_data_ranker(world.observed_fleet, entries),
        seed=world.spec.seed,
        draws=DRAWS,
        confounders=[tuple(p) for p in world.spec.patterns.confounders],  # type: ignore[misc]
    )


def _oracle_runs(paths: ChainPaths) -> list[Path]:
    return [p for p in (paths.later, paths.fresh) if (p / "run.json").is_file()]


# ---- fresh replays ----------------------------------------------------------------------------


def fresh_jobs(world: World, needs: Sequence[ReplayNeed]) -> tuple[list[Job], list[str]]:
    """Replay jobs for the needed fault lists, and the ones the engine would refuse
    (`fleetlab.oracle.fresh.fresh_jobs`)."""
    return oracle_fresh_jobs({i.episode_id: i for i in world.identities("scoring")}, needs)


def jobs_file(paths: ChainPaths) -> Path:
    return paths.root / "fresh" / "jobs.jsonl"


def fresh_manifest_file(paths: ChainPaths) -> Path:
    """Every job the fresh run was handed, in order (`fleetlab.oracle.run.run_here`)."""
    return paths.root / "fresh" / "manifest.jsonl"


def fresh_manifest(paths: ChainPaths, jobs: Sequence[Job]) -> list[Job]:
    """The fresh run's manifest: what it was handed before, then the new replays and their
    new proof replays (`fleetlab.oracle.fresh.fresh_proofs`); the run grows, never changes."""
    from fleetlab.oracle.fresh import fresh_proofs
    from fleetlab.oracle.plan import read_manifest

    path = fresh_manifest_file(paths)
    old = read_manifest(path) if path.is_file() else []
    have = {j.key for j in old}
    new = [j for j in jobs if j.key not in have]
    if not new:
        return old
    replays = [j for j in old if j.remove or j.add] + new
    proofs = [p for p in fresh_proofs(replays) if p.key not in have]
    return [*old, *new, *proofs]


def step_fresh(paths: ChainPaths, world: World, *, workers: int = 10, **_: Any) -> None:
    """The replays the scorer needs beyond the oracle, as a heavy job if there are any.

    The fresh run is replay-only, so it is proven by its proof replays
    (`fleetlab.oracle.proof`); a run stopped before its proof is finished first, because
    the oracle table refuses a run without a held proof.
    """
    from fleetlab.integration.chain import analyses, heavy, note_step
    from fleetlab.oracle.plan import read_manifest
    from fleetlab.oracle.run import run_complete
    from fleetlab.scoring import replays_needed_for

    stored = fresh_manifest_file(paths)
    ran = stored.is_file() and (paths.fresh / "run.json").is_file()
    if ran and not run_complete(paths.fresh, read_manifest(stored)):
        heavy(paths, "fresh-heavy", ["--workers", str(workers)])
    needs = [
        need
        for root in [None, *analyses(paths).values()]
        for need in replays_needed_for(scoring_inputs(world, _oracle_runs(paths), root))
    ]
    jobs, refused = fresh_jobs(world, needs)
    target = jobs_file(paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(j.model_dump_json() + "\n" for j in jobs))
    todo = bool(jobs) and not run_complete(paths.fresh, fresh_manifest(paths, jobs))
    note_step(paths, "fresh", needed=len(needs), jobs=len(jobs), todo=todo, refused=refused)
    if todo:
        heavy(paths, "fresh-heavy", ["--workers", str(workers)])


def run_fresh(paths: ChainPaths, world: World, workers: int) -> None:
    """The heavy part of `step_fresh`, in its own process: the replays and their proof
    replays (`fleetlab.oracle.run.run_here`). A replay-only run has no run job for the
    engine's chain sample, so the proof compares unmodified replays with the as-run chains
    of the observed and later runs."""
    from fleetlab.oracle.run import run_here

    jobs = [Job.model_validate_json(line) for line in jobs_file(paths).read_text().splitlines()]
    manifest = fresh_manifest(paths, jobs)
    if manifest:
        hash_file = paths.world / "hash"
        run_here(
            paths.fresh,
            manifest,
            [paths.observed, paths.later],
            world.spec.episodes_per_robot_day,
            manifest_file=fresh_manifest_file(paths),
            world_hash=hash_file.read_text().strip() if hash_file.is_file() else None,
            extend=True,
            workers=workers,
        )


# ---- score ------------------------------------------------------------------------------------


def step_score(paths: ChainPaths, world: World, **_: Any) -> dict[str, Any]:
    """Score every finished analysis against the hidden fault table and the oracle:
    `score/<condition>/report.json` and `report.md`."""
    from fleetlab.integration.chain import analyses, note_step
    from fleetlab.scoring import report_markdown, score

    started = time.time()
    out: dict[str, Any] = {}
    for condition, root in analyses(paths).items():
        inputs = scoring_inputs(world, _oracle_runs(paths), root)
        report = score(inputs)
        refused = [c for c in inputs.tool_log if c.get("delivered") is False]
        if refused:
            calls = ", ".join(
                f"session {c.get('session')} seq {c.get('seq')} ({c.get('tool')})" for c in refused
            )
            note = (
                f"the CLI refused to show the model the results of {calls}: too large for its "
                "MCP limit; they are not sources of the traceability, and the agent "
                "worked without them"
            )
            report = report.model_copy(update={"notes": [*report.notes, note]})
        target = paths.score / condition
        target.mkdir(parents=True, exist_ok=True)
        (target / "report.json").write_text(report.model_dump_json(indent=1) + "\n")
        (target / "report.md").write_text(report_markdown(report))
        out[condition] = {
            "analysis": root.name,
            "needs_fresh_replay": len(report.needs_fresh_replay),
        }
    note_step(paths, "score", started=started, ended=time.time(), analyses=out)
    return out


# ---- fixtures ---------------------------------------------------------------------------------


def step_fixtures(paths: ChainPaths, world: World, *, version: str | None = None, **_: Any) -> Path:
    """Copy the chain's files to the fixture directory; write the manifest in the repository."""
    from fleetlab.integration.chain import analyses, note_step
    from fleetlab.integration.fixtures import copy_fixtures
    from fleetlab.worlds.mini import VERSION

    version = version or VERSION
    extra = {
        "world_id": world.spec.world_id,
        "world_hash": (paths.world / "hash").read_text().strip(),
        "analyses": {c: r.name for c, r in analyses(paths).items()},
        "made_from": str(paths.root),
        "layout": "src/fleetlab/integration/chain.py (ChainPaths)",
    }
    started = time.time()
    manifest = copy_fixtures(paths.root, version, extra)
    note_step(
        paths,
        "fixtures",
        started=started,
        ended=time.time(),
        version=version,
        manifest=str(manifest),
    )
    return manifest
