"""The oracle's steps on a world root: plan, run (heavy), report, fresh; each resumable.

```
plan    the replay manifest and its counts      oracle/jobs.jsonl, oracle/plan.json
run     the replays (one heavy job), the proof  oracle/runs/<world>-oracle/
report  table, gains, intervals, chance, checks oracle/oracle.json, summary.md, episodes.parquet;
                                                truth/oracle.parquet in the run with the replays
fresh   the lookups some fix sets miss          oracle/fresh/<name>/: jobs.jsonl, run/, values.json
```

`Oracle` opens a world root with its runs and the oracle directory; its `table()` reads
every run the world and the oracle have (and any extra run directories named), so that
the miniature fleet, whose replays sit in its `w10-later` run, is read like a world with
an oracle run of its own.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fleetlab.contracts.actions import FixSet
from fleetlab.contracts.faults import FaultSpec
from fleetlab.contracts.oracle import POWER_SET_CAP, ReplayNeed
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.oracle.causes import (
    CAUSES_FILE,
    MeasuredCauses,
    curve_observed_estimate,
    measure_causes,
    observed_requests,
    required_observed_episodes,
)
from fleetlab.oracle.fresh import (
    Needs,
    catalogue_fix_sets,
    chance_fix_sets,
    fresh_jobs,
    fresh_proofs,
    needs_of_fix_sets,
    needs_of_oracle,
    value_all,
)
from fleetlab.oracle.plan import (
    OraclePlan,
    check_jobs,
    manifest_text,
    plan_world,
    read_manifest,
    write_plan,
)
from fleetlab.oracle.report import (
    OracleReport,
    compute_report,
    write_report,
    write_text,
    write_truth_table,
)
from fleetlab.oracle.run import (
    WORKERS,
    RunRefused,
    execute_third,
    note,
    proof_failed,
    run_complete,
    run_heavy,
    run_here,
)
from fleetlab.oracle.table import RunRows, TableBuild, build_table
from fleetlab.oracle.verification import Execution, ReplayVerifier, once_replay_risk
from fleetlab.oracle.world import (
    DEFAULT_REGISTRY,
    OracleLayout,
    OracleWorld,
    all_runs,
    load_world,
    recorded_sources,
    world_ready,
    world_runs,
)


@dataclass
class Oracle:
    """A world root, its runs and its oracle directory."""

    root: Path
    world: OracleWorld
    layout: OracleLayout
    extra_runs: list[Path] = field(default_factory=list)
    confirm: bool | str = True
    cap: int = POWER_SET_CAP
    proof: bool = True  # proof replays in the plan: the replays run in a run of their own
    claim_windows: tuple[str, ...] = ("scoring",)

    @classmethod
    def open(
        cls,
        root: Path | str,
        oracle_dir: Path | str | None = None,
        extra_runs: Sequence[Path | str] = (),
        *,
        confirm: bool | str = True,
        cap: int = POWER_SET_CAP,
        proof: bool = True,
        registry: str = DEFAULT_REGISTRY,
        windows: Sequence[str] = (),
    ) -> Oracle:
        base = Path(root).expanduser().resolve()
        world = load_world(base, registry)
        layout = OracleLayout(
            base / "oracle" if oracle_dir is None else Path(oracle_dir).expanduser().resolve(),
            world.world_id,
        )
        recorded = recorded_sources(layout, world.world_hash)
        sources = list(
            dict.fromkeys(p.expanduser().resolve() for p in [*recorded, *map(Path, extra_runs)])
        )
        if set(windows) - {"scoring", "observed", "experiment"}:
            raise ValueError("claim windows are scoring, observed or experiment")
        return cls(
            base,
            world,
            layout,
            sources,
            confirm,
            cap,
            proof,
            tuple(dict.fromkeys(["scoring", *windows])),
        )

    @property
    def steps(self) -> Path:
        return self.layout.steps

    def world_runs(self) -> list[Path]:
        """The runs that hold the world as run (for the proof): the world's and the extras."""
        return [*world_runs(self.root), *self.extra_runs]

    def runs(self) -> list[Path]:
        return all_runs(self.root, self.layout, self.extra_runs)

    def remember_sources(self) -> None:
        """Persist explicit row sources so a later invocation needs no --runs."""
        write_text(
            self.layout.root / "sources.json",
            json.dumps(
                {
                    "world_hash": self.world.world_hash,
                    "runs": [str(p.resolve()) for p in self.runs()],
                },
                indent=1,
            )
            + "\n",
        )

    def plan(self) -> OraclePlan:
        return plan_world(self.world, confirm=self.confirm, cap=self.cap, proof=self.proof)

    def table(self, *, all_windows: bool = False) -> TableBuild:
        """Scoring outcomes by default; all-window lookups for cited examples on request."""
        try:
            verifier = self.verifier()
        except RunRefused as exc:
            build = build_table(
                self.world.identities if all_windows else self.world.scoring(),
                self.world.excluded,
                self.runs(),
                self.world.world_hash,
            )
            build.runs.append(RunRows(self.layout.root / "verification", refused=str(exc)))
            return build
        verifier.validate()
        return build_table(
            self.world.identities if all_windows else self.world.scoring(),
            self.world.excluded,
            self.runs(),
            self.world.world_hash,
            verifier,
        )

    def verifier(
        self,
        *,
        execute: bool = False,
        workers: int = WORKERS,
        overrides: dict[str, Any] | None = None,
    ) -> ReplayVerifier:
        windows = {
            i.episode_id: name
            for name in ("observed", "experiment", "scoring")
            for i in self.world.window(name)
        }
        callback = (
            (
                lambda folder, job: execute_third(
                    folder,
                    job,
                    self.world.spec.episodes_per_robot_day,
                    self.world.world_hash,
                    workers,
                    overrides,
                )
            )
            if execute
            else None
        )
        verifier = ReplayVerifier(
            self.layout.root / "verification", windows, self.world.world_hash, callback
        )
        if execute:
            for name, ids in self.verification_required().items():
                verifier.required[name] = sorted(set(verifier.required.get(name, [])) | set(ids))
        return verifier

    def verification_required(self) -> dict[str, list[str]]:
        required = {
            name: [i.episode_id for i in self.world.window(name)] for name in self.claim_windows
        }
        plan = self.layout.root / "causes" / "plan.json"
        if plan.is_file():
            record = json.loads(plan.read_text())
            if record["world_hash"] != self.world.world_hash:
                raise RunRefused("observed plan belongs to another world")
            selected = record["observed_episodes"]
            required["observed"] = sorted(set(required.get("observed", [])) | set(selected))
        return required

    def step_verification_plan(self) -> dict[str, Any]:
        """Count reusable seconds without simulation or a new engine manifest."""
        from fleetlab.oracle.table import rows_of_run

        required = self.verification_required()
        seconds: set[str] = set()
        for path in self.runs():
            _, rows = rows_of_run(path, self.world.by_id(), self.world.world_hash)
            seconds.update(
                e.job.identity.episode_id
                for e in rows.executions
                if e.job.mode is JobMode.replay and not (e.job.remove or e.job.add)
            )
        counts = {name: len(set(ids) - seconds) for name, ids in required.items()}
        result = {
            "second_executions_remaining": sum(counts.values()),
            "windows": counts,
            "gpu_minutes_estimate": sum(counts.values()) / 1500 * 60,
            "basis": "1,500/hour; excludes queue and disputed attempts' further executions",
        }
        return result

    def step_verify(
        self, *, workers: int = WORKERS, here: bool = False, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Add the second executions; reuse existing proofs and all finished jobs."""
        from fleetlab.oracle.table import rows_of_run

        if waiting := world_ready(self.root):
            raise RunRefused(f"the world's runs are not ready: {waiting}")
        if not here:
            verifier = self.verifier()
            required = self.verification_required()
            if verifier.path.is_file() and all(
                set(ids) <= set(verifier.required.get(name, [])) for name, ids in required.items()
            ):
                try:
                    verifier.validate()
                    if not any(c["missing"] for c in verifier.summary()["coverage"]):
                        checked = build_table(
                            self.world.identities,
                            self.world.excluded,
                            self.runs(),
                            self.world.world_hash,
                            verifier,
                        )
                        if checked.problems():
                            raise RunRefused("; ".join(checked.problems()))
                        return {"jobs_run_now": 0, **verifier.summary()}
                except RunRefused as exc:
                    if "execution required" not in str(exc):
                        raise
            status = run_heavy(
                [
                    "verify",
                    str(self.root),
                    "--oracle-dir",
                    str(self.layout.root),
                    "--workers",
                    str(workers),
                ]
                + [a for name in self.claim_windows for a in ("--window", name)]
                + [a for p in self.extra_runs for a in ("--runs", str(p))],
                self.steps,
                "verify",
            )
            if status:
                raise RunRefused(f"verification ended with exit status {status}")
            return dict(json.loads(self.steps.read_text())["verify"])
        verifier = self.verifier(execute=True, workers=workers, overrides=overrides)
        for name, ids in self.verification_required().items():
            verifier.required[name] = sorted(set(verifier.required.get(name, [])) | set(ids))
        verifier.save()
        # A completed pre-rule first pass may have stopped at its old proof gate.
        # Reclassify its existing files before deciding which seconds are absent.
        # The engine is not loaded when all of that manifest's jobs are finalized.
        if self.layout.jobs.is_file() and (self.layout.run / "run.json").is_file():
            from fleetlab.contracts.rollout import RunLayout
            from fleetlab.oracle.table import jobs_merged

            old_manifest = self.stored_manifest()
            if (
                old_manifest
                and not run_complete(self.layout.run, old_manifest)
                and all(RunLayout(self.layout.run).is_done(j) for j in old_manifest)
                and jobs_merged(self.layout.run)
            ):
                problems = check_jobs(self.world, old_manifest)
                if problems:
                    raise RunRefused("the old manifest does not validate: " + "; ".join(problems))
                run_here(
                    self.layout.run,
                    old_manifest,
                    self.world_runs(),
                    self.world.spec.episodes_per_robot_day,
                    manifest_file=self.layout.jobs,
                    world_hash=self.world.world_hash,
                    workers=workers,
                    overrides=overrides,
                    verifier=verifier,
                )
        wanted = {e for episodes in verifier.required.values() for e in episodes}
        originals: dict[str, Execution] = {}
        seconds: dict[str, Execution] = {}
        # Prefer this oracle's proof: it was already compared by run_here.
        for path in sorted(self.runs(), key=lambda p: p != self.layout.run):
            _, source_info = rows_of_run(path, self.world.by_id(), self.world.world_hash)
            for execution in source_info.executions:
                job = execution.job
                episode = job.identity.episode_id
                if episode not in wanted or job.remove or job.add:
                    continue
                if job.mode is JobMode.run:
                    originals.setdefault(episode, execution)
                else:
                    seconds.setdefault(episode, execution)
        missing_originals = wanted - originals.keys()
        if missing_originals:
            raise RunRefused(f"missing proven as-run executions: {sorted(missing_originals)}")
        folder = self.layout.root / "verification"
        manifest_path = folder / "jobs.jsonl"
        stored = read_manifest(manifest_path) if manifest_path.is_file() else []
        keys = {j.key for j in stored}
        additions = [
            Job(identity=self.world.by_id()[e], mode=JobMode.replay)
            for e in self.world.by_id()
            if e in wanted and e not in seconds
        ]
        manifest = [*stored, *(j for j in additions if j.key not in keys)]
        for job in manifest:
            if (
                job.identity != self.world.by_id().get(job.identity.episode_id)
                or job.mode is not JobMode.replay
                or job.remove
                or job.add
                or job.from_snapshot
            ):
                raise RunRefused("the verification manifest is not unmodified world episodes")
        info: dict[str, Any] = {"jobs_run_now": 0}
        if manifest:
            info = run_here(
                folder / "run",
                manifest,
                self.world_runs(),
                self.world.spec.episodes_per_robot_day,
                manifest_file=manifest_path,
                world_hash=self.world.world_hash,
                extend=True,
                workers=workers,
                overrides=overrides,
                verifier=verifier,
            )
            _, rows = rows_of_run(folder / "run", self.world.by_id(), self.world.world_hash)
            seconds.update({x.job.identity.episode_id: x for x in rows.executions})
        for episode in sorted(wanted):
            if episode not in seconds:
                raise RunRefused(f"{episode}: the second execution is missing")
            verifier.settle(originals[episode], seconds[episode])
        # The other gate uses this same classifier: conflicting changed rows.
        build_table(
            self.world.identities, self.world.excluded, self.runs(), self.world.world_hash, verifier
        )
        verifier.save()
        info = {**info, **verifier.summary()}
        self.remember_sources()
        note(self.steps, "verify", ended=time.time(), **info)
        return info

    # ---- plan ----------------------------------------------------------------------------------

    def step_plan(self, *, replace: bool = False) -> tuple[OraclePlan, bool, list[str]]:
        """Write the manifest (unless the same is there); (plan, written, problems)."""
        started = time.time()
        plan = self.plan()
        problems = check_jobs(self.world, plan.manifest())
        if problems:
            raise ValueError("the planned manifest does not validate: " + "; ".join(problems[:5]))
        written = write_plan(self.layout, self.world, plan, replace=replace)
        self.remember_sources()
        note(self.steps, "plan", started=started, ended=time.time(), **plan.counts())
        return plan, written, problems

    def stored_manifest(self) -> list[Any]:
        if not self.layout.jobs.is_file():
            raise FileNotFoundError(f"{self.layout.jobs}: plan first")
        return read_manifest(self.layout.jobs)

    # ---- run -----------------------------------------------------------------------------------

    def run_is_complete(self) -> bool:
        return self.layout.jobs.is_file() and run_complete(self.layout.run, self.stored_manifest())

    def step_run(
        self, *, workers: int = WORKERS, here: bool = False, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the stored manifest: in this process with `here`, else as a heavy job."""
        manifest = self.stored_manifest()
        if not manifest:
            note(self.steps, "run", skipped="nothing to replay")
            return {"jobs": 0}
        waiting = world_ready(self.root)
        if waiting:
            raise RuntimeError(f"the world's runs are not ready: {waiting}")
        failed = proof_failed(self.layout.run, manifest, allow_adjudication=True)
        if failed:
            raise RunRefused(failed)
        self.remember_sources()
        if not here:
            if self.run_is_complete():
                verification = self.step_verify(workers=workers)
                return {"jobs": len(manifest), "complete": True, "verification": verification}
            status = run_heavy(
                [
                    "run",
                    str(self.root),
                    "--oracle-dir",
                    str(self.layout.root),
                    "--workers",
                    str(workers),
                ]
                + [a for name in self.claim_windows for a in ("--window", name)]
                + [a for p in self.extra_runs for a in ("--runs", str(p))],
                self.steps,
                "run",
            )
            if status != 0:
                raise RuntimeError(f"the heavy run ended with exit status {status}")
            return dict(json.loads(self.steps.read_text())["run"])
        problems = check_jobs(self.world, manifest)
        if problems:
            raise ValueError("the stored manifest does not validate: " + "; ".join(problems[:5]))
        note(self.steps, "run", started=time.time())
        verifier = self.verifier(execute=True, workers=workers, overrides=overrides)
        info = run_here(
            self.layout.run,
            manifest,
            self.world_runs(),
            self.world.spec.episodes_per_robot_day,
            manifest_file=self.layout.jobs,
            world_hash=self.world.world_hash,
            workers=workers,
            overrides=overrides,
            verifier=verifier,
        )
        info["verification"] = self.step_verify(workers=workers, here=True, overrides=overrides)
        note(self.steps, "run", ended=time.time(), **info)
        return info

    # ---- report --------------------------------------------------------------------------------

    def measured_causes(self, build: TableBuild | None = None) -> MeasuredCauses:
        from fleetlab.oracle.causes import logged_outcomes

        build = build or self.table(all_windows=True)
        measured = measure_causes(self.world, build.table, logged_outcomes(build))
        verified = set(build.verification.get("verified_episodes", []))
        for name, ids in {
            "scoring": [i.episode_id for i in self.world.scoring()],
            "observed": required_observed_episodes(self.world, measured),
        }.items():
            if missing := set(ids) - verified:
                measured.problems.append(f"{name}: {len(missing)} baselines not executed twice")
                measured.complete = False
        return measured

    def check_measured_causes(self, measured: MeasuredCauses) -> None:
        path = self.layout.root / CAUSES_FILE
        if path.is_file():
            old = MeasuredCauses.model_validate_json(path.read_text())
            if old.world_hash != measured.world_hash:
                raise RunRefused("measured causes belong to another world")
            if old.complete and not measured.complete:
                raise RunRefused("complete measured causes preserved; use another --oracle-dir")

    def write_measured_causes(self, measured: MeasuredCauses) -> None:
        self.check_measured_causes(measured)
        write_text(self.layout.root / CAUSES_FILE, measured.model_dump_json(indent=1) + "\n")

    def check_cause_manifest(self, manifest: Sequence[Job]) -> None:
        identities = {i.episode_id: i for i in self.world.window("observed")}
        for job in manifest:
            if (
                job.identity != identities.get(job.identity.episode_id)
                or job.mode is not JobMode.replay
                or job.add
                or job.from_snapshot
            ):
                raise RunRefused("the observed manifest is not a removal from this world's reset")

    def step_causes_plan(self, *, estimate: bool = False) -> dict[str, Any]:
        """Plan the next measured stage, before queueing; retain every completed job."""
        from fleetlab.contracts.rollout import RunLayout

        if estimate:
            return curve_observed_estimate(self.world)
        build = self.table(all_windows=True)
        if build.problems():
            raise RunRefused("; ".join(build.problems()))
        measured = self.measured_causes(build)
        verified = set(build.verification.get("verified_episodes", []))
        if any(i.episode_id not in verified for i in self.world.scoring()):
            raise RunRefused(
                "scoring measurements required with double execution; run verify first"
            )
        self.check_measured_causes(measured)
        requests = observed_requests(self.world, measured, build.table)
        jobs, refused = fresh_jobs(self.world.by_id(), requests)
        if refused:
            raise RunRefused("observed cause requests refused: " + str(refused))
        folder = self.layout.root / "causes"
        path = folder / "jobs.jsonl"
        stored = read_manifest(path) if path.is_file() else []
        self.check_cause_manifest(stored)
        existing_seconds = {
            e.job.identity.episode_id
            for r in build.runs
            for e in r.executions
            if e.job.mode is JobMode.replay and not (e.job.remove or e.job.add)
        }
        selected = {j.identity.episode_id for j in stored} | set(
            required_observed_episodes(self.world, measured)
        )
        # Every observed removal baseline is checked, not just the proof sample.
        seconds = [
            Job(identity=self.world.by_id()[e], mode=JobMode.replay)
            for e in sorted(selected - existing_seconds)
        ]
        have = {j.key for j in stored}
        additions = {j.key: j for j in [*jobs, *seconds] if j.key not in have}
        manifest = [*stored, *additions.values()]
        folder.mkdir(parents=True, exist_ok=True)
        write_text(path, manifest_text(manifest))
        layout = RunLayout(folder / "run")
        pending = [j for j in manifest if not layout.is_done(j)]
        record = {
            "world_hash": self.world.world_hash,
            "stage": "measured",
            "observed_episodes": sorted(selected),
            "jobs": len(manifest),
            "removal_replays": sum(bool(j.remove or j.add) for j in manifest),
            "second_executions": sum(not (j.remove or j.add) for j in manifest),
            "jobs_remaining": len(pending),
            "removal_replays_remaining": sum(bool(j.remove or j.add) for j in pending),
            "second_executions_remaining": sum(not (j.remove or j.add) for j in pending),
            "gpu_minutes_estimate": [len(pending) / 1500 * 60, len(pending) / 1500 * 60],
            "gpu_rate_basis": "1,500 replays/hour from entry 15; queue/adjudications excluded",
            "measurement_complete": measured.complete,
            "problems": measured.problems,
        }
        write_text(folder / "plan.json", json.dumps(record, indent=1) + "\n")
        self.write_measured_causes(measured)
        self.remember_sources()
        note(self.steps, "causes", planned=time.time(), **record)
        return record

    def step_causes_run(
        self,
        *,
        workers: int = WORKERS,
        here: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute only the stated stage; a following plan may request joint rows."""
        folder = self.layout.root / "causes"
        path = folder / "jobs.jsonl"
        manifest = read_manifest(path)
        self.check_cause_manifest(manifest)
        self.verification_required()
        if not manifest:
            result = {
                "jobs_run_now": 0,
                "verification": self.step_verify(workers=workers, here=here, overrides=overrides),
            }
            self.write_measured_causes(self.measured_causes())
            return result
        if waiting := world_ready(self.root):
            raise RunRefused(f"the world's runs are not ready: {waiting}")
        if not here:
            if run_complete(folder / "run", manifest):
                result = {"jobs_run_now": 0, "verification": self.step_verify(workers=workers)}
                self.write_measured_causes(self.measured_causes())
                return result
            status = run_heavy(
                [
                    "causes",
                    str(self.root),
                    "--oracle-dir",
                    str(self.layout.root),
                    "--run-only",
                    "--workers",
                    str(workers),
                    *[a for name in self.claim_windows for a in ("--window", name)],
                    *[a for p in self.extra_runs for a in ("--runs", str(p))],
                ],
                self.steps,
                "causes",
            )
            if status:
                raise RunRefused(f"observed replays ended with exit status {status}")
            return dict(json.loads(self.steps.read_text())["causes"])
        # An interrupted target is resumed; partial rows are not borrowed proofs.
        verifier = self.verifier(execute=True, workers=workers, overrides=overrides)
        verifier.validate()
        build = build_table(
            self.world.identities,
            self.world.excluded,
            [p for p in self.runs() if p != (folder / "run").resolve()],
            self.world.world_hash,
            verifier,
        )
        if build.problems():
            raise RunRefused("; ".join(build.problems()))
        selected = set(json.loads((folder / "plan.json").read_text())["observed_episodes"])
        references = [
            e
            for r in build.runs
            for e in r.executions
            if e.job.identity.episode_id in selected
            and e.job.mode is JobMode.replay
            and not (e.job.remove or e.job.add)
        ]
        own_keys = {j.key for j in manifest}
        references = list({e.job.key: e for e in references if e.job.key not in own_keys}.values())
        # Keep an already existing proof rather than execute it again in this run.
        note(self.steps, "causes", started=time.time())
        info = run_here(
            folder / "run",
            manifest,
            self.world_runs(),
            self.world.spec.episodes_per_robot_day,
            manifest_file=path,
            world_hash=self.world.world_hash,
            extend=True,
            workers=workers,
            overrides=overrides,
            verifier=verifier,
            proof_replays=references,
        )
        info["verification"] = self.step_verify(workers=workers, here=True, overrides=overrides)
        self.write_measured_causes(self.measured_causes())
        note(self.steps, "causes", ended=time.time(), **info)
        return info

    def parquet_run(self) -> Path | None:
        """The run that gets `truth/oracle.parquet`: the oracle's own, if it ran."""
        if (self.layout.run / "run.json").is_file():
            return self.layout.run
        return None

    def step_report(
        self,
        *,
        boot: int = 1000,
        draws: int = 200,
        parquet_into: Path | None = None,
        truth_table: bool = True,
        build: TableBuild | None = None,
    ) -> OracleReport:
        """Compute and write the report; `truth/oracle.parquet` goes into `parquet_into`,
        else into the oracle's own run if it ran, else nowhere; never with
        `truth_table=False`. The world's runs are written only when named (the miniature
        fleet's chain names `runs/w10-later`, where its replays are)."""
        started = time.time()
        build = build or self.table()
        all_windows = self.table(all_windows=True)
        # The plan as stored (what ran), else the plan this oracle would make.
        stored = self.layout.plan
        plan = json.loads(stored.read_text()) if stored.is_file() else self.plan().counts()
        report, episodes = compute_report(
            self.world,
            build,
            plan,
            boot=boot,
            draws=draws,
            cap=self.cap,
            measured=self.measured_causes(all_windows),
        )
        measured = MeasuredCauses.model_validate(report.measured_causes)
        self.check_measured_causes(measured)
        report.replay_verification = build.verification
        report.replay_verification["residual"] = once_replay_risk(
            [e for r in build.runs for e in r.executions], report.scoring_episodes
        )
        report.replay_verification["residual_by_window"] = {
            name: once_replay_risk(
                [
                    e
                    for r in all_windows.runs
                    for e in r.executions
                    if e.job.identity.day in getattr(self.world.spec.windows, name)
                ],
                len(self.world.window(name)),
            )
            for name in ("scoring", "observed")
        }
        for coverage in report.replay_verification.get("coverage", []):
            if coverage["missing"]:
                report.problems.append(
                    f"incomplete double execution of {coverage['window']}: "
                    f"{len(coverage['missing'])} attempts"
                )
        write_report(self.layout.root, report, episodes)
        self.write_measured_causes(measured)
        self.remember_sources()
        # Never into a world run unless named: the world's runs are read, not written.
        target = (parquet_into or self.parquet_run()) if truth_table else None
        if target is not None:
            write_truth_table(Path(target), report, self.world.spec.episodes_per_robot_day)
        note(
            self.steps,
            "report",
            started=started,
            ended=time.time(),
            rows=report.rows,
            truth_table=None if target is None else str(target),
            problems=len(report.problems),
        )
        return report

    # ---- fresh ---------------------------------------------------------------------------------

    def fresh_needs(
        self,
        fix_sets: Sequence[tuple[str, FixSet]],
        *,
        catalogue: bool = False,
        oracle_items: bool = False,
        build: TableBuild | None = None,
    ) -> Needs:
        table = (build or self.table()).table
        truth, current = self.world.truth(), self.world.current()
        needs = needs_of_fix_sets(fix_sets, table, truth, current)
        if catalogue:
            needs_of_fix_sets(
                catalogue_fix_sets(self.world.catalogue()), table, truth, current, needs
            )
        if oracle_items:
            needs_of_oracle(table, needs)
        return needs

    def step_fresh_plan(
        self,
        name: str,
        fix_sets: Sequence[tuple[str, FixSet]],
        *,
        catalogue: bool = False,
        oracle_items: bool = False,
        chance: bool = False,
        draws: int = 200,
        requests: Sequence[ReplayNeed] = (),
    ) -> dict[str, Any]:
        """Collect what the fix sets miss and write the fresh set's manifest.

        Once the set has run, its manifest stays: planning again (after the run, the
        table holds its rows and the needs shrink) keeps it as long as the fix sets need
        nothing it lacks, and refuses otherwise (another name, another set)."""
        folder = self.layout.fresh(name)
        if chance:
            fix_sets = [
                *fix_sets,
                *chance_fix_sets(self.world.catalogue(), self.world.spec.seed, draws),
            ]
        needs = self.fresh_needs(fix_sets, catalogue=catalogue, oracle_items=oracle_items)
        if requests:
            examples = self.table(all_windows=True).table
            needs.add(
                "requested episodes",
                [r for r in requests if examples.success(r.episode_id, r.faults) is None],
            )
        jobs, refused = fresh_jobs(self.world.by_id(), needs.unique())
        jobs_file = folder / "jobs.jsonl"
        if (folder / "run" / "run.json").is_file() and jobs_file.is_file():
            stored = read_manifest(jobs_file)
            have = {j.key for j in stored}
            lacking = [j for j in jobs if j.key not in have]
            if lacking:
                raise ValueError(
                    f"{jobs_file} has run and lacks {len(lacking)} replays these fix sets need; "
                    "use another name"
                )
            jobs = [j for j in stored if j.remove or j.add]
            proofs = [j for j in stored if not (j.remove or j.add)]
        else:
            proofs = fresh_proofs(jobs) if jobs else []
            # A requested healthy outcome on a fault-free episode is itself an
            # unmodified proof replay. Include it once, even outside the sample.
            unmodified = [j for j in jobs if not (j.remove or j.add)]
            proofs = list({j.key: j for j in [*proofs, *unmodified]}.values())
            jobs = [j for j in jobs if j.remove or j.add]
            folder.mkdir(parents=True, exist_ok=True)
            jobs_file.write_text(manifest_text([*jobs, *proofs]))
        record = {
            "name": name,
            "labels": [label for label, _ in fix_sets],
            "fix_sets": [{"label": label, **fs.model_dump(mode="json")} for label, fs in fix_sets],
            "catalogue": catalogue,
            "oracle_items": oracle_items,
            "chance": chance,
            "chance_seed": self.world.spec.seed if chance else None,
            "chance_draws": draws if chance else None,
            "requests": [
                {
                    "episode_id": r.episode_id,
                    "faults": [f.model_dump(mode="json") for f in r.faults],
                }
                for r in requests
            ],
            "needs": {label: len(n) for label, n in needs.by_label.items()},
            "errors": needs.errors,
            "jobs": len(jobs),
            "proof": len(proofs),
            "refused": refused,
        }
        (folder / "plan.json").write_text(json.dumps(record, indent=1) + "\n")
        self.remember_sources()
        note(self.steps, f"fresh-{name}", planned=time.time(), jobs=len(jobs), proof=len(proofs))
        return record

    def step_fresh_run(
        self,
        name: str,
        *,
        workers: int = WORKERS,
        here: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        folder = self.layout.fresh(name)
        manifest = read_manifest(folder / "jobs.jsonl")
        if not manifest:
            return {"jobs": 0}
        step = f"fresh-{name}"
        waiting = world_ready(self.root)
        if waiting:
            raise RuntimeError(f"the world's runs are not ready: {waiting}")
        failed = proof_failed(folder / "run", manifest, allow_adjudication=True)
        if failed:
            raise RunRefused(failed)
        self.remember_sources()
        if not here:
            if run_complete(folder / "run", manifest):
                return {
                    "jobs": len(manifest),
                    "complete": True,
                    "verification": self.step_verify(workers=workers),
                }
            status = run_heavy(
                [
                    "fresh",
                    str(self.root),
                    "--oracle-dir",
                    str(self.layout.root),
                    "--name",
                    name,
                    "--run-only",
                    "--workers",
                    str(workers),
                ]
                + [a for name in self.claim_windows for a in ("--window", name)]
                + [a for p in self.extra_runs for a in ("--runs", str(p))],
                self.steps,
                step,
            )
            if status != 0:
                raise RuntimeError(f"the heavy fresh run ended with exit status {status}")
            return dict(json.loads(self.steps.read_text())[step])
        note(self.steps, step, started=time.time())
        info = run_here(
            folder / "run",
            manifest,
            self.world_runs(),
            self.world.spec.episodes_per_robot_day,
            manifest_file=folder / "jobs.jsonl",
            world_hash=self.world.world_hash,
            workers=workers,
            overrides=overrides,
            verifier=self.verifier(execute=True, workers=workers, overrides=overrides),
        )
        info["verification"] = self.step_verify(workers=workers, here=True, overrides=overrides)
        note(self.steps, step, ended=time.time(), **info)
        return info

    def fresh_fix_sets(self, name: str) -> list[tuple[str, FixSet]]:
        """The fix sets a fresh set was planned for (its plan.json)."""
        record = json.loads((self.layout.fresh(name) / "plan.json").read_text())
        return [
            (str(f["label"]), FixSet.model_validate({k: v for k, v in f.items() if k != "label"}))
            for f in record["fix_sets"]
        ]

    def step_fresh_value(
        self, name: str, fix_sets: Sequence[tuple[str, FixSet]] | None = None
    ) -> list[dict[str, Any]]:
        """Value the fix sets with every row the runs hold now (`values.json`)."""
        fix_sets = self.fresh_fix_sets(name) if fix_sets is None else fix_sets
        build = self.table()
        values = value_all(fix_sets, build.table, self.world.truth(), self.world.current())
        folder = self.layout.fresh(name)
        record = json.loads((folder / "plan.json").read_text())
        if record.get("requests"):
            examples = self.table(all_windows=True).table
            for request in record["requests"]:
                faults = tuple(FaultSpec.model_validate(f) for f in request["faults"])
                success = examples.success(request["episode_id"], faults)
                values.append(
                    {
                        "label": "requested episode",
                        **request,
                        "success": success,
                        "missing": None if success is not None else "replay outcome missing",
                    }
                )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "values.json").write_text(json.dumps(values, indent=1) + "\n")
        return values
