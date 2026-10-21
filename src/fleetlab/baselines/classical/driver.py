"""Explicit truth-side queue runner; never imported by the blind baseline.

    uv run --locked python -m fleetlab.baselines.classical.driver \\
        --queue /new/analysis/queues/session-01.json --truth /world/world_truth.json \\
        --root /new/engine-output

The CLI launches itself through tools/heavy-run. It prints the reply export
only after every request completed or was refused, its arm episodes are present, and the
engine finalized the run with its chain checks passing. Use that export with
classical.experiments update. --resume accepts only this driver's own root
with identical inputs, source and configuration; existing cache roots are refused.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fleetlab.agents.sessions import Analysis, EngineReply, ExperimentQueue, arm_problems
from fleetlab.contracts.log import DESCRIPTION_FILE
from fleetlab.contracts.rollout import Job
from fleetlab.engine.experiments import WorldTruth, queue_dirs, run_queue
from fleetlab.engine.fleet import RunConfig
from fleetlab.log import run_status
from fleetlab.log.leaks import export_digest
from fleetlab.runtime import configure_process, repo_root
from fleetlab.scoring.provenance import certified_export
from fleetlab.toolserver.public_results import EXECUTION_FAILED, request_results_problems
from fleetlab.worlds.layout import WorldPaths

FORMAT = "fleetlab.classical-queue-driver/1"
MIOPEN_DIR = Path.home() / ".cache/robots-probes/engine/miopen"


class DriverError(RuntimeError):
    """A queue cannot be run safely, or its completion has not been established."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _tree(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise DriverError(f"missing input directory: {path}")
    return {str(p.relative_to(path)): _hash(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _source() -> dict[str, Any]:
    repo = repo_root()
    names = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "src",
            "third_party",
            "tools/heavy-run",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    # Include this entry point even during development before its first commit.
    names.append(str(Path(__file__).resolve().relative_to(repo)))
    files = {name: _hash(repo / name) for name in sorted(set(names) - {""})}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"commit": commit, "sha256": _digest(files), "files": files}


def _write(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _in_heavy_scope() -> bool:
    """tools/heavy-run names the memory-limited scope robots-heavy-<pid>."""
    return (
        re.search(r"/robots-heavy-\d+\.scope(?:/|$)", Path("/proc/self/cgroup").read_text())
        is not None
    )


def _analysis(queue_path: Path, queue: ExperimentQueue, path: Path | None) -> Analysis:
    analysis = Analysis(path.resolve() if path else queue_path.parent.parent)
    if (analysis.analysis_id, analysis.world_id) != (queue.analysis_id, queue.world_id):
        raise DriverError("queue and analysis identities differ")
    queued = {q.experiment_id: q for q in queue.experiments}
    known = {q.experiment_id: q for q in analysis.requests()}
    if not queued or len(queued) != len(queue.experiments):
        raise DriverError("queue must have distinct, nonempty experiment requests")
    if any(known.get(key) != request for key, request in queued.items()):
        raise DriverError("queue requests differ from the analysis outbox")
    if queue.session < 1 or any(q.session > queue.session for q in queue.experiments):
        raise DriverError("queue precedes a request's session")
    if Path(queue.analysis_id).name != queue.analysis_id or queue.analysis_id in (".", ".."):
        raise DriverError("analysis id must be a single directory name")
    return analysis


def _bind_world(queue: ExperimentQueue, truth_path: Path, world: WorldTruth) -> str:
    """Bind the blind public name to the truth through certified export bytes."""
    world_root = (
        truth_path.parent.parent
        if truth_path.parent.name in ("files", "world")
        else truth_path.parent
    )
    paths = WorldPaths.from_root(world_root)
    if paths.world_id != world.world_id:
        raise DriverError("truth and world records identities differ")
    export, digest = certified_export(world_root, world.world_id)
    public_id = json.loads((export / DESCRIPTION_FILE).read_text())["fleet"]["id"]
    if queue.world_id != public_id or export_digest(Path(queue.export)) != digest:
        raise DriverError("queue export belongs to another world")
    return digest


def _inputs(
    queue_path: Path,
    truth_path: Path,
    analysis: Analysis,
    queue: ExperimentQueue,
    world: WorldTruth,
) -> dict[str, Any]:
    """Fresh hashes of the same explicitly named inputs at every checkpoint."""
    return {
        "queue": {"path": str(queue_path), "sha256": _hash(queue_path)},
        "truth": {"path": str(truth_path), "sha256": _hash(truth_path)},
        "analysis": {"path": str(analysis.root), "sha256": _hash(analysis.root / "analysis.json")},
        "requests_sha256": _digest([q.model_dump(mode="json") for q in analysis.requests()]),
        "export": {"path": queue.export, "sha256": _digest(_tree(Path(queue.export)))},
        "base_logs": {
            str(Path(p).resolve()): _digest(_tree(Path(p) / "log")) for p in world.base_runs
        },
    }


def _check_snapshots(root: Path, inputs: dict[str, Any]) -> None:
    if (
        _hash(root / "queue.json") != inputs["queue"]["sha256"]
        or _hash(root / "world_truth.json") != inputs["truth"]["sha256"]
    ):
        raise DriverError("input snapshots changed")


def _check_reply(queue: ExperimentQueue, reply: EngineReply, root: Path) -> dict[str, Any]:
    expected = {q.experiment_id: q.request for q in queue.experiments}
    problems = request_results_problems(reply.results, expected)
    if any(result.message == EXECUTION_FAILED for result in reply.results):
        problems.append("experiment execution failed; resume the run before updating an answer")
    episodes = [
        episode for result in reply.results for arm in result.arms for episode in arm.episode_ids
    ]
    run_dir, export_dir = queue_dirs(root, queue.analysis_id, queue.session)
    if reply.export.resolve() != export_dir or not export_dir.is_dir():
        problems.append("reply must use the new engine root's export")
    if problems:
        raise DriverError("; ".join(problems))
    problems = arm_problems(reply)
    if problems:
        raise DriverError("; ".join(problems))
    plan_file = run_dir.parent / f"{run_dir.name}.plan.json"
    plan = json.loads(plan_file.read_text())
    if plan.get("results") != [r.model_dump(mode="json") for r in reply.results]:
        raise DriverError("reply differs from the saved engine plan")
    jobs = [Job.model_validate(j) for j in plan.get("jobs", [])]
    if sorted(j.key for j in jobs) != sorted(episodes):
        raise DriverError("saved engine jobs differ from the replied arm episodes")
    completion: dict[str, Any] = {"jobs": len(jobs), "plan_sha256": _hash(plan_file)}
    if jobs:
        status = run_status(run_dir, jobs)
        record_file = run_dir / "run.json"
        record = json.loads(record_file.read_text())
        if not status.finalized or status.pending:
            raise DriverError("engine run is not finalized or has pending jobs")
        if record.get("chain_ok") is not True:
            raise DriverError("engine chain checks did not pass")
        completion.update(run_sha256=_hash(record_file), chain_ok=True)
    completion["export_sha256"] = _digest(_tree(export_dir))
    return completion


def run(
    queue_path: Path,
    truth_path: Path,
    root: Path,
    *,
    analysis_path: Path | None = None,
    workers: int = 10,
    batch_size: int = 4,
    precision: str = "fp16_res32",
    miopen_dir: Path = MIOPEN_DIR,
    record_video: bool = True,
    resume: bool = False,
) -> EngineReply:
    """Run only inside a heavy scope. Preserve actual replies even on failed checks."""
    if not _in_heavy_scope():
        raise DriverError("use this module's CLI: simulation must run through tools/heavy-run")
    if workers < 1 or batch_size < 1:
        raise DriverError("workers and batch size must be positive")
    queue_path, truth_path, root = queue_path.resolve(), truth_path.resolve(), root.resolve()
    queue_bytes, truth_bytes = queue_path.read_bytes(), truth_path.read_bytes()
    queue = ExperimentQueue.model_validate_json(queue_bytes)
    world = WorldTruth.model_validate_json(truth_bytes)
    if world.policy.precision != precision:
        raise DriverError("queue/world identity or nominal policy precision differs")
    certified_digest = _bind_world(queue, truth_path, world)
    analysis = _analysis(queue_path, queue, analysis_path)
    if not resume and any(q.experiment_id in analysis.results_by_id() for q in queue.experiments):
        raise DriverError("queue results are already recorded; resume their original driver root")
    protected = [analysis.root, queue_path, truth_path, Path(queue.export).resolve()]
    protected.extend(Path(p).resolve() for p in world.base_runs)
    if any(root == p or root in p.parents or p in root.parents for p in protected):
        raise DriverError("engine root must be separate from the inputs and original world runs")
    config = RunConfig(out="", workers=workers, batch_size=batch_size, precision=precision)
    inputs = _inputs(queue_path, truth_path, analysis, queue, world)
    if (
        inputs["queue"]["sha256"] != hashlib.sha256(queue_bytes).hexdigest()
        or inputs["truth"]["sha256"] != hashlib.sha256(truth_bytes).hexdigest()
    ):
        raise DriverError("inputs changed while initializing the queue")
    inputs["certified_export_digest"] = certified_digest
    engine_queue = queue.model_copy(update={"world_id": world.world_id})
    provenance = {
        "format": FORMAT,
        "inputs": inputs,
        "source": _source(),
        "config": asdict(config),
        "record_video": record_video,
        "miopen_dir": str(miopen_dir.resolve()),
        "miopen_manifest_sha256": _hash(miopen_dir / "pinned/pinned.json")
        if (miopen_dir / "pinned/pinned.json").is_file()
        else None,
    }
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    elif not (root / "driver.json").is_file():
        raise DriverError("resume requires a root created by this driver")
    with (root / ".driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / "driver.json"
        if resume:
            if json.loads(manifest.read_text()) != provenance:
                raise DriverError("resume inputs, source or configuration differ; use a new root")
        else:
            _write(manifest, provenance)
            (root / "queue.json").write_bytes(queue_bytes)
            (root / "world_truth.json").write_bytes(truth_bytes)
            _write(root / "engine-queue.json", engine_queue.model_dump(mode="json"))
        _check_snapshots(root, inputs)
        if (
            ExperimentQueue.model_validate_json((root / "engine-queue.json").read_text())
            != engine_queue
        ):
            raise DriverError("engine queue snapshot changed")
        attempts = root / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / f"{len(list(attempts.glob('*.json'))) + 1:04d}.json"
        record: dict[str, Any] = {"started": time.time(), "complete": False}
        _write(attempt, record)
        try:
            os.environ["FLEETLAB_DATA_DIR"] = str(root / "runtime")
            os.environ["FLEETLAB_MIOPEN_DIR"] = str(miopen_dir.resolve())
            configure_process()
            reply_file = root / "reply.json"
            if reply_file.is_file():
                reply = EngineReply.model_validate_json(reply_file.read_text())
            else:
                reply = run_queue(
                    root / "engine-queue.json",
                    world,
                    root,
                    config=config,
                    record_video=record_video,
                )
            record["reply"] = reply.model_dump(mode="json")
            _write(attempt, record)
            record["completion"] = _check_reply(queue, reply, root)
            if _source() != provenance["source"]:
                raise DriverError("source changed during queue execution")
            record["inputs_verified"] = False
            fresh_inputs = _inputs(queue_path, truth_path, analysis, queue, world)
            fresh_inputs["certified_export_digest"] = certified_digest
            record["inputs_sha256"] = _digest(fresh_inputs)
            if fresh_inputs != inputs:
                raise DriverError("inputs changed during queue execution")
            _bind_world(queue, truth_path, world)
            _check_snapshots(root, inputs)
            if (
                ExperimentQueue.model_validate_json((root / "engine-queue.json").read_text())
                != engine_queue
            ):
                raise DriverError("engine queue snapshot changed")
            record["inputs_verified"] = True
            completion_file = root / "reply-completion.json"
            if completion_file.is_file():
                if json.loads(completion_file.read_text()) != record["completion"]:
                    raise DriverError("saved completion artifacts changed")
            else:
                _write(completion_file, record["completion"])
            if not reply_file.is_file():
                _write(reply_file, reply.model_dump(mode="json"))
            existing = analysis.results_by_id()
            for result in reply.results:
                if result.experiment_id in existing and existing[result.experiment_id] != result:
                    raise DriverError(
                        f"analysis already has a different result for {result.experiment_id}"
                    )
            analysis.record_results([r for r in reply.results if r.experiment_id not in existing])
            # Public request refusals belong to the method's record. Machine
            # failures instead raise before any reply is published, retaining
            # the engine plan/checkpoint for --resume.
            record.update(
                complete=True,
                completed_requests=sum(r.status == "done" for r in reply.results),
                refused_requests=[r.experiment_id for r in reply.results if r.status == "refused"],
                reply_sha256=_hash(reply_file),
            )
            return reply
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["ended"] = time.time()
            _write(attempt, record)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--analysis", type=Path, help="analysis directory; inferred from the queue by default"
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16_res32", "fp16", "fp32"), default="fp16_res32")
    parser.add_argument(
        "--miopen-dir",
        type=Path,
        default=Path(os.environ.get("FLEETLAB_MIOPEN_DIR", str(MIOPEN_DIR))),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if not _in_heavy_scope():
        command = [
            str(repo_root() / "tools/heavy-run"),
            sys.executable,
            "-m",
            "fleetlab.baselines.classical.driver",
            *arguments,
        ]
        return subprocess.run(command, check=False).returncode
    try:
        reply = run(
            args.queue,
            args.truth,
            args.root,
            analysis_path=args.analysis,
            workers=args.workers,
            batch_size=args.batch_size,
            precision=args.precision,
            miopen_dir=args.miopen_dir,
            record_video=not args.no_video,
            resume=args.resume,
        )
    except (DriverError, OSError, ValueError) as error:
        print(f"classical queue driver: {error}", file=sys.stderr)
        return 1
    print(reply.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
