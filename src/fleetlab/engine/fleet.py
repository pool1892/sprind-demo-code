"""A run: a manifest of jobs on worker processes and one inference server (contracts, section 10).

Layout (xvla-speed section 5, the fastest measured here): the policy is loaded
in this process before the workers start; `workers` simulation processes (two
to three per row) each own one environment at a time; work is handed out on
request by a small thread, preferring the task a worker has built; each worker
process is replaced after `life` jobs, the first lives staggered so that the
workers are not replaced at once; a worker writes a finished job's files while
it waits for its next chunk. The server serves the requests that have waited
longest, `batch_size` rows per call, padded with row 0 (`server.Batcher`).

Replay proof: after the manifest's jobs, every tenth "run" job (at least 20,
`contracts.rollout.chain_sample`) is replayed unmodified, after all other work
has been handed out, and must reproduce its chain; `chain_ok` in `run.json`
says whether all did.

Resumable: a job is done when its chain exists (and, for a run job, its steps
file), and a run started again on the same directory skips done jobs.

Memory: the run refuses to start when the machine's GPU memory in use or the
memory available says it would not fit, and stops its processes when they
pass the limits while it runs. It is meant to run under `tools/heavy-run`.

`run_in_process` runs the same jobs through the same loop in this process,
without workers: for tests, and for small checks.
"""

from __future__ import annotations

import functools
import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from fleetlab.contracts.identity import REPLAY_RULES, SoftwareVersion
from fleetlab.contracts.rollout import Job, JobMode, RunLayout, RunRecord, chain_sample
from fleetlab.engine.chain import (
    BROKEN,
    IDENTICAL,
    RENDER_TRANSIENT,
    Chain,
    SampleCheck,
    classify,
)
from fleetlab.engine.episode import (
    ChunkRequest,
    DirectoryStore,
    DirectSource,
    EngineError,
    EpisodeContext,
    call_policy,
    run_job,
)
from fleetlab.engine.factories import build, resolve
from fleetlab.engine.records import LogSink
from fleetlab.engine.server import Batcher

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
GTT_GLOB = "/sys/class/drm/card*/device/mem_info_gtt_used"
MAX_JOB_FAILURES = 3
MAX_STARTUP_LOSSES_PER_SLOT = 3


@dataclass(frozen=True)
class RunConfig:
    """How a run is made. Strings name factories (`engine.factories`), so workers can build them."""

    out: str
    batch_size: int = 4
    workers: int = 10
    precision: str = "fp16_res32"
    device: str = "cuda"
    env_factory: str = "fleetlab.sim.libero_sim:LiberoSimEnv"
    policy_factory: str = "fleetlab.policies.xvla:XVLAPolicy"
    check_factory: str = "fleetlab.sim.libero_success:FleetCheck"  # "placed"
    fault_registry: str = "fleetlab.faults:REGISTRY"
    supervisors: str = "fleetlab.engine.supervisors:make"
    teleoperator: str | None = None
    continuation_steps: int | None = None  # the step budget of a hand-over continuation
    defer_continuations: bool = False  # collect hand-overs before deterministic admission
    # The world's episodes per robot and day fixes the visible start times. Without it the
    # run refuses, unless the manifest may stand in for the world (tests, the engine's checks).
    allow_manifest_day_length: bool = False
    watchdog_s: float = 900.0  # watches every life from startup, including before its first ask
    worker_exit_grace_s: float = 2.0  # let a reporting worker flush its queues and exit
    memory_sample_s: float = 30.0  # how often run.json's memory series is sampled
    video: bool = False  # record video for every run job (a job may ask for it alone)
    life: int = 16
    stagger: bool = True
    chain_sample: bool = True
    # A candidate render transient is settled by `fleetlab.engine.rerender` in a process of its
    # own, through `tools/heavy-run --light` (False: a plain subprocess, for the toy tests).
    rerender_light: bool = True
    miopen: str | None = "pinned"  # the pinned database's name; None without GPU convolutions
    server_threads: int = 4
    # The world's episodes per robot and day, for the visible start times
    # (contracts.identity.started_at_for); None: one more than the largest
    # episode index of the manifest.
    episodes_per_robot_day: int | None = None
    max_workers: int = 12
    gtt_start_limit_gib: float = 17.0
    gtt_abort_gib: float = 21.0
    mem_floor_gib: float = 8.0
    worker_gib: float = 2.1
    server_gib: float = 4.0


@dataclass
class RunResult:
    layout: RunLayout
    record: RunRecord
    summaries: list[dict[str, Any]]


# ---- the worker side ---------------------------------------------------------------------------


class ServerClient:
    """A worker's chunk source: a request to the server, an answer through the pipe.

    `deferred` holds the previous job's file writing, which runs while the
    worker waits for its next chunk.
    """

    def __init__(self, wid: int, req_q: Any, conn: Any, life_no: int = 0) -> None:
        self.wid, self.req_q, self.conn = wid, req_q, conn
        self.life_no = life_no
        self.deferred: list[Any] = []

    def run_deferred(self) -> None:
        while self.deferred:
            self.deferred.pop(0)()

    def chunk(self, request: ChunkRequest) -> Any:
        self.req_q.put(("req", self.wid, request, self.life_no))
        self.run_deferred()
        answer = self.conn.recv()
        if isinstance(answer, str):
            raise EngineError(f"the server could not serve {request.key}: {answer}")
        return answer


def _context(config: RunConfig, layout: RunLayout) -> dict[str, Any]:
    module = importlib.import_module(config.fault_registry.partition(":")[0])
    load_all = getattr(module, "load_all", None)
    if callable(load_all):
        load_all()  # every fault type module registers its kinds (fleetlab.faults)
    registry = resolve(config.fault_registry)
    teleop = functools.partial(_teleoperator, config.teleoperator) if config.teleoperator else None
    name = resolve(config.teleoperator).spec.name if config.teleoperator else None
    return {
        "fault_types": registry.get,
        "make_check": resolve(config.check_factory),
        "teleoperator": teleop,
        "teleoperator_name": name,
        "supervisors": resolve(config.supervisors),
        "store": DirectoryStore(layout.root),
    }


def _teleoperator(path: str) -> Any:
    """A fresh in-worker policy for one episode or continuation."""
    from fleetlab.contracts.policy import check_policy_flags

    policy = build(path)
    check_policy_flags(policy)
    policy.load("cpu", "fp32")
    return policy


def _env_key(job: Job) -> tuple[str, int, int]:
    return (job.identity.suite, job.identity.task_index, job.identity.image_size)


def worker_main(
    wid: int,
    life_no: int,
    config: RunConfig,
    noise: tuple[tuple[int, ...] | None, str | None],
    supervised_sample: frozenset[str],
    req_q: Any,
    job_q: Any,
    conn: Any,
) -> None:
    """One simulation worker: asks for jobs, runs them, writes their files, until told to stop."""
    try:
        from fleetlab.runtime import configure_process

        configure_process()
        import fleetlab  # noqa: F401  (torch before mujoco)

        layout = RunLayout(Path(config.out))
        assert config.episodes_per_robot_day is not None
        sink = LogSink(layout.root, config.episodes_per_robot_day)
        parts = _context(config, layout)
        client = ServerClient(wid, req_q, conn, life_no)
        env: Any = None
        env_key: tuple[str, int, int] | None = None
        born = time.perf_counter()
        times = {"build": 0.0, "run": 0.0, "write": 0.0, "ask": 0.0}
        done = 0
        while True:
            s = time.perf_counter()
            job_q.put(("ask", wid, env_key, life_no))
            message = conn.recv()
            times["ask"] += time.perf_counter() - s
            if message is None:
                break
            message, sample_requested = message
            job = Job.model_validate_json(message)
            key = _env_key(job)
            built = False
            if key != env_key:
                if env is not None:
                    env.close()  # rule 6: one scene per process at a time
                s = time.perf_counter()
                env = build(config.env_factory, *key)
                times["build"] += time.perf_counter() - s
                env_key, built = key, True
            if job.mode is JobMode.continue_:
                client.run_deferred()
            ctx = EpisodeContext(
                source=client,
                noise_shape=noise[0],
                noise_dtype=noise[1],
                continuation_steps=config.continuation_steps,
                record_video=job.record_video,
                task_text=env.task_description,
                **parts,
            )
            s = time.perf_counter()
            record = run_job(job, env, ctx)
            times["run"] += time.perf_counter() - s
            done += 1
            record.stats.update(
                worker=wid,
                worker_life=life_no,
                worker_pid=os.getpid(),
                worker_job=done,
                env_built=built,
                t_end=time.time(),
            )
            for spawned in record.spawned:  # on disk first: a resumed run finds it
                persist_followup(layout, spawned)
                job_q.put(("spawn", wid, spawned.model_dump_json(), life_no))
            if job.mode is JobMode.run and (
                job.identity.episode_id in supervised_sample or sample_requested
            ):
                sample = sample_job_for(record.job, record.decisions)
                persist_followup(layout, sample)
                job_q.put(("sample", wid, sample.model_dump_json(), life_no))

            def finish(record: Any = record, text: str = env.task_description) -> None:
                s = time.perf_counter()
                info = sink.write(record, text)
                times["write"] += time.perf_counter() - s
                info["worker_pss_mib"] = pss_mib()
                req_q.put(("done", wid, info, life_no))

            client.deferred.append(finish)
        client.run_deferred()
        if env is not None:
            env.close()
        req_q.put(
            (
                "bye",
                wid,
                {
                    "worker": wid,
                    "life": life_no,
                    "jobs": done,
                    "alive_s": time.perf_counter() - born,
                    "pss_mib": pss_mib(),
                    **times,
                },
                life_no,
            )
        )
    except BaseException:  # report, then let the process die
        req_q.put(("error", wid, traceback.format_exc(), life_no))
        raise


# ---- the server side ---------------------------------------------------------------------------


def _read_int(path: str) -> int:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except OSError:
        return -1


def gtt_path() -> str | None:
    """The GPU memory counter, found as tools/heavy-run finds it (the first card that has one)."""
    import glob

    found = sorted(glob.glob(GTT_GLOB))
    return found[0] if found else None


def gtt_gib() -> float:
    path = gtt_path()
    return _read_int(path) / 2**30 if path else -1.0


def mem_available_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return -1.0


def pss_mib(pid: int | str = "self") -> float:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def loadavg() -> str:
    return " ".join(Path("/proc/loadavg").read_text().split()[:3])


def sample_plan(jobs: Sequence[Job]) -> tuple[list[Job], frozenset[str]]:
    """The jobs that prove a run: every tenth run job, at least 20, replayed unmodified.

    An unsupervised run job's replay is known now. A supervised one's is a
    "scripted" job from its recorded decisions, made when the run job ends
    (`sample_job_for`): returned here are the episode ids that wait for it.
    """
    runs = [j for j in jobs if j.mode is JobMode.run]
    picked = set(chain_sample([j.identity.episode_id for j in runs]))
    now = [
        Job(identity=j.identity, mode=JobMode.replay)
        for j in runs
        if j.identity.episode_id in picked and j.supervisor is None
    ]
    later = frozenset(
        j.identity.episode_id for j in runs if j.identity.episode_id in picked and j.supervisor
    )
    return now, later


def sample_replays(jobs: Sequence[Job]) -> list[Job]:
    """The sample replays known before the run (the unsupervised run jobs')."""
    return sample_plan(jobs)[0]


def sample_job_for(job: Job, decisions: Sequence[Any]) -> Job:
    """The sample replay of a supervised run job: its recorded decisions, replayed.

    Without a decision the supervisor never fired, and an unmodified replay
    is the same episode.
    """
    if not decisions:
        return Job(identity=job.identity, mode=JobMode.replay)
    return Job(identity=job.identity, mode=JobMode.scripted, decision_trace=list(decisions))


def followup_path(layout: RunLayout, job: Job) -> Path:
    """Where a job made at run time is kept: a continuation next to its snapshot
    (`RunLayout.continuation_file`), a supervised job's sample replay under
    `truth/samples/`."""
    if job.from_snapshot is not None:
        stem = Path(job.from_snapshot.path).name.removesuffix(".pkl")
        key, _, step = stem.rpartition(".")
        return layout.continuation_file(job.from_snapshot.episode_id, key, int(step))
    return layout.truth_dir / "samples" / f"{job.key}.json"


def persist_followup(layout: RunLayout, job: Job) -> None:
    path = followup_path(layout, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(job.model_dump_json())
    temporary.replace(path)


def persisted_followups(layout: RunLayout) -> tuple[list[Job], list[Job]]:
    """The continuations and the sample replays an earlier run of this directory made."""
    continuations = layout.continuations()
    samples = [
        Job.model_validate_json(p.read_text())
        for p in sorted((layout.truth_dir / "samples").glob("*.json"))
    ]
    return continuations, samples


def _unique(jobs: Sequence[Job]) -> list[Job]:
    seen: set[str] = set()
    out = []
    for job in jobs:
        if job.key not in seen:
            seen.add(job.key)
            out.append(job)
    return out


def check_jobs(config: RunConfig, jobs: Sequence[Job], software: SoftwareVersion) -> None:
    keys = [j.key for j in jobs]
    if len(set(keys)) != len(keys):
        raise EngineError("job keys in a manifest must be unique")
    for job in jobs:
        ident = job.identity
        if ident.batch_size != config.batch_size:
            raise EngineError(
                f"{job.key}: batch size {ident.batch_size}, run has {config.batch_size}"
            )
        if ident.policy.precision != config.precision:
            raise EngineError(
                f"{job.key}: precision {ident.policy.precision}, run has {config.precision}"
            )
        if ident.software != software:
            raise EngineError(f"{job.key}: made for other software than this run's ({software})")
    if config.device != "cpu" and config.miopen is not None:
        check_images(config, jobs)


def check_images(config: RunConfig, jobs: Sequence[Job]) -> None:
    """Replay rule 7 per call: the images each job's calls carry must be pinned.

    The effective image views come from the policy input hooks (a
    configuration fault may change them), so they are computed here per job.
    """
    from fleetlab.engine import miopen
    from fleetlab.engine.episode import effective_spec

    module = importlib.import_module(config.fault_registry.partition(":")[0])
    load_all = getattr(module, "load_all", None)
    if callable(load_all):
        load_all()
    registry = resolve(config.fault_registry)
    needed: dict[int, str] = {}
    for job in jobs:
        pairs = [
            (registry.get(f.kind), registry.get(f.kind).new_episode(f, job.identity))
            for f in job.effective_faults
        ]
        views = effective_spec(job.identity, pairs).image_views
        needed.setdefault(miopen.images_per_call(config.batch_size, views), job.key)
    for n_images, key in sorted(needed.items()):
        try:
            miopen.require_images(n_images, config.precision, config.miopen or "")
        except miopen.UnpinnedError as error:
            raise EngineError(f"{key}: {error}") from None


def prepare(config: RunConfig) -> SoftwareVersion:
    """Pin the convolution database if the run uses the GPU, and return the software version."""
    from fleetlab.engine import software

    if config.device == "cpu" or config.miopen is None:
        return software.current("none")
    from fleetlab.engine import miopen

    if gtt_path() is None:
        raise EngineError("no GPU memory counter (mem_info_gtt_used): the memory guard cannot work")
    miopen.activate(config.miopen)
    return software.current(miopen.require(config.batch_size, config.precision, config.miopen))


def kept_rules(config: RunConfig) -> list[str]:
    names = [rule.name for rule in REPLAY_RULES]
    if config.device == "cpu" or config.miopen is None:
        names.remove("pinned_conv_database")
    return names


def load_policy(config: RunConfig) -> Any:
    from fleetlab.contracts.policy import check_policy_flags

    policy = build(config.policy_factory)
    check_policy_flags(policy)
    policy.load(config.device, config.precision)
    return policy


def with_video(config: RunConfig, jobs: Sequence[Job]) -> list[Job]:
    """`config.video` records every run job's video: it sets the jobs' own flag."""
    if not config.video:
        return list(jobs)
    return [
        j.model_copy(update={"record_video": True}) if j.mode is JobMode.run else j for j in jobs
    ]


def with_day_length(config: RunConfig, jobs: Sequence[Job]) -> RunConfig:
    """The world's episodes per robot and day, or, if allowed, one more than the manifest's
    largest episode index (the visible start times depend on it)."""
    if config.episodes_per_robot_day is not None or not jobs:
        return config
    if not config.allow_manifest_day_length:
        raise EngineError(
            "give the world's episodes per robot and day (RunConfig.episodes_per_robot_day, "
            "--episodes-per-robot-day), or allow the manifest's (--allow-manifest-day-length)"
        )
    longest = 1 + max(j.identity.episode_index for j in jobs)
    return replace(config, episodes_per_robot_day=longest)


def open_writer(config: RunConfig, software: SoftwareVersion) -> Any:
    """The log's run writer, opened before any worker starts (it clears unfinished jobs)."""
    from fleetlab.log import RunWriter

    record = RunRecord(
        software=software, batch_size=config.batch_size, replay_rules=kept_rules(config)
    )
    assert config.episodes_per_robot_day is not None
    return RunWriter(Path(config.out), record, episodes_per_robot_day=config.episodes_per_robot_day)


def run(config: RunConfig, jobs: Sequence[Job]) -> RunResult:
    """Run the jobs in the worker layout and write the run directory."""
    software = prepare(config)
    jobs = with_video(config, jobs)
    config = with_day_length(config, jobs)
    check_jobs(config, jobs, software)
    writer = open_writer(config, software)
    try:
        return _run(config, jobs, software, writer)
    finally:
        writer.close()
        release_memory(config)


def release_memory(config: RunConfig) -> None:
    """Free the run's policy now, not when the garbage collector next comes round.

    The first nightly (2026-10-08) ran four runs in one process; the server's
    PSS was 2.3 GiB in the first two and 4.2 and 4.6 GiB in the next two, flat
    within each run: an earlier run's model still held when the next loaded
    its own. Collecting the cycles and emptying torch's cache after a run
    keeps one model in memory at a time. It changes no computation.
    """
    import gc

    gc.collect()
    if config.device != "cpu" and "torch" in sys.modules:
        import torch

        torch.cuda.empty_cache()


def _run(
    config: RunConfig, jobs: Sequence[Job], software: SoftwareVersion, writer: Any
) -> RunResult:
    t_start = time.perf_counter()
    layout = RunLayout(Path(config.out))
    if config.chain_sample:
        replays, supervised_sample = sample_plan(jobs)
    else:
        replays, supervised_sample = [], frozenset()
    earlier_continuations, earlier_samples = persisted_followups(layout)
    replays = _unique(replays + earlier_samples)
    todo = writer.pending(
        _unique(list(jobs) + ([] if config.defer_continuations else earlier_continuations))
    )
    todo_replays = writer.pending(replays)
    n_workers = min(config.workers, config.max_workers, max(1, len(todo) + len(todo_replays)))
    gtt0, avail0, load0 = gtt_gib(), mem_available_gib(), loadavg()
    if config.device != "cpu":
        if gtt0 > config.gtt_start_limit_gib:
            raise EngineError(f"GPU memory in use is {gtt0:.1f} GiB; not starting")
        need = n_workers * config.worker_gib + config.server_gib
        if avail0 - need < config.mem_floor_gib:
            raise EngineError(f"{avail0:.1f} GiB available, about {need:.0f} needed; not starting")

    import torch

    torch.set_num_threads(config.server_threads)
    t0 = time.perf_counter()
    policy = load_policy(config)
    load_s = time.perf_counter() - t0
    noise = (policy.noise_shape, policy.noise_dtype) if policy.needs_noise else (None, None)

    ctx = mp.get_context("spawn")
    req_q: Any = ctx.Queue()
    job_q: Any = ctx.Queue()
    conns: dict[int, Any] = {}
    procs: dict[int, Any] = {}
    lives: dict[int, int] = dict.fromkeys(range(n_workers), 0)
    given: dict[int, int] = dict.fromkeys(range(n_workers), 0)
    delivered: dict[int, int] = dict.fromkeys(range(n_workers), 0)
    startup_losses: dict[int, int] = dict.fromkeys(range(n_workers), 0)
    normal_exit_seen: dict[int, float] = {}
    on_task: dict[int, tuple[str, int, int] | None] = dict.fromkeys(range(n_workers))
    life_of = {
        w: config.life if not config.stagger else max(1, round(config.life * (w + 1) / n_workers))
        for w in range(n_workers)
    }
    lock = threading.Lock()
    pending: list[list[Job]] = [list(todo), list(todo_replays)]  # phase 0, then the sample
    spawned_jobs: list[Job] = list(earlier_continuations)
    sample_jobs: list[Job] = list(replays)
    sample_keys = {j.key for j in sample_jobs}
    known = {j.key for j in list(jobs) + spawned_jobs + sample_jobs}
    send_locks = {w: threading.RLock() for w in range(n_workers)}
    # The watchdog's books: the jobs handed to each worker and not yet reported done, when a
    # worker was last heard from, how often each job was lost with its worker.
    assigned: dict[int, list[Job]] = {w: [] for w in range(n_workers)}
    last_seen: dict[int, float] = dict.fromkeys(range(n_workers), time.perf_counter())
    failures: dict[str, int] = {}
    prior_extra = cast(dict[str, Any], writer.record.extra)
    prior_session = int(
        prior_extra.get("session_count")
        or (1 if "incidents" in prior_extra or "total_s" in prior_extra else 0)
    )
    session = prior_session + 1
    prior_incidents = [
        {
            **i,
            "session": i.get("session", 1),
            "replay_checks": {
                key: {"checked": False, "verdict": None, **check}
                for key, check in (i.get("replay_checks") or {}).items()
            },
        }
        for i in prior_extra.get("incidents") or []
        if isinstance(i, dict)
    ]
    incidents: list[dict[str, Any]] = []
    incident_by_life: dict[tuple[int, int], dict[str, Any]] = {}
    progress: dict[tuple[int, int], dict[str, str]] = {}
    handed_out: set[str] = set()
    completed: set[str] = set()
    alive: set[int] = set()
    stop = threading.Event()
    thread_events: queue.Queue[tuple[str, Any, str]] = queue.Queue()
    recovering: set[int] = set()

    def thread_main(name: str, target: Any) -> None:
        try:
            target()
        except BaseException:
            if not stop.is_set():
                thread_events.put(("error", name, traceback.format_exc()))

    def unpack(message: Any) -> tuple[str, int, Any, int]:
        kind, wid, payload, *generation = message
        # Legacy custom workers/probes use triples. The engine's own workers
        # always tag messages, so late requests/done/error from a lost life
        # cannot be applied to its replacement.
        return kind, wid, payload, generation[0] if generation else lives[wid] - 1

    def send(wid: int, payload: Any) -> None:
        with send_locks[wid]:
            conns[wid].send(payload)

    def pick(wid: int, built: tuple[str, int, int] | None) -> Job | None:
        with lock:
            phase = next((p for p in pending if p), None)
            if phase is None or given[wid] >= life_of[wid]:
                on_task[wid] = None
                return None
            same = next(
                (j for j in phase if built is not None and _env_key(j) == tuple(built)), None
            )
            if same is None:
                left: dict[tuple[str, int, int], int] = {}
                for j in phase:
                    left[_env_key(j)] = left.get(_env_key(j), 0) + 1
                crowd: dict[tuple[str, int, int], int] = {}
                for w, k in on_task.items():
                    if k is not None and w != wid:
                        crowd[k] = crowd.get(k, 0) + 1
                order = list(left)
                best = max(order, key=lambda k: (left[k] / (1 + crowd.get(k, 0)), -order.index(k)))
                same = next(j for j in phase if _env_key(j) == best)
            phase.remove(same)
            given[wid] += 1
            on_task[wid] = _env_key(same)
            assigned[wid].append(same)
            handed_out.add(same.key)
            progress.setdefault((wid, lives[wid] - 1), {})[same.key] = "not_delivered"
            last_seen[wid] = time.perf_counter()
            return same

    def dispatcher() -> None:
        while not stop.is_set():
            try:
                message = job_q.get(timeout=0.1)
            except queue.Empty:
                continue
            kind, wid, payload, generation = unpack(message)
            if kind in ("spawn", "sample"):
                job = Job.model_validate_json(payload)
                with lock:
                    if job.key in known:
                        continue
                    known.add(job.key)
                    if kind == "spawn":
                        if not config.defer_continuations:
                            pending[0].append(job)
                        spawned_jobs.append(job)
                    else:
                        pending[1].append(job)
                        sample_jobs.append(job)
                        sample_keys.add(job.key)
                continue
            with send_locks[wid]:
                if (
                    wid not in alive
                    or generation != lives[wid] - 1
                    or wid in recovering
                    or not procs[wid].is_alive()
                ):
                    continue
                last_seen[wid] = time.perf_counter()
                process = procs[wid]
                given_job = pick(wid, payload)
                try:
                    send(
                        wid,
                        None
                        if given_job is None
                        else (
                            given_job.model_dump_json(),
                            config.chain_sample and given_job.key in failures,
                        ),
                    )
                    if given_job is not None:
                        delivered[wid] += 1
                        startup_losses[wid] = 0
                        progress[(wid, generation)][given_job.key] = "execution_unconfirmed"
                except OSError:
                    recovering.add(wid)
                    thread_events.put(("lost", (wid, process), traceback.format_exc()))

    def spawn_locked(wid: int) -> bool:
        with lock:
            if not any(pending):
                return False
            given[wid] = 0
            delivered[wid] = 0
            if lives[wid] > 0:
                life_of[wid] = config.life
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        for v in THREAD_VARS:
            os.environ[v] = "1"
        parent, child = ctx.Pipe()
        if wid in conns:
            conns[wid].close()
        conns[wid] = parent
        process = ctx.Process(
            target=worker_main,
            args=(wid, lives[wid], config, noise, supervised_sample, req_q, job_q, child),
        )
        process.start()
        child.close()  # only the worker owns the receiving end; a dead worker must break sends
        for v, value in saved.items():
            if value is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = value
        lives[wid] += 1
        procs[wid] = process
        alive.add(wid)
        normal_exit_seen.pop(wid, None)
        last_seen[wid] = time.perf_counter()
        recovering.discard(wid)
        return True

    def spawn(wid: int) -> bool:
        with send_locks[wid]:
            return spawn_locked(wid)

    for w in range(n_workers):
        spawn(w)
    dispatch_thread = threading.Thread(
        target=thread_main, args=("dispatcher", dispatcher), daemon=True
    )
    dispatch_thread.start()

    peaks = {
        "gtt_gib": gtt0,
        "mem_available_min_gib": avail0,
        "pss_total_mib": 0.0,
        "pss_worker_max_mib": 0.0,
        "pss_server_mib": 0.0,
    }
    abort: dict[str, str | None] = {"why": None}

    series: list[list[float]] = []  # the soak test's record: memory over the run

    def monitor() -> None:
        last_sample = 0.0
        while not stop.wait(1.0):
            gtt, avail = gtt_gib(), mem_available_gib()
            server = pss_mib()
            workers = [pss_mib(p.pid) for p in list(procs.values()) if p.is_alive()]
            now = time.perf_counter() - t_start
            if now - last_sample >= config.memory_sample_s:
                last_sample = now
                series.append(
                    [round(now, 1), round(gtt, 2), round(avail, 2), round(server / 1024, 2),
                     round(sum(workers) / 1024, 2), len(workers),
                     round(max(workers, default=0.0) / 1024, 3)]
                )  # fmt: skip
            peaks["gtt_gib"] = max(peaks["gtt_gib"], gtt)
            peaks["mem_available_min_gib"] = min(peaks["mem_available_min_gib"], avail)
            peaks["pss_server_mib"] = max(peaks["pss_server_mib"], server)
            peaks["pss_worker_max_mib"] = max([peaks["pss_worker_max_mib"], *workers])
            peaks["pss_total_mib"] = max(peaks["pss_total_mib"], server + sum(workers))
            if config.device != "cpu" and (
                gtt > config.gtt_abort_gib or avail < config.mem_floor_gib - 2
            ):
                abort["why"] = f"GPU memory {gtt:.1f} GiB, memory available {avail:.1f} GiB"

    monitor_thread = threading.Thread(target=thread_main, args=("monitor", monitor), daemon=True)
    monitor_thread.start()

    batcher: Batcher[int] = Batcher(config.batch_size)
    summaries: list[dict[str, Any]] = []
    byes: list[dict[str, Any]] = []
    calls: list[tuple[float, int, float]] = []  # (since start, rows, seconds)
    serve0 = time.perf_counter()
    infer_s = 0.0

    def kill_all() -> None:
        for p in procs.values():
            if p.is_alive():
                p.kill()
        for p in procs.values():
            p.join(10)

    def lose_locked(wid: int, why: str, worker_error: str | None = None) -> None:
        """Let a reporting worker exit; kill only if it stays, then retry its lost assignments."""
        process = procs[wid]
        generation = lives[wid] - 1
        # Receiving the report does not mean its feeder has released the shared
        # Queue writer lock. Normal process exit flushes and joins those feeders.
        process.join(config.worker_exit_grace_s)
        killed = False
        if process.is_alive():
            process.kill()
            killed = True
        process.join(10)
        if process.exitcode == 0 and worker_error is None:
            # It may have exited normally since the watchdog or grace sampled
            # its status. Use the final joined status before recording a loss.
            # Leave its queued goodbye (or error report) to the main loop.
            normal_exit_seen.setdefault(wid, time.perf_counter())
            recovering.discard(wid)
            return
        batcher.waiting.pop(wid, None)
        with lock:
            lost, assigned[wid] = assigned[wid], []
            for job in lost:
                completed.discard(job.key)
                failures[job.key] = failures.get(job.key, 0) + 1
        if delivered[wid] == 0:
            startup_losses[wid] += 1
        incident = {
            "worker": wid,
            "worker_life": generation,
            "session": session,
            "why": why[-2000:],
            "exitcode": process.exitcode,
            "parent_killed": killed,
            "received_job": delivered[wid] > 0,
            "worker_error": worker_error[-2000:] if worker_error else None,
            "jobs": [j.key for j in lost],
            "attempts": {j.key: failures[j.key] for j in lost},
            "attempts_kind": "lost_assignments",
            "assignment_status": {
                j.key: progress.get((wid, generation), {}).get(j.key, "execution_unconfirmed")
                for j in lost
            },
            "recovered": False,
            "ended_run": False,
            "replay_checks": {j.key: {"checked": False, "verdict": None} for j in lost},
        }
        incidents.append(incident)
        incident_by_life[(wid, generation)] = incident
        print(
            f"worker {wid} lost ({why.splitlines()[-1] if why else ''}); "
            f"running {len(lost)} jobs again",
            flush=True,
        )
        for job in lost:
            if failures[job.key] >= MAX_JOB_FAILURES:
                incident["ended_run"] = True
                raise EngineError(f"job {job.key} failed {MAX_JOB_FAILURES} times; last:\n{why}")
            with lock:
                pending[1 if job.key in sample_keys else 0].insert(0, job)
                if config.chain_sample and job.mode is JobMode.run and job.supervisor is None:
                    sample = Job(identity=job.identity, mode=JobMode.replay)
                    if sample.key not in known:
                        known.add(sample.key)
                        sample_keys.add(sample.key)
                        sample_jobs.append(sample)
                        pending[1].append(sample)
                        persist_followup(layout, sample)
        with lock:
            work_left = any(pending)
        if work_left and startup_losses[wid] >= MAX_STARTUP_LOSSES_PER_SLOT:
            incident["ended_run"] = True
            raise EngineError(
                f"worker {wid} slot lost before receiving any job "
                f"{startup_losses[wid]} times; cannot make progress; last:\n{why}"
            )
        if not spawn(wid):
            alive.discard(wid)
        recovering.discard(wid)

    def lose(wid: int, why: str, expected: Any = None, worker_error: str | None = None) -> None:
        # Only this main thread replaces workers. Serialize replacement with
        # job assignment/sends, and discard a queued loss already handled by
        # an exit-code check or another failed send from the same worker life.
        with send_locks[wid]:
            if wid not in alive or (expected is not None and procs[wid] is not expected):
                return
            lose_locked(wid, why, worker_error)

    def late_report(kind: str, wid: int, payload: Any, generation: int) -> bool:
        incident = incident_by_life.get((wid, generation))
        if incident is None:
            return False
        if kind == "error":
            incident["worker_error"] = str(payload)[-2000:]
        elif kind == "done" and payload["key"] in incident["assignment_status"]:
            incident["assignment_status"][payload["key"]] = "completed_report_late"
        return True

    def thread_updates() -> None:
        while True:
            try:
                kind, target, why = thread_events.get_nowait()
            except queue.Empty:
                return
            if kind == "error":
                raise EngineError(f"{target} failed:\n{why}")
            wid, process = target
            lose(wid, why, expected=process)

    def reply(wid: int, payload: Any) -> None:
        with send_locks[wid]:
            process = procs[wid]
            try:
                send(wid, payload)
            except OSError:
                lose(wid, traceback.format_exc(), expected=process)

    def handle(message: Any) -> None:
        kind, wid, payload, generation = unpack(message)
        if late_report(kind, wid, payload, generation):
            return
        if generation != lives[wid] - 1:
            return
        last_seen[wid] = time.perf_counter()
        if kind == "req":
            progress.setdefault((wid, generation), {})[payload.key] = "execution_started"
            batcher.add(wid, payload)
        elif kind == "done":
            with lock:
                assigned[wid] = [j for j in assigned[wid] if j.key != payload["key"]]
                completed.add(payload["key"])
            payload["t_done"] = time.perf_counter() - t_start  # end to end, as xvla-speed counts
            summaries.append(payload)
            n = len(summaries)
            if n % 20 == 0:
                ok = sum(s["success"] for s in summaries)
                print(
                    f"{n} jobs done, {ok} successful, {time.perf_counter() - t_start:.0f} s",
                    flush=True,
                )
        elif kind == "bye":
            byes.append(payload)
            procs[wid].join(30)
            if not spawn(wid):
                alive.discard(wid)
        elif kind == "error":
            lose(wid, payload, worker_error=payload)

    try:
        while alive:
            thread_updates()
            # Startup/import/first ask is watched too. Check on every cycle,
            # even if other workers keep the request queue busy.
            now = time.perf_counter()
            for w in sorted(alive):
                if procs[w].exitcode == 0:
                    normal_exit_seen.setdefault(w, now)
                elif now - last_seen[w] > config.watchdog_s:
                    lose(w, f"silent for {now - last_seen[w]:.0f} s")
            if abort["why"]:
                raise EngineError(f"stopped to protect the machine: {abort['why']}")
            try:
                message = req_q.get_nowait() if batcher.waiting else req_q.get(timeout=2)
            except queue.Empty:
                thread_updates()
                now = time.perf_counter()
                for w in sorted(alive):
                    process = procs[w]
                    # A normal exit flushes its goodbye, but it may have
                    # arrived since get() found the shared queue empty.
                    if process.exitcode not in (None, 0):
                        lose(w, f"exit code {process.exitcode}")
                    elif process.exitcode == 0:
                        ended = normal_exit_seen.setdefault(w, now)
                        if now - ended > config.watchdog_s:
                            raise EngineError(
                                f"worker {w} exited normally but no goodbye was received "
                                f"within {config.watchdog_s:g} s"
                            ) from None
                with lock:
                    work_left = any(pending)
                for w in range(n_workers):  # a job queued after its slot's worker ended
                    if work_left and w not in alive and spawn(w):
                        alive.add(w)
                if not batcher.waiting:
                    continue
                message = None
            if message is not None:
                thread_updates()
                handle(message)
            while True:
                try:
                    handle(req_q.get_nowait())
                except queue.Empty:
                    break
            chosen = batcher.next_call()
            if not chosen:
                continue
            s = time.perf_counter()
            try:
                results = call_policy(policy, [r for _, r in chosen])
            except Exception as error:
                for w, _ in chosen:
                    reply(w, f"{type(error).__name__}: {error}")
                raise
            e = time.perf_counter()
            infer_s += e - s
            calls.append((s - serve0, len(chosen), e - s))
            for (w, _), actions in zip(chosen, results, strict=True):
                reply(w, actions)
        thread_updates()
        with lock:
            queued = {j.key for phase in pending for j in phase}
            missing = handed_out - completed - queued
        if missing:
            raise EngineError(
                f"assigned jobs neither completed nor pending: {', '.join(sorted(missing))}"
            )
        if queued:
            raise EngineError(f"no live workers for pending jobs: {', '.join(sorted(queued))}")
    finally:
        original = sys.exception()
        stop.set()
        kill_all()
        # Never read a shared queue after killing workers. Even get_nowait()
        # can block forever inside a frame whose feeder died partway through.
        # Only reports already handled by the live main loop are retained.
        # A killed worker can leave a Queue's writer lock held. Sending a sentinel
        # through that queue starts a parent feeder that can never finish, and its
        # implicit join then hangs interpreter exit. The dispatcher polls `stop`;
        # abandon queue buffers only now, after all workers have stopped.
        dispatch_thread.join(2)
        monitor_thread.join(2)
        for conn in conns.values():
            conn.close()
        for q in (job_q, req_q):
            q.cancel_join_thread()
            q.close()
        for incident in incidents:
            incident["recovered"] = (
                not incident["ended_run"]
                and all(key in completed for key in incident["jobs"])
                and (bool(incident["jobs"]) or original is None)
            )
        try:
            writer.update_record(
                extra={"incidents": prior_incidents + incidents, "session_count": session}
            )  # also keep aborted runs' losses
        except Exception as error:
            if original is None:
                raise
            original.add_note(f"could not record incidents during shutdown: {error}")
    serve_s = time.perf_counter() - serve0

    # ---- outputs ---------------------------------------------------------------------------
    status = writer.finalize()
    sample_ids, chain_ok, comparisons = prove(layout, jobs, sample_jobs, config)
    checked = {c["episode_id"]: c for c in comparisons}
    incident_jobs = {j.key: j for j in list(jobs) + spawned_jobs + sample_jobs}
    for incident in incidents:
        for key in incident["jobs"]:
            job = incident_jobs[key]
            check = checked.get(job.identity.episode_id)
            supported = job.mode is JobMode.run or job.key in sample_keys
            incident["replay_checks"][key] = {
                "checked": supported and check is not None,
                "verdict": check["verdict"] if supported and check else None,
            }
    main_keys = {j.key for j in jobs}
    # A resumed run keeps the earlier runs' per-job records (the truth tables hold every job).
    earlier = writer.record.extra.get("jobs_detail") or []
    rerun = {s["key"] for s in summaries}
    detail = [d for d in earlier if isinstance(d, dict) and d.get("key") not in rerun] + summaries
    ends = sorted(s["t_done"] for s in summaries if s["key"] in main_keys)
    all_ends = sorted(s["t_done"] for s in summaries)
    rows = np.array([c[1] for c in calls]) if calls else np.zeros(0)
    record = writer.update_record(
        chain_sample=sample_ids,
        chain_ok=chain_ok,
        extra=_json_safe(
            {
                "config": asdict(config),
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code": code_version(),
                "jobs": len(jobs),
                "jobs_run_now": len(todo),
                "spawned": len(spawned_jobs),
                "workers": n_workers,
                "policy_load_s": load_s,
                "total_s": time.perf_counter() - t_start,
                "serve_s": serve_s,
                "throughput": throughput(ends),  # the manifest's jobs, from the run's start
                "throughput_with_sample": throughput(all_ends),  # and the chain sample
                "calls": len(calls),
                "share_of_rows_in_use": float(rows.sum() / (len(rows) * config.batch_size))
                if len(rows)
                else None,
                "calls_by_rows": {
                    int(k): int((rows == k).sum()) for k in range(1, config.batch_size + 1)
                },
                "inference_s_total": infer_s,
                "inference_s_per_call_median": float(np.median([c[2] for c in calls]))
                if calls
                else None,
                "server_busy_share": infer_s / serve_s if serve_s else None,
                "successes": sum(s["success"] for s in summaries if s["key"] in main_keys),
                "jobs_done": len(status.done),
                "finalized": status.finalized,
                "comparisons": comparisons,
                "chain_checks": chain_summary(comparisons, chain_ok),
                "peaks": peaks,
                "gtt_start_gib": gtt0,
                "mem_available_start_gib": avail0,
                "load_start": load0,
                "load_end": loadavg(),
                "worker_lives": byes,
                "incidents": prior_incidents + incidents,
                "session_count": session,
                # seconds, GPU memory machine-wide, memory available, server PSS, workers' PSS
                # (GiB), live workers, the largest worker (GiB); every memory_sample_s
                "memory_series": series,
                "jobs_detail": detail,
            }
        ),
    )
    return RunResult(layout=layout, record=record, summaries=summaries)


def code_version() -> dict[str, Any]:
    """The checkout's commit and whether it had changes: which code made the run."""
    import subprocess

    from fleetlab.runtime import repo_root

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=False
        )
        return done.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def throughput(ends: Sequence[float]) -> dict[str, float | None]:
    """Jobs per hour end to end (from the run's start: policy load, workers starting, the
    tail) and in steady state (between the 10% and the 90% mark of completions)."""
    n = len(ends)
    if n < 2:
        return {"per_hour_overall": None, "per_hour_steady": None}
    lo, hi = int(0.1 * n), int(0.9 * n) - 1
    steady = 3600 * (hi - lo) / (ends[hi] - ends[lo]) if hi > lo and ends[hi] > ends[lo] else None
    return {
        "per_hour_overall": 3600 * n / ends[-1] if ends[-1] > 0 else None,
        "per_hour_steady": steady,
    }


def prove(
    layout: RunLayout,
    jobs: Sequence[Job],
    replays: Sequence[Job],
    config: RunConfig | None = None,
) -> tuple[list[str], bool | None, list[dict[str, Any]]]:
    """Classify each sample replay against its original; `chain_ok` over all of them.

    Each comparison is identical, a render transient or broken
    (`chain.classify`; docs/design/contracts.md section 4). A candidate
    transient is settled by rendering the episode again without the model
    (`settle_transient`). The run's chain holds when no comparison is broken
    and at most one is a transient: a second transient breaks it.
    """
    originals = {j.identity.episode_id: j for j in jobs if j.mode is JobMode.run}
    # Resumed continuation phases can inherit samples from an earlier run phase.
    replays = [r for r in replays if r.identity.episode_id in originals]
    if not replays:
        return [], None, []
    config = config or RunConfig(out=str(layout.root))
    out = []
    checks: list[SampleCheck] = []
    for replay in replays:
        original = originals[replay.identity.episode_id]
        try:
            chain = Chain.load(layout.chain(replay))
            reference = Chain.load(layout.chain(original))
        except FileNotFoundError as error:
            check = SampleCheck(BROKEN, f"missing: {error.filename}")
        else:
            check = classify(chain, reference, consumed_frames(layout, original, config))
            if check.verdict == RENDER_TRANSIENT:
                check = settle_transient(layout, original, replay, check, config)
        checks.append(check)
        out.append(
            {
                "episode_id": replay.identity.episode_id,
                "key": replay.key,
                "identical": check.verdict == IDENTICAL,
                "what": check.what,
                **check.as_json(),
            }
        )
    transients = sum(c.verdict == RENDER_TRANSIENT for c in checks)
    ok = all(c.verdict != BROKEN for c in checks) and transients <= 1
    return [r.identity.episode_id for r in replays], ok, out


def chain_summary(comparisons: Sequence[dict[str, Any]], chain_ok: bool | None) -> dict[str, Any]:
    """`run.json`'s `extra.chain_checks`: the verdicts counted, the transients, the frames."""
    verdicts = (IDENTICAL, RENDER_TRANSIENT, BROKEN)
    counts = {v: sum(c.get("verdict") == v for c in comparisons) for v in verdicts}
    transients = [
        {
            "episode_id": c["episode_id"],
            "key": c.get("key"),
            "step": c["frames"][0][0],
            "camera": c["frames"][0][1],
            "odd": c.get("odd"),
        }
        for c in comparisons
        if c.get("verdict") == RENDER_TRANSIENT
    ]
    reason = ""
    if chain_ok is False and not counts[BROKEN] and len(transients) > 1:
        reason = f"{len(transients)} render transients in one run (at most one is allowed)"
    return {
        **counts,
        "compared_frames": sum(int(c.get("compared_frames") or 0) for c in comparisons),
        "transients": transients,
        "broken_by": reason,
    }


def consumed_frames(layout: RunLayout, original: Job, config: RunConfig) -> Any:
    """Which frames of an episode something used: `(step, camera) -> bool`.

    A policy call uses both cameras' images of its step (`chunk_step` 0 in the
    steps table; an in-worker policy is called every step), a video frame
    both images of its step (`frame_index` set), and a supervisor that may
    read pixels every frame. A run without its steps table counts every frame
    as used.
    """
    from fleetlab.log.runfiles import read_table

    if original.supervisor is not None:
        module = importlib.import_module(config.supervisors.partition(":")[0])
        reads: Any = getattr(module, "reads_pixels", lambda _name: True)
        if reads(original.supervisor.name):
            return lambda step, camera: True
    try:
        rows = read_table(layout.root, "steps", original.identity.episode_id).to_pylist()
    except (FileNotFoundError, OSError):
        return lambda step, camera: True
    used = {
        int(r["step"]) for r in rows if r.get("chunk_step") == 0 or r.get("frame_index") is not None
    }
    return lambda step, camera: step in used


def settle_transient(
    layout: RunLayout, original: Job, replay: Job, check: SampleCheck, config: RunConfig
) -> SampleCheck:
    """Render the episode again without the model and say which rendering was the odd one.

    Runs `fleetlab.engine.rerender` in a process of its own (a light job:
    one simulator process, no model); a failure to name the odd one breaks
    the comparison.
    """
    import subprocess
    import tempfile

    from fleetlab.engine.rerender import odd_rendering
    from fleetlab.runtime import repo_root

    with tempfile.TemporaryDirectory(prefix="rerender-") as tmp:
        out = Path(tmp) / "rerender.json"
        command = [
            sys.executable, "-m", "fleetlab.engine.rerender", str(layout.root), original.key,
            "--against", replay.key, "--passes", "2", "--json-out", str(out),
            "--env-factory", config.env_factory, "--check-factory", config.check_factory,
            "--fault-registry", config.fault_registry,
        ]  # fmt: skip
        heavy_run = repo_root() / "tools" / "heavy-run"
        if config.rerender_light and heavy_run.is_file():
            command = [str(heavy_run), "--light", "-m", "6G", *command]
        # The interpreter's own path, as a spawned worker gets it (factories outside the package).
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
        except (OSError, subprocess.TimeoutExpired) as error:
            return check.settled(None, f"the re-rendering did not run: {error}")
        if done.returncode != 0 or not out.is_file():
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            return check.settled(None, f"the re-rendering failed: {tail[0][:200]}")
        result = json.loads(out.read_text())
    odd, why = odd_rendering(result, original.key, replay.key, check.frames[0])
    return check.settled(odd, why)


def _json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(value, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
    )


def run_in_process(config: RunConfig, jobs: Sequence[Job], policy: Any | None = None) -> RunResult:
    """The same jobs through the same loop in this process, one after the other, no workers."""
    software = prepare(config)
    jobs = with_video(config, jobs)
    config = with_day_length(config, jobs)
    check_jobs(config, jobs, software)
    writer = open_writer(config, software)
    try:
        return _run_in_process(config, jobs, policy, writer)
    finally:
        writer.close()


def _run_in_process(
    config: RunConfig, jobs: Sequence[Job], policy: Any | None, writer: Any
) -> RunResult:
    layout = RunLayout(Path(config.out))
    if config.chain_sample:
        replays, supervised_sample = sample_plan(jobs)
    else:
        replays, supervised_sample = [], frozenset()
    earlier_continuations, earlier_samples = persisted_followups(layout)
    replays = _unique(replays + earlier_samples)
    if policy is None:
        policy = load_policy(config)
    source = DirectSource(policy)
    parts = _context(config, layout)
    assert config.episodes_per_robot_day is not None
    sink = LogSink(layout.root, config.episodes_per_robot_day)
    queue_: list[Job] = writer.pending(
        _unique(list(jobs) + ([] if config.defer_continuations else earlier_continuations))
        + replays
    )
    spawned: list[Job] = list(earlier_continuations)
    summaries = []
    env: Any = None
    env_key: tuple[str, int, int] | None = None
    while queue_:
        job = queue_.pop(0)
        if _env_key(job) != env_key:
            if env is not None:
                env.close()
            env_key = _env_key(job)
            env = build(config.env_factory, *env_key)
        ctx = EpisodeContext(
            source=source,
            noise_shape=policy.noise_shape if policy.needs_noise else None,
            noise_dtype=policy.noise_dtype if policy.needs_noise else None,
            continuation_steps=config.continuation_steps,
            record_video=job.record_video,
            task_text=env.task_description,
            **parts,
        )
        record = run_job(job, env, ctx)
        for extra in record.spawned:
            persist_followup(layout, extra)
            spawned.append(extra)
            if not config.defer_continuations:
                queue_.insert(0, extra)
        if job.mode is JobMode.run and job.identity.episode_id in supervised_sample:
            sample = sample_job_for(job, record.decisions)
            persist_followup(layout, sample)
            replays.append(sample)
            queue_.append(sample)
        summaries.append(sink.write(record, env.task_description))
    if env is not None:
        env.close()
    status = writer.finalize()
    sample_ids, chain_ok, comparisons = prove(layout, jobs, replays, config)
    record_ = writer.update_record(
        chain_sample=sample_ids,
        chain_ok=chain_ok,
        extra=_json_safe(
            {
                "config": asdict(config),
                "in_process": True,
                "jobs_done": len(status.done),
                "finalized": status.finalized,
                "comparisons": comparisons,
                "chain_checks": chain_summary(comparisons, chain_ok),
                "jobs_detail": summaries,
            }
        ),
    )
    return RunResult(layout=layout, record=record_, summaries=summaries)
