"""One job, from its reset (or its snapshot) to its end: the rollout loop with its hooks.

The loop knows the hook points of `contracts.faults.HOOK_ORDER` and nothing
about any fault type. Per episode:

1. every active fault gets a fresh state (`new_episode`);
2. the policy input hooks give the effective `PolicySpec`; the environment is
   reset with it, so a changed control mode reaches the arm controller;
3. the environment runs its reset sequence and applies the cell hooks;
4. a job that continues from a snapshot puts the snapshot back.

Per step t, in this order (the log listens where `HOOK_ORDER` says):

- [decision] a supervisor's decision whose hold has run out is applied:
  `stop` ends the attempt, `hand_over` saves a snapshot and queues a
  "continue" job for the teleoperator, `retry` runs the recovery primitive;
- [snapshot] a requested snapshot is saved, before any hook of the step;
- [observation hook] what the policy and the log see; the visible row of step
  t is logged after it;
- [end] the episode ends on the check's final verdict ("done"): success, or a
  failure the check has concluded (the fleet's check gives up `tail_steps`
  after LIBERO's first fire); else at the step limit, a failure with reason
  "step limit";
- [supervisor] once the risk estimator has fired, the supervisor decides; the
  robot holds still (`contracts.live.hold_action`) for the steps the decision
  took, at least one;
- [policy] the next commanded action: from a hold, from the recovery
  primitive, or from the action queue, which a policy call refills; logged
  as commanded;
- [action hook] between the queue and the simulator; the chain records the
  executed action;
- [simulator] one control step; the chain records the state and image hashes;
- [success check, data hook] the data hook sees every step's verdict but may
  flip only a final one, so that a flipped verdict never ends an episode
  early; the final verdict after the hook is the visible outcome, the one
  before it and LIBERO's first fire are truth.

Replay rules kept here: the noise of every call comes from `noise_for` (rule
2), the rollout owns the action queue and the step count (rule 4), a restore
puts back the full snapshot and uses its saved observation (rule 5).
"""

from __future__ import annotations

import math
import pickle
import time
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import (
    PIXEL_KEYS,
    REASON_HANDED_OVER,
    REASON_STEP_LIMIT,
    REASON_STOPPED,
    Camera,
    CellHandle,
    CellHook,
    Observation,
    SimEnv,
    SuccessCheck,
    SuccessVerdict,
    Undo,
)
from fleetlab.contracts.faults import (
    FaultSpec,
    FaultState,
    FaultType,
    FaultTypeBase,
    check_data_hook,
)
from fleetlab.contracts.hashing import hash_array
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, noise_for
from fleetlab.contracts.live import (
    Decision,
    DecisionRecord,
    OperatorQueue,
    RiskEstimator,
    StepTelemetry,
    Supervisor,
    SupervisorCallFailure,
    SupervisorContext,
    hold_action,
)
from fleetlab.contracts.log import EventKind
from fleetlab.contracts.policy import Action7, ControlMode, NoiseSample, Policy
from fleetlab.contracts.rollout import Job, JobMode, Snapshot, SnapshotRef
from fleetlab.engine.chain import Chain
from fleetlab.engine.records import CONTROL_HZ, FRAME_EVERY, EpisodeRecord, StepRows
from fleetlab.engine.recovery import RETRY_STEPS, Pregrasp, Recovery, primitive

TELEMETRY_WINDOW = 20
RECENT_FRAMES = 8


class EngineError(RuntimeError):
    """A job the engine refuses to run, or a broken invariant."""


@dataclass(frozen=True)
class ChunkRequest:
    """One row's request for a policy call."""

    key: str
    observation: Observation
    instruction: str
    noise: NoiseSample | None
    spec: PolicySpec  # the effective spec: the server groups by it
    row: int | None
    batch_size: int
    # The episode's policy call this is; the first is cut without the latency compensation.
    chunk_index: int = 0


class ChunkSource(Protocol):
    """Where the rollout gets its next actions from: the inference server or the policy itself."""

    def chunk(self, request: ChunkRequest) -> npt.NDArray[np.float32]:
        """The actions to queue, already converted to `Action7` and cut to the executed count."""
        ...


@dataclass
class DirectSource:
    """Calls a loaded server policy in this process, one row padded to the batch size.

    Padding repeats row 0, as the server does; for a row-independent policy the
    chunk is the same bits as in the fleet layout (the rows check proves it).
    """

    policy: Policy
    calls: int = 0

    def chunk(self, request: ChunkRequest) -> npt.NDArray[np.float32]:
        return call_policy(self.policy, [request])[0]


def call_key(spec: PolicySpec) -> tuple[Any, ...]:
    """What requests must share to go into one policy call.

    The sampling loop runs once per batch, so the settings it reads must agree:
    the policy, its revision and precision, the sampling steps and the image
    views. The control mode and the number of executed actions only change
    what happens to a row's chunk afterwards.
    """
    return (spec.name, spec.revision, spec.precision, spec.sampling_steps, spec.image_views)


def call_policy(policy: Policy, requests: Sequence[ChunkRequest]) -> list[npt.NDArray[np.float32]]:
    """One policy call for requests that share a call key, padded to the batch size.

    Rows: a request with a fixed row goes there, the others fill the free rows
    in order; empty rows are copies of row 0 (or of the first filled row). The
    group's effective spec goes with the call. Returns each request's actions,
    converted for its control mode and cut to its `actions_executed_per_chunk`,
    after skipping its `latency_compensation_steps` stale head, except in an
    episode's first call: the robot waits for that chunk, so nothing in it is
    stale when it arrives (at least the chunk's last action is kept; with no
    skip the cut is the plain `[:executed]`).
    """
    if not requests:
        return []
    key = call_key(requests[0].spec)
    batch_size = requests[0].batch_size
    if any(call_key(r.spec) != key or r.batch_size != batch_size for r in requests):
        raise EngineError("a policy call serves requests of one call key and batch size")
    if len(requests) > batch_size:
        raise EngineError(f"{len(requests)} requests do not fit a batch of {batch_size}")
    slots: list[ChunkRequest | None] = [None] * batch_size
    for r in requests:
        if r.row is not None:
            if slots[r.row] is not None:
                raise EngineError(f"two requests for row {r.row} in one call")
            slots[r.row] = r
    free = iter(i for i in range(batch_size) if slots[i] is None)
    for r in requests:
        if r.row is None:
            slots[next(free)] = r
    position = {n: next(i for i, s in enumerate(slots) if s is r) for n, r in enumerate(requests)}
    first = slots[0] if slots[0] is not None else next(s for s in slots if s is not None)
    filled = [s if s is not None else first for s in slots]
    chunks = policy.act(
        [r.observation for r in filled],
        [r.instruction for r in filled],
        [r.noise for r in filled],
        requests[0].spec,  # the group's effective spec: what the sampling loop reads
    )
    out = []
    for n, r in enumerate(requests):
        actions = policy.convert(chunks[position[n]], r.spec.control_mode)
        executed = np.stack([np.asarray(a, np.float32) for a in actions])
        skip = min(r.spec.latency_compensation_steps, len(executed) - 1) if r.chunk_index else 0
        out.append(executed[skip : skip + r.spec.actions_executed_per_chunk])
    return out


@dataclass(frozen=True)
class SupervisorParts:
    """A supervisor with its risk estimator and the risk at which the estimator fires."""

    estimator: RiskEstimator
    supervisor: Supervisor
    threshold: float = 0.5
    # False when the estimator requires history absent from the snapshot contract.
    supports_snapshot_start: bool = True
    should_call: Callable[[SupervisorContext], bool] | None = None
    operator_queue: Callable[[], OperatorQueue] = field(default=lambda: OperatorQueue(0, 0))
    # A visible terminal failure after choosing to continue reaches this deterministic rule.
    failure_fallback: Decision | None = None

    def __post_init__(self) -> None:
        if self.failure_fallback not in (None, Decision.hand_over, Decision.stop):
            raise ValueError("failure fallback must hand over or stop")


class SnapshotStore(Protocol):
    def save(self, snapshot: Snapshot, episode_id: str, key: str) -> SnapshotRef: ...

    def load(self, ref: SnapshotRef) -> Snapshot: ...


@dataclass
class DirectoryStore:
    """Snapshots as pickles under `truth/snapshots/` of the run directory (hidden)."""

    root: Path  # the run directory

    def save(self, snapshot: Snapshot, episode_id: str, key: str) -> SnapshotRef:
        from fleetlab.contracts.rollout import RunLayout

        path = RunLayout(self.root).snapshot_file(episode_id, key, snapshot.step)
        relative = path.relative_to(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(path)
        return SnapshotRef(episode_id=episode_id, step=snapshot.step, path=relative.as_posix())

    def load(self, ref: SnapshotRef) -> Snapshot:
        snapshot = pickle.loads((self.root / ref.path).read_bytes())
        if not isinstance(snapshot, Snapshot):
            raise EngineError(f"{ref.path} does not hold a snapshot")
        return snapshot


@dataclass
class MemoryStore:
    """Snapshots in memory, for tests."""

    snapshots: dict[str, Snapshot] = field(default_factory=dict)

    def save(self, snapshot: Snapshot, episode_id: str, key: str) -> SnapshotRef:
        path = f"memory/{key}.{snapshot.step}"
        self.snapshots[path] = snapshot
        return SnapshotRef(episode_id=episode_id, step=snapshot.step, path=path)

    def load(self, ref: SnapshotRef) -> Snapshot:
        return self.snapshots[ref.path]


@dataclass
class EpisodeContext:
    """What a worker hands the loop besides the job: its parts and settings."""

    fault_types: Callable[[str], FaultType]  # kind -> type (a registry's `get`)
    make_check: Callable[[], SuccessCheck]
    source: ChunkSource | None = None  # for server policies
    noise_shape: tuple[int, ...] | None = None  # the server policy's noise, if it needs any
    noise_dtype: str | None = None
    # The in-worker policy: it runs "continue" jobs, and any job whose nominal
    # policy has its name (the teleoperator from initial states). A factory: an in-worker
    # policy is built per episode or continuation, so it carries nothing across them.
    teleoperator: Callable[[], Policy] | None = None
    teleoperator_name: str | None = None  # its spec's name: the jobs whose policy it is
    supervisors: Callable[[str, Mapping[str, Any]], SupervisorParts] | None = None
    store: SnapshotStore | None = None
    snapshot_steps: Sequence[int] = ()
    continuation_steps: int | None = None  # a hand-over's own step budget (Job.max_steps)
    record_video: bool = False
    task_text: str = ""


def image_hashes(observation: Observation) -> tuple[str, str]:
    pixels = observation["pixels"]
    return hash_array(pixels[PIXEL_KEYS[Camera.scene]]), hash_array(
        pixels[PIXEL_KEYS[Camera.wrist]]
    )


def _bound_cell_hook(fault_type: FaultType, state: FaultState) -> CellHook:
    def hook(handle: CellHandle) -> Undo | None:
        return fault_type.cell_hook(handle, state)

    return hook


@dataclass
class _Pending:
    record: DecisionRecord
    apply_at: int


class _IdleCheck:
    """Freeze the check's clock during operator waits; physics still steps."""

    def __init__(self, check: Any) -> None:
        self.check = check
        self.idle = False
        self.idle_steps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.check, name)

    def update(self, state: Any, step: int) -> SuccessVerdict:
        self.idle_steps += int(self.idle)
        return self.check.update(state, step - self.idle_steps)  # type: ignore[no-any-return]

    def snapshot(self) -> bytes:
        return b"idle-check/1\0" + pickle.dumps((self.idle_steps, self.check.snapshot()))

    def restore(self, blob: bytes) -> None:
        if blob.startswith(b"idle-check/1\0"):
            self.idle_steps, blob = pickle.loads(blob[len(b"idle-check/1\0") :])
        self.check.restore(blob)


def effective_spec(
    identity: EpisodeIdentity, pairs: Sequence[tuple[FaultType, FaultState]]
) -> PolicySpec:
    spec = identity.policy
    for fault_type, state in pairs:
        spec = fault_type.policy_input_hook(spec, state)
    return spec


def run_job(job: Job, env: SimEnv, ctx: EpisodeContext) -> EpisodeRecord:
    """Run one job on `env` (built for its task) and return what it produced."""
    from fleetlab.faults.command import action_context, in_policy_frame

    if (
        job.mode is JobMode.scripted
        and job.from_snapshot is not None
        and job.from_snapshot.step > 0
    ):
        raise EngineError("scripted decision replay must start from reset, not mid-attempt")
    sup: SupervisorParts | None = None
    if job.supervisor is not None:
        if ctx.supervisors is None:
            raise EngineError("this worker has no supervisors")
        sup = ctx.supervisors(job.supervisor.name, job.supervisor.params)
    if (
        job.from_snapshot is not None
        and job.from_snapshot.step > 0
        and sup is not None
        and not sup.supports_snapshot_start
    ):
        raise EngineError(
            "supervisor estimator history is not saved in snapshots; replay the trace from reset"
        )
    t_start = time.perf_counter()
    identity = job.identity
    faults: list[FaultSpec] = job.effective_faults
    snapshot: Snapshot | None = None
    if job.from_snapshot is not None:
        if ctx.store is None:
            raise EngineError("a job from a snapshot needs a snapshot store")
        snapshot = ctx.store.load(job.from_snapshot)

    teleop_name = _teleoperator_name(ctx)
    in_worker = job.mode is JobMode.continue_ or (
        teleop_name is not None and identity.policy.name == teleop_name
    )
    # built only for the jobs that run it, one per episode or continuation
    policy: Policy | None = (
        ctx.teleoperator() if in_worker and ctx.teleoperator is not None else None
    )
    if in_worker and (policy is None or not policy.in_worker):
        raise EngineError("a continue job needs an in-worker teleoperator")
    if not in_worker and ctx.source is None:
        raise EngineError("a job with a server policy needs a chunk source")

    if snapshot is not None:
        refuse_cell_changes(job, ctx)
    # 1. a fresh state per fault, before the cell hooks (contracts.env)
    types = [ctx.fault_types(f.kind) for f in faults]
    pairs = [(ft, ft.new_episode(f, identity)) for ft, f in zip(types, faults, strict=True)]
    # 2. the effective spec; the environment takes its control mode from it
    spec = effective_spec(identity, pairs)
    if spec.precision != identity.policy.precision:
        raise EngineError("a fault cannot change the precision of a loaded policy")
    env_identity = (
        identity if spec == identity.policy else identity.model_copy(update={"policy": spec})
    )
    mode = spec.control_mode
    # 3. the reset sequence with the cell hooks
    check = ctx.make_check()  # the environment resets it with the task's goal
    configure_takeover = getattr(policy, "configure_takeover", None)
    if callable(configure_takeover):
        if snapshot is None:
            raise EngineError("a human teleoperator needs a hand-over snapshot")
        configure_takeover(job.teleoperator_params, getattr(snapshot, "last_commanded", None))
    if (
        callable(configure_takeover)
        or sup is not None
        or job.mode is JobMode.scripted
        or (snapshot is not None and snapshot.success_check.startswith(b"idle-check/1\0"))
    ):
        check = _IdleCheck(check)
    env.set_success_check(check)
    cell_hooks = [_bound_cell_hook(ft, st) for ft, st in pairs]
    t0 = time.perf_counter()
    obs_raw = env.reset(env_identity, cell_hooks)
    reset_s = time.perf_counter() - t0

    queue: deque[Action7] = deque()
    chunk_index = 0
    chunk_step = 0
    t = 0
    if snapshot is not None:
        # 4. continue from the snapshot: environment, queue, chunk number, step, fault states
        obs_raw = env.restore(snapshot.env)
        check.restore(snapshot.success_check)
        queue = deque(np.asarray(a, np.float32) for a in snapshot.queued_actions)
        chunk_index = snapshot.chunk_index
        t = snapshot.step
        chunk_step = _chunk_step_after(snapshot, spec)
        restored = []
        for ft, st in pairs:
            blob = snapshot.fault_states.get(st.spec.fault_id)
            restored.append((ft, ft.restore(blob) if blob is not None else st))
        pairs = restored
    # after the reset or the restore: every model array a cell hook can change (section 4)
    chain = Chain(fixture_hash=env.fixture_hash().hex())
    scene_h, wrist_h = image_hashes(obs_raw)
    chain.add(t, env.state_hash().hex(), scene_h, wrist_h, None)
    first_step = t

    if policy is not None:
        policy.begin_episode(identity, t)  # type: ignore[attr-defined]

    trace = {d.step: d for d in job.decision_trace or []}
    if len(trace) != len(job.decision_trace or []):
        raise EngineError("decision trace contains duplicate steps")

    rows = StepRows()
    events: list[tuple[int, EventKind, str]] = []
    decisions: list[DecisionRecord] = []
    frames: list[tuple[int, npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = []
    snapshots: list[SnapshotRef] = []
    spawned: list[Job] = []
    telemetry: deque[StepTelemetry] = deque(maxlen=TELEMETRY_WINDOW)
    recent: dict[Camera, deque[npt.NDArray[np.uint8]]] = {
        c: deque(maxlen=RECENT_FRAMES) for c in Camera
    }
    frame_steps: deque[int] = deque(maxlen=RECENT_FRAMES)
    snapshot_steps = set(ctx.snapshot_steps) | set(job.snapshot_steps)
    record_video = ctx.record_video and job.mode is JobMode.run
    pending: _Pending | None = None
    holding = 0
    recovery: Recovery | None = None
    pregrasp = Pregrasp()
    retry_used = bool(getattr(snapshot, "retry_used", False))
    retry_started_at: int | None = None
    retry_active_steps = 0
    robot_active_steps = 0
    degraded = bool(getattr(snapshot, "degraded", False))
    decision_hold_steps = 0
    last_cmd: Action7 | None = getattr(snapshot, "last_commanded", None)
    risk: float | None = None
    true_fired: int | None = None  # the step the true verdict ended the episode a success
    policy_done = False  # an in-worker policy said it has finished
    # A continuation may have its own budget beyond the suite's limit (Job.max_steps).
    budget = getattr(job, "step_budget", identity.max_steps)
    if job.mode is JobMode.continue_ and job.max_steps is not None:
        budget += first_step
    elif snapshot is not None:
        remaining = getattr(snapshot, "remaining_active_steps", None)
        if remaining is None:
            # Pre-budget snapshots retained the cumulative idle clock in the check envelope.
            idle_steps = check.idle_steps if isinstance(check, _IdleCheck) else 0
            remaining = max(0, budget - (first_step - idle_steps))
        budget = first_step + remaining
    ended_by = "limit" if t >= budget else ""
    verdict_true = SuccessVerdict(False, False, "")
    visible = SuccessVerdict(False, False, "")
    native_first: int | None = None
    waits_ms: list[float] = []
    step_s = 0.0
    calls = 0
    last_gripper = float(last_cmd[6]) if last_cmd is not None else 0.0

    while True:
        if trace and min(trace) < t:
            raise EngineError(f"unapplied decision trace entry at step {min(trace)}")
        applied = False
        # A terminal failure is visible to a deployed supervisor too. This backstop
        # is a rule, not another model call; it consumes no hold tick or operator time.
        if ended_by in ("check", "limit", "policy done") and not visible.success:
            terminal = trace.get(t) if job.mode is JobMode.scripted else None
            if terminal is not None and terminal.call_status == "rule":
                if terminal.hold_steps != 0 or terminal.decision not in (
                    Decision.hand_over,
                    Decision.stop,
                ):
                    raise EngineError("invalid terminal fallback trace entry")
                trace.pop(t)
            elif sup is not None and sup.failure_fallback is not None:
                terminal = DecisionRecord(
                    step=t,
                    decision=sup.failure_fallback,
                    hold_steps=0,
                    decision_ms=0,
                    model_ms=0,
                    call_status="rule",
                    rationale="visible failure backstop after the policy's opportunity to finish",
                )
            else:
                terminal = None
            if terminal is not None:
                decisions.append(terminal)
                events.append((t, EventKind.supervisor_call, f"decided {terminal.decision.value}"))
                pending = _Pending(terminal, t)
                ended_by = ""
        # [decision] apply a decision whose hold has run out
        if pending is not None and t >= pending.apply_at and not ended_by:
            applied = True
            record, pending = pending.record, None
            decision = record.decision
            if decision is Decision.stop:
                ended_by = "stop"
            elif decision is Decision.hand_over and job.mode is not JobMode.run:
                # A scripted replay proves the run job's chain up to the hand-over; the
                # continuation exists already, from the run job.
                ended_by = "hand_over"
            elif decision is Decision.hand_over:
                ref = _save(
                    ctx,
                    env,
                    check,
                    queue,
                    chunk_index,
                    t,
                    pairs,
                    identity,
                    job.key,
                    last_cmd,
                    remaining_active_steps=max(0, budget - t),
                    degraded=degraded,
                    retry_used=retry_used,
                )
                snapshots.append(ref)
                spawned.append(
                    Job(
                        identity=identity,
                        mode=JobMode.continue_,
                        from_snapshot=ref,
                        max_steps=ctx.continuation_steps,
                    )
                )
                events.append((t, EventKind.handover, "handed to a teleoperator"))
                ended_by = "hand_over"
            elif decision in (Decision.retry, Decision.recover):
                assert last_cmd is not None
                if decision is Decision.retry:
                    if (
                        retry_used
                        or mode is ControlMode.relative
                        or pregrasp.command is None
                        or identity.max_steps <= RETRY_STEPS
                    ):
                        raise EngineError(
                            "retry is unavailable: needs one tracked absolute grasp approach"
                        )
                    if record.retry_budget_steps is None:
                        raise EngineError(
                            "legacy retry trace has no retry budget; rerun with current semantics"
                        )
                    if record.retry_budget_steps != identity.max_steps:
                        raise EngineError("retry budget must equal the attempt's nominal budget")
                    retry_used = True
                    retry_started_at = t
                    budget = t + record.retry_budget_steps
                recovery = primitive(decision, mode, last_cmd, rows.eef_pos[-1], pregrasp.command)
                queue.clear()
                events.append((t, EventKind.retry, decision.value + ": release and retreat"))
            elif decision is Decision.degrade:
                degraded = True
                queue.clear()
        # [snapshot] before any hook of the step
        if t in snapshot_steps and not ended_by:
            snapshots.append(
                _save(
                    ctx,
                    env,
                    check,
                    queue,
                    chunk_index,
                    t,
                    pairs,
                    identity,
                    job.key,
                    last_cmd,
                    remaining_active_steps=max(0, budget - t),
                    degraded=degraded,
                    retry_used=retry_used,
                )
            )
        # [observation hook] then log what the fleet sees
        obs = obs_raw
        for ft, st in pairs:
            obs = ft.observation_hook(obs, st)
        _log_state(rows, t, obs)
        if record_video:
            if t % FRAME_EVERY == 0:
                frames.append((t, *_upright(obs)))
                rows.frame_index.append(len(frames) - 1)
            else:
                rows.frame_index.append(None)
        else:
            rows.frame_index.append(None)
        if sup is not None and t % FRAME_EVERY == 0:
            for camera in Camera:
                recent[camera].append(obs["pixels"][PIXEL_KEYS[camera]].copy())
            frame_steps.append(t)
        # [end] on the check's final verdict ("done"), or at the step limit
        if ended_by:
            break
        # [supervisor] or the recorded decision of a scripted replay
        if pending is None and holding == 0 and not recovery and not applied:
            made = _decide(
                job,
                sup,
                trace,
                t,
                risk,
                telemetry,
                recent,
                ctx,
                obs,
                t - first_step - decision_hold_steps,
                budget - t,
                decisions,
                frame_steps,
                in_worker,
                not retry_used
                and identity.max_steps > RETRY_STEPS
                and pregrasp.available(mode, last_cmd, rows.eef_pos[-1]),
            )
            if made is not None:
                decisions.append(made)
                events.append((t, EventKind.supervisor_call, f"decided {made.decision.value}"))
                pending = _Pending(made, t + made.hold_steps)
                holding = made.hold_steps
        # [policy] the commanded action
        c_index: int | None = None
        c_step: int | None = None
        decision_idle = holding > 0
        # Holds, queued chunks and recovery use the saved policy command frame.
        command_world_frame = False
        if holding > 0:
            if last_cmd is None:
                raise EngineError("the robot cannot hold still before its first command")
            cmd = hold_action(mode, last_cmd)
            holding -= 1
            decision_hold_steps += 1
        elif recovery:
            cmd = recovery.action(obs["robot_state"]["eef"]["pos"])
            command_world_frame = recovery.commands_in_world_frame
        elif in_worker:
            assert policy is not None
            cmd = np.asarray(policy.act_one(obs, env), np.float32)
            frame = getattr(policy, "commands_in_world_frame", None)
            if not isinstance(frame, bool):
                raise EngineError("an in-worker policy must declare commands_in_world_frame")
            command_world_frame = frame
            c_index, c_step = chunk_index, 0
            chunk_index += 1
            if getattr(policy, "done", False) and not ended_by:
                policy_done = True  # the continuation ends after this step's verdict
        else:
            if not queue:
                assert ctx.source is not None
                noise = _noise(identity, chunk_index, ctx.noise_shape, ctx.noise_dtype)
                request = ChunkRequest(
                    key=job.key,
                    observation=obs,
                    instruction=env.task_description,
                    noise=noise,
                    spec=spec,
                    row=identity.row,
                    batch_size=identity.batch_size,
                    chunk_index=chunk_index,
                )
                t1 = time.perf_counter()
                actions = ctx.source.chunk(request)
                waits_ms.append(1000 * (time.perf_counter() - t1))
                calls += 1
                queue = deque(np.asarray(a, np.float32) for a in actions)
                if degraded:
                    queue = deque(list(queue)[:4])
                chunk_index += 1
                chunk_step = 0
            cmd = queue.popleft()
            c_index, c_step = chunk_index - 1, chunk_step
            chunk_step += 1
        cmd = np.asarray(cmd, np.float32)
        if command_world_frame:
            cmd = in_policy_frame(cmd, pairs, mode)
        rows.commanded.append(cmd)
        rows.chunk_index.append(c_index)
        rows.chunk_step.append(c_step)
        gripper = float(cmd[6])
        if c_index is not None:
            pregrasp.observe(cmd, rows.eef_pos[-1])
        if last_cmd is not None and np.sign(gripper) != np.sign(last_gripper) and gripper != 0:
            kind = EventKind.grasp if gripper > 0 else EventKind.release
            events.append((t, kind, "gripper commanded " + ("closed" if gripper > 0 else "open")))
        last_gripper = gripper
        last_cmd = cmd
        if sup is not None:
            tel = _telemetry(rows, t, cmd, c_index, c_step)
            telemetry.append(tel)
            risk = sup.estimator.update(tel)
        # [action hook]
        executed = cmd
        with action_context(mode):
            for ft, st in pairs:
                executed = np.asarray(ft.action_hook(executed, st), np.float32)
        # [simulator]
        idle = decision_idle or bool(getattr(policy, "idle", False))
        if isinstance(check, _IdleCheck):
            check.idle = idle
        if idle:
            budget += 1  # waiting time counts, but does not spend the active attempt budget
        else:
            robot_active_steps += int(job.mode is not JobMode.continue_)
            if retry_started_at is not None:
                retry_active_steps += 1
        t2 = time.perf_counter()
        result = env.step(executed)
        step_s += time.perf_counter() - t2
        obs_raw = result.observation
        t += 1
        scene_h, wrist_h = image_hashes(obs_raw)
        chain.add(t, env.state_hash().hex(), scene_h, wrist_h, executed)
        # [success check, data hook]
        true = result.verdict
        if result.native_success and native_first is None:
            native_first = t
        if not true.done and (t >= budget or policy_done):
            true = SuccessVerdict(False, True, REASON_STEP_LIMIT)
            ended_by = "policy done" if policy_done and t < budget else "limit"
        shown = true
        for ft, st in pairs:
            hooked = ft.data_hook(shown, st)
            try:
                check_data_hook(shown, hooked)
            except ValueError as error:
                raise EngineError(f"fault {st.spec.fault_id} ({st.spec.kind}): {error}") from None
            shown = hooked
        if shown.done:  # unchanged by the hooks: the episode ends here
            verdict_true, visible = true, shown
            if true.success:
                true_fired = t
            ended_by = ended_by or "check"

    if trace:
        raise EngineError(f"unapplied decision trace entries: {sorted(trace)}")
    if ended_by in ("stop", "hand_over"):
        # ended by the supervisor, without a verdict of the check
        reason = REASON_STOPPED if ended_by == "stop" else REASON_HANDED_OVER
        verdict_true = visible = SuccessVerdict(False, True, reason)
    if record_video and frames and frames[-1][0] != t:
        frames.append((t, *_upright(obs)))
        rows.frame_index[-1] = len(frames) - 1
    rows.commanded.append(None)
    rows.chunk_index.append(None)
    rows.chunk_step.append(None)
    env_close = getattr(env, "end_episode", None)
    if callable(env_close):
        env_close()  # undo the cell hooks now; the next reset would do it otherwise
    stats = {
        "first_step": first_step,
        "policy_calls": calls,
        "chunk_index_end": chunk_index,
        "wait_ms": [round(w, 1) for w in waits_ms],
        "reset_s": round(reset_s, 4),
        "step_s": round(step_s, 4),
        "wall_s": round(time.perf_counter() - t_start, 4),
        "decision_hold_steps": decision_hold_steps,
        "retry_started_at": retry_started_at,
        "retry_budget_steps": identity.max_steps if retry_started_at is not None else 0,
        "retry_active_steps": retry_active_steps,
        "robot_active_steps": robot_active_steps,
        "robot_time_s": robot_active_steps / CONTROL_HZ
        + sum(d.decision_ms for d in decisions) / 1000,
    }
    metrics = getattr(policy, "metrics", None)
    if callable(metrics):
        stats["human"] = metrics()
    return EpisodeRecord(
        job=job,
        effective_faults=faults,
        effective_spec=spec,
        in_worker=in_worker,
        steps=rows,
        chain=chain,
        true_success=true_fired is not None,
        success_step=true_fired,
        placed_step=true_fired if getattr(check, "reports_placed", False) else None,
        true_reason=verdict_true.reason,
        visible_success=visible.success,
        libero_first_success_step=native_first,
        events=events,
        decisions=decisions,
        frames=frames,
        snapshots=snapshots,
        spawned=spawned,
        ended_by=ended_by,
        stats=stats,
    )


def _teleoperator_name(ctx: EpisodeContext) -> str | None:
    """The name of the in-worker policy's spec, without building one if it can be helped."""
    if ctx.teleoperator is None:
        return None
    if ctx.teleoperator_name is not None:
        return ctx.teleoperator_name
    spec = getattr(ctx.teleoperator, "spec", None)  # a class
    if spec is None:
        spec = ctx.teleoperator().spec
    return str(spec.name)


def uses_cell_hook(fault_type: FaultType) -> bool:
    """Whether a fault type changes the live model (overrides `cell_hook`)."""
    return getattr(type(fault_type), "cell_hook", None) is not FaultTypeBase.cell_hook


def refuse_cell_changes(job: Job, ctx: EpisodeContext) -> None:
    """A replay from a snapshot may not remove or add a fault that uses a cell hook.

    The snapshot's state was simulated with the cell as it was; removing a
    camera turn or a furniture shift from it would continue a scene the fault
    shaped (contracts, section 10, version 1).
    """
    by_id = {f.fault_id: f for f in job.identity.faults}
    changed = [by_id[i] for i in job.remove] + list(job.add)
    named = sorted({f.kind for f in changed if uses_cell_hook(ctx.fault_types(f.kind))})
    if named:
        raise EngineError(
            f"{job.key}: a replay from a snapshot cannot remove or add a fault with a cell hook "
            f"({', '.join(named)}); replay it from the reset"
        )


def _chunk_step_after(snapshot: Snapshot, spec: PolicySpec) -> int:
    """Where in its chunk the next queued action sits (for the visible chunk_step column)."""
    return max(0, spec.actions_executed_per_chunk - len(snapshot.queued_actions))


def _noise(
    identity: EpisodeIdentity,
    chunk_index: int,
    shape: tuple[int, ...] | None,
    dtype: str | None,
) -> NoiseSample | None:
    """The noise of one call, from the identity (replay rule 2); None for a policy without."""
    if shape is None or dtype is None:
        return None
    values = noise_for(identity, chunk_index, tuple(shape), dtype)
    return NoiseSample(noise_seed=identity.noise_seed, chunk_index=chunk_index, values=values)


def _save(
    ctx: EpisodeContext,
    env: SimEnv,
    check: SuccessCheck,
    queue: deque[Action7],
    chunk_index: int,
    step: int,
    pairs: Sequence[tuple[FaultType, FaultState]],
    identity: EpisodeIdentity,
    key: str,
    last_commanded: Action7 | None = None,
    *,
    remaining_active_steps: int | None = None,
    retry_used: bool = False,
    degraded: bool = False,
) -> SnapshotRef:
    if ctx.store is None:
        raise EngineError("snapshots need a snapshot store")
    snapshot = Snapshot(
        env=env.snapshot(),
        queued_actions=[np.asarray(a, np.float32).copy() for a in queue],
        chunk_index=chunk_index,
        step=step,
        fault_states={st.spec.fault_id: ft.snapshot(st) for ft, st in pairs},
        success_check=check.snapshot(),
        last_commanded=None if last_commanded is None else last_commanded.copy(),
        remaining_active_steps=remaining_active_steps,
        retry_used=retry_used,
        degraded=degraded,
    )
    return ctx.store.save(snapshot, identity.episode_id, key)


def _decide(
    job: Job,
    sup: SupervisorParts | None,
    trace: MutableMapping[int, DecisionRecord],
    t: int,
    risk: float | None,
    telemetry: Sequence[StepTelemetry],
    recent: Mapping[Camera, Sequence[npt.NDArray[np.uint8]]],
    ctx: EpisodeContext,
    obs: Observation,
    active_step: int,
    remaining_active_steps: int,
    decisions: Sequence[DecisionRecord],
    frame_steps: Sequence[int],
    in_worker: bool,
    retry_available: bool,
) -> DecisionRecord | None:
    if job.mode is JobMode.scripted:
        record = trace.pop(t, None)
        if record is None:
            return None
        if record.call_status == "rule" or record.hold_steps < 1:
            raise EngineError("terminal rule trace entry before the attempt's visible failure")
        return record
    if sup is None or risk is None or not (risk >= sup.threshold):
        return None
    context = SupervisorContext(
        recent_frames={c: list(v) for c, v in recent.items()},
        telemetry_window=list(telemetry),
        risk=risk,
        task_text=obs["task_description"],
        options=[
            d
            for d in Decision
            if not (in_worker and d is Decision.degrade)
            and not (d is Decision.retry and not retry_available)
        ],
        operator_queue=sup.operator_queue(),
        active_step=active_step,
        remaining_active_steps=remaining_active_steps,
        previous_decisions=tuple(d.decision for d in decisions),
        frame_steps=tuple(frame_steps),
    )
    if sup.should_call is not None and not sup.should_call(context):
        return None
    try:
        answer = sup.supervisor.decide(context)
    except SupervisorCallFailure as failure:
        fallback = Decision.hand_over if Decision.hand_over in context.options else Decision.stop
        if not math.isfinite(failure.decision_ms) or failure.decision_ms < 0:
            raise EngineError("failed call timing must be finite and nonnegative") from failure
        if failure.model_ms is not None and (
            not math.isfinite(failure.model_ms) or failure.model_ms < 0
        ):
            raise EngineError(
                "failed call model timing must be finite and nonnegative"
            ) from failure
        return DecisionRecord(
            step=t,
            decision=fallback,
            hold_steps=max(1, math.ceil(failure.decision_ms * CONTROL_HZ / 1000)),
            decision_ms=failure.decision_ms,
            model_ms=failure.model_ms,
            call_status="failed" if failure.called else "skipped",
            error=str(failure),
            rationale=f"model route failed; declared fallback: {fallback.value}",
        )
    if answer.decision not in context.options:
        raise EngineError("supervisor chose an unavailable decision")
    if not all(math.isfinite(v) and v >= 0 for v in (answer.decision_ms, answer.model_ms)):
        raise EngineError("supervisor timing must be finite and nonnegative")
    hold = max(1, math.ceil(answer.decision_ms * CONTROL_HZ / 1000.0))
    return DecisionRecord(
        step=t,
        decision=answer.decision,
        hold_steps=hold,
        decision_ms=answer.decision_ms,
        model_ms=answer.model_ms,
        rationale=answer.rationale,
        retry_budget_steps=job.identity.max_steps if answer.decision is Decision.retry else None,
    )


def _log_state(rows: StepRows, t: int, obs: Observation) -> None:
    state = obs["robot_state"]
    rows.step.append(t)
    rows.joint_pos.append(np.asarray(state["joints"]["pos"], np.float64))
    rows.joint_vel.append(np.asarray(state["joints"]["vel"], np.float64))
    rows.gripper_qpos.append(np.asarray(state["gripper"]["qpos"], np.float64))
    rows.eef_pos.append(np.asarray(state["eef"]["pos"], np.float64))
    rows.eef_quat.append(np.asarray(state["eef"]["quat"], np.float64))


def _telemetry(
    rows: StepRows, t: int, cmd: Action7, c_index: int | None, c_step: int | None
) -> StepTelemetry:
    return StepTelemetry(
        step=t,
        time_s=t / CONTROL_HZ,
        joint_pos=rows.joint_pos[-1],
        joint_vel=rows.joint_vel[-1],
        gripper_qpos=rows.gripper_qpos[-1],
        eef_pos=rows.eef_pos[-1],
        eef_quat=rows.eef_quat[-1],
        commanded_action=np.asarray(cmd, np.float32),
        chunk_index=-1 if c_index is None else c_index,
        chunk_step=-1 if c_step is None else c_step,
    )


def _upright(obs: Observation) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Both camera images upright, for the videos (the simulator renders them rotated)."""
    pixels = obs["pixels"]
    scene = np.ascontiguousarray(pixels[PIXEL_KEYS[Camera.scene]][::-1, ::-1])
    wrist = np.ascontiguousarray(pixels[PIXEL_KEYS[Camera.wrist]][::-1, ::-1])
    return scene, wrist
