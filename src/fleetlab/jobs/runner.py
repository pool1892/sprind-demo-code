"""Job programs played from starts on the physics simulator, recorded as the job tables.

One `LiberoSimEnv` of the base task serves every start of a request: each start resets it
from the start's identity (`LiberoEnv.reset(init_state, seed)` needs both), and the
program's steps then run in that one continuous simulator state (`contracts.jobs`):

- a policy step asks the policy for chunks under its instruction, as the engine does
  (`engine.episode.call_policy`, X-VLA predicting 30 actions and executing the program's
  `actions_per_chunk` prefix, or all 30 when unset, the noise of
  call k from `noise_for(identity, k)` with k counted over the whole run), and ends when its
  goal holds (`reached`), when its step count is used (`reached`) or at `max_steps`
  (`timeout`, then `on_timeout`). The goal is looked at before every action, so a goal that
  holds when the step begins ends it at once. Whatever is left of a chunk is dropped at
  every step boundary;
- a motion step plays its segment like a one-segment probe (`contracts.probes.
  program_trajectory`) from the end effector's position and orientation measured at the
  step's start, the gripper keeping its last command unless the segment opens or closes it;
- a wait repeats the last command (`contracts.live.hold_action`), or holds the measured
  pose with the gripper open when nothing was commanded yet;
- a home step sends the hand back to the pose it reported after the reset, the gripper
  commanded open: the target moves in a straight line at `HOME.max_speed_m_s` (the probes'
  bound, 2.5 cm a control step) from where the hand is, its orientation turning along by
  spherical interpolation, then stays there until the hand is within `HOME.reach_m` of the
  start position and slower than `HOME.rest_m_s` (`reached`) or the step's budget is used
  (`timeout`, then `on_timeout`).

After a program whose last step is a policy step that `reached`, the policy goes on under
that instruction until the delivery check holds or `DELIVERY.tail_steps` (60) steps have
passed, the same for every program (the chunk is not dropped there: it is the same step
going on). Every requested goal is evaluated at every control step with LIBERO's own
predicates (`jobs.check.SceneGoals`); the delivery check and the order rules are evaluated
on the recorded trace at the end (`jobs.check`, `contracts.jobs.order_verdicts`).

Everything a replay needs is recorded: the actions as they reached the simulator and the
hash of MuJoCo's integration state after the reset and after every step
(`LiberoSimEnv.state_hash`). There are no action hooks and no faults, so the executed
action is the commanded one.

The batch command runs one session's queue (lane T's driver), as one heavy job:

    FLEETLAB_MIOPEN_DIR=$HOME/.cache/robots-probes/engine/miopen TMPDIR=/var/tmp/robots-jobs \\
        tools/heavy-run uv run --locked python -m fleetlab.jobs.runner --batch B.json --out DIR

`B.json` is `{"format": "fleetlab.job-batch/1", "state": <RequestState>, "jobs": [{"job_id":
"j0001", "request": <JobRequest>, "first_episode": <optional int>}]}`. Without
`first_episode` the episodes of job `jNNNN` are numbered NNNN x 100 + 1, + 2, ... The output
is described in `fleetlab.jobs.tables`. `--policy hold` replaces X-VLA by a policy that holds
still, for a check of the machinery without the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.env import PIXEL_KEYS, Camera, Observation
from fleetlab.contracts.identity import EpisodeIdentity, PolicySpec, SoftwareVersion
from fleetlab.contracts.ids import IdKind, neutral_id
from fleetlab.contracts.jobs import (
    DELIVERY,
    GOAL_SUITE,
    GOAL_SUITE_TASKS,
    HOME,
    JOB_ID_PATTERN,
    DeliveryVerdict,
    GoalPredicate,
    GoalTrace,
    JobEpisode,
    JobProgram,
    JobRequest,
    JobStep,
    JobStepKind,
    OnTimeout,
    ProgramStepResult,
    RequestedGoal,
    RequestState,
    StartIdentity,
    StepOutcome,
    StepUntil,
    order_verdicts,
    program_problems,
)
from fleetlab.contracts.policy import ControlMode
from fleetlab.contracts.probes import ProbePrimitive, program_trajectory
from fleetlab.jobs import tables
from fleetlab.jobs.check import DeliveryState, SceneGoals, delivery_verdict

__all__ = [
    "BATCH_FORMAT",
    "FRAME_EVERY",
    "IMAGE_SIZE",
    "ChunkSource",
    "GoalSuite",
    "HoldSource",
    "JobRunner",
    "Played",
    "XVLASource",
    "load_xvla",
    "main",
    "naive_program",
    "run_batch",
]

BATCH_FORMAT = "fleetlab.job-batch/1"
RUN_FORMAT = "fleetlab.job-run/1"
IMAGE_SIZE = 360  # what the policy sees; a re-render can use another (fleetlab.jobs.replay)
PRECISION = "fp16_res32"
CONTROL_HZ = 20
FRAME_EVERY = 4  # the engine's video cadence: every fourth control step, 5 frames a second
NAIVE_NAME = "naive-chain"


# ---- the goal suite ------------------------------------------------------------------------


class GoalSuite:
    """The goal suite as LIBERO's benchmark lists it: per plan task number, LIBERO's
    benchmark index, the task's language (the instruction the engine gives the policy) and
    the goal its BDDL file states."""

    def __init__(self) -> None:
        from fleetlab.runtime import configure_process

        configure_process()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        tasks = benchmark.get_benchmark_dict()[GOAL_SUITE]()
        by_name = {tasks.get_task(i).name: i for i in range(tasks.n_tasks)}
        self.bench = tuple(by_name[stem] for stem in GOAL_SUITE_TASKS)
        self.language = tuple(tasks.get_task(b).language for b in self.bench)
        folder = Path(get_libero_path("bddl_files")) / GOAL_SUITE
        goals = []
        for stem in GOAL_SUITE_TASKS:
            parsed = robosuite_parse_problem(str(folder / f"{stem}.bddl"))["goal_state"]
            goals.append(tuple(parsed_goal(p) for p in parsed))
        self.goals: tuple[tuple[RequestedGoal, ...], ...] = tuple(goals)

    def instructions(self) -> dict[str, str]:
        """Each task's single goal (as `RequestedGoal.text`) and the task's instruction."""
        return {g[0].text: lang for g, lang in zip(self.goals, self.language, strict=True)}


def parsed_goal(predicate: Sequence[str]) -> RequestedGoal:
    """A goal as LIBERO's BDDL parser returns it (`['on', 'a', 'b']`) as a `RequestedGoal`."""
    by_lower = {p.value.lower(): p for p in GoalPredicate}
    return RequestedGoal(predicate=by_lower[predicate[0].lower()], objects=list(predicate[1:]))


def naive_program(goals: Sequence[RequestedGoal], instructions: dict[str, str]) -> JobProgram:
    """The naive chain: each goal's skill under its own instruction until
    LIBERO's predicate of the goal holds, at most 300 steps each, nothing in between."""
    steps = []
    for i, goal in enumerate(goals):
        if goal.text not in instructions:
            raise ValueError(f"no goal-suite skill has the goal {goal.text}")
        steps.append(
            JobStep(
                kind=JobStepKind.policy,
                instruction=instructions[goal.text],
                until=StepUntil(goal=i, steps=None),
                max_steps=300,
                segment=None,
                steps=None,
            )
        )
    return JobProgram(
        name=NAIVE_NAME,
        purpose="the skills chained, each switching on its own task's native success",
        steps=steps,
        notes="host-built naive chain (demo plan, section 3)",
    )


# ---- policies --------------------------------------------------------------------------------


class ChunkSource(Protocol):
    """Where a policy step's actions come from: the policy, one call per chunk."""

    @property
    def spec(self) -> PolicySpec: ...

    def chunk(
        self, observation: Observation, instruction: str, identity: EpisodeIdentity, index: int
    ) -> npt.NDArray[np.float32]:
        """The actions of call `index` of the run, as `Action7`s to execute in order."""
        ...


@dataclass
class XVLASource:
    """X-VLA called as the engine calls it, one row, batch size one."""

    policy: Any
    calls: int = 0
    seconds: float = 0.0

    @property
    def spec(self) -> PolicySpec:
        spec: PolicySpec = self.policy.spec
        return spec

    def chunk(
        self, observation: Observation, instruction: str, identity: EpisodeIdentity, index: int
    ) -> npt.NDArray[np.float32]:
        from fleetlab.contracts.identity import noise_for
        from fleetlab.contracts.policy import NoiseSample
        from fleetlab.engine.episode import ChunkRequest, call_policy

        shape = tuple(self.policy.noise_shape)
        values = noise_for(identity, index, shape, self.policy.noise_dtype)
        noise = NoiseSample(noise_seed=identity.noise_seed, chunk_index=index, values=values)
        request = ChunkRequest(
            key=identity.episode_id,
            observation=observation,
            instruction=instruction,
            noise=noise,
            spec=identity.policy,
            row=None,
            batch_size=1,
            chunk_index=index,
        )
        t0 = time.perf_counter()
        actions = call_policy(self.policy, [request])[0]
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return np.asarray(actions, np.float32)


@dataclass
class HoldSource:
    """A stand-in policy that holds the measured pose with the gripper as given (no model)."""

    gripper: float = -1.0
    length: int = 30
    calls: int = 0
    seconds: float = 0.0
    spec: PolicySpec = field(default_factory=lambda: _hold_spec())

    def chunk(
        self, observation: Observation, instruction: str, identity: EpisodeIdentity, index: int
    ) -> npt.NDArray[np.float32]:
        self.calls += 1
        action = hold_here(observation, self.gripper)
        return np.repeat(action[None], self.length, axis=0)


def _hold_spec() -> PolicySpec:
    return PolicySpec(
        name="hold-still",
        revision="1",
        precision="fp32",
        control_mode=ControlMode.absolute,
        actions_executed_per_chunk=30,
        sampling_steps=None,
        image_views=1,
    )


def load_xvla() -> tuple[XVLASource, SoftwareVersion]:
    """X-VLA on the GPU in fp16_res32 with the pinned convolution database, batch size one."""
    from fleetlab.engine import miopen, software
    from fleetlab.policies.xvla import XVLAPolicy

    miopen.activate()
    soft = software.current(miopen.require(1, PRECISION))
    policy = XVLAPolicy()
    policy.load("cuda", PRECISION)
    return XVLASource(policy), soft


def hold_here(observation: Observation, gripper: float = -1.0) -> npt.NDArray[np.float32]:
    """Absolute control: the measured position and orientation, the gripper as given."""
    from scipy.spatial.transform import Rotation

    eef = observation["robot_state"]["eef"]
    action = np.zeros(7, np.float64)
    action[:3] = np.asarray(eef["pos"], np.float64)
    action[3:6] = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
    action[6] = gripper
    return action.astype(np.float32)


# ---- the scene -------------------------------------------------------------------------------


def make_env(bench_index: int, image_size: int) -> Any:
    """A `LiberoSimEnv` of a goal-suite task, also for the two tasks with a unary goal.

    `LiberoSimEnv` builds its scene index with `build_index`, which unpacks the task's goal
    into three names and so refuses `Turnon flat_stove_1` and `Open
    wooden_cabinet_1_middle_region`. The index does not depend on the goal, so for those it
    is built with the one name repeated (lane S's runner does the same); the adapter's own
    goal is then a placeholder the jobs runtime never reads (`SceneGoals` evaluates the
    requested goals).
    """
    from fleetlab.sim.libero_sim import LiberoSimEnv, build_index

    class JobSimEnv(LiberoSimEnv):
        def _index(self) -> Any:
            if self._scene is None:
                rs = self._rs
                goal = rs.parsed_problem["goal_state"]
                if len(goal) == 1 and len(goal[0]) == 2:
                    rs.parsed_problem["goal_state"] = [[goal[0][0], goal[0][1], goal[0][1]]]
                    try:
                        self._scene = build_index(rs)
                    finally:
                        rs.parsed_problem["goal_state"] = goal
                else:
                    self._scene = build_index(rs)
            return self._scene

    return JobSimEnv(GOAL_SUITE, bench_index, image_size)


def episode_identity(
    start: StartIdentity,
    episode_id: str,
    spec: PolicySpec,
    soft: SoftwareVersion,
    bench_index: int,
    image_size: int,
    max_steps: int,
) -> EpisodeIdentity:
    """The engine's identity of one job episode: the start, the policy, the software."""
    return EpisodeIdentity(
        world_id=neutral_id(IdKind.world, 1),
        episode_id=episode_id,
        robot_id=neutral_id(IdKind.robot, 1),
        site_id=neutral_id(IdKind.site, 1),
        day=0,
        episode_index=start.init_state,
        suite=GOAL_SUITE,
        task_index=bench_index,
        start_state=start.init_state,
        scene_seed=start.scene_seed,
        noise_seed=start.noise_seed,
        policy=spec,
        batch_size=1,
        row=None,
        image_size=image_size,
        max_steps=max_steps,
        faults=[],
        software=soft,
    )


def _upright(observation: Observation, camera: Camera) -> npt.NDArray[np.uint8]:
    return np.ascontiguousarray(observation["pixels"][PIXEL_KEYS[camera]][::-1, ::-1])


# ---- one episode -----------------------------------------------------------------------------


@dataclass
class Played:
    """One program played from one start: the record and what the tables and replay need."""

    episode: JobEpisode
    identity: EpisodeIdentity
    recorder: tables.EpisodeRecorder
    executed: list[npt.NDArray[np.float32]]
    hashes: list[str]
    frames: dict[Camera, list[npt.NDArray[np.uint8]]]
    tail_instruction: str | None
    order_detail: str
    stats: dict[str, Any]


class _Episode:
    """The loop of one episode on a built environment."""

    obs: Observation
    goals: SceneGoals
    recorder: tables.EpisodeRecorder
    priv: Any  # the adapter's PrivilegedState at the latest step
    holds: tuple[bool, ...]
    wall_s: float
    start_pos: npt.NDArray[np.float64]  # the hand's position after the reset
    start_rotation: npt.NDArray[np.float64]  # its orientation, as the controller's rotation vector
    start_quat: npt.NDArray[np.float64]  # its orientation as the robot reports it

    def __init__(
        self,
        env: Any,
        source: ChunkSource,
        identity: EpisodeIdentity,
        state: RequestState,
        record_video: bool,
        frame_every: int,
    ) -> None:
        self.env, self.source, self.identity, self.state = env, source, identity, state
        self.record_video, self.frame_every = record_video, frame_every
        self.t = 0
        self.chunk_index = 0
        self.queue: deque[npt.NDArray[np.float32]] = deque()
        self.last_cmd: npt.NDArray[np.float32] | None = None
        self.executed: list[npt.NDArray[np.float32]] = []
        self.hashes: list[str] = []
        self.frames: dict[Camera, list[npt.NDArray[np.uint8]]] = {c: [] for c in Camera}
        self.goal_rows: list[tuple[bool, ...]] = []
        self.sim_s = 0.0

    def _observe(self) -> None:
        priv = self.env.privileged_state()
        self.priv = priv
        self.holds = self.goals.holds()
        self.goal_rows.append(self.holds)
        frame_index = None
        if self.record_video and self.t % self.frame_every == 0:
            for camera in Camera:
                self.frames[camera].append(_upright(self.obs, camera))
            frame_index = len(self.frames[Camera.scene]) - 1
        eef = self.obs["robot_state"]["eef"]
        self.recorder.observe(
            self.t,
            eef_pos=eef["pos"],
            eef_quat=eef["quat"],
            gripper_opening=priv.gripper_opening,
            poses=priv.object_poses,
            velocities=priv.object_velocities,
            contacts=priv.contacts,
            goals_hold=self.holds,
            frame_index=frame_index,
        )
        self.hashes.append(self.env.state_hash().hex())

    def _do(
        self,
        action: npt.ArrayLike,
        *,
        instruction: str | None,
        program_step: int | None,
        phase: str,
    ) -> None:
        cmd = np.asarray(action, np.float32).reshape(7)
        self.recorder.command(cmd, instruction=instruction, program_step=program_step, phase=phase)
        self.executed.append(cmd.copy())
        t0 = time.perf_counter()
        result = self.env.step(cmd)
        self.sim_s += time.perf_counter() - t0
        self.obs = result.observation
        self.last_cmd = cmd
        self.t += 1
        self._observe()

    def _policy_action(self, instruction: str) -> npt.NDArray[np.float32]:
        if not self.queue:
            actions = self.source.chunk(self.obs, instruction, self.identity, self.chunk_index)
            self.chunk_index += 1
            # A source may return its full prediction. Keep only this episode's executed
            # prefix; the same queue continues into the tail and is cleared at step changes.
            horizon = self.identity.policy.actions_executed_per_chunk
            self.queue = deque(np.asarray(a, np.float32) for a in actions[:horizon])
        return self.queue.popleft()

    def delivery(self) -> DeliveryVerdict:
        priv = self.priv
        command = None if self.last_cmd is None else float(self.last_cmd[6])
        scene = DeliveryState(
            goals_hold=self.holds,
            contacts=tuple(priv.contacts),
            velocities=priv.object_velocities,
            gripper_opening=priv.gripper_opening,
            gripper_command=command,
        )
        return delivery_verdict(scene, self.state.requested_goals, self.goals.named)

    def play(
        self, program: JobProgram, job_id: str
    ) -> tuple[JobEpisode, tables.EpisodeRecorder, str | None, str]:
        env = self.env
        t_start = time.perf_counter()
        self.obs = env.reset(self.identity, [])
        from scipy.spatial.transform import Rotation

        eef = self.obs["robot_state"]["eef"]
        self.start_pos = np.asarray(eef["pos"], np.float64).copy()
        self.start_quat = np.asarray(eef["quat"], np.float64).copy()
        matrix = np.asarray(eef["mat"], np.float64)
        self.start_rotation = Rotation.from_matrix(matrix).as_rotvec()
        self.goals = SceneGoals(env._rs, self.state.requested_goals)
        free = sorted(env._index().object_dofs)
        self.recorder = tables.EpisodeRecorder(
            self.identity.episode_id, free, len(self.state.requested_goals)
        )
        self._observe()
        results: list[ProgramStepResult] = []
        stopped = False
        for i, step in enumerate(program.steps):
            if stopped:
                results.append(
                    ProgramStepResult(
                        step=i,
                        kind=step.kind,
                        outcome=StepOutcome.skipped,
                        steps_used=0,
                        start_control_step=None,
                        end_control_step=None,
                    )
                )
                continue
            self.queue.clear()  # the chunk of an earlier step is dropped at every boundary
            begin = self.t
            outcome = self._run_step(i, step)
            if outcome is StepOutcome.timeout and step.on_timeout is not OnTimeout.continue_:
                stopped = True
            results.append(
                ProgramStepResult(
                    step=i,
                    kind=step.kind,
                    outcome=outcome,
                    steps_used=self.t - begin,
                    start_control_step=begin,
                    end_control_step=self.t,
                )
            )
        tail = 0
        last = program.steps[-1]
        tail_instruction = None
        if last.kind is JobStepKind.policy and results[-1].outcome is StepOutcome.reached:
            assert last.instruction is not None
            tail_instruction = last.instruction
            while tail < DELIVERY.tail_steps and not self.delivery().delivered:
                self._do(
                    self._policy_action(last.instruction),
                    instruction=last.instruction,
                    program_step=None,
                    phase="tail",
                )
                tail += 1
        verdict = self.delivery()
        self.recorder.finish()
        if self.record_video and self.frames[Camera.scene] and self.t % self.frame_every != 0:
            for camera in Camera:
                self.frames[camera].append(_upright(self.obs, camera))
            self.recorder.robot[-1]["frame_index"] = len(self.frames[Camera.scene]) - 1
        held = np.asarray(self.goal_rows, bool).reshape(len(self.goal_rows), -1)
        first: list[int | None] = []
        for g in range(held.shape[1]):
            hits = np.flatnonzero(held[:, g])
            first.append(int(hits[0]) if len(hits) else None)
        verdicts = order_verdicts(list(self.state.order_rules), first)
        episode = JobEpisode(
            job_id=job_id,
            episode_id=self.identity.episode_id,
            start=_start_of(self.identity, self.state.base_task),
            program_digest=program.digest,
            delivered=verdict.delivered,
            goals=[
                GoalTrace(goal=g, first_held_step=first[g], held_at_end=bool(held[-1, g]))
                for g in range(held.shape[1])
            ],
            program_steps=results,
            tail_steps=tail,
            control_steps=self.t,
            order_verdicts=verdicts,
            order_kept=all(v.kept for v in verdicts),
            delivery=verdict,
            end_state_hash=self.hashes[-1],
            start_eef_pos=[float(x) for x in self.start_pos],
            start_eef_quat=[float(x) for x in self.start_quat],
        )
        self.wall_s = time.perf_counter() - t_start
        detail = "; ".join(
            f"rule {v.rule}: {'kept' if v.kept else 'broken'}, {v.detail}" for v in verdicts
        )
        return episode, self.recorder, tail_instruction, detail or "no order rules"

    def _run_step(self, i: int, step: JobStep) -> StepOutcome:
        if step.kind is JobStepKind.policy:
            assert step.until is not None and step.max_steps is not None
            assert step.instruction is not None
            n = 0
            while True:
                if step.until.goal is not None and self.holds[step.until.goal]:
                    return StepOutcome.reached
                if step.until.steps is not None and n >= step.until.steps:
                    return StepOutcome.reached
                if n >= step.max_steps:
                    return StepOutcome.timeout
                self._do(
                    self._policy_action(step.instruction),
                    instruction=step.instruction,
                    program_step=i,
                    phase="program",
                )
                n += 1
        if step.kind is JobStepKind.motion:
            from scipy.spatial.transform import Rotation

            assert step.segment is not None
            trajectory = program_trajectory([step.segment])
            eef = self.obs["robot_state"]["eef"]
            start = np.asarray(eef["pos"], np.float64).copy()
            rotation = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
            gripper = -1.0 if self.last_cmd is None else float(self.last_cmd[6])
            sets_gripper = step.segment.primitive in (
                ProbePrimitive.open_gripper,
                ProbePrimitive.close_gripper,
            )
            for k in range(len(trajectory.segment)):
                action = np.zeros(7, np.float64)
                action[:3] = start + trajectory.offset[k]
                action[3:6] = rotation
                action[6] = float(trajectory.gripper[k]) if sets_gripper else gripper
                self._do(action, instruction=None, program_step=i, phase="program")
            return StepOutcome.reached
        if step.kind is JobStepKind.home:
            assert step.duration_steps is not None
            return self._home(i, step.duration_steps)
        assert step.steps is not None
        from fleetlab.contracts.live import hold_action

        for _ in range(step.steps):
            if self.last_cmd is None:
                hold = hold_here(self.obs)
            else:
                hold = hold_action(ControlMode.absolute, self.last_cmd)
            self._do(hold, instruction=None, program_step=i, phase="program")
        return StepOutcome.reached

    def _home(self, i: int, budget: int) -> StepOutcome:
        """Back to the start pose at the probes' speed bound, gripper open, then settle."""
        import math

        from scipy.spatial.transform import Rotation, Slerp

        eef = self.obs["robot_state"]["eef"]
        origin = np.asarray(eef["pos"], np.float64).copy()
        turn = Slerp(
            [0.0, 1.0],
            Rotation.concatenate(
                [
                    Rotation.from_matrix(np.asarray(eef["mat"], np.float64)),
                    Rotation.from_rotvec(self.start_rotation),
                ]
            ),
        )
        per_step = HOME.max_speed_m_s / CONTROL_HZ
        ramp = max(1, math.ceil(float(np.linalg.norm(self.start_pos - origin)) / per_step - 1e-9))
        previous = origin
        for n in range(budget):
            action = np.zeros(7, np.float64)
            if n < ramp:
                fraction = (n + 1) / ramp
                action[:3] = origin + (self.start_pos - origin) * fraction
                action[3:6] = turn([fraction]).as_rotvec()[0]
            else:
                action[:3] = self.start_pos
                action[3:6] = self.start_rotation
            action[6] = -1.0
            self._do(action, instruction=None, program_step=i, phase="program")
            position = np.asarray(self.obs["robot_state"]["eef"]["pos"], np.float64)
            speed = float(np.linalg.norm(position - previous)) * CONTROL_HZ
            previous = position
            near = float(np.linalg.norm(position - self.start_pos)) < HOME.reach_m
            if n + 1 >= ramp and near and speed < HOME.rest_m_s:
                return StepOutcome.reached
        return StepOutcome.timeout


def _start_of(identity: EpisodeIdentity, base_task: int) -> StartIdentity:
    return StartIdentity(
        base_task=base_task,
        init_state=identity.start_state,
        scene_seed=identity.scene_seed,
        noise_seed=identity.noise_seed,
    )


# ---- the runner ------------------------------------------------------------------------------


@dataclass
class _RunTables:
    episodes: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class JobRunner:
    """Programs played from starts on one scene at a time, written to a run directory."""

    def __init__(
        self,
        source: ChunkSource,
        software: SoftwareVersion,
        *,
        image_size: int = IMAGE_SIZE,
        frame_every: int = FRAME_EVERY,
        suite: GoalSuite | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        if frame_every < 1 or CONTROL_HZ % frame_every:
            raise ValueError(f"frame_every divides {CONTROL_HZ}, not {frame_every}")
        self.source, self.software = source, software
        self.image_size, self.frame_every = image_size, frame_every
        self.suite = suite or GoalSuite()
        self.log = log
        self._env: Any = None
        self._env_bench: int | None = None
        self._tables: dict[Path, _RunTables] = {}

    def env(self, base_task: int) -> Any:
        bench = self.suite.bench[base_task]
        if self._env is None or self._env_bench != bench:
            self.close()
            self._env = make_env(bench, self.image_size)
            self._env_bench = bench
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_bench = None

    def play(
        self,
        program: JobProgram,
        state: RequestState,
        start: StartIdentity,
        job_id: str,
        episode_id: str,
        *,
        record_video: bool,
        allow_release: bool = False,
    ) -> Played:
        """One program from one start, not yet written anywhere."""
        if start.is_release and not allow_release:
            raise ValueError(f"initial state {start.init_state} is a release start")
        if start.base_task != state.base_task:
            raise ValueError(f"start of task {start.base_task}; the request's is {state.base_task}")
        problems = program_problems(program, state)
        if problems:
            raise ValueError("; ".join(problems))
        env = self.env(start.base_task)
        budget = program.max_control_steps + DELIVERY.tail_steps
        identity = episode_identity(
            start,
            episode_id,
            self.source.spec.model_copy(
                update={"actions_executed_per_chunk": program.actions_per_chunk or 30}
            ),
            self.software,
            self.suite.bench[start.base_task],
            self.image_size,
            budget,
        )
        calls0, policy0 = getattr(self.source, "calls", 0), getattr(self.source, "seconds", 0.0)
        run = _Episode(env, self.source, identity, state, record_video, self.frame_every)
        episode, recorder, tail_instruction, detail = run.play(program, job_id)
        stats = {
            "wall_s": round(run.wall_s, 3),
            "sim_s": round(run.sim_s, 3),
            "policy_calls": getattr(self.source, "calls", 0) - calls0,
            "policy_s": round(getattr(self.source, "seconds", 0.0) - policy0, 3),
            "load_average": round(os.getloadavg()[0], 2),
        }
        return Played(
            episode=episode,
            identity=identity,
            recorder=recorder,
            executed=run.executed,
            hashes=run.hashes,
            frames=run.frames,
            tail_instruction=tail_instruction,
            order_detail=detail,
            stats=stats,
        )

    def write(self, root: Path, played: Played, program: JobProgram) -> None:
        """An episode's files, then its records, then the run-level tables rewritten."""
        from fleetlab.video import write_mp4

        episode_id = played.episode.episode_id
        tables.write_episode_tables(root, episode_id, played.recorder.tables())
        videos: dict[Camera, str | None] = {c: None for c in Camera}
        if played.frames[Camera.scene]:
            from fleetlab.log._files import staged

            for camera in Camera:
                relative = tables.video_relative(episode_id, camera)
                with staged(root / "log" / relative, root) as tmp:
                    write_mp4(
                        tmp,
                        played.frames[camera],
                        CONTROL_HZ // self.frame_every,
                        crf=23,
                        preset="veryfast",
                        repeatable=True,
                    )
                videos[camera] = relative
        tables.save_replay_record(
            root,
            episode_id,
            np.asarray(played.executed, np.float32).reshape(-1, 7),
            played.hashes,
            played.identity.model_dump_json(),
        )
        tables.append_episode(root, played.episode)
        store = self._tables.setdefault(root, _load_tables(root))
        store.episodes.append(tables.episode_row(played.episode, videos, played.order_detail))
        store.events.extend(
            tables.step_event_rows(played.episode, program, played.tail_instruction)
        )
        tables.write_run_tables(root, store.episodes, store.events)

    def run_starts(
        self,
        root: Path,
        job_id: str,
        program: JobProgram,
        state: RequestState,
        starts: Sequence[StartIdentity],
        *,
        record_video: bool,
        first_episode: int,
        allow_release: bool = False,
        resume: bool = False,
        record: dict[str, Any] | None = None,
    ) -> list[JobEpisode]:
        """A program on several starts, each written as it finishes; returns the episodes.

        The run directory's record (`run.json`, hidden) holds the request state, the
        programs and the measured timings; a run directory serves one request state. Its
        top-level policy is the source's default spec; each episode records its effective
        policy, including the program's executed horizon, beside its timings and in its
        replay identity. An episode the run directory holds already is refused, unless
        `resume`, and then kept
        only when it is this job's run of this program from this start with every file
        (`resume_problems`).
        """
        out = []
        existing = {e.episode_id: e for e in tables.read_episodes(root)}
        if record is None:
            path = root / tables.RUN_RECORD
            record = json.loads(path.read_text()) if path.is_file() else {"format": RUN_FORMAT}
        dumped = state.model_dump(mode="json")
        if record.setdefault("state", dumped) != dumped:
            raise ValueError(f"{root} holds runs of another request state")
        record.setdefault("software", self.software.model_dump(mode="json"))
        record.setdefault("policy", self.source.spec.model_dump(mode="json"))
        record.setdefault("image_size", self.image_size)
        record.setdefault("frame_every", self.frame_every)
        record.setdefault("tail_steps", DELIVERY.tail_steps)
        problems = resume_problems(
            root,
            record,
            job_id,
            program,
            starts,
            first_episode=first_episode,
            record_video=record_video,
        )
        if problems:
            raise ValueError(f"{job_id} cannot continue in {root}: " + "; ".join(problems))
        jobs = record.setdefault("jobs", [])
        if not any(j.get("job_id") == job_id for j in jobs):
            jobs.append(
                {
                    "job_id": job_id,
                    "program_digest": program.digest,
                    "program": program.model_dump(mode="json"),
                    "first_episode": first_episode,
                    "starts": [s.model_dump(mode="json") for s in starts],
                }
            )
        tables.write_json(root, tables.RUN_RECORD, record)
        for k, start in enumerate(starts):
            episode_id = neutral_id(IdKind.episode, first_episode + k)
            if episode_id in existing:
                if not resume:
                    raise ValueError(f"{root} holds {episode_id} already")
                out.append(existing[episode_id])  # checked by resume_problems above
                continue
            played = self.play(
                program,
                state,
                start,
                job_id,
                episode_id,
                record_video=record_video,
                allow_release=allow_release,
            )
            self.write(root, played, program)
            record.setdefault("episodes", {})[episode_id] = {
                "job_id": job_id,
                "init_state": start.init_state,
                "policy": played.identity.policy.model_dump(mode="json"),
                **played.stats,
            }
            tables.write_json(root, tables.RUN_RECORD, record)
            e = played.episode
            ends = [
                "-" if r.end_control_step is None else r.end_control_step for r in e.program_steps
            ]
            steps = " ".join(
                f"{r.outcome.value[:3]}@{end}" for r, end in zip(e.program_steps, ends, strict=True)
            )
            verdict = "delivered" if e.delivered else "not delivered"
            self.log(
                f"{job_id} {episode_id} start {start.init_state:02d}: {verdict:<13} {steps} "
                f"tail={e.tail_steps} t={e.control_steps} "
                f"order={'kept' if e.order_kept else 'BROKEN'} "
                f"wall={played.stats['wall_s']:.1f}s calls={played.stats['policy_calls']}"
                + ("" if e.delivered else f" ({e.delivery.reason})")
            )
            out.append(e)
        return out


def resume_problems(
    root: Path,
    record: dict[str, Any],
    job_id: str,
    program: JobProgram,
    starts: Sequence[StartIdentity],
    *,
    first_episode: int,
    record_video: bool,
) -> list[str]:
    """Why a job cannot continue in a run directory: the program, starts or numbering it ran
    with there, or an episode of it that is not its run from that start with every file."""
    problems = []
    entry = next((j for j in record.get("jobs", []) if j.get("job_id") == job_id), None)
    if entry is not None:
        if entry.get("program_digest") != program.digest:
            problems.append(f"{job_id} ran another program here")
        if entry.get("first_episode") != first_episode:
            problems.append(f"{job_id}'s episodes are numbered from {entry.get('first_episode')}")
        if entry.get("starts") != [s.model_dump(mode="json") for s in starts]:
            problems.append(f"{job_id} ran from other starts here")
    existing = {e.episode_id: e for e in tables.read_episodes(root)}
    listed = tables.recorded_episode_ids(root)
    for k, start in enumerate(starts):
        episode_id = neutral_id(IdKind.episode, first_episode + k)
        episode = existing.get(episode_id)
        if episode is not None:
            problems += tables.episode_problems(
                root,
                episode,
                job_id=job_id,
                program_digest=program.digest,
                start=start,
                record_video=record_video,
                in_table=episode_id in listed,
            )
    return problems


def _load_tables(root: Path) -> _RunTables:
    """The run-level rows already written (a resumed run), read back from their files."""
    import pyarrow.parquet as pq

    store = _RunTables()
    for name, rows in (("job_episodes", store.episodes), ("job_step_events", store.events)):
        path = tables.table_path(root, name)
        if path.is_file():
            rows.extend(pq.read_table(path).to_pylist())
    return store


# ---- the batch -------------------------------------------------------------------------------


def _job_number(job_id: str) -> int:
    import re

    if not re.fullmatch(JOB_ID_PATTERN, job_id):
        raise ValueError(f"{job_id!r} is not a job id ({JOB_ID_PATTERN})")
    return int(job_id[1:])


def _problems(error: Exception) -> list[str]:
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        return [f"{'.'.join(map(str, e['loc'])) or 'request'}: {e['msg']}" for e in error.errors()]
    return [str(error)]


def run_batch(
    batch: dict[str, Any], out: Path, runner: JobRunner, *, resume: bool = False
) -> dict[str, Any]:
    """One session's queue: every job of the batch, each refused, done or failed."""
    if batch.get("format") != BATCH_FORMAT:
        raise ValueError(f"not a job batch: format {batch.get('format')!r}, want {BATCH_FORMAT}")
    state = RequestState.model_validate(batch["state"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "log").mkdir(exist_ok=True)
    before = tables.read_episodes(out)
    if not resume and before:
        raise ValueError(f"{out} holds a run already; pass --resume to continue it")
    record: dict[str, Any] = {
        "format": RUN_FORMAT,
        "state": state.model_dump(mode="json"),
        "software": runner.software.model_dump(mode="json"),
        "policy": runner.source.spec.model_dump(mode="json"),
        "image_size": runner.image_size,
        "frame_every": runner.frame_every,
        "tail_steps": DELIVERY.tail_steps,
        "jobs": [],
    }
    old = out / tables.RUN_RECORD
    if resume and old.is_file():
        # The run being continued keeps its record; the batch must be the same run.
        kept = json.loads(old.read_text())
        changed = [k for k in record if k != "jobs" and kept.get(k) != record[k]]
        if changed:
            raise ValueError(f"cannot resume {out}: the batch's {', '.join(changed)} differ")
        record = kept
    elif before:
        raise ValueError(f"cannot resume {out}: it holds episodes but no run record")
    result: dict[str, Any] = {"jobs": []}
    for entry in batch["jobs"]:
        job_id = str(entry.get("job_id", ""))
        status: dict[str, Any] = {
            "job_id": job_id,
            "status": "refused",
            "episode_ids": [],
            "program_digest": None,
            "problems": [],
        }
        result["jobs"].append(status)
        try:
            number = _job_number(job_id)
            request = JobRequest.model_validate(entry["request"])
            status["program_digest"] = request.program.digest
            problems = program_problems(request.program, state)
            problems += [
                f"start {s.init_state} is of task {s.base_task}; the request's is {state.base_task}"
                for s in request.starts
                if s.base_task != state.base_task
            ]
            saved = record.get("requests", {}).get(job_id)
            if saved is not None and saved != request.model_dump(mode="json"):
                problems.append(f"{job_id} was run here with another request; it is not resumed")
            if not problems:
                problems = resume_problems(
                    out,
                    record,
                    job_id,
                    request.program,
                    request.starts,
                    first_episode=int(entry.get("first_episode") or number * 100 + 1),
                    record_video=request.record_video,
                )
            if problems:
                raise ValueError("; ".join(problems))
        except Exception as error:
            status["problems"] = _problems(error)
            tables.write_json(out, tables.BATCH_RESULT, result)
            runner.log(f"{job_id}: refused: {'; '.join(status['problems'])}")
            continue
        first = int(entry.get("first_episode") or number * 100 + 1)
        status["episode_ids"] = [
            neutral_id(IdKind.episode, first + k) for k in range(len(request.starts))
        ]
        record.setdefault("requests", {})[job_id] = request.model_dump(mode="json")
        try:
            runner.run_starts(
                out,
                job_id,
                request.program,
                state,
                request.starts,
                record_video=request.record_video,
                first_episode=first,
                resume=resume,
                record=record,
            )
            status["status"] = "done"
        except Exception as error:
            status["status"] = "failed"
            status["problems"] = [
                f"{type(error).__name__}: {error}",
                traceback.format_exc()[-2000:],
            ]
            runner.log(f"{job_id}: failed: {error}")
            runner.close()  # a fresh scene for the next job
        tables.write_json(out, tables.RUN_RECORD, record)
        tables.write_json(out, tables.BATCH_RESULT, result)
    tables.write_json(out, tables.BATCH_RESULT, result)
    return result


def make_source(name: str) -> tuple[ChunkSource, SoftwareVersion]:
    if name == "xvla":
        return load_xvla()
    if name == "hold":
        from fleetlab.engine import software

        return HoldSource(), software.current("none")
    raise ValueError(f"unknown policy {name!r}: xvla or hold")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetlab.jobs.runner", description="Run one session's queue of jobs."
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", default="xvla", choices=("xvla", "hold"))
    parser.add_argument("--frame-every", type=int, default=FRAME_EVERY)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    batch = json.loads(args.batch.read_text())
    t0 = time.perf_counter()
    source, soft = make_source(args.policy)
    print(
        f"runner: policy {source.spec.name} ready in {time.perf_counter() - t0:.1f} s", flush=True
    )
    runner = JobRunner(
        source,
        soft,
        image_size=args.image_size,
        frame_every=args.frame_every,
        log=lambda line: print(line, flush=True),
    )
    try:
        result = run_batch(batch, args.out, runner, resume=args.resume)
    finally:
        runner.close()
    counts = {
        s: sum(j["status"] == s for j in result["jobs"]) for s in ("done", "refused", "failed")
    }
    print(f"runner: {counts} in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
