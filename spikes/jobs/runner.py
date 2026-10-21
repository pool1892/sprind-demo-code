"""Lane S spike: chain goal-suite skills into one job in one continuous simulator state.

A thin standalone runner (lane S of the two-day demo). One
`LiberoSimEnv` built from the job's base task (its first step's task) and
reset once per start; one `XVLAPolicy` loaded once in `fp16_res32`; each
step's goal evaluated with LIBERO's own predicate functions on the loaded
scene; the remaining chunk dropped at every step boundary; every control step
logged; a scene video; the host-fixed delivery check; a failure tag per run.

Switch variants (how a step hands over to the next):

- `naive`: switch the instruction the moment the step's native LIBERO
  predicate holds, drop the stale chunk, no release, no settle.
- `placed`: switch when the fleet's placed criterion holds for the step
  (native, no robot body touching the step's object, the object at rest),
  or 60 steps after native first held (the placed check's tail), whichever
  comes first. The policy keeps acting under the step's instruction meanwhile.
- `release`: switch on native, then a host-written transition from the probe
  vocabulary (`contracts.probes`): open the gripper and hold 15 steps, ramp
  5 cm up in 20 steps, hold 15 steps; then the next instruction.
- `home`: switch on native, open the gripper and hold 10 steps, then move the
  hand back to where it was at the job's start in ramps of at most 5 cm on one
  axis each (z first, then x, then y; 5 steps per ramp, 0.2 m/s), orientation
  held, and hold 10 steps. Each ramp is a valid one-segment probe; the whole
  move is not one probe (it leaves the 5 cm box), so it is in the vocabulary
  only if consecutive motion steps re-anchor at their own start.
- `extend30` (`extendN`): switch on native, drop the chunk, then a second policy step
  under the same instruction for exactly N control steps (fresh chunks), then the next
  instruction: what the plan's vocabulary expresses today (`until: steps N`).
- `retry`: switch as `placed`; a later step gets two attempts of 150 control steps,
  the second after the `release` transition (open, 5 cm up, settle), with a fresh
  chunk; expressible as a program only if a step's timeout does not end the job.
- `teleport` (host diagnostic, outside the vocabulary): switch on native, put
  the arm's joints back to their configuration at the job's start, hold the
  start pose with the gripper open for 10 steps, then the next instruction.

Every job ends the same way in every variant: once the last step's native
predicate holds, the policy keeps acting under the last instruction until the
delivery check holds or 60 steps have passed (the placed check's tail), within
the job budget. So differences between variants come from the switches only.

Heavy (loads the model):

    FLEETLAB_MIOPEN_DIR=$HOME/.cache/robots-probes/engine/miopen TMPDIR=/var/tmp/robots-spike \\
        tools/heavy-run uv run --locked python spikes/jobs/runner.py --batch B.json --out DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import fleetlab  # noqa: F401  first: torch before mujoco

SUITE = "libero_goal"
REPO = Path(__file__).resolve().parents[2]
BDDL_DIR = REPO / "third_party/libero/libero/libero/bddl_files/libero_goal"
IMAGE_SIZE = 360
PRECISION = "fp16_res32"
WORLD_SEED = 20261020  # scene and noise seeds come from it and the start index
STEP_MAX = 300  # control steps per policy step
JOB_MAX = 700  # control steps per job, transitions and the end tail included
TAIL = 60  # the placed check's tail (PLACED_THRESHOLDS["tail_steps"])
DEV_STARTS = range(0, 30)  # development init-state indices; 30 to 49 stay for release runs
OPEN_MIN_M = 0.06  # gripper open: fingers at least 0.06 m apart (widest ~0.08 m)
MOVED_M = 0.03  # a target counts as moved when it is 3 cm from where it was

# The plan's combinations, in tasks_info.txt numbering.
JOBS: dict[str, tuple[int, ...]] = {
    "J1": (6, 4),
    "J2": (6, 9),
    "J3": (4, 6),
    "J4": (8, 4),
    "J5": (4, 8),
    "J6": (6, 1),
    "J7": (2, 9),
    "J8": (7, 4),
    "J9": (6, 3),
    "J10": (6, 4, 8),
    # spike-only pairs that share no object, to look for combinations that just work
    "K1": (5, 4),  # wine bottle on the rack, then bowl on plate
    "K2": (4, 5),  # bowl on plate, then wine bottle on the rack
    "K3": (6, 5),  # cheese in bowl, then wine bottle on the rack
    "K4": (5, 6),  # wine bottle on the rack, then cheese in bowl
    "K5": (2, 7),  # turn on the stove, then open the middle drawer
    "K6": (0, 4),  # wine bottle on top of the cabinet, then bowl on plate
    "K7": (3, 6),  # bowl on top of the cabinet, then cheese in bowl: J9 reversed
}
VARIANTS = ("naive", "placed", "release", "home", "teleport", "retry", "extend30")


def plan_tasks() -> list[str]:
    """tasks_info.txt order (the plan's task numbers) as BDDL stems."""
    lines = (BDDL_DIR / "tasks_info.txt").read_text().split()
    return [Path(line).stem for line in lines if line.endswith(".bddl")]


def job_steps(name: str) -> tuple[int, ...]:
    """A job's steps in plan numbering; `S<n>` is the single skill n."""
    if name.startswith("S"):
        return (int(name[1:]),)
    return JOBS[name]


@dataclass(frozen=True)
class Goal:
    """One step's goal: the task, its instruction, its BDDL goal and the bodies it names."""

    number: int  # tasks_info.txt index (the plan's numbering)
    stem: str
    bench: int  # LIBERO's benchmark index (what LiberoEnv takes)
    instruction: str  # LIBERO's language for the task: what the engine passes as task_description
    predicates: tuple[tuple[str, ...], ...]  # parsed goal_state, as LIBERO evaluates it
    mover: str  # the body the step moves (the first argument's owner)
    target: str | None  # the second argument's owner, if any
    named: tuple[str, ...]  # bodies the goal names (objects, region owners), never the table


class Catalogue:
    """The goal suite's tasks: stems, benchmark indices, instructions, parsed goals."""

    def __init__(self) -> None:
        from fleetlab.runtime import configure_process

        configure_process()
        from libero.libero import benchmark
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        tasks = benchmark.get_benchmark_dict()[SUITE]()
        self.bench = {tasks.get_task(i).name: i for i in range(tasks.n_tasks)}
        self.language = {tasks.get_task(i).name: tasks.get_task(i).language for i in range(10)}
        self.stems = plan_tasks()
        self.parsed = {
            stem: robosuite_parse_problem(str(BDDL_DIR / f"{stem}.bddl")) for stem in self.stems
        }

    def goal(self, number: int, rs: Any) -> Goal:
        stem = self.stems[number]
        preds = tuple(tuple(p) for p in self.parsed[stem]["goal_state"])
        sites = getattr(rs, "object_sites_dict", {})

        def owner(name: str) -> str:
            if name not in sites:
                return name
            parent = sites[name].parent_name
            return "main_table" if parent is None else str(parent)  # table regions: no parent

        for p in preds:  # the shared scene must hold every name this goal uses
            for arg in p[1:]:
                if arg not in rs.object_states_dict:
                    raise KeyError(f"{stem}: {arg} is not in the loaded scene")
        first = preds[0]
        mover = owner(first[1])
        target = owner(first[2]) if len(first) == 3 else None
        named = []
        for p in preds:
            for arg in p[1:]:
                body = owner(arg)
                if body != "main_table" and body not in named:
                    named.append(body)
        return Goal(
            number=number,
            stem=stem,
            bench=self.bench[stem],
            instruction=self.language[stem],
            predicates=preds,
            mover=mover,
            target=target,
            named=tuple(named),
        )


def holds(rs: Any, predicates: tuple[tuple[str, ...], ...]) -> bool:
    """LIBERO's own evaluation of a goal on the loaded scene (unary and binary predicates)."""
    from libero.libero.envs.predicates import eval_predicate_fn

    result = True
    for name, *args in predicates:
        value = eval_predicate_fn(name, *[rs.object_states_dict[a] for a in args])
        result = bool(value) and result
    return result


def make_env(bench: int) -> Any:
    """A `LiberoSimEnv` for a goal-suite task, unary goals included.

    `build_index` unpacks three elements of the task's single goal predicate, so the
    adapter cannot be built for the two tasks with a unary goal (`Turnon flat_stove_1`,
    `Open wooden_cabinet_1_middle_region`). The scene index does not depend on the goal,
    so for those the index is built with the argument repeated; the env's own goal
    (`env.goal`) is then a placeholder that this runner never reads.
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

    return JobSimEnv(SUITE, bench, IMAGE_SIZE)


def start_identity(start: int, bench: int, spec: Any, soft: Any) -> Any:
    """The start's identity: base task, init-state index `start`, seeds from the start index."""
    from fleetlab.engine import manifests

    if start not in DEV_STARTS:
        raise ValueError(f"start {start} is outside the development starts 0 to 29")
    return manifests.identity(
        world_seed=WORLD_SEED,
        episode_number=start + 1,
        robot=1,
        site=1,
        day=0,
        episode_index=start,
        suite=SUITE,
        task_index=bench,
        start_state=start,
        policy=spec,
        batch_size=1,
        software=soft,
        image_size=IMAGE_SIZE,
        max_steps=JOB_MAX,
    )


def release_program() -> Any:
    from fleetlab.contracts.probes import ProbeProgram, ProbeSegment

    return ProbeProgram(
        segments=[
            ProbeSegment(
                primitive="open_gripper",
                axis=None,
                amplitude_m=None,
                duration_steps=15,
                period_steps=None,
            ),
            ProbeSegment(
                primitive="ramp", axis="z", amplitude_m=0.05, duration_steps=20, period_steps=None
            ),
            ProbeSegment(
                primitive="hold", axis=None, amplitude_m=None, duration_steps=15, period_steps=None
            ),
        ]
    )


class Video:
    """Scene and wrist views side by side, streamed to an H.264 MP4 at 20 fps."""

    def __init__(self, path: Path) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._w = imageio.get_writer(
            str(path),
            fps=20,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            ffmpeg_params=["-crf", "26", "-preset", "veryfast"],
        )

    def add(self, obs: Any) -> None:
        from fleetlab.sim.libero_env import camera_frame

        self._w.append_data(camera_frame(obs))

    def close(self) -> None:
        self._w.close()


class Run:
    """One job on one start in one variant: the loop, the trace, the verdict."""

    def __init__(
        self,
        env: Any,
        policy: Any,
        soft: Any,
        goals: list[Goal],
        job: str,
        variant: str,
        start: int,
        out: Path | None,
    ) -> None:
        self.env, self.policy, self.goals = env, policy, goals
        self.job, self.variant, self.start, self.out = job, variant, start, out
        self.identity = start_identity(start, goals[0].bench, policy.spec, soft)
        self.rs = env._rs
        self.t = 0
        self.chunk = 0
        self.calls: list[float] = []
        self.sim_s = 0.0
        self.last_cmd: np.ndarray | None = None
        self.rows: dict[str, list[Any]] = {
            k: []
            for k in (
                "t",
                "prog",
                "phase",
                "action",
                "eef",
                "opening",
                "pos",
                "quat",
                "lin",
                "ang",
                "robot_contact",
                "native",
                "placed",
                "env_native",
            )
        }
        self.contacts: list[list[str]] = []
        self.video: Video | None = None
        self.state: Any = None
        self.native: list[bool] = []
        self.placed: list[bool] = []

    # ---- one control step --------------------------------------------------------------

    def _observe(self) -> None:
        from fleetlab.sim.libero_success import at_rest, robot_touches

        state = self.env.privileged_state()
        self.state = state
        self.native = [holds(self.rs, g.predicates) for g in self.goals]
        placed = []
        for g, nat in zip(self.goals, self.native, strict=True):
            ok = nat and not robot_touches(state, g.mover)
            if g.mover in state.object_velocities:
                ok = ok and at_rest(state, g.mover)
            placed.append(ok)
        self.placed = placed

    def _log(self, prog: int, phase: int, action: np.ndarray, env_native: bool) -> None:
        from fleetlab.sim.libero_success import robot_touches

        st = self.state
        r = self.rows
        r["t"].append(self.t)
        r["prog"].append(prog)
        r["phase"].append(phase)
        r["action"].append(np.asarray(action, np.float32))
        r["eef"].append(np.asarray(self.obs["robot_state"]["eef"]["pos"], np.float64))
        r["opening"].append(st.gripper_opening)
        r["pos"].append([st.object_poses[n][:3] for n in self.objects])
        r["quat"].append([st.object_poses[n][3:] for n in self.objects])
        r["lin"].append([np.linalg.norm(st.object_velocities[n][:3]) for n in self.free])
        r["ang"].append([np.linalg.norm(st.object_velocities[n][3:]) for n in self.free])
        r["robot_contact"].append([robot_touches(st, n) for n in self.objects])
        r["native"].append(list(self.native))
        r["placed"].append(list(self.placed))
        r["env_native"].append(env_native)
        self.contacts.append([f"{a}|{b}" for a, b in st.contacts])

    def _step(self, action: np.ndarray, prog: int, phase: int) -> None:
        s = time.perf_counter()
        result = self.env.step(np.asarray(action, np.float32))
        self.obs = result.observation
        self.t += 1
        self.last_cmd = np.asarray(action, np.float32)
        self._observe()
        self.sim_s += time.perf_counter() - s
        self._log(prog, phase, action, bool(result.native_success))
        if self.video is not None:
            self.video.add(self.obs)

    def _chunk(self, instruction: str) -> deque[np.ndarray]:
        from fleetlab.contracts.identity import noise_for
        from fleetlab.contracts.policy import NoiseSample
        from fleetlab.engine.episode import ChunkRequest, call_policy

        values = noise_for(self.identity, self.chunk, (30, 20), "float32")
        noise = NoiseSample(
            noise_seed=self.identity.noise_seed, chunk_index=self.chunk, values=values
        )
        request = ChunkRequest(
            key=f"{self.job}-{self.variant}-{self.start}",
            observation=self.obs,
            instruction=instruction,
            noise=noise,
            spec=self.policy.spec,
            row=None,
            batch_size=1,
            chunk_index=self.chunk,
        )
        s = time.perf_counter()
        actions = call_policy(self.policy, [request])[0]
        self.calls.append(time.perf_counter() - s)
        self.chunk += 1
        return deque(np.asarray(a, np.float32) for a in actions)

    # ---- the delivery check (host-fixed) ---------------------------------------------

    def delivery(self) -> dict[str, Any]:
        from fleetlab.sim.libero_success import at_rest, robot_touches

        st = self.state
        named: list[str] = []
        for g in self.goals:
            named += [n for n in g.named if n not in named]
        contact = {n: robot_touches(st, n) for n in named}
        rest = {n: at_rest(st, n) for n in named if n in st.object_velocities}
        commanded_open = self.last_cmd is None or float(self.last_cmd[6]) < 0
        gripper_open = bool(st.gripper_opening >= OPEN_MIN_M and commanded_open)
        goals = list(self.native)
        ok = all(goals) and not any(contact.values()) and all(rest.values()) and gripper_open
        return {
            "delivered": bool(ok),
            "goals": goals,
            "contact": contact,
            "rest": rest,
            "gripper_opening_m": float(st.gripper_opening),
            "gripper_commanded_open": bool(commanded_open),
            "gripper_open": gripper_open,
        }

    # ---- the job ---------------------------------------------------------------------

    def run(self, video: bool) -> dict[str, Any]:
        t_run = time.perf_counter()
        load0 = os.getloadavg()[0]
        s = time.perf_counter()
        self.obs = self.env.reset(self.identity, [])
        reset_s = time.perf_counter() - s
        index = self.env._index()
        self.objects = sorted(index.object_bodies)
        self.free = sorted(index.object_dofs)
        self._observe()
        eef0 = self.obs["robot_state"]["eef"]
        self.home_eef = np.asarray(eef0["pos"], np.float64).copy()
        self.home_mat = np.asarray(eef0["mat"], np.float64).copy()
        robot = self.rs.robots[0]
        self.arm_q = list(robot._ref_joint_pos_indexes)
        self.arm_v = list(robot._ref_joint_vel_indexes)
        self.home_q = self.env._sim.data._data.qpos[self.arm_q].copy()
        self.motion_segments: list[int] = []
        if video and self.out is not None:
            self.video = Video(self.out / "scene.mp4")
            self.video.add(self.obs)
        steps: list[dict[str, Any]] = []
        queue: deque[np.ndarray] = deque()
        prog = 0
        mismatch = 0  # steps where this runner's base-task goal differs from the env's own check
        n_goals = len(self.goals)
        for k, g in enumerate(self.goals):
            last = k == n_goals - 1
            rec: dict[str, Any] = {
                "index": k,
                "task": g.number,
                "stem": g.stem,
                "instruction": g.instruction,
                "start_t": self.t,
                "native_first": None,
                "placed_first": None,
                "native_hash": None,
            }
            queue = deque()  # the previous step's chunk is dropped here
            retry = self.variant == "retry" and k > 0
            attempt_max = STEP_MAX // 2 if retry else STEP_MAX
            budget = min(STEP_MAX, JOB_MAX - self.t)
            n = 0
            attempt_n = 0
            rec["attempts"] = 1
            outcome = None
            while n < budget:
                if retry and attempt_n >= attempt_max and rec["attempts"] == 1:
                    # the first attempt ran out: release, retreat, settle, try again afresh
                    self._transition(prog)
                    queue = deque()
                    rec["attempts"] = 2
                    attempt_n = 0
                    budget = min(n + attempt_max, JOB_MAX - self.t + n)
                    if n >= budget:
                        break
                if not queue:
                    queue = self._chunk(g.instruction)
                self._step(queue.popleft(), prog, 0)
                if self.rows["env_native"][-1] != self.native[0]:
                    mismatch += 1
                n += 1
                attempt_n += 1
                if self.native[k] and rec["native_first"] is None:
                    rec["native_first"] = self.t
                    rec["native_hash"] = self.env.state_hash().hex()
                if self.placed[k] and rec["placed_first"] is None:
                    rec["placed_first"] = self.t
                if self.variant in ("placed", "retry") and not last:
                    if self.placed[k]:
                        outcome, rec["switch_on"] = "reached", "placed"
                        break
                    first = rec["native_first"]
                    if first is not None and self.t - first >= TAIL:
                        outcome, rec["switch_on"] = "reached", "native, placed tail ran out"
                        break
                elif self.native[k]:
                    outcome, rec["switch_on"] = "reached", "native"
                    break
            if outcome is None:
                outcome = "timeout"
                rec["why"] = "step budget" if budget == STEP_MAX else "job budget"
            rec["outcome"] = outcome
            rec["end_t"] = self.t
            rec["steps"] = n
            rec.update(self._switch_facts(g))
            steps.append(rec)
            prog += 1
            if outcome != "reached":
                for j in range(k + 1, n_goals):
                    later = self.goals[j]
                    steps.append(
                        {"index": j, "task": later.number, "stem": later.stem, "outcome": "skipped"}
                    )
                break
            if not last and self.variant.startswith("extend"):
                self._extend(prog, g.instruction, int(self.variant.removeprefix("extend")))
                prog += 1
            if not last and self.variant in ("release", "home", "teleport"):
                {"release": self._transition, "home": self._home, "teleport": self._teleport}[
                    self.variant
                ](prog)
                prog += 1
        tail_steps = 0
        if steps[-1]["outcome"] == "reached" and steps[-1]["index"] == n_goals - 1:
            # The end of every job: the last instruction goes on until delivered or TAIL steps.
            g = self.goals[-1]
            limit = min(TAIL, JOB_MAX - self.t)
            while tail_steps < limit and not self.delivery()["delivered"]:
                if not queue:
                    queue = self._chunk(g.instruction)
                self._step(queue.popleft(), prog, 2)
                tail_steps += 1
        verdict = self.delivery()
        trace = {k: np.asarray(v) for k, v in self.rows.items()}
        trace["objects"] = np.asarray(self.objects)
        tag, reason = classify({"steps": steps, "delivery": verdict}, trace)
        if self.video is not None:
            self.video.close()
        wall = time.perf_counter() - t_run
        summary = {
            "job": self.job,
            "variant": getattr(self, "label", self.variant),
            "switch_variant": self.variant,
            "start": self.start,
            "base_task": self.goals[0].number,
            "base_stem": self.goals[0].stem,
            "base_bench_index": self.goals[0].bench,
            "init_state": self.identity.start_state,
            "scene_seed": self.identity.scene_seed,
            "noise_seed": self.identity.noise_seed,
            "steps": steps,
            "tail_steps": tail_steps,
            "motion_segments": self.motion_segments,
            "control_steps": self.t,
            "delivery": verdict,
            "delivered": verdict["delivered"],
            "tag": tag,
            "reason": reason,
            "end_hash": self.env.state_hash().hex(),
            "base_check_mismatches": mismatch,
            "timing": {
                "wall_s": wall,
                "reset_s": reset_s,
                "policy_calls": len(self.calls),
                "policy_call_mean_s": float(np.mean(self.calls)) if self.calls else None,
                "policy_s": float(np.sum(self.calls)),
                "sim_s": self.sim_s,
                "load_start": load0,
                "load_end": os.getloadavg()[0],
            },
        }
        if self.out is not None:
            self._write(summary)
        return summary

    def _transition(self, prog: int) -> None:
        """The host-written release: open, retreat 5 cm up, settle (probe vocabulary)."""
        from scipy.spatial.transform import Rotation

        from fleetlab.contracts.probes import program_trajectory

        traj = program_trajectory(release_program().segments)
        eef = self.obs["robot_state"]["eef"]
        start = np.asarray(eef["pos"], np.float64).copy()
        rot = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
        for k in range(len(traj.segment)):
            if self.t >= JOB_MAX:
                break
            action = np.zeros(7, np.float64)
            action[:3] = start + traj.offset[k]
            action[3:6] = rot
            action[6] = float(traj.gripper[k])
            self._step(action.astype(np.float32), prog, 1)

    def _extend(self, prog: int, instruction: str, steps: int) -> None:
        """A policy step under the same instruction for a fixed number of control steps."""
        queue: deque[np.ndarray] = deque()  # a step boundary: the chunk is dropped
        for _ in range(steps):
            if self.t >= JOB_MAX:
                return
            if not queue:
                queue = self._chunk(instruction)
            self._step(queue.popleft(), prog, 3)

    def _hold_pose(self, pos: np.ndarray, rot: np.ndarray, steps: int, prog: int) -> None:
        for _ in range(steps):
            if self.t >= JOB_MAX:
                return
            action = np.zeros(7, np.float64)
            action[:3], action[3:6], action[6] = pos, rot, -1.0
            self._step(action.astype(np.float32), prog, 1)

    def _home(self, prog: int) -> None:
        """Open, then back to the start position in bounded one-axis ramps, then settle."""
        from scipy.spatial.transform import Rotation

        from fleetlab.contracts.probes import ProbeProgram, ProbeSegment

        eef = self.obs["robot_state"]["eef"]
        target = np.asarray(eef["pos"], np.float64).copy()
        rot = Rotation.from_matrix(np.asarray(eef["mat"], np.float64)).as_rotvec()
        self._hold_pose(target, rot, 10, prog)
        segments = 0
        for axis in (2, 0, 1):
            remaining = float(self.home_eef[axis] - target[axis])
            while abs(remaining) > 1e-4:
                piece = float(np.clip(remaining, -0.05, 0.05))
                ProbeProgram(  # each ramp is a valid one-segment probe from its own start
                    segments=[
                        ProbeSegment(
                            primitive="ramp",
                            axis="xyz"[axis],
                            amplitude_m=piece,
                            duration_steps=5,
                            period_steps=None,
                        )
                    ]
                )
                base = target[axis]
                for k in range(5):
                    if self.t >= JOB_MAX:
                        return
                    target[axis] = base + piece * (k + 1) / 5
                    action = np.zeros(7, np.float64)
                    action[:3], action[3:6], action[6] = target, rot, -1.0
                    self._step(action.astype(np.float32), prog, 1)
                remaining -= piece
                segments += 1
        self.motion_segments.append(segments)
        self._hold_pose(target, rot, 10, prog)

    def _teleport(self, prog: int) -> None:
        """Diagnostic only: the arm's joints back to the start, then hold the start pose."""
        import mujoco
        from scipy.spatial.transform import Rotation

        model, data = self.env._sim.model._model, self.env._sim.data._data
        data.qpos[self.arm_q] = self.home_q
        data.qvel[self.arm_v] = 0.0
        mujoco.mj_forward(model, data)
        rot = Rotation.from_matrix(self.home_mat).as_rotvec()
        self._hold_pose(self.home_eef.copy(), rot, 10, prog)

    def _switch_facts(self, g: Goal) -> dict[str, Any]:
        from fleetlab.sim.libero_success import robot_touches

        st = self.state
        cmd_closed = self.last_cmd is not None and float(self.last_cmd[6]) > 0
        touch = robot_touches(st, g.mover)
        speed = st.object_velocities.get(g.mover)
        return {
            "held": bool(cmd_closed and touch),
            "robot_touches_mover": bool(touch),
            "gripper_commanded_closed": bool(cmd_closed),
            "opening_m": float(st.gripper_opening),
            "mover_speed_m_s": None if speed is None else float(np.linalg.norm(speed[:3])),
            "hash": self.env.state_hash().hex(),
        }

    # ---- outputs ---------------------------------------------------------------------

    def _write(self, summary: dict[str, Any]) -> None:
        assert self.out is not None
        self.out.mkdir(parents=True, exist_ok=True)
        r = self.rows
        np.savez_compressed(
            self.out / "trace.npz",
            t=np.asarray(r["t"], np.int32),
            prog=np.asarray(r["prog"], np.int16),
            phase=np.asarray(r["phase"], np.int8),
            action=np.asarray(r["action"], np.float32),
            eef=np.asarray(r["eef"], np.float64),
            opening=np.asarray(r["opening"], np.float64),
            pos=np.asarray(r["pos"], np.float64),
            quat=np.asarray(r["quat"], np.float64),
            lin=np.asarray(r["lin"], np.float64),
            ang=np.asarray(r["ang"], np.float64),
            robot_contact=np.asarray(r["robot_contact"], bool),
            native=np.asarray(r["native"], bool),
            placed=np.asarray(r["placed"], bool),
            env_native=np.asarray(r["env_native"], bool),
            objects=np.asarray(self.objects),
            free=np.asarray(self.free),
            goals=np.asarray([g.stem for g in self.goals]),
        )
        with gzip.open(self.out / "contacts.json.gz", "wt") as f:
            json.dump(self.contacts, f)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, default=_plain))


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


# The body each goal-suite task moves and the body it moves it to (Catalogue.goal agrees).
STEM_BODIES: dict[str, tuple[str, str | None]] = {
    "put_the_wine_bottle_on_top_of_the_cabinet": ("wine_bottle_1", "wooden_cabinet_1"),
    "open_the_top_drawer_and_put_the_bowl_inside": ("akita_black_bowl_1", "wooden_cabinet_1"),
    "turn_on_the_stove": ("flat_stove_1", None),
    "put_the_bowl_on_top_of_the_cabinet": ("akita_black_bowl_1", "wooden_cabinet_1"),
    "put_the_bowl_on_the_plate": ("akita_black_bowl_1", "plate_1"),
    "put_the_wine_bottle_on_the_rack": ("wine_bottle_1", "wine_rack_1"),
    "put_the_cream_cheese_in_the_bowl": ("cream_cheese_1", "akita_black_bowl_1"),
    "open_the_middle_drawer_of_the_cabinet": ("wooden_cabinet_1", None),
    "push_the_plate_to_the_front_of_the_stove": ("plate_1", "main_table"),
    "put_the_bowl_on_the_stove": ("akita_black_bowl_1", "flat_stove_1"),
}


def held_at_switch(steps: list[dict[str, Any]], trace: Any, j: int) -> bool:
    """Whether step j-1's object was in the closed hand when step j's instruction began: the
    gripper commanded closed and a robot body touching the object at the last control step
    before step j's first policy action (after any transition motion)."""
    if j < 1 or j >= len(steps) or steps[j].get("start_t") in (None, 0):
        return False
    row = steps[j]["start_t"] - 1
    mover = STEM_BODIES[steps[j - 1]["stem"]][0]
    obj = {str(n): i for i, n in enumerate(trace["objects"])}
    action = np.asarray(trace["action"])
    contact = np.asarray(trace["robot_contact"])
    return bool(action[row, 6] > 0 and mover in obj and contact[row, obj[mover]])


def classify(summary: dict[str, Any], trace: Any) -> tuple[str, str]:
    """The failure tag of a run, from its step records, its delivery verdict and its trace.

    Precedence: delivered; `timeout` when the first step did not reach its goal; then
    the earliest step, in program order, whose goal was reached and is lost at the end
    ("undone") or that timed out. An undone goal is `held_at_switch` when its object was
    still in the closed hand at the instruction switch after it and in contact when the
    goal was lost (carried out), else `item_tipped`. A later step that timed out is
    `held_at_switch` when the previous step's object was in the closed hand at the
    instruction switch and stayed in contact for over half the step; `timeout` when the
    job's budget cut it; `target_moved` when its target had moved more than 3 cm from
    where it was at the job's start; else `second_skill_failed`. Every goal holding but
    the end check failing is `other`. "The instruction switch" is the last control step
    before the next step's first policy action, after any transition motion.
    """
    steps, verdict = summary["steps"], summary["delivery"]
    if verdict["delivered"]:
        return "delivered", "every goal holds, robot clear, objects at rest, gripper open"
    native = np.asarray(trace["native"], bool)
    contact = np.asarray(trace["robot_contact"], bool)
    pos = np.asarray(trace["pos"], np.float64)
    obj = {str(n): i for i, n in enumerate(trace["objects"])}
    bodies = [STEM_BODIES[r["stem"]] for r in steps]

    def held_into(j: int) -> bool:
        return held_at_switch(steps, trace, j)

    first = steps[0]
    if first["outcome"] == "timeout":
        return "timeout", f"step 1 ({first['stem']}) did not reach its goal ({first['why']})"
    problem: tuple[int, str] | None = None
    for k, rec in enumerate(steps):
        if rec["outcome"] == "reached" and not verdict["goals"][k]:
            problem = (k, "undone")
            break
        if rec["outcome"] == "timeout":
            problem = (k, "timeout")
            break
    if problem is None:
        parts = []
        touched = [n for n, v in verdict["contact"].items() if v]
        moving = [n for n, v in verdict["rest"].items() if not v]
        if touched:
            parts.append("the robot still touches " + ", ".join(touched))
        if moving:
            parts.append(", ".join(moving) + " not at rest")
        if not verdict["gripper_open"]:
            parts.append(f"gripper not open ({verdict['gripper_opening_m']:.3f} m)")
        return "other", "every goal holds at the end but " + "; ".join(parts)
    k, kind = problem
    rec = steps[k]
    mover, target = bodies[k]
    if kind == "undone":
        col = native[:, k]
        drops = [i for i in range(1, len(col)) if col[i - 1] and not col[i]]
        t_lost = drops[-1] if drops else len(col) - 1
        carried = bool(contact[max(0, t_lost - 3) : t_lost + 2, obj[mover]].any())
        if carried and held_into(k + 1):
            return (
                "held_at_switch",
                f"{mover} was still in the hand when the next instruction began and was "
                f"carried out of its goal at control step {t_lost + 1}",
            )
        how = "knocked out by the hand" if carried else "without the hand touching it"
        if target in obj:
            moved = float(
                np.linalg.norm(pos[t_lost, obj[target]] - pos[rec["end_t"] - 1, obj[target]])
            )
            if moved > MOVED_M:
                how = f"when its target {target} moved {moved * 100:.0f} cm"
        return (
            "item_tipped",
            f"step {k + 1}'s goal ({rec['stem']}) held at its switch and was lost at control "
            f"step {t_lost + 1}, {how}",
        )
    prev_mover = bodies[k - 1][0]
    a, b = rec["start_t"], rec["end_t"]
    share = float(contact[a:b, obj[prev_mover]].mean()) if prev_mover in obj and b > a else 0.0
    if held_into(k) and share > 0.5:
        return (
            "held_at_switch",
            f"{prev_mover} was still in the hand when step {k + 1} ({rec['stem']}) began and "
            f"stayed in contact for {share:.0%} of it; the step timed out",
        )
    if rec.get("why") == "job budget":
        return "timeout", f"step {k + 1} ({rec['stem']}) ran out of the job's {JOB_MAX} steps"
    if target in obj:
        d = float(np.linalg.norm(pos[a - 1, obj[target]] - pos[0, obj[target]]))
        if d > MOVED_M:
            return (
                "target_moved",
                f"step {k + 1} ({rec['stem']}) timed out; its target {target} had moved "
                f"{d * 100:.0f} cm since the job began",
            )
    return "second_skill_failed", f"step {k + 1} ({rec['stem']}) timed out from the new state"


def load_policy() -> tuple[Any, Any]:
    """X-VLA on the GPU in fp16_res32 with the pinned convolution database (batch size 1)."""
    from fleetlab.engine import miopen, software
    from fleetlab.policies.xvla import XVLAPolicy

    miopen.activate()
    soft = software.current(miopen.require(1, PRECISION))
    policy = XVLAPolicy()
    policy.load("cuda", PRECISION)
    return policy, soft


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True, help="JSON: {runs: [...], video}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="skip runs with a summary")
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    catalogue = Catalogue()
    t0 = time.perf_counter()
    policy, soft = load_policy()
    print(f"runner: policy loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    env = None
    env_bench = None
    results = args.out / "results.jsonl"
    for entry in batch["runs"]:
        job, variant = entry["job"], entry.get("variant", "naive")
        if variant not in VARIANTS and not re.fullmatch(r"extend\d+", variant):
            raise ValueError(f"unknown variant {variant}")
        numbers = job_steps(job)
        base_bench = catalogue.bench[catalogue.stems[numbers[0]]]
        if env is None or env_bench != base_bench:
            if env is not None:
                env.close()
            s = time.perf_counter()
            env = make_env(base_bench)
            env_bench = base_bench
            print(f"runner: built task {numbers[0]} in {time.perf_counter() - s:.1f} s", flush=True)
        goals = [catalogue.goal(n, env._rs) for n in numbers]
        # free-text instruction overrides per step index; the goals stay the tasks' own
        for index, text in entry.get("instructions", {}).items():
            goals[int(index)] = dataclasses.replace(goals[int(index)], instruction=text)
        label = entry.get("label", variant)
        for start in entry["starts"]:
            out = args.out / job / label / f"s{start:02d}"
            if args.resume and (out / "summary.json").exists():
                continue
            run = Run(env, policy, soft, goals, job, variant, start, out)
            run.label = label
            summary = run.run(bool(batch.get("video", True)))
            with results.open("a") as f:
                f.write(json.dumps(summary, default=_plain) + "\n")
            steps = " ".join(f"{s['outcome'][:3]}@{s.get('end_t', '-')}" for s in summary["steps"])
            tm = summary["timing"]
            print(
                f"{job} {variant} s{start:02d}: {summary['tag']:<20} {steps} "
                f"t={summary['control_steps']} wall={tm['wall_s']:.1f}s calls={tm['policy_calls']} "
                f"({(tm['policy_call_mean_s'] or 0) * 1000:.0f} ms) "
                f"mism={summary['base_check_mismatches']}",
                flush=True,
            )
    if env is not None:
        env.close()
    print(f"runner: done in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
