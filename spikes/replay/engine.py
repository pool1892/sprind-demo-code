"""A small rollout loop with everything that affects an episode under explicit control.

Differences from lerobot-eval's loop, each one there to remove a source of
run-to-run variation that this spike measured:

* numpy's global generator is seeded right before each reset, because LIBERO
  draws the fixture positions (cabinet, stove) from it;
* the sampler noise is drawn per episode from (noise seed, chunk index);
* each episode has its own action queue and is dropped from the batch when it
  ends, instead of being reset and run on;
* an environment's OpenGL context is made current before every reset, step and
  close, because robosuite never switches contexts and several environments
  share this process;
* the batch the policy sees is chosen explicitly: every episode alone, all
  episodes that need a new chunk together, or batches of one fixed size that
  are filled up with copies of a row when too few episodes are left.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch

import common as C


@dataclass
class Spec:
    task: int
    init: int
    seed: int  # seeds the scene (fixture placement)
    noise_seed: int  # seeds the policy's sampler noise
    suite: str = "libero_spatial"
    slot: int = 0  # the episode's row in the inference batch, for batching="slotsN"


@dataclass
class Run:
    spec: Spec
    env: object = None
    obs: dict = None
    queue: list = field(default_factory=list)
    chunk_index: int = 0
    t: int = 0
    slot: int = 0
    done: bool = False
    success: bool = False
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    proprio: list = field(default_factory=list)
    imgs: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    infer_obs: list = field(default_factory=list)


def img_digest(obs: dict) -> np.ndarray:
    out = []
    for k in C.CAMERAS:
        h = hashlib.sha256(np.ascontiguousarray(obs["pixels"][k]).tobytes()).digest()[:8]
        out.append(np.frombuffer(h, np.uint64)[0])
    return np.array(out, np.uint64)


def reset_episode(env, spec: Spec) -> dict:
    """Reset so that the scene depends only on (suite, task, init, seed).

    Three things in the stock reset path make the scene depend on history instead:
    the wrapper builds the simulator lazily with a reset of its own; LIBERO draws
    fixture positions and initial joint settings from numpy's global generator,
    which the seed passed to reset() does not reach on robosuite 1.5; and LIBERO
    appends its joint-setting samplers to a list at every hard reset without ever
    emptying it, so each further reset draws one more number before the fixtures
    are placed.
    """
    env._ensure_env()
    C.make_current(env)
    rs = C.rs_env_of(env)
    if rs.hard_reset and hasattr(rs, "object_property_initializers"):
        # refilled by the model rebuild inside a hard reset; a soft reset neither adds to
        # the list nor refills it, so there it must be left alone
        rs.object_property_initializers.clear()
    np.random.seed(spec.seed)
    env.init_state_id = spec.init
    obs, _ = env.reset(seed=spec.seed)
    return obs


class EnvPool:
    """Reuses simulator objects per (suite, task); a hard reset rebuilds the scene anyway."""

    def __init__(self, size: int = 360):
        self.size = size
        self.free: dict[tuple, list] = {}

    def take(self, spec: Spec):
        key = (spec.suite, spec.task)
        if self.free.get(key):
            return self.free[key].pop()
        return C.make_env(
            spec.suite, spec.task, observation_height=self.size, observation_width=self.size
        )

    def give(self, spec: Spec, env):
        self.free.setdefault((spec.suite, spec.task), []).append(env)

    def close(self):
        for envs in self.free.values():
            for e in envs:
                C.make_current(e)
                e.close()
        self.free = {}


def record(run: Run):
    run.states.append(C.get_mj_state(run.env))
    run.proprio.append(C.flat_proprio(run.obs))
    run.imgs.append(img_digest(run.obs))


def rollout(
    policy,
    specs: list[Spec],
    pool: EnvPool,
    batching: str = "alone",
    amp: str = "fp32",
    max_steps: int | None = None,
    keep_infer_obs: bool = False,
    before_step=None,
) -> list[Run]:
    """Run the episodes in specs from their reset to their end."""
    runs = [Run(spec=s, slot=s.slot) for s in specs]
    for r in runs:
        r.env = pool.take(r.spec)
        r.obs = reset_episode(r.env, r.spec)
        record(r)
    advance(policy, runs, batching, amp, max_steps, keep_infer_obs, before_step)
    for r in runs:
        pool.give(r.spec, r.env)
    return runs


def advance(
    policy,
    runs: list[Run],
    batching: str = "alone",
    amp: str = "fp32",
    max_steps: int | None = None,
    keep_infer_obs: bool = False,
    before_step=None,
) -> None:
    """Step the given runs until each has ended. Runs may be fresh or restored in mid-episode."""
    limit = max_steps or runs[0].env._max_episode_steps
    while True:
        active = [r for r in runs if not r.done and r.t < limit]
        if not active:
            break
        if before_step is not None:
            for r in active:
                before_step(r)
        need = [r for r in active if not r.queue]
        if need:
            pad_to, slotted = 0, False
            if batching == "alone":
                groups = [[r] for r in need]
            elif batching == "together":
                groups = [need]
            elif batching.startswith("slots"):
                # Batches of exactly N rows in which every episode keeps its own row for life.
                # Rows whose episode has ended or needs no chunk are filled with a copy of
                # another row and their output is thrown away.
                pad_to, slotted = int(batching.removeprefix("slots")), True
                groups = [need]
            else:  # "fixedN": batches of exactly N rows, filled up with copies of the first row
                pad_to = int(batching.removeprefix("fixed"))
                groups = [need[i : i + pad_to] for i in range(0, len(need), pad_to)]
            for g in groups:
                if slotted:
                    by_slot = {r.slot: r for r in g}
                    assert len(by_slot) == len(g) and max(by_slot) < pad_to, "slot clash"
                    rows = [by_slot.get(i, g[0]) for i in range(pad_to)]
                    index = {id(r): r.slot for r in g}
                else:
                    rows = g + [g[0]] * max(0, pad_to - len(g))
                    index = {id(r): i for i, r in enumerate(g)}
                batch = policy.batch([r.obs for r in rows], [r.env.task_description for r in rows])
                noise = torch.stack([policy.noise(r.spec.noise_seed, r.chunk_index) for r in rows])
                chunk = policy.chunks(batch, noise, amp=amp)
                acts = policy.to_env_actions(chunk)
                acts = {id(r): acts[index[id(r)]] for r in g}
                chunk = {id(r): chunk[index[id(r)]] for r in g}
                for r in g:
                    r.queue = list(acts[id(r)])
                    r.chunks.append(chunk[id(r)].numpy())
                    r.chunk_index += 1
                    if keep_infer_obs:
                        r.infer_obs.append(
                            {
                                "pixels": {k: r.obs["pixels"][k].copy() for k in C.CAMERAS},
                                "robot_state": r.obs["robot_state"],
                            }
                        )
        for r in active:
            a = r.queue.pop(0)
            C.make_current(r.env)
            r.obs, _, terminated, _, info = r.env.step(a)
            r.actions.append(np.asarray(a, np.float32))
            r.t += 1
            record(r)
            if terminated:
                r.done = True
                r.success = bool(info["is_success"])


def pack(runs: list[Run], prefix: str = "") -> dict:
    out = {}
    for i, r in enumerate(runs):
        p = f"{prefix}e{i}_"
        out[p + "spec"] = np.array([r.spec.task, r.spec.init, r.spec.seed, r.spec.noise_seed])
        out[p + "actions"] = np.stack(r.actions)
        out[p + "states"] = np.stack(r.states)
        out[p + "proprio"] = np.stack(r.proprio)
        out[p + "imgs"] = np.stack(r.imgs)
        out[p + "chunks"] = np.stack(r.chunks) if r.chunks else np.zeros((0, 1, 1), np.float32)
        out[p + "success"] = np.array(r.success)
    return out
