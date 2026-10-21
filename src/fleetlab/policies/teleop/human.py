"""Seeded human stand-in and deterministic operator reservations.

All times are simulated control ticks. Reservations are made in a separate
admission pass, in fleet timestamp order, never in worker completion order.
This module reads identities and snapshot references, never a truth table.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import Field, model_validator
from scipy.special import ndtr

from fleetlab.contracts._base import ContractModel
from fleetlab.contracts.identity import EpisodeIdentity, noise_for, started_at_for
from fleetlab.contracts.live import hold_action
from fleetlab.contracts.policy import Action7, ControlMode
from fleetlab.contracts.rollout import Job, JobMode
from fleetlab.engine.records import CONTROL_HZ
from fleetlab.policies.teleoperator import TELEOPERATOR_SPEC, Teleoperator

HUMAN_NOISE_STREAM = 604  # separate from X-VLA's executed sampling (sample 0)


class HumanSettings(ContractModel):
    """Scenario assumptions, not estimates of a measured human population."""

    operators: int = Field(default=2, ge=1)
    reaction_min_s: float = Field(default=2.0, ge=0)
    reaction_max_s: float = Field(default=4.0, ge=0)
    pace: int = Field(default=2, ge=1)  # control ticks per expert command
    fumble_probability: float = Field(default=0.1, ge=0, le=1)
    expert_steps: int = Field(default=600, ge=1)
    service_s: float = Field(default=90.0, gt=0)

    @model_validator(mode="after")
    def _fits_slot(self) -> HumanSettings:
        if self.reaction_max_s < self.reaction_min_s:
            raise ValueError("reaction_max_s must be at least reaction_min_s")
        if self.service_steps < self.reaction_max_steps + self.active_budget:
            raise ValueError("service slot must cover maximum reaction plus paced expert budget")
        return self

    @property
    def service_steps(self) -> int:
        return math.ceil(self.service_s * CONTROL_HZ)

    @property
    def reaction_max_steps(self) -> int:
        return math.ceil(self.reaction_max_s * CONTROL_HZ)

    @property
    def active_budget(self) -> int:
        return (self.expert_steps + 1) * self.pace


class Admission(ContractModel):
    settings: HumanSettings = Field(default_factory=HumanSettings)
    operator: int = Field(default=0, ge=0)
    arrival_tick: int = Field(default=0, ge=0)
    start_tick: int = Field(default=0, ge=0)
    queue_steps: int = Field(default=0, ge=0)


def draws(identity: EpisodeIdentity, step: int, settings: HumanSettings) -> tuple[int, bool]:
    """Uniform reaction and Bernoulli fumble from the contract's noise stream."""
    uniform = ndtr(noise_for(identity, step, (2,), "float64", sample=HUMAN_NOISE_STREAM))
    reaction = settings.reaction_min_s + float(uniform[0]) * (
        settings.reaction_max_s - settings.reaction_min_s
    )
    return math.ceil(reaction * CONTROL_HZ), bool(uniform[1] < settings.fumble_probability)


def arrival_tick(job: Job, episodes_per_robot_day: int) -> int:
    if job.from_snapshot is None:
        raise ValueError("operator admission needs a hand-over snapshot")
    hours, minutes, seconds = map(
        int, started_at_for(job.identity, episodes_per_robot_day).split(":")
    )
    return (
        job.identity.day * 86400 + hours * 3600 + minutes * 60 + seconds
    ) * CONTROL_HZ + job.from_snapshot.step


def admit(jobs: Sequence[Job], settings: HumanSettings, episodes_per_robot_day: int) -> list[Job]:
    """FIFO, earliest free operator; simultaneous arrivals tie by neutral episode id.

    A fixed slot is reserved even if the expert finishes early. It covers
    reaction and the maximum active budget; queue time is robot idle time,
    not operator time. Reserving the full slot is deliberately conservative.
    """
    if any(j.mode is not JobMode.continue_ for j in jobs):
        raise ValueError("only hand-over continuations enter the operator queue")
    if len({j.identity.episode_id for j in jobs}) != len(jobs):
        raise ValueError("admit one hand-over per episode")
    free = [0] * settings.operators
    result = []
    for job in sorted(
        jobs, key=lambda j: (arrival_tick(j, episodes_per_robot_day), j.identity.episode_id)
    ):
        arrived = arrival_tick(job, episodes_per_robot_day)
        operator = min(range(settings.operators), key=lambda n: (free[n], n))
        start = max(arrived, free[operator])
        admission = Admission(
            settings=settings,
            operator=operator,
            arrival_tick=arrived,
            start_tick=start,
            queue_steps=start - arrived,
        )
        free[operator] = start + settings.service_steps
        result.append(
            job.model_copy(
                update={
                    "teleoperator_params": admission.model_dump(mode="json"),
                    "max_steps": settings.active_budget,
                }
            )
        )
    return result


class HumanTeleoperator(Teleoperator):
    """The expert with real hold steps, slower commands, and a physical retry."""

    # Retain the expert spec name; the human adapter requires a hand-over snapshot.
    spec = TELEOPERATOR_SPEC.model_copy(update={"revision": "teleop-absolute-v1-human-v1"})

    @property
    def commands_in_world_frame(self) -> bool:
        # The reaction wait holds the saved policy command. Active commands and
        # slower-cadence holds come from the expert's native world coordinates.
        return not self.idle

    def __init__(self) -> None:
        super().__init__()
        self.admission = Admission()
        self.idle = False
        self._last: Action7 | None = None
        self._mode = ControlMode.absolute
        self._remaining = 0
        self._ticks = 0
        self._reaction = 0
        self._fumble = False
        self._idle_ticks = 0

    def configure_takeover(
        self, params: dict[str, Any] | None, last_commanded: Action7 | None
    ) -> None:
        self.admission = Admission.model_validate(params or {})
        if last_commanded is None:
            raise ValueError("a human takeover needs the snapshot's last commanded action")
        self._last = np.asarray(last_commanded, np.float32).copy()

    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None:
        super().begin_episode(identity, step)
        settings = self.admission.settings
        self._reaction, self._fumble = draws(identity, step, settings)
        self._remaining = self.admission.queue_steps + self._reaction
        self._mode = identity.policy.control_mode

    def act_one(self, observation: Any, env: Any) -> Action7:
        self._mode = ControlMode(env.spec.control_mode)
        if self._last is None:
            raise ValueError("human policy must start from a configured hand-over")
        self.idle = self._remaining > 0
        if self.idle:
            self._remaining -= 1
            self._idle_ticks += 1
            self.status = (
                "waiting for operator"
                if self._idle_ticks <= self.admission.queue_steps
                else "operator reacting"
            )
            return hold_action(self._mode, self._last)
        if self._ticks % self.admission.settings.pace:
            action = hold_action(self._mode, self._last)
        else:
            if self._expert is None:
                from fleetlab.policies.teleop import ScriptedExpert

                self._expert = ScriptedExpert(
                    env.handle().env,
                    control_mode=self._mode.value,
                    max_steps=self.admission.settings.expert_steps,
                    seed=self._seed,
                    fumble_first=self._fumble,
                )
            action = np.asarray(self._expert.act(), np.float32)
            self.done, self.status = bool(self._expert.done), str(self._expert.status)
        self._last = action
        self._ticks += 1
        return action

    def metrics(self) -> dict[str, Any]:
        return {
            "admission": self.admission.model_dump(mode="json"),
            "reaction_steps": self._reaction,
            "queue_steps": self.admission.queue_steps,
            "idle_steps": self._idle_ticks,
            "active_steps": self._ticks,
            "operator_used_s": (max(0, self._idle_ticks - self.admission.queue_steps) + self._ticks)
            / CONTROL_HZ,
            "operator_reserved_s": self.admission.settings.service_steps / CONTROL_HZ,
            "fumble_draw": self._fumble,
            "fumble_applied": bool(self._expert is not None and self._expert.fumble_applied),
            "status": self.status,
        }
