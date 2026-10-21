"""Causal telemetry features shared verbatim by training and the live hook."""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from fleetlab.contracts.live import StepTelemetry

HZ = 20
WINDOW = 20
TIME_KNOTS = tuple(range(20, 280, 20))
BASE_NAMES = (
    "elapsed",
    "eef_x",
    "eef_y",
    "eef_z",
    "height_from_start",
    "opening",
    "closing_command",
    "since_gripper_change",
    "gripper_changes",
    "joint_speed",
    "eef_speed",
    "command_speed",
    "window_path",
    "window_displacement",
    "window_opening_change",
    "window_mean_joint_speed",
    "stall_steps",
    "chunk_phase",
    "eef_rotation_change",
    "joint_displacement",
)
# Interactions let the association between progress and failure change over the attempt.
INTERACTIONS = (
    "height_from_start",
    "opening",
    "closing_command",
    "window_path",
    "stall_steps",
)
FEATURE_NAMES = (
    *BASE_NAMES,
    *(f"time_after_{k}" for k in TIME_KNOTS),
    *(f"late_{name}" for name in INTERACTIONS),
)
FloatArray = npt.NDArray[np.float64]


def _norm(x: FloatArray) -> float:
    return float(np.linalg.norm(x))


class FeatureWindow:
    """One instance per episode; strictly ordered, consecutive steps from zero.

    The hook supplies the state at t and command for t, before the action hook.
    Predictions can first affect the decision at t+1. No images, identifiers,
    events, object pose, outcomes, or wall-clock measurements enter here.
    """

    def __init__(self) -> None:
        self._history: deque[tuple[FloatArray, float, float, float]] = deque(maxlen=WINDOW + 1)
        self._last: StepTelemetry | None = None
        self._initial: StepTelemetry | None = None
        self._switch_step = 0
        self._switches = 0
        self._stall = 0

    def update(self, t: StepTelemetry) -> FloatArray:
        expected = 0 if self._last is None else self._last.step + 1
        if t.step != expected:
            raise ValueError(
                f"telemetry must start at zero and be consecutive: {t.step} != {expected}"
            )
        for name, size in (
            ("joint_pos", 7),
            ("joint_vel", 7),
            ("gripper_qpos", 2),
            ("eef_pos", 3),
            ("eef_quat", 4),
            ("commanded_action", 7),
        ):
            value = np.asarray(getattr(t, name))
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid telemetry {name}")
        # Copy retained arrays: callers may reuse their observation buffers.
        t = StepTelemetry(
            t.step,
            t.time_s,
            t.joint_pos.copy(),
            t.joint_vel.copy(),
            t.gripper_qpos.copy(),
            t.eef_pos.copy(),
            t.eef_quat.copy(),
            t.commanded_action.copy(),
            t.chunk_index,
            t.chunk_step,
        )
        previous = self._last or t
        initial = self._initial or t
        if np.sign(t.commanded_action[-1]) != np.sign(previous.commanded_action[-1]):
            self._switch_step = t.step
            self._switches += 1
        distance = _norm(t.eef_pos - previous.eef_pos)
        self._stall = self._stall + 1 if t.step > 0 and distance < 0.001 else 0
        opening = float(np.abs(t.gripper_qpos).sum())
        speed = _norm(t.joint_vel)
        # A command may be an absolute target or a relative delta. Subtracting
        # measured position from it is not a valid mode-independent tracking error.
        self._history.append((t.eef_pos.copy(), opening, speed, distance))
        old = self._history[0]
        # The first distance belongs to the step BEFORE this window's first position.
        path = sum(row[3] for row in list(self._history)[1:])
        values = [
            t.step / 280,
            *t.eef_pos.tolist(),
            t.eef_pos[2] - initial.eef_pos[2],
            opening,
            float(t.commanded_action[-1] > 0),
            min(t.step - self._switch_step, 280) / HZ,
            self._switches,
            speed,
            distance * HZ,
            _norm((t.commanded_action[:3] - previous.commanded_action[:3]).astype(np.float64)) * HZ,
            path,
            _norm(t.eef_pos - old[0]),
            opening - old[1],
            sum(row[2] for row in self._history) / len(self._history),
            min(self._stall, 280) / HZ,
            max(0, t.chunk_step) / 30,
            min(_norm(t.eef_quat - previous.eef_quat), _norm(t.eef_quat + previous.eef_quat)),
            _norm(t.joint_pos - initial.joint_pos),
        ]
        late = max(0.0, (t.step - 140) / 140)
        features = np.asarray(
            [
                *values,
                *(max(0, t.step - k) / 280 for k in TIME_KNOTS),
                *(values[BASE_NAMES.index(name)] * late for name in INTERACTIONS),
            ],
            dtype=np.float64,
        )
        self._last, self._initial = t, initial
        return features
