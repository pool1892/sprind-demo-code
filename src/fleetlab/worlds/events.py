"""Event detection beyond the engine's grasp and release (contracts section 11).

The engine writes `grasp` and `release` (the sign of the commanded gripper) and the
supervisor's own events. The fleet runner adds four kinds from what a fleet records
of every episode, its visible telemetry (`steps`) and outcome:

    events = episode_events(steps, outcome, max_steps)   # one episode, pure
    counts = add_events(run_dir)                         # every run job of a finalized run

Every rule below was calibrated on recorded runs: the dose-response calibration
(one run per fault type with a healthy twin for every episode, 1,000 healthy
episodes), a nightly world of 480 episodes and the miniature fleet's two runs, 8,996
episodes in all. The detectors are clues,
not labels: each one fires rarely on healthy successes and more often on failures and
under some faults, never only under one.

**drop**: the held object slipped out of the closed gripper. While the gripper is
commanded closed, its fingers first stop on something (the opening, the distance
between the two fingers, stays at or above `HOLD_MIN_M` and changes by at most
`HOLD_STEADY_M` per step for `HOLD_STEPS` steps), then close on through to
`EMPTY_MAX_M` before the gripper is commanded open. Healthy fingers closing on nothing
pass 2 mm within nine steps and settle at 1.0 mm; a held bowl rim keeps them at 3.7
mm or more. The event is at the first step below `HOLD_MIN_M`. Rate per episode:
healthy successes 0.3 %, healthy failures 22 %, failures under the heaviest batch 45 %.
A worn gripper whose fingers no longer close fully stops above 2 mm and hides the
slip from this rule.

**stall**: the end effector stays within `STALL_RADIUS_M` of each window's starting
position for `STALL_STEPS` steps (2 s), after the first `SETTLE_STEPS`; qualifying
windows make one event, at its first step, with their combined length. The bound
applies to each local window, not to the combined interval. Healthy successes 0.1 %, healthy
failures 12 %, failures under a changed policy setting 56 % and under a shifted
table 38 %. The detail says how far the commanded target moved meanwhile: far for an
arm held back or lagging its command, hardly for a policy that hesitates.

**collision**: the end effector stops abruptly against something. It moved at
`MOVING_MIN_M` per step or more (mean of three steps), its next step is at most
`STOP_RATIO` of that, it stays stopped (`STOPPED_MAX_M` per step for `AFTER_STEPS`
steps; a target that jumps makes the arm rebound instead), and the commanded target
moves on at least `TARGET_ON_M` in the old direction meanwhile, so the arm was not
told to stop. At most one event per `COLLISION_GAP` steps. Healthy successes 0.2 %,
healthy failures 16 %, a delayed command line 10 % of all its episodes. It does not
see an arm that flails (relative control): that arm never stays stopped.

**near_miss**: a success that nearly was not. One event at the episode's last step
when its outcome is a success and at least one of: an earlier drop; an earlier grasp
that closed on nothing (the opening reached `EMPTY_MAX_M` with no hold before it,
over at least `MISS_MIN_STEPS` closed steps); the success came in the last
`1 - LATE_SHARE` of the step limit. Healthy successes 0.8 %; successes with a delayed
command line 33 %, after a software release that carried one 66 %.

The `detail` is plain fleet language: what the telemetry showed, never a cause.

**Contacts** are not in the visible log. `contact_events(run_dir)` replays a run's
episodes without the policy on the CPU (`workers` processes, up to ten inside a heavy job;
four or fewer as a light job): the engine's reset
with the episode's faults, then the executed actions of its chain, the state hash
checked at every step; an episode that leaves its chain gets no events. Its collision
events (`scene_contact_events`) go to `add_events(run_dir, contacts=...)`. Measured on
757 replayed episodes of every fault type (all bit for bit, 1.9 s each in a worker):
some contact collision in 12 % of healthy successes and 41 % of healthy failures,
none at all under relative control (the arm flails in the air). The telemetry rules
hold up against the replay: 46 of 51 telemetry collisions coincide with a robot contact
within three steps; where the replayed bowl shows a drop and the telemetry rule fires,
they agree (28 of 28), the telemetry four steps (0.2 s) after the fingers lost it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from fleetlab.contracts.log import EventKind, Outcome
from fleetlab.contracts.rollout import JobMode, RunLayout, RunRecord
from fleetlab.log import RunWriter, run_status
from fleetlab.log._files import PARTS_DIR, write_text
from fleetlab.log._rows import decode_part, encode_part
from fleetlab.log.writer import EPISODES_PER_ROBOT_DAY

__all__ = [
    "DERIVED_KINDS",
    "AddEventsError",
    "ContactPass",
    "SceneContacts",
    "add_events",
    "contact_events",
    "contact_pass",
    "episode_events",
    "robot_part",
    "scene_contact_events",
]

DERIVED_KINDS: frozenset[EventKind] = frozenset(
    {EventKind.drop, EventKind.collision, EventKind.near_miss, EventKind.stall}
)

type Event = tuple[int, EventKind, str]
type Array = npt.NDArray[np.float64]

CONTROL_DT_S = 0.05  # 20 Hz, when a table has fewer than two rows to tell

# drop and missed grasp: the gripper opening, finger to finger, in metres
HOLD_MIN_M = 0.003
HOLD_STEADY_M = 0.0006
HOLD_STEPS = 3  # steady changes in a row: four states
HOLD_CLOSED_IN_M = 0.010  # a hold sits at least 1 cm inside the period's widest opening
EMPTY_MAX_M = 0.002
MISS_MIN_STEPS = 8

# stall
SETTLE_STEPS = 5
STALL_STEPS = 40
STALL_RADIUS_M = 0.015
STALL_TARGET_MOVED_M = 0.03  # the detail calls the commanded target's motion "far" from here

# collision
MOVING_MIN_M = 0.006  # per step: 0.12 m/s
STOP_RATIO = 0.3
STOPPED_MAX_M = 0.0025
AFTER_STEPS = 3
TARGET_ON_M = 0.010
COLLISION_GAP = 10

# near miss
LATE_SHARE = 0.8


@dataclass(frozen=True)
class _Telemetry:
    """One episode's columns as arrays; NaN where a value is missing."""

    step: npt.NDArray[np.int64]
    time_s: Array
    eef: Array  # (n, 3)
    opening: Array  # (n,)
    target: Array  # (n, 3) commanded position; NaN in the last row
    gripper_cmd: Array  # (n,) commanded gripper; > 0 closed

    @property
    def n(self) -> int:
        return len(self.step)

    @property
    def dt(self) -> float:
        if self.n < 2:
            return CONTROL_DT_S
        span = float(self.time_s[-1] - self.time_s[0])
        steps = int(self.step[-1] - self.step[0])
        if steps <= 0 or not math.isfinite(span) or span <= 0:
            return CONTROL_DT_S
        return span / steps


def _matrix(column: Sequence[Any] | None, n: int, width: int) -> Array:
    out = np.full((n, width), np.nan)
    if column is None:
        return out
    for i, value in enumerate(column):
        if value is None:
            continue
        row = np.asarray([np.nan if v is None else v for v in value], dtype=np.float64)
        if row.shape == (width,):
            out[i] = row
    return out


def _telemetry(steps: Mapping[str, Sequence[Any]]) -> _Telemetry:
    n = len(steps["step"])
    step = np.asarray(steps["step"], dtype=np.int64)
    time_s = np.asarray(
        [np.nan if v is None else v for v in steps.get("time_s", [None] * n)], dtype=np.float64
    )
    grip = _matrix(steps.get("gripper_qpos"), n, 2)
    action = _matrix(steps.get("commanded_action"), n, 7)
    return _Telemetry(
        step=step,
        time_s=time_s,
        eef=_matrix(steps.get("eef_pos"), n, 3),
        opening=grip[:, 0] - grip[:, 1],
        target=action[:, :3],
        gripper_cmd=action[:, 6],
    )


def _closed_periods(tel: _Telemetry) -> list[tuple[int, int]]:
    """(first, last) rows of each run of steps commanded closed, as the states they act on:
    a command at row t shapes the state at row t + 1, so the period covers the states
    first + 1 ... last + 1 (bounded by the table)."""
    closed = np.nan_to_num(tel.gripper_cmd, nan=-1.0) > 0
    periods = []
    t = 0
    while t < tel.n:
        if closed[t]:
            first = t
            while t < tel.n and closed[t]:
                t += 1
            periods.append((first, t - 1))
        else:
            t += 1
    return periods


@dataclass(frozen=True)
class _Drop:
    row: int  # first state below HOLD_MIN_M
    hold_row: int  # the state where the fingers had stopped on the object


@dataclass(frozen=True)
class _Miss:
    row: int  # first state at or below EMPTY_MAX_M


def _grasps(tel: _Telemetry) -> tuple[list[_Drop], list[_Miss]]:
    drops: list[_Drop] = []
    misses: list[_Miss] = []
    o = tel.opening
    for first, last_cmd in _closed_periods(tel):
        last = min(last_cmd + 1, tel.n - 1)
        hold = None
        for t in range(first + 1 + HOLD_STEPS, last + 1):
            window = o[t - HOLD_STEPS : t + 1]
            if not np.all(np.isfinite(window)):
                continue
            # The fingers must have closed onto something first: a steady opening near the
            # widest one of the period is a gripper still open (a delayed close command).
            widest = float(np.nanmax(o[first + 1 : t + 1]))
            closed_in = widest - float(window.max()) >= HOLD_CLOSED_IN_M
            steady = np.all(np.abs(np.diff(window)) <= HOLD_STEADY_M)
            if window.min() >= HOLD_MIN_M and steady and closed_in:
                hold = t
                break
        empty = next(
            (t for t in range(first + 1, last + 1) if np.isfinite(o[t]) and o[t] <= EMPTY_MAX_M),
            None,
        )
        if hold is not None:
            later = next(
                (t for t in range(hold + 1, last + 1) if np.isfinite(o[t]) and o[t] <= EMPTY_MAX_M),
                None,
            )
            if later is not None:
                row = next(t for t in range(hold + 1, later + 1) if o[t] < HOLD_MIN_M)
                drops.append(_Drop(row=row, hold_row=hold))
        if empty is not None and (hold is None or hold > empty) and last - first >= MISS_MIN_STEPS:
            misses.append(_Miss(row=empty))
    return drops, misses


def _cm(metres: float) -> str:
    return f"{metres * 100:.0f} cm"


def _drop_events(tel: _Telemetry, drops: Sequence[_Drop]) -> list[Event]:
    events: list[Event] = []
    for drop in drops:
        rise = float(tel.eef[drop.row, 2] - tel.eef[drop.hold_row, 2])
        held = (drop.row - drop.hold_row) * tel.dt
        if math.isfinite(rise):
            where = f"{_cm(abs(rise))} {'above' if rise >= 0 else 'below'} where it was grasped"
        else:
            where = "after it was grasped"
        detail = (
            f"held object slipped out of the closed gripper {where}, "
            f"{held:.1f} s after the fingers stopped on it"
        )
        events.append((int(tel.step[drop.row]), EventKind.drop, detail))
    return events


def _window_extent(points: Array, window: int) -> Array:
    """For each row t with t + window in range: the largest distance from points[t] within
    rows t .. t + window; NaN elsewhere or where a point is missing."""
    n = len(points)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    views = np.lib.stride_tricks.sliding_window_view(points, window + 1, axis=0)  # (m, 3, w+1)
    deltas = views - points[: n - window, :, None]
    out[: n - window] = np.sqrt((deltas**2).sum(axis=1)).max(axis=1)
    return out


def _stall_events(tel: _Telemetry) -> list[Event]:
    extent = _window_extent(tel.eef, STALL_STEPS)
    covered = np.zeros(tel.n, dtype=bool)
    for t in range(SETTLE_STEPS, tel.n - STALL_STEPS):
        if np.isfinite(extent[t]) and extent[t] < STALL_RADIUS_M:
            covered[t : t + STALL_STEPS + 1] = True
    events: list[Event] = []
    t = 0
    while t < tel.n:
        if not covered[t]:
            t += 1
            continue
        start = t
        while t < tel.n and covered[t]:
            t += 1
        end = t - 1
        seconds = (end - start) * tel.dt
        targets = tel.target[start : end + 1]
        targets = targets[np.all(np.isfinite(targets), axis=1)]
        moved = (
            float(np.sqrt(((targets - targets[0]) ** 2).sum(axis=1)).max()) if len(targets) else 0
        )
        if moved >= STALL_TARGET_MOVED_M:
            target = f"the commanded target moved up to {_cm(moved)} away meanwhile"
        else:
            target = f"the commanded target stayed within {_cm(STALL_TARGET_MOVED_M)}"
        detail = (
            f"{STALL_STEPS * tel.dt:.1f} s windows of end-effector motion "
            f"within {STALL_RADIUS_M * 100:.1f} cm of each window's start "
            f"for {seconds:.1f} s; {target}"
        )
        events.append((int(tel.step[start]), EventKind.stall, detail))
    return events


def _collision_events(tel: _Telemetry) -> list[Event]:
    p = tel.eef
    v = np.diff(p, axis=0)  # v[t]: from state t to t + 1
    speed = np.sqrt((v**2).sum(axis=1))
    events: list[Event] = []
    last = -COLLISION_GAP - 1
    for t in range(2, tel.n - 1 - AFTER_STEPS):
        before = v[t - 2 : t + 1].mean(axis=0)
        moving = float(np.sqrt((before**2).sum()))
        if not math.isfinite(moving) or moving < MOVING_MIN_M:
            continue
        after = speed[t + 1 : t + 1 + AFTER_STEPS]
        if not (speed[t + 1] <= STOP_RATIO * moving and np.all(after <= STOPPED_MAX_M)):
            continue
        a, b = tel.target[t + 1], tel.target[t + 1 + AFTER_STEPS]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        on = float(np.dot(b - a, before / moving))
        if on < TARGET_ON_M:
            continue
        if t - last > COLLISION_GAP:
            detail = (
                f"end effector stopped abruptly from {moving / tel.dt:.2f} m/s and stayed "
                f"stopped while the commanded target moved on {on * 100:.1f} cm"
            )
            events.append((int(tel.step[t + 1]), EventKind.collision, detail))
        last = t
    return events


def _near_miss_event(
    tel: _Telemetry,
    outcome: str,
    max_steps: int,
    drops: Sequence[_Drop],
    misses: Sequence[_Miss],
) -> list[Event]:
    if outcome != Outcome.success.value or tel.n == 0:
        return []
    last = int(tel.step[-1])
    reasons = []
    if drops:
        step = int(tel.step[drops[0].row])
        reasons.append(f"only after the held object slipped out at step {step}")
    if misses:
        step = int(tel.step[misses[0].row])
        reasons.append(f"only after a grasp closed on nothing at step {step}")
    if max_steps > 0 and last >= LATE_SHARE * max_steps:
        left = (max_steps - last) * tel.dt
        reasons.append(f"late, with {left:.1f} s left")
    if not reasons:
        return []
    return [(last, EventKind.near_miss, "succeeded " + "; ".join(reasons))]


def episode_events(
    steps: Mapping[str, Sequence[Any]], outcome: str, max_steps: int
) -> list[tuple[int, EventKind, str]]:
    """Events of one episode from its visible steps table (columns as a dict of lists, the way
    pyarrow's to_pydict() gives them: step, time_s, joint_pos[7], joint_vel[7], gripper_qpos[2],
    eef_pos[3], eef_quat[4], commanded_action[7] (None in the last row), chunk_index, chunk_step,
    frame_index) and its visible outcome ("success", "failure", "handed_over"). Pure and
    deterministic: the same table gives the same events in any process."""
    if not steps.get("step"):
        return []
    tel = _telemetry(steps)
    drops, misses = _grasps(tel)
    events = [
        *_drop_events(tel, drops),
        *_stall_events(tel),
        *_collision_events(tel),
        *_near_miss_event(tel, str(outcome), int(max_steps), drops, misses),
    ]
    return sorted(events, key=lambda e: (e[0], e[1].value, e[2]))


# ---- into a run ------------------------------------------------------------------------------


class AddEventsError(RuntimeError):
    """The run cannot take derived events: not finalized, or events that do not fit."""


def _event_row(episode_id: str, event: tuple[int, EventKind | str, str]) -> dict[str, Any]:
    step, kind, detail = event
    return {
        "episode_id": episode_id,
        "step": int(step),
        "kind": EventKind(kind).value,
        "detail": str(detail),
    }


def _derived_rows(
    episode_id: str,
    events: Sequence[tuple[int, EventKind | str, str]],
    last_step: int,
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        row = _event_row(episode_id, event)
        if EventKind(row["kind"]) not in DERIVED_KINDS:
            raise AddEventsError(
                f"{episode_id}: an event of kind {row['kind']} is not one this module adds "
                f"({sorted(k.value for k in DERIVED_KINDS)})"
            )
        if not 0 <= row["step"] <= last_step:
            raise AddEventsError(
                f"{episode_id}: event at step {row['step']} outside 0..{last_step}"
            )
        rows.append(row)
    unique = {json.dumps(r, sort_keys=True): r for r in rows}
    return [unique[k] for k in sorted(unique, key=lambda k: _sort_key(unique[k]))]


def _sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (int(row["step"]), str(row["kind"]), str(row["detail"]))


def add_events(
    run_dir: Path, contacts: Mapping[str, list[tuple[int, EventKind, str]]] | None = None
) -> dict[str, int]:
    """For every finished "run" job of a finalized engine run: compute episode_events from its
    visible steps file and episodes row (plus any contact-derived events given for that
    episode id), put them into the job's staged part beside the engine's own events, replacing
    any events of DERIVED_KINDS from an earlier call (idempotent: a second call changes no
    byte), then rebuild the run-level tables with fleetlab.log.RunWriter(...).finalize().
    Returns the counts per kind."""
    root = Path(run_dir)
    layout = RunLayout(root)
    if not layout.run_record.is_file():
        raise AddEventsError(f"{root} has no run record")
    status = run_status(root)
    if not status.finalized:
        raise AddEventsError(
            f"{root} is not finalized ({len(status.partial)} unfinished jobs, "
            f"{len(set(status.done) ^ set(status.merged))} jobs not in the tables)"
        )
    record = RunRecord.model_validate_json(layout.run_record.read_text())
    per_day = record.extra.get(EPISODES_PER_ROBOT_DAY)
    if not isinstance(per_day, int) or isinstance(per_day, bool):
        raise AddEventsError(f"{root}: the run record has no {EPISODES_PER_ROBOT_DAY}")
    given = dict(contacts or {})
    counts: Counter[str] = Counter({k.value: 0 for k in DERIVED_KINDS})
    # The writer's lock keeps the engine out while the parts change.
    with RunWriter(root, record, episodes_per_robot_day=per_day) as writer:
        changed: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for key in status.done:
            path = root / PARTS_DIR / f"{key}.json"
            text = path.read_text()
            part = decode_part(text)
            if part.get("mode") != JobMode.run.value:
                continue
            episode_id = str(part["episode_id"])
            seen.add(episode_id)
            tables = part["tables"]
            episode = tables["episodes"][0]
            max_steps = int(tables["identities"][0]["max_steps"])
            steps = pq.read_table(layout.table("steps", episode_id)).to_pydict()
            found = episode_events(steps, episode["outcome"], max_steps)
            rows = _derived_rows(
                episode_id, [*found, *given.get(episode_id, [])], int(episode["steps"])
            )
            engine = tables.get("events", [])
            kept = [r for r in engine if EventKind(r["kind"]) not in DERIVED_KINDS]
            tables["events"] = kept + rows
            counts.update(row["kind"] for row in rows)
            new = encode_part(part)
            if new != text:
                changed.append((path, new))
        unknown = sorted(set(given) - seen)
        if unknown:  # refused before anything is written
            raise AddEventsError(f"contact events for episodes not in the run: {unknown[:5]}")
        for path, new in changed:
            write_text(new, path, root)
        writer.finalize()
    return dict(sorted(counts.items()))


# ---- contacts, by replaying the run on the CPU -----------------------------------------------

# The robot's bodies by part, from their MuJoCo names (robosuite's Panda with its gripper).
_FINGER_MARKS = ("finger",)  # left and right finger, and the pads at their tips
_GRIPPER_PREFIX = "gripper0_"
_ARM_PREFIX = "robot0_"
_IGNORED = ("robot0_base", "fixed_mount0_")  # the arm's own mount

CONTACT_GAP = 5  # steps without the contact before it counts as a new one
PUSH_M = 0.01  # a non-target object moved this far while the gripper touched it
PUSH_TAIL = 5  # steps after the contact ends that its push is measured over
MAX_CONTACT_WORKERS = 10  # inside a heavy job; a light job stays at four or fewer


def robot_part(body: str) -> str | None:
    """ "arm", "wrist", "gripper" (its body, not the fingers), "finger", or None for a body
    that is not the robot's (or is its mount)."""
    if body.startswith(_IGNORED):
        return None
    if body.startswith(_GRIPPER_PREFIX):
        return "finger" if any(m in body for m in _FINGER_MARKS) else "gripper"
    if body.startswith(_ARM_PREFIX):
        return "wrist" if body.endswith("_hand") else "arm"
    return None


def _plain(name: str) -> str:
    """An object's or fixture's name in words: "wooden_cabinet_1" -> "wooden cabinet"."""
    words = [w for w in name.split("_") if w and not w.isdigit() and w != "main"]
    return " ".join(words) or name


@dataclass(frozen=True)
class SceneContacts:
    """What a replay saw, step by step: the robot's contacts by part, and where the objects were.

    `contacts[t]`: the (part, thing) pairs in contact at state t, `thing` an object's or
    fixture's name, or a MuJoCo body name ("table"); `positions[name][t]`: the object's
    position at state t; `target`: the object the task moves; `movable`: the objects that
    are not fixtures."""

    contacts: Sequence[frozenset[tuple[str, str]]]
    positions: Mapping[str, Array]
    target: str
    movable: frozenset[str]


def scene_contact_events(scene: SceneContacts) -> list[Event]:
    """Collision events from a replay's contacts, one per contact onset (a pair not in
    contact for `CONTACT_GAP` steps before): the arm or wrist touching anything; the
    gripper's body (not its fingers) touching anything but the object it is to pick up;
    any part of the gripper pushing another movable object `PUSH_M` or more."""
    events: list[Event] = []
    last_seen: dict[tuple[str, str], int] = {}
    n = len(scene.contacts)
    for t, pairs in enumerate(scene.contacts):
        for pair in sorted(pairs):
            onset = t - last_seen.get(pair, -CONTACT_GAP - 1) > CONTACT_GAP
            last_seen[pair] = t
            if not onset:
                continue
            part, thing = pair
            words = _plain(thing)
            if part in ("arm", "wrist"):
                events.append((t, EventKind.collision, f"the {part} touched the {words}"))
            elif part == "gripper" and thing != scene.target:
                events.append((t, EventKind.collision, f"the gripper body struck the {words}"))
            elif thing in scene.movable and thing != scene.target:
                end = t
                while end + 1 < n and pair in scene.contacts[end + 1]:
                    end += 1
                track = scene.positions.get(thing)
                if track is None:
                    continue
                stop = min(end + PUSH_TAIL, len(track) - 1)
                moved = float(np.sqrt(((track[t : stop + 1] - track[t]) ** 2).sum(axis=1)).max())
                if moved >= PUSH_M:
                    detail = f"the gripper pushed the {words} {moved * 100:.0f} cm"
                    events.append((t, EventKind.collision, detail))
    return sorted(events, key=lambda e: (e[0], e[1].value, e[2]))


@dataclass(frozen=True)
class ContactPass:
    """What `contact_pass` found: the events of the episodes whose replay matched the chain
    at every step; the episodes that did not, with the first step whose state differed (0:
    the state after the reset); the seconds each replay took."""

    events: dict[str, list[Event]]
    mismatched: dict[str, int]
    seconds: dict[str, float]


def contact_events(
    run_dir: Path, episode_ids: Sequence[str] | None = None, workers: int = 4
) -> dict[str, list[tuple[int, EventKind, str]]]:
    """Collision events from contacts, for `add_events(run_dir, contacts=...)`: every run
    episode (or those given) replayed on the CPU from its chain, in `workers` processes.
    An episode whose replay leaves its chain gets no events and a warning names it."""
    import warnings

    result = contact_pass(run_dir, episode_ids, workers)
    if result.mismatched:
        listed = ", ".join(f"{e} (step {s})" for e, s in sorted(result.mismatched.items())[:10])
        warnings.warn(
            f"{len(result.mismatched)} episodes did not replay bit for bit and have no "
            f"contact events: {listed}",
            stacklevel=2,
        )
    return result.events


def contact_pass(
    run_dir: Path, episode_ids: Sequence[str] | None = None, workers: int = 4
) -> ContactPass:
    """Replay a run's episodes without the policy and derive collision events from contacts.

    Each episode is reset as the engine reset it (its faults' fresh states, the effective
    policy spec, the cell hooks, the start state and scene seed), then stepped with the
    executed actions of its chain (`truth/chain/<episode>.npz`), the state hash checked
    against the chain at every step. Episodes of one task share an environment; the
    cameras are switched off after the reset (nothing renders, the physics is unchanged).
    `workers` > 1 runs that many spawned processes (at most `MAX_CONTACT_WORKERS`; more than
    four is a heavy job); 1 runs here."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    root = Path(run_dir).resolve()
    identities = pq.read_table(
        root / "truth" / "identities.parquet", columns=["episode_id", "identity_json"]
    ).to_pydict()
    by_id = dict(zip(identities["episode_id"], identities["identity_json"], strict=True))
    visible = set(pq.read_table(root / "log" / "episodes.parquet").column("episode_id").to_pylist())
    wanted = sorted(visible) if episode_ids is None else list(episode_ids)
    unknown = [e for e in wanted if e not in visible or e not in by_id]
    if unknown:
        raise ValueError(f"episodes not run in {root}: {unknown[:5]}")
    groups: dict[tuple[str, int, int], list[tuple[str, str]]] = {}
    for episode_id in wanted:
        data = json.loads(by_id[episode_id])
        key = (str(data["suite"]), int(data["task_index"]), int(data["image_size"]))
        groups.setdefault(key, []).append((episode_id, by_id[episode_id]))
    tasks = [
        (str(root), key, items[i : i + 25])
        for key, items in sorted(groups.items())
        for i in range(0, len(items), 25)
    ]
    workers = max(1, min(int(workers), MAX_CONTACT_WORKERS))
    results: list[tuple[str, list[Event] | None, int | None, float]] = []
    if workers == 1 or len(tasks) < 2:
        for task in tasks:
            results.extend(_replay_group(*task))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=context) as pool:
            for done in pool.map(_replay_group, *zip(*tasks, strict=True)):
                results.extend(done)
    events: dict[str, list[Event]] = {}
    mismatched: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for episode_id, found, mismatch, took in results:
        seconds[episode_id] = took
        if mismatch is not None or found is None:
            mismatched[episode_id] = -1 if mismatch is None else mismatch
        else:
            events[episode_id] = found
    return ContactPass(events=events, mismatched=mismatched, seconds=seconds)


def _replay_group(
    root: str, key: tuple[str, int, int], items: Sequence[tuple[str, str]]
) -> list[tuple[str, list[Event] | None, int | None, float]]:
    """One environment for one task; its episodes replayed one after the other."""
    import time

    from fleetlab.runtime import configure_process

    configure_process()
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.contracts.identity import EpisodeIdentity
    from fleetlab.engine.episode import _bound_cell_hook, effective_spec
    from fleetlab.faults import REGISTRY, load_all
    from fleetlab.sim.libero_sim import LiberoCheck, LiberoSimEnv, build_index

    load_all()
    env = LiberoSimEnv(*key)
    out: list[tuple[str, list[Event] | None, int | None, float]] = []
    try:
        rs = env._rs
        model, data = env._sim.model._model, env._sim.data._data
        index = build_index(rs)
        import mujoco

        raw = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            for b in range(model.nbody)
        ]
        parts = [robot_part(name) for name in raw]
        things = [index.body_names[b] for b in range(model.nbody)]
        cameras = [n for n in rs.observation_names if n.endswith("_image")]
        movable = frozenset(rs.objects_dict) - frozenset(rs.fixtures_dict)
        for episode_id, identity_json in items:
            started = time.perf_counter()
            identity = EpisodeIdentity.model_validate_json(identity_json)
            chain = np.load(Path(root) / "truth" / "chain" / f"{episode_id}.npz")
            hashes = [str(h) for h in chain["state_hash"]]
            executed = np.asarray(chain["executed_action"], np.float64)
            pairs = [(REGISTRY.get(f.kind), f) for f in identity.faults]
            states = [(ft, ft.new_episode(f, identity)) for ft, f in pairs]
            spec = effective_spec(identity, states)
            if spec != identity.policy:
                identity = identity.model_copy(update={"policy": spec})
            env.set_success_check(LiberoCheck())
            env.reset(identity, [_bound_cell_hook(ft, st) for ft, st in states])
            for name in cameras:
                rs.modify_observable(name, "enabled", False)
            mismatch: int | None = None
            # The cell as the hooks left it must be the recorded one (older chains have none).
            recorded = str(chain["fixture_hash"]) if "fixture_hash" in chain.files else None
            if recorded is not None and env.fixture_hash().hex() != recorded:
                mismatch = 0
            contacts: list[frozenset[tuple[str, str]]] = []
            positions: dict[str, list[Array]] = {n: [] for n in index.object_bodies}
            try:
                for t in range(len(hashes) if mismatch is None else 0):
                    if t > 0:
                        env._inner._env.step(np.asarray(executed[t], np.float32))
                    if env.state_hash().hex() != hashes[t]:
                        mismatch = t
                        break
                    step: set[tuple[str, str]] = set()
                    for c in range(int(data.ncon)):
                        contact = data.contact[c]
                        b1 = int(model.geom_bodyid[contact.geom1])
                        b2 = int(model.geom_bodyid[contact.geom2])
                        for a, b in ((b1, b2), (b2, b1)):
                            other = parts[b] is None and not raw[b].startswith(_IGNORED)
                            if parts[a] is not None and other:
                                step.add((str(parts[a]), things[b]))
                    contacts.append(frozenset(step))
                    for name, body in index.object_bodies.items():
                        positions[name].append(np.array(data.xpos[body], np.float64))
            finally:
                for name in cameras:
                    rs.modify_observable(name, "enabled", True)
                env.end_episode()
            found = None
            if mismatch is None:
                scene = SceneContacts(
                    contacts=contacts,
                    positions={n: np.array(v) for n, v in positions.items()},
                    target=index.goal.object_body,
                    movable=movable,
                )
                found = scene_contact_events(scene)
            out.append((episode_id, found, mismatch, time.perf_counter() - started))
    finally:
        env.close()
    return out
