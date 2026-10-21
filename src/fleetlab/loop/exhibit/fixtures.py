"""Invented records in the real formats, for the exhibit's tests and example layouts.

`write_fixture` writes, under a directory it marks with a `FIXTURE` file, an engagement
as lane driver's `run_engagement` records it (`engagement.json`, the analysis directory
with submissions, requests, results, notes and session records, one export per round),
lane state's evaluation (`summary.json`, `attempts.parquet`, the one-look ledger and the
frozen one-look list), lane learn's reuse comparison (`comparison.json`) and an agent's
learned skill draft. Outcomes are drawn from fixed rates with a seeded generator; every
rationale and note says it is a fixture. Small synthetic videos (plain colour frames) can
be written for the experiment arms so that the film path runs without the simulator.

Everything here is invented. The exhibit marks every page, table and figure built from it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fleetlab.loop.evaluate import mcnemar_exact, wilson
from fleetlab.loop.exhibit.records import FIXTURE_MARKER

__all__ = ["CAMERA", "DELAY", "write_fixture"]

TASKS = (
    "pick up the black bowl and place it on the plate",
    "pick up the milk and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
)
NOTE = "FIXTURE: invented for the exhibit's layout"
PER_ARM = 8


def CAMERA(robot: str) -> dict[str, Any]:
    return {
        "verb": "recalibrate",
        "target": {
            "scope": robot,
            "component": "scene_camera",
            "config_key": None,
            "condition_key": None,
        },
        "value": None,
    }


def DELAY(scope: str) -> dict[str, Any]:
    return {
        "verb": "revert",
        "target": {
            "scope": scope,
            "component": None,
            "config_key": "control.action_delay_steps",
            "condition_key": None,
        },
        "value": None,
    }


def _episodes_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            ("episode_id", pa.string()),
            ("robot_id", pa.string()),
            ("site_id", pa.string()),
            ("day", pa.int32()),
            ("task_text", pa.string()),
            ("steps", pa.int32()),
            ("outcome", pa.string()),
            ("duration_s", pa.float64()),
            ("video_scene", pa.string()),
            ("video_wrist", pa.string()),
            ("experiment_id", pa.string()),
        ]
    )
    return pa.Table.from_pylist([dict(r) for r in rows], schema=schema)


def _write_export(export: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    (export / "log").mkdir(parents=True, exist_ok=True)
    pq.write_table(_episodes_table(rows), export / "log" / "episodes.parquet")


def _video(path: Path, success: bool, frames: int = 10) -> None:
    from fleetlab.video import write_mp4

    colour = (40, 150, 60) if success else (180, 60, 40)
    images = []
    for i in range(frames):
        image = np.full((64, 64, 3), colour, dtype=np.uint8)
        image[:, : 6 * (i % 10) + 4] = 230
        images.append(image)
    write_mp4(path, images, fps=5, repeatable=True)


class _Ids:
    def __init__(self, start: int = 1) -> None:
        self.next = start

    def __call__(self) -> str:
        value = f"e{self.next:06d}"
        self.next += 1
        return value


def write_fixture(
    root: Path,
    *,
    world_id: str = "w99",
    robots: Mapping[str, str] | None = None,
    rounds: Sequence[int] = (1, 2),
    first_session: int = 1,
    days_per_round: int = 2,
    first_day: int = 0,
    per_robot_day: int = 4,
    weak: Mapping[str, float] | None = None,
    seed: int = 0,
    videos: bool = False,
    learned: bool = True,
    reuse: bool = True,
    probes: bool = False,
    comparators: bool = False,
) -> Path:
    """Write an invented engagement under `root` and return its `engagement.json`; beside
    it, `root/exhibit.json` assembles the engagement with the reuse comparison.

    `weak` gives the failing robots' success rate before a change (the others succeed
    at 0.9); round k's decision adopts a change on the k-th weak robot, which then
    succeeds at 0.85. Round 1 of `rounds` may be absent (e.g. rounds=(2,) after a real
    round 1), in which case earlier ledger entries are invented too.
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / FIXTURE_MARKER).write_text(NOTE + "\n")
    robots = dict(robots or {"r01": "s1", "r02": "s1", "r03": "s2", "r04": "s2"})
    weak = dict(weak or {"r02": 0.3, "r03": 0.4})
    weak_robots = list(weak)
    rng = random.Random(seed)
    analysis = root / "engagement" / f"{world_id}-fixture"
    for sub in ("memory", "outbox", "results", "sessions", "queues"):
        (analysis / sub).mkdir(parents=True, exist_ok=True)
    (analysis / "analysis.json").write_text(
        json.dumps({"analysis_id": analysis.name, "world_id": world_id, "label": NOTE}) + "\n"
    )
    ids = _Ids()
    fixed: list[str] = []  # robots whose change is in force
    all_ops: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    record_rounds: list[dict[str, Any]] = []
    notes = ""
    session = first_session
    x = 1
    for k in range(1, (min(rounds) if rounds else 1)):
        robot = weak_robots[(k - 1) % len(weak_robots)]
        fixed.append(robot)
        entries.append(
            {
                "entry_id": f"L{len(entries) + 1}",
                "round": k,
                "adopted_day": first_day,
                "action": CAMERA(robot),
                "status": "active",
                "replaced_by": None,
            }
        )

    def p(robot: str) -> float:
        return 0.85 if robot in fixed else weak.get(robot, 0.9)

    day = first_day
    for number in rounds:
        days = (day, day + days_per_round - 1)
        ops = []
        for d in range(days[0], days[1] + 1):
            for robot, site in sorted(robots.items()):
                for _ in range(per_robot_day):
                    ok = rng.random() < p(robot)
                    ops.append(
                        {
                            "episode_id": ids(),
                            "robot_id": robot,
                            "site_id": site,
                            "day": d,
                            "task_text": rng.choice(TASKS),
                            "steps": 200 if ok else 280,
                            "outcome": "success" if ok else "failure",
                            "duration_s": 10.0,
                            "video_scene": None,
                            "video_wrist": None,
                            "experiment_id": None,
                        }
                    )
        all_ops += ops
        export = root / "rounds" / f"r{number}" / "export"
        _write_export(export, all_ops)
        target = weak_robots[(number - 1) % len(weak_robots)]
        other = next(r for r in robots if r not in weak)
        s1, s2 = session, session + 1
        submissions.append(
            {
                "kind": "diagnosis",
                "session": s1,
                "receipt_id": f"diagnosis-{s1}",
                "answer": _diagnosis([(target, 0.5), (other, 0.3)]),
            }
        )
        exp_id = f"x{x:04d}"
        x += 1
        arms = [[], [CAMERA(target)], [CAMERA(other)]]
        request = {
            "arms": [{"actions": a} for a in arms],
            "robots": [target],
            "episodes_per_arm": PER_ARM,
            "paired_starts": True,
            "power_note": f"{NOTE}: power note for {target}",
        }
        requests.append({"experiment_id": exp_id, "request": request, "session": s1})
        arm_ids: list[list[str]] = [[], [], []]
        arm_rows: list[dict[str, Any]] = []
        exp_day = days[1] + 1
        for i in range(PER_ARM):
            task = rng.choice(TASKS)
            for a in range(3):
                # Fixed outcomes: the control and the other robot's camera arm succeed once,
                # the target's change fails once, so the change rescues PER_ARM - 2 pairs.
                ok = i != PER_ARM - 1 if a == 1 else i == 0
                eid = ids()
                arm_ids[a].append(eid)
                arm_rows.append(
                    {
                        "episode_id": eid,
                        "robot_id": target,
                        "site_id": robots[target],
                        "day": exp_day,
                        "task_text": task,
                        "steps": 200 if ok else 280,
                        "outcome": "success" if ok else "failure",
                        "duration_s": 10.0,
                        "video_scene": f"video/{eid}.scene.mp4" if videos else None,
                        "video_wrist": None,
                        "experiment_id": exp_id,
                    }
                )
        reply = root / "rounds" / f"r{number}" / "reply"
        _write_export(reply, all_ops + arm_rows)
        all_ops += arm_rows
        if videos:
            for row in arm_rows:
                _video(
                    reply / "log" / "video" / f"{row['episode_id']}.scene.mp4",
                    row["outcome"] == "success",
                )
        results.append(
            {
                "experiment_id": exp_id,
                "status": "done",
                "message": "done",
                "arms": [{"arm_index": a, "episode_ids": arm_ids[a]} for a in range(3)],
            }
        )
        submissions.append(
            {
                "kind": "diagnosis",
                "session": s2,
                "receipt_id": f"diagnosis-{s2}",
                "answer": _diagnosis([(target, 0.85)]),
            }
        )
        fixed.append(target)
        entry_id = f"L{len(entries) + 1}"
        entries.append(
            {
                "entry_id": entry_id,
                "round": number,
                "adopted_day": exp_day + 1,
                "action": CAMERA(target),
                "status": "active",
                "replaced_by": None,
            }
        )
        decision = {
            "adopt": [{"action": CAMERA(target), "rationale": f"{NOTE}: {exp_id} supports it"}],
            "replace": [],
            "undo": [],
            "predicted_set_gain": {"low": 2.0, "mid": 6.0, "high": 10.0},
            "expectations": [f"{NOTE}: {target} succeeds as often as the others"],
            "stop": number == max(rounds),
            "ranked_candidates": [
                {
                    "action": DELAY(robots[other]),
                    "expected_gain": 1.0,
                    "uncertainty": {"low": 0.0, "high": 3.0},
                    "rationale": NOTE,
                }
            ],
        }
        submissions.append(
            {"kind": "decision", "session": s2, "receipt_id": f"decision-{s2}", "answer": decision}
        )
        state = analysis / "session-state" / f"session-{s2:02d}" / "before" / "memory"
        state.mkdir(parents=True, exist_ok=True)
        notes += f"{NOTE}. Round {number}, session {s1}: suspect {target}; requested {exp_id}.\n"
        (state / "notes.md").write_text(notes)
        notes += f"{NOTE}. Round {number}, session {s2}: adopted the change on {target}.\n"
        state = analysis / "session-state" / f"session-{s2 + 1:02d}" / "before" / "memory"
        state.mkdir(parents=True, exist_ok=True)
        (state / "notes.md").write_text(notes)
        for n, ex in ((s1, export), (s2, reply)):
            (analysis / "sessions" / f"session-{n:02d}.json").write_text(
                json.dumps(
                    {
                        "number": n,
                        "state": "waiting" if n == s1 else "done",
                        "export": str(ex),
                        "attempts": [
                            {
                                "ok": True,
                                "seconds": 300.0,
                                "run_dir": str(root / "runs"),
                                "usage": {
                                    "input_tokens": 1000,
                                    "cached_input_tokens": 800,
                                    "output_tokens": 100,
                                    "reasoning_tokens": 10,
                                },
                            }
                        ],
                    }
                )
                + "\n"
            )
        record_rounds.append(
            {
                "round": number,
                "operation": {"days": list(days), "runs": [], "seconds": 1.0},
                "export": {"path": str(export), "seconds": 1.0},
                "sessions": [
                    {"number": s1, "state": "waiting", "error": None, "starts": 1},
                    {"number": s2, "state": "done", "error": None, "starts": 1},
                ],
                "costs": {
                    "sessions": 2,
                    "model_starts": 2,
                    "failed_starts": 0,
                    "starts_without_usage": 0,
                    "model_seconds": 600.0,
                    "tokens": {
                        "input_tokens": 2000,
                        "cached_input_tokens": 1600,
                        "output_tokens": 200,
                        "reasoning_tokens": 20,
                    },
                    "tool_calls": 30,
                    "api_equivalent_usd": {"status": "no price for this model"},
                    "experiments": [{"experiment_id": exp_id, "attempts": 3 * PER_ARM}],
                    "requests": 1,
                    "requests_refused": 0,
                    "arm_attempts": 3 * PER_ARM,
                    "queues": [
                        {
                            "state": "done",
                            "timing": {"simulation_seconds": 120.0},
                            "simulated": {
                                "run_jobs": 3 * PER_ARM,
                                "replays": 2,
                                "failed_confirmed": 1,
                                "partial_jobs": 0,
                                "unconfirmed": 0,
                                "simulated_confirmed": 3 * PER_ARM + 3,
                            },
                        }
                    ],
                },
                "decision": decision,
                "stopped": None,
                "ledger": {
                    "format": "fleetlab.public-ledger/1",
                    "round": number + 1,
                    "entries": [dict(e) for e in entries],
                },
            }
        )
        if probes:
            # Round 1 ran a probe to the end, round 2's probe failed part way.
            part = completed_probe_costs(s1) if number == min(rounds) else failed_probe_costs(s1)
            costs = record_rounds[-1]["costs"]
            costs["queues"][0].update(part.pop("queue"))
            costs.update(part)
        session += 2
        day = exp_day + 1
    (analysis / "memory" / "notes.md").write_text(notes)
    _lines(analysis / "outbox" / "submissions.jsonl", submissions)
    _lines(analysis / "outbox" / "experiments.jsonl", requests)
    _lines(analysis / "results" / "experiments.jsonl", results)
    learned_record = None
    if learned:
        # The draft comes from the last round's deciding session (round 2, session 4 with
        # the defaults), and the host froze it into a library in the real record shape.
        session_of_draft = session - 1
        _lines(analysis / "outbox" / "learned.jsonl", [_learned(session_of_draft)])
        learned_record = _library(
            root / "engagement" / "learned", analysis, world_id, session_of_draft
        )
    engagement: dict[str, Any] = {
        "kind": "engagement",
        "label": f"fixture engagement {world_id}",
        "world_id": world_id,
        "agent": {
            "backend": "codex",
            "model": "fixture-model",
            "effort": "xhigh",
            "service_tier": "fast",
        },
        "budget": {"attempts": 120, "text": NOTE},
        "rounds_planned": len(rounds),
        "analysis": str(analysis),
        "rounds": record_rounds,
        "ended_because": f"round {max(rounds)}: the agent stopped",
        "states": ["original", *[f"round-{n}" for n in rounds], "one-look"],
        # The driver's one-look record: requested cutoff, listed, adoptable, applied.
        "one_look": {"requested_cutoff": 2, "listed": 2, "adoptable": 2, "applied": 1},
        "outcome": "stopped",
    }
    if learned_record is not None:
        engagement["learned"] = learned_record
    look = [CAMERA(weak_robots[0]), DELAY(robots[next(r for r in robots if r not in weak)])]
    (root / "engagement" / "one-look-list.json").write_text(json.dumps(look, indent=1) + "\n")
    evaluation = root / "evaluation"
    _evaluation(evaluation, robots, weak, weak_robots, rounds, day, rng, entries, videos)
    engagement["evaluation"] = {"path": str(evaluation), "seconds": 900.0}
    path = root / "engagement" / "engagement.json"
    path.write_text(json.dumps(engagement, indent=1) + "\n")

    # The driver's record beside the engagement: every simulation by purpose.
    def jobs(run: int, replays: int) -> dict[str, int]:
        return {
            "run_jobs": run,
            "replays": replays,
            "partial_jobs": 0,
            "failed_confirmed": 0,
            "unconfirmed": 0,
            "simulated_confirmed": run + replays,
        }

    simulated = {
        "operation": jobs(len(robots) * per_robot_day * days_per_round * (len(rounds) - 1), 3),
        "experiments": jobs(3 * PER_ARM * len(rounds), 2 * len(rounds)),
        "evaluation": jobs(len(robots) * 8 * (len(rounds) + 3), 5),
    }
    (root / "driver.json").write_text(json.dumps({"simulated": simulated}, indent=1) + "\n")
    manifest: dict[str, Any] = {
        "format": "fleetlab.exhibit-manifest/1",
        "label": f"fixture engagement {world_id}",
        "rounds": [{"engagement": "engagement/engagement.json", "round": n} for n in rounds],
        "evaluation": "evaluation",
        "one_look_list": "engagement/one-look-list.json",
    }
    if reuse:
        (root / "reuse").mkdir()
        (root / "reuse" / "comparison.json").write_text(json.dumps(_reuse(), indent=1) + "\n")
        manifest["reuse"] = "reuse/comparison.json"
    if comparators:
        _plain_search(root / "plain-search", robots, weak_robots[0], evaluation)
        manifest["comparators"] = [
            {
                "name": "plain-search",
                "kind": "plain_search",
                "label": "plain search (told the site and the setting)",
                "record": "plain-search/plain_search.json",
                "attempts": "plain-search/evaluation/attempts.parquet",
            },
            {
                "name": "classical",
                "kind": "pending",
                "label": "fixed statistical procedure",
                "status": "evaluation queued; to be filled once it finishes",
            },
        ]
    (root / "exhibit.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return path


def _probe_block(run_jobs: int, replays: int, partial: int, unconfirmed: int) -> dict[str, Any]:
    """`loop.costs.simulated_jobs` of a probe run directory."""
    return {
        "run_dir": "fixture/probes/session",
        "run_jobs": run_jobs,
        "replays": replays,
        "partial_jobs": partial,
        "failed_confirmed": 0,
        "unconfirmed": unconfirmed,
        "not_delivered": 0,
        "incidents": 1 if unconfirmed else 0,
        "simulated_confirmed": run_jobs + replays,
    }


def completed_probe_costs(session: int) -> dict[str, Any]:
    """A round's probe accounting in the shapes lane probe records (loop/probe aa3c761):
    `loop.costs.probe_costs` and the queue record's `probes`, `simulated_probes`,
    `simulated_all.probes`, `probe_outcomes` and `charged_probe_attempts`. One probe of
    six episodes ran to the end; the engine also replayed six (the chain sample)."""
    block = _probe_block(6, 6, 0, 0)
    return {
        "probes": [
            {
                "probe_id": "p0001",
                "session": session,
                "robots": ["r02"],
                "repeats": 6,
                "steps_per_episode": 40,
                "status": "done",
                "attempts_executed": 6,
                "attempts_reserved": 0,
                "attempts_charged": 6,
            }
        ],
        "probe_requests": 1,
        "probe_requests_refused": 0,
        "probe_attempts": 6,
        "probe_attempts_executed": 6,
        "probe_attempts_reserved": 0,
        "queue": {
            "probes": ["p0001"],
            "planned_probe_attempts": 6,
            "simulated_probes": block,
            "simulated_all": {"probes": block},
            "probe_outcomes": [
                {
                    "probe_id": "p0001",
                    "status": "done",
                    "attempts_executed": 6,
                    "attempts_reserved": 0,
                }
            ],
            "charged_probe_attempts": 6,
        },
    }


def failed_probe_costs(session: int) -> dict[str, Any]:
    """As `completed_probe_costs`, for a probe of six that broke off: the engine finished
    one episode and left one partly written (known execution: two), two assignments it
    could not confirm, and the outcome keeps the other four attempts reserved."""
    block = _probe_block(1, 0, 1, 2)
    return {
        "probes": [
            {
                "probe_id": "p0002",
                "session": session,
                "robots": ["r03"],
                "repeats": 6,
                "steps_per_episode": 40,
                "status": "failed",
                "attempts_executed": 2,
                "attempts_reserved": 4,
                "attempts_charged": 6,
            }
        ],
        "probe_requests": 1,
        "probe_requests_refused": 0,
        "probe_attempts": 6,
        "probe_attempts_executed": 2,
        "probe_attempts_reserved": 4,
        "queue": {
            "probes": ["p0002"],
            "planned_probe_attempts": 6,
            "simulated_probes": block,
            "simulated_all": {"probes": block},
            "probe_outcomes": [
                {
                    "probe_id": "p0002",
                    "status": "failed",
                    "attempts_executed": 2,
                    "attempts_reserved": 4,
                }
            ],
            "charged_probe_attempts": 6,
        },
    }


def aborted_probe_costs(session: int) -> dict[str, Any]:
    """As `completed_probe_costs`, for a queue run (`HeavyRoundEngine`) that aborted before
    its reply: the probe request of six stays `queued` in `probe_costs` (charged 6,
    executed 0, reserved 0: no outcome was recorded), the queue record is `failed` with
    no `probe_outcomes` and no `charged_probe_attempts`, and its telemetry shows one
    finished probe episode."""
    block = _probe_block(1, 0, 0, 0)
    return {
        "probes": [
            {
                "probe_id": "p0003",
                "session": session,
                "robots": ["r02"],
                "repeats": 6,
                "steps_per_episode": 40,
                "status": "queued",
                "attempts_executed": 0,
                "attempts_reserved": 0,
                "attempts_charged": 6,
            }
        ],
        "probe_requests": 1,
        "probe_requests_refused": 0,
        "probe_attempts": 6,
        "probe_attempts_executed": 0,
        "probe_attempts_reserved": 0,
        "queue": {
            "state": "failed",
            "error": "RuntimeError",
            "probes": ["p0003"],
            "planned_probe_attempts": 6,
            "simulated_probes": block,
            "simulated_all": {"probes": block},
        },
    }


def _plain_search(root: Path, robots: Mapping[str, str], robot: str, panel: Path) -> None:
    """A plain search's record in the shape of a real one (plain_search.json: search,
    choice, adopted, ledger, costs with scripted sessions and no model use, raw_costs as
    the engagement counted them, and its own evaluation on the same panel), with its
    evaluation attempts: one setting changed, which rescues the searched robot's
    attempts that the agent's change also rescued."""
    root.mkdir(parents=True)
    (root / "evaluation").mkdir()
    rows = [dict(r) for r in pq.read_table(panel / "attempts.parquet").to_pylist()]
    start = [r for r in rows if r["state"] == "original"]
    after = []
    for r in start:
        success = bool(r["success"]) or r["robot_id"] == robot
        after.append(dict(r, state="round-1", success=success))
    pq.write_table(pa.Table.from_pylist(start + after), root / "evaluation" / "attempts.parquet")
    n = len(start)
    ok0 = sum(bool(r["success"]) for r in start)
    ok1 = sum(bool(r["success"]) for r in after)
    rescued = sum(1 for a, b in zip(start, after, strict=True) if not a["success"] and b["success"])
    action = {
        "verb": "set",
        "target": {
            "scope": robots[robot],
            "component": None,
            "config_key": "control.latency_compensation_steps",
            "condition_key": None,
        },
        "value": 15,
    }
    zero = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    costs = {
        "sessions": 2,
        "model_starts": 0,
        "failed_starts": 0,
        "starts_without_usage": 0,
        "usage_complete": True,
        "usage_note": None,
        "model_seconds": 0.0,
        "tokens": zero,
        "tool_calls": 15,
        "cost_usd_list_price": 0.0,
        "cost_usd_list_price_status": "complete",
        "api_equivalent_usd": {
            "status": "bounded_api_equivalent",
            "low_usd": 0.0,
            "high_usd": 0.0,
            "basis": "no model was called",
        },
        "experiments": [{"experiment_id": "x0001", "attempts": 144}],
        "requests": 1,
        "requests_refused": 0,
        "arm_attempts": 144,
        "queues": [
            {
                "state": "done",
                "simulated": {
                    "run_jobs": 144,
                    "replays": 21,
                    "partial_jobs": 0,
                    "failed_confirmed": 0,
                    "unconfirmed": 0,
                },
            }
        ],
        "scripted_sessions": 2,
    }
    raw = dict(
        costs, model_starts=2, starts_without_usage=2, usage_complete=False, model_seconds=0.87
    )
    summary = {
        "world_id": "fixture",
        "days": [0],
        "attempts": n,
        "panel_sha256": "0" * 64,
        "states": [
            {
                "name": "original",
                "ledger_version": 0,
                "changes": 0,
                "attempts": n,
                "successes": ok0,
                "rate": ok0 / n,
                "interval": list(wilson(ok0, n)),
            },
            {
                "name": "round-1",
                "ledger_version": 1,
                "changes": 1,
                "attempts": n,
                "successes": ok1,
                "rate": ok1 / n,
                "interval": list(wilson(ok1, n)),
            },
        ],
        "pairs": [
            {
                "first": "original",
                "second": "round-1",
                "rescued": rescued,
                "broken": 0,
                "net": rescued,
                "p_mcnemar": mcnemar_exact(rescued, 0),
            }
        ],
    }
    record = {
        "search": {
            "scope": robots[robot],
            "key": "control.latency_compensation_steps",
            "grid": [0, 5, 10, 15, 20, 25],
            "per_arm": 24,
            "alpha": 0.05,
        },
        "choice": {"chosen": 15, "reason": NOTE},
        "adopted": [action],
        "ledger": "fixture",
        "costs": costs,
        "evaluation": summary,
        "engagement": "fixture",
        "raw_costs": raw,
        "extra": {"outcome": "stopped"},
    }
    (root / "plain_search.json").write_text(json.dumps(record, indent=1) + "\n")


def _lines(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _diagnosis(causes: Sequence[tuple[str, float]]) -> dict[str, Any]:
    return {
        "causes": [
            {
                "target": CAMERA(robot)["target"],
                "direction": "degraded",
                "confidence": c,
                "evidence": [
                    {
                        "episode_id": None,
                        "step": None,
                        "table": "episodes",
                        "query": "SELECT robot_id, count(*) FROM episodes GROUP BY robot_id",
                    }
                ],
                "failure_mode": f"{NOTE}: grasps miss",
            }
            for robot, c in causes
        ],
        "mode_summaries": [f"{NOTE}: one failure mode"],
        "warnings": [],
        "retrain": [],
    }


def _evaluation(
    root: Path,
    robots: Mapping[str, str],
    weak: Mapping[str, float],
    weak_robots: Sequence[str],
    rounds: Sequence[int],
    day: int,
    rng: random.Random,
    entries: Sequence[Mapping[str, Any]],
    videos: bool = False,
) -> None:
    root.mkdir(parents=True)
    # A reference method on the same attempts, as the evaluation would record one.
    names = ["original", *[f"round-{n}" for n in rounds], "one-look", "fixed-procedure"]
    first = min(rounds)
    fixed_by: dict[str, set[str]] = {"original": set()} | {
        f"round-{n}": set(weak_robots[: n - first + 1 + (first - 1)]) for n in rounds
    }
    fixed_by["one-look"] = set(weak_robots[:1])
    fixed_by["fixed-procedure"] = set(weak_robots[1:2])
    panel = [(f"e{500000 + i:06d}", robot) for i, robot in enumerate(sorted(robots) * 8)]
    draws = {eid: rng.random() for eid, _ in panel}
    # One attempt the change breaks: the first panel attempt of the first weak robot
    # succeeds before the change and fails after it, so rescued and broken both occur.
    first_weak = next(eid for eid, robot in panel if robot == weak_robots[0])
    rows = []
    for name in names:
        for eid, robot in panel:
            rate = 0.85 if robot in fixed_by[name] else weak.get(robot, 0.9)
            if eid == first_weak:
                draws[eid] = 0.2 if robot not in fixed_by[name] else 0.95
            rows.append(
                {
                    "state": name,
                    "episode_id": eid,
                    "robot_id": robot,
                    "site_id": robots[robot],
                    "day": day,
                    "suite": "fixture",
                    "task_index": 0,
                    "start_state": 0,
                    "success": draws[eid] < rate,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), root / "attempts.parquet")
    if videos:
        # Each state's engine run keeps its visible log: episodes rows and scene videos.
        for name in names:
            mine = [r for r in rows if r["state"] == name]
            log_rows = [
                {
                    "episode_id": r["episode_id"],
                    "robot_id": r["robot_id"],
                    "site_id": r["site_id"],
                    "day": day,
                    "task_text": TASKS[0],
                    "steps": 200 if r["success"] else 280,
                    "outcome": "success" if r["success"] else "failure",
                    "duration_s": 10.0,
                    "video_scene": f"video/{r['episode_id']}.scene.mp4",
                    "video_wrist": None,
                    "experiment_id": None,
                }
                for r in mine
            ]
            _write_export(root / name, log_rows)
            for r in log_rows:
                _video(root / name / "log" / str(r["video_scene"]), r["outcome"] == "success", 4)
    outcome: dict[tuple[str, str], bool] = {
        (str(r["state"]), str(r["episode_id"])): bool(r["success"]) for r in rows
    }
    states = []
    for name in names:
        ok = sum(outcome[(name, e)] for e, _ in panel)
        states.append(
            {
                "name": name,
                "ledger_version": 0 if name == "original" else 1,
                "changes": len(fixed_by[name]),
                "attempts": len(panel),
                "successes": ok,
                "rate": ok / len(panel),
                "interval": list(wilson(ok, len(panel))),
            }
        )
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rescued = sum(1 for e, _ in panel if not outcome[(a, e)] and outcome[(b, e)])
            broken = sum(1 for e, _ in panel if outcome[(a, e)] and not outcome[(b, e)])
            pairs.append(
                {
                    "first": a,
                    "second": b,
                    "rescued": rescued,
                    "broken": broken,
                    "net": rescued - broken,
                    "p_mcnemar": mcnemar_exact(rescued, broken),
                }
            )
    summary = {
        "world_id": "fixture",
        "days": [day],
        "attempts": len(panel),
        "panel_sha256": "0" * 64,
        "states": states,
        "pairs": pairs,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (root / "one-look.ledger.json").write_text(
        json.dumps({"entries": [dict(entries[0], status="active")]}, indent=1) + "\n"
    )


def _reuse() -> dict[str, Any]:
    """The comparison of fleetlab.loop.reuse, with the shape of an arm whose usage is
    incomplete: the reuse harness wraps the driver's own lower-bound price record in a
    second lower-bound wrapper (nested), and that arm's usage figures are lower bounds."""
    bounded = {"status": "bounded_api_equivalent", "low_usd": 12.5, "high_usd": 20.0, "basis": NOTE}
    reason = (
        "1 of 3 model starts have no usage record: tokens, model seconds and prices are "
        "lower bounds"
    )

    def arm(score: float, reads: int, complete: bool) -> dict[str, Any]:
        api: dict[str, Any] = bounded
        if not complete:
            inner = {"status": "lower_bound", "reason": reason, "partial": bounded}
            api = {"status": "lower_bound", "reason": reason, "partial": inner}
        return {
            "costs": {
                "learned_skill_reads": {"fixture-skill": reads} if reads else {},
                "usage_complete": complete,
                "usage_note": None if complete else reason,
                "api_equivalent_usd": api,
            },
            "outcome": {"score": score},
        }

    rows = {
        "sessions": (3, 2),
        "tool_calls": (90, 60),
        "experiments_requested": (3, 2),
        "attempts_simulated": (120, 60),
        "model_seconds": (1800.0, 1200.0),
        "input_tokens": (3_000_000, 2_000_000),
        "output_tokens": (40_000, 30_000),
        "cost_usd_list_price": (None, None),
        "score": (0.71, 0.74),
    }
    differences = {}
    for key, (a, b) in rows.items():
        if key in {"model_seconds", "input_tokens", "output_tokens"}:
            differences[key] = {
                "without": a,
                "with": b,
                "with_minus_without": None,
                "status": "unavailable",
                "reason": f"incomplete in ['without']: {reason}",
            }
        elif a is None or b is None:
            differences[key] = {
                "without": a,
                "with": b,
                "with_minus_without": None,
                "status": "unavailable",
                "reason": "no price for this model",
            }
        else:
            differences[key] = {
                "without": a,
                "with": b,
                "with_minus_without": b - a,
                "status": "complete",
            }
    return {
        "case": f"{NOTE}: later fleet",
        "arms": {"without": arm(0.71, 0, False), "with": arm(0.74, 2, True)},
        "differences": differences,
        "negative_transfer": [],
    }


def _library(parent: Path, analysis: Path, world_id: str, session: int) -> dict[str, Any]:
    """A frozen library as `fleetlab.loop.learned.freeze` writes it (the shape of the
    library.json of a real freeze on 13 October 2026): skills under `skills/<name>/`,
    each with `provenance` and `source` {analysis, world_id, fleet_label, round, session,
    receipt_id}. Three skills: this engagement's draft with the round left out (as the
    driver freezes without a session map, so the reader must map the session), one
    carried from an earlier library of another investigation, and one with no source."""
    from fleetlab.contracts.learned import LearnedSkillDraft, render_learned_skill

    digest = "f1x7" + "0" * 60
    root = parent / f"learned-{digest[:16]}"
    draft = LearnedSkillDraft.model_validate(_learned(session)["draft"])
    line = f"learned by the analysis agent on {NOTE}, session {session}; checked and kept"
    other = draft.model_copy(update={"name": "fixture-carried-skill"})
    bare = draft.model_copy(update={"name": "fixture-skill-without-source"})
    skills = [
        (
            draft,
            line,
            {
                "analysis": str(analysis),
                "world_id": world_id,
                "fleet_label": NOTE,
                "round": None,
                "session": session,
                "receipt_id": f"learned-{session}",
            },
            None,
        ),
        (
            other,
            f"learned on another fleet ({NOTE})",
            {
                "analysis": "/elsewhere/another-analysis",
                "world_id": "w98",
                "fleet_label": NOTE,
                "round": 1,
                "session": 2,
                "receipt_id": "learned-1",
            },
            "c" * 64,
        ),
        (bare, None, None, None),
    ]
    entries = []
    for skill, provenance, source, carried in skills:
        folder = root / "skills" / skill.name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(render_learned_skill(skill, provenance))
        entry: dict[str, Any] = {
            "name": skill.name,
            "provenance": provenance,
            "replaces": None,
            "carried_from": carried,
            "sha256": "0" * 64,
        }
        if source is not None:
            entry["source"] = source
        entries.append(entry)
    (root / "library.json").write_text(
        json.dumps(
            {
                "format": "fleetlab.learned-library/1",
                "digest": digest,
                "kept_on": "fixture",
                "previous": None,
                "skills": entries,
                "checks": NOTE,
            },
            indent=1,
        )
        + "\n"
    )
    return {
        "drafts": [draft.name],
        "library": str(root),
        "digest": digest,
        "rounds": {draft.name: None},
        "kept": [draft.name],
        "refused": {"fixture-refused-draft": ["host-side reason, not shown"]},
    }


def _learned(session: int) -> dict[str, Any]:
    return {
        "receipt_id": f"learned-{session}",
        "session": session,
        "draft": {
            "name": "fixture-one-robot-misses",
            "description": f"{NOTE}: when one robot misses grasps from the first day",
            "applies_when": "one robot fails more than the others on every task from the first day",
            "parameters": [{"name": "robot", "meaning": "the robot that fails"}],
            "evidence": [
                {
                    "claim": "{robot} fails more than the fleet",
                    "sql": "SELECT robot_id, avg(CAST(outcome = 'success' AS INT)) AS rate "
                    "FROM episodes WHERE robot_id = $robot GROUP BY robot_id",
                    "result": "{robot} well below the others",
                }
            ],
            "alternatives": [
                {
                    "explanation": "a site condition",
                    "how_to_tell": "the other robots at the site are not worse",
                }
            ],
            "next_experiment": "recalibrate the robot's scene camera against a control arm",
            "check_with": [],
        },
    }
