"""What the recorded traces of the screen show, per run of a program.

HOST-AUTHORED analysis of host feasibility runs; never an input of an engineering agent.

    uv run --locked python spikes/jobs-reliability/traces.py STAGE_DIR [STAGE_DIR ...] --out DIR

Reads only the run directories' public tables (episodes.jsonl, run.json, log/*_steps) and
writes DIR/trace_mechanisms.json (one row per episode) and prints a per-arm summary:

- at the switch (the end of program step 0): the last command and the gripper opening of
  program step 0 (row k of robot_steps is the k-th action), whether a
  robot body touches the first goal's object (held at the switch), and the hand's distance
  from its start pose;
- every goal's losses after it first held, told apart (`goal_losses`): the first loss and
  how many there were, the transient ones (the predicate holds again later, by the end) and
  the final, unrecovered one (the predicate does not hold at the end; its step is where the
  last unbroken stretch of not holding begins). A loss is a change of LIBERO's predicate,
  which for `On` needs contact, so a transient loss is not by itself the object leaving: for
  each loss the goal's object's largest displacement from its position just before the
  loss, over the loss, is recorded. For the first and the final loss: the phase and
  program step in force and whether the robot touched the object within 10 steps before;
- the second goal: first held step, or the outcome of the step that should reach it;
- the bowl's largest lift above its height at the reset during the second step (C3).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

# The free object each requested goal moves, by goal text (the screens' goals).
GOAL_OBJECT = {
    "On akita_black_bowl_1 wooden_cabinet_1_top_side": "akita_black_bowl_1",
    "On akita_black_bowl_1 plate_1": "akita_black_bowl_1",
    "On plate_1 main_table_stove_front_region": "plate_1",
    "On cream_cheese_1 akita_black_bowl_1": "cream_cheese_1",
    "Open wooden_cabinet_1_middle_region": None,
    # the expanded screen's
    "On wine_bottle_1 wooden_cabinet_1_top_side": "wine_bottle_1",
    "On wine_bottle_1 wine_rack_1_top_region": "wine_bottle_1",
    "Turnon flat_stove_1": None,
}
LOOKBACK = 10


def _variant(name: str) -> str:
    if name == "naive-chain":
        return "naive"
    if name.startswith("host-single-"):
        return "single"
    return name.split("-", 2)[2] if name.startswith("host-c") else name


def _goal_texts(record: dict[str, Any]) -> list[str]:
    return [" ".join([g["predicate"], *g["objects"]]) for g in record["state"]["requested_goals"]]


def goal_losses(trace: dict[int, bool]) -> dict[str, Any]:
    """A goal's losses from its per-step predicate values: each loss as (the first step not
    holding after a step holding, the first step holding again or None), the first loss, and
    the final loss (the start of the stretch that runs to the end without holding)."""
    steps = sorted(trace)
    held = [s for s in steps if trace[s]]
    losses: list[tuple[int, int | None]] = []
    lost_at: int | None = None
    for prev, step in pairwise(steps):
        if trace[prev] and not trace[step]:
            lost_at = step
        elif not trace[prev] and trace[step] and lost_at is not None:
            losses.append((lost_at, step))
            lost_at = None
    if lost_at is not None:
        losses.append((lost_at, None))
    return {
        "first_held": held[0] if held else None,
        "held_end": trace[steps[-1]] if steps else None,
        "losses": losses,
        "first_loss": losses[0][0] if losses else None,
        "final_loss": losses[-1][0] if losses and losses[-1][1] is None else None,
    }


def _displacement(track: dict[int, dict[str, Any]], first: int, last: int) -> float | None:
    """The object's largest distance, over steps first..last, from where it was at first - 1."""
    if first - 1 not in track:
        return None
    origin = np.asarray(track[first - 1]["position"])
    far = [
        float(np.linalg.norm(np.asarray(track[s]["position"]) - origin))
        for s in range(first, last + 1)
        if s in track
    ]
    return round(max(far), 3) if far else None


def episode_row(run: Path, episode: dict[str, Any], program: dict[str, Any], goals: list[str]):
    import pyarrow.parquet as pq

    eid = episode["episode_id"]
    robot = pq.read_table(run / "log/robot_steps" / f"{eid}.parquet").to_pylist()
    objects = pq.read_table(run / "log/object_steps" / f"{eid}.parquet").to_pylist()
    goal_rows = pq.read_table(run / "log/goal_steps" / f"{eid}.parquet").to_pylist()
    holds: dict[int, dict[int, bool]] = defaultdict(dict)
    for r in goal_rows:
        holds[r["goal"]][r["step"]] = r["holds"]
    obj: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in objects:
        obj[r["object"]][r["step"]] = r
    by_step = {r["step"]: r for r in robot}
    steps = episode["program_steps"]
    switch = steps[0]["end_control_step"]
    start_pos = np.asarray(robot[0]["eef_pos"])
    row: dict[str, Any] = {
        "run": run.name,
        "episode": eid,
        "start": episode["start"]["init_state"],
        "variant": _variant(program["name"]),
        "delivered": episode["delivered"],
        "reason": episode["delivery"]["reason"],
        "step_outcomes": [s["outcome"] for s in steps],
    }
    first_obj = GOAL_OBJECT.get(goals[0])
    if switch is not None and switch - 1 in by_step:
        r = by_step[switch - 1]  # row k is the k-th action: the first step's last command
        row["switch_step"] = switch
        row["switch_gripper_command"] = r["gripper_command"]
        row["switch_gripper_opening"] = round(r["gripper_opening"], 3)
        row["switch_hand_from_start_m"] = round(
            float(np.linalg.norm(np.asarray(r["eef_pos"]) - start_pos)), 3
        )
        if first_obj is not None and switch in obj[first_obj]:
            row["switch_first_object_touched"] = obj[first_obj][switch]["in_contact_with_robot"]
    for g, text in enumerate(goals):
        found = goal_losses(holds[g])
        moved = GOAL_OBJECT.get(text)
        end = max(holds[g]) if holds[g] else 0
        row[f"goal{g}_first_held"] = found["first_held"]
        row[f"goal{g}_held_end"] = found["held_end"]
        row[f"goal{g}_losses"] = len(found["losses"])
        row[f"goal{g}_transient_losses"] = sum(b is not None for _, b in found["losses"])
        row[f"goal{g}_loss_regained"] = [b is not None for _, b in found["losses"]]
        if moved is not None:
            track = obj[moved]
            row[f"goal{g}_loss_max_displacement_m"] = [
                _displacement(track, a, end if b is None else b - 1) for a, b in found["losses"]
            ]
        row[f"goal{g}_loss_steps"] = [
            (b if b is not None else end + 1) - a for a, b in found["losses"]
        ]
        for which, step in (("first", found["first_loss"]), ("final", found["final_loss"])):
            if step is None:
                continue
            r = by_step.get(step, {})
            row[f"goal{g}_{which}_loss_step"] = step
            row[f"goal{g}_{which}_loss_phase"] = r.get("phase")
            row[f"goal{g}_{which}_loss_program_step"] = r.get("program_step")
            if moved is not None:
                row[f"goal{g}_{which}_loss_robot_touch"] = any(
                    obj[moved].get(s, {}).get("in_contact_with_robot", False)
                    for s in range(max(0, step - LOOKBACK), step + 1)
                )
    if len(goals) > 1 and "akita_black_bowl_1" in obj and switch is not None:
        bowl = obj["akita_black_bowl_1"]
        z0 = bowl[0]["position"][2]
        later = [bowl[s]["position"][2] for s in bowl if s > switch]
        row["bowl_max_lift_after_switch_m"] = round(max(later) - z0, 3) if later else None
    return row


def analyse(stage: Path) -> list[dict[str, Any]]:
    rows = []
    for run in sorted(p for p in stage.iterdir() if (p / "episodes.jsonl").is_file()):
        record = json.loads((run / "run.json").read_text())
        programs = {j["job_id"]: j["program"] for j in record["jobs"]}
        goals = _goal_texts(record)
        for line in (run / "episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            row = episode_row(run, e, programs[e["job_id"]], goals)
            row["stage"] = stage.name
            rows.append(row)
    return rows


def summary(rows: list[dict[str, Any]]) -> None:
    arms: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        arms[(r["stage"], r["run"], r["variant"])].append(r)
    for (stage, run, variant), rs in sorted(arms.items()):
        n = len(rs)
        k = sum(r["delivered"] for r in rs)
        touched = sum(bool(r.get("switch_first_object_touched")) for r in rs)
        closed = sum((r.get("switch_gripper_command") or -1) > 0 for r in rs)
        anyloss = [r for r in rs if r.get("goal0_losses")]
        final = [r for r in rs if r.get("goal0_final_loss_step") is not None]
        regained = [r for r in anyloss if r.get("goal0_held_end")]
        where = Counter(
            f"{r['goal0_final_loss_phase']}:{r['goal0_final_loss_program_step']}"
            + (" touch" if r.get("goal0_final_loss_robot_touch") else "")
            for r in final
        )
        never1 = sum(r.get("goal1_first_held") is None for r in rs if "goal1_held_end" in r)
        print(
            f"{stage:10} {run:15} {variant:12} {k:2}/{n:<2} held-at-switch {touched:2} "
            f"closed-at-switch {closed:2} goal0: any-loss {len(anyloss):2} "
            f"all-regained {len(regained):2} unrecovered {len(final):2} {dict(where)} "
            f"goal1-never {never1:2}"
        )
        if anyloss and "switch_step" in rs[0]:
            first = sorted(r["goal0_first_loss_step"] - r["switch_step"] for r in anyloss)
            last = sorted(r["goal0_final_loss_step"] - r["switch_step"] for r in final)
            transient = [
                (span, move)
                for r in anyloss
                for span, move, back in zip(
                    r["goal0_loss_steps"],
                    r.get("goal0_loss_max_displacement_m", [None] * len(r["goal0_loss_steps"])),
                    r["goal0_loss_regained"],
                    strict=True,
                )
                if back
            ]
            moves = [m for _, m in transient if m is not None]
            print(
                f"    goal0 first loss after switch {first}; final loss after switch {last}; "
                f"{len(transient)} transient losses lasting {sorted(t for t, _ in transient)} "
                f"steps, object moved at most {max(moves) if moves else '-'} m during them"
            )
        for r in rs:
            if not r["delivered"]:
                print(f"    start {r['start']:2}: {r['step_outcomes']} {r['reason'][:110]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stages", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = [r for s in args.stages for r in analyse(Path(s))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_mechanisms.json").write_text(json.dumps(rows, indent=1) + "\n")
    summary(rows)
    print(f"{len(rows)} episodes; rows in {out / 'trace_mechanisms.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
