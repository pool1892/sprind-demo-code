"""Host feasibility screen of replacement compound jobs.

HOST-AUTHORED. The programs, batches and results of this screen are the host's development
evidence. They are never an input of an engineering agent: the engineering agent of a
later investigation receives the request, the generic skills and its own
evidence, not these programs or their counts.

Every batch is a `fleetlab.job-batch/1` file for the runtime's own runner
(`python -m fleetlab.jobs.runner --batch B.json --out DIR`), so the screen measures the
transitions the agent runtime executes, not the lane S spike's. One batch per requested-goal
set (a batch carries one `RequestState`); every batch loads base task 6's scene (the goal
suite's tasks share it) and runs from the host's development starts.

    generate  the screen's batches and manifest from LIBERO's goal suite (imports LIBERO;
              no simulation): requested goals and instructions parsed by
              `fleetlab.jobs.runner.GoalSuite`, the naive chain by `runner.naive_program`
    validate  every batch against the job contracts and the manifest, without LIBERO
    hold      the hold check: initial state 0 and a program of every step kind
    confirm   confirmation batches for chosen candidate:variant pairs on starts 10 to 29
    generate-expanded
              the expanded screen: five further candidates, the same
              runner, starts and transitions, its own label; `confirm` takes its picks too
    report    counts, step outcomes, failure mechanisms and paired comparisons of run dirs

See README.md for the invocations, the queue commands and the estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fleetlab.contracts.jobs import JobProgram, RequestedGoal

HERE = Path(__file__).resolve().parent
BATCHES = HERE / "batches"
MANIFEST = "manifest.json"
BASE_TASK = 6  # put the cream cheese in the bowl: the shared scene and its stored starts
SCREEN_STARTS = range(0, 10)  # the legacy deterministic development starts of the spike
CONFIRM_STARTS = range(10, 30)  # the rest of the development allocation
RELEASE_FIRST = 30  # initial states 30 to 49 are reserved; this screen never touches them
LABEL = (
    "HOST-AUTHORED feasibility screen (robots-qzfk.44.3): host development evidence, never "
    "an input of an engineering agent"
)

# Requested goals by the plan's task numbers (`contracts.jobs.GOAL_SUITE_TASKS`): each
# goal is that task's own BDDL goal, its skill that task's instruction.
CANDIDATES: dict[str, tuple[int, ...]] = {
    "c3": (8, 3),  # push the plate to the stove front, then the bowl onto the cabinet
    "c1": (6, 7),  # the cheese into the bowl, then open the middle drawer
    "c2": (6, 8),  # the cheese into the bowl, then push the plate to the stove front
    "c4": (3, 7),  # the bowl onto the cabinet, then open the middle drawer
    "c5": (4, 7),  # the bowl onto the plate, then open the middle drawer
}
SINGLES: dict[str, int] = {"drawer": 7, "push": 8, "cabinet": 3}

# The generic transitions compared, from the job vocabulary as it stands (contracts.jobs):
#   naive        the runtime's naive chain (switch the moment the goal holds)
#   home         goal holds, then `home` (200 steps, continue on timeout), then the next skill
#   finish-home  goal holds, the same instruction 30 more steps, home, the next skill
#   lift-home    goal holds, open the gripper (10 steps), ramp 5 cm up (20 steps), home,
#                the next skill
# At most two beside naive per candidate, chosen from the spike's evidence (README.md).
VARIANTS = ("naive", "home", "finish-home", "lift-home")
SCREEN_ARMS: dict[str, tuple[str, ...]] = {
    "c3": ("naive", "home", "lift-home"),
    "c1": ("naive", "finish-home", "lift-home"),
    "c2": ("naive", "finish-home", "lift-home"),
    "c4": ("naive", "home", "lift-home"),
    "c5": ("naive", "lift-home"),
}
BATCH_ORDER = ("single-drawer", "single-push", "single-cabinet", "c4", "c3", "c1", "c2", "c5")

# The expanded screen: five further pairs of goal-suite tasks, each with
# the naive chain and the same generic home and finish-home transitions. Hypotheses, not
# expected wins; no singles, no order rule, no new starts. Its transition programs carry their
# own label in their notes; the naive chain stays the runtime's canonical one.
EXPANDED_LABEL = (
    "HOST-AUTHORED task-selection evidence, expanded feasibility screen (robots-qzfk.44.7): "
    "host development evidence withheld from engineering agents, never an input of an "
    "engineering agent"
)
EXPANDED_CANDIDATES: dict[str, tuple[int, ...]] = {
    "c6": (0, 7),  # the wine bottle onto the cabinet, then open the middle drawer
    "c7": (5, 7),  # the wine bottle onto the rack, then open the middle drawer
    "c8": (0, 4),  # the wine bottle onto the cabinet, then the bowl onto the plate
    "c9": (5, 3),  # the wine bottle onto the rack, then the bowl onto the cabinet
    "c10": (2, 3),  # turn on the stove, then the bowl onto the cabinet
}
EXPANDED_ARMS: dict[str, tuple[str, ...]] = {
    c: ("naive", "home", "finish-home") for c in EXPANDED_CANDIDATES
}
EXPANDED_CONFIRM_PICKS = 2  # each with its naive chain: at most 80 executions on 10 to 29


@dataclass(frozen=True, eq=False)
class Screen:
    """Which screen a batch belongs to: its id, label, candidates and batch order."""

    bead: str
    label: str
    candidates: dict[str, tuple[int, ...]]
    order: tuple[str, ...]
    singles: dict[str, int] = field(default_factory=dict)


ORIGINAL = Screen("robots-qzfk.44.3", LABEL, CANDIDATES, BATCH_ORDER, SINGLES)
EXPANDED = Screen(
    "robots-qzfk.44.7", EXPANDED_LABEL, EXPANDED_CANDIDATES, tuple(EXPANDED_CANDIDATES)
)


def screen_of(batch: str) -> Screen:
    """The screen a batch (a candidate or a single skill) belongs to."""
    return EXPANDED if batch in EXPANDED_CANDIDATES else ORIGINAL


HOME_STEPS = 200
FINISH_STEPS = 30
OPEN_STEPS = 10
LIFT_M = 0.05
LIFT_STEPS = 20
POLICY_STEPS = 300
BUDGET = {
    "starts_per_request": 20,
    "requests_per_session": 4,
    "sessions": 6,
    "starts_per_investigation": 400,
}


# ---- building ------------------------------------------------------------------------------


def _policy(instruction: str, *, goal: int | None, steps: int | None, max_steps: int) -> dict:
    return {
        "kind": "policy",
        "instruction": instruction,
        "until": {"goal": goal, "steps": steps},
        "max_steps": max_steps,
        "segment": None,
        "steps": None,
        "on_timeout": "stop",
    }


def _motion(primitive: str, duration: int, *, axis: str | None = None, m: float | None = None):
    segment = {
        "primitive": primitive,
        "axis": axis,
        "amplitude_m": m,
        "duration_steps": duration,
        "period_steps": None,
    }
    return {
        "kind": "motion",
        "instruction": None,
        "until": None,
        "max_steps": None,
        "segment": segment,
        "steps": None,
    }


def _home() -> dict:
    return {
        "kind": "home",
        "instruction": None,
        "until": None,
        "max_steps": None,
        "segment": None,
        "steps": None,
        "on_timeout": "continue",
        "duration_steps": HOME_STEPS,
    }


def variant_program(
    name: str, variant: str, instructions: Sequence[str], label: str = LABEL
) -> dict:
    """A two-goal program of a generic transition between the goals' own skills."""
    first, second = instructions
    head = [_policy(first, goal=0, steps=None, max_steps=POLICY_STEPS)]
    tail = [_policy(second, goal=1, steps=None, max_steps=POLICY_STEPS)]
    middle: list[dict] = {
        "home": [_home()],
        "finish-home": [
            _policy(first, goal=None, steps=FINISH_STEPS, max_steps=FINISH_STEPS),
            _home(),
        ],
        "lift-home": [
            _motion("open_gripper", OPEN_STEPS),
            _motion("ramp", LIFT_STEPS, axis="z", m=LIFT_M),
            _home(),
        ],
    }[variant]
    return {
        "name": name,
        "purpose": f"Host screen: the two skills with the generic '{variant}' transition.",
        "steps": head + middle + tail,
        "notes": label,
    }


def suite_tables() -> dict[str, Any]:
    """The goal suite as the runtime parses it: per plan task, goal and instruction."""
    import fleetlab  # noqa: F401  (torch before mujoco)
    from fleetlab.contracts.jobs import GOAL_SUITE, GOAL_SUITE_TASKS
    from fleetlab.jobs.runner import GoalSuite

    suite = GoalSuite()
    from libero.libero import get_libero_path

    folder = Path(get_libero_path("bddl_files")) / GOAL_SUITE
    tasks = []
    for t, stem in enumerate(GOAL_SUITE_TASKS):
        (goal,) = suite.goals[t]  # every goal-suite task states one goal
        bddl = (folder / f"{stem}.bddl").read_bytes()
        tasks.append(
            {
                "task": t,
                "stem": stem,
                "bench_index": suite.bench[t],
                "goal": goal.model_dump(mode="json"),
                "goal_text": goal.text,
                "instruction": suite.language[t],
                "bddl_sha256": hashlib.sha256(bddl).hexdigest(),
            }
        )
    return {"suite": suite, "tasks": tasks}


def _state(text: str, goals: list[dict]) -> dict:
    return {
        "request_text": text,
        "requested_goals": goals,
        "base_task": BASE_TASK,
        "order_rules": [],  # independent final goals: no order rule is invented
        "budget": BUDGET,
        "acceptance": None,
        "message_log": [],
    }


def _starts(states: range) -> list[dict]:
    from fleetlab.jobs.starts import start_identity

    if states.stop > RELEASE_FIRST:
        raise SystemExit(f"initial states {RELEASE_FIRST} on are reserved; not {states}")
    return [start_identity(BASE_TASK, k).model_dump(mode="json") for k in states]


def _batch(name: str, text: str, goals: list[dict], jobs: list[dict]) -> dict:
    screen = screen_of(name)
    return {
        "format": "fleetlab.job-batch/1",
        "host_screen": {"label": screen.label, "batch": name, "bead": screen.bead},
        "state": _state(text, goals),
        "jobs": jobs,
    }


def _job(number: int, kind: str, program: dict, starts: list[dict]) -> dict:
    return {
        "job_id": f"j{number:04d}",
        "variant": kind,  # read by report; the runner ignores it
        "request": {"program": program, "starts": starts, "record_video": False},
    }


def build(
    tables: dict[str, Any], arms: dict[str, tuple[str, ...]], states: range, singles: bool
) -> dict[str, dict]:
    """The batches of a stage, by name: single skills, then the candidates' arms."""
    from fleetlab.contracts.jobs import RequestedGoal
    from fleetlab.jobs.runner import naive_program

    suite = tables["suite"]
    by_task = tables["tasks"]
    instructions = suite.instructions()
    starts = _starts(states)
    out: dict[str, dict] = {}
    if singles:
        for name, task in SINGLES.items():
            row = by_task[task]
            goal = RequestedGoal.model_validate(row["goal"])
            program = naive_program([goal], instructions).model_dump(mode="json")
            program["name"] = f"host-single-{name}"
            program["notes"] = LABEL
            text = f"Host screen: the skill '{row['instruction']}' alone, delivered check."
            out[f"single-{name}"] = _batch(
                f"single-{name}", text, [row["goal"]], [_job(1, "single", program, starts)]
            )
    for cand, kinds in arms.items():
        tasks = screen_of(cand).candidates[cand]
        rows = [by_task[t] for t in tasks]
        goals = [RequestedGoal.model_validate(r["goal"]) for r in rows]
        text = "Host screen: leave " + " and ".join(g.text for g in goals) + " true at the end."
        jobs = []
        for n, kind in enumerate(kinds, start=1):
            if kind == "naive":
                program = naive_program(goals, instructions).model_dump(mode="json")
            else:
                program = variant_program(
                    f"host-{cand}-{kind}",
                    kind,
                    [instructions[g.text] for g in goals],
                    screen_of(cand).label,
                )
            jobs.append(_job(n, kind, program, starts))
        out[cand] = _batch(cand, text, [r["goal"] for r in rows], jobs)
    return out


def _write(folder: Path, batches: dict[str, dict], manifest: dict) -> None:
    files = {f"{name}.batch.json": batch for name, batch in batches.items()}
    texts = {n: json.dumps(b, indent=1) + "\n" for n, b in {**files, MANIFEST: manifest}.items()}
    changed = [
        n for n, t in texts.items() if (folder / n).exists() and (folder / n).read_text() != t
    ]
    if changed:  # a stage's batches are evidence: regenerating may rewrite them only identically
        raise SystemExit(f"{folder} holds other {', '.join(changed)}; choose another --out")
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (folder / name).write_text(text)
    print(f"wrote {len(batches)} batches and {MANIFEST} to {folder}")


def _manifest(
    stage: str,
    tables: dict[str, Any],
    batches: dict[str, dict],
    states: range,
    screen: Screen = ORIGINAL,
):
    from fleetlab.contracts.jobs import JobProgram

    rows = []
    for name, batch in batches.items():
        for job in batch["jobs"]:
            program = JobProgram.model_validate(job["request"]["program"])
            rows.append(
                {
                    "batch": name,
                    "job_id": job["job_id"],
                    "variant": job["variant"],
                    "program": program.name,
                    "digest": program.digest,
                    "starts": len(job["request"]["starts"]),
                }
            )
    return {
        "label": screen.label,
        "stage": stage,
        "base_task": BASE_TASK,
        "init_states": [states.start, states.stop - 1],
        "tasks": tables["tasks"],
        "candidates": {c: list(t) for c, t in screen.candidates.items()},
        "singles": screen.singles,
        "order": [b for b in screen.order if b in batches],
        "jobs": rows,
        "executions": sum(r["starts"] for r in rows),
    }


def cmd_generate(args: argparse.Namespace) -> int:
    tables = suite_tables()
    batches = build(tables, SCREEN_ARMS, SCREEN_STARTS, singles=True)
    _write(Path(args.out), batches, _manifest("screen-01", tables, batches, SCREEN_STARTS))
    return 0


def cmd_generate_expanded(args: argparse.Namespace) -> int:
    tables = suite_tables()
    batches = build(tables, EXPANDED_ARMS, SCREEN_STARTS, singles=False)
    stage = Path(args.out).name
    _write(Path(args.out), batches, _manifest(stage, tables, batches, SCREEN_STARTS, EXPANDED))
    return 0


def machinery_program(cand: str, instructions: Sequence[str]) -> dict:
    """Every step kind of the screen once, ending on step counts, for the hold check: with a
    policy that holds still a goal never holds, so the real programs stop at their first
    step's timeout and only this one reaches the motions, the home step and the tail."""
    first, second = instructions
    return {
        "name": f"host-{cand}-machinery",
        "purpose": "Host hold check: every step kind of the screen once, on step counts.",
        "steps": [
            _policy(first, goal=None, steps=5, max_steps=5),
            _motion("open_gripper", OPEN_STEPS),
            _motion("ramp", LIFT_STEPS, axis="z", m=LIFT_M),
            _home(),
            _policy(second, goal=None, steps=5, max_steps=5),
        ],
        "notes": LABEL,
    }


def cmd_hold(args: argparse.Namespace) -> int:
    """The screen's batches on initial state 0 only, each candidate with the machinery
    program beside its arms: a check of goals, regions and steps with `--policy hold`."""
    tables = suite_tables()
    batches = build(tables, SCREEN_ARMS, range(0, 1), singles=True)
    instructions = tables["suite"].instructions()
    for cand in SCREEN_ARMS:
        batch = batches[cand]
        texts = [
            instructions[" ".join([g["predicate"], *g["objects"]])]
            for g in batch["state"]["requested_goals"]
        ]
        starts = batch["jobs"][0]["request"]["starts"]
        number = len(batch["jobs"]) + 1
        batch["jobs"].append(_job(number, "machinery", machinery_program(cand, texts), starts))
    _write(Path(args.out), batches, _manifest("hold-01", tables, batches, range(0, 1)))
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    arms: dict[str, tuple[str, ...]] = {}
    known = [*CANDIDATES, *EXPANDED_CANDIDATES]
    for pick in args.pick:
        cand, _, kind = pick.partition(":")
        if cand not in known or kind not in VARIANTS:
            raise SystemExit(f"--pick {pick!r}: candidate:variant, of {known}")
        arms[cand] = (*arms.get(cand, ()), kind)
    screens = {screen_of(c) for c in arms}
    if len(screens) != 1:
        raise SystemExit("a confirmation stage confirms candidates of one screen")
    (screen,) = screens
    if screen is EXPANDED:  # at most two screened arms, each paired with its naive chain
        if len(args.pick) > EXPANDED_CONFIRM_PICKS or not args.with_naive:
            raise SystemExit(
                f"the expanded screen confirms at most {EXPANDED_CONFIRM_PICKS} picks, "
                "with --with-naive"
            )
        for cand, kinds in arms.items():
            if not set(kinds) <= set(EXPANDED_ARMS[cand]) - {"naive"}:
                raise SystemExit(f"{cand}: pick of {EXPANDED_ARMS[cand][1:]}, as screened")
    if args.with_naive:
        arms = {c: k if "naive" in k else ("naive", *k) for c, k in arms.items()}
    for cand, kinds in arms.items():
        if len(kinds) != len(set(kinds)):
            raise SystemExit(f"{cand}: a variant picked twice")
    out = Path(
        args.out or BATCHES / ("expanded-confirm-01" if screen is EXPANDED else "confirm-01")
    )
    tables = suite_tables()
    batches = build(tables, arms, CONFIRM_STARTS, singles=False)
    _write(out, batches, _manifest(out.name, tables, batches, CONFIRM_STARTS, screen))
    return 0


# ---- validating, without LIBERO ------------------------------------------------------------


def batch_problems(batch: dict, manifest: dict) -> list[str]:
    """What is wrong with one batch against the job contracts and the stage's manifest."""
    from fleetlab.contracts.jobs import JobRequest, RequestedGoal, RequestState, program_problems
    from fleetlab.jobs.starts import start_identity

    problems: list[str] = []
    name = batch.get("host_screen", {}).get("batch", "?")
    screen = screen_of(name)
    if batch.get("host_screen") != {"label": screen.label, "batch": name, "bead": screen.bead}:
        problems.append(f"{name}: host_screen is not the {screen.bead} screen's label and bead")
    if manifest["label"] != screen.label:
        problems.append(f"{name}: the manifest's label is not the {screen.bead} screen's")
    if batch.get("format") != "fleetlab.job-batch/1":
        problems.append(f"{name}: format {batch.get('format')!r}")
    state = RequestState.model_validate(batch["state"])
    if state.base_task != BASE_TASK:
        problems.append(f"{name}: base task {state.base_task}, want {BASE_TASK}")
    if state.order_rules:
        problems.append(f"{name}: order rules {state.order_rules}; the screen states none")
    tasks = manifest["tasks"]
    wanted = (
        [SINGLES[name.removeprefix("single-")]]
        if name.startswith("single-")
        else list(screen.candidates[name])
    )
    want_goals = [RequestedGoal.model_validate(tasks[t]["goal"]) for t in wanted]
    if state.requested_goals != want_goals:
        problems.append(f"{name}: requested goals are not the BDDL goals of tasks {wanted}")
    instructions = {tasks[t]["goal_text"]: tasks[t]["instruction"] for t in range(len(tasks))}
    lo, hi = manifest["init_states"]
    ids = [j["job_id"] for j in batch["jobs"]]
    if len(ids) != len(set(ids)):
        problems.append(f"{name}: job ids repeat")
    listed = {(r["batch"], r["job_id"]): r for r in manifest["jobs"]}
    for job in batch["jobs"]:
        jid = f"{name}/{job['job_id']}"
        try:
            request = JobRequest.model_validate(job["request"])
        except ValueError as error:  # pydantic's ValidationError is one
            problems.append(f"{jid}: refused by the contract: {error}")
            continue
        problems += [f"{jid}: {p}" for p in program_problems(request.program, state)]
        if not 1 <= len(request.starts) <= state.budget.starts_per_request:
            problems.append(f"{jid}: {len(request.starts)} starts")
        for s in request.starts:
            if s.init_state >= RELEASE_FIRST or not lo <= s.init_state <= hi:
                problems.append(f"{jid}: initial state {s.init_state} outside {lo} to {hi}")
            if s != start_identity(BASE_TASK, s.init_state):
                problems.append(f"{jid}: start {s.init_state} is not the host's start identity")
        if [s.init_state for s in request.starts] != list(range(lo, hi + 1)):
            problems.append(f"{jid}: starts are not initial states {lo} to {hi} in order")
        for i, step in enumerate(request.program.steps):
            if step.instruction is not None and step.instruction not in instructions.values():
                problems.append(f"{jid} step {i}: instruction is not a goal-suite instruction")
            if step.until is not None and step.until.goal is not None:
                goal_text = state.requested_goals[step.until.goal].text
                if step.instruction != instructions[goal_text]:
                    problems.append(f"{jid} step {i}: instruction is not goal {goal_text}'s skill")
        row = listed.get((name, job["job_id"]))
        if row is None or row["digest"] != request.program.digest:
            problems.append(f"{jid}: program digest is not the manifest's")
        if job.get("variant") == "naive" and request.program.name != "naive-chain":
            problems.append(f"{jid}: the naive arm is not the runtime's naive chain")
        expected = expected_program(name, job.get("variant"), want_goals, instructions)
        if expected is None or expected.digest != request.program.digest:
            problems.append(f"{jid}: program is not the screen's {job.get('variant')} program")
    return problems


def expected_program(
    name: str, variant: Any, goals: Sequence[RequestedGoal], instructions: dict[str, str]
) -> JobProgram | None:
    """The program a screen job of this batch and arm must carry, rebuilt from the goals and
    their instructions; None for an arm the screen does not have."""
    from fleetlab.contracts.jobs import JobProgram
    from fleetlab.jobs.runner import naive_program

    program: dict[str, Any]
    texts = [instructions[g.text] for g in goals]
    label = screen_of(name).label
    if variant == "naive" or variant == "single":
        program = naive_program(goals, instructions).model_dump(mode="json")
        if variant == "single":
            program |= {"name": f"host-{name}", "notes": label}
    elif variant == "machinery" and len(texts) == 2:
        program = machinery_program(name, texts)
    elif variant in VARIANTS and len(texts) == 2:
        program = variant_program(f"host-{name}-{variant}", variant, texts, label)
    else:
        return None
    return JobProgram.model_validate(program)


def cmd_validate(args: argparse.Namespace) -> int:
    folder = Path(args.dir)
    manifest = json.loads((folder / MANIFEST).read_text())
    files = sorted(folder.glob("*.batch.json"))
    names = {f.name.removesuffix(".batch.json") for f in files}
    problems = []
    if names != set(manifest["order"]):
        problems.append(f"batch files {sorted(names)} are not the manifest's {manifest['order']}")
    executions = 0
    for f in files:
        batch = json.loads(f.read_text())
        problems += batch_problems(batch, manifest)
        executions += sum(len(j["request"]["starts"]) for j in batch["jobs"])
    if executions != manifest["executions"]:
        problems.append(f"{executions} executions; the manifest says {manifest['executions']}")
    for p in problems:
        print("PROBLEM", p)
    print(
        f"{folder.name}: {len(files)} batches, {executions} executions, "
        f"initial states {manifest['init_states'][0]} to {manifest['init_states'][1]}: "
        + ("valid" if not problems else f"{len(problems)} problems")
    )
    return 0 if not problems else 1


# ---- reporting -----------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% Wilson interval of k of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mechanism(episode: dict) -> str:
    """One label per run, by the first thing in program order that went wrong."""
    if episode["delivered"]:
        return "delivered"
    goals = episode["goals"]
    for g in goals:
        if g["first_held_step"] is None:
            return f"goal{g['goal']}_never_held"
    for g in goals:
        if not g["held_at_end"]:
            return f"goal{g['goal']}_lost_after_holding"
    delivery = episode["delivery"]
    if delivery["robot_touches"]:
        return "goals_hold_robot_touching"
    if delivery["not_at_rest"]:
        return "goals_hold_object_moving"
    return "goals_hold_gripper_not_open"


def _step_outcomes(episode: dict, program: dict) -> str:
    parts = []
    for r, step in zip(episode["program_steps"], program["steps"], strict=True):
        parts.append(f"{step['kind'][:3]}:{r['outcome'][:3]}")
    return " ".join(parts)


def variant_of(program_name: str) -> str:
    """The arm a program of this screen belongs to, from its name."""
    if program_name == "naive-chain":
        return "naive"
    if program_name.startswith("host-single-"):
        return "single"
    if program_name.startswith("host-c"):
        return program_name.split("-", 2)[2]
    return program_name


def summarise_run(run: Path) -> list[dict]:
    """Per job of a run directory: counts, step outcomes, mechanisms, per-start rows."""
    record = json.loads((run / "run.json").read_text())
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    result = json.loads((run / "batch_result.json").read_text())
    status = {j["job_id"]: j for j in result["jobs"]}
    out = []
    for job in record.get("jobs", []):
        jid = job["job_id"]
        program = job["program"]
        mine = [e for e in episodes if e["job_id"] == jid]
        mine.sort(key=lambda e: e["start"]["init_state"])
        rows = [
            {
                "start": e["start"]["init_state"],
                "episode": e["episode_id"],
                "delivered": e["delivered"],
                "mechanism": mechanism(e),
                "steps": _step_outcomes(e, program),
                "first_held": [g["first_held_step"] for g in e["goals"]],
                "held_at_end": [g["held_at_end"] for g in e["goals"]],
                "control_steps": e["control_steps"],
                "reason": e["delivery"]["reason"],
            }
            for e in mine
        ]
        n, k = len(rows), sum(r["delivered"] for r in rows)
        entry = {
            "run": str(run),
            "job_id": jid,
            "variant": variant_of(program["name"]),
            "program": program["name"],
            "digest": job["program_digest"],
            "status": status.get(jid, {}).get("status"),
            "n": n,
            "delivered": k,
            "goal_reached_any": [
                sum(r["first_held"][g] is not None for r in rows) for g in range(_goals(rows))
            ],
            "goal_held_at_end": [
                sum(r["held_at_end"][g] for r in rows) for g in range(_goals(rows))
            ],
            "mechanisms": dict(Counter(r["mechanism"] for r in rows).most_common()),
            "step_outcomes": dict(Counter(r["steps"] for r in rows).most_common()),
            "rows": rows,
        }
        if n >= 20:
            entry["wilson95"] = [round(x, 3) for x in wilson(k, n)]
        out.append(entry)
    return out


def _goals(rows: list[dict]) -> int:
    return len(rows[0]["first_held"]) if rows else 0


def paired(jobs: list[dict]) -> list[dict]:
    """Each non-naive arm against the naive arm of the same run, on the shared starts."""
    naive = next((j for j in jobs if j["variant"] == "naive"), None)
    if naive is None:
        return []
    base = {r["start"]: r["delivered"] for r in naive["rows"]}
    out = []
    for j in jobs:
        if j is naive:
            continue
        mine = {r["start"]: r["delivered"] for r in j["rows"]}
        shared = sorted(set(base) & set(mine))
        out.append(
            {
                "variant": j["variant"],
                "same_starts": len(shared),
                "naive_delivered": sum(base[s] for s in shared),
                "variant_delivered": sum(mine[s] for s in shared),
                "rescued": [s for s in shared if mine[s] and not base[s]],
                "broken": [s for s in shared if base[s] and not mine[s]],
            }
        )
    return out


def report_label(root: Path) -> str:
    """The label of the stage's manifest that run_stage.sh copied beside the results; the first
    screen's when there is none (older result folders)."""
    copied = root / "batches" / MANIFEST
    if not copied.is_file():
        return LABEL
    label = json.loads(copied.read_text()).get("label")
    if label not in (ORIGINAL.label, EXPANDED.label):
        raise SystemExit(f"{copied}: label {label!r} is neither screen's")
    return str(label)


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.root)
    runs = sorted(p for p in root.iterdir() if (p / "episodes.jsonl").is_file())
    label = report_label(root)
    report: dict[str, Any] = {"label": label, "root": str(root), "runs": {}}
    lines = [f"# {root.name}: host screen counts", "", label + ".", ""]
    lines += [
        "| Run | Arm | Delivered | 95% interval | Goal reached / held at end | Mechanisms |",
        "|---|---|---|---|---|---|",
    ]
    pairs_md = []
    for run in runs:
        jobs = summarise_run(run)
        pairs = paired(jobs)
        report["runs"][run.name] = {"jobs": jobs, "paired": pairs}
        for j in jobs:
            interval = (
                f"{j['wilson95'][0]:.0%} to {j['wilson95'][1]:.0%}" if "wilson95" in j else "-"
            )
            goals = ", ".join(
                f"{a}/{b}"
                for a, b in zip(j["goal_reached_any"], j["goal_held_at_end"], strict=True)
            )
            mech = ", ".join(f"{m} {c}" for m, c in j["mechanisms"].items())
            lines.append(
                f"| {run.name} | {j['variant']} | **{j['delivered']}/{j['n']}** | {interval} "
                f"| {goals} | {mech} |"
            )
        for p in pairs:
            pairs_md.append(
                f"| {run.name} | {p['variant']} | {p['same_starts']} | {p['naive_delivered']} "
                f"| {p['variant_delivered']} | {len(p['rescued'])} {p['rescued']} "
                f"| {len(p['broken'])} {p['broken']} |"
            )
    if pairs_md:
        lines += [
            "",
            "Paired against naive on the same starts:",
            "",
            "| Run | Arm | Starts | Naive | Arm | Rescued | Broken |",
            "|---|---|---|---|---|---|---|",
            *pairs_md,
        ]
    out = Path(args.out) if args.out else root
    (out / "screen_report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "screen_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="screen.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="the screen's batches from LIBERO's goal suite")
    gen.add_argument("--out", default=str(BATCHES / "screen-01"))
    gen.set_defaults(func=cmd_generate)
    val = sub.add_parser("validate", help="a stage's batches against the contracts, no LIBERO")
    val.add_argument("dir")
    val.set_defaults(func=cmd_validate)
    hold = sub.add_parser("hold", help="the hold check's batches: initial state 0, machinery")
    hold.add_argument("--out", default=str(BATCHES / "hold-01"))
    hold.set_defaults(func=cmd_hold)
    con = sub.add_parser("confirm", help="confirmation batches on initial states 10 to 29")
    con.add_argument("--pick", action="append", required=True, help="candidate:variant")
    con.add_argument("--with-naive", action="store_true", help="the naive chain beside, paired")
    con.add_argument("--out", help="default: batches/confirm-01, or expanded-confirm-01")
    con.set_defaults(func=cmd_confirm)
    exp = sub.add_parser("generate-expanded", help="the expanded screen")
    exp.add_argument("--out", default=str(BATCHES / "expanded-screen-01"))
    exp.set_defaults(func=cmd_generate_expanded)
    rep = sub.add_parser("report", help="counts and mechanisms of a stage's run dirs")
    rep.add_argument("root", help="the stage's results directory (one run dir per batch)")
    rep.add_argument("--out", help="where the report goes (default: the root)")
    rep.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
