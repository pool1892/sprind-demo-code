"""Calibration records, exact-context harvesting and bounded task-cell refinement.

The legacy helpers preserve version-0 dose descriptions for historical worlds.
New calibration retains every active mechanism, source run and physical attempt,
with paired controls for contextual decoys and preregistered continuations.

Read from the faults' measured curves (`fleetlab.faults.records.measured`,
`src/fleetlab/faults/curves.json`): X-VLA in fp16_res32, four batch
rows, the fleet's success check (placed), 100 to 300 episodes of `libero_spatial` per
point against the same episodes healthy. `loss_points` is the paired difference to the
healthy twins with its sign turned, so that harm is positive: what removing the fault
gives back per episode that carries it, in points.

The legacy description predates object calibration; the current measured-context
path uses each scheduled suite and task and never transfers a bowl rate to objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from fleetlab.contracts.world import DayRange
    from fleetlab.faults._common import DosePoint


class CalibratedPoint(NamedTuple):
    kind: str
    label: str  # the calibration point's label in the fault type's `calibration`
    successes: int
    episodes: int
    healthy_successes: int  # the same episodes without the fault
    loss_points: float  # paired difference to the healthy twins, harm positive
    run: str
    verdict: str  # the type's verdict in the curves: "kept" or "decoy_unconfirmed"


def point(kind: str, label: str) -> CalibratedPoint:
    """The calibrated point of a kind by its label; KeyError if it was not measured."""
    from fleetlab.faults.records import measured

    curve = measured(kind)
    for p in curve["points"]:
        if p["label"] == label:
            return CalibratedPoint(
                kind=kind,
                label=label,
                successes=int(p["successes"]),
                episodes=int(p["n"]),
                healthy_successes=int(p["healthy_successes"]),
                loss_points=round(-100.0 * float(p["difference"][0]), 3),
                run=str(p["run_id"]),
                verdict=str(curve["verdict"]),
            )
    raise KeyError(f"no calibrated point {label!r} for {kind}")


def describe(p: CalibratedPoint) -> str:
    """One line naming the point and its numbers, for the fault table."""
    return (
        f"{p.kind} '{p.label}': {p.successes}/{p.episodes} placed against "
        f"{p.healthy_successes}/{p.episodes} for the healthy twins, paired loss "
        f"{p.loss_points:+.1f} points (run {p.run}, type verdict {p.verdict}; "
        "libero_spatial, applied to libero_object unmeasured)"
    )


def extra_doses(kind: str) -> tuple[DosePoint, ...]:
    """Dose descriptors used by the new calibration, without changing fault mechanisms.

    Descriptors are not measurements. Curves only use one when its record exists.
    Keeping them here avoids renumbering the library's historical calibration blocks.
    """
    from fleetlab.contracts.vocabulary import Component, ConfigKey
    from fleetlab.faults._common import DosePoint

    if kind == "calibration_update":
        from fleetlab.faults.calibration_update import NOMINAL_WRIST_CALIBRATION

        return (
            DosePoint(
                "wrist calibration rewritten",
                1.0,
                {"value": [0.002, -0.003, 0.048, 0.0, 0.4, 180.2]},
                ConfigKey.camera_wrist_calibration.value,
                NOMINAL_WRIST_CALIBRATION,
            ),
        )
    if kind == "camera_degradation":
        return (DosePoint("3 px blur", 3.0, {"mode": "blur"}, Component.scene_camera.value),)
    if kind == "action_latency":
        return (DosePoint("10 steps", 10.0, {}, ConfigKey.control_action_delay_steps.value, 0),)
    if kind == "policy_knob":
        return tuple(
            DosePoint(
                f"{chunk} actions per chunk",
                1.0,
                {"value": chunk},
                ConfigKey.policy_actions_executed_per_chunk.value,
                30,
            )
            for chunk in (15, 20)
        )
    return ()


# Fourth-generation task cells use absolute outcomes, never products of suite ratios.
# The older materialized calibration remains an immutable replay/seal input.
def mechanism_members(faults: object) -> tuple[tuple[str, str], ...]:
    """Canonical measured mechanisms; a release has exactly its carried mechanisms.

    Only a calibration rewrite is physically inert. In particular, never erase a
    decoy or an apparently harmless policy setting from a measured context.
    """
    import json
    from collections.abc import Sequence

    from fleetlab.contracts.faults import FaultSpec
    from fleetlab.faults.software_release import ReleaseParams, carried_spec
    from fleetlab.worlds.curves import calibrated_label

    if not isinstance(faults, Sequence):
        raise TypeError("fault sequence required")
    members: list[tuple[str, str]] = []
    for f in faults:
        if not isinstance(f, FaultSpec):
            raise TypeError("FaultSpec required")
        if f.kind == "calibration_update" or f.level == 0:
            continue
        if f.kind == "software_release":
            for c in ReleaseParams.model_validate(f.params).carries:
                members.extend(
                    mechanism_members([carried_spec(c, f.target.scope, f.schedule, f.fault_id)])
                )
        else:
            label = calibrated_label(f)
            if label is None:
                label = "dose:" + json.dumps(
                    {
                        "level": f.level,
                        "params": dict(f.params),
                        "target": str(f.target.thing),
                        "nominal": f.nominal,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            members.append((f.kind, label))
    return tuple(sorted(members))


def world_task_cells(roots: object) -> dict[str, object]:
    """Read complete runs, preserving every active mechanism and deduplicating attempts.

    These are observational calibration cells, not paired interaction evidence. A
    world containing a decoy is NOT pooled with its decoy-free counterpart.
    """
    import json
    from collections import Counter
    from pathlib import Path

    from fleetlab.worlds.supplement import _task_rows, _task_signature

    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen: dict[tuple[Any, ...], tuple[int, int]] = {}
    totals: Counter[str] = Counter()
    assert isinstance(roots, (list, tuple))
    for root in roots:
        root = Path(root)
        wid = json.loads((root / "files/spec.json").read_text())["world_id"]
        for window in ("observed", "later"):
            path = root / "runs" / f"{wid}-{window}"
            for identity, outcome in _task_rows(str(path)):
                totals[f"{wid}:attempts"] += 1
                try:
                    members = mechanism_members(identity.faults)
                except KeyError:
                    totals[f"{wid}:continuous_dose"] += 1
                    continue
                key = (identity.suite, identity.task_index, members)
                attempt = (key, _task_signature(identity))
                value = (int(outcome.true_success), int(outcome.logged_success))
                if attempt in seen:
                    if seen[attempt] != value:
                        raise ValueError("conflicting duplicate measurement")
                    continue
                seen[attempt] = value
                row = cells.setdefault(
                    key,
                    {
                        "suite": identity.suite,
                        "task_index": identity.task_index,
                        "members": members,
                        "n": 0,
                        "successes": 0,
                        "logged_successes": 0,
                        "sources": {},
                    },
                )
                row["n"] += 1
                row["successes"] += value[0]
                row["logged_successes"] += value[1]
                source = row["sources"].setdefault(
                    str(path), {"n": 0, "successes": 0, "logged_successes": 0}
                )
                for field, number in (
                    ("n", 1),
                    ("successes", value[0]),
                    ("logged_successes", value[1]),
                ):
                    source[field] += number
                totals[f"{wid}:mechanisms={len(members)}"] += 1
    return {
        "method": "exact active mechanisms; no neutral-fault erasure; releases expanded; "
        "observational counts only",
        "totals": dict(totals),
        "cells": [cells[key] for key in sorted(cells)],
    }


# Doses are fixed before the queued job is admitted. Existing mid-range cells
# receive top-ups first. New overlaps remain forbidden unless the job completes
# their cell and its counterfactual cells; a budget limit never grants a fallback.
REFINEMENT_MENU = {
    "libero_spatial": (
        (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk")),
        (("camera_pose_offset", "6 deg scene pan"),),
        (("action_latency", "17 steps"),),
        (("camera_degradation", "3 px blur"),),
        (("gripper_wear", "0.25 wear"),),
        (("gripper_wear", "0.5 wear"),),
        (("gripper_wear", "0.75 wear"),),
        (("gripper_wear", "1 wear"),),
        (("object_batch_variant", "4 mass doublings"),),
        (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings")),
    ),
    "libero_object": (
        (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk")),
        (("camera_pose_offset", "3 deg scene pan"),),
        (("action_latency", "14 steps"),),
        (("camera_degradation", "3 px blur"),),
        (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings")),
    ),
}


def refinement_contexts(
    menu: tuple[tuple[tuple[str, str], ...], ...],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Include the counterfactual cells needed to remove every member of a pair.

    A precise joint with a thin single cannot support a removal forecast. Put
    subsets first so a bounded admission does not spend its last attempts on a
    joint whose prerequisites were omitted. This does not alter sealed manifests.
    """
    from itertools import combinations

    return tuple(
        dict.fromkeys(
            tuple(sorted(subset))
            for members in menu
            for size in range(1, len(members) + 1)
            for subset in combinations(members, size)
        )
    )


def refinement_plan(root: object) -> dict[str, object]:
    """Seal manifests from measured counts. CPU only; no model or simulator is loaded."""
    import hashlib
    import json
    from pathlib import Path

    from fleetlab.contracts.rollout import Job
    from fleetlab.engine import software as soft
    from fleetlab.worlds import curves
    from fleetlab.worlds.supplement import _task_collect, _task_rows, _task_signature, fault

    root = Path(str(root))
    root.mkdir(parents=True, exist_ok=False)
    record = json.loads(Path(__file__).with_name("curves_supplement.json").read_text())
    raw = record["task_calibration"]
    controls: dict[str, list[Any]] = {}
    for suite, entry in raw["healthy"].items():
        unique = {}
        for path in entry["run_dirs"]:
            for i, _ in _task_rows(path):
                if i.suite == suite and not i.faults:
                    unique[_task_signature(i)] = i
        controls[suite] = sorted(
            unique.values(), key=lambda i: (i.task_index, i.episode_index, i.scene_seed)
        )
    stages: dict[str, list[Job]] = {"thin": [], "decoys": [], "overlaps": []}
    serial = 0

    def add(stage: str, i: Any, members: tuple[tuple[str, str], ...]) -> None:
        nonlocal serial
        serial += 1
        fs = [fault(k, label, n + 1) for n, (k, label) in enumerate(members)]
        stages[stage].append(
            Job(
                identity=i.model_copy(
                    update={
                        "episode_id": f"e{serial:06d}",
                        "faults": fs,
                        "software": soft.current(i.software.miopen_db_hash),
                    }
                )
            )
        )

    selected = []
    folded = curves.measured_cells()
    for suite, menu in REFINEMENT_MENU.items():
        for members in refinement_contexts(menu):
            rows = [
                p
                for p in raw["points"]
                if p["suite"] == suite
                and tuple(sorted(map(tuple, p["members"]))) == tuple(sorted(members))
            ]
            if len(rows) != 1:
                raise ValueError(("missing or repeated calibration row", suite, members))
            row = rows[0]
            used = set(_task_collect(row["run_dirs"], suite, members))
            # Further plans must not select an identity already added by a
            # completed refinement or an observational world. Releases expand
            # to mechanisms here just as they do in the estimator's cells.
            sources = {
                path
                for prior in record.get("task_refinement", {}).get("cells", [])
                if prior["suite"] == suite and tuple(map(tuple, prior["members"])) == members
                for path in prior["sources"]
            }
            for path in sorted(sources):
                used.update(
                    _task_signature(i)
                    for i, _ in _task_rows(path)
                    if i.suite == suite and mechanism_members(i.faults) == members
                )
            for cell in row["tasks"]:
                current_cell = folded.get((suite, cell["task_index"], tuple(sorted(members))))
                if current_cell is not None and not current_cell.sufficient:
                    candidates = [
                        i
                        for i in controls[suite]
                        if i.task_index == cell["task_index"] and _task_signature(i) not in used
                    ]
                    before = current_cell.n
                    needed = 50 - before
                    if len(candidates) < needed:
                        raise ValueError("not enough disjoint healthy identities")
                    for i in candidates[:needed]:
                        add("thin", i, members)
                    selected.append(
                        {
                            "suite": suite,
                            "members": members,
                            "task": cell["task_index"],
                            "before": before,
                            "added": needed,
                        }
                    )
    # A single fleet level limits the overlap grid. Same identities in every arm.
    blur = (("camera_degradation", "3 px blur"),)
    spatial = "libero_spatial"
    for task in range(10):
        group = [i for i in controls[spatial] if i.task_index == task][:20]
        for members in (
            blur,
            (*blur, ("illumination_change", "2 stops dimmer")),
            (*blur, ("backdrop_change", "1 walls and floor")),
        ):
            for i in group:
                add("decoys", i, members)
    # All individual removals of the new triple have their own measured arm.
    companions = (
        (("action_latency", "8 steps"),),
        (("policy_knob", "15 actions per chunk"),),
        (("action_latency", "8 steps"), ("policy_knob", "15 actions per chunk")),
        (("camera_pose_offset", "6 deg scene pan"),),
        (("action_latency", "17 steps"),),
        (("gripper_wear", "0.5 wear"),),
        (("gripper_wear", "0.75 wear"),),
        (("gripper_wear", "1 wear"),),
        (("object_batch_variant", "4 mass doublings"),),
        (("gripper_wear", "0.5 wear"), ("object_batch_variant", "4 mass doublings")),
    )
    for task in range(10):
        group = [i for i in controls[spatial] if i.task_index == task][:10]
        for companion in companions:
            for i in group:
                add("overlaps", i, blur + companion)
    receipt: dict[str, Any] = {
        "selected_cells": selected,
        "stages": {},
        "budget_seconds": 5400,
        "method": "fixed doses and fixed order; no selection after outcomes; "
        "full completed stages only; 50 attempts for existing mid-range cells; "
        "new thin mid-range overlap cells remain ineligible",
    }
    chunks = {"decoys": stages["decoys"]}
    for family in ("thin", "overlaps"):
        jobs = stages[family]
        for offset in range(0, len(jobs), 400):
            chunks[f"{family}-{offset // 400 + 1}"] = jobs[offset : offset + 400]
    # Continuation identities and doses are sealed now, before any outcomes.
    # Only their preregistered inclusion rule is evaluated after a parent stage.
    expanded: dict[str, list[Job]] = {}
    parents: dict[str, str] = {}
    for name, jobs in chunks.items():
        expanded[name] = jobs
        if not (name == "decoys" or name.startswith("overlaps-")):
            continue
        followup = f"{name}-refine"
        stages[followup] = []
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for job in jobs:
            i = job.identity
            key = i.suite, i.task_index, mechanism_members(i.faults)
            groups.setdefault(key, []).append(i)
        for (suite, task, members), group in groups.items():
            prior_identities = {_task_signature(i) for i in group}
            candidates = [
                i
                for i in controls[suite]
                if i.task_index == task
                and _task_signature(i.model_copy(update={"software": group[0].software}))
                not in prior_identities
            ]
            for i in candidates[: 50 - len(group)]:
                add(followup, i, members)
        expanded[followup] = stages[followup]
        parents[followup] = name
    receipt["order"] = list(expanded)
    receipt["continuation_rule"] = (
        "Use the sealed next identities to reach 50 when the existing-plus-parent "
        "rate is in [0.15,0.85]; for a decoy task, continue all three paired arms "
        "to 50 if any arm needs it. The hard total execution cap takes priority."
    )
    for name, jobs in expanded.items():
        text = "".join(j.model_dump_json() + "\n" for j in jobs)
        (root / f"{name}.jsonl").write_text(text)
        receipt["stages"][name] = {
            "jobs": len(jobs),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if name in parents:
            receipt["stages"][name]["parent"] = parents[name]
    (root / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# Round 6c: the one further admission after the first job's import. Everything below is
# fixed before that job is admitted (README, "Round 6c preregistration"); the order is the
# priority, and a stage the budget does not reach is not calibration.
_D8 = ("action_latency", "8 steps")
_D14 = ("action_latency", "14 steps")
_D17 = ("action_latency", "17 steps")
_C15 = ("policy_knob", "15 actions per chunk")
_M4 = ("object_batch_variant", "4 mass doublings")
_BLUR3 = ("camera_degradation", "3 px blur")
_BLUR4 = ("camera_degradation", "4 px blur")
_DIMMER = ("illumination_change", "2 stops dimmer")
_BACKDROP = ("backdrop_change", "1 walls and floor")
# Existing cells between 15 and 85% with fewer than fifty attempts, brought to fifty.
# First the cells every composition needs (the interaction's single removals on both
# suites, the drift's quarter-wear knot, the batch on object tasks, a 6 degree pan on
# a robot of the mixed site); later the cells of further plain faults.
FOLLOWUP_TOPUPS: tuple[dict[str, tuple[tuple[tuple[str, str], ...], ...]], ...] = (
    {
        "libero_spatial": ((_D8,), (_C15,), (("gripper_wear", "0.25 wear"),)),
        "libero_object": (
            (_D8,),
            (_C15,),
            (_M4,),
            (("camera_pose_offset", "6 deg scene pan"),),
        ),
    },
    {
        "libero_spatial": ((_D14,), (_BLUR4,)),
        "libero_object": ((_D17,), (_BLUR4,)),
    },
)
# A decoy on a site that also carries a site-wide 17-step delay: the delay alone and the
# delay with each candidate decoy on the same attempts, twenty per spatial task, then
# the sealed continuation. The control's members come first in every arm, so that the
# control fault keeps its identity. The second set is the same under the 3-pixel blur.
FOLLOWUP_ARMS: tuple[tuple[str, str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]], ...]
FOLLOWUP_ARMS = (
    ("decoys-d17", "libero_spatial", (_D17,), (_DIMMER, _BACKDROP)),
    ("decoys-blur-d17", "libero_spatial", (_BLUR3, _D17), (_BACKDROP,)),
)
# A new joint context: ten attempts per task, then the sealed continuation.
FOLLOWUP_CONTEXTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("libero_spatial", (_D17, _M4)),
)
FOLLOWUP_ORDER = (
    "topups-1",
    "decoys-d17",
    "decoys-d17-refine",
    "contexts-1",
    "contexts-1-refine",
    "topups-2",
    "decoys-blur-d17",
    "decoys-blur-d17-refine",
)


class _Pool:
    """The canonical healthy attempts, and which of them a context has already used.

    A calibration job takes its identities from here: new for the context they measure,
    none that the base cohort or an imported refinement already used for it.
    """

    def __init__(self) -> None:
        import json
        from pathlib import Path

        from fleetlab.worlds.supplement import _task_rows, _task_signature

        self.record = json.loads(Path(__file__).with_name("curves_supplement.json").read_text())
        self.raw = self.record["task_calibration"]
        self.controls: dict[str, list[Any]] = {}
        for suite, entry in self.raw["healthy"].items():
            unique = {}
            for path in entry["run_dirs"]:
                for i, _ in _task_rows(path):
                    if i.suite == suite and not i.faults:
                        unique[_task_signature(i)] = i
            self.controls[suite] = sorted(
                unique.values(), key=lambda i: (i.task_index, i.episode_index, i.scene_seed)
            )
        self.serial = 0

    def job(self, i: Any, members: tuple[tuple[str, str], ...]) -> Any:
        from fleetlab.contracts.rollout import Job
        from fleetlab.engine import software as soft
        from fleetlab.worlds.supplement import fault

        self.serial += 1
        fs = [fault(k, label, n + 1) for n, (k, label) in enumerate(members)]
        return Job(
            identity=i.model_copy(
                update={
                    "episode_id": f"e{self.serial:06d}",
                    "faults": fs,
                    "software": soft.current(i.software.miopen_db_hash),
                }
            )
        )

    def used(self, suite: str, members: tuple[tuple[str, str], ...]) -> set[tuple[Any, ...]]:
        from fleetlab.worlds.supplement import _task_collect, _task_rows, _task_signature

        canonical = tuple(sorted(members))
        out: set[tuple[Any, ...]] = set()
        for p in self.raw["points"]:
            if p["suite"] == suite and tuple(sorted(map(tuple, p["members"]))) == canonical:
                out.update(_task_collect(p["run_dirs"], suite, canonical))
        sources = {
            path
            for prior in self.record.get("task_refinement", {}).get("cells", [])
            if prior["suite"] == suite and tuple(map(tuple, prior["members"])) == canonical
            for path in prior["sources"]
        }
        for path in sorted(sources):
            out.update(
                _task_signature(i)
                for i, _ in _task_rows(path)
                if i.suite == suite and mechanism_members(i.faults) == canonical
            )
        return out

    def fresh(self, suite: str, task: int, taken: set[tuple[Any, ...]], count: int) -> list[Any]:
        from fleetlab.worlds.supplement import _task_signature

        found = [
            i
            for i in self.controls[suite]
            if i.task_index == task and _task_signature(i) not in taken
        ]
        if len(found) < count:
            raise ValueError(("not enough unused healthy identities", suite, task, count))
        return found[:count]


def followup_plan(root: object) -> dict[str, object]:
    """Seal the follow-up job's manifests from the measured counts. CPU only.

    Identities come from the canonical healthy attempts and are new for the context
    they measure: none that the base cohort or an imported refinement already used.
    """
    import hashlib
    import json
    from pathlib import Path

    from fleetlab.contracts.rollout import Job
    from fleetlab.worlds import curves

    root = Path(str(root))
    root.mkdir(parents=True, exist_ok=False)
    pool = _Pool()
    job, used, fresh = pool.job, pool.used, pool.fresh
    folded = curves.measured_cells()

    stages: dict[str, list[Job]] = {}
    parents: dict[str, str] = {}
    selected = []
    for number, menu in enumerate(FOLLOWUP_TOPUPS, start=1):
        jobs = stages.setdefault(f"topups-{number}", [])
        for suite, contexts in menu.items():
            for members in contexts:
                canonical = tuple(sorted(members))
                taken = used(suite, members)
                for task in range(10):
                    cell = folded.get((suite, task, canonical))
                    if cell is None or cell.sufficient:
                        continue
                    needed = 50 - cell.n
                    jobs.extend(job(i, members) for i in fresh(suite, task, taken, needed))
                    selected.append(
                        {
                            "suite": suite,
                            "members": members,
                            "task": task,
                            "before": cell.n,
                            "added": needed,
                        }
                    )
    assays = []
    for name, suite, control, decoys in FOLLOWUP_ARMS:
        arms = (control, *((*control, decoy) for decoy in decoys))
        taken = used(suite, control)
        stages[name], stages[f"{name}-refine"] = [], []
        parents[f"{name}-refine"] = name
        for task in range(10):
            group = fresh(suite, task, taken, 50)
            for stage, part in ((name, group[:20]), (f"{name}-refine", group[20:])):
                for members in arms:
                    stages[stage].extend(job(i, members) for i in part)
        assays.append(
            {
                "suite": suite,
                "control": control,
                "decoys": decoys,
                "stages": [name, f"{name}-refine"],
            }
        )
    stages["contexts-1"], stages["contexts-1-refine"] = [], []
    parents["contexts-1-refine"] = "contexts-1"
    for suite, members in FOLLOWUP_CONTEXTS:
        taken = used(suite, members)
        for task in range(10):
            group = fresh(suite, task, taken, 50)
            stages["contexts-1"].extend(job(i, members) for i in group[:10])
            stages["contexts-1-refine"].extend(job(i, members) for i in group[10:])
    if set(stages) != set(FOLLOWUP_ORDER) or any(len(jobs) > 900 for jobs in stages.values()):
        raise ValueError("follow-up stages differ from their fixed order or size")
    receipt: dict[str, Any] = {
        "selected_cells": selected,
        "stages": {},
        "budget_seconds": 5400,
        "method": "fixed doses and fixed order; no selection after outcomes; full completed "
        "stages only; fifty attempts for mid-range cells; identities unused for their context",
        "order": list(FOLLOWUP_ORDER),
        "continuation_rule": (
            "Use the sealed next identities to reach 50 when the existing-plus-parent rate "
            "is in [0.15,0.85]; for a task of a decoy stage, continue every paired arm to 50 "
            "if any arm needs it. The hard total execution cap takes priority."
        ),
        "assays": assays,
    }
    for name in FOLLOWUP_ORDER:
        text = "".join(j.model_dump_json() + "\n" for j in stages[name])
        (root / f"{name}.jsonl").write_text(text)
        receipt["stages"][name] = {
            "jobs": len(stages[name]),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if name in parents:
            receipt["stages"][name]["parent"] = parents[name]
    (root / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# Round 6c, third job (README, "Round 6c, third job: preregistration"): what the recipes
# with a fleet-wide release need under its 3-pixel blur. Fixed before admission; the order
# is the priority, every stage is followed by its sealed continuation, and a stage the
# budget does not reach or does not finish is not calibration.
_W25 = ("gripper_wear", "0.25 wear")
_W50 = ("gripper_wear", "0.5 wear")
_W75 = ("gripper_wear", "0.75 wear")
_W100 = ("gripper_wear", "1 wear")
_SPATIAL, _OBJECT = "libero_spatial", "libero_object"
# name, suite, then either ("arms", control, decoys): the control context and the control
# with each decoy on the same attempts, twenty per task, continued together to fifty; or
# ("contexts", members, ...): ten attempts per task each, continued to fifty.
THIRD_STAGES: tuple[tuple[Any, ...], ...] = (
    # The decoy on a site that carries the release's blur and a site-wide 17-step delay.
    ("decoys-blur-d17", _SPATIAL, "arms", (_BLUR3, _D17), (_BACKDROP,)),
    # The interaction on a robot of the mixed site: its pair and both single removals.
    ("object-pair", _OBJECT, "contexts", (_BLUR3, _D8, _C15)),
    ("object-members", _OBJECT, "contexts", (_BLUR3, _D8), (_BLUR3, _C15)),
    # Worn pads on a robot of the mixed site: the drift's upper knots and a plain fault.
    ("object-wear", _OBJECT, "contexts", (_BLUR3, _W100), (_BLUR3, _W75)),
    # A robot with a 17-step delay at the site that carries the heavier batch.
    ("spatial-d17-batch", _SPATIAL, "contexts", (_BLUR3, _D17, _M4)),
    # The drift's lower knots on object tasks; half wear is also the additive pair's.
    ("object-wear-low", _OBJECT, "contexts", (_BLUR3, _W50), (_BLUR3, _W25)),
    # The heavier batch at the mixed site, alone and with the additive pair's worn pads.
    ("object-batch", _OBJECT, "contexts", (_BLUR3, _M4), (_BLUR3, _W50, _M4)),
    # The quarter-wear knot on spatial tasks, and milder and stronger delays on a robot.
    ("spatial-wear-low", _SPATIAL, "contexts", (_BLUR3, _W25)),
    ("object-d17", _OBJECT, "contexts", (_BLUR3, _D17)),
    ("spatial-d14", _SPATIAL, "contexts", (_BLUR3, _D14)),
    ("object-d14", _OBJECT, "contexts", (_BLUR3, _D14)),
)
THIRD_ORDER = tuple(n for stage in THIRD_STAGES for n in (stage[0], f"{stage[0]}-refine"))
THIRD_BUDGET_SECONDS = 10800


def third_plan(root: object) -> dict[str, object]:
    """Seal the third job's manifests. CPU only; identities as in `followup_plan`."""
    return context_plan(root, THIRD_STAGES, THIRD_BUDGET_SECONDS)


# Round 7 job 1: all six missing spatial wear/batch contexts, in fixed order.
ROUND7_WEAR_STAGES: tuple[tuple[Any, ...], ...] = tuple(
    (f"wear-batch-{index}", _SPATIAL, "contexts", members)
    for index, members in enumerate(
        ((wear, _M4, *cover) for wear in (_W25, _W75, _W100) for cover in ((), (_BLUR3,))),
        start=1,
    )
)


def context_plan(
    root: object, definitions: tuple[tuple[Any, ...], ...], budget_seconds: int = 10800
) -> dict[str, object]:
    """Seal a fixed ordered context menu and its predetermined continuations."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(str(root))
    root.mkdir(parents=True, exist_ok=False)
    pool = _Pool()
    stages: dict[str, list[Any]] = {}
    parents: dict[str, str] = {}
    assays = []
    order = tuple(n for stage in definitions for n in (stage[0], f"{stage[0]}-refine"))
    for name, suite, kind, *payload in definitions:
        refine = f"{name}-refine"
        stages[name], stages[refine] = [], []
        parents[refine] = name
        if kind == "arms":
            control, decoys = payload
            arms = (control, *((*control, decoy) for decoy in decoys))
            taken = pool.used(suite, control)
            for task in range(10):
                group = pool.fresh(suite, task, taken, 50)
                for stage, part in ((name, group[:20]), (refine, group[20:])):
                    for members in arms:
                        stages[stage].extend(pool.job(i, members) for i in part)
            assays.append(
                {"suite": suite, "control": control, "decoys": decoys, "stages": [name, refine]}
            )
            continue
        for members in payload:
            taken = pool.used(suite, members)
            for task in range(10):
                group = pool.fresh(suite, task, taken, 50)
                stages[name].extend(pool.job(i, members) for i in group[:10])
                stages[refine].extend(pool.job(i, members) for i in group[10:])
    if tuple(stages) != order or any(len(jobs) > 900 for jobs in stages.values()):
        raise ValueError("context stages differ from their fixed order or size")
    receipt: dict[str, Any] = {
        "stages": {},
        "budget_seconds": budget_seconds,
        "method": "fixed doses and fixed order; no selection after outcomes; full completed "
        "stages only; fifty attempts for mid-range cells; identities unused for their context",
        "order": list(order),
        "continuation_rule": (
            "Use the sealed next identities to reach 50 when the existing-plus-parent rate "
            "is in [0.15,0.85]; for a task of a decoy stage, continue every paired arm to 50 "
            "if any arm needs it. The hard total execution cap takes priority."
        ),
        "assays": assays,
    }
    for name in order:
        text = "".join(j.model_dump_json() + "\n" for j in stages[name])
        (root / f"{name}.jsonl").write_text(text)
        receipt["stages"][name] = {
            "jobs": len(stages[name]),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if name in parents:
            receipt["stages"][name]["parent"] = parents[name]
    (root / "plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def continuation_counts(
    parent: dict[tuple[Any, ...], tuple[int, int, int]],
    existing: dict[Any, Any],
    *,
    paired_decoys: bool = False,
) -> dict[tuple[Any, ...], int]:
    """Preregistered continuation, never a choice of doses after their outcomes."""
    needed = {}
    for key, (n, successes, logged) in parent.items():
        previous = existing.get(key)
        total = n + (previous.n if previous else 0)
        rates = (
            (successes + (previous.successes if previous else 0)) / total,
            (logged + (previous.logged_successes if previous else 0)) / total,
        )
        if total < 50 and any(0.15 <= rate <= 0.85 for rate in rates):
            needed[key] = 50 - total
    if paired_decoys:
        tasks = {key[:2] for key in needed}
        needed = {key: 50 - values[0] for key, values in parent.items() if key[:2] in tasks}
    return needed


def paired_stage(name: str) -> bool:
    """Stages of same-attempt decoy arms continue all their arms together.

    The first job's stage was "decoys"; later ones are named "decoys-<context>".
    """
    return name == "decoys" or name.startswith("decoys-")


def continuation_jobs(root: object, parent: str, candidates: list[Any]) -> list[Any]:
    """Apply the fixed inclusion rule to a complete, chain-checked parent."""
    from collections import Counter
    from pathlib import Path

    from fleetlab.worlds import curves
    from fleetlab.worlds.supplement import _task_rows

    counts: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    for i, o in _task_rows(str(Path(str(root)) / parent / "run")):
        key = i.suite, i.task_index, mechanism_members(i.faults)
        n, success, logged = counts.get(key, (0, 0, 0))
        counts[key] = n + 1, success + int(o.true_success), logged + int(o.logged_success)
    needed = continuation_counts(
        counts, curves.measured_cells(), paired_decoys=paired_stage(parent)
    )
    seen: Counter[tuple[Any, ...]] = Counter()
    selected = []
    for job in candidates:
        i = job.identity
        key = i.suite, i.task_index, mechanism_members(i.faults)
        if seen[key] < needed.get(key, 0):
            selected.append(job)
            seen[key] += 1
    if any(seen[key] != n for key, n in needed.items()):
        raise ValueError("sealed continuation lacks enough distinct identities")
    return selected


# A stage is started only when this bound on its duration fits the remaining budget:
# seconds per attempt at six workers and a fixed part for loading and the chain check.
# The slowest stage of the first two jobs took 2.99 s per attempt, everything included.
STAGE_SECONDS_PER_JOB = 3.2
STAGE_SECONDS_FIXED = 150.0
STAGE_WOULD_NOT_FIT = 75


def stage_fits(jobs: int, seconds: float | None) -> bool:
    """Whether a stage of this many attempts is certain to end inside the seconds left."""
    return seconds is None or jobs * STAGE_SECONDS_PER_JOB + STAGE_SECONDS_FIXED <= seconds


def refinement_run(root: object, workers: int, seconds: float = 5400) -> int:
    """One admitted heavy job, hard bounded including chain checks and shutdown.

    A stage whose bound (`stage_fits`) exceeds what is left is not started and ends the
    job: a stage that is started is meant to finish, and one that the cap still stops
    is not calibration.
    """
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    from datetime import UTC, datetime
    from pathlib import Path

    root = Path(str(root))
    # Ninety admitted GPU minutes unless the sealed plan names its own budget; no plan
    # may name more than three hours.
    limit = float(json.loads((root / "plan.json").read_text()).get("budget_seconds", 5400))
    if not 1 <= workers <= 6 or not 0 < seconds <= limit <= 10800:
        raise ValueError("at most six workers and the plan's admitted GPU budget")
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "started": datetime.now(UTC).isoformat(),
        "seconds_limit": seconds,
        "stages": [],
        "state": "running",
        "workers": workers,
    }
    path = root / "execution.json"
    if path.exists():
        raise FileExistsError("an execution receipt already exists; no implicit rerun")
    try:
        # The independent decoy arms are first, then the highest-leverage pair cells.
        for stage in json.loads((root / "plan.json").read_text())["order"]:
            remaining = seconds - (time.monotonic() - started)
            if remaining < 60:
                receipt["state"] = "budget exhausted before next stage"
                return 124
            command = [
                sys.executable,
                "-m",
                "fleetlab.worlds.calibration",
                "stage",
                "--root",
                str(root),
                "--stage",
                stage,
                "--workers",
                str(workers),
                "--seconds",
                str(remaining - 15),
            ]
            with (root / f"{stage}.log").open("x") as log:
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                began = time.monotonic()
                try:
                    status = proc.wait(timeout=max(1, remaining - 15))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    receipt["state"] = (
                        "hard execution budget exhausted; unfinished stage not calibration"
                    )
                    return 124
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    receipt["stages"].append(
                        {
                            "name": stage,
                            "seconds": round(time.monotonic() - began, 3),
                            "exit": proc.returncode,
                        }
                    )
                    receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    path.write_text(json.dumps(receipt, indent=2) + "\n")
                if status == STAGE_WOULD_NOT_FIT:
                    receipt["state"] = "budget would not cover the next stage; it was not started"
                    return 124
                if status:
                    receipt["state"] = "stage failed; inspect memory guard and chain record"
                    return status
        receipt["state"] = "complete"
        return 0
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        path.write_text(json.dumps(receipt, indent=2) + "\n")


def refinement_stage(root: object, stage: str, workers: int, seconds: float | None = None) -> int:
    import hashlib
    import json
    import os
    from pathlib import Path

    from fleetlab.contracts.rollout import Job
    from fleetlab.engine import fleet, miopen
    from fleetlab.engine import software as soft

    root = Path(str(root))
    os.environ.setdefault(
        "FLEETLAB_MIOPEN_DIR", str(Path.home() / ".cache/robots-probes/engine/miopen")
    )
    miopen.activate()
    current = soft.current(miopen.require(4, "fp16_res32"))
    text = (root / f"{stage}.jsonl").read_text()
    planned = json.loads((root / "plan.json").read_text())["stages"][stage]
    if hashlib.sha256(text.encode()).hexdigest() != planned["sha256"]:
        raise ValueError("manifest differs from preregistered plan")
    jobs = [Job.model_validate_json(line) for line in text.splitlines()]
    if any(j.identity.software != current for j in jobs):
        raise ValueError("frozen software differs from manifest")
    if "parent" in planned:
        jobs = continuation_jobs(root, planned["parent"], jobs)
        with (root / f"{stage}.selected.jsonl").open("x") as selected:
            selected.write("".join(job.model_dump_json() + "\n" for job in jobs))
    if not jobs:
        return 0
    if not stage_fits(len(jobs), seconds):
        print(f"{len(jobs)} attempts would not fit {seconds} seconds; not started")
        return STAGE_WOULD_NOT_FIT
    result = fleet.run(
        fleet.RunConfig(
            out=str(root / stage / "run"),
            workers=workers,
            batch_size=4,
            precision="fp16_res32",
            episodes_per_robot_day=1000,
        ),
        jobs,
    )
    return 0 if result.record.chain_ok is True else 1


def refinement_main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Preregistered task-cell refinement")
    parser.add_argument("action", choices=("plan", "plan-followup", "plan-third", "run", "stage"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.action == "plan":
        print(json.dumps(refinement_plan(args.root), indent=2))
        return 0
    if args.action == "plan-followup":
        print(json.dumps(followup_plan(args.root), indent=2))
        return 0
    if args.action == "plan-third":
        print(json.dumps(third_plan(args.root), indent=2))
        return 0
    if args.action == "stage":
        if not args.stage or not 1 <= args.workers <= 6:
            parser.error("stage and one to six workers required")
        return refinement_stage(args.root, args.stage, args.workers, args.seconds)
    return refinement_run(args.root, args.workers, 5400 if args.seconds is None else args.seconds)


def refinement_record(
    world_roots: object, calibration_root: object | None = None
) -> dict[str, object]:
    """Materialize only complete, chain-checked cohorts; no duplicate attempt counts.

    `calibration_root` is one root (the first job; recorded as `calibration_root`) or
    a list of roots in the order their jobs ran (recorded as `calibration_roots`).

    A folded world is compared with its own manifest, so that an attempt removed from
    every table of its runs is noticed, and the manifest with the hash stored beside it.
    A continuation is compared with the stop rule, recomputed from its parent's outcomes
    and the cells of everything before its job (the task calibration, the folded worlds
    and the earlier roots): a selected list that is one attempt short of the rule is
    refused, although every attempt in it is one of the sealed ones. That recheck takes
    the order of the roots as the order in which the jobs read the record (README,
    "Round 9": a job queued before the previous import was committed can be refused).
    """
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path

    from fleetlab.contracts.rollout import Job
    from fleetlab.contracts.world import verify_world
    from fleetlab.worlds import curves
    from fleetlab.worlds.supplement import _task_rows, _task_signature

    raw = json.loads(Path(__file__).with_name("curves_supplement.json").read_text())
    baseline = raw["task_calibration"]
    seen: dict[tuple[Any, ...], tuple[int, int]] = {}
    # Record exact baseline identities, not just source paths: arms share directories.
    for entry in [*baseline["healthy"].values(), *baseline["points"]]:
        members = tuple(sorted(map(tuple, entry.get("members", ()))))
        for path in entry["run_dirs"]:
            for identity, outcome in _task_rows(path):
                try:
                    found = mechanism_members(identity.faults)
                except KeyError:
                    continue
                if found == members:
                    seen[(found, _task_signature(identity))] = (
                        int(outcome.true_success),
                        int(outcome.logged_success),
                    )
    assert isinstance(world_roots, (tuple, list))
    paths: list[Path] = []
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    def add(path: Path) -> None:
        """Count a cohort's attempts into the cells, each physical attempt once."""
        paths.append(path)
        for identity, outcome in _task_rows(str(path)):
            try:
                members = mechanism_members(identity.faults)
            except KeyError:
                counts[f"{path}:continuous"] += 1
                continue
            key = members, _task_signature(identity)
            value = int(outcome.true_success), int(outcome.logged_success)
            if key in seen:
                if seen[key] != value:
                    raise ValueError(f"conflicting duplicate calibration attempt in {path}")
                counts[f"{path}:duplicate"] += 1
                continue
            seen[key] = value
            counts[f"{path}:added"] += 1
            cell_key = identity.suite, identity.task_index, members
            cell = cells.setdefault(
                cell_key,
                {
                    "suite": identity.suite,
                    "task_index": identity.task_index,
                    "members": members,
                    "n": 0,
                    "successes": 0,
                    "logged_successes": 0,
                    "sources": {},
                },
            )
            source = cell["sources"].setdefault(
                str(path), {"n": 0, "successes": 0, "logged_successes": 0}
            )
            for field, number in (
                ("n", 1),
                ("successes", value[0]),
                ("logged_successes", value[1]),
            ):
                cell[field] += number
                source[field] += number

    for root in world_roots:
        root = Path(root)
        wid = json.loads((root / "files/spec.json").read_text())["world_id"]
        runs = [root / "runs" / f"{wid}-{window}" for window in ("observed", "later")]
        ran = [i for run in runs for i, _ in _task_rows(str(run))]
        # The manifest is the world's own only while the files still match their hash.
        try:
            sealed_files = verify_world(root / "files")
        except OSError:
            sealed_files = False
        if not sealed_files:
            raise ValueError(f"folded world {wid}: its files differ from their stored hash")
        own = {
            j.identity.episode_id: j.identity
            for line in (root / "files/manifest.jsonl").read_text().splitlines()
            for j in [Job.model_validate_json(line)]
        }
        if len(ran) != len(own) or {i.episode_id: i for i in ran} != own:
            raise ValueError(f"folded world {wid} differs from its own manifest")
        for run in runs:
            add(run)
    several = isinstance(calibration_root, (tuple, list))
    roots = (
        [Path(str(r)) for r in calibration_root]
        if isinstance(calibration_root, (tuple, list))
        else []
        if calibration_root is None
        else [Path(str(calibration_root))]
    )
    skipped = []
    comparisons: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        plan = json.loads((root / "plan.json").read_text())
        # The table this job's stop rule read: everything counted before its root.
        before = curves.baseline_cells()
        for (suite, task, members), cell in cells.items():
            key = suite, task, tuple(sorted(members))
            old = before.get(key, curves.MeasuredCell(0, 0, 0, ()))
            before[key] = curves.MeasuredCell(
                old.n + cell["n"],
                old.successes + cell["successes"],
                old.logged_successes + cell["logged_successes"],
                (),
            )
        verified: list[Path] = []
        for stage in plan["order"]:
            manifest = root / f"{stage}.jsonl"
            payload = manifest.read_bytes()
            sealed = plan["stages"][stage]
            if hashlib.sha256(payload).hexdigest() != sealed["sha256"]:
                raise ValueError(f"manifest differs from preregistered plan: {stage}")
            text = payload.decode("utf-8")
            original = [Job.model_validate_json(line) for line in text.splitlines()]
            if len(original) != sealed["jobs"]:
                raise ValueError(f"manifest count differs from preregistered plan: {stage}")
            path = root / stage / "run"
            try:
                rows = _task_rows(str(path))
            except (ValueError, OSError):
                skipped.append(stage if not several else f"{root.name}/{stage}")
                continue
            if "parent" in plan["stages"][stage]:
                manifest = root / f"{stage}.selected.jsonl"
            planned = {
                j.identity.episode_id: j.identity
                for line in manifest.read_text().splitlines()
                for j in [Job.model_validate_json(line)]
            }
            permitted = {j.identity.episode_id: j.identity for j in original}
            if any(permitted.get(key) != value for key, value in planned.items()):
                raise ValueError(f"selected manifest differs from sealed identities: {stage}")
            if "parent" in sealed:
                parent = sealed["parent"]
                try:
                    parent_rows = _task_rows(str(root / parent / "run"))
                except (ValueError, OSError):
                    # Without its parent's outcomes the rule cannot be rechecked.
                    skipped.append(stage if not several else f"{root.name}/{stage}")
                    continue
                tally: dict[tuple[Any, ...], tuple[int, int, int]] = {}
                for i, o in parent_rows:
                    key = i.suite, i.task_index, mechanism_members(i.faults)
                    n, success, logged = tally.get(key, (0, 0, 0))
                    tally[key] = (
                        n + 1,
                        success + int(o.true_success),
                        logged + int(o.logged_success),
                    )
                needed = continuation_counts(tally, before, paired_decoys=paired_stage(parent))
                taken: Counter[tuple[Any, ...]] = Counter()
                ruled = []
                for job in original:
                    i = job.identity
                    key = i.suite, i.task_index, mechanism_members(i.faults)
                    if taken[key] < needed.get(key, 0):
                        ruled.append(i)
                        taken[key] += 1
                if ruled != list(planned.values()):
                    raise ValueError(f"selected continuation differs from the stop rule: {stage}")
            if {i.episode_id: i for i, _ in rows} != planned:
                raise ValueError(f"incomplete or changed calibration stage {stage}")
            verified.append(path)
        for path in verified:
            add(path)
        # The first job's plan names no comparisons: its two decoys under the blur.
        for assay in plan.get(
            "assays",
            [
                {
                    "suite": "libero_spatial",
                    "control": [list(_BLUR3)],
                    "decoys": [list(_DIMMER), list(_BACKDROP)],
                    "stages": ["decoys", "decoys-refine"],
                }
            ],
        ):
            comparisons.append((root, assay))
    assays = []
    for root, assay in comparisons:
        decoy_paths = [path for name in assay["stages"] if (path := root / name / "run") in paths]
        if not decoy_paths:
            continue
        grouped: dict[tuple[Any, ...], dict[tuple[Any, ...], tuple[Any, Any]]] = {}
        for decoy_path in decoy_paths:
            for i, outcome in _task_rows(str(decoy_path)):
                grouped.setdefault(mechanism_members(i.faults), {})[_task_signature(i)] = (
                    i,
                    outcome,
                )
        control_key = tuple(sorted(map(tuple, assay["control"])))
        controls = grouped[control_key]
        for kind, label in assay["decoys"]:
            full = tuple(sorted((*control_key, (kind, label))))
            treated = grouped[full]
            if controls.keys() != treated.keys():
                raise ValueError("decoy assay lacks same-attempt controls")
            tasks = []
            for task in range(10):
                pairs = [
                    (int(treated[k][1].true_success), int(v[1].true_success))
                    for k, v in controls.items()
                    if v[0].task_index == task
                ]
                tasks.append(
                    {
                        "task_index": task,
                        "n": len(pairs),
                        "paired_counts": [
                            pairs.count(pair) for pair in ((1, 1), (1, 0), (0, 1), (0, 0))
                        ],
                    }
                )
            assays.append(
                {
                    "suite": assay["suite"],
                    "members": full,
                    "decoy": [kind, label],
                    "tasks": tasks,
                    "sources": [str(path) for path in decoy_paths],
                }
            )
    where: dict[str, object] = {}
    if several:
        where = {"calibration_roots": [str(root) for root in roots]}
    elif roots:
        where = {"calibration_root": str(roots[0])}
    return {
        **where,
        "method": "absolute task-context cells; same physical attempts counted once; "
        "observational world counts never change paired interaction qualification",
        "counts": dict(counts),
        "skipped_stages": skipped,
        "cells": [cells[key] for key in sorted(cells)],
        "decoy_assays": assays,
    }


# The decoy rule's margin: the same-attempt interval of a decoy's effect on its carriers
# must include zero and exclude a loss of this size. As round 6 built it, the margin is
# two points of success on the carriers themselves, and that stays the default here. A
# recipe states its own (`WorldRecipe.decoy_margin`, `decoy_margin_of`): the fourth
# generation takes the two points as points of FLEET success, as entry 15 of the design
# note separates decoy from cause, which on the carriers is two points divided by their
# share of the fleet's attempts (`compose.Composer.decoy_margin`).
DECOY_MARGIN = 0.02


def decoy_assays(
    kind: str, label: str, jobs: tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    """Every same-attempt comparison of this decoy: alone, and in each measured context.

    With `jobs`, the comparisons as of those calibration jobs (`curves.refinement`).
    """
    return [a for a in _all_decoy_assays(jobs) if a["decoy"] == [kind, label]]


def _all_decoy_assays(jobs: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    from fleetlab.worlds import curves

    path = Path(__file__).with_name("curves_supplement.json")
    stamp = path.stat().st_mtime_ns
    if _ASSAY_CACHE.get("stamp") != stamp:
        _ASSAY_CACHE.clear()
        _ASSAY_CACHE["stamp"] = stamp
    if jobs not in _ASSAY_CACHE:
        raw = json.loads(path.read_text())
        assays = list(curves.refinement(jobs).get("decoy_assays", []))
        for row in raw["task_calibration"]["points"]:
            if len(row["members"]) == 1:
                assays.append({**row, "decoy": list(row["members"][0])})
        _ASSAY_CACHE[jobs] = assays
    found: list[dict[str, Any]] = _ASSAY_CACHE[jobs]
    return found


_ASSAY_CACHE: dict[Any, Any] = {}


def decoy_contexts(
    decoy: object, schedule: object, slots: object
) -> dict[tuple[str, tuple[tuple[str, str], ...]], dict[int, int]] | None:
    """Per suite and covering context, how often each task carries the decoy.

    None when a carrier's context has a dose between calibrated points.
    """
    from collections import Counter

    from fleetlab.contracts.faults import FaultSpec
    from fleetlab.worlds import base

    assert isinstance(decoy, FaultSpec) and isinstance(schedule, (tuple, list))
    assert isinstance(slots, (tuple, list))
    mixes: dict[tuple[str, tuple[tuple[str, str], ...]], Counter[int]] = {}
    for slot in slots:
        active = base.in_force(schedule, slot.robot_id, slot.site_id, slot.day)
        if not any(f.fault_id == decoy.fault_id for f in active):
            continue
        try:
            members = mechanism_members(active)
        except KeyError:
            return None
        mixes.setdefault((slot.suite, members), Counter())[slot.task.task_index] += 1
    return {key: dict(mix) for key, mix in mixes.items()}


def decoy_flat(
    kind: str,
    label: str,
    mixes: dict[tuple[str, tuple[tuple[str, str], ...]], dict[int, int]] | None,
    margin: float = DECOY_MARGIN,
    jobs: tuple[str, ...] | None = None,
) -> bool:
    """Flat in every covering context: one matched comparison each, flat on its task mix.

    `margin` is the loss on the carriers that every context's interval must exclude.
    """
    if not mixes:
        return False
    assays = decoy_assays(kind, label, jobs)
    for (suite, members), mix in mixes.items():
        found = [
            a
            for a in assays
            if a["suite"] == suite and tuple(sorted(map(tuple, a["members"]))) == members
        ]
        if len(found) != 1 or not decoy_mix_interval(found[0]["tasks"], mix, margin)["flat"]:
            return False
    return True


def context_decoy_confirmed(
    decoy: object,
    schedule: object,
    slots: object,
    margin: float = DECOY_MARGIN,
    jobs: tuple[str, ...] | None = None,
    *,
    scoring_days: DayRange | None = None,
) -> bool:
    """Paired flatness on the scoring-window task mix in EACH covering context.

    A context needs a matched assay for every carried task. Its task-weighted
    Newcombe interval must include zero and exclude a loss of `margin` on the
    carriers; Kish's effective sample size penalizes task mixes unlike the calibration
    allocation. This approximate interval is reported as such, not as new exact
    McNemar evidence.
    """
    from fleetlab.contracts.faults import FaultSpec
    from fleetlab.worlds import curves
    from fleetlab.worlds.recipe import SMALL_WINDOWS

    assert isinstance(decoy, FaultSpec)
    label = curves.calibrated_label(decoy)
    if label is None:
        return False
    assert isinstance(slots, (tuple, list))
    window = SMALL_WINDOWS.scoring if scoring_days is None else scoring_days
    scoring_slots = [slot for slot in slots if slot.day in window]
    return decoy_flat(
        decoy.kind, label, decoy_contexts(decoy, schedule, scoring_slots), margin, jobs
    )


def decoy_mix_interval(
    tasks: object, mix: dict[int, int], margin: float = DECOY_MARGIN
) -> dict[str, object]:
    """Weighted same-attempt difference, with an explicit effective sample size."""
    import math

    from fleetlab.dose_response import newcombe_paired

    assert isinstance(tasks, list)
    table = {t["task_index"]: t for t in tasks}
    if not mix or any(
        t not in table or table[t]["n"] < 10 or "paired_counts" not in table[t] for t in mix
    ):
        return {"flat": False, "reason": "missing paired task cell"}
    total = sum(mix.values())
    weights = {t: n / total for t, n in mix.items()}
    n_eff = 1 / sum(w * w / table[t]["n"] for t, w in weights.items())
    # Newcombe/Wilson algebra accepts positive real counts. Here these are
    # effective counts for a stratified weighted sample, explicitly approximate.
    effective = [
        sum(weights[t] * table[t]["paired_counts"][j] / table[t]["n"] for t in mix) * n_eff
        for j in range(4)
    ]
    interval = newcombe_paired(*effective)
    return {
        "flat": interval[1] >= -margin and interval[1] <= 0 <= interval[2],
        "margin": margin,
        "difference": list(interval),
        "effective_n": math.floor(n_eff),
        "method": "task-weighted paired Newcombe, Kish effective n (approximate)",
    }


if __name__ == "__main__":
    raise SystemExit(refinement_main())
