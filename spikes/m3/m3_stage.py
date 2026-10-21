"""M3's Python steps, staged and never run until M3 starts (docs/design/m3-recipe.md, 9).

The launch sequence is `spikes/m3/m3_launch.sh`; this script is its part that needs the
Python API. It reads the recorded draw and never draws a seed:

    DRAW.json = {"lead":    {"seed": <LEAD_SEED>,    "world_id": "<LEAD_WORLD_ID>",
                             "public_fleet_id": "<LEAD_PUBLIC_ID>"},
                 "variant": {"seed": <VARIANT_SEED>, "world_id": ..., "public_fleet_id": ...},
                 "later":   {"seed": <LATER_SEED>,   "world_id": ..., "public_fleet_id": ...}}

    compose --draw DRAW                        CPU: the three worlds' files and m3.json
    operate --draw DRAW --fleet F              heavy: round 1's operation at 8 workers
    site --draw DRAW --fleet F                 CPU: the fleet's condition site (truth side)
    source --lead-a DIR --lead-b DIR --out F   CPU: section 4's source library, or none
    references-prep --loop L --out O           CPU: the two reference ledgers off L
    references-evaluate --world W --loop L --out O
                                               heavy: their states in O's own loop
    exhibit-extras --run R --run-exit X --build-exit Y --site S --bundle B
                                               CPU: a run's status and film, linked

The recipes come from `fleetlab.worlds.loop_fleets` (`m3_lead`, `m3_variant`, `m3_later`,
the later fleet placed by the lead's resolved condition site) and are never registered in
`RECIPES`. Every step writes its record beside its outputs; the launcher keeps the exit
receipts and the queue-log lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

FLEETS = ("lead", "variant", "later")
WORKERS = 8
PANEL_PER_ROBOT_DAY = 12
VIDEO_INDEX = 0
REFERENCE_STATES = ("classical", "build_rule")
# PENDING: the later fleet's two swap options (release-blur, arm-0.4) wait on the w71
# reuse pair (docs/design/m3-recipe.md, sections 2 and 7); w71's parts until decided.
LATER_OPTIONS: tuple[str, ...] = ()
SOURCE_RULE = (
    "lead run A's frozen library, whatever A's outcome; lead run B's only if A froze no "
    "skill; neither: no reuse pair, and 'no-source' is the reported outcome "
    "(docs/design/m3-recipe.md, section 4)"
)


# ---- the fleets ----------------------------------------------------------------------------


def recipes(draw: dict[str, Any]) -> dict[str, Any]:
    """The three M3 recipes of a recorded draw (no seed is drawn here)."""
    from fleetlab.worlds import loop_fleets as lf

    for fleet in FLEETS:
        entry = draw[fleet]
        if not isinstance(entry.get("seed"), int) or entry["seed"] < 1 << 62:
            raise ValueError(f"{fleet}: the recorded seed is missing or not a world seed")
    lead = lf.m3_lead(
        draw["lead"]["seed"], draw["lead"]["world_id"], draw["lead"]["public_fleet_id"]
    )
    variant = lf.m3_variant(
        draw["variant"]["seed"], draw["variant"]["world_id"], draw["variant"]["public_fleet_id"]
    )
    later = lf.m3_later(
        draw["later"]["seed"],
        draw["later"]["world_id"],
        draw["later"]["public_fleet_id"],
        lf.condition_site(lead),
        options=LATER_OPTIONS,
    )
    return {"lead": lead, "variant": variant, "later": later}


def _paths(recipe: Any) -> Any:
    from fleetlab.worlds.layout import WorldPaths

    return WorldPaths.default(recipe.world_id)


def compose(draw: dict[str, Any]) -> dict[str, Any]:
    """Write the three worlds' files (CPU) and the truth-side `m3.json` beside each."""
    from fleetlab.worlds import loop_fleets as lf
    from fleetlab.worlds import runner

    out = {}
    software = runner.engine_software()
    for fleet, recipe in recipes(draw).items():
        world = lf.build(recipe, software)
        paths = _paths(recipe)
        with runner.runner_lock(paths):
            value = runner.step_files(world, paths, lf.write_world)
        lf.write_beside(world, paths.root)
        record = {
            "fleet": fleet,
            "recipe": recipe.name,
            "world_id": recipe.world_id,
            "seed": recipe.seed,
            "world_hash": value,
            "condition_site": lf.condition_site(recipe),
            "later_options": list(LATER_OPTIONS) if fleet == "later" else None,
        }
        (paths.root / "m3.json").write_text(json.dumps(record, indent=1) + "\n")
        out[fleet] = record
    return out


def operate(draw: dict[str, Any], fleet: str) -> dict[str, Any]:
    """Round 1's operation of one fleet (heavy: run it under tools/heavy-run): its files,
    the observed window at 8 workers, records, events and the world export."""
    from fleetlab.worlds import loop_fleets as lf
    from fleetlab.worlds import runner

    recipe = recipes(draw)[fleet]
    world = lf.build(recipe, runner.engine_software())
    paths = _paths(recipe)
    engine = runner.EngineSettings(workers=WORKERS)
    runner.run_world(world, paths, engine, write=lf.write_world, through="observed-events")
    runner.run_world(world, paths, engine, write=lf.write_world, only="export")
    lf.write_beside(world, paths.root)
    status = lf.run_status(world, paths)
    if not (status["observed_chain_ok"] and status["export_check_clean"]):
        raise SystemExit(f"{fleet}: round 1's operation is not complete and clean: {status}")
    return status


# ---- the source library (section 4) -----------------------------------------------------------


def _freeze_state(run_dir: Path) -> dict[str, Any]:
    """What a lead run's freeze left, from its engagement record's `learned`:

    - `froze`: a completed freeze kept skills into a library that exists;
    - `empty`: a completed freeze kept no skill (no drafts, or every draft refused);
    - `error`: no engagement record, no freeze in it (the engagement stopped before its
      freeze), or an inconsistent or unavailable library. An error is never read as
      "no skill".

    The engagement's later evaluation does not matter: the freeze comes before it."""
    record = run_dir / "engagement" / "engagement.json"
    state: dict[str, Any] = {"engagement": str(record), "library": None}
    if not record.is_file():
        return {**state, "state": "error", "why": "no engagement record"}
    data = json.loads(record.read_text())
    state["engagement_outcome"] = data.get("outcome")
    learned = data.get("learned")
    if not isinstance(learned, dict):
        return {**state, "state": "error", "why": "no freeze in the engagement record"}
    library, kept = learned.get("library"), list(learned.get("kept") or [])
    state |= {"kept": kept, "refused": learned.get("refused"), "drafts": learned.get("drafts")}
    if library:
        if not Path(str(library)).is_dir():
            return {
                **state,
                "state": "error",
                "why": f"the recorded library {library} is unavailable",
            }
        if not kept:
            return {**state, "state": "error", "why": "a library recorded without kept skills"}
        return {**state, "state": "froze", "library": str(library), "why": None}
    if kept:
        return {**state, "state": "error", "why": "kept skills recorded without a library"}
    return {**state, "state": "empty", "why": "a completed freeze that kept no skill"}


def source(lead_a: Path, lead_b: Path, out: Path) -> dict[str, Any]:
    """Section 4's rule, recorded in `out`: A's library if A froze a skill; B's only if A's
    freeze completed empty and B froze one; 'no-source' only if both completed empty. Any
    error in what the rule needs stops it with a named reason (SystemExit), not
    'no-source'."""
    a, b = _freeze_state(lead_a), _freeze_state(lead_b)
    chosen: str | None = None
    error: str | None = None
    if a["state"] == "froze":
        chosen = "A"
    elif a["state"] == "error":
        error = f"lead run A: {a['why']}"
    elif b["state"] == "froze":
        chosen = "B"
    elif b["state"] == "error":
        error = f"lead run B: {b['why']}"
    outcome = "error" if error else "source" if chosen else "no-source"
    record = {
        "rule": SOURCE_RULE,
        "A": a,
        "B": b,
        "source": chosen,
        "library": a["library"] if chosen == "A" else b["library"] if chosen == "B" else None,
        "outcome": outcome,
        "error": error,
        "skipped_if_no_source": ["the reuse pair", "its scoring", "its exhibits"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    if error:
        raise SystemExit(f"source: {error}")
    return record


# ---- the references, in a loop directory of their own ---------------------------------------


def _final_ledger(loop: Path) -> Path:
    found = sorted(
        loop.glob("ledgers/ledger-r*.json"), key=lambda p: int(p.stem.removeprefix("ledger-r"))
    )
    if not found:
        raise FileNotFoundError(f"{loop}: no ledger")
    return found[-1]


def _round1_export(loop: Path) -> Path:
    found = sorted(p for p in (loop / "rounds" / "r1" / "exports").glob("start*") if p.is_dir())
    if not found:
        raise FileNotFoundError(f"{loop}: no round-1 export")
    return found[0]


def references_prep(loop: Path, out: Path) -> dict[str, Any]:
    """The two references on the engagement's round-1 export (`fleetlab.loop.references`),
    cut at its final number of changes; their ledgers go to `out/ledgers`, never into the
    engagement's loop."""
    from fleetlab.loop import references as R
    from fleetlab.loop.layout import LoopLayout
    from fleetlab.loop.state import Ledger, changes_in_force

    layout = LoopLayout.model_validate_json((loop / "layout.json").read_text())
    final = Ledger.load(_final_ledger(loop))
    k = changes_in_force(final)
    export = _round1_export(loop)
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    costs, ok = {}, {}
    for run in R.run_references(export, out / "references"):
        ledger = None
        if run.ok:
            ledger = R.reference_ledger(run, final.world_id, k, layout.adoption_day(1))
            (out / "ledgers" / f"reference-{run.state}.json").write_text(
                ledger.model_dump_json(indent=1) + "\n"
            )
        costs[run.state], ok[run.state] = R.reference_costs(run, ledger), run.ok
    record: dict[str, Any] = {
        "k": k,
        "export": str(export),
        "final_ledger": str(_final_ledger(loop)),
        "ok": ok,
        "costs": costs,
    }
    (out / "references.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    return record


def _digests(evaluation: Path) -> dict[str, str]:
    """Every file of an evaluation directory, by content: the engagement's must not move."""
    return {
        str(p.relative_to(evaluation)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(evaluation.rglob("*"))
        if p.is_file()
    }


def references_evaluate(world_root: Path, loop: Path, out: Path) -> dict[str, Any]:
    """The reference states on the engagement's panel, in `out/loop` (heavy). As on w70:
    the layout and the panel are copied from the engagement (the panel byte for byte, and
    `evaluate` mints it again and refuses a difference); the original state is read from
    the engagement's evaluation, never simulated again; and nothing of the engagement's
    evaluation is written (checked by content before and after)."""
    from fleetlab.engine.fleet import RunConfig
    from fleetlab.loop import evaluate as ev
    from fleetlab.loop.layout import LoopLayout
    from fleetlab.loop.operate import EngineSetup
    from fleetlab.loop.paths import LoopPaths
    from fleetlab.loop.state import Ledger
    from fleetlab.loop.world import LoopWorld

    engagement = loop / "evaluation"
    before = _digests(engagement)
    layout = LoopLayout.model_validate_json((loop / "layout.json").read_text())
    paths = LoopPaths(out / "loop")
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    paths.layout.write_text(layout.model_dump_json(indent=1))
    panel = paths.evaluation / "panel.json"
    if not panel.exists():
        shutil.copyfile(engagement / "panel.json", panel)
    if panel.read_bytes() != (engagement / "panel.json").read_bytes():
        raise ValueError("the references' panel is not the engagement's")
    missing = [
        n for n in REFERENCE_STATES if not (out / "ledgers" / f"reference-{n}.json").is_file()
    ]
    if missing:
        raise SystemExit(f"references: no ledger for {missing}; both are evaluated or neither")
    states = {
        name: Ledger.load(out / "ledgers" / f"reference-{name}.json") for name in REFERENCE_STATES
    }
    setup = EngineSetup(
        config=RunConfig(out="", batch_size=4, workers=WORKERS, precision="fp16_res32"),
        panel_video_indices=(VIDEO_INDEX,),
    )
    world = LoopWorld.load(world_root)
    ev.evaluate(world, layout, states, paths, setup, per_robot_day=PANEL_PER_ROBOT_DAY)
    original = ev._outcomes(engagement / "original")
    comparison = {}
    for name in states:
        mine = ev._outcomes(paths.evaluation / name)
        rescued = sum(1 for e, ok in mine.items() if ok and not original[e])
        broken = sum(1 for e, ok in mine.items() if not ok and original[e])
        comparison[name] = {
            "successes": sum(mine.values()),
            "attempts": len(mine),
            "original_successes": sum(original[e] for e in mine),
            "rescued": rescued,
            "broken": broken,
            "mcnemar_p": ev.mcnemar_exact(rescued, broken),
        }
    if _digests(engagement) != before:
        raise RuntimeError(f"{engagement} changed while the references were evaluated")
    record: dict[str, Any] = {
        "states": sorted(states),
        "against": "original, read from the engagement",
        "comparison": comparison,
        "engagement_evaluation_unchanged": True,
    }
    (out / "comparison.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# ---- the film, by the accepted rule --------------------------------------------------------


def _row(state_run: Path, episode_id: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    rows = pq.read_table(state_run / "log" / "episodes.parquet").to_pylist()
    return next(r for r in rows if r["episode_id"] == episode_id)


def clips(loop: Path, site: str, bundle: Path) -> dict[str, Any]:
    """The before-and-after film of a run's reserved panel by `fleetlab.loop.clips`'s rule
    (recorded attempts only, panel order, at the condition's site and then elsewhere),
    rendered by the exhibit's own `render_pair` into the bundle the exhibit built with
    `--no-clips` (its default selector is not used); `clips.json` and `clips.md` record
    the choice, a missing clip by its reason."""
    from fleetlab.loop.clips import RULE, select_clips
    from fleetlab.loop.exhibit.clips import ClipPair, render_pair

    evaluation = loop / "evaluation"
    chosen = select_clips(evaluation, site)
    lines = ["# Before and after, on the reserved attempts", "", f"The rule: {RULE}.", ""]
    for key, title in (("at_site", "At the condition's site"), ("elsewhere", "Elsewhere")):
        pick = chosen[key]
        lines += [f"## {title}", ""]
        if "none" in pick:
            lines += [f"No clip: {pick['none']}.", ""]
            continue
        before_run, after_run = evaluation / chosen["before"], evaluation / chosen["after"]
        pair = ClipPair(
            item=f"panel-clip-{key}",
            before_log=before_run / "log",
            before_row=_row(before_run, pick["episode_id"]),
            after_log=after_run / "log",
            after_row=_row(after_run, pick["episode_id"]),
            before_label=f"before ({chosen['before']})",
            after_label=f"after ({chosen['after']})",
            rule=RULE,
            files=(),
        )
        try:
            rendered = render_pair(pair, bundle / "clips", f"panel-{key}")
        except Exception as error:  # a broken video is reported, never replaced
            lines += [f"Attempt {pick['episode_id']}: not rendered ({error}).", ""]
            continue
        lines += [f"Attempt {pick['episode_id']}, robot {pick['robot_id']}, day {pick['day']}."]
        if rendered.gif:
            lines += ["", f"![Before and after, moving](clips/{rendered.gif})"]
        if rendered.strip:
            lines += ["", f"![Before and after, six frames each](clips/{rendered.strip})"]
        lines += ["", *(f"- [{v}](clips/{v})" for v in rendered.videos)]
        lines += [*(f"- problem: {p}" for p in rendered.problems), ""]
    (bundle / "clips.json").write_text(json.dumps(chosen, indent=1) + "\n")
    (bundle / "clips.md").write_text("\n".join(lines) + "\n")
    return chosen


def _evaluation_unavailable(evaluation: Path) -> str | None:
    """Why a run's evaluation cannot give a film, or None: the panel, the original state's
    and the final state's episode logs must exist."""
    from fleetlab.loop.clips import final_state

    if not (evaluation / "panel.json").is_file():
        return "the run has no reserved panel (its evaluation did not run)"
    for state in ("original", final_state(evaluation)):
        if not (evaluation / state / "log" / "episodes.parquet").is_file():
            return f"the state {state} was not evaluated"
    return None


def _receipt(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else "no receipt"


LINKS_MARK = "<!-- m3: status and film -->"


def exhibit_extras(
    run: Path, run_exit: Path, build_exit: Path, site: str, bundle: Path
) -> dict[str, Any]:
    """What every attempted run's exhibit shows beside what the exhibit built: its status
    (the run's exit receipt, the engagement's recorded outcome and error, the exhibit's
    own build receipt) in `status.md`, and its film by the accepted rule in `clips.md`,
    or why there is none. Both are linked from `index.md`; a bundle the exhibit could not
    build from the records gets an `index.md` that says so. Then the bundle is checked as
    the exhibit's `--check` does."""
    from fleetlab.dashboard.hub import check_bundle

    bundle.mkdir(parents=True, exist_ok=True)
    index = bundle / "index.md"
    if not index.is_file():
        index.write_text(
            f"# {run.name}\n\nThe exhibit could not be built from this run's records "
            f"(the build's exit receipt: {_receipt(build_exit)}). Its status follows.\n"
        )
    record_file = run / "engagement" / "engagement.json"
    record = json.loads(record_file.read_text()) if record_file.is_file() else None
    evaluation = run / "loop" / "evaluation"
    unavailable = _evaluation_unavailable(evaluation)
    status = {
        "run_exit": _receipt(run_exit),
        "engagement_outcome": None if record is None else record.get("outcome"),
        "engagement_error": None if record is None else record.get("error"),
        "exhibit_build_exit": _receipt(build_exit),
        "evaluation": "available" if unavailable is None else f"unavailable: {unavailable}",
    }
    lines = ["# This run's status", ""]
    lines += [
        f"- The run's exit code: {status['run_exit']}.",
        "- The engagement's recorded outcome: "
        + ("no engagement record" if record is None else str(status["engagement_outcome"]))
        + ".",
    ]
    if status["engagement_error"]:
        lines += [f"- Its recorded error: {status['engagement_error']}."]
    lines += [
        f"- The exhibit's build exit code: {status['exhibit_build_exit']}.",
        f"- The reserved evaluation: {status['evaluation']}.",
    ]
    (bundle / "status.md").write_text("\n".join(lines) + "\n")
    if unavailable is None:
        status["clips"] = clips(run / "loop", site, bundle)
    else:
        (bundle / "clips.md").write_text(
            f"# Before and after, on the reserved attempts\n\nNo clip: {unavailable}.\n"
        )
        status["clips"] = None
    text = index.read_text()
    if LINKS_MARK not in text:
        index.write_text(
            text.rstrip("\n")
            + f"\n\n{LINKS_MARK}\n## Status and film\n\n"
            + "- [This run's status](status.md)\n"
            + "- [Before and after, on the reserved attempts](clips.md)\n"
        )
    status["hub_check"] = [str(w) for w in check_bundle(bundle)]
    (bundle / "status.json").write_text(json.dumps(status, indent=1, default=str) + "\n")
    return status


# ---- command line ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spikes/m3/m3_stage.py")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compose", "operate", "site"):
        p = sub.add_parser(name)
        p.add_argument("--draw", type=Path, required=True, help="the recorded draw (JSON)")
        if name != "compose":
            p.add_argument("--fleet", choices=FLEETS, required=True)
    s = sub.add_parser("source")
    s.add_argument("--lead-a", type=Path, required=True)
    s.add_argument("--lead-b", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    for name in ("references-prep", "references-evaluate"):
        r = sub.add_parser(name)
        r.add_argument("--loop", type=Path, required=True, help="the first engagement's loop")
        r.add_argument("--out", type=Path, required=True, help="the references' own directory")
        if name == "references-evaluate":
            r.add_argument("--world", type=Path, required=True)
    x = sub.add_parser("exhibit-extras")
    x.add_argument("--run", type=Path, required=True, help="the run's output directory")
    x.add_argument("--run-exit", type=Path, required=True)
    x.add_argument("--build-exit", type=Path, required=True)
    x.add_argument("--site", required=True)
    x.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    result: Any
    if args.command == "compose":
        result = compose(json.loads(args.draw.read_text()))
    elif args.command == "operate":
        result = operate(json.loads(args.draw.read_text()), args.fleet)
    elif args.command == "site":
        from fleetlab.worlds import loop_fleets as lf

        print(lf.condition_site(recipes(json.loads(args.draw.read_text()))[args.fleet]))
        return 0
    elif args.command == "source":
        result = source(args.lead_a, args.lead_b, args.out)
    elif args.command == "references-prep":
        result = references_prep(args.loop, args.out)
        if not all(result["ok"].values()):
            failed = sorted(name for name, ok in result["ok"].items() if not ok)
            print(json.dumps(result, indent=1, default=str))
            print(f"references: {failed} failed; the set is incomplete", file=sys.stderr)
            return 1
    elif args.command == "references-evaluate":
        result = references_evaluate(args.world, args.loop, args.out)
    else:
        result = exhibit_extras(args.run, args.run_exit, args.build_exit, args.site, args.bundle)
    print(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    import fleetlab  # noqa: F401  (torch before anything that imports mujoco)

    raise SystemExit(main())
