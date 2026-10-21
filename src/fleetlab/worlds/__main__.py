"""The worlds' command line.

    uv run --locked python -m fleetlab.worlds check [D1 D2 D3]          # CPU: compose and report
    uv run --locked python -m fleetlab.worlds generate D1 [--root DIR]   # CPU: the world files
    tools/heavy-run uv run --locked python -m fleetlab.worlds run D1 [--root DIR] [--workers N]
        [--through STEP | --only STEP] [--provisional]                  # the whole world, resumable
    uv run --locked python -m fleetlab.worlds summary D1 [--root DIR]    # CPU: measured vs expected
    tools/heavy-run uv run --locked python -m fleetlab.worlds supplement run [--root DIR]
    uv run --locked python -m fleetlab.worlds supplement analyse [--root DIR]

Test and reserve worlds (design note, entry 16; README, "Drawing a test world"):

    python -m fleetlab.worlds recipe-hash                                # the recipe's hash
    python -m fleetlab.worlds ledger open --ledger LEDGER --beacon STRING # once: fixes every seed
    python -m fleetlab.worlds recipe-file --world-id W --public-fleet-id P --name T1 --out FILE
    python -m fleetlab.worlds generate --recipe FILE --root ROOT --ledger LEDGER --place T1
        # exit 0: composed and written; 3: this seed does not compose (draw again);
        # 2: refused, or the attempt failed and the draw stays open (the same command
        # again). The seed is derived from the beacon, never drawn or handed in.
    python -m fleetlab.worlds ledger summary --ledger LEDGER             # places and candidates
    python -m fleetlab.worlds ledger head --ledger LEDGER [--line N]     # lines, hashes
    python -m fleetlab.worlds ledger check --ledger LEDGER --place T1
        # entry 16's check from the runner's record: exit 0 kept, 4 discarded
    python -m fleetlab.worlds ledger verify --ledger LEDGER              # every draw again

A world's root defaults to `~/.cache/robots-probes/worlds/<world id>/`
(`fleetlab.worlds.layout`). `run` refuses a world whose interaction rests on an assumed
joint point unless `--provisional`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fleetlab  # noqa: F401  (torch before anything that imports mujoco)


def _recipe(name: str) -> Any:
    from fleetlab.worlds.recipe import DEVELOPMENT

    if name not in DEVELOPMENT:
        raise SystemExit(f"no world {name!r}; known: {', '.join(DEVELOPMENT)}")
    return DEVELOPMENT[name]


def _paths(recipe: Any, root: Path | None) -> Any:
    from fleetlab.worlds.layout import WorldPaths

    if root is None:
        return WorldPaths.default(recipe.world_id)
    return WorldPaths(root, recipe.world_id)


def _report(world: Any) -> dict[str, Any]:
    if world.recipe.kind in ("test", "reserve"):
        return {"private": True}
    d = world.detect
    comp = world.composition
    return {
        "world": world.recipe.name,
        "world_id": world.spec.world_id,
        "attempt": comp.attempt,
        "assumed_joint": comp.assumed_joint,
        "expected": d.window_success,
        "detectable_counted": len(d.counted(world.recipe.provisional)),
        "observed_counted": len(d.observed_counted(world.recipe.provisional)),
        "cause_forecast": {
            "count": len(d.counted(world.recipe.provisional)),
            "windows": ["observed", "scoring"],
            "measured": False,
            "rule": "entry 15; either-window forecast, measured causes require replay",
        },
        "predictions": world.predictions,
        "provisional": world.recipe.provisional,
        "detectable_by_rule": sum(1 for u in d.units if u.detectable),
        "qualified_by_observed": sum(
            1 for u in d.counted(world.recipe.provisional) if "observed" in u.qualified_by
        ),
        "faults": [
            {
                "fault_id": f.fault_id,
                "role": comp.roles.get(f.fault_id),
                "kind": f.kind,
                "scope": f.target.scope,
                "level": f.level,
                "onset": f.schedule.onset_day,
            }
            for f in world.spec.fault_schedule
        ],
        "causes": [
            {
                "unit": u.unit,
                "observed": round(u.effect["observed"], 2),
                "scoring": round(u.effect["scoring"], 2),
                "counts": u.counts,
            }
            for u in d.units
        ],
    }


def _refused(kind: str, reason: str) -> int:
    print(f"Creation refused before any seed was drawn ({kind}): {reason}.", file=sys.stderr)
    return 2


def _create(args: argparse.Namespace) -> int:
    """Create a test or reserve world: one draw, its seed derived, one of three outcomes.

    0: the seed composed and the world's files are written. 3: no composition within
    the recipe's drafts; the seed gives no candidate and the next draw replaces it.
    2: refused before a draw, or the attempt failed after it, with the class of the
    error. Everything that needs no seed is checked first, and an error there writes
    nothing. The seed is not drawn: it follows from the ledger's beacon, the recipe's
    hash, the place and the draw's number, and it is in the ledger before it is
    composed. Only a composition or its absence closes a draw; after a kill, an
    interrupt or an error the same command repeats the same seed. No line printed here
    shows truth.
    """
    import hashlib

    from fleetlab.contracts.world import HASH_FILE, verify_world
    from fleetlab.worlds import curves, runner
    from fleetlab.worlds import ledger as book
    from fleetlab.worlds import recipe as recipes
    from fleetlab.worlds import world as worlds
    from fleetlab.worlds.compose import NoComposition
    from fleetlab.worlds.layout import WorldPaths

    if args.ledger is None or not args.place:
        return _refused("arguments", "a test or reserve world needs --ledger and --place")
    try:
        root = recipes.private_root(args.root)
        ledger_file = book.ledger_path(args.ledger, root)
        recipes.private_path(args.recipe)
    except (ValueError, OSError):
        return _refused("path", "root, recipe and ledger lie outside every checkout and apart")
    try:
        file_digest = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
        stated = recipes.code_recipe(args.recipe)
    except (ValueError, OSError):
        return _refused("recipe", "the file is not the running code's recipe for its identity")
    try:
        record = list(curves.measured_sources(tuple(stated["calibration_jobs"])))
    except (ValueError, OSError):
        record = []
    if record != stated["calibration_sources"]:
        return _refused("calibration", "the record's digest is not the one the recipe pins")
    place, world_id = args.place, stated["world_id"]
    paths = WorldPaths(root, world_id)
    stored = root / "files" / "generator.json"
    try:
        with book.locked(ledger_file):
            try:
                ledger = book.Ledger.read(ledger_file)
            except (ValueError, OSError):
                return _refused("ledger", "the ledger cannot be read or a line does not hold")
            if ledger.beacon is None:
                return _refused("ledger", "the ledger is not open; `ledger open --beacon` first")
            draws = ledger.draws()
            pending = book.open_draw(draws)
            mine = pending is not None and pending["drawn"]["place"] == place
            if stored.is_file() and not mine:
                # The world exists: nothing is drawn and nothing new written, if it is the
                # place's candidate; anything else under this root is refused.
                try:
                    recipe = recipes.external_recipe(args.recipe, stored=stored)
                except (ValueError, OSError):
                    return _refused("recipe", "the file differs from the recipe stored there")
                if not any(
                    draw["drawn"]["place"] == place
                    and book.state_of(draw) in ("standing", "kept")
                    and book.stored_candidate(root, draw)
                    for draw in draws
                ):
                    return _refused("place", "the root holds a world that is not its candidate")
                world = worlds.generate(recipe, runner.engine_software())
                with runner.runner_lock(paths):
                    value = runner.step_files(world, paths, worlds.write_world)
                print(json.dumps({"hash": value, **_report(world)}, indent=1))
                return 0
            if not mine:
                why = book.refusal(draws, stated["kind"], place, world_id)
                if why is not None:
                    return _refused("ledger" if pending is not None else "place", why)
            try:
                code = book.code_state()
            except ValueError:
                return _refused("code", "the code's commit cannot be read; use a git checkout")
            if code["clean"] is not True:
                return _refused("code", "the tree is not clean; draw from the committed code")
            try:
                software = runner.engine_software()
            except Exception:
                return _refused("software", "the engine's software cannot be read")
            now = {
                "world_id": world_id,
                "recipe_file_sha256": file_digest,
                "recipe_hash": recipes.stated_hash(stated),
                "software": software.model_dump(mode="json"),
                "root": str(root),
            }
            if pending is not None:
                # An open draw of this place: the same seed again, if nothing it was
                # made under has changed. Otherwise nothing is written.
                drawn = pending["drawn"]
                same_commit = drawn["code"]["commit"] == code["commit"]
                if not same_commit or any(drawn[key] != now[key] for key in book.RESUMED_ON):
                    return _refused(
                        "ledger", "an open draw was made under another id, recipe, code or root"
                    )
                seed, draw_hash = int(drawn["seed"]), str(pending["hash"])
            else:
                # The draw. Its seed is derived, and on the disk before it is composed.
                number = 1 + sum(1 for draw in draws if draw["drawn"]["place"] == place)
                seed = book.derived_seed(str(ledger.beacon), now["recipe_hash"], place, number)
                draw_hash = ledger.append(
                    {
                        book.KEY: book.DRAWN,
                        "place": place,
                        "n": number,
                        "seed": seed,
                        "public_fleet_id": stated["public_fleet_id"],
                        "name": stated["name"],
                        "kind": stated["kind"],
                        "band": stated["band"],
                        "record": record,
                        "code": code,
                        **now,
                    }
                )
            line: dict[str, Any] = {"place": place, "world_id": world_id, "draw": draw_hash}
            interrupted: BaseException | None = None
            try:
                recipe = recipes.WorldRecipe.model_validate({**stated, "seed": seed})
                world = worlds.generate(recipe, software)
                hash_file = paths.files / HASH_FILE
                if hash_file.is_file() and (
                    hash_file.read_text().strip() != worlds.world_hash(world)
                    or not verify_world(paths.files)
                ):
                    # Files of another world under this root are never overwritten.
                    raise FileExistsError("the root holds other files")
                with runner.runner_lock(paths):
                    value = runner.step_files(world, paths, worlds.write_world)
            except NoComposition as error:
                line.update(
                    {book.KEY: "no_composition", "drafts": stated["attempts"]},
                    refusals=error.reasons,
                )
            except BaseException as error:
                # An error or an interrupt closes nothing: the draw stays open.
                line.update({book.KEY: book.FAILED, "error": type(error).__name__})
                if not isinstance(error, Exception):
                    interrupted = error
            else:
                line.update(
                    {book.KEY: "composed"},
                    draft=world.composition.attempt,
                    world_hash=value,
                    forecast_seal=world.composition.expected.get("forecast_seal"),
                )
            ledger.append(line)
    except book.LedgerBusy:
        return _refused("ledger", "the ledger is in use by another command")
    if interrupted is not None:
        raise interrupted
    if line[book.KEY] == "composed":
        print(json.dumps({"hash": value, **_report(world)}, indent=1))
        return 0
    if line[book.KEY] == "no_composition":
        print(
            "No composition within the recipe's drafts: this seed gives no candidate and "
            "is recorded in the ledger. Draw again for the same place.",
            file=sys.stderr,
        )
        return 3
    print(
        f"The attempt failed after the draw ({line['error']}); the draw stays open and "
        "keeps its seed. Give the same command again; truth details are withheld.",
        file=sys.stderr,
    )
    return 2


def _ledger(args: argparse.Namespace) -> int:
    """open, summary, head, check and verify. Nothing printed shows a seed or a composition."""
    from fleetlab.worlds import ledger as book
    from fleetlab.worlds.recipe import private_root

    try:
        path = book.ledger_path(args.ledger)
    except (ValueError, OSError):
        print("Refused (path): the ledger lies inside a checkout or is no file.", file=sys.stderr)
        return 2
    if args.action == "summary":
        ledger = book.Ledger.read(path, strict=False)
        print(json.dumps(book.summary(ledger), indent=1))
        return 0 if ledger.problem is None else 1
    try:
        if args.action == "open":
            if not args.beacon:
                print("Refused (arguments): opening a ledger needs --beacon.", file=sys.stderr)
                return 2
            with book.locked(path):
                try:
                    lines, last = book.open_ledger(path, args.beacon).head()
                except FileExistsError as error:
                    print(f"Refused (ledger): {error}.", file=sys.stderr)
                    return 2
                except ValueError as error:
                    print(f"Refused (arguments): {error}.", file=sys.stderr)
                    return 2
            print(json.dumps({"lines": lines, "head": last}))
            return 0
        if args.action == "check":
            if not args.place:
                print("Refused (arguments): the check needs --place.", file=sys.stderr)
                return 2
            with book.locked(path):
                ledger = book.Ledger.read(path)
                try:
                    root = None if args.root is None else private_root(args.root)
                    shown = book.check(ledger, args.place, root)
                except LookupError as error:
                    print(f"Refused (place): {error}.", file=sys.stderr)
                    return 2
                except ValueError as error:
                    if isinstance(error, book.BrokenLedger):
                        raise
                    print(f"Refused (world): {error}.", file=sys.stderr)
                    return 2
            print(json.dumps(shown))
            return 0 if shown["decision"] == "kept" else 4
        ledger = book.Ledger.read(path)
    except book.LedgerBusy:
        print("Refused (ledger): the ledger is in use by another command.", file=sys.stderr)
        return 2
    except (book.BrokenLedger, OSError) as error:
        print(f"Refused (ledger): {error}.", file=sys.stderr)
        return 2
    if ledger.beacon is None:
        print(
            "Refused (ledger): the ledger has no beacon line; it was never opened.", file=sys.stderr
        )
        return 2
    if args.action == "head":
        lines, last = ledger.head()
        shown = {"lines": lines, "head": last}
        if args.line is not None:
            if not 1 <= args.line <= lines:
                print("Refused (arguments): the ledger has no such line.", file=sys.stderr)
                return 2
            shown.update(line=args.line, hash=book.line_hash(ledger.texts[args.line - 1]))
        print(json.dumps(shown))
        return 0
    # Where it runs is part of what it says: the ledger's commit, and a clean tree.
    try:
        code = book.code_state()
    except ValueError:
        code = {"commit": None, "clean": False}
    if code["clean"] is not True or ledger.commit not in (None, code["commit"]):
        print(
            "Refused (code): verify runs from a clean checkout of the commit the ledger's "
            "draws name.",
            file=sys.stderr,
        )
        return 2
    try:
        counts = book.verify(ledger)
    except book.RecordChanged as error:
        print(f"Refused (record): {error}; nothing was composed.", file=sys.stderr)
        return 2
    print(json.dumps(counts, indent=1))
    bad = ("differ", "recipe_differs", "checks_differ", "open")
    return 1 if any(counts[name] for name in bad) else 0


def _recipe_file(args: argparse.Namespace) -> int:
    """Write the code's recipe for one identity to a new file outside every checkout."""
    from fleetlab.worlds.recipe import private_path, private_recipe

    try:
        path = private_path(args.out)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        stated = private_recipe(args.world_id, args.public_fleet_id, name=args.name, kind=args.kind)
        with path.open("x") as handle:
            handle.write(json.dumps(stated, indent=1) + "\n")
    except FileExistsError:
        print("Refused (path): the file exists; a recipe file is not replaced.", file=sys.stderr)
        return 2
    except (ValueError, OSError):
        print(
            "Refused (path or identity): a recipe file lies outside every checkout, in an "
            "existing directory, and names a valid world id, fleet id and name.",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.worlds")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="compose worlds on the CPU and report the gate's sizing")
    check.add_argument("names", nargs="*", default=["D1", "D2", "D3"])
    check.add_argument("--fake-software", action="store_true", help="no engine software needed")
    check.add_argument(
        "--provisional",
        action="store_true",
        help="let causes on transferred or assumed points count (before the supplement)",
    )
    gen = sub.add_parser("generate", help="write a world's files (CPU)")
    gen.add_argument("name", nargs="?")
    gen.add_argument(
        "--recipe", type=Path, help="private test recipe outside this checkout; requires --root"
    )
    gen.add_argument("--root", type=Path)
    gen.add_argument("--ledger", type=Path, help="the ledger of draws; required with --recipe")
    gen.add_argument("--place", help="the place the draw is for (T1 to T5); with --recipe")
    led = sub.add_parser("ledger", help="the record of draws for test and reserve worlds")
    led.add_argument("action", choices=["open", "summary", "head", "check", "verify"])
    led.add_argument("--ledger", type=Path, required=True)
    led.add_argument("--beacon", help="open: the public random string every seed follows from")
    led.add_argument("--line", type=int, help="head: also print the hash of this line")
    led.add_argument("--place", help="check: the place whose candidate ran")
    led.add_argument("--root", type=Path, help="check: optional; must be the draw's own root")
    rfile = sub.add_parser("recipe-file", help="write a place's recipe file from the code")
    rfile.add_argument("--world-id", required=True)
    rfile.add_argument("--public-fleet-id", required=True)
    rfile.add_argument("--name", required=True)
    rfile.add_argument("--kind", choices=["test", "reserve"], default="test")
    rfile.add_argument("--out", type=Path, required=True)
    rhash = sub.add_parser("recipe-hash", help="the hash of the recipe test worlds are drawn from")
    rhash.add_argument("--kind", choices=["test", "reserve"], default="test")
    run = sub.add_parser("run", help="run a whole world (heavy, resumable)")
    run.add_argument("name", nargs="?")
    run.add_argument(
        "--recipe", type=Path, help="the external template used at creation; requires --root"
    )
    run.add_argument("--root", type=Path)
    run.add_argument("--workers", type=int, default=10)
    run.add_argument("--through")
    run.add_argument("--only")
    run.add_argument("--provisional", action="store_true", help="run a provisional world")
    summ = sub.add_parser("summary", help="measured against expected (CPU)")
    summ.add_argument("name", nargs="?")
    summ.add_argument("--recipe", type=Path)
    summ.add_argument("--root", type=Path)
    supp = sub.add_parser("supplement", help="the libero_object and interaction calibration")
    supp.add_argument("action", choices=["run", "analyse"])
    supp.add_argument("--root", type=Path)
    supp.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    if args.command == "recipe-hash":
        from fleetlab.worlds.recipe import recipe_hash

        print(recipe_hash(args.kind))
        return 0
    if args.command == "ledger":
        return _ledger(args)
    if args.command == "recipe-file":
        return _recipe_file(args)

    from fleetlab.worlds import runner
    from fleetlab.worlds.world import generate

    if args.command == "supplement":
        from fleetlab.worlds import supplement

        root = args.root or supplement.DEFAULT_ROOT
        if args.action == "run":
            return supplement.run(root, args.workers)
        doc = supplement.analyse(root)
        print(json.dumps({"points": len(doc["points"]), "joints": len(doc["joints"])}))
        return 0

    if args.command == "check":
        failed = 0
        for name in args.names:
            recipe = _recipe(name)
            if args.provisional:
                recipe = recipe.model_copy(update={"provisional": True})
            if args.fake_software:
                from fleetlab.contracts.identity import SoftwareVersion

                software = SoftwareVersion(
                    lock_hash="0" * 16,
                    libero_patch_set="1" * 16,
                    mujoco_version="3.3.7",
                    miopen_db_hash="2" * 16,
                )
            else:
                software = runner.engine_software()
            world = generate(recipe, software)
            report = _report(world)
            if not args.fake_software:
                # Replay rule 7: every job's images per call must be in the pinned database,
                # or the engine refuses the run at its start.
                from fleetlab.engine.fleet import check_images

                check_images(runner.EngineSettings().config(Path("/nonexistent"), 1), world.jobs)
                report["images_per_call_pinned"] = True
            print(json.dumps(report, indent=1))
            if report["cause_forecast"]["count"] < recipe.min_detectable or recipe.provisional:
                failed += 1
        return 1 if failed else 0

    if args.recipe is not None:
        from fleetlab.worlds.recipe import external_recipe, private_root

        if args.name is not None or args.root is None:
            parser.error("--recipe requires --root and no development-world name")
        if args.command == "generate":
            return _create(args)
        try:
            args.root = private_root(args.root)
            stored = args.root / "files" / "generator.json"
            if args.command != "generate" and not stored.is_file():
                parser.error("create the private world with generate --recipe before resuming it")
            recipe = external_recipe(args.recipe, stored=stored)
        except (ValueError, OSError):
            print(
                "Private recipe or output path is invalid; truth details withheld.", file=sys.stderr
            )
            return 2
    else:
        if args.name is None:
            parser.error("supply a development-world name or --recipe")
        recipe = _recipe(args.name)
    if getattr(args, "provisional", False):
        recipe = recipe.model_copy(update={"provisional": True})
    paths = _paths(recipe, args.root)
    if args.command == "summary":
        from fleetlab.worlds.summary import summary

        world = generate(recipe, runner.engine_software())
        report = summary(world, paths)
        if recipe.kind in ("test", "reserve"):
            report = {"private": True, "ready": report.get("ready", False)}
        print(json.dumps(report, indent=1, default=str))
        return 0
    try:
        world = generate(recipe, runner.engine_software())
    except (ValueError, KeyError):
        if recipe.kind not in ("test", "reserve"):
            raise
        print("Private composition failed qualification; truth details withheld.", file=sys.stderr)
        return 2
    if args.command == "generate":
        from fleetlab.worlds.world import write_world

        with runner.runner_lock(paths):
            value = runner.step_files(world, paths, write_world)
        print(json.dumps({"hash": value, **_report(world)}, indent=1))
        return 0
    shaky = [u.unit for u in world.detect.counted(True) if u.transferred or u.assumed]
    if (world.composition.assumed_joint or shaky) and not args.provisional:
        print(
            "Private world lacks measured support."
            if recipe.kind in ("test", "reserve")
            else f"{args.name}: causes {shaky} rest on libero_spatial points applied to "
            "libero_object or on an assumed joint point; run the supplement first (make "
            "supplement) or pass --provisional",
            file=sys.stderr,
        )
        return 2
    record = runner.run_world(
        world,
        paths,
        runner.EngineSettings(workers=args.workers),
        through=args.through,
        only=args.only,
    )
    print(json.dumps({k: record.get(k) for k in ("world_id", "ready")}, indent=1))
    return 0 if record.get("ready") or args.through or args.only else 1


if __name__ == "__main__":
    raise SystemExit(main())
