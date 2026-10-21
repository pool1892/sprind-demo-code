"""Build an exhibit bundle from a finished run, and check it with the review site's renderer.

    uv run --locked python -m fleetlab.loop.exhibit build RUN --out DIR [--check] [--no-clips]

RUN is an `engagement.json`, an investigation's `run.json`, an `exhibit.json` manifest, or
a directory holding one of them. `--truth-world ROOT` adds the planted truth of a
development world (refused unless `dashboard/worlds.json` lists the world as development
material). `--check` runs `hub check` on the bundle; nothing is published.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fleetlab.loop.exhibit.records import Sources, load_run, load_truth
from fleetlab.loop.exhibit.render import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop.exhibit")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the bundle")
    b.add_argument("run", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--truth-world", type=Path, default=None)
    b.add_argument("--worlds", type=Path, default=Path("dashboard/worlds.json"))
    b.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    b.add_argument("--no-clips", action="store_true")
    b.add_argument("--check", action="store_true", help="run hub check on the bundle")
    args = parser.parse_args(argv)

    relocate = dict(item.split("=", 1) for item in args.relocate)
    view = load_run(args.run, sources=Sources(relocate=relocate))
    if args.truth_world is not None or view.truth is not None:
        from fleetlab.dashboard.data import WorldRegistry

        registry = WorldRegistry.load(args.worlds)
        if not registry.truth_allowed(view.world_id):
            print(
                f"refused: world {view.world_id} is {registry.kind(view.world_id)!r} in "
                f"{args.worlds}; its truth may not be shown",
                file=sys.stderr,
            )
            return 2
    if args.truth_world is not None:
        view.truth = load_truth(
            view.sources, args.truth_world.expanduser().resolve(), view.world_id
        )
    if view.truth is not None:
        from fleetlab.dashboard.data import WorldRegistry

        view.truth.kind = WorldRegistry.load(args.worlds).kind(view.world_id)
    out = build(view, args.out, clips=not args.no_clips)
    print(out)
    if args.check:
        from fleetlab.dashboard.hub import HubError, check_bundle

        try:
            warnings = check_bundle(out)
        except HubError as error:
            print(f"hub check failed: {error}", file=sys.stderr)
            return 1
        for warning in warnings:
            print(f"hub check: {warning}")
        print("hub check: no content errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
