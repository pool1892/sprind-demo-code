"""python -m fleetlab.dashboard build|publish --runs RUN_DIR... [--world-roots ROOT...]
                                --out BUNDLE_DIR [options]

    build     write the page bundles: one per run, one per world root and the progress page
    publish   build, then `hub check` and `hub publish` each page; ids go to dashboard/ids.json

See README.md in this directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fleetlab.dashboard.build import BuildRefused, Options, build, publish
from fleetlab.dashboard.data import WorldRegistry
from fleetlab.dashboard.hub import HubError, Manifest
from fleetlab.dashboard.sources import gather


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fleetlab.dashboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "publish"):
        p = commands.add_parser(name)
        p.add_argument(
            "--runs",
            nargs="*",
            type=Path,
            default=[],
            metavar="DIR",
            help="run directories or exports, one page each",
        )
        p.add_argument(
            "--world-roots",
            nargs="*",
            type=Path,
            default=[],
            metavar="ROOT",
            help="world roots (fleetlab.worlds.layout), one world page each",
        )
        p.add_argument("--out", type=Path, required=True, help="the build directory")
        p.add_argument(
            "--truth",
            action="store_true",
            help="add the hidden side (faults, oracle); development worlds only",
        )
        p.add_argument("--sample", type=int, default=6, help="episodes with video per run page")
        p.add_argument("--frames", type=int, default=6, help="frames per strip")
        p.add_argument("--no-video", action="store_true", help="strips only, no video files")
        p.add_argument("--nightly", type=Path, help="Markdown of the last nightly comparison")
        p.add_argument("--no-beads", action="store_true", help="do not call bd")
        p.add_argument("--no-queue", action="store_true", help="do not read the heavy-run queue")
        p.add_argument("--manifest", type=Path, default=Path("dashboard/ids.json"))
        p.add_argument("--worlds", type=Path, default=Path("dashboard/worlds.json"))
        p.add_argument(
            "--allow-test-worlds",
            action="store_true",
            help="build pages of test and reserve worlds (after the test only)",
        )
        p.add_argument(
            "--allow-unregistered-worlds",
            action="store_true",
            help="build pages of worlds not in dashboard/worlds.json (no truth for them)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = Options(
        truth=args.truth,
        sample=args.sample,
        frames=args.frames,
        video=not args.no_video,
        allow_sealed=args.allow_test_worlds,
        allow_unregistered=args.allow_unregistered_worlds,
    )
    try:
        registry = WorldRegistry.load(args.worlds)
        manifest = Manifest.load(args.manifest)
        progress = gather(
            Path.cwd(), beads=not args.no_beads, queue=not args.no_queue, nightly=args.nightly
        )
        pages = build(
            args.runs,
            args.out,
            options,
            registry,
            manifest,
            progress,
            world_roots=args.world_roots,
        )
        for page in pages:
            print(f"built {page.key}: {page.directory}")
        if args.command == "publish":
            publish(pages, args.out, registry, manifest, progress)
            if manifest.path is not None:
                print(f"ids in {manifest.path}: commit it")
    except (BuildRefused, HubError, FileNotFoundError, ValueError) as error:
        print(f"dashboard: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
