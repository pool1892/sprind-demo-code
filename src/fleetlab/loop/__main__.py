"""The loop's command line: the probe recipe check of story B.

    tools/heavy-run --light uv run --locked python -m fleetlab.loop check \
        LIB/recipes/NAME.json --world WORLD --robots r05,r01 --out DIR [--param name=value]

`RECIPE` is the recipe file of a frozen learned library (`loop.learned`, `recipes/<name>.json`
beside its `skills/`); the robots, comma-separated, go to the probe's robot parameters in
order. The same as `python -m fleetlab.loop.probes --library LIB --skill NAME --robot r05
--robot r01 --world WORLD --out DIR` (`loop.probes.recipe_check`): a light probe run, the
skill's evidence SQL on its export, and the result page `DIR/index.md`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    from fleetlab.loop import probes
    from fleetlab.loop.learned import RECIPES_DIR

    parser = argparse.ArgumentParser(prog="python -m fleetlab.loop")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="run a learned skill's probe recipe on named robots")
    check.add_argument("recipe", type=Path, help=f"LIB/{RECIPES_DIR}/NAME.json")
    check.add_argument("--world", type=Path, required=True)
    check.add_argument("--robots", required=True, help="comma-separated, in parameter order")
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--param", action="append", default=[], help="name=value")
    check.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    recipe = args.recipe.expanduser().resolve()
    if recipe.parent.name != RECIPES_DIR or recipe.suffix != ".json":
        parser.error(f"a recipe is LIB/{RECIPES_DIR}/NAME.json, not {args.recipe}")
    forwarded = ["--library", str(recipe.parent.parent), "--skill", recipe.stem]
    forwarded += ["--world", str(args.world), "--out", str(args.out)]
    forwarded += ["--workers", str(args.workers)]
    for robot in args.robots.split(","):
        forwarded += ["--robot", robot.strip()]
    for item in args.param:
        forwarded += ["--param", item]
    return probes.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
