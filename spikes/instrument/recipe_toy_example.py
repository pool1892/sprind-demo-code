"""An example result page of the probe recipe command on the probe lane's toy world (CPU).

    tools/heavy-run --light -m 2G uv run --locked python spikes/instrument/recipe_toy_example.py OUT

The learned skill and its library are the command test's (tests/test_loop_recipe_command.py);
the recipe runs on the toy world's linked robot r04 and on r01. Toy cell, not robots: the
figure carries the toy watermark.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":  # the engine spawns its workers, which re-import this module
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from test_loop_recipe_command import CLEAN, LINKED, SKILL, draft, frozen

    from fleetlab.loop.probe_toy import write_link_toy_world
    from fleetlab.loop.probes import RESULT_PAGE, recipe_check

    out = Path(sys.argv[1])
    world, setup = write_link_toy_world(out / "world-root")
    library = frozen(world, out, draft())
    recipe_check(world, setup, library.root, SKILL, [LINKED, CLEAN], out / "check")
    print(out / "check" / RESULT_PAGE)
