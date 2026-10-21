"""The exhibit: what an observer sees of a finished loop run (docs/design/loop.md, 12).

    view = load_run("…/engagement/engagement.json")   # or run.json, or an exhibit.json
    build(view, Path("bundle"))                        # index.md, round-N.md, provenance

Truth side: it reads the reserved evaluation's outcomes and, for a development world, the
planted truth, after the run. Never mounted for an agent.
"""

from fleetlab.loop.exhibit.records import (
    RunView,
    Sources,
    load_engagement,
    load_investigation,
    load_manifest,
    load_run,
)
from fleetlab.loop.exhibit.render import build

__all__ = [
    "RunView",
    "Sources",
    "build",
    "load_engagement",
    "load_investigation",
    "load_manifest",
    "load_run",
]
