"""Fixed round-7 calibration menus, committed before their respective admissions."""

from __future__ import annotations

import argparse
import json
from typing import Any

from fleetlab.worlds.calibration import (
    _BACKDROP,
    _BLUR3,
    _C15,
    _D8,
    _M4,
    _OBJECT,
    _SPATIAL,
    context_plan,
)

PAIR_STAGES: tuple[tuple[Any, ...], ...] = tuple(
    (f"pair-batch-{index}", _SPATIAL, "contexts", members)
    for index, members in enumerate(
        (
            (*part, _M4, *cover)
            for part in ((_D8,), (_C15,), (_D8, _C15))
            for cover in ((), (_BLUR3,))
        ),
        start=1,
    )
)
_SHIFT = ("furniture_shift", "4 cm sideways")
SITE_STAGES: tuple[tuple[Any, ...], ...] = (
    ("decoys-blur-shift", _SPATIAL, "arms", (_BLUR3, _SHIFT), (_BACKDROP,)),
    ("decoys-shift", _SPATIAL, "arms", (_SHIFT,), (_BACKDROP,)),
    ("shift-batch", _SPATIAL, "contexts", (_SHIFT, _M4)),
    ("blur-shift-batch", _SPATIAL, "contexts", (_BLUR3, _SHIFT, _M4)),
    ("object-blur-shift", _OBJECT, "contexts", (_BLUR3, _SHIFT)),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("pair", "site"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    plan = context_plan(args.root, PAIR_STAGES if args.job == "pair" else SITE_STAGES)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
