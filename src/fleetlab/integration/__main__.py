"""`python -m fleetlab.integration --dir ROOT`: the miniature fleet's chain (chain.py)."""

import sys

import fleetlab  # noqa: F401  (torch before anything that imports mujoco)
from fleetlab.integration.chain import main

if __name__ == "__main__":  # the engine's spawned workers import this module as __mp_main__
    sys.exit(main())
