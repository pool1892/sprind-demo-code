"""The job contracts, re-exported from their home `fleetlab.contracts.jobs`.

They live under `fleetlab.contracts` so that the tool server, which may import only the
contracts and the log, can use them (coordinator, 19 October).
"""

from __future__ import annotations

from fleetlab.contracts.jobs import *  # noqa: F403
from fleetlab.contracts.jobs import __all__ as __all__
