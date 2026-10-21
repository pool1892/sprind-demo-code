"""The version of the contracts (docs/design/contracts.md, section 15).

A change that breaks a caller bumps it. It is recorded in
every episode's `SoftwareVersion`, so a replay can tell which contracts its
original ran under.
"""

from __future__ import annotations

CONTRACTS_VERSION = "1.0"
