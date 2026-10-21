"""The jobs runtime: goal-suite skills chained into one job in one simulator state.

The interfaces are in docs/design/jobs.md. The contracts
live in `fleetlab.contracts.jobs` (re-exported by `fleetlab.jobs.contracts`), so that the
tool server can import them; the five job tables are registered in
`fleetlab.contracts.log.SCHEMA`.

| module | what |
|---|---|
| `check` | the host-fixed delivery check and LIBERO's goal predicates on the loaded scene |
| `tables` | a run's records turned into the five job tables, and runs merged into one tree |
| `runner` | one `LiberoSimEnv` and the policy: programs played from starts; the batch CLI |
| `library` | the installation's library of jobs, checks and recipes; checks run over runs |
| `replay` | recorded actions re-executed from the same start, verified, re-rendered |

`python -m fleetlab.jobs` is the operator's command line (`run`, `check`, `list`,
`replay`); `python -m fleetlab.jobs.runner --batch` runs a session's queue.

Importing the package loads nothing of the simulator: `runner` and `replay` load LIBERO
when they build a scene, after `fleetlab` has put torch before mujoco.
"""

from __future__ import annotations
