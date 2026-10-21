"""The thin dashboard: a progress page and one page per run, published with the `hub` CLI.

| module | what |
|---|---|
| `data` | a run or export read for the pages: outcomes, Wilson intervals, the hidden side |
| `figures` | PNG figures (success by robot and day, the dose-response) and frame strips |
| `sources` | beads (`bd`), the heavy-job queue, the nightly note |
| `pages` | the Markdown of the run page and the progress page |
| `build` | writing the page bundles and publishing them |
| `hub` | the `hub` CLI and the manifest of artifact ids (`dashboard/ids.json`) |

`python -m fleetlab.dashboard build|publish`; see README.md in this directory.
The dashboard reads `truth/` through `fleetlab.log.runfiles`; never hand it, or
a page built with `--truth`, to an agent.
"""
