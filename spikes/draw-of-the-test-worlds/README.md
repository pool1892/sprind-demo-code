# Archived scripts for the withdrawn test-world draw


> Withdrawn before the beacon on 2026-10-13. The test-world directory was empty
> and no draw process was running. The five-world campaign did not run. Do not
> resume the commands below. Script bytes and historical checksums are preserved.

A record, not a tool. The scripts here are copies, byte for byte, of what the
orchestrating session started for the draw of the five test worlds and for the steps that
follow it without a model (design note, section 10, entries 16, 21 and 22). The job ran
the copies under `~/.cache/robots-probes/orchestrator-logs/20261011/`, never the files in
this directory: a job must not read its scripts from a checkout that changes under it.

| File | What it does |
|---|---|
| `draw-and-chain.sh` | The one job. It waits for the round of the public random value, runs the draw, and then runs the chain for every kept world in the order of the places. |
| `draw-worlds.py` | The draw. It gives the ledger's commands (`src/fleetlab/worlds/README.md`, "Drawing a test world") in their order and decides nothing: every seed follows from the round's value, the recipe's hash, the place and the number of the draw. |
| `world-chain.sh` | What follows the draw for one world: the check that the files an agent sees hold none of the world's hidden terms, the oracle's replays, the baselines that need no model, the rule that analyses nothing, and the preflight of the frozen analysis checkout. |
| `w50-launch-audit.py` | The preflight that the chain calls. |
| `finish-chains.sh` | Not part of the job. Only if the job ends while the chains are running: it starts the chains still owed, world by world, with new output directories, and only for worlds that the job's status file names as kept after a complete draw. |

`sha256sum -c SHA256SUMS` in this directory checks the copies. The same sums, the command
line and everything else the draw depends on stand in the announcement
"Announcement: the draw of the five test worlds", recorded in the project's tracker before
the round.

The scripts were rehearsed twice on reserve worlds that are not part of the test (`w60` on
2026-10-11, `w62` and `w63` on 2026-10-12).
