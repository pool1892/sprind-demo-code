# fleetlab.dashboard: the thin dashboard

A script turns run directories into Markdown pages and publishes them to a
review hub with the `hub` CLI: one progress page, one page per run, and one
page per world root of the version-1 generator, which merges the world's two
runs. Not pretty, regenerated after every run. What the hub would need to do
better is not worked around here.

## Build and publish

```sh
make dashboard RUNS="~/.cache/robots-probes/engine/runs/mix_a ..."   # build and publish
make dashboard                                                       # the progress page only
make dashboard RUNS=... TRUTH=1                                      # with the hidden side
make dashboard WORLDS="~/.cache/robots-probes/worlds/w41" \
               RUNS="~/.cache/robots-probes/worlds/w41/runs/w41-observed ~/.cache/robots-probes/worlds/w41/runs/w41-later"

uv run --locked python -m fleetlab.dashboard build --runs RUN_DIR... --out data/dashboard
uv run --locked python -m fleetlab.dashboard publish --runs RUN_DIR... --out data/dashboard
uv run --locked python -m fleetlab.dashboard build --world-roots ROOT... --out data/dashboard
```

`build` writes the bundles and stops; `publish` builds, runs `hub check` on
each page (an error stops it) and `hub publish --project robots --tag dashboard`
(plus `--tag truth` on a page built with `--truth`). Run it from the
repository's root: it reads `dashboard/worlds.json` and `dashboard/ids.json`
there and calls `bd` there. After a publish, commit `dashboard/ids.json`.

| option | what |
|---|---|
| `--runs DIR...` | run directories or exports, one page each, named after the directory |
| `--world-roots ROOT...` | world roots (`fleetlab.worlds.layout`), one world page each, named after the world id; the world's runs get run pages only when they are also in `--runs` |
| `--out DIR` | the build directory (`make dashboard`: `data/dashboard`) |
| `--truth` | add the hidden side; refused unless every world of the run (or of the world root and its runs) is registered as development material |
| `--sample N`, `--frames N` | episodes with video per run page (6), frames per strip (6) |
| `--no-video` | strips only, no video files in the bundle |
| `--nightly FILE` | a Markdown note of the last nightly comparison, included on the progress page |
| `--no-beads`, `--no-queue` | leave out `bd` and the queue (the page says so) |
| `--allow-test-worlds` | build pages of test and reserve worlds; for after the test only |
| `--allow-unregistered-worlds` | build pages of worlds not in `dashboard/worlds.json` (never with their truth) |
| `--manifest`, `--worlds` | other paths for the two files below |

## What the pages show

```
data/dashboard/
  .dashboard               marks the directory as a dashboard build; page directories are cleared
  progress/index.md        the progress page: one bundle, one hub artifact
  runs/<run>/index.md      a run page with its PNG figures, frame strips and scene videos
  worlds/<world>/index.md  a world page with its PNG figures
```

**A run page** (`Dashboard: run <run>`): the run's state (finalized; complete
but without a table a newer log version writes; or what is partial, unmerged
or missing, from `fleetlab.log.run_status`), its world and the world's kind,
the hash-chain check and the software from `run.json` (hidden from the agents
by the contracts, but no fault truth, so shown without `--truth`); success by robot, site, day
and task with Wilson's 95% interval, by the logged outcome (what the agents
see); figures of success by robot and by day; a sample of episodes with video,
half failures, evenly spaced in episode order, each with a strip of frames
(evenly spaced logged frames from the first to the last, scene camera above the
wrist camera) and its scene video. A missing or unreadable video costs its row
of the strip or its copy, with a note, not the page. A render transient of the
chain sample (`extra.chain_checks` in `run.json`, docs/design/contracts.md
section 4) is a warning (`> [!WARNING]`, amber where the hub renders alerts)
under the facts: episode, step, camera and which rendering was the odd one. A run of a world
registered as `calibration` (`w90`) also gets the dose-response, headed and
marked as truth: success by fault kind and level, by the true outcome, against
the healthy episodes. The calibration
runs a healthy run and one run per fault kind; a kind's page without healthy
episodes of its own pools those of the other calibration runs of the same
world in the same build (`make dashboard RUNS="CAL/healthy CAL/camera_pose_offset"`)
and names them. The kinds and levels are the calibration's design, read from
the identities, so this part shows without `--truth`, for that world kind
only. The paired statistics
(McNemar, Newcombe) stay with `python -m fleetlab.dose_response analyse`.

**With `--truth`**, marked *truth* everywhere and introduced by a note at the
top: the faults active per robot, site and day as extra columns, the fault list
of each sampled episode and its true verdict, a table of the active faults with
the logged success of the episodes they were active in, the oracle's gain of
removing each fault (`contracts.oracle.fault_effect` over the episodes replayed
without it, from `runfiles.oracle_rows`), `truth/oracle.parquet` as written,
and the logged outcome against the true verdict. A fault whose level differs
between episodes (a drift) shows its range. Hidden tables a run does not have
yet read as empty, and the gains say which are missing.

**A world page** (`Dashboard: world <id>`, `world_page.py`), from a world root
(`files/`, `runs/<id>-observed`, `runs/<id>-later`, `runner.json`). Visible,
always: the world, its kind and note from `dashboard/worlds.json`; its shape
(sites, robots per site, days, the three windows and their days, episodes per
robot and day, tasks), read from `files/spec.json`'s shape fields only; each
run's state, episodes, success and hash chains, linked to its run page where
the manifest has one (a run page built in the same build but not published yet
says so), and the runner's steps and `ready` flag; success by the logged
outcome with Wilson's interval over both runs by window, site, robot and day
(site and robot also per window), a site-by-day table and a figure of success
by day, one panel per site; the health checks, a figure per component and
metric (one panel per site, one line per robot) with a table of each robot's
first, last, lowest and highest reading, from each run's `log/health.parquet`
(or `files/health.jsonl` cut to a run's days while its records are not
written); the change and maintenance logs as the observed run's records hold
them, then the entries the later run's records add after the observed window;
event counts per kind and window. A run that has not started, or cannot be
read yet, shows as such and costs nothing else.

**A world page with `--truth`**, marked *truth* and introduced by a note: the
fault schedule (fault, role from `generator.json`, kind, scope, target, level
with a drift's range over the days it is in force, onset, duration), the
patterns and small causes; the detectability list of `detectability.json`
(every cause with its kinds, roles, scopes, expected effect in points over the
three windows, observed episodes, qualifying windows, detectable, confirmed,
counts for gate 3, transferred from `libero_spatial`, assumed joint), the
count that meets gate 3's sizing rule against the recipe's minimum (six), the
neutral faults and the interaction pairs; expected success per window (logged
and true) beside the measured one, logged and true (the runs' verdicts); the
runner's summary flattened, its gate fields first; per fault and run, the
logged success of the episodes carrying it beside the mean of their expected
logged success, recomputed with `fleetlab.worlds.estimate.probability` from
the runs' identities and the generator's context rebuilt from the world's
files (a comparison of rates over the same episodes, not an effect estimate;
episodes of another world are left out and counted; a page built with other
curves than the world's says so); the generator's recipe and expectations.

**The progress page** (`Dashboard: progress`): every run of this build and of
the manifest with its state, world, success and a link to its page; the world
pages of this build and of the manifest with their kind, runner state, runs,
episodes, success and a link (`## World pages`); the worlds
with their kinds; render transients per million frames compared by the chain
samples of the runs that record them; the queue of `tools/heavy-run` (read from `/proc`: a heavy
invocation whose child is `flock` is waiting, otherwise running; light jobs do
not queue and are not listed); the gates (`Gate N` beads, closed ones with
their date), the beads waiting for a human (`bd human list --json`), the
beads in progress and the open ones (`bd list --all -n 0 --json`); the last
nightly comparison if `--nightly` names one.

## Two files in the repository

- `dashboard/worlds.json`: the kind of each world id. A page of a world not
  listed there is not built without `--allow-unregistered-worlds`, and one of a
  `test` or `reserve` world not without `--allow-test-worlds` (a test
  world the generator makes but nobody registers must not reach the hub). `--truth` is allowed for the kinds `probe`, `smoke`, `nightly`,
  `miniature`, `calibration` and `development` and refused for all others. A
  run whose world cannot be read yet (in progress, no identities) shows its
  state only. `w01` is registered as `probe` because the engine's probe runs
  and the tests' toys use it, `w90` as the calibration; register each world as
  the generator makes it.
- `dashboard/ids.json`: the hub artifact of each page (`progress`, per run its
  id, URL, run directory and world, and under `worlds` per world page its id,
  URL and world root). A page with an id is republished in
  place with `hub publish --id`; a new page's id is written as soon as the hub
  returns it. A run is refused when the manifest has it from another
  directory, since pages are named after directories, and a world when the
  manifest has it from another root. The build writes the
  known id as the entry's front matter (`id: ...`), the line `hub publish`
  would otherwise add to the generated file, so the bundle on disk is what the
  hub stores. `hub publish --json` answers with a path (`"url": "/a/<id>"`);
  the dashboard makes it absolute with the address `hub config` reports.

Published so far (2026-10-07, from the engine's probe runs, with `--truth`):
the progress page, `mix_a` and `smoke`.

## Determinism

Pages are a function of the run's files, the two repository files and, for the
progress page, of what `bd` and `/proc` say at the moment. The only clock value
is the `_Generated at ..._` line. Rows are sorted; PNG files carry no software
or date chunk; strips are raw pixels; videos are copied byte for byte. Tested:
two builds of the same runs differ in that one line only, and so do two builds
of the same world root with its truth.

A world page is published after the run pages and before the progress page;
when a run page got a new link in the same publish, the world page is rebuilt
with it first, and the progress page after a new world link.

## What the hub accepts today

Probed on 2026-10-07 with `hub check` on test bundles in the scratch directory
(nothing was published for it):

- Every file of a bundle directory is sent, whatever its type: PNG, SVG, JPEG,
  GIF, WebP, MP4, WebM, MKV, PDF, CSV, JSON, HTML, Parquet, NPZ and others were
  all counted. Files whose name starts with a dot are left out.
- The CLI skips a file larger than 25 MB (`hub: warning: skipped big.mp4
  because it is larger than 25 MB`): a file of exactly 25 MiB was kept, one of
  26 MiB skipped; a page that links the skipped file then gets a warning. The
  dashboard leaves out a video above that size and says so on the page.
- No limit on the bundle as a whole showed up: 601 files, and 26 files of 569
  MB in all, passed `hub check` (the time it took grew with the size, so the
  files are uploaded for the check). Whether `hub publish` stores bundles that
  large was not tried.
- Markdown images, HTML `<img>` tags and Markdown links are resolved against
  the bundle: a missing target is a warning in a bundle and an error in a
  single file. `<video src>` and `<source src>` are not looked at: a video tag
  pointing at a missing file gives no finding. So the check does not treat a
  video as a bundle reference, and whether the published page plays it is not
  known (below).
- `<details>`, `<script>` and `<iframe>` give no findings either; what the
  renderer keeps of them is not visible through the CLI.
- An artifact page answers 401 without credentials, so its rendered HTML could
  not be looked at from here without the hub's token, which this work does not
  read.

The run pages therefore show the frame strips as images (resolved) and embed
each scene video twice: as an HTML `<video>` tag and as a plain link to the
file in the bundle. Which of the two works in the browser was not checked from
here.

## Not verified

- How the hub renders the pages: the `<video>` tag, the links to MP4 files in
  a bundle, wide tables (the task column, the progress page's bead lists).
  `hub check` of the stored artifacts reports no findings and counts the MP4
  files among the bundle's 15 files; that is all the CLI shows.
- Run pages link to the progress page only from their second publish on: the
  progress page is published last, so its URL exists only then.
- The dose-response on a finished calibration: tested on toy runs in world
  `w90` (one mixed run; a healthy run and a kind's run built together). The
  real calibration's healthy run built (300 episodes, healthy row only); its
  first kind run was still running, so no real kind page was looked at.
- Runs of tens of thousands of episodes: the progress page reads each known
  run's `episodes.parquet` and `run_status` walks its files; neither was timed
  at that size.
- The queue on a machine where `/bin/sh` is not what runs `tools/heavy-run`, or
  where `/proc/<pid>/task/<pid>/children` is missing (older kernels). It was
  read once live (one heavy job running, one waiting) and is tested on a fake
  `/proc`.
- World pages on a real world root: built and tested on a fixture only (the
  development recipe D2 generated on the CPU, its two runs written through the
  log's writers from one episode per robot and day, outcomes drawn from the
  episode number, no simulation). The runner's summary
  (`fleetlab.worlds.summary`) did not exist yet, so its rendering is generic
  (every field flattened, the gate fields first). How the hub renders the wide
  detectability table was not looked at. The world page has not been reviewed
  independently yet.
- Reviewed independently (findings applied: a missing video or hidden table
  no longer stops the build, unregistered worlds are refused, the
  dose-response is marked as truth, the oracle gains are tested numerically).
