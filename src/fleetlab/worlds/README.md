# fleetlab.worlds: calibrated fleet composition and execution

Current handoff: **`w48` (D2) and `w49` (D3), the fourth generation's development
worlds, are composed and sealed; this builder ran neither. The template of the test
worlds composes on 4 of 20 seeds, short of the 15 it needs.** Three settings decided
by the orchestrating session on 2026-10-10, and three plain faults per mechanism and
kind of scope for the recipes with a fleet-wide release, made the two worlds possible;
all four are fields of the recipe. The w46/w47 records remain unchanged. Earlier
deliveries below are historical.

## Loop fleets: designed fleets for the two-round loop (2026-10-13, lane fleet)

`fleetlab.worlds.loop_fleets` builds the development fleets of the feasibility
demonstration's loop (`docs/design/loop.md`). It states a fleet's fault schedule
instead of searching for one: the composer below admits a world by a forecast, and its
measured estimator refuses every joint context nobody measured, while a loop fleet is
designed, each fault with its part in the story. Everything else is the version-1
generator's, unchanged (`world.changes`, `world.maintenance`, `health.rows`,
`base.slots`, `world.jobs`, the hashed world files), and `generator.json` names the
version-1 generator, so `fleetlab.loop.world.LoopWorld` derives health under a ledger and
the reserved panel's slots with the same functions. The world's windows are the loop's:
observed = round 1's operation (days 0-5), experiment = round 1's experiments (6-7),
scoring = round 2's operation (8-11); `layout_for_world` appends round 2's experiments
(12-13) and the evaluation (14-17). Twelve attempts per robot and day.

```sh
uv run --locked python -m fleetlab.worlds.loop_fleets check LEAD --fake-software  # CPU: design
uv run --locked python -m fleetlab.worlds.loop_fleets generate LATER               # CPU: files
tools/heavy-run uv run --locked python -m fleetlab.worlds.loop_fleets run LEAD     # round 1
tools/heavy-run --light uv run --locked python -m fleetlab.worlds.loop_fleets export-r1 LEAD --out DIR
uv run --locked python -m fleetlab.worlds.loop_fleets report LEAD                  # tables
```

`run` is round 1's operation only: files, the observed window, its records and events,
and the world export with its leak audit. The world's later run is not part of the
loop and is not run, so `runner.json` never says `ready` (the oracle's and part one's
marker); the export's own receipt is what `run_status` reads. `export-r1` builds round
1's export as the loop does (`fleetlab.loop.export.export_round`, empty ledger) with
its pre-export checks. Beside `files/`, outside what the leak audit reads as world
files, `design.json` holds the as-composed truth table and the design forecast (measured
cells: the exact joint where measured, else the members' ratios; guidance, the run
decides), `loop_layout.json` the layout, `report.json` what the run measured.

- **LEAD** (hidden `w70`, public `w01`): s1's four robots driven over a remote link, 17
  control steps, from day 0 and lasting (spatial tasks only); s2's release 4.3.0 (a 4 px
  scene blur) on the day its scene calibration record was rewritten (the coincidence);
  s3's tables moved 2 cm on the day s3 got the harmless release 4.3.1 (the decoy); under
  each site's first cause a second waits: r03's worn pads (behind the link), r06's
  14-step command buffer (logged, behind the release), r10's knocked scene camera (behind
  the tables). Five repairs and one adaptation; nothing starts after day 3.
- **LATER** (hidden `w71`, public `w02`), for the reuse comparison: the same kind of
  link at s3, 14 steps, object tasks; s1 carries the same site note over an 8-step link
  measured flat on its spatial tasks and drops on day 2 for release 4.4.0 (a 24-step scene
  camera lag); s2 is worse from day 0 for heavier objects beside a harmless sampling-steps
  change; r06's arm loses torque from day 2.

The recipes are truth side: the agents' tool bundle cannot import this module (it
imports `fleetlab.contracts.world` and `fleetlab.faults`, both forbidden there).

## Round 6c, continued: three settings, a third calibration job, `w48` and `w49` sealed (2026-10-10)

Round 6c had stopped because no recipe composes under the rules as built (the next
section). The orchestrating session then decided three settings for the fourth
generation's development worlds. This part of the round makes them fields of the
recipe, composes and seals `w49` under them, measures what the recipes with a
fleet-wide release lack, and composes and seals `w48`. Commits: `b02328c` (the third
job's plan, sealed before admission), `cc454da` (the three settings; `w49` sealed),
`e2a8939` (three plain faults per mechanism and scope for the fleet recipes),
`c644339` (import of the third job), `659a780` (`w48` sealed; the assessment of the
fleet recipes), and this note.

### The three settings, and which recipe uses which

Each is a field of `WorldRecipe`, so that a recipe states the rule it was composed
under. The fields' defaults are the rules as round 6 built them; nothing was changed in
place, and a recipe of the older estimators cannot set them.

| | Floor per robot and window (`robot_success_floor`, plus a reserve of 0.03) | Decoy rule's margin (`decoy_margin`, `decoy_margin_of`) | Retained forecast error (`retained_error`) |
|---|---|---|---|
| Round 6 as built: the fields' defaults and the candidates until `02e3275` | 0.40 | two points of success on the decoy's own attempts (`0.02`, `"carriers"`) | the third generation's (`"generation-3"`): minus 0.7 to plus 6.0 points on the observed window |
| Fourth generation: `planned_recipe("D2")`, `("D3")`, `("planned")` and the sealed `DEVELOPMENT["w48"]`, `["w49"]` | 0.30 | two points of fleet success (`0.02`, `"fleet"`) | none (`"none"`) |
| `w46`, `w47` (third generation, `DEVELOPMENT["D2"]`, `["D3"]`, estimator `suite-v1`) | 0.40 per robot and day, as sealed | not applicable: no measured contexts | not applicable |
| Test worlds | open | two points of fleet success | to be the error that `w48` and `w49` show |

- **The decoy rule's margin.** A decoy is confirmed where the same-attempt interval of
  its effect, on the task mix of its carriers and under whatever else covers them,
  includes zero and excludes a loss of the margin. The design note's entry 15
  separates decoy from cause at two points of fleet success over the scoring window,
  so with `"fleet"` the margin on the carriers is two points divided by their share of
  that window's attempts: six points for a decoy on one of three sites
  (`Composer.decoy_margin`). Zero must still lie inside, and a context without a
  same-attempt comparison is still refused. Under the 3-pixel blur the changed
  backdrop passes at six points (plus 0.4, interval minus 2.4 to plus 3.2) and the
  dimmer light stays refused (minus 5.7, interval minus 9.8 to minus 2.1: zero is
  outside). Under a site-wide 17-step delay the backdrop is confirmed on 5 of the 40
  spatial sites of the twenty assessment seeds at two points on the carriers and on
  all 40 at two points of the fleet.
- **The retained error.** `estimate.RETAINED_ERRORS` names the errors a forecast's
  acceptance interval can carry. `"generation-3"` is what `w46` and `w47` showed and
  stays the default. `"none"` is for worlds that measure the forecast's error anew:
  they are accepted when the forecast with its own spread (the calibration cells'
  uncertainty and the chance of a window's attempts) lies inside 60 to 75%. What
  `w48` and `w49` show is to become a further entry, for the test worlds.
- **The floor.** 0.30 per robot and window instead of 0.40, with the reserve of 0.03
  unchanged, so a robot's forecast in a window is at least 33%. The band of 60 to 75%
  is frozen; the floor was added after the freeze (entry 17) and gives way where both
  cannot be met.

`tests/test_worlds_settings.py` shows each rule at both values: the margin on a
synthetic comparison, on the measured decoys under the blur and through the composer
on the twenty seeds; the interval with and without the error, and admission at 65, 68
and 72%; a robot window at 32, 36 and 44% under both floors.

The search's target (`aim`) for the fourth generation is 64.0 to 71.5%: with no error
retained the forecast's own spread, about 3.5 points down and 2.5 up, stays inside the
band from both ends. The interval remains the admission criterion.

### D3: the assessment, and why the recipe is still D3

`planned_recipe("D3")` now has its interaction on one robot and four plain faults; the
third generation had the interaction on a whole site and five plain faults. With the
interaction on a site no seed composes on measured contexts: its four robots would
each need the pair and both single removals measured under everything else on the
site, and four robots near 60% beside a decoy's site leave the band no room. What
section 1 of the design note asks of D3 is unchanged: one confounder pair, one decoy
(it allows one or two), one interaction, one drift, and at least two thirds of the
faults from configuration and software; besides these the additive pair of entry 2.
Section 1 names neither the interaction's scope nor the number of plain faults; it
plans about ten active faults, and `w49` has ten besides its decoy and its inert
entry.

On the twenty assessment seeds `(1 << 62) + 1100000 + k`, with 400 drafts each, **D3
composes on 16**. The sixteen forecasts of the observed window lie between 66.8 and
71.4%, their intervals between 63.4 and 73.7%; six measured causes hold in at least
99.6% of the resamples; the first accepted draft is among the first twelve on nine
seeds and later than the hundredth on three. The four seeds that do not compose:

| Seed | What refused its 400 drafts |
|---|---|
| 3 | 181 forecasts outside the target, 216 with fewer than four plain faults on measured contexts, 3 with a decoy not confirmed under what covers it |
| 6 | 388 with no carrier for the interaction above the floor (its mixed site favours object tasks, where the pair leaves 28%), 11 short of plain faults, 1 outside the target |
| 7 | 141 outside the target, 258 short of plain faults, 1 without a confounder |
| 13 | 77 outside the target, 323 short of plain faults |

### `w49`: composed and sealed, not run

`DEVELOPMENT["w49"]` is `planned_recipe("D3")` with a private seed, the two calibration
jobs it stands on (`calibration_jobs`), their pinned digest and the world's hash. The
seed was drawn from the operating system's random source. Two draws were needed: the
first did not compose (385 of 400 drafts found no carrier for the interaction, as on
assessment seed 6) and was dropped before anything else of it was looked at; the
second composed on its eighth draft. No outcome of any run was involved. The world:
12 robots at 3 sites, 10 days, 1,680 planned attempts (1,347 on spatial tasks, 333 on
object tasks), neutral fleet id `w01`, opaque fault identities.

| What | Where | From day |
|---|---|---|
| Confounder: a 17-step action delay, with an inert calibration record on one robot the same day | site s1 (the record on r01) | 1 |
| Decoy: the light two stops dimmer | site s1 | 2 |
| Additive pair: objects four mass doublings heavier, and pads worn to 0.5 on one robot | site s3; r11 | 1; 0 |
| Interaction: an 8-step delay, then 15 actions per chunk | r07 (mixed site s2) | 0; 1 |
| Drift: pad wear rising linearly by 0.11 per day | r05 (s2) | 0 |
| Plain: a 17-step delay | r06 (s2) | 2 |
| Plain: a software release carrying a 3-pixel blur | r09 (s3) | 2 |
| Plain: a software release carrying a 17-step delay | r10 (s3) | 4 |
| Plain: a 17-step delay | r12 (s3) | 4 |

Eight of the eleven entries besides the decoy are configuration or software (73%).

The sealed forecast, on the outcome the fleet logs, with the forecast's own 95%
interval and no retained error:

| Window | Attempts | Forecast | Interval |
|---|---:|---:|---:|
| Observed (days 0 to 5) | 1,008 | 71.01% | 67.36 to 73.52% |
| Experiment (days 6 to 7) | 336 | 62.78% | 57.14 to 67.56% |
| Scoring (days 8 to 9) | 336 | 65.37% | 59.52 to 70.19% |

Only the observed window is held inside 60 to 75% by rule; the later windows are
forecast, with a floor of 53% on the scoring window's expectation.

Measured causes, forecast for the scoring window (removing one fault alone, the others
staying): **seven** faults gain at least two points of fleet success, and at least six
do in 99.95% of the resamples of the calibration cells. In points: the site-wide delay
9.9 (interval 8.9 to 11.1), the heavier batch 6.2, the delay on r06 5.2, the two
members of the interaction 3.5 each, the release with the delay on r10 3.3, the delay
on r12 2.9. Not causes by the forecast: the worn pads of the additive pair (1.1), the
drift (0.5 in the scoring window; its own robot crosses five points there, as the
rule on drifts asks), the release with the blur on r09 (minus 0.1: under the heavier
batch the blur costs nothing measurable) and the decoy (minus 0.1, interval minus 1.4
to plus 1.4).

The lowest robot-window is r12 in the scoring window at 33.8% (the floor with its
reserve is 33%); single days go lower, r06 on day 4 to 23%. The three largest
single-fault fixes by scoring gain are the site-wide delay at s1 (9.9 points,
configuration, in the change log), the heavier batch at s3 (6.2 points, a physical
condition, visible only in behaviour) and the delay on r06 (5.2 points, configuration,
in the change log): two of the three are logged settings.

The seals, in `tests/worlds_generation4_seals.json` and checked by
`tests/test_worlds_generation4.py` on every run of the tests:

- world hash `d9aa350ba380f0e080b03859822907f0bf2e57cfdc4932e5034c46f78e2cf202`;
- forecast seal `a81ce7dced2368543a0521182f6a26dc926409f9b49586d9602d2b7b39ea91f0`,
  the SHA-256 of the three windows' forecasts with their intervals and of the
  measured-cause forecast as the composition holds them under the final fault
  identities (`compose.forecast_seal`; in the world's `files/generator.json` under
  `expected.window_forecasts`, `expected.measured_cause_forecast` and
  `expected.forecast_seal`);
- the cells: the record as of the calibration jobs `20261010-task-cells-r6` and
  `20261010-task-cells-r6c` (`curves.refinement`), digest `ed44717b1a2cf3a6`. The
  third job, imported later, does not move them: the recipe refuses to compose on
  anything else.

One thing outside this builder's files: `fleetlab.worlds.world` builds its own
estimation context without the recipe's settings, so the `predictions` block it
writes into `generator.json` shows, beside the same expectation and the same interval
of the forecast's own spread (`conditional_prediction_interval`), a
`prediction_interval` widened by the third generation's error (observed 66.66 to
79.52%). The sealed forecast is the one above; `summary` reports it as `forecast`.
Passing `retained_error=comp.recipe.retained_error` and the recipe's cells in
`world.context` would make the block follow the recipe.

### D2 and the template: three plain faults per mechanism and kind of scope

`planned_recipe("D2")` and `("planned")` (the template of the test worlds; the two are
the same recipe under two names) now set `repeats` to 3: up to three plain faults of
one mechanism at one kind of scope, a release counting as what it carries. D3 always
had three; the third generation's D2 had one. The recipes changed in this respect
because with one they cannot compose at all. Their fleet-wide release holds the
software version and the scene camera on every robot, which takes the camera faults
and every further release off the menu. What is left is a delay on a robot, a delay
at a site, worn pads on a robot, a detector error on a robot, a batch at a site and a
furniture shift at a site, of which the delay at a site and the furniture shift
exclude each other (one heavy site-wide fault per world): five candidates for the
confounder's fault and four plain faults, each of which would have to stand on
measured contexts above the floor at once. A planning run that gave every unmeasured
context of the menu a guessed cell composed on 0 of 20 seeds with one fault per
mechanism and scope, and on 9 of 20 with three (150 drafts per seed; guesses, used
only to decide what to measure and reported in the third job's preregistration).
Section 1 of the design note fixes the patterns of a world, the fleet-wide release,
one scope per plain fault and about ten active faults; it does not limit a mechanism
to one fault per scope (orchestrating session, 2026-10-10). `repeats` is a field of
the recipe like the three settings; `tests/test_worlds_search.py` shows the cap
holding at one and at three.

### What the third calibration job measured

The job (root `~/.cache/robots-probes/worlds-v4/20261010-task-cells-r6d`, frozen
`b02328c`; plan in "Round 6c, third job: preregistration" at the end of this file,
committed at 17:06:57, admitted at 17:07:24) ended complete after 6,195 of its 10,800
seconds. All 22 stages of the sealed plan were reached; 18 ran, and four continuations
had nothing to continue because every task of their context lay outside 15 to 85%
after ten attempts. 2,540 attempts, 384 chain comparisons, all identical, no
incident, 103 GPU minutes; no stage was stopped and nothing was left out of the plan.
The record now holds 891 context cells from 11,981 attempts and five same-attempt
decoy comparisons (import `c644339`). On an even mix of the ten tasks of a suite,
under the 3-pixel blur:

- **The decoy under the blur with a site-wide 17-step delay.** On the same 230
  attempts the changed backdrop moves success by minus 1.3 points (interval minus 4.2
  to plus 1.6): zero lies inside and a loss of six points is excluded, so a decoy may
  stand on a spatial site that carries the release and the delay. The delay under the
  blur leaves 56.0% there (410 attempts), with the backdrop 55.8%.
- **The interaction on object tasks.** The 8-step delay alone leaves 84.8%, 15
  actions per chunk alone 88.6%, both together **2.0%** (2 of 100; without the blur
  the pair leaves 28.0%). The blur alone leaves 81.2%. A robot of the mixed site that
  carries the interaction under the release therefore succeeds on its spatial tasks
  (68.6%) and hardly ever on its object tasks, and stays above the floor only where
  it runs mostly spatial tasks.
- **Worn pads on object tasks** cost nothing under the blur: 85.4, 85.8, 88.0 and
  88.2% at 0.25, 0.5, 0.75 and full wear. On spatial tasks quarter wear leaves 98.0%.
- **The heavier batch on object tasks** leaves 88.8%, with half wear besides 87.2%.
- **Delays.** On spatial tasks the 17-step delay with four mass doublings leaves
  47.0%, the 14-step delay 91.6%. On object tasks the 14-step delay leaves 27.6% and
  the 17-step delay 0 of 100.

### D2 and the template after the import: 4 of 20

With the third job imported, the three settings and three plain faults per mechanism
and kind of scope, on the twenty assessment seeds with 400 drafts each, **D2 composes
on 4** (seeds 9, 10, 14 and 19) and so does the template `planned`, which is the same
recipe under another name. The four forecasts of the observed window lie between
69.2 and 71.5%, their intervals between 65.2 and 73.5%; six measured causes hold in
at least 99.6% of the resamples; the lowest robot-window lies between 33.7 and 41.7%.
On the sixteen seeds that do not compose, what refused their 6,400 drafts (a draft
that `check` refuses counts under each rule it breaks):

| Refusal | Drafts |
|---|---:|
| fewer than four plain faults fit measured contexts above the floor (3,950: three fit 1,615 times, two 1,701, one 461, none 173) | 3,950 |
| no carrier for the interaction on measured contexts above the floor | 1,433 |
| the observed forecast outside the target of 64.0 to 71.5% | 796 |
| the fleet release without a positive measured removal interval on the task mix | 104 |
| no site on which a decoy is confirmed under what covers it | 96 |
| six measured causes in fewer than 95% of the resamples | 40 |
| the confounder's fault not detectable | 15 |
| no confounder, a decoy no longer confirmed, a robot window below the floor | 19 |

Four seeds (3, 6, 8 and 16) fail on the interaction: their mixed site runs so many
object tasks that a robot carrying the pair under the blur, at 2% on object tasks,
falls below the floor, and on a spatial site the pair has no measured context beside
the batch or beside the delay with the decoy. On the other twelve the drafts run out
of plain faults.

### More drafts, and what would lift the template

The template of the test worlds needs 15 of 20; it has 4. Each row below is the same
assessment (twenty seeds, 400 drafts unless said) with one named departure. The rows
on guessed cells give every context of a named family that is not measured a cell of
fifty attempts at the product of its measured parts: they say which contexts the
search would use, not what a measurement would show.

| The template | Seeds of 20 |
|---|---:|
| as it stands | 4 |
| with 1,600 drafts per seed instead of 400 (one more seed, at its 647th draft) | 5 |
| with a floor of 0.25 instead of 0.30 | 6 |
| with three plain faults instead of four | 7 |
| with three plain faults and a floor of 0.25 | 10 |
| without the fleet-wide release | 16 |
| guessed cells: pads worn to 0.25, 0.75 and fully, together with the heavier batch, with and without the blur | 9 |
| guessed cells: those, and the interaction pair and each of its members together with the batch, with and without the blur | 11 |
| guessed cells: every context of the menu | 14 |

More drafts hardly move it. What holds the template down is the fleet-wide release:
every context a robot can be in has to be measured a second time under the blur, on
both suites where the robot stands at the mixed site, and the blur turns the
interaction on object tasks into a near-total failure. Three things follow from the
table. The release is the one departure tested here that clears the bar, and
dropping it is a change to section 1 of the design note, which plans a fleet-wide
fault. Calibration alone does not clear it: with every context guessed the template
reaches 14, and of the measurable families the worn pads with the batch (six
contexts on spatial tasks, 1,000 to 1,500 attempts at the third job's sizes) would
lift it most, to about 9 if the guesses hold; with the interaction beside the batch
(twelve contexts) to about 11. And once contexts are no longer short, the drafts are
refused for a forecast outside the target (1,327 of the 3,600 drafts of the nine
refused seeds in the second guessed row) and for two faults starting on the same day
at overlapping scopes (671), which the search could avoid instead of finding at the
end; that is a repair of the search this round did not make.

### `w48`: composed and sealed, not run

`DEVELOPMENT["w48"]` is `planned_recipe("D2")` with a private seed, the three
calibration jobs it stands on, their pinned digest and the world's hash, sealed by the
procedure of `w49`. Seeds were drawn from the operating system's random source until
one composed: the ninth did, on its 105th draft; the first eight were dropped, and
of a dropped draw nothing was looked at but the count of refusals per rule (four ran
short of plain faults at three, one at two, two had no carrier for the interaction,
one forecast outside the target). No outcome of any run was involved. The bar of 15
of 20 is for the test worlds' recipe, not for composing one development world
(orchestrating session, 2026-10-10). The world: 12 robots at 3 sites, 10 days, 1,680
planned attempts (1,540 on spatial tasks, 140 on object tasks), neutral fleet id
`w01`, opaque fault identities.

| What | Where | From day |
|---|---|---|
| Fleet-wide: a software release carrying a 3-pixel blur | every robot | 4 |
| Decoy: the changed backdrop | site s1 | 3 |
| Confounder: a 17-step action delay, with an inert calibration record on the same robot the same day | r10 (s3) | 3 |
| Additive pair: objects four mass doublings heavier, and pads worn to 0.5 on one robot | site s3; r12 | 1; 0 |
| Interaction: an 8-step delay, then 15 actions per chunk | r06 (mixed site s2) | 0; 1 |
| Drift: pad wear rising linearly by 0.096 per day | r07 (s2) | 0 |
| Plain: a 17-step delay | site s1 | 1 |
| Plain: a 17-step delay | r09 (s3) | 0 |
| Plain: a 17-step delay | r11 (s3) | 0 |
| Plain: pads worn to 0.75 | r05 (s2) | 1 |

Two plain faults and the confounder's fault are 17-step delays on robots: three of
one mechanism at one kind of scope, which is the cap the recipe now states; a fourth
delay stands at a site.

The sealed forecast, on the outcome the fleet logs, with the forecast's own 95%
interval and no retained error:

| Window | Attempts | Forecast | Interval |
|---|---:|---:|---:|
| Observed (days 0 to 5) | 1,008 | 70.72% | 66.86 to 72.72% |
| Experiment (days 6 to 7) | 336 | 66.38% | 59.82 to 70.84% |
| Scoring (days 8 to 9) | 336 | 63.09% | 56.84 to 67.56% |

Measured causes, forecast for the scoring window: **eight** faults gain at least two
points of fleet success by the point estimate, and at least six do in 98.6% of the
resamples (between six and eight in 95% of them). In points: the site-wide delay 11.8
(interval 9.6 to 12.7), the heavier batch 5.8, the two members of the interaction 3.5
and 3.2, the delay on r11 3.1, the confounder's delay on r10 2.9, the delay on r09
2.7, and the fleet-wide release 2.5 (interval 0.4 to 5.2; it reaches two points in
73% of the resamples and is the least certain of the eight). Not causes by the
forecast: the worn pads of the additive pair (0.4), the drift (0.2 in the scoring
window), the pads worn to 0.75 on r05 (minus 0.3: on that robot's tasks wear costs
nothing measurable under the blur) and the decoy (minus 0.3, interval minus 1.5 to
plus 1.1).

The lowest robot-window is r10 in the scoring window at 37.1% (the floor with its
reserve is 33%); its lowest single day is day 8 at 28.6%. The three largest
single-fault fixes by scoring gain are the site-wide delay at s1 (11.8 points,
configuration, in the change log), the heavier batch at s3 (5.8 points, a physical
condition, visible only in behaviour) and the 8-step delay of the interaction on r06
(3.5 points, configuration, in the change log): two of the three are logged settings.

The seals, in `tests/worlds_generation4_seals.json` and checked by
`tests/test_worlds_generation4.py`:

- world hash `9c2ca2cfa8201313ad2a4096a70e9cc797ecfc1af4dc9223ff11d604e93bff96`;
- forecast seal `5d0c5b77038abc19c677b883dc75119d99fd22e4067b63cbc94aaf6547cd4ff4`;
- the cells: the record as of all three calibration jobs, digest `3c66af84118dcafd`.

As for `w49`, the `predictions` block that `fleetlab.worlds.world` writes shows a
`prediction_interval` widened by the third generation's error (observed 66.16 to
78.72%); the sealed forecast is the one above.

### Frozen run commands (neither was run by this builder)

Each world has a detached checkout of its own at its sealing commit, with its own
environment (`make setup` exited 0) and with the world's files generated from it on
the CPU into the default root; the stored hash and the forecast seal equal the
repository's fixture, every job's images per call are in the pinned database, and no
cause rests on a transferred or an assumed point.

```sh
cd /home/cs/.cache/robots-worlds-run-w49    # cc454da, root ~/.cache/robots-probes/worlds/w49
make world WORLD=w49 WORLD_FLAGS="--workers 6"
cd /home/cs/.cache/robots-worlds-run-w48    # 659a780, root ~/.cache/robots-probes/worlds/w48
make world WORLD=w48 WORLD_FLAGS="--workers 6"
```

`make world` uses `tools/heavy-run`, the pinned MIOpen database, batch size four and
`fp16_res32`, and runs both windows, the records and events, the export and the
summary, without the oracle. Keep a checkout frozen while its command is queued or
running. The orchestrating session starts the runs; it queued `w49` behind the third
calibration job, and a run of `w49` from its checkout was in the process list from
18:52 on 2026-10-10, two minutes after that job had ended.

`python -m fleetlab.worlds check w49` exits 1: that command counts causes over the
observed window only, as before the design note's entry 15 (five for `w49`, six for
`w48`; seven and eight by either window). `run` does not use that count.

### Verified in this part, and not

Verified. The third job's manifests against the sealed plan before admission (the
test of the plan), and its 384 chain comparisons. Both sealed worlds: generated on the
CPU from their frozen checkouts, the stored hash and forecast seal equal to the
fixture, and `tests/test_worlds_generation4.py` recomposes each to its seal and
checks the band, the six causes, the floor, the decoy under the fleet margin, the
patterns and the cap on plain faults; `w49` recomposes to its seal on the record
after the third job's import. Each rule at its old and its new value in
`tests/test_worlds_settings.py`. Forty deliberate breaks of the rules' code, each in a
copy of the source, are each caught by the test named for it: the fifteen of the first
part and twenty-five for this part (the margin's basis, its share, its cache and its
use by the search and by `check`; zero outside the interval; the named error in the
interval, in the context and at zero; the floor's value and its reserve; the cap on
plain faults and its value for the fleet recipes; the named jobs in the record, in
the composer and in the sealed-calibration check; the seal's coverage of the final
identities, the causes and the later windows; the stage bound, the driver's stop and
the plan's cap). One of them, moving a field's default, is stopped earlier: the
recipe module refuses to load.

Not verified, or outside this builder's files. No world was run here, so neither
forecast has been compared with a measurement; that is what the two runs are for.
The rows on guessed cells are not measurements. The bound that keeps a stage from
starting when it would not fit was never needed in the job (every stage fit) and is
covered by its unit test only. `fleetlab.worlds.world` builds its own estimation
context: its `predictions` block carries the third generation's error, and it
records the whole record's digest in `generator.json`, so a private recipe that
names `calibration_jobs` would be refused when it is resumed from that file until
`world.build` passes the recipe's jobs to `curves.measured_sources`. No private
recipe names jobs today. The fleet margin is applied to each covering context of a
decoy by itself (each must exclude six points on a site's carriers), which is
sufficient for less than two points of the fleet and stricter than pooling them.

## Round 6c: both calibration jobs imported, the search repaired, and why no world composes (2026-10-10)

The round was finished by a second builder on `stage3/worlds-r6c`,
from the first builder's head `f966c99`. Commits: `2081d34` (import of the first job),
`1cdece0` (the follow-up job's plan, sealed before admission), `24e93d3` (record
builder over several roots, the decoy rule in parts), `03df29a` (measured-context
search), `0796f12` (a test for the decoy margin), `bb333fc`, `734821a` and `b346015`
(assessment and forecast bound), `7ba8307` (import of the follow-up job), and this
note.

### What the two jobs measured

The first job (root `~/.cache/robots-probes/worlds-v4/20261010-task-cells-r6`, frozen
`e42f61f`) ran all thirteen stages of its sealed plan: 4,294 attempts in 156 GPU
minutes, 448 chain comparisons, all identical, one recovered worker abort (`e004184`). The first builder's driver ended with its session after three
stages; the orchestrating session ran the other ten with the same stage command from
the same checkout in the plan's order and lifted its own 90-minute bound for them.
On an even mix of the ten spatial tasks:

- Under the 3-pixel blur the dimmer light costs 5.7 points (same-attempt interval
  −9.8 to −2.1). It is not a decoy there. The changed backdrop moves success by +0.4
  points (−2.4 to +3.2): zero lies inside, a two-point loss is not excluded.
- The blur alone leaves 94.3% on spatial tasks (96.9% without it) and 81.2% on object
  tasks (98.5%).
- The interaction pair (delay 8 with 15 actions per chunk) leaves 59.5% on spatial
  tasks (634 attempts with the folded worlds) and 28.0% on object tasks (300). Under
  the blur it leaves 68.6% on spatial tasks: the blur rescues attempts the pair
  breaks (task 2: 10 of 10 against 82 of 122; task 5: 10 of 10 against 45 of 66).
  That is the overlap the last review found in `w47`, now measured.
- The 17-step delay leaves 56.2% alone and 54.2% under the blur; the 6 degree pan
  54.8% and 59.2%; full wear 73.5% and 76.6%; four mass doublings 87.2% and 88.0%;
  half wear with four mass doublings 78.1% and 84.0%.
- No context under the blur was measured on object tasks.

The follow-up job (root `~/.cache/robots-probes/worlds-v4/20261010-task-cells-r6c`,
frozen `1cdece0`; its plan is in "Round 6c preregistration" at the end of this file,
committed at 13:41 before the job was queued, admitted at 14:02) ran under one
admission and its hard 5,400 seconds. Six of its eight stages completed with exit 0
and a passing chain sample: 1,827 attempts, 205 chain comparisons, all identical, no
incident, 76 GPU minutes. The seventh, the decoy comparison under the blur with the
17-step delay, was stopped by the budget at 300 of 400 attempts and is not
calibration; its continuation never started. It measured:

- the thin single cells that refused every interaction carrier, the drift and a batch
  on the mixed site: delay 8 alone on spatial task 7 is now 45 of 50 (object tasks 7
  and 9: 41 of 50 each), 15 actions per chunk on object task 7 41 of 50, quarter wear
  on spatial task 9 45 of 50, four mass doublings on object task 5 41 of 50, the 6
  degree pan on object task 5 29 of 50;
- under a 17-step delay, on the same attempts (320 pairs per decoy, even mix of the
  ten spatial tasks): the dimmer light moves success by −0.7 points (−4.0 to +2.6),
  the changed backdrop by +0.7 points (−2.5 to +3.9). The delay alone leaves 55.9%
  (500 attempts), with the dimmer light 55.6%, with the backdrop 57.0%;
- the 17-step delay with four mass doublings: 48.4% (254 attempts with the folded
  world's);
- further plain faults: delay 14 leaves 91.8% on spatial tasks, delay 17 5.6% on
  object tasks, the 4-pixel blur 76.6% on spatial and 45.9% on object tasks.

The refinement record now holds 741 task-context cells from 9,441 attempts (552 and
3,360 when the round began). With the published cohorts the table has 1,912 task
cells, 1,369 of them precise under the rule of fifty attempts between 15 and 85%.
`calibration.refinement_record` rebuilds the committed record from the two folded
worlds and both roots; a test compares, and checks that no attempt of the follow-up
job is counted twice.

### The assessment

`compose.assess()` (`python -m fleetlab.worlds.compose`) composes each recipe on the
twenty seeds `ASSESSMENT_SEEDS` and counts, for the refused seeds, the attempts each
rule refused.

As the code stood after the first import (nothing changed but the cells): D2 0 of 20,
D3 0 of 20, planned 0 of 20. "No interaction carrier meets task precision, paired
support and robot-window floor" on 19, 20 and 19 seeds, before any attempt: removing
the chunk setting leaves the 8-step delay alone, whose spatial task-7 cell held 12 of
15. On the remaining seed of D2 and of planned all 400 attempts ended "draft
incomplete".

At the end, with both jobs' cells and the repaired search: D2 0 of 20, D3 0 of 20,
planned 0 of 20.

- D2 and planned. On 16 seeds every one of the 400 attempts stops at "no site on
  which a decoy is confirmed flat under what covers it": under the blur the backdrop
  passes the two-point margin on a site of 4 seeds only. On those 4 seeds every
  attempt stops at "no interaction carrier left on measured contexts above the
  floor": the decoy and the batch take both spatial sites, and nothing under the blur
  is measured on object tasks, so the mixed site can carry only the release.
- D3 (site-wide interaction, five plain faults). 3,846 attempts stop at "no
  interaction carrier left on measured contexts above the floor" and 3,754 at "no
  robot left for a drift crossing in the scoring window": a site-wide pair, a decoy
  site and a batch site leave the drift no robot on measured ground. One seed has no
  site whose four robots all pass the carrier preflight.

### What was repaired

- **The search** (`03df29a`). The older draft drew the batch's site, the decoy's site
  and the drift's robot without looking at what was measured, and found out at the
  end. `Composer.draft_measured` places the fleet release, the decoys, the additive
  pair, the interaction and the drift first, each on the first candidate (in an order
  drawn from the attempt's generator) that `supported()` accepts: every context and
  single removal of the robots it reaches measured and precise, their windows above
  the floor, every decoy flat under what covers its carriers. An attempt that gives
  up records where, so the refusals name a rule. `settle_measured` takes the supported
  move that brings the observed forecast closest to the aim's middle. The rules of
  `check` are unchanged, and so are the older estimators' draft and settle: both
  third-generation seals reproduce. The measured branches inside the older `draft`
  are no longer reached.
- **Three type errors** in the tests of the first builder's last commits (`mypy` failed
  at `f966c99`), and that planner test's dependence on the cells of the day.
- **A break that passed**: dropping the two-point margin from the decoy rule failed no
  test. `test_a_decoy_is_flat_only_when_its_interval_excludes_the_margin` now does.
- The record builder takes several calibration roots and the same-attempt
  comparisons a plan names; the decoy rule reads its comparisons once per file state.

### Why no world composes

Three rules of this round meet, and the cells do not let them be met together.

1. **The band rule.** The observed forecast with its retained error (−0.7 and +6.0
   points) must lie inside 60 to 75%. The forecast's own spread, on the compositions
   of the what-if runs below, is about 3.5 points down and 2.5 up, so the forecast
   must lie between about 64.3 and 66.3%. The recipes aim at 64.0 to 65.7%.
2. **The decoy rule.** A decoy is confirmed flat (same-attempt interval that includes
   zero and excludes a loss of two points on its carriers) only where the comparison
   exists and is narrow enough: alone (some site and kind on 20 of 20 seeds); under
   the blur with the changed backdrop (a site on 4 of 20 seeds); under a site-wide
   17-step delay (a site whose four robots also keep the floor, on 5 of 20 seeds).
   Elsewhere the decoy's site of four robots carries nothing else and stays at about
   97%.
3. **The overlap rule and the floor.** Every other robot may carry only a context
   measured on each of its tasks, and must keep 0.40 plus the reserve of 0.03 in each
   window.

The arithmetic where the decoy's site carries nothing else: its four robots (96.9%
each), the drift's robot (by rule under five points of loss before day 8, so at least
91.9%), the additive pair's worn robot (78.1%) and the interaction's robot (59.5%)
hold 51.4 of the fleet's points. For a forecast of 65.7% the other five robots would
have to average 34%; for 66.3%, 36%. The floor with its reserve is 43%.

`compose.forecast_bound()` makes this a bound from below for each seed: every robot
outside the decoy's site in the most damaging precise context that keeps its later
windows at the floor, counted on each observed attempt at the lowest rate among that
context and the contexts of fewer of its faults; the decoy's site in a context that
confirms a decoy there; one carrier each for the interaction, the additive pair and
the drift; nothing else respected. Over the twenty seeds, for the recipes as they
stand:

| Recipe | Seeds with no site for a confirmed decoy | Bound on the others: lowest, median, highest | Seeds at or below 65.7% | at or below 66.3% |
|---|---:|---|---:|---:|
| D2, planned | 16 | 77.6%, 78.3%, 78.7% | 0 | 0 |
| D3 | 0 | 54.8%, 70.1%, 76.0% | 4 | 5 |

The five seeds of D3 are those where a decoy is confirmed under the site-wide
delay. With the interaction on one robot instead of a site, D3's median is
66.6% and five seeds lie at or below 65.7%.

What the search reaches lies above the bound, because a real composition also has one
fault per thing, site-wide faults that reach whole sites, onsets after day 0 and a
menu of fault types. For the recipes as they stand no attempt completes a draft. With
the interaction on one robot and four plain faults (a variant tried for this note,
not a recipe of the round), 358 of 3,000 attempts complete on 16 seeds; their
observed forecasts lie between 74.1 and 90%, the lowest per seed between 74.1 and
87.2% (median 80.2%), and all but one are refused by the aim. On the five seeds where
a decoy is confirmed under the site-wide delay the search does put the delay on the
decoy's site; restricted to the heaviest plain faults it still ends at 75.0% or
above on the three of them that have an interaction carrier. The four seeds whose
mixed site favours object tasks have none: the pair leaves 28% on object tasks.

### What would compose

No single value does; for the recipe without a fleet fault, two or three together do.

By the bound, one lever at a time:

- **The floor.** At 0.30 (0.33 with its reserve) the bound for D3 falls to a median of
  68.8% (at or below 65.7% on 2 seeds), at 0.20 to 67.5% (5 seeds). For the fleet
  recipes it does not move (77.4 to 78.2% on the four seeds that have a decoy site).
- **The retained error.** At zero the forecast could be as high as about 71.5%. The
  bound allows that on 11 seeds for D3 and on none for the fleet recipes.
- **The margin of the decoy rule** (`calibration.DECOY_MARGIN`, two points of success
  on the decoy's carriers; entry 15 of the design note separates decoy from cause at
  two points of FLEET success, which for a site of four robots is six points on its
  carriers). Under the blur the backdrop passes on a site of 4 seeds at two points,
  of 15 at three, of all 20 at four. Under the 17-step delay a decoy passes, on a
  site that keeps the floor, on 5 seeds at two points, 10 at three, 16 at four (16
  seeds have such a site). This lever has the most weight: if the decoy's site could
  carry faults like any other site, the bound would fall to a median of 52.6% for D3
  and 62.8% for the fleet recipes (interaction on one robot).

By the composer itself, on the twenty seeds with 400 attempts each. The runs are for
D3 with the interaction on one robot and four plain faults (two parameters of the
recipe; as it stands, with a site-wide interaction and five plain faults, it completes
no draft under any of these settings). Every other rule is as it stands. The values
were set in a script outside the repository; nothing in the repository was changed.

| Floor | Margin of the decoy rule | Retained error, aim | Seeds on which D3 composes |
|---|---|---|---:|
| 0.40 | two points | −0.7 and +6.0, aim to 65.7% (all as they stand) | 0 |
| 0.40 | six points | none, aim to 71.5% | 2 |
| 0.35 | six points | none, aim to 71.5% | 6 |
| 0.30 | two points | none, aim to 71.5% | 3 |
| 0.30 | six points | −0.7 and +6.0, aim to 65.7% | 5 |
| 0.30 | six points | none, aim to 71.5% | 16 |

Every composition among these has six causes in at least 97% of the resamples and
its whole interval inside 60 to 75%. With the floor at 0.30 their lowest robot window
lies between 33 and 45%. They all look alike: the decoy on a spatial site that also
carries a site-wide 17-step delay (the confounder's fault, or a release), the batch
on the other spatial site with delayed robots beside the worn one, the interaction
and the drift on the mixed site. Of the four seeds that do not compose in the last
row, three end outside the aim and one has no robot that can carry the pair (its
mixed site favours object tasks, where the pair leaves 28%).

For the fleet recipes none of this composes. With the floor at 0.30, the margin at six
points and no retained error, planned composes on 0 of 20 seeds: 7,904 of 8,000
attempts stop at "no interaction carrier left on measured contexts above the floor".
What the fleet recipes need first is evidence: every context
under the blur on object tasks (the pair, its two members, wear, the batch), and the
decoy compared on the same attempts under the blur with a site-wide fault, which
the budget cut. Until then the mixed site can carry only the release. That is a
calibration job of its own, and the test worlds' private recipe meets the same
refusals as "planned".

### Not done, and what was verified

No `w48`, no `w49`, no forecast sealed, no run command: the rules as they stand are
not met, and none was relaxed. The margin of the decoy rule, the floor, its reserve
and the retained errors (`estimate.MODEL_ERRORS`) are unchanged.

- The one full `make check` ran from a detached checkout at `b346015`, where code and
  cells are final (this note changes text only): exit **0**; lint, formatting and
  types passed (389 files); **2,581 passed, 2 skipped, 65 deselected** in 2,379.85
  seconds. A first run, at `bb333fc`, ended with one failure,
  `tests/test_engine_fleet.py::test_shutdown_with_a_request_larger_than_the_pipe_is_bounded[interrupt]`:
  it had been started with `nohup` from a background shell, which hands interrupts
  down ignored, and that test sends itself one. Started with default handling it
  passes, alone and in the final run. A test run's temporary directory must lie
  outside the home directory and must not have `worlds/` in its path (an audit test
  refuses that word in a public text); both are tracker memories now.
- Fifteen breaks each fail a test: the review's six, and nine of this round's search,
  lookup, drift, decoy rule and continuation (a copy of `src/` per break, at
  `b346015`).
- Both third-generation seals reproduce (`tests/test_worlds_round3.py`, in the full
  check).
- The follow-up job's manifests on disk are the plan that `1cdece0` names, and the
  record recomputes from the stored runs of both jobs and both folded worlds.

Not verified: no world was run, so the estimator is still unchecked on a fresh world.
The decoy under the blur with the 17-step delay was not measured. Nothing under the
blur is measured on object tasks. The bound counts the drift's robot at five points
below its rate without the drift, which the rule guarantees for the drift alone;
under the blur the wear cells lie within about three points of those without it.
The what-if runs and the variant's numbers come from scripts kept with their outputs
in `~/.cache/robots-probes/worlds-v4/20261010-round6c-analysis` (`whatif.py`,
`explore.py`, `final_numbers.py`), not from the repository; `compose.assess` and
`compose.forecast_bound` are in it.

Left in place: the detached checkouts `~/.cache/robots-worlds-run-r6c` (`1cdece0`, the
job's) and `~/.cache/robots-worlds-check-r6c` (`b346015`, the check's). The blocker is
filed separately; the planning defect is closed.

## Round 6: measured contexts and window forecasts (2026-10-10)

`2614bbb` merges the record-based root identity fix. New/private recipes use
`measured-v3`; `task-v2` is retained for historical diagnostics and cannot be
selected through the private creation path. Both historical development seals
continue to use their original calibration. The additional `task_refinement`
record has its own digest, required by new recipe seals; it cannot silently
change an old seal.

All **3,360** w46/w47 attempts are now task-context calibration, grouped in
**552** exact dose/task/context cells before the new job. A release expands to
its carried mechanisms. Only the calibration rewrite is erased by construction;
lighting, backdrop and the five-sampling-step setting remain in their contexts.
Drifting doses have exact numerical descriptors. Each count names its source run;
physical attempts are deduplicated, conflicting outcomes refused, and the raw
source recomputation is tested. These observational counts do not alter the
previous same-attempt interaction test. The two worlds cease to be validation
worlds. Their reports and exports are not edited.
The review's **476/966** solo-or-pair counts reproduce when labelled neutral
faults are dropped. Keeping the actual mechanisms gives **238/966** such
attempts for w46/w47; all the others remain in their complete contexts. In
particular, calling lighting or five sampling steps harmless does not erase
an unmeasured overlap from a calibration cell.

The source audit found one recovered incident in `w46-later`: worker 1 exited
with code −6 on `e000735`. Its record marks recovery and an identical replay;
the completed attempt is retained. The other three runs report no incidents.
Thus all four chain samples are clean, although the later w46 incident count
is one. This is separate from the archived w45 chain failure.

A new prediction uses each scheduled task's **absolute measured context rate**.
There is no product-of-single-ratios fallback. Linear dose interpolation requires
measured context-grid corners; all individual-removal contexts must also exist.
For a continuous drift, the interpolation knots are those measured in its
covering context. A solo quarter-wear measurement does not create a joint knot
under blur; that context can interpolate between zero and half wear if those
joint endpoints exist. A measured but thin intermediate knot cannot be skipped.
Every used mid-range cell (15–85% success, including the logged outcome) requires
50 attempts, and other cells require at least ten. A new measurement can turn a
formerly extreme cell into a mid-range one; that cell becomes ineligible until it
has fifty observations. Decoy qualification additionally needs same-attempt
context controls for every carried task. A task-weighted paired Newcombe interval
uses Kish effective sample size, includes zero and excludes a two-point loss.
This is an explicitly approximate stratified interval, not an exact paired test.

The floor is applied to each robot's **observed, experiment and scoring window**,
with the existing three-point engineering margin. Individual days may be lower.
Scoring removal gains share each calibration-cell draw across the world and all
counterfactuals. Each qualifying member counts separately; only an interaction
with neither qualifying member can use the pair fallback. Additive pairs cannot.
Admission requires at least six in 95% of the resamples. This forecasts expected
oracle gains; finite-scoring-sample and replay-pair covariance uncertainty are not
included in that probability. The actual gate remains pending the oracle.

The prediction interval combines calibration draws, conservative binomial window
noise and a retained model-error envelope. The observed envelope is **−0.7/+6.0
points**, as directed. The later envelopes also retain the errors already seen:
**−3.7/+6.6** for experiment and **−4.8/+5.4** for scoring. These are conditional
sensitivity intervals with an empirical error reserve, not validated 95% coverage.
Observed admission requires the entire outward-rounded interval inside 60–75%.
The trial recipes aim at 64.0–65.7% while settling doses/onsets, centred at
64.85% after the asymmetric error reserve. This guides the search; it does
not replace or weaken the interval check.
The reported band probability is the share of resamples inside the band for every
error in that envelope; no error distribution is inferred from two worlds.

A fleet candidate needs measured cells on every acted task and a **positive lower
removal-gain bound on its actual scoring task mix**. It need not harm every task or
suite: the task-mix removal, rather than an old suite-wide role tag, establishes
its effect. Historical recipes retain their original suite-tag checks. New
confounders use an inert calibration rewrite; a wrist record has a benign
counterpart even under a fleet scene-image fault. Its value uses the same four-
decimal distribution as benign records. This avoids assuming that five sampling
steps stay harmless in an unmeasured overlap. Small additive members and the drift
remain small causes; a separate overlapping small fault is not required.

`test_worlds_measured.py` covers contexts, missing counterfactuals, fifty-attempt
precision, windows, reserves, shared resampling, the retained error, fleet effect,
private paths and private summaries. Separate source mutations of all six surviving
review breaks each fail its focused test. The one full `make check`, from detached
CPU checkpoint `f0e739a`, exited **0**: lint/format/types passed, and **2,556 tests
passed, 2 skipped, 65 deselected** (5,296.06 seconds). The sole warning is a
third-party robosuite string escape. Later world-only changes through `e0e5e1e`
have 30 measured-rule tests passing; 96 guard/review tests passed just before its
last whole-window regression. Final cell-data verification is recorded after the
calibration completes. No world run or smoke check is part of this repair.

The preregistered calibration and its hard budget are recorded near the end of
this file. **Admission is not yet evidence that either recipe composes.** Initial
preflight rejected all 20 CPU seeds of D2, D3 and the planned-difficulty template
because the interaction's task cells were too thin for eligible carriers. This
is rechecked after the calibration; thresholds will not be waived to produce files.

The post-admission coverage audit found a planning defect:
the joint's solo counterfactuals were omitted from the top-up menu. Spatial
delay 8/task 7 is 12/15 after the world fold, already mid-range in the
confirmation (8/10), and needs fifty attempts. The earlier published solo
cohort adds 9/10 there; even pooling it would leave 21/25, still thin. The
selection screen remains excluded. Future plans now close their menu over
every nonempty subset, measuring the singles before their joint; a real
CPU-manifest test checks the omitted cells. That audit also found spatial
quarter-wear/task 9 (8/10), an interpolation knot needed by the linear drift;
future plans include its forty-attempt top-up. These repairs do **not** amend
the running `e42f61f` job or its sealed manifests. A lower robot floor or a
smaller band reserve cannot repair a missing counterfactual measurement.
Future plans use the current combined count, skip sufficient cells, and
exclude physical identities already present in every recorded refinement source.
The interaction carrier preflight now checks both single-removal contexts,
including tasks that occur only after onset; it refuses those gaps before
repeating hundreds of doomed complete-draft searches.

## Round 5: task-specific qualification before test worlds (2026-10-10)

The batch-4 merge is `fb5830a`. All **105 single measurements and six joints** are
retained, with suite-specific verdicts and findable source/control directories.
There is no numerical change to a shared cell. Both world seals and both prediction
seals of w46/w47 still reproduce. The stored worlds and their running checkout are
untouched; their historical `suite-v1` estimator stays explicit. That estimator is
accepted only for development worlds with the seven recorded ID/seed pairs, never
for a private test/reserve recipe. New recipes default to `task-v2`.

`curves_supplement.json:task_calibration` materializes **133 authoritative single or
joint cells as 1,330 task cells** from the existing calibration runs, with separate
**1,000-attempt healthy baselines per suite**. `supplement.task_calibration_record()`
reproduces the whole table exactly from the named runs, requires held chain samples,
checks published count totals, and rejects missing twins or conflicting duplicate
attempts. Twins match suite, task, start state, scene/noise seeds, nominal policy,
batch/image/step settings, and simulator/asset fingerprints. The dependency lock
hash is retained in the source run but excluded from this matching key: historical
healthy extensions cross a lock-file update. No new simulation or calibration job
was run. The independent 100-attempt confirmation is used for the mild pair; its
screen is excluded. Three false-positive detector cells conditioned on a camera
fault are retained in the library but deliberately unavailable as standalone task
ratios. Missing task evidence fails closed.

Expected success, removal gains, drift crossing and every robot-day now use the
scheduled **task index**, including measured joints and releases carrying them.
Prediction intervals resample those task cells and share each sampled value across
all its scheduled uses. They remain nominal, conditional sensitivity intervals:
independent Jeffreys draws omit paired covariance, within-robot dependence, unknown
interactions and model error. Gains above the healthy rate are capped rather than
credited as improvements from sampling noise.

New-world admission also requires **3 percentage points** above the robot and scoring
floors and inside both ends of the fleet band, plus **0.5 points** beyond the two-point
threshold for each cause counted toward sizing. These are explicit engineering
reserves, not statistical coverage guarantees. Section-5 detectability and fault
flags retain the two-point/carrier rule; the extra sizing reserve is separate.
The settling and final floor checks use the same task-based computation.

Decoy calibration currently measures each decoy alone. A new composition therefore
refuses every overlapping companion, including a release or another decoy, unless
the companion is an inert calibration record with no mechanism. Nonoverlapping
carriers/windows remain eligible. A solo flat curve cannot certify a joint context;
there are no measured decoy contexts in the current library. **w46 is unchanged**;
its oracle confirmation and any later truth correction remain the orchestrator's work.

A drift in `task-v2` must cross five points of its own harm in the scoring window,
regardless of the historical `warnable_drift` flag. Only historical `suite-v1`
recipes retain that flag's original behavior.
Its health precursor must additionally shift a metric by **at least two daily-noise
standard deviations on at least two observed days before that crossing**. The noise
includes accumulated benign-slope dispersion; a force-loss signal uses the lowest
baseline force. A nonzero camera shift smaller than daily noise is refused. The
composer searches only rates satisfying both conditions and refuses missing evidence.

Private CLI creation validates the resolved output root, `files` directory and any
existing file symlink before loading/drawing the recipe. Neither recipes nor outputs
may live in this or another checkout. Private generation prints only a hash and a
private-status marker; qualification failures and summaries suppress truth details.
`write_world` enforces the same output boundary. Before opaque-ID assembly returns,
a rule guard scans visible description/maintenance/health text for both final tokens
and the counted draft handles.

`test_worlds_task_guards.py` covers task-dependent success/gains/floors/predictions,
missing task evidence, all three reserves, contextual decoys, the noise threshold,
the carrier and same-window confirmation rules, the single-loss acceptance guard,
private-path/output boundaries, and a deliberately leaked draft handle. Historical
suite fixtures remain explicitly historical; the real w46/w47 seal regression also
requires their old numerical truth files to match after only enumerated provenance
substitutions and the estimator-version annotation. The seed-guessing attack test
now supplies the public derivation algorithm itself because batch-4 correctly removes
the identity module from the mounted bundle; its successful old-coupling control stays.

Read-only task-v2 recomputation of the existing schedules (no new seals or records).
These use the independent confirmation and exact canonical cohorts on both suites;
they are conditional model estimates, not the reviewer's pooled-cohort approximation
and not observed run results. Expected successes are percentages; cause counts use
the observed window (1,008 attempts), experiment/scoring each contain 336 attempts.
The minimum is over all 120 robot-days and both true/logged outcomes. Decoy context
neutrality remains an unmeasured assumption for this numerical audit.

| World | Observed logged % | Experiment logged % | Scoring logged % | Observed causes >=2 points | Causes >=2.5 points | Lowest robot-day % | Robot | Day | Robot-days <40% |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| w46 | 70.97 | 54.51 | 56.27 | 5 | 4 | 24.28 | r11 | 5 | 8 |
| w47 | 69.10 | 61.29 | 60.70 | 5 | 5 | 21.86 | r11 | 2 | 10 |

No new production world was composed, and no world run, GPU job, model session, oracle
or smoke job was started. `make check` ran once on `dd9ff30` and exited **0**:
**2,512 passed, two skipped, 65 deselected**, with lint, formatting and types passing.
After that check, the drift rule was made mandatory for `task-v2` even when the
historical flag is false. The complete world test selection then exited **0**:
**202 passed, one simulator test deselected**, including the 19 new rule cases and
both world/prediction seal regressions. Ruff lint/format checks and mypy (386 source
files) also exited **0** for the follow-up. The full check was not repeated, as requested.

## Round 4: opaque fault identities (2026-10-10)

New compositions assign `fx` followed by **32 lower-case hexadecimal characters**
to every fault. Tokens contain both letters and digits, so cannot be a word or
a count. HMAC-SHA256, keyed by the private world seed and domain-separated as
`fleetlab/fault-id/v1`, derives 128-bit tokens from private draft draw handles.
No fault kind, scope or accepted-schedule position enters the derivation. Reordering
the schedule preserves the mapping; the same recipe reproduces it. The 62-bit
world seed remains the entropy limit, and development seeds remain public as agreed.
Duplicate tokens are refused. No token is put in the visible fleet, description,
export or mounted filenames. `contracts/ids.py` accepts the opaque form alongside
canonical counted IDs for old records; opaque tokens have no ordinal number.

Assignment occurs after composition has passed its existing rules. Assembly retains
private draw handles for authors and evaluation order, then renames truth references
in the spec, jobs, patterns, fault table and detectability. Thus the assignment does
not redraw authors, change the schedule, or change numerical predictions. The fault
engine itself is untouched: its random streams include fault IDs, so newly identified
worlds have new stochastic fault identities. Neither world had run; this is not a
claim that a future rollout equals a rollout with the old IDs.

The immutable round-3 directories `worlds/w46` and `worlds/w47` stay intact. New
files use `~/.cache/robots-probes/worlds/w46-opaque-v1/` and `w47-opaque-v1/`;
their world IDs are still **w46** and **w47**, and public fleet IDs still **w01**.
The truth-side `base_runs` path follows this relocation; no modeled value changes.

| World | New world SHA-256 | Unchanged prediction SHA-256 |
|---|---|---|
| D2 / w46 | `beaac37883079862454e86fde1e3420b113f420636e81473a81b61fffb62790b` | `0258accd9e6f5923e9c161694142d73b10c4925bef76ac941641ac4b2d9b1cb5` |
| D3 / w47 | `f03a31c6868d3efcef0813cc2cc3b7d56a7163c2518e0fa9e1c136aa984d6849` | `dcd5319ca1cb4dcac12f2d3c93ad04d1863485140ea2221a311ba702945f9ead` |

`tests/worlds_round3_seals.json` records the 15-ID and 13-ID bijections and the
SHA-256 digests of all eleven original files per world, read before this change.
`test_development_recipe_reproduces_its_real_sealed_world_and_predictions` reverses
only those tokens and the consequent seal, using the original truth-side run path,
and requires **all 22 file digests** to match. This includes the entire 1,680-job
manifest for each world, authors, health, cause flags, drift and predicted intervals.
`model.json`, `description.json`, health and task texts are byte-identical outright.
The f01-style labels in the historical round-3 tables below refer to this bijection.

`test_worlds_fault_ids.py` checks deterministic tokens, seed separation, assignment
independent of order/type/scope, every scheduled fault receiving an ID, contract
rejection of malformed tokens, and legacy identity/replay-job round trips. Its CPU
toy export includes runs, replays and a continuation: no token survives in decoded
Parquet, other file bytes, description or filenames. Deliberate token leaks in
Parquet, a filename, description text and mounted-code text are all caught by the
existing export audit. Exporter-owned code is unchanged.

The read-only legacy audit verified all **five w41–w45 world hashes**, loaded all
**8,400 manifest jobs** and **8,400 stored run identities**, and constructed the
**840 unchanged replay inputs** selected by the runner. No stored world file was
rewritten. This verifies loading and replay input compatibility, not a fresh physical
replay: no world run, GPU job, oracle, model session or smoke job is authorized here.
Implementation and seals: **`3830f4e`**. The one final **`make check` exited 0**:
lint and formatting passed, mypy checked 273 source files, and pytest reported
**1,342 passed, 2 skipped, 62 deselected** in 522.95 seconds. No second full check
was run. The focused ID/export/seal regression run had 17 passing tests.

The detached run checkout is clean at **`3830f4e26f537f54252bffe1a29c0ffb1b101c6b`**.
Its own `make setup` and both CPU `generate` commands exited 0. The independent
frozen-file audit confirmed the 22-file comparison, all 3,360 unique identities and
derived scene/noise seed pairs, unchanged prediction seals and no run directories.
Its durable receipt is
`~/.cache/robots-probes/worlds/opaque-id-reseal-20261010.json`. Original w41–w47
world files were checked byte for byte again after generation and remain untouched.

### Current frozen run commands (not executed here)

The clean detached checkout is `/home/cs/.cache/robots-worlds-run-r4`; its exact
commit is recorded in the verification receipt above. Run from there, without editing
it while either command is queued or running:

```sh
cd /home/cs/.cache/robots-worlds-run-r4
make world WORLD=D2 WORLD_FLAGS="--workers 6 --root /home/cs/.cache/robots-probes/worlds/w46-opaque-v1"
make world WORLD=D3 WORLD_FLAGS="--workers 6 --root /home/cs/.cache/robots-probes/worlds/w47-opaque-v1"
```

`make world` uses `tools/heavy-run`, the pinned MIOpen database, batch size four
and `fp16_res32`. It runs both windows, records/events, export and summary, with no
oracle. These commands supersede the round-3 commands below and use new roots so
the original seals remain records. **Neither command was run in this round.**

## Round 3 measurement plan, sealed before execution (2026-10-09 PDT)

The four candidate (delay steps, actions per chunk) doses are **(8,10),
(8,15), (10,15), (10,20)**, in that order. Eight delay steps had no observed
object-suite loss in the earlier 100-pair record; ten is between that point
and the harmful fourteen. Chunks of fifteen and twenty are the one-quarter
and halfway restorations from ten toward the nominal thirty. The grid varies
the delay relative to the feedback/replanning interval (0.50 to 0.80), rather
than guessing a dose after looking at outcomes. All four doses are measured
on both spatial and object tasks. The previous (14,10) point stays a record.

Screening uses 100 new attempts per cell: healthy, each distinct single
(delay 8/10; chunks 10/15/20), and all four joints, per suite: **2,000
attempts**. All four arms of a comparison have identical task, start state,
scene/noise seeds, policy, precision and software; only faults and episode
identifiers differ. Calibration world w94, seed 941010, tasks 0–9, indices
0–99; nominal X-VLA fp16_res32, batch size four, at most six workers.

Choose **one** candidate for independent confirmation on indices 100–299:
200 fresh attempts per arm and suite, **1,600 attempts**, never pooled with
screening for acceptance. A screen-eligible suite has corrected joint success
at least 0.45, each single's loss at most ten percentage points, and the
paired 95% interval of `joint - A - B + healthy` wholly below zero. Prefer
candidates eligible on both suites; then the joint success closest to 0.60,
then the fixed order above. The 0.45 screen buffer aims to leave room above
the required 0.40 floor. Confirmation requires a corrected joint success at
least 0.40 and the same small-single and clear-interaction rules on at least
one suite; placement may use only an eligible suite. Report every screen and
confirmation result, including failures; do not choose a second candidate
after inspecting confirmation. Intervals are paired common-index bootstrap
95% (10,000 draws, seed 0); point effects also have paired Newcombe intervals.
These are suite estimates, not guarantees for every individual task.
Either stage stops if a healthy control succeeds on fewer than 90% of its
attempts; that is a failed calibration control, not a fault effect.

For decoys, use the explicitly permitted placement restriction: their site
runs only spatial tasks, where the recorded 1,000-pair dimming, backdrop and
warm-light points are confirmed flat. Other sites retain both suites. No
new decoy dose is asserted flat or measured in this round; this saves the
budget for the interaction. Restriction is in the truth-side recipe and the
ordinary visible task mix, never an agent-visible pattern label.

One queued calibration driver, from a new detached checkout, runs screening
and conditional confirmation. A hard timeout bounds all heavy execution
including startup/replays below **10,800 seconds**; the driver records elapsed
time and does not start a stage without enough forecasted time. A memory
refusal/stop is reported and a smaller-worker retry uses only the remaining
allowance. No world run or smoke-world is queued. Existing stored records
are immutable; all new calibration outputs go under
`~/.cache/robots-probes/worlds-v3/20261010-mild-pair/`.

### Resource-only confirmation contingency (declared while screening is blind)

At about fifty minutes of execution, the screen was still running. Before
reading any screen outcome, declare one sample-size contingency: if the bounded
driver refuses to **start** its 200-attempt-per-arm confirmation solely because
its forecast exceeds the remaining budget, use **100 fresh attempts per arm and
suite** (indices 100–199, 800 attempts total) for the same selected candidate.
The dose grid, candidate ranking, floor, ten-point single limit, interaction
interval test and prohibition on pooling/reselection are unchanged. A
200-attempt confirmation that has started is never truncated or reinterpreted
as this fallback. Sample size is selected by resource availability, not an
observed effect. The reduced stage still has to fit the remaining total heavy
allowance; no extra time is granted. Any conditional blur measurement then uses
those same 100 healthy controls per suite (200 faulted attempts total), rather
than claiming 200 controls that were never run. Its intervals will be wider.

### Conditional fleet dose, also declared before any round-3 result

CPU search with a labelled synthetic interaction (never saved as a world) shows
that the current fleet-wide four-pixel image blur also consumes most of the
robot floor, particularly on object tasks. If a mild interaction confirms but
that fleet dose prevents composition, measure **three-pixel blur**: the midpoint
between the recorded flat two-pixel dose and harmful four-pixel dose. This is
one predeclared fallback, not a dose search after looking at outcomes. Use the
confirmation stage's 200 independent healthy attempts per suite and run the
same 400 attempts with blur, from the same frozen checkout and identical
identities apart from fault and episode id. Require a held replay sample and a
measured loss beyond noise on both suites. No result is assumed in advance.
This conditional second job runs only if its forecast fits the **remaining**
10,800-second total allowance; it never extends the budget. If the interaction
fails confirmation, stop without this job or composition.

The CPU changes keep restrictions consistent across composition, sealed jobs,
experiment task menus, drift crossing and robot-day accounting. The dose search
rejects floor-breaking placements and settling moves before final acceptance.
Warnable drifts start at 1% of the calibrated maximum and grow linearly; this
avoids the engine's zero-start drift omission without
changing the required day-8/9 drift-only crossing. A mild interaction may cover
one robot or a site; every carrier must use only confirmation-eligible suites
and meet the same daily floor. This permits a
smaller per-attempt effect to be a detectable fleet cause without increasing its
dose. Image defects already measured on both suites can be carried by an
individual release as well as by the fleet release. New dose descriptors alone
are never treated as measured calibration. Historical calibration block numbers
and runtime fault mechanisms remain unchanged.

`confounder_companion` can select the harmless sampling setting or calibration
record. Calibration companions always target an individual, camera-free robot,
just as benign calibration records do. If the real fault is site-wide, that
robot belongs to its site and the two events share their onset. This closes
another visible cue found during preparation: a planted
site-scoped calibration record had no benign counterpart. D2 can use this
second record type; a fleet image defect occupies every scene camera, so D3
uses the harmless sampling setting. A conflicting explicit request is refused.
The focused scope test and the three legacy fixture hashes pass; stored worlds
are not regenerated.

The fault-library handoff `a57f337` adds the object-suite inert calibration
record: 98/100 placed successes against the same 98/100 healthy successes,
paired difference 0.0 points (95% interval −4.04 to +4.04), with all 100 full
chains identical. This is the library builder’s measurement, not a new run
here. Its `curves.json` is imported unchanged. The nine additional stage-3
object mechanisms remain unmeasured and cannot supply effective causes.
The later library handoff `8c3e6e2` changes verdict/provenance metadata, with
all measured values unchanged. The world reader now prefers explicit
`run_dirs` on points and joints, retaining the legacy `run_id` fallback. New
independent records carry source, healthy-twin and single-arm directories;
its separate library API and engine migration remains with the orchestrator.
The library's spatial `records.measured` API still reads only the original
type table, whereas the world reader also reads these newer supplement points.
That library-side integration is tracked separately; the composer reads
the independent counts correctly. Metadata changes deliberately require an
explicit recipe-pin review before composition.

## Round 3 screen results (selection evidence only)

The frozen `c37d5c1` screen completed all 2,000 planned attempts and all **200/200
replay comparisons were identical**, with no recorded incidents. Healthy controls
succeeded on 98/100 spatial and 99/100 object attempts. Each row below uses
100 identical attempts in all four arms. A/B are delay/chunk successes; joint
loss is healthy minus joint success, in percentage points, with its paired
Newcombe 95% interval. Excess loss is A + B − joint − healthy, with the
predeclared paired-bootstrap 95% interval. Corrected success applies the largest
recorded suite healthy rate to joint/healthy; it is the composer's floor quantity.

| Suite | Delay / chunk | A / B / joint successes | Joint loss, points (95%) | Excess loss, points (95%) | Corrected success |
|---|---|---|---|---|---|
| Spatial | 8 / 10 | 100 / 92 / 46 | 52 (41.33–61.46) | 48 (37–59) | 45.44% |
| Spatial | 8 / 15 | 100 / 99 / 63 | 35 (25.36–44.72) | 38 (28–49) | 62.23% |
| Spatial | 10 / 15 | 98 / 99 / 40 | 58 (47.19–67.11) | 59 (48–70) | 39.51% |
| Spatial | 10 / 20 | 98 / 97 / 70 | 28 (19.09–37.49) | 27 (17–38) | 69.14% |
| Object | 8 / 10 | 93 / 97 / 33 | 66 (55.33–74.48) | 58 (47–69) | 32.83% |
| Object | 8 / 15 | 93 / 98 / 25 | 74 (63.69–81.50) | 67 (56–78) | 24.87% |
| Object | 10 / 15 | 98 / 98 / 2 | 97 (90.31–98.67) | 95 (90–100) | 1.99% |
| Object | 10 / 20 | 98 / 97 / 36 | 63 (52.27–71.77) | 60 (48–71) | 35.82% |

The fixed rule selects **8 / 15 on spatial tasks** (closest eligible success to
60%; no candidate qualifies on both suites). Every single costs at most six
points in this screen. The other doses and their failures remain records;
none is substituted after confirmation. The screen is not imported as usable
calibration and is not pooled with the independent stage.

The driver used 6,404.858 seconds and exited 124 because its forecast for the
unstarted 200-per-arm confirmation was 5,952.469 seconds, beyond the remainder.
This was a resource refusal, not a simulation timeout. Conservatively charge
6,415 seconds including startup/shutdown, leaving 4,385 of the 10,800-second
allowance. The predeclared 100-per-arm contingency uses the same dose, frozen
`d077a13`, six workers and 800 independent attempts; its forecast was 3,006.234
seconds. Its first launch exited 127 before simulation because `/usr/bin/time`
is absent. That log is preserved. A file-only watcher missed the pre-stage
failure; direct process polling caught it about thirteen minutes after admission.
The retry uses Bash's built-in timer and direct process polling. Conservatively
charge the failed launch 60 seconds: 6,475 seconds were charged before retry,
leaving 4,325. The retry's outer timeout is 4,300 seconds plus ten seconds of
kill grace, with startup/shutdown margin. It finished at exit 0 in 2,147.502
seconds (2,158 charged including margin), leaving 2,167 seconds of the allowance.
Raw manifests, chain evidence, analyses and timing receipts are under
`~/.cache/robots-probes/worlds-v3/20261010-mild-pair/`.

### Independent confirmation of 8 steps / 15 actions

All 800 planned attempts completed: 100 per arm per suite, indices 100–199,
disjoint from the screen. The healthy controls succeeded 98/100 on each suite.
All **81/81** held replay comparisons were identical. Worker 5 aborted with
signal 6 on `e004187`; the engine recovered that attempt and its additional
replay was identical. Its native-abort cause is unknown; this
was not a reported memory-guard refusal. Original, partial and recovery files
are preserved. The clean frozen source is `d077a13`.

| Suite | A / B / joint successes (100 each) | Joint loss, points (paired 95%) | Excess loss, points (bootstrap 95%) | Corrected joint success |
|---|---|---|---|---|
| Spatial | 97 / 99 / 55 | 43 (32.75–52.74) | 43 (33–53) | 54.33% |
| Object | 95 / 97 / 31 | 67 (56.23–75.32) | 63 (52–74) | 31.16% |

Spatial qualifies: each single costs at most one point, the excess interval
excludes zero, and the joint remains above the 40% floor. Object does not qualify
for placement. The corresponding single-effect bootstrap loss intervals are:
spatial delay 1 point [0,3], chunk −1 [−3,0]; object delay 3 [−1,8], chunk 1
[−2,5]. Nothing is pooled with the screen and no second candidate is selected.

Using these independent points, D2 passes the composer at attempt 16 with a
site-scoped interaction, six observed causes and 67.75% expected observed
success. D3 still fails 100 attempts with the measured 4-pixel fleet release.
The predeclared 3-pixel fallback was therefore queued: 200 faulted attempts with
the confirmation's 200 healthy twins, same frozen source, six workers. Its
forecast is 677.407 seconds; the hard timeout is 2,140 seconds plus ten seconds
of kill grace, within the remaining 2,167 seconds. A sandbox-only invocation
failed on the read-only queue lock before admission and consumed no heavy
execution; its log is preserved. The host-level invocation completed normally.
No world is sealed or run from this preliminary composition.

The fallback completed all 200 attempts with **20/20** replay comparisons identical,
no incidents, and exit 0. Spatial success was 95/100 versus 98 healthy: loss
3 points, paired 95% interval **[−1.48,8.66]**, exact paired p=.25. Object success
was 80/100 versus 98: loss 18 points **[9.39,27.10]**, p=.000121. The spatial
point is a small cause, not a confirmed effect; consequently **3 pixels is not
accepted as the fleet fault** under the declared both-suite criterion. These
negative results are retained; no new blur dose is selected from them. The stage
took 323.306 seconds. Including conservative startup/teardown margins and the
failed wrapper launch, total heavy execution charged is **8,967 of 10,800 seconds**
(2 hours 29 minutes 27 seconds). No further GPU calibration is planned.

The spatial-only decoy placement uses existing measurements, not new GPU runs:

| Dose | Faulted / healthy successes (1,000 pairs) | Faulted minus healthy, points (paired 95%) |
|---|---|---|
| Lighting, 2 stops dimmer | 980 / 968 | +1.2 (+0.03 to +2.45) |
| Backdrop, walls and floor, blend 1 | 979 / 968 | +1.1 (+0.02 to +2.27) |
| Warm lamps, level 1 | 964 / 968 | −0.4 (−1.74 to +0.92) |

These retain the library's decoy rule: the exact paired test does not reject
zero change and the interval excludes a two-point loss. The first two Newcombe
intervals barely exclude zero on the improvement side; this is not a claim of
identical outcomes. All three are restricted to spatial tasks, including the
agent's experiment task menu; the harmful object measurements are not ignored.

`w44` and `w45` remain untouched integration records. Their replacements follow.

## Round 3 replacement worlds, sealed before runs (2026-10-10)

D2 is **w46**, D3 **w47**. Both show neutral fleet id `w01`. Each has 12 robots,
3 sites, 10 days and 1,680 planned attempts: 1,008 observed (days 0–5), 336
experiment (6–7), and 336 scoring (8–9). Sites s1 and s3 run spatial tasks only;
s2 retains the seeded mixture of both suites. Experiment task menus obey the same
restrictions. This places the confirmed interaction and flat decoys only on their
eligible suite. The two development seeds remain private to truth-side artifacts
and pinned development recipes; the separate external test-recipe path is unchanged.

The resulting task emphasis is substantial: w46 has 1,510 spatial / 170 object
attempts (900 / 108 in the observed window); w47 has 1,538 / 142 (928 / 80 observed).
Both retain all twenty task types. These are measured manifest counts, not a claim
that the two suites receive equal representation.

The measured 4-pixel fleet release is in **D2**, because the 3-pixel candidate did
not confirm a spatial effect. D3 has no fleet release and keeps its two-thirds
configuration/software share (8 of 12 effective/inert entries, excluding its decoy).
D2 uses four plain faults, a 64.5–72.5% composition aim and no required clean robot;
D3 uses five plain faults and the original 64.5–70.5% aim. All mandatory guards,
including six observed causes, the 40% robot-day floor, both pairs and a warnable
linear drift, stay enabled. D2 accepted deterministic draft 341; D3 accepted draft 7.

Predictions below are logged success, with nominal 95% **prediction** intervals
(512 independent Jeffreys-binomial calibration/sampling draws). They are conditional
on suite ratios: coverage is not validated and excludes paired covariance, task
heterogeneity, within-robot dependence and model misspecification. D2’s observed
interval extends above 75%; the prediction does not guarantee the measured gate.

| World/window | Expected success | Nominal 95% prediction interval |
|---|---:|---:|
| D2 observed | 72.38% | 68.65–76.39% |
| D2 experiment | 54.69% | 46.43–62.57% |
| D2 scoring | 54.05% | 45.47–62.04% |
| D3 observed | 65.13% | 60.30–70.34% |
| D3 experiment | 56.69% | 49.93–62.20% |
| D3 scoring | 56.65% | 49.70–61.90% |

Removing one fault alone defines every gain below; other faults remain. The two
interaction members count separately when each reaches two observed points. The
additive pair gets no joint cause. A data-detector gain is logged success only,
not a physical-success benefit or a claim-3 fix. “Behaviour” means no planted change
log or health signal; dose effects must be inferred from outcomes/telemetry.

### D2: w46

World hash: `11097c8af424ac60cf238dffcddfb8510c8a0e142a813ac75e3fea377209fae2`. Prediction seal: `0258accd9e6f5923e9c161694142d73b10c4925bef76ac941641ac4b2d9b1cb5`.
513 catalogue actions / 15 planted entries = 34.20 actions per entry.
Lowest expected robot-day success (minimum of true/logged): **40.0323%**, r02, day 4.
At that row, true success is 63.2768% and logged success 40.0323%.

| Fault | Type | Class | Scope | Role | Visible in | Observed loss (points) |
|---|---|---|---|---|---|---:|
| f01 | action_latency | configuration | r11 | interaction | change_log | 2.724 |
| f02 | policy_knob | configuration | r11 | interaction | change_log | 2.587 |
| f03 | object_batch_variant | physical | s1 | additive_pair | behaviour | 2.520 |
| f04 | gripper_wear | physical | r04 | additive_pair | health | 0.491 |
| f05 | software_release | software | fleet | fleet | change_log | 7.238 |
| f06 | gripper_wear | physical | r03 | drift | health | 0.060 |
| f07 | furniture_shift | site | s2 | confounder | behaviour | 2.557 |
| f08 | policy_knob | configuration | s2 | inert | change_log | 0.000 |
| f09 | backdrop_change | site | s1 | decoy | behaviour | 0.000 |
| f10 | illumination_change | site | s3 | decoy | behaviour | 0.000 |
| f11 | action_latency | configuration | r01 | plain | change_log | 0.417 |
| f12 | gripper_wear | physical | r10 | plain | health | 1.582 |
| f13 | success_detector_error | data | r02 | plain | behaviour | 2.500 |
| f14 | object_batch_variant | physical | s2 | plain | behaviour | 2.389 |
| f15 | gripper_wear | physical | r07 | small | health | 0.020 |

Observed causes (7): f01 (2.724 points), f02 (2.587 points), f03 (2.520 points), f05 (7.238 points), f07 (2.557 points), f13 (2.500 points), f14 (2.389 points).
Drift: f06 on r03, own five-point crossing day **8**.
Three largest single-fault fixes over scoring: f05 at fleet, software_release, **21.588 points**; f03 at s1, object_batch_variant, **3.171 points**; f14 at s2, object_batch_variant, **3.125 points**.

### D3: w47

World hash: `88754d6f8811fe1019cc03cc7902436906b8b391ea60aac57c29e57a5fe9f30e`. Prediction seal: `dcd5319ca1cb4dcac12f2d3c93ad04d1863485140ea2221a311ba702945f9ead`.
513 catalogue actions / 13 planted entries = 39.46 actions per entry.
Lowest expected robot-day success (minimum of true/logged): **40.4677%**, r03, day 2.
At that row, true success is 40.4677% and logged success 40.4677%.

| Fault | Type | Class | Scope | Role | Visible in | Observed loss (points) |
|---|---|---|---|---|---|---:|
| f01 | action_latency | configuration | s1 | interaction | change_log | 11.216 |
| f02 | policy_knob | configuration | s1 | interaction | change_log | 10.654 |
| f03 | object_batch_variant | physical | s3 | additive_pair | behaviour | 2.323 |
| f04 | gripper_wear | physical | r09 | additive_pair | health | 0.411 |
| f05 | camera_pose_offset | sensor | r05 | drift | health | 0.182 |
| f07 | action_latency | configuration | s3 | confounder | change_log | 8.453 |
| f08 | calibration_update | configuration | r11 | inert | change_log | 0.000 |
| f09 | illumination_change | site | s3 | decoy | behaviour | 0.000 |
| f10 | software_release | software | r07 | plain | change_log | 2.361 |
| f11 | action_latency | configuration | r06 | plain | change_log | 1.140 |
| f12 | software_release | software | r08 | plain | change_log | 2.756 |
| f13 | software_release | software | r03 | plain | change_log | 0.770 |
| f14 | camera_pose_offset | sensor | r04 | plain | health | 0.370 |

Observed causes (6): f01 (11.216 points), f02 (10.654 points), f03 (2.323 points), f07 (8.453 points), f10 (2.361 points), f12 (2.756 points).
Drift: f05 on r05, own five-point crossing day **8**.
Three largest single-fault fixes over scoring: f01 at s1, action_latency, **13.120 points**; f07 at s3, action_latency, **12.680 points**; f02 at s1, policy_knob, **12.523 points**.

D2’s interaction is delay 8/chunk 15 on r11; D3’s is the same pair at s1.
Both additive pairs are half-worn pads plus four object-mass doublings: D2 r04/s1,
D3 r09/s3. D2’s confounder is furniture shift plus five sampling steps at s2;
D3’s is delay 17 at s3 plus the inert camera-calibration record on r11, both day 2.
The latter robot has no camera fault; its scope, four-decimal values and author
distribution match benign records. Both drifts start above zero on day 0 and move
linearly on all ten days, with health readings moving before their day-8 crossing.

`tests/test_worlds_round3.py` recomposes these exact real-data recipes using pinned
software identifiers and task text, verifies both world hashes and prediction seals,
and checks sizing, patterns, drift, floor, task restrictions and catalogue capacity.
No cache path is required by these tests. The earlier rule-boundary tests still test
refusal of broken guards and external private test-recipe creation. Negative 3-pixel
evidence is imported and tested as **rejected for fleet use**, not discarded.

No full world simulation, export, oracle, smoke-world or additional agent session is part of this
round’s handoff. The calibration’s 3,000 planned attempts completed; its 301/301
held comparisons were identical. The recovered confirmation SIGABRT remains an
open engine incident. The single final **`make check` exited 0**:
lint and formatting passed, mypy checked 272 source files, and pytest reported
**1,329 passed, 2 skipped, 62 deselected** in 334.02 seconds. No second full check
was run. The engine/simulator implementation is unchanged; no separate check-sim
or smoke job was started. Actual new-dose runtime evidence is the calibration and
its held replays, not a claim about the unrun worlds.

### Frozen run commands (not executed in this round)

The detached checkout `/home/cs/.cache/robots-worlds-run-r3` is at **`a7d2da1`**,
with its own environment (`make setup` exited 0). This round did not change the old
`b81fe81` run checkout or any w44/w45 record. From the new checkout, the orchestrator
can queue these commands, one world per heavy job:

```sh
cd /home/cs/.cache/robots-worlds-run-r3
tools/heavy-run uv run --locked python -m fleetlab.worlds run D2 --workers 6
tools/heavy-run uv run --locked python -m fleetlab.worlds run D3 --workers 6
```

These use batch size 4, `fp16_res32` and the pinned MIOpen database. They run the
observed and later windows, records/events, export and summary; they do not run
the oracle. Keep this checkout frozen while queued or running. The world's
`files/generator.json` records all window predictions and their seal before any run.

CPU generation of both worlds exited 0 from that frozen checkout. Their written
contract hashes and prediction seals match the repository fixtures. An independent
file audit checked all 1,680 unique identities per world, every derived scene/noise
seed, all 20 task types, the spatial restrictions, neutral description id `w01`,
and absence of the private seed from the description. Neither world has a run
directory. The eight exporter-owned files still exactly equal `7741ee5`.

## Round 2 review repairs and feasibility, 2026-10-09

This section records the historical `d02cc05` delivery, before the round-3 measurements above.

There are two calibration boundaries. They are refused, not relaxed:

- The only confirmed measured interaction is delay14/chunk10. Its joint
  succeeds 27/300 spatial attempts against 296/300 healthy and 0/300 object
  attempts against 294/300 healthy. With the corrected spatial healthy
  reference 968/1,000, the highest predicted success for a carrier is
  `(968/1000)*(27/296) = 8.8297%`; any mixture with object tasks is worse.
  No placement can meet a 40% robot-day floor under this estimator. A milder
  measured pair or an explicit change to that requirement is necessary.
- The completed object-decoy measurements imported verbatim from `cc2df59`
  are dimming 969/1,000 versus 985 healthy, backdrop 970/1,000 versus 985,
  and warmer light 584/600 versus 593. All three are classified as effects;
  none is confirmed flat. The current mixed-suite site schedules therefore
  have no eligible decoy placement, even when the floor is disabled only
  for a diagnostic feasibility check. No optional extension acquired
  both-suite evidence in that partial handoff. No calibration was invented.

Consequently there is no new composition, cause list, drift day, minimum
robot success, top-three fix ranking, sealed prediction or runnable frozen
commit to hand off. `check D2 D3` refuses at the interaction/floor boundary.
The new rules are tested on CPU fixtures; those fixtures explicitly use
synthetic flat object decoys and omit the production floor. Their hashes are
regression seals only, **not development-world predictions or measurements**.

The repair imports `main` and the final exporter `7741ee5`. All eight owned
exporter code/test/document files are byte-identical to that commit; this
shared README preserves its final repair text separately. Both stored exports
passed its read-only audit: 3,033 files per world, zero hits, zero structural
problems (359,620,818 bytes for w44 and 399,120,750 for w45). No export or
historical runner receipt was rewritten or recertified.

The previously omitted w44-later incident was worker 0 exiting with code
−6 (SIGABRT) while assigned `e000071`. The job was retried. Its stored as-run
and replay chains match over all 151 steps, including state, both cameras
and executed actions. The abort's cause is not established by those files.
The separate w45 rerun belongs to the orchestrating session.

CPU checks cover all twelve review requirements and the previously untested
composer guards. An isolated, sequential mutation probe caught 24/24 targeted
breaks on assertions (not collection errors). That is a bounded probe, not an
exhaustive proof of test strength. Detailed receipts are under
`~/.cache/robots-probes/worlds-v2/20261010-worlds-r2/`.
One full `make check` ran: 1,282 passed, two skipped, 62 deselected, with
twelve dashboard fixture errors and one oracle error-message assertion.
The dashboard still requested the deliberately refused production recipe;
it now uses the explicitly synthetic CPU fixture. The oracle assertion now
checks the earlier contract rejection of an unknown additive-pair member.
The simulator test's fixture was switched in the same way, without running
the simulator. A failed-test-only follow-up first stopped at a raw-regex
lint correction, then `PYTEST_ADDOPTS=--lf make check` exited **0**: lint,
formatting and types passed, and all thirteen failed tests passed. Across
the full run and the repair run: **1,295 passed, two skipped**; the full
suite was not repeated. These checks used one test process at a time.
The 47 focused regressions are in `tests/test_worlds_review.py`. Key checks:

| Rule | Regression (function suffix after `test_`) |
|---|---|
| 1, observed sizing | `observed_sizing_is_distinct_from_either_window_detectability`; `summary_gate_uses_observed_causes_frozen_band_and_catalogue_ratio` |
| 2, cause units and flags | `causes_remove_each_member_in_the_composed_world`; `additive_pairs_never_become_joint_causes`; `overlapping_pairs_are_not_merged`; `fault_table_and_detectability_flags_cannot_disagree` |
| 3, measured fleet effect | `fleet_fault_requires_measured_effect_on_all_its_suites` |
| 4, linear warnable drift | `linear_drift_crossing_uses_its_own_scheduled_task_mix`; `warnable_drift_requires_a_visible_precursor` |
| 5, daily floor | `robot_day_floor_checks_the_combination_and_none_disables_it`; `known_interaction_cannot_meet_the_production_floor` |
| 6, harmless records | `calibration_record_occupies_its_camera`; `planted_calibration_uses_benign_values_and_authors` |
| 7, pins | `recipe_regression_hashes_fail_on_recomposition`; `sealed_hash_blocks_any_changed_world_before_writing`; `new_calibration_requires_a_new_recipe_even_if_the_points_are_unused` |
| 8, private test creation | `private_recipe_cli_creation_and_resume_are_truth_only`; `seed_draw_uses_all_62_bits`; `no_test_recipe_is_in_repository_and_symlinks_do_not_bypass_boundary` |
| 9, exporter | Eight-file byte comparison with `7741ee5`, its unchanged `test_log_export.py`, and both complete stored-export audits |
| 10, estimator | `healthy_reference_keeps_the_larger_measurement`; `prediction_intervals_are_reproducible_and_acknowledge_their_limits`; `summary_and_generator_record_include_the_sealed_prediction_intervals` |
| 11, contract and incident | `patterns_schema_is_closed_in_both_modes`; direct 151-step comparison of w44's retried episode |
| 12, negative guards | Paired support, confounder/drift detectability, four-fault cap, twenty carriers, transfer, recipe count, frozen band/catalogue, configuration share and per-suite decoy tests |

No GPU job, MuJoCo acceptance check, smoke-world, production oracle, agent
model session or production test world was run in this round.

This package implements the worlds of the experimental design and the
contracts in `docs/design/contracts.md`. The fault library and its calibration
method are documented in `docs/design/faults.md`. This README is
implementation and measurement evidence;
pattern confirmation by the oracle is a separate handoff.

## Running and reproducing

```sh
uv run --locked python -m fleetlab.worlds check D2 D3
uv run --locked python -m fleetlab.worlds generate D2
make world WORLD=D2
make world WORLD=D3
uv run --locked python -m fleetlab.worlds summary D2
```

A GPU run must start from a detached checkout at the chosen commit through
`tools/heavy-run`; never edit that checkout while a job runs. The runner
writes new roots under `~/.cache/robots-probes/worlds/<world id>/`:

- `files/`: sealed specification, fleet model, visible description and
  manifest; hidden fault table, detectability, health, recipe and curve sources;
- `runs/<id>-observed/` and `runs/<id>-later/`: as-run episodes, records,
  health checks, derived events, and chain receipts;
- `export/`: the observed window for analysis agents;
- `runner.json`: progress, resumable steps and measured-versus-predicted summary.

The generator and runner do not run the oracle. Existing oracle results lie
under a world's `oracle/` and are owned by that separate pipeline. The
version-0 miniature, nightly and smoke generators remain separate.

## Hidden generation seeds and public identity

New recipes draw their hidden generation seed with `secrets.randbits(62)`
and set bit 62, giving 62 independent random bits in the signed-int64 domain
`[2**62, 2**63)`. D2 and D3 pin independent draws in their truth-side recipes
for reproducibility. Validation rejects small seeds and a public fleet label
that equals the hidden world id or encodes the seed. Historical D1 explicitly
uses `legacy_identity=True`, an exception restricted to D1/w41/seed 41.
Composition revalidates recipe copies, whose Pydantic `model_copy` otherwise
skips validation. D1's original seed and identity remain unsafe for
blind evaluation, and its stored records remain untouched.

The visible fleet label is `w01` in each independent replacement export.
The truth-side directory identifiers `w44` and `w45` occur in hidden world
specifications, manifests and run receipts, never in the exported tree.
Robot and site identifiers are fixed neutral counters. Episode identifiers
are a permutation driven by the hidden seed; their visible order is sorted
by identifier. Filenames use only these neutral episode identifiers and
camera labels. Clock times are a public function of the within-day episode
index. The clocks still reveal that index; the hidden seed remains necessary
to derive the scene and noise streams.

Other visible quantities influenced by the hidden seed are task selection and
site emphasis, start states (through images), component service histories,
benign calibration and maintenance records, change authors and dates,
software deployment dates (version numbers follow a fixed sequence), fault
placement and timing (through their operational
effects), health readings, telemetry, outcomes and video. These outputs offer
offline checks of a guessed seed; their protection rests on the large random
seed domain, not secrecy of the generator algorithm. The generation seed is
stored as a Python/JSON integer and hashed as its exact decimal integer; it
fits signed int64. Scene seeds remain unsigned 32-bit values and noise seeds
signed 63-bit values in the existing int64 identity columns. The replay hash
scheme is unchanged.

`test_worlds_blinding.py` sends a generated export and the agents' actual
source bundle to the seed-leak attacker calculation. Only the verifier sees
truth. It checks that the visible fleet suffix and clocks recover neither the
world seed nor either episode seed, and scans names and decoded table values
for the hidden seed/id. Its telemetry is synthetic, so this is an export and
identifier regression, not authenticated sandbox or simulator evidence.
The same attack is also applied to the completed live exports at delivery.

## Paired interaction evidence

`curves.load()` reads both the world supplement and the library's
`faults/curves.json`, including its supplement points, healthy rates and
joints. The library's entries supersede matching older world-supplement
entries. Each source's digest is recorded. A historical joint without
same-attempt singles can inform a joint success prediction but cannot qualify
an interaction.

For an interaction verdict, the record must contain healthy, A-only, B-only
and joint success counts on the same attempts, plus the paired bootstrap
interval of `joint - A - B + healthy`. Only an interval wholly below zero
qualifies as an interaction. An interval containing zero is marked additive:
this means consistent with addition at this measurement's precision, not
proof of exactly zero interaction. The composer checks the scoring robot's
actual suite mix; it weights the suite intervals by that mix and requires
measured support on every represented suite. These weighted bounds are not a
fresh pooled bootstrap interval. Effects in metadata use attempts as their
denominator, not healthy successes.

The bowl-task library measurements are:

| Pair | Attempts | Healthy | A alone | B alone | Together | Interaction, percentage points (95% interval) |
|---|---:|---:|---:|---:|---:|---|
| pads worn 0.5 + four mass doublings | 300 | 296 | 281 | 258 | 234 | −3.0 (−7.0 to +1.0), additive |
| 14-step action delay + 10 actions per chunk | 300 | 296 | 277 | 273 | 27 | −75.67 (−80.67 to −70.33), interacting |

The object-task handoff (`fix/faults-review` commit `6bbe093`, 2026-10-09)
measured the delay/chunk pair on the same 300 attempts: 294 healthy successes,
188 with delay only, 282 with short chunks only, and zero jointly. Extra loss
beyond the two singles is 58.67 percentage points, paired bootstrap 95%
interval 52.33–64.67 (10,000 common-index resamples). All four runs' replay
samples passed. Only that commit's `faults/curves.json` was imported; it was
measured from `cfedf66` before the fault builder's engine repairs. Both suites
therefore support the interaction. No `OBJECT CURVES MEASURED` handoff was
available when the replacement recipes were composed, so optional extension
kinds requiring object calibration were not selected or waited for.

The legacy world supplement used a different seed stream for its bowl-task
joints than the borrowed singles. Its analysis now never borrows singles:
all four outcome sets must come from the supplement itself, cover the same
indices and have matching attempt identities. Missing singles stay missing.
Object-task wear/mass singles were already in that run; their paired
statistics were recomputed locally from the saved records. No old result
file was rewritten. Future supplement runs append the missing bowl singles
without renumbering existing blocks.

## Composition and hidden truth

The interaction is now a 14-step action delay and ten actions per policy
chunk on the same robot. Worn pads with a heavier site batch are also
planted, labelled `additive_pair` in metadata and recorded in
`Patterns.additive_pairs`. That field has the same pair-of-fault-ids shape
as `interactions`; it is omitted when empty to preserve old serialized
contracts. Both members remain effective faults for oracle replay.
Switching only the pair label changes neither jobs, health nor visible fleet
records. Section 10 entry 8 now judges every effective fault by removing it
alone from the composed world, including its interaction contribution. Two
qualifying members are two causes; one qualifying member is one cause plus a
small cause. Only if neither qualifies can a qualifying marked interaction
pair count once. Overlapping pairs are not merged. Additive pairs never get
that fallback. The fault table copies the same per-fault effects, qualifying
windows and flags. `Patterns` has a closed schema in both validation and
serialization modes while retaining omission of empty additive pairs.

A software release can carry either one calibrated member or both members,
using a calibration label or `kind|label + kind|label`. The estimator prefers the release's own measured point on the requested
suite (8 successes in 100 attempts against 98 healthy successes for the bowl-task
two-member release, ratio 8/98), then the measured joint (27 successes in
300 attempts against 296 healthy successes, ratio 27/296 on bowls), including when its version has been renumbered.
It does not multiply the two single ratios when that joint is available.
The development worlds use separate configuration changes so both fixes
can be considered independently.

Decoy neutrality is suite-specific. Placement now requires a confirmed-flat
point on every suite the target's scheduled attempts carry, with at least
1,000 calibration pairs. The estimator retains measured harm for an
unconfirmed point; it does not silently zero an object effect. The
`calibration_update` companion is inert by construction. Its camera key and
the corresponding physical camera share occupancy, so it cannot shadow a
camera fault, including a carried image-pipeline fault. Planted and benign
calibration records use the same seeded, four-decimal value distribution
and the same author distribution. Subsequent records start from the preceding
value in chronological order. The older sampling-steps companion remains
an option with its measured uncertainty.

The extension menu admits a level only when both task suites have measured
points. It includes lens contamination, camera latency, gripper force loss,
sticky release, arm torque loss, controller overload, image processing
releases and table height changes. Warm-light decoys become available when
their object curve is present. New image processing releases receive the
same deployment-version handling as carried releases. Extended recipes use
the vector forms of lighting and placement uniformly at every site, even
when no fault of those kinds was selected; neutral condition values and
calibration lookup accept equivalent zero padding.

`success_false_positive` is deliberately absent from this multiplicative
menu: its calibration is conditional on failed attempts and companion faults,
not a success ratio applicable to an arbitrary scene. New extension faults
are planted at measured levels. Drift interpolation remains an assumption;
it is not used to invent extension levels below a measured knee.

A requested effective fleet fault must have a measured effect on every
suite it acts on. The fleet menu uses a logged software release carrying the
measured four-pixel image blur, rather than calling five sampling steps an
effective fleet fault. This is an available composition choice, not a new run.

Drifts are linear from day 0 through day 9. Every fault-table row has
`drift_crossing_day`: the first scheduled day on which that drift **alone**
adds at least five percentage points of failure on its target's actual task
mix, or null. Warnable recipes require that day in scoring (8 or 9), and a
nonzero health precursor before it. A late crossing can be a small cause;
it need not supply one of the six observed causes. The draft can use camera
pose drift, or gripper wear when a fleet image release occupies the cameras.
The crossing below measured dose points relies on explicitly stated linear
interpolation; it is not a measured onset of failure.

`robot_success_floor` is optional. With `None` the floor check is disabled;
with 0.40 it checks true and logged expected success on **every** robot-day,
using all simultaneous faults and that day's actual suite mix. Checking
singles separately cannot satisfy it. D3 still requests at least two thirds
configuration/software faults. There is no exception for the interaction.

## Runner and export repair handoff

Export review repair, 2026-10-09:
the export step checks the observed chain receipt before any skip. A published
export is freshly checked unless a receipt matches both its content digest and
the current checker and term-list hash. It never trusts a saved `leaks_clean`
flag alone, including partial calls that stop before the export step. A kill
after publication and before
writing the result to `runner.json` therefore recovers without deleting or
rewriting the export. Readiness is cleared at the beginning of a runner call,
so a later failed check cannot leave a previously ready world marked ready.
The full fleet supplies hidden names only; the observed fleet supplies records.
New receipts are appended under `export-checks/`; legacy `runner.json` records
need no rewrite. The layout, oracle, live-data, teleoperation and dashboard
readers all use this same readiness rule. A refused check clears `leaks_clean`.
At a fresh audit, truth discovery includes sibling runs and the world's oracle
directory. The receipt freezes those terms; future oracle/scoring record names
also come from code declarations. Later oracle and reader writes cannot
invalidate a receipt for unchanged export bytes and unchanged checker code.
The runner audits a new staging tree once before publication, recording
`audit_seconds`. Its receipt binds the checked relative names and bytes to the
final export path, and only a clean tree is atomically renamed there. A refused
tree never stands at `<world>/export`. A saved receipt can therefore recover
publication interrupted before the runner saves its result;
a matching receipt needs no fresh audit. A partial call that ends with an
uncertified export clears the saved `leaks_clean` flag.
Changed source logs and missing per-episode files are separate integrity work,
beyond word/structure checking.

Stall events keep their previous steps and durations. Their text describes
two-second windows, each bounded relative to that window's start;
it no longer bounds the whole merged interval by 1.5 cm. A synthetic arm
drifting 0.3 mm per step demonstrates why the distinction matters. CPU fixtures
exercise publication followed by interruption, repeated recovery with byte-
identical exports, and a planted trait leak after a saved clean receipt. No
simulator, GPU or authenticated agent is exercised by this repair.
Adjacent qualifying windows can merge without overlapping; their details make
no overlap claim. Stored `w42` and `w43` retain their historical texts by the
orchestrator's decision, since they retire when `w44` and `w45` finish. D1 uses
the already rebuilt export below for procedure development, with provisional
scores under the recorded seed-leak decision.

The new D1 export is
`~/.cache/robots-probes/worlds/w41/export-review-20261009T173751Z/`, with
35 corrected stall texts and no other file differences. The bounded audit
also confirmed a seed leak through D1's public identifier, so
a passing word scan does not clear D1 for a blind analysis. Experiment refusal
messages and per-session gates have separate follow-ups.
Full receipts are under
`~/.cache/robots-probes/export-review/20261009T173751Z/`.

The first follow-up's final check exited 0 with 1,156 tests passed, 2 skipped
and 62 simulator/model tests deselected. The four complete scans (stored
`w41`, `w42`, `w43` and rebuilt D1) each covered 3,033 files and found zero
word hits and zero structural problems. Fresh receipts bind them to the
current checker and export digests, while the historical runner records stay
unchanged. The evidence, including exact term lists and the D1 comparison, is
under `~/.cache/robots-probes/export-review/20261009T203851Z/`. These are
CPU/data checks; no simulator, authenticated agent or real kill was exercised.

The second follow-up tests the eleven tool descriptions, standing prompts and
engine event details against 429 real `w41` truth-side key names, and runs a
CPU toy world through the runner, oracle plan, oracle execution and resumed
execution while checking that it stays ready. Both regressions fail on
`88901a8` and pass on the repair; no oracle code changes are needed. Four
formerly surviving mutations now fail on assertions. The check caught 127 of
the same 165 review plants, up from 126, with no lost detections.

Recertify a stored world after checker changes without running it or rewriting
its records:

```sh
uv run --locked python -m fleetlab.log certify /home/cs/.cache/robots-probes/worlds/w41
```

Use `--export PATH` for an export beside the original. This appends a receipt
under `export-checks/` and exits 0 only when clean. Reader refusals explain the
stale receipt and include the command for the actual path.

The second follow-up's `make check` exited 0: 1,168 tests passed, 2 skipped and
62 simulator/model tests deselected. Each of the four complete stored scans
again searched 3,033 files, with zero word hits and zero structural problems.
Certification took 102.58 seconds for stored `w41`, 103.39 for `w42`, 102.94 for
`w43` and 100.07 for rebuilt D1, with the four commands running concurrently on
the shared CPU. These durations include tree digests and the audit, excluding
receipt writes. Export and historical runner hashes stayed unchanged.
Evidence is under `~/.cache/robots-probes/export-review/20261009T215026Z/`.
No simulator, GPU oracle, authenticated agent or real kill was run; `w44` and
`w45` were left unopened as instructed.

The third follow-up restores main's literal underscore substring matching,
including plurals and digit suffixes, while keeping boundaries on spaced and
joined forms. All 252 tested suffix forms of main's 84 terms are detected.
The three new regressions fail on isolated `811cda9` source and pass on the
repair: suffix detection, a leaking run never published (one audit, with a
rename spy), and certification when fleet id and world name differ. `certify`
now uses `WorldPaths` with the root name, just like the readers. Summary keys
are truth terms derived from their producer even before a summary is written;
the surrounding engine and runner bookkeeping remains excluded.

The expanded role/pattern prose policy has no conflicts with 40 gate/prompt
and rendered-event texts on main; those texts also pass against all three
stored worlds' actual truth-side names. Ordinary `level` is now name-only as
decided. That changes the same 165-plant battery from 127 to 124 detections:
`level 2.0`, `level -> 2.0` and a list of `level`/`2.0` now pass the word search;
the list is structurally refused. Spaced compound hidden names followed by
numbers remain deliberate hits (`image size 360`, `max steps 280`).

`make check` exited 0: 1,172 passed, 2 skipped and 62 simulator/model tests
deselected. The four full scans again covered 3,033 files each with zero hits
and zero structural problems; their audit times were 106.20/108.65/109.87/105.53
seconds for stored w41/w42/w43 and rebuilt D1, concurrently on the shared CPU.
Historical exports and runner records stayed unchanged. Evidence is under
`~/.cache/robots-probes/export-review/20261009T223340Z/`. No model session,
simulator, GPU oracle, authenticated agent or real kill was exercised;
`w44` and `w45` stayed unopened as instructed.

## Reading a world under any root name

`WorldPaths.from_root(root)` reads the world ID from its records. The composed
world's `files/spec.json` is authoritative; `runner.json` must name the same ID
when it includes one, and provides the ID before the spec is written. Legacy
`world/spec.json` and standalone `spec.json` layouts remain supported. No ID,
invalid IDs, unreadable records or conflicting IDs are refused with the record
paths in the error. A root directory name and a public fleet ID are not world IDs.

The oracle readiness check, export certification, live data preparation (including
dataset labels), teleoperation readiness and dashboard use this resolver. Synthetic
tests build a checked world under `arbitrary-root` with ID `w77` and a different
fleet ID using the existing toy helpers. They check both run paths, caller behavior,
record disagreement and refusal diagnostics. Stored w41–w45 records and paths were
compared read-only; w46/w47 resolve under their existing suffixed roots. No
certification or oracle execution on stored worlds, simulator/GPU job, full check
or smoke run is part of this repair; the orchestrator owns those acceptance checks.
Regressions for all six caller modules fail against the original `70c7230` source;
the oracle and teleop tests also inspect the paths passed to real readiness checks,
since a toy export receipt can otherwise mask the incorrect identity.
Focused verification passed: repository lint, formatting and mypy, plus 422
distinct unit test cases across the layout/runner, oracle, export, live, teleop
and dashboard suites. The oracle/runner suites exercise the CPU toy engine;
they do not validate a stored world's simulator or model execution.

## Prediction and checks

The difficulty model multiplies calibrated success ratios, except for
marked pairs, which use their measured joint. Other overlapping faults are
assumed independent. Effects are marginal removal gains over the actual
world's task mix. Section 5 detectability accepts a marginal gain of at least
two percentage points in either observed or scoring, with twenty observed carriers and an
identifiable target. Section 1 sizing and the summary's gate mark separately
require at least **six confirmed causes in the observed window**. Measured
support on each relevant suite and paired interaction support remain required.
Observed success must lie inside the recipe aim (normally 64.5–70.5%, within
the frozen 60–75% band), scoring success stays above its configured floor,
and no episode carries more than four effective faults. Catalogue sufficiency
is independently tested at its frozen ten-actions-per-fault boundary.

`generator.json` and the world's summary now include both the point prediction
and a nominal 95% interval for each window and outcome. A deterministic
512-draw independent Jeffreys-binomial sensitivity model samples calibration rates
once per point for all episodes that share it, including nonzero uncertainty
after zero observed successes; the predictive interval also
samples the finite window. It reports calibration standard deviation in
percentage points separately. The interval is conditional on the suite-ratio
model. It omits paired covariance, within-robot dependence, task heterogeneity
and model misspecification; its coverage has not been validated. It cannot
make the historical 3.4- and 5.9-point experiment-window misses disappear.
The largest healthy measurement wins: 968/1,000 spatial is no longer replaced
by 98/100; current object healthy support is 985/1,000.

The truth-side generator record pins its recipe to the resulting 64-digit
world hash and calibration source digests, and hashes its prediction record.
Generation rejects a changed seal before writing; new calibration requires a
new recipe. The archived integration recipes also retain their original
world hashes and calibration digests, so current calibration cannot silently
regenerate w44/w45. Historical reproduction uses frozen source `b81fe81`.
The CPU fixture hash regressions exercise this independently of self-comparison.

A test/reserve world is created through an **external recipe file** that omits
`seed`. Its path must resolve outside a Git checkout (including symlinks and sibling
worktrees).
Creation draws the seed; only truth-side files retain it. Resumption reads
`files/generator.json`, reuses that seed and both pins, and refuses a changed
template. The template itself stays unchanged. For an authorized future test:

```sh
uv run --locked python -m fleetlab.worlds generate --recipe /private/recipe.json --root /private/world
```

After a valid development composition has been sealed and its accepted commit
is known, the orchestrator can prepare a **new**, detached run checkout and
run each world there with the usual heavy queue. The existing frozen checkout
is occupied by the orchestrator's w45 rerun and must not be reset:

```sh
git worktree add --detach /home/cs/code/robots-worlds-v3-run ACCEPTED_COMMIT
cd /home/cs/code/robots-worlds-v3-run
tools/heavy-run uv run --locked python -m fleetlab.worlds run D2 --workers 6
tools/heavy-run uv run --locked python -m fleetlab.worlds run D3 --workers 6
```

These are deliberately conditional commands, not queued jobs or a claim that
the present recipes compose. No accepted commit or prediction can be supplied
until the two calibration boundaries above are resolved.

## Historical runs, 2026-10-08

These measurements are retained for comparison; they do not describe the
replacement worlds. Each observed denominator is 1,008 attempts and each
scoring denominator is 336 attempts, X-VLA `fp16_res32`, placed success.

| World | Observed logged success | Predicted observed | Scoring logged success | Predicted scoring |
|---|---|---|---|---|
| D1 `w41` | 651/1,008 (64.6%) | 64.7% | 204/336 (60.7%) | 60.2% |
| old D2 `w42` | 702/1,008 (69.6%) | 68.1% | 180/336 (53.6%) | 51.3% |
| old D3 `w43` | 685/1,008 (68.0%) | 66.9% | 208/336 (61.9%) | 57.3% |

The old D2's true observed success was 71.7%; a detector fault reduced what
it logged. Both old D2 and D3 mislabelled the additive wear/mass pair as an
interaction.

## Limits

Calibration uncertainty and differences between the world's task weights and
the evenly sampled calibration tasks remain. Weighted suite intervals do not
account for variation within a suite. No held-out test world, medium world,
analysis-agent performance, cross-machine replay or oracle-confirmed pattern
claim is part of this delivery. No oracle is run here. The runner and export
review repairs are owned by the export builder; its handoff commits are carried
here without changing its implementation. Only the shared README overlaps
were resolved locally. Measurement receipts and their limits are recorded below.

## Historical integration composition, 2026-10-09 (before review)

The following are predictions from the actual handed-off calibration, before
running the worlds. Both compositions passed on attempt index 7. Their
observed window contains 1,008 attempts; their scoring window contains 336.
Logged success includes the planted success-detector faults; true success
means placement according to the independent check.

| World | Observed logged / true prediction | Scoring logged / true prediction | Detectable causes counted | Planted faults | Catalogue actions per fault |
|---|---:|---:|---:|---:|---:|
| D2, w44 | 68.47% / 69.21% | 56.44% / 57.44% | 7 | 15 | 513 / 15 = 34.2 |
| D3, w45 | 66.57% / 68.68% | 54.15% / 57.30% | 8 | 14 | 513 / 14 = 36.64 |

Both contain the confounder, decoy, interaction, drift and additive pair.
D2 places the interacting configuration changes on r05 (23 bowl and five
object scoring attempts); D3 on r07 (21 bowl and seven object attempts). The
composer prefers a bowl-emphasized site, then checks the actual scoring
suite mix. No task emphasis or task assignment was manually changed to make
the interaction qualify. The additive pair's wear is on r06 in each world;
its heavier batch is site-wide at s2. D3's fleet configuration and step drift
are described above. The calibration-record companion is used in D3; D2
uses the older reduced-sampling companion. D2's two decoys are distinct.

Preflight checks on this implementation and the imported measurement:

- `make check` exited 0: 1,107 passed, two skipped, 62 simulator/model tests
  deselected; lint, formatting and full type checks passed.
- `make check-sim` exited 0: 58 passed, including every selected development
  fault applied and undone through LIBERO. No model is loaded by this check.
- The old dashboard fixture now follows D2's recipe id rather than hard-coding
  `w42`; no dashboard production code was changed.
- D1's newly composed specification equals the stored `w41/files/spec.json`.
  This was a read-only comparison, not a rerun or rewrite of the old world.
- `make smoke-world` exited 0 from frozen `af211b8`: all 16 of 16 chains
  matched the stored reference, with no missing or changed chains. The later
  changes through `b81fe81` affect calibration data, recipe validation and
  documentation/tests, not the smoke runtime. The reference was not changed.
- Both replacement fleet measurements use frozen `b81fe81` and completed all
  scheduled attempts and exports. D2 is ready; D3 is not ready because one
  later-window replay comparison failed, as detailed below.

## Historical integration run evidence, 2026-10-09 (before rerun)

D2 `w44` completed all 1,680 attempts and its runner exited 0. Its observed
and later chain samples held (101 and 68 replay attempts respectively).
D3 `w45` also completed all 1,680 attempts and its export, but is not ready
because of the later replay failure described below.

Independent local checks passed for both observed exports. Each contains
exactly its 1,008 observed attempts; all 1,680 executed identities per world
equal the corresponding sealed manifest. Both full-trait leak audits passed.
The identifier-and-clock seed attack against each actual export with the
agent tool source bundle matched zero scene seeds and zero noise seeds
across all 1,008 episodes in that world. The private recipe module and seed
literals are absent from the bundle. This exercises actual exports and the
bundle locally; it is not an authenticated agent run. The comprehensive
verifier exited 0 for D2 and 1 for D3 on the explicit `ready` assertion;
passing export checks do not clear the failed replay.

| World and window | Logged successes / attempts | Logged prediction | Physical successes / attempts | Physical prediction |
|---|---:|---:|---:|---:|
| D2 observed, days 0–5 | 687 / 1,008 = 68.15% | 68.47% | 696 / 1,008 = 69.05% | 69.21% |
| D2 experiment, days 6–7 | 204 / 336 = 60.71% | 57.34% | 208 / 336 = 61.90% | 58.55% |
| D2 scoring, days 8–9 | 195 / 336 = 58.04% | 56.44% | 198 / 336 = 58.93% | 57.44% |
| D3 observed, days 0–5 | 674 / 1,008 = 66.87% | 66.57% | 699 / 1,008 = 69.35% | 68.68% |
| D3 experiment, days 6–7 | 206 / 336 = 61.31% | 55.39% | 215 / 336 = 63.99% | 58.59% |
| D3 scoring, days 8–9 | 185 / 336 = 55.06% | 54.15% | 195 / 336 = 58.04% | 57.30% |

D2's observed logged success is 0.32 percentage points below prediction;
D3's is 0.30 points above prediction. Both lie within the 60–75% gate band.
Their nominal binomial Wilson 95% intervals are 65.21–70.96% and
63.90–69.70%, respectively. Counts are exact for these fixed episodes;
these intervals do not describe uncertainty between worlds or account for
within-robot dependence. Calibration uncertainty remains separate.
D2's runner took approximately 56 minutes and D3's successful execution pass
approximately 57 minutes, excluding queue waits.

D3's first launch exited 1 before any episode: 28.4 GiB was available, while
ten workers needed an estimated 25 GiB allocation plus the engine's required
8 GiB reserve. Its retry uses six workers, estimated at 16.6 GiB plus the
same reserve. The sealed world, model settings, batch size and source commit
are unchanged. The reserve was not bypassed, and the run checkout was not
edited. The retry completed all 1,680 attempts and its observed export.
Its runner exited 1 and correctly retained `ready=false` because the later
chain sample failed. Its observed sample passed all 101 comparisons; the
runner's observed-export leak audit passed.

Of D3's 68 later comparisons, 67 are
identical and one is broken: `e001417`, r07, day 8, `libero_spatial` task 3.
State, both image hashes and executed actions first differ at step 255.
There are 16 differing finite-action rows; the largest absolute action
difference is approximately 0.0003784. Both original and replay end in
physical failure. The run recorded no engine incidents. This is not an
accepted rendering transient. The mismatch is established with high
confidence; its cause is unresolved and belongs to the engine owner.
No repeat simulation, recorded-chain replacement or reference update was
made in response. The initial D3 scoring window was not cleared for replay-based
evaluation. The orchestrator now owns its rerun and current status.

D3's programmed drift reaches 6.1 degrees on day 8 and 9 degrees on day 9.
Its carrier r09 succeeds physically in two of fourteen as-run attempts on
each day. These unpaired counts do not establish growth in the drift's
causal harm. The steeper programmed profile is verified; causal growth
still requires the oracle's removal comparisons.

Receipts, run logs, the first D3 refusal and verification scripts are
under `~/.cache/robots-probes/worlds-v2/20261009-589ea72/`. The final source
used for both fleet launches is `b81fe8121ad9d05969332f23804cc0cfa05dacae`.
The D3 audit receipt is named `w45-export-audit-and-replay-failure.json`;
`w45-replay-discrepancy.json` records the mismatching episode and chain paths.
No oracle was run; confirmation of individual causes, the interaction,
additivity and drift remains the orchestrating session's next measurement.

## Batch-4 calibration merge (2026-10-10)

The record retains all 105 single measurements and six joints from both parents.
Common numerical cells are unchanged; each type now has suite-specific verdicts,
and every point names its source and healthy runs (joints also name their singles).
The independent mild confirmation remains authoritative, separate from the earlier
8-step spatial cohort. D2/w46 and D3/w47 reproduce their existing world and prediction
hashes without resealing; only the recipe's calibration-file provenance pin changes.
The seal regression reverses an explicit list of provenance-only substitutions when
comparing the historical truth files. Both stored worlds and their run checkout are untouched.

### Round 6 preregistration (2026-10-10, before GPU admission)

The observed/experiment/scoring floor is now per robot **and window**, on
its scheduled tasks. The measured observed acceptance band remains 60–75%.
The forecast adds the observed estimator discrepancy range, −0.7 to +6.0
percentage points, to the conditional predictive interval. The cause target
is six single-fault removals gaining two fleet points in the scoring window,
with at least 95% probability under shared calibration-cell resampling.

The next calibration has one admission and a hard 5,400-second execution
budget, six workers, batch four, fp16_res32 and the ordinary chain sample.
`calibration.REFINEMENT_MENU` fixes the doses before admission: the 8-step /
15-action joint on both suites; spatial camera pan 6°, delay 17, blur 3,
wear .5/.75/1, mass 4 and wear .5 + mass 4; object pan 3°, delay 14, blur 3,
and wear .5 + mass 4. Each existing task cell in [15%,85%] with fewer than
50 attempts is topped up to 50. This is 47 cells and 1,680 added attempts
before folding the development runs (the extra independent observations
remain useful even when a folded cell already reaches 50).

Only **3-pixel blur** is considered as a new fleet release in this plan.
First, 200 same-attempt spatial controls at that level and 200 with each of
the two candidate decoys (2 stops dimmer; walls/floor level 1) measure their
context effects. These are paired anew under the current software, not
assumed equivalent to archived controls with another lock hash. Next come
the thin-cell additions, in fixed menu/task order and batches of at most
400; last come eight 100-attempt spatial contexts: blur with delay 8,
chunk 15, their joint, pan 6°, delay 17, and wear .5/.75/1. Counterfactual
cells must exist independently; a measured triple does not license a
product for its missing subsets. These last exploratory contexts need
further sampling if their task cells are mid-range and still below 50.
The full list is 3,080 attempts: it is a priority list, **not a promise that
it fits ninety minutes**. The hard budget wins; incomplete stages are not
calibration, and missing or thin cells cannot be used to compose a world.
No dose or priority is chosen after seeing these outcomes.

**Preadmission amendment:** the first command was withdrawn while blocked on
the GPU lock, with zero execution and no stage log. The corrected plan also
includes blur + mass 4 and blur + wear .5 + mass 4: these are mandatory subsets
of D2's fleet/additive overlap, omitted from the original list. The folded
world counts also make spatial task 7 at wear 1 a mid-range cell (12/16);
its 34-attempt top-up is included. There are now 3,314 initial attempts.
Before admission, the plan also seals continuation
identities and this rule: a new context whose existing-plus-new true or logged
rate is in [15%,85%] receives enough additional attempts to reach 50; a decoy
task continues all three same-attempt arms together. Up to 4,900 continuation
attempts are available, but the same 90-minute cap applies to the whole job.
Continuations follow each completed new-context stage, before the next one.
This is a fixed stopping rule, not a change of doses after outcomes; intervals
remain Bayesian calibration sensitivity, without a frequentist coverage claim.
Only complete, chain-checked stages enter the record.

Manifests use the canonical healthy attempts as a fixed sampling frame,
with current software identity recorded honestly. These new absolute
outcomes refine prediction cells; they do not add unpaired outcomes to the
previous interaction's paired significance calculation. The two old worlds
are observational calibration, with every active mechanism retained,
including decoys and the five-step sampling setting. Only a truly inert
calibration rewrite is erased. No world is resealed or run by this job.

### Round 6c preregistration (2026-10-10, before GPU admission)

Round 6 is finished by a second builder (branch `stage3/worlds-r6c`). The first job's
thirteen stages are imported (`2081d34`: 705 context cells from 7,614 attempts). With
them, and no code changed, none of the three recipes composes on any of the twenty
seeds `(1 << 62) + 1100000 + k`: the interaction has no carrier on 19, 20 and 19 seeds,
because removing the chunk setting leaves the 8-step delay alone, whose spatial task-7
cell holds 12 of 15; on the remaining seed of D2 and of the planned
recipe all 400 drafts end incomplete.

One further job is planned, with one admission and the same hard 5,400 seconds, six
workers, batch four and `fp16_res32`. `calibration.followup_plan` wrote its manifests
to `~/.cache/robots-probes/worlds-v4/20261010-task-cells-r6c` before this commit;
`tests/worlds_r6c_plan.json` holds the order, the sizes and the SHA-256 of every
manifest, and a test compares the files on disk with it. The order is the priority.
A stage the budget does not reach is not calibration.

1. `topups-1`, 275 attempts: the existing cells between 15 and 85% that every
   composition needs, brought to fifty. Delay 8 alone on spatial task 7 (35) and on
   object tasks 7 and 9 (40 each), 15 actions per chunk on object task 7 (40),
   quarter wear on spatial task 9 (40), four mass doublings on object task 5 (40),
   the 6 degree pan on object task 5 (40).
2. `decoys-d17`, 600 attempts, and its continuation (at most 900): the 17-step delay
   alone, with the dimmer light and with the changed backdrop, on the same attempts,
   twenty per spatial task.
3. `contexts-1`, 100 attempts, and its continuation (at most 400): the 17-step delay
   with four mass doublings, ten per spatial task.
4. `topups-2`, 380 attempts: delay 14 on spatial tasks 4 and 7, delay 17 on object
   tasks 1 and 2, the 4-pixel blur on three spatial and five object tasks.
5. `decoys-blur-d17`, 400 attempts, and its continuation (at most 600): the 3-pixel
   blur with the 17-step delay, alone and with the changed backdrop, on the same
   attempts.

The continuation rule is the first job's: a cell whose existing and new attempts
together number fewer than fifty and whose true or logged rate lies between 15 and 85%
gets the sealed next identities up to fifty; in a stage of decoy arms every arm of
such a task continues together. All identities are canonical healthy attempts that
the context they measure has not used before. The largest possible job is 3,655
attempts and does not fit; about 1,900 are expected through stage 4.

Why these cells. Stage 1 closes the gaps that refuse every interaction carrier, the
drift and a batch on the mixed site. Stages 2 and 3 are for the conflict that remains
once those are closed and that this note states before any of its cells is measured:
the observed forecast, with its retained error of minus 0.7 and plus 6.0 points, must
lie inside 60 to 75%, so the forecast itself must lie between about 64.2 and 65.5%;
a decoy is confirmed flat only where its carriers bear nothing else (alone, or under
the 3-pixel blur on 4 of the 20 seeds), so its site of four robots stays near 97%; the
drift's robot stays near 95% over the observed window and the additive pair's worn
robot at 78%. Those six robots hold 46.7 of the 65.5 points. The other six would have
to average under 38%, below a floor of 0.40 with its reserve of 0.03. A decoy measured
under a site-wide fault lets its site carry that fault. The 17-step delay is the
site-wide fault of the menu whose tasks either fail or succeed (three spatial tasks at
0 of 10, five at 9 of 10 or more), which keeps the same-attempt comparison narrow.
Whether the decoy then passes the two-point margin of the decoy rule is not expected
from at most fifty pairs per task; the job measures the effect and its interval, and
the report states per margin how many seeds compose. No margin, floor or reserve is
changed by this plan.

What the job measured, and what followed from it, is in "Round 6c" at the top of this
file. Six of the eight stages completed; the fifth item, the decoy comparison under the
blur, was stopped by the budget and is not calibration.

### Round 6c, third job: preregistration (2026-10-10, before GPU admission)

After the stop described at the top of this file the orchestrating session decided
three settings for the fourth-generation development worlds (the decoy rule's margin
as two points of fleet success, no retained error at composition, a floor of 0.30)
and asked for one further calibration job of at most 180 GPU minutes for the recipes
that carry a fleet-wide release. Those recipes place the 3-pixel blur on every robot
from the release's day on, and no context under the blur is measured on object
tasks, so nothing can stand on the mixed site once the release has started.

Which contexts to measure was read from the search, before any of them is measured.
A planning run gave every unmeasured context that contains the blur a guessed cell
(fifty attempts at the product of its measured parts) and the comparison of the
backdrop under the blur with the 17-step delay a stand-in copied from the comparison
under the delay alone, with the three settings, 150 drafts per seed, on the twenty
assessment seeds. These are guesses, not measurements and not forecasts; the only
thing taken from them is the list of contexts the search places:

- The fleet recipe as it stands (one fault per mechanism and scope) composes on 0 of
  20 seeds, also when every context of the menu is guessed: with the release holding
  the software version and the scene camera on every robot, the menu has too few
  plain faults left. This is a matter of the recipe, not of calibration, and it is
  reported instead of being measured around.
- With three faults per mechanism and scope, as the D3 recipe has, it composes on 9
  of 20 seeds, always in one layout: the decoy on a spatial site that also carries
  the site-wide 17-step delay, the heavier batch and the additive pair's worn robot
  on the other spatial site with two or three delayed robots beside it, the
  interaction, the drift and one worn robot on the mixed site. All nine use the pair
  and both of its members, the wear knots at 0.5, 0.75 and 1 on object tasks, the
  backdrop comparison under the blur with the delay, and the delay with the batch
  under the blur on spatial tasks; seven use the quarter-wear knots, three the
  17-step delay on object tasks, two the 14-step delay on both suites.

The job: one admission, six workers, batch four, `fp16_res32`, a hard cap of 10,800
seconds that the plan itself names (`refinement_run` reads the cap from the sealed
plan, ninety minutes where a plan names none, never more than three hours).
`calibration.third_plan` wrote the manifests to
`~/.cache/robots-probes/worlds-v4/20261010-task-cells-r6d` before this commit;
`tests/worlds_r6d_plan.json` holds the order, the sizes and the SHA-256 of every
manifest, and a test compares the files on disk with it. Every stage is followed by
its sealed continuation under the rule of the first two jobs (to fifty attempts
where the existing and new attempts of a cell number fewer than fifty and the true
or logged rate lies between 15 and 85%; the arms of a decoy comparison continue
together). All identities are canonical healthy attempts that the context they
measure has not used. The order is the priority:

1. `decoys-blur-d17`, 400 attempts, continuation at most 600: on spatial tasks the
   3-pixel blur with the 17-step delay, alone and with the changed backdrop, on the
   same attempts, twenty per task.
2. `object-pair`, 100, continuation at most 400: on object tasks the blur with the
   8-step delay and 15 actions per chunk.
3. `object-members`, 200, at most 800: the blur with the 8-step delay, and the blur
   with 15 actions per chunk (the interaction's single removals).
4. `object-wear`, 200, at most 800: the blur with full wear and with 0.75 wear.
5. `spatial-d17-batch`, 100, at most 400: on spatial tasks the blur with the 17-step
   delay and four mass doublings.
6. `object-wear-low`, 200, at most 800: the blur with 0.5 and with 0.25 wear.
7. `object-batch`, 200, at most 800: the blur with four mass doublings, and with 0.5
   wear besides (the additive pair on the mixed site).
8. `spatial-wear-low`, 100, at most 400: the blur with 0.25 wear on spatial tasks.
9. `object-d17`, 100, at most 400: the blur with the 17-step delay on object tasks.
10. `spatial-d14` and `object-d14`, 100 each, at most 400 each: the blur with the
    14-step delay.

Sizing. The guessed rates give about 3,450 attempts for the whole list; at the 2.3
to 3.0 seconds per attempt of the second job's stages (everything included) that is
130 to 170 minutes, so the list is expected to fit and may not. A stage is therefore
started only when a bound on its duration fits what is left of the cap (3.2 seconds
per attempt and 150 seconds; `calibration.stage_fits`), and the job ends at the first
stage that does not fit. A stage that was started and that the cap still stops is
not calibration, as before. Items 1 to 5 are what every composed layout of the
planning run needs and are expected to end within the first ninety minutes.

The comparison that the second job did not finish. Item 1 is the stage that the
second job's budget stopped after 300 of its 400 attempts. Its design (arms, doses,
twenty per task, continuation) was sealed in `1cdece0` before any of its outcomes
existed, and it is not changed: the planner, applied to today's record, selects the
same attempts in the same arms with the same fault ids (compared with the sealed
manifests of the second job; only the episode numbers differ). The whole stage is
run again; no outcome of the stopped stage is read, imported or used. One number of
it has been seen, that 165 of its 300 completed attempts succeeded, both arms
together, which says nothing about the difference between the arms. What did change
is its place, from last to first, because the orchestrating session asked for this
comparison and every layout of the planning run needs it; that choice does not
depend on the number.

Left out, for the budget: the 6-degree pan and the other camera faults under the
blur (the release holds the scene camera, so the recipes cannot place them); the
two decoys under the blur on object tasks and the dimmer light under the blur with
the delay (dimming under the blur stays refused); the batch at two and six
doublings, the furniture shift and the detector error under the blur; worn pads
together with the batch beyond half wear, with or without the blur, which the
search asked for most often and which would let a worn robot or the drift stand at
the batch's site; the 20-step delay; and every context of three or more faults
besides the blur. No margin, floor, reserve or band is changed by this plan, and no
dose or cell will be chosen from its outcomes.

## Round 7, first calibration preregistration (2026-10-11, before admission)

One job, six workers, batch four, fp16_res32, at most 10,800 admitted seconds.
The root is `~/.cache/robots-probes/worlds-v4/20261011-task-cells-r7a`;
`tests/worlds_r7a_plan.json` seals every manifest and the execution order.
`calibration.ROUND7_WEAR_STAGES` measures the six missing spatial contexts:
pads at 0.25 wear with four mass doublings, then the same under 3-pixel blur;
0.75 wear with the batch, then with blur; full wear with the batch, then with
blur. Each starts with ten attempts on each of the ten spatial tasks, followed
immediately by its sealed continuation to fifty where the existing-plus-new
true or logged rate is between 15 and 85%. Doses, order, physical identities
and that stop rule are fixed before admission. No context is selected after
its outcome. All subsets already have measured cells. The maximum is 3,000
attempts; at the previous job's 1,476 attempts/hour that is about 122 minutes,
plus stage overhead, so the 180-minute cap may truncate the list. The existing
stage-duration bound stops before a stage that would not fit; only complete
stages with passing chains enter calibration. w48 and w49 remain estimator
checks and contribute no cells. A detached on-disk checkout at this plan's
commit runs the job while the working branch is repaired.

## Round 7, second and third calibration preregistrations (2026-10-11)

Both plans are fixed before admission, in `calibration_round7.py` and the
manifest digests of `tests/worlds_r7b_plan.json` and `worlds_r7c_plan.json`.
Each has one separate heavy admission, six workers, batch four, fp16_res32,
a detached disk checkout and a 180-minute hard cap. Only complete stages
with passing chains enter calibration; the existing duration bound may end
the plan before its next stage. No dose or priority depends on these outcomes.

Job 2 (`20261011-task-cells-r7b`) measures six spatial contexts in this order:
delay 8 plus mass 4, then with blur 3; chunk 15 plus mass 4, then with blur;
the delay/chunk pair plus mass 4, then with blur. Each begins at ten per task
and immediately continues to fifty when the existing-plus-new true or logged
rate is in [15%,85%]. All identities, including continuations, are sealed.
At most 3,000 attempts; roughly two hours at the last job's throughput before
fixed overhead. It supplies both single removals for the pair beside the batch.
It adds no more trials of the near-zero object interaction under blur.

Job 3 (`20261011-task-cells-r7c`) tests another effective site mechanism:
the 4 cm sideways furniture shift (solo spatial calibration 59/100 successes).
First shift plus blur, with and without backdrop; then shift without blur,
with and without backdrop. Those matched arms start at twenty per task and
continue all arms together to fifty if any needs the fixed mid-range rule.
Then shift plus batch, shift plus batch plus blur, and finally shift plus
blur on object tasks, each ten per task with the same continuation to fifty.
The first two comparisons make the decoy and removal contexts available; the
batch contexts allow alternative site layouts. Maximum 3,500 attempts; a
priority list which may exceed three hours with overhead, not a promise that
all fit. The last object stage is lower priority. These plans do not change
the floor, band, margin, estimator, or the stored w48/w49 records.

## Round 7 contract repairs (2026-10-11)

Entry 20 is now expressed by the `planned` assessment template: floor 0.30
plus the unchanged 0.03 reserve, decoy margin 0.02 of scoring fleet success,
and `retained_error="generation-4"`. That named reserve moves both ends of
the forecast's own interval outward by 0.021, pooled over the three windows
of w48/w49. D2 and D3 keep `retained_error="none"`. The test template's search
aim is 66.0–69.5%. As built in this round that aim was a third admission rule
beside the frozen 60–75% band and the widened interval: a draft whose forecast
lay outside it was refused, which entry 20 does not ask for. Since round 8 the
interval inside the band admits alone and the aim only guides the search
(`admission="interval"`, see "Round 8" below). **A recipe's aim is read on the
raw measured-cell point proportion.** The conditional interval uses Beta(successes+1/2, failures+1/2)
samples, with their shrunken means. No point correction is fitted from w48/w49,
and neither world is added to calibration. Two worlds cannot validate 95%
coverage; the retained discrepancy is an engineering reserve.

Decoy confirmation now uses the scoring window's tasks and contexts in both
the grouped proposal check and the final slot-by-slot check. A constructed
decoy with +8 points on one task for six days and -12 on another for four
pooled to zero before this repair; it is now refused because the scoring
loss is four fleet points. Other windows cannot average that loss away.
The optional feasibility-bound diagnostic uses the same scoring mixture;
its regression refuses a variant whose five helpful post-onset days hid
four harmful days when pooled.

`world.context`, `build`, and `generator_record` retain the recipe's named
jobs, error setting and source digest. Rebuilt predictions match the sealed
intervals. `world.recompose_record` reads the two original records without
writing them: it recognizes only their exact known erroneous whole-record
digests and otherwise identical published recipes, then requires both the
old world hash and old forecast seal to reproduce. Unknown or changed pins
remain errors. This is explicit compatibility for the two damaged metadata
records, not a general exception to calibration pinning.

The import checks outcome completeness against identities, the original
stage manifest's bytes against its committed hash and count (including a
regression for changed line endings), selected continuation
identities against that manifest, and passing chains. A failed or unfinished
stage contributes no cells. The CPU `check` command names its cause count as
an either-window forecast; actual causes still require replay.

`search_revision="r7"` avoids onset collisions during drafting, checks floors
on actual overlaps rather than discarding candidates on standalone floors,
and settles full level/onset changes, pattern onsets, then pairs of changes.
Each accepted move uses measured contexts; additive estimates only order
which paired moves to try. Existing seals retain `"r6"` proposal order;
final scoring-window checks apply to both revisions. CPU assessments report
three largest scoring-window forecast fixes per accepted seed and distinct
unordered combinations of kind and scope for the largest two. Assessment
seeds are fixed public diagnostics; no test world is drawn or sealed.

While drafting, each plain-fault placement ranks its supported level/onset
combinations by distance to the raw-point target. It retains the whole level
menu for settling: a level unsupported at one onset can become supported at
another. A synthetic regression selects the supported day-2, 68% candidate
instead of the old day-0, 64% candidate; reverting that draft step fails it.
Thirteen separate guard-removal experiments fail their intended regression
assertions, and removing the interaction onset guard fails the structural
draft test. These synthetic checks prove the guards, not simulator behavior.
The interaction also tries all free second-member onsets, rather than
discarding a carrier when the first free onset fails. Its regression makes
day 1 fail a synthetic window floor and day 2 pass; the earlier search
refused that draft. This leaves the archived `r6` proposal sequence unchanged.

### Before the new calibration

On the existing 891 cells, at `017eb02`, the fixed twenty public assessment
seeds and 400 drafts per seed give planned **4/20**, D2 **4/20**, D3 **16/20**.
Their distinct unordered top-two (kind, scope) combinations number 4, 4 and
14 respectively. The sixteen failed template seeds account for 6,400 drafts:
1,718 stop with two of four plain faults placed, 1,693 with three, and 1,439
with no interaction carrier. The completed drafts include 588 off-target
refusals. A draft can fail several final rules, so refusal totals need not
sum to 6,400. The earlier `af35e15` search has the same acceptance counts;
repairing search alone has not met the bar.

A three-plain-fault variant at `af35e15` accepts 6/20 seeds, with five
top-two combinations, 7–8 forecast causes and at-least-six probabilities
98.24–100%. This is an assessment variant, not yet the selected template.
All these assessments precede the round-7 cell import; their raw rows are
under `~/.cache/robots-probes/worlds-v4/20261011-r7-evidence/` in
`before-calibration`, `before-calibration-plain3`, and
`target-before-calibration`. No actual test seed or world is created.

### Job 1: the wear/batch grid

`20261011-task-cells-r7a`, from frozen `7e262bf`, completed all twelve
preregistered stages: 1,560 attempts, 60 task-context cells, 252 chain
comparisons of which 251 are identical and one has an odd camera frame (stage
`wear-batch-4-refine`, episode `e001977`, the wrist camera at step 1 of the
replay), which the engine's chain check classes as a rendering transient and
accepts: no chain is broken and every stage's chain passes (corrected in round
8; this said "252 identical"), no incidents, 58.4 admitted minutes (driver elapsed,
including chain checks, excluding queue time). No attempts duplicate the
existing record. The refinement table becomes 951 cells / 13,541 attempts.

The following rates average the ten spatial tasks equally, using the new
attempts alone; they are calibration proportions, not world outcomes.

| Wear with four mass doublings | Attempts, no blur | Success, no blur | Attempts, blur 3 | Success, blur 3 |
|---|---:|---:|---:|---:|
| 0.25 | 180 | 86.2% | 140 | 86.4% |
| 0.75 | 300 | 69.4% | 220 | 71.8% |
| 1.00 | 380 | 51.2% | 340 | 49.6% |

This supplies the six missing context grids needed to consider a wear drift
beside the batch. Their contribution to composition is assessed separately;
having those cells does not itself guarantee the floor, six forecast causes,
or a fleet-release effect. The old w48/w49 job selections and seals remain
fixed. Jobs 2 and 3 use their already committed frozen plans.

### Job 2: the interaction/batch grid

`20261011-task-cells-r7b`, from frozen `b6f4388`, completed its twelve
preregistered stages, including three empty continuations: 880 attempts,
60 task-context cells, 180 identical chain comparisons, no incidents,
60.9 admitted minutes. No attempts duplicate the existing record. The
refinement table becomes 1,011 cells / 14,421 attempts.

These rates again average the ten spatial tasks equally, on this job's
attempts alone. Each context includes the batch of four mass doublings.

| Additional mechanism | Attempts, no blur | Success, no blur | Attempts, blur 3 | Success, blur 3 |
|---|---:|---:|---:|---:|
| Delay of 8 steps | 100 | 88.0% | 140 | 86.8% |
| Chunk of 15 actions | 100 | 88.0% | 100 | 88.0% |
| Delay and chunk together | 220 | 57.2% | 220 | 59.4% |

The joint context is above the 30% floor plus its three-point reserve on
the equally weighted suite; individual robot task mixtures, removals and cause
forecasts still have to pass composition. These absolute context cells
do not replace the earlier same-attempt qualification of the interaction.
No further object-task interaction attempts were taken.

With job 1 alone, `7691ad8` accepts planned 11/20, D2 13/20 and D3 16/20
public seeds, with 4/9/13 distinct top-two kind/scope combinations.
The three-plain variant accepts 10/20 (five combinations), so it does not
justify replacing the four-plain template. Trying later interaction onsets
at `0f4620f` still gives planned 11/20 (four combinations), on different
accepted drafts. The full comparison with jobs 2 and 3 remains separate.

The complete A+B assessment at `a132bf5` gives planned **15/20**, D2
**17/20**, D3 **18/20**, with 6/10/14 distinct top-two kind/scope
combinations. Four plain faults remain. The five failed template seeds
account for 2,000 drafts, including 1,002 off-target refusals and 840
incomplete plain-fault placements. The interaction-carrier refusal is absent
from these failed seeds. Final-rule refusals can overlap on one draft.

### Job 3: an alternative site mechanism

`20261011-task-cells-r7c`, also from frozen `b6f4388`, completed all ten
preregistered stages: 2,980 attempts, 70 task-context cells, 344 identical
chain comparisons, no incidents, **110.8 admitted minutes**. No attempts
duplicate the existing record. The complete refinement table is now
**1,081 cells / 17,401 attempts**. All three jobs finished within their
180-minute caps; their admitted times include stage and replay checks,
and exclude time waiting for the GPU lock.

These are this job's equally weighted task rates, not world outcomes.
The spatial rows average ten spatial tasks; the object row averages ten
object tasks. A shift is four centimetres sideways, blur is three pixels,
and the batch has four mass doublings.

| Context | Attempts | Success |
|---|---:|---:|
| Spatial shift | 500 | 59.8% |
| Spatial shift + backdrop | 500 | 60.6% |
| Spatial shift + blur | 500 | 67.0% |
| Spatial shift + blur + backdrop | 500 | 66.0% |
| Spatial shift + batch | 460 | 48.6% |
| Spatial shift + blur + batch | 380 | 56.8% |
| Object shift + blur | 140 | 25.4% |

The backdrop comparisons use 500 matched attempt pairs each. Its change
in success is +0.8 carrier points without blur (95% paired interval
−2.68 to +4.27), and −1.0 with blur (−4.71 to +2.71). Both intervals
include zero and exclude a loss of six carrier points, the two-fleet-point
margin for a full site. This is the equal-task summary using the existing
approximate task-weighted paired interval; composition still checks each
actual scoring-window task/context mix. These are matched **decoy**
comparisons; other removal forecasts use the measured context cells and
the unchanged estimator. No new world or removal oracle was run.

The shift therefore has measured support beside the decoy and batch,
including under the fleet release. Object performance under blur is much
lower than spatial performance, so a mixed site's exact task mixture and
the per-robot/window floor still constrain its placement. The final
assessment below measures the resulting layouts rather than assuming that
new cells must improve acceptance.

### The final assessment, on all three jobs

The round's first builder was cut off after the import
above; a second agent checked and committed that import (`aed22da`) and made this
assessment. It is the same measurement as the earlier ones: the round's `assess.py`
from a frozen checkout of `aed22da`, the table of 1,081 cells, the twenty fixed
public assessment seeds, 400 drafts per seed, the recipes exactly as
`planned_recipe` gives them, three worker processes. CPU only: `compose.assess`
drafts and checks in memory under the world id `w99`. No test seed is drawn and
nothing is written, sealed or run.

| Recipe | Composes | Distinct combinations of the two largest fixes | Observed forecast | Forecast causes | At least six causes, share of resamples |
|---|---:|---:|---:|---:|---:|
| `planned`, the test worlds' template | **15/20** | 7 | 66.8–69.4% | 7–10 | 95.5–100% |
| D2 | **17/20** | 10 | 66.8–71.4% | 7–10 | 95.5–100% |
| D3 | **18/20** | 14 | 65.5–71.5% | 6–8 | 96.4–100% |

The template reaches the bar of 15 of 20 and no more, under the settings of entry
20: a floor of 0.30 with the reserve of 0.03, the decoy rule at two points of fleet
success, the fourth generation's retained error of 2.1 points outward at each end
of the forecast's own interval, the fleet-wide release, four plain faults. The
accepted intervals begin between 61.2 and 63.6% and end between 71.4 and 73.7%.
D2 is the same recipe without the retained error and with the wider target that
follows from that (64.0–71.5% against 66.0–69.5%). It composes on the template's
fifteen seeds and on seeds 5 and 18 as well. What costs the template those two
seeds is its target, not the 2.1 points (corrected in round 8; this said the
retained error costs them): the review of round 7 accepted by the widened interval
alone, with the search unchanged, and seeds 5 and 18 composed too, at forecasts of
69.89 and 70.04% with intervals of 64.26–74.62% and 63.77–74.13%, 17 of 20.

Job 3 did not move the count: the same fifteen seeds composed on the table of jobs
1 and 2. It changed which draft is accepted, or its largest fixes, on six of them
(2, 3, 6, 12, 16, 19). The combinations of the two largest fixes went from 6 to 7,
and a furniture shift at a site, which was among the three largest fixes of no
accepted draft before, is now the largest fix on seeds 3, 6 and 16.

**Not one layout, but few.** A combination is the unordered pair of (kind, scope)
of the two fixes with the largest forecast gain over the scoring window, so the
same two mechanisms at other sites count as another combination. The template's
fifteen:

| The two largest fixes | Seeds |
|---|---:|
| a delay at site s1 and the heavier batch at site s3, as in `w48` and `w49` | 6 |
| a delay at site s3 and the batch at site s1 | 4 |
| a delay at site s2 and the batch at site s3 | 1 |
| a furniture shift at site s3 and a delay on robot r04 | 1 |
| a furniture shift at site s3 and a delay on robot r05 | 1 |
| a furniture shift at site s1 and the batch at site s3 | 1 |
| a delay at site s3 and a delay on robot r01 | 1 |

Eleven of the fifteen have a site-wide delay and a site-wide batch on top. With the
sites and robots left out there are four combinations of mechanisms. The largest
single fix is a site-wide delay on ten seeds, a furniture shift on three and the
batch on two.

Every composing seed of the template follows. Seeds and drafts are counted from
zero. A fix is the removal of one fault, with its forecast gain in points of fleet
success over the scoring window; these are forecasts from the cells, not gains
measured by replay.

| Seed | Accepted draft | Largest fix | Second | Third |
|---:|---:|---|---|---|
| 1 | 6 | `action_latency@s1` (16.54) | `object_batch_variant@s3` (5.27) | `gripper_wear@r09` (4.35) |
| 2 | 55 | `action_latency@s1` (14.11) | `object_batch_variant@s3` (5.00) | `action_latency@r07` (3.18) |
| 3 | 6 | `furniture_shift@s3` (11.55) | `action_latency@r04` (4.40) | `software_release@fleet` (4.30) |
| 4 | 191 | `action_latency@s3` (14.32) | `object_batch_variant@s1` (5.41) | `action_latency@r02` (4.40) |
| 6 | 22 | `furniture_shift@s3` (9.86) | `action_latency@r05` (4.38) | `action_latency@r08` (4.02) |
| 8 | 11 | `action_latency@s3` (14.03) | `object_batch_variant@s1` (6.36) | `software_release@fleet` (4.28) |
| 10 | 6 | `action_latency@s3` (16.97) | `action_latency@r01` (4.49) | `action_latency@r02` (4.35) |
| 11 | 2 | `action_latency@s1` (16.74) | `object_batch_variant@s3` (5.51) | `gripper_wear@r12` (3.85) |
| 12 | 138 | `object_batch_variant@s3` (8.49) | `action_latency@s2` (5.55) | `action_latency@r01` (3.98) |
| 13 | 131 | `action_latency@s3` (11.08) | `object_batch_variant@s1` (5.18) | `action_latency@r07` (3.99) |
| 14 | 17 | `action_latency@s1` (12.19) | `object_batch_variant@s3` (6.66) | `software_release@fleet` (3.41) |
| 15 | 196 | `object_batch_variant@s3` (10.16) | `action_latency@s1` (7.71) | `gripper_wear@r11` (3.70) |
| 16 | 32 | `furniture_shift@s1` (8.08) | `object_batch_variant@s3` (8.04) | `software_release@fleet` (3.80) |
| 17 | 334 | `action_latency@s1` (14.75) | `object_batch_variant@s3` (7.47) | `action_latency@r05` (4.14) |
| 19 | 35 | `action_latency@s3` (12.21) | `object_batch_variant@s1` (8.80) | `action_latency@r05` (5.33) |

**The five seeds that do not compose** (0, 5, 7, 9 and 18) are the five of the
earlier table. Their 2,000 drafts end as follows. The first three rows are stages
and add up to 2,000; the last three rows count rules that one of the 184 drafts
breaks, and a draft can break several.

| Where a draft ends | Drafts |
|---|---:|
| fewer than four plain faults fit measured contexts above the floor (three fit 450 times, two 269, one 84, none 26) | 829 |
| complete, with the observed forecast outside the target of 66.0 to 69.5% | 987 |
| complete and on target, then refused by a rule on the causes | 184 |
| of those 184: the fleet-wide release has no positive measured removal interval on the fleet's mix of tasks | 182 |
| of those 184: six causes in fewer than 95% of the resamples | 74 |
| of those 184: the confounder's fault (35) or the interaction (30) not detectable | 65 |

On four of the five seeds drafts do reach the target, 26 to 60 of 400 each, and
none on seed 9. Nearly all of them fall at one rule: over the scoring window,
removing the release has to gain fleet success on that seed's mix of tasks, with
the whole 95% interval of the calibration's resamples above zero, and on these
seeds it does not. D2's three refused seeds (0, 7 and 9) show the same,
170 of their 172 drafts on target. D3 has no fleet-wide release; its two refused
seeds (13 and 17) run out of plain faults (533 of 800 drafts) or miss the target
(260). What would lift the five was not measured: the bar is met, and no setting
was varied in this round after the calibration.

The rows of all three recipes, seed by seed with every refusal count and a receipt
naming the commit, are in
`~/.cache/robots-probes/worlds-v4/20261011-r7-evidence/after-job-c-assessment/`.

**What these numbers are not.** Twenty public seeds show whether the recipe
composes; they are not the rate of acceptance on fresh seeds, and 15 of 20 leaves
no margin. The share of resamples with six causes is the forecast's sensitivity to
the calibration's sampling, not the probability of six causes measured by replay.
No world of the template exists, so its interval with the 2.1 points is untested on
a world composed from this table.

### Round 7: verified at the end, and not

- **The last import.** Before `aed22da` the record was rebuilt from the six
  calibration roots and the two folded worlds (`w46`, `w47`) and compared with the
  file: identical, and identical to the first builder's collected snapshot. Nothing
  but `task_refinement` differs from the commit before it, every older cell keeps
  its sources, and the 2,980 attempts of job 3 are unique within the job and
  against every other source. No stage of jobs 1 and 3 is skipped; the three
  skipped stages of job 2 are continuations for which the committed rule selected
  no attempt. Every stage of the three jobs that ran has exit 0 and a passing
  chain. No source is `w48` or `w49`. The tables of the three job sections above
  were recomputed from the record and agree.
- **The guards, one removed at a time.** In a copy of the code, 25 single removals
  or reversals of what points 1 to 6 of the round built were each run against the
  test meant to catch it; every test passes on the unchanged copy first. 24 make
  their test fail. The one that does not: removing the choice of a free day for a
  plain fault before its levels are ranked (`_draw_plain`). The ranking that
  follows skips days that are not free, so recipes with a floor, which is all of
  the fourth generation, lose nothing; a recipe of revision `r7` without a floor
  would rely on that first choice alone, and no test covers it.
- **Tests.** The 100 focused world tests (round 7, its plans, measured, generation
  4, search, settings) pass on the import, among them the rebuild of the record
  from its sources and the recomposition of `w48` and `w49` from their stored
  records without a change to a seal.
- **Not done, by assignment.** No GPU job beyond the three calibration jobs, no
  world composed for writing, sealed or run, no test world, no fold of `w48` or
  `w49` into the cells. Not measured: what would make the five refused seeds
  compose, the template with three plain faults on the final table (10 of 20
  against 11 of 20 with four on job 1's table), more than 400 drafts per seed, and
  whether a world composed from this table lands inside its interval.

## Round 8: the repairs before the test worlds are drawn (2026-10-11)

This round answers the review of round 7 (2026-10-11: merge after named
repairs; the test worlds not yet from the template as it stood). Two of its
findings needed a decision, and the orchestrating session took both. The round
builds them, produces the test worlds' recipe by code, closes the gaps the
review found in the tests and in the import, and counts again. No world was
composed for writing, drawn, sealed or run, and no private recipe file exists.
`w48` and `w49` are untouched: they keep the rules they were composed under
and recompose to their hashes and seals.

### Decision 1: the interval admits, and nothing else

Entry 20 of the design note accepts a test world when its forecast's own 95%
interval, each end moved outward by 2.1 points, lies inside 60 to 75%. The
template of round 7 also refused every draft whose forecast lay outside its
search target of 66.0 to 69.5%, a rule entry 20 does not name. A recipe now
states its admission (`WorldRecipe.admission`):

- `"interval"`: the interval inside the band, with the recipe's other rules.
  The target (`aim`) still tells the search where to steer and when to stop
  settling; it refuses no draft. `planned_recipe` gives this to the template
  and to the two development recipes D2 and D3 (which retain no error, so
  theirs is the forecast's own interval, as entry 20 says for them).
- `"target"`: the forecast inside the target and the interval inside the
  band, as every world sealed so far was composed. It is the field's default,
  left out of a stored record, and `w48` and `w49` state it.

`Composer.check` asks the interval before the rules on causes, because it is
cheap; which drafts are accepted does not depend on that order.

### Decision 2: onset days as round 6 drew them

The search of round 7 ranked a plain fault's day by the forecast together with
its level, and moved the interaction's members while settling. Faults ended up
starting early: of the plain faults on one robot in its fifteen composing
drafts, 48 of 60 started on day 0, and both members of the interaction started
together on day 0 in 8 of 15. `w48` and `w49`, composed by the search of round
6, have 2 of 8 on day 0 and the members on days 0 and 1. Nobody decided that
faults should start earlier, and a method that looks for a change over time has
less to see when they do. Search revision `"r8"` is `"r7"` with the days of
those two groups handled as round 6 handled them:

- **The interaction's members** start on days 0 and 1, always. Round 7 tried
  days 2 and 3 for the second member when day 1 was not supported, and moved
  both members between days 0 and 3 while settling.
- **A plain fault's day** is drawn uniformly from its range (days 0 to 4 on a
  robot, 1 to 5 at a site, 1 to 3 for the confounder's fault). While settling
  it moves one day earlier or later at a time, and only while that brings the
  forecast nearer the target; the confounder's day does not move. Round 7
  took, at the draft and at every settling step, whichever day of the range
  brought the forecast nearest.

What round 7 repaired is kept: a plain fault's **level** is the supported one
nearest the target at its day, and settling may move it to any level; **no two
faults start on the same day at overlapping scopes**, already in the draft;
an interaction carrier and a plain fault are judged on their **actual
overlap**, so that one which only another fault lifts above the floor is
kept; settling still tries two moves together, and still moves the days of the
release, the decoy and the batch.

Three things of round 7 could not be combined with the days as drawn, and the
days were kept:

1. *A level and a day chosen together.* Round 7's ranking found the one day on
   which a level was supported. With the day drawn first, a placement that is
   unsupported on that day is a miss, and the draft draws another fault.
2. *A later day for the interaction's second member.* Avoiding a same-day
   onset now means passing over a carrier on which another fault starts on
   day 1, instead of moving the member to day 2 or 3.
3. *A day reached in one move.* A fault that is drawn after the day on which
   the fleet-wide release starts cannot step past it: the release covers every
   robot, so that day is taken. Round 6 had the same limit.

### The test worlds' recipe is produced by code

`recipe.private_recipe(world_id, public_fleet_id, name=...)` returns what a
test world's private file states: the template with the six calibration jobs it
was assessed on named (`TEST_WORLD_JOBS`: the three of round 6 and the three of
round 7), the digest of the record as of those jobs pinned
(`TEST_WORLD_SOURCES`), every one of the recipe's 56 fields stated, and no
seed. The three arguments are all that differs between T1 and T5. The function
writes nothing and draws nothing; creation (`generate --recipe`) draws the seed
as before.

Three things were open here. *The cells:* the template names no jobs, so a
recipe copied from it stood on "the record as it is" and would have moved with
the next import; with the jobs named a later job leaves it alone, and with the
digest pinned any other change of the record under it is refused at
composition. *The defaults:* the settings added since the third generation have
the older rule as their default, so that old records keep their bytes, and a
file that left one out composed without complaint under the older rule (the
review composed three other worlds that way, by leaving out the retained error,
the search revision or the target). `external_recipe` now refuses a file that
does not state every field, and one that states no calibration jobs. *The pin:*
`tests/test_worlds_round8.py` holds every field of the produced recipe and of
the template as literal values, so a changed setting fails a test before a
world is drawn from it.

Not in the recipe: the unseen fault types. The recipe has no field for a fault
kind or for how many unseen faults a world carries, and the template was
assessed without them.

### The import compares its sources

`calibration.refinement_record` passed two damaged sources in the review.

- *A folded world with one attempt removed from its identities, verdicts and
  episodes alike* imported 1,679 of 1,680 attempts, because the run's tables
  were only compared with one another. The run is now compared with the
  world's own manifest (`files/manifest.jsonl`), and the world is refused.
- *A continuation one attempt short of the stop rule* (the selected list and
  its run holding 39 of the 40 attempts the rule selects) imported 139 of 140,
  because the list was only checked to contain sealed identities. The rule is
  now recomputed from the parent stage's outcomes and from the cells counted
  before the job (the task calibration, the folded worlds and the earlier
  jobs, in the order given), and a selected list that differs is refused. A
  continuation whose parent cannot be read is skipped with it.

The record rebuilds identically from its six calibration roots and two folded
worlds under both checks (1,081 cells, 17,401 attempts): every continuation of
all six jobs is what the rule selects, and both folded worlds equal their
manifests.

### Tests, and the breaks they catch

`tests/test_worlds_round8.py` holds 22 tests for this round. Those for the
two decisions, the recipe and the two damaged sources fail on the code as
round 7 left it. Those for the breaks below pass there, as they must: the
code was right and only the test was missing, so each is shown by making its
break. The review had made 32 single-place breaks in round 7's code, and six
passed all 100 focused tests. Each of the six now fails a test written for
it:

| The break | The test that fails |
|---|---|
| the template's target set back to 64.0–71.5% | every field of the template and of the produced recipe is pinned |
| `world.recompose_record` no longer compares the stored forecast seal | a record with a changed seal is refused |
| the import no longer compares a continuation's selected identities with the sealed manifest | a selected list with a foreign identity is refused |
| the interaction's carrier filtered on its standalone floor again | carriers that fail alone are still offered, in `r7` and `r8`, and not in `r6` |
| settling accepts a move onto a day on which an overlapping fault starts | the move is refused, in `r7` and `r8` |
| settling accepts a move without the check of floor, cells and decoys | the move is refused, in `r7` and `r8` |

In a scratch copy of the code, those six and eighteen more breaks of what
this round built were each made alone and the 22 tests run (they pass on the
unchanged copy): all 24 make a test fail. The eighteen: admission by the
target again, in `check` or in the template; the interval tested at one end
only; `w48` without its admission by target; the interaction on later days, or
on a taken day 1; a plain fault's day ranked again, a taken drawn day not
replaced, days moved freely or the confounder's day moved while settling, the
template on revision `r7`; a recipe file with a field or its jobs left out;
the produced recipe with five jobs, without the pinned digest, or with a floor
of 0.25; a folded world not compared with its manifest; a continuation not
compared with the stop rule. A first run of these breaks reported all of them
caught within a third of a second each: the scratch copy lacked the project
file and no test had started. The counts here are from the second run, in
which every break is caught by a named failing test.

### The assessment with both decisions in

The same measurement as round 7's: the recipes exactly as `planned_recipe`
gives them (revision `r8`, admission by the interval), the table of 1,081
cells, 400 drafts per seed, in memory under the world id `w99`, from a frozen
copy of commit `15aa55c`. Two sets of twenty public seeds: the assessment
seeds of round 6 (`compose.ASSESSMENT_SEEDS`), and the twenty other seeds the
review of round 7 chose (`(1 << 62) + 7700000 + k`). Neither set holds a
world's seed. CPU only; nothing is written, sealed or run.

| Recipe | Seeds | Composes | Round 7 | Distinct combinations of the two largest fixes | Observed forecast | Forecast causes |
|---|---|---:|---:|---:|---:|---:|
| `planned`, the test worlds' template | assessment | **12/20** | 15/20 | 5 | 67.4–70.7% | 8–10 |
| `planned` | the review's other twenty | **12/20** | 13/20 | 7 | 68.8–71.0% | 8–10 |
| D2 | assessment | **15/20** | 17/20 | 6 | 68.8–73.2% | 6–9 |
| D3 | assessment | **18/20** | 18/20 | 15 | 67.2–72.8% | 6–9 |

**The template is below the bar of 15 of 20 on both sets: 24 of 40.** No
setting was lowered and no fault dropped to change that. It composes on
assessment seeds 1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 14 and 19, and not on 0, 5,
7, 9, 15, 16, 17 and 18. Round 7's template composed on three of those eight
(15, 16 and 17); seeds 0, 5, 7, 9 and 18 it refused as well. D2 is the
template without the 2.1 points; it composes on the template's twelve and on
seeds 5, 16 and 18, so under these two decisions the 2.1 points cost three
seeds of twenty. D3, which has no fleet-wide release, is unchanged at 18,
with the same two seeds refused (13 and 17).

The accepted intervals of the template begin between 61.6 and 64.9% and end
between 71.8 and 75.0%; the lowest robot window in any accepted draft is
33.25%, against the floor of 30% and its reserve of three points. Eleven of
the 24 accepted drafts have a forecast above the target's upper end of 69.5%
(three on the assessment seeds, eight on the others), up to 71.0%: decision 1
admits them, and their intervals end at 74.1 to 75.0%, the last one exactly
on the band's edge, which is inside. The share of resamples with at least six
causes is 97.7 to 100%.

**What refuses.** The eight refused seeds of each set, 3,200 drafts each. The
first five rows are where a draft ends and add up to 3,200; the rows below
them count rules that an admitted draft breaks, and a draft can break several.

| Where a draft ends | Assessment seeds | The other seeds |
|---|---:|---:|
| fewer than four plain faults fit measured contexts above the floor | 1,249 | 954 |
| no carrier takes the interaction on days 0 and 1 | 223 | 889 |
| no confounder placed | 5 | 1 |
| complete, the interval outside the band | 1,485 | 1,246 |
| complete and admitted by the interval, then refused by a rule on the causes | 238 | 110 |
| of those: the fleet-wide release has no positive measured removal interval | 233 | 108 |
| of those: six causes in fewer than 95% of the resamples | 78 | 63 |
| of those: the confounder's fault (45, 30) or the interaction (14, 3) not detectable | 59 | 33 |

Where the interval was recorded, every refusal is at its upper end. Drafts
400 to 799 of the eight refused assessment seeds were run with it recorded:
1,262 are refused for it, all 1,262 end above 75%, none begins below 60%. The
fleet of a draft that fails is too healthy, never too broken: the search does
not find enough loss in the observed window. The 223 drafts without a carrier
are all on seed 16; in the other set they are on four seeds, each of which
loses 210 to 234 of its 400 drafts there.

**Where the days fall.** A plain fault on one robot includes the confounder's
fault where that sits on a robot; "complete drafts" are all drafts of the
template that place every fault, after settling, on the twenty assessment seeds
with twelve drafts each (the method of the review's `onsets.py`), whether or
not they are admitted. Days are counted from 0.

| | Plain faults on one robot: day 0 / 1 / 2 / 3 / 4 | On day 0 | The site-wide plain fault: day 1 / 2 / 3 / 4 / 5 | The interaction's members |
|---|---|---:|---|---|
| template `r8`, composing drafts, assessment seeds (12) | 24 / 5 / 4 / 11 / 4 | 50% | 12 / 0 / 0 / 0 / 0 | days 0 and 1 in 12 of 12 |
| template `r8`, composing drafts, other seeds (12) | 33 / 8 / 5 / 3 / 0 | 67% | 9 / 2 / 0 / 0 / 0 | days 0 and 1 in 12 of 12 |
| `w48` | 2 / 1 / 0 / 1 / 0 | 2 of 4 | day 1 | days 0 and 1 |
| `w49` | 0 / 0 / 2 / 0 / 2 | 0 of 4 | day 1 | days 0 and 1 |
| round-6 search (`r6`), complete drafts (154) | 350 / 43 / 77 / 142 / 47 | 53% | 36 / 30 / 28 / 8 / 9 | days 0 and 1 in 154 of 154 |
| `r8`, complete drafts (133) | 361 / 39 / 86 / 76 / 13 | 63% | 53 / 19 / 15 / 2 / 1 | days 0 and 1 in 133 of 133 |
| round-7 search (`r7`), complete drafts (149) | 525 / 87 / 23 / 7 / 0 | 82% | 94 / 9 / 0 / 0 / 0 | both on day 0 in 129, days 0 and 1 in 15, later in 5 |
| round 7's template, composing drafts (15; the review's count) | 48 of 60 on day 0 | 80% | day 1 in 15 of 15 | both on day 0 in 8 of 15 |
| round-6 search (`r6`), composing drafts, assessment seeds (9; the review of round 8's count) | 19 of 36 on day 0 | 53% | day 1 in 14 of 15, both sets together | days 0 and 1 in every draft |
| round-6 search (`r6`), composing drafts, other seeds (6; the same review) | 16 of 24 on day 0 | 67% | | days 0 and 1 in every draft |

The interaction's members are where round 6 and both development worlds have
them, in every draft. Over complete drafts the plain faults start earlier than
under round 6's search, although the rule for their days is round 6's: 63% of
those on a robot start on day 0, against 53% under `r6` and 82% under `r7`,
and the site-wide fault starts on day 1 in 59%, against 32% and 91%. The draw
is the same; what differs is settling. (Corrected in round 9. The review of
round 8 recorded the generator's own day draws for both searches on all forty
seeds. This paragraph had said that `r8` still moves levels while settling,
and that composing drafts favour early days; both were wrong.)

- **Settling in `r8` moves no level:** 0 of 936 plain faults, against 260 of
  1,248 under `r6`. The level is ranked at the draft, so every settling step
  of `r8` is a day; where round 6 raised a level, `r8` moves a day. 60% of
  `r8`'s single-robot faults end earlier than drafted, by up to four days
  (`r6`: 54%), and 3 end later (`r6`: 100).
- **Only `r8` moves the days of the patterns:** the release's in 154 of 234
  complete drafts, the decoy's in 165 and the batch's in 38 (`r6`: none of
  312), and it tries two moves together.
- **A taken drawn day is replaced by a drawn free one in both searches** (210
  of 849 single-robot faults in `r8`, at the draft; 283 of 1,133 in `r6`, when
  settling starts). Day 0 is always free, which lifts its share from the
  19 to 21% that are drawn to 27% before any settling.

**The drafts that compose do not favour early days.** They have 50% and 67% of
their single-robot faults on day 0 (the two sets of seeds), against 63% and 65%
in all complete drafts; and among the review's complete drafts those whose
interval lies inside the band have 54% and 56%, against 65% and 66% in the
rest. Against round 6's own search nothing is shifted where it matters: with
admission by the interval `r6` composes on 9 and 6 of the twenty seeds, and its
composing drafts have 19 of 36 and 16 of 24 single-robot faults on day 0 (53%,
67%; `r8`: 50%, 67%) and the site-wide plain fault on day 1 in 14 of 15 (`r8`:
21 of 23). Against the two development worlds themselves a test world will
start more single-robot faults on day 0, a half to two thirds against a
quarter (2 of 8), as round 6's own composing drafts would have.

**What is selected instead** (the same review, 234 complete drafts: drafts 0 to
11 of all forty seeds). 118 of them have a site-wide delay or furniture shift
at the decoy's site, 116 have none. 39 have their interval inside the band: 38
of the 118, and one of the 116. In those 39 the largest fix has a median
forecast gain of 12.7 and 12.3 points (the two sets) and is that site-wide
fault in 36; in the 195 others the median is 7.5 and 7.3 points. So a draft
composes almost only when it carries a site-wide delay or shift beside the
decoy, and that fault is then its largest fix by a wide margin. This is why
the largest fix can be named from the build (the section "One structure under
every composed world" below).

**The two largest fixes.** A combination is the unordered pair of (kind, scope)
of the two fixes with the largest forecast gain over the scoring window. The
template's twelve on the assessment seeds give five (round 7: seven on fifteen
seeds), its twelve on the other seeds seven:

| The two largest fixes | Assessment seeds | Other seeds |
|---|---:|---:|
| a delay at site s1 and the heavier batch at site s3, as in `w48` and `w49` | 7 | 3 |
| a delay at site s3 and the batch at site s1 | 2 | 3 |
| a furniture shift at site s3 and the batch at site s1 | 1 | 0 |
| a delay at site s3 and a delay on one robot (r01, r05) | 2 | 0 |
| a delay at site s2 and the fleet-wide release | 0 | 2 |
| a delay at site s1 or s3 and the fleet-wide release | 0 | 2 |
| a delay at site s2 and the batch at site s3 | 0 | 1 |
| the batch at site s1 and worn pads on robot r04 | 0 | 1 |

With the sites and robots left out there are three combinations of mechanisms
on each set. The largest single fix is a site-wide delay in 22 of the 24, a
furniture shift in one and the batch in one. The template's composing seeds of
the assessment set, seeds and drafts counted from zero, gains in points of
fleet success over the scoring window as forecast from the cells:

| Seed | Accepted draft | Observed forecast | Interval | Largest fix | Second | Third |
|---:|---:|---:|---:|---|---|---|
| 1 | 35 | 69.1% | 63.2–73.3% | `action_latency@s1` (16.54) | `object_batch_variant@s3` (4.77) | `gripper_wear@r09` (4.35) |
| 2 | 81 | 67.8% | 61.8–71.9% | `action_latency@s1` (14.11) | `object_batch_variant@s3` (5.34) | `action_latency@r12` (3.04) |
| 3 | 389 | 70.7% | 64.7–74.9% | `furniture_shift@s3` (11.55) | `object_batch_variant@s1` (4.68) | `software_release@fleet` (3.79) |
| 4 | 207 | 69.2% | 63.2–73.3% | `action_latency@s3` (14.32) | `object_batch_variant@s1` (6.74) | `action_latency@r07` (4.79) |
| 6 | 46 | 70.6% | 64.6–74.7% | `action_latency@s3` (11.76) | `action_latency@r05` (4.38) | `action_latency@r08` (4.02) |
| 8 | 52 | 70.4% | 64.3–74.5% | `action_latency@s1` (13.86) | `object_batch_variant@s3` (6.70) | `gripper_wear@r11` (3.68) |
| 10 | 6 | 68.9% | 63.2–73.2% | `action_latency@s3` (16.97) | `action_latency@r01` (4.49) | `action_latency@r02` (4.35) |
| 11 | 19 | 67.7% | 62.0–72.1% | `action_latency@s1` (16.74) | `object_batch_variant@s3` (6.42) | `action_latency@r11` (4.46) |
| 12 | 228 | 69.3% | 63.7–73.8% | `action_latency@s3` (14.38) | `object_batch_variant@s1` (11.69) | `action_latency@r08` (4.10) |
| 13 | 22 | 67.4% | 61.6–71.8% | `action_latency@s1` (14.30) | `object_batch_variant@s3` (5.67) | `action_latency@r08` (3.33) |
| 14 | 17 | 69.3% | 63.6–73.5% | `action_latency@s1` (12.19) | `object_batch_variant@s3` (5.21) | `action_latency@r08` (5.00) |
| 19 | 45 | 68.4% | 62.8–72.8% | `action_latency@s1` (14.99) | `object_batch_variant@s3` (9.77) | `action_latency@r06` (4.54) |

### One structure under every composed world

The review of round 7 found that the composing drafts, for all their different
seeds, are built alike, and that a reader who knew the pattern could name the
largest fixes without analysing anything. It holds as much after this round,
and a report on the test worlds has to say it. In plain terms:

Two of the three sites run only the `libero_spatial` tasks, and the third runs
both suites. In all 24 composing drafts of the template (twelve on each set of
seeds) the heavier object batch sits at one of the two spatial-only sites and
the decoy, always a change of backdrop, at the other. 23 of the 24 carry
exactly one site-wide plain fault, a delay in 22 and a furniture shift in
one; in 20 it sits at the decoy's site, in 3 at the mixed site. That site-wide fault is the largest fix in 23 of the 24, and it and the
batch are the two largest fixes in 17. Everything else is the same in every
draft: thirteen faults; the fleet-wide release carrying a 3-pixel blur from
day 2, 3, 4 or 5; the interaction of an 8-step delay with chunks of 15 actions
on one robot, from days 0 and 1; a drift of worn gripper pads from day 0; and
four or five plain faults on single robots, which are full pad wear (37 of 97),
a 17-step delay (33), a 14-step delay (26) or wear of 0.75 (1). `w48` and
`w49` have the same build: a decoy and a site-wide delay at site s1, the batch
at site s3.

So a rule that reads no outcome at all ("revert the site-wide change at the
site whose backdrop changed, then the batch at the other spatial-only site")
names the largest fix in 20 of the 24 drafts and both of the two largest in 16
(in round 7's fifteen: 13 and 11).

Most of this follows from what the calibration record holds, not from the
seed. A decoy is accepted on its robots only where its effect was measured on
the same attempts with and without it, under every combination of faults that
reaches them (entry 20). The record has seven such comparisons, all on
`libero_spatial` tasks: the backdrop under the blur, under a 17-step delay,
under both, under a furniture shift, and under the shift with the blur; the
dimmer light under the blur and under the delay, but not under both. So under
a fleet-wide blur the decoy can only be the backdrop, only at a spatial-only
site, and the only faults that may reach its robots are a 17-step delay or a
furniture shift. Not the batch, which therefore goes to the other spatial-only
site, and no worn pads: of the 38 plain faults of worn pads on single robots,
none is at the decoy's site (33 at the batch's site, 5 at the mixed site).
Why the site-wide fault is then so often the largest fix was not measured in
round 8; the review of that round measured it ("What is selected instead",
above): a draft's interval lies inside the band almost only when it carries
that fault at the decoy's site. Whatever the share of each cause, a method could name
the largest fix on such a world from its build alone, without diagnosing
anything, and results on worlds drawn from this template have to be read with
that in mind.

### What would lift the template, measured

The template is three seeds short of the bar on the assessment seeds. The
refusals above say where: the fleet of a refused draft is too healthy. Every
change of the search that was measured to lift the count does it by starting
faults earlier, which is what decision 2 took away; the one exception is a
larger budget of drafts. Each row below is the template on
the twenty assessment seeds with admission by the interval, changed in one
respect. The three partial variants ran from scratch copies of the code and
are not in the repository; `r7` is the revision as it stands in the code.

| The template, with | Composes | Against the 12 | Plain faults on one robot on day 0, composing drafts | Interaction's members |
|---|---:|---|---:|---|
| nothing changed (`r8`, 400 drafts) | 12/20 | | 24 of 48 (50%) | days 0 and 1 in 12 of 12 |
| 800 drafts per seed | 14/20 | seeds 16 (draft 612) and 17 (draft 557) more | not counted | days 0 and 1 |
| the interaction's days as in `r7`, the rest `r8` | 13/20 | 16 and 18 more, 4 fewer | 28 of 52 (54%) | both on day 0 in 11 of 13 |
| a plain fault's day ranked by the forecast at the draft, as in `r7`; settling as `r8` | 15/20 (14/20 on the other seeds) | 5, 7 and 16 more | 45 of 61 (74%; 51 of 57 on the other seeds) | days 0 and 1 in 15 of 15 |
| a plain fault's day moved freely while settling, as in `r7`; the draft as `r8` | 14/20 | 16 and 18 more | 33 of 57 (58%) | days 0 and 1 in 14 of 14 |
| all three: revision `r7` | 17/20 | 5, 15, 16, 17 and 18 more | 56 of 69 (81%) | both on day 0 in 10 of 17 |
| no 2.1 points (D2, not open to the template: entry 20 sets them) | 15/20 | 5, 16 and 18 more | 29 of 61 (48%) | days 0 and 1 in 15 of 15 |

Read plainly:

- **Decision 2 costs five seeds of twenty.** With admission by the interval,
  round 7's search composes on 17 (the review's count, found again on this
  code), and `r8` on 12. Seeds 0, 7 and 9 compose under neither.
- **No part of it comes back free.** Each of the three parts of round 7's
  handling of days lifts the count by one to three seeds when it alone is
  restored, and each brings back some of what was decided against: with the
  interaction's days restored both members start on day 0 in 11 of 13
  composing drafts; with the day ranked at the draft 74% of the single-robot
  faults start on day 0; with free moves while settling 58%, against 50%.
  Only the ranking at the draft reaches 15 by itself, and on the other
  twenty seeds it gives 14, with 51 of 57 single-robot faults on day 0.
- **A larger search budget gives two seeds** and changes nothing about where
  faults start. Entry 20 states its counts at 400 drafts per seed; 800 would
  be a change to that, and 14 is still below the bar.
- **The 2.1 points cost three seeds**, the difference between the template and
  D2. They are entry 20's and were not touched.

Not measured: the other variants on the other twenty seeds; two parts
restored together; more than 800 drafts; and what further calibration would
do. Calibration is the one lever that asks no rule, no decision and no
condition of entry 20 to give way. The record's seven decoy comparisons and
the measured contexts beside the fleet-wide release decide how many builds
exist at all (see the section above); whether more of them would give a draft
enough loss without early starts is a guess until it is counted.

### Round 8: verified at the end, and not

- **`make check`** exits 0 on the round's code: lint, format, types, and 2,659
  tests passed, 2 skipped (neither a world test), 65 deselected, in 29
  minutes. Among them the 22 tests of this round, the rebuild of the record
  from its six calibration roots and two folded worlds under the two new
  checks of the import, and the recomposition of `w48` and `w49`, from their
  recipes and from their stored records, to their hashes and forecast seals
  without a change to a stored file.
- **The breaks, one at a time:** 24 of 24 caught, as described above.
- **The counts** are from a frozen copy of commit `15aa55c`, whose search code
  is byte for byte the final one. The rows, seed by seed with every refusal
  count, the scripts, the variants as patches and the logs are in
  `~/.cache/robots-probes/worlds-v4/20261011-r8-evidence/` (`out/A` and
  `out/C` for the two sets of seeds; `out/r7`, `out/vi`, `out/vii`,
  `out/viii`, `out/viiC` and `out/A800` for the variants).
- **Not done, by assignment:** no GPU job, no world composed for writing,
  drawn, sealed or run, no private recipe file written, no fold of `w48` or
  `w49` into the cells, no merge. `make smoke-world` was not run.
- **Not measured:** the template with its interval on a world that ran (none
  exists); D2 and D3 on the other twenty seeds; the search with a plain
  fault's day never moved after the draw, a stricter reading of decision 2
  whose count was started and stopped for time; the unseen fault types, which
  the recipe cannot carry and the 2.1 points were measured without.
- **Not checked:** that the stop rule's recomputation would also hold for a
  job that ran on an older table than the order of its root says. For the six
  jobs imported it holds; the import takes "everything before the root" as
  the table the job saw.

## Round 9: a draw leaves a record, and the search is pinned (2026-10-11)

Round 8 is merged. Its review found nothing wrong in the recipe, in admission
or in recomposition, and two things open before a draw: creation kept no
record of a seed that does not compose and reported it like a failure of the
tool, and the template is below entry 20's bar (a human's decision, not this
round's). The orchestrating session decided three things, and this round builds
them: a seed that does not compose is replaced and recorded; the recipe file is
the code's recipe; the search is frozen. No test world was drawn. The worlds
created here are scratch worlds under `/var/tmp` with the made-up id `w97`.

### Drawing a test world

The commands in the order an operator uses them, in their final form (round
9c). `FILE`, `ROOT` and `LEDGER` lie on the truth side, outside every checkout;
the ledger lies outside the world's root as well. All commands are
`uv run --locked python -m fleetlab.worlds …`, given from one frozen checkout
of the commit that is named on the tracker, with a clean tree: a draw from a
tree that is not clean is refused (class `code`), a new one and a repeated one.

**What the operator decides: nothing but when to give the commands.** The seed
of a test or reserve world is not drawn and cannot be handed in. It follows
from a public random string, the beacon, that the ledger's first line fixes:

    seed = 2^62 + (SHA-256("BEACON|RECIPE_HASH|PLACE|N") mod 2^62)

The four parts are joined by `|` as UTF-8 text: the beacon string as it stands
in the ledger, the recipe's hash (64 hexadecimal digits), the place, and N in
decimal, the number of the place's draw counted from 1 across discarded
candidates. Anyone can compute it with a hash tool. Worked example with a
made-up beacon:
`drand:default:1000:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff`,
the recipe hash below, place `T1`, N = 1 give the SHA-256
`c4a12292…937e2a24` and the seed `6164635393569204772`. So a ledger whose tail
was cut away draws the same seeds again, a second ledger with the same beacon
holds the same seeds, and a wrapper around the command changes none: `read`
recomputes every seed and refuses a line whose seed is not the derived one.
What a reader has to compare, because no code can: the beacon string in the
ledger's first line with its public source, and the recipe's hash and the
commit in the `drawn` lines with what the tracker named before the draw.

**What follows from that, and is not this round's to settle:** a seed derived
from public values is no secret. Whoever has the beacon string, this checkout
(which holds `private_recipe`, and so the recipe and its hash) and the place
can compute a test world's seed and compose its truth. Until round 9b the seed
came from the operating system and stood only with the world's truth (entry
16). Keeping test worlds blind now rests on the analysis side never running
the generator on these inputs.

**The source of the beacon** for the test worlds: the `randomness` of one
round of the drand network's default chain (chain hash
`8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce`, one round
every 30 seconds). The round is announced on the tracker before its time. The
operator fetches it from two relays, which have to agree
(`https://api.drand.sh/public/ROUND` and
`https://drand.cloudflare.com/public/ROUND`). The code makes no network call:
the operator fetches, the code records.

**The beacon has one writing, to the letter:** `drand:default:ROUND:RANDOMNESS`,
with ROUND in decimal without leading zeros and RANDOMNESS exactly the 64
lower-case hexadecimal characters of the relays' field `randomness`; nothing
before, between or after, for test and reserve worlds alike. `ledger open`
refuses every other writing (capital digits, a note after it, a leading zero,
the randomness alone, a blank anywhere), because the same value written
otherwise would be other seeds, and a ledger whose first line is written
otherwise does not read. `ledger summary` prints the string, so that a reader
can compare it with both relays.

**The round is final.** The moment the round is out, every seed and so every
world of the draw can be computed by anyone who has this checkout. Going on
with that beacon, or asking for another round, is therefore itself a choice
among worlds. It is not the operator's: the round announced on the tracker is
the round, and anything else goes to a human before a ledger is opened.

**Before the first draw, on the tracker** (the orchestrating session's part):
the ledger's path, the frozen checkout and its commit, the recipe's hash, the
drand round, and the five places with their world ids and public fleet ids.
The operator chooses none of them.

```sh
python -m fleetlab.worlds recipe-hash
```

It is the SHA-256 of the canonical JSON (keys sorted, no whitespace) of what
`private_recipe` returns, without the three fields in which the places differ:
world id, public fleet id and name. At this commit:
`d64334423b024b4461ffa13e06b26da0d4c424ec479609f76d55f4501699bb86` for test
worlds. A reserve world has another recipe (`recipe-hash --kind reserve`), so
other seeds.

**Open the ledger,** once, with the beacon. It refuses a ledger that exists:

```sh
python -m fleetlab.worlds ledger open --ledger LEDGER --beacon 'drand:default:ROUND:RANDOMNESS'
```

**For a place, write its recipe file** from the code, with the identity the
tracker gives. The command refuses a file that exists and a path inside a
checkout:

```sh
python -m fleetlab.worlds recipe-file --world-id wNN --public-fleet-id wMM --name T1 --out FILE
```

**Draw.** One call is one draw:

```sh
python -m fleetlab.worlds generate --recipe FILE --root ROOT --ledger LEDGER --place T1
```

| Exit | What happened | What to do |
|---:|---|---|
| 0 | The seed composed. The world's files are under `ROOT`, and the place has its candidate. | Run it (`tools/heavy-run … run --recipe FILE --root ROOT`). |
| 3 | No composition within the recipe's 400 drafts. The seed gives no candidate; nothing is under `ROOT`. | The same command again: the place's next draw. |
| 2, "refused before any seed was drawn" | Nothing was drawn and nothing written. The class is in brackets: `arguments`, `path`, `recipe`, `calibration`, `ledger`, `place`, `code`, `software`. | Read the class and put it right; nothing is lost. |
| 2, "the attempt failed after the draw" | The draw is open and keeps its seed. The ledger has a line `failed` with the exception's type. | The same command again. If it fails again: stop and report. Do not clear anything by hand first. |

**A draw is never interrupted.** It can take long: the review of rounds 9 and
9b measured 340 seconds for a draw without a composition and 52 seconds for
one with, on a machine that was busy; on a free one a composition takes under
ten seconds. Only a composition or its absence closes a draw. Every other end,
a kill, a power cut, Ctrl-C or an error, leaves the draw open with its seed
recorded, and the same command (the same file and root, the same commit, a
clean tree) repeats that seed. While a draw is open no draw is taken for any
place, and a call with another world id, recipe file, commit, software or
root is refused (class `ledger`), so an open draw cannot be passed over or
replaced. If the world's files were already written, the repeated draw finds
them equal to what it composes; files that differ are an error
(`FileExistsError`) and are not overwritten.

**The places.** A test world's places are `T1` to `T5` and nothing else, in
that order: a draw for a place is refused while the place before it is not
kept. A reserve world's places are `R1`, `R2` and so on without an upper end,
in order in the same way (`recipe-file --kind reserve`), so that the number
of a place chooses no seed. Creation refuses a place whose candidate was
kept (it is filled) or stands unchecked, a world id that the ledger holds
under another place or that was a checked candidate before. Entry 16's stop is
the whole draw's: after three discarded candidates for one place or six in
all, and after ten draws in a row without a composition for one place, no draw
is taken for any place, and the matter goes to a human.

The same command on a root that already holds the place's candidate draws
nothing, changes no byte and prints no seed.

**Read the state** (this is what the orchestrating session reads: per place
the counts, the failed attempts and its candidates in order, each with its
world id and its state, open, standing, kept or discarded, and once checked
its successes, episodes and rate; whether the beacon line is there and whether
the draw is stopped; no seed, no draft, no hash of a world). It exits 1 and
says which line is wrong when a line does not hold:

```sh
python -m fleetlab.worlds ledger summary --ledger LEDGER
```

**After the candidate ran,** entry 16's check is a command. It takes the root
the draw was made into, as the ledger records it (a `--root` that is another
directory is refused), asks the runner's own readiness test on that root
(every step done, both chains held, the export checked by the current code; a
`"ready": true` in the record is not enough), reads the fleet's logged success
over the observed window, successes and episodes, from the runner's record,
counts the same two numbers for itself from the export's table of episodes,
the public table an analysis agent sees, and refuses when they differ. It
compares the rate with the recipe's band (60 to 75%, both ends inside), writes
number and decision into the ledger, and prints one line: place, world id,
successes, episodes, rate, band, decision. Nobody opens the record by hand and
nobody types a number:

```sh
python -m fleetlab.worlds ledger check --ledger LEDGER --place T1
```

Exit 0: kept. The candidate is the place's test world, the place is filled,
and the next place may draw. Exit 4: discarded. Write a new recipe file with
the next world id the tracker gives and draw again for the place under another
root. Exit 2: refused, with the class (`place`: no standing candidate without
a check; `world`: the root does not hold that candidate, the record is another
world's, or the runner's test says it is not ready).

**After every command that writes a line** (`ledger open`, each `generate`,
each `ledger check`), post the head on the tracker: the number of lines and
the last line's hash. Each line carries the hash of the line before it, so a
posted hash fixes every line above it. To compare an earlier posting with the
ledger as it stands, ask for the hash at that line:

```sh
python -m fleetlab.worlds ledger head --ledger LEDGER
python -m fleetlab.worlds ledger head --ledger LEDGER --line N
```

**A cut-off last line** (a power cut in the middle of a write): every command
but `summary` refuses the ledger. Stop and report. Nobody edits a ledger by
hand; what happens to it is the orchestrating session's decision, on the
tracker.

**At the end, and at any time,** `ledger verify` is run from a clean checkout
of the ledger's commit, the one every `drawn` line names. It says where it
runs: from a tree that is not clean, or from another commit, it refuses (exit
2, class `code`) and composes nothing. The reader runs it, not only the
operator:

```sh
python -m fleetlab.worlds ledger verify --ledger LEDGER
```

Reading the ledger already recomputes every seed from the beacon, every
draw's number (1, 2, 3 and so on for a place), and every check's decision from
its own number, and refuses a ledger without its beacon line, a draw with two
outcomes or two checks, a line whose place or root is not its draw's, and
what no command could have written: a `drawn` line from a tree that was not
clean, `drawn` lines that name two commits, and a draw for a place while the
place before it is not kept.
`verify` then asks the running code for each draw's recipe (its hash and its
band must be the recorded ones) and composes every draw again, an open one
too: draft, world hash and forecast seal for a composed draw, no composition
for the others. For every checked candidate the root must still hold it, and
the runner's record and the export's own count the recorded number; a root
that is gone counts as different. It prints counts only. Exit 0: everything repeats. Exit 1:
something differs, or a draw is open. Exit 2: the ledger does not hold (class
`ledger`), or the calibration record is no longer the one the draws were made
on (class `record`; nothing is composed).

**The ledger's lines.** Each is one JSON object with `line` (its kind), `time`
in UTC and `prev`, the SHA-256 of the line before it (of its bytes without the
newline; the first line carries 64 zeros).

- `beacon`, the first line: the beacon string.
- `drawn`, before the seed is composed: the place, the draw's number `n`, the
  seed, the world id, the public fleet id, name and kind, the SHA-256 of the
  recipe file, the recipe's hash, the band, the code's commit, the record's
  digest, the engine's software and the root.
- The outcome that closes the draw, naming its `drawn` line by that line's
  hash (`draw`): `composed` with the accepted draft's number, the world's hash
  and the forecast seal; `no_composition` with the number of drafts and how
  often each rule refused.
- `failed`, naming the same line: an attempt that ended in an error or an
  interrupt, with the exception's type. It closes nothing.
- The check, naming the same line: `kept` or `discarded` with successes,
  episodes, rate, band and root.

Commands that write (`ledger open`, `generate`, `ledger check`) hold an
exclusive lock on the ledger's directory for their whole duration; a second
one fails at once with exit 2, class `ledger`, "in use".

### The recipe file is the code's recipe

`external_recipe` already refused a file that leaves a field out. Stated as
`null`, six fields were still accepted, among them the pinned digest and the
floor, and any other valid value was accepted too (a floor of 0.25, no
retained error, revision `r7`, 4,000 drafts). Now `external_recipe` refuses
`null` wherever the code's recipe states a value, and creation takes only a
file that equals `private_recipe(world_id, public_fleet_id, name=…, kind=…)`
of the running code (`recipe.code_recipe`). The five files of T1 to T5 differ
in the three identity fields and in nothing else, which is what the recipe's
hash says. Since round 9b the file is written by a command, `recipe-file`,
and not by a line of Python typed at the draw.

### The search cannot change unnoticed

Written first, on the code as round 8 left it, and passing unchanged at the
end: `tests/test_worlds_round9.py` pins, for the world the produced recipe
composes under the id `w97` on two public seeds, the accepted draft, the
world's hash and the forecast seal.

| Seed | Draft | World hash | Forecast seal |
|---|---:|---|---|
| assessment seed 10 | 6 | `b72feef4b5c7…c195793c` | `65dc11838466…b3b45a26` |
| the review's other seed 13 | 2 | `7c735b6febbb…a782a3ee` | `10f20f1a1618…c139eb6a` |

The review had made 19 single-place breaks of round 8's code and five passed
every test. Each now fails a test of its own: a taken drawn day replaced by
the earliest day of the range instead of a drawn free one (which also moves a
pinned world); that replacement onto day 0 for a site-wide fault; settling
that moves a plain fault's day only earlier; a folded world compared with its
manifest by episode ids only; a continuation imported although its parent
cannot be read.

### The import: three statements

- **A folded world's files are compared with their hash.** The review removed
  an attempt from a folded world's runs and from its manifest alike, and 1,679
  of 1,680 attempts imported, because the manifest was compared with nothing.
  The import now verifies the world's four files against the `hash` file
  beside them before it compares the runs with the manifest. That closes the
  accident, not the forgery: the review of rounds 9 and 9b wrote the hash file
  anew over the shortened files, and 1,679 of 1,680 attempts imported again. A
  hash file is as rewritable as a manifest. What stands is the world's hash in
  `recipe.py` (`sealed_hash` of the folded world's recipe, in the repository's
  history); the import does not compare a folded world's files with it yet.
- **The recheck of the stop rule assumes an order.** It takes "everything
  before a job's root, in the order given" as the table the job read. Jobs
  r7b and r7c were started five and six minutes before the previous job's
  import was committed, so they did not read that table, or not from their
  start. For the six jobs
  imported this changes nothing (the review tried every table a job can have
  read: all 34 continuations with a readable parent select the same
  attempts). But the assumption can refuse a legitimate job: with five
  attempts counted before job r7b that it did not see, on a cell it continued,
  the whole import raises "selected continuation differs from the stop rule".
  The code is unchanged; a job queued before the previous import is committed
  needs its roots given in the order in which the jobs read the record, or
  the refusal read with this in mind.
- **Nothing is folded into the record after the first draw.** The cohorts of
  a folded world belong to no calibration job, so they count for every
  recipe, also for one that names its jobs. A world folded later, `w48` and
  `w49` included, changes the record's digest under the test worlds' recipe,
  and every world drawn from it then refuses to recompose ("sealed recipe
  calibration changed"). That is the right failure, and it means the record
  is closed with the first draw.

### Rounds 9, 9b and 9c: tests, and what was verified

Round 9b (the same day) closed the gaps round 9 had named in its own ledger: a
seed recorded before it is composed and an interrupted draw repeated with it,
a lock, the chain, entry 16's check as a command, the `recipe-file` command.
Round 9c answers the review of both: only a composition or its absence closes
a draw; the seed is derived from a public beacon and nobody's to choose; the
check asks the runner's own readiness test on the draw's own root, and
`verify` asks the recipe; a tree that is not clean is refused; places in
order and entry 16's stop for the whole draw; `ledger head --line`. The search
was not touched in any of the three: the two pins pass unchanged.

`tests/test_worlds_round9.py` holds 26 tests. Seven (the two pins and the five
breaks of round 8's review) pass on the code as round 8 left it, as decided:
they guard a search that was right. The others fail there. Two of them end a
child process for real: SIGKILL between the seed's line and the composition
and again after the world's files were written, and SIGINT (Ctrl-C) while
composing; each time the test's own process then repeats the draw. A test's
beacon is a made-up string; no test and no code makes a network call.

The generator's ten test files pass on the final code (214 tests), with lint,
format and types as `make check` runs them. Round 9's full `make check` (2,938
passed) was on commit `86e9fe5`, before rounds 9b and 9c; the orchestrating
session runs the whole of it on the merge.

In a scratch copy of the final code (a throwaway git repository; the 26 tests
pass on the unchanged copy), 28 single-place breaks were each made alone, and
each makes a named test fail. Eight are the ones the review of rounds 9 and 9b
found surviving: an open draw repeated although the software changed, or from
another commit; the tree always recorded as clean; a ledger whose front was
cut away read as whole; the check reading a record that names another world; a
folded world without a readable hash file taken as sealed; lines no longer
synced to the disk; and the command on a root that holds the candidate
printing its seed. Twenty are of this round: a failed attempt closing its
draw; the seed drawn at random again; `read` not recomputing the seed, or not
checking the draw's number; a ledger that was never opened taking a draw, or
opened a second time; the check taking any root, or the ready flag alone; a
tree that is not clean drawing; a place drawing before the one before it is
kept; any name taken as a place; ten draws without a composition, or entry
16's limit, not stopping every place; a draw with two outcomes; a check's
decision not recomputed, or its place not its draw's; `head --line` printing
the last hash; `verify` passing an open draw, skipping a checked root that is
gone, or not comparing the band with the recipe's. Earlier: 22 of 22 on round
9b's ledger and 21 of 21 on round 9's code. The first run on the unchanged
copy failed one test: the copy's tests were started in the background, where a
process inherits an ignored SIGINT, and the interrupt test's child never got
its Ctrl-C. The child now puts the default back before it sends the signal;
the unchanged copy passes all 26, and the two breaks that only that test
catches (a failed attempt closing its draw, `verify` passing an open draw)
were made again and are caught.

The commands were run once as an operator gives them, each as its own process
and unpatched, in scratch paths under `/var/tmp`, at commit `baf3bb6` with a
clean tree and a made-up beacon: `recipe-hash`; `ledger open`; `recipe-file`;
`generate` for T1 with Ctrl-C (SIGINT) half a second after the seed's line
appeared (exit by the signal; a `drawn` and a `failed` line; the summary shows
one open draw with one failed attempt); the same command again (exit 0 after
7 seconds; `composed`, draft 11, naming the same `drawn` line); once more (no
line); `ledger check` without a record and then with a hand-written record
that says ready (exit 2 both times: the runner's own test says no);
`ledger head` and `ledger head --line 2`; `ledger verify` (exit 0, one draw
repeats). The transcript is in
`~/.cache/robots-probes/worlds-v4/20261011-r9-evidence/transcript9c.log`
(rounds 9 and 9b: `transcript.log`, `transcript9b.log`).

**Round 9d** (the last before the merge; commit `b361fdd`) answers the review
of round 9c. The beacon has one writing, and `ledger summary` prints it;
reserve places are drawn in order; `read` refuses a `drawn` line from a tree
that was not clean, `drawn` lines naming two commits, and a draw for a place
whose predecessor is not kept; `ledger verify` refuses from a tree that is not
clean and from another commit than the ledger's; `ledger check` counts the
observed window from the export's table of episodes and refuses when the
runner's summary says otherwise. That the seed is public is decided and stays:
every party that has the checkout also has the truth files on this machine.
`tests/test_worlds_round9.py` now holds 27 tests; the made-up beacon of the
tests has the one form. The generator's ten test files pass on that commit
(215 tests), with lint, format and types. Ten single-place breaks of this round's code were made
in a scratch copy: nine make a named test fail, and the tenth (the form not
checked in `open_ledger`) changes nothing, because the line itself is refused
when it is written. The transcript above is of round 9c: its made-up beacon
would no longer open a ledger, and it was not run again.

Not done: no test world, no world run, no GPU job, no smoke world, no full
`make check` in rounds 9b, 9c and 9d. Open, and not this round's to settle: a
derived seed is computable by anyone with the beacon and this checkout (see
"Drawing a test world"). Not built: the import comparing a folded world's
files with the hash in `recipe.py`; any fetching of the beacon (by decision).
Not checked: `ledger check` passing on a record and an export that a real run
wrote (the runner's readiness test needs a real export and its receipt; the
tests stand in for that test and write the export's table by hand, with the
one column the count reads, and the transcript shows only the refusal); the commands
from another checkout than this one; the drand relays and the format of their
answer (no network call was made).
