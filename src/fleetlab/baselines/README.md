# Claim-3 baselines 1–4

`baselines-v4` completes the `set` effect rule and records one rerun of a lost
runtime answer, following a third independent review. Baseline selection uses agent-visible exports. The command also
reads the oracle report's **seed and world ID only**, as explicitly requested
for reproducing the chance draws; no oracle gains, fault instances or scoring
identities enter selection or model prompts. Evaluation happens separately.
The classical baseline belongs to the other builder and is untouched.

## Run and scorer handoff

```sh
make baselines EXPORT=~/.cache/robots-probes/worlds/w41/export-review-20261009T173751Z \
  BASELINES_OUT=~/.cache/robots-probes/baselines/NEW-RUN
```

The CLI accepts `--export`, `--out`, `--seed`, `--chance-table`, `--draws`
(default 200), and `--offline`. Make forwards `SEED`, `CHANCE_TABLE` and
`BASELINES_FLAGS`. Default seed: the integer `seed` recorded in
`EXPORT/../oracle/oracle.json`, checked against the export's world ID. A copied
export can use `--chance-table` to name that world's report, or explicit `--seed`.
There is no map from world IDs to seeds. This reads the standard world report;
a report made with a bootstrap-seed override must not replace the chance-table
record. The seed never reaches either model.

An export contains only `log/` and `data_dictionary.md`, without symlinks.
Run/world roots are refused before querying or creating output. Output must be
a new directory separate from the export. Existing records are never rewritten.

Both model methods run twice. Each repeat makes one no-data call and one
six-video call: four subscription calls per world, requested `gpt-6-astra`,
xhigh, fast service tier, no tools or skills. **Repeat 1 is scored**, even if
invalid or missing; repeat 2 never substitutes for it. Login/rate-limit errors
record the failure and stop all subsequent calls. Other failed calls or invalid
answers are recorded missing for that method/repeat and allow the other calls.
Each timeout or identified transport failure **without any answer** gets exactly
one rerun with the same prompt, images, schema, model, effort, tier, subscription
and 600-second limit. Both starts are retained in `attempts`; the rerun occupies
the same repeat, including its scored flag. Vendor 5xx is transport failure;
`cli_error` requires an explicit connection/transport diagnostic. Other CLI
failures, malformed/schema-invalid answers and semantic invalidity are not rerun.
A historical lost start may be supplied to `model_call(previous_attempt=...)`;
this permits only its single remaining start. The caller must replay its saved
prompt and images unchanged. Two lost starts mean missing, never zero: the
manifest marks that repeat's comparison unavailable with null margin/ratio;
a missing scored repeat also makes the canonical comparison unavailable. The
scorer must retain this absence rather than dropping a competitor.
Exit 2 means any requested repeat is missing. `--offline` explicitly skips both
repeats and exits 0. There are no silent fixed-prior substitutions.
The runtime owns authentication; the package never accesses login material.

```
manifest.json                         # wording, seed, repeat paths, missing/stopped
random/draw-000/ ... random/draw-199/
worst_health/
fixed_prior_fallback/                 # separately labelled stage-2 prior
no_data/                             # repeat 1, scorer's canonical path
dramatic_videos/                      # repeat 1
repeats/repeat-2/no_data/
repeats/repeat-2/dramatic_videos/
video-inputs/                        # repeat 1 selection and decoded filmstrips
repeats/repeat-2/video-inputs/
runtime/<call-id>/                    # actual prompts, responses, accounting
```

Each method/draw contains stage-2 `RankedFixSet` in `outbox/fix_set.json`,
budget rankings in `budgets/{count-3,count-6,cost-4,cost-8}/outbox/`, and
**authoritative FixSets** in `fix_sets/{count-3,count-6,cost-4,cost-8}.json`.
`scoring.baselines.read_baseline_fix_set(root,budget)` validates those cuts.
Metadata records repeat/scored flags and wording. Expected gain zero and
interval [-100,100] are labelled unestimated schema placeholders.

## Recorded decisions and information boundaries

* Random: 200 uniform permutations without replacement, NumPy default_rng(seed).
  Count budgets take first k; cost budgets walk that permutation, skipping
  actions that do not fit. Rankings are uniform, cost-feasible sets are not.
  No filtering by truth, effect or lookup availability occurs.
* All methods use the stage-2 operator value rule: `set` writes the recorded
  fleet standard, `change_condition` writes the site's recorded condition,
  training requests use `any failure mode`. Unrecorded conditions are omitted
  because they cannot be completed. Models select indices, **never values**.
  Filled values are not shown to either model. Both prompts explain this rule,
  robot > site > fleet precedence, unchanged writes and exact-scope reversion.
  The full rule in both prompts is: “A set writes the recorded standard value
  of that key at that scope, so it changes something only where the current
  value at that scope differs from the standard, and at fleet scope it changes
  nothing on a robot or site that carries a value of its own.” The standard is
  the fleet's recorded value; no current values or overrides are disclosed.
  Fleet-scope entries remain eligible; the model decides their position.
* Both prompts use the analysis agents' exact scoring rule: “Neither changes
  what the robots do in the scoring window, so their expected gain in fleet
  success is zero; they belong at the bottom of the list, and in the diagnosis
  as a retrain indicator.” This identifies `route_to_teleop` and
  `request_training_data`. It is a scoring rule, not world data.
* Accepted model actions must belong exactly to the filled catalogue, preserving
  teleoperation task classes. Recorded values are checked against applicable
  simulator domains: positive integer policy counts, supported X-VLA precision,
  control mode, nonnegative latency, numeric condition shapes and clutter range;
  nonfinite values are refused. Invalid output is missing, never scored zero.
  These checks establish applicability, **not existing replay coverage**. A
  valid action can still require new oracle replays; report it as unvalued.
* No-data sees only topology, component names, vocabulary, costs and catalogue
  verb/target pairs, plus static teleoperation task classes. No current values,
  per-robot overrides, health, dates, logs, outcomes or videos. It ranks 32
  distinct indices, or all if fewer; the schema requires at least ten. Tests
  alter fleet/site/robot settings and require byte-identical prompts.
* Fixed prior keeps the stage-2 helper, labelled `fixed_prior_fallback`. Its
  implementation lives in the pure `baselines.prior` module; the scorer
  reexports that same helper. Blind baseline imports never load the scorer.
* Health: temporal median per robot/component/metric; standardize within metric
  by median and 1.4826 × median absolute deviation. Both cameras share the
  calibration-residual population. Zero scale falls back to population standard
  deviation, then 1 for identical readings. Lower grip force is worse; larger
  residual, cycles, torque and CPU are worse. Benign metrics remain eligible.
  Component severity is its worst standardized metric. Ties: robot ID, component
  name. Nonfinite readings ignored; unmeasured components last/listed; unknown
  metrics fail. Cameras recalibrate; grippers/arms/controllers service.
* Drama selects failures with video by duration descending, total logged
  end-effector Euclidean travel descending, then episode ID. Motion only breaks
  duration ties; no robot diversity quota. One call sees the first six clips
  (all if fewer; none means missing), each as 16 evenly spaced decoded frames
  per available camera, including start/end with camera timelines separated.
  Sampled viewing can miss brief events. The model gets task text, IDs, duration,
  static description and catalogue, without telemetry/settings/health/logs.
* Each clip permits every offered verb on that robot's components/keys or its
  site's conditions/keys. Other robots/sites and fleet scope are excluded.
  One nomination per clip, in frozen clip order. For count-3, take the **first
  three clips before deduplicating**. Later clips do not fill vacancies caused
  by repeated actions. Count-6 and cost cuts use all six, deduplicated in first
  occurrence order. Both prompts are neutral about being baselines or repeats.
* Video rankings pad to ten solely for the stage-2 schema. Labelled FORMAT
  PADDING ONLY entries **never enter FixSets**; score explicit FixSets, not a
  recut padded ranking. Both repeats derive every budget from their one answer.
* Budgets/costs retain the existing contract: 3/6 actions, 4/8 hours. The
  historical cost prose/table conflict remains open; no costs changed here.
  Chance intervals remain the scoring builder's responsibility.

## Evidence and limits

The import repair on `fix/baselines-imports`, based on batch-3 `752b601`,
makes package exports lazy and separates the unchanged fixed-prior helper
from scoring imports. Fresh interpreters assert that the package, classical
package and simple/model/video entry points load no `fleetlab.engine`,
`oracle`, `worlds`, `faults`, `scoring` or `sim` module. Access through both
leaf modules and the public package is checked, including the CLI import.
The seed reader is now `baselines._seed`, with standard-library imports only.
Only a call to `baselines.random.random_fixes` without an explicit seed loads
and invokes it. The CLI derives its manifest seed from those returned draws;
`simple.random_fixes` remains a compatibility reexport. No wording changed.

Stored D1 selection replay compares the pre-repair code and the repair to the
same immutable records: all 200 random draws, health, fixed prior, both v4
no-data repeats and both retained video answers. Captured model responses
are replayed; no model is called. All ranked answers, four budget cuts and
metadata bytes match, retaining the original runtime/wording provenance.
Classical source is unchanged; its stored diagnosis/ranking/cuts/evidence
are reserialized without fitting or selecting new statistical hypotheses.
The comparison covers **2,067 JSON records**, including that classical
serialization. Receipts and the replay script are external, under
`~/.cache/robots-probes/baselines/20261009-import-isolation-752b601/`.
This verifies selection/serialization stability on D1 with high confidence;
it makes no new live-model, statistical-refit or simulator claim. Only
baseline and classical test files run for this repair, with one pytest process
and sequential cold-import children; no `make check`, GPU job or smoke queue.

The review-3 repair was measured at executable commit **`6b3a75a`**, in detached
`~/.cache/robots-baselines-v4-run-20261009`. The new records are outside this
package, under `~/.cache/robots-probes/baselines/`:

* `20261009-6b3a75a-v4-development-results.json` and the adjacent `.md` contain
  the six live v4 no-data repeats and the historical D1 video rerun, every
  budget, exact FixSets, rescued/broken counts, gains/shares, lookup coverage,
  world hashes, oracle run paths and prompt/image hashes.
* `20261009-6b3a75a-v4-wNN-no-data/` holds the two revised no-data repeats for
  each old world. `20261009-6b3a75a-v3-w41-video-rerun/` holds the **one remaining
  start** for the old lost video repeat. It is **v3 wording**, replaying the
  original prompt, schema and six images byte-for-byte; v4 is its rerun policy.
  `same-condition-verification.json` proves this and retains both starts.
  The accepted answer occupies scored repeat 1; the old v3 repeat 2 is unchanged.
  Future video calls use v4's completed rule. No independent model sessions ran.
* `20261009-6b3a75a-v4-verification.json` records `make check` **exit 0**:
  lint, formatting, strict types, **1,139 passed, 2 skipped, 62 deselected**.
  The 56 focused tests pass. Each of the five formerly surviving mutations
  independently fails its targeted test; logs are in
  `20261009-6b3a75a-v4-mutation-evidence/`; the recorded failures are the
  intended assertions, not import errors.

All seven new starts succeeded: six no-data calls and the single authorized
historical video rerun. Every new budget cut is valued; **none is unvalued**.
No fresh replay/simulation was needed. All six no-data prompts are byte-identical
and retain the static allowlist; `20261009-6b3a75a-v4-final-validation.json`
records that check. No-data now puts fleet-scope **revert software.version,
revert control.mode, revert control.action_delay_steps** first in every repeat;
those three-fix cuts gain zero. The following replaces v3's no-data cells and
its missing D1 video first repeat. Other v3 video repeats, health and random
records remain unchanged. Cells are **gain in percentage points / share % of
the same-budget oracle ceiling**; each world has 336 scoring episodes.

| World, method, repeat | 3 fixes | 6 fixes | 4 hours | 8 hours |
|---|---:|---:|---:|---:|
| w41 no-data v4, 1 | 0 / 0 | 0 / 0 | 0 / 0 | −0.298 / −0.81 |
| w41 no-data v4, 2 | 0 / 0 | 0 / 0 | 0 / 0 | −0.298 / −0.81 |
| w41 video v3 rerun, 1 | 5.655 / 28.36 | 5.655 / 15.32 | 5.655 / 20.00 | 5.655 / 15.32 |
| w42 no-data v4, 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| w42 no-data v4, 2 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| w43 no-data v4, 1 | 0 / 0 | 0 / 0 | 2.976 / 9.09 | 2.976 / 8.40 |
| w43 no-data v4, 2 | 0 / 0 | 0 / 0 | 3.274 / 10.00 | 3.274 / 9.24 |

Signed net episodes behind these rounded gains: D1 no-data at 8h is −1
(4 rescued, 5 broken), its video rerun is +19 (19 rescued, 0 broken) at every
budget; D3 no-data cost cuts are +10 (12 rescued, 2 broken) in repeat 1 and
+11 (13 rescued, 2 broken) in repeat 2. Exact counts and FixSets are in the
external JSON. Lookup confidence is high; this does not establish performance
on new worlds. The 22:09 UTC
readiness check found no parent comment beginning `NEW WORLDS READY`, so
w44/w45 were not opened or measured. The original runtime-loss cause remains
unknown; the requested model is not independent confirmation of the serving model.
The stage-4 scoring branch was not run in this checkout; its handoff is the
explicit FixSets and missing/unavailable manifest records described above.

Historical v1 and v2 runs remain immutable in
`~/.cache/robots-probes/baselines/20261009-3898b12-D1/` and
`~/.cache/robots-probes/baselines/20261009-bb4dfda-v2-wNN/` respectively.
The former package results record is archived byte-for-byte at
`~/.cache/robots-probes/baselines/20261009-v3-review-archive/development_results_v2.json`.
Those model results are superseded by v3: they lacked the teleoperation scoring
rule, guessed values, and used separate vision calls for different budgets.

Live v3 **executable commit `7eea162`** was measured in the clean detached
`~/.cache/robots-baselines-v3-run-20261009` checkout. Results are external:

* `~/.cache/robots-probes/baselines/20261009-7eea162-v3-development-results.json`
  contains both repeats, all four budgets, exact FixSets, rescued/broken counts,
  points/shares, ceilings, missing replay lists, validation and call accounting.
* The adjacent `.md` gives the model-budget table. Per-world directories
  `20261009-7eea162-v3-w41`, `...-w42`, `...-w43` hold original runtime inputs,
  results, manifest, every answer, `oracle-lookup.json` and contract checks.
* `20261009-7eea162-v3-verification.json` records `make check` exit 0: lint,
  formatting, strict types, **1,123 passed, 2 skipped, 62 deselected**. Forty
  focused baseline tests passed. Archive hashes are recorded separately.

D1 used rebuilt `export-review-20261009T173751Z`; old D2/D3 used `export`.
All twelve final-wording subscription calls ran, with **eleven accepted answers**.
D1 video repeat 1 timed out after 600.036 seconds with only thread/turn-start
runtime events, no authentication/rate-limit event. It is **missing at every
budget**, never zero in the original v3 run. Repeat 2 succeeded at identical wording
and images in 49.963 seconds. The cause is unknown.
No retries or login-file changes occurred. D2/D3 commands exited 0; D1 exited 2
because of the missing first video answer. Requested model was `gpt-6-astra`; the CLI
does not independently identify the serving model.

Valid no-data answers still have **oracle coverage gaps**: w41 repeat 1 at 4/8h
needs 336 rows per cut, repeat 2 at 8h needs 112; w43 repeat 1 at 4/8h needs 28
per cut. These are applicable recorded-value actions, not invalid prose values.
Exact actions and required states are in the external record. Random is complete
for all D1 budgets and D2 count-3; other means/shares are withheld: D2 count-6,
4h,8h have 1,4,7 missing draws; D3 count-3,count-6,4h,8h have 1,2,10,24.
No incomplete sample is presented as the 200-draw chance level.

All saved FixSets fit their budgets and contain valid recorded catalogue actions.
Runtime prompt checks confirm the scoring sentence and static allowlist; the six
no-data prompt bytes are identical across worlds/repeats. Points mean
100 × (rescued − broken) / 336 scoring episodes; share means net gain / oracle
ceiling at the **same budget**. Repeat 1 is scored. Inputs/world hashes/oracle
run paths are pinned in each receipt. These exact lookups have high confidence;
new-world performance and the cause of the timeout remain unknown.

A fresh, non-overwriting recheck from an environment at the measured commit:

```sh
uv run --locked python ~/.cache/robots-probes/baselines/20261009-7eea162-v3-reproduce.py \
  ~/.cache/robots-probes/worlds/w41 \
  ~/.cache/robots-probes/baselines/20261009-7eea162-v3-w41 --commit 7eea162 \
  --receipt oracle-recheck-NEW.json --validation contract-recheck-NEW.json
```

No simulator-running source changed, so no new check-sim/smoke was required or
run. No fresh missing-row simulations were started in this revision. The previous
D3 chance replay had no completed run record available to this lookup. The latest
readiness check had no comment beginning `NEW WORLDS READY`;
w44/w45 remain unopened and their runs await that explicit comment. No results
record ships inside the Python package. No merge, push or docs/skills edits.

## The build rule: a fix set read off the generator's build (not a baseline)

Written on 2026-10-11, before the first test world is drawn, and repaired the
same day after its review (the sentence, the evaluation, one refusal in the
scorer; no fix set changed). Every world the stage-4 recipe composes is built
alike (`src/fleetlab/worlds/README.md`, "One structure under every composed
world"), so a rule that analyses nothing can name the largest fixes from the
build. `build_rule.py` is that rule, so that a report on the test worlds can
show how far it gets. It is **none of the baselines of the design**: nobody
who runs a fleet has its generator's build, `python -m fleetlab.baselines` and
`make baselines` do not start it, the scorer refuses its directory, and no
claim's margin is read against it.

```sh
python -m fleetlab.baselines.build_rule --export EXPORT --out NEW_DIR
python -m fleetlab.baselines.build_rule --evaluate NEW_DIR --world-root WORLD --out NEW_FILE.json
```

**The rule** (`build-rule-v1`, the constant `WORDING`; a test holds a literal
copy). The sentence says what the code does; where the two part, the sentence
is wrong:

> One revert for every scope and key that has an entry in the change log made
> at a site, then one for every scope and key with an entry made for the whole
> fleet, each group in the order of its earliest entry (day, site number,
> key). Then, for every site at which every task run is among the ten
> sentences of one suite and for which no site entry was read, the action that
> replaces its batch of objects by the site's recorded batch, by site number.
> An entry whose scope or key the vocabulary does not hold is left out; an
> action the catalogue does not hold is skipped and the next moves up. The
> first three actions are the answer at three and at six fixes; at four and
> eight hours, those of the three that fit, in order. No episode outcome, no
> health reading and no video is read.

What stands behind its words:

* **What is read.** Three queries: `day, scope, key` of `change_log`, the
  distinct `site_id, task_text` of `episodes`, and the catalogue. Besides
  them the system description: the list of sites, each site's recorded batch,
  and the values the catalogue's entries are filled with. A test records every
  query and finds these three; flipping every outcome and health reading
  leaves the answer unchanged.
* **The ten sentences of one suite.** An export names no suite. The module
  holds the two task lists of the recipe (`SUITES`: the ten `libero_spatial`
  and the ten `libero_object` sentences), and membership is literal: one other
  task at a site, or one written differently, ends that site's batch, and
  `metadata.json` says why. So does a site with no episode.
* **A site entry that was read** is an entry at a site whose scope and key the
  vocabulary holds, also when its revert was then skipped for want of a
  catalogue entry. A site whose only entry has a key outside the vocabulary
  keeps its batch; the entry is listed under `unread_change_entries`.
* **One revert for a scope and key.** Two entries of one key at one scope give
  one action. Every export has two fleet entries of `software.version`: a
  harmless patch on day 0 and a later one. The oracle values the revert as
  removing every fault whose change was made for the whole fleet on that key
  (`contracts.actions.resolve`); it is not "undo the last entry". The reason
  recorded with it (`because`) names the earliest entry and how many there
  are, so on every export it names the patch of day 0, not the release behind
  the fault.
* **Hours.** A revert costs 0.1 h and a batch 4 h. In every export a revert
  stands before the batch, so the 4-hour set holds no batch. With no site
  entry (2 of the 24 worlds below) the three actions are the fleet revert and
  two batches, and the 8-hour set holds one batch only.
* **Nothing exists at one fix.** The module writes sets at three and six
  fixes and at four and eight hours, the scoring budgets.

The output directory has the layout of one baseline method without its
`budgets/`: `outbox/fix_set.json`, the four files in `fix_sets/`, and
`metadata.json`. The ranked list is padded to the ten entries the stage-2
format wants, labelled FORMAT PADDING ONLY; the files in `fix_sets/` are the
answer, and `metadata.json` says so (`explicit_fix_sets_authoritative`). It also holds the wording, the named
actions with what each was read from, what was skipped, left unread or cut
off by the three, each site's suite, the SHA-256 of the four files read, and
`baseline_of_the_experiment: false`.

### Its evaluation, outside the claims

`--evaluate` values exactly what this module's rule names from the world's own
export. It computes the rule again from `WORLD/export` and refuses the
directory unless its four fix sets, the rule's version and its wording equal
that: a fix set cut by hand, a directory of another version or wording, and
the rule run on another export under the world's root are each refused (three
tests, which the first delivery failed). The recomputed answer then goes
through `scoring.decisions.score_decisions`, the function that values every
baseline's fix sets, at `SCORING_BUDGETS`, over the oracle table restricted to
the scoring episodes. The world, the table and the replay check are loaded as
`scoring.__main__.score_world` loads them: a world that is not ready and an
oracle report of another world are refused, and without full replay
verification nothing is valued. Whether the export holds a clean receipt of
the current export check is recorded as `export_receipt_problem`. The command
writes one JSON file of its own: per budget the actions, hours, rescued and
broken episodes, points and share, and the achievable gain.

**The scorer refuses the rule.** `scoring.loading` refuses a `--baseline`
source whose own record says `baseline_of_the_experiment: false`, under any
name. That is the one change to the scorer in this delivery; a source without
that word is read as before. The refusal looks for the record beside the list
it is handed, so the rule's directory holds no list without it: a baseline's
`budgets/<label>/outbox/fix_set.json`, a ranked list per budget with no record
beside it, is not written for the rule (`write_rule`). The second commit still
wrote it, and the scorer accepted `DIR/budgets/<label>` and cut the padding
into six, nine and ten actions. Now every path into the directory that the
scorer can read is refused: the directory, `metadata.json`, `fix_sets/` and
each file in it, `outbox/` and its file.

**What remains possible.** The four fix sets copied out of the directory
without `metadata.json` are an ordinary list to the scorer: it values them
and lists them under whatever name they are given. Nothing inside a fix set
says where it came from, so only this instruction stands against it: do not
hand the rule's fix sets to `python -m fleetlab.scoring`. The best baseline
and the ratios would not move even then, since they come from the fixed
opponents alone.

The tests show: the refusal, for every path into the directory; the scorer's `report.json` and `report.md` byte for
byte the same with and without the rule's directory and its evaluation file
present (beside the baselines of the same producer run, in the world's root,
in its oracle directory, beside the analysis); the same four fix sets as an
ordinary list valued by the scorer as the evaluation values them; and the
rule's name not among the scorer's fixed opponents
(`comparisons.BASELINE_METHODS`), from which alone the best baseline and the
ratio come.

**Beside the baselines the rule is read at three fixes only.** Its evaluation
file holds all four budgets, but the rule never has more than three actions.
At six fixes and at eight hours it stands with three next to methods that use
six (shares 0.68 and 0.61 on `w48`, 0.52 and 0.50 on `w49`), which understates
what the build gives away; at four hours it is the two reverts alone and
equals the stored no-data ranker on both worlds (13.39 and 10.42 points).

### The count on the public seeds, by which the wording was chosen

The recipe composes on 24 of forty public seeds, twelve on each set (round 8).
For each of the 24: what a public change log would show (at site scope the
site-wide delay, in 22 worlds one entry, in two none, since the furniture
shift and the batch are conditions and write none), the sites of one suite,
and the forecast gains of the worlds README's table. A variant is counted by
the faults its at most three actions remove: whether the largest fix is among
them, whether both of the two largest are, and the share named below.

| Order of steps | Largest fix named | Both of the two largest | Mean share of the best three |
|---|---:|---:|---:|
| The worlds README's ("the site-wide change at the site whose backdrop changed, then the batch at the other spatial-only site"); needs the decoy's site, which no export shows without video | 21 | 16 | 71.4% |
| Site entries, then the batch (the wording this started from) | 23 | 16 | 75.4% |
| **Site entries, then fleet entries, then the batch (chosen)** | **23** | **19** | **90.2%** |
| Fleet entries, then site entries, then the batch | 23 | 19 | 90.2% |
| Site entries, then the batch, then fleet entries | 23 | 18 | 88.3% |
| Site entries, then the batch, then the robots' delays, largest first | 23 | 18 | 87.5% |

Two labels belong with these numbers.

* **The share is a share of forecast gains, not of the oracle's achievable
  gain.** "The best three" are the three faults with the largest forecast
  gain of a single fix (points of fleet success in the scoring window); the
  share is the summed forecast of the faults the rule's actions remove over
  the sum of those three, a forecast that adds single gains. On the two
  worlds that have both, the review found them to agree: forecast share 0.956
  and 0.755 on `w48` and `w49`, oracle share at three fixes 0.958 and 0.764.
* **The chosen order is first of 85 and not alone.** All 85 orderings of one
  to three of five steps were counted (the three above, the robots' delays,
  the robots' policy entries), ranked by largest, then both, then share.
  Fleet, site, batch ties with the chosen order on all three; the tie is
  broken by the first action, which is the largest fix in 22 worlds for the
  chosen order and in none for the other.

What each step adds:

| Steps | Largest fix named | Both of the two largest | Mean share |
|---|---:|---:|---:|
| Site entries alone | 22 | 0 | 50.1% |
| Site and fleet entries | 22 | 4 | 66.7% |
| All three | 23 | 19 | 90.2% |

The worlds README counts its own wording at 20 and 16: there "names the largest
fix" means that the site-wide change it names is the largest fix. That is
found again exactly; as a set of actions its wording also holds the largest
fix in the one world where that is the batch, hence 21. Every one of the 24
worlds has exactly one fleet release fault, which is what the fleet step
removes. Where the delay sits at the mixed site, three worlds, the chosen
order keeps the fleet revert and the batch at the lower site number and drops
the second batch. The one world whose largest fix it misses has a furniture
shift, which no log shows.

The chosen order was also run by the module's own `nominations` on the 24
drafts recomposed in memory, with `contracts.actions.resolve` deciding what
each action removes: 23, 19 and 90.2% again, the same actions and removals in
every world. The review recounted it three ways.

### On the development worlds (2026-10-11)

At three fixes, by the command itself (both worlds are certified for the
current export check since the review):

| World | The three actions, in order | Rule | Achievable | Classical statistics | Dramatic videos | No-data ranker | Worst health | Random (mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `w48` | revert the delay at s1; revert the fleet's software version; replace the batch at s3 | 20.24 (0.96) | 21.13 | 18.45 (0.87) | 2.98 (0.14) | 1.79 (0.08) | 0.60 (0.03) | 0.36 |
| `w49` | the same three actions | 16.37 (0.76) | 21.43 | 18.45 (0.86) | 0.30 (0.01) | 0.00 (0.00) | 1.79 (0.08) | 0.47 |
| `w50` | revert the delay at s3; revert the fleet's software version; replace the batch at s1 | not evaluated: its oracle was still running | | | | | | |

Points of fleet success over the 336 scoring episodes, and in brackets the
share of the achievable gain at three fixes. `w48`: 80 episodes rescued, 12
broken; `w49`: 58 and 3. No action needed a fresh replay. The baselines'
numbers are the stored score reports' (first repeat of the two model
baselines), `scoring-checks/gen4-score-r8-3lvneCNJ/`.

On `w49` the fleet revert gains nothing: that world's recipe has no fleet
release fault and both fleet entries in its log are harmless (the review's
finding), and the rule's first action alone and its first two both give 10.42
points. On `w48` they give 13.10 and 13.39. Every world of the test recipe
has exactly one fleet release fault (all 24 recomposed drafts), so there the
second action removes a fault.

### What changed after the first run on a development export

The order of steps and the fix sets did not. Between the first run on the
three exports and the first delivery the module learned to leave out an entry
of the log whose scope or key the vocabulary does not hold, where before it
stopped with an error, and gained the two metadata fields that list such
entries and sites; no development export has one, and the fix sets were byte
for byte the same before and after. After the review the sentence was
rewritten to say what the code does, the recorded reasons were reworded,
`--evaluate` was made to compute the rule again, and the scorer got its one
refusal; a third commit stopped writing `budgets/`, whose padded lists the
scorer still read. The fix sets of the first delivery and of this one are byte for byte
the same on the three development exports and on 17 constructed exports that
take the sentence apart clause by clause (a test pins those 17), and the
count on the 24 worlds is unchanged.

### Evidence, verified and not

Scripts, logs, the 24 recomposed drafts, the rule's output directories and
the evaluation files are in
`~/.cache/robots-probes/baselines/20261011-build-rule/` (`r2/` for the second
commit, `r3/` for the third).

* Verified: lint, format and strict types over the repository (the three
  commands of `make check`, in the main checkout's environment); the 41 tests
  of `tests/test_build_rule.py`, the scorer's own five test files and the
  baselines' two (245 tests in all, the scorer's five last run on the second
  commit); 27
  single-place breaks of the module and of the scorer's refusal, each caught
  by a test, among them six of the seven the review found uncaught (the
  seventh, `export = read_from`, is gone with the recomputation, and its
  counterpart, the rule recomputed from the path the directory names, is
  caught); the count and its recount; the three runs and two evaluations
  above, by the command.
* Not run: the whole of `make check` and `make smoke-world` (the merge runs
  them); no simulator, model or world run was needed or started.
* Not verified here: the forecast shares 0.956 and 0.755, which are the
  review's; `w50`'s evaluation. A test world's export with a task outside the
  twenty sentences at a site would leave that site without its batch, and
  `metadata.json` would say why.
