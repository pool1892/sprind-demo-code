# The oracle

The oracle replays each scoring episode of a world under subsets of its faults and keeps
the outcomes in a table, so that any fix set that only removes faults is valued by
lookup, signed (rescued minus broken), with both parts kept. Everything the claims read
from the truth side follows from that table: the true gain of each ranked action and of
each fix set (claims 2 and 3), the achievable gain at every budget, the episodes that
still fail with every fault removed (claim 5), whether each detectable cause shows in
the oracle, the chance level of random fixes, and an interval for every gain.
Claim 5 counts failures of the policy itself on a healthy robot, the measure behind
"needs new training data". A partial removal can still rescue such an attempt; the
healthy outcome is measured directly, independently of catalogue coverage.

Specification: the contracts (`docs/design/contracts.md`) sections 8 and 9, the
experimental design, sections 1 and 4 to 6. The
contract's data types and lookups (`OracleRow`, `OracleTable`, `Gain`,
`value_fix_set`, `fault_effect`, `interaction`, `achievable_gain`) are in
`fleetlab.contracts.oracle`; this package plans and runs the replays, builds the table
and computes what follows. It grew out of the miniature fleet's first oracle
(`fleetlab.integration.oracle`), which now goes through it.

## Running it on a world

```sh
uv run --locked python -m fleetlab.oracle plan ROOT      # the manifest and its counts, before anything runs
uv run --locked python -m fleetlab.oracle run ROOT       # the replays: one heavy job (tools/heavy-run inside)
uv run --locked python -m fleetlab.oracle report ROOT    # table, gains, intervals, chance, checks; prints summary.md
uv run --locked python -m fleetlab.oracle all ROOT       # the three in a row

uv run --locked python -m fleetlab.oracle estimate ROOT/files/spec.json   # an upper bound from a spec alone
uv run --locked python -m fleetlab.oracle check ROOT     # the stored manifest against the world, no simulator
uv run --locked python -m fleetlab.oracle fresh ROOT --name NAME --fix-set FILE [--catalogue] [--oracle-items]
uv run --locked python -m fleetlab.oracle fresh ROOT --name chance --chance --draws 200
uv run --locked python -m fleetlab.oracle verify ROOT --oracle-dir DIR --window observed --window experiment
```

Options: `--confirm full|single|none` (the confirmation replays, below; `full` by
default), `--oracle-dir DIR`, `--runs RUN`, `--workers N` (`0`: as many as the engine's memory guard admits when the heavy job starts), `--boot N` and `--draws N`
(report), `--no-truth-table`, `--replace` (plan), `--here` (run a heavy step in this
process: what the heavy job itself is started with).

`ROOT` is a world root as the world generator and the fleet runner leave it (the miniature fleet's chain root is one too):

```
<root>/files/           the world files (contracts section 9) and the truth side beside them:
                        fault_table.json, world_truth.json, task_texts.json (w10: world/)
<root>/runs/<name>/     the world as run; read, never written
<root>/runner.json      the fleet runner's record: a heavy run waits for "ready": true
<root>/oracle/          this package's, nobody else's:
  plan.json, jobs.jsonl           the replay manifest and its counts
  runs/<world>-oracle/            the engine run of the replays; truth/oracle.parquet here
  oracle.json, summary.md         every gain with its interval, as JSON and as Markdown
  episodes.parquet                per scoring episode: faults, outcome as run and with every fault removed
  fresh/<name>/                   fresh replays: plan.json, jobs.jsonl, run/, values.json
  sources.json                    world hash and absolute source-run paths
  verification/                   seconds, comparison evidence, third/fourth executions
  measured_causes.json             realised cause list beside the curve list
  causes/                          staged observed replay plans and resumable run
  steps.json                      when each step ran, its queue time and throughput
```

`--oracle-dir DIR` writes elsewhere (the tests read the mini-v1 fixtures that way, which
must stay unchanged). `--runs RUN` adds run directories outside the root. Every step
resumes: planning again writes nothing new and refuses a plan that differs from the
stored one (`--replace` on purpose, and never once the stored plan's replays have run:
plan into another oracle directory then); the run resumes job by job through the engine,
is skipped when done and proven. A pre-rule proof failure can be classified from
existing files; a final refusal under the new rule stays refused. A fresh set keeps its manifest once it has run and refuses fix sets that
need replays it lacks; the report is recomputed from whatever rows the runs hold. `sources.json` records
source runs during plan, run, fresh planning/running and reporting; reopening the same directory
recovers them without `--runs`. Older reports supply paths from `oracle.json.runs`.
A source record from another world is refused. A complete report cannot be replaced
with a refused/incomplete report, fewer rows, fewer draws or fewer bootstrap resamples;
use a new `--oracle-dir` for such a comparison. This guard precedes every report-file,
truth-table and step-record write.

From Python: `fleetlab.oracle.Oracle.open(root)` with `step_plan`, `step_run`,
`step_report`, `step_fresh_plan`, `step_fresh_run`, `step_fresh_value`, and `table()`
for the `OracleTable` itself (with `world.truth()` and `world.current()` it is all that
`contracts.oracle.value_fix_set` needs).

## What it does

**Plan** (`plan.py`). Per scoring episode, its faults split into the toggled ones (the
effective faults) and the excluded ones (decoys and inert confounder entries,
`Patterns.excluded_from_replay`). Three kinds of replay job, all "replay" from the reset:

- *contract*: every non-empty subset of the toggled faults removed, the excluded ones in
  force (`contracts.oracle.power_set_replays`), 2^n - 1 for n toggled faults; an episode
  with more than four is listed as above the cap and gets none;
- *confirmation*: each of those removal sets, and the empty one, once more with every
  non-empty subset of the episode's excluded faults removed too. So decoys and inert
  entries are confirmed by replay (gate 3), every action on them is a lookup, and the
  all-faults-removed row exists. With one excluded fault that doubles the episode's
  replays (w10: 65 and 65);
- *proof*: unmodified replays of every tenth scoring episode, at least 20, because the
  engine's chain sample only replays run jobs and a replay-only run has none.
  Their chains are compared with the as-run chains in the world's runs,
  step by step, and the verdict goes into the run's run.json (`proof.py`).

`check_jobs` validates a manifest without the simulator: every job round-trips as a
`Job`, is a replay from the reset of exactly the world's identity (bit-equal to the
manifest's), removes only the episode's own faults, has a unique key and fault list,
and its seeds are `seeds_for(world seed, robot, day, index)`. `estimate_from_spec` counts
from a spec before generation (robots numbered site by site; configuration shadowing
ignored, so an upper bound).

**Run** (`run.py`). The manifest goes to `fleetlab.engine.fleet.run` in one heavy job
under `tools/heavy-run` (the light command starts it and waits), with the world's batch
size and precision, ten workers by default (`--workers 0` fits them to the free memory
when the heavy job holds the lock), no video, the pinned convolution database under
`~/.cache/robots-probes/engine/miopen`; then the proof. Queue time and throughput go to
`steps.json`.

Before the engine starts, `run.py` writes `oracle-origin.json` in a new run directory:
the world hash, manifest path, manifest hash and job count. This independently binds
execution to the planned identities even if the first run is interrupted before its
proof. Resumption checks that origin (or the proven run record's existing provenance),
including the world hash. Cached files or a run record with no known origin are refused;
they cannot acquire provenance merely by being proven again. A manifest extension must
preserve the recorded prefix. Existing valid runs retain their old provenance format.

**Table** (`table.py`). Rows from `truth/job_outcomes.parquet` of every run that holds
them (the world's and the oracle's own): a "run" job is the episode under its full fault
list (its true verdict, before the data hook), a "replay" from the reset is the episode
minus `removed` plus `added`. Only runs with `chain_ok` true give rows; an as-run
identity must equal the world's; a replay's job key must be the key of the job rebuilt
from the world's identity with its `removed` and `added`, which ties the row to the
episode and to the change, but not to the identity (a job key does not hash it). So the
oracle's own runs record their provenance in `run.json` (`extra.oracle`: the world hash,
the manifest file and its SHA-256), and every row of such a run must be a job of that
manifest whose identity equals the world's bit for bit; a run made for another world
hash, or whose manifest changed or is gone, gives no rows. A world run's replays (the
miniature fleet's oracle in `w10-later`) require an independently recorded, matching
identity in its identities table. A replay with neither source is refused. Two rows
of one fault list that disagree use the same majority rule as the proofs, below;
an unresolved disagreement stops the build.

## Recorded-action replay rule

`run` and `all` execute every scoring attempt twice: the world's as-run execution
and one unmodified replay from reset. The `causes` step also checks every observed
attempt whose removal outcome it uses, including rows reused from older runs.
Experiment is off by default; `--window experiment` opts in. `--window observed`
can explicitly check the whole observed window. Existing proven unmodified samples
and oracle proofs satisfy coverage, without another execution for the proof sample.

The rule is `recorded-action-four-executions-v2`, implementing design entry 14.
Compare reset hashes, then each executed action, then its resulting state and camera
hashes. A first difference in a state, image, fixture or length refuses. A first
different executed action is eligible only with every recorded prefix identical.
There is no numeric tolerance. Two further executions are then required: a real
engine `JobMode.run` job in an oracle scratch run, and a `JobMode.replay` job.
Of the four executions there must be two of each kind, and exactly three must agree
in every recorded step and physical outcome. The majority stands. Two matching
run executions against two matching replay executions are a replay defect, even
for just one attempt. All other patterns and unavailable further executions refuse.
For duplicate changed rows, which are already two replay executions, the two further
jobs both use the engine's normal run path with the effective fault list in their
identities. They do not use another replay to stand in for that path.

If the replay was odd the baseline stays; if the as-run version was odd all oracle
comparisons and scorer lookups use the reproducible outcome. World records stay
unchanged. Evidence in `verification/comparisons.json` binds the world hash, jobs,
physical outcomes and complete chain SHA-256. Reads recompute classifications and
ceilings. Third and fourth executions have separate, resumable directories with
independently recorded origins. Old majority-v1 evidence can be resumed: identical
comparisons are reused; a disputed comparison needs the missing run-kind execution.
Legacy events are upgraded before saving coverage. Classified evidence is retained
even when the completed-window ceiling refuses the run. When a staged engine run
grows, cached outcomes/call counts are refreshed before its new proof is checked.
Old JSON reports remain readable. Reports without coverage say their windows were
not executed twice. The source world's own proof gate remains strict.

The ceiling uses a Poisson count. `MEASURED_EVENTS_UPPER_95` is the unrounded Garwood
upper endpoint for three events, `chi2.ppf(.975, 8)/2 = 8.767273069742323`.
`POLICY_CALL_RATE_BOUND = MEASURED_EVENTS_UPPER_95 / 26928 = 0.00032558203616095974`.
For n second-execution calls, `event_ceiling(n)` is the smallest k with
`Poisson(n * POLICY_CALL_RATE_BOUND).sf(k) < .001`. It is 5 for n=3,300 and 3,400,
and 6 for 3,500; zero calls have ceiling zero. Scope is the source run and window.
Only second executions contribute calls; originals and adaptive third/fourth
executions do not. Comparisons deduplicate the second execution's run/job key.
The ceiling is evaluated when the required window's comparisons are complete;
`ceiling_pending: true` labels a partial proof sample. Partial coverage cannot make
a measured truth file complete. `verified_episodes` lists baselines compared to
their actual as-run execution; two replay rows alone do not satisfy that coverage.
The independence/rate assumption is fixed, never fitted to the current events.
A systematic run/replay path difference refuses on its two-versus-two split.
An event cannot distinguish a disturbed call from a rare dependence on the other
batch rows computed beside an attempt. Both further engine jobs run alone in their
scratch runs; they do not reproduce the original batch composition or row. Such a
dependency can therefore produce an accepted three-to-one event. Only the event
ceiling bounds this case; the rule does not prove row independence.

Today's `run.json.extra.jobs_detail[].policy_calls` counts robot requests, not server
batches. A ledger missing a second execution's total recovers it from that attempt's
engine record if available. Otherwise verification and report refuse, naming the run
and job. Finished replay chains contain no request count; nominal action chunks,
executed steps and server-batch counts cannot reconstruct it. A killed engine session
can thus require a new verification directory: completed replays lacking saved totals
cannot establish a ceiling. The ceiling's unit never changes. Events count first
chain divergences, not later consequences of the changed input. Reported per-million
rates retain their exact 95% Clopper-Pearson intervals for comparison opportunities.

Changed replays are ordinarily executed once; disturbances there remain undetected
unless another row disagrees. Residual fields use the fixed upper rate and the
changed executions' recorded policy-call totals. Only a changed job with a missing
total uses its identity's maximum step budget. The JSON names those jobs and their
count; the summary states the number using that fallback and prints the recorded-call
component separately from the combined bound. Duplicated jobs count once at their
largest exposure. Assuming
every affected replay changes its outcome, 100 times expected affected replays /
scoring attempts bounds expected wrong-outcome points per net-outcome table;
Markov gives the conservative 95% bound, capped at 100 points. Interaction contrasts
use up to four outcomes, so their point bounds are four times the table bounds.

From the frozen compatible measurement checkout, the two resumption commands are:

```sh
uv run --locked python -m fleetlab.oracle verify ROOT --oracle-dir DIR --workers 0
uv run --locked python -m fleetlab.oracle causes ROOT --oracle-dir DIR --workers 0
```

Both submit their own heavy jobs. `verify --plan-only` counts missing seconds without
simulation. `causes --plan-only` writes the next measured plan before a job queues;
`causes --run-only` runs only that stated stage. Plain `causes` prints a standalone
stage, runs it, then plans any newly eligible joint stage outside the heavy queue,
prints that count and submits it. Another resume executes no completed job again.
Run `report ROOT --oracle-dir DIR` after completion to write the summary.
The observed planner requires completed scoring baseline verification. A finalized
older first pass that lacks its proof is reproved from its existing files by
`verify`; this does not load the policy or repeat its completed replay jobs.

At the design's measured 1,500 replays/hour, the known curve-only estimates are:

| World | Observed removal replays | Distinct observed attempts needing a second | GPU minutes before sample reuse |
| --- | ---: | ---: | ---: |
| w47 | 364 | 364 | 29.12 |
| w46 | 658 | 616 | 50.96 |

These are **estimates from curves**, not measured plans. The actual faults are
chosen only after scoring removals exist, and joint jobs only after standalone
observed results exist. For standard w47 (1,680 total attempts: 1,008 observed,
336 experiment, 336 scoring), the completed first-pass oracle's 34 scoring proofs
leave **at most 302** scoring seconds, or **12.08 GPU minutes**, before further world
sample reuse. If the curve-estimated five faults are the observed candidates,
observed adds 364 removal + at most 364 second executions: **728 jobs / 29.12 minutes**
before sample reuse. The combined conditional estimate is **1,030 jobs / 41.20 minutes**,
excluding queue waits, adjudications, any joint fallback, and faults whose realised
scoring harm falls short of their curve estimate. This is not an upper bound on the
measured plan. Exact `jobs_remaining` and `second_executions_remaining` are printed by
the planner once scoring results exist. A first pass already made with full scoring
verification has zero scoring seconds left. No GPU timing was measured this round.
The executed directory is `w47-opaque-v1`, using the opaque-id contract in the
first pass's `8e3771b`. The loader/resumption check requires that compatible
contract; this branch does not change the shared identifier contract. As the first
pass completed, read-only inspection of its 3,584 table rows found the following
counts. These supersede the earlier curve-based planning estimate for this run:

| w47 addition | New removal replays | Missing second executions | GPU minutes at 1,500/hour |
| --- | ---: | ---: | ---: |
| Scoring verification | 0 | 268 | 10.72 |
| Observed standalone stage | 644 | 517 | 46.44 |
| Combined | 644 | 785 | 57.16 |

The scoring count reuses 68 existing unmodified executions. The observed stage
would cover 574 distinct attempts, reusing 57 existing seconds. Its six candidates
come from the completed scoring table, rather than the five curve candidates.
**Observed membership remains provisional until full scoring baseline verification**;
a settled odd as-run outcome can change a gain. The queued planner requires that
verification and prints the definitive standalone count before submission. Any joint
fallback is planned only after those standalone observed outcomes exist. Counts here
were computed without changing the world's files or running a replay. Times are
estimates, excluding queue waits and disputed attempts' further executions.

For this executed world, set `ROOT` to
`~/.cache/robots-probes/worlds/w47-opaque-v1` and `DIR` to its existing `oracle`
directory in the two commands above. The frozen integration checkout must combine
the opaque-id contract with this oracle. An older first pass with `chain_ok: null`
is reproved from its completed files before missing seconds are planned.
For a window itself containing 1,680
attempts, 168 existing proofs leave at most 1,512 seconds, or 60.48 minutes at this
rate; the standard world's scoring window contains 336, not 1,680.

Current records provide executed actions, state hashes and raw camera hashes.
They do not contain the hooked observation tensors, returned chunks, call times,
or batch row order for every policy request. Consequently this rule establishes
recorded-chain majority and an action-first divergence; it does not prove which
GPU call was disturbed or check every later chunk against its own inputs. The
engine's forthcoming call record would add that attribution and a direct audit of
policy inputs and row independence. The existing world readiness/engine proof gate
continues to apply before the oracle can start.

### CPU evidence

`tests/test_oracle_verification.py` executes toy policy answers through the real
episode runner and writes the engine's ordinary recorded chains. These are
constructed CPU cases, not observations of a GPU disturbance:

| Case | Required result and regression |
| --- | --- |
| One answer value changed by `7e-5` in the second execution | Accepted, replay odd; `test_one_policy_answer_is_settled_and_the_majority_outcome_is_used[replay]`. |
| One answer value changed by `7e-5` in the as-run execution | Accepted, as-run odd; the marginal toy outcome flips, the gain uses the majority outcome, and every original file retains its SHA-256; the same test's `[run]` case. |
| Row-dependent answers with both actual scratch jobs alone | Accepted as one event; `test_row_dependence_can_be_an_event_when_both_scratch_jobs_run_alone`. Only the ceiling bounds rare batch dependence. |
| Changed state before any different executed action | Refused without executing a third; `test_changed_state_is_not_an_event`. |
| Too many individually settled events | Refused; `test_more_one_off_events_than_ceiling_refuse`. |
| Three pairwise different executions | Refused; `test_three_executions_pairwise_different_refuse`. |

The full toy-world resume test reuses its 12 scoring proof executions, adds 24
executions for the other two windows, then adds zero on another resume, checking
that existing replay files remain unchanged. Its deterministic CPU source is
marked proven with the world's sample omitted: this exercises oracle coverage
and resumption, rather than the lower engine's proof gate. A separate test executes
an actual
third-engine job and proves that resumption cannot invoke the engine again.
Additional regressions cover missing thirds, conflicting changed rows, preceding
camera/state/fixture differences, tampered saved evidence, pre-rule failed proofs,
and readable legacy reports. These tests run through `tools/heavy-run --light`;
they do not exercise a pretrained policy or the real GPU exception.

Round-six code check at `be1e637`: `make check` exited 0, with lint, formatting and
types passing across 382 files, and 2,415 unit tests passing, two skipped and 65
deselected (4,316.31 seconds). All 19 replay-verification tests passed.
`make check-sim` exited 0 at `13fec80`: 61 passed, 2,421 deselected
(239.27 seconds). Runtime code is identical between those two revisions; only
the marginal CPU toy test changed. Both checks used frozen detached checkouts
and the light queue. A fixed-revision focused run also passed all 117 selected
oracle/scorer tests. One earlier run overlapping a checkout advance was discarded.
The complete record, including log hashes, is
[replay-rule-verification.json](replay-rule-verification.json).

Round-seven final focused verification at `8cbe0f7` passed all **233** oracle/scorer
tests (254.04 seconds), including 28 replay-rule cases and 15 measured-cause cases.
Lint, formatting and types passed for 384 files at that revision. The single capped
`make check` at `a566842` exited **2**: 2,434 passed, two skipped, 65 deselected,
and three failures (3,543.04 seconds). The failures were a legacy report's old exit-0
expectation and two proof stubs missing new keyword arguments; the final focused run
passes their repaired tests. The full command was not repeated, respecting the
assignment's one-invocation limit. `make check-sim` exited **0** at `a566842`:
61 passed, 2,443 deselected (509.81 seconds). Later runtime edits are confined to
oracle ledger upgrade, cache refresh and saving refusal evidence; the new regression
cases are included in the final focused run. The exact record and log hashes are in
[replay-measured-verification.json](replay-measured-verification.json).

No GPU policy job, stored-world replay, smoke job or model session was run in round
seven. The CPU evidence proves the recorded-chain rule and staged resumption, not
real hardware disturbance attribution or pretrained-policy timing. The measured
truth of w47 remains uncomputed until the orchestrating session runs the two additions.

No GPU policy job, stored-world replay, own model session or smoke job was run
for round six. Actual hardware attribution and new-world GPU time were not
measured. Changes after the final code check are this documentation, the receipt
and module docstrings only. The lower source-engine readiness/proof gate remains
a precondition; universal exclusion of rare unrecorded dependencies is not proved.

### Output additions and changed meanings for replay settlement

The added top-level field is `replay_verification` (default empty for older reports).
It holds `rule`, `unit`, `policy_call_rate_bound`, `step_rate_bound`,
`ceiling_coverage`, `verified_episodes`, `coverage` per requested window, `windows` per source engine
run/window, and `residual`. Coverage gives `window`, `attempts`,
`attempts_executed_twice` and `missing`. Each run/window gives `compared_calls`,
`compared_steps`, `run_dir`, `window`, `attempts_executed_twice`, `events`, `event_ceiling`, `ceiling_pending`, `ceiling_unit`,
`rate_per_million_calls`, `rate_interval_95_per_million_calls`, and corresponding
`*_steps` fields. Events name `episode_id`, `step`, `odd_side`, `odd_run`, `third_run`, `fourth_run`,
`outcome_differs` and `reproducible_success`.

Residual fields: `changed_replays`, `expected_affected_replays_upper`,
`expected_wrong_outcome_points_upper_per_table`,
`wrong_outcome_points_95_upper_per_table`, and the explicit caveat `text`.
`residual_by_window` adds these same bounds separately for scoring and observed
removal tables, with each full-window denominator. The legacy `residual` is scoring.
No existing output field is renamed. `episodes.parquet.as_run_success` and all
baseline comparisons now mean the reproducible physical outcome if an as-run odd
execution was settled; the original world/export outcome is still unchanged.
`chain_ok` now accepts an action-first majority settlement below the fixed ceiling;
the proof's `identical` count still counts only bit-identical pairs.
The added run metadata key `run.json.extra.replay_refusal` records a terminal
classification failure; its run remains `chain_ok: false` on resumption.

**Gains** (`gains.py`). Each gain is a sum over scoring episodes of +1 (rescued), -1
(broken) or 0, computed as a vector, so the **bootstrap over scoring episodes** is a
matrix product: 1,000 resamples of the N episodes with replacement (seeded with the
world seed), the 2.5th and 97.5th percentiles of the net in points. Items:
`fault_effect` of every fault (decoys and inert entries too), by day for drifting
faults; `interaction` of the marked pairs and of every other pair that occurs together;
`achievable` at k = 1 to 6 (the contract's `achievable_gain`, the k = 3 and 6 scoring
budgets among them) and at 4 and 8 hours (the scorer's rule: the cheapest fixing action
per fault, every subset that fits), each maximised again in every resample; `all
faults`, every fault removed, with the per-episode verdicts in episodes.parquet. A
lookup the table lacks is reported missing with the replays it needs, never guessed.
The point values equal the contract's functions (tests).

**Chance** (`chance.py`, `valuation.py`). 200 random draws from the world's filled
catalogue, cut at k = 3, 6, 4 h and 8 h, valued through the table; the 5th to 95th
percentile of the net in points and as a share of the achievable gain. These statistics
require all draws to be valued. If any draw lacks replays or fails resolution, the
statistics are null and `missing` states the counts. `bounds` gives worst-case lower
and upper bounds on each statistic over **all** draws: a missing draw can break at most
the baseline successes or rescue at most the baseline failures (an unknown baseline
outcome permits either). These bounds describe missing-data uncertainty, not sampling
intervals. The legacy `skipped_*` counters remain, but no draw is silently excluded.
`fresh --chance` collects the joint replay needs of the original seeded draws at every
scoring budget (200 by default, configurable with `--draws`), deduplicates jobs and adds
proofs. `--plan-only` writes the counts before a heavy job is submitted. Once it has
run, recompute `report` using the same oracle directory and draw count; completion
never substitutes new draws. Draw selection and valuation are shared with the scorer;
its aggregation must also retain every draw. This branch makes no assertion about an
unmerged scoring implementation.

`chance[].exact` is the chance floor at count budgets (design section 10, entry 5).
It counts every distinct complete catalogue set, grouping identical resolutions and
weighting choices by binomial coefficients. A streaming traversal counts combinations
of resolution classes without storing every set. `EXACT_STATE_LIMIT = 100000` limits
class combinations deterministically, visiting smaller class subsets first so the
uncomputed tail is small when most entries change nothing. At the limit, unvisited sets keep their full
multiplicity as unknown mass; their worst/best bounds replace a refusal.
[capacity-receipt.json](capacity-receipt.json) records the measured CPU limit: 21
independent resolution classes plus 100 no-change entries were exact at three/six
fixes in 0.015/0.499 seconds; 40 classes were exact at three in 0.086 seconds and
bounded at six in 0.639 seconds. The six-fix population still contains every
9,381,724,380 set, with 79,989,630 uncomputed sets (0.853% of the population). This fixture checks the
known histogram against hypergeometric counting; it establishes capacity for this
topology, not a universal wall-clock guarantee for denser episode incidences.
Full six-fix enumeration above the measured limit is deliberately bounded to keep
reporting short, including a world with 40 classes. Cost orders can only be enumerated
for small catalogues, up to
`EXACT_PERMUTATION_LIMIT = 40320`; otherwise their exact level is unavailable with a
reason, and the seeded draws remain the level there.

The distribution contract, for each `chance[].exact`:

- `histogram`: sorted `[net, sets]` pairs; integer net is rescued minus broken over the
  scoring episodes, and sets is the multiplicity with that known value.
- `sets`: the whole population. The histogram counts **plus** `unvalued_sets` equal it.
- `unvalued_sets`: sets lacking replay values, plus the explicitly counted uncomputed
  remainder if enumeration reaches the cap. `net_low` and `net_high` bound any such
  set by the negative baseline successes and positive baseline failures. Unknown
  baseline outcomes permit either. Only added `config_change` faults on the engine's
  `fleetlab.faults.config_change.NO_MECHANISM` keys are omitted. That sole list is
  camera.scene.calibration, camera.wrist.calibration and software.version. Their
  ConfigChange hooks read no policy/environment input; a sentinel regression
  exercises every hook and fails if that guarantee changes. No zero measured gain
  establishes this rule. Original faults, including calibration updates and releases
  carrying mechanisms, stay in force. The table, draws, exact distribution, agent
  and baseline fix sets, and fresh plans all use this same engine guarantee. A minimal
  contract extension, `OracleTable.resolution`, lets the engine-backed table normalize
  a resolution before lookup. Thus a guaranteed no-change gain is zero even without
  an outcome row; no healthy outcome is invented, and policy-failure counts remain
  incomplete until their actual outcomes exist. The default contract resolver is unchanged.
- `mean_points`, `p5_points`, `p95_points`: populated when complete **or when that
  statistic is invariant to all unknown values before rounding**. Other statistics
  are null; `bounds` maps only those uncertain names to `[low, high]`, from putting every
  unvalued set at `net_low` or `net_high`. Monotonicity guarantees containment whatever
  the missing values are. Bounds round outward to 0.001 points; complete statistics
  retain nearest rounding and numpy's linear-percentile convention.
- `missing`: null for a complete or bounded distribution. Unresolvable actions and
  missing catalogue/episodes make the level unavailable with a reason. `counting_limited`
  states a reached enumeration cap; `uncomputed_sets` is its unknown remainder,
  already included in `unvalued_sets`. Such a bounded result has no `missing` refusal.

At cost budgets, `chance[].p95_points_interval` is a distribution-free confidence
interval for the **population lower 95th quantile**, distinct from the sample's linear
`p95_points`. `p95_interval_order_stats` gives one-based ranks; 0 and draws+1 mean the
known support ends. `p95_interval_coverage` is the binomial coverage (at least
`PERCENTILE_COVERAGE = 0.95`, equal tails, probability 0.95). At 200 draws the ranks are
184 and 197. Atoms only make coverage conservative. Missing draw values are placed at
their support ends before taking these ranks, adding missing-value uncertainty without
losing the coverage guarantee. Endpoints round outward to 0.001 points. No cost-budget
pass mark is read.

### Cited-example replay requests

`fresh ROOT --name examples --request score.json` accepts a whole scorer report with
`needs_fresh_replay`, or a JSON list of its records. `--request` is repeatable and can
be combined with existing `--fix-set`, `--catalogue`, `--oracle-items` and `--chance`.
Existing fix-set request formats remain valid. Example:

```json
{"needs_fresh_replay": [{"episode_id": "e000001", "fault_ids": [], "faults": [],
                         "needed_for": ["claim 5 cited example"]}]}
```

`episode_id` names **any as-run episode in this world's manifest, in any window**.
`faults` is the full effective list to leave in force; `[]` removes every fault,
including decoys and inert entries. The optional `fault_ids` must agree with `faults`;
`needed_for` and other scorer annotations are ignored. The world supplies the original
identity, reset, seeds and software record. A foreign episode or altered retained
fault is refused. An already recorded healthy as-run outcome needs no replay.
If a fault-free episode lacks its outcome, its required unmodified replay is stored
once among the proofs and supplies that healthy outcome as well.

The manifest, origin binding and unmodified proof chains use exactly the scoring
replay path. Fresh `plan.json.requests` preserves these requests; `values.json` appends
records with `label: "requested episode"`, `episode_id`, `faults`, `success` and
`missing`. A healthy failure is `success: false`, not missing. The scorer's lookup is
`Oracle.open(ROOT, oracle_dir=DIR, extra_runs=RUNS).table(all_windows=True).table.success(episode_id, ())`.
Default `table()` and reports retain the scoring window and its denominator. Keep
that default table for scoring gains; use the all-window table only for example lookups. If the
fresh run is in another oracle directory, name its `fresh/examples/run` with `--runs`
or `extra_runs` once; the oracle records these sources for later lookup. Replan/resume the same
named request after proof, then score again; do not report retraining precision while
any cited outcome is absent.

### Cause-list compatibility

The loader, joint-removal calculation and detectability check consume the generator's
explicit `fault_ids` per cause. They neither compute connected components of marked
pairs nor require their members to belong to a multi-fault cause. Single members are
looked up in fault effects; a multi-fault cause uses a joint removal, independent of
pair membership. Splitting a previous group therefore needs no oracle algorithm
change. Cause descriptions below that refer to old joined groups describe those stored
worlds only; their lists are not rewritten. A regression checks separate marked-pair
members and an unrelated joint cause.

**Detectability** (`detect.py`) retains the generator's list as an audit of the
curve forecast against scoring outcomes. The current CLI writes measured membership
from `causes.py` (entry 15) beside this audit; a forecast disagreement does not refuse
that report. Only missing required measurements remain incomplete. Direct legacy
`compute_report` callers that provide no measured object keep their previous audit
behavior. Data faults are measured on original logged/true differences, not physical
replay gain. See the measured truth-file schema below.
Patterns: a decoy and an inert entry are confirmed when their interval holds zero; an
interaction when its term is negative and its interval lies strictly below zero
(otherwise "not confirmed"); a drift when the per-episode effect rises from the first to the last scoring day.

The stored interaction term keeps the contract's sign: joint removal gain J minus
the single-removal gains A and B, T = J - A - B. A negative T means the faults harm
more together than separately. Their standalone harms, holding other faults fixed,
are J - B and J - A, summing to J + T. This is distinct from complementary fixes,
whose positive T must not confirm a super-additive fault interaction.
Truth can also mark `patterns.additive_pairs`, with the same fault-id-pair shape as
`patterns.interactions`. Both kinds get a term, interval and pattern verdict. An additive pair has only **contradicted**, **below the sum** and **not contradicted**
verdicts. Define excess E = -T (joint harm minus the summed standalone harms).
`ADDITIVE_RELATIVE_TOLERANCE = 0.10` defines the band ±10% of the nonnegative
measured summed harm, converted to scoring-window points. It is contradicted only
when E's entire 95% bootstrap interval lies strictly above that band; below the sum
when the entire interval lies strictly below it; otherwise not contradicted. Equality
at either boundary does not contradict. Additivity rests on the world's calibration,
not this window. The report says this once and never calls an additive pair confirmed.

Each additive pair reports E and its interval, the summed harm and its interval,
and `max_excess_not_excluded_points` (E's upper interval endpoint). The previous
observed-SE advisory was not an 80% power threshold; its legacy fields
`min_reliable_excess_points` and `reliable_excess_power` are now always null, and
`reliable_excess_method` says withdrawn. The observed `excess_se_points` remains a
spread diagnostic only. Never quote an old advisory value as a detection threshold.

The replacement `sufficient_excess_power_bound_points` is a **conservative sufficient
bound**, not a minimum detectable excess. It depends only on the designed carrier
count, total scoring count, bootstrap resamples and tolerance. At a true expected
excess at least this large, the actual contradicted verdict has probability at least
`POWER = 0.80`, **provided carrier counterfactual vectors are independent, bootstrap
draws are independent of them, and noncarriers contribute zero**. Carrier vectors
need not be identically distributed; their four binary outcomes can have any coupling.
The guarantee describes full counterfactual measurement: all required replay outcomes
must be available for the verdict. The design-only bound can be planned before that
measurement, but makes no power promise for an incomplete report. This is probability
over samples and bootstrap draws, not over a fixed replay table.
It does not guarantee power under dependence between attempts. The method, assumptions,
guaranteed probability and any design-only unavailability accompany the field.
Bootstrap disabled makes the bound unavailable; zero observed deviations do not.

The proof in `power.py` uses every possible binary counterfactual vector. Let X be
positive excess per carrier, H standalone harm, S the observed total excess, D its
expectation, and m the carrier count. X lies in [-2,2] and H+X is the joint removal
gain, at most 1. Thus the actual tolerance is at most 0.10*max(0,m-S).
Conditional on S>=c, a bootstrap sum has variance at most 4m and a centered lower
increment at most 2+2m/N. Bernstein's inequality bounds the chance that one sum is
at or below the guarded band. A binomial Chernoff bound makes the chance that enough
of the finite bootstrap draws fall there at most 0.05. The guard of 0.001 points
exceeds nearest-rounding error and preserves the report's strict inequality.
Independent carriers give Var(S)<=4m-D^2/m; Cantelli's inequality chooses D so
P(S<c)<=0.15. The union bound leaves power at least 1-0.15-0.05=0.80. The printed
bound rounds upward. Tests check the inequalities against exact binomial tails.

This broad guarantee is deliberately much more conservative than a Bernoulli model
with deviations only on the excess side and a zero band (the review's approximately
5.3 expected deviations at 28 carriers). It can exceed a monotone fault model's
attainable excess; it does not promise that these worlds can detect ordinary excesses
with 80% power. Calibration, not this bound, remains the evidence for additivity.
With missing replays or no joint episodes the pair is missing. Interaction confirmations
retain their strict negative-term/`high < 0` rule. No stored gain or interval changes.

With `--boot 0`, points remain available and intervals render as "unavailable";
interval-dependent confirmations remain unavailable. An incomplete healthy outcome
table makes `still_failing` and its clearer alias `policy_failures_on_healthy_robot`
null. `all_removed_known_failures` and `all_removed_missing` expose the count's bounds;
a missing fault-free baseline invalidates the policy count, but its removal gain
remains the known zero. A missing outcome on a faulted attempt invalidates the gain.
`report` and the report step of `all` exit **0** when complete, **1** for refused
rows, resolution errors, measured detectability failures or an unavailable required count floor, and **2**
for missing replay outcomes, including incomplete draws. Refusals take precedence.
An unavailable optional cost permutation population or an uncertain pair verdict
alone does not fail the command; bounded count floors are available results.
A newly failed replay proof raises `RunRefused`, so direct, heavy, all and fresh paths
stop with a nonzero exit before reporting or valuing the failed run.

**Fresh** (`fresh.py`). `value_fix_set` raises `MissingReplays` for an action that adds a
change and for an episode above the cap. `fresh` collects what a list of fix sets misses
(a ranked list counts as each action alone and its cut at every scoring budget, as the
scorer values it), optionally every catalogue entry alone and the oracle's own items,
turns each missing lookup into a replay with `remove` and `add`, adds proof replays,
runs them as one heavy job into `oracle/fresh/<name>/run` and values the fix sets again
(`values.json`). A change of the precision or of the image views is refused and listed
(the engine fixes the precision per run, and the pinned database holds the fleet's
images per call only).

**Files** (`report.py`). `oracle.json` (`OracleReport`), `summary.md`, `episodes.parquet`,
and `truth/oracle.parquet` (the contract's five columns: fault_effect, interaction,
achievable with items `k=1`..`k=6`, `4 h`, `8 h`, and fix_set `all faults`) in the
oracle's own run once it has run, or in a run named with `--parquet-into` (the
miniature fleet's chain names `w10-later`, where its replays are); never otherwise into
a world run.
No wall-clock time in these files, so the same table gives the same bytes.

## Fields added by the review repair

No output field is renamed. Existing gains, interval signs and estimate shapes stay
the same. Incomplete chance statistics and healthy-policy failure counts are null
instead of presenting a partial population as complete.

| Field | Shape and meaning |
| --- | --- |
| `OracleReport.marked_additive_pairs` | List of two-fault-id lists marked as additive in truth. |
| `OracleReport.policy_failures_on_healthy_robot` | Nullable episode count; alias of the retained `still_failing` measure. |
| `OracleReport.all_removed_known_failures` | Nullable count of known failures among healthy outcomes. |
| `OracleReport.all_removed_missing` | Count of scoring episodes without a healthy outcome. |
| Interaction `detail.standalone_harm_net` | Two signed episode counts, J - B and J - A. |
| Interaction `detail.summed_harm_net` | Their signed sum, J + T. |
| Interaction `detail.summed_harm_low`, `detail.summed_harm_high` | Nullable 95% bootstrap endpoints of that sum, in scoring-window points. |
| `chance[].exact` | Nullable exact/bounded population; `method`, `sets`, point statistics and unavailable `missing` reason are retained. |
| `chance[].exact.histogram`, `unvalued_sets`, `net_low`, `net_high`, `bounds` | Valued integer-net multiplicities, unknown set count, net support ends and outward-rounded statistic bounds. |
| `chance[].p95_points_interval`, `p95_interval_coverage`, `p95_interval_order_stats` | Cost-budget population-quantile endpoints, coverage fraction and one-based ranks. |
| Additive interaction `detail.excess_net`, `excess_points`, `excess_low`, `excess_high` | Signed excess harm and its 95% bootstrap interval; opposite sign to retained T. |
| Interaction `detail.excess_se_points` | Unrounded paired full-window standard error, in points. |
| Additive interaction `detail.summed_harm_points` | Signed summed standalone harm, in points; existing net and interval fields remain. |
| Additive interaction `detail.max_excess_not_excluded_points` | Upper excess interval endpoint, nullable without bootstrap. |
| Additive interaction `detail.min_reliable_excess_points`, `reliable_excess_power`, `reliable_excess_method` | Withdrawn advisory: the numeric fields are null; method records its withdrawal. |
| Additive interaction `detail.sufficient_excess_power_bound_points`, `excess_power_guarantee`, `excess_power_method`, `excess_power_assumptions`, `excess_power_unavailable` | Design-only sufficient excess bound, conditional power guarantee, method, assumptions and unavailable reason. |
| Additive interaction `detail.additive_tolerance_points` | Nonnegative half-width of the relative tolerance band. |
| `chance[].exact.counting_limited`, `uncomputed_sets` | Nullable cap reason and remaining population included in unvalued_sets. |
| `sources.json.world_hash`, `sources.json.runs` | World binding and absolute source-run paths recovered on reopen. |
| Fresh `plan.json.requests` | List of requested `episode_id` and effective `faults`. |
| Requested `values.json` record | `label`, `episode_id`, `faults`, nullable boolean `success`, nullable `missing` reason. |
| Fresh `plan.json`: `chance`, `chance_seed`, `chance_draws` | Whether seeded draw collection was requested; its seed and draw count (nullable if disabled). |
| `chance[].bounds` | Nullable map from statistic name to `[lower, upper]` over all draws. |

The bounds use `mean_points`, `p5_points`, `p95_points`, `above_zero`, and, when the
achievable gain is positive and known, `mean_share`, `p5_share`, `p95_share`.
The input adapter also adds `OracleWorld.additive_pairs`, a tuple of fault-id pairs;
the truth input field is `patterns.additive_pairs`.

## Measured cause truth file (design entry 15)

`causes.py` measures each planted effect fault's standalone scoring removal.
Below two fleet points it measures removal over every carrying observed attempt,
using **all observed attempts** as the denominator. Decoys and inert entries use
scoring only: at two points they can become causes. A decoy below two points stays
a decoy; one at or above two points without an identifiable target or 20 observed
carriers is `small`, on neither the cause nor decoy scoring list. `no_decoy_left`
counts only the decoys that remain. An identifiable target and at least 20 observed carriers, from
`fault_table.json`, remain necessary. Each marked interaction is considered alone;
there is no connected-group assumption. Only when neither member qualifies in either
window is joint removal measured, first scoring, then observed if scoring falls short.
Marked additive pairs have no joint-cause treatment. A qualifying member stays its own
cause, and the other member stays small.

Data faults use original true/visible outcomes without simulation. Their `points`
counts **misreported outcomes in either direction**, `(rescued + broken) * 100 / N`,
while `net` retains corrected successes minus false successes. With a single active
data fault the difference is attributable to it. Coactive data hooks cannot be
attributed individually from today's final-outcome records; their measurements remain
explicitly missing with that reason, never guessed. The current worlds have no such
ambiguous overlap. Missing true/visible records are also explicit. A world with no
remaining measured decoy is flagged; the report says decoy rejection is unavailable.
The evidence needed for overlapping data hooks is an open item; no
engine instrumentation was added in this round.

The scorer reads `DIR/measured_causes.json`, not the planner, curves or replay internals.
Its three-line form (JSON objects; schema version 1) is:

1. Root: `schema_version`, `world_id`, `world_hash`, `rule`, `threshold_points`, `min_observed_carriers`, `episodes_per_window`, `complete`, `problems`, `no_decoy_left`.
2. `faults[]` / `pairs[]`: `item`, `fault_ids`, `pair_kind`, `planted_role`, `curve_class`, `measured_class`, `data`, `identifiable`, `observed_carriers`, `qualified_by`, `windows.{scoring,observed}`; each window has `status`, `episodes`, `carrying`, `known`, `missing`, `rescued`, `broken`, `net`, `points`, `metric`, `reason`.
3. `causes[]`: `{unit, fault_ids, qualified_by}`, `cause_count`, `curve_causes[]` (unit strings). `measured_class` is `cause`, `small`, `decoy`, `inert`, or null pending required measurements; the curve class is retained independently.

Window status is `measured`, `missing`, `not_needed`, or `not_applicable`.
A null point with `not_needed` is an unexecuted window that cannot change the current
qualification; `missing` is required data still absent. The file's cause list is
provisional until `complete: true`; consumers must check this. `oracle.json.measured_causes`
embeds the same object, and the summary shows both lists side by side. The legacy
`detectability` rows remain a curve-versus-scoring audit; their disagreement no longer
refuses the measured report. Missing required measurements produce exit 2.
`complete` also requires double-execution coverage of every scoring baseline and
every observed baseline whose removal measurement is used. A completed truth file
is never replaced with an incomplete one.

Observed plans are in `causes/plan.json`, `causes/jobs.jsonl`, with results in
`causes/run/`. Existing removal rows and unmodified proofs in any proven source are
reused, including external proof references rather than repeat simulation. The
manifest grows by appending new jobs, with its prior origin prefix preserved. The
plan is bound to the world and every job must be its observed identity from reset.
An interrupted stage resumes with `causes --run-only`; a completed stage can be
followed by `causes --plan-only` for the next joint decision. `causes --estimate`
prints curve-only counts and cannot submit a replay before scoring measurements exist.

`tests/test_oracle_causes.py` checks thresholds and whole-window counts, a curve
misclassification, observed qualification, both joint stages, additive exclusions,
identifiable/20-carrier eligibility, data misreports, decoy promotion/inert retention,
missing-scoring planning refusal, and real CPU engine resumption with/without prior
samples. It proves that only selected observed attempts are doubled, experiment is
absent by default, and previously recorded chains remain byte-identical. It does not
measure LIBERO/X-VLA trajectories or certify GPU behavior.

## Changed meanings for readers of oracle.json

No field is renamed. Consumers must update their models for the added fields above;
old code with extra fields forbidden cannot read new files, while this code reads old
files. Since the review repairs:

- `chance[]` statistics are null if any draw lacks a value; previously they described
  only valued draws. `missing` gives counts. Legacy `skipped_*` draws are bounded,
  not skipped. Count-budget `exact` is the floor, with histograms and missing-mass
  bounds instead of refusal for absent replay values or enumeration truncation.
  No-mechanism additions now use the same canonical table lookup everywhere,
  so their replay requirement disappears. Statistics invariant to unknown sets
  retain their scalar value; `bounds` omits those names. Cost-budget
  `p95_points_interval` describes
  population-quantile sampling uncertainty, distinct from the sample percentile.
- `still_failing` and its alias describe healthy-policy failures, not catalogue
  unfixability; both are null for missing healthy outcomes. `all_faults.net` is null
  for missing outcomes on faulted attempts. Fault-free attempts have a known zero
  removal gain even when their baseline outcome and the policy count are unknown.
- `interactions[].detail.marked` includes additive pairs. Marked pairs without joint
  episodes now get a row with null net and `missing: "no episode carries both"`.
- `patterns[]` includes `additive_pair`, now with contradicted, below the sum or
  not contradicted verdicts for valued terms. The positive excess-harm sign in the new
  numeric detail fields is opposite to the retained interaction term. An interaction's
  former weak verdict is
  now not confirmed. Decoy, inert and confounder verdicts can be missing when
  bootstrap is disabled. No term sign or existing gain/interval changes.
- The observed-SE power advisory is withdrawn: legacy numeric fields are null.
  `sufficient_excess_power_bound_points` is a sufficient bound under the explicitly
  stated independence assumptions, not a minimum or a physical calibration.
- `detectability[]` remains a curve audit; under the measured rule its disagreement
  does not invalidate a report. New `measured_causes` supplies realised membership
  and retains the old curve list beside it. Missing required measurements are
  incomplete (exit 2). Unavailable required exact floors still exit 1.
- `replay_verification.rule` is now four-executions-v2. Its ceiling is Poisson at
  the revised upper rate, and call/step exposure counts second executions only.
  Events add `fourth_run`; gain/baseline fields keep their canonical-outcome meaning.
- `run_here` raises `RunRefused` on a failed proof. Nonempty run directories without
  early origin or recorded provenance are refused; old interrupted runs need a new
  directory. `report` now fails on refused rows and incomplete measured results.

## Unified valuation: current reports (2026-10-09)

The oracle of record for each stored development world is now
`~/.cache/robots-probes/worlds/{w41,w42,w43}/oracle-unified-20261009`.
In particular w41/oracle stays untouched; its supplementary chance replays remain
in oracle-review-20261009/fresh/chance/run. All prior directories are historical.
No new robot replay was run in this round. Every existing gain and bootstrap interval
matches oracle-distributions-20261009, and each new report reproduces from its own
recorded sources without `--runs`. Reports exit 0 for all three worlds and took
5.537, 9.172 and 5.678 seconds respectively (single BLAS thread).

The full before/after tables, source hashes and replay plan counts are in
[unified-comparison.json](unified-comparison.json). In the tables below, each triple
is **mean / 5th / 95th percentile**, in percentage points over 336 scoring attempts;
brackets bound the missing values of the old distribution. Each new seeded table
contains all 200 draws at every budget. Cost population-quantile confidence intervals
remain separate fields in the linked table.

| World | Budget | Before: seeded triple | After: seeded triple |
| --- | --- | --- | --- |
| w41 | k = 3 | 0.353 / 0.000 / 0.000 | 0.353 / 0.000 / 0.000 |
| w41 | k = 6 | 0.807 / 0.000 / 8.333 | 0.807 / 0.000 / 8.333 |
| w41 | 4 h | 1.720 / 0.000 / 8.333 | 1.720 / 0.000 / 8.333 |
| w41 | 8 h | 2.335 / 0.000 / 8.943 | 2.335 / 0.000 / 8.943 |
| w42 | k = 3 | 0.424 / 0.000 / 2.470 | 0.424 / 0.000 / 2.470 |
| w42 | k = 6 | [0.369, 0.869] / [0.000, 0.000] / [4.182, 4.464] | 0.664 / 0.000 / 4.182 |
| w42 | 4 h | [-0.357, 1.643] / [-0.298, -0.298] / [4.464, 4.866] | 0.798 / -0.298 / 4.464 |
| w42 | 8 h | [-0.320, 3.180] / [-0.595, -0.312] / [7.440, 15.774] | 1.750 / -0.312 / 7.440 |
| w43 | k = 3 | [-0.021, 0.479] / [0.000, 0.000] / [3.274, 3.289] | 0.286 / 0.000 / 3.274 |
| w43 | k = 6 | [0.060, 1.060] / [-0.298, -0.298] / [5.357, 5.655] | 0.673 / -0.298 / 5.357 |
| w43 | 4 h | [-1.155, 3.845] / [-3.943, -0.595] / [7.738, 16.324] | 2.104 / -0.595 / 8.348 |
| w43 | 8 h | [-5.210, 6.790] / [-61.905, -0.595] / [8.929, 38.095] | 2.551 / -0.595 / 9.226 |

| World | Count budget | Before: exact triple | After: exact triple |
| --- | --- | --- | --- |
| w41 | k = 3 | [0.256, 0.403] / [0.000, 0.000] / [3.273, 3.274] | 0.348 / 0.000 / 3.274 |
| w41 | k = 6 | [0.253, 0.958] / [0.000, 0.000] / [5.654, 5.953] | 0.695 / 0.000 / 5.655 |
| w42 | k = 3 | [0.311, 0.391] / [0.000, 0.000] / [2.380, 2.381] | 0.358 / 0.000 / 2.381 |
| w42 | k = 6 | [0.484, 0.875] / [-0.298, 0.000] / [4.464, 4.465] | 0.715 / 0.000 / 4.464 |
| w43 | k = 3 | [0.244, 0.347] / [0.000, 0.000] / [3.273, 3.274] | 0.309 / 0.000 / 3.274 |
| w43 | k = 6 | [0.309, 0.800] / [-0.298, 0.000] / [4.464, 5.060] | 0.619 / 0.000 / 4.464 |

All new exact count histograms are complete: unknown mass is zero. Their populations
remain 22,369,536 three-fix sets and 24,582,576,566,016 six-fix sets. Thus count-budget
95th-percentile floors are numbers again: w41 3.274/5.655, w42 2.381/4.464, w43
3.274/4.464 points at three/six fixes. No possible set was discarded.

Chance plans against the stored sources contained 0/196/476 counterfactual replays
and 0/23/21 proof jobs for w41/w42/w43 before; all are now zero. Removing w41's
supplementary chance source gave 924 replays and 23 proofs before, zero afterwards.
The five surviving mutation families are caught by assertion regressions, including
both seed-error routes: [mutation-receipt-final.json](mutation-receipt-final.json)
records six mutated-code runs, each exit 1, with its unchanged baseline exit 0.
The hook test, signed pair tests, missing-mass completions, source reopening and
byte-preserving downgrade refusals are constructed cases. That round's normal power benchmark
has since been withdrawn; native simulator chains were not rerun in this round. Smoke is left
to the orchestrating session at merge.

Final source verification at `4092e6f`: `make check` exited **0**, with 1,176
passed, two skipped and 62 simulator/model cases deselected (331.57 seconds).
Lint, formatting and types also passed. An earlier full check exposed two miniature
fixture comparisons to the old missing-draw semantics; their updated independent
all-draw valuation and CLI expectations pass. The missing-baseline no-mechanism
regression fails against `b32a4dc` with `MissingReplays` and passes after the minimal
resolution callback. Final source reproduces each stored JSON report and summary
byte for byte, reopening its directory without `--runs`. The actual CLI at
`b32a4dc` reproduced JSON, summary and episode parquet for all three worlds at exit 0;
its reduced-draw/bootstrap attempts exited 1 and preserved every directory file.
No CLI or replay execution code changed in `4092e6f`. All six mutation runs report
baseline exit 0 and mutated exit 1. The 38,408-file original inventory and recorded
historical/current report hashes remain unchanged. No simulator source changed;
`make check-sim` was not repeated and smoke was not queued in this round, as requested.
The complete verification record is [verification-receipt-final.json](verification-receipt-final.json).

These are plans only. The old marked interactions keep their not confirmed verdicts;
if those pairs were marked additive they would be not contradicted (their intervals
include the tolerance band). No old truth mark is changed. Neither w44 nor w45 was opened.

## Power and incomplete-status follow-up (2026-10-10)

The observed-spread advisory in `726c85b` is withdrawn. The replacement has a
conditional analytic guarantee for the report's own tolerance, finite bootstrap,
linear percentiles, rounding and strict verdict. It is deliberately a sufficient
bound across independent binary-counterfactual laws, not a minimum. The assumptions
and derivation are stated above; dependence between attempts is not covered.

`uv run --locked python -m fleetlab.oracle.power_calibration --output FILE`
reproduces [power-bound-calibration.json](power-bound-calibration.json). The CPU study
ran as a light job from detached source `73d3843`, without replays or a GPU model.
Each row below used 1,000 independent synthetic windows per law and 1,000 bootstrap
resamples over 336 scoring attempts. The two laws use maximal +/-2 contrast spread
and independent counterfactual outcomes. Their true expected excess is set to the
printed bound; it is the sum of binary contrasts, not a count of deviating episodes.
All old marked pairs have 28 carriers and therefore the identical design bound,
regardless of whether zero, one or three deviations were observed.

| Design | Carriers | Sufficient bound, fleet points | Expected excess net | Contradicted trials: maximal spread / independent outcomes |
| --- | ---: | ---: | ---: | ---: |
| w42 f01+f02 | 28 | 14.099 | 47.37264 | 1,000/1,000 / 1,000/1,000 |
| w42 f01+f03 | 28 | 14.099 | 47.37264 | 1,000/1,000 / 1,000/1,000 |
| w43 f01+f02 | 28 | 14.099 | 47.37264 | 1,000/1,000 / 1,000/1,000 |
| Larger carrier window | 112 | 33.652 | 113.07072 | 1,000/1,000 / 1,000/1,000 |

These Monte Carlo results illustrate the conservativeness; they do not prove
100% power. The analytic lower guarantee is 80% under the stated independence
assumptions. No physical power or calibration was newly measured. The earlier
synthetic calibration study of the verdict itself remains valid; the verdict did
not change. Old stored gains, intervals, chance levels and pair verdicts are untouched.
The retained observed-SE field remains a spread diagnostic. Old numeric advisory
fields are null in new reports, and the changed-meaning guide lists their replacements.

The actual `report` CLI on w41 with only as-run rows, no planned replay source,
`--boot 0 --no-truth-table` and a new scratch `--oracle-dir` exits **2**. All seven
fault gains and 196 healthy outcomes are missing; all detectability problems now
say incomplete. The case fails against `726c85b` with status 1. A missing gain is
not a measured detectability failure. Refused rows, unresolvable actions and unavailable
required count floors still exit 1. Complete reports still exit 0.

Four targeted mutated-code runs fail their intended assertions while all unmodified
baselines pass: contradicting a stored no-mechanism row, dropping an original fault,
dropping a fault-free request outside the proof sample, and treating an unavailable
required exact floor as incomplete. The unsampled-request case plans 41 requested
outcomes, retaining every unmodified replay exactly once rather than only the proof
sample. These are manifest and lookup tests; no replay was executed. The receipt is
[power-status-mutations.json](power-status-mutations.json). Final `make check` at `47f95ee` ran **once** and exited **0**: 1,189 passed, two
skipped, 62 deselected in 315.50 seconds. Lint, format and types passed. The final
receipt is [power-status-verification.json](power-status-verification.json). The original
38,408-file inventory is unchanged. Simulator paths were not changed; check-sim was
not repeated. No smoke job was queued, and neither w44 nor w45 was opened.


The following review-round sections preserve historical measurements and verdict
names at those commits; the rules and oracle of record above supersede them.

## Second review: revised verdicts and exact chance populations

The second-round reports are in `~/.cache/robots-probes/worlds/{w41,w42,w43}/oracle-followup-20261009`.
For w41 this was the report oracle of record, superseded by the distribution round below. Its complete
200-draw level uses the replay record `oracle-review-20261009/fresh/chance/run`, plus
the original oracle/catalogue and world runs, all explicitly listed in its `runs`
field. `w41/oracle` and `w41/oracle-review-20261009` are preserved historical records;
neither is overwritten. To reproduce, supply those recorded run directories with
`--runs` and write to a new `--oracle-dir` with `--no-truth-table`.

No new robot replays were needed in this follow-up. All existing gains and bootstrap
intervals equal the preceding reviewed reports; only interaction wording/verdicts,
added summed-harm intervals, and the optional exact chance population change. The
JSON hashes of all six old/new reports in the preceding comparison still match.
Report exit statuses are w41 **0**, w42 **1**, w43 **1**: the last two deliberately
signal their still-incomplete seeded draws (counts below), despite complete exact
count-budget floors. Details are in [followup-results.json](followup-results.json).

Every world has 513 complete catalogue entries. Exact populations contain 22,369,536
three-entry sets and 24,582,576,566,016 six-entry sets. Points in this table are
percentage points over each world's 336 scoring attempts; all exact 5th percentiles
are zero. Hour-budget exact permutation enumeration exceeds the stated cap, so only
seeded draws are reported there. No budget is assigned an exact result from a subset
of available values.

| World | Three fixes: exact mean / 95th percentile | Six fixes: exact mean / 95th percentile | Missing seeded draws at k=3 / k=6 / 4h / 8h (of 200 each) |
| --- | ---: | ---: | --- |
| w41 | 0.348 / 3.274 | 0.695 / 5.655 | 0 / 0 / 0 / 0 |
| w42 | 0.358 / 2.381 | 0.715 / 4.464 | 0 / 1 / 4 / 7 |
| w43 | 0.309 / 3.274 | 0.619 / 4.464 | 1 / 2 / 10 / 24 |

Thus w41's three-fix floor is 3.274 points (3.3 to one decimal), while its unchanged
200 draws have a 95th percentile of zero. The real `fresh --chance --plan-only` CLI
collected 800 fix sets (200 draws at four budgets) against w41's completed table:
zero outstanding jobs, proofs, resolution errors or refusals. Synthetic tests check
that missing joint rows become exactly the necessary jobs. No new heavy job was
submitted; the new planning option has not been exercised on the new worlds.

There is no marked pair in w41. The two w42 pairs and the w43 pair now read **not
confirmed** as interactions; their negative terms' intervals include zero (unchanged
endpoints: w42 f01+f02 -2.083 to 0.000, w42 f01+f03 -0.900 to 0.000, w43 f01+f02
-1.786 to 1.190 scoring-window points). If marked additive, all three are
**indeterminate** under that round's rule. This is a hypothetical interpretation of the
old pairs, not a change to their truth marks. Neither new world was opened.

### Synthetic power study under the current rule

`uv run --locked python -m fleetlab.oracle.pair_calibration --output FILE` reproduces
[additive-calibration-final.json](additive-calibration-final.json): seed 20261009,
300 synthetic worlds per cell, 1,000 bootstrap resamples over 336 scoring attempts;
28 or 112 carry the pair. Healthy success probability is 0.90 and standalone harms
are 5.0 and 12.7 points among carriers. The four counterfactuals use either a shared
uniform difficulty or independent difficulties. This is statistical simulation,
not new robot replay evidence. Excess below is points among carriers, not fleet points.
Each percentage cell is shared/independent difficulty; Monte Carlo 95% error is at
most about six points per rate with 300 worlds. These outcome-model assumptions do
not establish a universal power guarantee or confirm additivity.

| Carriers | True excess | Contradicted % | Below the sum % | Not contradicted % |
| ---: | ---: | ---: | ---: | ---: |
| 28 | 0 | 1.3 / 2.7 | 1.0 / 2.0 | 97.7 / 95.3 |
| 28 | 10 | 20.3 / 6.7 | 0.0 / 0.3 | 79.7 / 93.0 |
| 28 | 20 | 53.0 / 22.3 | 0.0 / 0.0 | 47.0 / 77.7 |
| 112 | 0 | 1.0 / 2.7 | 0.3 / 1.3 | 98.7 / 96.0 |
| 112 | 10 | 56.0 / 18.7 | 0.0 / 0.0 | 44.0 / 81.3 |
| 112 | 20 | 97.0 / 67.7 | 0.0 / 0.0 | 3.0 / 32.3 |

## Review repair measurements (2026-10-09)

The three stored development worlds were reported again at `0fae966`, into
`<root>/oracle-review-20261009`. Their original primary `plan.json` and `jobs.jsonl`
were copied into those new directories, preserving the estimates for the plans that
actually ran. The original oracle run and the original `fresh/catalogue/run` were
read through `--runs`; `--no-truth-table` kept report writing inside the new
directories. All three report commands and manifest checks exited 0, with no report
problems. The complete numeric comparison is [review-comparison.md](review-comparison.md),
with a machine-readable version and source-report hashes in
[review-comparison.json](review-comparison.json).

For D1, the original 200 seeded draws at each budget lacked values for 1, 2, 14 and
28 draws at k = 3, k = 6, 4 hours and 8 hours respectively. Their 45 fix sets needed
924 distinct counterfactual replays. Those and 23 unmodified proof replays ran from
the frozen detached checkout at `2beb1ea`: 947 jobs in 1,291.6 seconds, with eight
workers, excluding the outer heavy-queue wait. The command exited 0 and all 23 sampled
chains were identical to the as-run chains. Every one of the 45 fix sets then valued
without a missing replay or resolution error. The table grew from 2,156 to 3,080
distinct rows; the report now values all 200 draws at each budget.

These are gains in percentage points over D1's 336 scoring episodes. Old means used
only the draws with replays; new means and percentiles use all 200 draws.

| Budget | Old subset mean | New all-draw mean | New 5th to 95th percentile |
| --- | ---: | ---: | ---: |
| k = 3 | 0.313 | 0.353 | 0.000 to 0.000 |
| k = 6 | 0.744 | 0.807 | 0.000 to 8.333 |
| 4 hours | 1.470 | 1.720 | 0.000 to 8.333 |
| 8 hours | 2.059 | 2.335 | 0.000 to 8.943 |

D2 and D3 use only stored replays, as assigned. Out of 200 draws at each of the same
four budgets, D2 still lacks 0, 1, 4 and 7 values, and D3 lacks 1, 2, 10 and 24.
Their incomplete statistics are null with explicit missing counts and bounds over
all draws. D2's complete k = 3 mean remains 0.424 points. Every existing fault gain,
interaction term, achievable gain and all-faults gain, including their intervals,
matches its stored report. Healthy-policy failures remain 6 of 336 episodes in D1,
8 of 336 in D2 and 11 of 336 in D3, with no missing healthy outcomes. D2's two marked
interaction verdicts and D3's one change from "not confirmed" to "weak"; their
negative terms have intervals containing zero. The old truth marks are retained.

`make smoke-world` at `2beb1ea` exited 0: all 16 chains were identical to the stored
reference, with no differences, missing chains or software changes. No reference
was updated. An inventory of 38,408 original files under these worlds' `files`,
`runs` and `oracle` directories found no size or modification-time changes.

Confidence is high in the repaired paths covered by the regressions and sampled
physical proofs. These measurements do not establish the additive-pair verdicts on
the new worlds, empirically calibrate the ten-percent tolerance, compare every new
counterfactual chain with a second execution, or force a failed proof or interrupted
GPU run. The failure guards and missing-data cases are tested on constructed cases.

## Counts and timings (before the review)

These sections preserve the original measurements. Chance figures computed from
fewer than all draws describe only the valued subset and are superseded by the
2026-10-09 repair's reports. The linked review comparison records every numeric change.

### The miniature fleet, w10 (mini-v1 fixtures)

From `~/.cache/robots-fixtures/mini-v1` (read only, `--oracle-dir` elsewhere):

| | |
|---|---|
| Scoring episodes | 104 (day 3); 39 without a toggled fault, 65 with one |
| Planned | **130 replays** (65 contract, 65 confirmation, the decoy f03 removed), the integration's count; 1.25 per scoring episode; 21 proof replays on top when run on their own; none above the cap |
| `estimate_from_spec` | 130, the same (no shadowing in w10) |
| Job keys | the 130 planned keys are exactly the 130 replays in `runs/w10-later` |
| Table | 234 distinct rows (104 as run, 130 replays; the later run's chain-sample replays agree) |
| Plan, report | 0.7 s; 2.2 s (1,000 resamples, 200 draws) |

**The replays again, on the GPU, in a run of their own** (2026-10-08, 05:17 to 05:22,
from a frozen checkout at 0d3e5b4: `python -m fleetlab.oracle all
~/.cache/robots-probes/mini/w10-v2 --oracle-dir ~/.cache/robots-probes/oracle/w10
--workers 4`; the world's root read only). 151 jobs (130 replays, 21 proof replays) in
292 s, 1,868 per hour end to end and 2,049 steady with **four workers**: the engine's
memory guard refused ten at 24 GiB available (another project's process held 19 GiB).
The proof held, 21 of 21 chains identical to the as-run chains in `w10-later`; **all
130 replays are bit-identical (state, both images, executed actions at every step) to
the integration's replays of the same jobs in `w10-later`**, made about six hours
earlier in another checkout with ten workers, and the table built from both runs (and
the world's as-run episodes) found no disagreement: 234 distinct rows, the same gains
to the digit.

**Fresh replays on the GPU** (06:10 to 06:15 after 48 minutes in the heavy queue,
`python -m fleetlab.oracle fresh ... --name delay8 --fix-set w10-delay8.json
--catalogue --workers 4`): the fix set `set control.action_delay_steps = 8 at s2` (a
non-nominal value: it removes the 17-step delay f02 and adds a delay of 8) raised
`MissingReplays` for the 52 scoring episodes of s2; `--catalogue` added the three
catalogue entries that add a change (`set camera.scene.calibration` to the fleet's value
at the fleet, s2 and r08, because r08 holds a benign recalibration record: 13 episodes
each, with three different added fault ids). 91 replays with `add`, 26 proof replays,
117 jobs in 324 s (1,304 per hour, four workers), proof 26 of 26. Valued again: delay 8
at s2 rescues 41 and breaks none, +39.42 points, exactly what removing the delay gains
(eight steps late does nothing here, as the curves say up to 14); the three calibration
sets are 0 (the key moves nothing). After it, every one of the 334 catalogue entries
with a replay is valued by lookup (net non-zero for 11 of them), and the table holds
325 rows. The random draws then value more draws (k = 3: 200 of 200, +1.06 points, +0.00
to +6.83; k = 6: 199; 4 h: 190; 8 h: 188): a draw that combines a change-adding entry
with a fault's fix still needs its own fresh replay.

| Item | Rescued | Broken | Net | Points | 95% bootstrap interval |
|---|---|---|---|---|---|
| f01 camera, r04 | 8 | 1 | 7 | +6.73 | +1.92 to +12.50 |
| f02 delay, s2 | 41 | 0 | 41 | +39.42 | +29.81 to +49.04 |
| f03 decoy, s1 | 0 | 1 | -1 | -0.96 | -2.88 to +0.00 (confirmed: zero within noise) |
| interaction f01, f03 | | | +1 | | +0.00 to +2.88 |
| achievable k = 1 | 41 | 0 | 41 | +39.42 | +29.81 to +49.04 |
| achievable k = 2 to 6, 4 h, 8 h | 49 | 1 | 48 | +46.15 | +36.54 to +55.77 |
| every fault removed | 49 | 1 | 48 | +46.15 | 2 of 104 episodes still fail |

Random fixes (200 draws): k = 3 +1.09 points (+0.00 to +7.40), k = 6 +1.86 (+0.00 to
+9.62), 4 h +5.71 (-0.96 to +39.42), 8 h +7.59 (-0.96 to +39.42), with 6, 12, 31 and 41
draws skipped for a fresh replay: the scorer's numbers in the integration README.
Detectability: f01 and f02 "ok" (expected 5.1 and 22.5 points from the calibrated
curves, measured 6.7 and 39.4). The two analyses' fix sets, cut at the four budgets and
valued through this table, reproduce the scorer's gains (net 48 at every budget for
both) and every ranked action's true gain (`tests/test_oracle.py`).

### The development worlds

World files of stage3/worlds at 2f1c613 (the compositions after the libero_object and
interaction calibration), 336 scoring episodes each (days 8 and 9), none above the cap,
34 proof replays, full confirmation. Each ran as `python -m fleetlab.oracle all
~/.cache/robots-probes/worlds/<wid> --workers 0` from a frozen checkout at f05679f on
2026-10-08; the reports were recomputed at 7ab768a. Results per world are in
`<root>/oracle/summary.md`.

| World | Scoring episodes by toggled faults (0/1/2) | Excluded faults | Contract | Confirmation | Jobs | Run | Proof | Rows |
|---|---|---|---|---|---|---|---|---|
| D1 (w41) | 140 / 140 / 56 | decoy f01 | 308 | 336 | 678 | 720 s, ten workers, 3,399 per hour, after 1 h 56 min in the queue | 34 of 34 | 980 |
| D2 (w42) | 28 / 140 / 168 | inert f06, decoys f07, f08 | 644 | 1,456 | 2,134 | 2,256 s, ten workers, 3,408 per hour, after 2 h 11 min in the queue | 34 of 34 | 2,436 |
| D3 (w43) | 28 / 196 / 112 | inert f05, decoy f06 | 532 | 672 | 1,238 | 1,364 s, ten workers, 3,271 per hour, after 1 h 27 min in the queue | 34 of 34 | 1,540 |

D2's three excluded faults sit on robots with effective faults (an s3 robot carries two
toggled and two excluded faults: 3 contract and 12 confirmation replays), so full
confirmation costs 2.3 times its contract replays; `--confirm single` would make it 1,238
jobs (560 confirmation replays), `--confirm none` 678. All three fit the design's row of
about 3,000 per small world with full confirmation.

**D1 (w41)**, seven faults (few, clear, strong; one decoy):

| Fault | Kind, scope | Episodes | Rescued / broken / net | Points | 95% interval | Expected | Verdict |
|---|---|---|---|---|---|---|---|
| f01 | decoy: illumination_change, s1 | 112 | 4 / 5 / -1 | -0.30 | -2.08 to +1.49 | 0 | decoy, confirmed |
| f02 | action_latency 20, r12 | 28 | 20 / 0 / 20 | +5.95 | +3.57 to +8.63 | 7.00 | ok |
| f03 | action_latency 20, r07 | 28 | 28 / 0 / 28 | +8.33 | +5.36 to +11.31 | 7.62 | ok |
| f04 | camera_pose_offset 9, r11 | 28 | 18 / 0 / 18 | +5.36 | +2.98 to +7.74 | 6.36 | ok |
| f05 | object_batch_variant 6, s1 | 112 | 13 / 2 / 11 | +3.27 | +1.19 to +5.66 | 2.91 | ok |
| f06 | camera_pose_offset 9, r02 | 28 | 11 / 1 / 10 | +2.98 | +0.89 to +5.06 | 4.92 | ok |
| f07 | control_mode_change, r04 | 28 | 19 / 0 / 19 | +5.66 | +3.27 to +8.33 | 6.94 | ok |

All six detectable causes show at least two points; the decoy is zero within noise.
Achievable gain: k = 1 +8.33 points (f03), k = 3 +19.94 (f02, f03, f07; +17.56 to +24.70),
k = 6 and 8 h +36.90 (all six), 4 h +28.27 (five fixes in 1.3 h). Every fault removed: 128
rescued, 2 broken, +37.50; 6 of 336 episodes still fail with every fault removed. Random
fixes before the catalogue's fresh replays: k = 3 +0.35 points (0 to 0), k = 6 +0.84 (0 to
+8.33), 4 h +1.15, 8 h +1.82, with 24, 42, 87 and 120 of 200 draws skipped: 22 catalogue
entries add a change in this world (`set software.version` at 16 scopes and the two
calibration keys at 3 each, where a robot's own record differs from the fleet's value).
Their fresh replays (`python -m fleetlab.oracle fresh <root> --name catalogue --catalogue
--workers 0`, 18:22 to 18:44): 1,176 replays with `add` and 34 proof replays in 1,360 s
(3,206 per hour), proof 34 of 34. After them every one of the world's 492 catalogue
entries with a replay is valued by lookup (12 with a non-zero net; the 22 change-adding
sets all 0, as a key without a mechanism should), the table holds 2,156 rows, and the
random draws are nearly all valued: k = 3 +0.31 points (199 of 200 draws; 0 to 0), k = 6
+0.74 (198; 0 to +6.31), 4 h +1.47 (186; 0 to +8.33), 8 h +2.06 (172; 0 to +8.47). The
draws still skipped combine a change-adding entry with a fault's fix on one robot.

**D2 (w42)**, fourteen faults:

| Fault | Kind, scope | Episodes | Rescued / broken / net | Points | 95% interval |
|---|---|---|---|---|---|
| f01 | object_batch_variant, s2 | 112 | 16 / 2 / 14 | +4.17 | +1.79 to +6.85 |
| f02 | gripper_wear 0.5, r06 | 28 | 2 / 0 / 2 | +0.60 | +0.00 to +1.49 |
| f03 | gripper_wear 0.5, r08 | 28 | 1 / 0 / 1 | +0.30 | +0.00 to +0.90 |
| f04 | camera_pose_offset, drifting, r04 | 28 | 14 / 0 / 14 | +4.17 | +2.37 to +6.25 |
| f05 | furniture_shift, s3 | 112 | 53 / 0 / 53 | +15.77 | +12.20 to +19.94 |
| f06 | inert: policy_knob, s3 | 112 | 4 / 6 / -2 | -0.60 | -2.68 to +1.19 |
| f07 | decoy: illumination_change, s2 | 112 | 5 / 3 / 2 | +0.60 | -1.19 to +2.09 |
| f08 | decoy: backdrop_change, s3 | 112 | 8 / 11 / -3 | -0.89 | -3.28 to +1.49 |
| f09 | camera_degradation, r12 | 28 | 9 / 1 / 8 | +2.38 | +0.60 to +4.17 |
| f10 | control_mode_change, r11 | 28 | 15 / 0 / 15 | +4.46 | +2.38 to +6.55 |
| f11 | software_release, r03 | 28 | 25 / 0 / 25 | +7.44 | +4.76 to +10.12 |
| f12 | gripper_wear 0.75, r07 | 28 | 8 / 0 / 8 | +2.38 | +0.89 to +4.17 |
| f13 | success_detector_error, r05 | 28 | 0 / 0 / 0 | 0 | a data fault: zero by construction |
| f14 | gripper_wear 0.25, r01 | 28 | 0 / 0 / 0 | 0 | |

- *Detectability*: every cause the generator calls detectable shows at least two points:
  the interaction unit f01+f02+f03 jointly +3.87 (expected 4.05), f04 +4.17 (6.38), f05
  +15.77 (12.69), f09 +2.38 (1.39; qualified by the observed window alone), f10 +4.46
  (3.10), f11 +7.44 (7.55). f13 is listed as `data`; f14 is small; f12, a small cause
  in the generator's list (expected 0.56), measures +2.38: `stronger`. No miss.
- *Patterns*: both decoys confirmed (zero within the interval, although they flip 8 and
  19 near-threshold episodes each way); the confounder confirmed (f05 +15.77, its inert
  twin f06 zero within the interval); the drift confirmed (f04 rescues 4 of 14 on day 8
  and 10 of 14 on day 9). **The interaction is not confirmed**: on r06's 28 scoring
  episodes removing the batch f01 nets +2, the wear f02 +2, both together +1
  (interaction -3); on r08's: +2, +1 and +2 (interaction -1). The joint effect is below
  the sum. The review corrected the meaning of that sign: both terms are negative
  evidence for excess joint harm, but their intervals include zero, so both verdicts
  are **weak**, not confirmed (recomputed 2026-10-09).
- *Achievable gain*: k = 1 +15.77 points (f05), k = 3 +30.95 (f05, f10, f11), k = 6 +41.07
  (f01, f04, f05, f09, f10, f11), 4 h +20.83, 8 h +39.29. Every fault removed: 139
  rescued, 1 broken, +41.07; 8 of 336 episodes still fail with every fault removed.
- *Random fixes* (200 draws): k = 3 +0.37 points (5th to 95th percentile 0 to +2.38), k = 6
  +0.61 (0 to +4.17), 4 h +0.73 (0 to +4.46), 8 h +1.48 (-0.30 to +7.44), with 12, 22, 60
  and 74 draws skipped for an action that adds a change (10 catalogue entries here).
  After the catalogue's fresh replays (364 with `add`, 23 proofs, 414 s, 3,375 per hour,
  proof 23 of 23): all 492 entries valued by lookup (22 non-zero), 2,800 rows, and k = 3
  +0.42 (200 of 200 draws; 0 to +2.47), k = 6 +0.66 (199; 0 to +4.20), 4 h +0.79 (196;
  -0.30 to +4.46), 8 h +1.72 (193; -0.42 to +7.44).

**D3 (w43)**, eleven faults (shifted toward configuration and software faults):

| Fault | Kind, scope | Episodes | Rescued / broken / net | Points | 95% interval |
|---|---|---|---|---|---|
| f01 | object_batch_variant 4, s3 | 112 | 11 / 2 / 9 | +2.68 | +0.60 to +4.76 |
| f02 | gripper_wear 0.5, r09 | 28 | 2 / 1 / 1 | +0.30 | -0.60 to +1.49 |
| f03 | camera_pose_offset, drifting, r07 | 28 | 10 / 1 / 9 | +2.68 | +0.89 to +4.77 |
| f04 | action_latency 14, s2 | 112 | 13 / 2 / 11 | +3.27 | +1.19 to +5.66 |
| f05 | inert: policy_knob, s2 | 112 | 0 / 3 / -3 | -0.89 | -2.08 to +0.00 |
| f06 | decoy: illumination_change, s3 | 112 | 1 / 3 / -2 | -0.60 | -1.79 to +0.60 |
| f07 | action_latency 14, r02 | 28 | 1 / 0 / 1 | +0.30 | +0.00 to +0.89 |
| f08 | control_mode_change, r05 | 28 | 17 / 0 / 17 | +5.06 | +2.68 to +7.74 |
| f09 | action_latency 17, r04 | 28 | 14 / 0 / 14 | +4.17 | +2.08 to +6.55 |
| f10 | control_mode_change, r08 | 28 | 19 / 0 / 19 | +5.66 | +3.27 to +8.33 |
| f11 | software_release, r03 | 28 | 12 / 0 / 12 | +3.57 | +1.79 to +5.66 |

- *Detectability*: every detectable cause shows at least two points: f01+f02 jointly
  +2.68 (expected 3.61), f03 +2.68 (4.79), f04 +3.27 (2.53), f08 +5.06 (5.77), f09 +4.17
  (5.32), f10 +5.66 (5.68), f11 +3.57 (5.18); f07 small. No miss.
- *Patterns*: the decoy confirmed; the confounder confirmed (f04 +3.27; its inert twin
  f05 breaks 3 and rescues none, the interval just holds zero); the drift only **weak**
  (f03 rescues 4 of 14 on day 8 and 5 of 14 on day 9); the interaction is **weak** after
  the sign repair (on r09's 28 episodes: f01 +2, f02 +1, both +2, term -1; its interval
  -1.79 to +1.19 points contains zero).
- *Achievable gain*: k = 1 +5.66 points (f10), k = 3 +19.35 (f04, f08, f10), k = 6 +32.44,
  4 h +32.74 (seven fixes), 8 h +35.42. Every fault removed: 119 rescued, 2 broken,
  +34.82; 11 of 336 episodes still fail with every fault removed.
- *Random fixes* (200 draws): k = 3 +0.28 points (0 to +3.27), k = 6 +0.69 (-0.24 to +5.30),
  4 h +1.75 (-0.60 to +7.69), 8 h +2.18 (-0.60 to +8.93), with 7, 15, 58 and 69 draws
  skipped (8 catalogue entries add a change). After the catalogue's fresh replays (280
  with `add`, 23 proofs, 366 s, 2,988 per hour, proof 23 of 23): all 492 entries valued
  (24 non-zero), 1,820 rows, and k = 3 +0.29 (199 of 200 draws; 0 to +3.27), k = 6 +0.69
  (198; -0.30 to +5.36), 4 h +2.04 (190; -0.59 to +7.74), 8 h +2.52 (176; -0.59 to +9.00).

Across the three worlds: 4,050 planned jobs in 72 minutes of GPU and 1,900 fresh ones
for the catalogues in 36 (about 3,000 to 3,400 per hour with ten workers), every proof
held, no disagreement between rows. The completion criteria on these worlds: every fault
and every catalogue entry with a replay valued by lookup (492 per world); the decoys at
zero within noise (four of four, each flipping some near-threshold episodes both
ways); "rescued by a fault" counted (the broken column: 2 episodes for w41's f05, 2 for
w42's f01, and so on). After the sign repair, the marked interaction verdicts in both
old worlds are weak because their intervals contain zero. D3's drift remains weak.

### Size

Measured once on synthetic tables with random outcomes (not a test): a small world's
320 scoring episodes with 10 faults take 0.2 s for every gain with its 1,000-resample
bootstrap, a medium world's 960 with 16 faults 6 s, almost all of it the achievable gain
under the hour budgets (every subset of fixes that fits the hours is valued).

## Tests

The second review adds `tests/test_oracle_followup.py` (24 cases) and
`tests/test_oracle_exact.py` (8 cases). All nine surviving review mutations are killed
by their intended regressions ([followup-mutations.json](followup-mutations.json));
19 of the new follow-up cases fail against `83b168c`. Exact count sets and small cost
permutations are checked against brute-force enumeration, including duplicate
resolution classes, no-change entries, missing lookups, and preservation of additions
that actually have a mechanism. The status tests isolate incomplete-result causes
and include a complete-result success case.

Final follow-up source `89a98e6`: `make check` exited **0**, with 1,143 passed,
two skipped and 62 simulator/model cases deselected. The 32 new follow-up/exact tests
pass, all nine surviving mutations fail at their intended regressions, and 19 of the
24 follow-up cases fail against the reviewed `83b168c` source. The final raw-table
check reproduces all gains, intervals and pattern verdicts and renders both report
files byte for byte; it reuses the measured chance objects because their aggregation
code is unchanged. Every existing seeded chance field also equals the preceding
reviewed report. The synthetic calibration rerun is byte-identical to its artifact.

No engine, fault hook, replay execution or proof code changes in this follow-up;
`make check-sim` and `make smoke-world` were therefore not repeated. Their earlier
results below apply to the unchanged simulator paths. No new heavy-queue replay job
ran; the fresh option was tested by planning the real completed w41 draws and by
synthetic missing-row tests, not by a new model execution. All six prior report hashes
still match and the inventory of 38,408 original files still has no size/mtime changes.
Neither w44 nor w45 was opened or run. The new tolerance's physical calibration and
large cost-budget exact distributions remain unmeasured; the latter exceed the
explicit enumeration cap. These additional commits await the next vendor review.

The review regressions are in `tests/test_oracle_review.py`: 28 cases covering the
interaction sign and point/interval agreement, the revised additive-pair rule, the truth
extension, absent joint episodes, unbound replay identities and interrupted-run
origins, a healthy-policy failure rescued by a partial removal, missing healthy
outcomes, all-draw chance bounds, unchanged seeded draws, failed-proof command status,
and populated reports with the bootstrap disabled. The initial defect cases failed
against the unrepaired code and pass after the changes; the additional origin and
zero-point cases also failed against their preceding commits before their repairs.
Failed-proof status and interrupted-run checks use mocks; they do not demonstrate a
physical failed proof or interruption of a GPU run.

At report commit `0fae966`, `make check` exited 0: 1,111 passed and two skipped.
`make check-sim` at frozen replay commit `2beb1ea` exited 0: 58 passed. `0fae966`
changes only the interaction verdict predicate relative to that replay
revision; engine, origin and replay execution code are identical.

`tests/test_oracle.py` in `make check` (about 20 s): a toy table counted by hand
(planning, the cap, the three confirmation modes against the estimate, the manifest
check, resumable plans, the gains equal to the contract's functions, the hour budgets
equal to the scorer's, the seeded bootstrap, every fault removed, missing lookups, the
chance draws, the detectability verdicts, the pattern checks, an added change and an
episode above the cap needing fresh replays and then being lookups); the mini-v1
fixtures when present (the planned count 130 and its keys, the table of 234 rows from
job_outcomes, the two fix sets' gains and every ranked action's, the chance level equal
to the scorer's, the report, the command line); and the toy world end to end through
the engine's toy cell on the CPU (`fleetlab.oracle.toy`: a world run, the plan, a report
before the replays that writes into no run, the replays with their proof, a resumed run
that runs nothing, the report, a fix set that sets a delay on a healthy robot:
`MissingReplays`, a fresh set of 4 replays and 4 proofs, its value, planning it again
after the run; then a re-plan over the run replays refused, a changed manifest and
another world hash refusing the run's rows, a failed proof not run again).

## Choices where the design is silent

- **Confirmation replays** are the full power set over the excluded faults on top of
  each contract removal set. With one excluded fault per episode, as in w10, that is
  the integration's scheme; with two it costs four times the contract's replays for
  that episode. The count is in the plan before anything runs; `--confirm single` keeps only each excluded fault removed alone and every fault removed (m + 1 replays for m excluded faults), `--confirm none` drops them.
- **The proof** of a replay-only run compares with the as-run chains in the world's
  runs, which must therefore stay as they are.
- **The bootstrap** resamples episodes, not robots or days; episodes of one robot share
  their faults, so the intervals are narrower than a clustered bootstrap's would be.
  The achievable gain's interval is that of the best set in each resample.
- **Every fault removed** means decoys and inert entries too: the healthy episode.
- **Detectability** is checked on the net over the scoring window; a cause qualified
  only by the observed window still has to show there, and a miss names the window
  that qualified it.
- **`finalized`**: a run made before a run-level table joined the schema (the visible
  catalogue, contracts batch 9) is read when its job tables are complete; the mini-v1
  runs are such runs.
- **Interactions** are computed for every pair that occurs together; the marked ones are
  flagged and checked as patterns.
- **The table's intervals** live in oracle.json and summary.md; the parquet keeps the
  contract's five columns.

## What I did not verify

- **Claims 2 and 3 on a development world**: no analysis has run on D1 to D3 yet, so no
  ranked list was valued there; the valuation path is the one tested on w10's two fix
  sets and on every catalogue entry of the three worlds.
- **The `single` and `none` confirmation modes on real replays**: every world ran with
  `full`; the two others are tested on the toy table and counted for D2.
- **An episode above the cap on real replays**: no world has one (the composer refuses
  more than four effective faults); the path is tested on the toy table.
- **The pattern verdicts' thresholds**: a decoy is "confirmed" when its interval holds
  zero, a drift when the gain rises by ten points of its carrying episodes; both are
  my choices, and with 28 episodes per robot the intervals are wide.
- **Whether the marked interaction is real** in the old D2 and D3: after the sign
  repair their verdicts are weak, with intervals containing zero (the
  curves' joint point for batch plus wear was an assumed one).
- **Exactness across days or checkouts beyond today's**: every proof compared replays
  with as-run chains made hours earlier on the same machine and software (w10: another
  checkout; D1 to D3: the world generator's); exactness holds per machine only.
- **The medium-size timing** is on a synthetic table with random outcomes.
- **A crash in the middle of a heavy run on real data**: resumption is tested on the toy
  cell (and the engine's own tests); the one real interruption was the memory guard
  refusing to start, after which the same command ran.

## Distribution and cited-example round (2026-10-09)

Merged main's design decisions as `b9f82ec`; implementation `20805cb`, explicit-pair
fixture correction `670775b`. Reports are new records under
`~/.cache/robots-probes/worlds/{w41,w42,w43}/oracle-distributions-20261009`.
For w41 this supersedes `oracle-followup-20261009` as the report oracle of record;
its completed seeded replay source remains `oracle-review-20261009/fresh/chance/run`.
The original `w41/oracle` and both preceding review reports remain untouched.

Each stored world has 336 scoring episodes and 513 distinct complete catalogue
entries. Count populations are 22,369,536 sets at k=3 and 24,582,576,566,016 at k=6.
Before below means `oracle-followup-20261009`, which canonicalized no-hook additions;
after retains actual joint replay requirements and bounds missing values. Each cell
shows old value → new range, in points of fleet success. The JSON's scalar statistics
are null for bounded populations; the ranges live in `chance[].exact.bounds`.

| World | Fixes | Mean points, before → after | 5th percentile, before → after | 95th percentile, before → after |
|---|---:|---|---|---|
| w41 | 3 | 0.348 → [0.256, 0.403] | 0.000 → [0.000, 0.000] | 3.274 → [3.273, 3.274] |
| w41 | 6 | 0.695 → [0.253, 0.958] | 0.000 → [0.000, 0.000] | 5.655 → [5.654, 5.953] |
| w42 | 3 | 0.358 → [0.311, 0.391] | 0.000 → [0.000, 0.000] | 2.381 → [2.380, 2.381] |
| w42 | 6 | 0.715 → [0.484, 0.875] | 0.000 → [-0.298, 0.000] | 4.464 → [4.464, 4.465] |
| w43 | 3 | 0.309 → [0.244, 0.347] | 0.000 → [0.000, 0.000] | 3.274 → [3.273, 3.274] |
| w43 | 6 | 0.619 → [0.309, 0.800] | 0.000 → [-0.298, 0.000] | 4.464 → [4.464, 5.060] |

Unvalued set counts at 3/6 fixes: w41 32,766 / 172,915,426,222; w42 17,775 /
95,839,878,181; w43 22,731 / 120,659,454,772. Every histogram's multiplicities plus
its unvalued count equal the entire population. No unvalued mass was dropped.
All existing gains, bootstrap intervals, cause checks, pair verdicts and seeded-draw
statistics compare equal. Reports took 4.57, 4.79 and 3.70 seconds on this machine;
report statuses are 0, 1, 1 respectively, because w42/w43 still lack seeded draw values.
No new D2/D3 replay was requested. The full distributions, before/after values and cost
interval fields are in [distribution-comparison.json](distribution-comparison.json).

At cost budgets, the 200-draw population-quantile interval has binomial coverage
0.9671521638567256 (96.715%), with ranks 184 and 197. w41's complete 4-hour draws give
[8.333, 13.691] points for the population 95th quantile; 8-hour draws give
[8.333, 13.989]. Their sample 95th percentiles remain 8.333 and 8.943 points. The
other worlds' intervals also include missing-draw uncertainty; no cost pass mark is
read. Synthetic tests establish binomial coverage and bound containment, not a
new physical calibration of these worlds.


Final code verification at `17965f1`: `make check` exited 0 (1,158 passed, two
skipped). The unavailable-required-count-floor regression failed before `1c4876e`
and its three cases passed afterwards; the fault-free cited episode with an absent
outcome failed before `17965f1` and passed afterwards through the toy engine, proof,
lookup and resume. There are 15 distribution/request regression cases, including
small-catalogue exhaustive counting, missing-value completion containment, binomial
coverage, origin refusals and cause/pair independence. No cause computation assumes
pair connectivity or requires a marked pair's members to form a multi-fault cause.

The physical check used w41 observed-window episode `e001540` (robot r01, day 0,
object-batch fault f05). It failed as run; with every fault removed it succeeded.
`fresh --request` executed one healthy replay and one unmodified proof; exit status 0,
with 1/1 chains identical. HEAD's `table(all_windows=True)` found the healthy success;
the default scoring table excluded the example. Recomputing w41's report added this
zero-scoring-row run to its sources and left every statistic unchanged. The full
request-to-lookup receipt is [cited-example-receipt.json](cited-example-receipt.json).
This example proves the replay path; its healthy success makes it a negative example
for the policy-failure retraining indicator.

Measurements ran from the clean detached checkout at `670775b`, frozen throughout
the queue and jobs. Later code commits change the report status guard and the
fault-free missing-outcome planning case; this faulted example's manifest is identical
at `17965f1`, and engine execution, origin and proof code are unchanged.
`make check-sim` exited 0 (58 passed); `make smoke-world` exited 0 (16/16 chains
identical to the stored reference, none differing or missing). A size/mtime inventory
of 38,408 original files is unchanged. No new D2/D3 replay, physical fault-free episode
with a missing outcome, or interrupted GPU run was exercised in this round. The
fault-free missing-outcome and resume paths were exercised on the CPU toy engine;
missing exact and seeded values remain explicitly bounded in the stored comparisons.

## Round 8: repairs after the rounds 5–7 review

Runtime and regression changes are in `1bc0d94`; fixture typing is repaired in
`dd7de56`. `round8-verification.json` records the source hashes, mutation inventory,
CPU kill/resume evidence, saved-world comparison and final check results.

The ceiling is always in second-execution policy requests. The missing-total test
refuses rather than substituting steps; the saved-missing-total test recovers the
exact count from the engine record when it exists. The reviewer's real CPU probe
was killed with ten of twelve second executions finished and three planted events;
its resume exits 1 and names the missing request count. The engine's persistence
was not changed. Such completed replays need a new verification directory if their
counts were never retained.

Sparse carrier fixtures include a removal with three rescues and one broken attempt:
its net is two points of the 100-attempt window, not ten points of its 20 carriers.
Exactly two points and exactly 20 observed carriers qualify; a pair requires both
members. Pending membership, canonical outcomes, full-chain/outcome agreement,
saved evidence on report reads, missing coverage, refused ledgers and strict Poisson
tails all have direct regressions. All 49 oracle mutations in the reviewer's
50-break inventory are caught. S10 changes the scorer and was not run or edited:
the hand-over assigns that delivery to its own builder. No oracle survivor remains.

An ineligible decoy measuring at least two points is `small`, neither a measured
cause nor a scoring decoy. The actual scratch-engine row-dependence test accepts a
single event: both further jobs run alone in row zero. That event cannot establish
independence from the original batch companions; only the ceiling limits it.

`oracle report` exits 0 on a reflink copy of the completed w47 oracle. The complete
measured-cause object (six causes), all thirteen scoring gains and all six observed
gains are exactly unchanged. Its 910 identical comparisons still cover 8,926 second
policy requests, with scoring/observed ceilings 6/7 and no event. The residual has
12,866 recorded changed-replay calls: 4.18894 expected affected replays and 1.24671
expected wrong-outcome points per scoring table. Twenty changed replays have no
saved total; their explicitly labelled 5,600-step budgets raise the combined figures
to 6.01220 replays and 1.78934 points. The review's approximate 4.2/1.26 describes the recorded-call component; those
twenty missing totals require the fallback as well. Observed removals all have totals: 5,511
calls, 1.79428 expected affected replays, 0.17800 points. These are conditional rate
bounds, not measurements of actual wrong outcomes.

The pre-repair regressions failed six cases. The oracle suite passed 203 tests,
plus the final strict-tail test checked separately. The single capped
`make check` invocation exited 2 at mypy before pytest, on eight errors in the new
fixture helpers. They are fixed in `dd7de56`. After that repair, full lint and format
checks exit 0; full mypy exits 0 across 394 files; the complete pytest run exits 0
with 2,619 passed, two skipped, 65 deselected and one existing robosuite warning in
2,895.01 seconds. The make invocation was not repeated or relabelled as a pass.

Camera-first differences (finding 2), ceiling scope per source run/window (finding
8), misreports counted in both directions (finding 9), the engine and the scorer
remain unchanged. No GPU policy or LIBERO measurement, new model session,
publication, merge or push was performed. Confidence is high in the arithmetic and saved-evidence
checks; this round does not establish live GPU behavior or the rate assumptions.
