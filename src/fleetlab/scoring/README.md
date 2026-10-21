# Development-world scoring

```sh
make score SCORE_FLAGS="--world-root WORLD --analysis ANALYSIS --out NEW_SCORE"
make score SCORE_FLAGS="--reports SCORE1/report.json SCORE2/report.json --world-roster FROZEN_ROSTER.json --out NEW_COMPARISON"
```

An analysis has `analysis.json`, `outbox/diagnosis.json`, `outbox/fix_set.json`
and runtime `sessions/`. Output directories must be new. The scorer writes JSON
and Markdown without changing input records. World and oracle data are read
through the public oracle readers. Claims 1–3 and 5, runtime accounting (claim 6),
all frozen section-8 marks, errors and the k=3 headline are included. Claim 4
requires its separate live experiment and remains unavailable here. Pass marks
live only in `thresholds.py`; their values are unchanged.

## Comparisons

The required opponents and accepted producer method names are defined once in
`comparisons.BASELINE_METHODS` (`REQUIRED_BASELINES` is derived from it):
no-data ranker, worst health metric, dramatic videos and classical statistics.
Every one must have a valued three-fix result before the margin or ratio exists.
For **with experiments**, `classical statistics with experiments` is additionally
required (entry 13); the comparison uses the stronger of that method and the
best of the four logs-only opponents. Both are shown. **With memory** and
**knowledge ablation** use those same four logs-only opponents. Supply it with
`--baseline 'classical statistics with experiments=UPDATE_OUTPUT'`, where the
directory was written by `fleetlab.baselines.classical.experiments update ... --out`.
Logs-only evidence cannot stand in for experiment evidence, or vice versa.
Extra methods cannot replace a missing required one. The miniature chain's
`trivial_no_data_ranker` is an explicitly labelled deterministic prior, not a
substitute for the required model baseline.

```sh
make score SCORE_FLAGS="--world-root WORLD --analysis ANALYSIS --condition 'logs alone' --repeat 1 --baseline 'no-data ranker=PRIOR1' --baseline 'no-data ranker repeat 2=PRIOR2' --baseline 'worst health metric=HEALTH' --baseline 'dramatic videos=VIDEO1' --baseline 'dramatic videos repeat 2=VIDEO2' --baseline 'classical statistics=STATS' --out NEW_SCORE"
```

A bare model-baseline label denotes its first repeat; `NAME repeat 1` is also
accepted and takes precedence. `NAME repeat 2` is displayed and never decides
a mark or ratio. A required source must carry its own method record, and a model
source must explicitly record repeat 1 to enter comparisons. Method/repeat
mismatches and two names sharing a resolved source directory are refused.
Copied fix sets without their producer record are displayed, but leave margins
and ratios unavailable. Simple methods bind through their producer's
`manifest.json`: its export bytes must match the analysis's recorded, certified export,
its method output
pointer must name the supplied method directory's `outbox/fix_set.json`, and
model repeat numbers come from `model_repeats`. Classical statistics binds
through its own output directory's `evidence.json`: every recorded `input_sha256`
must match that same certified export; hashes of description and observed
episodes are required. The public `world_id` in this producer record is checked
against that export's public fleet name, then the binding records the real world ID.
The real identity comes from `WorldPaths.from_root`; the export's fleet name is
never its binding identity. This allows opaque worlds sharing public `w01` while
refusing another world's differing export bytes. Readiness receipts must match
the current checker; `fleetlab.log certify WORLD` appends a new receipt when needed.
`metadata.json`, `analysis.json` and `run.json` provide consistency checks;
they cannot substitute for either producer record. Conflicts are refused.
Older model records without a producer-recorded repeat cannot decide comparisons.
Missing second repeats are shown explicitly. Baselines accept
ranked answers, individual `FixSet` files, or directories of budget records.
Every producer with `fix_sets/` is read through that directory, also when an
individual budget path is supplied. If its metadata marks
`explicit_fix_sets_authoritative: true`, a supplied `outbox` or ranked
`outbox/fix_set.json` path is refused. Later ranks cannot fill vacancies. Explicit sets must
survive the specified count/cost budget unchanged; missing budget files are
unavailable. Baselines are valued by the same oracle as the agent. Only the
producer's pointed-to answer or its sibling `fix_sets/*.json` can feed required
comparisons; unrelated files and outputs escaping that directory are refused.
JSON retains the producer record path/hash, world binding and file hashes.
Neither current `manifest.json` nor `evidence.json` has an output-hash field for
`fix_sets/*.json`; the latter hashes inputs only. The available binding is to
the producer directory. Scorer-captured output hashes identify bytes scored;
they do not prove those bytes have remained unchanged since production.
**Known limit (review finding 9): edited producer outputs still bind.** A changed
fix set or rewritten producer export pointer cannot be detected without hashes
recorded by the producer when it ran; the scorer does not invent that evidence.
The experiment producer additionally records `condition=experiments`, request
and completed-request counts, request/result hashes, and experiment-input hashes.
Its description must be the same as its logs input. It records no path to the
request/result files, so those two hashes cannot be independently checked against
their origin; they are retained producer assertions. It also has no full-export
digest: world binding uses its recorded visible-input hashes against the export
whose whole-byte digest readiness certifies. The producer is unchanged.

The blind baseline default-seed reader uses the standard library only. It reads
`seed`, `world_id`, and `world_hash` from the specified oracle report, checks root
identity/hash metadata, and binds the export bytes to a clean receipt. It never
reads a chance value. The scorer's public readiness reader separately requires
the receipt's current checker identity and vocabulary digest.

Entry 11 reads the runtime's fields: `lost_twice` in an analysis's `run.json`
(including its recorded parent run) or `sessions/session-NN.json`; `attempts`
and `rerun` in baseline metadata; and `manifest.json: missing.<method>.attempts`
when a lost baseline has no method directory. Each start records its error kind,
detail, seconds and whether an answer arrived. Two lost starts make a baseline
missing, never a valued zero, even if stale files exist or its absent path was
supplied explicitly. One lost start and a recovered replacement are not
exhaustion. The analysis runtime's `lost_twice` decision overrides any outbox
contents. Legacy records keep answer-readability checks. This package starts no
runtime retry or model call.

An analysis `run.json` must contain the world's hash. Its `export` selects the
comparison snapshot: a current clean receipt must certify those exact bytes,
and every baseline must bind to them too. A different certified snapshot of the
same world is allowed, including w41's original review export. Missing analysis
export provenance leaves margin and ratio unavailable; a foreign export,
conflicting records, absent world hash or mismatched baseline inputs is refused.

Without full scoring-window coverage in the oracle report's
`replay_verification.coverage`, claims 2 and 3, their marks, the headline and
pooled comparisons are unavailable. Coverage must name exactly one scoring row,
with both attempt counts equal to the scoring window and an empty missing list.
Claim 1 retains its separate measured-membership gate. The report gives the
world-specific `fleetlab.oracle verify ... --window scoring` command and the
`fleetlab.oracle report ...` command needed to refresh the report afterward.
Both commands name a new `*-verified-NEW` directory and retain the recorded run
sources; replace `NEW` with a fresh destination suffix. Report generation uses
`--no-truth-table` to preserve the original runs.
Legacy oracle reports qualify for this refusal too. Integer nets decide the
three-point margin; chance uses the unrounded, rational 95th percentile of the
population net histogram, placing unvalued sets at their upper net bound. An
exact chance tie misses, and an exact three-point margin meets the mark.

The scorer reads `chance[]` from the selected oracle report. It computes no
world chance distribution. At count budgets it displays the exact population,
its bounds and the 200 draws side by side. The mark, headline and world ratio
use the exact k=3 95th percentile, or its upper bound when replays are missing.
The comparison is strict. Histogram counts plus unvalued sets must equal the
recorded total number of sets, in both world and pooled comparisons. An absent
or inconsistent accounting record makes the exact level unavailable.
A missing exact level gives an unavailable result
with the oracle report command. At cost budgets, the oracle's 200-draw estimates,
missing-draw bounds and binomial order-statistic percentile interval are shown;
no mark is read at those budgets. `--oracle-dir NEW_ORACLE` and repeated
`--extra-run RUN` select alternative immutable records. Produce a new oracle
report after adding replay sources; extra rows do not silently recompute chance.

Pooled metrics retain arithmetic mean/minimum/maximum and identify missing
worlds. The CLI requires `--world-roster FILE`; the Python API requires
`pool(reports, roster=Path(...))`. Typed world lists are refused. The JSON file
has this shape (use real hashes from the world records):

```json
{"worlds": [{"world_id": "w41", "world_hash": "<64 hexadecimal SHA-256 characters>"}]}
```

Each entry includes `world_hash`, explicitly null only where the world has no
hash. Supplied report hashes must equal their roster entries. The report prints
the roster's resolved filename and SHA-256 of its exact bytes, in JSON and
Markdown, so the orchestration can compare it with the frozen test-world record.
This file must list every world in the
intended comparison, including worlds with no delivered analysis. A missing
world leaves every mark and ratio for its cell undecided. Unverified run labels
are refused at scoring and pooling; older reports without
`metadata.run_labels_verified=true` must be rescored against their run records.
A pooled mark reads the complete mean, with the count of individually
passing worlds alongside it; above chance must hold in every world. Undefined
correlations leave that mark undecided and name the world. Repeats and conditions
remain separate. The pooled ratio uses the best required opponent in each world;
the ratio to the single opponent best on average is shown alongside it. The floor
is the 95th percentile of independent random sets' mean share, convolving oracle
exact histograms with integer weights. Shares are rounded down to a 1/4000 grid
before their mean; the percentile error is bounded by 0.00025 of share, below
0.001. Missing sets are placed at the oracle's worst/best net values. Comparisons
use the upper endpoint, including that rounding allowance. Missing histograms,
worlds or required baseline values leave the ratio unavailable.

## Measured causes (claim 1)

Entry 15 makes the oracle's `ORACLE_DIR/measured_causes.json` the sole membership
authority for recall, both precisions, decoy rejection, error feedback and type
splits. The scorer reads its public `MeasuredCauses` schema and verifies the real
world ID/hash and frozen rule. A complete list with no recorded problems is
required; absent or incomplete measurements leave claim 1 unavailable, with:

```sh
uv run --locked python -m fleetlab.oracle causes WORLD --oracle-dir ORACLE_DIR
```

That command belongs to the oracle and can simulate; scoring itself does not.
There is no fallback to generator membership. All members of a measured joint
cause must be named in accepted directions. A promoted measured decoy is a cause;
one below the threshold remains a decoy. With no remaining measured decoy, its
rejection mark is unavailable. The measured list and complete window measurements
are in JSON; the curve list and curve-derived diagnosis numbers are displayed
alongside, explicitly descriptive. Pooling also labels curve metrics descriptive.
Generator units/flags that disagree appear in that curve audit; they do not make
the independent measured diagnosis provisional. A missing diagnosis scores zero
recall once complete measured causes exist; missing truth remains unavailable.

## Claim 5 and truth consistency

Drift warnings use the truth table's own `drift_crossing_day`. Absence of that
field makes the measure unavailable; null means the drift never crosses within
the world. The daily scope-versus-fleet crossing, its separate verdict, forecast
error and lead time remain alongside the calibrated measure (latest entry 9).
Already-crossed, later-crossed-and-warned, and later-crossed-without-warning are
separate states. An early crossing reports whether the agent noticed the drift,
without claiming lead time. Forecast error is expected day minus drift day.
Warnings at scopes with no drifting fault are counted as false alarms, including
in worlds without drift.

Every missing healthy example is listed in the native `needs_fresh_replay`
format, with an empty effective fault list. The oracle accepts the score itself:

```sh
uv run --locked python -m fleetlab.oracle fresh WORLD --oracle-dir NEW_ORACLE --name NEW_NAME --request SCORE/report.json --plan-only
```

Its planner includes observed-window identities. Run simulations through
`tools/heavy-run` from an immutable detached checkout; this scoring command
starts none. Add completed, proven replay sources and score again. No indicator
precision in an analysis is published or pooled until all examples of all its
indicators have healthy outcomes. The measure says "the policy itself fails this
attempt on a healthy robot"; it does not mean no other removal can rescue it.

The generator's curve units remain an audit, while measured units decide claim 1.
The scorer changes neither input cause list. Directions come from
`FaultTruth.directions`: for object batches
`changed` is accepted, giving six of six recall on the first stored D1 answer;
the former class-only wording would give five of six.

## Choices where the design is silent

- The roster is an input record, not inferred from delivered reports. Its hash
  identifies the supplied scope; the scorer cannot certify that it is the
  orchestrator's frozen roster. Constructed validation creates a labelled
  single-world fixture roster. Development checks use development rosters only.
- Simple-baseline export bytes must equal the analysis-selected certified export, even
  when a producer used a byte-identical copy elsewhere. Classical input hashes
  must match that same export, including an observed-episode hash; a
  description-only record cannot bind worlds sharing a public name. With no producer
  output hashes, directory binding is the strongest available check. The
  producers are unchanged; no synthetic hash is represented as their receipt.
- Measured membership is projected onto the fault table for target/direction
  matching, without altering it on disk. A targetless claim qualifies as vague
  only where the measured window points show an effect at that exact scope;
  generator-curve expected points no longer establish that qualification.
- Exhausted calls are recognized through the runtime receipts described above.
  An unrecorded timeout is not inferred from files or from a zero-valued fix set.
- Scope and target matching are exact. A site fault named on one robot is a
  false positive; it does not count as naming the site decoy. These are review
  findings 12 and 18's retained rules. Confounder resolution counts the effective
  partner as named in any direction, provided the inert partner is not named;
  this is distinct from a direction-correct recall hit.
- Duplicate targets at the same scope count once before the confidence cut.
  Highest confidence selects the representative; tied duplicates use list order,
  including conflicting directions. Ties at the n+2 cut use list order.
- Verdict priority is hit, wrong direction, small, decoy, inert. Wrong directions
  and small effective faults are neither hits nor false positives. These counts
  and targetless-cause counts are over named causes only. A targetless cause is
  vague only at the exact scope of an effective fault with a nonzero expected
  effect. A scope containing only a decoy or inert change is a false positive.
  Primary precision counts all targetless causes as false positives. The older
  precision excluding qualifying vague causes is shown second (latest entry 8).
- Spearman ties receive average ranks. Constant series have undefined correlation
  or slope. Partial singleton results remain feedback but cannot pass the mark.
- Accepted run-record condition aliases are `logs` and `experiments`. Condition,
  repeat, hash and issue-day mismatches are refused. Older records derive their
  issue day from their recorded condition's window endpoint. If a record lacks
  either condition or repeat, scoring is refused. Moving an analysis away from
  its enclosing run record does not allow caller labels to replace that record.
  Explicit recorded issue days take precedence over window defaults. Later memory
  checkpoints need their own recorded issue day. `expected_day` is a forecast.
- Warnings match drift scopes exactly, regardless of confidence. Mean lead time
  includes warned future crossings only. Retraining examples are deduplicated
  within an indicator; observed fault-free as-run outcomes are already healthy.
- Shares use signed net gains, can exceed one under partial-scope repairs, and
  are undefined when achievable gain is zero. Claim-2 top-k shares reuse claim 3.
  Cost-budget achievable gain tries subsets of the cheapest applicable fixing
  action for each active fault, restoring nominal values; verb-order breaks ties.
  Faults without a fixing action are omitted with a note. Count budgets use the
  contract's achievable gain. Missing replays are never replaced by zero.
- No-replay verbs retain their budget place and gain zero under claim 3; claim 2
  excludes them. Explicit empty fix sets are the no-intervention floor. An invalid
  or unreadable submission scores as no answer, with no causes or fixes, plus the
  parse error. Each absent half is scored independently: no diagnosis finds no
  causes (zero recall where causes exist), and no ranking chooses no fixes
  (zero net gain). Its supplied, readable other half is retained. Precision and
  prediction statistics with no denominator remain undefined. An absent or
  unreadable diagnosis earns no decoy-rejection credit. An invalid run identity
  or a mislabelled run is refused. Constructed validation answers carry an
  explicitly synthetic run record with their condition, repeat and issue day.
- The additive-pair wording check reads a named cause's `failure_mode`, catches
  the word `interaction` except simple adjacent denials, and displays it only in
  `remarks`. It misses synonyms such as "synergy" and "interact", and can flag
  "interaction unlikely" or complex denials. Neither precision nor any measure
  reads these remarks. Oracle additive verdicts are confirmed/contradicted/
  indeterminate; interactions are confirmed/not confirmed. No synonym of these
  verdicts is treated as measured confirmation.
- Traceability checks numbers at half their last written decimal, allowing rate
  scaling by 100 and ignoring signs. Successful, delivered query/describe/frames/
  experiment results are sources; skills, notes, submissions and request receipts
  are not. Calls whose arguments carry a matching number are echoes. Text integers
  below ten without decimals are counted separately; numeric prediction fields
  are always checked. Confidences are judgement, not traced data. Every session's
  calls are searched, and undelivered results are excluded.
- False positives with no matching truth type, including targetless causes, stay
  in global precision and the split's `unmatched_false_positives` list. Split
  precisions omit them rather than assign a fault type without evidence.

## Accounting and type splits

`--ledger FILE` overrides discovery from recorded session run directories, adjacent
`agent-runs/`, then the shared runtime root. Failed calls and retries are charged.
Missing session or retry rows make normalized costs unavailable. The denominator
is distinct observed attempts plus delivered experiment attempts, overridable
with `--attempts-analysed`. Wall time sums CLI process times and excludes the
experiment queue. Input includes cached tokens; reasoning is within output.

CLI-recorded list-price equivalent is preferred. Otherwise `--prices FILE` maps
exact model names to per-million USD input/cached-input/output rates, with `source`
and `as_of`. Uncached and cached input are charged separately. No price or alias
is guessed. `--unseen-types FILE` supplies exact type names (development default
`[]`). Cross-type units/actions remain mixed; decisions use disjoint known-only,
any-unseen and no-effective-fault episode cohorts, retaining world-point gains.
No interaction effect is arbitrarily attributed to one member.

## Verification and limits

Measurements below through the fourth repair round use the former curve-based
claim-1 rule. They are historical records, not measured-membership results under
entry 15. The fifth round's missing-file result and constructed fixtures are
reported separately below.

`python -m fleetlab.scoring.validation --world-root WORLD --out NEW_DIR` creates
labelled hidden-truth and empty answers, names every member of every counted
cause, and checks ceiling/floor from stored public records. Its random section
is a direct replay-coverage audit, **not validation of chance percentiles**.
The previous same-seed 200-draw comparison was circular as a chance check; its
190/196/193/188 inside-interval counts remain historical coverage observations.
Unit tests use hand-valued comparisons and toy outcome tables, including exact
versus sampled chance, pooling, provisional truth, warnings, native replay
requests, malformed submissions and ledger gaps. No regenerated
world, held-out world, live retraining experiment or live billing is established
by these fixtures. The regenerated-world round is intentionally separate.

The supplied review's `mutations.txt` contains **22** surviving rows, although
its prose says 21. All 22 consequential errors are now caught: 13 mutations apply
at their original sites and nine equivalent mutations apply at the corresponding
refactored sites. None survives. `review-mutations.json` names each case and the
failing test; the independent review's runner was used on a new scratch copy,
with no review implementation copied into this package.

A stored D1 repeat-2 rescore exposed scalar `message` values in CLI streams.
The minimal companion fix in `integration/chain.py` skips non-object messages
while preserving structured refusal marking; a regression checks both together.
This is the only source edit outside the assigned scoring paths.

The specification on main advanced to a7b574b during this repair. Its entries
8 and 9 take precedence over the handoff message: primary precision penalizes
targetless causes (the older rule is secondary), and both logged-rate and
calibrated warning measures retain their states and lead times.

Cause-unit caveats remain on pooled diagnosis metrics and recall marks, naming
each affected world and fault also at the top of the standalone Markdown.
Provisional marks retain their numeric value and frozen threshold; their displayed
result is explicitly provisional. Exact count-budget chance and cost-budget
percentile confidence intervals are shown beside the descriptive 200 draws.

### Repair-round stored measurements

The original two D1 logs-alone analyses were rescored at executable `84b532a`
with the latest stored baselines-v3 first repeats and frozen classical `b022bbf`.
Every existing second model repeat is displayed. These are stored agent answers,
not new agent runs. The stored `oracle-distributions-20261009` report was read
through the public oracle reader, without simulations or source-record changes.

| D1 analysis | Recall | Primary precision | Older precision | Exact k3 p95 bounds (points) | Margin | Ratio |
|---|---|---|---|---|---|---|
| logs alone, repeat 1 | 6/6 | 6/7 | 6/6 | 3.273–3.274 | unavailable | unavailable |
| logs alone, repeat 2 | 6/6 | 6/8 | 6/7 | 3.273–3.274 | unavailable | unavailable |

The first video baseline timed out; its second repeat succeeds but cannot replace
it. World and pooled margin/ratio checks explicitly require that missing first
repeat. The exact comparison uses the upper bound, 3.274 percentage points of
fleet success over 336 scoring attempts. These population bounds retain 32,766
unvalued sets among 22,369,536 three-fix sets. The headline's pooled floor is
convolved from oracle histograms with error bounded by 0.00025 of achievable
share, below the required 0.001. Latest reports and assertion receipt:
`~/.cache/robots-probes/scoring-checks/review3-84b532a/`.

A separate historical v2 four-baseline arithmetic check gives zero points of
margin and ratio 1 for both analyses; removing classical statistics makes both
unavailable. It is not a comparison against the latest baseline protocol.
Those v2 model records have no repeat number: the current scorer makes both
comparisons unavailable even when all four labels are supplied.
Records: `~/.cache/robots-probes/scoring-checks/review-repair-11729ae/`.
The D1 constructed-truth check there reaches recall, singleton Spearman and k3
share 1; an empty answer reaches recall/share 0. All 200 random draws per budget
have replay coverage, which does not independently validate the chance population.

The previous round's stored round-8 three-world pool gave recall
0.8015873015873015, with its mean mark met and two of three worlds meeting it
alone. Its diagnosis metrics were provisional, naming `w42/f01` and `w43/f01`.
This is historical evidence; old reports lacking verified labels now require
rescoring. The caveat regression still checks preservation of those flags.
These old worlds have not
received the generator's repaired cause lists or calibrated drift-day fields;
own-day warnings stay unavailable until those truth fields are delivered.

### Third repair round

All eight survivors of `mutations2.txt` are caught by assertions: repeat
substitution, any-world instead of mean marks, incomplete-world ratios,
low-end floor comparison, 90th percentile, dropped unvalued sets, omitted
rounding allowance, and upward rounding. Seven use the reviewer's exact
replacement; repeat substitution uses the equivalent lookup after refactoring.
The scratch module path was checked before running its unmutated 70-test suite,
which passed. None of the eight survives. Receipt: `mutations.txt` in the latest
measurement directory above.

The reviewer's attacks were rerun against `84b532a`, using new scratch copies.
Producer relabelling/shared sources are refused, copied fixes lacking records
cannot yield comparisons, and moved runs lacking labels are refused. The old
scalar exact record cannot decide a chance mark. The CLI pool with only D1
reports and an explicit w41/w42/w43 roster leaves every mark and ratio
unavailable, naming the two absent worlds; no omitted-roster call is accepted.
Malformed and missing-half variants carry an unchanged copy of the original
run record so the answer path is exercised separately from label refusal.
No diagnosis earns zero recall/decoy credit; no ranking earns zero net gain.
The supplied readable half retains its measured score. Receipt:
`head-attacks.txt`; these altered answers are synthetic checks.

D1 constructed validation was also rerun at this commit: ceiling recall,
singleton correlation and k3 share are one; empty recall/share are zero.
The 200 random sets at each of the four budgets have replay coverage; this
continues to make no claim to validate the oracle's population percentiles.
The real calibrated drift-day path, regenerated worlds, with-experiments
analyses and simulations remain outside this round.

`make check` exited 0 at executable `84b532a`: lint, formatting and types passed;
1,205 tests passed, two skipped and 62 simulator/model tests were deselected.
This round changed only scoring code and its tests/README, so simulator checks
were not run.

### Fourth repair round

Executable `2d3f059`, with test typing corrected at `6967ced`, binds required
values to producer manifests/evidence and scored-world outputs. Both remaining
review attacks were asserted against `6967ced`: w42's producers on w41 are
refused; three copied fix sets with handwritten metadata leave margin and ratio
unavailable. The real producers still qualify. Neither producer has hashes of
its output fix sets, so the verified directory/input binding and this limit
remain explicit. No producer code or stored record was changed.

Both original D1 logs-alone analyses were rescored with all four required
producers, including the successful first-repeat video rerun. Existing second
model repeats are displayed separately. Recall is 6/6 in both; primary precision
is 6/7 and 6/8, and older precision is 6/6 and 6/7. Each recovers 67 net successes
over 336 scoring attempts (19.94047619047619 percentage points). Each has zero
points of margin and a ratio of one against the best baseline; omitting classical
statistics makes both comparisons unavailable. Exact k3 chance uses the upper
end, 3.274 points, of the oracle's 3.273–3.274 p95 bounds.

The pooled check reads `development-roster.json`, listing the public world
hashes of w41/w42/w43, rather than a typed list. Its SHA-256 is
`83f4fb42d493c0188d24b9b96b556f55bcd83f9088c263395bc193c5940d3d70`, identifying
this development scope only. Supplying the two D1 reports leaves all pooled
marks and ratios unavailable, naming w42 and w43. JSON and Markdown name the
roster and print its hash. Tests refuse typed lists, missing roster/hash fields
and disagreeing world hashes.

The focused suite passes 74 tests. Both new surviving review mutations are
killed by `test_core_comparison_requires_method_record_and_unshared_source`:
dropping the method-record check and tolerating a shared required source. The
reviewer's mutation runner used a fresh scratch package whose import path was
verified; no implementation was copied from the review. Artifacts and receipts:
`~/.cache/robots-probes/scoring-checks/review4-6967ced/`.
These are stored-data checks and constructed attacks, with no new analysis,
simulation, regenerated world or held-out-world access.

`make check` exited 0 at `6967ced`: lint, formatting and types passed;
1,209 tests passed, two skipped and 62 simulator/model tests were deselected.
No simulator code changed in this round, so simulator checks were not run.
The successful CLI pool has the same values as the API pool, including its
roster identity and undecided marks. Receipts include `make-check.txt`,
`final-head.txt`, `mutations.txt` and `cli-verification.json` in that directory.

### Fifth repair round: third-generation w47

Executable `f9a9b36` scores both stored logs-only analyses of
`~/.cache/robots-probes/worlds/w47-opaque-v1`, whose real ID is w47 and public
fleet name is w01. All four required logs-only producer sources bind to its
certified export. Oracle random fixes are the fifth section-6 baseline; both
model repeats are displayed. The blind reader finds the oracle seed
6132661456928807008 by default. Fresh clean certification receipts were appended
to w47 and w41 for the current checker; exports and stored runs were unchanged.

The oracle's measured-causes file was absent during both scores. Primary recall,
both precisions, decoy rejection and their marks therefore remain unavailable.
Constructed `MeasuredCauses` fixtures check decoy promotion at two points,
below-threshold decoys, all-members joint hits, world/hash/rule binding, list
accounting and missing/incomplete results. They establish scorer behavior, not
w47's measured membership. Curve-only recall is 5/6 in both runs; primary-style
curve precision is 5/7 and 5/5, and older curve precision is 5/6 and 5/5. None of
these curve numbers decides a mark.

| w47 logs-only repeat | Singleton Spearman | Net successes at k3 | Share of achievable | Margin (points) | Ratio |
|---|---|---|---|---|---|
| 1 | 0.451220 | 69 | 69/78 = 0.884615 | 0 | 1 |
| 2 | 0.725191 | 73 | 73/78 = 0.935897 | 1.190476 | 1.057971 |

There are 336 scoring attempts; 78 is the oracle's maximum net rescued minus
broken at three fixes. Net fleet gains are 20.535714 and 21.726190 percentage
points. The strongest baseline, classical logs, recovers 69 net attempts.
No-data recovers zero, health recovers minus one, and video recovers ten; both
model repeats have the same net as their first repeat here. Exact random k3 p95
is the oracle's 3.571 fleet points over all 22,369,536 sets, none unvalued.
Both margin marks miss the unchanged three-point threshold. The drift's own
crossing day is 8, versus issue day 5: both analyses are **later crossing, not
warned**, with no credited lead or forecast error. The separate logged-rate
crossing never occurs. Their retraining precisions remain unavailable: three
and four cited examples respectively lack healthy outcomes and are listed for
fresh replay.

An earlier w47 table reports video net 17 (+5.06 points). Its oracle
`fresh/baselines-1/values.json` values the first three ranked actions, including
`set r12 control.action_delay_steps = 0`. The producer's authoritative
`dramatic_videos/fix_sets/count-3.json` contains only lighting s3 and latency r11;
this scorer values that two-action set at net ten (+2.976190 points), as the
existing FixSet contract requires. No record was changed. The strongest opponent
is classical in both cases, so this difference affects neither margin nor ratio.

Stored reports: `~/.cache/robots-probes/scoring-checks/w47-gen3-rescore-aWZUuQN2/`.
The measured file now exists. Rescore both analyses to a new
directory with this exact command (it starts no model or simulation):

```sh
bash /home/cs/.cache/robots-probes/scoring-checks/gen3-f9a9b36/rescore-w47.sh
```

The round-3/4 stored attacks (foreign w42 producers on w41, copied fixes with
handwritten metadata, and shared producer labels) yield no margin or ratio.
The isolated metadata forgery on w47 reaches comparison and explicitly leaves
both unavailable, naming the three absent producer bindings. The new attack
uses constructed w47/w43 exports with an identical public description and
different observed bytes: simple, classical logs and classical experiments
foreign producers are refused, while each genuine counterpart binds. No other
third-generation world root was opened. Attack receipts are in
`/tmp/scoring-r5-f9a9b36/`; the 146-test focused suite passed.

With-experiments producer binding, stronger-opponent selection, incomplete
comparison/pool refusal and two-lost-start handling are constructed checks.
No stored experimental classical output or real lost-start record was used.
Round 5 implemented no retry and tested an inferred receipt contract; round 6
replaces it with the actual runtime fields above. No GPU, simulator or new model
analysis was started, and no oracle/generator source was edited.

## Round 6 repair evidence (2026-10-11)

`tests/data/scoring/runtime-records.json` was produced by the actual writers in
`fix/lost-call-rule` at `0353c6d`, with stand-in results and no model, simulator
or GPU call. It contains a failed analysis session, a run-level loss flag derived
from that returned session, two lost baseline starts, one ordinary failed start and a recovered
replacement. The fixture is stored here; running the tests needs no other
checkout. No runtime source was merged into this branch.

| Guarantee | Regression evidence |
|---|---|
| Runtime loss records override stale answers; absent baseline paths stay missing | `test_runtime_attempt_records`, `test_runtime_session_and_run_override_stale_outbox`, `test_lost_manifest_without_method_directory_is_missing`, `test_cli_lost_analysis_is_no_answer` |
| Unverified scoring attempts yield no claims 2/3, marks, headline or pooled ratio | `test_score_world_blocks_unverified_claims_and_pool`: absent legacy coverage, partial execution, wrong total, missing episodes, duplicate scoring coverage and malformed coverage; advice preserves old results and retains replay sources |
| Authoritative producer sets decide every budget path; ranked paths are refused | `test_authoritative_producer_refuses_ranked_path_and_normalizes_budget_paths` |
| The analysis and both producer forms bind to one certified snapshot | `test_same_alternate_certified_bytes_bind_analysis_and_both_producers`, `test_analysis_foreign_export_and_missing_hash_are_refused` |
| Integer-net thresholds avoid rounded chance ties and float subtraction errors | `test_nets_decide_exact_chance_tie_and_three_point_margin`, `test_exact_margin_uses_net_despite_float_subtraction` |

The tests use constructed world/oracle I/O for the CLI path, with the real
loader, membership, valuation, report and pool functions. They establish those
paths, not a live authenticated model retry or a complete experimental analysis.
The seed-reader checks include unclean receipts and receipts for another export
path. The experiment-producer tests vary each required field independently.
The measured-membership tests deliberately disagree with the generator's cause
units, roles and expected effects, including the known/new-type splits.

The full original 52-break review list is adapted in
`review-mutations-round6.json`, with targeted test selectors. S34–36 now mutate
the runtime `attempts` decision, S48 mutates conflicting `attempts` records,
and S33/S44 mutate the integer-net decisions. These retain the original breaks'
meaning; the obsolete record field and rounded-point decisions were removed.
Fresh isolated source trees, one per lane, share only the existing interpreter;
import-path receipts identify the code tested. At most three test processes run
at once. No Python environment is built in `/tmp`.

Round-6 evidence is under
`~/.cache/robots-probes/scoring-checks/round6-codex/`. The score-schema repair adds
record fields to the checker vocabulary, changing its identifier to
`3d619cd1f6c3343f696c090ca2f98950905ffaceeaf7f502ee567794f4a69623`.
Consequently old readiness receipts are refused. Fresh full audits are saved
in the new `certifications/` directory, without rewriting any world record.
Their clean receipts need to be discoverable in each world's `export-checks/`
before the unchanged README command can score those original roots.

The adapted mutation audit catches **52 of 52** original single breaks,
including **all 25** that survived round 5. None of that list remains uncaught.
The unchanged targeted control passes; every mutated run exits 1 with a test
failure. Results and per-break failing assertions are in
`mutations-2/results.json` and `mutations-2/S01.txt` through `S52.txt`; source
imports are recorded in each lane. The later classifier import adjustment only
accommodates the runtime helper's move between modules; its four relevant
regressions pass, and isolated typing against the lost-call runtime also passes.
No surviving break is excused as a known limit. Edited output binding (finding
9) remains the separate producer-provenance limit described above.

The three fresh audits are clean: w47's canonical export, w41's canonical
export, and w41's original review export. A read-only harness lets the unchanged
receipt validator discover those actual audit receipts in the new evidence
directory. It retains the checker identity, export path and digest checks; it
does not manufacture readiness. **These are audit-backed harness checks, not
successful standalone runs of the README command:** that command still refuses
the old world receipts. Installing the three clean receipts requires the
orchestrator's exception to the boundary on existing result directories.

The harness calls the normal CLI with the README's w47 arguments. At three
fixes, over 336 scoring attempts, the two analyses rescue net 69 and 73 attempts
(+20.54 and +21.73 fleet percentage points). Classical statistics rescues net
69 (+20.54), dramatic videos net 10 (+2.98), worst health net −1 (−0.30), and
no-data net zero (0.00). The achievable value is net 78 (+23.21). Claim 1 uses
six measured causes: both analyses find five of six; their precisions are five
of seven and five of five. The new reports are in `audit-backed-w47-1/` and
`audit-backed-w47-2/`, and explicitly record the harness qualification.

The w41 round-4 sources contradict the hand-over's claim that all producers
used the review export. Five genuine baseline records bind again to the
analysis's `export-review-20261009T173751Z` digest. The classical evidence
instead records the canonical export's `log/events.parquet` hash beginning
`f0f2294e`; the review export's hash begins `8fcc0055`. The complete original
source set is correctly refused. `w41-source-mismatch.json` records both full
hashes and the historical report. Scoring the other original sources in
`audit-backed-w41-original-remaining-final/` retains the matching bindings and makes
claims 2/3, their marks, headline and ratio unavailable because the original
oracle lacks full second-execution coverage. A new classical producer result
on the analysis-selected export, and oracle verification/report generation,
are needed outside this scorer's ownership. No historical record was rewritten.

Confidence is high in the guarded paths exercised by the regressions and the
recorded real-world bytes. Live model retry, a complete experimental condition,
and standalone checks against refreshed world receipts remain unverified.

The 132-test focused scoring suite passed. The initial 42 new regression cases
were also run against the original `8e28a01` scorer: 17 failed and 25 passed,
establishing failures before the repairs rather than only current-code passes.
The final six coverage regressions pass, including malformed coverage and the
new destination/source arguments; the w41 harness report was regenerated to a
new directory with that advice. Its clean certificates still validate against
the unchanged checker identifier above.

`make check` ran **once** and exited **2** at mypy: 12 optional-fixture typing
errors in the new tests. Those were corrected in `63cd9b7`. Subsequent full lint,
format and type phases exit 0 (400 Python files). The complete unit suite at that
commit exited **1**: **2,650 passed, 2 skipped, 65 deselected, 1 failed**, in
3,091.53 seconds. The sole failure is
`test_classical_statistics.py::test_evaluator_generic_root_and_informative_denominator`,
which expects primary confounders without supplying measured causes. The same
test fails against the untouched original scorer; its import path and failure
are recorded in `original-classical-evaluator.txt`. The caller/test needs repair
by its owner; the primary measured-cause gate was preserved.
No second `make check` was invoked. The advice/coverage changes after that full
suite have the six-case focused check and full lint/format/types passing.

The w41 input discrepancy remains open. Independent review,
receipt installation and the two owner handoffs remain open; this delivery
does not claim a green project gate. No simulator, GPU or new model job, merge,
push, historical-result rewrite or oracle/runtime source edit was performed.
