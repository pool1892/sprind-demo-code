# Classical statistics baseline

This is baseline 5 of the frozen experimental design. The runtime sees only
an observed export and the action catalogue. It never imports the generator,
fault registry, oracle or scoring code, and uses no language model or videos.
`evaluate.py` is a separate, explicitly truth-side measurement program.
`driver.py` is the explicit engine-side executor; neither is imported by the blind runtime.

## Reply compatibility: classical-frozen-4

`classical-frozen-4` differs from `classical-frozen-3` **only in how an experiment
reply is read and handled**. Its statistics, experiment selection, numerical update,
diagnosis rule and fix-set selection are unchanged. This integration repair starts
from `merge/batch-4` (`4042967`) on `fix/classical-reply-contract`.

The driver and updater use the tool server's `public_results.request_results_problems`
and constants, with no local status/message allow-list. Fixtures use its `COMPLETED`
and refusal codes. The capacity fixture now submits distinct, public-valid arms;
it still verifies exact job allocation, refusal at the capacity limit, and the
original midnight-overflow reason in the host log rather than in the public reply.

A valid request refusal stays recorded against the method's request budget. It
adds no arm measurements to an estimate; the CLI returns its
verified export for the updater. Driver `complete` means validated reply delivery;
`completed_requests` and `refused_requests` distinguish the outcomes. An engine
exception leaves no published reply and requires `--resume` with the original
inputs and checkpoint. The public `EXECUTION_FAILED` code is also treated as an
interruption, not a method refusal. Missing, queued, malformed or execution-failed
results cannot produce an updated answer. Resuming a delivered refusal does not
rerun the engine or manufacture successful arms.

The stored w42 replies predate this public contract and contain descriptive
messages. For comparison only, a **new copy** replaces each completed message with
the shared `COMPLETED` value, preserving requests, arm IDs, outcomes and all stored
originals. Frozen-3 rerun on the original replies reproduces all seven stored JSON
files byte for byte. Frozen-3 and frozen-4 reading the same translated copy produce
seven byte-identical files. Their diagnosis, ranked answer and four fix sets also
match the original six submitted files byte for byte. Against the original evidence,
the sole difference is `result_sha256`, the hash of the translated reply file; no
statistical value changes. This is an offline replay of the existing 960 arm
attempts, not a new experiment or an oracle evaluation.

The replay helper is `tests/classical_reply_replay.py`, invoked with the visible
reply export, saved logs answer, analysis directory and a new output directory.
Put either tag's `src` on `PYTHONPATH` to compare it, with
`OPENBLAS_NUM_THREADS=1`. [freeze4_verification.json](freeze4_verification.json)
records exact inputs, commands, hashes and checks. The 209 classical tests pass,
including all fifteen reported failures; the strengthened capacity host-log check
also passes separately. Mypy passes all 377 source files; targeted lint and format
checks pass. No existing substantive check was removed. No new worlds were opened, and
no GPU, model, simulation, oracle scoring, `make check` or `make smoke-world` was run.

## Run and outputs

```sh
env OPENBLAS_NUM_THREADS=1 uv run --locked python -m fleetlab.baselines.classical \
  /path/to/export --out /new/answer
uv run --locked python -m fleetlab.baselines.classical.evaluate \
  /path/to/world /new/answer --out /new/score
```

The first command reads the dictionary, description, episodes, events, health,
change log, maintenance, catalogue and cost tables. An explicit `--catalogue`
may supply the last catalogue separately. Only days 0–5 enter the regression;
ordinary rows outside that window are rejected, and experimental episodes are
excluded. Symlinks escaping the export are refused. `handed_over` counts as an
unsuccessful autonomous attempt, and does not abort the analysis.

Every answer directory is new. It contains contract-valid
`outbox/diagnosis.json`, `outbox/fix_set.json`, and
`fix_sets/{count-3,count-6,cost-4,cost-8}.json`. All four cuts follow the ranked
list, with the exported operator-hour costs. The ten ranked entries have
**singleton** expected gains in percentage points of fleet success. Their list
order is selected for joint gains, so singleton estimates need not descend.
`evidence.json` holds hashes, all candidate estimates, uncertainty flags,
label checks, exposure histories, model selection losses and joint forecasts.
No MCP provenance is fabricated for the direct computation.

## Statistical choices, fixed without world identifiers

1. **Outcome check before regression.** For each task, infer a failure horizon
   only when at least two robots and at least half the robots with failures
   agree on it; each robot votes when at least 60% of its failures end at
   the same step. Robots with no failures supply no endpoint information.
   A short failure ends before 95% of that horizon. Flag a robot only with
   at least five short failures, a
   majority of its failures short, and a one-sided Fisher test at familywise
   1% after multiplying by the number of robots. Replace suspect labels by
   the peer success probability among short episodes of that task, using
   Jeffreys half-counts. Name this hypothesis at robot scope, direction
   `changed`; the visible signature does not identify a hardware component.
   The former controller/degraded mapping was development-truth-informed and
   has been removed.
   Estimate the missed-success fraction as diagnostic metadata only.
   Gains target physical task success: correcting a recording fault does not
   itself improve a task outcome. Such a fault can therefore be named without
   attributing a positive physical repair gain.
   This is an explicit label-error sensitivity model: a short episode alone
   is not proof of success, and the uncertainty of those inferred labels is
   not fully propagated through the outcome model.
2. **Configuration history.** Reconstruct scoped old/new values from the
   observed change log, with robot overrides taking precedence over site and
   fleet settings. Infer initial inheritance when a local old value equals its
   parent's initial value; the export does not distinguish an explicit identical
   override. Encode every observed categorical level, with symmetric ridge shrinkage.
   Full encoding lets either level carry a nonpositive hazard under the survival
   model; choosing a modal reference must never forbid a beneficial reversion. A reversion restores the last old value at that exact scope.
   Never equate an unseen restoration value with the reference: its estimate
   is withheld and its interval marked uninformative. Constant exposures
   likewise supply no fitted contrast. This handles versions without assuming
   that numeric or lexicographic order measures quality.
3. **Health histories.** For each component/metric, estimate measurement noise
   from pooled absolute second differences: median × 1.4826 / sqrt(6), floored
   at 1e-6. Fit a constant level, a line, a step at each observed boundary, or
   a continuous hinge. A step needs at least one pre and two post readings; a hinge is considered with at least four readings. Every nonconstant history needs
   two observed departures in its fitted direction beyond two pooled noise
   scales, so a single late spike cannot establish persistence. Student-t(4)
   residual likelihood plus `(df-1)*log(n)` selects the shape; scanned onsets additionally pay `2*log(max(1,n-2))`. Maintenance
   dates are inspected boundaries, not assumed successful repairs. Thus a
   late step is not spread back over healthy days, and a noisy robot does
   not determine its own noise threshold. The reference is the median fitted
   initial reading, with scale max(1.4826*MAD, pooled noise, 1e-6). Adverse
   exposure is `log1p(max(adverse_z-1,0))`, without a hard saturation ceiling.
   Grip force is adverse downward; calibration residual, torque RMS and CPU
   load upward. Cycle count is cumulative usage and is excluded. These are
   declared engineering priors; **unknown metrics use both tails**, not an
   assumed upward direction. Selected trends extrapolate to day 8.5; a flat
   forecast at the last reading supplies a sensitivity envelope.
4. **Outcome model and validation.** Compare penalized binomial logistic and
   multiplicative-survival models. The latter is
   `p=logistic(task/site/robot/day baseline)*exp(exposure effects)` and constrains
   exposure coefficients nonpositive: an explicit restoration prior. Logistic
   coefficients may have either sign. Base penalties are intercept .01, task
   1, site 8, robot 16, day 8, configuration/condition .5, health 1. Compare
   multipliers .25, 1 and 4, leaving the intercept unchanged. Choose minimum
   held-out log loss across five deterministic robot-day folds (first four
   SHA256 bytes modulo five); exact ties prefer stronger regularization.
   These seven relative penalties were retained from v1, not oracle-tuned.
   They impose exchangeability assumptions on unexplained site/robot effects;
   a persistent exposure exactly collinear with a unit cannot be causally
   identified without those assumptions. Predictions use each robot's observed
   task distribution, persistent configuration and day-five common time effect.
5. **Uncertainty and diagnosis.** Default 200 positive exponential multiplier
   draws, seed 0, multiply at site, robot and robot-day levels and normalize
   to the observed sample size. No exposure cell is ever omitted. This adds
   between-unit uncertainty for static exposures and day uncertainty for
   transitions, but cannot identify arbitrary unit confounding. The 5th–95th
   percentiles envelope drift and flat forecasts. Add a conservative finite
   scoring-panel envelope: the design has two scoring days against six
   observed days, and counterfactual Bernoulli discordance uses its maximum
   feasible variance **only on rows changed by that action**, capped by the
   affected fleet population. Bounds use an accurate sum and eight outward floating-point ulps to avoid excluding an exactly representable population fraction through summation order. Unaffected robots contribute zero repair variance. These remain model-conditional ranges, not guaranteed
   90% causal coverage. Complementary levels of the same categorical key
   are one contrast, not aliases. Other aliases with |correlation| ≥ .98 and unresolved causes
   get explicit ignorance ranges; report their coverage separately.
   Name candidate effects ≥ .5 fleet points using Benjamini–Hochberg at .10
   on bootstrap negative-tail fractions. These are conditional stability
   scores, not calibrated causal p-values. Shared drop signatures and label
   checks use their separately multiplicity-controlled tests.
6. **Onsets and fleet changes.** Scan all five day boundaries at every robot,
   site and fleet scope using task-stratified CMH scores, with Bonferroni over
   all scopes × five boundaries. Retain every cut, not only the strongest.
   For a logged edit followed by a distinct significant later decline, check
   the earlier period separately: no earlier evidence (`z > -1`) withholds
   the edit in ranking and diagnosis together. A second break does not itself
   veto an earlier cause. A fleet edit has **no per-site veto or minimum
   effect threshold for ranking**. Its gain now uses a local pooled level contrast:
   last two days before the release versus first two days at/after it, truncated
   at observed boundaries, with unpenalized robot and shared task effects and observed robot weights.
   Two days is a fixed noise/locality tradeoff, chosen before scoring; it excludes
   remote declines rather than assigning their effect to the release. The step is estimated after projecting out unpenalized robot/task effects. This permits rotating tasks without requiring every robot/task cell on both sides. Noise pools Bernoulli rates by task/phase with Jeffreys half-count smoothing; the residualized step gives its sampling variance, and 200 normal draws by default represent uncertainty. A time effect exactly aliased with robot/task composition remains unidentified. This assumes shared task effects and a constant local step; there is no site-significance veto or nuisance penalty.
   A common final-success log-odds shift transports this estimate into the
   regression forecast for all joint cuts. The regression coefficient is not
   used as a second independent piece of release evidence. Other actions retain
   their fitted feature-union effects; multiple release shifts add on log odds.
   Diagnosis uses the same .5-point/BH screen as every identified candidate.
   There is no positive minimum effect for ranking; five points is the smallest
   planted effect checked, not a guaranteed detectable effect.
   Other causes must be stable **inside this local window**: same-day local
   faults, common shocks, and a decline one day before the release remain
   confounded. A fleet has no untreated time comparator. The task-standardized
   robot heterogeneity statistic remains descriptive metadata, with pooled
   phase/task Bernoulli noise; it never vetoes an effect.
   Reserve the highest-estimated logged fleet change for experimentation,
   including apparently inert changes. This does not cover every fleet change.
7. **Unexplained failures and event signatures.** Negative site/robot effects
   nominate otherwise unsupported actions: allocate a site's unexplained
   deficit equally among the distinct condition keys in its action catalogue, or a robot's among
   its distinct candidate components (two camera verbs share one component
   allocation and their latent repair union is counted once). This is a stated equal-cause prior, not measured repair
   identification. Task-adjusted incidence contrasts test every event kind
   except grasp/release commands, Bonferroni across scopes and kinds. Excess
   drops significant at .05 across at least two robots shift the site prior
   to 75% object batch and 25% shared by other available conditions (100% if
   it is the sole available condition; equal allocation if it is unavailable) and support a
   `changed` condition diagnosis. Events are downstream symptoms; they are
   not put into the outcome regression as causal covariates. Conditions are
   restored to their observed modal reference. Unchanged keys are never
   padded in as reversions. Camera calibration residuals map to recalibration;
   other image-health readings map to cleaning. Both verbs remain candidates,
   and other components use service. This mapping is an engineering hypothesis;
   logs need not distinguish cleaning from replacement.
8. **Joint decisions.** Search all triples of candidates for maximum joint
   predicted gain, with total cost then a fixed SHA256 lottery over action JSON
   (`classical-ties-v2:` prefix) as deterministic ties. Equal-prior conditions
   therefore have no alphabetical priority.
   Next find the best three additions preserving that triple. Order within
   each group by marginal joint gain and fill to ten in the same way. Apply
   repairs to the same nonlinear forecast, combining overlapping feature
   changes only once. Signed threshold increments with identical action
   memberships compact the exact union calculation; this is an execution
   optimization, checked against direct unions on noisy signed overlapping
   fixtures for both links and sets of one, two, three and six actions.
   This captures masking. Optimal sets at different
   budgets need not nest: exact k=3 is the priority; k=6 is exact conditional
   on that triple, and hour budgets are the contract's cuts, not independent
   knapsack optima. No claim of simultaneously globally optimal nested sets. This prioritization
   resolves the mathematical conflict between globally optimal non-nested sets
   and the scorer's single-list contract.
9. **Warnings.** An adverse selected health trend with at least .9 bootstrap
   support for an adverse outcome coefficient triggers a forecast check over
   the last observed day through the end of scoring (days 5–9 here). Emit a
   `failure_rate` warning only at a **future** five-point fleet-relative crossing
   prevented by freezing that particular health trend at its last reading.
   An already elevated robot supplies no positive lead time and no warning.
   Evidence records issue day, predicted crossing and lead days. Warning confidence is the smaller of
   trend-sign confidence and coefficient stability. No retraining claims are
   fabricated from unresolved failures.

## The experiment condition

```sh
uv run --locked python -m fleetlab.baselines.classical.experiments plan \
  /path/to/export /existing/logs-answer --analysis /new/analysis
tools/heavy-run uv run --locked python -m fleetlab.baselines.classical.driver \
  --queue /new/analysis/queues/session-01.json --truth /world/files/world_truth.json \
  --root /new/engine-output --workers 10 --batch-size 4 --precision fp16_res32
uv run --locked python -m fleetlab.baselines.classical.experiments update \
  /path/to/reply-export /existing/logs-answer --analysis /new/analysis --out /new/ab-answer
```

The rule is fixed before results: reserve the highest-estimated logged fleet
change when one exists, then all catalogue conditions at the site with the
strongest significant decline alarm, falling back to greatest summed latent gain
when no site has an alarm. Fill unused slots from the ten-entry joint ranking, then remaining singleton
gains if capacity skips exhaust that list, skipping
duplicates and any request exceeding the public limit of 160 arm episodes per
robot/day. A plan records every allocation and skipped candidate; requests use
sorted robot IDs so planning and engine allocation agree. Eight requests, two arms,
60 attempts per arm, paired starts, affected robots only. `--requests` and
`--episodes-per-arm` permit a smaller **explicitly labelled pilot**, not an
unreported substitute for the full condition. Each request states the
conservative 80%-power, two-sided 5%-level unpaired Bernoulli detectable effect
`2.80*sqrt(.5/n)` within affected robots (25.6 points at n=60).
Task/start slots match, but scene and noise seeds differ. This does not imply independent outcomes: shared task/robot/start difficulty can correlate both arms. The planning detectable-effect calculation remains a conservative independence benchmark, not measured power.

`plan` calls the same `ToolBox.request_experiment` and
`Analysis.write_queue` as agents. The external `engine.experiments.run_queue`
runs both arms through `tools/heavy-run`; use a detached checkout and a new
engine output root. Its reply is checked by `arm_problems`, recorded with
`Analysis.record_results`, and supplied to `update`. The blind program never
loads `WorldTruth`, starts a simulator or discovers a world directory.

Updates validate matched robot/task identities, experiment IDs and the two-day
window following the observed boundary (public design sections 1 and 7). Raw
arm means are standardized by observed robot weights. Use 10,000 Dirichlet draws per robot over the four matched outcome pairs (both fail, rescued, broken, both succeed), with half a count in each cell. This estimates covariance rather than imposing independence or perfect pairing. Perfect agreement still has nonzero uncertainty. Different robots are treated as independent sampling strata; the short panels cannot estimate every shared temporal correlation.
Combine measurement and observational mean by inverse variance; observational
variance is the bootstrap variance of the mean effect, excluding future-panel
noise. Unidentified assignments and collapsed variances (≤1e-8 points²) have
zero observational precision. If two informative estimates differ by more than
1.96 joint standard deviations, inflate observational variance to at least the
squared discrepancy minus measurement variance, then precision-pool: randomization
governs a large conflict instead of inheriting false observational precision.
For an unidentified candidate, an experiment whose 90% interval includes zero
may lower but cannot raise its hypothesis-prior point. This declared pretest
safeguard prevents noise alone promoting unsupported candidates; the unscreened
precision mean and interval remain in evidence. It is a screened estimate,
not an unbiased mean or posterior expectation. Finite-panel uncertainty is added
only on affected rows to the final interval in both conditions. The exact
oracle proxy has zero measurement variance and is explicitly labelled.
A bounded logistic joint forecast is calibrated to **pooled** singleton gains
and re-optimized. **Exactly the same naming rule applies in both conditions**:
current expected gain at least .5 points, BH .10 on current negative-tail
probabilities, with the separately screened signature route for untested
symptoms. Testing replaces that action's signature shortcut with its pooled
repair evidence. Existing diagnoses are re-screened and take the strongest
currently supported confidence among their target's repair verbs. A precise
contradicting test can withdraw a cause; a weak nonsignificant test does not
by itself erase strong logs. A null cleaning result cannot erase separately
supported recalibration. No stale pre-experiment confidence is retained.
These remain model-conditional confidence scores; the conflict rule cannot
detect bias smaller than the joint uncertainty.
Transport from days 6–7 to days 8–9, and the joint log-odds interaction model,
remain assumptions. Effects at boundaries may be clipped in the joint forecast;
the measured singleton stays unchanged and clipping is recorded.

`evaluate --oracle-proxy` applies the same fixed selection/update rule to
noise-free singleton oracle lookups **in the scoring window**. It creates no
fake arm rows or queue results. This optimistic diagnostic is not a real
with-experiments condition and is never an out-of-sample A/B score.

## Third-freeze repair checks and measurement protocol

This round uses planted cases to choose repairs and consults the w41–w43
oracle only once, after the final executable-code commit. All three worlds
remain **in-sample**, including every stability figure: earlier development
used their results. No replacement world is opened before the orchestrator's
`NEW WORLDS READY` gate, after tagging `classical-frozen-3`.

The planted rank fixture has 12 robots, three sites, 1,008 noisy Bernoulli
attempts, four logged 40-point robot faults from day 1, and three unlogged
40-point robot faults starting before the release. An inert day-5 release
competes with the four genuine repairs. A real release from day 3 reduces
success by 5, 10, 20 or 40 points, with probabilities floored at zero; the
unlogged faults then start at day 1. Twelve predetermined seeds per effect
exercise ranking, not just positive gain or experiment reservation. The
smallest effect must reach the top three in most fits, while the inert release
must stay out in most fits. An independent invariance test changes every
outcome in remote prehistory and requires an identical release contrast.
This demonstrates separation of earlier declines, not identification against
coincident shocks. The two-day bandwidth is not selected from world scores. A preliminary visible-only run exposed missing robot/task cells under rotating schedules; its incomplete answers are preserved under `classical-stat-v4/measured-75295b5`. Before any oracle scoring, a pooled unpenalized robot/task contrast replaced the all-cells requirement and was checked on synthetic rotations.

The warning rule is unchanged. **This baseline gives no warnings for drifts
already above its five-point threshold at day 5.** It therefore gave no warning
for either real drifting fault in the old worlds and earns no lead time there.
A noisy prospective planted crossing verifies positive lead time; it is not
evidence of warning sensitivity to already-elevated faults.

The real experiment update reuses the immutable w42 arms from
`classical-stat-v3/engine-w42-9a39de8/exports/ab-w42-9a39de8/session-01` and the
associated analysis queue. That earlier end-to-end run completed 960 arm
attempts with 96/96 replay checks identical; no new GPU job, simulation,
smoke-world or model session is run in this round. The new selection is compared
with the stored requests and any mismatch is reported, not filled with invented
arms. Oracle-proxy results use exact scoring-window lookups and are explicitly
not real A/B results.

Stability uses 100 episode resamples within each robot/day, with health and
change histories fixed, seed 20261009 and 200 uncertainty draws per fit. Each
resample chooses its own proxy requests. Reused real-arm stability holds those
arms fixed, so excludes repeated experimental noise and request-selection
uncertainty. This distinction matters especially on w43, whose apparently
perfect earlier canonical share ranged from 46.2% to 100% under resampling.

Historical records are retained in [freeze2_results.json](freeze2_results.json),
[freeze2_planted.json](freeze2_planted.json), and
[freeze2_verification.json](freeze2_verification.json), with older results in
[revision_results.json](revision_results.json). Those tables describe earlier
methods and must not be quoted as the third freeze's performance.

The unchanged reviewer probes also meet their original guarantees: its inert
release is in the top three in **1/12** fits (previously 9/12), while its real
five-point release remains in the top three in **8/12**. These are noisy
synthetic rankings, not oracle scores. The independently generated four-real-fix
fixture passes the majority criterion for 0, 5, 10, 20 and 40 points.

[freeze3_planted.json](freeze3_planted.json) records these probes and **22/22**
targeted removal checks caught by the added tests. This rechecks the formerly
surviving choices and four new repairs; it is not a rerun of the reviewer's
entire 45-removal suite. Tests cover both signs of arm covariance, precise
contradictions versus weak null tests, robot-only label scope, shared naming
multiplicity, onset isolation, joint transport, the interval envelope and
outward physical bounds. The focused check peaked at 331,864 KiB RSS.

[freeze3_validation.json](freeze3_validation.json) repeats the final pooled
method's synthetic checks: 181 focused tests pass and all 22 targeted removals
are caught. On the independent fixture, the inert release enters the top three
in 1/12 fits and each genuine 5–40-point release in 12/12. The reviewer fixture
remains 1/12 inert and 8/12 genuine five-point. Thirty noisy rotating-task panels
retain an identifiable five-point effect; a completely confounded task/time
schedule stays unidentified. These controls concern the final method; the
preliminary record above is retained to make the correction auditable.

## Final measurements for classical-frozen-3

All rows below are **in-sample**: w41, w42 and w43 were development worlds. This final
method was selected using synthetic planted cases and code review, without fresh oracle
scores during this repair round; these worlds were scored in one batch after the final
runtime commit. Earlier rounds did use their oracle scores. Each world has 1,008 observed
attempts and 336 scoring attempts. The recorded executable source is `2acc093`; later
commits add measurement records only.

Recall is matched detectable causes / detectable causes. Precision is matched causes /
(matched causes + false positives), with the scorer’s small-effect exclusions. Rho is
Spearman correlation between predicted and true singleton gains over ten valued actions.
Share is the joint top-three gain divided by the oracle’s best gain at k=3. The oracle proxy
supplies eight noise-free scoring-window singleton effects; it is not a real A/B experiment.
The stored-real w42 row reuses the earlier eight requests with 60 attempts per arm (960
simulated attempts); this round runs no simulator.

| World | Condition (in-sample) | Recall | Precision | Rho | k=3 gain / achievable points | Share | Rescued / broken |
| --- | --- | --- | --- | --- | --- | --- | --- |
| w41 | logs | 6/6 | 6/6 | 0.957 | 19.94 / 19.94 | 100.0% | 67 / 0 |
| w41 | oracle proxy | 6/6 | 6/6 | 0.957 | 19.94 / 19.94 | 100.0% | 67 / 0 |
| w42 | logs | 3/7 | 3/4 | 0.094 | 11.01 / 30.95 | 35.6% | 43 / 6 |
| w42 | oracle proxy | 4/7 | 4/5 | 0.735 | 30.95 / 30.95 | 100.0% | 104 / 0 |
| w42 | stored real A/B | 4/7 | 4/4 | 0.680 | 30.95 / 30.95 | 100.0% | 104 / 0 |
| w43 | logs | 6/7 | 6/9 | 0.644 | 19.35 / 19.35 | 100.0% | 67 / 2 |
| w43 | oracle proxy | 6/7 | 6/9 | 0.863 | 19.35 / 19.35 | 100.0% | 67 / 2 |

Chance draws three actions from the catalogue using the scorer’s random baseline.
Unavailable fresh replays are omitted and counted; the denominator is valued / requested
draws.

| World | Chance mean share | Chance 5th–95th | Draws |
| --- | --- | --- | --- |
| w41 | 1.6% | 0.0%–0.0% | 199/200 |
| w42 | 1.4% | 0.0%–8.0% | 200/200 |
| w43 | 1.5% | 0.0%–16.9% | 199/200 |

**Stability:** 100 episode resamples within robot/day per world, seed 20261009; health
readings and changes stay fixed. Every refit uses 200 uncertainty draws. These are
distributions of the selected set’s true k=3 share, not confidence intervals for a causal
effect. Proxy actions are reselected for each refit. Real w42 stability reuses the one
measured arm sample, so it does **not** measure repeated-experiment variability or request
reselection under fresh arms.

| World | Condition (in-sample) | Mean share | 5th–95th | Min–max | Oracle-optimal / valued |
| --- | --- | --- | --- | --- | --- |
| w41 | logs | 99.3% | 98.5%–100.0% | 85.1%–100.0% | 68/100 |
| w41 | oracle proxy | 99.4% | 98.5%–100.0% | 85.1%–100.0% | 68/100 |
| w42 | logs | 35.0% | 35.6%–35.6% | 11.5%–38.5% | 0/100 |
| w42 | oracle proxy | 79.5% | 38.5%–100.0% | 38.5%–100.0% | 59/100 |
| w42 | stored real A/B | 99.8% | 100.0%–100.0% | 76.0%–100.0% | 99/100 |
| w43 | logs | 85.1% | 46.2%–100.0% | 46.2%–100.0% | 72/100 |
| w43 | oracle proxy | 85.6% | 46.2%–100.0% | 46.2%–100.0% | 71/100 |

Calibration compares singleton effects in fleet-success percentage points. Informative
intervals exclude explicitly unidentified assignments and aliases; all-action coverage
includes them. Median interval width refers to informative entries only. The finite-panel
envelope is conservative and model-conditional, and these selected in-sample entries do not
establish nominal coverage. Exact proxy coverage is partly tautological.

| World | Condition (in-sample) | All coverage | Informative covered / total | Informative rho | MAE points | Slope | Median width points | Confounders resolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w41 | logs | 100.0% | 8/8 | 0.976 | 0.378 | 1.041 | 3.60 | 0/0 |
| w41 | oracle proxy | 100.0% | 10/10 | 0.957 | 0.170 | 0.999 | 2.43 | 0/0 |
| w42 | logs | 90.0% | 7/8 | 0.228 | 2.039 | 0.195 | 6.60 | 0/1 |
| w42 | oracle proxy | 100.0% | 6/6 | 0.986 | 0.694 | 0.964 | 3.56 | 1/1 |
| w42 | stored real A/B | 100.0% | 7/7 | 0.964 | 1.399 | 1.185 | 4.11 | 1/1 |
| w43 | logs | 100.0% | 9/9 | 0.583 | 1.293 | 0.914 | 5.38 | 0/1 |
| w43 | oracle proxy | 100.0% | 9/9 | 0.883 | 0.708 | 1.085 | 5.05 | 0/1 |

Other budget shares use each budget’s own achievable gain. Cost limits are exported
operator-hours. Warning counts measure emitted warnings only; this scorer has no claim-5
warning accuracy or lead-time score.

| World | Condition (in-sample) | k=6 share | 4-hour share | 8-hour share | Decoy rejection | Warnings |
| --- | --- | --- | --- | --- | --- | --- |
| w41 | logs | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| w41 | oracle proxy | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| w42 | logs | 37.0% | 72.9% | 46.2% | 100.0% | 0 |
| w42 | oracle proxy | 87.0% | 75.7% | 89.4% | 100.0% | 0 |
| w42 | stored real A/B | 85.5% | 75.7% | 89.4% | 100.0% | 0 |
| w43 | logs | 81.7% | 97.3% | 89.9% | 100.0% | 0 |
| w43 | oracle proxy | 81.7% | 97.3% | 89.9% | 100.0% | 0 |

[freeze3_results.json](freeze3_results.json) contains every summary, all 300 resampled
decisions and values, canonical diagnoses and action lists, file hashes, request comparisons
and measured arm statistics. Logs-only informative coverage is 24/25; coverage including
ignorance ranges is 29/30. Proxy informative coverage is 25/25; stored-real w42 coverage is
7/7. These small, selected in-sample sets do not establish nominal coverage. Calibration
slope regresses true singleton gain on predicted gain. The designated decoy counts are 1, 2
and 1 for w41, w42 and w43; inert edits are separate false positives.

**Remaining failures.** W42 logs still attribute the site-3 placement deficit to an inert
sampling edit, missing placement, the milder site-2 batch effect and r12's camera
degradation. The r05 recording anomaly remains a robot-only hypothesis: the scorer treats it
as vague, not a recovered controller fault. The local fleet contrast removes the inert
release from the canonical top three, but it still enters in 4/100 resamples. Within-window
time changes remain a causal confounder, and the local contrast assumes shared task effects.
The canonical logs triple rescues 43 scoring attempts and breaks six, for 37 net gains
against an achievable 104; the resampling mean share is 35.0%.

The stored w42 sampling measurement is -0.56 fleet points, with its paired 90% interval
-2.88 to 1.71. Its pooled confidence is 0.390, and the common naming screen withdraws the
cause. The observed and randomized evidence now resolve that designated confounder. The
proxy also resolves it, but re-screening all current evidence admits an untested r10 arm
association at confidence 0.91, a false positive. Conditional stability scores are not
calibrated causal probabilities; re-screening can change an untested hypothesis's
multiplicity threshold. No code was changed after these scores were read.

On w43, the drop signature points to batch at the wrong site, an arm-health association
names an unaffected arm, and an inert sampling edit remains named. The milder true batch
effect is missed. Its canonical optimal triple occurs in 72/100 logs resamples and 71/100
proxy resamples, with a 46.2% minimum share in both. That sensitivity is retained alongside
the 100% canonical result.

**Stored real arms and limits.** The final planner selects the same eight w42 actions as the
existing real run, in a different request order. The 960 arm attempts and their original
timing/noise are reused, not regenerated in the new order. The 99.8% mean real-arm stability
holds those data fixed; the proxy instead reselects its requests in every resample (site 3
in 59/100 and site 1 in 41/100). It is not evidence that noisy experiments outperform the
oracle. Repeated experimental noise, fresh request-order execution, real A/B on w41 or w43,
and replacement-world performance remain unmeasured here. No warning is emitted for the old
worlds; their real drifts have no warning lead time under this prospective-crossing rule.

**Verification.** [freeze3_verification.json](freeze3_verification.json) records one `make
check`, exit 0: 1,264 passed, two skipped, 62 deselected, 291.85 seconds. The final focused
statistical suite passed 181 tests; the subsequent rotating-task coverage check passed as
well. The final 22-removal audit passed; earlier preliminary records remain preserved. At
most two fitting workers ran together; their peak RSS values were 390,848, 454,800 and
467,720 KiB for w41–w43. When jobs overlapped, combined systemd memory ceilings were 3,900
MiB, below 4 GiB. No new simulator, GPU, model session, check-sim or smoke-world run was
started. Runtime Python source remains identical to `2acc093`; later commits only record
results and verification. All changes stay within the classical package and its tests.

## Third-generation plumbing repair (2026-10-11)

The frozen statistical method, thresholds, selection and blind outputs are unchanged.
The offline evaluator passes `oracle/measured_causes.json` through the scorer's identity,
hash and frozen-rule reader. Summary `claim1_basis` says `measured causes` when that record
exists, or `descriptive curves` when it is absent. An incomplete record stays unavailable;
it never silently falls back. `claim1_missing` retains the reason. Public-name answers
are bound through the scorer's certified producer-input check.

The truth-side driver reads the world's ID from its records and checks that the queue's
export digest matches the world's currently certified export. The public fleet name is
checked separately and can differ from the world ID. Foreign bytes are refused even
when the public name matches. The original queue is preserved as `queue.json`; only
`engine-queue.json`, a separate engine snapshot, substitutes the recorded world ID.
Both snapshots and input/source receipts are checked across execution and resume.

Planning requires an existing logs-only answer, not a new empty answer directory.
The request list is the queue's `experiments` array, not a `requests` field.
Synthetic evaluator and driver tests cover these contracts; synthetic certification
receipts prove byte binding, not leak-audit coverage. The w47 dry start stops before
`fleet.run` executes; it proves real-input planning, not GPU execution or completed
experiment results. No simulator engine, scoring module or frozen reference is edited.

The detached checkout `/home/cs/.cache/robots-classical5-run` at `86388c8` reached
`fleet.run` with eight requests and 960 planned arm jobs, first job `e001681`.
A temporary external harness intercepted that call: zero GPU jobs executed.
The original w47 export lacked a current certification receipt. Its byte-identical
copy under `/home/cs/.cache/robots-probes/classical-gen3-86388c8/w47` passed the real
checker using the original observed run; the original world was not modified.
Evidence and dry-start harnesses are under that new results directory.

To execute later from that detached checkout, these fish variables choose new
analysis and engine outputs; `L` is the existing logs-only answer:

```fish
set W $HOME/.cache/robots-probes/classical-gen3-86388c8/w47
set A $HOME/.cache/robots-probes/classical-gen3-86388c8/w47-live
set L $HOME/.cache/robots-probes/analysis-fresh/w47-classical-exp/answer
uv run --locked python -m fleetlab.baselines.classical.experiments plan "$W/export" "$L" --analysis "$A/analysis"
uv run --locked python -m fleetlab.baselines.classical.driver --queue "$A/analysis/queues/session-01.json" --truth "$W/files/world_truth.json" --root "$A/engine" --workers 6 --batch-size 4 --precision fp16_res32
uv run --locked python -m fleetlab.baselines.classical.experiments update "$A/engine/exports/analysis/session-01" "$L" --analysis "$A/analysis" --out "$A/ab-answer"
```

The driver itself invokes `tools/heavy-run`; the later run must wait for the GPU.
The updater must wait for the driver's successful completion. Neither completed arms
nor the update have been measured in this repair. No `check-sim` or GPU smoke run
was attempted under this assignment's dry-start-only boundary.

Final full verification: `TMPDIR=/var/tmp/codex-classical-tests
OPENBLAS_NUM_THREADS=1 make check` exited 0: 2,659 passed, two skipped,
65 deselected, one dependency SyntaxWarning, 3,540.42 seconds of pytest.
Ruff lint and formatting passed; mypy checked all 400 files successfully.
The first full-check invocation exited 2 at mypy before pytest because a new test's
nested dictionary needed an explicit `dict[str, Any]` annotation; `aa6adba` fixes it.
The successful full-check log is `make-check-final.log` in the new evidence directory.
The focused driver suite passed 54 tests; the evaluator contract passed all three
record states. Independent review, GPU arms, final update and GPU smoke remain open.
