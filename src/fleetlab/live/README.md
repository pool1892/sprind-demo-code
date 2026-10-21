# Telemetry risk estimator

**This experiment does not establish useful early warning beyond elapsed
time, or an operating point with substantially earlier calls and few false
alarms. Keep timeout as the reference.** Most failures terminate at the known
280-step limit. Near-perfect aggregate discrimination largely identifies
that deadline, rather than a failure developing in the robot. In the main
5-second transition stratum (steps 160–200), telemetry loses to elapsed time.
It has higher AUROC in two sparse strata, but worse probability error in
every stratum; those exceptions do not establish an early-warning advantage.

The deterministic CPU estimator and supervisor plumbing work. After the
review correction described below, the calibration-selected 5-second
threshold is **0.47**: it calls on 181/181 held-out failures, with 17 false
calls in 420 episodes (17/239 successes; 7.11%) and 6.15 seconds median lead.
Timeout calls on the same failures with 15 false calls and 5.30 seconds lead.
Telemetry's 5-second Brier error is 0.0407, versus 0.0224 for timeout's
probability forecast and 0.0074 for time only. More lead alone does not prove
that a supervisor can rescue an attempt or save operator time.

Confidence is high in the recorded counts and CPU path checks, but low in
additional predictive value on new robots. Only three robots in one development
world supply these test results, and **the test numbers were looked at once
before this fix**. This is a post-review correction, not a fresh blind
validation. It is an implementation README, not the gate-4 report or claim 4.

## The timeout artefact and the measured tables

The measurement files in `docs/results/live-risk/` contain all horizons,
reliability tables, raw metrics, the selection receipt and CPU benchmark;
large experiment files no longer ship in this package.
The runtime models are
[telemetry](results/telemetry.json) and [time only](results/time_only.json).

D2 (`w42`) was read after its runner reported ready. The unchanged split has
840 fitting episodes (140 failures), 420 calibration episodes (258 failures)
and 420 test episodes (181 failures; 83,528 command rows). Of the test failures,
180 end at step 280; the other ends at step 224. Test successes end at median
step 137, 90th percentile 161 and maximum 247. Thus every failure row from step
180 is positive for the 5-second label, while most successes have already ended.

The 5-second time-stratified comparison is below. AUROC measures rank ordering
of positive versus negative command rows (higher is better); NA means one
class is absent. Brier is mean squared probability error (lower is better).
The sparse early stratum's 36 positive rows all belong to the single early
failure. The 200–240 stratum is already 98.2% positive. These are not convincing
evidence that telemetry spots developing failures earlier than a clock.

| Steps, upper bound excluded | Positive / rows | Telemetry AUROC | Time-only AUROC | Telemetry Brier | Time-only Brier |
|---|---|---|---|---|---|
| 0–40 | 0/16,800 | NA | NA | 0.0083 | 0.0000 |
| 40–80 | 0/16,800 | NA | NA | 0.0090 | 0.0000 |
| 80–120 | 0/15,985 | NA | NA | 0.0208 | 0.0000 |
| 120–160 | 36/11,620 | 0.6588 | 0.5960 | 0.0845 | 0.0031 |
| 160–200 | 3,640/7,761 | 0.9371 | 0.9749 | 0.1709 | 0.0575 |
| 200–240 | 7,224/7,355 | 0.7548 | 0.6738 | 0.0449 | 0.0176 |
| 240–280 | 7,200/7,207 | 0.8704 | 0.9125 | 0.0189 | 0.0010 |

The aggregate AUROC belongs beside that qualification, not in the headline:

| Horizon | Telemetry Brier | Time-only Brier | Telemetry AUROC | Time-only AUROC | Telemetry threshold |
|---|---|---|---|---|---|
| 1 s | 0.0063 | 0.0010 | 0.9990 | 0.9991 | 0.14 |
| 2 s | 0.0162 | 0.0021 | 0.9996 | 0.9995 | 0.30 |
| 5 s | 0.0407 | 0.0074 | 0.9968 | 0.9987 | 0.47 |

The 5-second telemetry Brier's descriptive 95% robot-bootstrap interval is
0.0228–0.0540; false calls per episode span 0.0214–0.0643. Resampling only three
robots cannot establish new-world generalization, and perfect observed recall
makes its bootstrap interval degenerate. The selected successful-episode
alarm cap was 5% on calibration (8/162), but it rises to 7.11% on test.

The miniature fleet is an external transfer check: 416 episodes, 157 failures,
80,677 command rows, and no miniature fitting or threshold selection. At 5 s,
telemetry catches 157/157 with 45 false calls and 6.45 s median lead; timeout
catches 157/157 with 37 false calls and 5.30 s lead. Their Brier scores are
0.0622 and 0.0372; time only scores 0.0287. D1 (`w41`) was not ready at the
original fit and remains unmeasured here. Its runner's aggregate outcome
summary was inspected later while monitoring the queue, after this model was
frozen; no D1 features or predictions were evaluated. Future D1 work must not
describe those aggregate outcomes as unseen.

## What the review correction changed

The first implementation subtracted measured end-effector position from the
commanded position. That is invalid when a command is a relative delta.
Calibration robot r11 uses relative control under the visible control-mode
configuration. Its 39,200 rows were 41.5% of calibration; its original raw
5-second probability averaged about 0.010 against a label mean of 0.357.
The pooled sigmoid compressed the scores (slope 0.247), lifting healthy scores
and producing the misleading original 0.51 threshold and 0.0843 test Brier.
This was a concrete feature bug, not merely unexplained distribution sensitivity.

The correction removes `position_error`, its window mean and its late-time
interaction. It does not infer control mode from hidden truth or add a new
telemetry field. The remaining command-speed feature measures changes between
commands, without subtracting an absolute observed position. This avoids the
invalid pose-error calculation; it does not guarantee all features transfer
across control modes.

The same source-run SHA-256 digests and the same robot groups were used to
rebuild features, refit all heads and sigmoid calibration, and reselect
thresholds on calibration robots only. No test outcome enters the fitting or
threshold objective. Nevertheless the defect was found after test inspection,
so these revised test results must not be described as untouched validation.
The former models and measurements are preserved by commit b9f557f.

The revised 5-second sigmoid slope is 0.4701, intercept 1.0073. Calibration
still compromises between r11 and the other robots: r11 remains poorly
predicted before calibration, while calibration worsens Brier on r03/r07.
The feature correction reduces the problem; it does not solve model misspecification.
Here raw means the uncalibrated head; final includes sigmoid calibration and
the nondecreasing-horizon adjustment:

| Calibration robot | Rows | Label mean | Raw mean | Final mean | Raw Brier | Final Brier |
|---|---|---|---|---|---|---|
| r03 | 31,628 | 0.259 | 0.229 | 0.362 | 0.0302 | 0.0507 |
| r07 | 23,630 | 0.152 | 0.152 | 0.264 | 0.0036 | 0.0303 |
| r11 | 39,200 | 0.357 | 0.061 | 0.227 | 0.2812 | 0.1781 |

Fit plus calibration of three telemetry heads took 2.79 s
on one BLAS thread. The revised benchmark uses the first 32 sorted episodes in
`w42-observed`: 5,440 complete updates, median 38.5 microseconds,
95th percentile 44.0 microseconds. Offline versus streaming probability
error is at most 3.33e-16. Shared-machine timings are descriptive.

## Labels, features and separation of robots

At command step `t`, the label at horizon `h` is one exactly when the
unassisted attempt fails at terminal step `T` with `t < T <= t+h`.
One second is 20 steps. True placed-check verdicts supply labels; terminal
rows have no command and are excluded. Successes have zero labels at every
horizon. Early LIBERO-only endings use their actual terminal step. This
predicts terminal failure, not the instant of a drop or an irreversible fault.

There are 38 causal features: elapsed step; end-effector position and movement;
finger opening and gripper command changes; joint movement; command movement;
a one-second movement window; stationary-step count; chunk phase;
piecewise-linear time terms and five late-time interactions. Inputs contain
no robot/site/task IDs, object poses, images, events, fault labels, verdicts,
or future measurements. Absolute command-minus-position error is absent.

Each horizon has an L2 logistic fit (fixed penalty 0.001), equally weighted
command rows, training-only means/scales and standardized clipping to [-20,20].
Separate robots fit the sigmoid calibration; a negative slope falls back to
calibration prevalence. A cumulative maximum makes risk nondecreasing with
horizon. Selection and evaluation both use those final probabilities.

| Use, every day and window | Robot IDs within D2 |
|---|---|
| Fit | r01, r02, r05, r06, r09, r10 |
| Calibration and threshold selection | r03, r07, r11 |
| Test | r04, r08, r12 |

Identities are scoped to their world. No source run is modified. Feature caches
record digests of the episode table, verdicts and every steps file, with
relative filenames. Caches and models validate the exact feature schema and
refuse the old 41-feature version. Robot-cluster intervals use 500 resamples
with a fixed seed, keeping each robot's rows and episodes together. They do
not capture unobserved fault classes, worlds or correlated site effects.

## Threshold proposal and costs

Keep timeout as the reference. For a controlled telemetry comparison, use the
frozen **5-second threshold 0.47**, replacing the invalid model's 0.51.
Both the timeout rule and time-only model first call at step 174 here.
This is a supervisor-call threshold, not an automatic hand-over decision.
The operating rule is a separate decision.

Selection permits calls on at most 5% of successful calibration episodes,
then maximizes total first-call failure lead divided by all failures.
Thresholds range from 0.01 to 1.00 in 0.01 increments; 1.00 disables calls.
Timeout selects its elapsed step; stall selects consecutive sub-millimetre
end-effector movement steps, with no call before step 40. Both follow the
same calibration constraint and objective. Their two-valued probability
forecasts are training-only event frequencies with half-count smoothing;
the alarm itself remains the rule, independent of those probabilities.

The low-threshold evidence rules out claiming a useful early operating point.
Before the correction, threshold **0.25 called all 239 successful test episodes**.
After correction it calls 98/239 (41.0%) for 7.95 s median lead; thresholds
0.01–0.10 still call every success, effectively at the start. The revised
sweep is shown below. These are descriptive test operating points, not
additional threshold selection:

| Revised threshold | Caught / failures | False calls / successes | Median failure lead |
|---|---|---|---|
| 0.01 | 181/181 | 239/239 | 13.95 s |
| 0.05 | 181/181 | 239/239 | 13.95 s |
| 0.10 | 181/181 | 239/239 | 13.95 s |
| 0.25 | 181/181 | 98/239 | 7.95 s |
| 0.35 | 181/181 | 37/239 | 6.45 s |
| 0.50 | 181/181 | 15/239 | 5.75 s |
| 0.75 | 181/181 | 10/239 | 4.95 s |

At the selected 0.47 the median lead is only 0.85 s longer than timeout and
there are two more false calls. Earlier opportunities do not establish useful
rescues. A call is worthwhile when its avoidable failure cost exceeds decision
idle time plus intervention/operator time, including queue wait. Rescue
probability, actual latency and operator service time remain unmeasured here.
The illustrative threshold sweep does not establish an optimal cost policy.

The engine scores command `t` before the action hook; a decision can occur at
`t+1` only while the attempt is active. Every lead statistic includes that
delay. False calls per episode count called eventual successes divided by all
episodes; the success-only denominator is also reported. An additional
outside-horizon metric counts scores more than `h` steps before failure.
Only the first call in an unassisted trace is evaluated. Interventions change
the trajectory and require the separate replay-scoring task.

## Running it

After `make setup`, use these CPU-only commands (fish syntax). The feature
caches are distinct from the original pre-review caches:

```fish
set cache ~/.cache/robots-risk/codex-risk
uv run --locked python -m fleetlab.live prepare --world w10 \
    --run ~/.cache/robots-fixtures/mini-v2/runs/w10-observed \
    --run ~/.cache/robots-fixtures/mini-v2/runs/w10-later --out $cache/mini-v2.npz
uv run --locked python -m fleetlab.live prepare-world \
    ~/.cache/robots-probes/worlds/w42 --out $cache/w42-v2.npz
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m fleetlab.live train \
    --data $cache/w42-v2.npz --mini $cache/mini-v2.npz \
    --out $cache/refit-results --models-out $cache/refit-models
```

`prepare-world` refuses unfinished worlds. Training writes frozen model files
and the selection receipt before scoring test robots. Models contain JSON
numbers, not executable pickles. Runtime uses NumPy; fitting uses the locked
SciPy dependency. The delivered compact model files live under this package;
full measurement outputs belong under `docs/results/live-risk/` or a cache.

```python
from fleetlab.live import RiskModel, TelemetryRiskEstimator
from fleetlab.contracts.rollout import SupervisorSpec

model = RiskModel.load(model_path)
estimator = TelemetryRiskEstimator(model, horizon_steps=100)
risk = estimator.update(step_telemetry)  # fresh instance for each episode
spec = SupervisorSpec(name="risk", params={
    "model": model.to_dict(), "horizon_steps": 100,
    "decision": "stop", "decision_ms": 500,
})
```

Embedding the model freezes it in the job. A file instead requires both
`model_path` and `model_sha256`; changed contents are refused. The fixed
`decision` is explicit. This adapter exercises the hook; it is not the future
vision-language supervisor. The actual probability reaches
`SupervisorContext.risk`. A continued attempt can call again when its hold
ends; suppression and cooldowns belong to the supervisor task.

## Replay and verification boundaries

Runtime uses no clock or randomness. Retained arrays are copied, steps must
be consecutive from zero, and invalid telemetry or model values are refused.
The estimator does not sample the policy or alter its noise, batch, action
queue or rendering. Nonzero snapshot starts lack feature history and are
refused before reset through `SupervisorParts.supports_snapshot_start=False`,
without naming a particular supervisor in the engine. Scripted decision
replay and unsupervised snapshot continuations remain supported. Snapshot
history support is a separate contracts proposal.

The new CPU test uses the actual fitted JSON model in toy episodes, wraps its
estimator to capture every hook probability, writes the engine record through
`LogSink` and `RunWriter`, then rebuilds offline features with `build_dataset`.
The per-step probabilities agree within 1e-12, both without decisions and
with continue/hold steps (including null chunk indices). Probabilities vary,
so zero coefficients cannot hide a feature-path mismatch. The same episodes
also reproduce their chains with ordinary or scripted replay.

Review verification on the combined main/risk code (e6994cc):

- Focused live/engine checks: 49 passed.
- `make check`: exit 0; lint, formatting and types passed; 798 tests passed,
  2 skipped, 43 deselected, 174.12 seconds. No shutdown workaround was needed.
- `make check-sim`: exit 0; all 39 simulator tests passed, 804 deselected,
  286.11 seconds under concurrent load.
- `make smoke-world`: exit 0 on e1712fc (only this README changed since
  e6994cc); 8/8 episodes succeeded and all 16 stored chain comparisons were
  identical, with no differing or missing chains and no reference update.
  The heavy lock admitted the job at 17:45 PDT on 2026-10-08, before the
  post-lock 20:30 deadline; the smoke measurement took 28.8 seconds.

Main's engine changes were incorporated before testing. Its subsequent
changes through 971f53f affect only documentation, so the
merged code is the code checked above. The earlier
intermittent test-shutdown issue remains open.

The learned estimator's live **X-VLA path remains unverified**; smoke-world
checks existing GPU chains but does not enable this estimator. The optional
`fleetlab.live.probe` accepts `--data`, `--model`, `--out` and a mandatory
`--latest-start` deadline, checked after the heavy lock before model loading.
Run it through `tools/heavy-run` only during an authorized GPU window.

Action-spread value/cost, another development world, actual rescues, operator
time, measured supervisor latency and post-intervention calibration are also
unverified. Fresh development/model improvement and the optional spread
experiment remain follow-ups. No policy-sampling or contracts change is
needed for this correction. Generalization beyond this simulated fleet,
policy, 20 Hz rate and 280-step budget remains uncertain.

## Claim-4 repair, round 2

A pass selects every true failure in the scoring window and 50 seeded true
successes, retaining their exact identities and faults. The checked source
export, complete outcomes and passing chain sample are required. Successes
carry inverse inclusion weights; failures have weight one. No truth or source
paths enter the model context. Records at `359a3d0` are historical development
evidence, **not a claim-4 baseline**: they precede the merged engine and supervisor.
Every compared pass must be made from the same clean frozen commit.

All requests enter the same FIFO queue. Admission uses arrival time, then neutral
attempt identity, with the earliest available operator reserving a fixed service
slot. An optional maximum wait is the queue's sole refusal rule; no risk estimate
or cost ratio judges another rule's request. A refused nonterminal request is
changed to continue inside the existing supervisor, so its subsequent calls and
physical policy continue. Terminal refusals have exhausted the robot's active
budget and are scored from policy continuation, not the stopped proposal.
Consumed reaction and active-control seconds are the operator axis. Reservation
seconds and robot queue seconds remain separate.

The economic estimator rule is one simple baseline, with strict risk times ratio
(times the declared avoidable fraction) greater than one. Ratio one cannot fire
and is not run in the grid. Its sweep evaluates all ratios against **one recorded
unassisted telemetry run** per attempt; physical prefixes and human continuations
at different threshold crossings still need simulation. This is not a promise
that three different physical interventions can be simulated in one attempt.
A supervisor is run separately for each of its three ratios. The wrapper supplies
the ratio in its task context and records the queue counts it supplies.

The queue provider currently reads already persisted requests in fixed fleet-start
order, clips them to the call's fleet timestamp, and supplies the number waiting
and the number free. Proposal execution uses one worker so completion races do
not change this view. **This is the harness's known reservation state, not a fully
chronological scheduler across overlapping attempts.** A later-started attempt
may make an earlier request, unknown while the earlier job is running. Closing
that gap requires an engine event scheduler that exposes every live attempt's
next fleet tick, yields at supervisor calls, applies delayed handovers at their
actual ticks, and reports completion, including terminal backstops. The existing
`SupervisorParts.operator_queue` hook alone does not provide those events.
No engine or VLM-owned source is changed in this repair.

Canonical engine decisions are reconciled with recorded successful and failed
calls. Terminal backstops are rule decisions with declared risk one, not model
calls. Failed-call risk, timing and error survive; unavailable model timers stay
null. A skipped route cannot be scored as a model measurement. Prefix counters
replay autonomous intervention history and must reproduce its physical chain.
The first applied intervention supplies the episode counterfactual; individual
counters are retained. Both as-run and policy-from-intervention readings are shown.

An interrupted attempt that lacks a committed chain/part/steps triple loses its
abandoned continuation and sample-job files before restart. Completed attempts
and their records are retained. The raw request manifest is frozen after the
proposal phase; admitted jobs never replace it. Changed source, settings or code
cannot resume a pass. Varying-answer interruption fixtures cover every attempt
boundary, changed hold lengths, a changed answer to continue, sample jobs and
unique request counts. They prove toy-protocol resume, not a live model interruption.

The pass mark compares the **window net of rescues minus weighted broken
successes** with the highest simple-rule net at exactly the supervisor's expanded
consumed operator seconds. Every measured segment and the implicit zero-cost,
zero-net as-run point are considered. There is no extrapolation. Missing range
and nonpositive baseline nets are named; they cannot receive a percentage mark.
The point threshold stays 20%. Paired attempt resamples, stratified by as-run
outcome, reselect the frontier and provide intervals for both nets and their
difference. These intervals condition on this world and cohort; transfer and
human-model calibration remain unmeasured.

Each supervisor pass with deciding holds also runs one hold-only control, from
those recorded holds. It invokes no model and performs no retry, degrade, stop or handover.
An immutable `hold-source/record.json` supplies those holds; both its digest and
the control record's digest are retained. Attaching the control to the parent
therefore cannot invalidate the control's input hash. A toy interruption inside
the control resumes from that same source, despite changed supervisor answers.
Operator waits and reaction holds are excluded. Holds whose steps a control never
reaches are not forced after its natural end. Timeout and random have zero assigned
deciding latency and receive no control. The control net is displayed beside the
supervisor net as its hold-only reference; **nothing is subtracted from either
net for the mark**, as reviewed design entry 19 specifies. Raw rescues, broken
successes, weighted net, and actual model-call counts remain visible.

The figure plots window success and net against expanded operator minutes and
subscription/model deciding time on a separate axis. An unavailable model timer
is omitted from its plotted axis, rather than fabricated as zero.
If a supervisor never holds to decide, it has a zero hold reference and creates
no control record; the grid scores its pass without requesting a missing file.

### Grid and frozen launcher

`run_grid.sh REVISION WORLD NEW_OUT [SUPERVISOR_SPEC FACTORY]` creates a detached
checkout under the new output root, builds its own locked environment on disk,
and runs the grid in one heavy scope. Resume accepts only that exact clean commit.
The supervisor profile must be tracked in that revision and is read from the
frozen checkout; a changed caller copy cannot change the measured profile.
`GRID_DRY_START=1` builds the environment and validates the cohort, then stops
before the first GPU stage. It starts no model call. `GRID_WORKERS` defaults to two;
proposal jobs with rules use one worker for the known-state queue view.

The development grid uses timeouts at steps **1, 174 and 240**, with **two and six
operators**, one matched-count random comparator per operator setting (matched to
the late timeout), economic ratios two and four from one telemetry tape, and the
no-supervisor replay. Supplying a supervisor adds its ratios one, two and four and
one control for each. That is **17 full-cohort passes** including controls, or
**3,332 attempts for archived w42's 196-attempt cohort**, before additional prefix
and human continuations. This is deliberately not called an affordable test grid.
The early timeout requests every still-running attempt and extends the measured
operator range; if the supervisor still exceeds it, the scorer says so.

Decided test grid (`GRID_MODE=test`, 2026-10-11): two operators; timeouts at steps one and
240; one random comparator matched to the latter; economic ratio four; supervisor
ratios one/two/four and their three controls. The implicit as-run zero replaces an
extra no-supervisor run. **Ten full-cohort passes cost 1,000 to 1,400 attempts
when the cohort has 100 to 140 attempts.** The plan's budget row is an estimate,
not a cap. Larger cohorts retain every failure and openly exceed the estimate.
For a cohort of 196 the decided grid costs 1,960 base attempts. Prefix proofs,
sample replays and human continuations are additional executions; the full report
must count them from the run records rather than call base attempts the total.

Dropped test points and what they would show: timeout 174 samples the middle of
the timing curve; six operators test queue capacity sensitivity; extra random
points test additional matching rates; economic ratio two tests a more conservative
estimator threshold; the fresh none pass checks unsupervised replay again. The
three supervisor ratios and their controls are retained. The design's budget and
cohort cannot be guaranteed simultaneously for an arbitrary world's failure count.

Example (fish; the orchestrator executes the full measurement):

```fish
env TMPDIR=/var/tmp/robots-live bash src/fleetlab/live/run_grid.sh COMMIT \
    ~/.cache/robots-probes/worlds/w42 ~/.cache/robots-probes/live-claim4/NEW_ROUND2 \
    src/fleetlab/live/supervisor_spec.json fleetlab.engine.supervisors:make
```

Grid random matching is on cohort handover counts, not expanded window rate or
operator seconds. Window estimates condition on cohort queue demand: omitted
successes might also make requests. No weighting scheme simulates those missing
queue interactions. These limits must accompany any claim-4 report.

### Round-2 verification and remaining boundaries

The reviewer's 75 original single-place breaks are accounted for in
[round2_mutations.json](round2_mutations.json): **72 caught**, **3 retired**.
Breaks 28 and 29 modify the removed estimator gate; break 43 forces zero wait
inside that same retired price gate. All requests now use one queue policy.
No fault survives the executed/adapted list. The receipt explains the equivalent
new probes for in-place refusal, net comparisons, formatted expressions and
paired attempt resampling; it retains each selector and raw log path. A finite
mutation list does not prove all possible errors absent.
The entire list was repeated on runtime `d80a04b` after the final owned fixes:
all 72 applicable faults were caught again. The receipt's source hashes match
that runtime. Later changes add a shell dispatch guard and its CPU fixture;
the Python runtime checked by the simulator suite is unchanged.

Owned final CPU checks passed **40 tests**, with **one simulator test deselected**,
in **36.52 seconds**. The additional queue-predicate and physical-latency fixtures
pass repaired code and fail when their two old behaviors are restored; the
immutable control source and frozen-profile checks also exercise interruption
and a changed caller profile. The no-hold grid dispatch fixture fails on the
prior shell script and passes with the guard; it isolates dispatch using a fake
CLI and is not a live-model or GPU measurement. Lint, formatting, shell syntax and focused types
(25 source files) pass. `make check-sim` from clean detached
`b1a740a` exited **0**, with **62 simulator tests passed**, **2,591 deselected**,
in **494.19 seconds**. Later runtime changes enrich the call predicate's context,
choose the physical latency timer for refusal, and freeze the control source.
Their frozen simulator receipt is recorded at hand-over.
That repeat from clean detached `d80a04b` exited **0**: **62 passed** and
**2,599 deselected**, in **783.79 seconds**, including the real LIBERO pass,
physics, renderer, reset and recovery checks. This does not exercise the
pretrained GPU policy, a live model session, or a person.

The requested single whole `make check` invocation exited **2**: **2,583 passed**,
**two skipped**, **66 deselected**, **three failed**, in **2,962.37 seconds**.
Lint, formatting and types (401 source files) passed. Its collected suite used
the committed candidate `135a667`; the subsequent owned fixes are checked by
the final CPU and simulator receipts. The three failures are unchanged
supervisor-owned assertions of ratio-one denial or `StopIteration`. Their exact
required updates belong to the terminal and failed-call harness repair; they were
not hidden by a skip.

The w42 export was recertified by appending a new clean receipt (no existing
export or run record changed). Frozen `b1a740a` dry start exited **0**, prepared
its own environment, selected the 196-attempt cohort, planned 17 full-cohort
passes (3,332 attempts), and stopped before its first GPU stage. It loaded no
policy on the GPU and made no model call. The final-commit dry start and whole
`make check` receipt are recorded at hand-over. No full w42 grid or fresh smoke
was run: the hand-over prohibits GPU jobs beyond dry start.

Round-2 scope boundaries at delivery: the supervisor builder owns three tests whose old
expectations assert ratio-one denial or `StopIteration`; their exact required
updates belong to the terminal and failed-call harness repair. Complete overlapping-attempt queue chronology needs
the engine scheduling events described above. The proposed test grid cannot
promise the frozen attempt budget for cohorts larger than 140, nor count its
additional physical continuations inside that full-cohort estimate. These are
not demonstrated as completed or quietly hidden by a skip.
The budget/point decision is the test grid above. A failed call refused by a bounded
queue still raises `queue refusal escaped the retained supervisor`:
the engine needs an explicit nullable-timer failed-call fallback to continue.
The unbounded grid completes the failed-call fixture; this does not prove the
bounded case. Call predicates now see the same known queue as actual calls;
successful-answer refusal uses the latency route's physical deciding time.

### Integration on stage5/integration, 2026-10-11

The harness and supervisor now share one branch. The three obsolete assertions
are replaced by positive coverage: all three cost ratios admit the six toy VLM
requests; terminal-rule handovers after continue and failed-call handovers both
complete and retain their canonical trace, call counts, holds, errors and nullable
model timing. A test through the actual harness and VLM adapter captures model
inputs with zero waiting/two free operators and one waiting/no free operators.
The captured fields equal the harness's recorded queue observations. This proves
the known-reservation interface, not full scheduling of overlapping attempts.

Published entry 19's mark already uses the weighted window nets at equal consumed
operator time, subtracts no control, refuses extrapolation and leaves nonpositive
baseline nets unmarked. The integration adds an interval to every table row,
including hold-only controls. Candidate/best-baseline/difference intervals use
the same supported paired resamples; a draw outside the measured operator range
is omitted from both compared net intervals. Hold-only intervals remain descriptive
on all draws. Controls can only reference supervisor passes, never simple rules.
Regression fixtures change the control's outcome without changing the mark and
check the undefined zero-baseline case. The point threshold remains 20%.

The failed-call fallback is still open. Its positive regression is a strict expected failure;
running it with `--runxfail` produces exactly `queue refusal escaped the retained
supervisor`. The read-only engine in `episode._decide` hardcodes failed calls to
hand over or stop. The required change outside this assignment is an optional,
validated fallback Decision on `contracts.live.SupervisorCallFailure`, preserving
the current default, and engine application of that fallback with failed/skipped
status, measured decision hold and nullable model timer intact. The live wrapper
must then probe admission after the appropriate hold, record requested/refused,
and choose continue on refusal. It must retain its rule, allow later calls and
reserve no refused operator time. Returning an apparently successful answer with
a zero model timer would lose failure evidence and is not implemented.

One command prepares and measures the full development grid from an immutable
checkout of the supplied commit:

```fish
env GRID_MODE=development GRID_WORKERS=2 TMPDIR=/var/tmp/robots-stage5 \
    bash src/fleetlab/live/run_grid.sh FINAL_COMMIT \
    ~/.cache/robots-probes/worlds/w42 ~/.cache/robots-probes/live-claim4/NEW_STAGE5 \
    src/fleetlab/live/vlm/profiles/one_early_look.json fleetlab.engine.supervisors:make
```

Add `GRID_DRY_START=1` to exercise only startup. The launcher reads the tracked
profile from its own clean detached checkout and builds an independent environment
on disk. The CLI accepts either a SupervisorSpec or a plain VLM settings profile.
The profile selects `gpt-6-astra` at low effort, one assessment at active step 120, with the existing
fast subscription route. Preflight verifies every frozen identity's software and
pinned convolution database and runs the existing free Codex gates before any
GPU execution. Its private runtime defaults to `/var/tmp/robots-stage5-grid-UID`.

For stored w42, the cohort is 146 failures plus 50 sampled successes. The grid
plans 17 full-cohort passes/3,332 base attempts. There are 186 source attempts
still active after step 120, giving **558 expected calls across three ratios**;
the cohort-wide ceiling is 588. Controls and replay proofs call no model. Resource
planning is uncertain: the test-grid estimate scales to about **1.7 GPU-job
hours for base passes**, plus **1.42 hours of model waiting** at the historical
9.15-second offline low-effort median, plus unmeasured physical holds, prefix
proofs and operator continuations. These are queue occupation estimates, not a
measured inference time or a guaranteed runtime. Two workers plus the policy
server have an engine admission estimate of 8.2 GiB resident memory and require
16.2 GiB available including the 8 GiB reserve. The heavy scope limits RAM to
24 GiB; the engine refuses GPU admission above 17 GiB in use and aborts above
21 GiB. The supervisor proposal stage uses one worker. Actual peaks are unmeasured.

`NEW_STAGE5/checkout` holds the frozen code. `NEW_STAGE5/grid` holds cohort and
preflight receipts, none/simple-2/simple-6/random-2/random-6/costs-2/supervisor-2
passes, and `scores/table.json`, `scores/table.md` and `scores/success.png`.
Each pass retains manifests, raw calls, physical chains, proofs, counters,
operator metrics and its record; supervisor controls sit under each ratio's
`hold-control`. A dry start instead writes `gpu-boundary` and `model-boundary`
manifests/configuration/receipts, then stops before `fleet.run`. It also encodes
a clearly labelled synthetic startup input and intercepts the adapter immediately
before `run_agent`, recording model/effort/tier without invoking it. That verifies
model request preparation, not actual world observations, authentication of a
paid call, GPU inference, rescue quality or a full measurement. No result record
or score is written by that dry start.

Focused integration checks passed 48 tests with one expected engine-boundary
failure and one simulator test deselected. The six final preflight/resume checks
also pass. The full unit run exited 0: 2,605 passed, two skipped, 66 deselected
and one strict expected engine-boundary failure, in 3,055.75 seconds. The frozen
`make check-sim` at `5938ea6` exited 0: 62 passed, 2,612 deselected, in 552.03
seconds. Final lint, formatting and mypy (402 files) each exit 0. The single
allowed `make check` invocation exited 2 before pytest at a test-only mypy import
error. That import is fixed; all four check stages were completed directly with
exit 0, without a second `make check` invocation. This is equivalent stage evidence,
not an observed green exit of that one make invocation. The final test refinement
allows a later accepted handover and verifies the failed/nullable canonical trace
of each refused call; its focused run retains one strict expected failure. Runtime
code and selected simulator tests are unchanged from the frozen simulator run.
Exact receipts were recorded at delivery.
The orchestrator owns independent review, the real
grid and smoke/GPU verification. No engine, design document, model-routing file,
stored result or smoke reference is changed here.

### Historical round-1 receipts (not claim-4 evidence)

On 2026-10-09, `make check` exited 0 on `b8d0abb`: lint, formatting and
types passed; 1,093 tests passed, 2 were skipped and 63 tests were
deselected, in 403.47 seconds. `make check-sim` exited 0 on `359a3d0`:
59 simulator tests passed and 1,099 other tests were deselected, in 699.16
seconds. The difference between those commits only narrows a test's JSON
type assertions; their runtime code is identical.

The new coverage includes strict expected-cost boundaries, FIFO admission,
same-revision comparison guards, sampled-success weighting, paired frontier
intervals, both latency routes, interrupted-pass resume, and physical-prefix
counterfactuals after earlier `continue` assessments. The simulator test uses
real LIBERO states for queued hand-overs and an autonomous intervention; it
does not exercise the VLM, a human operator, or the full D2 population.

The earlier smoke checks exited 0 on `7e88913` and `c1eb3c7`, with all eight
episodes successful and all 16 stored chain comparisons identical. A fresh
`make smoke-world` also exited 0 from clean detached `359a3d0`, in 50.9
seconds: eight episodes succeeded, all 16 chain comparisons were identical,
and no chains differed, were missing, or had changed software. The reference
was untouched. The guarded controller advanced to the D2 grid only after
that successful exit. The grid is queued at the same revision; no completed
D2 record is claimed in these receipts yet.

The subsequent custom-factory comparator fix and portable grid launcher passed
`make check` (exit 0: 1,094 passed, 2 skipped, 63 deselected; 311.24 seconds)
and `make check-sim` (exit 0: 59 passed, 1,100 deselected; 522.72 seconds).
The focused CPU file passed all 11 tests. Bash syntax and a separately
labelled dry dispatch check confirmed the launcher's 16 default / 22 optional
record paths, including arguments containing spaces. That dry check ran no
simulator, model, or scoring. The frozen D2 controller calls the same 16
native operating points directly; the optional VLM matrix remains unmeasured.
