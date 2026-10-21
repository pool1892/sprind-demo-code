# Vision-language supervisor

The 2026-10-11 harness integration is documented in [the live README](../README.md#integration-on-stage5integration-2026-10-11).
Its frozen grid uses `profiles/one_early_look.json` directly: `gpt-6-astra` at low effort, one look at
active step 120. The harness-to-adapter fixture now proves different known queue
states reach the actual model input; obsolete StopIteration/ratio-one-denial
assertions have positive replacements. Failed-call refusal by a bounded queue
still needs the read-only engine's explicit fallback interface (still open).
No revised live outcome or GPU measurement is claimed by this integration.

`SupervisorSpec(name="vlm", params={})` uses the subscription runtime's
sandboxed Codex route, model `gpt-6-astra`, effort `xhigh`, fast service tier.
`backend="claude"` selects `claude-opus-5-5`; either model can be overridden.
It sees only live `SupervisorContext` data and the two general supervisor
skills. No simulator handle, object geometry, fault labels, original outcome
or identity enters the call. Three spaced upright images per camera accompany
the full telemetry window, risk, task, options, prior choices and queue state.
There are no tools in the model call. The runtime's schema and local validation
must both accept the answer. Auth or usage failures stop that route; a shared
stop receipt prevents subsequent workers from issuing calls. Failed calls become charged truth-side decisions with a declared hand-over
fallback. Earlier decisions survive. Subsequent attempts see the disabled
route, record a `skipped` event and take the same fallback without calling
the subscription again. Such a run is **refused as a measurement**, before the
operator pass or rescue scoring; `measurement-refused.json` identifies the
skipped calls. Calling `report` separately enforces the same refusal. The run,
earlier decisions, failed call and skipped-call charges remain available for
diagnosis and replay. Missing model timing is null, never a fabricated zero.
This covers adapter/runtime failures, not a worker process killed during a call;
worker-loss persistence belongs to the live-layer work.

The supervisor runs at 160 active ticks (8 seconds at 20 Hz) and, if still
active, at least 70 ticks later, at most twice. After a primitive it waits
for that primitive's duration plus 40 policy ticks: retry is reassessed after
110 active ticks, recover after 74. This avoids paying for a second opinion
before the policy has used the recovered pose. Decision holds are excluded
from this clock. Grasp/release events are only commands, so they are not used
as evidence of a drop. The existing risk model supplies context but does not
suppress the scheduled inspection: on this development fleet it mostly
identifies elapsed time. This is a declared development choice, not an
optimized or held-out call policy. On the final call the options are continue,
stop and hand over, so another unassessed primitive is not started. If the
policy still fails at its visible terminal check or active limit, a zero-call,
zero-hold `hand_over` rule saves a snapshot and enters the existing operator
path. A successful finish does not trigger this rule.

Physical options:

- Retry opens, lifts 8 cm, travels above the retained pre-grasp command,
  returns to that command, then gives control to the policy. It takes 70
  ticks. Eligibility requires absolute control, the first open-to-close grasp
  approach tracked within 3 cm, a current commanded pose tracked within 3 cm,
  a return distance no greater than 35 cm, and a nominal budget above the
  70 primitive ticks (so at least one policy tick remains). The anchor is the last open
  command at that first approach: later closes during transport never replace
  it. Both that command and the observed close must be within 3 cm vertically
  of the lowest end-effector position observed so far. A later lower observation
  can invalidate the anchor too. This rejects closes during a high excursion;
  there is no initial-pose or later-close fallback. The model still judges whether the
  object remains at that site from the images; telemetry cannot establish
  contact or guarantee recovery under an unseen fault. No world reset occurs.
- Recover opens for 10 ticks, lifts 8 cm over 16 ticks, and settles for 8.
  Its visual precondition is the correct object already over the destination.
  It is a release/retreat primitive, not a new grasp planner.
- Degrade discards queued actions and runs at most four commands from each
  subsequent policy chunk before requesting fresh visual inference. It keeps
  the task, noise rules, policy, faults and model unchanged; additional
  inference is real. It is unavailable for in-worker policies.
- Continue resumes the policy; stop ends safely after the hold; hand over
  records a snapshot for the existing human-model teleoperator.

Absolute recovery targets stay in commanded coordinates. Relative recovery
uses observed end-effector position with LIBERO's 5 cm/unit translation scale
and a 0.3 command cap, while holding its current orientation. Relative
**retry is unavailable** until pose return is implemented and validated;
relative recover remains available. Absolute orientation
interpolation follows the shortest SO(3) rotation, including across the
rotation-vector branch cut. Recovery needs no private object geometry. It does not
repair camera calibration, restore objects, turn off faults, or invoke the
privileged scripted teleoperator as an autonomous skill.

The six inspected observed-window D2 failures were e000011, e000302,
e000582, e000811, e001051 and e001359. Scene and wrist frames at ticks 40,
100, 160, 220 and 278, plus end-effector/gripper telemetry, showed empty
carries after missed grasps, an object resting at a basket rim, and large
arm excursions. Inspection contact sheets are in
`~/.cache/robots-supervisor/inspection/`. These examples motivated the
primitives; they do not establish their rescue rates. Automated visual
regrasp planning and calibrated contact recovery are omitted because neither
is available through the observable contract or validated here.

Every hold physically steps the simulator and its action faults, and is
charged as robot time, but cannot exhaust the active-attempt or success-check
timeout. Success may still occur during a hold. The check's clock offset is
preserved in its opaque snapshot bytes; old snapshots remain readable. A
continue decision permits an active policy tick before another inspection.
Supervisor-free execution retains the fault repair's command path; the
supervisor integration adds no transformation to it.

Stage-5 design choice accepted in the repair request: **at most one retry per
attempt, with its own active budget equal to the nominal attempt budget**.
The budget starts when the retry primitive begins, after its decision hold;
the 70 primitive ticks consume that budget. `DecisionRecord.retry_budget_steps`
records the grant. Scripted replay applies that same grant at the same physical
tick. Snapshots record `remaining_active_steps`, `retry_used`, and `degraded`,
so a resumed policy receives the remaining active budget and cannot retry twice.
An operator continuation with its own `max_steps` still gets that own budget.
Old snapshots recover their remaining budget from the saved idle-check clock.
Mid-attempt scripted starts are refused: replay the full trace from reset.

`run.json` job details report `retry_started_at`, `retry_budget_steps`,
`retry_active_steps`, `robot_active_steps` and `robot_time_s`. Robot time is
active policy/primitive ticks divided by 20 plus full subscription decision
seconds; none of a retry is operator service time. The pilot report totals
robot and retry time from original run jobs only, excluding proof replays.
Older reports without these stats return null for these new totals.
The current prompt states that retry gets its own nominal budget once,
recover spends the remaining budget, and decision holds do not spend it.
That factual budget correction was authorized in round 3; the instructions
and all other prompt wording remain unchanged. Historical measurements below
identify the earlier prompt versions they used.

`decision_ms` measures the whole adapter call, including setup, frame encoding,
CLI startup and response. `model_ms` is the provider's `duration_api_ms` on
Claude. Codex supplies no separate provider timer in this runtime: its value
is the measured turn-start to turn-completion envelope, explicitly a
production estimate containing unseparated overhead, not pure model time.
Each runtime run has a `supervisor.json` receipt explaining that basis.
The truth-side decision table now includes rationale; scripted replay uses
the recorded decision, hold ticks and retry grant, never another model call.
Duplicate entries or entries inside a hold, primitive, or after termination
fail the replay. Terminal rule records must occur at a visible failure.
Legacy decision tables/parts acquire optional truth-column defaults only in
memory when read or finalized, including SQL reads; source records are not
rewritten. A legacy retry trace without the explicit grant is refused for
scripted execution because its old budget semantics differ.

`SupervisorParts.should_call` and its `operator_queue` callback let the full
live-layer evaluator supply a schedule and current queue state. Contract
additions are Decision.recover/degrade, optional SupervisorContext.active_step,
remaining_active_steps, previous_decisions and frame_steps, and
DecisionRecord.rationale. This repair adds `SupervisorCallFailure`, nullable
recorded `model_ms`, `call_status`, `error`, `retry_budget_steps`, the snapshot
fields above, and `SupervisorParts.failure_fallback`. The additional log
columns remain hidden. Successful `Supervisor.decide` still returns the same
four-field `SupervisorDecision`; existing constructors remain valid.

The full live harness should construct `make_vlm(params)` (or use registry
name `vlm`), provide its live `OperatorQueue` callback on `SupervisorParts`,
and honor both `should_call` and `failure_fallback` in the engine hook. It
should admit/deny the resulting continuation snapshots through its own queue,
compare the same identities against all simple rules at each cost setting,
and score actual operator service separately from queue wait and robot time.
Use `decision_ms` as primary cost; label the nullable `model_ms` estimate.
Trace `rule` and `skipped` entries are not model calls. The existing pilot's
historical `calls` field counts trace records; `model_call_attempts` distinguishes
actual VLM attempts. No second claim-4 harness is implemented here.
For that harness's CLI the interface is `--rule supervisor --supervisor SPEC.json
--factory fleetlab.engine.supervisors:make --latency-route subscription`, where
the file is a `SupervisorSpec` such as `{"name":"vlm","params":{}}`. Populate
its cost parameters for the chosen harness ratio/service assumptions instead
of silently retaining the pilot defaults. Use a new shared `stop_file` per run.

The `measure` CLI freezes a seeded, stratified sample, runs either the VLM or
timeout at tick 201, admits handovers with the existing human model, and
reports rescues and damaged successes against the original as-run outcomes.
This sample harness supplies a **static** queue observation (zero waiting,
two free), then uses actual FIFO reservations for operator accounting. It is
not the other builder's dynamic queue or equal-operator-time cost sweep.
The prompt prices operator time at 0.2 failed-attempt equivalents per minute
and robot time at 0.02, with 25.5 seconds as expected operator service time.
These are declared development settings, not tuned cost-ratio results.
The timeout rule has zero decision latency (one minimum physical hold tick).
Source results are read only; each selection and run needs a new directory.

Runtime gates use an isolated `ROBOTS_AGENT_ROOT` outside the repository.
The deferred rerun below selects Codex only; no new Claude probe is requested. Simulation commands run
through `tools/heavy-run` in a detached checkout; keep it immutable while a
job runs. `select WORLD OUT --failures 60 --successes 20` freezes the D2
sample; `run SAMPLE OUT --method vlm|timeout` measures it. Use the same sample
for both. Results and verification evidence are recorded below.

## Verification and miniature-fleet evidence

At 5d548d4, `make check` exited 0 with 1,095 passing tests and two skipped
(509.52 seconds). This includes image orientation and context boundaries,
schema/error handling, a failed-call stop across fresh supervisor instances,
call scheduling, physical-command semantics, truth-side rationale logging,
long holds, and scripted replay. After the remaining-budget correction,
`make check-sim` exited 0 at 5d548d4 with 61 tests passing (539.23 seconds).
Three of those tests exercise retry, recover and degrade with actual LIBERO
physics and identical scripted chains. Twelve focused supervisor tests also
passed, including preservation of the remaining budget across a prior hold.
Both vendor runtime gates passed again for the a9ca7e6 tool environment.
The Codex gate passed again at 5d548d4 before the corrected measurement.
`make smoke-world` exited 0 again at 5d548d4: 16 of 16 chains matched
the stored reference, with no changed software or missing chains (57.5
seconds reported by the smoke command). The reference was not changed.

One **offline** Codex subscription call on decoded miniature-fleet video and
recorded telemetry succeeded. This is authentication/image/schema evidence,
not an in-loop experiment or rescue. At active tick 160 it chose retry and
cited an empty gripper and the bowl still in the drawer. The complete record
is `/home/cs/.cache/robots-supervisor/offline-codex-probe.json`.

| Evidence | Calls | Full decision time | Codex turn-envelope proxy |
|---|---:|---:|---:|
| Cached miniature observations, Codex gpt-6-astra requested, xhigh/fast | 1 | 33.121 s | 30.784 s |

An earlier full unit invocation found two stale skill-version assertions,
which were corrected, and encountered the existing queue-feeder shutdown
hang. The first full invocation at 5d548d4 stalled after 299 passing tests
in the worker-loss cases; interrupting it also exposed the queue-finalizer
hang. The unchanged candidate then completed the
entire suite and exited 0 with full-trace/faulthandler diagnostics enabled.
No worker-loss code was changed, finalizer suppressed or test disabled.

The miniature Codex run at a9ca7e6 exited 0 on two originally failed attempts and one
originally successful attempt. It rescued neither failure autonomously and
handed one to the operator, who rescued it using 8.95 seconds of operator
time and no queue waiting. The other failure stayed a failure. The original
success finished without a supervisor call. All three sampled replay chains
matched (6,526 compared frames). These three cases prove integration, not a
rescue-rate advantage.

| Miniature attempt | Active tick | Decision | Full decision time | Codex turn-envelope proxy |
|---|---:|---|---:|---:|
| e000075, originally failed | 160 | retry | 55.529 s | 53.649 s |
| e000075, originally failed | 270 | hand over | 23.314 s | 22.582 s |
| e000183, originally failed | 160 | continue | 16.741 s | 15.025 s |
| e000183, originally failed | 230 | continue | 34.431 s | 33.673 s |

The four calls used 130.015 seconds in total, with a median of 28.872 seconds.
The replay proof uses physical ticks, including each call's hold; the table
uses active ticks to show the schedule. Results, rationales and operator
metrics are in `/home/cs/.cache/robots-supervisor/mini-codex-v2/`.

The queued initial probes were cancelled while still blocked in `flock`,
before any simulator or model work, so the detached checkout could move to
a9ca7e6. Their empty progress logs are preserved.

The single-attempt Claude probe on e000075 passed its gate but its one real
model call timed out at the 120-second ceiling. It requested/reported
`claude-opus-5-5`, effort `xhigh`; CLI wall time was 120.488 seconds, with
initialization at 0.738 seconds and thinking-progress events but no answer.
There is no completed decision_ms/model_ms pair, intervention outcome or
replay proof for this route. The stop receipt prevented the engine's episode
retry from making a second call. This is a timeout, not evidence of a login
failure. No login file was changed. Evidence is in `mini-claude-v2` under
`/home/cs/.cache/robots-supervisor/`, with runtime receipt
`/var/tmp/robots-supervisor-agents-1000/runs/20261009T123900-0f9f96fc/result.json`.
The Codex path remains the default.

The miniature exposed an input omission: e000183's final continue choice had
only 50 active ticks (2.5 seconds) left, but the model had not been given the
remaining budget. The adapter now receives that visible budget from the
engine and explains that primitives consume it while decision holds do not.
This is a development correction before the D2 model measurement; the
miniature results above used the earlier context. The queued `d2-codex-v1`
command was cancelled in `flock`, before any model or simulator work.

The corrected miniature recheck at 5d548d4 exited 0 on the single original
failure e000183. It continued at active tick 160, then handed over at tick
230; the second rationale explicitly cited the remaining 2.5 seconds. The
operator rescued the attempt using 10.15 seconds, with no queue waiting.
There was no autonomous rescue. Its scripted chain matched over 2,932
compared frames. This deliberate development rerun verifies the new context
and integration; a stochastic rerun with different hold durations does not
isolate the budget field's causal effect.

| Corrected miniature attempt | Active tick | Decision | Full decision time | Codex turn-envelope proxy |
|---|---:|---|---:|---:|
| e000183, originally failed | 160 | continue | 17.424 s | 15.539 s |
| e000183, originally failed | 230 | hand over | 44.266 s | 43.363 s |

The two calls used 61.690 seconds in total. Evidence is in
`/home/cs/.cache/robots-supervisor/mini-budget-codex-v1/`.

## Historical D2 comparison (superseded engine)

These numbers describe the original engine only. The reviewed fault-frame
repair makes 93 of 153 stored D2 takeovers stale, so the 23-versus-28 result
cannot serve as either arm of the next comparison. Both arms below are
regenerated on the combined engine; no old takeover or timeout result is reused.

D2's frozen `d2-sample-v1` contains 60 originally failed and 20 originally
successful attempts from scoring days 8–9, selected with seed 602. The
timeout comparator, `d2-timeout-v1`, exited 0 at a9ca7e6. It handed over all
60 original failures, rescued 28 with the operator and none autonomously,
and left all 20 original successes uncalled and successful. Operator time
was 1,455.4 seconds in total (24.257 seconds per handover); operator queue
waiting was a separate 1,440.0 seconds. Its 60 rule decisions made no model
calls and have zero configured decision latency, with the documented one
minimum physical hold tick each.

All 20 sampled supervised replay chains matched (7,328 compared frames).
The operator continuations were measured but not replayed again in this
sample harness. The later remaining-budget context addition does not alter
the timeout rule or its physical commands.

The corrected candidate's `d2-codex-v2` comparison exited 0 at 5d548d4 after
its required checks and miniature recheck passed. The same frozen 80 episode
IDs and original outcomes were used for both methods. This development
point **does not demonstrate the required advantage**: the supervisor
rescued 23 of 60 original failures versus 28 for timeout, while using less
total operator time. It did not make autonomous retry or degrade effective
on this sample.

| Quantity on the same 60 failures + 20 successes | Timeout | Codex supervisor |
|---|---:|---:|
| Original failures rescued before handover | 0 / 60 | 1 / 60 |
| Original failures rescued after handover | 28 / 60 | 22 / 60 |
| Total original failures rescued | 28 / 60 | 23 / 60 |
| Attempts handed over | 60 / 80 | 48 / 80 |
| Original successes receiving a decision | 0 / 20 | 2 / 20 |
| Original successes lost | 0 / 20 | 0 / 20 |
| Operator time, summed across handovers | 1,455.400 s | 1,311.450 s |
| Operator time per handover | 24.257 s | 27.322 s |
| Operator queue waiting, separately summed | 1,440.000 s | 641.800 s |
| Decisions (subscription model calls) | 60 (0) | 111 (111) |
| Full subscription decision time, sum | 0 s | 2,933.521 s |
| Full decision time, median / 95th percentile | 0 / 0 s | 23.887 / 48.571 s |
| Codex turn-envelope production proxy, sum | Not applicable | 2,833.758 s |

All 20 sampled supervisor chains matched over 32,818 compared frames, with
no render transients or broken chains. Neither the supervised run nor its
operator pass recorded worker incidents. The main measurement and operator
pass exited successfully; no failed model route or paid episode retry was
hidden in these numbers. As for the comparator, operator continuations were
measured but not replayed again in this harness.

The decisions were 23 retries, three degrades, 37 continues and 48 handovers;
recover and stop were never selected. The one rescue before handover,
e000384, followed two continue choices. No attempt receiving retry or
degrade succeeded before handover. The two originally successful attempts
that received calls, e000928 and e001564, each received continue and still
succeeded. Every operator continuation included active operator steps, so
the 22 post-handover rescues are not queue-only placements.

Pairing the 60 original failures: both methods rescued 21, only timeout
rescued seven, only the supervisor rescued two, and neither rescued 30.
Confidence is high in these recorded sample counts and timing receipts.
They do not isolate the value of visual reasoning from physical waiting,
choice timing or altered policy observations, and they do not estimate
performance on other worlds or real humans. The static queue observation,
single development setting and unmatched total operator budgets preclude
an equal-time claim-4 conclusion. The full live-layer builder's matched
revision, dynamic queue and cost sweep remain separate work.

Complete per-attempt outcomes and rationales are in
`/home/cs/.cache/robots-supervisor/d2-codex-v2/report.json`; the paired
comparator is `d2-timeout-v1/report.json` under the same root. `started.json`
and `measurement.json` preserve the source revision, sample hash, prompt
cost settings and human model; `supervised-run.json` preserves the replay
proof before the operator pass. The full subscription route was exercised
here. A physical run using the production timing proxy was not measured.

## Round-2 repair verification and deferred measurements

Repairs are in ce00caa, 083275a and e5a38f9. The 114-test focused suite passed at exit 0;
the three LIBERO recovery/replay tests passed at exit 0 (21.13 s), using a CPU
expert only as a realistic grasp-approach fixture, with no learned policy or
model call. The retry test checks return within 3 cm of the first grasp pose
and within 0.1 rad of its observed wrist orientation. A separate rotation
branch-cut test bounds every commanded orientation along the 3-degree short
arc. These prove a controlled physical path, not D2 rescue performance.

The old decision-table check read **271 tables, 3,428 rows**: the frozen
inventory of stored tables under the probe and fixture roots, without changing
them. The file-by-file receipt is `round2/legacy-decisions-final.json` under
`~/.cache/robots-supervisor/`.
Tests also remove all newly added columns from a staged run, read it through
Arrow and SQL, append another attempt, and finalize it. Snapshot tests cover
10-, 100- and 600-tick holds with both explicit and legacy-derived budgets.
They retain 50 active ticks of a nominal 60 and preserve a separate 17-tick
operator continuation budget. Mid-attempt scripted replay is rejected by
both the Job validator and the engine.

Regression tests now exercise the review's missed guarantees: frozen check
clock, unavailable choices, negative/nonfinite timers, rounded-up hold ticks,
discarded queued chunks, first tracked grasp anchor, recent frame cadence,
physical recover lift, final-call options, exact prompt-field allow-list,
subscription versus model timing, actual `vlm` factory wiring, risk clock
through idle/primitive rows, original-failure rescue counts and actual operator
service counts. Failure integration uses a successful first call and a failed
second call with synthetic clocks, persists both, and proves scripted replay
from the stored trace. No authenticated runtime claim follows from mocks.
The new regressions are in `tests/test_supervisor_repairs.py` and
`tests/test_vlm_repairs.py`; physical assertions are in `tests/test_sim_vlm.py`.
The review mutation identifiers covered are E02/E05/E06/E07/E09/E10/E11/E14,
R05, V01/V03/V04/V05/V06 and S01/S02. E10 now asserts the requested first
tracked grasp anchor, superseding the old latest-close behavior. The original
mutation runner was not rerun; no mutation-score claim is made.
The instruction function and payload expressions compare identically to
941caba, as do both skill files; receipt: `round2/prompt-freeze.json` under
`~/.cache/robots-supervisor/`.

`tests/data/supervisor_rule_reference.json` is the new CPU rule reference,
measured from detached e5a38f9 via `tools/heavy-run --light`: **240 attempts,
200 scripted traces identical**, with 40 short-budget retry cases correctly
receiving no call. `tests/test_supervisor_references.py` both
reproduces and verifies it. Every new reference value is committed; none was
written into an old stored run. Compared to the review's pre-supervisor base,
40/40 retry, 40/40 continue, 10/40 hand-over **and 10/40 stop** chains move.
Continue/stop/hand-over changes arise from held active/check clocks; retry
also changes its primitive, anchor eligibility and active budget. Against
941caba only 40/40 retry chains in the shared cohort move; all 120 other
shared rule run chains remain exact. Another 80 cases use a 100-tick budget to
exercise completed retry primitives; they have no pre-supervisor reference.
The complete before/after key inventory is
`results/reference_transition.json`.

The only pre-existing committed chain reference is
`src/fleetlab/engine/smoke_reference.json`: unsupervised and unchanged.
The on-disk nightly references that may move are the live digest maps in:

- `~/.cache/robots-probes/nightly/2026-10-08/scores.json`
- `~/.cache/robots-probes/nightly-check/2026-10-08/scores.json` (copy of that night)
- `~/.cache/robots-probes/nightly-check/2026-10-09/scores.json`
- `~/.cache/robots-probes/nightly-dry/2026-10-08/scores.json`

Their prior file hashes and every live key/digest are inventoried in the same
JSON. Scripted proof filenames also change because job keys hash the full
decision trace, including the added default fields, even when physical chains
are identical. The inventory identifies every old scripted key separately;
compare the episode/proof role as well as the filename set. These reference files are records, so remain untouched. New GPU nightly
digests are **not generated in this round**; exact changed keys cannot be
asserted without that run. Unsupervised world/canary digests should not move.
The later authorized refresh must use a fresh root; for the full fixed/date-seeded
2026-10-08 reference, from the clean detached `robots-supervisor-run` checkout
at **8dead7d** (the tested merge commit):

```sh
bash src/fleetlab/live/vlm/refresh_nightly.sh 8dead7d \
  /home/cs/.cache/robots-probes/nightly/2026-10-08 \
  /home/cs/.cache/robots-supervisor/nightly-reference-merged-v1
```

This command is prepared, **not run or queued**. The orchestrator owns its
scheduling and the smoke run at merge. The wrapper reads the prior date and
day length (40 for the full night, 2 for the dry night), queues the existing
nightly runner through `tools/heavy-run`, and writes both worlds, their live
passes, canary, and new digest maps to `NEW_ROOT/DATE/scores.json`. It also
writes `reference-transition.json` with **every** differing key and before/after
digest in both worlds' world/live maps, plus the existing nightly comparator's
`reference-comparison.json` and `.md` against the explicitly supplied old night.
This is separate from the runner's ordinary `comparison.*`, which has no prior
night in the fresh root. A missing/renamed scripted key is recorded as null on
the absent side; it is not asserted to be a physical divergence. The wrapper
fails if unsupervised world digests change, any run's replay check fails, or
the canary differs. Live changes remain a candidate for review, not automatic
approval. No old scores file or smoke reference is overwritten.

For 2026-10-09 or the dry reference, supply that earlier night and another new
root; the wrapper retains its date and day length automatically. The
`nightly-check/2026-10-08` copy shares the canonical night's input/reference.
Treat new nightly digests as the next reference only after reviewing
the comparison, not by replacing prior records or suppressing mismatches.

The D2 rerun is prepared in `repeat_d2.sh` and uses the existing pilot harness,
not a new scorer. After moving the idle detached run checkout to the reviewed
merged commit **8dead7d**, run there:

```sh
bash src/fleetlab/live/vlm/repeat_d2.sh 8dead7d \
  /home/cs/.cache/robots-supervisor/d2-sample-v1 \
  /home/cs/.cache/robots-supervisor/d2-merged-v1
```

The script requires a clean detached checkout at exactly that commit and a
new output directory, runs timeout and Codex VLM on the same frozen 60 failed
and 20 successful identities, and uses a separate runtime root. Its preflight
checks the manifest's stored SHA-256, unique identities and 60/20 baseline
counts. Both arms go through the GPU queue. Each writes its own `manifest.jsonl`,
`started.json`, replayable `run/` (including truth-side decisions),
`supervised-run.json`, `measurement.json`, and `report.json` under
`NEW_OUTPUT/timeout/` and `NEW_OUTPUT/vlm/`; subscription receipts go under
`NEW_OUTPUT/agent-runtime/`. Each report compares its outcomes to the same
frozen original outcomes. Compare those two reports on their identical episode
IDs for autonomous/operator rescues, hand-overs, disturbed successes, actual
operator service, robot time and model-call decision time. A skipped route
instead writes a refusal receipt and stops without a VLM rescue report.
The script does not invoke the full claim-4 cost/queue sweep or reselect/tune
the cohort. It and the nightly wrapper are prepared, not run or queued.
Results above remain historical evidence for
5d548d4; they do not measure the repaired retry or backstop. No GPU policy run,
model call, new D2 outcome, or smoke validation is claimed here.

Read-only integration review of `stage5/live-rest` at 0acb882 found a concrete
remaining requirement: its recording wrapper sees only successful
`Supervisor.decide` returns. It must retain context/risk on
`SupervisorCallFailure`, propagate the exception for engine fallback, and
admit `call_status=rule` terminal hand-overs from the engine's trace too
(declared failure risk 1 after a visible terminal failure). Looking only for
hand-over calls in the wrapper's receipt misses both paths. Its model-time
sums must support null, and its prefix replay must retain retry grants and
terminal rules from the engine trace. This is left to
the live-layer builder; combined-branch integration is not verified here.

Final checks on 0f04acb (executable code e5a38f9) both returned **exit 0**:
`make check`: **1,131 passed, 2 skipped**, lint/format/types clean, 360.09 s;
`make check-sim`: **61 passed**, 484.29 s. The latter includes all three
recovery tests with the final accounting assertions. Timing is shared-machine
check duration, not supervisor decision latency. Logs are in
`~/.cache/robots-supervisor/round2/`; the committed summary is
`results/repair_verification.json`. Later edits only record this evidence and
reference inventory; they change no executable source. No worker-loss paths,
login stores, design documents, model instructions or smoke reference were
edited. Minimal ownership exceptions were `contracts/rollout.py` for snapshot
budgets and scripted validation, `contracts/log.py` for hidden truth columns,
and `log/_rows.py` / `log/runfiles.py` for legacy table/SQL compatibility.
Confidence is high in the tested mechanical guarantees. Rescue improvement
and combined-harness behavior remain unmeasured until their deferred runs.

## Round 3: second review repairs

A–B: route failures still stop further subscription calls, preserving the login
boundary. Runs containing any skipped call are ineligible for scoring. The
eight-attempt real-registry fixture fails on call three (timeout or unoffered
answer), preserves two earlier calls and the failed call, records six later
skips, and proves refusal before operator service and from standalone `report`.
`decision_s_total/median/p95` and production model-time estimates include only
actual VLM calls (`ok` or `failed`); rules/skips contribute only to the separate
`all_decisions_s_total` and charged robot time. `calls` retains its historical
meaning of all trace records; `model_call_attempts` is the model denominator.
For a timeout-rule report the model count/total is zero and its percentiles
are null. The review's 20/21/22-second fixture now has a 21-second median,
63-second sum and 21.9-second p95, despite two rule and two skipped records.

C: exactly one factual prompt sentence changes: recover spends the remaining
ticks, holds do not, and retry gets its own nominal attempt budget once per
attempt. All other prompt content and both skill instructions remain unchanged.

D: `results/round3_eligibility.json` records a read-only check of the same 60
failed D2 trajectories through the first look at tick 160: the added height
guard reduces eligible anchors from 13 to 10. It rejects e000661 and e001295
(the reported 32/26 cm high closes) and conservatively e001669 (3.5 cm high).
No future telemetry or object geometry is used. Source-hashed excerpts in
`tests/data/supervisor_pregrasp_d2.json` test the two mid-air rejections and
retention of e000148's low approach. These are eligibility checks, not new
rescue outcomes. General object presence remains a visual judgement; physical
return at the bowl is tested with the existing CPU LIBERO expert fixture.

E–F: resumed jobs initialize gripper history from the saved command. Twelve
replay/continuation fixtures check both real transitions and absence of phantom
events. Tests assert the exact options-guard error, the saved retry-used flag
and its effect after restore. The repeated-hold test limits its fixture to two
calls, so removal of the policy-tick guard fails promptly rather than looping.
These tests live in `test_supervisor_round3.py`, with the strengthened earlier
tests beside them. Deferred-command tests check the frozen cohort, complete
reference-key inventory and rejection of unsupervised/canary/replay drift.

The live harness must apply the same skipped-route measurement refusal and
actual-call timing denominator when using this adapter, in addition to the
factory, queue, failure/backstop trace and robot-time requirements above. No
second claim-4 harness was built. No GPU run, model call or smoke-world run was
started in this round; the two commands above remain for the orchestrating
session. New nightly values and repaired D2 effectiveness remain unverified.

Final round-3 checks on **92b8b66** both exited **0**: `make check` passed
**1,160 tests**, with 2 skipped (lint, formatting and types also clean);
`make check-sim` passed **61 tests**, including all three physical supervisor
tests. Their pytest durations were 483.81 and 540.91 seconds respectively,
not model latency. The existing 240-case toy chain reference is unchanged.
Four focused mutations (missing options guard, missing saved/restored retry-used
flag, and missing policy tick after a hold) each fail the intended test within
six seconds, without hitting the 25-second subprocess limit. This is not a
rerun of the entire review mutation suite. Logs are under
`~/.cache/robots-supervisor/round3/`; the source commit, hashes and check receipts
are recorded in `results/round3_verification.json`. Subsequent delivery edits
only record these results. Confidence is high in these mechanical regressions;
effectiveness on the repaired D2 run remains unknown.


## Integration with the reviewed fault engine

The branch first merges `merge/batch-3` at `e4fd448`, then the reviewed
fault repairs at `8c3e6e2`. The two `episode.py` conflicts keep both pieces:
the local fault-command import precedes the snapshot-start guard, and the
loop sets both the decision-idle flag and the command-frame flag. The import
stays local to avoid the fault registry's bootstrap cycle. The effective
control mode wraps all action hooks. In-worker commands are normalized
before logging, holding, pre-grasp tracking or snapshotting, after reading
the frame of the command just produced. Snapshot remaining budgets, the
one-use retry grant, frozen hold clocks and strict scripted traces stay intact.
The Makefile merged without a conflict: check, scoring, baselines and soak
commands from both parents remain. No line in the fault library or its
reviewed tests is edited beyond accepting that branch verbatim.

Command frames in the combined engine:

| Producer | Frame and reason |
| --- | --- |
| Retry and recover (`Recovery.commands_in_world_frame=False`) | Policy frame: absolute targets derive from saved canonical commands; relative recover uses differences of observed positions. Placement acts once. |
| Decision hold | Policy frame: reuses the already-normalized saved command (zero displacement in relative mode). The loop declares `command_world_frame=False`. |
| Degrade / continue | Use the underlying producer's declared frame; degrade only truncates server chunks and never changes their coordinates. |
| In-worker expert fixtures / `Teleoperator` / `ToyTeleoperator` / `LiftPolicy` | World frame (`True`): they construct target poses in simulator coordinates. The fault branch supplies the adapters' declarations. |
| Human-model teleoperator | Policy frame during saved-command reaction holds, world frame during expert movement; its existing dynamic declaration is read after producing the action. |
| Snapshot-event fixture `Operator` | Explicitly policy frame (`False`), because it reissues a saved command rather than computing a world target. |

The new numerical-cell integration tests exercise all offered interventions
with three real placement hook types, both a server policy and an in-worker
world-target policy, and scripted proof of each result. They check that
holds and primitives are not translated twice, retry retains its full active
budget, and a hand-over snapshot retains its canonical command and remaining
budget. A further three cases switch effective control to relative and test
recovery without subtracting a placement offset from a displacement. These
are command/replay tests, not D2 physics or rescue evidence.

The deferred wrappers already run fresh outputs at the exact detached commit,
so their execution logic needs no integration change. Use the two commands
above with the final merged branch commit, in the detached run checkout.
`repeat_d2.sh` regenerates **both** timeout and VLM arms, including their human
continuations, from the same 80 frozen identities (60 original failures,
20 successes); both `measurement.json` receipts must name that commit.
The original outcomes define the fixed cohort, not a reused timeout arm.
The nightly's camera/delay/gripper world inputs do not include placement
faults; its unsupervised-digest/canary refusal remains appropriate. New live
reference digests require the deferred GPU run, which has not run here.

Final integration verification on **8dead7d**: `make check` exited **0**
(**1,810 passed, 2 skipped**, lint/format/types clean); `make check-sim`
exited **0** (**61 passed**, including all three physical supervisor tests).
Each was invoked once, sequentially. The simulator check used
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Pytest durations were 2,850.45
and 341.78 seconds respectively; these are shared-machine verification
durations, not supervisor decision latency. The unit run spent substantial
time in the newly imported statistical tests.

The focused combined suite passed **115 tests**, including **36** new
placement/supervisor cases. Four in-memory engine mutations (nominal rather
than effective mode, missing world normalization, missing explicit-frame
guard, and reading the frame before producing the command) each failed its
intended reviewed fault test in under five seconds. Neither the checkout nor
the reviewed fault tests were modified for those probes.

From the detached checkout, all **975** unsupervised toy cases (25 identities,
13 budgets, healthy/camera/delay) matched `b34d7ad`: complete chain arrays,
outcomes and lengths. All **240** cases in the existing supervised toy
reference matched too, with **200** scripted proofs. **No stored toy reference
moved or was rewritten.** The only extra reference file is a fresh candidate
in the new evidence directory. The smoke reference also remains untouched;
its GPU proof and the nightly live reference transition remain deferred.

`results/merged_engine_verification.json` records source commits, exact
counts, artifact hashes and both deferred command forms. Evidence is under
`~/.cache/robots-supervisor/merged-engine/`. The detached run checkout remains
at **8dead7d** and is idle. Later delivery edits record evidence only. The
fault-library tree and reviewed fault-engine test are byte-identical to
`8c3e6e2`; the list of additionally edited lines there is **empty**. No prompt
content or model instruction was changed. Confidence is high in these tested
mechanical guarantees; repaired D2 effectiveness and new nightly digests
remain unmeasured until the two deferred commands run.

## Follow-up protocol, 2026-10-10 (predeclared)

The retry ceiling uses the unchanged frozen `d2-sample-v1` 60 failures at
steps **120, 160, 200** (6, 8, 10 active seconds). The first is after the
initial grasp for most failing attempts (58/60 close at all; first-close
quartiles 49/57.5/105.5 ticks); the next is the existing first look; the last
is the timeout rule. Previously inspected D2 frames show missed grasps,
empty transport and basket-rim stalls across these times. No outcome from
this new probe was available when these times were chosen.

`python -m fleetlab.live.vlm.ceiling SAMPLE NEW_OUTPUT` runs all three arms,
with two workers, no model, no operator, and the current eligibility guards.
An unavailable retry produces one continue decision, never a delayed retry.
Every available retry retains its own nominal budget, once. Zero decision
latency still incurs the engine's one physical hold tick. Each arm writes
its complete run and sampled scripted proof; `ceiling.json` reports retried,
ineligible, rescued and independently successful IDs, plus the union of
rescues over the three times. This is a best-of-three ceiling under today's
primitive and guards, not a mathematical upper bound over every possible
trigger time. The one heavy invocation has an outer 1,490-second timeout
plus a five-second kill grace, counted after obtaining the GPU lock.

The merged-engine result at `d2-merged-v4` has 53 rescues versus timeout's
49, with 1,086 versus 1,077 operator-seconds. Its two autonomous rescues are
`e001624` (retry then continue) and `e000384` (continue twice); only **one**
of the six retries finished without an operator. Four retries handed over
at 110 active ticks after starting the reset: 70 primitive + 40 policy.
The ceiling probe deliberately grants the whole retry budget.

### Scratch live-harness integration

The read-only `stage5/live-rest` revision `943f434` was merged only in
`~/.cache/robots-supervisor/harness-checkout`, not into this branch. The
scratch Makefile conflict keeps both sets of targets. At scratch `b78f67e`,
`tests/test_supervisor_harness.py` passes all three light toy checks. Seven
source attempts are actually run and chain-checked. Only the schema-valid
model response, toy expert and generated toy-world readiness are fixtures;
VLM input construction/scheduling, engine, FIFO human admission, physical
queue holds, counterfactual continuation and scoring use their real code.
The CLI hardwires the LIBERO RunConfig, so the test calls its underlying
`pass_run.execute` with the toy RunConfig; it builds no parallel harness.

Six hand-over proposals at ratios 1, 2, 4 admit **0, 6, 6** respectively.
The latter two and timeout each have **30 robot queue-seconds**, two operators
and seven final successes (six rescues). Ratio 1 keeps the one source success.
All four records go through `scoring.score_files`. These numbers are fixture
integration evidence, not measured model judgment or human performance.
Records: `~/.cache/robots-supervisor/harness-toy-v4/` (three tests, exit 0,
16.72 seconds). The tests assert the scorer's exact rescue counts, queue time
and operator time against timeout: both accepted ratios and timeout consume
6.7 operator-seconds. They also distinguish real adapter subscription overhead
from the fixture's zero model timer. `results/harness_20261010.json` records
these checks and artifact hashes.

The other two checks expose real integration blockers, tracked as
the terminal and failed-call harness repair: terminal-backstop and failed-call hand-overs both raise
`StopIteration` in pass_run's risk lookup. Its RecordingSupervisor sees only
successful decide returns, while the engine also generates these records.
Before a VLM claim-4 pass, the harness must reconcile that event stream with
the canonical engine trace, retain failure context/risk and the declared
terminal risk, handle nullable model time, reject skipped-route measurements,
and distinguish model calls from rule records. Dynamic queue feedback to the
proposal model remains absent; only admission sees occupancy.
The model's cost settings must also match each admission ratio explicitly.
No live-rest source was changed.

### Launch preflight repair

`repeat_d2.sh` now defaults its sandbox root to
`/var/tmp/robots-supervisor-repeat-UID`, independently of output location.
`ROBOTS_AGENT_ROOT` can still select another valid private root outside home.
It pins the existing machine convolution database via `FLEETLAB_MIOPEN_DIR`,
checks every frozen identity's software/database before queueing, and runs
free runtime gates before either GPU arm. It does not create a new database.

The optional fourth argument is a JSON object of VLMSettings. `--dry-start`
runs the same manifest/settings/factory/configuration path for both arms and
stops immediately before `engine.fleet.run`; no simulation, model call or GPU
queue entry occurs. Each arm writes `started.json`, `manifest.jsonl` and
`dry-start.json`, never a completed measurement/report. A real invocation
writes both fresh arms, including human continuations; each uses the same
80 frozen identities and this checkout's engine. Existing outputs are refused.

The repaired command was **actually dry-started this round** from clean,
detached `a9e0bed` in `~/.cache/robots-supervisor/revision-checkout`:

```sh
bash src/fleetlab/live/vlm/repeat_d2.sh a9e0bed \
  /home/cs/.cache/robots-supervisor/d2-sample-v1 \
  /home/cs/.cache/robots-supervisor/dry-start-v1 --dry-start
```

It exited 0, verified database `7751323820effb28`, passed the Codex runtime
gates and reached both `fleet.run` boundaries. Receipts are in each arm's
`dry-start.json`; the log is `~/.cache/robots-supervisor/dry-start-v1.log`.
This verifies the real startup path, not GPU inference or a revised outcome.

### Offline call comparison protocol

`python -m fleetlab.live.vlm.offline STORED_RUNTIME_RUNS MEASUREMENT NEW_OUTPUT`
verifies all 92 runtime receipts against the measured decision table and requires
the stored system instructions to equal the reviewed live instructions. It hashes
the retained prompts and images, then submits the unchanged requests once each
at `gpt-6-astra` medium, `gpt-6-astra` low and `gpt-6.1-sol` medium: **276 calls maximum**, no retries,
two concurrent slots, rotating the setting order for each input. Agreement is
with xhigh's decision, not with a known correct intervention. Failed/invalid
answers stay in the denominator. Timing is offline subscription wall time,
including gate recheck, PNG encoding, CLI and answer validation; model time is
separate. Shared service load and warm-runtime/cache effects are uncontrolled.

The initial `offline-efforts-v1` attempt made **zero model calls**: an extra
generic analysis-text scan rejected the reviewed live payload's `index`,
`nominal` and `units`. That was the wrong audit interface for SupervisorContext.
The comparator now calls the **unchanged existing live-supervisor gate**
(`measure.gate`), including its instruction/source, sandbox and login checks,
then requires its verification record on every call. Neither runtime controls
nor input text are changed. The failed scan remains in its original output.
The existing assignment authorizes this established route; no new permission
or prompt change is needed. A regression proves a refusal of the actual live
gate prevents every call.

### Offline comparison results

At `f53cd8f` the comparison completed with exit 0: **276/276 valid calls**,
92 per setting, with no retry, login failure or skipped call. The stored xhigh
column below is the original in-loop reference, not another 92 calls.

| Model / effort | Agreement with stored xhigh | Decision median | Decision p95 | Total decision time |
| --- | ---: | ---: | ---: | ---: |
| gpt-6-astra xhigh, stored | reference (92 calls) | 21.38 s | 40.08 s | 2,173.84 s |
| gpt-6-astra medium, offline | 83/92 (90.2%) | 9.03 s | 11.22 s | 835.31 s |
| gpt-6-astra low, offline | 86/92 (93.5%) | 9.15 s | 11.50 s | 848.50 s |
| gpt-6.1-sol medium, offline | 82/92 (89.1%) | 10.27 s | 12.73 s | 947.35 s |

All three settings retain all six original retry decisions. Low has the most
agreement in this sample; medium and low have similar latency. Confidence is
high in these observed timings and agreements, but this does not establish
that either effort rescues the same attempts: different holds change physics,
and agreement with xhigh is not correctness. The timing conditions above also
prevent attributing the whole speed difference solely to reasoning effort.

The 276 system/prompt pairs and 1,656 PNGs were verified byte-identical to
their stored sources. All 276 CLI argument records match the requested model,
effort and fast tier; every reported model matches. Compact results and raw
artifact hashes are in `results/offline_efforts_20261010.json`; raw requests,
responses and timings remain in `~/.cache/robots-supervisor/offline-efforts-v2/`
and its recorded `/var/tmp` runtime root. The failed v1 scan made zero calls,
so the assignment's total subscription use is 276, below its 300-call cap.

The paired pilot's queue and robot-time scopes need care. Although every model
input says two operators are free, the continuation FIFO actually accumulated
**1,440.0 queue-seconds for timeout and 869.4 for VLM**. These match physical
idle ticks in the continuation metrics, not merely unused reservations. Its
735/2,944-second robot totals cover the original attempts (including VLM
holds), excluding those continuation waits and operator motion. Every measured
continuation consumed nonzero operator time, so this qualification changes no
rescue count. The full live harness must score its actual queue and all robot
time; this pilot is not an equal-operator-time claim-4 comparison.

### Retry ceiling and the predeclared revision

The single bounded GPU job at `56776f7` exited **0**. Each arm ran the same
60 original failures and 20 sampled scripted proofs; **all 60 proofs matched**.
The three arm durations sum to 1,180.42 seconds, excluding queue waiting and
process startup. The outer 1,490-second timeout was not reached.

| Forced retry at active tick | Eligible retries | Rescued by retry / 60 | Success without retry |
| --- | ---: | ---: | ---: |
| 120 | 9 | 4 | 1 |
| 160 | 10 | 2 | 0 |
| 200 | 10 | 3 | 0 |

The union is **4/60 (6.7%)**: e001083, e001406, e001624 and e001662, all
already rescued at tick 120. The extra success e001018 at 120 used continue,
so it is excluded from the retry ceiling. This is the best of these three
times with the current guards and one minimum hold tick, not an upper bound
over every possible trigger or model hold duration. No person or model was
used. Raw runs are in `~/.cache/robots-supervisor/retry-ceiling-v1/`; compact
counts, hashes and replay evidence are in `results/retry_ceiling_20261010.json`.
Three of those four attempts were already rescued by the historical timeout
operator; e001083 was not. The autonomous ceiling is therefore not four
additional total rescues against timeout, and does not predict the outcome
of a later operator continuation after a failed retry.

**Revision 1, declared after these results and before measuring it:** use
`profiles/one_early_look.json`: `gpt-6-astra` at low effort, one model assessment at active tick
120, with the existing eligible options. A retry uses its full once-only
nominal budget with no second assessment; continue/recover/degrade use the
remaining policy budget, and hand over happens either when this one assessment
chooses it or through the existing terminal-failure backstop. Stop remains
available. No instructions or prompt wording change; the live call count and
options correctly describe this one-assessment schedule. The default xhigh,
two-assessment configuration remains available for reproducing the old design.

The reason for 120 is its four rescues versus two and three at the later
times. Its successful retries took **151, 175, 207 and 236 active ticks**,
all beyond the old second assessment at 110 retry ticks; they support allowing
the whole budget to run. Low retained 86/92 stored decisions, including all
six retries, at a 9.15-second median offline cost. That supports testing it,
not declaring equivalent rescue quality. The earlier look may interrupt more
of the 20 originally successful attempts; an early hand-over may also consume
unnecessary operator time. These are explicit costs to measure, not reasons
to filter those attempts out.

The prepared paired run must rerun both timeout and this profile on the same
80 frozen identities, and compare autonomous/person rescues, damaged successes,
actual model-call timing, operator time, original robot/retry time and separately
reported queue wait with `d2-merged-v4`. This joint schedule/effort revision does
not isolate each change's causal effect and is still a development pilot;
claim 4 belongs to the repaired live harness. No revised D2 outcome has been
measured in this round.

The runnable code and profile are frozen at **5029b50**. That exact prepared
command was started from the clean detached `revision-checkout` with
`--dry-start`, exiting **0** and reaching `fleet.run` for **both arms**.
The two generated manifests preserve the exact same 80 identity objects;
the VLM receipt confirms `gpt-6-astra` low, tick 120 and
one call. The runtime root is `/var/tmp/robots-supervisor-repeat-1000`, and
the pinned database hash is `7751323820effb28`. Evidence is in
`~/.cache/robots-supervisor/dry-start-revision-v1/` and the adjacent `.log`.
The dry start performs no model call or simulation and queues no GPU job.

The round-5 command below is historical and cannot be reused: its output
directory contains an interrupted timeout arm. Use the round-6 command at the
end of this README instead:

```sh
cd /home/cs/.cache/robots-supervisor/revision-checkout
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash src/fleetlab/live/vlm/repeat_d2.sh 5029b50 \
  /home/cs/.cache/robots-supervisor/d2-sample-v1 \
  /home/cs/.cache/robots-supervisor/d2-one-early-look-v1 \
  src/fleetlab/live/vlm/profiles/one_early_look.json
```

The script runs two queued GPU jobs and writes fresh `timeout/` and `vlm/`
directories, each with the full run, scripted proofs, human continuations,
`measurement.json` and `report.json`. Neither the old `d2-merged-v4` arms
nor the sample is modified or reused as an intervention result. The old
reports remain the historical comparator; the new timeout arm controls for
the engine used by the revised supervisor. Subsequent commits recording
verification in this README do not change the runnable commit above.

Final verification on the **5029b50** code: lint and formatting exit 0;
mypy checks **383 files** with exit 0; the focused supervisor suite exits 0
with **127 passed, one scratch-harness skip** (15.45 seconds). The separate
scratch harness check passes all three tests at `b78f67e`. `make check-sim`
exits **0**, **61 passed, one scratch-harness skip, 2,403 deselected**
(489.28 seconds), including the physical retry/recover/degrade tests. The
one-call regression proves a full 280-tick retry, a later terminal hand-over
and an identical scripted chain; it also covers continue-to-terminal and
retention of the default final-reassessment options.

`results/followup_verification_20261010.json` records commands, source commits,
dry-start settings and artifact hashes. No full `make check`, smoke-world,
revised D2 run or additional live model call was run. No live-rest source,
engine source, stored reference or original result was edited in this round;
the only main integration was the requested `2c5813e` merge. Confidence is
high in the observed ceiling, offline agreement and mechanical checks;
the revised supervisor's rescue rate and cost remain unmeasured.


## Round 6: review repairs and a fresh measurement command

The toy harness test pins its own synthetic code revision,
`supervisor-harness-fixture-v1`, with `dirty=False` on its pass records. It no
longer treats the current checkout as measured code. This is a fixture for the
real scorer, whose refusal of dirty production measurements is unchanged.
In an isolated checkout of the harness at `adf82f5`, with these test files
modified, all three harness tests pass, including the cost-ratio scorer test.
The two cases asserting the existing `StopIteration` remain until the harness
builder repairs the terminal and failed-call handling; they are not evidence of correct terminal or
failed-call handling.

Regression tests now intercept the real adapter call with a fixture response
to check the profile's effort and a nondefault model. Offline comparison tests
intercept every replay call to assert unchanged system and prompt text, image
bytes and order, schema, model and effort. Ceiling tests reject a failed replay
proof and verify that only the 60 original failures enter each arm, excluding
the 20 source successes. These use synthetic inputs, with no subscription or
GPU call. The engine-to-adapter test supplies two different queue callbacks:
zero waiting/two operators free, then three waiting/none free. Only the queue
field changes in the model's prompt. The existing adapter already serializes
both fields; dynamic chronological scheduling in the full harness remains the
parallel builder's part of the live queue feedback.

The offline effort evidence remains 92 observations at ticks 160, 230 and 270,
with a second assessment available, not the revision's one assessment at 120.
Low and medium had similar median latency; agreement cannot establish quality
at the earlier assessment. Tick 120 was chosen on the same 60 failures used
in the paired pilot, and those failures overlap the claim-4 world cohort.
This is development selection, not independent validation. Confidence is high
in the mechanical checks; the revised profile's rescue quality remains unknown.

The revised runnable revision is **0ba4874**. Its clean detached checkout is
`~/.cache/robots-supervisor/round6-dry-checkout`. The fresh dry start exited 0,
passed the Codex gates and database check (`7751323820effb28`), and stopped
before `engine.fleet.run` in both arms. Each manifest preserves the same 80
source identity objects; the VLM settings are `gpt-6-astra` low, tick 120, one call.
Evidence: `~/.cache/robots-supervisor/dry-start-round6-v1/`. This proves
preflight and launch preparation, not a model call, physics run or rescue.

To start the actual measurement, from that immutable checkout:

```sh
cd /home/cs/.cache/robots-supervisor/round6-dry-checkout
env TMPDIR=/var/tmp/robots-supervisor OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  bash src/fleetlab/live/vlm/repeat_d2.sh 0ba4874 \
  /home/cs/.cache/robots-supervisor/d2-sample-v1 \
  /home/cs/.cache/robots-supervisor/d2-one-early-look-v2 \
  src/fleetlab/live/vlm/profiles/one_early_look.json
```

The new output name was absent at preparation. The wrapper refuses an existing
name, preserving both the stopped v1 run and any later results. It requires
about **74 model calls**, the reviewer's estimate of 60 failures and 14 source
successes reaching tick 120, and **two queued GPU jobs**, one per arm, with
proofs and operator continuations. It must wait for available Codex allowance.
No actual paired run, model session, GPU job, full `make check`, simulator suite
or smoke world was run in round 6; the hand-over restricts verification to
lint, types and owned tests. Harness integration and independent review remain
with the orchestrating session.

Round-6 checks: scoped lint and formatting exit 0; mypy checks 393 files and
exits 0; the owned nonsimulator suite passes 145 tests with one harness-module
skip, exit 0. The typed queue callback cleanup was followed by all eight
revision tests passing. All seven named single-place breaks from finding 14
fail their new regression in the isolated scratch checkout. Receipts, artifact
hashes and the exact evidence scope are in
`results/round6_verification_20261011.json`.
