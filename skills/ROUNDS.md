# Development analysis rounds

Builder record only. This file is outside the per-skill directories and is
never mounted into an analysis sandbox. Each result directory is a new record;
no result or world is overwritten. Scores remain provisional until the oracle
repairs on `fix/oracle-review` are incorporated.

## Round 0 — development runner, before the first analysis

Started 2026-10-09 at 17:33 UTC. The six-hour session limit is 23:33 UTC.

Changes: a `make analyse` command for either condition, gpt-6-astra at xhigh
and fast service tier; public day ranges from the world specification;
experiment queues under a new output root; source audit with the full fleet
before the first session and after every experiment queue; scoring even when
a session leaves no answer. The scoring builder's first command was merged
from commit `89f988a`. No analysis skills have changed yet.

Why: the miniature command had fixed dates and a Claude-only backend, and
wrote experiments inside its world directory. Development runs must use the
cleared export and leave stored world records unchanged.

Before/after scores: not measured yet. D1 is waiting for its `D1 EXPORT CHECK:`
clearance; no analysis has been started without that clearance.

Preflight follow-up: the runtime's Codex gate initially rejected the CLI's
empty SQLite maintenance lock. A repair adds an exact-name allowance only
for an empty regular file; nonempty files, directories and links are refused.
The Codex-only `make agents-gate` then exited 0; 54 focused tests passed
(50 runtime, 4 development-command). The runtime change is `agents/checks.py`.

An independent boundary review found an accidental cross-world input gap:
an export with the right day range could belong to another fleet. The command
now requires the fleet id and visible episode rows to match the world's base
runs. Experiment-result text joins the export in the audit before being given
to the agent. The reviewer found no remaining actionable boundary issue by
static inspection. Live analysis remains unverified; the final independent
delivery review is still the orchestrator's next step.

### Stopped at the export boundary, 2026-10-09

The export builder's 17:52 UTC comment reports that D1
cannot be cleared: a probe reproduced recovery of both hidden episode
seeds for all 1,008 observed episodes using the public world id and clocks
plus mounted identity code. The word/structure scan is clean, but that does
not establish blinding. No fault schedule was recovered in that probe; no
model or authenticated sandbox attack was run. This assignment explicitly
requires stopping on a failed D1 export check. No analysis model call or
experiment request has been made by this builder.

| Condition | Completed repeats / required | Detectable-cause recall | Decoy rejected | Singleton-gain Spearman | Share of achievable gain at k = 3 |
|---|---:|---|---|---|---|
| Logs alone | 0 / 2 | not measured | not measured | not measured | not measured |
| With experiments | 0 / 2 | not measured | not measured | not measured | not measured |

The loop has no measured before/after comparison and no evidence about what
the development-world analysis agent gets wrong. The next action is to resolve
the public-to-hidden seed channel and obtain explicit clearance for the exact
export. Runtime findings on refusal messages and on audits at the common launch
boundary are also open. This command audits
replies and each new export, but it does not repair all other runtime callers.

Code and checks at handoff:

- Scorer `89f988a` merged as `6aee23d`; runner `7b67490`; boundary and runtime
  lock fix `07789f6`; export repair `38f4ffa` merged as `053a95e`.
- `make check` at `07789f6` exited 0: 1,090 passed, two skipped. The earlier
  sandboxed run stalled after 303 passes and was terminated (exit 143);
  the host rerun passed without an engine edit. Evidence and causal limits
  are recorded in `/tmp/codex-skills/check-07789f6.log`.
- `make check-sim` from detached `7b67490` exited 0: 58 passed. No physics or
  engine change was authored. The smoke check remained queued and was
  cancelled at the failed export boundary; no stored chain comparison ran.
- After the export repair merge: 129 focused tests passed, and the Codex
  `make agents-gate` exited 0. The stricter audit also caught scheduling-key
  labels in the public prompt; the prompt now uses ordinary prose and both
  conditions have a passing prompt-audit regression (four command tests).
- No full `make check` or simulator suite was rerun after that final prompt
  adjustment. Final independent review, smoke verification, live analysis,
  experiment-queue execution, scores and repeated pass marks remain undone.

No analysis skill procedure was tuned. D2 and D3 were not analysed. Stored
worlds, fixtures and prior results were neither rewritten nor removed.

### Resumed under explicit development clearance, 2026-10-09 17:58 UTC

The orchestrating session cleared D1 for procedure development in its decision,
despite the identifier weakness above. The exact rebuilt
export is `/home/cs/.cache/robots-probes/worlds/w41/export-review-20261009T173751Z`.
Its term and structural scans passed; the original export remains untouched.
The seed channel does not recover a fault schedule, trait or start state from
the mounted bundle. All D1 scores remain development-only and provisional;
reportable evidence requires a neutral fleet id or a regenerated world as
well as the oracle repairs. The renewed six-hour limit is 23:58 UTC.

The first logs-only run uses committed runner `53eb75d` in detached checkout
`robots-skills-run-r00c`, with new results at
`~/.cache/robots-probes/skills/20261009-r00/logs-1`. Before the second run,
the launch-boundary repair will enforce a current audit at the lowest launch path. Before
any experiment run, the reply repair will sanitize and validate queue replies.
Removing seed derivation from the mounted closure is also assigned. Prompts
and skills must not invite identifier interpretation or seed derivation;
any such attempt in a tool log will be recorded as a finding.

## Round 1 — launch boundary, public replies and first D1 attempt

The first logs-only attempt at `53eb75d` ran for 94.417 seconds at gpt-6-astra xhigh.
It completed 14 tool calls (description, skills and one counting query), then
lost the MCP transport during the triage outcomes block. Five subsequent tool
attempts returned `Transport closed`, including the attempted memory write.
Neither diagnosis nor ranked fixes was submitted. The scorer ran: recall,
decoy rejection, gain correlation and k = 3 share are all unavailable, not zero.
No experiment was queued. Inspection found no identifier interpretation or
seed derivation attempt.

The exact SQL succeeds on the host and in the original tool sandbox, including
replay of all 14 preceding calls. This does not identify the cause. The next
run gets request-start and native stderr logs; a transport lost after real
work is classified as a failure and is not automatically rerun unchanged.

Changes required before the next attempt: a fresh audit at `run_agent` over
current export, source, skills, prompts, schema, box and all persisted state;
content-bound CLI/environment verification; fixed experiment refusal codes
with host-only exception details and strict request/result/export validation;
queue-hashed cached plans; lazy contract exports that exclude identity/seed,
oracle, world and rollout modules. No analysis procedure has been tuned.

An independent review found and prompted repairs for direct stored-result
validation, malformed JSONL records, omitted cached replies, CLI override hooks,
and the synthetic runtime measurement's invalid arm records. The runtime's
small measurement fixture now explicitly exercises refusal/continuation; it
no longer pretends that five existing episodes can supply eight new arm
attempts. Successful experiment execution is a separate engine test and the
pending real experiment condition. This review supplements the orchestrator's
required later independent review.

Before/after score comparison is pending the first completed analysis. The
logs-only attempt above remains an infrastructure failure in the denominator.
Runtime checks, final focused regressions and final commits are recorded below
when complete. Scores from all D1 runs remain development-only and provisional.

Round-1 verification: 71 focused tests passed (launch audit, public replies,
tool server and engine experiment queue). Codex-only `make agents-gate`
exited 0, including the local wire capture and isolation check. An earlier
combined pass had 123 tests pass and one obsolete fixture fail; that fixture
had passed observed episodes off as experiment arms and now checks their
rejection before recording a valid refusal. Full `make check`, simulator and
smoke verification are still pending the coherent delivery commit.

The scoring extension through `28fe52a` was merged as `df0bedd`, and oracle
repairs through `0fae966` as `af85dbb`. The runner now labels both scoring
conditions explicitly and records repeat numbers. New analyses have unique ids
so the shared runtime ledger cannot conflate their costs. D1 remains provisional
because the identity-cleared replacement export/world and additional oracle
coverage are not yet used; merging repair code does not invent missing rows.

## Round 2 — reproduce and repair the tool lifetime failure

The second logs-only attempt at `ecf30ef` ran for 75.940 seconds at gpt-6-astra xhigh
and completed 15 tool calls. Its next SQL call never reached the server:
request-start and returned-call counts both remained 15, with empty server
stderr. The runtime classified this as `tool_server` and did not retry. No
diagnosis, fix set, memory or experiment was submitted. All four agent pass
metrics remain unavailable; this is the second infrastructure failure in the
denominator. No identifier interpretation or seed derivation was observed.
Results are in `~/.cache/robots-probes/skills/20261009-r01/logs-1`.

A CPU probe reproduced SIGKILL (-9) after the thread that spawned a nested
`bwrap --die-with-parent` exited, while its parent process remained alive.
This is a concrete mechanism consistent with both failures, not yet proof
of the live CLI's cause. Nested tools now rely on the outer PID namespace
for cleanup; the outer CLI keeps its parent-death signal. A new free gate
checks the old flag as a failing control, successful tool calls after thread
exit without that flag, and descendant cleanup when the outer sandbox exits.
It passes. Gate receipts are also keyed by environment and implementation,
preventing a parallel builder's gate from replacing a running analysis's
receipt. No analysis skill has yet been tuned.

Verification: the Codex-only gate exited 0 with the lifetime test; 68 runtime
and launch tests passed, followed by three focused receipt regressions.
At `ecf30ef`, full `make check` exited 0 (1,170 passed, two skipped), and
`make check-sim` exited 0 (58 passed). Its smoke job is still queued; no
reference has been changed. The next fresh run will test the lifetime repair
on the real analysis path. Before/after diagnosis quality remains unmeasured.

## Round 3 — first completed analysis and cleared experiment continuation

At `ef43a6e`, logs-only repeat 1 completed in 898.442 model seconds, with
93 successful tool calls and both outputs submitted. It read scene and wrist
frames, checked symptoms against telemetry, controlled for task mix and other
affected robots, and excluded suspect outcome labels from rate estimation
without crediting their apparent failures as recoverable gains. No identifier
interpretation or seed derivation was observed. This is the first completed
analysis after two infrastructure failures, not a selected best run.

| Condition / repeat | Detectable-cause recall | Decoy rejection | Singleton-gain Spearman | Achievable share at k = 3 |
|---|---:|---:|---:|---:|
| Logs alone / 1 | 6/6 = 1.00 | 1.00 | 0.9461 | 1.00 |
| Logs alone / 2 | pending | pending | pending | pending |
| With experiments / 1 | pending | pending | pending | pending |
| With experiments / 2 | pending | pending | pending | pending |

The top three recovered a net 19.9405 points of whole-fleet success over
336 scoring attempts (67 rescued, none broken), matching the attainable
three-action ceiling. Correlation uses the eight replay-bearing actions;
the remaining two are teleoperation/training recommendations, excluded by
the design. Mean absolute singleton-gain error was 0.5580 points. Six of
eight true gains fell inside the reported intervals. Baseline margin and
chance remain unavailable; these four session marks do not establish the
full gate-4 evidence claim. All numbers remain development-only.

The agent also named a bare-site outcome-checker/logging problem based on
four short labelled failures whose last frames looked successful. There is
no planted logging cause. The scorer marks this seventh cause vague, neither
a hit nor a false positive; its reported precision of 1.00 must therefore
not be read as seven correct diagnoses. The second and third gain estimates
are reversed relative to their true gains. A high-gain delay repair was
slightly above its interval, while the lighting decoy was assigned exactly
zero with a zero-width interval although its repair loses one scoring
attempt. These are uncertainty-calibration and unsupported-attribution
findings; no procedure has been tuned yet.

The first experiment-enabled session also runs at `ef43a6e`. Static inspection
found that its queue would republish the original observed run,
restoring the historical stall strings removed from the cleared export.
The audit would refuse them. The pending heavy-queue waiter is being held
before simulation; the completed model session will be continued, not repeated.
`2f154a1` adds `--base-run` with exact cleared-export reproduction and a
new-output `--resume-from` for an ended first queue that has not simulated.
Independent review prompted checks for source/output containment, exact saved
queue/session/request agreement, and original export-manifest identity.
Six runner regressions and mypy pass; real continuation remains pending.

`make check` at `ef43a6e` exited 0: 1,171 passed, two skipped. The latest
scorer through `ddd2e9c` was merged as `e2d9207`; its 20 focused tests pass.
The first completed score above was produced by the immutable run checkout's
scorer and will also be rescored into a new directory with that merge.

## Round 4 — preserve saved analysis prose at the launch boundary

The first experiment session completed in 874.614 model seconds with 86 tool
calls and five requests totalling 720 arm attempts. Its queue was held before
simulation. The cleared-base continuation at `50035b1` passed exact export
reproduction and resume validation, but inspection caught another blocker
before simulations: the current-data audit treated ordinary notes such as
“camera faults”, “success=.6” and “windows: observed days” as hidden data.
That queue was held again; no model session was repeated and no simulation
was discarded. The second experiment repeat is being held at the same boundary.

A repair narrows the exception to those standalone words in tool-owned
notes and output files, then rescans complete content with the existing scanner.
It preserves the scanner's public vocabulary rules and rejects later hidden
compounds, actual kinds, traits and canaries, including when ordinary prose
appears first. Filenames, extra files, results and the CLI box remain strict.
Independent review found a vocabulary regression in an earlier implementation;
the final implementation reuses `leak_search` and includes that positive case.
All 31 launch-audit tests, lint/types and the Codex-only gate pass. Runtime
file changed in this round: `agents/audit.py`; no analysis procedure changed.

The first logs-only repeat rescored at `50035b1` into a new directory has the
same four marks: 1.00 recall, 1.00 decoy rejection, 0.9461 correlation and 1.00
top-three share. Both experiments still lack final submissions, so their four
marks are unavailable, not zero. A completed session's costs and provenance
survive continuation. Full `make check` at `50035b1` exited 0: 1,174 passed,
two skipped. The new audit repair awaits its full check and real continuation.

The second experiment repeat then finished its first session: three requests,
480 arm attempts, 86 successful tool calls, and saved notes. It too was held
before simulation. Checking those real notes exposed plural `faults:` plus
abbreviations of public configuration keys and a statistical alias
`min_detectable` (the note's shorthand for the power helper's returned minimum
difference). The prose exception now includes compound suffixes derived from
the public `ConfigKey` enumeration and that statistical alias. Full hidden
kinds/traits/canaries remain protected even if their name collides with a public
term. Numeric settings attached to a name are accepted, but letters, underscores
or hyphens after the number are rejected; review caught and tested that boundary.
Both real saved states now pass. The first full check of `2b92e07` stopped on a
formatting discrepancy; it was corrected before further measurement.

## Round 5 — second logs-only repeat and experiment continuation

Logs-only repeat 2 at `50035b1` completed in 1,070.094 model seconds with
98 successful tool calls and both submissions. Automatic scoring then failed:
the old Claude refusal-trace reader assumed every CLI event's `message` was an
object, whereas a Codex event carried a string. `2a53302` checks envelope types
and retains real refusal detection (six integration tests and
mypy passed). The existing output was scored into a new directory without
another model call. No identifier interpretation or seed derivation was found.

| Condition / repeat | Detectable-cause recall | Decoy rejection | Singleton-gain Spearman | Achievable share at k = 3 |
|---|---:|---:|---:|---:|
| Logs alone / 1 | 6/6 = 1.00 | 1.00 | 0.9461 | 1.00 |
| Logs alone / 2 | 6/6 = 1.00 | 1.00 | 0.9538 | 1.00 |
| With experiments / 1 | awaiting experiments | unavailable | unavailable | unavailable |
| With experiments / 2 | awaiting experiments | unavailable | unavailable | unavailable |

Both completed repeats select the same top three: 67 rescues, no broken
attempts, net 19.9405 points over 336 scoring attempts, equal to the ceiling.
Repeat 2 values nine of eleven ranked actions, with mean absolute gain error
0.5671 points and interval coverage 7/9. Its eight named causes include six
hits, one placement alternative at confidence 0.20 that still counts under
the frozen top-n rule, and the same vague bare-site logging claim as repeat 1.
Precision is therefore 6/7 = 0.8571 among non-vague named causes. The placement
alternative gets an expected 0.64 points against a true zero; the lighting
decoy's zero-width interval again misses a small negative true gain. These
are alternative-management and uncertainty findings, despite meeting the four
marks. Baseline margin and chance are still unavailable; development-only
clearance and oracle coverage limitations remain.

The final saved-state repair (`35fbfb0`) passed 41 focused tests, lint/types,
the Codex-only gate and checks of both real notes/outboxes/results; independent
review found no remaining issue. Both saved experiment repeats continue from
immutable `2a53302` into new `20261009-r05/experiments-{1,2}` outputs, preserving
their original analysis ids and call ledgers. Their first queues total 1,200
arm attempts. No analysis skill or general fault knowledge was changed.

Independent review of the agent's interval warning found a general helper
defect: a Wilson-adjusted rate centre is combined with a Wald
half-width while the point estimate uses the raw rate. On synthetic data,
0/10 successes against 90/90 yields gain 10.00 and interval [8.61, 8.61].
The task-standardized comparison also uses the total comparison count for
its variance: adding 1,000 unrelated-task observations can narrow an interval
without adding relevant evidence. These CPU-only probes motivate a coherent
boundary-safe, task-weighted interval and invariance tests; they do not estimate
D1's coverage. The diagnosis skill also explicitly invites
0.2–0.4 alternatives into `causes` but omits the frozen top-n naming rule.
Confidence is not an abstention flag. Both changes are held until the current
four-run comparison finishes; the next procedure round should address them
and demand corroboration before calling a logging defect from final frames.

## Round 6 — shared-runtime stop rule found in final boundary review

Review found that the generic `run_analysis` loop could send pending
requests to the engine after a session queued them and then ended with an
authentication or rate-limit error. Its `waiting` state took precedence over
the error; the batch noticed only after the whole loop returned. The development
command already had the required stop, so neither current continuation uses
the defective route. No actual login error occurred in these runs.

`agents/sessions.py` now stops before queue execution on either error. Four
focused session regressions, lint/types and the Codex-only gate pass. A queued
request remains recorded, with zero subsequent model or engine calls in the
synthetic regression. The final boundary review found no additional supported
provenance, export-binding or public-reply defect. External independent review
remains with the orchestrator. Scores are unchanged from Round 5; both experiment
conditions still await the shared heavy queue. The intermediate full check
passed 1,194 tests with two skipped; final-code full verification continues.

The scoring builder then completed D1's missing chance replays. Both finished
logs analyses were rescored into new `logs-{1,2}-score-chance-baselines`
directories, supplying the completed random-k3 and oracle-review chance runs
and the corrected baseline records at `baselines/20261009-3898b12-D1`.
All 200 random draws are now valued at every budget. At k = 3, chance's
95th percentile is zero points; mean gain is 0.3527 points. The best supplied
baseline is `worst_health`, net 8.3333 points (29 rescues, one broken attempt),
against each agent repeat's 19.9405 points. Their margin is 11.6071 points
and ratio 2.3929. The four primary scores are unchanged. The no-data and video
baselines score zero; the fixed-prior fallback scores 2.9762 points.

This is comparison with the supplied baselines only: the classical statistical
baseline is absent. Their manifest records the original export and reused
model responses after a gripper-wording correction, not new calls on our
cleared export. These limitations and D1's development-only identity clearance
prevent treating the added numbers as frozen gate evidence. No replay or model
call was started for this rescore. The scorer's documentation-only `964bf61`
handoff was merged; the executable scorer is unchanged.

Final-code verification at `4433cc6`: `make check` exited 0 with 1,200 passed,
two skipped and 62 simulator tests deselected (325.48 seconds). No code changed
afterward, only scoring-builder documentation and these result notes. The
earlier `8cad738` check also exited 0 (1,198 passed, two skipped). The only
remaining mandatory execution check is the queued unchanged-reference smoke;
the simulator suite passed 58 tests at `ecf30ef`, before later runtime-only
and integration-adapter changes.

Execution verification completed at 19:49 UTC. `make smoke-world` at the
queued immutable `ecf30ef` checkout exited 0: all 16 comparisons identical,
no missing/different chains and no software differences, reference untouched.
Its run is `robots-skills-run-r00c/data/smoke-world/w30-ecf30ef76926`.
`make check-sim` was additionally repeated at final immutable `f77ed68` and
exited 0 (58 passed, 1,206 deselected, 328.32 seconds), covering the later
queue-adapter changes too. Engine, identity, fault and policy sources and the
environment lock are unchanged between the smoke checkout and final code.
Both experiment readouts still await their queues; these checks do not stand
in for live A/B result delivery.

Builder-side inspection clarified the recurring logging diagnosis.
All four short failed records the agents questioned (`e000608`, `e001059`,
`e000613`, `e001666`) have a false pre-hook success verdict and the same visible
failure label, with reason “LIBERO only”. Each ended 60 steps after the coarse
task check first held. There was no label flip in these records. The nominal
fleet criterion also requires no robot-object contact and an object at rest;
the verdict does not identify which of contact or motion prevented completion.
The final scene image can look successful without proving those requirements.
The public dictionary does not explain this distinction, while episode-cards
explicitly suggests a recording problem when the last frame looks done. The
next general procedure correction should explain nominal completion and require
release/settling evidence before asserting label corruption or excluding a
failure. These hidden verdict observations and episode ids belong only in this
builder log, never in the skills. No simulation or model call was used for this
inspection, and the measured procedures remain unchanged.

## Round 7 — old D2 and D3 logs-only comparison while D1 experiments wait

The orchestrator's 20:23 UTC comment clears the stored
`w42/export` and `w43/export` for development on D1's terms: repaired scans
are clean, but the identifier weakness still prevents an evidence claim.
It explicitly directs two logs-only repeats per world without waiting for D1's
experiments. These four analyses started around 20:26 UTC from immutable
`f77ed68`, with separate fresh memories and outputs under
`skills/20261009-r07/{w42,w43}-logs-{1,2}` in the probes cache. Procedures,
model and effort are unchanged; the initial state is awaiting results.

These older worlds are temporary development material, not tuning targets.
Their nominal interaction is actually additive; comparisons will retain that
ground-truth limitation and distinguish complete from incomplete chance
valuation. The replacement worlds are not opened. Both saved D1 experiment
repeats remain queued, with no engine run yet at this round's start.

All four analyses completed with both submissions and no runtime failure.
The unchanged skill library at `f77ed68` gives:

| World / logs repeat | Detectable-cause recall | Decoy rejection | Gain Spearman (valued actions) | Achievable share at k = 3 |
|---|---:|---:|---:|---:|
| D2 / 1 | 4/7 = 0.5714 | 1.00 | 0.0634 (8) | 0.3558 |
| D2 / 2 | 4/7 = 0.5714 | 1.00 | 0.3822 (12) | 0.3558 |
| D3 / 1 | 6/7 = 0.8571 | 1.00 | 0.8616 (10) | 0.7385 |
| D3 / 2 | 6/7 = 0.8571 | 1.00 | 0.7658 (12) | 0.7385 |

D2's two top-three sets each rescue 43 and break six of 336 scoring attempts:
net 37, or 11.0119 fleet points, against a 30.9524-point ceiling. D3's each
rescue 48 and break none: 14.2857 points against a 19.3452-point ceiling.
The recall denominator is the scorer's seven detectable cause units, retaining
the old worlds' acknowledged grouping limitation. D2 fails three of the four
session marks; D3 meets all four, without beating the supplied classical baseline.
First repeats remain primary; repeats are not pooled into a best-run result.

The frozen classical baseline is `classical-frozen-1` (`9f6ca2f`, measured
behavior `b022bbf`), not the previous `83387da` rescore. All nine recorded
input hashes for each world match its current export (18/18). Its unchanged
answer files were adapted without inventing model sessions and rescored with
the current evaluator into `20261009-r07/frozen1-current-scorer-285328f`.
It finds 4/7 causes on D2, with rho 0.3685 and k3 share 0.3558; on D3 it finds
6/7, with rho 0.6261 and k3 share 1.00. These are single frozen analyses, not
resampling distributions. The orchestrator's reported resampling range has
not been traced to a sample receipt here.

Agent comparisons are in `20261009-r07/scored-with-frozen1-fa9c14c`.
D2's agent margin over this baseline is zero points. D3's is -5.0595 points;
its chance comparison remains unavailable (199/200 draws valued at k3).
D2 k3 chance is complete (200/200, mean 0.4241 points, 95th percentile
2.4702 points). Other budgets retain their own completeness checks. These
rescores launched no model or replay and changed no answers.

Model wall times for D2 repeats 1/2 were 739.086/729.047 seconds, with
81/87 tool calls; D3 took 857.999/702.866 seconds, with 83/73 calls.
D3 repeat 1 had four SQL parser/binder errors and recovered; the other three
had no tool errors. All four read the seven analysis skills. No identifier
interpretation or seed-derivation attempt was found in the request scan.
Interval coverage was 4/8 and 9/12 on D2, 9/10 and 9/12 on D3, so these
coverage and correlation numbers have different action denominators.

Reading the D2 traces with an independent reviewer shows several distinct
failures. Both agents assigned a synchronized site decline to the sole logged
sampling edit with confidence 0.8 and expected gain 14.05/14.75 points;
its replay gain is -0.5952. An empty `same_day_overlapping` field was treated
as causal isolation even though it covers only recorded changes. The export
does not identify the competing physical mechanism from that temporal contrast
alone. Matched site/task failures were also absorbed into separate component
or residual-policy explanations. This is a general procedure gap.

Both D2 agents noticed apparent outcome corruption and assigned zero physical
repair gain, but submitted a bare robot target and `changed`; the scorer
requires a component and a different direction, and reports this as vague.
This is an ontology/reporting gap, not simple failure to notice
the discrepancy. D1's false recording suspicion remains a separate completion-
semantics problem. The new procedure must preserve that distinction.

The helper's bad intervals appeared directly in both D2 traces. Their custom
repairs still diluted task-specific uncertainty with pooled comparator counts.
Assuming half restoration also halved upper bounds without evidence that full
restoration was impossible. Drift identification was stronger than gain
prediction: a linear projection exceeded a robot's full fleet exposure, then
was capped and promoted to an almost complete-recovery point estimate.
The interval repair addresses sampling intervals; attribution, repair efficacy and
drift-scenario uncertainty remain separate concerns.

D3 missed the shared object-condition cause and named both members of a
confounded settings pair. Its top-three deficit is not all due to the missed
condition: the frozen baseline selects complementary site and robot fixes,
while the agents select larger singleton estimates. The launch prompt demands
descending singleton gain but `rank-actions` explicitly orders for additional
set benefit. Both D2 agents likewise retained overlapping repairs in their
top six while recognizing that overlap. This is an instruction
conflict; it is not evidence that the agents failed to understand redundancy.

The scoring builder's `b78ad9e` Makefile ordering fix was merged as `fa9c14c`,
retaining both `analyse` and `score` with `setup` as the default. Dry runs of
all three targets pass. `make check` at that merge exited 0: 1,200 passed,
two skipped, 62 deselected, 533.03 seconds. Measured code and skill contents
were unchanged during all four analyses.

## Round 8 — general procedure corrections, before new measurements

The next version separates singleton gain estimates from set ordering in the
launch prompt, requires attention to unlogged and shared
site/task explanations, keeps unresolved alternatives out of
asserted causes, and explains nominal release/settling before
calling a recording error. None of these changes names a world,
robot, site, planted target or world-specific value. The public recording-target
ontology remains a separate unresolved issue; no hidden target
has been taught as a shortcut. Drift scenario calibration is filed separately.

Independent review found that the first recording paragraph could prohibit
even corroborated recording diagnoses because exact contacts/velocities are
not exported. It now retains ambiguity only when genuine incompletion remains
comparably plausible, allows supported diagnoses with calibrated uncertainty,
and asks for separate justification before excluding any episode's label.
No identifier/seed or hidden-result hint was found in the revised procedures.

The 35 helper/dry-run/development-command checks passed, as did 80 contract and
library checks. The latter initially found four obsolete assertions that every
procedure's revision must remain `1`; the manifest still requires a version,
while individual procedures can now advance without changing the library's
roster. Full merged verification and all after-change scores are pending.
The general sampling-interval correction is being implemented separately from
synthetic data in an isolated checkout. Existing D1 experiment continuations
keep their original immutable procedures and remain queued; their eventual
results must not be presented as measurements of this new round.

The isolated implementation delivered `15da1fb`: Wilson MOVER
intervals on disjoint observation cells, with coefficients for reused controls
combined before uncertainty is calculated and pre-period comparator uncertainty
included. Its 32 new synthetic tests plus 26 existing synthetic helper checks
pass, with focused lint, format and types also passing. Independent review ran
the 32 tests and 64 additional old/new synthetic comparisons; existing point
estimates and fallback behavior were preserved, including missing task labels.
These checks establish arithmetic properties, not empirical coverage on
clustered fleet observations. The skill states the conditional interpretation.

`f2f2c34` and `19d6ef8` remove the unsupported fixed-half repair example and
separate uncertainty in attribution/repair efficacy from conditional sampling
uncertainty: the point uses a supported central fraction, while bounds cover
the plausible scenarios, including signed gains. A separate review found no
new issue or restricted information. The public recording ontology and bounded
drift forecast remain future work; neither was silently changed.

The combined revision is frozen at `19d6ef8`. The Codex-only `agents-gate`
exited 0; full `make check` is running. Four new logs-only analyses use fresh
memory under `skills/20261009-r08`: two D1 repeats and one transfer check each
on the old D2 and D3, all at gpt-6-astra/xhigh/fast. Before-change scores are above;
after-change scores are pending. These are measurements of a new procedure
version, not additional unchanged repeats of Round 7.

Full `make check` for this revision exited 0: 1,232 passed, two skipped,
62 deselected, 540.50 seconds. The interval arithmetic and procedure changes
add 32 tests; no simulator source changed in this round.

The D1 baseline comparison is now complete for the frozen classical method.
Its saved `b022bbf` answers, rescored without a new model or replay, also find
6/6 causes and recover all 67 net successes out of 336 at k3 (19.9405 points),
with rho 0.9567. Both original agent repeats therefore tie this baseline on
k3: margin zero, ratio 1.00. The earlier positive margin concerned only the
four baselines supplied then and must not be presented as beating all baselines.
Outputs are under `20261009-r08/scored-with-frozen1-3b6d85d`, including the
unchanged original agent answers and their full chance valuation.

This baseline ran against the original D1 export. Eight of its nine recorded
input hashes match the cleared rebuild. Only `events.parquet` differs, in its
`detail` column; all 5,046 episode-id/event-kind pairs are identical. The frozen
source consumes events only in `event_profiles` and `event_signatures`, both
using those two unchanged fields. This supports feature equivalence, not a
claim that the baseline was freshly executed on the rebuilt export. The
identifier weakness and development-only restriction remain.

All four Round 8 analyses completed and were scored. The measured revision is
`19d6ef8`; no follow-up candidate edits entered these sessions.

| World / logs repeat | Detectable-cause recall | Decoy rejection | Gain Spearman (valued actions) | Achievable share at k = 3 |
|---|---:|---:|---:|---:|
| D1 / 1 | 5/6 = 0.8333 | 1.00 | 0.8462 (9) | 1.00 |
| D1 / 2 | 6/6 = 1.00 | 1.00 | 0.8889 (10) | 1.00 |
| D2 / 1 | 5/7 = 0.7143 | 1.00 | 0.7214 (10) | 0.8846 |
| D3 / 1 | 6/7 = 0.8571 | 1.00 | 0.8651 (10) | 1.00 |

Each D1 top-three set rescues 67 and breaks none of 336 scoring attempts
(19.9405 points). D2 rescues 92 and breaks none (27.3810 points against a
30.9524-point ceiling); D3 rescues 67 and breaks two (19.3452 points, its
ceiling). These four runs, their automatic scores, and the separate unchanged-
answer rescores with frozen classical comparisons all exited 0. The latter
are under `20261009-r08/scored-with-frozen1-3b6d85d`. D1 and D3 tie the
frozen classical k3 gain; D2 exceeds it by 16.3690 points, ratio 2.4865.
D1/D2 k3 chance is fully valued; D3 remains unavailable at 199/200 draws.
All identifiers and old-world interaction caveats above remain in force.

Against the earlier procedure, D1 retains all four marks in both repeats;
its first repeat leaves the shared site mechanism unlocalized and loses one
hit. Both stop asserting the earlier false recording diagnosis and exclude
no labels. Their correlation denominators differ from the earlier answers,
so the lower correlations are not a fixed-action ablation. Mean absolute
errors are 0.7348/0.5373 points; coverage is 7/9 and 8/10, with every miss
caused by the inward-rounded full-exposure cap described in Round 9 below.

D2 improves from 4/7 recall and 0.3558 k3 share in both earlier repeats to
5/7 and 0.8846; it still fails recall. Independent trace review finds the
unlogged-placement procedure used in calls 68 and 70–73: matched pre/post
views support placement rather than the contemporaneous sampling edit.
Shared task/site evidence is investigated in calls 67, 74–77, 84 and 88;
the batch action reaches rank six but the diagnosis remains unlocalized.
The recording discrepancy also remains a bare target, before Round 9's
public ontology repair. Precision 1.00 excludes these two vague entries;
it does not mean all seven submitted causes were localized correctly.

D2 interval calibration remains poor: 5/10 overall, and on eight actions
common to all three runs coverage falls from 4/8 and 5/8 to 3/8, although
their mean absolute error improves from 2.9955/3.0455 to 1.5540 points.
The clipped drift becomes a nearly complete-recovery estimate; conditional
peer comparisons and gripper discounts miss effects; the sampling alternative's
zero lower bound excludes its negative gain. These are comparator, forecast,
efficacy and harm-model issues beyond conditional sampling arithmetic.
K6 share improves from 0.4420 to 1.00, but four-hour share falls from 0.8714
to 0.7571 because the expensive placement fix consumes that budget first.

D3's set ordering selects complementary site timing and robot-mode repairs,
raising k3 share from 0.7385 to 1.00. It no longer asserts both confounded
settings as causes, but still leaves the shared object-condition mechanism
unlocalized and now asserts a false fleet-level recording diagnosis. The
completion-semantics correction therefore helped D1 without eliminating this
error generally. D3 mean absolute error is 0.6029 points and coverage 9/10;
the missed interval belongs to a harmless setting whose revert breaks three
scoring attempts. No claimed probability or causal attribution is revised
after seeing these truths. One harder-world repeat cannot isolate which
procedure changes caused the improvements.

Model wall times were 923.080/880.705 seconds on D1, 1,140.377 on D2 and
1,059.891 on D3; successful calls completed with 93/88/98/97 tool calls.
D1 repeat 1 recovered from one SQL error and D3 from two; the other runs
had none. No identifier interpretation or seed-derivation attempt was found
in the tool-request scan or reviewed notes. D1 experiment repeats still wait
for the shared heavy queue and retain the original procedures.

## Round 9 — public target mapping, bounded drift and endpoint reporting

The known-class passage now supplies the documented outcome-detector target:
`controller` with direction `degraded`. This is sourced from the public
fault-library design, not a development truth table (`docs/design/faults.md`,
the class tables and direction rule). It remains inside the separable class-
knowledge section. Unknown or unlocalized recording inconsistencies do not
inherit that mapping or its zero true-success gain; suspicious end frames
still require the episode-card comparisons. Independent review supported this
boundary, and 80 library/contract checks passed. This addresses the
recording-target ontology gap.
The running Round 8 analyses retain their immutable earlier version, so none
of their eventual scores measure this additional guidance.

The drift follow-up now bounds a projected loss to the range implied by a future
success rate in [0, 1], holding the observed comparator fixed. The raw
projection remains a diagnostic, with a clipping flag and usable observation
day count/range; missing, single-day or nonfinite fits remain unavailable.
The observed gain and its sampling intervals are unchanged. The procedure
separates persistence/trend scenarios from an expected repair gain, rejects
clipping as evidence of complete recovery, and requires a workload-weighted
average of bounded daily forecasts before using a trend for the whole scoring
window. The follow-up does not claim a validated forecasting model.

Independent review found no actionable issue and reran all 42 synthetic
interval/projection tests, with additional signed and pre-period checks. The
combined helper/dry-run tests passed before the final overflow guard (70 tests);
the final 42 arithmetic tests include that guard. These corrections have no
development identifiers or measured calibration values. A live comparison
and full merged verification remain pending for this candidate.

Round 8 also exposed inward rounding of a physical exposure cap: both D1
answers use an upper bound of 8.33 where the exact full-exposure gain is
8.333333333333334. The two equivalent actions are the only interval misses
in each D1 answer. A reporting repair keeps exact caps or rounds endpoints outward
in the reporting procedure and the gains block. Point estimates stay unchanged;
signed fractional-exposure tests establish that displayed bounds contain the
unrounded conditional interval. Floating arithmetic at an integer boundary may
consume one extra outward reporting unit. Independent review passed all 43
arithmetic checks; the combined helper, dry-run, library and contract suite
exited 0 with 152 passed. Saved answers and their reported coverage were not
altered. This guidance is part of the next candidate, not the measured Round 8.

Oracle follow-ups through `89a98e6` and their current owner branch were merged
as `9202097`. That branch also brings the orchestrator's updated instructions
and post-freeze scoring decisions. Its bookkeeping file was retained verbatim when
resolving the sole conflict; no new builder bookkeeping was added there.
In particular, the revised design requires all four nonrandom baselines and
exact count-budget chance comparisons, and changes cause grouping and warning
evaluation. The scores above retain their original evaluator and denominator;
their supplied-baseline ratios are historical comparisons, not passes under
the newly clarified full-baseline rule. Updated scorer integration and fresh
rescores are pending; no stored reports will be overwritten.

The next procedure measurement is frozen at `d94c147` in a new detached
checkout. Fresh logs analyses under `20261009-r09` comprise two D2 repeats
and one regression run each on D1 and D3. Full merged `make check` is running;
152 focused checks and the independent procedure/arithmetic reviews passed.
These calls use the same cleared exports, gpt-6-astra/xhigh/fast and fresh
memory. All four later completed with both submissions; all four `make`
commands exited 0. Their immutable procedure is `d94c147`, before the further
endpoint clarification recorded in Round 11. Fresh settled scores at
`20261009-settled-94432fa/scores/r09-*` preserve all original submissions;
the later scorer refresh described below reads the same answers.

| World/repeat | Recall | Strict precision | Decoy rejection | Singleton rho (valued actions) | Three-fix share | Interval coverage | Mean absolute error, points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1, 1 | 6/6 | 6/6 | 1.00 | 0.9012 (10) | 1.0000 | 8/10 | 0.6095 |
| D2, 1 | 6/7 | 6/6 | 1.00 | 0.7135 (11) | 0.8846 | 8/11 | 2.1671 |
| D2, 2 | 6/7 | 6/6 | 1.00 | 0.3650 (12) | 0.8846 | 11/12 | 2.7312 |
| D3, 1 | 6/7 | 6/7 | 1.00 | 0.9019 (10) | 1.0000 | 9/10 | 0.4874 |

D2/D3 diagnosis metrics retain the old cause-unit disagreement and are
provisional. Strict precision includes vague causes; small real faults are
reported separately, not counted as false positives. D1 recovers 67 and
breaks zero of 336 scoring attempts (+19.9405 points); each D2 run recovers
92 and breaks zero (+27.3810); D3 recovers 67 and breaks two (+19.3452).
All replay-bearing actions are valued. D1, D2 repeat 1 and D3 meet the four
session marks numerically; D2 repeat 2 misses rank correlation. D1/D3 tie
the frozen classical three-fix gain; D2 retains its 16.3690-point margin
and 2.4865 ratio over the required baselines. D1's overall comparison still
lacks its first video baseline and is unavailable.

CLI wall times were 868.301/890.087/1085.437/1195.614 seconds in table order;
tool calls were 93/88/101/105, including 2/2/0/0 recovered SQL parser errors.
Each read all seven logs-condition skills. Identifier/seed inference was
not found in the calls: the identifier hashes select samples, including a
custom matched-case sample; they do not interpret identifiers or recover seeds.

The D1 answer names all six causes without the previous vague recording
claim. Its only interval misses are mathematically equal full-exposure caps
separated by floating evaluation order (Round 11). D3 also removes the false
fleet recording diagnosis, retains one vague site mechanism, and improves
gain error and rank correlation relative to Round 8. Its lone interval miss
is again the harmless sampling alternative: lower bound zero versus a signed
gain of -0.8929 points.

The independent read-only review of D2 repeat 1 found actual use of the new
procedures. Tool call 11 delivers the public controller mapping; calls
51–54 and 74 add terminal comparisons; 81 and 86 give the recording mechanism
the documented target and direction. That supplies the additional credited
cause, chiefly a mapping correction with added observational checks, not an
intervention-based causal validation. Calls 65 and 68 use the bounded helper
and separate scoring days; a declared 50/50 mixture of persistence 6.12 and
trend 8.20 yields 7.16 points. Call 77 widens the lower bound to 1.44, covering
the true 4.17. The mixture weight is judgment. Calls 67/77 round outward;
69 preserves a raw cap as the supplied instructions then permitted.

On nine actions shared with Round 8 D2, coverage improves 4/9 to 6/9 but
mean absolute error worsens 1.823 to 2.290 points. Both new D2 repeats choose
the same three-fix set, but point estimates outside that set vary enough to
fail ranking in repeat 2. Both omit the batch repair from the first six:
their six-fix share falls from 1.0000 to 0.8986, four-hour share to 0.7286,
and eight-hour share to 0.8864. The remaining shared site/task cause is
assigned to retraining too readily; its cited healthy replay outcomes are
not available, so the assertion that no repair can help is unestablished.

The traces also expose a general error in the existing class guidance:
it invents partial repair efficacy where the public contract removes a known
mechanism completely. D2 repeat 1 gives camera service 75% efficacy (3.23
predicted points, true zero) and discounts gripper service relative to
replacement although both remove the same known mechanism. Confounded-pair
guidance still forces a nonnegative interval and an equal-attribution point.
These source-supported repairs are recorded separately; their implementation
and review are a new round, not a reinterpretation of these saved answers.

## Round 10 — experiment startup memory refusal and bounded retry

Both original D1 experiment continuations in `20261009-r05` were refused
at 21:15 UTC, before `load_policy` or any episode: the ten-worker estimate
was 25 GiB plus the 8 GiB memory floor, against 29.3/29.1 GiB available.
Both `make` calls exited 2 and their automatic scores report unavailable
diagnosis, ranking and gain. Each engine directory contains only initialization
metadata, empty log/staging directories and its canary; no result reached an
agent. The first analysis calls and all failed continuation outputs remain.

The 21:20 and later still-queued status statements were stale: they inferred
our state from the shared lock rather than rereading our ended records. The
timestamps and empty episode stores establish the actual refusal time above.
No authentication error
occurred, no memory/GPU guard was lowered and no other job was altered.

At 21:34 UTC the same original ended-waiting snapshots were resumed into
new `20261009-r10/experiments-{1,2}` outputs with `--workers 4`, retaining
the original immutable `2a53302` procedures, analysis ids, requests and call
provenance. Four workers require an estimated 12.4 GiB plus the same 8 GiB
floor. Policy batch size remains four. This is a changed execution setting,
not an independent analysis repeat or an unchanged retry; no completed model
session is repeated. The queues still request 720/480 arm attempts and will
go through all normal reply/export checks before any further session.

## Round 11 — settled evaluator and full baseline comparison

The scorer's `ead9dcc` and oracle distribution/replay-request repairs merged
as `94432fa`. The preceding full `make check` at `d94c147` exited 0:
1,275 passed, two skipped and 62 simulator/model cases deselected. Full merged
verification is running. No analysis procedure changed in this integration.

Fresh oracle reports at
`~/.cache/robots-probes/skills/20261009-settled-94432fa/oracles/{w41,w42,w43}`
read exactly the prior reports' accepted run sources, copy the original plan
records, and pass `--no-truth-table`. There are no new simulations or writes
to stored sources. Before/after source report hashes match; world hash, fault
effects, interactions, achievable gains, all-fault results, causes and daily
effects compare equal. D1 report exits 0; D2/D3 exit 1 for their incomplete
seeded draws, with no table problems. Reproduction scripts, command exits and
the field comparison are retained in the new result directory.

The exact count distribution now preserves unvalued combinations rather than
claiming every combination has a lookup. Of 22,369,536 three-entry sets,
32,766/17,775/22,731 lack values on D1/D2/D3. Conservative 95th-percentile
bounds in fleet-success points are [3.273,3.274], [2.380,2.381] and
[3.273,3.274]; the upper endpoints decide the comparison. The seeded 200-draw
results remain alongside these bounds and do not supply the count-budget floor.

Fourteen ended records were rescored without model or simulation calls:
ten completed logs analyses, both early transport failures and the two failed
experiment continuations. The same new scorer will score the four active
Round 9 runs and the two Round 10 continuations when they end. Submission
hashes are retained and checked unchanged during scoring. Final-wording
simple baselines `7eea162` and frozen classical `b022bbf` use authoritative
per-budget sets; first model repeats are primary. Original baseline inputs
and the classical feature-equivalence caveat remain as recorded above.

The completed analyses' numeric recall, singleton correlation and recovery
are unchanged. The new strict precision counts vague causes as false positives:

| Procedure/run | Recall | Strict precision | Singleton rho | Three-fix share |
| --- | ---: | ---: | ---: | ---: |
| Original D1 repeat 1 | 6/6 | 6/7 | 0.9461 | 1.0000 |
| Original D1 repeat 2 | 6/6 | 6/8 | 0.9538 | 1.0000 |
| Round 7 D2 repeat 1 | 4/7 | 4/6 | 0.0634 | 0.3558 |
| Round 7 D2 repeat 2 | 4/7 | 4/6 | 0.3822 | 0.3558 |
| Round 7 D3 repeat 1 | 6/7 | 6/7 | 0.8616 | 0.7385 |
| Round 7 D3 repeat 2 | 6/7 | 6/7 | 0.7658 | 0.7385 |
| Round 8 D1 repeat 1 | 5/6 | 5/6 | 0.8462 | 1.0000 |
| Round 8 D1 repeat 2 | 6/6 | 6/6 | 0.8889 | 1.0000 |
| Round 8 D2 repeat 1 | 5/7 | 5/7 | 0.7214 | 0.8846 |
| Round 8 D3 repeat 1 | 6/7 | 6/8 | 0.8651 | 1.0000 |

All ten reject the decoy. Every listed correlation has outcomes for all its
replay-bearing actions; no-replay verbs are excluded. D2/D3 recall and both
precision variants are explicitly provisional: the old stored cause units
disagree with the per-fault detectable flag for `f01`. The evaluator does not
rewrite those world records. The four no-answer execution records now have
zero recall and recovery, with undefined correlation; this supersedes the
earlier evaluator's unavailable treatment, without erasing the failure reason.
The two refused experiment continuations are failed branches of the same
saved analyses, not independent repeats or evidence about A/B effectiveness.

D1's first final-wording video baseline timed out, so the required-baseline
margin/ratio is unavailable. Its successful second repeat is shown, never
substituted. The agents tie the completed classical baseline. D2 and D3 have
all four required three-fix baseline values. Round 8 D2 beats their maximum
by 16.3690 points, ratio 2.4865; D3 ties it, ratio 1.0000. Both are one revised
repeat on old development worlds, not held-out or pooled evidence. Missing
cost-budget baseline replays and warning/retraining evidence stay missing.

The interval, instruction-conflict and site/task items were closed for their scoped
procedure repairs and checked live/synthetic behavior, with remaining causal
and calibration limits named. A separate ordering
follow-up is recorded: compare hour-budget cuts among permutations that leave both count
sets unchanged, without changing singleton gains or post-hoc reordering a
stored answer. This was not added to the active Round 9 procedure.

Full merged `make check` at `94432fa` exited 0: 1,301 passed, two skipped,
62 deselected, 462.83 seconds. A read-only merge-specific boundary review
found the 24-module toolserver closure unchanged and free of oracle/scorer
imports, including nested imports. Runtime checks, public-result validation
and mounted skills are unchanged. This is an additional source
check, not the orchestrator's later delivery review.

The Round 8 pool is retained in `round8-pool` beside the new scores. Its
standalone report loses the provisional diagnosis flags when pooling D2/D3;
this was reported to the scoring owner and an adjacent
`PROVISIONAL-NOTICE.md` preserves the caveat without altering the generated
report. No pooled recall pass is claimed as settled evidence. The scorer
watcher was moved, after its first fourteen reports and with source unchanged,
to detached `robots-skills-score-94432fa`; stopping the waiting watcher exited
143, and the new watcher skips those completed outputs. No model or simulator
was interrupted.

The first completed Round 9 D1 answer exposed a final reporting edge case:
it retains an unrounded cap of 8.333333333333332 where the mathematically
equal oracle gain is 8.333333333333334. Both equivalent actions are marked
uncovered; the difference is 1.776e-15 points. The guidance had offered full
precision as an alternative to outward rounding, which does not protect
against different floating evaluation orders. The follow-up now requires
physical clipping first and outward rounding last, including custom bounds.
Helper mathematics and point estimates are unchanged. The first focused
command named a nonexistent test file and exited 4 without running tests;
the corrected helper/library/contract/dry-run command exited 0 with 152 passed.
This prose clarification is not in the active Round 9 runs, and no stored
answer or scoring tolerance was changed.

## Round 12 — documented repair semantics and honest attribution

The Round 9 trace review found a contradiction between the sensitivity skill
and the public design. The skill described gripper
service as a partial repair and implied that camera service could recover a
fraction of an image-degradation loss. The public fixing-verb tables in
`docs/design/faults.md` sections 1 and 8.1 and action resolution in
`docs/design/contracts.md` section 8 specify removal of the entire known
mechanism. Uncertainty about which mechanism is present remains separate.

Revision 5 of `sensitivity-from-logs` replaces those claims with a separable
general mechanism-to-verb table. Equal removals receive equal singleton
effects and uncertainty, with different costs; unsupported verbs receive no
invented fractional efficacy. Co-occurring mechanisms and unknown classes
remain explicit. The general procedure also removes a forced equal split
between confounded alternatives and a forced zero lower bound. It calls for
declared, evidence-supported attribution scenarios and signed harm bounds.
No world identifiers, fault schedules or development measurements were added.

Before-change evidence is the immutable Round 9 table above. In particular,
old D2 repeats 1/2 have singleton rho 0.7135/0.3650 despite identical 0.8846
three-fix recovery. Repeat 2 splits a site's confounded loss equally between
placement and a sampling change; their oracle effects differ in sign and
size. Both repeats discount gripper service and give camera service an
unsupported benefit. These observations motivate general contract fidelity,
not a per-world repair list. After-change scores are NOT MEASURED.

The focused helper/library/contract/dry-run command exited 0 with 152 passed.
The full merged `make check` exited 0 with 1,304 passed, two skipped and 62
deselected in 514.22 seconds. That check started at `6e9fd8d`; the prose-only
revision was made while it ran, before the skill tests. A clean committed
check and Codex gate follow. Under the updated project instructions, the orchestrator
arranges the independent leak review; this builder starts no reviewer model.
The candidate remains unmeasured pending that review and a fresh analysis.

At 22:06 UTC both Round 10 experiment continuations still wait for their
first simulation. Their initial model sessions are preserved; no extra call
or repeated failed simulation has been made. The six-hour deadline remains
23:58 UTC, so experiment completion remains the main delivery uncertainty.

A consistency pass also removes a universal stronger-repair implication from
`rank-actions` and `scope`: maintenance remains observational evidence whose
scope, timing and concurrent changes matter. Those skills point to the same
separable mapping instead of inventing a second one. Package READMEs now
reflect the completed logs analyses and the still-pending A/B delivery.
The Codex gate on `3b5966c` exited 0. Its initial invocation inside the outer
sandbox failed on read-only `/var/tmp`; the same command outside that sandbox
passed all source, layout, lifetime, wire, isolation and authentication checks.
This establishes a sandbox restriction, not a login failure, and makes no
model call. The consistency follow-up still needs its own skill checks/review.

Verification follow-up at 22:15 UTC: `f8c203d` passed the focused 152 checks
in 19.00 seconds and the Codex gate, both exit 0. The full `make check` also
exited 0 with 1,304 passed, two skipped and 62 deselected in 469.37 seconds;
it started at `3b5966c` and finished after the prose-only `f8c203d` follow-up.
Both current prompt variants and the complete mounted bundle pass the source
audit on the exact cleared D1 export, with no failures. The host-only receipt
is `~/.cache/robots-probes/skills/20261009-preflight-f8c203d/audit.json`.
Independent review is requested for `d3223d4..f8c203d` and
remains pending. There is still no after-change model score.

At 22:20 UTC, measurement started from detached `robots-skills-run-r12` at
`f8c203d`: one D1 logs regression and two old-D2 logs repeats, all fresh
gpt-6-astra xhigh/fast sessions. `make setup` exited 0. Independent review
remains requested; the assignment requires it after each larger round, not
as approval before an authorized development measurement. This supersedes
the builder's earlier self-imposed wait. Results go to new
`~/.cache/robots-probes/skills/20261009-r12/` directories. The immutable
`6e9fd8d` evaluator watches for completed records and writes new scores to
`20261009-r12-settled-6e9fd8d`, with the same oracle reports and baseline
protocol as the Round 9 comparison. No running checkout is edited.

An additional host-side identifier screen covers all 18 original tool logs
through Round 9, including the two failed starts and the two saved first A/B
sessions: 1,479 tool calls. No arguments match the seed, identifier or known
derivation-route terms in either screen. File hashes, patterns and zero-hit
lists are under `20261009-identifier-review-f8c203d`. This supports the earlier
manual trace reviews; it is not proof against encoded attempts and does not
remove D1's known identifier weakness. The Round 10 queues still have zero
engine records; no experiment result has been delivered.

Further oracle reading explains old D2's 12-attempt top-three shortfall.
Placement alone rescues 53 attempts; placement plus
the control repair rescues 79, so control adds 26 after placement, versus
15 alone. The selected camera repair rescues 14 independently. With the
software repair's 25, the selected set rescues 92 instead of the achievable
104: 12/336 = 3.571429 fleet points missed. This is distinct from the old
world's planned interaction pair being additive. Causes that each prevent
the same success can make their repairs complementary; shared scope does
not imply only redundancy. The current rank procedure explains subtracting
overlap but not this case. A general joint-response procedure and a small
synthetic counterexample are proposed separately, not added to the running
Round 12 checkout. Its remaining weaker/stronger-verb wording is also noted.

### Round 12 results

All three analysis commands and the immutable settled-score watcher exited 0.
The measured procedure is `f8c203d`; `99f3821` later imports only the scoring
owner's README and mutation-evidence record, with no executable change.

| Logs run | Detectable recall | Strict precision | Decoy rejected | Singleton rho (valued actions) | Three-fix share | Interval coverage | Mean absolute error, fleet points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 repeat 1 | 6/6 | 6/6 | 1.00 | 0.9256 (11) | 1.0000 | 9/11 | 0.7029 |
| Old D2 repeat 1 | 6/7 | 6/7 | 1.00 | 0.7278 (11) | 0.8846 | 8/11 | 2.5910 |
| Old D2 repeat 2 | 6/7 | 6/7 | 1.00 | 0.7820 (12) | 1.0000 | 12/12 | 1.3403 |

Every replay-bearing action is valued; routing/training entries have no
immediate robot-success replay and are excluded. D2 diagnosis remains
provisional for the stored cause-unit disagreement. The three-fix sets rescue
67, 92 and 104 attempts respectively, breaking none, out of 336 scoring
attempts per world. D1 recovers the full achievable gain at all four budgets.
D2 repeat 1 has six-fix/four-hour/eight-hour shares 0.8986/0.8714/0.9394;
repeat 2 has 0.9565/0.7571/1.0000. The four-hour selection still depends
strongly on ordering: repeat 2 spends its whole budget on placement first.

Before/after on D2: Round 9 rho 0.7135/0.3650 becomes 0.7278/0.7820, and
top-three share 0.8846/0.8846 becomes 0.8846/1.0000. Both new repeats clear
the requested four numerical marks, but neither identifies the shared batch
cause precisely; both now include a vague site cause, so strict precision
falls from 6/6 to 6/7. Absolute gain error worsens in repeat 1 and improves
in repeat 2. This is not a uniformly better or independently replicated
causal attribution method. It remains old-world development, with changed
action lists and several general wording corrections in the same round.

The trace confirms the intended repair-semantics behavior in both D2 runs:
cleaning and replacement of the supported image mechanism get equal effects;
gripper service and replacement get equal effects; unsupported camera service
is omitted. The gain attributed to the sampling change differs sharply.
Repeat 1 declares a 55/45 placement/sampling split as an assumption, predicts
6.05 points for sampling reversion, and retains a signed harm envelope. Repeat
2 gives the visible placement mechanism central weight and sampling reversion
zero, retaining the same wide signed alternative envelope. Their true sampling
gain is -0.5952 points. Removing a forced equal split permits better reasoning;
it does not identify a unique attribution from confounded observations.

Repeat 2's query 86, `coherent_site_counterfactuals`, explains its better set.
It builds task-matched before/after comparison regimes, keeping a component
problem present or removed in each. The control repair is predicted to add
8.03 points after placement restoration, versus 4.20 alone, so it precedes
an independent 7.83-point software repair without falsifying its singleton
field. Conversely, cleaning's additional benefit falls after placement.
This is a successful explicit treatment of both complementarity and overlap,
supporting the general joint-response follow-up; comparator transport and
mechanism assumptions remain observational, not measured interventions.

D1's two interval misses are no longer rounding: query 77 clips an otherwise
wider interval to an empirical historical failure fraction, yielding high 7.74
for equivalent delay repairs whose future gain is 8.3333. This is a
general error: observed failure counts are not a deterministic bound on
another window's recoveries. Point estimates and selected sets are unchanged.

Model times were 871.467/978.681/1160.209 seconds; tool calls 92/93/105,
with 1/1/2 recoverable custom-SQL errors. All seven logs skills were read.
The errors were bare aliases and a mixed aggregate, corrected without builder
intervention. Trace receipts and source hashes are beside the settled scores.
One identifier-screen hit in repeat 2 is the ordinary phrase “not unique cause
identifiers,” about telemetry signatures; manual inspection finds no identifier
interpretation or seed derivation. No replacement world was accessed.

At 22:43 UTC both original D1 A/B continuations still await their first
simulation. Their current four-worker retries have waited about 67 minutes;
there is still no delivered experiment result or completed experiment-condition
score. Independent review of the procedure changes remains requested from
the orchestrating session, and all D1 results remain development-only.

## Round 13: conditional repair response and future-window bounds

Round 12's second D2 repeat successfully compared four observational regimes
before choosing its complementary repairs; the first repeat did not. The
general `rank-actions/pair_scenarios` block now computes the two singleton
gains, their joint gain and each conditional gain from four stated response
rates, weighted by disjoint target-workload strata. It preserves signed
effects and separates alternative scenarios. Invalid rates, missing values,
nonfinite values, duplicate strata and impossible total exposure return no
gains for their group. It cannot validate counterfactual assumptions or infer
higher-order interactions from pairs. The procedure distinguishes redundancy
from complementarity and tells the agent to keep singleton estimates honest
when conditional gains change the set order.

`sensitivity-from-logs` now distinguishes a future workload bound from the
realized failure count in the observed window. Historical
failure rates must not clip a future uncertainty envelope as if they were
known future counts. Remaining weaker-verb wording is replaced with measured
partial-response reasoning. The experiment procedure describes an optional
both-actions arm for plausible masking mechanisms within the existing budget;
it does not require tests of equivalent documented repairs just because their
costs differ. These are general procedures with no development identifiers or
calibration values. No simulator, runtime or scorer source changes this round.

Before measurement, Round 12 D1 recall/rho/k3 share are 6/6, 0.9256, 1;
old D2 repeats are 6/7 each, 0.7278/0.7820 and 0.8846/1. Their interval
coverage is 9/11, 8/11 and 12/12. Round 13's after scores are not yet
measured. The initial focused check exposed four test-fixture type-coercion
errors and an old assumption that every SQL block has scalar parameters;
both were corrected without weakening the scenario checks. The Codex runtime
gate exited 0; focused and full checks are recorded below when complete.
Independent skill leak review is requested from the orchestrating session
after this larger round; no reviewer session is launched by this builder.

### Round 13 results

All three `make analyse` commands and the original settled-score watcher exited
0. The newest immutable `ee176ff` evaluator also scores all three unchanged
submissions with verified run labels. Each scored world has 336 future attempts.

| Logs run | Recall | Strict precision | Decoy rejected | Singleton rho (valued actions) | Three-fix share | Interval coverage | Mean absolute error, fleet points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 repeat 1 | 5/6 | 5/6 | 1.00 | 0.8977 (11) | 1.0000 | 11/11 | 0.6602 |
| Old D2 repeat 1 | 6/7 | 6/7 | 1.00 | 0.8182 (12) | 1.0000 | 12/12 | 1.6319 |
| Old D2 repeat 2 | 6/7 | 6/7 | 1.00 | 0.6125 (11) | 0.8846 | 9/11 | 2.0253 |

D1 retains the four requested marks and now covers every valued gain. Its
trace reports the delay interval's exposure-clipped, outward-rounded high 8.34,
instead of the previous historical-failure cap 7.74. The site cause is now
left unlocalized, losing one recall hit and adding one vague claim; the batch
repair remains an explicitly uncertain ranked hypothesis. All four budget
shares are 1, with top-three 67 rescued/0 broken and top-six 125/1.

Both D2 traces use `pair_scenarios`, but repeat 2 fails the ranking mark. Their
top-three sets rescue 104/0 and 92/0 attempts. Six-fix shares are 1 and 0.9493;
four-hour shares are 0.8714 each; eight-hour shares are 0.9394 each. D2 diagnosis
remains provisional for the stored cause-unit disagreement. Both retain the
vague shared-site claim instead of identifying its condition or the whole
counted several-fault unit. This remains a real uncertainty in the available
evidence, not a reason to teach the planted condition to the skill.

Repeat 1 uses the explicit complementary control/placement pair and predicts
zero for the unresolved sampling alternative, with a full signed identification
envelope. Repeat 2 assumes 75/25 placement/sampling mechanism weights. That
reduces control's expected contribution after placement to 7.07, just below a
7.10 drift-repair estimate, changing the selected three. It predicts 3.47 for
sampling reversal, whose actual gain is -0.5952; the signed interval covers it.
Both still overpredict some camera gains. Repeat 2's two interval misses are
equivalent image repairs with low 2.83 versus true 2.381. Repeat 1 retains a
broader across-regime envelope that covers them. The helper makes conditional
arithmetic explicit but does not identify causal scenario weights or validate
comparator transport. The results are mixed, not a uniform improvement over
Round 12 (D2 rho 0.7278/0.7820, three-fix shares 0.8846/1).

Both D2 repeats independently improved the internal cost order before Round
14's clarification: cheap actions precede the four-hour action within the same
count set. Thus any subsequent improvement cannot be attributed to that wording
alone. Model times were 659.594/995.135/864.524 seconds; calls 87/86/91, errors
0/0/3. All seven skills were read and each run used the new helper. The three
errors are recoverable bare SQL aliases. The identifier screen's one hit is
the label “late recovery” near a normal `episode_id` lookup in D1 query 62;
manual inspection finds no identifier interpretation or seed derivation.
Receipts and tool-log hashes are in `20261009-r13-trace-review`.

Round 13 verification: all 173 focused checks passed in 19.43 seconds.
`make check` on `1ce4a8b` exited 0: 1,325 passed, two skipped, 62 deselected,
506.32 seconds. The three fresh analyses run from immutable
`~/.cache/robots-skills-run-r13`, with new `20261009-r13` outputs; the original
A/B continuations still have no engine records at 23:00 UTC.

### New evaluator provenance checks

Merge `f727956` imports the scoring owner's `84b532a` and `ee176ff`, binding
condition/repeat and baseline labels to producer records and requiring an
explicit roster for pooled comparisons. Immutable evaluator `ee176ff` rescored
17 completed logs answers and three ended no-answer records to new
`20261009-final-ee176ff` directories. Successful-run recall, ranking and
recovery do not change. The three no-answer records now receive zero decoy
credit as well as zero recall/recovery. D1's missing-primary-video comparison
remains unavailable. The original first transport failure (`r00`) is refused
because its old run record lacks a repeat field. That original record and its
earlier scores are preserved; no repeat was inserted retrospectively. The new
receipt lists this refusal explicitly, with the 20 completed reports and input
hashes. No model or simulation was repeated for this evaluator update.

## Round 14: preserve count sets while comparing cost cuts

The Round 12 D2 second agent explicitly computed a cheaper four-hour order
while retaining both count sets, but rejected it because the skill required
marginal-benefit order. `rank-actions` now permits permutations within the
first-three block and within ranks four through six, comparing the actual
four-hour/eight-hour sets under the same response scenarios. Singleton fields
remain unchanged; no saved ranking is edited. A hand-valued synthetic case
demonstrates a four-hour improvement from 12 to 21 points at 3.95 hours while
both count sets and the eight-hour set remain identical.

`write-diagnosis` also distinguishes a localized symptom from a causal target.
If the component/key/condition is unresolved, preserve the
finding in mode summaries instead of asserting a bare scope or guessing a key.
An unfamiliar mechanism with a supported target remains reportable as unknown;
the procedure does not force it into a known class. This responds to the D2
trace's explicit uncertainty about its site/task anomaly, not to a hidden
instruction to name the planted condition.

Before measurement, Round 12 D2 strict precision is 6/7 in both repeats and
four-hour recovery shares are 0.8714/0.7571. The intervening Round 13 results
are still pending; Round 14 has no live after score yet. The changes are general
prose plus a synthetic budget test, with no runtime or simulator change.
Independent review remains the orchestrating session's responsibility.

Before the Round 14 measurement, a final general correction separates policy
configuration keys' dose-response knowledge. The old paragraph
called all settings mostly mild, despite the public library's severe loss with
most image views removed, near-flat/small tested sampling reductions, modest
chunk execution alone with a strong documented delay combination, and no
fault-library precision calibration. The replacement states these distinct
qualitative priors and their transport limits; it contains no calibration
numbers or development-world values and remains in the separable knowledge
section. A large observed association still needs current mechanism evidence;
the prior neither caps the effect nor automatically clears a setting. This
correction joins the cost-order/diagnosis candidate before any Round 14 model
call, so their effects will not be claimed as independently isolated.

### Final evaluator integration and verification

The scoring owner's reviewed `2d3f059`/`6967ced` (README `84bfa91`) now binds
baseline producers to the scored world, their recorded output directories and
available input hashes. A direct read-only probe accepts the correct-world
classical producer and refuses another world's producer with an explicit
`ValueError`; receipt `20261009-baseline-world-binding-repair`.
The producer records still lack output-hash attestations, a separate freeze
follow-up. Pooling now takes a file roster, not typed world IDs.
The unchanged Round 9 answers pool under the public development roster hash
`83f4fb42d493c0188d24b9b96b556f55bcd83f9088c263395bc193c5940d3d70`, retaining
the D2/D3 cause-unit caveats. This is not a held-out roster.

Immutable evaluator `84bfa91` creates new `20261009-final-84bfa91` reports.
The first 23 reports (20 completed logs answers and three ended no-answer
records) retain every diagnosis, ranking, interval and recovery value from
`ee176ff`. The old `r00` provenance refusal remains. The baseline owner's
successful primary video rerun, `6b3a75a` with manifest wording `baselines-v4`,
now fills D1's missing method; repeat 2 is still not substituted. D1's
three-fix baseline margin is therefore available at zero points and its ratio
is 1, tied with classical statistics. Prior missing-baseline comparisons and
the original timed-out baseline stay unchanged. This evaluator update made
no model or simulation call and did not change any submitted answer.

The full check on the final procedure code stalled in the known older engine's
deliberate worker-crash test, after 354 passed tests. SIGINT after
over 15 minutes without progress reported `selectors.py:415`; interpreter
shutdown then needed SIGTERM of the verified own pytest/resource-tracker pair.
That `make` ended with exit 2. No test was weakened or skipped, and no engine
change was made. A fresh full check on `67268cd` exited 0: 1,339 passed, two
skipped, 62 deselected, 489.32 seconds; lint, formatting and types passed.
Logs are `/tmp/codex-skills/check-r14-full.log` and
`check-r14-final-retry.log`. This branch does not include the soak owner's
later engine repair, so this recurrence does not test their final code.
The subsequent scorer merge is checked separately below.

### Round 14 results

Both final `make analyse` commands exited 0 from immutable `67268cd`; the
separate `ee176ff` score watcher exited 0 and the final `84bfa91` reports agree.
Every score below uses 336 future attempts. The first repeat meets the four
requested marks; the second misses recall. Both remain old-world development
results, with diagnosis provisional for the stored cause-unit disagreement.

| Old D2 logs run | Recall | Strict precision | Decoy rejected | Singleton rho (valued actions) | Three-fix share | Interval coverage | Mean absolute error, fleet points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat 1 | 6/7 | 6/7 | 1.00 | 0.9574 (13) | 0.8846 | 13/13 | 1.1803 |
| Repeat 2 | 5/7 | 5/5 | 1.00 | 0.8141 (11) | 0.5192 | 10/11 | 1.3761 |

Before this combined revision, Round 13 D2 recall was 6/7 in each repeat,
rho 0.8182/0.6125 and three-fix share 1/0.8846. Round 14 therefore improves
singleton correlation in both repeats but lowers recovery. It is not a
consistent improvement. Repeat 1's seventh diagnosis is a real gripper effect
below the detectable-cause threshold, which strict precision penalizes; neither
new repeat submits a targetless cause. The shared site/task cause unit remains
unresolved. Repeat 2 also abstains from the placement cause and leaves its
repair out of the candidate list, even though its memory records a plausible
unlogged physical/input explanation for the shared step.

Three-fix sets rescue 92/54 and break zero attempts. Six-fix sets rescue
132/83 and break 0/8, net 132/75, shares 0.9565/0.5435. Four-hour sets rescue
71/67 and break 1/6, net 70/61, shares 1/0.8714. Eight-hour sets rescue
132/83 and break 0/8, shares 1/0.5682. The primary baseline is classical
statistics at 37 net rescues; three-fix margin is 16.3690/5.0595 fleet points,
ratio 2.4865/1.4595. The provisional diagnosis caveat does not disappear in
these budget comparisons.

Repeat 1 uses the cost rule explicitly: moving cheap actions before the
four-hour action preserves the three/six-fix sets and improves its stated
four-hour scenario from 13.43 to 25.34 predicted points without worsening the
other cuts. Its actual four/eight-hour shares are 1. This demonstrates live
use, not an isolated wording effect: candidate choices also changed, and prior
agents sometimes found good cost orders unaided. It computes control's
conditional contribution after placement as 8.0818, above the drift repair's
6.8919 singleton, but still selects drift in the leading three. The helper
does not force consistent set comparisons across its alternative scenarios.
Its remaining calibration error is mostly optimistic camera recovery.

Repeat 2 has no false-positive diagnosis but fails to carry the latent physical
repair into its ranking. This is distinct from correctly declining to assert
an unresolved cause. It also gives the uncertain sampling reversal [0, 20.43]
despite the signed-uncertainty procedure; its actual -0.5952 points fall outside.
Repeat 1 uses the full signed identification envelope [-33.34, 33.34], explicitly
not a statistical confidence interval. The next general
work: separate repair-candidate generation from diagnosis assertion, retain
evidence-supported latent-target alternatives, and compare consistent response
scenarios without invented weights or automatic zero attribution. This should
be developed with a synthetic latent-condition case and transferred to newly
cleared worlds, not by teaching a planted condition or further tailoring to
these old worlds. No further procedure change is made in this session.

Both runs read all seven skills and used `pair_scenarios` once. Their agent CLI
elapsed times are 922.338/753.130 seconds; calls 88/85, one error each. Repeat
1 recovers from aggregate/nonaggregate mixing; repeat 2 from the reserved SQL
alias `both`. The only identifier-screen hit is repeat 1 query 58's “late
recovery” symptom beside an ordinary `episode_id` lookup. Manual trace and notes
inspection finds no identifier interpretation or seed derivation; lexical
screening alone cannot rule out an encoded attempt. Tool-log hashes and manual
disposition are in `20261009-r14-trace-review`.

After merge `89f7a4e` and notes `18068d3`, the full `make check` exited 0:
1,343 passed, two skipped, 62 deselected, 473.95 seconds; lint, formatting and
types passed. Diagnostic `faulthandler_timeout=120` did not require a retry.
The final skill revision's 174 focused checks and Codex `agents-gate` also
exited 0 before its measurements. Engine, fault, policy, identity and locked
dependency sources match the previous verified smoke revision `ecf30ef`:
`check-sim` passed 58 tests and
`smoke-world` reproduced 16/16 chains, with the stored reference untouched.
Those simulator checks were not repeated for these later procedure/scorer
changes. Subsequent edits are result notes only; their library/dry-run checks
exited 0 with 72 passed in 14.85 seconds.

Final evaluator receipt `20261009-final-84bfa91/rescoring-receipt-after-r14.json`
identifies all 25 reports: 22 completed logs answers and three ended no-answer
records, with unchanged answer hashes and metrics relative to `ee176ff`.
The original `r00` provenance refusal and two still-active experiment
continuations are explicit, not silently dropped. The three no-answer records
include failed continuations of existing repeats, not three independent
analysis draws. At 23:39 UTC both original A/B continuations still have no
engine records and no completed experiment-condition score.

## Session handoff

Prepared at 23:45 UTC for the renewed six-hour limit at 23:58 UTC. The frozen
numeric pass marks have been met by both original D1 logs-only repeats, but the
experiment condition has no completed repeat. Later logs procedures were
measured separately; their results do not substitute for those missing runs.

| Development logs result | Procedure round | Recall | Decoy rejected | Singleton rho | Three-fix achievable share |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 original repeat 1 | Original procedures | 6/6 | 1 | 0.9461 | 1 |
| D1 original repeat 2 | Original procedures | 6/6 | 1 | 0.9538 | 1 |
| D1 latest single run | 13 | 5/6 | 1 | 0.8977 | 1 |
| Old D2 latest repeat 1 | 14 | 6/7 | 1 | 0.9574 | 0.8846 |
| Old D2 latest repeat 2 | 14 | 5/7 | 1 | 0.8141 | 0.5192 |
| Old D3 latest single run | 9 | 6/7 | 1 | 0.9019 | 1 |

All gains use the world's 336 future attempts; the reported recovery share
divides net gain by that world's achievable gain. D1's original singleton
correlations cover 8/10 and 9/11
replay-scored actions; the later displayed runs have every replay-scored action
valued. All results are development-only. D1 ties classical statistics on
three-fix recovery, rather than demonstrating a gain over the strongest
baseline. Old D2/D3 diagnosis retains its cause-unit caveat. No replacement or
held-out world was opened; the required `NEW WORLDS READY` clearance has not
appeared.

`~/.cache/robots-probes/skills/20261009-session-handoff/handoff.json` contains
the exact ongoing run, queue, baseline, evaluator and verification paths. The
two `20261009-r10/experiments-{1,2}` processes continue from the original
`20261009-r03/experiments-{1,2}` waiting snapshots, using immutable
`~/.cache/robots-skills-run-r05` at `2a53302`. Their first batches have five
and three requests, 720 and 480 arm attempts, with four workers and policy
batch size four. Their own process trees were verified waiting in `flock`;
no engine run record or simulated attempt exists. The failed ten-worker
startup branches stay in `20261009-r05`; neither they nor the saved model
sessions are repeated or overwritten. Do not relaunch an active continuation
or edit its checkout. Partial-simulation recovery is not implemented.

The latest external scorer watcher expires at 23:56 UTC; if the queues finish
later, create a new `84bfa91`-or-later report with the preserved reproduction
inputs. The runtime will still write its own result and original-version
automatic score. No authentication error occurred. The development runner at
the original procedure revision already stops on authentication/rate-limit
errors; the later generic-runtime fix does not change that route.

This branch contains oracle repairs through `670775b`. Later oracle commits
`5888ae5`, `1c4876e` and `17965f1` are not merged here: the latter changes add
a hard failure for unavailable required count floors and a proof replay path
for missing fault-free cited outcomes. The stored reports used here have
populated count distributions; no new healthy-example replay was requested or
measured. Keep this limit explicit alongside the pending independent skill
review. The reviewed scoring repair is merged as `89f7a4e`; all final
procedure code is `67268cd`, measured results `fb69249`.

Next work is to finish and score the preserved A/B repeats, then test the
general candidate-generation and consistent-set-comparison follow-ups on newly
cleared development worlds. The two final repeats show why: even a high rank
correlation can coexist with a missing useful action, and correctly computed
conditional gains do not ensure that the selected set uses them. Do not
rewrite the saved answers or insert a world's planted condition into a skill.

Authored scope exceptions earlier in this session were the lazy public exports
in `contracts/__init__.py`, the type-only identity reference in
`contracts/actions.py`, and the small scanner classmethod rename in
`log/leaks.py`. Before the orchestrator's direction, its bookkeeping file received an
interim boundary-review row (`07789f6`) and a stopped-delivery row (`53eb75d`);
it has been left to the orchestrator since. No authored `docs/` change, push,
main merge, publication, login-store edit or stored reference update was
made. The main work items remain open for the missing experiment evidence and review.

At 23:56 UTC the final `84bfa91` score watcher exited 1 at its stated waiting
deadline: the two experiment continuations were still pending, and the original
`r00` record was refused for missing repeat metadata. Its 25 completed reports
remain valid outputs; this exit is not a successful experiment score or a new
analysis failure. The dedicated completed Round 14 watcher exited 0. Both
experiment processes were still alive, with zero engine records. The latest
queue snapshots and closing receipt are kept beside the handoff JSON.

## Round 15 — review repairs and the freeze candidate (2026-10-10 UTC)

Decision: the orchestrating session's review of `3ea810d` clears the recorded
results as reproducible but requires pipeline repairs and a stricter knowledge
boundary. This round starts the library from Round 12 (`f8c203d`). No Round 13
or 14 procedure is carried forward: the four-state scenario helper, cost-order
permutations, added abstention wording and policy priors are withdrawn. Their
historical runs, scores and notes above remain records. No score in this file
is a measurement of the new freeze candidate. No analysis, oracle scoring,
model call, new-world inspection or GPU/smoke job is authorized in this round.
The orchestrating session schedules one independent evaluation later.

The generator's own fault measurements are no longer admissible skill knowledge,
even as qualitative paraphrases. Public description, dictionary, catalogue,
vocabulary and general analysis/arithmetic remain admissible. No fault-library
source, calibration output, replacement development world or unseen fault type
was consulted to choose the revised procedures.

Removed passages (including earlier revisions restored at the Round-12 reset):

| Passage / location | Removal reason |
| --- | --- |
| “Tested reductions in sampling steps were near flat” and the supplementary-suite qualifier; “actions per chunk were modest alone but strongly compounded a command delay”; the mild/severe image-view claim and absence-of-precision-calibration claim (`sensitivity-from-logs`, Round 14) | Test-bed measurement priors, including the planted interaction. Entire paragraph withdrawn, not merely its numbers. |
| “Policy settings … mostly mild on healthy robots, but … make other causes much worse together than apart” (Round 12) | Earlier qualitative restatement of the same private response knowledge. |
| Sensor-pose small-error tolerance, steep loss and harmless-then-costly drift; image degradation's graded response | Empirical shapes from the fault library are not public fleet facts. |
| Wear's object/task response and complete removal by gripper service or replacement | Measured selectivity and private fixing semantics do not establish this fleet's response. |
| Control timing's threshold and near-total loss under changed control mode; reverting removes the whole effect | Private response and efficacy assertions. |
| Software reversion “removes all of it, and nothing else does” | A private exclusivity/completeness assertion beyond public `revert` semantics. |
| Site-condition restoration removes the effect; lighting/background “often harmless” and expected near-zero gain (`sensitivity-from-logs` and `episode-cards`) | Fault-library response/decoy knowledge. |
| Mechanism-to-verb rows for displaced camera, image degradation, stale frames, gripper wear/force loss, delayed opening, arm torque/controller cycles, and detector errors; “each listed verb removes … completely”; service/cleaning/replacement equivalence examples | The public catalogue specifies eligibility, not these mechanism-specific complete-removal mappings. Replaced with component-to-permitted-verb rows only. |
| Controller detector diagnosis, prescribed degraded direction and zero true-success gain for the known detector class | System ontology is moved to the public glossary; the diagnosis/efficacy prescription is withdrawn. Recording symptoms alone do not establish a cause. |
| Steady fleet-wide residuals imply a policy weakness that no repair removes (`sensitivity-from-logs`, `failure-modes`) | Scope and rate alone cannot establish this private mechanism or its irreparability. |
| Deterministic symptom-to-mechanism/repair suggestions in `episode-cards`, including periodic motion's policy-key list | Replaced by explicitly tentative comparisons grounded in the public component/key definitions, without response priors. |
| “Configuration, software and site causes do not show in health at all” (`scope`) | Test-bed-specific telemetry behavior; the replacement asks what the public metric actually measures. |
| Known-mechanism complete-removal/equal-gain wording in `rank-actions` | Private efficacy rules; catalogue eligibility and price do not determine efficacy. |
| Miniature README interpretation attributing closed-on-nothing episodes to its planted camera change | Historical truth interpretation does not belong in the library guidance. Public telemetry threshold measurements are retained with their limited scope. |

The glossary is served by `toolserver/public_system.py`, in the `vocabulary`
part of `describe` (also `all`) for every export, including old immutable ones.
It describes `controller` as the policy/control computer including the robot's
outcome detector and outcome recording. It contains no fault schedule, response
curve or diagnosis instruction. Other component descriptions remain the public
vocabulary's definitions.

Retained class-knowledge passages in `episode-cards`, `failure-modes`,
`sensitivity-from-logs` and `rank-actions` have explicit begin/end markers.
`--ablate-fault-classes` strips them while materializing a separately hashed
library. Malformed markers fail closed; the statistical SQL remains. A test
checks the actual bundled source and stripped library through the common launch
audit, with execution stubbed before any model or CLI session.

Pipeline repairs in this round:

- Merge `main`, then `merge/batch-2` because main lacks `7741ee5`; resolve
  `log/leaks.py` and `tests/test_integration.py` with the export branch's versions.
  Agent-written prose is handled at JSON value/line locations. Structural keys,
  extra files, hidden names and later hidden tokens remain subject to the scan.
- Bind an initial export to the full fleet model's public label and exact
  visible base episode records, independently of the hidden world id.
- Check all experiment actions against public domains, target coverage and
  visible revert history before charging the request. Identical arms refuse.
  Cached engine plans receive the same preflight before any saved job runs.
  Engine refusals retain ids, free budget and permit a final answer. Detailed
  engine errors remain on the host; only fixed public codes are returned, with
  public explanations in `describe vocabulary`. Budget refunds validate the
  stored result rather than trusting a bare `refused` field.
- Require experiment-day bounds at every launch containing arm episodes; check
  all such rows, including uncited rows. Reuse and recheck completed queue replies
  when resuming an ended checkpoint into a new directory. Copy engine history
  and preserve queue bytes for the engine's existing plan/resume mechanism.
  An interrupted caller cannot certify that its systemd worker ended: without
  the worker's completion record, copying refuses and durations remain unknown.
- Test `analyse()` with synthetic logs and stub calls, heavy dispatch and scorer.
  Its tests cover automatic repeat allocation, model configuration, auth/rate
  stops, refusal readout, finalization and interruption after a completed queue.
- Reserve repeat ordinals before launch in a shared append-only receipt registry
  keyed by world hash, condition and actual library digest. Continuations keep
  the same reservation; independent failed runs keep their ordinal.
- Record model tokens/time, queue time, simulation time and launch/queue audits.
  Price equivalents use a dated official Fast-price snapshot. CLI aggregate
  usage lacks per-request context-tier/cache-write detail, so cost is an explicit
  lower/upper bound, not an invented exact bill; per-thousand normalization uses
  the recorded observed-episode count. Missing call results and interrupted
  phases are marked incomplete/unknown. Subscription charges are not inferred.

Mechanical checks find literal/structural leaks; they cannot establish semantic
skill generality. A paraphrased schedule or unnamed oracle prior can pass.
Independent reading remains required, as do human controls on test-world access
and the freeze. This distinction is now stated in both library and integration
READMEs. The full experiment command is prepared there, not executed here.

Implementation: `dc3a88c`; saved-plan fail-fast repair: `ad772fe`. Required
merges: `8635feb` (main) and `1780b02` (merge/batch-2). The scanner conflict
resolution takes the export branch's files as directed.

Verification at 01:42 UTC:

- The one full `make check` exited **2**: lint and formatting passed, mypy
  passed on 304 source files, and pytest reported **1,524 passed, 2 skipped,
  62 deselected, 1 failed** in 628.04 seconds. The failure was
  `test_cached_plan_cannot_omit_replies_or_answer_another_queue`: public
  preflight had begun reading an export before rejecting a malformed saved
  plan. `ad772fe` restores structural/hash rejection first, then checks public
  applicability. This is a repaired failure, not a passing full gate.
- After that repair, all **9** tests in `test_engine_experiments.py` and
  `test_experiment_validation.py` passed (15.68 seconds), including the original
  failing case, legacy-plan applicability and host-only exception details.
- The final **12** `test_analysis_pipeline.py` cases passed (10.23 seconds),
  including the last interruption guard added while the full check was running.
  It leaves a queue's end state unconfirmed if its caller is interrupted;
  a live systemd worker must not be copied as an ended checkpoint.
- The earlier **180** launch-boundary, tool-server, library and CPU queue tests
  passed (123.40 seconds); the full check also passed their cases apart from
  the saved-plan failure described above. Final changed-source lint, format
  and type checks passed.

Receipts: `/tmp/codex-skills/repair2/make-check.log`, `queue-final.log`,
`interruption-final.log`, `final-focused.log` and `types-final.log`. At most two
pytest drivers ran concurrently. No second full check has been started: the
assignment limits it to once, so permission to repeat it has been requested
to meet the separate exit-0 delivery requirement. The green full-gate claim
therefore remains pending. No fresh CLI wire gate, simulator/GPU/smoke run,
analysis or development-world scoring was performed. The agent-runtime files
changed here are `audit.py`, `run.py`, `sessions.py` and `toolenv.py`; their
synthetic checks do not replace the deferred `make agents-gate` wire checks.

All numerical agent scores remain unchanged historical records;
revised-library performance is **NOT MEASURED**.

## Round 16 — second review's launch and execution repairs (2026-10-10 UTC)

The orchestrating session's full check of `e0d1954` superseded Round 15's
pending gate: exit 0, **1,526 tests passed**.
This round follows the complete second review. No analysis, oracle
score, model call, calibration, GPU job or new-world inspection is authorized.
All past scores retain their original revisions; no performance claim is made
for this library.

The real-world launch regression uses D1's actual fleet, base manifest and
cleared export, and reads the stored D1/D2/D3 analysis states without modifying
them. Before the scanner repair it reproduced **3,924 terms**, the `queue_s`
hit in the session instructions and `summary` hit in the schema, and ordinary
prose hits in saved experiment notes. This is not addressed by deleting prompt
words or weakening this branch's scanner. `log/leaks.py` belongs to the export
builder; its delivered `fix/export-terms` repair is the required dependency.

Skill changes and provenance:

- Only the tentative fault-class hypotheses in `episode-cards` are marked for
  ablation. Markers were removed around `failure-modes` scope comparisons,
  `sensitivity-from-logs` public verb table and `rank-actions` action semantics
  and scoring instructions. Their text stays in both conditions. The ablation
  regression requires exactly `episode-cards` to contribute removed passages.
- Retained method developed on the second development world: `scope`'s
  same-day confounding and unlogged alternatives (reviewed lines 59–75), its
  task-conditioned site comparisons and support limitations (97–101), and
  `failure-modes`'s shared site/task residual comparisons (74–78). These are
  provenance disclosures for the freeze report, not evidence from an unseen
  world and not measured response priors. None was added in this round.
- The stale `write-diagnosis` reference to the old failure-modes step 4 is
  corrected: a broad, steady spread alone is insufficient to establish that
  no catalogue action reaches a mode.
- The completion criterion and distinction between a reported outcome and
  physical completion are served by `toolserver/public_system.py` through
  `describe(dictionary)` for every export. The same module continues to serve
  the controller glossary through `describe(vocabulary)`. These are system
  semantics, not a skill's predicted benefit for a fault class.

Pipeline changes:

- Condition preflight and the public domain glossary follow the registered
  condition-change application: signed lighting stops or stops/warmth; two- or
  three-coordinate placement; two-value object batch; clutter blend in [0,1].
  A domain regression resolves each form and invokes the registered hook's
  `new_episode`, including the public zero values. Only the application dispatch
  was inspected for this repair; no dose-response finding or calibration output
  was consulted or added to the library.
- Only a request-specific planning failure returns a fixed public refusal for
  that request. A machine or worker failure stops the analysis with its exact
  queue/plan intact for continuation. `run.json.experiments` records requested,
  completed, refused and pending requests and delivered arm episode counts.
- Repeats are reserved at the first model-call dispatch, after gate, audit and
  process creation, before task input is sent. The fixed OS-account register
  cannot be redirected by runtime/cache environment variables or `--out`.
  Its key includes world hash, condition, library digest, model and effort.
  Receipts carry the first call and an integrity hash, are checked on resume
  and before scoring, and survive failed dispatched calls. A refused preflight
  or failed process creation gets no ordinal. A launch ordinal does not assert
  that the provider accepted a request.

Verification and delivered commits are recorded below after the final checks.

Focused checks before the final gate: **43** synthetic pipeline/experiment cases
passed, **177** tool/library/launch cases passed, **3** repeat-boundary cases
passed (including a plain Python stdin child), and each of the real D1
condition-value and planted-secret tests passed. The initial focused tool test
still expected the old positive-only lighting-domain message; that assertion
was corrected to the signed-stop domain. The new real-data tests' initial
fixtures needed the tool's required explicit arguments and case-normalized
canary comparison; both were corrected. Changed-source lint/types pass.

At the final-check decision, there is no delivery of `fix/export-terms`
and that branch still points to its starting commit `644de01`. As explicitly
allowed by the assignment, the two positive real-world regressions are marked
**strict expected failures, waiting on that delivery**. The planted-secret and
public-condition regressions are ordinary passing tests. The waiting markers
must be removed when merging the delivered repair; an unexpected pass is a
failure so this cannot silently remain waived. No launch compatibility with the
real development terms is claimed on this branch before that merge.

Final verification on `561b845`: the single `make check` exited **0**. Ruff
lint/format passed; mypy passed on **305 files**; pytest reported **1,560 passed,
2 skipped, 62 deselected and 2 expected failures** in **1,078.53 seconds**.
The two expected failures are exactly the waiting real-world positive audit
regressions. The real-world protected-name/seed and public-condition tests
passed. Full log: `/tmp/codex-skills-repair4-make-check.log`. Only one full check
was run, and no source changed while it ran. Implementation commits are
`3607224` and `561b845` (waiting-marker line length).

The final dependency check found candidate `f4017c6` on `fix/export-terms`, but
**no delivery comment**, so the conditional merge was not performed. The export
builder's 02:34 UTC comment also reports a specification conflict:
two unchanged saved notes contain `gripper-wear` or `gripperwear`, forms of a
protected registered kind. Accepting every saved state while refusing every
such kind form requires the orchestrating session's decision. This branch does
not weaken that protection. The comment reports successful candidate
pipeline/schema/library checks, but that candidate is neither merged nor
independently verified by this round's tests.

Runtime edits are limited to `agents/run.py` and `agents/sessions.py`; the
engine edit is confined to the additionally owned `engine/experiments.py`.
No other builder's implementation file, `docs/design/` or bookkeeping file was edited.
No model/analysis call, development-world score, GPU job, full simulation,
`make agents-gate` or smoke replay was run. The scheduled full analysis must
still establish the live CLI and simulated-queue continuation after the scanner
repair/decision. Its command is in `integration/README.md`.

## Round 17 — integrate the delivered scanner surfaces

The orchestrating session delivered `fix/export-terms` at `86f7cc4` and directed
an immediate merge; merge commit `b494f91` brings that exact delivery into this
branch. This supersedes Round 16's waiting decision and resolves its vocabulary
policy question: natural-form agent hypotheses may pass, while multipart
identifiers of planted kinds, roles, traits and hidden fleet properties remain
protected alongside actual seeds, canaries and fault ids. The two real-world
positive regressions no longer have expected-failure marks.

No skill procedure, class hypothesis or helper changed, and no score was
computed. `log/leaks.py` was imported by the merge and was not edited here.
The runtime changes are `agents/checks.py`, `agents/audit.py` and
`agents/measure.py`:

- `source_audit` explicitly selects `ours` for exports, tool descriptions,
  prompts, schemas, source and both library variants.
- `audit_state` first checks all state on `ours`; only recognized notes and
  exact schema locations containing agent prose are checked on `agent` in
  place of that first result. Results, the CLI box, filenames, keys, envelopes,
  configuration/condition values and unknown fields/files remain `ours`. The
  public contract's text-valued training-data/teleoperation descriptions are
  agent prose, including their appearances as action values. Prose is passed as
  JSON strings, so embedded JSON still receives seed/identifier checks. Ambiguous
  structural keys cannot impersonate an agent prose location. The old masking
  of public words is removed.
- The legacy `measure.blinding` diagnostic retains the conservative `ours`
  surface for mixed measurement records that also contain pipeline content.

The surface regression inspects the actual scanner arguments at the stubbed
launch boundary and rejects any pipeline text passed as agent prose. Other
cases check natural hypotheses, protected identifier spellings, malformed
location keys, embedded JSON, pipeline envelopes/results and the conservative
measurement diagnostic. The real-data regressions use D1's actual term inputs,
cleared export, both libraries and all stored D1/D2/D3 analysis states. Existing
worlds and results remain unchanged. The runtime and integration READMEs name
the policy's limits; a lexical check does not establish complete blinding.

Verification: the real D1 pipeline/schema/source/full-library/stripped-library
case passes, as does its public condition-value case. The final stored-state
and protected-name run exits **0**, with **2 passed, 2 deselected** in **229.03
seconds**: all **26** saved states pass (9 D1 logs, 2 D1 experiments, 11 D2
logs, 4 D3 logs), while the real-world canary, kind identifier, trait identifier,
fault id and seed still refuse. Both former waiting tests are ordinary tests.
Logs: `/tmp/codex-skills-surfaces-real.log` and
`/tmp/codex-skills-surfaces-real-state-final.log`.

The first real-state attempt exposed one additional prose location: a saved
training-data action's `value` said “nominal scenes”. The public action contract
makes that field text; it now receives the agent policy, with a six-case
regression separating text-valued verbs from configuration settings and testing
both natural and identifier spellings. No phrase was changed in the saved state.

Focused runtime tests passed **53** cases; **47** launch cases passed in the
launch-only run. The new provenance guard initially stopped before reaching the
scanner because its fixture omitted required request fields, then because its
invented `extra.txt` filename matched a strict structural term. Those fixture
errors were corrected, keeping pipeline scans strict. The final guard plus the
six text-valued action cases exit **0** (**7 passed, 47 deselected**, 13.57
seconds). Thus the focused pass set covers all **54** launch cases, including
malformed-key, embedded-JSON and mixed-measurement boundaries. Logs:
`/tmp/codex-skills-surfaces-focused.log`,
`/tmp/codex-skills-surfaces-launch-final.log` and
`/tmp/codex-skills-surfaces-guard-final.log`. Focused Ruff lint/format, mypy on
the three runtime files, and `git diff --check` also exit **0**.

No model/analysis call, score, GPU/simulation job, `make check`, fresh CLI wire
gate or smoke replay was run in this round. The orchestrating session owns the
full check and scheduled live experiment analysis. These checks establish the
requested content/surface boundary, not successful authenticated continuation.

## Round 18 — final scanner API integration

Merge `f65bb21` imports the requested `fix/export-terms` delivery `bee34d3`.
The API no longer has a surface selector: trees and mounted code are always
strict. `source_audit` supplies our prompts/schema/tool descriptions as `texts`
and the bundle/libraries as `code`; `measure.blinding` keeps mixed archives as
strict trees. `audit_state` retains the approved extraction and supplies only
source-named notes, request power notes and answer prose as `agent_texts`, with
an empty tree. JSON-string wrappers preserve nested-JSON checking without
sending any pipeline fields as agent text. Text-valued actions in answers
remain prose; requests contribute only their power note.

The provenance guard now checks the exact `agent_texts` source/value mapping.
The real-world protection fixture supplements D1's actual audit inputs with
one synthetic non-countable fault id (`x9e52a18c`), since the retired world's
actual ids are all countable under the delivered one-letter/two-digit rule.
No stored run, truth file or agent state was changed. A further identifier
probe checks a planted kind embedded inside a longer identifier. No skill
procedure/helper, tool behavior or `log/leaks.py` was edited.

The integration exposed a semantic difference from approved `26e9d4c`, beyond
the API change. The final scanner classifies undeclared names as planted;
`min_detectable`, a saved statistical alias, now refuses in agent prose because
it is also a recipe field. The scanner delivery's saved-state measurements used
the older `e0d1954` caller, which masked this alias. The approved caller removed
masking in Round 17. We do not reintroduce an exception or alter the agent's
stored notes to obtain a pass. The existing regressions remain ordinary tests,
not expected failures, so the mismatch remains visible. It is reported to the
orchestrating session; repairing the scanner's policy is outside this round's ownership.

The focused launch suite exits **1**, with **53 passed and 2 failed** in
**238.26 seconds**. Both failures are the existing positive agent-prose cases
containing `min_detectable`; the separate reproducer fails for the same term
in **5.92 seconds**. The final-source provenance/strict-tree/nested-JSON subset
exits **0**, **3 passed, 52 deselected** in **15.29 seconds**. The real-world
protected-name/seed probes and public-condition-value test also pass. Logs:
`/tmp/codex-skills-final-api-launch.log`,
`/tmp/codex-skills-final-api-prose.log`,
`/tmp/codex-skills-final-api-guards.log` and
`/tmp/codex-skills-final-api-real.log`.

The first real-world run timed out while auditing both libraries in one test
under the existing 300-second deadline, and stopped its saved-state check at
the first refusal. Each library and each saved session now has its own test
case; no assertion or deadline was weakened. This lets the complete stored
set be checked despite a refusal.

The complete saved-state run exits **1**: **25 passed, 1 failed, 4 deselected**
in **325.24 seconds**. It checked the memory and outbox of every one of the
**26** stored sessions. The sole refusal is
`skills/20261009-r03/experiments-2/run.json`, `memory/notes.md`,
`min_detectable`; no other term or location refuses. Evidence:
`/tmp/codex-skills-final-api-states.log`.

Both separate D1 pipeline cases pass with the actual export and term list:
prompts, schema, tool descriptions, mounted source and the full and ablated
libraries. The command exits **0**, **2 passed, 28 deselected** in **359.57
seconds** (full **171.78 seconds**, ablated **187.46 seconds**), recorded in
`/tmp/codex-skills-final-api-pipeline.log`. Focused Ruff lint/format checks,
mypy on the three changed runtime modules and `git diff --check` exit **0**.

The API adaptation and provenance protections are verified; the requested
all-states acceptance guarantee is **not met** because of the remaining
scanner-policy mismatch. No failure was marked expected or waived. No model
call, analysis, scoring, GPU/simulation job, fresh agent gate, smoke replay or
`make check` was run. The orchestrating session owns the full check and the
decision on the scanner's treatment of this statistical shorthand.
