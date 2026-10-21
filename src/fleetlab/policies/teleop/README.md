# Human teleoperator in the fleet loop

Implemented on branch `stage5/teleop`. The existing
scripted expert remains available as `fleetlab.policies.teleoperator.Teleoperator`.
`HumanTeleoperator` adds a seeded human stand-in and a limited operator queue.
It takes over from the engine's saved mid-episode state and issues real actions
through the same simulator, fault hooks and placed-success check. A placement
requires LIBERO's goal to hold, the robot to be clear of the object, and the
object to be at rest. The simulator and the scripted expert provide privileged
geometry; this is a stand-in for an operator, not a vision-only human interface.
LIBERO is the manipulation benchmark used here; X-VLA is the original learned
robot policy.

The primary recovery number is the share of would-fail hand-overs rescued:
64/144 (44.4%) after the timeout rule and 27/63 (42.9%) after the random rule,
using the original policy's same-snapshot replay. By the original as-run outcome
the shares are 67/146 (45.9%) and 30/65 (46.2%). The rules are nearly equal on
rescue share; this measurement does not show a difference on that metric.
Entire-window placements are reported below alongside these conditional shares.
No held-out world is used. These results measure this human stand-in, not
claim 4's comparison with a learned supervisor.

The original human-model numbers were measured on 2026-10-08 at `a00a8da`,
before main's later engine changes. They have not been remeasured. The review
corrections merge main `2c83a27` as `7b2e35f`; the expert-only comparison is a
separate 2026-10-09 measurement at `be5facb`. Its identical snapshots and settings
are stated below, with the different code provenance kept explicit.

## What runs

The timeout supervisor first makes its decision. The robot repeats its last
command while that decision is pending. At hand-over the engine saves the
simulator, the original policy's remaining actions, the faults' internal state,
the success check and the last commanded pose and gripper command.

For deterministic admission, the collection pass uses
`RunConfig.defer_continuations=True`. It writes each continuation next to its
snapshot and completes the original episodes first. `admit` then orders all
hand-overs by simulated fleet time: episode day and scheduled start time plus
the snapshot's control step. Simultaneous requests tie by neutral episode id.
The earliest available operator gets the request; ties use the operator number.
Every admission reserves a fixed service slot. Worker completion order and
wall-clock performance never enter this queue. Admission overwrites the raw
continuation file with its complete human settings and queue assignment, so
a resumed fleet run finds the same assignment.

The continuation restores the snapshot, holds the saved command during its
queue wait and reaction delay, then starts a fresh expert. Holds use the
episode's control mode: repeat the target pose in absolute control, zero motion
in relative control, and preserve the gripper command in either case. Every
hold is a simulator step and goes through active fault hooks. The adapter's
open-gripper no-op is not used. Queue and reaction steps count as elapsed robot
time but do not spend the continuation's active budget or the success check's
failure timeout. Physics still runs and a real placement may finish the job
during the wait.

The paced expert advances one command every `pace` control ticks and holds the
last target between commands. This slows its command and script cadence; the
robot controller still runs at 20 Hz, and motion duration need not multiply
exactly by `pace`. Gripper commands remain unchanged on intermediate holds.
A fumble leaves the gripper open at the first grasp closure, then uses the
expert's ordinary retry. If the object is already held, the first fumble opens
the gripper and the expert attempts to pick it up again. Neither case resets
or edits the simulator. A fumble can cause a real loss. Natural expert failures,
including an upside-down bowl or an inaccessible object, remain losses.

## Parameters and evidence

These are explicit scenario assumptions, not measurements of human operators.
They are configurable with a JSON `HumanSettings` file. Their purpose is to
make delay, capacity, pace and retry cost measurable before human data exist.

| Parameter | Default | Meaning and justification |
|---|---:|---|
| `operators` | 2 | Two shared operators; an assumed limited-staff scenario. |
| `reaction_min_s`, `reaction_max_s` | 2, 4 seconds | Uniform reaction delay after an operator becomes free; assumed, not a sourced population estimate. |
| `pace` | 2 | One expert command every two 50 ms control steps, or 10 commands/s versus the expert's 20; assumed slower cadence. |
| `fumble_probability` | 0.10 | One Bernoulli draw per hand-over; assumed chance of a failed first attempt. Draws and actually applied fumbles are reported separately. |
| `expert_steps` | 600 | The measured expert's logical step budget, retained from the scripted-teleoperator spike. |
| `service_s` | 90 seconds | A fixed operator reservation including reaction and active control. It covers the worst configured reaction plus `(expert_steps + 1) * pace / 20` seconds of active budget, with slack. Smaller invalid reservations are refused. |

The service reservation is deliberately conservative: an operator stays booked
for the whole slot even if placement finishes early. Reports distinguish
reserved operator seconds from used operator seconds (consumed reaction plus
active control). Queue time is robot idle time, not operator time. This
choice makes capacity reproducible without estimating duration from privileged
outcomes. It does not model early release of an operator or cancellation of a
reservation when an object settles before service begins.

The reaction and fumble draws use the contracts' `noise_for(identity,
snapshot_step, sample=604)`. A normal cumulative-distribution transform gives
uniform draws; the distributions and stream number are fixed. The expert's
existing grasp-planner seed still comes from the identity's noise seed. No
draw uses a process id, current time, worker number, global random generator,
fault label or outcome table. A continuation's parameters enter its job key.
Jobs without human parameters keep their existing keys.

Sources for the control convention and expert behavior are the bundled
[actions.py](actions.py) and [expert.py](expert.py). The earlier empirical
expert results are recorded in repository documents
`spikes/teleop-absolute/README.md` and `spikes/teleop-expert/README.md`:
999/1,000 placements from start states, 1,174/1,193 on constructed disturbances,
and 474/535 when taking over after X-VLA's original budget. Those are expert-alone
results, not results for this human model. The queue and default human values
are assumptions made here; no citation is being presented as empirical support
for them.

## Checks and measurements

The CPU proof uses LIBERO, the scripted server policy `SwayPolicy`, the timeout
supervisor, actual snapshots, the fleet worker layout and the default human
settings. It does not load X-VLA. Tasks 0, 1, 4 and 7 of `libero_spatial`, stored
start state 2, are handed over at step 31 after a decision at step 21 and ten
supervisor hold steps. The continuations run again in reverse order with one
worker instead of two. Every state hash, both camera image hashes, executed
action and chain endpoint is compared.

| CPU check | Result |
|---|---:|
| Supervisor episodes against recorded-decision replay | 4/4 exact; `chain_ok=true` |
| Human continuations against repeat in another layout | 4/4 exact; 4,590 chain rows compared |
| Human placements | 4/4 |
| Requests with a 90-second operator queue wait | 2/4 |
| Default fumble draws in this small sample | 0/4 |
| Collection, continuations and repeat, total wall time | 174.5 seconds |

The proof is at `~/.cache/robots-probes/teleop-loop/cpu-v2/proof.json`.
Its four hand-overs used 11.80, 13.10, 12.25 and 12.15 operator seconds;
each reserved 90 seconds. Reaction consumed 3.95, 2.10, 3.90 and 3.30 seconds;
queue waits were 0, 0, 90 and 90 seconds. The first driver trial under
`cpu-default` completed simulations but failed during run-level proof
finalization; it is not used as a completed measurement.

A paired check removes reaction, queueing and fumbles and changes only command
cadence, from the same four snapshots. Both settings placed 4/4 objects.
The slower cadence is a modest slowdown, not a twofold duration multiplier:

| Spatial task | 20 commands/s, active seconds | 10 commands/s, active seconds | Duration increase |
|---|---:|---:|---:|
| 0 | 7.05 | 7.60 | 7.8% |
| 1 | 9.70 | 10.30 | 6.2% |
| 4 | 7.65 | 8.45 | 10.5% |
| 7 | 7.90 | 8.85 | 12.0% |

This evidence is `cpu-v2/pace-proof.json`. It supports a slower command-paced
operator in these four cases, not a claim about a person's actual movement speed.

A separate real-physics test forces fumble probability to one, queues the
episode for 320 steps (16 seconds, longer than the original 280-step budget),
and runs the same continuation twice. The fumble was applied, the expert
retried and placed, and the chains matched. That test passed in 35.0 seconds.
`make check-sim` passed all 40 simulator tests in 259.2 seconds on this busy
machine, including the forced-fumble test. Targeted engine, queue and report
tests passed. The full `make check` exited zero: lint and strict type checks
passed for 195 files, with 780 tests passed, two skipped and 44 simulator tests
deselected, in 193.7 seconds.

The world pass also repeated four X-VLA-origin human continuations in reverse
order with three workers instead of four. All four were exact over 3,223 chain
rows, including both camera hashes, simulator state, actions and endpoints.
These four cases ended without placement; none drew a fumble or queued for an
operator. The separate CPU checks exercise successful placement, queueing and
a forced fumble. Both world collection passes also passed their sampled
recorded-decision replays (34/34 episodes per pass).

Confidence is high in these measured counts and replay comparisons. They do
not establish replay for every fault or configuration, or predict another
world's recovery rate. Confidence in resemblance to real operators is low:
the numerical parameters are assumptions, and the model inherits a privileged
expert's strengths and missing skills. Human recordings and further development
worlds could change those conclusions. No independence-based error bar is
attached to correlated episodes and overlapping fault cohorts.

## Development-world scoring

`measure world` samples the entire observed or scoring window with a separate
identity-seeded stream. It chooses no episode using a failure or fault label.
The stand-in timeout trigger is step 200 by default; the current engine calls
it one step after that threshold. Decision delay is a fixed, assumed 500 ms,
not a measured model or subscription latency.

After collecting and admitting the timeout hand-overs, the baseline chooses
the same number of episode identities uniformly with another fixed seed
stream. Selected episodes hand over at the early step-20 timeout, so normal
placements have less opportunity to finish before their scheduled hand-over.
The measured run matched the actual hand-over counts, 153 in each method.
This is a matched-count random-episode baseline. Its timing differs from the
late timeout, so it cannot isolate the value of episode selection or establish
supervisor superiority.
The actual hand-over counts are reported, rather than silently assuming they
match. Both methods keep every active fault in force.

Each hand-over is also replayed from the exact saved snapshot with the original
policy and queued actions, no supervisor and no teleoperator. This is the
remaining original attempt budget, not the longer operator budget. The report
counts hand-overs, placements, counterfactual successes, rescues and breakages.
This measures control, delay and the different attempt budgets together.
It also compares with the world's original as-run outcomes, which have no
supervisor hold. That distinction matters: the supervisor's physical hold can
itself change a result.

Only the separate `report` command reads outcome truth for scoring and fault
types for stratification. Fault-type groups are overlapping cohorts of active
faults, not causal attributions; their counts may add to more than the total.
The report refuses a missing as-run denominator. Supervisor decision time and
actual hold steps are charged separately as robot idle time. Human queue and
reaction time, used and reserved operator time, fumble draws and applied fumbles
are recorded. Each labelled recovery can be reconstructed from its snapshot,
persisted settings and the executed-action chain under `truth/chain/`; this
work does not claim to demonstrate retraining from those trajectories.

### Measured window and recovery

The ready development world w42 contributes all 336 scoring episodes: 12 robots,
14 episodes per robot per day, across days 8 and 9. It uses X-VLA in the world's
fixed batch size of four and its pinned `fp16_res32` precision (16-bit model
weights with selected calculations kept at 32 bits). Measurement ran from
clean commit `a00a8da`, with four simulator workers. Both methods used six
operators and otherwise the defaults above. The timeout decides at step 201,
normally handing over at snapshot step 211; the random trigger decides at step
21 and normally hands over at step 31. Each control step is 50 milliseconds.

In the tables, H is the number of hand-overs, P the number placed after takeover,
C the number placed by the original policy replayed from the same snapshot,
and A the number that placed in the original world run before any intervention.
E is the number placed by the expert alone from those same snapshots, with the
human model switched off. P, C, A and E use the same H episodes within each method.

Rescue share comes first: its denominator includes only hand-overs that would
fail under the chosen counterfactual, rather than every hand-over.

| Method | Same-snapshot rescues / would fail | As-run rescues / would fail |
|---|---:|---:|
| Timeout | 64/144 (44.4%) | 67/146 (45.9%) |
| Random | 27/63 (42.9%) | 30/65 (46.2%) |

| Method | H | P/H: human model | E/H: expert alone | C/H: snapshot without takeover | A/H: original as-run |
|---|---:|---:|---:|---:|---:|
| Timeout | 153 | 71/153 (46.4%) | 70/153 (45.8%) | 9/153 (5.9%) | 7/153 (4.6%) |
| Random | 153 | 107/153 (69.9%) | 105/153 (68.6%) | 90/153 (58.8%) | 88/153 (57.5%) |

The timeout P count splits into 69 placements after at least one operator
action and two placements while waiting for the operator to react. Episode
`e000126` placed after three idle ticks (0.15 seconds), and `e000835` after
17 (0.85 seconds), both with zero active operator ticks. Both also succeed
under C and A, so neither is a rescue. All 107 random placements had an
active operator action.

A rescue means takeover placed and the same-snapshot replay did not. A breakage
means the reverse. Entire-window placement includes the unselected episodes.

| Method | Rescues | Breakages | Entire window placed |
|---|---:|---:|---:|
| No hand-over, world as run | — | — | 190/336 (56.5%) |
| Timeout | 64 | 2 | 254/336 (75.6%) |
| Random | 27 | 10 | 209/336 (62.2%) |

The random method's higher P/H includes many episodes that would succeed anyway.
The timeout method targets more would-fail episodes and therefore produces more
rescues and placements across this window, while its rescue share is nearly
the same as the random rule's.
The distinct trigger times and longer operator budget prevent interpreting that
comparison as the isolated value of episode selection. C also differs from A:
both replay baselines start after the supervisor's physical hold.

The expert-only pass retains every fault, the saved snapshot, the original
fixed batch size and precision, four workers, the 600-command expert limit and
the fleet continuation budget of 1,202 active ticks. It disables queueing,
reaction delay, command pacing and fumbling. It uses the unchanged
`Teleoperator` factory, which normally finishes before the outer budget.
The two 153-snapshot batches took 257.8 seconds in one heavy job (145.1 seconds
for timeout and 111.7 for random), starting before the 11:30 PDT cutoff, with
no worker incidents. Their source includes main's merged engine changes;
E-versus-P differences therefore retain this code-provenance limitation.

The expert alone also rescued 64/144 timeout would-fail cases and 27/63 random
ones. Within the timeout cohort, nine cases placed only with the expert alone
and ten only with the human model. In random, none placed only with the expert
and two only with the human model. There is no aggregate recovery penalty from
the assumed human settings in this sample; this does not calibrate real humans
or establish that delay and fumbling are beneficial.

### Active fault cohorts

Each row counts episodes carrying that active fault type. A policy knob is a
changed inference setting; a software release is a robot-software change
carrying another fault; an object batch variant changes object properties.
Other row names describe the changed camera, gripper, lighting, backdrop,
furniture, controller mode or success detector. Healthy means no active injected
fault. These rows overlap: for example, backdrop change, furniture shift and policy knob share
the same hand-over cohorts here. Their counts must not be summed or read as
effects caused independently by each fault.

| Active cohort | Timeout H | P | E | C | Random H | P | E | C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Backdrop change | 93 | 24 | 18 | 3 | 50 | 14 | 14 | 8 |
| Camera degradation | 26 | 0 | 0 | 0 | 11 | 0 | 0 | 1 |
| Camera pose offset | 15 | 15 | 15 | 1 | 17 | 17 | 17 | 9 |
| Control mode change | 28 | 23 | 17 | 0 | 14 | 14 | 14 | 0 |
| Furniture shift | 93 | 24 | 18 | 3 | 50 | 14 | 14 | 8 |
| Gripper wear | 13 | 11 | 10 | 0 | 50 | 50 | 50 | 46 |
| Healthy | 0 | 0 | 0 | 0 | 11 | 11 | 11 | 11 |
| Illumination change | 17 | 13 | 12 | 0 | 50 | 50 | 50 | 45 |
| Object batch variant | 17 | 13 | 12 | 0 | 50 | 50 | 50 | 45 |
| Policy knob | 93 | 24 | 18 | 3 | 50 | 14 | 14 | 8 |
| Software release | 27 | 18 | 24 | 5 | 12 | 2 | 0 | 4 |
| Success detector error | 5 | 3 | 3 | 0 | 13 | 13 | 13 | 12 |

### Where the losses sit

Of 82 timeout takeover failures, 63 have exactly
`backdrop_change + furniture_shift + policy_knob`, optionally with
`camera_degradation`, and end with `fail:attempts_exhausted`. The corresponding
random count is 36 of 46 failures. All 26 timeout camera-degradation failures
belong to that four-fault combination; the camera row is not a separate causal
camera result. Additional timeout failures carry other fault combinations or
end for other reasons.

Eight timeout continuations end at the success check's 60-active-tick
"LIBERO only" limit while the expert is still acting; seven carry
`software_release`. The check remembers its first LIBERO success even if the
native predicate later becomes false. For `e000744` it was already remembered
in the saved snapshot, so the continuation has no new LIBERO-first-fire row.
These are check-ended losses, distinct from exhausted expert attempts.

The furniture-shift investigation found a command-frame defect, rather than
stale pre-shift collision geometry or an unreachable object, in representative
episode `e000148`. `Scene.body_pos`, `Scene.geom_points` and `Scene.eef_pos`
read current MuJoCo world positions; the cached surface points are local to
each geom and transformed by its current world pose. The expert then emits
world-frame absolute targets. `FurnitureShift` exposes a shifted observation
frame and subtracts the 4 cm offset from absolute commands, but the expert
forms its command from the native world pose instead of that observation frame.

At its first active tick the native end-effector y is -0.260408 m, while the
observation reports -0.220408 m. Its initial vertical-rise plan preserves
native y, but the action hook sends the commanded target another 0.040000 m
sideways. The exact diagnostic replay matches all 762 original chain rows and
logs repeated 26–27 mm stalls, with no robot contacts or joints at their limits,
before exhausting five grasp attempts. A diagnostic-only replay compensating
that active command offset places the object, with all faults and the reaction
wait retained. It first diverges after the 68 reaction ticks, as intended.

This establishes the frame mechanism for one case with high confidence; it
does not prove the cause of every cohort loss. The defect is recorded as an
open bug. The expert remains unchanged in this branch, and the compensated
diagnostic is not included in any recovery table or presented as a production fix.

### Operator and computation cost

| Cost, across 153 hand-overs | Timeout | Random |
|---|---:|---:|
| Used operator seconds, total | 3,901.85 | 2,896.85 |
| Used operator seconds per hand-over | 25.50 | 18.93 |
| Reserved operator seconds per hand-over | 90 | 90 |
| Reserved operator seconds, total | 13,770 | 13,770 |
| Consumed reaction seconds, total | 470.6 | 460.4 |
| Robot queue seconds, total | 630 | 1,170 |
| Episodes queued; maximum wait | 7; 90 seconds | 13; 90 seconds |
| Fumbles drawn; physically applied | 10; 4 | 24; 16 |
| Supervisor decisions | 155 | 153 |
| Assumed decision seconds, total | 77.50 | 76.50 |
| Actual supervisor hold seconds, total | 76.95 | 76.50 |

Two timeout decisions ended before hand-over, so decisions and hand-overs have
different denominators. Scheduled and consumed human queue seconds agreed in
this run. Drawn fumbles can remain unapplied when an episode ends or the expert
never reaches a grasp. Operator time charges unsuccessful attempts too. The
decision rule can trade failures against used time or conservative reservations;
the choice of cost axis remains with the cost-ratio sweep.

| Heavy-job phase | Wall seconds |
|---|---:|
| Timeout collection and its sampled replay checks | 818.1 |
| Timeout human continuations | 207.4 |
| Random collection and its sampled replay checks | 566.6 |
| Random human continuations | 190.0 |
| Timeout original-policy counterfactuals | 115.3 |
| Random original-policy counterfactuals | 265.6 |
| Four human continuation repeats | 19.7 |
| Whole driver, including phase overhead | 2,185.7 (36.4 minutes) |

This excludes waiting for the heavy-job lock and excludes the earlier CPU
checks. There were no recorded worker incidents. Peak job proportional resident
memory, which apportions shared pages between processes, was 10.78 GiB; at least
23.61 GiB of system memory remained available. Peak machine-wide GPU translation
table memory was 10.05 GiB; that includes other GPU users and is not this job's
isolated GPU allocation. Human-only phases made zero inference calls, although
the current fleet runner still loaded the original model in their server.
Avoiding that load is left as an idea.

Bundled [evidence.json](evidence.json) preserves the aggregate report, both CPU
proofs, world replay comparisons, the expert-only comparison, the single-case
frame diagnosis and per-phase resource records. Full snapshots,
chains and per-episode outcomes remain in the local run directory documented
below. No truth table was used for trigger selection, human draws or admission.

## Run it

From the repository root, after `make setup`:

```fish
tools/heavy-run --light uv run --locked python -m fleetlab.policies.teleop.measure cpu-proof ~/.cache/robots-probes/teleop-loop/my-cpu --workers 2
tools/heavy-run --light uv run --locked python -m fleetlab.policies.teleop.measure pace-proof ~/.cache/robots-probes/teleop-loop/my-cpu --workers 2
```

For a development world, first require its `runner.json` to say `"ready": true`.
The driver also checks this before reading its spec or identities. Four workers
and a limit of 480 episodes are a modest initial pass; w42's entire scoring
window has 336 episodes, so the measured pass uses all of it. Use six operators
for this first 12-robot world to keep simulated queue-step cost modest; this is
a declared scenario change from the default two, not an estimated staffing
requirement. Both methods use identical settings. Lower the worker count if
the memory guard refuses it. The driver uses this machine's pinned convolution
database and validates the world's software and fixed batch size. Simulation
of the world, including the human continuations and baselines, is one heavy
job. The deadline is checked after the heavy queue lock is acquired.
These are the recorded world-pass commands; for a new pass change the output
root and latest-start deadline to an authorized compute window.

```fish
echo '{"operators":6}' > ~/.cache/robots-probes/teleop-loop/w42-settings.json
tools/heavy-run uv run --locked python -m fleetlab.policies.teleop.measure world ~/.cache/robots-probes/teleop-loop/w42-scoring-6ops --world ~/.cache/robots-probes/worlds/w42 --window scoring --limit 480 --workers 4 --settings ~/.cache/robots-probes/teleop-loop/w42-settings.json --latest-start 2026-10-08T20:30:00-07:00
uv run --locked python -m fleetlab.policies.teleop.measure report ~/.cache/robots-probes/teleop-loop/w42-scoring-6ops
```

The review's separate expert-only measurement used the existing saved snapshots;
it did not rerun either original world pass:

```fish
tools/heavy-run uv run --locked python -m fleetlab.policies.teleop.measure expert-only ~/.cache/robots-probes/teleop-loop/w42-scoring-6ops --latest-start 2026-10-09T11:30:00-07:00
uv run --locked python -m fleetlab.policies.teleop.measure report ~/.cache/robots-probes/teleop-loop/w42-scoring-6ops
```

`expert-only/measurement.json` and its timeout/random layouts retain the separate
expert code revision and phase records. The report adds E only when that cohort
is present and refuses a mismatched episode denominator.

`sample.jsonl` fixes the denominator; `timeout.jsonl` and `random.jsonl` fix the
triggers. `admitted.jsonl` and the per-snapshot continuation files fix the human
queue assignment. `human-metrics.json` preserves used time even across resume;
phase run records preserve resource and timing evidence. `measurement.json`
records counts, code revision, elapsed time and four X-VLA-origin continuation
repeat comparisons. `report.json` holds both baselines and the detailed rows.
Reusing the same output path resumes completed jobs. Use a new path to change
settings, sample size or trigger time, because a run job's file key is its
episode id. The `--settings` argument accepts a JSON file of `HumanSettings`.

## Contract choices and limits

The additive fields `Job.teleoperator_params` and `Snapshot.last_commanded`
are proposed as a contracts change; the orchestrator owns the written
contracts. A legacy snapshot without a saved command remains usable by the
expert-alone policy; the human policy refuses it rather than guessing a hold.
The continuation's own `max_steps` is now counted beyond the snapshot step;
the earlier engine treated it as an absolute endpoint. A deferred continuation
pass filters earlier run samples out of its run-level proof; they cannot prove
a different continuation. Exactness of the human continuation is established
by an explicit independent repeat instead.

Admission is a batch of the collected hand-overs. It does not feed live queue
occupancy back into `SupervisorContext`; integration of that state and the
cost-ratio sweep are separate work.
The real supervisor, autonomous recovery comparison and learned decision rule
are outside this delivery. No held-out measurement, real operator calibration,
new recovery skill, model training or cross-machine replay is claimed.

An independent review of `93493ec` raised findings; they are addressed in this revision. The
required merged-branch checks are `make check`, `make check-sim` and
`make smoke-world`, with 16/16 stored-reference chains identical and unchanged
software. The original
world run verifies the chosen six-operator scenario, not the default
two-operator world scenario or sensitivity to the assumed human parameters.
The command-frame defect is deferred; no expert repair or new
smoke reference is part of this delivery.
