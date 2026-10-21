# Fault library and review repairs

The registry contains 23 calibrated fault types. Each type supplies per-episode
state and hooks for the scene, observations, commands, policy configuration or
logged outcomes. `curves.json` is the measured library record; object-suite
points and interactions belong in `supplement.points` and `supplement.joints`,
keyed by kind/pair, suite and level label. Spatial measurements are preserved.
Type verdicts are stored in `types[kind].verdicts[suite]`; `records.measured`
returns the selected suite's verdict and points. A decoy judgement applies
only to the task family queried. At the measured object levels illumination,
backdrop and warmer light are small causes by decision, and
must not be planted or scored as object decoys. Their statistical point roles
remain `effect`; the object type verdict is `kept`. The adaptive intervals
are descriptive, including in the record itself.

## Command convention

The engine scopes each action-hook application with `action_context` from
`command.py`. It carries the effective controller mode and the command
producer's coordinate frame. This also reaches hooks delegated through a
software release or a generic condition change. No hook inspects fault truth
to decide who produced a command, and no command convention is cached from the
nominal episode identity.

Absolute policy targets retain exactly the former float32 offset subtraction.
Relative commands bypass origin translation. The scripted expert advertises
world-frame targets. The engine applies inverse transforms explicitly declared
by the placement hooks before any buffering hooks. FurnitureShift supplies the
transform; table-height changes inherit it and condition changes delegate it.
Carriers forward the inverse transforms of their inner mechanisms in reverse
hook order, including placement changes carried by a software release.
Sensor errors in an observation do not declare a coordinate transform and
therefore cannot displace expert targets. This makes buffered
policy commands and new expert targets share one convention; the placement hook
then applies its usual translation. Direct world-frame hook calls bypass it. The human adapter
advertises policy coordinates during its queue/reaction wait, because it holds
the snapshot's last policy command; after that its targets and cadence holds
are world-frame commands. Context is reset after every action-hook application
and is absent from snapshots, so restored fault state cannot retain a stale
mode or producer frame.
Every in-worker policy must explicitly declare `commands_in_world_frame` as
a boolean after producing each command. The engine refuses missing or invalid
declarations; it reads the declaration after `act_one` because a human
adapter's idle-to-active transition changes the frame of that command.

The minimal edits outside the fault/calibration paths are
`engine/episode.py` (scope the hook loop and prepare world targets), `policies/teleoperator.py`
(command-frame property), `policies/teleop/human.py` (frame during the reaction
wait), and `tests/test_teleop_human.py` (check the transition). No live risk,
decision-rule or scoring code is changed.
The toy teleoperator in `engine/toy.py` and the scripted lift teleoperator in
`engine/scripted.py` also explicitly declare world coordinates. These two
one-line declarations are the additional edits outside the original owned
paths required to enforce the in-worker contract.

## Calibration validation

The stage-3 runner validates the exact planned episode set, a passing replay
sample, and the healthy-success floor for both fresh and completed resumed
runs. A rejected run retains `calibration-validation.json` and remains rejected
on restart even if a later invocation lowers the floor. New evidence requires
a new directory. Analysis rejects failed/missing replay validation, preserves
`chain_ok` on points and joints, and publication requires it to be true.
Object-only publication preserves the spatial stage-3 metadata. Data-fault
tables label logged successes separately from physical placements.

## Measurements and reproduction

The 2026-10-09 assignment's new outputs are under
`~/.cache/robots-probes/faults/codex-faults-20261009/`. The `pair` subdirectory
holds 300 common object attempts and healthy twins for the 14-step delay,
10-actions-per-chunk setting, and their joint, launched from main `cfedf66`
before the repairs. All four engine run records independently identify clean
`cfedf668b1344e792abd6a98a79e7f1df54b1e60`; the detached editable installation
points at that checkout's own `src/`. Its scratch driver and execution receipt
are retained.
The paired interaction is `joint - A - B + healthy`, bootstrapped over the
same episode indices (10,000 resamples, seed 0). Point differences use paired
Newcombe 95% intervals and exact McNemar tests.
Every stored point now names absolute source directories in `run_dirs` and
its twins in `healthy_run_dirs`; joints also name both `single_run_dirs`.
Merged extensions retain all contributing directories. The repeated
`object-healthy` names are qualified by their respective `pair`, `objects`
or `object-tail` roots. The top-level `runs` field is the legacy spatial
root only. Backfilling these paths changes no counts, intervals or point roles.

`python -m fleetlab.faults.object_calibration NEW_ROOT --budget-seconds SECONDS`
starts with 300 healthy attempts and 300 pairs at each of the three world
decoy levels. It extends decoys to 600 and then 1,000 pairs only while either
95% interval bound is more than 1.5 percentage points from zero, adding only
the necessary healthy twins. This precision-driven stopping is prespecified in
the driver; the intervals are descriptive under that adaptive sampling. A
copied completed 300-attempt baseline can be reused without repeating it. It next measures the explicitly listed one or two useful levels
of each new type at 100 pairs. It covers one plantable level of every type
first, prioritizing the gate's established 12-step camera lag and 0.2 torque
loss before their stronger alternatives. It extends effects of 2–15 percentage
points to 300, then uses remaining time for second levels and their extensions. Light colour is already covered by the decoy pass. It reserves a
conservative measured time allowance before starting each run, writes progress
and analysis after each accepted run, and stops when another run would exceed
the remaining budget. A wall-time alarm also stops and cleans up a slow or
interrupted run at the budget limit; only prior completed validated runs enter
the saved analysis. Run it through `tools/heavy-run` from a detached commit.
The budget starts after acquisition of the heavy lock; subtract the pair's
runtime and any failed launch time from the 9,000-second calibration allowance.

`python -m fleetlab.faults.verify_takeovers SOURCE NEW_OUT --count 20`, through
`tools/heavy-run --light`, samples 20 evenly spaced sorted jobs from the
preserved 63 timeout takeover failures in D2's moved-table cohort. SOURCE is
`~/.cache/robots-probes/teleop-loop/w42-scoring-6ops/timeout`. It repeats their
exact snapshots and human settings without policy inference, records the
repair's success, retains chains, verifies that every pre-active row matches
the stored chain, and verifies executed active targets against the expert's original world
targets within float32 rounding (at most 0.1 micrometre). This is a repair probe of selected failures, not a
remeasurement of all takeovers or the live decision rule. The completed live
simulator probe rescued 19 of these 20 stored failures. All 20 reaction/wait
prefixes match their original chains through the last idle action. All 3,284
active command rows match the original expert world targets with maximum XYZ
error 1.49e-8 metres. The remaining failure is retained with its metrics and
chain; a correct command frame does not guarantee takeover success. The probe
exited 0, took 134.4 seconds, and preserves its manifest, per-job results and
chains under `takeovers-20`. Its new `first-active-offset-proof.json` also
compares each first active command from the identical pre-active state: all
20 corrections match the stored table's four-centimetre displacement (along
y), within float32 rounding. This 20-case absolute-control sample does not
cover the complete set of takeovers affected by both repairs.

An independent review of `a57f337` established that 93 of the 153 stored D2
takeovers are stale: 65 under absolute control and 28 under relative control
at the moved table. The effective-mode repair changes relative-control
takeovers too, including their idle holds. The reviewer repeated three of
the 28: all chains changed from their first step and all three outcomes
flipped (two rescues became failures, one failure became a rescue). Alongside
the builder's 20 cases this leaves 70 of the 93 affected continuations
unmeasured. The zero-of-98 result below concerns policy attempts only.
No old D2 takeover figure is valid for reuse until all 93 affected takeovers
are rerun on the merged engine by the orchestrating session. This assignment
does not rerun them or replace any stored live-layer figure.

The object interaction measurement is complete at 300 common attempts per
condition: healthy 294 placed objects, delay alone 188, short chunks alone 282,
and the joint zero. Paired Newcombe 95% intervals for the losses are 35.33
percentage points [29.71, 40.97], 4.00 [0.86, 7.44], and 98.00 [95.38, 99.08],
respectively. The extra loss beyond adding the two single effects is 58.67
percentage points, paired-bootstrap 95% interval [52.33, 64.67]. All four runs
have exactly 300 outcomes and passing replay samples. The completed driver's
receipt records 3,674.3 seconds; the earlier stopped launcher adds a conservative
60 seconds to the heavy-queue allowance. The library stores these under
`libero_object`, alongside the existing spatial pair. The completed object
decoy batch and bounded remainder are recorded below. The existing health
readings remain unchanged. No result under an existing probe directory is
removed or rewritten.

The first object calibration batch exited 0 after 4,144.7 seconds. All eleven
source runs (including healthy and fault extensions) passed their replay and
exact-episode validations. The healthy cohort has 985 physical placements in
1,000 attempts; the 600-attempt prefix has 593. The recorded object curves are:

| World level | Pairs | Fault / healthy placements | Loss, percentage points (nominal 95% interval) |
|---|---:|---:|---:|
| Illumination, 2 stops dimmer | 1,000 | 969 / 985 | 1.60 (0.48 to 2.83) |
| Backdrop, 1 walls and floor | 1,000 | 970 / 985 | 1.50 (0.36 to 2.75) |
| Light colour, 1 warmer | 600 | 584 / 593 | 1.50 (0.09 to 3.10) |

These are adaptively sampled descriptive intervals, as specified above. None
fits wholly within 1.5 percentage points of zero. The analysis assigns all
three object points `role=effect`; their nominal decoy designation on the
spatial suite does not establish a near-zero object effect. Spatial records
and the measured interaction in both suites are unchanged. The requested
object calibration is therefore incomplete. The bounded remainder exited 0
in `object-tail`: the inert calibration update placed 98 objects in 100 pairs,
exactly matching its healthy twins (zero loss; nominal 95% interval -4.04 to
4.04 percentage points). All 100 entire chains match the preserved healthy
baseline from clean main. This also checks that the repaired runtime retained
the ordinary policy path on these object cases. Its automatic replay sample
passed; the receipt records 241.3 seconds including baseline validation and
238.7 seconds for the fault run.

Light colour and calibration update are two of the eleven requested stage-3
types. The other nine remain unmeasured on objects: lens contamination,
camera latency, gripper force loss, sticky release, arm torque loss,
controller overload, image crop, table height change and false-positive
success detection. Their one/two plantable levels remain explicitly listed
in `object_calibration.py`. No 300-pair extension or second level of those
types was measured. The three nominal decoys' interval target also remains
unmet; those levels are treated as small object causes by decision. No full
`OBJECT CURVES MEASURED` completion marker was posted.

The final `heavy-budget-final.json` charges 8,893.6 of 9,000 seconds
(148.2 of 150 minutes), including both mode probes, both smoke checks, the
stopped launcher, conservative startup reserves and an upper bound for the
first batch. Other builders' queue waits are excluded. The remaining 106.4
seconds cannot fit the conservative 304.5-second forecast for another
100-pair run, so measurement stopped. `object-final-analysis.json` combines
the accepted points while retaining the 1,000-attempt healthy summary.

`python -m fleetlab.faults.verify_d2_modes WORLD NEW_OUT`, through
`tools/heavy-run`, replays all 98 shifted-table attempts whose effective mode
differs from their nominal identity in the preserved D2 manifest. The source
is `~/.cache/robots-probes/worlds/w42`. It requires their outcomes to remain
unchanged, checks the new run's own replay sample, and compares eight evenly
spaced absolute-mode shifted-table policy chains with their stored originals.
This extra acceptance check was added on 2026-10-09.

The final explicit-transform runtime at `57f086e` passed this probe in
`d2-modes-validated`: zero outcomes moved in the 98-case cohort, all eight
absolute-policy chains were identical, and its automatic replay sample passed
(`chain_ok=true`). The probe exited 0 and took 338.3 seconds. An earlier
`d2-modes` probe measured the same unchanged outcomes and controls but was
configured as replay-only jobs, which disables recursive replay validation;
its `chain_ok=null` guard correctly failed. That failed probe is retained and
charged to the budget, rather than represented as a passing acceptance run.

The same final runtime repeated the 20 takeover cases in
`takeovers-explicit-20`, again rescuing 19 and matching expert targets within
1.49e-8 metres. All 20 entire repaired chains equal those from `takeovers-20`;
the new `repair-chain-comparison.json` records this comparison. The final
probe exited 0 and took 229.9 seconds as a light job.

Checks for the previous candidate through `a57f337` exited 0: `make check` at `cc2df59`
(1,110 tests passed, two skipped; lint, formatting and types passed),
`make check-sim` (58 passed), and final `make smoke-world` from the frozen
`57f086e` runtime (16 of 16 stored chains identical, no missing or changed
chains, eight physical successes, 34.1 seconds). Commits `57f086e` and
`72b89e1` change test annotations/metadata only; runtime code is the same as
`353f0fd`. The smoke reference
was not rewritten. Check logs live in the assignment's new cache directory.
After the final inert-point addition, focused calibration, fault-library and
world-loader checks also exited 0 (386 passed, two skipped). Runtime code
remained unchanged, so the passing simulator and smoke checks apply to the
final records too. This verifies the measured paths and sampled continuations;
it does not measure the other 70 affected takeovers or the nine remaining
object types. The older stage-2 execution-stop gap found during inspection is
tracked separately; the reviewed stage-3 validation repair is
covered here.

## Checks added after the independent review

`tests/test_faults_engine_commands.py` runs the engine loop with real placement
and mode hooks, a numerical toy environment, and the real scripted teleoperator
adapter with a fixed expert target. Only the placement cell hooks are disabled;
it measures command semantics, not scene physics or policy success. It covers
all three placement types composed with control-mode changes in both orders,
bit-identical policy commands, expert targets from resets and snapshots, the
per-command frame transition, explicit frame declarations, and software
releases carrying furniture/table shifts with delay in both orders and modes.
These tests run in `make check`.

`tests/test_fault_records.py` checks suite-scoped decoy verdicts and source
metadata for every stored point/joint, and rejects untraceable publication.
`tests/test_dose_response_stage3.py` now exercises object analysis with healthy
extensions, singles/joints and data-fault twins, low healthy floors, exact
episode sets, persistent rejection receipts and direct replay guards.
`tests/test_object_calibration_driver.py` checks the actual driver against
stored synthetic outcomes and real paired analysis with a virtual clock:
precision-driven extensions, no repeated indices, moderate-effect extension,
budget refusal and exclusion of failed runs. These prove orchestration and
analysis without inference or new calibration.

Sequential source-mutation trials caught all four engine breaks named by the
reviewer, plus missing-frame fallback, a carrier without its inverse, wrong
episode sets, ignored rejection receipts, dropped analysis/point replay
guards, a dropped object healthy floor, a looser stopping threshold, repeated
extension indices and a removed budget check (14 of 14). Each trial restored
the exact original source bytes. New review-round receipts and logs are in
`~/.cache/robots-probes/faults/codex-faults-review2-20261009/`. A read-only
record comparison confirmed all 99 points and four joints retain every old
value; all 61 named source/twin directories have findable `run.json` records.
No stored measurement was rewritten, and no model or GPU job was launched.

The command helpers are imported once at `run_job` entry, after the engine
module is initialized. Moving that import to module scope caused a circular
import through registry discovery of `verify_d2_modes`; fresh workers failed
before receiving any job. The single final `make check` invocation initially
retried that bootstrap error and continued on fresh workers after the import
was corrected. `tests/test_faults_bootstrap.py` adds a fresh-interpreter
import check for future merges; it was added after
that invocation's collection and is checked separately. Lint and types are
checked again after the import correction. The controller's independent
infinite retry of errors before assignment is filed separately; its
repair belongs outside the fault builder's ownership.

Final review-round checks exited 0: the single `make check` invocation
(1,156 passed, two skipped, 62 deselected; 1,305.8 seconds including the
initial bootstrap retries), the separately added fresh-import test (one
passed), final-source lint/format/types (272 files), and the single
`make check-sim` through `tools/heavy-run --light` (58 passed, 537.3 seconds).
The focused preflight passed 88 tests. Runtime is `984350c`; engine/carrier
guards are `ef1d6da` and record/calibration repairs are `e9280d8`.
The four named engine mutations and ten additional guard mutations were
caught before the import-only correction; that correction changes import
timing, not the tested command or analysis logic. Check logs and
`delivery.json` are in the review-round directory above. Only one pytest
invocation ran at a time.

No new smoke-world or model check was run, as the review assignment forbids
them. The historical smoke result above does not verify this review round's
carried-placement or startup changes. No D2 takeover was rerun, no live
decision/scoring rule was changed, and no additional object curve was
measured. The 93 stale takeovers and nine unmeasured object types remain for
their respective owners; original measurement records and the smoke
reference are intact. No merge or push was performed.
