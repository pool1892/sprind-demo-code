# Narration: the installation takes on a new job

One section per beat of the film, headed `## <beat id>: <title>`; the ids are the
manifest's, and the beats are those of the plan's two stories (story A, "make this one
job"; story B, "make that a check the next shift can run"). Everything under a heading
is read aloud except HTML comments (notes for the editor, never read) and lines starting
with `>`. Written for a male standard American voice: short sentences, plain words, no
abbreviations.

**No number is written into this script.** Every number is a placeholder, `{name}`,
defined in `fields.json` beside this file as a field of a named record (a run's
`job_run.json`, a release receipt, a table). `python -m fleetlab.film bind` reads the
cut's records (`fleetlab.film-records/1`) and writes the bound script with
`bindings.json`, which names each value's file and its SHA-256. A placeholder whose
record is missing stays as `{name}`: a rehearsal shows it in the captions; a voice
recording and a final cut refuse it. A note marked `depends:` names the result a
sentence assumes; rewrite the sentence if the record says otherwise. A sentence claims
only what a run record shows.

`python -m fleetlab.film narration <bound script> --out DIR` writes the captions and
the spoken length at a normal pace; `tools/film-narrate render` reads it with the
chosen voice.

## open: Opening

This is a simulated robot installation: one robot arm at a kitchen table. Its policy,
the model that turns its camera images, a written instruction and the hand's measured
pose into motion, is fixed. Agents watch the
installation, run experiments on it, and change how the installation uses that policy.

## a0-title: Story A

First story: make this one job.

## a1-skills: The skills

The robot already has skills. Each skill is one task on this table, done by the policy
under its own instruction. Here, each one runs alone and delivers. The policy does not
change anywhere in this film.

<!-- depends: both clips are single-skill runs whose record says delivered. -->

## a2-request: The request

A shift lead asks for one job: the cheese into the black bowl, and the loaded bowl onto
the cabinet top. The cheese goes in before the bowl moves. The shift lead's bar:
{lead_deliveries} deliveries in every {lead_starts} fresh starts, {release_k} of
{release_n}. The agent offers to be judged on {acceptance_k} of {acceptance_n}.

<!-- depends: the release receipt's lead_fraction and K (the lead's bar, set 2026-10-20 08:15 Pacific; job_run.json release_reissued) and the third investigation's acceptance proposal (acceptance.K). The first investigation's message log is not shown: it carries the earlier bar of two in three. -->

## a3-naive: The naive attempt

The obvious chain switches to the next skill the moment the first one reports success.
That is the signal published scores count. On the first {naive_starts} development
starts, it delivers on {naive_delivered}. A success signal is not a delivery.

<!-- depends: the first investigation's first batch, the naive chain on development starts. -->

## a4-investigation: The investigation

The agent reads the object traces, looks at the frames, and writes its own check: was
the cheese released and settled when the bowl task began, and did it stay in the bowl to
the end? Then it tests transitions on the installation. Letting the cheese instruction
run on a little after its goal first holds delivers on {tuned_delivered} of the same
{tuned_starts} starts. On {fresh_starts} fresh starts, the same program delivers on
{fresh_delivered}. Adding a return home at the end breaks starts that had delivered.

<!-- depends: the agent's write_check and run_check calls; the batches of sessions two to four; the pair on screen is the first start the return home broke. -->

## a5-account: The account

The first shift writes its account to the shift lead and hands on what it built: a check
that finds where a delivery is lost, and a skill on why a return home can break a
delivery. It took {job_one_sessions} sessions, {job_one_starts} development executions on
{job_one_distinct} distinct starts, and {job_one_model_minutes} minutes of the agent's
model time.

<!-- depends: the first investigation's job_account.json (its handover lines, quoted on screen), its library (post-goal-release-retention, goal-triggered-home-can-break-delivery) and its costs. -->

## a6-second-shift: The second shift

The second shift starts from that library. It runs the first shift's check on its own
baseline, then builds a sharper one that finds where the load is lost in transport. Its
longer continuation delivers on {second_delivered} of {second_starts} starts it had
already examined. It took {job_two_sessions} sessions, {job_two_starts} executions on
{job_two_distinct} starts and {job_two_model_minutes} minutes. The third shift starts
from the library of both. Its shorter continuation delivers on {third_delivered} of
{third_starts} development starts, and the agent saves it as the job, in
{job_three_sessions} sessions, {job_three_starts} executions on {job_three_distinct}
starts and {job_three_model_minutes} minutes.

<!-- depends: the second investigation's library_mounted, its run_check of the first shift's check, its loaded-transfer-diagnosis check, its batch of session four and its account; the third investigation's mounted library (both shifts' checks) and its saved job. The accounts stand side by side: never say a later shift was cheaper or better because of an earlier one. -->

## a7-release: The release

The saved job runs on the {release_n} reserved starts no shift had touched, and the naive
chain runs on the same starts. The job delivers on {release_delivered}; the naive chain on
{release_naive_delivered}. Rescued: {release_rescued}. Broken: {release_broken}. Order
violations: {release_order_violations}. Against the lead's bar of {release_k}, the job is
released.

<!-- depends: the release run's receipt (re-issued at the lead's bar: passed true, released true; the earlier receipt beside it as receipt.json.k14) and its recorded episodes; the pair on screen is the first rescued start. -->

## a8-close: What changed

The robot policy stayed the same. The installation now runs a released job it did not
have before the first shift: the cheese in the bowl, and the loaded bowl on the cabinet.
Behind it, the library holds the job, {library_checks} checks and the skills of every
shift. The second job, the bowl onto the plate, builds on job one's library over three
shifts: the first two each save a check of their own, and the third builds a guarded
placement retry that delivers on {plate_dev_delivered} of {plate_dev_starts} development
starts. On the {plate_release_n} reserved starts it delivers on {plate_release_delivered},
level with the naive chain's {plate_release_naive_delivered}, so the installation's release
check holds it back, and the library keeps the program and {plate_library_checks} checks
for the next shift.

<!-- depends: the release receipt (released true), the release library's index and the three investigations' saved skills; job two's three shifts: run-1 (job one's run-5 library mounted; saved handoff-versus-transfer-failure), run-2 (job one's released library; saved handoff-versus-delivery-audit), run-3 (run-2's library mounted; saved the job loaded-bowl-plate-placement-retry, whose index purpose is an extended goal-guarded placement attempt, j0009 in session four on development starts 0 to 9, and loaded-transfer-stage-audit). run-3's release receipt: released false, delivered equal to the naive chain's; the passage says the release check held it back and claims no release. The library count is run-3's release library index (checks). A later shift that releases gets its own chapter, a9-second-job, between a8-close and b0-title. -->

## b0-title: Story B

Second story: make that a check the next shift can run.

## b1-constraint: The fleet and the lag that misleads

A fleet of robot arms works at several sites. At one site, inference stays in the
customer's data center, so commands arrive late, and nobody on site can undo that. Task
telemetry shows a lag there, and a lag on one robot at another site as well. From the
telemetry alone, the agent cannot tell what kind of lag each one is.

<!-- depends: the agent's notes in round 1 (the alignment-lag query: s1 and r06 both lag; "growing vs fixed delay, needs known-input probe"). The figure is the exhibit's success by site and day. -->

## b2-instrument: The instrument

So the agent builds a measurement. From a bounded set of motions it composes a step
probe, and asks for it on a robot at the delayed site, on the lagging robot, on a healthy
reference and on a robot at the third site. Commanded against measured: at the delayed
site, the arm starts late by the same amount every time. On the lagging robot, the delay
changes from one repeat to the next, and grows within the probe. One is a fixed delay.
The other is a backlog.

<!-- depends: the agent's request_probe call (its transcript) and the probe's recorded tables, drawn by the film (probe-first-fleet.png and its record). -->

## b3-change: The change

The probe picks candidates; task experiments on paired starts decide. The agent adopts
its changes, each with its reason: compensate for the delay at the first site rather
than remove it, roll back the software update at the third site, and service the lagging
robot's controller.

<!-- depends: the round-1 ledger's adopted entries and their rationales, verbatim on screen; the clips are the experiments' recorded attempts, control against the adopted arm on the same robot, task and initial state. -->

## b4-evaluation: The evaluation

Then the changed fleet runs {b_attempts} reserved attempts the agent never saw, beside
the original state on the same attempts. Before: {b_original_success} of {b_attempts}
succeed. After: {b_final_success}. Rescued: {b_rescued}. Broken: {b_broken}. The lagging
robot goes from {b_lagging_original} to {b_lagging_final} of {b_lagging_attempts}.

<!-- depends: the evaluation's summary and attempts table. -->

## b5-second-use: The second use

The probe and its analysis became a recipe in the library, runnable on named robots. On
a later fleet, robots fail again, and the recipe runs. At the third site, the arms first
move {b_later_delay_first} rows after the command and are half-way {b_later_delay_half}
rows after it: a fixed delay. The robot that fails on its own responds like the healthy
reference. Its first movement comes {b_later_isolated_first} row after the command; the
reference's, {b_later_reference_first} row after it. So the
skill's rule points at the images, not at the controller service that fixed the first
fleet. The operator's recorded decision follows it: no service for that robot; inspect its
images and its change log, and test the first fleet's adaptation at the third site before
adopting it.

<!-- depends: the recipe check on the later fleet (recipe-check.json, probe.png; first_movement_rows and half_way_rows count rows after the command, not absolute rows; "row" is singular because the bound values for r06 and r05 are 1), the skill's rule, verbatim on screen, and the operator's dated decision (decision.md, by the host acting as operator; nothing was adopted from it). The film claims what the rule and the decision point at, not what the cause was. -->

## close: The record

Every clip in this film is a recorded run. The record lists each one.
