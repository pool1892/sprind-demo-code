---
title: "The record: a robot installation takes on two new jobs"
---

<!--
Source of the reliability record page. Bind it, never publish this source as it is:
python -m fleetlab.film.record bind site/reliability/record/record.md
  --fields site/reliability/record/fields.json
  --records site/reliability/record/records-final-2-2026-10-20.json --out DIR --final
The bound page is DIR/page/index.md. Every number and quotation is a placeholder defined
in this folder's fields.json; the page carries no number of its own. HTML comments never
reach the bound page.

The reserved start-31 pair of the second job shows the plate knocked out of place during
the bowl task; the plate-push failures in the narrative come from the development runs.
Keep the two apart. Host feasibility runs shape the one methods sentence, never a finding.
-->

# Two new jobs from a robot's existing skills: {r1_delivered} of {r1_release_n} and {r2_delivered} of {r2_release_n} delivered

A simulated robot arm at a kitchen table was given two jobs it has no single skill for.
An AI engineering agent built each one from the robot's existing skills. On
{r1_release_n} fresh arrangements of the scene (*starts*), held back until the end, the
first job delivered on {r1_delivered} and the plain chain of the same skills on
{r1_naive_delivered}; on {r2_release_n} such starts for the second job, the counts were
{r2_delivered} and {r2_naive_delivered}. Rescued starts, failed with the chain and
delivered by the job: {r1_rescued} and {r2_rescued}. Broken starts, the reverse:
{r1_broken} and {r2_broken}.

The arm is driven by X-VLA, a public vision-language-action policy that turns camera
images, the hand's position and a written instruction into motion; its weights never
changed, and the agent cannot alter them. It writes programs (a skill under its
instruction until a goal holds, a scripted hand motion, a wait, a return home), reads
every recorded control step and both cameras, and keeps what it learns in the installation's library.

## The first job: the bowl on the cabinet, the drawer open

The request: leave the bowl on top of the cabinet and the cabinet's middle drawer open,
both at once, with the robot clear of the bowl and the cabinet, both at rest, and the
gripper open.

**What failed.** The plain chain starts the drawer skill the moment the bowl is on the
cabinet. On {r1_plain_n} practice starts it delivered on {r1_plain_k}. On all
{r1_plain_n}, the traces showed the switch coming while the bowl still touched the hand
and moved; in the failures, the hand turning toward the drawer hit the bowl again. The
chain's recorded run on one reserved start shows the same: the bowl lands on the cabinet,
the hand swings away with the bowl at its fingertips, and the bowl drops to the table.

**What separated the causes.** The agent wrote a *check* for exactly this hand-off, a
saved query over the run records, then changed one thing at a time on the same starts.
Opening the gripper in place delivered on {r1_open_only_k} of {r1_open_only_n}: open
fingers could still drag the bowl. Going home first delivered on {r1_home_only_k}, but
the way home could snag the bowl. Running the bowl skill on for {r1_finish_steps} more
steps, a move the library already held from an earlier job, delivered on
{r1_finish_only_k} alone and on {r1_finish_home_k} of {r1_finish_home_n} with the move
home after it: the policy released the bowl and lifted the hand, and from the move home on nothing touched the bowl again. The hand started home higher than in any open-then-home run; height,
orientation and the policy's next camera view changed together, as the agent notes.

**Release.** The unchanged procedure delivered on {r1_confirm_k} of {r1_confirm_n}
practice starts it had never used, then on {r1_delivered} of the {r1_release_n} reserved
starts, against {r1_naive_delivered} for the plain chain.

## The second job: the plate at the stove, the bowl on the cabinet

The request: leave the plate at the front of the stove and the bowl on top of the
cabinet, under the same end conditions, now for the plate, the bowl and the cabinet.

**What failed.** The plain chain, plate then bowl, delivered on {r2_plain_k} of
{r2_plain_n} practice starts. The weak step was the plate. Pushed, it ran into the cream cheese lying in its
path, or the hand took hold of the cheese instead, and on some starts the plate tipped or
never reached the stove front. Going home between the two tasks
helped the bowl, {r2_home_only_k} of {r2_home_only_n}, but the plate failures stayed.
Other variants, the first job's continuation among them, did not remove them.

**Moving the bottleneck.** Loading the cheese into the bowl first cleared the plate's
way, and every plate push held, but carrying the loaded bowl failed: {r2_loaded_k} of
{r2_loaded_n} delivered. So the agent kept the preparation and dropped the loading. Its
procedure runs the cheese skill for {r2_grasp_steps} steps, long enough to grasp the
cheese but not to load it, slides it {r2_slide_cm} cm sideways, opens the gripper for
{r2_open_steps} steps to set it on the table, then goes home, pushes the plate, goes home
again and puts the empty bowl on the cabinet: {r2_relocate_k} of {r2_relocate_n} on the
same starts. The cheese is not part of the request; the job moves it on purpose. Setting
it aside also changes the timing, the camera view and the hand's path, and the hand
brushes the bowl on the way; the agent records that these changes are not separated from
clearing the plate's path.

**Release.** The unchanged procedure delivered on {r2_confirm_k} of {r2_confirm_n}
practice starts it had never used. In {r2_flicker} of those runs the bowl's goal went off
and back on while the hand finished letting go: the goal holds once the bowl arrives,
release comes after, and delivery is judged at the end. On the {r2_release_n} reserved starts the job delivered on {r2_delivered} and the plain chain on
{r2_naive_delivered}.

## What the installation keeps

Each investigation leaves a job that an operator runs with one command, a check that
flags a fragile hand-off even in a delivered run, and a *learned skill* for the next
agent: the first tells an unfinished release from a collision on the way out or on the
approach to the next task; the second, that clearing an obstacle can move the bottleneck
downstream, and how to test a relocation without loading. The second investigation tested
the first job's continuation and built its check by extending the first job's.

The two investigations took {r1_wall_minutes} and {r2_wall_minutes} minutes from request
to final account, {r1_model_minutes} and {r2_model_minutes} of them model time, with
{r1_practice_runs} and {r2_practice_runs} practice runs.

## The agent builds an instrument

On a fleet of {b_robots} simulated arms at {b_sites} sites, an agent built its own
measurement, a scripted hand motion with the policy switched off that times how each hand
follows its commands, and with it told a fixed network delay at one site from a
controller falling behind on one robot, which ordinary telemetry lumps together. It
adopted {b_round1_count} changes after paired experiments; on {b_eval_final_n} later task
attempts it never saw, success rose from {b_eval_original_k} to {b_eval_final_k}, with
{b_eval_rescued} rescued and {b_eval_broken} broken. Saved as a recipe, the probe reran
with one command on a later fleet.

Everything ran in simulation (LIBERO household tasks, MuJoCo physics); each count is a
literal number of recorded runs, from {r1_plain_n} to {b_eval_final_n}.

## The counts

| Count | Result | 95% interval |
|---|---|---|
| First job, reserved starts {r1_release_range} | {r1_delivered} of {r1_release_n} delivered ({r1_release_pct}) | {r1_release_low_pct} to {r1_release_high_pct} |
| Its plain chain, the same starts | {r1_naive_delivered} of {r1_release_n} delivered ({r1_naive_pct}) | {r1_naive_low_pct} to {r1_naive_high_pct} |
| The same starts compared one by one | {r1_rescued} rescued, {r1_broken} broken; both delivered on {r1_both}, neither on {r1_neither} | |
| First job, practice starts {r1_confirm_range}, unused before | {r1_confirm_k} of {r1_confirm_n} delivered ({r1_confirm_pct}) | {r1_confirm_low_pct} to {r1_confirm_high_pct} |
| Second job, reserved starts {r2_release_range} | {r2_delivered} of {r2_release_n} delivered ({r2_release_pct}) | {r2_release_low_pct} to {r2_release_high_pct} |
| Its plain chain, the same starts | {r2_naive_delivered} of {r2_release_n} delivered ({r2_naive_pct}) | {r2_naive_low_pct} to {r2_naive_high_pct} |
| The same starts compared one by one | {r2_rescued} rescued, {r2_broken} broken; both delivered on {r2_both}, neither on {r2_neither} | |
| Second job, practice starts {r2_confirm_range}, unused before | {r2_confirm_k} of {r2_confirm_n} delivered ({r2_confirm_pct}) | {r2_confirm_low_pct} to {r2_confirm_high_pct} |
| Of those, runs whose bowl goal stopped holding after the bowl arrived and held again by the end | {r2_recovered_k} of {r2_recovered_n} ({r2_recovered_pct}) | {r2_recovered_low_pct} to {r2_recovered_high_pct} |
| Second job's reserved runs, cheese at the end untouched by the robot | {r2_cheese_robot_free_k} of {r2_cheese_robot_free_n} ({r2_cheese_robot_free_pct}) | {r2_cheese_robot_free_low_pct} to {r2_cheese_robot_free_high_pct} |
| The same runs, cheese at the end touching the table | {r2_cheese_on_table_k} of {r2_cheese_on_table_n} ({r2_cheese_on_table_pct}) | {r2_cheese_on_table_low_pct} to {r2_cheese_on_table_high_pct} |
| The same runs, cheese touching only the table | {r2_cheese_only_table_k} of {r2_cheese_only_table_n} ({r2_cheese_only_table_pct}) | {r2_cheese_only_table_low_pct} to {r2_cheese_only_table_high_pct} |
| The same runs, cheese touching the table and the plate | {r2_cheese_with_plate_k} of {r2_cheese_with_plate_n} ({r2_cheese_with_plate_pct}) | {r2_cheese_with_plate_low_pct} to {r2_cheese_with_plate_high_pct} |
| Fleet with the agent's changes, reserved attempts | {b_eval_final_k} of {b_eval_final_n} succeeded ({b_eval_final_pct}) | {b_eval_final_low} to {b_eval_final_high} |
| Fleet as it was, the same attempts | {b_eval_original_k} of {b_eval_original_n} succeeded ({b_eval_original_pct}) | {b_eval_original_low} to {b_eval_original_high} |
| The same attempts compared one by one | {b_eval_rescued} rescued, {b_eval_broken} broken | |

For each job the bar was set before any program ran: at least {r1_release_k} deliveries
on the {r1_release_n} reserved starts for the first and {r2_release_k} of
{r2_release_n} for the second, each proposed by the agent and approved by the shift lead
who made the request. The first proposal: "{r1_proposal}" The second: "{r2_proposal}"
The first job's receipt records that it {r1_verdict}; the second's, that it
{r2_verdict}. The coordinator reconciled the raw release episodes, each job's and its
chain's, one by one against the receipts: all {r1_raw_verified} of the first and all {r2_raw_verified} of the second. The requests were
selected through separate host feasibility runs, whose procedures and results the
engineering agent did not receive.

## Every program run on practice starts

The first job:

| What happens after the bowl lands | The agent's name for the program | Practice starts | Delivered |
|---|---|---|---|
| straight to the drawer (the plain chain) | `{r1_plain_program}` | {r1_plain_range} | {r1_plain_k} of {r1_plain_n} |
| bowl skill for {r1_finish_steps} more steps | `{r1_finish_only_program}` | {r1_finish_only_range} | {r1_finish_only_k} of {r1_finish_only_n} |
| gripper opens in place | `{r1_open_only_program}` | {r1_open_only_range} | {r1_open_only_k} of {r1_open_only_n} |
| hand returns home | `{r1_home_only_program}` | {r1_home_only_range} | {r1_home_only_k} of {r1_home_only_n} |
| gripper opens, then hand returns home | `{r1_open_home_program}` | {r1_open_home_range} | {r1_open_home_k} of {r1_open_home_n} |
| bowl skill for {r1_finish_steps} more steps, then hand returns home | `{r1_finish_home_program}` | {r1_finish_home_range} | {r1_finish_home_k} of {r1_finish_home_n} |
| drawer first, then bowl | `{r1_drawer_first_program}` | {r1_drawer_first_range} | {r1_drawer_first_k} of {r1_drawer_first_n} |
| the winning row's procedure, unchanged, on starts not used before | `{r1_confirm_program}` | {r1_confirm_range} | {r1_confirm_k} of {r1_confirm_n}{r1_confirm_interval} |

The first saved job runs that procedure in four steps: "{r1_step0_instruction}" until the
bowl is on the cabinet, at most {r1_step0_max} control steps; the same instruction for
{r1_finish_steps} more; the hand home with the gripper open, at most {r1_home_max}; then
"{r1_step3_instruction}" until the drawer is open, at most {r1_step3_max}.

The second job:

| What the program does | The agent's name for the program | Practice starts | Delivered |
|---|---|---|---|
| plate, then bowl (the plain chain) | `{r2_plain_program}` | {r2_plain_range} | {r2_plain_k} of {r2_plain_n} |
| bowl, then plate | `{r2_bowl_first_program}` | {r2_bowl_first_range} | {r2_bowl_first_k} of {r2_bowl_first_n} |
| plate, then bowl, the policy re-planning more often | `{r2_replan_program}` | {r2_replan_range} | {r2_replan_k} of {r2_replan_n} |
| plate, hand home, bowl | `{r2_home_only_program}` | {r2_home_only_range} | {r2_home_only_k} of {r2_home_only_n} |
| bowl, hand home, plate | `{r2_bowl_home_program}` | {r2_bowl_home_range} | {r2_bowl_home_k} of {r2_bowl_home_n} |
| bowl, the first job's continuation, home, plate | `{r2_bowl_finish_home_program}` | {r2_bowl_finish_home_range} | {r2_bowl_finish_home_k} of {r2_bowl_finish_home_n} |
| plate with a second attempt, home, bowl | `{r2_plate_retry_program}` | {r2_plate_retry_range} | {r2_plate_retry_k} of {r2_plate_retry_n} |
| plate, home, a second plate attempt, home, bowl | `{r2_home_retry_program}` | {r2_home_retry_range} | {r2_home_retry_k} of {r2_home_retry_n} |
| plate, home, bowl, the policy re-planning more often | `{r2_home_replan_program}` | {r2_home_replan_range} | {r2_home_replan_k} of {r2_home_replan_n} |
| bowl with the continuation, home, plate, home at the end | `{r2_bowl_finish_home_end_program}` | {r2_bowl_finish_home_end_range} | {r2_bowl_finish_home_end_k} of {r2_bowl_finish_home_end_n} |
| plate push cut short, home, plate again, home, bowl | `{r2_push_cut_program}` | {r2_push_cut_range} | {r2_push_cut_k} of {r2_push_cut_n} |
| plate push cut short, a scripted sideways slide, plate again, home, bowl | `{r2_push_slide_program}` | {r2_push_slide_range} | {r2_push_slide_k} of {r2_push_slide_n} |
| cheese loaded into the bowl, home, plate, home, bowl | `{r2_loaded_program}` | {r2_loaded_range} | {r2_loaded_k} of {r2_loaded_n} |
| cheese loaded into the bowl, home, bowl with the continuation, home, plate, home | `{r2_loaded_bowl_first_program}` | {r2_loaded_bowl_first_range} | {r2_loaded_bowl_first_k} of {r2_loaded_bowl_first_n} |
| cheese picked up, slid aside and set down on the table, home, plate, home, bowl | `{r2_relocate_program}` | {r2_relocate_range} | {r2_relocate_k} of {r2_relocate_n} |
| the same, with a lift before the slide | `{r2_relocate_lift_program}` | {r2_relocate_lift_range} | {r2_relocate_lift_k} of {r2_relocate_lift_n} |
| cheese loaded, home, plate, home, the bowl instruction naming the loaded bowl | `{r2_loaded_named_program}` | {r2_loaded_named_range} | {r2_loaded_named_k} of {r2_loaded_named_n} |
| cheese loaded, home, plate, home, the bowl instruction asking for a rim grasp | `{r2_loaded_rim_program}` | {r2_loaded_rim_range} | {r2_loaded_rim_k} of {r2_loaded_rim_n} |
| the cheese-aside procedure, unchanged, on starts not used before | `{r2_confirm_program}` | {r2_confirm_range} | {r2_confirm_k} of {r2_confirm_n}{r2_confirm_interval} |

The second saved job runs that procedure in {r2_steps} steps: "{r2_cheese_instruction}"
for {r2_grasp_steps} control steps; a scripted slide of {r2_slide_cm} cm over
{r2_slide_steps} steps; the gripper open for {r2_open_steps}; the hand home, at most
{r2_home_max}; "{r2_plate_instruction}" until the plate is at the stove front, at most
{r2_plate_max}; the hand home again; then "{r2_bowl_instruction}" until the bowl is on the
cabinet, at most {r2_bowl_max}. In both jobs a step that runs out of time stops the
program.

## In the agent's words

The first job:

- The request it was given: "{r1_request}"
- After its first results: "{r1_switch}" Its report of the session: "{r1_diagnosis}"
- On the plain chain: "{r1_said_chain}"
- On going home first: "{r1_said_home}"
- On opening, then going home: "{r1_said_open}"
- On the saved procedure: "{r1_said_finish}"
- On hand height, from its learned skill, where the comparator is open-then-home and the
  candidate the saved procedure, on the same {r1_finish_home_n} practice starts:
  "{r1_skill_heights}"
- On what its evidence does not separate: "{r1_said_unisolated}"
- On readiness: "{r1_said_ready}"

The second job:

- The request it was given: "{r2_request}"
- On the plain chain: "{r2_said_plain}" And later: "{r2_said_wrong_object}"
- On going home between the tasks: "{r2_said_home}"
- On the first job's continuation: "{r2_said_continuation}"
- On loading the cheese: "{r2_said_loaded}"
- From its learned skill, on the same practice starts: "{r2_skill_evidence}"
- On the saved procedure's first step: "{r2_said_grasp}" And: "{r2_said_deliberate}"
- On the confirmation runs: "{r2_said_flicker}"
- On what its evidence does not separate: "{r2_said_unisolated}"
- On the library it started from: "{r2_said_library}"
- On readiness: "{r2_said_ready}"

## What the installation keeps

- The jobs `{r1_job_name}` and `{r2_job_name}`.
- The check `{r1_check_name}`, with {r1_check_params} parameters so that it runs on other
  jobs: "{r1_check_purpose}"
- The check `{r2_check_name}`, with {r2_check_params} parameters: "{r2_check_purpose}"
- The learned skill `{r1_skill_name}`: "{r1_skill_description}"
- The learned skill `{r2_skill_name}`: "{r2_skill_description}"

The first investigation started from a library with {r1_mounted_jobs} saved job and
{r1_mounted_checks} checks, left by the investigations of an earlier job. That saved job,
`{r1_inherited_name}`, runs its first instruction, "{r1_inherited_instruction}", for
{r1_inherited_steps} more control steps after its goal first holds. The agent's account:
"{r1_said_library}" The second investigation started from the library the first left,
{r2_mounted_jobs} saved jobs and {r2_mounted_checks} checks. The library now holds
{r2_library_jobs} saved jobs, the two released jobs above and that earlier one, and
{r2_library_checks} checks.

## How this was recorded

Everything ran in simulation: LIBERO's household tasks on the MuJoCo physics engine, one
simulated arm driven by X-VLA with the same weights in every run. The engineering agent is {r1_agent_model} at reasoning effort {r1_agent_effort} in the first
investigation and {r2_agent_model} at {r2_agent_effort} in the second. It made
{r1_tool_calls} tool calls in {r1_sessions} sessions on the first job and
{r2_tool_calls} in {r2_sessions} on the second.

A *start* is one of a scene's stored arrangements of the objects. The agent could run
programs on practice starts. The reserved starts, {r1_release_range} and {r2_release_range} of each job's own pool,
were refused to it until the single final run, which ran the saved job and the plain
chain on each, {r1_reserved_runs} and {r2_reserved_runs} runs.

*Delivered* is the installation's own check at the end of every run: every requested goal
holds, as LIBERO's goal predicates judge it; no part of the robot touches the objects the
goals name; those objects are at rest; and the gripper is open. The agent could read the
goals and the delivery check and could not change them. It was never given the answer:
the host's feasibility runs were withheld from it, and of each final run it received only
the receipt. Neither request set an order for its two goals.

The cheese is not named by any goal, so the delivery check does not judge it. Its final
recorded state in the second job's {r2_cheese_runs} reserved runs: untouched by the robot
in {r2_cheese_untouched}, touching only the table in {r2_cheese_table_only}, and the table
and the plate in {r2_cheese_table_plate}.

For each job, the two runs on one reserved start ({r1_clip_start} and {r2_clip_start}), the
plain chain's and the job's, were re-rendered from their recorded actions for the scene and the wrist camera. A
re-render is accepted only if every recorded physics-state hash and the final verdict
match the original run: {both_rerender_accepted} of {both_rerender_reports} were
accepted, with {both_rerender_hashes} hashes compared and {both_rerender_mismatches}
mismatches. In the second job's pair, the chain pushes the plate to the stove front and
then runs out of time on the bowl, which never reaches the cabinet; the plate ends tipped
beside it.

Counts are literal. A count resting on at least 20 runs carries a 95% Wilson score
interval, a standard range for a proportion; versions run on the same starts are compared
start by start as rescued and broken.

These are counts from tens to hundreds of runs, in one simulator.

## The instrument, as recorded

The agent adopted {b_round1_count} changes: {b_round1_actions}. On a reserved panel of
task attempts on days after every recorded day, which the agent never ran or saw, the
fleet with these changes succeeded on {b_eval_final_k} of {b_eval_final_n}; the fleet as
it was, on {b_eval_original_k} of {b_eval_original_n}. Attempt by attempt,
{b_eval_rescued} were rescued and {b_eval_broken} broken. The probe's recipe, kept with
the learned skill `{b2_recipe_skill}`, ran with one command on a later fleet; that run is
recorded as {b2_outcome}.

## Provenance

Every record this page is bound to, by its identifier and the first characters of its
SHA-256 checksum, a fingerprint that changes if a single byte of the file changes:

{provenance_records}

Every number and quotation on this page, with its meaning and source:

{provenance_numbers}
