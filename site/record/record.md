---
title: "The record behind the film: the installation takes on a new job"
---

<!--
Source of the record page. Bind it before reading or publishing it:
python -m fleetlab.film.record bind site/record/record.md --records <cut records> --out DIR
writes DIR/page/index.md, the page for the site; `site BUNDLE`
binds it and builds the bundle's site with it. Every number is a placeholder defined in
site/record/fields.json; [gap: ...] and [one of: ...] mark what the runs decide, and the
final writer replaces them from the records named in the gap. --final refuses an unbound
number or a gap; any bind refuses a page that shows a name fields.json forbids. These HTML
comments are notes for that writer and never reach the bound page. Host knowledge (the spikes) may
shape a definition here, never a finding.

Tone: results
first, stated with conviction; every count literal and bound to its record; the one line
on sample size and the simulator stays and nothing else hedges. The shift lead's bar for a
release (three deliveries in every five reserved starts, set 2026-10-20) is said once, in
its order: the 14 of 20 approved during the request, then the lead's bar applied to the
recorded release run (a review, 2026-10-20); the re-issued receipt and its
provenance are records of the page.
-->

# The record behind the film

This page is the record behind the film *The installation takes on a new job*: what was
run, what was counted and which recorded run each number and clip comes from. The tables
at the end list every record by an identifier and a SHA-256 checksum, a fingerprint that
changes if a single byte of the file changes. Everything happens in simulation, in LIBERO,
a public benchmark of household manipulation tasks, on the MuJoCo physics engine.

## The results

The robot's policy never changed. What changed is what the installation around it can do.

- **Job one, released.** Three shifts of an AI agent turned two of the robot's skills into
  one job. On {release_n} arrangements of the scene held back for the release, which no
  program had run on, the released program delivered on {release_delivered} of
  {release_n}{release_interval}; the obvious chain of the same two skills, which switches
  skills the moment the first one's success signal fires, delivered on
  {release_naive_delivered} of {release_n}{release_naive_interval}. Start by start, the
  program delivered on {release_rescued} where the chain did not and missed {release_broken}
  where the chain delivered, with {release_order_violations} violations of the required
  order, cheese before bowl.
- **Job two, three shifts on the installation's library.** A second request, the loaded
  bowl onto a plate, has had three shifts, each starting from the installation's library
  and handing over its own checks and learned skills. The third, with a starting skill on
  the robot's own control loop and gripper, built a placement step that tries again when
  the bowl has not reached the plate, and it delivered on {j2_saved_k} of {j2_saved_n}
  development starts. On {j2_release_n} reserved starts it delivered on
  {j2_release_delivered} of {j2_release_n}{j2_release_interval}, the obvious chain on
  {j2_release_naive_delivered}; it delivered on {j2_release_rescued} where the chain did
  not and missed {j2_release_broken} where the chain delivered, so the installation's
  release check held it back, and the program is in the library for the next shift.
- **Story B: a measurement built, then used again.** On a fleet of {b_robots} simulated
  arms at {b_sites} sites, an agent composed a step-response probe, a short scripted
  motion that times how each robot's hand follows its commands, tested candidate changes
  in paired experiments and adopted {b_round1_count}. On a held-back set of task attempts
  the agent never saw, the fleet with those changes succeeded on {b_eval_final_k} of
  {b_eval_final_n}{b_eval_final_interval}, the fleet as it was on {b_eval_original_k} of
  {b_eval_original_n}{b_eval_original_interval}; {b_eval_rescued} attempts that failed
  before succeeded after, {b_eval_broken} the other way round. Saved as a recipe, the probe
  ran with one command on a later fleet and told that fleet's fixed transport delay apart
  from a failing robot whose hand moves promptly.

The sections below define each term in full and give every count with its record.

## The installation and its fixed policy

An *installation* here is one simulated robot arm at a kitchen table, with its objects,
the software that runs it and the records it keeps. The arm is driven by a *policy*: a
trained model, X-VLA, that turns camera images, the hand's measured position and
orientation, and a written instruction into motion. The policy is fixed; none of its
weights change anywhere in this film. What changes is how the installation uses it.

A *skill* is one task the policy does on its own under its own instruction, such as putting
an item into a bowl. This installation has {policy_skills} skills, the tasks of LIBERO's
goal suite, all in one scene. The film opens with {a1_clips} recorded runs of single
skills; by the delivery check described below, {a1_delivered} of them delivered.

## Story A: the request

The *shift lead* asks the installation for work and sets the bar a release must pass; here
the lead's words are scripted. The request, in the lead's words: "{job_one_request_text}"

The *requested goals*, in LIBERO's spelling, are {job_one_goals}. The *order rule* is that
{job_one_order_rules}. It is checked on the recorded trace of every run, and a program
whose runs broke it cannot be released. A *start* is one fresh beginning of the scene, one
of LIBERO's stored arrangements of the objects. The installation keeps *development
starts*, on which the agent may try anything, apart from *release starts*, reserved for one
qualifying run at the end and refused to the agent until then.

*Delivered* is the installation's own *delivery check* at the end of every run: every
requested goal holds, as LIBERO's predicate functions judge it; no part of the robot
touches the objects the goals name; those objects are at rest, slower than one centimetre
and 0.15 radians a second; and the gripper is open.

Before any program ran, the agent proposed the *acceptance condition*, one run of the
frozen program on N = {acceptance_n} release starts with at least K = {s3_offered_k}
deliveries, and the lead approved it for the request. After the release run, the shift lead
set the bar for releases at {release_bar_num} deliveries in every {release_bar_den}
reserved starts ({release_bar_pct}) and judged the recorded run against it, K =
{release_k} of {release_n}; the recorded runs on all {release_n} starts are unchanged.

## Story A: the naive chain

The *naive chain* is the obvious way to chain skills, run by the host (the people and code
that run the installation, outside the agent's reach) as the comparison at the release. It
switches to the next skill's instruction the moment LIBERO's native success signal fires,
the signal every published LIBERO score counts, and does nothing in between. The first
shift's agent began with that chain itself: on development starts {s1_chain_starts} it
delivered on {s1_chain_k} of {s1_chain_n}. Its account: "{s1_chain_quote}"

## Story A: three shifts build the job

A *job* is several skills done in one continuous scene. The agent writes it as a
*program*: up to eight *steps*, each a policy step with an instruction and a condition that
ends it, a short scripted motion of the hand or gripper, a wait, or a return of the hand
to where it started. A program is named by a digest of its exact text.

The agent is an AI model, {job_one_agent_model} at reasoning effort {job_one_agent_effort}.
A *shift* is one investigation, in several sessions; between them the installation runs
the queued programs and records every control step. A *check* is a database query over
those records, saved with its purpose so that anyone can run it again. A *learned skill*
is a note an agent writes for later agents: when a method applies and what to look at; the
installation audits it for leaked answers before keeping it. The *job library* is the
installation's list of saved jobs, checks and probe recipes, which an operator can run
with one command. Each shift starts with the library the shifts before it left.

The first two shifts handed over checks and skills rather than a program, each keeping the
reserved starts for a program it judged ready:

- The first shift found the transition that matters: continuing the cheese instruction for
  {s1_cont_steps} steps after its goal first holds delivered on {s1_cont_k} of the chain's
  {s1_cont_n} starts and on {s1_fresh_k} of {s1_fresh_n} fresh development starts,
  {s1_fresh_starts}{s1_fresh_interval}, and the shift saved {job_one_learned_skills}
  learned skill and the checks {job_one_check_name}.
- The second shift, with that library mounted, tested both checks against its own baseline
  before it built on them, delivered on {s2_short_k} of {s2_short_n} with a
  {s2_short_steps}-step continuation and on {s2_long_k} of {s2_long_n} development starts,
  {s2_long_starts}{s2_long_interval}, with a {s2_long_steps}-step one, and saved the check
  {s2_check_name} and {s2_learned_skills} learned skills.

The third shift, with both shifts' library mounted ({s3_mounted_checks} checks), built the
job {s3_job_name}: the cheese instruction until its goal holds, a
{s3_job_continuation}-step continuation of it, then the bowl instruction. On development
starts {s3_saved_starts} it delivered on {s3_saved_k} of {s3_saved_n}. Its account of the
design: "{s3_finding}" It saved the check {s3_check_name}: "{s3_check_quote}" And it kept
{s3_learned_skills} learned skills.

## Story A: the release

The installation froze the program (digest beginning {release_digest}) and ran it once on
the {release_n} reserved release starts, {release_starts}, with the naive chain on the
same starts. The program delivered on {release_delivered} of
{release_n}{release_interval}; the naive chain delivered on {release_naive_delivered} of
{release_n}{release_naive_interval}. *Rescued* counts the starts the program delivered and
the naive chain did not: {release_rescued}. *Broken* counts the reverse: {release_broken}.
Both delivered on {release_both}, neither on {release_neither}. The program broke the
order rule {release_order_violations} times. Judged against the shift lead's bar of
{release_k} of {release_n}, the receipt (SHA-256 beginning {release_receipt_sha}) records
that the program {release_outcome}. The released job is in the installation's library,
which an operator runs with one command.

The three shifts side by side:

| | First shift | Second shift | Third shift |
|---|---|---|---|
| Agent sessions | {job_one_sessions} | {s2_sessions} | {s3_sessions} |
| Tool calls | {job_one_tool_calls} | {s2_tool_calls} | {s3_tool_calls} |
| Program executions (distinct development starts) | {job_one_starts} ({s1_distinct_starts}) | {s2_starts} ({s2_distinct_starts}) | {s3_starts} ({s3_distinct_starts}) |
| Minutes of model time | {job_one_model_minutes} | {s2_model_minutes} | {s3_model_minutes} |

## Story A: a second job

<!--
Job two is bound shift by shift: j2s1_* read the records job_two_s1* (job-two/run-1),
j2s2_* read job_two_s2* (run-2), j2s3_* read job_two_s3* (run-3); the request's fields
(j2_*) read the latest shift. The release of the third shift's saved program is bound as
job one's: records job_two_release (its receipt), job_two_saved_job and job_two_exposure,
fields j2_release_* mirroring release_*, and `library` is its release library. A further
shift adds a bullet, a column of the table and its own records and fields (j2s4_*); a
further release moves job_two_release, job_two_saved_job, job_two_exposure and `library`.
-->

A second request asked the installation for another job: "{j2_request_text}" Its goals,
in LIBERO's spelling, are {j2_goals}; its order rule is that {j2_order_rules}. Three
shifts have worked on it, each starting from the installation's library and handing over
its own checks and learned skills:

- The first shift, with job one's library as its second shift left it, tested the
  inherited checks on its first program: "{j2s1_library_quote}" Its best program, which
  names the bowl holding the cheese in its instruction and retries the transfer,
  delivered on {j2s1_best_k} of {j2s1_best_n} development starts. It handed over a check:
  "{j2s1_check_quote}"
- The second shift, with job one's released library, ran job one's saved check on its
  first program: "{j2s2_library_quote}" Its finding: "{j2s2_finding}" It handed over the
  check {j2s2_check_name}: "{j2s2_check_quote}"
- The third shift, with the second shift's library ({j2s3_mounted_jobs} released job,
  {j2s3_mounted_checks} checks) and a starting skill on the installation itself, in its
  task's words "{j2s3_installation_quote}", saved the job {j2_job_name}: the cheese
  instruction, a continuation of it, then the bowl instruction with a second attempt when
  the bowl has not reached the plate. It delivered on {j2_saved_k} of {j2_saved_n}
  development starts, {j2_saved_starts}; in its account, "{j2s3_finding}" It handed over
  the check {j2s3_check_name}: "{j2s3_check_quote}"

The installation froze that program (digest beginning {j2_release_digest}) and ran it once
on the {j2_release_n} reserved release starts, {j2_release_starts}, with the naive chain
on the same starts. The program delivered on {j2_release_delivered} of
{j2_release_n}{j2_release_interval}; the naive chain delivered on
{j2_release_naive_delivered} of {j2_release_n}{j2_release_naive_interval}. Rescued:
{j2_release_rescued}. Broken: {j2_release_broken}. Both delivered on {j2_release_both},
neither on {j2_release_neither}. The program broke the order rule
{j2_release_order_violations} times. Judged against the shift lead's bar of {j2_release_k}
of {j2_release_n}, the receipt (SHA-256 beginning {j2_release_sha}) records that the
program {j2_release_outcome}, so the installation's release check held it back. The saved
program is in the library for the next shift.

The shifts on job two side by side:

| | First shift | Second shift | Third shift |
|---|---|---|---|
| Library mounted | job one's after its second shift: {j2s1_mounted_checks} checks | job one's released library: {j2s2_mounted_jobs} released job, {j2s2_mounted_checks} checks | the second shift's: {j2s3_mounted_jobs} released job, {j2s3_mounted_checks} checks |
| Programs (program executions) | {j2s1_programs} ({j2s1_executions}) | {j2s2_programs} ({j2s2_executions}) | {j2s3_programs} ({j2s3_executions}) |
| Distinct development starts | {j2s1_distinct_starts} | {j2s2_distinct_starts} | {j2s3_distinct_starts} |
| Best program run, delivered | {j2s1_best_k} of {j2s1_best_n} | {j2s2_best_k} of {j2s2_best_n} | {j2s3_best_k} of {j2s3_best_n} |
| Checks and learned skills handed over | {j2s1_new_checks} and {j2s1_learned} | {j2s2_new_checks} and {j2s2_learned} | {j2s3_new_checks} and {j2s3_learned} |
| Agent sessions | {j2s1_sessions} | {j2s2_sessions} | {j2s3_sessions} |
| Tool calls | {j2s1_tool_calls} | {j2s2_tool_calls} | {j2s3_tool_calls} |
| Minutes of model time | {j2s1_model_minutes} | {j2s2_model_minutes} | {j2s3_model_minutes} |

## Story A: what the installation keeps

After both jobs, the installation's job library holds {library_jobs} saved jobs, job
one's released and job two's for its next shift, and {library_checks} checks, every one
written by an agent, with the learned skills each shift kept. The robot's policy is the
same.

## Story B: the fleet

A *fleet* here is {b_robots} simulated arms at {b_sites} sites doing ordinary
pick-and-place tasks from LIBERO day after day, with the same fixed policy. The agent sees
what an operator would see: each attempt's outcome, the robots' telemetry, a change log,
maintenance and costs.

A *lasting condition* lowers success and cannot be removed by any action an operator or
the agent can take. The public note of site {b_linked_site} says: "{b_linked_site_note}"
Commands from a distant server reach those arms late. In regular work on days
{b_ops_days}, before the agent's first decision, that site succeeded on
{b_delayed_site_success} of {b_delayed_site_episodes}{b_delayed_site_interval}
attempts, the other sites on {b_other_sites_success} of
{b_other_sites_episodes}{b_other_sites_interval}.

## Story B: the lag that misleads

Ordinary telemetry gives one rough number per robot: how many control steps (50
milliseconds each) its hand trails its commands. A slow command path, a controller that
falls behind and a policy that reacts late all raise it. The agent wrote the lesson into
the skill it kept: "{b_lag_lesson}"

## Story B: the probe

A *step-response probe* is a short scripted motion with the policy switched off: hold
still, step the hand along one axis, step back, hold, repeated, while the installation
records where the hand is. The *response shapes* it can show, in the agent's starting
skill: a fixed dead time (the hand starts late by the same amount every time), a dead time
that grows as a backlog builds, a prompt start that falls short as in a weak arm, and a
prompt healthy response, which places the cause elsewhere, such as the cameras.

In round one the agent composed probe {b_probe_id} and ran it on {b_probe_robots} robots,
{b_probe_repeats} times each. Read from its records, after the step out and the step back:
{b_probe_summary}. The agent's reading: "{b_probe_reading}"

## Story B: the decision

The probe proposes candidates; experiments decide. An *experiment* runs ordinary task
attempts on chosen robots with and without a change, on matched starts, so *rescued* and
*broken* compare them attempt by attempt. A *latency compensation* of n control steps
makes the arm skip the first n commands of every action sequence the policy predicts after
the first, so the commands it runs are the ones meant for the moment they arrive. Round
one's experiments:

{b_experiments_table}

The agent adopted {b_round1_count} changes: {b_round1_actions}. The fleet's recorded cost
table, its estimate of operator time, puts them at {b_hours_each} operator hours,
{b_hours_total} in all. In round two the agent adopted {b_round2_actions}; round one's
changes are the fleet's final state.

## Story B: the evaluation

The *reserved evaluation panel* is a set of task attempts on days after every recorded
day, never run or seen by the agent; each state of the fleet is simulated on exactly these
attempts. With the agent's changes the fleet succeeded on {b_eval_final_k} of
{b_eval_final_n}{b_eval_final_interval}; as it was, on {b_eval_original_k} of
{b_eval_original_n}{b_eval_original_interval}. {b_eval_rescued} attempts were rescued and
{b_eval_broken} broken. By site and for the serviced robot:

{b_eval_breakdown}

## Story B: the second use

A *validated probe recipe* is the probe and its analysis saved with a learned skill, the
robots as parameters, checked when saved and audited before it runs, with one command, on
another fleet. The installation kept {b_kept_skills} learned skills from this engagement
(library digest beginning {b_library_digest}); {b2_recipe_skill} carries the probe.

A later fleet has its own remote-link site, {b2_linked_site}. The robots were chosen, and a
decision recorded from the readings, by {b2_decider}, from that fleet's public records of
its regular work, dated {b2_decision_at}. At the linked site, {b2_fixed} succeeded on
{b2_fixed_k} of {b2_fixed_n}{b2_fixed_interval} and {b2_upstream} on {b2_upstream_k} of
{b2_upstream_n}{b2_upstream_interval}. Outside it, the weakest robot was
{b2_lowest_robot}, {b2_lowest_k} of {b2_lowest_n}{b2_lowest_interval}; {b2_reference}, the
reference, succeeded on {b2_reference_k} of {b2_reference_n}{b2_reference_interval}. The
recipe run finished with the recorded outcome {b2_outcome}. After the step out and the step
back, {b2_fixed}'s hand first moved after {b2_fixed_first_move} control steps and
{b2_upstream}'s after {b2_upstream_first_move}; {b2_backlog}'s after
{b2_backlog_first_move}, like the reference's {b2_reference_first_move}.

The skill's rule: "{b2_rule}" The recipe tells the two cases apart: on the first fleet the
isolated failing robot's delay grew from step to step, and the answer was a controller
service; on the later fleet the failing robot's hand moves promptly, so the rule points to
its images and change log. The recorded decision's next actions: "{b2_next_r06}" and
"{b2_next_s3}"

## How this was recorded

Nothing in the film is staged or re-shot; every clip is a recorded run, listed below. A
*verified re-render* re-executes a run's recorded actions at a higher resolution and is
used only if every step's physics state and the final delivery check match the original,
which is kept. The cut uses {rerenders} re-renders; {rerenders_accepted} matched on all
{rerender_steps} compared steps.

The agent was never given the answer. It could read the requested goals, the definition of
delivered and the order rule, and could not change them. The release starts, the
evaluation panel, the host's earlier measurements and the causes the host put into the
fleets were withheld from it.

Three attempts at job one are not shown, the first, second and fourth: the host stopped
the first to change the request (its record holds {run1_sessions} session), and the second
and fourth ended at the launch audit, which took an ordinary English word in the agent's
own database query for a key of a hidden record (the second after {run2_sessions}
sessions). In story B the loop driver stopped at the freeze of the learned skills on a
labelling problem in its launch; the host froze the skills and evaluated both states with
the driver's own functions, changing no decision of the agent.

Counts are literal. A count resting on at least 20 runs carries a 95% Wilson interval, a
standard range for a proportion; versions run on the same starts are compared start by
start as rescued and broken.

These are counts from tens to hundreds of runs, in one simulator.

## Every program run

Every program run on development starts, with its deliveries, as its episode records hold
them.

Job one, first shift:

{s1_programs_table}

Job one, second shift:

{s2_programs_table}

Job one, third shift:

{s3_programs_table}

Job two, first shift:

{j2s1_programs_table}

Job two, second shift:

{j2s2_programs_table}

Job two, third shift:

{j2s3_programs_table}

## Provenance

The records this page is bound to:

{provenance_records}

Every number on this page, with its source:

{provenance_numbers}

Every clip in the film:

{provenance_clips}
