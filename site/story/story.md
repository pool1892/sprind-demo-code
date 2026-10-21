---
title: Can an AI agent engineer a robot?
subtitle: Two experiments in simulation
kicker: Simulation evidence
healthy: r05 r07 r08
job_r1: Job one: bowl on the cabinet, drawer open
job_r2: Job two: plate at the stove, bowl on the cabinet
label.r2.j0006: bowl, let the placement finish, home, plate
label.r2.j0010: bowl, let the placement finish, home, plate, home
label.r2.j0011: plate push stopped early, home, plate again, bowl
label.r2.j0012: plate push stopped early, hand slid sideways, plate again, bowl
label.r2.j0016: cheese lifted, then set aside
label.r2.j0017: cheese loaded, the instruction names the loaded bowl
label.r2.j0018: cheese loaded, the instruction asks for a grasp at the rim
glance.r1: Experiment one · job one | Bowl on the cabinet, / drawer open | locked exam layouts completed
glance.r2: Experiment one · job two | Plate at the stove, / bowl on the cabinet | locked exam layouts completed, / cream cheese in the plate's way
glance.fleet: Experiment two · fleet | Problems planted in {b_robots} arms / at {b_sites} sites | success on {b_eval_final_n} attempts / the agent never saw
glance.marks.jobs: plain chain of skills | the agent's job
glance.marks.fleet: as the fleet was | with the agent's {b_round1_count} changes
glance.legend: completed | failed: one dot per locked exam layout | share of attempts that succeeded
---

<!--
The public page of the demo. It tells the
film's story (site/explainer/script.md, script v4) in its own, readable form. No number is
typed here: every number is a placeholder filled at build time from the film's bound data
(fleetlab.film.story; the values are listed in the built site's story-values.json). Blocks
between `::: figure` and `:::` become charts drawn from the same data; their first line is
the title, the rest the caption. Words the page must not use fail the build.
-->

**The question.** Can an AI agent do a robot's engineering on its own: find out what needs to
change, change it, and prove that it works?

**The evidence.** In two experiments in simulation, it did, with the robot's model unchanged
and no person involved after setup. Two new jobs built from the robot's existing skills
completed {r1_delivered} of {r1_release_n} and {r2_delivered} of {r2_release_n} unseen
layouts, against {r1_naive_delivered} and {r2_naive_delivered} for the skills simply chained.
In a fleet with planted faults, the agent built its own instrument, found all three problems
and lifted success on {b_eval_final_n} unseen attempts from {b_eval_original_pct} to
{b_eval_final_pct}.

**Why it matters.** Turning robot skills into dependable work takes engineering, and done by
specialists, it adds cost and time when products and rules change. Here, agents did that
engineering. Mayfly is building installations that take on new
work themselves, starting with fulfilment providers whose assortments and packing rules keep
changing.

## The results at a glance {#results data-toc="Results"}

In both experiments the agent did the engineering itself, and every result was measured on
cases it had never seen.

::: figure glance r1 r2 fleet
The results, before and after the agent's work
Each job: the plain chain of skills against the agent's job, on the same locked exam layouts.
The fleet: its success on {b_eval_final_n} attempts the agent never saw, as it was and with the
agent's changes.
:::

<ul class="conditions">
<li><strong>Same robot model throughout.</strong> X-VLA, an open vision-language-action model, unchanged in every run.</li>
<li><strong>Judged on unseen cases.</strong> Exam layouts and fleet attempts stayed locked away until the agent's work was final.</li>
<li><strong>No person in the loop.</strong> After setup, the agent chose every experiment and wrote every solution.</li>
<li><strong>Request to proven job in minutes.</strong> On the prepared bench, from the request to a job proven on the locked exam: {r1_wall_minutes} minutes for the first job, {r2_wall_minutes} for the second.</li>
</ul>

<p class="pointers">Every count above comes with its source: <a href="record/index.html">the jobs record</a> · <a href="fleet-record.html">the fleet record</a> · <a href="story-values.json">every number on this page</a> · <a href="https://github.com/pool1892/sprind-demo-code">the code on GitHub</a></p>

## Why it matters: skills are not jobs {#why data-toc="Why it matters"}

Robot models are learning skills fast, but a business needs whole jobs done reliably in its
own setting, and its jobs keep changing. Turning skills into reliable work is engineering, a
loop:

<ol class="loop">
<li>Watch the failures</li>
<li>Find the cause</li>
<li>Change the system</li>
<li>Prove it</li>
</ol>

Today specialists run that loop by hand, job by job and change after change, which adds cost
and time whenever the work changes. Mayfly hands the loop to agents. Before building that on real
robots, we tested the core question in simulation: can an agent find out on its own what needs
to change, change it, and prove that it works?

## How we tested it {#method data-toc="How we tested it"}

A bench where every failure can be seen and measured, a robot model nobody touched, and an
agent that never sees the answers.

### The bench

A physics simulator, MuJoCo, runs a kitchen scene from LIBERO, a public robotics benchmark:
every start can be repeated exactly, every motion measured, and the failures are easy to
follow. The robot runs X-VLA, an open vision-language-action model that turns camera images and
a written instruction into motion. We kept it fixed, so every gain comes from the engineering
around it.

### The sandbox

Two AI models play different parts. The engineering agent reads the records, writes programs
and chooses experiments, in a sandbox; X-VLA, the robot's model, executes the skills,
unchanged. The agent works from the evidence; the grading stays out of its reach.

<div class="sandbox">
<div class="sb sb-sees"><h4>Sees</h4><ul>
<li>camera frames from its trials</li>
<li>the simulator's measurements: where the hand is, where each object is, what touches what</li>
<li>in the fleet, an operator's records: results, telemetry, site notes, change and maintenance logs</li>
</ul></div>
<div class="sb sb-does"><h4>Does</h4><ul>
<li>writes its own checks over the records</li>
<li>builds programs from the robot's skills</li>
<li>runs experiments and test motions</li>
<li>saves what works in the installation's library</li>
</ul></div>
<div class="sb sb-locked"><h4>Fixed boundaries</h4><ul>
<li>hidden from it: the problems we planted</li>
<li>locked away until its work is final: the exam layouts and the reserved attempts</li>
<li>fixed: the grading rules, which it knows, and the robot model's weights</li>
</ul></div>
</div>

We built the bench, wrote the requests and fixed the grading. After that, no person took part.

### A fair test

Each try starts from a stored arrangement of the table, a layout. The agent compares its
programs on practice layouts; {r1_release_n} more stay locked away for a final exam, and the
bar is set before any program runs. The plain chain, the skills one after the other, runs on
the same {r1_release_n}: that is the baseline. In the planned hardware programme, agents will
practise on simulated copies of the station and qualify their solutions on the real robots.

## Experiment one: a new job from skills the robot already has {#experiment-one data-toc="Experiment one"}

Twice, chaining the robot's skills failed the job. Twice, the agent found the physical cause,
changed how the job is done, and passed the locked exam: {r1_delivered} of {r1_release_n},
then {r2_delivered} of {r2_release_n}.

### Job one: the bowl on the cabinet, the drawer open

The request, as a shift lead would write it: bowl on top of the cabinet, middle drawer open,
everything at rest, hand clear. The bar: the whole job completed without a failure on at least
{r1_release_k} of {r1_release_n} fresh layouts.

Chained plainly, the job falls apart: the drawer skill starts the moment the bowl counts as
placed, while the hand still holds the bowl, and drags it off. The agent wrote a check for
exactly this hand-off: in all {r1_plain_n} first runs, the next skill began while the bowl was
still touched and moving.

Then it changed one thing at a time. Opening the fingers first: {r1_open_only_k} of
{r1_open_only_n}; open fingers still catch the bowl. Going home first: {r1_home_only_k} of
{r1_home_only_n}; the way home snags it. Letting the placement finish, so the hand lets go and
lifts, then going home: {r1_finish_home_k} of {r1_finish_home_n}, and {r1_confirm_k} of
{r1_confirm_n} on practice layouts it had never used.

::: figure notebook r1
The agent's lab notebook, job one
Every program the agent ran, in order; each dot is one practice layout. The last row is the
confirmation on layouts it had never used.
:::

### Job two: the plate, the bowl, and the cream cheese in the way

The second request is harder: the plate at the front of the stove, the bowl on top of the
cabinet, and a pack of cream cheese in the plate's way. Chained plainly, the job succeeds on
{r2_naive_delivered} of {r2_release_n} exam layouts.

Starting from the check it saved in job one, the agent tried {r2_programs} programs. Loading
the cheese into the bowl cleared the plate's way, but the loaded bowl would not carry:
{r2_loaded_k} of {r2_loaded_n}. Fixing one problem had created the next. Its answer: set the
cheese aside, push the plate, then place the empty bowl. {r2_relocate_k} of {r2_relocate_n} on
practice, {r2_confirm_k} of {r2_confirm_n} on layouts it had never used.

::: figure notebook r2 mute=other
The agent's lab notebook, job two
The plain chain, the cheese loaded into the bowl, the cheese set aside, and the confirmation.
:::

### The final exam

Then the locked layouts opened.

::: figure exam r1 r2
Plain chain against the agent's job, layout by layout
One dot per locked layout, the same layout in the same column. Filled: the whole job
completed without a failure.
:::

From request to proven job, the agent took {r1_wall_minutes} and {r2_wall_minutes} minutes.
Each job and its check went into the installation's library for the next request.

## Experiment two: problems planted in a fleet {#experiment-two data-toc="Experiment two"}

When working robots start to fail, can the agent find out why? It told look-alike problems
apart with an instrument it built itself, tested every repair, and lifted the fleet from
{b_eval_original_pct} to {b_eval_final_pct} on attempts it had never seen.

We built a fleet of {b_robots} simulated arms at {b_sites} sites, doing pick-and-place work,
and planted problems in it:

- **At {b_link_site},** the model that drives the robots runs in the customer's distant data
  centre, and the line adds a delay. The customer requires that link, so it stays.
- **At {b_overload_site},** one robot's controller is overloaded and falls behind.
- **At {b_update_site},** a software update crops part of every camera image. The same day, a
  camera's calibration record is rewritten, which changes nothing: our false lead.

The agent is not told the causes. It sees what an operator sees: results, telemetry, the site
notes (one says the link at {b_link_site} is the customer's requirement), change and
maintenance logs, and the actions an operator can take.

### What the agent sees

The robots at {b_link_site} trail their commands and succeed far less often; robot
{b_overload_robot} at {b_overload_site} trails the same way. The robots at {b_update_site} fall
short of the healthy robots, and the change log shows two changes there on one day. Same
symptom, same cause? Which change is to blame?

::: figure fleet
The regular work, robot by robot
Successes per robot before the agent's work; the line marks the healthy robots,
{b_healthy_robots}: {b_healthy_k} of {b_healthy_n}. Intervals and sources are in the
[fleet record](fleet-record.html).
:::

### The agent builds an instrument

The task records cannot tell the first two apart, so the agent built an instrument: a small
test motion with the robot's model switched off, timing how each arm follows a command. At
{b_link_site} the hand starts late by the same amount each way, {b_probe_r01_out_s} seconds: a
delay on the line. Robot {b_overload_robot} starts nearly on time, {b_probe_r06_out_s} seconds,
then falls behind, {b_probe_r06_back_s} seconds on the way back: its controller cannot keep up.
They look alike in the task records but need different repairs.

::: figure probe r01="the same wait out and back: a delay on the line" {b_overload_robot}="on time out, far behind back: a backlog" r08="a healthy robot, for comparison"
The agent's test motion, as recorded
The commanded hand position and the measured one; the shading marks the wait before the hand
moves. One control step is {b_step_ms} milliseconds.
:::

### Every repair tested on its own

Each repair is tested on the same attempts with and without it before it is adopted. The link
at {b_link_site} has to stay, so the agent has the robots skip commands that would be out of
date by the time they arrive: a latency compensation, tested at {b_comp_settings} settings, the
best one kept. The agent selects the catalogue's simulated controller service for
{b_overload_robot}. At {b_update_site} it undoes each same-day change in a separate test: the
software rollback brings the site back, the calibration rollback shows no clear effect, so the
calibration stays as it is.

::: table repairs
The tested repairs
Each change against the same attempts without it. Rescued: failed without the change and
succeeded with it; broken: the reverse. Intervals and sources are in the
[fleet record](fleet-record.html).
:::

### The test that counts

Then {b_eval_final_n} attempts the agent never saw. As the fleet was, {b_eval_original_k}
succeeded; with the agent's {b_round1_count} changes, {b_eval_final_k}: from
{b_eval_original_pct} to {b_eval_final_pct}. Against what we planted: every problem found, each
with a tested repair, and the false lead left alone.

::: figure fleet-result
{b_eval_final_n} attempts the agent never saw
Each square is one attempt, in the same place in both grids, sorted by outcome.
:::

## What we learned {#learned data-toc="What we learned"}

The agent worked like a good engineer: it read the evidence, found the physical cause, changed
one thing at a time, built a measurement when the records were not enough, and kept what
worked, the two jobs, their checks and its test motion, in the installation's library.

For the second job, it tested and extended the check it had saved from the first. In the fleet,
it built a test motion when the task records could not tell two faults apart. That is what Mayfly
builds: the measurements, experiment interfaces and qualification that let agents engineer an
installation, so that every tested job and diagnostic becomes a starting point for the next
requirement.

## What comes next {#next data-toc="Next"}

Next, real robots: a complete packing operation on two mobile two-armed robots from different
makers, turning picked items into correct, protected parcels ready for dispatch. An operator
brings an unfamiliar item or a new packing rule, and the agents develop the change: synchronized
video, motion and force records explain the failures, repeated orders give cheap new attempts,
and the safety functions stay outside their reach. Frozen versions then face unseen orders in
separate autonomous tests, without teleoperation, retraining or manual changes between trials;
scans, weight checks and independent parcel audits test the result. The measure: all the human
work and time each new requirement needs, at fixed parcel quality, throughput and reliability.

## What it means {#what-it-means data-toc="What it means"}

<div class="so-what">
<p>What stands between a robot's skills and paid work is engineering, which specialists do by
hand today. Here an agent did that engineering from start to finish and proved it on cases it
had never seen.</p>
<p>Our ambition is the economics that follow: smaller installations that recover their setup
costs, shorter runs that carry their changeover costs, and customers who buy the result while
Mayfly is responsible for delivering it.</p>
<p class="so-what-line">That is what Mayfly is building: robot installations where taking on
new work is part of the product, because agents do the engineering.</p>
</div>

## The evidence {#evidence data-toc="Evidence"}

These experiments are newer than the simulation evidence attached to our application: they add
a diagnostic instrument the agent built itself, which that evidence had not yet shown. Every
clip in the film is a recorded run, and every count on this page comes from the run records.

<table class="results">
<thead><tr><th scope="col">Result</th><th scope="col">Before</th><th scope="col">After the agent's work</th><th scope="col">Rescued · broken</th></tr></thead>
<tbody>
<tr><th scope="row">Job one, final exam</th><td data-label="before">plain chain: <strong>{r1_naive_delivered} of {r1_release_n}</strong><br><span class="ci">{r1_naive_pct} ({r1_naive_low_pct} to {r1_naive_high_pct})</span></td><td data-label="after"><strong>{r1_delivered} of {r1_release_n}</strong><br><span class="ci">{r1_release_pct} ({r1_release_low_pct} to {r1_release_high_pct})</span></td><td data-label="rescued · broken">{r1_rescued} · {r1_broken}</td></tr>
<tr><th scope="row">Job two, final exam</th><td data-label="before">plain chain: <strong>{r2_naive_delivered} of {r2_release_n}</strong><br><span class="ci">{r2_naive_pct} ({r2_naive_low_pct} to {r2_naive_high_pct})</span></td><td data-label="after"><strong>{r2_delivered} of {r2_release_n}</strong><br><span class="ci">{r2_release_pct} ({r2_release_low_pct} to {r2_release_high_pct})</span></td><td data-label="rescued · broken">{r2_rescued} · {r2_broken}</td></tr>
<tr><th scope="row">Fleet, reserved attempts</th><td data-label="before">as it was: <strong>{b_eval_original_k} of {b_eval_original_n}</strong><br><span class="ci">{b_eval_original_pct} ({b_eval_original_low} to {b_eval_original_high})</span></td><td data-label="after"><strong>{b_eval_final_k} of {b_eval_final_n}</strong><br><span class="ci">{b_eval_final_pct} ({b_eval_final_low} to {b_eval_final_high})</span></td><td data-label="rescued · broken">{b_eval_rescued} · {b_eval_broken}</td></tr>
</tbody>
</table>

<p class="table-note">Under each count: its share, and in brackets its {ci_level} Wilson interval.</p>

<div class="evidence-links">
<a class="evidence-link" href="record/index.html"><strong>The jobs record</strong><span>Every program, count and quotation behind the two jobs, each with its source.</span></a>
<a class="evidence-link" href="fleet-record.html"><strong>The fleet record</strong><span>The regular work robot by robot, the test motion's waits, every tested repair and the evaluation, each count with its interval and source.</span></a>
<a class="evidence-link" href="sources.html"><strong>Clips and provenance</strong><span>Every clip in the film by its recorded run, every file with its digest.</span></a>
<a class="evidence-link" href="https://github.com/pool1892/sprind-demo-code"><strong>The code</strong><span>github.com/<wbr>pool1892/<wbr>sprind-demo-code</span></a>
</div>

<p class="method">How this was recorded: LIBERO household tasks on MuJoCo physics; robot
model X-VLA, its weights unchanged in every run; engineering agent {r1_agent_model}. Each job
was examined on {r1_release_n} locked layouts and the fleet on {b_eval_final_n} attempts, all in
simulation.</p>
