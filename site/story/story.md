---
title: Can an AI agent engineer a robot?
subtitle: Two experiments in simulation: making new jobs work, and finding faults in a fleet
kicker: Simulation evidence
teaser: A robot can put a bowl on a cabinet and open a drawer, yet asked to do both, it drags the bowl back off. Finding out why and fixing it is engineering. In two experiments in simulation, an AI agent did that engineering on its own.
healthy: r05 r07 r08
job_r1: Job one: put the bowl on the cabinet, then open the drawer
job_r2: Job two: push the plate to the stove, put the bowl on the cabinet
label.r1.j0001: straight to the drawer (skills in sequence)
label.r2.j0001: plate, then bowl (skills in sequence)
label.r2.j0006: bowl, let the placement finish, home, plate
label.r2.j0010: bowl, let the placement finish, home, plate, home
label.r2.j0011: plate push stopped early, home, plate again, bowl
label.r2.j0012: plate push stopped early, hand slid sideways, plate again, bowl
label.r2.j0016: cheese lifted, then set aside
label.r2.j0017: cheese loaded, the instruction names the loaded bowl
label.r2.j0018: cheese loaded, the instruction asks for a grasp at the rim
glance.r1: Experiment one · job one | Put the bowl on the cabinet, / then open the drawer | final tests passed
glance.r2: Experiment one · job two | Push the plate to the stove, / put the bowl on the cabinet | final tests passed, / cream cheese in the plate's way
glance.fleet: Experiment two · fleet | Find the faults planted in / {b_robots} robot arms at {b_sites} sites | of {b_eval_final_n} final test attempts / succeeded
glance.marks.jobs: skills in sequence | the agent's program
glance.marks.fleet: before the repairs | after the agent's {b_round1_count} repairs
glance.legend: test passed | test failed: one dot per final test | share of attempts that succeeded
---

<!--
The public page of the demo. It tells the
film's story (site/explainer/script.md, script v4) in its own, readable form. No number is
typed here: every number is a placeholder filled at build time from the film's bound data
(fleetlab.film.story; the values are listed in the built site's story-values.json). Blocks
between `::: figure` and `:::` become charts drawn from the same data; their first line is
the title, the rest the caption. Words the page must not use fail the build. Each task comes
before its score, every number names what it counts, and "final test" means the held-out cases.
-->

**The question.** Can an AI agent do a robot's engineering on its own: find out what needs to
change, change it, and prove that it works?

**The evidence.** In two experiments in simulation, it did. The agent made two new jobs work
from skills the robot already had: both passed every one of their final tests. And it found
all three faults we planted in a fleet of robots, building its own measurement instrument when
the records were not enough; its repairs raised the fleet's success rate from
{b_eval_original_pct} to {b_eval_final_pct}. After setup, no person took part, the robot's model
stayed unchanged, and every final test used cases the agent had never seen.

**Why it matters.** Turning robot skills into dependable work takes engineering; done by
specialists, it adds cost and time when products and rules change. Here, agents did that
engineering. Mayfly is building robot installations that do it themselves, so that customers
can change the work without starting a new integration project. We start with fulfilment
providers whose assortments and packing rules keep changing.

## The results at a glance {#results data-toc="Results"}

In both experiments the agent did the engineering itself. Then we tested its finished work on
cases it had never seen: these are the final results.

::: figure glance r1 r2 fleet
The final results, before and after the agent's work
Each job: the robot's skills run one after another, then the agent's program, on the same
{r1_release_n} final tests. The fleet: before and after the agent's repairs, on the same
{b_eval_final_n} final test attempts.
:::

<ul class="conditions">
<li><strong>The robot model stayed the same.</strong> X-VLA, an open vision-language-action model, with its weights unchanged; the agent improved the programs and the system around it.</li>
<li><strong>Judged on unseen cases.</strong> The final test layouts and the fleet's final test attempts stayed locked away until the agent's work was final.</li>
<li><strong>No person in the loop.</strong> People prepared the test environment, the requests and the grading; after that, the agent chose every experiment and wrote every solution.</li>
<li><strong>Request to proven job in minutes.</strong> On the prepared test environment, from the request to a job proven on its final test: {r1_wall_minutes} minutes for the first job, {r2_wall_minutes} for the second.</li>
</ul>

<p class="pointers">The evidence behind every number: <a href="record/index.html">the two jobs, program by program</a> · <a href="fleet-record.html">the fleet investigation</a> · <a href="story-values.json">all numbers (JSON)</a> · <a href="https://github.com/pool1892/sprind-demo-code">the code on GitHub</a></p>

## Why it matters: skills are not jobs {#why data-toc="Why it matters"}

Robot models are learning skills fast: placing an object, opening a drawer. But a business needs
whole jobs done reliably in its own setting, with its own products, equipment and rules, and its
jobs keep changing. Turning skills into reliable work is engineering, a loop:

<ol class="loop">
<li>Watch the failures</li>
<li>Find the cause</li>
<li>Change the system</li>
<li>Prove it</li>
</ol>

Today specialists run that loop by hand, job by job and change after change, which adds cost
and time whenever the work changes. Mayfly hands the loop to agents. Before building that on
real robots, we tested the core question in simulation: can an agent find out on its own what
needs to change, change it, and prove that it works?

## How we tested it {#method data-toc="How we tested it"}

A test environment where every failure can be seen and measured, a robot model nobody touched,
and an agent that never sees the answers.

### The test environment

MuJoCo, a physics simulator, runs kitchen tasks from LIBERO, a public robotics benchmark: every
starting layout can be repeated exactly, every motion measured, and the failures are easy to
see.

X-VLA, an open vision-language-action model, controls the robot, turning camera images and a
written instruction into motion. Its weights stayed fixed throughout, so every gain comes from
the engineering around the model.

### What the engineering agent could do

Two AI models have different jobs. X-VLA executes the robot's skills. A separate engineering
agent inspects the results, writes programs and chooses experiments. It works in a restricted
environment: it can investigate and change the system through the tools it is given, but it
cannot change the grading or see the final tests.

<div class="sandbox">
<div class="sb sb-sees"><h4>What it could inspect</h4><ul>
<li>camera frames from its trials</li>
<li>the simulator's measurements of the hand, the objects and what touches what</li>
<li>in the fleet experiment: operating results, motion and timing measurements, site notes, and change and maintenance logs</li>
</ul></div>
<div class="sb sb-does"><h4>What it could do</h4><ul>
<li>write checks to analyse the records</li>
<li>build programs from the robot's existing skills</li>
<li>run experiments and test motions</li>
<li>save programs and checks for later use</li>
</ul></div>
<div class="sb sb-locked"><h4>Fixed rules and hidden information</h4><ul>
<li>grading rules: known to the agent, and fixed</li>
<li>the robot model's weights: fixed</li>
<li>the causes of the faults we planted: hidden</li>
<li>the final tests: locked away until its work was finished</li>
</ul></div>
</div>

We built the test environment, wrote the requests and fixed the grading rules. After that, no
person took part.

### A fair test

Each trial starts from a stored arrangement of the objects on the table, called a layout. For
each job, the agent developed and compared its programs on practice layouts; another
{r1_release_n} layouts were reserved for the final test, and the pass mark was fixed before any
program ran.

On those same {r1_release_n} final layouts we ran two programs: the robot's skills run one after
another (skills in sequence), as the baseline, and the agent's finished program. The difference
is what the agent's engineering changed.

## Experiment one: making new jobs work with existing skills {#experiment-one data-toc="Experiment one"}

Twice, the robot's skills run in sequence failed the job. Twice, the agent found the physical
cause, changed how the job is done, and its program passed every final test.

### Job one: put the bowl on the cabinet, then open the drawer

The request: put the bowl on top of the cabinet, open the middle drawer, and finish with the
objects at rest and the robot's hand clear. To pass, the program had to complete the whole job
in at least {r1_release_k} of {r1_release_n} final tests.

Run one after another, the skills fail. The drawer skill starts as soon as the placement skill
reports success, but the hand is still holding the bowl, so the next motion drags it off the
cabinet. The agent wrote a check for this hand-off: in all {r1_plain_n} first trials, the drawer
skill began while the hand was still touching the bowl and the bowl was still moving.

Then it tested ways to finish the placement before opening the drawer, one change at a time.
Opening the fingers first: {r1_open_only_k} of {r1_open_only_n}; the open fingers still caught
the bowl. Returning the arm to its home position first: {r1_home_only_k} of {r1_home_only_n};
the hand snagged the bowl on the way. The program that worked lets the placement finish, so the
hand releases the bowl and lifts away, then returns the arm home before opening the drawer:
{r1_finish_home_k} of {r1_finish_home_n} in practice, and {r1_confirm_k} of {r1_confirm_n} on
further practice layouts it had never used.

::: figure notebook r1
The agent's lab notebook, job one
Every program the agent tested for this job, in the order it tested them; each dot is one
practice layout, filled when the whole job was completed. "Home" is the arm's home position.
The last row checks the chosen program on further practice layouts; the final test follows
below.
:::

### Job two: the plate, the bowl, and the cream cheese in the way

The second job is harder: push the plate to the front of the stove and put the bowl on top of the
cabinet, with a pack of cream cheese in the plate's path. Run one after another, the skills
completed the job in {r2_naive_delivered} of {r2_release_n} final tests.

The agent reused and extended the check from job one, then tried {r2_programs} programs. Putting
the cream cheese into the bowl cleared the plate's path, but the robot struggled to move the
loaded bowl: the whole job succeeded in {r2_loaded_k} of {r2_loaded_n} practice trials. Fixing
one problem had created the next. The program that worked moves the cream cheese aside, pushes
the plate into place, then places the empty bowl: {r2_relocate_k} of {r2_relocate_n} in
practice, and {r2_confirm_k} of {r2_confirm_n} on further practice layouts it had never used.

::: figure notebook r2 mute=other
The agent's lab notebook, job two
Skills in sequence, the cream cheese loaded into the bowl, the cream cheese moved aside, and the
check on further practice layouts; the full list holds all {r2_programs}
programs.
:::

### The final test

Once each program was finished, we ran it on the {r1_release_n} layouts reserved for that job,
which the agent had not seen. Skills in sequence and the agent's program faced the same
layouts.

::: figure exam r1 r2
Skills in sequence against the agent's program, final test by final test
Each column is one final test layout, the same for both programs; a filled dot means the whole
job was completed.
:::

From request to proven job, the agent took {r1_wall_minutes} minutes for job one and
{r2_wall_minutes} for job two. Each program and its check went into the installation's library
for the next request.

## Experiment two: finding faults in a fleet {#experiment-two data-toc="Experiment two"}

When robots that worked start to fail, can an agent find out why? It did: it told two
look-alike problems apart with an instrument it built itself, tested every repair, and raised
the fleet's success rate from {b_eval_original_pct} to {b_eval_final_pct} on final test attempts
it had never seen.

We planted three problems in a fleet of {b_robots} simulated robot arms at {b_sites} sites, all
doing pick-and-place work, and added a misleading maintenance entry, to see whether the agent
could tell a cause from a coincidence:

- **Delayed commands at {b_link_site}.** The model that drives the robots runs in the
  customer's distant data centre, and the network connection delays its commands. The customer
  requires that connection, so it stays.
- **An overloaded controller at {b_overload_site}.** One robot's controller cannot keep up with
  its commands.
- **Cropped camera images at {b_update_site}.** A software update crops the images the robot
  model sees. The same day, a camera's calibration record is rewritten, which changes nothing:
  the false lead.

The agent is not told the causes. It sees what an operator sees: operating results, motion and
timing measurements, site notes (one says the connection at {b_link_site} must stay), change and
maintenance logs, and the actions an operator can take.

### The same symptom, different causes?

The robots at {b_link_site} moved later than commanded and succeeded far less often; robot
{b_overload_robot} at {b_overload_site} showed a similar lag. The operating records could not
tell whether the two delays had the same cause. At {b_update_site}, success had fallen after two
changes on the same day: which one mattered?

::: figure fleet
The fleet before the investigation, robot by robot
Successes per robot in the recorded operating work, before the agent's changes; the line marks
the combined rate of the healthy robots, {b_healthy_robots}: {b_healthy_k} of {b_healthy_n}.
Intervals and sources are in the [fleet record](fleet-record.html).
:::

### The agent builds an instrument

The operating records cannot tell the two lags apart, so the agent built an instrument: a short
test motion, sent directly without X-VLA, that measures how long the arm takes to follow a small
movement out and back. At {b_link_site} the hand started {b_probe_r01_out_s} seconds late in both directions: a
delay on the connection. Robot {b_overload_robot} started after {b_probe_r06_out_s} seconds on
the way out but fell {b_probe_r06_back_s} seconds behind on the way back: its controller could
not keep up. In ordinary work the two failures looked alike; the new measurement told them apart
and pointed to different repairs.

::: figure probe r01="the same wait out and back: a delay on the connection" {b_overload_robot}="on time out, far behind back: a backlog" r08="a healthy robot, for comparison"
The agent's instrument: its test motion, as recorded
The commanded hand position and the measured one; the shading marks the wait before the hand
moves. One control step is {b_step_ms} milliseconds.
:::

### Every repair tested on its own

The agent tested every repair before adopting it, on the same tasks and starting layouts with
and without the change.

- **The delayed commands.** The connection at {b_link_site} had to stay, so the agent
  compensated for the delay: the robots skip commands that would be out of date by the time
  they arrive. It tested {b_comp_settings} settings and kept the best.
- **The overloaded controller.** The agent tested the simulated maintenance action for robot
  {b_overload_robot}'s controller: with it, the robot completed every test attempt, so the agent
  adopted it.
- **The two changes on one day.** At {b_update_site} the agent undid each change in a separate
  test. Rolling back the software update restored the site; rolling back the calibration showed
  no clear effect, so the calibration stayed as it was.

::: table repairs
The tested repairs
Each change against the same tasks and starting layouts without it. Improved: failed without
the change, succeeded with it; worsened: the reverse. Intervals and sources are in the
[fleet record](fleet-record.html).
:::

### The test that counts: the final fleet test

Then we tested the fleet before and after the agent's {b_round1_count} repairs on the same
{b_eval_final_n} final test attempts, which the agent had not seen. Before the repairs,
{b_eval_original_k} succeeded; after them, {b_eval_final_k}: from {b_eval_original_pct} to
{b_eval_final_pct}. Against what we planted: every problem found, each with a tested repair,
and the false lead left alone.

::: figure fleet-result
{b_eval_final_n} final test attempts, before and after the repairs
Each square is one attempt, in the same place in both grids, sorted by outcome.
:::

## What we learned {#learned data-toc="What we learned"}

The agent worked like a good engineer: it read the evidence, found the physical causes, changed
one thing at a time, and proved its solutions on final tests it had never seen. When the
operating records could not tell two faults apart, it built its own measurement.

It also left tools behind. The second job reused and extended a check from the first, and the
two finished programs, their checks and its test motion went into the installation's
library. That is what Mayfly builds: the tools that let agents measure failures, run experiments
and qualify their changes on an installation, so that every tested job and diagnostic becomes a
starting point for the next requirement.

## What comes next {#next data-toc="Next"}

Next, we take the approach to real robots: two mobile robots with two arms each, from different
makers, running a complete packing operation that turns picked items into correct, protected
parcels ready for dispatch.

An operator brings an unfamiliar item or a new packing rule, and the agents develop the change:
synchronized video, motion and force records explain the failures, and repeated orders give
cheap new attempts. The agents practise on simulated copies of the station and qualify their
solutions on the real robots; the safety functions stay outside their reach.

Each version is then frozen and tested on unseen orders in separate autonomous tests, without
teleoperation, retraining or manual changes between trials; scans, weight checks and
independent parcel audits check the result. The measure: all the human work and time each new
requirement needs, at fixed parcel quality, throughput and reliability.

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

These experiments extend our earlier simulation evidence with a diagnostic instrument the agent
built itself. Every clip in the film is a recorded run, and every count on this page comes
from the run records.

<table class="results">
<thead><tr><th scope="col">Result</th><th scope="col">Before</th><th scope="col">After the agent's work</th><th scope="col">Improved · worsened</th></tr></thead>
<tbody>
<tr><th scope="row">Job one, final test</th><td data-label="before">skills in sequence: <strong>{r1_naive_delivered} of {r1_release_n}</strong><br><span class="ci">{r1_naive_pct} ({r1_naive_low_pct} to {r1_naive_high_pct})</span></td><td data-label="after"><strong>{r1_delivered} of {r1_release_n}</strong><br><span class="ci">{r1_release_pct} ({r1_release_low_pct} to {r1_release_high_pct})</span></td><td data-label="improved · worsened">{r1_rescued} · {r1_broken}</td></tr>
<tr><th scope="row">Job two, final test</th><td data-label="before">skills in sequence: <strong>{r2_naive_delivered} of {r2_release_n}</strong><br><span class="ci">{r2_naive_pct} ({r2_naive_low_pct} to {r2_naive_high_pct})</span></td><td data-label="after"><strong>{r2_delivered} of {r2_release_n}</strong><br><span class="ci">{r2_release_pct} ({r2_release_low_pct} to {r2_release_high_pct})</span></td><td data-label="improved · worsened">{r2_rescued} · {r2_broken}</td></tr>
<tr><th scope="row">Fleet, final test</th><td data-label="before">before the repairs: <strong>{b_eval_original_k} of {b_eval_original_n}</strong><br><span class="ci">{b_eval_original_pct} ({b_eval_original_low} to {b_eval_original_high})</span></td><td data-label="after"><strong>{b_eval_final_k} of {b_eval_final_n}</strong><br><span class="ci">{b_eval_final_pct} ({b_eval_final_low} to {b_eval_final_high})</span></td><td data-label="improved · worsened">{b_eval_rescued} · {b_eval_broken}</td></tr>
</tbody>
</table>

<p class="table-note">Under each count: its share and, in brackets, its {ci_level} Wilson confidence interval. Improved: failed before, succeeded after; worsened: the reverse.</p>

<div class="evidence-links">
<a class="evidence-link" href="record/index.html"><strong>The two jobs, program by program</strong><span>Every program, count and quotation behind the two jobs, each with its source.</span></a>
<a class="evidence-link" href="fleet-record.html"><strong>The fleet investigation</strong><span>The fleet's work robot by robot, the motion test's timings, every tested repair and the final test, each count with its interval and source.</span></a>
<a class="evidence-link" href="sources.html"><strong>Recorded clips and sources</strong><span>Every clip in the film by its recorded run, every file with its digest.</span></a>
<a class="evidence-link" href="https://github.com/pool1892/sprind-demo-code"><strong>The code</strong><span>github.com/<wbr>pool1892/<wbr>sprind-demo-code</span></a>
</div>

<p class="method">Test environment: LIBERO household tasks in the MuJoCo physics simulator.
X-VLA executed the robot's skills with its weights unchanged; {r1_agent_model} was the
engineering agent. Each job had {r1_release_n} final test layouts and the fleet's final test
{b_eval_final_n} attempts. All results on this page are from simulation.</p>
