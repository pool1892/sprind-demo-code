# Narration: the explainer, script v4

<!--
The narration of the explainer film. One section per beat, headed
`## <beat id>: <title>`; the beat ids are the scene ids of the Remotion composition in
site/explainer/film. Everything under a heading is read aloud except HTML comments. No
number is typed into this script: every number is a placeholder {name} defined in
site/explainer/fields.json and bound from the records named in site/explainer/records.json:

    uv run --locked python -m fleetlab.film.record bind site/explainer/script.md \
        --fields site/explainer/fields.json --records site/explainer/records.json \
        --out ~/.cache/robots-probes/demo/explainer/script-v4 --final

Script v3 (21 October) says why we ran this, what we wanted to show, the two experiments,
why the setup is built this way and the sandbox, story before detail. Script v4 (21 October,
evening) revises it after cut 3: no attention rings; the engineering loop as its own
graphic, in step with the voice; the robot model named as a vision-language-action model;
the bar said explicitly; practice in simulation and the exam in the real world, once on
real robots; more room for the results; a summary closing experiment one; experiment two
introducing everything planted, the false lead included, before the agent solves each
problem; the code repository on the end card.

A `purpose:` comment says what the beat must leave with the viewer, a `screen:` comment what
is on screen, a `source:` comment what each sentence rests on. Beats without narration
(title, end) carry only comments. Pace: about 155 words a minute; the whole film about 9:30.
No rings, circles or boxes drawn around objects anywhere: these scenes are calm enough
without guidance.

Acts: I why (cold-open to question), II the bench and the rules (station to rules),
III experiment one (request to r-summary), IV experiment two (b-fleet to b-result),
V what we learned (results to end).
-->

## cold-open: One job, two skills

This robot can put a bowl on a cabinet. It can open a drawer. Ask for both in one job, and
it gets it right in just {r1_naive_delivered} of {r1_release_n} fresh tries.

<!-- purpose: the hook; skills work, the combined job does not. -->
<!-- screen: as cut 3. Side camera: skill one alone (bowl onto the cabinet) ✓, skill two
alone (drawer opens) ✓, then the plain chain: the bowl lands, the arm turns to the drawer,
the bowl slides off (half speed on the fall). Freeze, the count as dots. No ring. -->
<!-- source: single skills: host screen runs single-cabinet e000102 and single-drawer
e000102, verified re-renders. Chained: job 1 plain chain on reserved start 31
(replacement-01 release e000202, verified re-render); count = the plain chain on the 20
reserved starts (receipt). -->

## title: Can an AI agent engineer a robot?

<!-- screen: "Can an AI agent engineer a robot?", subtitle "Two experiments in simulation",
Mayfly wordmark. Music hit; no voice. -->

## why: Skills are not jobs

Robot AI models are learning skills fast. But a business needs whole jobs done reliably,
in its own setting, and its jobs keep changing.

Turning skills into reliable work is engineering: watch the failures, find the cause,
change the system, prove it. Today, specialists do that by hand, for every change, and
that makes automation slow and expensive.

<!-- purpose: the problem: the engineering between a skill and a working job, again at
every change. -->
<!-- screen: two graphics in turn, never both at full size. First, small and quick, with
the first two sentences: skill tiles tick green and snap into a job; the seam cracks; change
cards (new product, new packing rule, a worn gripper) send the job back to the start.
Then, with "Turning skills into reliable work", the engineering loop takes the centre at
full size: four steps in a ring, each lighting exactly on its words (watch the failures →
find the cause → change the system → prove it; cues watch, cause, change, prove); on
"specialists", a specialist at a desk runs the loop by hand, a clock and a cost counter
running beside. -->
<!-- source: no number. -->

## question: The question

Mayfly is building robot installations whose engineering is done by AI agents. Before
building that on real robots, we tested the core question in simulation: can an agent
find out on its own what needs to change, change it, and prove that it works?

We ran two experiments, one for each kind of engineering an installation keeps needing.
The first: a new job, made from skills the robot already has. Every new product or packing
rule asks for a job like this, and today a specialist has to make it reliable. The second:
faults we planted in a fleet of robots. When robots that worked yesterday start to fail,
someone has to find out why, and repair it.

<!-- purpose: the intention and what we try to show, said once, plainly; the two
experiments named, and why these two, before either begins. Recorded with longer pauses. -->
<!-- screen: as cut 4: the specialist's seat in the loop taken by the agent glyph; the
question card, its three verbs lighting. Then the two experiment cards, one at a time, each
with its reason in one short line under it as the voice gives it: "Experiment 1 · A new job
from existing skills", then "new products and rules mean new jobs"; "Experiment 2 · Faults
planted in a fleet", then "working robots start to fail". -->
<!-- source: the demo's purpose; no number. -->

## station: The test bench

The test bench is a physics simulator with a kitchen scene from a public robotics
benchmark. In simulation, every start can be repeated exactly and every motion measured,
and the kitchen makes the failures easy to follow. The robot runs X-VLA, an open robot
vision-language-action model: it turns camera images and a written instruction into
motion. For these experiments we kept it fixed, so every gain comes from the engineering
around it.

<!-- purpose: what the bench is and why it is built this way: visible mechanics, and a fixed
robot model so the engineering's effect is measurable. -->
<!-- screen: establishing camera on the station, labels pin to bowl, plate, cream cheese,
stove, cabinet and drawers; beside it the diagram camera images + instruction → X-VLA,
"open robot vision-language-action model" (padlock, "fixed for these experiments") → arm
motion. -->
<!-- source: LIBERO goal-suite scene on MuJoCo; X-VLA weights fixed in every run
(job_run.json policy records of both investigations). -->

## agent: The engineer and its sandbox

The engineer is an AI agent in a sandbox. Like a robotics engineer at a desk, it looks
at camera frames from its trials and reads the measurements the simulator records: where
the hand is, where each object is, what touches what. It writes its own checks, builds
programs from the robot's skills and runs experiments. It knows the rules it is graded by,
and cannot change them. The faults we planted stay hidden from it, and the exam layouts
stay locked away until its solution is final. We built the bench, wrote the requests and
fixed the grading. After that, no person took part: the agent chose every experiment and
wrote every solution.

<!-- purpose: the division of responsibility; who supplied the engineering judgment; the
agent cannot see or touch the answers. -->
<!-- screen: condensed: one clean box, the agent glyph inside, and three
short groups that appear one after another with the voice and stay small: "Sees" (camera
frames, measurements), "Does" (checks, programs, experiments), "Cannot see" behind a lock
(planted faults, exam layouts); the rules as one line on the box's edge ("rules: known, not
changeable"); on the last two sentences a single line under the box, "People: built the
bench, wrote the requests, fixed the grading", which then gives way to the punchline,
large and held to the end of the beat: "After that, no person took part: the agent chose
every experiment and wrote every solution." No arrows between groups,
no extra icons, nothing that moves without a word to move it. -->
<!-- source: the job tools (get_frames, query, write_check, submit_job_program) and the
fleet tools (experiments, probes, interventions); the release starts, the evaluation panel, the host's earlier
measurements and the planted causes are withheld; the delivery check, order rule, start
allocation and release check stay with the host. -->

## rules: A fair test

Each try starts from a stored arrangement of the table: a layout. The agent compares its
programs on the same practice layouts. {r1_release_n} more stay locked away for a final
exam it never sees in advance. On those same {r1_release_n}, we also run the plain chain,
the skills simply one after the other. That is the baseline. Later, on real robots, the
practice layouts will be simulated copies of the real station, and the final exam will
take place in the real world.

<!-- purpose: define layout, practice, final exam and baseline once, with the dot legend, so
every count later reads at a glance; say how the same split carries over to real robots
(real to sim to real). -->
<!-- screen: a flick through a few practice layouts; practice tiles on the left; twenty
tiles slide into a locked vault; the plain chain's two skill tiles clip together beside the
vault. Legend that stays for the film: "● one dot = one try: green delivered, red not". On
the last sentence, two small tags: under the practice tiles "on real robots: simulated
copies of the real station"; under the vault "on real robots: the real world". -->
<!-- source: practice starts 0 to 29 (tuning 0 to 9, confirmation 10 to 29), reserved
starts 30 to 49 refused to the agent until the final run (both job runs). -->

## request: Experiment one: the request

Experiment one. The request, as a shift lead would write it: bowl on top of the cabinet,
middle drawer open, everything at rest, hand clear. The bar: the whole job completed without
a failure on at least {r1_release_k} of {r1_release_n} fresh layouts.

<!-- purpose: the composite job and its standard, set before any test, said explicitly. -->
<!-- screen: chapter card "Experiment 1 · A new job from existing skills", then the request
as recorded (verbatim: "{r1_request}"), the delivery check as ticks, the bar as 20 outlined
dots with 18 marked and the words "at least 18 of 20 fresh layouts completed without a
failure" (numbers from the film data). -->
<!-- source: request text and acceptance from replacement-01 job_run.json; the lead's
parameter fixed the bar before any program ran. -->

## r1-fail: Watch it fail

Chained plainly, the job falls apart. The drawer skill starts the moment the bowl counts as
placed. But the hand still holds the bowl, and drags it off.

<!-- purpose: the failure, visible and physical. -->
<!-- screen: as cut 3 (practice layout 2, the agent's own first runs): the switch label "✓
bowl on cabinet → next skill: open the drawer", the contact glow while the hand touches the
bowl, the fall at half speed; wrist inset. No ring on the bowl. -->
<!-- source: session-01 e000103 (practice 2), verified re-render; its recorded table. -->

## r1-measure: Measure why

The agent studies its first runs and writes a check for exactly this hand-off. In all
{r1_plain_n}, the next skill began while the bowl was still touched and moving. The placing
skill reports success before the job is done.

<!-- purpose: the diagnosis, as a measurement; the general lesson in one sentence. -->
<!-- screen: as cut 3: bowl height, hand contact and the bowl goal over time, the switch
moment marked; the check's finding as ten amber hand-off markers. -->
<!-- source: check handoff-retention-and-delivery (session 2); traces of practice 2
(session-01 e000103). -->

## r1-experiment: One change at a time

Then it changes one thing at a time, on the same practice layouts. Open the fingers first?
They still catch the bowl. Go home first? The way home snags it. What works: let the
placement finish, so the hand lets go and lifts, then go home, then open the drawer:
{r1_finish_home_k} of {r1_finish_home_n} successful jobs.

<!-- purpose: the method made visible: hypotheses, rejected alternatives, the working one. -->
<!-- screen: as cut 3, the lab notebook, one row per program, ten dots per row; the working
row's count reads "10 of 10 successful jobs" (numbers from the film data); side by side on practice layout 2: open-then-home loses the bowl, finish-then-home
keeps it; then the working program plays in full. -->
<!-- source: per-start outcomes of j0001 to j0007 (sessions 1 to 3); clips session-03
e000503 and e000603. -->

## r1-prove: The final exam

Before the exam, it confirms the program on {r1_confirm_n} practice layouts it had never
used. {r1_confirm_k} of {r1_confirm_n}.

Then the final exam, on the {r1_release_n} locked layouts. The plain chain gets
{r1_naive_delivered} right. The agent's job: {r1_delivered} of {r1_release_n}.

<!-- purpose: proof on unseen layouts; confirmation and exam clearly separate in time; room
for the result to sink in: a longer pause before each count, and the
final tiles held for several seconds after the voice ends. -->
<!-- screen: confirmation dots fill (practice, grey frame); a pause; the vault opens; twenty
tiles play at once, the plain chain (borders turn red, one green), a beat of stillness,
then the agent's job (all green), held; small: "{r1_rescued} rescued, {r1_broken} broken". -->
<!-- source: confirmation j0008 (starts 10 to 29); the receipt (starts 30 to 49), reconciled in
coordinator-verification.json; tiles: the 40 release episodes, verified re-renders. -->

## r1-keep: Kept for the next request

The job and its check go into the installation's library, ready for the next request. From
request to proven job, the agent took {r1_wall_minutes} minutes.

<!-- purpose: what persists; how long the agent's work took. -->
<!-- screen: library shelf: job station-bowl-and-open-drawer, check
handoff-retention-and-delivery; the timer. -->
<!-- source: replacement-01 library; r1_wall_minutes from job_run.json wall_seconds. -->

## r2-fail: A harder job

The second request is harder. Two things must end up in place: the plate at the front of
the stove, and the bowl on top of the cabinet. The robot has a skill for each: one pushes
the plate, the other places the bowl.

Chained plainly, the job succeeds on {r2_naive_delivered} of {r2_release_n} fresh layouts.
The trouble is a pack of cream cheese in the plate's way: the push runs into it, or the
hand grabs the cheese instead.

<!-- purpose: a different engineering problem: something no goal names is in the way; the
task described with room before the failure. -->
<!-- screen: the request card with its two goals, each lighting with its words; the two
skills as tiles ("push the plate", "place the bowl"); then hero-plate camera, recorded
plain-chain plate failures (practice 1 and 5: the plate tipping, the hand on the cheese);
the exam baseline as dots, {r2_naive_delivered} of {r2_release_n}. No rings around the
cheese: the footage shows it plainly. -->
<!-- source: plate-cabinet-01 session 1 j0001; the receipt for the plain chain. -->

## r2-bottleneck: The bottleneck moves

Starting from the check it saved in the first job, the agent tries {r2_programs} programs.
Loading the cheese into the bowl clears the way, and every plate push works. But now the
loaded bowl won't carry: the job delivers {r2_loaded_k} of {r2_loaded_n}. Fixing one
problem created the next.

<!-- purpose: reuse of the saved check; the bottleneck moves and the agent sees it. -->
<!-- screen: as cut 3. -->
<!-- source: check station-goal-and-handoff-audit (its purpose: extends the library handoff
check); the 18 programs of sessions 1 to 5 on practice starts 0 to 9; loaded j0013. -->

## r2-fix: Set the cheese aside

Its answer: pick up the cheese and set it aside. Then the plate, then the empty bowl:
{r2_relocate_k} of {r2_relocate_n}. On the final exam, {r2_delivered} of {r2_release_n},
against {r2_naive_delivered} for the plain chain.

<!-- purpose: the solution and its proof. -->
<!-- screen: as cut 3. -->
<!-- source: j0015 (session 5); confirmation j0019 {r2_confirm_k} of {r2_confirm_n}; receipt
reconciled in plate-cabinet-01/coordinator-verification.json. -->

## r-summary: What experiment one showed

What did the first experiment show? Twice, a new job failed when the robot's skills were
simply chained. Twice, the agent found the physical cause, changed how the job is done, and
proved its solution on layouts it had never seen, with no person in the loop. That
matters, because this is the engineering that makes every new job slow and expensive
today. Here, an agent did it, in {r1_wall_minutes} and {r2_wall_minutes} minutes.

<!-- purpose: close Act III with what it showed and why it matters -->
<!-- screen: calm, two cards side by side, one per job: the request in a few words, the
exam before and after as two dot rows ({r1_naive_delivered} → {r1_delivered} of
{r1_release_n}; {r2_naive_delivered} → {r2_delivered} of {r2_release_n}); under both, three
words lighting with the voice: found the cause · changed the job · proved it on unseen
layouts; then the two times, {r1_wall_minutes} and {r2_wall_minutes} minutes. -->
<!-- source: the two receipts; r1_wall_minutes and r2_wall_minutes from the job runs'
wall_seconds. -->

## b-fleet: Experiment two: planted faults

Experiment two asks a different question: when working robots start to fail, can the agent
find out why? We built a fleet of {b_robots} simulated arms at {b_sites} sites, doing
pick-and-place work, and planted problems in it. The agent is told nothing about them. It
sees what an operator sees: results, telemetry, change and maintenance logs, and the
actions an operator can take.

<!-- purpose: the second experiment's premise said aloud: planted faults, hidden from the
agent, an operator's view. -->
<!-- screen: chapter card "Experiment 2 · Faults planted in a fleet"; fleet map, three
sites of four arms; a sealed envelope "planted problems" goes into a locked drawer; the
operator console: results, telemetry, change log, maintenance log, action catalogue. -->
<!-- source: demo record sections 2.3 and 2.4 (what the fleet agent sees; planted causes
withheld). -->

## b-planted: What we planted

Here is what we planted. At the first site, the model that drives the robots runs in the
customer's distant data centre, and the line adds a delay. The customer requires that
link, so it stays. At the second site, one robot's controller is overloaded and falls
behind. At the third site, a software update crops part of every camera image before the
model sees it. On the same day, a camera's calibration record is rewritten, which changes
nothing at all. That one is our false lead.

<!-- purpose: everything planted, the false lead included, introduced together before the
agent's work begins, so no problem comes from nowhere. The viewer knows
what the agent does not. -->
<!-- screen: the drawer from b-fleet opens towards the viewer only (the agent's console
stays dark, "hidden from the agent"); the fleet map with three sites; one plain card per
site as the voice reaches it: s1 "a delay on the line to the data centre (the link stays)";
s2, robot r06 "an overloaded controller"; s3 "a software update that crops the camera
images" and, beside it in grey, "a calibration record rewritten the same day: no effect,
the false lead". No internal fault names, no world ids. -->
<!-- source: story B's planted problems (the host recipe: the remote link at s1, lasting;
r06's overloaded controller at s2; at s3 release 4.3.0 cropping a fifth of each image and a
scene calibration record rewritten the same day with no mechanism); the public site note
for s1. -->

## b-symptom: What the agent sees

The agent is not told what causes the failures. It sees the operating records. The robots
at the first site trail their commands and succeed far less often. One robot at the second
site trails just the same. The robots at the third site fail more often than the healthy
robots, and the change log shows two changes on the same day. Same symptom, same cause? And
which of the two changes is to blame?

<!-- purpose: the diagnostic problem as the agent meets it: two look-alike symptoms, and
two candidate causes at the third site. The agent does see the site note and the change
log; what it is not told is what causes the failures. -->
<!-- screen: the operator's view: success by robot in the regular work, s1's four robots
and r06 low, s3's four robots in between, the healthy robots (r05, r07, r08) high; no site
is labelled healthy (every site has a planted problem); the same "lag" badge on the s1
robots and on r06 (its label coloured, no box around it); s3's change log with its two
entries on one day. -->
<!-- source: the fleet's public site note; the regular work of days 0 to 5 (s1 174 of 288,
s2 261 of 288, s3 251 of 288, film data); the change log; demo record section 4.2. -->

## b-probe: The agent builds an instrument

The task records can't tell the first two apart. So the agent builds an instrument: a
small test motion, with the robot's model switched off, timing how each arm follows a
command. At the first site, the hand always starts late, by the same amount: a delay on
the line. The robot at the second site starts nearly on time, then falls further and
further behind: its controller can't keep up. They look alike, but they are two different
problems, and they need different repairs.

<!-- purpose: the agent creates the measurement it needs; the two curves told apart; no
slogan. -->
<!-- screen: as cut 3: the probe drawn from its recorded steps: commanded against measured
hand position for r01 (first site), r06 and a healthy arm; the fixed delay and the growing
backlog animated separately; first-move delays labelled from the records. -->
<!-- source: probe p0001, its recorded steps (first moves: r01 18 and 18 control steps every
time; r06 4, 12 and 5 out, 37, 45 and 39 back; r08 and r10 1 and 1). -->

## b-fix: Tested repairs

Each repair is tested on its own before it is adopted. The first site's link has to stay,
so the agent has the robots skip commands that would be out of date by the time they
arrive. It tests three compensation settings and keeps the best. The overloaded controller
at the second site is serviced. At the third site, it undoes each of the two changes in a
separate test. Rolling back the software update brings the site back. Rolling back the
calibration shows no clear effect. So it adopts the software rollback, and leaves the
calibration alone.

<!-- purpose: experiments decide; each planted problem met in turn, the false lead included.
The agent's own reading of the calibration test was "inconclusive";
"false lead" is our knowledge and is said only in b-planted and b-result. -->
<!-- screen: one experiment card per site, in the order of the voice, each with its dots
(numbers from the film data): s1, three compensation settings side by side (8, 17 and 26
control steps: 40, 43 and 38 of 44, against 29 of 44 without), 17 kept; s2, the controller
service on r06 (13 of 24 → 24 of 24); s3, the two rollbacks side by side: software update
(49 of 66 → 66 of 66, adopted) and calibration (6 rescued, 3 broken, not adopted). -->
<!-- source: round one's three experiments x0001 to x0003 (demo record section 4.3; film
data fleet.experiments); latency compensation skips the early commands of each predicted
action sequence (demo record section 4.3). -->

## b-result: The fleet result

Then the test that counts: {b_eval_final_n} attempts the agent never saw. As the fleet was,
{b_eval_original_k} successes. With the agent's {b_round1_count} changes, {b_eval_final_k}.
From {b_eval_original_pct} to {b_eval_final_pct}. Against what we planted: all three
problems found, each with a tested repair, and the false lead left alone.

<!-- purpose: the proof on unseen attempts, and the score against the planted truth. -->
<!-- screen: the reserved panel as 576 dots: {b_eval_original_k} → {b_eval_final_k};
"{b_eval_rescued} rescued, {b_eval_broken} broken"; the agent's forecast beside the measured
gain; the checklist against b-planted's cards: delay compensated ✓, controller serviced ✓,
software rolled back ✓, calibration left alone ✓. -->
<!-- source: the reserved panel evaluation (engagement record, state round-2 against
original); the planted causes (host recipe). -->

## results: The results

In both experiments, the agent did the engineering itself. Two new jobs: from
{r1_naive_delivered} to {r1_delivered} of {r1_release_n}, and from {r2_naive_delivered} to
{r2_delivered}. A fleet: from {b_eval_original_pct} to {b_eval_final_pct}. The robots'
vision-language-action model and their overall design stayed the same throughout.

<!-- purpose: the three results together, with the condition that makes them meaningful
(the model named precisely). -->
<!-- screen: three result cards; "Same robot model and design throughout"; "Request to
proven job: {r1_wall_minutes} and {r2_wall_minutes} minutes." -->
<!-- source: the receipts and the fleet evaluation; X-VLA in every job run and in the
fleet's setup (policy XVLAPolicy). -->

## lessons: What we learned

What we learned: the agent worked like a good engineer. It looked at the evidence, found
the physical cause, changed one thing at a time, built a measurement when the records
weren't enough, and proved its solutions on cases it had never seen. These were first,
compact tests, and in each one the agent carried the investigation through to a working
result.

<!-- purpose: the general finding: the engineering loop works with an agent in it. -->
<!-- screen: as cut 3: the loop as five cards, each with a thumbnail from the film. -->

## next: Next: real robots

Next, we take this to real robots: a complete packing operation on two-armed mobile
robots, where agents engineer new work on the hardware and prove it there. The question we
will answer: how much less human work each new job needs, once agents do the engineering.

<!-- purpose: what comes next, as the narration says it. -->
<!-- screen: a stylised packing cell with two two-armed mobile robots standing on the floor
(wheels on the ground, shadows), driving between shelf and packing table and carrying
boxes and items (static and grounded if motion is too much), boxes and a
packaging machine; request cards arrive; an "engineer hours per new job" bar shrinks as a
question mark, not a result. -->

## mayfly: What it means

That is what Mayfly is building: robot installations where taking on new work is part of
the product, because agents do the engineering.

<!-- screen: zoom out from the packing cell; request cards flow through the loop onto a
growing library shelf; Mayfly wordmark and the line. -->

## end: How this was recorded

<!-- screen, no voice: "Simulation: LIBERO household tasks, MuJoCo physics. Robot model:
X-VLA, weights unchanged in every run. Engineering agent: {r1_agent_model}. Every clip is a
recorded run; every count is literal, from {r1_plain_n} to {b_eval_final_n} runs." Then two
addresses: the record (pool1892.github.io/sprind-demo-page) and the code
(github.com/pool1892/sprind-demo-code). -->
