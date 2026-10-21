# Narration: the reliability film, final cut 1

The revised demo film: two new jobs that an engineering agent made reliable, then story B,
the instrument an agent built for a fleet, told briefly and unchanged in its results. About
three minutes. Beat ids are those of `site/manifests/reliability-final-2-2026-10-20.json`
(final cut 1, `reliability-final-1-2026-10-20.json`, bound this file as of b50d84e).
The private drafts of the first case (`reliability-first-case-2026-10-20.json` and
`reliability-first-case-2-2026-10-20.json`) bound earlier versions of this file; they are
kept at commits 82222f9 and 7cb3947.

The rules are those of `site/narration/script.md`: one section per beat, headed
`## <beat id>: <title>`; everything under a heading is read aloud except HTML comments and
lines starting with `>`; **no number is written into this script**, every number is a
placeholder `{name}` defined in `site/reliability/fields.json` and bound from the cut's
records (`site/manifests/reliability-final-2-2026-10-20.records.json`). Only spoken
placeholders are bound, so a number on a card is spoken too. A `source:` note names what
each sentence rests on. Until a voice is recorded the build is silent, with these words as
captions.

## open: Opening

The robot's policy never changes. An engineering agent improves the installation around it.

<!-- source: X-VLA weights fixed throughout (job_run.json agent and policy records of both investigations; draft section 1). "Engineering agent": gpt-6-astra through the job driver. -->

## r1-request: The first job

First job: the bowl on the cabinet and the middle drawer open, together, on at least
{r1_release_k} of {r1_release_n} fresh starts. The robot has a skill for each half.

<!-- source: replacement-01 job_run.json request.request_text (sentences 1 and 2, shown verbatim on screen), policy_skills, acceptance; K and N bound from the receipt. The agent proposed the bar and the lead approved it before any program ran (message_log 2 and 3). -->

## r1-failure: What fails

Chained plainly, the hand turns to the drawer still touching the bowl, and knocks it off.
On the agent's practice starts: {r1_plain_dev_delivered} of {r1_dev_starts}.

<!-- source: the agent's session-2 diagnosis and account; count j0001, session 1. Footage: the plain chain on reserved start 31 (e000202), verified 720px scene re-render; the label says it is a reserved start. -->

## r1-experiment: The experiment

The agent writes a check: does the bowl stay in place while the next skill runs? Then it
changes one thing at a time. Open, then go home: {r1_open_home} of {r1_dev_starts}. Let the
placement run {r1_finish_steps} steps longer, then go home: {r1_finish_home} of
{r1_dev_starts}.

<!-- source: check handoff-retention-and-delivery (session 2), purpose and expects, summarised on the card and labelled as a summary; counts j0005 and j0006, session 3, the same ten practice starts as j0001. Footage: practice start 2, e000503 against e000603, original 360px. -->

## r1-release: Released

Confirmed {r1_confirm_delivered} of {r1_confirm_starts} on unused starts, then
{r1_delivered} of {r1_release_n} reserved starts, against {r1_naive_delivered} for the plain
chain: {r1_rescued} rescued, {r1_broken} broken.

<!-- source: confirmation j0008 (session 4, starts 10 to 29); receipt (starts 30 to 49, starts_fresh), reconciled in replacement-01/coordinator-verification.json. Footage: first rescued reserved start 31 by `film pairs`, e000202 against e000102, verified 720px scene re-renders. -->

## r1-kept: What it keeps

It keeps the job, the check, and a lesson: an open, still hand at one instant has not
cleared the path that follows.

<!-- source: replacement-01 library: job station-bowl-and-open-drawer, check handoff-retention-and-delivery, skill quiet-handoff-versus-path-clearance (description shown verbatim; the sentence paraphrases it). -->

## r2-request: The second job

Second job: the plate at the front of the stove and the bowl on the cabinet. Again, at
least {r2_release_k} of {r2_release_n}.

<!-- source: plate-cabinet-01 job_run.json request.request_text (sentences 1 and 2, shown verbatim on screen), acceptance; K and N bound from the receipt. No order rule. -->

## r2-failure: What fails

Plate first, then bowl: {r2_plain_dev_delivered} of {r2_dev_starts} practice starts. Here
the plate push runs out of time short of the stove.

<!-- source: count j0001, session 1. Footage: practice start 1, e000102 of session 1 (the plate step times out at 300 steps; the bowl step is skipped), original 360px; the first start j0015 rescued against j0001 by `film pairs`. This is the development failure, not the reserved start shown later. -->

## r2-loaded: The experiment

The agent extends the first job's check to both goals and tests changes. Going home
between skills: {r2_home} of {r2_dev_starts} delivered. Loading the cheese into the bowl
gets every plate to the stove, but the loaded bowl becomes the bottleneck: {r2_loaded} of
{r2_dev_starts} delivered.

<!-- source: account (library check tested and extended as station-goal-and-handoff-audit; j0013 "all plate goals acquired and retained, but loaded-bowl pickup/placement became the bottleneck"); counts j0004 and j0013 on the same starts 0 to 9 (j0015 in r2-relocated). The cheese is moved aside deliberately; the film does not claim it was the only thing that changed or an isolated cause. Root's final-trace check: the cheese rests on the table in all 20 release episodes (relocated-cheese-final-positions.json). Footage: practice start 0, e001301 (session 4, loaded), original 360px; the first start j0015 rescued against j0013 by `film pairs`. The counts are deliveries. -->

## r2-relocated: The scene, prepared

So the agent grasps the cheese, slides it aside onto the table, and leaves the bowl
empty: {r2_relocated} of {r2_dev_starts} delivered.

<!-- source: the saved program (cheese skill stopped at 60 steps, a 20-step slide of -0.05 m in x, 20 steps open, home) and j0015's count, session 5. Footage: the same practice start 0, e001501 (session 5), original 360px: grasp, slide and release first (steps 0 to 130), then plate and bowl (130 to 420). Root's final-trace check: the cheese rests on the table in all 20 release episodes. -->

## r2-release: Released

Confirmed {r2_confirm_delivered} of {r2_confirm_starts} on unused starts, then
{r2_delivered} of {r2_release_n} reserved starts, against {r2_naive_delivered} for the plain
chain: {r2_rescued} rescued, {r2_broken} broken. Here the plain chain reaches the stove, but
its bowl attempt leaves neither goal standing.

<!-- source: confirmation j0019 (session 6, starts 10 to 29, the unchanged j0015 program); receipt, reconciled in plate-cabinet-01/coordinator-verification.json. Footage: first rescued reserved start 31 by `film pairs`, e000202 (plate goal at step 145, bowl step times out at 445, both goals false at the end) against e000102 (delivered; plate 132 to 270, bowl 307 to 389), verified 720px scene re-renders (replay-film-reader-verification.json). -->

## r2-kept: What it keeps

It keeps the job, the extended check, and a lesson: preparing the scene can move the
bottleneck to the next skill.

<!-- source: plate-cabinet-01 library: job station-plate-and-bowl-prepared, check station-goal-and-handoff-audit, skill scene-preparation-can-move-the-bottleneck (description shown verbatim; the sentence paraphrases its name). No claim that reuse made the second job faster. -->

## b-instrument: An instrument for a fleet

Story two: a fleet of arms at several sites, with the same lag at one site and on one
robot elsewhere. The agent builds its own probe, commanded against measured: one lag is a
fixed network delay, the other a controller falling behind.

<!-- source: story B as in the earlier final cuts, unchanged: the agent's round-1 notes (s1 and r06 both lag), the probe's recorded tables (probe-first-fleet.png and its record). -->

## b-result: The fleet, repaired

After paired experiments it compensates for the delay, rolls back an update and services
the controller. On {b_attempts} reserved attempts it never saw, success rises from
{b_original_success} to {b_final_success}: {b_rescued} rescued, {b_broken} broken.

<!-- source: the round-1 ledger (L1 to L3) and the evaluation summary, the same fields and file as the earlier final cuts. -->

## close: The record

Two jobs at {r1_delivered} and {r2_delivered} of {r1_release_n} reserved starts, and a
fleet improved by an instrument the agent built. The policy never changed; every clip is a
recorded run.

<!-- source: both receipts and the story B evaluation; every clip resolves to a recorded run in provenance.json. -->
