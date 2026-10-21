---
name: episode-cards
description: Describe sampled episodes one by one from their telemetry and video frames, as short cards with the phase where it went wrong, the symptom, the evidence frames and a candidate cause in the fleet's vocabulary; with general knowledge of timing in a control loop (latency, feedback delay, buffering, action chunks, replanning) and the competing explanations a timing symptom can have.
when_to_use: After triage has drawn a sample, and whenever a mode or a hypothesis needs more examples. A card is evidence about one episode, not about a robot. Also when motion looks hesitant, late or oscillating, or a site or robot may be working under a delay.
tools_used: [query, get_frames, memory_read, memory_write]
inputs:
  - episode ids from the sample of triage (or from failure-modes or scope)
outputs:
  - one card per episode, kept in memory in the card format below
  - for a timing symptom, the competing explanations with what each predicts, and the readings that tell them apart
version: "4"
---
# Episode cards

A card says what happened in one episode, where, and what would explain it,
with the frames that show it. Cards are the raw material of `failure-modes`.
Look before you explain: write the symptom from what the telemetry and the
frames show, and only then the candidate cause.

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## Procedure

1. Run `profile` (below) with the sampled episode ids as its input rows, up
   to about twenty at a time. One row per episode gives:
   - `close_step`: the step at which the fingers closed for the grasp: the
     first time their opening fell below `closed_below` of the episode's
     widest opening and stayed there for `settle_steps`; empty if they never
     closed.
   - `held_ratio`: the finger opening once the fingers have settled
     (`settle_steps` to ten steps more after closing), as a share of the
     widest. The fingers close slowly and creep on for a while when nothing
     is between them, so a reading taken sooner says nothing. Below
     `empty_below` the fingers met: they closed on nothing (a missed grasp).
     Above it something is held: a thin rim settles low (a bowl held by its
     rim reads a few hundredths), a bottle or a box much higher. Empty when
     the fingers never stayed closed that long.
   - `collapse_step`: the first later step, before the fingers open again,
     at which an object that was held is no longer held (the fingers met): a
     drop or a slip.
   - `reopen_step`: the first step after the grasp at which the fingers opened
     again: a release (wanted or not).
   - `lift_cm`: how far the gripper rose after closing.
   - `still_steps`: steps in which the arm barely moved (below `still_below`
     of its fast speed, the 90th percentile): hesitation or a stall.
   - `reversals`: how often the first joint changed direction at speed:
     oscillation or overshoot around a target.
   - `events`: the fleet's own events with their steps. Read what they
     report before you lean on them (`SELECT kind, detail, count(*) FROM
     events GROUP BY ALL`): a grasp or release event whose detail speaks of
     the gripper being commanded closed or open says when the policy asked
     for it, not whether an object was caught.
   - `end_phase`: where the episode ended up: approach, grasp, carry, place,
     or completed. A drop event or a collapse says the object was lost; the
     settled finger opening says whether anything was held.
   - `frame_steps`: the logged frames nearest to the moments that matter
     (start, closing, just after, a collapse, the first notable event, the
     release, the end). Empty when the episode has no video.
2. For each episode, call `get_frames` with its `frame_steps` (at most eight)
   and `camera` `scene`. Add the `wrist` camera at the closing step when the
   grasp is in question: the wrist view shows whether the fingers are around
   the object. Compare with the frames of a success of the same task (the
   sample has one per robot): the difference is the evidence.
3. Write the card. One line per episode, in this format, in the fleet's
   vocabulary (components, configuration keys and conditions as `describe`
   lists them):

   `<episode> | <robot> | day <d> | <task class> | <outcome> | phase: <phase> | symptom: <what went wrong> | frames: <steps> | cause: <component, key or condition>, <direction> | confidence: <low, medium or high>`

   Directions are `degraded`, `misconfigured`, `changed` and `drifting`.
   Write `cause: unclear` rather than a guess. A success with a near miss gets
   a card too: it shows a mode before it turns into failures.
4. Keep the cards in memory (`memory_write` with the whole updated notes),
   grouped by robot. Cards are short on purpose: twenty cards must fit in a
   few kilobytes.

<!-- fault-class-knowledge:begin -->
## Public vocabulary for hypotheses

Use the public description to name candidates, then test them against matched
rates and episode evidence. The following are hypotheses, not response priors:

- A difference in images suggests inspecting the cameras and the site's
  lighting. A camera's pose and image quality are different observations.
- A held object that is lost suggests inspecting the gripper and the objects
  in use. Compare robots, sites and task classes before locating the cause.
- A difference in command timing suggests inspecting the documented buffering
  and control-mode keys. Use the change log to test chronology, including
  concurrent changes and absent comparison support.
- A different cell layout suggests inspecting the site's placement condition.
- A disagreement between frames and outcomes requires checking the dictionary's
  completion criterion. Compare late frames, finger opening and a matched
  successful episode. A final frame alone does not establish a recording error.

A symptom can fit several targets or an unknown mechanism. Retain alternatives
when the available measurements cannot distinguish them.
<!-- fault-class-knowledge:end -->

<!-- fault-class-knowledge:begin -->
## Timing in the control loop

General engineering knowledge, at the level of a controls textbook or a
robot vendor's integration manual: how timing problems arise, how they show
and what engineers commonly do about them. It does not say how large any of
these effects is in this fleet, which explanation holds here, or which
setting works best. Those are for the records and your experiments to show.

### The loop and where the time goes

A learned manipulation policy closes a loop. Sensors capture the scene and
the robot's state; the policy turns that observation into commands; the
commands pass through buffers and links to the robot's controller; the
controller drives the joints toward each command; the motion changes what
the sensors see next. Every stage takes time:

- **sensing**: a camera's exposure, readout, encoding and transfer;
- **inference**: the policy's computation, which grows with the model, the
  number of sampling or denoising steps, the numerical precision and the
  number of image views;
- **queueing**: commands that wait in a buffer before they are executed;
- **transport**: a fieldbus, a network or a remote link between the computer
  that runs the policy and the controller;
- **the controller**: its servo cycle, and the tracking lag every real
  controller has, since it reaches a commanded target over several cycles,
  not at once.

What matters for behaviour is the total delay from the moment the world
looked as the observation shows to the moment the resulting command acts on
the robot. A steady delay and a varying one (jitter) do different damage.

### Action chunks and replanning

Many current policies (action-chunking transformers, diffusion policies,
vision-language-action models) predict a chunk: a sequence of future actions
from one observation. The executor runs some of them, the execution horizon,
and then calls the policy again with a fresh observation.

- Running the whole chunk is open loop for its length: smooth and cheap in
  computation, but blind to what happens meanwhile (a slip, a nudged object,
  an approach that is off).
- Running a few actions and replanning is closer to closed loop: more
  reactive, but the policy runs more often, and each switch to a new chunk
  can bring a discontinuity, because consecutive chunks need not agree (two
  plausible ways to grasp, for example).
- Synchronous execution waits for each new chunk, so the robot pauses at
  every chunk boundary for the inference time. Asynchronous execution keeps
  running the old chunk while the next one is computed; the new chunk then
  arrives late, and its first actions are meant for a moment already past.
- Smoothing across chunks (averaging the overlapping predictions of several
  chunks, as temporal ensembling does) trades responsiveness for smoothness.

### What delay does

- In a feedback loop, delay erodes the stability margin. A loop that
  corrects often and strongly, tuned without delay, starts to overshoot and
  oscillate as the delay grows: it keeps correcting an error that commands
  already in flight will remove, then corrects its own overshoot. In linear
  control the faster a loop corrects, the less delay it tolerates. A learned
  policy is not a linear controller, so treat this as a hypothesis to test,
  not a rule.
- In an open-loop stretch, a steady delay mostly shifts the motion in time.
  It hurts where timing between things matters (closing the gripper as the
  arm arrives, releasing over a target) or where the world moves meanwhile.
- Commands computed from an old view aim at where things were: approaches
  that end beside the object, grasps that close a moment late, corrections
  that reverse soon after they start.
- A buffer that fills faster than it drains adds a delay that grows over an
  episode. Flushing stale commands when a new plan arrives bounds it.

### What engineers commonly do

1. **Measure before changing anything.** Timestamp each stage where the
   system allows it; otherwise estimate the delay from the records (below).
   Compare with healthy robots doing the same tasks, because every
   controller lags its commands a little.
2. **Remove the delay at its source** where that is possible: shorter or
   flushed buffers, a local link instead of a remote one, faster inference
   (fewer sampling steps, lower precision, fewer views), each at the cost of
   whatever that setting buys.
3. **Compensate for a known delay** that cannot be removed. Predict forward
   over the delay (the idea of the Smith predictor), or, for a chunked
   policy, skip the head of each new chunk that is already in the past when
   it arrives, or condition the new chunk on the actions still in flight so
   that it continues them (as real-time chunking does). Compensation needs
   the delay's size: too little leaves the error, too much skips actions
   that were still due.
4. **Retune how much of each plan runs before replanning.** A longer
   execution horizon is less exposed to delay-driven oscillation and to
   chunk switches but reacts later to errors; a shorter one reacts sooner
   but meets the delay at every switch. Which side wins depends on the size
   of the delay, the task, the policy and the other settings, so engineers
   sweep it on the affected robots instead of reasoning it out.
5. **Slow down where precision matters**: lower speeds or controller gains
   near contact, so that a late correction does less harm.

When a delay cannot be removed, adapting to it is the usual answer: the
condition stays and the operating point moves. Settings like these act
together (a compensation that suits one horizon may not suit another), so
a change of one can shift the best value of the other.

### Reading timing from the records

The data dictionary's `steps` table holds, per control step, the commanded
action, the measured state (`joint_pos`, `eef_pos` and others), and which
policy call each action came from and its place within that call
(`chunk_index`, `chunk_step`). Read each column's convention in the
dictionary before you build on it (whether a commanded position is a
displacement or a target, for example).

- **Lag between commanded and measured motion.** For a range of shifts k,
  correlate the commanded motion at step t with the measured motion at step
  t + k: for displacement commands, the end effector's measured change of
  position from one step to the next; for target commands, its position.
  The shift with the strongest correlation estimates the delay in steps. A
  sharp peak says the delay is steady; a broad or low one says it varies.
  Use the translation components and the stretches where the arm moves
  (the correlation means nothing while it stands still), and compare robots
  on the same tasks.
- **The execution horizon in use.** Count the steps of each `chunk_index`
  within an episode.
- **Pauses and jumps at chunk boundaries.** Still steps that cluster at the
  start of each chunk suggest waits for inference. A jump in the commanded
  action from the last action of one chunk to the first of the next
  suggests chunks that disagree or arrive late.
- **Oscillation.** Reversals of the arm (the `profile` block's `reversals`)
  and approaches that overshoot and come back.
- **Scope and moment.** Which robots, sites and days show it, and whether it
  began with a logged change or was there from the start, and what the
  public description says about the sites and their equipment.

### Competing explanations that look alike

In the outcome tables these all show the same way: lower success, hesitant
or oscillating approaches, late or missed grasps. They differ in what they
predict in the records and in an experiment:

- **Command delay** (commands reach the controller late, through a buffer, a
  link or a transport). The measured motion follows the commands faithfully
  but late, by a shift that is about the same in every episode, phase and
  task of the robots concerned, whatever the robot is doing. The commands
  look normal, and the camera views agree with the joint record. It covers
  every robot behind the same buffer or link, often a whole site, from the
  moment the cause began. In an experiment it reacts to compensation sized
  to the delay and to the execution horizon, not to faster inference.
- **An overloaded controller** (the servo loop misses its cycle). The timing
  is irregular rather than late by a fixed amount: jitter, held or skipped
  commands, a tracking error that grows with speed and with demanding
  motion, worse in some phases or tasks than in others. The lag estimate's
  peak is broad, or it moves between episodes. A fixed compensation does
  little against it.
- **Sensor or camera lag** (the views are old when the policy sees them).
  The controller executes commands promptly, so the lag between commanded
  and measured motion is normal, but the commands aim at the past: the
  policy steers toward where the object or the gripper was, overshoots and
  reverses, more so when things move fast. If the recorded frames are what
  the policy saw, they trail the joint record. It belongs to a camera or the
  image path, not to the controller's link.
- **A slow policy** (inference takes longer than before, from a heavier
  setting, version or precision). With synchronous execution the robot
  pauses at every chunk boundary, more often the more often the policy is
  called; with asynchronous execution the chunks arrive stale and the
  commanded actions jump at the boundaries. The lag within a chunk is
  normal. It reacts to faster inference.

Other causes can pass for timing: a gripper that closes slowly or weakly
(late grasps without any lag in the arm), a camera calibration error
(approaches that are offset rather than late), a new object or placement
(misses that depend on the task rather than the robot). Keep two or more
explanations open until a measurement separates them, and choose the
experiment whose result the competing explanations predict differently.
<!-- fault-class-knowledge:end -->

Two episodes are an anecdote; a card is worth most when it can be compared
with the same task on a healthy robot.

## Blocks

```sql
-- helper: profile
WITH params AS (SELECT 0.85 AS closed_below, 0.025 AS empty_below, 25 AS settle_steps, 0.1 AS still_below),
-- input: picked(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')),
picked AS (SELECT episode_id FROM episodes WHERE outcome = 'failure' ORDER BY md5(episode_id) LIMIT 5),
-- end input
raw AS (
  SELECT s.episode_id, s.step, s.frame_index, s.eef_pos[3] AS z,
    abs(s.gripper_qpos[1]) + abs(s.gripper_qpos[2]) AS width,
    sqrt(s.joint_vel[1] ^ 2 + s.joint_vel[2] ^ 2 + s.joint_vel[3] ^ 2 + s.joint_vel[4] ^ 2
      + s.joint_vel[5] ^ 2 + s.joint_vel[6] ^ 2 + s.joint_vel[7] ^ 2) AS speed,
    s.joint_vel[1] AS first_joint
  FROM steps s JOIN (SELECT DISTINCT episode_id FROM picked) USING (episode_id)
),
scaled AS (
  SELECT *, width / nullif(max(width) OVER per_episode, 0) AS opening,
    speed / nullif(quantile_cont(speed, 0.9) OVER per_episode, 0) AS relative_speed,
    max(abs(first_joint)) OVER per_episode AS joint_peak,
    lag(first_joint) OVER (PARTITION BY episode_id ORDER BY step) AS previous_joint
  FROM raw WINDOW per_episode AS (PARTITION BY episode_id)
),
-- stretches of closed fingers; the grasp is the first one long enough to settle
stretches AS (
  SELECT x.episode_id, x.step,
    x.step - row_number() OVER (PARTITION BY x.episode_id ORDER BY x.step) AS stretch
  FROM scaled x, params p WHERE x.opening < p.closed_below
),
grasp AS (
  SELECT s.episode_id,
    coalesce(min(s.start) FILTER (WHERE s.n >= p.settle_steps), min(s.start)) AS close_step
  FROM (SELECT episode_id, stretch, min(step) AS start, count(*) AS n FROM stretches
    GROUP BY episode_id, stretch) s, params p
  GROUP BY s.episode_id
),
closing AS (
  SELECT x.episode_id, any_value(gr.close_step) AS close_step, max(x.step) AS last_step,
    count(*) FILTER (WHERE x.relative_speed < p.still_below) AS still_steps,
    count(*) FILTER (WHERE sign(x.first_joint) * sign(x.previous_joint) < 0
      AND abs(x.first_joint) > 0.2 * x.joint_peak) AS reversals,
    bool_or(x.frame_index IS NOT NULL) AS has_video
  FROM scaled x LEFT JOIN grasp gr USING (episode_id), params p GROUP BY x.episode_id
),
grip AS (
  -- the opening once the fingers have settled: something held, or nothing between them
  SELECT c.episode_id,
    median(x.opening) FILTER (WHERE x.step BETWEEN c.close_step + p.settle_steps
      AND c.close_step + p.settle_steps + 10 AND x.opening < p.closed_below) AS held_ratio,
    min(x.z) FILTER (WHERE x.step = c.close_step) AS z_close,
    max(x.z) FILTER (WHERE x.step >= c.close_step) AS z_top
  FROM closing c JOIN scaled x USING (episode_id), params p GROUP BY c.episode_id
),
later AS (
  SELECT c.episode_id,
    min(x.step) FILTER (WHERE x.step > c.close_step + p.settle_steps
      AND x.opening >= p.closed_below) AS reopen_step,
    min(x.step) FILTER (WHERE x.step > c.close_step + p.settle_steps + 10
      AND x.opening < p.empty_below) AS meet_step
  FROM closing c JOIN scaled x USING (episode_id), params p
  GROUP BY c.episode_id
),
lost AS (
  SELECT l.*, CASE WHEN g.held_ratio >= p.empty_below
    AND l.meet_step < coalesce(l.reopen_step, 1000000) THEN l.meet_step END AS collapse_step
  FROM later l JOIN grip g USING (episode_id), params p
),
ev AS (
  SELECT episode_id, string_agg(kind || '@' || step, ' ' ORDER BY step) AS events,
    min(step) FILTER (WHERE kind IN ('drop', 'collision', 'stall', 'near_miss')) AS event_step,
    -- a drop event says the object was lost; grasp and release events report the
    -- gripper's command, not the object, so whether something is held is read off the fingers
    bool_or(kind = 'drop') AS dropped, bool_or(kind = 'release') AS released
  FROM events WHERE episode_id IN (SELECT episode_id FROM picked) GROUP BY episode_id
),
summary AS (
  SELECT e.episode_id, e.robot_id, e.day, e.task_text, e.outcome, e.steps, c.close_step,
    round(g.held_ratio, 2) AS held_ratio, l.collapse_step, l.reopen_step,
    round(100 * (g.z_top - g.z_close), 1) AS lift_cm, CAST(c.still_steps AS INT) AS still_steps,
    CAST(c.reversals AS INT) AS reversals, coalesce(ev.events, '') AS events, c.has_video,
    CASE WHEN e.outcome = 'success' THEN 'completed'
      WHEN coalesce(ev.dropped, false) OR l.collapse_step IS NOT NULL THEN 'carry: lost the object'
      WHEN g.held_ratio IS NULL THEN 'approach: never closed'
      WHEN g.held_ratio < p.empty_below THEN 'grasp: closed on nothing'
      WHEN coalesce(ev.released, false) OR l.reopen_step IS NOT NULL
        THEN 'place: released, not placed'
      ELSE 'carry or place: held to the end' END AS end_phase,
    [0, c.close_step, c.close_step + p.settle_steps, l.collapse_step, ev.event_step, l.reopen_step,
      c.last_step] AS moments
  FROM episodes e JOIN closing c USING (episode_id) JOIN grip g USING (episode_id)
    JOIN lost l USING (episode_id) LEFT JOIN ev USING (episode_id), params p
),
nearest AS (
  SELECT m.episode_id, m.moment,
    arg_min(f.step, abs(f.step - m.moment) * 100000 + f.step) AS frame_step
  FROM (SELECT episode_id, unnest(moments) AS moment FROM summary) m
    JOIN raw f ON f.episode_id = m.episode_id AND f.frame_index IS NOT NULL
  WHERE m.moment IS NOT NULL
  GROUP BY m.episode_id, m.moment
)
SELECT s.* EXCLUDE (moments),
  coalesce((SELECT list(DISTINCT n.frame_step ORDER BY n.frame_step) FROM nearest n
    WHERE n.episode_id = s.episode_id), []) AS frame_steps
FROM summary s
ORDER BY s.robot_id, s.episode_id
```
