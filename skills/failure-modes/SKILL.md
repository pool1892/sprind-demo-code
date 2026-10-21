---
name: failure-modes
description: Group episode cards into failure modes, give each mode a telemetry fingerprint, and count every mode over all failures of the log by robot, site, day and task, so that a mode seen in a handful of cards becomes a number.
when_to_use: After episode-cards, once there are cards for most robots that lose points. Again whenever new cards do not fit the existing modes.
tools_used: [query, get_frames, memory_read, memory_write]
inputs:
  - the cards from episode-cards (in memory)
  - the outcome table from triage (in memory)
outputs:
  - a list of failure modes, each with a name, its fingerprint, example episodes, and its counts by robot, site, day and task
  - for each mode, whether it is concentrated or spread, and candidate scopes to investigate
version: "3"
---
# Failure modes

A failure mode is a way of failing: the phase where it goes wrong and the
symptom, such as "carry: the object is lost after a good grasp". It is not
yet a cause. One mode can have several causes (two robots that drop objects
for different reasons), and one cause can produce several modes. The point
of this skill is to turn cards, which describe a few episodes, into counts
over every episode, which is what the later skills need.

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## Procedure

1. Read the cards from memory (`memory_read`). Group them by phase and
   symptom first, by candidate cause second. Name each group in words,
   phase first. Leave a card that fits nowhere on its own: it may be the
   first of a mode the sample under-represents.
2. Run `cards_check` (below) with one input row per card: the episode and
   the mode you gave it. It shows, for each mode, the telemetry signature of
   its episodes (the same rules as `profile` in episode-cards, applied to
   every episode). Whether something was held is read off the settled
   finger opening; a grasp or release event that reports the gripper's
   command says only when it was asked to close or open:
   - `lost the object`: a drop event, or something held after closing and
     then the fingers met before they opened again;
   - `never closed`: the fingers never closed, or never stayed closed long
     enough to settle;
   - `closed on nothing`: the fingers settled with nothing between them;
   - `collision`, `stalled`: the fleet's events;
   - `released, not placed`: the gripper opened again (or a release event),
     but the task failed;
   - `held to the end`: still holding when the episode ran out.
   If the cards of a mode share one signature, that signature is the mode's
   fingerprint. If they split, the mode mixes two things, or a card is
   wrong: look at the frames again (`get_frames`) before going on.
3. Run `signatures` (below). One row per signature over all failures in the
   window: how many, their share of failures, and their counts per robot,
   site, day and task (`k/n`: failures with this signature out of all
   episodes of that robot, site, day or task). The counts are what the
   cards could not give.
4. Read each mode's spread. Compare rates at each scope before attribution.


   Public target vocabulary suggests the following hypotheses:
   - **Concentrated on one robot**: inspect robot components and configuration. Its scope is the robot.
   - **Concentrated on one site**, on all its robots: a condition of the site
     (lighting, placement, object batch, clutter) or a configuration set at
     the site.
   - **Starting on a day**, on the robots of one scope: a change made that
     day. The change log names recorded candidates; it need not contain the
     responsible change (see `scope`).
   - **Growing day by day**: inspect time-varying measurements and changes.
   - **Concentrated on a task or an object**: compare the same task across robots; on one site only, look at the site's object batch
     or placement.
   - **Spread over the whole fleet**: consider fleet-wide configuration,
     common circumstances and the policy. A broad scope alone does not show
     that no repair can help.
   Compare task-conditioned rates across sites before calling residual
   failures the policy's own weakness. A site with high overall success may
   still have a shared failure on a small class of objects or tasks. Keep an
   unexplained site/task mode in view even when no logged key or familiar
   component mechanism accounts for it; do not force it into a known class.

5. Check the arithmetic against triage: the excess failures of a mode on its
   scope (its failures there minus what the rest of the fleet shows at the
   same rate) should account for the points that scope loses in the outcome
   table. If a scope loses more than its modes explain, another mode hides
   there: draw more cards from it (triage's `sample` with another `salt`).
6. Write the modes to memory: name, fingerprint, two or three example
   episodes with their frames, counts by scope, and the reading of step 4.

Next: `scope` for each concentrated mode.

## Blocks

```sql
-- helper: cards_check
WITH params AS (SELECT 0.85 AS closed_below, 0.025 AS empty_below, 25 AS settle_steps),
-- input: cards(episode_id, mode) AS (VALUES ('<episode id>', '<mode>'), ('<episode id>', '<mode>')),
cards AS (SELECT episode_id, 'example mode' AS mode FROM episodes WHERE outcome = 'failure' ORDER BY md5(episode_id) LIMIT 6),
-- end input
raw AS (
  SELECT episode_id, step, abs(gripper_qpos[1]) + abs(gripper_qpos[2]) AS width
  FROM steps WHERE episode_id IN (SELECT episode_id FROM cards)
),
grip AS (
  SELECT episode_id, step, width / nullif(max(width) OVER (PARTITION BY episode_id), 0) AS opening
  FROM raw
),
-- stretches of closed fingers; the grasp is the first one long enough to settle
stretches AS (
  SELECT g.episode_id, g.step,
    g.step - row_number() OVER (PARTITION BY g.episode_id ORDER BY g.step) AS stretch
  FROM grip g, params p WHERE g.opening < p.closed_below
),
closing AS (
  SELECT s.episode_id,
    coalesce(min(s.start) FILTER (WHERE s.n >= p.settle_steps), min(s.start)) AS close_step
  FROM (SELECT episode_id, stretch, min(step) AS start, count(*) AS n FROM stretches
    GROUP BY episode_id, stretch) s, params p
  GROUP BY s.episode_id
),
held AS (
  -- the opening once the fingers have settled: something held, or nothing between them
  SELECT c.episode_id, c.close_step,
    median(g.opening) FILTER (WHERE g.step BETWEEN c.close_step + p.settle_steps
      AND c.close_step + p.settle_steps + 10 AND g.opening < p.closed_below) AS held_ratio,
    min(g.step) FILTER (WHERE g.step > c.close_step + p.settle_steps + 10
      AND g.opening < p.empty_below) AS meet_step,
    min(g.step) FILTER (WHERE g.step > c.close_step + p.settle_steps
      AND g.opening >= p.closed_below) AS reopen_step
  FROM closing c JOIN grip g USING (episode_id), params p GROUP BY c.episode_id, c.close_step
),
ev AS (
  -- a drop event says the object was lost; grasp and release events report the gripper's
  -- command, not the object, so whether something is held is read off the fingers
  SELECT episode_id, bool_or(kind = 'drop') AS dropped, bool_or(kind = 'release') AS released,
    bool_or(kind = 'collision') AS collided, bool_or(kind = 'stall') AS stalled
  FROM events GROUP BY episode_id
),
signed AS (
  SELECT c.mode, e.episode_id, e.robot_id, CASE
    WHEN e.outcome = 'success' THEN 'success'
    WHEN coalesce(ev.dropped, false) OR (h.held_ratio >= p.empty_below
      AND h.meet_step < coalesce(h.reopen_step, 1000000)) THEN 'lost the object'
    WHEN h.held_ratio IS NULL THEN 'never closed'
    WHEN h.held_ratio < p.empty_below THEN 'closed on nothing'
    WHEN coalesce(ev.collided, false) THEN 'collision'
    WHEN coalesce(ev.stalled, false) THEN 'stalled'
    WHEN coalesce(ev.released, false) OR h.reopen_step IS NOT NULL THEN 'released, not placed'
    ELSE 'held to the end' END AS signature
  FROM cards c JOIN episodes e USING (episode_id) LEFT JOIN held h USING (episode_id)
    LEFT JOIN ev USING (episode_id), params p
)
SELECT mode, signature, CAST(count(*) AS INT) AS cards,
  string_agg(DISTINCT robot_id, ' ' ORDER BY robot_id) AS robots,
  string_agg(episode_id, ' ' ORDER BY episode_id) AS episodes
FROM signed GROUP BY mode, signature ORDER BY mode, cards DESC
```

```sql
-- helper: signatures
WITH params AS (SELECT 0.85 AS closed_below, 0.025 AS empty_below, 25 AS settle_steps, 0 AS from_day, 9999 AS to_day),
ep AS (
  SELECT e.* FROM episodes e, params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
),
raw AS (
  SELECT s.episode_id, s.step, abs(s.gripper_qpos[1]) + abs(s.gripper_qpos[2]) AS width
  FROM steps s JOIN ep USING (episode_id) WHERE ep.outcome = 'failure'
),
grip AS (
  SELECT episode_id, step, width / nullif(max(width) OVER (PARTITION BY episode_id), 0) AS opening
  FROM raw
),
-- stretches of closed fingers; the grasp is the first one long enough to settle
stretches AS (
  SELECT g.episode_id, g.step,
    g.step - row_number() OVER (PARTITION BY g.episode_id ORDER BY g.step) AS stretch
  FROM grip g, params p WHERE g.opening < p.closed_below
),
closing AS (
  SELECT s.episode_id,
    coalesce(min(s.start) FILTER (WHERE s.n >= p.settle_steps), min(s.start)) AS close_step
  FROM (SELECT episode_id, stretch, min(step) AS start, count(*) AS n FROM stretches
    GROUP BY episode_id, stretch) s, params p
  GROUP BY s.episode_id
),
held AS (
  -- the opening once the fingers have settled: something held, or nothing between them
  SELECT c.episode_id, c.close_step,
    median(g.opening) FILTER (WHERE g.step BETWEEN c.close_step + p.settle_steps
      AND c.close_step + p.settle_steps + 10 AND g.opening < p.closed_below) AS held_ratio,
    min(g.step) FILTER (WHERE g.step > c.close_step + p.settle_steps + 10
      AND g.opening < p.empty_below) AS meet_step,
    min(g.step) FILTER (WHERE g.step > c.close_step + p.settle_steps
      AND g.opening >= p.closed_below) AS reopen_step
  FROM closing c JOIN grip g USING (episode_id), params p GROUP BY c.episode_id, c.close_step
),
ev AS (
  -- a drop event says the object was lost; grasp and release events report the gripper's
  -- command, not the object, so whether something is held is read off the fingers
  SELECT episode_id, bool_or(kind = 'drop') AS dropped, bool_or(kind = 'release') AS released,
    bool_or(kind = 'collision') AS collided, bool_or(kind = 'stall') AS stalled
  FROM events GROUP BY episode_id
),
signed AS (
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, e.task_text, CASE
    WHEN e.outcome = 'success' THEN 'success'
    WHEN coalesce(ev.dropped, false) OR (h.held_ratio >= p.empty_below
      AND h.meet_step < coalesce(h.reopen_step, 1000000)) THEN 'lost the object'
    WHEN h.held_ratio IS NULL THEN 'never closed'
    WHEN h.held_ratio < p.empty_below THEN 'closed on nothing'
    WHEN coalesce(ev.collided, false) THEN 'collision'
    WHEN coalesce(ev.stalled, false) THEN 'stalled'
    WHEN coalesce(ev.released, false) OR h.reopen_step IS NOT NULL THEN 'released, not placed'
    ELSE 'held to the end' END AS signature
  FROM ep e LEFT JOIN held h USING (episode_id) LEFT JOIN ev USING (episode_id), params p
),
scoped AS (
  SELECT 'robot' AS dimension, robot_id AS level, signature FROM signed
  UNION ALL SELECT 'site', site_id, signature FROM signed
  UNION ALL SELECT 'day', CAST(day AS VARCHAR), signature FROM signed
  UNION ALL SELECT 'task', task_text, signature FROM signed
),
counts AS (
  SELECT kinds.kind AS signature, x.dimension, x.level,
    count(*) FILTER (WHERE x.signature = kinds.kind) AS k, count(*) AS n
  FROM scoped x, (SELECT DISTINCT signature AS kind FROM signed WHERE signature <> 'success') kinds
  GROUP BY kinds.kind, x.dimension, x.level
),
listed AS (
  SELECT signature, dimension,
    string_agg(level || ' ' || CAST(k AS INT) || '/' || CAST(n AS INT), ', ' ORDER BY k / n DESC, level)
      AS spread
  FROM counts GROUP BY signature, dimension
)
SELECT t.signature, CAST(t.failures AS INT) AS failures,
  round(t.failures / (SELECT count(*) FILTER (WHERE signature <> 'success') FROM signed), 3)
    AS share_of_failures,
  round(100 * t.failures / (SELECT count(*) FROM signed), 2) AS per_100_episodes,
  (SELECT spread FROM listed l WHERE l.signature = t.signature AND dimension = 'robot') AS by_robot,
  (SELECT spread FROM listed l WHERE l.signature = t.signature AND dimension = 'site') AS by_site,
  (SELECT spread FROM listed l WHERE l.signature = t.signature AND dimension = 'day') AS by_day,
  (SELECT spread FROM listed l WHERE l.signature = t.signature AND dimension = 'task') AS by_task
FROM (SELECT signature, count(*) AS failures FROM signed WHERE signature <> 'success'
  GROUP BY signature) t
ORDER BY t.failures DESC
```
