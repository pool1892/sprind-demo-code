---
name: jobs-traces
description: Read what happened in a job's episodes from their records. The job tables (job_episodes, job_step_events, object_steps, goal_steps, robot_steps), how they join and how they are timed; an overview of the job runs; one episode's timeline; the moments that matter in an episode with the frames to look at; and the state of every object at chosen steps. How to run the SQL blocks of the job skills.
when_to_use: Whenever job runs have come back and before any explanation of what went wrong or right; again for every episode another job skill points at. Read it first among the job skills, because the others run their blocks as it describes.
tools_used: [query, get_frames, job_results, operator_state, memory_read, memory_write]
inputs:
  - the job ids of runs that have come back (job_results)
  - episode ids to look at (from the runs block, or from another job skill)
outputs:
  - an overview of the job runs with literal counts
  - for each episode looked at, a card in memory - what happened, at which control step, with the frames that show it
version: "2"
---
# Reading job traces

A job is a program of steps played in one continuous scene: policy steps (the
policy acting under an instruction until a goal holds or a step budget runs
out), motion steps (one scripted motion of the hand or the gripper) and waits.
Every control step of every episode is recorded. This skill is how to read
those records. Look before you explain: write down what the records and the
frames show, and only then what might explain it.

## The tables

Twenty control steps make one second. In every per-step table, row `step` = k
is one instant: the state measured after k commands, then command k issued.
A command acts during the next control step, so its effect shows first in row
k + 1. The last row of an episode is the state after its last command and has
no command. The per-step tables have one row per control step from 0 to
`job_episodes.control_steps` and join on `(episode_id, step)`. Vectors are
lists indexed from 1 (`position[3]` is the height); positions are in metres
in the world frame, z up.

- `job_episodes`: one row per episode (one start played through one
  program), keyed by `job_id` (the run, `j0001` and so on) and `episode_id`.
  The program by its `program_digest`; the start by `base_task` and
  `init_state` (the same two denote the same starting conditions, so two
  episodes with the same two had the same start); `start_eef_pos` and
  `start_eef_quat`, the hand's pose when the job started; `control_steps` and
  `tail_steps` (below); the host's verdicts: `delivered` with
  `delivery_detail` in words, and what the delivery check found at the end
  (`goals_held_at_end`, one boolean per requested goal in order;
  `robot_touching_at_end` and `moving_at_end`, the named objects the robot
  touched or that were not at rest; `gripper_opening_at_end`); `order_kept`
  with `order_detail`; `video_scene` and `video_wrist`, empty without video.
- `job_step_events`: one row per program step per episode: `program_step`
  (from 0), `kind`, `start_step` and `end_step` (empty when skipped; the
  next program step starts at `end_step`), `steps_used`, `outcome`
  (`reached`, `timeout`, `skipped`) and the `instruction` of a policy step.
  A row of kind `tail` with an empty `program_step` is the delivery tail.
- `goal_steps`: one row per control step per requested goal: `goal` (the
  goal's position in the request, from 0; `operator_state` lists the goals)
  and `holds`, the goal's own predicate evaluated on that state.
- `object_steps`: one row per control step per movable object of the scene:
  `object`, `position`, `quaternion` (w, x, y, z), `linear_speed` (m/s),
  `angular_speed` (rad/s), `in_contact_with_robot`, and `in_contact_with`:
  every body touching it, the robot's bodies by their own names.
- `robot_steps`: one row per control step: `eef_pos`, `eef_quat`,
  `gripper_opening` (metres between the fingers), `commanded_action` and
  `gripper_command` (-1 open, 1 closed) as issued, `instruction` (the
  instruction in force, empty during motion and wait steps), `program_step`
  (empty in the tail and the last row), `phase` (`program`, `tail`, or `end`
  for the last row) and `frame_index` (the step's frame in the videos, empty
  on steps without one).

**The delivery tail.** When a program's last step is a policy step with
outcome `reached` (its `until` goal held or its `until.steps` ran out), the
installation lets the policy go on under that step's
instruction until the delivery check holds or a fixed number of control
steps has passed (`operator_state` gives it), the same for every program.
Those steps are the tail: `phase` `tail` in `robot_steps`, a `tail` row in
`job_step_events`, their count in `job_episodes.tail_steps`. A program that
ends otherwise has no tail.

**Imported jobs.** In a numbered investigation the tables also hold the
episodes of earlier investigations of this work, under job ids from `j1001`
up; `job_results` lists those jobs with status `imported`, the part each
was, its own ids there and the request it ran for. Their goal indices are
that request's own: `goal_steps.goal` and `goals_held_at_end` of an imported
job count its `requested_goals`, which may differ from this request's, so
compare an imported episode with yours only on the goals both requests
share. Their frames come from `get_frames` as yours do, and every block below
runs on them. The notes, accounts and programs of those investigations are in
`job_library` (history); an id written in their notes is their own id there.

Check what the tables hold before you rely on a column:
`SELECT table_name, column_name, data_type FROM information_schema.columns
WHERE table_name IN ('job_episodes', 'job_step_events', 'goal_steps',
'object_steps', 'robot_steps') ORDER BY ALL`. If a column a block uses is
named differently, use the name the table has.

Two facts make these records better than a video for most questions. Contact
is read from the physics state, so `in_contact_with_robot` says whether the
robot touches an object even where the camera cannot see it. And `holds` is
evaluated at every control step, so a goal that holds for a moment and is
then lost shows up, even if it holds again later. Frames exist only on some
control steps (`frame_index`); the blocks give frame steps that have one.

## How to run the SQL blocks of the job skills

Each block in the job skills starts with a `-- helper: <name>` line. Copy
the block whole and change only two things:

- **The `params` line**, `WITH params AS (SELECT 20 AS every, ...)`: change the
  values, keep the names. Text in single quotes, numbers bare, `TRUE` or
  `FALSE`.
- **The input rows**, where a block takes rows: everything from the line
  after `-- input:` to the line before `-- end input`. The default there is a
  SELECT that makes the block run as it stands (usually on the first job run
  or episode). Replace that whole CTE by one of the form the `-- input:`
  comment shows, for example
  `picked(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')),`: the
  input's name and columns as the comment gives them, one row in parentheses
  per item, text in single quotes, numbers bare, and a comma after the last
  parenthesis. Keep the two marker lines.

Then run it with `query`. Keep the `-- helper:` line: the tool log then shows
which block produced a number, and every number you report must come from a
tool result. `query` returns at most 200 rows (the result then says
`truncated: true`) and stops a call after 60 seconds. If a result is
truncated, narrow it (fewer episodes, a shorter range of steps, a larger
`every`) and run it again; never read a truncated result as complete.

## Procedure

1. Call `memory_read` and continue from your notes. Call `operator_state` for
   the requested goals and their indices, and `job_results` for the runs that
   have come back, their job ids and the names of their programs (the tables
   name a program by its digest).
2. Run `runs` for an overview: one row per job run with its program digest,
   its episodes, how many were delivered, how many broke an order rule, how
   many program steps timed out or were skipped, how many episodes had a
   tail, the control steps used, and how many episodes have video. These are
   counts of a few starts: report them as counts ("7 of 10"), not as rates.
3. Pick the episodes to look at: every failure of a small run, and at least
   one delivered episode of the same program for comparison. Choose them
   before you look at any of them, and say how you chose them.
4. Run `moments` with those episodes (up to about twenty at a time). One row
   per episode:
   - `delivery_detail` and `goals_held_at_end`: the host's verdict in words,
     and which goals held at the end;
   - `switches`: each program step after the first, with the control step at
     which it started (`1@64` is program step 1 starting at step 64) and how
     the step before it ended; `tail` when the episode had one, from the step
     it started at;
   - `first_held` and `lost`: for each goal, the step at which it first held
     and, if it stopped holding later, the first step at which it no longer
     held (`0@90`);
   - `robot_touch`: for each object, the spells in which the robot touched
     it (`name 40-68`; a spell still open at the end runs to the last step);
   - `end_moving`: objects still moving in the last row, with their speed;
   - `frame_steps`: the steps worth seeing (the start, each switch, the tail,
     each first hold and loss, the start and end of each touch, and the end),
     each moved to the nearest step that has a frame. At most eight, in
     order.
5. For each episode with video, call `get_frames` with its `frame_steps` on
   the scene camera (and the wrist camera where the hand matters and
   `video_wrist` is set). Describe what each frame shows in plain words, then
   compare with the row: does the frame show what the records say (the object
   where the goal says, the hand where contact says)? Where they disagree,
   trust neither until a closer look settles it.
6. Where a moment needs a closer look, run `timeline` for that episode over a
   short range of steps (`from_step`, `to_step`, `every` 1 to 5). Each row is
   a control step: the phase, program step and instruction, the gripper
   command and opening, the hand's height, which goals hold, which objects
   the robot touches, which objects move, and whether the step has a frame.
   Rows at the regular spacing are joined by every step at which something
   changes (a switch, the tail, a gripper command, a goal, a touch), so a
   change between two regular rows is never missed.
7. To compare objects at two moments (before and after a switch, the moment
   a goal held against the end), run `objects_at` with those steps: each
   object's position, how far it moved and turned since the episode's first
   row, its speeds and its contacts.
8. Write a card per episode to memory (`memory_write`, the full updated
   notes):

   ```
   <episode id> (<job id>, <program name>, start <base_task>/<init_state>): delivered | not delivered
   what happened: <in order, with control steps: what the robot did, what each object did, which goal held when>
   where it went wrong: <program step, control step, what the records and frames show>
   evidence: <block names and the values; frame steps looked at>
   open questions: <what the records do not settle>
   ```

   A card is about one episode. It says what happened, not why the program
   is wrong; that comes from several cards together (jobs-preserve,
   jobs-transitions, jobs-practice).

## Blocks

```sql
-- helper: runs
WITH ev AS (
  SELECT episode_id,
    count(*) FILTER (WHERE outcome = 'timeout' AND kind <> 'tail') AS timeouts,
    count(*) FILTER (WHERE outcome = 'skipped') AS skipped
  FROM job_step_events GROUP BY episode_id
)
SELECT j.job_id, left(any_value(j.program_digest), 12) AS digest,
  count(DISTINCT j.program_digest) AS programs, count(*) AS episodes,
  count(*) FILTER (WHERE j.delivered) AS delivered,
  count(*) FILTER (WHERE NOT j.delivered) AS not_delivered,
  count(*) FILTER (WHERE NOT j.order_kept) AS order_broken,
  CAST(sum(coalesce(ev.timeouts, 0)) AS INT) AS step_timeouts,
  CAST(sum(coalesce(ev.skipped, 0)) AS INT) AS steps_skipped,
  count(*) FILTER (WHERE j.tail_steps > 0) AS with_tail,
  min(j.control_steps) AS min_steps, CAST(median(j.control_steps) AS INT) AS median_steps,
  max(j.control_steps) AS max_steps,
  count(*) FILTER (WHERE coalesce(j.video_scene, '') <> '') AS with_video
FROM job_episodes j LEFT JOIN ev USING (episode_id)
GROUP BY j.job_id
ORDER BY j.job_id
```

```sql
-- helper: moments
WITH params AS (SELECT 0.01 AS moving_above),
-- input: picked(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')),
picked AS (SELECT episode_id FROM job_episodes WHERE NOT delivered ORDER BY episode_id LIMIT 5),
-- end input
ep AS (SELECT j.* FROM job_episodes j WHERE j.episode_id IN (SELECT episode_id FROM picked)),
last_row AS (
  SELECT episode_id, max(step) AS last_step FROM robot_steps
  WHERE episode_id IN (SELECT episode_id FROM picked) GROUP BY episode_id
),
sw AS (
  SELECT e.episode_id,
    string_agg(CASE WHEN e.kind = 'tail' THEN 'tail@' || CAST(e.start_step AS VARCHAR)
      ELSE CAST(e.program_step AS VARCHAR) || '@' || CAST(e.start_step AS VARCHAR)
        || ' after ' || coalesce(p.outcome, '?') END, ', '
      ORDER BY e.start_step) AS switches,
    list(e.start_step ORDER BY e.start_step) AS switch_steps
  FROM job_step_events e
  LEFT JOIN job_step_events p ON p.episode_id = e.episode_id AND p.program_step = e.program_step - 1
  WHERE e.episode_id IN (SELECT episode_id FROM picked) AND e.start_step > 0
    AND e.outcome <> 'skipped'
  GROUP BY e.episode_id
),
g AS (
  SELECT episode_id, goal, step, holds,
    lag(holds) OVER (PARTITION BY episode_id, goal ORDER BY step) AS before
  FROM goal_steps WHERE episode_id IN (SELECT episode_id FROM picked)
),
gfirst AS (
  SELECT episode_id, goal, min(step) FILTER (WHERE holds) AS first_held FROM g GROUP BY ALL
),
glost AS (
  SELECT g.episode_id, g.goal, min(g.step) AS lost_at
  FROM g JOIN gfirst f USING (episode_id, goal)
  WHERE g.before AND NOT g.holds AND g.step > f.first_held GROUP BY ALL
),
gs AS (
  SELECT f.episode_id,
    string_agg(CAST(f.goal AS VARCHAR) || '@' || coalesce(CAST(f.first_held AS VARCHAR), 'never'), ' '
      ORDER BY f.goal) AS first_held,
    string_agg(CAST(f.goal AS VARCHAR) || '@' || CAST(l.lost_at AS VARCHAR), ' '
      ORDER BY f.goal) FILTER (WHERE l.lost_at IS NOT NULL) AS lost,
    list(f.first_held) FILTER (WHERE f.first_held IS NOT NULL) AS held_steps,
    list(l.lost_at) FILTER (WHERE l.lost_at IS NOT NULL) AS lost_steps
  FROM gfirst f LEFT JOIN glost l USING (episode_id, goal) GROUP BY f.episode_id
),
o AS (
  SELECT episode_id, object, step, in_contact_with_robot AS touch,
    lag(in_contact_with_robot) OVER (PARTITION BY episode_id, object ORDER BY step) AS before
  FROM object_steps WHERE episode_id IN (SELECT episode_id FROM picked)
),
spells AS (
  SELECT *, sum(CASE WHEN touch AND NOT coalesce(before, false) THEN 1 ELSE 0 END)
    OVER (PARTITION BY episode_id, object ORDER BY step) AS spell
  FROM o
),
touch AS (
  SELECT episode_id, object, spell, min(step) AS first_step, max(step) AS last_step
  FROM spells WHERE touch GROUP BY ALL
),
ts AS (
  SELECT episode_id,
    string_agg(object || ' ' || CAST(first_step AS VARCHAR) || '-' || CAST(last_step AS VARCHAR), ', '
      ORDER BY first_step) AS robot_touch,
    list(first_step) || list(last_step) AS touch_steps
  FROM touch GROUP BY episode_id
),
moving AS (
  SELECT o.episode_id,
    string_agg(o.object || ' ' || CAST(round(o.linear_speed::DOUBLE, 3) AS VARCHAR) || ' m/s', ', ') AS end_moving
  FROM object_steps o JOIN last_row l ON l.episode_id = o.episode_id AND l.last_step = o.step, params p
  WHERE o.linear_speed > p.moving_above GROUP BY o.episode_id
),
wanted AS (
  SELECT e.episode_id, unnest(list_distinct(list_filter(
      [0, l.last_step] || coalesce(sw.switch_steps, []) || coalesce(gs.held_steps, [])
        || coalesce(gs.lost_steps, []) || coalesce(ts.touch_steps, []),
      x -> x IS NOT NULL))) AS want
  FROM ep e JOIN last_row l USING (episode_id)
  LEFT JOIN sw USING (episode_id) LEFT JOIN gs USING (episode_id) LEFT JOIN ts USING (episode_id)
),
framed AS (
  SELECT episode_id, step FROM robot_steps
  WHERE episode_id IN (SELECT episode_id FROM picked) AND frame_index IS NOT NULL
),
snapped AS (
  SELECT w.episode_id, arg_min(f.step, abs(f.step - w.want)) AS step
  FROM wanted w JOIN framed f USING (episode_id) GROUP BY w.episode_id, w.want
),
frames AS (
  SELECT episode_id, list_sort(list_distinct(list(step))) AS candidates FROM snapped GROUP BY episode_id
)
SELECT e.episode_id, e.job_id, left(e.program_digest, 12) AS digest, e.init_state, e.delivered,
  e.delivery_detail, e.goals_held_at_end, e.order_kept, l.last_step, sw.switches, gs.first_held,
  gs.lost, ts.robot_touch, moving.end_moving,
  coalesce(e.video_scene, '') <> '' AS has_video,
  CASE WHEN f.candidates IS NULL THEN NULL WHEN len(f.candidates) <= 8 THEN f.candidates
    ELSE list_sort(list_distinct([f.candidates[1]] || list_transform(range(1, 7),
      i -> f.candidates[1 + CAST(round(i * (len(f.candidates) - 1) / 7.0) AS INT)]) || [f.candidates[-1]]))
    END AS frame_steps
FROM ep e JOIN last_row l USING (episode_id)
LEFT JOIN frames f USING (episode_id)
LEFT JOIN sw USING (episode_id) LEFT JOIN gs USING (episode_id)
LEFT JOIN ts USING (episode_id) LEFT JOIN moving USING (episode_id)
ORDER BY e.job_id, e.episode_id
```

```sql
-- helper: timeline
WITH params AS (SELECT 20 AS every, 0 AS from_step, 100000 AS to_step, 0.01 AS moving_above),
-- input: picked(episode_id) AS (VALUES ('<episode id>')),
picked AS (SELECT min(episode_id) AS episode_id FROM job_episodes),
-- end input
one AS (SELECT min(episode_id) AS episode_id FROM picked),
r AS (
  SELECT r.step, r.phase, r.program_step, r.instruction, r.gripper_command, r.gripper_opening,
    r.eef_pos[3] AS hand_z, r.frame_index IS NOT NULL AS has_frame
  FROM robot_steps r WHERE r.episode_id = (SELECT episode_id FROM one)
),
g AS (
  SELECT step, string_agg(CAST(goal AS VARCHAR) || CASE WHEN holds THEN ' yes' ELSE ' no' END, ', '
    ORDER BY goal) AS goals
  FROM goal_steps WHERE episode_id = (SELECT episode_id FROM one) GROUP BY step
),
o AS (
  SELECT o.step,
    coalesce(string_agg(o.object, ', ' ORDER BY o.object) FILTER (WHERE o.in_contact_with_robot), '') AS robot_touches,
    coalesce(string_agg(o.object || ' ' || CAST(round(o.linear_speed::DOUBLE, 3) AS VARCHAR), ', ' ORDER BY o.object)
      FILTER (WHERE o.linear_speed > p.moving_above), '') AS moving
  FROM object_steps o, params p WHERE o.episode_id = (SELECT episode_id FROM one) GROUP BY o.step
),
joined AS (
  SELECT r.*, g.goals, o.robot_touches, o.moving,
    lag(r.program_step) OVER w AS prev_program_step, lag(r.phase) OVER w AS prev_phase,
    lag(r.gripper_command) OVER w AS prev_command,
    lag(g.goals) OVER w AS prev_goals, lag(o.robot_touches) OVER w AS prev_touches,
    max(r.step) OVER () AS last_step
  FROM r LEFT JOIN g USING (step) LEFT JOIN o USING (step)
  WINDOW w AS (ORDER BY r.step)
),
marked AS (
  SELECT j.*,
    concat_ws(' ',
      CASE WHEN j.step > 0 AND j.phase = 'program' AND j.program_step IS DISTINCT FROM j.prev_program_step
        THEN 'switch' END,
      CASE WHEN j.step > 0 AND j.phase = 'tail' AND j.prev_phase IS DISTINCT FROM 'tail' THEN 'tail' END,
      CASE WHEN j.step > 0 AND j.gripper_command IS NOT NULL
        AND j.gripper_command IS DISTINCT FROM j.prev_command THEN 'gripper' END,
      CASE WHEN j.step > 0 AND j.goals IS DISTINCT FROM j.prev_goals THEN 'goal' END,
      CASE WHEN j.step > 0 AND j.robot_touches IS DISTINCT FROM j.prev_touches THEN 'touch' END,
      CASE WHEN j.step = j.last_step THEN 'end' END) AS changed
  FROM joined j
)
SELECT m.step, m.phase, m.program_step, left(m.instruction, 60) AS instruction, m.gripper_command,
  round(m.gripper_opening::DOUBLE, 4) AS gripper_opening, round(m.hand_z::DOUBLE, 3) AS hand_z,
  m.goals, m.robot_touches, m.moving, m.has_frame, m.changed
FROM marked m, params p
WHERE m.step BETWEEN p.from_step AND p.to_step
  AND ((m.step - p.from_step) % p.every = 0 OR m.changed <> '')
ORDER BY m.step
```

```sql
-- helper: objects_at
WITH
-- input: picked(episode_id, step) AS (VALUES ('<episode id>', 0), ('<episode id>', 100)),
picked AS (
  SELECT e.episode_id, unnest([0, e.control_steps]) AS step
  FROM (SELECT episode_id, control_steps FROM job_episodes ORDER BY episode_id LIMIT 1) e
),
-- end input
start_pose AS (
  SELECT o.episode_id, o.object, o.position AS p0, o.quaternion AS q0
  FROM object_steps o WHERE o.step = 0 AND o.episode_id IN (SELECT episode_id FROM picked)
)
SELECT o.episode_id, o.step, o.object,
  round(o.position[1]::DOUBLE, 3) AS x, round(o.position[2]::DOUBLE, 3) AS y,
  round(o.position[3]::DOUBLE, 3) AS z,
  round(100 * sqrt(((o.position[1] - f.p0[1]) ^ 2 + (o.position[2] - f.p0[2]) ^ 2
    + (o.position[3] - f.p0[3]) ^ 2)::DOUBLE), 1) AS moved_cm,
  round(degrees(2 * acos(least(1.0, abs(o.quaternion[1] * f.q0[1] + o.quaternion[2] * f.q0[2]
    + o.quaternion[3] * f.q0[3] + o.quaternion[4] * f.q0[4])::DOUBLE))), 1) AS turned_deg,
  round(o.linear_speed::DOUBLE, 3) AS linear_speed, round(o.angular_speed::DOUBLE, 3) AS angular_speed,
  o.in_contact_with_robot, o.in_contact_with
FROM object_steps o
JOIN picked k ON k.episode_id = o.episode_id AND k.step = o.step
JOIN start_pose f ON f.episode_id = o.episode_id AND f.object = o.object
ORDER BY o.episode_id, o.step, o.object
```

Next: `jobs-preserve` for what the program kept and lost, `jobs-transitions`
for what happened at the switches, and `jobs-practice` for the loop of
changes.
