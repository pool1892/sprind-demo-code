---
name: jobs-preserve
description: Check that the later steps of a job keep what the earlier steps achieved, and find the moment a goal was lost. For every episode and requested goal, when it first held, whether it held at the end and when it stopped holding; for every loss, the program step, the instruction in force and what the robot and the objects were doing just before; and the order rules recomputed from the trace.
when_to_use: After a job run has come back, before changing the program, and whenever an episode ends with fewer goals holding than it achieved along the way. Also to confirm that a change kept every goal it should keep.
tools_used: [query, get_frames, operator_state, memory_read, memory_write]
inputs:
  - the job ids of runs that have come back
  - the requested goals and order rules (operator_state)
outputs:
  - per job run, how many episodes lost a goal after achieving it, which goal, in which program step
  - for each loss, its moment and what happened in the steps before it, in memory
  - the order rules as the trace shows them, beside the host's verdict
version: "1"
---
# Does a later step keep what an earlier one achieved?

A job is several steps in one scene, and each step changes the scene the next
one starts from. A step can achieve its goal and a later step can undo it:
a moved object takes another with it or knocks it over, the hand carries away
something it should have left, or something that only just held stops holding
once nobody holds it. The delivery check at the end says whether
every goal holds then. This skill says when each goal held along the way and,
when one was lost, the exact control step and what was going on.

Run the blocks as `jobs-traces` describes: copy a block whole, change only
its `params` values and the rows between `-- input` and `-- end input`, run it
with `query`. Goals are numbered as the request lists them, from 0
(`operator_state`).

## Procedure

1. Run `goal_timeline` on the job runs to check (the default takes the first
   run). One row per episode and goal:
   - `first_held`: the first control step at which the goal held; empty if
     it never did;
   - `held_at_end`: whether it holds in the episode's last row;
   - `losses`: how many times it went from holding to not holding after it
     first held; `first_lost` is the first of those steps, and
     `lost_in_phase`, `lost_in_step` and `lost_under` are the phase
     (`program` or `tail`), the program step and the instruction in force
     then (a loss in the delivery tail has no program step);
   - `held_share`: the share of the steps from `first_held` to the end in
     which it held. A goal that comes and goes (low share, several losses)
     differs from one that was achieved once and then undone (one loss, held
     before, not after).
   - `achieved_then_lost`: held at some point, not at the end.
2. Run `losses_by_step` for the summary per job run: for each goal and
   program step (and the delivery tail, `tail`), in how many episodes the
   goal was first lost during that step, out of how many in which it had
   held by then. These are counts of a few episodes; report them as counts.
3. Run `loss_context` for the losses: one row per first loss, with the
   window of control steps before it (`window` steps, default 20, one
   second). It shows which objects the robot touched in the window, which
   objects moved more than `moved_cm` in it and by how much, how far the
   hand moved, the gripper commands issued, and whether the objects the goal
   names still touch each other at the loss (`in_contact_with`). Read it
   with the goal's predicate in mind: a goal that needs one object on or in
   another is lost when they part or when one is no longer where the
   predicate wants it.
4. Look at frames around each loss (`get_frames` at steps with a frame near
   the window's first step, its middle and the loss step; `timeline` in
   `jobs-traces` shows which steps have one) and write what they show. Add to the
   episode's card in memory: the goal, the loss step, the program step and
   instruction, and what the robot and the objects did. Read `jobs-traces`
   `timeline` over the window when the frames leave it open.
5. Group the losses. The same goal lost in the same program step in most
   episodes, with the same thing going on in the window, is one pattern of
   the program; scattered losses with different contexts are not. Write the
   patterns to memory with their counts, and the episodes that show each.
6. Run `order_check` with the request's order rules as input rows (from
   `operator_state`, each rule's `earlier_goal` and `later_goal`). It
   recomputes each rule from `first_held`, one row per episode and rule
   (`from_trace`). The host's `order_kept` is one verdict per episode for all
   its rules together (`order_detail` gives it in words), so a kept rule can
   stand beside a false `order_kept` when another rule of the same episode
   was broken. Compare `order_kept` with `all_kept_from_trace`, which says
   whether every recomputed rule of the episode is kept, not with each row.
   They should agree; where they do not, trust the host's verdict (it decides
   whether a program can be saved) and find out why from the trace.

What this skill does not decide: why a goal was lost. The context of a loss
gives candidates (`jobs-transitions` for the moment of a switch), and a change
to the program tested on the same starts settles which candidate it was
(`jobs-practice`).

## Blocks

```sql
-- helper: goal_timeline
WITH
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT min(job_id) AS job_id FROM job_episodes),
-- end input
ep AS (SELECT job_id, episode_id, delivered FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
g AS (
  SELECT g.episode_id, g.goal, g.step, g.holds,
    lag(g.holds) OVER (PARTITION BY g.episode_id, g.goal ORDER BY g.step) AS before,
    max(g.step) OVER (PARTITION BY g.episode_id) AS last_step
  FROM goal_steps g WHERE g.episode_id IN (SELECT episode_id FROM ep)
),
firsts AS (
  SELECT episode_id, goal, min(step) FILTER (WHERE holds) AS first_held, max(last_step) AS last_step
  FROM g GROUP BY ALL
),
per AS (
  SELECT g.episode_id, g.goal, f.first_held, f.last_step,
    bool_or(g.holds AND g.step = g.last_step) AS held_at_end,
    count(*) FILTER (WHERE g.before AND NOT g.holds AND g.step > f.first_held) AS losses,
    min(g.step) FILTER (WHERE g.before AND NOT g.holds AND g.step > f.first_held) AS first_lost,
    count(*) FILTER (WHERE g.holds AND g.step >= f.first_held) AS held_steps
  FROM g JOIN firsts f USING (episode_id, goal)
  GROUP BY g.episode_id, g.goal, f.first_held, f.last_step
)
SELECT e.job_id, p.episode_id, p.goal, p.first_held, p.held_at_end, CAST(p.losses AS INT) AS losses,
  p.first_lost, r.phase AS lost_in_phase, r.program_step AS lost_in_step,
  left(r.instruction, 60) AS lost_under,
  round(p.held_steps / nullif(p.last_step - p.first_held + 1, 0), 2) AS held_share,
  p.first_held IS NOT NULL AND NOT p.held_at_end AS achieved_then_lost, e.delivered
FROM per p JOIN ep e USING (episode_id)
LEFT JOIN robot_steps r ON r.episode_id = p.episode_id AND r.step = p.first_lost - 1
ORDER BY e.job_id, p.episode_id, p.goal
```

```sql
-- helper: losses_by_step
WITH
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT job_id FROM job_episodes GROUP BY job_id),
-- end input
ep AS (SELECT job_id, episode_id FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
g AS (
  SELECT g.episode_id, g.goal, g.step, g.holds,
    lag(g.holds) OVER (PARTITION BY g.episode_id, g.goal ORDER BY g.step) AS before
  FROM goal_steps g WHERE g.episode_id IN (SELECT episode_id FROM ep)
),
firsts AS (SELECT episode_id, goal, min(step) FILTER (WHERE holds) AS first_held FROM g GROUP BY ALL),
lost AS (
  SELECT g.episode_id, g.goal, min(g.step) AS first_lost
  FROM g JOIN firsts f USING (episode_id, goal)
  WHERE g.before AND NOT g.holds AND g.step > f.first_held GROUP BY ALL
),
steps AS (
  SELECT s.episode_id, coalesce(CAST(s.program_step AS VARCHAR), 'tail') AS program_step, s.kind,
    s.start_step, s.end_step
  FROM job_step_events s
  WHERE s.episode_id IN (SELECT episode_id FROM ep) AND s.outcome <> 'skipped'
)
SELECT e.job_id, f.goal, s.program_step, any_value(s.kind) AS kind,
  count(*) FILTER (WHERE f.first_held <= s.end_step) AS held_by_then,
  count(*) FILTER (WHERE l.first_lost > s.start_step AND l.first_lost <= s.end_step) AS first_lost_here
FROM firsts f JOIN ep e USING (episode_id)
JOIN steps s ON s.episode_id = f.episode_id
LEFT JOIN lost l ON l.episode_id = f.episode_id AND l.goal = f.goal
GROUP BY e.job_id, f.goal, s.program_step
ORDER BY e.job_id, f.goal, min(s.start_step)
```

```sql
-- helper: loss_context
WITH params AS (SELECT 20 AS window, 1.0 AS moved_cm),
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT min(job_id) AS job_id FROM job_episodes),
-- end input
ep AS (SELECT job_id, episode_id FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
g AS (
  SELECT g.episode_id, g.goal, g.step, g.holds,
    lag(g.holds) OVER (PARTITION BY g.episode_id, g.goal ORDER BY g.step) AS before
  FROM goal_steps g WHERE g.episode_id IN (SELECT episode_id FROM ep)
),
firsts AS (SELECT episode_id, goal, min(step) FILTER (WHERE holds) AS first_held FROM g GROUP BY ALL),
lost AS (
  SELECT g.episode_id, g.goal, min(g.step) AS loss_step
  FROM g JOIN firsts f USING (episode_id, goal)
  WHERE g.before AND NOT g.holds AND g.step > f.first_held GROUP BY ALL
),
win AS (
  SELECT l.*, greatest(0, l.loss_step - p.window) AS from_step FROM lost l, params p
),
touched AS (
  SELECT w.episode_id, w.goal,
    string_agg(DISTINCT o.object, ', ' ORDER BY o.object) FILTER (WHERE o.in_contact_with_robot) AS robot_touched
  FROM win w JOIN object_steps o ON o.episode_id = w.episode_id AND o.step BETWEEN w.from_step AND w.loss_step
  GROUP BY ALL
),
moved AS (
  SELECT w.episode_id, w.goal, a.object,
    100 * sqrt(((b.position[1] - a.position[1]) ^ 2 + (b.position[2] - a.position[2]) ^ 2
      + (b.position[3] - a.position[3]) ^ 2)::DOUBLE) AS cm,
    100 * (b.position[3] - a.position[3])::DOUBLE AS rise_cm
  FROM win w
  JOIN object_steps a ON a.episode_id = w.episode_id AND a.step = w.from_step
  JOIN object_steps b ON b.episode_id = w.episode_id AND b.step = w.loss_step AND b.object = a.object
),
moved_list AS (
  SELECT m.episode_id, m.goal,
    string_agg(m.object || ' ' || CAST(round(m.cm, 1) AS VARCHAR) || ' cm (up '
      || CAST(round(m.rise_cm, 1) AS VARCHAR) || ')', ', ' ORDER BY m.cm DESC) AS objects_moved
  FROM moved m, params p WHERE m.cm > p.moved_cm GROUP BY ALL
),
hand AS (
  SELECT w.episode_id, w.goal,
    100 * sqrt(((b.eef_pos[1] - a.eef_pos[1]) ^ 2 + (b.eef_pos[2] - a.eef_pos[2]) ^ 2
      + (b.eef_pos[3] - a.eef_pos[3]) ^ 2)::DOUBLE) AS hand_cm,
    100 * (b.eef_pos[3] - a.eef_pos[3])::DOUBLE AS hand_rise_cm
  FROM win w
  JOIN robot_steps a ON a.episode_id = w.episode_id AND a.step = w.from_step
  JOIN robot_steps b ON b.episode_id = w.episode_id AND b.step = w.loss_step
),
grip AS (
  SELECT w.episode_id, w.goal,
    string_agg(DISTINCT CAST(r.gripper_command AS VARCHAR), ' ') AS gripper_commands
  FROM win w JOIN robot_steps r ON r.episode_id = w.episode_id AND r.step BETWEEN w.from_step AND w.loss_step - 1
  GROUP BY ALL
),
at_loss AS (
  SELECT w.episode_id, w.goal,
    string_agg(o.object || ': ' || array_to_string(o.in_contact_with, ' '), '; ' ORDER BY o.object) AS contacts_at_loss
  FROM win w JOIN object_steps o ON o.episode_id = w.episode_id AND o.step = w.loss_step
  GROUP BY ALL
)
SELECT e.job_id, w.episode_id, w.goal, w.loss_step, w.from_step,
  r.phase, r.program_step AS in_step, s.kind, w.loss_step - s.start_step AS steps_into_step,
  left(r.instruction, 60) AS instruction, t.robot_touched, ml.objects_moved,
  round(h.hand_cm, 1) AS hand_moved_cm, round(h.hand_rise_cm, 1) AS hand_rise_cm,
  gr.gripper_commands, al.contacts_at_loss
FROM win w JOIN ep e USING (episode_id)
LEFT JOIN robot_steps r ON r.episode_id = w.episode_id AND r.step = w.loss_step - 1
LEFT JOIN job_step_events s ON s.episode_id = w.episode_id
  AND w.loss_step > s.start_step AND w.loss_step <= s.end_step
LEFT JOIN touched t USING (episode_id, goal) LEFT JOIN moved_list ml USING (episode_id, goal)
LEFT JOIN hand h USING (episode_id, goal) LEFT JOIN grip gr USING (episode_id, goal)
LEFT JOIN at_loss al USING (episode_id, goal)
ORDER BY e.job_id, w.episode_id, w.goal
```

```sql
-- helper: order_check
WITH
-- input: rules(earlier_goal, later_goal) AS (VALUES (0, 1)),
rules AS (SELECT 0 AS earlier_goal, 1 AS later_goal),
-- end input
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT job_id FROM job_episodes GROUP BY job_id),
-- end input
ep AS (
  SELECT job_id, episode_id, order_kept, order_detail FROM job_episodes
  WHERE job_id IN (SELECT job_id FROM runs)
),
firsts AS (
  SELECT episode_id, goal, min(step) FILTER (WHERE holds) AS first_held
  FROM goal_steps WHERE episode_id IN (SELECT episode_id FROM ep) GROUP BY ALL
)
SELECT e.job_id, e.episode_id, r.earlier_goal, a.first_held AS earlier_goal_held, r.later_goal,
  b.first_held AS later_goal_held,
  CASE WHEN b.first_held IS NULL THEN 'kept'
    WHEN a.first_held IS NULL OR b.first_held < a.first_held THEN 'broken' ELSE 'kept' END AS from_trace,
  bool_and(b.first_held IS NULL OR (a.first_held IS NOT NULL AND b.first_held >= a.first_held))
    OVER (PARTITION BY e.job_id, e.episode_id) AS all_kept_from_trace,
  e.order_kept AS host_verdict, e.order_detail AS host_detail
FROM ep e CROSS JOIN rules r
LEFT JOIN firsts a ON a.episode_id = e.episode_id AND a.goal = r.earlier_goal
LEFT JOIN firsts b ON b.episode_id = e.episode_id AND b.goal = r.later_goal
ORDER BY e.job_id, e.episode_id, r.earlier_goal, r.later_goal
```

Next: `jobs-transitions` for losses that start at or just after a switch,
`jobs-check` to keep what you check here as a saved check, and
`jobs-practice` to test a change.
