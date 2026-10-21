---
name: jobs-transitions
description: How a job passes from one step to the next and how to examine it. What ends a step, what a goal predicate does and does not establish, why a task's own success signal can fire before the work is finished (an object still held, still moving, not yet resting on its own), what the policy faces after a switch, the transitions the program vocabulary can express with what each risks, and blocks that measure the state of the scene at every switch and how long it takes to come to rest.
when_to_use: When composing a program's transitions, and after a run whenever losses, failures or timeouts start at or just after a switch between steps. Also before choosing what to change at a switch, to know what the records say happened there.
tools_used: [query, get_frames, operator_state, memory_read, memory_write]
inputs:
  - the job ids of runs that have come back
  - the objects the requested goals name (operator_state)
outputs:
  - for each switch of each episode, the state of the scene at the switch and what became of the goals that held then
  - per job run and switch, counts of switches made in each state
  - the steps the named objects took to come to rest after each switch, where they did
version: "1"
---
# Transitions between the steps of a job

A single task ends when its goal is reached, and a score for that task needs
no more than that. In a job the scene goes on: the next step starts from
whatever the last one left, and anything the last one left unfinished is now
part of the next step's problem. The transition (when one step ends, and what
happens before the next begins) is part of the program you write.

## What ends a step

- A **policy step** runs the policy under its instruction until its `until`
  condition holds or `max_steps` runs out (`outcome` `timeout`). With
  `until: {goal: i}` it ends at the first control step at which requested
  goal `i` holds; with `until: {steps: n}` after `n` control steps. After a
  timeout, the step's `on_timeout` either ends the program (`stop`; the later
  steps are `skipped`) or goes on with the next step (`continue`).
- A **motion step** plays one segment of the probe vocabulary (hold, step,
  ramp, sine, open_gripper, close_gripper, with the probe bounds: the target
  at most a few centimetres from where the hand was when the step started,
  limited speed and duration), from the hand's position and orientation as
  measured at the step's start. The gripper keeps its last command through
  hold, step, ramp and sine; open_gripper and close_gripper change it. The
  step ends when the segment ends.
- A **wait** holds the robot's last command for its steps; the scene goes on
  under physics.
- A **home step** commands the gripper open and ramps the hand back to the
  pose the robot reported when the job started, then holds until the hand is
  within 1 cm of it and at rest (`reached`) or its `duration_steps` run out
  (`timeout`, then `on_timeout` as for a policy step).
- At every boundary the runner discards whatever actions the policy had
  planned and not yet executed. The next policy step starts from a fresh
  prediction under its own instruction, from the scene exactly as it is.
- After the program, when its last step is a policy step with outcome
  `reached` (its `until` goal held or its `until.steps` ran out), the
  installation runs the delivery tail described in `jobs-traces`,
  the same for every program. A program that ends with a motion or a wait, or
  with a timeout, has no tail.

## What a goal predicate establishes, and what it does not

The requested goals are the benchmark's own predicates, evaluated on the
physics state at one instant. In the predicate definitions the benchmark
publishes: `In a b` holds when `a` touches `b` and `a` lies inside `b`'s
containing region; `On a b` when `a` touches `b`, is not below it and is
centred close to it horizontally. Where `b` is a region of the table or of a
fixture (a name such as `..._region` or `..._side`), `On a b` holds when `a`
is over the region and touches what the region belongs to, and `In a b` when
`a` lies inside the region. `Open`, `Close`, `Turnon` and `Turnoff` read a
fixture's joint state; `Up a` reads an object's height. Read the definition
of each goal of your request before you rely on it.

None of these asks whether the robot still holds or touches the object,
whether anything is still moving, or whether the state will last once nothing
holds it. A placement in the ordinary engineering sense is finished when the
object has been let go, the hand is clear of it, and the object rests where it
should under its own weight. A predicate can therefore hold:

- while the object is still in the gripper, held where the predicate wants
  it;
- while the object is still sliding, rocking or settling, before it comes to
  rest, possibly somewhere else;
- for a moment only, as the object passes through the region;
- in a state that depends on the hand's support and ends when the support
  goes.

So "the goal holds now" is a different question from "the work of this step
is done and the next may start". Which of the two a transition waits for is a
choice in the program, and the records tell you which one each switch
actually got: `object_steps.in_contact_with_robot`, `linear_speed` and
`angular_speed` at the switch, and whether the goal still held later.

## What the policy faces after a switch

The policy learned its behaviour from its training data, one task at a time.
What it does when its own task's goal already holds, and what it does under a
new instruction when it still holds something, are not part of any single
task, and nothing guarantees one behaviour or the other. Do not assume it
opens the fingers before it moves on, or that it does not; read it in the
traces (`gripper_command`, `gripper_opening` and contact in the steps after a
switch) and in the frames. Whatever the hand holds when it starts the next
task goes where the hand goes until the fingers open.

## The transitions the program can express

These are the means the vocabulary gives you. They are listed with the
question each answers and what each can cost; they are not ranked, and no one
of them is a default. Which, if any, a job needs is a question for its
records.

| Transition | What it does | What it can cost |
| --- | --- | --- |
| switch on the goal | the next step starts the first control step the goal holds | starts the next step in whatever state the goal was first reached in |
| switch on a step count | ends the step after a fixed number of control steps | ignores whether the goal holds; too short cuts the work off, too long spends steps |
| a wait after the step | lets motion die out with no new command | the hand stays where it was and keeps whatever it holds; steps spent |
| open the gripper (motion) | lets go of what the fingers hold | an object let go before it is supported falls or tips |
| move the hand (motion: step or ramp, within the bounds) | moves the hand away from where it was | anything still held or snagged moves with it |
| return the hand to the start pose (home) | brings the hand back to the pose it had when the job started, gripper open | lets go of whatever is held where the hand is; the way back can pass through the scene; steps spent |
| run a policy step again | a second step with the same instruction and the same goal | see in `job_step_events` how a step behaves when its goal already holds at its start; it may disturb what holds |
| another instruction wording | the instruction is free text | the policy was trained on its own instructions; a new wording is a change like any other and is tested like one |
| another order of the steps | the policy's skills chained in another order | allowed only where the order rules allow it; each order leaves a different scene for the next step |

A transition can combine several of these. Change one element at a time and
test it on the same starts (`jobs-practice`), so that each part's effect is
known before parts are combined.

## Procedure

Run the blocks as `jobs-traces` describes. Put the objects the requested
goals name (from `operator_state`) into the `named` input where a block has
one; the default takes every logged object. The `rest_speed` and `rest_turn`
parameters are what the blocks call at rest; set them to the thresholds the
delivery check publishes (`operator_state`).

1. Run `at_switch` on a run. One row per switch of each episode (program step
   `to_step` starting at control step `switch_step`), with the state the
   next step started in:
   - `after`: how the step before ended (`reached`, `timeout`);
   - `goals_holding`: the goals that hold at the switch;
   - `robot_touches`: the named objects the robot touches at the switch, and
     `touch_lasts`: for how many steps after the switch the robot still
     touches any of them (to the end of the next step at most; empty when it
     touched none);
   - `moving`: the named objects not at rest at the switch, with their
     speeds;
   - `gripper_before`: the last gripper command of the step that ended, and
     `opening`: the fingers' opening at the switch;
   - `lost_in_next`: goals that held at the switch and no longer hold when
     the next step ends; `lost_by_end`: those that no longer hold at the end.
2. Run `switch_summary` for counts per run and switch: of the episodes that
   reached each switch, how many switched while the robot touched a named
   object, while one moved, with the gripper last commanded closed, and how
   many lost a goal that held at the switch during the next step or by the
   end. Read the counts side by side: losses that occur mostly after switches
   made in one state, and rarely after the others, point to the state at the
   switch. A pattern in a few episodes is a hypothesis, not a finding.
3. Run `settling` to see, for each switch, how many control steps passed
   until every named object was untouched by the robot and at rest
   (`quiet_after`), and whether the goals that held at the switch still held
   then. Empty means it never happened before the episode ended. This is what
   your own runs show about how long things take to come to rest under the
   program as it stands; it changes when the program changes.
4. Look at the frames at a few switches (`get_frames` at the steps with a
   frame nearest `switch_step` and a few steps after) in episodes with and
   without a loss, and write what they show next to the counts.
5. Write to memory what the switches looked like, with the counts and the
   episodes, and the candidate change to the transition. Test it as
   `jobs-practice` describes: one change, the same starts.

## Blocks

```sql
-- helper: at_switch
WITH params AS (SELECT 0.01 AS rest_speed, 0.15 AS rest_turn),
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT min(job_id) AS job_id FROM job_episodes),
-- end input
-- input: named(object) AS (VALUES ('<object>'), ('<object>')),
named AS (SELECT DISTINCT object FROM object_steps),
-- end input
ep AS (SELECT job_id, episode_id FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
sw AS (
  SELECT e.episode_id, e.program_step AS to_step, e.kind AS to_kind, e.start_step AS switch_step,
    e.end_step AS next_end, p.outcome AS after
  FROM job_step_events e
  JOIN job_step_events p ON p.episode_id = e.episode_id AND p.program_step = e.program_step - 1
  WHERE e.episode_id IN (SELECT episode_id FROM ep) AND e.program_step > 0 AND e.outcome <> 'skipped'
),
last_row AS (
  SELECT episode_id, max(step) AS last_step FROM goal_steps
  WHERE episode_id IN (SELECT episode_id FROM ep) GROUP BY episode_id
),
at_obj AS (
  SELECT w.episode_id, w.to_step,
    string_agg(o.object, ', ' ORDER BY o.object) FILTER (WHERE o.in_contact_with_robot) AS robot_touches,
    list(o.object) FILTER (WHERE o.in_contact_with_robot) AS touched,
    string_agg(o.object || ' ' || CAST(round(o.linear_speed::DOUBLE, 3) AS VARCHAR) || ' m/s '
      || CAST(round(o.angular_speed::DOUBLE, 2) AS VARCHAR) || ' rad/s', ', ' ORDER BY o.object)
      FILTER (WHERE o.linear_speed >= p.rest_speed OR o.angular_speed >= p.rest_turn) AS moving
  FROM sw w JOIN object_steps o ON o.episode_id = w.episode_id AND o.step = w.switch_step, params p
  WHERE o.object IN (SELECT object FROM named)
  GROUP BY w.episode_id, w.to_step
),
touch_rows AS (
  SELECT episode_id, step, coalesce(list(object) FILTER (WHERE in_contact_with_robot), []) AS objs
  FROM object_steps
  WHERE episode_id IN (SELECT episode_id FROM ep) AND object IN (SELECT object FROM named)
  GROUP BY episode_id, step
),
touch_end AS (
  SELECT w.episode_id, w.to_step, min(r.step) AS untouched_from
  FROM sw w JOIN at_obj a USING (episode_id, to_step)
  JOIN touch_rows r ON r.episode_id = w.episode_id AND r.step BETWEEN w.switch_step AND w.next_end
  WHERE len(coalesce(a.touched, [])) > 0 AND NOT list_has_any(r.objs, a.touched)
  GROUP BY w.episode_id, w.to_step
),
goals AS (
  SELECT w.episode_id, w.to_step,
    list(g.goal ORDER BY g.goal) FILTER (WHERE g.step = w.switch_step AND g.holds) AS held,
    list(g.goal ORDER BY g.goal) FILTER (WHERE g.step = w.next_end AND g.holds) AS held_next,
    list(g.goal ORDER BY g.goal) FILTER (WHERE g.step = l.last_step AND g.holds) AS held_end
  FROM sw w JOIN last_row l USING (episode_id)
  JOIN goal_steps g ON g.episode_id = w.episode_id AND g.step IN (w.switch_step, w.next_end, l.last_step)
  GROUP BY w.episode_id, w.to_step
)
SELECT e.job_id, w.episode_id, w.to_step, w.to_kind, w.switch_step, w.after,
  array_to_string(coalesce(g.held, []), ' ') AS goals_holding,
  a.robot_touches,
  CASE WHEN a.touched IS NULL OR len(a.touched) = 0 THEN NULL
    ELSE coalesce(t.untouched_from, w.next_end) - w.switch_step END AS touch_lasts,
  a.moving, rb.gripper_command AS gripper_before, round(rs.gripper_opening::DOUBLE, 4) AS opening,
  array_to_string(list_filter(coalesce(g.held, []), x -> NOT list_contains(coalesce(g.held_next, []), x)), ' ')
    AS lost_in_next,
  array_to_string(list_filter(coalesce(g.held, []), x -> NOT list_contains(coalesce(g.held_end, []), x)), ' ')
    AS lost_by_end
FROM sw w JOIN ep e USING (episode_id)
LEFT JOIN at_obj a USING (episode_id, to_step) LEFT JOIN touch_end t USING (episode_id, to_step)
LEFT JOIN goals g USING (episode_id, to_step)
LEFT JOIN robot_steps rb ON rb.episode_id = w.episode_id AND rb.step = w.switch_step - 1
LEFT JOIN robot_steps rs ON rs.episode_id = w.episode_id AND rs.step = w.switch_step
ORDER BY e.job_id, w.episode_id, w.to_step
```

```sql
-- helper: switch_summary
WITH params AS (SELECT 0.01 AS rest_speed, 0.15 AS rest_turn),
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT job_id FROM job_episodes GROUP BY job_id),
-- end input
-- input: named(object) AS (VALUES ('<object>'), ('<object>')),
named AS (SELECT DISTINCT object FROM object_steps),
-- end input
ep AS (SELECT job_id, episode_id FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
sw AS (
  SELECT e.episode_id, e.program_step AS to_step, e.start_step AS switch_step, e.end_step AS next_end
  FROM job_step_events e
  WHERE e.episode_id IN (SELECT episode_id FROM ep) AND e.program_step > 0 AND e.outcome <> 'skipped'
),
last_row AS (
  SELECT episode_id, max(step) AS last_step FROM goal_steps
  WHERE episode_id IN (SELECT episode_id FROM ep) GROUP BY episode_id
),
state AS (
  SELECT w.episode_id, w.to_step,
    bool_or(o.in_contact_with_robot) AS touching,
    bool_or(o.linear_speed >= p.rest_speed OR o.angular_speed >= p.rest_turn) AS moving
  FROM sw w JOIN object_steps o ON o.episode_id = w.episode_id AND o.step = w.switch_step, params p
  WHERE o.object IN (SELECT object FROM named)
  GROUP BY w.episode_id, w.to_step
),
kept AS (
  SELECT w.episode_id, w.to_step,
    bool_or(a.holds AND NOT b.holds) AS lost_next, bool_or(a.holds AND NOT c.holds) AS lost_end
  FROM sw w JOIN last_row l USING (episode_id)
  JOIN goal_steps a ON a.episode_id = w.episode_id AND a.step = w.switch_step
  JOIN goal_steps b ON b.episode_id = w.episode_id AND b.step = w.next_end AND b.goal = a.goal
  JOIN goal_steps c ON c.episode_id = w.episode_id AND c.step = l.last_step AND c.goal = a.goal
  GROUP BY w.episode_id, w.to_step
)
SELECT e.job_id, w.to_step, count(*) AS switches,
  count(*) FILTER (WHERE s.touching) AS robot_touching,
  count(*) FILTER (WHERE s.moving) AS object_moving,
  count(*) FILTER (WHERE r.gripper_command > 0) AS gripper_closed_before,
  count(*) FILTER (WHERE k.lost_next) AS lost_goal_in_next_step,
  count(*) FILTER (WHERE k.lost_end) AS lost_goal_by_end,
  count(*) FILTER (WHERE k.lost_end AND s.touching) AS lost_by_end_and_touching,
  count(*) FILTER (WHERE k.lost_end AND NOT s.touching) AS lost_by_end_not_touching
FROM sw w JOIN ep e USING (episode_id)
LEFT JOIN state s USING (episode_id, to_step) LEFT JOIN kept k USING (episode_id, to_step)
LEFT JOIN robot_steps r ON r.episode_id = w.episode_id AND r.step = w.switch_step - 1
GROUP BY e.job_id, w.to_step
ORDER BY e.job_id, w.to_step
```

```sql
-- helper: settling
WITH params AS (SELECT 0.01 AS rest_speed, 0.15 AS rest_turn),
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT min(job_id) AS job_id FROM job_episodes),
-- end input
-- input: named(object) AS (VALUES ('<object>'), ('<object>')),
named AS (SELECT DISTINCT object FROM object_steps),
-- end input
ep AS (SELECT job_id, episode_id FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)),
sw AS (
  SELECT e.episode_id, e.program_step AS to_step, e.start_step AS switch_step
  FROM job_step_events e
  WHERE e.episode_id IN (SELECT episode_id FROM ep) AND e.program_step > 0 AND e.outcome <> 'skipped'
),
flags AS (
  SELECT o.episode_id, o.step,
    NOT o.in_contact_with_robot AND o.linear_speed < p.rest_speed AND o.angular_speed < p.rest_turn AS still
  FROM object_steps o, params p
  WHERE o.episode_id IN (SELECT episode_id FROM ep) AND o.object IN (SELECT object FROM named)
),
quiet AS (SELECT episode_id, step FROM flags GROUP BY episode_id, step HAVING bool_and(still)),
first_quiet AS (
  SELECT w.episode_id, w.to_step, w.switch_step, min(q.step) AS quiet_step
  FROM sw w LEFT JOIN quiet q ON q.episode_id = w.episode_id AND q.step >= w.switch_step
  GROUP BY ALL
),
goals AS (
  SELECT f.episode_id, f.to_step,
    list(a.goal ORDER BY a.goal) FILTER (WHERE a.holds) AS held_at_switch,
    list(a.goal ORDER BY a.goal) FILTER (WHERE a.holds AND NOT coalesce(b.holds, false)) AS gone_when_quiet
  FROM first_quiet f
  JOIN goal_steps a ON a.episode_id = f.episode_id AND a.step = f.switch_step
  LEFT JOIN goal_steps b ON b.episode_id = f.episode_id AND b.step = f.quiet_step AND b.goal = a.goal
  GROUP BY f.episode_id, f.to_step
)
SELECT e.job_id, f.episode_id, f.to_step, f.switch_step, f.quiet_step - f.switch_step AS quiet_after,
  r.phase AS quiet_in_phase, r.program_step AS quiet_in_step,
  array_to_string(coalesce(g.held_at_switch, []), ' ') AS goals_at_switch,
  CASE WHEN f.quiet_step IS NULL THEN NULL ELSE array_to_string(coalesce(g.gone_when_quiet, []), ' ') END
    AS lost_before_quiet
FROM first_quiet f JOIN ep e USING (episode_id)
LEFT JOIN goals g USING (episode_id, to_step)
LEFT JOIN robot_steps r ON r.episode_id = f.episode_id AND r.step = f.quiet_step
ORDER BY e.job_id, f.episode_id, f.to_step
```

Next: `jobs-practice` to test a change to a transition on the same starts,
and `jobs-preserve` to confirm that the change kept every goal.
