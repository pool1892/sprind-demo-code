---
name: instrument-step-response
description: Measure how a robot's hand answers a known command with a step probe, read the response's dead time, rise, settled position and start position against a robot that does well, and tell apart a fixed command delay, a delay that keeps growing, an arm that cannot reach its targets and a motion path that is fine.
when_to_use: When some robots or sites fail more in motions that need timely corrections, when the motion records show the hand trailing its commands, or when two places show a similar lag and their causes must be told apart before a remedy is chosen. Needs the probe tools (request_probe, probe_results).
tools_used: [query, request_probe, probe_results, request_experiment]
inputs:
  - robots under suspicion, and a robot that does well, ideally one that works in the same scene
  - each robot's alignment lag from the motion records, if it has been computed
outputs:
  - per robot, the shape of its response: shifted whole, shifted by a growing amount, prompt but short or low, or prompt and like the robot that does well
  - the remedy class each shape suggests, and the task experiment that decides
version: "1"
---
# Step response

A step probe commands a known motion and records what the hand does. In a
task episode the policy keeps changing its commands, so a lag between
command and hand mixes the robot with the policy's plan. A step is a command
whose answer is known: on a healthy robot the hand starts toward the new
target at once, rises smoothly, perhaps overshoots a little, and settles on
it. How a robot departs from that answer tells where, between the command and
the hand, the trouble lies. Compare every robot with the same probe on a
robot that does well; the absolute numbers belong to the arm and its
controller, and only the difference is evidence.

## What to read in a response

For each commanded step (each change of `target_pos` in `probe_steps`; the
data dictionary gives the timing: the state is measured first, then the
command is issued, and a command acts in the next row), read four things:

- **Dead time**: rows from the row the step was commanded in to the first row
  in which the hand has clearly moved toward it (a millimetre is clear). A
  healthy robot moves in the next row or so.
- **Rise**: rows until the hand has covered half of the step, and from a
  tenth to nine tenths of it: the speed of the arm and its controller.
- **Settled position**: measured minus commanded on the step's axis just
  before the next step is commanded, and on the other axes, since a hand that
  sinks shows in z.
- **Start position**: where the hand was in row 0, against a robot that does
  well in the same scene. The probe's targets are offsets from this position,
  so an arm that already sags when the probe starts hides part of its sag
  from the targets.

Dead time and half-way alone are not enough: an arm that starts on time and
rises at a normal speed can still end well short of every target. Read where
the hand settles and where it starts as well.

## Four shapes

Each shape comes with the explanations an engineer would consider first and
the class of remedy it suggests. Which remedy works, and how well, is for a
task experiment to decide; the shape only says where to look.

1. **Shifted whole.** The response has the healthy shape but starts later,
   by the same number of rows at every step and in every repetition.
   Something on the command path holds every command back by a fixed time,
   such as a slow link between the policy and the robot or a command buffer.
   Remedy class: remove the delay where it can be removed, otherwise adapt to
   it with settings that act on the command stream.
2. **Shifted by a growing amount.** The delay is longer at a later step than
   at an earlier one of the same probe, and differs between repetitions.
   Commands queue up faster than they are executed and the queue never
   catches up, as when a computer between the policy and the drives falls
   behind. Remedy class: service. Whether adapting the command stream helps
   against a delay that keeps changing is for an experiment to show.
3. **Prompt but short or low.** The hand starts on time and rises about as
   fast as on a robot that does well, but settles short of each target,
   usually lower than commanded, and may already start lower than the other
   robot's hand in the same scene. The drives cannot deliver what is
   commanded, as with weak motors or losses in a gearbox. Remedy class:
   service.
4. **Prompt and like the robot that does well.** The motion path is fine. If
   the task still fails on this robot, the cause lies before the motion: what
   the robot sees (images that arrive late, a camera that moved, images
   prepared differently) or the scene itself. Remedy class: look at the
   images, together with the change log and the health records; changing how
   commands are sent addresses a motion path that is not at fault.

The probe shows the motion path, not the task. It selects candidate causes; a
task experiment on the robots concerned, the remedy against the current state
on the same starts, decides.

## The alignment lag from the motion records

The lag that best aligns commanded and measured hand positions over task
episodes is a cheap first look: a healthy robot shows its controller's own
tracking lag, and commands that arrive late add to it. As one number per
robot it cannot tell a fixed delay from a growing one, because a backlog that
grows within each episode also averages out to some lag. Read it per episode,
and in an early and a late window of long episodes: a fixed delay gives the
same lag in every episode and in both windows; a growing one gives a spread of
lags, a later window that lags more than an earlier one, and a larger residual
distance at the best lag. An arm that falls short shows little lag but a
steady offset between command and hand, often downward.

## Procedure

1. From the records, find where the hand trails its commands (the alignment
   lag per robot) and where tasks fail. Pick a suspect robot at each place
   and a robot that does well, ideally in the same scene.
2. Compose a step probe with `request_probe`: a short hold, a step of a few
   centimetres on one axis, a step back, a short hold. Use at least two
   steps, spaced wider than the longest delay you expect, because a late
   response runs into the next step; two or three repetitions per robot; the
   same program on every robot. Name the probe and say what you expect.
3. In the next session read `probe_steps` per robot and step: dead time,
   rise, settled position and start position. Compare the first step with the
   second and the repetitions with each other.
4. Name each robot's shape and the remedy class it suggests, and test the
   remedy with `request_experiment` on the robots concerned before adopting
   it.
5. If the question will come back on another fleet, keep the probe: a learned
   skill can carry it, with its robots as parameters, and the query that reads
   it with `$probe`.

One way to read dead time and half-way for every step on x, per robot and
repetition (replace `$probe` by the probe's id in single quotes and run it
with `query`; for another axis use `target_pos[2]` and `eef_pos[2]`, or
`[3]` for z):

```sql
WITH s AS (
  SELECT e.robot_id, e.repeat, st.episode_id, st.step,
         st.target_pos[1] AS target_x, st.eef_pos[1] AS hand_x,
         lag(st.target_pos[1]) OVER (PARTITION BY st.episode_id ORDER BY st.step) AS target_before
  FROM probe_steps AS st JOIN probe_episodes AS e USING (episode_id)
  WHERE e.probe_id = $probe
),
changes AS (
  SELECT episode_id, robot_id, repeat, step AS issued, target_x - target_before AS change,
         hand_x AS hand_at_issue
  FROM s
  WHERE target_before IS NOT NULL AND abs(target_x - target_before) > 1e-6
)
SELECT c.robot_id, c.repeat, c.issued, round(c.change * 1000, 1) AS change_mm,
       min(s.step) FILTER (WHERE sign(c.change) * (s.hand_x - c.hand_at_issue) >= 0.001)
         - c.issued AS first_move_rows,
       min(s.step) FILTER (WHERE sign(c.change) * (s.hand_x - c.hand_at_issue) >= abs(c.change) / 2)
         - c.issued AS half_way_rows
FROM changes AS c JOIN s ON s.episode_id = c.episode_id AND s.step > c.issued
GROUP BY c.robot_id, c.repeat, c.issued, c.change
ORDER BY c.robot_id, c.repeat, c.issued
```
