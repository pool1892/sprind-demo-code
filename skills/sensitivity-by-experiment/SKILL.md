---
name: sensitivity-by-experiment
description: Plan the round's experiments against the attempt budget that is left (every attempt of every arm is charged, the control arm and failed attempts included), state in advance the attempts each test needs from a power calculation, request them with paired starts, and read the results back into gains with intervals.
when_to_use: In a round with experiments, after sensitivity-from-logs, when a candidate's gain is uncertain enough to change what you adopt, when two candidates are confounded, when a setting's best value is unknown, or when the set you intend to adopt has not been seen together. In the next session, to read the results.
tools_used: [query, request_experiment, experiment_results, experiment_budget, memory_read, memory_write]
inputs:
  - candidate actions with their log-based gains and intervals (sensitivity-from-logs)
  - confounded pairs (scope)
  - observed success rates of the robots concerned (triage, scope)
  - the round's attempt budget and what is left of it (the task, the tools)
outputs:
  - a spending plan for the round, written to memory before the first request
  - experiment requests, each with arms, robots, attempts per arm, paired starts and a power note
  - after the experiments ran, a measured gain with an interval for each tested action or set
version: "5"
---
# Sensitivity by experiment

An experiment runs the same work on the same robots with and without a
change and compares. It is the only way to separate two changes the log
cannot tell apart, and the most direct way to measure what a change wins.
Every arm starts from the fleet as it currently is, with the changes
adopted in earlier rounds in place: the control arm (no actions) is the
current state, and a treatment arm's actions act on top of it.

## The budget is attempts, and every attempt counts

A round has a budget of robot attempts for its experiments. What a request
costs is its attempts per arm times its arms:

- **every arm is charged**, the control arm included: two arms of 40 cost
  80, four arms of 30 cost 120;
- **every attempt is charged the same**, whether it succeeds or fails;
- **an accepted request is charged in full when it is accepted**, before it
  runs; a request the tool server refuses costs nothing;
- requests of earlier rounds do not count against this round's budget.

Where the budget comes from: your task states it, the description of
`request_experiment` states the limits in force (attempts in all, attempts
per arm, a count of requests, and the session from which this round
counts), and every refusal says how many attempts are already requested.
If your tools include `experiment_budget`, call it: it reports the budget
and what is left of it (`attempts`, `attempts_used`, `attempts_left`,
`max_per_arm`, `requests`, `requests_used`, `counted_from_session`). Keep your own
count in memory all the same (each request's id, its arms, its attempts per
arm, its cost, and what is left after it), and do not rely on a refusal to
stop you.

There is no fixed ceiling per arm. A run may set a limit per arm, and then
the task, the description and `max_per_arm` show it; without one,
`max_per_arm` equals the whole budget and an arm may take any part of what
is left. An arm larger than the power calculation asks for buys little and
leaves less for the questions that come later.

Experiments run between sessions. A session that requests experiments ends
waiting for them; the next session starts with `memory_read` and finds the
episodes of the experiments in the visible tables. Memory is the only thing
that carries over, so write the plan down before the session ends.

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## Choosing what to test

Rank the open questions by how much the answer can change what you adopt:

1. **Confounded pairs.** Two changes on the same day and scope, one of
   which is likely harmful: one arm reverts the first, another the second,
   a control arm reverts nothing. Whichever arm recovers names the cause.
2. **Candidates at the edge of the decision.** A round adopts at most three
   changes. An action whose interval straddles zero, or straddles the gain
   of the weakest action you would otherwise adopt, is decided by a test.
3. **A setting whose best value is unknown** (an operating point rather
   than a repair): a sweep of a few arms at different values against the
   current value, on the robots concerned. Spread the values widely enough
   that a difference can show; a sweep of neighbouring values usually
   cannot separate them at these sizes. Read it as a curve with intervals,
   not as a contest that the highest point wins.
4. **The set you intend to adopt, together.** When two or more candidate
   changes reach the same robots, one arm with all of them measures what
   they win together, overlap and interaction included, which separate arms
   cannot. This is the measurement your set forecast most needs.
5. **Earlier changes**, from the second round on: an arm that undoes or
   replaces an earlier change tests whether it still earns its place.
6. **Large but uncertain gains**: a wide interval, an attribution from few
   cards, a drift projection.
7. **Competing actions on one component** (clean against replace, service
   against replace): when the cheaper one might do.

Do not test what the log already settles (a large gain whose interval is
far from the decision's edge), and never `route_to_teleop` or
`request_training_data`: they have no simulated effect and are refused.

## Planning the round's spend

Before the first request, write a spending plan to memory:

1. List the questions in the order above, with the comparison each needs.
2. Run `power` for each (below), with `max_per_arm` set to the attempts per
   arm you plan to request, and note the attempts it needs in all
   (`attempts_needed`) and what it costs at your planned size
   (`attempts_at_max`).
3. Add them up against `attempts_left`. If the list does not fit, drop the
   least valuable question, merge questions that share robots into one
   request with a shared control arm (three treatments and one control cost
   four arms, not six), or accept a coarser question and say what it can no
   longer detect. Never cut the control arm to save attempts.
4. Keep a reserve for what the first results will raise: a follow-up on an
   inconclusive test, or the combined arm of the set you settle on. Say how
   large the reserve is and what it is for.

## Designing a request

- **Arms.** Arm 0 is the control and has no actions. Each other arm holds
  the actions to test: usually one, all of them for a combined arm, one
  value each for a sweep. All arms run on the same robots.
- **Robots.** The robots the action reaches: the robot itself for a
  component, robots of the site for a site action. More robots spread the
  work over more cells; the actions must cover every robot listed.
- **Paired starts.** Set `paired_starts` to true: every arm then runs the
  same tasks from the same start states, which removes the task mix from the
  comparison. The power calculation below ignores the pairing, so it is
  conservative.
- **Attempts per arm** (`episodes_per_arm`): from the power calculation,
  within what is left, and at most the run's limit per arm if it sets one.
- **Power note** (`power_note`), written before the request, in one to
  three sentences: the two rates and where they come from (the control
  arm's expected rate is the robots' observed rate since the cause began,
  from `outcomes`, `breakdown` or `gains`; the treatment rate is what you
  expect if the change works), the attempts per arm needed for 80% power,
  the attempts per arm you request, the smallest difference that number
  detects (`smallest_detectable_at_max`; empty when no difference up to the
  ceiling of success reaches 80% power, which means the test cannot settle
  the question), the request's cost and the attempts left after it.
- **Several comparisons.** With more than one treatment arm, the chance that
  one of them clears zero by luck grows. Name the comparison that matters
  most in the power note, and read the others with a wider interval: `z` of
  `readout` at 2.39 for three comparisons, 2.50 for four (Bonferroni at an
  overall 5%).
- **No peeking.** Decide the attempts before the request and read the
  result once. An inconclusive result followed by more attempts is a second
  test; say so, and report the two as such rather than stopping whenever
  the interval happens to clear zero.

## Procedure

1. `memory_read`; read the budget in force and what is left (above).
2. Write the spending plan to memory.
3. For each question, run `power` and write the request. Call
   `request_experiment`; the receipt's id names the experiment. Update the
   count in memory after every receipt, accepted or not.
4. Before the session ends, write to memory: each experiment's id, its
   hypothesis, its arms in words, its cost, the attempts left, and what
   each possible result would mean for what you adopt. End the session by
   saying that you are waiting for experiments.
5. In the next session: `memory_read`, then `experiment_results`. For each
   experiment that is done, run `readout` (below) with one row per episode
   id and its arm, and with `share` set to the share of the fleet's episodes
   the action's scope covers (the `share` that `gains` reported for it).
   The arm episodes are now in the visible tables too, with their
   `experiment_id`; the other blocks leave them out of their rates, so the
   observed window's numbers do not move. Find an experiment's episodes with
   `SELECT experiment_id, episode_id FROM episodes WHERE experiment_id IS
   NOT NULL`, or take them from `experiment_results`.
6. Read the result. `diff` is the arm's success rate minus the control's,
   `diff_low` and `diff_high` its interval. `gain_points`, `gain_low` and
   `gain_high` are the measured gain of the arm's actions in points of
   fleet success with its interval (`100 x diff x share`), computed by the
   block. An interval that excludes zero settles the question; one that
   includes zero with a small upper end says the change wins little; one
   that includes zero and a large upper end says the test was too small.
   A sweep is read arm by arm against the control, and the shape matters as
   much as the best arm: a flat curve says the setting does not matter here
   within the range tried.
7. Replace the log-based estimate with the measured one where the
   experiment tested exactly that action (or that set, for a combined arm);
   note the experiment id as the source. A refused experiment is reported as
   refused, an inconclusive one as inconclusive, and a hypothesis the result
   contradicts is withdrawn in memory with the experiment that withdrew it.
8. Before the round's decision, write the round's spend to memory: attempts
   used, attempts left, and what each experiment changed in the decision.

## Blocks

```sql
-- helper: power
WITH params AS (SELECT 0.6 AS p_control, 0.85 AS p_treatment, 1.96 AS z_alpha, 0.8416 AS z_power, 60 AS max_per_arm, 2 AS arms, 480 AS attempts_left),
p AS (
  SELECT CAST(p_control AS DOUBLE) AS p0, CAST(p_treatment AS DOUBLE) AS p1,
    CAST(z_alpha AS DOUBLE) AS za, CAST(z_power AS DOUBLE) AS zb, max_per_arm,
    CAST(arms AS INT) AS arms, CAST(attempts_left AS INT) AS attempts_left FROM params
),
normal AS (
  -- a close approximation of the standard normal distribution function
  SELECT *, 1 / (1 + exp(-0.07056 * zb * zb * zb - 1.5976 * zb)) AS target_power FROM p
),
grid AS (
  SELECT n.*, d.step / 100.0 AS d,
    least(0.999, n.p0 + d.step / 100.0) AS p_alt
  FROM normal n, range(1, 100) d(step)
),
grid_power AS (
  SELECT *, (abs(p_alt - p0) * sqrt(max_per_arm) - za * sqrt(2 * ((p0 + p_alt) / 2) * (1 - (p0 + p_alt) / 2)))
    / sqrt(p0 * (1 - p0) + p_alt * (1 - p_alt)) AS x
  FROM grid WHERE p0 + d < 1
),
needed AS (
  -- attempts per arm for the target power, and the power at max_per_arm
  SELECT *,
    CAST(ceil(power((za * sqrt(2 * ((p0 + p1) / 2) * (1 - (p0 + p1) / 2))
      + zb * sqrt(p0 * (1 - p0) + p1 * (1 - p1))), 2) / power(p1 - p0, 2)) AS INT) AS n_needed,
    (abs(p1 - p0) * sqrt(max_per_arm) - za * sqrt(2 * ((p0 + p1) / 2) * (1 - (p0 + p1) / 2)))
      / sqrt(p0 * (1 - p0) + p1 * (1 - p1)) AS x_max
  FROM normal
)
SELECT p0 AS p_control, p1 AS p_treatment, round(p1 - p0, 3) AS difference,
  n_needed AS n_per_arm_needed, max_per_arm,
  round(1 / (1 + exp(-0.07056 * power(x_max, 3) - 1.5976 * x_max)), 3) AS power_at_max,
  (SELECT min(g.d) FROM grid_power g
    WHERE 1 / (1 + exp(-0.07056 * g.x * g.x * g.x - 1.5976 * g.x)) >= g.target_power)
    AS smallest_detectable_at_max,
  -- what it costs: every arm, the control included, at its attempts per arm
  arms, CAST(arms * n_needed AS INT) AS attempts_needed,
  CAST(arms * max_per_arm AS INT) AS attempts_at_max,
  CAST(attempts_left - arms * max_per_arm AS INT) AS attempts_left_after
FROM needed
```

```sql
-- helper: readout
WITH params AS (SELECT 1.96 AS z, 1.0 AS share),
-- input: arms(arm, episode_id) AS (VALUES (0, '<episode id>'), (1, '<episode id>')),
arms AS (SELECT CAST(dense_rank() OVER (ORDER BY robot_id) - 1 AS INT) AS arm, episode_id FROM episodes WHERE robot_id IN (SELECT DISTINCT robot_id FROM episodes ORDER BY robot_id LIMIT 2)),
-- end input
joined AS (
  SELECT CAST(a.arm AS INT) AS arm, e.episode_id, e.robot_id, e.task_text,
    CAST(e.outcome = 'success' AS INT) AS ok
  FROM arms a JOIN episodes e USING (episode_id)
),
ev AS (
  SELECT episode_id, count(*) FILTER (WHERE kind = 'drop') AS drops,
    count(*) FILTER (WHERE kind = 'collision') AS collisions
  FROM events GROUP BY episode_id
),
per_arm AS (
  SELECT j.arm, count(*) AS n, sum(j.ok) AS successes, sum(coalesce(ev.drops, 0)) AS drops,
    sum(coalesce(ev.collisions, 0)) AS collisions, count(DISTINCT j.task_text) AS tasks,
    string_agg(DISTINCT j.robot_id, ' ' ORDER BY j.robot_id) AS robots
  FROM joined j LEFT JOIN ev USING (episode_id) GROUP BY j.arm
),
wilson AS (
  SELECT a.*, a.successes / a.n AS rate,
    (a.successes / a.n + p.z * p.z / (2 * a.n)) / (1 + p.z * p.z / a.n)
      - p.z * sqrt(a.successes / a.n * (1 - a.successes / a.n) / a.n + p.z * p.z / (4 * a.n * a.n))
      / (1 + p.z * p.z / a.n) AS low,
    (a.successes / a.n + p.z * p.z / (2 * a.n)) / (1 + p.z * p.z / a.n)
      + p.z * sqrt(a.successes / a.n * (1 - a.successes / a.n) / a.n + p.z * p.z / (4 * a.n * a.n))
      / (1 + p.z * p.z / a.n) AS high
  FROM per_arm a, params p
)
SELECT w.arm, CAST(w.n AS INT) AS n, CAST(w.successes AS INT) AS successes, round(w.rate, 3) AS rate,
  round(w.low, 3) AS rate_low, round(w.high, 3) AS rate_high,
  round(w.rate - c.rate, 3) AS diff,
  round(w.rate - c.rate - sqrt(power(w.rate - w.low, 2) + power(c.high - c.rate, 2)), 3) AS diff_low,
  round(w.rate - c.rate + sqrt(power(w.high - w.rate, 2) + power(c.rate - c.low, 2)), 3) AS diff_high,
  round((w.rate - c.rate) / nullif(sqrt(w.rate * (1 - w.rate) / w.n + c.rate * (1 - c.rate) / c.n), 0), 2)
    AS z_statistic,
  round(w.drops / w.n, 3) AS drops_per_ep, round(w.collisions / w.n, 3) AS collisions_per_ep,
  CAST(w.tasks AS INT) AS tasks, w.robots,
  round(100 * (w.rate - c.rate) * p.share, 2) AS gain_points,
  round(100 * (w.rate - c.rate - sqrt(power(w.rate - w.low, 2) + power(c.high - c.rate, 2))) * p.share, 2)
    AS gain_low,
  round(100 * (w.rate - c.rate + sqrt(power(w.high - w.rate, 2) + power(c.rate - c.low, 2))) * p.share, 2)
    AS gain_high
FROM wilson w JOIN wilson c ON c.arm = 0, params p
ORDER BY w.arm
```
