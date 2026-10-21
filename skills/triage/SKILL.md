---
name: triage
description: First pass over a fleet's log. Success rates with intervals by site, robot, day and task, corrected for the task mix, with near-miss and event rates, and a stratified sample of episodes to look at.
when_to_use: At the start of every analysis, before any hypothesis, and again after new episodes arrive (an experiment's results, a new session).
tools_used: [describe, query, memory_read, memory_write]
inputs:
  - the visible tables (episodes, events)
  - the observed window as a range of days (all days in the export unless told otherwise)
outputs:
  - an outcome table by fleet, site, robot, day and task with intervals and points below expectation
  - a short list of groups that stand out, with their numbers
  - a stratified sample of episode ids for episode-cards
version: "2"
---
# Triage

The goal is a map of where success is lost, with honest uncertainty, before
any explanation. Do not guess causes here; that comes later.

## How to run the SQL blocks of a skill

Each block below starts with a `-- helper: <name>` line. Copy the block
whole and change only two things:

- **The `params` line**, `WITH params AS (SELECT 3 AS failures_per_robot,
  ...)`: change the values, keep the names. Text in single quotes, numbers
  bare, `TRUE` or `FALSE`.
- **The input rows**, where a block takes rows: everything from the line
  after `-- input:` to the line before `-- end input`. The default there is a
  SELECT that makes the block run as it stands. Replace that whole CTE by
  one of the form the `-- input:` comment shows, for example
  `picked(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')),`:
  the input's name and columns as the comment gives them, one row in
  parentheses per item, text in single quotes, numbers bare, `NULL` for an
  empty value, and a comma after the last parenthesis. Keep the two marker
  lines. An input marked optional can stay as it is.

Then run it with `query`. Keep the `-- helper:` line: the tool log then
shows which block produced a number, and every number you report must come
from a tool result.

`query` returns at most 200 rows (the result then says `truncated: true`)
and stops a call after 60 seconds. Every block is written to stay under 200
rows for a fleet of a few dozen robots. If a result is truncated or a call
times out, narrow it (a shorter range of days with `from_day` and `to_day`,
fewer episodes in an input) and run it again; never read a truncated result
as complete.

## Procedure

1. Call `memory_read`. If notes exist, this is a later session: read them and
   continue where they stop instead of starting over. A later session's log
   also holds the episodes of the experiments you requested, marked by
   `episodes.experiment_id`. Every block leaves them out of its rates; read
   them only through the experiment's result (`readout` in
   sensitivity-by-experiment). A block's `excluded` input is for leaving out
   other episodes.
2. Call `describe` twice: with `part` `dictionary` for the tables and their
   columns, then with `part` `fleet` and `include_logs` true for the sites,
   robots, components, the configuration at each scope, the change and
   maintenance logs and the task classes. Each part is one result of at
   most 30,000 characters; `vocabulary` and `costs` are parts of their own,
   and `all` is refused for a larger world. The actions the fleet can take
   are not in the description but in the table `catalogue`, queried like
   any other (`SELECT verb, count(*) FROM catalogue GROUP BY verb`). Note the
   number of sites, robots, days and task classes and the configuration
   keys set at each scope, and which days the log covers:
   `SELECT min(day), max(day), count(*) FROM episodes`.
3. Run `outcomes` (below). It gives one row per group: the fleet, each site,
   robot, day and task. Read these columns:
   - `rate` with `rate_low` and `rate_high`: the success rate and its 95%
     interval (Wilson). Twenty episodes give an interval of about plus or
     minus twenty points; do not rank groups by rates whose intervals
     overlap widely.
   - `task_expected`: the rate the group would have if each of its episodes
     succeeded as often as the rest of the fleet does on the same task. Sites
     and robots run different task mixes, so compare `rate` with
     `task_expected`, not with the fleet's rate. A site with a low raw rate
     and a low `task_expected` is doing harder work, not worse work.
   - `points_below_expected`: how many points of fleet success the group
     loses against that expectation (positive means it loses). This is the
     triage ranking: the large positive values are where the analysis goes
     next.
   - `stands_out`: `worse` when the whole interval lies below the
     expectation, `better` when above. With many groups, one or two will
     stand out by chance; a group that stands out and loses several points
     is a lead, one that stands out by a hair is not.
   - The event columns (`drops_per_ep`, `collisions_per_ep`,
     `stalls_per_ep`, `near_misses_per_success`): where failures leave
     traces. Near misses among successes (an object that slipped and was
     caught) mark a group that is fragile before it fails; a rising near-miss
     rate is an early warning.
4. For the `day` rows, look for a step (a day after which rates drop and stay
   down) or a slope (a steady decline). A step suggests a change on that day;
   a slope suggests a gradual change. Note the day.
5. Run `sample` (below) to choose the episodes to look at. It takes a few
   failures and one success per robot, spread over the days, chosen by a
   fixed hash of the episode id with your `salt`, so the choice is
   reproducible and not cherry-picked; within a day, episodes with video
   come first (cards need frames), and successes with a near miss or another
   notable event come first among successes. `failures_per_robot` applies to
   every robot alike: to look harder at the robots that lose many points,
   run it again with a larger value and keep their rows. Take a second
   sample with another `salt` when a later skill needs more examples. The
   one success per robot is not necessarily of the same task as its
   failures; for a comparison on the same task, find one with `query`
   (`SELECT episode_id FROM episodes WHERE task_text = '<task>' AND outcome =
   'success' AND robot_id <> '<robot id>' LIMIT 3`). The sample is for describing failures, never
   for rates: rates come from the full tables.
6. Write to memory (`memory_write`, replacing the notes with the full
   updated text): the window, the fleet rate with its interval, the groups
   that stand out with `rate`, `task_expected` and `points_below_expected`,
   any step or slope in the days, and the sampled episode ids. Name the block
   that produced each number.

Next: `episode-cards` on the sample, then `failure-modes`. If one group
dominates the loss, `scope` can come first.

## Blocks

```sql
-- helper: outcomes
WITH params AS (SELECT 1.96 AS z, 0 AS from_day, 9999 AS to_day),
-- input: excluded(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')), optional
excluded AS (SELECT episode_id FROM episodes WHERE false),
-- end input
ep AS (
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, e.task_text,
    CAST(e.outcome = 'success' AS INT) AS ok
  FROM episodes e, params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
    AND e.episode_id NOT IN (SELECT episode_id FROM excluded)
),
ev AS (
  SELECT episode_id,
    count(*) FILTER (WHERE kind = 'drop') AS drops,
    count(*) FILTER (WHERE kind = 'near_miss') AS near_misses,
    count(*) FILTER (WHERE kind = 'collision') AS collisions,
    count(*) FILTER (WHERE kind = 'stall') AS stalls
  FROM events GROUP BY episode_id
),
joined AS (
  SELECT ep.*, ep.task_text AS task, coalesce(ev.drops, 0) AS drops,
    coalesce(ev.near_misses, 0) * ep.ok AS success_near_misses,
    coalesce(ev.collisions, 0) AS collisions, coalesce(ev.stalls, 0) AS stalls
  FROM ep LEFT JOIN ev USING (episode_id)
),
cells AS (
  SELECT
    CASE WHEN robot_id IS NOT NULL THEN 'robot' WHEN site_id IS NOT NULL THEN 'site'
      WHEN day IS NOT NULL THEN 'day' WHEN task_text IS NOT NULL THEN 'task'
      ELSE 'fleet' END AS dimension,
    coalesce(robot_id, site_id, CAST(day AS VARCHAR), task_text, 'fleet') AS level,
    task, count(*) AS n, sum(ok) AS ok, sum(drops) AS drops,
    sum(success_near_misses) AS success_near_misses, sum(collisions) AS collisions,
    sum(stalls) AS stalls
  FROM joined
  GROUP BY GROUPING SETS ((robot_id, task), (site_id, task), (day, task), (task_text, task), (task))
),
task_total AS (SELECT task, n AS tn, ok AS tok FROM cells WHERE dimension = 'fleet'),
fleet AS (SELECT sum(tn) AS fn, sum(tok) AS fok FROM task_total),
grouped AS (
  SELECT c.dimension, c.level, sum(c.n) AS n, sum(c.ok) AS ok, sum(c.drops) AS drops,
    sum(c.success_near_misses) AS success_near_misses, sum(c.collisions) AS collisions,
    sum(c.stalls) AS stalls,
    sum(c.n * (t.tok - c.ok) / (t.tn - c.n)) FILTER (WHERE t.tn > c.n)
      / sum(c.n) FILTER (WHERE t.tn > c.n) AS task_expected
  FROM cells c JOIN task_total t USING (task)
  GROUP BY c.dimension, c.level
),
rated AS (
  SELECT g.*, g.ok / g.n AS rate,
    (f.fok - g.ok) / nullif(f.fn - g.n, 0) AS rest_rate, f.fn,
    (g.ok / g.n + p.z * p.z / (2 * g.n)) / (1 + p.z * p.z / g.n) AS centre,
    p.z * sqrt(g.ok / g.n * (1 - g.ok / g.n) / g.n + p.z * p.z / (4 * g.n * g.n))
      / (1 + p.z * p.z / g.n) AS half
  FROM grouped g, fleet f, params p
)
SELECT dimension, level, CAST(n AS INT) AS n, CAST(ok AS INT) AS successes,
  round(rate, 3) AS rate, round(centre - half, 3) AS rate_low, round(centre + half, 3) AS rate_high,
  round(rest_rate, 3) AS rest_rate, round(task_expected, 3) AS task_expected,
  round(100 * (coalesce(task_expected, rest_rate) - rate) * n / fn, 2) AS points_below_expected,
  CASE WHEN centre + half < coalesce(task_expected, rest_rate) THEN 'worse'
    WHEN centre - half > coalesce(task_expected, rest_rate) THEN 'better' ELSE '' END AS stands_out,
  round(drops / n, 3) AS drops_per_ep, round(collisions / n, 3) AS collisions_per_ep,
  round(stalls / n, 3) AS stalls_per_ep,
  round(success_near_misses / nullif(ok, 0), 3) AS near_misses_per_success
FROM rated
ORDER BY CASE dimension WHEN 'fleet' THEN 0 WHEN 'site' THEN 1 WHEN 'robot' THEN 2
  WHEN 'day' THEN 3 ELSE 4 END, points_below_expected DESC NULLS LAST
```

```sql
-- helper: sample
WITH params AS (SELECT 3 AS failures_per_robot, 1 AS successes_per_robot, 'a' AS salt, 0 AS from_day, 9999 AS to_day),
ev AS (
  SELECT episode_id, string_agg(kind || '@' || step, ' ' ORDER BY step) AS events,
    bool_or(kind IN ('near_miss', 'drop', 'collision', 'stall')) AS notable
  FROM events GROUP BY episode_id
),
candidates AS (
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, e.task_text, e.outcome, e.steps,
    coalesce(ev.events, '') AS events, e.video_scene IS NOT NULL AS has_video,
    e.outcome = 'success' AND coalesce(ev.notable, false) AS notable_success,
    md5(p.salt || e.episode_id) AS draw,
    md5(p.salt || e.robot_id || CAST(e.day AS VARCHAR)) AS day_draw
  FROM episodes e LEFT JOIN ev USING (episode_id), params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
),
within_day AS (
  SELECT *, row_number() OVER (PARTITION BY robot_id, outcome, day
    ORDER BY has_video DESC, notable_success DESC, draw) AS turn
  FROM candidates
),
ranked AS (
  -- one episode per day in turn, so that the picks spread over the days
  SELECT *, row_number() OVER (PARTITION BY robot_id, outcome
    ORDER BY notable_success DESC, turn, day_draw) AS pick
  FROM within_day
)
SELECT r.robot_id, r.site_id, r.outcome, CAST(r.pick AS INT) AS pick, r.episode_id, r.day,
  r.task_text, r.steps, r.events, r.has_video
FROM ranked r, params p
WHERE (r.outcome = 'failure' AND r.pick <= p.failures_per_robot)
   OR (r.outcome = 'success' AND r.pick <= p.successes_per_robot)
ORDER BY r.site_id, r.robot_id, r.outcome, r.pick
```
