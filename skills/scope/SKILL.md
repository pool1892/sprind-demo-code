---
name: scope
description: Break a failure mode down by robot, site, day and configuration, read the change log, the maintenance log and the health checks against it, and find the narrowest scope and the moment that explain it, including when two changes cannot be told apart.
when_to_use: For each failure mode that is concentrated somewhere (failure-modes, step 4), and for each group triage flags. Run it again when an experiment's results arrive.
tools_used: [describe, query, memory_read, memory_write]
inputs:
  - a failure mode with its fingerprint (a signature), or failures in general
  - the fleet's part of the system description (configuration per scope) and the change_log and maintenance tables
outputs:
  - for each mode, the narrowest scope that explains it (a robot, a site or the fleet), the day it started, and whether it grows
  - the change-log and maintenance entries that line up with it, with their before and after effect, and which of them are confounded
  - the health readings that support or contradict a component cause
version: "4"
---
# Scope

Where and since when does a mode happen, and what changed there and then?
The answer narrows the candidate causes to a target (a component of a robot,
a configuration key at a scope, a condition at a site) and a direction
(`degraded`, `misconfigured`, `changed`, `drifting`).

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## Procedure

1. Run `breakdown` (below) with `signature` set to the mode's fingerprint
   (as `signatures` in failure-modes names it), or `any` for all failures.
   Per robot and per site: the failures of that kind out of all episodes,
   the daily counts, and the slope per day. Decide the scope:
   - one robot far above the others, the other robots of its site normal: a
     robot scope;
   - all robots of a site raised together, the other sites normal: a site
     scope;
   - every robot raised alike: the fleet, or the policy itself (compare with
     the first days, before anything changed).
2. Find the moment. In the daily counts, a mode that starts on a day and
   stays is a step; one that grows steadily is a drift. Run `trend` (below)
   for the slopes: each robot and site's daily success gap against the rest
   of the fleet (task mix removed), its slope per day, and the day the
   fitted gap crosses minus `margin`; for a robot also the gap against the
   other robots of its own site (`within_site_slope`, `within_site_r2`). A
   step shows as a gap that jumps between two days.
   **A site's step is not a drift.** A slope that every robot of a site
   shares, together with a jump in `breakdown`'s daily counts and an entry in
   `changes` on that day, belongs to the site: a change from a day on looks
   like a slope when a straight line is fitted across it. Call a robot
   drifting, and warn for it, only when its `within_site_slope` is clearly
   negative with a fit that is not poor; then its own slope and
   `crossing_day` are the ones to use.
3. Run `changes` (below), with `to_day` the last observed day. For every
   change-log entry and every maintenance entry: success on the scope it touches in the `days_around` days before
   and after, the same for the rest of the fleet, and the difference of the
   two differences (`did`, task mix removed) with its interval, and the
   points of fleet success it amounts to. Read it with care:
   - An entry whose `did` interval lies clearly below zero lines up with a
     loss. That is a correlation in time, not yet a cause.
   - **Confounded entries.** `same_day_overlapping` lists other entries made
     on the same day on a scope that overlaps. The log cannot separate them:
     both show the same `did`, because they cover the same episodes from the
     same day. Two entries that land together, one harmful and one
     harmless, look identical here. Keep both as candidates, compare their
     support in the episode evidence and public description, and plan an
     experiment that reverts one of them alone (`sensitivity-by-experiment`).
     Lower your confidence in either until that is done.
   - **Unlogged alternatives.** An empty `same_day_overlapping` means no
     other recorded entry, not that the edit is the only possible cause.
     A shared change in physical conditions, input objects or timing may
     leave no entry. Compare the symptom and size of the loss with the
     proposed mechanism. Return to episode-cards for matched tasks before
     and after the step, viewing both examples and a comparison scope.
     Record which observation distinguishes the alternatives. If none does,
     retain the scope and onset as findings but leave the mechanism open;
     the scope's loss is not an identified gain from reverting its logged edit.
   - A fleet-wide entry has no rest of the fleet to compare with; its
     before and after also contain every other trend of those days.
   - An entry inside a drift picks up the drift: a robot that worsens every
     day shows a negative `did` for any change that day.
   - Maintenance is an observational test of a mechanism, not proof by itself.
     Compare matched work before and after, the component's exposure and
     concurrent changes. No recovery can mean the verb addresses a different
     mechanism, another cause remains, or the comparison is too weak; it does
     not establish that a more expensive verb will work. Recovery supports
     the component hypothesis only to the extent that alternatives are excluded.
   - An entry with no effect is information too: a change that moved
     nothing is not a cause, however suspicious it looks.
4. Run `health` (below). Per robot, component and metric: the readings'
   first and last value, their slope, and how far the robot stands from the
   fleet on the same days (`mean_z`, `last_z`), and how the metric moves
   with the robot's daily failures (`corr_with_failure`). Health readings
   are noisy and some drift for every robot (a counter that grows with use):
   a reading supports a component cause only when it stands apart from the
   fleet *and* lines up with the failures in time. A normal health reading
   excludes only what that metric measures; use the dictionary to identify
   its coverage.
   Before naming the same component on several robots, compare their shared
   site and task pattern with robots elsewhere doing the same work. A shared
   condition can produce different symptoms on components of different health.
   Sparse or missing comparison tasks limit the conclusion; they do not prove
   independent component failures.
5. Read the fleet's part of the system description (`describe` with `part`
   `fleet`; the change and maintenance logs are also the tables
   `change_log` and `maintenance`): the configuration values at the fleet,
   at each site and at each robot, with the day each version started. A robot that sets a key itself is not touched by a site or fleet
   change of that key. The configuration in force on a robot is its own
   value, else its site's, else the fleet's.
6. For each mode write to memory: the scope, the start day or the slope,
   the candidate targets with their direction, the entries that line up
   (and those confounded with them), the health readings that support or
   contradict, and the block and parameters behind each number.

Directions: a component that works worse is `degraded`; a configuration
value that is wrong is `misconfigured`, or `changed` when it is a change
that came with a version or an update; anything that worsens day by day is
also `drifting`.

Next: `sensitivity-from-logs` for each candidate target.

## Blocks

```sql
-- helper: breakdown
WITH params AS (SELECT 'any' AS signature, 0.85 AS closed_below, 0.025 AS empty_below, 25 AS settle_steps, 0 AS from_day, 9999 AS to_day),
-- input: excluded(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')), optional
excluded AS (SELECT episode_id FROM episodes WHERE false),
-- end input
ep AS (
  SELECT e.* FROM episodes e, params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
    AND e.episode_id NOT IN (SELECT episode_id FROM excluded)
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
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, CASE
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
marked AS (
  SELECT s.*, CAST(CASE WHEN p.signature = 'any' THEN s.signature <> 'success'
    ELSE s.signature = p.signature END AS INT) AS hit
  FROM signed s, params p
),
daily AS (
  SELECT 'robot' AS dimension, robot_id AS level, site_id AS site, day, sum(hit) AS k, count(*) AS n
  FROM marked GROUP BY robot_id, site_id, day
  UNION ALL
  SELECT 'site', site_id, site_id, day, sum(hit), count(*) FROM marked GROUP BY site_id, day
  UNION ALL
  SELECT 'fleet', 'fleet', '', day, sum(hit), count(*) FROM marked GROUP BY day
)
SELECT dimension, level, site, CAST(sum(k) AS INT) AS hits, CAST(sum(n) AS INT) AS n,
  round(sum(k) / sum(n), 3) AS share,
  list(CAST(k AS INT) ORDER BY day) AS daily_hits, list(CAST(n AS INT) ORDER BY day) AS daily_n,
  list(day ORDER BY day) AS days,
  round(regr_slope(k / n, day), 3) AS slope_per_day
FROM daily
GROUP BY dimension, level, site
ORDER BY CASE dimension WHEN 'fleet' THEN 0 WHEN 'site' THEN 1 ELSE 2 END, share DESC
```

```sql
-- helper: trend
WITH params AS (SELECT 0.05 AS margin, 0 AS from_day, 9999 AS to_day),
-- input: excluded(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')), optional
excluded AS (SELECT episode_id FROM episodes WHERE false),
-- end input
ep AS (
  SELECT e.robot_id, e.site_id, e.day, e.task_text, CAST(e.outcome = 'success' AS INT) AS ok
  FROM episodes e, params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
    AND e.episode_id NOT IN (SELECT episode_id FROM excluded)
),
task_rate AS (SELECT task_text, avg(ok) AS task_ok FROM ep GROUP BY task_text),
resid AS (SELECT ep.*, ok - task_ok AS r FROM ep JOIN task_rate USING (task_text)),
scopes AS (
  SELECT DISTINCT 'robot' AS dimension, robot_id AS level, site_id AS site FROM ep
  UNION ALL SELECT DISTINCT 'site', site_id, site_id FROM ep
),
daily AS (
  SELECT s.dimension, s.level, x.day,
    avg(x.r) FILTER (WHERE s.level IN (x.robot_id, x.site_id))
      - avg(x.r) FILTER (WHERE s.level NOT IN (x.robot_id, x.site_id)) AS gap,
    -- a robot against the other robots of its own site: a step of the whole site cancels out
    CASE WHEN s.dimension = 'robot' THEN avg(x.r) FILTER (WHERE x.robot_id = s.level)
      - avg(x.r) FILTER (WHERE x.site_id = s.site AND x.robot_id <> s.level) END AS site_gap,
    count(*) FILTER (WHERE s.level IN (x.robot_id, x.site_id)) AS n
  FROM scopes s, resid x GROUP BY s.dimension, s.level, s.site, x.day
),
fit AS (
  SELECT dimension, level, regr_slope(gap, day) AS slope, regr_intercept(gap, day) AS intercept,
    regr_r2(gap, day) AS r2, max(day) AS last_day,
    regr_slope(site_gap, day) AS site_slope, regr_r2(site_gap, day) AS site_r2,
    list(round(gap, 2) ORDER BY day) AS daily_gap,
    list(round(site_gap, 2) ORDER BY day) AS daily_site_gap,
    list(CAST(n AS INT) ORDER BY day) AS daily_n
  FROM daily WHERE gap IS NOT NULL GROUP BY dimension, level
)
SELECT f.dimension, f.level, round(f.slope, 3) AS slope_per_day, round(f.r2, 2) AS r2,
  round(f.intercept + f.slope * f.last_day, 3) AS fitted_gap_last_day, f.daily_gap, f.daily_n,
  CASE WHEN f.slope < 0 THEN CAST(ceil((-p.margin - f.intercept) / f.slope) AS INT) END
    AS crossing_day,
  round(f.site_slope, 3) AS within_site_slope, round(f.site_r2, 2) AS within_site_r2,
  CASE WHEN f.dimension = 'robot' THEN f.daily_site_gap END AS daily_within_site_gap
FROM fit f, params p
ORDER BY CASE f.dimension WHEN 'robot' THEN 0 ELSE 1 END, coalesce(f.site_slope, f.slope)
```

```sql
-- helper: changes
WITH params AS (SELECT 2 AS days_around, 1.96 AS z, 0 AS from_day, 9999 AS to_day),
ep AS (
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, e.task_text,
    CAST(e.outcome = 'success' AS INT) AS ok
  FROM episodes e, params p
  WHERE e.day BETWEEN p.from_day AND p.to_day AND e.experiment_id IS NULL
),
task_rate AS (SELECT task_text, avg(ok) AS task_ok FROM ep GROUP BY task_text),
resid AS (SELECT ep.*, ok - task_ok AS r FROM ep JOIN task_rate USING (task_text)),
robots AS (SELECT DISTINCT robot_id, site_id FROM episodes),
entries AS (
  SELECT * FROM (
    SELECT 'change' AS kind, day, scope, key AS what, old || ' -> ' || new AS detail, author
    FROM change_log
    UNION ALL
    SELECT 'maintenance', day, robot_id, component || ' ' || verb, note, '' FROM maintenance
  ) x WHERE x.day BETWEEN (SELECT from_day FROM params) AND (SELECT to_day FROM params)
),
numbered AS (SELECT row_number() OVER (ORDER BY day, kind, scope, what) AS entry, * FROM entries),
coincident AS (
  SELECT a.entry, string_agg(b.what || ' at ' || b.scope, '; ' ORDER BY b.entry) AS same_day
  FROM numbered a JOIN numbered b ON a.day = b.day AND a.entry <> b.entry
  WHERE a.scope = 'fleet' OR b.scope = 'fleet' OR a.scope = b.scope
    OR EXISTS (SELECT 1 FROM robots r WHERE (r.robot_id = a.scope AND r.site_id = b.scope)
      OR (r.robot_id = b.scope AND r.site_id = a.scope))
  GROUP BY a.entry
),
windows AS (
  SELECT n.entry, x.r, x.ok,
    n.scope IN ('fleet', x.robot_id, x.site_id) AS covered,
    x.day >= n.day AS after
  FROM numbered n JOIN resid x ON x.day BETWEEN n.day - (SELECT days_around FROM params)
    AND n.day + (SELECT days_around FROM params) - 1
),
cells AS (
  SELECT entry,
    count(*) FILTER (WHERE covered AND NOT after) AS n_before,
    count(*) FILTER (WHERE covered AND after) AS n_after,
    avg(ok) FILTER (WHERE covered AND NOT after) AS rate_before,
    avg(ok) FILTER (WHERE covered AND after) AS rate_after,
    avg(ok) FILTER (WHERE NOT covered AND NOT after) AS rest_before,
    avg(ok) FILTER (WHERE NOT covered AND after) AS rest_after,
    avg(r) FILTER (WHERE covered AND after) - avg(r) FILTER (WHERE covered AND NOT after)
      - (avg(r) FILTER (WHERE NOT covered AND after) - avg(r) FILTER (WHERE NOT covered AND NOT after))
      AS did,
    sqrt(coalesce(var_samp(r) FILTER (WHERE covered AND after) / count(*) FILTER (WHERE covered AND after), 0)
      + coalesce(var_samp(r) FILTER (WHERE covered AND NOT after) / count(*) FILTER (WHERE covered AND NOT after), 0)
      + coalesce(var_samp(r) FILTER (WHERE NOT covered AND after) / count(*) FILTER (WHERE NOT covered AND after), 0)
      + coalesce(var_samp(r) FILTER (WHERE NOT covered AND NOT after) / count(*) FILTER (WHERE NOT covered AND NOT after), 0))
      AS se
  FROM windows GROUP BY entry
),
shares AS (
  SELECT n.entry, avg(CAST(n.scope IN ('fleet', x.robot_id, x.site_id) AS INT)) AS share
  FROM numbered n, ep x GROUP BY n.entry
)
SELECT n.entry, n.kind, n.day, n.scope, n.what, n.detail, n.author,
  CAST(c.n_before AS INT) AS n_before, CAST(c.n_after AS INT) AS n_after,
  round(c.rate_before, 3) AS rate_before, round(c.rate_after, 3) AS rate_after,
  round(c.rest_before, 3) AS rest_before, round(c.rest_after, 3) AS rest_after,
  round(c.did, 3) AS did, round(c.did - p.z * c.se, 3) AS did_low, round(c.did + p.z * c.se, 3) AS did_high,
  round(100 * c.did * s.share, 2) AS fleet_points,
  coalesce(o.same_day, '') AS same_day_overlapping
FROM numbered n JOIN cells c USING (entry) JOIN shares s USING (entry)
  LEFT JOIN coincident o USING (entry), params p
ORDER BY n.day, n.entry
```

```sql
-- helper: health
WITH params AS (SELECT 0 AS from_day, 9999 AS to_day),
readings AS (
  SELECT h.* FROM health h, params p WHERE h.day BETWEEN p.from_day AND p.to_day
),
fleet_day AS (
  SELECT day, component, metric, count(*) AS k, sum(value) AS total, sum(value * value) AS squares
  FROM readings GROUP BY day, component, metric
),
others AS (
  -- each reading against the other robots' readings of the same day
  SELECT r.*, (f.total - r.value) / (f.k - 1) AS mean_others,
    sqrt(greatest(0, (f.squares - r.value * r.value
      - (f.total - r.value) * (f.total - r.value) / (f.k - 1)) / nullif(f.k - 2, 0))) AS sd_others
  FROM readings r JOIN fleet_day f USING (day, component, metric) WHERE f.k > 2
),
scored AS (SELECT *, (value - mean_others) / nullif(sd_others, 0) AS z FROM others),
daily_failure AS (
  SELECT robot_id, day, 1 - avg(CAST(outcome = 'success' AS INT)) AS failure_rate
  FROM episodes WHERE experiment_id IS NULL GROUP BY robot_id, day
),
per_robot AS (
  SELECT s.robot_id, s.component, s.metric, count(*) AS days,
    arg_min(s.value, s.day) AS first_value, arg_max(s.value, s.day) AS last_value,
    regr_slope(s.value, s.day) AS slope, avg(s.z) AS mean_z, arg_max(s.z, s.day) AS last_z,
    corr(s.value, d.failure_rate) AS corr_with_failure
  FROM scored s LEFT JOIN daily_failure d USING (robot_id, day)
  GROUP BY s.robot_id, s.component, s.metric
),
fleet_slope AS (
  SELECT component, metric, avg(slope) AS fleet_slope FROM per_robot GROUP BY component, metric
)
SELECT p.robot_id, p.component, p.metric, CAST(p.days AS INT) AS days,
  round(p.first_value, 3) AS first_value, round(p.last_value, 3) AS last_value,
  round(p.slope, 4) AS slope_per_day, round(p.slope - f.fleet_slope, 4) AS slope_vs_fleet,
  round(p.mean_z, 2) AS mean_z, round(p.last_z, 2) AS last_z,
  round(p.corr_with_failure, 2) AS corr_with_failure
FROM per_robot p JOIN fleet_slope f USING (component, metric)
ORDER BY abs(p.mean_z) DESC NULLS LAST
LIMIT 150
```
