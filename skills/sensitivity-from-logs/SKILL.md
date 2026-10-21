---
name: sensitivity-from-logs
description: Estimate, from the log alone, how many points of fleet success each candidate action would win back, with an interval, using the gap between the affected scope and comparable episodes, the scope's share of the fleet's work, a drift projection, and uncertainty about the repair response.
when_to_use: After scope has named candidate targets, before rank-actions. In the logs-only condition this is the only source of expected gains; with experiments, it decides which candidates are worth testing.
tools_used: [query, memory_read, memory_write]
inputs:
  - candidate targets with their scope and start day (from scope)
  - for each, the share of the scope's excess failures it explains (from failure-modes counts)
outputs:
  - for each candidate, the expected gain in points of fleet success over the scoring window, with a low and a high value, and the numbers behind it
version: "8"
---
# Sensitivity from logs

The question for each candidate: if the action is taken, how many more
episodes succeed in the scoring window, as points of fleet success? From the
log alone the estimate is the loss the scope shows today, carried forward,
corrected for drift and for what the action can and cannot undo.

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## The estimate

For a candidate whose scope is a robot or a site, from its start day:

- `gap`: the share of episodes the scope loses against comparable
  episodes: those on the same days that no other candidate covers, task by
  task (so a harder task mix is not counted as a loss, and another cause
  does not lower the comparison). `n_comparison` says how many there were;
  with `exclude_suspects` set to FALSE the block compares with every
  episode outside the scope instead.
- `share`: the scope's share of all the fleet's episodes. Robots work about
  equally, so the scoring window will have about the same share.
- `attributable`: the share of the gap this cause explains. One when it is
  the only mode on the scope; less when the scope shows two modes (take the
  ratio from the mode counts of failure-modes, the excess failures of this
  mode over the excess failures of all modes there).
- `gain_points = 100 x gap x share x attributable`, with `gain_low` and
  `gain_high` from an approximate conditional sampling interval combining
  Wilson bounds for the scope's and comparison's task rates
  ([MOVER](https://www.sciencedirect.com/science/article/pii/S0167947308004623)).
- `within_site_gap`, for a robot: the gap against the other robots of its
  own site on the same days, task by task. Another robot cause at the same
  site lowers that comparison.

For a change made on a day (a configuration change, a software version),
the scope may have differed from the rest before the change too. The block
also gives `gap_before`, the same gap before `from_day`, and
`gain_points_change`, which counts only the difference: use it for changes.
A fleet-wide scope has nothing to compare with on the same days; use the
`changes` block of `scope` (before and after) instead, and say so.

The intervals treat observed task weights, fleet share, selected candidates
and comparators, and attribution as fixed. They do not cover dependence
within robots or days, selection, unmeasured confounding, repair-efficacy
uncertainty, future workload, or drift. Missing comparison support or an
absent pre-period leaves the corresponding estimate and interval unavailable.
The default `z = 1.96` gives approximate 95% bounds; `z` must be positive
and finite, and each `attributable` must be in [0, 1].
The block rounds interval endpoints outward. For custom estimates, derive
physical caps from full-precision counts, apply them before the final outward
rounding, and retain the outward-rounded endpoints. A raw floating-point cap
can differ from an equivalent ratio in its last digit; it must not replace
the reported bound. Do not derive a cap from the displayed, rounded `share`.

## Procedure

No arithmetic in your head: every gain, low and high you will report is a
value the block returns. When a number needs a factor (a weaker repair, an
uncertain attribution, a confounded pair), that factor goes into an input
row's `attributable`, and the block does the multiplication.

1. Write one input row per candidate action: a label (the action in words,
   unique), the scope (a robot id, a site id), the day the cause began (0 if
   it was there from the start), and `attributable`. Several actions on the
   same scope are separate rows: the two entries of a confounded pair, and
   each verb on one component. Different repair responses are separate
   scenarios, justified by evidence in this export or its experiments.
   Interpret maintenance comparisons with their timing and concurrent changes;
   a price difference alone does not establish an efficacy difference.
   An action on a target that is not the cause gets a row with zero.
2. Run `gains` (below). Set `last_day` to the end of the observed window.
   For a drifting cause, set `project_to_day` to the middle of the scoring
   window for an initial scenario: the block fits loss against day, weighted
   by observed attempts (`slope_per_day`). `projected_gain_points` bounds
   the implied future success rate to [0, 1], holding the comparator rate
   fixed. `raw_projected_gain_points` is the unbounded diagnostic;
   `projection_clipped` flags when it exceeds those probability bounds.
   The usable observed day count and range accompany the projection; fewer
   than two distinct usable days, missing support or nonfinite inputs leave
   it unavailable. This scenario is neither an expected repair gain nor a
   confidence bound. Clipping signals a linear extrapolation failure, not
   evidence of almost complete recovery. Compare persistence with the trend,
   using within-site rates, task mix and episode evidence to justify a central
   forecast. If a trend supplies the scoring-window point, query each scoring
   day and average the bounded predictions with workload weights: clipping
   the midpoint alone need not give the window average. Record the chosen
   scenario and the assumed comparator stability.
3. Write the reason for each row's `attributable` beside its numbers. Separate
   uncertainty about the mechanism from how much of that mechanism the action
   removes. Use repair response measured from the available observations or experiments;
   a cheaper verb is not automatically a partial repair. Removing one
   mechanism leaves co-occurring causes in place, so it need not recover the
   entire scope gap. A verb that addresses none of the supported mechanisms
   gets no gain merely because it acts on the same component.
4. Check for double counting: `shared_failures` counts the scope's failures
   that another candidate also covers. Gains of candidates that cover the
   same failures do not add up; when two causes act on the same episodes,
   removing either may win back most of their joint loss (an interaction),
   or only its own part. Say which you assume. A robot inside a site that
   is itself a candidate shares the site's failures: read its
   `within_site_gap`. If it is no worse than the other robots of its site
   (a gap near zero), its loss is the site's, and its own row gets a small
   `attributable`, often zero.
5. Set the uncertainty, each end from a returned value:
   - by default, low `gain_low` and high `gain_high` of the action's row
     (`gain_change_low` and `gain_change_high` for a change);
   - **confounded alternatives**: evaluate the plausible mechanism scenarios,
     including each candidate alone and joint effects where supported. Set
     the central attribution from the matched observations and mechanism
     evidence; do not split the loss equally merely because two candidates
     coincide. If their relative support cannot be resolved, state the central
     assumption and show its sensitivity rather than presenting it as measured.
     Take the envelope of scenario bounds, including no benefit where plausible.
     Gain is signed: a changed setting can help current attempts, so undoing it
     can break some of them. Include supported harm scenarios and retain negative
     contrast bounds; zero is not an automatic lower bound for a proposed fix;
   - **a drift**: cover both the conditional sampling bounds and plausible
     persistence/trend scenarios, taking their envelope with `query`. Keep
     the central forecast separate from the envelope. Neither clipping nor
     extending a trend establishes that its extreme is the expected gain;
   - **uncertain attribution or repair efficacy**: evaluate the plausible
     mechanisms and any partial-response fractions supported by measurements
     or the cards as separate rows. State assumptions about efficacy when the
     evidence does not resolve it. Keep the best-supported fraction for the
     point estimate and take the minimum low and maximum high across the
     scenarios with `query`. An expected partial restoration does not make
     full restoration impossible; do not cap the upper bound at the assumed
     central fraction without evidence.
   An interval that honestly spans zero is better than a narrow one that is
   wrong.
6. Write each candidate's numbers to memory with the block's parameters,
   so that every number in the fix set can be traced to this query.

## Public action vocabulary

The public catalogue states which verbs apply to which targets. It does not
specify a response curve or promise complete removal of a mechanism.

| Public component | Catalogue verbs (robot scope) |
| --- | --- |
| Scene camera or wrist camera | recalibrate, clean, service, replace |
| Gripper | clean, service, replace |
| Arm | recalibrate, service, replace |
| Controller | service, replace |

Configuration keys allow `set` and `revert`; site conditions allow
`change_condition`. Check the actual catalogue row and the public description
for the target and value. Use observed maintenance, matched comparisons or an
experiment to estimate which action changes the supported failure mode.
A low price does not imply a partial repair, and a high price does not imply
complete recovery. If the mode fits no described class, keep its mechanism
unknown and reason from the evidence to a component, key or condition.

## Blocks

```sql
-- helper: gains
WITH params AS (SELECT 1.96 AS z, 0 AS first_day, 9999 AS last_day, -1 AS project_to_day, TRUE AS exclude_suspects),
-- input: candidates(label, scope, from_day, attributable) AS (VALUES ('<label>', '<robot or site id>', 0, 1.0), ('<label>', '<robot or site id>', 2, 0.5)),
candidates AS (SELECT 'everything at ' || robot_id AS label, robot_id AS scope, 0 AS from_day, 1.0 AS attributable FROM (SELECT DISTINCT robot_id FROM episodes)),
-- end input
-- input: excluded(episode_id) AS (VALUES ('<episode id>'), ('<episode id>')), optional
excluded AS (SELECT episode_id FROM episodes WHERE false),
-- end input
interval_params AS (
  SELECT CASE WHEN CAST(z AS DOUBLE) > 0 AND isfinite(CAST(z AS DOUBLE)) THEN CAST(z AS DOUBLE)
    ELSE error('z must be positive and finite') END AS z FROM params
),
ep AS (
  SELECT e.episode_id, e.robot_id, e.site_id, e.day, e.task_text,
    CAST(e.outcome = 'success' AS INT) AS ok
  FROM episodes e, params p
  WHERE e.day BETWEEN p.first_day AND p.last_day AND e.experiment_id IS NULL
    AND e.episode_id NOT IN (SELECT episode_id FROM excluded)
),
flags AS (
  SELECT c.label, c.scope, CAST(c.from_day AS INT) AS from_day,
    CASE WHEN CAST(c.attributable AS DOUBLE) BETWEEN 0 AND 1
      THEN CAST(c.attributable AS DOUBLE)
      ELSE error('attributable must be in [0, 1]') END AS attributable, x.*,
    c.scope IN ('fleet', x.robot_id, x.site_id) AS covered,
    x.day >= CAST(c.from_day AS INT) AS active
  FROM candidates c, ep x
),
suspects AS (
  -- episodes that another candidate (not fleet-wide) covers while its cause is active
  SELECT a.label, a.episode_id, bool_or(b.covered AND b.active AND b.scope <> 'fleet') AS suspect
  FROM flags a JOIN flags b ON a.episode_id = b.episode_id AND a.label <> b.label
  GROUP BY a.label, a.episode_id
),
marked AS (
  SELECT f.*, NOT f.covered AND NOT (p.exclude_suspects AND coalesce(s.suspect, false)) AS comparison
  FROM flags f LEFT JOIN suspects s USING (label, episode_id), params p
),
task_rates AS (
  SELECT label, active, task_text, avg(ok) FILTER (WHERE comparison) AS clean_ok,
    avg(ok) FILTER (WHERE NOT covered) AS rest_ok,
    count(*) FILTER (WHERE comparison) AS clean_n,
    count(*) FILTER (WHERE NOT covered) AS rest_n
  FROM marked GROUP BY label, active, task_text
),
rest AS (
  SELECT label, active, avg(ok) FILTER (WHERE comparison) AS clean_rate,
    avg(ok) FILTER (WHERE NOT covered) AS rest_rate,
    count(*) FILTER (WHERE comparison) AS n_comparison,
    count(*) FILTER (WHERE NOT covered) AS rest_n
  FROM marked GROUP BY label, active
),
scored AS (
  SELECT m.*, coalesce(t.clean_ok, r.clean_rate, t.rest_ok, r.rest_rate) AS expected,
    r.n_comparison
  FROM marked m LEFT JOIN task_rates t USING (label, active, task_text)
    LEFT JOIN rest r USING (label, active)
),
summary AS (
  SELECT label, any_value(from_day) AS from_day, any_value(attributable) AS attributable,
    count(*) FILTER (WHERE covered AND active) AS n,
    sum(ok) FILTER (WHERE covered AND active) AS successes,
    avg(expected) FILTER (WHERE covered AND active) AS expected_rate,
    max(n_comparison) FILTER (WHERE active) AS n_comparison,
    count(*) FILTER (WHERE covered AND NOT active) AS n_before,
    avg(expected - ok) FILTER (WHERE covered AND NOT active) AS gap_before,
    avg(CAST(covered AS INT)) AS share,
    count(DISTINCT day) FILTER (WHERE covered AND active AND expected IS NOT NULL)
      AS projection_observed_days,
    min(day) FILTER (WHERE covered AND active AND expected IS NOT NULL) AS projection_first_day,
    max(day) FILTER (WHERE covered AND active AND expected IS NOT NULL) AS projection_last_day,
    regr_slope(expected - ok, day) FILTER (WHERE covered AND active) AS slope,
    regr_intercept(expected - ok, day) FILTER (WHERE covered AND active) AS intercept
  FROM scored GROUP BY label
),
robot_site AS (
  -- a robot candidate's site, for the comparison with its own site's other robots
  SELECT label, any_value(site_id) AS site FROM flags WHERE covered AND scope = robot_id
  GROUP BY label
),
site_rates AS (
  SELECT f.label, f.task_text, avg(f.ok) AS site_ok
  FROM flags f JOIN robot_site r ON r.label = f.label AND f.site_id = r.site
  WHERE f.active AND NOT f.covered GROUP BY f.label, f.task_text
),
site_rest AS (
  SELECT f.label, avg(f.ok) AS site_rate
  FROM flags f JOIN robot_site r ON r.label = f.label AND f.site_id = r.site
  WHERE f.active AND NOT f.covered GROUP BY f.label
),
within AS (
  SELECT f.label, avg(coalesce(s.site_ok, o.site_rate)) - avg(f.ok) AS within_site_gap
  FROM flags f JOIN robot_site r USING (label) LEFT JOIN site_rates s USING (label, task_text)
    LEFT JOIN site_rest o USING (label)
  WHERE f.covered AND f.active GROUP BY f.label
),
shared AS (
  SELECT a.label, count(DISTINCT a.episode_id) AS shared_failures
  FROM flags a JOIN flags b ON a.episode_id = b.episode_id AND a.label <> b.label
  WHERE a.covered AND a.active AND b.covered AND b.active AND a.ok = 0
  GROUP BY a.label
),
cells AS (
  -- Each observation belongs to one period/task/membership cell per candidate.
  SELECT label, active, task_text,
    CASE WHEN covered THEN 'target' WHEN comparison THEN 'clean' ELSE 'other' END AS membership,
    count(*) AS n, avg(ok) AS rate
  FROM marked GROUP BY label, active, task_text, membership
),
target_weights AS (
  SELECT c.*, c.n / sum(c.n) OVER (PARTITION BY c.label, c.active) AS weight,
    -- Match scored's coalesce order exactly, including pooled clean before task rest.
    CASE WHEN t.clean_ok IS NOT NULL THEN 'clean_task'
      WHEN r.clean_rate IS NOT NULL THEN 'clean_pool'
      WHEN t.rest_ok IS NOT NULL THEN 'rest_task'
      WHEN r.rest_rate IS NOT NULL THEN 'rest_pool' END AS comparison_pool,
    CASE WHEN t.clean_ok IS NOT NULL THEN t.clean_n
      WHEN r.clean_rate IS NOT NULL THEN r.n_comparison
      WHEN t.rest_ok IS NOT NULL THEN t.rest_n
      WHEN r.rest_rate IS NOT NULL THEN r.rest_n END AS pool_n
  FROM cells c LEFT JOIN task_rates t USING (label, active, task_text)
    LEFT JOIN rest r USING (label, active)
  WHERE c.membership = 'target'
),
coefficient_terms AS (
  SELECT label, active, task_text, membership, n, rate, -weight AS coefficient
  FROM target_weights
  UNION ALL
  SELECT c.*, w.weight * c.n / w.pool_n AS coefficient
  FROM target_weights w JOIN cells c ON c.label = w.label AND c.active = w.active
    AND c.membership <> 'target'
    AND CASE w.comparison_pool
      WHEN 'clean_task' THEN c.membership = 'clean' AND c.task_text = w.task_text
      WHEN 'clean_pool' THEN c.membership = 'clean'
      WHEN 'rest_task' THEN c.task_text = w.task_text
      WHEN 'rest_pool' THEN TRUE ELSE FALSE END
),
coefficients AS (
  -- A control cell can serve several tasks or overlapping pools: add its weights first.
  SELECT label, active, task_text, membership, n, rate, sum(coefficient) AS coefficient
  FROM coefficient_terms GROUP BY label, active, task_text, membership, n, rate
),
wilson AS (
  SELECT c.*,
    (rate + p.z * p.z / (2 * n)
      - p.z * sqrt(rate * (1 - rate) / n + p.z * p.z / (4.0 * n * n)))
      / (1 + p.z * p.z / n) AS low,
    (rate + p.z * p.z / (2 * n)
      + p.z * sqrt(rate * (1 - rate) / n + p.z * p.z / (4.0 * n * n)))
      / (1 + p.z * p.z / n) AS high
  FROM coefficients c, interval_params p
),
support AS (
  SELECT label, active, bool_and(comparison_pool IS NOT NULL) AS supported
  FROM target_weights GROUP BY label, active
),
period_uncertainty AS (
  -- MOVER uses Wilson distances from the raw rate; no second multiplication by z.
  SELECT label, active,
    CASE WHEN bool_and(s.supported) THEN
      sum(coefficient * coefficient * CASE WHEN coefficient > 0
        THEN pow(rate - low, 2) ELSE pow(high - rate, 2) END) END AS low_sq,
    CASE WHEN bool_and(s.supported) THEN
      sum(coefficient * coefficient * CASE WHEN coefficient > 0
        THEN pow(high - rate, 2) ELSE pow(rate - low, 2) END) END AS high_sq
  FROM wilson JOIN support s USING (label, active) GROUP BY label, active
),
intervals AS (
  SELECT s.*, s.successes / s.n AS rate, sqrt(a.low_sq) AS low_distance,
    sqrt(a.high_sq) AS high_distance,
    -- Post minus pre reverses every pre coefficient, exchanging its two distances.
    sqrt(a.low_sq + b.high_sq) AS change_low_distance,
    sqrt(a.high_sq + b.low_sq) AS change_high_distance
  FROM summary s LEFT JOIN period_uncertainty a ON a.label = s.label AND a.active
    LEFT JOIN period_uncertainty b ON b.label = s.label AND NOT b.active
  WHERE s.n > 0
),
projections AS (
  SELECT i.*, CASE WHEN CAST(p.project_to_day AS DOUBLE) >= 0
      AND isfinite(CAST(p.project_to_day AS DOUBLE))
      AND i.projection_observed_days >= 2 AND isfinite(i.slope)
      AND isfinite(i.intercept) AND isfinite(i.expected_rate)
      AND isfinite(i.intercept + i.slope * CAST(p.project_to_day AS DOUBLE))
      AND isfinite(100 * (i.intercept + i.slope * CAST(p.project_to_day AS DOUBLE))
        * i.share * i.attributable)
    THEN i.intercept + i.slope * CAST(p.project_to_day AS DOUBLE) END AS projected_gap
  FROM intervals i, params p
)
SELECT i.label, i.from_day, CAST(i.n AS INT) AS n, round(i.rate, 3) AS rate,
  round(i.expected_rate, 3) AS expected_rate, CAST(i.n_comparison AS INT) AS n_comparison,
  round(i.expected_rate - i.rate, 3) AS gap,
  floor(1000 * (i.expected_rate - i.rate - i.low_distance)) / 1000 AS gap_low,
  ceil(1000 * (i.expected_rate - i.rate + i.high_distance)) / 1000 AS gap_high,
  round(i.share, 3) AS share, i.attributable,
  round(100 * (i.expected_rate - i.rate) * i.share * i.attributable, 2) AS gain_points,
  floor(10000 * (i.expected_rate - i.rate - i.low_distance) * i.share * i.attributable) / 100 AS gain_low,
  ceil(10000 * (i.expected_rate - i.rate + i.high_distance) * i.share * i.attributable) / 100 AS gain_high,
  CAST(i.n_before AS INT) AS n_before, round(i.gap_before, 3) AS gap_before,
  round(100 * (i.expected_rate - i.rate - i.gap_before) * i.share * i.attributable, 2)
    AS gain_points_change,
  floor(10000 * (i.expected_rate - i.rate - i.gap_before - i.change_low_distance)
    * i.share * i.attributable) / 100 AS gain_change_low,
  ceil(10000 * (i.expected_rate - i.rate - i.gap_before + i.change_high_distance)
    * i.share * i.attributable) / 100 AS gain_change_high,
  round(i.slope, 4) AS slope_per_day,
  i.projection_observed_days, i.projection_first_day, i.projection_last_day,
  round(100 * i.projected_gap * i.share * i.attributable, 2) AS raw_projected_gain_points,
  CASE WHEN i.projected_gap IS NOT NULL THEN round(100 *
    least(i.expected_rate, greatest(i.expected_rate - 1, i.projected_gap))
    * i.share * i.attributable, 2) END AS projected_gain_points,
  CASE WHEN i.projected_gap IS NOT NULL THEN
    i.projected_gap NOT BETWEEN i.expected_rate - 1 AND i.expected_rate END AS projection_clipped,
  CAST(coalesce(sh.shared_failures, 0) AS INT) AS shared_failures,
  round(w.within_site_gap, 3) AS within_site_gap
FROM projections i LEFT JOIN shared sh USING (label) LEFT JOIN within w USING (label)
ORDER BY gain_points DESC NULLS LAST
```
