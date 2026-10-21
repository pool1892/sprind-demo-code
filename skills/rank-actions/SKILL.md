---
name: rank-actions
description: Turn diagnosed causes into catalogue actions with an honest gain, interval and cost each, rank them so that the first k are the best set of k, decide which at most three to adopt this round (possibly none) and which earlier changes to undo or replace, and forecast the gain of the whole set with a range instead of adding single gains.
when_to_use: After the gains are estimated (sensitivity-from-logs, and sensitivity-by-experiment where it ran), before write-diagnosis. Again whenever a gain changes. In a later round, also for the changes adopted earlier.
tools_used: [query, memory_read, memory_write]
inputs:
  - the causes with their targets and directions (scope)
  - the expected gains with intervals (sensitivity-from-logs, sensitivity-by-experiment)
  - the action catalogue, with costs and task classes (the table catalogue)
  - from the second round on, the changes adopted earlier (their ledger entry ids, rationales and forecasts, from the task and memory)
outputs:
  - a ranked list of candidate actions, each with expected gain, low and high, cost in hours and a rationale
  - the changes adopted this round (at most three, possibly none), the earlier changes undone or replaced, or the decision that no further change is worth making
  - the forecast of the whole set's gain, with low, mid and high
version: "6"
---
# Rank actions

A round ends with a decision: which changes the fleet adopts now and keeps.
The decision holds at most three changes, new ones and replacements of
earlier ones counted together, and none is a legitimate answer. It can also
undo an earlier change; an undo does not count against the three. It
carries one forecast for the set as a whole, with a range, and, if you
wish, the ranked alternatives: the further actions you weighed and did not
adopt, best first. Alternatives are optional; none is valid, and a decision
to change nothing or to stop needs none.

Your working list, adopted actions on top and alternatives below, does
three jobs:

- it is how you choose the set: the first k actions should be the best set
  of k you can name;
- it records the alternatives you weighed, each with an honest gain;
- round one's adopted actions in order, then its alternatives in order,
  cut at the number of changes in force at the end, are the plan you would
  have fixed after a single look, which is compared with what the second
  round reached. So order round one's list as you would adopt if you could
  only look once, and list as alternatives only actions that could be
  adopted.

Run the blocks as `triage` describes: copy a block whole, change only its
`params` values and the rows between `-- input` and `-- end input`, run it
with `query`.

## From a cause to an action

An action is a verb from the catalogue on a target at a scope. The
catalogue is the visible table `catalogue`: one row per valid verb, scope
and component, configuration key or condition, with its cost in `hours`;
`value` holds the task class of a `route_to_teleop` row and is empty where
you choose the value (`set`, `change_condition`, `request_training_data`).
Check every action against it (`SELECT * FROM catalogue WHERE scope =
'<scope>' AND verb = '<verb>'`): one that has no row there is invalid.

- **A component of a robot that is degraded**: compare the permitted verbs
  at robot scope. Estimate efficacy from this fleet's observations or an
  experiment. Neither eligibility in the catalogue nor price establishes
  which mechanism an action repairs or how much it recovers.
- **A configuration value that was changed or is wrong**: `revert` the key
  at the scope where the change was made, exactly that scope: a revert at
  another scope does nothing, because a revert undoes the change recorded
  there. Or `set` the key to the value it had before, at that scope or a
  narrower one; a value set at a robot overrides its site and the fleet for
  that robot only. When the change is in the change log, `revert` at its
  scope is the safer of the two: it needs no value and cannot set a wrong
  one. `set` needs its `value`: the change log's `old`, with its JSON
  decoded (`0` stays the number 0, `"1.4.0"` becomes the text 1.4.0, without
  the quotes).
- **A software version**: `revert` `software.version` at the scope of the
  change.
- **A site condition**: `change_condition` on that key at the site, with the
  value to change to (what the other sites have, or what the site had).
- **A circumstance no action removes** (something the public description
  says stays, or a cause the catalogue cannot reach): look for a setting
  under which the robots work better despite it, and measure it
  (sensitivity-by-experiment, a sweep). A `set` of a setting is an action
  like any other, with its own gain and interval.
- **A mode no repair removes** (the policy's own weakness):
  `route_to_teleop` and `request_training_data` have no simulated effect
  on what the robots do, so they cannot be adopted as changes and win
  nothing in fleet success. Report the mode in the diagnosis' retrain
  indicators instead, and keep them out of the decision and its ranked
  candidates.

An action already in force (adopted earlier and still active) is not a new
change; the decision refuses it.

## Earlier changes (from the second round on)

Every change adopted in an earlier round is an entry of the fleet's ledger
(`L1`, `L2`, ...), with its rationale and the forecast made for it. The
fleet has operated with it since. For each active entry:

- compare what the records since its adoption show with what was predicted
  for it (scope's `changes` and `breakdown` around the adoption day, on the
  robots it reaches);
- keep it when it did what it was meant to do;
- **undo** it when the records show no gain or a loss, and nothing suggests
  the records are too few to tell;
- **replace** it when a better action on the same cause has appeared (a
  setting whose better value an experiment measured, a cheaper repair that
  does as well).

Only settings and conditions can be undone or replaced. A component verb
(a recalibration, a cleaning, a new part) is physical work and cannot be
taken back; a later component verb is a new change. An undo or a
replacement is part of this round's decision and of its set forecast, with
its own gain over the fleet as it now operates.

## When no further change is worth making

Adopt nothing, and say so, when no candidate has evidence of a gain: every
interval reaches zero and no experiment is left to settle it, or the
remaining failures sit where no catalogue action reaches. A change with no
evidence still costs operator time and can break attempts that work today.
"Nothing further is worth changing" is a result, not a failure; the
decision's `stop` says it.

An uncertain change can still be worth adopting when it is cheap,
reversible and the round's experiments could not settle it: the next round's
records test it, and an undo takes it back. Say so in its rationale and
let the forecast's range show the uncertainty.

## Procedure

1. List the candidate actions for every cause, with the alternatives: the
   other member of a confounded pair, different permitted verbs on a
   component, the same change at the site and at its robots, the values of
   a setting that were measured. Take each expected gain, low and high from
   the gain estimates; a measured gain from an experiment replaces a
   log-based one.
2. `expected_gain` is the gain of the action **on its own**, in points of
   fleet success over the fleet as it now operates. An action on a healthy
   target wins nothing: give it zero, not a small hopeful number.
3. Order the list for the set, not only by the gain: put first the action
   that wins most; then the one that adds most to what is above it. Two
   actions shown to fix the same failures do not add: the second adds only
   the failures the first leaves, so it moves down even if its own gain is
   large. Say so in its rationale. Never list an action twice.
4. Decide how many of the top actions to adopt: none to three. Take an
   action only when its evidence says it helps (or when the reason above
   for an uncertain one holds), and stop at the first action that does not
   earn its place, even if fewer than three are taken. Decide the undos and
   replacements of earlier entries. A replacement's new action stands among
   the adopted ones at the top of the working list too, and it counts
   against the three. The rows below the adopted ones are the alternatives
   the decision may carry. They pass through `cut` like the changes
   (catalogue, values, simulated effect), whatever the decision tool itself
   checks of them.
5. Run `cut` (below) with the list, one input row per action: rank, verb,
   scope, the component or key, the value as text in single quotes (`'0'`,
   `'1.4.0'`; `NULL` when the verb takes none: text for every value, so
   that rows with a number and rows with a word fit one column), expected
   gain, low, high; and `adopt`, the number of top actions the decision
   takes (new changes and replacements). It
   looks each action up in the table `catalogue`, takes its hours from
   there, adds the running hours, marks the `adopted` actions (down the
   list in order, skipping a repeat), and reports problems in `checks`. Fix
   what `checks` reports. An expensive action high up may have a cheaper
   action on the same cause below it (recalibrate before replace) that does
   as well; note the trade-off in the rationale.
6. Run `set_gain` (below) for the set: every adopted change, replacement and
   undo, one row each, grouped in parts. Its last row is the forecast for the
   whole set.
7. Write to memory: the final list, each action with its gain, interval,
   hours, rationale and the tool calls its numbers come from; the adopted
   set, the undos and replacements; the set forecast; and what the next
   round's records should show if the set works.

Next: `write-diagnosis`.

## The set forecast

Single gains are not added up. Repairs that reach the same robots overlap
(two of them may remove the same failures) and can interact, and the sum of
their singles overstates what the set wins. `set_gain` makes the forecast
from parts:

- **A part** is a group of robots that no other part's changes reach: a
  robot, or a site's robots. Changes that reach overlapping robots (a site
  change and a change at one of its robots, two changes at one robot) go in
  the same part. Give each row the part's `share` of the fleet's attempts
  and its current success `rate` (from `outcomes` or `gains`); the most a
  part can win is what it loses today, `100 x share x (1 - rate)` points.
- **Within a part**, one change gives its own gain. Several changes not
  measured together are combined as if each removed its own fraction of the
  failures still left (independent causes): more than the best of them, less
  than their sum; expected harms (negative gains) add up, at most to every
  success the part has. The range allows
  for the worst overlap on each side. At the low end the benefits may be the
  very same failures, so only the best positive low end counts, and the
  harms may fall on different attempts that succeed today, so negative low
  ends add up, at most to every success the part has. At the high end the
  benefits may be different failures, so positive high ends add up, at most
  to what the part loses, and the harms may be the same attempts, so only
  the mildest certain harm counts. Whenever a change's interval reaches
  below zero, the block warns, whatever the point estimate says: changes
  that each might cost a little can together cost more, and only a test of
  the set settles it. When an experiment measured the changes together (a
  combined arm), give the part one row with that measurement instead, and
  name all its changes in the label.
- **Across parts**, gains add, because fleet success is the average over
  robots and the parts share none. The range combines the parts' half-widths
  as independent errors (the square root of the sum of their squares).
- An undo or a replacement is a row like any other: its gain is what the
  fleet gains by it over the current state, which is negative if you expect
  it to cost something (then say why you take it).

The forecast is points of fleet success over the fleet as it now operates;
its low, mid and high are the set row's `low`, `gain` and `high`. When you
doubt the model (changes that might interfere, a part measured on few
attempts), widen the range in `query` and say why. With no change, the
forecast is zero.

## Costs

Operator hours per action, parts included, are on every row of the table
`catalogue`, and per verb in the table `costs` (the verb, the component or
the kind of key it names, the hours). A verb that does not apply to a
component has no row in either; read the actual catalogue for eligibility.
`request_training_data` is not costed (empty hours); `cut` counts it as
zero in the sums. Hours are information for the choice between actions on
the same cause; the round's limit is the number of changes, not hours.

## Blocks

```sql
-- helper: cut
WITH params AS (SELECT 3 AS adopt, 3 AS max_adopt),
-- input: ranked(rank, verb, scope, thing, value, expected_gain, low, high) AS (VALUES (1, '<verb>', '<scope>', '<component or key>', NULL, 4.0, 1.5, 6.5), (2, 'set', '<scope>', '<key>', '<value as text>', 2.0, 0.0, 6.0)),
ranked(rank, verb, scope, thing, value, expected_gain, low, high) AS (VALUES
  (1, 'recalibrate', '<robot id>', 'scene_camera', NULL, 4.0, 1.5, 6.5),
  (2, 'revert', '<site id>', 'software.version', NULL, 3.0, 0.5, 6.0),
  (3, 'replace', '<robot id>', 'gripper', NULL, 2.5, 1.0, 4.0),
  (4, 'set', '<site id>', 'software.version', '<old version>', 2.0, 0.0, 6.0),
  (5, 'route_to_teleop', 'fleet', NULL, '<task class>', 0.0, 0.0, 0.0)
),
-- end input
costed AS (
  -- each action's row in the table `catalogue`, which carries its hours; a task class is
  -- part of a route_to_teleop row, other values are the agent's to choose
  SELECT CAST(r.rank AS INT) AS rank, r.verb, r.scope, r.thing, CAST(r.value AS VARCHAR) AS value,
    CAST(r.expected_gain AS DOUBLE) AS gain, CAST(r.low AS DOUBLE) AS low,
    CAST(r.high AS DOUBLE) AS high, k.hours, k.verb IS NOT NULL AS known,
    row_number() OVER (PARTITION BY r.verb, r.scope, r.thing, CAST(r.value AS VARCHAR)
      ORDER BY r.rank) = 1 AS first
  FROM ranked r LEFT JOIN catalogue k ON k.verb = r.verb AND k.scope = r.scope
    AND coalesce(k.component, k.config_key, k.condition_key) IS NOT DISTINCT FROM r.thing
    AND (k.value IS NULL OR k.value = CAST(r.value AS VARCHAR))
),
-- down the list in order, skipping a repeat: the first `adopt` distinct actions are adopted
checked AS (
  SELECT c.*, row_number() OVER (ORDER BY c.rank) AS position,
    sum(CAST(c.first AS INT)) OVER (ORDER BY c.rank) AS taken,
    lag(c.gain) OVER (ORDER BY c.rank) AS gain_above,
    count(*) OVER () AS list_size
  FROM costed c
),
marked AS (
  SELECT c.*, c.first AND c.taken <= p.adopt AS adopted, p.max_adopt
  FROM checked c, params p
)
SELECT c.rank, c.verb, c.scope, c.thing, c.value, c.gain AS expected_gain, c.low, c.high, c.hours,
  round(sum(CASE WHEN c.first THEN coalesce(c.hours, 0) ELSE 0 END) OVER (ORDER BY c.rank), 2)
    AS cum_hours,
  c.adopted,
  round(c.gain / nullif(c.hours, 0), 2) AS gain_per_hour,
  CAST(c.list_size AS INT) AS list_size,
  concat_ws('; ',
    CASE WHEN c.adopted AND c.taken > c.max_adopt
      THEN 'more changes adopted than a round allows (at most three)' END,
    CASE WHEN c.rank <> c.position THEN 'ranks must run 1, 2, 3 without gaps or repeats' END,
    CASE WHEN NOT c.first THEN 'the same action is listed twice' END,
    CASE WHEN NOT c.known THEN 'not in the catalogue: no such verb, scope and target' END,
    CASE WHEN c.known AND c.hours IS NULL THEN 'not costed, counted as zero hours' END,
    CASE WHEN c.low > c.high THEN 'low is above high' END,
    CASE WHEN c.gain < c.low OR c.gain > c.high THEN 'expected gain outside its interval' END,
    CASE WHEN c.gain > c.gain_above
      THEN 'larger gain than the action above: right only if it overlaps a higher action' END,
    CASE WHEN c.verb IN ('set', 'change_condition', 'route_to_teleop', 'request_training_data')
      AND c.value IS NULL THEN 'needs a value' END,
    CASE WHEN c.adopted AND c.verb IN ('route_to_teleop', 'request_training_data')
      THEN 'no simulated effect: cannot be adopted' END,
    CASE WHEN c.adopted AND c.low <= 0
      THEN 'adopted with an interval that reaches zero: say why in its rationale' END
  ) AS checks
FROM marked c
ORDER BY c.rank
```

```sql
-- helper: set_gain
WITH
-- input: changes(part, kind, label, share, rate, gain, low, high) AS (VALUES ('<robot or site id>', 'adopt', '<the change in words>', 0.25, 0.6, 4.0, 1.5, 6.5), ('<robot or site id>', 'undo', '<entry id and change in words>', 0.25, 0.6, 1.0, -0.5, 2.5)),
changes(part, kind, label, share, rate, gain, low, high) AS (VALUES
  ('<site id>', 'adopt', '<a change at the site>', 0.25, 0.55, 6.0, 2.0, 9.0),
  ('<site id>', 'adopt', '<another change reaching the same robots>', 0.25, 0.55, 4.0, 0.5, 7.0),
  ('<robot id>', 'replace', '<an earlier change replaced, at a robot of another site>', 0.083, 0.4, 3.0, 1.0, 4.5)
),
-- end input
typed AS (
  SELECT CAST(part AS VARCHAR) AS part, CAST(kind AS VARCHAR) AS kind, CAST(label AS VARCHAR) AS label,
    CAST(gain AS DOUBLE) AS gain, CAST(low AS DOUBLE) AS low, CAST(high AS DOUBLE) AS high,
    -- the most this part can win: the points it loses today; the most it can lose: its successes
    100 * CAST(share AS DOUBLE) * (1 - CAST(rate AS DOUBLE)) AS ceiling,
    100 * CAST(share AS DOUBLE) * CAST(rate AS DOUBLE) AS successes
  FROM changes
),
parts AS (
  SELECT part, count(*) AS n, string_agg(kind || ': ' || label, '; ' ORDER BY gain DESC) AS changes,
    max(ceiling) AS ceiling,
    -- independent removal: each change wins back its own fraction of what is still lost;
    -- expected harms add, at most every success of the part
    greatest(-max(successes),
      max(ceiling) * (1 - product(1 - least(1, greatest(0, gain) / nullif(ceiling, 0))))
        + sum(least(gain, 0))) AS mid,
    max(ceiling) * (1 - product(1 - least(1, greatest(0, gain) / nullif(ceiling, 0))))
      + sum(least(gain, 0)) < -max(successes) - 1e-9 AS beyond,
    -- the low end: the benefits may be the same failures (only the best positive low end
    -- counts) and the harms may be different successes (negative low ends add up, at most
    -- every success of the part)
    greatest(-max(successes),
      max(greatest(low, 0)) + sum(least(low, 0))) AS low,
    -- the high end: the benefits may be different failures (positive high ends add up, at
    -- most what the part loses) and the harms the same successes (only the mildest counts)
    greatest(-max(successes),
      least(max(ceiling), sum(greatest(high, 0))) + max(least(high, 0))) AS high,
    max(ceiling) - min(ceiling) > 1e-9 AS uneven, bool_or(gain < 0) AS harms,
    bool_or(low < 0) AS may_harm,
    bool_or(gain < low OR gain > high OR low > high) AS outside,
    bool_or(gain > ceiling + 1e-9) AS above,
    bool_or(kind NOT IN ('adopt', 'replace', 'undo')) AS unknown_kind
  FROM typed GROUP BY part
),
part_rows AS (
  SELECT 0 AS is_set, part, CAST(n AS INT) AS changes_in_part, changes, round(ceiling, 2) AS ceiling,
    -- outward to two decimals, after rounding away floating-point noise
    round(mid, 2) AS gain, floor(round(100 * least(low, mid), 6)) / 100 AS low,
    ceil(round(100 * greatest(high, mid), 6)) / 100 AS high,
    CASE WHEN n = 1 THEN 'one change'
      ELSE 'changes on the same robots: combined, not added; harms may add up' END AS how,
    concat_ws('; ',
      CASE WHEN may_harm AND n > 1 THEN 'the changes may cost points together: their harms '
        || 'can add up on the same robots; measure the set together (a combined arm)' END,
      CASE WHEN may_harm AND n = 1 THEN 'the change may cost points: its interval reaches '
        || 'below zero' END,
      CASE WHEN uneven THEN 'share or rate differ within the part: one part, one share and rate' END,
      CASE WHEN harms THEN 'a change expected to cost points: say why it is taken' END,
      CASE WHEN outside THEN 'a gain outside its interval' END,
      CASE WHEN above THEN 'a gain above what the part loses today' END,
      CASE WHEN beyond THEN 'harms beyond what the part can lose (its successes): held there' END,
      CASE WHEN unknown_kind THEN 'kind must be adopt, replace or undo' END) AS checks
  FROM parts
),
set_row AS (
  -- parts reach different robots: their gains add; the range combines their half-widths
  SELECT 1 AS is_set, 'set' AS part, CAST(sum(n) AS INT) AS changes_in_part,
    CAST(count(*) AS VARCHAR) || ' parts' AS changes, round(sum(ceiling), 2) AS ceiling,
    round(sum(mid), 2) AS gain,
    floor(round(100 * (sum(mid) - sqrt(sum(power(mid - least(low, mid), 2)))), 6)) / 100 AS low,
    ceil(round(100 * (sum(mid) + sqrt(sum(power(greatest(high, mid) - mid, 2)))), 6)) / 100
      AS high,
    'the forecast for the whole set: low, gain (mid) and high' AS how,
    concat_ws('; ',
      CASE WHEN sum(mid) - sqrt(sum(power(mid - least(low, mid), 2))) < 0
        THEN 'the set may cost points: its low end is below zero' END,
      CASE WHEN bool_or(may_harm OR uneven OR harms OR outside OR above OR beyond OR unknown_kind)
        THEN 'see the parts' END) AS checks
  FROM parts
)
SELECT part, changes_in_part, changes, ceiling, gain, low, high, how, checks
FROM (SELECT * FROM part_rows UNION ALL SELECT * FROM set_row)
ORDER BY is_set, part
```
