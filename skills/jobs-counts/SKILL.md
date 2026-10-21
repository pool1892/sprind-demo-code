---
name: jobs-counts
description: Counts and intervals for job runs. Literal counts below twenty starts and a 95% Wilson interval from twenty; two programs compared on the same starts as counts of rescued and broken starts.
when_to_use: Whenever a number about job runs goes into notes, a decision or the account to the shift lead, and whenever two programs are compared on the same starts.
tools_used: [query, job_results, operator_state, memory_read, memory_write]
inputs:
  - the job ids of runs that have come back
  - for a comparison, two runs on the same starts
outputs:
  - delivered counts per run, with an interval where at least twenty starts allow one
  - rescued and broken starts between two programs, as counts with the starts named
version: "2"
---
# Counts and intervals for job runs

A job run is a few starts, so its numbers are small and exact: "7 of 10
delivered" says what happened and nothing more. This skill keeps the numbers
honest at that size.

Run the blocks as `jobs-traces` describes.

## The rules for numbers

- **Counts first.** Report a run as "d of n delivered", with the program and
  the starts. Below twenty starts give the count only, with no rate and no
  interval: a rate with decimals suggests a precision a count of a few starts
  does not have.
- **Intervals from twenty starts.** Where a count rests on at least twenty
  starts, give the 95% Wilson interval with it (`delivery_counts`; it is the
  interval the fleet skills use too). Twenty-five starts with 22 delivered
  give an interval of roughly 70% to 96%: wide, and that width is the honest
  statement.
- **Same starts for comparisons.** A start is a base-task identifier and a
  stored initial-state index. The same public start identifier denotes the
  same starting conditions, so two programs compared on the same starts
  differ only by the program. From the first moment the two programs act
  differently, though, the policy sees different scenes, so a start can come
  out differently for reasons that have nothing to do with the change.
  Compare them start by start (`paired`): a *rescued* start is delivered by
  the new program and not by the old; a *broken* start the other way round.
  Report both as counts, with the starts they are, and open those starts in
  the traces to see whether the change is what made the difference.
- **Fresh starts for claims.** Counts on the starts you tuned a program on
  are optimistic: changes were kept because they helped there. A count that
  says how often a program delivers comes from starts it was not tuned on.
- **No pooling across programs.** Each count belongs to one program digest.
  Do not add the runs of two programs into one count.
- **Imported counts stay their own.** A count of an imported job belongs to
  its investigation, its request and its starts: report it with them and
  never add it to a count of this investigation's runs. An earlier
  investigation's final evaluation ran on starts that are examined now: for
  this investigation it is evidence about what failed, not a fresh
  confirmation of anything.

## The acceptance condition and the release

The acceptance condition is set before any program runs: the released
program runs once on N reserved starts and must be delivered on at least K of
them (`jobs-release`). The numbers that go beside it later are counts: the
program's count on fresh development starts, and the release's count against
K. Report them as they are; the condition is the shift lead's bar, and a
count below twenty starts is given without a rate or an interval.

## Procedure

1. `delivery_counts` for every run you report: `episodes`, `delivered`,
   `order_broken`, and from `min_n` (twenty) starts the rate with its 95%
   interval. Below `min_n` the rate and interval stay empty on purpose.
   `programs` should be 1: a run plays one program.
2. `paired` for two runs on the same starts (the input names the earlier run
   as `job_a` and the changed program as `job_b`): pairs matched, both
   delivered, rescued, broken, neither, the starts that were rescued and
   broken, and the starts that were not in both runs.
3. Write the numbers to memory with the block that produced each, and use
   them as they are in the account to the shift lead.

## Blocks

```sql
-- helper: delivery_counts
WITH params AS (SELECT 1.96 AS z, 20 AS min_n),
-- input: runs(job_id) AS (VALUES ('<job id>'), ('<job id>')),
runs AS (SELECT job_id FROM job_episodes GROUP BY job_id),
-- end input
c AS (
  SELECT job_id, left(any_value(program_digest), 12) AS digest, count(DISTINCT program_digest) AS programs,
    count(*) AS n, count(*) FILTER (WHERE delivered) AS d,
    count(*) FILTER (WHERE NOT order_kept) AS order_broken
  FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs) GROUP BY job_id
),
w AS (
  SELECT c.*, c.d / c.n AS rate,
    (c.d / c.n + p.z * p.z / (2 * c.n)) / (1 + p.z * p.z / c.n) AS centre,
    p.z * sqrt(c.d / c.n * (1 - c.d / c.n) / c.n + p.z * p.z / (4 * c.n * c.n)) / (1 + p.z * p.z / c.n) AS half,
    p.min_n
  FROM c, params p
)
SELECT job_id, digest, CAST(programs AS INT) AS programs, CAST(n AS INT) AS episodes,
  CAST(d AS INT) AS delivered, CAST(order_broken AS INT) AS order_broken,
  CAST(d AS VARCHAR) || ' of ' || CAST(n AS VARCHAR) AS count_text,
  CASE WHEN n >= min_n THEN round(rate, 3) END AS rate,
  CASE WHEN n >= min_n THEN round(centre - half, 3) END AS rate_low,
  CASE WHEN n >= min_n THEN round(centre + half, 3) END AS rate_high
FROM w ORDER BY job_id
```

```sql
-- helper: paired
WITH
-- input: pair(job_a, job_b) AS (VALUES ('<earlier job id>', '<changed job id>')),
pair AS (SELECT min(job_id) AS job_a, max(job_id) AS job_b FROM job_episodes),
-- end input
a AS (
  SELECT j.base_task, j.init_state, j.delivered, j.episode_id
  FROM job_episodes j WHERE j.job_id = (SELECT job_a FROM pair)
),
b AS (
  SELECT j.base_task, j.init_state, j.delivered, j.episode_id
  FROM job_episodes j WHERE j.job_id = (SELECT job_b FROM pair)
),
m AS (
  SELECT a.delivered AS da, b.delivered AS db, a.episode_id AS ea, b.episode_id AS eb,
    coalesce(a.init_state, b.init_state) AS init_state
  FROM a FULL JOIN b USING (base_task, init_state)
),
t AS (
  SELECT count(*) FILTER (WHERE ea IS NOT NULL AND eb IS NOT NULL) AS pairs,
    count(*) FILTER (WHERE da AND db) AS both_delivered,
    count(*) FILTER (WHERE NOT da AND db) AS rescued,
    count(*) FILTER (WHERE da AND NOT db) AS broken,
    count(*) FILTER (WHERE NOT da AND NOT db) AS neither,
    count(*) FILTER (WHERE ea IS NULL OR eb IS NULL) AS unmatched,
    list(init_state ORDER BY init_state) FILTER (WHERE NOT da AND db) AS rescued_starts,
    list(init_state ORDER BY init_state) FILTER (WHERE da AND NOT db) AS broken_starts
  FROM m
)
SELECT (SELECT job_a FROM pair) AS job_a, (SELECT job_b FROM pair) AS job_b,
  CAST(pairs AS INT) AS pairs, CAST(both_delivered AS INT) AS both_delivered,
  CAST(rescued AS INT) AS rescued, CAST(broken AS INT) AS broken, CAST(neither AS INT) AS neither,
  CAST(unmatched AS INT) AS unmatched_starts, rescued_starts, broken_starts
FROM t
```
