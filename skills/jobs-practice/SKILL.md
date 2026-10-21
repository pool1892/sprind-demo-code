---
name: jobs-practice
description: The practice loop for making a job work with few starts. Run about ten development starts, look at every failure, name the moment it went wrong and one explanation, change one thing in the program, run the changed program on the same starts, compare start by start, keep what helped and drop what did not, and confirm on fresh starts before a release. With a ledger of every program tried and blocks that list the programs tried and the starts of a run for reuse.
when_to_use: From the first program to the one you save. At the start of every session after the first, to see where the practice stands; before every submit_job_program, to choose the change and the starts.
tools_used: [query, submit_job_program, job_results, operator_state, memory_read, memory_write]
inputs:
  - the budget (operator_state)
  - the runs so far (job_results) and the cards and patterns in memory
outputs:
  - the ledger of programs tried, kept in memory
  - the next job requests, each with one change and the starts it runs on
version: "2"
---
# Practice with few starts

A job is made to work the way a technician makes a machine work: try it,
watch what goes wrong, change one thing, try again on the same pieces, keep
what helped. Each start is one try. A few of them, looked at closely, say more
about why a program fails than many of them counted. Counting comes at the
end, on starts the program was not tuned on (`jobs-counts`).

Run the blocks as `jobs-traces` describes.

## The loop

In a numbered investigation, begin with the earlier investigations' records
(`jobs-traces`, Imported jobs): what they tried, what their final evaluation
found, and the failures that remained. The patterns their evaluation failed
on are the first to explain; a change that was tried there and did not help
needs a reason before it is tried again.

1. **A first program.** The simplest program that could do the job
   (`jobs-decompose`), on about ten development starts with video. Its
   failures are the first evidence; nothing is known before it.
2. **Look at every failure.** Cards for each (`jobs-traces`), what each goal
   did along the way (`jobs-preserve`), and the state at each switch
   (`jobs-transitions`). Look at a delivered episode of the same program too:
   it shows what the failures lack.
3. **Name the moment and one explanation.** For the most frequent pattern:
   the program step and control step where it goes wrong, what the records
   show there, and one explanation with what it predicts for the next run
   ("if this step times out because its budget ends while the policy is still
   making progress, then a larger budget should turn those timeouts into
   reached steps and change nothing in the others").
4. **Change one thing.** One element of one step: an `until` condition, a
   step budget, a motion, a wait, an instruction, the order of two steps where
   the rules allow it. A change whose parts only make sense together counts
   as one; say so in its notes. Give the changed program a new name and a
   `notes` line that names its parent and the change.
5. **Same starts.** Run the changed program on the same starts as its parent
   (`starts_of` gives their stored initial states, which is how
   `submit_job_program` names starts). Then compare start by start
   (`jobs-counts` `paired`): rescued starts, broken starts.
6. **Look before you keep.** Open the rescued starts: did they deliver for the
   reason the explanation predicted? Open the broken ones: what did the change
   break? A change that rescues for another reason than the one you expected
   is a new finding, not a confirmation.
7. **Keep or drop.** Keep a change that rescued starts, broke none or fewer,
   and works by the mechanism you named. Drop one that does not, and go back
   to its parent. Record the decision either way.
8. **Next pattern.** Repeat from step 3 with the failures that remain.
9. **Confirm on fresh starts.** When the program delivers on the starts it
   was tuned on, run it on development starts it has not run on, as many as
   the budget allows (twenty or more gives an interval). That count, not the
   tuned one, is the evidence for a release (`jobs-counts`
   `delivery_counts`).

## Using the budget well

`operator_state` gives the budget: starts per request, requests per session,
sessions, starts in all. Requests run after the session ends, so a session is
one round of the loop. Plan the rounds before the first request, in memory:
how many rounds of changes, how many starts each, and what is kept back for
the fresh-start confirmation. A plan that spends everything on tuning has
nothing left to confirm with.

- Several requests in one session can test alternatives side by side: two
  different changes of the same parent, each on the same starts, compared
  with the parent in the next session. Each is still one change.
- The same public start identifier denotes the same starting conditions;
  repeating the same program on it is not a fresh trial: spend starts on
  changes and on fresh starts. Two different programs, though, make the
  policy see different scenes from the first moment they differ, so a start
  can turn out differently for reasons unrelated to the change. That is why
  rescued and broken starts are opened and read, not only counted, and why a
  claim rests on fresh starts.
- Record video in every development run you will look at: the frames are
  the only way to see what the records describe.
- Starts that every program delivers and starts that no program delivers say
  little about the difference between programs. The informative ones are
  those that some programs deliver and others do not.

## The ledger

Keep one line per program in memory, and update it after every run:

```
<program name> (digest <first 12>), parent <name>, change: <one line>
  run <job id> on <which starts>: <d> of <n> delivered; against parent: rescued <r>, broken <b>
  what the traces showed: <one line>; kept | dropped, because <one line>
```

`tried` lists from the tables every program that has run, with its runs,
starts and counts, so that the ledger can be checked against the records.

## When to stop

Save when you judge the program ready for its release, not when one batch
first reaches the condition. Ready means three things: the program has
delivered on a confirmation of development starts it was not tuned on; you
have read the trace of every failure that remains and know what each one
did; and that count, set beside the approved condition, makes you willing to
stake the release on it (`jobs-release`). A count that only just reaches the
condition, or that rests on the starts the program was tuned on, is a reason
to confirm further, not to save.

When two changes in a row to the same approach leave its count where it was,
try a materially different intervention before tuning that approach again:
another transition, another order where the rules allow it, another use of
the vocabulary. Record in the ledger why the earlier approach stalled.

Saving freezes the program and spends the investigation's one release. Stop
without a release when the budget is spent, or when the failures that remain
need something the program's vocabulary cannot express; say which, with the
evidence, in the account (`submit_job_account` with `job` null). Stopping
without a release is a valid outcome and is recorded as one.

## Blocks

```sql
-- helper: tried
WITH ev AS (
  SELECT episode_id,
    string_agg(CAST(program_step AS VARCHAR), ' ' ORDER BY program_step)
      FILTER (WHERE outcome = 'timeout' AND kind <> 'tail') AS timeouts
  FROM job_step_events GROUP BY episode_id
)
SELECT left(j.program_digest, 12) AS digest,
  string_agg(DISTINCT j.job_id, ' ' ORDER BY j.job_id) AS runs,
  count(*) AS episodes,
  count(DISTINCT (j.base_task, j.init_state)) AS distinct_starts,
  count(*) FILTER (WHERE j.delivered) AS delivered,
  count(*) FILTER (WHERE NOT j.order_kept) AS order_broken,
  count(*) FILTER (WHERE ev.timeouts IS NOT NULL) AS with_timeout,
  list(DISTINCT j.init_state ORDER BY j.init_state) AS init_states
FROM job_episodes j LEFT JOIN ev USING (episode_id)
GROUP BY left(j.program_digest, 12)
ORDER BY min(j.job_id)
```

```sql
-- helper: starts_of
WITH
-- input: runs(job_id) AS (VALUES ('<job id>')),
runs AS (SELECT min(job_id) AS job_id FROM job_episodes),
-- end input
s AS (
  SELECT job_id, base_task, init_state, delivered
  FROM job_episodes WHERE job_id IN (SELECT job_id FROM runs)
)
SELECT job_id, any_value(base_task) AS base_task, count(DISTINCT base_task) AS base_tasks,
  list(init_state ORDER BY init_state) AS init_states,
  list(init_state ORDER BY init_state) FILTER (WHERE delivered) AS delivered_on,
  list(init_state ORDER BY init_state) FILTER (WHERE NOT delivered) AS not_delivered_on
FROM s GROUP BY job_id ORDER BY job_id
```

Next: `jobs-counts` for every comparison and count, `jobs-check` to keep the
question you kept asking as a saved check, `jobs-release` at the end.
