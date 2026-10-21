---
name: jobs-check
description: Write, test and save a check - a named SQL query with parameters over the job tables that answers one question about a run's episodes, saved with its purpose and what a healthy run returns, so that the next shift can run it from the command line and a later investigation can call it with run_check. What makes a check useful beyond the delivery verdict, its format, how to test it against episodes you have looked at, and how to use a check from an earlier job.
when_to_use: When the same question has been asked of several runs during practice and its answer decides whether a run is good; before saving a job, whose check goes with it; and when job_library lists checks from earlier requests.
tools_used: [query, write_check, run_check, job_library, memory_read, memory_write]
inputs:
  - the question you kept asking of runs during practice, and the blocks that answered it
  - runs whose episodes you have looked at, for testing
outputs:
  - a saved check (write_check) with its name, purpose, parameters, SQL and expectation
  - the check's result on runs you know, compared with your cards
version: "1"
---
# Checks the next shift can run

The delivery check says, at the end of an episode, whether the job was
delivered. It does not say why not, or which part of the job is fragile even
when it was delivered. A check is your own question, kept: one SQL query
over the job tables that you asked of several runs while making the job
work, written so that someone else can ask it of a new run without you. The
operator runs it from the command line on any run directory; a later
investigation calls it with `run_check`. It is read-only and runs on the
recorded tables, never on the robot.

## What a check is

A check (`write_check`) has:

- `name`: short, lower case letters, digits and hyphens, three to forty
  characters, saying what it checks;
- `purpose`: one or two sentences: the question it answers and when to run
  it;
- `parameters`: a list of `{name, type}`, the type one of `int`, `float`,
  `str` and `bool`; the SQL refers to each as `$name`. Every parameter the
  SQL uses is declared, and every declared one is used;
- `sql`: one SELECT over the job tables (`job_episodes`, `job_step_events`,
  `goal_steps`, `object_steps`, `robot_steps`);
- `expects`: what a healthy run returns, in words ("every row has verdict
  pass"; "no rows").

`write_check` takes the check and, in `check_with`, a value for each of its
parameters: the installation runs the SQL once with them to make sure it
works, and does not keep them. `run_check` runs a saved check with values for
its parameters, over one job run's episodes (its `job_id`) or over every run
so far (`job_id` null), and returns its rows. The operator runs it the same
way from the command line.

## What makes a check useful

- **One question, one row per episode** (or per episode and goal, or per
  episode and step), with a verdict column and the evidence columns that
  justify it. A reader sees at once which episodes fail it and why.
- **Parameters for everything that belongs to the request**: the run
  (`$job_id`), goal indices, object names, step numbers, thresholds. A check
  that names this request's objects or numbers in its SQL works only here; one
  that takes them as parameters can serve the next request.
- **It asks what the delivery verdict does not.** Delivered or not is already
  recorded. A check earns its place by saying something more: which part of
  the job failed, at which moment, or that a delivered episode came close to
  failing.
- **Its expectation is testable.** `expects` states what a good run returns,
  so that a run that does not return it stands out.

## Procedure

1. From your notes, name the question that decided your changes during
   practice, and the blocks that answered it (from `jobs-preserve`,
   `jobs-transitions`, `jobs-traces`). The check distils that question; it
   need not reproduce every column of the blocks.
2. Write the SQL with `$name` parameters. Test it with `query` first, with
   the parameters written in as values, on a run whose episodes you have
   looked at.
3. Compare its rows with your cards: every episode you know failed for the
   reason the check is about should fail it; the episodes you know were fine
   should pass. Run it also on a run of another program (a parent and its
   changed program). A check that agrees with your cards on both is ready; one
   that does not is wrong, or your cards are, and you find out which before you
   save it.
4. Save it with `write_check`, with the values you tested it with in
   `check_with`. Run it once with `run_check` on the same run, to confirm the
   saved form returns what the tested form did.
5. Write its name, purpose and the test result to memory. `save_job` names the
   check that goes with the job.

## A check from an earlier job

When `job_library` lists checks from earlier requests, read each one's
purpose, parameters and expectation. A check made for another job answers its
question on this job's tables only if the question applies here: run it on a
run you have looked at and compare with your cards before you rely on it.
Using it, adapting it into a new check of your own, or ignoring it are all
fine; say which and why in your notes.

## The format, by example

A check of a different question, to show the form only: which episodes of a
run had a given program step time out.

```
name: step-timeouts
purpose: Which episodes of a run had the given program step run out of steps before
  its until condition held. Run after any job run.
parameters: [{"name": "job_id", "type": "str"}, {"name": "program_step", "type": "int"}]
expects: every row has verdict pass
```

```sql
-- check: step-timeouts
SELECT j.episode_id, s.program_step, s.outcome, s.end_step - s.start_step AS steps_used,
  CASE WHEN s.outcome = 'timeout' THEN 'fail' ELSE 'pass' END AS verdict
FROM job_episodes j JOIN job_step_events s USING (episode_id)
WHERE j.job_id = $job_id AND s.program_step = $program_step
ORDER BY j.episode_id
```

Next: `jobs-release` to save the job with its check.
