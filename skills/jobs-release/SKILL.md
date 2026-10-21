---
name: jobs-release
description: The beginning and the end of a request. Propose the acceptance condition (N reserved release starts, K required deliveries) before anything runs and get the shift lead's approval; at the end, save the job with its check for release, or stop without a release; and submit the account to the shift lead in plain words with every number from a tool result.
when_to_use: First, before any program runs (the acceptance proposal); last, when practice has a program ready or the budget or the vocabulary has run out; and when operator_state shows the release receipt, to submit the account.
tools_used: [operator_state, submit_acceptance_proposal, save_job, write_check, run_check, job_library, job_results, submit_job_account, write_learned_skill, memory_read, memory_write]
inputs:
  - the request and the budget (operator_state)
  - at the end, the program to release, its counts on fresh development starts (jobs-counts), and its check
outputs:
  - an approved acceptance condition
  - a saved job with its check, or a decision to stop without a release, with its reasons
  - the account to the shift lead
version: "2"
---
# Proposing, releasing and reporting a job

## Before anything runs: the acceptance condition

The release of a job is one run of the program you save on N reserved
starts, which neither you nor any program of yours has run on, and the job
passes if it is delivered on at least K of them. You propose N and K; the
shift lead approves them or says what is missing. No program runs before the
condition is approved, so the bar cannot be fitted to results.

1. Read the request and the budget (`operator_state`). Ask what the shift
   lead needs: a job the next shift can rely on, and the least it asks for
   if the request says so. Decide what "rely on" should mean in deliveries
   out of starts, from evidence rather than a round number: a job of several
   skills delivers less often than its weakest skill, and the condition you
   propose is the one you will be held to. A job released against a
   condition it meets serves the next shift better than one that fails a
   stricter condition it set itself.
2. Submit it with `submit_acceptance_proposal`: N, K and a short reason in
   plain words: what the condition asks of the job and why that is what the
   next shift needs.
3. Read the reply in `operator_state`. If the shift lead says something is
   missing, revise and propose again. Write the approved condition to memory.

## At the end: release or stop

The release is the investigation's final evaluation: one run of the frozen
program on the reserved starts, judged once against the approved condition.
Save when your evidence says the program will meet the condition there, not
as soon as one development batch reaches it (`jobs-practice`, When to stop).
A release that misses is the investigation's result and is reported as it
is; its records stay with the installation for whoever continues the work.

When `jobs-practice` has a program whose count on fresh development starts
(`jobs-counts`), set beside the approved condition, makes you willing to
release it:

1. Write its check (`jobs-check`) and test it with `run_check` on the job id
   you will save: a saved job keeps only a check that was run on its own runs.
2. Save the job with `save_job`: a name the next shift will recognise, the
   job id of a run of that program (the program is saved as it ran there,
   with the notes you gave it, so let its notes say what each step does and
   why), a purpose the next shift understands, and the name of its check.
   A program whose recorded runs broke an order rule cannot be saved.
3. The job saved last in a session is released after that session: the
   installation runs its program, frozen by its digest, once on the N
   reserved release starts, against the approved condition. The release
   receipt appears in `operator_state` in the next session: the starts, the
   delivered count of N and the order-rule verdicts. There is one release per
   investigation, and no programs and no saves after it, so save only the
   program you mean to release, and save it last.

When no program is ready (the budget is spent, or the failures that remain
need something the program's vocabulary cannot express), save none. Stopping
without a release is a valid outcome and is recorded as such; the account
says why.

If you found a method that a later job could use (a way of reading the
traces, a test that decided a question), `write_learned_skill` keeps it as a
procedure, with the request's objects, goals and steps as parameters, never
named. Write it before you submit the account.

## The account to the shift lead

The shift lead did not watch the work. Write for them, in plain words, every
term explained the first time it is used, every number with what it counts
and where it came from (a tool result of this investigation):

1. **What was asked**: the job in their words, the goals and the rule.
2. **The condition**: N and K as approved, and what they mean.
3. **What was tried**: the programs in order, each with what it changed, the
   starts it ran on and its delivered count ("<d> of <n> on development
   starts <which>"), and what the records showed about its failures, at
   which moment. Literal counts; intervals only from twenty starts. In a
   numbered investigation, say what the earlier investigations left (their
   final evaluations' counts, with their investigation) and what this one
   did with it.
4. **What was released, or why nothing was**: the saved program step by step
   in plain words, and why it is built that way; or the reason for stopping,
   with its evidence.
5. **The result**: the release's count against the condition, from the
   receipt; or, when you stopped, that nothing was released.
6. **The check**: its name, what it checks, how to run it, and what a healthy
   run returns.
7. **What is not known**: what the starts do not show (how many there were,
   what they did not cover), and what to watch on the next shift.

Submit it with `submit_job_account`: after the release, naming the released
job; when you stop without a release, with `job` null. The account ends the
investigation, and it is refused while a release is pending. It ends this
investigation, not the installation's work on the job: write it so that the
next one can start from it.
