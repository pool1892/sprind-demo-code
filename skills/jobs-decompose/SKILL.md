---
name: jobs-decompose
description: Turn a requested job into a program the installation can run. Read the request, its requested goals and its order rules; match each goal to a skill the policy has; find what the goals share and how one step changes the scene for the next; list the orders the rules allow; and write a program of policy, motion and wait steps in the program format, starting from the simplest program that could do the job.
when_to_use: At the start of a request, after the acceptance proposal, before the first job run; again whenever practice shows that the program's structure (its steps or their order), not only a transition, is in question.
tools_used: [operator_state, submit_job_program, job_library, memory_read, memory_write]
inputs:
  - the request, its requested goals and order rules, the skills the policy has, the development starts and the budget (operator_state)
outputs:
  - in memory, the goals matched to skills, what they share, the permitted orders, and the reasoning for the first program
  - the first program, submitted on development starts
version: "1"
---
# From a request to a program

The shift lead asks for a job in their own words and names what must be true
at the end: the requested goals. The policy cannot do a job; it can do single
tasks, each under its own instruction, each ending in its own goal. A program
is the bridge: steps in one continuous scene, in an order, with transitions
between them. The requested goals, the order rules and the delivery check are
the installation's and you cannot change them; the program is yours.

## Read the request

Call `operator_state`. Note in memory:

- the request in the shift lead's words, and the requested goals with their
  index (from 0), predicate and objects. The goals, not the words, are what
  the delivery check tests at the end: every requested goal holds, the robot
  touches none of the objects the goals name, those objects are at rest, and
  the gripper is open, with the thresholds the check publishes;
- the order rules: each names an `earlier_goal` and a `later_goal`; the
  later goal may not hold before the earlier one has held. They are checked
  on the recorded trace of every run, whatever the instructions say, and a
  program whose recorded runs break a rule cannot be saved;
- the skills the policy has: each with its goal-suite task, its instruction
  and the goal it achieves when run on its own;
- the base-task identifier, the development starts you may use, the delivery check's thresholds, and the budget.

If `job_library` lists jobs or checks from earlier requests, read their
purpose and notes: they were made for other requests and may or may not
apply here.

## Match goals to skills

For each requested goal, find the skill whose own goal is that goal: the same
predicate on the same objects. Write the match down. A goal no skill
achieves on its own cannot be reached by chaining these skills alone; say so
to the shift lead in the account instead of guessing.

## What the goals share, and what each step leaves

Steps in one scene interact. For each pair of goals, note:

- **Shared objects.** An object named by two goals is moved or used by both
  steps. Whatever one step does to it is the state the other starts from.
- **Containment and support.** A goal that puts one object in or on another
  ties them: moving the lower one later moves or disturbs the upper one, and
  placing onto something that has moved means placing somewhere else than
  where it stood.
- **Fixtures.** A fixture's state (open, closed, on, off) can change what the
  next step can reach and what it needs to avoid.
- **Independence.** Goals with no shared object and no fixture between them
  may still interact through the hand: the hand ends one step where it ends
  it, with whatever it holds.

These notes say where to look first in the traces. They are not findings.

## The orders the rules allow

List the orders of the steps the order rules permit. For each, note what
scene the later steps start from in your notes above. Where several orders
are permitted, the first program takes one and says why; another order is a
change like any other, tested on the same starts (`jobs-practice`).

## The program format

A program has a `name` (lower case letters, digits and hyphens, three to
forty characters), a `purpose` (one line), `steps` (at most eight) and
`notes` (what it changes against its parent, if it has one). Its digest is
the SHA-256 of its canonical JSON, and the digest includes the name, purpose,
steps and notes: a program renamed or given other notes has another digest.
The tables, the release, the receipt and the library refer to a program by
its digest. The tool's argument schema is authoritative. Every step
carries every field, null where its kind does not use it. There are four
kinds:

- a **policy step**: `kind` `policy`; an `instruction` (free text; the
  policy was trained on its skills' instructions as they are written); an
  `until` with exactly one of `goal` (a requested goal's index: the step ends
  the first control step that goal holds) and `steps` (the step ends after
  that many control steps), the other null; `max_steps` (at most 300, twenty
  control steps a second); and `on_timeout`, `stop` (the default: the
  program ends there and the later steps are skipped) or `continue` (the next
  step runs), which decides what happens when `max_steps` run out before the
  `until` condition holds;
- a **motion step**: `kind` `motion` with one `segment` of the probe
  vocabulary (`primitive` hold, step, ramp, sine, open_gripper or
  close_gripper, with `axis`, `amplitude_m`, `duration_steps` and
  `period_steps` as `request_probe` states them and within its bounds),
  played from where the hand is when the step starts;
- a **wait**: `kind` `wait` with `steps` (at most 100), holding the last
  command;
- a **home step**: `kind` `home` with `duration_steps` (at most 200) and
  `on_timeout`: with the gripper commanded open, the hand's target ramps
  from where the hand is to the pose (position and orientation) the robot
  reported when the job started, no faster than the probes' speed bound,
  then holds until the hand is within 1 cm of that position and at rest
  (`reached`) or `duration_steps` run out (`timeout`, then `on_timeout` as
  for a policy step).

A request (`submit_job_program`) is a program, the development starts it
runs on, named by their stored initial states (`operator_state` gives the
range), and whether to record video. Release starts are refused.
`jobs-transitions` describes what each kind of step does at a boundary, and
`jobs-traces` the delivery tail after the program.

## The first program

Start from the simplest program that could do the job: one policy step per
requested goal, in a permitted order, each with its matched skill's
instruction as written and `until` its goal, with a `max_steps` large enough
for the skill to finish on its own. It is a baseline to watch, not a guess at
the final program: its failures are the first evidence of what the job needs.
Run it on about ten development starts with video (`submit_job_program`;
it runs after the session ends) and write its name, the starts and the
reasoning above to memory. Then follow `jobs-practice`.
