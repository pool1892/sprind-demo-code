---
name: choose-intervention
description: For the live supervisor. Choose continue, retry, recover, degrade, stop or hand over for a running attempt from the risk assessment, the options available, the operator queue and the cost of operator time, by comparing the expected cost of each option.
when_to_use: Right after assess-risk, every time the supervisor is called. The decision must come back quickly; the robot holds still meanwhile.
tools_used: []
inputs:
  - the assessment from assess-risk (phase, signs, judged chance of failure, recoverable, retry used)
  - the options available for this attempt
  - the operator queue (hand-overs waiting, operators free)
  - the cost of an operator minute relative to a failed attempt (the run's cost ratio)
outputs:
  - one decision of the options available, with a one-sentence rationale naming the sign it rests on
version: "2"
---
# Choose an intervention

Six options, of which the context lists the ones available:

- **continue**: the robot carries on as it is.
- **retry**: a scripted recovery: open the gripper, lift, return to a
  pre-grasp pose, hand control back to the policy. No operator; costs a few
  seconds.
- **recover**: open the gripper for half a second, then retreat vertically by
  8 cm. Use only when the correct object is visibly over its destination
  and releasing there is appropriate. It cannot pick up an object elsewhere.
- **degrade**: discard queued policy commands and execute at most four commands
  before replanning from new observations. This trades inference throughput
  for more frequent visual feedback. Consider it for oscillation or inaccurate
  open-loop motion; it does not fix an unreachable object or bad calibration.
- **stop**: end the attempt safely. It counts as a failed attempt, but it
  ends a doomed attempt early instead of letting it run to the step limit,
  which frees the robot for the next one.
- **hand_over**: a teleoperator takes over from where the robot stands. It
  may save a recoverable attempt and costs operator time, plus waiting when no
  operator is free.

## The decision rule

Compare the expected cost of each available option, in failed attempts:

- continue: the chance of failure if the robot carries on, plus, if it
  fails, the time a doomed attempt still wastes;
- retry: the chance of failure after a retry, plus, if it fails, the time
  it still wastes, plus its own small time cost;
- hand_over: the chance the operator fails too (small when the scene is
  recoverable), plus the operator minutes times the cost of an operator
  minute, plus the waiting when the queue is long;
- stop: one failed attempt, with no time wasted.

For recover and degrade, compare the judged residual failure chance plus
robot time, as for retry. These are uncertain forecasts, not measured rescue
rates. Do not assume the operator always succeeds. The queue is a supplied
observation: zero free operators without a waiting-time estimate does not
by itself establish that waiting is unaffordable.

Choose the cheapest. In words, that comes to:

1. **Continue** when the frames show normal progress, even slow progress,
   and whenever the estimator fired but the frames contradict it. Most calls
   should end here: interrupting a healthy attempt is the commonest mistake.
2. **Retry** when the attempt is failing in a way the recovery primitive
   undoes (a missed or weak grasp, a slip early in the lift, an object
   nudged but upright and in reach), the scene is recoverable, and no retry
   has been used on the same symptom in this attempt. A second retry for
   the same symptom rarely works; go to 3 or 4 instead.
3. **Recover** only for a visible release/retreat problem over the correct
   target. **Degrade** for a plausible benefit from more frequent visual
   feedback when retry would simply repeat an inaccurate approach. Each
   autonomous primitive is offered at most once; respect the actual options.
4. **Hand over** when the attempt will fail without help, the situation is
   beyond the primitive (an object in an awkward pose, dropped near the
   target, a jam, a repeat after a retry), the scene is recoverable by a
   person, and an operator is free or the queue is short, and the operator
   time is worth it at this cost ratio: the more an operator minute costs,
   the surer the failure and the higher the chance of rescue must be.
5. **Stop** when the attempt cannot be saved (the object left the work
   area, the scene is disturbed), when handing over is not worth its cost
   (no operator free and a long queue, or a high cost ratio with a doubtful
   rescue), or when the arm is pressing against the cell and carrying on
   risks damage.

Write the rationale in one sentence that names the sign the decision rests
on and the step or frame that shows it.

`choose_intervention_helpers.py` beside this file computes the same
comparison from numbers (the chances, minutes and cost ratio), for a
harness that wants to check or log the rule; the decision stays yours, made
from what the frames show.
