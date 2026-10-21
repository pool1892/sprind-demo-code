---
name: assess-risk
description: For the live supervisor. Judge how a running attempt is going from the recent camera frames, the telemetry window and the risk estimator's number, name the phase and the signs of trouble with their evidence, and say whether the attempt can still be saved.
when_to_use: Every time the risk estimator fires and the supervisor is called, before choose-intervention.
tools_used: []
inputs:
  - the recent frames of the scene and wrist cameras, oldest first
  - the telemetry window (joints, gripper, end effector, commanded actions)
  - the risk estimator's probability of failure within its horizon
  - the task text
  - optionally, the telemetry signs the harness computed (assess_risk_helpers.py)
outputs:
  - the phase of the attempt
  - the signs of trouble seen, each with the frame or step that shows it
  - a judged chance of failure if the robot carries on, and whether the attempt is recoverable
version: "2"
---
# Assess risk

The robot holds still while you decide, and that time counts. Be quick and
concrete: look, name what you see, judge. The risk estimator reads only the
telemetry; elapsed time can dominate its score. A scheduled inspection does not prove failure. Your frames can confirm trouble or show a false alarm.

## Procedure

1. **Phase.** From the gripper and the frames: approach (gripper open,
   moving towards the object), grasp (closing on it), lift and carry
   (closed with the object, rising or moving), place (over the target,
   lowering), release and retreat. The task text names the object and the
   target; check that the robot is working on that object.
2. **Signs of trouble.** Look for each, and note the frame or step that
   shows it:
   - *closed on nothing*: the fingers are shut and the wrist view shows no
     object between them;
   - *slipping*: the object tilts or slides in the fingers while carried,
     the finger opening shrinks;
   - *dropped*: the object lies away from the target while the gripper is
     empty;
   - *knocked*: the object was pushed or toppled before the grasp;
   - *stalled*: the arm has barely moved for many steps and the task is not
     done;
   - *oscillating*: the arm moves back and forth around a point, commands
     reversing again and again;
   - *pressing*: the arm or the gripper is in contact with the table or a
     fixture and pushing;
   - *wrong object or lost view*: approaching another object, or the object
     out of the scene camera's view.
   If the harness gave telemetry signs (`closed_on_nothing`,
   `lost_object_at`, `longest_still_run`, `reversals` and the others of
   `assess_risk_helpers.signs`), use them as a second witness; trust the
   frames where the two disagree, and say so.
3. **Recoverable?** The attempt can be saved when the object is still
   within reach and graspable where it lies, the gripper is free, and
   nothing is jammed. It cannot when the object has left the work area,
   lies where the arm cannot reach, or the scene is disturbed beyond the
   task (another object knocked into the way, the object broken apart from
   its container).
4. **Judge the chance of failure if the robot carries on.** Start from the
   estimator's number. Lower it when the frames show normal progress (a slow
   but correct approach, a grasp that holds). Raise it to near certain when
   you see a miss, a drop or a stall with no progress. Policies rarely
   recover by themselves from a drop or from closing on nothing; they
   sometimes recover from hesitation and from a first missed approach.
5. Hand on to choose-intervention: the phase, the signs with evidence, the
   judged chance, recoverable or not, and whether a retry has already been
   used in this attempt.

## Typical trouble by phase (general knowledge)

- Approach: wrong object, oscillation near the object (control timing), an
  object knocked by the arm, a target the camera places wrongly (the gripper
  heads beside the object, consistently).
- Grasp: closing beside the object, closing too early or too late, a weak
  grip on a heavy or slippery object.
- Carry: slipping and dropping, worse with worn pads or heavy objects;
  collisions with fixtures on the way.
- Place: overshoot near the target, release too early (the object falls
  beside the target), not releasing at all (the attempt times out holding).
