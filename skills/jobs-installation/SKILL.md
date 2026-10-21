---
name: jobs-installation
description: The installation as an engineer knows it from its interfaces. The arm and its gripper, the physics they run on, the control loop and the form of a command, the pose controller and what it implies for tracking, the gripper's travel and how its command persists, the two cameras and what the policy perceives, the bounds of the program vocabulary and of a home step, the numbers of the delivery check, scales to reason with at 20 control steps a second, and which tool publishes which number at run time.
when_to_use: Once, early, before composing the first program, so that steps, waits and thresholds are reasoned in the installation's own units; and again whenever a number about the robot, a step's bounds or the delivery check is needed, to know where to confirm it.
tools_used: [operator_state, submit_job_program, job_results, query, get_frames, memory_write]
inputs:
  - nothing beyond this text; the run-time numbers come from operator_state, submit_job_program's description and the job tables
outputs:
  - the installation's fixed numbers, in its units (control steps, metres, seconds), written to memory where a later step relies on them
version: "2"
---
# The installation: the arm, its control loop, its gripper and the numbers the checks use

This is what an installation engineer knows about the robot before watching it
work: its build, its interfaces and the numbers fixed in them. It says nothing
about how the policy behaves in this scene, how long its tasks take or where
they go wrong; that comes from the records you make. Where a number below is
also published by a tool at run time, the tool is the authority: confirm the
number there before a program or a check depends on it.

## The robot

- **The arm**: a seven-joint arm of the Franka Panda type, on a fixed mount
  beside the work surface. Seven joints give one more degree of freedom than a
  hand pose needs, so the same hand pose can be held with different arm
  postures; the controller settles that redundancy itself.
- **The gripper**: the Panda's parallel two-finger gripper. Both fingers move
  together, symmetrically, towards or away from the centre line between them.
- **The physics**: the scene is simulated in MuJoCo (through robosuite and the
  LIBERO benchmark): rigid bodies under gravity, contacts with friction, an
  integration step of 2 ms, so 25 physics steps pass in every control step.
  Movable objects are free bodies with three linear and three angular
  velocities; fixtures (furniture, appliances) are fixed or move only along
  their own joints, such as a door's hinge.

## The control loop and the form of a command

- **20 control steps a second** (`CONTROL_HZ`): one control step is 50 ms. All
  step counts in programs, waits, budgets and tables count these steps.
- **A command is seven numbers**, absolute, in the frame in which the robot
  reports its hand (z points up):
  1. three: the target position of the hand, in metres;
  2. three: the target orientation of the hand as a rotation vector (the axis
     of rotation scaled by the angle, in radians);
  3. one: the gripper command, -1 to open and +1 to close.

  Every command the installation issues has this form, the policy's included;
  `robot_steps.commanded_action` and `robot_steps.gripper_command` record each
  one as it was issued.
- **The controller** is an operational-space pose controller (robosuite's
  `OSC_POSE`, fixed impedance: stiffness 150, damping ratio 1). At every physics
  step it computes joint torques that pull the hand towards the commanded pose,
  like a spring with a damper attached to the target, with the arm's own
  inertia accounted for. In general terms that means:
  - the hand follows a moving target with a lag, and a target that jumps is
    approached over several control steps, not reached in the step it was
    sent;
  - a damping ratio of 1 is chosen to approach without ringing in free space,
    but joint limits, the arm's posture and contacts change the response; how
    this arm actually tracks is something to read from the records (commanded
    against measured), not to assume;
  - in contact, the target is not where the hand goes but how hard it pushes:
    a target behind a surface or inside an object turns into a force that
    grows with the distance, so the hand presses, pushes or drags rather than
    stops;
  - position and orientation are controlled together; a large turn of the
    hand is tracked with lag like a large move.
- **Measured and commanded**: `robot_steps.eef_pos` is the hand's measured
  position after the step, at the same point the command targets; the
  difference from the commanded position is the position tracking error at
  that step. `robot_steps.eef_quat` is the orientation of the hand's body,
  while a command's rotation vector is the orientation of the controller's
  frame at the gripper; the two differ by a fixed 90-degree turn, so
  `eef_quat` set beside a commanded rotation shows an error of about 90
  degrees even when the hand tracks perfectly. Compare positions; for
  orientation, compare `eef_quat` across steps, not with the command.

## The gripper

- **Travel**: its widest opening is about 0.08 m between the fingers.
  `robot_steps.gripper_opening` records the measured distance between the
  fingers at every step.
- **Speed**: the fingers move at a finite speed, so a change of command takes
  effect over several control steps. Closing on an object stops where the
  fingers meet it: the opening while holding something is roughly its width at
  the grasp, and closing on nothing ends near zero. A grasp holds by friction
  between the fingers and the object.
- **Persistence**: the gripper keeps its last command until something changes
  it. A motion step's `hold`, `step`, `ramp` and `sine` and a `wait` keep it;
  `open_gripper` and `close_gripper` set it; a home step commands it open; a
  policy step sets it with every action it issues. Before anything has been
  commanded, a wait holds the measured pose with the gripper open
  (`jobs-transitions` describes the steps in full).

## The cameras, and what the policy perceives

- **Two cameras**, as the records name them: `scene`, fixed, overlooking the
  work area, and `wrist`, on the hand, looking along the gripper. The policy
  receives both at 360 by 360 pixels. The recorded videos keep every fourth
  control step (5 frames a second) and the last; `robot_steps.frame_index`
  names the frames, and `get_frames` returns them.
- **The policy** (X-VLA, a vision-language-action model) is fixed: it is never
  retrained or tuned here. At each call it is given the two camera images, the
  written instruction and the hand's measured position and orientation, and
  nothing else: not the gripper's opening, not the positions of objects, not
  the requested goals, not contacts, not the program or what the step before
  did. It predicts 30 actions at a call. The program's `actions_per_chunk`
  setting chooses how many of those actions to execute in order before
  observing and calling the policy again: an integer from 5 to 30. Omitted
  or `null` means 30 actions (1.5 s at 20 Hz), the default. This changes the
  executed prefix, not the number predicted or the policy's weights. At every
  step boundary the actions not yet executed are dropped, so a new policy
  step starts from a fresh call.
- **Closed loop, chunk by chunk**: the policy sees the scene anew at every
  call, so it responds to what has happened; within the executed prefix of
  one call it does not look again. Shorter horizons observe and replan more
  often, requiring more policy inference for the same number of control
  steps. The program-wide setting also applies during the delivery tail;
  entering the tail continues the current prefix instead of discarding it.
  The engineering agent chooses this setting between runs; it does not
  intervene while the robot moves.
- **It does not know when it is done**: it learned single tasks, each under its
  own instruction, and goes on acting under its instruction after its task's
  goal first holds. Only a step's `until` condition, its `max_steps` or the
  delivery tail ends a policy step. `operator_state` lists the instructions its
  tasks were learned under; what it does under other wording is not given by
  anything here.

## How goals and the delivery check are decided

- **Goals** are decided by the LIBERO benchmark's own goal predicates,
  evaluated on the physics state at every control step; `jobs-transitions`
  gives their definitions and what they do not establish.
- **Delivered** is the installation's check at the end of a run: every
  requested goal holds; no body of the robot (arm links, fingers) touches an
  object the goals name; those objects are at rest; the gripper is open. Its
  numbers:
  - at rest: an object's linear speed below 0.01 m/s and its angular speed
    below 0.15 rad/s, from the velocities of its free body as the physics
    reports them; a fixture without a free body is checked for contact only;
  - touching: any contact between a body of the robot and the object in the
    physics state's contact list at that step;
  - open: the last gripper command was open and the fingers are at least
    0.06 m apart;
  - the delivery tail: after a program whose last step is a policy step with
    outcome `reached` (its `until` goal held, or its `until.steps` ran out;
    not a `timeout`), the policy goes on under that instruction until the
    check holds or 60 control steps (3 s) have passed (`jobs-traces`).

## The bounds of a program

Fixed by the installation; `submit_job_program`'s description repeats them.

| What | Bound | Meaning |
|---|---|---|
| steps of a program | at most 8 | policy, motion, wait and home steps together |
| a program's `actions_per_chunk` | 5 to 30, or `null` for 30 | actions executed from each policy prediction before a fresh observation and call; also applies to the delivery tail |
| a policy step's `max_steps` | at most 300 (15 s) | the most one instruction runs, the step budget of a single task |
| a wait's `steps` | at most 100 (5 s) | the last command held, physics going on |
| a motion segment | at most 200 steps (10 s) | one segment per motion step |
| a motion target | at most 0.05 m per axis | the offset on x, y or z from the hand's position when the step starts; orientation held as it was then |
| a ramp's or a sine's speed | at most 0.5 m/s | 2.5 cm a control step for the target |
| a sine's period | at least 4 steps | at most 5 Hz |
| a `step` segment | a jump | the target moves at once; the hand follows as the controller allows |
| a home step's speed | at most 0.5 m/s | the target moves in a straight line to the job's start pose, orientation turning along, gripper open |
| a home step's arrival | within 0.01 m, slower than 0.01 m/s | of the start position, the speed measured between consecutive control steps (`reached`) |
| a home step's `duration_steps` | at most 200 (10 s) | ramp and settling together, then `timeout` |

## Scales to reason with

- One control step is 50 ms; 20 steps are a second; 60 steps are 3 s, 300
  steps 15 s.
- The rest threshold of 0.01 m/s is less than 0.5 mm of movement between two
  control steps; 0.15 rad/s is 7.5 mrad (about 0.4 degrees) a step. The
  check reads the physics' velocities at one step, not the images: recorded
  frames are four control steps apart, and the wrist camera moves with the
  hand, so movement seen in frames is a cue for where to query
  `object_steps`, not a verdict on rest.
- A home step's target moves at most 2.5 cm a step (the ramp's length is
  rounded up to whole steps and divided equally): 20 cm of return is a ramp of
  8 steps, and the hand then needs its settling time to come within 1 cm and
  below 1 cm/s.
- An object let go above a surface falls under gravity, t = sqrt(2h / g):
  from 1 cm in about 45 ms (one step), from 5 cm in about 0.10 s (two steps),
  from 10 cm in about 0.14 s (three steps). It lands within a few steps; what
  it does then (stays, bounces, slides, tips) the records show.
- The checks look at single instants. A speed or a contact read at one step
  can catch a passing moment; read it over several consecutive steps before
  calling a state settled.

## Where to confirm a number at run time

- `operator_state`: `delivery_thresholds` (`linear_speed_below_m_s`,
  `angular_speed_below_rad_s`, `gripper_open_at_least_m`, `tail_steps`), the
  policy's tasks with their instructions and goals (`policy_skills`), the
  development starts, the release starts reserved and the budget.
- `submit_job_program`'s description: the bounds of the step kinds, the motion
  vocabulary, the home step and the program, and the starts a request may name.
- The job tables, read with `query` (`jobs-traces` describes them):
  `robot_steps` for the hand and gripper (`eef_pos`, `eef_quat`,
  `gripper_opening`, `commanded_action`, `gripper_command`, `phase`,
  `frame_index`); `object_steps` for the objects (`linear_speed`,
  `angular_speed`, `in_contact_with_robot`, `in_contact_with`); `goal_steps`
  and `job_step_events` for goals and step outcomes; `job_results` to list the
  runs. `get_frames` for the images.

If a tool and this text disagree, the tool is right; write the difference to
memory with `memory_write`. Next: `jobs-release`, for the acceptance proposal.
