# What counts as success: LIBERO's check, or the object placed and let go

Spike of 2026-10-07, run on this machine. Sections 1 to 4
come from light jobs only (no model on the graphics chip); section 5 adds one
heavy run with X-VLA after a review of the first version.
The scripted expert ran in this checkout's environment (`make setup`: MuJoCo
3.3.7, robosuite 1.5.2 with the port, the patched LIBERO in `third_party/`,
relative control). The recorded policy episodes were replayed in the
environment they were recorded in (`~/.cache/robots-probes/baseline-venv-mj337`:
the same versions, LeRobot 0.6.1). Other agents' jobs ran throughout (load
average 10 to 18).

Words used throughout:

- **LIBERO's check** is `env.check_success()`, the test on which LeRobot's
  evaluation ends an episode and which every published LIBERO success rate
  counts.
- **The check step** is the first control step after which LIBERO's check
  holds. A control step is 50 ms.
- **Identical** means bit for bit, compared through a hash of MuJoCo's full
  state, never with a tolerance.
- **Touching** means an entry in MuJoCo's contact list between a collision
  geometry of the robot (arm, hand, fingers) and one of the object. It is the
  test LIBERO uses for its own contact condition.
- **At rest** means the object moves slower than 1 cm/s and turns slower than
  0.15 rad/s. These are the thresholds the scripted expert already used to call
  an object settled, chosen before the runs and not tuned.
- **Placed** is the definition recommended here, (d) in section 3.

## Verdict

**Use "placed" as the fleet's success, and log LIBERO's check beside it.**
Placed holds at a control step when LIBERO's check holds, nothing of the robot
touches the object, and the object is at rest. It is a function of one
simulator state, so an original and its replay get the same label bit for bit.
The episode runs on after LIBERO's check until placed holds, for at most 60
more steps and never past the step limit. An episode where LIBERO's check held
but placed never did is a failure, with a label of its own.

**Measured with X-VLA acting on its own after the check** (section 5; 600
episodes in the fleet's 16-bit setting, identical to the earlier runs up to the
check in 532 of 532 successes): X-VLA lets go by itself, commanding the gripper
open within 60 steps in all 394 healthy successes and in 129 of 138 faulted
ones (the other 9 ran into the step limit first). Healthy, placed costs
it nothing on the bowl (195 of 200 by either definition) and 3 episodes on the
basket (199 to 196 of 200, items left hanging on an open finger). In the
faulted mix it costs 12 episodes (138 to 126 of 200), 10 of them because the
step limit left less than 60 steps after a late check, nearly all under the
17-step action delay. Placed holds a median of 13 (bowl) and 8 (basket) steps
after LIBERO's check, at most 32 and 39 healthy and 57 under the delay; that
is 0.3 to 0.4 extra policy calls per success healthy and 0.5 in the faulted
mix.

- **LIBERO's check asks for much less than a placed object** (section 1). For
  the bowl it holds at the first touch of the plate, while the bowl is still
  gripped. For the basket it asks only that the item's origin lies inside a box
  over the basket, with no contact, so it holds while the item is held above
  the basket or falling into it. Neither the gripper nor any velocity enters.
- **At the check step, not one of 850 recorded policy successes is placed**
  (section 2). Replayed without the policy, all 850 identical at every step:
  X-VLA still touches the bowl in 372 of 392 successes, the gripper commanded
  closed in 368.
  In the basket suite it has already let go in mid-air in 115 of 199, and the
  item is falling at a median of 89 cm/s when the episode is scored.
- **12 of the 850 LIBERO successes do not end placed** when a script opens the
  gripper at the check step and lifts the arm (section 2): the bowl slides off
  the plate's rim or settles just outside LIBERO's 3 cm circle, the milk carton
  or the orange juice comes up with the open gripper, or a finger stays on the
  bowl. That is 2 of 434 for healthy X-VLA (both basket items, both partly an
  artefact of the scripted lift; healthy bowls 0 of 235), 6 of 157 (3.8%) for
  X-VLA under faults, and 4 of 259 for SmolVLA. Faults make marginal placements
  more common, so LIBERO's check understates what a fault costs.
- **On 1,000 expert episodes with three continuations after the check**
  (section 3), placed fired in none of the 879 continuations that carried the
  object away, apart from 2 in which the bowl had been set down and let go
  before LIBERO's check fired. LIBERO's check had scored all 879 as successes.
  "Nothing touches the object" alone fired on 18 of them, at a step where the
  bowl was free of the fingers for a moment but still moving; "gripper open"
  missed all 119 objects that came to rest in place while the gripper stayed
  closed.
- **It costs few steps** (sections 4 and 5). X-VLA acting on its own: placed
  holds a median of 13 steps after LIBERO's check on the bowl and 8 on the
  basket (90%: 21 and 20; at most 32 and 39), 0.28 and 0.44 extra policy calls
  per success (+7% and +9%). For the scripted expert: 3 and 15 (at most 12 and
  44).
- **The scripted release was close to the best case, and X-VLA came close to
  it** (section 5). The prediction from the scripted release: stage-1 rates of
  96.0% (spatial) and 98.5% (object) become about 96.0% and 97.5%. Measured with
  X-VLA itself on the speed spike's manifests: spatial 97.5% stays 97.5%, object
  99.5% becomes 98.0%. Under faults the loss is larger than the scripted release
  suggested: on the same 138 faulted successes the scripted release lost 1 and
  X-VLA on its own 12, because a late check leaves no room for the release
  before the step limit.

| | Bowl onto plate (`libero_spatial`) | Item into basket (`libero_object`) |
|---|---|---|
| LIBERO's check asks | bowl touches plate; bowl origin within 3 cm of plate origin, not below it | item origin inside a 12 x 12 x 15 cm box over the basket |
| X-VLA at its check step: object touched / gripper commanded closed | 372 / 368 of 392 | 84 / 68 of 199 |
| X-VLA: placed at its check step | 0 of 392 | 0 of 199 |
| Not placed after a scripted release | X-VLA 0 of 235 healthy, 6 of 157 faulted; SmolVLA 4 of 259 | X-VLA 2 of 199 |
| Expert: steps from LIBERO's check to placed, median / 90% / max | 3 / 5 / 12 | 15 / 19 / 44 |
| X-VLA, scripted release: steps to placed, median / 90% / max | 6 / 15 / 18 (healthy) | 8 / 17 / 25 |
| **X-VLA acting on its own, healthy: LIBERO's check / placed, of 200** | **195 / 195** | **199 / 196** |
| X-VLA acting on its own, faulted mix (spatial): LIBERO's check / placed, of 200 | 138 / 126 (none 40 / 40, camera 6 degrees 42 / 39, delay 17 steps 56 / 47) | |
| X-VLA acting on its own: steps to placed, median / 90% / max | 13 / 21 / 32 (healthy), 16 / 34 / 57 (mix) | 8 / 20 / 39 |
| X-VLA acting on its own: extra policy calls per LIBERO success | 0.28 (healthy), 0.54 (mix; delay 0.84) | 0.44 |

## 1. What LIBERO's check asks for

Read from `third_party/libero/libero/libero/` (`envs/predicates/base_predicates.py`,
`envs/object_states/base_object_states.py`, `envs/objects/site_object.py`) and
the task files under `bddl_files/`. Every task of both suites has one goal with
one predicate:

| suite | tasks | goal in the task file | what has to hold at the step |
|---|---|---|---|
| `libero_spatial` | all 10 (bowl from the table, the drawer, the stove, the cookie box, ...) | `(On akita_black_bowl_1 plate_1)` | the plate's body origin is not above the bowl's; **some** collision geometry of the bowl is in MuJoCo's contact list with some of the plate's; the two body origins are less than 3 cm apart horizontally |
| `libero_object` | all 10 (alphabet soup, cream cheese, salad dressing, bbq sauce, ketchup, tomato sauce, butter, milk, chocolate pudding, orange juice) | `(In <item>_1 basket_1_contain_region)` | the item's body origin lies strictly inside the basket's site `contain_region`: half sizes 6.1 x 6.1 x 6.9 cm, centred 7.2 cm above the basket's origin, turned with the basket and then taken axis-aligned, with the lower face lowered by 1 cm. LIBERO's contact test for a site always returns true, so the item **need not touch the basket** |

In neither case does the gripper enter, nor any velocity, nor how long the
condition has held. The check is evaluated after every control step, and
LeRobot's `LiberoEnv.step` returns `terminated` the first time it holds, which
ends the episode in `lerobot-eval` and in our fleet runners.

My reading, which the measurements below confirm: the bowl check fires at the
first touch between bowl and plate, which for a grasp from above happens while
the bowl is still between the fingers. The basket check fires as soon as the
item's origin is over the opening and below about 14 cm above the basket's
origin, whether the item is held, falling or resting.

The replica of LIBERO's check in `defs.py` (`libero_check`) agreed with
`env.check_success()` at every recorded state of every run of this spike:
545,792 states, 0 disagreements (expert runs 373,465, of which 360 belong to
the two episodes where the check never fired; policy replays 172,327).

## 2. The state at LIBERO's check, from recorded episodes

### What the stored data hold

| source | per step | object pose | enough for this question |
|---|---|---|---|
| `stack-reference/*` (LeRobot's `lerobot-eval`) | `eval_info.json` and videos | no | no |
| `skeleton/runs/*`, `xvla-speed/runs/*`: `log/steps/*.parquet` | joints, finger positions, end-effector pose, commanded action | no | gripper opening only |
| same runs: `truth/chain/*.npz` | every action as it reached the simulator (`exe`), a hash of MuJoCo's full state after every step | no | **yes, by replay** |

The telemetry has the gripper opening but nothing about the object. The truth
chain, however, holds every executed action, and the episode record holds the
start (task, initial state, scene seed). Stepping the same environment with the
same actions, without the policy, reproduces the episode, and the stored hashes
say whether it did. `replay_policy.py` reproduced **850 of 850 successful
episodes identically at every step**, X-VLA in absolute and SmolVLA in relative
control, healthy and faulted. Every replayed state was then read with
`defs.features`.

Runs replayed (all their successful episodes):

| run (under `~/.cache/robots-probes/`) | policy | suite | faults | successes |
|---|---|---|---|---|
| `skeleton/runs/healthy200` | X-VLA, 32-bit | spatial | none | 195 |
| `xvla-speed/runs/p_obj_res32` | X-VLA, 16-bit | object | none | 199 |
| `xvla-speed/runs/m200v_fp16res32_r4w10` | X-VLA, 16-bit | spatial | 40 none, 80 camera turned 6 degrees, 80 actions 17 steps late | 138 |
| `xvla-speed/runs/p_cam9_res32` | X-VLA, 16-bit | spatial | camera turned 9 degrees | 59 |
| `xvla-speed/runs/h200v_smolvla_r4w10` | SmolVLA | spatial | none | 166 |
| `xvla-speed/runs/m200v_smolvla_r4w10` | SmolVLA | spatial | mix as above | 93 |

Replay `replay850`, 37 minutes with two processes. The tables of this section
and the next come from `analyze.py --expert expert1000 --replay replay850`; the
full output is in `~/.cache/robots-probes/success/tables_expert1000_replay850.md`.

### At the check step

"Above its resting height" is the object's origin above where it ends up
resting on the target (per task, the median at the end of the expert runs).

| policy | suite | successes | object touched by the robot | gripper still commanded closed | gripper opening, mm, median (10% to 90%) | above its resting height, cm, median / 90% / max | object touches the target | placed |
|---|---|---|---|---|---|---|---|---|
| X-VLA | spatial, healthy | 235 | 229 | 227 | 4 (3 to 7) | 0.2 / 0.6 / 1.3 | 235 | 0 |
| X-VLA | spatial, faulted | 157 | 143 | 141 | 4 (3 to 9) | 0.4 / 0.9 / 1.8 | 157 | 0 |
| X-VLA | object, healthy | 199 | 84 | 68 | 52 (36 to 69) | 5.7 / 9.9 / 11.2 | 5 | 0 |
| SmolVLA | spatial, healthy | 202 | 195 | 190 | 4 (3 to 8) | 0.3 / 0.7 / 1.3 | 202 | 0 |
| SmolVLA | spatial, faulted | 57 | 54 | 53 | 4 (3 to 12) | 0.2 / 0.7 / 2.5 | 57 | 0 |

For the bowl the picture is uniform: the bowl sits on the plate, the fingers
are closed on its rim (4 mm between the pads) and squeeze it with a median of
7 to 14 N depending on the task. For the basket it depends on the item:

| task | item | mass, g | expert: touched at its check, of 50 | expert: above resting height, cm | X-VLA: touched at its check, of 20 | X-VLA: above resting height, cm, median (max) |
|---|---|---|---|---|---|---|
| 0 | alphabet soup | 26 | 50 | 8.0 | 5 | 7.2 (8.4) |
| 1 | cream cheese | 6 | 0 | 9.3 | 0 | 7.2 (11.0) |
| 2 | salad dressing | 22 | 50 | 3.7 | 19 | 4.6 (4.7) |
| 3 | bbq sauce | 12 | 50 | 6.4 | 6 | 5.5 (6.4) |
| 4 | ketchup | 23 | 50 | 4.1 | 7 | 3.8 (4.8) |
| 5 | tomato sauce | 27 | 2 | 7.5 | 5 | 6.8 (8.1) |
| 6 | butter | 5 | 50 | 10.9 | 3 | 10.3 (11.2) |
| 7 | milk | 32 | 50 | 4.5 | 19 of 19 | 4.8 (5.0) |
| 8 | chocolate pudding | 10 | 50 | 10.5 | 0 | 7.1 (10.6) |
| 9 | orange juice | 32 | 50 | 4.1 | 20 | 4.7 (5.0) |

X-VLA still holds the salad dressing, the milk and the orange juice at its
check step (squeezed with about 40 N) and has let go of most of the others in
mid-air. Of the 115 items already let go, the median one falls at 89 cm/s at
the check step (10%: 48 cm/s): LeRobot scores them while they fall. The expert
holds most items 4 to 11 cm above where they will rest, and lets go of the
cream cheese and the tomato sauce in the air before the check fires.

### If the policy had let go at the check step

Two scripted continuations from the replayed check step, 40 steps each:
`open_now` keeps the arm at its last target and opens the gripper; `open_lift`
does the same and after 10 steps raises the target by 1 cm per step to 10 cm.
Neither is what the policy would have done; section 5 measures that.

| policy | suite | successes | continuation | LIBERO's check still holds after 40 steps | placed within 40 steps | steps to placed, median / 90% / max |
|---|---|---|---|---|---|---|
| X-VLA | spatial, healthy | 235 | open_now | 235 | 146 | 4 / 8 / 20 |
| X-VLA | spatial, healthy | 235 | open_lift | 235 | 235 | 6 / 15 / 18 |
| X-VLA | spatial, faulted | 157 | open_now | 152 | 79 | 5 / 10 / 16 |
| X-VLA | spatial, faulted | 157 | open_lift | 151 | 151 | 12 / 17 / 21 |
| X-VLA | object, healthy | 199 | open_now | 199 | 194 | 8 / 17 / 25 |
| X-VLA | object, healthy | 199 | open_lift | 197 | 197 | 8 / 17 / 25 |
| SmolVLA | spatial, healthy | 202 | open_now | 197 | 154 | 4 / 8 / 36 |
| SmolVLA | spatial, healthy | 202 | open_lift | 199 | 199 | 5 / 14 / 19 |
| SmolVLA | spatial, faulted | 57 | open_now | 57 | 44 | 5 / 14 / 38 |
| SmolVLA | spatial, faulted | 57 | open_lift | 57 | 56 | 6 / 14 / 21 |

With the arm left where it was (`open_now`), X-VLA's hand stays inside the
bowl and keeps touching it, so placed does not hold in a third of the bowl
episodes; that is the frozen arm, not the bowl. In 72 of the 850 `open_now`
tails the held-still arm sagged back onto the object after placed had first
held (61 still touching at the end); the engine will not see this, because it
ends the episode at the first step placed holds. With the lift, every episode
whose object stays put becomes placed within 21 steps (bowl) and 25 (basket).
Against the end state of `open_lift` (LIBERO's check holds and the robot does
not touch the object 40 steps after the check), placed and placed with K=5
have no false success and no miss in any of the five groups; LIBERO's check has
12 false successes, "nothing touches" 3, "carried by the target" 1, and
K=20 misses 10 that are placed only later.

The episodes that separate LIBERO's check from placed, all with `open_lift`:

| episode | policy, faults | what happens after the release | video |
|---|---|---|---|
| `p_cam9_res32/c9-t0-i4` | X-VLA, camera 9 degrees | the bowl was set on the plate's rim; released, it slides off | `p_cam9_res32__c9-t0-i4_open_lift.mp4` |
| `p_cam9_res32/c9-t0-i5`, `c9-t1-i13`, `c9-t3-i16`, `c9-t8-i7` | X-VLA, camera 9 degrees | the bowl leaves the plate's 3 cm circle after release (not looked at one by one) | |
| `m200v_fp16res32_r4w10/m2-t0-i5` | X-VLA, camera 6 degrees | the same kind | `m200v_fp16res32_r4w10__m2-t0-i5_open_lift.mp4` |
| `p_obj_res32/o1-t7-i17` | X-VLA, healthy | the milk carton is wedged on the open fingers and comes up with them | `p_obj_res32__o1-t7-i17_open_lift.mp4` |
| `p_obj_res32/o1-t9-i3` | X-VLA, healthy | the orange juice comes up with the open gripper | `p_obj_res32__o1-t9-i3_open_lift.mp4` |
| `h200v_smolvla_r4w10/h1-t5-i1`, `h1-t9-i4`, `m200v_smolvla_r4w10/m1-t1-i3` | SmolVLA | the bowl was held 2.8 to 3.0 cm off the plate's centre; released, it settles 1 to 3 mm further out and LIBERO's check turns false, with the bowl still on the plate | `h200v_smolvla_r4w10__h1-t5-i1_open_lift.mp4` |
| `m200v_smolvla_r4w10/m2-t1-i3` | SmolVLA, camera 6 degrees | LIBERO's check still holds, but one finger keeps touching the bowl after the scripted lift | |

The two basket cases and the finger left on the bowl are partly artefacts of
the scripted lift, straight up while the object hangs on a finger; a policy
might shake it off. The bowl cases are not: the bowl rests on the plate's rim,
or within LIBERO's 3 cm, only because the gripper holds it there. Three of the
SmolVLA cases are borderline: the bowl ends on the plate, a millimetre outside
LIBERO's own tolerance. Under any definition that contains LIBERO's check they
are failures, and that is LIBERO's 3 cm, not the release.

## 3. Candidate definitions, scored on the scripted expert

All definitions are in `defs.py`, each a function of one simulator state
(the ones with K of a streak of states). Every one contains LIBERO's check.

| name | holds at a step when |
|---|---|
| (a) LIBERO | LIBERO's check holds |
| (b) released | (a), and nothing of the robot touches the object |
| (b'') not between both fingers | (a), and not both fingers touch the object |
| (b') gripper open | (a), and the pads are more than 70 mm apart (fully open is 78) |
| (c) K=10, K=20 | (b) and at rest, at each of the last K steps |
| **(d) placed** | (b), and the object is at rest (a first version also asked for contact with the target; dropped after the review, no number changed, section 4) |
| (d) K=5 | (d) at each of the last 5 steps |
| (e) carried by the target | (b''), the object touches the target, at rest, and the robot's contacts push on it with less than half its weight |

The expert of `spikes/teleop-expert` ran from LIBERO's initial states 0 to 49
of all 20 tasks: run `expert1000`, 1,000 episodes, one hour with two
processes. In 998 LIBERO's check fired; in 2 (bowl in the drawer) the expert
found no grasp. From the state at the check step, three continuations ran 80
steps each; the second and third by resetting and replaying the expert's own
actions up to the check step, identical in 1,996 of 1,996 replays:

- `expert`: the expert carries on (sets down, lets go, backs off), then holds
  still with the gripper open;
- `open_now`: the arm holds where it is and the gripper opens at once, a
  policy that lets go the moment the check fires;
- `carry_off`: the gripper stays closed, the arm rises 15 cm and moves 15 cm
  toward the robot, a policy that never lets go.

**Success within 40 steps after LIBERO's check** (spatial: 498 episodes,
object: 500):

| definition | spatial: expert | open_now | carry_off | object: expert | open_now | carry_off |
|---|---|---|---|---|---|---|
| (a) LIBERO | 498 | 498 | 498 | 500 | 500 | 500 |
| (b) released | 498 | 498 | 37 | 500 | 493 | 100 |
| (b'') not between both fingers | 498 | 498 | 91 | 500 | 500 | 100 |
| (b') gripper open | 498 | 498 | 0 | 500 | 500 | 0 |
| (c) K=10 | 498 | 497 | 19 | 494 | 478 | 100 |
| (c) K=20 | 498 | 497 | 19 | 485 | 458 | 100 |
| **(d) placed** | 498 | 498 | 21 | 499 | 487 | 100 |
| (d) K=5 | 498 | 497 | 19 | 497 | 485 | 100 |
| (e) carried by the target | 498 | 498 | 22 | 499 | 492 | 100 |

**Against the end state.** An episode counts as placed if 80 steps after the
check LIBERO's check holds and the robot does not touch the object (rest and
support are left out on purpose, so that (c), (d) and (e) are not compared
with themselves). "False success": the definition fired within 40 steps but
the object is not placed at the end. "Missed": placed at the end, but the
definition did not fire within 40 steps.

| definition | spatial: expert | open_now | carry_off | object: expert | open_now | carry_off |
|---|---|---|---|---|---|---|
| (a) LIBERO | 0 / 0 | 2 / 0 | **479** / 0 | 0 / 0 | 5 / 0 | **400** / 0 |
| (b) released | 0 / 0 | 2 / 0 | 18 / 0 | 0 / 0 | 5 / 7 | 0 / 0 |
| (b'') not between both fingers | 0 / 0 | 2 / 0 | 72 / 0 | 0 / 0 | 5 / 0 | 0 / 0 |
| (b') gripper open | 0 / 0 | 2 / 0 | 0 / 19 | 0 / 0 | 5 / 0 | 0 / 100 |
| (c) K=10 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 6 | 0 / 17 | 0 / 0 |
| (c) K=20 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 15 | 0 / 37 | 0 / 0 |
| **(d) placed** | 0 / 0 | 2 / 0 | 2 / 0 | 0 / 1 | 1 / 9 | 0 / 0 |
| (d) K=5 | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 3 | 0 / 10 | 0 / 0 |
| (e) carried by the target | 0 / 0 | 2 / 0 | 3 / 0 | 0 / 1 | 5 / 8 | 0 / 0 |

What is behind the non-zero cells for placed, read from the feature rows of
each episode:

- The 2 in spatial `carry_off` (task 7, initial states 10 and 49, bowl on the
  stove): the expert had set the bowl down and let go before LIBERO's check
  fired, so the bowl was placed at the check step; the scripted continuation
  then closed the gripper on it again and carried it off.
- The 2 in spatial `open_now` and the 1 in object `open_now`: placed when it
  fired; later the held-still arm sagged until a finger rested on the object
  again (0.003 to 0.1 N).
- The 9 misses in object `open_now`: the milk carton hangs on one open finger
  for 50 to 76 steps before it slides into the basket. Placed is right not to
  fire while it hangs; LIBERO's check fired the whole time.
- The 1 miss in object `expert` (milk, initial state 41): placed held after 44
  steps, the expert's slowest placement.

**Steps from LIBERO's check until the definition first holds**, expert
continuation (median / 90% / max):

| definition | spatial | object |
|---|---|---|
| (b) released | 1 / 2 / 4 | 13 / 17 / 38 |
| (b'') not between both fingers | 1 / 1 / 2 | 13 / 16 / 17 |
| (b') gripper open | 11 / 11 / 11 | 18 / 20 / 20 |
| (c) K=10 | 12 / 14 / 21 | 24 / 28 / 53 |
| (c) K=20 | 22 / 24 / 31 | 34 / 38 / 63 |
| **(d) placed** | 3 / 5 / 12 | 15 / 19 / 44 |
| (d) K=5 | 7 / 9 / 16 | 19 / 23 / 48 |
| (e) carried by the target | 3 / 5 / 12 | 15 / 19 / 44 |
| (the expert reports itself done) | | 20 / 26 / 51 (both suites) |

## 4. Recommendation

**Definition: placed, (d).** At a control step: LIBERO's check holds; no
collision geometry of the robot is in contact with the object; the object's
free joint moves slower than 1 cm/s and turns slower than 0.15 rad/s. One
state, no history, no threshold on forces.

A first version also asked that the object touch the target. The review
pointed out that this is redundant for the bowl (LIBERO's `On` already asks for
contact with the plate) and that for the basket it would turn an item resting
on another object inside the basket into a failure. No recorded state, expert
or policy, had the object released and at rest without touching the target,
so dropping the condition changes no number: every table of this report was
scored again with the new `defs.def_d` and came out identical
(`tables_*.before_s4.md` against `tables_*.md` under
`~/.cache/robots-probes/success/`).

Why this one:

- (a) counts every carried-away object as a success (879 of 879 in the expert
  runs) and 12 of 850 policy successes that do not survive a scripted release,
  and 15 of 532 X-VLA successes that do not end placed when X-VLA acts on
  (section 5).
- (b) fires the moment the object is free of the robot, whether or not it has
  come to rest: it fired on 18 bowls that were then carried away, at a step
  where the bowl was off the fingers for a moment but still moving at 2 to
  4 cm/s. The rest condition of (d) is what kept (d) from firing there.
- (b') and (b'') look at the gripper, not at the object. (b') missed all 119
  objects that came to rest in place while the gripper stayed closed (they had
  been let go before the check, or slipped out); (b'') fired on 72 bowls that
  were then carried away, at a step where one finger touched them, and counts
  a milk carton hanging from one finger. With X-VLA under the action delay,
  (b') holds within 60 steps in only 26 of 56 successes, against 47 for (d).
- (c) is slower by 9 steps (K=10) or 19 (K=20) and misses slow placements
  (K=20: 15 expert basket episodes, 10 policy releases within 40 steps; with
  X-VLA under the action delay 4 of 56 within 60).
- (d) with K=5 differed from (d) only by being 4 steps later on the expert; on
  X-VLA it misses 8 more under the action delay and 2 more on the basket.
  Nothing in these runs needed the streak.
- (e) behaves like (d) except where a finger still touches the object: 10
  false successes in the expert runs against 5 for (d), 1 in the policy
  releases against none, and it needs a force threshold. With X-VLA acting on
  its own, (e) and (d) give the same outcome in all 532 successes (and the same
  step in 512 of the 517 placed): X-VLA does not leave a finger resting on the
  object, so (e) is not needed.

**Orientation.** Placed has no condition on how the object stands, and the
data give no reason for one. A tilt feature (the angle between the object's
starting up axis and the vertical) was added after the review and all policy
episodes were replayed again with it (`replay850t`, `tail60t`; identical, same
scores). At the first placed step no bowl leaned more than 31 degrees (X-VLA
acting on its own: median 1.7, at most 18; scripted continuations at most 31;
none upside down), and basket items lay at up to 90 degrees, on their side,
which "in the basket" does not forbid (`analyze.py --tilt`, output
`tables_tilt.md`). The bound I would hold placed to is "no bowl above 45
degrees"; nothing came near it.

**What it means for the engine.**

- Do not end an episode on LeRobot's `terminated`. Evaluate LIBERO's check at
  every step as now, and from the first step it holds evaluate placed too.
  End the episode at the first step placed holds (success); 60 steps after
  LIBERO's check first held (failure, labelled "LIBERO only"); or at the step
  limit (failure), whichever comes first.
- 60 steps covers every placement measured with X-VLA acting on its own (at
  most 32 steps on the bowl and 39 on the basket healthy, 57 under the action
  delay), by the expert (at most 44), and after a scripted release with lift
  (at most 25). It does not cover a carton hanging on a still finger: in the
  expert's `open_now` continuation 9 such episodes became placed only 50 to 76
  steps after the check, 7 of them after more than 60 (milk, task 7, initial
  states 2, 5, 7, 23, 32 and 42 at 69 to 76 steps; orange juice, task 9,
  initial state 37 at 66). With X-VLA the 3 hanging items had not come free
  after 60 steps either.
- **The step limit decides how much the action delay costs.** In the faulted
  mix 28 of 138 LIBERO successes came less than 60 steps before the 280-step
  limit (25 of them under the delay), and 10 of the 12 "LIBERO only" episodes
  are such cut tails. Two ways to handle it: keep the limit for the whole
  episode (the robot's time budget includes letting go; the delay fault then
  costs 47 instead of 56 of 80 successes), or let the tail run up to 60 steps
  past the limit (a fault then costs under placed only what it costs in
  release; not measured, the run stopped at 280). I recommend the first, as the
  simpler rule; either way the delay fault's dose-response, calibrated with
  LIBERO's check, has to be measured again under placed.
- Log both outcomes per episode: `libero_success` with its step (comparable
  with published rates and with the stage-1 numbers) and `success` (placed)
  with its step. A "LIBERO only" episode is a natural near miss for the live
  layer and the oracle.
- Placed is a function of the state, and the engine's replay is identical, so
  original and replay get the same label without extra work. The index of
  object, target and robot geometry (`defs.GoalIndex`) is built once per scene,
  right after the reset. `defs.features` costs 1.35 ms per step in the bowl
  scene (87 contacts) and 0.59 ms in the basket scene (31 contacts), against
  13.6 and 10.8 ms for a control step without rendering: 5 to 10% of a step,
  measured once with `bench.py` at a load average of 5 to 7
  (`logs/bench.log`). It loops over MuJoCo's contact list in Python and only
  needs to run from LIBERO's check on.
- Cost, measured with X-VLA acting on its own: 0.28 extra policy calls per
  LIBERO success on the bowl and 0.44 on the basket (healthy), 0.54 in the
  faulted mix (0.84 under the action delay), against 4.1 and 5.0 calls per
  successful episode today: +7% and +9% on healthy successes. Failures are
  unchanged. SmolVLA was not run past the check; with the scripted release it
  would pay 0.7 extra calls on 10.7.
- The thresholds go into the gate-2 pre-registration as they are: 1 cm/s and
  0.15 rad/s.

**How much the rest thresholds matter.** `sweep.py` scores placed over the
recorded feature rows for linear thresholds of 0.5, 1, 2, 5 and 10 cm/s and
angular ones of 0.05, 0.1, 0.15, 0.3, 0.5 and 1 rad/s and none, as above
(output `~/.cache/robots-probes/success/sweep_expert1000_replay850.md`):

- At 0.15 rad/s the linear threshold changes nothing between 1 and 10 cm/s
  (expert: 5 false successes, 10 misses; policy releases: 0 and 0; 12 not
  placed; median steps to placed 3 bowl, 15 basket). At 0.5 cm/s the expert has
  one false success fewer and its slowest placement takes 45 steps instead of
  44.
- The angular threshold carries the result. Up to 0.15 rad/s nothing changes.
  At 0.3 rad/s one policy release counts as placed that does not stay so. At
  0.5 rad/s and above the expert's false successes rise to 9 (at 1 cm/s) and up
  to 22 (with a loose linear threshold): those are the bowls that were off the
  fingers for a moment while the gripper re-closed on them, still turning. The
  12 not-placed policy releases fall to 11 and, with no angular limit and 5 cm/s
  or more, to 9.
- The expert's median steps to placed move by at most 2; the policy releases'
  median goes from 6 (chosen) to 8 at 0.05 rad/s and to 3 with no limits.

So the result does not hinge on the exact values as long as the angular
threshold stays at or below about 0.15 rad/s.

## 5. What X-VLA does after LIBERO's check

`policy_tail.sh 60`, started from this session on 2026-10-07 at 17:55; the
first run of `fleet_tail.py`. X-VLA in the fleet's 16-bit setting
(`fp16_res32`, batches of 4, 10 simulation workers), the speed spike's three
manifests, step limit 280, and the policy keeps acting for 60 steps after
LIBERO's check first holds. `healthy200` ran at once (until 18:02); `object200`
and `mix200` waited behind the pi0.5 spike's lock and ran until 18:26 and
18:59; the replay (bit for bit, 532 of 532) and the scoring followed as light
jobs. Tables: `analyze.py --policy-run tail60 --window 60` with `--compare`
against the original runs; full output in
`~/.cache/robots-probes/success/tables_tail60_w60.md` (and `_w40`).

**Same episodes as before up to the check.** Every LIBERO success of the three
runs is a LIBERO success of the original run with the same settings
(`h200v_fp16res32_r4w10`, `p_obj_res32`, `m200v_fp16res32_r4w10`) and no other:
195, 199 and 138. In all 532 the check fired at the same step with the same
simulator state (hash). So what follows the check is the continuation of
exactly the episodes measured in stage 1 and replayed in section 2.

**Results, 60 steps after the check:**

| run | episodes | LIBERO successes | X-VLA commands open | (b) released | **(d) placed** | (d) K=5 | LIBERO only | rate: LIBERO / placed | steps to placed, median / 90% / max | extra calls per LIBERO success |
|---|---|---|---|---|---|---|---|---|---|---|
| `healthy200_tail60` (spatial) | 200 | 195 | 195 | 195 | **195** | 195 | 0 | 97.5% / 97.5% | 13 / 21 / 32 | 0.28 |
| `object200_tail60` (object) | 200 | 199 | 199 | 196 | **196** | 194 | 3 | 99.5% / 98.0% | 8 / 20 / 39 | 0.44 |
| `mix200_tail60` (spatial, faulted mix) | 200 | 138 | 129 | 129 | **126** | 118 | 12 | 69.0% / 63.0% | 16 / 34 / 57 | 0.54 |
| of these: healthy | 40 | 40 | 40 | 40 | 40 | 40 | 0 | 100% / 100% | 12 / 22 / 26 | 0.25 |
| of these: camera turned 6 degrees | 80 | 42 | 41 | 40 | 39 | 39 | 3 | 52.5% / 48.8% | 14 / 22 / 25 | 0.43 |
| of these: actions 17 steps late | 80 | 56 | 48 | 49 | 47 | 39 | 9 | 70.0% / 58.8% | 29 / 39 / 57 | 0.84 |

"X-VLA commands open": a gripper action below zero reached the simulator within
the 60 steps (or before the step limit). X-VLA does so a median of 6 steps
after the check on the bowl (90%: 12, at most 22), at once on the basket items
(median 0, at most 12; it had let go of most of them already), and a median of
21.5 steps after the check under the action delay, whose commands reach the
arm 17 steps late. With a 40-step window instead of 60 the mix has 123 placed
instead of 126; healthy is unchanged.

**The 15 LIBERO-only episodes:**

| episodes | what happens |
|---|---|
| `object200_tail60` `o1-t7-i17` (milk), `o1-t9-i3`, `o1-t9-i13` (orange juice) | X-VLA opens the gripper, but the item stays wedged on one finger and hangs inside the basket for the whole 60 steps; LIBERO's check holds throughout. The first two are the ones the scripted release lost too. |
| `mix200_tail60`, 9 under the action delay (`m3-t1-i1`, `i2`, `i3`, `i6`, `m3-t7-i0`, `i1`, `i3`, `i6`, `i7`) and `m2-t0-i2` (camera) | LIBERO's check fired late (steps 227 to 280), leaving 0 to 53 steps before the step limit; in 8 of these 10 no open command reached the arm in time. `m2-t0-i2` fired at step 280 itself. |
| `mix200_tail60` `m2-t0-i5` (camera 6 degrees) | the bowl slides off the plate's rim after X-VLA lets go, as with the scripted release |
| `mix200_tail60` `m2-t5-i7` (camera 6 degrees) | X-VLA opens at step 10 after the check and then closes on the bowl again; at the end it holds it on the plate |

**After placed.** Left running, X-VLA comes back to the bowl in 16 of 195
healthy episodes after it was placed (in 10 it grips it again with both
fingers), and at the end of the 60 steps LIBERO's check no longer holds; in the
faulted mix this happens in 6 more. The engine ends the episode at the first
step placed holds, so this does not enter the label. It does mean that a
check at a fixed step after the check would score these as failures: the
policy has no notion of being done.

## Surprises

1. **X-VLA lets go of basket items in mid-air** (115 of 199 successes) and
   LIBERO scores them while they fall. The scripted expert does the same with
   small items: its release height is limited by the open fingers clearing the
   basket walls, so it lets go of the cream cheese about 12 cm above the basket
   floor. Every item still landed in the basket.
2. **The gripper opening says little about release.** The alphabet soup can is
   62 mm wide and is held with the pads 62 mm apart; the fully open gripper is
   at 78 mm. A threshold on the opening either calls a held can released or
   waits for full opening, and it ignores objects that fall from a closed
   gripper.
3. **The hand, not only the fingers, keeps touching the bowl.** After opening
   with the arm held where it was, X-VLA's hand rests inside the bowl and in
   the episode in the video pushes on it with 15 to 21 N (the bowl weighs
   6 g); only moving the arm frees it.
4. **X-VLA has no notion of being done.** Left running after it has placed
   the bowl, it comes back to it in 16 of 195 healthy episodes and grips it
   again in 10 (section 5). Under LeRobot's evaluation this never shows,
   because the episode ends at the check.
5. **Recorded fleet runs can be read back without the policy.** The truth
   chain was meant for checking replays; it is also enough to recover every
   simulator state of a past run as a light job, about 2 s per episode with
   rendering. Later questions about object states in the stage-1 runs can be
   answered without the graphics chip.

## What I did not verify

- **SmolVLA after LIBERO's check.** Only X-VLA was run past the check
  (section 5); SmolVLA's numbers come from the scripted release.
- **A tail past the step limit.** The policy-side run stopped at 280 steps, so
  how many of the 10 cut tails in the faulted mix would have ended placed with
  more time is not known.
- **Orientation on the expert runs.** The tilt feature was added after the
  expert runs and measured on the policy replays only.
- **Other suites.** Only `libero_spatial` and `libero_object`. `libero_goal`
  and `libero_10` have drawers, stoves and two-object goals, where placed needs
  another form (a drawer has no object to let go of).
- **The engine's own loop.** The definition ran in this spike's runners, not in
  the engine being written.
- **The faulted-run separations one by one.** Of the six X-VLA faulted bowl
  episodes that do not end placed, I looked at two videos.
- **Whether LIBERO's training demonstrations include the release.** If they
  end at the check, the policies never learned to let go.

## Videos

Under `~/.cache/robots-probes/success/videos/`, each with an eight-frame
`_strip.png` beside it. The text line on every frame gives the step from
LIBERO's check and whether (a) LIBERO, (b) released, (d) placed and (c) K=20
hold, with the gripper opening, touch and the object's speed.

| file | what it shows |
|---|---|
| `p_cam9_res32__c9-t0-i4_{recorded,open_now,open_lift}.mp4` | X-VLA, camera 9 degrees: the bowl on the plate's rim at LIBERO's check; released, it slides off |
| `m200v_fp16res32_r4w10__m2-t0-i5_*.mp4` | X-VLA, camera 6 degrees: the same kind |
| `p_obj_res32__o1-t7-i17_*.mp4` | X-VLA: the milk carton comes up with the open gripper |
| `p_obj_res32__o1-t9-i3_*.mp4` | X-VLA: the orange juice comes up with the open gripper |
| `p_obj_res32__o1-t1-i12_*.mp4` | X-VLA: the cream cheese let go 11 cm above its resting height; LIBERO's check fires while it falls |
| `healthy200__h1-t0-i0_*.mp4` | X-VLA, healthy: the usual bowl case, gripped at the check; with `open_now` the hand stays in the bowl |
| `h200v_smolvla_r4w10__h1-t5-i1_*.mp4` | SmolVLA: the bowl held at the edge of LIBERO's 3 cm circle; released, it settles just outside |
| `healthy200_tail60__h1-t3-i2_recorded.mp4` | X-VLA acting on its own: lets go, the bowl is placed 3 steps after the check, then X-VLA comes back and grips it again |
| `object200_tail60__o1-t7-i17_recorded.mp4` | X-VLA acting on its own: opens, but the milk carton stays wedged on a finger for 60 steps ("LIBERO only") |
| `mix200_tail60__m3-t7-i1_recorded.mp4` | X-VLA, actions 17 steps late: LIBERO's check at step 253, the step limit comes before the release |
| `libero_spatial_t0_i0_{expert,open_now,carry_off}.mp4` | expert, bowl: carried away after LIBERO's check |
| `libero_object_t0_i0_*.mp4` | expert, alphabet soup: held 8 cm high at the check; carried away |
| `libero_object_t1_i0_*.mp4` | expert, cream cheese: let go 12 cm above the floor |
| `libero_spatial_t7_i10_*.mp4` | expert, bowl on the stove: placed before LIBERO's check fired; `carry_off` grips it again |

## Follow-up items

- (idea) Release faults (gripper that does not open, opens late
  or partly, sticky object), visible only under placed.
- (idea) Do LIBERO's demonstrations include the release?
- The scripted teleoperator drops small basket items from up to
  12 cm.
- Re-measure the action-delay dose-response under placed.

## How to run

```sh
make setup                                      # this checkout's environment
# 1,000 expert episodes with three continuations (1 hour with two processes on a busy machine)
spikes/success/run.sh expert1000 expert_tail.py --name expert1000 --inits 0-49 --tail 80 --workers 2 --chunk 5
# replay recorded policy successes with the two scripted continuations (37 minutes, two processes)
R=~/.cache/robots-probes/xvla-speed/runs
spikes/success/run.sh --baseline replay850 replay_policy.py --name replay850 --only-success --workers 2 --per-job 20 \
    --run ~/.cache/robots-probes/skeleton/runs/healthy200 --run $R/p_obj_res32 --run $R/m200v_fp16res32_r4w10 \
    --run $R/p_cam9_res32 --run $R/h200v_smolvla_r4w10 --run $R/m200v_smolvla_r4w10
# the tables (no simulator)
.venv/bin/python spikes/success/analyze.py --expert expert1000 --replay replay850
# a video of one episode
spikes/success/run.sh v video.py expert libero_object 0 0
spikes/success/run.sh --baseline v video.py policy $R/p_obj_res32 o1-t7-i17
# the policy-side run (heavy, needs the graphics chip; about 5 minutes per manifest once it has the lock)
spikes/success/policy_tail.sh 60
# its tables, with the check against the original runs
X=~/.cache/robots-probes/xvla-speed/runs
.venv/bin/python spikes/success/analyze.py --policy-run tail60 --window 60 --compare healthy200_tail60=$X/h200v_fp16res32_r4w10 \
    --compare object200_tail60=$X/p_obj_res32 --compare mix200_tail60=$X/m200v_fp16res32_r4w10
# threshold sweep (no simulator) and the cost of the features per step (light job)
.venv/bin/python spikes/success/sweep.py
spikes/success/run.sh bench bench.py
```

`run.sh` runs a script as a light job (at most four simulator processes, a
16 GB ceiling) and logs to `~/.cache/robots-probes/success/logs/`. It takes the
script's path relative to `spikes/success/` (it prefixes that directory).

`~/.cache/robots-probes/success/discarded/` holds the partial output of the
first two attempts at the expert and replay runs, stopped after a few minutes
to add per-finger contacts and contact forces to the features. Nothing in this
report comes from it.

## Files

| file | what it is |
|---|---|
| `defs.py` | The candidate definitions, LIBERO's check replicated, `features(m, d, ix)` |
| `expert_tail.py` | The expert past LIBERO's check, three continuations, features at every step |
| `replay_policy.py` | Replays recorded policy episodes from their truth chain, checks every state hash, adds the scripted continuations |
| `analyze.py` | All tables of this report, and `analysis_*.json` with the numbers |
| `video.py` | Annotated videos of single episodes |
| `fleet_tail.py`, `policy_tail.sh` | The policy-side run: X-VLA acting on after the check (section 5) |
| `sweep.py` | The threshold sweep of section 4 |
| `bench.py` | The per-step cost of `defs.features` |
| `run.sh` | Light-job wrapper |
