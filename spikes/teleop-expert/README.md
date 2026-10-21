# Can a scripted expert stand in for the teleoperator?

Spike of 2026-10-06. Everything below was measured on this
machine in the venv `~/.cache/robots-probes/libero-port` (robosuite 1.5.2 with
the port patch, hf-libero 0.1.4, mujoco 3.8.1), on the suites `libero_spatial`
(bowl onto plate) and `libero_object` (grocery item into basket).

## Verdict

**Build it.** A scripted controller that reads the true object poses from the
simulator and drives the arm through the same 7-number action a policy sends is
realistic, and most of it now exists in this directory.

- **From LIBERO's own initial states it finished 1000 of 1000 episodes** (all 50
  initial states of all 20 tasks), 976 of them with the first grasp. A typical
  episode takes 143 control steps, which is 7 seconds at 20 steps per second.
- **Taking over in mid-episode it finished 1177 of 1195 episodes (98.5%)** in the
  six kinds of disturbed state it can handle in principle: object moved
  elsewhere, object dropped from the gripper, object knocked onto its side,
  gripper closed on nothing, object held in mid-air, arm left in an awkward
  pose.
- **On real failures of the SmolVLA policy it recovered 21 of 22** (one run of 50
  episodes on `libero_spatial`).
- **One kind it does not handle at all: a bowl lying upside down (0 of 100).**
  The gripper cannot get hold of it from above. The same test with the grocery
  items standing on their heads was 100 of 100.
- The other 18 failures have four causes: the object ended up behind the
  cabinet, where the forearm hits the cabinet (6); the planner found no grasp
  (6; in the two I looked into, the scene was fine at hand-over and the
  expert's own first moves, which are blind, had disturbed it); the arm had
  been wound up to its joint limits before the hand-over (4); repeated stalls
  on the way to the grasp (2).
- **It is hard to break with the faults I tried.** With the gripper's friction
  cut to a fiftieth it still finished 200 of 200, and with the object fifty
  times heavier 198 of 200. It took both at once (a tenth of the friction and
  fifty times the mass) to break it: 38 of 200. At two hundred times the mass,
  where the objects weigh 1 to 6 kg, it finished 163 of 200.

What this gives the project that a cost-and-probability model cannot: every
handed-over episode gets a real outcome (recovered or not, and after how many
steps) that depends on the actual state the policy left behind. "The
teleoperator could not have saved this one either" becomes something the
simulation shows instead of something we assume.

What it does not give: human behaviour. This expert sees the true state, never
hesitates and never slips. It is an upper bound on a teleoperator. Reaction
time, slower hands and human error would have to be added on top as a model
(see "Recommendation" at the end).

**One finding matters beyond this spike:** on the robosuite 1.5 port a LIBERO
reset is not repeatable. The cabinet and the stove land up to 1.7 cm apart from
one reset to the next, with the same seed and the same initial state. That
breaks replay-based ground truth. Cause and a two-line workaround are under
"Surprises".

## What was built

| file | what it is |
|---|---|
| `expert.py` | The controller, `ScriptedExpert`. Reads the simulator, returns one action per step, never writes to the simulator. |
| `grasp.py` | Grasp planner. Works on the object's current shape in the world, so it does not care whether the object stands, lies or sits on something. |
| `scene.py` | The privileged view: true poses, collision shapes as surface points, contacts. |
| `disturb.py` | Creates the disturbed start states and applies the model faults. |
| `runner.py`, `evaluate.py` | Run one episode; run batches in four worker processes and write one line per episode. |
| `common.py` | Builds the environment exactly as LeRobot's wrapper does, plus the reset fix. |
| `campaign.sh`, `tables.py` | The complete measurement run and the tables below. |
| `try_one.py`, `show.py`, `summarize.py`, `strip.py` | Look at single episodes, event logs, videos. |
| `takeover_smolvla.py`, `run_smolvla.sh` | Hand SmolVLA's real failures to the expert. Done once for `libero_spatial`, see the results. |
| `probes/` | The small experiments behind the design decisions and the surprises. |

How the expert works, in plain terms:

1. **It has no memory.** Whenever it is switched on it looks at the scene and
   decides where to continue. Task already satisfied: stop. Object between the
   fingers: carry it to the target. Otherwise: open the gripper, rise clear of
   everything, plan a grasp, pick, carry, set down. This is what makes it usable
   for taking over.
2. **Grasp planning is geometric, not per object.** The gripper always comes
   straight down. The planner tries a few thousand combinations of position and
   turning angle over the object and keeps those where the open fingers come
   down into free space, the palm clears the object and its neighbours, and the
   finger pads end up level with a part of the object that is narrower than the
   gripper opening. A bowl is wider than the gripper, so the planner ends up at
   the rim with one finger inside and one outside, without being told that it
   is a bowl. Planning takes 0.05 to 0.4 seconds.
3. **It carries the object's reference point over the target, not the
   gripper.** LIBERO's success check for the bowl wants the bowl's centre within
   3 cm of the plate's centre, and a bowl held by its rim hangs 4.7 cm off the
   gripper's centre.
4. **It sets the object down, then lets go**, and checks afterwards that the
   object is not hanging on a finger.
5. **It retries.** A missed grasp, a slip or a drop sends it back to step 1 with
   a different grasp, up to five times and within a budget of 600 steps.

"Success" in all tables is stricter than LIBERO's own check: LIBERO's check must
be true after the gripper has let go, retreated, and the scene has had 15 more
steps to settle. LIBERO's check alone turns true earlier, for the basket tasks
even while the object is still in the gripper above the basket floor (see
"Surprises").

## Results

All numbers come from one run of `spikes/teleop-expert/campaign.sh`
(3800 episodes, about 40 minutes), which wrote
`~/.cache/robots-probes/teleop/results/final_*.jsonl`. The tables are printed by
`spikes/teleop-expert/run.sh tables.py`. Steps are control steps of 50 ms.

### From the initial states

All 50 initial states LIBERO ships for each task. Command: `run.sh evaluate.py --name final_init --scenario init --inits 0-49`.

| suite | task | instruction | success | of which first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|---|
| spatial | 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 50/50 | 50 | 139 / 141 / 157 |  |
| spatial | 1 | pick up the black bowl next to the ramekin and place it on the plate | 50/50 | 50 | 164 / 190 / 193 |  |
| spatial | 2 | pick up the black bowl from table center and place it on the plate | 50/50 | 50 | 168 / 170 / 172 |  |
| spatial | 3 | pick up the black bowl on the cookie box and place it on the plate | 50/50 | 50 | 154 / 155 / 156 |  |
| spatial | 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 50/50 | 44 | 173 / 184 / 311 |  |
| spatial | 5 | pick up the black bowl on the ramekin and place it on the plate | 50/50 | 34 | 127 / 210 / 329 |  |
| spatial | 6 | pick up the black bowl next to the cookie box and place it on the plate | 50/50 | 50 | 201 / 209 / 216 |  |
| spatial | 7 | pick up the black bowl on the stove and place it on the plate | 50/50 | 48 | 245 / 280 / 310 |  |
| spatial | 8 | pick up the black bowl next to the plate and place it on the plate | 50/50 | 50 | 138 / 142 / 180 |  |
| spatial | 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 50/50 | 50 | 176 / 186 / 196 |  |
| **spatial total** |  |  | **500/500** | **476** | **165 / 205 / 329** |  |
| object | 0 | pick up the alphabet soup and place it in the basket | 50/50 | 50 | 149 / 150 / 152 |  |
| object | 1 | pick up the cream cheese and place it in the basket | 50/50 | 50 | 120 / 121 / 122 |  |
| object | 2 | pick up the salad dressing and place it in the basket | 50/50 | 50 | 132 / 134 / 134 |  |
| object | 3 | pick up the bbq sauce and place it in the basket | 50/50 | 50 | 130 / 146 / 151 |  |
| object | 4 | pick up the ketchup and place it in the basket | 50/50 | 50 | 147 / 149 / 151 |  |
| object | 5 | pick up the tomato sauce and place it in the basket | 50/50 | 50 | 142 / 144 / 146 |  |
| object | 6 | pick up the butter and place it in the basket | 50/50 | 50 | 141 / 142 / 143 |  |
| object | 7 | pick up the milk and place it in the basket | 50/50 | 50 | 129 / 138 / 152 |  |
| object | 8 | pick up the chocolate pudding and place it in the basket | 50/50 | 50 | 128 / 135 / 138 |  |
| object | 9 | pick up the orange juice and place it in the basket | 50/50 | 50 | 130 / 132 / 136 |  |
| **object total** |  |  | **500/500** | **500** | **132 / 148 / 152** |  |
| **both suites** |  |  | **1000/1000** | **976** | **143 / 190 / 329** |  |

### Take-over from disturbed states

Ten start states per task and kind, so 100 per suite and 200 per kind. A fresh expert with no knowledge of what happened before is switched on in the disturbed state. Command per kind: `run.sh evaluate.py --name final_KIND --scenario KIND --inits 0-9`. A few states could not be set up (for example the set-up pick itself failed); they are left out, not counted as failures. "First grasp" means the first grasp attempt carried through. The failure labels are the expert's own: `unreachable` means a move stalled before arriving.

| start state | suite | success | of which first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|
| displaced | spatial | 97/100 | 92 | 191 / 227 / 428 | unreachable 3 |
| displaced | object | 100/100 | 100 | 147 / 172 / 191 |  |
| **displaced** | **both** | **197/200** | **192** | **161 / 218 / 428** |  |
| dropped | spatial | 96/99 | 91 | 156 / 219 / 313 | unreachable 3 |
| dropped | object | 100/100 | 100 | 116 / 143 / 162 |  |
| **dropped** | **both** | **196/199** | **191** | **136 / 196 / 313** | 1 states could not be set up |
| on_side | spatial | 98/100 | 91 | 201 / 240 / 478 | unreachable 2 |
| on_side | object | 100/100 | 93 | 162 / 190 / 265 |  |
| **on_side** | **both** | **198/200** | **184** | **173 / 232 / 478** |  |
| inverted | spatial | 0/100 | 0 | - | grasp_missed 84, unreachable 16 |
| inverted | object | 100/100 | 100 | 152 / 177 / 189 |  |
| **inverted** | **both** | **100/200** | **100** | **152 / 177 / 189** |  |
| closed_empty | spatial | 96/98 | 93 | 170 / 215 / 440 | no_grasp_found 2 |
| closed_empty | object | 100/100 | 100 | 137 / 151 / 165 |  |
| **closed_empty** | **both** | **196/198** | **193** | **148 / 196 / 440** | 2 states could not be set up |
| held | spatial | 98/98 | 93 | 77 / 140 / 291 |  |
| held | object | 100/100 | 100 | 71 / 94 / 124 |  |
| **held** | **both** | **198/198** | **193** | **74 / 104 / 291** | 2 states could not be set up |
| awkward | spatial | 97/100 | 87 | 214 / 331 / 601 | no_grasp_found 3 |
| awkward | object | 95/100 | 87 | 168 / 292 / 400 | unreachable 4, no_grasp_found 1 |
| **awkward** | **both** | **192/200** | **174** | **198 / 320 / 601** |  |

### How the object actually lay after tipping and dropping

Tipping an object does not decide how it comes to rest, so this table sorts the same episodes by the measured lean of the object when the expert took over.

| start state | suite | object came to rest | success | failures |
|---|---|---|---|---|
| on_side | spatial | upright (< 30 deg) | 98/100 | unreachable 2 |
| on_side | object | on its side | 100/100 |  |
| inverted | spatial | upside down (> 150 deg) | 0/100 | grasp_missed 84, unreachable 16 |
| inverted | object | upside down (> 150 deg) | 100/100 |  |
| dropped | spatial | upright (< 30 deg) | 93/96 | unreachable 3 |
| dropped | spatial | on its side | 3/3 |  |
| dropped | object | upright (< 30 deg) | 100/100 |  |

### The awkward arm poses, by kind

Same 200 episodes as the `awkward` rows above.

| arm pose | success | of which first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|
| tilted | 41/42 | 39 | 179 / 237 / 388 | unreachable 1 |
| stretched | 44/47 | 38 | 254 / 343 / 601 | no_grasp_found 3 |
| pressing | 37/37 | 34 | 167 / 210 / 400 |  |
| folded | 37/37 | 36 | 156 / 187 / 313 |  |
| twisted | 33/37 | 27 | 259 / 352 / 528 | unreachable 3, no_grasp_found 1 |

### Faulted gripper and heavier object

From the first ten initial states of every task, with MuJoCo model parameters edited after the reset. `friction_X` multiplies the friction of the gripper's finger and pad surfaces by X and gives them priority, because MuJoCo otherwise uses the larger friction of the two touching bodies and a slippery gripper would have no effect. `mass_X` multiplies the mass and inertia of the target object by X. The objects are light to begin with: 5 g for the bowl, up to 32 g for the cartons. Command per level: `run.sh evaluate.py --name final_fault_F --scenario init --fault F --inits 0-9`.

| fault | suite / meaning | success | of which first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|
| none (first 10 states of final_init) |  | 200/200 | 193 | 144 / 185 / 310 |  |
| friction_0.5 | spatial | 100/100 | 98 | 164 / 199 / 312 |  |
| friction_0.5 | object | 100/100 | 100 | 133 / 149 / 152 |  |
| **friction_0.5** | pad friction 1.0 (normal 2.0) | **200/200** | **198** | **143 / 184 / 312** |  |
| friction_0.1 | spatial | 100/100 | 98 | 164 / 199 / 312 |  |
| friction_0.1 | object | 100/100 | 100 | 132 / 149 / 152 |  |
| **friction_0.1** | pad friction 0.2 (normal 2.0) | **200/200** | **198** | **142 / 185 / 312** |  |
| friction_0.02 | spatial | 100/100 | 84 | 168 / 255 / 548 |  |
| friction_0.02 | object | 100/100 | 100 | 133 / 149 / 152 |  |
| **friction_0.02** | pad friction 0.04 (normal 2.0) | **200/200** | **184** | **144 / 202 / 548** |  |
| mass_10 | spatial | 99/100 | 94 | 164 / 201 / 360 | no_grasp_found 1 |
| mass_10 | object | 100/100 | 99 | 132 / 149 / 154 |  |
| **mass_10** | object mass 52 to 317 g | **199/200** | **193** | **146 / 186 / 360** | **no_grasp_found 1** |
| mass_50 | spatial | 98/100 | 77 | 177 / 249 / 402 | no_grasp_found 2 |
| mass_50 | object | 100/100 | 100 | 137 / 154 / 169 |  |
| **mass_50** | object mass 262 to 1583 g | **198/200** | **177** | **153 / 214 / 402** | **no_grasp_found 2** |
| mass_200 | spatial | 67/100 | 12 | 264 / 460 / 541 | grasp_missed 17, slipped_in_lift 9, no_grasp_found 7 |
| mass_200 | object | 96/100 | 30 | 487 / 592 / 601 | unreachable 2, slipped_in_lift 2 |
| **mass_200** | object mass 1049 to 6333 g | **163/200** | **42** | **339 / 580 / 601** | **grasp_missed 17, slipped_in_lift 11, no_grasp_found 7, unreachable 2** |
| friction_0.1+mass_50 | spatial | 0/100 | 0 | - | slipped_in_lift 100 |
| friction_0.1+mass_50 | object | 38/100 | 33 | 130 / 140 / 306 | slipped_in_lift 61, dropped_in_transit 1 |
| **friction_0.1+mass_50** | pad friction 0.2 (normal 2.0); object mass 262 to 1583 g | **38/200** | **33** | **130 / 140 / 306** | **slipped_in_lift 161, dropped_in_transit 1** |

### Why take-overs failed

All kinds except the upside-down bowl. Causes read from the event logs (`run.sh show.py final_KIND --failed`).

Six kinds without 'inverted': 1177 of 1195 (98.5%). With the upside-down grocery items: 1277 of 1295.

| cause | episodes | where |
|---|---|---|
| forearm runs into the cabinet (object behind it) | 6 | displaced 3, dropped 3 |
| planner found no collision-free grasp | 6 | awkward (stretched) 3, closed_empty 2, awkward (twisted) 1 |
| arm kept stalling at its joint limits until steps or attempts ran out | 4 | awkward (twisted) 3, awkward (tilted) 1 |
| repeated stalls on the way to the grasp until steps or attempts ran out | 2 | on_side 2 |

All fault-free runs together: 2277 successes; steps median 148, 90% within 210, max 601. LIBERO's own success check turned true a median of 21 steps before the expert finished.

### Take-over from real failures of SmolVLA

`lerobot/smolvla_libero` on `libero_spatial`, five episodes per task, seed 1000, the evaluation
set-up of an earlier run on the port venv. Every episode the policy had not solved after its 280
steps was handed to the expert as it stood. Command: `spikes/teleop-expert/run_smolvla.sh
libero_spatial 5` (5.5 minutes under the heavy lock, 7.1 GB memory peak). Results in
`~/.cache/robots-probes/teleop/results/smolvla_takeover_libero_spatial.jsonl`.

| | episodes |
|---|---|
| run | 50 |
| solved by the policy | 28 |
| handed to the expert | 22 |
| recovered by the expert | **21 of 22**, 18 with the first grasp; median 181 steps, at most 424 |

What the policy had left behind in those 22 episodes (measured at hand-over):

- In 16 the bowl was not on the plate: 14 to 56 cm from the plate's centre, in all four
  drawer episodes still in the drawer. The gripper was open, half closed or closed on nothing.
- In 5 the bowl was **on the plate but 3 to 6 cm off its centre**, which LIBERO does not count
  (its limit is 3 cm). The expert picked it up and put it down again in the middle.
- In 1 the bowl was still between the closed fingers, resting on the table. The expert carried
  on from there (54 steps).
- In 2 of the 22 the arm was stretched out with the elbow straight (the first trap under
  "Surprises"). Both were recovered.
- The single failure is one of the 16: the event log shows the ramekin on the plate (the
  gripper touched it there, and the bowl could not be lowered closer than 3 cm to the plate).
  The expert set the bowl down on top of it three times and ran out of steps. **A target
  occupied by another object is a case it does not handle.**

This is one small run. It says the made-up disturbed states above are not wildly off, and it
adds one kind I had not thought of. The share of near-miss placements (5 of 22 failures) may
also be worth a look for the open question why SmolVLA scores only about 60% here.

## Which disturbed states it handles

| kind of state | how the test state was made | verdict |
|---|---|---|
| Object dropped on the table away from where it started (`displaced`, `dropped`) | `displaced`: the object is put down upright at a random free spot within reach. `dropped`: the expert picks it up, carries it to a random spot and lets go in mid-air. | **Handled**, 393 of 399. All failures are the same case: the bowl lies behind the cabinet as seen from the robot, and the forearm runs into the cabinet. The expert plans only for the gripper, not for the rest of the arm. |
| Object knocked over, lying on its side (`on_side`) | The object is tipped by 90 degrees, dropped from just above the surface and left to settle. | **Handled**, 198 of 200. Bottles, cans and cartons stay on their side and are picked up lying; a bottle is longer than the basket is wide, so the expert turns it to lie along the basket's diagonal. A bowl does not stay on its side: it rolls back upright, so for bowls this test is the same as `displaced`. |
| Object upside down (`inverted`) | Tipped by 180 degrees and dropped. | **Grocery items: handled, 100 of 100. Bowl: not handled, 0 of 100.** An upside-down bowl is a dome wider than the gripper, and its foot is a shallow cone the pads slide off. A person with this gripper would have to flip it by pushing it against something. |
| Gripper closed on nothing (`closed_empty`) | The gripper is driven down next to the object and closed. | **Handled**, 196 of 198. |
| Object held in the gripper in mid-air (`held`) | The expert picks the object up, carries it somewhere at a random height and stops. A fresh expert takes over. | **Handled**, 198 of 198, and fast (median 74 steps). |
| Arm stopped in an awkward pose (`awkward`) | Five kinds, picked at random: gripper tilted 25 to 50 degrees; arm stretched to the edge of its reach; gripper pressed onto the table and still pushing; gripper pulled in next to the robot's base; wrist turned about the vertical for four seconds until the joints run into their limits. | **Mostly handled**, 192 of 200. Tilted, pressing and folded poses are no problem. An arm stretched until the elbow is straight, or wound up until the forearm joint is at its limit, cannot always be recovered through this action interface. |

## Surprises

1. **A LIBERO reset is not repeatable on the port (measured).** Resetting the
   same task to the same initial state with the same seed, the way LeRobot's
   wrapper does it, put the cabinet at x = 0.0207, 0.0333, 0.0206 m in three
   consecutive resets, and somewhere else again in a second process; the
   largest difference seen was 1.7 cm. The
   initial-state files fix the movable objects and the robot, but not the
   fixtures, which LIBERO re-draws from a small range at every reset. Two
   separate causes:
   - LIBERO draws those positions from numpy's global random generator. On
     robosuite 1.5 `env.seed()` no longer reaches it: robosuite 1.5 stores
     `self.seed = None` on the environment, which hides LIBERO's `seed()`
     method, so the port's shim seeds a different generator that LIBERO's
     placement code does not use (`run.sh probes/seed_reaches.py`).
   - LIBERO keeps a list (`object_property_initializers`) that grows by one
     entry at every hard reset and is never cleared. Each entry draws one random
     number before the fixtures are placed, so even a correctly seeded reset
     number n differs from reset number 1. This part is LIBERO's own and should
     affect the pinned robosuite 1.4 stack too (not checked there).

   Workaround, used in `common.py: reset_to_init`: clear that list and call
   `np.random.seed(seed)` directly before `env.reset()`. With it, the state
   after reset, after 60 fixed actions and after a complete expert episode of
   163 steps are bit-identical over three resets and two separate processes
   (`run.sh probes/check_reset_repeatable.py libero_spatial 4 pin`, run twice).
   LeRobot's wrapper does neither, so an episode on `libero_spatial` cannot at
   present be reproduced by resetting with the same seed and initial state.
   `libero_object` has no fixtures and was repeatable as it is.
2. **A bowl held by its rim with upright pads gets squeezed out.** The bowl's
   wall leans outward by 14.5 degrees. Flat upright pads touch it at two
   different heights, the bowl slowly turns and creeps out of the grip within
   one to two seconds. Measured over 240 rim grasps each, lifting and then
   moving the bowl around: 71% held with the gripper upright, 88% with the
   gripper leaned to match the wall, and 68 of the 70 leaned grasps (97%) that
   had the pads 14 to 22 mm below the rim (`probes/bowl_grasp_study.py`). The
   expert now measures the lean of a thin wall and leans the gripper with it.
   My inference, not measured: the same effect is probably behind some of the
   bowls that pretrained policies drop.
3. **The arm has three traps, and a policy can leave it in any of them.**
   Through the Cartesian action interface (move the gripper, turn the gripper)
   the arm cannot be steered joint by joint, and
   - with the elbow straight at the edge of reach, small position commands do
     nothing (`probes/straight_elbow.py`: pulling back at full command for three
     seconds left the elbow straight and folded the hand over instead; pulling
     back while pitching the fingertips toward the robot bent it at once);
   - close to its own base the arm is folded up, and turning the gripper there
     twists the forearm instead of the last joint and dragged the gripper up to
     18 cm off course in my traces;
   - turning the gripper too far about the vertical winds the arm up until two
     joints sit at their limits, although the gripper's direction looks fine.

   The expert has an escape move for each, and they work most of the time
   (stretched 44 of 47, wound up 33 of 37). For the live layer this means: "arm
   in an awkward pose" is a real reason for a hand-over to fail.
4. **LIBERO's success check fires early.** For the basket tasks it only tests
   whether the object's centre is inside a box around the basket, so it turns
   true while the object still hangs in the gripper: in 400 of the 500 basket
   episodes from the initial states it was already true before the gripper
   began to open, by a median of 7 steps. For the bowl it turns true the moment
   the bowl touches the plate, still in the gripper. LeRobot's evaluation ends
   the episode at that step, so it never sees whether the object stays put. The
   expert finished letting go and backing off a median of 21 steps later.
5. **The milk carton can hook itself onto an open finger.** The ridge of its
   roof sticks out exactly as far as its sides. When the carton was released
   5 mm above the basket floor it dropped onto the finger pad's upper edge and
   came up again with the gripper (3 of 5 episodes in an early version). Setting
   objects down before opening, and stepping away from a finger that still
   touches the object, removed it.
6. **One environment process needs 1.5 to 2.7 GB and grows by 30 to 65 MB per
   reset** (`probes/memory.py`). Four workers hit the 8 GB ceiling within
   seconds until I stopped importing torch in the workers (0.6 GB each) and
   restarted each worker after five episodes. Without the offscreen renderer it
   is about 0.15 GB less. Worth knowing for the fleet's throughput.
7. **The simulated gripper is very forgiving.** See the fault table: the
   objects weigh a few grams and the pads have a friction coefficient of 2.0, so
   friction or mass alone can change by a factor of fifty with almost no effect
   on success, and only the two together make objects slip. A fault study on
   this gripper needs large steps, or faults in the grip force, to show an
   effect. One more detail, an estimate from the model's parameters and not a
   measurement: the fingers are position-controlled with a stiffness of
   1000 N/m, so on a thin object like the bowl's 2.6 mm wall they press with
   about 1 N, far below the 20 N limit that applies to thick objects.

## What I did not verify

- **Real policy failures.** One run: 50 episodes of one policy on
  `libero_spatial`, 22 hand-overs. Nothing on `libero_object`, nothing with a
  second policy, and nothing with injected faults, where a policy fails in
  other ways.
- **Other suites.** `libero_goal`, `libero_10` and `libero_90` contain tasks
  this expert has no skill for: opening and closing drawers, turning the stove
  on, pushing, putting things on a region of a fixture, two objects in a row. It
  refuses any goal other than "object on object" and "object in container"
  with an error.
- **Faults in the arm itself.** I changed gripper friction and object mass, as
  asked. I did not test a weak or sticky arm joint, wrong controller gains, a
  slow gripper or delayed actions. The expert closes the loop on the true
  gripper position at every step, so I expect it to tolerate moderate ones, but
  that is an expectation.
- **Disturbed states combined with faults**, and several disturbances at once
  (for example the object on its side and the arm stretched).
- **The pinned robosuite 1.4 stack.** Everything ran on the 1.5.2 port. The
  expert uses robosuite 1.5 names (`part_controllers["right"]`) and would need
  the same small shim LeRobot's wrapper has.
- **Four of the six "no grasp" failures.** I looked into two (`probes/why_no_grasp.py`:
  111 and 193 grasp candidates right after the disturbance, none 30 steps into
  the expert's own first moves). I assume the others are the same and did not
  check. I also did not check what exactly was pushed.
- **The disturbed states are mine.** They are plausible, not sampled from what
  policies actually do. In particular I chose how far objects are thrown (40 to
  70 cm from the robot's base) and that share decides how often "behind the
  cabinet" comes up.
- **Whether a human would do better or worse** on the cases the expert fails.
- **Step counts are for an expert at full speed** (up to 0.25 m/s). I did not
  model a human's pace.

## Recommendation

Build it, as the teleoperator for the pick-and-place suites, and put a small
model of the human on top.

**Remaining work, my estimate: two to three days** for `libero_spatial` and
`libero_object`.

| item | effort | note |
|---|---|---|
| Wire it into the fleet harness | half a day | `ScriptedExpert(env)` needs the inner LIBERO environment, which LeRobot's wrapper holds as `_env`. `takeover_smolvla.py` shows the hand-over in 20 lines. |
| Tests, tidy-up, fixed regression set | one day | The spike has no unit tests. The campaign is the regression test. |
| Human model on top | half a day to a day | Hand-over delay, a speed factor, optionally a probability of fumbling. This is where "cost" comes back in, on top of a measured outcome instead of in place of it. |
| Look before the first move | half a day | After taking over, the expert opens the gripper, rises and straightens up without checking what is next to it. In the drawer that can push things around. |
| Keep the forearm off the furniture | one to two days, optional | Would remove the "behind the cabinet" failures. The alternative is to leave them: a teleoperator who cannot reach is realistic. |
| Target occupied by another object | half a day to a day, optional | Seen once with SmolVLA. Needs "clear the target first", which is a second pick and place. |
| Upside-down bowl | uncertain, optional | Needs a new skill (flip it by pushing). I would leave it as a case the teleoperator loses. |

**Expected success rate once that is done:** 99% or better from initial states,
and 97 to 99% when taking over, with the upside-down bowl as a known zero. These
are the measured rates of the spike; I would not promise more than they show.

**Where to model instead of build:** any suite with drawers, knobs or
multi-step tasks (`libero_goal`, `libero_10`). Each new skill is roughly one to
two days. If the fleet has to cover those before the deadline, use the scripted
expert where it has the skill and a cost-and-probability model for the rest,
and say so in the evidence.

## How to run

```sh
# one episode with its event log, and a video of it
spikes/teleop-expert/run.sh try_one.py libero_spatial 4 0 --video
spikes/teleop-expert/run.sh try_one.py libero_object 2 1 --scenario on_side --video
spikes/teleop-expert/run.sh try_one.py libero_spatial 2 0 --fault friction_0.1+mass_50

# a batch: 10 start states per task, four worker processes, one line per episode
spikes/teleop-expert/run.sh evaluate.py --name mytest --scenario dropped --inits 0-9
spikes/teleop-expert/run.sh show.py mytest --failed      # event logs of the failures

# everything behind this report
spikes/teleop-expert/campaign.sh && spikes/teleop-expert/run.sh tables.py
```

`run.sh` puts every command under `tools/heavy-run --light` (8 GB ceiling) and
limits numpy to one thread per process. Scenarios: `init`, `displaced`,
`dropped`, `on_side`, `inverted`, `closed_empty`, `held`, `awkward`. Faults:
`friction_X` multiplies the gripper's friction by X, `mass_X` multiplies the
target object's mass by X, and `+` joins several.

In code:

```python
expert = ScriptedExpert(env)        # env: LIBERO's OffScreenRenderEnv, in whatever state it is
while not expert.done:
    env.step(expert.act())
print(expert.status, expert.events)
```

## Videos and frame strips

Each video shows the take-over only, from two cameras side by side (LIBERO's
`agentview` and `frontview`). The `_strip.png` next to it is eight frames of the
same video in one picture, for looking at over SSH.

| file (under `~/.cache/robots-probes/teleop/videos/`) | what it shows | outcome |
|---|---|---|
| `libero_spatial_t0_i0_init_none.mp4` | bowl onto plate from the initial state; the gripper leans to match the bowl's wall | success, 140 steps |
| `libero_spatial_t4_i0_init_none.mp4` | bowl out of the open drawer, gripper pre-closed to 44 mm to fit beside the drawer wall | success |
| `libero_object_t7_i2_init_none.mp4` | milk carton into the basket, set down before release | success |
| `libero_object_t2_i1_on_side_none.mp4` | bottle lying on its side, turned to the basket's diagonal | success |
| `libero_object_t7_i3_held_none.mp4` | takes over with the carton already in the gripper | success |
| `libero_spatial_t5_i2_dropped_none.mp4` | bowl dropped from mid-air, picked up where it landed | success |
| `libero_spatial_t6_i1_closed_empty_none.mp4` | gripper closed on nothing with the elbow straight; folds the arm back first | success |
| `libero_spatial_t2_i0_awkward_none.mp4` | arm wound up at its joint limits; unwinds, then picks | success |
| `libero_spatial_t2_i1_inverted_none.mp4` | bowl upside down: five grasp attempts slide off | **failure** |
| `libero_spatial_t7_i1_displaced_none.mp4` | bowl behind the cabinet: the forearm runs into the cabinet | **failure** |
| `libero_object_t0_i4_awkward_none.mp4` | arm wound up and not recovered within 600 steps | **failure** |
| `libero_spatial_t0_i0_init_friction_0.1+mass_50.mp4` | slippery gripper and a 280 g bowl: the bowl slips out of the grip at every attempt | **failure** |

Each has a `..._strip.png` beside it. Any episode of the campaign can be recorded again with
`run.sh try_one.py SUITE TASK INIT --scenario KIND --fault F --video`; the runs are repeatable.
