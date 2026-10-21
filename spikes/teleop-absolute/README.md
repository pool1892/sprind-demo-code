# The scripted teleoperator in X-VLA's control mode

Spike run on 2026-10-07 on this machine, in the locked
environment of this repository (`uv sync --locked` at `8a90ced`: MuJoCo 3.3.7,
robosuite 1.5.2, the vendored LIBERO with the port and the reset patch,
fleetlab's adapter `fleetlab.sim.libero_env.LiberoEnv`). Every simulation ran
as a light job (`tools/heavy-run --light`), four worker processes at most, no
model and no GPU inference. Other agents' jobs ran throughout (load average 6
to 15 in the run records).

Words used throughout:

- **Absolute control** is X-VLA's mode: each action is a target pose of the
  gripper in world coordinates. **Relative control** is the mode the expert
  was built and measured in: each action is a small change of pose.
- **Take-over**: the expert is switched on in a state someone else left
  behind and finishes the task from there.
- **Through a snapshot**: the complete simulator state at the hand-over is
  saved, the environment is reset to another start state with another scene
  seed and stepped, the snapshot is put back, and only then does the expert
  start. This is how the rollout engine will hand over.
- **Identical** means bit for bit, compared through SHA-256 hashes of every
  action and of MuJoCo's complete integration state after every step.
- **Success** is the strict outcome of `spikes/teleop-expert`: LIBERO's check
  is true after the expert has let go, backed off and waited 15 steps, and no
  finger touches the object. A step is 50 ms.
- **Earlier** numbers are those of `spikes/teleop-expert` (relative control,
  MuJoCo 3.8.1, another venv).

## Verdict

**The expert transfers to absolute control cleanly. Run the teleoperator in
X-VLA's mode; relative control stays available per episode and gives the
same outcomes.** The one number that moves is new: taking over from X-VLA's
own failures after its full 280 steps it recovers 89%, not 98%, because
X-VLA fails in ways the made-up disturbances did not cover; from the middle
of X-VLA's motion it recovers 99.5%.

| measured in absolute control, MuJoCo 3.3.7 | now | earlier |
|---|---|---|
| From LIBERO's stored start states, all 50 of all 20 tasks (`c_init_abs`) | **999 / 1000** | 1000 / 1000 |
| Same 200 start states, absolute against relative control (`c_init_abs`, `c_init_rel`) | 199 and 199, the same episodes | not measured |
| Take-over from the six kinds of disturbed state, through a snapshot (`c_displaced` .. `c_awkward`) | **1174 / 1193 (98.4%)** | 1177 / 1195 (98.5%) |
| Bowl upside down (`c_inverted`) | **0 / 50**, still a known loss | 0 / 100 |
| Take-over from X-VLA's recorded failures, after its 280 steps (`c_xvla_end`) | **474 / 535 (88.6%)** | not measured |
| Take-over in the middle of X-VLA's motion (`c_xvla_mid`) | **199 / 200** | not measured |
| The same, switched to relative control at the hand-over (`c_xvla_mid_rel`) | 199 / 200, the same episodes | not measured |
| Episodes run again in other processes, other order (`determinism.py`) | **135 of 135 identical** | |
| Take-overs without the snapshot against through it | **111 of 111 identical** | |
| Snapshot restores that gave back the saved state | 2178 of 2178 | |

1. **The port is a conversion, not a new controller.** The expert still plans
   every step as a small change of pose; `teleop/actions.py` turns that into
   the target pose robosuite would itself compute from it, in X-VLA's
   convention (position and rotation vector in world coordinates, gripper
   -1/+1). In both modes the arm is driven toward the same goal, so outcomes
   agree episode for episode (200 of 200 paired start states, 200 of 200
   paired mid-motion take-overs). The paths are not bit-identical: the
   expert's reading of the gripper pose is 2 ms older than the controller's,
   and the contact-rich simulation amplifies that (median path difference
   under 1 mm in 13 of 20 episodes, see "Control modes compared").
2. **The take-over works from X-VLA directly, without a GPU.** X-VLA episodes
   recorded by the walking skeleton and the xvla-speed spike replay bit for
   bit in the locked environment from their stored actions, so every X-VLA
   failure on disk can be handed to the expert exactly as X-VLA left it. I
   checked the state hash against the recording after every step of every
   one of the 935 X-VLA replays in the campaign.
3. **X-VLA's failures include three situations the expert has no skill
   for** (61 failed take-overs, causes read from the scene at the hand-over):
   - *another object on the target* (27): X-VLA put the ramekin (14), the
     other black bowl (9) or the cookie box (4) on the plate. The expert sets
     the right bowl down on top of it, which LIBERO does not count. Seen once
     with SmolVLA in the earlier spike.
   - *the bowl in a drawer that X-VLA pushed partly shut* (19): the slide
     joint of the cabinet's top drawer had moved from 14 to 16 cm at the
     start to 2 to 13 cm at the hand-over, toward closed (the video shows
     the drawer pushed in over the bowl). The expert grasps only from above,
     and the cabinet top covers the bowl.
   - *the bowl held with the arm stretched out* (8, seven of them in the
     drawer task): the elbow is at or near its limit, and the expert's escape
     move does not bring the bowl over the plate.
   A person at the controls would clear the plate or pull the drawer open, so
   on these the expert understates a teleoperator.
4. **Exact replay holds with the expert in the loop**, in another process,
   another order, through a snapshot or not, and with other image sizes (the
   13 videos are the measured episodes, bit for bit).

## What was built

| file | what it is |
|---|---|
| `teleop/` | The expert as one importable package: `actions.py` (the action convention), `expert.py` (`ScriptedExpert`), `grasp.py` (grasp planner), `scene.py` (read-only view of the simulator). `grasp.py` and `scene.py` are unchanged copies from `spikes/teleop-expert`; `expert.py` has the same motion logic with the conversion added. No global state; it imports only numpy, scipy, mujoco and `fleetlab` (for the import order). |
| `disturb.py` | The six kinds of disturbed state and the upside-down object, ported: every set-up action goes through the expert's conversion, so the set-up also runs in absolute control. Random draws as in the earlier spike. |
| `snapshot.py` | Complete simulator snapshot and restore, copied from `spikes/replay/snapshot.py` and pointed at fleetlab's adapter. |
| `xvla_record.py`, `xvla_select.py` | Read X-VLA's recorded episodes; choose the hand-overs from them. |
| `episode.py` | One episode from reset to verdict: start state, set-up or X-VLA replay with hash checks, snapshot, reset elsewhere, restore, take-over, success check, hash chain. |
| `evaluate.py`, `campaign.sh` | Batches in four worker processes; the whole campaign, run from a frozen copy of this directory (`code/8ffe649`). |
| `determinism.py`, `videos.py`, `tables.py`, `try_one.py` | Re-runs compared with the campaign; the videos; the tables below; one episode with its event log. |
| `probes/` | `replay_xvla_check.py` (do recorded X-VLA episodes replay here?), `forward_neutral.py` (does the hand-over procedure leave the physics alone?), `mode_paths.py` (do both modes drive the arm the same way?), `handover_scene.py` (what X-VLA left behind in the failed take-overs). |

### The action convention

The expert's `act()` returns the environment's 7 numbers in the episode's
control mode (`ScriptedExpert(env, control_mode=...)`):

| | absolute (default, X-VLA's) | relative (fallback) |
|---|---|---|
| `a[0:3]` | target position of the grip site, world frame, metres | change of position in units of 5 cm, in [-1, 1] |
| `a[3:6]` | target orientation as a rotation vector (axis times angle, angle in [0, pi]), world frame | change of orientation as a rotation vector in units of 0.5 rad, in [-1, 1] |
| `a[6]` | gripper: -1 open, +1 close, 0 hold | the same |

Internally the expert plans a unit command `u` per step, as before. In
absolute control the target is `p + 0.05 u[0:3]` and `exp(0.5 u[3:6]) R`,
with `p, R` the grip site's present pose: the goal robosuite's controller
computes itself from a relative command. `act()` checks at every step that the
arm controller is in the expert's mode (and, in absolute mode, reads targets
in world coordinates), and the constructor checks the
controller's scaling (5 cm and 0.5 rad per unit).

One difference to X-VLA's convention: robosuite's gripper uses only the sign
of `a[6]`, and X-VLA's converted actions are always -1 or +1. The expert also
sends 0 ("hold the present opening") while it approaches a grasp with the
gripper half closed in a tight spot. That happened in 178 of the 3,385
campaign episodes, 135 of them in the drawer task (spatial 4), the rest in
eight other spatial tasks.

### How a take-over runs (`episode.py`)

1. `LiberoEnv(suite, task, image_size=64, control_mode="absolute",
   hard_reset=False)`, robosuite's episode horizon raised from 1000 to 5000
   steps (set-up plus 600 expert steps can pass 1000). Images play no part in
   the expert; 64 pixels keep rendering cheap, and the videos show that the
   physics does not depend on the size.
2. `reset(init_state=i, seed=1000 + i)` for the stored start states, or the
   recording's own start state and scene seed for X-VLA.
3. Set-up: `disturb.apply_scenario` in absolute control, or X-VLA's recorded
   actions up to the hand-over step, the state hash compared with the
   recording after every step.
4. Snapshot (`snapshot.take`); reset to start state `(i + 25) mod 50` with
   seed `+ 7919` and three steps there; `snapshot.restore`; the restored state
   hash must equal the saved one.
5. Arm controller set to the expert's mode; `mj_forward`, so that positions
   and contacts belong to the restored state (after `mj_setState` they still
   belong to the scrambled one); a fresh expert with no memory of the set-up;
   at most 600 expert steps, then 15 steps holding still.

Scene seeds: 1000 + start state index, counting on from the reference
evaluation's seed 1000. They are not the seeds `lerobot-eval` gives its
environments, so the fixture positions differ from the reference runs by a
few millimetres; the stored start states are the same.

## Results

All numbers come from one run of `spikes/teleop-absolute/campaign.sh` at
commit `8ffe649` (3,385 episodes, 16:52 to 17:57, 65 minutes), which
wrote `~/.cache/robots-probes/teleop-absolute/results/c_*.jsonl` with a
`c_*.run.json` beside each (command, code hashes, time, load). The tables are
printed by `run.sh tables.py` (`python tables.py` works too, it simulates
nothing). Run ids are the file names.

### From the initial states (`c_init_abs`)

All 50 stored start states of all 20 tasks, absolute control, 558 s.

| suite | task | success | first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|
| spatial | 0 | 50/50 | 50 | 140 / 142 / 156 |  |
| spatial | 1 | 50/50 | 42 | 164 / 199 / 282 |  |
| spatial | 2 | 50/50 | 50 | 168 / 171 / 173 |  |
| spatial | 3 | 50/50 | 50 | 154 / 155 / 156 |  |
| spatial | 4 | 49/50 | 43 | 175 / 183 / 279 | no_grasp_found 1 |
| spatial | 5 | 50/50 | 50 | 152 / 155 / 160 |  |
| spatial | 6 | 50/50 | 50 | 202 / 209 / 216 |  |
| spatial | 7 | 50/50 | 48 | 253 / 291 / 353 |  |
| spatial | 8 | 50/50 | 50 | 139 / 143 / 171 |  |
| spatial | 9 | 50/50 | 50 | 177 / 187 / 196 |  |
| **spatial total** |  | **499/500** | 483 | 163 / 206 / 353 | no_grasp_found 1 |
| object | 0 | 50/50 | 50 | 150 / 151 / 152 |  |
| object | 1 | 50/50 | 50 | 121 / 123 / 124 |  |
| object | 2 | 50/50 | 50 | 129 / 134 / 134 |  |
| object | 3 | 50/50 | 50 | 131 / 147 / 151 |  |
| object | 4 | 50/50 | 50 | 149 / 151 / 151 |  |
| object | 5 | 50/50 | 50 | 142 / 144 / 144 |  |
| object | 6 | 50/50 | 50 | 143 / 144 / 144 |  |
| object | 7 | 50/50 | 50 | 130 / 137 / 151 |  |
| object | 8 | 50/50 | 50 | 132 / 140 / 142 |  |
| object | 9 | 50/50 | 50 | 131 / 132 / 136 |  |
| **object total** |  | **500/500** | 500 | 134 / 149 / 152 |  |
| **both suites** |  | **999/1000** | 983 | 147 / 188 / 353 | no_grasp_found 1 |

95% interval for both suites: 99.4 to 100.0%. Earlier: 1000/1000, 976 with
the first grasp. The one failure, spatial task 4 (bowl in the drawer) start
state 1: the first rim grasp, the only kind the planner found (6 candidates),
slipped while lifting, and then the planner found none. It fails the same way
in relative control. Spatial task 5 (bowl on the ramekin) is now 50 of 50
with the first grasp, earlier 34: MuJoCo 3.3.7 settles the bowl in the
ramekin where 3.8.1 left it on the rim. Spatial task 1 needed
a second grasp in 8 episodes, earlier none: the bowl came off the fingers in
transit after a rim grasp. Not looked into further.

### Control modes compared (`c_init_abs` start states 0 to 9, `c_init_rel`)

| suite | absolute | relative | both succeed | only absolute | only relative | both fail | same step count | steps median abs / rel |
|---|---|---|---|---|---|---|---|---|
| spatial | 99/100 | 99/100 | 99 | 0 | 0 | 1 | 44 | 165 / 163 |
| object | 100/100 | 100/100 | 100 | 0 | 0 | 0 | 30 | 133 / 133 |
| **both** | 199/200 | 199/200 | 199 | 0 | 0 | 1 | 74 | 147 / 145 |

The grip site's path, start state 0 of all 20 tasks in both modes
(`probes/mode_paths.py`, `results/mode_paths.json`): the same outcome in 20 of
20, the same sequence of events in 19 (object task 7 differs: 130 against 137
steps), the same step count in 9. The paths
separate by more than 1 mm after 7 to 17 steps; the median distance over an
episode is under 1 mm in 13 of 20 and at most 24 mm, the largest single
distance 108 mm. The large distances are timing: when a move ends one step
earlier in one mode, everything after it is shifted by a step, and at full
speed a step is 12 mm. Cause of the first separation, read from robosuite's
code: with `lite_physics` the controller computes the goal of a relative
command after recomputing the arm's pose for the first physics substep, while
the expert reads the pose the last substep of the previous step left, 2 ms
older.

### Take-over from disturbed states, through a snapshot

Ten start states per task and kind, as earlier; a few states could not be set
up and are not counted. The disturbances are drawn with the earlier spike's
formula but from other seeds (the scene seed is 1000 + start state here, 0 +
start state there), so 98.4% against 98.5% compares two samples of the same
kind of disturbance, not paired episodes.

| start state | suite | success | could not be set up | first grasp | steps median / 90% / max | failures | earlier |
|---|---|---|---|---|---|---|---|
| displaced | spatial | 96/100 | 0 | 89 | 189 / 236 / 540 | unreachable 4 | |
| displaced | object | 100/100 | 0 | 100 | 145 / 173 / 204 |  | |
| **displaced** | **both** | **196/200** | 0 | 189 | 161 / 221 / 540 | unreachable 4 | 197/200 |
| dropped | spatial | 96/98 | 2 | 93 | 166 / 211 / 307 | unreachable 2 | |
| dropped | object | 100/100 | 0 | 100 | 116 / 147 / 172 |  | |
| **dropped** | **both** | **196/198** | 2 | 193 | 139 / 196 / 307 | unreachable 2 | 196/199 |
| on_side | spatial | 99/100 | 0 | 93 | 199 / 234 / 601 | unreachable 1 | |
| on_side | object | 100/100 | 0 | 88 | 159 / 204 / 401 |  | |
| **on_side** | **both** | **199/200** | 0 | 181 | 173 / 234 / 601 | unreachable 1 | 198/200 |
| closed_empty | spatial | 97/98 | 2 | 89 | 165 / 250 / 499 | no_grasp_found 1 | |
| closed_empty | object | 100/100 | 0 | 100 | 137 / 148 / 159 |  | |
| **closed_empty** | **both** | **197/198** | 2 | 189 | 147 / 198 / 499 | no_grasp_found 1 | 196/198 |
| held | spatial | 97/97 | 3 | 94 | 75 / 112 / 283 |  | |
| held | object | 100/100 | 0 | 100 | 72 / 94 / 100 |  | |
| **held** | **both** | **197/197** | 3 | 194 | 73 / 98 / 283 |  | 198/198 |
| awkward | spatial | 94/100 | 0 | 86 | 215 / 291 / 369 | no_grasp_found 3, unreachable 2, grasp_missed 1 | |
| awkward | object | 95/100 | 0 | 87 | 165 / 250 / 579 | unreachable 5 | |
| **awkward** | **both** | **189/200** | 0 | 173 | 196 / 285 / 579 | unreachable 7, no_grasp_found 3, grasp_missed 1 | 192/200 |
| **six kinds** | **both** | **1174/1193** | 7 | 1119 | | | 1177/1195 |
| inverted (bowl) | spatial | **0/50** | 0 | 0 | - | grasp_missed 43, unreachable 7 | 0/100 |

Six kinds: 98.4%, 95% interval 97.5 to 99.0%; earlier 98.5%. The awkward
poses by kind: tilted 32/33, stretched 31/35, pressing 44/44, folded 43/43,
twisted 39/45. Causes of the 19 failures, sorted automatically from the
expert's event log: the arm wound up or stuck at joint limits 8 (six of them
in the twisted and stretched poses), stalls on the way to the grasp 4, no
grasp found 4, the forearm against the cabinet 3. Earlier the same four causes
appeared (6, 6, 4, 2 in another order).

### Take-over from X-VLA's recorded failures (`c_xvla_end`)

Every distinct failed X-VLA episode recorded in 32-bit floats by the walking
skeleton (`healthy200`, `dose_cam_p1`, `dose_cam_p2`, `dose_delay_p1`,
`dose_delay_p2`; `fleet` added nothing new) and the xvla-speed spike
(`h200v_fp32_r4w10`, `m200v_fp32_r4w10`, `p_cam3_fp32`, `p_cam6a_fp32`,
`p_cam6b_fp32`, `p_cam9_fp32`, `p_delay17_fp32`, `p_obj_fp32`), handed over
as it stood after its 280 steps. 534 are `libero_spatial`, 1 is
`libero_object` (X-VLA hardly fails there). The faults in those runs, a
turned scene camera and late actions, act on what X-VLA sees or when its
actions arrive, not on the physics, so the executed actions replay in a
healthy simulator; `xvla_select.py` drops an episode recorded twice.

| X-VLA's episode | handed over | recovered | first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|
| no fault | 12 | 9 | 6 | 151 / 278 / 292 | no_grasp_found 2, placed_wrong 1 |
| camera turned 3 deg | 27 | 24 | 23 | 157 / 211 / 339 | placed_wrong 2, no_grasp_found 1 |
| camera turned 6 deg | 195 | 177 | 162 | 190 / 296 / 588 | placed_wrong 10, no_grasp_found 7, dropped_in_transit 1 |
| camera turned 9 deg | 176 | 157 | 141 | 179 / 264 / 509 | placed_wrong 10, no_grasp_found 8, dropped_in_transit 1 |
| camera turned 12 deg | 33 | 28 | 27 | 171 / 205 / 240 | placed_wrong 3, no_grasp_found 1, unreachable 1 |
| actions 14 steps late | 2 | 1 | 1 | 72 / 72 / 72 | placed_wrong 1 |
| actions 17 steps late | 67 | 58 | 52 | 47 / 188 / 572 | unreachable 7, no_grasp_found 2 |
| actions 20 steps late | 23 | 20 | 20 | 52 / 154 / 202 | no_grasp_found 1, unreachable 1, placed_wrong 1 |
| **all** | **535** | **474** | 432 | 169 / 271 / 588 | placed_wrong 28, no_grasp_found 22, unreachable 9, dropped_in_transit 2 |

95% interval 85.6 to 91.0%. By task (spatial): 0 36/36, 1 21/21, 2 27/38,
3 15/15, 4 56/83, 5 60/66, 6 75/80, 7 74/84, 8 24/25, 9 86/87.

What X-VLA had left, from the hand-over record:

| state at hand-over | episodes | recovered | failures |
|---|---|---|---|
| object between the fingers | 86 | 78 | unreachable 8 |
| object at the target (within 6 cm), not counted | 70 | 64 | placed_wrong 6 |
| object tipped over | 3 | 3 |  |
| object elsewhere | 376 | 329 | placed_wrong 22, no_grasp_found 22, dropped_in_transit 2, unreachable 1 |

Why the 61 failed, from `probes/handover_scene.py` (what touches the target
and the object at the hand-over, which fixture joints moved;
`logs/handover_scene_end.log`, which also holds the one failed mid-motion
take-over below):

| situation at the hand-over | failed take-overs | tasks |
|---|---|---|
| another object on the plate: the ramekin 14, the other black bowl 9, the cookie box 4 | 27 | 5, 7 (ramekin), 2 (bowl), 6 (box) |
| the bowl in the top drawer, its slide joint moved by X-VLA from 14 to 16 cm to 2 to 13 cm, toward closed | 19 | 4 |
| the bowl between the fingers, arm stretched out, elbow at or near its limit | 8 | 4 (7), 9 (1) |
| the other bowl or the ramekin moved next to the plate, not touching it; placed but not counted | 3 | 2, 7 |
| no grasp found, other scenes (drawer opened further, two others) | 3 | 4, 6, 8 |
| could not reach | 1 | 7 |

### Take-over in the middle of X-VLA's motion (`c_xvla_mid`, `c_xvla_mid_rel`)

100 X-VLA episodes per suite drawn at random from `h200v_fp32_r4w10`
(spatial) and `p_obj_fp32` (object), handed over after a random number of
steps between 10 and the step before the episode ended (10 to 219, median
66), so before LIBERO counted a success; 198 of the 200 were successes for
X-VLA. 110 were handed over with the object between the fingers.

| expert's mode | suite | handed over | recovered | first grasp | steps median / 90% / max | failures |
|---|---|---|---|---|---|---|
| absolute | spatial | 100 | 99 | 96 | 115 / 174 / 265 | no_grasp_found 1 |
| absolute | object | 100 | 100 | 100 | 78 / 130 / 160 |  |
| **absolute** | **both** | **200** | **199** | 196 | 85 / 154 / 265 | no_grasp_found 1 |
| relative | spatial | 100 | 99 | 96 | 115 / 174 / 264 | no_grasp_found 1 |
| relative | object | 100 | 100 | 100 | 78 / 130 / 163 |  |
| **relative** | **both** | **200** | **199** | 196 | 87 / 154 / 264 | no_grasp_found 1 |

Paired: the same 199 recovered in both modes, the same step count in 112.
The failure is the drawer again (`h2-t4-i3` at step 101: drawer joint moved
from 14.7 to 13.3 cm, no grasp found; from `logs/handover_scene_end.log`). In
relative control the controller is switched from X-VLA's absolute mode at the
hand-over; that switch costs nothing measurable.

### Determinism (`determinism.py`, `results/det_summary.json`)

A sample of 135 campaign episodes (12 per batch, a quarter of them failures
where there were failures; the 3 upside-down ones), run again in new worker
processes, in reverse order and in groups of three instead of ten. Then the
111 take-overs among them again without the snapshot: the expert takes over
in the environment the set-up or X-VLA left, with no save, reset elsewhere or
restore.

| comparison | episodes | identical (every action and every state, 25,134 steps) |
|---|---|---|
| campaign against re-run | 135 | **135** |
| through a snapshot against directly | 111 | **111** |
| 64-pixel campaign against the 256-pixel video runs | 13 | **13** |

Two more checks on X-VLA's recordings (`probes/forward_neutral.py`, 12
episodes interrupted at a random step and played on to the end, every state
hash compared with the recording):

| interruption | identical to the end |
|---|---|
| `mj_forward` (the take-over does this) | 12 of 12 |
| snapshot, reset elsewhere, restore | 12 of 12 |
| both | 12 of 12 |
| restore of the lean set only: MuJoCo's state with warm start, the model's small arrays (fixture poses), the gripper controller, the sensor clocks; not the arm controller, the composite controller, the robot's buffers, robosuite's counters or LIBERO's trackers | 12 of 12 |

So in absolute control, as in relative, the arm controller's goal need not be
in a snapshot: the next action replaces it entirely. The replay spike had
listed this as unverified.

### Recorded X-VLA episodes replay here (`probes/replay_xvla_check.py`)

7 recorded episodes (2 successes from `healthy200` with soft resets, 2
failures from `healthy200` with hard resets, 3 failures with the camera
turned 9 and 12 degrees from `dose_cam_p2`) replay from their executed
actions with every state hash equal to the recording, at 64-pixel images
where they were recorded at 360. I ran the probe before the campaign without
keeping its output and again after the review with it kept:
`logs/replay_xvla_check.log`, 7 of 7 identical. The campaign checked every
step of 935 more replays (`episode._replay_xvla` at `8ffe649`: a hand-over
whose replay differs from the recording is skipped and marked; none was).

### Videos

`~/.cache/robots-probes/teleop-absolute/videos/`, written by `run.sh videos.py`,
each with an eight-frame `_strip.png` for looking at over SSH. Scene camera and
wrist camera side by side, 10 frames per second; the bar on top is orange
while the set-up or X-VLA acts and blue once the expert has taken over from
the restored snapshot. `index.json` lists each with its event log. All 13 are
the measured episodes bit for bit.

| file | what it shows | outcome |
|---|---|---|
| `xvla_end_healthy_held.mp4` | X-VLA without fault still holds the bowl after 280 steps; the expert carries it to the plate | success, 84 steps |
| `xvla_end_camera9_elsewhere.mp4` | camera turned 9 degrees, bowl left on the table | success, 147 |
| `xvla_end_delay17.mp4` | actions 17 steps late | success, 45 |
| `xvla_end_at_target.mp4` | bowl on the plate but off centre, not counted; the expert picks it up and sets it down in the middle | success, 135 |
| `xvla_end_failure.mp4` | X-VLA put the other black bowl on the plate; the expert sets the right one down on top of it three times and runs out of steps | failure (step budget) |
| `xvla_end_drawer_pushed_in.mp4` | X-VLA pushed the drawer shut over the bowl; no grasp from above | failure (no grasp) |
| `xvla_mid_spatial_held.mp4` | handed over at step 77 while X-VLA carries the bowl | success, 44 |
| `xvla_mid_object.mp4` | handed over at step 70 while X-VLA carries the cream cheese | success, 64 |
| `dropped_spatial.mp4` | bowl dropped from mid-air | success, 180 |
| `on_side_object.mp4` | cream cheese on its side | success, 143 |
| `closed_empty_spatial.mp4` | gripper closed on nothing next to the bowl | success, 147 |
| `awkward_twisted.mp4` | wrist wound up to its limit; unwinds, then picks | success, 220 |
| `inverted_bowl_failure.mp4` | bowl upside down: the grasps slide off | failure |

### Run times

| batch | episodes | wall time | episodes per minute | median s per episode in a worker | load average at the end |
|---|---|---|---|---|---|
| c_init_abs | 1000 | 558 s | 107 | 1.8 | 12.4 |
| c_init_rel | 200 | 107 s | 112 | 1.4 | 11.3 |
| c_displaced | 200 | 163 s | 74 | 2.4 | 10.6 |
| c_dropped | 200 | 189 s | 63 | 2.7 | 8.4 |
| c_on_side | 200 | 243 s | 49 | 3.1 | 9.1 |
| c_closed_empty | 200 | 277 s | 43 | 3.9 | 9.5 |
| c_held | 200 | 250 s | 48 | 3.7 | 9.6 |
| c_awkward | 200 | 322 s | 37 | 4.7 | 8.7 |
| c_inverted | 50 | 151 s | 20 | 8.7 | 7.4 |
| c_xvla_end | 535 | 1182 s | 27 | 7.9 | 7.2 |
| c_xvla_mid | 200 | 216 s | 56 | 2.7 | 6.6 |
| c_xvla_mid_rel | 200 | 221 s | 54 | 2.8 | 10.1 |

Four worker processes. A worker's time includes the set-up or X-VLA replay,
the snapshot (about 0.2 s), the reset elsewhere and the restore. The expert's
own step costs about as much as the simulator's.

### Changes after the review

The review asked for three small changes to the expert, made after the
campaign (commit `43a1654`): it takes robosuite's environment, the object the
contracts' `handle()` will provide (LIBERO's wrapper is still accepted and
unwrapped), and asks it for success directly; after it has finished, `act()`
holds the pose with the last gripper command instead of opening, so stopping
does not drop a carried object; `hold()` checks the controller's mode like
`act()`. Checked against the campaign by re-running every failed episode and
4 successes per batch (`determinism.py --prefix recheck --per-file 4
--all-failures --repeat-only`, `results/recheck_summary.json`):

| re-run with the changed expert | episodes | identical | outcome changed |
|---|---|---|---|
| successes | 44 | 44 | 0 |
| failures | 134 | 68 | 0 |

The 66 failures that differ (40 upside-down bowls, 24 X-VLA take-overs, 2
displaced) differ in the hash chain and final state only, not in steps or
outcome: the step after the expert gave up now holds with its last gripper
command (closed) where it used to open. All tables above are the campaign at
`8ffe649`.

## What I conclude

Kept apart from the measurements above.

- **Absolute control is the mode to use for the teleoperator.** It changes
  nothing measurable in the expert's results, and with the policy in the same
  mode the hand-over needs no controller switch. The fallback the plan names
  (relative control per episode) also works, including a switch in the middle
  of an episode, so nothing is lost by choosing absolute.
- **For X-VLA's failures the take-over rate lies between about 89% and
  99.5%, not at the 98% of made-up states.** About 89% is a lower bound:
  those are end-of-budget states (X-VLA had all 280 steps to make things
  worse), 431 of the 535 under a camera fault, 534 on one suite, and they
  come from 255 stored start states (401 of the episodes share theirs with
  another), so the interval of 85.6 to 91.0% overstates how independent they
  are. 99.5% (199 of 200) from the middle of mostly healthy motion is the
  upper bound. The live layer's case, a failing episode handed over before
  its end on a risk signal, lies between and was not measured. Within the
  failures, most failed take-overs come from two tasks (2 and 4) and one
  mechanism each. Under faults that make X-VLA push the drawer or put the
  wrong object on the plate, the teleoperator will look worse than a human
  would. If the stage-5 scoring treats "the teleoperator failed too" as "not
  recoverable", these cases need either the two skills (clear the target,
  open the drawer) or a label saying the expert lacks the skill. I would add
  "clear the target" first (27 of 61, and SmolVLA showed it too).
- **The take-over from a snapshot is ready for the engine** as far as the
  expert is concerned: deterministic, identical with or without the restore,
  and the lean snapshot of the contracts is enough in absolute control. One
  requirement follows for the engine: after a restore, MuJoCo's derived
  quantities must be recomputed (`mj_forward`) before an in-worker policy
  or the privileged state reads positions or contacts; it does not change
  the physics.
- **Moving the code into `fleetlab`:** `teleop/` moves as it is. The expert
  needs robosuite's environment of the task (today `LiberoEnv._env.env`;
  under the contracts the robosuite environment that `handle()` provides;
  LIBERO's wrapper is accepted too), and its `act()` is the in-worker
  policy's `act_one`. The hold command (0) is the one value outside X-VLA's
  action set that the log and the human model will meet.

## What I did not verify

- **X-VLA in the loop.** The heavy lock was taken and no model was loaded:
  every X-VLA state comes from replaying recorded episodes. They replay bit
  for bit (checked at every step), so the hand-over states are X-VLA's, but
  the X-VLA side of a live hand-over (the engine stopping the policy's chunk,
  switching to the expert in the same worker) was not run.
- **X-VLA failures on `libero_object`:** one episode. X-VLA failed once in the
  recordings there. The failure sample is also dominated by camera faults
  (431 of 535) and by the 32-bit model; 16-bit X-VLA was not used.
- **Take-over from a failing X-VLA episode before its end**, the live layer's
  case: not measured. Only end-of-budget failures (280 steps) and mid-motion
  hand-overs of mostly successful episodes were.
- **Independence of the X-VLA failures:** 535 episodes from 255 stored start
  states, several faults and doses of the same start; the 95% interval treats
  them as independent.
- **Other faults and suites for the X-VLA take-overs:** camera turns and late
  actions only, `libero_spatial` only in practice.
- **The fleet's success check.** This spike's strict check (LIBERO's check
  after release, 15 settle steps, no finger contact) is not the fleet's
  "placed" check of the success spike (`spikes/success`; object at rest, a 60-step tail), so the
  numbers under the engine may move a little.
- **The three missing skills** (clear the target, open the drawer, recover a
  stretched arm while holding) were found, not built.
- **Why spatial task 1 needed a second grasp in 8 of 50 episodes** where the
  earlier spike needed none, and why start state 1 of task 4 fails. Both
  fail the same way in relative control, so the control mode is not the
  cause; MuJoCo version and fixture positions are candidates I did not test.
- **The automatic sorting of failure causes** for the disturbed states rests
  on the expert's event log; I read a sample of the logs, not all 19.
- **Faults in the arm or gripper** together with take-overs, and the human
  model: not in scope.
- **The upside-down object in the grocery suite**: only the bowl was rerun
  (0 of 50), as asked; the earlier 100 of 100 for groceries was not repeated.
- **Speed on a quiet machine.** Times were taken with other agents' jobs
  running.
- **Hard resets** in the campaign: soft resets throughout (the recorded X-VLA
  episodes replayed with both).

## How to run

```sh
# one episode with its event log, optionally a video (256 pixels)
spikes/teleop-absolute/run.sh try_one.py libero_spatial 4 0
spikes/teleop-absolute/run.sh try_one.py libero_object 2 1 --scenario on_side --video /tmp/x.mp4
spikes/teleop-absolute/run.sh try_one.py --xvla ~/.cache/robots-probes/xvla-speed/runs/p_cam6a_fp32 c6-t4-i13

# a batch, four workers, one line per episode in results/NAME.jsonl
spikes/teleop-absolute/run.sh evaluate.py --name mytest --scenario dropped --inits 0-9
spikes/teleop-absolute/run.sh evaluate.py --name mytest --scenario init --mode relative

# everything behind this report (65 minutes), then the checks and the tables
spikes/teleop-absolute/run.sh probes/replay_xvla_check.py healthy200 h1-t0-i0 h2-t0-i0
spikes/teleop-absolute/run.sh xvla_select.py
spikes/teleop-absolute/campaign.sh
spikes/teleop-absolute/run.sh determinism.py
spikes/teleop-absolute/run.sh determinism.py --prefix recheck --per-file 4 --all-failures --repeat-only
spikes/teleop-absolute/run.sh probes/forward_neutral.py
spikes/teleop-absolute/run.sh probes/mode_paths.py
spikes/teleop-absolute/run.sh probes/handover_scene.py c_xvla_end
spikes/teleop-absolute/run.sh videos.py
spikes/teleop-absolute/run.sh tables.py
```

`run.sh` runs a script of this directory as a light job (8 GB ceiling) in the
locked environment with one numpy thread per process. `campaign.sh` refuses
to run with uncommitted changes here and runs from a copy of the committed
directory, so editing the code cannot reach a running campaign.

In code:

```python
import fleetlab
from fleetlab.sim.libero_env import LiberoEnv
from teleop import ScriptedExpert

env = LiberoEnv("libero_spatial", 0, control_mode="absolute")
env.reset(init_state=0, seed=1000)
env._env.sim.forward()                 # after a restore; harmless otherwise
expert = ScriptedExpert(env._env.env, control_mode="absolute")   # robosuite's environment
while not expert.done:
    env.step(expert.act())
print(expert.status, expert.events)
```
