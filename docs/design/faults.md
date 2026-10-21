# Faults: the mechanism, the library and the dose-response curves

Written 2026-10-08 for stage 2 on
branch `stage2/faults`; the measured curves (section 6) were added on
`stage2/faults-calibration`. Stage 3 (branch
`stage3/faults`) added eleven types, the interaction, the drift and the
fault-by-fault table: sections 8 and 9. The specification is [the contracts](contracts.md)
(sections 3, 7 and 8) and the fault catalogue of
the experimental design (section 2). The code is
`src/fleetlab/faults/`, one module per fault type, and the calibration script
`src/fleetlab/dose_response.py`.

## 1. What exists

Fourteen fault types are registered in `fleetlab.faults`, each a module that
registers its class on import (importing the package loads them all). Twelve
are the catalogue of the experimental design, two are the generic kinds that the
resolution of an added change needs.

| Kind | Design name | Class | Acts on | Hooks | Level | Fixed by | Change log |
|---|---|---|---|---|---|---|---|
| `camera_pose_offset` | same | sensor | `scene_camera` or `wrist_camera` | cell | degrees (turn) or mm (shift) | recalibrate, replace | none |
| `camera_degradation` | same | sensor | a camera | observation | blur sigma px, noise sd, exposure stops | clean, replace | none |
| `action_latency` | `action_delay` | configuration | `control.action_delay_steps` | action | steps | set, revert | the key, 0 to n |
| `policy_knob` | same | configuration | `policy.sampling_steps`, `policy.actions_executed_per_chunk`, `policy.image_views`, `policy.precision` | policy input | 1 (the value is in params) | set, revert | the key, old to new |
| `control_mode_change` | same | configuration | `control.mode` | policy input | 1 | set, revert | `control.mode`, absolute to relative |
| `software_release` | `software_version` | software | `software.version` | those of what it carries | 1 | set, revert | `software.version` only |
| `gripper_wear` | same | physical | `gripper` | cell | wear, 0 to 1 | replace, service | none |
| `object_batch_variant` | `object_batch` | physical | condition `object_batch` | cell | doublings of mass (or halvings of friction) | change_condition | none |
| `furniture_shift` | same | site | condition `placement` | cell, observation, action | cm | change_condition | none |
| `illumination_change` | `lighting` | site | condition `lighting` | cell | stops | change_condition | none |
| `backdrop_change` | `background` | site | condition `clutter` (provisional) | cell | blend, 0 to 1 | change_condition | none |
| `success_detector_error` | same | data | `controller` | data | share of episodes logged wrong, 0 to 1 | service, replace | none |
| `config_change` | (added change) | configuration | any configuration key | dispatches by key | 1 | set, revert | the key |
| `condition_change` | (added change) | site | any condition | dispatches by condition | 1 | change_condition | none |

Every type carries its `Params` model, `dose_levels` (starting at 0, no
fault), a `TruthMapping` and `calibration`, the points the dose-response run
measures (a level with its params, target and nominal value). Component
faults are robot-scoped, condition faults site-scoped, configuration and
software faults may sit at any scope. Accepted diagnosis directions follow
the experimental design's claim 1: degraded (and drifting) for physical, sensor and data
faults, misconfigured or changed for configuration, changed for software,
changed (and drifting) for site.

`fleetlab.faults.records` gives what a fault writes into the fleet's records:
`change_entry(spec, day, author)` (a `Change` for the configuration and
software classes, None otherwise), `condition_value(spec)` (the true value of
the condition a site fault changed, in the units of section 4) and
`health_shift(spec)` (the camera's calibration residual in pixels, the
gripper's grip-force loss; nothing for the other types).

## 2. Choices, and where they depart from the experimental design

- **Four names changed.** The contracts' test that no fault kind can be read
  off the vocabulary (`test_the_vocabulary_names_no_fault_kind`) searches the
  contracts' sources for each kind name. `action_delay`, `software_version`,
  `object_batch` and `lighting` occur there as parts of configuration keys or
  as condition keys, so the kinds are `action_latency`, `software_release`,
  `object_batch_variant` and `illumination_change`; `background` became
  `backdrop_change`, a word the export's tables and skills will not use.
- **The command delay is a configuration fault.** The experimental design lists it as
  software, fixed by reverting `control.action_delay_steps`. Section 3 of the
  contracts reserves software faults to a `Change` of `software.version`, and a
  `Change` of `control.action_delay_steps` with its true values is a
  configuration fault. So `action_latency` is configuration, and the software
  exemplar is `software_release` carrying it (or any other type): the release
  shows only `software.version` in the change log, and its episodes are those
  of what it carries, bit for bit (tested). "Service controller", the second
  fix the experimental design names, would be a third type (a slow controller, a
  component fault); it is not built.
- **A turned camera is also fixed by replacing it**, at its higher cost: a new
  camera is mounted and calibrated.
- **Gripper wear** acts through two mechanisms, scaled by the wear (0 to 1):
  the pads' sliding friction falls by `friction_loss` (half at full wear),
  and the fingers stop short of closing, leaving a gap of `closed_gap_mm`
  (3 mm at full wear; `fingers="one"` puts all of it on one finger, a bent
  finger). The gap is set through the finger actuators' control range, which
  robosuite reads when it builds the gripper controller at every reset. Thinner
  pads would have been the literal "worn pad", but the finger collision meshes
  sit 1 mm apart when closed, so pads receding by more than about 0.4 mm only
  hand the grip to the bare fingers. The fingers are position-controlled at
  1000 N/m, so the gap takes the grip force on thin parts (a bowl's 2.6 mm
  rim) and leaves thick ones alone: on `libero_spatial`, all bowls, the
  effect is strong; on `libero_object` it should be small. The friction half
  does little at the default loss: at full wear the pads' 1.0 still exceeds
  the bowl's 0.95, and the teleoperator spike measured no effect at pad
  friction 1.0. The gap carries the fault.
- **Object batches** are measured in doublings of mass because LIBERO's
  objects are very light (5.6 g for a bowl): a real ceramic bowl is about 50
  times heavier (5.6 doublings). The solver constants MuJoCo derives from the
  masses are scaled with them, exactly as `mj_setConst` computes them for a
  free body (tested), without its rounding changes elsewhere. A slipperier
  batch gets priority in its contacts, because MuJoCo uses the larger friction
  of two touching geoms and the gripper pads' 2.0 would hide any change (the
  teleoperator spike found the same for the pads); with priority the object's
  contact parameters (solref, solimp, condim) apply to its contacts too, the
  pads' softer 0.01 and 0.5 among them. A control point (priority, friction
  unchanged) measures what that alone does.
- **A shifted table** is not done through LIBERO's placement sampler, as
  the experimental design proposed: the sampler moves fixtures but not the objects that
  stand on or in them (their poses come from the recorded start state after
  the reset), so a bowl in a moved drawer would start inside its wall.
  Instead the robot's base and the scene camera move the other way in the
  world (cell hook), the robot reports its end effector in its own believed
  frame (observation hook), and absolute targets land in that frame (action
  hook). The policy sees and acts exactly as if the table, with everything
  on it, had moved. "Recalibrate camera" does not fix it, so the only fixing
  verb is `change_condition placement`.
- **The surroundings have no condition key.** `backdrop_change` sits on
  `clutter` until the contracts decide.
- **The success detector errs only where the episode ends anyway.** The data
  hook may change the `success` of a verdict that is `done` and never `done`
  itself (contracts, section 7), so the episode ends on the same step and the
  robot does what it does without the fault. It misses a success
  (`mode="missed"`, the default) or records one where the fleet's check gave
  up on a placement that only LIBERO's check saw (`"false_success"`, the
  verdict "LIBERO only"), or both. A false success on an attempt that failed
  otherwise would need a done verdict there. Which episodes
  err is drawn once per episode, nested across levels.
- **A switched control mode is a policy input hook.** The effective spec
  carries the new mode and the engine sets the arm controller from the
  effective spec; X-VLA's adapter emits absolute target poses whatever the
  mode, so a controller switched to relative control reads each target pose
  as a change of pose, clipped and scaled to 5 cm and 0.5 rad per step. A
  first version emulated the switched controller in the observation and
  action hooks; the independent review measured it within 2.6 mm of a real
  relative controller after 20 steps, and the real controller is simpler and
  exact.
- **The precision knob is not calibrated.** The engine loads one policy per
  run in one precision, so a run cannot mix precisions; spikes/xvla-speed
  section 8 measured 32 against 16 bits on 1,200 paired episodes (798 each).
- **Calibration points beyond the experimental design.** The wrist camera's pan, three
  degradation modes, image views down to one (the engine's adapter takes 1
  to 4), a slipperier batch, a brighter light and a repainted table top are
  measured too, so that the stage-3 library starts from curves.

## 3. What the engine relies on

Recorded for the engine: the handle's `env` is robosuite's environment
(LIBERO's wrapper is unwrapped); `upload_texture` uploads into the
environment's render context; cell hooks run before the seeded reset and are
undone after the episode in reverse order; observation hooks return a new
dictionary that both the policy and the log receive; the environment takes
its control mode from the effective spec; the data hook changes only the
success of a done verdict. The engine's adapter (`LiberoSimEnv` on
`stage2/engine`) does all of this as written.

One constraint the review found: the engine's environment snapshot saves and
restores `body_pos` and `body_quat` of every body (the fixture poses) and the
gripper controller's state, which includes the control range it was built
from. `furniture_shift` (and `condition_change` on `placement`) moves the
robot's base through `body_pos`, and `gripper_wear` sets the control range. A
replay from a snapshot taken under one of these faults with the fault
*removed* would restore the faulted base or range without an undo, and the
change would outlive the episode; one *added* to a replay from a healthy
snapshot would be overwritten by the restore. Replays from the reset and
continuations with the same faults are unaffected. The engine should restore
only the fixtures' poses (LIBERO's `fixtures_dict`) and re-apply the cell
hooks after a restore, or refuse a from-snapshot job that removes or adds a
cell-hook fault (recorded for the engine).

## 4. The generic kinds and the values of conditions

`config_change` (`params = {"value": v}`, level 1, `nominal` the value it
replaced) dispatches on its key to the type that key works through, with the
level and params a fault of that type would carry: the four `policy.*` keys
to `policy_knob`, `control.mode` to `control_mode_change`,
`control.action_delay_steps` to `action_latency`. The camera calibrations are
the poses the fleet believes, which the policy never reads, and a version set
by an action carries nothing, so those keys do nothing. A `set` to a fault's
own value reproduces that fault's episodes exactly (tested).

`condition_change` calls the mechanism of the condition with the value
itself. The value of each condition, against the cell as LIBERO sets it up
(the nominal value in brackets):

| Condition | Value | Nominal |
|---|---|---|
| `lighting` | stops, signed (negative dimmer) | 0.0 |
| `placement` | the table's offset [x, y] in cm (x away from the robot, y to its left) | [0.0, 0.0] |
| `object_batch` | [mass doublings, friction halvings] | [0.0, 0.0] |
| `clutter` | blend of walls and floor toward another colour, 0 to 1 (provisional) | 0.0 |

The world generator writes these values into `Site.conditions` and into the
`nominal` of site faults; a change to a fault's own `condition_value`
reproduces the fault exactly (tested).

## 5. Tests

`make check` (`tests/test_faults.py`, on a toy MuJoCo cell with LIBERO's
names, and `tests/test_dose_response.py`): every type registered with the
truth mapping, class and fixing verbs above; every calibration point a valid
fault that each of its fixing verbs removes through `contracts.actions.resolve`
without adding a change; each type implements only the hooks it needs; the
delay line and the sensor noise are episode state that a snapshot restores;
blur, noise and exposure have the intended image statistics and never change
the input; the detector's misses are about the level's share and nested;
every cell hook at every calibration point changes the model and its undo
restores every array of the model and the headlight byte for byte, uploading
a changed texture on apply and again on undo; the solver constants of a
heavier batch equal `mj_setConst`'s; change entries come only from
configuration and software faults, with the true key, old and new values,
and no kind name, fault id or the word fault; the generic kinds and a
release reproduce the faults they stand for; the calibration statistics
reproduce the published numbers of spikes/xvla-speed section 8.

`make check-sim` (`tests/test_sim_faults.py`, LIBERO `libero_spatial` task 4,
no model): every cell-hook type at its strongest point changes the images or
the state as intended, and after its undo a healthy reset, also after a
healthy episode in between, has the fingerprint (simulator state with warm
start, fixture placement, both images) of a healthy reset in a fresh process.
A turned scene camera changes the scene image and not the wrist image; a worn
gripper at 0.5 closes 1.5 mm short; a batch 16 times as heavy presses 16 times
as hard on the table; a slipperier batch sets its contacts' friction; a
shifted table keeps the end effector's reported position and its targets in
the robot's frame; dimmer lights darken both images; a repainted backdrop
changes part of the scene image; blur, noise and exposure measured on
rendered images.

Through the engine (2026-10-08, a scratch merge of `stage2/engine` at
`40850b1` with this branch, CPU only, the engine's scripted in-worker policy,
`libero_spatial` task 4, 60 steps): fifteen cases, every type at one or two
calibration points, ran through `run_job`; each faulted episode run again
later in the same process gave the identical chain (fixture hash, state and
both image hashes at every step), and a healthy episode run after all of them
had the chain of the same healthy episode before them and in a fresh process.
The scripted policy ignores images and policy settings, so the degradation,
the knob and the detector left its chain as it was healthy; the other twelve
cases changed it.

## 6. Dose-response

**How it ran.** Each calibration point ran on 100 episodes of `libero_spatial`
(ten tasks, start states 0 to 9), the candidate decoys on 300 (start states 0
to 29); the healthy run holds 300 episodes, of which the first 100 are the
twins of every 100-episode point: same task, start state, scene seed and noise
seed, so every comparison is paired. X-VLA in `fp16_res32`, four batch rows,
eight workers (ten did not fit the engine's memory floor beside the other
jobs), the fleet's check ("placed"), no video. Run on 2026-10-08
with `python -m fleetlab.dose_response run --workers 8` from a scratch merge
(`calib/faults-engine` at `69d44ad`: `stage2/engine` at `8c5db04` with
`stage2/faults` at `291848f`). Software of every run: lock `12dd1f1237b40724`,
LIBERO patch set `cd1acebf8d9c749c`, MuJoCo 3.3.7, MIOpen database
`7751323820effb28`. The runs are under `~/.cache/robots-probes/faults/runs/`,
named by the run ids in the tables; each run replayed its sample of every
tenth episode unmodified (630 sample replays over the fourteen runs) and all
fourteen have `chain_ok` true. The two decoy levels were then extended to
1,000 pairs (the orchestrator's option a, 2026-10-08): 700 more episodes each
of the healthy run, lighting 2 stops dimmer and backdrop walls and floor at
blend 1, episodes 300 to 999 (start states 30 to 49, then 50 to 99, which
wrap to LIBERO's 50 recorded states with other scene and noise seeds), from
the same checkout and software, into their own run directories
(`healthy-1000`, `illumination_change-1000`, `backdrop_change-walls-1000`;
the log writer does not resume a run with another number of episodes per
day); all three have `chain_ok` true. The machine-readable curves and
verdicts are `src/fleetlab/faults/curves.json`
(`fleetlab.faults.records.measured`).

**Healthy baseline.** 968 of 1,000 (96.8%, 95% interval 95.5 to 97.7) by the
fleet's check and 970 by LIBERO's: 296 of the first 300 by both checks (the
same 296 episodes) and 672 of the 700 added for the decoys. The first 100,
the twins of every 100-episode point, 98 by both. The added 700 succeed
less often than the first 300 (96.0 against 98.7%, two-sample p about
0.03): they cover other start states; the comparisons are paired, so this
does not enter them.

**Columns.** *Placed*: the fleet's check before the data hook, which ends the
episode; *LIBERO's check*: whether LIBERO's own predicate fired at any step;
*Logged*: the visible outcome after the data hook; *Healthy twins*: the same
episodes without the fault, by the fleet's check; *Broken / rescued*: twins
that succeed only healthy / only faulted; the difference is faulted minus
healthy in points, with Newcombe's 95% interval. For the detector the
comparison is on the logged outcome; for every other type on the fleet's check.

**Against the stage-1 yardstick** (spikes/skeleton and xvla-speed section 8,
LIBERO's check, 200 other episodes, another runner):

| Fault | Stage 1 | Here, LIBERO's check (95% interval) | Here, placed | Here minus stage 1 (LIBERO's), points, p |
|---|---|---|---|---|
| scene camera 3 degrees | 179/200 (89.5%) | 86/100 (77.9 to 91.5) | 86/100 | -3.5, p 0.37 |
| scene camera 6 degrees | 109/200 (54.5%) | 62/100 (52.2 to 70.9) | 57/100 | +7.5, p 0.22 |
| scene camera 9 degrees | 54/200 (27%) | 36/100 (27.3 to 45.8) | 35/100 | +9.0, p 0.11 |
| delay 14 steps | "nothing" | 95/100 (twins 98) | 92/100, p 0.11 | no effect beyond noise in either |
| delay 17 steps | 145/200 (72.5%) | 69/100 (59.4 to 77.2) | 53/100 | -3.5, p 0.53 |

The curves reproduce within noise on LIBERO's check (two-sample tests,
independent episodes). Under the fleet's check the delay costs much more: at
17 steps LIBERO's check fired in 16 episodes that never ended placed, at 20
steps in 28 (51 against 23 of 100); the late commands keep moving the hand
after the bowl reaches the plate. The camera loses little to the stricter
check (5 episodes at 6 degrees).

**Two checks of the mechanism in the runs.** The software release carrying a
17-step delay produced the delay's episodes exactly: its 100 chain files
(every state and image hash, every executed action) equal those of
`action_latency` at 17 steps, byte for byte. The success detector left every
robot's true outcome and episode length as its healthy twin's (300 of 300);
it logged 77 and 62 successes at shares 0.2 and 0.4 (about 78 and 59
expected), and at a false-success share of 0.5 flipped one of the two
episodes the check ended as a failure.

**Verdicts**, by the rule: kept when removing the fault changes outcomes
beyond noise at a realistic level; a decoy at a level whose interval excludes
a loss of two points; a flat level whose interval does not is a small cause
(the experimental design, section 5). No type was dropped.

| Type | Class | Verdict | Levels with an effect (placed, points) | Levels flat at the sample size |
|---|---|---|---|---|
| `camera_pose_offset` | sensor | kept | scene pan 3, 6, 9 degrees: -12, -41, -63 | wrist pan 5 and 10 degrees (-1, -6) |
| `camera_degradation` | sensor | kept | blur 4 px: -25 | blur 1, 2 px; noise 10, 25; exposure 1, 2 stops darker, 1 brighter |
| `action_latency` | configuration | kept | 17, 20 steps: -45, -75 | 8, 14 steps (-2, -6) |
| `policy_knob` | configuration | kept | 1 image view: -85 | 5, 2, 1 sampling steps (+2 each); 10 actions per chunk (-4); 3, 2 image views (-3, -2) |
| `control_mode_change` | configuration | kept | relative control: -98 | |
| `software_release` | software | kept | carrying the 17-step delay: -45 | |
| `gripper_wear` | physical | kept | wear 0.75, 1: -11, -21 | 0.25, 0.5 (-1, -3) |
| `object_batch_variant` | physical | kept | 4, 6 mass doublings: -12, -18 | 2 doublings (-3); 2 friction halvings (-2); priority alone (-1) |
| `furniture_shift` | site | kept | 4 cm: -39 | 1, 2 cm (0, -2) |
| `illumination_change` | site | decoy at 2 stops dimmer (1,000 pairs: +1.2, interval +0.0 to +2.5) | none | 1 stop dimmer, 1 stop brighter at 300 pairs: -1.7, -1.0, lower bounds -4.2, -3.5 (small causes) |
| `backdrop_change` | site | decoy at walls and floor, blend 1 (1,000 pairs: +1.1, interval +0.0 to +2.3) | none | blend 0.5 at 300 pairs: -1.0, lower bound -3.5; table top at 100: -1.0, lower bound -6.7 (small causes) |
| `success_detector_error` | data | kept (on the logged outcome) | 0.2, 0.4 missed: -21, -36 logged; true outcomes unchanged | false successes 0.5 (+1) |

On realism: the camera at 3 degrees, the delay at 17 steps, a table moved by
4 cm, a batch of objects 16 to 64 times as heavy as LIBERO's (90 to 360 g
for a bowl; a ceramic bowl weighs about 300 g) and worn grippers that close
2 to 3 mm short are within what a fleet sees. Relative control and a single
image view are total: every episode on the robot fails, so they are
realistic as events but trivially detectable, and suit a world as a clear,
strong fault (development world D1). The flat levels are the "small causes"
of the experimental design's section 5 and the material for its confounder (a change in
the change log with no effect): five, two and one sampling steps, ten actions
per chunk and three or two image views changed nothing measurable on 100
episodes. A finding beside the faults: X-VLA does as well with one sampling
step as with ten here (100 of 100).

**The decoys.** At 1,000 pairs both decoy levels exclude a loss of two
points. Lighting 2 stops dimmer: 980 against 968, 11 broken and 23 rescued,
McNemar p 0.058, difference +1.2 points (95% interval +0.0 to +2.5).
Backdrop walls and floor at blend 1: 979 against 968, 9 broken and 20
rescued, p 0.061, +1.1 (+0.0 to +2.3). Both lean slightly the other way,
toward more successes, short of significance; a flip is about as likely in
either direction for an episode close to failing, and two decoys leaning
alike on the same added episodes is worth a look if a world shows the same.
These are the levels to plant as decoys. The other levels of the two types
(lighting 1 stop dimmer or brighter, a blend of 0.5, the repainted table top)
are flat but unconfirmed, so they count as small causes, not decoys; 1 stop
dimmer (-1.7 at 300 pairs) may carry a real loss of one or two points.

### The curves

#### camera_pose_offset: kept

| camera_pose_offset | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 3 deg scene pan | 100 | 86 (86.0%) | 77.9 to 91.5 | 98 | 86 | 86 | 14 / 2 | 0.0042 | -12.0 (-20.3 to -4.4) | `camera_pose_offset` |
| 6 deg scene pan | 100 | 57 (57.0%) | 47.2 to 66.3 | 98 | 62 | 57 | 42 / 1 | 1e-11 | -41.0 (-50.9 to -30.5) | `camera_pose_offset` |
| 9 deg scene pan | 100 | 35 (35.0%) | 26.4 to 44.7 | 98 | 36 | 35 | 65 / 2 | 3.1e-17 | -63.0 (-72.0 to -51.2) | `camera_pose_offset` |
| 5 deg wrist pan | 100 | 97 (97.0%) | 91.5 to 99.0 | 98 | 97 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | `camera_pose_offset` |
| 10 deg wrist pan | 100 | 92 (92.0%) | 85.0 to 95.9 | 98 | 93 | 92 | 8 / 2 | 0.11 | -6.0 (-13.2 to +0.5) | `camera_pose_offset` |

#### camera_degradation: kept (blur)

| camera_degradation | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 1 px blur | 100 | 98 (98.0%) | 93.0 to 99.4 | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | `camera_degradation` |
| 2 px blur | 100 | 98 (98.0%) | 93.0 to 99.4 | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | `camera_degradation` |
| 4 px blur | 100 | 73 (73.0%) | 63.6 to 80.7 | 98 | 74 | 73 | 26 / 1 | 4.2e-07 | -25.0 (-34.5 to -15.8) | `camera_degradation` |
| 10 noise | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 96 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `camera_degradation` |
| 25 noise | 100 | 98 (98.0%) | 93.0 to 99.4 | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | `camera_degradation` |
| 1 stops darker | 100 | 98 (98.0%) | 93.0 to 99.4 | 98 | 98 | 98 | 1 / 1 | 1 | +0.0 (-4.9 to +4.9) | `camera_degradation` |
| 2 stops darker | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 98 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `camera_degradation` |
| 1 stop brighter | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 96 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `camera_degradation` |

#### action_latency: kept

| action_latency | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 8 steps | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 96 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `action_latency` |
| 14 steps | 100 | 92 (92.0%) | 85.0 to 95.9 | 98 | 95 | 92 | 8 / 2 | 0.11 | -6.0 (-13.2 to +0.5) | `action_latency` |
| 17 steps | 100 | 53 (53.0%) | 43.3 to 62.5 | 98 | 69 | 53 | 46 / 1 | 6.8e-13 | -45.0 (-54.8 to -34.3) | `action_latency` |
| 20 steps | 100 | 23 (23.0%) | 15.8 to 32.2 | 98 | 51 | 23 | 75 / 0 | 5.3e-23 | -75.0 (-82.3 to -64.6) | `action_latency` |

#### software_release: kept (what it carries)

| software_release | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| release with a 17-step delay | 100 | 53 (53.0%) | 43.3 to 62.5 | 98 | 69 | 53 | 46 / 1 | 6.8e-13 | -45.0 (-54.8 to -34.3) | `software_release` |

#### policy_knob: kept (one image view)

| policy_knob | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 5 sampling steps | 100 | 100 (100.0%) | 96.3 to 100.0 | 98 | 100 | 100 | 0 / 2 | 0.5 | +2.0 (-2.0 to +7.0) | `policy_knob` |
| 2 sampling steps | 100 | 100 (100.0%) | 96.3 to 100.0 | 98 | 100 | 100 | 0 / 2 | 0.5 | +2.0 (-2.0 to +7.0) | `policy_knob` |
| 1 sampling step | 100 | 100 (100.0%) | 96.3 to 100.0 | 98 | 100 | 100 | 0 / 2 | 0.5 | +2.0 (-2.0 to +7.0) | `policy_knob` |
| 10 actions per chunk | 100 | 94 (94.0%) | 87.5 to 97.2 | 98 | 99 | 94 | 5 / 1 | 0.22 | -4.0 (-10.5 to +1.6) | `policy_knob` |
| 3 image views | 100 | 95 (95.0%) | 88.8 to 97.8 | 98 | 96 | 95 | 5 / 2 | 0.45 | -3.0 (-9.4 to +2.8) | `policy_knob` |
| 2 image views | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 96 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `policy_knob` |
| 1 image view | 100 | 13 (13.0%) | 7.8 to 21.0 | 98 | 13 | 13 | 86 / 1 | 1.1e-24 | -85.0 (-90.7 to -74.9) | `policy_knob` |

#### control_mode_change: kept

| control_mode_change | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| relative | 100 | 0 (0.0%) | 0.0 to 3.7 | 98 | 0 | 0 | 98 / 0 | 6.3e-30 | -98.0 (-99.4 to -91.8) | `control_mode_change` |

#### gripper_wear: kept

| gripper_wear | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 0.25 wear | 100 | 97 (97.0%) | 91.5 to 99.0 | 98 | 98 | 97 | 2 / 1 | 1 | -1.0 (-6.4 to +4.0) | `gripper_wear` |
| 0.5 wear | 100 | 95 (95.0%) | 88.8 to 97.8 | 98 | 95 | 95 | 4 / 1 | 0.38 | -3.0 (-9.2 to +2.4) | `gripper_wear` |
| 0.75 wear | 100 | 87 (87.0%) | 79.0 to 92.2 | 98 | 87 | 87 | 11 / 0 | 0.00098 | -11.0 (-18.7 to -4.8) | `gripper_wear` |
| 1 wear | 100 | 77 (77.0%) | 67.8 to 84.2 | 98 | 77 | 77 | 21 / 0 | 9.5e-07 | -21.0 (-30.0 to -13.0) | `gripper_wear` |

#### object_batch_variant: kept (heavier)

| object_batch_variant | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 2 mass doublings | 100 | 95 (95.0%) | 88.8 to 97.8 | 98 | 97 | 95 | 5 / 2 | 0.45 | -3.0 (-9.4 to +2.8) | `object_batch_variant` |
| 4 mass doublings | 100 | 86 (86.0%) | 77.9 to 91.5 | 98 | 87 | 86 | 14 / 2 | 0.0042 | -12.0 (-20.3 to -4.4) | `object_batch_variant` |
| 6 mass doublings | 100 | 80 (80.0%) | 71.1 to 86.7 | 98 | 81 | 80 | 20 / 2 | 0.00012 | -18.0 (-27.1 to -9.4) | `object_batch_variant` |
| 2 friction halvings | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 96 | 96 | 3 / 1 | 0.62 | -2.0 (-7.8 to +3.2) | `object_batch_variant` |
| contact priority only (control) | 100 | 97 (97.0%) | 91.5 to 99.0 | 98 | 97 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | `object_batch_variant` |

#### furniture_shift: kept

| furniture_shift | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 1 cm sideways | 100 | 98 (98.0%) | 93.0 to 99.4 | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | `furniture_shift` |
| 2 cm sideways | 100 | 96 (96.0%) | 90.2 to 98.4 | 98 | 97 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | `furniture_shift` |
| 4 cm sideways | 100 | 59 (59.0%) | 49.2 to 68.1 | 98 | 62 | 59 | 39 / 0 | 3.6e-12 | -39.0 (-48.8 to -29.0) | `furniture_shift` |

#### illumination_change: decoy at 2 stops dimmer; the other levels small causes

| illumination_change | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 1 stops dimmer | 300 | 291 (97.0%) | 94.4 to 98.4 | 296 | 293 | 291 | 7 / 2 | 0.18 | -1.7 (-4.2 to +0.5) | `illumination_change` |
| 2 stops dimmer | 1000 | 980 (98.0%) | 96.9 to 98.7 | 968 | 981 | 980 | 11 / 23 | 0.058 | +1.2 (+0.0 to +2.5) | `illumination_change + illumination_change-1000` |
| 1 stop brighter | 300 | 293 (97.7%) | 95.3 to 98.9 | 296 | 294 | 293 | 6 / 3 | 0.51 | -1.0 (-3.5 to +1.3) | `illumination_change` |

#### backdrop_change: decoy at walls and floor, blend 1; the other levels small causes

| backdrop_change | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 0.5 walls and floor | 300 | 293 (97.7%) | 95.3 to 98.9 | 296 | 294 | 293 | 6 / 3 | 0.51 | -1.0 (-3.5 to +1.3) | `backdrop_change-walls` |
| 1 walls and floor | 1000 | 979 (97.9%) | 96.8 to 98.6 | 968 | 979 | 979 | 9 / 20 | 0.061 | +1.1 (+0.0 to +2.3) | `backdrop_change-walls + backdrop_change-walls-1000` |
| 1 table top | 100 | 97 (97.0%) | 91.5 to 99.0 | 98 | 97 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | `backdrop_change-table` |

#### success_detector_error: kept (logged outcome)

| success_detector_error | Episodes | Placed | 95% interval | Healthy twins, placed | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| healthy | 1000 | 968 (96.8%) | 95.5 to 97.7 | | 970 | | | | | `healthy + healthy-1000` |
| 0.2 missed | 100 | 77 (77.0%) | 67.8 to 84.2 | 98 | 98 | 77 | 21 / 0 | 9.5e-07 | -21.0 (-30.0 to -13.0) | `success_detector_error` |
| 0.4 missed | 100 | 62 (62.0%) | 52.2 to 70.9 | 98 | 98 | 62 | 36 / 0 | 2.9e-11 | -36.0 (-45.7 to -26.3) | `success_detector_error` |
| 0.5 false successes | 100 | 99 (99.0%) | 94.6 to 99.8 | 98 | 98 | 99 | 0 / 1 | 1 | +1.0 (-3.2 to +5.8) | `success_detector_error` |

## 7. Not verified, and what is left

- The decoys are confirmed at one level each (section 6); their other
  levels only as small causes.
- **`libero_object` is measured by the world generator, not here.** Every
  curve of this file is on `libero_spatial`. The world generator's
  supplement (`fleetlab.worlds.supplement`, world `w91`,
  100 episodes per point against healthy twins of the same suite; committed
  as `src/fleetlab/worlds/curves_supplement.json` on `stage3/worlds` at
  `2f1c613`) measured 25 points on `libero_object` and the wear-and-batch
  pairs on both suites. It confirmed what this section expected for the
  gripper (`gripper_wear` flat at every level, +1 to +2 points: the thick
  objects are wider than the closed gap) and found the heavier batch flat too
  (-2 and -3 at 4 and 6 doublings), and found `libero_object` far more
  sensitive than `libero_spatial` to the image, frame and timing faults: the
  scene camera turned 3, 6 and 9 degrees -44, -71 and -97 points; blur 4 px
  -55; the delay 14, 17 and 20 steps -35, -93 and -98; the table moved 2 and
  4 cm -34 and -68. The detector's misses and relative control behave as on
  `libero_spatial`. Lighting 2 stops dimmer (-2, interval -7.8 to +3.2) and
  the repainted walls and floor (0, -4.9 to +4.9) are flat there at 100 pairs
  but not confirmed: **the decoys are confirmed on `libero_spatial` only**,
  and a world that plants them where robots run both suites rests the
  `libero_object` share on 100 pairs (section 8.6). The stage-3 types are not measured
  there (section 8.6); the column "On libero_object" of section 9 lists what
  is.
- Rendering under a changed texture was checked on LIBERO's own render
  context, not through the engine's `upload_texture`, which calls robosuite's
  `upload_texture` on the same context.
- `furniture_shift` was checked for a sideways shift. A shift toward the
  robot by several centimetres could bring the robot's mount into the table;
  not checked.
- The health shifts are first models (residual from the focal length,
  grip-force loss on a 4 mm test plate); the world generator adds noise.

## 8. Stage 3: the library at twenty-three types

Written 2026-10-08 on `stage3/faults`. The specification is the same as for
stage 2 (contracts sections 3 and 7, the experimental design's sections 1, 2 and 5), and
the method is section 6's: every calibration point on 100 `libero_spatial`
episodes, paired with healthy twins, X-VLA in `fp16_res32`, the fleet's
check; decoy candidates at 300 pairs, then 1,000 at the level a world would
use; the verdict rule of section 6 (kept, decoy, dropped), with one more
verdict, "inert", for a type that has no mechanism at all.

### 8.1 The eleven new types

| Kind | Class | Acts on | Hooks | Level | Fixed by | Change log |
|---|---|---|---|---|---|---|
| `lens_contamination` | sensor | `scene_camera` or `wrist_camera` | observation | film density, 0 to 1 | clean, replace | none |
| `camera_latency` | sensor | a camera | observation (frame buffer) | steps the frames are old | service, replace | none |
| `gripper_force_loss` | physical | `gripper` | cell (finger actuators) | share of force lost, below 1 | replace, service | none |
| `gripper_sticky_release` | physical | `gripper` | action (gripper command) | steps the opening lags | clean, service, replace | none |
| `arm_torque_loss` | physical | `arm` | cell (arm motors' gain) | share of torque lost, up to 0.9 | service, replace | none |
| `controller_overload` | physical | `controller` | action (growing buffer) | chance of a missed cycle, up to 0.5 | service, replace | none |
| `calibration_update` | configuration | `camera.scene.calibration`, `camera.wrist.calibration` | none (inert) | 1 | set, revert | the key, old pose to new |
| `image_pipeline_release` | software | `software.version` | observation (both cameras) | 1 (swap), percent cropped, 100 minus JPEG quality | set, revert | `software.version` only |
| `table_height_change` | site | condition `placement` | cell, observation, action | cm lower (or higher) | change_condition | none |
| `light_color_shift` | site | condition `lighting` | cell (lights) | share of the way to the new lamps' colour | change_condition | none |
| `success_false_positive` | data | `controller` | observation (listens only), data | share of eligible failures logged as successes | service, replace | none |

Two stage-2 types gained points, appended so that no episode id of stage 2
changed: `camera_pose_offset` the scene camera tilted down by 2, 4 and 6
degrees (a mount that creeps: the drift with a known kind of curve), and
`software_release` a release carrying two faults at once (a 14-step delay
and 10 of 30 actions per chunk, each flat alone at 100 episodes).

Every class has at least one new type: sensor two, physical four,
configuration one, software one, site two, data one. With the twelve
calibrated types of stage 2 that makes twenty-three; the two generic kinds
of an added change (`config_change`, `condition_change`) have no curve of
their own and are not counted.

### 8.2 Mechanisms and choices

- **`lens_contamination`** blends the image toward a blurred (10 px at full
  density, at 360 px) and hazed (35% toward a bright grey) version of
  itself by `level * map`, the map 1 at a smear's centre (the image's middle
  by default) falling to 0.3 far from it. A dirty lens is the natural drift:
  dust settles every day until someone cleans it, and `clean` is the
  cheapest fix in the catalogue. The health checks do not see it (the
  calibration residual measures pose, not clarity).
- **`camera_latency`** hands the policy and the log the frame of `level`
  steps ago, the joints current; the first frame is repeated until the
  buffer fills. The commands are on time, which tells it apart from
  `action_latency` in the logs (the picture lags the robot's own state).
  X-VLA reads the cameras once per chunk of 30 steps, so the levels are 6,
  12, 24 and 36 steps (the last longer than a chunk: the policy never sees
  a frame newer than its previous call); 3 steps, planned first, was
  dropped as certainly flat after the review.
- **`gripper_force_loss`** scales the finger servos' stiffness (1000 N/m)
  and force limit (20 N). Measured on LIBERO (`libero_spatial` task 4, the
  gripper closing on nothing): the finger joints carry 1 N of friction and
  heavy damping of their own, so a softer servo also closes more slowly and
  stalls short of closed. The opening after 20 and 60 steps of closing is
  1.9 and 1.0 mm healthy, 2.9 and 1.0 mm with 0.25 of the force lost, 5.7
  and 2.1 mm with 0.5 (still inside a bowl's 2.6 mm rim), 17.7 and 7.0 mm
  with 0.75 (no grip on a rim at all); at 0.9 the fingers take seconds to
  close. The levels were set to 0.25, 0.5 and 0.75 from that. It is the
  partner of a heavier batch in the exploratory interaction.
- **`gripper_sticky_release`** holds the executed gripper command at its last
  closing value for `level` steps after the command opens; a new closing
  command ends the hold. The commanded action in the log opens on time, the
  measured finger opening does not. Under LIBERO's check it would be
  invisible; under the fleet's placed check it costs success.
- **`arm_torque_loss`** scales the gain of the seven arm motors: the
  controller's gravity compensation and its pull to the target fall short
  together, so the arm sags. Measured with the hand held at its start pose:
  it settles 1.0, 2.5, 6.0 and 9.8 mm low, and about as far short toward
  the base, with 0.05, 0.1, 0.2 and 0.3 of the torque lost; on a reach 15
  cm forward and 10 cm down, 3, 8 and 13 mm low with 0.1, 0.2 and 0.3, and
  9 cm with 0.5, where the arm cannot hold the reach (the independent
  review's probe). The calibration runs 0.1, 0.2, 0.3 and 0.4: a millimetre
  at 0.05 is below the policy's own scatter, and 0.5 is total. The
  commanded targets in the log are unchanged; the measured pose falls short
  of them. Recalibrating the arm does not fix it.
- **`controller_overload`** is section 2's unbuilt slow controller, made
  distinct from the configured delay: a missed cycle (drawn per step from
  the episode's own generator) repeats the last command and leaves the new
  one waiting in the buffer, which never catches up, so the delay grows
  within each episode by about `level` steps per step. A fixed delay as a
  component fault would have produced `action_latency`'s episodes exactly,
  under another truth mapping.
- **`calibration_update`** writes a `Change` of a camera's believed pose and
  implements no hook: every episode is its healthy twin bit for bit (the
  engine pass in `make check-sim` checks the chain). It is the confounder
  entry the worlds need, "a change in the change log with no effect"
  (the experimental design, section 1), and its verdict is "inert".
- **`image_pipeline_release`** is a release whose version prepares images
  differently: red and blue swapped (BGR handed to an RGB model), a centred
  crop of 10 or 20% scaled back to full size (a driver's new region of
  interest; 5%, 9 pixels a side, was dropped as certainly flat), or JPEG
  compression at quality 30 or 10. It is a software type with a mechanism of its own, which
  `software_release` (a carrier of other types) cannot express. OpenCV's
  8-bit resize and its JPEG codec are deterministic (tested twice in a row
  and against a reference resize).
- **`table_height_change`** is `furniture_shift`'s mechanism in the
  vertical: the robot's base and the scene camera move up for a lower table,
  the robot reports and reaches in its own frame. The `placement` condition
  now takes [x, y, z]; a value written as [x, y] means what it meant, and
  `condition_change` counts a value equal to the nominal once the shorter is
  padded with zeros as no change ([0, 0, 0] is [0, 0]; for lighting [0, 0]
  is 0). `contracts.actions.resolve` compares literally, so a world should
  write a condition's nominal in the form its faults' values take.
- **`light_color_shift`** multiplies every light and the headlight per
  channel toward warm (about 3000 K) or cool (about 10000 K) lamps at the
  same luminance. Measured on the rendered images: the mean red over the
  mean blue rises 1.43 times at half the way to warm lamps and 2.1 times
  (scene camera; 1.9 wrist) at the full way, and falls to 0.84 times for
  cool lamps, while the mean luminance changes by 1 to 5%. The `lighting`
  condition now takes a number (stops) or [stops, warmth].
- **`success_false_positive`** answers section 2's open false success without a contract
  change: its observation hook only listens (returns the observation
  unchanged) to the finger opening, and on a done failure (LIBERO only, or
  the step limit, which the engine passes through the data hooks) of an
  attempt that closed and reopened the gripper it logs a success, in a share
  of episodes drawn once per episode and nested across levels. Its effect is
  the level times the share of eligible failures, so it is calibrated on
  top of a scene camera turned 6 degrees (`DosePoint.companions`), and
  compared with the same episodes' true outcome, which it cannot change.
  It should not sit on the same controller as `success_detector_error`: a
  success the other misses becomes a done failure, which this one would
  flip back.

The fixture hash of the engine's adapter covers every model array the new
cell hooks change (actuator gains, biases and force ranges, body and camera
positions, the lights); the headlight, which `light_color_shift` and
`illumination_change` scale, is not in it (a gap of stage 2, unchanged).
Nothing a new hook changes is in the adapter's lean snapshot, so the
engine's rule that a replay from a snapshot cannot add or remove a cell-hook
fault covers them (an open item for the engine). The chain
hashes the images as rendered, before the observation hook, so a
non-deterministic observation hook would show only through the policy's
actions; the image types' determinism rests on OpenCV's fixed-point 8-bit
code and libjpeg-turbo, checked by repetition in the tests.

### 8.3 Tests

`make check` (`tests/test_faults.py`, `tests/test_faults_library.py`,
`tests/test_dose_response_stage3.py`): every type registered with its class,
target, fixing verbs and hooks; every calibration point valid and removed by
each of its fixing verbs; every cell hook undone byte for byte on the toy
cell (which gained seven arm joints with torque motors); change entries only
from configuration and software types, in the fleet's words; a condition
change to a fault's own value reproduces every condition point exactly, the
height and the lamp colour included; per type, the mechanism (the dirty
lens softer where the smear lies and brighter overall; frames exactly
`level` steps old and replayed from a snapshot; servo, force limit and motor
gains scaled and nothing else; the opening held for exactly `level` steps;
the overloaded controller's delay never shrinking, nested by level, replayed
from a snapshot; the calibration record changing nothing; swap, crop and
JPEG as specified and repeatable; the lower table lifting the robot and
keeping its frame; the lamps' tints at unit luminance; the detector
flipping only failed placements, in a nested share); the release carrying
two faults equals the two; the stage-3 plan (no episode id repeated between
runs, every faulted episode with a healthy twin on its suite, joints and
companions in the same episodes, the 1,000-pair extension at indices 300 to
999) and its analysis on toy runs (inert, the detector on its own episodes,
an interaction, the merge into a copy of curves.json, twice without
duplicates).

`make check-sim` (`tests/test_sim_faults.py`, `tests/test_sim_faults_engine.py`):
on LIBERO `libero_spatial` task 4, the new cell hooks change the cell and
their undo restores a healthy reset's fingerprint, also after a healthy
episode in between; the tilted camera looks 6 degrees further down; the weak
arm sags monotonically with the level; the weak gripper closes more slowly
and stalls short; the sticky gripper opens 10 steps late; the lower table is
reached in the robot's own frame; warm lamps change colour, not brightness;
the dirty lens, the BGR swap and the late camera on rendered images; a
placement motion makes a failure eligible for the detector and idling does
not. Through the engine's own adapter and loop (`run_job` on `LiberoSimEnv`,
the scripted sway policy): every new kind, the tilt and the two-fault release
at their strongest points give the same chain when run twice; the types the
sway policy can feel change the chain, the image types and the calibration
record leave it as it is healthy; the detector logs a success on a failed
attempt and nothing else does; a healthy episode after all of them has the
chain it had before them and in a fresh process.

The runner's skips and conditions are tested with a stand-in for the
engine (`tests/test_dose_response_stage3.py`): a complete run is skipped, the
700 healthy episodes only when the chains differ from stage 2's, the
1,000-pair extension only when no level of the type loses, optional runs
only when named, and the healthy floor stops the job. A dry run of the
runner through the real engine on the CPU with the sway policy (two
workers, ten episodes per run, every kind of run of the plan) wrote, resumed
and skipped its runs and analysed them.

### 8.4 How the calibration runs

`python -m fleetlab.dose_response_stage3 run`, one resumable heavy job, from
a worktree of this branch at a fixed commit, into
`~/.cache/robots-probes/faults/runs/s3-*` beside the stage-2 runs. Order:
the healthy run (300) and a bit-for-bit comparison of its chains with stage
2's healthy run, which ran with another lock file (stage 2 `12dd1f1237b40724`,
stage 3 `a302a3256e62bf73`; the difference names only matplotlib, which
LIBERO already installed) (only if they differ, 700
more healthy episodes, else stage 2's episodes 300 to 999 are the twins of
the 1,000-pair decoy); the new kinds at every point and the appended points
(100 each); the interaction (both singles and the joint on the same 300
episodes; an exploratory pair at 100); the decoy candidate at 300 pairs per
level and, unless the 300 already show a loss, 1,000 at warm lamps; two
filmed episodes of every new kind with their healthy twins.

How it ran: queued at 04:23 from `/home/cs/code/robots-faults3-run`; the
first start (05:16) was refused by the engine's memory guard while another
project's build held 20 GiB, so the runner learned to wait and resume
(`cdaa5e8`); the second start ran the healthy run, was stopped for memory at
240 of 300 episodes and resumed, and then stopped on its healthy floor
because the engine's count after a resume covers only the resumed jobs, so
the runner now counts the run directory (`2e2c0b9`); the third start
(06:31, `2e2c0b9`) ran everything else until 09:00, 6,928 episodes in 19
runs plus their replay samples, in two and a half hours of queue, X-VLA in
`fp16_res32`, eight workers, every run's replay sample held (`chain_ok`
true in all). Two reviews of the results added runs, made by a fourth start
(14:20 to 15:17, from `594f025`, the branch merged with `main`): the pair
the curves suggested (700 episodes) and the five effects found at the edge
of significance, extended to 300 pairs (1,000 episodes); all `chain_ok`
true.

`libero_object` is left to the world generator's supplement
(`fleetlab.worlds.supplement`, world `w91`), which measures
`gripper_wear` at 0.5, 0.75 and 1, `object_batch_variant` at 2, 4 and 6
doublings and three wear-and-batch pairs on that suite, and was queued first;
the same runs exist here as optional runs (`s3-object-*`) made only when
named.

The interaction of a pair A, B on the same episodes is, per episode on the
fleet's check (1 or 0), `AB - A - B + healthy`, averaged: the joint effect
minus the sum of the single effects, negative when the two together cost
more than their sum, with a percentile bootstrap interval over episodes
(10,000 resamples, seed 0). Its role is "interaction" when the interval lies
below zero, "additive" when it covers zero, "sub-additive" above.

The decoy candidate's 1,000-pair extension runs only if no level of the type
loses successes beyond noise at 300 pairs: a type with an effect is kept,
and its flat levels are small causes, which 700 more pairs would not make a
decoy a world can plant. The inert type is checked chain by chain against
its healthy twins (`checks` in the analysis), not only by its outcomes.

The drift: the world generator moves a level linearly over the days
(contracts section 7 leaves drift semantics to it) and interpolates between
calibrated points of one family. The drift types' families are
`lens_contamination` on the scene camera (film 0.25, 0.5, 0.75 and 1: a film
growing by 0.1 a day passes them on days 2.5, 5, 7.5 and 10) and
`camera_pose_offset` tilted down (2, 4 and 6 degrees: a mount creeping 0.6
degrees a day passes them on days 3.3, 6.7 and 10).

### 8.5 Results

**Healthy twins.** The stage-3 healthy run, 300 episodes, made with
another lock file than stage 2's, has stage 2's chains bit for bit, all 300
(state, both images and the executed action at every step; stage 2's chain
files carry no fixture hash, so that one is not compared), and the same 296
successes. So stage 2's healthy run of 1,000 stays
the twin set (968 of 1,000), and stage 2's runs stay comparable: the 100
episodes of `gripper_wear` at 0.5 and of `object_batch_variant` at 4
doublings that stage 3 ran again equal stage 2's chain for chain (100 of 100
each).

**Verdicts.** Twenty-three calibrated types: nine new ones kept, one new
decoy, one inert; with stage 2's twelve, that is nineteen kept, three decoys
(`illumination_change`, `backdrop_change`, `light_color_shift`) and one
inert (`calibration_update`). None was dropped. Every class gained at least
one type: sensor two with an effect (`lens_contamination`,
`camera_latency`), physical four with an effect, software one
(`image_pipeline_release`), site one with an effect (`table_height_change`)
and one decoy (`light_color_shift`), data one (`success_false_positive`),
and configuration one without an effect: the inert `calibration_update`,
which is what the confounder's entry needs (configuration has no key left
whose change would be a new mechanism; its effects are stage 2's
`action_latency`, `policy_knob` and `control_mode_change`).

| Type | Class | Verdict | Levels with an effect (placed, points) | Flat levels |
|---|---|---|---|---|
| `lens_contamination` | sensor | kept | scene film 1: -10.7 (300 pairs) | scene film 0.25, 0.5, 0.75 (+1, +1, -3); wrist film 0.5, 1 (+1, 0) |
| `camera_latency` | sensor | kept | scene 24, 36 steps: -44, -39 | scene 6, 12 steps (+1, +1); wrist 12 steps (-2) |
| `gripper_force_loss` | physical | kept | 0.5, 0.75 lost: -56, -98 | 0.25 lost (-6, p 0.07) |
| `gripper_sticky_release` | physical | kept | 20, 40 steps late: -61, -98 | 5, 10 steps (-2, -3) |
| `arm_torque_loss` | physical | kept | 0.3, 0.4 lost: -13, -42 | 0.1, 0.2 lost (0, -4) |
| `controller_overload` | physical | kept | 0.1 missed: -8.3 (300 pairs); 0.2: -42 | 0.02, 0.05 missed (+1, +1) |
| `calibration_update` | configuration | inert | none | the rewritten record: every episode its twin's, chain for chain |
| `image_pipeline_release` | software | kept | crop 10%: -7.0 (300 pairs); 20%: -27 | red and blue swapped (-1); JPEG quality 30, 10 (+1, 0) |
| `table_height_change` | site | kept | 2, 4 cm lower: -12, -59; 2 cm higher: -9.7 (300 pairs) | 1 cm lower (-1) |
| `light_color_shift` | site | decoy at warm lamps (1,000 pairs: -0.4, interval -1.7 to +0.9) | none | half warm (-1.7), cool (-0.7) at 300 pairs: small causes |
| `success_false_positive` | data | kept (logged outcome) | 0.25, 0.5, 1 on a 6-degree camera: +12, +23, +43 logged | none |
| `camera_pose_offset`, added | sensor | kept | scene tilt down 4 degrees: -10.7 (300 pairs); 6 degrees: -24 | 2 degrees (-1) |
| `software_release`, added | software | kept | carrying a 14-step delay and 10 actions per chunk: -90 (the interaction below) | none |

Notes on the curves:

- **Thresholds, not slopes.** Most new types are flat at their mild levels
  and steep past a knee: the late camera between 12 and 24 steps (measured;
  why the knee lies there is not established: X-VLA reads the cameras once
  per 30-step chunk, but 24 and 36 steps, on either side of a chunk, cost
  about the same), the sticky gripper between 10 and 20 steps (the fleet's
  check allows 60 steps after LIBERO's first fire for the robot to let go,
  and X-VLA's own release takes up to about 40), the weak gripper between
  0.25 and 0.5 (where the stalled fingers no longer meet a bowl's rim), the
  overloaded controller between 0.05 and 0.1. A world generator that
  interpolates linearly from level 0 overstates the harm of a level below
  the knee.
- **The sticky gripper is invisible to LIBERO's check:** at 20 and 40 steps
  LIBERO's check still fired in 96 of 100 episodes, against 37 and 0 placed.
  It is the clearest case for the placed check.
- **The weak arm and the overloaded controller show in LIBERO's check less
  than in placed** (0.4 lost: 79 against 56; 0.2 missed: 73 against 56): the
  object gets there, the hand does not leave it at rest in time.
- **The false-positive detector** ran on top of a scene camera turned 6
  degrees (57 placed of 100, 43 failures), compared with the same episodes'
  true outcome, which it cannot change (0 broken by construction): it logged
  12, 23 and 43 of the 43 failures as successes at 0.25, 0.5 and 1; every
  failure there had closed and reopened the gripper, so the eligible share
  was 1. Its logged effect in a world is level x eligible share x failure
  rate, not a ratio of success.
- **The two-fault release is the strongest interaction measured:** a
  14-step delay (-6 alone, stage 2) and 10 of 30 actions per chunk (-4
  alone) together leave 8 of 100 (-90): with the policy called every 10
  steps and its commands arriving 14 late, every chunk is planned from a
  state its own previous commands have not reached yet.

The curves of the new types and of the points added to stage-2 types:

#### lens_contamination: kept

| lens_contamination | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 scene film | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-lens_contamination` |
| 0.5 scene film | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-lens_contamination` |
| 0.75 scene film | 100 | 95 (95.0%) | 98 | 96 | 95 | 5 / 2 | 0.45 | -3.0 (-9.4 to +2.8) | small_cause | `s3-lens_contamination` |
| 1 scene film | 300 | 264 (88.0%) | 296 | 265 | 264 | 36 / 4 | 1.9e-07 | -10.7 (-14.9 to -6.8) | effect | `s3-lens_contamination + s3-lens_contamination-300` |
| 0.5 wrist film | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-lens_contamination` |
| 1 wrist film | 100 | 98 (98.0%) | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | small_cause | `s3-lens_contamination` |

#### camera_latency: kept

| camera_latency | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 steps scene lag | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-camera_latency` |
| 12 steps scene lag | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-camera_latency` |
| 24 steps scene lag | 100 | 54 (54.0%) | 98 | 56 | 54 | 46 / 2 | 8.4e-12 | -44.0 (-54.0 to -32.8) | effect | `s3-camera_latency` |
| 36 steps scene lag | 100 | 59 (59.0%) | 98 | 61 | 59 | 40 / 1 | 3.8e-11 | -39.0 (-48.9 to -28.6) | effect | `s3-camera_latency` |
| 12 steps wrist lag | 100 | 96 (96.0%) | 98 | 96 | 96 | 4 / 2 | 0.69 | -2.0 (-8.1 to +3.6) | small_cause | `s3-camera_latency` |

#### gripper_force_loss: kept

| gripper_force_loss | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 force lost | 100 | 92 (92.0%) | 98 | 92 | 92 | 7 / 1 | 0.07 | -6.0 (-13.0 to +0.1) | small_cause | `s3-gripper_force_loss` |
| 0.5 force lost | 100 | 42 (42.0%) | 98 | 42 | 42 | 56 / 0 | 2.8e-17 | -56.0 (-65.2 to -45.2) | effect | `s3-gripper_force_loss` |
| 0.75 force lost | 100 | 0 (0.0%) | 98 | 0 | 0 | 98 / 0 | 6.3e-30 | -98.0 (-99.5 to -91.8) | effect | `s3-gripper_force_loss` |

#### gripper_sticky_release: kept

| gripper_sticky_release | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 steps late open | 100 | 96 (96.0%) | 98 | 97 | 96 | 2 / 0 | 0.5 | -2.0 (-7.2 to +2.3) | small_cause | `s3-gripper_sticky_release` |
| 10 steps late open | 100 | 95 (95.0%) | 98 | 97 | 95 | 3 / 0 | 0.25 | -3.0 (-8.7 to +1.5) | small_cause | `s3-gripper_sticky_release` |
| 20 steps late open | 100 | 37 (37.0%) | 98 | 96 | 37 | 61 / 0 | 8.7e-19 | -61.0 (-69.9 to -50.2) | effect | `s3-gripper_sticky_release` |
| 40 steps late open | 100 | 0 (0.0%) | 98 | 96 | 0 | 98 / 0 | 6.3e-30 | -98.0 (-99.5 to -91.8) | effect | `s3-gripper_sticky_release` |

#### arm_torque_loss: kept

| arm_torque_loss | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 torque lost | 100 | 98 (98.0%) | 98 | 98 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | small_cause | `s3-arm_torque_loss` |
| 0.2 torque lost | 100 | 94 (94.0%) | 98 | 96 | 94 | 6 / 2 | 0.29 | -4.0 (-10.7 to +2.1) | small_cause | `s3-arm_torque_loss` |
| 0.3 torque lost | 100 | 85 (85.0%) | 98 | 91 | 85 | 15 / 2 | 0.0024 | -13.0 (-21.5 to -5.2) | effect | `s3-arm_torque_loss` |
| 0.4 torque lost | 100 | 56 (56.0%) | 98 | 79 | 56 | 43 / 1 | 5.1e-12 | -42.0 (-51.9 to -31.4) | effect | `s3-arm_torque_loss` |

#### controller_overload: kept

| controller_overload | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 cycles missed | 100 | 99 (99.0%) | 98 | 99 | 99 | 0 / 1 | 1 | +1.0 (-3.2 to +5.8) | small_cause | `s3-controller_overload` |
| 0.05 cycles missed | 100 | 99 (99.0%) | 98 | 100 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-controller_overload` |
| 0.1 cycles missed | 300 | 271 (90.3%) | 296 | 293 | 271 | 27 / 2 | 1.6e-06 | -8.3 (-12.2 to -5.0) | effect | `s3-controller_overload + s3-controller_overload-300` |
| 0.2 cycles missed | 100 | 56 (56.0%) | 98 | 73 | 56 | 42 / 0 | 4.5e-13 | -42.0 (-51.7 to -31.8) | effect | `s3-controller_overload` |

#### calibration_update: inert

| calibration_update | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| scene calibration rewritten | 100 | 98 (98.0%) | 98 | 98 | 98 | 0 / 0 | 1 | +0.0 (-4.0 to +4.0) | inert | `s3-calibration_update` |

#### image_pipeline_release: kept

| image_pipeline_release | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| red and blue swapped | 100 | 97 (97.0%) | 98 | 97 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | small_cause | `s3-image_pipeline_release` |
| 10% cropped | 300 | 275 (91.7%) | 296 | 278 | 275 | 25 / 4 | 0.0001 | -7.0 (-10.8 to -3.6) | effect | `s3-image_pipeline_release + s3-image_pipeline_release-300` |
| 20% cropped | 100 | 71 (71.0%) | 98 | 72 | 71 | 28 / 1 | 1.1e-07 | -27.0 (-36.6 to -17.6) | effect | `s3-image_pipeline_release` |
| jpeg quality 30 | 100 | 99 (99.0%) | 98 | 99 | 99 | 1 / 2 | 1 | +1.0 (-3.7 to +6.1) | small_cause | `s3-image_pipeline_release` |
| jpeg quality 10 | 100 | 98 (98.0%) | 98 | 99 | 98 | 2 / 2 | 1 | +0.0 (-5.2 to +5.2) | small_cause | `s3-image_pipeline_release` |

#### table_height_change: kept

| table_height_change | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 cm lower | 100 | 97 (97.0%) | 98 | 97 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | small_cause | `s3-table_height_change` |
| 2 cm lower | 100 | 86 (86.0%) | 98 | 87 | 86 | 13 / 1 | 0.0018 | -12.0 (-20.2 to -4.8) | effect | `s3-table_height_change` |
| 4 cm lower | 100 | 39 (39.0%) | 98 | 41 | 39 | 59 / 0 | 3.5e-18 | -59.0 (-68.0 to -48.2) | effect | `s3-table_height_change` |
| 2 cm higher | 300 | 267 (89.0%) | 296 | 285 | 267 | 32 / 3 | 4.2e-07 | -9.7 (-13.8 to -6.0) | effect | `s3-table_height_change + s3-table_height_change-300` |

#### light_color_shift: decoy

| light_color_shift | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 warmer | 300 | 291 (97.0%) | 296 | 293 | 291 | 7 / 2 | 0.18 | -1.7 (-4.2 to +0.5) | small_cause | `s3-light_color_shift` |
| 1 warmer | 1000 | 964 (96.4%) | 968 | 968 | 964 | 23 / 19 | 0.64 | -0.4 (-1.7 to +0.9) | decoy | `s3-light_color_shift + s3-light_color_shift-1000` |
| 1 cooler | 300 | 294 (98.0%) | 296 | 295 | 294 | 4 / 2 | 0.69 | -0.7 (-2.9 to +1.3) | small_cause | `s3-light_color_shift` |

#### success_false_positive: kept

| success_false_positive | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 on a 6 deg camera | 100 | 69 (69.0%) | 57 | 62 | 69 | 0 / 12 | 0.00049 | +12.0 (+5.3 to +18.5) | effect | `s3-success_false_positive` |
| 0.5 on a 6 deg camera | 100 | 80 (80.0%) | 57 | 62 | 80 | 0 / 23 | 2.4e-07 | +23.0 (+14.4 to +31.3) | effect | `s3-success_false_positive` |
| 1 on a 6 deg camera | 100 | 100 (100.0%) | 57 | 62 | 100 | 0 / 43 | 2.3e-13 | +43.0 (+33.0 to +52.8) | effect | `s3-success_false_positive` |

#### camera_pose_offset: points added in stage 3

| camera_pose_offset | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 deg scene tilt down | 100 | 97 (97.0%) | 98 | 98 | 97 | 3 / 2 | 1 | -1.0 (-6.7 to +4.4) | small_cause | `s3-camera_pose_offset-tilt` |
| 4 deg scene tilt down | 300 | 264 (88.0%) | 296 | 275 | 264 | 36 / 4 | 1.9e-07 | -10.7 (-14.9 to -6.8) | effect | `s3-camera_pose_offset-300 + s3-camera_pose_offset-tilt` |
| 6 deg scene tilt down | 100 | 74 (74.0%) | 98 | 77 | 74 | 26 / 2 | 3e-06 | -24.0 (-33.6 to -14.6) | effect | `s3-camera_pose_offset-tilt` |

#### software_release: points added in stage 3

| software_release | Episodes | Placed | Twins | LIBERO's check | Logged | Broken / rescued | McNemar p | Difference, points (95% interval) | Role | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| release with a 14-step delay and 10 actions per chunk | 100 | 8 (8.0%) | 98 | 19 | 8 | 90 / 0 | 1.6e-27 | -90.0 (-94.2 to -81.4) | effect | `s3-software_release-two` |

**The interaction.** Each pair measured jointly on 300 `libero_spatial`
episodes (the exploratory one on 100), each single on the same episodes,
paired per episode (method in 8.4); the last column is the interaction's
role:

| Pair | Suite | Episodes | Healthy | A alone | B alone | Together | Effect A | Effect B | Joint effect | Interaction (95% interval) | Role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wear+heavier | libero_spatial | 300 | 296 | 281 | 258 | 234 | -5.0 (-8.0 to -2.7) | -12.7 (-17.0 to -8.7) | -20.7 (-25.3 to -16.0) | -3.0 (-7.0 to +1.0) | additive |
| weaker+heavier | libero_spatial | 100 | 98 | 92 | 86 | 77 | -6.0 (-12.0 to -1.0) | -12.0 (-20.0 to -5.0) | -21.0 (-30.0 to -13.0) | -3.0 (-9.0 to +3.0) | additive |
| delay+short-chunks | libero_spatial | 300 | 296 | 277 | 273 | 27 | -6.3 (-9.7 to -3.3) | -7.7 (-11.3 to -4.3) | -89.7 (-93.0 to -86.0) | -75.7 (-80.7 to -70.3) | interaction |

curves -> /home/cs/code/robots-faults3/src/fleetlab/faults/curves.json

The pair planned for the worlds, `gripper_wear` 0.5 with four mass doublings,
is additive: the two together cost 20.7 points, the sum of 5.0 and 12.7 alone,
within its interval (interaction -3.0, -7.0 to +1.0). The world generator's
supplement measured the same pair at 100 `w91` episodes (76 of 100, -22,
consistent with -20.7 here); against singles taken from this file's
100-episode curves (-3 and -12) that reads like an interaction of about -7
points, but with the singles measured on the same episodes (-5.0 and -12.7
on 300) it is within noise of additive. The exploratory pair (a weak
gripper, 0.25, with the same batch) is additive too. The pair the curves suggested, the 14-step delay with 10
actions per chunk, is the one to plant as the worlds' interaction: on the
same 300 episodes the delay alone costs 6.3 points (interval 3.3 to 9.7),
10 actions per chunk alone 7.7 (4.3 to 11.3), and the two together 89.7
(86.0 to 93.0): 27 of 300 placed against 296 healthy, an interaction of
-75.7 points (-80.7 to -70.3), far beyond the sum. Each alone is a small
cause a fleet would hardly notice; together they stop the robot. The
release that carries both (section 8.5, 100 episodes) is the joint chain
for chain (100 of 100 episodes identical), so the pair can be planted
either as two configuration changes (at a site, perhaps on different days:
a confounder-like history) or inside one release. The joint ran from the
branch after its merge with `main` (`594f025`), the release from `2e2c0b9`:
the identical chains also show that the merge changed no episode.

**The drift.** Two new families serve a drift: the scene camera tilting
down (2, 4, 6 degrees: -1, -10.7, -24) and a dirty scene lens (film 0.25,
0.5, 0.75, 1: +1, +1, -3, -10.7). These are points on the episodes of the robot
that carries the fault. A mount creeping 0.6 degrees a day from day 0
passes 2 degrees on day 3.3, 4 on day 6.7 and 6 on day 10: on that robot
it costs 0 to about 9 points in the observed window (days 1 to 6, 3.6
degrees by day 6, interpolated) and 20 to 24 in the scoring window (days 9
and 10). The experimental design's section-5
rule is in points of fleet success: on one robot of twelve, the same drift
costs the fleet about 0.7 points by day 6, 1.7 on day 9 and 2.0 on day 10,
so it reaches the detectability line only in the scoring window, the
"counts by the scoring window" case of that section, and only at the edge;
a fleet-wide or site-wide drift (a software or condition drift) or a
steeper creep is needed for more. The lens on one robot (-10.7 at a full film)
never reaches two fleet points: a drift below the line. The world generator
as of `stage3/worlds` `2f1c613` drifts the scene camera's pan (stage 2's
family, 3 to 9 degrees), not the tilt; either family works the same way.

### 8.6 Not verified, and what is left

- **The stage-3 types on `libero_object`.** Not measured. The supplement
  shows that suite reacting much more strongly to image, frame and timing
  faults, so `camera_latency`, `lens_contamination`, `image_pipeline_release`,
  `table_height_change`, `controller_overload` and the delay-with-short-chunks
  pair are likely stronger there, and the gripper types likely weaker (thick
  objects); a world that plants them on `libero_object` tasks applies the
  `libero_spatial` numbers and should say so. The optional runs
  `s3-object-*` of the plan exist for the gripper and batch types.
- **One hundred episodes per point** for the new types: a level reads as an
  effect from about 7 points of loss. Five effects sat at p 0.02 to 0.05 on
  100 pairs (`lens_contamination` film 1, p 0.023; the tilt of 4 degrees,
  the 10% crop, the table 2 cm higher and `controller_overload` at 0.1, p
  0.04 each); with about sixty tests at 0.05, two or three false effects are
  expected among all points. They were extended to 300 pairs and all five
  held, at p 1e-4 or below: -10.7, -10.7, -7.0, -9.7 and -8.3 points.
  The flat levels are small causes, not decoys, except warm lamps at 1,000
  pairs.
- **The decoys are confirmed on `libero_spatial` only.** On `libero_object`
  the supplement has lighting 2 stops dimmer and the backdrop flat at 100
  pairs, not confirmed; warm lamps were not run there. A world whose decoy
  site runs both suites should keep the decoy's expected effect to the
  spatial share or have the decoy levels run at 1,000 pairs on
  `libero_object` (open).
- **The worlds' interaction and inert entries are not yet this library's.**
  As of `stage3/worlds` `2f1c613` the generator plants worn pads with a
  heavier batch as its interaction, which is additive here; reads joint
  points only from its own supplement (with roles "effect" and "assumed",
  which curves.json's joints now also write as `role`, the interaction's
  own classification as `interaction_role`); multiplies a release's carried
  ratios (for the two-fault release that predicts about -10 points against
  the measured -90); and uses `policy.sampling_steps` 10 to 5 (flat at 100
  pairs, +2, not confirmed) as the confounder's inert entry, where
  `calibration_update` is inert by construction. All open for the world
  generator.
- **The false-positive detector's twins are not healthy**: its points
  compare the logged outcome with the same episodes' true outcome on top of
  a 6-degree camera (`companions` and `compared_with` in curves.json); a
  reader that takes `healthy_successes` as healthy twins gets a ratio above
  one. Its effect in a world is level x eligible share x failures.
- **The pair's singles from stage 2.** The 14-step delay and 10 actions per
  chunk on episodes 0 to 99 are stage 2's runs, made under the old lock,
  and were not run again; their comparability rests on the healthy run and
  the two re-run points being stage 2's chain for chain (300 and 200 of 200)
  and on neither type's code having changed since.
- **Levels between the calibrated points** are not measured; the knees of
  8.5 lie between them.
- **The films** (two episodes per new type, [faults-films](faults-films/index.md)) are chosen by
  index, not by outcome, so some show a faulted episode that still succeeds,
  and two filmed levels are flat ones: `camera_latency` at 12 steps (+1)
  and `arm_torque_loss` at 0.2 (-4); they show the mechanism, not a failure.
- **The health model**: `arm_torque_loss` and `controller_overload` move no
  health metric, by the frozen design; whether they should is an open
  question. The headlight is not in the engine's fixture hash
  (an open item for the engine).
- **`gripper_force_loss` and `gripper_wear` overlap in mechanism**: both end
  with fingers that stop short of a thin rim (a stall against the fingers'
  own friction, or a shortened range). They differ in the closing speed, the
  force limit on thick objects and the health reading only in degree; an
  analysis that blames the gripper is right for both, and the fixing verbs
  are the same.

## 9. The fault-by-fault table

All twenty-three calibrated types (the two generic kinds of an added change
act through these and have none of their own). Effects are the placed check's
paired difference to healthy twins in points, `libero_spatial`, 100 episodes
per level unless stated (300 for the decoy candidates' other levels, the
three interaction pairs and the five edge effects extended, 1,000 for the
confirmed decoy levels), from `curves.json`;
the `libero_object` column is the world generator's supplement (`w91`, 100
episodes per level, section 7). A data type's effect is on the logged
outcome. Films of every new type: [faults-films/index.md](faults-films/index.md),
two episodes each beside their healthy twins as strips of five frames,
built from run `s3-films` with `python -m fleetlab.dose_response_stage3
films docs/design/faults-films`; the videos themselves (8 MB of MP4) stay in
the run directory, which the page names.

| Kind | Class | Verdict | Mechanism | Levels with an effect (points) | Decoy and small-cause levels (points) | On libero_object (supplement) | Fixed by | Realism, with a source |
|---|---|---|---|---|---|---|---|---|
| `camera_pose_offset` | sensor | kept | a camera turned or shifted on its mount (cell hook on the camera's pose); tilted down by a creeping mount (stage 3) | 3 deg scene pan -12; 6 deg scene pan -41; 9 deg scene pan -63; 4 deg scene tilt down -11; 6 deg scene tilt down -24 | 5 deg wrist pan -1; 10 deg wrist pan -6; 2 deg scene tilt down -1 | 3 deg scene pan -44; 6 deg scene pan -71; 9 deg scene pan -97 | recalibrate, replace | Amazon's stowing robot: failures in its simplest motions "primarily stem from calibration errors and uncertainty in the grasped item pose and shape" ([Stow][stow]); DROID's calibrations "are not always accurate" ([DROID][droid]); camera pose breaks LIBERO policies ([LIBERO-Plus][liberoplus]) |
| `camera_degradation` | sensor | kept | blur, noise or wrong exposure on one camera's image (observation hook) | 4 px blur -25 | 1 px blur +0; 2 px blur +0; 10 noise -2; 25 noise +0; 1 stops darker +0; 2 stops darker -2; 1 stop brighter -2 | 2 px blur +1; 4 px blur -55 | clean, replace | "the most common failed sensor is the camera" ([Carlson et al.][carlson]); image noise costs LIBERO policies 9 to 57 points ([LIBERO-Plus][liberoplus]) |
| `lens_contamination` | sensor | kept | a film of dirt on the lens: blur and haze, densest in a smear (observation hook); the drift type, dust settling day by day | 1 scene film -11 | 0.25 scene film +1; 0.5 scene film +1; 0.75 scene film -3; 0.5 wrist film +1; 1 wrist film +0 | not measured | clean, replace | the camera as the most failed sensor ([Carlson et al.][carlson]); no source on dirty lenses in the background research |
| `camera_latency` | sensor | kept | frames `level` steps old while the joints are current (observation hook with a frame buffer) | 24 steps scene lag -44; 36 steps scene lag -39 | 6 steps scene lag +1; 12 steps scene lag +1; 12 steps wrist lag -2 | not measured | replace, service | a stale third-person view takes pi0.5 from 97.2 to 21.4 ([LIBERO-VPro][vpro]); connectors as the major cause of faults ([RoboCup survey][steinbauer]) |
| `action_latency` | configuration | kept | commands executed `level` steps late, a configured buffer (action hook) | 17 steps -45; 20 steps -75 | 8 steps -2; 14 steps -6 | 8 steps +0; 14 steps -35; 17 steps -93; 20 steps -98 | revert, set | OpenVLA loses 13 points when its loop runs slower ([OpenVLA][openvla]) |
| `policy_knob` | configuration | kept | sampling steps, actions per chunk or image views changed in the configuration (policy input hook) | 1 image view -85 | 5 sampling steps +2; 2 sampling steps +2; 1 sampling step +2; 10 actions per chunk -4; 3 image views -3; 2 image views -2 | 5 sampling steps -4 | revert, set | SmolVLA 82.8% at 10 actions per chunk, 51.8% at 50 ([SmolVLA][smolvla]); a missing processing step gives a silent 0% ([lerobot 3674][lr3674]) |
| `control_mode_change` | configuration | kept | the controller switched to relative control (policy input hook) | relative -98 | none | relative -98 | revert, set | X-VLA scored 0% until control was set to absolute ([lerobot 3401][lr3401]) |
| `calibration_update` | configuration | inert | a camera's believed pose rewritten in the configuration; no mechanism (inert) | none | scene calibration rewritten (inert, identical) | not measured | revert, set | camera calibrations "are not always accurate" ([DROID][droid]), so their records get rewritten; most outages follow a change ([Google SRE][sre-book]), which is what a confounder imitates |
| `software_release` | software | kept | a release that carries other faults; only `software.version` in the change log | release with a 17-step delay -45; release with a 14-step delay and 10 actions per chunk -90 | none | not measured | revert, set | Waymo's recall traced to one software release ([NHTSA][waymo]); 79% of medical software recalls came from changes ([FDA][fda]) |
| `image_pipeline_release` | software | kept | a release whose image preparation swaps red and blue, crops or compresses (observation hook, both cameras) | 10% cropped -7; 20% cropped -27 | red and blue swapped -1; jpeg quality 30 +1; jpeg quality 10 +0 | not measured | revert, set | a processing step missing from a checkpoint's pipeline: "0% success and the failure is silent" ([lerobot 3674][lr3674]); a pinned library overwritten ([openvla-oft 135][oft135]) |
| `gripper_wear` | physical | kept | worn pads lose friction and the fingers close short (cell hook) | 0.75 wear -11; 1 wear -21 | 0.25 wear -1; 0.5 wear -3 | 0.25 wear +2; 0.5 wear +2; 0.75 wear +2; 1 wear +1 | replace, service | finger tape "wears out over time" ([ALOHA 2][aloha2]); pick success 94.4% in Amazon's Robin evaluation ([Robin][robin]) |
| `gripper_force_loss` | physical | kept | a weaker gripper drive: softer finger servos, lower force limit (cell hook) | 0.5 force lost -56; 0.75 force lost -98 | 0.25 force lost -6 | not measured | replace, service | pick success of 94.4% in Amazon's Robin evaluation ([Robin][robin]); no source on gripper drive force in the background research |
| `gripper_sticky_release` | physical | kept | the fingers open `level` steps after the command to open (action hook) | 20 steps late open -61; 40 steps late open -98 | 5 steps late open -2; 10 steps late open -3 | not measured | clean, replace, service | thin or deformable items "dropped unnoticed" ([Stow][stow]) is the nearest; our own success spike; no direct source |
| `arm_torque_loss` | physical | kept | the arm's drives deliver less torque than commanded; the arm sags (cell hook) | 0.3 torque lost -13; 0.4 torque lost -42 | 0.1 torque lost +0; 0.2 torque lost -4 | not measured | replace, service | no source on drive derating in the background research |
| `controller_overload` | physical | kept | the controller misses cycles; commands back up, the delay grows (action hook) | 0.1 cycles missed -8; 0.2 cycles missed -42 | 0.02 cycles missed +1; 0.05 cycles missed +1 | not measured | replace, service | "the control system was the most common source of failures (32%)" ([Carlson et al.][carlson]) |
| `object_batch_variant` | physical | kept | a batch of heavier or slipperier objects at a site (cell hook) | 4 mass doublings -12; 6 mass doublings -18 | 2 mass doublings -3; 2 friction halvings -2; contact priority only (control) -1 | 2 mass doublings +1; 4 mass doublings -2; 6 mass doublings -3 | change_condition | physics variation takes pi0.5 from 82.7 to 31.6 ([RoboTwin-Phys][rtphys]); thin and deformable items cause failures ([Stow][stow]) |
| `furniture_shift` | site | kept | the table and everything on it shifted (cell, observation and action hooks) | 4 cm sideways -39 | 1 cm sideways +0; 2 cm sideways -2 | 2 cm sideways -34; 4 cm sideways -68 | change_condition | "variation in the arm setup among workcells" ([Robin][robin]); object layout breaks LIBERO policies ([LIBERO-PRO][liberopro]) |
| `table_height_change` | site | kept | the table lower or higher (the same hooks, vertical) | 2 cm lower -12; 4 cm lower -59; 2 cm higher -10 | 1 cm lower -1 | not measured | change_condition | "variation in the arm setup among workcells" ([Robin][robin]) |
| `illumination_change` | site | decoy | the site's lights dimmer or brighter (cell hook) | none | 1 stops dimmer -2; 2 stops dimmer +1.2 (decoy, 1000 pairs); 1 stop brighter -1 | 1 stops dimmer -1; 2 stops dimmer -2 | change_condition | light barely affects policies with a wrist camera ([LIBERO-Plus][liberoplus]); "Maintaining the same visibility across the robot fleet is also challenging" ([TRI][tri]) |
| `light_color_shift` | site | decoy | the lamps replaced by warmer or cooler ones (cell hook) | none | 0.5 warmer -2; 1 warmer -0.4 (decoy, 1000 pairs); 1 cooler -1 | not measured | change_condition | visibility across a fleet ([TRI][tri]); reflective surfaces leave colour unreliable ([Eppner et al.][eppner]) |
| `backdrop_change` | site | decoy | walls, floor or table top repainted (cell hook on textures) | none | 0.5 walls and floor -1; 1 walls and floor +1.1 (decoy, 1000 pairs); 1 table top -1 | 1 walls and floor +0 | change_condition | pi0 at 0% on an unseen background ([Penn study][penn]); background barely affects policies with a wrist camera ([LIBERO-Plus][liberoplus]) |
| `success_detector_error` | data | kept | the detector misses a share of successes (data hook) | 0.2 missed -21; 0.4 missed -36 | 0.5 false successes +1 | 0.2 missed -15; 0.4 missed -41 | replace, service | "only about 50 percent of robot users have monitoring" ([review][juniper]); no source on detector errors |
| `success_false_positive` | data | kept | the detector logs a share of failed placements as successes (data hook, listening to the gripper) | 0.25 on a 6 deg camera +12; 0.5 on a 6 deg camera +23; 1 on a 6 deg camera +43 | none | not measured | replace, service | section 2's open false success; no source on detector errors in the background research |

[stow]: https://arxiv.org/html/2505.04572v1
[droid]: https://arxiv.org/html/2403.12945
[robin]: https://arxiv.org/html/2305.10272
[aloha2]: https://arxiv.org/html/2405.02292
[eppner]: https://www.roboticsproceedings.org/rss12/p36.pdf
[penn]: https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/
[carlson]: http://nelsonrobotics.org/paper_archive_nelson/carlson-icra-2004.pdf
[steinbauer]: https://link.springer.com/content/pdf/10.1007/978-3-642-39250-4_31.pdf
[openvla]: https://arxiv.org/html/2406.09246
[tri]: https://arxiv.org/html/2507.05331v1
[waymo]: https://static.nhtsa.gov/odi/rcl/2025/RCLRPT-25E084-7732.pdf
[fda]: https://www.fda.gov/media/73141/download
[juniper]: https://juniperpublishers.com/raej/RAEJ.MS.ID.555624.php
[smolvla]: https://arxiv.org/html/2506.01844
[lr3401]: https://github.com/huggingface/lerobot/issues/3401
[lr3674]: https://github.com/huggingface/lerobot/issues/3674
[oft135]: https://github.com/moojink/openvla-oft/issues/135
[sre-book]: https://sre.google/sre-book/introduction/
[liberoplus]: https://arxiv.org/abs/2510.13626
[liberopro]: https://arxiv.org/abs/2510.03827
[vpro]: https://arxiv.org/abs/2609.24350
[rtphys]: https://arxiv.org/abs/2609.26292
