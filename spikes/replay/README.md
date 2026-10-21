# Does an episode replay exactly?

Measured on 2026-10-06 on this machine, in the probe environment
`~/.cache/robots-probes/libero-port` (MuJoCo 3.8.1, robosuite 1.5.2 with the
port patch, hf-libero 0.1.4, LeRobot 0.6.1, PyTorch 2.11.0 on ROCm 10.0.0,
policy `lerobot/smolvla_libero`, tasks from `libero_spatial` unless another
suite is named). Five other checks used the machine at the same time.

Words used throughout:

- **Identical** always means bit for bit: the stored numbers are the same
  bytes, not merely close. Recordings were compared through hashes or byte by
  byte, never with a tolerance.
- An **episode** is one attempt at a task: a LIBERO task, one of LIBERO's 50
  stored initial states for it, and up to 280 control steps of 50 ms.
- A **fixture** is a piece of furniture that cannot move during an episode
  (cabinet, stove, wine rack). LIBERO places fixtures at reset, at random
  within about 2 cm, and this placement is not part of its stored initial
  states.
- A **chunk** is what one call of the policy returns: 50 actions. The policy
  is called again only when they are used up, so a full episode has at most
  six calls.
- The **sampler noise** is the random starting point from which SmolVLA
  computes a chunk (ten refinement steps turn noise into actions). Different
  noise gives a different chunk for the same camera images.
- A **row** is one episode's place in a batch, when the policy computes chunks
  for several episodes in one call.

## Verdict

**Exact replay is possible**, with the policy running on the Radeon, and it
needs six rules. With them, 50 of 50 full closed-loop episodes repeated
identically in a fresh process and in a different order: every action, every
simulator state, every camera image, every outcome. Without them an identical
setup does not repeat: the same 50 episodes with nothing changed but how the
policy calls were batched flipped 5 of 50 outcomes between success and
failure.

The six rules. Each is backed by a measurement below, except rule 4, which
follows from reading LeRobot's loop:

1. **Pin the scene.** Seed numpy's global random generator immediately before
   every reset, after the simulator object exists, and on hard resets empty
   one list inside LIBERO first. The seed that LeRobot passes to `reset()` does
   not reach the fixture placement on robosuite 1.5.
2. **Give every episode its own noise.** Draw the sampler noise from a
   generator seeded with (episode noise seed, chunk number) and hand it to the
   policy. By default the noise comes from one global stream on the GPU and
   depends on call order.
3. **Keep batch size and row fixed.** On the Radeon a chunk depends on how many
   rows the batch has and on which row the episode sits in. It does not depend
   on what the other rows contain. So either call the policy for one episode
   at a time, or always use batches of one fixed size in which an episode
   keeps its row, filling unused rows with anything.
4. **One loop per episode.** Each episode has its own queue of pending actions
   and leaves the batch when it ends. LeRobot's loop keeps finished episodes
   running and resets them, which moves the random streams of all others.
5. **Save more than positions and velocities** when an episode is to be
   continued from the middle: the solver's warm start, the gripper
   controller's internal command, the fixture positions, and the observation
   that was delivered at that instant. LIBERO's own state restore is not exact.
6. **One rendering context at a time.** Either one environment per process, or
   switch the OpenGL context explicitly before every use of an environment.
   Otherwise images come out wrong without any error.

Hard resets are not needed for any of this (section 8), and neither is a
particular number format: 32-bit floats, bfloat16 and float16 all repeat
identically call for call (section 5).

The statistical fallback, if a rule cannot be kept: with fresh noise and
everything else equal, 11 of 50 outcomes flipped (22%). That is the noise
floor a statistical replacement would have to average away (section 7).

## The measurements

### 1. Simulator alone: yes

Same task, same initial state, same 280 actions. The actions were a fixed
script that grasps a bowl, lifts and drops it and then sweeps the closed
gripper through all other objects (51 to 100 contact points per step, all five
objects moved by 6 to 24 cm). Recorded per step: MuJoCo's complete integration
state, 441 numbers.

| repeated | result |
|---|---|
| in the same process after a hard reset, 3 times | identical |
| in the same process after a soft reset, 4 times | identical, and identical to the hard-reset runs |
| in a fresh process | identical |
| in 4 processes at the same time, 2 runs each | all 8 identical |
| 10 tasks from four suites (`libero_spatial` 3, 5, 8; `libero_10` 0, 3, 6, 9; `libero_goal` 2, 7; `libero_object` 4), each 2 runs in each of 2 simultaneous processes | 4 of 4 identical for every task |

Condition: the fixtures must be pinned (rule 1). Without that, two of the ten
tasks (`libero_10` 3 and 9, the ones with a drawer or door that starts open or
closed) gave a different state trace on the second reset than on the first,
in the same process and with the same seed. The cause is in "Defects found"
below.

MuJoCo is single-threaded here and no setting had to be changed for this.
Scripts: [sim_trace.py](sim_trace.py), [sim_compare.py](sim_compare.py),
[sim_sweep.sh](sim_sweep.sh), [scripted_actions.py](scripted_actions.py).

### 2. Rendering: yes, with two conditions

The same simulator state gave the same pixels in every comparison of section
1: 281 frames from each of the two cameras at 360x360, across processes, in
four parallel processes and across hard resets (which create a new rendering
context). Rendering ran on the Radeon through EGL (`radeonsi`, Mesa 26.2.2,
4x multisampling), not in software.

Condition one: the scene must be the same, which again means pinned fixtures.
With unpinned fixtures 8 to 12% of the pixel values of the main camera differ
between two resets with the same seed.

Condition two: the rendering context must be the right one (rule 6).
robosuite gives every environment its own OpenGL context and never switches
back to it. Measured with [glctx_check.py](glctx_check.py):

| situation | images |
|---|---|
| two different scenes alive in one process, stock code | the environment that was not built or hard-reset last renders garbage: 99% of pixels wrong, no error |
| the same task twice in one process, different fixture layouts, stock code | correct; presumably because both contexts then hold the same textures and meshes in the same order |
| two different scenes in one process, context made current before each reset and step | correct in every case |
| one environment closed, then another task built in the same process | correct |

So a process that rebuilds its one environment for a new task is safe. A
process that holds two environments is safe only with the explicit switch
(`gl_ctx.make_current()` on the environment's render context, one line, shown
in `common.make_current`).

Changing what is rendered on a live model, as a visual fault would (one
environment in the process): a new camera position, light colour, object
colour and material colour are all picked up by the next render and, when
undone, give back the identical image. Changed texture pixels are **not**
picked up until the texture is uploaded again (`mujoco.mjr_uploadTexture`);
with the upload they are, and undoing them restores the identical image. Model
changes survive a soft reset and are lost at a hard reset.

One more fact that matters for rule 5: the image a policy is given does not
show the end of the control step. robosuite samples its sensors after 24 of
the 25 physics substeps, so the delivered image and joint readings are 2 ms
old (measured: median age 2.00 ms). Rendering the saved end-of-step state
again gives a slightly different picture (3% of pixel values in a fast
motion). This is deterministic and harmless, as long as nobody re-renders in
place of the delivered image. Script: [obs_phase.py](obs_phase.py).

### 3. Save and restore: yes, if the right things are saved

A snapshot was taken after step 57 (bowl in the gripper), 110 (bowl carried)
and 185 (pushing objects) of the contact-rich script, the episode was played
to its end as the reference, then the snapshot was put back and the rest was
played again.

| restored into | everything saved | LIBERO's own `set_state` (positions and velocities) |
|---|---|---|
| the same simulator, later in the same process | identical, 3 of 3 | never identical |
| the same simulator after a hard reset with another seed (new model, moved fixtures, new controllers) | identical, 3 of 3 | never identical |
| a fresh process whose scene was reset with another seed | identical, 2 of 2 | never identical |

What has to be saved, found by leaving out one group at a time (6 trials per
group):

| group | needed? | what happens without it |
|---|---|---|
| time, positions, velocities, controls | yes | (the baseline) |
| solver warm start (`qacc_warmstart`, 43 numbers: where the contact solver starts its search) | **yes** | never identical; differences of 1e-14 to 1e-6 in the next step grow to between 1e-4 and 0.14 (metres, radians or their rates) by the end of the episode, 95 to 223 steps later; images differ in most frames |
| gripper controller: `goal_qvel` and the gripper model's `current_action` (the gripper integrates its open/close command) | **yes** | the gripper jumps and the episode diverges from the next step, in 4 of 6 trials; in the other two the stale value happened to be the right one |
| model: fixture positions and orientations (`body_pos`, `body_quat`) | **yes** when the scene was reset differently | physics unchanged in these trials, the main camera's image differs in 12% of values |
| the observation delivered at that instant | **yes** for a policy | a re-rendered image differs, see section 2 |
| arm controller (goal pose) | no | recomputed from the current pose at every action in relative ("delta") control |
| composite controller, robot's recent-value buffers, robosuite's step counter and clock, LIBERO's object-state trackers | no | no effect in any trial |
| per-sensor sampling clocks | no effect in any trial | keep them anyway: they decide which substep a sensor is read at, and LIBERO's `set_state` advances them by one substep |

The lean set (MuJoCo's integration state with warm start, gripper state,
fixture poses, delivered observation) was identical in 6 of 6 trials. It is
about 4 kB plus the two camera images (0.78 MB uncompressed). Putting a
snapshot back takes 0.6 ms. Taking the diagnostic snapshot of this spike takes
159 ms, nearly all of it for copying the whole model; a lean one would not.

On the policy side this loop has only two things to save: the actions still
queued from the current chunk and the number of the next chunk (it selects the
noise). With the policy in the loop on the Radeon, 5 of 5 episodes continued
from a snapshot (after step 37 or 130, with two to five later policy calls)
were identical to the uninterrupted episode, after the simulator had been
reset to a different episode and moved in between.

Wrapper counters: LeRobot's wrapper advances the index of the next initial
state at every reset, and the 280-step limit is counted by the evaluation
loop, not by the environment. The engine has to own both.

Scripts: [snapshot.py](snapshot.py), [save_restore.py](save_restore.py),
[closed_loop_restore.py](closed_loop_restore.py).

### 4. Policy alone: yes

Sixteen recorded observations (eight tasks, two instants each), each with its
own fixed noise, on the Radeon:

- The same observation and noise three more times in a row: identical, in
  32-bit floats, bfloat16 and float16.
- The same 16 single calls and a batch of four in a fresh process: 16 of 16
  and 4 of 4 identical, in all three number formats.
- On the CPU: identical across two fresh processes, and identical between 4
  and 8 threads.

Where the noise comes from: `lerobot/policies/common/flow_matching.py`,
`sample_noise`, calls `torch.normal(..., device=device)` once per policy call
for the whole batch, shape (batch, 50, 32). That draws from PyTorch's global
generator on the GPU, which `lerobot-eval` seeds once at program start.
Measured: the same seed and shape give the same noise, also in a fresh
process, but the next call gives different noise. So an episode's noise
depends on how many policy calls happened before and on its row.

How to give each episode its own stream: `select_action` and
`predict_action_chunk` accept a `noise` argument of shape (batch, 50, 32).
`policy_lib.Policy.noise` draws each row on the CPU from a generator seeded
with (noise seed, chunk number). This is what all closed-loop runs below used.

Scripts: [policy_lib.py](policy_lib.py), [policy_probe.py](policy_probe.py),
[compare_probe.py](compare_probe.py), [make_obs.py](make_obs.py).

### 5. Batch composition: no, except in one respect

On the Radeon, the same in 32-bit floats, bfloat16 and float16:

| what changes | a row's output |
|---|---|
| the other rows' contents, batch size and row kept (batches of 4, 8 and 10) | identical, 12 of 12 in each format |
| the row's position, batch size kept | not identical, 0 of 24 in each format |
| the batch size (1 against 2, 3, 4, 8, 10) | not identical, 0 of 20 in each format |

The differences are not small: for 32-bit floats a batch of four against the
row alone gives a median difference of 0.003 and a largest of 0.05, where a
typical output value is 0.76 (the policy's normalised action units). That is
as large as the difference between 32-bit floats and bfloat16 (median 0.003),
or between the Radeon and the CPU (median 0.003). The network amplifies
rounding differences that far.

On the CPU the row position did not matter in the few cases tried, only the
batch size did.

Inference from these facts, tested in closed loop in section 6: batching is
compatible with exact replay if the batch always has the same number of rows
and an episode always sits in the same row.

### 6. Closed loop: yes

Full episodes with SmolVLA on the Radeon in 32-bit floats, rules 1 to 4 in
force, every episode's policy calls made alone (batch of one). Compared:
every action, the simulator state after every step, both images of every
step, the outcome.

| comparison | episodes identical |
|---|---|
| 6 episodes twice in one process | 6 of 6 |
| the same 6 in another process, with other environments beside them and a second policy process running on the GPU at the same time | 6 of 6 |
| 50 episodes (10 tasks, 5 initial states) in a fresh process, in reverse order | 50 of 50 |
| 50 episodes with inference on the CPU, fresh process, reverse order | 50 of 50 |

The second run of the 50 also differed from the first in two ways that
turned out not to matter: it switched the rendering context explicitly, and
no other policy process ran beside it. The first run held five environments
in one process without the switch (rule 6 was found later); its images were
right because the five were always the same task.

Fixed rows, the batching that rule 3 allows (20 episodes of tasks 0 to 3,
batches of exactly five rows, the row given by the episode's initial-state
index, unused rows filled with a copy of another row):

| comparison | episodes identical |
|---|---|
| batch mates from the same task, against batch mates from four different tasks in another order | 20 of 20 |
| 5 of the episodes each alone in its row with four filler rows, against both runs above | 5 of 5 |
| the same 20 with every policy call made alone (a batch of one) | 0 of 20, 2 outcomes flipped |

So a batch of fixed size with fixed rows is exact, and it is a different
episode from the one a batch of one produces. The batching regime belongs to
the episode's identity.

Scripts: [engine.py](engine.py), [closed_loop.py](closed_loop.py),
[compare_runs.py](compare_runs.py), [gpu_jobs.sh](gpu_jobs.sh).

### 7. When it is not exact: how often the outcome flips

Each line is 50 paired episodes (10 tasks, 5 initial states) that differ in
exactly one thing. A flip is success in one run and failure in the other.

| the one thing that differs | identical episodes | outcome flips | paths more than 1 cm apart |
|---|---|---|---|
| nothing (fresh process, reverse order), Radeon | 50 of 50 | 0 | 0 |
| nothing (fresh process, reverse order), CPU | 50 of 50 | 0 | 0 |
| how policy calls are batched: alone, or five episodes of a task in one batch that shrinks as episodes end; Radeon | 0 of 50 | 5 (10%) | 23 of 50, typically from step 82 |
| the same, CPU | 0 of 50 | 5 (10%) | 22 of 50, typically from step 142 |
| the device: Radeon or CPU | 0 of 50 | 1 (2%) | 26 of 50, typically from step 107 |
| the sampler noise (CPU) | 0 of 50 | 11 (22%) | 44 of 50, typically from step 53 |

How fast paths diverge: with a different batch layout the very first chunk
already differs, the gripper paths are more than 1 mm apart in 49 of 50
episodes (typically from step 38) and more than 1 cm apart in 23 of 50. The
simulator itself is just as sensitive: a wrong starting point for the contact
solver, worth 1e-14 to 1e-6 in the next step, grew to between 1e-4 and 0.14
(metres, radians or their rates) within 95 to 223 steps of the contact-rich
script (section 3).

These are single samples of 50. A count of 5 in 50 is compatible with a true
rate of roughly 3 to 22%, 11 in 50 with 12 to 36%, 1 in 50 with up to 11%.
What they do show: there is no "nearly exact". Any deviation at the level of
rounding moves a noticeable share of outcomes, and fresh noise moves about
twice as many. Success counts stayed between 27 and 32 of 50 throughout, so
the average hides the flips.

### 8. Resets and what ran before in the process

Does exact replay need hard resets (which rebuild model and renderer, and by
the throughput check cost about 1 s per episode)? No. On ten tasks from four
suites the contact-rich script gave the identical state trace, image stream
and fixture layout in all four combinations: hard or soft reset, directly
after start or directly after an unrelated episode (another initial state and
scene seed, 120 steps of other actions, gripper left closed, objects
displaced). All of that with the reset of rule 1.

Does an episode's start depend on what the process built before? With the
stock reset, yes; with the reset of rule 1, no. `libero_spatial` task 0,
initial state 0, seed 1000, either as the first thing in a process or after
another environment had been built, reset and closed:

| reset | fixture layout | first image | positions and velocities after reset |
|---|---|---|---|
| stock (numpy seeded once at program start, then `reset(seed)`) | differs | differs | identical for this task |
| rule 1 (`engine.reset_episode`) | identical | identical | identical |

"Identical for this task" is luck: nothing rests on a fixture in task 0. Over
all ten `libero_spatial` tasks, the simulator state right after reset
differed between two fixture layouts in five (tasks 3, 4, 6, 7, 9), by 1e-17
to 6e-10. That is small, and section 3 shows what differences of that size
grow into.

Scripts: [soft_reset_check.py](soft_reset_check.py),
[init_state_check.py](init_state_check.py).

## What the rollout engine has to do

Measured facts are above; this section is my recommendation.

**Identity of an episode.** An episode is (suite, task, initial-state index,
scene seed, noise seed, fault set, software versions, batch size and row).
Nothing else may influence it, in particular not which worker ran it, what
ran before it in that process, or what ran beside it.

**Seeding.**

- Scene: use `engine.reset_episode`. It builds the simulator if needed,
  empties LIBERO's sampler list on a hard reset, seeds numpy's global
  generator with the scene seed and resets. After the reset, store the fixture
  poses (`model.body_pos`, `model.body_quat`) in the episode record and
  compare them on every replay. Better still, let the engine set the fixture
  poses itself; then they can also become a controlled site condition.
- Noise: one generator per (noise seed, chunk number), drawn on the CPU,
  passed through the `noise` argument. Never rely on a global seed.
- Nothing else needs a seed. MuJoCo, the controllers and the policy contain no
  other randomness that these runs could detect.

**Batching.** This is the one place where the layout from the throughput check
(eight workers, one inference process that batches whichever requests are
waiting) and exact replay collide: a batch whose size varies from call to call
changes every episode's actions. Three ways out. The costs are calculated
from the throughput check's fit (65 ms per call plus 43 ms per row), not
measured, and which of the first two is cheaper for the fleet has to be
measured in the prototype.

1. *One episode per call.* The simplest rule, with the most evidence (50
   episodes on each device), and nothing to record. 108 ms per call; the
   prototype's calls then queue instead of merging.
2. *Fixed rows.* The inference process always computes a batch of exactly N
   rows. Worker k's request always goes into row k; rows with no request
   waiting are filled with any valid input and their output is dropped. The
   episode record stores N and the row, and a replay uses the same N and row.
   409 ms per call for eight rows, 237 ms for four, against about 130 ms at
   the prototype's mean batch of 1.5. N need not be the number of workers:
   with N = 4, two workers share a row and never go into the same call.
3. *Free batching for the fleet, exact pairs for the ground truth.* The fleet
   runs at full speed and its episodes are not reproducible. Ground truth for
   a fault then never compares the original episode with a replay. It runs
   the episode twice under way 1 or 2, once with the fault and once without,
   and compares those two. The original outcome and the re-run with the fault
   still in will disagree in roughly one episode in ten (5 of 50 measured).

I would start with way 1 and move to way 2 only if the prototype shows that
the queue costs real throughput.

**Resets.** Soft resets are enough (section 8). Two cautions. A
fault that changed the model (camera pose, light, colours, friction, masses)
survives a soft reset and has to be undone by the engine; a hard reset wipes
it. And a changed texture needs an upload both when applied and when undone.

**Processes and rendering.** One environment per worker process, as measured
by the throughput check, needs no extra care, also when the worker moves to a
new task. Any process with two environments must make the environment's
context current before every reset, step and close.

**Saving state.** For "continue from here" save the lean set of section 3 plus
the sensor clocks, the queued actions, the chunk number and the step count.
Restore by writing them back; do not call LIBERO's `set_state` or
`regenerate_obs_from_state`, and hand the policy the saved observation
instead of rendering a new one. Both branches of a comparison ("the policy
carries on" against "someone takes over") start from the same restored
snapshot.

**Loop.** One action queue per episode; an episode that ends leaves the loop
and its worker starts the next one; no automatic reset. The engine counts the
steps.

**Guard rails.** Store a hash of state, action and images per step with every
episode. A replay with nothing removed must reproduce the hash chain; run a
few such replays with every campaign, because a driver, library or kernel
update can end exactness without any error. In every process that uses both,
import `torch` before `mujoco` (see below).

## Defects found on the way

1. **The episode seed does not reach the scene** (LIBERO port on robosuite
   1.5). robosuite 1.5 stores the seed in an attribute `env.seed`, which hides
   LIBERO's method `seed()` that used to seed numpy. The port's wrapper then
   seeds a generator that LIBERO's placement code does not use. Fixture
   positions therefore come from numpy's global stream, which `lerobot-eval`
   seeds once per program run. Measured with LeRobot's own environment
   factory: the episode with seed 1001 got its cabinet at (0.0336, -0.2692)
   with batch size 2 and at (0.0385, -0.2788) with batch size 1. A whole run
   repeats; a single episode taken out of it does not. For the same reason an
   environment's first image depends on what was built earlier in the process
   (the throughput check saw this as 3% of pixels; section 8). Filed with
   defect 2 as follow-up work.
2. **LIBERO's reset is not repeatable for tasks with drawers or doors.**
   `object_property_initializers` gains one more sampler at every hard reset
   and is never emptied; each draws one number before the fixtures are placed.
   The scene therefore depends on how many resets the environment object has
   seen. Upstream LIBERO, not the port.
3. **LIBERO's `set_state` is not a restore.** It leaves the warm start and the
   gripper's command as they were and advances the sensor clocks.
4. **robosuite never switches the rendering context** (section 2).
5. **Import order.** In a process that imports `mujoco` before `torch`,
   copying the policy's weights to the Radeon failed with `CUDA error: invalid
   argument` in 15 of 15 attempts; with `torch` first it never failed,
   including a direct comparison of the two orders
   ([import_order.py](import_order.py)). Cause not investigated.
   `lerobot-eval` happens to import `torch` first.

## Not verified

- bfloat16 and float16 in closed loop. They were checked call for call only.
- Closed loop with the policy on other suites than `libero_spatial`, and any
  other policy.
- The eight-worker prototype itself. All closed-loop runs used the
  single-process loop in `engine.py`; the fixed-row rule was tried there with
  batches of five, not with an inference process that receives requests at
  arbitrary times, and not with eight rows.
- Soft resets with the policy in the loop (simulator only, ten tasks).
- Faults. Nothing here injected one; how each fault type interacts with
  snapshots and soft resets is open, beyond the visual parameters in section 2.
- Absolute control mode, where the arm controller's goal would have to be
  saved.
- Stability over time: hundreds of resets in one process, a reboot, a driver
  or library update, a first run with an empty MIOpen cache.
- The flip rate with fresh noise on the Radeon (measured on the CPU), and
  tighter bounds on any flip rate: each is one sample of 50.
- What the batching options cost in throughput. The figures given are
  calculated from the throughput check's fit.
- Exactness while the GPU is short of memory or shared with other heavy jobs.
  One run had a second policy process of mine beside it; nothing heavier.
- Why batch size and row position change the output this much, and why
  `mujoco` before `torch` breaks the copy to the GPU.

## Files

Recordings and logs are in `~/.cache/robots-probes/replay/`. Scripts, all run
with the probe environment's Python from the repository root:

| script | what it does |
|---|---|
| `common.py`, `engine.py`, `policy_lib.py`, `snapshot.py` | shared: environment set-up, the rollout loop, the policy with explicit noise, state snapshots |
| `scripted_actions.py`, `sim_trace.py`, `sim_compare.py`, `sim_sweep.sh` | sections 1 and 2 |
| `glctx_check.py`, `obs_phase.py`, `init_state_check.py`, `soft_reset_check.py` | rendering contexts, age of observations, dependence on process history, soft against hard resets |
| `save_restore.py`, `closed_loop_restore.py` | section 3 |
| `make_obs.py`, `policy_probe.py`, `compare_probe.py` | sections 4 and 5 |
| `closed_loop.py`, `compare_runs.py`, `gpu_jobs.sh` | sections 6 and 7 |
| `import_order.py` | defect 5 |
