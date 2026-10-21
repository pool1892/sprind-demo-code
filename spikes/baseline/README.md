# A healthy baseline policy on LIBERO

Spike run on 2026-10-06. Question: a pretrained policy has to
succeed at least 85% of the time on LIBERO pick-and-place tasks before faults
are injected. SmolVLA reached about 60%. Why, and which setup is healthy?

Everything below was measured on this machine unless it says otherwise. What I
conclude from the measurements is marked as such.

## Verdict

**The low score had two causes, both in the evaluation setup. Neither is the
robosuite port, and neither is the third camera.**

1. **The checkpoint runs blind for 2.5 seconds at a time.** SmolVLA predicts the
   next 50 actions in one go (an "action chunk"). `lerobot/smolvla_libero` is
   stored with the setting to execute all 50 before it looks at the cameras
   again. Executing 10 and then predicting afresh raises `libero_spatial` from
   58 to 79 of 100 episodes on the same start states.
2. **MuJoCo 3.8.1 breaks the start of four tasks.** In `libero_spatial` task 5
   the bowl starts tilted on the rim of the ramekin instead of sitting in it.
   SmolVLA solved that task in 1 of 50 episodes on MuJoCo 3.8.1 and in 18 of 30
   on MuJoCo 3.3.7. Three `libero_10` tasks are affected in a different way
   (objects still in the air in the first camera image); that part is not in
   the upstream bug report.

**With both corrected, this SmolVLA checkpoint is borderline, not healthy:** 79
of 100 on `libero_spatial` (95% interval 70 to 86%) and 88 of 100 on
`libero_object` (80 to 93%).

**X-VLA clears the bar.** `lerobot/xvla-libero`, 20 episodes on each of the 10
tasks of the two pick-and-place suites, and 10 per task on `libero_goal`:

| Suite | Successes | Rate | 95% interval |
|---|---|---|---|
| `libero_spatial` | 192/200 | 96.0% | 92.3 to 98.0% |
| `libero_object` | 197/200 | 98.5% | 95.7 to 99.5% |
| `libero_goal` | 99/100 | 99.0% | 94.6 to 99.8% |

It costs about the same machine time as SmolVLA here: 6.7 s per episode on
`libero_spatial` and 5.7 s on `libero_object`, against 6.5 s and 5.2 s for
SmolVLA. Its model is
slower per call, but its episodes are shorter and it calls the model only every
30 steps. That is 540 to 630 episodes per hour in one process.

**Recommendation.** Use X-VLA as the fleet's nominal policy and the corrected
SmolVLA setup as a second, weaker "policy version". Pin MuJoCo below 3.4
(3.3.7 is what I measured). X-VLA needs one fix to the robosuite port first,
which my runner applies at run time. pi0.5 also runs here (9 of
10 on the one task tried) but is the heaviest and has a licence and a
tokenizer reservation; see "Policies compared".

## The setup that works

```sh
# from the repository root; one task per run, 20 episodes each, resumable
BASELINE_ABS_WORLD=1 BASELINE_EXPECT_GIB=8 MJ=337 \
  spikes/baseline/run_suite.sh g1_xvla_spatial libero_spatial \
  --policy.path=lerobot/xvla-libero --env.control_mode=absolute \
  --eval.n_episodes=20 --eval.batch_size=5
# same with g2_xvla_object libero_object
~/.cache/robots-probes/libero-port/bin/python spikes/baseline/summarize.py --suite g1_xvla_spatial g2_xvla_object
```

What that expands to, per task:

```sh
LIBERO_CONFIG_PATH=~/.cache/robots-probes/baseline/libero-config-mj337 \
MIOPEN_FIND_MODE=2 MUJOCO_GL=egl BASELINE_NO_VIDEO=1 BASELINE_ABS_WORLD=1 \
tools/heavy-run ~/.cache/robots-probes/baseline-venv-mj337/bin/python spikes/baseline/measured_eval.py \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --policy.path=lerobot/xvla-libero --policy.device=cuda \
  --env.control_mode=absolute \
  --eval.n_episodes=20 --eval.batch_size=5 --eval.use_async_envs=false \
  --seed=1000 --output_dir=~/.cache/robots-probes/baseline/runs/g1_xvla_spatial_t0
```

The parts that matter, and why:

- **`lerobot/xvla-libero`**, revision `12e8783e`, used exactly as stored:
  32-bit floats, 30 actions executed per chunk, 10 denoising steps. No renaming
  of camera inputs is needed.
- **`--env.control_mode=absolute`.** X-VLA commands target poses of the gripper,
  not small movements. LeRobot's X-VLA page gives this flag.
- **`BASELINE_ABS_WORLD=1`.** On the robosuite 1.5.2 port, absolute commands
  are read in the wrong coordinate frame and the arm flies off (see "Side
  findings"). This switch makes `measured_eval.py` correct the frame. Without
  it X-VLA cannot work on the port.
- **MuJoCo 3.3.7** (`MJ=337`): my copy of the shared `libero-port` environment
  with only that package changed.
- **Episode limit left at LeRobot's default**, 280 steps for these two suites.
  LeRobot's X-VLA page uses 800. X-VLA's successful episodes here took 73 to
  242 steps (median 106 on `libero_spatial`, 134 on `libero_object`), so the
  shorter limit costs little and bounds the run time.
- **Five environments at a time**, 20 episodes per task in four rounds. Five
  instead of ten keeps the GPU memory of the process at 7.2 GiB.
- **Start states.** LIBERO stores 50 start states per task. The first round
  uses states 0 to 4. Later rounds do not continue with 5 to 19: in this mode
  LeRobot resets an environment that finished early and lets it run on until
  the round ends, and each such reset moves that environment on by five start
  states (see "Side findings"). The 20 counted episodes of a task therefore
  start from 20 stored states spread over the 50, not from the first 20 (a
  state could repeat if the counter passes 50; with these episode lengths that
  is unlikely). Which
  ones depends on when episodes ended; the runner did not record them for
  these runs. It does now: in the pi0.5 run the second round started from
  states 15, 16, 7, 18 and 19, and in `libero_goal` task 0 from 10, 6, 12, 13
  and 14.
- **Seed 1000, and every task in its own process.** A task's result then does
  not depend on which tasks ran before it.
- `measured_eval.py` is LeRobot's `lerobot-eval` with timers added. It does not
  change what the policy or the simulator do, except for the two switches
  named above and for not rendering videos.

Per-task successes, out of 20 for the first two suites and out of 10 for
`libero_goal`:

| Suite | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `libero_spatial` | 20 | 20 | 20 | 20 | 16 | 18 | 20 | 19 | 19 | 20 |
| `libero_object` | 20 | 20 | 20 | 20 | 20 | 18 | 20 | 19 | 20 | 20 |
| `libero_goal` | 10 | 10 | 10 | 10 | 10 | 9 | 10 | 10 | 10 | 10 |

The weakest task, `libero_spatial` 4 ("bowl in the top drawer of the wooden
cabinet"), is at 16 of 20, which on its own has an interval of 58 to 92%. A
further ten episodes of that task for the video sample gave 7 of 10.

## Why SmolVLA scored about 60%

All runs in this section: `lerobot/smolvla_libero` (revision `31d453f7`),
`libero_spatial`, seed 1000, 10 environments at a time, 10 episodes per task on
start states 0 to 9, robosuite 1.5.2 port, inference on the Radeon. The run
names refer to the table under "Runs".

### Cause 1: 50 actions executed per chunk

`n_action_steps` is the number of actions executed from each predicted chunk
before the policy looks again. The checkpoint's `config.json` says 50, which is
the framework default and the whole chunk. At 20 control steps per second that
is 2.5 seconds without feedback.

| MuJoCo | `n_action_steps` | Successes | Per task (0 to 9) | Run |
|---|---|---|---|---|
| 3.8.1 | 50 (as stored) | 58/100 | 8 5 6 9 6 0 8 6 3 7 | `cmp_libero_spatial_libero-port` (an earlier run) |
| 3.8.1 | 10 | 79/100 | 10 10 9 9 8 0 10 7 9 7 | `f1_spatial_mj381_nas10` |

Same start states in both rows. 27 episodes were solved only with 10, 6 only
with 50 (`compare_runs.py`, exact two-sided p = 0.0003). The SmolVLA paper
evaluates in simulation with a new prediction after every single action, and
LeRobot's LIBERO page uses 10 for its pi0.5 reference numbers.

On task 5 alone, executing 1 action per chunk did not help either while the
start state was broken: 0/10 at 50, 1/10 at 10, 0/10 at 1 (`d01`, `d02`).
That pointed to a second cause.

### Cause 2: MuJoCo 3.8.1 changes how four tasks start

LIBERO's stored start states are not at rest. Some objects are stored above
or inside the surface they belong on, and the physics is left to settle them
during reset. In `libero_spatial` task 5 the bowl is stored 11 cm above the
ramekin and drops onto it. In the `libero_10` basket scenes the ketchup, the
orange juice and the milk are stored about 3 cm inside the floor. MuJoCo 3.3.7
lifts them out gently; MuJoCo 3.8.1 throws them half a metre into the air.

Measured without any policy (`probe_settle_all.py`, start state 0 of all 40
tasks of the four suites, robosuite 1.5.2 in both cases): the scene after reset
is identical to under 0.05 mm in 36 tasks and differs in four.

| Task | MuJoCo 3.3.7 | MuJoCo 3.8.1 |
|---|---|---|
| `libero_spatial` 5, "bowl on the ramekin" | bowl sits in the ramekin, 1.45 cm off its centre | bowl hangs on the rim, 3.7 cm off centre |
| `libero_10` 0, 1 and 7 (the basket scenes) | all objects at rest | ketchup, orange juice and milk are 0.4 to 0.55 m in the air in the first camera image and land within the next second |

The two distances match those in LeRobot issue 4390 (1.45 and 3.72 cm). The
`libero_10` effect is not in that issue. Pictures: `~/.cache/robots-probes/baseline/probe/sp5_mj381_vs_mj337_vs_dataset.png`
(MuJoCo 3.8.1, 3.3.7, and a frame from the training data) and
`.../probe/l10/t0_cmp.png`.

Effect on SmolVLA, task 5 only:

| MuJoCo | `n_action_steps` | Successes on task 5 | Runs |
|---|---|---|---|
| 3.8.1 | 50, 50, 10, 10, 1 | 0, 0, 1, 0, 0 of 10 each: 1/50 | baseline, `d01`, `d02_nas10`, `f1`, `d02_nas1` |
| 3.3.7 | 50, 10, 10 | 5, 8, 5 of 10 each: 18/30 | `d04_nas50`, `d04_nas10`, `f2` |

On the whole suite the correction did not show in the total:

| MuJoCo | `n_action_steps` | Successes | Per task (0 to 9) | Run |
|---|---|---|---|---|
| 3.8.1 | 10 | 79/100 | 10 10 9 9 8 0 10 7 9 7 | `f1_spatial_mj381_nas10` |
| 3.3.7 | 10 | 79/100 | 10 10 8 8 5 5 9 8 7 9 | `f2_spatial_mj337_nas10` |

Task 5 gained five episodes and the other nine tasks lost five between them.
With 10 episodes per task that is within chance (11 episodes solved only on
3.8.1, 11 only on 3.3.7), so these two runs neither show nor exclude a loss on
the other tasks. The same configuration also gave 8/10 and 5/10 on task 5 in
two runs, once alone and once as part of the suite, which changes the random
numbers the policy draws. One hundred episodes cannot resolve a difference of
five points.

### Not a cause: the third camera

The checkpoint's `config.json` lists three image inputs and a 6-number robot
state. Both are leftovers from the model it was fine-tuned from
(`lerobot/smolvla_base`, built for a different robot). This is read from the
checkpoint's files and LeRobot's code, and agrees with LeRobot issue 4517:

- The checkpoint's own `train_config.json` shows it was trained on
  `lerobot/libero`, which has two cameras, with exactly the mapping we used
  (`image` to `camera1`, `image2` to `camera2`).
- Its stored normalisation statistics have 8 state numbers, not 6.
- SmolVLA's code skips an image input that is absent (`empty_cameras` is 0), in
  training and in evaluation alike. So the model never saw a third image.

Measured as well: feeding the model a blank third image that is marked as
absent (`--policy.empty_cameras=1`) changed nothing on the 20 episodes tried.
Tasks 0 and 7 gave 10/10 and 8/10 with it (`d05`) and 10/10 and 8/10 without
(`f2`).

### Also checked and found in order

- **Image orientation and content.** The frame the policy receives matches a
  training frame of the same task in orientation, viewpoint and brightness
  (mean pixel value 119.9 against 118.3 for the main camera, 127.1 against
  126.8 for the wrist camera). `probe_init_state.py`, picture
  `.../probe/sp5_mj381/cmp.png`.
- **Robot state.** Same 8 numbers in the same order and units as in the
  training data (position, axis-angle orientation, two finger positions).
  Position agrees within 2 mm. On the robosuite 1.5.2 port the reported
  orientation is about 2 degrees off what robosuite 1.4.0 reports for the same
  start state; 1.4.0 matches the training data. Effect on success not measured.
- **Checkpoint against this LeRobot version.** Names and shapes of all 500
  weight tensors match the model LeRobot 0.6.1 builds (788 of 788 for
  `HuggingFaceVLA/smolvla_libero`). `check_checkpoint_keys.py`. X-VLA's file
  uses an older layout; LeRobot renames the entries on load and then loads
  strictly, which fails on any mismatch, and it did not fail.

## Policies compared

Rates are successes over episodes. "s/episode" is time inside the rollouts
divided by episodes, with 10 environments at a time unless noted.

| Policy | Setup | `libero_spatial` | `libero_object` | s/episode | Model call |
|---|---|---|---|---|---|
| X-VLA, `lerobot/xvla-libero`, 0.88 B parameters | as above, 5 environments | 192/200 | 197/200 | 6.7 and 5.7 | 2.1 s per 5 environments, every 30 steps |
| SmolVLA, `lerobot/smolvla_libero`, 0.45 B | MuJoCo 3.3.7, `n_action_steps=10` | 79/100 | 88/100 | 5.2 to 9.0 | 0.5 s per 10 environments, every 10 steps |
| SmolVLA, same | as stored (50), MuJoCo 3.8.1 | 58/100 | not run | 5.3 (with videos) | 0.57 s, every 50 steps |
| SmolVLA, `HuggingFaceVLA/smolvla_libero` | as stored (1 action per chunk), MuJoCo 3.8.1 | task 5 only: 6/10 | not run | 26.9 | 0.68 s, every step |
| pi0.5, `lerobot/pi05_libero_finetuned`, 4.1 B | MuJoCo 3.3.7, `n_action_steps=10` as LeRobot's page prescribes, 5 environments | task 5 only: 9/10 | not run | 11.8 | 1.95 s per 5 environments, every 10 steps |

Published figures for comparison (not measured here): SmolVLA paper 90 and 96
with one action per chunk; X-VLA paper 98.2 and 98.6; LeRobot's pi0.5 page 97
and 99.

pi0.5 runs here, with two reservations. Its tokenizer (the component that
turns the instruction text into numbers) sits in a Hugging Face repository
that requires accepting Google's licence and a login token, and this machine
has neither. I rebuilt it from the public file the original pi0.5 code uses
(`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`) and
checked on 200 prompts in pi0.5's format that it produces the same numbers as
that file read directly. I could not compare it with the gated original. And
it is the heaviest of the three: 13.3 GiB of GPU memory and 17.5 GiB of
process memory with five environments, and 11.8 s per episode on task 5
against 8.8 s for X-VLA on the same task.

## Speed and memory

Measured inside the evaluation process, videos off, on a machine that other
spikes were using at the same time (load average between 6 and 32). Treat the
times as upper bounds; the first SmolVLA row was measured twice and came out at
6.5 and 9.0 s per episode.

| Setup | Envs at a time | s per episode | Env steps per s | Episodes per hour | Peak process memory | Peak GPU memory: PyTorch / whole process |
|---|---|---|---|---|---|---|
| X-VLA, `libero_spatial`, 200 episodes | 5 | 6.67 | 17.1 | 540 | 7.8 GiB | 4.1 / 7.2 GiB |
| X-VLA, `libero_object`, 200 episodes | 5 | 5.68 | 24.2 | 634 | 7.8 GiB | 4.1 / 6.9 GiB |
| X-VLA, `libero_goal`, 100 episodes | 5 | 4.75 | 23.3 | 758 | 7.8 GiB | 4.1 / 6.8 GiB |
| X-VLA, `libero_spatial` task 5 (`x02`) | 10 | 7.92 | 12.8 | 455 | 7.8 GiB | 4.9 / 10.5 GiB |
| SmolVLA, 10 actions per chunk, `libero_spatial` (`f1`, `f2`) | 10 | 6.50, 9.02 | 21.6, 15.7 | 554, 399 | 9.0 GiB | 1.4 / 7.3 GiB |
| SmolVLA, 10 actions per chunk, `libero_object` (`f3`) | 10 | 5.20 | 30.3 | 692 | 8.3 GiB | 1.4 / 6.6 GiB |
| pi0.5, 10 actions per chunk, `libero_spatial` task 5 (`p01`) | 5 | 11.8 | 9.5 | 305 | 17.5 GiB | 10.1 / 13.3 GiB |
| SmolVLA, 1 action per chunk, task 5 (`d02_sp5_nas1`) | 10 | 16.9 | 16.6 | 213 | 8.0 GiB | 1.4 GiB / not recorded |
| SmolVLA `HuggingFaceVLA`, 1 action per chunk, task 5 (`d03`) | 10 | 26.9 | 6.3 | 134 | 8.0 GiB | 1.7 GiB / not recorded |

- "s per episode" is the time inside the rollouts divided by the episodes. In
  the 200-episode runs, wall time including start-up of each process is about
  0.6 s per episode more.
- "Env steps per s" counts the steps of each episode until it ended and
  divides by the same rollout time. A policy that fails more runs more steps to
  the limit, so this number is not comparable across policies; seconds per
  episode is.
- "Whole process" GPU memory is what the kernel accounts to the evaluation
  process: PyTorch plus one renderer per environment at roughly 0.6 GiB each.
- Where X-VLA's time goes on `libero_spatial`: model 34% (204 calls of 2.2 s
  for five environments), simulator steps 26%, resets 15%, the rest is
  conversion of observations and Python overhead.
- A counted simulator step cost 10 to 20 ms inside the evaluations, against
  6.5 ms for a lone environment in `bench_sim.py`. The difference is simulation
  that is thrown away: environments that finished early are rebuilt and keep
  running until the slowest one of the round ends (see "Side findings"). In
  `f1` the simulator ran 23,040 steps for 14,009 that counted, in the X-VLA
  runs 27,840 for 22,821 and 31,500 for 27,518.
- With videos on, LeRobot renders the cameras a second time on every step for
  the first 10 episodes of each task. I did not measure that cost cleanly. Ten
  X-VLA episodes of its weakest task with videos, three of them failures
  (episodes 2, 3 and 8):
  `~/.cache/robots-probes/baseline/runs/v1_xvla_sp4_video/videos/libero_spatial_4/`.

## What changes with MuJoCo 3.3.7

- **Start states:** four of 40 tasks, as listed above. `libero_object` and
  `libero_goal` are not affected.
- **Speed:** no difference I could measure. One environment, 600 steps, four
  runs each, CPU time per step 5.6 to 6.1 ms on 3.3.7 and 5.4 to 6.4 ms on
  3.8.1 (`bench_sim.py`). Wall-clock steps per second were 137 to 154 against
  131 to 159, on a machine that other jobs were using.
- **Compatibility:** robosuite 1.5.2 requires MuJoCo 3.3.0 or newer and
  hf-libero allows anything below 3.9, so 3.3.7 satisfies both. Nothing else
  in the environment changed.
- **Which version:** I measured 3.3.7 and 3.8.1 only. Issue 4390 places the
  change at 3.4.0 and reports 3.3.0 and 3.3.7 as behaving alike.

## Side findings

- **Absolute control is broken on the robosuite 1.5.2 port**.
  Commanding the pose the arm already has moves it 0.9 m; on robosuite 1.4.0 it
  stays within 0.1 mm. robosuite 1.5 reads absolute commands relative to the
  robot's base, 1.4 read them in world coordinates. Setting the controller's
  `input_ref_frame` to `"world"` restores 0.1 mm. `probe_absolute_control.py`.
  SmolVLA and pi0.5 command small movements and are not affected.
- **Finished environments keep simulating, and skip start states.** LeRobot
  has a wrapper that freezes an environment once its episode ends, but applies
  it only to the asynchronous environment runner, which crashes on this
  machine. With `--eval.use_async_envs=false`, an environment that finishes is
  reset on the next step, plays a second episode that is thrown away, and its
  start-state counter moves on. Cost: 13 to 39% of the simulated steps in my
  runs. Consequence for results: none when a task is run in one round (all the
  10-episode runs here, start states 0 to 9); with several rounds the later
  rounds start from other stored states than one would expect. Applying the
  wrapper in the synchronous runner as well is filed as follow-up work. Read
  from `lerobot/envs/utils.py` and `libero.py`, and consistent with the step
  counts above; I did not test the fix.
- **The simulator's renderers, not the model, hold most of the GPU memory.**
  A SmolVLA evaluation with 10 environments holds 7.3 GiB on the GPU, of which
  PyTorch has 1.4 GiB; X-VLA with 5 environments 7.2 GiB, of which PyTorch has
  4.1 GiB. That is roughly 0.6 GiB per environment for rendering.
- **The stored start state does not fix the furniture.** Stove, cabinet, wine
  rack and similar get their exact position from NumPy's global random
  generator at every reset. Two unseeded runs put them 5 to 18 mm apart;
  seeded, they agree. Noted in the replay spike (`spikes/replay`).
- **`tools/heavy-run` stopped one of my runs at start-up** because GPU memory on
  the whole machine was above 24 GiB at that moment. My process had not loaded
  anything yet, so the memory was held by other jobs.
  `measured_eval.py` now refuses to start when its expected use would take the
  machine above 20 GiB, and `run_suite.sh` retries.
- **X-VLA in 16-bit floats does not run** in this LeRobot version
  (`--policy.dtype=bfloat16` ends in `TypeError: Got unsupported ScalarType
  BFloat16`).

## Not verified

About the winning setup:

- **One seed.** LeRobot recommends averaging over three seeds. All runs here
  use seed 1000.
- **Repeatability beyond one case.** `libero_spatial` task 4 was run twice
  with the same seed (`g1_xvla_spatial_t4` and `v1_xvla_sp4_video`). The first
  ten episodes agree exactly: the same seven successes and three failures, and
  the same length in steps for every episode. That is one task; the replay spike
  (`spikes/replay`) covers the question properly.
- **Which start states.** Not the first 20 per task, as explained above. The
  rate is over 20 stored start states per task either way.
- **X-VLA on robosuite 1.4.0.** Not run. All X-VLA numbers are on the 1.5.2
  port with the frame correction applied at run time. The correction is not
  in the port's patch file yet.
- **X-VLA with the 800-step limit** of LeRobot's X-VLA page, and X-VLA on
  MuJoCo 3.8.1: not run.
- **`libero_10`** for X-VLA: not run. `libero_goal` has 100 episodes, not 200.
- **Speed on a quiet machine.** Every timing was taken while other spikes were
  running.

About the causes:

- **Whether MuJoCo 3.3.7 changes the other nine `libero_spatial` tasks.** The
  start states are identical, the totals of the two 100-episode runs are equal,
  and the per-task differences are within chance. A loss or gain of a few
  points on those tasks would not have shown.
- **Whether the airborne objects cost success on `libero_10` tasks 0, 1 and 7.**
  I measured the start state only, no policy on `libero_10`.
- **Start states other than state 0** in the 40-task comparison (states 0 and 1
  for `libero_spatial` task 5). Issue 4390 reports all 50 states of task 5 as
  identical in this respect.
- **MuJoCo versions other than 3.3.7 and 3.8.1.**
- **SmolVLA with one action per chunk on a whole suite.** That is how the
  SmolVLA paper evaluates. Users in LeRobot issue 4614 report 83% against 76%
  with 10 on `libero_spatial`, with task 5 broken. It took 16.9 s per episode
  here on task 5, where every episode ran to the limit, against 6.0 s with 10.
  X-VLA is both better and faster, so I did not spend the machine time.
- **`HuggingFaceVLA/smolvla_libero`** beyond 10 episodes of one task.
- **Camera resolution.** The environment renders 360 by 360 pixels, the
  training data is 256 by 256, and the policies rescale either. Not compared.
- **The 2 degree orientation offset on the port**: its effect on success.

About pi0.5: one task, ten episodes, with a tokenizer I rebuilt and could not
compare with the original. Its licence is Google's Gemma terms, not Apache
2.0 like the other two; I did not read what they allow.

## Environments

- `~/.cache/robots-probes/libero-port`: the shared environment, MuJoCo 3.8.1.
  Not modified.
- `~/.cache/robots-probes/baseline-venv-mj337`: mine. A copy of `libero-port`
  (`cp -a --reflink`), then
  `uv pip install --python .../bin/python mujoco==3.3.7 sentencepiece`.
  [`venv-mj337.freeze.txt`](venv-mj337.freeze.txt) lists every package; it
  differs from `libero-port` in those two lines only. LIBERO needs a config
  file that points into the environment it runs from:
  `~/.cache/robots-probes/baseline/libero-config-mj337/config.yaml`.
- Checkpoints are in the Hugging Face cache, except pi0.5, which is in
  `~/.cache/robots-probes/baseline/pi05_libero_finetuned_local`. In that copy
  `policy_preprocessor.json` names my rebuilt tokenizer,
  `~/.cache/robots-probes/baseline/paligemma-tokenizer-local`, instead of
  `google/paligemma-3b-pt-224`; the original file is kept beside it as
  `.json.orig`. The tokenizer was built by putting the public
  `paligemma_tokenizer.model` into a directory as `tokenizer.model` with a
  `tokenizer_config.json` naming `GemmaTokenizer` (adds the start token, no end
  token), loading that with `AutoTokenizer` and saving it.
- The training data used for comparison is a copy of `lerobot/libero` in
  `~/.cache/robots-probes/baseline/ds/lerobot_libero` (1.9 GB).
- All run outputs: `~/.cache/robots-probes/baseline/runs/<name>/` with
  `eval_info.json` (LeRobot's result), `measure.json` (timings and memory) and
  `<name>.log` beside it.

## Files here

| File | Purpose |
|---|---|
| `run.sh` | One measured evaluation behind the heavy-job lock. |
| `run_suite.sh` | A suite as one run per task; resumable. |
| `measured_eval.py` | `lerobot-eval` plus timers, memory accounting, a GPU memory guard and the two switches. |
| `summarize.py` | Results per run, or added up per suite with a 95% interval. |
| `compare_runs.py` | Paired comparison of two runs on the same start states. |
| `probe_init_state.py` | What the policy sees at the start of one task, against a training frame. |
| `probe_settle_all.py` | Where objects rest after reset in all 40 tasks; compares two MuJoCo versions. |
| `probe_absolute_control.py` | Hold-and-move test of absolute control. |
| `bench_sim.py` | Simulator speed without a policy. |
| `check_checkpoint_keys.py` | Whether a checkpoint's weights match the model this LeRobot version builds. |
| `venv-mj337.freeze.txt` | Package list of my environment. |

## Runs

Every run is `spikes/baseline/run.sh NAME FLAGS`, with the environment variables
shown. `run.sh` adds `--env.type=libero --eval.n_episodes=10
--eval.batch_size=10 --eval.use_async_envs=false --policy.device=cuda
--seed=1000`, sets `MIOPEN_FIND_MODE=2 MUJOCO_GL=egl` and turns videos off;
flags given here override those. `MJ=337` selects my MuJoCo 3.3.7 environment,
no `MJ` the shared one with 3.8.1. `RM` stands for
`--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'`,
`SV` for `--policy.path=lerobot/smolvla_libero`, `XV` for
`--policy.path=lerobot/xvla-libero --env.control_mode=absolute` with
`BASELINE_ABS_WORLD=1`.

| Run | Environment | Flags |
|---|---|---|
| `cmp_libero_spatial_libero-port` | 3.8.1 | an earlier run: `lerobot-eval` with `SV RM --env.task=libero_spatial`, videos on; output in `~/.cache/robots-probes/runs/` |
| `d01_sp5_before` | 3.8.1 | `SV RM --env.task=libero_spatial --env.task_ids='[5]'` (result in the log only; `eval_info.json` was not written because of a bug in my runner, fixed since) |
| `d02_sp5_nas10`, `d02_sp5_nas1` | 3.8.1 | as `d01` plus `--policy.n_action_steps=10`, `=1` |
| `d03_sp5_hfvla` | 3.8.1 | `--policy.path=HuggingFaceVLA/smolvla_libero --env.task=libero_spatial --env.task_ids='[5]'` |
| `d04_sp5_mj337_nas50`, `d04_sp5_mj337_nas10` | `MJ=337` | as `d01`, and as `d01` plus `--policy.n_action_steps=10` |
| `d05_sp07_mj337_nas10_emptycam` | `MJ=337` | `SV RM --env.task=libero_spatial --env.task_ids='[0,7]' --policy.n_action_steps=10 --policy.empty_cameras=1` |
| `f1_spatial_mj381_nas10` | 3.8.1 | `SV RM --env.task=libero_spatial --policy.n_action_steps=10` |
| `f2_spatial_mj337_nas10` | `MJ=337` | same |
| `f3_object_mj337_nas10` | `MJ=337` | `SV RM --env.task=libero_object --policy.n_action_steps=10` |
| `x01_xvla_sp0_mj337` | `MJ=337` | `XV --env.task=libero_spatial --env.task_ids='[0]' --env.episode_length=800` |
| `x02_xvla_sp5_mj337` | `MJ=337` | `XV --env.task=libero_spatial --env.task_ids='[5]'` |
| `x03_xvla_sp5_mj337_bf16` | `MJ=337` | as `x02` plus `--policy.dtype=bfloat16` (fails, see side findings) |
| `g1_xvla_spatial_t0` .. `_t9` | `MJ=337` | through `run_suite.sh`: `XV --env.task=libero_spatial --env.task_ids='[N]' --eval.n_episodes=20 --eval.batch_size=5` |
| `g2_xvla_object_t0` .. `_t9` | `MJ=337` | same with `libero_object` |
| `g3_xvla_goal_t0` .. `_t9` | `MJ=337` | same with `libero_goal` and `--eval.n_episodes=10` |
| `v1_xvla_sp4_video` | `MJ=337` | `XV --env.task=libero_spatial --env.task_ids='[4]' --eval.batch_size=5` with `BASELINE_NO_VIDEO=0` |
| `p01_pi05_sp5_mj337` | `MJ=337` | `--policy.path=~/.cache/robots-probes/baseline/pi05_libero_finetuned_local --env.task=libero_spatial --env.task_ids='[5]' --policy.n_action_steps=10 --policy.compile_model=false --eval.batch_size=5` |

Probes without a model (all through `tools/heavy-run --light`, with
`MUJOCO_GL=egl` and the `LIBERO_CONFIG_PATH` of the environment used):

```sh
python spikes/baseline/probe_init_state.py --suite libero_spatial --task 5 --n 2 --size 256 \
    --out ~/.cache/robots-probes/baseline/probe/sp5_mj381 --dataset ~/.cache/robots-probes/baseline/ds/lerobot_libero
python spikes/baseline/probe_settle_all.py --out ~/.cache/robots-probes/baseline/probe/settle_seeded_mj381.json
python spikes/baseline/probe_settle_all.py --compare .../settle_seeded_mj337.json .../settle_seeded_mj381.json
python spikes/baseline/probe_absolute_control.py [--world]
python spikes/baseline/bench_sim.py --size 360
python spikes/baseline/check_checkpoint_keys.py lerobot/smolvla_libero HuggingFaceVLA/smolvla_libero
python spikes/baseline/compare_runs.py A/eval_info.json B/eval_info.json
```
