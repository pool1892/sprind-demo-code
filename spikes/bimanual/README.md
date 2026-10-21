# Bimanual settings that run on this machine without CUDA

Spike run on this machine on 2026-10-07. The question was
which two-arm manipulation settings work here, on a Radeon 8060S with no
NVIDIA GPU and no CUDA, judged by four things: a public pretrained policy
with high success, deterministic headless simulation (bit-exact replay),
speed, and the cost of putting the setting behind our adapters. A
desk-research survey from sources ran in
parallel; this report is the hands-on half and only states what was run or
read in code here.

Everything ran on the CPU for the policy (the heavy-job lock was held by the
pi0.5 spike all evening) and on the Radeon for rendering, as light jobs with
at most four simulation processes. **The machine was busy throughout:** the
load average was 3 to 21 in the run records, partly from this probe's own
parallel runs, and the GPU was at times 99% busy with the other spike's work
(seen live in the GPU monitor, not recorded), so timings are upper bounds,
good to perhaps 20 to 30%.
All outputs are under `~/.cache/robots-probes/bimanual/` (called `$P` below):
venvs, clones, downloads, and `$P/runs/` with one directory per measurement.

Words used throughout:

- An **episode** is one attempt at a task from a seeded start. A **policy
  call** is one forward pass of the model; ACT and the diffusion policy return
  a chunk of actions per call, so most steps take an action from a queue
  without calling the model.
- **Identical** means bit for bit, compared by SHA-256 over every observation
  array and every action of an episode, never with a tolerance.
- **ACT** (Action Chunking Transformer) is the policy published with the
  ALOHA simulation. **DDPM** is the diffusion sampler of the robomimic policy:
  100 denoising passes of a network per call.

## Verdict

**Two bimanual settings run here end to end without CUDA, with a public
policy, bit-exact replay and nothing but MuJoCo underneath: gym-aloha with
LeRobot's ACT, and robosuite's TwoArmTransport with robomimic's diffusion
policy.** ACT on transfer cube succeeds in 81% of 200 seeds (its card says
83%) at 2 to 4 s per episode with the policy on the CPU, and gym-aloha
installs beside our pins (MuJoCo 3.3.7, robosuite 1.5.2, numpy 1.26.4,
LeRobot 0.6.1) with dm_control 1.0.34. Its second task, insertion, reaches
only 12.5%. The Transport policy succeeds in 20 of 20 on exactly our robosuite
and MuJoCo, but its 100-step sampler makes an episode take 7 minutes on the
CPU; sampled with 10 steps instead it took 46 s and succeeded in 19 of 20.

**RoboTwin 2.0 runs here only in part.** SAPIEN renders on the Radeon through
Vulkan, rasterised and ray-traced, with no display. But stock RoboTwin cannot
even import a task without CuRobo, which is NVIDIA-only, and its evaluation
runs a CuRobo-planned expert on every seed. With CuRobo stubbed out, or with
the `RLinf_support` branch's mplib planner, tasks build, step and replay
bit-exactly under joint-space actions; but the scripted experts mostly fail
with mplib, the images lose their (CUDA-only) denoiser, and the policies are
VLA-sized checkpoints in other frameworks. BiGym and ManiSkill3's two-arm
tasks run and replay, but have no usable pretrained policy.

None of these is industrial: research arms, one or two objects on a bare
table, handovers in mid-air. The cheapest to add is gym-aloha transfer cube,
about three days.

| Candidate | Runs here without CUDA | Policy available | Success measured here | Seconds per episode (policy on CPU; busy machine, upper bounds) | Deterministic (bit-exact) | Simulator | Days to add behind our adapters |
|---|---|---|---|---|---|---|---|
| **gym-aloha transfer cube + ACT** | yes | yes, `lerobot/act_aloha_sim_transfer_cube_human` | **162 of 200 (81%)**; card 83% | **2.2** (top camera only; 3.7 under more load; 5.9 as shipped, three cameras) | yes, across processes, with a fixed PyTorch thread count | MuJoCo 3.3.7 via dm_control 1.0.34 (ships with 3.8.1) | **3**: same LeRobot policy path; a dm_control environment wrapper with our own per-episode seeding instead of gym-aloha's module-level pose; top camera only; faults are MuJoCo model edits as for LIBERO. Plus about 2 for a scripted expert: gym-aloha ships none, and its end-effector tasks raise `NotImplementedError` |
| gym-aloha insertion + ACT | yes | yes, `lerobot/act_aloha_sim_insertion_human` | 25 of 200 (12.5%); card 20.6% | 2.5 | yes | same | half a day on top of transfer cube; too weak for a healthy baseline |
| **robosuite TwoArmTransport + robomimic diffusion policy** | yes | yes, `robomimic/robomimic_models` (one task) | **20 of 20** (100-step DDPM); 19 of 20 with 10 DDIM steps | **418** (DDPM); 46 (DDIM 10) | yes, policy noise seeded per episode (2 of 2 repeats) | MuJoCo 3.3.7, robosuite 1.5.2: our pins | **4**: environment from our robosuite code; a new robomimic policy adapter (frame stacking, seeded sampler); robomimic's old pins (diffusers 0.11.1, transformers 4.41.2) probably need their own process; speed on the GPU unmeasured |
| robosuite TwoArmLift, TwoArmHandover | yes | none found | none | environment only: 155 to 165 steps per second, no camera | yes | same | not without training a policy |
| RoboTwin 2.0 | **partly**: stock code no (CuRobo); with a stub or the RLinf branch's mplib, yes | yes, 11 official co-trained checkpoints (3 to 31 GB) and many others on the hub | none run | simulator alone 0.3 s per action with observation, so 2 to 8.5 minutes at the step limit (400 to 1,700 actions, by task) | environment yes under joint actions; the mplib expert no | SAPIEN 3.0.0b1 | **10 to 15**: own process and environment (torch 2.4.1, SAPIEN 3.0.0b1); CuRobo patched out and the expert check replaced; images without denoiser; every fault mechanism new on SAPIEN; policies need the GPU and their own serving (openpi JAX, X-VLA remote code) |
| BiGym | yes, with an install workaround | no official one | none | environment only: 310 to 350 steps per second with one camera | yes | MuJoCo 3.1.5 (pinned for its demos) | not without training a policy |
| ManiSkill3 two-robot tasks | yes, CPU physics | none found | none | environment only: 97 to 126 steps per second with RGB | yes | SAPIEN 3.0.3 | not without training a policy |

The day estimates are mine and assume the stage-2 rollout engine with its two
adapters exists; they cover a working healthy baseline with replay and the
first faults, not the full fault catalogue.

## gym-aloha with LeRobot's ACT

**Set-up.** `$P/venv-aloha`: LeRobot 0.6.1 with its `aloha` extra, which
brings gym-aloha 0.1.4, and PyTorch 2.11.0 for the CPU. Only two ACT
checkpoints exist under `lerobot/` on the Hugging Face hub:
`act_aloha_sim_transfer_cube_human` (revision `ba73b276`) and
`act_aloha_sim_insertion_human` (revision `33259aa8`). The `_scripted`
variants and LeRobot's diffusion and VQ-BeT ALOHA checkpoints do not exist
(the hub answers 401 for each; `$P/hub/`). Both checkpoints predate LeRobot's
processor format and were converted with LeRobot's own
`migrate_policy_normalization` script (`$P/ckpt/`, logs in `$P/logs/`).
`aloha_eval.py` runs one episode per seed, seeded as `lerobot-eval` does
(`env.reset(seed=s)`, seeds counting up from 1000), with LeRobot's ALOHA limit of 400
steps and its observation preprocessing, and hashes every observation and
action. Four threads for PyTorch unless stated. The checkpoint's
configuration lists `observation.images.top` (3 by 480 by 640) as its only
image input, with `chunk_size` and `n_action_steps` both 100 and
`temporal_ensemble_coeff` null, so rendering only the top camera (below)
does not change what the policy sees.

**What the tasks are.** Two 6-joint ViperX arms face each other across a bare
grey table, seen by one top-down camera at 640 by 480; the policy commands
14 joint positions at 50 Hz. In *transfer cube* the right arm picks a small
red cube and hands it to the left arm in mid-air; in *insertion* each arm
picks one part and they push a peg into a socket in mid-air. Videos:
`$P/runs/tc_a/seed100{0..3}.mp4` and `$P/runs/ins_a/seed100{0..3}.mp4`.
Nothing about it is industrial: one object on an empty table, no bins,
fixtures, conveyor or clutter, one camera, constant light, a research arm,
and policies trained on 50 human demonstrations each.

| | Transfer cube | Insertion |
|---|---|---|
| **Success, MuJoCo 3.3.7 (our pin), 200 seeds 1000 to 1199** | **162 (81.0%; 95% interval 75 to 86%)** | **25 (12.5%; 8.6 to 17.8%)** |
| Success, MuJoCo 3.3.7, first 50 seeds | 38 (76%) | 7 (14%) |
| Success, MuJoCo 3.8.1, first 50 seeds | 37 (74%) | 8 (16%) |
| Success on the model card (500 episodes, versions not stated) | 83.0% | 20.6% |
| Mean steps per episode, MuJoCo 3.3.7, 200 seeds | 311 | 392 |
| Policy calls per episode (one per 100 steps) | 3.4 | 4.0 |
| One policy call, median (4 threads) | 82 to 139 ms, by load | 82 to 110 ms, by load |
| Seconds per episode, MuJoCo 3.8.1, 3 cameras rendered as shipped | 5.9 (`tc_b`) | 6.8 (`ins_b`) |
| Seconds per episode, top camera only, MuJoCo 3.8.1 / 3.3.7 | 3.7 / 2.2 | not run / 2.5 |

The first runs, `tc_a` and `ins_a`, gave 7.6 and 8.6 s per episode as
shipped because twelve of their episodes ran beside the other spike's GPU
job, at 18 to 55 s each; `tc_b` and `ins_b` are the fair baseline.

Transfer cube matches its card. Insertion is below its card on 200 seeds
(20.6% lies outside the interval). On MuJoCo 3.8.1 the first 50 seeds gave
16%, too few to say whether the version matters; I did not find the cause. The insertion repository's `eval_info.json` is a byte-for-byte
copy of the transfer cube one (83%); its model card says 20.6%. Runs:
`$P/runs/tc_mj337`, `tc_mj337_more`, `ins_mj337`, `ins_mj337_more` (MuJoCo
3.3.7), `tc_a`, `ins_a` (3.8.1).

**Speed.** The simulator, not the policy, takes the time: ACT is called once
per 100 steps, so its share of an episode was 10 to 30%. gym-aloha renders
three 640 by 480 cameras every step (`top`, `angle`, `front_close`) and the
policy sees only `top`. Rendering only `top` (`--top-only`, a monkeypatch of
`get_observation`) cut the simulator from 14.4 to 7.2 ms per step and the
episode from 5.9 to 3.7 s (`tc_b` against `tc_top`, run side by side; 3.4 s
for a success, 4.4 s for a 400-step failure), and gave **identical episodes,
50 of 50**. That is about 980
episodes per hour per simulation process with the policy on the CPU, at load
averages between 6 and 14 (`$P/runs/tc_top/`). One policy call took a median
of 86 ms on four threads at load 4 to 9, 139 to 167 ms at load 12 to 14,
and 355 ms on one thread. The first ten minutes ran beside the other spike's
GPU job at 99% GPU use (seen live in the GPU monitor, not recorded);
episodes then took up to 55 s each, almost all of it in the simulator step.

**Determinism.** Every check was by hash over the whole episode.

| Check | Result | Run directories |
|---|---|---|
| Same seeds in a second process, transfer cube | identical, 50 of 50 | `tc_a`, `tc_b` |
| Same seeds in a second process, insertion | identical, 50 of 50 | `ins_a`, `ins_b` |
| Three cameras rendered against one | identical, 50 of 50 | `tc_a`, `tc_top` |
| Seed s, then seed s+10000, then seed s again, one process | identical, 6 of 6 | `tc_repeat` |
| 1 PyTorch thread against 4 | **different, 10 of 10**; seed 1003 fails with 1 thread and succeeds with 4 | `tc_t1`, `tc_a` |

The start state follows the seed, and nothing carries over between episodes
in a process. gym-aloha registers both tasks with `nondeterministic=True`
and a comment that "even after seeding, the rendered observations are
slightly different"; that did not happen here. ACT itself draws nothing at
inference (its latent is zero and dropout is off). Three things to know for
replay: a reset without a seed draws the object pose from fresh entropy;
the object pose goes through a module-level variable (`BOX_POSE`) shared by
every environment in a process, which is safe only while resets are
sequential; and **the PyTorch thread count is part of an episode's
identity on the CPU**: the same observation gives other actions with
another thread count, most likely through the order of floating-point
sums in the CPU kernels; not confirmed.

**Versions.** gym-aloha 0.1.4 asks for `mujoco>=3.0.0,<3.9.0` and
`dm-control>=1.0.15`. Left alone, the resolver picks dm_control 1.0.41,
which needs MuJoCo 3.8.1, and the runs above used those. dm_control 1.0.34
needs MuJoCo 3.3.6 or later, so with our pins (MuJoCo 3.3.7, robosuite
1.5.2, numpy 1.26.4 forced as in our `pyproject.toml`, LeRobot 0.6.1) the
whole set resolves into one environment (`$P/venv-aloha-mj337`,
`$P/logs/install-aloha-mj337.log`). It runs there: the 200-seed rows of
the table above come from that environment, with gym-aloha on dm_control
1.0.34 and robosuite 1.5.2 importable beside it. dm_control 1.0.36 and
later need MuJoCo 3.4 or newer, so dm_control has to be pinned. The first
observation of every episode was identical between MuJoCo 3.3.7 and 3.8.1;
the dynamics then differ (0 of 50 episodes identical), as expected.

## SAPIEN 3 rendering on Vulkan

`sapien_render.py` builds a small table scene, renders RGB offscreen at 256
by 256 and 640 by 480, times 100 frames after a warm-up, renders one state
twice, and steps CPU PhysX twice from scratch. It ran with `DISPLAY` and
`WAYLAND_DISPLAY` unset, in two venvs: `$P/venv-sapien` with SAPIEN 3.0.3
(ManiSkill's version) and `$P/venv-robotwin` with 3.0.0b1 (RoboTwin's pin).

- **Both versions pick the Radeon through RADV with no display and no CUDA.**
  `sapien.render.get_device_summary()`: "GPU: AMD Radeon 8060S Graphics
  (RADV STRIX_HALO) ... cudaId: -1, rayTrace: 1".
- **The rasterizer and the ray tracer both work.** Pictures:
  `$P/runs/sapien/default/frame_640x480.png` and
  `rt_none/frame_640x480.png`.
- **The denoiser needs CUDA and fails quietly.** With the OIDN denoiser
  selected, SAPIEN logs "CUDA Error: cudaErrorInsufficientDriver" and
  "Failed to initialize denoiser", raises nothing, and returns the same
  bytes as with no denoiser (`rt_oidn` and `rt_none`, same hashes). The
  picture stays grainy.
- **Repeatable.** The same state rendered twice gave identical pixels in
  every mode, the ray tracer included, and two fresh scenes stepped 300 times
  ended in identical poses.
- SAPIEN 3.0.0b1 imports `pkg_resources`, which a fresh uv venv lacks;
  `setuptools==69.5.1` (as RoboTwin's script installs) fixes it.

| Milliseconds per frame, render only | 256 by 256 | 640 by 480 |
|---|---|---|
| 3.0.3, rasterizer (`default` shader) | 1.0 | 7.4 (3.6 with a physics step between frames, so timing noise) |
| 3.0.3, rasterizer (`minimal` shader) | 1.1 | 2.6 |
| 3.0.3, ray tracer, 32 samples per pixel | 9.1 | 24 |
| 3.0.0b1, rasterizer | 1.6 | 5.2 |
| 3.0.0b1, ray tracer, 32 samples per pixel | 9.6 | 27 |

These were taken with the GPU 24 to 74% busy from other jobs (seen live in the GPU monitor, not recorded) and a load
average of 13 to 14 (`$P/runs/sapien/clean_*`). The first runs, beside the
ALOHA jobs and the other spike, were up to ten times slower and are not in
the table. **The gate for RoboTwin and ManiSkill3 is passed.**

## RoboTwin 2.0

**Read from the code** (clone of `RoboTwin-Platform/RoboTwin` at
`ea8b2112`, 2026-09-24, in `$P/clones/robotwin`):

- `scripts/requirements.txt` pins SAPIEN 3.0.0b1, mplib 0.2.1, torch 2.4.1,
  numpy 1.26.4 and scipy 1.10.1 (no wheel for Python 3.12; I used 1.17.1).
  `scripts/_install.sh` adds pytorch3d from source and **CuRobo v0.7.8**,
  plus warp-lang. CuRobo builds five `CUDAExtension` modules from `.cu`
  kernels (`$P/clones/curobo/setup.py`); this machine has neither `nvcc`
  nor `hipcc`, and there is no ROCm port to install.
- **CuRobo is needed to load any task, evaluation included.**
  `envs/robot/robot.py` line 15 imports `CuroboPlanner` unconditionally;
  `planner.py` defines it only if `import curobo` succeeds, so without
  CuRobo every task module fails to import (measured:
  `$P/runs/robotwin/as_shipped/result.json`). And `load_robot()` always
  builds a CuRobo planner per arm (`envs/_base_task.py` line 391), in
  evaluation mode too.
- **The scripted experts plan with CuRobo** (`left_plan_path`,
  `plan_batch`, all on `.cuda()` tensors). The evaluation script runs the
  expert on every seed first and skips seeds the expert cannot solve
  (`expert_check`, on by default, `scripts/eval_policy_xpolicylab.py` lines
  629 to 646), so the official seed list itself needs CuRobo. The check can
  be switched off; the seeds then differ from the published protocol.
- **Executing a policy's actions does not need CuRobo in joint mode.**
  `take_action(action_type="qpos")` time-parameterises the path with
  mplib's TOPP on the CPU. End-effector mode (`"ee"`) plans each step with
  CuRobo.
- pytorch3d is optional: it is only used to subsample point clouds, which a
  camera-only policy does not request.
- **Assets** (Hugging Face dataset `TianxingChen/RoboTwin2.0`): background
  textures 10.97 GB, objects 3.74 GB, embodiments 0.22 GB. The clean setting
  (`demo_clean.yml`) loads no background textures, so 3.96 GB suffice for it;
  the randomised setting needs all 14.9 GB. The objects archive is needed
  even to import the package (`envs/utils/rand_create_cluttered_actor.py`
  reads its list at import time).
- **Public checkpoints, seen on the hub:** the same dataset repository has
  `cotrain_ckpt/` with one archive each for AHA_WAM, Abot_M0, EventVLA,
  FastWAM, GalaxeaVLA, Pi_05, Spatial_Forcing, X_VLA, X_WAM,
  Xiaomi_Robotics_0 and starVLA, 3.2 to 31 GB each, which the README calls
  the "official benchmark checkpoints". By their file names six are
  joint-space (Abot_M0, FastWAM, GalaxeaVLA, Pi_05, Xiaomi_Robotics_0,
  starVLA), two end-effector (X_VLA, X_WAM), and three do not say.
  Separately, `2toINF/X-VLA-RoboTwin2` (3.5 GB, action mode `ee6d`, so its
  actions go through the end-effector path), per-task OpenVLA-OFT models
  from RLinf, `StarVLA/Qwen3-VL-OFT-Robotwin2`,
  `motus-robotics/Motus_robotwin2` and many more match "robotwin" (the hub
  search stops at 100; `$P/hub/search_robotwin.json`,
  `robotwin_ckpt_all.json`). I did not
  download or run any of them, and their success rates are for the desk
  research to report.

**Run here** (`$P/venv-robotwin`: SAPIEN 3.0.0b1, mplib 0.2.1, torch 2.4.1
for the CPU, numpy 1.26.4; clean setting, aloha-agilex robot, evaluation
mode; assets: embodiments and objects only).

1. *As shipped:* importing any task fails, `ImportError: cannot import name
   'CuroboPlanner'` (`$P/runs/robotwin/as_shipped/`).
2. *Main branch with a stub for CuRobo* (`robotwin_probe.py --stub`: a
   do-nothing class with CuRobo's numpy-only gripper ramp; anything that
   plans raises). `handover_block` builds in 0.2 to 0.5 s and renders four
   320 by 240 cameras (head, front, two wrists). Twenty joint-space actions
   (`take_action(action_type="qpos")`) with an observation after each took
   0.29 s per action. **Two builds of the same seed in fresh task objects
   gave identical first observations and identical 20-action rollouts**
   (`$P/runs/robotwin/stub_handover/`).
3. *The `RLinf_support` branch with `planner_backend: mplib`* (clone at
   `dca9ec68`, 2026-09-22, `robotwin_mplib_probe.py`). It imports without
   CuRobo (`CuroboPlanner` falls back to `None`) and builds environments.
   - **The scripted experts mostly fail with mplib.** `handover_block`: 0 of
     10 seeds planned or succeeded, 23 to 143 s each, with mplib
     reporting collisions of the arm's links with each other and with the
     table at the start state;
     with RoboTwin's install-time change to mplib applied in-process
     (`--patch-mplib`: screw motions no longer fail on collision), still 0 of 5.
     `lift_pot` with that change: 1 of 5 succeeded, 2 planned but failed the
     success check, 2 found no grasp ("target_pose cannot be None").
     The evaluation protocol runs the expert on every seed first, so with
     mplib most seeds would be skipped or the seed list would differ from
     the published one.
   - **One observation takes 0.19 to 0.23 s** (four ray-traced 320 by 240
     cameras; the first call 3 s), median of 10 per task at load 10 to 16.
   - **End-effector control works through mplib.** Five `"ee"` actions that
     raise both hands by 1 cm each moved the left hand from z = 0.942 to
     0.952, 0.961, 0.971, 0.981 and 0.991 m, at 0.3 to 0.4 s per action (the
     first 1.4 s).
   - **Same seed: the start repeats, the expert does not.** Run three times in
     one process (the third time right after another seed's episode),
     `lift_pot` seed 100000
     had an identical first observation each time, but the expert's final
     state differed every time and its success check came out true, false,
     true. The joint-space rollouts of point 2 were identical, so the
     difference comes from the expert; mplib's sampling planner (RRT) is the
     likely source, which I did not confirm.
4. *Denoiser.* RoboTwin renders every camera with the ray tracer at 32
   samples per pixel and the OIDN denoiser (`envs/_base_task.py` lines 215 to
   218). Here OIDN fails as in the SAPIEN probe, so the images are grainy
   (`$P/runs/robotwin/stub_handover/strip.png`), while the published
   checkpoints were trained on denoised images. AMD's Ryzers recipe sets the
   denoiser to `"none"` and accepts this; whether policies tolerate it is
   untested.

The open Ryzers pull request (AMDResearch/Ryzers #96, head `b0e5056f`,
`$P/downloads/ryzers-pr96/`) packages RoboTwin for this chip with the same
moves: CuRobo replaced by mplib in `robot.py` when CuRobo is missing, the
denoiser off, `setuptools<81` for `pkg_resources`, mplib installed without
its dependencies, numpy 1.26.4 last, pytorch3d built for gfx1151, on an
older RoboTwin commit (`bf44be51`). It publishes no success rates. The
co-trained pi0.5 checkpoint in `cotrain_ckpt/Pi_05/` is an Orbax (JAX)
checkpoint in openpi's layout, step 29999 (`params/` with OCDBT shards; read
from the first 20 MB of the archive, `$P/downloads/robotwin-pi05-head/`), not
a LeRobot one, so LeRobot 0.6.1 cannot load it as it is.

## robosuite two-arm tasks and robomimic's Transport policy

**The environments.** `$P/venv-robosuite`: robosuite 1.5.2 and MuJoCo 3.3.7,
our pins. `robosuite_twoarm.py` builds TwoArmLift and TwoArmHandover (two
Pandas side by side) and TwoArmTransport (two Pandas facing each other), all
with 14-dimensional actions (end-effector pose control and a gripper per arm),
headless through EGL. Frames: `$P/runs/robosuite/twoarm/*.png`. Lift is a pot
lifted by two handles, Handover a hammer passed from one hand to the other,
Transport a hammer taken out of a lidded bin on one table and dropped into a
bin on the other while the second arm clears a piece of trash. Of the three,
Transport is the nearest thing to a work cell, and it is still a lab scene.

| | TwoArmLift | TwoArmHandover | TwoArmTransport |
|---|---|---|---|
| Steps per second, no camera | 165 | 156 | 125 |
| One 256 by 256 camera | 90 | 94 | 85 |
| Two cameras | 72 | 86 | 73 |
| Same seed, two environments: start state and observation | identical | identical | identical |
| 200 seeded random actions: state | identical | identical | identical |
| 200 seeded random actions: observations, 1 + 8 repeats | 8 of 9 | 9 of 9 | 9 of 9 |
| Another seed gives another start | yes | yes | yes |
| Reseed in place, reset, 50 steps, twice | identical | identical | identical |

Control steps at 20 Hz, load average about 7 (`$P/runs/robosuite/twoarm/result.json`,
`obs_diff_repeats.log`). The one mismatch was in the first run, with two
environments rendering in the same process while the ALOHA runs shared the
GPU: the states stayed identical and an observation differed. Eight
repeats per task in fresh processes found nothing (24 of 24 identical;
they ran as a script in the scratch area, copied into this directory
afterwards as `robosuite_obs_diff.py`; results in `obs_diff_repeats.log`). Our
replay rules already say "one rendering context at a time", which this fits;
the first run did not record which key differed. robosuite 1.5.2 takes its
seed at construction and shares one numpy generator between the object
placement, the objects and the robots' start noise; a per-episode seed means
resetting that generator's state in place, which works (last row).

**robomimic's Transport policy.** The only pretrained two-arm robosuite
policy I found from the robomimic team is in `robomimic/robomimic_models`
(revision `bb6c8757`, February 2026): a diffusion policy (U-Net, DDPM with 100
denoising passes per call, 8 actions per call, two observation frames) on four
84 by 84 cameras and the two grippers' poses, trained on the `transport/ph`
image data for robosuite 1.5. The checkpoint itself records `env_version:
1.5.2`. robomimic's file names log its own success during training; I took
`model_epoch_400_image_v15_success_0.96.pth` (1.9 GB), and its config rolls
out 50 episodes of at most 600 steps. PyPI has robomimic only up to 0.3.0, so I installed the GitHub master
(`d309eaec`, version 0.5.0) into the same robosuite 1.5.2 venv with PyTorch
for the CPU: it loads and runs with no CUDA-only package (it pins
`diffusers==0.11.1`, `transformers==4.41.2`, `huggingface_hub==0.23.4` and
builds `egl_probe` with CMake). `robomimic_eval.py` builds the environment
from the checkpoint's own metadata, seeds the robosuite generator and
PyTorch's generator per episode, and hashes everything.

| 20 episodes, seeds 0 to 19, 4 processes of 4 threads | Result |
|---|---|
| Success | **20 of 20** |
| Steps per episode | mean 455, from 392 to 576 (limit 600) |
| Seconds per episode | **418** (96.5% of it in the policy) |
| One policy call (100 denoising passes) | median 7.2 s, 90th percentile 8.2 s |
| Simulator with four 84 by 84 cameras | 30 ms per step |
| Load average | 10 to 21 |

The four Transport processes ran beside this probe's other runs, so the
load of 10 to 21 here, and of 13 to 18 during the 200-seed ALOHA runs, was
partly the probe's own. Runs in `$P/runs/robomimic/ddpm_s{0,5,10,15}/`, a video in
`ddpm_s0/seed0.mp4`. On the CPU the setting is correct but slow: about nine
episodes per hour per process.

**Determinism.** Seeds 0 and 5 run again in a fresh process (seed 5 now
second in its process, before first) gave identical first states,
observations, actions and final states (`$P/runs/robomimic/ddpm_repeat/`).
The 20 seeds gave 20 different start states. The sampler draws its noise from
PyTorch's global generator, so the adapter has to seed it per episode, as
the script does; robomimic's `run_trained_agent.py` seeds numpy and PyTorch
once per run, so an episode there depends on the ones before it.

**A faster sampler.** The same network sampled with 10 DDIM steps instead of
100 DDPM steps (`--ddim 10`, a change the checkpoint was not evaluated with)
succeeded in **19 of 20** (seed 13 failed) at **46 s per episode**, 0.60 s
per call, 78% of the time in the policy, load 9 to 11, from the same 20 start
states as the DDPM runs (`$P/runs/robomimic/ddim10/`). That is about 80
episodes per hour per process on the CPU, and a change of the sampler that
would need its own validation before use.

## BiGym and ManiSkill3

**BiGym** (clone of `chernyadev/bigym` at `d9812445`, version 4.1.0, in
`$P/venv-bigym`). It pins MuJoCo 3.1.5 and dm_control 1.0.19, with the
comment "WARNING: recorded demos might break when updating Mujoco", and a fork
of Gymnasium from git whose metadata uv rejects (a duplicate extra,
`classic-control`). With upstream gymnasium 0.29.1 and BiGym installed
without dependencies it runs: `PickBox` (a Unitree H1 with two grippers on a
floating base, 15 joint-position actions) renders a 256 by 256 head camera
headless, steps at 310 to 350 per second with that camera at load 13, and
two runs of the same seed and actions were identical over 200 steps
(`$P/runs/bigym/pickbox/`). There is no official pretrained policy; the only
model on the hub is a third-party `wangbin08/DM05-BiGym-CutleryLong-50pct`,
not tried. The MuJoCo pin would have to move to 3.3.7 and the demos be
re-checked before it could share our environment. A note, not a candidate.

**ManiSkill3** (`mani_skill` 3.0.1 on SAPIEN 3.0.3, CPU PhysX). Its two-arm
tasks `TwoRobotPickCube-v1` and `TwoRobotStackCube-v1` (two Pandas) and the
humanoid `UnitreeG1TransportBox-v1` build, render RGB observations through
Vulkan (ManiSkill warns that the render device "sapien_cuda" was not found
and falls back; the pictures are fine), run at 97 to 126 steps per second
per process at load about 17, and two runs of the same seed and actions were
identical over 100 steps (`$P/runs/maniskill/twoarm/`). No pretrained policy
for any of them is on the hub (searches for "TwoRobot" and "TransportBox"
return nothing; the first 100 results for "maniskill" hold no two-arm model).
A note too: runnable, but a policy would have to be trained.

## Against the desk research's open points

The desk research's section "Not confirmed, for the probe",
point by point.

- **Confirmed:** `lerobot[aloha]` resolves with dm_control 1.0.34 and MuJoCo
  3.3.7; a seed gives identical states and images under EGL on the Radeon;
  ACT repeats on the CPU (with a fixed thread count); the robomimic
  checkpoint loads and rolls out on robosuite 1.5.2 and MuJoCo 3.3.7 with no
  CUDA-only package, and records robosuite 1.5.2 as its training version;
  stock RoboTwin fails without CuRobo; SAPIEN 3.0.0b1 renders headless on
  RADV, rasterised and ray-traced, and the denoiser setting does nothing
  without NVIDIA; the `RLinf_support` branch with mplib builds environments
  and runs end-effector control; a seed and joint-space actions repeat bit
  for bit; the co-trained pi0.5 is an Orbax checkpoint that LeRobot cannot
  load as it is.
- **Refined:** the mplib experts mostly fail (0 of 15 on `handover_block`,
  1 of 5 on `lift_pot`) and do not repeat. gym-aloha ships no scripted
  expert and raises `NotImplementedError` for the end-effector tasks that the
  ACT repository's expert needs (`gym_aloha/env.py` lines 120 to 129), so a
  mid-episode take-over needs a port of that expert.
- **Not run:** ACT on the Radeon; whether FastWAM loads in LeRobot; any
  RoboTwin policy; a measured fleet throughput.
- **Day estimates differ** between the two documents: settings.md gives 4 to
  5 (gym-aloha), 4 to 6 (robosuite Transport) and 8 to 12 (RoboTwin), this
  report 3 plus 2 for an expert, 4, and 10 to 15. Both are their author's
  estimates, not measurements.

## What I did not verify

About policies and success:

- **Nothing ran on the GPU.** The heavy-job lock was held all evening, so ACT
  and the diffusion policy ran on the CPU only. Whether ACT is identical run
  to run on the Radeon, and how fast the diffusion policy is there, are open.
- **Why insertion is below its card** (12.5% against 20.6% on 200 seeds). The
  card does not state its MuJoCo or gym-aloha version.
- **Transport beyond 20 episodes**, and the DDIM variant beyond 20.
- **No RoboTwin policy was downloaded or run**, so no success rate on this
  machine, and nothing on whether policies trained on denoised images
  tolerate the grainy ones, or whether X-VLA's end-effector actions work
  through mplib in closed loop.
- **No pretrained policy for TwoArmLift, TwoArmHandover, BiGym or ManiSkill3**
  was found; the hub searches are in `$P/hub/` and are not exhaustive.

About the environments:

- **The converted ACT checkpoints.** LeRobot's
  `migrate_policy_normalization` rewrote both checkpoints into the current
  format; that they behave as the hub files did is shown only by transfer
  cube's success matching its card.
- **Our own lock file.** gym-aloha resolved with our pins in a probe venv
  with the same numpy override and constraints as `pyproject.toml`; I did not
  run `uv lock` on the project (no changes there were allowed). Whether
  robomimic's pins coexist with LeRobot 0.6.1 in one environment was not
  tried.
- **The one robosuite mismatch**: which observation differed, and whether two
  rendering contexts in one process cause it.
- **RoboTwin's mplib expert**: whether the self-collisions it reports come
  from collision pairs that are not filtered, whether seeding mplib's planner
  makes it repeat, and how the experts do on more tasks and seeds. The
  randomised setting (11 GB of textures) was not downloaded.
- **Replay on the CPU across machines or library versions**: only one
  machine, one evening. The PyTorch thread count changes ACT's actions, so it
  belongs to an episode's identity on the CPU.

About speed:

- **A quiet machine.** Load averages were 3 to 21 in the run records and the GPU was shared with
  the other spike; every timing is an upper bound, good to perhaps 20 to 30%.
- **Throughput of a fleet**: one process per run, no batching, no parallel
  layout measured.

## Files

Scripts in this directory; outputs under `$P/runs/`, install logs under
`$P/logs/`, hub listings and model cards under `$P/hub/` (the API answer for
`robomimic/robomimic_models` in `robomimic_models.json`). Two downloads went
to the shared Hugging Face cache `~/.cache/huggingface/hub/` instead of
`$P`: the insertion checkpoint (about 200 MB) and the CLIP tokenizer files
robomimic loads (3.6 MB).

| File | Purpose |
|---|---|
| `aloha_eval.py` | ACT on gym-aloha, one episode per seed, success, timings, hashes, videos; `--top-only`, `--repeat`, `--threads`. |
| `aloha_summary.py` | Success and timing per run, and hash comparison between runs of one task. |
| `sapien_render.py` | SAPIEN offscreen rendering, rasterizer and ray tracer, timings and repeatability. |
| `robosuite_twoarm.py` | The three two-arm tasks: frames, same-seed checks, in-place reseed, step rates. |
| `robosuite_obs_diff.py` | Two same-seed environments in one process, first differing observation. |
| `robomimic_eval.py` | A robomimic checkpoint on robosuite, seeded per episode, hashes; `--ddim N`. |
| `robotwin_probe.py` | RoboTwin main branch, as shipped or with a stub for CuRobo: build, render, joint-space steps, same-seed check. |
| `robotwin_mplib_probe.py` | RoboTwin `RLinf_support` with mplib: expert per seed, observation time, end-effector control, same-seed repeats; `--patch-mplib`. |
| `bigym_probe.py`, `maniskill_probe.py` | Install-and-run checks for BiGym and ManiSkill3. |

| Run directory under `$P/runs/` | What |
|---|---|
| `tc_a`, `tc_b`, `ins_a`, `ins_b` | ACT, MuJoCo 3.8.1, three cameras, two processes per task |
| `tc_top`, `tc_t1`, `tc_repeat` | top camera only; one thread; carry-over within a process |
| `tc_mj337`, `tc_mj337_more`, `ins_mj337`, `ins_mj337_more` | ACT on MuJoCo 3.3.7, 200 seeds per task |
| `sapien/` | SAPIEN probes; `clean_*` are the timings in the table |
| `robosuite/twoarm`, `robosuite/obs_diff_repeats.log` | robosuite two-arm probe and repeats |
| `robomimic/ddpm_s*`, `ddpm_repeat`, `ddim10` | Transport diffusion policy |
| `robotwin/as_shipped`, `stub_handover`, `mplib_*` | RoboTwin runs |
| `bigym/pickbox`, `maniskill/twoarm` | BiGym and ManiSkill3 |
