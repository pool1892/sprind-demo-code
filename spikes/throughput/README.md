# Throughput: where rollout time goes and how far this machine scales

Measured on 2026-10-06 on this machine, in the probe
environment `~/.cache/robots-probes/libero-port` (robosuite 1.5.2 port, LeRobot
0.6.1, torch 2.11.0+rocm10.0.0, MuJoCo 3.8.1), with `lerobot/smolvla_libero` on
`libero_spatial`.

**Read every number with this in mind:** five other checks ran on the machine
at the same time. Its load average (roughly: how many CPU threads are busy)
was between 1 and 34 during my runs and changed from minute to minute. Each
table says what it was. Numbers taken under load are pessimistic; treat any
single number as good to about 20%. The measurements at the start of section 1
were taken on a nearly idle machine.

Terms used throughout:

- An **environment step** is one control step of the robot: 50 ms of simulated
  time, which is 25 physics steps of 2 ms with the arm controller run before
  each, followed by one image from each of the two cameras.
- An **action chunk** is what one inference call returns: 50 actions. Today the
  policy executes all 50 before it looks at the cameras again, so inference
  runs once every 50 environment steps.
- **GPU memory** on this machine is ordinary system memory handed to the
  graphics chip. The kernel calls it GTT. It is counted system-wide, it is not
  part of any process's own memory figure, and the machine is reported to hang
  when it fills its 31 GiB cap. I did not go near that.
- **Pss** is a process's memory with shared pages split fairly between the
  processes that share them, so the Pss of several workers can be added up.
- A **hard reset** rebuilds the MuJoCo model and the renderer at the start of
  every episode. It is LeRobot's default. A **soft reset** keeps both and only
  sets the state.

## Verdict

**Layout.** One inference process that holds the only copy of the policy on the
GPU, and eight simulation worker processes with one environment each. A worker
steps its environment on its own, sends its camera images to the inference
process only when it has used up its actions, and waits for the answer. The
inference process takes every request that is waiting at that moment and runs
them as one batch. An episode ends when the task succeeds, and the worker then
starts its next episode without waiting for anyone.

**Expected throughput.** A prototype of exactly this layout
([fleet_proto.py](fleet_proto.py)) ran 200 `libero_spatial` episodes at
**6,200 episodes per hour** in steady state and 5,460 per hour including
start-up, against about 570 per hour for `lerobot-eval` today. That is a factor
of ten. The success rate was 112 of 200 (56%) with the same per-task pattern as
the known runs (58 and 61 of 100), which is what one expects if the prototype
does the same work.

The number rests on these conditions, all measured in that run:

| condition | value |
|---|---|
| workers | 8, hard resets, memory returned to the system after each reset |
| actions executed per inference call | 50 |
| cameras fed to the policy | 2, rendered at 360x360 |
| number format | 32-bit floats |
| mean episode length | 185 steps (successful episodes 110, failed ones the full 280) |
| inference requests per episode | 4.2 |
| other load on the machine | load average 10 to 15 from other jobs |

Section 10 shows how to recompute it when one of these changes. The one that
matters most is the number of actions executed per inference call. With 10
instead of 50, four workers on tasks 0 to 3 reached 2,440 episodes per hour
with the inference process busy 89% of the time. The GPU is then the
bottleneck and more workers add little: the estimate for eight workers is
2,200 to 3,800 per hour, depending on how long the episodes are.

**Safe limits.**

- **Eight workers.** The layout's own footprint with eight workers is about
  28 GiB: 18 GiB of process memory (1.9 GiB per worker, 2.5 GiB for the
  inference process) and 10 GiB of GPU memory, which rose from 3.1 to
  13.2 GiB. During that run the machine's free memory fell from 38 GiB to
  12 GiB at the lowest point; other jobs' memory moved at the same time, so
  not all of that fall was this run's. Twelve workers would have needed about
  11 GiB more, which was not safely there while other projects used the
  machine.
- **Memory, not CPU, is what limits the worker count.** Each worker needs about
  2.8 GiB of system memory in total (1.9 GiB of its own plus 0.8 to 1.0 GiB of
  GPU memory for its textures). The CPU is not the limit: eight workers and
  the inference process together used about ten of the 32 CPU threads.
- **Never more than 12 rendering workers, on any day.** Sixteen took GPU memory
  to 19.6 GiB, too close to the cap that hangs the machine, and simulated no
  more steps than twelve did: rendering had reached its limit.
- **The heavy-job lock does not bound GPU memory.** Light jobs render too, at
  0.8 GiB per environment. At 17:58 system-wide GPU memory stood at 19.9 GiB
  with none of my jobs running: one heavy job and several light ones from the
  other checks. [gpu_mem_by_process.py](gpu_mem_by_process.py) lists who holds
  how much, without root. About 3 GiB is held permanently by the desktop's
  `voxtype-vulkan` daemon, which is not part of this project.
- **`tools/heavy-run` at its default 24 GB ceiling fits this layout** (18 GiB
  of process memory). The ceiling does not see GPU memory, so the fleet runner
  itself should refuse to start, or stop adding workers, when GTT use is above
  about 17 GiB or free memory below about 10 GiB.
- **One inference process.** Not because of the ROCm reports, which did not
  reproduce here (section 8), but because a second copy of the policy would
  cost memory and the two would slow each other down on the one GPU.
- **One environment per process, or at least one scene per process.** robosuite
  does not switch the GPU rendering context between environments in the same
  process. Two environments with different scenes in one process produce wrong
  images without any error (section 4, with a picture).

**What I would not do.** bfloat16 through PyTorch's automatic mixed precision:
on this GPU it is 20 to 30% slower than 32-bit floats and saves no memory
(section 6). Lower camera resolution to gain speed: it saves almost nothing
(section 2). More than eight workers to gain speed: the gain from eight to
twelve is small and it costs 11 GiB.

## Where the time goes today

`lerobot-eval` with ten environments in one process needed 66 s for ten
episodes of one task (543 episodes per hour, consistent with the known 570).
Section 3 has the table. In short:

- **Half of it is simulation that runs on one CPU core**, one environment after
  the other, and most of that simulation is thrown away: an environment whose
  episode has succeeded is reset and keeps running a new episode that nobody
  reads, until the slowest environment of the batch has finished.
- **A quarter is building ten environments** for every task.
- **Nearly a fifth is converting and preparing all ten environments' images on
  every step**, although the policy looks at them only every 50th step.
- **Inference is 6%.** It is called six times per batch of ten episodes.

So the policy is not what makes today's setup slow, and the GPU sits idle most
of the time.

## Measurements

### 1. One environment step

One environment alone in a process, no policy, 360x360, two cameras, small
random actions. Script: [profile_step.py](profile_step.py). Load average 0.8.

| part of one environment step | ms | share |
|---|---|---|
| physics (25 MuJoCo steps of 0.076 ms) | 1.90 | 31% |
| arm controller (25 calls of robosuite's Python controller, 0.09 ms each) | 2.26 | 37% |
| rendering, camera `agentview` | 0.95 | 16% |
| rendering, camera `robot0_eye_in_hand` | 0.61 | 10% |
| other sensors and observation bookkeeping | 0.28 | 5% |
| everything else (success check, wrapper, Python in between) | 0.10 | 2% |
| **total** | **6.09** | 164 steps per second |

Rendering time is almost entirely waiting for the graphics chip to hand the
picture back; the draw call itself returns after 0.04 ms.

Per episode a worker also pays, outside the steps:

| | seconds |
|---|---|
| building the environment for a new task (once per task) | 1.3 to 2.0 |
| hard reset, including the 10 settle steps LeRobot adds | 0.57 |
| soft reset, including the settle steps | 0.09 to 0.17 |

A full 280-step episode therefore costs one worker about 1.7 s of stepping and
0.6 s of reset.

Thread settings (`OMP_NUM_THREADS=1` and the like) made no difference for a
single environment: 5.95 ms per step with the defaults, 6.29 ms with one
thread, which is within the noise of that moment (load 6).

### 2. Camera resolution

The policy's configuration lists its inputs as 256x256, presumably the size of
its training images. LeRobot's LIBERO environment renders 360x360 by
default. Whatever arrives, the policy scales it to 512x512 before its vision
encoder, so the encoder's work does not depend on the rendering resolution.

Same script, one environment, all five resolutions in one run. Load 9 to 12,
so compare the rows with each other, not with section 1.

| resolution | rendering both cameras, ms | whole step, ms |
|---|---|---|
| 128x128 | 1.65 | 6.74 |
| 256x256 | 1.83 | 7.08 |
| 360x360 | 1.78 | 6.47 |
| 512x512 | 2.35 | 8.02 |
| 720x720 | 2.45 | 7.64 |

Rendering at 256 instead of 360 saves nothing measurable. The cost of a frame
is dominated by fixed work per frame (275 objects to draw, one round trip to
the graphics chip), not by the number of pixels. Rendering at the training
resolution may still be worth trying for the success rate; that is the
baseline check's question, not a throughput one. Not measured: whether lower
resolution raises the graphics chip's limit when many workers render at once.

### 3. Today's loop, stage by stage

LeRobot's own rollout loop, repeated call for call with a timer around each
stage. Ten environments of task 0, ten episodes. Script:
[eval_breakdown.py](eval_breakdown.py). Load 4 at the start, 16 at the end.

| stage | seconds | share |
|---|---|---|
| stepping the ten environments (`env.step`) | 31.8 | 48% |
| building the ten environments at the first reset | 15.7 | 24% |
| converting observations to tensors, on every step | 8.4 | 13% |
| model inference: 6 calls of 656 ms for a batch of ten | 3.9 | 6% |
| LIBERO observation processing (flip images, build state), every step | 3.0 | 5% |
| collecting video frames | 1.8 | 3% |
| policy preprocessing (tokenize, copy to GPU), every step | 1.0 | 2% |
| the rest | 0.6 | 1% |
| **total** | **66.4** | 543 episodes per hour |

The stepping line needs explaining, because 31.8 s over 280 loop iterations is
114 ms per iteration, far more than ten steps of 6 ms. The ten episodes needed
1,101 environment steps in total up to their ends. The loop simulated about
2,800: in this configuration (`--eval.use_async_envs=false`) an environment
that has finished is reset on the next iteration, which is a hard reset of
half a second or more, and then steps through a new episode that the
evaluation ignores, until the last environment of the batch is done or 280
iterations have passed. LeRobot has a wrapper that freezes a finished
environment, but it is only applied on the asynchronous path, the one that
crashed here. [sync_autoreset_check.py](sync_autoreset_check.py) shows the
mechanism directly: of two environments built the way `lerobot-eval` builds
them, the one that reported "finished" after its fifth step was reset a second
time and had simulated 29 steps after 30 iterations.

Inference is called once every 50 steps: six times for a batch that runs the
full 280 steps, each call serving all ten environments. In the layout
recommended here, where an episode stops at success, it was 4.2 requests per
episode on average.

The process used 4.4 CPU cores on average although its work is sequential.
Several of its threads sat at about a quarter of a core each, which I take to
be PyTorch's CPU threads spinning between the small tensor operations of every
step; I did not confirm that. It held 5.8 GiB of its own memory. Its GPU
memory was not measured in isolation; from section 5 it should be about 7 GiB
(ten rendering contexts and the policy).

### 4. Simulation scaling

N worker processes, one environment each, all stepping at once, no policy.
Script: [sim_scaling.py](sim_scaling.py).

Each worker that builds its own environment needs 1.5 to 3 GiB (section 5), so
16 to 32 independent workers do not fit into this machine's memory. To measure
the scaling anyway the script has a second mode: the parent process builds one
environment without a renderer and forks the workers, which then share the
model's memory, and each worker opens its own rendering context and renders
both cameras after every step. The work per step is the same, the memory is a
fraction. One forked worker alone does 143 steps per second in that run
against 164 in section 1, most likely because of the load on the machine (29
at that moment).

**With rendering, 360x360, two cameras.** Load 16 to 29 from other jobs.

| workers | steps per second, all workers | ms per step in a worker | of which physics | controller | rendering | GPU busy | GPU memory in use |
|---|---|---|---|---|---|---|---|
| 1 | 143 | 6.98 | 1.94 | 2.44 | 2.19 | 23% | 6.9 GiB |
| 2 | 282 | 7.08 | 1.99 | 2.43 | 2.32 | 27% | 5.4 GiB |
| 4 | 518 | 7.73 | 2.14 | 2.66 | 2.51 | 38% | 7.0 GiB |
| 8 | 814 | 9.83 | 2.61 | 3.45 | 3.20 | 59% | 10.9 GiB |
| 12 | 1,052 | 11.41 | 2.97 | 3.90 | 3.88 | 76% | 16.1 GiB |
| 16 | 1,050 | 15.24 | 3.69 | 4.96 | 5.67 | 79% | 19.6 GiB |
| 24, 32 | not run: GPU memory would have passed 20 GiB | | | | | | |

GPU memory includes what other jobs held at the time; each worker added 0.8
to 1.0 GiB.

**Without rendering** (physics and controller only), to see the CPU alone.
Load 22 to 34.

| workers | steps per second, all workers | ms per step in a worker |
|---|---|---|
| 1 | 188 | 5.32 |
| 2 | 426 | 4.69 |
| 4 | 925 | 4.32 |
| 8 | 1,772 | 4.51 |
| 12 | 2,577 | 4.66 |
| 16 | 3,016 | 5.30 |
| 24 | 3,406 | 7.05 |
| 32 | 3,856 | 8.30 |

What this shows:

- **The CPU scales well.** Without rendering, throughput grows in proportion
  up to 16 workers, one per core. The second thread of each core adds another
  28% at 32 workers.
- **Rendering on the shared graphics chip becomes the limit at about 1,050
  environment steps per second**, which is about 2,100 camera images per
  second, reached with 12 workers. Sixteen workers produce no more. Rendering
  time per step grows from 2.2 ms with one worker to 5.7 ms with sixteen.
- **Replays that need no images** are therefore two to three times cheaper per
  step at eight workers and more, if the replay can skip the cameras.
- **A worker is one busy thread.** With rendering it uses 0.9 of a core, the
  rest being time blocked on the graphics chip. Whether thread settings
  (`OMP_NUM_THREADS` and the like) matter for eight independent workers was
  not measured; for one alone they do not (section 1).

**Several environments in one process** is what `lerobot-eval` does. Script:
[multi_env_one_process.py](multi_env_one_process.py). Stepping is still one
after the other, so it adds no throughput, but two results matter:

- Memory: one environment in a process 2.1 GiB, two 2.7 GiB, four 3.6 GiB. A
  further environment of the same task costs 0.45 to 0.65 GiB of process
  memory and 0.53 GiB of GPU memory, well under half of a separate process.
- Correctness: robosuite makes a rendering context current only when it creates
  it. With two environments of different scenes in one process, the first is
  drawn inside the second one's context. Same process, same simulation state,
  same camera, the only change being which context is current: 75% of the
  pixels differ, and the picture shows the other scene's objects and textures
  at this scene's positions (`~/.cache/robots-probes/throughput/glctx_compare.png`,
  left correct, right what a step returns). `lerobot-eval` gets away with it
  because all environments of a batch share one task. A fleet with mixed tasks,
  or with faults that change textures or meshes, must not put different scenes
  into one process without switching the context itself.

### 5. Memory per worker

| what | own memory (Pss) | GPU memory | script |
|---|---|---|---|
| worker, soft resets, over 150 resets | 1.71 to 1.87 GiB, flat | 0.8 to 1.0 GiB | [reset_memory.py](reset_memory.py) `--soft` |
| worker, hard resets, over 300 resets | 2.6 to 3.2 GiB, no upward trend | 0.8 to 1.0 GiB | [reset_memory.py](reset_memory.py) |
| worker, hard resets, garbage collected and freed memory returned after each reset | 1.93 GiB mean, 1.99 GiB highest of eight | 0.8 to 1.0 GiB | [fleet_proto.py](fleet_proto.py) `--trim` |
| worker, soft resets, in the prototype (peak over a run of 10 or 20 episodes) | 1.36 and 1.70 GiB mean in the two runs | 0.8 to 1.0 GiB | [fleet_proto.py](fleet_proto.py) `--soft` |
| inference process, policy loaded | 2.5 to 3.2 GiB | 1.2 GiB, plus the batch (section 6) | [infer_bench.py](infer_bench.py) |
| `lerobot-eval`, ten environments and the policy in one process | 5.8 GiB | not measured in isolation, about 7 GiB expected | [eval_breakdown.py](eval_breakdown.py) |

Hard resets do not leak: over 300 resets the memory moved up and down between
2.6 and 3.2 GiB and ended where it started (+0.01 MiB per reset). They are
bigger than soft resets because the old model and renderer are only freed by
a garbage-collector pass and the C allocator keeps the freed pages. Collecting
and returning them after each reset brings a hard-reset worker down to the
size of a soft-reset one, at the price of a slower reset (section 9).

Where a worker's memory goes: 0.5 to 0.7 GiB is Python with robosuite, LIBERO,
LeRobot and PyTorch imported, 0.26 GiB is the compiled MuJoCo model (0.18 GiB
of that textures, three of them 4096x4096), and about 0.9 GiB more is
allocated while the scene is built and never given back. The GPU memory is the
same textures uploaded: 0.53 GiB for each rendering context, plus about
0.3 GiB for the first context of a process.

### 6. Inference by batch size and number format

One call produces one chunk of 50 actions for every environment in the batch.
Times are wall-clock with the GPU's queue flushed before and after each call.
Script: [infer_bench.py](infer_bench.py). Two cameras.

**32-bit floats, batch 1 to 64.** Load 14 to 28.

| batch | ms per call | of which images and language | of which 10 denoising steps | ms per environment | PyTorch's peak GPU memory | GPU memory in use, system-wide |
|---|---|---|---|---|---|---|
| 1 | 107 | 34 | 74 | 107 | 0.97 GiB | 5.9 GiB |
| 2 | 141 | 63 | 78 | 71 | 0.99 | 5.3 |
| 4 | 219 | 115 | 104 | 55 | 1.05 | 5.4 |
| 8 | 368 | 199 | 169 | 46 | 1.16 | 5.6 |
| 12 | 530 | 308 | 222 | 44 | 1.27 | 5.8 |
| 16 | 692 | 402 | 290 | 43 | 1.38 | 5.9 |
| 24 | 1,076 | 617 | 459 | 45 | 1.61 | 6.3 |
| 32 | 1,431 | 823 | 608 | 45 | 1.83 | 6.5 |
| 48 | 2,182 | 1,274 | 908 | 45 | 2.28 | 8.0 |
| 64 | 3,234 | 1,862 | 1,372 | 51 | 2.72 | 8.6 |

A good fit is **65 ms plus 43 ms per environment**. Batching stops paying at
about eight: beyond that every environment costs the same 44 to 46 ms. The
GPU's capacity is therefore about 22 action chunks per second, which at 50
actions per chunk is 1,100 environment steps per second, the same order as
what the simulation can deliver. Memory is no concern at any batch size
tried: the policy has 450 million parameters and the batch adds little.

The system-wide column includes other jobs' GPU memory (it started at 4.5 GiB
before the policy was loaded), so only its growth is meaningful.

On top of each call the inference process spends CPU time to turn the images
into tensors and tokenize the instruction: 8 to 45 ms in this run, 35 ms per
call in the prototype. Turning the chunk back into robot actions takes under
1 ms.

**32-bit floats against bfloat16 and float16**, each batch size measured back
to back so the three see the same load (14 to 17). bfloat16 and float16 here
mean PyTorch's automatic mixed precision, which is what
`--policy.use_amp=true` switches on (LeRobot's switch picks float16).

| batch | fp32, ms per call | bfloat16 | float16 | largest difference of a bfloat16 action from fp32 | PyTorch's peak GPU memory, fp32 / bfloat16 |
|---|---|---|---|---|---|
| 1 | 112 | 116 | 157 | 0.03 | 0.97 / 0.97 GiB |
| 4 | 268 | 279 | 292 | 0.03 | 1.05 / 1.05 |
| 8 | 388 | 467 | 495 | 0.03 | 1.16 / 1.17 |
| 16 | 745 | 903 | 947 | 0.03 | 1.38 / 1.41 |
| 32 | 1,497 | 1,960 | 1,766 | 0.04 | 1.83 / 1.88 |

The half-precision formats are slower, not faster, on this GPU with this
software, and they save no memory, because mixed precision keeps the weights
in 32 bits. The action differences are in the policy's normalized units, where
actions have a standard deviation of one. Stay with 32-bit floats. Not tried:
converting the weights themselves to bfloat16.

**Three cameras instead of two**, which is what the checkpoint expects and
what the baseline fix may introduce. The third image was a copy of the first.
Load 9.

| batch | 2 cameras, ms per call | 3 cameras, ms per call | ms per environment with 3 |
|---|---|---|---|
| 8 | 388 | 522 | 65 |
| 16 | 745 | 1,058 | 66 |

A third camera makes inference about 40% more expensive, almost all of it in
the image part. It would also add a third render per step, 0.6 to 1 ms going
by the two cameras of section 1; that was not measured.

### 7. Rendering and inference on the same GPU

Eight rendering simulation workers and the policy at batch eight, in phases of
eight seconds, back to back. Script: [contention.py](contention.py). Load 15
to 17.

| phase | environment steps per second | ms per inference call | GPU busy | CPU used by the inference process |
|---|---|---|---|---|
| simulation only | 824 | | 59% | 0.01 cores |
| inference only | | 437 | 89% | 1.03 cores |
| both at once | 496 | 610 | 97% | 1.00 cores |
| simulation only, again | 618 | | 81% | 0.00 cores |
| inference only, again | | 397 | 94% | 1.04 cores |
| everything idle, policy loaded | | | 11% | 0.00 cores |

Yes, in both directions. With both running flat out, inference takes about
1.5 times as long and the simulation loses 20 to 40% of its steps. The two
"simulation only" rows differ by a quarter between themselves, which is the
other jobs on the machine, so the size of the effect is uncertain; its
existence is not. In the real layout the two overlap only part of the time,
because a worker that waits for actions does not render. The prototype's
calls took about 1.35 times as long as the same batch sizes in section 6.

### 8. The two ROCm reports

Script: [rocm_idle.py](rocm_idle.py). Child processes initialise the GPU
through PyTorch, run one small computation and then do nothing for ten
seconds, while the parent reads their consumed CPU time and the kernel's GPU
busy figure. Load 4 to 8; other jobs were rendering at the time.

| children | CPU used by each while idle | GPU busy, mean |
|---|---|---|
| none | | 4% |
| one plain Python process (control) | 0.0 cores | 9% |
| one process with PyTorch on the CPU only (control) | 0.0 cores | 11% |
| one process with ROCm initialised | 0.0 cores | 16% |
| two processes with ROCm initialised | 0.0, 0.0 cores | 14% |
| three processes with ROCm initialised | 0.0, 0.0, 0.0 cores | 5% |
| none, again | | 11% |

**Report 1, "every process that loads the ROCm runtime keeps one CPU core
spinning": not true here for an idle process.** An idle process with ROCm
loaded uses no CPU, and so does an inference process with the policy loaded
that is waiting for work (last row of section 7). What is true: while a
process is running inference it uses one full CPU core for the whole duration
of each call (1.0 cores in section 7). Whether that thread is computing or
waiting for the GPU in a busy loop I did not determine. Plan one core for the
inference process.

**Report 2, "two or more such processes keep the GPU pinned": not true here.**
GPU busy does not rise with the number of idle ROCm processes: 16%, 14% and 5%
with one, two and three, against 4% and 11% with none. The differences are
other jobs rendering at the time.

Both tests cover idle processes over ten seconds. Two processes that run
inference at the same time were not tested.

Neither report is a reason for the single inference process. Memory and the
mutual slowdown of section 7 are.

### 9. The layout, end to end

Script: [fleet_proto.py](fleet_proto.py). Eight workers and the inference
process. The workload is `lerobot-eval`'s: every task of `libero_spatial`,
episodes 0 to 9 with seeds 1000 to 1009 and LIBERO's stored initial states 0
to 9. A worker takes a task, builds its environment, runs the ten episodes and
takes the next task.

| run | episodes | episodes per hour, steady state | including start-up | success | mean batch | ms per call | worker: step / wait / reset / build | ms per step | reset, s | worker memory (Pss) | GPU memory peak | load |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 workers, hard resets with memory return, 50 actions per call, all ten tasks twice | 200 | **6,221** | 5,458 | 112/200 | 1.5 | 168 | 40% / 30% / 22% / 6% | 9.81 | 1.12 | 1.93 GiB | 13.2 GiB | 10 to 15 |
| 4 workers, soft resets, 10 actions per call, tasks 0 to 3 | 40 | 2,440 | 1,880 | 38/40 | 1.4 | 158 | 24% / 70% / 2% / 2% | 10.03 | 0.10 | 1.36 GiB | 12.2 GiB | 5 to 19 |
| 4 workers, soft resets, 50 actions per call, tasks 0 to 3 | 40 | 3,907 | 2,945 | 31/40 | 1.2 | 246 | 45% / 46% / 4% / 4% | 12.80 | 0.16 | 1.70 GiB | 13.3 GiB | 19 to 23 |

The GPU memory column is the system-wide peak and includes other jobs; in the
first run it started from 3.1 GiB, in the third from 9.1 GiB.

The two four-worker runs are what fitted when the machine briefly had memory
free. Eight workers with soft resets, eight workers with 10 actions per call
and a run with fewer PyTorch threads in the inference process were prepared
and queued but did not run: the queue for the GPU was long, and when their
turn came the other jobs had left less free memory than they need. The
commands are at the end of this file.

"Steady state" is the rate over the middle 80% of episode completions, which
leaves out loading the policy, building the first environments, and the tail
where some workers have run out of tasks. Success by task in the first run:
18, 15, 14, 15, 6, 0, 18, 9, 5, 12 of 20, against 8, 5, 6, 9, 6, 0, 8, 6, 3, 7
of 10 in the known `lerobot-eval` run on the same port. The episodes are not
identical to that run, because the policy's sampling noise depends on how
requests are batched.

What the first run shows about the layout:

- **A worker spends 40% of its time stepping, 30% waiting for actions, 22% in
  resets and 6% building environments.** A step takes 9.8 ms with eight
  workers, against 6.1 ms for one alone on an idle machine.
- **The inference process is busy 91% of the time with batches of only 1.5 on
  average.** Requests seldom pile up, because a call for one or two
  environments is over in 170 ms. Small batches are the expensive kind (112 ms
  per environment instead of 46), so the GPU is working at well under half of
  what it could deliver. I expect that holding a call back to collect more
  requests would make the GPU's work cheaper without shortening anyone's wait,
  and so would not raise throughput at this worker count. The script has the
  option (`--linger-ms`); it was not run.
- **The hard reset with memory return costs 1.1 s per episode**, against
  0.57 s for a plain hard reset alone on an idle machine and 0.69 s in the
  300-reset run under load.
- **The inference process used 4.9 CPU cores**, one for the call itself
  (section 8) and the rest probably PyTorch's CPU threads, as in section 3.

What the two four-worker runs add:

- **With 10 actions per call the GPU is the bottleneck.** A worker waits for
  actions 70% of its time, and four workers already keep the inference process
  busy 89% of the time. More workers would mostly wait.
- **Soft resets take 0.10 to 0.16 s** against 1.12 s, and a soft-reset worker
  stays at 1.4 to 1.7 GiB.
- **The third run was disturbed by other jobs**: its inference calls took
  246 ms at a mean batch of 1.2, twice what section 6 gives, and a step took
  12.8 ms with only four workers. Its 3,907 per hour is therefore low.
- **A side observation for the baseline check, from 40 episodes each:** on
  tasks 0 to 3, executing 10 actions per call succeeded 38 times of 40, and
  50 actions per call 31 times of 40 (62 of 80 in the first run, with hard
  resets). Episodes were also shorter, 102 steps against 144 on average. A
  difference of that size comes up by chance about once in forty tries, so it
  is worth a proper test and not yet a result.

### 10. Recomputing for another policy or setting

A worker's episode costs

    reset + steps x step time + requests x wait

with `requests = steps / actions per call`, rounded up, and `wait` the time
from sending an observation to receiving the chunk. The fleet does
`workers / episode time` episodes per second. The wait is at least one
inference call for the batch the request lands in, and about half a call more
when the inference process is busy.

Component times to plug in, all from the sections above:

| component | value | changes with |
|---|---|---|
| environment step in a worker | 6.1 ms alone, 9.8 ms with 8 workers, 11.4 ms with 12 | worker count, cameras (0.7 ms each) |
| reset | 0.57 s hard, 1.1 s hard with memory return, 0.15 s soft | |
| building an environment | 1.3 to 2.6 s, once per task change | |
| steps per episode | 110 when the task succeeds, 280 when it fails (libero_spatial) | suite, policy quality |
| inference call | 65 ms + 43 ms per environment with 2 cameras, about 65 ms per environment with 3 | policy, cameras |
| slowdown of a call while workers render | 1.35 | worker count |
| CPU work per call (images to tensors, tokenizer) | 35 ms | |

[estimate.py](estimate.py) solves this, including the dependence of the wait
on how busy the inference process is:

    python spikes/throughput/estimate.py --workers 8 --step-ms 9.8 --reset-s 1.27 --steps 185

| case | estimate, episodes per hour | measured |
|---|---|---|
| 8 workers, hard resets with memory return, 50 actions per call, 185 steps per episode | 6,600 | 6,221 |
| 4 workers, soft resets, 10 actions per call, 102 steps per episode | 2,800 | 2,440 |
| 4 workers, soft resets, 50 actions per call, 144 steps per episode (the disturbed run) | 4,900 | 3,907 |
| 8 workers, soft resets, 50 actions per call, 185 steps | 7,800 | not run |
| 8 workers, hard resets, a policy that succeeds 90% of the time (127 steps) | 8,400 | not run |
| 8 workers, hard resets, 25 actions per call, 185 steps | 4,400 | not run |
| 8 workers, soft resets, 10 actions per call, 185 steps | 2,200 | not run |
| 8 workers, soft resets, 10 actions per call, 102 steps | 3,800 | not run |
| 12 workers, hard resets, 50 actions per call (11.4 ms per step) | 8,600 | not run, no memory for it |

For the measured cases the estimate uses each run's own step and reset times,
so it tests only the part of the model that predicts the waiting. It was 6%,
14% and 26% too high. The model is optimistic, the more so the more other
jobs use the GPU, which it knows nothing about. Read the predictions as upper
values and take 20 to 25% off for a shared machine. With 10 or 25 actions per
call the inference process is busy all the time, and the result depends on
how requests happen to be batched more than on anything else.

A third camera makes each call about 40% more expensive (section 6). At 50
actions per call, where waiting for actions is 30% of a worker's time, that
should cost roughly a tenth of the throughput. At 10 actions per call it
would cost close to the full 40%.

## Surprises

1. **Simulation, not the policy, is the bottleneck.** I expected the 450
   million parameter policy on an integrated GPU to dominate. It needs 46 ms
   per environment every 50 steps; the simulation needs 300 ms for those 50
   steps.
2. **Most of what `lerobot-eval` simulates in this configuration is discarded**
   (section 3).
3. **A worker costs 2.8 GiB of system memory**, a third of it GPU memory for
   textures, so memory caps the fleet at eight to twelve workers long before
   the 32 CPU threads do.
4. **The arm controller, plain Python, is the largest part of a step** (37%),
   larger than the physics.
5. **Half precision is slower** on this GPU.
6. **Wrong images without an error** when different scenes share a process
   (section 4).
7. **Both ROCm reports did not reproduce.**
8. **Fewer actions per inference call may buy success at the price of
   throughput** (section 9, small sample).

## Not verified

- **Quiet-machine numbers for sections 4, 6, 7 and 9.** All were taken while
  other jobs loaded the CPU and the GPU. I expect them to be better on an idle
  machine, by an amount I did not measure.
- **Whether soft resets give the same episodes as hard resets.** They are
  faster (0.15 s against 0.57 s) and smaller (1.8 GiB against 2.6 to 3.2 GiB),
  and LeRobot's own comment warns that observations after the settle steps
  can differ. The one soft-reset run with 50 actions per call succeeded 31
  times of 40 on tasks 0 to 3, the same rate as the hard-reset run on those
  tasks (62 of 80), which shows no gross difference and nothing finer. This
  belongs to the replay check.
- **Eight workers with soft resets, and eight workers with 10 actions per
  call.** Only the four-worker versions ran.
- **Twelve workers end to end.** There was not enough free memory to try.
- **Long runs.** The longest prototype run was about two minutes, the longest
  single-worker run 300 resets. Nothing here says the layout survives ten
  hours, and ROCm's long-run stability is still open.
- **Whether `tools/heavy-run`'s memory ceiling counts GPU memory.** I assume
  not, from how the kernel accounts it, and did not test it.
- **Other suites and other policies.** Everything is `libero_spatial` with
  SmolVLA. `libero_10` episodes are up to 520 steps and its scenes differ, so
  step time and memory per worker will differ.
- **Recording.** No run wrote videos, images or logs to disk. The fleet will,
  and that costs time per step that is not in any number here.
- **Two environments per worker process**, stepped alternately so that one
  runs while the other waits for actions. From section 4 it should use the 30%
  waiting time at 0.5 GiB instead of 2.8 GiB per extra environment. Not tried.
- **Why the inference process and `lerobot-eval` use four to five CPU cores.**
  I saw several threads at about a quarter of a core each and take them to be
  PyTorch's CPU threads spinning between small operations. Limiting them
  (`torch.set_num_threads(1)`) should free those cores. Prepared, not run.
- **Thread settings with eight independent workers.** For one environment
  alone they made no difference, and the prototype's workers ran with one
  thread each. The comparison at eight workers was prepared, not run.
- **The exact number of discarded steps in today's loop.** Section 3 has the
  mechanism and the totals; the instrumented repeat that counts simulated
  steps and resets was prepared, not run.
- **Lower resolution at high worker counts**, and textures scaled down to save
  GPU memory. Both change the images the policy sees.

## Scripts

All run through `spikes/throughput/run.sh SCRIPT.py ARGS`, which sets the probe
environment's variables, wrapped in `tools/heavy-run` (GPU, or more than four
simulation processes) or `tools/heavy-run --light`. Raw results are JSON files
in `~/.cache/robots-probes/throughput/`, one per run, named after the script.

| script | what it measures |
|---|---|
| [profile_step.py](profile_step.py) | parts of one environment step, reset times, at any resolution |
| [eval_breakdown.py](eval_breakdown.py) | stages of LeRobot's own rollout loop |
| [sim_scaling.py](sim_scaling.py) | simulation throughput and memory against worker count |
| [multi_env_one_process.py](multi_env_one_process.py) | several environments in one process: cost, memory, and the wrong-image check (`--mixed`) |
| [reset_memory.py](reset_memory.py) | memory of one worker over many resets |
| [infer_bench.py](infer_bench.py) | inference time and GPU memory against batch size and number format |
| [contention.py](contention.py) | rendering and inference at the same time |
| [rocm_idle.py](rocm_idle.py) | the two ROCm reports |
| [fleet_proto.py](fleet_proto.py) | the recommended layout, end to end |
| [estimate.py](estimate.py) | expected episodes per hour from component times |
| [sync_autoreset_check.py](sync_autoreset_check.py) | whether `lerobot-eval` keeps simulating finished environments |
| [gpu_mem_by_process.py](gpu_mem_by_process.py) | which process holds how much GPU memory, without root |
| [common.py](common.py), [policy_setup.py](policy_setup.py), [run.sh](run.sh) | shared helpers |

## Prepared but not run

These need the GPU lock and 25 to 30 GiB of free memory, which were not
available together before I stopped. Run them from the repository root on a
quiet machine; each takes one to three minutes.

```sh
# Eight workers: soft resets, 10 actions per call, fewer PyTorch threads in the inference process
tools/heavy-run spikes/throughput/run.sh fleet_proto.py --workers 8 --rounds 1 --soft
tools/heavy-run spikes/throughput/run.sh fleet_proto.py --workers 8 --rounds 1 --soft --n-action-steps 10
tools/heavy-run spikes/throughput/run.sh fleet_proto.py --workers 8 --rounds 1 --soft --server-threads 1

# Today's loop again, counting simulated steps and resets, then with one CPU thread
tools/heavy-run spikes/throughput/run.sh eval_breakdown.py --tasks 0 --tag _instrumented
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/heavy-run spikes/throughput/run.sh eval_breakdown.py --tasks 0 --threads 1 --tag _1thread

# Thread settings with eight independent workers
tools/heavy-run spikes/throughput/run.sh sim_scaling.py --mode spawn --soft --workers 8 --tag _threads_default
tools/heavy-run spikes/throughput/run.sh sim_scaling.py --mode spawn --soft --workers 8 --threads 1 --torch-threads 1 --tag _threads_1

# Does lower resolution lift the graphics chip's limit at 12 workers?
for r in 360 256 128; do tools/heavy-run spikes/throughput/run.sh sim_scaling.py --mode fork --res $r --workers 12 --gtt-stop 16 --tag _res; done

# The whole scaling curve and the inference table again, on a quiet machine
tools/heavy-run spikes/throughput/run.sh sim_scaling.py --mode fork --workers 1 2 4 8 12 16 --gtt-stop 17
tools/heavy-run spikes/throughput/run.sh infer_bench.py --batches 1 2 4 8 16 32 --amp fp32 bf16
```
