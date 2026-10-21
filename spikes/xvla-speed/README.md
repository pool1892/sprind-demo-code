# X-VLA on this machine: where a call's time goes, and how to get 3,000 faulted episodes per hour

Spike run on this machine in the night of 2026-10-06 to 07. The walking skeleton
found that with X-VLA the graphics chip is the bottleneck: 1,577 healthy or
1,074 faulted episodes per hour, where medium-sized worlds need about 2,100 at
faulted-fleet success rates. This spike measures where the time goes, what
makes it faster, and what the fleet then delivers end to end.

Everything below was measured here unless it says otherwise; what I conclude
from the measurements is marked as such. Simulator and policy ran in the
baseline spike's environment (`~/.cache/robots-probes/baseline-venv-mj337`:
MuJoCo 3.3.7, robosuite 1.5.2 port with the absolute-control correction,
LeRobot 0.6.1, PyTorch 2.11.0 on ROCm 10.0.0). The policy is
`lerobot/xvla-libero` revision `12e8783e`. **The machine was never quiet:**
another agent's jobs ran throughout, the load average (roughly: busy CPU
threads) was between 0.5 and 7, and each table says what it was.

Words used throughout:

- An **episode** is one attempt at a `libero_spatial` task, at most 280 control
  steps of 50 ms. A **chunk** is what one call of the policy returns: the next
  30 actions. A **row** is one episode's place in a batch, when the policy
  computes chunks for several episodes in one **call**.
- **Identical** always means bit for bit, compared through hashes or byte by
  byte, never with a tolerance.
- **32-bit floats** (float32) are the number format the checkpoint is stored
  in and the skeleton ran in: about seven significant digits. **float16** has
  about three and a half digits and no numbers above 65,504. **bfloat16** has
  float32's range and about two and a half digits.
- **Today's** settings are the walking skeleton's: 32-bit floats, 10 sampling
  steps, batches of eight with fixed rows, `cudnn.benchmark` switched on.
- A **sampling step** is one pass of X-VLA's action transformer; ten of them
  turn random noise into a chunk.
- **MIOpen** is the library PyTorch uses for convolutions on the Radeon (the
  counterpart of NVIDIA's cuDNN). It has several algorithms per convolution
  and keeps a **database** of which was fastest.
- **TFLOPS**: 10^12 floating-point operations per second, the usual measure
  of how fast a chip multiplies matrices.

## Verdict

**With the model in 16-bit floats one call is 3.3 times faster, and the fleet
runs about 4,900 healthy or 3,100 faulted episodes per hour, end to end with
video, under the same replay rules. Medium worlds fit, with half as much again
to spare.**

| 200 episodes, end to end, both videos and all tables written | Today (skeleton) | 32-bit, new batching | **16-bit, recommended** (`fp16_res32`) | 16-bit, plain (`fp16`) |
|---|---|---|---|---|
| Healthy episodes per hour (in steady state) | 1,577 (1,683) | 1,856 (1,952) | **4,962** (5,753) | 4,850 (5,529) |
| Faulted mix per hour (in steady state) | 1,074 (1,117) | 1,154 (1,196) | **3,093** (3,378) | 3,170 (3,447) |
| Success, healthy 200 | 195 | 194 | 195 | 195 |
| Success, faulted mix | 92 of 120 (77%) | 140 of 200 (70%) | 138 of 200 (69%) | 137 of 200 (68.5%) |
| Chunks per episode, healthy / mix | 4.2 / 6.4 | 4.2 / 7.0 | 4.2 / 7.0 | 4.2 / 7.0 |
| One call inside the fleet | 3.1 s for 8 rows | 1.66 s for 4 rows | **0.54 s for 4 rows** | 0.51 s for 4 rows |
| Share of rows in use | 87% and 91% | 99% | 98% | 98 to 99% |
| Workers | 8 | 10 | 10 | 10 |
| Peak memory of the run's processes | 17.5 GiB | 20.8 to 22.3 GiB | 20.8 GiB | 20.8 to 22.6 GiB |
| Peak GPU memory: the run's processes / whole machine | 10.4 / 13.4 GiB | 10.2 / 13.3 GiB | 9.4 / 12.4 GiB | 9.4 / 12.5 GiB |
| Least memory available on the machine during the run | 22 GiB | 21.7 GiB | 21.2 GiB | 22.6 GiB |

The faulted mix is the skeleton's two faults at the skeleton's levels: 40
healthy episodes, 80 with the scene camera turned by 6 degrees, 80 with
actions executed 17 steps late. The skeleton's own faulted run had 120
episodes in equal thirds, so it is a little easier (6.4 chunks per episode
against 7.0); the three right-hand columns are the same 200 episodes. The two
16-bit columns do not differ by more than the machine's load did between
their runs.

**1. Recommended inference settings** (`fp16_res32` in `fleet.py`).

- *All weights in float16, converted once when the model is loaded.* Not
  PyTorch's "automatic mixed precision", which keeps 32-bit weights and
  converts them at every use and is slower (section 2).
- *In the action transformer, each token's running sum and the layer norms
  stay in 32 bits*; only the two sub-layers of each of its 24 layers, the
  attention and the MLP, compute in 16. This costs 9% of a call against
  converting everything, costs nothing measurable end to end, and brings the
  commanded positions four times nearer to today's.
- *Every linear layer's weight stored transposed.* Needed for the batching
  rule below: PyTorch's usual way of calling the matrix product gives, in 16
  bits and for some layer shapes, a row another result in another row of the
  batch. Stored transposed it does not, and it is 7% faster (section 3).
- *`torch.backends.cudnn.benchmark = False`, with a convolution database that
  was searched once on a quiet machine and is read-only afterwards.* Needed
  for exact replay whatever the number format: today each process picks its
  convolution algorithms by timing them at start-up (my reading of the
  measurements in section 4), and another algorithm gives other bits.
- 10 sampling steps and 30 actions per chunk, as stored.

One call for four rows then takes 0.48 s alone and 0.54 s inside the fleet,
against 1.56 s and 1.66 s in 32 bits.

*Is it still the same policy?* Not bit for bit. On 24 recorded observations
the commanded positions differ from today's by up to 0.3 mm and the rotations
by up to 0.0014 rad; no gripper command differs. (For plain `fp16`: 1.2 mm
and 0.003 rad.) In closed loop, the same episodes in 32 bits and in 16:

| Same episodes | 32-bit | **Recommended** | Plain `fp16` | Episodes that succeed in one format only: 32-bit / recommended | 32-bit / plain |
|---|---|---|---|---|---|
| 100 healthy (10 per task, start states 0 to 9) | 100 | 100 | 100 | 0 / 0 | 0 / 0 |
| The skeleton's 200 healthy (32-bit: the skeleton's run) | 195 | 195 | 195 | 0 / 0 | 1 / 1 |
| Faulted mix, all 200 | 140 | 138 | 137 | 9 / 7 | 10 / 7 |
| of these: healthy robot, 40 | 39 | 40 | 40 | 0 / 1 | 0 / 1 |
| camera turned by 6 degrees, 80 | 45 | 42 | 39 | 7 / 4 | 9 / 3 |
| actions 17 steps late, 80 | 56 | 56 | 58 | 2 / 2 | 1 / 3 |

(32-bit ran at batch size 8 in the first two lines, the skeleton's size, and
at 4 in the faulted mix; 16-bit always at 4.) No difference on healthy
episodes: with the recommended setting the very same 195 of 200 succeed.
Under faults the three formats are at 70, 69 and 68.5%, and splits like 9
against 7 are what chance gives.

In these first samples the camera fault showed an order, 45, 42, 39 of 80,
that followed the numerical distance from today's model. **A larger paired
sample has since settled that for the recommended variant (section 8): it was
chance.** The same episodes in 32 bits and in the recommended 16-bit variant,
same seeds, noise and batch size:

| Case | Pairs | 32-bit | Recommended 16-bit | Only 32-bit succeeds | Only 16-bit succeeds | Exact paired test, p | A true loss this size would have shown 4 times in 5 |
|---|---|---|---|---|---|---|---|
| `libero_object`, no fault | 200 | 199 | 199 | 0 | 0 | 1.00 | 3.9 points |
| `libero_spatial`, camera turned by 6 degrees | 400 | 221 | 221 | 17 | 17 | 1.00 | 5.4 points |
| `libero_spatial`, actions 17 steps late | 200 | 145 | 142 | 5 | 2 | 0.45 | 6.4 points |
| These three together | 800 | 565 | 562 | 22 | 19 | 0.76 | 2.9 points |
| With the camera fault at 3 and at 9 degrees as well | 1,200 | 798 | 798 | 32 | 32 | 1.00 | 2.3 points |

So a loss of two to three points overall, or of five to six under the camera
fault at 6 degrees, would very probably have shown, and none did; the
dose-response curve of the camera fault does not shift either (-1.0, -0.5 and
+2.5 points at 3, 6 and 9 degrees). A loss of one point cannot be excluded by
1,200 pairs, and neither can a gain. The fleet's truth does not depend on any
of this: original and replay run in the same format and are identical.

*Does it repeat?* Yes: call for call, in a fresh process, and in closed loop
(point 3).

**2. Batching rule.** The batch size is fixed and belongs to an episode's
identity; the row does not. Any waiting request takes any free row, longest
wait first; rows nobody asked for are filled with a copy of row 0. Run two and
a half workers per row, so that when a call ends the next requests are already
waiting. **Four rows and ten workers** gave the most episodes per hour within
the limits (section 5); three rows are as good, six and eight are worse with
ten workers because the rows no longer fill.

**3. Exact replay is kept.** In the new layout, with free rows (480 of the
1,200 pairs below carry a fault):

| Compared | What differs between the two runs | Pairs | Identical |
|---|---|---|---|
| Faulted mix, recommended setting | 10 workers against 8, other processes, order, batch mates, the row of every call; 160 of the pairs carry a fault | 200 | 200 |
| the same against a third run | as above, and video on | 200 | 200 |
| Healthy, recommended setting: the 200 with video against the 100 of the success-rate run | processes, order, batch mates, rows, video | 100 | 100 |
| Faulted mix, plain `fp16` | 10 workers with video against 8 without; 160 with a fault | 200 | 200 |
| 200 healthy, plain `fp16` | 10 workers against 8 | 200 | 200 |
| the same against a third run | video on, workers replaced at other moments | 200 | 200 |
| 32-bit, 100 episodes, against the **skeleton's** run of the same episodes | another runner, free rows instead of fixed ones, 10 workers instead of 8, pinned database instead of timing | 100 | 100 |
| **All** | | **1,200** | **1,200** |

Policy alone, 80 comparisons of each kind per batch size: section 3.

**4. Worker count and memory.** Ten workers. Process memory peaked at 20.8 GiB
(21.6 to 23.0 GiB in a few runs, see section 8) of the 24 GiB that
`tools/heavy-run` allows, GPU memory
of the run's processes at 9.4 GiB (machine-wide 12.5 GiB, limit 17 GiB), and
the machine never had less than 18 GiB available. Twelve workers would by the
same arithmetic pass the 24 GiB ceiling (calculated), so I did not try them.
Eight workers hold 17.5 GiB and cost 5 to 9% (measured without video: 5,090
against 5,335 healthy per hour with plain `fp16`, 2,866 against 3,117 in the
faulted mix with the recommended setting).

**5. Against the plan.** Medium worlds need about 2,100 episodes per hour at
faulted-fleet success rates; the faulted mix ran at 3,093 on a machine that
other jobs were using. At that rate the plan's 255,000 episodes are 82 hours
of queue.

**6. If 16 bits are not wanted.** In 32 bits nothing I tried makes a call
faster (section 2). The new batching alone gives 1,856 healthy and 1,154
faulted episodes per hour: small worlds, not medium ones. The pinned database
and the free rows are worth having either way.

**7. Two settings that change the policy** would buy more, and are knowledge,
not a recommendation (section 6): five sampling steps instead of ten make a
call 1.5 times faster (98 of 100 healthy episodes instead of 100), and
LeRobot feeds this checkpoint one empty camera image more than its
configuration asks for, which costs 15% of every call.

**8. SmolVLA as the second policy** (section 7) ran 3,630 healthy and 2,648
faulted-mix episodes per hour in the same layout, in 32 bits. Its chunks do
depend on the row, so these runs are not replayable as they stand.

## 1. Where a call's time goes

`pipeline.sh profile` (`profile_call.py`: every stage with the GPU's queue
flushed before and after, median of three calls after a warm-up; real
observations recorded by `make_obs.py`). Load 0.6 to 3.4 during the 32-bit
run and 4.5 during the 16-bit run.

X-VLA has 0.88 billion parameters in three parts, and one call runs them in
this order:

| Stage of one call for 8 rows | Today, ms | Share | 16-bit (plain `fp16`), ms | Share |
|---|---|---|---|---|
| Preparation on the CPU: stack the observations, images to float tensors, flip, build the state, tokenize the instruction | 2 to 50 | about 1% | 6 to 50 | about 2% |
| Copy to the GPU, with the tokenizer (part of the line above) | 2 | | 1.4 | |
| Resize both images to 224 x 224 on the GPU | 0.7 | 0% | 0.7 | 0% |
| **Vision tower** (Florence-2's image encoder, 363 M parameters) on 16 images | 534 | 17% | 236 | 28% |
| Language encoder (12 layers over 50 image and 50 text tokens; 208 M parameters with the token embeddings) | 40 | 1% | 10 | 1% |
| **Sampling loop**: 10 steps of the action transformer (24 layers, 309 M parameters) over 312 tokens | 2,477 | **81%** | 601 | **71%** |
| Copy the chunk to the CPU | 0.1 | 0% | 0.1 | 0% |
| Chunk to environment actions (6D rotation to axis-angle, gripper threshold) | 20 | 1% | 5 | 1% |
| **Whole call** | **3,054** | | **850** | |

- The recommended variant differs from the 16-bit column only in the loop,
  which takes 675 instead of 601 ms (926 ms for the whole call).
- Preparation jumped between 2 and 50 ms from call to call in these
  single-process measurements. Inside the fleet runs it was 4 to 8 ms per
  call.
- **Copies between host and GPU are negligible**: 2 ms in, 0.1 ms out.
  The one oddity is that today's conversion to environment actions sends each
  of the 30 actions to the GPU and back (20 ms); done in one pass on the CPU it
  is 13 ms and gives the same bits.
- **The loop is the call.** Each of the ten steps runs all 24 layers over all
  312 tokens again, 248 ms a step. Nothing can be carried from one step to
  the next: the 30 action tokens change at every step and every other token
  attends to them, so every token's value changes too. I looked for work to
  hoist and found only the two input projections of the image and text tokens
  and the lookup of the soft prompts; computing those once per call instead of
  ten times is bit-identical and saves less than the measurement noise.
- Nothing can be done once per episode either: the language encoder reads the
  scene image together with the instruction, so its output changes with every
  call. Only the tokenizer's work repeats, and it takes a millisecond.

**What the 312 tokens are**, because it matters for section 6: 30 actions; 100
from the language encoder (50 from the scene camera, 50 for the instruction,
of which about 20 are words and the rest padding that the model does attend
to); 150 "auxiliary" image tokens, 50 from the wrist camera and **100 that are
all zeros**, standing for two cameras that do not exist; and 32 learned
"soft prompt" tokens.

**Inside the loop** (a timer around every layer; these timers flush the GPU
thousands of times per call, so read the shares):

| Part of the sampling loop, 8 rows | Today, ms | Share | 16-bit (plain), ms | Share |
|---|---|---|---|---|
| Linear layers (matrix products) | 1,793 | 68% | 489 | 75% |
| Attention itself (softmax of q k^T, times v) | 598 | 23% | 43 | 7% |
| GELU | 89 | 3% | 53 | 8% |
| Reshapes, additions, timer overhead | 105 | 4% | 46 | 7% |
| Layer norm | 31 | 1% | 21 | 3% |

**Why 16 bits are so much faster on this chip** (`pipeline.sh micro`,
`micro_bench.py`, load 0.7 to 1.6): the same matrix products alone, in the
action transformer's shapes for 8 rows.

| | float32 | float16 | bfloat16 |
|---|---|---|---|
| MLP, 1024 to 4096 | 2.66 ms, 7.9 TFLOPS | 0.57 ms, 37 TFLOPS | 0.59 ms, 36 TFLOPS |
| MLP, 4096 to 1024 | 2.51 ms, 8.4 | 0.72 ms, 29 | 0.68 ms, 31 |
| Attention inputs, 1024 to 3072 | 1.72 ms, 9.2 | 0.43 ms, 37 | 0.40 ms, 40 |
| Attention output, 1024 to 1024 | 0.63 ms, 8.3 | 0.18 ms, 28 | 0.18 ms, 29 |
| One 4096 x 4096 product | 34 ms, 4.0 | 3.4 ms, 40 | 3.4 ms, 41 |
| Attention for one layer | 2.25 ms | 0.14 ms | 0.14 ms |

The Radeon multiplies 16-bit matrices four to five times as fast as 32-bit
ones, and computes attention sixteen times as fast. My explanation, which I
did not verify: this chip generation has dedicated matrix instructions for
16-bit numbers only, and PyTorch has a fused attention routine for 16-bit
numbers only on this software stack (its warning says so). The measured fact
stands either way.

**By batch size** (whole call, ms):

| Rows | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Today | 565 | 903 | | 1,559 | | | 3,054 | | 4,548 | 6,093 |
| per row | 565 | 451 | | 390 | | | 382 | | 379 | 381 |
| 16-bit (plain) | 127 | 219 | 308 | 455 | 540 | 610 | 850 | 1,111 | 1,381 | 1,805 |
| per row | 127 | 110 | 103 | 114 | 108 | 102 | 106 | 111 | 115 | 113 |

A row costs the same from four rows up (32-bit) or from two rows up (16-bit).
The recommended variant took 479 ms for four rows and 926 ms for eight.
Large batches buy nothing; they only make everyone wait longer. PyTorch's own
peak GPU memory is 3.3 GiB at any batch size in 16 bits and 3.4 to 4.6 GiB in
32 bits.

## 2. Every change tried

`pipeline.sh variants` (`variants.py`: one fresh process per line; the same 24
recorded observations as three batches of eight with fixed noise; time is the
median of five calls of the first batch, three for the slow 32-bit lines; load
0.5 to 7). "Call" is everything from the resized images to the environment
actions, for 8 rows. "Largest difference" compares the 24 chunks with
today's: in the policy's own units, and for the commanded position in
millimetres. Success rates are from fleet runs of the same 100 episodes
(`pipeline.sh success`); a dash means I did not spend the machine time,
because the change is not faster or not a candidate.

| Change | Call, ms | Faster by | Output against today's | Success, 100 episodes | Notes |
|---|---|---|---|---|---|
| **Today**: 32-bit, `cudnn.benchmark` on, the machine's shared MIOpen database | 3,069 | 1.00 | | **100/100** | second process: 3,079 ms, identical |
| *PyTorch settings, 32-bit* | | | | | |
| `cudnn.benchmark` off, database holds the convolutions | 3,086 | 0.99 | identical | | this is the pinned setup |
| `cudnn.benchmark` off, empty database | 3,208 | 0.96 | differs, 1.7e-6 | - | falls back to a slower algorithm |
| Channels-last memory format for the vision tower | 3,120 | 0.98 | identical | | |
| `torch.no_grad` instead of inference mode | 3,093 | 0.99 | identical | | today already uses inference mode |
| Attention written out instead of PyTorch's fused call | 3,045 | 1.01 | differs, 7e-7 | - | within noise |
| Linear weights stored transposed | 3,417 | 0.90 | differs, 1e-6 | - | slower in 32 bits |
| `torch.compile` of the action transformer's 24 layers | 3,122 | 0.98 | differs, 6e-7 | - | it works on this stack; no gain |
| *MIOpen, 32-bit* | | | | | |
| Empty database, first process (searches for 9 s in its first call) | 3,095 | 0.99 | identical | | |
| The same database, second process | 3,092 | 0.99 | identical | | |
| Full search with tuning (`MIOPEN_FIND_MODE=1`, `MIOPEN_FIND_ENFORCE=3`) | 3,076 | 1.00 | identical | | picks the same algorithms |
| Reading that tuned database | 3,134 | 0.98 | identical | | |
| MIOpen's Winograd algorithms forbidden | 3,174 | 0.97 | differs, 1.7e-6 | - | shows that the algorithm decides the bits |
| *Work moved out of the loop or the call, 32-bit* | | | | | |
| Loop-invariant projections computed once per call | 3,099 | 0.99 | identical | | no gain |
| Chunk to environment actions in one pass | 3,069 | 1.00 | identical | | 20 ms become 13 |
| Both, with the pinned database (`fp32` in the fleet runner) | 3,064 | 1.00 | identical | 100/100, identical to the skeleton's episodes | |
| *16-bit floats* | | | | | |
| Automatic mixed precision, bfloat16 | 1,067 | 2.88 | differs, 9.8e-3, 5.8 mm | - | |
| Automatic mixed precision, float16 | 1,046 | 2.93 | differs, 1.1e-3, 0.8 mm | - | |
| Action transformer's weights in float16 | 1,242 | 2.47 | differs, 2.0e-3, 1.2 mm | - | |
| ... in bfloat16 | 1,240 | 2.47 | differs, 2.6e-2, 12 mm | - | |
| ... float16 only inside its attention and MLP sub-layers | 1,327 | 2.31 | differs, 1.6e-4, 0.04 mm | - | the nearest to today's numbers |
| Whole model in float16 | 910 | 3.37 | differs, 2.0e-3, 1.1 mm | - | row position matters (section 3) |
| Whole model in bfloat16 | 916 | 3.35 | differs, 2.6e-2, 12 mm | - | |
| Whole model in float16, sub-layers only in the action transformer | 988 | 3.11 | differs, 1.4e-3, 0.2 mm | - | |
| Whole model in float16, weights transposed, pinned database (`fp16`) | 849 | 3.61 | differs, 2.0e-3, 1.2 mm | 100/100 | the fastest; second process identical |
| **The same with sub-layers only in the action transformer** (`fp16_res32`) | **926** | **3.31** | differs, 1.1e-3, 0.3 mm | **100/100** | **recommended** |
| The same as `fp16` in bfloat16 (`bf16`) | 1,050 | 2.92 | differs, 2.2e-2, 12 mm | 98/100 | its convolutions are not in the pinned database, hence slower |
| *PyTorch settings and MIOpen again, on the whole model in float16* | | | *against that model* | | |
| `cudnn.benchmark` off, database holds the convolutions | 938 | | identical | | |
| `cudnn.benchmark` off, empty database | 1,016 | | differs, 1e-3, 1 mm | - | vision tower 342 ms instead of 236 |
| Channels-last | 918 | | identical | | |
| Attention written out | 1,136 | | differs | - | slower: the fused 16-bit routine is the fast one |
| Loop-invariant work and one-pass conversion | 906 | | identical | | |
| Full MIOpen search, and reading it | 908, 910 | | identical | | |
| Winograd forbidden | 916 | | differs, 2e-3 | - | |
| `torch.compile` of the action transformer's 24 layers, on `fp16` | 796 | | differs, 2.4e-3, 1 mm | - | 7% faster than `fp16`; not qualified, see below |

Reading it:

- **In 32 bits nothing helps.** No PyTorch or MIOpen setting made a call more
  than 1% faster, which is inside the noise, and several made it slower. The
  convolutions that MIOpen could tune are 78 of the 3,054 ms.
- **MIOpen, as asked.** A one-time tuned search with a saved database does
  not make 32-bit layers faster than today's setup, because today's setup
  already searches. With `cudnn.benchmark` on, the first call of every
  process takes 4.6 s, against 3.8 s with it off and 3.1 s for later calls
  (14 s when the kernel cache is empty as well), and the timings stored in
  the database change. I read that as: MIOpen times the algorithms again in
  *every process*, and `MIOPEN_FIND_MODE=2` does not prevent it. The saved database
  matters for another reason, exactness, and in 16 bits also for speed: with
  `cudnn.benchmark` off and nothing stored for the batch size in use, MIOpen
  falls back to a default that makes the vision tower 45% slower (342 against
  236 ms). Section 4.
- **Converted weights beat automatic mixed precision** (849 against 1,046 ms)
  and are simpler to reason about. The throughput spike found mixed precision
  20 to 30% *slower* than 32 bits for SmolVLA; for X-VLA it is nearly three
  times faster. I did not look into why SmolVLA differs.
- **float16 is nearer to today's numbers than bfloat16** by a factor of ten
  at the same speed, and nothing overflowed: every output was finite. LeRobot's
  own `dtype="bfloat16"` switch for X-VLA, which fails in 0.6.1, would give
  the bfloat16 line.
- **Which 16-bit variant.** `fp16` is the fastest per call. `fp16_res32`
  keeps each token's running sum and the layer norms of the action transformer
  in 32 bits, stays four times nearer to today's commanded positions and
  costs 9% of a call. Both passed every test here: row check, replay check,
  100 healthy, 200 healthy, 200 faulted. End to end they came out equal,
  because with calls this short the simulation workers share the limit. So I
  recommend the nearer one. If the GPU becomes the only limit again (more
  workers, a quieter machine), `fp16` is 9% of a call cheaper.
- **`torch.compile` works here** and makes the 16-bit loop 8% faster (549
  against 599 ms). Its output differs from the uncompiled 16-bit model by as
  much as 16 bits differ from 32, so it would need its own row check, replay
  check and success rates, and it ties exactness to generated code. I did not
  qualify it. In 32 bits it gains nothing.

## 3. Batching: is a row independent of its position and of the other rows?

`pipeline.sh rows` (`rows_check.py`; 80 recorded observations, each with its
own instruction and noise, laid out in batches; a comparison counts as
identical only if all 600 numbers of the chunk are the same bits).

| Model | Batch size | Same content in another row | Same row, other content in every other row | Both at once | Half the batch filled with copies of row 0 |
|---|---|---|---|---|---|
| Today, 32-bit | 8 | 80/80 identical | 80/80 | 80/80 | 80/80 |
| | 4 | 80/80 | 80/80 | 80/80 | 80/80 |
| 16-bit, PyTorch's usual linear layers | 8 | **0/80**, up to 2.9e-3 | 80/80 | 0/80 | 40/80 |
| | 4, 6, 12 | 0/80, 0/78, 0/72 | all identical | none | half |
| 16-bit `fp16`, weights stored transposed | 8 | 80/80 | 80/80 | 80/80 | 80/80 |
| | 4 | 80/80 | 80/80 | 80/80 | 80/80 |
| | 6 | 78/78 | 78/78 | 78/78 | 78/78 |
| **Recommended** (`fp16_res32`, weights transposed) | 4 | 80/80 | 80/80 | 80/80 | 80/80 |
| | 8 | 80/80 | 80/80 | 80/80 | 80/80 |

So for X-VLA in 32 bits the answer is yes on both counts, as the skeleton's
single swap suggested. In 16 bits the answer is yes only after one change.
The 32-bit and `fp16` checks ran at load 2 to 7, the recommended variant's at
load 3 to 6.

**What goes wrong in 16 bits, and the fix** (`rows_diag.py`, `rows_diag2.py`).
I ran every layer of the real 16-bit model a second time on its own input
with the rows rotated. Convolutions, layer norms, GELU and attention gave
every row the same bits in any position (attention with its inputs copied to
plain memory first; on the vision tower's own strided inputs 10 of 24 calls
differed, but that test changes the memory layout along with the position and
so does not tell). Sixteen of the 29 kinds of linear layer did not, among
them the attention inputs and outputs of the action transformer, while its
two MLP layers, the same routine at other shapes, did.
All 29 are position-free when the same product is written in any of three
other ways; the cheapest is to store the weight matrix transposed, so that
the library is called without its "transpose" flag. My reading: the
library's 16-bit product has a code path for transposed inputs that treats
rows differently depending on where they lie in the matrix. I did not look
further. In 32 bits the same change is 10% slower and changes the last bit,
so it belongs to the 16-bit setup only.

**The batch size itself changes a chunk**, in both formats, so it stays fixed
and stays part of an episode's identity:

| First 16 contents against their chunk at batch size 8 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Today, 32-bit: identical of 16 | 0 | 0 | | 0 | | 0 | | 16 | | 0 | 0 |
| largest difference | 1.3e-6 | 7e-7 | | 1.6e-6 | | 9e-7 | | 0 | | 8e-7 | 1e-6 |
| 16-bit (`fp16`, transposed): identical of 16 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 16 | 16 | 12 | 0 |
| largest difference | 1.2e-3 | 1.2e-3 | 1.5e-3 | 1.0e-3 | 1.6e-3 | 1.0e-3 | 0 | 0 | 0 | 1.0e-3 | 2.8e-3 |

That sizes 7, 8 and 10 agree in 16 bits is a curiosity, not something to
build on. For the recommended variant I compared sizes 4 and 8 only: none of
16 identical, up to 1.2e-3. In 32 bits the difference between two batch sizes
is a thousand times smaller than in 16 bits, and it still changes episodes:
the 200 healthy episodes gave 195 successes at batch size 8 and 194 at batch
size 4.

## 4. The convolution database and exact replay

This was not asked for and matters for the engine whatever the number format.

X-VLA's vision tower has eight kinds of convolution, and MIOpen has two to
four algorithms for each. Measured (`variants.py`, the lines with "database"
and "Winograd" above):

- Another algorithm gives other bits: 1.7e-6 in 32 bits, 1e-3 in 16 bits,
  where it moves a commanded position by a millimetre.
- With `cudnn.benchmark = True` (what `lerobot-eval` and the walking skeleton
  set) the first call of a process takes 0.8 s longer than with it off, and
  my private copy of the database had its stored timings rewritten by such
  runs (one entry went from 5.01 to 5.28 ms). I read that as: each process
  times the algorithms itself and takes the fastest.
- For several convolutions the two fastest algorithms are within 9% of each
  other (2.82 against 3.06 ms, for example), and in a fleet run the timing
  happens while the simulation workers start up beside it.
- With `cudnn.benchmark = False` PyTorch takes the fastest algorithm *stored
  in the database* and measures nothing, and the database is not written
  (file times unchanged). With an empty database it takes a default, which is
  a different algorithm again.
- The database lives in `~/.config/miopen` and is shared by every project of
  this user on this machine.

So today an episode's bits depend on a timing race in each process and on a
file that other projects write. The skeleton's 496 identical pairs say the
race came out the same way every time that day; nothing guarantees it. This
is an inference from the measurements above; I did not catch a flip.

What this spike did instead, and what I recommend for the engine
(`pipeline.sh miopen-pin`, `miopen_pin.py`): search once per batch size on a
quiet machine into a database of the project's own
(`MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`), make it read-only, run
everything with `cudnn.benchmark = False`, and treat the database file as part
of the software version an episode was made with. The search takes 11 to 34 s
per batch size in 16 bits. The database is keyed by the number of images in a
call and by the number format, so a batch size that is not in it silently gets
the slow default. With the pinned database the 32-bit model reproduces the
skeleton's episodes bit for bit (verdict, point 3).

## 5. Batch size and worker count, end to end

`pipeline.sh scan`, `headline`, `exact` (`fleet.py`, this spike's copy of the
skeleton's runner; changes listed under "Files"). 200 healthy episodes, the
same as the skeleton's healthy batch, no video in the scan. The scan was run
with plain `fp16`, before I settled on the recommended variant; the two do
not differ in what the scan is about.

| Rows | Workers | Per hour, end to end | In steady state | One call, ms | Rows in use | Server busy | Workers waiting | Process memory peak | Load at start |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 10 | 5,409 | 6,292 | 351 | 99% | 82% | 56% | 20.8 GiB | 3.7 |
| 4 | 8 | 5,090 | 5,685 | 495 | 98% | 80% | 71% | 17.5 GiB | 1.8 |
| **4** | **10** | **5,335** | **6,175** | 503 | 98% | 86% | 74% | 20.8 GiB | 1.6 |
| 5 | 10 | 4,916 | 5,586 | 593 | 95% | 77% | 66% | 22.1 GiB | 2.5 |
| 6 | 10 | 4,522 | 5,340 | 637 | 91% | 66% | 80% | 20.8 GiB | 3.0 |
| 8 | 10 | 3,774 | 4,163 | 871 | 69% | 74% | 83% | 22.3 GiB | 3.1 |

- **Keep the batch small and the workers at two to three per row.** With
  eight rows and ten workers only two requests are waiting when a call ends;
  the other eight are simulating, the call starts half empty and costs the
  same. Because a row costs the same at any batch size from two up (section
  1), a small batch loses nothing on the GPU.
- Three and four rows are equal within what one run on a shared machine can
  tell; the three-row run and the five-row run had a busier machine (a
  simulation step took 14 to 17 ms instead of 9).
- "End to end" counts from the start of the runner to the last episode's
  files: loading the policy (5.9 s), starting the workers, 25 to 29
  environment builds, replacing workers, and the tail. "Steady state" is the
  rate over the middle 80% of episode completions. For a 200-episode run the
  difference is 8 to 16%; for a campaign of thousands the steady-state
  figure is the relevant one.

**The headline runs**, four rows and ten workers, both videos and all tables
written, each in one go (`pipeline.sh headline`):

| Run | Episodes | Successes | Per hour, end to end | Steady state | Steps / chunks per episode | One call, ms | Rows in use | Server busy | Process memory peak | GPU memory: run / machine | Least memory available | Load, start to end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `h200v_fp16res32_r4w10`, healthy, **recommended** | 200 | 195 | **4,962** | 5,753 | 111 / 4.2 | 533 | 98% | 84% | 20.8 GiB | 9.4 / 12.4 GiB | 23.0 GiB | 3.7 to 4.7 |
| `m200v_fp16res32_r4w10`, faulted mix, **recommended** | 200 | 138 | **3,093** | 3,378 | 193 / 7.0 | 543 | 98% | 88% | 20.8 GiB | 9.4 / 12.4 GiB | 21.2 GiB | 2.8 to 6.1 |
| `h200v_fp16_r4w10`, healthy, plain `fp16` | 200 | 195 | 4,850 | 5,529 | 111 / 4.2 | 510 | 98% | 79% | 20.8 GiB | 9.4 / 12.5 GiB | 22.6 GiB | 2.9 to 5.2 |
| `m200v_fp16_r4w10`, faulted mix, plain `fp16` | 200 | 137 | 3,170 | 3,447 | 194 / 7.0 | 512 | 99% | 85% | 22.6 GiB | 9.4 / 12.5 GiB | 22.9 GiB | 5.0 to 4.7 |
| `h200v_fp32_r4w10`, healthy, 32-bit | 200 | 194 | 1,856 | 1,952 | 112 / 4.2 | 1,658 | 99% | 94% | 20.8 GiB | 10.2 / 13.2 GiB | 21.7 GiB | 2.9 to 3.1 |
| `m200v_fp32_r4w10`, faulted mix, 32-bit | 200 | 140 | 1,154 | 1,196 | 192 / 7.0 | 1,668 | 99% | 96% | 22.3 GiB | 10.2 / 13.3 GiB | 21.9 GiB | 3.6 to 3.5 |
| `m200_fp16res32_r4w10`, faulted mix, recommended, no video | 200 | 138 | 3,117 | 3,374 | 193 / 7.0 | 537 | 99% | 86% | 20.8 GiB | 9.4 / 12.5 GiB | 23 GiB | 3.5 |
| `m200_fp16res32_r4w8`, the same with 8 workers | 200 | 138 | 2,866 | 3,090 | 193 / 7.0 | 523 | 98% | 78% | 17.5 GiB | 8.3 / 11.3 GiB | 25 GiB | 3.1 |
| `m200_fp16_r4w8`, faulted mix, plain `fp16`, 8 workers, no video | 200 | 137 | 3,164 | 3,503 | 194 / 7.0 | 494 | 98% | 82% | 17.5 GiB | 8.3 / 11.3 GiB | 25.5 GiB | 2.8 to 4.0 |

How to read it:

- **The GPU is still the bottleneck, but no longer by much.** In the 16-bit
  runs the server is busy 78 to 88% of the time and the workers wait 62 to
  74% of theirs. The rows are full whenever the server runs; what remains is
  about twenty gaps of a second each per run, when several workers are
  building an environment or being replaced at once and fewer than four
  requests arrive. A call takes 0.51 to 0.54 s in the fleet against 0.46 to
  0.48 s alone, because the workers render on the same chip.
- **The ceiling** for four rows at 0.54 s is 7.4 chunks per second: 6,350
  healthy or 3,800 faulted-mix episodes per hour (calculated). The
  recommended runs reached 90 and 89% of that in steady state.
- **Faults still cost in proportion to their chunks**: 7.0 per episode in the
  mix against 4.2.
- **In 32 bits the server is busy 94 to 96%** and the gain over the skeleton
  is the fuller rows (99% against 87 to 91%) and fewer environment builds (28
  against 84, because a worker now stays with the task it has built).
- **Video** made no difference I can show: 4,850 against 5,335 healthy
  episodes per hour with plain `fp16` on a busier machine, 3,093 against 3,117
  in the faulted mix with the recommended variant.
- **Environment builds take 3 to 5 s each** in these runs, against 1 to
  1.5 s in the skeleton. Ten workers build at the same moment at the start,
  beside video encoding and another agent's jobs; I did not test which of
  these is the cause.
- **A worker held 1.5 to 1.8 GiB** when it ended (1.9 to 2.1 GiB at its
  largest), where the skeleton's ended at 1.9 to 2.1 GiB; I did not find out
  why. The server's peak is 3.9 GiB while the checkpoint loads (5.2 to 5.9 GiB
  in six of the runs) and 2.7 GiB afterwards.

## 6. Settings that change what the policy does

**These are not recommendations for the nominal fleet.** They are
configuration settings a simulated fleet could be given, with what each costs
and buys. `pipeline.sh knobs`: plain `fp16`, four rows, ten workers, the 100
healthy episodes of the success-rate sample, no video. The stored values are
in bold.

| Setting | Value | Successes of 100 | Chunks per episode | One call for 4 rows, ms | Per hour, end to end | Steady state | Against the stored value, same episodes |
|---|---|---|---|---|---|---|---|
| Sampling steps | 5 | 98 | 4.2 | 329 | 5,459 | 7,770 | 2 episodes fail that succeed with 10 |
| | **10** | **100** | 4.1 | 506 | 4,877 | 6,438 | |
| | 20 | 99 | 4.2 | 848 | 3,228 | 3,692 | 1 episode fails that succeeds with 10 |
| Actions executed per chunk | 10 | 99 | 12.4 | 484 | 1,967 | 2,318 | 1 episode fails that succeeds with 30; episodes 12 steps longer |
| | 20 | 100 | 5.8 | 491 | 3,715 | 4,540 | the same 100 succeed |
| | **30** | **100** | 4.1 | 506 | 4,877 | 6,438 | |
| Image views fed to the model | 3 | 98 | 4.2 | 419 | 5,025 | 6,537 | 2 episodes fail that succeed with 4 |
| | **4** | **100** | 4.1 | 506 | 4,877 | 6,438 | |

- **Sampling steps.** The loop's time is proportional to the steps; the
  vision tower's is not, so 5 steps make a call 1.5 times faster and 20 steps
  1.7 times slower. Five steps give nearly the same chunk as ten: on the 24
  recorded observations the commanded positions differ by at most 1.0 mm,
  which is as little as 16 bits differ from 32. With five steps the run was
  short of work for the server (busy 65%), so its end-to-end figure
  understates it.
- **Actions per chunk.** 30 is the whole chunk, so the neighbours lie below.
  Looking again after 20 or 10 actions costs calls in proportion and bought
  nothing on healthy episodes, where there is nothing left to gain. Whether
  it helps a faulted robot was not measured.
- **Image views.** The checkpoint's configuration asks for three views (scene
  camera, wrist camera, one empty). LeRobot 0.6.1 counts the empty one twice
  and feeds four, so the action transformer carries 50 tokens of zeros more
  than configured: 312 instead of 262. With three, a call for 8 rows takes 718
  instead of 849 ms. The chunks differ substantially (up to 11 mm), so this
  is another policy, not a rounding change. Whether three is what the model
  was trained with I did not check; the configuration suggests it. On the
  100 healthy episodes it solved 98, which a sample of this size cannot tell
  from 100.

## 7. SmolVLA as the second policy

`pipeline.sh smolvla`. `lerobot/smolvla_libero` as the baseline spike corrected
it: 10 of its 50 predicted actions executed per chunk, relative control,
MuJoCo 3.3.7, 32-bit floats, `policy_smolvla.py`. Four rows, ten workers,
both videos, the same two manifests. The faults keep the levels that were
chosen for X-VLA.

| Run | Episodes | Successes | Per hour, end to end | Steady state | Steps / chunks per episode | One call, ms | Rows in use | Server busy | Workers waiting | Process memory peak | GPU memory: run / machine | Load at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `h200v_smolvla_r4w10`, healthy | 200 | 166 (83%) | **3,630** | 4,026 | 133 / 13.6 | 237 | 99% | 86% | 80% | 19.8 GiB | 7.5 / 10.5 GiB | 2.3 |
| `m200v_smolvla_r4w10`, faulted mix | 200 | 93 (46.5%) | **2,648** | 2,879 | 197 / 19.9 | 235 | 99% | 90% | 81% | 20.2 GiB | 7.5 / 10.8 GiB | 4.7 |

- **The GPU is the bottleneck for SmolVLA too** once it looks every 10 steps:
  a call is short (0.24 s for four rows), but an episode needs 14 to 20 of
  them.
- **The same two faults hit SmolVLA much harder**: 46.5% in the mix against
  69% for X-VLA. I did not break that down by fault.
- **These two runs are not replayable as they stand.** Policy alone
  (`rows_check.py --model smolvla`): at batch sizes 4 and 8 the same content
  in another row gave another chunk in 80 of 80 comparisons, by up to 0.03 in
  the policy's units at batch size 4 and up to 2.0 at batch size 8 (I did not
  look at which action component), and other batch mates changed nothing (80
  of 80 identical). That is what the replay spike found. Storing the
  weights transposed, the change that cures X-VLA in 16 bits, does **not**
  cure SmolVLA in 32 bits (0 of 80). So under exact replay SmolVLA needs the
  skeleton's fixed rows, or its position-dependent layer has to be found the
  way `rows_diag2.py` found X-VLA's. Neither was done here.
- **One SmolVLA call by batch size**: 98, 144, 219, 367 and 702 ms for 1, 2,
  4, 8 and 16 rows. Unlike X-VLA, a row gets cheaper up to eight rows (98, 72,
  55, 46, 44 ms), so SmolVLA would gain from a larger batch if the rows can be
  kept full.
- **16 bits for SmolVLA were not measured.** Converting its weights fails
  inside LeRobot's SmolVLA code (`mat1 and mat2 must have the same dtype`);
  I did not pursue it. Given what 16 bits do for X-VLA's matrix products on
  this chip, it is the obvious next thing to try, by automatic mixed
  precision or by fixing the conversion.

## 8. 32-bit against 16-bit, outcome by outcome

Added on 2026-10-07 to settle the caution the first version of this report
carried: on 80 camera-fault episodes the 32-bit model, the recommended 16-bit
variant and the plain one had solved 45, 42 and 39, in the order of their
numerical distance. Was that the start of a real loss, or chance?

`pipeline.sh paired`. Every episode below ran twice, once in 32 bits (`fp32`)
and once in the recommended 16-bit variant (`fp16_res32`), with the same task,
start state, scene seed, noise seed and batch size (4 rows, 10 workers, no
video). Nothing else ran on the machine; the load average of 2 to 6 is these
runs' own. The fault levels are the skeleton's; cases 4 and 5 were added to
see the camera fault's whole dose-response curve.

| Case | Pairs | 32-bit | 16-bit | Both succeed | Only 32-bit | Only 16-bit | Neither | Exact paired test, p | 16-bit minus 32-bit, points (95% interval) | Smallest true loss this sample would have shown 4 times in 5 |
|---|---|---|---|---|---|---|---|---|---|---|
| **1.** `libero_object`, no fault (10 tasks x start states 0 to 19) | 200 | 199 (99.5%) | 199 (99.5%) | 199 | 0 | 0 | 1 | 1.00 | 0.0 (-2.1 to +2.1) | 3.9 points |
| **2.** `libero_spatial`, scene camera turned by 6 degrees (10 x 40) | 400 | 221 (55.2%) | 221 (55.2%) | 204 | 17 | 17 | 162 | 1.00 | 0.0 (-2.9 to +2.9) | 5.4 points |
| **3.** `libero_spatial`, actions 17 steps late (10 x 20) | 200 | 145 (72.5%) | 142 (71.0%) | 140 | 5 | 2 | 53 | 0.45 | -1.5 (-4.4 to +1.3) | 6.4 points |
| **4.** Camera turned by 3 degrees (the first 200 episodes of case 2, at another level) | 200 | 179 (89.5%) | 177 (88.5%) | 171 | 8 | 6 | 15 | 0.79 | -1.0 (-5.1 to +3.0) | 7.9 points |
| **5.** Camera turned by 9 degrees (the same 200) | 200 | 54 (27.0%) | 59 (29.5%) | 52 | 2 | 7 | 139 | 0.18 | +2.5 (-0.6 to +5.7) | 6.9 points |
| Cases 2 and 3 together (the two faults asked for) | 600 | 366 (61.0%) | 363 (60.5%) | 344 | 22 | 19 | 215 | 0.76 | -0.5 (-2.6 to +1.6) | 3.8 points |
| Cases 1 to 3 together (what was asked for) | 800 | 565 (70.6%) | 562 (70.2%) | 543 | 22 | 19 | 216 | 0.76 | -0.4 (-2.0 to +1.2) | 2.9 points |
| **Cases 1 to 5 together** | **1,200** | **798 (66.5%)** | **798 (66.5%)** | 766 | **32** | **32** | 370 | **1.00** | **0.0 (-1.3 to +1.3)** | **2.3 points** |
| Not in the sums, from section 5's runs: `libero_spatial`, no fault, the skeleton's 200 | 200 | 194 (97.0%) | 195 (97.5%) | 194 | 0 | 1 | 5 | 1.00 | +0.5 (-1.6 to +2.9) | 4.4 points |

How the columns are made (`analyze.py paired`):

- *Exact paired test*: McNemar's test in its exact form. It looks only at the
  pairs whose outcome differs and asks how often a fair coin would split them
  at least this unevenly. A p of 1.00 means the split is as even as it can be.
- *Interval* for the difference in success rate: Newcombe's score interval for
  paired proportions, built from the usual intervals of the two rates and the
  correlation between the paired outcomes. In a simulation of samples like
  these it contained the true difference 95 to 96 times in 100. A second
  method, exact for the pairs that differ, gives the same within about a
  point.
- *Smallest true loss*: how many points the 16-bit model would truly have to
  lose for the test to come out below 0.05 in 80 of 100 samples of this size,
  if the pairs that differ for other reasons keep flipping at the rate seen
  in the sample (calculated).

**What these samples say.**

- **The camera-fault hint was chance.** With 400 pairs the two formats solve
  exactly as many episodes, 221 each, and the 34 pairs that differ split 17
  to 17. The earlier 80 episodes are the first 80 of these 400 and came out
  the same again, 45 against 42; in the 320 new ones it is 176 against 179,
  the other way round.
- **On `libero_object` nothing differs at all**: the same 199 of 200 succeed
  in both formats and the same one fails.
- **Under the delay fault** 145 against 142, with seven pairs that differ,
  five of them against 16 bits. A fair coin does that or worse 45 times in
  100.
- **Over the three cases asked for, 800 pairs,** the 16-bit model is 0.4
  points behind, 22 against 19 of the pairs that differ. **Over all 1,200
  pairs** it is level: 798 successes each, and the 64 pairs that differ split
  32 to 32.

**What they could and could not have detected.**

- **Camera fault.** A true loss of 5.4 points would have shown four times in
  five, one of 6 points nearly nine times in ten, one of 4 points six times
  in ten and one of 3 points four times in ten. The interval puts the
  difference between 2.9 points behind and 2.9 ahead. The earlier 80 pairs
  could only have shown a loss of about 18 points, so they could never have
  settled it.
- **Healthy `libero_object`.** A loss of 4 points would probably have shown;
  the interval allows 2.1 either way.
- **Delay fault.** A loss of 6 to 7 points would probably have shown; the
  interval reaches from 4.4 behind to 1.3 ahead, so this is the least settled
  of the cases.
- **The three cases asked for, 800 pairs.** A loss of 3 points would have
  shown four times in five, one of 2 points only every second time; the
  interval reaches from 2.0 behind to 1.2 ahead.
- **All 1,200 pairs.** A loss of 2.5 points would have shown nearly nine times
  in ten, one of 2 points seven times in ten, one of 1 point only one time in
  four. The interval is 1.3 points either way.
- **A loss of one or two points cannot be excluded** in any single case, and
  neither can a gain of the same size. Under the camera fault, where one pair
  in twelve flips anyway, showing two points would take about 2,000 pairs and
  showing one point about 7,500 (calculated).
- Not covered: other fault kinds, other levels of the delay, `libero_goal`
  and `libero_10`, and the plain `fp16` variant, which was not rerun here.

**Does the whole dose-response curve shift?** No. The same 200 episodes at
three levels of the camera fault:

| Camera turned by | 32-bit | 16-bit | Only 32-bit | Only 16-bit | 16-bit minus 32-bit |
|---|---|---|---|---|---|
| 3 degrees | 179 (89.5%) | 177 (88.5%) | 8 | 6 | -1.0 points |
| 6 degrees | 109 (54.5%) | 108 (54.0%) | 11 | 10 | -0.5 points |
| 9 degrees | 54 (27.0%) | 59 (29.5%) | 2 | 7 | +2.5 points |

The two curves lie on top of each other: a point behind at the mild level,
level in the middle, two and a half ahead at the severe one, each well inside
its interval. A shifted curve would be behind at every level. Over the 800
camera-fault pairs of all three levels it is 454 against 457.

**Why outcomes differ at all.** Under the camera fault at 6 degrees the two
formats disagree on 34 of 400 episodes (8.5%), under the delay on 7 of 200
(3.5%), on healthy episodes on 1 of 400. That is the picture of section 3: any change in the last
digits moves an episode that is close to failing, in either direction, and a
change of batch size in 32 bits alone does the same. In 245 of the 400
camera-fault pairs the two episodes have the same length.

**By task, camera fault, 400 pairs:**

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Pairs | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 400 |
| 32-bit successes | 22 | 35 | 30 | 38 | 21 | 22 | 0 | 4 | 37 | 12 | 221 |
| 16-bit successes | 23 | 37 | 29 | 38 | 18 | 21 | 0 | 4 | 36 | 15 | 221 |
| Only 32-bit | 3 | 0 | 1 | 1 | 4 | 5 | 0 | 0 | 2 | 1 | 17 |
| Only 16-bit | 4 | 2 | 0 | 1 | 1 | 4 | 0 | 0 | 1 | 4 | 17 |

No task carries a difference; tasks 4 and 9 lean opposite ways by three
episodes each.

**Replay, once more.** These runs share 80 camera-fault and 80 delay episodes
with the faulted mix of section 5, run hours earlier in other processes: 160
of 160 identical in 32 bits and 160 of 160 in 16 bits.

**Speed and memory of these runs** (no video, machine otherwise idle; twelve
runs of 200 episodes, 82 minutes of queue of which the 32-bit half took 60):

| Case | 32-bit, per hour (steady state) | 16-bit, per hour (steady state) | Process memory peak | Machine-wide GPU memory peak | Least memory available |
|---|---|---|---|---|---|
| `libero_object`, no fault | 1,598 (1,658) | 4,367 (4,869) | 21.6 GiB | 12.6 GiB | 18.8 GiB |
| Camera 3 degrees | 1,615 (1,734) | 4,212 (4,669) | 20.8 GiB | 13.3 GiB | 19.7 GiB |
| Camera 6 degrees, two runs of 200 | 1,178 and 1,189 (1,254, 1,244) | 3,322 and 3,369 (3,756, 3,686) | 20.8 GiB | 13.3 GiB | 19.5 GiB |
| Camera 9 degrees | 962 (986) | 2,659 (2,839) | **23.0 GiB** | 13.3 GiB | 19.7 GiB |
| Delay 17 steps | 962 (1,011) | 2,653 (2,833) | 20.8 GiB | 13.3 GiB | 19.5 GiB |

The 16-bit model was 2.6 to 2.8 times as fast end to end in every case. One
run came within a gigabyte of the 24 GiB ceiling: while the checkpoint loads,
the inference process briefly holds up to 6.2 GiB instead of 3.9, and in that
run the ten workers were already at full size. Nine workers, or loading the
policy before the workers start, would take that edge off; I did not try
either.

## Surprises

1. **16-bit floats are four to five times faster on this Radeon**, where the
   throughput spike had found them slower for SmolVLA. For X-VLA it is the
   whole answer.
2. **In 32 bits there was nothing to find.** No PyTorch or MIOpen setting
   made a call measurably faster.
3. **In 16 bits a chunk depended on its row**, through one way of calling the
   matrix product, and storing the weights transposed removed it.
4. **Today's setup appears to re-time the convolution algorithms in every
   process**, and the choice decides the last bits. Exact replay has rested on
   that race coming out the same.
5. **LeRobot feeds X-VLA an empty camera too many.** A third of the action
   transformer's tokens are zeros standing for cameras that do not exist.
6. **Small batches are best.** A row costs the same from two rows up, so the
   only thing a larger batch does is leave rows empty.
7. **Five sampling steps give almost the same chunk as ten.**
8. **The batch size changes outcomes even in 32 bits**: 195 against 194 of the
   same 200 episodes at batch sizes 8 and 4.
9. **The two 16-bit variants are equally fast end to end** although one has 9%
   longer calls: with calls of half a second the simulation shares the limit.
10. **SmolVLA is not cured by the same change**: its chunks depend on the row
    in 32 bits, transposed weights or not.
11. **An order of 45, 42, 39 of 80 that looked like a trend was chance**: with
    400 pairs it is 221 against 221, and over 1,200 pairs in five cases 798
    against 798 (section 8).

## What I did not verify

About 16 bits:

- **A loss of about one point of success in 16 bits.** Section 8's 1,200
  pairs exclude a loss of two to three points overall and of five to six
  under the camera fault at 6 degrees with fair confidence, and nothing
  smaller. Under the delay fault alone the interval still allows 4.4 points.
  Other fault kinds, `libero_goal` and `libero_10` were not run, and the plain
  `fp16` variant has only the 200 faulted pairs of section 5.
- **Why the 16-bit product depends on the row position** for some shapes, and
  whether another driver or library version brings it back. The check is
  `rows_check.py`, three minutes; it belongs to the guard rails after any
  update.
- **Batch sizes other than 4 and 8** for the row check of the recommended
  variant (plain `fp16`: 4, 6, 8), and any row check or replay for `bf16`.
- **Why 16 bits are fast here**: instruction set and fused attention are my
  explanation, not a measurement.
- **bfloat16 with its convolutions in the pinned database.** Its call time
  (1,050 ms) includes the slow default for the vision tower.

About exactness:

- **A flip of the convolution algorithm between two processes** with
  `cudnn.benchmark` on. I measured that timings are taken again, that close
  timings exist and that another choice changes bits; I did not see it happen.
- **One machine state**: one evening, one driver and library version, no
  reboot. A pinned database from another library version was not tried.
- **Mid-episode snapshots**, as in the skeleton: not used.
- **Long runs.** The longest run was 200 episodes and ten minutes; a worker
  was replaced at most twice.

About speed:

- **A quiet machine.** Every timing was taken beside another agent's jobs.
  Single numbers are good to about 10%.
- **Twelve workers**: over the 24 GiB process-memory ceiling by my estimate
  (2.1 GiB each plus the server), so not run.
- **`torch.compile` in closed loop.** It runs and is 7% faster on the 16-bit
  model; row check, replay and success rate were not run with it.
- **Preparation on the CPU** (4 to 8 ms per call in the fleet) was left as it
  is. Moving the image conversion to the GPU would save a few milliseconds
  and would have to be checked for identical bits.
- **Why a worker is smaller and an environment build slower** than in the
  skeleton's runs.
- **Video on against off** on an equally loaded machine.

About the knobs: 100 healthy episodes each, one seed, with plain `fp16`.
Under faults, where they would matter, none was run.

About SmolVLA: two runs, 32 bits only, with free rows although its chunks
depend on the row; no replay, no 16 bits, no other batch size end to end.

## Follow-up items

- A decision: may the fleet run X-VLA in 16-bit floats?
- Pin the convolution database and switch `cudnn.benchmark` off
  in the rollout engine.
- LeRobot feeds this checkpoint four image views where its
  configuration says three.
- SmolVLA's chunks depend on the row, and its weights do not
  convert to 16 bits.
- The scheduling rule and its numbers, recorded with the work on scheduling.
- Section 8's paired comparison, recorded with the 16-bit decision.

## Files

| File | Purpose |
|---|---|
| `pipeline.sh` | The commands of every step; the tables name the step. Written afterwards as the record, not run from top to bottom. |
| `run.sh`, `fleet.sh` | One job behind the heavy-job lock: a script, or a fleet run. |
| `xs_common.py` | Environment set-up, pinned reset, hashes, memory readers (from the skeleton), and the private MIOpen directories. |
| `policy_fast.py` | X-VLA with the noise passed in, in stages, with a switch for every change tried. `chunks()` is the skeleton's method unchanged. |
| `make_obs.py` | Records 80 real observations for the policy-only checks. |
| `profile_call.py`, `micro_bench.py` | Section 1. |
| `variants.py` | Section 2: one process per variant, and the comparison table. |
| `miopen_pin.py` | Builds the pinned convolution database. |
| `rows_check.py`, `rows_diag.py`, `rows_diag2.py` | Section 3. |
| `fleet.py` | The runner. Against the skeleton's: free rows, more workers than rows, work handed out on request with a preference for the task a worker has built, first worker lives staggered, inference settings selectable, per-call records in `run.json`. |
| `manifests.py`, `faults.py` | Episode lists; the skeleton's two faults, copied. |
| `analyze.py` | Run summary, exactness comparison, paired success comparison with exact test, interval and detectable difference (section 8). |
| `policy_smolvla.py` | SmolVLA with the same interface (section 7). |

Outputs are in `~/.cache/robots-probes/xvla-speed/`: `runs/` (each with
`log/`, `truth/` and `run.json`), `results/` (profiles, variants, row checks),
`logs/`, `obs/`, `manifests/` and `miopen/` (the pinned database in
`miopen/pinned`, a copy of the shared one as it stood in `miopen/today`, and
the throw-away ones of the MIOpen variants). The shared files in
`~/.config/miopen` were read once to make that copy and never written; their
checksums before and after are the same.
