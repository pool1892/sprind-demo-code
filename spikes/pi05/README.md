# pi0.5 on LIBERO here: success, speed, replayability, robustness under the two faults

Spike run on this machine on 2026-10-07. Question: is pi0.5
(`lerobot/pi05_libero_finetuned_v044`, LeRobot 0.6.1) a usable policy for a
simulated LIBERO fleet, as the main policy or as the second one for the
transfer check, compared with X-VLA (`lerobot/xvla-libero`)?

Everything below was measured here unless it says otherwise; what I conclude
from the measurements is marked as such. **The machine was shared:** other
agents' simulation and evaluation jobs ran beside most of these measurements
(load average up to 24), and each table says what the load was.

Words as in `spikes/xvla-speed/README.md`: an **episode** is one attempt at a
task, at most 280 control steps of 50 ms; a **chunk** is what one call of the
policy returns (pi0.5: 50 actions, of which the first 10 are executed); a
**row** is one episode's place in a batch; **identical** means bit for bit.

## Verdict

**pi0.5 runs here, in bfloat16, and is as good as X-VLA on healthy episodes.
Under the two measured faults it behaves the opposite way: it shrugs off the
turned camera that halves X-VLA's success, and it fails every episode under
the 17-step action delay that X-VLA mostly survives. It is about a ninth as
fast as X-VLA in 16 bits. My reading: usable as the second policy for the
transfer check, where its different failure profile is the point; not as the
main policy, for speed alone.**

| Same episodes as X-VLA's | X-VLA, 32 bits | pi0.5, bfloat16 |
|---|---|---|
| `libero_spatial`, healthy, 200 | 194 (97.0%) | **195 (97.5%)** |
| `libero_object`, healthy, 200 | 199 (99.5%) | **193 (96.5%)**, p = 0.07 |
| Scene camera turned 6 degrees, 400 | 221 (55.2%) | **391 (97.8%)** |
| Actions executed 17 steps late, 200 | 145 (72.5%) | **0 (0.0%)** |
| Healthy `libero_spatial` episodes per hour, end to end with video | 4,962 (16 bits), 1,856 (32 bits) | **552** |

1. **Load and call.** The checkpoint loads and runs through ROCm in both
   formats. bfloat16: 483 ms per call for one row, 1,666 ms for four, 3,019
   ms for eight; 9.1 GiB on the GPU after loading; process memory peaks at
   17.5 GiB while loading, 5 GiB after. float32: 3.5 s, 10.9 s and 23.0 s,
   15.8 GiB on the GPU: it works but is useless in closed loop. In bfloat16
   the vision-language prefix is three quarters of a call (SigLIP 28%,
   language model 39 to 53%), the action expert's ten denoising steps 20 to
   29% (section 1).
2. **Nominal success** on the X-VLA spike's own episode lists, so pair by
   pair: equal on `libero_spatial`, three points behind on `libero_object`
   (seven episodes fail only for pi0.5, one only for X-VLA; p = 0.07).
   Failures are repeated grasp attempts until the step limit (section 2).
3. **Replayability.** LeRobot's pi0.5 takes an externally supplied noise
   sample (`predict_action_chunk(batch, noise=...)`), and the same
   observation and noise give the same bits in another process (80 of 80).
   But in bfloat16 a chunk depends on its row in a larger batch (0 of 80
   identical when a content moves to another row, by up to 1.8 normalised
   units), not on its batch mates. The cause is PyTorch's default matrix
   library here, hipBLASLt, in the wide bfloat16 linear layers; rocBLAS cures
   it at 63% more time per call, X-VLA's transposed-weight trick does not.
   So the fleet runs pi0.5 at batch size 1 (or would need fixed rows), and
   all runs here are at batch size 1. Whole episodes replay: 24 of 24
   episodes run again in another fleet run, with other workers and without
   video, were bit-identical step by step, 16 of them with a fault
   (section 3).
4. **Faults**, on exactly the episodes behind X-VLA's 55.2% and 72.5%: camera
   97.8% against 55.2%, delay 0% against 72.5%. The delay hits pi0.5 so hard
   because it commands movements, not poses: a late movement overshoots, a
   late pose is merely late (my reading). It fits that SmolVLA, the other
   relative-control policy measured here, solved 0 of 80 of the same delay
   episodes in the X-VLA spike, where X-VLA solved 56 of those 80 in both
   precisions. The 17-step level was chosen for X-VLA; at which delay pi0.5
   starts to fail was not measured (section 4).
5. **Throughput** in the X-VLA spike's fleet layout with a pi0.5 adapter
   (`policy_pi05.py`): 552 healthy `libero_spatial`, 440 `libero_object`,
   about 610 camera-fault and 249 delay-fault episodes per hour end to end,
   with the GPU busy 97 to 99% of the time. Medium worlds need about 2,100
   (section 5).

**pi0** (`lerobot/pi0_libero_finetuned_v044`), measured alongside as asked:
it loads and runs the same way, a call is a little cheaper (428 ms for one
row), but it solved only 35 of 50 healthy `libero_spatial` episodes where
pi0.5 solved 48 and X-VLA 49 or 50 of the same ones (depending on the run;
section 6). Not a candidate.

**For gate 2** (my reading of the measurements above):

- *Fit.* Healthy parity on both suites (195 against 194 on `libero_spatial`,
  193 against 199 on `libero_object`, p = 0.07) and the opposite fault
  profile (camera turned: 97.8% against 55.2%; delay: 0% against 72.5%). As
  the second policy of the transfer check it would show plainly that a
  fault's effect depends on the policy.
- *Condition.* At X-VLA's fault levels pi0.5 gives no camera signal (it does
  not fail) and no delay survivor (it never succeeds), so neither level works
  for a pi0.5 transfer world as it stands. Such a world needs its own
  dose-response for each exemplar fault first. That is not measured here: the
  runs at delays of 2 and 5 steps and the camera at 9 degrees were its start
  and were dropped for the lock. Roughly three levels of 100 episodes per
  fault, at 400 to 600 episodes per hour, is half an hour to 45 minutes of
  queue per fault (calculated).
- *Queue cost.* A small world of about 7,600 episodes takes
  about 14 hours at pi0.5's healthy rate (552 per hour) and 19 to 21 hours at
  a faulted-mix rate (about 360 to 400 per hour, section 5), against about
  2.5 hours for X-VLA in 16 bits (3,090 per hour in its faulted mix).
- *What the engine needs to run pi0.5*: relative control per episode; batch
  size 1, or fixed rows, in bfloat16 under hipBLASLt (rocBLAS is free of the
  row dependence at 63% more time per call); the policy loaded before the
  workers start (17.5 GiB of process memory while loading, 9.1 GiB on the GPU
  after); a (50, 32) float32 noise tensor per chunk drawn from the episode's
  noise seed and chunk number; `n_action_steps` 10; `torch.compile` off; the
  rebuilt tokenizer, offline; a pinned MIOpen entry for every batch size used
  (SigLIP's patch convolution); and `rows_pi05.py` as a guard rail after any
  update of PyTorch, ROCm or the checkpoint.

## The setup

- **Checkpoint.** `lerobot/pi05_libero_finetuned_v044`, revision `7152f13e`,
  used as LeRobot 0.6.1 loads it, except two settings: 10 actions executed per
  chunk (`n_action_steps`, as LeRobot's LIBERO documentation sets it for its
  pi0.5 results, "matching the original OpenPI implementation": read on
  2026-10-07 in `docs/source/libero.mdx` on LeRobot's main branch; the
  checkpoint says 50, the whole chunk) and `torch.compile` off (the checkpoint
  asks for it in its slowest mode, `max-autotune`). 4.14 billion parameters:
  3.61 billion in bfloat16 (the two Gemma stacks) and 0.53 billion that
  LeRobot keeps in 32 bits even in bfloat16 mode (SigLIP, the image projector,
  every norm). Relative control: pi0.5 commands small movements, so the
  absolute-control correction X-VLA needs does not apply.
- **The checkpoint is the one the baseline spike already had.** Its local copy
  of `lerobot/pi05_libero_finetuned` is the same commit, and the SHA-256 of
  `model.safetensors` equals the Hub's hash for `_v044`
  (`ckpt/pi05_sha256.txt`, `ckpt/pi05_hub_tree.json`). I copied it rather than
  downloading 7.5 GB again.
- **Tokenizer.** pi0.5 names `google/paligemma-3b-pt-224`, which is gated on
  the Hub. I rebuilt it from the public file the original openpi code reads
  (`paligemma_tokenizer.model` from Google's `big_vision` bucket), as the
  baseline spike did, with `make_tokenizer.py`. On 250 prompts in pi0.5's and
  pi0's formats it gives the same ids as the SentencePiece file read directly
  with the start token in front (0 mismatches, `tokenizer/check.json`).
  Against the baseline spike's rebuild, part by part
  (`tokenizer/compare_with_baseline.json`, written by `make_tokenizer.py`):
  vocabulary, normaliser, pre-tokenizer, post-processor, decoder and added
  tokens are identical; the `tokenizer.json` files differ in the stored
  padding and truncation settings, and the `tokenizer_config.json` files in
  `clean_up_tokenization_spaces` (used only when decoding; unset against
  `false`) and in the path of the staged source file. I could not compare it
  with the gated original.
- **Environment.** `~/.cache/robots-probes/pi05/venv`, a copy of the baseline
  spike's environment: MuJoCo 3.3.7, robosuite 1.5.2 with the port's patch,
  LeRobot 0.6.1, transformers 5.5.4, PyTorch 2.11.0 on ROCm 10.0.0. pi0.5
  needs nothing the baseline environment lacks; the package lists are the same
  (`venv.freeze.txt`). The locked project environment was not touched.
- **Episodes.** The fleet runner is `spikes/xvla-speed/fleet.py`, copied, with
  a pi0.5 adapter (`policy_pi05.py`) that calls LeRobot's own
  `predict_action_chunk` with the starting noise passed in. The episode lists
  are generated by a copy of the X-VLA spike's `manifests.py` and are
  byte-identical to the lists behind the X-VLA numbers I compare with, so every
  pi0.5 episode has an X-VLA twin with the same task, start state, scene seed
  and fault. Each chunk's noise is drawn from the episode's noise seed as for
  X-VLA, in pi0.5's own shape.
- **Inference.** bfloat16, **batch size 1**, three simulation workers. Batch
  size 1 because in bfloat16 a row's chunk depends on its position in a larger
  batch (section 3); at batch size 1 a chunk has the same bits in any process.
- **MIOpen database** pinned as in the X-VLA spike: searched once into this
  spike's own directory, read-only afterwards, `cudnn.benchmark` off.

## 1. Does it load and run, and where does a call's time go?

`pipeline.sh profile` (`profile_pi05.py`, one process per number format, the
first B of the X-VLA spike's 80 recorded `libero_spatial` observations, each
with its own noise; median of five whole calls after a warm-up; stages with
the GPU's queue flushed between them). Load average 1.3 to 2.5 for bfloat16,
1.7 to 3.8 for 32 bits: the machine was nearly quiet.

**Yes in both formats, but only bfloat16 is usable.**

| | bfloat16 | float32 |
|---|---|---|
| Loads | yes, 55 s | yes, 53 s |
| Process memory: while loading / after | 17.5 / 5.3 GiB | 17.5 / 5.1 GiB |
| GPU memory of the process after loading (PyTorch's share) | 9.1 GiB (8.7) | 15.8 GiB (15.5) |
| GPU memory of the process at batch size 8 | 10.8 GiB | 19.1 GiB, machine-wide 22.5 of the 24 GiB stop |
| One call, batch size 1 | **483 ms** | 3,475 ms |
| batch size 4 | 1,666 ms (417 per row) | 10,928 ms |
| batch size 8 | 3,019 ms (377 per row) | 23,006 ms |
| The same call repeated in the same process | identical | identical |

- **Process memory peaks at 17.5 GiB while loading in either format**:
  LeRobot builds the whole model in 32 bits in process memory first and
  converts it afterwards. Beside ten grown simulation workers that would pass
  the 24 GiB ceiling of `tools/heavy-run`, so my runner loads the policy
  before it starts the workers.
- **float32 runs but is useless in closed loop**: 3.5 s for one row, where an
  episode needs 8 to 28 calls. Its language model is 12 to 16 times slower than
  in bfloat16 (next table). On the same 8 observations and noise, the 32-bit
  chunks differ from the bfloat16 ones by at most 0.016 in the executed
  position commands (relative control, in the environment's units of -1 to 1)
  and never in the gripper's direction (`profile_fp32.json`, `against`). I did
  not run 32 bits in closed loop.
- **A row costs nearly the same at any batch size**: 483 ms alone, 417 ms in
  a batch of four, 377 ms in a batch of eight. Batching buys 14 to 22% per
  row, not more, because the work is large already for one row (next table).

Where the time of one call goes, bfloat16 (milliseconds, `stages_ms_median`):

| Stage | 1 row | Share | 4 rows | Share | 8 rows | Share |
|---|---|---|---|---|---|---|
| Preparation on the CPU (images to tensors, state, normalisation, tokenizer) | 18 | 4% | 22 | 1% | 27 | 1% |
| Image resize and scaling on the GPU | 0.3 | 0% | 0.4 | 0% | 0.5 | 0% |
| **SigLIP** on three images (two cameras and one empty), in 32 bits | 133 | 28% | 460 | 28% | 802 | 27% |
| **Language model over the prefix** (Gemma 2B, 18 layers, 968 tokens), which fills the key-value cache | 188 | 39% | 817 | 49% | 1,592 | 53% |
| **Denoising loop**: 10 steps of the action expert (Gemma 300M) over 50 action tokens, attending to the cache | 138 | 29% | 368 | 22% | 594 | 20% |
| Copy back, conversion to environment actions | 1 | 0% | 1 | 0% | 1 | 0% |
| **Whole call** | **483** | | **1,666** | | **3,019** | |

The same in float32: SigLIP 133, 456 and 1,013 ms (it is in 32 bits in both
formats), language model 3,070, 9,949 and 19,784 ms, denoising loop 250, 997
and 1,633 ms.

- **The vision-language prefix is three quarters of a call**, the action
  expert's loop one quarter. This is the reverse of X-VLA, whose sampling loop
  was 81% of a call: pi0.5 reads the images once per call and caches the
  result, X-VLA runs its whole transformer again at every step.
- **400 of the prefix's 968 tokens are masked out** and computed anyway: 256
  for the empty third camera (configured as `empty_cameras: 1`, an image of
  -1 that the attention mask hides) and on average 144 of padding, because
  the instruction is padded to 200 tokens and the real prompt is about 56
  (`prefix_tokens_real_mean`: 568). The third image also costs a third of
  SigLIP's time. Leaving masked tokens out would compute the same function
  with fewer operations (my reading; not tried, filed as an idea).
- SigLIP has one convolution (its patch embedding), so the pinned MIOpen
  database holds one entry per batch size. SigLIP on one camera's image took
  0.9 s in the call that searched and 45 ms afterwards (`profile_bf16.json`,
  `pin`).

## 2. Nominal success

`pipeline.sh main` (`fleet.py`, batch size 1, three workers, both videos).
The episodes are the X-VLA spike's: for `libero_spatial` the walking
skeleton's 200 healthy episodes (`healthy200`: 10 tasks, start states 0 to 11
of one robot and 0 to 3 of two more, each robot with its own scene seed), for
`libero_object` 10 tasks x start states 0 to 19 (`object200`). X-VLA ran
exactly these episodes in 32 bits (`h200v_fp32_r4w10`, `p_obj_fp32` in
`~/.cache/robots-probes/xvla-speed/runs/`), so the two policies are compared
pair by pair (`analyze.py paired`, McNemar's exact test as in the X-VLA
spike).

These are not the start states of the baseline spike's X-VLA reference
(192/200 and 197/200): LeRobot's evaluation loop there did not record which
stored states it used (`spikes/baseline/README.md`, "Start states"). The
fleet's lists are the nearest well-defined equivalent, and X-VLA's results on
them are within one or two episodes of its reference.

| Same episodes | Pairs | X-VLA, 32 bits | pi0.5 | Both | Only X-VLA | Only pi0.5 | Neither | Exact paired test, p | pi0.5 minus X-VLA, points (95% interval) | Mean steps, X-VLA / pi0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `libero_spatial`, the skeleton's 200 | 200 | 194 (97.0%) | **195 (97.5%)** | 189 | 5 | 6 | 0 | 1.00 | +0.5 (-3.1 to +4.2) | 112 / 106 |
| `libero_object`, 10 tasks x start states 0 to 19 | 200 | 199 (99.5%) | **193 (96.5%)** | 192 | 7 | 1 | 0 | 0.07 | -3.0 (-6.6 to -0.1) | 135 / 142 |

(`results/paired_xvla32.md`; runs `h200v_pi05_r1w3`, `o200v_pi05_r1w3`.)
Against X-VLA's recommended 16-bit variant the numbers are the same: 195
against 195 on `libero_spatial`, 199 against 193 on `libero_object`.

Per task (`results/tasks.md`):

| Run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| pi0.5, `libero_spatial` | 20 | 20 | 19 | 19 | 19 | 19 | 20 | 20 | 19 | 20 |
| X-VLA, `libero_spatial` | 20 | 19 | 20 | 20 | 18 | 19 | 20 | 19 | 19 | 20 |
| pi0.5, `libero_object` | 20 | 20 | **17** | 19 | 18 | 20 | 20 | 19 | 20 | 20 |
| X-VLA, `libero_object` | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 19 | 20 | 20 |

- **On `libero_spatial` pi0.5 and X-VLA are equal**: 195 against 194, and no
  episode fails for both. LeRobot's LIBERO documentation reports 97% for
  pi0.5 on this suite (10 episodes per task, batch size 1).
- **On `libero_object` pi0.5 is three points behind**, 193 against 199: seven
  episodes fail only for pi0.5, one only for X-VLA. A fair coin splits seven
  against one this unevenly 7 times in 100 (p = 0.07), and the interval for
  the difference just excludes zero, so this is probably a real, small loss,
  not a settled one. LeRobot's documentation reports 99% here. I do not know
  which part of my setup costs the difference, if any: the rebuilt tokenizer,
  the 360-pixel render (training images are 256), 10 actions per chunk, or
  chance; I did not test them.
- **Every failure ran to the 280-step limit** with the gripper closing and
  opening again and again: 2 to 21 changes of the gripper command in the 12
  failed episodes (median 6; per run in `results/gripper.md`, written by
  `analyze.py motion`). On `libero_spatial` 189 of the 195 successes
  close the gripper once and never open it; on `libero_object` successes
  regrasp too (median 2 changes, up to 12). So the failures are grasp
  attempts that never come off, not an arm that wanders away (my reading of
  these counts and of five contact sheets). Videos of both cameras
  for every episode are in each run's `log/video/`; the failed ones are
  listed in `results/failures.md`, and contact sheets of five are in
  `results/frames/` (`figures.py sheet`).
- **Every success number in this report is at batch size 1.** The batch size
  changes a chunk's bits (section 3), so at another batch size single
  episodes could come out differently. The only other data point is the
  24-episode smoke run at four free rows (`smoke_bf16_r4w6`): healthy 8 of 8,
  camera fault 8 of 8, delay 0 of 8; the same 24 episodes at batch size 1
  (`smoke_r1w2`) had the same outcome episode by episode.
- **pi0.5's episodes are as long as X-VLA's** (106 and 142 steps on average),
  but it looks again every 10 steps where X-VLA looks every 30: 11.0 and 14.6
  calls per episode against 4.2 and 5.0.

## 3. Replayability

`pipeline.sh rows` (`rows_pi05.py`, adapted from the X-VLA spike's
`rows_check.py`: the 80 recorded observations, each with its own instruction
and noise, laid out in batches; a comparison counts only if all 50 x 7 numbers
of the chunk are the same bits).

**Does LeRobot's pi0.5 take an externally supplied noise sample? Yes.**
`PI05Policy.predict_action_chunk(batch, noise=...)` hands keyword arguments on
to `sample_actions`, which uses `noise` as the starting point of its sampling
loop instead of drawing one. The keyword is not in the type declaration of
those arguments (`ActionSelectKwargs` lists only the real-time-chunking ones),
but Python passes it. Checked two ways: a copy of `sample_actions` that feeds
the noise into the loop explicitly gives the same bits as LeRobot's call in
every batch size and both formats (`timed_equals_lerobot_call`), and calls
with the same observation and noise repeat bit for bit. pi0 (`PI0Policy`)
passes it the same way (read from the code). So no
copy of the sampling loop is needed, unlike for X-VLA.

**Is a chunk the same across processes, rows and batch mates?**

| Compared, bfloat16 | Batch size 1 | Batch size 4 | Batch size 8 |
|---|---|---|---|
| The same batch again, same process | identical | 4/4 | 8/8 |
| **The same batches in another process** | **80/80** (`rows_bf16_p3` against `_p2`) | **80/80** (`rows_bf16_p2` against `rows_bf16`) | not run |
| Same content in another row, same mates | (one row only) | **0/80**, up to 1.8 | **0/80**, up to 1.8 |
| Same row, other mates in every other row | | 80/80 | 80/80 |
| Both at once | | 0/80 | 0/80 |
| Half the batch filled with copies of row 0 | | 40/80 (the copies sit in other rows) | 40/80 |

Differences are in the policy's normalised action units (standard deviations
of the training data's actions); 1.8 is a different command, not rounding.

**So in bfloat16 a chunk depends on its row position, not on its batch mates,
and it does not depend on the process.** Two consequences for the fleet:
free rows, which the X-VLA runner uses, are not replayable with pi0.5 as
LeRobot runs it; fixed rows (an episode keeps its row, as in the walking
skeleton) or batch size 1 are. The main runs below use batch size 1.

**The batch size changes a chunk too**, as for X-VLA: the first 16 contents
at batch sizes 1, 2, 8 and 16 against their chunk at batch size 4, 0 of 16
identical each, by up to 0.04 to 0.08. So the batch size stays part of an
episode's identity.

**Which layers** (`rows_diag_pi05.py`, the X-VLA spike's method: every layer of
one real call of four rows runs a second time on its own input with the rows
rotated, so a difference can only come from that layer):

| Gives a row other bits in another position | Position-free |
|---|---|
| Language model, all 18 layers: the query projection (2,048 to 2,048), the attention output (2,048 to 2,048), the MLP's gate, up and down projections (2,048 to 16,384 and back). Action expert: the query projection (1,024 to 2,048), in all 180 calls | Language model's key and value projections (2,048 to 256); the expert's other projections; every SigLIP layer and the image projector (all in 32 bits); the attention computations of all three stacks; norms, activations, embeddings, the convolution |

All of the position-dependent ones are bfloat16 matrix products with a wide
output. **Storing the weights transposed**, which cured X-VLA's 16-bit
products, **does not cure these** (`rows_bf16_lt`: 0/80). **Switching
PyTorch's matrix library from hipBLASLt (its default here) to rocBLAS does**
(`torch.backends.cuda.preferred_blas_library("cublas")`, `rows_bf16_rocblas`:
80/80 in all four tests), but a call took 63% longer: 1,010 against 610 to
630 ms at batch size 1 and 3,158 against 1,957 ms at batch size 4, measured
at the same load average of about 20. My reading: hipBLASLt splits some
bfloat16 products differently for different parts of the output, so where a
row lands decides the order of its additions; I did not look into the library.

**Closed loop.** `pipeline.sh main` ends with `smoke_r1w2`: 24 episodes that
the main runs had already run (8 healthy, 8 with the camera fault, 8 with the
delay; same identity), run again at batch size 1 with two workers instead of
three and without video, in new processes and in another order.
`analyze.py compare` (the X-VLA spike's comparison: hashes of the simulator
state and of both camera images after every step, the full robot state, the
commanded and executed actions, the furniture, length and outcome):

| Compared | Pairs | With a fault | Run by another worker slot or at another point of a worker's life | Identical |
|---|---|---|---|---|
| `smoke_r1w2` against `h200v_pi05_r1w3`, `c6av_pi05_r1w3`, `d17v_pi05_r1w3` | 24 | 16 | 21 | **24** |

(`results/replay_smoke.md`.) So at batch size 1 pi0.5's episodes replay
exactly, as X-VLA's do with free rows.

## 4. Robustness under the two faults

`pipeline.sh main`, the same runner and settings as section 2, on exactly the
episodes behind X-VLA's 55.2% and 72.5% (`spikes/xvla-speed/README.md`,
section 8): the scene camera turned by 6 degrees on `libero_spatial`, 10 tasks
x start states 0 to 39 (`cam6_a`, `cam6_b`), and every action executed 17
control steps late, 10 tasks x start states 0 to 19 (`delay17`). Same task,
start state, scene seed and fault for both policies, paired as in section 2.
The fault code is the X-VLA spike's, copied unchanged. I checked that the
camera fault reaches pi0.5: in all 40 pairs of episodes that share a scene
with and without the fault (robot 2's healthy episodes in `healthy200`), the
first scene image differs and the wrist image, the simulator state and the
furniture are the same (`results/cam_fault_reaches_policy.txt`, from
`figures.py camcheck`; picture:
`results/frames/scene_healthy_vs_cam6_t0_i0.png`).

| Same episodes | Pairs | X-VLA, 32 bits | pi0.5 | Both | Only X-VLA | Only pi0.5 | Neither | Exact paired test, p | pi0.5 minus X-VLA, points (95% interval) | Mean steps, X-VLA / pi0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Camera turned 6 degrees | 400 | 221 (55.2%) | **391 (97.8%)** | 214 | 7 | 177 | 2 | below 0.001 | +42.5 (+37.2 to +47.6) | 188 / 107 |
| Actions 17 steps late | 200 | 145 (72.5%) | **0 (0.0%)** | 0 | 145 | 0 | 55 | below 0.001 | -72.5 (-78.2 to -65.7) | 233 / 280 |

(`results/paired_xvla32.md`; runs `c6av_pi05_r1w3`, `c6bv_pi05_r1w3`,
`d17v_pi05_r1w3`. X-VLA's recommended 16-bit variant gives the same totals on
these episodes, 221 and 142.)

**The camera fault costs pi0.5 nothing that 400 episodes can show**: 97.8%
with the camera turned, 97.5% on the healthy `libero_spatial` episodes of
section 2. X-VLA drops from 97% healthy to 55%. By task:

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X-VLA | 22 | 35 | 30 | 38 | 21 | 22 | 0 | 4 | 37 | 12 | 221 |
| pi0.5 | 40 | 37 | 40 | 40 | 40 | 37 | 40 | 40 | 37 | 40 | 391 |

(of 40 each, `results/bytask_cam6.md`). Where X-VLA fails most (task 6: 0 of
40, task 7: 4 of 40) pi0.5 succeeds in 80 of 80. Why pi0.5 is so much less sensitive I did
not investigate. A guess: openpi's training code augments the scene image
with a random crop to 95% and a rotation of up to 5 degrees
(`src/openpi/models/model.py` on GitHub, read on 2026-10-07), and this
checkpoint descends from openpi's weights (its model card names
`lerobot/pi05_libero_base` as the base). Its own fine-tuning in LeRobot had
image augmentation switched off (`train_config.json`). Whether the base
model saw such augmentation on LIBERO I did not check.

**The action delay defeats pi0.5 completely: 0 of 200**, where X-VLA solves
145 of the same episodes: at least 13 of 20 in eight of the ten tasks and
none in tasks 4 and 9 (`results/bytask_delay17.md`). Every pi0.5 episode ran
to the 280-step limit. What the arm does (from the step logs of all 200,
against the 200 healthy `libero_spatial` episodes): it still closes the
gripper in 191 of 200 episodes, but changes the gripper command 10 times per
episode (median) against once, and moves further from where it started (0.51
against 0.43 m at the furthest, medians), although more slowly on average
(path length over the episode's duration: 0.129 against 0.188 m/s, medians;
`results/gripper.md`). Picture:
`results/frames/smoke_k3-t0-i0.png`, an episode of the smoke run: the hand
travels about 10 cm further along the table than it ever does in the healthy
episode of the same task and start state (`k1-t0-i0`), and in 280 steps it
closes the gripper for 3.

My reading of why, not tested beyond these numbers: pi0.5 commands small
movements relative to where the hand is (relative control, as LeRobot's
LIBERO environment gives it); X-VLA commands where the hand should be
(absolute control). A movement executed 0.85 s late is applied to a hand that
has moved since, so corrections overshoot and the loop oscillates; a target
pose executed late is still the right target, only reached later. Until the
delay line has filled, the fault repeats the episode's first command for 17
steps; that does not derail the episode: after the first 17 steps the hand is
3.5 cm from its start (median) against 15.7 cm in healthy episodes, so the
warm-up mostly holds it still. A second relative-control policy fails the same
way: SmolVLA (10 of its 50 actions executed per chunk) solved 0 of 80 of these
delay episodes (start states 0 to 7) in the X-VLA spike's faulted mix
(`~/.cache/robots-probes/xvla-speed/runs/m200v_smolvla_r4w10`), where X-VLA
solved 56 of the same 80 in 32 and in 16 bits (`m200v_fp32_r4w10`,
`m200v_fp16res32_r4w10`) and pi0.5 0. **So the delay fault
at 17 steps is not the same severity for the two policies**: it was chosen for
X-VLA, and for a relative-control policy it is a different, much harsher
fault.


## 5. Episodes per hour in the fleet layout

The adapter exists, so every run above is a fleet run: the X-VLA spike's
runner, one inference process, simulation workers in their own processes,
both videos and all tables written. Batch size 1 and three workers, because
batch size 1 is what replays (section 3) and with one row three workers keep
it busy. Load average at the start of each run between 6 and 13 (other
agents' jobs).

| Run | Episodes | Successes | Per hour, end to end | Steady state | Steps / calls per episode | One call, ms | Server busy | Workers waiting | Process memory peak | GPU memory: run / machine |
|---|---|---|---|---|---|---|---|---|---|---|
| `libero_spatial`, healthy (`h200v_pi05_r1w3`) | 200 | 195 | **552** | 579 | 106 / 11.0 | 540 | 97% | 88% | 11.2 GiB | 11.6 / 18.4 GiB |
| `libero_object`, healthy (`o200v_pi05_r1w3`) | 200 | 193 | **440** | 461 | 142 / 14.6 | 522 | 98% | 91% | 11.8 GiB | 11.3 / 18.6 GiB |
| Camera 6 degrees, first half (`c6av_pi05_r1w3`) | 200 | 197 | 614 | 649 | 105 / 11.0 | 487 | 97% | 90% | 11.3 GiB | 11.8 / 17.1 GiB |
| Camera 6 degrees, second half (`c6bv_pi05_r1w3`) | 200 | 194 | 603 | 643 | 109 / 11.3 | 481 | 97% | 91% | 11.3 GiB | 11.7 / 18.1 GiB |
| Actions 17 steps late (`d17v_pi05_r1w3`) | 200 | 0 | 249 | 257 | 280 / 28.0 | 495 | 99% | 92% | 11.3 GiB | 11.6 / 17.5 GiB |

(`results/summary.md`, from each run's `run.json`.)

- **The GPU is the whole limit.** One row is in use in every call, the server
  is busy 97 to 99% of the time, and the workers wait 88 to 92% of theirs. A
  call in the fleet takes 481 to 540 ms against 483 ms alone. So the rate is
  simply about 1.9 calls per second divided by the calls an episode needs: 11
  for a healthy `libero_spatial` episode, 14.6 for `libero_object`, 28 for
  one that runs to the limit.
- **Against X-VLA on the same 200 healthy `libero_spatial` episodes**, end to
  end with video (`spikes/xvla-speed/README.md`, section 5): 4,962 per hour
  in X-VLA's recommended 16-bit setting and 1,856 in 32 bits, against 552 for
  pi0.5: **a ninth of X-VLA in 16 bits, under a third of X-VLA in 32 bits.**
  Two factors, each measured: a pi0.5 call for one row costs about what an
  X-VLA 16-bit call for four rows costs (483 against 479 ms), so a chunk is
  about four times dearer; and pi0.5 looks again every 10 steps where X-VLA
  looks every 30, so it needs 2.6 times as many chunks (11.0 against 4.2 per
  episode).
- **Faults that make episodes run to the limit cost in proportion**: pi0.5
  under the delay fails every episode at the full 280 steps and manages 249
  per hour. A faulted mix like the X-VLA spike's (40 healthy, 80 camera, 80
  delay) would run at about 360 to 400 per hour (calculated: these runs'
  calls per episode, 11.0, 11.1 and 28.0, give 17.9 for that mix, at the 1.8
  to 2.0 calls per second they reached).
- **Memory is modest**: the run's processes peaked at 11.2 to 11.8 GiB of
  process memory and 11.3 to 11.8 GiB of GPU memory (machine-wide at most
  18.6 GiB, with other agents' jobs). The inference process holds 2.5 to 5.7
  GiB after loading and 9.1 GiB on the GPU; loading it takes about 60 s.
- **Batches with fixed rows would buy little**: by the call times of section
  1, four rows in one call give 2.4 chunks per second against 2.1 for single
  rows, about 16% more (calculated; with free rows replay is lost, section
  3). In the 24-episode smoke run with four free rows and six workers a call
  took 1.84 s: 2.2 chunks per second for full calls, 1.9 as run, with 3.56 of
  4 rows filled on average (`smoke_bf16_r4w6`).
- **Against the plan** (my reading): medium worlds need about 2,100 episodes
  per hour at faulted-fleet success rates (`spikes/xvla-speed/README.md`).
  pi0.5 delivers 250 to 650 depending on how long its episodes run, so as the
  only policy it fits small worlds at best. As the second policy beside X-VLA,
  for a share of the episodes, it costs about nine X-VLA episodes of machine
  time per pi0.5 episode.


## 6. pi0 alongside

`pipeline.sh pi0`. `lerobot/pi0_libero_finetuned_v044`, revision `45dcc8fc`,
downloaded from the Hub (`ckpt/pi0_sha256.txt` equals the Hub's hash), with
the same rebuilt tokenizer and the same two settings as pi0.5 (10 actions per
chunk, the checkpoint says 50; `torch.compile` off), bfloat16, batch size 1,
three workers.

| | pi0 | pi0.5 |
|---|---|---|
| Parameters | 4.03 billion (3.61 in bfloat16, 0.42 kept in 32 bits) | 4.14 billion |
| Process memory while loading / after; GPU memory after loading | 17.0 / 4.9 GiB; 8.7 GiB | 17.5 / 5.3 GiB; 9.1 GiB |
| One call, batch size 1 / 4 / 8 | 428 / 1,446 / 2,632 ms | 483 / 1,666 / 3,019 ms |
| Healthy `libero_spatial`, 10 tasks x start states 0 to 4 (robot 1 of `healthy200`) | **35 of 50 (70%)** | 48 of 50 (96%); X-VLA on the same 50: 50 in `s100_fp32_r8w10` and in `h200v_fp16res32_r4w10`, 49 in `h200v_fp32_r4w10` |
| Episodes per hour on those, end to end | 437 (522 steady), 160 steps and 16.3 calls per episode | |

(`results/profile_pi0_bf16.json`, `results/paired_pi0.md`, run
`s50v_pi0_r1w3`; load average 1.7 to 4.6.) Against pi0.5 on the same 50
episodes: 14 succeed only for pi0.5, 1 only for pi0 (p below 0.001). By task
pi0 solved 4, 3, 5, 4, 2, 3, 4, 5, 1 and 4 of 5. I did not look into why, nor
try pi0 with the 50 actions per chunk its checkpoint asks for; the stage times,
the row check and replay were not measured for pi0.

## Side findings

- **The fleet's memory guard fired once.** The first healthy run (four rows,
  six workers) was stopped by the runner itself after a few minutes, with
  0.2 GiB of memory available on the machine: other agents' light jobs had
  grown to about 20 GiB beside it (`logs/aborted_h200v_pi05_r4w6_lowmem.log`).
  Two more attempts stopped: the queued `libero_object` run of the same
  layout, which I stopped while it loaded (`logs/aborted_o200v_pi05_r4w6_stopped.log`),
  and a run with four workers that the runner refused to start twice for lack
  of memory before I went to three (`logs/refused_h200v_pi05_r1w4.log`). No
  episode of the three is used. The X-VLA runner's start check counted 4 GiB for
  the inference process, pi0.5's is 15.5 GiB with its GPU memory, which on
  this machine is system memory. My runner now counts that, checks again after
  loading, and `fleet.sh` waits for memory before it queues for the lock.
- **LeRobot builds pi0.5 in 32 bits in process memory before converting it**,
  so loading needs 17.5 GiB of process memory even for bfloat16.
- **The checkpoint asks for `torch.compile` in its slowest mode**
  (`compile_model: true`, `max-autotune`) and for 50 actions per chunk.
  LeRobot's evaluation would honour both unless told otherwise.

## What I did not verify

About the policy and its setup:

- **The gated tokenizer.** Mine matches the public SentencePiece file on 250
  prompts; I could not compare it with `google/paligemma-3b-pt-224` itself.
- **Why pi0.5 is three points behind on `libero_object`**, or whether it is:
  p = 0.07 on 200 pairs. Not tested: 50 actions per chunk as stored, or 5 as
  openpi's LIBERO example does (`replan_steps = 5`); renders of 256 pixels;
  another seed.
- **float32 in closed loop.** It loads and runs (section 1) but at 3.5 s per
  call I did not spend the machine time.
- **`torch.compile`**, which the checkpoint asks for: switched off throughout.
- **Start states of the baseline spike's X-VLA reference** (192/200, 197/200):
  not reproducible, because they were not recorded; I compared on the X-VLA
  spike's episode lists instead, where X-VLA's own results are within two
  episodes of that reference.

About the faults:

- **Other levels of either fault for pi0.5.** Where pi0.5 starts to fail
  under a shorter delay, and where under a larger camera turn, is not
  measured. Runs at delays of 2 and 5 steps (50 episodes each) and the camera
  at 9 degrees were planned and dropped, because other agents' jobs were
  queueing for the heavy-job lock. Without them the delay result says only
  that 17 steps is far beyond what pi0.5 tolerates.
- **Why the camera fault does not hurt pi0.5**: not investigated (a guess in
  section 4).

About replay:

- **Closed-loop replay beyond 24 episodes** and beyond the two runs compared
  (other worker counts, video on and off). The X-VLA spike compared 1,200
  pairs; this is a first check, not that.
- **Fixed rows with batches of four** (replayable by section 3's numbers,
  about 16% faster by section 1's call times) were not built or run.
- **One machine state**: one ROCm and PyTorch version, one day, no reboot.
  Whether hipBLASLt's row dependence appears or goes with another version is
  open; `rows_pi05.py` takes about four minutes and belongs to the guard
  rails after any update.
- **Why hipBLASLt does it.** I measured which layers, not the library's
  reason.

About speed:

- **A quiet machine.** Every fleet run shared the machine with other agents'
  jobs (load 6 to 24 at the start of runs); single numbers are good to about
  10%. The policy-alone timings in section 1 were taken nearly quiet.
- **More workers or other batch sizes end to end.** With one row the server is
  busy 97 to 99% of the time and three workers wait 88 to 92% of theirs, so
  more workers cannot help (calculated from these shares); batch sizes above 1
  were measured only in the 24-episode smoke run.
- **The idea of leaving the masked prefix tokens out** and
  other speed-ups: 20 actions per chunk, fewer denoising steps, SigLIP in 16
  bits. None was tried; each changes what the policy computes.

## Follow-up items

- bfloat16 products here make a chunk depend on its row in the
  batch (hipBLASLt); batch size 1 or fixed rows for replay; rocBLAS cures it
  at 63% more time per call.
- (idea) Leave pi0.5's 400 masked prefix tokens out of each call.
- SmolVLA's row dependence might have the same
  cause; one row check with rocBLAS would tell.

## Files

| File | Purpose |
|---|---|
| `pipeline.sh` | The commands of every step; the tables name the step. Written as the record, not run from top to bottom. |
| `run.sh`, `fleet.sh` | One job behind the heavy-job lock: a script, or a fleet run. `fleet.sh` waits until enough memory is available before it queues for the lock, and retries when the runner refuses to start. |
| `p5_common.py` | Environment set-up, pinned reset, hashes, memory readers (copied from the X-VLA spike), this spike's paths. |
| `make_tokenizer.py` | Rebuilds the PaliGemma tokenizer from the public file, checks it, compares it part by part with the baseline spike's rebuild, points the local checkpoints at it. |
| `policy_pi05.py` | pi0.5 and pi0 behind the X-VLA runner's policy interface, with the noise passed in; `chunks_timed` for the stage times; switches for transposed weights, rocBLAS and row-by-row computing. |
| `profile_pi05.py` | Section 1: load, memory, call time by batch size, stages; pins the MIOpen database. |
| `rows_pi05.py`, `rows_diag_pi05.py` | Section 3: row position, batch mates, batch size, other processes; and which layers. |
| `fleet.py` | The X-VLA spike's runner for pi0.5 and pi0. Changed: the policy, relative control, the policy loads before the workers start, pi0.5's memory in the start checks, the checkpoint in `run.json`. |
| `manifests.py`, `faults.py` | Copies; `manifests.py` adds `spatial50` and the lower delays and otherwise gives byte-identical episode lists. |
| `analyze.py` | Copy with the two policies as the sides of the paired tables, X-VLA's runs looked up in its spike's output area, a list of failures, and the gripper and hand statistics (`motion`). |
| `figures.py` | The contact sheets, the camera-fault picture and the check that the camera fault reaches the policy. Written as one-off commands during the spike and collected here afterwards; they reproduce the same files. |

Outputs are in `~/.cache/robots-probes/pi05/`: `runs/` (each with `log/`
with steps and videos, `truth/` and `run.json`), `results/` (profiles, row
checks, chunks, the tables in Markdown, contact sheets in `frames/`), `logs/`,
`manifests/`, `miopen/pinned/`, `ckpt/` (the two checkpoints and their
hashes), `tokenizer/`, `venv/` and `venv.freeze.txt`.
