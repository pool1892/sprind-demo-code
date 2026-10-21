# Walking skeleton: from one fault to a scored diagnosis

Spike run on 2026-10-06 on this machine. It is the first run
of the whole idea in miniature: a small fleet with faults, the truth obtained
by replay, a log, a blind agent and a mechanical score. It is built from the
day's measurement code before any interface is fixed, so that the interfaces
can learn from it. It is glue with honest numbers, not a framework.

Everything below was measured here unless it says otherwise. The simulator and
the policy ran in the baseline spike's environment
(`~/.cache/robots-probes/baseline-venv-mj337`: MuJoCo 3.3.7, robosuite 1.5.2
port with the absolute-control correction applied at run time, LeRobot 0.6.1,
PyTorch 2.11.0 on ROCm 10.0.0). The policy is X-VLA, `lerobot/xvla-libero`
revision `12e8783e`, used as stored: 32-bit floats, 10 denoising steps, 30
actions per call. The agent is Claude Opus 5.5 through `claude` 2.1.292.

Words used throughout:

- An **episode** is one attempt at a task: one of the ten `libero_spatial`
  tasks, one of LIBERO's stored start states for it, and at most 280 control
  steps of 50 ms. An episode that is not done after 280 steps has failed.
- A **chunk** is what one call of the policy returns: the next 30 actions. The
  policy is called again when they are used up.
- A **row** is one episode's place in a batch, when the policy computes chunks
  for several episodes in one call.
- **Identical** always means bit for bit, compared through hashes or byte by
  byte, never with a tolerance.
- A **replay** runs an episode again from its reset with the same identity
  (seeds, batch size, row) and a fault set of our choosing.
- The **truth** is everything the agent must not see: which robot carries
  which fault, the seeds, the replays' outcomes, simulator internals.
- The **workspace** is the directory the agent's tools can read.

## Verdict

**The chain ran end to end, and replay is exact inside the multi-process
layout.**

1. **Exactness.** Across nine runs in separate sets of processes, 496
   pairs of episodes with the same identity were compared and all were
   identical: every action, the simulator state and both camera images after
   every step, the outcome. (The MP4 files are the exception: the encoder does
   not repeat itself, see section 5.) The two checks asked for: the 40 replays of the
   healthy robot equal the originals (40 of 40), and the 80 faulted episodes
   replayed without their fault equal the same 80 episodes run healthy from
   the start in workers that never saw a fault (80 of 80).
2. **Two faults with a usable dose-response.** The scene camera turned by 6
   degrees brings its robot to 24 of 40 (60%). Actions executed 17 control
   steps late bring theirs to 28 of 40 (70%). Each curve has five levels of 40
   episodes.
3. **Truth by replay, per episode.** Of robot 2's 16 failures the camera fault
   caused 15; one fails with and without it. Two of its episodes succeed
   *only* with the fault. Of robot 3's 12 failures the delay caused 11.
4. **Blind diagnosis.** Three runs of one Opus agent in the two nested
   sandboxes, each given the log of 120 episodes and a description of the
   system. All three named both faulted robots, the right component for each,
   the right fix, and raised no false alarm. Each took about 65 seconds, 16 or
   17 tool calls and about 90,000 input tokens. **This task is too easy to say
   much about the agent**: two large faults, one per robot, nothing misleading,
   and a description that lists the affected setting. See "What this does not
   show".
5. **Throughput for the sizing decision:** 1,577 episodes per hour end to
   end (1,683 in steady state) for a batch of 200 healthy episodes with full
   logging, with X-VLA under the replay rules. Peak process memory 17.5 GiB,
   peak GPU memory of the run's processes 10.4 GiB, load average 3 to 4 at the
   time. A fleet with faults is slower, because failed episodes run
   to the step limit: 1,074 per hour for the 120-episode fleet, 800 to 910 for
   the dose-response runs. The policy on the GPU is the bottleneck, not the
   simulation.
6. **Disk:** 0.29 MB per healthy episode for everything the agent may see
   (tables and both videos), 0.42 MB for the faulted fleet's longer episodes,
   plus 0.03 to 0.05 MB of truth-side records per episode. Disk is not a
   constraint at any world size in the plan.

## The scenario

Three robots at one site run X-VLA on `libero_spatial`. Each runs the same 40
episodes: ten tasks times stored start states 0 to 3. Every robot stands in its
own cell, so each has its own furniture placement (scene seed) and its own
policy noise.

| Robot | Fault | Class | Level | Success |
|---|---|---|---|---|
| `robot_1` | none | | | 40/40 |
| `robot_2` | the fixed scene camera is turned about the vertical axis, as if its mount had been knocked | changes the simulated cell | 6 degrees | 24/40 |
| `robot_3` | every commanded action is executed some control steps late | acts in the control loop | 17 steps (0.85 s) | 28/40 |

How the run is laid out (from the throughput spike): one process holds the
only copy of X-VLA on the GPU and serves chunks; eight simulation worker
processes hold one LIBERO environment each. The six replay rules as applied:

1. *Pinned scene.* numpy's global generator is seeded right before every
   reset (`sk_common.reset_pinned`).
2. *Per-episode noise.* Each row's starting noise is drawn on the CPU from the
   episode's (noise seed, chunk number). LeRobot's X-VLA wrapper cannot do
   this: it accepts a `noise` argument and ignores it. `policy_xvla.py` repeats
   the sampling loop with the noise passed in.
3. *Fixed batch size, fixed row.* Every call computes exactly eight rows. An
   episode's row is part of its identity, and the worker for that row is the
   only one that runs it. Rows with no request waiting are filled with a copy
   of another row and their output is dropped.
4. *One action queue per episode*, in the worker. An episode that ends leaves
   the loop; nothing resets on its own.
5. *Snapshots in mid-episode* are not used: every replay starts at a reset.
6. *One rendering context at a time.* One environment per worker process; a
   worker closes its environment before it builds the next task's.

Resets are soft (the MuJoCo model and the renderer are kept between episodes),
and a worker process is replaced after 16 episodes.

## Numbers

Every command is run from the repository root. `spikes/skeleton/pipeline.sh
STEP` holds the commands of each step; the step is named under each table.
I ran these commands one by one as the work went along; the script as a whole
was written afterwards as the record and has not been run from top to bottom.
Runs are in `~/.cache/robots-probes/skeleton/runs/<name>/`.

### 1. The policy alone

`pipeline.sh probe` (twice `tools/heavy-run ... probe_policy.py`, two processes)

| Question | Result |
|---|---|
| One X-VLA call for 1, 2, 4, 8 rows | 0.57, 0.90, 1.58, 3.06 s: about 0.21 s plus 0.36 s per row |
| Same batch again in the same process | identical |
| Same batch in a second process | identical (8 rows, 30 actions each) |
| A row's chunk when other rows hold other content | identical (two rows checked while the other six were changed) |
| The same content in another row of the batch | identical in the one case tried |
| The same content alone, batch of one | not identical, largest difference 5e-7 |
| Scene camera turned on the live model | 96% of the scene image's values change, the wrist image and the physics do not; after undoing it the next reset gives the identical images and state |

So rule 3 holds for X-VLA on the Radeon in its weak form: the batch size
matters, the other rows' content does not. SmolVLA's chunk also depended on
the row position (replay spike); X-VLA's did not in the one swap tried.

### 2. Dose-response of the two faults

`pipeline.sh dose` (four runs of 100 episodes, then `analyze.py dose`)

The same 40 episodes at every level, so the levels of one fault differ in the
fault and in nothing else (same scene, same noise).

Scene camera of robot 2 turned by:

| Degrees | Successes | Rate | 95% interval | Mean steps | Successes by task 0 to 9 (of 4) |
|---|---|---|---|---|---|
| 0 | 37/40 | 92.5% | 80 to 97% | 119 | 4 3 4 4 3 4 4 4 3 4 |
| 3 | 34/40 | 85.0% | 71 to 93% | 136 | 3 4 4 4 4 3 2 4 3 3 |
| **6** | **24/40** | **60.0%** | 45 to 74% | 182 | 2 3 3 4 3 4 0 1 4 0 |
| 9 | 10/40 | 25.0% | 14 to 40% | 233 | 3 3 0 3 0 0 0 0 1 0 |
| 12 | 7/40 | 17.5% | 9 to 32% | 260 | 4 2 0 0 0 0 0 0 1 0 |

Actions of robot 3 executed late by:

| Control steps | Successes | Rate | 95% interval | Mean steps | Successes by task 0 to 9 (of 4) |
|---|---|---|---|---|---|
| 0 | 39/40 | 97.5% | 87 to 100% | 110 | 4 4 4 4 3 4 4 4 4 4 |
| 8 | 40/40 | 100.0% | 91 to 100% | 147 | 4 4 4 4 4 4 4 4 4 4 |
| 14 | 38/40 | 95.0% | 83 to 99% | 198 | 4 4 4 4 3 3 4 4 4 4 |
| **17** | **28/40** | **70.0%** | 55 to 82% | 236 | 4 3 4 4 0 2 4 3 4 0 |
| 20 | 17/40 | 42.5% | 29 to 58% | 259 | 3 0 4 4 0 1 2 0 3 0 |

Both faults give a monotone curve within chance, and one level of each falls
into the wanted band of 40 to 75%: 6 degrees and 17 steps. For the delay, 20
steps (42.5%) would also have qualified; 17 is nearer the middle.

How the delay hurts, as far as these numbers show: it does not spoil the
grasp, it slows the robot. X-VLA commands target poses, so a late command is
still a correct one. But each new chunk is planned from a pose the arm has not
left yet, and its first part is spent on the stale end of the previous chunk.
Episodes that succeed with and without the 17-step delay take 217 steps
instead of 100. Up to 14 steps of delay there is time to spare inside the
280-step limit; from 17 on the slower tasks run out of it. That the mechanism
is exactly this is my reading of the step counts, not a separate measurement.

### 3. The fleet run and the truth by replay

`pipeline.sh fleet`, `pipeline.sh replay`, `pipeline.sh checks`

The 120 episodes ran as one batch with video (run `fleet`). Then every episode
was run again with its fault set emptied, read from the first run's truth
records, in new processes and in reverse order (run `replay`, 136 episodes:
the 120, plus every fifth faulted episode once more with its fault left in).

| Robot | As run | With the fault removed | Fault caused the failure | Failed either way | Succeeded either way | Succeeded only with the fault |
|---|---|---|---|---|---|---|
| `robot_1` | 40/40 | 40/40 | - | - | - | - |
| `robot_2` | 24/40 | 37/40 | 15 | 1 | 22 | 2 |
| `robot_3` | 28/40 | 39/40 | 11 | 1 | 28 | 0 |

The per-episode table is `runs/fleet/truth/attribution.json`. Two things in
it are worth knowing before the oracle is designed. A fault can rescue an
episode: `e0059` and `e0017` of robot 2 fail without the camera fault and
succeed with it, so the fault's net effect (13 episodes) is smaller than the
number of failures it causes (15). And one failure per faulted robot is not
the fault's doing; counting every failure of a faulted robot against the
fault would overstate it.

### 4. Exactness inside the layout

`pipeline.sh checks` (`analyze.py compare A B` matches episodes of two runs by
identity and compares every stored array byte by byte)

| Compared | What differs between the two runs | Pairs | Identical | Outcome flips |
|---|---|---|---|---|
| **Check 1.** Robot 1's episodes in `fleet` against their replays in `replay` | processes, order, batch mates, video on or off; another agent's job held the GPU in between | 40 | 40 | 0 |
| **Check 2.** Robots 2 and 3 replayed without their fault (`replay`) against the same episodes in `healthy200` | processes, order, batch mates, video; in `replay` 16 fault-free replays ran directly after a faulted episode on the same simulator, in `healthy200` no worker ever applies a fault | 80 | 80 | 0 |
| Faulted episodes replayed with the fault left in (`replay`) against `fleet` | as check 1 | 16 | 16 | 0 |
| Robots 2 and 3 in `fleet` against the same level in the dose runs | processes, order; in the dose runs each worker steps through five fault levels on one environment | 80 | 80 | 0 |
| Fault-free replays of robots 2 and 3 against level 0 of the dose runs | as above | 80 | 80 | 0 |
| Robot 1 in `fleet` against `healthy200` | processes, batch mates | 40 | 40 | 0 |
| Robots 2 and 3 in `healthy200` against level 0 of the dose runs | processes, batch mates | 80 | 80 | 0 |
| Robot 1's replays against `healthy200` | processes, order, batch mates, video | 40 | 40 | 0 |
| Faulted episodes replayed with the fault left in against the dose runs | processes, order | 16 | 16 | 0 |
| The day's first 24-episode test against the same 24 episodes run last, with the code as delivered | the server (the first one did not wait for its rows, and workers wrote files before asking), workers replaced after 2 or after 16 episodes, 80 minutes apart | 24 | 24 | 0 |
| **All** | | **496** | **496** | **0** |

"Identical" covers, per step: a hash of MuJoCo's complete integration state, a
hash of each camera image, the robot state in full precision, the commanded
and the executed action; and per episode the fixture layout, the length and
the outcome. No pair differed anywhere.

Whether a fault leaves something behind on a simulator that is reused is
tested most often by the fourth row. In `fleet` every worker runs robot 1,
robot 2 and robot 3 one after the other on the same simulator, so all 40 of
robot 3's episodes start right after a camera-faulted one. They equal the
dose runs' episodes, whose workers never turned a camera.

What this adds to the replay spike: X-VLA instead of SmolVLA, absolute control,
soft resets with the policy in the loop, an inference process that receives
requests at arbitrary times from eight workers, padded batches of eight rows,
worker processes that are replaced in mid-run, and faults that are switched on
and off on a live simulator.

### 5. The log, the truth, and their size

`pipeline.sh fleet` writes both; `pipeline.sh export` builds the workspace.

What the agent can see, and nothing else:

| Part | Content | Bytes per episode, fleet run (mean 176 steps) | Healthy batch (mean 111 steps) |
|---|---|---|---|
| Episode table (Parquet) | episode number, robot, task and its instruction, start state, length, duration, success | 33 | 21 |
| Per-step table (Parquet) | joint positions and velocities, finger positions and velocities, measured gripper pose, the commanded action, which chunk it came from; 37 columns, the measurements as 32-bit floats | 38,429 as one file per episode; 30,117 in one file for all | 28,292 |
| Scene camera video (MP4, H.264, 360x360, 5 frames per second) | every fourth control step and the last one | 112,048 | 82,135 |
| Wrist camera video | the same | 269,206 | 184,462 |
| **Sum** | | **0.42 MB** | **0.29 MB** |

That is 2.4 kB per control step, and the wrist video is 64% of it: its picture
changes completely as the arm moves. A failed episode costs 0.60 MB, a
successful one 0.36 MB in the fleet run.

What stays on the truth side, in a directory the workspace does not contain:

| Part | Content | Bytes per episode |
|---|---|---|
| `truth/episodes.jsonl` | identity (seeds, batch size, row), fault set, outcome, fixture hash, which worker process ran it, waiting times | about 700 |
| `truth/chain/<id>.npz` | per step: hashes of simulator state and of both images, robot state in 64-bit floats, commanded and executed action | 47,457 (fleet), 33,449 (replays) |
| `truth/fault_table.json`, `canary.txt` | which entry of the system description each fault belongs to, and its fix; a random marker | once per run |
| `truth/attribution.json` | the table of section 3 per episode | once per run |

Time in the log is robot time (step number times 50 ms). How long a worker
waited for the inference process is a property of the test bed, not of the
robot, and is kept on the truth side.

**The video files do not repeat, although the images do.** The 24 episodes of
the test manifest were run twice (runs `smoke` and `smoke_final`). Every
camera image was identical by hash, and all 24 per-step Parquet files are the
same bytes. Of the 48 MP4 files, 18 differ (12 scene, 6 wrist), and after
decoding 399 of their 636 frames differ, by up to 46 of 255 in a pixel value.
The encoder (libx264 through PyAV, threads left on automatic) is the only
thing between the identical images and the differing files. I did not look
for the cause or try a single-threaded encode.

### 6. The blind diagnosis

`pipeline.sh export`, `pipeline.sh diagnose`

**Before the first call.** `leak_check.py` searched the workspace for 16 words:
the two fault kinds, telling words such as "fault", "truth", "replay", "seed",
"delay", "degrees", and a random canary token that exists only in the truth
directory. It searched the raw bytes of all 242 files (49.4 MB), their names,
all 253 distinct text values and the column names of the Parquet files, 1,680
video tags, and the four texts the agent is given (system prompt, prompt,
answer schema, the tool server's source, which is readable inside the
sandbox). **No hit.** On the first try the search did report hits, all false:
"steps" (a column name and also the delay fault's parameter name) and "fault"
inside "default". The search now ignores parameter names and requires that no
letter stands before the word.

`sandbox_check.py` then ran three things in the same two nested sandboxes the
agent call uses, without a model:

| Seen from | Truth directory, fault table, canary, run directory, repository | Workspace | Login file | Network |
|---|---|---|---|---|
| the tool server's sandbox | hidden | visible | hidden | closed |
| the command-line tool's sandbox | hidden | visible | **visible** | **open** |

A real session with the tool server through both sandboxes worked (tables,
frames), and six attempts to get past `query` were refused (`read_csv` and
`read_text` on a path, `COPY`, `ATTACH`, switching file access back on, two
statements in one call).
The only parts of the home directory that exist inside are the tool server's
Python environment and interpreter. One call with a model on a test workspace
confirmed that eight frames per tool call arrive as images.

**The call.** `diagnose.py`: `run_agent("claude", "opus", ...)` with
`sandbox="bwrap"`, the tools `query(sql)` and `get_frames(episode, times)`, and
an answer schema. The prompt describes the three cells, seven components per
cell, fourteen settings with their nominal values, what is normal (192 of 200
in acceptance tests), the log, and seven generic actions. It asks, per robot:
problem or not, which component, which setting, what is wrong, how sure, on
what evidence, which action. It does not say how many robots have a problem.

**The score**, mechanical (`score.py`: every cell is a comparison of two
strings against `truth/fault_table.json`, which was written before the first
call):

| Run | Right robot | Right component | Right setting (robot 3) | Right action | False alarms | Seconds | Tool calls (queries / frame calls / images) | Input tokens (of which read from cache) | Output tokens |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2/2 | 2/2 | yes | 2/2 | 0 | 67 | 16 (12 / 3 / 8) | 84,682 (64,307) | 6,727 |
| 2 | 2/2 | 2/2 | yes | 2/2 | 0 | 62 | 17 (11 / 5 / 9) | 102,797 (80,635) | 6,025 |
| 3 | 2/2 | 2/2 | yes | 2/2 | 0 | 63 | 17 (10 / 6 / 18) | 88,511 (70,010) | 5,599 |

The tool call counts include the final call that hands in the answer. At list
prices the three runs would have cost 0.31, 0.31 and 0.27 dollars; the
subscription is not charged per call. The three ran at the same time.

How much the runs vary: not at all in the scored fields. Confidence was 0.80
to 0.85 for robot 2 and 0.85 to 0.88 for robot 3, and 0.90 that robot 1 is
healthy. The evidence was of the same kind in all three:

- *Robot 3.* The measured gripper position matches the command issued 21
  steps earlier, against 4 steps on the other two robots (4 steps is the arm's
  normal lag). All three runs computed this in SQL and concluded "about 17
  steps of buffering instead of 0", which is the injected value. I checked the
  21 and the 4 against the dose runs' logs myself: lag 4, 12, 18, 21, 24 for
  delays of 0, 8, 14, 17, 20.
- *Robot 2.* As the agents report it: command tracking is normal (lag 4), so
  arm and controller are ruled out; the scene camera's picture at the start
  of an episode is shifted and turned compared with the other two robots on
  the same task and start state, while the wrist picture is the same; the
  gripper closes 3 to 6 cm beside where robot 1 closes it. Of this I checked
  one thing: the first scene frames of `e0013`, `e0116` and `e0027` (task 6,
  start state 0, robots 1, 2, 3) side by side. Robot 2's picture is shifted
  by about 40 pixels and tilted; the other two match
  (`probe/t6_first_frames_r1_r2_r3.png`). The 3 to 6 cm I did not check.

After the runs, the canary and the two fault kinds were searched for in
everything the three runs left behind (command lines, event streams with all
tool results, tool call logs: 15 files). No hit.

### 7. Throughput and memory

`pipeline.sh healthy` (run `healthy200`), and the other runs' `run.json`

| Run | Episodes | Successes | Episodes per hour, end to end | In steady state | Steps / chunks per episode | Rows in use, of 8 | Peak process memory | Peak GPU memory: these processes / whole machine | Load average at start |
|---|---|---|---|---|---|---|---|---|---|
| **`healthy200`**, video | 200 | 195 | **1,577** | 1,683 | 111 / 4.2 | 7.0 | **17.5 GiB** | **10.4** / 13.4 GiB | 3.1 |
| `fleet`, video | 120 | 92 | 1,074 | 1,117 | 176 / 6.4 | 7.3 | 17.5 GiB | 10.1 / 13.4 GiB | 2.3 |
| `replay` | 136 | 127 | 1,355 | 1,630 | 122 / 4.6 | 6.7 | 17.5 GiB | 10.3 / 13.3 GiB | 4.1 |
| `dose_cam_p1` | 100 | 67 | 890 | 1,145 | 165 / 6.0 | 5.4 | 17.5 GiB | 9.8 / 12.9 GiB | 0.8 |
| `dose_cam_p2` | 100 | 45 | 811 | 924 | 207 / 7.5 | 6.2 | 17.5 GiB | 9.8 / 12.8 GiB | 4.1 |
| `dose_delay_p1` | 100 | 84 | 909 | 1,065 | 181 / 6.6 | 6.1 | 17.5 GiB | 10.0 / 13.6 GiB | 3.2 |
| `dose_delay_p2` | 100 | 78 | 804 | 922 | 199 / 7.2 | 5.9 | 17.5 GiB | 10.0 / 14.0 GiB | 5.5 |

The first row is the number asked for: 200 episodes in 457 s, with both videos
and all tables written. Load average over the run: 3.1 at the start, 4.3 at
the end (1-minute values), from other jobs on the machine and these nine
processes.

- "End to end" counts from the start of the runner to the last episode's
  files: loading the policy (5.5 s), starting eight worker processes, building
  84 environments, replacing each worker once, and the tail in which rows run
  out of work. "Steady state" is the rate over the middle 80% of episode
  completions.
- "Process memory" is the sum over the inference process and the eight
  workers of each one's proportional share (Pss), sampled once a second. The peak is the same 17.5 GiB in every run, a 24-episode test included, so
  it is reached at start-up; I take that to be the moment the policy file is
  loaded and did not confirm it. Free memory on the machine never fell below
  22 GiB during any run.
- "GPU memory, these processes" is what the kernel accounts to the nine
  processes. The whole-machine figure includes other users of the GPU; it stood
  at 3.1 GiB before the run.

How to read it:

- **The policy is the bottleneck.** One call costs about 3.1 s for eight rows,
  whether or not all eight are in use. A worker simulates its 30 steps in about
  0.3 s and then waits for the next call: workers wait 87 to 91% of their
  time. The GPU's ceiling is one chunk per 0.36 s per row; a healthy episode
  needs 4.2 chunks, a failed one 10.
- **Rows in use decide the speed under rule 3.** The first version of the
  server took whichever requests were waiting, as the throughput spike's
  prototype does. It used 2.6 of 8 rows per call and reached 586 episodes per
  hour on a 24-episode test, because every call costs the full eight rows.
  After two changes the runs with evenly loaded rows used 6.7 to 7.3 rows:
  after a call the server waits up to 0.65 s for the rows it has just served,
  and a worker writes an episode's files while it waits for the next
  episode's first chunk. The same 24 episodes with the final server took 78 s
  instead of 147 s and used 5.9 rows instead of 2.6. That is not a clean
  before and after: the first test also replaced every worker after two
  episodes, the last one after 16, and the two changes were not measured one
  at a time.
- **Faults make the fleet slower.** A failed episode runs 280 steps and needs
  10 chunks. Mean chunks per episode: 4.2 healthy, 6.4 in the fleet
  run, up to 7.5 in the dose runs.
- **Replays cost as much as the original.** X-VLA needs both images at every
  call, so a replay renders and infers like the first run. Only the video
  files are saved; writing all of an episode's files took 0.14 s in the fleet
  run and happens while the worker waits anyway.
- **Memory.** A worker holds 1.76 GiB after its first episode and grew by 1
  to 53 MiB per episode, typically about 10, over the 15 or 16 episodes of its
  life (soft resets), ending at 1.9 to 2.1 GiB; the inference process peaks at
  3.9 GiB. GPU memory of the nine processes: 10
  GiB, of which the inference process has 4.8 GiB (PyTorch 4.0 GiB) and each
  worker 0.59 GiB.
- All of this on a machine that other jobs were using (load average 1 to
  5). Two of my runs waited behind another job for the lock, 5 and 11
  minutes; those waits are not in the numbers.

Against the plan's sizing table, which asks for about 800 episodes per hour
for small worlds and about 2,100 for medium ones: the plan's fleets succeed
60 to 75% of the time. In the five runs here that succeeded 45 to 84% of the
time (the fleet run and the dose runs) this machine delivered 800 to 1,100
per hour with X-VLA under the replay rules,
and 1,400 to 1,600 where most episodes succeed (the replays and the healthy
batch). Small worlds fit with little to spare; medium worlds do not. The
ceiling is the GPU: at 0.36 s per row and 4.2 chunks per episode it would be
about 2,400 healthy episodes per hour if every row of every call were in use
and nothing else took time (calculated). The healthy batch reached two thirds
of that.

## What the interfaces must provide

Drawn from what this spike needed and from what was awkward.

**Fault hook points.**

1. *A cell hook that the engine calls at a defined moment of the reset.* The
   camera fault changes the live MuJoCo model. It has to be applied after the
   simulator exists and before the reset renders its first image, and undone
   after the episode, because with soft resets a model change outlives the
   episode. LeRobot's wrapper builds its simulator inside `reset()`, so I had
   to call its private `_ensure_env()` to get in between. With hard resets
   there is no such moment at all: the model is rebuilt inside the reset. The
   engine has to own the reset sequence: build, apply cell faults, set the
   start state, settle, first observation.
2. *An action hook between the policy's queue and the simulator*, with a call
   that clears its state when an episode starts. The delay fault is a queue of
   17 commands; that queue is part of the episode's state.
3. *An observation hook* (between the sensors and what the policy and the log
   receive) and *a hook on the policy side* were not needed for these two
   faults and will be for the next ones. They follow the same pattern.
4. *Every hook has to say on which side of it the log listens.* The log
   records the commanded action, before the delay; the executed action is
   truth. It records the camera image after the camera fault, because that is
   what the camera saw. A column "executed action" in the log would have
   given the delay away in one query. This belongs in the data dictionary,
   signal by signal.
5. *A fault is data*: a kind and parameters in the episode's record, with
   level zero meaning no fault. The runner's loop knows the two hook points
   and nothing about any fault; the two faults are the 110 lines of
   `faults.py`.
6. *A fault must undo itself completely, and that is tested, not assumed*:
   check 2 above, in every campaign.
7. *A separate table on the truth side maps each fault kind to the system
   description*: component, setting, fixing action (`score.py`,
   `TRUTH_OF_KIND`). It has to exist before any agent runs.

**Episode identity.** What had to be fixed for two runs to produce the same
episode: suite, task, start state, scene seed, noise seed, the size of the
inference batch, the row, the step limit, the fault set; and per run the
software versions, the policy revision, 30 actions per call, absolute control,
360x360 images, soft resets. Not part of it, and shown not to matter: the
worker process, what ran before in it, what ran beside it, the order, whether
video is written, which other rows of a call were in use.

- The visible episode number (`e0001`) is neutral and says nothing about the
  identity. The map from number to identity lives on the truth side. Replays
  need it, and so will experiments the agents request.
- Batch size and row belong to the identity, which ties an episode to a worker
  slot and a run to one worker count. For X-VLA the row may not matter (one
  swap tried); if that holds, any worker could run any episode.
- Seeds are a pure function of (robot, task, start state)
  (`manifests.identity`). The world generator needs the same property for
  whatever it adds: day, site, configuration version.

**What a snapshot contains.** This spike needed none: both faults act from the
first step and every replay starts at a reset, so the identity is the whole
snapshot, verified by the fixture hash and the hash chain. For a takeover or
a fault that starts in mid-episode, the lean set of the replay spike has to
grow by what this spike put between policy and simulator: the actions still
queued, the chunk number and the noise seed, the step count, every active
fault's internal state (the delay line's 17 commands; a cell fault's saved
original values), and in absolute control the arm controller's goal. None of
that was tested here.

**What the log must carry for replay.** Nothing, and it should not. The
agent-visible log holds no seed, no row and no fault set. Replay runs from the
truth-side record: identity, then the fixture hash and the per-step hash chain
to prove that the replay with nothing removed reproduces the original. That
record is 33 to 47 kB per episode. So there are three things to design, not
two: the log, the fault truth, and the replay record. And the proof of an
exact replay has to rest on the hashes of the raw images, not on the video
files: those differ between two encodings of the same images (section 5). A
sealed world's file hash therefore fixes one encoding; regenerating the world
would not reproduce it unless the encoder is made repeatable.

**How truth is kept apart.** In this spike by five means, none of them a
guarantee:

1. The worker writes two directory trees, `log/` and `truth/`. The visible
   episode table is built from a list of allowed columns, not by removing
   forbidden ones.
2. The export reads `log/` only and refuses a run whose episode ids are not
   neutral numbers (the dose runs name their episodes after fault levels).
3. A word search over the export, with the words taken from the truth side,
   and a canary; again after the agent runs.
4. The two sandboxes, checked without a model before every campaign.
5. The tool server's source contains nothing about how the data was made.

The weak point is that one process, the worker, knows both sides, and the
separation is the discipline of its writer. The word search finds names, not
numbers: a column whose values encode a fault would pass it.

**The engine's scheduling**, which no interface list had so far:

- A padded call costs the same with two rows in use as with eight. The engine
  has to keep rows full: balanced work per row, a short wait for returning
  rows, file writing moved into waiting time.
- A row's worker is replaced periodically. The row is empty for about 2.5 s
  (measured, 16 replacements) plus the time to build an environment, 1 to 2 s.
- Runs behind the shared lock should be minutes long. Splitting a manifest by
  task leaves the rows unevenly loaded (5.4 of 8 rows in use in one half).

**The agent side.**

- The tool server loads the Parquet tables into memory and then switches file
  access off; it needs its own Python environment, mounted read-only.
- `get_frames` needs the frame times to be derivable from the episode table
  (every fourth step and the last one), or a frame index in the log.
- The answer schema and the scorer share two vocabularies, components and
  action verbs. They must come from one place; here they are written twice
  (`diagnose.py`, `score.py`).

## Surprises

1. **LeRobot's X-VLA wrapper ignores the `noise` argument.** Exact replay
   needs a copy of its sampling loop (30 lines) until that is fixed upstream.
2. **With X-VLA the GPU is the bottleneck**, the reverse of the throughput
   spike's finding for SmolVLA. Workers wait about 90% of their time.
3. **The throughput prototype's batching rule and replay rule 3 do not go
   together.** A server that takes whichever requests are waiting used a third
   of its rows in a first test, while every call cost all eight (section 7).
4. **The action delay does not break grasps; it makes the robot slow**, and
   the success rate falls off a cliff between 14 and 20 steps when episodes run
   into the step limit. Below that it is invisible in the success rate.
5. **A fault can rescue an episode** (2 of robot 2's 40).
6. **The agent found both faults in about a minute, three times.** The log
   gives the delay away to anyone who lines up command and motion (21 against
   4 steps), and the camera fault shows in the first frame.
7. **The leak search's first hits were all false** ("steps", "default").
8. **The wrist video is 2.4 times the size of the scene video.**
9. **Worker memory grew by about 10 MiB per episode** (1 to 53 across 24
   worker lives), less than the 30 to 65 MiB per reset seen in the
   teleoperation spike. That spike used hard resets through its own
   environment set-up.
10. **The MP4 files of the same episode differ from run to run** while the
    images they were made from are identical (section 5).
11. **Robot 1 succeeded 40 times of 40**; the same tasks without fault on
    robots 2 and 3 gave 37 and 39 of 40. The healthy batch as a whole gave
    195 of 200 (97.5%, 95% interval 94 to 99%), against 192 of 200 in the
    baseline spike, which used hard resets and LeRobot's own loop.

## What this does not show

About the diagnosis:

- **The task is easy by construction.** One site, three robots, one large
  fault on each of two, no decoy, no confounder, no drift, no change log, a
  healthy robot to compare with on the same tasks and start states, and a
  description whose list of fourteen settings contains the one that is off.
  Three correct answers say that the chain works, not how good the agent is.
- **One model, three runs, one prompt.** No cheaper model, no OpenAI model
  (the Codex CLI's login on this machine was damaged at the time; see the
  agent-runtime spike).
- **Only the structured fields are scored.** What the agents wrote as
  evidence I read but did not check line by line, apart from the lag figures
  and one set of three frames (section 6).
- **The fault table's vocabulary is mine.** Whether "controller" and
  "revert_setting" are the right labels for a delay is a design decision of
  this spike; another vocabulary could score the same answer differently.
- **Recommendations were not executed.** The plan values a recommendation by
  applying it. Here the two right fixes are known by replay to recover 13 and
  11 episodes, but no agent action was fed back into the simulator.

About exactness:

- **One machine state.** All runs within an hour and a half, one driver and library
  version, no reboot in between. Other jobs loaded the CPU at the time
  (load average up to 5.5 when a run started). On the GPU, at most about
  1 GiB belonged to other jobs during any run (whole-machine peak minus these
  processes, less the 3.1 GiB the desktop holds), which is a rendering
  simulation or two and no second model.
- **Row position for X-VLA**: one swap of two rows. Every comparison in
  section 4 kept the row fixed.
- **Hard resets** were not used at all. Whether soft resets change the success
  rate against LeRobot's default is covered only by the healthy batch's rate.
- **Mid-episode snapshots**, absolute control included, and faults that start
  in mid-episode.
- **Other suites, other policies**, and worker counts other than eight.
- **Long runs.** The longest run was 200 episodes. A worker was replaced at
  most once per run.

About the faults:

- **Five levels, 40 episodes each.** The 95% intervals are about 15 points
  wide either way. The chosen levels were not repeated with other seeds; the
  fleet run's 24 and 28 are the same episodes as the dose runs', not a second
  sample.
- **Why the delay costs what it costs** is inferred from step counts.
- **Other fault classes**: sensor degradation, site conditions, configuration
  changes with a change log, inputs outside the training data.

About speed and size:

- **A quiet machine.** Every timing was taken with other jobs running.
- **Four rows and four workers.** From the call times (1.58 s for four rows
  against 3.06 s for eight) the same throughput should be reachable with half
  the worker memory. Calculated, not run.
- **Video settings.** Sizes are for H.264 at quality 23, preset "veryfast",
  360x360. Smaller is certainly possible; I did not try.
- **Why the video encoder does not repeat itself**, and whether one thread or
  another codec would. Also whether the agent's reading of a frame can change
  with the encoding; the differences are small next to a 6-degree turn of the
  camera, but I did not test it.
- **Token refresh inside the sandbox** did not happen during
  these calls.
- **The command-line tool's own isolation.** `claude` was the version the
  agent-runtime spike measured the same day (2.1.292). That spike's wire and
  isolation checks were not repeated here; the sandbox check of section 6 was.

## Files

| File | Purpose |
|---|---|
| `pipeline.sh` | The commands of every step (see the note under "Numbers"). |
| `run.sh` | One fleet run behind the heavy-job lock. |
| `fleet.py` | The runner: inference server, eight workers, the log and the truth records. |
| `sk_common.py` | Environment set-up, the pinned reset, hashes, memory readers. |
| `policy_xvla.py` | X-VLA with the sampler noise passed in. |
| `faults.py` | The two faults and their two hook points. |
| `manifests.py` | Episode lists: dose-response, fleet, replay, healthy batch; the identity function. |
| `analyze.py` | Dose-response tables, exactness comparison, attribution, run summary. |
| `probe_policy.py` | The policy-side checks of section 1. |
| `export.py`, `leak_check.py` | The agent's workspace, and the search for anything that names the truth. |
| `fleet_tools.py` | The tool server: hardened `query`, `get_frames`. |
| `sandbox_check.py` | What the two sandboxes show; a tool session through both, without a model. |
| `diagnose.py` | The system description, the question, the answer schema, the agent call. |
| `score.py` | The hidden fault table and the mechanical score. |
| `agent_runtime.py` | Copied unchanged from `spikes/agent-runtime/`. |

Outputs are in `~/.cache/robots-probes/skeleton/`: `runs/` (each with `log/`,
`truth/` and `run.json`), `manifests/`, `workspace/`, `agent/` (the three
answers with their scores), `agent-state/runs/` (command line, event stream and
tool call log of every agent call), `probe/`, `logs/`, and `venv/` for the
tools. About 670 MB in all, of which 440 MB is the tools' Python environment.

Agent calls made by this spike: four (one tool test, three diagnoses).
