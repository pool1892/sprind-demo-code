# The rollout engine

Runs manifests of jobs (`contracts.rollout.Job`) on the LIBERO adapter with
X-VLA, in the worker layout of the X-VLA speed spike, with the fault hooks in
their fixed order, the supervisor hook, snapshots, exact replay proven by hash
chains, and the episode log written through `fleetlab.log`. Specification:
[docs/design/contracts.md](../../../docs/design/contracts.md), sections 4, 5,
6, 10 and 11.

## Verdict

**The engine works and meets the done criteria, with one borderline
number.** It runs manifests of jobs with X-VLA in the worker layout, keeps
the seven replay rules, proves exact replay in every run, restores a snapshot
bit for bit inside the layout, hands over to the scripted teleoperator
through the same loop, and is within 15% of the stage-1 throughput. The
borderline number: on `libero_object` in 16 bits the engine scored 200 of
200, which lies above the reference's 95% interval (95.7 to 99.5%) by half a
point; 200 against the reference's 197 is no difference a test can show
(Fisher's exact test, p = 0.25), and in 32 bits, the reference's own
precision, the engine scored 198 of 200, inside the interval.

| Criterion | Measured | Met |
|---|---|---|
| Success inside the reference's 95% interval, 200 episodes per suite | `libero_spatial` 194/200 (fp16_res32), 195/200 (fp32), reference 192/200, interval 92.3 to 98.0%; `libero_object` 200/200 (fp16_res32), 198/200 (fp32), reference 197/200, interval 95.7 to 99.5% | spatial yes; object yes in fp32, 0.5 points above in fp16_res32 |
| 200 faulted-mix episodes bit-identical across two runs with other worker counts and video on and off | 200 of 200 (10 workers with video against 6 without); both runs' chain samples held | yes |
| A mid-episode restore inside the fleet layout bit-identical in 20 of 20 | 20 of 20 | yes |
| The scripted teleoperator or a toy in-worker policy through the same loop | both: 8 of 8 hand-overs from X-VLA continued by the scripted teleoperator, which placed the object in 8 of 8; 8 of 8 with the toy policy | yes |
| Throughput within 15% of 3,093 faulted episodes per hour in 16 bits | 2,789 per hour end to end (−10%), 3,006 in steady state | yes |

## The done criteria, measured

Run directories are under `~/.cache/robots-probes/engine/runs/`, logs under
`~/.cache/robots-probes/engine/logs/`. The runs were made from a frozen
checkout of this branch (`../robots-engine-run`): `smoke` and the model tests
at 5f7a99c, `ref_spatial`, `ref_object`, `mix_a`, `mix_b`, `handover` and the
restore's first half at 8ee1535, the restore's second half, `mix_placed`,
the fp32 references, `mix_real` and `handover_teleop` at c8654e0. Later runs
record their commit in `run.json` (`extra.code`). All runs: X-VLA, 4 rows, the pinned
database `7751323820effb28`, the stand-in camera and delay faults unless
said otherwise, on 2026-10-07 between 18:28 and 21:00 with other agents'
jobs on the machine (load average 5 to 10; timings are good to 10 to 20%).

### Exactness

| What | Result | Run |
|---|---|---|
| Faulted mix (stand-in camera 6 degrees, delay 17 steps), run A: 10 workers, video, staggered lives of 16 | 143/200 successes; chain sample 20 of 20 identical | `mix_a` |
| The same manifest, run B: 6 workers, no video | 143/200; chain sample held; **200 of 200 episodes identical to run A** (state, both images, executed action at every step) | `mix_b`, `python -m fleetlab.engine compare mix_a mix_b` |
| The same episodes with `fleetlab.faults`' types (`camera_pose_offset` 6, `action_latency` 17) in place of the stand-ins | 143/200; **200 of 200 identical to run A** | `mix_real` |
| Restore: 20 faulted-mix episodes with a snapshot each at steps 25 to 44 (before, at and after a chunk boundary), continued in a second run with 8 workers | **20 of 20 continuations identical** to the uninterrupted episodes, 43 to 255 steps each | `restore`, `restore/restore_report.json` |
| Every other run's chain sample (every tenth run job, at least 20, replayed after all other work) | held in all of them (`chain_ok` true) | all |
| `make check-model` | 4 of 4: rows check in fp32 and fp16_res32, the same chunk bits in another process (and other bits for other noise: the noise handed in is used), the worker layout against one process on 10 faulted jobs | `logs/stage1.log` |
| Rows check, batch size 4, 16 recorded observations | 52 of 52 comparisons identical in each precision (again 4/4, other position 16/16, other mates 16/16, padded with row 0 16/16) | `rows_fp32.json`, `rows_fp16_res32.json` |

### Rerun after the code review (2026-10-07, 22:40 to 23:00, commit a75009d)

The review's fixes changed what a snapshot holds (the lean set), the fixture
hash (now over every model array a cell hook can change, stored in the chain
file), the verdict reasons and the proof of supervised jobs. Rerun under
`tools/heavy-run` (log `logs/stage4.log`), with `fleetlab.faults`' types and
the fleet's check:

| What | Result | Run |
|---|---|---|
| Faulted mix (camera_pose_offset 6, action_latency 17), 10 workers, video | 125/200 placed; chain sample 20 of 20; **2,913 per hour end to end** (−6% against 3,093), 3,326 steady, rows in use 92% | `r2_mix_a` |
| The same, 6 workers, no video | 125/200; chain sample held; **200 of 200 identical to `r2_mix_a`**, fixture hashes included | `r2_mix_b` |
| Restore with the lean snapshot, 20 snapshots at steps 25 to 44, continued in a second run | **20 of 20 identical** | `r2_restore` |
| 8 supervised episodes handed to the scripted teleoperator, chain sample on | 8 scripted sample replays, **8 of 8 identical** (`chain_ok` true); the teleoperator placed the object in 8 of 8, at the same steps as in `handover_teleop` | `r2_handover` |
| `make check-model` (at e37434b, after fixing two tests that still used the old `act` signature and no day length) | 4 of 4 | |

### Success against the reference evaluation

`fleetlab.reference_eval` (lerobot-eval, fp32, LIBERO's check) measured
192/200 on `libero_spatial` and 197/200 on `libero_object` in stage 1. The
engine ran the same sample size, 10 tasks by start states 0 to 19, ending on
LIBERO's check (`--check libero`), with its own scene and noise seeds:

| Suite | fp16_res32 | fp32 | Reference | Reference's 95% interval |
|---|---|---|---|---|
| `libero_spatial` | 194/200, 97.0% (`ref_spatial`) | 195/200, 97.5% (`ref_spatial_fp32`) | 192/200 | 92.3 to 98.0% |
| `libero_object` | 200/200, 100% (`ref_object`) | 198/200, 99.0% (`ref_object_fp32`) | 197/200 | 95.7 to 99.5% |

### Throughput and memory

| Run | Episodes | Per hour, end to end | Steady state | Rows in use | Call, s | Peak memory of the run |
|---|---|---|---|---|---|---|
| `mix_a`, faulted mix, 10 workers, video, LIBERO's check | 200 | **2,789** | 3,006 | 87% | 0.57 | 19.6 GiB (PSS), GPU 14.8 GiB machine-wide |
| `mix_real`, the same with the real fault types | 200 | 2,766 | 2,998 | 82% | 0.55 | |
| `mix_placed`, faulted mix with the fleet's check (placed) | 200 | 2,723 | 2,926 | 84% | 0.54 | |
| `ref_spatial`, healthy, fp16_res32, no video | 200 | 4,165 | 4,929 | 91% | 0.59 | 18.7 GiB |
| `ref_object`, healthy, fp16_res32 | 200 | 3,901 | 4,419 | 96% | | |
| `ref_spatial_fp32` / `ref_object_fp32` | 200 / 200 | 1,830 / 1,587 | 1,962 / 1,705 | 98% / 99% | 1.68 | |

xvla-speed's comparable run (`m200v_fp16res32_r4w10`) made 3,093 per hour
end to end and 3,378 in steady state at a load of 2.8 to 6.1, with 98% of the
rows in use; here the load was 7 to 8 during `mix_a` and a simulation step
took 16.5 ms against 9.6 ms on a quieter machine (`smoke`). The server was
busy 90% of `mix_a`. What a worker spends per step outside the simulator,
the policy wait and the reset (the hooks, the hashes, the logged row, the
success check): 0.8 ms beside a 9.6 ms simulation step in `smoke` (6
workers), 0.8 ms beside 16.4 ms in `ref_spatial` and 1.3 ms beside 16.5 ms
in `mix_a` (10 workers, load 5 to 8; the faulted episodes' observation and
action hooks and video frames add to it). Workers waited for the server
68 to 73% of their time, so this overhead does not set the throughput. The gap in rows in use is likely the spike's
0.25 s wait for a full batch, which this server does not have; measuring it
is left as an idea.

### Hand-over

`handover` and `handover_teleop`: 8 healthy `libero_spatial` episodes, the
timeout supervisor decides at step 41 to hand over, the robot holds still for
its 500 ms (10 steps), the engine saves a snapshot and queues a "continue"
job. With the toy in-worker policy (`scripted.LiftPolicy`) the 8
continuations ran from step 51 to the step limit; with the scripted
teleoperator (`policies.teleoperator.Teleoperator`, a budget of 700 steps)
it placed the object in **8 of 8** by the fleet's check, ending between steps
109 and 377. In the simulator tests a take-over by the teleoperator replays
bit for bit (`tests/test_sim_engine.py`).

A continuation's `Job.max_steps`, including the value supplied through
`--continuation-steps`, is its own budget counted from the saved snapshot step.
For example, a snapshot at step 170 with a 700-step continuation budget permits
700 further active steps, up to step 870. Earlier engine versions treated 700
as an absolute endpoint and cut that continuation short. The scripted expert
and human teleoperator both use the corrected meaning. The human model's queue
and reaction holds advance physical time but do not spend its active budget.
The nightly live-pass counts (`handed_over`, `teleoperator_placed`) may move once,
at the first nightly after this correction is merged; the longer continuation
budget is a semantic correction, not a changed smoke-chain reference.

### The fleet's check against LIBERO's, on the same faulted episodes

`mix_placed` against `mix_a`: the same 200 episodes, ended by placed
instead of LIBERO's check. Up to LIBERO's first fire both runs are the same
episodes (200 of 200 with the same first-fire step); after it, X-VLA kept
acting:

| Episodes | LIBERO's check | Placed | Steps from LIBERO's fire to placed, median / max |
|---|---|---|---|
| healthy, 40 | 40 | 40 | 12.5 / 25 |
| camera 6 degrees, 80 | 43 | 42 (1 "LIBERO only") | 12.5 / 35 |
| delay 17 steps, 80 | 60 | 43 (17 reached the step limit after LIBERO's check fired late) | 30 / 57 |

X-VLA releases and backs off by itself in every healthy episode; under the
delay fault, LIBERO's check fires so late that the step limit of 280 cuts the
60-step tail in 17 of 60 cases.

## How to run

```sh
# once per machine, heavy: search and pin the convolution algorithms (replay rule 7)
tools/heavy-run uv run --locked python -m fleetlab.engine.miopen pin --batch-sizes 1 4 8 \
    --precisions fp32 fp16_res32

# a manifest (JSON lines of contracts.rollout.Job); the engine's own kinds:
uv run --locked python -m fleetlab.engine.manifests mix200 M.jsonl     # or reference, handover

# run it, heavy (loads the policy; ten simulation processes)
tools/heavy-run uv run --locked python -m fleetlab.engine run M.jsonl RUN_DIR \
    [--workers 10] [--batch-size 4] [--precision fp16_res32] [--video] \
    [--check fleet|libero] [--faults fleetlab|standin] [--snapshot-at STEP] [--no-chain-sample] \
    [--teleoperator fleetlab.policies.teleoperator:Teleoperator --continuation-steps 700]

# compare two runs' chains job by job; replay a run's snapshots and compare
uv run --locked python -m fleetlab.engine compare RUN_A RUN_B --manifest M.jsonl
tools/heavy-run uv run --locked python -m fleetlab.engine restore RUN_DIR M.jsonl
```

From Python: `fleetlab.engine.fleet.run(RunConfig(out=..., ...), jobs)`
returns a `RunResult` with the run record; `run_in_process` runs the same jobs
through the same loop in one process. A run started again on the same
directory resumes: `fleetlab.log.RunWriter` discards unfinished jobs and the
engine runs only the jobs whose chain does not exist.
`--continuation-steps N` sets the number of additional active steps after the
snapshot for each spawned continue job, rather than an absolute episode end
step. See the continuation budget and nightly note under Hand-over above.

The pinned database lives under `FLEETLAB_MIOPEN_DIR` (default
`<checkout>/data/miopen`); the measurements here used
`~/.cache/robots-probes/engine/miopen`, so that every checkout of this
machine reads the same database.

## The experiment queue

`experiments.run_queue(queue_path, world, world_dir, config=...)` answers the
runtime's `ExperimentQueue` with an `EngineReply`; `experiments.QueueEngine`
is the runtime's `Engine` for one world. The world comes as a `WorldTruth`
(`<world dir>/world_truth.json`, truth side): the fleet model as the
experiment window starts, the windows, the fault schedule (with per-day
levels for drift), the as-run day length, the tasks, the base runs of the
first export and the first free episode number. The shape was agreed with the
miniature fleet's world generator.

Per experiment and arm: a slot per episode, robots in turn, then the
experiment window's days; a fresh episode index per robot and day after the
as-run ones and a fresh episode id, both kept per analysis in
`<world dir>/experiments/<analysis>/allocations.json` across its sessions
(analyses are alternative histories of one world and each export holds only
its own experiments, so every analysis restarts after the as-run episodes and
episode ids are unique within an analysis); seeds from
`seeds_for`; the task and start state drawn from (world seed, experiment id,
slot), the same for both arms with `paired_starts`; `experiment_id` set; the
faults in force for that robot and day with the arm's actions resolved
against them (`resolve_all`) as the identity's fault list. Both arms are
always simulated; a request the world cannot run is answered "refused" with
the reason. The plan is written before the run
(`experiments/<analysis>/session-NN.plan.json`), so answering a queue again
resumes it; the run is `experiments/<analysis>/session-NN/`, the new export
`exports/<analysis>/session-NN/` over the base runs and the analysis' own
experiment runs (never another analysis'). Arm episodes take the indices
after the day's as-run episodes and are written with the world's own day
length: the contracts' start time puts them from 16:00 every three minutes,
160 per robot and day, after the day's regular work; a request
whose arm episodes would start after 23:59:59, counting the analysis' earlier
experiments on that robot and day, is refused before minting with the public
code `exceeds_robot_day_capacity`, whose fixed message states the limit.

Measured on 2026-10-07 at 23:11 (`python -m fleetlab.engine.probe`, commit
76a0a9f, `~/.cache/robots-probes/engine/queue/probe1`): a three-robot world
on `libero_spatial`, 24 observed episodes, then the runtime's
`run_analysis(check_arms=True)` with a scripted agent requesting one
experiment on r02, the control arm against `recalibrate scene_camera`, 10
episodes each with paired starts. Sessions "waiting" then "ended"; the reply
passed the runtime's arm check; the control arm (camera turned 6 degrees)
placed 1 of 10, the recalibrated arm 10 of 10; the experiment run's chain
sample, all 20 arm episodes, held. Tests on a toy world: minting, pairing,
refusals, resumption, a second session's fresh indices, the export's counts,
and `run_analysis` reading the reply (`tests/test_engine_experiments.py`).

## The ladder of worlds

`ladder.py`. **`make smoke-world`**: the smoke world `w30` (one robot, two
days of four `libero_spatial` episodes, the scene camera turned 3 degrees on
day 1), a heavy job; every chain is compared bit for bit with
`smoke_reference.json` and the exit status is non-zero when one differs.
The run directory is `data/smoke-world/w30-<commit>`, so a killed run
resumes and a new commit runs afresh. `--update-reference` writes the reference again,
which is a decision to make on purpose: the reference is this machine's, with
the pinned MIOpen database under `~/.cache/robots-probes/engine/miopen`
(`MIOPEN_DIR` in the Makefile).

**`make nightly`**: one heavy job under `~/.cache/robots-probes/nightly/<date>/`,
the runs named after world and date (`deterministic/w20-<date>`,
`deterministic/w20-<date>-live`, the same for w21, `canary-<date>`), because
the dashboard names a page after its run directory.
Two worlds of one site, four robots, three days (observed, experiment,
scoring) at 40 episodes per robot and day, faults at calibrated levels:
`w20` with the fixed seed 20 (camera 6 degrees on r02, latency 17 steps on
r03, gripper wear 1.0 on r04 from day 1, r01 healthy) and `w21` with the date
as seed (the same three types on three robots drawn, the camera's and the
delay's level drawn from the calibrated points, wear at 1.0, the only one
calibrated, onsets on day 0 or 1). Both come from the world generator
(`fleetlab.worlds`), whose world files go to `<world>/files/`
(the world hash is compared night to night with the chains). Per world:
480 episodes as run with video, the oracle's power set of each scoring
episode's faults (`contracts.oracle`: 2^n - 1 replays for n faults), then a
live pass over the 160 scoring episodes with the timeout supervisor, which
hands an attempt still running at step 160 to the scripted teleoperator.
Then the canary replays w20's first ten episodes, hours after they ran.
`scores.json` holds success by window and robot, each fault's effect and the
best net gain at k = 1, 2, 3 fixes from the oracle, the live pass and each
run's memory peaks and worker processes; `comparison.md` compares w20's chains
and outcomes with the previous night's and sets w20 beside w21 (for
`make dashboard DASHBOARD_FLAGS="--nightly FILE"`). When w20's world files change
but its chains and outcomes do not (a new generator version that leaves the
episodes alone), the comparison says so, names the files that changed and
whether the generator's sources changed, and that night becomes the reference
for the world hash. `--per-day N` shrinks
both worlds for a dry run in another `--root`. Every run's `run.json` keeps a
memory series every 30 s (`extra.memory_series`: seconds, GPU memory, memory
available, server and workers' PSS, workers, largest worker), the soak test's
record. Workers are fitted to the memory available; the layout
changes no bit.

The smoke world is built in `ladder.py` from contracts types, apart from the
generator: it tests the engine, and a new generator version must not move its
reference.

Measured on 2026-10-08. **Smoke world** (commit 15e6382): 32 s from start to
exit (8 episodes, 8 of 8 placed, 8 chain-sample replays), the reference
written; a second run reproduced all 16 chains. A dry nightly at 2 episodes a
day: 4 minutes. **The first nightly** (15e6382,
`~/.cache/robots-probes/nightly/2026-10-08`), 00:15 to 00:46, 31 minutes:

| | w20 | w21 (seed 20261008) |
|---|---|---|
| as run, observed / experiment / scoring day (of 160) | 130, 105, 103 | 141, 105, 94 |
| by robot (of 120) | r01 115, r02 65, r03 66, r04 92 | r01 69, r02 59, r03 117, r04 95 |
| fault effect, net, scoring day | camera 6 deg +20, latency 17 +17, wear +13 | camera 6 deg +20 (21 rescued, 1 broken), latency 20 +31, wear +13 |
| best net gain at k = 1, 2, 3 fixes | 20, 37, 50 | 31, 51, 64 |
| live pass: handed over at step 160 / placed by the teleoperator / placed alone | 79 / 70 / 81 | 81 / 65 / 79 |
| world run: jobs with the chain sample, minutes, per hour | 648, 12.0, 3,217 | 648, 11.6, 3,359 |
| live run: jobs, minutes | 259, 3.4 | 261, 3.8 |

w21 drew camera 6 deg on r02 from day 0, latency 20 on r01 from day 1, wear
on r04 from day 1. Every chain sample held (48 per world run, 20 per live
run), no incident, 49 and 51 worker processes in the world runs; the canary
replayed w20's first 10 episodes half an hour later, 10 of 10 identical.

**The identity check, ahead of the second night** (3a00893, `--root
~/.cache/robots-probes/nightly-check --date 2026-10-09` over a copy of the
first night's scores, 00:54 to 01:25): w20's chains 648 of 648 identical in
the world run and 259 of 259 in the live pass, outcomes, fault effects and
world hash the same, canary 10 of 10.

**Memory.** The server's PSS is flat within a run (slope about 0
GiB per hour). In the first night it was 2.3 GiB in the first two runs and
4.2 and 4.6 GiB in the next two: the process held an earlier run's model
while loading the next. `fleet.run` now releases it (`release_memory`), and
the second night stayed at 2.3 to 2.5 GiB in all five runs. Workers' PSS at
the end of a life 1.3 to 1.5 GiB, the same in the first and last quarter of
the lives; memory available never below 17 GiB; GPU memory at most 13 GiB.
The longest single run is 12 minutes and a night 31 minutes: the six hours of
the soak test are not run yet.

## The soak test

`make soak-test` (`python -m fleetlab.engine.ladder soak-run --root DIR
--hours 6`): randomized nightly worlds from the generator one after another,
world k with the seed `base * 100 + k` (base: the date), each with its 480
episodes as run, the oracle's replays and the live pass, and after each the
canary: world 0's first ten episodes replayed and compared with their chains,
so that the last canary compares episodes hours apart. A world starts while
the worlds done ran less than the hours. `soak.json` keeps the worlds and the
sessions; a rerun resumes, and the record counts a soak as unattended only if
one session ran all its worlds. The record (`soak-summary.json`,
`soak-summary.md`): per run its hour, jobs, minutes, throughput, the server's
PSS peak, the workers' PSS at the end of their lives, GPU memory, memory
available, incidents (watchdog and dead workers) and the chain sample; per
world the canary; over the hours the slopes of server PSS, worker PSS and
throughput. Since **2026-10-09**, so that a timeout is retried a couple
of times before it counts as failure, a job may be lost twice and
retried; its third loss aborts the run (`MAX_JOB_FAILURES = 3`). Passed: the
hours reached in one session, no unrecovered incident, at most one incident
per running hour on average (`SOAK_MAX_INCIDENTS_PER_HOUR = 1.0`), no broken
chain sample and every canary identical. Before that date, the second loss
of a job aborted the run and any incident failed the soak, even after recovery.

Each incident keeps its worker, cause and lost job keys, and adds the assignment
loss number of each job, whether all those jobs subsequently completed,
and whether each repeat was checked by replay (and its verdict). Incidents
from earlier sessions are retained, with a session number on every incident.
A repeated run job joins the chain sample, including a supervised run's scripted replay;
continuations and other jobs the sample does not cover are recorded unchecked.
Explicitly disabling the chain sample also leaves repeats unchecked. The soak
record counts total, recovered and unrecovered incidents, states the rate
using unrounded running time, and shows each cause even when the soak passes.
It names every failed pass condition. When rounding would hide a violation,
the text shows a bound (`<6` hours or `>1.00` incidents per hour); JSON keeps
the exact running hours and rate.
For older records without recovery flags, the reader infers recovery only
from a completed job entry and its stored chain; it does not rewrite the files.

The `attempts` field counts lost assignments, not proven failed executions
(`attempts_kind: lost_assignments`). `assignment_status` distinguishes an
undelivered job, unconfirmed execution, a received policy request proving
execution started, and a completion report that arrived after the loss.
The parent still repeats unconfirmed work. Incidents also retain the worker
life, exit status, whether the parent requested a kill, and the worker's own
error report even when a failed send noticed the death first, if the live main
loop receives that report. Shutdown abandons unread queue messages after killing
workers: even a nonblocking queue read can hang inside a cut-off image frame.

Every worker life is watched from startup. An isolated loss before a first job
is replaced, or its slot retired if nothing is pending. Three consecutive such
losses in the same slot with pending work abort with an explicit no-progress
error (`MAX_STARTUP_LOSSES_PER_SLOT = 3`); a delivered job resets that count.
An incident ending the run is never recovered (`ended_run`), and an empty-job
incident is recovered only when the session completes. A normal exit waits for
its queued goodbye and does not count as a loss, including an exit observed
during the loss handler's grace period or final join. An empty queue still missing
that goodbye a watchdog interval after a status-0 exit yields an explicit
protocol error, so an abnormal clean exit cannot wait indefinitely.
Loss handling allows `worker_exit_grace_s` (default
two seconds) for a reporting worker to flush its queues and exit before
killing one that remains. A retired slot cannot take a new assignment, and
the run refuses to finish with a handed-out job neither completed nor pending.
These are mitigations: the shared worker-to-parent queues can still be blocked
by a worker dying while its feeder holds their writer lock, or a cut-off
request frame blocking the live main loop's read. Channels isolated
per worker remain open work.

Computed from the stored `soak.json` and `run.json` files, the second run
started on **2026-10-08** (`2026-10-08b`) passes the new rule: 24 worlds in
one session, 21,828.347365626032 seconds of running (6.06 hours displayed),
one incident and one recovered incident, 0.16492315884935307 incidents per
hour, all 48 runs with a chain sample held (the 24 canary runs have no
chain sample) and all 240 canaries identical; its one repeated job,
`e000290.continue.cd3be50c948b3715`, was a continuation and
was not checked by replay.
Both historical soaks' stored records remain unchanged; the first still
fails for insufficient running time and its broken chain sample.

**First attempt, 2026-10-08, 01:47 to 05:16** (3ada3f7,
`~/.cache/robots-probes/soak/2026-10-08`, record in `soak-summary.md` there):
**not passed.** It stopped after 3.47 hours, in world 13. 13 worlds finished
(seeds 2026100800 to 2026100812), 11,884 jobs, 1,094 worker lives, one
session.

- *Stopped by the memory guard* at 05:16, a minute into world 13: memory
  available 5.9 GiB, under the floor of 6 GiB (`mem_floor_gib` 8 less 2).
  The fleet's own memory did not grow (below), so the memory went to other
  jobs on the machine: other agents' light simulator tests ran beside the
  soak all night, and the lowest memory available in a finished run fell
  from about 20 GiB in the first hour to 7.2 GiB.
- *One incident, a GPU fault*: at 05:04 (hour 3.3, world 12) the kernel
  reset the graphics ring after a timeout ("ring gfx_0.0.0 timeout ... ring
  reset succeeded") and Mesa (radeonsi) aborted the context of the worker
  rendering in it (signal 6, core dump in libgallium). The watchdog
  recorded the death and ran the episode again; the run finished with its
  chain sample held.
- *One broken chain sample*: world 7's live pass (hour 2.0), episode
  e000319, its scripted replay differed in one wrist image, at step 91 of
  148. State, scene image and executed action were identical at every step,
  and so were the wrist images before and after it. One frame rendered
  differently, in the live run or in the replay; which one cannot be told
  without a third rendering. 38 of the 39 chain samples held.
- *Canary*: 130 of 130 identical, the last one 3.4 hours after world 0 ran.
- *Memory*: the server's PSS peaked at 2.2 to 2.5 GiB per run, with no
  trend (slope -0.24 GiB per hour over the soak). Five of the 39 runs
  peaked at 4.4 to 4.6 GiB, which then fell back in the next run. Workers'
  PSS at the end of a life was 1.41 GiB in the first world run and 1.41 GiB
  in the last (slope +0.007 GiB per hour). GPU memory peaked at 15.6 GiB.
- *Throughput* of the world runs fell from 3,566 to 3,163 episodes an hour
  (slope -118 an hour). The lowest, 2,725 in world 6, ran while other jobs
  loaded the machine.

## The soak's GPU faults

**The differing frame was the replay's.** World 7's live pass, episode
e000319 (r01, healthy), step 91, wrist camera: the live run rendered
`9f63e66860e9e8e3` and the chain sample's replay `2908e0a3e8fa4331`. The
frame was rendered 15 more times on 2026-10-08, and every rendering gave the
live run's hash, at every one of the 149 steps and in both cameras:

- 10 times with the model (5 in-process runs of the live job, each with its
  sample replay), in a heavy job;
- 5 times without the model, stepping the environment with the chain's
  executed actions (`fleetlab.engine.rerender`, a light job), the state hash
  equal to the chain's at every step.

So the run's data were right, and the replay rendered one frame differently
once. That frame was not consumed:
- the policy was called at steps 0, 30, 60, 90 and 120 (step 91 is chunk 3,
  step 1);
- the timeout supervisor reads no pixels;
- live jobs record no video, and video takes every 4th step only.

The odd hash equals no other frame of either chain, so it was not a stale copy
of a neighbouring frame.

What was ruled out, and what was not:
- **Memory pressure.** GPU memory was 12.2 to 13.0 GiB around both renderings
  (run peak 14.9) and memory available 17 to 19 GiB.
- **The ring timeout.** No kernel message near 03:51 or 03:54; the timeout
  came 70 minutes later.
- **A fresh context.** Neither rendering ran in a new process or environment:
  the live run was worker 7's 5th job of its first life, the replay worker
  0's 13th job of its third.
- **A shader cache miss** would change compile time, not output; not examined
  further.
- **Not ruled out:** a transient error in the GPU's rendering or read-back
  under the night's load (10 EGL contexts beside ROCm inference). The replay
  ran among the chain sample's replays at the end of the run, when the
  server's answers took 2.4 s instead of 1.

**The ring timeout** (05:04:16, world 12, episode e000450):
- The kernel's gfx ring timed out on a submission of worker pid 3009548
  (thread `python:cs0`) and reset the ring.
- In the core dump, the worker's main thread is in MuJoCo's `mjr_readPixels`,
  waiting in libgallium for its frame. The abort comes from Mesa's
  command-submission thread, which found its context lost
  ("guilty of a hard recovery").
- No OOM.
- Nothing like it in the kernel log since 2026-09-17 (three boots).
- Kernel 7.2.5 and Mesa 26.2.2 have not changed since September.
- libgallium has no debug symbols on Arch's debuginfod, so its frames stay
  unnamed.
- The engine handled it as designed: the watchdog reran the episode, and the
  run's chain sample held.

Both faults are in the rendering path, under hours of concurrent rendering
and inference: 1 frame in about 300,000 compared frames, and 1 ring timeout
in 3.5 hours.

**The rule, adopted** (docs/design/contracts.md section 4; implemented on
`stage2/transients`): `fleet.prove` classifies every sample comparison with
`chain.classify` as identical, a render transient or broken. A candidate
transient is settled at once by `settle_transient`: it runs `rerender` (two
passes) in a process of its own through `tools/heavy-run --light`, and the
comparison stays a transient only if the re-rendering names the odd
rendering. The frames something used come from the run's steps table: a
policy call's step (`chunk_step` 0, so every step for an in-worker policy)
and a video frame's step (`frame_index` set). A supervisor that may read
pixels uses every frame (`supervisors.reads_pixels`: noop and timeout do
not). `chain_ok` holds when nothing is broken and at most one comparison is a
transient. `run.json` keeps each comparison's verdict, frames, compared
frames and odd rendering in `extra.comparisons`, and the summary in
`extra.chain_checks` (counts, compared frames, transients, and what broke the
run if two transients did). `verdicts` and `job_outcomes` do not carry it yet:
that needs a column in the contracts. The dashboard shows a
transient in amber on the run page and the rate per million compared frames
on the progress page.

**The proposal as it was made:** Today any difference makes the sample broken,
`chain_ok` false, and the run gives the oracle no rows. A narrower verdict,
"render transient", would hold when all of the following are true:
1. the state hash and the executed action are equal at every step;
2. the image hashes differ in at most one frame of the episode;
3. no policy call, supervisor or video consumed that frame (for an in-worker
   policy, every frame counts as consumed);
4. `rerender` shows which rendering was the odd one.

If the as-run frame was the odd one and it went into a video, that episode's
video is marked. A run whose comparisons are all identical or render
transients keeps `chain_ok`; more than one transient in a run, or any in a
consumed frame, still makes it broken.

This is sound for what the oracle and the scores use: outcomes follow from
state and actions, which the sample shows to be reproduced exactly. It gives
up one thing: detecting image nondeterminism on its own. So a glitched frame
that is consumed, about one in 30,000 episodes at the rate seen here, changes
that episode's actions without the sample seeing it unless that episode is
sampled. The oracle keeps its own guard, since two rows of one fault list
that disagree are refused.

How to show it:
- **Run page:** the hash-chain line in amber ("held; 1 render transient:
  e000319 step 91, wrist, the replay's frame, not consumed").
- **Progress page:** render transients per million compared frames over the
  runs, so that a rising rate, the sign of a GPU going bad, shows.
- **Oracle:** uses such runs as it does any other.

## Where things are

| module | what |
|---|---|
| `episode.py` | one job: the reset with its cell hooks, the per-step loop with the hooks in their fixed order, the chain, snapshots, supervisor decisions, hold, recovery, hand-over |
| `server.py` | the scheduling rule of the inference server (`Batcher`) |
| `fleet.py` | a run: workers, work on request, staggered lives, the server loop, memory guards, the chain sample, `run.json` |
| `records.py` | what a job produced (`EpisodeRecord`) and its writing through `fleetlab.log` (`LogSink`) |
| `chain.py` | the hash chain and its comparison |
| `miopen.py` | the pinned convolution database and the refusal of an unpinned setting |
| `software.py` | the `SoftwareVersion` of a run |
| `manifests.py` | identities and the engine's own manifests |
| `supervisors.py` | the no-op and timeout supervisors (the real one is stage 5) |
| `scripted.py` | model-free policies for LIBERO: an in-worker one (lift) and a server one (sway), for tests |
| `standin_faults.py` | stage 1's camera and delay faults, which the measurements here used; bit-identical to `fleetlab.faults`' `camera_pose_offset` and `action_latency` |
| `toy.py` | a toy cell, toy policies and toy faults for the CPU tests |
| `experiments.py` | the experiment queue: `WorldTruth`, `run_queue`, `QueueEngine`, a scripted analysis |
| `probe.py` | the probe world and one queue through the runtime on LIBERO |
| `ladder.py` | the smoke world and the nightly worlds, their scores and the comparison of nights |
| `rerender.py` | an episode rendered again from its chain's executed actions, without the model: which rendering was the odd one |
| `../sim/libero_sim.py` | the LIBERO adapter as `SimEnv`: reset sequence, snapshot and restore, privileged state |
| `../sim/libero_success.py` | the fleet's success check, placed |
| `../policies/xvla.py` | X-VLA with our own sampling loop, fp32, fp16, fp16_res32 |
| `../policies/rows_check.py` | the rows check that `row_independent` rests on |
| `../policies/teleoperator.py` | the scripted teleoperator as an in-worker policy |
| `../policies/teleop/` | the scripted expert itself, moved from spikes/teleop-absolute as measured |

Tests: `tests/test_engine_loop.py` and `tests/test_engine_fleet.py` (`make
check`, toy cell, CPU), `tests/test_sim_engine.py` (`make check-sim`, the real
adapter without a model), `tests/test_model_engine.py` (`make check-model`).

## Choices where the contracts are silent, and workarounds

The loop (`episode.run_job`), per job:

1. every active fault gets a fresh state (`new_episode`), then the policy
   input hooks give the effective `PolicySpec`; the environment is reset with
   it, so the control mode reaches the arm controller (the faults agent's
   `control_mode_change` keeps the nominal spec and works in the hooks);
2. the environment runs its reset sequence: undo what is left of the last
   episode, the cell hooks through the `CellHandle`, seed and reset, start
   state, settle, control mode, first observation;
3. a job from a snapshot restores it after the reset: environment (lean set
   plus the cheap groups, then `mj_forward`), queued actions, chunk number,
   step, every fault's state, the success check's state.

Per step: a supervisor decision whose hold has run out is applied; a
requested snapshot is taken (before any hook of the step); the observation
hook, then the visible row of the step (the log listens after it); the end
test; the supervisor, if the risk estimator fired (the robot holds still at
least one step while it decides, `contracts.live.hold_action`); the commanded
action (hold, recovery primitive, queue, or a policy call that refills the
queue), logged as commanded; the action hook; the simulator; the chain row
(state and images as rendered, the executed action); the check's verdict
through the data hooks (`check_data_hook` on every one). The episode ends on
a done verdict, or at the step limit ("step limit").

Choices:

- **Noise** is drawn in the worker from `noise_for(identity, chunk_index)`
  and handed to the server with the request; the server never draws.
- **Policy calls** group requests by what the sampling loop reads (`call_key`:
  name, revision, precision, sampling steps, image views); control mode and
  executed actions per chunk are applied per row afterwards. The group's
  effective spec goes with the call (`Policy.act(..., spec)`). Rows are free for a row-independent policy; a request with a
  fixed row goes into it. Padding repeats row 0.
- **Conversion** to `Action7` is per chunk (30 rows at a time), so that a
  row's actions never depend on the batch's other rows through the CPU's
  vector code.
- **Precision** is a per-run setting; a fault that changes it is refused
  (the model is loaded once).
- **Episode end and verdicts**: the reasons are the contracts': "placed",
  "LIBERO only", "step limit", "LIBERO's check" (a run on LIBERO's own check,
  the reference mode), "stopped" and "handed over" (the supervisor ended it);
  a handed-over episode's visible outcome is `handed_over`. The environment
  resets the check with the task's goal. The runs of 2026-10-07 before 21:30
  (all in the tables above) were made before these reasons existed: their
  LIBERO-check successes record "placed" and their handed-over episodes "step
  limit" with outcome failure.
- **CPU threads**: the inference process runs with `RunConfig.server_threads`
  (4) and every worker with one thread; neither is in the identity. The
  policy computes on the GPU; its CPU work is the observation's preprocessing
  and the conversion of each chunk, done one chunk at a time with elementwise
  operations and reductions over 3 to 6 numbers, which threads do not split.
  So the thread count should not change a bit; it was 4 in every run here and
  another value was not tried (on the CPU it does change a policy's actions,
  spikes/bimanual).
- **Hold and recovery steps** are real control steps: they go through the
  action hook and into the chain; their visible rows have no chunk index.
  Stage 5 excludes decision holds from the active-attempt and success-check
  timeout while still allowing physical success during a hold. The check's
  clock offset survives snapshots. Retry now opens, lifts 8 cm, returns via
  a high waypoint to the retained pre-grasp command, and hands back to the
  policy. Recover releases and retreats vertically; degrade executes at most
  four actions per fresh policy chunk. Details and evidence are in the
  [supervisor README](../live/vlm/README.md).
- **Snapshots** are requested by a run job's `snapshot_steps` and stored as
  pickles at `RunLayout.snapshot_file`. A hand-over snapshot is taken at the
  top of the step after the hold, before the step's observation hook, so the
  continuation applies the hook once. The adapter's snapshot is the lean set
  (fixture bodies only, never the controllers' ranges), and a replay from a
  snapshot that removes or adds a fault with a cell hook is refused (version
  1). A restore ends with `mj_forward`, so an in-worker policy reads the
  restored scene; its continuation then replays itself, but is not what the
  uninterrupted episode would have done at that step, which saw MuJoCo's
  positions one substep old. The restore check is therefore made with the
  server policy, whose input after a restore is the saved observation.
- **Continue jobs** run the in-worker policy named by `RunConfig.teleoperator`
  (`policies.teleoperator.Teleoperator`; `scripted.LiftPolicy` in the tests),
  built for that job alone, which learns its episode through
  `begin_episode(identity, step)` and ends the continuation when `done`. A
  continuation is written next to its snapshot (`RunLayout.continuation_file`)
  the moment it is made, and a resumed run finds and finishes it. They write
  truth only. Their `max_steps` counts additional active steps from the snapshot,
  as does `--continuation-steps`; human queue/reaction holds are elapsed physical
  time outside that active budget.
- **Events**, version 1: grasp and release when the commanded gripper changes
  sign (0, "hold the opening", is neither), and supervisor_call, handover,
  retry from the decisions.
- **The chain sample** is handed out after every other job, so the replays
  run in other processes than their originals as far as the layout allows. A
  supervised run job's sample replay is a "scripted" job from its recorded
  decisions, made when the run job ends and kept under `truth/samples/`; a
  scripted replay ends at its hand-over without spawning a continuation.
  Runs made without a sample (`--no-chain-sample`): `restore` (both halves),
  `handover`, `handover_teleop`; every other run held its sample.
- **Lost workers**: a worker that dies, raises, or is silent for 15 minutes
  (`RunConfig.watchdog_s`) is killed, its unfinished jobs (including one whose
  files it had not yet written) run again on a new worker, and a job lost
  three times stops the run (`MAX_JOB_FAILURES`); `run.json` lists the
  incidents, failed attempt numbers, recovery and replay checks.
  A failed pipe send also goes through worker-loss recovery, whether it was
  answering a job request or an inference request. Unexpected dispatcher or
  monitor errors stop the run with a message instead of leaving workers waiting.
- **Throughput** counts from the run's start (policy load, workers starting,
  the tail), as xvla-speed's "end to end"; "steady" is the rate between the 10%
  and 90% marks of completions. The chain sample's replays are excluded.
- **The pinned database** is searched for the vision tower's number format
  (fp32, or fp16 for both 16-bit settings) and the batch size; its manifest
  records the file hashes, and a run refuses a database whose files changed.
  It is checked per images per call (batch size times the real views, at
  most the two cameras), with every job's effective image views after the
  policy input hooks, so a configuration fault that changes the views needs
  its own pinned entry.
- **The visible start times** need the world's episodes per robot and day
  (`--episodes-per-robot-day`); the engine refuses to guess it unless told to
  take the manifest's (`--allow-manifest-day-length`, the engine's own checks).
- **Software version**: hash of `uv.lock`, hash of `third_party/*.patch`,
  MuJoCo's version, the pinned database's hash; a job made for other software
  is refused.

Contract points raised while the engine was built are in contracts
batch 6 and used: the hashes on `SimEnv` (the fixture hash over every model
array a cell hook can change), the verdict on `StepResult`, the effective
spec in `Policy.act`, `begin_episode`, `Job.snapshot_steps`, the reasons for
LIBERO's own check and the supervisor's endings, the goal for the check, the
fixture hash in the chain file, the continuation files.

## What I did not verify

- **Twelve of the fourteen fault types** in the engine: only
  `camera_pose_offset` and `action_latency` ran here (bit-identical to the
  stand-ins). The faults agent ran its types through the loop on LIBERO in its
  own tests and calibrates through this engine.
- **Stage-5 supervisor measurements** are tracked separately in the
  [supervisor README](../live/vlm/README.md). The original stage-2 evidence
  above covered only no-op/timeout calls; stage 5 adds the subscription model
  route and LIBERO tests of retry, recover, degrade and their scripted chains.
- **The teleoperator's human model** and its take-over from
  X-VLA's own failures at scale: 8 hand-overs from healthy episodes here.
- **The image-views knob** (3 views) against the spike's variant of the same
  name: it changes the output (CPU test), equality of bits not checked.
- **The rows check at the spike's scale**: 16 observations at batch size 4,
  against the spike's 80 at several sizes.
- **The convolution search on a quiet machine**: the database was pinned at a
  load of 10 to 14. Bits are fixed by the pin whatever the load; whether
  another search would have chosen faster algorithms is not known (the call
  time, 0.54 to 0.57 s, matches the spike's).
- **Runs longer than ten minutes**, exactness across reboots or
  driver updates, and two engine runs at once (the log's writer locks a run
  directory; the GPU guard is per run).
- **The memory guard**: it refused two 10-worker runs at 32.7 GiB available
  (estimate 25 GiB plus an 8 GiB floor) that would have fitted (measured peak
  19.6 GiB); `--mem-floor` lowers it.
- **A row-dependent policy** (SmolVLA): fixed rows are implemented and tested
  in the server's scheduling only.
- **The fleet's check on `libero_object`** under faults: placed ran on the
  `libero_spatial` mix and the smoke run.
