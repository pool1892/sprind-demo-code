# Host feasibility screen: replacement compound jobs

**HOST-AUTHORED.** These programs, batches and results are the host's development
evidence. They are never an input of an engineering agent: the engineering agent of a
later investigation gets the request, the generic skills and its own
evidence, not these programs, counts or mechanisms.

## What it measures

Five candidate two-goal jobs and three single skills, all played by the runtime's own
runner (`python -m fleetlab.jobs.runner`, `fleetlab.job-batch/1`), not the lane S spike's
runner, whose transitions differ. Requested goals and instructions are parsed from the goal
suite by `fleetlab.jobs.runner.GoalSuite` (LIBERO's BDDL files and benchmark language); the
naive arm is `runner.naive_program`. Every batch loads **base task 6's scene** (the goal
suite's tasks share it; the BDDL files differ only in language and goal) and runs from the
host's development starts (`fleetlab.jobs.starts.start_identity(6, k)`, the spike's seeds):
initial states **0 to 9** in the screen, **10 to 29** in a confirmation. States 30 to 49
are never used. No order rules (the final goals are independent); delivery check, tail and
thresholds are the runtime's, unchanged.

| Batch | Requested goals (plan task) | Arms (10 starts each) |
|---|---|---|
| single-drawer | open the middle drawer (7) | single skill |
| single-push | plate to the stove front (8) | single skill |
| single-cabinet | bowl on top of the cabinet (3) | single skill |
| c4 | bowl on cabinet (3), then open middle drawer (7) | naive, home, lift-home |
| c3 | push plate (8), then bowl on cabinet (3) | naive, home, lift-home |
| c1 | cheese in bowl (6), then open middle drawer (7) | naive, finish-home, lift-home |
| c2 | cheese in bowl (6), then push plate (8) | naive, finish-home, lift-home |
| c5 | bowl on plate (4), then open middle drawer (7) | naive, lift-home |

170 executions. Transitions, all from the existing job vocabulary:

- **naive:** switch the moment the first goal holds.
- **home:** then `home` (200 steps, continue on timeout), then the second skill.
- **finish-home:** then the first instruction 30 more steps (`until.steps`), home, the
  second skill.
- **lift-home:** then `open_gripper` (10 steps), `ramp` z +5 cm over 20 steps, home, the
  second skill.

Why these arms (spike evidence, `spikes/jobs/README.md` sections 3 and 4): after a push the
hand stays at the plate (J4) and a return to the start rescued 5 of 10, so c3 gets home and
a lift before home (a straight way back from low at the plate may drag it). A plain return
right after the cheese made J1 worse (0 of 20) and once pulled the cheese (job-two run 3),
so c1 and c2 get finish-home and lift-home. The bowl is still held when bowl-on-cabinet or
bowl-on-plate fires (K7 10/10, J3 20/20) and 30 more steps of the cabinet skill lost the
bowl in K7 (10/10), so c4 gets home and lift-home and c5 lift-home only (release-then-retreat
kept bowl-on-plate on 16 of 20 in J3). The spike's home moved differently, which is why the
runtime's is measured here.

## Files

- `screen.py`: `generate`, `hold`, `confirm`, `validate`, `report`, and `generate-expanded`
  for the expanded screen (below).
- `traces.py`: per-episode mechanisms from the public trace tables (below).
- `batches/screen-01/`, `batches/hold-01/`, `batches/confirm-01/` (root's picks, `b5efc4b`):
  the batches and `manifest.json` (goals,
  instructions, BDDL hashes, program digests). Reproducible: `generate`/`hold` rewrite them
  byte for byte.
- `batches/expanded-screen-01/`: the expanded screen's five batches and manifest
  (`generate-expanded` rewrites them byte for byte).
- `run_stage.sh STAGE [xvla|hold]`: validates, copies the batches beside the results, runs
  each batch as its own runner process in the manifest's order (`--resume` when a run dir
  holds episodes), writes `screen_report.{md,json}`; exits non-zero when a batch fails or
  the report cannot be written (the report failure used to be swallowed; fixed after the
  runs, which all wrote their reports). `RELIABILITY_HEAVY_RUN` replaces `tools/heavy-run`
  only for the test.
- `test_screen.py`: CPU checks (batches validate, 170 executions, report on a synthetic run,
  `run_stage.sh`'s exit status with stand-ins for `uv` and heavy-run).

## Commands (from the repository root of a checkout whose `src/` is main `ac2d73e`)

CPU only, no simulator:

```sh
uv run --locked python spikes/jobs-reliability/screen.py validate spikes/jobs-reliability/batches/screen-01
tools/heavy-run --light -m 2G uv run --locked pytest spikes/jobs-reliability -p no:cacheprovider -q
```

Hold check before the GPU (light, no model; 22 executions on initial state 0, each
candidate also with `host-cN-machinery`, which runs every step kind on step counts so the
motions, the home step and the tail are reached; about 3 to 5 minutes, mostly process
start-up). It checks that every requested goal and region builds on base task 6's scene
and every program is accepted:

```sh
spikes/jobs-reliability/run_stage.sh hold-01 hold
# results: ~/.cache/robots-probes/demo/2026-10-20/reliability/hold-01/
```

Expect `{'done': N, 'refused': 0, 'failed': 0}` per batch; the real programs end
`not delivered` at their first step's timeout, and only the machinery jobs run every step.

**First heavy batch** (X-VLA, one heavy job per batch, 8 batches):

```sh
spikes/jobs-reliability/run_stage.sh screen-01
# results: ~/.cache/robots-probes/demo/2026-10-20/reliability/screen-01/<batch>/
# logs:    ~/.cache/robots-probes/demo/2026-10-20/reliability/screen-01/<batch>.log
```

The script sets `FLEETLAB_MIOPEN_DIR=$HOME/.cache/robots-probes/engine/miopen` (the pinned
cache) and `TMPDIR=/var/tmp/robots-reliability`; heavy-run's defaults apply (24G RAM, 24
GiB GPU). One batch by hand:

```sh
FLEETLAB_MIOPEN_DIR=$HOME/.cache/robots-probes/engine/miopen TMPDIR=/var/tmp/robots-reliability \
  tools/heavy-run uv run --locked python -m fleetlab.jobs.runner \
  --batch spikes/jobs-reliability/batches/screen-01/c4.batch.json \
  --out ~/.cache/robots-probes/demo/2026-10-20/reliability/screen-01/c4
```

**Estimate:** about 18 minutes of the heavy slot (30 single-skill starts at about 2 s,
140 chained starts at about 6 s, 8 policy loads at 10 to 20 s); at most about 30 minutes if
every chained start runs its full budget (naive 660 control steps, lift-home 890). Memory as
the 21 October job batches (X-VLA in 16-bit, one LIBERO scene).

## Confirmation (after the screen; the coordinator picks the arms)

```sh
tools/heavy-run --light -m 4G uv run --locked python spikes/jobs-reliability/screen.py confirm \
  --pick c4:lift-home --pick c3:home            # example picks; add --with-naive for a paired naive arm
uv run --locked python spikes/jobs-reliability/screen.py validate spikes/jobs-reliability/batches/confirm-01
spikes/jobs-reliability/run_stage.sh confirm-01
```

20 starts per pick (initial states 10 to 29), the same program digests as the screen. Two
picks are 40 executions, about 5 minutes; `--with-naive` adds 20 per candidate.

## Report

`run_stage.sh` writes it; by hand: `uv run --locked python spikes/jobs-reliability/screen.py
report <stage results dir>`. Per arm: delivered of N (a 95% Wilson interval when N >= 20),
per goal reached at any time / held at the end, a mechanism per failed run (the first goal
never held, a goal lost after holding, or goals holding but the robot touching, an object
moving or the gripper not open), step outcomes, and per-start rows; paired against naive
on the same starts: rescued and broken with start numbers. Raw run directories stay as
written. In the "Goal reached / held at end" column each goal is shown as reached/held.

## Expanded screen

**HOST-AUTHORED task-selection evidence, withheld from engineering agents.** A second
screen inside the same replacement search, with the first screen's runner, starts, delivery
check, horizons and caps unchanged. Five further pairs of goal-suite tasks (canonical BDDL
goals and benchmark instructions, as `GoalSuite` parses them; the manifest's task table is
the first screen's, byte for byte), each with the same three generic arms: **naive**,
**home** and **finish-home** (definitions above). Hypotheses, not expected wins; a candidate
that already delivers naively is kept in the counts and is unsuitable as an improvement story.

| Batch | Requested goals (plan task) |
|---|---|
| c6 | wine bottle on cabinet (0), then open middle drawer (7) |
| c7 | wine bottle on rack (5), then open middle drawer (7) |
| c8 | wine bottle on cabinet (0), then bowl on plate (4) |
| c9 | wine bottle on rack (5), then bowl on cabinet (3) |
| c10 | stove on (2), then bowl on cabinet (3) |

5 x 3 x 10 = **150 executions** on historical pool 6 legacy development starts 0 to 9
(`start_identity(6, k)`); no singles, no order rule. The home and finish-home programs carry
the expanded label (`EXPANDED_LABEL`) in their notes; the naive arm is the
runtime's canonical naive chain, unlabelled. No expanded digest equals a first-screen digest
(the goal pairs are new). Each batch's `host_screen` and the manifest's label name
the expanded screen, and `report` takes its label from the stage's copied manifest. The goals
and regions are base task 6's scene: the BDDL files of tasks 0, 2 and 5 differ from task 6's
only in language, objects of interest and goal, and the runner's unary-goal index
(`make_env`) and `SceneGoals` already take `Turnon`. `traces.py` knows the wine bottle as the
object those goals move.

`validate` now also rebuilds every job's program from the manifest's goals and instructions
(naive chain, single, machinery, or the screen's transition under its label) and checks the
batch's label and screen id against its screen; the first screen's batches still validate with
their recorded digests, and `generate`, `hold` and `confirm --pick c4:lift-home --pick
c3:home --with-naive` still rewrite `screen-01`, `hold-01` and `confirm-01` byte for byte.
Generation refuses to overwrite a stage's files with different content.

```sh
uv run --locked python spikes/jobs-reliability/screen.py validate spikes/jobs-reliability/batches/expanded-screen-01
spikes/jobs-reliability/run_stage.sh expanded-screen-01
# results: ~/.cache/robots-probes/demo/2026-10-20/reliability/expanded-screen-01/<batch>/
```

Five heavy batches; about 15 to 20 minutes of the heavy slot (150 chained starts at about
6 s, 5 policy loads), at most about 25 minutes if every start runs its full budget (naive
660 control steps, home 860, finish-home 890, as counted above).

**Confirmation** after the screen, at most two picks of the screened home/finish-home arms,
each candidate with its naive chain (at most 80 executions on starts 10 to 29; never 30 to 49):

```sh
tools/heavy-run --light -m 4G uv run --locked python spikes/jobs-reliability/screen.py confirm \
  --pick cX:ARM [--pick cY:ARM] --with-naive          # writes batches/expanded-confirm-01
uv run --locked python spikes/jobs-reliability/screen.py validate spikes/jobs-reliability/batches/expanded-confirm-01
spikes/jobs-reliability/run_stage.sh expanded-confirm-01
```

`confirm` refuses more than two expanded picks, picks without `--with-naive`, an arm the
expanded screen did not run, and a stage mixing both screens; its programs have the expanded
screen's digests. The engineering agent later receives only a selected request, never these
programs, counts or mechanisms.

## Results (internal selection evidence; host feasibility runs, not agent results)

Sources: `~/.cache/robots-probes/demo/2026-10-20/reliability/screen-01/` (170 executions,
17 jobs, 8 batches, all done, none refused or failed) and `confirm-01/` (80 executions, 4
jobs: C3 naive and home, C4 naive and lift-home, starts 10 to 29). Recounted independently
from every `episodes.jsonl` (250 episodes): each count below matches both stages'
`screen_report.md`, every episode's start equals `start_identity(6, k)` with k < 30, every
program digest equals its manifest's, and episode ids do not repeat. Per-episode
mechanisms: `analysis-01/trace_mechanisms.json`, from

```sh
R=~/.cache/robots-probes/demo/2026-10-20/reliability
tools/heavy-run --light -m 4G uv run --locked python spikes/jobs-reliability/traces.py \
  $R/screen-01 $R/confirm-01 --out $R/analysis-01
```

| Job | Arm | Starts 0-9 | Starts 10-29 | Paired vs naive on 10-29 |
|---|---|---|---|---|
| C4 bowl on cabinet, then drawer | naive | 4/10 | 4/20 (8 to 42%) | |
| | home | 2/10 | - | |
| | **lift-home** | **10/10** | **20/20 (84 to 100%)** | 16 rescued, 0 broken |
| C3 push plate, then bowl on cabinet | naive | 6/10 | 10/20 (30 to 70%) | |
| | **home** | **10/10** | **19/20 (76 to 99%)** | 9 rescued, 0 broken |
| | lift-home | 9/10 | - | |
| C1 cheese, then drawer | naive / finish-home / lift-home | 10 / 9 / 9 | - | |
| C2 cheese, then push plate | naive / finish-home / lift-home | 7 / 9 / 9 | - | |
| C5 bowl on plate, then drawer | naive / lift-home | 10 / 10 | - | |
| Single skills | cabinet / drawer / push | 10 / 9 / 8 | - | |

Over starts 0 to 29 the same programs give C4 lift-home 30/30 (89 to 100%) against naive
8/30, and C3 home 29/30 (83 to 99%) against naive 16/30; starts 0 to 9 chose the arms, so
the 20-start confirmation is the cleaner count. C1 and C5 already deliver naively and show
no improvement; C2 gains 2 of 10. Root selected C4 lift-home as the primary replacement and
C3 home as the backup.

What the traces show (90 start groups: on every start the arms are identical up to the
switch, same switch step, hand position and gripper opening, so a paired difference is the
transition's; root's review compared the full robot, object and goal rows up to the
switch on all 90 and found no difference):

A goal "loss" below is a change of LIBERO's predicate, which for `On` needs contact; a
transient loss (the predicate holds again later) is not by itself the object leaving, so
`traces.py` records each loss's span and how far the object moved over it. "Final loss" is
the start of the unbroken stretch without the goal that runs to the end.

- **C4, the bowl is still gripped when its goal fires.** On all 30 starts the cabinet
  skill's last command before the switch is closed, the fingers 4 to 16 mm apart and the
  bowl in contact with the hand.
  - Naive: the bowl-on-cabinet predicate drops on 30 of 30 runs, first 1 to 26 steps after
    the switch, as 1- to 9-step flickers while the drawer skill moves with the bowl still
    in the hand (122 transient losses; 5 moved the bowl more than 2 cm, at most 10 cm). On 8
    runs every loss is regained. On 22 the final loss comes 50 to 58 steps after the switch
    with the hand touching the bowl, and the bowl ends 21 to 26 cm from where it stood:
    pulled off the cabinet top. The drawer never opens on 17 of those 22.
  - Home: the hand opens while already moving. The predicate drops 1 to 4 steps after the
    switch on 10 of 10; 2 regain it. On 8 the final loss is 1 to 8 steps after the switch,
    the hand touching the bowl, which ends 22 to 30 cm away: dragged off.
  - Lift-home (open in place, a vertical 5 cm lift, then home): the predicate drops on 15
    of 30 runs, 1 to 4 steps after the switch, each time for 1 or 2 steps with the bowl
    moving at most 4 mm, and is regained every time: contact flicker as the fingers let go,
    not the bowl leaving. No final loss; the bowl is on the cabinet and the drawer open at
    the end on 30 of 30.
- **C3, the hand is left at the plate, 39 to 45 cm from its start pose.** Naive (29 of 30
  pushes reached): the bowl step timed out on 8 runs, 7 of them lifting the bowl at most
  3 cm. The plate-at-stove-front predicate drops on 9 runs; 1 regains it; on 8 the final
  loss comes 44 to 164 steps after the switch, 6 with the hand touching the plate in the 10
  steps before, the plate moving 0.6 to 9.7 cm from where it was (under 1 cm on 3: the
  plate sits near the region's edge after the push, so a small nudge ends the goal).
  - Home: one transient loss in 30 runs (confirm start, 3 steps after the switch, 65 steps
    long, the plate moving 3 mm), regained; no final loss, and every bowl reached the
    cabinet. Its one failure (start 20) is the push itself timing out on its first step,
    as in the naive arm on that start: a first-skill failure, not the transition's.
  - Lift-home's one failure (start 7): opening the gripper against the plate ended the
    plate goal 6 steps after the switch, for good, with the plate moving 1 mm: it was at
    the region's edge. Home without the lift did not touch it there.
- Residual single-skill misses: drawer start 0 and push start 0 end with the hand touching
  the fixture or plate (delivery check), push start 3 loses the plate in the tail. C4's
  drawer step ended delivered on 30 of 30 under lift-home.

Limits: development pool 6 under the legacy seeds only, one execution per start and arm;
the counts say nothing about pool 2 or another seed profile, where the prepared C4 request
will run. A clean 20/20 bounds the rate at about 84% from below at 95%. These programs are
host-authored; the engineering agent receives none of them, nor these counts or mechanisms.
