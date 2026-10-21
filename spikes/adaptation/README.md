# Adapting to a lasting condition: commands that reach one site late

Spike for lane adapt of
[the loop's design note](../../docs/design/loop.md) (section 7), night of
13 to 14 October 2026. Everything below was simulated on this machine with
X-VLA (`lerobot/xvla-libero`, fp16_res32, four rows per call) in LIBERO and
judged by the fleet's own success check (the object placed). No response curve
is planted anywhere: every number is the outcome of simulated attempts.

## Verdict

**Yes: the adaptation story holds in the physics simulation.** A site whose
robots get every command a fixed number of control steps late, over a remote
link that stays, loses a third to a half of its attempts with the default
settings. One ordinary setting that every robot has, the new
`control.latency_compensation_steps`, set to the delay at that site, brings it
back to within a few points of a healthy site. Nothing removes the condition
itself. The numbers below are from p2, the measurement after the review's startup
check (section 3).

| Under the lasting condition, 60 paired attempts per cell | Default | Compensation = the delay | Gain, points (95% interval) | Half the delay |
|---|---|---|---|---|
| Spatial tasks (bowl onto plate), 17 steps late | 30 of 60 (50%) | 58 of 60 (96.7%) | +46.7 (+32.0 to +59.2) | 56 of 60 (+43.3) |
| Object tasks (item into basket), 17 steps late | 4 of 60 (6.7%) | 59 of 60 (98.3%) | +91.7 (+79.9 to +95.9) | |
| Object tasks, 14 steps late | 38 of 60 (63%) | 58 of 60 (96.7%) | +33.3 (+19.7 to +46.2) | 56 of 60 (+30.0) |
| Spatial tasks, 14 steps late | 54 of 60 (90%) | 56 of 60 (93.3%) | +3.3 (-7.8 to +14.6), not significant | |

For comparison, the same attempts on a healthy site: 38 of 40 spatial, 40 of
40 object. The first and third rows are p2, the second and fourth p3.

This is development evidence with settings chosen by us, not a discovery by an
agent: it shows that a better operating point exists under the condition and
what it is worth, which is what the loop's agent would have to find.

What makes it an engineering problem rather than a lookup:

- **The setting harms a site without the delay, badly.** On a healthy site,
  spatial tasks fell from 38 of 40 to 8 of 40 with compensation 17 (-75
  points), to 18 of 40 with 14 (-50) and to 31 of 40 with 8 (-17.5); object
  tasks to 37 of 40 with 14 (-7.5, not significant). Set fleet-wide, the
  setting that rescues the remote site would wreck the others: the agent has to
  set it at the site.
- **The value has to be found, but the optimum is broad.** On spatial tasks 17
  steps late: none 30 of 60, 8 gives 56, 17 gives 58 and 20 gives 58. On object
  tasks 14 steps late: 7 gives 56, 14 gives 58, 17 gives 55. Compensating the
  lag the telemetry shows (the delay plus the controller's 3 steps) is as good
  as compensating the delay on spatial tasks and costs at most a few points on
  object tasks.
- **The delay is measurable from the visible telemetry,** without any planted
  readout: the lag that best aligns the commanded and the measured end-effector
  positions is 3 control steps on a healthy robot and the delay plus 3 under
  the condition, as the median and both quartiles, with rare one-step
  deviations (section 6).
- **The one replanning-only setting tested failed.** With 13 actions per chunk
  and no compensation, spatial tasks 17 steps late succeeded in 8 of 60, 22
  fewer than the default. That one comparison does not separate the skip from
  the replanning: its first chunk is 13 actions where compensation's is 30, and
  how skipping and replanning interact is not isolated. Every gain here is the
  effect of the whole setting (section 2).

## 1. The condition: `remote_link_latency`

`src/fleetlab/faults/remote_link_latency.py`. The site's robots are driven by
policy servers in the customer's own data centre: every command crosses a
wide-area link and reaches the robot `level` control steps (of 50 ms) after the
policy issued it. Mechanically it is a delay line in the action hook, the same
code as the calibrated `action_latency`, but its own: it is not a
configuration value.

- **No action removes it.** Its truth mapping names the controller component
  of each robot at the site (a component fault is one per robot) with no fixing
  verb. `set` or `revert` of `control.action_delay_steps` acts on a separate
  delay line in series and leaves this one in force; so does every other
  catalogue action, alone or all together
  (`tests/test_adaptation.py::test_every_catalogue_action_leaves_the_condition_in_force`,
  a review's must-fix). A review of 611448c found no leak in removal or
  severity.
- **What an operator knows is public, nothing more.** The site gets a note in
  its public description (`Site.notes`, written by both world generators from
  the schedule through `records.site_notes`): "The robots at this site are
  driven by policy servers in the customer's own data centre, over a wide-area
  network link. This is the customer's requirement and stays." No size, no
  effect, no change-log entry, no health reading. The note is left out of every
  serialized form when empty, so older descriptions keep their bytes.
- **It is the calibrated physics, chain for chain.** The measurement uses the
  calibration's seeds. Spatial tasks 17 steps late repeat the calibration's
  `action_latency` 17-step episodes (60 of 60 outcomes equal, 20 of 20 chains
  compared identical), and the healthy spatial cell repeats the calibration's
  healthy run (40 of 40, 20 of 20). The world generator reads the type's curve
  off `action_latency`'s (`curve_kind`), with the same point labels.
- **The warm-up is a model choice.** Until the line has filled, the robot
  executes the first command it was sent, as `action_latency` always has; a
  real link would deliver nothing for those steps. Since the first chunk is
  never compensated (section 2), the warm-up is the same in every cell.

## 2. The settings

- **`control.latency_compensation_steps`** (new, nominal 0, in the vocabulary
  and in the configuration of every fleet whose spec opts in, section 5): the
  rollout skips that
  many actions at the head of every chunk after an attempt's first, then
  executes the next `policy.actions_executed_per_chunk`, keeping at least the
  chunk's last action. A chunk computed while the robot moves arrives with that
  many stale actions; skipping them makes each executed action land at the time
  its chunk meant it for, as real-time chunking does. The robot waits for its
  first chunk, so nothing in it is stale and it is cut as without the setting.
  The cut lives in `engine.episode.call_policy`; the value is a `PolicySpec`
  field set through the policy input hook like the other policy settings
  (`policy_knob`).
- **The setting's complete effect.** With X-VLA's chunk of 30 actions, a skip
  of c also means the robot replans after at most 30 - c actions instead of 30.
  Every gain here is the gain of the whole setting, skip and replanning
  together, not of "latency compensation" alone (a review's reporting point).
- **`policy.actions_executed_per_chunk`**, the existing knob: 30 is already the
  whole chunk, so it can only make the robot replan more often. The two values
  tested under a delay did worse than the default: 10 actions per chunk at 14
  steps late (the calibration: 27 of 300 against 277 of 300 with 30,
  `docs/design/faults.md` section 8) and 13 at 17 steps late (p3: 8 of 60
  against 30). Values between 13 and 30 were not tested.
- **Compensation 0 changes nothing.** The cut is the old `[:executed]`, the
  field is left out of every serialized `PolicySpec`, and `make smoke-world`
  found 16 of 16 chains identical to the stored reference (at 611448c; again at
  the candidate commit, section 8).

## 3. The measurements

`src/fleetlab/adaptation.py`. A cell is one suite, one delay (0: a healthy
site) and one setting, on paired attempts: attempt i is task i % 10 at start
state i // 10 with the calibration's scene and noise seeds (world w90, robot
r01, day 1, index i), so attempt i is the same start in every cell of every
phase. A setting is applied as the loop applies an adopted action: `resolve`
of `set KEY = v at s1` against the attempt's identity, which adds a
`config_change` and removes nothing. Success has Wilson's 95% interval; each
setting is compared with the default cell of its suite and delay on the same
attempts (Newcombe's paired interval, exact McNemar). The default cells ran
once, in p1: with no setting the cut is the same in every phase, chain for
chain, so p2 and p3 pair with them.

    uv run --locked python -m fleetlab.adaptation analyse p1 p2 p3

### p2: the startup check (320 attempts; the first chunk never compensated)

Run `/var/tmp/robots-loop/adapt/runs/p2`, launched at 42839e6 from this
worktree with a clean tree (`logs/p2-launch.txt`); `run.json` records 42839e6
as dirty because the README and `docs/design/loop.md` were being written during
the run (no source file changed). Replay sample held.

| Cell | Setting | Successes | 95% interval | Against p1's default: rescued / broken | Difference, points (95% interval) | McNemar p | Mean steps |
|---|---|---|---|---|---|---|---|
| spatial, 17 steps late | compensation 8 | 56 of 60 (93.3%) | 84.1 to 97.4 | 27 / 1 | +43.3 (+28.5 to +56.0) | 2e-07 | 186 |
| spatial, 17 steps late | compensation 17 | 58 of 60 (96.7%) | 88.6 to 99.1 | 29 / 1 | +46.7 (+32.0 to +59.2) | 6e-08 | 140 |
| object, 14 steps late | compensation 7 | 56 of 60 (93.3%) | 84.1 to 97.4 | 21 / 3 | +30.0 (+15.1 to +43.5) | 0.0003 | 217 |
| object, 14 steps late | compensation 14 | 58 of 60 (96.7%) | 88.6 to 99.1 | 21 / 1 | +33.3 (+19.7 to +46.2) | 1e-05 | 162 |
| spatial, healthy | compensation 8 | 31 of 40 (77.5%) | 62.5 to 87.7 | 0 / 7 | -17.5 (-31.9 to -4.5) | 0.016 | 139 |
| object, healthy | compensation 14 | 37 of 40 (92.5%) | 80.1 to 97.4 | 0 / 3 | -7.5 (-19.9 to +2.5) | 0.25 | 107 |

**The gain does not come from the startup.** An independent review verified all 320 outcomes
and 32 replays, and that in every compensated attempt the first 30 steps match
its p1 default attempt in the commands and in the joint and end-effector
positions. Without any head start the gain is as large as in p1 or larger. The one cell that moved, full compensation on spatial
tasks, moved up: from 39 of 60 in p1 to 58 of 60. Skipping the first chunk's
head had made the robot reach for a pose 17 steps ahead before it had moved at
all, which spoiled the start of the spatial tasks; p1's finding that full
compensation underperforms on spatial tasks was that artifact.

### p1: the first look (520 attempts; compensation applied to every chunk)

Run `/var/tmp/robots-loop/adapt/runs/p1`, launched at 611448c from this
worktree with only `spikes/adaptation/` untracked; `run.json` records badaf58
(dirty) because the engine reads git at completion and badaf58, which added
only `spikes/adaptation/lag.py`, landed at 20:50:48, after the launch:
mechanism unchanged. Replay sample held. An independent review matched all 520 identities,
verdicts and visible records, the ten cells' counts, rescued and broken pairs
and p-values, and 52 of 52 replays. In this phase the skip also applied to an
attempt's first chunk, so the compensated cells carry the startup artifact; the
default cells do not (no skip) and serve every phase.

| Cell | Setting | Successes | 95% interval | Against the default: rescued / broken | Difference, points (95% interval) | McNemar p | Mean steps |
|---|---|---|---|---|---|---|---|
| spatial, 17 steps late | default | 30 of 60 (50.0%) | 37.7 to 62.3 | | | | 254 |
| spatial, 17 steps late | compensation 8 | 54 of 60 (90.0%) | 79.9 to 95.3 | 26 / 2 | +40.0 (+24.5 to +53.1) | 3e-06 | 184 |
| spatial, 17 steps late | compensation 17 | 39 of 60 (65.0%) | 52.4 to 75.8 | 19 / 10 | +15.0 (-2.6 to +31.4) | 0.14 | 171 |
| object, 14 steps late | default | 38 of 60 (63.3%) | 50.7 to 74.4 | | | | 265 |
| object, 14 steps late | compensation 7 | 56 of 60 (93.3%) | 84.1 to 97.4 | 20 / 2 | +30.0 (+15.6 to +43.3) | 0.0001 | 210 |
| object, 14 steps late | compensation 14 | 60 of 60 (100%) | 94.0 to 100 | 22 / 0 | +36.7 (+24.1 to +49.3) | 5e-07 | 146 |
| spatial, healthy | default | 38 of 40 (95.0%) | 83.5 to 98.6 | | | | 129 |
| spatial, healthy | compensation 17 | 8 of 40 (20.0%) | 10.5 to 34.8 | 0 / 30 | -75.0 (-85.2 to -56.3) | 2e-09 | 233 |
| object, healthy | default | 40 of 40 (100%) | 91.2 to 100 | | | | 144 |
| object, healthy | compensation 14 | 35 of 40 (87.5%) | 73.9 to 94.5 | 0 / 5 | -12.5 (-26.1 to -1.3) | 0.062 | 110 |

### p3: the recipe's questions (500 attempts; the first chunk never compensated)

Run `/var/tmp/robots-loop/adapt/runs/p3`, launched at 8659c9e from this
worktree with a clean tree (`logs/p3-launch.txt`); `run.json` records 8659c9e,
not dirty. Replay sample held. New default cells for the two new conditions;
the others pair with p1's.

| Cell | Setting | Successes | 95% interval | Against the default: rescued / broken | Difference, points (95% interval) | McNemar p | Mean steps |
|---|---|---|---|---|---|---|---|
| spatial, 17 steps late | compensation 20 (the lag) | 58 of 60 (96.7%) | 88.6 to 99.1 | 28 / 0 (p1's default) | +46.7 (+32.6 to +58.9) | 7e-09 | 129 |
| spatial, 17 steps late | 13 actions per chunk, no compensation | 8 of 60 (13.3%) | 6.9 to 24.2 | 1 / 23 (p1's default) | -36.7 (-49.0 to -22.4) | 3e-06 | 277 |
| object, 14 steps late | compensation 17 (the lag) | 55 of 60 (91.7%) | 81.9 to 96.4 | 21 / 4 (p1's default) | +28.3 (+12.9 to +42.3) | 0.0009 | 157 |
| spatial, 14 steps late | default | 54 of 60 (90.0%) | 79.9 to 95.3 | | | | 217 |
| spatial, 14 steps late | compensation 14 | 56 of 60 (93.3%) | 84.1 to 97.4 | 6 / 4 | +3.3 (-7.8 to +14.6) | 0.75 | 140 |
| object, 17 steps late | default | 4 of 60 (6.7%) | 2.6 to 15.9 | | | | 279 |
| object, 17 steps late | compensation 17 | 59 of 60 (98.3%) | 91.1 to 99.7 | 55 / 0 | +91.7 (+79.9 to +95.9) | 6e-17 | 165 |
| spatial, healthy | compensation 17 | 8 of 40 (20.0%) | 10.5 to 34.8 | 0 / 30 (p1's default) | -75.0 (-85.2 to -56.3) | 2e-09 | 239 |
| spatial, healthy | compensation 14 | 18 of 40 (45.0%) | 30.7 to 60.2 | 0 / 20 (p1's default) | -50.0 (-64.4 to -31.8) | 2e-06 | 192 |

## 4. What a fleet recipe would use

The lead fleet on `loop/fleet` (02cbee6, `worlds/loop_fleets.py`) already
plants this condition: s1's four robots 17 steps late from day 0, s1 running
spatial tasks. The measurements support that choice:

- **Site and delay.** One site of four robots, every command 17 control steps
  (0.85 s) late from day 0 with no end, the site's work spatial tasks. A
  site that also runs object tasks loses even more there (6.7% at 17 steps),
  which makes the condition louder but no harder to find. At 14 steps spatial
  tasks lose too little to matter (90% against 95%).
- **What the default costs.** Spatial tasks at the site succeed about half the
  time (30 of 60) against 95% (38 of 40) on a healthy site: about 45 points
  at the site, or 15 points of fleet success when it is one site of three.
- **What the agent would have to find.** `set
  control.latency_compensation_steps = 17 at s1` (58 of 60 on the same
  attempts). Anything from about half the delay to the delay plus 3 works (8:
  56, 17: 58, 20: 58), so the telemetry lag can be used as it is.
- **What it must not do.** Set it at the fleet or at a healthy site: there
  compensation 17 takes spatial tasks from 95% to 20%. The one setting of
  the existing knob tested at the site, 13 actions per chunk without
  compensation, gave 13% (8 of 60).
- **Where the evidence is.** The site's note in the description; the site
  under-performing from day 0 with nothing in the change log to explain it;
  the command-tracking lag of its robots (20 steps against 3, section 6).
- **Mind the cut.** `loop/fleet` merged badaf58, where the skip also applied to
  an attempt's first chunk; under that cut compensation 17 on spatial tasks
  gives only 39 of 60. Merge 42839e6 or later before any compensated run.

## 5. Old fleets, and which fleets have the setting

A fleet has the setting only if its spec says so:
`WorldSpec.later_settings = [control.latency_compensation_steps]` (left out of
the serialized spec when empty). The generators then record the nominal value
0 at fleet scope (`generate.nominal_config`), the description's vocabulary
lists the key and the catalogue offers `set` and `revert` of it; a site or a
robot may override the fleet default but never hold the key alone (`Fleet`
refuses that). Every fleet composed before the key existed, the development
worlds w41 to w63, the mini worlds and the sealed recipes included, has none of
this: its spec, model, description, catalogue and hash are what they were, and
every sealed world among them still recomposes to its stored hash.

Two development fleets fall between: w70 and w71, the loop's lead and later
fleets, were composed while the key was briefly part of every fleet's nominal
configuration (611448c to 66befcc). Their stored models carry the fleet default
0 and their specs have no `later_settings`. They load from those saved models
(`LoopWorld.fleet_on` starts from the stored `WorldTruth.fleet`), not by being
regenerated under the new spec semantics, and as loaded they have the setting:
it is in their public catalogue and vocabulary, an arm or a ledger entry sets
it at a remote site, and the engine's effective policy then runs that value
with the lasting condition still in force
(`tests/test_loop_w70_compensation.py`, which a review confirmed on the stored w70
at days 5, 11 and 14). The current code would not recompose them bit for bit
(the key's description text changed at 42839e6, and a new composition carries
`later_settings`); nothing reads them that way, and they stay as stored.
Newly composed loop fleets opt in (`loop_fleets.compose`). The stored worlds
are frozen and are not changed; no bridge was built because none is needed.

## 6. Measuring the condition from the telemetry

The visible `steps` table logs the commanded action before the action hook
(its first three numbers are the commanded end-effector position, in the world
frame under absolute control) and the measured end-effector position after
each step. The lag that best aligns the two is the controller's own tracking
lag plus any delay on the way:

| Run | Robots' commands | Episodes | Lag, median (quartiles) | Episodes off the median |
|---|---|---|---|---|
| calibration `healthy` | on time | 244 | 3 steps (3 to 3) | not counted |
| calibration `action_latency` | 8 steps late | 100 | 11 (11 to 11) | not counted |
| calibration `action_latency` | 14 steps late | 100 | 17 (17 to 17) | not counted |
| calibration `action_latency` | 17 steps late | 100 | 20 (20 to 20) | not counted |
| calibration `action_latency` | 20 steps late | 100 | 23 (23 to 23) | not counted |
| p1 to p3, every delay-17 cell, any setting | 17 steps late | 522 | 20 (20 to 20) | 1 at 21 |
| p1 to p3, every delay-14 cell, any setting | 14 steps late | 479 | 17 (17 to 17) | 1 at 18 |
| p1 to p3, healthy cells, any setting | on time | 201 | 3 (3 to 3) | 5 at 2, 2 at 4 |

(Episodes shorter than 100 steps are left out. The last column counts, for the
p1 to p3 cells, the episodes whose best lag is not the median: one step off
in every case.) The setting does not hide the
condition: the compensated robot's commands still arrive late, so the lag
stays. `spikes/adaptation/lag.py` computes it per episode;
`spikes/adaptation/lag.sql` is the same estimate as one DuckDB query over the
visible `steps` and `episodes` tables, per robot, as the agents' query tool
runs it (1.8 s on the calibration runs, whose lags it reproduces). An agent
that compares the remote site's robots with the others reads the delay off as
the difference of the two lags, with no planted readout and no hidden table.
The loop's stretch goal (design note, section 10) could start from this as a
composed measurement.

## 7. What is shown and what is not

Shown: under a lasting command delay that no action removes, setting a generic
knob to the delay at the affected site wins back 33 to 92 points on paired
attempts, to within a few points of a healthy site, on both task suites, where
the default loses much (at 14 steps spatial tasks lose too little to gain
from it); the gain survives the startup check; the one replanning-only setting
tested (13 actions per chunk, no compensation) failed; the setting harms a site
without the delay, by up to 75 points; the delay is measurable from visible
telemetry.

Not shown, or limited:

- One policy (X-VLA), two LIBERO suites, one robot's seeds, 40 to 60 attempts
  per cell: the intervals are wide, and a cell of 60 cannot separate 93 from
  97%.
- The warm-up of the delay line is a model choice (section 1).
- A gain is the complete effect of the setting, skip and replanning together
  (section 2).
- Whether an agent finds the setting, and at what cost compared with a plain
  search over its values, is the loop's question, not this spike's.

## 8. The plain search, the comparator for adaptation

`src/fleetlab/loop/plain_search.py`. Told the affected site and the setting
(and the page says so), it runs one experiment there from the fleet's current
state through the agent's own experiment path, chooses on that experiment's
visible outcomes only, adopts its choice as a ledger entry and stops; the
driver evaluates it on the reserved panel. Its design was fixed before any of
its results, as an engineer's default for an integer setting below the
policy's 30-action chunk, not from the measurements above: the values 0, 5,
10, 15, 20 and 25; 24 paired attempts per value with the current value as the
control arm (144 attempts at a site whose setting is still 0); the most
successes wins, a tie going to the value nearest the current one; adoption
only if the winner beats the current value by a one-sided exact sign test on
the paired attempts at 5% divided by the values tried. Its cost receipt is
the agent's format, with its sessions as scripted ones and no model. Lane
state's toy world checks the mechanics on the CPU; it has not run on a
simulated fleet. From the measurements, 10, 15 and 20 all sit in the broad
optimum for spatial tasks 17 steps late (8 to 20 recovered 56 to 58 of 60),
so the search would likely find a good value at that site; what it costs
against the agent is the comparison the loop makes.

## 9. Provenance and checks

- The measurements' provenance is in section 3: p1 launched at 611448c, p2 at
  42839e6 and p3 at 8659c9e, from this worktree, through `tools/heavy-run`,
  each with a start and end line in the lanes' queue log; every replay sample
  held, and an independent review re-derived every count, pair and p-value from the raw
  records.
- `make smoke-world`: 16 of 16 chains identical to the stored reference at
  611448c and again at 72088f5, the last change to the engine.
- `make check`: ruff, ruff format and mypy clean at 72088f5 and at 7596966.
  The full test suite was stopped at 34% on 72088f5, with no failure, at the
  coordinator's request, because the full suite runs once on the frozen
  integrated candidate. Focused suites, all passed: the adaptation, plain
  search, stored-w70/w71 probe, loop fleets, loop costs, build rule,
  baselines, classical, contracts and all world tests (including the six world
  files whose sealed worlds must recompose).

## Files

| File | What it is |
|---|---|
| `src/fleetlab/faults/remote_link_latency.py` | the lasting condition |
| `src/fleetlab/engine/episode.py` (`call_policy`, `ChunkRequest.chunk_index`) | the cut with the compensation |
| `src/fleetlab/contracts/identity.py`, `vocabulary.py` | the `PolicySpec` field and the configuration key |
| `src/fleetlab/contracts/model.py`, `_base.py` | `Site.notes`; `visible_dump` honours `exclude_if` |
| `src/fleetlab/faults/records.py` | `site_notes`; `measured` reads a `curve_kind` |
| `src/fleetlab/adaptation.py` | the measurement: phases, plan, run, paired tables |
| `tests/test_adaptation.py` | the focused tests |
| `spikes/adaptation/lag.py`, `lag.sql` | the telemetry lag estimate |
| `/var/tmp/robots-loop/adapt/runs/<phase>` | run directories, with `<phase>.jsonl` manifests |
| `/var/tmp/robots-loop/adapt/logs/` | run logs and launch receipts (`<phase>-launch.txt`) |
