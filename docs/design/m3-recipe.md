# M3: the demonstration's recipe and run selection (proposal)

*Lane fleet, 14 October 2026. A proposal: three new
fleets never used in development, five investigations, every outcome shown. Nothing here
is composed or run. Section 7 holds M2's lessons; the choices that depend on the w71
reuse pair stay open.*

*Amended: see the amendment below.*

## Amendment, 14 October: the configuration actually run

M3's configuration was frozen on 14 October 2026, before any of its fleets was created,
and M3 runs with the configuration below. Where a later section of this note says
otherwise, this amendment holds; those sections are kept as proposed.

**Runtime.** Every M3 step runs the fixed, checked runtime at commit `67f5a70`: its full
check exited 0, and so did the retry of its smoke-world replay. It runs from a frozen,
detached checkout of that commit, which is never edited.

**Workers and sessions.** Every heavy step runs at 4 workers, not the 8 of sections 5 and
9: during the w71 reuse pair, an 8-worker job was refused by the engine's memory check.
Each round allows the agent 8 sessions, not the 4 the staged launcher passes, as in M2
and in both arms of the w71 reuse pair. The rest is unchanged: 480 attempts per round,
576 reserved cases per state, probes off and learning on.

**Later recipe.** The later fleet's recipe is unchanged from section 2. Its two optional
swaps, release-blur and arm-0.4, are declined for this campaign and are no longer pending.

**No wait for w71.** The earlier rule that M3 waits for the result of the w71 reuse pair
is removed. M3's reuse pair mounts M3's own source library, chosen by the rule of section
4, and the w71 pair's scoring and report run beside M3.

**Order: the first result first.** The priority is the first complete result of lead run
A, which starts as soon as the lead fleet's round-1 operation ends: lead run B and the
variant's investigation are not started alongside A just for throughput, although the
variant's and the later fleet's round-1 operations may fill the allowed windows while an
engagement's model is reasoning. Those two operations go through guarded commands that
submit them only while the w71 pair has no heavy job running or queued; the M3
engagements queue their own heavy jobs, which simply alternate with the w71 pair's in
`tools/heavy-run`'s one-at-a-time queue, and the plain searches and the references start
only after the w71 pair has finished (14 October). The source rule of section 4 is
applied as soon as lead run A has a valid freeze, and the library it picks is never
changed for its content or results.

**Overrides of the staged defaults**, stated explicitly:

- The Python steps run from a run-specific configured copy of `spikes/m3/m3_stage.py`
  whose only change is `WORKERS = 8` to `WORKERS = 4` (sha256 of the original begins
  9aa6d417, of the configured copy 852708b7).
- Explicit, reviewed command lines replace the staged launcher `spikes/m3/m3_launch.sh`.
- The record of this configuration lives in the run's own `FREEZE.md`, outside the
  repository.

**Differences from M2's agent.** The task text M3's agent receives each round carries the
sentence added to the task text: before requesting experiments, write down provisional
competing explanations, each with a rough confidence, and revise them when the results
arrive. Its tool bundle differs too: the bundle's digest (`tool_env`) is fb4c9d04, against
3422ebca for M2 and the w71 reuse pair, because the bundle now holds the probe contracts
and tables (probes stay off) and the tool server changed. Both are accepted changes on
main, not choices made for M3. The system prompt, the base skills and the agent settings
are identical. Any comparison between M2 and M3 must say this.

**Draw.** The three seeds were drawn once, at 07:30:58 on 14 October (machine time,
UTC-7), and the fleets were composed from them at 07:31. The hidden and public ids are:
the lead fleet w72 (public w03), the variant w73 (public w04) and the later fleet w74
(public w05). The seeds and the fleets' truth stay outside this note.

## 1. Fixed before any demonstration fleet exists

*Amended: M3 runs at 4 workers; see the amendment at the top.*

- **One starting configuration for all five investigations** (section 8). It is the
  latest fully checked candidate with the accepted prompt sentence and the panel
  video support: `5032f73` once its full check and smoke pass, otherwise the next checked
  candidate. Probes are off by default. Its model, effort, tier, tools and tool
  environment, its commit and its prompt and skill digests are recorded before any fleet
  is created.
- **The recipe, as code:** three new recipes in `fleetlab.worlds.loop_fleets` (`M3-LEAD`,
  `M3-VARIANT`, `M3-LATER`), each with a seed drawn once at creation
  (`recipe.new_world_seed()`), new hidden and public ids, and the placement rules of
  section 2.
- **The rest of this note:** the runs, their order and the worker count, 8 (section 5);
  the source rule (section 4); and the panel, states and video rule (section 3).

All of it is recorded, with the commit, the agent's digests and each fleet's
world hash, before the first run. Every run is shown, including failed and inconclusive
ones.

## 2. The fleets

All three keep the shape of w70 and w71:

- 12 robots at 3 sites, 12 attempts per robot and day, built through `loop_fleets` by the
  version-1 generator;
- the loop layout: round 1's operation on days 0-5 (864 attempts), its experiments on
  days 6-7, round 2's operation on days 8-11 (576 attempts), its experiments on days
  12-13, and the reserved evaluation on days 14-17;
- every fault starting by day 3, none between rounds;
- the compensation setting at its nominal 0 everywhere.

**Placement drawn from the seed.** w70 and w71 put each part at a named site and robot by
hand. The demonstration recipes keep every part: its kind, calibration point, kind of
scope, onset and the site's suite pinning. Each fleet's seed draws which site and which
robots carry each part, by a seeded permutation of the three sites and of the four robots
within each site. This is bookkeeping only; no level, mechanism or pattern changes. M1
ran on the miniature fleet w10, and M2 runs on w70. Without the draw, the M3 lead fleet
would carry w70's answer key.

**The later fleet's condition site** is drawn from the two sites other than the M3 lead
fleet's condition site, so the reuse comparison's "another site" holds by construction.
An independent permutation would put it back at the lead's site one time in three.

- The rule takes the two other site ids in order and picks index
  `sha256("m3-later-condition-site:" + str(later_seed))[0] % 2`.
- The later fleet's other two parts then take the two remaining sites by its own seeded
  permutation.
- Both seeds are drawn at creation, before any outcome exists.

**Lead fleet**, from w70, which worked as designed in round 1. Its parts are unchanged:

- the lasting remote link, 17 steps, at a site with spatial tasks only;
- the coincidence: a release carrying a 4 px scene blur and an inert scene calibration
  record, at one site on day 1;
- the decoy: a release with a 2 px smoothing on day 2, the day another site's tables
  move 2 cm (unlogged);
- the masked second layers: worn pads behind the link (day 2), a logged 14-step command
  buffer behind the blur release (day 3), a scene camera knocked 3 degrees behind the
  tables (day 3).

On w70 these parts gave a 66% start and the designed site and day patterns. The command
lag was readable in the visible records, and the second layers stayed hidden in round 1.

**Modest variant**, new, investigated once. It is the lead fleet without the masked
second layers: the same lasting condition, the coincidence, and the decoy with the
unlogged tables. That leaves three repairs and the adaptation, one more than a round may
make, so round 2 has little to do and stopping is a fair outcome to show. The start
should be near the lead's: w70's round-1 site rates (58%, 59% and 82%) came almost
entirely from these parts.

**Later fleet**, from w71. These parts stay:

- the condition at another site, 14 steps, object tasks only;
- the same site note at a second site, over an 8-step link that is flat on its spatial
  tasks;
- a site worse from day 0 (heavier objects, 6 mass doublings) beside a harmless
  sampling-steps change.

Two candidate changes, both to existing calibration points:

- **The logged release at the note site.** In w71 it carried a 24-step scene-camera lag.
  Under the short link that cost about 5 points (97% before day 2, 92% after), against
  54% when the lag was calibrated alone. The candidate is a 4 px scene blur instead: 78%
  on 378 calibration attempts, and independent of timing. It shares the mechanism of the
  lead's release, but at another site, version and day; the reuse comparison is about the
  lasting condition, not releases.
- **The weak arm.** A torque loss of 0.3 cost nothing visible in w71 (90% after day 2,
  siblings 75-83%). The candidate is 0.4, calibrated at 56% on 100 attempts.

With both, the expected start is about 75%, from w71's measured site rates with these two
parts at their calibration points. Both stay open until the w71 reuse pair is in (section
7). Neither follows automatically from any single outcome.

*Amended: both swaps are declined; see the amendment at the top.*

## 3. Panel, states and video

- **Panel.** 12 attempts per robot and day on days 14-17: 576 per state, 192 of them at
  the condition's site. The identities are minted from the seed and never exported. Every
  state of a fleet is simulated on the same identities.
- **States per engagement, as built.** Original, round 1, round 2 if the agent did not
  stop, and one-look (round 1's ranked list cut at the final number of changes).
  - Every engagement evaluates its own original state; no cache is shared across
    engagements.
  - Every named state is simulated, also when two have the same ledger: `evaluate`
    builds no deduplication, and none is added.
- **References, on the same panel, as built:**
  - **The fixed statistical procedure and the build rule:** once per fleet, beside the
    fleet's first engagement (lead run A, the variant's run, the later fleet's run
    without the library), cut at that engagement's number of changes
    (`fleetlab.loop.references`, then `evaluate --state`).
  - **The plain search:** once per fleet, told the site and the setting: 6 arms of 24
    attempts in a one-round engagement of its own. That engagement evaluates its own
    original, adopted and one-look states.
- **Video.** In every state, only the panel attempts with index 0 of each robot and day
  record both cameras: 48 per state, 8%.
  - The same identities across states give before-and-after pairs.
  - The clip shown is chosen only among those recorded attempts: the first in panel
    order at the condition's site that failed under original and succeeded under the
    final state; and the first such attempt elsewhere in the fleet.
  - If no recorded attempt was rescued, the report says so and shows no clip. An attempt
    without video is never selected, and nothing is replayed to film it.
  - Built and on main (`0ce3b6e`, landed via `5a1c76b`): `EngineSetup.panel_video_indices`,
    set by `--panel-video-index 0` on the driver, the reuse comparison and the plain
    search, and the selector `fleetlab.loop.clips`.

## 4. The source engagement

M3's own source rule, fixed before any outcome:

- **Lead run A**, the first started, is the source, whatever its outcome.
- **Lead run B** is the source only if run A froze no skill (it wrote none, or the
  freeze refused it).
- If neither froze one, the reuse pair is not run, and that is the outcome reported.

The library is never chosen by its content or by any evaluation. Lane learn's runbook
([reuse-run.md](reuse-run.md) on `loop/learn` and `loop/integ`, section 1) describes M2's
single source; M3 has two lead runs, hence this rule.

## 5. Runs, order and compute

*Amended: M3 runs at 4 workers and 8 sessions per round, in the order set out in the
amendment at the top of this note.*

**Workers.** Every M3 heavy step runs at **8 workers**, under `tools/heavy-run`'s memory
guard: one heavy job at a time, killed above 24 GiB of RAM or above 24 of the GPU's
31 GiB. That covers each operation, each round's experiments and each evaluation. The
driver's default is 4 and `loop_fleets run` defaults to 10, so the setting is passed
explicitly.

**Rates, with explicit denominators.** The estimate is in primary attempts (the planned
attempts, chain-sample replays not counted) per hour of a job's total elapsed time; the
elapsed time already holds that job's replays and start-up. Each record's own rate field
is shown beside it, since the fields use other denominators.

| Workers | Job | Primary attempts | Total elapsed | Per hour of elapsed | The record's field |
|---|---|---|---|---|---|
| 8 | one real job | 360 | 638.6 s | 2,029 | 2,223 (the manifest's completion rate) |
| 10 | w70's round-1 operation | 864 (+ 87 replays) | 1,263 s | 2,463 | 2,696 |
| 10 | w71's round-1 operation | 864 (+ 87 replays) | 1,199 s | 2,595 | 2,868 |
| 4 | M2, larger job | 360 (+ 36 replays) | 1,144 s | 1,133 | 1,255 |
| 4 | M2, smaller job | 120 (+ 20 replays) | 501 s | 862 | 987 |

M3 is sized on the one 8-worker job: one job, not a guarantee. Replays ran at 10% in the
operations and 10% and 17% in M2's queues, so a job with more replays runs slower per
primary attempt. The other rows are scenarios for comparison, not bounds. The hours are
updated from M3's first real jobs; there is no benchmark campaign.

The attempts below are at most what is planned, with each round's experiments at the
full budget of 480.

| # | Step | Planned attempts (at most) |
|---|---|---|
| 1 | Freeze record; compose the three fleets (CPU); hashes recorded | 0 |
| 2 | Lead fleet: round 1's operation | 864 |
| 3 | Lead run A, with learning: 2 x 480 experiments, round 2's operation 576, four states 2,304 | 3,840 |
| 4 | Lead run B, with learning, independent | 3,840 |
| 5 | Freeze the source library by section 4 | 0 |
| 6 | Variant: round 1's operation, then its run | 864 + 3,840 |
| 7 | Later fleet: round 1's operation, then the reuse pair (without, then with) | 864 + 2 x 3,840 |
| 8 | Per fleet: statistical procedure 576 and build rule 576 (after the fleet's first engagement decides), plain search 144 + its adopted state 576 (any time) | 3 x 1,872 |
| | **Subtotal** | **27,408** |
| 8b | Per fleet, as built: the plain search's own original and one-look states | 3 x 1,152 |
| 9 | Score the reuse pair (reads the later fleet's truth), then the report | 0 |
| | **Planned primary attempts** | **30,864** |

| Rate (primary attempts per hour of elapsed) | Hours of simulation for 30,864 |
|---|---|
| 2,029, the one 8-worker job (2,223 by its record's field) | 15.2 (13.9) |
| 2,463-2,595, the 10-worker operations | 11.9-12.5 |
| 862-1,133, the 4-worker M2 jobs | 27.2-35.8 |

- **Savings.** A run that stops after round 1 saves up to 1,632 planned attempts: round
  2's operation, round 2's experiments and the round-2 state.
- **Additions.** Evaluating the two references beside every engagement, instead of once
  per fleet, adds 1,152 per further engagement (lead run B and the run with the library:
  2,304).
- **Model time.** About 3.3 to 7.5 hours for the five agent engagements: 40 to 90 minutes
  each, lane learn's estimate.
- **What can wait on no agent, and what must.**
  - The plain searches and the variant's and later fleet's round-1 operations depend on no
    agent decision and can be queued at any time.
  - The statistical procedure and the build rule are cut at their fleet's first
    engagement's final number of changes (section 3). So they cannot be queued before that
    engagement's last decision: the lead fleet's after run A decides, the variant's after
    its run, the later fleet's after the run without the library. They can then fill the
    next engagement's model time.
- **Filling the model time.** Steps 8 and 8b and the two round-1 operations come to 10,800
  planned primary attempts that could fill model time when the GPU would otherwise sit
  idle: 5.3 hours at the 8-worker job's rate.
- **Elapsed time, as scenarios that assume useful overlap** (the work above actually
  fills the model time, within the order these dependencies allow):
  - at the 8-worker job's 2,029 an hour: about **15 to 17.5 hours**;
  - at the 10-worker operations' rates: about 12 to 16 hours;
  - at the 4-worker M2 jobs' rates the simulation sets it: about 27 to 36 hours.
  - Less overlap means more time, up to the simulation and the model time added.
- **Nightly.** The nightly job takes the heavy queue at 04:30 for half an hour each night
  the schedule spans.

## 6. What this proposal does not add

No extra hardening against the statistical procedure or against remembered settings, and
no new calibration, search or benchmark jobs. Every level and mechanism comes from w70,
w71 and the existing calibration cells.

## 7. M2 lessons

*What M2 showed on w70, from the audited records, and what stays open until the w71 reuse
pair is in.*

**The rounds.**
- **Round 1** adopted three changes: the software revert at s2 (L1), compensation 18 at
  s1 (L2) and a gripper replacement at r03 (L3). Their forecast was +19.68 points.
- **Round 2** adopted one more, the revert of r06's command buffer (L4), and stopped.
- **Not adopted:** the knocked camera at r10, and nothing at s3.

**The reserved panel**, 288 attempts per state in M2 (6 per robot and day):

| State | Of 288 | s1, of 96 | s2, of 96 | s3, of 96 |
|---|---|---|---|---|
| original | 171 | 51 | 42 | 78 |
| round 1 | 255 | 90 | 87 | 78 |
| final, and one-look (identical) | 263 | 90 | 95 | 78 |

- **Original to final:** 94 attempts rescued and 2 broken.
- **Round 1 to final:** 8 rescued, all at r06, none broken.

**The comparators on the same panel:**
- The plain search, told the site and the setting: 205, with one action and 144
  experiment attempts.
- The fixed statistical procedure and the build rule: 218 each, with 4 and 3 actions,
  no experiments, and identical outcomes.
- The agent: 4 changes from 816 experiment attempts. The effort is unequal, and that is
  shown.

**What this means for M3:**
- **The second layers partly surfaced.** r03's pads were fixed already in round 1,
  r06's buffer in round 2, and r10's camera not at all. Round 2 added a small gain, 8 of
  288. This is a lesson for reading the lead and the variant, not a reason to make
  either harder. The lead keeps its parts, and the variant, without second layers, is
  where a stop after round 1 is a fair outcome.
- **Probes (M4).** A real agent with probes enabled requested no motion probe. That
  non-use is an honest result, and probes stay off by default in M3.
- **Throughput.** One real 8-worker job ran 360 primary attempts in 638.6 s of total
  elapsed time: about 2,029 an hour by total elapsed, or 2,223 by the manifest's
  completion-rate field. Section 5 sizes M3 on it, with the denominators named. It is one
  job, not a guarantee.

**Still open, pending the w71 reuse pair:** the later fleet's two swap options (section
2) and anything else that depends on reuse. None follows automatically from M2's
lead-fleet outcome.

*Amended: swaps declined, no wait for the w71 pair; see the amendment at the top.*

## 8. The starting configuration

All five investigations use one starting configuration:

- **Which candidate:** the latest fully checked candidate with the accepted
  prompt sentence (provisional competing explanations, each with a rough confidence) and
  the panel video support. That is `5032f73` once its full check and smoke pass,
  otherwise the next checked candidate.
- **Probes** are off by default.
- **The record:** its commit and its prompt and skill digests are recorded before any
  fleet is created, and the same configuration applies to all five investigations.
- **No earlier prompt:** there is no setting to recreate an earlier commit's prompt.
- **Not a go-ahead:** this does not allow creating fleets before the M2 and reuse
  lessons are settled.

*Amended: M3 does not wait for the reuse lessons; see the amendment at the top.*

## 9. Staged commands (not run)

*Amended: M3 runs from explicit reviewed commands and a configured copy of `m3_stage.py`
at 4 workers, not from this launcher; see the amendment at the top of this note.*

*Prepared against the accepted interfaces so that M3 can start once section 8's commit,
the recorded draw and the source are fixed. Nothing here has been run.* The sequence is
`spikes/m3/m3_launch.sh`. Its Python steps are `spikes/m3/m3_stage.py`. The launcher
refuses to run until `M3_LAUNCH=go` and its placeholders are filled:

- `START`: a detached checkout of the starting commit, after `make setup` there;
- `C`: the starting commit;
- `DRAW`: the recorded draw, with three seeds and their hidden and public ids, the ids
  unused in `dashboard/worlds.json`.

The later fleet's swap options stay pending (`LATER_OPTIONS = ()` in `m3_stage.py`).

*Amended: both swaps are declined; see the amendment at the top.*

**The steps, in order:**

1. Compose the three fleets from the draw (CPU), with `m3_lead`, `m3_variant` and
   `m3_later`; the recipes are never registered and no seed is drawn.
2. Each fleet's round-1 operation, 864 attempts at 8 workers.
3. Lead run A, then lead run B: the driver's `rounds`, learning on and probes off (the
   defaults), `--panel-per-robot-day 12 --panel-video-index 0 --workers 8`.
4. The source (section 4), recorded in `source.json`, from each lead run's freeze:
   - lead run A's library if A's freeze kept a skill, also when A's later evaluation
     failed;
   - otherwise, if A's freeze completed and kept none, lead run B's library if B's kept
     one;
   - if both freezes completed and kept none, the outcome "no-source". That is a
     reported outcome, not an error: the reuse pair, its scoring and its exhibits are
     skipped, and so are the later fleet's references, which would be cut at the later
     fleet's first engagement. The other investigations and the lead's and variant's
     references are unaffected.
   - A missing engagement record, a record without a freeze (an engagement that stopped
     before it), or a recorded library that is missing or inconsistent is an error, not
     "no skill". The step stops with the reason named, and the reuse pair waits.
5. The variant's run.
6. With a source, the reuse pair on the later fleet (`reuse_run compare`, without then
   with) and then its score.
7. The plain search per fleet, told the site.
8. The two references per fleet.
   - They are cut at the fleet's first engagement's final number of changes and
     evaluated in a loop directory of their own (`<fleet>-references/loop`), as on w70.
   - If either reference fails, the preparation step fails, so the evaluation never
     takes one reference as the complete set.
   - The layout and the panel are copied from the engagement (the panel byte for byte,
     and `evaluate` mints it again and refuses a difference), with video for index 0.
   - The original state is read from the engagement's evaluation and never simulated
     again.
   - The engagement's evaluation directory is never written: its files are compared by
     content before and after, and a change fails the step. This keeps every agent
     state that the reuse scoring, the exhibits and the clips read.
9. The exhibit per attempted run, whatever its outcome, built with `--no-clips`.
   - Reporting needs only that a run was attempted. A failed or partial run's exhibit is
     built from its records; where that is not possible, its bundle says so.
   - Every bundle gets `status.md`: the run's exit code, the engagement's recorded
     outcome and error, the exhibit's build code, and whether the reserved evaluation is
     available.
   - Its film is chosen by the accepted rule (`fleetlab.loop.clips`): recorded attempts
     only, in panel order, at the condition's site and then elsewhere. It is rendered by
     the exhibit's own `render_pair` into `clips.md`, with `clips.json`. Where the
     evaluation is unavailable, `clips.md` gives the reason instead.
   - `index.md` links both pages, and the bundle is checked as `--check` does.

**Receipts and status:**

- Every step writes its exit receipt, `$M3/<step>.exit`, and its wrapper returns that
  step's own status.
- A step runs only when the receipts of what it needs read 0.
- The launcher exits non-zero, naming the failed steps, if any receipt does not read 0.
- Heavy steps of our own write queued, started and ended lines to the shared queue log;
  the orchestrators log their own heavy steps through `--queue-log`.

**Checked on the CPU, with stubs** (scratch under `/var/tmp`):

- **Wrappers:** they return a failing command's status (7 and 5) with matching
  receipts and queue-log lines.
- **Gating:** when composition fails, nothing else runs and the launcher exits 1. When the
  lead's operation fails, its runs, references and exhibits are skipped with the reason
  given, and the launcher exits 1, naming it.
- **The source:** each of the three branches produces the step's record, and the launcher
  passes A's or B's library or skips the reuse steps. Two failed engagements, a missing
  record and an unavailable library each stop the step with the reason named, and only
  two completed, empty freezes give "no-source". A valid freeze in A whose evaluation
  failed still gives A. A failing source step is reported as a failed step.
- **The references:** one failed reference of two makes the preparation exit 1.
- **The references:** a stubbed evaluation leaves the engagement's evaluation unchanged,
  byte for byte.
- **The clips:** the selector picks the filmed attempt first in panel order, not an
  unfilmed one that comes first by episode id.
- **The exhibits:** a failed lead run still gets its exhibit and status page, its
  references are skipped, and the launcher exits 1 naming it. A run whose exhibit could
  not be built gets a bundle that says so. `index.md` links `status.md` and `clips.md`.

| Step | Inputs | Outputs |
|---|---|---|
| Composition | the draw | the world files under `$W/<id>/` and `m3.json` (truth side) beside them |
| Round-1 operation | the files | `runs/<id>-observed`, `export/` and the receipt |
| Each run | the world | `<out>/loop/` (layout, ledgers, rounds, evaluation) and `<out>/engagement/engagement.json` |
| The source | lead runs A and B | `source.json` (A's library, B's, or "no-source") |
| The reuse pair | the world and the source's library | `$M3/reuse/engagements/{without,with}/` |
| The references | the first engagement's round-1 export, final ledger, layout and panel | `$M3/<fleet>-references/` (ledgers, `loop/evaluation`, `comparison.json`) |
| The exhibit and its film | each run's `engagement.json` and `<loop>/evaluation` | `$M3/exhibits/<run>/`, with `clips.json` and `clips.md` |

Every exit code is in its `.exit` file.
