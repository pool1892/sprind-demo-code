# The skill library, version 1

Procedures an analysis agent follows with the tool server's tools, one
directory per skill, in the format of the contracts
([`docs/design/contracts.md`](../docs/design/contracts.md), section 13).
Claude and Codex read the same files: the tool server lists them
(`list_skills`, which returns each skill's front matter) and serves them
(`read_skill`, which returns the `SKILL.md` text); the CLIs' own skill
loading stays switched off.

## What is here

| Skill | For | Step | Gives |
|---|---|---|---|
| `triage` | analysis | first look | outcome table by fleet, site, robot, day and task with intervals and task-mix expectation; a stratified sample |
| `episode-cards` | analysis | look at episodes | one card per sampled episode: phase, symptom, evidence frames, candidate cause; general knowledge of timing in a control loop and the competing explanations of a timing symptom |
| `failure-modes` | analysis | group | modes with a telemetry fingerprint, counted over all failures |
| `scope` | analysis | locate | the narrowest scope and the moment; change-log, maintenance and health readings against it; confounded entries |
| `sensitivity-from-logs` | analysis | estimate | expected gain per candidate action in points of fleet success, with an interval, from the log |
| `sensitivity-by-experiment` | analysis, with experiments | measure | a spending plan against the round's attempt budget; A/B requests with a power note and their cost; their read-out |
| `rank-actions` | analysis | decide | catalogue actions in order; at most three adopted (or none), undos and replacements of earlier changes, and one forecast for the whole set |
| `write-diagnosis` | analysis | answer | the round's decision and the `Diagnosis`, submitted |
| `assess-risk` | live supervisor | judge | phase, signs of trouble, chance of failure, recoverable or not |
| `choose-intervention` | live supervisor | decide | continue, retry, stop or hand over, by expected cost |

Each directory holds `SKILL.md` (front matter of
`contracts.skills.SkillManifest`, then the procedure) and, where the skill
computes something, `<skill>_helpers.py`. Which skills each role gets is
named once, in `fleetlab.skill_library` (`ANALYSIS_SKILLS`,
`SUPERVISOR_SKILLS`, `skill_dirs(role)`), for the runtime to mount.

## How an agent composes them

The ordinary path through an analysis is the order of the table:
triage, episode-cards, failure-modes, scope, sensitivity-from-logs, then
sensitivity-by-experiment when experiments are allowed, rank-actions and
write-diagnosis. Each skill ends by naming the next one and writes what it
found to memory, which is the thread between skills and the only state that
crosses sessions.

A hard case does not follow the table. The skills are written to be called
again and out of order:

- a scope that loses more points than its modes explain sends the agent
  back to triage's `sample` (another `salt`) and to episode-cards;
- cards that fit no mode reopen failure-modes;
- a confounded pair found in scope goes to sensitivity-by-experiment before
  any gain is trusted;
- results of experiments, in the next session, go through scope (`changes`,
  `breakdown`) and the gain estimate again, and rank-actions re-orders;
- a fleet-wide scope has no comparison group, so its gain comes from
  scope's `changes` instead of sensitivity-from-logs' `gains`;
- write-diagnosis can be run early for a provisional answer.

## SQL blocks and helpers

The computations an agent should not do by hand (rate tables with Wilson
intervals and task-mix expectations, a stratified sample, telemetry
signatures, differences in differences, gap-based gains with intervals, a
power calculation with a request's cost, the adopted top of a ranked list and
the forecast of a set) are single SQL statements in
the `SKILL.md` text, each starting with a `-- helper: <name>` line. The
agent copies a block, edits only its `params` line and the input rows, and
runs it with `query`. An input sits between a `-- input: name(columns) AS
(VALUES ...)` comment, which shows the form of its rows, and a `-- end
input` line; the agent replaces the default there (a SELECT that makes the
block run as it stands) by a `VALUES` list in that form. Four blocks
(`outcomes`, `gains`, `breakdown`, `trend`) have an optional `excluded`
input for leaving episodes out. Every block that computes rates leaves out
the episodes of requested experiments (`episodes.experiment_id`), which
are read only through `readout`. `query` returns at most 200 rows (marked
`truncated`) and stops after 60 seconds; the blocks stay under 200 rows for
a fleet of a few dozen robots, and triage tells the agent to narrow the days
when a result is truncated.

This is the whole mechanism; the tool server needs nothing beyond `query`,
and the agent has no shell to run anything else in. Running the helpers
through `query` has a property the design needs: the tool log records the
SQL and its result, so every number an agent reports can be traced to a
call and re-run by the scorer. The `-- helper:` line
in the logged SQL says which block produced it.

`<skill>_helpers.py` is the reference runner of a skill's blocks: it reads
the blocks from `SKILL.md` (one source for the agent's text and the tested
code), fills parameters and input rows, and runs them through
`fleetlab.log.LogReader`, opened materialized and locked as the tool server
opens it. The tests drive it; a person can run it:

```sh
uv run python skills/triage/triage_helpers.py <export dir> outcomes --params '{"from_day": 0}'
uv run python skills/rank-actions/rank_actions_helpers.py <export dir> cut --sql   # print, run nothing
```

The runner part of every helper is identical (a test checks it), so that a
skill directory is complete on its own when it is mounted. If a later
version needs computation SQL cannot express, the tool server would need a
tool that runs a helper; that is a change to the contracts' tool list and is
filed as an idea, not built.

The two supervisor skills have no tool server: the supervisor gets its
context in the prompt. Their helpers are for the harness that builds that
context: `assess_risk_helpers.signs` summarises a telemetry window, and
`choose_intervention_helpers.choose` computes the decision rule's expected
costs from the numbers, to check or log the supervisor's choice.

## The leak rules and the review that reads them

A skill holds analysis methods and knowledge grounded in the public system
description, dictionary, action catalogue and vocabulary. It holds no world's
identifiers, planted schedule, fault kinds, traits, oracle values or empirical
claims about the test bed's fault response curves. This excludes qualitative
restatements too: which settings are harmless, which pairs interact and which
verbs completely remove a planted mechanism. Development scores are records of
performance, not a source of priors about a mechanism.

The freeze candidate starts from Round 12. Rounds 13 and 14 remain historical
records in [ROUNDS.md](ROUNDS.md), but their procedure additions are withdrawn.
The removal inventory there identifies the earlier fault-library claims that
were also removed. The component glossary now belongs to the public `describe`
response; the library does not prescribe a controller diagnosis for recording
symptoms.

Retained passages of class knowledge are bracketed by
`<!-- fault-class-knowledge:begin -->` and `<!-- fault-class-knowledge:end -->`.
`make analyse ANALYSE_FLAGS="... --ablate-fault-classes"` materializes a separate
content-addressed library with those passages removed from every mounted file.
Malformed markers fail closed. Statistical SQL and procedural steps remain.
The run records the ablation choice and digest of the actual mounted library;
the public system description remains available in both conditions.

Mechanical checks reject literal identifiers, known hidden names, canaries,
seed values and some encoded representations. The current-content launch audit
hashes and scans the exact export, sources, skills, prompts and agent-written
state. Synthetic tests cover the ablated library through that launch boundary.
They do **not** prove semantic generality: a paraphrased planted schedule or an
unnamed oracle-derived response prior can pass. Independent reading after large
changes and at the freeze remains necessary. Nor can these checks establish
that a builder never used a test world, or that a reported first run was chosen
without selection; provenance and the launch-time repeat registry support that
audit, while development/test access and the freeze remain operator obligations.

The SQL constants are statistical thresholds, public budget limits, or telemetry
heuristics measured from the miniature fleet's exported episodes (below).
Those telemetry thresholds are not dose-response measurements of planted faults.
They remain provisional outside the measured gripper and policy.

## Learned skills, beside this library

An analysis agent can write a diagnostic skill of its own during an
investigation (`write_learned_skill`; [docs/design/learned.md](../docs/design/learned.md)).
Learned skills never enter this directory: the host checks and freezes them
into a separate, content-addressed library that a later analysis receives when
it is created and that is mounted at `/learned`. `list_skills` lists them after
these skills with `origin: learned`, `applies_when` and `provenance`; they
cannot take or shadow a name from here, so this library stays byte-identical
in an analysis with or without them.

## Choices made where the contracts are silent

- **Comparison groups.** `gains` compares a scope with the episodes no other
  candidate covers (`exclude_suspects`), because in a world with ten causes
  the "rest of the fleet" is not healthy, and its interval combines the
  uncertainty of the scope's rate and of the comparison's. Each robot also
  gets its gap against the other robots of its own site (`gains`:
  `within_site_gap`; `trend`: `within_site_slope`), which is how a site's
  step is told from a robot's drift. `health` scores each reading against
  the other robots of the same day, so one outlier does not hide itself by
  widening the spread.
- **Signatures.** Whether something was held is read off the finger
  opening once it has settled, relative to the episode's own widest
  opening; a drop event counts as a loss; grasp and release events give
  only the moment the gripper was commanded. The thresholds are measured
  (next section).
- **No arithmetic in the agent's head.** A supported factor on a gain (a
  measured partial response or an uncertain attribution) is an input row's
  `attributable`, so the block returns the number; sensitivity-from-logs
  names the returned value behind every low and high.
- **The catalogue, costs and the decision** follow the contracts: the
  actions are the visible table `catalogue` with their hours, the description
  comes in parts (`describe` with `part` `dictionary`, `fleet`, `vocabulary`,
  `costs` or `all`, each at most 30,000 characters), and a round's
  decision (`RoundDecision`, docs/design/loop.md section 8) adopts at most
  three changes. rank-actions' `cut` block looks every action up in
  `catalogue` and marks the adopted top of the list; its `set_gain` block
  forecasts the set from disjoint parts of the fleet (changes on the same
  robots: gains combined as independent removals, a range that lets their
  benefits overlap and their possible harms add up; parts added, half-widths
  combined in quadrature), never as a sum of single gains.
- **Helper names.** `<skill>_helpers.py`, unique per skill, so that mypy can
  check them all in one run.

## The finger thresholds, measured

Measured on 2026-10-08 on the miniature fleet's fixtures
(`~/.cache/robots-fixtures/mini-v1/`, the runs of its observed and later days):
416 X-VLA episodes on libero_spatial (black bowls, 228) and libero_object
(bottles and boxes, 188), 259 successes and 157 failures by the fleet's
check. The opening is `|gripper_qpos[1]| + |gripper_qpos[2]|` over the
episode's widest (0.079 to 0.081 m everywhere).

**The events say nothing about holding.** Every one of the 416 episodes has
a grasp event (947 in all), with 822 release events and no drop event:
the engine reports the gripper's *command* ("gripper commanded closed",
"gripper commanded open"), not whether an object was caught. The rule of
the review's first round (no grasp event means closed on nothing, a release
event means released) therefore found no missed grasp and no lost object
on these episodes and called all 157 failures "released, not placed". The
blocks now read holding off the fingers and take a drop event, where a
detector reports one, as a loss.

**The fingers close slowly.** Two steps after the close command the opening
is still 0.65 to 1.0 of the widest; ten steps after, a bowl rim still reads
0.14 to 0.23. The old reading at two steps after closing called every one
of the 416 episodes "held". With nothing between them the fingers creep
down for 20 to 40 steps and settle near 0.013 of the widest (0.0009 m at
the narrowest).

**Settled readings by what was held**, over closed spells of at least 30
steps (from a grasp event to the next release event):

| held | spells | settled opening (share of the widest) |
|---|---|---|
| nothing | 13, all in failures | 0.010 to 0.02 |
| a bowl by its rim | 182 in successes | 0.031 at the lowest, 0.041 at the 2nd percentile, median 0.06 |
| a bottle or box | 90 in successes | 0.42 and above |

So the rule of 0.15 of the widest, read once the fingers settle, would call
159 of the 169 successful bowl episodes "closed on nothing", and 0.05 still
23 of them. The blocks now use `empty_below` 0.025 and read the median
opening from `settle_steps` (25) to ten steps more after the fingers close,
from the first closed stretch long enough to settle. Over the 416 episodes:

| settle window | successes read as held | failures: closed on nothing / lost the object / never closed / released, not placed |
|---|---|---|
| 15 steps | 259 of 259 | 6 / 11 / 1 / 139 |
| 20 steps | 259 of 259 | 7 / 6 / 2 / 142 |
| **25 steps (the default)** | **259 of 259** | **10 / 3 / 1 / 143** |
| 30 steps | 259 of 259 | 13 / 2 / 3 / 139 |

Shorter windows read a slow creep onto nothing as a hold and then a "loss";
25 steps catches most of the 13 spells that settle on nothing while keeping
every held bowl. The supervisor's `assess_risk_helpers.signs` uses the same values and agrees
with the profile on all 416 episodes (10 on nothing, 3 lost, 403 neither).
The grasp settles within 25 steps for this gripper and this policy; a
policy that closes faster or slower needs the window measured again.

## Measured use and remaining limits

- Earlier miniature and development analyses exercised earlier library
  revisions. Their scores and limitations remain in [ROUNDS.md](ROUNDS.md).
  No previous score is evidence for this revised freeze candidate. This repair
  round makes no analysis call, performs no oracle scoring and reads no new
  world. The orchestrating session schedules its independent evaluation.
- The ablation and restored SQL blocks have synthetic checks. They establish
  the transformation, launch compatibility and arithmetic, not analysis quality.
- The finger thresholds are measured on 416 recorded episodes (above), the
  stillness and reversal measures are not. Most tests run on toy exports
  written through the log writer; the block tests also run on the
  miniature fleet's export when its fixtures are on the machine.
- Full development exports have now exercised the telemetry blocks through
  the production tool server, including its query timeout. Successful calls
  and custom-SQL errors are preserved in the tool logs. This is observed
  feasibility at the measured size, not a standalone memory or scaling benchmark.
- The power calculation uses the normal approximation for two proportions
  and ignores paired starts; it matches the textbook formula in the tests,
  not a simulation.
- The experiment budget: a round's budget is counted in attempts over every
  arm of every charged request (the round driver sets it, with a limit per
  arm only where a run asks for one; the old pipeline's default stays 8
  requests of at most 60). Publicly invalid requests are refused before
  debit; engine refusals release their debit without reusing request
  identifiers.
- A site's step against a robot's drift: the within-site slope separates
  them on the toy fleet (one drifting robot, one site with a step); a site
  whose robots drift together, or a drift that starts on the day of a site
  change, was not tried.
- With 24 robots or more, the text columns of `signatures` grow long; the
  row limit holds, the readability for a cheap model was not tried.
