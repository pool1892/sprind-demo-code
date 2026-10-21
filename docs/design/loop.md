# The two-round loop: interfaces for the builders

*Working contract, 13 October 2026, for the feasibility demonstration. It fixes what the lanes
hand each other, not how each lane builds its part. A builder who finds a
better shape changes this note in the same branch and says so in the handoff.
The inherited runtime, tool server and engine are implementation choices we may
revise: a thin new path that works beats repairs to old machinery, provided it
keeps the authorized model provider, the public-data boundary, real operation
under the adopted state, and an honest comparison.*

## What one run of the loop is

```
round 1: operate (original state) -> export -> agent investigates, experiments from current state -> decision -> ledger L1
round 2: operate (state L1)       -> export -> agent investigates, experiments from current state -> decision -> ledger L2
evaluate: the reserved panel, simulated under L0 (original), L1, L2 (+ the one-look plan), identical identities
reuse:   a related later fleet, the same model and base skills, with and without what the agent wrote down
```

Nothing is injected between rounds to force a second round. The lasting
condition (section 7) is in force from the first day of the fleet.

## 1. Chronology (`fleetlab.loop.layout`, lane state)

`LoopLayout`: `rounds: [RoundWindows(ops: DayRange, experiment: DayRange)]`,
`evaluation: DayRange`, `world_days` (the stored horizon), in strictly increasing day
order. `adoption_day(k)` is the day round k's decision takes effect: the next round's
first operation day, or the evaluation's first day after the last round. A new world
states the layout directly. On an existing development world
(`~/.cache/robots-probes/worlds/wNN`, `mini/w10*`), `layout_for_world(spec)`: round 1
is the world's observed and experiment windows; round 2's operation is the world's
scheduled regular work on its old scoring days, run under L1; round 2's experiment days
and the evaluation days are appended after (as many as the world's experiment and
scoring windows hold, by default). Faults without an end persist; a drift level beyond
the stored table holds its last value (`LoopWorld.faults_on`).

What the world forces beyond this (state lane's choices): experiment days carry only the
agent's arms. The world's own as-run work on round 1's experiment days is in no round's
operation and in no export: those are test days. Regular work past the stored horizon is
refused, not invented; a two-round layout on a stored world never needs it.

The loop's own development fleets (lane fleet, `fleetlab.worlds.loop_fleets`: the lead
fleet `w70`, the later fleet `w71`) are built in this layout: observed = round 1's
operation (days 0-5), experiment = round 1's experiments (6-7), scoring = round 2's
operation (8-11), so that `layout_for_world` appends round 2's experiments (12-13) and
the evaluation (14-17); each also writes `loop_layout.json` beside its files. Every fault
starts within round 1's operation; none is injected between rounds.

## 2. Retained state (`fleetlab.loop.state`, lane state)

```
LedgerEntry  entry_id ("L1", "L2", ...; unique over the whole history), round, adopted_day, action: Action,
             rationale (public), status ("active"|"replaced"|"undone"), retired_day, retired_round,
             retire_rationale, replaced_by, replaces
Ledger       world_id, version (0 = original), entries; active(day) -> [Action]; active_steps(day) -> [Step];
             adopt(round, day, adopt=[Adoption(action, rationale)], replace=[Replacement(entry_id, action,
             rationale)], undo=[Undo(entry_id, rationale)]) -> Ledger (the next version);
             save(dir) -> ledger-r<N>.json (never rewritten with other content)
```

`active(day)` answers from adoption and retirement days, so every stored version keeps
its history exact. `adopt` refuses a round out of order, a day before the ledger's last,
a retired or unknown entry, an action already in force, an incomplete or unsimulated
action, and the undo or replacement of a component verb (a recalibration or a new part is
physical and cannot be taken back).

**One interpretation of a state.** `materialize(world_faults, RobotDay, steps, truth,
current, layers) -> RobotState` gives a robot's state on a day; `LoopWorld.state_on(robot,
day, ledger, arm=())` is the entry point operation, experiments and the evaluation all
use, and `apply_arm` folds an arm's actions on top as a delta. Rules: (1) `revert` undoes
the most recent change at exactly its scope and key: if an earlier step made it (a `set`,
or a revert), the two cancel, as if neither had been taken; otherwise it undoes the
world's recorded change there; a second revert undoes the first. (2) Configuration is
layered: every scope covering the robot (fleet, site, robot) keeps a stack of the fleet's
base value, the world's own change at that scope (a fault, else a benign own value) and
the steps' sets; the narrowest scope with something on its stack decides, so a wider
setting that a narrower one shadows is kept and comes back when the narrower one is
removed (review of 2e151ff). The precedence of what runs is the order of the faults in
force, the order their hooks run in: the world's own behaviour first (its faults in the
world's order, releases included), then the ledger's retained changes in adoption order,
then an arm's. A deciding set adds a generic `config_change` whose nominal is the value
in force before it (for a policy setting what the engine's hooks before it give, else the
base value), so an explicit setting always takes effect, also against a release that
carries the same setting (review of cc1ff68). A table of combinations (a release with and
without the setting, a retained set, an arm override, a revert, an undo) is checked
against the engine's own `effective_spec` (`tests/test_loop_policy.py`); a release's
carried setting shows only in what the attempt runs, never in the records. (3) Component verbs and condition changes
resolve in order with `resolve`; a fault no verb fixes (a lasting condition) stays.
(4) A world change on the same scope and key after a step is refused, not guessed (no
stored world has one). (5) A change a ledger entry adds carries the entry's own fault id
(`entry_fault_id`) and its adoption day as onset. Without steps the state is the world's
own: on w10, w41, w60, w62 and w63 it reproduces every as-run identity's faults.

The adopted actions are the fleet's own record (`believed_fleet`, `ledger_records`): the
configuration after the steps by the same rule (1), a change-log entry by `ops` on the
day a change takes effect (an adoption, or a retirement that takes one back) for every
scope whose own value the ledger changes, a maintenance entry for a component verb, and
the site's believed condition for `change_condition`. There is no condition log in the
visible schema, so a condition change shows only as the site's condition in the
description. The components' latest health in the description is that
day's reading under the ledger, the same number the health table holds (review of
2e151ff). Nothing visible marks whether an adopted action removed a fault; a test checks
that for every robot and key the records and the simulated values agree.

Conventions the driver and the evaluator share: a decision counts `adopt + replace`
against `MAX_CHANGES_PER_ROUND` (3), an undo counts nothing (`decision_changes`); the
number of changes of a state is its entries in force at the end (`changes_in_force`); the
one-look plan is round 1's ranked list, adoptable actions only, cut at the final state's
number of changes and adopted on round 1's adoption day (`one_look_ledger`).

## 3. Operation under a state (`fleetlab.loop.operate`, lane state)

`operate(world, layout, round, ledger, paths, setup) -> run_dir`: the world's scheduled
regular work for the round's operation days, as the world's own as-run identities
(manifest: episode ids, tasks, start states, seeds) with the current software and each
attempt's faults materialized under the ledger in force on its day, through the normal
engine and log writer; then the fleet's records into the finished run (description and
logs after the ledger on the round's last operation day, health of its days under the
ledger, derived events). Round 1 under the original state is the world's stored observed
run itself. Visible tables as in contracts section 11.

## 4. Experiments from the current state (`fleetlab.loop.experiments`, lane state)

`RoundEngine(world, layout, round, ledger, paths, setup)` is the runtime's `Engine` for
one round. `plan_queue` (engine) takes the round's experiment days and a hook that gives
each arm attempt's faults: `LoopWorld.state_on(robot, day, ledger, arm.actions)`, so arms
resolve their actions as deltas against the ledger's effective state (section 2) and the
control arm is the current state. A request is checked against the current catalogue,
and an action outside it is refused (`not_in_catalogue`, in
`toolserver.action_validation`). Allocations are kept per analysis and round; a round's
episode ids continue after every id an earlier round allocated, so one export holds all
arms. Refusal codes stay public and fixed.

## 5. The export a round's agent reads (`fleetlab.loop.export`, lane state)

`export_round(world, layout, round, ledger, paths, out, analyses=[...])`: the operation
runs of rounds 1..K, the agent's own arm runs of rounds 1..K so far, the description,
change log, maintenance log and catalogue from `LoopWorld.public_fleet(last ops day,
ledger)`, and the daily health of every day up to the round's last operation day, each
day under the ledger in force on it (`export_runs` takes the health rows). The pre-export
leak checks of contracts section 11 run on every export, with the truth vocabulary of the
loop's runs and of the world's own runs and files, and the world's full model as the
audit fleet. No evaluation result, oracle value or fault name ever enters it.

## 6. The reserved evaluation (`fleetlab.loop.evaluate`, lane state)

`evaluate(world, layout, states: {name: Ledger}, paths, setup) -> EvaluationSummary`:
the panel is minted once, deterministically, on the evaluation days, and written to
`evaluation/panel.json` before any state runs: per robot and day `episodes_per_robot_day`
attempts, seeds from `seeds_for`, task and start state by the world generator's own slot
rule (version 1 with `generator.json`'s site emphasis, version 0 for the miniature
fleet), ids in a block of their own. The panel lies past the stored horizon, so none of
its identities has ever been run or exported. Every state is simulated on those
identities with its own ledger materialized per attempt (state `original` is the empty
ledger), so no oracle lookup is needed. Output per state and per attempt (truth side),
and `summary.json`: success per state with a 95% Wilson interval, and attempts rescued
and broken between each pair of states, with McNemar's exact p. A panel day inside the
stored world reuses the as-run identities, and state `original` must then reproduce the
as-run outcomes (`reproduces_as_run`). The oracle's fault-removal ceiling is optional.

**The reference methods as states** (`fleetlab.loop.references`, lane state): the fixed
statistical procedure, one pass, logs only (`fleetlab.baselines.classical`, the
`classical-frozen-5` lineage), and the rule that analyses nothing
(`fleetlab.baselines.build_rule`) each read round 1's export, and each ranked list
becomes a state by the one-look plan's convention (`state.one_look_ledger`: adoptable
actions only, cut at the final state's number of changes, adopted on round 1's adoption
day), evaluated on the same panel beside the agent's states. The methods run unchanged.
Each run is recorded (inputs' hashes, status, measured compute seconds) with its costs in
the shared format of `loop/costs.py` (no model session: zero and complete; no
experiment); a failed reference is recorded missing and has no state. CLI: `python -m
fleetlab.loop.references`.

## 7. The lasting condition and how to adapt to it (lane adapt)

A site-level condition the operator cannot undo, in force from day 0, for
example a remote controller link that delivers commands late. The public
description says what an operator would know (for example that the site's
controllers run remotely), never its size or effect, and the catalogue holds no
action that removes it. Adaptation happens through ordinary, generic settings
that every robot has, for example `policy.actions_executed_per_chunk` and a
new latency compensation that drops the stale head of each new chunk. Every
effect comes from the physics simulation; no response curve is planted. The
lane first measures whether some non-default setting beats the default under
the condition. If none does, it finds another principled lasting condition.

The condition must stay in force under every action an operator or the agent
can take (a review finding). Removing a verb from the catalogue is not enough:
the delay is today a configuration key that `set` and `revert` can clear. The
external delay therefore lives in a mechanism that no action resolves away,
while the compensation settings still act on it.

As built (lane adapt; measurements in
[spikes/adaptation](../../spikes/adaptation/README.md)):

- The condition is the fault kind `remote_link_latency`: one fault per robot of
  the site, on its `controller` component, level the delay in control steps,
  onset day 0, no duration, no fixing verb. Its delay line is its own, in series
  with any `control.action_delay_steps`. The generators write the operator's
  note into the site's public description (`Site.notes`); nothing else is
  visible. The fault table reads its curve off `action_latency`.
- The setting is the configuration key `control.latency_compensation_steps`
  (nominal 0): the engine skips that many actions at the head of every chunk
  after an attempt's first. With X-VLA's 30-action chunk it also shortens the
  replanning interval to at most 30 minus the value; a gain is the effect of the
  whole setting.
- A fleet has the setting when its spec lists it in `WorldSpec.later_settings`:
  the fleet default is then recorded, listed in the description's vocabulary and
  offered in the catalogue. Every loop fleet that should have it sets this; every
  world composed before it keeps its bytes and recomposes to its sealed hash.

## 8. Rounds, decisions and the budget (`fleetlab.loop.driver`, lane driver)

```
RoundDecision  adopt: [{action, rationale}] (at most 3 per round), replace: [{entry_id, action, rationale}],
               undo: [{entry_id, rationale}], predicted_set_gain: {low, mid, high} (points over the current state),
               expectations: [str] (what the next round's records should show if the set works),
               stop: bool, ranked_candidates: [RankedAction] (optional; zero is valid)
```

The ranked alternatives are checked by the same public action and completeness
checks as the adoptions, and refusals are reported back to the agent. The
one-look plan is round one's adoptions followed by its valid alternatives, up
to the final number of changes. The record holds the requested cutoff and the
number of changes actually available and applied. A shorter list gives an
honestly shorter plan, and nothing is invented to pad it.

The models live in `fleetlab.contracts.decisions` (mounted with the contracts);
the agent submits with the tool `submit_decision`. Every action must have a
simulated effect; replaced and undone entries must be active in the ledger.
How replacements and undos count against the number of changes is lane state's
convention in `fleetlab.loop.state`.

The driver runs the rounds, carries the agent's notes and its own earlier
decisions and predictions into the next round (never their evaluation), and
records every cost: attempts simulated per experiment and arm (refused requests
cost nothing, failed simulations count), sessions, tool calls, model wall time
and tokens. The budget is set in attempts per round from what the machine runs
(about 1,500 attempts an hour), not from the old 8 requests of at most 60:
`fleetlab.loop.driver.AttemptBudget`, by default 480 attempts per round (about
twenty minutes of the heavy queue). An arm is limited only by what is left of
the round's attempts, unless a run sets a limit per arm (`--per-arm`, the
miniature run's 20); the historical 60 per arm is only the tool server's
default for paths that set no budget. The engine still refuses an arm that does
not fit the experiment days (160 slots per robot and day after the regular
work). The tool server enforces the budget from the round's first session
(`Limits.budget_first_session`), and the tool `experiment_budget` reports it and
what is left (`ExperimentBudgetStatus`: attempts, attempts_used, attempts_left,
max_per_arm, requests, requests_used, counted_from_session).

What crosses rounds is carried two ways. The round task (strict pipeline text)
names the round, says that stopping is a valid choice, and lists the ledger's
actions, days and status and each earlier round's session numbers. The agent's
own earlier decisions, reasons, predicted gains and expectations stay in its
outbox (`round_decision.json`, `submissions.jsonl`), are audited as agent prose
at every launch, and are read back with the tool `earlier_decisions`; they are
never pasted into the task text. One analysis spans the engagement, so the
notes carry over by construction and experiment ids stay unique.

An engagement ends `completed` (every round decided) or `stopped` (the agent
stopped), both exit 0, or `aborted` (a round without a decision: an auth or
rate limit, a lost call, sessions exhausted), exit 3 and not evaluated unless
asked. Learning is on by default: `write_learned_skill` is a round tool, the
task carries lane learn's paragraph, and after the rounds the host freezes the
drafts (`freeze_learned`, each with its round) and records the library's root
and digest; `--learned <root>` verifies such a library and mounts only its
skills tree into a later engagement.

## 9. What the agent learns and keeps (`fleetlab.loop.learned`, lane learn)

*How it is built, the rules and the reuse harness: [learned.md](learned.md); the
reuse run's specification and runbook: [reuse-run.md](reuse-run.md).*

The agent can write a diagnostic skill during an investigation: `SKILL.md` in
the existing skill format (contracts section 13: front matter, SQL blocks run
through `query`) with the required fields `applies_when`, `evidence` (what it
saw, as queries and results), `next_experiment` and `provenance` (fleet label,
round). After the investigation the host checks it (the agent-text leak rules,
and no identifiers of the source fleet's robots, sites, episodes or
experiments except as parameters), freezes it into a library with a digest,
and a later investigation can be given that library beside the unchanged base
skills. The reuse comparison runs the same later case with the same model and
base skills with and without the library, and compares investigation cost and
final outcome. Each arm starts as a separate, fresh analysis with an identical
export, model, base skills and budget. The learned library is the only thing
transferred.

## 10. Stretch: measurements the agent composes (lane probe)

Generic measurement channels with a cost, for example command and sensing
timestamps, commanded against measured joint positions at a higher rate, and
gripper force. The agent composes a probe from them and labels it as its own.
A channel measures behaviour; it never exposes a planted parameter.

As built (lane probe; `fleetlab.contracts.probes`, `fleetlab.engine.probe_player`,
`fleetlab.loop.probes`). It is an opt-in probe path: exercised by scripted agents on CPU
fixtures and by a host-selected scripted check on the physics simulator, not yet by a
real model.

- **What the agent composes.** `request_probe(ProbeRequest)`: its own name for the
  probe, the question it answers, what it expects to see, one to four robots, one to five
  repetitions per robot, and a program of up to twelve segments played in order, each a
  primitive of a small public vocabulary: `hold`, `step`, `ramp`, `sine` on the x, y or z
  axis of the frame the robot reports its hand in, and `open_gripper`, `close_gripper`.
  Targets are offsets from the hand's position at the probe's start; the orientation is
  held; the gripper starts open. Public bounds (`PROBE_BOUNDS`): offsets within 5 cm, at
  most 0.5 m/s in a ramp or sine, periods of at least 4 steps (5 Hz), at most 200 steps a
  segment and 400 (20 s) a probe. The arm takes end-effector targets (operational-space
  control), so there are no joint targets; joints are recorded as measured. There is no
  force channel: a free-space probe touches nothing, so it would read zero.
  `probe_results` echoes the agent's own definitions with each probe's status, episodes
  and charged attempts.
- **Where it runs.** A probe episode is an engine episode whose identity carries the
  program (`EpisodeIdentity.probe`, left out of the dump when absent, so no existing
  identity or chain moves) and whose policy is the probe player, an in-worker policy
  (contracts section 6) that commands `start + offset[k]` in the frame the robot reports,
  through every action hook in force, like the policy's commands. Its faults are the
  robot's current state under the ledger (`LoopWorld.state_on`), on the round's
  experiment days; its ids come from the analysis' allocations under the arms' lock; it
  runs in a probe run of its own (`rounds/r<K>/experiments/<analysis>/probes/session-NN`)
  that serves no policy (CPU, light). The run's chain sample replays probe episodes like
  any others (`chain_ok`).
- **What it costs.** Each probe episode is one attempt of the round's experiment budget,
  reported apart from experiment-arm (task) attempts in the receipt and in
  `experiment_budget` (`task_attempts_used`, `probe_attempts_used`); at most four probe
  requests a budget period. Outcomes (`ProbeOutcome`) are `done` (every episode, charged),
  `refused` before anything ran (free) or `failed` (the run broke off: the whole request
  stays charged, as `attempts_executed`, the episodes the run left a record of, and
  `attempts_reserved`, the rest, which the runner cannot confirm either way, because an
  in-worker probe that a lost worker was playing leaves no record; partial records are
  withheld). Outcomes are bound to their request and to the export's probe rows, visible
  only in a later session, and any contradiction refuses them as a whole, as for
  experiment results (tool server, round driver and launch audit alike). The loop's cost
  records count probes and their runs apart (`loop.costs.probe_costs`, `queue_costs`,
  `WorldLoop.simulated()["probes"]`, the heavy queue's records and log lines).
- **What it records.** Visible tables only an export holds (`TableSpec.source = "export"`;
  never part of a run, absent from an export without probes, whose bytes do not change):
  `probe_episodes`, `probe_segments` and `probe_steps`. Timing of `probe_steps`: row k is
  one instant, `time_s` = k / 20 s after the probe started; the state is measured first
  (hand position and orientation, joint positions and velocities, gripper opening), then
  command k is issued (`target_pos`, `gripper_command`) as the probe submitted it, before
  anything on the command path; with nothing on the path a command shows first in row
  k + 1; the last row has no command.
- **Honesty.** The tables hold commanded targets and measured state, nothing of the
  command path's own: no receipt or arrival time, no configured value, no fault name, no
  executed command. The export's leak audit reads them (a planted term in a probe table is
  caught) and searches the probe runs' truth as well. A lag an agent reads from them, as
  `loop.probes.step_lags` does, is a threshold-crossing measurement of the hand's response
  to a commanded step, not the delay of the command path itself: it includes the
  controller's own response time.
- **Keeping a probe.** A learned skill (section 9) can carry a probe's definition in its
  next experiment, with robots as parameters, and the query that reads it as evidence; a
  later investigation can request it again.
- **The switch.** `Engagement.probes` (`run_rounds(probes=True)`, the driver's
  `--probes`) offers both tools, adds the round task's probe paragraph
  (`loop.probes.PROBE_TASK`, which hints at no condition) and puts the tools' texts into
  the launch gate's audited texts. Without it, nothing of a run changes.
- **Measured** (host-selected scripted check on the physics simulator, miniature fleet
  w10-v2, round 1 under the original state, one step program of 100 steps: hold 10, step
  +3 cm on x for 40, step back for 40, hold 10, three repetitions on each of two robots):
  r01, at the site without a command delay, first moved one row after each commanded step
  and covered half of it in 4 rows; r05, at the site whose command delay setting is 17
  steps, first moved 18 rows after each commanded step and covered half of it in 21 rows,
  in every repetition. No probe table carries the condition's name or value. Under a
  retained state (round 2, a ledger whose entry, adopted on day 3, takes back that site's
  command delay setting; the same program on day 4, twice on r05 and once on r01, read
  from the probe run's own steps, without an export, which would need round 2's operation
  run), r05 first moved one row after each commanded step and covered half of it in 4
  rows, like r01: the probe ran on the robot as the ledger left it. The chain sample of
  both probe runs replayed exactly. These checks were selected and scripted by the host;
  no model composed them.

## 11. Working rules for every lane

- Hidden information never reaches the agent. Never open a path containing
  `unseen` or `HELD`, the branch `held/unseen-faults`, or test-world truth. The retired campaign and its draw stay
  retired.
- Simulator and GPU work goes through `tools/heavy-run` (`--light` for CPU
  simulation), from a committed checkout, with scratch under
  `/var/tmp/robots-loop/<lane>/`; each heavy job is logged in
  `/var/tmp/robots-loop/queue.log` (start, end, commit, purpose, attempts).
- Builders commit on their own branch in their own worktree. They do not merge
  to main, push, publish, or start model reviews. The integration owner merges
  lanes into `loop/integ`.
- Shared files (`contracts/tools.py`, `toolserver/server.py`,
  `agents/sessions.py`) take small additive edits, named in the handoff.

## 12. What an observer sees (`fleetlab.loop.exhibit`, lane exhibit)

`python -m fleetlab.loop.exhibit build RUN --out DIR [--check]` turns a finished run into
a Markdown bundle for publishing: `index.md` (the summary), `round-<N>.md`, `provenance.md`
and `provenance.json`, `figures/*.png`, `clips/*`. RUN is an engagement record
(`engagement.json`), an older single-round investigation (`run.json`), or a manifest
(`exhibit.json`, format `fleetlab.exhibit-manifest/1`) that assembles rounds, the
evaluation, the reuse comparison, a frozen learned library, the world for the planted
truth and reviewers' annotations from separate records. Truth side, never mounted. A
manifest's `hub_id` (the artifact a bundle was published as) is written as the
summary page's front-matter `id:`, so every re-render keeps the identity; without one,
none is written.

- **Per round:** the fleet's regular work by robot, site and day with Wilson intervals,
  and a fixed rule for where it is weak; the working table (each target the agent's
  diagnoses named or its experiments acted on, its confidence before and after the first
  test, withdrawn when the last diagnosis drops it); the experiment ledger (each arm
  against control, paired rescued and broken with McNemar's exact p, an interval for
  unpaired arms, and what the decision did with the arm's change); the decision with the
  predicted set gain labelled as a forecast; the next round's regular work beside that
  forecast, labelled as not the evidence; the learned procedure verbatim; the round's cost.
- **Summary:** the plan's headline placeholders filled where the records have them; the
  reserved panel by state with intervals, the one-look plan's counts, rescued and broken
  between every pair, by robot; the planted truth per round (development worlds only,
  `dashboard/worlds.json`); the reuse comparison side by side; the cost table.
- **Numbers:** every table and figure is an item in `provenance.json` with its files
  (SHA-256) and derivation. A missing figure is "not recorded", a usage figure with model
  starts that reported no usage is "at least", a lower-bound price (also nested, as the
  reuse comparison wraps the driver's) stays "at least"; nothing missing is shown as zero.
  Experiment task attempts and probe attempts (section 10) are counted apart, each with
  executed, failed, unconfirmed and replayed simulations; a failed probe's reserved rest
  makes its count "at least"; a record from before probes says probes are not recorded.
- **Origin:** a file under a directory holding a `FIXTURE` marker is a fixture; a run
  with the scripted agent is a toy run. Pages, captions and figures built from either say
  so (banner, `[FIXTURE]`/`[TOY]` in the caption, a mark across the figure).
- **Film:** the publishing site renders images and drops `<video>` (its sanitizer has no
  video tag; checked with `hub check`), so each pair is an animated GIF of
  both scene videos side by side, a six-frame strip of each, and the two MP4s linked. Pairs
  are chosen by rule: the first rescued pair of each changed arm that beat its control; on
  the reserved panel the first attempt rescued and the first broken between the original
  and the final state.

The records give "confidence before testing" only when the agent submits a diagnosis
before its experiments' results arrive; a run without one shows "not stated".
