# The miniature fleet end to end

## Development-world analyses (stage 4)

`make analyse` runs the existing sandbox runtime on a cleared observed-window
export, then invokes the development-world scorer. Use a detached checkout at
the commit being measured, with its own `make setup` environment:

```sh
make analyse ANALYSE_FLAGS="--world-root /path/to/world --export /path/to/cleared-export --out /path/to/new-result --condition logs"
make analyse ANALYSE_FLAGS="--world-root /path/to/world --export /path/to/cleared-export --out /path/to/another-new-result --condition experiments"
```

The default model is `gpt-6-astra` through Codex, subscription authentication,
Fast tier and `--effort xhigh`. `--model` is recorded from the actual run
configuration and used to select a reviewed API-price snapshot. The command
uses only Codex. No hidden world identifier enters the system/task prompt or
mounted files: only the public day ranges and fleet label are visible.

The initial export is bound to the truth-side fleet model's **public label**
and the exact visible episode records of its base runs. Its label need not
equal the hidden world id. It must contain observed-window episodes only.
The operator must obtain the repaired export check's clearance first.

Output is a new directory outside the stored world. It contains the analysis's
memory, submissions, requests and sessions; `agent-runs/` contains model-call
records and tool logs; `engine/` contains experiment runs; `score/` contains the
host-only evaluator report. The agent never receives world files or scores.
`run.json` records source commit, world hash, model configuration, actual library
digest, ablation choice, repeat reservation and cost accounting.

Repeat indices are allocated atomically at the first model-call dispatch:
after the gate, per-session audit and successful CLI process creation, immediately
before sending its task. A refused gate/audit or failed process start has no
index. The key is world hash, condition, library content digest, model and effort.
Immutable numbered receipts live at the OS account's home directory under
`.cache/robots-probes/analysis-repeats-v2/`; neither `ROBOTS_AGENT_ROOT`, `HOME`,
`XDG_CACHE_HOME` nor an output path selects another register. `run.json` carries
the key, ordinal, receipt path and SHA-256, first-call path and launch timestamp.
`accounting.verify_repeat` checks these against the fixed register before scoring
and on continuation. A failed dispatched call keeps its number; this is a launch
ordinal, not proof that the provider accepted the request. There is no
caller-selected `--repeat`. Continuations retain the original receipt and require
the same group. These checks cannot prevent a host operator from editing records
or deleting the register; the reviewer still checks the retained run history.

For a rebuilt export, pass `--base-run PATH` for each corrected historical run.
Before any experiment analysis, the pipeline rebuilds the selected base runs
into `checked-base-export/` and requires every byte to match the cleared export.
The engine uses those same base runs when it extends the export.

An ended waiting checkpoint can continue with `--resume-from OLD_RESULT` and a
new `--out`. It copies memory, requests, results, queues and engine state,
preserves the analysis id and original call paths, and never changes the source.
A saved completed queue reply is revalidated and reused without dispatching it
again; already recorded results must agree and are not appended twice. A pending
queue retains its original bytes so the engine can resume its saved plan.
Active or delivered analyses are refused. The synthetic loop test covers a
completed queue followed by interruption and successful continuation. Recovery
of a partially executed real simulation is not measured in this round.
A stopped caller cannot mark a started worker as ended: an interrupted queue
without a worker completion record refuses copying until its state is resolved.

Only experiment arms take the shared heavy lock. `--workers` defaults to four;
the policy batch size and the engine's memory/GPU guards remain in force. Public
preflight checks every action in every arm: value domains, target existence,
coverage of requested robots, visible history for `revert`, and distinct arm
sets. A refused request takes no request or episode budget. Engine refusals
release the debit, retain their id, and allow a later session to answer. The
runner currently cannot switch a loaded policy's precision or introduce new
camera-pose/software settings; these limits are public in `describe`. A recorded
revert is still permitted. Catalogue eligibility does not promise a gain.

Request-specific resolution failures produce fixed public refusal codes for
that request alone. Machine failures (including the memory guard and a lost
worker) propagate, stop the analysis in its waiting state, and leave the exact
queue and saved plan for `--resume-from`; they never become blanket refusals.
`run.json.experiments` counts requests, delivered completions, refusals, pending
requests and delivered arm episodes. A label of experiments alone is not evidence
that any experiment was delivered.

The host keeps exception details. Results use fixed public refusal codes;
request/result counts and episode membership are validated before mounting.
The launch audit also requires every experiment episode to lie in the configured
experiment days, including rows not cited by a result. Scoring-day arms refuse.
There are up to eight charged requests, sixty attempts per arm and seventeen
sessions by default, allowing readouts after refusals. A login or rate-limit
error stops before any additional queue or model call. An unanswered analysis that dispatched a model call
still has its record and evaluator result; it is never silently removed.

Claim-6 accounting includes every saved CLI attempt, including failed attempts
and prior continuation segments, token totals, model wall time, launch-audit
time, queue waiting time, simulation time and queue-export audit time. Queue
waiting includes dispatch and interpreter startup after acquiring the lock;
simulation time brackets `fleet.run`, including its worker startup/cleanup.
Queue dispatch ids prevent copied continuation histories from being counted twice.
The observed-episode count is the per-thousand denominator.
`wall_seconds` is the current invocation's elapsed time; the model and queue
measures include prior continuation segments.

API equivalents use a dated snapshot of the
[official API prices](https://developers.openai.com/api/docs/pricing), with Fast
rates selected for the configured model. They are not subscription charges.
Because Codex's totals do not expose each request's context tier or cache writes,
the record retains lower and upper price bounds, including the per-thousand
bounds. It does not invent an exact bill. Missing usage is explicitly unavailable.
An interrupted CLI call without a result makes the aggregate cost and model
duration incomplete. Interrupted queue phases with no duration remain unknown.
No price or timing from a new full analysis was measured in this repair round.

`--ablate-fault-classes` removes explicitly marked passages from a separately
materialized skill library. The run records the resulting digest. Only the hypothesis block in `episode-cards` is marked. Scope comparisons,
public action semantics and ranking arithmetic stay in both libraries. The public
description and catalogue remain available. `describe(dictionary)` supplies the
placement completion criterion and explains that recording changes affect the
record rather than physical success; `describe(vocabulary)` supplies the
controller glossary and the executable condition-value domains. The freeze candidate starts from
Round 12, with empirical fault-library claims removed; earlier scores apply only
to their recorded revisions.

The honesty checks catch literal hidden names, canaries, seeds, forbidden source
imports, invalid export structure, mismatched observed records and unsafe replies.
They check exact mounted bytes at each launch, including agent-written state.
Pipeline prompts/schemas/tool descriptions use `texts`; trees and mounted code
are always strict. Only recognized, source-named agent prose uses `agent_texts`,
which permits natural-form hypotheses, record-layout names and countable fault
ids but refuses seeds, canaries, non-countable fault ids and identifier spellings
of planted/hidden names, including within longer identifiers. It cannot flag single-word hidden
names or natural spellings in that prose. The runtime README lists the exact
prose locations and the conservative treatment of unknown state fields/files.
They cannot detect a skill that paraphrases a planted schedule or an unnamed
oracle-derived prior; such a skill can launch. Independent reading remains
necessary. They also cannot prove that development avoided test worlds, or that
a builder froze its pipeline before seeing them. World provenance, revision
records and automatically reserved repeat indices make those obligations auditable.

Verification includes read-only development records as well as synthetic fixtures: `test_analysis_pipeline.py` exercises `analyse()` with
stub model calls, dispatch and scoring, including auth stops, finalization and
continuation after a saved successful queue. `test_agent_launch_audit.py` checks
public prose, hidden-name refusals, experiment-day isolation and the ablated
library with the real bundled source. `test_toolserver.py` and
`test_experiment_validation.py` check the malformed requests, no budget debit,
public glossary and exception isolation. `test_engine_experiments.py` exercises
the CPU toy queue. `test_real_world_launch_audit.py` derives thousands of terms
from the stored, retired development worlds and checks the real pipeline texts,
schema, both libraries, bundle and stored states, plus planted-name/seed refusals.
The site-condition regression resolves each supported form and invokes the
registered hook's `new_episode`, without constructing a simulator or measuring a
response. None proves a full development-world experiment analysis.
No model session, GPU job or smoke replay is part of this repair round; a fresh
CLI wire gate is deferred to the scheduled analysis's automatic preflight.

Prepared full run (the orchestrating session supplies an approved world/export
and a new output path, from an immutable run checkout at the delivered commit):

```sh
make analyse ANALYSE_FLAGS="--world-root /approved/world --export /approved/export --out /new/experiment-result --condition experiments --model gpt-6-astra --effort xhigh --workers 4"
```

Add `--base-run /approved/corrected-observed-run` when the cleared export was
rebuilt from corrected historical logs. Add `--ablate-fault-classes` for the
separate ablation. This command has been prepared, not executed in this round.

Historical stage-4 measurements (2026-10-09; not measurements of the freeze
candidate): the orchestrator cleared the rebuilt D1
export for development despite its identifier/seed weakness.
Its numbers remain provisional and development-only until a neutral export or
regenerated world is used. Two logs-only attempts made no submissions after
tool transport failures. After the nested sandbox lifetime repair, the first
two completed logs-only repeats met all four session marks: each has recall
6/6, decoy rejection 1.00 and achievable top-three share 1.00; their gain-rank
correlations are 0.9461 and 0.9538. The frozen classical baseline also recovers
100% at three fixes, so the agents tie it on this measure. Its original inputs
have been checked for feature equivalence to the rebuilt export; its answers
were reused, not freshly executed. Experiment continuations are still pending.
The orchestrator then cleared the old D2/D3 exports on the same development-only
terms. Two logs-only repeats each at the unchanged procedures found 4/7 causes
on D2 and recovered 35.6% of top-three achievable gain; both miss the session
marks. Both D3 repeats found 6/7 and recovered 73.8%, meeting the four marks
but trailing the frozen classical baseline's 100%. Their traces exposed
unlogged confounding, shared site/task attribution, interval defects and a
singleton-sort versus set-order instruction conflict. Round 7 records the
full scores, denominators, frozen baseline provenance and evidence limits.
The first measured revision (`19d6ef8`, Round 8) retains the four D1 marks in
both repeats (recall 5/6 and 6/6), improves one D2 run to 5/7 recall and 88.5%
top-three share, and one D3 run to 6/7 recall and 100% share. D2 still misses
the recall mark and its intervals remain poorly calibrated; D3 still makes a
false recording claim. These are development measurements, with one revised
repeat per harder world, not a completed experiment-condition or held-out result.
The refusal-trace reader tolerates other Codex event shapes while preserving
Claude refusal recognition, so a string status message cannot abort scoring.
See `skills/ROUNDS.md` for every run and its evidence limits.

Settled scoring integration (`94432fa`) rescored these unchanged submissions
with fresh oracle reports and the final-wording simple baselines (`7eea162`),
alongside frozen classical `b022bbf`. Original D1 marks and Round 7/8 numeric
recall, ranking and recovery are unchanged. D2/D3 recall and precision are
explicitly provisional because stored cause units disagree with one fault's
detectability flag. Precision now also reports vague causes as false positives.
D1's overall baseline margin was unavailable in that evaluator round: the
first video repeat timed out and its second was not substituted. Revised D2 has a 16.369-point
margin and ratio 2.4865 over the strongest required baseline; revised D3 ties it.
These remain old-world development comparisons. Exact three-fix chance bounds
have upper 95th percentiles 3.274/2.381/3.274 points on D1/D2/D3; missing random
sets are bounded, not silently omitted. Fresh records and reproduction scripts
are under `~/.cache/robots-probes/skills/20261009-settled-94432fa`.

Round 9 (`d94c147`) adds public recording-target guidance and bounded drift
scenarios. D1's new repeat finds 6/6 causes, rejects the decoy, reaches gain
correlation 0.9012 and recovers all achievable top-three gain. D2 repeats find
6/7, with correlations 0.7135 and 0.3650 despite identical 88.5% top-three
recovery; D3 finds 6/7, correlation 0.9019 and 100% recovery. D2/D3 diagnosis
remains provisional. Strict precision is 6/6, 6/6, 6/6 and 6/7 respectively.
All gains use 336 scoring attempts per world. The scorer at `6e9fd8d`
also preserves provisional cause warnings in pooled reports; fresh unchanged
answer scores are in `~/.cache/robots-probes/skills/20261009-final-6e9fd8d`.
The trace review found arbitrary equal attribution between confounded
alternatives, unsupported partial repair assumptions and overconfident drift
forecasts. Round 12 (`f8c203d`) exercised general repair semantics: its D2
correlations were 0.7278/0.7820, with three-fix shares 0.8846/1. Round 13
(`1ce4a8b`) added conditional repair arithmetic and future-window bounds.
All three agents used the new helper. D1 reached recall 5/6, correlation
0.8977, three-fix share 1 and interval coverage 11/11. D2 repeats reached
6/7 each, correlations 0.8182/0.6125 and three-fix shares 1/0.8846. This is
mixed improvement: the helper does not identify causal weights or validate
transport of response rates. The final combined policy-prior, cost-order and
unsupported-target revision (`67268cd`) is measured on two fresh D2 repeats:
recall 6/7 and 5/7, correlations 0.9574 and 0.8141, three-fix shares 0.8846
and 0.5192. Both reject the decoy. The first reaches optimal four/eight-hour
cuts; the second has strict precision 5/5 but omits a useful latent-condition
repair from its ranking. Thus improved precision/correlation does not establish
better repair selection. The general follow-up is open.
Independent skill review remains requested from the orchestrator.

The reviewed scorer `84bfa91` binds baseline inputs and outputs to producer
records and the scored world; pooling requires a file roster with world hashes.
Its fresh reports in `~/.cache/robots-probes/skills/20261009-final-84bfa91`
retain the same diagnosis, ranking and recovery values for 25 preserved records
(22 completed logs answers and three ended no-answer records). The earliest
transport failure is refused for missing repeat metadata, without rewriting
its record. The baseline owner's primary video rerun (`6b3a75a`, manifest
`baselines-v4`) now supplies D1's missing method: its original two agents tie
classical statistics, zero-point margin and ratio 1 at three fixes. The older
timeout and incomplete-comparison reports remain preserved. Producer output
hashes are not attested by those old manifests; directory/input bindings and
new scored-file hashes do not retroactively create that attestation.

Both initial experiment continuations were refused at startup for insufficient
available memory, before policy load or any episode. Their records remain.
Round 10 continues the same saved first sessions into new directories using
four workers, with policy batch size and memory guards unchanged; these are
continuations, not new repeats. At 23:39 UTC both still await their first
simulation. No experiment-conditioned pass or benefit is established.

## Stage-2 miniature chain

The integration of stage 2: a world of 8 robots at 2 sites with 3 faults over 4
simulated days is generated, run with X-VLA, replayed for the oracle, exported,
diagnosed by one agent through the runtime and the skills, and scored against the
hidden fault table. Its files are the fixtures the stage-4 agents develop against:
`~/.cache/robots-fixtures/mini-v2/` (manifest `fixtures/mini-v2.json`), the export with
the tables `costs` and `catalogue` and the description read in parts, and the first
version `mini-v1` (manifest `fixtures/mini-v1.json`), kept unchanged for the tests that
pin it. The section "mini-v2" below says what changed; the sections after it describe
the mini-v1 analyses, whose runs mini-v2 reuses.

```sh
make mini                                   # the whole chain; resumes in MINI_DIR (w10-v2)
make mini MINI_DIR=DIR MINI_FLAGS="--only score"     # one step; --through STEP, --workers N, --model M
make dashboard RUNS="DIR/runs/w10-observed DIR/runs/w10-later" TRUTH=1
```

Code: `fleetlab.worlds` (the version-0 generator and the miniature fleet's spec),
`fleetlab.integration` (the chain: `chain.py`, `oracle.py`, `scoring_step.py`,
`fixtures.py`), `fleetlab.scoring` (the scorer, version 1). Tests:
`tests/test_worlds.py`, `tests/test_integration.py`, `tests/test_scoring.py`.

The chain's oracle now goes through `fleetlab.oracle`
(src/fleetlab/oracle/README.md): `integration.oracle.oracle_jobs` plans the same 130
replays with the same job keys, the table is read from `job_outcomes` (the same 234
rows), and the oracle step writes `oracle/oracle.json` as the package's report (every
gain with a bootstrap interval, the chance level, the detectability check), with
`summary.md` and `episodes.parquet` beside it, and `truth/oracle.parquet` into
`w10-later` as before. The scorer's valuation and random draws live in
`fleetlab.oracle.valuation`. A fresh run is replay-only, so it is proven by proof
replays against the as-run chains (`fleetlab.oracle.proof`; the engine's chain sample
replays run jobs only). The files described below for mini-v1 and mini-v2
were made before; their numbers are unchanged.

## mini-v2: the export with the catalogue table (2026-10-08)

Contracts batch 9 moved the action catalogue out of the system description into the
visible table `catalogue` and made `describe` return one part at a time under a
30,000-character cap (Claude Code refused the 93 kB description, so the
mini-v1 analyses never saw it). World w10 was re-exported from the same runs, no GPU:

- **World.** Regenerated on main: spec, model, manifest, fault table and health readings
  byte-identical to mini-v1's; only `description.json` changed (costs and the number of
  catalogue entries in place of the catalogue), so the world hash is now
  `4729eb7880fbeb339e808df56d057283ecd808a94c8054a96eeee05caa9c4dae`.
- **Runs.** `runs/w10-observed` and `runs/w10-later` copied from the mini-v1 root (the
  episodes' identities are unchanged, and the runs were shown to reproduce byte for
  byte); their records rewritten (description, change and maintenance logs, the new
  `catalogue` table, health). The chain now counts a run whose jobs are all done and
  merged as complete even when a newer log version adds a run-level table, so that the
  records step, not the GPU, fills it in.
- **Oracle.** Unchanged: 234 rows, the same gains.
- **Export.** `export/`: the same 208 episodes; besides `data_dictionary.md` and
  `description.json` (20 kB), the only new file is `log/catalogue.parquet` (355 actions
  with their hours); `costs.parquet` was there already. Every other file is
  byte-identical to mini-v1's export. Leak audit clean.
- **Analysis.** One logs-only analysis with opus on the new export
  (`analyses/w10-20261008T022844-b93170`, one session of 213 s, 34 tool calls, 733,081
  input tokens of which 633,927 cached, 26,504 output, list price 1.45 USD). The tool
  log shows the description read and delivered: one `describe` call with
  `part: "all"`, 19,371 characters (every part fits under the cap at this world's
  size, so the agent took them in one call rather than part by part), and no result
  refused in the CLI's stream; and the catalogue queried as a table (call 23,
  `SELECT verb, scope, component, config_key, condition_key, hours FROM catalogue
  WHERE ...`). With the task classes in view it proposed `route_to_teleop` for the
  milk task, which the mini-v1 analyses could not name.
- **The experiments condition** waits for the GPU (the soak holds it until about
  08:00); mini-v2 holds the logs-only analysis only.
- **Fixtures.** `~/.cache/robots-fixtures/mini-v2/`: 3,128 files, 281 MB (world, runs,
  export, oracle, the analysis with its agent run, score, step records); manifest
  `fixtures/mini-v2.json`, `verify('mini-v2')` gives `[]`. The mini-v1 tree was not
  touched and still matches its manifest.

The score of the new analysis next to the earlier logs-only ones (same world, runs and
oracle; mini-v1 is the scored analysis of that version, repeat 2 the one from an empty
directory):

| Logs only | mini-v1 | mini-v1, repeat 2 | **mini-v2** |
|---|---|---|---|
| Saw the system description | no (refused) | no (refused) | **yes** |
| Claim 1: recall / precision / decoy rejection | 1.00 / 0.50 / 1.00 | 1.00 / 0.67 / 1.00 | **1.00 / 0.67 / 1.00** |
| Claim 1: named / hits / false positives | 4 / 2 / 2 | 3 / 2 / 1 | **3 / 2 / 1** (r08's calibration record at 0.10) |
| Claim 2: Spearman / mean absolute error (points) | 0.89 / 1.37 | 0.87 / 4.23 | **0.96 / 1.83** |
| Claim 2: slope / true gain inside the interval | 1.17 / 0.91 | 1.80 / 0.67 | **1.18 / 0.88** |
| Claim 3: share of the achievable gain at k = 3, 6, 4 h, 8 h | 1.00 at each | 1.00 at each | **1.00 at each** (+46.15 points) |
| Traceability: traced / untraced / small | 65 / 0 / 26 | 64 / 0 / 24 | **56 / 0 / 20** (22 of 34 calls searched) |
| Tool calls; list price | 42; 1.42 USD | 32; 1.13 USD | **34; 1.45 USD** |

The baselines are unchanged (random fixes at k = 3: +1.09 points, 95th percentile +7.40;
no-data ranker 0). Seeing the description changed the answer little on this world: the
two causes, their order and the fix sets were already right from the tables; the
expected gains came from the same `gains` helper (34.07 points for the delay against
39.42 true, 7.81 for the camera against 6.73). The visible differences are the
`route_to_teleop` entry and one false positive fewer than the scored mini-v1 analysis,
within the spread of the mini-v1 repeats.

## The score (mini-v1)

| Metric | logs | experiments |
|---|---|---|
| Claim 1: recall of the 2 detectable causes | 1.00 | 1.00 |
| Claim 1: precision among the named causes | 0.50 | 1.00 |
| Claim 1: decoy rejection | 1.00 | 1.00 |
| Claim 1: named / hits / false positives | 4 / 2 / 2 | 2 / 2 / 0 |
| Claim 2: Spearman, expected against true gain | 0.89 | 0.74 |
| Claim 2: mean absolute error (points) | 1.37 | 0.97 |
| Claim 2: slope of true on expected | 1.17 | 1.08 |
| Claim 2: true gain inside the agent's interval | 0.91 | 0.90 |
| Claim 3 at k = 3: agent's net points (share of achievable), hours | +46.15 (1.00), 0.7 h | +46.15 (1.00), 0.7 h |
| Claim 3 at k = 6: agent's net points (share of achievable), hours | +46.15 (1.00), 4.3 h | +46.15 (1.00), 1.4 h |
| Claim 3 at 4 h: agent's net points (share of achievable), hours | +46.15 (1.00), 2.25 h | +46.15 (1.00), 2.25 h |
| Claim 3 at 8 h: agent's net points (share of achievable), hours | +46.15 (1.00), 5.25 h | +46.15 (1.00), 5.25 h |
| Traceability: traced / echo / untraced / small (calls searched) | 65 / 0 / 0 / 26 (28 of 42) | 86 / 0 / 0 / 18 (33 of 58) |
| Fresh replays the score waits for | 0 | 0 |

Baselines (the same for both analyses):

| Budget | Achievable (points) | Random fixes: mean, 5th to 95th percentile (points); valued / skipped draws | No-data ranker (points) |
|---|---|---|---|
| k = 3 | +46.15 | +1.09, +0.00 to +7.40; 194 / 6 | +0.00 |
| k = 6 | +46.15 | +1.86, +0.00 to +9.62; 188 / 12 | +6.73 |
| 4 h | +46.15 | +5.71, -0.96 to +39.42; 169 / 31 | +6.73 |
| 8 h | +46.15 | +7.59, -0.96 to +39.42; 159 / 41 | +6.73 |

How to read it. Both analyses found both detectable causes with an accepted direction
and left the decoy alone, and both fix sets reach the oracle's best at every budget:
reverting the delay at s2 and recalibrating r04's scene camera are the only two
actions that change the scoring window, and they lead both lists. The difference
between the conditions is in precision and calibration. Named means a confidence of at
least 0.5 or a place in the top n + 2 = 4 by confidence, so the logs-only agent's two
low-confidence alternatives (r08's same-day calibration record at 0.1, s2's object
batch at 0.2) count as named false positives; the agent with experiments tested the
r08 record (x0002: 14 of 40 reverted against 15 of 40) and dropped it. Its expected
gains come from its own A/B tests (36.25 points for the delay, 5.62 for the camera,
against true 39.42 and 6.73). The Spearman correlation is over all ranked actions,
most of them at zero expected and zero true gain, so the ties carry most of it; it is
lower with experiments because two zero-gain actions there carry small negative
expectations. The random draws at the hour budgets cut long lists of cheap 0.1-hour
`set` and `revert` actions, so their 95th percentile reaches the best single fix. The
headline's best baseline, the trivial no-data ranker, recovers nothing at k = 3, so
the report gives no ratio. Neither agent saw the system description: `describe()` is
too large for Claude Code's MCP limit, so both worked from the tables;
the report's notes say so and those two calls are not traceability sources.

## The agent's answer, verbatim

### mini-v2, logs only: analysis `w10-20261008T022844-b93170`

`outbox/diagnosis.json`:

```json
{
 "causes": [
  {"target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "direction": "misconfigured", "confidence": 0.85, "evidence": [{"episode_id": null, "step": null, "table": "change_log", "query": "-- helper: changes (scope skill block, params days_around 2, from_day 0, to_day 1): entry control.action_delay_steps at s2 day 1, 0 -> 17, rate_before 0.942, rate_after 0.269, did -0.624 (-0.795, -0.453)"}, {"episode_id": null, "step": null, "table": "episodes", "query": "SELECT robot_id, site_id, day, count(*) n, sum(CAST(outcome='success' AS INT)) ok, sum(CAST(outcome='handed_over' AS INT)) ho FROM episodes GROUP BY ALL ORDER BY robot_id, day"}, {"episode_id": "e000242", "step": 126, "table": null, "query": null}, {"episode_id": "e000006", "step": 127, "table": null, "query": null}], "failure_mode": "carry/place: late, hesitant motion, object picked up late and not placed before the episode times out"},
  {"target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "direction": "degraded", "confidence": 0.75, "evidence": [{"episode_id": null, "step": null, "table": "health", "query": "SELECT * FROM health ORDER BY robot_id, component, metric, day"}, {"episode_id": null, "step": null, "table": "episodes", "query": "-- helper: gains (sensitivity-from-logs block, last_day 1): 'recalibrate r04 scene_camera', r04, from_day 0: rate 0.346, expected_rate 0.971, gap 0.625, within_site_gap 0.654"}, {"episode_id": "e000409", "step": 0, "table": null, "query": null}, {"episode_id": "e000409", "step": 30, "table": null, "query": null}], "failure_mode": "grasp/place: r04 misses or mis-handles the object on both days, scene view shifted"},
  {"target": {"scope": "r08", "component": null, "config_key": "camera.scene.calibration", "condition_key": null}, "direction": "changed", "confidence": 0.1, "evidence": [{"episode_id": null, "step": null, "table": "change_log", "query": "-- helper: changes (scope skill block): entry camera.scene.calibration at r08 day 1, did -0.502, same_day_overlapping control.action_delay_steps at s2"}, {"episode_id": "e000367", "step": 0, "table": null, "query": null}], "failure_mode": "carry/place: late, hesitant motion, object picked up late and not placed before the episode times out"}
 ],
 "mode_summaries": [
  "carry/place: late, hesitant motion, object picked up late and not placed before the episode times out; fingerprint 'released, not placed' with the grasp around step 98-127 instead of about 50 and the episode running to 280 steps; 38 failures at s2 on day 1, all four robots (3 to 5 successes of 13 each, against 12 to 13 on day 0); cause: control.action_delay_steps set from 0 to 17 at s2 on day 1.",
  "grasp/place: r04 misses or mis-handles the object on both days, scene view shifted; fingerprints 'closed on nothing' (4, all on r04), 'never closed' (1, r04) and 'released, not placed' (12 on r04); 17 failures of 26 episodes on r04, present from day 0; cause: r04 scene camera pose (calibration_residual 69.26 then 98.15, the highest in the fleet).",
  "milk: the policy fails 'pick up the milk and place it in the basket' in 9 of 10 episodes, including day 0 on robots r05, r06 and r07 that were otherwise healthy; no cause assigned, a weakness of the policy itself.",
  "harmless changes: policy.sampling_steps 10 to 5 at s1 (r01 to r03 stay at 13 of 13 on day 1), camera.scene.calibration at r08 (r08 no worse than the other s2 robots, view unchanged), software.version 4.2.1 and the maintenance entries show no loss of their own."
 ],
 "warnings": [
  {"scope": "r04", "metric": "scene_camera.calibration_residual", "expected_day": 2, "confidence": 0.4}
 ],
 "retrain": [
  {"failure_mode": "milk: the carton is grasped and then released or lost before it reaches the basket", "episode_ids": ["e000056", "e000282", "e000171"], "why_no_action_fixes_it": "It occurs on day 0 on three s2 robots before any change and with no component fault, so no repair or setting reaches it.", "data_wanted": "More demonstrations of picking up the milk carton and carrying it to the basket, from varied start positions."}
 ]
}
```

`outbox/fix_set.json`:

```json
{
 "actions": [
  {"action": {"verb": "revert", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": null}, "expected_gain": 34.07, "uncertainty": {"low": 27.21, "high": 40.94}, "rationale": "Removes the day-1 change of the command delay from 0 to 17 at s2, which lines up with the collapse of all four s2 robots; gain is gain_points_change of the gains block. Largest and cheapest, so first."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 7.81, "uncertainty": {"low": 5.25, "high": 9.88}, "rationale": "r04 fails on both days with a visibly shifted scene view and the highest calibration residual in the fleet; gain from the gains block (gap 0.625 on an eighth of the work). Independent of the s2 fix, so second."},
  {"action": {"verb": "set", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": 0}, "expected_gain": 34.07, "uncertainty": {"low": 27.21, "high": 40.94}, "rationale": "Same cause as the revert at rank 1, by setting the delay back to its old value; on its own it wins the same, but adds nothing once the revert is done."},
  {"action": {"verb": "clean", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.78, "uncertainty": {"low": 0.0, "high": 0.99}, "rationale": "Alternative for r04's scene camera if the fault were dirt rather than pose; the images look sharp, so a tenth of the r04 gap at most (gains block, attributable 0.1)."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "wrist_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.78, "uncertainty": {"low": 0.0, "high": 0.99}, "rationale": "Next suspect on r04: its wrist camera residual is unremarkable against the fleet, so a tenth of the r04 gap at most (gains block, attributable 0.1)."},
  {"action": {"verb": "replace", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 7.03, "uncertainty": {"low": 4.73, "high": 8.89}, "rationale": "Stronger verb on the same r04 camera; fixes the same failures as the recalibration at rank 2 at eight hours, so it adds little after it and stands low."},
  {"action": {"verb": "revert", "target": {"scope": "r08", "component": null, "config_key": "camera.scene.calibration", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 1.29}, "rationale": "Confounded in time with the s2 delay change, but the change is tiny, r08 is no worse than its site peers and its view is normal; judged harmless."},
  {"action": {"verb": "revert", "target": {"scope": "s1", "component": null, "config_key": "policy.sampling_steps", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.99}, "rationale": "The day-1 change to 5 sampling steps at s1 moved nothing: r01 to r03 stayed at 13 of 13; a small upside only if it worsens r04."},
  {"action": {"verb": "route_to_teleop", "target": {"scope": "fleet", "component": null, "config_key": null, "condition_key": null}, "value": "pick up the milk and place it in the basket"}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The milk task fails 9 of 10 times, also on healthy robots; routing it away does not change what the robots do, so no gain in fleet success."},
  {"action": {"verb": "request_training_data", "target": {"scope": "fleet", "component": null, "config_key": null, "condition_key": null}, "value": "milk carton: grasped then released before reaching the basket"}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The policy's own weakness on the milk task; no effect within the scoring window."}
 ]
}
```

### mini-v1, logs only: analysis `w10-20261008T004146-af1148`

`outbox/diagnosis.json`:

```json
{
 "causes": [
  {"target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "direction": "changed", "confidence": 0.8, "evidence": [{"episode_id": null, "step": null, "table": "change_log", "query": "SELECT * FROM change_log ORDER BY day"}, {"episode_id": null, "step": null, "table": "episodes", "query": "SELECT site_id, robot_id, day, count(*) n, sum(CAST(outcome='success' AS INT)) ok, round(avg(steps),1) steps, count(video_scene) vids FROM episodes GROUP BY ALL ORDER BY 1,2,3"}, {"episode_id": "e000240", "step": 90, "table": null, "query": null}, {"episode_id": "e000019", "step": 279, "table": null, "query": null}], "failure_mode": "place: late, slow motion after the delay change, released but not placed, timeout at 280 steps (site s2, day 1)"},
  {"target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "direction": "degraded", "confidence": 0.7, "evidence": [{"episode_id": null, "step": null, "table": "health", "query": "SELECT robot_id, day, component, metric, round(value,2) v FROM health WHERE component IN ('scene_camera','wrist_camera') OR metric NOT IN ('calibration_residual','grip_force','cycle_count','joint_torque_rms','cpu_load') ORDER BY component, metric, robot_id, day"}, {"episode_id": null, "step": null, "table": "episodes", "query": "SELECT site_id, robot_id, day, count(*) n, sum(CAST(outcome='success' AS INT)) ok, round(avg(steps),1) steps, count(video_scene) vids FROM episodes GROUP BY ALL ORDER BY 1,2,3"}, {"episode_id": "e000409", "step": 32, "table": null, "query": null}, {"episode_id": "e000075", "step": 52, "table": null, "query": null}], "failure_mode": "grasp: gripper closes beside the object and retries until timeout (r04, both days)"},
  {"target": {"scope": "r08", "component": null, "config_key": "camera.scene.calibration", "condition_key": null}, "direction": "changed", "confidence": 0.1, "evidence": [{"episode_id": null, "step": null, "table": "change_log", "query": "SELECT * FROM change_log ORDER BY day"}, {"episode_id": "e000367", "step": 0, "table": null, "query": null}], "failure_mode": "place: late, slow motion after the delay change, released but not placed, timeout at 280 steps (site s2, day 1)"},
  {"target": {"scope": "s2", "component": null, "config_key": null, "condition_key": "object_batch"}, "direction": "changed", "confidence": 0.2, "evidence": [{"episode_id": null, "step": null, "table": "episodes", "query": "SELECT episode_id, robot_id, day, outcome, steps FROM episodes WHERE task_text='pick up the milk and place it in the basket' ORDER BY day, robot_id"}, {"episode_id": "e000282", "step": 66, "table": null, "query": null}], "failure_mode": "milk into basket: fails on otherwise healthy robots on day 0"}
 ],
 "mode_summaries": [
  "place: late, slow motion, released but not placed, timeout at 280 steps: site s2 on day 1, all four robots (s2 success 0.942 on day 0, 0.269 on day 1; signature 'released, not placed', s2 41/104); cause: control.action_delay_steps changed 0 -> 17 at s2 on day 1. The r08 scene calibration change of the same day is confounded but r08 is no worse than its site-mates, so it is judged harmless.",
  "grasp: gripper closes beside the object and retries until timeout: r04 on both days (17/26 failures, rate 0.346 against 0.971 expected); the scene image is displaced against other robots on the same task and the scene camera calibration residual is the fleet's highest (69.26 then 98.15); cause: r04 scene camera pose.",
  "milk into basket: 1 success in 10, including 3 failures on day 0 on s2 robots with no other cause; policy weakness on this object or the s2 object batch; no cause confirmed.",
  "No effect found for policy.sampling_steps 10 -> 5 at s1 (r01-r03 stay at or near 1.0), the fleet software version 4.2.1, or the maintenance entries."
 ],
 "warnings": [],
 "retrain": [
  {"failure_mode": "milk into basket: fails on otherwise healthy robots on day 0", "episode_ids": ["e000056", "e000282", "e000171"], "why_no_action_fixes_it": "These failures occur on s2 robots on day 0, before the delay change and with no component or configuration fault on those robots, so no repair or setting reaches them.", "data_wanted": "More demonstrations of grasping and carrying the milk carton into the basket, across start positions and neighbouring objects."}
 ]
}
```

`outbox/fix_set.json`:

```json
{
 "actions": [
  {"action": {"verb": "revert", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": null}, "expected_gain": 34.07, "uncertainty": {"low": 27.21, "high": 40.94}, "rationale": "Undoes the day-1 change 0 -> 17 at s2, after which all four s2 robots fell from 0.942 to 0.269; gain is gain_points_change of the gains block. Largest loss, cheapest action, so first."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 7.81, "uncertainty": {"low": 5.25, "high": 9.88}, "rationale": "r04 misses grasps on both days (0.346 against 0.971 expected), its scene view is displaced and its calibration residual is the fleet's highest; recalibration removes a pose error. Gain from the gains block; independent of action 1."},
  {"action": {"verb": "revert", "target": {"scope": "r08", "component": null, "config_key": "camera.scene.calibration", "condition_key": null}, "value": null}, "expected_gain": 0.47, "uncertainty": {"low": 0.0, "high": 4.75}, "rationale": "Confounded with the s2 delay change on day 1, but r08 is no worse than its site-mates (within-site gap -0.067) and the calibration moved only slightly; small expected gain, high end if it alone were the cause."},
  {"action": {"verb": "set", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": 0}, "expected_gain": 34.07, "uncertainty": {"low": 27.21, "high": 40.94}, "rationale": "Same fix as action 1 by setting the old value; on its own it wins the same, but it adds nothing once action 1 is taken, so it stands below."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "wrist_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.39, "uncertainty": {"low": 0.0, "high": 0.49}, "rationale": "Alternative for r04's missed grasps if the error is in the wrist view; wrist residual is unremarkable, so a small share of r04's gap."},
  {"action": {"verb": "service", "target": {"scope": "r04", "component": "gripper", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.39, "uncertainty": {"low": 0.0, "high": 0.49}, "rationale": "Next suspect on r04; no drops and normal grip force, so a small share of its gap; three hours."},
  {"action": {"verb": "clean", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "r04's image is sharp but displaced; cleaning does nothing for a pose error."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r08", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 4.75}, "rationale": "Overlaps action 3; r08's view looks normal on day 1, so no gain expected."},
  {"action": {"verb": "revert", "target": {"scope": "s1", "component": null, "config_key": "policy.sampling_steps", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The 10 -> 5 change at s1 moved nothing: r01-r03 stay at or near 1.0 on day 1."},
  {"action": {"verb": "revert", "target": {"scope": "fleet", "component": null, "config_key": "software.version", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "Version 4.2.1 was in force on day 0 when seven of eight robots ran at 0.92 or above; not a cause."},
  {"action": {"verb": "request_training_data", "target": {"scope": "fleet", "component": null, "config_key": null, "condition_key": null}, "value": "milk into basket: fails on otherwise healthy robots"}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "Milk task succeeds 1 in 10, including on healthy robots on day 0; no change in the scoring window."},
  {"action": {"verb": "replace", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 7.81, "uncertainty": {"low": 5.25, "high": 9.88}, "rationale": "Would also fix r04's camera pose but overlaps action 2 at sixteen times the hours; last so it never crowds out cheaper actions."}
 ]
}
```

### mini-v1, with experiments: analysis `w10-20261008T004200-364749`

`outbox/diagnosis.json`:

```json
{
 "causes": [
  {"target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "direction": "misconfigured", "confidence": 0.92, "evidence": [{"episode_id": null, "step": null, "table": "episodes", "query": "SELECT site_id, day, count(*) AS n, sum(CAST(outcome = 'success' AS INT)) AS ok FROM episodes WHERE experiment_id IS NULL GROUP BY site_id, day ORDER BY site_id, day"}, {"episode_id": null, "step": null, "table": "episodes", "query": "SELECT (CAST(substr(episode_id, 2) AS INT) - 417) % 2 AS arm, count(*) AS n, sum(CAST(outcome = 'success' AS INT)) AS ok FROM episodes WHERE experiment_id = 'x0001' GROUP BY 1 ORDER BY 1"}, {"episode_id": "e000196", "step": 100, "table": null, "query": null}, {"episode_id": "e000240", "step": 100, "table": null, "query": null}], "failure_mode": "delayed control: late gripper close, reversals and repeated grasp/release, object released but not placed"},
  {"target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "direction": "degraded", "confidence": 0.9, "evidence": [{"episode_id": null, "step": null, "table": "episodes", "query": "SELECT robot_id, day, count(*) AS n, sum(CAST(outcome = 'success' AS INT)) AS ok FROM episodes WHERE experiment_id IS NULL AND site_id = 's1' GROUP BY robot_id, day ORDER BY robot_id, day"}, {"episode_id": null, "step": null, "table": "episodes", "query": "SELECT (CAST(substr(episode_id, 2) AS INT) - 617) % 3 AS arm, count(*) AS n, sum(CAST(outcome = 'success' AS INT)) AS ok FROM episodes WHERE experiment_id = 'x0003' GROUP BY 1 ORDER BY 1"}, {"episode_id": "e000409", "step": null, "table": null, "query": null}, {"episode_id": "e000075", "step": null, "table": null, "query": null}], "failure_mode": "shifted scene view: repeated regrasps on r04, object released but not placed"}
 ],
 "mode_summaries": [
  "delayed control: late gripper close, reversals and repeated grasp/release, object released but not placed; all four s2 robots (r05-r08) from day 1, when s2 fell from 49 of 52 to 14 of 52 successes; cause: control.action_delay_steps set from 0 to 17 at s2 on day 1; experiment x0001 reverting it gave 40 of 40 against 11 of 40 in the control arm.",
  "shifted scene view: repeated regrasps on r04, object released but not placed; r04 on both days (4 of 13 and 5 of 13 successes) while r01-r03 are near perfect; cause: r04 scene camera pose/calibration degraded (calibration residual far above the other robots); experiment x0003: recalibrate gave 40 of 40 against 22 of 40 in control, cleaning did not help (19 of 40).",
  "milk into basket failures: the milk task failed on s2 on day 0 before the delay change (e000056, e000171, e000282), but it succeeded in every treated experiment arm, so no separate cause is assigned; a few episodes only, treated as noise or an unconfirmed policy weakness.",
  "harmless changes: s1 policy.sampling_steps 10 to 5 on day 1 (r01-r03 stayed at 13 of 13) and the r08 scene camera calibration update on day 1 (experiment x0002: reverting it gave 14 of 40 against 15 of 40 in control, while reverting the s2 delay on r08 gave 39 of 40); neither is a cause."
 ],
 "warnings": [],
 "retrain": []
}
```

`outbox/fix_set.json`:

```json
{
 "actions": [
  {"action": {"verb": "revert", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": null}, "expected_gain": 36.25, "uncertainty": {"low": 27.42, "high": 41.95}, "rationale": "Removes the day-1 change of the action delay at s2 (0 to 17) that broke all four s2 robots. Measured in experiment x0001 (readout, share 0.5): 40 of 40 against 11 of 40 in control. Largest gain, cheapest action, so first."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 5.62, "uncertainty": {"low": 3.53, "high": 7.52}, "rationale": "Restores the scene camera pose of r04, which fails on both days. Measured in experiment x0003 (readout, share 0.125): 40 of 40 against 22 of 40 in control. Independent of the s2 cause, so it adds fully to rank 1."},
  {"action": {"verb": "set", "target": {"scope": "s2", "component": null, "config_key": "control.action_delay_steps", "condition_key": null}, "value": 0}, "expected_gain": 36.25, "uncertainty": {"low": 27.42, "high": 41.95}, "rationale": "Same fix as rank 1 by setting the old value 0; on its own it wins the same as the revert (x0001), but it adds nothing once the revert is applied, so it stands below the two independent fixes."},
  {"action": {"verb": "revert", "target": {"scope": "s1", "component": null, "config_key": "policy.sampling_steps", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The s1 sampling steps change on day 1 is harmless: r01-r03 stayed at 13 of 13 afterwards. No gain expected; listed as the next suspect."},
  {"action": {"verb": "revert", "target": {"scope": "fleet", "component": null, "config_key": "software.version", "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The fleet software update on day 0 shows no effect: healthy robots and the treated experiment arms succeed almost always. No gain expected."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r04", "component": "wrist_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "Alternative camera on r04; the scene camera recalibration alone brought r04 to 40 of 40 in x0003, so nothing is left for the wrist camera."},
  {"action": {"verb": "recalibrate", "target": {"scope": "r08", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": -2.83, "high": 2.24}, "rationale": "r08's scene calibration update on day 1 is harmless (x0002: reverting it changed nothing, the delay revert alone gave 39 of 40), so recalibrating wins nothing; interval taken from the x0002 revert arm."},
  {"action": {"verb": "service", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 5.62}, "rationale": "Stronger, costlier verb on the r04 scene camera; untested, and the fault is a pose error that recalibration already removes, so no gain is claimed beyond rank 2."},
  {"action": {"verb": "request_training_data", "target": {"scope": "fleet", "component": null, "config_key": null, "condition_key": null}, "value": "milk into basket: released, not placed"}, "expected_gain": 0.0, "uncertainty": {"low": 0.0, "high": 0.0}, "rationale": "The milk task failed on s2 before the delay change, but succeeded in the treated arms; unconfirmed weakness. Does not change the scoring window, so zero gain."},
  {"action": {"verb": "revert", "target": {"scope": "r08", "component": null, "config_key": "camera.scene.calibration", "condition_key": null}, "value": null}, "expected_gain": -0.31, "uncertainty": {"low": -2.83, "high": 2.24}, "rationale": "Tested in x0002 (readout, share 0.125): 14 of 40 against 15 of 40 in control, no effect; the calibration change is not a cause."},
  {"action": {"verb": "clean", "target": {"scope": "r04", "component": "scene_camera", "config_key": null, "condition_key": null}, "value": null}, "expected_gain": -0.94, "uncertainty": {"low": -3.49, "high": 1.73}, "rationale": "Tested in x0003 (readout, share 0.125): 19 of 40 against 22 of 40 in control; cleaning does not fix a pose error, so it stands last."}
 ]
}
```


## What each step produced

All under the chain's root, `~/.cache/robots-probes/mini/w10/` (copied to the fixtures).

| Step | What | Produced |
|---|---|---|
| world | `fleetlab.worlds.mini.mini_spec()` and `generate(spec, software)` | `world/`: `spec.json`, `model.json` (truth, with traits), `description.json` (as the agents receive it), `manifest.jsonl` (416 run jobs), `hash` `31633759e0bf63ba...`; truth-side beside the hash: `fault_table.json`, `health.jsonl`, `task_texts.json`, `world_truth.json` (the engine's `WorldTruth`), `generator.json` |
| run | `fleet.run` twice, one heavy job (made in the first pass, below) | `runs/w10-observed`: days 0 and 1, 208 episodes with video, 21 chain-sample replays, `chain_ok` true; `runs/w10-later`: days 2 and 3 as run (208 episodes) and the 130 oracle replays of day 3, 21 chain-sample replays, `chain_ok` true; the fleet's records (description, change log, maintenance, health) written into each run |
| oracle | `integration.oracle` | 234 rows over the 104 scoring episodes (104 as run, 65 power-set replays, 65 decoy-confirmation replays; the chain sample's unmodified replays agree with their originals); `truth/oracle.parquet` in `w10-later`; `oracle/oracle.json` |
| export | `export_runs([w10-observed], fleet=..., task_classes=...)` | `export/`: the visible tree of 208 episodes (36,976 step rows, 416 videos) and `data_dictionary.md`; leak audit clean |
| analysis | `Batch.after_gate` then `run_analysis`, opus, eight skills | two analyses, both opus (claude-opus-5-5), the eight analysis skills: **logs** `analyses/w10-20261008T004146-af1148`, one session, done, 42 tool calls, without the two experiment tools; **experiments** `analyses/w10-20261008T004200-364749`, session 1 "waiting" with 3 requests (x0001 to x0003, 40 episodes per arm, paired starts), the engine's queue ran 320 arm episodes in `world/experiments/<id>/session-01` (chain sample held) and built `world/exports/<id>/session-01`, session 2 "done", 22 tool calls; each `outbox/` holds `diagnosis.json` and `fix_set.json`, each session's CLI call its `box/toolbox/tool_calls.jsonl` under `agent-runs/` |
| fresh | `scoring.replays_needed_for` | none needed: every valuation is a lookup (the agent's `set` puts back the nominal value) |
| score | `scoring.score` per analysis | `score/<condition>/report.json`, `report.md` |
| fixtures | `integration.fixtures.copy_fixtures` | `~/.cache/robots-fixtures/mini-v1/`: 6,429 files, 666 MB (world with the experiment runs and the session-2 export, runs, export, oracle, analyses, agent runs, scores, step records); `fixtures/mini-v1.json` in the repository: every file's size and SHA-256 prefix, a directory of more than 40 files as one entry, and a digest over the whole list; `python -c "from fleetlab.integration.fixtures import verify; print(verify('mini-v1'))"` gives `[]` |
| dashboard | `make dashboard ... TRUTH=1` | run pages `w10-observed` and `w10-later`, with the truth side; the progress page |

### The world

| | |
|---|---|
| World | `w10`, seed 10, kind `miniature` in `dashboard/worlds.json` |
| Fleet | s1: r01 to r04; s2: r05 to r08; five components each; two hidden traits per robot |
| Days and windows | 0 to 3: observed 0 and 1, experiment 2, scoring 3 |
| Episodes | 13 per robot and day, 416 as run; ids a seeded permutation of e000001 to e000416 |
| Tasks | the 20 of `libero_spatial` and `libero_object`; s1 leans 2:1 to spatial, s2 2:1 to object; start state 0 to 49 |
| f01 | `camera_pose_offset`, r04's scene camera turned 6 degrees, from day 0 (the s1 robot with the largest mounting tolerance); no Change |
| f02 | `action_latency`, `control.action_delay_steps` 0 to 17 at s2 from day 1, with its Change (day 1, s2, author `ops`) |
| f03 | decoy: `policy_knob`, `policy.sampling_steps` 10 to 5 at s1 from day 1, with its Change (day 1, s1, author `auto`) |
| Benign records | `software.version` 4.2.0 to 4.2.1 fleet-wide on day 0 (vendor); r08's `camera.scene.calibration` record nudged on day 1 (auto); maintenance: r02 controller service (day 0), r07 scene camera clean (day 0), r03 arm service (day 1) |

Fleet success by day and site (the fleet's check, as logged):

| Day | s1 | s2 | Fleet |
|---|---|---|---|
| 0 (observed) | 42/52 | 49/52 | 91/104 |
| 1 (observed) | 44/52 | 14/52 | 58/104 |
| 2 (experiment) | 42/52 | 14/52 | 56/104 |
| 3 (scoring) | 43/52 | 11/52 | 54/104 |

The observed window's fleet success is 71.6% (149 of 208), inside the design's
60 to 75% for a small world. r04 placed 9 of 26 in the observed window.

### The oracle (scoring window, 104 episodes)

| Item | Rescued | Broken | Net | Points |
|---|---|---|---|---|
| f01 (camera, r04) | 8 | 1 | 7 | 6.73 |
| f02 (delay, s2) | 41 | 0 | 41 | 39.42 |
| f03 (decoy, s1) | 0 | 1 | -1 | -0.96 |
| interaction f01, f03 | | | 1 | |
| achievable, k = 1 | 41 | 0 | 41 | 39.42 |
| achievable, k = 2 to 6 | 49 | 1 | 48 | 46.15 |

The generator expected 5.1 points for f01 and 22.5 for f02 over the scoring window,
from the calibrated curves on `libero_spatial`. The delay hurts much more on
`libero_object`, which s2 runs two thirds of the time: s2 placed 3 of 67 object
episodes and 22 of 37 spatial ones over days 2 and 3.

## The chain again, from an empty directory

`make mini MINI_DIR=~/.cache/robots-probes/mini/w10-check MINI_FLAGS="--through score"`
(one command; `--through score` so that the fixtures stay those of `w10`), started
01:02 from a frozen checkout at e5ab913: the world regenerated to the same hash
(`31633759...`; spec, model, description, manifest, fault table and health readings
byte-identical); the heavy job queued 22.8 minutes behind another agent's ladder run,
then ran both runs in 684 s, both chain samples held; **the visible trees of both runs
are byte-identical to those of the first runs (631 files each, the videos included),
and `python -m fleetlab.engine compare` finds all 229 and 359 jobs' hash chains
identical** (two runs of one world, 2 hours apart, in other processes and checkouts).
The same command then ran both analyses anew (a second repeat of each condition;
the experiments condition's queue job got the lock at once and ran its arms in 298 s)
and scored them, 48 minutes in all from the empty directory, 23 of them in the queue:

| Repeat 2 (`w10-check`) | logs | experiments |
|---|---|---|
| Claim 1: recall / precision / decoy rejection | 1.00 / 0.67 / 1.00 | 1.00 / 1.00 / 1.00 |
| Claim 1: named / hits / false positives | 3 / 2 / 1 (r08's calibration record at 0.15) | 2 / 2 / 0 |
| Claim 2: Spearman / mean absolute error (points) | 0.87 / 4.23 | 0.95 / 1.80 |
| Claim 3: share of the achievable gain at k = 3 | 1.00 | 1.00 |
| Traceability: traced / untraced | 64 / 0 | 76 / 0 |
| Sessions; list price | 1, 171 s; 1.13 USD | 2 (3 experiments), 188 s and 130 s; 1.89 USD |

The repeats agree on everything but the precision of the logs-only condition and the
calibration of the gains; the scored analyses are those of `w10`, which the fixtures
hold. `w10-check` stays in `~/.cache/robots-probes/mini/` and is not in the fixtures.

## Timings and queue

All times on 2026-10-07/08, Pacific time, with other agents' heavy and light jobs on
the machine (load 5 to 10).

| Step | Wall clock | Queue for the heavy lock | Notes |
|---|---|---|---|
| world | under 2 s | - | generation 0.8 s plus LIBERO's task texts |
| run (first pass, 23:22 to 23:33) | 678 s | 1.0 s | observed: 208 episodes and 21 chain-sample replays in 269 s, 3,039 per hour end to end; later: 338 jobs (208 episodes, 130 oracle replays) and 21 sample replays in 408 s, 3,199 per hour; policy load 12.6 s per run |
| records into the runs (light) | under 1 s | - | rewritten for this world's health model |
| oracle | under 2 s | - | 234 rows |
| export | 26 s | - | with the leak audit |
| analysis, logs | 234 s | - | the gate's records were current; one session of 209 s; 949,324 input tokens (858,268 cached), 26,055 output; list price 1.42 USD |
| analysis, experiments | 875 s | 102 s | session 1: 152 s, 1.08 USD; the queue job: 102 s waiting behind another agent's nightly world, then 468 s for 320 arm episodes (2,884 per hour) and the export; session 2: 131 s, 0.67 USD |
| fresh | under 1 s | - | no replay needed |
| score | 1.5 s | - | both analyses, 200 random draws each |
| fixtures | 4 s | - | 666 MB copied, manifest written |
| `make mini` over the finished root | 10.7 s | - | every step checks its inputs and skips what is done |

The two analyses ran in parallel. End to end, a chain from an empty directory needs
about 12 minutes of heavy work for the runs, 4 minutes for the logs-only analysis and
15 minutes for the experiments condition, plus whatever the heavy lock's queue costs:
in the first pass the experiments analysis waited 44 minutes behind the faults agent's
decoy confirmation and the nightly ladder before it was stopped (below).

## Choices, and where they depart from the brief

- **The decoy is a configuration change, not lighting or backdrop.** The coordinator
  later suggested lighting or backdrop; the runs had been made by then. I chose
  `policy.sampling_steps` 10 to 5 at s1 because it is calibrated flat (100 of 100
  against 98 of 100 healthy twins) and it writes a Change, so it is visible in the
  records the agent reads and decoy rejection tests something: a site-condition
  decoy writes no Change and shows only in the images. Its curve verdict is "kept"
  as a type (one image view breaks the policy), the point used here is flat.
- **Two runs, not one.** `export_runs` exports whole runs, so the observed window is
  its own run directory; the experiment and scoring windows as run and the oracle's
  replays share the second.
- **Decoy-confirmation replays.** The contract leaves decoys out of the power set.
  The oracle replays each scoring episode's fault lists once more with the decoy
  removed (65 replays): it confirms the decoy by replay and lets the scorer value an
  action on it by lookup. `achievable_gain` still excludes it, as the contract says.
- **Health readings and the fleet's records** are written into each run after the
  engine finishes (`RunWriter.write_model`, `write_table("health")`); the engine
  writes none of them.
- **The random baseline's catalogue** fills every value from the visible records:
  `set` writes the fleet-wide value of its key, `change_condition` keeps the site's
  recorded condition. Draws holding an action that adds a change (on this world the
  `set` of r08's scene calibration back to the fleet value) need a fresh replay and
  are skipped and counted, as the scorer reports per budget.
- **The no-data ranker is trivial** (a fixed verb and component prior, robots in id
  order); the agent baseline of the experimental design (section 6) replaces it.
- **Heavy steps run as their own processes under `tools/heavy-run`**, the runs, the
  fresh replays and each answer to the experiment queue, so the analysis does not
  hold the heavy lock while the model thinks.
- **The fixtures hold the agents' CLI run directories** (`agent-runs/`): the stream,
  the tool log, the prompts and `argv.json`, which records the host paths of the
  sandbox's mounts, the login file's path among them (the runtime's choice),
  never its content; the copy refuses any symbolic link.
- **The generator is version 0** (the stage-3 world generator replaces it): fixed rules for the
  benign entries and the health model, no drift, no confounder, no interaction, the
  calibrated curves of `libero_spatial` applied to both suites, every target taken as
  identifiable.

## The first pass, and what the review changed

The chain ran first in `~/.cache/robots-probes/mini/v1/` (kept there, not in the
fixtures). Its logs-only analysis found both causes (recall 1.00, precision 0.67,
decoy rejected, k = 3 share 1.00). An independent review of that pass found, among others,
that the cameras' calibration residual had a third of the noise the experimental design asks
for, so r04's readings stood apart from every other robot's on every day: a label,
not a clue, and the agent cited it. The fix changes `model.json` and
`description.json` and so the world's hash (ccabb72d... before, 31633759... now); the
manifest and spec are byte-identical, so the runs carried over and their records
were rewritten. The other changes after the review: "drifting" accepted only for a
fault with a drift schedule; configuration shadowing resolved against each day's
fleet; traceability reads only the data tools and only results the CLI showed the
model, and sets small whole numbers apart; an unanswered analysis is scored as
missing and a waiting one resumes. The first pass's experiments analysis was stopped
after its first session while it waited 44 minutes for the heavy lock, since its
export carried the old health readings; both analyses were then run anew.

## Findings filed

- `describe()` returns 93 kB on this export; Claude Code refuses an MCP
  result that large and saves it to a file the sandboxed agent cannot read, so both
  analyses worked without the system description and the catalogue. Fixed by
  contracts batch 9 and the tool server: `describe(part)` with a 30,000-character
  cap, the catalogue a table; the chain's prompt says so.
- `value_fix_set` treated an added change as needing
  a fresh replay even when its row was in the table. Fixed in contracts batch 9; the
  scorer's workaround is gone.

## What I did not verify

- **A failure in the middle of a step on real data**: the chain's one command ran from
  an empty directory to the score (`w10-check`, above) and over the finished root,
  so the skip paths are exercised, but no step was interrupted to see it resume
  (the engine's own resumption is tested on its side).
- **Resuming an analysis stopped while waiting for its experiments** (`_resume`):
  written after the review, not run on real data.
- **The fresh-replay path on real data**: neither analysis needed a fresh replay, so
  `fresh_jobs`, the engine's `add` and the scorer's lookup of an added change
  rest on the unit tests.
- **The effect of the faults on `libero_object`**: uncalibrated. The generator used
  the `libero_spatial` curves for detectability; the oracle measured the delay at 39
  points over the scoring window against the 22.5 expected, because s2 runs mostly
  object tasks and the delay costs more there.
- **Run-to-run variation beyond two repeats**: two analyses per condition (`w10` and
  `w10-check`); the first pass's logs-only analysis on the earlier health model
  scored the same recall and k = 3 share with precision 0.67.
- **The health model** is the version-0 generator's guess at the experimental design's section
  2, not measured on anything; the camera residual's noise was widened after the
  review so that it is a clue and not a label (r04 reads 69 and 98 px in the observed
  window, r01 55 px on day 0, r05 49 px on day 1).
- **The traceability rule on agent prose** was checked on these two answers only (0
  untraced numbers each, after small whole numbers in the text were set apart and
  the skills' text stopped counting as a source); no number was traced by hand.
- **That the decoy is a decoy beyond this world**: `policy_knob` at 5 sampling steps
  is flat in the calibration (100 of 100 against 98 healthy twins) and in this
  world's oracle (removing it breaks one episode of 52); the type as a whole is
  "kept" because one image view breaks the policy.
- **The dashboard's rendering** of the two run pages (videos, wide tables): published
  with `hub check` clean, not looked at in a browser.
- **The skills' fixture tests** still use the toy fleet (`tests/skills_fleet.py`);
  moving them onto these fixtures is stage 4's.
- **Exactness of the copied runs**: the runs were made once for the first pass and
  copied into this root; the world's manifest and spec are byte-identical between the
  two (the review's changes touch only the health readings and the fault table), and
  both runs' chain samples held, but no episode was replayed after the copy.
