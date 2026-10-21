# The reuse run: specification and runbook (M2, the learning half)

*Lane learn, 14 October 2026. What has to hold, what is measured, and the
exact commands for the comparison of a fresh agent with an agent that has its
own learned library, on the later fleet. The mechanics are described in
[learned.md](learned.md) and section 9 of [the loop note](loop.md). The code is
`fleetlab.loop.reuse_run` on top of `fleetlab.loop.reuse` and the round driver
(`fleetlab.loop.driver`, `fleetlab.loop.rounds`), whose learning wiring (the
paragraph in the round task, the tool, the freeze after the rounds) it uses. Everything here was proven
on CPU fixtures with scripted agents (`tests/test_reuse_run.py`). No model has
run this yet.*

## 1. The source: where the library comes from

**The investigation.** The library comes from the lead fleet's engagement:
lane fleet's lead fleet (world `w70`, `fleetlab.worlds.loop_fleets` recipe
`LEAD`), two rounds of the round driver, run by lane driver as the
demonstration's main run. The library is not made by a run of its own. The
round driver learns by default (`loop/driver` `8bfe442`; `--no-learning` turns
it off). Then:

- the round task carries the learning paragraph
  (`fleetlab.loop.learned.LEARNING_INSTRUCTIONS`), once;
- `write_learned_skill` is one of the round tools;
- after the rounds, the driver checks and freezes the drafts
  (`fleetlab.loop.rounds.freeze_learned`). The engagement record's `learned`
  names the library and its digest.

**What the agent is asked to write, and when.** The paragraph says:

- write once the evidence from this fleet has established an explanation, or
  ruled one out;
- write it as a competent engineer would want it the next time a similar fleet
  shows a similar pattern:
  - when it applies, as the pattern in the records in general terms;
  - the queries that showed it, as SQL with parameters, and what each
    returned;
  - the competing explanations, and the observation that tells them apart;
  - the experiment that settles the question;
- name no robot, site, episode or experiment, because they differ on the next
  fleet;
- write it before finishing, and revise it by writing it again under its name;
- treat a skill with `origin: learned` from an earlier investigation as
  something to test before relying on it.

The agent may write up to three skills per engagement. Drafts persist across
the engagement's sessions and rounds. The library takes the latest draft of
each name.

**Which engagement is the source.** This is fixed now: M2, the lead fleet's
real two-round engagement (`/var/tmp/robots-loop/driver/m2-w70-astra-2342`,
from the frozen `aa8731b`), with learning on by default. If M2 kept no skill,
the reuse comparison is not run, and "the agent wrote nothing worth keeping"
is reported as the outcome.

The library is never chosen by its content or by any evaluation.

**Freezing and checking.** The driver freezes after the rounds and before the
evaluation. It reads only the analysis' outbox and lane learn's checks
(`fleetlab.loop.learned.freeze`):
- Each draft is rendered with the host's provenance: the fleet label
  (`--fleet-label`, never an identifier), the session, and the day it was
  kept. Lane driver may add the round of the session that wrote it
  (`Source.session_rounds`).
- It is checked against the truth side of everything the source agent saw.
  That is the world's runs plus every finished run of the engagement's loop
  (operation and experiment arms):
  - the draft rules, `contracts.learned.draft_problems`: no neutral
    identifier, declared parameters, filled fields;
  - the rules of a mounted skill, `validate_skill_general` with `code_terms`:
    fault kinds in identifier form, trait names, seeds, fault ids and canaries;
  - the agent-text leak rules: canaries, fault ids, planted names and seeds;
  - the learned front matter and its sections.
- A skill that fails is dropped and listed with the host's reasons, by name
  only, and never mounted. The other skills are kept. Nothing is
  half-written.
- The library is content-addressed: `learned-<digest16>/skills/<name>/SKILL.md`,
  with the host's record `library.json` beside it, which is never mounted.

## 2. The later fleet

Lane fleet's recipe `LATER` (world `w71`, public fleet id `w02`;
`fleetlab.worlds.loop_fleets`). This specification requires the following of
it. As far as it can be checked without reading the fleet's outcomes, the
recipe in `fa981f8` meets each point.

1. **The same kind of lasting condition, elsewhere and different.**
   - It is a remote-link delay at one site that no action removes.
   - The site's public note says what an operator knows; no size and no effect
     is visible.
   - The site differs from the lead fleet's, and so do the delay and the task
     suite. The remembered site is wrong.
   - The remembered setting value may still help. A changed delay does not
     rule out reusing it: in lane adapt's p3, object tasks 14 steps late with
     compensation 17 (the lead fleet's delay) succeeded on 55 of 60 attempts,
     against 38 of 60 at the default. A gain in the "with" arm can therefore
     come from replaying a remembered value. Whether the agent used the skill
     as a procedure is shown from its traces (section 4, "Procedural use"),
     not inferred from the changed delay.
2. **Adaptable through the same public setting.**
   - Some value of `control.latency_compensation_steps` at the condition's site
     must beat the default under the condition. Lane adapt measured +33 points
     (95% interval +19.7 to +46.2) for object tasks at 14 steps late, with the
     compensation set to the delay.
3. **Other faults than the lead's.** Different robots, components and logged
   changes, so the lead fleet's other answers do not transfer either.
4. **Lookalikes that make negative transfer measurable.**
   - A second site with the same remote-link note, whose link is harmless
     (a decoy in the patterns). Compensation there costs success.
   - A site that is worse from day 0 for a reason other than late commands.
   - A skill applied without its competing explanations then adapts the wrong
     site.
5. **The same shape as the lead fleet.**
   - Twelve robots at three sites, 12 attempts per robot and day.
   - The loop layout of `layout_for_world`: round 1 is days 0 to 5 as run,
     with experiments on days 6 and 7; round 2 operates on days 8 to 11, with
     experiments on days 12 and 13; the reserved panel is minted on days 14 to
     17.
6. **Never used in development.**
   - Nobody runs an agent on it before the comparison.
   - Its truth is read only by `reuse_run score` after both arms have
     finished.
   - Lane fleet generates it and runs its round-1 operation; nothing else.

## 3. The arms

`reuse_run compare` runs the later fleet as two engagements of the round
driver.

**What is identical.** Each arm is a separate analysis that the harness
creates fresh (`reuse-<comparison id>-<arm>`). Its memory, outbox, results,
queues and sessions are empty at the start. The two arms are identical in:
- the stored later fleet (its world hash);
- the model, effort and service tier;
- the base skills (digest) and the tool environment (digest);
- the system prompt and tools, with learning on in both;
- the attempt budget, the sessions per round, the layout and the panel.

The "with" arm differs in one thing only: the frozen library, copied into its
analysis at creation and mounted read-only at `/learned`.

**How it is enforced.**
- The harness records the digests per arm (`inputs.json`) and refuses to run
  when anything but the library differs.
- It checks after each arm that the stored fleet and the mounted library are
  unchanged.
- It requires round 1's export to be the same bytes in both arms; otherwise
  `validity` lists the difference and the comparison does not count.
- The launch audit of every session re-checks the library against the later
  fleet's truth side.

**Budget per arm.** The lead fleet's per-round budget, the same in both arms:
- 480 experiment attempts per round (`AttemptBudget`, no limit per arm, none
  on the number of requests);
- at most 8 sessions per round, as in M2, the source engagement, so that the
  later arms get what the source agent had;
- two rounds;
- a reserved panel of 6 attempts per robot and day over the 4 evaluation days,
  so 288 attempts per state, 96 of them at the condition's site.

**Repeats.** One pair is run (section 6). A second pair, with the order
reversed, is optional and follows only once the three main capabilities have
real evidence.
- Model variability is large. In development the agent blamed a harmless
  lighting change in three of four runs on the same fleets. A single pair
  therefore shows what happened once; it cannot separate the library's effect
  from run-to-run variation.
- A second pair is run when the heavy queue allows, with the arm order
  reversed: pair 1 runs `without,with`, pair 2 `with,without`.
- Results are reported per pair. No test is pooled across pairs, and one pair
  is called one pair.

## 4. Metrics, fixed before any run

Each metric is computed by `reuse_run score` from the arm's own submissions,
its engagement record and its reserved-panel attempts. The truth enters
through `ConditionTruth.from_world` alone:
- the condition's site and robots are the targets of the fleet's
  remote-link-delay faults that are not decoys;
- the lookalike sites are those of its decoy remote links.

**The diagnosis targets the affected controller** (a proxy for identification).
This is the first accepted `submit_diagnosis` of the arm, in session order,
with a cause that targets the condition at its site:
- confidence at least 0.5;
- the target's scope is the site, or robot scopes covering at least half of
  its robots;
- the target names one of:
  - the component `controller`;
  - the configuration key `control.action_delay_steps` or
    `control.latency_compensation_steps`;
  - nothing, with a failure mode that says late, latency, lag or delay.

This shows where the diagnosis points. It does not show that the reasoning was
right: a diagnosis that blames the controller while calling its timing normal
counts too. There is no semantic classifier. The matching diagnosis is
reported verbatim beside the metric, with the session's call log, for a reader
to judge. Only the agent's own diagnosis counts; an adopted action does not.
Reported with it:
- **attempts until targeted**: the experiment attempts delivered before that
  session, from requests of earlier sessions with results;
- **model time and tokens until targeted**: every start of every session up to
  and including that one, failed starts included; a lower bound when a start
  has no usage record;
- every diagnosis that targets a lookalike site, or half its robots, the same
  way. These are recorded on their own, whether or not the condition was ever
  targeted;
- if no diagnosis qualifies: "not targeted within the budget", with the
  totals.

**Effective setting adopted.** Every round whose decision adopts, or replaces
into, `set control.latency_compensation_steps` with a value above 0 at the
condition's site or on its robots is listed, with the paired effect of that
round's state at the condition's robots.

"Effective" means that, on the reserved panel, the condition's robots under
that round's whole state succeed at least 10 points more often than under
`original`, with McNemar's exact two-sided p below 0.05 on their paired
attempts. It says that the round's state improved; it does not attribute the
gain to the setting. Only the agent's own experiments can do that, and they are
shown with their results.

- **Attempts until effective**, the headline: the experiment attempts of
  rounds 1 to k, where k is the first effective round. An earlier ineffective
  choice is skipped, and a later replacement counts. If no adoption was
  effective, the value is "none"; every adoption is still listed, and all the
  arm's experiment attempts are given.
- The final outcome is reported separately (below). It is not the state of the
  first effective round.
- **Settings elsewhere**: a setting of the same key at a lookalike site, on a
  lookalike's robots, fleet-wide or elsewhere is recorded on its own, whatever
  the adoption at the condition was. It counts as measured harm only if the
  paired outcomes at the lookalike's robots show a loss: fewer successes under
  that round's state than under `original`, with McNemar p below 0.05.

**Procedural use**, in the "with" arm only, from its call logs:
- which learned skills it read;
- which of their SQL blocks it ran (they carry `-- helper: evidence-N`);
- the values it set, with the decision's rationale.

Reuse of a procedure is shown from these traces. A gain alone could be a
remembered value replayed (section 2).

**Total investigation cost**, per arm, summed over every model start
(`fleetlab.loop.costs.session_costs`):
- sessions, model starts and failed starts, tool calls;
- model seconds, tokens, and the API-equivalent bounds;
- experiment requests and simulated arm attempts.

Unknowns are kept. A start without a usage record makes the arm's tokens,
seconds and prices lower bounds, and no difference of them is reported. The
loop's own simulation (operation and evaluation) is reported beside the cost,
not inside it.

**Final outcome.**
- The reserved panel's success under each arm's last state, fleet-wide with a
  Wilson 95% interval, and at the condition's site.
- Between the arms (`between_arms`): both engagements mint the same panel
  from the same world and layout, which is checked through the panel's digest.
  Their final states are therefore paired attempt by attempt: rescued, broken,
  the difference in points, and McNemar's p.

**Validity.** A comparison counts only if:
- both engagements ended with a decision in every round they ran, completed or
  stopped by the agent;
- both have a reserved-panel evaluation;
- round 1's export is the same bytes in both arms.

Otherwise `validity` lists the reason, `compare` exits non-zero, and no result
of the pair is claimed.

**Headline numbers** (the plan's [X], [Y], [R1], [R0]):
- [X] and [Y]: attempts until effective, without and with the library. If an
  arm never reaches an effective adoption, all its experiment attempts, marked
  "no effective adoption".
- [R0] and [R1]: the final reserved-panel success, without and with.

## 5. What is checked and what is shown

**Before reuse:**
- the freeze's checks (section 1);
- the launch audit of every session in the "with" arm, against the later
  fleet's truth:
  - the agent-text leak rules;
  - the rules of a mounted skill;
  - the learned front matter and sections;
  - no base skill's name;
  - nothing in a skill directory but `SKILL.md`;
- the comparison's own invariants (section 3).

**Shown verbatim:**
- every frozen `SKILL.md`, byte for byte as the later agent reads it;
- the library's digest;
- each skill's provenance;
- the list of checks with their result.

Refused drafts are not shown: their text can carry what the check caught. A
refusal is reported by name and reason only. The leak rules catch identifiers,
canaries, planted names and seeds. They do not refuse a remembered setting
value. Such a value can still help on the later fleet (section 2,
requirement 1), so a gain in the "with" arm is read together with the arm's
procedural use (section 4).

## 6. Commands, compute and model calls

**The source checkout** is the frozen `/home/cs/code/robots-loop-main-aa8731b`.
It is the source of M2 as well, which keeps the comparison on the same code and
tool environment as the engagement that made the library. The harness is in it:
`7c7bc94` was merged by `7507dc6`, an ancestor of `aa8731b`. The comparison needs
nothing from later commits; the probe lane came after it and is not used here.

**The source library** is M2's: the lead fleet `w70`, two rounds, run dir
`/var/tmp/robots-loop/driver/m2-w70-astra-2342`. Its `engagement.json` names the
frozen library in `learned` (`library`, `digest`, `kept`, `refused`). If M2
kept no skill, the comparison is not run, and that is reported.

**The later fleet** is stored: `~/.cache/robots-probes/worlds/w71`. Round 1's
operation is its as-run observed run, which lane fleet ran and checked; the
export checked at `/var/tmp/robots-loop/fleet/exports/w71-r1`. Nothing about
`w71` is generated or run again. Each arm's engagement makes its own round-1
export from that stored run. The harness requires the two to be the same bytes,
and the dry run below checked them against lane fleet's.

```sh
cd /home/cs/code/robots-loop-main-aa8731b
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# The library M2 froze (after M2 has finished)
LIB=$(python3 -c 'import json; print(json.load(open("/var/tmp/robots-loop/driver/m2-w70-astra-2342/engagement/engagement.json"))["learned"]["library"])')

# One pair on w71: arm "without", then arm "with" (M2's library mounted), identical otherwise.
# Lane driver launches it (the only owner of model calls). It runs each simulation step as
# its own heavy job and logs it in the queue log.
uv run --locked python -m fleetlab.loop.reuse_run compare \
    --world-root ~/.cache/robots-probes/worlds/w71 \
    --library "$LIB" \
    --out /var/tmp/robots-loop/runs/reuse-w71-pair1 \
    --backend codex --model gpt-6-astra --effort xhigh --service-tier fast \
    --attempts 480 --max-sessions 8 --panel-per-robot-day 6 --workers 4 \
    --queue-log /var/tmp/robots-loop/queue.log --order without,with

# Afterwards only (reads w71's truth): the registered metrics, written to metrics.json.
uv run --locked python -m fleetlab.loop.reuse_run score \
    --comparison /var/tmp/robots-loop/runs/reuse-w71-pair1 \
    --world-root ~/.cache/robots-probes/worlds/w71
```

What happens:
- `compare` exits 0 only for a valid comparison (section 4, "Validity").
- Before each round's first model call, the driver runs the CLI gate (a record
  for this tool environment exists from M2), the source audit of the round's
  export and prompts, and the login check. The launch audit then runs before
  every session.
- `score` reports, per arm:
  - `identification.first`, with the diagnosis verbatim and its call logs;
  - `adoption.attempts_until_effective`, the headline: experiment attempts
    through the first effective round, None if there was none, with
    `adoption.attempts` listing every adoption;
  - `adoption.adaptation_elsewhere`;
  - `procedural_use`;
  - `outcome`, the final state on the reserved panel, reported separately from
    the first effective round;
  - and for both arms together, `between_arms`, their final states paired
    attempt by attempt;
  - `validity`.
- A second pair, with `--order with,without` and its own `--out`, is optional.
  It follows only once the three main capabilities have real evidence.

**Dry run** (CPU only, no model, no simulation; `scripts`: see the handoff).
It did what `compare` does up to the first model call of each arm:
- prepared both arms with their digests;
- built each arm's round-1 export;
- built the round-1 task exactly as the driver does;
- ran the launch audit of each arm's first session, with M1's real frozen
  library (`48619b46…`) mounted in "with".

M1's library served only the dry run. The real "with" arm mounts M2's. The
receipt is `/var/tmp/robots-loop/learn/dryrun-w71/dryrun-receipt.json`
(14 October, 257 s). It found:
- the stored loop layout equals `layout_for_world`;
- the arms' recorded inputs differ only in `library` and `mounted_library`:
  - world hash `37faadb5…`, base skills `b093708d…` and tool environment
    `3422ebcab59730bf` are the same in both;
  - so are the model, the budget (480 attempts and 8 sessions per round, the
    14 round tools), the system prompt and the runner's inputs;
- a passing CLI gate record exists for that tool environment, from 13 October
  at 23:01;
- both round-1 exports are the same bytes, `64dbec69…`, equal to lane fleet's
  checked `w71-r1`;
- both round-1 tasks are identical, with the learning paragraph once;
- both launch audits passed: 75 s for "without", and 59 s for "with" with the
  `learned` root mounted.

Before the launch, the same check can be repeated with M2's library. It takes
about 4 minutes on the CPU, from the frozen checkout:

```sh
cd /home/cs/code/robots-loop-main-aa8731b
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --locked python \
    /var/tmp/robots-loop/learn/scripts/w71_dryrun.py /var/tmp/robots-loop/learn/preflight-w71 "$LIB"
```

**Compute** (attempts simulated, at about 1,500 attempts an hour):

| Step | Attempts | Hours |
|---|---|---|
| The later fleet's round-1 operation: stored already, not run again | 0 | 0 |
| Per arm: round-1 experiments, at most | 480 | 0.3 |
| Per arm: round-2 operation, under the arm's own state | 576 | 0.4 |
| Per arm: round-2 experiments, at most | 480 | 0.3 |
| Per arm: evaluation of 4 states (original, round 1, round 2, one-look) at 288 each | 1,152 | 0.8 |
| **Per arm, at most** | **2,688** | **1.8** |
| **One pair** | **5,376** | **3.6** |

- An arm that stops after round 1 skips its round-2 operation, experiments and
  one state: about 1,000 attempts less.
- The engine's chain-sample replays add a few percent.
- The `original` state is the same in both arms and is simulated in each:
  288 attempts that a shared evaluation could save, which is not built.

**Model calls.** Per engagement, at most 8 sessions per round over 2 rounds:
16 sessions, each with at most one replacement start for a lost call. For one
pair: at most 32 sessions of `gpt-6-astra` at effort xhigh on the fast tier.
A round usually ends well before its eighth session, when
the decision is accepted.

For reference, the driver's part-one run on the miniature fleet (two
sessions, `p1-astra-2049`) spent:
- 21 minutes of model time;
- 5.8 million input tokens (5.6 million cached) and 50,000 output tokens;
- 106 tool calls;
- about $20 to $40 at API-equivalent prices.

On this fleet of twelve robots, expect 40 to 90 minutes of model time per
engagement. This is an estimate, not a measurement. Model time and simulation
alternate rather than overlap, so allow 3 to 4 hours of wall time per arm.

## 7. What this run shows, and what it does not

**Shows**, for one pair, or two:
- whether the agent with its own skill targeted the affected controller, and
  reached an effective adaptation, sooner, in experiment attempts and model
  time;
- whether it ended better on the same reserved attempts;
- whether it was misled by a lookalike;
- whether it used the skill as a procedure (its traces) or only arrived at a
  remembered value.

**Does not show:**
- that learning carries over in general: one source fleet, one later fleet,
  one or two pairs;
- that the difference is caused by the skill and not by run-to-run variation;
  a single pair cannot separate them;
- that an effective round's gain comes from the adaptation setting rather than
  the round's other changes, unless the agent's own experiments show it.
