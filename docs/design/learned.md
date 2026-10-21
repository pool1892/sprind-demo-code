# Learned skills and the reuse comparison: how section 9 is built

*Lane learn, 14 October 2026. Builds
[section 9 of the loop note](loop.md): an analysis agent writes an
inspectable diagnostic skill during an investigation, the host checks and
freezes it, and a later investigation of a different fleet receives it beside
the unchanged base skills. Mechanics are proven on CPU fixtures with scripted
agents; no model session has used them yet.*

## The path of a learned skill

```
agent (session n)        write_learned_skill(LearnedSkillDraft)          contracts.learned, toolserver.server
  tool server            checks the draft, runs each SQL once here,      outbox/learned.jsonl (latest line per name)
                         keeps it without the check values
host, after the          stored_drafts -> render with provenance ->      fleetlab.loop.learned.freeze
investigation            audit_learned_skill (source fleet's truth) ->
                         learned-<digest16>/{skills/<name>/SKILL.md, library.json}
later analysis           Analysis.create(..., learned=library.skills)    agents.sessions (copied in, digest recorded)
every later session      mounted read-only at /learned; launch audit     agents.sandbox, agents.audit.launch_audit
                         checks it against the later fleet's truth;
                         list_skills / read_skill serve it after the base skills
```

**The draft** (`LearnedSkillDraft`, the tool's argument): `name`,
`description`, `applies_when` (the pattern in the records that makes it
relevant), `parameters` (name and meaning), `evidence` (one to eight items of
`claim`, `sql`, `result`: what the query shows, the SELECT with `$name`
parameters, what it returned on the fleet it was written on), `alternatives`
(the competing explanations and the observation that tells each apart),
`next_experiment`, and `check_with` (a value of this fleet per SQL parameter,
used once to run the query and never stored). At most three learned skills
per analysis; writing one again under its name replaces it.

**The rendered skill** is an ordinary `SKILL.md` (contracts, section 13):
front matter with `origin: learned`, `applies_when` and `provenance` added to
the usual keys, and the sections Applies when, Parameters, Evidence (each SQL
in a ```` ```sql ```` block headed `-- helper: evidence-N`, as the base skills
do), Competing explanations, Next experiment and Provenance. A short note under
the title tells the later agent that the skill comes from an earlier
investigation and that this fleet's records win where they disagree.
Rendering is deterministic, so the library's digest is stable.

**Provenance** is the host's, not the agent's: a label for the fleet (never
its identifier; a label with an identifier is refused), the round, the session
and the day it was kept. The host's own record (`library.json`, beside the
mounted `skills/`, never mounted) adds the source analysis, its world id, the
receipt, what each skill replaced and which earlier library a skill was
carried from.

## The rules and where they hold

| Rule | Tool server (sandbox, no truth) | Host freeze (source truth) | Launch (later truth) |
|---|---|---|---|
| no neutral identifier (robot, site, episode, experiment, fault, world) | yes, with a message naming the parameter to use | yes | yes |
| SQL parameters and prose placeholders declared, fields filled, sizes | yes | yes (again, on the stored draft) | front matter and sections |
| each SQL one SELECT that runs on this fleet | yes | no | no |
| agent-text leak rules (`audit_export` with `agent_texts`: canaries, fault ids, planted names in identifier form, seeds) | no | yes | yes |
| world-specific terms of a mounted skill (`validate_skill_general` with `code_terms`) | no | yes | yes |
| no base skill's name; nothing but `SKILL.md` in a skill | yes (name) | yes | yes |
| the draft in the outbox at the next session's launch | | | agent prose (`_LEARNED_PROSE`), like a diagnosis |

Natural-language forms of hidden names ("mount tolerance") pass, as in an
agent's notes; identifier forms (`mount_tolerance_mm`) and canaries do not.
The template text, the tool's description and the learning paragraph pass the
strict audit against both test fleets (a test).

## Base skills stay the same in both arms

The learned library is a separate mount (`/learned`) and a separate field of
the analysis, not part of the tool environment. `ToolEnv.skills`, its digest
and therefore the gate record are the same with and without a library;
`list_skills` returns the base manifests byte for byte as before
(`exclude_none`) and lists learned skills after them, and a learned skill can
neither take a base skill's name nor shadow one.

## The reuse comparison (`fleetlab.loop.reuse`)

`compare(case, agent, env, library, out, budget=..., engine_for=..., score=...,
runner=...)` runs one `LaterCase` (world id, export, system prompt, task,
truth-side audit inputs) in two arms, "without" and "with". Enforced:

- each arm is a new analysis created for the comparison
  (`out/analyses/reuse-<arm>`); its memory, outbox, results, queues and
  sessions are empty at the start, and the library, copied in at creation, is
  the only thing it receives;
- per arm the harness records the digests of the export, the base skills, the
  tool environment and the library (frozen and as mounted), the model spec,
  the budget, and hashes of the prompts (`inputs.json`); it refuses to run when
  anything but the library differs, when the mounted copy is not the frozen
  library, and when the export or the library changes during an arm.

Per arm it reports sessions and their states, starts, tool calls (and by tool,
from the call logs), reads of learned skills, experiments requested, done and
refused, attempts requested and simulated, model wall time, tokens and list
price, the learned skills the arm wrote itself, and the outcome; then the
differences (with minus without) and possible negative transfer: a lower
score with the library, more spent for no better outcome, or a library never
read. `comparison.json` and `comparison.md` go to `out`.

Costs are summed over every model start of every session, a failed or
replaced start included (`fleetlab.loop.costs.session_costs`, the driver
lane's helper, which both lanes use): a session's record keeps only its last
start's usage. A start without a usage record makes the arm's tokens, model
seconds and prices lower bounds (`usage_complete: false`, with the reason), and
then no difference of them is reported, only the bounds; counts that do not
depend on usage are still compared.

Each comparison has its own id (`out/reuse.json`) and its arms the analysis
ids `reuse-<id>-<arm>`, so two comparisons on one later world never share the
queue engine's per-analysis plans and exports. `compare(..., resume=True)`
reopens the same analyses, refuses inputs that changed (compared in their
stored, serialized form), skips finished arms and continues an interrupted
one: `run_session` recovers a session in progress, and a waiting session's
saved queue goes to the engine again, which resumes its own plan.

**Seams for the driver.** `runner(ArmSetup)` runs one prepared arm; the
default is `run_analysis`, and the round driver can pass its own (a whole
round with operation, experiments and a decision), returning extra costs that
the harness records beside its own. `score(analysis)` is the truth-side
outcome, for example reserved-panel success after the arm's decision is
adopted, with a number under `score`. `engine_for(arm)` gives each arm its own
engine. The learning paragraph (`LEARNING_INSTRUCTIONS`, appended by
`with_learning` and by `integration.analyse.task_prompt`) is in both arms'
task alike.

## The related later case (proposal for the demonstration)

*Superseded by the specification in [reuse-run.md](reuse-run.md); kept as the
proposal lane fleet built `LATER` from.*

With lane adapt's lasting condition (commands that reach one site's robots
late, adapted through actions per chunk and latency compensation):

- **Source.** The lead fleet's investigation, round 1 or 2: the agent finds
  the late-command site, measures settings by experiment and, if it does what
  the paragraph asks, writes a skill such as "one site worse from the first
  day across tasks and robots, no logged change, the arm lagging the commanded
  path in the motion records; rule out a site condition; test actions per
  chunk and compensation by experiment at that site".
- **Later fleet.** A new fleet from the same recipe family, created after the
  agent is fixed and never used in development: the same kind of condition at
  a *different* site and of a different size, among *different* other faults
  (other robots, other components, a different harmless change). To make
  negative transfer measurable, a second site with the same public note (its
  controllers also run remotely) but no delay, and a site-level lookalike
  (an object batch or lighting condition that lowers success from day 0):
  a skill applied without its competing explanations would adapt the wrong
  site.
- **Measured.** One round of the later fleet per arm, with the same budget:
  sessions, tool calls, experiments and attempts until the decision, model time
  and tokens, and the outcome on the later fleet's reserved panel under each
  arm's adopted state (lane state's `evaluate`). With two runs per arm if the
  budget allows; one run per arm is an anecdote and is shown as one.

## Before real runs

The tool server's source, the sandbox and the launch audit changed, so the
tool environment's digest and `implementation_digest` change: run
`make agents-gate` (free checks, no model call) before the first model session
on this branch, or `require_verified` refuses the launch.

## Not shown yet

That a model writes a useful skill when asked, reads it on a later fleet, or
spends less with it. That the leak rules catch every way a skill could carry
a fleet's answer: they catch identifiers, canaries and planted names; a value
(a setting that worked) is not refused. A remembered value can still help on a
later fleet with another delay (lane adapt's p3: object tasks 14 steps late
with compensation 17 succeeded on 55 of 60 attempts, against 38 of 60 at the
default). Whether a skill was used as a procedure is read from the arm's traces
([reuse-run.md](reuse-run.md), section 4).
