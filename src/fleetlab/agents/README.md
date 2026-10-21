# fleetlab.agents: the agent runtime

Sandboxed `claude -p` and `codex exec` calls with our tools, our skills, a
memory per world, schema-checked answers, token accounting and a tool log.
The tool server the agents use is `fleetlab.toolserver`. Built on the
stage-1 spike, [spikes/agent-runtime](../../../spikes/agent-runtime/README.md), which is
the reference for the CLIs' flags and behaviour; this package ports it and
changes what is listed under "What changed from the spike".

Stage-4 preflight repair (2026-10-09): Codex 0.161.0 leaves an
empty `.sqlite-maintenance.lock` in its private runtime home. The isolation
check accepts that exact name only as a zero-byte regular file, rejecting
content, directories and symlinks. This changes `checks.py` only; no login or
CLI command is changed. The 50 runtime unit tests and
`make agents-gate AGENTS_GATE_FLAGS=codex` pass, including wire, isolation and
authentication checks. The gate uses a local wire capture, not a live analysis.
`AGENTS_GATE_FLAGS` optionally selects a backend; leaving it empty preserves
the two-backend gate.

## Calling an agent

```python
from fleetlab.agents import run_agent, ToolAccess
from fleetlab.agents.audit import AuditContext
from fleetlab.agents.toolenv import build_tool_env
from fleetlab.agents.sessions import Analysis
from fleetlab.contracts.tools import Diagnosis

env = build_tool_env(role="analysis")                  # interpreter, packages, source, and the
                                                       # analysis skills (fleetlab.skill_library)
analysis = Analysis.create("w01", label="opus, run 1")  # a new memory for this world
audit = AuditContext(tuple(run_dirs), full_fleet)       # evaluator-only inputs
access = analysis.tool_access(env, export_dir, session=1, audit=audit)

res = run_agent("claude", "opus", SYSTEM, PROMPT, images=[...], tools=access, schema=Diagnosis)
res.ok, res.error           # True/None, or one of the failure kinds below
res.answer                  # a Diagnosis (validated); res.result is its JSON
res.usage                   # input_tokens (cached included), cached_input_tokens, output_tokens,
                            # reasoning_tokens; claude also cost_usd_list_price
res.seconds                 # wall clock of the CLI process; res.meta["phases"]["answer_s"]
                            # when the answer arrived, res.meta["api_ms"] (claude) model time
res.tool_log                # every tool call: arguments, full result text, image hashes
res.run_dir                 # argv.json, prompt.json, stdout.jsonl, stderr.txt, result.json,
                            # box/ (what was mounted), box/toolbox/tool_calls.jsonl (the call log)
```

Parameters: `backend` (`claude`, `codex`), `model` as the CLI names it
(`opus`, `fable`, `haiku`, `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-luna`),
`images` (PNG/JPEG paths), `tools` (None for a call without tools), `schema`
(a pydantic model, or a strict JSON schema dict, or None for text),
`effort`, `service_tier` (codex: `default` or `fast`, passed on
every codex call; `flex` is refused by the service on the subscription),
`thinking_tokens` (claude: `MAX_THINKING_TOKENS`),
`auth` (`subscription` or `api_key`), `timeout`, `run_root`, `cli_version`
(default: the pinned one). `run_agent_with_retry` repeats by failure kind.
There is no unsandboxed mode (see "What changed from the spike").

An analysis in sessions, with the engine answering the experiment queue:

```python
from fleetlab.agents.sessions import AgentSpec, run_analysis
record = run_analysis(analysis, export_dir, AgentSpec("codex", "gpt-6.1-sol", service_tier="default"),
                      SYSTEM, TASK, env, engine, max_sessions=4, audit=audit)
[s.state for s in record.sessions]       # e.g. ["waiting", "done"]
analysis.diagnosis(), analysis.fix_set(), analysis.notes()
```

A batch, which refuses to start on an unchecked CLI:

```python
from fleetlab.agents.batch import Batch, AnalysisJob
batch = Batch.after_gate(env, ["claude", "codex"], export_dir, run_dirs, fleet=fleet,
                         texts={"system": SYSTEM, "task": TASK}, concurrency=8)
records = batch.analyses([AnalysisJob("w01", export_dir, AgentSpec("claude", "opus"), SYSTEM, TASK)], engine)
batch.stopped               # None, or why no further calls started (auth, rate_limit, window)
```

`Batch.after_gate` runs the gate before every batch (about 25 s
per vendor; `rerun=False` runs it only where a record is missing or stale),
then, for this batch, the source audit over its own export, its prompts and
the mounted code, and the login check; `Batch(...)` alone only requires the
records.

## What is mounted where

Two nested bubblewrap sandboxes, as the spike recommended. The CLI runs in
the outer one; it starts the tool server, which runs in the inner one.

| Inside | CLI sandbox (outer) | Tool sandbox (inner) | On the host |
|---|---|---|---|
| `/usr`, `/etc/ssl`, `/etc/ca-certificates`, `/etc/resolv.conf` | read-only | `/usr` only | the system |
| `/home/agent` | tmpfs, the CLI's home | absent | nothing |
| `/home/agent/.claude/.credentials.json` or `/home/agent/.codex/auth.json` | the login file **alone**, read-write, subscription route only | absent | `~/.claude/.credentials.json`, `~/.codex/auth.json` |
| `/home/agent/.codex` (codex) | read-write | absent | a private home per concurrent slot, `<root>/codex-homes/NN` |
| `/opt/cli` | read-only | absent | the pinned CLI's install directory |
| `/run/agent` | read-write; the CLI's cwd is `/run/agent/cwd` | absent | this call's `<run>/box/` only: cwd, codex's system.txt, schema.json, images, last_message.txt, toolbox/ |
| `/opt/tools/python`, `/opt/tools/site`, `/opt/tools/src` | read-only | read-only | `<root>/toolenv/{python-3.12.15,site-<hash>,src-<hash>}` |
| `/work` | read-only | read-only | the export (`log/`, `data_dictionary.md`) |
| `/skills` | read-only | read-only | `<root>/toolenv/skills-<hash>`, a copy of the skill library |
| `/memory`, `/outbox` | read-write | read-write | the analysis' `memory/`, `outbox/` |
| `/results` | read-only | read-only | the analysis' `results/` |
| `/log` | read-write | read-write | the run directory's `box/toolbox/` |
| network | on | **off** | |
| environment | built from nothing (below) | `PATH`, `LANG` only, for the tool server (`--clearenv`) and for bwrap itself, PID 1 (`env -i`) | |

`<root>` is the agent root, `/var/tmp/robots-agents-<uid>` unless
`ROBOTS_AGENT_ROOT` says otherwise: outside the repository and outside the
home directory, private to the user. Only three things of the
home directory are mounted anywhere: the pinned CLI's install directory and
the one login file of the CLI, both in the outer sandbox, and nothing in the
inner one. No host path appears inside: every mount point has a neutral name,
and the run directory's own records (`argv.json` with the host paths of every
mount, `prompt.json`, `stdout.jsonl`, `stderr.txt`, `result.json`) lie beside
`box/`, outside the mount.

The agent root must be a directory, not a link, owned by the user and of mode
0700; `cli.agent_root` refuses it otherwise, because `/var/tmp` is writable by
everyone and someone could plant the path first.

**Process 1 of each sandbox.** Inside a bwrap sandbox bwrap itself is PID 1,
and `/proc/1/environ` and `/proc/1/cmdline` show its environment and command
line to every process inside. The tool sandbox's bwrap is started through
`env -i PATH=/usr/bin LANG=C.UTF-8`, so its PID 1 holds those two variables
(a static probe checks it). The CLI sandbox's PID 1 unavoidably shows the
outer command line (the sandbox's mount points, the CLI's flags, the system
prompt) and the CLI's own environment, an API key included on that route;
the CLI process holds all of that anyway.

**The codex slot homes persist** codex's caches between calls, and between
worlds: `installation_id`, `models_cache.json`, its SQLite databases
(goals, logs, memories, queue, state), `skills/.system` (codex's bundled
skills, not offered), `shell_snapshots/`, `tmp/arg0/` links to its own binary,
and the 0-byte mount point `auth.json`. Nothing of a call may stay there for
the next: `memories` and `shell_snapshot` are switched off, `--ephemeral`
keeps sessions out, and the isolation check reads every home (below), and
`run_agent` stops with `auth` when `auth.json` is anything but an empty file
after a call.

The CLI's environment: `HOME=/home/agent`, `PATH=/usr/bin`, `LANG=C.UTF-8`;
claude adds `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`,
`CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`, `DISABLE_AUTOUPDATER=1` (and
`MAX_THINKING_TOKENS` when asked); codex adds `CODEX_HOME`. It reaches bwrap
through the process environment, never through `--setenv`, so that an API
key never lands on a command line or in `argv.json`.

The tool server's environment (`toolenv`):

- `python`: a copy of the uv-managed CPython the project runs on
  (python-build-standalone, relocatable), run with `-I -S`: no environment
  variables, no `site`, no `.pth` files; `sys.path` is set to the bundle and
  the site directory by the launcher.
- `site`: only the tool server's packages (`duckdb`, `pyarrow`, `numpy`,
  `imageio`, `imageio-ffmpeg`, `pillow`, `pydantic` and what they pull in) at
  the project environment's versions, installed with `uv pip install
  --target` from uv's cache.
- `src`: the import closure of `fleetlab.toolserver`, computed by importing
  it in a fresh interpreter and copying exactly the modules loaded:
  `fleetlab/__init__`, `contracts/*`, `log/__init__`, `log/reader`,
  `log/dictionary`, `toolserver/*`. The log package imports its writer,
  export and leak search only on first use, so none of them is in the
  sandbox. Never `fleetlab.faults`, `sim`, `agents`. The
  contracts' source names the hidden tables and columns by design; the
  coordinator allowed that, and the source audit searches it for
  world-specific terms only (below).
- `skills`: a copy of the skills mounted for the run: `role="analysis"`
  takes the eight analysis skills of `fleetlab.skill_library` from the
  checkout's `skills/`, `role="supervisor"` the supervisor's two; or
  `skills=` and `skill_names=` for any other library (the measured tier mounts
  a toy skill). `make agents-gate` audits the analysis skills.

All four are content-addressed and built once; the gate records the tool
environment's digest, so a change to the tool server's source needs the
gate to run again.

## The tool server

Every tool of contracts section 12, over one export and one analysis
(`fleetlab.toolserver`, an MCP server written here: the part of the protocol
the CLIs use is five JSON-RPC methods, and the `mcp` package would bring an
HTTP stack into the sandbox).

| Tool | What it does here |
|---|---|
| `query(sql)` | one SELECT per call (DuckDB's statement type is checked; not even EXPLAIN, because DuckDB calls `EXPLAIN ANALYZE INSERT ...` an EXPLAIN and runs the INSERT), on a `LogReader(materialize=True)` locked with file access off; at most 200 rows; interrupted after 60 s; one reader per session, which no query can change |
| `get_frames(episode_id, steps, camera)` | up to 8 PNG images, each with a line naming the step asked for and the logged frame shown |
| `describe(part, include_logs)` | one part of the data dictionary and the system description as compact JSON (`contracts.tools.describe_part`): `dictionary`, `fleet` (the change and maintenance logs only with `include_logs`, left out for the no-data baseline), `vocabulary`, `costs`, or `all`; at most 30,000 characters per answer, a larger one refused with a pointer to the smaller parts and the tables `change_log`, `maintenance` and `catalogue`. The action catalogue is the table `catalogue` |
| `list_skills()`, `read_skill(name)` | the manifests; the whole `SKILL.md`, with helper files named (no tool runs them: see "Not verified") |
| `request_experiment(ExperimentRequest)` | checks robots and scopes against the fleet, refuses `route_to_teleop` and `request_training_data` (no simulated effect) and anything over the budget: the contracts' `EXPERIMENT_BUDGET` (8 requests per analysis; the request model refuses more than 60 episodes per arm) and 960 episodes per analysis unless the runtime raises it (`max_experiment_episodes`); the tool's description and every refusal say the limits; appends to `outbox/experiments.jsonl`; the receipt's `receipt_id` is the experiment id (`x0001`, ...) |
| `experiment_results()` | every request so far: its result when the runtime has one, else status `queued` |
| `submit_diagnosis(Diagnosis)`, `submit_fix_set(RankedFixSet)` | validated (scopes exist, cited episodes exist, a count budget holds); stored as `outbox/diagnosis.json`, `outbox/fix_set.json` and one line each in `outbox/submissions.jsonl` |
| `memory_read()`, `memory_write(notes)` | `{"notes": ...}`; replaces `memory/notes.md`, at most 64 KiB |

**Why `describe` answers in parts.** Claude Code 2.1.293 accepts an MCP tool
result of up to 25,000 tokens (`MAX_MCP_OUTPUT_TOKENS`, else a flag set by
the vendor, else 25,000; read from the binary). It estimates characters / 4
(plus 1,600 per image): at most half the limit, 50,000 characters of text,
passes without counting; above four times the limit is refused at once; in
between the API counts the tokens and refuses above 25,000. A refused result
is saved to a file in the CLI's home, which the sandboxed agent has no tool
to read: the miniature fleet's whole description (92,874 characters, the
catalogue its bulk) never reached the agent. The 30,000-character cap keeps
every answer in the uncounted band, with room for text that tokenizes
densely. On the miniature fleet the parts are 12 to 8,467 characters and
`all` 17,650. Codex has a `tool_output_token_limit` setting and truncates
instead of refusing (its default not read). Eight frames are estimated at
12,800 tokens and are therefore counted, and pass.

Invalid arguments come back as a tool error with pydantic's message, so the
agent can correct itself. A call to a tool that is not offered is answered
with an error and recorded like any other. The call log, one JSON line per call in
`box/toolbox/tool_calls.jsonl` (`run.call_log(run_dir)`): `seq`, `session`, `at`, `tool`, `arguments`,
`ok`, `error`, `result` (the full text returned), `images` (sha256 prefixes),
`seconds`. That is what the scorer needs to trace a number in an answer to a
tool result.

## Sessions

An experiment is a heavy job of minutes, so an analysis is a sequence of
sessions, each one CLI call (`sessions`):

1. A session runs with every tool. Its system prompt ends with the session
   rules (request, write memory, finish "waiting"; submit, finish "done"),
   and its answer schema is `SessionReport` (`status`, `summary`).
2. The runtime decides the state from what the tool server recorded, not from
   the agent's word: **waiting** if the session queued an experiment,
   **done** if it submitted and queued nothing, **ended** if neither,
   **failed** if the call failed. A failed session is retried by its failure
   kind only when it left nothing behind; a retry after a queued request
   would queue it twice.
3. While waiting, the runtime writes `queues/session-NN.json`, an
   `ExperimentQueue`, and hands it to the engine (`Engine.run_queue`).
4. The engine answers with an `EngineReply`: one
   `ExperimentResult` per request (status, the episode ids of each arm) and
   the export that now holds the new episodes. The runtime appends the results
   to `results/experiments.jsonl` (read-only for the agent).
5. The next session starts on the new export with a continuation prompt; the
   agent sees the results through `experiment_results` and its notes through
   `memory_read`. Memory is the only state that crosses sessions.

Without an engine, `run_analysis` stops at the first waiting session with the
queue file written.

The queue format (`fleetlab.experiment-queue/1`), for the engine:

```json
{"format": "fleetlab.experiment-queue/1", "world_id": "w01", "analysis_id": "w01-...",
 "session": 1, "created_at": "2026-10-07T18:20:00", "export": "/path/of/the/export/the/session/saw",
 "rules": {"window": "experiment", "both_arms_always_simulated": true,
           "arm_episodes_carry_experiment_id": true},
 "experiments": [{"experiment_id": "x0001", "session": 1,
                  "request": {"arms": [{"actions": []}, {"actions": [{"verb": "recalibrate", "target": {...}, "value": null}]}],
                              "robots": ["r02"], "episodes_per_arm": 4, "paired_starts": true, "power_note": "..."}}]}
```

and the reply: `{"results": [ExperimentResult, ...], "export": "/path/of/the/new/export"}`.
The rules field carries the contract's rules: every request runs in the
experiment window; both arms are always simulated, also when an action
targets something healthy, or the agent would learn health from identical
results; and every arm episode's identity carries the experiment's id
(`EpisodeIdentity.experiment_id`, visible as `episodes.experiment_id`), so
that a later session keeps the arms out of the observed rates.
`run_analysis` checks the last rule in the reply's export and raises
`EngineError` when an arm episode is missing or unmarked (`arm_problems`).
The measurement fixture exercises refused replies without fabricated arms.
How the engine spreads `episodes_per_arm` over the
robots is the engine's; the budget counts `episodes_per_arm` times the number
of arms.

## Memory per world

`Analysis.create(world_id, ...)` makes a new directory (by default under
`<root>/analyses/`) with an empty `memory/`, and refuses an existing one. A
new world, or the same world under another method or model, therefore never
sees another analysis' notes: the reset is by construction, not by deletion.

## Failures and retries

`run_agent` classifies each failure. Scored ex-post sessions follow design
section 10, entry 11: at most two starts, with one replacement only when the
first call was lost to the software around the model without an answer.
`lost_runtime_answer` in `run.py` is shared with the model baselines. It checks
all answer surfaces (structured result, final text, validated answer); an answer
that cannot be read stays one sample, as entry 7 requires. A failed start that
queued a request or submitted an answer is never restarted. A tool server lost
after recorded tool work is never restarted either.

| Caller | Rule |
|---|---|
| `integration.analyse`, `integration.chain`, and `Batch.analyses` through `run_session` / `run_analysis` | Scored ex-post rule: one replacement for an untouched lost call, two starts maximum per session |
| `agents.measure.measure_session` through `run_analysis` | Exercises that same session protocol on the toy fixture |
| `baselines.models.model_call` | Same lost-answer predicate and two-start limit; no tools, so tool-server losses do not apply; behaviour unchanged |
| `run_agent_with_retry` and its external callers | Generic `RETRY` / `MAX_ATTEMPTS` policy below, unchanged; not used for scored ex-post sessions |
| Direct `run_agent` calls in measurement and `live.vlm.supervisor.VLMSupervisor` | One start per invocation; the live supervisor's later decisions and model calls follow its own design |

The scored-session column assumes no answer arrived and no request or submission
was left behind. No combination of error kinds allows more than two starts.

| Kind | Scored ex-post session | Generic `run_agent_with_retry` |
|---|---|---|
| `timeout` | One replacement start | Once, immediately |
| `overloaded` | One replacement start | After 5, 20, 60 seconds, four starts maximum |
| `cli_error` | One replacement only for an identified transport failure; otherwise stop | Once, after 2 seconds |
| `tool_server` | One replacement only before any server tool work; otherwise stop | Once, after 1 second |
| `malformed_output` | Stop after one start, including empty or invalid JSON and schema failures | Once, immediately |
| `rate_limit` | Stop; a `Batch` also stops | Stop |
| `auth` | Stop; a `Batch` also stops; login needs a human | Stop |
| `refusal` | Stop | Stop |
| `bad_request` | Stop | Stop |

Session JSON now uses the baselines' `attempts` shape: each start has `ok`,
`error` (the kind), `detail`, `run_dir`, `seconds`, `usage`, `runtime_meta`, and
`answer_present`. `rerun` means there were two starts; the session's `seconds`
is their total. The replacement supplies the final result, usage and run
directory; per-start usage remains in `attempts`. `lost_twice` means both
starts were lost without an answer. This stops the analysis and is copied to
the top-level `lost_twice` boolean in `integration.analyse`'s `run.json` before
scoring, so the evaluator can score no answer without opening session files.
The replacement uses the same rules as the first start: a request makes the
session `waiting` and is dispatched, a submission with no request makes it
`done`, including when a timeout follows either. `lost_twice` is true only
when neither an answer nor a request/submission remains after the replacement.
Calling `run_analysis` again on a twice-lost analysis returns its stored record
and starts no further call.

Before the first start, the host snapshots the mounted writable `memory/` and
`outbox/` trees under `session-state/session-NN/before/`. Before a replacement,
the lost start's trees are kept under `lost-start-1/` and the original trees
are restored, including deleted files and removal of newly created files.
These records are outside the agent's mounts. Each CLI start already has a
fresh working directory in its own box; the lost box and stream remain in its
run directory and are never mounted into the replacement. Results and the
export are read-only. The prompt, tools, model settings and full time limit
stay the same.

Every completed start is saved immediately in `session-NN.json`, using an
atomic file replacement. `in_progress` distinguishes an unfinished session
from a completed one. The process-start callback also saves a provisional
receipt, so a caller interruption after dispatch consumes that start. A
continuation keeps these receipts and uses only the starts still available.
It recovers a saved child result or final stream message when available, and
otherwise judges the interrupted start using its recorded tool work. Unknown
time/usage is marked `runtime_meta.interrupted_without_result`; the provisional
receipt's zero seconds is not a measured duration. Pipeline accounting retains
its existing missing-result treatment. A failed preflight consumes no start.

Codex stream commentary stays in the raw stream and generic callers' text.
For scored tool sessions it does not count as an answer: the runtime requires
the last-message file, an explicit final message, or an unlabelled final message
followed by a completed turn. A malformed final answer still receives no retry.

Scorer-facing fields retain their existing names and locations:

| Record | Fields |
|---|---|
| Analysis `run.json` | `lost_twice` |
| `sessions/session-NN.json` | `lost_twice`, `attempts`, `rerun` |
| Model baseline `metadata.json` | `attempts`, `rerun` |
| Baseline `manifest.json` when a method is missing | `missing.<method>.attempts` |

Round 2's baseline comparison against the definition before round 1 covered
3,840 constructed results: zero differences in loss classification or attempt
receipts. The baseline call loop's syntax tree is unchanged. This is CPU
stand-in evidence, not a measurement of recovery through a live provider.

CPU tests use stand-in calls to cover every error kind, early and mid-work
tool-server loss, all answer surfaces, queued/submitted work, unchanged call
conditions, replacement success or failure, per-start receipts, and the
twice-lost signal in `run.json`. They also check that both starts belong to one
scored repeat and both count in accounting. These tests make no model call and
do not establish live provider recovery, simulator or GPU behaviour.

Every call adds a line to `<root>/ledger.jsonl`: backend, model asked and
reported, CLI version, effort, service tier, auth route, seconds, answer
time, model time, error, tokens, tool calls.

## The pinned CLIs and the gate

`cli.PINNED`: claude **2.1.293**, codex **0.161.0**. Every call runs the
real binary of the pinned version under
`~/.local/share/mise/installs/<tool>/<version>/`, never `latest` and never the
launcher scripts in `~/.local/bin` (they run `mise use -g` on every start).

`make agents-gate` (`python -m fleetlab.agents.checks`) runs, without a model
call, for each pinned version, on the toy world:

| Check | What it asserts |
|---|---|
| source audit | `fleetlab.log.audit_export` over the export, with the tool descriptions as `texts=` and every mounted file (source bundle and skills) as `code=`: no truth term in the data and texts, no fault kind, trait name or canary in the code |
| version | `--version` of the binary equals the pin |
| layout | the command lines `run_agent` builds, read back: no repository, no truth directory, no home directory but the CLI's install dir and the one login file, the login file bound alone and only in the CLI sandbox, no network and only `PATH`/`LANG` in the tool sandbox, nothing writable but memory, outbox, log and run dir, no link toward a login in a writable mount, the export is the visible tree only |
| static | plain processes in both sandboxes: canary outside the export, truth directory of the exported run, repository, home, both login stores, the other CLI's login: hidden; own login: in the CLI sandbox only; run directory: CLI sandbox only; export and skills: visible, not writable; memory writable; results and source not writable; network: CLI sandbox only; no file anywhere with the canary or the repository's text (with a positive control); `fleetlab.faults` not importable in the tool sandbox while the tool server's own modules are (positive control); the tool sandbox's environment, and that of its PID 1 (`/proc/1/environ`), is `LANG`, `PATH` |
| isolation | claude: a throwaway home with a user CLAUDE.md, auto-memory, hooks, a skill and an MCP server reaches nothing in the request; the control run without our flags shows them. codex (after the wire check, so that a home exists; none fails): per slot home, `auth.json` a 0-byte placeholder, nothing off an allowlist of codex's caches, no history or sessions, no `AGENTS.md`, `config.toml`, hooks, rules, user skills or links toward a login, the memories database empty but for its migrations and a progress marker at 0, no threads in the state database, no rows in the logs database |
| wire | the request the CLI would send, captured locally (headers, which carry the token, never stored): exactly our eleven tools (claude: plus `StructuredOutput`; codex: plus code mode's `exec`, `wait`, `request_user_input_async`, with no tool reachable inside `exec`), our system prompt, none of: planted CLAUDE.md/AGENTS.md, home path, repository name or text, the CLI's own system prompt |
| auth | `claude auth status` / `codex login status` inside the CLI sandbox succeeds (output not kept) |

The result is a record in `<root>/verified/<backend>-<version>-<content-key>.json`
with the binary's sha256 and the tool environment's digest. The content key binds
the environment and runtime implementation, so a parallel checkout's gate cannot
replace a passing receipt for a running analysis.
`checks.require_verified`, called by every `Batch` (and `Batch.after_gate`
runs the checks itself where a record is missing), refuses a version
without a passing record, a binary whose hash changed, and a tool environment
that changed since the checks. To move to a new CLI version: change
`PINNED`, run `make agents-gate`, read the failures, then `make check-agents`.

## The API-key route (for measured runs)

`auth="api_key"` reads `ROBOTS_ANTHROPIC_API_KEY` (claude, passed as
`ANTHROPIC_API_KEY`) or `ROBOTS_OPENAI_API_KEY` (codex, passed as
`CODEX_API_KEY` and `OPENAI_API_KEY`) from the runtime's environment at call
time and hands it to the CLI through its process environment only: never a
file, never a command line, never `argv.json`. No login file is mounted on
this route. The prefix keeps a key that happens to be in a shell from
switching a run to paid billing by accident. Estimate the cost first from a
subscription run of the same batch: claude reports the list price of every
call (`usage.cost_usd_list_price`, `total_cost_usd` in its stream), and
`Batch.usage(results)` sums it with the tokens; codex reports tokens only, so
its estimate is tokens times OpenAI's price list, which is not in this
repository. **Not tried**: there is no API key on this machine.

## Measured (`make check-agents`)

<!-- check-agents:begin -->
Measured 2026-10-07 20:30 with claude 2.1.293, codex 0.161.0, by `make check-agents`; raw records in `check_agents.json`. Every call in the bwrap sandboxes, one at a time except the last table. "Process" is the wall clock of the whole CLI process; "answer" is when the event with the answer arrived.

| Model | Tier | Codex service tier | Trivial: answer / process (s) | Four frames: answer / process (s) | Correct | Tokens in / out (trivial; frames) |
|---|---|---|---|---|---|---|
| `opus` (claude-opus-5-5) | strong | - | 1.7 / 2.2 | 2.6 / 3.2 | yes / yes | 1,026 / 53; 1,353 / 63 |
| `haiku` (claude-haiku-5-5) | cheap | - | 1.7 / 2.3 | 1.4 / 2.0 | yes / yes | 1,051 / 53; 1,378 / 63 |
| `gpt-6.1-sol` (gpt-6.1-sol) | strong | default | 4.0 / 4.4 | 3.3 / 4.0 | yes / yes | 1,572 / 15; 1,908 / 21 |
| `gpt-6.1-sol` (gpt-6.1-sol) | strong | fast | 4.3 / 4.7 | 4.2 / 4.7 | yes / yes | 1,572 / 15; 1,908 / 21 |
| `gpt-6-luna` (gpt-6-luna) | cheap | default | 2.5 / 3.0 | 2.9 / 3.3 | yes / yes | 1,572 / 15; 1,908 / 21 |
| `gpt-6-luna` (gpt-6-luna) | cheap | fast | 2.3 / 2.8 | 2.5 / 2.9 | yes / yes | 1,572 / 15; 1,908 / 21 |

Sessions with tools and the toy skill (session 1 ends "waiting" with an experiment queued, the toy engine answers, a later session submits a diagnosis and ends "done"; an agent may request more experiments on the way):

| Model | States | Seconds per session | Tool calls | Skill read | Frames | Diagnosis names the right robot | Tokens in / out (all sessions) |
|---|---|---|---|---|---|---|---|
| `opus` | waiting, done | 13.8, 16.3 | 14 | yes | 2 | yes | 93,575 / 2,713 |
| `haiku` | waiting, done | 10.4, 20.0 | 20 | yes | 2 | yes | 313,830 / 4,676 |
| `gpt-6.1-sol` | waiting, done | 42.0, 13.9 | 12 | yes | 2 | yes | 146,163 / 1,057 |
| `gpt-6-luna` | waiting, done | 14.1, 11.6 | 14 | yes | 2 | yes | 189,889 / 756 |

Eight trivial calls started at once:

| Model | When | Answered | Whole batch (s) | Fastest / slowest call (s) | Calls per minute | Errors | Subscription window (claude) |
|---|---|---|---|---|---|---|---|
| `haiku` | 2026-10-07 20:30 | 8 of 8 | 3.8 | 3.5 / 3.8 | 125.0 | none | allowed |
| `gpt-6-luna` | 2026-10-07 20:30 | 8 of 8 | 6.7 | 2.9 / 6.6 | 72.1 | none | - |

The other model names, one trivial call each:

| Model | Reported as | Accepted and answered | Answer / process (s) | Tokens in / out |
|---|---|---|---|---|
| `fable` | claude-fable-5-1 | yes | 2.4 / 2.9 | 1,049 / 53 |
| `gpt-6-astra` | gpt-6-astra | yes | 3.5 / 4.2 | 1,572 / 15 |

Blinding verdict: **held**. The gate passed for claude, codex; 42 run and analysis directories searched for 16 terms of the toy run's truth (canary, fault kinds, trait names): 0 hits.
<!-- check-agents:end -->

Reading the numbers:

- **Every call and every analysis was correct**, on both vendors, strong and
  cheap, and eight at once on both: the protocol (waiting, toy engine, done),
  the skill read through the tool server, frames through `get_frames`, memory
  carried across sessions, a schema-valid diagnosis naming the robot the
  agent's own query found. This run (20:30) is on the code after the first
  review, with main merged; the two before it (18:34, 19:54) gave the same
  verdicts, except as below. In the 19:54 run Luna requested a second
  experiment in its second session and submitted in the third, which the
  protocol allows.
- **The blinding search** covers the toy run's canary and trait name and the
  kinds of `fleetlab.faults`, now merged (16 terms).
- **The subscription window.** The 18:34 run hit the account's five-hour
  limit on Claude's eight-at-once row: every call came back `rate_limit`
  (HTTP 429, `usage_limit_reached`). The runtime classified it correctly and
  did not retry; most of the window had been used by other sessions on the
  account. Claude's stream reports `allowed_warning` near the limit. Since
  then the ledger records the window per call, and a `Batch` stops starting
  calls above `stop_at_window` (95 %).
- `haiku` now resolves to **claude-haiku-5-5** (the spike had 4.5, slower than
  Opus) and `fable` to **claude-fable-5-1**; `gpt-6-astra` is accepted too.
  Codex does not report which model answered.
- **Codex's fast tier shows no consistent gain** over three runs, one call per
  cell: Luna was 0.1 to 0.9 s faster with `fast` in all six of its pairs;
  Sol's four-frame call was 0.1 to 0.9 s slower with `fast` all three times,
  its trivial call faster twice (by 0.8 s, and against a 9.0 s first-call
  outlier) and slower once. The user's own codex config holds `default`; the
  runtime passes the tier on every call and defaults to `default`. `flex` is
  refused by the service on the subscription.
- **Compared with the spike** (one call per cell, a busy machine, so within
  about 20 %): for the trivial call (answer / process) Opus 1.7 / 2.2 s
  against the spike's 1.1 / 1.6 s; the codex models 2.3 to 4.3 s to the
  answer against 2.5 and 2.7 s. Sessions took 10 to 42 s, the spike's single
  sessions 11 to 19 s; these do more (a skill, frames, memory, a queued
  experiment, a diagnosis). Eight Luna calls at once took up to 6.6 s each in
  this run (3.8 s at 18:34 and 19:54); the spike saw the same spread at eight.
- **Startup**: the tool server takes about half a second to start (its
  imports took 0.54 s inside the tool sandbox before the log package became
  lazy, which removes the writer's imports). The sessions' tokens are mostly
  cached input (`input_tokens` includes the cache reads for both vendors, as
  in the spike).

## What changed from the spike

- **No unsandboxed mode, no Docker mode.** Without a sandbox, codex needs a
  private home holding a link to the login file, the kind of link that once
  broke the CLI's login on this machine, and claude writes into `~/.claude`. Docker
  added 0.2 s per call and nothing the runtime needs.
- **No copy of, link to, or read of a login file anywhere.** The checks test
  the login file with `test -r` inside the sandbox; nothing reads its content.
  The spike's `scrub` of login tokens is gone with the shell variants that
  needed it.
- **The tool server is a package with a closure-built bundle**, run from a
  dedicated interpreter and package directory under the agent root, instead
  of a script mounted from the repository with the spike's venv in the home
  directory.
- **A hand-written MCP server** instead of the `mcp` package.
- **A codex home per concurrent slot** (`codex-homes/NN`, held by a lock
  file) instead of one shared private home, so that parallel calls do not
  share codex's databases (the spike saw slow calls at 4 and 8 and did not
  know whether that was the cause).
- **`service_tier` on every codex call**; `tool_timeout_sec` set for our
  tool server (120 s; the server interrupts a query at 60 s).
- **Answers validated with the pydantic models** of the contracts, not only
  the JSON schema.
- **The gate**: the spike's three free checks as functions with expected
  values, a source audit, and records that a batch requires.

## After the first review (2026-10-07)

No blocking findings. Applied: `query` SELECT only (S1); `memories` and
`shell_snapshot` off for codex and every slot home checked (S2); the agent
root refused unless a private directory of the user (S3); the tool sandbox's
bwrap started with a clean environment, with a probe of `/proc/1/environ`
(S4); the contracts' experiment budget enforced, 960 episodes per analysis by
default (S5); only `<run>/box/` mounted (S6); `installed_version` without
autoupdate or auto-memory (M1); calls to tools not offered are logged (M2);
the tool environment's digest from the interpreter binary (M3); the slot's
`auth.json` checked after every codex call (M4); submitted-then-failed is
"done" (M5); the log package imports its writer, export and leak search
lazily, so the sandbox holds only the reader (M6); the codex isolation check
fails without a home (M7).

## Choices made where the contracts are silent

- The session-end answer (`SessionReport`) and the queue format
  (`ExperimentQueue`, `EngineReply`) live in this package; the engine imports
  them from `fleetlab.agents.sessions`. A move to the contracts is proposed.
- A session's state comes from the outbox, not from the agent's `status`.
- `query` allows one SELECT per call, so that a query's result depends on the
  export alone and the scorer can re-run it.
- The experiment budget counts `episodes_per_arm` times arms; besides the
  contracts' 8 requests and 60 per arm, a cap of 960 episodes per analysis
  (8 times 2 arms times 60) that a caller may raise.
- A session that submitted and then failed counts as "done" and keeps its
  error.
- `request_experiment` refuses the two verbs without a simulated effect.
- Experiment ids are numbered per analysis in request order.
- The run directories, analyses, ledger and gate records live under the agent
  root; a caller can put run directories and analyses elsewhere
  (`run_root`, `Analysis.create(base=...)`).

## Not verified

- **A token refresh inside a sandbox**, for either CLI. No refresh happened
  during these runs: both login files kept their size and modification time.
  Until it is tested, keep sandboxed sessions short (one CLI call per session is short) and treat an `auth`
  error as "stop and log in again", which `Batch` does.
- **The API-key route**: no key on this machine. Which flags or variables
  codex 0.161.0 needs for an API key in `exec` mode beyond `CODEX_API_KEY` /
  `OPENAI_API_KEY`, and whether the model names are the same on the API, is
  open.
- **A real engine**: the queue was answered by a toy engine that names logged
  episodes as the arms' episodes; the rollout engine has not
  read `ExperimentQueue` yet, and no export has grown between sessions.
- **Size**: the toy export has ten episodes. `query` materializes every visible
  table at the session's first tool call; for a large export that costs
  memory and seconds per session, not measured.
- **Rate limits beyond the one observed**: Claude's five-hour limit was hit
  once and classified correctly; how long a batch may run before it, the
  seven-day window, and Codex's limits (its stream reports none) are not
  known. More than eight calls at once and sustained load are not measured.
- **The miniature fleet**: everything ran on the toy export of
  `tests/log_toy.py` (ten episodes, 16-pixel frames); the miniature fleet's
  export does not exist yet.
- **The free checks in `make check`**: the layout check and the codex-home
  check run there; the static probes, the wire check, the isolation check and
  the login check start the CLI binaries or mount the login file, so they run
  in `make agents-gate` and before every batch, not in `make check`.
- **An inventive attack** on the final layout through a model. The static
  check (no model) is the wall's evidence; the measured tier only checks that
  ordinary analyses leave no truth term in their run directories. The spike's
  attack runs used its own layout.
- **That codex's `exec` isolate has no file or network access**: its own
  description says so, and the bwrap layer does not depend on it. The wire
  check asserts that no tool is reachable inside it.
- **Skill helper scripts**: served as text; no tool runs them.
- **Network filtering** of the CLI sandbox: the network is open
  for the CLI, closed for the tool server.
- **The warm session** (one CLI process per episode) for the live layer:
  the spike measured it for claude without tools; not part of this runtime.
- **Several sessions of one analysis at once**: the runtime runs one session
  per analysis at a time and numbers experiments in request order; two
  sessions on one analysis directory would race.
- **Protocol versions** of MCP other than the ones the two pinned CLIs ask for.

## Stage-4 launch and reply boundary

Tool-bearing `run_agent` calls now require `ToolAccess.audit`, an `AuditContext`
with the full host-side fleet and contributing runs. `run_session` and
`run_analysis` take the same `audit=` argument; `Batch.after_gate` supplies it
to analysis jobs. An optional experiment root adds newly completed runs to the
canary set. These objects are never mounted into either sandbox.

Immediately before each launch, including continuations and retries, the runtime
checks the pinned CLI gate, the selected Codex home's isolation, structural
export validity, public fleet identity, current source and skill bytes, system
and user prompts, answer schema, the prepared CLI box, notes, outbox and results.
It hashes the mounted data before and after the scan and writes `audit.json`
beside the box. Missing context, changed inputs and failed audits raise before
execution. CLI/environment overrides are rejected for analysis calls. CLI gate
records bind actual environment bytes and the audit/sandbox implementation.
The local wire-capture bootstrap has no production bypass flag.

The scanner always checks trees and mounted code strictly. `source_audit`
passes prompts, schemas and tool descriptions as `texts`, and source/skills as
`code`. `audit_state` first scans the complete state tree, then replaces findings
only at recognized agent prose locations using separately sourced `agent_texts`:
`notes.md`, request power notes, diagnosis descriptions/queries/warning metrics,
retraining prose and fix rationales (including submission copies), plus the
public contract's text-valued training-data and teleoperation descriptions in
answers. Each snippet names its original file and field; a JSON-string wrapper
preserves checks of nested JSON. The temporary tree for this second scan is
empty. No selector can make a tree or mounted code lenient.

Extraction follows exact schema paths; an envelope field with the same name is
not agent prose. Filenames, keys, configuration/condition values, extra files,
results and the launch box remain strict. There is no public-word masking.

In `agent_texts`, seeds, canaries and non-countable fault ids refuse, as do
identifier spellings of planted kinds, pattern roles, traits and hidden fleet
properties, including within longer identifiers. Public text passes. Actual
fault ids of one ASCII letter and two digits are countable and do not refuse
there; strict trees/texts/code still check them. Natural-word hypotheses,
record-layout names and single-word hidden names are outside this tripwire;
acceptance does not prove that no truth was learned. The legacy
`measure.blinding` diagnostic keeps its mixed archives as strict trees, since
they also contain pipeline prompts and schemas.

The final scanner API integration still has a recorded acceptance mismatch:
the saved statistical shorthand `min_detectable` refuses as an unknown protected
name. No caller-side mask was restored. Round 18 of
`skills/ROUNDS.md` records the failing regressions and the focused check results;
this integration does not yet establish that all stored agent states launch.

This assumes trusted host code does not modify inputs between the completed
check and process launch. The receipt is a content-bound scan, not protection
against a malicious concurrent host writer. Login stores are not scanned or
changed. Calls without tools do not carry a fleet export and retain their
existing generic runtime path.

Experiment refusals expose only a fixed public code from `PUBLIC_REFUSALS`
(`toolserver/public_results.py`): the preflight's codes, `exceeds_robot_day_capacity`
for a request past a robot-day's 160 experiment-arm episodes (public layout),
and `request_not_runnable` for any other engine exception;
success exposes `done`. Detailed exceptions stay in the engine's host log. Cached plans bind the exact
queue, replies must cover its requests, and result files are parsed strictly.
Before mounting and again before returning results, counts, unique arm episodes,
experiment membership and requested robots are checked. Earlier requests must
have results before a continuation. The lightweight `check-agents` fixture now
measures the refused-request/session protocol: it does not invent new A/B
outcomes from its five recorded episodes per robot. Successful queue execution
is covered separately by the toy-engine integration tests and real analyses.

The tool closure no longer contains identity/seed, oracle, world or rollout
contracts. Package-root contracts load lazily, preserving their public names.
Runtime changes are in `audit.py`, `batch.py`, `checks.py`, `measure.py`, `run.py`,
`sessions.py` and `toolenv.py`. The tool server records request starts and native
stderr beside its normal tool log, to diagnose a lost transport. A mid-analysis
transport loss stops the analysis rather than automatically repeating it.

The shared `run_analysis` loop also stops on `auth` or `rate_limit` even when
the session queued requests before failing. It preserves those
requests but starts neither their simulation nor another model call. The
development runner already enforced this rule; CPU regressions cover the
generic runtime path without accessing a login or making a model call.

Nested tool sandboxes omit `--die-with-parent`: Linux's parent-death signal
follows the thread that spawned a process, and a short-lived CLI spawn thread
can kill a healthy tool server. The outer CLI sandbox retains that flag and
its PID namespace cleans up nested descendants. The free `lifetime` gate
reproduces the thread-death failure as a negative control, verifies tool calls
after thread exit, and verifies descendant cleanup when the outer sandbox ends.
This repair changes `sandbox.py` and `checks.py`; live verification is recorded
in `skills/ROUNDS.md`.

Verification through `4433cc6`: `make check` exited 0 (1,200 passed, two skipped)
and the Codex-only free gate passed. Two real D1 logs-only analyses completed
93 and 98 successful tool calls with current launch receipts, verifying the
transport repair on the subscription path. Both saved experiment states pass
the launch-state audit; successful A/B delivery is still waiting on the heavy
queue. `make check-sim` at final immutable `f77ed68` passed 58 tests. The queued
`ecf30ef` smoke exited 0 with all 16 chains identical; simulator sources are
unchanged between that checkout and final code, and no reference was updated. Detailed
run provenance and provisional scores are in `skills/ROUNDS.md`.

### Review-repair launch contract

The launch audit understands field/line locations from the repaired export
scanner. The surface policy above is confined to known agent prose values;
structural keys and other files remain strict. An export containing experiment rows requires explicit experiment
day bounds in `AuditContext`; every arm row must lie inside them, including
uncited rows. The current-content scan's duration is recorded even on failure.

`build_tool_env(ablate_fault_classes=True)` creates a separate content-addressed
library with marked class-knowledge passages stripped. Unit tests drive the
common launch audit with the real source bundle and that library, stubbing
execution before any model call. This establishes source/skill audit compatibility,
not a new CLI wire-gate or a live model result. No agent session is run in this
repair round; `Batch.after_gate` still requires the appropriate current gate
before the later scheduled analysis.

The development pipeline passes a host-only `on_start` callback through
`run_session` and `run_agent`. `_execute` invokes it after the CLI process has
been created and before writing task input. This is the repeat-registration
boundary: a failed gate, audit or process creation invokes no callback. Callback
failure kills and reaps the process before propagating. `started.json` records
that dispatch outside the mounted box. Unit tests use a plain Python stdin
process to check this ordering; no model or login is involved.

The class-knowledge ablation removes only the hypothesis block in
`episode-cards`; all general analysis method and public action semantics remain.
The fixed repeat register and continuation checks are described in the integration
README. A fresh `make agents-gate` is deferred to the next
full run; the synthetic launch checks do not certify the current CLI wire.
