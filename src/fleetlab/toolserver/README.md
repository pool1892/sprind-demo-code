# Tools over a fleet export

The MCP server exposes SQL queries, video frames, system description, skills,
analysis memory, experiment requests/results and schema-checked submissions.
It runs inside the runtime's network-free sandbox with only the exported data,
public source bundle and per-analysis state. SQL is limited to SELECT over
materialized visible tables with external file access disabled.

`public_results.py` defines the public experiment reply boundary. Only fixed
completion/refusal messages are accepted; stored JSONL must contain valid
objects. Replies are checked against the persisted request and visible episode
records before being returned. Detailed engine exceptions never belong here.
The runtime repeats these checks before mounting results for a later session.

`tool_calls.jsonl` records returned content. `tool_requests.jsonl` records starts,
so a server death leaves the last attempted call. `server.stderr` captures native
and Python diagnostics in the same host-visible tool-log directory. These are
runtime diagnostics, not tools offered to the analysis agent.

The first two D1 development analyses lost this transport during triage.
Replaying the query and its complete preceding tool sequence succeeded through
the original sandbox bundle without a model. A focused probe reproduced a
SIGKILL when the spawning thread exited with `--die-with-parent`; the runtime
now uses the outer sandbox's PID namespace to clean up nested tools. Its free
gate checks both thread survival and cleanup. The first two successful live
repeats completed 93 and 98 tool calls; subsequent development rounds also
completed. Their records and errors are in `skills/ROUNDS.md`. The CPU tests
exercise tool contracts and refusal validation. Successful end-to-end A/B
delivery remains pending; the live logs analyses do not establish it.

### Review repairs: public experiments and system glossary

`action_validation.py` shares the public experiment preflight with the engine.
It checks every arm, rejects identical action sets, missing visible revert
history, unsupported/invalid setting values, absent targets and actions covering
none of the requested robots before the request is stored or charged. Domains describe executable
settings, not response curves. Precision is fixed while a policy is loaded;
new camera-pose/software settings are not supported by this runner, but visible
recorded reverts remain available. `describe` exposes these limits and the
controller's outcome-detector glossary through `public_system.py` for every
export without rewriting stored descriptions.

Engine refusal codes are allow-listed, explained in `describe vocabulary`, and
release request/episode budget while
preserving the request identifier. Exception details stay on the host. A refused
queue does not terminate analysis: the next session can read it and submit an
answer. Synthetic request and result tests cover these contracts; no full
world experiment analysis was measured in the review-repair round.

### Public system semantics and experiment value domains

`describe(dictionary)` adds the placement completion criterion and explains
that a recording error changes the recorded outcome, not the robot's physical
completion. `describe(vocabulary)` includes the controller's outcome detector
and recorder, plus the current executable action domains. The static additions
are in `public_system.py`; stored descriptions and dictionaries stay unchanged.

Site-condition forms follow the registered application hook: lighting is signed
stops, `[stops]` or `[stops, warmth]`; placement is `[x,y]` or `[x,y,z]` in
centimetres; object batch is `[mass doublings, friction halvings]`; clutter is
a blend from zero to one, as a number or one-number list. Numbers must be finite.
The regression resolves each domain through the registered condition hook's
`new_episode` and checks the public values in the retained development export.
It tests application shapes and units, not a dose-response curve or simulation.
