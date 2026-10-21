---
name: write-diagnosis
description: Write and submit the round's answers from what the earlier skills found, with every number traceable to a tool result. The decision (at most three adopted changes or none, undos and replacements of earlier ones, one forecast for the whole set with a range, what the next round should show, whether to stop, and optional ranked alternatives) and the diagnosis behind it (causes, mode summaries, warnings, retrain indicators).
when_to_use: At the end of a round, once rank-actions has the ranked list, the chosen set and its forecast. Also to submit a provisional answer before a session that waits for experiments ends, if the run asks for one.
tools_used: [query, memory_read, memory_write, submit_diagnosis, submit_decision, submit_fix_set]
inputs:
  - the notes in memory: modes, scopes, causes, gains, the ranked list, the chosen set and its forecast, warnings
  - from the second round on, the earlier decisions and their forecasts and expectations (the task and memory)
outputs:
  - the round's decision, submitted and accepted
  - the diagnosis, submitted and accepted when the task asks for it
  - a final note in memory for the next round
version: "7"
---
# Write the round's answers

Your task names the answers to submit and the tool for each. In a round of
the improvement loop that is the decision (`submit_decision`) and, when the
task asks for it, the diagnosis (`submit_diagnosis`). Each tool checks the
answer against its schema and refuses what does not fit, with a list of
problems; fix them and submit again. Where this text and a tool's schema
differ, the schema decides.

## Rule for numbers

Every number you write (a gain, an interval, a forecast, a confidence aside,
a day, a rate in a summary) must appear in the result of a tool call of this
analysis, so that anyone can trace it. Copy numbers as the blocks returned
them. Never compute one in your head: if you need a product or a sum, run it
through `query` (`SELECT round(100 * 0.29 * 0.167, 2) AS gain_points` is a
valid query and leaves a trace).

Compute exposure limits and other physical bounds from full-precision counts,
not a rounded display share. Apply any physical clipping before the final
reporting step, then round the interval outward in `query`: lower endpoints
with `floor(100 * low) / 100` and upper endpoints with `ceil(100 * high) /
100`. The reported interval must contain the unrounded clipped interval.
Point estimates may use ordinary rounding.

## The decision

- `adopt`: the new changes the fleet takes on now. Together with the
  replacements below they are at most three, and none is a valid answer.
  Each is `{"action": ..., "rationale": ...}`. The rationale says in a
  sentence or two which cause the change removes or which lasting condition
  it adapts to, where its gain comes from (block and experiment ids), and
  what could go wrong.
- `replace`: earlier changes swapped for another action, each
  `{"entry_id": "L<n>", "action": ..., "rationale": ...}`; each counts
  against the three.
- `undo`: earlier changes taken back, each `{"entry_id": "L<n>",
  "rationale": ...}`; an undo does not count against the three. The
  rationale says what the records since the change showed against what was
  predicted for it. Only settings and conditions can be undone or replaced:
  a component verb is physical work and cannot be taken back.
- `predicted_set_gain`: `{"low", "mid", "high"}`, the gain of the whole
  decision in points of fleet success over the fleet as it now operates:
  the `set` row of rank-actions' `set_gain` (`low`, `gain`, `high`), never a
  sum of single gains. With no change it is zero.
- `expectations`: a few concrete statements about the next round's records
  that would show the set works, each one a query could check: which robots
  or site should rise and to about what, which symptom should become rarer,
  which measurement should move (for example the lag between commanded and
  measured motion at a site). Also what would show it did not.
- `stop`: true when, after this decision, nothing further is worth changing
  (rank-actions, "When no further change is worth making"). It is a result
  like any other; say in the adopted changes' rationales or in memory why.
- `ranked_candidates`: optional alternatives, the further actions you
  weighed and did not adopt: the rows of rank-actions' list below the
  adopted ones, best first, with the same numbers as its `cut` result. None
  is valid, and a decision to change nothing or to stop needs none. Adopted
  and replacement actions are not repeated here. List only actions that
  could be adopted (rows of the catalogue, values filled, a simulated
  effect), checked through `cut` like the changes, with their honest gains,
  never invented ones. Round one's adopted actions in order, then these
  alternatives in order, cut at the number of changes in force at the end,
  are the plan fixed after a single look, so order them as you would adopt
  with one look.

An action is `{"verb", "target": {"scope", "component", "config_key",
"condition_key"}, "value"}`. `value` is null except for `set` and
`change_condition`; a `set` back to an earlier value takes the change log's
`old` with its JSON decoded: `0` stays the number `0`, `"1.4.0"` becomes the
string `"1.4.0"` in the answer's JSON (not `"\"1.4.0\""`). Where the change
is in the log, the `revert` at its scope is the safer action. A ranked
candidate is `{"action", "expected_gain", "uncertainty": {"low", "high"},
"rationale"}`, its gain the action on its own. Every action is a row of the
table `catalogue` (rank-actions' `cut` reports any that is not).

## The diagnosis

What you believe and why, for the record behind the decision.

- `causes`: one per supported target and direction. Unresolved alternatives
  belong in `mode_summaries` and memory, with the observation that would
  distinguish them. A low confidence is not an abstention flag.
  - `target`: `scope` (a robot id, a site id or `fleet`) and at most one of
    `component`, `config_key`, `condition_key`; the others null. Name the
    narrowest scope the evidence supports.
  - `direction`: a component that works worse is `degraded`; a wrong
    configuration value `misconfigured`; a change that came with an update
    or a version `changed`; anything that worsens day by day `drifting`.
  - `confidence` from 0 to 1, meant as your probability that the cause is
    real: about 0.9 when an experiment confirmed it; 0.7 to 0.8 when the
    rates, the symptom, the moment and the class of fault agree and nothing
    is confounded with it. If two candidates remain indistinguishable, say
    so in the mode summary rather than asserting both. Name what the
    evidence supports, with the confidence it supports, no more and no less.
  - `evidence`: at least one table reference and one or two episodes. A
    table reference is `{"episode_id": null, "step": null, "table": <the
    main visible table the query reads>, "query": <the exact SQL you ran,
    with its helper line>}`. An episode reference is `{"episode_id": <id>,
    "step": <the step of the evidence frame or event>, "table": null,
    "query": null}`.
  - `failure_mode`: the mode's name, as in the mode summaries.
- `mode_summaries`: one sentence per mode: its name, fingerprint, how many
  failures it accounts for and where, and the cause you assign (or none).
- `warnings`: for every scope that drifts towards failure (scope's `trend`
  block: for a robot, a clearly negative `within_site_slope`, so that a step
  of its whole site is not taken for its drift; for a site, a negative
  slope that `breakdown` and `changes` do not show as a step on one day):
  `scope` is the robot or site id, `metric` says what will cross (`success
  rate`, or a health metric such as `scene_camera.calibration_residual`),
  `expected_day` the day the block's `crossing_day` gives (the first day the
  scope's success falls five points below the rest of the fleet),
  `confidence` how sure you are of the trend. A warning only helps before
  the day, so issue it even when the fit is modest, with a matching
  confidence. A `crossing_day` inside the observed window (or below zero)
  means the scope has crossed already: name it as a cause, and give the
  warning the first day after the window.
- `retrain`: for every mode for which the evidence supports no available
  catalogue action (check the scope comparisons in failure-modes and scope;
  a broad or steady spread alone is insufficient): `failure_mode`, three to
  ten `episode_ids` of that mode from robots with no other cause,
  `why_no_action_fixes_it` (in a sentence: where the mode occurs and why no
  repair or setting reaches it), and `data_wanted` (which situations the
  policy needs more examples of).

## If the task asks for a ranked fix set instead

An older answer form, `submit_fix_set`, takes only a ranked list: at least
ten distinct catalogue actions, each with its value where the verb takes
one, best first, each gain the action on its own with its interval. Give
it rank-actions' list with the same honest gains; when your strong
candidates are fewer than ten, the rest are the plausible alternatives and
next suspects with their honest, often zero, gains.

## Procedure

1. `memory_read`. Gather causes, modes, gains, the ranked list, the chosen
   set and its forecast. If a number you need is not in a tool result of
   this session, re-run the block that produced it (the notes say which).
2. From the second round on: for each earlier change, set what the records
   now show beside the forecast and the expectations written for it, and
   say in memory whether each expectation was met, missed, or cannot yet be
   told. This is where an undo or a replacement gets its reason.
3. Write the diagnosis, if the task asks for it. Check: every cause has a
   target that names a scope that exists, a direction, evidence with the
   exact query text; every warning names a robot or a site, not the fleet;
   every retrain indicator lists episodes that exist. Submit it, fix and
   resubmit on refusal.
4. Write the decision from rank-actions' `cut` and `set_gain` results (same
   order, same numbers). Submit it, fix and resubmit on refusal.
5. `memory_write` a short final note for the next round: the receipt ids,
   the changes adopted, undone and replaced, the set forecast and the
   expectations, the hypotheses still open, and what you would look at
   next with more budget.

## Shape of the answers

Placeholders in angle brackets.

```json
{"adopt": [{"action": {"verb": "replace", "target": {"scope": "<robot id>",
   "component": "gripper", "config_key": null, "condition_key": null}, "value": null},
   "rationale": "..."}],
 "replace": [{"entry_id": "L1", "action": {"verb": "set", "target": {"scope": "<site id>",
   "component": null, "config_key": "<key>", "condition_key": null}, "value": "<value>"},
   "rationale": "..."}],
 "undo": [],
 "predicted_set_gain": {"low": 1.9, "mid": 4.4, "high": 7.2},
 "expectations": ["<robot id>: success rises from about 0.55 to 0.8 or more; carry drops become rare"],
 "stop": false,
 "ranked_candidates": [{"action": {"verb": "service", "target": {"scope": "<robot id>",
   "component": "gripper", "config_key": null, "condition_key": null}, "value": null},
   "expected_gain": 2.1, "uncertainty": {"low": 0.4, "high": 3.8}, "rationale": "..."}]}
```

```json
{"causes": [{"target": {"scope": "<robot id>", "component": "gripper", "config_key": null,
   "condition_key": null}, "direction": "degraded", "confidence": 0.8,
   "evidence": [{"episode_id": null, "step": null, "table": "episodes",
     "query": "-- helper: gains\nWITH params AS (...) ..."},
    {"episode_id": "<episode id>", "step": 7, "table": null, "query": null}],
   "failure_mode": "carry: the object is lost after a good grasp"}],
 "mode_summaries": ["carry: the object is lost after a good grasp; ..."],
 "warnings": [{"scope": "<robot id>", "metric": "success rate", "expected_day": 8,
   "confidence": 0.6}],
 "retrain": [{"failure_mode": "...", "episode_ids": ["<episode id>"],
   "why_no_action_fixes_it": "...", "data_wanted": "..."}]}
```
