# Jobs: the interface note

*Lane K, 19 October 2026, from the demo's agreed interfaces and lane J's contracts as
they stand on `demo/jobs` at `985582f` (`fleetlab.contracts.jobs`, the job
tables in `fleetlab.contracts.log`). Kept in step with both as the lanes change
them; a change goes through the integrator first and then here. Section 1 is a
faithful copy of the plan's interfaces; sections 2 to 5 give the concrete
forms the base skills (`skills/jobs-*`) and the task text
(`fleetlab.jobs.task_text`) are written against. Where the code says
otherwise, the code decides. Items still open between lanes are marked
**open**.*

## 1. The interfaces as the plan fixes them

```
RequestState   host-owned: request text, requested goals [{predicate, objects}],
               base task, order rules [{first: goal, then: goal}], budget
               {starts per request <= 20, requests per session <= 4,
               sessions <= 6, starts per investigation <= 400},
               acceptance {N, K} once approved, message log
JobStep        kind: policy | motion | wait
               policy: instruction (free text), until: {goal: <index into requested
                       goals> | steps: N}, max_steps (<= 300)
               motion:  one ProbeSegment (contracts.probes), bounds as there
               wait:    steps (<= 100)
JobProgram     name, purpose, steps[] (<= 8), notes; digest = sha256 of the
               canonical JSON
StartIdentity  base_task, init_state, scene_seed, noise_seed
JobRequest     program, starts [StartIdentity] (development allocation only;
               the runtime refuses release indices), record_video
JobEpisode     job_id, episode_id, StartIdentity, delivered, per requested goal:
               first step it held and whether it held at the end, per program step:
               outcome (reached | timeout | skipped), steps used, order-rule
               verdict, end state hash
JobTables      job_episodes (keys: job_id, episode_id); job_step_events (episode_id,
               program step, kind, start control step, end control step, outcome);
               object_steps (episode_id, control step, object, position, quaternion,
               linear speed, angular speed, in contact with robot: bool, in contact
               with: names); goal_steps (episode_id, control step, goal index, holds);
               robot_steps (episode_id, control step, end-effector position and
               orientation, gripper opening, gripper command, instruction in force,
               program step); frames reachable through get_frames by episode_id and
               control step; the schemas registered with the LogReader so query
               sees them
DeliveryCheck  host-fixed; reads requested goals from RequestState; thresholds in
               one place; unit-tested on constructed states
CheckSpec      name, purpose, parameters {name: type}, sql, expects (text);
               run_check(name, run, params) -> rows
ReleaseReceipt program digest, release starts, delivered count of N, naive count
               on the same starts, rescued and broken, order-rule verdicts
Tools          submit_acceptance_proposal(N, K), submit_job_program (queued; runs
               after the session like experiments), job_results, save_job,
               write_check(CheckSpec), run_check, job_library, operator_state;
               existing: query, get_frames, list_skills, read_skill, memory_read,
               memory_write, write_learned_skill
Library        jobs/<name>.json, checks/<name>.json, recipes/<name>.json, index.json
Operator CLI   python -m fleetlab.jobs run <job> --starts release|dev --n N --video
               python -m fleetlab.jobs check <check> --run <run dir> [--param k=v]
ProbeRecipe    name, purpose, request template (ProbeRequest with robots as
(story B)      parameters), analysis sql; python -m fleetlab.loop check <recipe>
               --world <world> --robots r01,r05 runs it as a light probe run
Results        ~/.cache/robots-probes/demo/2026-10-20/<lane>/<run>/
```

The runner keeps one simulator for the whole job, resets it from the start
identity, calls the policy with the step's instruction, drops the remaining
action chunk at every step boundary, evaluates every requested goal every
control step with LIBERO's own predicate functions on the loaded scene, logs
the five tables, writes the scene video, and evaluates the delivery check at
the end.

## 2. The job tables (lane J, `contracts.log`)

All five are visible tables of `query` (DuckDB), in a job run's tree only.
Row `step` = k of a per-step table is one instant, k control steps after the
reset, at 20 control steps a second: the state is measured first and then
the row's command is issued; a command acts during the next control step. The
last row is the state after the last command and has no command. The per-step
tables have one row per control step from 0 to `job_episodes.control_steps`
and join on `(episode_id, step)`. Vectors are fixed-length lists indexed from
1 in SQL. Scene names are LIBERO's. Job ids are the runtime's (`j0001`), not
one of the fleet's neutral id kinds; episode ids are unique across job runs.

**Start identity** (`contracts.jobs.start_identity(base_task, init_state)`):
`base_task` is the goal-suite task number in `tasks_info.txt` order (0 to 9)
and `init_state` LIBERO's stored initial state; the scene and noise seeds
derive from the initial state, so `(base_task, init_state)` names a start.
Development starts are initial states 0 to 29, release starts 30 to 49; a
`JobRequest` refuses release starts and names starts of one base task only.

**The delivery tail** (`DELIVERY.tail_steps`, 60): after a program whose last
step is a policy step that reached its goal, the policy goes on under that
step's instruction until the delivery check holds or 60 control steps have
passed, the same for every program. It shows as `phase` `tail`, a
`job_step_events` row of kind `tail` with an empty `program_step`, and
`job_episodes.tail_steps`.

**The delivery check** (`DELIVERY`): every requested goal holds; no robot
body touches an object the goals name; those objects move slower than
0.01 m/s and 0.15 rad/s; the last gripper command is open and the fingers are
at least 0.06 m apart.

**The program vocabulary** (`JobStep.kind`): `policy`, `motion` and `wait`
as in section 1, and `home` (lane J, `0c00b85`): with `duration_steps` (at
most 200) and `on_timeout`; the gripper commanded open, the target ramps to
the pose recorded right after the reset at no more than 0.5 m/s, then holds;
`reached` when the hand is within 1 cm of the start position and slower than
0.01 m/s (`HOME`), else `timeout`. The start pose is in `job_episodes`
(`start_eef_pos`, `start_eef_quat`).

### `job_episodes`: one row per run of a program from one start

`job_id`; `episode_id`; `program_digest` (SHA-256 of the program);
`base_task` (integer, 0 to 9); `init_state`; `start_eef_pos` and
`start_eef_quat` (the hand's pose when the job started; nullable);
`control_steps` (from the reset to
the end); `tail_steps`; `delivered` and `delivery_detail` (the verdict in
words); what the delivery check found at the end: `goals_held_at_end` (one
boolean per requested goal, in order), `robot_touching_at_end` and
`moving_at_end` (lists of the named objects), `gripper_opening_at_end`
(metres); `order_kept` and `order_detail`; `video_scene` and `video_wrist`
(paths relative to `log/`, empty without video). An agent reaches it with
`query` after `job_results` has named the job ids; the program's name is in
`job_results` (the table names a program by its digest only). Two episodes
had the same start when `base_task` and `init_state` agree.

### `job_step_events`: one row per program step of each run, and one for the tail

`episode_id`; `program_step` (from 0; empty for the tail); `kind` (`policy`,
`motion`, `wait` or `tail`); `start_step` and `end_step` (control steps; empty
when skipped; the next step starts at the previous step's `end_step`);
`steps_used`; `outcome` (`reached`, `timeout`, `skipped`; for the tail,
`reached` when delivered within it); `instruction` (of a policy step or the
tail). After a policy step's timeout, `on_timeout` `stop` skips the later
steps and `continue` runs the next one.

### `object_steps`: one row per control step and movable object

`episode_id`; `step`; `object` (the scene's name); `position` (world frame,
metres); `quaternion` (w, x, y, z); `linear_speed` (m/s); `angular_speed`
(rad/s); `in_contact_with_robot`; `in_contact_with` (every body touching the
object, the robot's bodies by their own names). Fixtures are not rows here;
their goals are in `goal_steps`. The angle between two orientations of one
object is `2 acos(abs(q1 . q2))`.

### `goal_steps`: one row per control step and requested goal

`episode_id`; `step`; `goal` (the requested goal's index, from 0); `holds`
(LIBERO's own predicate of the goal on the loaded scene). A goal's first held
step is `min(step) WHERE holds`; `goals_held_at_end` gives its end.

### `robot_steps`: one row per control step

`episode_id`; `step`; `eef_pos`; `eef_quat`; `gripper_opening` (metres);
`commanded_action` (seven numbers: position, rotation, gripper; empty in the
last row); `gripper_command` (-1 open, 1 closed; empty in the last row);
`instruction` (in force; empty during motion and wait steps); `program_step`
(the step issuing the command; empty in the tail and the last row); `phase`
(`program`, `tail`, or `end` for the last row); `frame_index` (the step's
frame in the videos; empty on steps without one, one frame every four control
steps by default).

### Frames

`get_frames(episode_id, steps, camera)` serves a job episode's frames when it
has video; the skills ask for steps that have a `frame_index`. **Open**
(lane T): whether `get_frames` also takes a step without a frame and serves
the nearest one.

## 3. Tools as the skills and the task text use them (lane T)

| Tool | What the skills rely on |
| --- | --- |
| `operator_state` | the request as mounted: `request_text`; `requested_goals` with `index`, `goal` (LIBERO's spelling), `predicate` and `objects`; `base_task` with its number and task; `order_rules` with `earlier_goal`, `later_goal` and `in_words`; what delivered means and the delivery check's thresholds; the development starts and how many release starts are reserved; the approved acceptance; the budget and what is left; the `message_log` (`author`, `words`, `session`); the skills the policy has, each with its goal-suite task, instruction and the goal it achieves on its own; and the release receipt once the release has run |
| `submit_acceptance_proposal(N, K, reason)` | N at most the 20 release starts; the scripted lead's rule (`acceptance_problems`) is the host's and is not stated in the skills or the task text |
| `submit_job_program` | `program` (`JobProgram`), `starts` (development starts as stored initial-state numbers) and `record_video`; refused before an approved condition, beyond the budget and after the release; queued, runs after the session |
| `job_results` | the requests so far with the agent's name and purpose for each program, their status, job id, episodes, deliveries and order-rule violations |
| `write_check(check, check_with)` | `check` a `CheckSpec`: `name`, `purpose`, `parameters` as a list of `{name, type}` with types `int`, `float`, `str`, `bool`, `sql` with `$name` parameters, all declared and all used, `expects`; `check_with` the values the SQL runs with once, not stored |
| `run_check(name, job_id, params)` | a saved check over one job's episodes, or over every job run so far with `job_id` null |
| `save_job(name, job_id, purpose, check)` | saves the program of a job that ran, with the name of its check (nullable); the job saved last in a session is released after that session on the reserved starts and its receipt appears in `operator_state` in the next session; one release per investigation, and no programs or saves after it; refused when a recorded run of the program broke an order rule |
| `job_library` | the installation's job list: saved jobs, checks and probe recipes, from earlier requests when a library is mounted and from this investigation |
| `submit_job_account(job, account)` | the account to the shift lead; after the release it names the released job, with `job` null it is the valid outcome of stopping without a release; it ends the investigation and is refused while a release is pending |

**Names in mounted state.** The launch audit scans the mounted files strictly
and refuses truth-side words as keys or file names (`text`, `first`, `jobs`,
`index`, `entries`, `pairs`, `rescued`, `broken`, `run_dir`). The mounted
request therefore writes the order rules' goals as `earlier_goal` and
`later_goal` and a message's text as `words`; `contracts.jobs` and the
integrator's request file keep their own names (`first`, `then`, `text`). The
skills name state fields in the mounted form. The request fixtures of gate 1
(`tests/fixtures/jobs/job-one.json`, `job-two.json` in the contracts' form,
validated with `RequestState` at `demo/jobs` `48e7288`; `*.mounted.json` in
the mounted form with the policy's skills, from lane T's `public_request`)
carry `acceptance` null: the scripted lead's minimum fraction is set at
launch.

## 4. What `task_text` reads of the request state

`fleetlab.jobs.task_text.task_text(state, session, library_mounted)` takes a
`RequestState` (`contracts.jobs`) or anything with the same attributes or
mapping keys, nested items too:

| Attribute | Form |
| --- | --- |
| `request_text` | the shift lead's request in their words |
| `requested_goals` | list of `{predicate, objects}`; rendered as LIBERO spells them (`On <object> <object>`), with the index |
| `base_task` | the goal-suite task number; rendered with the task's name from `GOAL_SUITE_TASKS` when `fleetlab.contracts.jobs` is importable |
| `order_rules` | list of `{first, then}` (the contracts) or `{earlier_goal, later_goal}` (mounted), goal indices; a goal itself is also accepted |
| `budget` | `{starts_per_request, requests_per_session, sessions, starts_per_investigation}` |
| `acceptance` | `{N, K}` once approved, otherwise empty |

`session` counts from 1; a later session's text starts with what to read
first and repeats the whole task. `library_mounted` adds the paragraph on an
earlier job's library. The text names no job, no condition and no difficulty.
Everything it does not take from the state (the policy's skills, the
development starts, the delivery thresholds, the shift lead's replies, a
release receipt) it sends the agent to `operator_state` for, so that those
facts have one source.

## 5. Open points between the lanes

- The roster test `tests/test_skills_library.py` admits only the version-1
  skills under `skills/` and only `SKILL.md` and helpers in a skill directory;
  the `jobs-*` skills need a jobs role in `fleetlab.skill_library`
  (`fleetlab.jobs.task_text.JOB_SKILLS` lists them in order) and that test
  widened, by the integrator. The fixtures live under
  `tests/fixtures/jobs/`, outside the skills.
- `contracts.skills.load_skill` refuses a skill whose `tools_used` names a
  tool not in `contracts.tools.TOOLS`; the `jobs-*` skills name the job tools
  of section 3, which lane T adds.
- Whether `get_frames` serves the nearest frame for a step without one; the
  skills ask for steps that have a `frame_index`.
- `src/fleetlab/jobs/__init__.py` is lane J's; until it exists,
  `fleetlab.jobs.task_text` imports as a namespace-package module.

## 6. Numbered investigations: pools, profiles and exposure

*20 October 2026. The continuation of a job line is a sequence of numbered
investigations, each with one final evaluation (its release run).*

- **Sessions.** `JOB_LIMITS.sessions` is 12; a request written for 6 stays
  valid. The driver's session cap is the request's budget unless
  `--max-sessions` lowers it.
- **Allocation profile.** `RequestState.allocation_profile` and
  `SavedJob.allocation_profile`: null (the legacy rule, seeds from the initial
  state alone, `seeds_for(JOB_SEED, "r01", 0, k)`) or `pool-v1`
  (`seeds_for(JOB_SEED + 1 + base_task, "r01", 0, k)`). Both fields are left
  out of the dump when null, so every earlier file keeps its bytes and digest.
  The profile is host configuration: `public_request` leaves it out of the
  mounted request, and `operator_state` explains `base_task` as the pool of
  stored starts of the shared scene, not the work. Every derivation names its
  profile (`jobs.starts.start_for`, `development_starts_for`,
  `release_starts_for`, keyword `profile`): the driver's development requests
  (`_batch`), the toy's release, the saved job, the operator's `run` and
  `_state`, `saved_job_from_run` (from the run's recorded request). A release
  receipt whose starts are not the saved job's profile-derived identities is
  refused (`release_receipt`, `driver._release`). Replay uses the recorded
  identities, unchanged.
- **Action horizon.** `JobProgram.actions_per_chunk`, 5 to 30 or null (the
  whole chunk, 30), left out of the dump when null; an explicit 30 is another
  program with another digest. The horizon lane's runner plays it.
- **Exposure.** A release attempt's exposure line records its full start
  identities and profile (`library.record_release_exposure`), before the
  attempt runs; a line written before that is read under the legacy profile
  (`library.exposure_identities`), which reproduces the identities job one's
  recorded receipt holds. Freshness compares full identities for any request
  (`library.exposed_identities`), finished or not. `carry_exposure` appends the
  lines a library lacks, byte for byte.
- **Preflight** (`jobs.prior.preflight`, before anything is written): a request
  with a profile names its cohort in the host's assignment record
  (`--assignment FILE --cohort NAME`, format `fleetlab.coordinator-cohorts/1`:
  `assignments[]` with `line`, `cohort`, `base_task`, `allocation_profile`,
  `status` reserved or active, `request_sha256`; `historical[]` with
  `base_task`); no other line and no historical entry holds the pool; every
  public start (base task, initial state) means one identity across the
  examined identities and the request's own pool, initial states 0 to 49; and
  none of the request's evaluation identities (30 to 49) was examined before.
  A refusal exits 2 with every reason and leaves no output directory.
- **Imported history** (`jobs.prior.read_prior`, `--prior-run DIR`, repeatable,
  in order). For each earlier investigation directory (`job_run.json`): the
  completed episodes (rows of a run's `job_episodes`, with their three
  per-episode tables and videos) of every development batch, interrupted ones
  included with their status, and of its final evaluation (`release/run/run`:
  the saved program j0001, part `release-candidate`, and the naive chain
  j0002, `release-naive`); the programs, the checks (outbox and libraries),
  the memory notes, the account, the job and check names, and the exposure
  lines of its libraries. Each episode is checked against its run's own record
  (program digest, its place among its job's starts, its start under the run's
  own profile); a receipt's pairs against the run; a legacy exposure line of
  the evaluation run against its receipt. Nothing of `truth/`, no run record,
  batch file or host record reaches the agent.
- **New ids.** Investigation i's jobs are numbered 1000 + 100 (i - 1) + n: a
  development job jNNNN keeps n = NN (at most 97), the evaluation's program 98
  and naive chain 99; an episode keeps its last two digits (e000401 of j0004
  in investigation 2 is e110401 of j1104). Every row of the five tables, the
  file names and the video columns are rewritten; nothing else changes. At
  most 89 investigations, so every id stays within six digits and beyond the
  48 job ids a request can reach.
- **What is written** (`Prior.write`, under DIR/prior, only after the
  preflight passed): `history/` (a job-run tree every export merges before the
  batches), `registry.json` (host: sources, file digests, each job and episode
  with its own id and full start identity, the examined identities),
  `exposure/` (the lines carried into every library the investigation
  freezes) and `history.json`, copied byte for byte to results/history.json.
  The importer refuses (exit 2, nothing written) when a source changes while
  it is read.
- **The tools** (`toolserver.jobs.PublicHistory`, `history_problems`):
  the history is validated as a whole against the export's `job_episodes`
  before anything is shown or counted: each imported job's episodes are
  exactly its rows, with its digest and starts; no id collides with a current
  request; every program is its digest's text. `outcomes` refuses a row that
  is neither a current outcome's nor imported. `job_results` lists imported
  jobs after the current ones (status `imported`, session 0, never charged,
  with `investigation`, `part`, `source_job_id`, `source_status`,
  `source_session`, `source_episode_ids`, `base_task`, `requested_goals`,
  `order_rules`: their goal indices are their own request's). `job_library`
  adds `history` (investigations with final evaluation, notes and account;
  programs; checks; names). `save_job` refuses an imported job, a name any
  earlier investigation used, and a program whose recorded runs broke an order
  rule here or in an imported job of the same goals and order rules (another
  request's verdicts do not count); `write_check` refuses a used check name.
  `query`, `get_frames` and `run_check` work on imported episodes as on
  current ones. `operator_state.history` points to them.
- **The launch audit** scans the agent prose of results/history.json
  (`agents.audit._HISTORY_PROSE`: imported jobs' names and purposes, the
  programs' text, the checks, notes, accounts and names) with the agent-text
  rules (canaries, hidden identifiers, seeds); every other key and value stays
  strict.
