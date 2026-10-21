# Contracts: the interfaces of the test bed

Version 1, written 2026-10-07 for stage 2 and revised the same night after an independent review (three
blocking findings, eighteen smaller ones, all applied). Owned by the
orchestrating session: implementation agents build against these interfaces
and propose changes, handled in batches. The Python side lives
in `src/fleetlab/contracts/` and carries `CONTRACTS_VERSION`; a change that
breaks a caller bumps it and names the change.

Kept deliberately lean for a first build: no measured latencies in the visible
log, no action-spread signal for the risk estimator, costs in hours only, no
realistic-range metadata on fault types, and the supervisor's `recover` and
`degrade` as stage-5 additions. Each is named where it was dropped.

Everything here was learned from the stage-1 spikes, mainly
[the skeleton](../../spikes/skeleton/README.md) (section "What the interfaces
must provide"), [replay](../../spikes/replay/README.md) and
[xvla-speed](../../spikes/xvla-speed/README.md), and from the
[agent runtime](../../spikes/agent-runtime/README.md). Where a rule has a
measured reason, the spike is named.

## 1. Principles

1. **A fault is data.** A fault is a declared change to a parameter of the
   system model, applied through a small number of hook points. The rollout
   loop knows the hook points and nothing about any fault type.
2. **Truth is kept apart from the log by construction.** Every stored column
   and every field of the system model carries a visibility tag. The
   agent-visible export and the system description are built from the list of
   visible columns and fields, never by removing hidden ones, and a test fails
   on any column or field without a tag.
3. **An episode is its identity.** Two runs with the same identity produce the
   same states and images, bit for bit, under the seven replay rules
   (section 4). Nothing else about the run (process, order, neighbours in the
   batch, video on or off) may matter, and the rollout proves it with hashes.
4. **Adapters, not forks.** The simulator and the policy sit behind two
   interfaces. A second policy (the scripted teleoperator today, pi0.5
   tomorrow) and a second simulator run through the same loop unchanged.
5. **One vocabulary.** Component names, configuration keys, condition keys,
   action verbs and the six generic fault classes come from one module,
   `fleetlab.contracts.vocabulary`, used by the world generator, the truth
   tables, the scorer, the tool schemas and the skills. The skeleton wrote
   them twice and they drifted. Fault *kinds* are not in it: the fault
   registry lives in `fleetlab.faults`, which is never mounted into a sandbox,
   so that the unseen fault types cannot be read off a module (section 12).
6. **Neutral identifiers.** Episode, robot and site identifiers visible to the
   agents carry no information: `e000123`, `r07`, `s2`. The map from
   identifier to identity lives on the truth side.

## 2. Vocabulary

- **World.** One generated fleet with its own structure, faults and history,
  made from a seed. Development, nightly, test and reserve worlds are all
  worlds.
- **Fleet, site, robot, component.** A fleet has sites; a site has robots and
  conditions; a robot has components and a versioned configuration.
  Components, version 1: `scene_camera`, `wrist_camera`, `gripper`, `arm`,
  `controller` (the computer that runs the policy and the control stack).
- **Configuration.** Versioned key-value settings at fleet, site or robot
  scope; a robot's effective configuration is the merge, robot over site over
  fleet. Version 1 keys: `policy.sampling_steps`,
  `policy.actions_executed_per_chunk` (how many of a returned chunk the
  rollout consumes; the chunk length is the policy's), `policy.image_views`,
  `policy.precision`, `control.mode`, `control.action_delay_steps`,
  `camera.scene.calibration`, `camera.wrist.calibration` (the pose the fleet
  believes its camera has), `software.version`. Every change to a value is a
  change-log entry.
- **Condition.** A site-level circumstance: lighting, table and furniture
  placement within installation tolerance, object batch (mass, friction,
  appearance), clutter.
- **Fault.** A declared change to a component, a configuration value or a
  condition, with a level, a scope and a schedule. A fault at level zero is no
  fault.
- **Action.** What a fleet operator can do: a verb from the catalogue applied
  to a target at a scope. A **fix set** is the actions a method proposes
  within a budget.
- **Episode.** One attempt at one task by one robot: a reset, a sequence of
  steps, an outcome.
- **Windows.** The observed window the agents analyse, the experiment window
  in which they may run A/B tests, the scoring window that measures
  predictions and fixes. Each is a range of simulated days.
- **Oracle.** The truth tables made by replay: the outcome of every scoring
  episode under subsets of its active faults, and what follows from that.
- **Run and job.** A run is one invocation of the rollout engine on a
  manifest of jobs; a job is one episode or one replay with its identity.
- **Skill.** A procedure an analysis agent can follow, as a file the runtime
  mounts into the sandbox.

## 3. The system model (`contracts.model`)

Pure data, serialisable to JSON, with the identifiers of principle 6.

```
Fleet      id, sites: [Site], config: ConfigVersion, change_log: [Change], maintenance_log: [Maintenance]
Site       id, robots: [Robot], conditions: {condition_key: value}, config: ConfigVersion
Robot      id, site_id, components: {component: ComponentState}, config: ConfigVersion, traits: {...}
ComponentState   component, installed_day, last_service_day, health: {metric: value}
ConfigVersion    version (int), values: {key: value}, since_day
Change     day, scope ("fleet" | site_id | robot_id), key, old, new, author ("ops" | "vendor" | "auto")
Maintenance day, robot_id, component, verb, note
```

Faults change this model (section 7); actions change it back or onward
(section 8); the agent-visible export of the model is the **system
description** the agents receive (section 11), together with the world's
task classes, so that the catalogue can list one `route_to_teleop` per task
class. One rule governs what the
description and the change log show, for the fault, log and world agents
alike: **they hold the fleet's own records, as the fleet believes them, and
nothing marks an entry as a fault.** A configuration fault is a `Change`
entry with its true key and value, its author drawn like a benign change's
(`ops`, `vendor` or `auto`). A software fault is a `Change` of
`software.version` only; what the version brings with it stays hidden in the
version. Physical, sensor, site and data faults produce no `Change`: a
knocked camera moves the real pose while `camera.scene.calibration` still
holds the believed one, and `recalibrate` brings the two together again. The
generator emits benign `Change` and `Maintenance` entries for healthy robots
too, so that an entry is never a giveaway, and the health values it writes
carry noise and benign variation.

Fields have visibility tags like columns: `Robot.traits` and anything derived
from a fault are hidden and dropped by the same allow-list mechanism, with a
test. Traits are per-robot constants the generator draws (a camera mounting
tolerance, a gripper pad age), used by the fault schedule to decide where
faults fall. They explain nothing by themselves and are never exported.

## 4. Episode identity (`contracts.identity`)

```
EpisodeIdentity
  world_id, episode_id            neutral ids (truth side maps them)
  robot_id, site_id, day, episode_index    episode_index numbers the robot's episodes within the day
  suite, task_index, start_state  LIBERO suite, task, recorded initial state
  scene_seed, noise_seed          seeds_for(world_seed, robot_id, day, episode_index): pure, documented, collision-resistant
  policy: PolicySpec              the nominal spec: name, revision, precision, control_mode, actions_executed_per_chunk,
                                  sampling_steps, image_views; a configuration fault's effective spec is recorded on the truth side
  batch_size, row                 batch size fixed per run; row None for a row-independent policy, fixed otherwise
  image_size, max_steps
  faults: [FaultSpec]             the active faults with levels (truth side only)
  software: SoftwareVersion       lock hash, LIBERO patch set, MuJoCo version, pinned MIOpen database hash, contracts version
```

`EnvSpec.control_mode` is set from the identity at `reset`, so the control
mode has one home, `PolicySpec`. The visible start time of an episode is a
pure function of the identity, `started_at_for(identity)`: the day's shift
starts at 08:00 simulated time and the robot's episodes follow at equal
intervals by `episode_index`, so two runs of one identity write the same
time. An identity lists only faults that are
active on its day and above level zero, and a configuration or condition
fault always carries the `nominal` value it replaced; the identity's
validator refuses anything else, so that the oracle never toggles a fault
that was not in force. Noise is derived by
one function, `noise_for(identity, chunk_index, shape, dtype)`, on a CPU
generator in `float32`; a policy declares `noise_shape` and `noise_dtype`
(X-VLA 30 by its action width, SmolVLA 50 by 32), and the function must never
change between worlds.

The seven replay rules the rollout keeps (replay and xvla-speed spikes):

1. The scene seed is set directly before the reset, and the recorded start
   state after it (`LiberoEnv.reset`).
2. The policy's sampling noise is drawn from `noise_seed` by the rollout and
   handed to the policy; a policy must not draw its own (LeRobot's X-VLA does,
   so the engine carries its own sampling loop).
3. The batch size is fixed for a run and recorded; the row is free for a
   policy whose `row_independent` flag is verified (X-VLA in 32-bit, and in
   16-bit with transposed linear weights), fixed otherwise (SmolVLA).
4. The rollout owns the action queue; the environment never auto-resets.
5. A mid-episode restore uses a full snapshot (section 5).
6. One rendering context is current at a time (`LiberoEnv._make_current`).
7. Convolution algorithms come from the pinned database with
   `cudnn.benchmark` off; the database hash is part of
   `software`, and the engine refuses a (batch size, precision) pair that the
   database does not hold, because MIOpen would silently fall back to another
   algorithm.

Not part of the identity, and shown not to matter: the worker process, what
ran before or beside it, the order of episodes, whether video is written, the
other rows of a call. Exactness holds per machine only.

**Proof of replay.** Every episode records a hash chain: the fixture hash
after the reset, and per step the hash of the simulator state and of each
camera image as rendered (the skeleton's `truth/chain`; videos are written
upright, hashes are over the rendered, rotated images). The hash is the first
8 bytes of SHA-256 over the contiguous bytes; the state is
`mj_getState(mjSTATE_INTEGRATION)` as float64, so that the engine and the
comparison tool agree; the environment exposes both hashes (`state_hash`,
`fixture_hash`), and the fixture hash covers the model arrays a cell hook
can change (body poses, camera poses, geometry friction, masses, lights,
actuator ranges, textures), stored in the chain file. In every run a sample
of episodes, every tenth and at least 20, is replayed unmodified and must
reproduce its chain; a supervised episode's sample replay is a "scripted"
job from its recorded decision trace. One broken chain invalidates the
whole run (`chain_ok` is a run-level flag), with one named exception found
by the soak test: a **render transient**, where the simulator
state and the executed action are equal at every step, the image hashes
differ in at most one frame of the episode, no policy call, supervisor or
video consumed that frame (for an in-worker policy every frame counts as
consumed), and a model-free re-rendering (`fleetlab.engine.rerender`) has
shown which rendering was the odd one. A run whose comparisons are all
identical or render transients keeps its chain verdict and its oracle rows
count, because outcomes follow from state and actions and the sample proves
those identical; a second transient in a run, or one in a consumed frame,
still breaks the run. The dashboard shows transients in amber on the run
page and their rate per million compared frames on the progress page, so a
failing GPU becomes visible. Measured once in 3.5 hours of rendering beside
inference: about one odd frame per 300,000 compared. A second named
exception holds on the oracle's side only (experimental design, section 10,
entry 14, 2026-10-10): a **disturbed policy call**, where one execution's
policy answer differs from another's with identical inputs. The oracle
executes a second time, unchanged, every attempt as run whose outcome
enters a comparison with a replay; a replay that changes something is
executed once. A difference is settled by two further executions, one made
the run's way and one as a replay. Three of the four agreeing step for step
with one alone is an event, and the version of the three stands; two
replays agreeing against two executions made the run's way is a replay
defect and stops the oracle, as does every other pattern, a first
difference that is not an executed action, or more events than the
ceiling of that entry. A render transient met by the oracle's second
executions is handled as entry 18 of the experimental design says:
harmless and counted, under a ceiling of its own per window, where the odd
rendering is the oracle's; judged by this section's rule for a world's own
run, including its limit of one per run, where it is the attempt as run.
A world's own run keeps the rule above: a
difference in its sample means the world is run again in full. Video files are never
hashed (encodings differ). Replay rule 7 is kept per number of images
in a call, which is what the convolution database is keyed by: the batch
size times the real image views, so a configuration fault that changes the
image views needs its own pinned entry or is refused.

## 5. The environment adapter (`contracts.env`)

```
class SimEnv(Protocol):
    spec: EnvSpec                         suite, task_index, image_size, control_mode, hard_reset (False in version 1)
    task_description: str
    max_steps: int
    def reset(self, identity: EpisodeIdentity, cell_hooks: [CellHook]) -> Observation
    def step(self, action: Action7) -> StepResult      observation, done, verdict (the configured check's), native_success (the simulator's own predicate)
    def state_hash(self) -> bytes;  def fixture_hash(self) -> bytes      the chain's two hashes (section 4)
    def snapshot(self) -> EnvSnapshot                   the lean set below
    def restore(self, snapshot: EnvSnapshot) -> Observation    returns the saved observation, does not re-render
    def privileged_state(self) -> PrivilegedState       object poses and velocities, contacts incl. every robot body against the objects,
                                                        gripper opening, LIBERO's own check result: success check and events
    goal: TaskGoal                                       the task's predicate, object body and target body, read from the BDDL goal
    def handle(self) -> CellHandle                       model, data, the robosuite env, upload_texture(id): cell hooks and in-worker policies
    def set_success_check(self, check: SuccessCheck) -> None
    def close(self) -> None
```

**The reset sequence is owned by the environment and has fixed moments**
(skeleton, hook point 1): reuse the simulator (version 1 runs soft resets,
`hard_reset=False`; a hard reset rebuilds the model inside `reset()` and
wipes a cell hook, so hard resets are unsupported with cell hooks); apply
cell hooks through the `CellHandle` (faults that change the live MuJoCo
model: camera pose, physics parameters, object batch, textures with their
upload), each fault's `new_episode` having been called first because the
cell hook works on the state it returns; seed and reset; set the recorded
start state; settle (`SETTLE_STEPS` no-op steps that bypass the observation
and action hooks, so the fault states see nothing of them); set the control
mode; first observation. Cell hooks are
undone at the end of the episode, with the texture upload repeated on undo,
and a test checks that the fixture hash of a healthy episode after a faulted
one equals that of a healthy episode in a fresh process.

**Observation** is the skeleton's dictionary: `pixels` per camera (uint8, as
rendered), `robot_state` (joints, gripper, end effector), `task_description`,
`step`. Images are stored as rendered; a policy adapter flips or resizes.

**Snapshot**, the lean set the replay and teleoperator spikes measured as
necessary: the MuJoCo state (`mjSTATE_INTEGRATION`) with the solver warm
start; the gripper controller's `goal_qvel` and `current_action` (the arm
controller's goal is rebuilt from the next action in both control modes, so
it is not saved: replay spike for relative, teleop-absolute for absolute);
the positions and orientations of the fixture bodies only, as placed by the
reset (never every body, and never the controller ranges, which the reset's
cell hooks set: a restore must not carry a cell-hook fault back into an
episode that removed it); the
per-sensor clocks; the success check's state; and the observation that was
delivered at that step, because re-rendering after a restore differs by one
substep. Restoring hands that observation to the policy instead of rendering
a new one, and ends with `mj_forward`: after the state is set, MuJoCo's body
and site positions and its contacts still belong to the state before the
restore, so the privileged state, the cell hooks and an in-worker policy
would read the wrong scene; the call changes nothing in the physics that
follows (teleop-absolute, 12 of 12 continuations identical) and does not
render. `EnvSpec.image_size` is required, not defaulted, and `reset` refuses an
identity whose image size differs from the spec, because robosuite fixes the
renderer's size at construction.

**Success check** is pluggable and stateful, because the
fleet's check is of the form "released, retreated, settled for n steps":

```
class SuccessCheck(Protocol):
    tail_steps: int                       steps after LIBERO's first fire before the check gives up
    def reset(self, goal: TaskGoal) -> None     the task's goal names the object and the target
    def update(self, state: PrivilegedState, step: int) -> SuccessVerdict(success, done, reason)
    def snapshot(self) -> bytes;  def restore(self, blob: bytes) -> None      its state is part of the Snapshot
```

Version 1 ships LIBERO's own check and the fleet's check, **placed**, as
measured: LIBERO's check holds, no collision geometry of the
robot touches the object, and the object is at rest (slower than 1 cm/s and
0.15 rad/s). The verdict's `done` ends the episode: `success and done` when
placed holds; `done` without success with reason "LIBERO only" when
`tail_steps` (60) have passed since LIBERO's check first held without placed;
the step limit ends it with reason "step limit" and the tail never extends
the limit, so an episode that has not released by the limit is a failure,
as it would be against a fleet's cycle budget. Two more reasons exist: "LIBERO's
check" when a run is configured with LIBERO's own check (the reference mode
the gate-2 pass mark compares with), and "stopped" or "handed over" when the
supervisor ended the attempt. A handed-over episode's visible outcome is
`handed_over`, a third value beside success and failure, because a fleet
knows an operator took over; the continuation's result is truth only. Measured with X-VLA running
60 steps past LIBERO's check: placed costs nothing on the bowl
tasks (195 of 195), 3 of 199 basket successes, and 12 of 138 faulted ones,
ten of them cut short by the step limit under the delay fault. The rollout
applies the same check to original and replay and ends on the verdict
*after* the data hook (section 7), which is what the fleet's own runner sees;
the data hook may flip `success` only on a `done` verdict, so that a flipped
verdict never ends an episode early. The truth side records the verdict
before the data hook, LIBERO's first-fire step and the fleet check's own
step (table `verdicts`), because the gate-2 pass mark compares with LeRobot's
reference, which uses LIBERO's check. The check's state, LIBERO's first-fire
step or none, is part of the snapshot, otherwise a restore inside the tail
restarts it and the continuation ends on another step.

**Control mode** is a per-episode setting (`EnvSpec.control_mode`), absolute
for X-VLA and for the teleoperator (the same outcomes in both
modes on 200 paired episodes, absolute chosen), relative for SmolVLA. `Action7` is the 7-vector LIBERO expects (position, rotation,
gripper) in the mode's convention; the conversion from a policy's native
output is the policy adapter's job.

## 6. The policy adapter (`contracts.policy`)

```
class Policy(Protocol):
    spec: PolicySpec
    chunk_length: int                     actions per call (X-VLA 30, SmolVLA 50, ACT 100)
    needs_noise: bool                     False for deterministic policies (ACT, the teleoperator)
    noise_shape, noise_dtype              what noise_for() produces for this policy
    row_independent: bool                 verified by the rows check, not assumed
    in_worker: bool                       True for a policy that runs inside the worker, per step, with the environment handle
    image_views: int
    def load(self, device: str, precision: str) -> None
    def act(self, batch: [Observation], instructions: [str], noise: [NoiseSample | None], spec: PolicySpec) -> [ActionChunk]
                                          server policies; spec is the group's effective spec (sampling steps, image views)
    def begin_episode(self, identity: EpisodeIdentity, step: int) -> None    in-worker policies, per episode or continuation
    def act_one(self, observation: Observation, env: SimEnv) -> Action7                                      in-worker policies
    def convert(self, chunk: ActionChunk, control_mode: str) -> [Action7]
```

Invariants, each with a check: the same observation, instruction and noise
give the same chunk bits in another process (replay spike); a row's chunk
does not depend on its position or on the other rows when `row_independent`
is true (xvla-speed `rows_check.py`, three minutes, run after every library
update); the batch size changes bits, so it is fixed per run. These checks
load the model, so they live in a third tier, `make check-model`, run under
`tools/heavy-run`; `make check` stays CPU-only and `make check-sim` loads no
model. The engine runs one inference server per run with `batch_size` rows
and about two and a half workers per row (xvla-speed section 5), loads the
policy before the workers start, and pads empty rows with a copy of row 0.
Requests whose effective `PolicySpec` differs (a configuration fault changed
the sampling steps or the image views for one robot) cannot share a batch,
because the sampling loop is per batch: the server groups waiting requests by
effective spec and pads each group to `batch_size`.

**Teleoperator.** The scripted expert is an in-worker policy
(`in_worker=True`, `needs_noise=False`): it reads the live simulator every
step through the environment handle (the robosuite environment with its
model and data: geometry, contacts, object dictionaries, the grasp planner's
surface samples) and never goes through the server. An in-worker policy is
constructed per episode or continuation and exposes `done` and a `status`;
the engine ends a continuation on `done`, and after `done` the policy returns
a hold with its last gripper command, never the adapter's open no-op. It is
the only policy with that flag, which is logged on the truth side. It may
send a gripper command of 0, "hold the opening", which X-VLA never sends.
Its human model (hand-over delay, slower pace, a chance of fumbling) draws
from the identity's noise through `noise_for`, so that take-overs replay.
Measured on the locked stack: 999 of 1,000 from the start
states, 98.4% on six kinds of disturbed state, 88.6% when taking over from
X-VLA's own recorded failures at the end of their budget.

## 7. Faults (`contracts.faults`)

```
FaultSpec     kind, params: {name: value}, level: float, target: Target, schedule: Schedule
Target        scope ("fleet" | site_id | robot_id), component | config_key | condition_key
Schedule      onset_day, duration_days | None, drift: None | ("linear" | "step", rate)

class FaultType(Protocol):
    kind: str
    fault_class: str                      "physical" | "sensor" | "configuration" | "software" | "site" | "data"
    Params: type[BaseModel]               the pydantic model of params
    dose_levels: [float]                  the levels calibration runs
    truth: TruthMapping                   component or key, fixing verbs, scope (section 8); whether it writes a Change;
                                          the accepted diagnosis directions for its class
    def new_episode(self, spec, identity) -> FaultState
    def cell_hook(self, handle: CellHandle, state) -> Undo | None  before the reset renders
    def observation_hook(self, observation, state) -> Observation  what the policy and the log see
    def action_hook(self, action, state) -> Action7                between the queue and the simulator
    def policy_input_hook(self, policy_spec, state) -> PolicySpec  configuration knobs; the effective spec is recorded on the truth side
    def data_hook(self, verdict, state) -> SuccessVerdict          the success detector's own error
    def snapshot(self, state) -> bytes;  def restore(self, blob) -> FaultState
```

Rules of the truth mapping, checked when a type is registered: a
configuration-key fault lists `set` and `revert` among its fixing verbs,
because a wrong value is fixed by setting it right by definition; a condition
fault lists `change_condition`; `writes_change` follows the class
(configuration and software write a `Change`, the other classes do not);
and a component fault is robot-scoped only, so that a bad gripper batch at a
site is one fault per robot and k fixes are always k actions, which keeps
`achievable_gain(k)` honest. The faults in force for a robot respect the
configuration precedence: a site- or fleet-scoped configuration fault does
not cover a robot whose own configuration sets that key.

Two generic kinds exist for the changes an action can *add* (section 8):
`config_change` and `condition_change`, one type each, `params = {"value":
v}`, level 1, `nominal` the value replaced, the target naming the key.
`config_change` dispatches on the key to the hook that key uses (sampling
steps and image views through the policy input hook, the action delay
through the action hook, and so on); a key with no mechanism is a no-op;
`software.version` is meaningful only through the software fault type's
definition of what each version carries. Their two kind names are the one
thing the tool server's import of `actions` carries into the sandbox; the
pre-export word search lists them as the known exception.

Rules: a fault type implements only the hooks it needs; `new_episode` creates
a fresh state per episode (the delay line's queued commands are episode
state) and is called before the cell hooks, which work on that state; a
fault undoes itself completely and the test of
section 5 proves it; the hook order inside the rollout is fixed (observation
hook, then policy, then action hook, then simulator); the log listens on a
defined side of every hook (section 11): it records the commanded action
before the action hook and every observation signal, images and robot state,
after the observation hook, because that is what a fleet logs. Drift
semantics (how a level moves with the day) are left to the world generator
in stage 3. A new fault type is a module registered under its `kind` in
`fleetlab.faults` and needs no change to the rollout; the unseen-fault author
works through this interface alone.

World-level patterns are not fault types: a **confounder** is two faults in
the schedule with the same onset where only one has an effect; a **decoy** is
a fault type whose dose-response is flat, confirmed by replay; an
**interaction** is two faults whose joint effect exceeds the sum; **drift** is
the schedule's drift field. The generator composes them (section 9).

## 8. Actions and fixes (`contracts.actions`)

Verbs, version 1: `recalibrate` (the two cameras and the arm), `clean` (the
two cameras and the gripper), `replace`, `service` (every component of a
robot); `set`, `revert` (a configuration key at a scope); `change_condition`
(at a site); `route_to_teleop` (a class of work, fleet-scoped);
`request_training_data` (a failure mode, fleet-scoped). The catalogue for a
world is every valid (verb, target) pair over its model: hundreds of actions
for about ten active faults. The costs of the experimental design (section 3
there) live in `contracts.actions.COSTS`, appear on every catalogue entry in
the system description, and are written as a visible table `costs` into the
export, because a fleet operator knows what a repair costs.

```
Action     verb, target: Target, value | None
Cost       hours: float                                per verb and component, from the design
FixSet     actions: [Action], budget: Budget           Budget = ("count", k) | ("cost", hours)
cut(ranked: [Action], budget, costs) -> FixSet       walk down the list, skip what no longer fits; the scorer applies it at k = 3, 6 and 4, 8 hours
Resolution resolve(action, identity, world_truth) -> (removed: set[fault_id], added: [FaultSpec])     per episode
```

`resolve` is the bridge from an action to the oracle, evaluated per episode
against the faults active in it. `recalibrate scene_camera on r07` removes a
camera-pose fault on that robot and nothing if there is none; `replace
gripper` removes gripper wear on that robot; component verbs are robot-scoped
and `change_condition` is site-scoped. Configuration actions follow an
exact-scope rule: `revert key at scope X` removes the fault whose `Change`
is at X, and covers the robots that scope covers; `set key = nominal` at
robot scope on a robot that inherits a site-scoped fault removes that fault
for that robot's episodes, as an override. An action that only removes
faults is a lookup in the oracle; `set` to a non-nominal value or
`change_condition` to a new value *adds* a change, which no replayed subset
covers, so `added` always means a fresh replay, counted in the budget row
"fixes that need a fresh simulation". An action on a healthy target removes
nothing and adds nothing, and counts as no change without simulation. The two verbs without a replay (`route_to_teleop`,
`request_training_data`) are scored under claim 5. The truth mapping of each
fault type (section 7) says which actions fix it, so the scorer and the
oracle agree by construction. The scalar of the cost budget is defined in the
experimental design.

## 9. Worlds and the oracle (`contracts.world`, `contracts.oracle`)

```
WorldSpec   seed, size ("small" | "medium"), sites, robots_per_site, days, windows: {observed, experiment, scoring},
            fault_schedule: [FaultSpec], patterns: {confounders, decoys, interactions, drifts}, policy: PolicySpec,
            tasks: [(suite, task_index)], episodes_per_robot_day
WorldFiles  spec.json, model.json (truth), description.json (agent-visible), manifest.jsonl (jobs), hash
```

Generation is a pure function of the spec; the hash of the world files is
recorded for the test worlds. Seeds of
every episode are derived from the world seed and the identity fields, never
drawn from a global generator.

```
OracleRow   episode_id, active_faults (canonical hash of the effective fault list in the replay), success
Gain        rescued: int, broken: int, net: int          over a set of episodes
value_fix_set(fix_set, world) -> Gain                     resolve per episode, then look up the remaining set
fault_effect(fault_id, world) -> Gain                     signed; a fault can rescue an episode (gate 1: 2 of 40)
interaction(fault_a, fault_b, world) -> int
achievable_gain(k, world) -> Gain                          the best k-set among the resolved fixes of all active faults
```

The oracle is a per-episode lookup. Each scoring episode is replayed under
the full power set of its effective active faults (decoys and inert
confounder entries excluded) when it has at most four (at most 15 replays;
most episodes carry one to three, about 9 replays on average), so that any
fix set that only removes faults, and the achievable gain at any budget k,
are exact lookups. Episodes above the cap are reported, and they and the
actions that add a change (section 8) need a fresh replay, counted in the
budget row "fixes that need a fresh simulation". The experimental design sizes the replay count
from the fault schedule. Rows come from runs whose unmodified sample
reproduced its hash chains (section 4); a run with a broken chain
contributes no rows. In the oracle's own runs a difference that is settled
as a disturbed policy call (section 4) does not break the run: the
reproducible version of that attempt supplies the row, and the event is
listed.

## 10. Rollout engine (`contracts.rollout`)

```
Job         identity: EpisodeIdentity, mode ("run" | "replay" | "continue" | "scripted"), remove: [fault_id], add: [FaultSpec],
            from_snapshot: SnapshotRef | None, record_video: bool, supervisor: SupervisorSpec | None,
            decision_trace: [DecisionRecord] | None,     "scripted" replays a supervised episode with its recorded decisions
            max_steps: int | None                        a "continue" job's own step budget, counted from its snapshot's step: a snapshot
                                                         at step S with budget N permits N more active steps (the teleoperator needed up
                                                         to 600 steps plus settling; the simulator's horizon is raised to match)
            teleoperator_params: JSON | None             a "continue" job's parameters for the in-worker teleoperator: the human model's
                                                         settings and the operator's admission; part of the job key only when present
            snapshot_steps: [int]                        where a run job saves snapshots (the restore check, the oracle's counterfactuals)
Run         manifest of jobs, run_dir layout (section 11), resumable: a job is done when its episode record and chain exist
Hooks       the fault hooks of section 7, in fixed order, plus:
            supervisor_hook(step, observation, telemetry_window) -> Decision      every step after the risk estimator fires
            takeover(snapshot, teleoperator) -> continuation                        a "continue" job from a snapshot
```

`Snapshot` (section 5 plus rollout state): the environment snapshot, the
actions still queued, the chunk index, the step count, every active fault's
state, the success check's state. A job spawned at run time from a snapshot
(a hand-over's continuation) is written to the truth side as a job file next
to its snapshot the moment it is created, so that a resumed run finds and
finishes it. In version 1 a replay from a snapshot may not add or remove a
fault that uses a cell hook; the engine refuses it. Snapshots serve two
things: the counterfactual replay that scores an
intervention ("would the episode have failed without it") and the hand-over
continuation. They are never used to undo the world for the robot itself
(section 14). The engine proves snapshot exactness the same way as replay:
continuing from a snapshot and running through must give the same chain as
the uninterrupted episode.

Added with the human teleoperator (2026-10-09),
all of it additive: a job or snapshot without the new fields keeps its key and
its meaning.

- `Snapshot.last_commanded`: the last commanded action before the action
  hooks, seven numbers in the episode's control mode, saved with a hand-over
  snapshot. It is what a waiting operator holds: the pose and the gripper
  command in absolute control, the gripper command with zero motion in
  relative control. A snapshot from before this change lacks it; a human
  teleoperator needs it and refuses to guess a hold.
- `Job.teleoperator_params`: a human continuation stores its complete human
  settings and its operator admission here, and the admitted continuation is
  written next to its snapshot before it is scheduled, so that a resumed run
  uses the same assignment.
- `RunConfig.defer_continuations`, false by default: when true, a hand-over
  still saves its snapshot and writes its continue job, but the fleet does not
  schedule continuations by itself; a caller admits them, for example through
  an operator queue in simulated time, and runs them in a later manifest.
- **The step budget of a continuation counts from the snapshot.** Until
  2026-10-09 the engine read `max_steps` as the episode's absolute end point,
  which left a teleoperator that took over at step 170 about 530 of its 700
  steps. The words above always said "own step budget"; the engine now does
  what they say. Results of continuations from before that date were run
  under the shorter budget: the nightly worlds' live-pass counts may move
  once at the first nightly after the change.
- **A hold spends no budget.** While an operator waits in the queue or
  reacts, simulated time and the faults in force go on, but neither the
  continuation's step budget nor the success check's tail of 60 steps is
  spent. A placement during a hold still ends the episode.

Scheduling (xvla-speed section 5): batch rows fixed, workers two to three per
row, work handed out on request with preference for the task a worker has
built, workers replaced after a fixed number of episodes (16 in the spikes),
staggered so that not all rows empty at once, file writing in waiting time,
the policy loaded before the workers start, everything under
`tools/heavy-run`.

## 11. Episode log, version 1 (`contracts.log`)

Parquet tables with zstd, one run directory per run, two trees:

```
run_dir/
  log/        agent-visible         episodes.parquet, steps/<episode_id>.parquet, events.parquet,
                                    health.parquet, change_log.parquet, maintenance.parquet, description.json,
                                    video/<episode_id>.scene.mp4, video/<episode_id>.wrist.mp4
  truth/      hidden                identities.parquet, faults_active.parquet (with params and nominal), effective_policy.parquet,
                                    chain/<job key>.npz (a replay has its own key), verdicts.parquet (true verdict and
                                    reason, LIBERO's first-fire step, the fleet check's step), replays.parquet,
                                    job_outcomes.parquet (every job of any mode: key, episode, mode, removed, added,
                                    snapshot step, success, steps; the live layer's scoring reads it), oracle.parquet,
                                    decisions.parquet, snapshots/
  run.json    hidden                software versions, timings, waits, memory, worker records
```

Everything visible is a deterministic function of the identities, so two
runs of the same world give byte-identical visible tables; measured times
(inference waits, wall clock) live in `run.json` and never in a visible
table, because they would differ between replays and churn the fixtures.
Every column is declared once in `contracts.log.SCHEMA` with its type and a
visibility tag (`visible` or `hidden`) and a one-line description; the data
dictionary the agents receive is generated from the visible part. Version 1
columns:

- `episodes` (visible): episode_id, robot_id, site_id, day, task_text,
  started_at (simulated time of day), steps, outcome (success or failure by
  the fleet's check), duration_s (simulated), video_scene, video_wrist,
  experiment_id (empty unless the episode was an arm of a requested
  experiment).
- `costs` (visible): verb, component or key kind, hours; from
  `contracts.actions.COSTS`.
  Hidden, in `identities`: everything of section 4, including `start_state`;
  the agents see the layout in the first frame, not an index.
- `steps` (visible, per episode): step, time_s (simulated), joint_pos[7],
  joint_vel[7], gripper_qpos[2], eef_pos[3], eef_quat[4], commanded_action[7]
  (before the action hook), chunk_index, chunk_step, frame_index (which steps
  have a frame in the video). Hidden, in `chain`: state hash, image hashes,
  executed_action[7] (after the action hook).
- `events` (visible): episode_id, step, kind (grasp, release, drop, collision,
  near_miss, stall, supervisor_call, handover, retry), detail. Event detection
  is the fleet runner's; version 1 ships grasp, release and
  drop from gripper and object telemetry.
- `health` (visible): day, robot_id, component, metric, value. Periodic
  component checks as a fleet runs them; the schema exists in version 1 and
  stage 3 fills it, with noise and benign variation.
- `change_log`, `maintenance` (visible): the model's logs of section 3, under
  the one rule of that section.
- `faults_active` (hidden): episode_id, fault_id, kind, level, target,
  schedule fields. `effective_policy` (hidden): the policy spec after the
  policy input hook. `replays` (hidden): the oracle rows. `oracle` (hidden):
  the derived gains. `decisions` (hidden): the supervisor's decision trace
  (step, decision, hold steps, decision_ms, model_ms).

Two rules from the skeleton stand: the visible tables are written from the
allow-list, and the export refuses a run whose episode identifiers are not
neutral numbers. A word search over the export with the truth side's
vocabulary and with the fault registry's kind names, and a canary, run before
any agent sees it; the search covers every file mounted into
the sandbox, the tool server's own source included.

## 12. Agent tools and answer schemas (`contracts.tools`)

The tool server runs inside the inner sandbox with the export mounted
read-only and no network (agent-runtime spike). It ships as its own small
package whose own code says nothing about how worlds are made, because its
source is readable inside the sandbox (skeleton, section 6). It may import
`fleetlab.contracts` and `fleetlab.log.reader`, whose source names the
hidden tables and columns: that is class knowledge, not world truth, and the
pre-export leak search over the mounted source uses world-specific terms
(fault kinds, trait names, canaries) with `actions.ADDED_CHANGE_KINDS` as the
documented exemption. The reader runs materialized and locked, with DuckDB's
file access off. Tools, version 1:

```
query(sql) -> rows                       DuckDB over the visible tables; file access off (DuckDB's file functions read the canary in the spike)
get_frames(episode_id, steps, camera) -> images     from the video at the logged frame_index
describe() -> data dictionary + system description
list_skills() -> [SkillManifest];  read_skill(name) -> text       the skill library, delivered by the server (section 13)
request_experiment(ExperimentRequest) -> receipt   queued; run after the session ends (below)
experiment_results() -> [ExperimentResult]          in the next session
submit_diagnosis(Diagnosis) -> receipt
submit_fix_set(RankedFixSet) -> receipt
memory_read() -> notes;  memory_write(notes)      the agent's own notes for this world; reset between worlds
write_learned_skill(LearnedSkillDraft) -> LearnedSkillReceipt   a diagnostic skill kept for later investigations (section 13)
```

**Sessions.** An experiment is a heavy job of minutes and a tool call inside
`claude -p` cannot wait for it. An analysis is therefore a sequence of
sessions: the agent requests experiments and ends the session in the state
"waiting"; the runtime runs every queued request as one job in the experiment
window, both arms always simulated (no healthy-target shortcut here, or the
agent would learn health from identical results); a new session starts with
`memory_read()` and the new episodes visible. Memory is the only state that
crosses sessions. The tool server enforces the experiment budget of the
design (`contracts.tools.EXPERIMENT_BUDGET`: 8 requests per analysis, 60
episodes per arm). Arm episodes carry the experiment's id in the visible
column `episodes.experiment_id` (a fleet knows which episodes were part of a
test it ran), so a later session keeps them out of the observed rates and
reads them only through the experiment's result.

```
Diagnosis          causes: [Cause], mode_summaries: [str], warnings: [Warning], retrain: [RetrainIndicator]
Cause              target: Target (carries the scope), direction ("degraded" | "misconfigured" | "changed" | "drifting"),
                   confidence (0..1), evidence: [EvidenceRef (episode_id, step | table, query)], failure_mode: str
RankedFixSet       actions: [RankedAction]                        at least ten, ordered by value; the scorer cuts it at every budget (section 8)
RankedAction       action: Action, expected_gain: float (points of fleet success over the scoring window), uncertainty: (low, high), rationale
ExperimentRequest  arms: [{actions: [Action]}], robots: [robot_id], episodes_per_arm: int, paired_starts: bool, power_note: str
Warning            robot_id | site_id, metric, expected_day, confidence
RetrainIndicator   failure_mode, episode_ids (examples of the mode), why_no_action_fixes_it, data_wanted
```

`describe()` takes a flag for the no-data baseline: the static structure
(sites, robots, components, keys, the catalogue) without the change and
maintenance logs. That flag alone is not what the no-data ranker of
the experimental design, section 6, is shown: the description also holds the current
configuration values and a health snapshot, which reveal faults and are
data. The ranker's prompt holds only: site and robot identifiers, component
names, the names of the keys and actions that may be used, the cost table,
and the catalogue's actions with what each does and without setting values
(clarified on 2026-10-09 after the review of the baselines).

The schemas are pydantic models; `claude -p` and `codex exec` both get their
JSON schema (OpenAI needs strict schemas, agent-runtime README). The scorer
reads the same models. Every number in a diagnosis or fix set must be
reproducible from a recorded tool call: the runtime
keeps the tool log next to the answer, and the scorer flags numbers that
appear in no tool result.

## 13. Skills (`contracts.skills`)

A skill is a directory `skills/<name>/` with `SKILL.md` (front matter: name,
description, when to use, tools used, inputs, outputs, version). The
computations a skill needs are SQL blocks inside the `SKILL.md`, each one
SELECT the agent copies, fills in and runs through `query`, so that every
number lands in the tool log with the statement that produced it; the helper
scripts beside a skill are reference runners for the tests, and no tool
executes them. The skills reach the agent
through the tool server (`list_skills`, `read_skill`) or inline in the system
prompt, not through the CLIs' own skill loading, which stays switched off: the
agent-runtime spike found that a private home still pulled the user's real
skills, and both CLIs run with their skill features disabled. Rules: general procedures and general
knowledge about fault classes only; no identifiers or values of a specific
world; nothing from a truth table except through the development worlds'
published scores. Each skill has a fixture test against the miniature fleet's
export, and the skill set is frozen with the pipeline at gate 4. Version 1
skills: triage, episode-cards, failure-modes, scope, sensitivity-from-logs,
sensitivity-by-experiment, rank-actions, write-diagnosis; supervisor:
assess-risk, choose-intervention.

**Learned skills** (docs/design/loop.md, section 9; `contracts.learned`,
`fleetlab.loop.learned`). An analysis agent writes them with
`write_learned_skill`; they have the same format, with `origin: learned`,
`applies_when` and `provenance` added to the front matter and the sections
Applies when, Parameters, Evidence, Competing explanations, Next experiment
and Provenance in the body. They hold no identifier of the fleet they were
written on (robots, sites, episodes and experiments are parameters, `$name`
in SQL and `{name}` in prose), pass the agent-text leak rules and the rules of
a mounted skill against that fleet's truth side, and are frozen by the host
into a library with a digest. A later analysis receives a library only when it
is created; it is mounted read-only at `/learned`, checked again at every
launch against the later fleet's truth side, and served by `list_skills` and
`read_skill` after the base skills, which it can neither replace nor change.

## 14. Live layer (`contracts.live`)

```
class RiskEstimator(Protocol):
    def update(self, telemetry: StepTelemetry) -> float      probability of failure within the horizon
Decision   "continue" | "retry" | "stop" | "hand_over"        version 1; "recover" and "degrade" are stage-5 additions
Supervisor decide(context: SupervisorContext) -> (Decision, rationale, decision_ms, model_ms)
SupervisorContext   recent frames, telemetry window, risk, task_text, options available, operator queue state
```

The robot holds still while the supervisor decides, counted as time, not as a
failed attempt. Holding still is defined per control mode: in absolute
control the last commanded pose is repeated with the gripper command
unchanged; in relative control a zero motion with the gripper command
unchanged. The adapter's `NO_OP_ACTION` (gripper open) is not a hold: it
would drop a held object. Every intervention is physical, in the live
simulation: `retry` is a scripted recovery primitive (open the gripper, lift,
return to a pre-grasp pose, hand control back to the policy); `stop` ends the
attempt safely; `hand_over` creates a "continue" job for the teleoperator
from the current snapshot. A snapshot is never used to put the world back for
the robot (section 10). The decision trace (step, decision, hold steps,
`decision_ms` through the subscription route, `model_ms` as the production
estimate) is recorded on the truth side, and a supervised episode replays in
"scripted" mode from that trace, so that its chain can be proved like any
other. If an action-spread signal is added later, its extra samples are
separate calls at the same batch size with noise from `noise_for(identity,
chunk_index, k)`, never extra rows in the executed call.

## 15. Versioning and change

`fleetlab.contracts.CONTRACTS_VERSION = "1.0"`. Implementation agents import
from `fleetlab.contracts` only; a needed change is a request labelled
`contracts`, and the orchestrator applies changes in batches and bumps the
version. The tests under `tests/test_contracts_*.py` are the executable part
of this note: schema and model tags complete, vocabulary single-sourced and
free of fault kinds, seed and noise derivation pure, action resolution per
episode against a toy truth including the exact-scope rule and an added
change, gains signed, the strict JSON schemas valid, a planted world
identifier caught in a skill. Checks that load a model are the third tier,
`make check-model`.

## 16. Who implements what (stage 2)

| Section | Owner |
|---|---|
| 3, 4, 7, 8, 9 data types; 11 schema; 12 schemas; 13 format; 15 tests | contracts agent from this note, reviewed independently |
| 5 adapter changes (soft resets with cell hooks, lean snapshot, handle, stateful check); 10 scripted replay of decisions | engine agent |
| 6 policy adapters: X-VLA with own sampling loop in both precisions, grouping by effective spec; teleoperator as in-worker policy | engine agent, teleop agent |
| 7 exemplar fault types and dose-response | faults agent |
| 11 writer, export, data dictionary; 10 run layout | log agent |
| 12 tool server and runtime; 13 mounting; 14 runtime side | runtime agent |
| 13 skills version 1 | skills agent |
| 9 oracle lookup and 8 resolve, first version on the miniature fleet | integration agent |
