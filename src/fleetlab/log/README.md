# fleetlab.log: the episode log, version 1

## Export review repair (2026-10-09)

`audit_export(export_dir, run_dirs, *, fleet=None, texts=None,
agent_texts=None, code=None, extra_terms=())` and
`leak_search(root, terms, texts=None, *, agent_texts=None, field_terms=(),
kinds=None, seeds=(), fault_ids=())` always check the tree and mounted code
strictly. Neither takes a `surface` argument. Our prompts, skills, schemas and
tool descriptions belong in `texts`; only separately supplied `agent_texts`
receive the agent policy. Both are mappings from the source file/field to its
text. For text alone use `search_texts(terms, texts=None, *, agent_texts=None,
field_terms=(), kinds=None, seeds=(), fault_ids=())`; no temporary tree is needed.
An empty source label is an error. A mixed call cannot relax an export or code.

In agent-written text, world seeds, canaries, non-countable world fault ids and
multipart identifier spellings of planted names or hidden properties refuse. Definitions
of fault kinds and pattern roles, trait producers and fleet visibility tags
establish planted names; execution schemas and report producers establish
record structure. Job/identity fields, hidden log columns, fault-parameter
keys and declared record-layout names never qualify on this surface. Unknown
truth names stay planted until a code definition establishes another class;
Declared recipe fields are configuration keys rather than planted names on
agent text; they stay strict on our own texts, including `worn_robots` and
`min_clean_robots`. A genuine kind, role or hidden property with the same name
still qualifies. Recipe fields come from their defining model schemas.
Public identifiers, including the actions contract's `config_change` and
`condition_change`, pass in agent text. A false refusal costs an analysis, so
for established classes this policy deliberately errs toward allowing its text.
For worlds with one-letter, two-digit countable ids, agent-written text is not
checked for those fault ids.

Multipart identifiers use underscores, dots or camelCase and also count within
longer identifiers, prefixes, plurals and suffixes. Spaced, hyphenated and joined
natural forms pass. A multipart name needs at least two nonempty parts: a
leading or trailing separator does not turn a single word into one. Single
words, even inside longer identifiers, never refuse except non-countable world fault
ids or canaries. Seed checks include integer/floating values, thousands separators,
hexadecimal and scientific notation. An agent's own label equal to a planted
kind still refuses, as does a non-countable world fault id used as an ordinary
token; a person must judge such refusals against the record.

On agent-written text, this check is a tripwire for generator identifiers,
non-countable fault ids, canaries and seeds; single-word hidden names cannot be
checked in agent prose. It cannot prove that nothing was learned from the world.

`truth_terms` returns a list-compatible `TruthTerms` with two matching classes.
Hidden injection, identity and execution contracts, robot-property producers,
registered fault kinds and the deliberate role/pattern vocabulary define the
flexible class. Task-name string columns selected from the `Task` contract and
hidden identity schema supply structural identifiers, including task-suite
names; public task prose also uses them. Generic generator, oracle and scoring report producers define structural
keys. Unknown names default to planted matching until a defining schema or record
producer establishes structure; the classifier does not infer semantics from
words such as `queue`, `count` or `summary`.

Structural keys match their exact defining spelling, ignoring case, as JSON/map
keys, columns, filenames and labels (`queue_s=2`). Their spaced and joined forms
do not match, and their bare occurrence in prose is allowed. A camel-case alias
defined by a record keeps that literal spelling rather than acquiring a new
underscore spelling. Flexible names retain the matching described below.

The export's `log/description.json` root keys come from its public producer; the fleet is checked
against its model schema and visible serialization. Declared maps, including
health readings, retain their public arbitrary keys. An undeclared or hidden
model field is a hit even if it is absent from the term list.
Other files named `description.json` receive the ordinary word search, so an
agent's public answer fields do not acquire the export record's schema rules.
This rejects hidden metadata blocks while allowing suite identifiers in public
task descriptions and catalogue values. It does not exempt protected fault
kinds, traits, canaries, fault ids or seeds inside those public strings.

Public field names are excluded regardless of a coincident hidden field. The
definitions are the visible model/table schemas, declared tool argument/answer
models, `agents.sessions.SessionReport`, `SkillManifest`, and the SQL reply
producer's dictionary. Incidental imports into `contracts.tools` do not declare
public models. This supersedes the previous restriction that only a visible log
column could make a hidden column name public.

Pass the `TruthTerms` object directly to `leak_search` to retain its policy.
`classify_truth_terms` attaches the current code-defined policy to a frozen list;
pass its dynamic `identity_names`, custom `kinds`, fault ids and full fleet when
applicable. Arbitrary plain term lists retain the legacy flexible search on
`ours`; for agent text known planted names and unclassified names qualify. Keep the policy
object to preserve custom fleet traits and their provenance.
Receipts hash the classes, planted provenance, literal spellings and public schema names as well as
the vocabulary and checker code. A class change or a newly declared public
field invalidates a receipt too, even if the static term list stays the same.

`leaks.hidden_field_terms` derives truth-only serialized names from the log,
identity, job, world recipe, experiment-world and fault-table schemas,
registered fault parameter and truth-mapping schemas, visibility-tagged model
fields and generator/detectability metadata producers. Shared visible names
are excluded using the visible schemas. Trait keys come from both generator
definitions and their actual trait dictionaries,
plus any full fleet passed to the audit. No copied list of hidden property
names needs updating when a property is added.

Flexible hidden names and trait keys are searched in any case, with word separators or
camelCase. Literal underscore spellings count wherever they occur, including
plurals and digit suffixes; boundaries apply to the spaced and joined forms.
Fault kinds are searched in identifier forms (snake, camel, hyphen,
dot and any case); separate words in fleet prose are allowed. Generic truth
words and canaries are searched in prose. A discovered single-word key is
searched as a name: a label (`row=2`), JSON key, Parquet column name or filename.
It is forbidden in ordinary prose only when deliberately listed by the checker.
The explicit prose vocabulary includes `nominal`, `params`, `schedule`,
`generator`, `haze`, `decoy`, `inert`, `confounder`, `superadditive`, `additive`,
`detectable`, `planted`, `injected`, `counterfactual`, `oracle` and `verdict`,
alongside the generic truth words above. Discovered plurals of those prose
terms do not inherit the prose ban: `verdicts` is a name, while `verdict` is
deliberately forbidden in prose. Ordinary `drift`, `level`, `outcome` and `interaction` remain
allowed in prose; names shared with visible contracts, such as `outcome`, are
public. The decision that `level` is ordinary supersedes its earlier prose
ban. Spaced/joined compound matches have boundaries at both ends: `as_run`
cannot match inside "has run". This single-word rule applies to every class,
including new fault kinds, hidden properties and unclassified names; discovering
one never adds an ordinary word to the deliberate prose list. Names still count
as keys, columns, filenames, labels, backtick identifiers and dot-path components.
Assignments (`frames=2`) and colon labels at a line or record-item boundary
(`frames: 2`) count; a prose phrase ending in a colon ("from the gripper and
the frames: approach") does not. Multipart names keep their class rules, and
the agent-written surface is unchanged. `traits` remains a data term
and is excluded from the narrower source-code terms. Parquet checks include
decoded nested field metadata and nested text values, even when the Arrow
schema or data pages encode the literal bytes.

The check also derives keys and columns from actual run truth, sibling runs,
world files and the oracle directory at audit time, including records written
by other modules. Engine `run.json` and runner bookkeeping are excluded.
Future oracle/scoring field names come from their code declarations and record
producers, before those files exist. Truth-summary names, including
`observed_carriers`, `detectable_causes`, `patterns_present` and
`expected_for_them`, come from `worlds.summary`'s dict keys and assignments;
the runner's surrounding bookkeeping remains excluded. The receipt retains the exact audited term
list and its digest. Matching it uses current checker/code-defined vocabulary
and export bytes, without rediscovering files a reader later wrote. Oracle
planning, replay and other reader writes cannot invalidate an unchanged export.
Visible request/answer schemas come from the declared tool interface, not
incidental imports; their declared model names are also public concepts.
Oracle runs contribute their truth; their visible log, bookkeeping records
and staging trees are excluded, and sources are read once. A frozen regression
retains the terms of `main` at `cfedf66`. Name-valued columns, map keys, CSV
headers and Arrow metadata receive name matching.

Spaced forms of flexible hidden names followed by a number are flagged on
purpose: `image size 360` and `max steps 280` still name hidden execution fields.
Structural record names do not acquire that rule.

Every seed in the truth identities is checked as a digit string in text and
as an integer in integer columns and as its representable floating-point value
in floating columns, including nested values and JSON numbers. Floating checks
cover double, single and half precision, including a seed rounded by its cast.
Fault identifiers are whole-word terms. Other low-entropy values, such as a fault
level of 2.0 or a robot property of 232.5, rest on the allow-list alone; the
search does not infer their meaning from an otherwise permitted number.

A word search cannot recognize shortened names or paraphrases. It also cannot
recover meanings encoded in transformed numbers, ordering or video pixels.
Those routes belong to the separate adversarial audit. A clean word search
establishes this literal boundary and does not establish complete blinding.

The decoded search covers supported Arrow text, binary, dictionary, list/view,
struct, map and JSON values. Unsupported columns, unreadable files and invalid
text are hits. UTF-8 and BOM-marked UTF-16 text are decoded, with suffix case
ignored. Packed files such as ZIP, gzip, NPZ and Feather in an export are
refused rather than searched incompletely. Visible MP4 containers receive the
raw-byte and structural checks; their image content needs a separate audit.

Raw-byte checks keep the existing literal generic words of at least five
characters. Flexible hidden names need at least eight letters or digits and
ASCII separators in opaque binary; shorter names retain the full decoded-text
search. This bounds incidental matches in compressed video without treating
arbitrary binary runs as Unicode separators. The analytic regression uses
independent uniform bytes, not a claim about the distribution of real videos;
the complete stored-video scans provide the measured evidence.

Host-side checks can append receipts beside an export under `export-checks/`.
Each receipt retains the audited term list and its digest, and binds the
export's names and bytes to the current code-defined vocabulary and
checker/allow-list code. A previous clean flag alone is insufficient. Receipt
matching does not prove source-log equivalence or per-episode completeness;
that work remains open.

For a new export, the runner audits the staging tree once and writes a receipt
bound to its relative names, bytes and intended final path. Only a clean tree
is atomically renamed to `<world>/export`. A refused tree never appears there,
and a reader still requires the published bytes to match the receipt. A kill
after publication can recover from that receipt without rebuilding the export.

From this checkout, certify a stored world without the runner:

```sh
uv run --locked python -m fleetlab.log certify /home/cs/.cache/robots-probes/worlds/w41
```

For the rebuilt D1 export, add
`--export /home/cs/.cache/robots-probes/worlds/w41/export-review-20261009T173751Z`.
The command prints the report and new receipt path, exits 0 only when clean,
and appends a receipt without rewriting the export or any stored record.
It finds runs through `WorldPaths` using the world root's name, independently
of a deliberately neutral visible fleet id.
When a live-data or teleoperation reader refuses an uncertified export, its
error explains the missing or stale receipt and includes this command for the
actual world and export path. Each fresh audit records its elapsed seconds.

Deriving hidden label definitions stored as values, rather than actual keys or
columns, needs a visibility rule: staged truth records also contain public probe
labels. A blanket rule was rejected for its visible `grip_force` regression.
Exported `metric` and `key` values still receive name matching against the
derived terms.

`export_runs(..., audit_fleet=full_fleet)` lets the check learn world-specific
hidden keys while `fleet=observed_fleet` supplies only the observed records.
The audit fleet never contributes exported records.

Regression evidence: producer-added property and schema-field injections,
trait-free fleet and no-fleet audits, code/data treatment of `traits`, hidden
metadata in an encoded nested Arrow schema, and separate audit/export fleets.
The first repair's complete stored exports of `w41`, `w42` and `w43` each passed the
check with 248 terms over all 3,033 files and no structural problems. There
were no hits to report by file, field or term. The rebuilt D1 export is
`~/.cache/robots-probes/worlds/w41/export-review-20261009T173751Z/`.
Its 1,008 episodes, 186,439 telemetry rows and 5,046 event rows are preserved.
The only differing file is `log/events.parquet`: 35 stall details changed;
every other file is byte-identical. Original stored exports, D1 run and world
files passed before/after SHA-256 comparison. The source audit also passed
over 11 tool-description texts and 43 mounted source/skill files. These are
CPU/data and source-bundle checks, with no authenticated agent launch.

Evidence is retained under
`~/.cache/robots-probes/export-review/20261009T173751Z/`, including the rebuilt
run copy, full hit reports, hashes, every changed event, source audit, and the
delivery's check log and source receipt. The D1 rebuild derives telemetry
events from a new run copy and retains every measured collision event; it
does not repeat contact simulation. The interruption regression is a CPU
fixture at publication, not a real SIGKILL or power-loss test.
`make check` exited 0: 1,090 tests passed, 2 were skipped and 62 simulator/model
tests were deselected. No simulator or stored-chain replay was run for this
data-only repair.

These checks establish the literal/structural boundary;
they do not prove that public identifiers cannot encode hidden seeds, or that
experiment replies and every agent launch are gated. Follow-ups from the
bounded audit and the metadata repair are tracked separately. No agent call or
simulator run is required here.

### First follow-up measurements (`88901a8`)

The final follow-up scans used the same repaired checker on every file, with
fresh receipts beside the stored worlds. No stored export or `runner.json`
was changed. Counts below are distinct derived terms, files searched, word
hits and structural problems respectively.

| Export | Terms | Files | Hits | Structural problems |
|---|---:|---:|---:|---:|
| stored `w41` | 3,866 | 3,033 | 0 | 0 |
| stored `w42` | 3,867 | 3,033 | 0 | 0 |
| stored `w43` | 3,862 | 3,033 | 0 | 0 |
| rebuilt D1 | 3,866 | 3,033 | 0 | 0 |

There are no hit file/field/term tuples to report. The complete term lists,
counts, hashes and receipts are retained under
`~/.cache/robots-probes/export-review/20261009T203851Z/`. Scan durations were
351.51, 384.23, 413.76 and 352.43 seconds in the table's order, on the shared
CPU while tests and other builders ran; these are not isolated benchmarks.
Earlier candidate scans and their incidental compressed-byte matches remain
as superseded evidence. The final `make check` exited 0: 1,156 passed,
2 skipped and 62 simulator/model tests deselected. Binary, oracle-source,
receipt-decoding and regression findings were repaired before these scans.
The subsequent external review found the gate-text and oracle-readiness
defects repaired below.
No simulator, real SIGKILL, authenticated agent or fresh mounted-source bundle
was exercised in this follow-up. D1 remains authorized for procedure development
with provisional scores; literal cleanliness does not prove
complete blinding.

### Second follow-up measurements (`811cda9`)

The eleven tool descriptions, standing prompts and rendered engine event
details pass against a fixture containing 429 actual `w41` truth-side key
names. This test and the CPU test that runs the runner, oracle planning,
oracle execution and resumed oracle execution both fail on `88901a8` and pass
on this repair. The oracle test checks readiness after each operation and
after an additional reader writes a file. No oracle code was changed.

Of the external review's same 165 planted leaks, `88901a8` caught 126 and this
repair catches 127; none of the previously caught plants now passes. The
additional detection is a seed in a floating column. The word search still
misses 38 plants, of which the structural allow-list also permits 22; these
are the documented shortened-name, paraphrase and low-entropy limitations.
Four separate mutation runs now fail on assertions: trusting a false-clean
receipt, omitting a check-source hash, subtracting a hidden column through
an incidental public-model import and ignoring an unsupported Arrow column.

All four final certification commands exited 0, with no hits or structural
problems. Each searched 3,033 files. The stored exports and `runner.json`
records have identical before/after SHA-256 hashes.

| Export | Distinct terms | Audit seconds | Hits | Structural problems |
|---|---:|---:|---:|---:|
| stored `w41` | 3,847 | 102.58 | 0 | 0 |
| stored `w42` | 3,848 | 103.39 | 0 | 0 |
| stored `w43` | 3,843 | 102.94 | 0 | 0 |
| rebuilt D1 | 3,847 | 100.07 | 0 | 0 |

The timings include both export digests, term discovery, word search and
structural checks, excluding the receipt write. The four commands ran
concurrently on the shared CPU; this is not an isolated benchmark. The search
now rejects impossible matches by a literal anchor before running each regex.
The runner makes at most one fresh audit per export per pass, records
`audit_seconds`, and reuses a matching receipt with zero fresh audits.

Evidence, per-export reports, exact term lists, baseline failures and mutation
results are under `~/.cache/robots-probes/export-review/20261009T215026Z/`.
`make check` exited 0: 1,168 passed, 2 skipped and 62 simulator/model tests
deselected. These establish the specified CPU/data guarantees. No simulator,
GPU oracle, authenticated analysis agent, real SIGKILL or new mounted-source
bundle was exercised in this round; `w44` and `w45` were left unopened as
instructed. D1's provisional-score and complete-blinding limitations above
remain in force.

### Third follow-up measurements

The three new regressions fail on isolated `811cda9` source and pass on this
repair. They check 252 plural/digit-suffixed variants of `main`'s 84 frozen
terms, refuse a leak planted in a run's own log without ever renaming that tree
to `export` (including a publication spy), and certify a world whose fleet id
differs from its root name. The previous gate-text and oracle-readiness tests
still pass. Fresh successful and refused exports both take one audit; receipt
reuse takes none, and refusals record their audit duration.

The role/pattern prose policy has no conflicts with 40 tool, prompt and rendered
event texts from `main` at `080a696c52381391af90a1cd6844bac1fd662f27`.
Those same texts pass against the actual truth-side files of each of `w41`,
`w42` and `w43`, with zero hits. The regression suite also retains the
cache-independent fixture of real `w41` key names.

The same 165-plant battery catches 124 plants, compared with 127 on `811cda9`.
The three lost detections are deliberate consequences of the explicit
ordinary-`level` decision: event prose `level 2.0`, event prose `level -> 2.0`,
and `list<string>` values `['level','2.0']`. The list's unknown column is still
structurally refused. Of the 41 word-search misses, 24 also pass the structural
allow-list. No `main` term was removed or lost its literal matching behavior.

All four certification commands exited 0, each searching 3,033 files and
finding zero hits and zero structural problems. Export and historical
`runner.json` hashes stayed unchanged; only new side receipts were appended.

| Export | Distinct terms | Audit seconds | Hits | Structural problems |
|---|---:|---:|---:|---:|
| stored `w41` | 3,868 | 106.20 | 0 | 0 |
| stored `w42` | 3,869 | 108.65 | 0 | 0 |
| stored `w43` | 3,864 | 109.87 | 0 | 0 |
| rebuilt D1 | 3,868 | 105.53 | 0 | 0 |

These durations include tree digests and the audit, excluding receipt writes.
The four commands ran concurrently on the shared CPU. Evidence, exact term
lists, old-head failures, gate texts and plant comparison are under
`~/.cache/robots-probes/export-review/20261009T223340Z/`.
`make check` exited 0: 1,172 passed, 2 skipped and 62 simulator/model tests
deselected. No model session, simulator, GPU oracle, authenticated analysis
agent or real kill was run in this round. `w44` and `w45` stayed unopened as
instructed; their certification is the orchestrating session's check. The
remaining blinding and integrity limits above are unchanged.

### Structural-name repair measurements (`fix/export-terms`)

The cache-independent fixture `tests/data/export-skills-e0d1954.json.gz`
freezes the three actual worlds' vocabulary and the pipeline, schema, libraries,
mounted source and 26 saved sessions selected by the skills review. The caller
is unchanged from `e0d1954`. On actual stored-world inputs, launch-text/schema
refusals fall from 6 of 6 to 0 of 6 (three worlds, two conditions). Both full
and stripped libraries pass on each world, as does the 26-file frozen mounted
bundle. Canary, fault-kind, trait, fault-id and seed probes still refuse on each
world.

Before the explicit surface decision, saved-state refusals fell from 24 of 26
to 23 of 26. That acceptance was blocked: unchanged notes in `20261009-r09/w41-logs-1` and
`20261009-r13/w42-logs-1` contain `gripper-wear` and `gripperwear`. Both must
match the registered `gripper_wear` under the required identifier-form rule.
Other notes contain deliberately forbidden role/pattern words and exact hidden
labels. Accepting all these states while refusing those forms needs a policy
decision, recorded separately; no protected name was silently dropped.
The fixture verifies the saved files' hashes and retains the original spellings.
The explicit surface policy above resolves those two collisions on agent text.

The same plant battery catches 124 of 165 before and 131 after, with no lost
detections. The seven additions are undeclared description metadata fields,
refused through their schema rather than inferred numeric meanings. The 588
forms of `main`'s 84 frozen terms retain all 576 detections; the same twelve
alphabetic prefixes do not match. There are 34 remaining search misses,
including 17 also allowed structurally; the literal-boundary limits above remain.

All four export checks exit 0. Each searches 3,033 files and finds no word hits
or structural problems, so there are no file/field/term hit tuples to list.

| Export | Distinct terms | Audit seconds | Hits | Structural problems |
|---|---:|---:|---:|---:|
| stored `w41` | 3,932 | 94.00 | 0 | 0 |
| stored `w42` | 3,933 | 96.67 | 0 | 0 |
| stored `w43` | 3,928 | 91.75 | 0 | 0 |
| rebuilt D1 | 3,932 | 89.92 | 0 | 0 |

These sequential audits ran on the shared CPU, alongside at most two other
data-check processes, and include tree digests and checks but exclude receipt
writes. Export and historical `runner.json` hashes remain unchanged. Only new
side receipts were appended; all match the current checker. Complete reports,
before/after state hits and excerpts, source hashes, fixture provenance and
final delivery validation are under
`~/.cache/robots-probes/export-terms/20261010T035515Z/`.

The one full `make check`, on `175e874`, exited 2: 1,418 tests passed, one
failed, two were skipped and 62 simulator/model tests were deselected. Its sole
failure was the integration export's legitimate suite identifier in public task
prose; task identifiers now use structural matching. The description schema
guard preserves the existing metadata-plant detection
and respects arbitrary public health-map keys. It applies only at the export's
contract path, leaving unrelated agent `description.json` files under ordinary
word matching. The final focused suite, including that integration regression,
passed all 129 tests in 192.89 seconds. Final lint, formatting and type checks
exit 0 over 295 source files. The full suite was not repeated under the explicit
one-full-check limit, so a full-head `make check` pass is not claimed.
No model session, authenticated agent launch, simulator, GPU job, `check-sim`
or `smoke-world` ran. `w44` and `w45` stayed unopened. These checks establish
the measured literal boundary, not complete blinding or successful agent runs.

### Explicit-surface measurements (`44282ac`)

This round's single-word agent policy is superseded by the multipart-only
policy and measurements below; these results retain the earlier denominator.

On the committed checker, 70 focused term/gate tests pass, including missing
surface errors, the two saved-note collisions, camelCase versus lowercase
joined forms, fault ids, canaries and numeric/floating-point seeds. An earlier
broader focused run passed 204 tests, including export, reader, event,
runner/oracle and integration regressions; the final changes after it affect
only agent matching and its additional tests. The same 165-plant battery
catches 131 before and after, losing no detection. All 576 previously caught
forms of the 588 forms of `main`'s 84 terms remain caught.

With actual stored-world inputs, pipeline/answer-schema checks pass in all six
world/condition cases, both libraries pass in all six world/library cases,
and all 30 protected checks refuse (five probe types, three worlds, both
surfaces). The unchanged `e0d1954` state-audit caller is run through an
evidence-only adapter that supplies the required `agent` argument, keeping
its state parsing and public-word handling intact. Of 26 saved sessions,
three pass and 23 refuse for exact identifier words/names/labels. The
remaining hits include `nominal`, `additive`, `detectable` and `max_steps`;
single-word identifiers and record labels can coincide with English, so a
hit does not establish where the agent learned the name. Neither saved
`gripper-wear`/`gripperwear` collision is a hit on agent text.

Stored `w41`, `w42`, `w43` and rebuilt D1 each still have zero hits and zero
structural problems across 3,033 files. Their term counts are respectively
3,932, 3,933, 3,928 and 3,932; audit durations are 121.63, 130.56, 128.15
and 138.11 seconds. These sequential shared-CPU durations include checks and
digests, excluding receipt writes, alongside at most two other data-check
processes. Export and original `runner.json` hashes remain unchanged; only
new side receipts were appended.

Complete session/file/field/term refusals, export reports, tree/record hashes,
plant comparisons, probe results and test logs are under
`~/.cache/robots-probes/export-surfaces/20261010T042623Z/`; see
`remaining-refusals.md` and `delivery.json`. Agent/skill source is untouched:
its owner must adapt the scanner calls to the required argument. The only
additional test call-site edit is in `test_log_reader.py`. This round ran
focused tests only; the orchestrating session runs `make check`. No live
agent launch, model session, GPU/simulator job or smoke run was performed;
`w44`/`w45` remain unopened. Confidence is high in the measured lexical
checks; they do not establish complete blinding.

### Multipart-only agent measurements (`d8c1f20`)

This round's execution-field refusal is superseded by the fleet-property and
pattern vocabulary below; it is retained here as the measured earlier result.

The same unchanged `e0d1954` state-audit caller, through the explicit-agent
adapter and actual `w41`/`w42`/`w43` truth inputs, now accepts 25 of 26 sessions
instead of 3. Only `20261009-r14/w42-logs-2` refuses: its `max_steps` SQL alias
names a hidden `Job` field. The two hits are `diagnosis.json` at
`/causes/4/evidence/0/query` and `submissions.jsonl` line 1 at
`/answer/causes/4/evidence/0/query`. No single-word or natural-form hit remains.
Public vocabulary namespaces supply public leaf names, so `sampling_steps`
shown as `control.sampling_steps` passes too. An actual fault id still refuses
even when its spelling is public or absent from the supplied term list.

All six actual-world pipeline/schema cases and six library/world cases pass.
All 30 protected checks refuse: canary, identifier kind, identifier trait,
fault id and seed on three worlds and both surfaces. The plant battery remains
131 of 165 before and after, with no lost detection; the 588 frozen main forms
retain the same 576 detections. The 84 focused term/gate tests pass at exit 0
in 120.62 seconds, including a cache-independent, hash-checked recreation of
all 26 states and checks of prose, keys, filenames, columns and bytes. Focused
lint, formatting and type checks also exit 0.

All four fresh export audits exit 0 over 3,033 files each, with zero hits and
zero structural problems. Stored `w41`, `w42`, `w43` and rebuilt D1 use
3,932 / 3,933 / 3,928 / 3,932 terms and take
119.32 / 134.13 / 143.40 / 140.40 seconds respectively. These sequential
shared-CPU audits include checks and digests, excluding receipt writes, with
at most two other data-check processes. Export and original `runner.json`
hashes remain unchanged; only fresh side receipts are appended.

Evidence is under `~/.cache/robots-probes/export-surfaces/20261010T044707Z/`,
including the complete remaining file/field/term hits, before/after states,
protected probes, plant comparison and export receipts. Agent/skill code is
untouched. No live agent launch, model session, simulator/GPU job or smoke run
was performed; `w44`/`w45` remain unopened. The orchestrating session owns the
full `make check`; this round verifies the measured lexical tripwire only.

### Fleet-property and pattern vocabulary (`f2a272e`)

Agent name eligibility now comes from fault kinds, the truth role contract and
assignments to the composer's role map, generator/full-fleet trait keys, and
hidden visibility tags on fleet-model attributes. Execution and truth-record
schemas retain their previous matching on `ours`; their layout keys never
qualify on `agent`. This includes `max_steps`, `true_success`, `image_size` and
`onset_day`, in both classified and plain vocabulary lists. A false refusal
costs an analysis, so the agent policy errs toward allowing its own text.

All 26 unchanged `e0d1954` saved-state audits now pass against actual
`w41`/`w42`/`w43` truth inputs, up from 25; there are no session/file/field/term
refusals to list. The evidence-only adapter supplies the explicit agent surface
without changing the caller's state parsing or public-word handling. The six
pipeline/schema and six library/world checks pass, and all 30 protected probes
refuse (five types, three worlds, both surfaces). All 83 focused term/gate tests
exit 0 in 127.10 seconds, including complete, hash-checked state recreation and
future hidden-fleet-field/pattern-role definitions. Focused lint, formatting
and mypy checks pass. The plant battery stays at 131 of 165 before and after,
with no losses, and the main-form battery retains all 576 of 588 detections.

The four fresh export audits each find zero hits and structural problems over
3,033 files. Stored `w41`/`w42`/`w43` and rebuilt D1 use
3,932 / 3,933 / 3,928 / 3,932 terms and take
120.16 / 104.84 / 96.64 / 96.63 seconds respectively. These sequential
shared-CPU durations include checks/digests and exclude receipt writes, with
at most two other data-check processes. Original export/runner hashes are
unchanged; only fresh side receipts are appended.

Evidence is under `~/.cache/robots-probes/export-surfaces/20261010T050327Z/`.
No agent/skill source was edited, and no authenticated model launch, simulator,
GPU job or smoke run was performed; `w44`/`w45` stay unopened. The orchestrating
session runs the full `make check`. These results establish the measured
tripwire, rather than successful live execution or complete blinding.

Implements section 11 of [the contracts](../../../docs/design/contracts.md)
on top of `fleetlab.contracts`, which it imports and
never changes. The schema (every column, its type and its visibility) is
`fleetlab.contracts.log.SCHEMA`; this package writes, exports and reads it.

## Layout

A run directory, as the contracts fix it:

```
run_dir/
  log/      visible   episodes.parquet, steps/<episode_id>.parquet, events.parquet, health.parquet,
                      change_log.parquet, maintenance.parquet, description.json,
                      video/<episode_id>.scene.mp4, video/<episode_id>.wrist.mp4
  truth/    hidden    identities, faults_active, effective_policy, verdicts, replays, job_outcomes,
                      oracle, decisions (.parquet); chain/<job key>.npz; canary.txt;
                      parts/<job key>.json (staged rows) and parts/_merged.json
  run.json  hidden    contracts.rollout.RunRecord; extra["episodes_per_robot_day"] is the world's
  .staging/           temporary files, cleared when a writer opens the run
  .writer.lock        held by the one RunWriter of the run
```

An export, which is what the tool server mounts:

```
out_dir/
  log/                the visible tree above, for the exported episodes only
  data_dictionary.md  every visible table and column, and the conventions
```

## Writing a run

```python
from fleetlab.log import RunWriter, EpisodeWriter

n = world_spec.episodes_per_robot_day
with RunWriter(run_dir, run_record, episodes_per_robot_day=n) as run:  # before workers start
    run.write_model(fleet, task_classes)              # description.json, change_log, maintenance
    run.write_table("health", rows)                   # run-scoped tables are written whole
    for job in run.pending(jobs):                     # after a crash: what is left
        ep = EpisodeWriter(run_dir, job, task_text=text, episodes_per_robot_day=n)  # in a worker
        ep.add_step(telemetry, frames)                # per state; frames optional, upright, both cameras
        ep.add_chain(step, state_hash, image_hashes, executed_action)  # None in the first row
        ep.add_event(step, "grasp", "closed on the bowl")
        ep.set_verdict(true_success, reason, libero_first_success_step, success_step,
                       placed_step=placed_step)       # reason from contracts.env.VERDICT_REASONS
        ep.finish(outcome=success_after_data_hook)
        # any other job: ep.finish(success=, steps=, placed_step=, libero_step=); nothing visible
    run.finalize()                                    # run-level tables from the finished jobs
    run.update_record(chain_sample=..., chain_ok=...) # extra is merged
```

Conventions: `steps` row k is the state after k actions and the action
commanded in it; the last row has no command (null). Chain row k holds the
hashes of state k and the executed action that led to it. A job's first chain
row, the state after the reset (step 0) or the state restored from a snapshot
(the snapshot's step), has no executed action: pass None, it is stored as NaN,
and a comparison of a continuation's chain with the uninterrupted episode's
skips that one action. A run job's steps and chain have one row per state, and
`finish` checks that their step numbers agree. `episodes.steps` and
`duration_s` come from the last step row; the ids, day and video paths from the
job; `started_at` from `contracts.identity.started_at_for(identity,
episodes_per_robot_day)`, so it is the caller's no longer. `identities`,
`faults_active` (with `params` and `nominal` as JSON), `effective_policy` (the
nominal spec unless `set_effective_policy` was called), the `replays` row of a
replay from the reset and the `job_outcomes` row of every job are derived from
the job and `finish`. Frames get their `frame_index` from the writer.

## Exporting and auditing

```python
from fleetlab.log import export_runs, audit_export, LogReader

report = export_runs([run_a, run_b], out_dir, fleet=fleet, task_classes=task_classes)
leaks = audit_export(out_dir, [run_a, run_b], fleet=fleet, texts={"prompt": prompt},
                     code={"tool server": source})
reader = LogReader(out_dir, materialize=True, lock=True)  # the tool server's way, the only closed one
reader.query("SELECT robot_id, avg((outcome = 'success')::int) FROM episodes GROUP BY 1")
reader.frames("e000003", [0, 40, 80], "scene")
```

`LogReader` with `lock=True` but without `materialize` keeps DuckDB inside
`log/` (it must read the Parquet files there), which still lets a query read
any file under `log/` and write there with `COPY ... TO`; on a run directory,
`truth/` and `run.json` stay out of reach. The tool server runs materialized
and locked: DuckDB then opens no file at all.

`fleetlab.log.runfiles` reads a whole run, hidden tables included
(`read_table`, `read_identities`, `oracle_rows`, `open_run` for the
dashboard, `full_data_dictionary`). Never hand it to an agent.

## The rules, and where each is enforced

1. **Visible tables are built from the allow-list.** A row for a visible table
   may carry exactly the columns of `contracts.log.allowed_columns`; a hidden
   or unknown column, or a missing one, raises `RowError`. The writer refuses
   to open at all while `contracts.log.schema_problems()` is not empty (a
   hidden column in a visible table, a visible measured time). The export
   reads every visible table through the allow-list, casts it to the declared
   types and writes it anew, so no column, metadata or stray row rides along.
2. **Neutral identifiers.** The export refuses a run whose `log/` tree fails
   `contracts.log.log_tree_problems` (non-neutral ids in file names or id
   columns, unknown files, hidden fields in the description) and any episode
   id that is not neutral or appears in two runs.
3. **No measured time in the visible tree.** Schema: no visible column with
   `clock="measured"`; files: no column named like one; videos: the MP4 movie,
   track and media headers must carry zero creation and modification times
   (`export.mp4_times`). Wall-clock times go to `run.json`.
4. **The visible tables are a deterministic function of the identities.**
   Run-level tables are sorted by episode (and step), so the order in which
   jobs finish does not matter; Parquet is written without timestamps; the
   video encode is repeatable (below). Tested: two runs that finish their jobs
   in opposite orders give byte-identical `log/` trees, videos included.
5. **A job is done when its chain exists.** Every file goes to `.staging/`
   first, is flushed to disk (`fsync`) and renamed into place, and its
   directory is flushed after the rename; the chain is written last. A resumed
   `RunWriter` refuses a record that differs in software, batch size, replay
   rules or episodes per robot and day (the rest of the record, `extra`
   included, is kept from the file), clears the staging directory and deletes
   the files of jobs without a chain (`discarded`), so `pending(jobs)` reruns
   exactly those. `finalize` merges only finished jobs and can run again at
   any time; the export refuses a run that is not finalized
   (`run_status(...).finalized`). A worker of the crashed engine that is still
   alive (an orphan) can go on writing into the resumed run; what it writes is
   the same deterministic content the rerun writes, file by atomic file, so the
   run stays consistent, but stop orphans before resuming all the same.
6. **Replays write nothing visible.** Only a "run" job may add steps, events or
   frames; a replay from the reset adds one `replays` row
   (`fault_list_hash(job.effective_faults)`, its true outcome) and its chain.
   Every job, of any mode, adds one row to `job_outcomes`: the faults removed
   and changes added (JSON), the snapshot step it started from, its true
   outcome, its last step, the first step the fleet's check found the object
   placed and the first step LIBERO's check fired. Verdict reasons must be
   one of `contracts.env.VERDICT_REASONS`.
7. **The canary.** A random token in `truth/canary.txt`, in the key-value
   metadata of every hidden Parquet table and in every staged part. The audit
   refuses to run when a run's canary is not planted.
8. **The word search** (`leaks.leak_search`) covers file and directory names,
   the raw bytes of binary files (literal generic words of five characters or
   more, flexible hidden names of eight letters/digits or more), every
   distinct text value, column name and metadata entry of each Parquet file,
   `description.json` key by key, and any texts given. Terms
   (`leaks.truth_terms`): generic words about how the data was made, hidden
   column names no visible column shares, the fault kinds of the registry
   (`fleetlab.faults.load_all()`) and of the runs, the robots' trait names,
   the canary tokens. Source code mounted into the sandbox is searched with
   the narrower `code_terms` (kinds, traits, canaries; the two kinds of
   `contracts.actions.ADDED_CHANGE_KINDS` exempt), because the contracts'
   source names the hidden tables by design. Occurrences inside names of the
   visible vocabulary do not count ("delay" in `control.action_delay_steps`),
   and the description's vocabulary block is skipped when it is the
   contracts' own. `export_runs` runs the audit before the export is renamed
   into place and refuses on any hit.
9. **Video bytes are never hashed.** The chain holds hashes of the rendered
   images; videos are derived data.

## Video encoding

The skeleton's MP4 files differed between two runs of identical images. The
cause was the encode path, not the images: the skeleton encoded through PyAV.
Measured 2026-10-07 on 40 of the skeleton's videos (scene and wrist, 360x360,
28 frames on average), each encoded four times, 16 encodes in parallel (load
average 7 to 9):

| path | files whose four encodes differ | seconds per file | bytes per file |
|---|---|---|---|
| ffmpeg through imageio, one thread, bit-exact (`write_mp4(..., repeatable=True)`) | 0 of 40 | 0.074 | 73,517 |
| ffmpeg through imageio, threads automatic | 0 of 40 | 0.080 | 74,253 |
| PyAV, as the skeleton (`spikes/skeleton/fleet.py`) | 3 of 40 | 0.079 | 76,352 |

PyAV also gave differing files with `thread_count=1` in a first probe, so the
thread count is not the cause there; I did not look further. The writer uses
the first row: libx264, crf 23, preset veryfast, one thread, `+bitexact`, no
metadata. It costs nothing measurable at the fleet's parallelism. Byte
identity holds for one installed build of imageio-ffmpeg's ffmpeg: libx264
writes its version and options into the stream, so another build gives other
bytes from the same frames.

## Choices made where the contracts are silent

- Staged rows per job (`truth/parts/<key>.json`) instead of one Parquet file
  per table and job: one file per job, written atomically before the chain.
- The run-scoped tables (`health`, `change_log`, `maintenance`, `oracle`) are
  written whole by `RunWriter.write_table`; `finalize` writes an empty file for
  any that was never written, so every visible table exists.
- Several runs export together; fleet-level tables come from the `fleet` if
  given, else from the runs, which must agree.
- `job_outcomes.steps` of a job that is not a run is the last step of its
  chain unless `finish(steps=...)` says otherwise (and then they must agree).
- Parquet in the visible tree is written without Arrow's stored schema,
  because pyarrow 25 cannot read back a null in a fixed-size list column
  (`commanded_action` of the last row) when that schema is present. Readers
  cast to the declared types; DuckDB reads the vectors as lists (`[1]` is the
  first element). Hidden tables keep the stored schema, since it carries the
  canary.
- The reader picks the logged frame nearest to a requested step, the earlier
  one on a tie.

## Not verified

- Anything with the simulator or a policy: all tests use toy data made from
  the contracts' types (`tests/log_toy.py`); the engine will
  be the first real writer.
- Scale: export and DuckDB over tens of thousands of per-episode steps files,
  memory of an episode's buffered frames (about 54 MB for 280 steps at 360x360
  with frames every fourth step), the time `finalize` takes with many jobs.
- Several writers in several processes at once on one run: the lock keeps out
  a second `RunWriter`, and episode writers in workers write disjoint files,
  but no test runs real worker processes.
- The MP4 time check reads the movie, track and media headers only; other
  boxes that could hold a date were not looked for.
- The reader in the real sandbox (bubblewrap, read-only mount): the lock was
  tested in-process only. Importing `fleetlab.log.reader` there brings the
  `contracts` modules with it, whose source names the hidden tables and
  columns; the contracts decided that this is class knowledge, not world
  truth (contracts note, section 12), and the leak search over mounted source
  uses world-specific terms only.
- Power loss: files are flushed to disk before each rename and directories
  after it, but no test cuts the power; a crash of the process is what was
  tested.


## Strict trees and sourced agent text (`3fb26d1`)

This replaces the surface selector on tree/code entry points with the signatures
at the top of this README. Trees and mounted code always use the strict policy;
only separately named agent text receives the agent policy. A regression puts
`true_success` and `onset_day` columns plus planted event text in an export,
supplies innocent agent notes in the same call, and still refuses the tree and
planted mounted code. Another regression verifies that passing `surface` to
either tree entry point is an error.

Fault/property identifiers and world fault ids now count within longer
identifiers, including aliases, prefixes, plurals and digit suffixes. Tests
cover both authorships' comma/underscore grouping, hexadecimal and scientific
seed notation. Hex is derived with `hex(seed)`; the review's literal
`0xe5fda6cc` represents 3858605772, whereas the cited seed 3858582668 is
`0xe5fd4c8c`. Public added-change identifiers pass on agent text while remaining
forbidden on our text. Dynamic model discovery replaces the three-model tuple;
a new execution model proves discovery without an added name list. Unknown
names stay planted, including recipe controls, until a schema or record
producer establishes structure. Keyword fields are learned from variadic
record writers' dictionary updates, rather than unrelated function options.

At this source commit, the unchanged frozen e0d1954 state reader, adapted only
to hand named state text to `agent_texts`, accepts 24 of 26 saved sessions
(previously 26). Its original JSON/prose parsing and public statistical alias
masks remain in effect. The complete reader source is hash-checked in the
cache-independent test fixture beside the original hashed state contents.
The two refusals are exclusively world fault-id substrings in `memory/notes.md`:

- `20261009-r08/w42-logs-1/run.json`: `f10` inside `F108`, and `f14` inside `of14`.
- `20261009-r14/w42-logs-2/run.json`: `f14` inside `of14`.

These lexical matches need human interpretation against the record; they do
not establish that those agents learned fault identities. Full source paths
and contexts are in `remaining-refusals.json` and `.md`. The actual checks use
w41/w42/w43 truth-side inputs; no live/model launch was performed.

All six pipeline/schema cases and six library/world cases pass. All 30 protected
probes refuse: five types (canary, kind identifier, trait identifier, fault id,
seed), three real worlds, two authorships. The plant battery remains 131 of
165, with no lost hits, and main's seven forms of 84 terms remain 576 of 588.
The four export audits each search 3,033 files with zero hits and structural
problems. Stored w41/w42/w43 and rebuilt D1 have 3,933 / 3,934 / 3,929 / 3,933
terms, with audit times 82.16 / 90.58 / 89.25 / 90.08 seconds respectively
(shared CPU, checks/digests included, receipt writes excluded). Original export
and runner hashes remain unchanged; only fresh side receipts were appended.

The four export-focused suites pass 199 tests in 203.43 seconds, and the runner
and event suites pass 40 tests in 109.08 seconds, with one simulator case
excluded. Focused Ruff lint/format and strict mypy pass over 11 owned source/test
files. No full `make check`, GPU/model/simulator job, smoke run or w44/w45 access
was performed. Confidence is high in the measured lexical guarantees; complete
blinding and live launch behavior remain outside these checks.

Evidence: `/home/cs/.cache/robots-probes/export-surfaces/20261010T055253Z/delivery.json`, the full state and probe reports,
plant comparison, four export receipts and before/after hashes, and test logs.

## Countable fault ids in agent text (`5c9ef9f`)

The orchestrator's countable-id decision supersedes the two saved-state refusals
reported above. Only actual world fault ids whose entire spelling is one ASCII
letter followed by two ASCII digits are exempt on agent text. Every other world
fault id still refuses, including inside longer identifiers and without a term
vocabulary. Our texts, trees and mounted code still check every world fault id;
the public function signatures are unchanged.

All 26 unchanged saved-state audits now pass with actual w41/w42/w43 inputs and
the original e0d1954 state reader's parsing and public-word masks. The
cache-independent acceptance test checks the reader and state-file hashes.
Before the source change, this acceptance test and four countable-id cases
failed; all pass afterwards. Nearby shapes such as `f001`, `f1`, `ff01` and
`1f01`, plus the made-up `x9e52a18c`, remain protected on agent text.

All six pipeline/schema cases and six library/world cases pass. All 30 protected
checks refuse: canary, kind identifier, trait identifier, fault id and seed on
three worlds and both authorships. The agent fault-id probe uses `x9e52a18c`;
the controlled-text probe still uses a real countable world id. A separate
agent-text check of each world's entire countable-id set passes.
Our plant battery remains 131 of 165, with zero lost detections; main's term
forms remain 576 of 588. The four export checks each cover 3,033 files with zero
hits and zero structural problems: stored w41/w42/w43 and rebuilt D1. Their
term counts are 3,933 / 3,934 / 3,929 / 3,933; audit times are 99.42 / 97.67 /
99.63 / 131.20 seconds (shared CPU, checks/digests included, receipt writes
excluded). Export and runner hashes stayed unchanged; only new side receipts
were appended.

208 focused export tests pass at exit 0 in 251.51 seconds. Ruff lint, formatting
and mypy each exit 0 over 11 owned source/test files. No full `make check`, live
model launch, GPU/simulator job, smoke run or w44/w45 access was performed.
Confidence is high in these measured lexical checks; live launches and complete
blinding remain unverified.

Evidence: `/home/cs/.cache/robots-probes/export-surfaces/20261010T060957Z/delivery.json`,
state and probe reports, plant comparison, four receipts and before/after
hashes, and focused check logs.

## Single-word aliases and recipe configuration (`155bd74`)

The reproduced `min_detectable` hit was for the entire recipe key, rather than
the single word `detectable`. Recipe model declarations now distinguish known
configuration fields from unknown planted names on agent text only. A future
recipe model test proves that this comes from the defining code; a real hidden
fleet property of the same name still wins. Unknown names retain their previous
policy. Our texts keep the same vocabulary, structural classes and matching.
An identifier needs two nonempty parts to qualify as a multipart planted name;
leading or trailing separators cannot promote a single word.

`min_detectable`, `n_additive` and `is_nominal` pass as prose, SQL aliases and
JSON keys with each real world's terms, both classified and plain. The longer
identifiers `illumination_change_level` and `gripper_pad_age_days_mean` still
refuse. All 26 raw saved-state audits pass against actual w41/w42/w43 inputs
without caller word masks; a second cache-independent test hashes and checks
every saved file directly. Seven new regressions fail on the preceding head
and pass after this change. The three public signatures are unchanged.

All six pipeline/schema and six library/world cases pass; all 30 protected
checks refuse. The strict plant battery remains 131 of 165 before and after,
with zero lost detections, and main's forms remain 576 of 588. Stored w41/w42/w43
and rebuilt D1 each have zero hits and zero structural problems over 3,033
files. Their term counts remain 3,933 / 3,934 / 3,929 / 3,933; audit times are
130.25 / 140.00 / 148.91 / 152.56 seconds (shared CPU, checks/digests included,
receipt writes excluded). Export and runner hashes are unchanged; only fresh
side receipts were appended.

122 focused term/gate tests exit 0 in 246.35 seconds. Ruff lint, formatting and
mypy exit 0 over 11 owned files. No agents/ or skills/ edits, full `make check`,
live model launch, GPU/simulator/smoke job or w44/w45 access was performed.
Confidence is high in the measured lexical checks; live launch behavior and
complete blinding remain unverified.

Evidence: `/home/cs/.cache/robots-probes/export-surfaces/20261010T064712Z/delivery.json`,
raw state/probe reports, plant comparison, four receipts/hashes and focused logs.

## Supervisor single-word gate repair (2026-10-10)

Scanner commits `366de89` and `ba61523` keep every non-deliberate single word
name-only on our surface, independently of class, and retain uppercase fault-id
checking. Unquoted colons in prose phrases are not record labels. The frozen
`export-supervisor-batch4.json.gz` fixture records `merge/batch-4` at `81a02b0`,
the instruction builder, its launch gate, the role registry and both supervisor
skills with their hashes; the test builds the gate's actual text against w42's
real truth vocabulary. Four new regressions fail on `2c1d00f` and pass here.
The earlier test banning bare `verdicts` now asserts the decided ordinary-prose
rule; singular `verdict` remains deliberately forbidden. The agent policy and
all three public signatures are unchanged.

The final live audit uses all 3,934 actual w42 terms: the system text and both
skills, including their two mounted helper files, have zero hits. All six
pipeline/schema cases and six analysis-library/world cases pass with actual
w41/w42/w43 inputs. All 26 raw saved-state audits pass without caller word masks;
all 30 protected probes refuse (five types, three worlds, two surfaces), with a
non-countable made-up fault id on the agent surface. The plant battery remains
131 of 165 caught, with zero lost detections; main's forms remain 576 of 588
caught. All 212 focused export tests pass in 286.69 seconds; Ruff lint/format
and mypy exit 0 on the four changed Python files.

Each stored w41/w42/w43 export and rebuilt D1 export has zero hits and zero
structural problems across 3,033 files. Their audit times are respectively
89.81, 109.18, 103.27 and 102.26 seconds, including checks/digests and excluding
receipt writes. Their term counts are respectively 3,933, 3,934, 3,929 and
3,933. Export and original runner hashes are unchanged; only new side receipts
were appended. These are lexical audits, not authenticated agent launches or
proof of complete blinding. No full `make check`, GPU/simulator/smoke run or
w44/w45 inspection was performed. Evidence: `/home/cs/.cache/robots-probes/export-surfaces/20261010T091734Z/delivery.json`.
