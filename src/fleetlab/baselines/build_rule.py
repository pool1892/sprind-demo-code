"""A fix set read off the generator's build. Not a baseline of the experiment.

Every world the stage-4 recipe composes is built alike (`fleetlab/worlds/README.md`,
"One structure under every composed world"): two of the three sites run the
tasks of one suite only, the heavier batch of objects sits at one of them, and
nearly every world carries one delay that covers a whole site and stands in the
change log. This module is the rule that names fixes from that build and from
nothing else, so that a report on the test worlds can show how far it gets:

    python -m fleetlab.baselines.build_rule --export E --out NEW_DIR
    python -m fleetlab.baselines.build_rule --evaluate RULE_DIR --world-root W --out NEW.json

The first command reads a public export (its change log, the task of every
episode with its site, the catalogue and the sites' recorded conditions) and
writes fix sets in the baselines' format. It reads no episode outcome, no
health reading and no video. The second computes the rule again from the
world's own export, refuses a directory that does not hold exactly that, and
values it with the function the scorer values a baseline's fix sets with, at
the same budgets, into a file of its own. The rule is never handed to the
scorer, which refuses its directory: it is none of the methods of
`python -m fleetlab.baselines`, `make baselines` does not start it, and no
claim's margin is read against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from fleetlab.baselines.common import (
    BaselineAnswer,
    budget_label,
    export_catalogue,
    ranking,
    validate_export,
    visible_fleet,
    write_answer,
    write_json,
)
from fleetlab.contracts.actions import SCORING_BUDGETS, Action, FixSet, cut
from fleetlab.contracts.faults import Target
from fleetlab.contracts.ids import IdKind, ScopeKind, id_number, scope_kind
from fleetlab.contracts.log import DESCRIPTION_FILE
from fleetlab.contracts.tools import MIN_RANKED_ACTIONS
from fleetlab.contracts.vocabulary import ConditionKey, ConfigKey, Verb
from fleetlab.log.reader import LogReader

METHOD = "build_rule"
RULE_VERSION = "build-rule-v1"
MAX_ACTIONS = 3
# The rule as it is stated to a reader: a sentence of what the code does. Tests hold a
# literal copy. A change of the code that changes a fix set needs a new version.
WORDING = (
    "One revert for every scope and key that has an entry in the change log made at a site, "
    "then one for every scope and key with an entry made for the whole fleet, each group in "
    "the order of its earliest entry (day, site number, key). Then, for every site at which "
    "every task run is among the ten sentences of one suite and for which no site entry was "
    "read, the action that replaces its batch of objects by the site's recorded batch, by "
    "site number. An entry whose scope or key the vocabulary does not hold is left out; an "
    "action the catalogue does not hold is skipped and the next moves up. The first three "
    "actions are the answer at three and at six fixes; at four and eight hours, those of the "
    "three that fit, in order. No episode outcome, no health reading and no video is read."
)
SITE_STEP = "site change"
FLEET_STEP = "fleet change"
BATCH_STEP = "single-suite batch"
PADDING = "FORMAT PADDING ONLY; not named by the build rule; never included in a FixSet."
# The label the rule's fix sets carry inside the scorer's function. It names no baseline.
EVALUATION_NAME = "build rule"
# What the first command reads of an export, and nothing besides.
INPUTS = (
    DESCRIPTION_FILE,
    "log/catalogue.parquet",
    "log/change_log.parquet",
    "log/episodes.parquet",
)

# The two task suites of the recipe, by the sentences an export shows. An export names no
# suite, so the rule knows them as the build has them (LIBERO's own task lists).
SUITES: dict[str, frozenset[str]] = {
    "libero_spatial": frozenset(
        f"pick up the black bowl {where} and place it on the plate"
        for where in (
            "between the plate and the ramekin",
            "next to the ramekin",
            "from table center",
            "on the cookie box",
            "in the top drawer of the wooden cabinet",
            "on the ramekin",
            "next to the cookie box",
            "on the stove",
            "next to the plate",
            "on the wooden cabinet",
        )
    ),
    "libero_object": frozenset(
        f"pick up the {thing} and place it in the basket"
        for thing in (
            "alphabet soup",
            "cream cheese",
            "salad dressing",
            "bbq sauce",
            "ketchup",
            "tomato sauce",
            "butter",
            "milk",
            "chocolate pudding",
            "orange juice",
        )
    ),
}


class Entry(NamedTuple):
    """An entry of the change log, as far as the rule reads it: no value, no author."""

    day: int
    scope: str
    key: str


class Nomination(NamedTuple):
    """An action the rule names, with the step that names it and what it was read from."""

    step: str
    verb: Verb
    target: Target
    because: str


def suite_of(tasks: Iterable[str]) -> tuple[str | None, str]:
    """The one suite whose tasks a site's robots ran, or None, and why in words."""
    texts = set(tasks)
    if not texts:
        return None, "no episode"
    unknown = sorted(t for t in texts if not any(t in suite for suite in SUITES.values()))
    if unknown:
        return None, f"{len(unknown)} of {len(texts)} tasks belong to no suite the rule knows"
    found = sorted(name for name, suite in SUITES.items() if texts & suite)
    if len(found) > 1:
        return None, "tasks of " + " and ".join(found)
    return found[0], f"{len(texts)} tasks, all of {found[0]}"


def _scope_number(scope: str) -> int:
    kind = scope_kind(scope)
    if kind is ScopeKind.fleet:
        return 0
    return id_number(IdKind.site if kind is ScopeKind.site else IdKind.robot, scope)


def nominations(changes: Iterable[Entry], suites: Mapping[str, str | None]) -> list[Nomination]:
    """The rule before the catalogue and the cap: every action it names, in order, each once.

    `changes` is the change log as far as it was read; `suites` gives, for
    every site, the one suite its robots run, or None. A revert stands where
    the earliest entry of its scope and key stands, and its reason names that
    entry: with several entries on one key it is one action, which the oracle
    values as removing every fault whose change was made at that scope on that
    key, not as undoing the last entry.
    """
    ordered = sorted(changes, key=lambda e: (e.day, _scope_number(e.scope), e.key))
    entries = Counter((e.scope, e.key) for e in ordered)
    named: list[Nomination] = []
    for kind, step in ((ScopeKind.site, SITE_STEP), (ScopeKind.fleet, FLEET_STEP)):
        named += [
            Nomination(
                step,
                Verb.revert,
                Target(scope=entry.scope, config_key=ConfigKey(entry.key)),
                f"change log: {entry.scope}, {entry.key}, "
                f"entries: {entries[entry.scope, entry.key]}, the earliest on day {entry.day}",
            )
            for entry in ordered
            if scope_kind(entry.scope) is kind
        ]
    changed_sites = {n.target.scope for n in named if n.step == SITE_STEP}
    for site in sorted(suites, key=_scope_number):
        suite = suites[site]
        if suite is not None and site not in changed_sites:
            named.append(
                Nomination(
                    BATCH_STEP,
                    Verb.change_condition,
                    Target(scope=site, condition_key=ConditionKey.object_batch),
                    f"every task run at {site} is of {suite}, and no site entry was read for it",
                )
            )
    once: dict[tuple[Verb, Target], Nomination] = {}
    for nomination in named:
        once.setdefault((nomination.verb, nomination.target), nomination)
    return list(once.values())


def _record(nomination: Nomination, action: Action | None = None) -> dict[str, Any]:
    target = nomination.target if action is None else action.target
    record: dict[str, Any] = {
        "step": nomination.step,
        "because": nomination.because,
        "verb": nomination.verb.value,
        "target": target.model_dump(mode="json"),
    }
    if action is not None:
        record["action"] = action.model_dump(mode="json")
    return record


def build_rule(export: Path) -> BaselineAnswer:
    """The rule's fix sets for one export, in the baselines' answer format.

    Reads the change log (day, scope, key), the site and task of every episode,
    the catalogue and the system description. An action the catalogue does not
    hold is skipped and listed under `skipped` in the metadata. An entry of the
    log whose scope or key the vocabulary does not know is left unread and
    listed, and so is a site that has episodes and no description.
    """
    export = export.expanduser().resolve()
    fleet = visible_fleet(export)
    with LogReader(export) as reader:
        logged = reader.query("SELECT day, scope, key FROM change_log").to_pylist()
        ran = reader.query("SELECT DISTINCT site_id, task_text FROM episodes").to_pylist()
    changes: list[Entry] = []
    unread: list[dict[str, Any]] = []
    for row in logged:
        entry = Entry(int(row["day"]), str(row["scope"]), str(row["key"]))
        try:
            scope_kind(entry.scope)
            ConfigKey(entry.key)
        except ValueError:
            unread.append({**entry._asdict(), "reason": "scope or key outside the vocabulary"})
        else:
            changes.append(entry)
    tasks: dict[str, set[str]] = {site.id: set() for site in fleet.sites}
    undescribed: set[str] = set()
    for row in ran:
        if str(row["site_id"]) in tasks:
            tasks[str(row["site_id"])].add(str(row["task_text"]))
        else:
            undescribed.add(str(row["site_id"]))
    sites = {site: suite_of(texts) for site, texts in tasks.items()}
    named = nominations(changes, {site: suite for site, (suite, _) in sites.items()})
    entries = export_catalogue(export)
    offered = {(a.verb, a.target): a for a in entries}
    chosen: list[tuple[Nomination, Action]] = []
    beyond: list[tuple[Nomination, Action]] = []
    skipped: list[Nomination] = []
    for nomination in named:
        action = offered.get((nomination.verb, nomination.target))
        if action is None:
            skipped.append(nomination)
        elif len(chosen) < MAX_ACTIONS:
            chosen.append((nomination, action))
        else:
            beyond.append((nomination, action))
    actions = [action for _, action in chosen]
    taken = {a.model_dump_json() for a in actions}
    # The stage-2 answer format wants ten ranked entries. The filler is labelled and
    # enters no fix set: the explicit fix sets below are the rule's answer.
    padding = [a for a in entries if a.model_dump_json() not in taken][
        : max(0, MIN_RANKED_ACTIONS - len(actions))
    ]
    changed_sites = {n.target.scope for n in named if n.step == SITE_STEP}
    return BaselineAnswer(
        METHOD,
        ranking(
            actions + padding,
            [f"{n.step}: {n.because}" for n, _ in chosen] + [PADDING for _ in padding],
        ),
        explicit_fix_sets={
            budget_label(b.kind, b.amount): cut(actions, b) for b in SCORING_BUDGETS
        },
        metadata={
            "rule_version": RULE_VERSION,
            "rule": WORDING,
            "baseline_of_the_experiment": False,
            "max_actions": MAX_ACTIONS,
            "named": [_record(n, a) for n, a in chosen],
            "beyond_the_cap": [_record(n, a) for n, a in beyond],
            "skipped": [
                {**_record(n), "reason": "the export's catalogue does not hold this action"}
                for n in skipped
            ],
            "unread_change_entries": unread,
            "sites_with_episodes_and_no_description": sorted(undescribed),
            "sites": {
                site: {
                    "suite": suite,
                    "tasks": why,
                    "change_at_site": site in changed_sites,
                }
                for site, (suite, why) in sorted(sites.items())
            },
            "reads": [
                "change_log: day, scope, key",
                "episodes: site_id, task_text",
                "catalogue",
                "description: sites and their recorded object batch",
            ],
            "values": "recorded-standard: a batch is replaced by the site's recorded one",
            "export": str(export),
            "input_sha256": {name: _sha256(export / name) for name in INPUTS},
            "explicit_fix_sets_authoritative": True,
            "format_padding_never_selected": True,
        },
    )


def write_rule(root: Path, answer: BaselineAnswer) -> None:
    """The rule's directory: `outbox/`, `fix_sets/` and `metadata.json`, and no `budgets/`.

    A baseline's directory also holds a ranked list per budget under
    `budgets/<label>/outbox/`, with no record beside it. The rule's lists are
    padded, and the scorer would read such a path as an ordinary ranking and cut
    the padding into a fix set. So the rule keeps none: every list it leaves
    lies under the directory whose `metadata.json` says that this is no
    baseline of the experiment, which the scorer refuses.
    """
    write_answer(root, answer)
    shutil.rmtree(root / "budgets")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valued(
    world_root: Path,
    fix_sets: Mapping[str, FixSet],
    source: Path,
    oracle_dir: Path | None,
    extra_runs: Sequence[Path],
) -> dict[str, Any]:
    """Fix sets by budget label, valued on a world as the scorer values a baseline's.

    The world, its oracle table and the replay check are loaded as
    `fleetlab.scoring.__main__.score_world` loads them, and the fix sets go
    through `fleetlab.scoring.decisions.score_decisions`, the function that
    values every baseline's fix sets at `SCORING_BUDGETS` and refuses a set that
    does not fit its budget.
    """
    # Imported here: reading an export must not load a module that holds a world's truth.
    from fleetlab.oracle.report import report_json
    from fleetlab.oracle.steps import Oracle
    from fleetlab.oracle.table import build_table
    from fleetlab.oracle.world import (
        OracleLayout,
        all_runs,
        load_world,
        recorded_sources,
        world_ready,
    )
    from fleetlab.scoring.decisions import score_decisions
    from fleetlab.scoring.models import ScoringInputs
    from fleetlab.scoring.verification import replay_problem

    world = load_world(world_root)
    if world.fault_table is None:
        raise ValueError("world lacks the hidden fault table")
    if problem := world_ready(world_root):
        raise ValueError(problem)
    layout = OracleLayout(oracle_dir or world_root / "oracle", world.world_id)
    source_runs = all_runs(
        world_root, layout, [*extra_runs, *recorded_sources(layout, world.world_hash)]
    )
    verifier = Oracle(world_root, world, layout).verifier()
    verifier.validate()
    built = build_table(world.identities, world.excluded, source_runs, world.world_hash, verifier)
    verification: dict[str, Any] = {}
    if layout.report.is_file():
        oracle_report = report_json(layout.report)
        if oracle_report.world_hash != world.world_hash:
            raise ValueError("oracle pattern report belongs to another world hash")
        verification = oracle_report.replay_verification
    inputs = ScoringInputs(
        fault_table=world.fault_table,
        oracle=built.table.restricted(world.scoring()),
        truth=world.truth(),
        current=world.current(),
        seed=world.spec.seed,
        baseline_fix_sets={EVALUATION_NAME: dict(fix_sets)},
        baseline_sources={EVALUATION_NAME: {"source": str(source)}},
        replay_missing=replay_problem(
            verification,
            len(world.scoring()),
            "python -m fleetlab.oracle verify, as the scorer's report of this world spells out",
        ),
    )
    decisions, needs, notes = score_decisions(inputs)
    waiting = f"claim 3: {EVALUATION_NAME} at "
    return {
        "world_id": world.world_id,
        "world_hash": world.world_hash,
        "world_root": str(world_root),
        "oracle_report": str(layout.report) if layout.report.is_file() else None,
        "scoring_episodes": len(inputs.oracle.identities),
        "budgets": [
            {
                "label": at.label,
                "budget": at.budget.model_dump(mode="json"),
                "rule": at.baselines[EVALUATION_NAME].model_dump(mode="json"),
                "achievable": at.achievable.model_dump(mode="json"),
            }
            for at in decisions.budgets
        ],
        "needs_fresh_replay": {
            label.removeprefix(waiting): len(items)
            for label, items in needs
            if label.startswith(waiting)
        },
        "notes": [*notes, *built.problems()],
    }


def evaluate(
    world_root: Path,
    rule_dir: Path,
    *,
    oracle_dir: Path | None = None,
    extra_runs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Value what this module's rule names from the world's own export, outside every claim.

    The rule is computed again from `WORLD/export`, and the directory is
    refused unless its four fix sets, its version and its wording equal that:
    a fix set cut by hand, another version of the rule, or the rule run on
    other bytes is not valued under the rule's name. What is valued is the
    recomputed answer. Whether the export holds a clean receipt of the current
    export check is recorded; the world's readiness, which asks for one, is
    required as the scorer requires it. Nothing of a score report is read or
    written.
    """
    from fleetlab.log.leaks import export_check_problem
    from fleetlab.scoring.decisions import budget_label as scorer_label

    world_root = world_root.expanduser().resolve()
    rule_dir = rule_dir.expanduser().resolve()
    metadata = json.loads((rule_dir / "metadata.json").read_text())
    if metadata.get("method") != METHOD:
        raise ValueError(f"{rule_dir} is not an output of the build rule")
    export = world_root / "export"
    validate_export(export)
    own = build_rule(export)
    named = {label: fix.model_dump(mode="json") for label, fix in own.fix_sets().items()}
    found = {
        p.stem: json.loads(p.read_text()) for p in sorted((rule_dir / "fix_sets").glob("*.json"))
    }
    if found != named:
        raise ValueError(
            f"{rule_dir}: its fix sets are not what the rule names from {export}; "
            "run the rule on this world's export and evaluate that"
        )
    if (metadata.get("rule_version"), metadata.get("rule")) != (RULE_VERSION, WORDING):
        raise ValueError(f"{rule_dir} was written by another version or wording of the rule")
    valued = _valued(
        world_root,
        {scorer_label(fix.budget): fix for fix in own.fix_sets().values()},
        rule_dir,
        oracle_dir,
        extra_runs,
    )
    return {
        "method": METHOD,
        "rule_version": RULE_VERSION,
        "rule": WORDING,
        "claims": "none: this file stands beside the score reports and enters no claim table",
        "valued_by": (
            "fleetlab.scoring.decisions.score_decisions, as it values a baseline's fix sets, "
            "at fleetlab.contracts.actions.SCORING_BUDGETS"
        ),
        "valued": (
            "the rule computed again from this world's export; the directory's four fix sets, "
            "version and wording equal it"
        ),
        "rule_dir": str(rule_dir),
        "export": str(export),
        "export_receipt_problem": export_check_problem(
            export, receipt_dir=world_root / "export-checks"
        ),
        "input_sha256": own.metadata["input_sha256"],
        "fix_set_sha256": {
            str(p.relative_to(rule_dir)): _sha256(p)
            for p in sorted((rule_dir / "fix_sets").glob("*.json"))
        },
        **valued,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--export", type=Path, help="a public export: log/ and data_dictionary.md")
    parser.add_argument(
        "--evaluate", type=Path, metavar="RULE_DIR", help="value the fix sets written earlier"
    )
    parser.add_argument("--world-root", type=Path, help="with --evaluate: the world to value on")
    parser.add_argument("--oracle-dir", type=Path, help="with --evaluate: default WORLD/oracle")
    parser.add_argument("--extra-run", action="append", type=Path, default=[])
    parser.add_argument(
        "--out", type=Path, required=True, help="a new directory, or with --evaluate a new file"
    )
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error(f"{out} exists; nothing is overwritten")
    if args.evaluate is not None:
        if args.export is not None or args.world_root is None:
            parser.error("--evaluate takes --world-root and no --export")
        try:
            result = evaluate(
                args.world_root,
                args.evaluate,
                oracle_dir=args.oracle_dir,
                extra_runs=args.extra_run,
            )
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        write_json(out, result)
        print(out)
        return 0
    if args.export is None or args.world_root is not None or args.oracle_dir or args.extra_run:
        parser.error("give --export, or --evaluate with --world-root")
    export = args.export.expanduser().resolve()
    if out.is_relative_to(export) or export.is_relative_to(out):
        parser.error("output must be separate from the export")
    try:
        validate_export(export)
        answer = build_rule(export)
    except ValueError as exc:
        parser.error(str(exc))
    write_rule(out, answer)
    print(out / "metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
