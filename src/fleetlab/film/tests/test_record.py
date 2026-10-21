"""The record page: derived values (Wilson intervals, rescued and broken, selections) and the
page's binding, shown as a draft and refused as final while numbers or gaps remain."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fleetlab.film import fields, record
from fleetlab.film.website import SITE_ROOT


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_wilson_matches_the_textbook_values() -> None:
    low, high = record.wilson(17, 20)
    assert (round(low, 3), round(high, 3)) == (0.640, 0.948)
    low, high = record.wilson(0, 20)
    assert low == 0.0 and round(high, 3) == 0.161
    low, high = record.wilson(20, 20)
    assert round(low, 3) == 0.839 and high == 1.0
    with pytest.raises(ValueError):
        record.wilson(3, 0)


def test_an_interval_needs_twenty_runs() -> None:
    assert record.proportion(17, 20)["interval"] == " (85%; 95% interval 64% to 95%)"
    small = record.proportion(7, 12)
    assert small["interval"] == "" and "low_pct" not in small and small["pct"] == "58%"


def test_paths_select_chain_and_flatten() -> None:
    data = {
        "message_log": [
            {"author": "lead", "text": "Make this one job", "session": None},
            {"author": "agent", "text": "Proposal: N=20, K=16", "session": 1},
            {"author": "lead", "text": "Not yet", "session": 1},
            {"author": "agent", "text": "Proposal: N=20, K=17", "session": 2},
            {"author": "lead", "text": "Approved: 17 of 20", "session": 2},
        ],
        "batches": [{"jobs": [{"d": "a"}, {"d": "b"}]}, {"jobs": [{"d": "a"}]}],
    }
    assert record.path_values(data, "message_log.author=lead.session=2.text") == [
        "Approved: 17 of 20"
    ]
    assert record.path_values(data, "batches.*.jobs.*.d") == ["a", "b", "a"]
    assert record.path_values(data, "batches.1.jobs.0.d") == ["a"]
    assert record.path_values(data, "nothing.here") == []


def _receipt(pairs: list[tuple[bool, bool]]) -> dict[str, object]:
    return {
        "N": len(pairs),
        "K": 16,
        "delivered": sum(p for p, _ in pairs),
        "naive_delivered": sum(n for _, n in pairs),
        "rescued": sum(p and not n for p, n in pairs),
        "broken": sum(n and not p for p, n in pairs),
        "passed": sum(p for p, _ in pairs) >= 16,
        "released": sum(p for p, _ in pairs) >= 16,
        "program_digest": "ab" * 32,
        "pairs": [{"delivered": p, "naive_delivered": n} for p, n in pairs],
    }


def test_derive_recounts_a_receipt_and_selects_from_a_record(tmp_path: Path) -> None:
    pairs = [(True, False)] * 9 + [(True, True)] * 8 + [(False, True)] + [(False, False)] * 2
    receipt = _write(tmp_path / "receipt.json", _receipt(pairs))
    run = _write(
        tmp_path / "job_run.json",
        {
            "message_log": [
                {"author": "lead", "text": "Make this one job"},
                {"author": "agent", "text": "Proposal: N=20, K=16"},
                {"author": "lead", "text": "Approved: 16 of 20"},
            ],
            "batches": [
                {"jobs": [{"program_digest": "a"}, {"program_digest": "b"}]},
                {"jobs": [{"program_digest": "a"}]},
            ],
            "request": {"goals": ["On a b", "On b c"]},
        },
    )
    spec = {
        "pairs": {"kind": "pairs", "record": "receipt"},
        "program": {
            "kind": "proportion",
            "k": {"record": "receipt", "path": "delivered"},
            "n": {"record": "receipt", "path": "N"},
        },
        "verdict": {"kind": "verdict", "record": "receipt"},
        "checksum": {"kind": "sha256", "record": "receipt", "chars": 12},
        "approval": {
            "kind": "text",
            "record": "run",
            "path": "message_log.author=lead.text",
            "startswith": "Approved",
        },
        "runs": {"kind": "count", "record": "run", "path": "batches.*.jobs.*"},
        "programs": {
            "kind": "distinct",
            "record": "run",
            "path": "batches.*.jobs.*.program_digest",
        },
        "goals": {"kind": "join", "record": "run", "path": "request.goals", "code": True},
        "skills": {"kind": "count", "record": "run", "path": "library.skills"},
        "story_b": {"kind": "evaluation", "record": "missing", "state": "original"},
    }
    derived = record.derive(spec, {"receipt": [receipt], "run": [run]})
    values = derived["values"]
    assert {k: values["pairs"][k] for k in ("rescued", "broken", "both", "neither")} == {
        "rescued": 9,
        "broken": 1,
        "both": 8,
        "neither": 2,
    }
    assert values["program"]["interval"] == " (85%; 95% interval 64% to 95%)"
    assert values["verdict"]["value"] == "met the shift lead's bar and was released"
    assert values["checksum"]["value"] == derived["files_read"][str(receipt)][:12]
    fresh = _write(tmp_path / "fresh.json", {**_receipt(pairs), "starts_fresh": True})
    verdict = record.derive({"v": {"kind": "verdict", "record": "f"}}, {"f": [fresh]})
    words = verdict["values"]["v"]["value"]
    assert words == "met the shift lead's bar on starts no program had run on and was released"
    assert values["approval"]["value"] == "Approved: 16 of 20"
    assert (values["runs"]["value"], values["programs"]["value"]) == (3, 2)
    assert values["goals"]["value"] == "`On a b`; `On b c`"
    assert values["pairs"]["inputs"][0]["sha256"] == derived["files_read"][str(receipt)]
    # a path the record does not have is unavailable, never a count of zero
    assert (
        "skills" not in values and "nothing at library.skills" in derived["unavailable"]["skills"]
    )
    assert derived["unavailable"]["story_b"] == "this cut provides no record missing"


def test_a_receipt_whose_pairs_disagree_is_not_derived(tmp_path: Path) -> None:
    bad = _receipt([(True, False)] * 20)
    bad["rescued"] = 19
    receipt = _write(tmp_path / "receipt.json", bad)
    derived = record.derive({"pairs": {"kind": "pairs", "record": "r"}}, {"r": [receipt]})
    assert "pairs" not in derived["values"] and "rescued 19" in derived["unavailable"]["pairs"]


def test_evaluation_states_and_pairs(tmp_path: Path) -> None:
    engagement = _write(
        tmp_path / "engagement.json",
        {
            "rounds": [
                {
                    "decision": {
                        "adopt": [
                            {
                                "action": {
                                    "verb": "set",
                                    "value": 10,
                                    "target": {
                                        "scope": "s1",
                                        "config_key": "control.latency_compensation_steps",
                                    },
                                }
                            }
                        ]
                    }
                }
            ],
            "evaluation": {
                "summary": {
                    "states": [
                        {"name": "original", "attempts": 576, "successes": 400},
                        {"name": "round-1", "attempts": 576, "successes": 470},
                    ],
                    "pairs": [
                        {"first": "original", "second": "round-1", "rescued": 80, "broken": 10}
                    ],
                }
            },
        },
    )
    spec = {
        "final": {"kind": "evaluation", "record": "e", "state": "round-1"},
        "pair": {
            "kind": "evaluation_pair",
            "record": "e",
            "first": "original",
            "second": "round-1",
        },
        "actions": {"kind": "actions", "record": "e", "path": "rounds.0.decision.adopt"},
    }
    values = record.derive(spec, {"e": [engagement]})["values"]
    assert (values["final"]["k"], values["final"]["n"]) == (470, 576)
    assert values["final"]["interval"].startswith(" (82%; 95% interval ")
    assert (values["pair"]["rescued"], values["pair"]["broken"]) == (80, 10)
    assert values["actions"]["value"] == "set control.latency_compensation_steps at s1 to 10"


PAGE = """---
title: "A record"
---

<!-- a note for the writer, never bound: {not_a_field} -->

# A record

The program delivered on {delivered} of {n}{interval}. The bar was {k}.

[gap: what the agent found, from its account]

{provenance_numbers}
"""


def _fields(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "fields.json",
        {
            "format": fields.FIELDS_FORMAT,
            "titles": {"receipt": "the release receipt"},
            "fields": {
                "delivered": {"record": "receipt", "field": "delivered", "meaning": "delivered"},
                "n": {"record": "receipt", "field": "N", "meaning": "release starts"},
                "k": {"record": "receipt", "field": "K", "meaning": "the bar"},
                "interval": {
                    "record": "derived",
                    "field": "program.interval",
                    "meaning": "share and interval",
                },
            },
            "derived": {
                "program": {
                    "kind": "proportion",
                    "k": {"record": "receipt", "path": "delivered"},
                    "n": {"record": "receipt", "path": "N"},
                    "how": "Wilson 95% interval",
                }
            },
        },
    )


def test_a_draft_page_shows_what_is_unbound_and_a_final_one_refuses(tmp_path: Path) -> None:
    page = tmp_path / "record.md"
    page.write_text(PAGE, encoding="utf-8")
    fields_path = _fields(tmp_path)
    records = _write(tmp_path / "records.json", {"format": fields.RECORDS_FORMAT, "records": {}})
    report = record.bind_record(page, fields_path, records, tmp_path / "draft")
    bound = (tmp_path / "draft" / "page" / "index.md").read_text()
    assert set(report["unbound"]) == {"delivered", "n", "k", "interval"}
    assert "**[unbound: delivered]**" in bound and "{delivered}" not in bound
    assert "<!--" not in bound and "not_a_field" not in bound
    assert bound.index("> **Draft.** 4 number(s)") > bound.index("# A record")
    assert bound.startswith('---\ntitle: "A record"\n---\n')
    assert report["gaps"] == ["[gap: what the agent found, from its account]"]
    code = record.main(
        [
            "bind",
            str(page),
            "--fields",
            str(fields_path),
            "--records",
            str(records),
            "--out",
            str(tmp_path / "final"),
            "--final",
        ]
    )
    assert code == 4


def test_a_bound_page_carries_numbers_and_their_sources(tmp_path: Path) -> None:
    page = tmp_path / "record.md"
    page.write_text(PAGE.replace("[gap: what the agent found, from its account]", ""))
    receipt = _write(
        tmp_path / "run" / "receipt.json", _receipt([(True, False)] * 17 + [(False, False)] * 3)
    )
    records = _write(
        tmp_path / "records.json",
        {"format": fields.RECORDS_FORMAT, "records": {"receipt": str(receipt)}},
    )
    fields_path = _fields(tmp_path)
    report = record.bind_record(page, fields_path, records, tmp_path / "out")
    bound = (tmp_path / "out" / "page" / "index.md").read_text()
    assert not report["unbound"] and not report["gaps"]
    assert "delivered on 17 of 20 (85%; 95% interval 64% to 95%). The bar was 16." in bound
    assert "Draft" not in bound
    assert re.search(r"\| interval \| +\(85%.*Wilson 95% interval from `", bound)
    code = record.main(
        [
            "bind",
            str(page),
            "--fields",
            str(fields_path),
            "--records",
            str(records),
            "--out",
            str(tmp_path / "final"),
            "--final",
        ]
    )
    assert code == 0


def test_an_undefined_placeholder_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(fields.FieldError, match="no field defines"):
        record.bind_page("# R\n\n{nobody}\n", {}, {}, {"values": {}})


def test_the_committed_page_and_fields_agree() -> None:
    page = (SITE_ROOT / "record" / "record.md").read_text(encoding="utf-8")
    data = json.loads((SITE_ROOT / "record" / "fields.json").read_text(encoding="utf-8"))
    specs = fields.load_fields(SITE_ROOT / "record" / "fields.json")
    used = set(fields.PLACEHOLDER.findall(re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)))
    assert used - set(record.GENERATED) == set(specs)
    for spec in specs.values():
        assert spec.record in data["records"] or spec.record == "derived"
        if spec.record == "derived":
            assert (spec.field or "").split(".")[0] in data["derived"]
    for name, item in data["derived"].items():
        assert item.get("how"), f"derived {name} says how it is derived"
    assert set(data["titles"]) == set(data["records"])
    # bound to no record at all, the page still binds, as a draft with every number marked
    binding = record.bind_page(page, specs, {}, {"values": {}, "unavailable": {}})
    assert not fields.PLACEHOLDER.search(binding.text)
    assert len(binding.unbound) == len(specs)
    assert "verdict" not in binding.text.lower()


def _parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_story_b_kinds_read_the_noted_site_quotes_and_the_panel(tmp_path: Path) -> None:
    description = _write(
        tmp_path / "description.json",
        {
            "fleet": {
                "sites": [
                    {"id": "s1", "robots": [{"id": "r01"}]},
                    {
                        "id": "s2",
                        "robots": [{"id": "r02"}],
                        "notes": ["Policy servers in the customer's data centre."],
                    },
                ]
            }
        },
    )
    skill = tmp_path / "SKILL.md"
    skill.write_text("Intro. A normal probe with failing tasks calls for a look. Then more.\n")
    ledger = _write(
        tmp_path / "ledger.json",
        {
            "entries": [
                {
                    "entry_id": "L1",
                    "action": {"verb": "set", "target": {"scope": "s2"}},
                    "rationale": "Probe p0007 shows it.",
                },
                {
                    "entry_id": "L2",
                    "action": {"verb": "service", "target": {"scope": "r02"}},
                    "rationale": "Costs 3.0 operator hours.",
                },
            ]
        },
    )
    rows = [
        {"state": s, "site_id": site, "robot_id": robot, "success": ok}
        for s, site, robot, ok in [
            ("original", "s1", "r01", True),
            ("original", "s2", "r02", False),
            ("original", "s2", "r02", True),
            ("final", "s1", "r01", True),
            ("final", "s2", "r02", True),
            ("final", "s2", "r02", True),
        ]
    ]
    attempts = _parquet(tmp_path / "attempts.parquet", rows)
    work = _parquet(
        tmp_path / "work.parquet",
        [
            {"robot_id": "r01", "site_id": "s1", "outcome": "failure"},
            {"robot_id": "r03", "site_id": "s1", "outcome": "success"},
            {"robot_id": "r02", "site_id": "s2", "outcome": "failure"},
        ],
    )
    linked = {"site_id": {"derived": "site"}}
    spec = {
        "site": {"kind": "noted_site", "record": "d", "note": "data centre"},
        "rule": {"kind": "quote", "record": "skill", "regex": r"A normal probe[^.]*\."},
        "probe": {
            "kind": "quote",
            "record": "ledger",
            "path": "entries.*.rationale",
            "regex": r"Probe (p[0-9]+)",
        },
        "hours": {
            "kind": "quote",
            "record": "ledger",
            "path": "entries.*.rationale",
            "regex": r"([0-9.]+) operator hours",
        },
        "serviced": {
            "kind": "text",
            "record": "ledger",
            "path": "entries.*.action.verb=service.target.scope",
        },
        "lowest": {
            "kind": "lowest",
            "record": "work",
            "where_not": linked,
            "group": "robot_id",
            "column": "outcome",
            "success": "success",
        },
        "table": {
            "kind": "breakdown",
            "record": "attempts",
            "states": ["original", "final"],
            "labels": ["Before", "After"],
            "column": "success",
            "success": True,
            "groups": [
                {"label": "site {site_id}", "where": linked},
                {"label": "robot {robot_id}", "where": {"robot_id": {"derived": "serviced"}}},
            ],
        },
    }
    records = {
        "d": [description],
        "skill": [skill],
        "ledger": [ledger],
        "attempts": [attempts],
        "work": [work],
    }
    derived = record.derive(spec, records)
    values = derived["values"]
    assert not derived["unavailable"]
    assert (values["site"]["value"], values["site"]["robots"]) == ("s2", 1)
    assert values["rule"]["value"] == "A normal probe with failing tasks calls for a look."
    assert (values["probe"]["value"], values["hours"]["value"]) == ("p0007", "3.0")
    assert values["serviced"]["value"] == "r02"
    assert (values["lowest"]["value"], values["lowest"]["k"], values["lowest"]["n"]) == (
        "r01",
        0,
        1,
    )
    assert values["table"]["value"].splitlines()[2:] == [
        "| site s2 | 1 of 2 | 2 of 2 |",
        "| robot r02 | 1 of 2 | 2 of 2 |",
    ]


def test_an_evaluation_whose_interval_is_not_wilsons_is_refused(tmp_path: Path) -> None:
    engagement = _write(
        tmp_path / "e.json",
        {
            "evaluation": {
                "summary": {
                    "states": [
                        {
                            "name": "original",
                            "attempts": 576,
                            "successes": 451,
                            "interval": [0.70, 0.80],
                        },
                    ]
                }
            }
        },
    )
    derived = record.derive(
        {"o": {"kind": "evaluation", "record": "e", "state": "original"}}, {"e": [engagement]}
    )
    assert "not Wilson's" in derived["unavailable"]["o"]
    low, high = record.wilson(451, 576)
    assert (round(low, 4), round(high, 4)) == (0.7475, 0.8147)


def test_the_probe_reading_follows_the_probe_route(tmp_path: Path) -> None:
    def steps(episode: str, delay: int) -> list[dict[str, object]]:
        out = []
        for k in range(60):
            target = 0.03 if 10 <= k < 40 else 0.0
            moved = 0.03 if 10 + delay <= k < 40 + delay else 0.0
            out.append(
                {
                    "episode_id": episode,
                    "step": k,
                    "target_pos": [target, 0.0, 1.0],
                    "eef_pos": [moved, 0.0, 1.0],
                }
            )
        return out

    episodes = _parquet(
        tmp_path / "probe_episodes.parquet",
        [
            {
                "probe_id": "p0001",
                "episode_id": f"e{i}",
                "robot_id": robot,
                "site_id": "s1",
                "repeat": rep,
            }
            for i, (robot, rep) in enumerate([("r01", 0), ("r01", 1), ("r02", 0), ("r02", 1)])
        ],
    )
    delays = [5, 5, 1, 2]
    files = [
        _parquet(tmp_path / "steps" / f"e{i}.parquet", steps(f"e{i}", d))
        for i, d in enumerate(delays)
    ]
    spec = {"p": {"kind": "probe", "episodes": "ep", "steps": "st", "probe": "p0001"}}
    values = record.derive(spec, {"ep": [episodes], "st": files})["values"]["p"]
    assert values["r01"]["first_move"] == "5 and 5"
    assert values["r02"]["first_move"] == "1 and 2 then 1 and 2"
    assert values["summary"].startswith("r01 (site s1): first moved after 5 and 5 control steps")


def test_a_page_that_would_show_a_forbidden_name_is_refused(tmp_path: Path) -> None:
    receipt = _write(tmp_path / "r.json", {"note": "the hidden world w80"})
    specs = {
        "note": fields.FieldSpec.parse(
            "note", {"record": "r", "field": "note", "meaning": "a note"}
        )
    }
    with pytest.raises(fields.FieldError, match="must never show"):
        record.bind_page(
            "# R\n\n{note}\n", specs, {"r": [receipt]}, {"values": {}}, forbidden=[r"\bw8[0-9]\b"]
        )


def test_shift_kinds_tabulate_programs_ranges_and_added_request_text(tmp_path: Path) -> None:
    run = _write(
        tmp_path / "job_run.json",
        {
            "request": {"request_text": "Make this one job. It must deliver."},
            "batches": [
                {
                    "jobs": [
                        {"job_id": "j0001", "program": "chain"},
                        {"job_id": "j0002", "program": "wait"},
                    ]
                }
            ],
        },
    )
    nxt = _write(
        tmp_path / "request.json",
        {"request_text": "Make this one job. It must deliver. A miss is fine."},
    )
    episodes = tmp_path / "episodes.jsonl"
    rows = [{"job_id": "j0001", "start": {"init_state": i}, "delivered": False} for i in range(3)]
    rows += [
        {"job_id": "j0002", "start": {"init_state": i}, "delivered": i % 4 != 0}
        for i in range(10, 30)
    ]
    episodes.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = {
        "table": {"kind": "jobs_table", "record": "run", "episodes": "ep"},
        "range": {
            "kind": "start_range",
            "record": "ep",
            "table": True,
            "where": {"job_id": "j0002"},
        },
        "added": {
            "kind": "added",
            "record": "next",
            "path": "request_text",
            "before": "run",
            "before_path": "request.request_text",
        },
    }
    values = record.derive(spec, {"run": [run], "next": [nxt], "ep": [episodes]})["values"]
    assert values["table"]["value"].splitlines()[2:] == [
        "| j0001 | chain | 0 to 2 | 0 of 3 |",
        "| j0002 | wait | 10 to 29 | 15 of 20 (75%; 95% interval 53% to 89%) |",
    ]
    assert values["range"]["value"] == "10 to 29"
    assert values["added"]["value"] == "A miss is fine."


def test_experiments_costs_and_safe_identifiers(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "experiment_id": "x0001",
                "request": {
                    "arms": [
                        {"actions": []},
                        {
                            "actions": [
                                {
                                    "verb": "service",
                                    "value": None,
                                    "target": {"scope": "r06", "component": "controller"},
                                }
                            ]
                        },
                    ]
                },
            }
        )
        + "\n"
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "experiment_id": "x0001",
                "status": "done",
                "arms": [
                    {"arm_index": 0, "episode_ids": ["e1", "e2", "e3"]},
                    {"arm_index": 1, "episode_ids": ["e4", "e5", "e6"]},
                ],
            }
        )
        + "\n"
        + json.dumps({"experiment_id": "x0002", "status": "refused", "arms": []})
        + "\n"
    )
    outcomes = {
        "e1": "success",
        "e2": "failure",
        "e3": "success",
        "e4": "success",
        "e5": "success",
        "e6": "failure",
    }
    episodes = _parquet(
        tmp_path / "episodes.parquet",
        [{"episode_id": e, "outcome": o} for e, o in outcomes.items()],
    )
    costs = _parquet(
        tmp_path / "costs.parquet",
        [
            {"verb": "service", "target": "controller", "hours": 3.0},
            {"verb": "set", "target": "config_key", "hours": 0.1},
        ],
    )
    decision = _write(
        tmp_path / "e.json",
        {
            "adopt": [
                {
                    "action": {
                        "verb": "set",
                        "value": 17,
                        "target": {"scope": "s1", "config_key": "k"},
                    }
                },
                {
                    "action": {
                        "verb": "service",
                        "value": None,
                        "target": {"scope": "r06", "component": "controller"},
                    }
                },
            ]
        },
    )
    words = {"service:controller": "a service of robot {scope}'s controller"}
    spec = {
        "table": {
            "kind": "experiments_table",
            "requests": "q",
            "results": "r",
            "episodes": "ep",
            "experiments": ["x0001"],
        },
        "hours": {"kind": "action_costs", "record": "d", "path": "adopt", "costs": "c"},
    }
    records = {"q": [requests], "r": [results], "ep": [episodes], "c": [costs], "d": [decision]}
    values = record.derive(spec, records, words)["values"]
    assert values["table"]["value"].splitlines()[2] == (
        "| x0001 | a service of robot r06's controller | 2 of 3 | 2 of 3 | 1 | 1 |"
    )
    assert (values["hours"]["value"], values["hours"]["total"]) == ("0.1 and 3.0", "3.1")
    hidden = Path.home() / ".cache" / "robots-probes" / "x" / "w80-2026" / "run.json"
    ref = record.safe_ref(hidden, [r"\bw8[0-9]\b"])
    assert "w80" not in ref and ref.startswith("probes:x/run#") and ref.endswith("/run.json")


def test_a_shortfall_and_release_starts_from_a_receipt(tmp_path: Path) -> None:
    receipt = _write(
        tmp_path / "receipt.json",
        {
            "K": 14,
            "delivered": 13,
            "release_starts": [{"init_state": i} for i in range(30, 50)],
        },
    )
    spec = {
        "short": {
            "kind": "difference",
            "a": {"record": "r", "path": "K"},
            "b": {"record": "r", "path": "delivered"},
        },
        "starts": {
            "kind": "start_range",
            "record": "r",
            "rows_path": "release_starts",
            "path": "init_state",
        },
    }
    values = record.derive(spec, {"r": [receipt]})["values"]
    assert values["short"]["value"] == 1
    assert (values["starts"]["value"], values["starts"]["count"]) == ("30 to 49", 20)


def test_the_highest_share_among_program_runs(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    rows = [
        {"job_id": j, "delivered": ok}
        for j, ok in [
            ("j0001", True),
            ("j0001", False),
            ("j0002", True),
            ("j0002", True),
            ("j0003", True),
        ]
    ]
    episodes.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = {
        "best": {
            "kind": "highest",
            "record": "ep",
            "table": True,
            "group": "job_id",
            "column": "delivered",
            "success": True,
        }
    }
    best = record.derive(spec, {"ep": [episodes]})["values"]["best"]
    assert (best["value"], best["k"], best["n"], best["interval"]) == ("j0002", 2, 2, "")
