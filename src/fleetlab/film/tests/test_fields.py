"""Narration numbers bound to record fields; a final cut and a voice refuse unbound ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleetlab.film import build, fields, manifest, narration, sources
from fleetlab.film.tests import fixtures
from fleetlab.film.website import SITE_ROOT

SCRIPT = """# Script

Not read: {not_a_field} in the preamble is never bound.

## open: Opening

The bar is {acceptance_k} of {acceptance_n}. Delivered: {dev_delivered} of {dev_starts}.

<!-- an editor's note may name {anything}. -->

## a5-release: The release

It delivered on {release_delivered}, in {job_one_model_minutes} minutes.
"""

FIELDS = {
    "format": fields.FIELDS_FORMAT,
    "fields": {
        "acceptance_k": {"record": "job_one", "field": "acceptance.K", "meaning": "K"},
        "acceptance_n": {"record": "job_one", "field": "acceptance.N", "meaning": "N"},
        "dev_delivered": {
            "record": "episodes",
            "count_true": "delivered",
            "where": {"job_id": "j0001"},
            "meaning": "delivered development starts",
        },
        "dev_starts": {
            "record": "episodes",
            "count_rows": True,
            "where": {"job_id": "j0001"},
            "meaning": "development starts",
        },
        "release_delivered": {"record": "release", "field": "delivered", "meaning": "delivered"},
        "job_one_model_minutes": {
            "record": "job_one",
            "field": "costs.model_seconds",
            "scale": 1 / 60,
            "format": "{:.0f}",
            "meaning": "minutes",
        },
    },
}


def _write(tmp_path: Path, records: dict[str, object]) -> tuple[Path, Path, Path]:
    script = tmp_path / "script.md"
    script.write_text(SCRIPT)
    spec = tmp_path / "fields.json"
    spec.write_text(json.dumps(FIELDS))
    (tmp_path / "job_run.json").write_text(
        json.dumps({"acceptance": {"N": 20, "K": 18}, "costs": {"model_seconds": 1500.0}})
    )
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"job_id": "j0001", "delivered": True},
                {"job_id": "j0001", "delivered": False},
                {"job_id": "j0002", "delivered": True},
            )
        )
        + "\n"
    )
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"format": fields.RECORDS_FORMAT, "records": records}))
    return script, spec, path


def test_bind_reads_fields_and_counts_and_leaves_the_rest(tmp_path: Path) -> None:
    script, spec, records = _write(
        tmp_path, {"job_one": "job_run.json", "episodes": ["episodes.jsonl"]}
    )
    bound, report = fields.bind_script(script, spec, records, tmp_path / "out")
    values = {k: v["value"] for k, v in report["values"].items()}
    assert values == {
        "acceptance_k": 18,
        "acceptance_n": 20,
        "dev_delivered": 1,
        "dev_starts": 2,
        "job_one_model_minutes": "25",
    }
    assert set(report["unbound"]) == {"release_delivered"}  # no release record in this cut
    assert report["values"]["acceptance_k"]["files"][0]["sha256"] == sources.sha256(
        tmp_path / "job_run.json"
    )
    beats = {b.id: b for b in narration.load_script(bound)}
    assert beats["open"].text == "The bar is 18 of 20. Delivered: 1 of 2."
    assert beats["a5-release"].unbound == ["release_delivered"]
    assert "{not_a_field}" in bound.read_text()  # the preamble is not bound
    assert report["bound_script_sha256"] == sources.sha256(bound)


def test_a_missing_field_stays_unbound_and_an_undefined_name_is_an_error(tmp_path: Path) -> None:
    script, spec, records = _write(tmp_path, {"job_one": "job_run.json", "release": "nope.json"})
    (tmp_path / "job_run.json").write_text(json.dumps({"acceptance": {"N": 20}}))
    _, report = fields.bind_script(script, spec, records, tmp_path / "out")
    assert "has no acceptance.K" in report["unbound"]["acceptance_k"]
    assert "does not exist" in report["unbound"]["release_delivered"]
    assert "provides no record episodes" in report["unbound"]["dev_starts"]
    script.write_text(SCRIPT + "\n## close: Close\n\nAnd {typo_name}.\n")
    with pytest.raises(fields.FieldError, match="typo_name"):
        fields.bind_script(script, spec, records, tmp_path / "out")


def test_the_repository_script_names_only_defined_fields() -> None:
    spec = fields.load_fields(SITE_ROOT / "narration" / "fields.json")
    text = (SITE_ROOT / "narration" / "script.md").read_text()
    binding = fields.bind(text, spec, {})
    assert binding.unbound and not binding.values  # every number waits for its record
    spoken = [n for b in narration.parse_script(text) for n in b.unbound]
    assert set(spoken) == set(binding.unbound) and set(spoken) <= set(spec)


def test_a_final_cut_narrates_a_bound_script_only(tmp_path: Path) -> None:
    fleet, chain = fixtures.fleet_run(tmp_path / "fleet"), fixtures.episode_run(tmp_path / "spike")
    script, spec, records = _write(tmp_path, {"job_one": "job_run.json"})
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain, script=script))
    with pytest.raises(build.FinalCutError, match="not a bound script"):
        build.build(m, tmp_path / "f0", draft=True, final=True, beats={"open"})
    bound, _ = fields.bind_script(script, spec, records, tmp_path / "bound")
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain, script=bound))
    with pytest.raises(build.FinalCutError, match="unbound"):
        build.build(m, tmp_path / "f1", draft=True, final=True, beats={"open"})
    # every record present: the cut passes and its provenance names the values
    (tmp_path / "release.json").write_text(json.dumps({"delivered": 19}))
    records.write_text(
        json.dumps(
            {
                "format": fields.RECORDS_FORMAT,
                "records": {
                    "job_one": "job_run.json",
                    "episodes": "episodes.jsonl",
                    "release": "release.json",
                },
            }
        )
    )
    bound, report = fields.bind_script(script, spec, records, tmp_path / "bound")
    assert not report["unbound"]
    result = build.build(m, tmp_path / "f2", draft=True, final=True, beats={"open"})
    assert result.provenance["bindings"]["values"]["acceptance_k"]["value"] == 18
    assert result.provenance["bindings"]["matches_script"] is True
    bound.write_text(bound.read_text().replace("18 of 20", "19 of 20"))  # typed by hand
    with pytest.raises(build.FinalCutError, match="changed after it was bound"):
        build.build(m, tmp_path / "f3", draft=True, final=True, beats={"open"})


def test_a_rerender_whose_original_hash_differs_is_refused(tmp_path: Path) -> None:
    run = fixtures.jobs_run(tmp_path / "jobs")
    video, report = fixtures.rerender(
        tmp_path / "rr", run, overrides={"original_video_sha256": "0" * 64}
    )
    ref = manifest.ClipRef(
        label="j", run=run, episode="e000101", rerender=manifest.Rerender(video, report)
    )
    refused = sources.resolve(ref)
    assert refused.rerender is not None and refused.rerender["used"] is False
    assert "original video's SHA-256" in refused.rerender["reason"]
    original = run / "log" / "video" / "e000101.scene.mp4"
    video, report = fixtures.rerender(
        tmp_path / "ok", run, overrides={"original_video_sha256": sources.sha256(original)}
    )
    ref = manifest.ClipRef(
        label="j", run=run, episode="e000101", rerender=manifest.Rerender(video, report)
    )
    assert sources.resolve(ref).rerender["used"] is True  # type: ignore[index]


def test_a_field_selects_a_list_entry_by_value(tmp_path: Path) -> None:
    record = tmp_path / "check.json"
    record.write_text(
        json.dumps({"episodes": [{"robot_id": "r05", "rows": 1}, {"robot_id": "r11", "rows": 18}]})
    )
    spec = {
        "far": fields.FieldSpec.parse(
            "far", {"record": "c", "meaning": "m", "field": "episodes.robot_id=r11.rows"}
        ),
        "none": fields.FieldSpec.parse(
            "none", {"record": "c", "meaning": "m", "field": "episodes.robot_id=r99.rows"}
        ),
    }
    binding = fields.bind("## b: B\n\nAt {far}, {none}.\n", spec, {"c": [record]})
    assert binding.values["far"]["value"] == 18 and "robot_id=r99" in binding.unbound["none"]


def test_a_probe_figure_is_drawn_from_its_tables_with_its_record(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fleetlab.film import probefig

    log = tmp_path / "log"
    (log / "probe_steps").mkdir(parents=True)
    episodes = []
    for i, (robot, delay) in enumerate((("r01", 18), ("r08", 1))):
        eid = f"e{i}"
        episodes.append({"probe_id": "p0001", "episode_id": eid, "robot_id": robot, "repeat": 0})
        rows = [
            {
                "step": s,
                "target_pos": [0.03 if s >= 20 else 0.0, 0.0, 1.0],
                "eef_pos": [0.03 if s >= 20 + delay else 0.0, 0.0, 1.0],
            }
            for s in range(80)
        ]
        pq.write_table(pa.Table.from_pylist(rows), log / "probe_steps" / f"{eid}.parquet")
    pq.write_table(pa.Table.from_pylist(episodes), log / "probe_episodes.parquet")
    out = tmp_path / "probe.png"
    record = probefig.draw(log, "p0001", [("delayed", "r01"), ("reference", "r08")], out)
    assert out.is_file() and record["figure_sha256"] == sources.sha256(out)
    assert [e["episode_id"] for e in record["episodes"]] == ["e0", "e1"]
    assert str(log / "probe_episodes.parquet") in record["files_read"]
    with pytest.raises(ValueError, match="did not run on r99"):
        probefig.draw(log, "p0001", [("x", "r99")], tmp_path / "no.png")


def test_a_list_of_names_is_counted_by_its_items(tmp_path: Path) -> None:
    record = tmp_path / "run.json"
    record.write_text(json.dumps({"library": {"checks": ["a", "b"], "skills": []}}))
    spec = {
        n: fields.FieldSpec.parse(
            n, {"record": "r", "meaning": "m", "field": f, "count_rows": True}
        )
        for n, f in (("checks", "library.checks"), ("skills", "library.skills"))
    }
    binding = fields.bind("## x: X\n\n{checks} and {skills}.\n", spec, {"r": [record]})
    assert binding.values["checks"]["value"] == 2 and binding.values["skills"]["value"] == 0


def test_cards_show_bound_numbers_and_a_final_cut_refuses_unbound_ones(tmp_path: Path) -> None:
    fleet, chain = fixtures.fleet_run(tmp_path / "fleet"), fixtures.episode_run(tmp_path / "spike")
    script, spec, records = _write(
        tmp_path, {"job_one": "job_run.json", "episodes": "episodes.jsonl"}
    )
    (tmp_path / "release.json").write_text(json.dumps({"delivered": 13}))
    data = json.loads(records.read_text())
    data["records"]["release"] = "release.json"
    records.write_text(json.dumps(data))
    bound, _ = fields.bind_script(script, spec, records, tmp_path / "bound")
    card: dict[str, object] = {"kind": "card", "title": "{release_delivered} of {acceptance_n}"}
    beats = [{"id": "open", "title": "Open", "shots": [card]}]
    m = manifest.load(
        fixtures.manifest(tmp_path / "m.json", fleet, chain, script=bound, beats=beats)
    )
    plans, _ = build.plan(m)
    assert plans[0].shots[0].shot.title == "13 of 20"
    card["title"] = "{release_delivered} of {nowhere}"
    m = manifest.load(
        fixtures.manifest(tmp_path / "m.json", fleet, chain, script=bound, beats=beats)
    )
    assert any("unbound nowhere" in p for p in build.final_cut_problems(build.plan(m)[0]))


def test_library_index_entries_read_as_messages() -> None:
    entries = [{"kind": "job", "name": "j", "purpose": "p"}, {"kind": "check", "name": "c"}]
    assert build._messages(entries, None, "w") == [("job", "j: p"), ("check", "c")]


def test_distinct_values_are_counted_across_tables(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps({"init_state": i}) for i in (0, 1, 2)) + "\n")
    b.write_text("\n".join(json.dumps({"init_state": i}) for i in (1, 2, 3)) + "\n")
    spec = {
        "d": fields.FieldSpec.parse(
            "d", {"record": "e", "meaning": "m", "count_distinct": "init_state"}
        )
    }
    assert fields.bind("## x: X\n\n{d}.\n", spec, {"e": [a, b]}).values["d"]["value"] == 4


def test_messages_can_show_one_key_of_each_entry() -> None:
    entries = [{"kind": "check", "name": "c", "purpose": "a long purpose"}]
    assert build._messages(entries, None, "w", "name") == [("check", "c")]
