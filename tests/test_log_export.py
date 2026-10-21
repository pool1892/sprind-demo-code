"""The export of the agent-visible tree, the data dictionary and the leak audit.

src/fleetlab/log/export.py and leaks.py; docs/design/contracts.md, sections 11 and 12.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from log_toy import (
    CAMERA_FAULT,
    CONFIG_FAULT,
    TASK_CLASSES,
    TRAIT,
    all_jobs,
    replay_jobs,
    run_jobs,
    toy_fleet,
    write_toy_run,
)

from fleetlab.contracts.actions import ADDED_CHANGE_KINDS
from fleetlab.contracts.log import (
    MEASURED_TIME_NAME,
    SCHEMA,
    Visibility,
    allowed_columns,
    visible_schema,
)
from fleetlab.log import ExportRefused, LogReader, audit_export, export_runs, leak_search
from fleetlab.log.dictionary import DICTIONARY_FILE
from fleetlab.log.export import export_problems, mp4_times
from fleetlab.log.leaks import code_terms, hidden_column_terms, hidden_field_terms, truth_terms


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A finalized toy run and its export."""
    base = tmp_path_factory.mktemp("export")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    return base / "run", base / "export"


def _copy_run(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run"
    shutil.copytree(run, target)
    return target


def _hidden_names() -> set[str]:
    visible = {c.name for t in visible_schema().values() for c in t.columns}
    return {
        c.name
        for t in SCHEMA.values()
        for c in t.columns
        if c.visibility is Visibility.hidden and c.name not in visible
    }


def test_the_export_holds_the_visible_tree_only(toy: tuple[Path, Path]) -> None:
    _, export = toy
    assert export_problems(export) == []
    assert sorted(p.name for p in export.iterdir()) == [DICTIONARY_FILE, "log"]
    assert not any(p.name == "truth" for p in export.rglob("*"))
    hidden = _hidden_names()
    for path in export.rglob("*.parquet"):
        names = pq.read_schema(path).names
        table = path.parent.name if path.parent.name == "steps" else path.stem
        assert names == list(allowed_columns(table)), path
        assert not hidden & set(names), path
        assert pq.read_schema(path).metadata is None, path
    episodes = LogReader(export).episode_ids()
    assert episodes == sorted(j.identity.episode_id for j in run_jobs())
    assert len(list((export / "log/steps").iterdir())) == len(episodes)
    assert len(list((export / "log/video").iterdir())) == 2 * sum(
        j.record_video for j in run_jobs()
    )


def test_no_measured_time_in_the_export(toy: tuple[Path, Path]) -> None:
    _, export = toy
    for table in visible_schema().values():
        for column in table.columns:
            assert column.clock != "measured"
    for path in export.rglob("*.parquet"):
        for name in pq.read_schema(path).names:
            assert not MEASURED_TIME_NAME.search(name), (path, name)
    for path in (export / "log/video").iterdir():
        times = mp4_times(path)
        assert times and not any(times), path


def test_a_measured_time_in_a_run_is_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    steps = run / "log/steps/e000002.parquet"
    data = pq.read_table(steps)
    pq.write_table(data.append_column("inference_ms", pa.array([41.5] * data.num_rows)), steps)
    with pytest.raises(ExportRefused, match="inference_ms"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())
    assert not (tmp_path / "export").exists()


def test_a_video_with_a_creation_time_is_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    video = run / "log/video/e000001.scene.mp4"
    data = bytearray(video.read_bytes())
    at = data.index(b"mvhd")
    data[at + 8 : at + 12] = struct.pack(">I", 3_800_000_000)  # version 0: creation time
    video.write_bytes(bytes(data))
    with pytest.raises(ExportRefused, match="creation or modification time"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())


def test_a_hidden_column_in_a_run_is_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    episodes = run / "log/episodes.parquet"
    data = pq.read_table(episodes)
    pq.write_table(data.append_column("scene_seed", pa.array([1] * data.num_rows)), episodes)
    with pytest.raises(ExportRefused, match="'scene_seed' is not on the visible allow-list"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())


def test_non_neutral_episode_ids_are_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    # A dose run names its episodes after the fault level.
    (run / "log/steps/e000002.parquet").rename(run / "log/steps/d2-degrees6.parquet")
    episodes = pq.read_table(run / "log/episodes.parquet")
    ids = ["d2-degrees6" if i == "e000002" else i for i in episodes["episode_id"].to_pylist()]
    episodes = episodes.set_column(0, "episode_id", pa.array(ids))
    pq.write_table(episodes, run / "log/episodes.parquet")
    with pytest.raises(ExportRefused) as refused:
        export_runs([run], tmp_path / "export", fleet=toy_fleet())
    text = str(refused.value)
    assert "d2-degrees6.parquet: episode id in the name is not neutral" in text
    assert "episode_id values not neutral" in text


def test_an_unfinished_run_is_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_toy_run(run, all_jobs(), finalize=False)
    with pytest.raises(ExportRefused, match="not finalized"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())


def test_two_runs_export_like_one(toy: tuple[Path, Path], tmp_path: Path) -> None:
    jobs = all_jobs()
    first = [j for j in jobs if j.identity.day < 2]
    second = [j for j in jobs if j.identity.day >= 2]
    fleet_a = write_toy_run(tmp_path / "a", first)
    write_toy_run(tmp_path / "b", second)
    export_runs(
        [tmp_path / "a", tmp_path / "b"],
        tmp_path / "export",
        fleet=fleet_a,
        task_classes=TASK_CLASSES,
    )
    single = toy[1]
    files = sorted(p.relative_to(single) for p in single.rglob("*") if p.is_file())
    assert files == sorted(
        p.relative_to(tmp_path / "export") for p in (tmp_path / "export").rglob("*") if p.is_file()
    )
    for relative in files:
        assert (single / relative).read_bytes() == (tmp_path / "export" / relative).read_bytes()
    with pytest.raises(ExportRefused, match=r"is in .* and"):
        export_runs([tmp_path / "a", tmp_path / "a"], tmp_path / "again", fleet=fleet_a)


def test_without_the_fleet_the_runs_must_agree(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    export_runs([run], tmp_path / "export")  # the run's own description and logs
    assert json.loads((tmp_path / "export/log/description.json").read_text()) == json.loads(
        (toy[1] / "log/description.json").read_text()
    )
    # A description with a hidden field is refused.
    description = json.loads((run / "log/description.json").read_text())
    description["fleet"]["sites"][0]["robots"][0]["traits"] = {TRAIT: 0.1}
    (run / "log/description.json").write_text(json.dumps(description))
    with pytest.raises(ExportRefused, match="hidden field 'traits'"):
        export_runs([run], tmp_path / "export2")


def test_an_export_is_replaced_only_on_request(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run, _ = toy
    out = tmp_path / "export"
    export_runs([run], out, fleet=toy_fleet())
    before = (out / "log/episodes.parquet").read_bytes()
    with pytest.raises(ExportRefused, match="exists and is not empty"):
        export_runs([run], out, fleet=toy_fleet())
    export_runs([run], out, fleet=toy_fleet(), overwrite=True)
    assert (out / "log/episodes.parquet").read_bytes() == before
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep")
    with pytest.raises(ExportRefused):
        export_runs([run], other, fleet=toy_fleet(), overwrite=True)
    assert (other / "notes.txt").read_text() == "keep"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_failed_replacement_keeps_the_earlier_export(
    toy: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = toy
    out = tmp_path / "export"
    export_runs([run], out, fleet=toy_fleet())
    before = sorted(p.relative_to(out) for p in out.rglob("*"))
    real_replace = os.replace

    def failing(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if ".export-" in Path(src).name:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(OSError, match="disk full"):
        export_runs([run], out, fleet=toy_fleet(), overwrite=True)
    monkeypatch.undo()
    assert sorted(p.relative_to(out) for p in out.rglob("*")) == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_the_dictionary_lists_every_visible_column(toy: tuple[Path, Path]) -> None:
    text = (toy[1] / DICTIONARY_FILE).read_text()
    for table in visible_schema().values():
        if table.source == "export":  # an export without probes does not list their tables
            assert f"## {table.name}" not in text
            continue
        assert f"## {table.name}" in text
        for column in table.columns:
            assert f"| {column.name} |" in text, (table.name, column.name)
    for name in hidden_column_terms():
        assert name not in text
    assert "## Conventions" in text
    run = toy[0]
    report = audit_export(toy[1], [run], fleet=toy_fleet(), texts={"d": text})
    assert report.clean and report.searched["texts"] == 1 and len(report.terms) > 40


def test_the_audit_of_the_toy_export_is_clean(toy: tuple[Path, Path]) -> None:
    run, export = toy
    report = audit_export(export, [run], fleet=toy_fleet())
    assert report.clean, report.summary()
    terms = set(report.terms)
    assert {CAMERA_FAULT, CONFIG_FAULT, TRAIT, "fault", "scene_seed"} <= terms
    assert any(t.startswith("canary-") for t in terms)
    assert report.searched["files"] > 20 and report.searched["text values"] > 100


def test_the_leak_search_finds_planted_terms(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run, export = toy
    planted = tmp_path / "export"
    shutil.copytree(export, planted)
    token = (run / "truth/canary.txt").read_text().strip()
    events = pq.read_table(planted / "log/events.parquet")
    details = events["detail"].to_pylist()
    details[0] = f"gripper slipped: {CAMERA_FAULT}"
    pq.write_table(
        events.set_column(3, "detail", pa.array(details)), planted / "log/events.parquet"
    )
    (planted / "log/video/e000003.scene.mp4").write_bytes(
        (planted / "log/video/e000003.scene.mp4").read_bytes() + token.encode()
    )
    report = audit_export(planted, [run], fleet=toy_fleet(), texts={"prompt": "the oracle says"})
    found = {(h.where.split(" ")[0], h.term) for h in report.hits}
    assert ("value", CAMERA_FAULT) in found
    assert ("bytes", token.lower()) in found
    assert ("prompt", "oracle") in found


def test_vocabulary_names_hide_no_kind(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("the scene_camera was set; control.action_delay_steps = 3")
    assert leak_search(tmp_path, ["delay", "scene_camera_shift"]).clean
    (tmp_path / "b.txt").write_text("scene_camera_shift at level 2")
    assert [h.term for h in leak_search(tmp_path, ["scene_camera_shift"]).hits] == [
        "scene_camera_shift"
    ]
    (tmp_path / "c.txt").write_text("default settings")
    assert not [h for h in leak_search(tmp_path, ["fault"]).hits if h.where == "c.txt"]


def test_a_symbolic_link_is_a_hit(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "target.txt")
    assert [h.term for h in leak_search(tmp_path, []).hits] == ["(symbolic link)"]


def test_code_is_searched_for_what_is_specific_to_a_world(toy: tuple[Path, Path]) -> None:
    run, export = toy
    narrow = code_terms([run], fleet=toy_fleet())
    assert CAMERA_FAULT in narrow and "fault" not in narrow and "scene_seed" not in narrow
    assert "traits" not in narrow
    assert not set(ADDED_CHANGE_KINDS) & set(narrow)
    source = "SCHEMA = {'faults_active': ..., 'scene_seed': ...}  # " + " ".join(ADDED_CHANGE_KINDS)
    clean = audit_export(export, [run], code={"contracts": source})
    assert clean.clean, clean.summary()
    leaking = audit_export(export, [run], code={"tool server": f"KINDS = ['{CONFIG_FAULT}']"})
    assert [h.term for h in leaking.hits] == [CONFIG_FAULT]
    data = audit_export(export, [run], texts={"prompt": "look up the robot traits"})
    assert [(h.where, h.term) for h in data.hits] == [("prompt", "traits")]


def test_trait_names_come_from_the_producer_without_a_full_fleet(
    toy: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.worlds import base

    original = base.traits_for
    monkeypatch.setattr(
        base,
        "traits_for",
        lambda seed, robot: {
            **original(seed, robot),
            "new_hidden_property": 2,
        },
    )
    stripped = toy_fleet().model_copy(deep=True)
    for robot in stripped.robots():
        object.__setattr__(robot, "traits", {})
    terms = {*original(0, "r01"), "new_hidden_property"}
    (tmp_path / "events.txt").write_text("\n".join(f"{t}=80" for t in terms))
    for fleet in (None, stripped):
        report = audit_export(tmp_path, [toy[0]], fleet=fleet)
        assert terms <= {hit.term for hit in report.hits}
        code = audit_export(
            toy[1],
            [toy[0]],
            fleet=fleet,
            code={"contracts": "traits: dict[str, float]", "leak": "new_hidden_property=2"},
        )
        assert [(h.where, h.term) for h in code.hits] == [("code leak", "new_hidden_property")]


def test_new_truth_schema_fields_are_covered_automatically(
    toy: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fleetlab.worlds import truth

    class FutureTable(truth.FaultTable):
        next_hidden_label: str = ""

    monkeypatch.setattr(truth, "FaultTable", FutureTable)
    expected = {
        "next_hidden_label",
        "fault_levels",
        "expected_loss_points",
        "qualified_by",
        "traits",
        "scoring_floor",
        "min_clean_robots",
        "counts_for_gate",
        "expected_effect_points",
        "writes_change",
    }
    assert expected <= set(hidden_field_terms())
    (tmp_path / "event.txt").write_text("next_hidden_label=answer; fault_levels=3")
    report = audit_export(tmp_path, [toy[0]])
    assert {"next_hidden_label", "fault_levels"} <= {hit.term for hit in report.hits}


def test_common_hidden_field_names_are_labels_not_ordinary_prose(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("the small bowl was placed in the first row")
    assert audit_export(tmp_path, [toy[0]]).clean
    (tmp_path / "log").mkdir()
    (tmp_path / "log/description.json").write_text('{"small": true, "row": 2}')
    pq.write_table(pa.table({"onset_day": [1]}), tmp_path / "data.parquet")
    (tmp_path / "labels.txt").write_text('nominal=3; "qualified_by": ["observed"]')
    report = audit_export(tmp_path, [toy[0]])
    assert {"small", "row", "onset_day", "nominal", "qualified_by"} <= {h.term for h in report.hits}


def test_arrow_field_metadata_and_nested_text_cannot_hide_in_encoded_schema(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    child = pa.field("note", pa.string(), metadata={b"gripper_pad_age_days": b"80"})
    dtype = pa.struct([child])
    schema = pa.schema([pa.field("detail", dtype)])
    data = pa.Table.from_pylist(
        [{"detail": {"note": "controller_board_revision=3"}}], schema=schema
    )
    path = tmp_path / "nested.parquet"
    pq.write_table(data, path, compression="zstd")
    assert pq.read_schema(path).metadata is None  # the hidden key lives in encoded ARROW:schema
    report = audit_export(tmp_path, [toy[0]])
    assert any(
        h.term == "gripper_pad_age_days" and "field metadata" in h.where for h in report.hits
    )
    assert any(h.term == "controller_board_revision" and "value of" in h.where for h in report.hits)


def test_generator_metadata_names_are_read_from_the_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fleetlab.worlds import world

    def future_record(world: object) -> dict[str, str]:
        return {"new_generator_truth": "hidden"}

    monkeypatch.setattr(world, "generator_record", future_record)
    assert "new_generator_truth" in truth_terms()


def test_emphasis_and_new_truth_table_names_are_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.worlds import base

    def future_emphasis(self: object) -> dict[str, int]:
        return {"future_emphasis_secret": 1}

    monkeypatch.setattr(base.Emphasis, "as_json", future_emphasis)
    monkeypatch.setitem(
        SCHEMA,
        "future_hidden_table",
        replace(
            SCHEMA["verdicts"], name="future_hidden_table", path="truth/future_hidden_table.parquet"
        ),
    )
    assert {"future_emphasis_secret", "future_hidden_table"} <= set(truth_terms())


def test_audit_fleet_contributes_only_hidden_names(toy: tuple[Path, Path], tmp_path: Path) -> None:
    full = toy_fleet().model_copy(deep=True)
    full.robots()[0].traits["unusual_hidden_property"] = 1
    stripped = toy_fleet().model_copy(deep=True)
    for robot in stripped.robots():
        robot.traits.clear()
    out = tmp_path / "clean"
    export_runs([toy[0]], out, fleet=stripped, audit_fleet=full, task_classes=TASK_CLASSES)
    assert "traits" not in (out / "log/description.json").read_text()
    run = _copy_run(toy[0], tmp_path)
    data = pq.read_table(run / "log/events.parquet")
    details = data["detail"].to_pylist()
    details[0] = "unusual_hidden_property=1"
    pq.write_table(data.set_column(3, "detail", pa.array(details)), run / "log/events.parquet")
    with pytest.raises(ExportRefused, match="unusual_hidden_property"):
        export_runs([run], tmp_path / "leak", fleet=stripped, audit_fleet=full)


def test_terms_cover_the_hidden_columns_and_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import fleetlab.faults

    assert "executed_action" in hidden_column_terms()
    assert "episode_id" not in hidden_column_terms()  # visible as well
    monkeypatch.setattr(fleetlab.faults, "load_all", lambda: ("unseen_wobble",))
    assert "unseen_wobble" in truth_terms()


def test_an_audit_without_a_canary_refuses(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    (run / "truth/canary.txt").unlink()
    with pytest.raises(ValueError, match="no canary planted"):
        audit_export(toy[1], [run])


def test_the_export_runs_the_audit(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    events = pq.read_table(run / "log/events.parquet")
    details = events["detail"].to_pylist()
    details[1] = "replay of the drop"
    pq.write_table(events.set_column(3, "detail", pa.array(details)), run / "log/events.parquet")
    with pytest.raises(ExportRefused, match=r"leak audit: value of log/events\.parquet\.detail"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())
    assert not (tmp_path / "export").exists()


def test_importing_the_log_package_does_not_import_the_fault_registry() -> None:
    code = (
        "import sys, fleetlab.log, fleetlab.log.reader; "
        "assert 'fleetlab.faults' not in sys.modules, 'fleetlab.faults imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_replays_never_reach_the_export(toy: tuple[Path, Path]) -> None:
    _, export = toy
    names = {p.name for p in export.rglob("*")}
    for job in replay_jobs():
        assert not any(job.key in name for name in names)


def test_a_video_without_logged_frames_is_refused(toy: tuple[Path, Path], tmp_path: Path) -> None:
    run = _copy_run(toy[0], tmp_path)
    steps = run / "log/steps/e000001.parquet"
    data = pq.read_table(steps)
    index = data.schema.get_field_index("frame_index")
    data = data.set_column(index, "frame_index", pa.nulls(data.num_rows, pa.int32()))
    pq.write_table(data, steps)
    with pytest.raises(ExportRefused, match="a video, no frame_index logged"):
        export_runs([run], tmp_path / "export", fleet=toy_fleet())


@pytest.mark.parametrize(
    "spelling",
    [
        "gripperPadAgeDays",
        "GRIPPER_PAD_AGE_DAYS",
        "gripper-pad-age-days",
        "gripper pad age days",
        "gripper.pad.age.days",
    ],
)
def test_property_spellings_are_found(
    toy: tuple[Path, Path], tmp_path: Path, spelling: str
) -> None:
    (tmp_path / "note.txt").write_text(spelling + " 80")
    assert "gripper_pad_age_days" in {h.term for h in audit_export(tmp_path, [toy[0]]).hits}


@pytest.mark.parametrize("spelling", ["trueReason", "TRUE_REASON", "true-reason", "true reason"])
def test_hidden_column_spellings_are_found(
    toy: tuple[Path, Path], tmp_path: Path, spelling: str
) -> None:
    (tmp_path / "note.txt").write_text(spelling + ": placed")
    assert "true_reason" in {h.term for h in audit_export(tmp_path, [toy[0]]).hits}


@pytest.mark.parametrize(
    "spelling",
    [
        "IlluminationChange",
        "ILLUMINATION_CHANGE",
        "illumination-change",
        "illumination.change",
        "illumination\u2010change",
    ],
)
def test_fault_kind_identifier_spellings_are_found(
    toy: tuple[Path, Path], tmp_path: Path, spelling: str
) -> None:
    (tmp_path / "note.txt").write_text(spelling)
    assert "illumination_change" in {h.term for h in audit_export(tmp_path, [toy[0]]).hits}


def test_fault_kinds_as_separate_prose_words_are_allowed(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    (tmp_path / "note.txt").write_text("illumination change at s1; software release 4.3.0")
    assert audit_export(tmp_path, [toy[0]]).clean


def test_seeds_and_fault_ids_are_values_not_just_field_names(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    ids = pq.read_table(toy[0] / "truth/identities.parquet")
    scene = ids["scene_seed"][0].as_py()
    noise = ids["noise_seed"][0].as_py()
    (tmp_path / "note.txt").write_text(f"rng {noise}; f01 active; xf01z is another word")
    pq.write_table(pa.table({"count": pa.array([scene], pa.int64())}), tmp_path / "numbers.parquet")
    report = audit_export(tmp_path, [toy[0]])
    assert {str(scene), str(noise), "f01"} <= {h.term for h in report.hits}
    assert sum(h.term == "f01" for h in report.hits) == 1
    assert any(h.term == str(scene) and "numbers.parquet.count" in h.where for h in report.hits)


def test_world_truth_files_supply_unknown_keys_and_columns(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    root = tmp_path / "world"
    run = root / "runs/w01-observed"
    shutil.copytree(toy[0], run)
    files = root / "files"
    files.mkdir()
    (files / "spec.json").write_text('{"new_composer_secret": {"interaction": 4}}')
    (files / "generator.json").write_text(
        '{"expected": {"healthy_robots": 2, "detectable_units": 1}}'
    )
    (root / "runner.json").write_text('{"summary": {"observed_carriers": 3}}')
    (files / "calibration.csv").write_text("new_curve_key\n2\n")
    pq.write_table(pa.table({"new_oracle_column": [2]}), run / "truth/new_oracle.parquet")
    terms = set(truth_terms([run]))
    assert {
        "new_composer_secret",
        "interaction",
        "healthy_robots",
        "detectable_units",
        "new_oracle_column",
        "new_curve_key",
    } <= terms
    assert "observed_carriers" in terms  # derived from the truth-summary producer
    assert "episode_id" not in terms
    out = tmp_path / "planted"
    out.mkdir()
    (out / "note.json").write_text('{"new_composer_secret": 2}')
    assert "new_composer_secret" in {h.term for h in audit_export(out, [run]).hits}


def test_oracle_and_sibling_runs_supply_truth_names_and_values(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    root = tmp_path / "world"
    observed = root / "runs/w01-observed"
    shutil.copytree(toy[0], observed)
    (root / "files").mkdir()
    (root / "files/spec.json").write_text("{}")
    oracle = root / "oracle"
    oracle.mkdir()
    (oracle / "oracle.json").write_text(
        '{"as_run_success": 0.5, "all_removed_success": 0.8, "achievable_net": 0.3, "runs": []}'
    )
    oracle_run = oracle / "runs/replay"
    shutil.copytree(toy[0], oracle_run)
    (oracle_run / "truth/oracle_writer.json").write_text('{"oracle_run_only_secret": 3}')
    (oracle_run / "log/description.json").write_text('{"visible_log_marker_not_truth": 4}')
    later = root / "runs/w01-later/truth"
    later.mkdir(parents=True)
    (later / "new_writer.json").write_text('{"sibling_only_secret": 2}')
    future_seed = 777000888111222333
    pq.write_table(pa.table({"new_seed": [future_seed]}), later / "identities.parquet")
    terms = set(truth_terms([observed]))
    assert {
        "as_run_success",
        "all_removed_success",
        "achievable_net",
        "sibling_only_secret",
        "oracle_run_only_secret",
        "new_seed",
        str(future_seed),
    } <= terms
    assert "visible_log_marker_not_truth" not in terms
    from fleetlab.log.leaks import _truth_sources

    sources = _truth_sources([observed])
    assert sources.count(oracle_run / "truth/oracle_writer.json") == 1
    out = tmp_path / "export"
    out.mkdir()
    (out / "note.txt").write_text("control runs at 20 steps per second")
    assert audit_export(out, [observed]).clean
    (out / "note.json").write_text('{"runs": []}')
    assert "runs" in {hit.term for hit in audit_export(out, [observed]).hits}
    (out / "note.txt").write_text(f"sibling_only_secret=2; rng {future_seed}")
    assert {"sibling_only_secret", str(future_seed)} <= {
        hit.term for hit in audit_export(out, [observed]).hits
    }


def test_direct_search_keeps_fault_kinds_identifier_only(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("gripper wear; illumination change")
    assert leak_search(tmp_path, ["gripper_wear", "illumination_change"]).clean
    (tmp_path / "note.txt").write_text("GripperWear and IlluminationChange")
    assert {"gripper_wear", "illumination_change"} <= {
        h.term for h in leak_search(tmp_path, ["gripper_wear", "illumination_change"]).hits
    }


def test_non_json_bracketed_text_is_decodable(toy: tuple[Path, Path], tmp_path: Path) -> None:
    pq.write_table(pa.table({"detail": ["[s1] remounted"]}), tmp_path / "note.parquet")
    assert audit_export(tmp_path, [toy[0]]).clean


def test_opaque_binary_separators_do_not_form_hidden_names(tmp_path: Path) -> None:
    # Reproduce the compressed-video matches without relying on random chance.
    payload = b"\0\0\0\x18ftypisom\xfft\xc7end\xffpss\xf3/\x13mib\xff"
    (tmp_path / "video.mp4").write_bytes(payload)
    assert leak_search(tmp_path, ["t_end", "pss_mib"]).clean
    (tmp_path / "video.mp4").write_bytes(payload + b"gripper\xffpad_age_days")
    assert leak_search(tmp_path, ["gripper_pad_age_days"]).clean
    (tmp_path / "video.mp4").write_bytes(payload + b"illumination change")
    assert leak_search(tmp_path, ["illumination_change"]).clean
    (tmp_path / "video.mp4").write_bytes(payload + b"IlluminationChange")
    assert any(
        h.term == "illumination_change" and h.where.startswith("bytes of")
        for h in leak_search(tmp_path, ["illumination_change"]).hits
    )
    (tmp_path / "note.txt").write_text("TEnd: 2; PssMib=3")
    assert {"t_end", "pss_mib"} <= {
        h.term for h in leak_search(tmp_path, ["t_end", "pss_mib"]).hits
    }
    (tmp_path / "video.mp4").write_bytes(payload + b"gripper-pad-age-days=80")
    assert any(
        h.term == "gripper_pad_age_days" and h.where.startswith("bytes of")
        for h in leak_search(tmp_path, ["gripper_pad_age_days"]).hits
    )


def test_raw_name_search_has_a_small_uniform_binary_chance_bound() -> None:
    from fleetlab.log.leaks import GENERIC_TERMS, _Search

    search = _Search(truth_terms(), surface="ours")
    # A conservative union bound for 400 MiB of independent uniform bytes.
    # Case folding gives at most two matching byte values. At most 38 ASCII
    # whitespace/punctuation bytes may be skipped between literal characters.
    # Actual compressed-video distributions are checked by the stored-world scans.
    bound = 0.0
    for term in search.raw:
        if term in GENERIC_TERMS:
            continue  # unchanged literal generic-word policy, measured separately
        literals = sum(c.isalnum() for c in term)
        bound += (2 / (256 - 38)) ** literals
    assert bound * (400 * 1024**2) < 0.0001


def test_importable_hidden_models_cannot_subtract_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    from fleetlab.contracts import tools
    from fleetlab.contracts.faults import FaultSpec
    from fleetlab.contracts.identity import EpisodeIdentity

    before = set(truth_terms())
    monkeypatch.setattr(tools, "FaultSpec", FaultSpec, raising=False)
    monkeypatch.setattr(tools, "EpisodeIdentity", EpisodeIdentity, raising=False)
    assert before <= set(truth_terms())
    assert {"fault_id", "onset_day", "scene_seed", "noise_seed", "start_state"} <= set(
        truth_terms()
    )


def test_main_cfedf66_terms_remain_terms() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "data/export-main-cfedf66-terms.json").read_text()
    )
    # Since 2026-10-19 max_steps is a public word of the job vocabulary (JobStep.max_steps),
    # as in tests/test_export_terms.py; the frozen fixture keeps main's list as it was.
    assert set(fixture["terms"]) - {"max_steps"} <= set(truth_terms())


def test_the_job_tools_public_words_are_not_terms() -> None:
    """`max_steps` (a job program step's field) and `params` (run_check's argument) are
    public names of the job tools since 19 October, so neither is a truth term."""
    assert not {"max_steps", "params"} & set(truth_terms())


# `params` left this list on 19 October: run_check's argument names it in public (lane T).
@pytest.mark.parametrize("term", ["level", "nominal", "schedule", "verdicts"])
def test_name_columns_map_keys_csv_and_metadata_use_name_matching(
    toy: tuple[Path, Path], tmp_path: Path, term: str
) -> None:
    schema = pa.schema([pa.field("metric", pa.string(), metadata={b"src": term.encode()})])
    pq.write_table(
        pa.Table.from_pylist([{"metric": term}], schema=schema), tmp_path / "health.parquet"
    )
    pq.write_table(
        pa.table({"data": pa.array([[(term, 2.0)]], pa.map_(pa.string(), pa.float64()))}),
        tmp_path / "maps.parquet",
    )
    (tmp_path / "labels.csv").write_text(f"episode_id,{term}\ne000001,2\n")
    report = audit_export(tmp_path, [toy[0]])
    for location in ("health.parquet.metric", "maps.parquet", "labels.csv"):
        assert any(h.term == term and location in h.where for h in report.hits), report.summary()
    assert any(h.term == term and "field metadata" in h.where for h in report.hits)


@pytest.mark.parametrize("column", ["metric", "key"])
@pytest.mark.parametrize(
    "dtype",
    [pa.binary(), pa.dictionary(pa.int32(), pa.string()), pa.dictionary(pa.int32(), pa.binary())],
)
def test_encoded_name_values_use_name_matching(
    toy: tuple[Path, Path], tmp_path: Path, column: str, dtype: pa.DataType
) -> None:
    value_type = dtype.value_type if pa.types.is_dictionary(dtype) else dtype
    value = b"row" if pa.types.is_binary(value_type) else "row"
    pq.write_table(pa.table({column: pa.array([value], dtype)}), tmp_path / "names.parquet")
    report = audit_export(tmp_path, [toy[0]])
    assert any(h.term == "row" and f"names.parquet.{column}" in h.where for h in report.hits)


@pytest.mark.parametrize(
    "dtype",
    [
        pa.string_view(),
        pa.binary(),
        pa.large_binary(),
        pa.binary_view(),
        pa.binary(100),
        pa.dictionary(pa.int32(), pa.binary()),
        pa.list_(pa.binary()),
        pa.list_view(pa.string()),
        pa.large_list_view(pa.string()),
        pa.struct([("note", pa.binary())]),
        pa.json_(),
    ],
)
def test_every_decodable_text_column_is_searched(
    toy: tuple[Path, Path], tmp_path: Path, dtype: pa.DataType
) -> None:
    text = "gripperPadAgeDays=232.5"
    if pa.types.is_struct(dtype):
        value: object = {"note": text.encode()}
    elif (
        pa.types.is_list(dtype)
        or pa.types.is_list_view(dtype)
        or pa.types.is_large_list_view(dtype)
    ):
        value = [text.encode()] if pa.types.is_binary(dtype.value_type) else [text]
    elif pa.types.is_dictionary(dtype):
        value = text.encode()
    elif isinstance(dtype, pa.BaseExtensionType):
        value = json.dumps({"note": text})
    elif pa.types.is_fixed_size_binary(dtype):
        value = text.encode().ljust(100, b" ")
    elif (
        pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_binary_view(dtype)
    ):
        value = text.encode()
    else:
        value = text
    pq.write_table(
        pa.table({"detail": pa.array([value], type=dtype)}),
        tmp_path / "nested.parquet",
        compression="zstd",
    )
    report = audit_export(tmp_path, [toy[0]])
    assert "gripper_pad_age_days" in {h.term for h in report.hits}, report.summary()


@pytest.mark.parametrize("suffix", [".gz", ".zip", ".npz", ".feather", ".parquet", ".bin"])
def test_undecodable_files_are_hits(toy: tuple[Path, Path], tmp_path: Path, suffix: str) -> None:
    (tmp_path / ("bad" + suffix)).write_bytes(b"\xff\xfe\xff")
    assert any(h.term == "(undecodable file)" for h in audit_export(tmp_path, [toy[0]]).hits)


def test_utf16_uppercase_suffixes_and_embedded_json_are_searched(
    toy: tuple[Path, Path], tmp_path: Path
) -> None:
    (tmp_path / "notes.LOG").write_bytes("true reason: placed".encode("utf-16"))
    (tmp_path / "notes.jsonl").write_text(json.dumps({"note": json.dumps({"schedule": 2})}) + "\n")
    report = audit_export(tmp_path, [toy[0]])
    assert {"true_reason", "schedule"} <= {h.term for h in report.hits}


def test_invalid_binary_column_is_a_hit(toy: tuple[Path, Path], tmp_path: Path) -> None:
    pq.write_table(pa.table({"detail": [b"\xff"]}), tmp_path / "bad.parquet")
    assert any(h.term == "(undecodable value)" for h in audit_export(tmp_path, [toy[0]]).hits)


def test_new_single_word_fault_kind_needs_a_name_context(
    toy: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.faults

    monkeypatch.setattr(fleetlab.faults, "load_all", lambda: ("haze", "drift", "row"))
    (tmp_path / "note.txt").write_text("the lens shows haze; drift on r03; row 2")
    assert {h.term for h in audit_export(tmp_path, [toy[0]]).hits} == {"haze"}
    (tmp_path / "note.txt").write_text("drift=2; row: 3")
    assert {"drift", "row"} <= {h.term for h in audit_export(tmp_path, [toy[0]]).hits}


def test_names_and_hidden_model_fields_have_independent_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fleetlab.log.leaks as leaks
    from fleetlab.contracts._base import HIDDEN
    from fleetlab.contracts.model import MODEL_CLASSES, Site

    class FutureSite(Site):
        concealed_site_property: Annotated[float, HIDDEN] = 0.0

    monkeypatch.setattr(leaks, "MODEL_CLASSES", (*MODEL_CLASSES, FutureSite))
    assert "concealed_site_property" in truth_terms()
    assert {t.name for t in SCHEMA.values() if t.tree == "truth"} <= set(truth_terms())
    (tmp_path / "trueReason").mkdir()
    pq.write_table(
        pa.table(
            {
                "data": pa.array(
                    [{"concealedSiteProperty": "x"}],
                    pa.struct([("concealedSiteProperty", pa.string())]),
                )
            }
        ),
        tmp_path / "nested.parquet",
    )
    report = audit_export(tmp_path, [])
    assert any(h.term == "true_reason" and h.where.startswith("name ") for h in report.hits)
    assert any(h.term == "concealed_site_property" and "field of" in h.where for h in report.hits)
