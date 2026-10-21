from __future__ import annotations

import json
from pathlib import Path

import pytest

from fleetlab.film import build, encode, manifest, sources
from fleetlab.film.manifest import ManifestError
from fleetlab.film.tests import fixtures


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    return fixtures.fleet_run(tmp_path / "fleet"), fixtures.episode_run(tmp_path / "spike")


def test_manifest_rejects_what_cannot_be_traced(tmp_path: Path) -> None:
    def load(beat: dict[str, object]) -> manifest.Manifest:
        data = {"format": manifest.FORMAT, "beats": [beat]}
        return manifest.parse(data, tmp_path / "m.json")

    with pytest.raises(ManifestError, match="needs its episode"):
        load({"id": "a", "shots": [{"kind": "clip", "clip": {"run": "x"}}]})
    with pytest.raises(ManifestError, match="literal count needs its source"):
        load({"id": "a", "panel": {"count": {"text": "3 of 4"}}, "shots": [{"kind": "slate"}]})
    with pytest.raises(ManifestError, match="unknown field"):
        load({"id": "a", "shots": [{"kind": "slate", "colour": "red"}]})
    m = load({"id": "a", "shots": [{"kind": "clip", "clip": {"run": "runs/x", "episode": "e1"}}]})
    assert m.beats[0].shots[0].clips[0].run == tmp_path / "runs" / "x"


def test_both_layouts_resolve_to_their_frames(runs: tuple[Path, Path]) -> None:
    fleet, chain = runs
    clip = sources.resolve(manifest.ClipRef(label="x", run=fleet, episode="e1"))
    assert clip.layout == "fleet-log" and clip.outcome == "failure"
    assert clip.frame_of_step[:9] == [0, 0, 0, 0, 1, 1, 1, 1, 2]  # the latest earlier frame
    assert clip.end == 40 and clip.info.frames == 11
    assert str(fleet / "log" / "episodes.parquet") in clip.files
    spike = sources.resolve(manifest.ClipRef(label="y", run=chain, episode="J1/naive/s00"))
    assert spike.layout == "episode-dir" and spike.outcome == "held_at_switch"
    assert spike.crop == (0, 0, 64, 64)  # the scene half of the side-by-side video
    assert (
        spike.program_index(0) == 0
        and spike.program_index(12) == 1
        and spike.program_index(30) == 2
    )
    with pytest.raises(ManifestError, match="not in"):
        sources.resolve(manifest.ClipRef(label="z", run=fleet, episode="e9"))
    release = sources.resolve(manifest.ClipRef(label="r", run=chain, episode="J1/release/s00"))
    assert [p.text for p in release.program] == [
        "put the cream cheese in the bowl",
        "transition motion",
        "put the bowl on the plate",
    ]
    assert release.program_index(15) == 1 and release.program_index(20) == 2


def test_counts_join_tables_and_refuse_to_count_twice(runs: tuple[Path, Path]) -> None:
    _, chain = runs
    tables = (chain / "results.jsonl", chain / "more.jsonl")
    spec = manifest.CountSpec(
        "{k} of {n}", tables=tables, where={"job": "J1"}, success="delivered", key=("start",)
    )
    value = sources.count_value(spec)
    assert value["text"] == "3 of 4" and len(value["tables"]) == 2
    twice = manifest.CountSpec(
        "{k} of {n}", tables=(tables[0], tables[0]), success="delivered", key=("job", "start")
    )
    with pytest.raises(ManifestError, match="appears twice"):
        sources.count_value(twice)


def test_a_fleet_clip_takes_no_rerender(runs: tuple[Path, Path], tmp_path: Path) -> None:
    fleet, _ = runs
    video = fleet / "log" / "video" / "e1.scene.mp4"
    record = tmp_path / "verify.json"
    record.write_text(json.dumps({"verified": True}))  # a generic claim is not a report
    ref = manifest.ClipRef(
        label="x", run=fleet, episode="e1", rerender=manifest.Rerender(video, record)
    )
    refused = sources.resolve(ref)
    assert refused.rerender is not None and refused.rerender["used"] is False
    assert "only for jobs runs" in refused.rerender["reason"] and refused.original is None


def test_build_writes_the_film_and_its_provenance(runs: tuple[Path, Path], tmp_path: Path) -> None:
    fleet, chain = runs
    script = tmp_path / "script.md"
    script.write_text("## naive: N\n\nThe chain fails. It is a test.\n")
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain, script))
    result = build.build(m, tmp_path / "out", draft=True)
    out = tmp_path / "out"
    for name in (
        "film.mp4",
        "captions.vtt",
        "chapters.json",
        "chapters.vtt",
        "provenance.json",
        "timeline.png",
        "timeline.md",
        "manifest.json",
        "poster.jpg",
    ):
        assert (out / name).is_file(), name
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["format"] == "fleetlab.film-provenance/1"
    by_id = {b["id"]: b for b in prov["beats"]}
    assert [b["status"] for b in prov["beats"]] == ["card", "placeholder", "record", "no footage"]
    naive = by_id["naive"]["shots"][0]
    assert naive["clips"][0]["video_sha256"] == sources.sha256(
        chain / "J1" / "naive" / "s00" / "scene.mp4"
    )
    (count,) = naive["counts"]
    assert count["text"] == "2 of 3 delivered" and count["kind"] == "computed"
    # the panel names the table the count was computed from, tied to its SHA-256
    naive_plan = next(b for b in result.beats if b.beat.id == "naive").shots[0]
    (on_screen,) = build._panel_state(naive_plan, 0, m.fps).counts
    table_sha = sources.sha256(chain / "results.jsonl")
    assert on_screen[2] is not None and table_sha[:8] in on_screen[2]
    assert "results.jsonl" in on_screen[2] and "/" not in on_screen[2].split(" ")[1][:1]
    pair = by_id["pair"]["shots"][0]["clips"]
    assert [c["episode"] for c in pair] == ["e1", "e2"]
    assert by_id["naive"]["narration"]["timing"] == "estimated reading pace"
    # the beat holds for its narration: lead-in, the estimated reading, the tail
    spoken = by_id["naive"]["narration"]["seconds"]
    assert by_id["naive"]["end_s"] - by_id["naive"]["start_s"] >= spoken + build.LEAD_S
    web = prov["outputs"][0]
    assert web["sha256"] == sources.sha256(out / "film.mp4")
    info = sources.video_info(out / "film.mp4")
    assert (info.width, info.height) == build.DRAFT
    assert info.frames == round(result.seconds * m.fps)
    assert "Stream" in " ".join(web["streams"].values()) and "aac" in " ".join(
        web["streams"].values()
    )
    assert "The chain fails." in (out / "captions.vtt").read_text()
    assert next(b for b in prov["beats"] if b["id"] == "naive")["narration"]["text"] == (
        "The chain fails. It is a test."
    )


def test_a_viewer_cut_draws_no_run_ids_or_digests(runs: tuple[Path, Path], tmp_path: Path) -> None:
    fleet, chain = runs
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain))
    assert m.sources_on_screen == "full"
    with pytest.raises(ManifestError, match="sources_on_screen"):
        manifest.load(fixtures.manifest(tmp_path / "x.json", fleet, chain, sources_on_screen="no"))
    viewer = manifest.load(
        fixtures.manifest(tmp_path / "v.json", fleet, chain, sources_on_screen="viewer")
    )
    plans, _ = build.plan(viewer)
    shots = {bp.beat.id: bp.shots[0] for bp in plans}
    full = " ".join(t for t, _ in build._source_segments(shots["pair"]))
    assert "e1" in full and shots["pair"].clips[0].video_sha256[:10] in full
    assert build._source_segments(shots["pair"], viewer=True) == []
    # placeholder and missing-footage texts stay; the computed count keeps its number only
    naive = " ".join(t for t, _ in build._source_segments(shots["naive"], viewer=True))
    assert naive == "Placeholder: a test run." and "J1" not in naive
    missing = build._source_segments(shots["missing"], viewer=True)
    assert [t for t, _ in missing] == ["No recorded run yet."]
    (count,) = build._panel_state(shots["naive"], 0, viewer.fps, viewer=True).counts
    assert count == ("2 of 3 delivered", None, None)
    # the provenance keeps every source and records the setting
    build.build(viewer, tmp_path / "out", draft=True, beats={"pair"})
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["options"]["sources_on_screen"] == "viewer"
    clips = prov["beats"][0]["shots"][0]["clips"]
    assert [c["episode"] for c in clips] == ["e1", "e2"] and all(c["video_sha256"] for c in clips)


def test_narration_audio_sets_the_beat_length(runs: tuple[Path, Path], tmp_path: Path) -> None:
    import numpy as np

    from fleetlab.film import narration

    fleet, chain = runs
    script = tmp_path / "script.md"
    script.write_text("## missing: M\n\nA beat that is read for longer than its slate lasts.\n")
    voice = tmp_path / "voice"
    voice.mkdir()
    wav = voice / "missing.wav"
    encode.write_wav(wav, np.full(encode.AUDIO_RATE * 3, 1000, dtype=np.int16))
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain, script))
    # audio that voice.json does not vouch for stops the build
    with pytest.raises(build.NarrationError, match=r"no voice\.json"):
        build.plan(m, narration_dir=voice, beats={"missing"})
    (voice / "missing.mp3").write_bytes(wav.read_bytes())
    wav.unlink()
    narration.voice_record(voice, script, "v", "m", "mp3_44100_128", ["missing"])
    result = build.build(m, tmp_path / "out", draft=True, narration_dir=voice, beats={"missing"})
    (beat,) = result.beats
    assert beat.audio == voice / "missing.mp3"
    assert beat.frames / m.fps == pytest.approx(build.LEAD_S + 3 + build.TAIL_S, abs=0.06)
    track = encode.decode_audio(tmp_path / "out" / "film.mp4")
    assert np.abs(track.astype(np.int32)).max() > 300  # the narration is in the sound track
    # a script revised after the recording, or audio changed since, stops the build
    script.write_text("## missing: M\n\nOther words now.\n")
    revised = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain, script))
    with pytest.raises(build.NarrationError, match="other words"):
        build.plan(revised, narration_dir=voice, beats={"missing"})
    script.write_text("## missing: M\n\nA beat that is read for longer than its slate lasts.\n")
    (voice / "missing.mp3").write_bytes(b"\x00" * 100)
    with pytest.raises(build.NarrationError, match="changed since"):
        build.plan(m, narration_dir=voice, beats={"missing"})


def test_a_jobs_run_and_its_verified_rerender(tmp_path: Path) -> None:
    run = fixtures.jobs_run(tmp_path / "jobs")
    clip = sources.resolve(manifest.ClipRef(label="j", run=run, episode="e000101"))
    assert clip.layout == "jobs-run" and clip.outcome == "delivered"
    assert [p.text for p in clip.program] == [
        "put the cream cheese in the bowl",
        "motion",
        "put the bowl on the plate",
    ]
    assert clip.program_index(5) == 0 and clip.program_index(12) == 1
    assert clip.program_index(20) == 3  # the tail: every program step is behind
    assert clip.program[2].outcome == "skipped"
    assert clip.frame_of_step[23] == 5 and clip.info.frames == 7
    video, report = fixtures.rerender(tmp_path / "rr", run)
    ref = manifest.ClipRef(
        label="j", run=run, episode="e000101", rerender=manifest.Rerender(video, report)
    )
    used = sources.resolve(ref)
    assert used.rerender is not None and used.rerender["used"] is True, used.rerender
    assert used.video == video and used.original is not None
    assert used.frame_of_step[:3] == [0, 1, 2] and used.frame_of_step[-1] == 24
    assert used.crop == (0, 0, 128, 128)  # the crop scaled to the re-render's size
    assert used.rerender["video_sha256"] == sources.sha256(video)
    assert used.rerender["comparison"]["hash_mismatches"] == 0
    assert used.rerender["comparison"]["hashes_compared"] == 25
    other = fixtures.jobs_run(tmp_path / "other")
    mismatches: dict[str, dict[str, object]] = {
        "not accepted": {"accepted": False, "label": "rejected re-render"},
        "another episode": {"episode_id": "e000202"},
        "another run": {"run_dir": str(other)},
        "another original": {"original_video": str(other / "log" / "video" / "e000101.scene.mp4")},
        "no hashes compared": {"hashes_compared": 0},
        "fewer steps": {"steps": 20},
        "another video": {"video": "e000101.rerender.scene.720px.mp4"},
        "frames": {"frames": 7},
        "a hash": {"video_sha256": "0" * 64},
    }
    for i, (name, change) in enumerate(mismatches.items()):
        video, report = fixtures.rerender(tmp_path / f"bad{i}", run, overrides=change)
        ref = manifest.ClipRef(
            label="j", run=run, episode="e000101", rerender=manifest.Rerender(video, report)
        )
        refused = sources.resolve(ref)
        assert refused.rerender is not None and refused.rerender["used"] is False, name
        assert refused.video != video and refused.original is None, name
    generic = tmp_path / "generic.json"
    generic.write_text(
        json.dumps({"verified": True, "state_hashes_match": True, "delivery_match": True})
    )
    ref = manifest.ClipRef(
        label="j", run=run, episode="e000101", rerender=manifest.Rerender(video, generic)
    )
    assert sources.resolve(ref).rerender["used"] is False  # type: ignore[index]


def test_messages_and_listings_come_from_records(runs: tuple[Path, Path], tmp_path: Path) -> None:
    fleet, chain = runs
    record = tmp_path / "run.json"
    log = [
        {"author": "lead", "text": "Cheese into the bowl, then bowl onto plate.", "session": 0},
        {"author": "agent", "text": "I propose 20 release starts and 16 deliveries.", "session": 0},
        {"author": "host", "text": "Approved.", "session": 0},
    ]  # fmt: skip
    record.write_text(json.dumps({"state": {"message_log": log}}))
    check = tmp_path / "check.json"
    check.write_text(
        json.dumps({"name": "undone", "purpose": "p", "sql": "SELECT 1\nFROM goal_steps"})
    )
    beats = [
        {"id": "req", "title": "The request", "shots": [
            {"kind": "messages", "file": str(record), "field": "state.message_log",
             "lines": [1, 2]}]},
        {"id": "check", "title": "The check", "shots": [
            {"kind": "text", "file": str(check), "field": "sql", "title": "undone"}]},
    ]  # fmt: skip
    path = fixtures.manifest(tmp_path / "m.json", fleet, chain, beats=beats)
    plans, _ = build.plan(manifest.load(path))
    assert plans[0].shots[0].messages == [
        ("Shift lead", log[0]["text"]),
        ("Agent", log[1]["text"]),
    ]
    assert plans[1].shots[0].text == "SELECT 1\nFROM goal_steps"
    assert str(record) in plans[0].shots[0].files
    bad = [
        {"id": "x", "title": "x", "shots": [{"kind": "text", "file": str(check), "field": "nope"}]}
    ]
    with pytest.raises(ValueError, match="does not lead anywhere"):
        build.plan(manifest.load(fixtures.manifest(tmp_path / "b.json", fleet, chain, beats=bad)))
    result = build.build(manifest.load(path), tmp_path / "out", draft=True)
    shots = [s for b in result.provenance["beats"] for s in b["shots"]]
    assert shots[0]["messages"][0]["from"] == "Shift lead" and shots[1]["field"] == "sql"


def test_the_agents_tool_calls_come_from_its_transcript(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    fleet, chain = runs
    transcript = tmp_path / "stdout.jsonl"
    events = [
        {"type": "thread.started"},
        {"type": "item.started", "item": {"type": "mcp_tool_call", "tool": "query"}},
        {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call",
         "tool": "query", "arguments": {"sql": "SELECT *\n  FROM goal_steps"},
         "status": "completed", "error": None}},
        {"type": "item.completed", "item": {"id": "i2", "type": "mcp_tool_call",
         "tool": "read_skill", "arguments": {"name": "x"}, "status": "completed"}},
        {"type": "item.completed", "item": {"id": "i3", "type": "mcp_tool_call",
         "tool": "run_check", "arguments": {"name": "undone", "params": {"episode": "e1"}},
         "status": "failed", "error": {"message": "no such check"}}},
        {"type": "item.completed", "item": {"id": "i4", "type": "agent_message", "text": "hi"}},
    ]  # fmt: skip
    transcript.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    beats = [
        {"id": "calls", "title": "The investigation", "shots": [
            {"kind": "calls", "file": str(transcript), "tools": ["query", "run_check"]}]},
    ]  # fmt: skip
    path = fixtures.manifest(tmp_path / "m.json", fleet, chain, beats=beats)
    plans, _ = build.plan(manifest.load(path))
    shot = plans[0].shots[0]
    assert shot.messages[0] == ("Agent · query", "SELECT * FROM goal_steps")
    assert shot.messages[1][0] == "Agent · run_check" and "not completed" in shot.messages[1][1]
    assert [c["id"] for c in shot.calls] == ["i1", "i3"]
    assert str(transcript) in shot.files
    bad = [{"id": "x", "title": "x", "shots": [{"kind": "calls", "tools": ["query"]}]}]
    with pytest.raises(ManifestError, match="transcript"):
        manifest.load(fixtures.manifest(tmp_path / "b.json", fleet, chain, beats=bad))
    result = build.build(manifest.load(path), tmp_path / "out", draft=True)
    shots = [s for b in result.provenance["beats"] for s in b["shots"]]
    assert shots[0]["calls"] == [{"id": "i1", "tool": "query"}, {"id": "i3", "tool": "run_check"}]


def test_a_bare_video_is_only_ever_a_labelled_placeholder(
    runs: tuple[Path, Path], tmp_path: Path
) -> None:
    fleet, chain = runs
    video = str(fleet / "log" / "video" / "e1.scene.mp4")

    def beat(placeholder: str | None, clip: dict[str, object]) -> dict[str, object]:
        return {"id": "b", "title": "B", "placeholder": placeholder,
                "shots": [{"kind": "clip", "clip": clip}]}  # fmt: skip

    def load(b: dict[str, object]) -> manifest.Manifest:
        return manifest.parse({"format": manifest.FORMAT, "beats": [b]}, tmp_path / "m.json")

    with pytest.raises(ManifestError, match="only in a beat marked as a placeholder"):
        load(beat(None, {"video": video, "steps_per_frame": 4}))
    with pytest.raises(ManifestError, match="frame cadence"):
        load(beat("an old video", {"video": video}))
    m = load(beat("an old video", {"video": video, "steps_per_frame": 4, "label": "old"}))
    plans, _ = build.plan(m)
    clip = plans[0].shots[0].clips[0]
    assert clip.placeholder and clip.layout == "video-file"
    record = clip.provenance()
    assert record["placeholder"] is True and record["run"] is None and record["episode"] is None
    assert build.final_cut_problems(plans)  # a final cut refuses it
    folder = chain / "J1" / "naive" / "s00" / "summary.json"
    folder.unlink()
    with pytest.raises(ManifestError, match="not a recorded episode"):
        sources.resolve(manifest.ClipRef(label="x", run=chain, episode="J1/naive/s00"))


def test_a_final_cut_holds_only_recorded_footage(runs: tuple[Path, Path], tmp_path: Path) -> None:
    fleet, chain = runs
    m = manifest.load(fixtures.manifest(tmp_path / "m.json", fleet, chain))
    with pytest.raises(build.FinalCutError, match="placeholder") as error:
        build.build(m, tmp_path / "out", draft=True, final=True)
    assert "no footage yet" in str(error.value)
    assert not (tmp_path / "out").exists()
    result = build.build(m, tmp_path / "out", draft=True, final=True, beats={"open", "pair"})
    assert result.provenance["final"] is True
