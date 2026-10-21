"""python -m fleetlab.film: build the film, check a manifest, build the site, prepare narration.

build MANIFEST --out DIR [--narration DIR] [--draft] [--repeatable] [--story A] [--beats a,b]
check MANIFEST [--narration DIR]           resolve every clip and count; print the timeline
site BUNDLE --out DIR                      the static site (BUNDLE: site.json or its folder)
bind SCRIPT --records FILE --out DIR [--fields FILE] [--final]
                                           the script with its numbers read from the records
narration SCRIPT --out DIR [--model M] [--require-bound]
                                           per beat: text, request bodies, captions
voices LISTING [--models LISTING]          two male American narration voices and the model
probe-figure EXPORT_LOG --probe ID --robot LABEL=ROBOT ... --out PNG
                                           a recorded probe drawn, commanded against measured
pairs --before RUN --after RUN [--before-where k=v] [--after-where k=v]
                                           rescued and broken starts; the first of each
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]


def _beats(value: str | None) -> set[str] | None:
    return None if not value else {v.strip() for v in value.split(",") if v.strip()}


def _cmd_build(args: argparse.Namespace) -> int:
    from fleetlab.film import build, manifest

    m = manifest.load(args.manifest)
    result = build.build(
        m,
        args.out.expanduser(),
        narration_dir=args.narration.expanduser() if args.narration else None,
        draft=args.draft,
        repeatable=args.repeatable,
        beats=_beats(args.beats),
        story=args.story,
        final=args.final,
    )
    for path in result.outputs:
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {result.seconds:.1f} s)")
    print(f"wrote {result.out / 'provenance.json'}, timeline.png, timeline.md, captions.vtt")
    for bp in result.beats:
        print(
            f"  {bp.beat.id:<14} {bp.start / m.fps:6.1f} s  {bp.frames / m.fps:5.1f} s  {bp.status}"
        )
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 1 if any("over GitHub Pages" in w for w in result.warnings) else 0


def _cmd_check(args: argparse.Namespace) -> int:
    from fleetlab.film import build, manifest

    m = manifest.load(args.manifest)
    plans, warnings = build.plan(
        m, narration_dir=args.narration.expanduser() if args.narration else None
    )
    fps = m.fps
    for bp in plans:
        print(f"{bp.beat.id:<14} {bp.start / fps:6.1f} s  {bp.frames / fps:5.1f} s  {bp.status}")
        for sp in bp.shots:
            for clip in sp.clips:
                print(
                    f"    {sp.shot.kind:<5} {clip.layout:<11} {clip.run} {clip.episode} "
                    f"steps {clip.start}-{clip.end} {clip.video_sha256[:12]} {clip.outcome or ''}"
                )
            for count in sp.counts:
                print(f"    count: {count['text']} ({count['kind']})")
    total = sum(bp.frames for bp in plans) / fps
    print(f"{len(plans)} beats, {total:.1f} s; every clip resolved")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _cmd_site(args: argparse.Namespace) -> int:
    from fleetlab.film import website

    result = website.build_site(args.bundle.expanduser(), args.out.expanduser())
    print(f"wrote {result.out} ({result.files} files, {result.bytes / 1e6:.1f} MB)")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 1 if result.errors else 0


def _cmd_narration(args: argparse.Namespace) -> int:
    from fleetlab.film import encode, narration

    beats = narration.load_script(args.script)
    out: Path = args.out.expanduser()
    unbound = {b.id: b.unbound for b in beats if b.unbound}
    if unbound and args.require_bound:
        for beat_id, names in unbound.items():
            print(f"{beat_id}: unbound {', '.join('{' + n + '}' for n in names)}", file=sys.stderr)
        print(
            "bind the script to its records first (python -m fleetlab.film bind)", file=sys.stderr
        )
        return 4
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_cues: list[narration.Cue] = []
    at = 0.0
    measured_any = False
    for i, beat in enumerate(beats):
        if not beat.text:
            continue
        (out / f"{beat.id}.txt").write_text(beat.text + "\n", encoding="utf-8")
        previous = beats[i - 1] if i > 0 else None
        following = beats[i + 1] if i + 1 < len(beats) else None
        bodies = narration.request_bodies(
            beat, model=args.model, previous=previous, following=following
        )
        for j, body in enumerate(bodies, 1):
            narration.write_json(out / f"{beat.id}.request-{j}.json", body)
        audio = out / f"{beat.id}.mp3"
        if audio.is_file():
            seconds = len(encode.decode_audio(audio)) / encode.AUDIO_RATE
            timing = "measured"
            measured_any = True
        else:
            seconds = narration.spoken_seconds(beat)
            timing = "estimated"
        cues = narration.cues(beat, seconds)
        (out / f"{beat.id}.vtt").write_text(narration.vtt(cues), encoding="utf-8")
        all_cues += narration.cues(beat, seconds, offset=at)
        at += seconds + 1.0
        rows.append(
            {
                "text_sha256": narration.text_digest(beat),
                "id": beat.id,
                "title": beat.title,
                "words": beat.words,
                "seconds": round(seconds, 1),
                "timing": timing,
                "requests": len(bodies),
            }
        )
    (out / "captions.vtt").write_text(narration.vtt(all_cues), encoding="utf-8")
    (out / "beats.tsv").write_text(
        "".join(f"{r['id']}\t{r['requests']}\n" for r in rows), encoding="utf-8"
    )
    info = narration.summary([b for b in beats if b.text])
    info["timing"] = "measured where a voice exists" if measured_any else "estimated reading pace"
    info["beats"] = rows
    info["seconds"] = round(sum(r["seconds"] for r in rows), 1)
    narration.write_json(out / "narration.json", info)
    print(
        f"{len(rows)} beats, {info['words']} words, {info['seconds']} s "
        f"({info['timing']}, {narration.WORDS_PER_MINUTE} words a minute where estimated)"
    )
    return 0


def _cmd_bind(args: argparse.Namespace) -> int:
    from fleetlab.film import fields

    records = args.records.expanduser() if args.records else None
    bound, report = fields.bind_script(args.script, args.fields, records, args.out.expanduser())
    for name, value in report["values"].items():
        source = (
            value.get("field")
            or value.get("count_true")
            or (f"distinct {value['count_distinct']}" if value.get("count_distinct") else "rows")
        )
        print(f"bound   {{{name}}} = {value['value']}  ({value['record']}: {source})")
    for name, why in report["unbound"].items():
        print(f"unbound {{{name}}}: {why}", file=sys.stderr)
    print(f"wrote {bound} and {bound.parent / 'bindings.json'}")
    return 4 if args.final and report["unbound"] else 0


def _cmd_voices(args: argparse.Namespace) -> int:
    from fleetlab.film import narration

    listing = json.loads(args.listing.read_text())
    models = json.loads(args.models.read_text()) if args.models else None
    picked = narration.pick_voices(listing, count=args.count)
    model = narration.pick_model(models if isinstance(models, list) else None)
    for voice in picked:
        print(
            f"voice\t{voice['voice_id']}\t{voice['name']}\t"
            + json.dumps(voice["labels"], sort_keys=True)
        )
    print(f"model\t{model}")
    if len(picked) < args.count:
        print(
            f"only {len(picked)} premade voice(s) are labelled male and American", file=sys.stderr
        )
        return 3
    return 0


def _cmd_voice_record(args: argparse.Namespace) -> int:
    from fleetlab.film import narration

    ids = [b for b in args.beats.split(",") if b]
    narration.voice_record(args.dir, args.script, args.voice, args.model, args.format, ids)
    print(f"wrote {args.dir / 'voice.json'} ({len(ids)} beats)")
    return 0


def _cmd_probe_figure(args: argparse.Namespace) -> int:
    from fleetlab.film import probefig

    robots = []
    for item in args.robot:
        label, _, robot = item.rpartition("=")
        robots.append((label.strip(), robot.strip()))
    record = probefig.draw(
        args.log.expanduser(), args.probe, robots, args.out.expanduser(), title=args.title
    )
    print(f"wrote {args.out} and its record ({len(record['episodes'])} probe episodes)")
    return 0


def _cmd_pairs(args: argparse.Namespace) -> int:
    from fleetlab.film import pick

    before: list[pick.Episode] = []
    after: list[pick.Episode] = []
    files: dict[str, str] = {}
    for runs, where, side in (
        (args.before, args.before_where, before),
        (args.after, args.after_where, after),
    ):
        for run in runs:
            episodes, read = pick.load_side(run, pick.parse_where(where))
            side += episodes
            files.update(read)
    report = pick.pair(before, after, files)
    print(json.dumps(report.summary(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetlab.film", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="assemble the film")
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--narration", type=Path, help="a directory of <beat id>.mp3 from tools/film-narrate"
    )
    p.add_argument("--draft", action="store_true", help="640 x 360 only, fast; no master")
    p.add_argument("--repeatable", action="store_true", help="one encoder thread, bit-exact")
    p.add_argument("--story", help="only this story's beats (and beats of no story)")
    p.add_argument("--beats", help="only these beat ids, comma-separated")
    p.add_argument(
        "--final", action="store_true", help="refuse placeholders, slates and bare videos"
    )
    p.set_defaults(run=_cmd_build)
    p = sub.add_parser("check", help="resolve every clip and count without rendering")
    p.add_argument("manifest", type=Path)
    p.add_argument("--narration", type=Path)
    p.set_defaults(run=_cmd_check)
    p = sub.add_parser("site", help="build the static site")
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=_cmd_site)
    p = sub.add_parser("bind", help="bind the script's numbers to fields of recorded runs")
    p.add_argument("script", type=Path)
    p.add_argument("--records", type=Path, help="the cut's fleetlab.film-records/1 file")
    p.add_argument(
        "--fields",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "site" / "narration" / "fields.json",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--final", action="store_true", help="fail when a number stays unbound")
    p.set_defaults(run=_cmd_bind)
    p = sub.add_parser("narration", help="per beat: text, request bodies and captions")
    p.add_argument("script", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="eleven_multilingual_v2")
    p.add_argument(
        "--require-bound", action="store_true", help="refuse a script with unbound numbers"
    )
    p.set_defaults(run=_cmd_narration)
    p = sub.add_parser("voices", help="choose voices and the model from the account's listings")
    p.add_argument("listing", type=Path, help="the JSON of GET /v1/voices")
    p.add_argument("--models", type=Path, help="the JSON of GET /v1/models")
    p.add_argument("--count", type=int, default=2)
    p.set_defaults(run=_cmd_voices)
    p = sub.add_parser("voice-record", help="voice.json for the beats a run recorded")
    p.add_argument("dir", type=Path)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--format", default="mp3_44100_128")
    p.add_argument("--beats", required=True, help="the recorded beat ids, comma-separated")
    p.set_defaults(run=_cmd_voice_record)
    p = sub.add_parser("probe-figure", help="draw a recorded probe, commanded against measured")
    p.add_argument("log", type=Path, help="the export's log/ with probe_episodes.parquet")
    p.add_argument("--probe", required=True)
    p.add_argument("--robot", action="append", required=True, help="LABEL=ROBOT, one per panel")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title")
    p.set_defaults(run=_cmd_probe_figure)
    p = sub.add_parser("pairs", help="before-and-after pairs on the same starts, by rule")
    p.add_argument("--before", type=Path, action="append", required=True, help="a run (repeat)")
    p.add_argument("--after", type=Path, action="append", required=True, help="a run (repeat)")
    p.add_argument("--before-where", action="append", default=[], help="column=value")
    p.add_argument("--after-where", action="append", default=[], help="column=value")
    p.set_defaults(run=_cmd_pairs)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
