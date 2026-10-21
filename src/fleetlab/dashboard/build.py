"""Building the page bundles and publishing them.

    out/
      .dashboard                 marks the directory as the dashboard's own (it is cleared per page)
      progress/index.md          the progress page
      runs/<run id>/index.md     a run page, with its figures, frame strips and videos
      worlds/<world id>/index.md a world page: a world root, both runs merged (`world_page`)

Each page directory is one hub bundle with `index.md` as its entry and is
published as an artifact of its own, so that a page keeps its id while runs
come and go.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fleetlab.contracts.env import Camera
from fleetlab.dashboard.data import (
    DosePoint,
    Rate,
    Source,
    Truth,
    WorldRegistry,
    dose_points,
    load_source,
    load_truth,
    rates_by,
    sample_episodes,
)
from fleetlab.dashboard.figures import dose_figure, frame_strip, rate_figure
from fleetlab.dashboard.hub import (
    Manifest,
    PageEntry,
    check_bundle,
    hub_origin,
    publish_bundle,
)
from fleetlab.dashboard.pages import (
    GENERATED_PREFIX,
    EpisodeCard,
    RunPageData,
    RunRow,
    WorldRow,
    link,
    progress_page,
    run_page,
)
from fleetlab.dashboard.sources import Progress
from fleetlab.dashboard.world_page import (
    WorldSource,
    load_world,
    load_world_truth,
    write_world_page,
)
from fleetlab.log import LogReader

__all__ = ["BuildRefused", "BuiltPage", "Options", "build", "generated_line", "publish"]

MARKER = ".dashboard"
VIDEO_LIMIT = 25 * 1024 * 1024  # the hub CLI skips a bundle file larger than 25 MB


class BuildRefused(RuntimeError):
    """The build was asked for something the dashboard does not do."""


@dataclass(frozen=True)
class Options:
    truth: bool = False  # show the hidden side (development worlds only)
    sample: int = 6  # episodes with video per run page
    frames: int = 6  # frames per strip
    video: bool = True  # copy the sampled episodes' scene videos into the bundle
    allow_sealed: bool = False  # build pages of test and reserve worlds
    allow_unregistered: bool = False  # build pages of worlds not in dashboard/worlds.json


@dataclass(frozen=True)
class BuiltPage:
    key: str  # "progress", the run id or the world id
    directory: Path
    title: str
    tags: tuple[str, ...] = ()
    source: Source | None = None  # a run page's run
    world: WorldSource | None = None  # a world page's world


def generated_line(now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{GENERATED_PREFIX}{moment:%Y-%m-%d %H:%M} UTC._"


def _prepare(out: Path) -> None:
    """Create the output directory, or accept one that is the dashboard's own."""
    if out.exists():
        if not out.is_dir():
            raise BuildRefused(f"{out} is not a directory")
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise BuildRefused(
                f"{out} is not empty and not a dashboard build directory ({MARKER} missing)"
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(
        "fleetlab.dashboard build directory: pages are cleared and rewritten\n"
    )


def _front(artifact_id: str) -> str:
    """The front matter `hub publish` writes into an entry; written here when the id is known,
    so that the bundle on disk is the one the hub stores."""
    return f"---\nid: {artifact_id}\n---\n" if artifact_id else ""


def _fresh(directory: Path) -> Path:
    if directory.is_symlink():
        raise BuildRefused(f"{directory} is a symbolic link")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def _check_worlds(sources: Sequence[Source], registry: WorldRegistry, options: Options) -> None:
    for source in sources:
        for world in source.world_ids:
            if registry.sealed(world) and not options.allow_sealed:
                raise BuildRefused(
                    f"{source.run_id}: world {world} is a {registry.kind(world)} world; nobody "
                    "on the development side looks at it before the test (--allow-test-worlds)"
                )
            if not registry.registered(world) and not options.allow_unregistered:
                raise BuildRefused(
                    f"{source.run_id}: world {world} is not in dashboard/worlds.json; register "
                    "its kind first, since a test world must not reach the hub before the test "
                    "(--allow-unregistered-worlds)"
                )
        if not options.truth:
            continue
        if not source.is_run:
            raise BuildRefused(f"{source.run_id} is an export: there is no truth to show")
        if not source.world_ids:
            raise BuildRefused(f"{source.run_id}: no world id, so no truth")
        for world in source.world_ids:
            if not registry.truth_allowed(world):
                raise BuildRefused(
                    f"{source.run_id}: world {world} is {registry.kind(world)} in "
                    "dashboard/worlds.json; the truth is shown for development worlds only"
                )


def _check_world_roots(
    worlds: Sequence[WorldSource], registry: WorldRegistry, options: Options
) -> None:
    """The run pages' rules for a world page: its own world and the worlds of its runs."""
    for world in worlds:
        for wid in sorted({world.world_id, *world.run_world_ids}):
            if registry.sealed(wid) and not options.allow_sealed:
                raise BuildRefused(
                    f"world root {world.root}: world {wid} is a {registry.kind(wid)} world; "
                    "nobody on the development side looks at it before the test "
                    "(--allow-test-worlds)"
                )
            if not registry.registered(wid) and not options.allow_unregistered:
                raise BuildRefused(
                    f"world root {world.root}: world {wid} is not in dashboard/worlds.json; "
                    "register its kind first, since a test world must not reach the hub before "
                    "the test (--allow-unregistered-worlds)"
                )
            if options.truth and not registry.truth_allowed(wid):
                raise BuildRefused(
                    f"world root {world.root}: world {wid} is {registry.kind(wid)} in "
                    "dashboard/worlds.json; the truth is shown for development worlds only"
                )


def load_worlds(roots: Sequence[Path | str]) -> list[WorldSource]:
    worlds = [load_world(r) for r in roots]
    seen: dict[str, Path] = {}
    for w in worlds:
        if w.world_id in seen:
            raise BuildRefused(
                f"two world roots hold world {w.world_id}: {seen[w.world_id]} and {w.root}; "
                "a world page is named after its world"
            )
        seen[w.world_id] = w.root
    return worlds


def _check_world_manifest(worlds: Sequence[WorldSource], manifest: Manifest) -> None:
    for w in worlds:
        entry = manifest.worlds.get(w.world_id)
        if entry and entry.root and Path(entry.root) != w.root:
            raise BuildRefused(
                f"world {w.world_id} was published from {entry.root}, not {w.root}: change its "
                "entry in dashboard/ids.json if the world moved"
            )


def _run_links(world: WorldSource, sources: Sequence[Source], manifest: Manifest) -> dict[str, str]:
    """Each run of a world as a link to its run page, where that page exists."""
    built = {s.root for s in sources}
    out = {}
    for run in world.runs:
        directory = run.directory.resolve()
        entry = manifest.runs.get(run.run_id)
        if entry and entry.url and (not entry.run_dir or Path(entry.run_dir) == directory):
            out[run.name] = link(run.run_id, entry.url)
        elif directory in built:
            out[run.name] = f"{run.run_id} (page built, not published yet)"
    return out


def build_world(
    world: WorldSource,
    out: Path,
    truth: bool,
    registry: WorldRegistry,
    manifest: Manifest,
    sources: Sequence[Source],
    generated: str,
) -> BuiltPage:
    page = _fresh(out / "worlds" / world.world_id)
    known = manifest.worlds.get(world.world_id)
    write_world_page(
        world,
        page,
        kind=registry.kind(world.world_id),
        note=registry.notes.get(world.world_id, ""),
        truth=load_world_truth(world) if truth else None,
        run_links=_run_links(world, sources, manifest),
        progress_url=manifest.progress_url(),
        generated=generated,
        front=_front(known.id if known else ""),
    )
    tags = ("truth",) if truth else ()
    return BuiltPage(world.world_id, page, f"Dashboard: world {world.world_id}", tags, world=world)


def _world_rows(
    worlds: Sequence[WorldSource],
    registry: WorldRegistry,
    manifest: Manifest,
    sources: Sequence[Source],
) -> list[WorldRow]:
    """The world pages of this build and those published before, for the progress page."""
    rows: dict[str, WorldRow] = {}
    known: list[tuple[WorldSource | None, str, str]] = [(w, w.world_id, "") for w in worlds]
    for world_id, entry in manifest.worlds.items():
        if any(w.world_id == world_id for w in worlds):
            continue
        try:
            known.append((load_world(entry.root), world_id, ""))
        except Exception as error:  # a listing of other worlds: one damaged root must not stop it
            known.append((None, world_id, f"not readable: {type(error).__name__}"))
    for world, world_id, note in known:
        url = manifest.world_url(world_id)
        if world is None:
            entry = manifest.worlds[world_id]
            rows[world_id] = WorldRow(
                world_id, url, registry.kind(world_id), root=entry.root, note=note
            )
            continue
        links = _run_links(world, sources, manifest)
        overall = world.overall
        rows[world_id] = WorldRow(
            world_id=world_id,
            url=url,
            kind=registry.kind(world_id),
            ready="ready" if world.ready else "not ready",
            runs=", ".join(links.get(r.name) or r.run_id for r in world.runs),
            episodes=str(overall.n),
            success=overall.text(),
            root=str(world.root),
            note="" if url else "not published yet",
        )
    return [rows[k] for k in sorted(rows)]


def _is_calibration(source: Source, registry: WorldRegistry) -> bool:
    return bool(source.world_ids) and all(
        registry.kind(w) == "calibration" for w in source.world_ids
    )


def _video_path(reader: LogReader, relative: object) -> Path | None:
    """The video file an `episodes` row names, if it exists inside the log tree."""
    if not relative:
        return None
    path = (reader.log_dir / str(relative)).resolve()
    if not path.is_relative_to(reader.log_dir) or not path.is_file():
        return None
    return path


def _strip_steps(reader: LogReader, episode_id: str, count: int) -> list[int]:
    """`count` logged frames, evenly spaced from the first to the last."""
    logged = [
        int(r["step"])
        for r in reader.query(
            "SELECT step FROM steps WHERE episode_id = ? AND frame_index IS NOT NULL ORDER BY step",
            [episode_id],
        ).to_pylist()
    ]
    if not logged:
        return []
    if count <= 1 or len(logged) == 1:
        return [logged[0]]
    k = min(count, len(logged))
    return [logged[round(i * (len(logged) - 1) / (k - 1))] for i in range(k)]


def _cards(source: Source, page: Path, options: Options, truth: Truth | None) -> list[EpisodeCard]:
    cards = []
    sample = sample_episodes(source.episodes, options.sample)
    with LogReader(source.root) as reader:
        for row in sample:
            eid = str(row["episode_id"])
            steps = _strip_steps(reader, eid, options.frames)
            problems = [] if steps else ["no logged frame"]
            cameras, frames = [], []
            for camera in Camera:
                video_file = _video_path(reader, row.get(f"video_{camera.value}"))
                if video_file is None:
                    if row.get(f"video_{camera.value}"):
                        problems.append(f"the {camera.value} video file is missing")
                    continue
                try:
                    images = [f.image for f in reader.frames(eid, steps, camera)] if steps else []
                except (ValueError, OSError) as error:  # a damaged file: say so, keep the page
                    problems.append(f"the {camera.value} video could not be read ({error})")
                    continue
                if images:
                    cameras.append(camera)
                    frames.append(images)
            strip = ""
            if frames:
                strip = f"{eid}.strip.png"
                frame_strip(page / strip, frames)
            video = ""
            scene = _video_path(reader, row.get("video_scene"))
            if options.video and scene is not None:
                if scene.stat().st_size > VIDEO_LIMIT:
                    problems.append(
                        f"the scene video is {scene.stat().st_size // 2**20} MB, over the limit"
                    )
                else:
                    video = f"{eid}.scene.mp4"
                    shutil.copyfile(scene, page / video)
            note = ("Not shown: " + "; ".join(problems) + ".") if problems else ""
            faults: tuple[str, ...] = ()
            verdict = ""
            if truth is not None:
                identity = truth.identities.get(eid)
                if identity is not None:
                    faults = tuple(
                        f"{f.fault_id} {f.kind} at {f.level:g}"
                        for f in sorted(identity.faults, key=lambda f: f.fault_id)
                    )
                known = truth.true_success.get(eid)
                verdict = "unknown" if known is None else ("success" if known else "failure")
            cards.append(
                EpisodeCard(
                    row=row,
                    strip=strip,
                    strip_steps=tuple(steps),
                    cameras=tuple(c.value for c in cameras),
                    video=video,
                    video_note=note,
                    faults=faults,
                    verdict=verdict,
                )
            )
    return cards


def build_run(
    source: Source,
    out: Path,
    options: Options,
    registry: WorldRegistry,
    manifest: Manifest,
    generated: str,
    dose: Sequence[DosePoint] = (),
) -> BuiltPage:
    page = _fresh(out / "runs" / source.run_id)
    truth = load_truth(source) if options.truth else None
    figures: dict[str, str] = {}
    overall = source.overall
    if source.episodes:
        rate_figure(
            page / "success-by-robot.png",
            rates_by(source.episodes, "robot_id"),
            overall,
            "Success by robot",
            "robot",
        )
        figures["robot"] = "success-by-robot.png"
        rate_figure(
            page / "success-by-day.png",
            rates_by(source.episodes, "day"),
            overall,
            "Success by day",
            "day",
        )
        figures["day"] = "success-by-day.png"
    if any(p.kind is not None for p in dose):
        dose_figure(page / "dose-response.png", dose)
        figures["dose"] = "dose-response.png"
    world_kinds = sorted({registry.kind(w) for w in source.world_ids}) or ["unknown"]
    notes = sorted({registry.notes.get(w, "") for w in source.world_ids} - {""})
    data = RunPageData(
        source=source,
        world_kind=", ".join(world_kinds),
        world_note="; ".join(notes),
        figures=figures,
        cards=_cards(source, page, options, truth),
        dose=list(dose),
        truth=truth,
        progress_url=manifest.progress_url(),
    )
    known = manifest.runs.get(source.run_id)
    (page / "index.md").write_text(_front(known.id if known else "") + run_page(data, generated))
    tags = ("truth",) if truth is not None else ()
    return BuiltPage(source.run_id, page, f"Dashboard: run {source.run_id}", tags, source)


def _doses(sources: Sequence[Source], registry: WorldRegistry) -> dict[str, list[DosePoint]]:
    """The dose-response points of the calibration runs of a build.

    The calibration runs one healthy run and one run per fault kind; a run
    without healthy episodes of its own gets those of the build's other
    calibration runs of the same world, pooled, so its page has the band.
    """
    points = {
        s.run_id: dose_points(load_truth(s, gains=False), s.episodes)
        for s in sources
        if s.is_run and s.episodes and _is_calibration(s, registry)
    }
    worlds = {s.run_id: s.world_ids for s in sources}
    out = {}
    for run_id, mine in points.items():
        if mine and all(p.kind is not None for p in mine):
            donors = [
                (other, p)
                for other, theirs in sorted(points.items())
                if other != run_id and worlds[other] == worlds[run_id]
                for p in theirs
                if p.kind is None
            ]
            if donors:
                rate = Rate(
                    sum(p.rate.n for _, p in donors), sum(p.rate.successes for _, p in donors)
                )
                runs = ", ".join(other for other, _ in donors)
                mine = [DosePoint(None, None, rate, runs), *mine]
        out[run_id] = mine
    return out


def _known_runs(sources: Sequence[Source], manifest: Manifest) -> list[RunRow]:
    rows = {s.run_id: RunRow(s.run_id, manifest.run_url(s.run_id), s) for s in sources}
    for run_id, entry in manifest.runs.items():
        if run_id in rows:
            continue
        try:
            source: Source | None = load_source(entry.run_dir, run_id)
            note = ""
        except Exception as error:  # a listing of other runs: one damaged run must not stop it
            source, note = None, f"not readable: {type(error).__name__}"
        rows[run_id] = RunRow(run_id, entry.url, source, note)
    return [rows[k] for k in sorted(rows)]


def build_progress(
    sources: Sequence[Source],
    out: Path,
    registry: WorldRegistry,
    manifest: Manifest,
    progress: Progress,
    generated: str,
    worlds: Sequence[WorldSource] = (),
) -> BuiltPage:
    page = _fresh(out / "progress")
    runs = _known_runs(sources, manifest)
    world_rows = _world_rows(worlds, registry, manifest, sources)
    known = manifest.progress.id if manifest.progress else ""
    (page / "index.md").write_text(
        _front(known) + progress_page(runs, registry, progress, generated, world_rows)
    )
    return BuiltPage("progress", page, "Dashboard: progress")


def load_sources(run_dirs: Sequence[Path | str]) -> list[Source]:
    sources = [load_source(d) for d in run_dirs]
    seen: dict[str, Path] = {}
    for s in sources:
        if s.run_id in seen:
            raise BuildRefused(
                f"two runs are named {s.run_id}: {seen[s.run_id]} and {s.root}; "
                "a page is named after its directory"
            )
        seen[s.run_id] = s.root
    return sources


def _check_manifest(sources: Sequence[Source], manifest: Manifest) -> None:
    for s in sources:
        entry = manifest.runs.get(s.run_id)
        if entry and entry.run_dir and Path(entry.run_dir) != s.root:
            raise BuildRefused(
                f"run {s.run_id} was published from {entry.run_dir}, not {s.root}: rename the "
                "directory, or change its entry in dashboard/ids.json"
            )


def build(
    run_dirs: Sequence[Path | str],
    out: Path,
    options: Options,
    registry: WorldRegistry,
    manifest: Manifest,
    progress: Progress,
    now: datetime | None = None,
    world_roots: Sequence[Path | str] = (),
) -> list[BuiltPage]:
    """Write a page per run, a page per world root and the progress page; returns them in
    that order."""
    sources = load_sources(run_dirs)
    worlds = load_worlds(world_roots)
    _check_worlds(sources, registry, options)
    _check_world_roots(worlds, registry, options)
    _check_manifest(sources, manifest)
    _check_world_manifest(worlds, manifest)
    _prepare(out)
    generated = generated_line(now)
    doses = _doses(sources, registry)
    pages = [
        build_run(s, out, options, registry, manifest, generated, doses.get(s.run_id, ()))
        for s in sources
    ]
    pages += [
        build_world(w, out, options.truth, registry, manifest, sources, generated) for w in worlds
    ]
    pages.append(build_progress(sources, out, registry, manifest, progress, generated, worlds))
    return pages


def publish(
    pages: Sequence[BuiltPage],
    out: Path,
    registry: WorldRegistry,
    manifest: Manifest,
    progress: Progress,
    now: datetime | None = None,
    log: Any = print,
) -> list[tuple[BuiltPage, PageEntry]]:
    """Check and publish the built pages: runs first, then the world pages, then the progress
    page, each with the links of those before.

    The manifest is saved after each publish. When a run page got a new id, the
    world pages and the progress page are rebuilt with its link before they are
    published; when a world page got one, the progress page is.
    """
    done = []
    sources = [p.source for p in pages if p.source is not None]
    origin = hub_origin()  # once, before anything is published
    new_link = False
    for page in pages:
        if page.source is None:
            continue
        for warning in check_bundle(page.directory):
            log(f"{page.key}: {warning}")
        known = manifest.runs.get(page.key)
        entry = publish_bundle(
            page.directory, page.title, known.id if known else None, page.tags, origin
        )
        entry.run_dir = str(page.source.root)
        entry.world = page.source.world
        new_link = new_link or known is None or known.url != entry.url
        manifest.runs[page.key] = entry
        manifest.save()
        done.append((page, entry))
        log(f"{page.key}: {entry.url or entry.id}")
    worlds = [p.world for p in pages if p.world is not None]
    run_link = new_link
    for page in pages:
        world = page.world
        if world is None:
            continue
        if run_link:
            truth = "truth" in page.tags
            page = build_world(world, out, truth, registry, manifest, sources, generated_line(now))
        for warning in check_bundle(page.directory):
            log(f"world {page.key}: {warning}")
        known = manifest.worlds.get(page.key)
        entry = publish_bundle(
            page.directory, page.title, known.id if known else None, page.tags, origin
        )
        entry.root = str(world.root)
        entry.world = page.key
        new_link = new_link or known is None or known.url != entry.url
        manifest.worlds[page.key] = entry
        manifest.save()
        done.append((page, entry))
        log(f"world {page.key}: {entry.url or entry.id}")
    progress_pages = [
        p for p in pages if p.key == "progress" and p.source is None and p.world is None
    ]
    if progress_pages:
        page = progress_pages[0]
        if new_link:
            page = build_progress(
                sources, out, registry, manifest, progress, generated_line(now), worlds
            )
        for warning in check_bundle(page.directory):
            log(f"progress: {warning}")
        known_id = manifest.progress.id if manifest.progress else None
        entry = publish_bundle(page.directory, page.title, known_id, (), origin)
        manifest.progress = entry
        manifest.save()
        done.append((page, entry))
        log(f"progress: {entry.url or entry.id}")
    return done
