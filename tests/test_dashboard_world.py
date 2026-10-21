"""The dashboard's world page on a world root: generator files, two runs, the runner's record.

src/fleetlab/dashboard/world_page.py. The world is the CPU mechanics fixture D2
(synthetic acceptance calibration),
(every pattern: a drift, an interaction, a confounder, decoys), generated on the
CPU in a few seconds; its two runs are written through the log's writers from
one episode per robot and day of the world's own manifest, with outcomes drawn
from the episode number (nothing is simulated). Nothing here calls bd or the
hub: publishing runs against a stand-in `hub` script.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import pytest
from log_toy import run_jobs, write_toy_run
from test_worlds_v1 import example_curves, recipes

from fleetlab.contracts.env import REASON_PLACED, REASON_STEP_LIMIT, Camera
from fleetlab.contracts.hashing import hash_bytes
from fleetlab.contracts.identity import REPLAY_RULES, SoftwareVersion, Task
from fleetlab.contracts.live import StepTelemetry
from fleetlab.contracts.log import EventKind
from fleetlab.contracts.rollout import Job, RunRecord
from fleetlab.dashboard.__main__ import main
from fleetlab.dashboard.build import BuildRefused, Options, build, publish
from fleetlab.dashboard.data import WorldRegistry
from fleetlab.dashboard.hub import Manifest, PageEntry
from fleetlab.dashboard.pages import GENERATED_PREFIX
from fleetlab.dashboard.sources import Progress
from fleetlab.dose_response import wilson
from fleetlab.log import EpisodeWriter, RunWriter
from fleetlab.log.runfiles import read_identities
from fleetlab.worlds import curves
from fleetlab.worlds.base import day_level
from fleetlab.worlds.estimate import probability
from fleetlab.worlds.layout import RUN_WINDOWS, WorldPaths
from fleetlab.worlds.recipe import TASKS
from fleetlab.worlds.world import WorldV1, context, generate, write_world

WORLD = recipes()["D2"].world_id
REGISTRY = WorldRegistry({WORLD: "development"}, {WORLD: "the test's world"})
NOW = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
SOFTWARE = SoftwareVersion(
    lock_hash="0" * 16, libero_patch_set="1" * 16, mujoco_version="3.3.7", miopen_db_hash="2" * 16
)
TEXTS = {Task(t.suite, t.task_index): f"do task {t.task_index} of {t.suite}" for t in TASKS}
ROLE_WORDS = ("decoy", "confounder", "inert", "interaction", "drift")
SUMMARY = {
    "observed_logged_success": {"measured": 0.31, "expected": 0.653},
    "gate3": {"band": False, "sizing": True},
}


# ---------------------------------------------------------------------------- the fixture


def write_episode(run_dir: Path, job: Job, text: str, per_day: int) -> None:
    """One episode of a few steps; success from the episode number and its faults."""
    identity = job.identity
    number = int(identity.episode_id[1:])
    rng = np.random.default_rng(number)
    writer = EpisodeWriter(run_dir, job, task_text=text, episodes_per_robot_day=per_day)
    last = 3
    for k in range(last + 1):
        writer.add_step(
            StepTelemetry(
                step=k,
                time_s=k * 0.05,
                joint_pos=rng.normal(size=7),
                joint_vel=rng.normal(size=7),
                gripper_qpos=rng.uniform(0, 0.04, size=2),
                eef_pos=rng.normal(size=3),
                eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
                commanded_action=(
                    np.full(7, np.nan, np.float32)
                    if k == last
                    else rng.normal(size=7).astype(np.float32)
                ),
                chunk_index=k // 2,
                chunk_step=k % 2,
            )
        )
        tag = f"{job.key}/{k}".encode()
        hashes = {c: hash_bytes(tag + c.value.encode()) for c in Camera}
        executed = None if k == 0 else rng.normal(size=7).astype(np.float32)
        writer.add_chain(k, hash_bytes(tag), hashes, executed)
    success = (number * 7) % 10 >= 2 + 2 * len(identity.faults)
    if number % 5 == 0:
        writer.add_event(1, EventKind.grasp, "closed on the object")
    if number % 7 == 0:
        writer.add_event(2, EventKind.drop, "the object fell")
    reason = REASON_PLACED if success else REASON_STEP_LIMIT
    writer.set_verdict(
        success, reason, 2, last if success else None, placed_step=1 if success else None
    )
    writer.set_fixture_hash(hash_bytes(f"fixture/{identity.episode_id}".encode()))
    writer.finish(outcome=success)


def write_run(run_dir: Path, world: WorldV1, name: str) -> None:
    """A finalized run of one episode per robot and day of the run's windows, with the
    fleet's records as the runner writes them."""
    spec = world.spec
    windows = RUN_WINDOWS[name]
    days = {
        d
        for w in windows
        for d in range(getattr(spec.windows, w).first, getattr(spec.windows, w).last + 1)
    }
    jobs = [
        j.model_copy(update={"record_video": False})
        for j in world.jobs
        if j.identity.day in days and j.identity.episode_index == 0
    ]
    per_day = spec.episodes_per_robot_day
    record = RunRecord(software=SOFTWARE, batch_size=4, replay_rules=[r.name for r in REPLAY_RULES])
    with RunWriter(run_dir, record, episodes_per_robot_day=per_day) as run:
        for job in jobs:
            text = world.task_texts[Task(job.identity.suite, job.identity.task_index)]
            write_episode(run_dir, job, text, per_day)
        run.write_model(
            world.observed_fleet if name == "observed" else world.fleet, world.task_classes
        )
        health = [r for w in windows for r in world.health_in(w)]  # type: ignore[arg-type]
        run.write_table(
            "health",
            sorted(health, key=lambda r: (r["day"], r["robot_id"], r["component"], r["metric"])),
        )
        run.finalize()


@pytest.fixture(scope="module")
def world() -> WorldV1:
    made = generate(recipes()["D2"], SOFTWARE, TEXTS, example_curves())
    assert made.spec.world_id == WORLD
    return made


@pytest.fixture(scope="module")
def world_root(world: WorldV1, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("worlds") / WORLD
    paths = WorldPaths(root, WORLD)
    write_world(world, paths.files)
    for name in RUN_WINDOWS:
        write_run(paths.run(name), world, name)
    steps = {
        s: {"done": True, "attempts": [{"seconds": 2.5}]} for s in ("files", "observed", "later")
    }
    paths.record.write_text(
        json.dumps({"world_id": WORLD, "ready": True, "steps": steps, "summary": SUMMARY})
    )
    return root


def progress_fixture() -> Progress:
    return Progress([], [], [], None)


def make(
    out: Path,
    worlds: list[Path],
    runs: list[Path] | None = None,
    now: datetime = NOW,
    manifest: Manifest | None = None,
    registry: WorldRegistry = REGISTRY,
    **options: Any,
) -> dict[str, Path]:
    pages = build(
        runs or [],
        out,
        Options(sample=0, **options),
        registry,
        manifest or Manifest(None),
        progress_fixture(),
        now=now,
        world_roots=worlds,
    )
    return {("world " if p.world else "") + p.key: p.directory for p in pages}


@pytest.fixture(scope="module")
def built(world_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The world page without and with the truth, built once for the tests that read them."""
    base = tmp_path_factory.mktemp("built")
    plain = make(base / "plain", [world_root])
    truth = make(base / "truth", [world_root], truth=True)
    return {
        "plain": plain[f"world {WORLD}"],
        "truth": truth[f"world {WORLD}"],
        "progress": plain["progress"],
        "truth_out": base / "truth",
    }


def section(markdown: str, heading: str, skip: int = 0) -> list[list[str]]:
    """The rows of a table after a heading (the first, or the one after `skip` others)."""
    lines = markdown.splitlines()
    start = lines.index(heading)
    tables: list[list[list[str]]] = [[]]
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            tables[-1].append([c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]])
        elif tables[-1]:
            tables.append([])
    rows = tables[skip]
    return [r for r in rows[1:] if not set("".join(r)) <= {"-"}]


def rate_text(outcomes: list[bool]) -> str:
    low, high = wilson(sum(outcomes), len(outcomes))
    rate = sum(outcomes) / len(outcomes)
    return f"{100 * rate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def episodes_of(root: Path) -> list[dict[str, Any]]:
    out = []
    for name in RUN_WINDOWS:
        run = WorldPaths(root, WORLD).run(name)
        rows: list[dict[str, Any]] = pq.read_table(run / "log" / "episodes.parquet").to_pylist()
        out += [{**r, "run": name} for r in rows]
    return out


def spec_of(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((root / "files" / "spec.json").read_text())
    return data


def window_of(spec: dict[str, Any], day: int) -> str:
    name: str = next(w for w, r in spec["windows"].items() if r["first"] <= day <= r["last"])
    return name


# ---------------------------------------------------------------------------- tests


def test_the_world_page_builds_with_its_figures(built: dict[str, Path]) -> None:
    page = built["plain"]
    text = (page / "index.md").read_text()
    assert page.name == WORLD and page.parent.name == "worlds"
    assert text.startswith(f"# Dashboard: world {WORLD}\n")
    referenced = set(re.findall(r"\]\(([^)]+)\)", text))
    assert referenced == {p.name for p in page.iterdir() if p.name != "index.md"}
    assert len([f for f in referenced if f.startswith("health-")]) == 6  # one per metric
    for name in referenced:
        image = imageio.imread(page / name)
        assert image.ndim == 3 and image.shape[0] > 200 and image.shape[1] > 600
    assert "- **runner**: not ready" in text  # fixture's legacy flag has no export receipt
    assert "- **shape**: 3 sites, 4 robots per site, 10 days, 14 episodes per robot and day" in text


def test_the_tables_are_the_runs_numbers(world_root: Path, built: dict[str, Path]) -> None:
    text = (built["plain"] / "index.md").read_text()
    spec = spec_of(world_root)
    episodes = episodes_of(world_root)
    everything = [e["outcome"] == "success" for e in episodes]
    assert f"- **success [95%]**: {rate_text(everything)}, {sum(everything)} of 120" in text
    rows = section(text, "### By window")
    assert [r[0] for r in rows] == ["observed", "experiment", "scoring"]
    for row in rows:
        outcomes = [
            e["outcome"] == "success" for e in episodes if window_of(spec, e["day"]) == row[0]
        ]
        assert row[2] == ("observed" if row[0] == "observed" else "later")
        assert row[4:] == [str(len(outcomes)), str(sum(outcomes)), rate_text(outcomes)]
    for heading, key in (
        ("### By site", "site_id"),
        ("### By robot", "robot_id"),
        ("### By day", "day"),
    ):
        rows = section(text, heading)
        values = sorted({e[key] for e in episodes}, key=lambda v: (isinstance(v, str), v))
        assert [r[0] for r in rows] == [str(v) for v in values], heading
        for row, value in zip(rows, values, strict=True):
            outcomes = [e["outcome"] == "success" for e in episodes if e[key] == value]
            numbers = row[2:5] if key == "robot_id" or key == "day" else row[1:4]
            assert numbers == [str(len(outcomes)), str(sum(outcomes)), rate_text(outcomes)], value
    site_day = section(text, "### By site and day")
    s1 = [e for e in episodes if e["site_id"] == "s1" and e["day"] == 0]
    assert site_day[0][1] == f"{sum(e['outcome'] == 'success' for e in s1)}/{len(s1)}"
    # the records: the observed run's, then what the later run adds after the observed window
    observed = WorldPaths(world_root, WORLD).observed
    changes = pq.read_table(observed / "log" / "change_log.parquet").to_pylist()
    assert len(section(text, "## Change log")) == len(changes) and changes
    later = pq.read_table(WorldPaths(world_root, WORLD).later / "log" / "maintenance.parquet")
    added = [r for r in later.to_pylist() if r["day"] > spec["windows"]["observed"]["last"]]
    assert f"they add {len(added)} entries after day 5" in text
    assert len(section(text, "## Maintenance log", skip=1)) == len(added)
    # events per kind and window, both runs
    kinds: dict[str, int] = {}
    for name in RUN_WINDOWS:
        events = pq.read_table(WorldPaths(world_root, WORLD).run(name) / "log" / "events.parquet")
        for r in events.to_pylist():
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    rows = section(text, "## Events")
    assert {r[0]: int(r[4]) for r in rows} == kinds


def test_the_health_comes_from_each_runs_days(world_root: Path, built: dict[str, Path]) -> None:
    text = (built["plain"] / "index.md").read_text()
    assert "days 0 to 5 from the observed run's `log/health.parquet`" in text
    assert "days 6 to 9 from the later run's `log/health.parquet`" in text
    rows = section(text, "### gripper: grip_force")
    readings = [
        json.loads(line)
        for line in (world_root / "files" / "health.jsonl").read_text().splitlines()
        if '"grip_force"' in line
    ]
    r01 = sorted((r["day"], r["value"]) for r in readings if r["robot_id"] == "r01")
    assert rows[0][:3] == ["r01", "s1", "10"]
    assert rows[0][3] == f"{r01[0][1]:.4g} (day 0)" and rows[0][4] == f"{r01[-1][1]:.4g} (day 9)"


def test_no_truth_without_the_flag(world_root: Path, built: dict[str, Path]) -> None:
    spec = spec_of(world_root)
    generator = json.loads((world_root / "files" / "generator.json").read_text())
    kinds = {f["kind"] for f in spec["fault_schedule"]}
    for key in ("plain", "progress"):
        page = built[key]
        text = (page / "index.md").read_text()
        names = " ".join(p.name for p in page.iterdir())
        for words in (text, names):
            assert not re.search(r"\bf\d{2}\b", words)  # no fault id
            for term in (*kinds, *ROLE_WORDS, "(truth)", "## Truth", "*truth*", "detectab"):
                assert term not in words, (key, term)
            for role in set(generator["roles"].values()):
                assert f"| {role} |" not in words, (key, role)
    world_text = (built["plain"] / "index.md").read_text()
    assert "gate" not in world_text.lower() and "expected" not in world_text


def test_the_truth_with_the_flag(world_root: Path, built: dict[str, Path]) -> None:
    text = (built["truth"] / "index.md").read_text()
    spec = spec_of(world_root)
    generator = json.loads((world_root / "files" / "generator.json").read_text())
    detect = json.loads((world_root / "files" / "detectability.json").read_text())
    assert "> **Truth: the hidden side.**" in text
    schedule = section(text, "### Fault schedule (truth)")
    assert [r[0] for r in schedule] == [f["fault_id"] for f in spec["fault_schedule"]]
    for row, fault in zip(schedule, spec["fault_schedule"], strict=True):
        assert row[1] == generator["roles"][fault["fault_id"]] and row[2] == fault["kind"]
        assert row[3] == fault["target"]["scope"] and int(row[6]) == fault["schedule"]["onset_day"]
    drift = next(r for r in schedule if r[1] == "drift")
    from fleetlab.contracts.faults import FaultSpec

    fault = FaultSpec.model_validate(
        next(f for f in spec["fault_schedule"] if f["fault_id"] == drift[0])
    )
    first = fault.schedule.onset_day
    assert drift[5].startswith(
        f"{day_level(fault, first):g} on day {first} to {day_level(fault, 9):g} on day 9"
    )
    assert (
        "- **interactions**: "
        + ", ".join(f"{a} with {b}" for a, b in spec["patterns"]["interactions"])
        in text
    )
    causes = section(text, "### Detectability (truth)")
    assert [r[0] for r in causes] == [c["unit"] for c in detect["causes"]]
    counted = sum(c["counts_for_gate"] for c in detect["causes"])
    assert f"**{counted} causes count for gate 3's sizing rule**" in text
    assert [r[9] for r in causes] == [
        "yes" if c["counts_for_gate"] else "no" for c in detect["causes"]
    ]
    windows = section(text, "### Success per window: expected and measured (truth)")
    episodes = episodes_of(world_root)
    for row in windows:
        want = detect["window_success"][row[0]]
        assert row[2:5] == [
            str(want["episodes"]),
            f"{100 * want['logged']:.1f}%",
            f"{100 * want['true']:.1f}%",
        ]
        outcomes = [
            e["outcome"] == "success" for e in episodes if window_of(spec, e["day"]) == row[0]
        ]
        assert row[6] == rate_text(outcomes)
    marks = section(
        text, "### The runner's summary: measured against expected, gate-3 marks (truth)"
    )
    assert marks == [["gate3.band", "no"], ["gate3.sizing", "yes"]]


def test_the_per_fault_comparison_counts_the_runs_episodes(
    world: WorldV1, world_root: Path, built: dict[str, Path]
) -> None:
    """Measured from the runs; expected with the generator's own context, which the page
    rebuilds from the world's files."""
    text = (built["truth"] / "index.md").read_text()
    rows = section(
        text, "### Per fault: logged success of its episodes against the expectation (truth)"
    )
    ctx = context(world.composition, curves.load())
    logged = {e["episode_id"]: e["outcome"] == "success" for e in episodes_of(world_root)}
    carried: dict[tuple[str, str], list[tuple[bool, float]]] = {}
    for name in RUN_WINDOWS:
        for identity in read_identities(WorldPaths(world_root, WORLD).run(name)):
            p = probability(identity.faults, identity.suite, ctx, logged=True)
            labels = [f.fault_id for f in identity.faults] or ["no fault"]
            for label in [*labels, "all"]:
                carried.setdefault((label, name), []).append((logged[identity.episode_id], p))
    assert {(r[0], r[3]) for r in rows} == set(carried)
    for row in rows:
        values = carried[(row[0], row[3])]
        outcomes = [s for s, _ in values]
        expected = sum(p for _, p in values) / len(values)
        measured = sum(outcomes) / len(outcomes)
        assert row[4:] == [
            str(len(outcomes)),
            rate_text(outcomes),
            f"{100 * expected:.1f}%",
            f"{100 * (measured - expected):+.1f}",
        ], row
    assert "**not an effect estimate**" in text


def test_the_truth_is_refused_where_it_does_not_belong(world_root: Path, tmp_path: Path) -> None:
    unregistered = WorldRegistry({})
    with pytest.raises(BuildRefused, match=re.escape("not in dashboard/worlds.json")):
        make(tmp_path / "a", [world_root], registry=unregistered)
    with pytest.raises(BuildRefused, match="development worlds only"):
        make(
            tmp_path / "a", [world_root], registry=unregistered, truth=True, allow_unregistered=True
        )
    assert not (tmp_path / "a").exists()  # refused before anything was written
    sealed = WorldRegistry({WORLD: "test"})
    with pytest.raises(BuildRefused, match="before the test"):
        make(tmp_path / "c", [world_root], registry=sealed)
    with pytest.raises(BuildRefused, match="development worlds only"):
        make(tmp_path / "c", [world_root], registry=sealed, allow_sealed=True, truth=True)
    with pytest.raises(BuildRefused, match="two world roots hold world"):
        make(tmp_path / "d", [world_root, world_root])
    with pytest.raises(FileNotFoundError, match="not a world root"):
        make(tmp_path / "e", [tmp_path])


def test_the_progress_page_links_the_world_page(
    world_root: Path, built: dict[str, Path], tmp_path: Path
) -> None:
    rows = section((built["progress"] / "index.md").read_text(), "## World pages")
    assert rows[0][:3] == [WORLD, "development", "not ready"] and rows[0][-1] == "not published yet"
    manifest = Manifest(None)
    url = "https://hub.example/a/idw"
    manifest.worlds[WORLD] = PageEntry(id="idw", url=url, root=str(world_root.resolve()))
    # a world page published before is listed without being built again
    pages = make(tmp_path / "a", [], manifest=manifest)
    assert sorted(pages) == ["progress"]
    rows = section((pages["progress"] / "index.md").read_text(), "## World pages")
    assert rows[0][0] == f"[{WORLD}]({url})" and rows[0][4] == "120" and rows[0][-1] == ""
    manifest.worlds[WORLD] = PageEntry(id="idw", url=url, root="/elsewhere")
    with pytest.raises(BuildRefused, match="published from /elsewhere"):
        make(tmp_path / "b", [world_root], manifest=manifest)
    manifest.worlds[WORLD] = PageEntry(id="idw", url=url, root="/gone")
    rows = section(
        (make(tmp_path / "c", [], manifest=manifest)["progress"] / "index.md").read_text(),
        "## World pages",
    )
    assert rows[0][0] == f"[{WORLD}]({url})" and rows[0][-1] == "not readable: FileNotFoundError"


def test_two_builds_differ_only_in_the_generated_line(
    world_root: Path, built: dict[str, Path], tmp_path: Path
) -> None:
    first = built["truth_out"]
    make(tmp_path / "two", [world_root], truth=True, now=datetime(2027, 1, 1, tzinfo=UTC))
    one = sorted(p.relative_to(first) for p in first.rglob("*"))
    two = sorted(p.relative_to(tmp_path / "two") for p in (tmp_path / "two").rglob("*"))
    assert one == two
    pngs = 0
    for relative in one:
        a, b = first / relative, tmp_path / "two" / relative
        if a.is_dir():
            continue
        if a.name == "index.md":
            lines_a, lines_b = a.read_text().splitlines(), b.read_text().splitlines()
            differ = [i for i, (x, y) in enumerate(zip(lines_a, lines_b, strict=True)) if x != y]
            assert len(differ) == 1 and lines_a[differ[0]].startswith(GENERATED_PREFIX)
        else:
            assert a.read_bytes() == b.read_bytes(), relative
            pngs += a.suffix == ".png"
    assert pngs == 7


def test_the_command_line_on_a_world_whose_later_run_has_not_started(
    world_root: Path, tmp_path: Path
) -> None:
    """Also: an unregistered world built with --allow-unregistered-worlds, and the page's id
    from the manifest written as its front matter."""
    root = tmp_path / WORLD
    shutil.copytree(world_root / "files", root / "files")
    shutil.copytree(WorldPaths(world_root, WORLD).observed, WorldPaths(root, WORLD).observed)
    registry = tmp_path / "worlds.json"
    registry.write_text("{}")
    ids = tmp_path / "ids.json"
    url = "https://hub.example/a/idw"
    ids.write_text(json.dumps({"worlds": {WORLD: {"id": "idw", "url": url, "root": str(root)}}}))
    out = tmp_path / "out"
    common = ["--no-beads", "--no-queue", "--worlds", str(registry), "--manifest", str(ids)]
    assert main(["build", "--world-roots", str(root), "--out", str(out), *common]) == 1
    assert not out.exists()  # refused: not registered
    command = ["build", "--world-roots", str(root), "--out", str(out), *common]
    assert main([*command, "--allow-unregistered-worlds"]) == 0
    text = (out / "worlds" / WORLD / "index.md").read_text()
    assert text.startswith(f"---\nid: idw\n---\n# Dashboard: world {WORLD}\n")
    assert f"- **world**: {WORLD} (unregistered)" in text
    assert f"| {WORLD}-later | experiment, scoring | 6 to 9 | not started |" in text
    assert "No `runner.json` yet." in text and "- **runner**: not ready" in text
    assert "days 6 to 9" not in text  # no health of days that did not run
    assert [r[0] for r in section(text, "### By window")] == ["observed", "experiment", "scoring"]
    assert [r[4] for r in section(text, "### By window")] == ["72", "0", "0"]
    progress = (out / "progress" / "index.md").read_text()
    assert section(progress, "## World pages")[0][:4] == [
        f"[{WORLD}]({url})",
        "unregistered",
        "not ready",
        f"{WORLD}-observed, {WORLD}-later",
    ]
    assert main([*command, "--allow-unregistered-worlds", "--truth"]) == 1
    nothing = ["build", "--world-roots", str(tmp_path / "out"), "--out", str(tmp_path / "x")]
    assert main([*nothing, *common]) == 1  # not a world root


def test_runs_of_another_world_do_not_stop_the_page(world_root: Path, tmp_path: Path) -> None:
    """The log's toy run (world w01) where the observed run belongs: shown as it is."""
    root = tmp_path / WORLD
    shutil.copytree(world_root / "files", root / "files")
    toy = [j.model_copy(update={"record_video": False}) for j in run_jobs()]
    write_toy_run(WorldPaths(root, WORLD).observed, toy)
    registry = WorldRegistry({WORLD: "development", "w01": "probe"})
    page = make(tmp_path / "out", [root], registry=registry, truth=True)[f"world {WORLD}"]
    text = (page / "index.md").read_text()
    episodes = pq.read_table(WorldPaths(root, WORLD).observed / "log" / "episodes.parquet")
    outcomes = [e["outcome"] == "success" for e in episodes.to_pylist()]
    assert f"- **success [95%]**: {rate_text(outcomes)}, {sum(outcomes)} of 10" in text
    assert f"the runs' episodes belong to world w01, not only to {WORLD}" in text
    assert f"Note: 10 episodes of the runs are left out: no identity of world {WORLD}" in text
    with pytest.raises(BuildRefused, match=re.escape("world w01 is not in dashboard/worlds.json")):
        make(tmp_path / "refused", [root])


FAKE_HUB = """#!{python}
import json, sys, pathlib
args = sys.argv[1:]
with pathlib.Path({log!r}).open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "config":
    print("url: https://hub.example (test)")
elif args[0] == "check":
    print(json.dumps({{"findings": [], "file_count": 1}}))
elif args[0] == "publish":
    given = args[args.index("--id") + 1] if "--id" in args else None
    new = given or "id-" + args[args.index("--title") + 1].split()[-1]
    print(json.dumps({{"id": new, "url": "/a/" + new}}))
"""


def test_publish_links_the_runs_the_world_and_the_progress_page(
    world_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "hub.log"
    hub = bin_dir / "hub"
    hub.write_text(FAKE_HUB.format(python=sys.executable, log=str(log)))
    hub.chmod(hub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    path = tmp_path / "ids.json"
    runs = [WorldPaths(world_root, WORLD).run(n) for n in RUN_WINDOWS]
    manifest = Manifest.load(path)
    pages = build(runs, tmp_path / "out", Options(sample=0, truth=True), REGISTRY, manifest,
                  progress_fixture(), world_roots=[world_root])  # fmt: skip
    publish(pages, tmp_path / "out", REGISTRY, manifest, progress_fixture(), log=lambda _: None)
    data = json.loads(path.read_text())
    assert data["worlds"][WORLD] == {
        "id": f"id-{WORLD}",
        "url": f"https://hub.example/a/id-{WORLD}",
        "world": WORLD,
        "root": str(world_root.resolve()),
    }
    text = (tmp_path / "out" / "worlds" / WORLD / "index.md").read_text()
    for name in RUN_WINDOWS:  # rebuilt with the run pages' new links before it was published
        assert f"[{WORLD}-{name}](https://hub.example/a/id-{WORLD}-{name})" in text
    progress = (tmp_path / "out" / "progress" / "index.md").read_text()
    assert f"[{WORLD}](https://hub.example/a/id-{WORLD})" in progress
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    titles = [c[c.index("--title") + 1] for c in calls if c[0] == "publish"]
    assert titles == [
        f"Dashboard: run {WORLD}-observed",
        f"Dashboard: run {WORLD}-later",
        f"Dashboard: world {WORLD}",
        "Dashboard: progress",
    ]
    world_call = next(
        c for c in calls if c[0] == "publish" and "world" in c[c.index("--title") + 1]
    )
    assert "truth" in [world_call[i + 1] for i, a in enumerate(world_call) if a == "--tag"]
