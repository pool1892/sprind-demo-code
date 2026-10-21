"""The dashboard on the log's toy run and export: bundles, tables, figures, strips, truth, repeat.

src/fleetlab/dashboard. Nothing here calls bd or the hub: the progress
sources are given, and publishing runs against a stand-in `hub` script.
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
from log_toy import (
    FRAME_SIZE,
    TASK_CLASSES,
    frame_value,
    identities,
    write_toy_run,
)

from fleetlab.contracts.env import Camera
from fleetlab.contracts.faults import FaultSpec, Schedule, Target
from fleetlab.contracts.rollout import Job
from fleetlab.contracts.vocabulary import Component
from fleetlab.dashboard.build import BuildRefused, Options, build, publish
from fleetlab.dashboard.data import Rate, WorldRegistry
from fleetlab.dashboard.figures import STRIP_GAP
from fleetlab.dashboard.hub import Manifest, PageEntry
from fleetlab.dashboard.pages import GENERATED_PREFIX
from fleetlab.dashboard.sources import Bead, Progress, QueueJob, read_beads, read_queue
from fleetlab.dose_response import wilson
from fleetlab.log import export_runs

REGISTRY = WorldRegistry({"w01": "probe", "w90": "calibration", "w07": "test"})
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
FAULT_TERMS = ("toy_camera_wobble", "toy_command_lag", "(truth)", "## Truth")


def progress_fixture() -> Progress:
    beads = [
        Bead("item-g2", "Gate 2: design freeze", "open", 1, "decision", ("human",)),
        Bead("item-g1", "Gate 1: sizing", "closed", 1, "decision", ("human",), "2026-10-07"),
        Bead("item-g10", "Gate 10: far away", "open", 1, "decision", ("human",)),
        Bead("item-a", "Build the engine", "in_progress", 1, "task"),
        Bead("item-b", "Write | the dashboard", "open", 2, "task"),
        Bead("item-c", "Done already", "closed", 0, "task"),
    ]
    human = [Bead("item-h", "Decide the hub's video", "open", 2, "task", ("human",))]
    queue = [
        QueueJob(10, "running", "uv run python -m fleetlab.engine run"),
        QueueJob(12, "waiting", "uv run python spikes/x.py"),
    ]
    return Progress(beads, human, queue, ("nightly.md", "# Nightly\n\nAll identical.\n"))


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("dashboard")
    fleet = write_toy_run(base / "run")
    export_runs([base / "run"], base / "export", fleet=fleet, task_classes=TASK_CLASSES)
    write_toy_run(base / "calibration", calibration_jobs())
    # as the calibration runs it: one healthy run, one run per fault kind
    write_toy_run(base / "healthy", calibration_jobs()[:4])
    write_toy_run(base / "wobble", calibration_jobs()[4:])
    names = ("run", "export", "calibration", "healthy", "wobble")
    return {name: base / name for name in names}


def calibration_jobs() -> list[Job]:
    """The toy's ten episodes in world w90: four healthy, three at level 3, three at level 6."""
    jobs = []
    for number, identity in enumerate(identities(), start=1):
        faults = []
        if number > 4:
            level = 3.0 if number <= 7 else 6.0
            faults = [
                FaultSpec(
                    fault_id="f01",
                    kind="toy_camera_wobble",
                    params={"degrees": level},
                    level=level,
                    target=Target(scope=identity.robot_id, component=Component.scene_camera),
                    schedule=Schedule(onset_day=0),
                )
            ]
        changed = identity.model_copy(update={"world_id": "w90", "faults": faults})
        jobs.append(Job(identity=changed, record_video=number % 2 == 1))
    return jobs


def make(
    toy: dict[str, Path],
    out: Path,
    names: tuple[str, ...] = ("run", "export"),
    now: datetime = NOW,
    manifest: Manifest | None = None,
    registry: WorldRegistry = REGISTRY,
    **options: Any,
) -> dict[str, Path]:
    options.setdefault("sample", 0)  # strips and videos cost an ffmpeg call each
    pages = build(
        [toy[n] for n in names],
        out,
        Options(**options),
        registry,
        manifest or Manifest(None),
        progress_fixture(),
        now=now,
    )
    return {p.key: p.directory for p in pages}


def section(markdown: str, heading: str) -> list[list[str]]:
    """The rows of the first table after a heading, as cells."""
    lines = markdown.splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        if line.startswith("|"):
            rows.append([c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]])
        elif rows:
            break
    return [r for r in rows[1:] if not set("".join(r)) <= {"-"}]


def episodes_of(run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(run / "log" / "episodes.parquet").to_pylist()
    return rows


def rate_text(outcomes: list[bool]) -> str:
    low, high = wilson(sum(outcomes), len(outcomes))
    rate = sum(outcomes) / len(outcomes)
    return f"{100 * rate:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


# ---------------------------------------------------------------------------- tests


def test_the_bundle_builds_with_its_figures(toy: dict[str, Path], tmp_path: Path) -> None:
    pages = make(toy, tmp_path / "out", sample=2)
    assert sorted(pages) == ["export", "progress", "run"]
    for key in ("run", "export"):
        page = pages[key]
        text = (page / "index.md").read_text()
        assert text.startswith(f"# Dashboard: run {key}\n")
        for figure in ("success-by-robot.png", "success-by-day.png"):
            assert f"]({figure})" in text
            image = imageio.imread(page / figure)
            assert image.ndim == 3 and image.shape[0] > 100 and image.shape[1] > 100
        # every file the page refers to is in the bundle, and nothing else is
        referenced = set(re.findall(r"\]\(([^)]+)\)", text))
        assert referenced == {p.name for p in page.iterdir() if p.name != "index.md"}
    assert (pages["progress"] / "index.md").is_file()
    assert (tmp_path / "out" / ".dashboard").is_file()


def test_the_tables_match_the_data(toy: dict[str, Path], tmp_path: Path) -> None:
    text = (make(toy, tmp_path / "out")["run"] / "index.md").read_text()
    episodes = episodes_of(toy["run"])
    for heading, key in (
        ("### By robot", "robot_id"),
        ("### By day", "day"),
        ("### By site", "site_id"),
        ("### By task", "task_text"),
    ):
        values = sorted({e[key] for e in episodes}, key=str)
        rows = section(text, heading)
        assert [r[0] for r in rows] == [str(v) for v in values], heading
        for row, value in zip(rows, values, strict=True):
            outcomes = [e["outcome"] == "success" for e in episodes if e[key] == value]
            assert row[-3:] == [str(len(outcomes)), str(sum(outcomes)), rate_text(outcomes)]
    everything = [e["outcome"] == "success" for e in episodes]
    assert f"- **success [95%]**: {rate_text(everything)}, {sum(everything)} of 10" in text
    assert "- **episodes**: 10" in text


def test_the_strip_is_made_from_the_right_frames(toy: dict[str, Path], tmp_path: Path) -> None:
    page = make(toy, tmp_path / "out", frames=3, sample=6)["run"]
    text = (page / "index.md").read_text()
    cards = re.findall(
        r"### (e\d{6}):.*?\n\n.*?\n\n!\[.*?\]\((.*?)\)\n\nFrames at steps ([\d, ]+);",
        text,
        flags=re.S,
    )
    assert [c[0] for c in cards] == ["e000001", "e000003", "e000005", "e000007", "e000009"]
    for episode_id, strip_name, step_text in cards:
        number = int(episode_id[1:])
        last = 3 + number % 3  # the toy's last step; frames at even steps and the last
        logged = sorted({*range(0, last + 1, 2), last})
        steps = [int(s) for s in step_text.split(", ")]
        k = min(3, len(logged))
        assert steps == [logged[round(i * (len(logged) - 1) / (k - 1))] for i in range(k)]
        assert steps[0] == 0 and steps[-1] == last
        strip = imageio.imread(page / strip_name)
        tile = FRAME_SIZE + STRIP_GAP
        assert strip.shape[:2] == (2 * tile - STRIP_GAP, len(steps) * tile - STRIP_GAP)
        for r, camera in enumerate((Camera.scene, Camera.wrist)):
            for c, step in enumerate(steps):
                pixels = strip[r * tile : r * tile + FRAME_SIZE, c * tile : c * tile + FRAME_SIZE]
                assert abs(float(pixels.mean()) - frame_value(number, step, camera)) < 4
        assert np.all(strip[FRAME_SIZE : FRAME_SIZE + STRIP_GAP] == 255)  # the gap is white
        video = page / f"{episode_id}.scene.mp4"
        assert video.read_bytes() == (toy["run"] / "log" / "video" / video.name).read_bytes()


def test_hidden_columns_appear_only_with_the_truth_flag(
    toy: dict[str, Path], tmp_path: Path
) -> None:
    plain = make(toy, tmp_path / "plain", names=("run",), sample=2)
    text = (plain["run"] / "index.md").read_text()
    for term in (*FAULT_TERMS, "f01", "f02", "true verdict"):
        assert term not in text, term
    truth = make(toy, tmp_path / "truth", names=("run",), truth=True, sample=2)
    text = (truth["run"] / "index.md").read_text()
    for term in (*FAULT_TERMS, "true verdict"):
        assert term in text, term
    robots = section(text, "### By robot")
    assert robots[0][-1] == "none"  # r01 has no fault
    assert robots[1][-1] == "f01 toy_camera_wobble at 0.5; f02 toy_command_lag at 1"
    faults = section(text, "### Active faults (truth)")
    rows = pq.read_table(toy["run"] / "truth" / "faults_active.parquet").to_pylist()
    for row in faults:
        mine = [r for r in rows if r["fault_id"] == row[0]]
        assert row[1] == mine[0]["kind"] and int(row[5]) == len(mine)
    verdicts = pq.read_table(toy["run"] / "truth" / "verdicts.parquet").to_pylist()
    true = {r["episode_id"]: r["true_success"] for r in verdicts}
    jobs = pq.read_table(toy["run"] / "truth" / "job_outcomes.parquet").to_pylist()
    gains = section(text, "### Oracle: the gain of removing each fault (truth)")
    for row in gains:
        fault = row[0]
        active = {r["episode_id"] for r in rows if r["fault_id"] == fault}
        alone = [  # replays from the reset with exactly this fault removed
            j
            for j in jobs
            if j["mode"] == "replay"
            and j["from_snapshot_step"] is None
            and json.loads(j["removed"]) == [fault]
            and j["episode_id"] in active
        ]
        rescued = sum(not true[j["episode_id"]] and j["success"] for j in alone)
        broken = sum(true[j["episode_id"]] and not j["success"] for j in alone)
        assert row[1:] == [
            str(len(active)),
            str(len(alone)),
            str(rescued),
            str(broken),
            str(rescued - broken),
        ], fault
    assert [r[0] for r in gains] == ["f01", "f02"] and gains[0][2] == "1"  # e000002's replay
    agree = section(text, "### Logged outcome against the true verdict (truth)")
    assert sum(int(c) for r in agree for c in r[1:]) == len(verdicts)


def test_the_truth_is_refused_where_it_does_not_belong(
    toy: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(BuildRefused, match="export"):
        make(toy, tmp_path / "a", names=("export",), truth=True)
    unregistered = WorldRegistry({})
    with pytest.raises(BuildRefused, match=re.escape("not in dashboard/worlds.json")):
        make(toy, tmp_path / "b", names=("run",), registry=unregistered)
    with pytest.raises(BuildRefused, match="development worlds only"):
        make(
            toy,
            tmp_path / "b",
            names=("run",),
            truth=True,
            registry=unregistered,
            allow_unregistered=True,
        )
    make(toy, tmp_path / "b", names=("run",), registry=unregistered, allow_unregistered=True)
    sealed = WorldRegistry({"w01": "test"})
    with pytest.raises(BuildRefused, match="before the test"):
        make(toy, tmp_path / "c", names=("run",), registry=sealed)
    pages = make(toy, tmp_path / "d", names=("run",), registry=sealed, allow_sealed=True)
    assert "toy_camera_wobble" not in (pages["run"] / "index.md").read_text()
    assert not (tmp_path / "a").exists()  # refused before anything was written


def test_two_builds_differ_only_in_the_generated_line(toy: dict[str, Path], tmp_path: Path) -> None:
    names = ("run", "export", "calibration")
    make(toy, tmp_path / "one", names=names, sample=6)
    make(toy, tmp_path / "two", names=names, sample=6, now=datetime(2027, 1, 1, tzinfo=UTC))
    one = sorted(p.relative_to(tmp_path / "one") for p in (tmp_path / "one").rglob("*"))
    two = sorted(p.relative_to(tmp_path / "two") for p in (tmp_path / "two").rglob("*"))
    assert one == two
    pngs = 0
    for relative in one:
        a, b = tmp_path / "one" / relative, tmp_path / "two" / relative
        if a.is_dir():
            continue
        if a.name == "index.md":
            lines_a, lines_b = a.read_text().splitlines(), b.read_text().splitlines()
            assert len(lines_a) == len(lines_b)
            differ = [i for i, (x, y) in enumerate(zip(lines_a, lines_b, strict=True)) if x != y]
            assert len(differ) == 1 and lines_a[differ[0]].startswith(GENERATED_PREFIX)
        else:
            assert a.read_bytes() == b.read_bytes(), relative
            pngs += a.suffix == ".png"
    assert pngs >= 10


def test_a_calibration_run_gets_its_dose_response(toy: dict[str, Path], tmp_path: Path) -> None:
    page = make(toy, tmp_path / "out", names=("calibration",))["calibration"]
    text = (page / "index.md").read_text()
    assert "](dose-response.png)" in text and (page / "dose-response.png").is_file()
    verdicts = {
        r["episode_id"]: r["true_success"]
        for r in pq.read_table(toy["calibration"] / "truth" / "verdicts.parquet").to_pylist()
    }
    groups = {"healthy": [1, 2, 3, 4], "3": [5, 6, 7], "6": [8, 9, 10]}
    rows = section(text, "## Dose-response (truth)")
    assert [(r[0], r[1]) for r in rows] == [
        ("healthy", ""),
        ("toy_camera_wobble", "3"),
        ("toy_camera_wobble", "6"),
    ]
    for row, numbers in zip(rows, groups.values(), strict=True):
        outcomes = [verdicts[f"e{n:06d}"] for n in numbers]
        assert row[2:] == [str(len(outcomes)), str(sum(outcomes)), rate_text(outcomes)]
    other = make(toy, tmp_path / "plain", names=("run",))["run"]
    assert "Dose-response" not in (other / "index.md").read_text()


def test_a_kind_run_gets_the_healthy_run_of_its_build(toy: dict[str, Path], tmp_path: Path) -> None:
    pages = make(toy, tmp_path / "out", names=("healthy", "wobble"))
    rows = section((pages["wobble"] / "index.md").read_text(), "## Dose-response (truth)")
    healthy = [
        r["true_success"]
        for r in pq.read_table(toy["healthy"] / "truth" / "verdicts.parquet").to_pylist()
    ]
    assert rows[0] == ["healthy (runs healthy)", "", "4", str(sum(healthy)), rate_text(healthy)]
    assert [r[1] for r in rows[1:]] == ["3", "6"]
    assert (pages["wobble"] / "dose-response.png").is_file()
    alone = make(toy, tmp_path / "alone", names=("wobble",))["wobble"]
    assert [
        r[0] for r in section((alone / "index.md").read_text(), "## Dose-response (truth)")
    ] == [
        "toy_camera_wobble",
        "toy_camera_wobble",
    ]


def test_the_progress_page(toy: dict[str, Path], tmp_path: Path) -> None:
    manifest = Manifest(None)
    manifest.runs["gone"] = _entry("idg", "/nonexistent/gone")
    text = (make(toy, tmp_path / "out", manifest=manifest)["progress"] / "index.md").read_text()
    runs = section(text, "## Runs")
    assert [r[0] for r in runs] == ["export", "[gone](https://hub.example/idg)", "run"]
    assert runs[2][1:6] == [
        "w01",
        "probe",
        "finalized",
        "10",
        rate_text([e["outcome"] == "success" for e in episodes_of(toy["run"])]),
    ]
    assert runs[1][3] == "not readable: FileNotFoundError"
    assert [r[0] for r in section(text, "## Gates")] == ["item-g1", "item-g2", "item-g10"]
    assert section(text, "## Gates")[0][2] == "2026-10-07"
    assert [r[0] for r in section(text, "## Waiting for a human")] == ["item-h"]
    assert [r[0] for r in section(text, "## Beads in progress")] == ["item-a"]
    assert section(text, "## Open beads") == [["item-b", "P2", "task", "Write \\| the dashboard"]]
    queue = section(text, "## Queue (tools/heavy-run)")
    assert [(r[0], r[1]) for r in queue] == [("running", "10"), ("waiting", "12")]
    assert "From `nightly.md`:\n\n### Nightly\n\nAll identical." in text


def _entry(artifact_id: str, run_dir: str) -> PageEntry:
    return PageEntry(id=artifact_id, url=f"https://hub.example/{artifact_id}", run_dir=run_dir)


def test_a_run_published_from_another_directory_is_refused(
    toy: dict[str, Path], tmp_path: Path
) -> None:
    manifest = Manifest(None)
    manifest.runs["run"] = _entry("id1", "/elsewhere/run")
    with pytest.raises(BuildRefused, match="published from /elsewhere/run"):
        make(toy, tmp_path / "out", names=("run",), manifest=manifest)


def test_a_foreign_output_directory_is_refused(toy: dict[str, Path], tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "notes.txt").write_text("mine")
    with pytest.raises(BuildRefused, match="not a dashboard build directory"):
        make(toy, tmp_path / "out", names=("run",))
    assert (tmp_path / "out" / "notes.txt").read_text() == "mine"


FAKE_HUB = """#!{python}
import json, sys, pathlib
log = pathlib.Path({log!r})
args = sys.argv[1:]
with log.open("a") as f:
    f.write(json.dumps(args) + "\\n")
if args[0] == "config":
    print("url: https://hub.example (test)")
elif args[0] == "check":
    print(json.dumps({{"findings": [], "file_count": 1}}))
elif args[0] == "publish":
    given = args[args.index("--id") + 1] if "--id" in args else None
    title = args[args.index("--title") + 1]
    new = given or "id-" + title.split()[-1]
    print(json.dumps({{"id": new, "url": "/a/" + new}}))
"""


def test_publish_keeps_ids_in_the_manifest(
    toy: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "hub.log"
    hub = bin_dir / "hub"
    hub.write_text(FAKE_HUB.format(python=sys.executable, log=str(log)))
    hub.chmod(hub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    path = tmp_path / "ids.json"
    for round_ in (1, 2):
        manifest = Manifest.load(path)
        progress = progress_fixture()
        pages = build([toy["run"]], tmp_path / "out", Options(), REGISTRY, manifest, progress)
        publish(pages, tmp_path / "out", REGISTRY, manifest, progress, log=lambda _: None)
        data = json.loads(path.read_text())
        assert data["runs"]["run"]["run_dir"] == str(toy["run"].resolve())
        run_url = data["runs"]["run"]["url"]
        assert run_url == "https://hub.example/a/id-run"  # made absolute with hub config
        progress_text = (tmp_path / "out" / "progress" / "index.md").read_text()
        assert f"[run]({run_url})" in progress_text
        if round_ == 2:
            run_text = (tmp_path / "out" / "runs" / "run" / "index.md").read_text()
            assert data["progress"]["url"] in run_text
            # the id the hub would write into the entry is there from the build on
            assert run_text.startswith("---\nid: id-run\n---\n# Dashboard: run run\n")
            assert progress_text.startswith("---\nid: id-progress\n---\n")
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    publishes = [c for c in calls if c[0] == "publish"]
    assert len(publishes) == 4 and len([c for c in calls if c[0] == "check"]) == 4
    assert all("--tag" in c and c[c.index("--tag") + 1] == "dashboard" for c in publishes)
    assert ["--id" in c for c in publishes] == [False, False, True, True]


def test_the_queue_is_read_from_proc(tmp_path: Path) -> None:
    def process(
        pid: int, argv: list[str], children: tuple[int, ...] = (), comm: str = "sh"
    ) -> None:
        d = tmp_path / str(pid)
        (d / "task" / str(pid)).mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "task" / str(pid) / "children").write_text(" ".join(map(str, children)))
        (d / "comm").write_text(comm + "\n")

    process(100, ["/bin/sh", "tools/heavy-run", "-m", "12G", "uv", "run", "a.py"], (101,))
    process(101, ["flock", "9"], comm="flock")
    process(200, ["/bin/sh", "/repo/tools/heavy-run", "uv", "run", "b.py"], (201,))
    process(201, ["systemd-run"], comm="systemd-run")
    process(300, ["/bin/sh", "tools/heavy-run", "--light", "pytest"])
    process(400, ["python", "tools/heavy-run"])
    assert read_queue(tmp_path) == [
        QueueJob(200, "running", "uv run b.py"),
        QueueJob(100, "waiting", "uv run a.py"),
    ]


def test_beads_unreadable_are_said_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    result = read_beads(tmp_path)
    assert isinstance(result, str) and "bd" in result


def test_a_rate_without_episodes() -> None:
    assert Rate(0, 0).text() == "no episodes"
    assert Rate(4, 3).text() == rate_text([True, True, True, False])


def damaged_copy(toy: dict[str, Path], tmp_path: Path, *remove: str) -> dict[str, Path]:
    """A copy of the toy run with some of its files removed."""
    copy = tmp_path / "damaged" / "run"
    shutil.copytree(toy["run"], copy)
    for relative in remove:
        (copy / relative).unlink()
    return {"run": copy}


def test_a_missing_video_costs_its_strip_row_not_the_page(
    toy: dict[str, Path], tmp_path: Path
) -> None:
    runs = damaged_copy(toy, tmp_path, "log/video/e000001.wrist.mp4", "log/video/e000003.scene.mp4")
    page = make(runs, tmp_path / "out", names=("run",), sample=6, truth=True)["run"]
    text = (page / "index.md").read_text()
    assert "Frames at steps 0, 2, 4; scene camera." in text  # e000001: the scene row only
    assert "Not shown: the wrist video file is missing." in text
    assert "Not shown: the scene video file is missing." in text
    assert not (page / "e000003.scene.mp4").exists() and (page / "e000003.strip.png").is_file()
    assert imageio.imread(page / "e000001.strip.png").shape[0] == FRAME_SIZE


def test_the_truth_of_a_run_without_replays(toy: dict[str, Path], tmp_path: Path) -> None:
    runs = damaged_copy(toy, tmp_path, "truth/replays.parquet", "log/costs.parquet")
    pages = make(runs, tmp_path / "out", names=("run",), truth=True)
    text = (pages["run"] / "index.md").read_text()
    assert "Not computed: missing truth/replays.parquet." in text
    assert "### Active faults (truth)" in text
    # 13 jobs: ten runs, two replays, one continuation
    state = "complete (13 jobs, all merged), but without truth/replays.parquet, log/costs.parquet"
    assert state in text and state in (pages["progress"] / "index.md").read_text()


def test_a_render_transient_is_shown_in_amber_and_counted(
    toy: dict[str, Path], tmp_path: Path
) -> None:
    run = tmp_path / "transient"
    shutil.copytree(toy["run"], run)
    record = json.loads((run / "run.json").read_text())
    record["chain_ok"] = True
    record["extra"]["chain_checks"] = {
        "identical": 9, "render_transient": 1, "broken": 0, "compared_frames": 2_000_000,
        "transients": [{"episode_id": "e000003", "key": "e000003.replay.x", "step": 91,
                        "camera": "wrist", "odd": "replay"}],
        "broken_by": "",
    }  # fmt: skip
    (run / "run.json").write_text(json.dumps(record))
    pages = build(
        [run], tmp_path / "out", Options(sample=0), REGISTRY, Manifest(None), progress_fixture(),
        now=NOW,
    )  # fmt: skip
    built = {p.key: p.directory for p in pages}
    page = (built["transient"] / "index.md").read_text()
    assert "> [!WARNING]" in page
    assert "episode e000003, step 91, wrist camera; the sample replay's rendering" in page
    assert "1 render transient (below)" in page
    progress = (built["progress"] / "index.md").read_text()
    assert "1 render transient in 2,000,000 frames" in progress
    assert "0.50 per million frames" in progress
    assert section(progress, "## Runs")[0][6] == "ok, 1 render transient"
